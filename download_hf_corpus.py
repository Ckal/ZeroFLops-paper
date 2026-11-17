#!/usr/bin/env python3
"""Download a subset of a Hugging Face dataset as a plain-text corpus file.

Examples:
    # web text, as before (the first 2 GB of the corpus)
    python download_hf_corpus.py --dataset HuggingFaceTB/smollm-corpus \\
        --size 2GB --output data/smollm_subset_2GB.txt

    # MORE web text that starts with exactly the same 2 GB (same order, no sampling)
    python download_hf_corpus.py --dataset HuggingFaceTB/smollm-corpus \\
        --size 8GB --output data/smollm_subset_8GB.txt

    # Python code, and English or German Wikipedia (for the "domains" stage)
    python download_hf_corpus.py --dataset codeparrot/codeparrot-clean-valid --config default \\
        --text-field content --size 20MB --output data/domain_code.txt
    python download_hf_corpus.py --dataset wikimedia/wikipedia --config 20231101.en \\
        --size 20MB --output data/domain_wiki_en.txt

The dataset is streamed, so nothing but the output file is stored. Documents
are written in the order the dataset has them; with --sample-ratio 1 (the
default) the same command always gives the same text, and a bigger --size
gives a file that begins with the smaller one.
"""

import argparse
import os
import random
import re
import sys
import time
from typing import Optional

try:
    from datasets import get_dataset_config_names, load_dataset
except Exception as exc:  # pragma: no cover
    print("ERROR: 'datasets' is not installed. Install it with: pip install datasets", file=sys.stderr)
    raise SystemExit(1) from exc


SIZE_SUFFIXES = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}

DEFAULT_TEXT_FIELDS = ("text", "content", "article", "body", "value", "sentence", "page", "document",
                       "sample", "prompt", "completion")

# configs used when none is given
DEFAULT_CONFIGS = {"HuggingFaceTB/smollm-corpus": "fineweb-edu-dedup"}


def parse_size(value: str) -> int:
    value = value.strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?", value)
    if not match:
        raise ValueError(f"invalid size string: {value!r}")
    suffix = (match.group(2) or "b").lower()
    if suffix not in SIZE_SUFFIXES:
        raise ValueError(f"unsupported size suffix: {suffix!r}")
    size = int(float(match.group(1)) * SIZE_SUFFIXES[suffix])
    if size <= 0:
        raise ValueError("size must be positive")
    return size


def _clean(text: str) -> str:
    """Exactly the cleaning of the earlier version of this script, so that new
    files begin with the same bytes as files made before."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n[ \t]+", "\n", text)


def normalize_text(text, keep_indent: bool) -> str:
    """Unify line ends and trim. Code keeps its indentation (--keep-indent)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not keep_indent:
        return _clean(_clean(text)).rstrip()           # applied twice, as before
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n").rstrip()


def extract_text(row, text_field: Optional[str]):
    if not isinstance(row, dict):
        return None
    if text_field:
        return row.get(text_field)
    for key in DEFAULT_TEXT_FIELDS:
        if row.get(key) is not None:
            return row[key]
    for value in row.values():
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            merged = "\n".join(str(part) for part in value if part is not None)
            if merged.strip():
                return merged
    return None


def utf8_prefix(data: bytes, n: int) -> bytes:
    """The first n bytes of data, shortened so no character is cut in half."""
    data = data[:n]
    while data:
        try:
            data.decode("utf-8")
            return data
        except UnicodeDecodeError:
            data = data[:-1]
    return data


def write_subset(dataset, args, limit_bytes: int):
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    part = args.output + ".part"                       # renamed when complete
    rng = random.Random(args.seed)
    sep = args.doc_sep.encode("utf-8").decode("unicode_escape").encode("utf-8")
    written, count, seen, t0, last = 0, 0, 0, time.time(), 0.0
    with open(part, "wb") as fout:
        for row in dataset:
            seen += 1
            if args.max_rows is not None and seen > args.max_rows:
                break
            if args.sample_ratio < 1.0 and rng.random() >= args.sample_ratio:
                continue
            text = normalize_text(extract_text(row, args.text_field), args.keep_indent)
            if not text or len(text) < args.min_chars:
                continue
            data = text.encode("utf-8") + sep
            if limit_bytes and written + len(data) > limit_bytes:
                chunk = utf8_prefix(data, limit_bytes - written)
                fout.write(chunk)
                written += len(chunk)
                break
            fout.write(data)
            written += len(data)
            count += 1
            if time.time() - last > 10:
                last = time.time()
                rate = written / max(1e-9, last - t0)
                eta = (limit_bytes - written) / max(rate, 1) / 60 if limit_bytes else 0
                print(f"  {written / 2 ** 20:9.1f} MB | {count:,} documents | {rate / 2 ** 20:5.1f} MB/s"
                      + (f" | {eta:5.1f} min left" if limit_bytes else ""), flush=True)
    os.replace(part, args.output)
    return written, count


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="e.g. HuggingFaceTB/smollm-corpus")
    ap.add_argument("--config", default=None,
                    help="config name (smollm-corpus: fineweb-edu-dedup if not given)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="hf_corpus.txt")
    ap.add_argument("--size", default=None, help="max output size, e.g. 10MB, 500KB, 2GB")
    ap.add_argument("--skip-rows", type=int, default=0,
                    help="skip this many documents first (to get text that does not overlap an earlier file)")
    ap.add_argument("--max-rows", type=int, default=None, help="read at most this many documents")
    ap.add_argument("--sample-ratio", type=float, default=1.0, help="share of documents to keep, (0, 1]")
    ap.add_argument("--text-field", default=None, help="the text column, if not obvious (code: content)")
    ap.add_argument("--keep-indent", action="store_true", help="keep leading spaces (use for code)")
    ap.add_argument("--doc-sep", default="\\n", help="written after each document (default: one newline)")
    ap.add_argument("--seed", type=int, default=42, help="for --sample-ratio")
    ap.add_argument("--min-chars", type=int, default=0, help="skip documents shorter than this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not (0.0 < args.sample_ratio <= 1.0):
        raise SystemExit("--sample-ratio must be in (0, 1]")
    if args.list_configs:
        try:
            print(f"Available configs for {args.dataset}: {get_dataset_config_names(args.dataset)}")
            return 0
        except Exception as exc:
            print(f"No configs found for {args.dataset}: {exc}", file=sys.stderr)
            return 2
    config = args.config or DEFAULT_CONFIGS.get(args.dataset)
    if "code" in args.dataset.lower() and not args.keep_indent:
        print("note: this looks like code; add --keep-indent to keep its indentation")
    limit_bytes = parse_size(args.size) if args.size else 0
    print(f"Dataset: {args.dataset} | config: {config or 'default'} | split: {args.split} | "
          f"size: {args.size or 'all'} | output: {args.output}")
    if args.dry_run:
        return 0
    try:
        kw = dict(name=config, split=args.split, streaming=True)
        if args.trust_remote_code:
            kw["trust_remote_code"] = True
        dataset = load_dataset(args.dataset, **kw)
    except Exception as exc:
        msg = str(exc)
        try:
            msg += f"\nAvailable configs: {get_dataset_config_names(args.dataset)}"
        except Exception:
            pass
        print(f"Failed to load dataset: {msg}", file=sys.stderr)
        return 2
    if args.skip_rows:
        dataset = dataset.skip(args.skip_rows)
    written, count = write_subset(dataset, args, limit_bytes)
    print(f"Wrote {written:,} bytes ({written / 2 ** 20:.1f} MB) in {count:,} documents to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
