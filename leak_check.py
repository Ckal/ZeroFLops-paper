"""
leak_check.py - how much of the validation text appears word for word in the
text the hard tables are counted from?

Samples random 64-byte and 256-byte snippets from the validation text and
checks, with a rolling hash over every position, how many of them occur
verbatim in each table text. A 256-byte match almost always means duplicated
content (the same article or paragraph twice), not natural language reuse.

  python leak_check.py --val_data data/smollm_subset_200MB.txt \
      --tables data/smollm_subset_200MB.txt data/smollm_subset_1024MB.txt
Each --tables file is used the way the trainer uses it with --val_data:
the whole file minus the validation text.
"""
import argparse, importlib.util, os, time

import numpy as np

P = 0x100000001B3
MOD = 1 << 64


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("trainer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def window_hashes(c, m):
    """64-bit polynomial hash of every window of length m in c (position independent)."""
    n = len(c)
    if n < m:
        return np.zeros(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        pw = np.cumprod(np.full(n, P, dtype=np.uint64))                         # P^(t+1)
        qw = np.cumprod(np.full(n, pow(P, -1, MOD), dtype=np.uint64))           # P^-(t+1)
        H = np.cumsum((c.astype(np.uint64) + np.uint64(1)) * qw) * pw          # sum b_i P^(t-i)
        prev = np.concatenate([np.zeros(1, dtype=np.uint64), H[:-m]])
        return H[m - 1:] - prev * np.uint64(pow(P, m, MOD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--tables", nargs="+", required=True, help="text files the tables are built from")
    ap.add_argument("--n", type=int, default=5000, help="snippets per length")
    ap.add_argument("--trainer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "hard_residual_lm.py"))
    ap.add_argument("--chunk", type=int, default=1 << 24)
    args = ap.parse_args()
    tr = load_trainer(args.trainer)
    _, val = tr.load_bytes(args.val_data, 0)
    rng = np.random.default_rng(0)
    snips = {}
    for m in (64, 256):
        pos = rng.integers(0, len(val) - m, args.n)
        snips[m] = np.unique(np.concatenate([window_hashes(val[p:p + m], m) for p in pos]))
    print(f"validation: {len(val):,} bytes, {args.n} snippets each of 64 and 256 bytes")
    for path in args.tables:
        t0 = time.time()
        data, _ = tr.load_bytes(path, 0, block=0)
        data = tr.remove_text(data, val)
        found = {m: set() for m in snips}
        for a in range(0, len(data), args.chunk):
            for m in snips:
                c = data[max(0, a - m + 1):a + args.chunk]
                hit = snips[m][np.isin(snips[m], window_hashes(c, m))]
                found[m].update(hit.tolist())
        rates = "  ".join(f"{m}-byte: {len(found[m]) / len(snips[m]):6.2%}" for m in snips)
        print(f"  {os.path.basename(path):32s} {len(data) / 1e6:8.1f} MB   found verbatim  {rates}"
              f"   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
