"""
trace.py - watch the model read one short text, byte by byte.

  python trace.py --feat runs\\p_e8l_cache_best.pt --alone runs\\p_e8f_alone_best.pt ^
      --text "The capital of France is Paris. The capital of France is Paris."

For every byte it prints how many bits each part needed for the TRUE next byte
(lower = more sure; 0.1 = nearly certain, 8 = no idea), and its top guess:

  tables    each counted expert on its own (short contexts, long contexts,
            word expert, document cache)
  layers    the transformer's answer if it had to stop after layer 0 (= only
            its inputs, memory included), after a few middle layers, and at the end
  no cache  the same transformer with its document cache blanked
  no long   ... with the long contexts (8+ bytes) blanked
  no mem    ... with all memory inputs blanked. It never trained that way, so
            this shows how much it relies on the memory, not how good a plain
            model would be; for that, give --alone
  plain     a transformer trained without memory (--alone), if given

Reading it: where "tables" are already low, the memory knew the answer and the
transformer only has to copy it. Where tables are high and the last layer is
low, the transformer worked it out itself. "layer 0" shows how much is known
the moment the memory is injected, before any attention has happened.

A short --text has no document before it, which the model rarely saw in
training. For a realistic picture read a piece of real text with its history:

  python trace.py --feat ... --alone ... --text_file data\\enwik8.valid ^
      --offset 1000000 --length 9000 --show 300

The last 513 bytes (the model's window) are traced, the 8,000+ bytes before
them are the document the cache has seen; --show 300 prints the last 300 rows.

A csv with all numbers is written next to the model (--out).
"""
import argparse, csv, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

import analyze as A

LN2 = math.log(2)


def show(b):
    c = chr(b)
    return {"\n": "\\n", " ": "_", "\t": "\\t"}.get(c, c if 32 < b < 127 else f"x{b:02x}")


def prep(tr, hm, text, T, dev):
    """The inputs for one text, as the trainer's score_text builds them."""
    H = hm.hist if hm is not None else 1
    b = list(text)[-(T + 1):]
    before = list(text)[:-len(b)][-H:]                  # real text before the window: the document so far
    npad = H - len(before)
    lead = [32] * npad + before
    fin = torch.tensor(lead + b[:-1], device=dev)[None]
    x, y = fin[:, H:], torch.tensor(b[1:], device=dev)[None]
    hard = None
    if hm is not None:
        wf = None
        if hm.word_tabs:
            full, _ = tr.word_features(np.array(lead + b[:-1], dtype=np.uint8), hm.cls_of)
            wf = torch.from_numpy(full[-(fin.shape[1] if hm.wf_full else x.shape[1]):]).to(dev)[None]
        hard = hm.query(fin, wf, pad=torch.tensor([npad], device=dev))
    return b, x, y, hard


def bits_top(logits, y):
    lp = F.log_softmax(logits.float(), -1)[0]
    return (-lp.gather(1, y[0][:, None])[:, 0] / LN2).tolist(), lp.argmax(-1).tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat", required=True, help="*_best.pt of a model trained with the memory")
    ap.add_argument("--alone", default="", help="*_best.pt of a plain model, for comparison")
    ap.add_argument("--load_hard", default="", help="other tables than the ones the model was trained with")
    ap.add_argument("--text", default="", help="the text to read (\\n for a line break)")
    ap.add_argument("--text_file", default="", help="or: read the text from this file")
    ap.add_argument("--offset", type=int, default=0, help="with --text_file: start at this byte")
    ap.add_argument("--length", type=int, default=0, help="with --text_file: read this many bytes (0 = all)")
    ap.add_argument("--show", type=int, default=0, help="print only the last N bytes (0 = all)")
    ap.add_argument("--out", default="", help="csv file (default: trace.csv next to the model)")
    ap.add_argument("--trainer", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    if not args.trainer:
        c = [os.path.join(here, n) for n in ("hard_residual_lm.py", "tg_train_hard.py")]
        args.trainer = max([p for p in c if os.path.exists(p)], key=os.path.getmtime)
    tr = A.TR = A.load_trainer(args.trainer)
    A.LN2 = LN2
    dev = args.device
    if args.text_file:
        with open(args.text_file, "rb") as f:
            f.seek(args.offset)
            text = f.read(args.length or -1)
    else:
        text = args.text.replace("\\n", "\n").encode("utf-8")
    if len(text) < 2:
        raise SystemExit("give --text or --text_file")

    model, hm, info = A.load_model(tr, args.feat, args.load_hard, dev)
    if hm is None:
        raise SystemExit("--feat must be a model trained with the memory")
    T = info["ctx"]
    b, x, y, hard = prep(tr, hm, text, T, dev)
    logs, stats = hard
    n = x.shape[1]
    cols, tops = {}, {}

    for name in model.names:                               # each counted expert on its own
        cols["tab:" + name], tops["tab:" + name] = bits_top(logs[name], y)

    hs = A.internals(model, x, hard)                       # hidden state after every layer
    L = len(hs) - 1
    for li in sorted({0, L // 4, L // 2, 3 * L // 4, L}):
        key = "layer %d" % li
        cols[key], tops[key] = bits_top(A.final_logits(model, hs[li], logs), y)
    last = "layer %d" % L

    abl = A.ablations(hm, model, logs, stats)
    for label, key in (("no document cache", "no cache"), ("no long contexts (8+ bytes)", "no long"),
                       ("no memory at all (uniform, no counts)", "no mem")):
        if label in abl:
            cols[key], tops[key] = bits_top(model(x, abl[label]), y)

    if args.alone:
        m2, h2, i2 = A.load_model(tr, args.alone, "", dev)
        if h2 is not None:
            raise SystemExit("--alone must be a plain model")
        _, x2, y2, _ = prep(tr, None, text, i2["ctx"], dev)
        c, t = bits_top(m2(x2, None), y2)
        pad = n - len(c)
        cols["plain"], tops["plain"] = [float("nan")] * pad + c[-n:], [32] * pad + t[-n:]

    names = list(cols)
    short = {k: k.replace("tab:ngram_long", "long").replace("tab:ngram_o2", "2-byte").replace("tab:ngram", "short")
             .replace("tab:word", "word").replace("tab:ccache", "ccache").replace("tab:cache", "cache")
             for k in names}
    w = 9
    print(f"\nmodel {args.feat} ({info['params'] / 1e6:.1f}M, {L} layers) reading {n + 1} bytes")
    print("bits for the true next byte (lower = more sure), then the top guess\n")
    print("seen".rjust(14) + " next | " + "".join(short[k].rjust(w) for k in names))
    print("-" * (22 + w * len(names)))
    first = max(0, n - args.show) if args.show else 0
    for i in range(first, n):
        seen = "".join(show(c) for c in b[max(0, i - 11):i + 1])[-13:]
        line = seen.rjust(14) + f"  {show(b[i + 1]):>3} | "
        for k in names:
            v = cols[k][i]
            line += ("  -".rjust(w) if v != v else f"{v:5.2f} {show(tops[k][i])[:2]:>2}".rjust(w))
        print(line)
    print("-" * (22 + w * len(names)))
    wm = tr.WORD_MASK
    ws = [i for i in range(n) if wm[b[i + 1]] and not wm[b[i]]]
    rest = [i for i in range(n) if i not in set(ws)]

    def mean(k, idx):
        v = [cols[k][i] for i in idx if cols[k][i] == cols[k][i]]
        return sum(v) / len(v) if v else float("nan")

    for title, idx in (("all bytes", range(n)), ("first letters", ws), ("other bytes", rest)):
        print(f"{'mean, ' + title:>20} | " + "".join(f"{mean(k, idx):5.2f}   ".rjust(w) for k in names))

    solved = [i for i in range(n) if cols[last][i] < 1.0]
    by_mem = [i for i in solved if min(cols[k][i] for k in names if k.startswith("tab:")) < 1.0]
    by_in = [i for i in solved if cols["layer 0"][i] < 1.0]
    print(f"\nof the {len(solved)} bytes the model is sure about at the end (under 1 bit):")
    print(f"  {len(by_mem):4d} a table was already sure about (the memory knew it)")
    print(f"  {len(by_in):4d} were already sure right after the inputs, before any attention (layer 0)")
    print(f"  {len(solved) - len(by_mem):4d} no table was sure about (the transformer worked them out)")
    ref = "plain" if "plain" in cols else "no cache" if "no cache" in cols else None
    if ref:
        ok = [i for i in range(n) if cols[ref][i] == cols[ref][i]]
        hurt = sorted(ok, key=lambda i: cols[ref][i] - cols[last][i], reverse=True)[:5]
        what = "the plain model" if ref == "plain" else "the same model without its document cache"
        print(f"\nwhere the memory helped most (bits: this model -> {what}):")
        for i in hurt:
            seen = "".join(show(c) for c in b[max(0, i - 15):i + 1])
            print(f"  ...{seen} -> '{show(b[i + 1])}'   {cols[last][i]:.2f} -> {cols[ref][i]:.2f}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.feat)), "trace.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["pos", "seen", "next"] + [short[k] for k in names] + [short[k] + " guess" for k in names])
        for i in range(n):
            wr.writerow([i, show(b[i]), show(b[i + 1])] + [f"{cols[k][i]:.4f}" for k in names] +
                        [show(tops[k][i]) for k in names])
    print(f"\nall numbers -> {out}")


if __name__ == "__main__":
    main()