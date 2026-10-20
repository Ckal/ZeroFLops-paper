"""Independent re-derivation of the n-gram baselines used in the paper.

The paper's headline claim is: "Hard BPC 3.071 beats the interpolated trigram
baseline 3.087" -- a margin of 0.016 BPC. This script recomputes bigram and
trigram BPC on exactly the split gf256_markov_lm.load_data() uses (90/10,
byte level) so that claim can be checked without trusting the paper's table.

Nothing here imports torch, so it runs anywhere numpy is available.

Usage:
    python src/baseline_check.py --data data/tinyshakespeare.txt
"""
import argparse
import math

import numpy as np

LN2 = math.log(2.0)


def load_data(path):
    """Identical split to gf256_markov_lm.load_data()."""
    raw = open(path, "rb").read()
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def _bpc(total_nats, n_chars):
    return total_nats / (n_chars * LN2)


def _pair_table(data, order):
    """Sparse (context, next) counts for the given n-gram order.

    A pair code is ctx * 256 + next, so a miss is a binary search. Contexts
    are the `order` bytes immediately preceding the target.
    """
    if order == 1:
        ctx = data[:-1].astype(np.int64)
    elif order == 2:
        ctx = data[:-2].astype(np.int64) * 256 + data[1:-1]
    elif order == 3:
        ctx = (data[:-3].astype(np.int64) * 65536
               + data[1:-2].astype(np.int64) * 256 + data[2:-1])
    else:
        raise ValueError("order must be 1..3")
    nxt = data[order:]
    uniq, counts = np.unique(ctx * 256 + nxt, return_counts=True)
    return uniq, counts.astype(np.float64)


def _ctx_index(table):
    """Sorted unique contexts of a pair table and their TOTAL occurrence counts.

    Must sum the counts, not count the rows -- a context with 30 distinct
    next-bytes still needs the sum of all 30 counts as its denominator.
    """
    uniq, counts = table
    ctx = uniq // 256
    u, inverse = np.unique(ctx, return_inverse=True)
    totals = np.zeros(len(u), dtype=np.float64)
    np.add.at(totals, inverse, counts)
    return u, totals


def _lookup(table, ctx, nxt):
    """(count, seen, P(next|ctx)) for each aligned (ctx, next) element."""
    uniq, counts = table
    codes = ctx * 256 + nxt
    idx = np.clip(np.searchsorted(uniq, codes), 0, len(uniq) - 1)
    seen = uniq[idx] == codes
    c = np.where(seen, counts[idx], 0.0)

    ctx_arr, tot_arr = _ctx_index(table)
    ci = np.clip(np.searchsorted(ctx_arr, ctx), 0, len(ctx_arr) - 1)
    ok = seen & (ctx_arr[ci] == ctx)
    return c, seen, np.where(ok, c / np.maximum(tot_arr[ci], 1.0), 0.0)


def bigram_bpc(train, val, smooth=1.0):
    """Dense add-k smoothed bigram.

    With smooth=0 any context byte unseen in training gives log(0) = inf, so
    the paper's "bigram 3.602" must come from a smoothed variant.
    """
    counts = np.full((256, 256), smooth, dtype=np.float64)
    np.add.at(counts, (train[:-1], train[1:]), 1.0)
    probs = counts / counts.sum(1, keepdims=True)
    nll = -np.log(probs[val[:-1], val[1:]]).sum()
    return _bpc(nll, len(val) - 1)


def stupid_backoff_bpc(train, val, alpha=0.4):
    """Brantsen et al. stupid backoff, add-one smoothed bigram underneath.

    S(x y | z) = S(x|y)   if c(x y z) > 0
               = a S(x|y) if c(y z) > 0
               = a^2 S(x|_)
    """
    uni = np.full(256, 1.0)
    np.add.at(uni, train, 1.0)
    up = uni / uni.sum()

    bg = np.full((256, 256), 1.0, dtype=np.float64)
    np.add.at(bg, (train[:-1], train[1:]), 1.0)
    bprobs = bg / bg.sum(1, keepdims=True)

    tri = _pair_table(train, 3)

    # targets at positions 3..len(val)-1
    nxt = val[3:].astype(np.int64)
    prev = val[2:-1].astype(np.int64)
    prev2 = val[1:-2].astype(np.int64)

    ctx3 = val[:-3].astype(np.int64) * 65536 + prev2 * 256 + prev
    _, _, p3 = _lookup(tri, ctx3, nxt)

    p2 = bprobs[prev, nxt]

    # stupid backoff: use the trigram conditional when the context was seen,
    # otherwise discount to the bigram, then to the unigram. A seen trigram
    # gives P(x|y,z) -- certainty (1.0) only for a context seen exactly once.
    prob = np.where(p3 > 0, p3,
                    np.where(p2 > 0, alpha * p2, (alpha * alpha) * up[nxt]))
    return _bpc(-np.log(np.maximum(prob, 1e-12)).sum(), len(nxt))


def interp_abs_discount_bpc(train, val, discount=0.75):
    """Interpolated absolute-discount n-gram -- the family the paper's phrase
    "interpolated trigram baseline" most plausibly refers to.

    Reported at several discounts because the paper never states which one it
    used, and the answer moves with it.
    """
    uni = np.full(256, 1.0)
    np.add.at(uni, train, 1.0)
    up = uni / uni.sum()

    bg = np.full((256, 256), 1.0, dtype=np.float64)
    np.add.at(bg, (train[:-1], train[1:]), 1.0)
    bprobs = bg / bg.sum(1, keepdims=True)

    bi = _pair_table(train, 2)
    tri = _pair_table(train, 3)

    nxt = val[3:].astype(np.int64)
    prev = val[2:-1].astype(np.int64)
    prev2 = val[1:-2].astype(np.int64)

    ctx3 = val[:-3].astype(np.int64) * 65536 + prev2 * 256 + prev
    _, _, p3 = _lookup(tri, ctx3, nxt)

    ctx2 = prev2 * 256 + prev
    _, _, p2 = _lookup(bi, ctx2, nxt)

    p1 = bprobs[prev, nxt]

    lam = discount
    p_hi = np.where(p3 > 0, p3, p2)
    w_hi = np.where(p3 > 0, lam, 0.0)
    p_lo = np.where(p2 > 0, p1, up[nxt])
    w_lo = np.where(p2 > 0, lam, 0.0)

    prob = w_hi * p_hi + (1 - w_hi) * (w_lo * p_lo + (1 - w_lo) * up[nxt])
    return _bpc(-np.log(np.maximum(prob, 1e-12)).sum(), len(nxt))


def coverage(train, val, order):
    """Fraction of validation positions whose full context was seen in train.

    Vectorised via a packed int64 code + sorted-array membership test, which
    matters: the naive Python loop over 100k positions dominated runtime.
    """
    n = len(val)
    if order == 1:
        seen = set(train.tolist())
        hit = np.fromiter((v in seen for v in val), dtype=bool, count=n)
    else:
        if order == 2:
            tcode = train[:-1].astype(np.int64) * 256 + train[1:]
            vcode = val[:-1].astype(np.int64) * 256 + val[1:]
        else:
            tcode = (train[:-2].astype(np.int64) * 65536
                     + train[1:-1].astype(np.int64) * 256 + train[2:])
            vcode = (val[:-2].astype(np.int64) * 65536
                     + val[1:-1].astype(np.int64) * 256 + val[2:])
        uniq = np.unique(tcode)
        pos = np.searchsorted(uniq, vcode)
        pos_c = np.clip(pos, 0, len(uniq) - 1)
        hit = uniq[pos_c] == vcode
    return float(hit.mean()), int((~hit).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    args = ap.parse_args()

    train, val = load_data(args.data)
    print(f"corpus: {len(train) + len(val)} bytes  train={len(train)}  val={len(val)}")
    print("split matches gf256_markov_lm.load_data (90/10, byte level)\n")

    rows = [("uniform (8 bits)", 8.0),
            ("bigram, add-one smoothing", bigram_bpc(train, val, 1.0))]
    for a in (0.2, 0.4, 0.6):
        rows.append((f"trigram stupid-backoff (alpha={a})", stupid_backoff_bpc(train, val, a)))
    for d in (0.5, 0.75, 0.9):
        rows.append((f"trigram interpolated abs-discount (d={d})",
                     interp_abs_discount_bpc(train, val, d)))

    print(f"{'baseline':<44}{'BPC':>8}")
    print("-" * 52)
    for name, v in rows:
        print(f"{name:<44}{v:>8.3f}")

    # Coverage: how many validation positions had their full context observed
    # in training. This is what decides whether a trigram is a fair bar -- if
    # most contexts are unseen, any backoff model is mostly guessing.
    print()
    print("validation context coverage (fraction of positions whose full")
    print("context appeared in train, and the count that did not):")
    for order in (1, 2, 3):
        frac, miss = coverage(train, val, order)
        name = {1: "unigram", 2: "bigram ", 3: "trigram"}[order]
        print(f"  {name}  {frac:.4f}   ({miss:,} unseen)")
    print()
    print("paper claims:  bigram 3.602 | trigram 3.087 | GF256-hard 3.071")


if __name__ == "__main__":
    main()