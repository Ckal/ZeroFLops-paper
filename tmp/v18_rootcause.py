"""Root-cause probe for V18's column-5 collapse.

Two questions:
  1. Is the 94.7% column-5 pick a COLUMN-MAPPING BUG? (i.e. the model
     learned the right shape but emits the wrong column index?)
     -> score every permutation of the model's picks against the labels.
  2. Is the model's column output near-constant (a collapse), or does it
     carry board-dependent signal that simply gets mis-indexed?
     -> inspect the raw column logits.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.c4.fmt import COL_BYTE_OFFSET, MOVE_OFFSET, PLY_LEN

SPLIT = "runs/c4/posdata_split"
RUN = Path("runs/c4/v7mixed_v18_seed32")
CTX = 191


def load_labels():
    z = np.load(f"{SPLIT}/holdout_targets.npz")
    T = z["target"]
    best = T.max(axis=1, keepdims=True)
    optimal = np.isclose(T, best, rtol=0.0, atol=1e-6)
    return T, optimal


def main():
    T, optimal = load_labels()
    seqs = [l for l in open(f"{SPLIT}/holdout_corpus.txt", "rb").read().split(b"\n") if l]

    c = np.load("runs/_probe/v18_picks_posdata_split.npz")
    idx = c["idx"]
    pick = c["pick"]
    legal = c["legal"]

    opt = optimal[idx]
    lg = legal.astype(bool)

    base = 100.0 * (opt[np.arange(len(idx)), pick] & lg[np.arange(len(idx)), pick]).mean()
    print("model identity mapping: %.3f%%  (rows=%d)" % (base, len(idx)))

    # ---- 1. permutation search ----
    print("\n=== best column permutation of the model's picks ===")
    best = None
    for perm in itertools.permutations(range(7)):
        pp = np.array([perm[p] for p in pick])
        s = 100.0 * (opt[np.arange(len(idx)), pp] & lg[np.arange(len(idx)), pp]).mean()
        if best is None or s > best[0]:
            best = (s, perm)
    print("best perm %s -> %.3f%%   (identity was %.3f%%)"
          % (best[1], best[0], base))

    # also: score of ALWAYS-C5 under the best perm
    print("\n=== all permutations ranked (top 8) ===")
    scores = []
    for perm in itertools.permutations(range(7)):
        pp = np.array([perm[p] for p in pick])
        s = 100.0 * (opt[np.arange(len(idx)), pp] & lg[np.arange(len(idx)), pp]).mean()
        scores.append((s, perm))
    scores.sort(reverse=True)
    for s, perm in scores[:8]:
        print("   %-28s %.3f%%" % (str(perm), s))

    # ---- 2. oracle / constant baselines on the same rows ----
    print("\n=== reference policies on the SAME 4000 rows ===")
    for cc in range(7):
        s = 100.0 * (opt[:, cc] & lg[:, cc]).mean()
        print("  always-c%d : %.3f%%" % (cc, s))
    # best legal constant (pick the legal column with the highest index)
    hits = 0
    for i in range(len(idx)):
        cand = [x for x in range(7) if lg[i, x]]
        if cand and opt[i, cand[-1]]:
            hits += 1
    print("  highest-legal-index : %.3f%%" % (100.0 * hits / len(idx)))
    hits = 0
    for i in range(len(idx)):
        cand = [x for x in range(7) if lg[i, x]]
        if cand and opt[i, cand[0]]:
            hits += 1
    print("  lowest-legal-index  : %.3f%%" % (100.0 * hits / len(idx)))
    # oracle over legal+optimal
    orc = 0
    for i in range(len(idx)):
        cand = [x for x in range(7) if lg[i, x] and opt[i, x]]
        if cand:
            orc += 1
    print("  legal-and-optimal available : %.3f%% of rows" % (100.0 * orc / len(idx)))

    # ---- 3. is the model output near-constant? ----
    print("\n=== raw column logits on a sample of holdout rows ===")
    from src.gf256_markov_lm_v6 import GFMarkovLM
    raw = torch.load(RUN / "final_float.pt", map_location="cpu", weights_only=False)
    meta = raw.get("meta", {})
    model = GFMarkovLM(d=meta.get("d", 32), layers=meta.get("layers", 3),
                       fan_in=meta.get("fan_in", 3), bigram_logp=None,
                       seed=meta.get("seed", 32))
    model.load_state_dict(raw["state"])
    model.eval()
    model.set_ste(False)

    from src.c4.transcript_v2 import decode_state

    def legal_from_header(b):
        cols, _l, _s = decode_state(bytes(b))
        out = np.zeros(7, dtype=bool)
        for cc, (h, _o) in enumerate(cols):
            if h < 6:
                out[cc] = True
        return out

    sample = idx[:400]
    picks = []
    logit_rows = []
    with torch.no_grad():
        for i in sample:
            seq = seqs[i]
            ctx = min(CTX, len(seq) - 1)
            start = max(0, len(seq) - (ctx + 1))
            window = seq[start:start + ctx + 1]
            window = window + b"\n" * (ctx + 1 - len(window))
            arr = np.frombuffer(window, dtype=np.uint8)
            x = torch.tensor(arr[:-1].astype(np.int64)).unsqueeze(0)
            logits = model(x)[0]
            pos = len(seq) - 1 - start - 1
            if not (0 <= pos < ctx):
                continue
            lgv = logits[pos][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].clone()
            hdr_at = pos + 1 - MOVE_OFFSET
            legal_v = (legal_from_header(x[0, hdr_at:hdr_at + 9])
                       if 0 <= hdr_at < ctx else np.ones(7, dtype=bool))
            m = torch.tensor(legal_v)
            lgv = lgv.masked_fill(~m, float("-inf"))
            picks.append(int(lgv.argmax()))
            logit_rows.append(lgv.numpy())

    L = np.array(logit_rows)
    fin = np.where(np.isfinite(L), L, np.nan)
    print("model argmax col distribution over %d sampled rows:" % len(picks))
    bc = np.bincount(picks, minlength=7)
    for k in range(7):
        print("   col %d : %5d  (%5.1f%%)" % (k, bc[k], 100.0 * bc[k] / len(picks)))
    print("\nmean (finite) logit per column:")
    print("  ", np.round(np.nanmean(fin, axis=0), 3).tolist())
    print("std  (finite) logit per column:")
    print("  ", np.round(np.nanstd(fin, axis=0), 3).tolist())
    # spread: how much does the logit of the model's chosen column vary?
    chosen = np.array([np.nanmax(fin[r]) for r in range(len(L))])
    print("\nchosen-col logit: mean %.3f  std %.3f  min %.3f  max %.3f"
          % (np.nanmean(chosen), np.nanstd(chosen), np.nanmin(chosen), np.nanmax(chosen)))
    # margin between best and 2nd best
    srt = np.sort(np.where(np.isfinite(fin), fin, -np.inf), axis=1)
    margin = srt[:, -1] - srt[:, -2]
    print("argmax margin (best - 2nd): mean %.4f  median %.4f  frac<0.01: %.3f"
          % (np.nanmean(margin), np.nanmedian(margin), float((margin < 0.01).mean())))


if __name__ == "__main__":
    main()
