"""Compute constant-column baselines on the v7mixed_v18_seed32 holdout.

Answers: is holdout_optimal_pct=30.73% distinguishable from a fixed-column
policy that ignores the board entirely?
"""
import json
from pathlib import Path

import numpy as np

RUN = Path("runs/c4/v7mixed_v18_seed32")
SPLIT = Path("runs/c4/posdata_split")
PLY_LEN = 9  # bytes per ply (header 9 + move bytes)

z = np.load(SPLIT / "holdout_targets.npz")
T = z["target"]
print("sidecar shape:", T.shape, "ndim:", T.ndim)

seqs = [l for l in (SPLIT / "holdout_corpus.txt").read_bytes().split(b"\n") if l]
print("holdout rows:", len(seqs), "sidecar rows:", T.shape[0])

per_ply = T.ndim == 3
if per_ply:
    ply_of = z["ply_of"] if "ply_of" in z else None
    n_plies = z["n_plies"] if "n_plies" in z else None
    assert ply_of is not None, "per-ply sidecar without ply_of"
    T_final = np.zeros((len(seqs), 7), dtype=T.dtype)
    for i in range(len(seqs)):
        nply = int(n_plies[i]) if n_plies is not None else len(seqs[i]) // PLY_LEN
        target_ply = max(0, nply - 1)
        slots = np.nonzero(ply_of[i] == target_ply)[0]
        if len(slots) == 0:
            lab = np.nonzero(ply_of[i] >= 0)[0]
            if len(lab) == 0:
                continue
            slots = lab[-1:]
        T_final[i] = T[i, slots[0]]
    T = T_final

best = T.max(axis=1, keepdims=True)
optimal = np.isclose(T, best, rtol=0.0, atol=1e-6)
n = optimal.shape[0]
print("scored positions:", n)
print("mean tie-set size: %.4f" % optimal.sum(axis=1).mean())

# Constant-column policies: always play column c (when legal).
print("\n=== constant-column policies (play column c whenever legal) ===")
rows = []
for c in range(7):
    # a constant policy only scores when c is legal; approximate legal with
    # the target being non-zero OR treat c as always-playable. Report the
    # unconstrained number too (c optimal) and a legality-aware number.
    hit = optimal[:, c].sum()
    rows.append((c, hit, 100.0 * hit / n))
for c, hit, pct in rows:
    print("  always-c%d : optimal %.3f%%  (%d/%d)" % (c, pct, hit, n))

# What the model actually did (from metrics.jsonl, last eval).
last = None
for line in (RUN / "metrics.jsonl").read_text().splitlines():
    line = line.strip()
    if line:
        last = json.loads(line)
print("\n=== model (last eval) ===")
print("  holdout_optimal_pct: %.3f" % last["holdout_optimal_pct"])
print("  holdout_n:", last.get("holdout_n"))
print("  pick_tot:", last.get("holdout_pick_tot"))
print("  pick_dist(hits):", last.get("holdout_pick_dist"))
tot = last.get("holdout_pick_tot") or []
if tot and sum(tot):
    p5 = tot[5] / sum(tot) * 100
    print("  fraction of picks on column 5: %.1f%%" % p5)

best_const = max(rows, key=lambda r: r[2])
print("\nBEST CONSTANT COLUMN: c%d at %.3f%%" % (best_const[0], best_const[2]))
print("MODEL: %.3f%%  -> delta vs best constant: %+.3f pp"
      % (last["holdout_optimal_pct"], last["holdout_optimal_pct"] - best_const[2]))

# Random-among-optimal oracle (upper bound of tie-breaking luck)
rand = optimal.sum(axis=1).mean() / 7 * 100
print("random-among-optimal oracle: %.3f%%" % rand)

# Legal-aware constant: only positions where column c has a non-zero target prob
print("\n=== legal-aware: score only where column c is a legal (non-zero prob) move ===")
for c in range(7):
    legal = T[:, c] > 0
    if legal.sum() == 0:
        print("  c%d: never legal" % c)
        continue
    hit = (optimal[legal, c]).sum()
    print("  always-c%d : %.3f%% over %d positions where legal" % (c, 100.0 * hit / legal.sum(), legal.sum()))
