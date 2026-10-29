#!/usr/bin/env python
"""Is the audit's 32.5% 'label vs best_move' rate explained by chance?

best_move() returns ONE arbitrary member of the optimal set on LOSS/DRAW
positions (src/c4/exact_solver.py:233). If the label is also an arbitrary
member of that set (it is: targets are a UNIFORM distribution over the
canonical best set, see tools/build_disjoint_holdout.py:191-195, so argmax
picks one member uniformly), then agreement should be ~ sum(1/|opt_i|)/n.

If observed 32.5% is in line with that chance expectation, the 'gap' is a
tie-break artefact, not a decode error.

Also computes a permutation null: shuffle the label assignment across rows
and re-measure agreement with best_move -- the null distribution the
observed rate must beat to mean anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.board import Board, Outcome          # noqa: E402
from src.c4 import transcript_v2 as TV2           # noqa: E402
from src.c4.exact_solver import ExactSolver, move_order, has_four, WIN, DRAW  # noqa: E402

SPLIT = ROOT / "runs/c4/_pilot_v23"
N_PROVED = 200


def decode_header_to_board(hdr):
    columns, _last, turn = TV2.decode_state(hdr)
    b = Board()
    b.p1 = b.p2 = 0
    b.mask = 0
    for c, (h, owners) in enumerate(columns):
        b.heights[c] = h
        for r, owner in enumerate(owners):
            if owner == 1:
                b.p1 |= 1 << (r * 7 + c)
            else:
                b.p2 |= 1 << (r * 7 + c)
    b.mask = b.p1 | b.p2
    b.turn = turn
    return b


def load():
    z = np.load(SPLIT / "holdout_targets.npz")
    Th = z["target"]
    lines = [l for l in (SPLIT / "holdout_corpus.txt").read_bytes().split(b"\n") if l]
    items = []
    for i in range(min(len(lines), len(Th))):
        raw = lines[i]
        if len(raw) % TV2.PLY_LEN:
            continue
        nrec = len(raw) // TV2.PLY_LEN
        hdr = raw[(nrec - 1) * TV2.PLY_LEN:(nrec - 1) * TV2.PLY_LEN + TV2.PLY_LEN - 1]
        try:
            b = decode_header_to_board(hdr)
        except Exception:
            continue
        if b.outcome() != Outcome.ONGOING:
            continue
        items.append((b, Th[i].copy(), raw[nrec * TV2.PLY_LEN - 1] - TV2.COL_BYTE_OFFSET))
    return items


def root_values(solver, b, side):
    mine, theirs = (b.p1, b.p2) if side == 1 else (b.p2, b.p1)
    heights = list(b.heights)
    empties = 42 - (b.p1.bit_count() + b.p2.bit_count())
    out = {}
    for col in move_order([c for c in range(7) if heights[c] < 6]):
        h = heights[col]
        bit = 1 << (h * 7 + col)
        heights[col] = h + 1
        try:
            if has_four(mine | bit):
                out[col] = 1000
            else:
                out[col] = -solver._negamax(theirs, mine | bit, heights, empties - 1)
        except TimeoutError:
            pass
        finally:
            heights[col] = h
    return out


def normalise(v):
    if v >= 1000:
        return WIN
    if v >= WIN:
        return WIN
    if v == DRAW:
        return DRAW
    return -1


def main() -> int:
    items = load()
    solver = ExactSolver(max_empty=16, budget=20_000_000)
    rows = []
    for b, t, played in items:
        e = 42 - int(sum(b.heights))
        if e > 16 or e < 8:
            continue
        try:
            status = solver.solve(b, side=b.turn)
            if status is None:
                continue
            bm = solver.best_move(b, side=b.turn)
            rv = root_values(solver, b, b.turn)
        except TimeoutError:
            continue
        if not rv:
            continue
        nv = {c: normalise(v) for c, v in rv.items()}
        mx = max(nv.values())
        opt = sorted(c for c, v in nv.items() if v == mx)
        rows.append({"label": int(np.argmax(t)), "bm": bm, "opt": opt,
                     "status": status, "target": t})
        if len(rows) >= N_PROVED:
            break

    n = len(rows)
    obs = 100.0 * sum(1 for r in rows if r["label"] == r["bm"]) / n
    chance = 100.0 * sum(1.0 / len(r["opt"]) for r in rows) / n

    print(f"n = {n}")
    print(f"observed agreement (label == best_move): {obs:.1f}%")
    print(f"chance expectation  E[1/|opt|]         : {chance:.1f}%")

    # permutation null: relabel rows at random, re-measure
    rng = np.random.default_rng(0)
    labels = np.array([r["label"] for r in rows])
    bms = np.array([r["bm"] for r in rows])
    null = []
    for _ in range(2000):
        perm = rng.permutation(labels)
        null.append(100.0 * np.mean(perm == bms))
    null = np.array(null)
    p = 100.0 * np.mean(null >= obs)
    print(f"permutation null: mean {null.mean():.1f}%  sd {null.std():.1f}%  "
          f"p95 {np.percentile(null, 95):.1f}%")
    print(f"observed is at the {100.0*np.mean(null <= obs):.1f}th percentile of the null "
          f"(p = {p:.3f})")

    # how many rows have a WIN-only optimum (where best_move is canonical)
    win_uniq = [r for r in rows if len(r["opt"]) == 1]
    win_rate = (100.0 * sum(1 for r in win_uniq if r["label"] == r["bm"]) / len(win_uniq)
                if win_uniq else float("nan"))
    print(f"\nunique-optimum rows: {len(win_uniq)}/{n} -> agreement {win_rate:.1f}%")
    print("  (here best_move is the ONLY correct answer, so agreement should be ~100%)")

    # status breakdown of unique-optimum rows
    from collections import Counter
    st = Counter(r["status"] for r in win_uniq)
    print(f"  unique-optimum rows by status: {dict(st)}")

    # per-status agreement over all rows
    print("\nagreement by position status (all rows):")
    allst = Counter(r["status"] for r in rows)
    for s in sorted(allst):
        sub = [r for r in rows if r["status"] == s]
        a = 100.0 * sum(1 for r in sub if r["label"] == r["bm"]) / len(sub)
        ch = 100.0 * sum(1.0 / len(r["opt"]) for r in sub) / len(sub)
        print(f"  status {s:+d}: n={len(sub):3d}  observed {a:5.1f}%  chance {ch:5.1f}%")

    out = {
        "n": n,
        "observed_agreement_pct": round(obs, 2),
        "chance_expectation_pct": round(chance, 2),
        "null_mean_pct": round(float(null.mean()), 2),
        "null_sd_pct": round(float(null.std()), 2),
        "null_p95_pct": round(float(np.percentile(null, 95)), 2),
        "permutation_p_value": round(float(p), 4),
        "unique_optimum_rows": len(win_uniq),
        "unique_optimum_agreement_pct": round(win_rate, 2),
        "unique_optimum_rows_by_status": {str(k): v for k, v in st.items()},
    }
    dst = ROOT / "runs/_probe/provenance_chance_baseline.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
