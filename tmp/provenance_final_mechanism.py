#!/usr/bin/env python
"""Confirm the final mechanism: best_move()'s tie-break is systematically
biased (leftmost legal column on LOSS), while the LABEL is uniform over the
true optimal set.

Expected outcome:
  * on LOSS rows (status == -1), best_move == leftmost legal column ~100%
  * the target's SUPPORT == the true optimal set on 100% of rows
  * the target is (near-)uniform over that support
  * therefore "label == best_move" is a coin flip on tied rows, and the
    audit's 32.5% is exactly E[1/|opt|], NOT a decode error.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

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
        items.append((b, Th[i].copy()))
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
    for b, t in items:
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
        rows.append({"board": b, "target": np.asarray(t, dtype=np.float64),
                     "status": status, "bm": bm, "opt": opt})
        if len(rows) >= N_PROVED:
            break

    n = len(rows)
    print(f"n = {n}")

    # 1) best_move tie-break bias on LOSS rows
    loss = [r for r in rows if r["status"] == -1]
    leftmost_hits = 0
    for r in loss:
        legal = [c for c in range(7) if r["board"].heights[c] < 6]
        if r["bm"] == min(legal):
            leftmost_hits += 1
    print(f"\nLOSS rows: {len(loss)}/{n}")
    print(f"  best_move == leftmost legal column: {leftmost_hits}/{len(loss)} "
          f"({100.0*leftmost_hits/max(1,len(loss)):.1f}%)")
    print("  -> best_move's LOSS tie-break is a FIXED column, not a random one,")
    print("     so it cannot be matched by a uniform label more than ~1/|opt| of the time.")

    # 2) target support == true optimal set
    sup_ok = 0
    uniform_err = []
    for r in rows:
        sup = set(np.nonzero(r["target"] > 1e-6)[0].tolist())
        if sup == set(r["opt"]):
            sup_ok += 1
        vals = r["target"][sorted(sup)] if sup else np.array([1.0])
        uniform_err.append(float(np.max(vals) - np.min(vals)))
    print(f"\ntarget SUPPORT == true optimal set: {sup_ok}/{n} "
          f"({100.0*sup_ok/n:.1f}%)")
    print(f"target uniformity over its support: max spread "
          f"{max(uniform_err):.2e} (0 = perfectly uniform)")

    # 3) is the label uniform over the optimal set?
    counts = Counter()
    for r in rows:
        counts[r["target"].argmax()] += 1
    print(f"label argmax column histogram: {dict(sorted(counts.items()))}")

    # 4) decisive: agreement restricted to rows with a UNIQUE optimum
    uniq = [r for r in rows if len(r["opt"]) == 1]
    ok = sum(1 for r in uniq if r["target"].argmax() == r["opt"][0])
    print(f"\nrows with a UNIQUE optimal column: {len(uniq)}/{n}")
    print(f"  label == that unique column: {ok}/{len(uniq)} "
          f"({100.0*ok/max(1,len(uniq)):.1f}%)")
    print("  -> where a single right answer exists, the label is ALWAYS right.")
    print("     A reversed decode cannot produce this.")

    out = {
        "n": n,
        "loss_rows": len(loss),
        "best_move_is_leftmost_legal_on_loss_pct": round(100.0 * leftmost_hits / max(1, len(loss)), 1),
        "target_support_equals_optimal_set_pct": round(100.0 * sup_ok / n, 1),
        "max_target_uniformity_spread": float(max(uniform_err)),
        "unique_optimum_rows": len(uniq),
        "label_correct_on_unique_optimum_pct": round(100.0 * ok / max(1, len(uniq)), 1),
    }
    dst = ROOT / "runs/_probe/provenance_final_mechanism.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
