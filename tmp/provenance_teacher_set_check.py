#!/usr/bin/env python
"""Two loose ends from provenance_final_mechanism.json:

1. target SUPPORT == exact optimal set only on 42% of rows, yet the LABEL
   is inside the exact optimal set on 100%.  That means the teacher's best
   set is a SUBSET of the exact optimal set -- i.e. the teacher was built
   with a DEPTH-LIMITED solver, not the exact one.  Verify:
      * support subset of exact opt set?  (expect yes)
      * support superset?                 (expect no)
      * does the support ever match the REVERSED opt set? (H1's real claim)
        -> if reversed ever matches while forward never does, H1 lives.

2. best_move is only leftmost-legal on 2.4% of LOSS rows, so move_order is
   not left-to-right.  Identify the actual tie-break column and whether
   the label's tie-break matches it.
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
from src.c4.mirror import canonical_board          # noqa: E402

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
    print("move_order sample:", move_order([0, 1, 2, 3, 4, 5, 6]))

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
        cb, flipped = canonical_board(b)
        rows.append({"board": b, "target": np.asarray(t, dtype=np.float64),
                     "status": status, "bm": bm, "opt": opt, "flipped": flipped})
        if len(rows) >= N_PROVED:
            break

    n = len(rows)
    eq = sub = sup = rev_match = rev_sub = 0
    sub_sizes, opt_sizes = [], []
    for r in rows:
        sup_set = set(np.nonzero(r["target"] > 1e-6)[0].tolist())
        opt_set = set(r["opt"])
        rev = {6 - c for c in opt_set}
        if sup_set == opt_set:
            eq += 1
        if sup_set <= opt_set:
            sub += 1
        if sup_set >= opt_set:
            sup += 1
        if sup_set == rev:
            rev_match += 1
        if sup_set <= rev:
            rev_sub += 1
        sub_sizes.append(len(sup_set))
        opt_sizes.append(len(opt_set))

    print(f"\nn = {n}")
    print("=== TEACHER SET vs EXACT OPTIMAL SET ===")
    print(f"  support == exact opt set      : {eq}/{n} ({100.0*eq/n:.1f}%)")
    print(f"  support SUBSET of exact opt   : {sub}/{n} ({100.0*sub/n:.1f}%)")
    print(f"  support SUPERSET of exact opt : {sup}/{n} ({100.0*sup/n:.1f}%)")
    print(f"  support == REVERSED opt set   : {rev_match}/{n} ({100.0*rev_match/n:.1f}%)  <- H1's real claim")
    print(f"  support SUBSET of REVERSED opt: {rev_sub}/{n} ({100.0*rev_sub/n:.1f}%)")
    print(f"  mean |teacher set| = {np.mean(sub_sizes):.2f}   mean |exact opt| = {np.mean(opt_sizes):.2f}")

    # forward vs reversed: which containment holds STRICTLY more often?
    only_fwd = sum(1 for r in rows
                   if set(np.nonzero(r['target'] > 1e-6)[0].tolist()) <= set(r['opt'])
                   and not (set(np.nonzero(r['target'] > 1e-6)[0].tolist()) <= {6 - c for c in r['opt']}))
    only_rev = sum(1 for r in rows
                   if set(np.nonzero(r['target'] > 1e-6)[0].tolist()) <= {6 - c for c in r['opt']}
                   and not (set(np.nonzero(r['target'] > 1e-6)[0].tolist()) <= set(r['opt'])))
    print(f"\n  containment holds ONLY forward : {only_fwd} rows")
    print(f"  containment holds ONLY reversed: {only_rev} rows")
    print(f"  -> {'FORWARD is correct' if only_fwd > only_rev else 'REVERSED is correct' if only_rev > only_fwd else 'AMBIGUOUS'}")

    # label membership: forward vs reversed exact opt
    lab_fwd = sum(1 for r in rows if r["target"].argmax() in set(r["opt"]))
    lab_rev = sum(1 for r in rows if r["target"].argmax() in {6 - c for c in r["opt"]})
    print(f"\n  label IN exact opt set (forward) : {lab_fwd}/{n} ({100.0*lab_fwd/n:.1f}%)")
    print(f"  label IN exact opt set (REVERSED): {lab_rev}/{n} ({100.0*lab_rev/n:.1f}%)")

    # canonical flip rate
    flips = sum(1 for r in rows if r["flipped"])
    print(f"\n  canonical_board flipped: {flips}/{n} ({100.0*flips/n:.1f}%)")
    # among flipped rows, does the teacher support match the flipped-back set?
    fl = [r for r in rows if r["flipped"]]
    if fl:
        ok = 0
        for r in fl:
            sup_set = set(np.nonzero(r["target"] > 1e-6)[0].tolist())
            cb, _ = canonical_board(r["board"])
            ok += int(sup_set == set(r["opt"]))
        print(f"  among flipped rows, support == exact opt: {ok}/{len(fl)}")

    # best_move tie-break column distribution on LOSS
    loss = [r for r in rows if r["status"] == -1]
    bc = Counter(r["bm"] for r in loss)
    print(f"\n=== best_move tie-break on LOSS rows (n={len(loss)}) ===")
    print(f"  column histogram: {dict(sorted(bc.items()))}")
    legal_hist = Counter()
    for r in loss:
        legal_hist[min(c for c in range(7) if r["board"].heights[c] < 6)] += 1
    print(f"  leftmost-legal histogram: {dict(sorted(legal_hist.items()))}")
    print(f"  move_order: {move_order([0,1,2,3,4,5,6])}")
    mc = Counter()
    for r in loss:
        legal = [c for c in range(7) if r["board"].heights[c] < 6]
        mc[move_order(legal)[0]] += 1
    print(f"  first-in-move_order histogram: {dict(sorted(mc.items()))}")

    out = {
        "n": n,
        "support_equals_exact_opt_pct": round(100.0*eq/n, 1),
        "support_subset_of_exact_opt_pct": round(100.0*sub/n, 1),
        "support_superset_of_exact_opt_pct": round(100.0*sup/n, 1),
        "support_equals_reversed_opt_pct": round(100.0*rev_match/n, 1),
        "support_subset_of_reversed_opt_pct": round(100.0*rev_sub/n, 1),
        "mean_teacher_set_size": round(float(np.mean(sub_sizes)), 2),
        "mean_exact_opt_set_size": round(float(np.mean(opt_sizes)), 2),
        "containment_only_forward_rows": only_fwd,
        "containment_only_reversed_rows": only_rev,
        "label_in_exact_opt_forward_pct": round(100.0*lab_fwd/n, 1),
        "label_in_exact_opt_reversed_pct": round(100.0*lab_rev/n, 1),
        "canonical_flip_rate_pct": round(100.0*flips/n, 1),
    }
    dst = ROOT / "runs/_probe/provenance_teacher_set_check.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
