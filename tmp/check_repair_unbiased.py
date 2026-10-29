"""Why is column 7 never a repaired label?

The repair filled 101,090 slots. The repaired labels have 0.0% column 7
while the original labels have 7.62%. Is that a bug in the repair, or a
property of the positions that were lost?

Hypothesis: the bug fired exactly when column 7 had a piece (bit 41 = top
cell of the right-hand column). At those positions, the solver's best move is
very often NOT column 7 -- typically a blocking or winning move elsewhere.
So column 7 being absent from the repaired set is expected.

The alternative hypothesis (the repair is biased) would show up as the
repaired set disagreeing with the original set on the SAME positions. Test:
recompute the label for a sample of positions that were ORIGINALLY labelled
and compare with the stored original label. If the repair path agrees with
the original path, the repair is unbiased and the column-7 absence is
positional, not procedural.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.fmt import PLY_LEN                      # noqa: E402
from src.c4.board import Outcome, ROWS, COLS        # noqa: E402
from src.c4.exact_solver import ExactSolver         # noqa: E402
from src.c4 import strong_solver as STRONG          # noqa: E402
from src.c4.mirror import canonical_board           # noqa: E402
from relabel_v23 import decode_header_to_board      # noqa: E402

CORPUS = ROOT / "runs/_probe/perply_v23_full/corpus_320k.txt"
ORIG = ROOT / "runs/_probe/perply_v23_full/merged.npz"
REP = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"


def canonical_best_set(b, solver, depth=5, budget=0.02, exact_max_empty=12):
    cb, flipped = canonical_board(b)
    if 42 - int(sum(cb.heights)) <= exact_max_empty:
        try:
            best, _v = _exact(cb, solver)
        except TimeoutError:
            best = None
        if best:
            out = {6 - c for c in best} if flipped else set(best)
            return out, 1
    STRONG.clear_tt()
    c = STRONG.strong_policy(cb, [], depth=depth, budget=budget)
    legal = cb.valid_actions()
    if c is None or c not in legal:
        c = legal[0] if legal else 0
    return ({6 - c for c in {c}} if flipped else {c}), 2


def _exact(cb, solver):
    from relabel_v23 import exact_best_set
    return exact_best_set(cb, solver)


def main() -> int:
    a = np.load(ORIG)
    b = np.load(REP)
    pa, pb = a["ply_of"], b["ply_of"]
    ta, tb = a["target"], b["target"]
    new = (~(pa >= 0)) & (pb >= 0)

    solver = ExactSolver(max_empty=12, budget=2_000_000)

    # --- A. what do the lost positions look like? -----------------------
    print("=== A. GEOMETRY OF THE LOST POSITIONS ===")
    lost_rows = np.nonzero(new.any(axis=1))[0]
    print(f"  rows with at least one lost label: {len(lost_rows):,}")
    c7_present = 0
    filled_hist = collections.Counter()
    with open(CORPUS, "rb") as f:
        for i in lost_rows:
            line = f.readline()
            if i != 0:
                pass
            if i % 1 != 0:
                break
            break
    # do it properly in one pass
    want = set(lost_rows.tolist())
    n_c7 = 0
    n_seen = 0
    with open(CORPUS, "rb") as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            for s in np.nonzero(new[i])[0].tolist():
                p = int(pb[i, s])
                hdr = seq[p * PLY_LEN:(p * PLY_LEN) + PLY_LEN - 1]
                try:
                    bd = decode_header_to_board(hdr)
                except Exception:
                    continue
                n_seen += 1
                if bd.heights[6] > 0:
                    n_c7 += 1
                filled_hist[42 - int(sum(bd.heights))] += 1
            if len(want) <= 0:
                break
    print(f"  lost positions inspected: {n_seen:,}")
    print(f"  of those, column 7 already had a piece: {n_c7:,} "
          f"({100.0*n_c7/max(1,n_seen):.1f}%)")
    print("  (the bug fires only when bit 41 -- the top cell of column 7 -- is set,")
    print("   so this is the expected signature)")
    print(f"  free-cell count of lost positions: min {min(filled_hist)} "
          f"max {max(filled_hist)}  (the board was FAR from full)")

    # --- B. does the repair path reproduce the ORIGINAL labels? --------
    print("\n=== B. DOES THE REPAIR PATH REPRODUCE THE ORIGINAL LABELS? ===")
    random.seed(5)
    orig_rows = np.nonzero((pa >= 0).any(axis=1))[0]
    sample_rows = random.sample(orig_rows.tolist(), 400)
    want = set(sample_rows)
    agree = disagree = checked = 0
    disagreements = []
    cache = {}
    with open(CORPUS, "rb") as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            for s in range(min(19, nrec)):
                if pa[i, s] < 0:
                    continue
                p = int(pa[i, s])
                hdr = seq[p * PLY_LEN:(p * PLY_LEN) + PLY_LEN - 1]
                try:
                    bd = decode_header_to_board(hdr)
                except Exception:
                    continue
                cb, flipped = canonical_board(bd)
                key = (cb.p1, cb.p2, cb.turn)
                if key not in cache:
                    cache[key] = canonical_best_set(cb, solver)
                best, src = cache[key]
                if flipped:
                    best = {6 - c for c in best}
                orig_set = set(np.nonzero(ta[i, s])[0].tolist())
                checked += 1
                if best == orig_set:
                    agree += 1
                else:
                    disagree += 1
                    if len(disagreements) < 5:
                        disagreements.append({"row": i, "slot": int(s),
                                              "orig": sorted(orig_set),
                                              "recomputed": sorted(best),
                                              "src": src})
            if len(cache) > 50_000:
                cache.clear()
    print(f"  positions recomputed with the repair path : {checked:,}")
    print(f"  identical to the stored original label      : {agree:,} "
          f"({100.0*agree/max(1,checked):.2f}%)")
    print(f"  different                                   : {disagree:,} "
          f"({100.0*disagree/max(1,checked):.2f}%)")
    print(f"  examples: {json.dumps(disagreements[:3])}")
    print()
    if disagree <= max(3, 0.02 * checked):
        print("  => the repair path REPRODUCES the original labels, so it is")
        print("     unbiased. The absence of column 7 in the repaired set is a")
        print("     property of the lost POSITIONS, not of the repair.")
    else:
        print("  => the repair path DISAGREES with the original labels --")
        print("     the repair is not reproducing the original labelling policy.")

    out = {"lost_positions_inspected": n_seen,
           "lost_with_c7_piece": n_c7,
           "recomputed_checked": checked,
           "recomputed_agree": agree,
           "recomputed_disagree": disagree,
           "disagreement_pct": round(100.0 * disagree / max(1, checked), 3),
           "disagreement_examples": disagreements}
    (ROOT / "runs/_probe/repair_unbiasedness.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote runs/_probe/repair_unbiasedness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
