"""Precise trigger of the is_full() bug: which column-7 state did it fire on?

The bug was:
    return bool((self.mask >> (ROWS * COLS - 1)) & 1)
i.e. bit 41 = ROWS*COLS-1 = the TOP cell of the right-hand column.

Cells fill bottom-up, so the top cell of a column is occupied ONLY when that
column is COMPLETELY FULL (6/6). Therefore the bug fired exactly when
column 7 was FULL -- not merely when column 7 had a piece.

That is a much sharper statement than "column 7 was occupied", and it
changes the mechanism story:

  * the false draw the user hit: both sides played column 7 six times ->
    column 7 full -> bug fired -> DRAW with 36 empty cells;
  * every lost LABEL is a position in which column 7 was full and
    therefore column 7 was ILLEGAL;
  * the consequence is not a regional deletion but a TAIL DELETION: once
    column 7 fills, every later ply of that game loses its label, so the
    game stops being supervised at that point.

Measure the height distribution of column 7 at the lost positions, and the
ply at which each truncated game lost supervision.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.c4.fmt import PLY_LEN                   # noqa: E402
from relabel_v23 import decode_header_to_board   # noqa: E402

CORPUS = ROOT / "runs/_probe/perply_v23_full/corpus_320k.txt"
ORIG = ROOT / "runs/_probe/perply_v23_full/merged.npz"
REP = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"


def main() -> int:
    a = np.load(ORIG)
    b = np.load(REP)
    pa, pb = a["ply_of"], b["ply_of"]
    new = (~(pa >= 0)) & (pb >= 0)

    rows = np.nonzero(new.any(axis=1))[0]
    want = set(rows.tolist())
    heights = collections.Counter()
    legal_c7 = 0
    n_seen = 0
    # at which ply did each affected game lose supervision?
    lost_plys = []
    game_end_plys = []
    filled_col_hist = collections.Counter()

    with open(CORPUS, "rb") as fh:
        for i, line in enumerate(fh):
            if i not in want:
                continue
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            sl = np.nonzero(new[i])[0].tolist()
            if not sl:
                continue
            # the ply range that lost supervision
            ps = sorted(int(pb[i, s]) for s in sl)
            lost_plys.extend(ps)
            # the last ply of the game
            game_end_plys.append(nrec - 1)
            for s in sl:
                p = int(pb[i, s])
                hdr = seq[p * PLY_LEN:(p * PLY_LEN) + PLY_LEN - 1]
                try:
                    bd = decode_header_to_board(hdr)
                except Exception:
                    continue
                n_seen += 1
                heights[bd.heights[6]] += 1
                if 6 in bd.valid_actions():
                    legal_c7 += 1
                filled_col_hist[int(sum(bd.heights))] += 1

    print("=== THE EXACT TRIGGER ===")
    print(f"  lost label positions inspected : {n_seen:,}")
    print(f"  column-7 height at those       : {dict(sorted(heights.items()))}")
    print(f"  column 7 LEGAL at those        : {legal_c7:,} "
          f"({100.0*legal_c7/max(1,n_seen):.2f}%)")
    print()
    if set(heights) == {6}:
        print("  => CONFIRMED: the bug fired ONLY when column 7 was FULL (6/6).")
        print("     bit 41 is the TOP cell of that column, and cells fill bottom-up,")
        print("     so the top cell is set only when the column is complete.")
        print("     Consequently column 7 was ILLEGAL in every lost position:")
        print("     the lost labels were never 'play column 7' labels.")
    else:
        print(f"  => NOT a pure column-7-full trigger: heights {dict(heights)}")
    print()

    print("=== IT IS A TAIL DELETION, NOT A REGIONAL ONE ===")
    lost_plys.sort()
    n = len(lost_plys)
    print(f"  lost labels: {n:,}")
    print(f"  lost-ply distribution: min {lost_plys[0]}, "
          f"median {lost_plys[n//2]}, max {lost_plys[-1]}")
    print(f"  lost labels at ply 0 (the opening move): "
          f"{sum(1 for p in lost_plys if p == 0):,}")
    # per affected game: how many trailing plies lost supervision
    with open(CORPUS, "rb") as fh:
        for i, line in enumerate(fh):
            if i not in want:
                continue
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            sl = np.nonzero(new[i])[0].tolist()
            ps = sorted(int(pb[i, s]) for s in sl)
            # contiguity: are the lost plies a contiguous tail?
            gaps = sum(1 for k in range(1, len(ps)) if ps[k] != ps[k-1] + 1)
            if gaps == 0:
                filled_col_hist["contiguous"] += 1
            else:
                filled_col_hist["noncontiguous"] += 1
    print(f"  affected games with a CONTIGUOUS lost tail : "
          f"{filled_col_hist['contiguous']:,}")
    print(f"  affected games with a NON-contiguous loss  : "
          f"{filled_col_hist['noncontiguous']:,}")
    print()
    print("  => A contiguous tail means the game simply STOPPED being supervised")
    print("     once column 7 filled. That is a truncation of the game's")
    print("     supervision, not the removal of a region.")

    out = {"lost_inspected": n_seen,
           "col7_height_hist": {str(k): v for k, v in sorted(heights.items())},
           "col7_legal_at_lost": legal_c7,
           "lost_ply_min": lost_plys[0],
           "lost_ply_median": lost_plys[n // 2],
           "lost_ply_max": lost_plys[-1],
           "lost_at_ply0": sum(1 for p in lost_plys if p == 0),
           "contiguous_tail_games": filled_col_hist["contiguous"],
           "noncontiguous_games": filled_col_hist["noncontiguous"]}
    (ROOT / "runs/_probe/isfull_exact_trigger.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote runs/_probe/isfull_exact_trigger.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
