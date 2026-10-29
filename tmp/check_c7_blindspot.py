"""Did the bug create a SYSTEMATIC blind spot in the c7 region?

The bug fired exactly when bit 41 was set -- the top cell of column 7 (the
right-hand column in internal 0..6 indexing, i.e. the LAST column the player
sees as column 7). So the relabeller skipped labels for positions in which
column 7 was occupied.

The model's diagnosed defect is an ECHO tendency: it answers column X with
column X. To refuse an echo, the model has to learn what to do in a position
where the opponent has ALREADY stacked column 7 -- i.e. exactly the region
that was never labelled.

Hypothesis: the training signal for "column 7 is occupied, do something else"
was largely DELETED by the bug. That would connect two findings that had
looked unrelated:
  1. the false draw after six column-7 moves
  2. the model's echo-column defect against a column-7 stacker

Measure:
  - fraction of all corpus positions in which column 7 was occupied;
  - fraction of those positions that got a label (original vs repaired);
  - the same for the mirror corpus;
  - what the repaired labels teach in that region (do they point AWAY from
    column 7?).

This is a causal-mechanism measurement, not a correlation: the repair changes
only the label set, so any difference between the original and repaired label
distribution IN THE c7 REGION is caused by the bug.
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


def scan(corpus_path, sidecar_path, max_rows=None):
    z = np.load(sidecar_path)
    ply_of = z["ply_of"]
    target = z["target"]
    P = ply_of.shape[1]

    tot_pos = 0
    c7_pos = 0
    c7_lab = 0
    lab_pos = 0
    c7_cols = collections.Counter()
    nonc7_cols = collections.Counter()
    legal_in_c7 = 0
    legal_in_c7_nolabel = 0

    with open(corpus_path, "rb") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            if i >= ply_of.shape[0]:
                break
            for s in range(min(P, nrec)):
                p = ply_of[i, s]
                if p < 0:
                    continue
                hdr = seq[p * PLY_LEN:(p * PLY_LEN) + PLY_LEN - 1]
                try:
                    b = decode_header_to_board(hdr)
                except Exception:
                    continue
                lab_pos += 1
                col = int(np.argmax(target[i, s]))
                if b.heights[6] > 0:
                    c7_pos += 1
                    c7_lab += 1
                    c7_cols[col] += 1
                    legal = b.valid_actions()
                    if 6 in legal:
                        legal_in_c7 += 1
                else:
                    nonc7_cols[col] += 1
    return dict(lab_pos=lab_pos, c7_lab=c7_lab, c7_cols=dict(c7_cols),
                nonc7_cols=dict(nonc7_cols),
                legal_in_c7=legal_in_c7)


def scan_positions(corpus_path, sidecar_path, max_rows=None):
    """Count ALL positions (labelled or not) and how many have c7 occupied."""
    z = np.load(sidecar_path)
    ply_of = z["ply_of"]
    P = ply_of.shape[1]
    n_pos = 0
    n_c7 = 0
    n_holes = 0
    n_holes_c7 = 0
    with open(corpus_path, "rb") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            if i >= ply_of.shape[0]:
                break
            for s in range(min(P, nrec)):
                p = ply_of[i, s]
                if p < 0:
                    # a hole: the slot was never labelled. Which ply is it?
                    n_holes += 1
                    continue
                hdr = seq[p * PLY_LEN:(p * PLY_LEN) + PLY_LEN - 1]
                try:
                    b = decode_header_to_board(hdr)
                except Exception:
                    continue
                n_pos += 1
                if b.heights[6] > 0:
                    n_c7 += 1
    return dict(positions=n_pos, c7_positions=n_c7, holes=n_holes)


def main() -> int:
    out = {}
    print("=== ORIGINAL CORPUS + ORIGINAL SIDECAR ===")
    a = scan("runs/_probe/perply_v23_full/corpus_320k.txt",
             "runs/_probe/perply_v23_full/merged.npz")
    print(f"  labelled positions        : {a['lab_pos']:,}")
    print(f"  of those, c7 occupied     : {a['c7_lab']:,} "
          f"({100.0*a['c7_lab']/max(1,a['lab_pos']):.2f}%)")
    print(f"  c7 legal in those         : {a['legal_in_c7']:,}")
    print(f"  label column when c7 occ : {a['c7_cols']}")
    print(f"  label column when c7 free: {a['nonc7_cols']}")
    out["original"] = a

    print("\n=== ORIGINAL CORPUS + REPAIRED SIDECAR ===")
    b = scan("runs/_probe/perply_v23_full/corpus_320k.txt",
             "runs/_probe/perply_v23_full/merged_repaired.npz")
    print(f"  labelled positions        : {b['lab_pos']:,}")
    print(f"  of those, c7 occupied     : {b['c7_lab']:,} "
          f"({100.0*b['c7_lab']/max(1,b['lab_pos']):.2f}%)")
    print(f"  c7 legal in those         : {b['legal_in_c7']:,}")
    print(f"  label column when c7 occ : {b['c7_cols']}")
    print(f"  label column when c7 free: {b['nonc7_cols']}")
    out["repaired"] = b

    print("\n=== THE DELTA (caused by the repair, i.e. by the bug's inverse) ===")
    dl = b["lab_pos"] - a["lab_pos"]
    dc7 = b["c7_lab"] - a["c7_lab"]
    print(f"  extra labels total        : {dl:,}")
    print(f"  extra labels in c7 region : {dc7:,} "
          f"({100.0*dc7/max(1,dl):.1f}% of all recovered labels)")
    print(f"  c7-region label coverage  : {a['c7_lab']:,} -> {b['c7_lab']:,} "
          f"({100.0*(b['c7_lab']-a['c7_lab'])/max(1,a['c7_lab']):.1f}% more)")
    print()
    # What do the recovered c7-region labels teach?
    recovered_c7 = {k: b["c7_cols"].get(k, 0) - a["c7_cols"].get(k, 0)
                    for k in set(b["c7_cols"]) | set(a["c7_cols"])}
    recovered_c7 = {k: v for k, v in recovered_c7.items() if v}
    tot = sum(recovered_c7.values())
    print(f"  RECOVERED labels in the c7 region: {tot:,}")
    for k in sorted(recovered_c7):
        print(f"      column {k+1}: {recovered_c7[k]:>7,} "
              f"({100.0*recovered_c7[k]/max(1,tot):.1f}%)")
    if 6 in recovered_c7:
        print(f"  -> the recovered labels point AT column 7 in "
              f"{100.0*recovered_c7[6]/max(1,tot):.1f}% of cases")
    else:
        print(f"  -> the recovered labels NEVER point at column 7 "
              f"({100.0*recovered_c7.get(6,0)/max(1,tot):.1f}%)")
    out["delta"] = {"extra_labels": dl, "extra_in_c7_region": dc7,
                     "recovered_c7_columns": recovered_c7}

    (ROOT / "runs/_probe/c7_region_blindspot.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote runs/_probe/c7_region_blindspot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
