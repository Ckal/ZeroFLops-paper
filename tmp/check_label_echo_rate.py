"""Does the TRAINING LABEL itself teach the echo behaviour?

The model copies the opponent's previous column ~50% of the time. Three
possible causes:

  (a) teacher-label bias: the labels themselves point at the opponent's
      previous column, so the model is learning correctly and the TEACHER
      is wrong;
  (b) coverage: the labels for the positions that should teach "do
      something else" were deleted by the terminal bug;
  (c) architecture/optimisation: the labels are fine, but the model cannot
      represent the rule.

This measures (a) directly: for every labelled position, compare the label
column against the opponent's PREVIOUS move in the same game. If the labels
echo as often as the model does, the model is faithful to its supervision and
the fault is upstream (the teacher). If the labels echo much less, the
model is NOT faithful and the fault is in the model.

It also measures (b): late-game label coverage, because the bug deleted the
TAIL of every game in which column 7 filled, and the tail is exactly where
"block the four / build my own four" is distinguishable.

Report BOTH the original and the repaired sidecar so the repair's effect on
the echo rate of the labels is visible.
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

from src.c4.fmt import PLY_LEN, MOVE_OFFSET, COL_BYTE_OFFSET   # noqa: E402

CORPUS = ROOT / "runs/_probe/perply_v23_full/corpus_320k.txt"
ORIG = ROOT / "runs/_probe/perply_v23_full/merged.npz"
REP = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"

MOVE_BYTES = set(range(COL_BYTE_OFFSET, COL_BYTE_OFFSET + 7))


def move_of(seq, ply):
    """Column index (0..6) of the move played at `ply`, or None."""
    mb = seq[ply * PLY_LEN + MOVE_OFFSET]
    if mb not in MOVE_BYTES:
        return None
    return mb - COL_BYTE_OFFSET


def measure(sidecar_path, label, max_rows=None):
    z = np.load(sidecar_path)
    ply_of = z["ply_of"]
    target = z["target"]

    echo = 0
    total = 0
    by_ply_echo = collections.Counter()
    by_ply_total = collections.Counter()
    prev_col_hist = collections.Counter()
    label_hist = collections.Counter()
    # echo rate in the LATE game (ply >= 12), where the terminal bug deleted
    # the supervision
    late_echo = late_total = 0
    early_echo = early_total = 0
    late_cols = collections.Counter()
    early_cols = collections.Counter()

    with open(CORPUS, "rb") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            seq = line.rstrip(b"\r\n")
            nrec = len(seq) // PLY_LEN
            if i >= ply_of.shape[0]:
                break
            for s in range(ply_of.shape[1]):
                p = int(ply_of[i, s])
                if p < 0:
                    continue
                if p == 0:
                    continue                    # no previous move
                if p >= nrec:
                    continue
                prev = move_of(seq, p - 1)
                if prev is None:
                    continue
                col = int(np.argmax(target[i, s]))
                total += 1
                prev_col_hist[prev] += 1
                label_hist[col] += 1
                if col == prev:
                    echo += 1
                if p >= 12:
                    late_total += 1
                    late_cols[col] += 1
                    if col == prev:
                        late_echo += 1
                else:
                    early_total += 1
                    early_cols[col] += 1
                    if col == prev:
                        early_echo += 1
                by_ply_total[p] += 1
                if col == prev:
                    by_ply_echo[p] += 1

    print(f"=== {label} ===")
    print(f"  labelled positions with a previous move: {total:,}")
    print(f"  label ECHOES the previous move          : {echo:,} "
          f"({100.0*echo/max(1,total):.2f}%)")
    print(f"    early game (ply < 12): {100.0*early_echo/max(1,early_total):.2f}% "
          f"of {early_total:,}")
    print(f"    late game  (ply >=12): {100.0*late_echo/max(1,late_total):.2f}% "
          f"of {late_total:,}")
    print(f"  label column histogram: "
          f"{ {k+1: round(100.0*v/max(1,total),2) for k,v in sorted(label_hist.items())} }")
    print(f"  echo rate by ply (ply: rate%):")
    for p in sorted(by_ply_total):
        print(f"      ply {p:>2}: {100.0*by_ply_echo[p]/max(1,by_ply_total[p]):5.1f}% "
              f"({by_ply_total[p]:>7,})")
    return {
        "total": total, "echo": echo,
        "echo_pct": round(100.0 * echo / max(1, total), 3),
        "early_total": early_total,
        "early_echo_pct": round(100.0 * early_echo / max(1, early_total), 3),
        "late_total": late_total,
        "late_echo_pct": round(100.0 * late_echo / max(1, late_total), 3),
        "label_hist": {str(k + 1): v for k, v in sorted(label_hist.items())},
        "late_label_hist": {str(k + 1): v for k, v in sorted(late_cols.items())},
        "early_label_hist": {str(k + 1): v for k, v in sorted(early_cols.items())},
    }


def main() -> int:
    out = {}
    out["original"] = measure(ORIG, "ORIGINAL sidecar (pre-repair)")
    print()
    out["repaired"] = measure(REP, "REPAIRED sidecar")
    print()
    a, b = out["original"], out["repaired"]
    print("=== WHAT THE REPAIR DID TO THE LABEL ECHO RATE ===")
    print(f"  overall echo rate : {a['echo_pct']}% -> {b['echo_pct']}%")
    print(f"  early game        : {a['early_echo_pct']}% -> {b['early_echo_pct']}%")
    print(f"  late game         : {a['late_echo_pct']}% -> {b['late_echo_pct']}%")
    print(f"  late-game labels  : {a['late_total']:,} -> {b['late_total']:,} "
          f"(+{b['late_total']-a['late_total']:,})")
    print()
    print("=== COMPARISON WITH THE MODEL'S OWN ECHO RATE ===")
    print("  model echo rate vs a column-7 stacker : ~50%")
    print("  model echo rate vs a centre bot       : ~80%")
    print("  model echo rate vs random              : ~31.4%")
    print(f"  LABEL echo rate (this measurement)     : {b['echo_pct']}%")
    print()
    if abs(b["echo_pct"] - 50.0) < 12.0:
        print("  => The LABELS echo about as much as the MODEL does.")
        print("     The model is faithfully reproducing its supervision; the")
        print("     ECHO FAULT IS IN THE TEACHER, not in the model architecture.")
    else:
        print("  => The labels echo much LESS than the model does.")
        print("     The model is NOT faithful to its supervision; the fault is")
        print("     in the model/optimisation, not the teacher.")

    (ROOT / "runs/_probe/label_echo_rate.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote runs/_probe/label_echo_rate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
