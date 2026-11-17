"""Decisive check: do the teacher sidecar labels agree with the ACTUAL move bytes?

The trainer has two different notions of "correct":
  * hard_acc / legal_acc  -> measured against REAL move bytes in the transcript
  * holdout_optimal_pct   -> measured against the teacher sidecar (npz targets)

If these two disagree, the run cannot optimise both at once. And the model's
static per-column bias (col 5) tells us which one it was actually fit to.
"""
import json
from pathlib import Path

import numpy as np

SPLIT = Path("runs/c4/posdata_split")
MOVE_OFFSET = 9
PLY_LEN = 10
COL_BYTE_OFFSET = 0x41


def rows(p):
    return [l for l in p.read_bytes().split(b"\n") if l]


def final_move_col(seq):
    """Column of the LAST move in the transcript, from the real move byte."""
    if len(seq) < PLY_LEN:
        return None
    # move byte of the final ply
    b = seq[(len(seq) // PLY_LEN - 1) * PLY_LEN + MOVE_OFFSET]
    return b - COL_BYTE_OFFSET if COL_BYTE_OFFSET <= b < COL_BYTE_OFFSET + 7 else None


def all_move_cols(seq):
    out = []
    for i in range(MOVE_OFFSET, len(seq), PLY_LEN):
        b = seq[i]
        if COL_BYTE_OFFSET <= b < COL_BYTE_OFFSET + 7:
            out.append(b - COL_BYTE_OFFSET)
    return out


def main():
    for name in ["train", "holdout"]:
        seqs = rows(SPLIT / f"{name}_corpus.txt")
        T = np.load(SPLIT / f"{name}_targets.npz")["target"]
        n = min(len(seqs), len(T))
        print("=" * 72)
        print("%s: rows=%d  sidecar=%d" % (name, len(seqs), len(T)))

        # --- actual played move columns over the WHOLE transcript ---
        allc = []
        for s in seqs:
            allc.extend(all_move_cols(s))
        allc = np.array(allc)
        cnt = np.bincount(allc, minlength=7)
        print("  ACTUAL move-byte column dist (all plies, n=%d):" % len(allc))
        print("    ", cnt.tolist())
        print("     pct:", np.round(100.0 * cnt / len(allc), 2).tolist())
        print("     majority: col %d @ %.2f%%" % (int(cnt.argmax()), 100.0 * cnt.max() / len(cnt)))

        # --- final-ply actual move vs teacher label ---
        agree = 0
        teacher_first = 0
        n_cmp = 0
        for i in range(n):
            c = final_move_col(seqs[i])
            if c is None:
                continue
            n_cmp += 1
            if np.isclose(T[i, c], T[i].max(), atol=1e-6):
                agree += 1
            if int(T[i].argmax()) == c:
                teacher_first += 1
        if n_cmp:
            print("  final-ply: teacher marks ACTUAL move as optimal : %d/%d (%.2f%%)"
                  % (agree, n_cmp, 100.0 * agree / n_cmp))
            print("  final-ply: teacher ARGMAX equals actual move      : %d/%d (%.2f%%)"
                  % (teacher_first, n_cmp, 100.0 * teacher_first / n_cmp))

        # --- what does the teacher's argmax distribution look like vs actual ---
        tam = T.argmax(axis=1)
        tcnt = np.bincount(tam, minlength=7)
        print("  TEACHER argmax dist:", tcnt.tolist())
        print("  TEACHER argmax pct:", np.round(100.0 * tcnt / len(tam), 2).tolist())

        # --- teacher label for the ACTUAL final move, probability mass ---
        prob = []
        for i in range(n):
            c = final_move_col(seqs[i])
            if c is None:
                continue
            prob.append(T[i, c])
        prob = np.array(prob)
        print("  teacher target prob assigned to ACTUAL final move:")
        print("     mean %.4f  median %.4f  frac==max(%.3f)  frac==0(%.3f)"
              % (prob.mean(), np.median(prob), np.isclose(prob, prob.max()).mean(),
                 (prob <= 1e-9).mean()))
        print()

    # --- what policy does the model's static bias correspond to? ---
    print("=" * 72)
    print("Model static column preference (from logits): "
          "col5 > col4 > col6 > col3 > col2 > col1 > col0")
    print("Model picks col5 on 3789/4000 = 94.7% of holdout positions.")
    for name in ["train", "holdout"]:
        seqs = rows(SPLIT / f"{name}_corpus.txt")
        allc = []
        for s in seqs:
            allc.extend(all_move_cols(s))
        allc = np.array(allc)
        pct5 = 100.0 * (allc == 5).mean()
        print("  %-8s: actual moves on col5 = %.2f%%" % (name, pct5))


if __name__ == "__main__":
    main()
