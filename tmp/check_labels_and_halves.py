"""Final characterisation of the corpus last-move-field defect.

Three things must be settled:

A) LABEL VALIDITY on the affected rows, checked with the CORRECT ply
   semantics (ply_of[i, slot] = the ply that slot refers to).  The board
   for the decision at ply p is the position after p moves, i.e. after
   mv[0..p-1].  The previous Q3 pass advanced the board in SLOT order,
   which is wrong when slots are not in ply order -- so its "496 illegal"
   may have been an artefact of that bug.  This pass rebuilds the board
   in PLY order.

B) WHICH HALF of the mirror corpus carries the defect.  corpus_320k_mirror_v2
   interleaves row 2*i = original game i, row 2*i+1 = mirrored game i.
   If the defect lives ONLY in the original half, the mirror builder
   CORRECTED it (reflecting reflect(c) = c), and the mirror A/B was
   therefore trained on a HALF-CLEAN, HALF-CORRUPTED dataset with a
   systematic difference between the halves -- a serious confound.

C) WHY the trainer's validate_stream gate did not catch it.  The gate
   probes only the FIRST row that is long enough, so a defect that starts
   at row 193,459 passes.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.transcript_v2 import PLY_LEN, COL_BYTE_OFFSET  # noqa: E402
from src.c4.board import Board                             # noqa: E402
from src.c4.movemask import validate_stream                 # noqa: E402

CORPUS_DIR = ROOT / "runs/_probe/perply_v23_full"


def flat_of(line: bytes) -> bytes:
    return line.replace(b"\n", b"")


def moves_from_bytes(flat: bytes):
    n = len(flat) // PLY_LEN
    return [flat[p * PLY_LEN + 9] - COL_BYTE_OFFSET for p in range(n)]


def last_of(flat: bytes, p: int):
    v = flat[p * PLY_LEN + 7]
    return None if v < COL_BYTE_OFFSET else v - COL_BYTE_OFFSET


def is_mirrored_last(flat: bytes) -> bool:
    mv = moves_from_bytes(flat)
    for p in range(1, len(mv)):
        l = last_of(flat, p)
        if l is None:
            continue
        if l == mv[p - 1]:
            continue
        return l == 6 - mv[p - 1]
    return False


def main() -> int:
    out = {}

    # ---- A: label validity, board rebuilt in PLY order --------------------
    import numpy as np  # local trusted artifact
    d = np.load(CORPUS_DIR / "merged_repaired.npz")
    tgt, ply_of = d["target"], d["ply_of"]
    p = CORPUS_DIR / "corpus_320k.txt"
    raw = p.read_bytes()
    lines = [ln for ln in raw.split(b"\n") if ln]

    res = {"affected": Counter(), "clean": Counter()}
    illegal_ex = []
    for i in range(0, len(lines), 53):
        ln = lines[i]
        flat = flat_of(ln)
        mv = moves_from_bytes(flat)
        affected = is_mirrored_last(flat)
        # Build the board state for EVERY ply, in ply order.  The board
        # for the decision at ply p is the position AFTER p moves.
        boards = []
        b = Board()
        boards.append((b.p1, b.p2, b.turn))
        for c in mv:
            b.apply_move(c)
            boards.append((b.p1, b.p2, b.turn))
        key = "affected" if affected else "clean"
        for slot in range(tgt.shape[1]):
            ply = int(ply_of[i, slot])
            if ply < 0:
                res[key]["no_label"] += 1
                continue
            if ply >= len(mv):
                res[key]["ply_out_of_range"] += 1
                continue
            lab = int(tgt[i, slot].argmax())
            b2 = Board()
            b2.p1, b2.p2, b2.turn = boards[ply]
            if lab in b2.valid_actions():
                res[key]["legal"] += 1
            else:
                res[key]["illegal"] += 1
                if len(illegal_ex) < 6:
                    illegal_ex.append({"row": i, "slot": slot, "ply": ply,
                                       "label": lab, "affected": affected})
    out["A_label_validity"] = {
        "affected_rows": dict(res["affected"]),
        "clean_rows": dict(res["clean"]),
        "illegal_examples": illegal_ex,
    }
    print("A:", json.dumps(out["A_label_validity"], default=str))

    # ---- B: which half of the mirror corpus carries the defect? ---------
    halves = {}
    for name in ("corpus_320k_mirror.txt", "corpus_320k_mirror_v2.txt"):
        pp = CORPUS_DIR / name
        if not pp.exists():
            continue
        rr = pp.read_bytes()
        lls = [ln for ln in rr.split(b"\n") if ln]
        even = sum(1 for ln in lls[0::2] if is_mirrored_last(flat_of(ln)))
        odd = sum(1 for ln in lls[1::2] if is_mirrored_last(flat_of(ln)))
        halves[name] = {
            "rows": len(lls),
            "even_rows_bad": even, "odd_rows_bad": odd,
            "even_rows": len(lls[0::2]), "odd_rows": len(lls[1::2]),
        }
    out["B_mirror_halves"] = halves
    print("B:", json.dumps(halves, default=str))

    # The ORIGINAL corpus is the even half of the mirror corpus; confirm.
    orig_bad = sum(1 for ln in lines if is_mirrored_last(flat_of(ln)))
    out["B_original_corpus_bad"] = orig_bad
    print(f"B: original corpus bad rows = {orig_bad}")

    # ---- C: why validate_stream did not catch it ------------------------
    probe_ok = True
    probe_err = ""
    try:
        validate_stream(flat_of(lines[0]), "v2")
    except Exception as e:
        probe_ok = False
        probe_err = str(e)
    # and a bad row
    bad_idx = next(i for i, ln in enumerate(lines) if is_mirrored_last(flat_of(ln)))
    bad_err = ""
    try:
        validate_stream(flat_of(lines[bad_idx]), "v2")
    except Exception as e:
        bad_err = str(e)
    out["C_validate_stream"] = {
        "first_row_valid": probe_ok, "first_row_error": probe_err,
        "first_bad_row": bad_idx,
        "first_bad_row_rejected": bool(bad_err), "first_bad_row_error": bad_err,
    }
    print("C:", json.dumps(out["C_validate_stream"], default=str))

    (ROOT / "runs/_probe/corpus_last_header_labels.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwrote runs/_probe/corpus_last_header_labels.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
