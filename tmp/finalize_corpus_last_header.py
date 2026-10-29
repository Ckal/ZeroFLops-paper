"""Decisive characterisation of the corpus last-move-field defect.

Established so far:
  * corpus_320k.txt rows 0..193,458 are fine; rows 193,459..319,999
    (126,537 rows, 39.5%) carry `last_header = 6 - own_previous_move`.
  * Re-encoding every row from its OWN MOVE BYTES differs from the file
    ONLY at the last-move header byte (record index 7), for all
    320,000 rows: 0 rows with any other difference, 0 rows with a wrong
    move byte, 0 wrong board bytes.
  * 1,567,958 plies have a wrong last-move byte.
  * The affected rows are mirror copies of earlier rows (100%).

So the board and move bytes are internally coherent, and ONLY the
last-move field contradicts the board.  The fix (re-encode from own move
bytes) is therefore provably minimal.

This script answers the remaining questions:
  Q1  Full-corpus decode rate: original vs repaired.
  Q2  Are the affected rows exact board-reflections of the earlier rows
      they duplicate?  (i.e. is the corpus game + its mirror?)
  Q3  Are the SIDECAR LABELS for affected rows still valid?  (checked with
      the CORRECT ply_of semantics: ply_of[i, slot] = the ply that slot
      refers to; -1 means no label.)
  Q4  Is the same defect present in the mirror corpus, the mirror-v2
      corpus, and the 1500-position holdout corpus?
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.transcript_v2 import (encode_game, decode_game, PLY_LEN,  # noqa: E402
                                  COL_BYTE_OFFSET)
from src.c4.board import Board  # noqa: E402

CORPUS_DIR = ROOT / "runs/_probe/perply_v23_full"


def flat_of(line: bytes) -> bytes:
    return line.replace(b"\n", b"")


def moves_from_bytes(flat: bytes):
    n = len(flat) // PLY_LEN
    return [flat[p * PLY_LEN + 9] - COL_BYTE_OFFSET for p in range(n)]


def last_headers(flat: bytes):
    out = []
    n = len(flat) // PLY_LEN
    for p in range(n):
        out.append(flat[p * PLY_LEN + 7] - COL_BYTE_OFFSET
                   if flat[p * PLY_LEN + 7] >= COL_BYTE_OFFSET else None)
    return out


def is_mirrored_last(flat: bytes) -> bool:
    mv = moves_from_bytes(flat)
    ls = last_headers(flat)
    for p in range(1, len(mv)):
        if ls[p] is None:
            continue
        if ls[p] == mv[p - 1]:
            continue
        return ls[p] == 6 - mv[p - 1]
    return False


def board_bytes(flat: bytes):
    """The 7 board bytes of each ply (records 0..6)."""
    n = len(flat) // PLY_LEN
    return [tuple(flat[p * PLY_LEN + c] for c in range(7)) for p in range(n)]


def main() -> int:
    out = {}

    # ---- Q1: full decode rate, original vs repaired ----------------------
    for name in ("corpus_320k.txt", "corpus_320k_mirror_v2.txt",
                 "corpus_320k_mirror.txt"):
        p = CORPUS_DIR / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        lines = [ln for ln in raw.split(b"\n") if ln]
        ok_o = bad_o = ok_r = bad_r = 0
        n_mirrored = 0
        for ln in lines:
            flat = flat_of(ln)
            try:
                decode_game(flat)
                ok_o += 1
            except Exception:
                bad_o += 1
            if is_mirrored_last(flat):
                n_mirrored += 1
            try:
                canon = encode_game(moves_from_bytes(flat)).replace(b"\n", b"")
                decode_game(canon)
                ok_r += 1
            except Exception:
                bad_r += 1
        out[name] = {
            "rows": len(lines),
            "original_decodes": ok_o,
            "original_fails": bad_o,
            "repaired_decodes": ok_r,
            "repaired_fails": bad_r,
            "rows_with_mirrored_last_field": n_mirrored,
        }
        print(f"{name}: {json.dumps(out[name], default=str)}")

    # ---- Q2: are affected rows board-reflections of earlier rows? --------
    p = CORPUS_DIR / "corpus_320k.txt"
    raw = p.read_bytes()
    lines = [ln for ln in raw.split(b"\n") if ln]
    flats = [flat_of(ln) for ln in lines]
    by_moves = {}
    for i, f in enumerate(flats):
        by_moves.setdefault(tuple(moves_from_bytes(f)), i)

    q2 = Counter()
    refl_exact_board = 0
    checked = 0
    for i, f in enumerate(flats):
        if not is_mirrored_last(f):
            continue
        mv = moves_from_bytes(f)
        r = tuple(6 - c for c in mv)
        j = by_moves.get(r)
        if j is None or j >= i:
            q2["no_earlier_reflection"] += 1
            continue
        q2["earlier_reflection"] += 1
        if checked < 200:
            checked += 1
            fb = board_bytes(f)
            jb = board_bytes(flats[j])
            n = min(len(fb), len(jb))
            # A reflected board has column c of one equal to column 6-c of
            # the other, with ownership bits reflected vertically.
            same = all(fb[p][c] == jb[p][6 - c] for p in range(n) for c in range(7))
            if same:
                refl_exact_board += 1
    out["Q2_affected_rows_are_reflections"] = {
        "affected_rows_with_earlier_reflection": q2["earlier_reflection"],
        "affected_rows_without": q2["no_earlier_reflection"],
        "sampled_reflection_board_bytes_match": refl_exact_board,
        "sampled": checked,
    }
    print(f"Q2: {json.dumps(out['Q2_affected_rows_are_reflections'], default=str)}")

    # ---- Q3: sidecar label validity for affected rows --------------------
    import numpy as np  # local trusted artifact
    npz = CORPUS_DIR / "merged_repaired.npz"
    d = np.load(npz)
    tgt, ply_of = d["target"], d["ply_of"]
    q3 = Counter()
    illegal = []
    for i, f in enumerate(flats):
        if i % 37:            # sample every 37th row
            continue
        if not is_mirrored_last(f):
            continue
        mv = moves_from_bytes(f)
        b = Board()
        for slot in range(tgt.shape[1]):
            ply = int(ply_of[i, slot])
            if ply < 0:
                continue
            if ply >= len(mv):
                q3["ply_out_of_range"] += 1
                continue
            lab = int(tgt[i, slot].argmax())
            if lab in b.valid_actions():
                q3["legal"] += 1
            else:
                q3["illegal"] += 1
                if len(illegal) < 5:
                    illegal.append({"row": i, "slot": slot, "ply": ply,
                                    "label": lab})
            # advance the board by the ply's move
            if ply < len(mv):
                b.apply_move(mv[ply])
    out["Q3_labels_on_affected_rows"] = {
        "legal": q3["legal"], "illegal": q3["illegal"],
        "ply_out_of_range": q3["ply_out_of_range"],
        "illegal_examples": illegal,
    }
    print(f"Q3: {json.dumps(out['Q3_labels_on_affected_rows'], default=str)}")

    # ---- Q4: is the defect in the holdout / mirror corpora too? ---------
    q4 = {}
    for rel in ("runs/_probe/ab1500/corpus.txt",
                "runs/_probe/perply_v23_full/corpus_320k_mirror.txt",
                "runs/_probe/perply_v23_full/corpus_320k_mirror_v2.txt"):
        pp = ROOT / rel
        if not pp.exists():
            q4[rel] = "MISSING"
            continue
        rr = pp.read_bytes()
        lls = [ln for ln in rr.split(b"\n") if ln]
        nm = sum(1 for ln in lls if is_mirrored_last(flat_of(ln)))
        q4[rel] = {"rows": len(lls), "rows_with_mirrored_last_field": nm}
    out["Q4_other_corpora"] = q4
    print(f"Q4: {json.dumps(q4, default=str)}")

    (ROOT / "runs/_probe/corpus_last_header_final.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwrote runs/_probe/corpus_last_header_final.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
