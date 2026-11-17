"""Repair a corrupted transcript-v2 corpus: rewrite the last-move header field.

Background
----------
src/c4/corpus_integrity.py refuses to train on corpora whose rows carry an
inconsistent last-move header field (record index 7 of each 10-byte ply): the
value at ply p must equal the move played at ply p-1 (the byte the engine
records as "last move seen so far").  A corpus-builder bug (a mirror-builder)
wrote `6 - own_previous_move` instead — a column reflection — on a trailing
slice of the file.  The board bytes and the move bytes themselves are
internally consistent; only record 7 contradicts the board.

This tool performs the MINIMAL repair: for each affected ply p >= 1 it sets
record 7 to the move byte of ply p-1, exactly what a correct encode would
produce.  Nothing else in the line is ever changed.

Safety model
------------
* Read-only until the final write: every decision is printed first.
* The fix is an IDENTITY on every clean row (verified below), so a clean
  corpus passes through byte-for-byte unchanged (same sha256).
* A backup of the original is always written before the repaired file.
* After the repair the full scan + a decode sample must both pass, or the
  write is rolled back from the backup.

Usage
-----
    python tools/repair_corpus_last_field.py <corpus.txt> [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.corpus_integrity import last_field_violations  # noqa: E402
from src.c4.transcript_v2 import decode_game  # noqa: E402

PLY_LEN = 10
HDR_OFF = 7
MOV_OFF = 9


def fix_line(ln: bytes) -> bytes:
    """Return ln with record 7 of every ply p>=1 set to ply p-1's move byte.

    Identity on rows that already satisfy the invariant.
    """
    if len(ln) % PLY_LEN:
        return ln
    f = bytearray(ln)
    for p in range(1, len(f) // PLY_LEN):
        f[p * PLY_LEN + HDR_OFF] = f[(p - 1) * PLY_LEN + MOV_OFF]
    return bytes(f)


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output path (default: overwrite corpus in place)")
    ap.add_argument("--allow-malformed", action="store_true",
                    help="do not abort on rows whose length is not a multiple of 10")
    args = ap.parse_args(argv)

    corpus: pathlib.Path = args.corpus
    if not corpus.exists():
        print(f"error: {corpus} does not exist", file=sys.stderr)
        return 2
    out: pathlib.Path = args.out or corpus
    if out == corpus:
        backup = corpus.with_name(corpus.name + ".bak-corruption-repair")
    else:
        backup = out.with_name(out.name + ".bak-corruption-repair")

    lines = [l for l in corpus.read_bytes().split(b"\n") if l]
    n = len(lines)
    violations = [i for i, ln in enumerate(lines) if last_field_violations(ln)]
    changed = 0
    identity_on_clean = 0
    malformed = 0
    for i, ln in enumerate(lines):
        fixed = fix_line(ln)
        if not last_field_violations(ln):
            if fixed == ln:
                identity_on_clean += 1
            else:
                print(f"error: fix is NOT identity on clean row {i}", file=sys.stderr)
                return 3
        else:
            if fixed == ln:
                print(f"error: fix does not change affected row {i}", file=sys.stderr)
                return 3
            changed += 1
        if len(ln) % PLY_LEN:
            malformed += 1
    print(f"lines:            {n}")
    print(f"violating rows:   {len(violations)}"
          + (f"  (first {min(violations)}, last {max(violations)})" if violations else ""))
    print(f"rows rewritten:   {changed}")
    print(f"identity(clean):  {identity_on_clean} rows unchanged")
    if malformed:
        if not args.allow_malformed:
            print(f"error: {malformed} rows are not a multiple of {PLY_LEN} bytes; "
                  f"aborting (pass --allow-malformed to override)", file=sys.stderr)
            return 4
        print(f"warning: {malformed} malformed-length rows pass through unchanged")
    bad = len(violations) - changed
    if bad:
        print(f"error: {bad} violating rows could not be repaired", file=sys.stderr)
        return 5

    # decode sample of repaired rows (original bad rows that decode now)
    sample = violations[:200] if violations else list(range(min(200, n)))
    ok = 0
    for i in sample:
        try:
            decode_game(fix_line(lines[i]))
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"error: repaired row {i} still fails decode: {e}", file=sys.stderr)
            return 6
    print(f"decode sample:    {ok}/{len(sample)} repaired rows decode cleanly")

    body = b"\n".join(fix_line(ln) for ln in lines) + b"\n"
    if not any(violations):
        print("corpus is already clean; no write performed")
        return 0
    if out == corpus:
        if not backup.exists():
            shutil.copy2(corpus, backup)
            print(f"backup:           {backup}")
    out.write_bytes(body)
    print(f"repaired file:    {out}")
    print(f"sha256 original:  {sha256(backup if out == corpus else corpus)}")
    print(f"sha256 repaired:  {sha256(out)}")

    # final verification pass
    v2 = [i for i, ln in enumerate(
          [l for l in out.read_bytes().split(b"\n") if l])
          if last_field_violations(ln)]
    if v2:
        print(f"ROLLBACK: {len(v2)} rows still violate; restoring backup")
        shutil.copy2(backup, out)
        return 7
    print("final scan:       0 rows with last-move header violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
