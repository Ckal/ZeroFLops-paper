"""Apply the canonical terminal predicate to every is_full() consumer.

Mechanical, auditable rewrite:
  * `X.is_win() or X.is_full()`  ->  `is_terminal(X)`
  * `X.is_win() != 0 or X.is_full()`  ->  `is_terminal(X)`
  * `winner or board.is_full()`  ->  `is_terminal(board)`   (winner == is_win())
  * a bare `X.is_full()` in a loop/terminal context -> `is_terminal(X)`

The bare cases are reviewed individually below; this script only rewrites
the explicit `is_win() or is_full()` forms, and reports the bare ones for
manual inspection so nothing is changed blind.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(r"C:\Users\Chris4K\projects\ZeroFLops-paper")
FILES = [
    "src/c4/agent.py",
    "src/c4/augment_corpus.py",
    "src/c4/bench_speed.py",
    "src/c4/corpus.py",
    "src/c4/corpus_quality.py",
    "src/c4/outcome_rl.py",
    "src/c4/play.py",
    "src/c4/threat_corpus.py",
    "src/c4/train.py",
]

# (pattern, replacement) -- explicit two-part terminal tests
EXPLICIT = [
    (re.compile(r"(\b[\w.]+)\.is_win\(\)\s*!=\s*0\s+or\s+\1\.is_full\(\)"), r"is_terminal(\1)"),
    (re.compile(r"(\b[\w.]+)\.is_win\(\)\s+or\s+\1\.is_full\(\)"), r"is_terminal(\1)"),
    (re.compile(r"\bwinner\s+or\s+(\b[\w.]+)\.is_full\(\)"), r"is_terminal(\1)"),
]

BARE = re.compile(r"\b([\w.]+)\.is_full\(\)")


def add_import(txt: str, path: pathlib.Path) -> str:
    """Add is_terminal to the board import, if the file imports from board."""
    if "is_terminal" in txt and "import" in txt.split("is_terminal")[0][-200:]:
        pass
    # find an existing `from .board import ...` or `from src.c4.board import ...`
    m = re.search(r"^(\s*)from (\.|src\.c4\.)board import ([^\n]+)$", txt, re.M)
    if not m:
        return txt
    indent, mod, names = m.group(1), m.group(2), m.group(3)
    if "is_terminal" in names:
        return txt
    new_names = names.rstrip()
    # keep the trailing comment if present
    comment = ""
    if "#" in new_names:
        new_names, comment = new_names.split("#", 1)
        new_names = new_names.rstrip()
        comment = "  #" + comment
    new_names = new_names + ", is_terminal"
    return (txt[:m.start()]
            + f"{indent}from {mod}board import {new_names}{comment}"
            + txt[m.end():])


def main() -> int:
    changed = []
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"SKIP {rel} (missing)")
            continue
        txt = p.read_text(encoding="utf-8")
        orig = txt
        for pat, rep in EXPLICIT:
            txt = pat.sub(rep, txt)
        if txt != orig:
            txt = add_import(txt, p)
            p.write_text(txt, encoding="utf-8")
            changed.append(rel)
            print(f"PATCHED {rel}")
        # report leftovers
        rest = BARE.findall(txt)
        if rest:
            print(f"  remaining bare is_full() in {rel}: {rest}")
    print(f"\nchanged: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
