#!/usr/bin/env python
"""Strip calendar dates and clock times from a tree of text files.

WHY
---
A public release should not carry the private working chronology of the
project. The research log is a dated journal; publishing it with every
timestamp in it makes the repository read like an internal transcript
rather than a release. The dates carry no information a reader needs: the
appendices are already ordered, and the dayparts ("late night",
"evening") carry the sense of sequence without pinning anything to a
calendar.

WHAT IS REMOVED
----------------
  * ISO dates          2026-10-29
  * ISO datetimes      2026-10-29 02:34
  * European dates     29.10.26, 29.10.2026
  * Spelled dates      October 29, 2026 / 28 Oct 2026
  * Clock times        02:34, 00:01:55, 21:25

WHAT IS KEPT
------------
  * Dayparts           "late night", "evening", "night"
  * Version/run labels V24, step 18000, Appendix D31
  * Every number that is a measurement
  * Python indentation and Markdown hard line breaks (two trailing spaces)

SAFETY
------
This script rewrites source files, so it must never reformat code. Two
rules enforce that:

  1. Only the specific token is removed. No whitespace is collapsed and
     no line is re-wrapped, because both would break Python indentation
     and Markdown hard breaks.
  2. Every rewritten .py file is byte-compiled before the change is
     accepted. If a file stops compiling, it is restored and the run
     fails loudly rather than leaving broken source behind.

Usage:
    python tmp/strip_dates.py <dir-or-file> [...]
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

TEXT_EXT = {
    ".md", ".py", ".txt", ".sh", ".yml", ".yaml", ".json", ".html",
    ".css", ".js", ".cfg", ".ini", ".toml",
}

# --- Phrase substitutions ------------------------------------------------
# Applied BEFORE the generic date removal so the sentences stay readable.

PHRASES: list[tuple[str, str]] = [
    # Status headers that would otherwise be left dangling.
    (r"\*\*As of \d{4}-\d{2}-\d{2}, late night\.\*\*", "**Current state.**"),
    (r"\*\*As of \d{4}-\d{2}-\d{2}, late\.\*\*", "**Current state.**"),
    (r"\*\*As of \d{4}-\d{2}-\d{2}\.\*\*", "**Current state.**"),
    (r"\*\*Date:\*\*\s*\d{4}-\d{2}-\d{2}\s*\(evening, before the midnight V24 launch\)",
     "**Snapshot:** before the midnight V24 launch"),
    (r"\*\*Date:\*\*\s*\d{4}-\d{2}-\d{2}\s*evening, before the midnight V24 launch\.",
     "**Snapshot:** before the midnight V24 launch."),
    (r"\*\*Date:\*\*\s*\d{4}-\d{2}-\d{2}\s*\(night addendum\)",
     "**Snapshot:** night addendum"),
    (r"\*\*Date:\*\*\s*\d{4}-\d{2}-\d{2}[^\n]*", "**Snapshot:** current revision"),
    (r"\*\*Opened:\*\*\s*\d{4}-\d{2}-\d{2}", "**Opened:** at release"),
    (r"\*\*Recorded:\*\*\s*\d{4}-\d{2}-\d{2}", "**Recorded:** at release"),
    (r"Recorded \d{4}-\d{2}-\d{2}\.", "Recorded at release."),
    (r"Recorded \d{4}-\d{2}-\d{2},", "Recorded at release,"),

    # Sentences where the date was the subject.
    (r"Found \d{4}-\d{2}-\d{2}, \*\*after\*\*", "Found **after**"),
    (r"Also resolved \d{4}-\d{2}-\d{2}:", "Also resolved:"),
    (r"Also, added \d{4}-\d{2}-\d{2}:", "Also added:"),
    (r"and, added \d{4}-\d{2}-\d{2}:", "and, added:"),
    (r"Re-verified \d{4}-\d{2}-\d{2} \d{2}:\d{2}", "Re-verified"),
    (r"Re-verified \d{4}-\d{2}-\d{2}", "Re-verified"),
    (r"written before \d{4}-\d{2}-\d{2}", "written by the older exporter"),
    (r"On \d{4}-\d{2}-\d{2} the machine slept", "The machine slept"),
    (r"the \d{4}-\d{2}-\d{2} sleep killed", "the host sleep killed"),
    (r"The host slept at ~\d{2}:\d{2} on \d{4}-\d{2}-\d{2} and killed",
     "The host slept and killed"),
    (r"written at \d{2}:\d{2}:\d{2} on \d{4}-\d{2}-\d{2}", "written during that run"),
    (r"\(launched \d{4}-\d{2}-\d{2} \d{2}:\d{2}\)", ""),
    (r"launched \d{4}-\d{2}-\d{2} \d{2}:\d{2}", "launched"),
    (r"Replaced by ``tmp/test_ui_isfull_v2\.py`` on \d{4}-\d{2}-\d{2}\.",
     "Replaced by ``tests/test_ui_isfull_v2.py``."),
    (r"Measured consequence on \d{4}-\d{2}-\d{2}:", "Measured consequence:"),
    (r"BUG \(fixed \d{4}-\d{2}-\d{2}\)", "BUG (fixed)"),
    (r"SCALE CONSISTENCY \(fixed \d{4}-\d{2}-\d{2}\)", "SCALE CONSISTENCY (fixed)"),
    (r"DEFAULT CHANGED 0\.2 -> 0\.0 \(\d{4}-\d{2}-\d{2}\)",
     "DEFAULT CHANGED 0.2 -> 0.0"),
    (r"WHY \(\d{4}-\d{2}-\d{2}\)", "WHY"),
    (r"WHY THIS EXISTS \(\d{4}-\d{2}-\d{2}\)", "WHY THIS EXISTS"),
    (r"PREMISE FIX \(\d{4}-\d{2}-\d{2}\)", "PREMISE FIX"),
    (r"ADDENDUM \d{4}-\d{2}-\d{2} \d{2}:\d{2} ", "ADDENDUM "),
    (r"ADDENDUM \d{4}-\d{2}-\d{2} ", "ADDENDUM "),
    (r"## 0c\. THE REAL ANSWER TO \"IS IT THE DATA\?\" — \d{4}-\d{2}-\d{2}",
     "## 0c. THE REAL ANSWER TO \"IS IT THE DATA?\""),
    (r"\*\*Date:\*\*\s*\(night addendum\)", "**Snapshot:** night addendum"),
    (r"\*\*Date:\*\*\s*\(evening, before the midnight V24 launch\)",
     "**Snapshot:** before the midnight V24 launch"),
    (r"(?m)^Date:\s*\(night addendum\)$", "**Snapshot:** night addendum"),
    (r"(?m)^Date:\s*\(.*?\)$", "**Snapshot:** current revision"),
    (r"\*\*Date:\*\*\s*\(night\)\s*·\s*\*\*Folder:\*\*\s*`[^`]*`",
     "**Snapshot:** current revision"),
]

# --- Generic patterns ----------------------------------------------------
#
# The patterns and the validity test live in `date_spec.py`, which the
# checker imports too. The two tools MUST agree: if the checker flagged
# tokens the scrubber was never going to remove, the staging gate would
# fail on things that are not dates at all -- channel-share triples such
# as `86/80/91`, board-fill fractions such as `21.0/42`, and section
# numbers such as `9.7.1`. A gate that cries wolf is a gate that gets
# switched off.
#
# The month patterns never let an abbreviation swallow a following word:
# "Dec" + "ided" used to match "decided", which made the scrubber delete
# real source lines. Two further rules apply to every month pattern:
#   * the month must not be followed by a letter, so "Decided" and
#     "Maybe" are not months;
#   * the gap between the month and the day is horizontal whitespace
#     only, so a date can never span a line break and delete the
#     code in between.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import date_spec as DS  # noqa: E402

RE_ISO_RANGE = DS.RE_ISO_RANGE
RE_ISO_DT = DS.RE_ISO_DT
RE_MONTH_DMY = DS.RE_MONTH_DMY
RE_DAY_MONTH_Y = DS.RE_DAY_MONTH_Y
RE_MONTH_DAY = DS.RE_MONTH_DAY
RE_DAY_MONTH = DS.RE_DAY_MONTH
RE_CLOCK = DS.RE_CLOCK

# The three-part numeric forms are matched by a callback, because only
# the shared validity test knows whether a token is really a date.
RE_NUMERIC_DATE = DS.NUMERIC_TRIPLE


def _ast_signature(source: str) -> str:
    """A structural fingerprint of a Python source file.

    Comments and string literals are dropped, because those are exactly
    what carries a date and is expected to change. Everything else --
    every operator, name, call, and argument -- must match exactly.
    """
    tree = ast.parse(source)

    class Strip(ast.NodeTransformer):
        def visit_Constant(self, node):  # noqa: N802
            if isinstance(node.value, str):
                return ast.Constant(value="")
            return node

    tree = Strip().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True)


# A removed date leaves a hole. To tidy ONLY those holes -- and never
# touch a "()" that the author wrote -- each removal is first replaced
# with a sentinel marker. The marker is resolved at the end, when the
# script knows exactly which characters it put there.
#
# The earlier implementation ran a global "remove empty parentheses"
# rule over the whole file. In Markdown that silently rewrote code
# references such as `Board.is_win()` into `Board.is_win`, which is a
# different call. The marker approach cannot do that: it only looks
# at the characters immediately around a removal.
SENTINEL = "\x00DATE\x00"


def _resolve_holes(text: str) -> str:
    """Remove the sentinels left by date removal, tidying only around them.

    The order below is deliberate. Each rule only acts on text that
    still contains a sentinel, so a sentence the author wrote is never
    altered -- only the neighbourhood of a removed date is.

      1. a date alone inside brackets      "(2026-10-29)"   -> ""
      2. a dangling preposition             "(night of <d>)" -> "(night)"
      3. the spaces a removal left behind
      4. the sentinel itself
    """
    # 1. A date alone inside brackets: "(<d>)" -> "". The brackets go
    #    with it, because a bracket pair with nothing inside is noise.
    text = re.sub(r"[\(\[]\s*" + SENTINEL + r"\s*[\)\]]", "", text)

    # 2. A preposition left dangling by a removed date: "(night of
    #    <d>)" -> "(night)".
    #
    #    The preposition is dropped only when the removed date was
    #    INSIDE a bracket pair, which is what leaves it dangling. In
    #    running text ("from <d>, the model") the preposition still
    #    has an object, so it stays.
    #    The words between the bracket and the preposition are kept:
    #    "(night of <d>)" -> "(night)".
    text = re.sub(
        r"([\(\[])([^()\[\]]*?)\b(?:of|on|at|from|since|until|to|by|for|before|after|during)\s+"
        + SENTINEL + r"\s*([\)\]])",
        r"\1\2\3", text)

    # 3. The spaces a removal left behind. Only the run that touches the
    #    sentinel is touched; spaces elsewhere in the file are the
    #    author's. A sentinel that sat between two words keeps exactly
    #    one space, so "on <d> the thing" becomes "on the thing" and not
    #    "onthe thing".
    text = re.sub(r"[ \t]{2,}" + SENTINEL, " " + SENTINEL, text)
    text = re.sub(SENTINEL + r"[ \t]{2,}", SENTINEL + " ", text)
    text = re.sub(r"[ \t]+" + SENTINEL + r"[ \t]+", " " + SENTINEL, text)
    # A sentinel at the start of a line keeps the indentation intact:
    # only spaces on the same line are touched.
    text = re.sub(r"(?m)^[ \t]*" + SENTINEL, SENTINEL, text)
    # A bracket-only removal leaves a space where the bracket was, at
    # the start of the line: "(<d>) The model" -> " The model". The
    # leading space goes with it.
    text = re.sub(r"(?m)^ \S", lambda m: m.group(0)[1:], text)

    # 4. The sentinel itself.
    text = text.replace(SENTINEL, "")
    return text


def _brace_spans(text: str) -> list[tuple[int, int]]:
    """Return the spans of balanced ``{...}`` groups in the text.

    A time-like token inside a brace group is a data label, not
    chronology: ``{0:0, 1:3, 2:54, 3:1367}`` is a histogram, and
    ``{3:16,4:54,5:35,6:7}`` is a column histogram. The scrubber must
    not touch those, so the clock rule skips any match that falls
    inside one of these spans.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    return spans


def _tidy_around_holes(text: str) -> str:
    """Tidy punctuation spacing ONLY around a removed date.

    Each rule is anchored on the sentinel, so the rule can only ever
    touch the characters immediately next to a date that was actually
    removed. A rule that is allowed to run globally is a bug: the
    global form of the punctuation rule rewrites `from .board import`
    into `from.board import` and `winner != draw` into `winner!= draw`,
    which in Python is a different program.

    The rules are applied inside a placeholder window so that the
    sentinel itself is not disturbed by the whitespace rules that
    follow it.
    """
    # Each window is the sentinel plus a little context on either side.
    # The context is protected by the same sentinel-based trick: the
    # window text is parked under a private marker, tidied, and put
    # back.
    def _tidy_window(m: re.Match) -> str:
        window = m.group(0)
        # Park the window so the global rules below cannot reach it.
        parked = "\x00WINDOW\x00"
        window = window.replace(SENTINEL, parked)
        window = re.sub(r"[ \t]+([,.;:!?])", r"\1", window)
        window = re.sub(r"\(\s+", "(", window)
        window = re.sub(r"[ \t]+\)", ")", window)
        window = re.sub(r"[ \t]+\]", "]", window)
        window = re.sub(r"[ \t]+—", " —", window)
        window = re.sub(r"—[ \t]+", "— ", window)
        return window.replace(parked, SENTINEL)

    # A window is the sentinel plus up to 24 characters on each side.
    return re.sub(
        r".{0,24}" + re.escape(SENTINEL) + r".{0,24}",
        _tidy_window, text, flags=re.DOTALL)


def strip_dates(text: str, tidy: bool = True) -> tuple[str, int]:
    """Return (cleaned_text, number_of_dates_removed).

    Only the date/time tokens themselves are removed. Whitespace and
    indentation are never collapsed: doing so would break Python blocks
    and Markdown hard line breaks. The only whitespace changes are the
    ones directly around a removed token, which the marker pass handles.
    """
    count = 0

    for pattern, repl in PHRASES:
        text, n = re.subn(pattern, repl, text)
        count += n

    # Ranges first: they contain two dates.
    text, n = re.subn(RE_ISO_RANGE, SENTINEL, text)
    count += n

    # Every generic removal becomes a marker, so the tidy pass knows
    # exactly which characters it is allowed to touch.
    for pattern in (RE_ISO_DT, RE_MONTH_DMY, RE_DAY_MONTH_Y,
                    RE_MONTH_DAY, RE_DAY_MONTH):
        text, n = re.subn(pattern, SENTINEL, text)
        count += n

    # The clock rule runs separately from the other patterns, because it
    # needs the brace guard: a time-like token inside `{...}` is a data
    # label (a histogram), not chronology.
    #
    # The brace spans MUST be computed here, after every other pattern
    # has run. A span computed earlier is stale: replacing a date with
    # the 7-character SENTINEL shifts every offset after it, so a
    # histogram that sat below the first date in the file was no longer
    # inside its span and its `3:16` was deleted. That turned
    # `{3:16,4:54,5:35,6:7}` into `{,,,6:7}` -- the corruption this
    # function is now guarding against.
    brace_spans = _brace_spans(text)

    def _in_brace(pos: int) -> bool:
        return any(lo <= pos < hi for lo, hi in brace_spans)

    clock_removed = 0

    def _clock_sub(match: re.Match) -> str:
        nonlocal clock_removed
        if _in_brace(match.start()):
            return match.group(0)
        clock_removed += 1
        return SENTINEL

    text = RE_CLOCK.sub(_clock_sub, text)
    count += clock_removed

    # The numeric triple goes through the shared validity test, so the
    # scrubber removes exactly what the checker would report.
    #
    # The count is taken from the callback, not from re.subn: subn
    # counts every substitution it performed, including the ones where
    # the callback returned the match unchanged. Counting those would
    # report removals that never happened.
    removed = 0

    def _count_numeric_date(match: re.Match) -> str:
        nonlocal removed
        if DS.numeric_triple_is_date(match):
            removed += 1
            return SENTINEL
        return match.group(0)

    text = re.sub(RE_NUMERIC_DATE, _count_numeric_date, text)
    count += removed

    if tidy:
        # Prose-only tidying, applied ONLY to the neighbourhood of a
        # removed date.
        #
        # These rules must never run globally. A global
        # `[ \t]+([,.;:!?])` -> `\1` rewrites `from .board import` into
        # `from.board import` and `winner != draw` into `winner!= draw`:
        # in Python that is a different program. The same is true of the
        # bracket rules on `Board.is_win()`.
        #
        # The marker approach is what makes them safe: each rule is
        # anchored on the sentinel, so it can only ever touch text that
        # is adjacent to a date that was actually removed.
        text = _tidy_around_holes(text)

    text = _resolve_holes(text)

    return text, count


def process_file(path: Path) -> int:
    """Strip dates from one file. Returns the number of tokens removed.

    For a .py file the candidate is byte-compiled first; if it no longer
    compiles, the original is left untouched and an error is raised.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return 0
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return 0

    # Source code is NOT tidied. The prose rules (removing empty
    # parentheses, tightening spaces before punctuation) are correct for
    # Markdown and wrong for Python: "Board()" would become "Board",
    # which is a different object entirely. For code the only edits are
    # the phrase substitutions and the bare date/token removal.
    is_code = path.suffix.lower() in {".py", ".js", ".sh", ".yml", ".yaml"}

    cleaned, count = strip_dates(text, tidy=not is_code)
    if not count or cleaned == text:
        return 0

    is_python = path.suffix.lower() == ".py"
    backup: Path | None = None

    if is_python:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / path.name
            probe.write_text(cleaned, encoding="utf-8")
            try:
                py_compile.compile(str(probe), doraise=True,
                                   cfile=str(Path(td) / "probe.pyc"))
            except py_compile.PyCompileError as exc:
                raise RuntimeError(
                    f"date-stripping broke the syntax of {path}:\n{exc}"
                ) from exc
        # The AST must be structurally identical to the original. A
        # syntax-only check would pass "Board()" -> "Board", which
        # compiles but is different code.
        if _ast_signature(text) != _ast_signature(cleaned):
            raise RuntimeError(
                f"date-stripping changed the code structure of {path}"
            )
        backup = path.with_suffix(path.suffix + ".dates-backup")
        shutil.copy2(path, backup)

    try:
        path.write_bytes(cleaned.encode("utf-8"))
    except Exception:
        if backup is not None:
            shutil.copy2(backup, path)
        raise

    if backup is not None:
        backup.unlink(missing_ok=True)

    return count


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    targets: list[Path] = []
    for arg in argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in TEXT_EXT:
                    targets.append(f)
        elif p.is_file():
            targets.append(p)

    total = 0
    touched = 0
    for f in sorted(set(targets)):
        try:
            n = process_file(f)
        except RuntimeError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        if n:
            total += n
            touched += 1
            print(f"  {n:3d} date(s)  {f}")

    print(f"stripped {total} date/time token(s) from {touched} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
