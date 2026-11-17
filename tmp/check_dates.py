#!/usr/bin/env python
"""Report every calendar date that still remains in a tree of text files.

WHY THIS EXISTS
---------------
The staging script used to run its "no calendar dates" gate with a PCRE
lookbehind inside `grep -E`. GNU grep's -E dialect has no lookbehind, so
the pattern matched nothing at all, the gate reported "clean" every time,
and the published repository kept every date in the research log.

This module does the same job with the same intent, but with patterns
that are actually evaluated, and it reports the file, line and matched
token so a failure is actionable.

ONE SHARED DEFINITION OF A DATE
-------------------------------
The patterns and the validity test live in `date_spec.py`, which the
scrubber imports too. The two tools MUST agree: if the checker flagged
tokens the scrubber was never going to remove, the staging gate would
fail on things that are not dates at all -- channel-share triples such
as `86/80/91`, board-fill fractions such as `21.0/42`, and section
numbers such as `9.7.1`. A gate that cries wolf is a gate that gets
switched off.

A bare four-digit year is also reported, but only when it sits in a
date-ish context (a month name on the same line). On its own, `2026` is
as likely to be a seed value or a step count as a year.

Usage:
    python tmp/check_dates.py <dir-or-file> [...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import date_spec as DS  # noqa: E402

TEXT_EXT = {
    ".md", ".py", ".txt", ".sh", ".yml", ".yaml", ".json", ".html",
    ".css", ".js", ".cfg", ".ini", ".toml", ".ipynb",
}

# A bare 4-digit number is only a date when it sits in a date-ish
# context. These are the measured values that appear all over the logs.
_NOT_YEAR_CONTEXT = re.compile(
    r"range\(\s*\d{4}\s*\)"
    r"|Random\(\s*\d{4}\s*\)"
    r"|default_rng\(\s*\d{4}\s*\)"
    r"|seed\D{0,3}\d{4}"
    r"|\bstep[\s-]\d{4}\b"
    r"|\b\d{4}\s*(?:steps?|epoch|ms|s\b)"
    r"|\[\s*\d{4}\s*:\s*\d{4}\s*\]"
    r"\|\s*\d{4}\s*\|"
    r"|\b\d+\.\d+\b"
    r"|\bd\w*\s*=\s*\d{4}\b"
    r"|\bn\s*=\s*\d{4}\b"
    r"|\b\d{4}/\d{3,4}\b"
    r"|\b\d{4}[,;)\s]"
    r"|https?://"
)

RE_BARE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _is_ip_or_version(line: str, start: int, end: int) -> bool:
    """True if the token is part of an IP address or a version triple."""
    window = line[max(0, start - 12):end + 12]
    if re.search(r"\d+\.\d+\.\d+\.\d+", window):
        return True
    if re.search(r"\d+\.\d+\.\d+", line[start:end]):
        return True
    return False


def find_dates(text: str) -> list[tuple[int, str, str]]:
    """Return (line_number, label, matched_text) for every date found."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in DS.RE_ISO_DT.finditer(line):
            token = m.group(0)
            if _is_ip_or_version(line, m.start(), m.end()):
                continue
            hits.append((lineno, "ISO date", token))

        for m in DS.find_numeric_dates(line):
            token = m.group(0)
            if _is_ip_or_version(line, m.start(), m.end()):
                continue
            hits.append((lineno, "numeric date", token))

        for regex, label in DS.DATE_PATTERNS[1:]:
            for m in regex.finditer(line):
                token = m.group(0)
                if _is_ip_or_version(line, m.start(), m.end()):
                    continue
                hits.append((lineno, label, token))

        for m in RE_BARE_YEAR.finditer(line):
            token = m.group(0)
            if _is_ip_or_version(line, m.start(), m.end()):
                continue
            if _NOT_YEAR_CONTEXT.search(line):
                continue
            # A bare year only counts as a date when the line also
            # carries a month name or a daypart.
            if not re.search(DS.MONTH, line, re.IGNORECASE):
                continue
            hits.append((lineno, "bare year", token))
    return hits


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
    for f in sorted(set(targets)):
        try:
            text = f.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits = find_dates(text)
        if not hits:
            continue
        lines = text.splitlines()
        for lineno, label, token in hits:
            total += 1
            context = lines[lineno - 1].strip()[:120] if lineno <= len(lines) else ""
            print(f"{f}:{lineno}: [{label}] {token!r}\n    {context}")

    print(f"{total} date-like token(s) found")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
