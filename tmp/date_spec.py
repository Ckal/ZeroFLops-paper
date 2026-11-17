#!/usr/bin/env python
"""ONE definition of "what is a calendar date", shared by the scrubber and
the checker.

WHY A SHARED MODULE
-------------------
The scrubber (`strip_dates.py`) removes dates. The checker
(`check_dates.py`) reports dates that survived. When those two used
different pattern sets they disagreed: the checker flagged tokens the
scrubber was never going to remove, so the staging gate failed on
things that were not dates at all -- channel-share triples such as
`86/80/91`, board-fill fractions such as `21.0/42`, step lists such as
`step0/30/60`, and section numbers such as `9.7.1`.

A gate that cries wolf is a gate that gets switched off. The two tools
must therefore agree by construction: they import the SAME patterns
from this module and apply the SAME validity test.

WHAT COUNTS AS A DATE
---------------------
  * ISO            2026-10-29   and   2026-10-29 02:34
  * European       29.10.26     and   29.10.2026
  * US / slashed   10/29/2026   and   29/10/2026
  * Year-first     2026/10/29
  * Spelled        October 29, 2026  /  29 October 2026
  * Month and day  26 October   /   October 26    (no year)

WHAT IS NOT A DATE
------------------
  * IPv4 addresses        127.0.0.60
  * Version triples       2.2024, 1.0.60, 26.6.1, 9.7.1
  * Model scores          511-88-1, 543-57-0, 223-376-1
  * Channel-share triples 86/80/91      (first component > 31)
  * Board-fill fractions  21.0/42       (mixed separators)
  * Step lists            step0/30/60, 0/500/1000
  * Column lists          3/4/5, 18/1/6
  * Seed values           random.Random(2026)
  * Measured numbers      0.1970, 1959/2000

THE VALIDITY TEST
-----------------
A three-part numeric token is a date only when its components CAN be
one:

    year-first   2026-10-29   -> 4-digit year 1900..2099, month <= 12,
                                 day <= 31
    year-last    29.10.2026   -> 4-digit year 1900..2099, and one of the
                                 first two components is a month (<= 12)
                                 while the other is a day (<= 31)
    European     29.10.26     -> DOTTED only, 2-digit year, day <= 31,
                                 month <= 12

That single rule rejects every false positive above. `86/80/91` fails
on the day, `0/30/60` fails on the day, `21.0/42` has mixed
separators, `511-88-1` fails on the day, `26.6.1` has a one-digit
year, and `9.7.1` has a one-digit year.

TWO DIGIT YEARS
---------------
A two-digit year is only accepted in the DOTTED European form
(`29.10.26`). The slashed and dashed forms are left alone, because
`21/10/18` in the research log is a channel-share triple and not the
21st of October 2018.
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------ months
_MONTH_FULL = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)"
)
_MONTH_ABBR = r"(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

# A month token never ends in a letter. Without this, "Dec" + "ided"
# matched "decided", and the scrubber deleted real source lines.
_MONTH_BODY = rf"(?:{_MONTH_FULL}|{_MONTH_ABBR})"
MONTH = rf"{_MONTH_BODY}(?![A-Za-z])"
MONTH_OPT_DOT = rf"{_MONTH_BODY}\.?(?![A-Za-z])"

# Horizontal whitespace only. A date must never span a line break: if it
# could, a match could delete the code in between.
GAP = r"[ \t]+"

# ------------------------------------------------------------------ shapes
# ISO date, optionally with a time.
RE_ISO_DT = re.compile(
    r"(?<![\d-])\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?![\d])")

# ISO date range: two times joined by an arrow.
RE_ISO_RANGE = re.compile(
    r"\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:→|->)\s*\d{1,2}:\d{2}(?::\d{2})?")

# Spelled dates.
RE_MONTH_DMY = re.compile(
    rf"\b{MONTH_OPT_DOT}{GAP}\d{{1,2}}(?:st|nd|rd|th)?(?:,)?{GAP}\d{{4}}\b",
    re.IGNORECASE)
RE_DAY_MONTH_Y = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?{GAP}{MONTH_OPT_DOT},?{GAP}\d{{4}}\b",
    re.IGNORECASE)

# Month and day with no year: "26 October", "October 26".
RE_MONTH_DAY = re.compile(
    rf"\b{MONTH_OPT_DOT}{GAP}\d{{1,2}}(?:st|nd|rd|th)?\b", re.IGNORECASE)
RE_DAY_MONTH = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?{GAP}{MONTH_OPT_DOT}\b", re.IGNORECASE)

# A clock time on its own is not reported by the checker: `a[1:2]` and
# `0:30` are code, not chronology, and a bare `21:25` carries no date
# information. The scrubber still removes clock times, because the
# working chronology is exactly what the public tree must not carry --
# so the scrubber is deliberately STRICTER than the checker here.
#
# The left guard is the important part. A bare `\d{1,2}:\d{2}` also
# matches the tail of a `host:port` URL (`127.0.0.1:7860` -> `127.0.0.60`)
# and of a labelled pair (`c2:64` -> `c`, `{3:16,4:54}` -> `{,`). A clock
# time must therefore be a standalone token: it may not be preceded by a
# digit, a dot, or a letter, and it must not be followed by a word
# character.
RE_CLOCK = re.compile(r"(?<![\w.])\d{1,2}:\d{2}(?::\d{2})?(?![\w:])")

# Three-part numeric shapes. The separator is back-referenced, so the
# three components must all use the SAME separator.
NUMERIC_TRIPLE = re.compile(
    r"(?<![\d./-])(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})(?![\d./-])")


def _plausible_year(year: int) -> bool:
    return 1900 <= year <= 2099


def numeric_triple_is_date(match: re.Match) -> bool:
    """True when a three-part numeric token can be a calendar date.

    The test is component-based, not shape-based. See the module
    docstring for the three accepted layouts.
    """
    first, sep, second, third = match.group(1), match.group(2), \
        match.group(3), match.group(4)
    a, b, c = int(first), int(second), int(third)

    # A zero month or a zero day is never a date. This is what keeps
    # the version triple `1.0.60` out of the date set.
    if b == 0 or c == 0:
        return False

    # Year first: 2026-10-29, 2026/10/29.
    if len(first) == 4 and _plausible_year(a):
        return b <= 12 and c <= 31

    # A four-digit year last: 29.10.2026, 10/29/2026, 29/10/2026.
    if len(third) == 4:
        if not _plausible_year(c):
            return False
        return (a <= 12 and b <= 31) or (b <= 12 and a <= 31)

    # A two-digit year last: the dotted European form only.
    if len(third) == 2:
        if sep != ".":
            return False
        return a <= 31 and b <= 12

    # A one-digit year is a version number or a section number.
    return False


def find_numeric_dates(text: str) -> list[re.Match]:
    """Every three-part numeric token in `text` that is a real date."""
    return [m for m in NUMERIC_TRIPLE.finditer(text)
            if numeric_triple_is_date(m)]


# The order matters: the first pattern that matches a position wins, so
# a full ISO date is not also reported as a bare European date.
DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (RE_ISO_DT, "ISO date"),
    (RE_MONTH_DMY, "spelled date"),
    (RE_DAY_MONTH_Y, "spelled date"),
    (RE_MONTH_DAY, "month and day"),
    (RE_DAY_MONTH, "month and day"),
]
