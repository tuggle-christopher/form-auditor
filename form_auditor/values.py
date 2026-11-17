"""Normalising and validating transcribed values.

normalize(fmt, text, allow_na, date_order) -> Value

Formats:
  text / id  - returned as written (ids upper-cased, spaces collapsed). Symbols such as
               # / - are kept.
  date       - many handwritten styles: 14 Oct 26, 14-Oct-26, 14-10-26, 14/10/2026,
               14.10.26, Oct 14 2026, 2026-10-14 ... -> ISO yyyy-mm-dd.
               Numeric dates are read day-first unless date_order="MDY". A month name the
               reader got slightly wrong (22 See 26) or digits read as letters (1O-1O-26)
               are still read, with a `doubt` saying what was guessed.
  time       - HH:MM (24 h). 9:05 is accepted unless strict two-digit hours are asked for.
  number     - e.g. 15.8, -18, 15.8°C, 15.8 C -> "15.8"
A written NA (NA, N/A, N.A., n/a) is recognised for every format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS.update({"sept": 9, "june": 6, "july": 7, "january": 1, "february": 2, "march": 3,
               "april": 4, "august": 8, "september": 9, "october": 10, "november": 11,
               "december": 12})
_NA = re.compile(r"^\s*n\s*[/.\\-]?\s*a\s*\.?\s*$", re.I)
# characters handwriting OCR commonly confuses inside numbers
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                            "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})


@dataclass
class Value:
    raw: str
    normalized: str
    valid: Optional[bool]          # None = nothing to validate (empty / unread)
    is_na: bool = False
    problem: str = ""
    doubt: str = ""                # read, but only by guessing at letters / digits: check it


def _year(y: str) -> Optional[int]:
    y = y.translate(_DIGIT_FIX)
    if not y.isdigit():
        return None
    v = int(y)
    if len(y) == 2:
        return 2000 + v
    if len(y) == 4 and 1990 <= v <= 2100:
        return v
    return None


def _num(t: str) -> Optional[int]:
    t = t.translate(_DIGIT_FIX)
    return int(t) if t.isdigit() else None


def _month(t: str) -> Optional[int]:
    t = t.lower().rstrip(".")
    if t in MONTHS:
        return MONTHS[t]
    if len(t) >= 3 and t[:3] in MONTHS:
        return MONTHS[t[:3]]
    return None


_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                "sept", "june", "july", "january", "february", "march", "april", "august",
                "september", "october", "november", "december"]
_LETTER_FIX = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "|": "l"})


def _edits(a: str, b: str) -> int:
    """Edit distance: insert, delete or change a letter, or swap two neighbours ('Spe')."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def guess_month(tok: str) -> Optional[int]:
    """A month name the handwriting reader got slightly wrong: 'See' -> Sep, '0ct' -> Oct,
    'Nuv' -> Nov, 'Septemper' -> September. One letter off for a short name, two for a long
    one, and only when exactly one month is that close."""
    t = tok.lower().rstrip(".").translate(_LETTER_FIX)
    if len(t) < 3 or not t.isalpha():
        return None
    if t in MONTHS:
        return MONTHS[t]
    best, hits = None, set()
    for name in _MONTH_NAMES:
        allowed = 1 if len(name) <= 4 else 2
        if abs(len(name) - len(t)) > allowed:
            continue
        d = _edits(t, name)
        if d <= allowed:
            if best is None or d < best[0]:
                best, hits = (d, name), {MONTHS[name]}
            elif d == best[0]:
                hits.add(MONTHS[name])
    return hits.pop() if best and len(hits) == 1 else None


def read_date(text: str, order: str = "DMY") -> tuple[Optional[date], str]:
    """-> (date, doubt). doubt is empty when the date was read as written; otherwise it says
    what had to be guessed (a misread month name, letters taken for digits)."""
    d = parse_date(text, order)
    if d is not None:
        # a month taken from its first letters only ('Septemper', 'Octo'): not a real name
        for w in re.findall(r"[A-Za-z]{3,}", text):
            if _month(w) and w.lower().rstrip(".") not in MONTHS:
                return d, f"the month '{w}' is not clear - taken as {date(2000, _month(w), 1):%b}"
        fixed = text.translate(_DIGIT_FIX)
        # digits written as letters (O for 0, S for 5, ...) were needed to read it
        letters_in_numbers = any(re.fullmatch(r"[0-9OoDIl|SsBZz]+", x) and re.search(r"[OoDIl|SsBZz]", x)
                                 and re.search(r"\d", x) for x in re.split(r"[\s/\-.,_']+", text))
        if letters_in_numbers and fixed != text:
            return d, "some digits look like letters"
        return d, ""
    if not re.search(r"[a-z]{3}", text.lower().translate(_DIGIT_FIX)):
        d = parse_date(text.translate(_DIGIT_FIX), order)          # 1O-1O-26 -> 10-10-26
        if d is not None:
            return d, "some digits look like letters"
    # a month name the reader got slightly wrong: 22 See 26
    toks = [x for x in re.split(r"([\s/\-.,_']+)", text)]
    for i, x in enumerate(toks):
        if not re.search(r"[A-Za-z]", x):
            continue
        core = x if len(re.findall(r"[A-Za-z]", x)) >= 2 and guess_month(x) else re.sub(r"^\d+|\d+$", "", x)
        m = guess_month(core) if core else None
        if m and core.lower().rstrip(".") not in MONTHS:
            name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
            fixed = "".join(toks[:i] + [x.replace(core, f" {name} ")] + toks[i + 1:])
            d = parse_date(fixed, order)
            if d is not None:
                return d, f"the month '{core}' is not clear - taken as {name}"
    return None, ""


def parse_date(text: str, order: str = "DMY") -> Optional[date]:
    toks = []
    for x in re.split(r"[\s/\-.,_']+", text.strip()):
        if not x:
            continue
        if re.fullmatch(r"[0-9OoDIl|SsBZz]+", x) and re.search(r"\d", x):
            toks.append(x.translate(_DIGIT_FIX))          # "O1" -> "01"
        else:                                             # "14Oct26" -> 14 Oct 26
            toks += [y for y in re.sub(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)", " ", x).split() if y]
    if len(toks) != 3:
        return None
    candidates = []
    mon_idx = [i for i, x in enumerate(toks) if _month(x) and not x.translate(_DIGIT_FIX).isdigit()]
    if len(mon_idx) == 1:
        mi = mon_idx[0]
        m = _month(toks[mi])
        rest = [toks[i] for i in range(3) if i != mi]
        if mi == 1:                                   # 14 Oct 26
            d, y = _num(rest[0]), _year(rest[1])
            candidates.append((y, m, d))
        elif mi == 0:                                 # Oct 14 26
            d, y = _num(rest[0]), _year(rest[1])
            candidates.append((y, m, d))
        else:                                         # 2026 14 Oct (rare)
            y, d = _year(rest[0]), _num(rest[1])
            candidates.append((y, m, d))
    elif not mon_idx:
        a, b, c = toks
        if len(a.translate(_DIGIT_FIX)) == 4:         # 2026-10-14
            candidates.append((_year(a), _num(b), _num(c)))
        elif order.upper() == "MDY":
            candidates.append((_year(c), _num(a), _num(b)))
        else:
            candidates.append((_year(c), _num(b), _num(a)))
    for y, m, d in candidates:
        if None in (y, m, d):
            continue
        try:
            return date(y, m, d)
        except ValueError:
            continue
    return None


def normalize(fmt: str, text: Optional[str], allow_na: bool = False, date_order: str = "DMY",
              strict_time: bool = False) -> Value:
    if text is None:
        return Value("", "", None)
    raw = " ".join(str(text).split())
    if not raw:
        return Value("", "", None)
    if _NA.match(raw):
        if allow_na:
            return Value(raw, "NA", True, True)
        return Value(raw, "NA", False, True, "'NA' written, but NA is not an accepted entry for this field")

    if fmt == "date":
        d, doubt = read_date(raw, date_order)
        if d is None:
            return Value(raw, "", False, problem="not a recognisable date")
        return Value(raw, d.isoformat(), True, doubt=doubt)
    if fmt == "time":
        t = raw.replace(" ", "")
        m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", t)
        if not m or (strict_time and len(m.group(1)) != 2):
            return Value(raw, "", False, problem="not in HH:MM format")
        return Value(raw, f"{int(m.group(1)):02d}:{m.group(2)}", True)
    if fmt == "number":
        t = re.sub(r"\s*(°\s*[cCfF]?|deg(rees)?\s*[cC]?(elsius)?|[cC])\s*$", "", raw).replace(",", ".")
        t = t.replace(" ", "")
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", t):
            return Value(raw, t.lstrip("+"), True)
        t2 = t.translate(_DIGIT_FIX)
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", t2):
            return Value(raw, t2.lstrip("+"), True, problem="read with character corrections")
        return Value(raw, "", False, problem="not a number")
    if fmt == "id":
        return Value(raw, re.sub(r"\s+", "", raw).upper(), True)
    return Value(raw, raw, True)
