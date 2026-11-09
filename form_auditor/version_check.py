"""Is the filled form the version the template was made for?

The version is printed on each page ("Document ID: <id> v31.1"). The handwriting reader
returns the page's printed text too, so the version is read there - in the box the template
builder found on the blank (`version_box`), else in the top band of the page - and compared
with the versions the template accepts (`versions` in the template YAML, from `version` in
the form config). A form of another version may look the same and still differ in its
rules or rows, so an unknown version is reported, never silently accepted.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .rules_common import Finding
from .template import norm_version

VERSION_RE = re.compile(r"(?<![A-Za-z0-9])[vV]\s?(\d+(?:\.\d+)*)(?![\d.]*\d)")
TOP_BAND = (0.0, 0.0, 1.0, 0.12)


def read_version(page_text, box_px) -> tuple[Optional[str], str]:
    """(version as printed, e.g. '32.0', or None; the text read in the box)."""
    x0, y0, x1, y1 = box_px
    words = []
    for w in getattr(page_text, "words", []) or []:
        c = np.asarray(w.polygon, float).reshape(-1, 2).mean(axis=0)
        if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
            words.append((w.line, c[0], w.text))
    words.sort()
    text = " ".join(t for _, _, t in words)
    m = VERSION_RE.search(text)
    return (m.group(1) if m else None), text


def check_version(runtime, page_text, shape, known: dict[str, str]) -> Optional[Finding]:
    """A finding for the page's printed version, or None when the template names no version.
    known: every loaded template's versions -> its name."""
    tpl, page = runtime.template, runtime.page
    if not tpl.versions:
        return None
    nbox = page.version_box or list(TOP_BAND)
    h, w = shape[:2]
    box_px = (nbox[0] * w, nbox[1] * h, nbox[2] * w, nbox[3] * h)
    accepted = ", ".join(f"v{v}" for v in tpl.versions)
    f = Finding("VERSION", "Form version", "version", "Printed form version", "form_version",
                "PASS", "", accepted, "", 1.0, boxes=[list(nbox)])
    if page_text is None or getattr(page_text, "error", ""):
        f.status, f.observed = "SKIP", "not read"
        f.reason = "The form version is read by the handwriting reader, which was not used"
        return f
    printed, text = read_version(page_text, box_px)
    f.value_read = text
    if printed is None:
        f.status, f.observed = "REVIEW", "unreadable"
        f.reason = (f"Could not read the form version at the top of the page (read: "
                    f"'{text[:80]}') - check it is {accepted}")
        return f
    found = norm_version(printed)
    f.observed, f.value_normalized = f"v{printed}", f"v{found}"
    if found in tpl.versions:
        return f
    other = known.get(found)
    f.status = "REVIEW"
    if other:
        f.reason = (f"Form version v{printed} is printed on the page, but the page matched "
                    f"{tpl.display_name} - check the page belongs to {other}")
    else:
        f.reason = (f"Form version v{printed} is not recognised - this tool knows "
                    f"{', '.join('v' + v for v in sorted(known)) or accepted}. The checks below "
                    f"follow {tpl.display_name}; confirm them by hand, and add a template "
                    f"for this version")
    return f
