"""Details / follow-up column: who wrote what, even when handwriting spills over.

Handwriting in the details column is grouped into text blobs (letters -> words ->
lines). Each blob is attributed to the row that holds most of its ink, so writing that
"overlaps to the next line" or "exceeds the space" counts for the row it was written
for and does not make the neighbouring row look filled. Overflow is reported as a note.

Also provides the extent of writing in free-text fields (Section A), so that long
reference numbers that run past the box are cropped in full for reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .context import PageContext
from .rules_common import classify_presence
from .settings import Settings
from .template import Box, Item


@dataclass
class DetailsReading:
    state: str                         # filled | blank | uncertain | occluded
    confidence: float
    ink_ratio: float
    visible: float
    blobs: list[tuple[int, int, int, int]] = field(default_factory=list)   # template px
    notes: list[str] = field(default_factory=list)
    continuation_of: Optional[str] = None

    @property
    def extent(self) -> Optional[tuple[int, int, int, int]]:
        if not self.blobs:
            return None
        return (min(b[0] for b in self.blobs), min(b[1] for b in self.blobs),
                max(b[2] for b in self.blobs), max(b[3] for b in self.blobs))


def _edge_sliver(gx: np.ndarray, left: int, right: int, max_w: int = 10, near: int = 14) -> bool:
    """Narrow vertical piece hugging a cell's left/right border: border residue."""
    w = gx.max() - gx.min() + 1
    cx = gx.mean()
    return w <= max_w and (abs(cx - left) < near or abs(cx - right) < near)


def _blobs(mask: np.ndarray, jx: int, jy: int, min_px: int):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, jx), max(1, jy)))
    grp = cv2.dilate(mask.astype(np.uint8), k)
    n, lab = cv2.connectedComponents(grp, connectivity=8)
    out = []
    for j in range(1, n):
        P = (lab == j) & mask
        if P.sum() >= min_px:
            out.append(P)
    return out


def read_section_details(ctx: PageContext, items: list[Item], base: Settings
                         ) -> dict[str, DetailsReading]:
    items = [i for i in items if i.type == "choice" and i.details is not None]
    if not items:
        return {}
    s = base
    H, W = ctx.shape
    boxes = [ctx.px(i.details.box) for i in items]
    row_h = int(np.mean([b[3] - b[1] for b in boxes]))
    cx0 = min(b[0] for b in boxes)
    cx1 = max(b[2] for b in boxes)
    rx0, rx1 = max(0, cx0 - 4), min(W, cx1 + s.details_overflow_px)
    ry0 = max(0, min(b[1] for b in boxes) - 4)
    ry1 = min(H, max(b[3] for b in boxes) + int(0.6 * row_h))
    sub = ctx.maps.added[ry0:ry1, rx0:rx1]

    assigned: list[list[np.ndarray]] = [[] for _ in items]
    outside_blobs: list = []
    notes: list[list[str]] = [[] for _ in items]
    ext: list[list[tuple]] = [[] for _ in items]
    for P in _blobs(sub, s.text_join_x, s.text_join_y, 2 * s.min_component_px):
        ys, xs = np.nonzero(P)
        if xs.mean() + rx0 < cx0 + 6:
            continue                       # spill from a circle in the response column
        if _edge_sliver(xs + rx0, cx0, cx1):
            continue                       # remnant of a cell border, not writing
        gy = ys + ry0
        counts = np.array([((gy >= b[1]) & (gy < b[3])).sum() for b in boxes])
        if counts.max() == 0:
            # below the last row: only writing that starts right under it is overflow
            if gy.min() >= boxes[-1][3] - 2 and gy.min() <= boxes[-1][3] + 0.3 * row_h:
                ri = len(items) - 1
            else:
                continue
        else:
            ri = int(np.argmax(counts))
        bx = (int(xs.min() + rx0), int(ys.min() + ry0), int(xs.max() + rx0 + 1), int(ys.max() + ry0 + 1))
        if bx[0] > cx1 + 2:
            outside_blobs.append((ri, P, bx))    # beyond the table: only as a continuation
            continue
        assigned[ri].append(P)
        ext[ri].append(bx)
        b = boxes[ri]
        outside = 1 - counts[ri] / len(ys)
        if len(ys) < s.overflow_note_min_px:
            continue                       # too small to comment on
        if outside > 0.2:
            others = [items[k].id for k in range(len(items)) if k != ri and counts[k] > 0.1 * len(ys)]
            where = f"into row {', '.join(others)}" if others else "below the table row"
            notes[ri].append(f"handwriting crosses the row line ({outside:.0%} of it {where})")
        elif bx[3] >= b[3] - 1 or bx[1] <= b[1] + 1:
            # ink on the printed line itself is removed with the form, so writing that runs
            # over the line shows up as writing that stops exactly at it
            notes[ri].append("handwriting runs into the row line (may overlap the next row)")
        if bx[2] > b[2] + 8:
            notes[ri].append("handwriting runs past the right edge of the box")

    # writing entirely to the right of the table counts only when it continues text that
    # starts inside the row's cell (otherwise it is a stray mark beside the form)
    for ri, P, bx in outside_blobs:
        if any(e[2] >= bx[0] - 40 and e[1] < bx[3] and e[3] > bx[1] for e in ext[ri]):
            assigned[ri].append(P)
            ext[ri].append(bx)
            notes[ri].append("handwriting runs past the right edge of the box")

    out: dict[str, DetailsReading] = {}
    for ri, it in enumerate(items):
        b = boxes[ri]
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
        ink = sum(int(P.sum()) for P in assigned[ri])
        ratio = ink / area
        vis = ctx.visible(*b)
        ss = s.merged(it.overrides)
        state, conf = classify_presence(ratio, vis, ss)
        out[it.id] = DetailsReading(state, conf, round(ratio, 4), round(vis, 3), ext[ri],
                                    sorted(set(notes[ri])))

    # writing that sits fully inside row r but is the continuation of row r-1's text
    for ri in range(1, len(items)):
        cur, prev = out[items[ri].id], out[items[ri - 1].id]
        if cur.state != "filled" or prev.state != "filled" or not cur.blobs or not prev.blobs:
            continue
        top_cur = min(bb[1] for bb in cur.blobs)
        bot_prev = max(bb[3] for bb in prev.blobs)
        if top_cur - bot_prev < 0.5 * row_h and top_cur < boxes[ri][1] + 0.6 * row_h:
            cur.continuation_of = items[ri - 1].id
    return out


def field_extent(ctx: PageContext, box: Box, s: Settings
                 ) -> tuple[Optional[tuple[int, int, int, int]], list[str]]:
    """Extent (template px) of the writing that belongs to a free-text box, including
    any part that runs outside it. Returns (extent or None, notes)."""
    H, W = ctx.shape
    x0, y0, x1, y1 = ctx.px(box)
    bw, bh = x1 - x0, y1 - y0
    ex0, ex1 = max(0, x0 - int(0.3 * bw)), min(W, x1 + int(0.6 * bw))
    ey0, ey1 = max(0, y0 - int(0.35 * bh)), min(H, y1 + int(0.35 * bh))
    sub = ctx.maps.added[ey0:ey1, ex0:ex1]
    keep = []
    for P in _blobs(sub, s.text_join_x, s.text_join_y, 2 * s.min_component_px):
        ys, xs = np.nonzero(P)
        gx, gy = xs + ex0, ys + ey0
        if _edge_sliver(gx, x0, x1):
            continue
        inside = ((gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)).mean()
        if inside >= 0.25:
            keep.append((int(gx.min()), int(gy.min()), int(gx.max() + 1), int(gy.max() + 1)))
    if not keep:
        return None, []
    e = (min(k[0] for k in keep), min(k[1] for k in keep),
         max(k[2] for k in keep), max(k[3] for k in keep))
    notes = []
    if e[2] > x1 + 8 or e[0] < x0 - 8:
        notes.append("writing runs past the edge of the box")
    if e[3] > y1 + 6 or e[1] < y0 - 6:
        notes.append("writing crosses the box's top/bottom line")
    return e, notes
