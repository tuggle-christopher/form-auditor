"""Page-level handwriting (Azure Document Intelligence "Layout") fused with the ink check.

The aligned page is sent to Document Intelligence; every word it finds comes back with
a position, so it can be placed exactly on the blank form. This module

  * keeps only the pen writing (printed words of the form are recognised by the
    template's own print at that spot and dropped; words inside answer cells are left
    to the mark reader),
  * gives each piece of writing to the box it was written in - a word that starts in a
    box and runs out of it stays with that box, and so do the other words of its line,
  * keeps writing that is entirely outside every box but right next to one, so a blank
    box with writing beside it goes to REVIEW instead of FAIL,
  * combines the result with the ink check (`fuse_presence`): two independent opinions
    on whether a box is empty. Faint writing that the ink check misses but Document
    Intelligence reads confidently counts as filled; anything only one of them half-sees
    goes to REVIEW. Neither can turn clear pen ink into "blank".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .settings import Settings

PxBox = tuple[int, int, int, int]


@dataclass
class PageWord:
    """One word as returned by the reader, in the coordinates of the image sent."""
    text: str
    confidence: float
    polygon: np.ndarray                  # (N, 2)
    line: int = -1                       # index of the reader's text line
    handwritten: Optional[bool] = None


@dataclass
class PageMark:
    """A selection mark (checkbox) found by the reader."""
    state: str                           # selected | unselected
    confidence: float
    polygon: np.ndarray


@dataclass
class PageText:
    words: list[PageWord] = field(default_factory=list)
    marks: list[PageMark] = field(default_factory=list)
    engine: str = ""
    error: str = ""

    def scaled(self, k: float) -> "PageText":
        """Polygons multiplied by k (image sent -> template px)."""
        return PageText([PageWord(w.text, w.confidence, w.polygon * k, w.line, w.handwritten)
                         for w in self.words],
                        [PageMark(m.state, m.confidence, m.polygon * k) for m in self.marks],
                        self.engine, self.error)


@dataclass
class PenWord:
    text: str
    confidence: float
    box: PxBox                           # template px
    line: int
    handwritten: Optional[bool]
    pen_px: int                          # pen ink found under it by the ink check

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class FieldText:
    """The pen writing that belongs to one box."""
    words: list[PenWord] = field(default_factory=list)       # in the box (or running out of it)
    outside: list[PenWord] = field(default_factory=list)     # entirely beside the box
    overflow: bool = False

    @staticmethod
    def _join(words: list[PenWord], sep: str) -> str:
        lines: dict[int, list[PenWord]] = {}
        for w in words:
            lines.setdefault(w.line, []).append(w)
        ordered = sorted(lines.values(), key=lambda ws: min(w.box[1] for w in ws))
        return sep.join(" ".join(w.text for w in sorted(ws, key=lambda w: w.box[0])) for ws in ordered)

    def text(self, multiline: bool = False) -> str:
        return self._join(self.words, "\n" if multiline else " ").strip()

    @property
    def outside_text(self) -> str:
        return self._join(self.outside, " ").strip()

    @property
    def confidence(self) -> float:
        return round(float(np.mean([w.confidence for w in self.words])), 3) if self.words else 0.0

    @property
    def extent(self) -> Optional[PxBox]:
        return _union([w.box for w in self.words]) if self.words else None

    @property
    def outside_extent(self) -> Optional[PxBox]:
        return _union([w.box for w in self.outside]) if self.outside else None


def _union(boxes) -> PxBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _inter(a: PxBox, b: PxBox) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _gap(a: PxBox, b: PxBox) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return float(np.hypot(dx, dy))


def _inside(pt, b: PxBox, tol: int = 2) -> bool:
    return b[0] - tol <= pt[0] <= b[2] + tol and b[1] - tol <= pt[1] <= b[3] + tol


# --------------------------------------------------------------------------- #
def pen_words(page: PageText, ctx, s: Settings) -> list[PenWord]:
    """Words of `page` (template px) that are pen writing, not the printed form."""
    H, W = ctx.shape
    printed = ctx.tpl_core if ctx.tpl_core is not None else np.zeros((H, W), bool)
    if ctx.rules_zone is not None:
        printed = printed & ~ctx.rules_zone         # table rules are not printed words
    pen = ctx.maps.added
    if getattr(ctx, "faint", None) is not None:
        pen = pen | ctx.faint.added
    out = []
    for w in page.words:
        if not w.text.strip() or w.polygon is None or len(w.polygon) == 0:
            continue
        xs, ys = w.polygon[:, 0], w.polygon[:, 1]
        b = (int(max(0, np.floor(xs.min()))), int(max(0, np.floor(ys.min()))),
             int(min(W, np.ceil(xs.max()))), int(min(H, np.ceil(ys.max()))))
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        p_print = printed[b[1]:b[3], b[0]:b[2]].sum() / area
        n_pen = int(pen[b[1]:b[3], b[0]:b[2]].sum())
        if p_print > s.printed_word_density and n_pen / area < p_print \
                and not (w.handwritten and n_pen >= s.pen_word_min_px):
            continue                                   # a printed word of the form
                                                       # (writing over print is kept)
        if not (w.handwritten or n_pen >= s.pen_word_min_px
                or w.confidence >= s.azure_accept_conf):
            continue                                   # unsure, unconfirmed: noise
        out.append(PenWord(w.text.strip(), float(w.confidence), b, w.line, w.handwritten, n_pen))
    return out


def assign_words(words: list[PenWord], fields: dict[str, PxBox], answer_cells: list[PxBox],
                 s: Settings) -> dict[str, FieldText]:
    """Give each pen word to the box it was written in.

    1. a word whose centre is in a box belongs to it (or, failing that, the box it
       overlaps most, if at least 30% of the word);
    2. other words of the same text line follow it - writing that starts in a box and
       runs out of it (past the edge, over a line) stays with that box;
    3. words still unplaced and within `outside_box_px` of a box (and in no answer cell)
       are kept as writing *beside* that box.
    Words inside the Yes / No / NA cells are left to the mark reader."""
    out = {k: FieldText() for k in fields}
    where: dict[int, Optional[str]] = {}
    for i, w in enumerate(words):
        c = w.centre
        if any(_inside(c, b, 0) for b in answer_cells):
            where[i] = "#answer"
            continue
        hit = [k for k, b in fields.items() if _inside(c, b)]
        if not hit:
            area = max(1, (w.box[2] - w.box[0]) * (w.box[3] - w.box[1]))
            best = max(fields, key=lambda k: _inter(w.box, fields[k]), default=None)
            if best is not None and _inter(w.box, fields[best]) >= 0.3 * area:
                hit = [best]
        where[i] = hit[0] if hit else None

    by_line: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        if w.line >= 0:
            by_line.setdefault(w.line, []).append(i)
    for i, w in enumerate(words):
        if where[i] is not None or w.line < 0:
            continue
        mates = [j for j in by_line.get(w.line, []) if where[j] not in (None, "#answer")]
        if mates:
            j = min(mates, key=lambda j: _gap(words[j].box, w.box))
            if _gap(words[j].box, w.box) <= 3 * s.outside_box_px:
                where[i] = where[j]
                out[where[j]].overflow = True

    # one written line split over cells stacked above each other (a line that sags over
    # the row line) belongs to the cell holding most of it - so a word that dips into the
    # next row does not make that row look filled. Cells side by side are left alone.
    for line, idx in by_line.items():
        ks = {where[i] for i in idx if where[i] not in (None, "#answer")}
        if len(ks) < 2:
            continue
        ks = sorted(ks)
        stacked = all(min(fields[a][2], fields[b][2]) - max(fields[a][0], fields[b][0]) > 0
                      for a in ks for b in ks if a < b)
        if not stacked:
            continue
        area = {k: sum((words[i].box[2] - words[i].box[0]) * (words[i].box[3] - words[i].box[1])
                       for i in idx if where[i] == k) for k in ks}
        main = max(ks, key=lambda k: area[k])
        for i in idx:
            if where[i] in ks and where[i] != main:
                where[i] = main
                out[main].overflow = True

    for i, w in enumerate(words):
        k = where[i]
        if k == "#answer":
            continue
        if k is not None:
            out[k].words.append(w)
            b = fields[k]
            if w.box[2] > b[2] + 8 or w.box[0] < b[0] - 8 or w.box[3] > b[3] + 6 or w.box[1] < b[1] - 6:
                out[k].overflow = True
            continue
        near = min(fields, key=lambda k2: _gap(w.box, fields[k2]), default=None)
        if near is not None and _gap(w.box, fields[near]) <= s.outside_box_px:
            out[near].outside.append(w)
    return out


def checkbox_mark(page: Optional[PageText], box: PxBox) -> Optional[PageMark]:
    """The reader's selection mark lying in `box` (template px), if any."""
    if page is None:
        return None
    best, best_a = None, 0
    for m in page.marks:
        xs, ys = m.polygon[:, 0], m.polygon[:, 1]
        mb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        a = _inter(mb, box)
        if a > best_a:
            best, best_a = m, a
    return best


# --------------------------------------------------------------------------- #
def nearby_pen_blobs(ctx, box: PxBox, all_boxes: list[PxBox], s: Settings) -> list[PxBox]:
    """Pen writing (ink check) entirely outside every box of the page but within
    `outside_box_px` of `box` - writing beside a box instead of in it."""
    H, W = ctx.shape
    m = s.outside_box_px
    x0, y0 = max(0, box[0] - m), max(0, box[1] - m)
    x1, y1 = min(W, box[2] + m), min(H, box[3] + m)
    sub = ctx.maps.added[y0:y1, x0:x1]
    if getattr(ctx, "faint", None) is not None:
        sub = sub | ctx.faint.added[y0:y1, x0:x1]
    if not sub.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (s.text_join_x, s.text_join_y))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(sub.astype(np.uint8), k))
    near_print = None
    if ctx.tpl_core is not None:
        near_print = cv2.dilate(ctx.tpl_core[y0:y1, x0:x1].astype(np.uint8),
                                np.ones((13, 13), np.uint8)) > 0
    out = []
    for j in range(1, n):
        P = (lab == j) & sub
        npx = int(P.sum())
        if npx < s.min_text_blob_px:
            continue
        ys, xs = np.nonzero(P)
        b = (int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0 + 1), int(ys.max() + y0 + 1))
        bw, bh = b[2] - b[0], b[3] - b[1]
        if min(bw, bh) <= 2 * s.sliver_px or min(bw, bh) < 0.12 * max(bw, bh):
            continue                                   # a line (rule residue, page edge)
        if near_print is not None and (P & near_print).sum() >= 0.7 * npx:
            continue                                   # hugs the printed form: residue
        if any(_inter(b, ob) > 0.15 * (b[2] - b[0]) * (b[3] - b[1]) for ob in all_boxes):
            continue                                   # (mostly) in some box: not "beside"
        if _gap(b, box) <= m:
            out.append(b)
    return out


def fuse_presence(strict: str, faint: Optional[str], ft: Optional[FieldText], s: Settings,
                  multiline: bool = False) -> tuple[str, Optional[float], list[str]]:
    """Combine the ink check (`strict`, and the sensitive pass `faint`) with the writing
    the reader found in the box. Returns (state, confidence or None to keep, notes).

      ink filled                       -> filled (reader not needed: signatures, scribbles)
      ink blank, reader reads it surely -> filled  ("faint writing")
      ink blank/unclear, reader unsure  -> uncertain (REVIEW, with what it read)
      ink blank, faint marks only       -> uncertain (REVIEW)
    """
    notes: list[str] = []
    state, conf = strict, None
    if strict == "blank" and faint in ("filled", "uncertain"):
        state = "uncertain"
        notes.append("faint marks in the box (light pen or pencil?)")
    has = ft is not None and bool(ft.words)
    if has and state in ("blank", "uncertain", "occluded"):
        txt = ft.text(multiline)
        # some ink seen + a reading = two weak signals that agree; no ink seen at all
        # needs a sure reading
        needed = s.ocr_min_conf if strict == "uncertain" else s.azure_accept_conf
        if ft.confidence >= needed and state != "occluded":
            state, conf = "filled", ft.confidence
            notes = [f"light / small writing found by the handwriting reader ('{txt}')"]
        elif state == "occluded":
            notes.append(f"the handwriting reader read '{txt}' in the hidden area")
        else:
            state = "uncertain"
            notes.append(f"the handwriting reader may have found writing here ('{txt}', "
                         f"confidence {ft.confidence:.2f})")
    return state, conf, notes
