"""Reading the circled / ticked / cancelled answer of each Yes / No / NA row.

Instead of measuring ink per cell, every pen stroke group in a section's response
column is analysed as a whole:

  circle  - a ring of ink whose convex hull encloses a printed word. The word it
            encloses is the answer, even when the circle spills over into the next
            cell ("circle overlaps Yes and No").
  tick    - an open stroke (check mark) inside a cell -> that cell's option.
  cross   - an X-shaped mark that is not a cancellation of a circled word -> counts like a
            tick.
  cancel  - an option that was circled and then crossed out (or simply crossed out).
            By default the option is WITHDRAWN: the answer is whatever else is circled,
            and a row whose only answer was cancelled has no valid response.
            See Settings.cancel_mark_means.
  written - writing in a cell that has no printed option (e.g. a handwritten "NA"
            on rows where NA is not printed) -> that option was used.

Strokes are grouped generously because removing the printed form from the scan cuts
small gaps into circles where they cross printed text or cell borders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .context import PageContext
from .settings import Settings
from .template import Item


@dataclass
class Mark:
    bbox: tuple[int, int, int, int]          # template px (x0, y0, x1, y1)
    ink: int
    kind: str              # circle | partial-circle | tick | cross | cancel | written | speck
    option: Optional[str]
    confidence: float
    enclosure: dict = field(default_factory=dict)
    note: str = ""

    def describe(self) -> str:
        return f"{self.kind} on {self.option}" if self.option else self.kind


@dataclass
class ChoiceReading:
    state: str                    # selected | none | ambiguous | uncertain | occluded | cancelled
    selected: Optional[str]
    confidence: float
    note: str
    marks: list[Mark] = field(default_factory=list)
    visible: float = 1.0
    stray_ink: float = 0.0

    @property
    def mark_type(self) -> str:
        """Kind of the main mark for the selected option (circle preferred, then the
        mark with the most ink); for unclear rows, a description of all marks."""
        sig = [m for m in self.marks if m.kind != "speck"]
        if self.selected:
            mine = [m for m in sig if m.option == self.selected]
            if not mine:
                return ""
            circ = [m for m in mine if m.kind == "circle"]
            return (circ[0] if circ else max(mine, key=lambda m: m.ink)).kind
        return ", ".join(m.describe() for m in sig)

    @property
    def has_content(self) -> bool:
        return self.state in ("selected", "ambiguous", "uncertain", "cancelled") or bool(
            [m for m in self.marks if m.kind != "speck"])


# --------------------------------------------------------------------------- #
def _shape_features(P: np.ndarray):
    """P: bool mask (local). Returns hull mask, coverage, ring fraction, quadrants."""
    ys, xs = np.nonzero(P)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    hull = cv2.convexHull(pts)
    hm = np.zeros(P.shape, np.uint8)
    cv2.fillConvexPoly(hm, hull, 1)
    M = cv2.moments(hm, binaryImage=True)
    if M["m00"] == 0:
        return hm.astype(bool), 0.0, 0.0, 0
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    # angles in box-normalised coordinates, so a flat ellipse counts like a circle
    bw, bh = max(xs.max() - xs.min(), 1), max(ys.max() - ys.min(), 1)
    ang = np.arctan2((ys - cy) / bh, (xs - cx) / bw)
    hist, _ = np.histogram(ang, bins=24, range=(-np.pi, np.pi))
    coverage = float((hist >= max(2, 0.01 * len(xs))).mean())
    dist = cv2.distanceTransform(hm, cv2.DIST_L2, 3)
    x0, y0, w, h = cv2.boundingRect(hull)
    ring = float((dist[ys, xs] <= max(3.0, 0.2 * min(w, h))).mean())
    bx0, by0 = xs.min(), ys.min()
    mx, my = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    q = [((xs < mx) & (ys < my)).mean(), ((xs >= mx) & (ys < my)).mean(),
         ((xs < mx) & (ys >= my)).mean(), ((xs >= mx) & (ys >= my)).mean()]
    quads = int(sum(v >= 0.08 for v in q))
    return hm.astype(bool), coverage, ring, quads


def _diag_ink(L: np.ndarray, anti: bool, tol: int = 3) -> float:
    """Share of points along a bbox diagonal (outer two thirds only - the crossing
    point is often erased together with the printed word beneath it) near ink."""
    h, w = L.shape
    d = cv2.dilate(L.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    t = np.concatenate([np.linspace(0.04, 0.33, 12), np.linspace(0.67, 0.96, 12)])
    xs = (t * (w - 1)).astype(int)
    ys = ((1 - t) * (h - 1) if anti else t * (h - 1)).astype(int)
    return float(d[ys, xs].mean())


def _is_cross(L: np.ndarray, quads: int) -> bool:
    """X: ink along BOTH diagonals of the mark's box."""
    h, w = L.shape
    if quads < 4 or not (0.45 <= w / max(h, 1) <= 2.2):
        return False
    return _diag_ink(L, False) >= 0.6 and _diag_ink(L, True) >= 0.6


def _thin(mask: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """Zhang-Suen thinning of a small boolean mask to 1-px centre lines. Thick pen strokes
    (blurred scans, felt pens) otherwise give the line detector many short horizontal
    pieces inside the blob instead of the stroke's real direction."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            c = P[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                changed = True
                img[1:-1, 1:-1][rm] = 0
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def word_struck(ctx: PageContext, wb: tuple[int, int, int, int], s: Settings,
                row_box: Optional[tuple[int, int, int, int]] = None) -> Optional[str]:
    """Is this printed option word crossed out?  Looks for straight pen strokes through the
    word: an X (two lines on opposite diagonals) or a horizontal strike through its middle.
    Curved circle strokes and the single long arm of a tick do not qualify, so a tick or a
    tight circle is never mistaken for a cancellation."""
    import math
    if ctx.tpl_core is None:
        return None
    x0, y0, x1, y1 = wb
    ww, wh = x1 - x0, y1 - y0
    # a stroke counts as crossing the word only if it passes this close to its centre;
    # measured on the word's smaller side, so the sides of a circle drawn tightly around
    # a tall word printed on two lines ("Ye/s") don't qualify
    wmin = min(ww, wh)
    H, W = ctx.shape
    X0, Y0 = max(0, int(x0 - 0.6 * ww)), max(0, int(y0 - 1.2 * wh))
    X1, Y1 = min(W, int(x1 + 0.6 * ww)), min(H, int(y1 + 1.2 * wh))
    if row_box is not None:                 # strokes of the rows above / below don't count
        Y0, Y1 = max(Y0, row_box[1] - 4), min(Y1, row_box[3] + 4)
    from .ink import ink_mask
    pad = 30
    big = ctx.maps.warped[max(0, Y0 - pad):Y1 + pad, max(0, X0 - pad):X1 + pad]
    oy, ox = Y0 - max(0, Y0 - pad), X0 - max(0, X0 - pad)
    ink = ink_mask(big, s)[oy:oy + (Y1 - Y0), ox:ox + (X1 - X0)]
    # printed letters, and a narrow band around the table rules (residue of slightly
    # misaligned rules would otherwise look like straight strokes), are excluded
    E = ink & ~ctx.tpl_core[Y0:Y1, X0:X1] & ~ctx.maps.occluded[Y0:Y1, X0:X1]
    if ctx.rules_near is not None:
        E &= ~ctx.rules_near[Y0:Y1, X0:X1]
    # strokes are looked for both as drawn and thinned to centre lines: thick strokes
    # fused with the circle (blurred scans, felt pens) are only found on the thinned mask,
    # strokes already cut into pieces by the printed letters only on the original one
    masks = [E.astype(np.uint8) * 255, _thin(E).astype(np.uint8) * 255]

    def hough(th, min_len, gap):
        found = [cv2.HoughLinesP(m, 1, np.pi / 180, threshold=th, minLineLength=min_len,
                                 maxLineGap=gap) for m in masks]
        found = [f.reshape(-1, 4) for f in found if f is not None]
        return np.vstack(found) if found else None

    segs = hough(18, max(10, int(0.45 * ww)), 12)
    if segs is None:
        return None
    # shorter pieces may confirm the second diagonal (its arms are often clipped)
    short = hough(10, max(8, int(0.25 * ww)), 8)
    px_, py_ = (x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0
    # the middle of the word (inner 60%): an X or a strike drawn through the word crosses
    # it; a circle drawn around the word - however tightly - only touches its edges
    ix0, ix1 = x0 + 0.2 * ww - X0, x1 - 0.2 * ww - X0
    iy0, iy1 = y0 + 0.2 * wh - Y0, y1 - 0.2 * wh - Y0

    ext = 0.5 * max(ww, wh)

    def through_word(xa, ya, xb, yb):
        # the parts of an X lying over the letters are removed with the print, so the
        # arms that are left are extended along their direction: an arm of an X points
        # through the word, the edge of a circle curves around it
        L = max(np.hypot(xb - xa, yb - ya), 1.0)
        ux, uy = (xb - xa) / L, (yb - ya) / L
        xa, ya, xb, yb = xa - ext * ux, ya - ext * uy, xb + ext * ux, yb + ext * uy
        t = np.linspace(0.0, 1.0, 48)
        xs_, ys_ = xa + (xb - xa) * t, ya + (yb - ya) * t
        return bool(((xs_ >= ix0) & (xs_ <= ix1) & (ys_ >= iy0) & (ys_ <= iy1)).any())

    reach = {"A": [np.inf, -np.inf], "B": [np.inf, -np.inf]}
    strike = False
    straight = []                     # strokes through the word: removed before the ring test
    for xa, ya, xb, yb in segs.reshape(-1, 4).astype(float):
        length = math.hypot(xb - xa, yb - ya)
        ang = math.degrees(math.atan2(-(yb - ya), xb - xa)) % 180
        d = abs((xb - xa) * (ya - py_) - (xa - px_) * (yb - ya)) / max(length, 1)
        cls = ("A" if 20 <= ang <= 70 else "B" if 110 <= ang <= 160 else None)
        if not through_word(xa, ya, xb, yb):
            continue
        if cls and d <= 0.4 * wmin:
            reach[cls][0] = min(reach[cls][0], xa, xb)
            reach[cls][1] = max(reach[cls][1], xa, xb)
            straight.append((xa, ya, xb, yb))
        elif (ang <= 12 or ang >= 168) and d <= 0.25 * wh and length >= 0.9 * ww:
            strike = True
            straight.append((xa, ya, xb, yb))
    # an X: one diagonal line through the word's centre reaches well past BOTH sides of it,
    # and a stroke on the opposite diagonal also passes through the centre. (Arms are often
    # clipped where they cross the circle or a row line, so only one needs to span.) A
    # curved circle stroke broken into short straight pieces stays at one side of the word;
    # the short arm of a tick does not pass through the word's centre.
    def spans(c):
        return reach[c][0] <= px_ - 0.3 * ww and reach[c][1] >= px_ + 0.3 * ww

    present_short = set()
    for xa, ya, xb, yb in (short.reshape(-1, 4).astype(float) if short is not None else []):
        length = math.hypot(xb - xa, yb - ya)
        ang = math.degrees(math.atan2(-(yb - ya), xb - xa)) % 180
        d = abs((xb - xa) * (ya - py_) - (xa - px_) * (yb - ya)) / max(length, 1)
        if d <= 0.4 * wmin and through_word(xa, ya, xb, yb):
            c = "A" if 20 <= ang <= 70 else "B" if 110 <= ang <= 160 else None
            if c:
                present_short.add(c)
                straight.append((xa, ya, xb, yb))
                # short pieces count towards the reach too: an X over a short word is cut
                # into pieces by the printed letters. A piece only qualifies if its line
                # passes through the word centre (d), which arcs of a circle never do.
                reach[c][0] = min(reach[c][0], xa, xb)
                reach[c][1] = max(reach[c][1], xa, xb)

    def present(c):
        return reach[c][1] > reach[c][0] or c in present_short
    if (spans("A") and present("B")) or (spans("B") and present("A")):
        how = "crossed out"
    elif spans("A") or spans("B"):
        # a single straight slash right across the word (a tick's long arm starts at its
        # vertex near the word's middle, so it never reaches past both sides)
        how = "crossed out (slash)"
    elif strike:
        how = "struck through"
    else:
        return None
    # A cancellation is a word that was CIRCLED and then crossed out. Without a ring
    # around the word, the strokes are an answer (a tick or X drawn over a narrow word),
    # not a cancellation.
    RX0, RY0 = max(0, int(x0 - 1.5 * ww)), max(0, int(y0 - 2.0 * wh))
    RX1, RY1 = min(W, int(x1 + 1.5 * ww)), min(H, int(y1 + 2.0 * wh))
    if row_box is not None:                 # never count the circles of the rows above / below
        RY0, RY1 = max(RY0, row_box[1] - 8), min(RY1, row_box[3] + 8)
    # the crossing strokes themselves are taken out first: the two arms of a tick (or of
    # an X) drawn over a short word reach out in several directions and would otherwise
    # pass for a circle around it
    ring_ink = ctx.maps.added[RY0:RY1, RX0:RX1].astype(np.uint8)
    for xa, ya, xb, yb in straight:
        # extend each piece along its direction: the detected pieces stop short of the
        # stroke's ends
        L = max(math.hypot(xb - xa, yb - ya), 1.0)
        ux, uy = (xb - xa) / L, (yb - ya) / L
        e = 6
        pa = (int(round(xa - e * ux + X0 - RX0)), int(round(ya - e * uy + Y0 - RY0)))
        pb = (int(round(xb + e * ux + X0 - RX0)), int(round(yb + e * uy + Y0 - RY0)))
        cv2.line(ring_ink, pa, pb, 0, s.strike_erase_px)
    ring = _ring_coverage(ring_ink.astype(bool), (x0 + x1) / 2 - RX0,
                          (y0 + y1) / 2 - RY0, ww, wh)
    if ring < s.cancel_ring_coverage:
        return None
    return how


def _ring_coverage(ink: np.ndarray, cx: float, cy: float, ww: float, wh: float,
                   bins: int = 16) -> float:
    """Share of directions around the word centre in which pen ink lies outside the
    word's core (a circle covers nearly all directions, an X or a tick only a few)."""
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return 0.0
    dx, dy = (xs - cx) / max(ww / 2, 1), (ys - cy) / max(wh / 2, 1)
    r = np.hypot(dx, dy)
    keep = r >= 0.8                                    # on or beyond the word's outline
    if not keep.any():
        return 0.0
    ang = (np.arctan2(dy[keep], dx[keep]) + np.pi) / (2 * np.pi)
    hit = np.bincount(np.minimum((ang * bins).astype(int), bins - 1), minlength=bins)
    return float((hit >= 2).mean())


def _is_tick(L: np.ndarray) -> bool:
    """Check mark: from its lowest point, arms rise on BOTH sides (a V / check shape)."""
    ys, xs = np.nonzero(L)
    if len(ys) < 10:
        return False
    h = ys.max() - ys.min() + 1
    bottom = ys >= ys.max() - max(2, 0.12 * h)
    bx = xs[bottom].mean()
    left, right = xs < bx - 2, xs > bx + 2
    if left.mean() < 0.12 or right.mean() < 0.12:
        return False
    rise = ys.max() - 0.35 * h
    # both arms must reach well above the bottom; the centre above the vertex is empty
    vx0, vx1 = int(bx - 2 - xs.min()), int(bx + 3 - xs.min())
    above = L[ys.min():int(ys.min() + 0.5 * h), max(0, vx0):max(0, vx1)]
    return ys[left].min() < rise and ys[right].min() < rise and not above.any()


def _ring_quality(P: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(P)
    if len(ys) < 10:
        return 0.0, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    _, coverage, ring, _ = _shape_features(P[y0:y1, x0:x1])
    return coverage, ring


def _merge_ring_pieces(pieces: list, s: Settings) -> list:
    """Within a row band, rejoin arcs of one circle that were cut apart where the circle
    ran over printed text, a cell border or a row line.

    Pieces closer than ring_bridge_px form clusters. A cluster is merged when the union is
    a clearly better ring than its best piece. Pieces that are already (nearly) complete
    circles are never merged with anything, so two separate circles stay two answers."""
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s.ring_bridge_px + 1,) * 2)
    out = []
    for band in sorted({p[0] for p in pieces}):
        items = [p[1] for p in pieces if p[0] == band]
        qual = [_ring_quality(P) for P in items]
        complete = [c >= 0.85 and r >= s.circle_ring for c, r in qual]
        n = len(items)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        dil = [cv2.dilate(P.astype(np.uint8), bridge).astype(bool) for P in items]
        for i in range(n):
            for j in range(i + 1, n):
                if not complete[i] and not complete[j] and (dil[i] & items[j]).any():
                    parent[find(i)] = find(j)
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        for idx in clusters.values():
            if len(idx) == 1:
                out.append([band, items[idx[0]]])
                continue
            union = np.zeros_like(items[idx[0]])
            for i in idx:
                union |= items[i]
            cu, ru = _ring_quality(union)
            best = max(qual[i][0] for i in idx)
            if cu >= best + 0.05 and ru >= s.circle_ring and cu >= s.circle_coverage:
                out.append([band, union])
            else:
                out += [[band, items[i]] for i in idx]
    return out


def _merge_fragments(pieces: list, s: Settings) -> list:
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s.mark_group_px + 1,) * 2)

    def complete(P):
        c, r = _ring_quality(P)
        return c >= 0.75 and r >= s.circle_ring

    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            if complete(pieces[i][1]):
                continue
            di = cv2.dilate(pieces[i][1].astype(np.uint8), near).astype(bool)
            for j in range(len(pieces)):
                if i == j or pieces[i][0] != pieces[j][0] or complete(pieces[j][1]):
                    continue
                if (di & pieces[j][1]).any():
                    pieces[i][1] = pieces[i][1] | pieces[j][1]
                    pieces.pop(j)
                    changed = True
                    break
            if changed:
                break
    return pieces


def _encloses_word(P: np.ndarray, words: dict, ox: int, oy: int, s: Settings) -> Optional[str]:
    """The printed word (template px boxes; the piece's sub-image starts at ox, oy) that
    the piece's outline surrounds, if any."""
    ys, xs = np.nonzero(P)
    if len(ys) < 10:
        return None
    gx0, gy0 = xs.min(), ys.min()
    L = P[gy0:ys.max() + 1, gx0:xs.max() + 1]
    hull, coverage, ring, _ = _shape_features(L)
    if coverage < 0.35:
        return None
    for o, wb in words.items():
        x0, y0, x1, y1 = wb[0] - ox - gx0, wb[1] - oy - gy0, wb[2] - ox - gx0, wb[3] - oy - gy0
        dx, dy = int(0.1 * (x1 - x0)), int(0.1 * (y1 - y0))
        ix0, iy0 = max(0, x0 + dx), max(0, y0 + dy)
        ix1, iy1 = min(hull.shape[1], x1 - dx), min(hull.shape[0], y1 - dy)
        area = max((x1 - x0 - 2 * dx) * (y1 - y0 - 2 * dy), 1)
        if ix1 > ix0 and iy1 > iy0 and hull[iy0:iy1, ix0:ix1].sum() >= s.enclose_min * area:
            return o
    return None


def _merge_around_words(pieces: list, rows: list[dict], ox: int, oy: int, s: Settings) -> list:
    """Within a row band, join pieces that together - but none alone - surround a printed
    word: a circle drawn tightly around a word loses the parts that ran over its letters,
    leaving e.g. a left and a right arc."""
    out = []
    for band in sorted({p[0] for p in pieces}):
        items = [p[1] for p in pieces if p[0] == band]
        words = rows[band]["words"]
        used = set()
        for o, wb in words.items():
            cell = rows[band]["cells"].get(o, wb)
            wh = wb[3] - wb[1]
            near = []
            for k, P in enumerate(items):
                if k in used or _encloses_word(P, words, ox, oy, s):
                    continue
                ys, xs = np.nonzero(P)
                # only pieces in this word's own cell (a circle's arcs around it), never
                # pieces of the neighbouring option's mark
                cx = (xs.min() + xs.max()) / 2 + ox
                by0, by1 = ys.min() + oy, ys.max() + oy
                if cell[0] - 8 <= cx <= cell[2] + 8 and by1 >= wb[1] - wh and by0 <= wb[3] + wh:
                    near.append(k)
            if len(near) < 2:
                continue
            union = np.zeros_like(items[near[0]])
            for k in near:
                union |= items[k]
            if _encloses_word(union, {o: wb}, ox, oy, s) == o:
                out.append([band, union])
                used.update(near)
        out += [[band, P] for k, P in enumerate(items) if k not in used]
    return out


def _merge_side_spills(pieces: list, rows: list[dict], ox: int, oy: int, s: Settings) -> list:
    """Within a row band, a small piece that surrounds no word and sits right next to a
    circle that does is part of that circle - typically where the circle ran over a cell
    border into the neighbouring option's cell and was cut there."""
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * s.mark_group_px + 1,) * 2)

    def on_other_word(Q, words, own):
        """Does the piece lie on a printed word other than the circle's own? (Then it is
        a mark of its own - a tick on No beside a circle on Yes is two answers.)"""
        ys, xs = np.nonzero(Q)
        qx0, qy0, qx1, qy1 = xs.min() + ox, ys.min() + oy, xs.max() + ox, ys.max() + oy
        area = max((qx1 - qx0) * (qy1 - qy0), 1)
        for o, wb in words.items():
            if o == own:
                continue
            w = min(qx1, wb[2]) - max(qx0, wb[0])
            h = min(qy1, wb[3]) - max(qy0, wb[1])
            if w > 0 and h > 0 and w * h > 0.15 * area:
                return True
        return False

    changed = True
    while changed:
        changed = False
        for i, (bi, P) in enumerate(pieces):
            own = _encloses_word(P, rows[bi]["words"], ox, oy, s)
            if own is None:
                continue
            grown = cv2.dilate(P.astype(np.uint8), near).astype(bool)
            for j, (bj, Q) in enumerate(pieces):
                if j == i or bj != bi or Q.sum() >= s.spill_merge_ratio * P.sum():
                    continue
                if _encloses_word(Q, rows[bj]["words"], ox, oy, s) is not None:
                    continue
                if on_other_word(Q, rows[bj]["words"], own):
                    continue
                if (grown & Q).any():
                    pieces[i][1] = P | Q
                    pieces.pop(j)
                    changed = True
                    break
            if changed:
                break
    return pieces


def _row_aware_groups(sub: np.ndarray, rows: list[dict], ry0: int, s: Settings,
                      rx0: int = 0) -> list[np.ndarray]:
    """Group stroke pieces into marks without merging marks of neighbouring rows.

    Pieces are grouped generously inside each row band (bridging the gaps that removing
    the printed form cuts into circles). Across a row line, two groups are merged only
    when they physically touch and one is much smaller than the other - a circle that
    spills over the line - so stacked circles in consecutive rows stay separate."""
    H = sub.shape[0]
    bounds = [0]
    for a, b in zip(rows[:-1], rows[1:]):
        bounds.append(int(round((a["box"][3] + b["box"][1]) / 2)) - ry0)
    bounds.append(H)
    k = max(3, s.mark_base_group_px) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    pieces = []                       # (band index, mask)
    for bi in range(len(rows)):
        y0, y1 = max(0, bounds[bi]), min(H, bounds[bi + 1])
        if y1 <= y0:
            continue
        band = np.zeros_like(sub)
        band[y0:y1] = sub[y0:y1]
        if not band.any():
            continue
        grp = cv2.dilate(band.astype(np.uint8), ker)
        grp[:y0] = 0
        grp[y1:] = 0
        n, lab = cv2.connectedComponents(grp, connectivity=8)
        for j in range(1, n):
            P = (lab == j) & band
            if P.any():
                pieces.append([bi, P])
    # rejoin pieces of one ring cut apart where it crossed printed text / cell borders,
    # then join remaining fragments (tick / cross pieces) - complete rings stay separate,
    # so two circles drawn close together remain two answers
    pieces = _merge_ring_pieces(pieces, s)
    pieces = _merge_around_words(pieces, rows, rx0, ry0, s)
    pieces = _merge_fragments(pieces, s)
    pieces = _merge_side_spills(pieces, rows, rx0, ry0, s)
    # Across a row line: a circle drawn over the line is cut in two there (the printed
    # line is removed with the form, leaving a gap of a few px in the pen stroke), and the
    # part beyond the line would read as a mark in the neighbouring row. Pieces whose ends
    # face each other across the line are joined when the result is a clearly better
    # circle and neither piece is already a complete circle of its own (two circles on
    # neighbouring rows stay two answers), or when one is just a small spill.
    gap = s.template_dilate_px + 8

    def crossings(P, y_line, above):
        """x-centres of the places where the stroke meets the line steeply, from the 3
        pixel rows of ink nearest the line: a stroke that crosses the line is a few px
        wide there; an arc that merely touches the line runs along it."""
        band = P[max(0, y_line - gap):y_line] if above else P[y_line:y_line + gap]
        rows_with_ink = np.nonzero(band.any(axis=1))[0]
        if not len(rows_with_ink):
            return []
        near = rows_with_ink[-3:] if above else rows_with_ink[:3]
        cols = band[near].any(axis=0)
        out, x = [], 0
        while x < len(cols):
            if cols[x]:
                x1 = x
                while x1 + 1 < len(cols) and cols[x1 + 1]:
                    x1 += 1
                if x1 - x + 1 <= 16:
                    out.append((x + x1) / 2)
                x = x1 + 1
            else:
                x += 1
        return out

    def continues(a, b, y_line):
        xa, xb = crossings(a, y_line, True), crossings(b, y_line, False)
        return any(abs(p - q) <= 12 for p in xa for q in xb)

    def own_answer(P, band):
        """Does this piece on its own surround a printed word of its own row?"""
        return _encloses_word(P, rows[band]["words"], rx0, ry0, s) is not None

    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                bi, bj = pieces[i][0], pieces[j][0]
                if bj != bi + 1:
                    continue
                a, b = pieces[i][1], pieces[j][1]          # a above the line, b below
                if not continues(a, b, bounds[bj]):
                    continue
                # two marks that each surround a word of their own row are two answers
                if own_answer(a, bi) and own_answer(b, bj):
                    continue
                small, big = (a, b) if a.sum() <= b.sum() else (b, a)
                join = small.sum() < s.spill_merge_ratio * big.sum()        # a small spill
                if not join:
                    cu, ru = _ring_quality(a | b)
                    join = (ru >= s.circle_ring and cu >= s.circle_coverage
                            and cu >= max(_ring_quality(a)[0], _ring_quality(b)[0]) + 0.05)
                if not join:
                    continue
                keep = i if big is a else j
                drop = j if keep == i else i
                pieces[keep][1] = a | b
                pieces.pop(drop)
                merged = True
                break
            if merged:
                break
    return [p[1] for p in pieces]


def read_section_choices(ctx: PageContext, items: list[Item], base: Settings
                         ) -> dict[str, ChoiceReading]:
    items = [i for i in items if i.type == "choice"]
    if not items:
        return {}
    s = base
    H, W = ctx.shape
    rows = []
    for it in items:
        cells = {o: ctx.px(b) for o, b in it.options.items()}
        x0 = min(c[0] for c in cells.values()); x1 = max(c[2] for c in cells.values())
        y0 = min(c[1] for c in cells.values()); y1 = max(c[3] for c in cells.values())
        if it.printed:
            words = {o: ctx.px(b) for o, b in it.printed.items()}
        else:
            # hand-calibrated templates have no word boxes: use the centre of each cell
            words = {o: (int(c[0] + 0.3 * (c[2] - c[0])), int(c[1] + 0.3 * (c[3] - c[1])),
                         int(c[2] - 0.3 * (c[2] - c[0])), int(c[3] - 0.3 * (c[3] - c[1])))
                     for o, c in cells.items()}
        rows.append(dict(item=it, cells=cells, box=(x0, y0, x1, y1), words=words))
    word_h = np.median([w[3] - w[1] for r in rows for w in r["words"].values()] or [16])
    row_h = min(r["box"][3] - r["box"][1] for r in rows)

    rx0 = max(0, min(r["box"][0] for r in rows) - 30)
    rx1 = min(W, max(r["box"][2] for r in rows) + 30)
    ry0 = max(0, min(r["box"][1] for r in rows) - int(0.5 * row_h))
    ry1 = min(H, max(r["box"][3] for r in rows) + int(0.5 * row_h))
    sub = ctx.maps.added[ry0:ry1, rx0:rx1]

    groups = _row_aware_groups(sub, rows, ry0, s, rx0)

    per_row: list[list[Mark]] = [[] for _ in rows]
    stray = [0 for _ in rows]
    for P in groups:
        ink = int(P.sum())
        if ink < s.min_component_px:
            continue
        ys, xs = np.nonzero(P)
        gx0, gy0 = int(xs.min() + rx0), int(ys.min() + ry0)
        gx1, gy1 = int(xs.max() + rx0 + 1), int(ys.max() + ry0 + 1)
        # which row: the one holding most of the ink
        counts = [int(((ys + ry0 >= r["box"][1]) & (ys + ry0 < r["box"][3])).sum()) for r in rows]
        ri = int(np.argmax(counts))
        if counts[ri] == 0:
            continue
        r = rows[ri]
        # ignore strokes that mostly sit outside the response cells (details writing,
        # question text) - they are handled elsewhere
        inside_x = ((xs + rx0 >= r["box"][0] - 8) & (xs + rx0 < r["box"][2] + 8)).mean()
        if inside_x < 0.5:
            continue
        size = max(gx1 - gx0, gy1 - gy0)
        if min(gx1 - gx0, gy1 - gy0) <= 2 * s.sliver_px:
            continue                       # thin straight remnant (border, highlight outline)
        if ink < s.mark_min_ink or size < 0.6 * word_h:
            stray[ri] += ink
            per_row[ri].append(Mark((gx0, gy0, gx1, gy1), ink, "speck", None, 0.0))
            continue

        # local mask around the group
        L = P[gy0 - ry0:gy1 - ry0, gx0 - rx0:gx1 - rx0]
        hull, coverage, ring, quads = _shape_features(L)

        def frac_in(box, mask):
            bx0, by0, bx1, by1 = box
            ix0, iy0 = max(bx0, gx0), max(by0, gy0)
            ix1, iy1 = min(bx1, gx1), min(by1, gy1)
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            area = max((bx1 - bx0) * (by1 - by0), 1)
            return float(mask[iy0 - gy0:iy1 - gy0, ix0 - gx0:ix1 - gx0].sum()) / area

        enclosure = {}
        for o, wb in r["words"].items():
            dx, dy = int(0.1 * (wb[2] - wb[0])), int(0.1 * (wb[3] - wb[1]))
            enclosure[o] = round(frac_in((wb[0] + dx, wb[1] + dy, wb[2] - dx, wb[3] - dy), hull), 2)
        share = {o: frac_in(c, L) * max((c[2] - c[0]) * (c[3] - c[1]), 1) / ink
                 for o, c in r["cells"].items()}
        ranked = sorted(enclosure, key=enclosure.get, reverse=True)
        best = ranked[0] if ranked else None
        second = enclosure[ranked[1]] if len(ranked) > 1 else 0.0

        is_ring = coverage >= s.circle_coverage and ring >= s.circle_ring
        # centre of the ring (hull centroid) - which cell is the circle drawn around?
        hm = cv2.moments(hull.astype(np.uint8), binaryImage=True)
        hcx = gx0 + (hm["m10"] / hm["m00"] if hm["m00"] else (gx1 - gx0) / 2)
        centre_cell = next((o for o, c in r["cells"].items() if c[0] <= hcx < c[2]), None)
        if best and enclosure[best] >= s.enclose_min and is_ring:
            def wcx(o):
                wb = r["words"][o]
                return (wb[0] + wb[2]) / 2
            centred = False
            if second >= s.enclose_min and centre_cell == best:
                d_best, d_other = abs(hcx - wcx(best)), abs(hcx - wcx(ranked[1]))
                centred = d_best <= 0.5 * d_other   # drawn around best, sloppy overlap
            if second >= s.enclose_min and not centred:
                both = [o for o in ranked if enclosure[o] >= s.enclose_min]
                m = Mark((gx0, gy0, gx1, gy1), ink, "circle", None, 0.3, enclosure,
                         f"one circle encloses {' and '.join(both)}")
            else:
                conf = 0.95 if enclosure[best] >= 0.8 else 0.8
                note = ""
                spill = [o for o, v in share.items() if o != best and v >= 0.25]
                if spill:
                    note = f"circle around {best} overlaps the {'/'.join(spill)} cell"
                m = Mark((gx0, gy0, gx1, gy1), ink, "circle", best, conf, enclosure, note)
        elif best and enclosure[best] >= 0.8 and second < s.enclose_min and coverage >= 0.35 \
                and ring >= s.circle_ring:
            # an open circle (cut where it crossed a printed line) clearly drawn around one
            # word, and around no other
            m = Mark((gx0, gy0, gx1, gy1), ink, "partial-circle", best, 0.8, enclosure,
                     f"circle around {best} is not closed (crosses a printed line)")
        elif sum(v >= s.enclose_min for v in enclosure.values()) >= 2:
            both = [o for o in ranked if enclosure[o] >= s.enclose_min]
            m = Mark((gx0, gy0, gx1, gy1), ink, "circle", None, 0.3, enclosure,
                     f"one mark surrounds {' and '.join(both)}")
        else:
            cell = max(share, key=share.get)
            note = ""
            conf = 0.85
            if share[cell] < 0.5:
                # straddles cells: decide by the mark's centre; a mark whose centre is
                # outside every response cell (border / details spill) is not an answer
                cxm = (gx0 + gx1) / 2
                inside = [o for o, c in r["cells"].items() if c[0] <= cxm < c[2]]
                if not inside:
                    stray[ri] += ink
                    per_row[ri].append(Mark((gx0, gy0, gx1, gy1), ink, "speck", None, 0.0))
                    continue
                cell = inside[0]
                conf = 0.5
                note = "mark straddles two cells"
            Lc = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            if cell not in r["words"]:
                kind = "written"            # e.g. handwritten NA in an unprinted cell
            elif _is_cross(Lc, quads):
                # an X-shaped mark on its own counts like a tick; cancellation is decided
                # only by the straight-stroke test on the printed word (word_struck)
                kind = "cross"
                conf = min(conf, 0.75)
            elif _is_tick(Lc):
                kind = "tick"
            else:
                kind = "partial-circle"     # an arc / incomplete ring around the answer
            m = Mark((gx0, gy0, gx1, gy1), ink, kind, cell, conf, enclosure, note)
        per_row[ri].append(m)

    out = {}
    for ri, r in enumerate(rows):
        it = r["item"]
        bx0, by0, bx1, by1 = r["box"]
        vis = ctx.visible(bx0, by0, bx1, by1)
        word_vis = min([ctx.visible(*wb) for wb in r["words"].values()] or [1.0])
        marks = per_row[ri]
        sig = [m for m in marks if m.kind != "speck"]
        area = max((bx1 - bx0) * (by1 - by0), 1)
        stray_ratio = stray[ri] / area

        # options whose printed word is crossed / struck out: circled, then cancelled
        struck = {o: k for o, wb in (r["words"].items() if it.printed else [])
                  if (k := word_struck(ctx, wb, s, r["box"]))}
        for m in sig:
            if m.option in struck and m.kind != "written":
                m.kind = "cancel"
                m.note = f"{m.option} circled then {struck[m.option]}"
        if struck and not any(m.option in struck for m in sig):
            sig.append(Mark(r["words"][next(iter(struck))], 0, "cancel", next(iter(struck)), 0.8,
                            note=f"{next(iter(struck))} {struck[next(iter(struck))]}"))
            marks = marks + [sig[-1]]
        # a mark around both words where one word is crossed out: the circled-and-cancelled
        # word drops out, the other one is the answer (also covers two adjacent circles that
        # were grouped together)
        if struck and s.cancel_mark_means == "withdrawn":
            for m in sig:
                if m.kind == "circle" and m.option is None and m.enclosure:
                    left = [o for o, v in m.enclosure.items() if v >= s.enclose_min and o not in struck]
                    gone = [o for o, v in m.enclosure.items() if v >= s.enclose_min and o in struck]
                    if len(left) == 1 and gone:
                        m.option, m.confidence = left[0], 0.75
                        m.note = f"{'/'.join(gone)} circled then {struck[gone[0]]}"
                        sig.append(Mark(m.bbox, 0, "cancel", gone[0], 0.8, note=m.note))
                        marks = marks + [sig[-1]]
        both = [m for m in sig if m.kind == "circle" and m.option is None]
        chosen = [m for m in sig if m.option is not None]
        cancels = [m for m in chosen if m.kind == "cancel"]
        withdrawn = sorted({m.option for m in cancels})
        if s.cancel_mark_means == "withdrawn" and cancels:
            # circled and crossed out = taken back; only the remaining marks count
            chosen = [m for m in chosen if m.kind != "cancel" and m.option not in withdrawn]
        opts = sorted({m.option for m in chosen})
        notes = [m.note for m in sig if m.note]
        if s.cancel_mark_means == "withdrawn" and withdrawn and opts:
            notes = [n for n in notes if "crossed out" not in n] + [
                f"{'/'.join(withdrawn)} was cancelled (crossed out); {'/'.join(opts)} counted"]

        if both:
            rd = ChoiceReading("ambiguous", None, 0.3, both[0].note, marks, vis, stray_ratio)
        elif len(opts) > 1:
            desc = ", ".join(m.describe() for m in chosen)
            rd = ChoiceReading("ambiguous", None, 0.3, f"more than one option marked ({desc})",
                               marks, vis, stray_ratio)
        elif len(opts) == 1:
            o = opts[0]
            conf = max(m.confidence for m in chosen if m.option == o)
            note = "; ".join(notes)
            if s.cancel_mark_means == "review" and any(m.kind == "cancel" for m in chosen):
                rd = ChoiceReading("uncertain", o, 0.5,
                                   f"'{o}' is crossed out rather than circled - confirm intent",
                                   marks, vis, stray_ratio)
            elif word_vis < s.min_visible_frac and len(r["words"]) > 1:
                hidden = [w for w, wb in r["words"].items() if ctx.visible(*wb) < s.min_visible_frac]
                rd = ChoiceReading("occluded", o, min(conf, 0.5),
                                   f"'{o}' looks marked but {'/'.join(hidden)} is obscured",
                                   marks, vis, stray_ratio)
            elif conf < 0.6:
                rd = ChoiceReading("uncertain", o, conf, note or f"unclear mark near '{o}'",
                                   marks, vis, stray_ratio)
            else:
                rd = ChoiceReading("selected", o, conf, note, marks, vis, stray_ratio)
        elif withdrawn and s.cancel_mark_means == "withdrawn":
            rd = ChoiceReading("cancelled", None, 0.6,
                               f"the only answer marked ({'/'.join(withdrawn)}) was crossed out - "
                               f"no valid response", marks, vis, stray_ratio)
        else:
            if vis < s.min_visible_frac or word_vis < s.min_visible_frac:
                rd = ChoiceReading("occluded", None, 0.2,
                                   f"no visible mark; response area {1 - vis:.0%} obscured",
                                   marks, vis, stray_ratio)
            elif stray_ratio > s.mark_lo:
                rd = ChoiceReading("uncertain", None, 0.3,
                                   "only faint / fragmentary marks in the response cells",
                                   marks, vis, stray_ratio)
            else:
                rd = ChoiceReading("none", None, 0.9, "", marks, vis, stray_ratio)
        out[it.id] = rd
    return out
