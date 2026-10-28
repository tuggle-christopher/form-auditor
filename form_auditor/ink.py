"""Isolating the human's ink: (ink on filled page) minus (ink already on the blank form),
ignoring coloured overlay boxes and anything outside the photographed area."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .settings import Settings
from .template import Box


def ink_mask(bgr: np.ndarray, s: Settings) -> np.ndarray:
    """Dark strokes that stand out from the local paper colour.

    Background = large median (immune to sparse strokes, follows shading and the average
    level of screen-pixel grids / moiré); the page is lightly blurred first so periodic
    screen patterns average out while pen strokes keep most of their contrast."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, s.ink_bg_kernel | 1)
    bg = cv2.medianBlur(gray, k).astype(np.float32)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s.ink_blur) if s.ink_blur > 0 \
        else gray.astype(np.float32)
    return ((bg - g) > s.ink_contrast) & (g <= s.dark_max)


def overlay_mask(bgr: np.ndarray, s: Settings) -> np.ndarray:
    """Saturated, bright blue regions (annotation callouts, highlighter-style boxes)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= s.overlay_hue_min) & (h <= s.overlay_hue_max)
         & (sat >= s.overlay_sat_min) & (v >= s.overlay_val_min)).astype(np.uint8)
    # close the white lettering inside the callouts, then grow slightly
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


def strip_straight_lines(mask: np.ndarray, length: int, near_rules: np.ndarray | None = None,
                         long_factor: float = 3.0) -> np.ndarray:
    """Remove straight horizontal/vertical runs that are not handwriting:
      * runs >= `length` lying on a printed table rule (rule residue after imperfect
        alignment), and
      * runs >= long_factor * `length` anywhere (outlines of highlight boxes, rulers).
    Flat tops/bottoms of wide circles are shorter than the second limit and are not on a
    rule, so they survive."""
    m = mask.astype(np.uint8)

    def runs(L):
        h = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        v = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        return h | v

    short = runs(length) > 0
    if near_rules is not None:
        short &= near_rules
    lines = short | (runs(int(length * long_factor)) > 0)
    lines = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~lines


def rule_zone(template_bgr: np.ndarray, px: int = 9) -> np.ndarray:
    """Band around the template's printed table rules."""
    from .align import is_crisp, line_masks
    hl, vl = line_masks(template_bgr, crisp=is_crisp(template_bgr))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.dilate((hl | vl).astype(np.uint8), k) > 0


def template_core(template_bgr: np.ndarray, s: Settings) -> np.ndarray:
    """The printed strokes themselves, grown just enough to cover scan blur."""
    m = ink_mask(template_bgr, s).astype(np.uint8)
    k = max(1, 2 * s.print_core_px + 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def template_ink(template_bgr: np.ndarray, s: Settings) -> np.ndarray:
    m = ink_mask(template_bgr, s).astype(np.uint8)
    k = max(1, s.template_dilate_px)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


@dataclass
class InkMaps:
    added: np.ndarray       # bool, new ink in template coordinates
    occluded: np.ndarray    # bool, pixels we cannot see (overlay / outside photo)
    warped: np.ndarray      # BGR filled page warped into template space


def compute_ink_maps(img_bgr: np.ndarray, transform, tpl_ink_dilated: np.ndarray,
                     s: Settings, rules_zone: np.ndarray | None = None,
                     tpl_core: np.ndarray | None = None) -> InkMaps:
    """`transform` is an align.PageTransform (homography + local correction)."""
    warped = transform.warp(img_bgr)
    valid = transform.valid_mask(img_bgr.shape)
    k = max(1, s.edge_margin_px)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    occluded = ~valid
    if s.mask_overlays:
        occluded |= overlay_mask(warped, s)

    ink = ink_mask(warped, s)
    added = ink & ~tpl_ink_dilated & ~occluded
    if s.recover_halo and tpl_core is not None:
        # Pen strokes crossing printed text lose everything inside the subtraction margin.
        # Give back ink in that margin (not on the printed strokes themselves) wherever it
        # continues an existing pen stroke.
        near_pen = cv2.dilate(remove_small(added, s.min_component_px).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        halo = tpl_ink_dilated & ~tpl_core
        added |= ink & halo & near_pen & ~occluded
    if s.strip_line_frac > 0:
        added = strip_straight_lines(added, max(15, int(added.shape[1] * s.strip_line_frac)),
                                     rules_zone)
    added = remove_small(added, s.min_component_px)
    return InkMaps(added, occluded, warped)


def box_to_px(box: Box, shape, inset_frac: float = 0.0) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    dx, dy = (x1 - x0) * inset_frac, (y1 - y0) * inset_frac
    return (int(round(x0 + dx)), int(round(y0 + dy)), int(round(x1 - dx)), int(round(y1 - dy)))


def measure(maps: InkMaps, box: Box, inset_frac: float, sliver_px: int = 0) -> tuple[float, float]:
    """Return (ink_ratio, visible_ratio) inside a normalised box."""
    x0, y0, x1, y1 = box_to_px(box, maps.added.shape, inset_frac)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    occ = maps.occluded[y0:y1, x0:x1]
    total = occ.size
    visible = total - int(occ.sum())
    if visible == 0:
        return 0.0, 0.0
    sub = maps.added[y0:y1, x0:x1]
    if sliver_px > 0 and sub.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
        thin = np.zeros(n, bool)
        thin[1:] = np.minimum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) <= sliver_px
        sub = sub & ~thin[lab]
    ink = int(sub.sum())
    return ink / visible, visible / total
