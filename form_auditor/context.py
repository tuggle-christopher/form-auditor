"""Everything the rule engine needs to know about one aligned page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .ink import InkMaps, box_to_px, measure
from .settings import Settings
from .template import Box


@dataclass
class PageContext:
    maps: InkMaps                 # ink / occlusion maps in template coordinates
    settings: Settings
    original: np.ndarray          # the input page as loaded (BGR)
    transform: object             # align.PageTransform
    tpl_core: Optional[np.ndarray] = None    # printed strokes of the blank form (template px)
    rules_zone: Optional[np.ndarray] = None  # band around the printed table rules
    rules_near: Optional[np.ndarray] = None  # narrow band around the rules (strike test)
    faint: Optional[InkMaps] = None          # second, more sensitive ink pass (faint marks)

    def faint_ctx(self) -> Optional["PageContext"]:
        """The same page seen through the sensitive ink pass."""
        if self.faint is None:
            return None
        from dataclasses import replace
        return replace(self, maps=self.faint, faint=None)

    @property
    def shape(self):
        return self.maps.added.shape

    def px(self, box: Box, inset: float = 0.0) -> tuple[int, int, int, int]:
        return box_to_px(box, self.shape, inset)

    def norm(self, x0, y0, x1, y1) -> Box:
        h, w = self.shape
        return (max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, x1 / w), min(1.0, y1 / h))

    def measure(self, box: Box, s: Settings) -> tuple[float, float]:
        return measure(self.maps, box, s.roi_inset_frac, s.sliver_px)

    def largest_text_blob(self, box: Box, s: Settings) -> int:
        """Ink pixels in the biggest word-like blob inside the box (thin border slivers
        ignored)."""
        x0, y0, x1, y1 = self.px(box, s.roi_inset_frac)
        sub = self.maps.added[y0:y1, x0:x1]
        if not sub.any():
            return 0
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (s.text_join_x, s.text_join_y))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(sub.astype(np.uint8), k))
        best = 0
        for j in range(1, n):
            P = (lab == j) & sub
            ys, xs = np.nonzero(P)
            if len(ys) == 0 or min(np.ptp(xs), np.ptp(ys)) <= 2 * s.sliver_px:
                continue
            best = max(best, int(P.sum()))
        return best

    def visible(self, x0, y0, x1, y1) -> float:
        occ = self.maps.occluded[max(0, y0):y1, max(0, x0):x1]
        return 1.0 - float(occ.mean()) if occ.size else 0.0

    def crop(self, x0, y0, x1, y1, scale: float = 2.0, pad: int = 6) -> np.ndarray:
        """Rectified crop of the ORIGINAL image (full resolution) for a template-pixel box."""
        h, w = self.shape
        region = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        return self.transform.warp(self.original, region=region, scale=scale)
