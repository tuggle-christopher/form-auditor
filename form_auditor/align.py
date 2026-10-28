"""Registers a filled page (scan or phone photo) onto the blank template.

SIFT keypoints + RANSAC homography handles rotation, scale and perspective
(photos of paper or of a screen). An optional ECC pass then refines the fit to
sub-pixel accuracy so that template subtraction leaves only the human's ink.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .settings import Settings


class PageTransform:
    """Input page -> template coordinates: a homography H followed by a small, smooth
    correction field (cubic polynomials dx(x,y), dy(x,y) in template space) that takes up
    lens distortion and paper curl left over after the homography.

    template point p corresponds to warped-homography point q = p + d(p), i.e. image point
    Hinv(q)."""

    def __init__(self, H: np.ndarray, tpl_shape, cx=None, cy=None, bounds=None):
        self.H = H
        self.Hinv = np.linalg.inv(H)
        self.shape = tpl_shape[:2]
        self.cx, self.cy = cx, cy
        self.bounds = bounds          # (x0, y0, x1, y1) where the correction was fitted

    @property
    def has_local(self) -> bool:
        return self.cx is not None

    def _basis(self, x, y):
        h, w = self.shape
        x0, y0, x1, y1 = self.bounds
        u = (np.clip(x, x0, x1) - x0) / max(x1 - x0, 1)
        v = (np.clip(y, y0, y1) - y0) / max(y1 - y0, 1)
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v,
                         u ** 3, u * u * v, u * v * v, v ** 3], -1)

    def offsets(self, x, y):
        if not self.has_local:
            return np.zeros_like(x, dtype=np.float64), np.zeros_like(y, dtype=np.float64)
        B = self._basis(np.asarray(x, np.float64), np.asarray(y, np.float64))
        return B @ self.cx, B @ self.cy

    def tpl_to_img(self, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        dx, dy = self.offsets(pts[:, 0], pts[:, 1])
        q = np.stack([pts[:, 0] + dx, pts[:, 1] + dy], -1).astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(q, self.Hinv).reshape(-1, 2)

    def maps(self, region=None, scale: float = 1.0, step: int = 4):
        """Remap grids for template region (x0, y0, x1, y1 in template px) at `scale`.

        Large grids are computed exactly on every `step`-th pixel and interpolated in
        between: the mapping is smooth (homography + cubic correction), so the error is
        far below a pixel, while memory and time drop ~step^2-fold."""
        h, w = self.shape
        x0, y0, x1, y1 = region or (0, 0, w, h)
        xs = np.arange(int(round((x1 - x0) * scale)), dtype=np.float64) / scale + x0
        ys = np.arange(int(round((y1 - y0) * scale)), dtype=np.float64) / scale + y0
        if len(xs) * len(ys) <= 1_000_000 or step <= 1 or len(xs) < 2 * step or len(ys) < 2 * step:
            X, Y = np.meshgrid(xs, ys)
            src = self.tpl_to_img(np.stack([X.ravel(), Y.ravel()], -1))
            return (src[:, 0].reshape(X.shape).astype(np.float32),
                    src[:, 1].reshape(X.shape).astype(np.float32))
        ix = np.unique(np.r_[np.arange(0, len(xs), step), len(xs) - 1])
        iy = np.unique(np.r_[np.arange(0, len(ys), step), len(ys) - 1])
        X, Y = np.meshgrid(xs[ix], ys[iy])
        src = self.tpl_to_img(np.stack([X.ravel(), Y.ravel()], -1))

        def up(c):                                     # coarse (len(iy), len(ix)) -> full
            j = np.clip(np.searchsorted(ix, np.arange(len(xs)), side="right") - 1, 0, len(ix) - 2)
            t = ((np.arange(len(xs)) - ix[j]) / (ix[j + 1] - ix[j])).astype(np.float32)
            rows = c[:, j] * (1 - t) + c[:, j + 1] * t
            i = np.clip(np.searchsorted(iy, np.arange(len(ys)), side="right") - 1, 0, len(iy) - 2)
            u = ((np.arange(len(ys)) - iy[i]) / (iy[i + 1] - iy[i])).astype(np.float32)[:, None]
            return (rows[i] * (1 - u) + rows[i + 1] * u).astype(np.float32)

        return (up(src[:, 0].reshape(X.shape).astype(np.float32)),
                up(src[:, 1].reshape(X.shape).astype(np.float32)))

    def warp(self, img, region=None, scale: float = 1.0, border=(255, 255, 255),
             interp=cv2.INTER_LINEAR):
        mx, my = self.maps(region, scale)
        return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def valid_mask(self, img_shape) -> np.ndarray:
        ones = np.full(img_shape[:2], 255, np.uint8)
        return self.warp(ones, border=0, interp=cv2.INTER_NEAREST) > 0


@dataclass
class Alignment:
    ok: bool
    H: Optional[np.ndarray]     # 3x3, maps INPUT pixel coords -> TEMPLATE pixel coords
    inliers: int
    inlier_ratio: float
    message: str = ""
    layout_score: float = 0.0
    transform: Optional[PageTransform] = None
    local_shift_px: float = 0.0     # largest local correction applied


@dataclass
class Features:
    scale: float
    keypoints: list
    descriptors: Optional[np.ndarray]


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def extract_features(img_bgr: np.ndarray, s: Settings) -> Features:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, s.align_max_dim / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if s.align_blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), s.align_blur)
    gray = _CLAHE.apply(gray)
    sift = cv2.SIFT_create(nfeatures=s.sift_features)
    kp, des = sift.detectAndCompute(gray, None)
    return Features(scale, kp, des)


def _quad_is_sane(H: np.ndarray, in_shape, tpl_shape) -> bool:
    """The template's corners, mapped back into the input, must form a convex quad
    of plausible size; otherwise RANSAC latched onto garbage."""
    th, tw = tpl_shape[:2]
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, Hinv).reshape(-1, 2)
    if not cv2.isContourConvex(q.astype(np.float32)):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    ih, iw = in_shape[:2]
    return 0.02 * ih * iw < area < 25 * ih * iw


def is_crisp(bgr: np.ndarray) -> bool:
    """True for a clean digital render: almost every pixel is near-white or near-black."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (g > 235).mean() > 0.75          # photos/scans: paper is rarely pure white


def line_masks(bgr: np.ndarray, crisp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Long horizontal / vertical strokes = the printed table grid (handwriting rarely
    produces long straight lines).

    crisp=True for clean digital renders (blank PDF templates): their rules can be a
    single pixel wide, which the median filter used against photo noise would erase."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if crisp:
        th = ((g < 170).astype(np.uint8) * 255)
    else:
        g = cv2.medianBlur(g, 5)
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    w = bgr.shape[1]
    hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 30), 1)))
    vl = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, w // 50))))
    return hl > 0, vl > 0


def layout_check(tpl_lines, warped: np.ndarray, valid: np.ndarray, s: Settings, tol: int = 6
                 ) -> tuple[float, int, int, int]:
    """Compare table rules (not pixels) between template and aligned page.

    Returns (recall, n_template_rules_checked, n_matched, n_extra):
      recall  - share of the template's rules found on the page;
      n_extra - long rules on the page that the template does not have (extra / moved
                rows => a different version of the form).
    Only the template's table area away from the photo border, and not under coloured
    overlay boxes, is compared."""
    from .ink import overlay_mask
    il = line_masks(warped, crisp=is_crisp(warped))
    allt = tpl_lines[0] | tpl_lines[1]
    ys, xs = np.where(allt)
    if not len(ys):
        return 0.0, 0, 0, 0
    area = np.zeros_like(valid)
    m = 15
    area[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1] = True
    vis = (cv2.erode(valid.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & area
    if s.mask_overlays:
        vis &= ~overlay_mask(warped, s)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    checked = matched = extra = 0
    for horiz, t, i in ((True, tpl_lines[0], il[0]), (False, tpl_lines[1], il[1])):
        tD = cv2.dilate(t.astype(np.uint8), k) > 0
        iD = cv2.dilate(i.astype(np.uint8), k) > 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(t.astype(np.uint8))
        for j in range(1, n):
            if max(st[j][2], st[j][3]) < 150:
                continue
            px = (lab == j) & vis
            if px.sum() < 0.3 * st[j][4]:
                continue                       # mostly hidden: no evidence either way
            checked += 1
            matched += bool(iD[px].mean() >= 0.6)
        n, lab, st, _ = cv2.connectedComponentsWithStats((i & vis).astype(np.uint8))
        for j in range(1, n):
            if (st[j][2] if horiz else st[j][3]) < 150:
                continue
            if tD[lab == j].mean() < 0.5:
                extra += 1
    return (matched / checked if checked else 0.0), checked, matched, extra


def _line_offsets(tpl_mask, img_mask, horizontal: bool, radius: int, step: int = 60):
    """For points along each template rule, find the offset (perpendicular to the rule)
    of the nearest matching rule in the image. Returns arrays x, y, d."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(tpl_mask.astype(np.uint8), connectivity=8)
    H_, W_ = tpl_mask.shape
    xs, ys, ds = [], [], []
    half = 10
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        length = w if horizontal else h
        if length < 150:
            continue
        for t in range((x if horizontal else y) + 20, (x + w if horizontal else y + h) - 20, step):
            if horizontal:
                rows = np.where(lab[:, t] == i)[0]
                if not len(rows):
                    continue
                c = rows.mean()
                lo, hi = int(c) - radius, int(c) + radius + 1
                if lo < 0 or hi > H_ or t - half < 0 or t + half >= W_:
                    continue
                prof = img_mask[lo:hi, t - half:t + half + 1].sum(1).astype(np.float64)
            else:
                cols = np.where(lab[t, :] == i)[0]
                if not len(cols):
                    continue
                c = cols.mean()
                lo, hi = int(c) - radius, int(c) + radius + 1
                if lo < 0 or hi > W_ or t - half < 0 or t + half >= H_:
                    continue
                prof = img_mask[t - half:t + half + 1, lo:hi].sum(0).astype(np.float64)
            if prof.max() < half:                     # no clear rule here (occluded, faded)
                continue
            peaks = np.where(prof >= 0.6 * prof.max())[0]
            # several separated peaks -> ambiguous (another rule nearby); skip
            if peaks.max() - peaks.min() > 6:
                continue
            off = float((peaks * prof[peaks]).sum() / prof[peaks].sum()) + lo - c
            xs.append(t if horizontal else c)
            ys.append(c if horizontal else t)
            ds.append(off)
    return np.array(xs), np.array(ys), np.array(ds)


def _robust_fit(B, d, iters=4):
    keep = np.ones(len(d), bool)
    coef = None
    for _ in range(iters):
        if keep.sum() < 12:
            return None, keep
        coef, *_ = np.linalg.lstsq(B[keep], d[keep], rcond=None)
        r = np.abs(B @ coef - d)
        mad = np.median(r[keep]) + 1e-6
        new = r < max(2.5, 3.5 * mad)
        if (new == keep).all():
            break
        keep = new
    return coef, keep


def local_refine(H: np.ndarray, img_bgr: np.ndarray, tpl_lines, tpl_shape, s: Settings
                 ) -> tuple[PageTransform, float]:
    """Fit the smooth correction field by snapping the image's table rules onto the
    template's. Returns (transform, max_abs_correction_px)."""
    base = PageTransform(H, tpl_shape)
    if not s.local_refine:
        return base, 0.0
    warped = base.warp(img_bgr)
    iH, iV = line_masks(warped, crisp=is_crisp(warped))
    tH, tV = tpl_lines
    r = s.local_refine_radius
    hx, hy, hd = _line_offsets(tH, iH, True, r)
    vx, vy, vd = _line_offsets(tV, iV, False, r)
    if len(hd) < 15 or len(vd) < 15:
        return base, 0.0
    allx, ally = np.concatenate([hx, vx]), np.concatenate([hy, vy])
    bounds = (float(allx.min()), float(ally.min()), float(allx.max()), float(ally.max()))
    probe = PageTransform(H, tpl_shape, np.zeros(10), np.zeros(10), bounds)
    cy, keep_y = _robust_fit(probe._basis(hx, hy), hd)
    cx, keep_x = _robust_fit(probe._basis(vx, vy), vd)
    if cx is None or cy is None:
        return base, 0.0
    t = PageTransform(H, tpl_shape, cx, cy, bounds)
    # sanity: corrections must be modest and must actually reduce the residuals
    gx, gy = np.meshgrid(np.linspace(bounds[0], bounds[2], 15), np.linspace(bounds[1], bounds[3], 15))
    dx, dy = t.offsets(gx.ravel(), gy.ravel())
    mx = float(max(np.abs(dx).max(), np.abs(dy).max()))
    before = np.median(np.abs(np.concatenate([hd[keep_y], vd[keep_x]])))
    after = np.median(np.abs(np.concatenate([
        hd[keep_y] - probe._basis(hx, hy)[keep_y] @ cy,
        vd[keep_x] - probe._basis(vx, vy)[keep_x] @ cx])))
    if mx > s.local_refine_max_px or after > before:
        return base, 0.0
    return t, mx


class TemplateMatcher:
    def __init__(self, template_bgr: np.ndarray, s: Settings):
        self.s = s
        self.template = template_bgr
        self.shape = template_bgr.shape[:2]
        self.feat = extract_features(template_bgr, s)
        self._gray_small = None
        self.lines = line_masks(template_bgr, crisp=is_crisp(template_bgr))

    def quick_inliers(self, feat: Features) -> int:
        """Feature matches consistent with one homography - a cheap 'is this the right
        page?' score used to decide which template page to align against first."""
        if feat.descriptors is None or self.feat.descriptors is None or len(feat.keypoints) < 10:
            return 0
        cv2.setRNGSeed(12345)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        pairs = matcher.knnMatch(feat.descriptors, self.feat.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.s.ratio_test * p[1].distance]
        if len(good) < 10:
            return len(good)
        src = np.float32([feat.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.feat.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.s.ransac_reproj_px)
        return int(mask.sum()) if mask is not None else 0

    def align(self, img_bgr: np.ndarray, feat: Optional[Features] = None) -> Alignment:
        s = self.s
        feat = feat or extract_features(img_bgr, s)
        if feat.descriptors is None or self.feat.descriptors is None or len(feat.keypoints) < 10:
            return Alignment(False, None, 0, 0.0, "too few keypoints")

        cv2.setRNGSeed(12345)          # FLANN + RANSAC are randomised: make audits repeatable
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        pairs = matcher.knnMatch(feat.descriptors, self.feat.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < s.ratio_test * p[1].distance]
        if len(good) < s.min_inliers:
            return Alignment(False, None, len(good), 0.0, f"only {len(good)} feature matches")

        src = np.float32([feat.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.feat.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, s.ransac_reproj_px)
        if Hs is None:
            return Alignment(False, None, 0, 0.0, "homography failed")
        inliers = int(mask.sum())
        ratio = inliers / len(good)

        Si = np.diag([feat.scale, feat.scale, 1.0])
        St = np.diag([self.feat.scale, self.feat.scale, 1.0])
        H = np.linalg.inv(St) @ Hs @ Si
        H /= H[2, 2]

        if inliers < s.min_inliers:
            return Alignment(False, H, inliers, ratio, f"only {inliers} RANSAC inliers")
        if not _quad_is_sane(H, img_bgr.shape, self.shape):
            return Alignment(False, H, inliers, ratio, "implausible geometry")

        msg = "sift"
        if s.ecc_refine:
            H, refined = self._ecc_refine(img_bgr, H)
            msg += "+ecc" if refined else ""

        transform, shift = local_refine(H, img_bgr, self.lines, self.shape, s)
        if transform.has_local:
            msg += f"+lines({shift:.0f}px)"
        warped = transform.warp(img_bgr)
        valid = transform.valid_mask(img_bgr.shape)
        recall, checked, matched, extra = layout_check(self.lines, warped, valid, s)
        score = round(recall - 0.05 * extra, 3)
        lay = f"rules {matched}/{checked}, {extra} extra"
        if recall < s.layout_min_recall or extra > s.layout_max_extra:
            return Alignment(False, H, inliers, ratio,
                             f"layout mismatch ({lay}) - different form or version?",
                             score, transform, shift)
        return Alignment(True, H, inliers, ratio, f"{msg}, {lay}", score, transform, shift)

    # ------------------------------------------------------------------ #
    def _ecc_refine(self, img_bgr: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, bool]:
        th, tw = self.shape
        k = min(1.0, 1200 / max(th, tw))
        if self._gray_small is None:
            g = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self._gray_small = cv2.GaussianBlur(cv2.resize(g, None, fx=k, fy=k), (5, 5), 0)
        warped = cv2.warpPerspective(img_bgr, H, (tw, th), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        w = cv2.GaussianBlur(cv2.resize(w, None, fx=k, fy=k), (5, 5), 0)
        W = np.eye(3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        try:
            cc, W = cv2.findTransformECC(self._gray_small, w, W, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error:
            return H, False
        K = np.diag([k, k, 1.0])
        Wf = np.linalg.inv(K) @ W.astype(np.float64) @ K   # template px -> warped px
        # Refinement must be a small nudge; reject anything large.
        if abs(Wf[0, 2]) > 0.02 * tw or abs(Wf[1, 2]) > 0.02 * th or cc < 0.2:
            return H, False
        if np.abs(Wf[:2, :2] - np.eye(2)).max() > 0.03:
            return H, False
        H2 = np.linalg.inv(Wf) @ H
        return H2 / H2[2, 2], True
