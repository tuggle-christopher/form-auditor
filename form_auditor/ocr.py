"""Reading handwriting: pluggable engines.

  none        - don't transcribe (presence checks only)
  azure       - Azure AI Document Intelligence "prebuilt-layout", PAGE mode: the
                straightened pages are sent (2 per request - the free tier's limit) and
                every word comes back with its position, so writing is placed on the form
                exactly (in the box, running out of it, or beside it) and combined with the
                ink check. Needs AZURE_DI_ENDPOINT and AZURE_DI_KEY.
  azure-read  - Azure "prebuilt-read" on crops of the filled-in boxes only (one request
                per page, all crops on one sheet). Cheaper per page, no position fusion.
  claude      - Claude vision via the Anthropic Messages API. Needs ANTHROPIC_API_KEY.
  tesseract   - local Tesseract OCR. Offline; weak on handwriting.
  mock        - canned answers from a JSON file (tests / dry runs).
  mock-layout - canned page-mode words from a JSON file (tests).

Crop engines return OcrResult(text, confidence 0..1) per box; page engines return a
layout.PageText (words with positions) per page.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

FORMAT_HINT = {
    "id": "an identifier: letters, digits and symbols such as # / - (may be long), or NA",
    "text": "free handwritten text (details / follow-up actions); may span several lines",
    "date": "a date written in any style, e.g. 14 Oct 26, 14-Oct-26, 14-10-26",
    "time": "a time, expected as HH:MM",
    "number": "a number, possibly with a decimal point, e.g. 15.8",
}


@dataclass
class OcrRequest:
    key: str
    image: np.ndarray          # BGR crop
    fmt: str                   # id | text | date | time | number
    hint: str = ""             # field label, e.g. "Reference number"


@dataclass
class OcrResult:
    text: str
    confidence: float
    engine: str
    illegible: bool = False
    error: str = ""


class ReaderHttpError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class FatalOcrError(RuntimeError):
    """Configuration problem (bad API key, unknown model): stop the run, don't flood the
    report with REVIEWs."""


class OcrEngine:
    name = "none"
    page_level = False          # True: analyse_pages() on whole pages instead of read()

    def read(self, requests: list[OcrRequest], where: dict) -> dict[str, OcrResult]:
        return {}


# --------------------------------------------------------------------------- #
class MockOcr(OcrEngine):
    """JSON: {"<file name>": {"<page>": {"<key>": "text"}}}  or a flat {"<key>": "text"}."""
    name = "mock"

    def __init__(self, path: str | Path):
        self.data = json.loads(Path(path).read_text())

    def read(self, requests, where):
        d = self.data.get(where.get("file", ""), {})
        d = d.get(str(where.get("page", 1)), d) if isinstance(d, dict) else {}
        out = {}
        for r in requests:
            v = d.get(r.key, self.data.get(r.key))
            if isinstance(v, str):
                out[r.key] = OcrResult(v, 0.95, self.name, illegible=(v == ""))
        return out


# --------------------------------------------------------------------------- #
class TesseractOcr(OcrEngine):
    name = "tesseract"
    WHITELIST = {
        "id": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#/-.",
        "date": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/. ",
        "time": "0123456789:.",
        "number": "0123456789.-",
    }

    def __init__(self, lang: str = "eng"):
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract is not installed (apt install tesseract-ocr)")
        self.lang = lang

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        """Binarise pen strokes against the local paper colour (robust to shading and
        screen moiré), drop printed rules, crop and scale to ~50 px text height."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        k = max(31, (min(g.shape[:2]) // 2) | 1)
        bg = cv2.medianBlur(g, min(k, 255)).astype(np.float32)
        d = bg - cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2.0)
        th = ((d > 40) * 255).astype(np.uint8)
        w = th.shape[1]
        lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 4), 1)))
        lines |= cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, th.shape[0] // 2))))
        th[cv2.dilate(lines, np.ones((5, 5), np.uint8)) > 0] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 25          # specks
        th = (keep[lab] * 255).astype(np.uint8)
        ys, xs = np.nonzero(th)
        if len(ys):
            th = th[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]
        h = th.shape[0]
        k = 50 / max(h, 1) if (h < 35 or h > 90) else 1.0
        if k != 1.0:
            th = cv2.resize(th, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        out = 255 - th
        return cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    def read(self, requests, where):
        out = {}
        for r in requests:
            img = self._prep(r.image)
            psm = "6" if r.fmt == "text" else "7"
            cfg = ["--psm", psm, "--oem", "1"]
            if r.fmt in self.WHITELIST:
                cfg += ["-c", f"tessedit_char_whitelist={self.WHITELIST[r.fmt]}"]
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "c.png"
                cv2.imwrite(str(p), img)
                try:
                    res = subprocess.run(["tesseract", str(p), "stdout", "-l", self.lang, *cfg, "tsv"],
                                         capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired:
                    out[r.key] = OcrResult("", 0.0, self.name, error="timeout")
                    continue
            words, confs, last_line = [], [], None
            for line in res.stdout.splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) < 12 or not parts[11].strip():
                    continue
                key = (parts[2], parts[3], parts[4])
                if last_line is not None and key != last_line and r.fmt == "text":
                    words.append("\n")
                last_line = key
                words.append(parts[11])
                confs.append(float(parts[10]))
            text = re.sub(r" ?\n ?", "\n", " ".join(words)).strip()
            conf = float(np.mean(confs)) / 100 if confs else 0.0
            out[r.key] = OcrResult(text, round(conf, 2), self.name, illegible=not text)
        return out


# --------------------------------------------------------------------------- #
class ClaudeOcr(OcrEngine):
    """Claude vision through the Anthropic Messages API (plain HTTPS, no SDK needed)."""
    name = "claude"
    URL = "https://api.anthropic.com/v1/messages"
    CONF = {"high": 0.95, "medium": 0.7, "low": 0.35}

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: Optional[str] = None,
                 max_images: int = 20, timeout: int = 120):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use --ocr claude")
        self.model = model
        self.max_images = max_images
        self.timeout = timeout

    @staticmethod
    def _png_b64(img: np.ndarray, max_side: int = 1200) -> str:
        h, w = img.shape[:2]
        k = min(1.0, max_side / max(h, w))
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf.tobytes()).decode()

    def _prompt(self, reqs: list[OcrRequest]) -> list[dict]:
        content = [{"type": "text", "text": (
            "You are transcribing handwriting from crops of a scanned paper checklist "
            "form. For each image, transcribe exactly what is "
            "handwritten - do not correct spelling, do not complete, reformat or guess "
            "missing characters. Keep symbols such as # / - . : exactly. Ignore printed "
            "form text and table lines. If a word is unreadable write [?] in its place; "
            "if nothing is legible return an empty string. A written 'NA' or 'N/A' must be "
            "returned as written.\n\nReturn ONLY a JSON object mapping each key to "
            '{"text": "...", "confidence": "high"|"medium"|"low"}.')}]
        for r in reqs:
            content.append({"type": "text",
                            "text": f"key: {r.key}\nfield: {r.hint}\nexpected: {FORMAT_HINT.get(r.fmt, r.fmt)}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": self._png_b64(r.image)}})
        return content

    def _call(self, content: list[dict]) -> str:
        body = json.dumps({"model": self.model, "max_tokens": 4000, "temperature": 0,
                           "messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        delay = 2.0
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403, 404):
                    raise FatalOcrError(f"Anthropic API error {e.code} - check ANTHROPIC_API_KEY "
                                        f"and --ocr-model ({self.model}): {msg}") from None
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Anthropic API error {e.code}: {msg}") from None
            except urllib.error.URLError as e:
                if attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"cannot reach Anthropic API: {e.reason}") from None
        raise RuntimeError("Anthropic API: too many retries")

    @staticmethod
    def _parse(text: str) -> dict:
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        m = re.search(r"\{.*\}", t, re.S)
        return json.loads(m.group(0) if m else t)

    def read(self, requests, where):
        out = {}
        for i in range(0, len(requests), self.max_images):
            chunk = requests[i:i + self.max_images]
            try:
                data = self._parse(self._call(self._prompt(chunk)))
            except FatalOcrError:
                raise
            except Exception as e:                      # keep auditing without values
                for r in chunk:
                    out[r.key] = OcrResult("", 0.0, self.name, error=str(e)[:200])
                continue
            for r in chunk:
                v = data.get(r.key)
                if not isinstance(v, dict):
                    out[r.key] = OcrResult("", 0.0, self.name, error="missing from response")
                    continue
                text = str(v.get("text", "")).strip()
                conf = self.CONF.get(str(v.get("confidence", "low")).lower(), 0.35)
                if "[?]" in text:
                    conf = min(conf, 0.5)
                out[r.key] = OcrResult(text, conf, self.name, illegible=not text)
        return out


# --------------------------------------------------------------------------- #
class _AzureClient:
    """REST client for Azure AI Document Intelligence (no SDK needed). Needs
    AZURE_DI_ENDPOINT (e.g. https://<resource>.cognitiveservices.azure.com) and
    AZURE_DI_KEY. Documents go only to your own Azure resource.

    Free tier (F0) limits are respected: at most 1 analyse request and 1 result poll per
    second; 429 / 5xx answers are retried with back-off."""

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None,
                 api_version: str = "2024-11-30", timeout: int = 180):
        self.endpoint = (endpoint or os.environ.get("AZURE_DI_ENDPOINT")
                         or os.environ.get("DOCUMENTINTELLIGENCE_ENDPOINT") or "").rstrip("/")
        self.key = (key or os.environ.get("AZURE_DI_KEY")
                    or os.environ.get("DOCUMENTINTELLIGENCE_API_KEY"))
        if not self.endpoint or not self.key:
            raise RuntimeError("Set AZURE_DI_ENDPOINT and AZURE_DI_KEY (e.g. in .env) to use "
                               "the Azure Document Intelligence reader")
        self.api_version = api_version
        self.timeout = timeout
        self._last_post = 0.0

    def model_url(self, model: str) -> str:
        # v4.0 (2024-11-30) lives under /documentintelligence, older versions under /formrecognizer
        base = "documentintelligence" if self.api_version >= "2023-10-31" else "formrecognizer"
        return f"{self.endpoint}/{base}/documentModels/{model}:analyze?api-version={self.api_version}"

    # ---- HTTP --------------------------------------------------------------
    def _request(self, req: urllib.request.Request):
        delay = 2.0
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.status, dict(resp.headers), resp.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403, 404):
                    raise FatalOcrError(
                        f"Azure Document Intelligence error {e.code} - check AZURE_DI_ENDPOINT, "
                        f"AZURE_DI_KEY and --azure-api-version ({self.api_version}); on the free "
                        f"tier this also happens when the monthly page allowance is used up: "
                        f"{body}") from None
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    time.sleep(float(ra) if ra and ra.isdigit() else delay)
                    delay *= 2
                    continue
                raise ReaderHttpError(e.code, f"Azure Document Intelligence error {e.code}: {body}") from None
            except urllib.error.URLError as e:
                if attempt < 5:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"cannot reach Azure Document Intelligence: {e.reason}") from None
        raise RuntimeError("Azure Document Intelligence: too many retries")

    def _analyze(self, data: bytes, url: str,
                 content_type: str = "application/octet-stream") -> dict:
        hdr = {"Ocp-Apim-Subscription-Key": self.key}
        wait_post = 1.05 - (time.time() - self._last_post)     # F0: 1 analyse request / s
        if wait_post > 0:
            time.sleep(wait_post)
        self._last_post = time.time()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={**hdr, "Content-Type": content_type})
        status, headers, _ = self._request(req)
        op = next((v for k, v in headers.items() if k.lower() == "operation-location"), None)
        if status != 202 or not op:
            raise RuntimeError(f"unexpected analyze response {status} (no Operation-Location)")
        t0 = time.time()
        wait = 1.5
        while time.time() - t0 < self.timeout:
            time.sleep(wait)
            _, h2, body = self._request(urllib.request.Request(op, headers=hdr))
            data = json.loads(body)
            st = str(data.get("status", "")).lower()
            if st == "succeeded":
                return data.get("analyzeResult") or {}
            if st in ("failed", "canceled"):
                raise RuntimeError(f"analysis {st}: {data.get('error')}")
            ra = next((v for k, v in h2.items() if k.lower() == "retry-after"), None)
            # F0 allows 1 result poll per second
            wait = max(1.05, min(float(ra), 5.0)) if ra and str(ra).replace(".", "", 1).isdigit() else 1.05
        raise RuntimeError("Azure Document Intelligence: timed out waiting for the result")



class AzureDocIntelOcr(_AzureClient, OcrEngine):
    """Azure "prebuilt-read" on crops (--ocr azure-read).

    All crops of a page are pasted onto one white sheet with gaps between them and sent
    as ONE request; each returned line is mapped back to its field by position."""
    name = "azure-read"
    GAP = 60                    # px of white between crops on the sheet
    MAX_H = 9500                # Read model limit is 10,000 px per side
    MAX_W = 4000

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None,
                 api_version: str = "2024-11-30", timeout: int = 180):
        super().__init__(endpoint, key, api_version, timeout)
        self.read_url = self.model_url("prebuilt-read")

    # ---- sheet building ----------------------------------------------------
    def _sheets(self, requests: list[OcrRequest]):
        """Yield (png_bytes, width, [(key, y0, y1)]) sheets of stacked crops."""
        rows, cur, h = [], [], 0
        for r in requests:
            img = r.image if r.image.ndim == 3 else cv2.cvtColor(r.image, cv2.COLOR_GRAY2BGR)
            if img.shape[1] > self.MAX_W:
                k = self.MAX_W / img.shape[1]
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            if img.shape[0] < 50:                       # service minimum is 50 px
                img = cv2.copyMakeBorder(img, 0, 50 - img.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if cur and h + img.shape[0] + self.GAP > self.MAX_H:
                rows.append(cur)
                cur, h = [], 0
            cur.append((r.key, img))
            h += img.shape[0] + self.GAP
        if cur:
            rows.append(cur)
        for row in rows:
            W = max(max(i.shape[1] for _, i in row), 50) + 2 * self.GAP
            H = sum(i.shape[0] + self.GAP for _, i in row) + self.GAP
            sheet = np.full((H, W, 3), 255, np.uint8)
            spans, y = [], self.GAP
            for key, img in row:
                sheet[y:y + img.shape[0], self.GAP:self.GAP + img.shape[1]] = img
                spans.append((key, y, y + img.shape[0]))
                y += img.shape[0] + self.GAP
            ok, buf = cv2.imencode(".png", sheet)
            yield buf.tobytes(), W, spans

    @staticmethod
    def _centre(poly, scale):
        xs, ys = poly[0::2], poly[1::2]
        return (sum(xs) / len(xs)) * scale, (sum(ys) / len(ys)) * scale

    def read(self, requests, where):
        out = {}
        for png, width, spans in self._sheets(requests):
            try:
                res = self._analyze(png, self.read_url)
            except FatalOcrError:
                raise
            except Exception as e:                      # keep auditing without values
                for key, _, _ in spans:
                    out[key] = OcrResult("", 0.0, self.name, error=str(e)[:200])
                continue
            pages = res.get("pages") or [{}]
            page = pages[0]
            # polygons are in pixels for images; convert defensively if not
            scale = 1.0
            if page.get("unit") not in (None, "pixel") and page.get("width"):
                scale = width / float(page["width"])

            def which(y):
                for key, y0, y1 in spans:
                    if y0 - self.GAP / 2 <= y < y1 + self.GAP / 2:
                        return key
                return None

            lines = {k: [] for k, _, _ in spans}
            for ln in page.get("lines") or []:
                if not ln.get("polygon"):
                    continue
                cx, cy = self._centre(ln["polygon"], scale)
                k = which(cy)
                if k:
                    lines[k].append((cy, cx, ln.get("content", "")))
            confs = {k: [] for k, _, _ in spans}
            for w in page.get("words") or []:
                if not w.get("polygon"):
                    continue
                _, cy = self._centre(w["polygon"], scale)
                k = which(cy)
                if k and w.get("confidence") is not None:
                    confs[k].append(float(w["confidence"]))
            fmt = {r.key: r.fmt for r in requests}
            for key, _, _ in spans:
                ls = sorted(lines[key])
                # group into visual lines (same height band) and read left to right
                text_lines, band = [], []
                for cy, cx, t in ls:
                    if band and abs(cy - band[-1][0]) > 15:
                        text_lines.append(" ".join(x[2] for x in sorted(band, key=lambda b: b[1])))
                        band = []
                    band.append((cy, cx, t))
                if band:
                    text_lines.append(" ".join(x[2] for x in sorted(band, key=lambda b: b[1])))
                sep = "\n" if fmt.get(key) == "text" else " "
                text = sep.join(t.strip() for t in text_lines if t.strip())
                conf = float(np.mean(confs[key])) if confs[key] else (0.0 if not text else 0.5)
                out[key] = OcrResult(text, round(conf, 3), self.name, illegible=not text)
        return out


# --------------------------------------------------------------------------- #
@dataclass
class _Tile:
    page: int                 # index of the input page
    sheet: int                # index of the sheet (= page of the request)
    x: int                    # top-left corner on the sheet (sheet px, before any shrinking)
    y: int
    w: int
    h: int


def plan_sheets(sizes: list[tuple[int, int]], n_sheets: int, max_px: int = 9900, gap: int = 120
                ) -> tuple[list[tuple[int, int]], list[_Tile]]:
    """Tile pages (w, h) onto at most `n_sheets` sheets, in order, spread evenly (3 pages on
    2 sheets: 2 + 1). Pages on a sheet go side by side, wrapping into rows so that no sheet
    side exceeds `max_px` where possible. Returns (sheet sizes, tiles)."""
    n = len(sizes)
    n_sheets = max(1, min(n_sheets, n))
    per = [n // n_sheets + (1 if i < n % n_sheets else 0) for i in range(n_sheets)]
    tiles, sheet_sizes, k = [], [], 0
    for si, cnt in enumerate(per):
        idx = list(range(k, k + cnt))
        k += cnt
        pw = max(sizes[i][0] for i in idx)
        cols = max(1, min(cnt, (max_px + gap) // (pw + gap)))
        x = y = row_h = sw = 0
        for j, i in enumerate(idx):
            if j and j % cols == 0:
                x, y, row_h = 0, y + row_h + gap, 0
            w, h = sizes[i]
            tiles.append(_Tile(i, si, x, y, w, h))
            sw = max(sw, x + w)
            row_h = max(row_h, h)
            x += w + gap
        sheet_sizes.append((sw, y + row_h))
    return sheet_sizes, tiles


class AzureLayoutOcr(_AzureClient, OcrEngine):
    """Azure "prebuilt-layout" on whole pages (--ocr azure).

    The free tier analyses only the first 2 pages of each request. To read a whole form in
    ONE request, the straightened pages are tiled onto at most `sheets` images (default 2:
    a 3-page form goes as page 1 + 2 side by side, and page 3) and sent as one multi-page
    TIFF; each word that comes back is mapped to the form page it lies on. Sheets stay
    within the service's 10,000 px limit, and the request is compressed to stay under
    `max_mb` (4 MB on the free tier). If the service rejects the TIFF, the same sheets are
    sent as a PDF instead (for the rest of the run).

    Each page comes back as a layout.PageText: every word with its polygon (in that page's
    image px), its text line, whether it is handwritten, plus the selection marks."""
    name = "azure"
    page_level = True
    MAX_PX = 9900             # service limit is 10,000 px per side

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None,
                 api_version: str = "2024-11-30", model: str = "prebuilt-layout",
                 timeout: int = 180):
        super().__init__(endpoint, key, api_version, timeout)
        self.model = model
        self.layout_url = self.model_url(model)
        self.requests_sent = 0
        self.format = "tiff"

    # ---- packing -------------------------------------------------------------
    @classmethod
    def _sheets(cls, images: list[np.ndarray], n_sheets: int):
        sizes, tiles = plan_sheets([(im.shape[1], im.shape[0]) for im in images], n_sheets,
                                   cls.MAX_PX)
        sheets = [np.full((h, w, 3), 255, np.uint8) for w, h in sizes]
        for t in tiles:
            sheets[t.sheet][t.y:t.y + t.h, t.x:t.x + t.w] = images[t.page]
        # a sheet that could not be kept under the limit (very many pages) is shrunk
        shrink = [min(1.0, cls.MAX_PX / max(sh.shape[:2])) for sh in sheets]
        sheets = [sh if f == 1.0 else cv2.resize(sh, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                  for sh, f in zip(sheets, shrink)]
        return sheets, tiles, shrink

    @staticmethod
    def _encode(sheets: list[np.ndarray], fmt: str, max_mb: float) -> tuple[bytes, float]:
        """Sheets -> one multi-page greyscale TIFF (JPEG-compressed) or PDF under max_mb.
        Returns (bytes, extra scale applied to fit)."""
        import io
        from PIL import Image
        scale, quality = 1.0, 88
        while True:
            pil = []
            for sh in sheets:
                if scale < 1.0:
                    sh = cv2.resize(sh, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                # greyscale: the reader needs no colour, and it is ~1/3 of the size, so
                # whole forms fit the free tier's 4 MB at full resolution
                pil.append(Image.fromarray(cv2.cvtColor(sh, cv2.COLOR_BGR2GRAY)))
            buf = io.BytesIO()
            if fmt == "tiff":
                pil[0].save(buf, "TIFF", save_all=True, append_images=pil[1:],
                            compression="jpeg", quality=quality)
            else:
                # PDF pages are kept within 17 in (an older documented limit)
                dpi = max(200.0, max(max(p.size) for p in pil) / 17.0)
                pil[0].save(buf, "PDF", save_all=True, append_images=pil[1:], resolution=dpi,
                            quality=quality)
            data = buf.getvalue()
            if len(data) <= max_mb * 1024 * 1024 * 0.97 or scale < 0.5:
                return data, scale
            if quality > 60:
                quality -= 14
            else:
                scale *= 0.85

    # ---- reading the answer ----------------------------------------------------
    @staticmethod
    def parse(result: dict, sizes: list[tuple[int, int]]) -> list:
        """analyzeResult -> one PageText per page (sheet) sent, polygons in its px."""
        from .layout import PageMark, PageText, PageWord
        content_hw = []                                      # handwritten spans
        for st in result.get("styles") or []:
            if st.get("isHandwritten"):
                content_hw += [(sp["offset"], sp["offset"] + sp["length"]) for sp in st.get("spans") or []]

        def hw(span):
            if not span:
                return None
            o = span.get("offset", -1)
            return any(a <= o < b for a, b in content_hw)

        out = []
        pages = result.get("pages") or []
        for i, (w_px, h_px) in enumerate(sizes):
            if i >= len(pages):
                out.append(PageText(engine="azure", error=(
                    f"the service analysed only {len(pages)} of {len(sizes)} images of the "
                    f"request (free tier: 2 per request - set azure_pages_per_request=2)")))
                continue
            pg = pages[i]
            # images: pixels; PDFs: inches - scale to the image that was sent
            kx = w_px / float(pg.get("width") or w_px)
            ky = h_px / float(pg.get("height") or h_px)

            def poly(p):
                a = np.asarray(p or [], np.float64).reshape(-1, 2)
                return a * np.array([kx, ky]) if len(a) else a

            line_spans = [[(sp["offset"], sp["offset"] + sp["length"]) for sp in ln.get("spans") or []]
                          for ln in pg.get("lines") or []]

            def line_of(span):
                if not span:
                    return -1
                o = span.get("offset", -1)
                for li, spans in enumerate(line_spans):
                    if any(a <= o < b for a, b in spans):
                        return li
                return -1

            words = [PageWord(w.get("content", ""), float(w.get("confidence", 0.0)),
                              poly(w.get("polygon")), line_of(w.get("span")), hw(w.get("span")))
                     for w in pg.get("words") or [] if w.get("polygon")]
            marks = [PageMark(str(m.get("state", "")).lower(), float(m.get("confidence", 0.0)),
                              poly(m.get("polygon")))
                     for m in pg.get("selectionMarks") or [] if m.get("polygon")]
            out.append(PageText(words, marks, "azure"))
        return out

    @staticmethod
    def _untile(sheet_texts: list, tiles: list[_Tile], factors: list[float], n_pages: int) -> list:
        """Words / marks of each sheet -> the form page whose tile they lie on, in that
        page's own px."""
        from .layout import PageMark, PageText, PageWord
        out = [PageText(engine="azure") for _ in range(n_pages)]
        for t in tiles:
            st = sheet_texts[t.sheet]
            if st.error:
                out[t.page].error = st.error
                continue
            f = factors[t.sheet]
            off = np.array([t.x, t.y], np.float64)

            def mine(poly):
                c = poly.mean(axis=0) / f - off
                return 0 <= c[0] < t.w and 0 <= c[1] < t.h

            for w in st.words:
                if len(w.polygon) and mine(w.polygon):
                    out[t.page].words.append(PageWord(w.text, w.confidence, w.polygon / f - off,
                                                      t.sheet * 100000 + w.line if w.line >= 0 else -1,
                                                      w.handwritten))
            for m in st.marks:
                if len(m.polygon) and mine(m.polygon):
                    out[t.page].marks.append(PageMark(m.state, m.confidence, m.polygon / f - off))
        return out

    def analyse_pages(self, images: list[np.ndarray], wheres: list[dict], max_mb: float = 4.0,
                      sheets: int = 2):
        """All `images` in ONE request, tiled onto at most `sheets` images."""
        from .layout import PageText
        tiled, tiles, shrink = self._sheets(images, sheets)
        res = None
        for attempt in range(2):
            data, k = self._encode(tiled, self.format, max_mb)
            factors = [f * k for f in shrink]
            try:
                self.requests_sent += 1
                res = self._analyze(data, self.layout_url,
                                    "image/tiff" if self.format == "tiff" else "application/pdf")
                break
            except FatalOcrError:
                raise
            except ReaderHttpError as e:
                if e.code in (400, 415) and self.format == "tiff" and attempt == 0:
                    self.format = "pdf"                   # TIFF not accepted: use PDF from now on
                    continue
                return [PageText(engine="azure", error=str(e)[:200]) for _ in images]
            except Exception as e:                        # keep auditing without the reader
                return [PageText(engine="azure", error=str(e)[:200]) for _ in images]
        sizes = [(int(round(sh.shape[1] * k)), int(round(sh.shape[0] * k))) for sh in tiled]
        return self._untile(self.parse(res, sizes), tiles, factors, len(images))


class MockLayoutOcr(OcrEngine):
    """Page-mode stand-in for tests. JSON: {"<file>": {"<page>": {"words": [{"text",
    "conf", "box": [x0, y0, x1, y1] (template px), "line", "handwritten"}], "marks":
    [{"state", "conf", "box"}]}}}"""
    name = "mock-layout"
    page_level = True

    def __init__(self, path: str | Path, template_width: int = 1700):
        self.data = json.loads(Path(path).read_text())
        self.tw = template_width
        self.requests_sent = 0
        self.pages_per_request: list[int] = []

    @staticmethod
    def _poly(b, k):
        x0, y0, x1, y1 = (float(v) * k for v in b)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def analyse_pages(self, images, wheres, max_mb: float = 4.0, sheets: int = 2):
        from .layout import PageMark, PageText, PageWord
        self.requests_sent += 1
        self.pages_per_request.append(len(images))
        out = []
        for im, wh in zip(images, wheres):
            k = im.shape[1] / self.tw
            d = self.data.get(wh.get("file", ""), {}).get(str(wh.get("page", 1)), {})
            out.append(PageText(
                [PageWord(w["text"], float(w.get("conf", 0.95)), self._poly(w["box"], k),
                          int(w.get("line", -1)), w.get("handwritten", True))
                 for w in d.get("words", [])],
                [PageMark(m["state"], float(m.get("conf", 0.9)), self._poly(m["box"], k))
                 for m in d.get("marks", [])], self.name))
        return out



def make_engine(name: str, model: Optional[str] = None, mock_file: Optional[str] = None,
                azure_endpoint: Optional[str] = None, azure_api_version: str = "2024-11-30",
                azure_model: str = "prebuilt-layout") -> OcrEngine:
    name = (name or "none").lower()
    if name == "none":
        return OcrEngine()
    if name == "tesseract":
        return TesseractOcr()
    if name == "claude":
        return ClaudeOcr(model=model or "claude-sonnet-5-5")
    if name == "azure":
        return AzureLayoutOcr(endpoint=azure_endpoint, api_version=azure_api_version,
                              model=azure_model)
    if name == "azure-read":
        return AzureDocIntelOcr(endpoint=azure_endpoint, api_version=azure_api_version)
    if name == "mock-layout":
        if not mock_file:
            raise ValueError("--ocr mock-layout needs --ocr-mock-file")
        return MockLayoutOcr(mock_file)
    if name == "mock":
        if not mock_file:
            raise ValueError("--ocr mock needs --ocr-mock-file")
        return MockOcr(mock_file)
    raise ValueError(f"unknown OCR engine '{name}' "
                     f"(none | azure | azure-read | claude | tesseract | mock | mock-layout)")
