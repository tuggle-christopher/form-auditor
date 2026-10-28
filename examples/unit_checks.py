"""Quick checks of the plumbing: the .env loader, cloud-storage staging (with stand-ins
for Databricks `dbutils` and the azure-storage-blob client), the Azure Document
Intelligence client (against a local stand-in of the service) and template validation.
Needs a template in templates/ (form.yaml, built from your blank form).

    python examples/unit_checks.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from form_auditor import storage                       # noqa: E402
from form_auditor.api import run_audit                 # noqa: E402
from form_auditor.env import load_env                  # noqa: E402
from form_auditor.template import load_template        # noqa: E402

BLANK_PAGE = ROOT / "templates" / "form_p1.png"        # a blank page 1 of your template
FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


# --------------------------------------------------------------------------- #
class FakeDbutils:
    """dbutils.fs over a local folder: abfss://box@acct.dfs.core.windows.net/<p> -> root/<p>"""
    PREFIX = "abfss://box@acct.dfs.core.windows.net/"

    def __init__(self, root: Path):
        self.root = root
        self.fs = self

    def _p(self, uri: str) -> Path:
        if uri.startswith("file:"):
            return Path(uri[5:])
        assert uri.startswith(self.PREFIX), uri
        return self.root / uri[len(self.PREFIX):]

    def ls(self, uri):
        p = self._p(uri)
        if not p.exists():
            raise FileNotFoundError(uri)
        return [types.SimpleNamespace(path=uri.rstrip("/") + "/" + c.name + ("/" if c.is_dir() else ""),
                                      name=c.name + ("/" if c.is_dir() else ""))
                for c in sorted(p.iterdir())]

    def cp(self, src, dst, recurse=False):
        s, d = self._p(src), self._p(dst)
        if s.is_dir():
            assert recurse
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(s, d)

    def mkdirs(self, uri):
        self._p(uri).mkdir(parents=True, exist_ok=True)


def install_fake_blob_sdk(root: Path):
    """A stand-in for azure.storage.blob.ContainerClient storing blobs under root/<container>."""
    class Downloader:
        def __init__(self, p):
            self.p = p

        def readinto(self, fh):
            fh.write(self.p.read_bytes())

    class ContainerClient:
        def __init__(self, url, container, credential=None):
            self.dir = root / container
            self.credential = credential

        @classmethod
        def from_connection_string(cls, conn, container):
            assert "AccountName=" in conn
            return cls("", container)

        def list_blobs(self, name_starts_with=""):
            return [types.SimpleNamespace(name=p.relative_to(self.dir).as_posix())
                    for p in sorted(self.dir.rglob("*")) if p.is_file()
                    and p.relative_to(self.dir).as_posix().startswith(name_starts_with)]

        def download_blob(self, name):
            return Downloader(self.dir / name)

        def upload_blob(self, name, data, overwrite=False):
            p = self.dir / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data.read())

    blob = types.ModuleType("azure.storage.blob")
    blob.ContainerClient = ContainerClient
    for name, mod in (("azure", types.ModuleType("azure")),
                      ("azure.storage", types.ModuleType("azure.storage")),
                      ("azure.storage.blob", blob)):
        sys.modules[name] = mod


# --------------------------------------------------------------------------- #
def azure_layout_checks():
    """A stand-in for the free tier: analyses at most 2 images per request, answers the
    first request with 429 (too many requests), can refuse TIFF. It finds each page's
    black marker square in the images it receives by itself and reports a word there - so
    the tiling of pages onto sheets and the mapping back are checked independently."""
    import http.server
    import io
    import threading
    import time
    import numpy as np
    import cv2
    import pypdfium2 as pdfium
    from PIL import Image
    from form_auditor.ocr import AzureLayoutOcr

    log = {"posts": [], "gets": [], "frames": [], "types": [], "n": 0, "refuse_tiff": False}
    results = {}

    def frames_of(body, ctype):
        """-> [(gray image, unit, width, height)] as the service would see the pages."""
        if ctype == "image/tiff":
            im = Image.open(io.BytesIO(body))
            out = []
            for i in range(getattr(im, "n_frames", 1)):
                im.seek(i)
                g = np.array(im.convert("L"))
                out.append((g, "pixel", g.shape[1], g.shape[0]))
            return out
        doc = pdfium.PdfDocument(body)
        out = []
        for i in range(len(doc)):
            wpt, hpt = doc[i].get_size()
            g = np.array(doc[i].render(scale=2).to_pil().convert("L"))
            out.append((g, "inch", wpt / 72, hpt / 72))
        return out

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            ctype = self.headers["Content-Type"]
            log["posts"].append(time.time())
            if len(log["posts"]) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.end_headers()
                return
            if ctype == "image/tiff" and log["refuse_tiff"]:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": {"code": "InvalidContent"}}')
                return
            frames = frames_of(body, ctype)
            log["frames"].append(len(frames))
            log["types"].append(ctype)
            pages, content = [], ""
            for g, unit, W_, H_ in frames[:2]:                        # free tier: 2 per request
                k = W_ / g.shape[1]                                    # unit per rendered px
                n, lab, st, _ = cv2.connectedComponentsWithStats((g < 60).astype(np.uint8))
                words = []
                for j in range(1, n):
                    x, y, w, h, a = st[j]
                    if a < 400:
                        continue
                    off = len(content)
                    content += "M "
                    words.append({"content": "M", "confidence": 0.9, "span": {"offset": off, "length": 1},
                                  "polygon": [v * k for v in (x, y, x + w, y, x + w, y + h, x, y + h)]})
                pages.append({"unit": unit, "width": W_, "height": H_, "words": words, "lines": []})
            log["n"] += 1
            results[log["n"]] = {"status": "succeeded", "analyzeResult": {"content": content, "pages": pages}}
            self.send_response(202)
            self.send_header("Operation-Location", f"http://127.0.0.1:{port}/r/{log['n']}")
            self.end_headers()

        def do_GET(self):
            log["gets"].append(time.time())
            n = int(self.path.split("/")[-1])
            res = results[n] if len(log["gets"]) % 2 == 0 else {"status": "running"}
            data = json.dumps(res).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

    def page(i):                     # a straightened page at 1.5 x 200 dpi with a marker
        im = np.full((3300, 2550, 3), 245, np.uint8)
        x, y = 300 + 400 * i, 500 + 300 * i
        im[y:y + 60, x:x + 60] = 0
        return im, (x, y)

    def placed(texts, marks):
        ok = True
        for t, (x, y) in zip(texts, marks):
            if t.error or len(t.words) != 1:
                return False
            p = t.words[0].polygon
            ok &= abs(p[:, 0].min() - x) <= 3 and abs(p[:, 1].min() - y) <= 3
        return ok

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng = AzureLayoutOcr(endpoint=f"http://127.0.0.1:{port}", key="k")
        pages3 = [page(i) for i in range(3)]
        out3 = eng.analyse_pages([p for p, _ in pages3], [{}] * 3)
        check(log["frames"] == [2] and log["types"] == ["image/tiff"],
              "3-page form: ONE request, pages tiled onto 2 images (TIFF)")
        check(placed(out3, [m for _, m in pages3]), "every word mapped back to its own page and spot")
        pages6 = [page(i) for i in range(6)]
        out6 = eng.analyse_pages([p for p, _ in pages6], [{}] * 6)
        check(log["frames"][-1] == 2 and placed(out6, [m for _, m in pages6]),
              "6-page PDF: still one request (3 pages per image)")
        out_cut = eng.analyse_pages([p for p, _ in pages3], [{}] * 3, sheets=3)
        check(out_cut[2].error and "2 per request" in out_cut[2].error and not out_cut[0].error,
              "an image the free tier skipped is reported, not silently blank")
        log["refuse_tiff"] = True
        outp = eng.analyse_pages([p for p, _ in pages3], [{}] * 3)
        check(eng.format == "pdf" and log["types"][-1] == "application/pdf" and placed(outp, [m for _, m in pages3]),
              "TIFF refused -> same sheets sent as PDF (inches mapped back correctly)")
    finally:
        srv.shutdown()
    posts, gets = log["posts"], log["gets"]
    check(all(b - a >= 1.0 for a, b in zip(posts, posts[1:])), "at most 1 analyse request per second")
    check(all(b - a >= 1.0 for a, b in zip(gets, gets[1:])), "at most 1 result poll per second")
    check(len(posts) == 6, "429 (too many requests) retried")
    rng = np.random.default_rng(0)
    noisy = [np.clip(245 + rng.normal(0, 25, (3300, 2550, 3)), 0, 255).astype(np.uint8) for _ in range(6)]
    sheets, _, _ = AzureLayoutOcr._sheets(noisy, 2)
    data, _ = AzureLayoutOcr._encode(sheets, "tiff", 4.0)
    check(len(data) <= 4 * 1024 * 1024 and max(max(sh.shape[:2]) for sh in sheets) <= 10000,
          f"6 noisy pages fit the free tier's limits ({len(data) / 1e6:.1f} MB, "
          f"sheets {[sh.shape[1::-1] for sh in sheets]})")


# --------------------------------------------------------------------------- #
def two_versions(tmp: Path):
    """A second version of the form = page 1 with one more table line. Documents of each
    version are audited with BOTH templates loaded: every page must go to its own
    version, and pages that are the same in both (2, 3) must stay with their form's."""
    import cv2
    import numpy as np
    from PIL import Image
    from form_auditor.template import read_yaml
    from form_auditor.calibrate import dump_yaml
    src = ROOT / "templates" / "form.yaml"
    d = tmp / "versions"
    d.mkdir()
    raw = read_yaml(src)
    if (src.parent / raw.get("rules_csv", "")).is_file():
        shutil.copy(src.parent / raw["rules_csv"], d / raw["rules_csv"])
    pages = [cv2.imread(str(src.parent / p["template_image"])) for p in raw["pages"]]
    g = cv2.cvtColor(pages[0], cv2.COLOR_BGR2GRAY)
    ink_rows = (g < 128).sum(axis=1)
    y = next(y for y in range(int(0.12 * g.shape[0]), int(0.5 * g.shape[0]))
             if ink_rows[y - 12:y + 12].max() == 0)                # an empty band on page 1
    v2_p1 = pages[0].copy()
    cv2.line(v2_p1, (int(0.08 * g.shape[1]), y), (int(0.92 * g.shape[1]), y), (0, 0, 0), 2)
    for name, imgs, ver in (("form", pages, "1"), ("form_v2", [v2_p1] + pages[1:], "2")):
        t = dict(raw, version=ver, pages=[])
        for i, (p, im) in enumerate(zip(raw["pages"], imgs)):
            cv2.imwrite(str(d / f"{name}_p{i + 1}.png"), im)
            t["pages"].append(dict(p, template_image=f"{name}_p{i + 1}.png"))
        t.pop("document_rules", None)
        dump_yaml(t, d / f"{name}.yaml", "# test copy\n")
    for name, imgs in (("doc_v1.pdf", pages), ("doc_v2.pdf", [v2_p1] + pages[1:])):
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs * 2]   # 2 forms
        pil[0].save(d / name, resolution=200, save_all=True, append_images=pil[1:])
    res = run_audit([d / "doc_v1.pdf", d / "doc_v2.pdf"], d / "out",
                    templates=[d / "form.yaml", d / "form_v2.yaml"], verbose=False)
    by_file: dict = {}
    for r in res["summary"]:
        by_file.setdefault(r["file"], []).append((r["template"].split(" v")[-1], r["form_no"]))
    v1, v2 = by_file.get("doc_v1.pdf", []), by_file.get("doc_v2.pdf", [])
    check([v for v, _ in v1] == ["1"] * len(pages) * 2,
          f"every page of an old-version document matched to the old template ({[v for v, _ in v1]})")
    check([v for v, _ in v2] == ["2"] * len(pages) * 2,
          f"every page of a new-version document matched to the new one ({[v for v, _ in v2]})")
    n = len(pages)
    check([f for _, f in v1] == [1] * n + [2] * n and [f for _, f in v2] == [1] * n + [2] * n,
          "pages grouped into whole forms")


# --------------------------------------------------------------------------- #
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fa_unit_"))
    try:
        print(".env loader")
        env = tmp / ".env"
        env.write_text('# comment\nexport FA_T1="quoted value"\nFA_T2=plain # trailing\n'
                       "FA_T3='single'\nFA_KEEP=from_file\n")
        os.environ["FA_KEEP"] = "already set"
        load_env(env)
        check(os.environ.get("FA_T1") == "quoted value", "quoted value, export prefix")
        check(os.environ.get("FA_T2") == "plain", "trailing comment stripped")
        check(os.environ.get("FA_T3") == "single", "single quotes")
        check(os.environ.get("FA_KEEP") == "already set", "real environment variable wins")

        print("abfss:// via dbutils")
        lake = tmp / "lake"
        (lake / "incoming").mkdir(parents=True)
        shutil.copy(BLANK_PAGE, lake / "incoming" / "blank.png")
        (lake / "incoming" / "notes.txt").write_text("not a form")
        storage.set_dbutils(FakeDbutils(lake))
        res = run_audit(FakeDbutils.PREFIX + "incoming/", FakeDbutils.PREFIX + "audited/run1",
                        verbose=False)
        out = lake / "audited" / "run1"
        check(res["pages"] == 1, "one page audited (the .txt file skipped)")
        check(res["failing"] == 1, "blank page fails")
        for f in ("all_findings.csv", "batch_summary.csv", "rule_summary.csv",
                  "blank/blank_audit.csv", "blank/blank_p1_audited.png"):
            check((out / f).is_file(), f"uploaded {f}")
        rs = (out / "rule_summary.csv").read_text()
        check("coverage" in rs.splitlines()[0], "rule summary says which rules were audited")

        print("https://<account>.blob.core.windows.net via azure-storage-blob")
        blobs = tmp / "blobs"
        (blobs / "forms" / "in").mkdir(parents=True)
        shutil.copy(BLANK_PAGE, blobs / "forms" / "in" / "blank.png")
        install_fake_blob_sdk(blobs)
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=x"
        res = run_audit("https://acct.blob.core.windows.net/forms/in",
                        "https://acct.blob.core.windows.net/forms/out", verbose=False)
        check(res["pages"] == 1, "one page audited")
        check((blobs / "forms" / "out" / "blank" / "blank_audited.pdf").is_file(),
              "annotated PDF uploaded")
        check((blobs / "forms" / "out" / "batch_summary.csv").is_file(), "summary uploaded")

        print("Azure Document Intelligence Layout client (local stand-in of the service)")
        azure_layout_checks()

        print("two versions of a form loaded together")
        two_versions(tmp)

        print("template validation")
        bad = tmp / "tpl"
        shutil.copytree(ROOT / "templates", bad)
        y = (bad / "form.yaml").read_text()
        y = y.replace("document_rules:", "old_document_rules:")
        y += ("document_rules:\n- id: TEST_RULE\n  type: checkbox_conflict\n  checkbox: NO_SUCH_BOX\n"
              "  item: NO_SUCH_ITEM\n  value: 'Yes'\n")
        (bad / "form.yaml").write_text(y)
        try:
            load_template(bad / "form.yaml")
            check(False, "a rule naming a missing item is rejected")
        except ValueError as e:
            check("NO_SUCH" in str(e), f"a rule naming a missing item is rejected: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("ALL OK" if not FAILS else f"{len(FAILS)} check(s) failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
