"""The work behind app.py (the upload-and-download web app): audit a batch of uploaded
files in a private temporary folder and collect what the page shows and offers for download.
Kept apart from the Streamlit screen so it can be tested and reused without it."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .env import load_env

ALLOWED = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WORK_PREFIX = "form_auditor_app_"
KEEP_HOURS = 2.0                    # results of earlier sessions are deleted after this
RANK = {"ERROR": 5, "FAIL": 4, "REVIEW": 3, "UNRECOGNISED": 2, "PASS": 1, "": 0}


@dataclass
class FileResult:
    name: str                        # as uploaded
    status: str                      # worst page status: ERROR, FAIL, REVIEW, UNRECOGNISED, PASS
    pages: int = 0
    n_fail: int = 0
    n_review: int = 0
    pdf: Optional[Path] = None       # annotated PDF
    csv: Optional[Path] = None       # this file's findings
    previews: list[Path] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)    # FAIL / REVIEW findings
    note: str = ""


@dataclass
class BatchResult:
    work: Path
    files: list[FileResult]
    zip_path: Path
    all_findings: Path
    document_report: Path
    documents: list[dict]
    reader_used: bool
    stopped: str = ""
    seconds: float = 0.0

    @property
    def totals(self) -> dict:
        return {"files": len(self.files),
                "fail": sum(f.status in ("FAIL", "ERROR") for f in self.files),
                "review": sum(f.status == "REVIEW" for f in self.files),
                "unrecognised": sum(f.status == "UNRECOGNISED" for f in self.files),
                "pass": sum(f.status == "PASS" for f in self.files)}


def reader_configured() -> bool:
    """Is the Azure handwriting reader set up (environment variables, or a .env file)?"""
    load_env(None)
    return bool(os.environ.get("AZURE_DI_ENDPOINT") and os.environ.get("AZURE_DI_KEY"))


def safe_name(name: str, taken: set) -> str:
    """A plain file name: no folders, no odd characters, unique within the batch."""
    base = Path(name.replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .") or "file"
    out, n = f"{stem}{ext.lower()}", 2
    while out.lower() in taken:
        out, n = f"{stem}_{n}{ext.lower()}", n + 1
    taken.add(out.lower())
    return out


def clean_old_work(root: Optional[Path] = None, keep_hours: float = KEEP_HOURS) -> None:
    root = Path(root or tempfile.gettempdir())
    cutoff = time.time() - keep_hours * 3600
    for d in root.glob(WORK_PREFIX + "*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def audit_uploads(uploads: list[tuple[str, bytes]], use_reader: bool = True,
                  show_pass: bool = False, templates: Optional[list] = None,
                  work_root: Optional[Path] = None) -> BatchResult:
    """uploads: [(file name, bytes)]. Writes them to a new private folder, audits them and
    returns where everything is. Unsupported file types are listed as errors."""
    from .api import run_audit
    clean_old_work(work_root)
    work = Path(tempfile.mkdtemp(prefix=WORK_PREFIX, dir=work_root))
    inp, out = work / "in", work / "out"
    inp.mkdir()
    t0 = time.time()
    taken: set = set()
    names, paths, skipped = {}, [], []
    for name, data in uploads:
        if Path(name).suffix.lower() not in ALLOWED:
            skipped.append(name)
            continue
        fn = safe_name(name, taken)
        (inp / fn).write_bytes(data)
        names[fn] = name
        paths.append(inp / fn)
    reader = use_reader and reader_configured()
    res = run_audit(paths, out, templates=templates, ocr="azure" if reader else "none",
                    show_pass=show_pass, verbose=False) if paths else \
        {"summary": [], "findings": [], "documents": [], "stopped": ""}
    out.mkdir(exist_ok=True)

    files = []
    for fn, original in names.items():
        stem = Path(fn).stem
        pages = [s for s in res["summary"] if s.get("file") == fn]
        rows = [r for r in res["findings"] if r.get("file") == fn]
        status = max((s.get("page_status", "") for s in pages), key=lambda x: RANK.get(x, 0),
                     default="ERROR")
        fdir = out / stem
        fr = FileResult(
            name=original, status=status or "ERROR", pages=len(pages),
            n_fail=sum(r.get("status") == "FAIL" for r in rows),
            n_review=sum(r.get("status") == "REVIEW" for r in rows),
            pdf=next(iter(sorted(fdir.glob("*_audited.pdf"))), None) if fdir.is_dir() else None,
            csv=(fdir / f"{stem}_audit.csv") if (fdir / f"{stem}_audit.csv").is_file() else None,
            previews=sorted(p for p in fdir.glob("*_audited.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if fdir.is_dir() else [],
            problems=[r for r in rows if r.get("status") in ("FAIL", "REVIEW")],
            note="; ".join(sorted({s.get("note", "") for s in pages if s.get("note")}))[:300])
        files.append(fr)
    for name in skipped:
        files.append(FileResult(name=name, status="ERROR",
                                note="not a PDF or image (PDF, JPG, PNG or TIFF only)"))
    files.sort(key=lambda f: (-RANK.get(f.status, 0), f.name.lower()))

    zip_path = work / "form_audit_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out))
    return BatchResult(work=work, files=files, zip_path=zip_path,
                       all_findings=out / "all_findings.csv",
                       document_report=out / "document_report.csv",
                       documents=[{**d, "file": names.get(d.get("file", ""), d.get("file", ""))}
                                  for d in res.get("documents", [])],
                       reader_used=reader,
                       stopped=res.get("stopped", ""), seconds=time.time() - t0)


# --------------------------------------------------------------------------- #
# Branding: the app's words and images (branding/branding.yaml); colours and fonts are in
# .streamlit/config.toml
# --------------------------------------------------------------------------- #
BRANDING_DEFAULTS = {
    "app_title": "Form auditor",
    "subtitle": ("Upload scanned or photographed checklist forms (PDF or photos). Each page is "
                 "checked against the blank form and the rules: problems are boxed and numbered "
                 "on the PDF, and the report lists what was checked, where and why."),
    "company_name": "", "logo": "", "logo_small": "",
    "footer": "Files are processed for this session only and are not kept.",
    "support_contact": "", "css": "",
}


def load_branding(root: Optional[Path] = None) -> dict:
    """branding/branding.yaml (else the example, else defaults), with file entries (logo,
    logo_small, css) turned into paths that exist, or ''."""
    import yaml
    root = Path(root or Path(__file__).resolve().parents[1])
    d = root / "branding"
    b = dict(BRANDING_DEFAULTS)
    for name in ("branding.yaml", "branding.example.yaml"):
        f = d / name
        if f.is_file():
            try:
                b.update({k: v for k, v in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).items()
                          if v is not None})
            except Exception:
                pass
            break
    for k in ("logo", "logo_small", "css"):
        v = str(b.get(k) or "").strip()
        p = (d / v) if v else None
        b[k] = str(p) if p is not None and p.is_file() and p.resolve().parent == d.resolve() else ""
    return b
