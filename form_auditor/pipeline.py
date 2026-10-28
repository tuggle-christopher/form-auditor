"""End-to-end: file(s) -> pages -> identify template page -> align -> measure -> rules -> outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .align import Alignment, TemplateMatcher, extract_features
from .context import PageContext
from .ink import compute_ink_maps, rule_zone, template_core, template_ink
from .io_utils import PageImage, load_pages
from .ocr import OcrEngine
from .report import (FINDING_COLUMNS, SUMMARY_COLUMNS, annotate, extracted_record,
                     finding_rows, page_status, save_pdf, write_csv)
from .rules import Finding, apply_document_rules, audit_page
from .settings import Settings
from .template import FormTemplate, TemplatePage


@dataclass
class _RuntimePage:
    template: FormTemplate
    page: TemplatePage
    matcher: TemplateMatcher
    tpl_ink: np.ndarray
    rules_zone: np.ndarray
    tpl_core: np.ndarray
    rules_near: np.ndarray
    tpl_ink_faint: Optional[np.ndarray] = None    # the form's print as the faint pass sees it


@dataclass
class PageResult:
    page_index: int
    template: Optional[str]
    template_page: Optional[str]
    status: str
    findings: list[Finding] = field(default_factory=list)
    alignment: Optional[Alignment] = None
    annotated: Optional[np.ndarray] = None
    note: str = ""
    image: Optional[np.ndarray] = None
    runtime: Optional["_RuntimePage"] = None
    ctx: Optional[PageContext] = None           # until audited
    where: dict = field(default_factory=dict)
    page_text: Optional[object] = None          # layout.PageText from a page-mode reader
    debug_dir: Optional[Path] = None
    page: Optional[PageImage] = None


class Auditor:
    def __init__(self, templates: list[FormTemplate], verbose: bool = True,
                 ocr: Optional[OcrEngine] = None):
        self.templates = templates
        self.verbose = verbose
        self.ocr = ocr or OcrEngine()
        self.runtime: list[_RuntimePage] = []
        for t in templates:
            sf = self._faint_settings(t.settings)
            for p in t.pages:
                img = p.image
                self.runtime.append(_RuntimePage(t, p, TemplateMatcher(img, t.settings),
                                                 template_ink(img, t.settings), rule_zone(img),
                                                 template_core(img, t.settings),
                                                 rule_zone(img, t.settings.strike_rule_band_px),
                                                 template_ink(img, sf)))

    @staticmethod
    def _faint_settings(s: Settings) -> Settings:
        return s.merged({"ink_contrast": s.faint_ink_contrast, "dark_max": s.faint_dark_max,
                         "edge_margin_px": s.faint_edge_margin_px})

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ #
    def identify(self, img: np.ndarray, prefer: Optional[FormTemplate] = None
                 ) -> tuple[Optional[_RuntimePage], Alignment]:
        """Rank template pages by a quick feature-match score, then fully align against
        them. With one template the first page that passes the layout check wins. With
        several (e.g. an old and a new version of the form), every likely page is aligned
        and the best-fitting layout wins; on a near tie the template of the previous page
        of the file (`prefer`) is kept, so one form is not split across versions."""
        cache: dict = {}

        def feats(rt):
            s = rt.template.settings
            key = (s.align_max_dim, s.sift_features, s.align_blur)
            if key not in cache:
                cache[key] = extract_features(img, s)
            return cache[key]

        scored = sorted(((rt, rt.matcher.quick_inliers(feats(rt))) for rt in self.runtime),
                        key=lambda p: -p[1])
        best_q = scored[0][1] if scored else 0
        several = len(self.templates) > 1
        best_al = Alignment(False, None, 0, 0.0, "no templates")
        found = []
        for rt, q in scored:
            if found and q < 0.5 * best_q:
                break                                   # clearly a different page
            al = rt.matcher.align(img, feats(rt))
            if al.ok:
                found.append((rt, al))
                if not several:
                    break
            elif (al.layout_score, al.inliers) > (best_al.layout_score, best_al.inliers):
                best_al = al
        if not found:
            return None, best_al

        def fit(c):
            rt, al = c
            return (al.layout_score + (0.02 if prefer is not None and rt.template is prefer else 0.0),
                    al.inliers)
        return max(found, key=fit)

    def analyse_page(self, pg: PageImage, debug_dir: Optional[Path] = None,
                     prefer: Optional[FormTemplate] = None) -> PageResult:
        """Identify and align one page and measure its ink. The rules are applied by
        audit_prepared (after read_pages, for a page-mode reader); nothing is drawn yet -
        cross-page rules may still add findings, see finish_form."""
        rt, al = self.identify(pg.image, prefer)
        label = f"{pg.source.name} p{pg.page_index + 1}"
        if rt is None:
            note = f"Page not recognised as any template ({al.message})"
            self._log(f"  [{label}] UNRECOGNISED - {al.message}")
            return PageResult(pg.page_index, None, None, "UNRECOGNISED", [], al, None, note,
                              image=pg.image)

        s = rt.template.settings
        maps = compute_ink_maps(pg.image, al.transform, rt.tpl_ink, s, rt.rules_zone, rt.tpl_core)
        faint = None
        if s.faint_check:
            faint = compute_ink_maps(pg.image, al.transform, rt.tpl_ink | rt.tpl_ink_faint,
                                     self._faint_settings(s), rt.rules_zone, rt.tpl_core)
            faint.added |= maps.added
        ctx = PageContext(maps, s, pg.image, al.transform, rt.tpl_core, rt.rules_zone,
                          rt.rules_near, faint)
        self._log(f"  [{label}] {rt.template.display_name}/{rt.page.page_id} "
                  f"inliers={al.inliers} ({al.message})")
        res = PageResult(pg.page_index, rt.template.display_name, rt.page.page_id, "",
                         [], al, None, image=pg.image, runtime=rt)
        res.ctx, res.where = ctx, {"file": pg.source.name, "page": pg.page_index + 1}
        res.debug_dir, res.page = debug_dir, pg
        return res

    def read_pages(self, results: list[PageResult]) -> None:
        """Page-mode reader: send the recognised pages of `results` (one request; the
        caller keeps it within azure_pages_per_request) and keep each page's words."""
        todo = [r for r in results if r.ctx is not None]
        if not todo or not getattr(self.ocr, "page_level", False):
            return
        s = todo[0].runtime.template.settings
        k = s.azure_page_scale
        images = [r.ctx.transform.warp(r.ctx.original, scale=k) for r in todo]
        texts = self.ocr.analyse_pages(images, [r.where for r in todo], s.azure_max_request_mb,
                                       sheets=max(1, s.azure_pages_per_request))
        for r, t in zip(todo, texts):
            r.page_text = t.scaled(1.0 / k)
            if t.error:
                self._log(f"  [{r.where['file']} p{r.where['page']}] handwriting reader: {t.error}")

    def audit_prepared(self, res: PageResult) -> PageResult:
        """Apply the rules to a page prepared by analyse_page (and read by read_pages)."""
        if res.ctx is None:
            return res
        rt, ctx = res.runtime, res.ctx
        res.findings = audit_page(rt.page, ctx, rt.template.settings, self.ocr, where=res.where,
                                  page_text=res.page_text)
        if res.debug_dir is not None:
            self._write_debug(res.debug_dir, res.page, rt, ctx.maps, res.findings)
        res.ctx = res.page = None                         # free memory
        return res

    def finish_form(self, pages: list[PageResult], label: str, show_pass: bool = False) -> None:
        """Apply cross-page rules to the pages of one form, then number and draw the
        problems on each page."""
        known = [p for p in pages if p.runtime is not None]
        if known:
            tpl = known[0].runtime.template
            by_id = {p.template_page: p for p in known}
            for page_id, f in apply_document_rules(tpl, {k: v.findings for k, v in by_id.items()},
                                                   tpl.settings):
                by_id[page_id].findings.append(f)
        for res in pages:
            name = f"{label} p{res.page_index + 1}"
            if res.runtime is None:
                res.annotated = annotate(res.image, [], (1, 1), None,
                                         f"{name} | UNRECOGNISED PAGE - NOT AUDITED "
                                         f"({res.alignment.message if res.alignment else ''}) | REVIEW")
                continue
            n = 0
            for f in res.findings:
                if f.status in ("FAIL", "REVIEW"):
                    n += 1
                    f.marker = n
            res.status = page_status(res.findings)
            c = {k: sum(f.status == k for f in res.findings) for k in ("FAIL", "REVIEW", "PASS", "SKIP")}
            rt = res.runtime
            header = (f"{name} | {rt.template.display_name} [{rt.page.page_id}] | "
                      f"{c['FAIL']} fail, {c['REVIEW']} review, {c['PASS']} pass | {res.status}")
            res.annotated = annotate(res.image, res.findings, rt.page.image.shape,
                                     res.alignment.transform, header, show_pass)
            self._log(f"  [{name}] -> {res.status} ({c['FAIL']} fail, {c['REVIEW']} review)")

    def audit_page_image(self, pg: PageImage, show_pass=False, debug_dir: Optional[Path] = None
                         ) -> PageResult:
        """Single page on its own (no cross-page rules)."""
        res = self.analyse_page(pg, debug_dir)
        self.read_pages([res])
        self.audit_prepared(res)
        self.finish_form([res], pg.source.name, show_pass)
        return res

    @staticmethod
    def _write_debug(debug_dir: Path, pg: PageImage, rt: _RuntimePage, maps, findings):
        from .ink import box_to_px
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = f"p{pg.page_index + 1}"
        vis = maps.warped.copy()
        vis[maps.occluded] = (0.5 * vis[maps.occluded] + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
        vis[maps.added] = (0, 0, 255)
        for f in findings:
            for b in f.boxes:
                x0, y0, x1, y1 = box_to_px(b, vis.shape, rt.template.settings.roi_inset_frac)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 160, 0), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_aligned.jpg"), maps.warped)
        for f in findings:
            for part in (f.metrics.get("marks") or "").split("; "):
                if "@(" in part:
                    x0, y0, x1, y1 = (int(v) for v in part.split("@(")[1].rstrip(")").split(","))
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 0, 200), 1)
                    cv2.putText(vis, part.split("@")[0], (x0, max(10, y0 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_ink_overlay.jpg"), vis)
        blend = cv2.addWeighted(maps.warped, 0.5, rt.page.image, 0.5, 0)
        cv2.imwrite(str(debug_dir / f"{stem}_alignment_check.jpg"), blend)

    # ------------------------------------------------------------------ #
    def audit_file(self, path: Path, out_dir: Path, show_pass=False, debug=False,
                   skip_unrecognised: bool = False,
                   dpi: Optional[int] = None, draw_pdf_annotations: Optional[bool] = None
                   ) -> tuple[list[dict], list[dict], list[dict]]:
        """Audit every page of one file. Consecutive pages p1, p2, p3 of a template form
        one filled form (form_no); a new p1 (or a page out of order) starts the next form.
        Returns (finding rows, page summary rows, one extracted-data row per form)."""
        base = self.templates[0].settings
        dpi = dpi or base.render_dpi
        draw = base.draw_pdf_annotations if draw_pdf_annotations is None else draw_pdf_annotations
        file_out = out_dir / path.stem
        file_out.mkdir(parents=True, exist_ok=True)
        self._log(f"Auditing {path}")

        all_rows, summary, annotated, records = [], [], [], []
        state = {"form_no": 0}

        def flush(group: list[PageResult]):
            if not group:
                return
            known = [r for r in group if r.runtime is not None]
            form_no = ""
            if known:
                state["form_no"] += 1
                form_no = state["form_no"]
            self.finish_form(group, path.name, show_pass)
            rec = {"file": path.name, "form_no": form_no,
                   "pages": " ".join(str(r.page_index + 1) for r in group)}
            for res in group:
                ext = ".png" if is_pdf else (path.suffix.lower() if path.suffix.lower() in
                                             (".jpg", ".jpeg", ".png") else ".png")
                img_out = file_out / f"{path.stem}_p{res.page_index + 1}_audited{ext}"
                if skip_unrecognised and res.status == "UNRECOGNISED":
                    img_out = None
                else:
                    cv2.imwrite(str(img_out), res.annotated)
                    annotated.append(res.annotated)
                rt = res.runtime
                rows = finding_rows(res.findings, path.name, res.page_index + 1,
                                    res.template or "", res.template_page or "",
                                    rt.page.image.shape if rt else None,
                                    res.alignment.transform if rt else None,
                                    res.alignment, rt.template.rule_text if rt else None)
                for r in rows:
                    r["form_no"] = form_no
                all_rows.extend(rows)
                c = {k: sum(f.status == k for f in res.findings) for k in ("FAIL", "REVIEW", "PASS", "SKIP")}
                summary.append({
                    "file": path.name, "page": res.page_index + 1, "form_no": form_no,
                    "template": res.template or "", "template_page": res.template_page or "",
                    "page_status": res.status,
                    "n_fail": c["FAIL"], "n_review": c["REVIEW"], "n_pass": c["PASS"], "n_skip": c["SKIP"],
                    "align_inliers": res.alignment.inliers if res.alignment else 0,
                    "align_method": res.alignment.message if res.alignment else "",
                    "annotated_output": str(img_out) if img_out else "", "note": res.note,
                })
                if rt:
                    rec.update(extracted_record(res.findings, {}))
                res.image = res.annotated = None              # free memory
            if known:
                tpl = known[0].runtime.template
                have = {r.template_page for r in known}
                missing = [p.page_id for p in tpl.pages if p.page_id not in have]
                rec["form_status"] = page_status([f for r in known for f in r.findings])
                rec["template"] = tpl.display_name
                rec["missing_pages"] = " ".join(missing)
                records.append(rec)

        group: list[PageResult] = []
        is_pdf = False
        # page-mode reader: the pages of a file are read together, as many as fit in one
        # request (azure_pages_per_request images x azure_pages_per_sheet pages each; free
        # tier: 2 x 3), so prepared pages wait until a request is full or the file ends
        batch_n = (max(1, base.azure_pages_per_request) * max(1, base.azure_pages_per_sheet)
                   if getattr(self.ocr, "page_level", False) else 1)
        pending: list[PageResult] = []

        def ready():
            self.read_pages(pending)
            done = [self.audit_prepared(r) for r in pending]
            pending.clear()
            return done

        last_tpl: dict = {"t": None}

        def results():
            for pg in load_pages(path, dpi=dpi, draw_annotations=draw):
                r = self.analyse_page(pg, file_out / "debug" if debug else None, last_tpl["t"])
                if r.runtime is not None:
                    last_tpl["t"] = r.runtime.template
                pending.append(r)
                if sum(r.ctx is not None for r in pending) >= batch_n:
                    yield from ready()
            yield from ready()

        for res in results():
            is_pdf = is_pdf or (res.image is not None and path.suffix.lower() == ".pdf")
            if group and res.runtime is not None:
                last = next((r for r in reversed(group) if r.runtime is not None), None)
                if last is not None:
                    order = [p.page_id for p in res.runtime.template.pages]
                    same = last.runtime.template is res.runtime.template
                    if not same or order.index(res.template_page) <= order.index(last.template_page):
                        flush(group)
                        group = []
            group.append(res)
        flush(group)

        if annotated:
            save_pdf(annotated, file_out / f"{path.stem}_audited.pdf")
        write_csv(file_out / f"{path.stem}_audit.csv", all_rows, FINDING_COLUMNS)
        write_csv(file_out / f"{path.stem}_summary.csv", summary, SUMMARY_COLUMNS)
        write_csv(file_out / f"{path.stem}_extracted.csv", records)
        return all_rows, summary, records
