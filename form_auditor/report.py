"""Drawing problem boxes on the ORIGINAL page and writing CSV reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .rules_common import Finding

COLORS = {"FAIL": (40, 40, 220), "REVIEW": (0, 150, 255), "PASS": (60, 170, 60)}

FINDING_COLUMNS = [
    "file", "page", "form_no", "template", "template_page", "marker", "section", "section_title",
    "item_id", "item_label", "check", "status", "observed", "expected", "reason",
    "value_read", "value_normalized", "mark_type", "notes",
    "rule_ref", "rule_instruction",
    "confidence", "ocr_engine", "ocr_confidence", "ink_ratio", "visible_ratio", "marks",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "bbox_units", "bbox_template_norm",
    "align_inliers", "align_method",
]
SUMMARY_COLUMNS = [
    "file", "page", "form_no", "template", "template_page", "page_status", "n_fail", "n_review",
    "n_pass", "n_skip", "align_inliers", "align_method", "annotated_output", "note",
]
RULE_SUMMARY_COLUMNS = ["rule_ref", "category", "instruction", "coverage", "n_checks", "n_fail",
                        "n_review", "n_pass", "n_skip", "pages_with_failures", "items"]


def project_box(box, tpl_shape, transform, n: int = 6) -> np.ndarray:
    """Normalised template box -> polygon in original-image pixels (edges sampled so the
    outline follows any lens-distortion correction)."""
    th, tw = tpl_shape[:2]
    x0, y0, x1, y1 = box[0] * tw, box[1] * th, box[2] * tw, box[3] * th
    t = np.linspace(0, 1, n)
    pts = np.concatenate([np.stack([x0 + (x1 - x0) * t, np.full(n, y0)], 1),
                          np.stack([np.full(n, x1), y0 + (y1 - y0) * t], 1),
                          np.stack([x1 - (x1 - x0) * t, np.full(n, y1)], 1),
                          np.stack([np.full(n, x0), y1 - (y1 - y0) * t], 1)])
    return transform.tpl_to_img(pts)


def page_status(findings: list[Finding]) -> str:
    st = {f.status for f in findings}
    return "FAIL" if "FAIL" in st else "REVIEW" if "REVIEW" in st else "PASS"


def _put_label(img, text, org, color, scale, thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    x, y = int(org[0]), int(org[1])
    y = max(h + base + 2, y)
    x = min(max(0, x), img.shape[1] - w - 4)
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def annotate(original: np.ndarray, findings: list[Finding], tpl_shape, transform,
             header: str, show_pass: bool = False) -> np.ndarray:
    img = original.copy()
    ih, iw = img.shape[:2]
    unit = max(iw, ih) / 2000.0
    thick = max(2, int(round(3 * unit)))
    fscale = 0.6 * unit
    fthick = max(1, int(round(1.6 * unit)))

    if transform is not None:
        order = {"PASS": 0, "REVIEW": 1, "FAIL": 2}   # draw FAIL last (on top)
        for f in sorted(findings, key=lambda f: order.get(f.status, 0)):
            if f.status not in ("FAIL", "REVIEW") and not (show_pass and f.status == "PASS"):
                continue
            color = COLORS[f.status]
            for b in f.boxes:
                poly = project_box(b, tpl_shape, transform).astype(np.int32)
                cv2.polylines(img, [poly], True, color, thick if f.status != "PASS" else 1,
                              cv2.LINE_AA)
                if f.marker is not None:
                    _put_label(img, f"#{f.marker} {f.item_id}", poly[0], color, fscale, fthick)

    def wrap(text, scale, th):
        words, out, cur = text.split(" "), [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0][0] > iw - 30 and cur:
                out.append(cur)
                cur = w
            else:
                cur = cand
        return out + [cur]

    lines = wrap(header, fscale * 1.1, fthick + 1)
    n_head = len(lines)
    for f in findings:
        if not f.boxes and f.status in ("FAIL", "REVIEW"):
            lines += wrap(f"#{f.marker} {f.status}: {f.reason}", fscale * 1.1, fthick)
    line_h = int(34 * unit) + 6
    banner = np.full((line_h * len(lines) + 12, iw, 3), 255, np.uint8)
    st = header.split("|")[-1]
    hdr_color = COLORS["FAIL"] if "FAIL" in st else COLORS["REVIEW"] if "REVIEW" in st else COLORS["PASS"]
    cv2.rectangle(banner, (0, 0), (iw - 1, banner.shape[0] - 1), hdr_color, thick)
    for i, t in enumerate(lines):
        head = i < n_head
        cv2.putText(banner, t, (10, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    fscale * 1.1, hdr_color if head else (20, 20, 20), fthick + (1 if head else 0),
                    cv2.LINE_AA)
    return np.vstack([banner, img])


def finding_rows(findings: list[Finding], file: str, page: int, template: str, tpage: str,
                 tpl_shape, transform, align, rule_text=None) -> list[dict]:
    rows = []
    for f in findings:
        bbox = ["", "", "", ""]
        if f.boxes and transform is not None:
            pts = np.vstack([project_box(b, tpl_shape, transform) for b in f.boxes])
            bbox = [int(pts[:, 0].min()), int(pts[:, 1].min()),
                    int(pts[:, 0].max()), int(pts[:, 1].max())]
        rows.append({
            "file": file, "page": page, "template": template, "template_page": tpage,
            "marker": f.marker if f.marker is not None else "",
            "section": f.section_id, "section_title": f.section_title,
            "item_id": f.item_id, "item_label": f.item_label, "check": f.check,
            "status": f.status, "observed": f.observed, "expected": f.expected,
            "reason": f.reason, "value_read": f.value_read, "value_normalized": f.value_normalized,
            "mark_type": f.mark_type, "notes": "; ".join(f.notes),
            "rule_ref": "; ".join(f.rule_refs),
            "rule_instruction": rule_text(f.rule_refs) if (rule_text and f.rule_refs) else "",
            "confidence": f.confidence, "ocr_engine": f.ocr_engine,
            "ocr_confidence": "" if f.ocr_confidence is None else f.ocr_confidence,
            "ink_ratio": f.metrics.get("ink_ratio", ""),
            "visible_ratio": f.metrics.get("visible_ratio", ""),
            "marks": f.metrics.get("marks", ""),
            "bbox_x0": bbox[0], "bbox_y0": bbox[1], "bbox_x1": bbox[2], "bbox_y1": bbox[3],
            "bbox_units": "px in original page (unannotated)" if bbox[0] != "" else "",
            "bbox_template_norm": json.dumps([[round(v, 4) for v in b] for b in f.boxes]) if f.boxes else "",
            "align_inliers": align.inliers if align else "",
            "align_method": align.message if align else "",
        })
    return rows


def extracted_record(findings: list[Finding], base: dict) -> dict:
    """One row per audited page: what was read from the form."""
    rec = dict(base)
    for f in findings:
        if f.section_id == "PAGE":
            continue
        if f.check == "response":
            rec[f.item_id] = f.observed
            if f.mark_type:
                rec[f"{f.item_id}.mark"] = f.mark_type
        elif f.check == "details":
            if f.observed == "filled":
                rec[f"{f.item_id}.details"] = f.value_normalized or f.value_read or "(written, not transcribed)"
            else:
                rec[f"{f.item_id}.details"] = "" if f.observed == "blank" else f.observed
        elif f.check == "na_checkbox":
            rec[f.item_id] = f.observed
        elif f.check == "item_in_use":
            rec[f"{f.item_id}.in_use"] = f.observed
        elif f.item_id.startswith("rule:"):
            continue
        elif f.check == "signature_present":
            rec[f.item_id] = {"filled": "signed", "blank": ""}.get(f.observed, f.observed)
        else:
            if f.observed == "filled":
                rec[f.item_id] = f.value_normalized or f.value_read or "(written, not transcribed)"
            else:
                rec[f.item_id] = "" if f.observed == "blank" else f.observed
            if f.value_read and f.value_normalized and f.value_read != f.value_normalized:
                rec[f"{f.item_id}.as_written"] = f.value_read
    return rec


def rule_summary(rows: list[dict], rules: dict, inactive: Optional[dict] = None) -> list[dict]:
    """`inactive`: rule id -> why it was not applied (e.g. switched off by a setting)."""
    inactive = inactive or {}
    agg: dict[str, dict] = {}
    for r in rows:
        for ref in [x.strip() for x in (r.get("rule_ref") or "").split(";") if x.strip()]:
            a = agg.setdefault(ref, {"rule_ref": ref, "n_checks": 0, "n_fail": 0, "n_review": 0,
                                     "n_pass": 0, "n_skip": 0, "_pages": set(), "_items": set()})
            a["n_checks"] += 1
            key = f"n_{r['status'].lower()}"
            if key in a:
                a[key] += 1
            if r["status"] == "FAIL":
                a["_pages"].add(f"{r['file']} p{r['page']}")
            a["_items"].add(r["item_id"])
    out = []
    order = list(rules) + sorted(k for k in agg if k not in rules)
    for ref in order:
        a = agg.get(ref, {"rule_ref": ref, "n_checks": 0, "n_fail": 0, "n_review": 0,
                          "n_pass": 0, "n_skip": 0, "_pages": set(), "_items": set()})
        rr = rules.get(ref)
        out.append({**{k: v for k, v in a.items() if not k.startswith("_")},
                    "category": rr.category if rr else "",
                    "instruction": rr.instruction if rr else "",
                    "coverage": "audited" if a["n_checks"] else
                                inactive.get(ref, "no finding cited this rule in this batch"),
                    "pages_with_failures": len(a["_pages"]),
                    "items": ", ".join(sorted(a["_items"]))})
    return out


def write_csv(path: Path, rows: list[dict], columns: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:                     # dynamic columns, first-seen order
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_pdf(images_bgr: list[np.ndarray], path: Path, dpi: int = 150) -> None:
    pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images_bgr]
    if pil:
        pil[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pil[1:])
