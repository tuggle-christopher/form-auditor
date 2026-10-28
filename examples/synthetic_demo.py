"""Synthetic end-to-end demo / regression test.

Builds a blank synthetic checklist (a made-up form), a calibrated template YAML,
and filled versions that reproduce the mistakes in the real sample photo
(blank Section A, 'Yes' with details written, 'No' without details, NA with
details, missing dates, a blue callout box hiding a row ...). The filled page is
then distorted like a phone photo. Finally the auditor runs and its verdicts are
compared with the known ground truth.

    python examples/synthetic_demo.py           # writes examples/demo/ and checks results
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from form_auditor.calibrate import dump_yaml          # noqa: E402
from form_auditor.pipeline import Auditor             # noqa: E402
from form_auditor.template import load_template       # noqa: E402

OUT = ROOT / "examples" / "demo"
W, H = 1700, 2200
rng = np.random.default_rng(7)
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (70, 35, 25)          # dark blue-black ballpoint (BGR)

# ----------------------------------------------------------------------------
# Layout (pixels on the blank page)
# ----------------------------------------------------------------------------
COLS_B = dict(q=(100, 1000), Yes=(1000, 1110), No=(1110, 1220), NA=(1220, 1330), det=(1330, 1600))
B_ROWS = [(620 + 90 * i, 710 + 90 * i) for i in range(5)]
C_ROWS = [(1420 + 70 * i, 1490 + 70 * i) for i in range(8)]
B_Q = ["1) All documents received", "2) Quantity matches the order",
       "3) Batch number and date match the certificate",
       "4) Labels match the system report", "5) Extra label applied"]
C_Q = ["1) Area is clean and in good condition",
       "2) Seal present and intact",
       "3) Cover in place, no gaps or holes",
       "4) No pests found before work starts",
       "5) No pests found after work is done",
       "6) No chemicals or spills present",
       "7) No excessive dirt or grease",
       "8) Protective cover present"]
NA_PRINTED = {"B5", "C2", "C8"}


def txt(img, s, x, y, scale=0.7, th=2):
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)


def rect(img, x0, y0, x1, y1, th=2):
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), th)


def make_blank() -> np.ndarray:
    img = np.full((H, W, 3), 255, np.uint8)
    txt(img, "DEMO-FORM v1.0", 520, 60, 1.0, 2)
    txt(img, "Demo Checklist", 520, 105, 1.0, 2)
    # Section A
    rect(img, 100, 150, 1600, 430, 3)
    txt(img, "Section A - General Information", 115, 192, 0.85, 2)
    for y in (210, 320):
        cv2.line(img, (100, y), (1600, y), (0, 0, 0), 2)
    for x in (500, 900, 1200):
        cv2.line(img, (x, 210), (x, 430), (0, 0, 0), 2)
    txt(img, "Reference number", 115, 255); txt(img, "(Enter NA if none)", 115, 295, 0.55, 1)
    txt(img, "Lot number", 115, 365); txt(img, "(Enter NA if none)", 115, 405, 0.55, 1)
    txt(img, "Order /", 915, 255); txt(img, "Document number", 915, 295)
    txt(img, "Date", 915, 385)

    def table(y_head, y_rows, rows, qs, sec, signoff_y):
        y0, y1 = y_head, y_rows[-1][1]
        rect(img, 100, y0, 1600, y1, 3)
        cv2.line(img, (1000, y0), (1000, y1), (0, 0, 0), 2)
        cv2.line(img, (1330, y0), (1330, y1), (0, 0, 0), 2)
        txt(img, f"Section {sec} - " + ("Document Checks" if sec == "B"
                                        else "Area Checks"), 115, y0 + 60, 0.8, 2)
        txt(img, "Response", 1100, y0 + 60, 0.8, 2)
        txt(img, "If No, provide details", 1340, y0 + 30, 0.5, 1)
        txt(img, "and follow-up actions", 1340, y0 + 55, 0.5, 1)
        txt(img, "(Leave blank if Yes/NA)", 1340, y0 + 80, 0.5, 1)
        for i, (r0, r1) in enumerate(rows):
            cv2.line(img, (100, r0), (1600, r0), (0, 0, 0), 2)
            for x in (1110, 1220):
                cv2.line(img, (x, r0), (x, r1), (0, 0, 0), 1)
            ty = (r0 + r1) // 2 + 8
            txt(img, qs[i], 115, ty, 0.62, 2)
            txt(img, "Yes", 1030, ty, 0.75, 2)
            txt(img, "No", 1145, ty, 0.75, 2)
            if f"{sec}{i + 1}" in NA_PRINTED:
                txt(img, "NA", 1252, ty, 0.75, 2)
        # sign-off
        s0 = signoff_y
        rect(img, 100, s0, 1600, s0 + 140, 3)
        cv2.line(img, (100, s0 + 45), (1600, s0 + 45), (0, 0, 0), 2)
        cv2.line(img, (1000, s0 + 45), (1000, s0 + 140), (0, 0, 0), 2)
        txt(img, f"Section {sec} Completed by", 115, s0 + 32, 0.75, 2)
        txt(img, "Sign", 115, s0 + 105, 0.8, 2)
        txt(img, "Date", 1015, s0 + 105, 0.8, 2)

    table(500, B_ROWS, B_ROWS, B_Q, "B", 1070)
    txt(img, "Sections A and B shall be completed before the form is handed over",
        110, 1255, 0.65, 2)
    txt(img, "for the sections below to be completed", 110, 1290, 0.65, 2)
    table(1320, C_ROWS, C_ROWS, C_Q, "C", 1990)
    # a bit of non-repeating texture so the footer is matchable
    txt(img, "Doc ref: DEMO-0001   Rev 1.0   Synthetic test form - not a real document", 300, 2170, 0.55, 1)
    return img


def n(x0, y0, x1, y1):
    return [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)]


def make_template_yaml(png_name: str) -> dict:
    def choice_items(prefix, rows, qs):
        items = []
        for i, (r0, r1) in enumerate(rows):
            items.append({
                "id": f"{prefix}{i + 1}", "label": qs[i][3:], "type": "choice",
                "options": {o: n(COLS_B[o][0], r0, COLS_B[o][1], r1) for o in ("Yes", "No", "NA")},
                "details": {"box": n(COLS_B["det"][0], r0, COLS_B["det"][1], r1)},
            })
        return items

    def signoff(prefix, s0):
        return [{"id": f"{prefix}.sign", "label": f"Section {prefix} signature", "type": "signature",
                 "box": n(300, s0 + 45, 1000, s0 + 140)},
                {"id": f"{prefix}.date", "label": f"Section {prefix} date", "type": "date",
                 "box": n(1100, s0 + 45, 1600, s0 + 140)}]

    det_defaults = {"details": {"required_when": ["No"], "must_be_blank_when": ["Yes", "NA"]}}
    return {
        "form_id": "DEMO-FORM", "version": "1.0", "name": "Demo checklist (synthetic)",
        "pages": [{
            "page_id": "p1", "template_image": png_name,
            "sections": [
                {"id": "A", "title": "General Information", "items": [
                    {"id": "A.ref_no", "label": "Reference number", "type": "text",
                     "box": n(500, 210, 900, 320)},
                    {"id": "A.lot_no", "label": "Lot number", "type": "text", "box": n(500, 320, 900, 430)},
                    {"id": "A.order_no", "label": "Order / document number", "type": "text",
                     "box": n(1200, 210, 1600, 320)},
                    {"id": "A.date", "label": "Date", "type": "date",
                     "box": n(1200, 320, 1600, 430)}]},
                {"id": "B", "title": "Document Checks", "choice_defaults": det_defaults,
                 "items": choice_items("B", B_ROWS, B_Q) + signoff("B", 1070)},
                {"id": "C", "title": "Area Checks", "choice_defaults": det_defaults,
                 "items": choice_items("C", C_ROWS, C_Q) + signoff("C", 1990)},
            ],
            "rules": [{"id": "A_B_before_C", "type": "complete_before",
                       "prerequisite_sections": ["A", "B"], "dependent_section": "C",
                       "description": "Sections A and B shall be completed before the form is handed over"}],
        }],
    }


# ----------------------------------------------------------------------------
# "Handwriting"
# ----------------------------------------------------------------------------
def write(img, s, x, y, scale=1.1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, INK, 3, cv2.LINE_AA)


def circle(img, opt, row):
    cx = (COLS_B[opt][0] + COLS_B[opt][1]) // 2
    cy = (row[0] + row[1]) // 2
    t = np.linspace(0, 2 * np.pi * 1.08, 60)
    ax, ay = 42 + rng.integers(-4, 5), (row[1] - row[0]) * 0.36
    pts = np.stack([cx + ax * np.cos(t) + rng.normal(0, 1.5, t.size),
                    cy + ay * np.sin(t) + rng.normal(0, 1.5, t.size)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, INK, 3, cv2.LINE_AA)


def signature(img, x0, y0):
    t = np.linspace(0, 1, 80)
    xs = x0 + 260 * t
    ys = y0 + 25 * np.sin(14 * t) + 15 * np.cos(31 * t)
    cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, INK, 3, cv2.LINE_AA)


def fill_like_sample(blank):
    """Reproduces the real photo: lots of mistakes."""
    img = blank.copy()
    # Section A: left blank (as in the sample)
    circle(img, "No", B_ROWS[0])                                  # B1 No, no details
    # B2: nothing visible - hidden under a blue callout later
    # B3: no response
    circle(img, "Yes", B_ROWS[3]); write(img, "Tally", 1350, B_ROWS[3][0] + 60)     # B4 Yes + details
    circle(img, "NA", B_ROWS[4]); write(img, "need label", 1340, B_ROWS[4][0] + 60)  # B5 NA + details
    signature(img, 420, 1165)                                     # B signed, no date
    circle(img, "No", C_ROWS[0])                                  # C1 No, no details
    circle(img, "NA", C_ROWS[1]); write(img, "Seal broken", 1338, C_ROWS[1][0] + 48, 0.95)
    # C3 no response
    for i in (3, 4, 5, 6, 7):
        circle(img, "Yes", C_ROWS[i])
    write(img, "chem spills", 1340, C_ROWS[4][0] + 48, 0.95)     # C5 Yes + details
    # C sign/date blank
    # blue annotation callout over B2's response cells (like "Page 1 Region 2")
    cv2.rectangle(img, (990, 715), (1570, 795), (230, 40, 30), -1)
    cv2.putText(img, "2. Page 1 Region 2  Category: Response", (1000, 750), FONT, 0.5, (255, 255, 255), 1)
    return img


def fill_correct(blank):
    img = blank.copy()
    write(img, "MSKU 4471203", 520, 280); write(img, "SL-88213", 520, 390)
    write(img, "INV-2210-77", 1215, 280); write(img, "21/10/26", 1230, 390)
    for i, r in enumerate(B_ROWS):
        circle(img, "Yes", r)
    signature(img, 420, 1165); write(img, "21/10/26", 1150, 1180)
    for i, r in enumerate(C_ROWS):
        if i == 2:
            circle(img, "No", r); write(img, "small gap, taped", 1338, r[0] + 48, 0.8)
        elif i == 1:
            circle(img, "NA", r)
        else:
            circle(img, "Yes", r)
    signature(img, 420, 2085); write(img, "21/10/26", 1150, 2100)
    return img


def photo_distort(img, strength=1.0):
    h, w = img.shape[:2]
    pad = 160
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 35, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = img
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    j = 60 * strength
    dst = src + rng.uniform(-j, j, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=(35, 35, 35))
    yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]]
    shade = 0.82 + 0.18 * (xx / out.shape[1]) * (1 - 0.3 * yy / out.shape[0])
    out = np.clip(out * shade[..., None] + rng.normal(0, 6 * strength, out.shape), 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# Ground truth for the "sample-like" page: (item_id, check) -> expected status
EXPECTED = {
    ("A.ref_no", "field_filled"): "FAIL", ("A.lot_no", "field_filled"): "FAIL",
    ("A.order_no", "field_filled"): "FAIL", ("A.date", "date_present"): "FAIL",
    ("B1", "response"): "PASS", ("B1", "details"): "FAIL",
    ("B2", "response"): "REVIEW",
    ("B3", "response"): "FAIL",
    ("B4", "response"): "PASS", ("B4", "details"): "FAIL",
    ("B5", "response"): "PASS", ("B5", "details"): "FAIL",
    ("B.sign", "signature_present"): "PASS", ("B.date", "date_present"): "FAIL",
    ("C1", "response"): "PASS", ("C1", "details"): "FAIL",
    ("C2", "response"): "PASS", ("C2", "details"): "FAIL",
    ("C3", "response"): "FAIL",
    ("C4", "details"): "PASS", ("C5", "details"): "FAIL", ("C6", "details"): "PASS",
    ("C7", "details"): "PASS", ("C8", "details"): "PASS",
    ("C.sign", "signature_present"): "FAIL", ("C.date", "date_present"): "FAIL",
    ("rule:A_B_before_C", "complete_before"): "FAIL",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blank = make_blank()
    cv2.imwrite(str(OUT / "blank_p1.png"), blank)
    dump_yaml(make_template_yaml("blank_p1.png"), OUT / "template.yaml",
              "# Auto-generated by synthetic_demo.py\n")

    sample = photo_distort(fill_like_sample(blank), 1.0)
    cv2.imwrite(str(OUT / "filled_like_sample_photo.jpg"), sample)

    # A 3-page "scanned" PDF: correct form, sample-like form, unrelated page
    junk = np.full((H, W, 3), 255, np.uint8)
    for k in range(30):
        txt(junk, f"Unrelated delivery note line {k}: {rng.integers(1e6)}", 120, 120 + 65 * k)
    pages = [photo_distort(fill_correct(blank), 0.3), photo_distort(fill_like_sample(blank), 0.3),
             photo_distort(junk, 0.3)]
    pil = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]
    pil[0].save(OUT / "batch_scan.pdf", "PDF", resolution=200, save_all=True, append_images=pil[1:])

    auditor = Auditor([load_template(OUT / "template.yaml")])
    rows, _, _ = auditor.audit_file(OUT / "filled_like_sample_photo.jpg", OUT / "audit_out", debug=True)
    _, summ, _ = auditor.audit_file(OUT / "batch_scan.pdf", OUT / "audit_out")

    got = {(r["item_id"], r["check"]): r["status"] for r in rows}
    bad = [(k, v, got.get(k)) for k, v in EXPECTED.items() if got.get(k) != v]
    print("\nGround-truth check on photo-like sample:",
          "ALL OK" if not bad else f"{len(bad)} mismatch(es)")
    for k, exp, g in bad:
        print(f"   {k}: expected {exp}, got {g}")
    exp_pages = ["PASS", "FAIL", "UNRECOGNISED"]
    got_pages = [s["page_status"] for s in summ]
    print("PDF page statuses:", got_pages, "OK" if got_pages == exp_pages else f"expected {exp_pages}")
    return 0 if not bad and got_pages == exp_pages else 1


if __name__ == "__main__":
    sys.exit(main())
