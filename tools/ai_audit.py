"""The "just ask an AI" baseline, with the same inputs and outputs as the form auditor.

The AI gets the PDF's pages as images, the same checklist the auditor works through
(every item of your template, by the same item ids and check names) and the same rule
texts (your rules CSV), and is asked for a verdict on each check: what it sees, PASS /
FAIL / REVIEW, why, and where on the page. Its answers are written in the auditor's
format, so the two can be compared row by row (tools/compare_results.py):

    <out>/ai_findings.csv                     one row per check, as all_findings.csv
    <out>/<file>/<file>_ai_audited.pdf        pages with the AI's numbered boxes
    <out>/<file>/<file>_ai_response.json      the AI's raw answer

Providers:

    --provider azure-openai   Azure OpenAI (a vision model such as gpt-4o / gpt-4.1);
                              .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,
                              AZURE_OPENAI_DEPLOYMENT
    --provider anthropic      Claude through the Anthropic API; .env: ANTHROPIC_API_KEY
    --provider manual         for a chat window (Copilot, Claude...): writes the prompt
                              and the page images; you paste them in, save the answer as
                              <file>_ai_answer.json, then run again with --import-answers

    python tools/ai_audit.py form1.pdf form2.pdf -o ai_out --provider azure-openai
    python tools/ai_audit.py form1.pdf -o ai_out --provider manual
    python tools/ai_audit.py form1.pdf -o ai_out --import-answers
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from form_auditor.env import load_env                     # noqa: E402
from form_auditor.io_utils import iter_input_files, load_pages   # noqa: E402
from form_auditor.report import save_pdf, write_csv       # noqa: E402
from form_auditor.template import FormTemplate, load_template    # noqa: E402

PRESENCE_CHECK = {"signature": "signature_present", "date": "date_present", "time": "time_present",
                  "number": "value_present", "text": "field_filled", "field": "field_filled"}
AI_COLUMNS = ["file", "page", "form_no", "template", "marker", "section", "item_id", "item_label",
              "check", "status", "observed", "reason", "value_read", "rule_ref", "confidence",
              "ai_model", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "bbox_units"]
MAX_SIDE = 2000                      # px, long side of each image sent


# --------------------------------------------------------------------------- #
# The checklist: exactly the checks the auditor makes, by the same ids
# --------------------------------------------------------------------------- #
def checklist(tpl: FormTemplate) -> list[dict]:
    out = []
    for pno, page in enumerate(tpl.pages, start=1):
        for sec in page.sections:
            for it in sec.items:
                base = {"page": pno, "section": sec.id, "item_id": it.id, "label": it.label}
                if it.type == "choice":
                    printed = [o for o in it.options if o in it.printed] or list(it.options)
                    refs = list(dict.fromkeys(list(it.rule_ref_map.values()) + it.rule_refs))
                    e = {**base, "check": "response", "options_printed": printed,
                         "answer_required": it.required, "rules": refs}
                    flag = list(dict.fromkeys(it.fail_when_selected + it.flag_when_selected))
                    if flag:
                        e["FAIL_if_answer"] = flag
                    out.append(e)
                    if it.details is not None:
                        d = {**base, "check": "details", "label": f"{it.label} - details column",
                             "rules": list(it.details.rule_refs)}
                        if it.details.must_be_blank_when:
                            d["FAIL_if_written_when_answer_is"] = list(it.details.must_be_blank_when)
                        if it.details.required_when:
                            d["FAIL_if_blank_when_answer_is"] = list(it.details.required_when)
                        if it.details.read:
                            d["read_the_writing"] = True
                        out.append(d)
                elif it.type == "checkbox":
                    out.append({**base, "check": "na_checkbox", "rules": list(it.rule_refs)})
                else:
                    e = {**base, "check": PRESENCE_CHECK[it.type], "required": it.required,
                         "rules": list(it.rule_refs)}
                    if it.read and not it.presence_only:
                        e["read_the_writing"] = True
                        e["NA_allowed"] = it.allow_na
                    out.append(e)
        for r in page.rules:
            if r.get("type") == "block":
                out.append({"page": pno, "section": page.sections[-1].id if page.sections else "",
                            "item_id": str(r.get("item")), "check": "item_in_use",
                            "label": r.get("description", ""), "parts": list(r.get("items", [])),
                            "rules": list(r.get("rule_ref", []))})
    for r in tpl.document_rules:
        if r.get("requires_setting") and not getattr(tpl.settings, r["requires_setting"], False):
            continue                                  # switched off, as in the auditor
        out.append({"page": None, "section": "PAGE", "item_id": f"rule:{r.get('id')}",
                    "check": "cross_check", "label": r.get("description", ""),
                    "rules": list(r.get("rule_ref", []))})
    for e in out:
        e["key"] = f"{e['item_id']}|{e['check']}"
    return out


def build_prompt(tpl: FormTemplate, items: list[dict], images: list[dict], as_pdf: bool = False) -> str:
    used = sorted({r for e in items for r in e["rules"]})
    def plain(t: str) -> str:                       # drop notes about the auditor's own options
        return re.sub(r"\s*\([^)]*--[a-z][^)]*\)", "", t)
    rules = "\n".join(f"- {r}: {plain(tpl.rules[r].instruction)}" for r in used if r in tpl.rules)
    na_rules = getattr(tpl.settings, "na_checkbox_rules", False)
    pages = "\n".join(f"- image {i + 1}: page {im['page']}" + (f" ({im['part']})" if im["part"] else "")
                      for i, im in enumerate(images))
    lines = [json.dumps({k: v for k, v in e.items() if k != "key"} | {"key": e["key"]}, ensure_ascii=False)
             for e in items]
    if as_pdf:
        intro = ("The attached PDF is the form, scanned or photographed. Look at each page as an image\n"
                 "(the handwriting and pen marks matter; any text layer does not). PDF page N is form\n"
                 "page N; in your answer \"image\" means the PDF page number.")
    else:
        intro = ("The images are pages of it (\"page\" in the checklist = the form's page number):\n" + pages)
    return f"""You are auditing a filled-in paper form ({tpl.name or tpl.form_id}), scanned or photographed.
{intro}

Check EVERY entry of the checklist below against the rules. Do not summarise and do not skip
entries: answer each one, by its "key".

How to read the form:
- "response": which printed option (one of options_printed) is marked - circled or ticked.
  An option circled and then crossed out does not count. An answer written in, or marked in
  an empty cell, is not a printed option. Two or more options marked (two circles, or one
  circle round two words): FAIL. A mark you cannot place: REVIEW. A circle that crosses a
  cell or row line but surrounds one word, touching no other printed option, counts for it.
- "details": whether anything is written in that row's details / follow-up column, and what.
- "*_present" / "field_filled": whether the box is filled in, and what is written.
- "na_checkbox": whether the NA box is ticked.{'' if na_rules else ' These NA boxes are recorded only: report ticked or blank, with status SKIP.'}
- "item_in_use": the item with sub-items listed in "parts": apply its rule to those parts.
- "cross_check": apply the rule named.

Fields on a checklist entry, where present, settle the rule for that entry:
- "answer_required" / "required": true = a blank is FAIL.
- "FAIL_if_answer": these answers are FAIL (otherwise a clear printed answer is PASS).
- "FAIL_if_written_when_answer_is": details written next to these answers is FAIL.
- "FAIL_if_blank_when_answer_is": blank details next to these answers is FAIL.
- "read_the_writing": put the handwriting in value_read; "NA_allowed": false = a written NA is FAIL.
- An option circled where it is not in options_printed (e.g. NA in an empty cell) is FAIL.

Rules:
{rules}

Statuses: PASS = the rule is met; FAIL = it is broken; REVIEW = you cannot tell (unclear,
faint, hidden, unreadable) - say REVIEW rather than guess; SKIP = the check does not apply.

Checklist (one JSON object per line):
{chr(10).join(lines)}

Answer with ONE JSON object and nothing else:
{{"findings": [{{"key": "<key>", "observed": "<what you see: the option marked / filled / blank / ticked ...>",
  "value_read": "<handwriting read, if any>", "status": "PASS|FAIL|REVIEW|SKIP",
  "reason": "<short reason when not PASS>", "image": <image number the item is on>,
  "bbox": [x0, y0, x1, y1] (the problem area as fractions 0-1 of that image, for FAIL / REVIEW; else null),
  "confidence": <0-1>}}, ...]}}
One finding per checklist key, in the same order."""


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def page_images(form: list[dict], halves: bool) -> list[dict]:
    """form: [{file_page, tpage, img}] -> [{tpage, file_page, part, y0, y1 (page fractions), jpeg}]"""
    out = []
    for pg in form:
        img = pg["img"]
        parts = [("", 0.0, 1.0)]
        if halves:                                   # overlapping halves: more pixels per word
            parts = [("top half", 0.0, 0.55), ("bottom half", 0.45, 1.0)]
        h = img.shape[0]
        for part, f0, f1 in parts:
            crop = img[int(f0 * h):int(f1 * h)]
            k = min(1.0, MAX_SIDE / max(crop.shape[:2]))
            if k < 1:
                crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append({"page": pg["tpage"], "file_page": pg["file_page"], "part": part,
                        "y0": f0, "y1": f1, "jpeg": buf.tobytes(),
                        "w": img.shape[1], "h": img.shape[0]})
    return out


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _post(url: str, headers: dict, body: dict, timeout: int = 600) -> dict:
    data = json.dumps(body).encode()
    delay = 5.0
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:600]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 5:
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
    raise RuntimeError("too many retries")


def ask_azure_openai(prompt: str, images: list[dict]) -> tuple[str, str]:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not (endpoint and key and model):
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT (e.g. in .env)")
    content = [{"type": "text", "text": prompt}]
    for im in images:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(im["jpeg"]).decode(), "detail": "high"}})
    body = {"model": model, "messages": [
                {"role": "system", "content": "You are a meticulous auditor of paper forms. Answer only with JSON."},
                {"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, "temperature": 0,
            "max_completion_tokens": 16000}
    url = f"{endpoint}/openai/v1/chat/completions"
    for _ in range(3):                    # some models refuse one of the optional parameters
        try:
            res = _post(url, {"api-key": key}, body)
            break
        except RuntimeError as e:
            msg = str(e)
            if "HTTP 400" in msg and "temperature" in msg and "temperature" in body:
                body.pop("temperature")
            elif "HTTP 400" in msg and "max_completion_tokens" in msg and "max_completion_tokens" in body:
                body["max_tokens"] = body.pop("max_completion_tokens")
            elif "HTTP 400" in msg and "response_format" in msg and "response_format" in body:
                body.pop("response_format")
            else:
                raise
    else:
        raise RuntimeError("Azure OpenAI refused the request")
    choice = res["choices"][0]
    if choice.get("finish_reason") == "length":
        print("  warning: the answer was cut off (max tokens) - some checks will be missing")
    return choice["message"]["content"], f"azure-openai:{model}"


def ask_anthropic(prompt: str, images: list[dict], model: str) -> tuple[str, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY (e.g. in .env)")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(im["jpeg"]).decode()}}
               for im in images]
    content.append({"type": "text", "text": prompt})
    res = _post("https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": 16000, "temperature": 0,
                 "messages": [{"role": "user", "content": content}]})
    return "".join(b.get("text", "") for b in res.get("content", [])), f"anthropic:{model}"


def parse_answer(text: str) -> list[dict]:
    """The findings list from an answer that may carry code fences or chatter."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the answer")
    data = json.loads(t[a:b + 1])
    return data.get("findings", data if isinstance(data, list) else [])


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def to_rows(file_name: str, form_no: int, tpl: FormTemplate, items: list[dict],
            answers: list[dict], images: list[dict], model: str) -> list[dict]:
    by_key = {str(a.get("key")): a for a in answers if isinstance(a, dict)}
    file_page = {im["page"]: im["file_page"] for im in images}
    rows, marker = [], 0
    for e in items:
        a = by_key.get(e["key"])
        st = str((a or {}).get("status", "MISSING")).upper()
        st = st if st in ("PASS", "FAIL", "REVIEW", "SKIP") else ("MISSING" if a is None else "REVIEW")
        row = {"file": file_name, "form_no": form_no, "template": tpl.display_name,
               "section": e["section"], "item_id": e["item_id"], "item_label": e["label"],
               "check": e["check"], "status": st, "rule_ref": "; ".join(e["rules"]),
               "observed": (a or {}).get("observed", ""),
               "reason": (a or {}).get("reason", "") if a else "the AI gave no answer for this check",
               "value_read": (a or {}).get("value_read", "") or "",
               "confidence": (a or {}).get("confidence", ""), "ai_model": model,
               "page": file_page.get(e["page"], "")}
        img_no, box = (a or {}).get("image"), (a or {}).get("bbox")
        if st in ("FAIL", "REVIEW", "MISSING"):
            marker += 1
            row["marker"] = marker
        try:
            ok = st in ("FAIL", "REVIEW") and len(box) == 4 and 1 <= int(img_no) <= len(images)
        except (TypeError, ValueError):
            ok = False
        if ok:
            im = images[int(img_no) - 1]
            w, h = im["w"], im["h"]
            x0, y0, x1, y1 = (float(v) for v in box)
            if max(x0, y0, x1, y1) > 1.5:            # answered in pixels of the image sent
                crop_h = h * (im["y1"] - im["y0"])
                k = min(1.0, MAX_SIDE / max(w, crop_h))
                sw, sh = w * k, crop_h * k
                x0, x1, y0, y1 = x0 / sw, x1 / sw, y0 / sh, y1 / sh
            x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
            y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
            span = im["y1"] - im["y0"]
            row.update(page=im["file_page"], bbox_units="px",
                       bbox_x0=round(x0 * w), bbox_x1=round(x1 * w),
                       bbox_y0=round((im["y0"] + y0 * span) * h), bbox_y1=round((im["y0"] + y1 * span) * h))
        rows.append(row)
    return rows


def annotate(form: list[dict], rows: list[dict], model: str) -> list[np.ndarray]:
    out = []
    for pg in form:
        im, fp = pg["img"].copy(), pg["file_page"]
        unit = max(im.shape[:2]) / 2000.0
        th_box, th_txt = max(2, int(3 * unit)), max(1, int(2 * unit))
        for r in (r for r in rows if r.get("page") == fp and r.get("bbox_units")):
            colour = (40, 40, 220) if r["status"] == "FAIL" else (0, 140, 255)
            p0, p1 = (int(r["bbox_x0"]), int(r["bbox_y0"])), (int(r["bbox_x1"]), int(r["bbox_y1"]))
            cv2.rectangle(im, p0, p1, colour, th_box)
            label = f"#{r['marker']} {r['item_id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit, th_txt)
            ty = max(p0[1], th + 8)
            cv2.rectangle(im, (p0[0], ty - th - 8), (p0[0] + tw + 8, ty), colour, -1)
            cv2.putText(im, label, (p0[0] + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit,
                        (255, 255, 255), th_txt, cv2.LINE_AA)
        mine = [r for r in rows if r.get("page") == fp]
        n_f = sum(r["status"] == "FAIL" for r in mine)
        n_r = sum(r["status"] in ("REVIEW", "MISSING") for r in mine)
        head = f"AI audit ({model}) | page {fp} (form page {pg['tpage']}) | {n_f} fail, {n_r} review"
        bar = np.full((int(44 * unit) + 10, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, head, (10, int(30 * unit) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * unit,
                    (40, 40, 40), th_txt, cv2.LINE_AA)
        out.append(np.vstack([bar, im]))
    return out


# --------------------------------------------------------------------------- #
# Which form page is each image? (so the AI is asked about the right checks)
# --------------------------------------------------------------------------- #
class PageSorter:
    """Groups a file's pages into forms the same way the auditor does, so both are asked
    about the same pages. identify=True uses the auditor's page matcher (local, no AI call:
    it only picks which part of the checklist to send); False takes the pages in order."""

    def __init__(self, templates: list[FormTemplate], identify: bool):
        self.templates, self.auditor = templates, None
        if identify:
            from form_auditor.pipeline import Auditor
            self.auditor = Auditor(templates, verbose=False)

    def forms(self, pages: list[np.ndarray]) -> list[tuple[FormTemplate, list[dict]]]:
        if self.auditor is None:
            tpl = next((t for t in self.templates if len(pages) % len(t.pages) == 0), self.templates[0])
            n = len(tpl.pages)
            return [(tpl, [{"file_page": i + j + 1, "tpage": j + 1, "img": pages[i + j]}
                           for j in range(min(n, len(pages) - i))]) for i in range(0, len(pages), n)]
        out, cur, prev_tpl, prev_no = [], [], None, 0
        for i, img in enumerate(pages, start=1):
            rt, _ = self.auditor.identify(img, prefer=prev_tpl)
            if rt is None:
                print(f"  page {i}: not recognised as a page of the form - not sent")
                continue
            no = rt.template.pages.index(rt.page) + 1
            if cur and (rt.template is not prev_tpl or no <= prev_no):
                out.append((prev_tpl, cur))
                cur = []
            cur.append({"file_page": i, "tpage": no, "img": img})
            prev_tpl, prev_no = rt.template, no
        if cur:
            out.append((prev_tpl, cur))
        return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    from form_auditor.api import default_templates
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="PDF / image files or folders")
    ap.add_argument("-o", "--out", default="ai_out")
    ap.add_argument("--provider", choices=["azure-openai", "anthropic", "manual"], default="azure-openai")
    ap.add_argument("--import-answers", action="store_true",
                    help="read <out>/<file>/<file>_ai_answer.json saved from a chat window")
    ap.add_argument("--model", default="claude-sonnet-5-5", help="model for --provider anthropic")
    ap.add_argument("-t", "--template", nargs="+", default=None, help="default: templates/")
    ap.add_argument("--halves", action="store_true",
                    help="send each page as two overlapping halves (more detail per word)")
    ap.add_argument("--no-identify", action="store_true",
                    help="take the pages in order (p1, p2, p3, p1...) instead of recognising them")
    ap.add_argument("--attach-pdf", action="store_true",
                    help="--provider manual: the prompt is for attaching the PDF itself, not page images")
    ap.add_argument("--env-file", default=None)
    a = ap.parse_args(argv)
    load_env(a.env_file)
    tpls = [load_template(Path(t)) for t in (a.template or default_templates())]
    sorter = PageSorter(tpls, identify=not a.no_identify)
    out = Path(a.out)
    all_rows = []
    for f in iter_input_files([Path(p) for p in a.inputs]):
        pages = [pg.image for pg in load_pages(f, dpi=200)]
        forms = sorter.forms(pages)
        fdir = out / f.stem
        fdir.mkdir(parents=True, exist_ok=True)
        rows_f, annotated = [], []
        for fi, (tpl, form) in enumerate(forms, start=1):
            shown = {pg["tpage"] for pg in form}
            items = [e for e in checklist(tpl) if e["page"] in shown
                     or (e["page"] is None and len(shown) == len(tpl.pages))]
            tag = f.stem if len(forms) == 1 else f"{f.stem}_form{fi}"
            images = page_images(form, a.halves)
            as_pdf = a.attach_pdf and not a.halves and len(forms) == 1 \
                and all(pg["file_page"] == pg["tpage"] == i for i, pg in enumerate(form, start=1))
            if a.attach_pdf and not as_pdf and not a.import_answers:
                print(f"  {f.name}: pages are not one form in page order - writing page images instead")
            prompt = build_prompt(tpl, items, images, as_pdf)
            if a.import_answers:
                ans = fdir / f"{tag}_ai_answer.json"
                if not ans.is_file():
                    print(f"  {f.name}: no {ans} - run with --provider manual first")
                    continue
                text, model = ans.read_text(encoding="utf-8"), "chat (pasted)"
            elif a.provider == "manual":
                (fdir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
                if as_pdf:
                    what = f"attach {f.name}"
                else:
                    for i, im in enumerate(images, start=1):
                        (fdir / f"{tag}_image{i}.jpg").write_bytes(im["jpeg"])
                    what = f"attach {tag}_image1..{len(images)}.jpg (in order)"
                print(f"{f.name}" + (f" form {fi}" if len(forms) > 1 else "") +
                      f": {what} to a new chat, paste {fdir / (tag + '_prompt.txt')}, and save the "
                      f"reply as {fdir / (tag + '_ai_answer.json')}")
                continue
            else:
                print(f"Asking the AI about {f.name}" + (f" (form {fi})" if len(forms) > 1 else "") + " ...")
                t0 = time.time()
                if a.provider == "azure-openai":
                    text, model = ask_azure_openai(prompt, images)
                else:
                    text, model = ask_anthropic(prompt, images, a.model)
                print(f"  answered in {time.time() - t0:.0f} s")
                (fdir / f"{tag}_ai_response.json").write_text(text, encoding="utf-8")
            try:
                answers = parse_answer(text)
            except Exception as e:
                print(f"  {f.name}: could not read the AI's answer as JSON ({e}); it is saved as is")
                continue
            rows = to_rows(f.name, fi, tpl, items, answers, images, model)
            rows_f += rows
            annotated += annotate(form, rows, model)
            missing = sum(r["status"] == "MISSING" for r in rows)
            print(f"  {f.name}: {len(rows)} checks - {sum(r['status'] == 'FAIL' for r in rows)} fail, "
                  f"{sum(r['status'] == 'REVIEW' for r in rows)} review"
                  + (f", {missing} not answered by the AI" if missing else ""))
        if rows_f:
            write_csv(fdir / f"{f.stem}_ai_audit.csv", rows_f, AI_COLUMNS)
            save_pdf(annotated, fdir / f"{f.stem}_ai_audited.pdf")
            all_rows += rows_f
    if a.provider == "manual" and not a.import_answers:
        print("\nThen run the same command again with --import-answers instead of --provider manual.")
    if all_rows:
        write_csv(out / "ai_findings.csv", all_rows, AI_COLUMNS)
        print(f"\nWrote {out / 'ai_findings.csv'} - compare with: python tools/compare_results.py "
              f"<auditor output folder> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
