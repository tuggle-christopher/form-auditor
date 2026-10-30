"""Build a template (YAML + one image per page) from the BLANK form PDF.

    python tools/build_template_from_pdf.py forms/blank.pdf
    python tools/build_template_from_pdf.py forms/blank_v2.pdf --out templates/form_v2.yaml --version 2

What is specific to your form (its id, the header fields, which rows the flag rules
cover, cross-checks between items) is read from a config file, by default
templates/form_config.yaml - kept out of git like the rest of templates/. See
tools/form_config.example.yaml.

Everything else is derived from the PDF itself, no manual drawing:
  * table cells come from the ruled lines of the rendered page;
  * section membership from the "Section X - Title" headings;
  * answer rows from the printed "Yes  No [NA]" words in the text layer (also "Ye/s"
    split over two lines); the exact box of each printed word is stored (used to tell
    which word a circle encloses). Rows without a third cell get only Yes / No;
  * a "... (HH:MM):" line becomes a time field (two lines in one cell: .start / .end),
    a "______ °C" line a number field, a "[ ] NA" box on a heading row a checkbox;
  * sub-items "n.1) ... n.k)" are grouped under their heading row "n)" (a block with its
    NA checkbox), see `rules:` in the generated YAML;
  * "Sign ... Date" rows become signature + date fields;
  * every field cites the rule(s) it checks in the rules CSV (rule_ref).

After building, the blank PDF is audited against the new template as a self-check
(every page must be recognised, and no box may read as filled on a blank form), and the
items are compared with any other template in the same folder (what was added /
removed), so a new version of the form can be checked before it is used.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from form_auditor.calibrate import dump_yaml  # noqa: E402
from form_auditor.template import read_yaml  # noqa: E402

DEFAULT_CONFIG = ROOT / "templates" / "form_config.yaml"
CFG: dict = {}                                 # the loaded config (see load_config)

DPI = 200
SIGN_OVERRIDES = {"fill_lo": 0.0015, "fill_hi": 0.003}


# --------------------------------------------------------------------------- #
def render(pdf_path: Path, page_no: int):
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_no - 1]
    pw, ph = page.get_size()
    img = page.render(scale=DPI / 72, draw_annots=False).to_pil()
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    k = DPI / 72

    tp = page.get_textpage()
    words, cur, box = [], "", None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch.strip():
            if cur:
                words.append((cur, box))
            cur, box = "", None
            continue
        l, b, r, t = tp.get_charbox(i)
        cb = (l * k, (ph - t) * k, r * k, (ph - b) * k)       # -> pixel coords, y down
        if box is not None and abs((cb[1] + cb[3]) / 2 - (box[1] + box[3]) / 2) > 0.8 * (box[3] - box[1]):
            # new line without a separating space (compare vertical centres, so that
            # hyphens and commas, which sit higher / lower, stay in the word)
            words.append((cur, box))
            cur, box = "", None
        cur += ch
        box = cb if box is None else (min(box[0], cb[0]), min(box[1], cb[1]),
                                      max(box[2], cb[2]), max(box[3], cb[3]))
    if cur:
        words.append((cur, box))
    pdf.close()
    return bgr, _join_split_words(words)


def _join_split_words(words):
    """'Ye' + 's' printed on two lines of a narrow cell -> one word 'Yes' (union box)."""
    out, skip = [], set()
    for i, (t, b) in enumerate(words):
        if i in skip:
            continue
        if t == "Ye":
            for j in range(i + 1, min(i + 6, len(words))):
                t2, b2 = words[j]
                cx = (b2[0] + b2[2]) / 2
                if t2 == "s" and b[0] - 2 <= cx <= b[2] + 2 and 0 <= b2[1] - b[3] < 30:
                    b = (min(b[0], b2[0]), b[1], max(b[2], b2[2]), b2[3])
                    t = "Yes"
                    skip.add(j)
                    break
        out.append((t, b))
    return out


def grid_lines(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = (g < 128).astype(np.uint8)
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)))
    # 30 px: short enough for narrow one-line rows (~33 px), longer than any printed glyph
    vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))

    def segs(mask, horiz):
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for x, y, w, h, _ in st[1:]:
            out.append(dict(pos=float(y + h / 2 if horiz else x + w / 2),
                            lo=int(x if horiz else y), hi=int(x + w if horiz else y + h)))
        return sorted(out, key=lambda d: d["pos"])
    H, V = segs(hl, True), segs(vl, False)
    # underscores of a printed "________ °C" line are text, not table rules
    H = [l for l in H if l["hi"] - l["lo"] > 400]
    return H, V


def h_above(H, x, y):
    c = [l for l in H if l["lo"] - 3 <= x <= l["hi"] + 3 and l["pos"] < y]
    return max(c, key=lambda l: l["pos"])["pos"]


def h_below(H, x, y):
    c = [l for l in H if l["lo"] - 3 <= x <= l["hi"] + 3 and l["pos"] > y]
    return min(c, key=lambda l: l["pos"])["pos"]


def v_left(V, x, y):
    c = [l for l in V if l["lo"] - 3 <= y <= l["hi"] + 3 and l["pos"] < x]
    return max(c, key=lambda l: l["pos"])["pos"]


def v_right(V, x, y):
    c = [l for l in V if l["lo"] - 3 <= y <= l["hi"] + 3 and l["pos"] > x]
    return min(c, key=lambda l: l["pos"])["pos"]


def v_between(V, x0, x1, y):
    return sorted(l["pos"] for l in V if l["lo"] - 3 <= y <= l["hi"] + 3 and x0 - 2 <= l["pos"] <= x1 + 2)


def cy_of(b):
    return (b[1] + b[3]) / 2


def load_config(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"Config not found: {path}\nCopy tools/form_config.example.yaml there and "
                 f"fill it in for your form.")
    c = read_yaml(path) or {}
    c.setdefault("form_id", "FORM")
    c.setdefault("name", "")
    c.setdefault("output_stem", "form")
    c.setdefault("rules_csv", f"{c['output_stem']}_rules.csv")
    c.setdefault("header_section", "A")
    c.setdefault("header_fields", [])
    c.setdefault("choice_defaults", {"flag_when_selected": ["No"],
                                     "details": {"required_when": [], "must_be_blank_when": ["Yes", "NA"]}})
    c.setdefault("flag_rows_upto", {})
    c.setdefault("cross_checks", [])
    c.setdefault("labels", {})
    for k, v in {"section": "Section", "sign": "Sign", "date": "Date", "response": "Response"}.items():
        c["labels"].setdefault(k, v)
    return c


# --------------------------------------------------------------------------- #
# Rule ids (see the rules CSV). Convention, per section X:
#   X-1 answer required, X-2 flagged answer (e.g. any No), X-3 details must be blank,
#   X-4 sign-off, X-<n> item n with sub-items (a block); G-1 only printed options,
#   G-2 how answers are read. Header fields cite what the config gives them.
def refs(sid: str, kind: str, block: str | None = None) -> list[str]:
    if kind == "response":
        # which rule a response finding cites depends on the outcome (see rules.audit_choice)
        return {"answer": f"{sid}-{block}" if block else f"{sid}-1", "flag": f"{sid}-2",
                "options": "G-1", "reading": "G-2"}
    elif kind == "details":
        r = [f"{sid}-3"]
    elif kind == "signoff":
        return [f"{sid}-4"]
    else:                                   # time / number inside a block
        r = []
    if block:
        r = [f"{sid}-{block}"] + r
    return r


def page_template(pdf: Path, page_no: int, out_dir: Path, stem: str) -> dict:
    bgr, words = render(pdf, page_no)
    h, w = bgr.shape[:2]
    H, V = grid_lines(bgr)
    pad = 4

    def nbox(x0, y0, x1, y1, inset=pad):
        return [round(float(x0 + inset) / w, 4), round(float(y0 + inset) / h, 4),
                round(float(x1 - inset) / w, 4), round(float(y1 - inset) / h, 4)]

    def wbox(b):
        return [round(float(v) / d, 4) for v, d in zip(b, (w, h, w, h))]

    # ---------------- sections from headings ---------------------------- #
    heads, titles = [], {}
    for i, (t, b) in enumerate(words):
        if t == CFG["labels"]["section"] and i + 2 < len(words) \
                and re.fullmatch(r"[A-Z]", words[i + 1][0]) and words[i + 2][0] in ("–", "-", "—"):
            sid = words[i + 1][0]
            heads.append((b[1], sid))
            rest = []
            for t2, b2 in words[i + 3:]:
                if abs(b2[1] - b[1]) > 8 or t2 == CFG["labels"]["response"]:
                    break
                rest.append(t2)
            titles[sid] = " ".join(rest)
    heads.sort()

    def section_at(y):
        s = [sid for hy, sid in heads if hy <= y + 2]
        return s[-1] if s else None

    def row_text(top, bot, x_right):
        q = [t2 for t2, b2 in words if b2[2] < x_right and top < cy_of(b2) < bot]
        txt = " ".join(q)
        m = re.match(r"^(\d+(?:\.\d+)?)\)\s*", txt)
        return (m.group(1) if m else None), re.sub(r"^\d+(?:\.\d+)?\)\s*", "", txt).strip()

    items_by_sec: dict[str, list[tuple[float, dict]]] = {}
    blocks: dict[tuple[str, str], dict] = {}          # (section, "5") -> block info

    def add(sid, y, item):
        items_by_sec.setdefault(sid, []).append((y, item))

    # ---------------- choice rows from printed Yes / No ----------------- #
    for t, b in words:
        if t != "Yes":
            continue
        cy = cy_of(b)
        same_line = [(t2, b2) for t2, b2 in words if abs(cy_of(b2) - cy) < 8 and b2[0] > b[2]]
        no = [x for x in same_line if x[0] == "No" and x[1][0] - b[2] < 160]
        if not no:
            continue                                  # "Yes or NA" in a header etc.
        no = no[0]
        na = [x for x in same_line if x[0] == "NA" and x[1][0] - no[1][2] < 160]
        cx = (b[0] + b[2]) / 2
        top, bot = h_above(H, cx, cy), h_below(H, cx, cy)
        xl = v_left(V, cx, cy)
        xr = max(l["pos"] for l in V if l["lo"] - 3 <= cy <= l["hi"] + 3)   # table edge
        cols = v_between(V, xl, xr, cy)
        cells = list(zip(cols[:-1], cols[1:]))
        sid = section_at(cy)
        num, label = row_text(top, bot, xl)
        if sid is None or num is None or len(cells) < 3:
            continue
        opts = {"Yes": nbox(*(cells[0][0], top, cells[0][1], bot), 2),
                "No": nbox(*(cells[1][0], top, cells[1][1], bot), 2)}
        if len(cells) >= 4:
            opts["NA"] = nbox(cells[2][0], top, cells[2][1], bot, 2)
        det = cells[-1]
        printed = {"Yes": wbox(b), "No": wbox(no[1])}
        if na:
            printed["NA"] = wbox(na[0][1])
        block = num.split(".")[0] if "." in num else None
        it = {"id": f"{sid}{num}", "label": label, "type": "choice", "options": opts, "printed": printed}
        if "NA" in opts and not na:
            it["fail_when_selected"] = ["NA"]
        if block:
            it["required"] = False                    # decided by the block rule
        it["rule_ref"] = refs(sid, "response", block)
        it["details"] = {"box": nbox(det[0], top, det[1], bot), "read": True,
                         "rule_ref": refs(sid, "details", block)}
        upto = (CFG["flag_rows_upto"] or {}).get(sid)
        if upto and not block and num.isdigit() and int(num) > upto:
            it["flag_when_selected"] = []             # a normal answer on this row
            it["details"]["must_be_blank_when"] = []  # details not audited (recorded only)
            it["rule_ref"] = {k: v for k, v in it["rule_ref"].items() if k != "flag"}
            it["details"]["rule_ref"] = []
        add(sid, top, it)

    # ---------------- NA checkboxes ( [ ] NA on a heading row ) ---------- #
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for t, b in words:
        if t != "NA":
            continue
        cy = cy_of(b)
        top, bot = h_above(H, b[0], cy), h_below(H, b[0], cy)
        if any(t2 in ("Yes", "No") and top < cy_of(b2) < bot for t2, b2 in words):
            continue                                   # NA of a Yes/No/NA row
        sid = section_at(cy)
        num, label = row_text(top, bot, b[0] - 60)
        if sid is None or num is None:
            continue
        x0 = int(b[0]) - 70
        sub = (g[int(top) + 3:int(bot) - 3, x0:int(b[0]) - 2] < 160).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(sub)
        sq = [(x + x0, y + int(top) + 3, ww, hh) for x, y, ww, hh, a in st[1:]
              if 14 <= ww <= 40 and 14 <= hh <= 40 and abs(ww - hh) <= 6 and a < 0.6 * ww * hh]
        if not sq:
            print(f"  warning: no checkbox square found left of NA on row {num}")
            continue
        x, y, ww, hh = sq[-1]
        cid = f"{sid}{num}.NA"
        hit = (x - 12, y - 12, max(x + ww, b[2]) + 12, max(y + hh, b[3]) + 12)
        add(sid, top, {"id": cid, "label": f"{label} - NA checkbox", "type": "checkbox",
                       "box": nbox(x, y, x + ww, y + hh, 0), "hit_box": nbox(*hit, 0),
                       "rule_ref": [f"{sid}-{num}"]})
        blocks[(sid, num)] = {"label": label, "checkbox": cid}

    # ---------------- time fields: "<label> (HH:MM):" -------------------- #
    hms = [(t, b) for t, b in words if t.startswith("(HH:MM)")]
    for t, b in hms:
        cy = cy_of(b)
        top, bot = h_above(H, b[0], cy), h_below(H, b[0], cy)
        xl = v_left(V, b[0], cy)                     # left edge of the cell with the label
        xr = v_right(V, b[2], cy)
        sid = section_at(cy)
        num, _ = row_text(top, bot, xl)
        if sid is None or num is None:
            continue
        # one cell may hold several "(HH:MM)" lines: split it between them
        lines = sorted(cy_of(b2) for t2, b2 in hms if top < cy_of(b2) < bot and abs(b2[0] - b[0]) < 400)
        k = lines.index(cy)
        y0 = top if k == 0 else (lines[k - 1] + cy) / 2
        y1 = bot if k == len(lines) - 1 else (cy + lines[k + 1]) / 2
        kind = ("start", "end")[k] if len(lines) == 2 else f"t{k + 1}"
        text = " ".join(t2 for t2, b2 in words if abs(cy_of(b2) - cy) < 8 and xl < b2[0] < b[0])
        block = num.split(".")[0] if "." in num else None
        add(sid, top + k * 0.25,
            {"id": f"{sid}{num}.{kind}", "label": f"{num}) {text} (HH:MM)", "type": "time",
             "box": nbox(b[2] + 6, y0, xr, y1, 3), "read": True, "required": block is None,
             "rule_ref": refs(sid, "time", block)})

    # ---------------- number field: "______ °C" -------------------------- #
    for t, b in words:
        if not re.fullmatch(r"_{4,}", t):
            continue
        cy = b[1] - 10
        unit = [b2 for t2, b2 in words if t2 in ("oC", "°C", "ºC") and abs(b2[3] - b[3]) < 30 and b2[0] > b[2]]
        top, bot = h_above(H, b[0], cy), h_below(H, b[0], cy)
        xl = v_left(V, b[0], cy)
        sid = section_at(cy)
        num, label = row_text(top, bot, xl)
        if sid is None or num is None:
            continue
        x1 = unit[0][0] - 2 if unit else v_right(V, b[2], cy)
        block = num.split(".")[0] if "." in num else None
        add(sid, top, {"id": f"{sid}{num}", "label": f"{num}) {label} (°C)", "type": "number",
                       "box": nbox(xl, top, x1, bot), "read": True, "required": block is None,
                       "value": {"format": "number"}, "rule_ref": refs(sid, "number", block)})

    # ---------------- sign / date rows ---------------------------------- #
    for t, b in words:
        if t != CFG["labels"]["sign"]:
            continue
        cy = cy_of(b)
        top, bot = h_above(H, b[0], cy), h_below(H, b[0], cy)
        split = v_right(V, b[2], cy)
        right = v_right(V, split + 2, cy)
        dw = [b2 for t2, b2 in words if t2 == CFG["labels"]["date"] and top < cy_of(b2) < bot]
        sid = section_at(cy)
        sb = nbox(b[2] + 12, top, split, bot)
        db = nbox((dw[0][2] + 12) if dw else split, top, right, bot)
        add(sid, top, {"id": f"{sid}.sign", "label": f"Section {sid} completed by - signature",
                       "type": "signature", "box": sb, "rule_ref": refs(sid, "signoff"),
                       "overrides": dict(SIGN_OVERRIDES)})
        add(sid, top + 0.5, {"id": f"{sid}.date", "label": f"Section {sid} completed by - date",
                             "type": "date", "box": db, "read": True, "rule_ref": refs(sid, "signoff"),
                             "value": {"format": "date", "presence_only": True},
                             "overrides": dict(SIGN_OVERRIDES)})

    # ---------------- header fields: printed label -> value cell to its right -- #
    hs = CFG["header_section"]
    a_items = []
    if any(sid == hs for _, sid in heads):
        for f in CFG["header_fields"]:
            hit = [b for t, b in words if t == str(f["find"])]
            if not hit:
                print(f"  warning: header label '{f['find']}' ({f['id']}) not found on page {page_no}")
                continue
            b = hit[0]
            cy = cy_of(b)
            top, bot = h_above(H, b[0], cy), h_below(H, b[0], cy)
            x0 = v_right(V, b[2], cy)
            x1 = v_right(V, x0 + 2, cy)
            fmt = str(f.get("format", "text"))
            a_items.append({"id": str(f["id"]), "label": str(f.get("label", f["id"])),
                            "type": "date" if fmt == "date" else "text",
                            "box": nbox(x0, top, x1, bot), "read": True,
                            "value": {"format": fmt, "allow_na": bool(f.get("allow_na", False))},
                            "rule_ref": [str(r) for r in f.get("rule_ref", [])]})

    # ---------------- assemble ------------------------------------------ #
    dd = CFG["choice_defaults"]
    sections = []
    if a_items:
        sections.append({"id": hs, "title": titles.get(hs, hs), "items": a_items})
    for _, sid in heads:
        if sid == hs:
            continue
        its = [it for _, it in sorted(items_by_sec.get(sid, []), key=lambda p: p[0])]
        sections.append({"id": sid, "title": titles.get(sid, sid), "choice_defaults": dd, "items": its})

    rules = []
    for (sid, num), info in blocks.items():
        members = [it["id"] for _, it in sorted(items_by_sec.get(sid, []), key=lambda p: p[0])
                   if re.fullmatch(rf"{sid}{num}\.\d+(\.(start|end|t\d+))?", it["id"])]
        rules.append({"id": f"{sid}{num}_block", "type": "block", "item": f"{sid}{num}",
                      "description": f"Item {num}) {info['label']}: {num}.1 onwards",
                      "items": members, "na_checkbox": info["checkbox"],
                      "rule_ref": [f"{sid}-{num}"]})

    img_name = f"{stem}_p{page_no}.png"
    cv2.imwrite(str(out_dir / img_name), bgr)
    page = {"page_id": f"p{page_no}", "template_image": img_name, "sections": sections}
    if rules:
        page["rules"] = rules
    n_choice = sum(1 for s in sections for i in s["items"] if i["type"] == "choice")
    kinds = {}
    for s in sections:
        for i in s["items"]:
            kinds[i["type"]] = kinds.get(i["type"], 0) + 1
    print(f"  page {page_no}: sections {[s['id'] for s in sections]}, {kinds}, {len(rules)} block(s)")
    return page


def self_check(out: Path, pdf: Path, pages: list[int]) -> bool:
    """Audit the blank PDF against the template just built: every page must be
    recognised, and no box may read as filled (a box over printed text would)."""
    from form_auditor.io_utils import load_pages
    from form_auditor.pipeline import Auditor
    from form_auditor.template import load_template
    tpl = load_template(out)
    aud = Auditor([tpl], verbose=False)
    ok, n_rec, problems = True, 0, []
    for pg in load_pages(pdf, dpi=DPI):
        if pg.page_index + 1 not in pages:
            continue
        res = aud.analyse_page(pg)
        if res.ctx is None:
            ok = False
            problems.append(f"page {pg.page_index + 1} is NOT recognised against its own template "
                            f"({res.alignment.message if res.alignment else ''})")
            continue
        n_rec += 1
        aud.audit_prepared(res)
        for f in res.findings:
            filled = (f.check == "response" and f.observed not in ("none", "")) or \
                     (f.check == "na_checkbox" and f.observed != "blank") or \
                     (f.check not in ("response", "na_checkbox") and f.observed in ("filled", "uncertain"))
            if filled and not f.item_id.startswith("rule:") and f.check not in ("item_in_use", "cross_check"):
                problems.append(f"page {pg.page_index + 1}: {f.item_id} ({f.check}) reads as "
                                f"'{f.observed}' on the BLANK form - its box may cover printed text")
    print(f"\nSelf-check on the blank PDF: {n_rec}/{len(pages)} page(s) recognised, "
          f"{len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    used = tpl.referenced_rules()
    missing = sorted(r for r in used if r not in tpl.rules)
    unused = sorted(r for r in tpl.rules if r not in used)
    if not tpl.rules:
        print(f"  - note: rules CSV {tpl.path.parent / CFG['rules_csv']} not found - copy it in "
              f"(the reports then quote each rule's text)")
    else:
        if missing:
            print(f"  - rule ids cited by the template but not in the rules CSV: {', '.join(missing)}")
        if unused:
            print(f"  - rules in the CSV that nothing on this form checks: {', '.join(unused)}")
    return ok and not problems


def compare(out: Path) -> None:
    """Items added / removed compared with the other templates in the same folder."""
    def items(path):
        raw = read_yaml(path) or {}
        return {str(p.get("page_id")): {str(i["id"]) for sct in p.get("sections", []) for i in sct.get("items", [])}
                for p in raw.get("pages", [])}, raw
    new, _ = items(out)
    for other in sorted(out.parent.glob("*.yaml")):
        if other.resolve() == out.resolve():
            continue
        old, raw = items(other)
        if not raw.get("pages"):
            continue                                   # not a template (e.g. the config)
        ver = str(raw.get("version") or "")
        print(f"\nCompared with {other.name} ({raw.get('form_id', '')}{' v' + ver if ver else ''}):")
        same = True
        for pid in sorted(set(new) | set(old)):
            a, r = sorted(new.get(pid, set()) - old.get(pid, set())), sorted(old.get(pid, set()) - new.get(pid, set()))
            if a or r:
                same = False
                print(f"  {pid}: added {', '.join(a) or '-'}; removed {', '.join(r) or '-'}")
        if same:
            print("  same items on every page")


def main():
    global CFG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdf", help="the BLANK form PDF")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="form config (default templates/form_config.yaml)")
    ap.add_argument("--out", default=None, help="template YAML to write (default "
                    "templates/<output_stem>.yaml from the config)")
    ap.add_argument("--pages", type=int, nargs="+", default=None, help="default: all pages")
    ap.add_argument("--version", default=None, help="version label (default: from the config)")
    ap.add_argument("--no-check", action="store_true", help="skip the self-check")
    a = ap.parse_args()
    CFG = load_config(Path(a.config))

    out = Path(a.out) if a.out else ROOT / "templates" / f"{CFG['output_stem']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(a.pdf)
    n = len(pdf)
    pdf.close()
    page_nos = a.pages or list(range(1, n + 1))
    pages = [page_template(Path(a.pdf), p, out.parent, out.stem) for p in page_nos]

    version = a.version if a.version is not None else str(CFG.get("version", ""))
    tpl = {"form_id": str(CFG["form_id"]), "version": version, "name": str(CFG["name"]),
           "rules_csv": CFG["rules_csv"], "settings": {"render_dpi": DPI, **(CFG.get("settings") or {})},
           "pages": pages}
    # cross-checks between items, added only when the form has both items
    ids = {it["id"] for p in pages for sec in p["sections"] for it in sec["items"]}
    rules = []
    for c in CFG["cross_checks"]:
        if str(c["checkbox"]) in ids and str(c["item"]) in ids:
            rules.append({"id": str(c["id"]), "type": "checkbox_conflict", "checkbox": str(c["checkbox"]),
                          "item": str(c["item"]), "value": str(c.get("value", "Yes")),
                          "requires_setting": "na_checkbox_rules",
                          "description": str(c.get("description", c["id"])),
                          "rule_ref": [str(r) for r in c.get("rule_ref", [])]})
        else:
            print(f"  cross-check {c['id']} not added: this form has no "
                  f"{' / '.join(x for x in (str(c['checkbox']), str(c['item'])) if x not in ids)}")
    if rules:
        tpl["document_rules"] = rules
        tpl["settings"]["na_checkbox_rules"] = True   # the form has what these rules need
    title = f"{CFG['form_id']}{' v' + version if version else ''}"
    header = f"""# {title} ({len(pages)} page(s)) - KEEP OUT OF GIT (built from your form)
# Generated by tools/build_template_from_pdf.py from {Path(a.pdf).name} with {Path(a.config).name}.
# Boxes are [x0, y0, x1, y1] as fractions of the page. `printed` = exact box of each
# printed option word (from the PDF text layer). Options without a printed word (the
# empty NA cell on most rows) may not be used: fail_when_selected: [NA].
# rule_ref = rule ids in {CFG['rules_csv']}.
"""
    dump_yaml(tpl, out, header)
    print(f"wrote {out}")
    compare(out)
    if not a.no_check:
        good = self_check(out, Path(a.pdf), page_nos)
        print("\nTemplate looks good." if good else
              "\nCheck the problems above (and the preview) before using this template.")
        print(f"Look at the boxes: python -m form_auditor preview {out} -o preview.png")


if __name__ == "__main__":
    main()
