"""Turns page readings into audit findings.

Statuses:
  PASS   - check satisfied
  FAIL   - check clearly violated
  REVIEW - the machine is not confident (faint marks, two options marked, field hidden
           under an overlay, handwriting it could not read, ...). A human must look.
  SKIP   - check could not be applied (e.g. details rule when no response is readable)

The guiding principle: never silently PASS something the auditor is unsure about.
"""
from __future__ import annotations

from typing import Optional

from .context import PageContext
from .details import DetailsReading, field_extent, read_section_details
from .layout import FieldText, PageText, assign_words, checkbox_mark, fuse_presence, \
    nearby_pen_blobs, pen_words
from .marks import ChoiceReading, read_section_choices
from .ocr import OcrEngine, OcrRequest, OcrResult
from .rules_common import Finding, classify_presence
from .settings import Settings
from .template import FormTemplate, Item, Section, TemplatePage
from .values import normalize

__all__ = ["Finding", "audit_page", "apply_document_rules", "classify_presence"]


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _norm(v: str) -> str:
    return v.strip().lower()


_PRESENCE_WORD = {"signature": "Signature", "date": "Date", "time": "Time", "number": "Value",
                  "text": "Field", "field": "Field"}
_CHECK_NAME = {"signature": "signature_present", "date": "date_present", "time": "time_present",
               "number": "value_present", "text": "field_filled", "field": "field_filled"}


# --------------------------------------------------------------------------- #
def _faint_presence(item_box, ctx: PageContext, s: Settings, vis: float) -> Optional[str]:
    fctx = ctx.faint_ctx()
    if fctx is None:
        return None
    ink, _ = fctx.measure(item_box, s)
    st, _ = classify_presence(ink, vis, s)
    if st == "blank" and fctx.largest_text_blob(item_box, s) >= s.min_text_blob_px:
        st = "filled"
    return st


def audit_presence(sec: Section, item: Item, ctx: PageContext, s: Settings,
                   ft: Optional[FieldText] = None,
                   beside: Optional[list] = None) -> Finding:
    """Is the box filled in?  Ink check, plus (when available) the faint ink pass and the
    words the handwriting reader found in / beside the box. `beside` = pen writing (px
    boxes) right next to the box but in no box."""
    ink, vis = ctx.measure(item.box, s)
    state, conf = classify_presence(ink, vis, s)
    if state in ("uncertain", "blank") and vis >= s.min_visible_frac:
        # short entries ("NA", a small initial) in a big box have a low ink ratio, but a
        # solid word-sized blob of writing is unmistakable
        if ctx.largest_text_blob(item.box, s) >= s.min_text_blob_px:
            state, conf = "filled", 0.8
    faint = _faint_presence(item.box, ctx, s, vis) if state == "blank" else None
    state, c2, fuse_notes = fuse_presence(state, faint, ft, s)
    conf = c2 if c2 is not None else conf
    word = _PRESENCE_WORD[item.type]
    f = Finding(sec.id, sec.title, item.id, item.label, _CHECK_NAME[item.type], "PASS",
                state, "", "", conf, [item.box],
                {"ink_ratio": round(ink, 4), "visible_ratio": round(vis, 3),
                 "has_content": state == "filled"}, rule_refs=list(item.rule_refs),
                notes=list(fuse_notes))
    beside = list(beside or [])
    if ft is not None and ft.outside:
        beside.append(ft.outside_extent)
    beside_txt = f" ('{ft.outside_text}')" if ft is not None and ft.outside else ""
    if state == "occluded":
        f.status, f.reason = "REVIEW", f"{word} is obscured ({vis:.0%} visible) - check manually"
    elif state == "uncertain":
        f.status = "REVIEW"
        f.reason = (fuse_notes[-1][0].upper() + fuse_notes[-1][1:] + " - confirm it is properly filled"
                    if fuse_notes else f"{word} has faint/partial marks - confirm it is properly filled")
    elif item.must_be_blank:
        f.expected = "blank"
        if state == "filled":
            f.status, f.reason = "FAIL", f"{word} must be left blank but contains writing"
    elif item.required:
        f.expected = "filled"
        if state == "blank":
            if beside:
                f.status = "REVIEW"
                f.reason = (f"{word} box is empty but there is writing right next to it"
                            f"{beside_txt} - check whether it was meant for this box")
                f.boxes = [item.box] + [ctx.norm(*b) for b in beside]
            else:
                f.status = "FAIL"
                f.reason = {"signature": "Signature missing", "date": "Date missing"}.get(
                    item.type, "Required field is blank")
    else:
        f.expected = "optional"
    if state == "filled" and beside:
        f.notes.append(f"more writing right next to the box{beside_txt}")
    return f


def read_checkbox(item: Item, ctx: PageContext, s: Settings) -> tuple[str, int, float]:
    """-> (state, ink px, visible ratio); state ticked | blank | uncertain | occluded.
    Pen ink anywhere in the box or on its NA label counts (a tick, a cross, a filled box,
    or NA circled instead of ticking)."""
    x0, y0, x1, y1 = ctx.px(item.hit_box or item.box)
    vis = ctx.visible(x0, y0, x1, y1)
    if vis < s.min_visible_frac:
        return "occluded", 0, vis
    sub = ctx.maps.added[y0:y1, x0:x1]
    ink = 0
    if sub.any():
        import cv2
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype("uint8"), connectivity=8)
        for j in range(1, n):
            if min(st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]) > s.sliver_px:
                ink += int(st[j, cv2.CC_STAT_AREA])
    if ink >= s.checkbox_ink_hi:
        return "ticked", ink, vis
    if ink <= s.checkbox_ink_lo:
        return "blank", ink, vis
    return "uncertain", ink, vis


def audit_checkbox(sec: Section, item: Item, ctx: PageContext, s: Settings,
                   page_text: Optional[PageText] = None) -> Finding:
    state, ink, vis = read_checkbox(item, ctx, s)
    notes = []
    if state == "blank" and ctx.faint_ctx() is not None:
        fstate, _, _ = read_checkbox(item, ctx.faint_ctx(), s)
        if fstate != "blank":
            state = "uncertain"
            notes.append("faint mark in the NA box (light pen or pencil?)")
    m = checkbox_mark(page_text, ctx.px(item.box))
    if m is not None:
        notes.append(f"handwriting reader: box {m.state} ({m.confidence:.2f})")
        if state == "blank" and m.state == "selected" and m.confidence >= s.ocr_min_conf:
            state = "uncertain"
    f = Finding(sec.id, sec.title, item.id, item.label, "na_checkbox", "PASS", state,
                "ticked or not", "", 1.0, [item.hit_box or item.box],
                {"ink_ratio": ink, "visible_ratio": round(vis, 3), "has_content": state == "ticked"},
                rule_refs=list(item.rule_refs), notes=notes)
    if not s.na_checkbox_rules:
        f.status, f.expected = "SKIP", "recorded only"
        f.reason = "Recorded only - the NA-checkbox rules are switched off (na_checkbox_rules)"
    elif state in ("uncertain", "occluded"):
        f.status = "REVIEW"
        f.reason = ("NA box is obscured" if state == "occluded" else
                    "NA box has a faint / small mark, or the ink check and the handwriting "
                    "reader disagree - confirm whether it is ticked")
    return f


def audit_choice(sec: Section, item: Item, rd: ChoiceReading, dr: Optional[DetailsReading],
                 s: Settings, ctx: PageContext) -> list[Finding]:
    opts = "/".join(o for o in item.options if o in item.printed) or "/".join(item.options)
    resp = Finding(sec.id, sec.title, item.id, item.label, "response", "PASS",
                   rd.selected if rd.state == "selected" else rd.state,
                   f"exactly one of {opts}", "", rd.confidence,
                   [_union(list(item.options.values()))],
                   {"has_content": rd.has_content, "visible_ratio": round(rd.visible, 3),
                    "marks": "; ".join(f"{m.describe()}@{m.bbox}" for m in rd.marks if m.kind != "speck")},
                   rule_refs=list(item.rule_refs), mark_type=rd.mark_type)
    if rd.note:
        resp.notes.append(rd.note)
    if rd.state == "none":
        if item.required:
            resp.status, resp.reason = "FAIL", "No response marked"
    elif rd.state == "ambiguous":
        resp.status, resp.reason = "REVIEW", f"Unclear response - {rd.note}"
    elif rd.state == "cancelled":
        resp.status, resp.reason = "REVIEW", rd.note[0].upper() + rd.note[1:]
    elif rd.state in ("uncertain", "occluded"):
        resp.status, resp.reason = "REVIEW", rd.note[0].upper() + rd.note[1:]
        if rd.selected:
            resp.observed = f"{rd.state} ({rd.selected}?)"
    if rd.state == "selected":
        sel = rd.selected
        if _norm(sel) in {_norm(x) for x in item.fail_when_selected}:
            how = "written in" if any(m.kind == "written" for m in rd.marks if m.option == sel) else "marked"
            resp.status = "FAIL"
            resp.reason = (f"'{sel}' {how} on a row where it is not a printed option - only the "
                           f"options printed on the sheet may be used")
        elif _norm(sel) in {_norm(x) for x in item.review_when_selected}:
            resp.status = "REVIEW"
            resp.reason = f"'{sel}' marked on a row where it is not a printed option - confirm it is allowed"
        elif _norm(sel) in {_norm(x) for x in item.flag_when_selected}:
            resp.status = "FAIL"
            resp.reason = f"Response is '{sel}' - every '{sel}' answer is flagged"
        if rd.mark_type in ("tick", "cross", "cancel", "written") and "was cancelled" not in rd.note:
            resp.notes.append({"tick": "answered with a tick instead of a circle",
                               "cross": "answered with an X mark instead of a circle",
                               "cancel": "answer was crossed out but counted (cancel_mark_means=selected)",
                               "written": "answer written in instead of circled"}[rd.mark_type])
    m = item.rule_ref_map
    if m:
        # cite the rule the outcome is about
        if resp.status == "PASS":
            pass
        elif rd.state == "none" and "answer" in m:
            resp.rule_refs = [m["answer"]]
        elif rd.state == "selected" and _norm(rd.selected) in {_norm(x) for x in item.flag_when_selected} \
                and "flag" in m:
            resp.rule_refs = [m["flag"]]
        elif rd.state == "selected" and "options" in m:
            resp.rule_refs = [m["options"]]
        elif "reading" in m:
            resp.rule_refs = [m["reading"]]
    findings = [resp]

    d = item.details
    if d is None or dr is None:
        return findings
    boxes = [d.box]
    if dr.extent:
        boxes = [_union([d.box, ctx.norm(*dr.extent)])]
    det = Finding(sec.id, sec.title, item.id, item.label, "details", "PASS",
                  dr.state, "", "", dr.confidence, boxes,
                  {"ink_ratio": dr.ink_ratio, "visible_ratio": dr.visible,
                   "has_content": dr.state == "filled"},
                  rule_refs=list(d.rule_refs), notes=list(dr.notes))

    if rd.state != "selected":
        det.status, det.expected = "SKIP", "depends on response"
        det.reason = f"Details rule not applied because response is '{rd.state}'"
        if dr.state == "filled":
            det.reason += " (details column contains writing)"
        findings.append(det)
        return findings

    sel = _norm(rd.selected)
    if sel in {_norm(x) for x in item.flag_when_selected}:
        resp.notes.append({"filled": "details written", "blank": "no details written"}.get(
            dr.state, f"details column {dr.state}"))
    req = {_norm(x) for x in d.required_when}
    blank = {_norm(x) for x in d.must_be_blank_when}
    if sel in req:
        det.expected = "filled"
        if dr.state == "blank":
            det.status = "FAIL"
            det.reason = f"Response is '{rd.selected}' but details / follow-up actions are blank"
    elif sel in blank:
        det.expected = "blank"
        if dr.state == "filled":
            if dr.continuation_of:
                det.status = "REVIEW"
                det.reason = (f"Response is '{rd.selected}' and the details cell has writing - it may be "
                              f"the continuation of row {dr.continuation_of}'s details; check")
            else:
                det.status = "FAIL"
                det.reason = (f"Response is '{rd.selected}' - details column must be left blank "
                              f"but contains writing")
    else:
        det.expected = "any"
    if det.status == "PASS" and dr.state in ("uncertain", "occluded") and det.expected in ("filled", "blank"):
        det.status = "REVIEW"
        det.reason = ("Details column is obscured" if dr.state == "occluded"
                      else "Details column has faint/partial marks") + f" - expected {det.expected}"
    findings.append(det)
    return findings


# --------------------------------------------------------------------------- #
# Values (transcription)
# --------------------------------------------------------------------------- #
def _apply_value(f: Finding, item: Item, fmt: str, res, s: Settings, is_details: bool) -> None:
    if res is None:
        f.notes.append("not transcribed")
        return
    f.ocr_engine = res.engine
    f.ocr_confidence = res.confidence
    if res.error:
        f.notes.append(f"transcription failed: {res.error}")
        if not is_details or f.expected == "filled":
            f.escalate("REVIEW", "Handwriting present but could not be transcribed (reader error)")
        return
    f.value_read = res.text
    if res.illegible or not res.text.strip():
        if item.presence_only and not is_details:
            f.notes.append("written, but the reader could not read it (only presence is checked)")
        elif not is_details or f.expected == "filled":
            f.escalate("REVIEW", "Handwriting present but not legible to the reader - check")
        return
    if is_details:
        f.value_normalized = " ".join(res.text.split())
        if f.expected == "filled" and res.confidence < s.ocr_min_conf:
            f.escalate("REVIEW", f"Details read with low confidence: '{f.value_normalized}' - verify")
        return
    v = normalize(fmt, res.text, item.allow_na, s.date_order, s.time_strict)
    f.value_normalized = v.normalized
    if v.problem and v.valid:
        f.notes.append(v.problem)
    confident = res.confidence >= s.ocr_fail_conf
    if v.valid is False:
        if v.is_na:
            f.escalate("FAIL" if confident else "REVIEW",
                       f"{v.problem} (read: '{v.raw}')")
        elif fmt == "time":
            f.escalate("FAIL" if confident else "REVIEW",
                       f"Time must be written as HH:MM - read '{v.raw}'")
        elif fmt == "date":
            if s.check_date_format:
                f.escalate("REVIEW", f"Could not interpret '{v.raw}' as a date - check")
            else:
                f.notes.append("not a recognisable date (date format is not checked)")
        else:
            f.escalate("REVIEW", f"Value '{v.raw}' is {v.problem} - check")
    elif res.confidence < s.ocr_min_conf:
        if item.presence_only:
            f.notes.append(f"read with low confidence as '{v.raw}' (only presence is checked)")
        else:
            f.escalate("REVIEW", f"Read with low confidence as '{v.raw}' - verify")


# --------------------------------------------------------------------------- #
# Page-level rules
# --------------------------------------------------------------------------- #
def _rule_complete_before(rule: dict, findings: list[Finding]) -> Finding:
    pre = [str(x) for x in rule["prerequisite_sections"]]
    dep = str(rule["dependent_section"])
    desc = rule.get("description", f"Sections {', '.join(pre)} must be complete before section {dep}")
    dep_active = any(f.metrics.get("has_content") for f in findings if f.section_id == dep)
    pre_fail = [f for f in findings if f.section_id in pre and f.status == "FAIL"]
    pre_review = [f for f in findings if f.section_id in pre and f.status == "REVIEW"]
    f = Finding("PAGE", "Page rules", f"rule:{rule.get('id', 'complete_before')}", desc,
                "complete_before", "PASS", "", "", "", 1.0, rule_refs=list(rule.get("rule_ref", [])))
    f.observed = f"section {dep} {'started' if dep_active else 'not started'}; " \
                 f"{len(pre_fail)} failure(s) in {'/'.join(pre)}"
    f.expected = f"{'/'.join(pre)} complete before {dep}"
    if dep_active and pre_fail:
        f.status = "FAIL"
        f.reason = f"Section {dep} was filled in while section(s) {'/'.join(pre)} still have " \
                   f"{len(pre_fail)} failure(s): " + ", ".join(sorted({x.item_id for x in pre_fail}))
    elif dep_active and pre_review:
        f.status, f.confidence = "REVIEW", 0.5
        f.reason = f"Section {dep} started; prerequisites have items needing review"
    return f


def _field_value(findings, item_id) -> tuple[Optional[str], Optional[Finding]]:
    for f in findings:
        if f.item_id == item_id and f.check not in ("details",):
            if f.check == "response":
                return (f.observed if f.status != "SKIP" else None), f
            return (f.value_normalized or None), f
    return None, None


def _rule_option_condition(rule: dict, findings: list[Finding]) -> Finding:
    """option_allowed_only_if: `option` of `item` may only be chosen when `field` is in `in`
       option_required_if:     `option` of `item` must be chosen when `field` is in `in`
    Matching is case-insensitive; `contains: true` matches substrings."""
    kind = rule["type"]
    item, option, fld = str(rule["item"]), str(rule["option"]), str(rule["field"])
    allowed = [str(v).strip().lower() for v in rule.get("in", [])]
    contains = bool(rule.get("contains", False))
    resp, rf = _field_value(findings, item)
    val, vf = _field_value(findings, fld)
    desc = rule.get("description", f"{kind}: {item}={option} / {fld} in {allowed}")
    f = Finding("PAGE", "Page rules", f"rule:{rule.get('id', kind)}", desc, kind, "PASS",
                f"{item}={resp}; {fld}={val}", "", "", 1.0,
                boxes=(rf.boxes if rf else []) + (vf.boxes if vf else []),
                rule_refs=list(rule.get("rule_ref", [])))

    def matches(v):
        v = (v or "").strip().lower()
        return any(a in v for a in allowed) if contains else v in allowed

    chosen = resp is not None and resp.lower() == option.lower()
    if kind == "option_allowed_only_if":
        f.expected = f"{option} only when {fld} is one of {rule.get('in')}"
        if chosen:
            if val is None:
                f.status, f.reason = "REVIEW", f"{item} is {option} but {fld} could not be read to confirm it is allowed"
            elif not matches(val):
                f.status, f.reason = "FAIL", f"{item} is {option}, which is only allowed for {fld} in {rule.get('in')} (read '{val}')"
    else:
        f.expected = f"{option} when {fld} is one of {rule.get('in')}"
        if val is None:
            if resp is not None and not chosen:
                f.status, f.reason = "REVIEW", f"{fld} could not be read - cannot confirm whether {item} must be {option}"
        elif matches(val) and not chosen:
            f.status, f.reason = "FAIL", f"{fld} is '{val}', so {item} must be {option} (found '{resp}')"
    return f


def _main_finding(findings, item_id) -> Optional[Finding]:
    """The response / presence / checkbox finding of an item (not its details)."""
    return next((f for f in findings if f.item_id == item_id and f.check != "details"), None)


def _has_content(f: Finding) -> bool:
    if f.check == "response":
        return f.observed not in ("none", "")
    return f.observed in ("filled", "uncertain")


def _rule_block(rule: dict, findings: list[Finding], s: Settings) -> Finding:
    """An item with sub-items (e.g. 5.1-5.5 under item 5) and an NA checkbox.

    na_checkbox_rules off: the block is required as soon as any part of it is filled in.
    na_checkbox_rules on:  the block is required unless its NA box is ticked.
    When required, every member must be answered / filled; blanks are FAIL."""
    head = str(rule.get("item", rule.get("id")))
    members = [m for m in (_main_finding(findings, str(i)) for i in rule.get("items", [])) if m]
    cb = _main_finding(findings, str(rule["na_checkbox"])) if rule.get("na_checkbox") else None
    cb_state = cb.observed if cb else "missing"
    filled = [m for m in members if _has_content(m)]
    f = Finding(members[0].section_id if members else "PAGE",
                members[0].section_title if members else "Page rules",
                head, rule.get("description", head), "item_in_use", "PASS", "", "", "", 1.0,
                rule_refs=list(rule.get("rule_ref", [])))
    f.metrics["has_content"] = bool(filled)
    span = f"{len(filled)} of {len(members)} part(s) filled in; NA box {cb_state}"
    if s.na_checkbox_rules:
        if cb_state in ("uncertain", "occluded", "missing"):
            f.status, f.observed = "REVIEW", span
            f.expected = "complete unless NA is ticked"
            f.reason = (f"Cannot tell whether item {head} applies: its NA box is {cb_state} "
                        f"({len(filled)} part(s) filled in)")
            f.boxes = cb.boxes if cb else []
            return f
        required = cb_state != "ticked"
        why = "its NA box is not ticked"
        f.expected = "complete (NA not ticked)" if required else "not required (NA ticked)"
        if not required and filled:
            f.status, f.observed = "REVIEW", span
            f.reason = (f"Item {head} NA box is ticked but {len(filled)} of its part(s) are "
                        f"filled in: " + ", ".join(m.item_id for m in filled))
            f.boxes = (cb.boxes if cb else []) + [b for m in filled for b in m.boxes]
            return f
    else:
        required = bool(filled)
        why = "part of it is filled in"
        f.expected = "all parts filled in, or none"
    f.observed = ("in use: " if required else "not used: ") + span
    blanks = [m for m in members if not _has_content(m)]
    for m in members:
        if not required:
            m.expected = f"not required (item {head} not used)" if not s.na_checkbox_rules \
                else f"not required (item {head} NA ticked)"
            continue
        m.expected = "Yes or No" if m.check == "response" else "filled"
        if m in blanks and m.status in ("PASS", "SKIP"):
            what = "No response marked" if m.check == "response" else "Value missing"
            m.status, m.reason = "FAIL", f"{what} - item {head} must be complete because {why}"
            m.rule_refs = list(rule.get("rule_ref", []))
    if required and blanks:
        f.reason = f"Item {head} is incomplete ({len(blanks)} blank part(s)) - see " + \
                   ", ".join(m.item_id for m in blanks)
    return f


def _page_rule(rule: dict, findings: list[Finding], s: Settings) -> Optional[Finding]:
    t = rule.get("type")
    req = rule.get("requires_setting")
    if req and not getattr(s, req):
        return None
    if t == "block":
        return _rule_block(rule, findings, s)
    if t == "complete_before":
        return _rule_complete_before(rule, findings)
    if t in ("option_allowed_only_if", "option_required_if"):
        return _rule_option_condition(rule, findings)
    raise ValueError(f"Unknown page rule type: {t}")


# --------------------------------------------------------------------------- #
def _faint_choices(ctx: PageContext, sec: Section, choices: dict[str, ChoiceReading],
                   base: Settings) -> None:
    """Rows with no mark: look again with the sensitive ink pass. A faint mark turns
    "no response" (FAIL) into "unclear" (REVIEW) with the guess."""
    empty = [i for i in sec.items if i.type == "choice" and choices[i.id].state == "none"]
    fctx = ctx.faint_ctx()
    if not empty or fctx is None:
        return
    faint = read_section_choices(fctx, sec.items, base)
    for it in empty:
        fr = faint[it.id]
        if fr.state == "none":
            continue
        what = fr.mark_type or "mark"
        on = f" on {fr.selected}" if fr.selected else ""
        choices[it.id] = ChoiceReading(
            "uncertain", fr.selected, 0.4,
            f"no clear mark, only a faint {what}{on} (light pen or pencil?) - confirm the answer",
            fr.marks, fr.visible)


def _page_fields(page: TemplatePage, ctx: PageContext):
    """(fields the reader's words are given to, Yes/No/NA cells + checkboxes, all boxes)"""
    fields, answers = {}, []
    for sec in page.sections:
        for it in sec.items:
            if it.type == "choice":
                answers += [ctx.px(b) for b in it.options.values() if b]
                if it.details is not None and it.details.box:
                    fields[f"{it.id}.details"] = ctx.px(it.details.box)
            elif it.type == "checkbox":
                answers.append(ctx.px(it.hit_box or it.box))
            elif it.box:
                fields[it.id] = ctx.px(it.box)
    return fields, answers, list(fields.values()) + answers


def audit_page(page: TemplatePage, ctx: PageContext, base: Settings,
               ocr: Optional[OcrEngine] = None, where: Optional[dict] = None,
               page_text: Optional[PageText] = None) -> list[Finding]:
    """`page_text`: words found on the whole page by a page-mode reader (--ocr azure), in
    template px. Without it, filled boxes are cropped and sent to a crop reader."""
    findings: list[Finding] = []
    reads = []          # crop readers: (finding, item, fmt, extent px, is_details)
    values = []         # page reader:  (finding, item, fmt, FieldText, is_details)
    fields, answers, all_boxes = _page_fields(page, ctx)
    texts: dict[str, FieldText] = {}
    if page_text is not None and not page_text.error:
        texts = assign_words(pen_words(page_text, ctx, base), fields, answers, base)
    reader_error = page_text.error if page_text is not None else ""

    for sec in page.sections:
        choices = read_section_choices(ctx, sec.items, base)
        if base.faint_check:
            _faint_choices(ctx, sec, choices, base)
        details = read_section_details(ctx, sec.items, base)
        fdetails = None
        for item in sec.items:
            s = base.merged(item.overrides)
            if item.type == "choice":
                dr = details.get(item.id)
                ft = texts.get(f"{item.id}.details")
                if dr is not None and item.details is not None:
                    faint = None
                    if dr.state == "blank" and ctx.faint_ctx() is not None:
                        if fdetails is None:
                            fdetails = read_section_details(ctx.faint_ctx(), sec.items, base)
                        fd = fdetails.get(item.id)
                        faint = fd.state if fd else None
                    st, _, notes = fuse_presence(dr.state, faint, ft, s, multiline=True)
                    if st != dr.state:
                        dr.state = st
                        dr.notes = list(dr.notes) + notes
                    if ft is not None and ft.extent:
                        dr.blobs = list(dr.blobs) + [ft.extent]
                    if dr.state == "blank" and ft is not None and ft.outside:
                        dr.state = "uncertain"
                        dr.notes = list(dr.notes) + [f"writing right next to the details cell "
                                                     f"('{ft.outside_text}')"]
                        dr.blobs = list(dr.blobs) + [ft.outside_extent]
                fs = audit_choice(sec, item, choices[item.id], dr, s, ctx)
                findings += fs
                det = fs[-1] if fs[-1].check == "details" else None
                if item.details and item.details.read and det and dr \
                        and dr.state in ("filled", "uncertain"):
                    if page_text is not None:
                        values.append((det, item, "text", ft, True))
                    elif dr.extent:
                        reads.append((det, item, "text", dr.extent, True))
            elif item.type == "checkbox":
                findings.append(audit_checkbox(sec, item, ctx, s, page_text))
            else:
                ft = texts.get(item.id)
                beside = []
                if item.required and not item.must_be_blank:
                    beside = nearby_pen_blobs(ctx, ctx.px(item.box),
                                              [b for b in all_boxes if b != ctx.px(item.box)], s)
                f = audit_presence(sec, item, ctx, s, ft, beside)
                findings.append(f)
                if item.type != "signature":
                    extent, notes = field_extent(ctx, item.box, s)
                    if f.observed != "filled":
                        notes = []
                    if ft is not None and ft.overflow and f.observed == "filled":
                        notes = list(notes) + ["writing runs past the edge of the box"]
                    f.notes += list(dict.fromkeys(notes))
                    if ft is not None and ft.extent and f.observed == "filled":
                        extent = ft.extent if extent is None else (
                            min(extent[0], ft.extent[0]), min(extent[1], ft.extent[1]),
                            max(extent[2], ft.extent[2]), max(extent[3], ft.extent[3]))
                    if extent:
                        bx = ctx.px(item.box)
                        if notes:
                            f.boxes = [_union([item.box, ctx.norm(*extent)])] + f.boxes[1:]
                        if item.read and f.observed in ("filled", "uncertain"):
                            if page_text is None:
                                ext = (min(extent[0], bx[0]), min(extent[1], bx[1]),
                                       max(extent[2], bx[2]), max(extent[3], bx[3]))
                                reads.append((f, item, item.value_format, ext, False))
                    if page_text is not None and item.read and f.observed in ("filled", "uncertain"):
                        values.append((f, item, item.value_format, ft, False))

    # values read by the page reader
    for f, item, fmt, ft, is_det in values:
        if reader_error:
            res = OcrResult("", 0.0, page_text.engine, error=reader_error)
        elif ft is None or not ft.words:
            res = OcrResult("", 0.0, page_text.engine, illegible=True)
        else:
            res = OcrResult(ft.text(multiline=is_det), ft.confidence, page_text.engine)
        _apply_value(f, item, fmt, res, base.merged(item.overrides), is_det)

    # values read by a crop reader
    if reads:
        engine = ocr or OcrEngine()
        reqs = []
        for f, item, fmt, ext, is_det in reads:
            key = f"{item.id}.details" if is_det else item.id
            hint = f"{item.label} - details / follow-up actions" if is_det else item.label
            reqs.append(OcrRequest(key, ctx.crop(*ext, scale=base.ocr_scale), fmt, hint))
        page_level = getattr(engine, "page_level", False)
        results = engine.read(reqs, where or {}) if engine.name != "none" and not page_level else {}
        for (f, item, fmt, ext, is_det), r in zip(reads, reqs):
            if engine.name == "none" or page_level:
                continue
            _apply_value(f, item, fmt, results.get(r.key), base.merged(item.overrides), is_det)

    for rule in page.rules:
        f = _page_rule(rule, findings, base)
        if f is not None:
            findings.append(f)
    return findings


# --------------------------------------------------------------------------- #
# Document rules: conditions between pages of one form (e.g. page 2 vs page 1)
# --------------------------------------------------------------------------- #
def apply_document_rules(template: FormTemplate, pages: dict[str, list[Finding]],
                         s: Settings) -> list[tuple[str, Finding]]:
    """`pages` maps template page id -> findings of that page in this form.
    Returns (page id to report on, finding) pairs."""
    out = []
    for rule in template.document_rules:
        req = rule.get("requires_setting")
        if req and not getattr(s, req):
            continue
        if rule.get("type") != "checkbox_conflict":
            raise ValueError(f"Unknown document rule type: {rule.get('type')}")
        cb_id, other, value = str(rule["checkbox"]), str(rule["item"]), str(rule["value"])
        cb_page, other_page = template.page_of_item(cb_id), template.page_of_item(other)
        if cb_page not in pages:
            continue                         # that page is not in this form - nothing to check
        cb = _main_finding(pages[cb_page], cb_id)
        oth = _main_finding(pages.get(other_page, []), other)
        f = Finding(cb.section_id if cb else "PAGE", cb.section_title if cb else "Page rules",
                    f"rule:{rule.get('id')}", rule.get("description", rule.get("id")),
                    "cross_check", "PASS", "", f"not ({other} = {value} and {cb_id} ticked)", "",
                    1.0, rule_refs=list(rule.get("rule_ref", [])))
        cb_state = cb.observed if cb else "missing"
        if oth is None:
            ans = f"page {other_page} not in this file"
        elif oth.status == "SKIP" or oth.observed in ("none", "ambiguous", "cancelled") \
                or oth.observed.startswith(("uncertain", "occluded")):
            ans = None
        else:
            ans = oth.observed
        f.observed = f"{other} = {ans if ans is not None else (oth.observed if oth else '?')}; {cb_id} {cb_state}"
        f.boxes = list(cb.boxes) if cb else []
        if cb_state == "ticked" and ans == value:
            f.status = "FAIL"
            f.reason = f"{cb_id} is ticked but {other} is '{value}'"
        elif cb_state in ("uncertain", "occluded") and (ans == value or ans is None):
            f.status, f.reason = "REVIEW", f"{cb_id} is {cb_state} and {other} is {ans or 'unclear'} - check"
        elif cb_state == "ticked" and (ans is None or oth is None):
            f.status = "REVIEW"
            f.reason = f"{cb_id} is ticked; {other} could not be confirmed ({f.observed.split(';')[0]})"
        out.append((cb_page, f))
    return out
