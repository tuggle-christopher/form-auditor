"""Form template model + YAML loader.

A template describes one paper form: one blank reference image per page, and for
each page a list of sections whose items have bounding boxes (normalised 0..1
coordinates on the blank image) and audit rules.

Templates are generated from the blank form by tools/build_template_from_pdf.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import re

import numpy as np
import yaml

from .io_utils import load_image_file
from .settings import Settings

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 normalised to page size


class YamlLoader(yaml.SafeLoader):
    """SafeLoader where only true/false are booleans. Plain YAML 1.1 would turn the
    option names Yes / No / On / Off into True / False."""


YamlLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
YamlLoader.add_implicit_resolver("tag:yaml.org,2002:bool",
                                 re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def read_yaml(path: Path) -> dict:
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=YamlLoader)


PRESENCE_TYPES = {"text", "date", "time", "number", "signature", "field"}
VALUE_FORMATS = {"text", "id", "date", "time", "number"}
_DEFAULT_FORMAT = {"text": "text", "field": "text", "date": "date", "time": "time", "number": "number"}
ITEM_TYPES = PRESENCE_TYPES | {"choice", "checkbox"}


@dataclass
class DetailsSpec:
    box: Optional[Box]
    required_when: list[str] = field(default_factory=list)
    must_be_blank_when: list[str] = field(default_factory=list)
    read: bool = False                  # transcribe the handwriting
    rule_refs: list[str] = field(default_factory=list)


@dataclass
class RuleRef:
    """One region / instruction from the customer's rules CSV."""
    label: str
    category: str
    instruction: str
    region_id: str = ""


@dataclass
class Item:
    id: str
    label: str
    type: str
    required: bool = True
    must_be_blank: bool = False
    box: Optional[Box] = None
    options: dict[str, Optional[Box]] = field(default_factory=dict)
    details: Optional[DetailsSpec] = None
    review_when_selected: list[str] = field(default_factory=list)   # e.g. NA where not printed
    fail_when_selected: list[str] = field(default_factory=list)     # answers that are never valid
    flag_when_selected: list[str] = field(default_factory=list)     # valid answers that are flagged
                                                                    # (FAIL), e.g. every "No"
    hit_box: Optional[Box] = None       # checkbox: area searched for a tick (box + its label)
    printed: dict[str, Box] = field(default_factory=dict)            # printed option words
    rule_refs: list[str] = field(default_factory=list)
    rule_ref_map: dict[str, str] = field(default_factory=dict)   # outcome -> rule id (choices)
    read: bool = False                  # transcribe the handwriting (OCR)
    value_format: str = "text"          # text | id | date | time | number
    allow_na: bool = False              # a written "NA" is an acceptable value
    overrides: dict = field(default_factory=dict)

    def all_boxes(self) -> list[tuple[str, Optional[Box]]]:
        if self.type == "choice":
            out = [(f"option:{k}", v) for k, v in self.options.items()]
            if self.details:
                out.append(("details", self.details.box))
            return out
        return [("box", self.box)]


@dataclass
class Section:
    id: str
    title: str
    items: list[Item]


@dataclass
class TemplatePage:
    page_id: str
    image_path: Path
    sections: list[Section]
    rules: list[dict] = field(default_factory=list)
    _image: Optional[np.ndarray] = None

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = load_image_file(self.image_path)
        return self._image

    def section(self, sid: str) -> Optional[Section]:
        return next((s for s in self.sections if s.id == sid), None)


@dataclass
class FormTemplate:
    form_id: str
    name: str
    version: str
    path: Path
    settings: Settings
    pages: list[TemplatePage]
    rules: dict[str, RuleRef] = field(default_factory=dict)
    document_rules: list[dict] = field(default_factory=list)   # rules spanning pages

    def page_of_item(self, item_id: str) -> Optional[str]:
        for p in self.pages:
            for s in p.sections:
                if any(i.id == item_id for i in s.items):
                    return p.page_id
        return None

    def referenced_rules(self) -> set[str]:
        """Rule ids (rules CSV labels) that some item or rule of this template can cite."""
        out = set()
        for p in self.pages:
            for r in p.rules:
                out.update(_as_list(r.get("rule_ref")))
            for s in p.sections:
                for i in s.items:
                    out.update(i.rule_refs)
                    out.update(i.rule_ref_map.values())
                    if i.details is not None:
                        out.update(i.details.rule_refs)
        for r in self.document_rules:
            out.update(_as_list(r.get("rule_ref")))
        return out

    def rule_text(self, refs: list[str]) -> str:
        out = []
        for r in refs:
            rr = self.rules.get(r)
            out.append(f"{r}: {rr.instruction}" if rr else r)
        return " | ".join(out)

    @property
    def display_name(self) -> str:
        return f"{self.form_id} v{self.version}" if self.version else self.form_id


# --------------------------------------------------------------------------- #
def _parse_box(raw, where: str, allow_missing: bool) -> Optional[Box]:
    if raw is None:
        if allow_missing:
            return None
        raise ValueError(
            f"{where}: bounding box missing. Run `python -m form_auditor calibrate` "
            f"to draw it on the blank template.")
    if len(raw) != 4:
        raise ValueError(f"{where}: box must be [x0, y0, x1, y1], got {raw}")
    x0, y0, x1, y1 = (float(v) for v in raw)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"{where}: box {raw} must be normalised (0..1) with x0<x1, y0<y1")
    return (x0, y0, x1, y1)


def _as_list(v) -> list[str]:
    if v is None:
        return []
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def load_template(path: str | Path, base: Settings | None = None,
                  allow_missing_boxes: bool = False) -> FormTemplate:
    path = Path(path)
    raw = read_yaml(path)
    settings = (base or Settings()).merged(raw.get("settings"))

    pages = []
    for p_i, p in enumerate(raw.get("pages", [])):
        page_id = str(p.get("page_id", f"p{p_i + 1}"))
        img_path = (path.parent / p["template_image"]).resolve()
        if not img_path.exists() and not allow_missing_boxes:
            raise FileNotFoundError(f"Template image not found: {img_path}")

        sections = []
        for s in p.get("sections", []):
            sid = str(s["id"])
            cdef = s.get("choice_defaults", {}) or {}
            cdef_details = cdef.get("details", {}) or {}
            items = []
            for it in s.get("items", []):
                iid = str(it["id"])
                where = f"{path.name}:{page_id}:{iid}"
                itype = str(it.get("type", "text")).lower()
                if itype not in ITEM_TYPES:
                    raise ValueError(f"{where}: unknown type '{itype}' (use one of {sorted(ITEM_TYPES)})")
                item = Item(
                    id=iid,
                    label=str(it.get("label", iid)),
                    type=itype,
                    required=bool(it.get("required", True)),
                    must_be_blank=bool(it.get("must_be_blank", False)),
                    overrides=it.get("overrides", {}) or {},
                )
                settings.merged(item.overrides)  # validate names early
                if itype == "choice":
                    opts = it.get("options") or {}
                    if not opts:
                        raise ValueError(f"{where}: choice item needs `options`")
                    item.review_when_selected = _as_list(it.get("review_when_selected"))
                    item.fail_when_selected = _as_list(it.get("fail_when_selected"))
                    item.flag_when_selected = _as_list(it.get("flag_when_selected",
                                                              cdef.get("flag_when_selected")))
                    item.options = {str(k): _parse_box(v, f"{where}:option {k}", allow_missing_boxes)
                                    for k, v in opts.items()}
                    d = it.get("details", None)
                    if d is not None or cdef_details:
                        d = d or {}
                        if d is False:
                            item.details = None
                        else:
                            item.details = DetailsSpec(
                                box=_parse_box(d.get("box"), f"{where}:details", allow_missing_boxes),
                                required_when=_as_list(d.get("required_when", cdef_details.get("required_when"))),
                                must_be_blank_when=_as_list(d.get("must_be_blank_when",
                                                                  cdef_details.get("must_be_blank_when"))),
                            )
                else:
                    item.box = _parse_box(it.get("box"), where, allow_missing_boxes)
                    if itype == "checkbox":
                        item.hit_box = _parse_box(it.get("hit_box"), where + ":hit_box", True) or item.box
                rr = it.get("rule_ref")
                if isinstance(rr, dict):        # {answer: B-1, flag: B-2, options: G-1, reading: G-2}
                    item.rule_ref_map = {str(k): str(v) for k, v in rr.items()}
                    rr = list(dict.fromkeys(item.rule_ref_map.values()))
                item.rule_refs = _as_list(rr)
                item.read = bool(it.get("read", False))
                val = it.get("value") or {}
                item.value_format = str(val.get("format", _DEFAULT_FORMAT.get(itype, "text")))
                if item.value_format not in VALUE_FORMATS:
                    raise ValueError(f"{where}: unknown value format '{item.value_format}'")
                item.allow_na = bool(val.get("allow_na", False))
                if itype == "choice":
                    item.printed = {str(k): _parse_box(v, f"{where}:printed {k}", False)
                                    for k, v in (it.get("printed") or {}).items()}
                    unknown = set(item.printed) - set(item.options)
                    if unknown:
                        raise ValueError(f"{where}: printed words {unknown} are not options")
                    if item.details is not None and isinstance(d, dict):
                        item.details.read = bool(d.get("read", False))
                        item.details.rule_refs = _as_list(d.get("rule_ref"))
                items.append(item)
            sections.append(Section(id=sid, title=str(s.get("title", sid)), items=items))

        pages.append(TemplatePage(page_id=page_id, image_path=img_path,
                                  sections=sections, rules=p.get("rules", []) or []))

    if not pages:
        raise ValueError(f"{path}: template defines no pages")
    rules = {}
    if raw.get("rules_csv"):
        csv_path = path.parent / raw["rules_csv"]
        if csv_path.is_file():
            rules = load_rules_csv(csv_path)
        else:                       # the reports then cite rule ids without their text
            import warnings
            warnings.warn(f"rules CSV not found: {csv_path}")
    document_rules = raw.get("document_rules", []) or []
    _check_rule_items(path, pages, document_rules)
    return FormTemplate(form_id=str(raw.get("form_id", path.stem)),
                        name=str(raw.get("name", "")),
                        version=str(raw.get("version", "")),
                        path=path, settings=settings, pages=pages, rules=rules,
                        document_rules=document_rules)


# keys of page / document rules that name items of the form
_RULE_ITEM_KEYS = ("item", "items", "na_checkbox", "checkbox", "field")


def _check_rule_items(path: Path, pages: list[TemplatePage], document_rules: list[dict]) -> None:
    """Every item a rule refers to must exist - a typo would otherwise switch a check off."""
    page_ids = {p.page_id: {i.id for s in p.sections for i in s.items} for p in pages}
    all_ids = set().union(*page_ids.values())
    todo = [(f"page {p.page_id} rule", r, page_ids[p.page_id]) for p in pages for r in p.rules]
    todo += [("document rule", r, all_ids) for r in document_rules]
    for kind, r, ids in todo:
        for key in _RULE_ITEM_KEYS:
            if key not in r or (key == "item" and r.get("type") == "block"):
                continue                                  # a block's `item` is just its name
            missing = [x for x in _as_list(r[key]) if x not in ids]
            if missing:
                raise ValueError(
                    f"{path.name}: {kind} '{r.get('id', r.get('type'))}' refers to "
                    f"{', '.join(missing)} ({key}), which is not an item of "
                    f"{'this page' if kind.startswith('page') else 'the form'}")


def load_rules_csv(path: Path) -> dict[str, RuleRef]:
    """Rules CSV with columns label (the rule id cited by rule_ref), category, instruction
    (other columns, e.g. page, are ignored)."""
    import csv
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            label = (r.get("label") or "").strip()
            if not label:
                continue
            text = " ".join((r.get("instruction") or "").split())
            out[label] = RuleRef(label, (r.get("category") or "").strip(), text,
                                 (r.get("region_id") or "").strip())
    return out
