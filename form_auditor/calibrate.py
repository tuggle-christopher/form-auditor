"""Template authoring helpers.

render  : rasterise a blank form (PDF/image) into the template PNG(s)
calibrate: walk through every box in a template YAML and drag it on screen
preview : draw all template boxes on the blank page to verify them
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .io_utils import load_pages
from .template import read_yaml


# ---------- YAML helpers: keep [x0, y0, x1, y1] on one line --------------- #
class _FlowList(list):
    pass


def _flow_repr(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.SafeDumper.add_representer(_FlowList, _flow_repr)


def _flowify(obj):
    if isinstance(obj, dict):
        return {k: _flowify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) for v in obj) or (obj and all(isinstance(v, str) for v in obj) and len(obj) <= 6):
            return _FlowList(obj)
        return [_flowify(v) for v in obj]
    return obj


def dump_yaml(data: dict, path: Path, header: str = "") -> None:
    text = yaml.safe_dump(_flowify(data), sort_keys=False, allow_unicode=True, width=120)
    path.write_text(header + text)


# --------------------------------------------------------------------------- #
def render_template(src: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pg in load_pages(src, dpi=dpi, draw_annotations=False):
        p = out_dir / f"{src.stem}_p{pg.page_index + 1}.png"
        cv2.imwrite(str(p), pg.image)
        paths.append(p)
        print(f"wrote {p}  ({pg.image.shape[1]}x{pg.image.shape[0]})")
    return paths


def _box_slots(raw: dict):
    """Yield (page, description, getter, setter) for every box in the raw YAML."""
    for page in raw.get("pages", []):
        for sec in page.get("sections", []):
            for it in sec.get("items", []):
                iid = it["id"]
                label = it.get("label", iid)
                if str(it.get("type", "text")).lower() == "choice":
                    for opt in list((it.get("options") or {}).keys()):
                        yield (page, f"{iid} - option '{opt}'  ({label})",
                               lambda it=it, o=opt: it["options"][o],
                               lambda v, it=it, o=opt: it["options"].__setitem__(o, v))
                    if it.get("details", None) is not False:
                        it.setdefault("details", {})
                        if it["details"] is None:
                            it["details"] = {}
                        yield (page, f"{iid} - details / follow-up box  ({label})",
                               lambda it=it: it["details"].get("box"),
                               lambda v, it=it: it["details"].__setitem__("box", v))
                else:
                    yield (page, f"{iid} - {it.get('type', 'text')}  ({label})",
                           lambda it=it: it.get("box"),
                           lambda v, it=it: it.__setitem__("box", v))


def calibrate(yaml_path: Path, only: str | None = None, redo: bool = False,
              max_display: int = 1000) -> None:
    raw = read_yaml(yaml_path)
    images: dict[str, np.ndarray] = {}
    win = "calibrate: drag box, ENTER/SPACE=accept, c=skip, ESC on empty=quit"
    print(__doc__)
    print("Tips: draw OPTION boxes around the whole Yes/No/NA cell (circles spill over);\n"
          "      draw DETAIL/TEXT boxes inside the cell borders.\n")
    for page, desc, get, set_ in _box_slots(raw):
        if only and not desc.startswith(only):
            continue
        if get() is not None and not redo:
            continue
        ip = (yaml_path.parent / page["template_image"]).resolve()
        if str(ip) not in images:
            images[str(ip)] = cv2.imread(str(ip))
        img = images[str(ip)]
        h, w = img.shape[:2]
        k = min(1.0, max_display / max(h, w))
        disp = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        cv2.putText(disp, desc, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print("Draw:", desc)
        x, y, bw, bh = cv2.selectROI(win, disp, showCrosshair=False)
        if bw == 0 or bh == 0:
            print("  skipped")
            continue
        box = [round(x / k / w, 4), round(y / k / h, 4), round((x + bw) / k / w, 4), round((y + bh) / k / h, 4)]
        set_(box)
        dump_yaml(raw, yaml_path)          # save after every box
        print("  ->", box)
    cv2.destroyAllWindows()
    print(f"Saved {yaml_path}")


def preview(yaml_path: Path, out: Path) -> list[Path]:
    """Draw every box with its id on the blank template page(s)."""
    raw = read_yaml(yaml_path)
    outs = []
    colors = {"option": (200, 120, 0), "details": (0, 140, 255), "box": (0, 160, 0),
              "printed": (200, 0, 200)}
    for pi, page in enumerate(raw.get("pages", [])):
        img = cv2.imread(str((yaml_path.parent / page["template_image"]).resolve()))
        h, w = img.shape[:2]
        for sec in page.get("sections", []):
            for it in sec.get("items", []):
                boxes = []
                if str(it.get("type", "text")).lower() == "choice":
                    boxes += [(f"{it['id']}:{o}", b, "option") for o, b in (it.get("options") or {}).items()]
                    boxes += [("", b, "printed") for o, b in (it.get("printed") or {}).items()]
                    if isinstance(it.get("details"), dict):
                        boxes.append((f"{it['id']}:details", it["details"].get("box"), "details"))
                else:
                    boxes.append((it["id"], it.get("box"), "box"))
                    if it.get("hit_box"):
                        boxes.append(("", it["hit_box"], "printed"))
                for name, b, kind in boxes:
                    if b is None:
                        continue
                    p0 = (int(b[0] * w), int(b[1] * h))
                    p1 = (int(b[2] * w), int(b[3] * h))
                    cv2.rectangle(img, p0, p1, colors[kind], 2)
                    cv2.putText(img, name, (p0[0] + 2, p0[1] + 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, colors[kind], 1, cv2.LINE_AA)
        p = out if len(raw["pages"]) == 1 else out.with_name(f"{out.stem}_p{pi + 1}{out.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        outs.append(p)
        print(f"wrote {p}")
    return outs
