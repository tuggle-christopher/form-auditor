"""Show why the tool reads a row the way it does.

    python -m form_auditor explain input/form1.pdf --item E5 -o explain_out

For each page holding the item, writes `<file>_p<N>_<item>.png`: the row and its
neighbours as lined up with the blank, next to what the tool sees there (the blank's print
in light blue, the pen ink it found in red, ink only the sensitive pass found in orange,
each mark boxed and labelled), and prints how each mark was read. It also measures how far
the page and the blank still differ around the row: a few pixels or more there means the
print was not fully removed, and leftover print can look like a mark.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .details import read_section_details
from .ink import box_to_px
from .io_utils import iter_input_files, load_pages
from .marks import read_section_choices
from .pipeline import Auditor
from .template import load_template

KIND_COLOUR = {"circle": (0, 150, 0), "partial-circle": (0, 150, 0), "tick": (160, 90, 0),
               "cross": (160, 90, 0), "cancel": (120, 0, 160), "written": (160, 90, 0),
               "spill": (0, 120, 230), "speck": (140, 140, 140)}


def _cells(item, shape):
    boxes = [b for b in (item.options or {}).values() if b]
    if item.box is not None:
        boxes.append(item.box)
    if item.details is not None and item.details.box:
        boxes.append(item.details.box)
    return [box_to_px(b, shape) for b in boxes]


def _offset(blank_g: np.ndarray, page_g: np.ndarray) -> tuple[float, float, float]:
    """How far the page's print sits from the blank's in this crop (dx, dy px, response)."""
    a = (255 - blank_g).astype(np.float32)
    b = (255 - page_g).astype(np.float32)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0, 0.0, 0.0
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def explain_page(res, item_id: str, out_png: Path) -> list[str]:
    rt, ctx = res.runtime, res.ctx
    s = rt.template.settings
    sec = next((x for x in rt.page.sections if any(i.id == item_id for i in x.items)), None)
    if sec is None:
        return []
    items = [i for i in sec.items if i.type == "choice"]
    item = next(i for i in sec.items if i.id == item_id)
    lines = [f"{item_id} ({item.label})  page {res.page_index + 1}, section {sec.id}"]
    shape = ctx.shape
    # rows shown: the item and its neighbours in the section
    k = items.index(item) if item in items else -1
    show = items[max(0, k - 1):k + 2] if k >= 0 else [item]
    boxes = [b for i in show for b in _cells(i, shape)]
    if not boxes:
        return lines + ["  (no boxes for this item in the template)"]
    x0 = max(0, min(b[0] for b in boxes) - 40); y0 = max(0, min(b[1] for b in boxes) - 30)
    x1 = min(shape[1], max(b[2] for b in boxes) + 40); y1 = min(shape[0], max(b[3] for b in boxes) + 30)

    warped = ctx.maps.warped[y0:y1, x0:x1]
    blank = rt.page.image[y0:y1, x0:x1]
    added = ctx.maps.added[y0:y1, x0:x1]
    faint = ctx.faint.added[y0:y1, x0:x1] & ~added if ctx.faint is not None else np.zeros_like(added)
    occl = ctx.maps.occluded[y0:y1, x0:x1]

    # what the tool sees
    vis = np.full_like(warped, 255)
    printed = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY) < 150
    vis[printed] = (235, 200, 160)                     # blank's print: light blue
    vis[occl] = (200, 200, 200)
    vis[faint] = (0, 150, 255)                         # sensitive pass only: orange
    vis[added] = (30, 30, 220)                         # pen ink: red
    choices = read_section_choices(ctx, sec.items, s) if items else {}
    details = read_section_details(ctx, sec.items, s) if items else {}

    big = 2
    W = cv2.resize(warped, None, fx=big, fy=big, interpolation=cv2.INTER_NEAREST)
    V = cv2.resize(vis, None, fx=big, fy=big, interpolation=cv2.INTER_NEAREST)
    for i in show:
        for b in _cells(i, shape):
            cv2.rectangle(V, ((b[0] - x0) * big, (b[1] - y0) * big), ((b[2] - x0) * big, (b[3] - y0) * big),
                          (190, 190, 190), 1)
        rd = choices.get(i.id)
        if rd is None:
            continue
        for n, m in enumerate(rd.marks, start=1):
            if m.kind == "speck":
                continue
            mx0, my0, mx1, my1 = m.bbox
            c = KIND_COLOUR.get(m.kind, (0, 0, 0))
            p0 = ((mx0 - x0) * big - 3, (my0 - y0) * big - 3)
            p1 = ((mx1 - x0) * big + 3, (my1 - y0) * big + 3)
            cv2.rectangle(V, p0, p1, c, 2)
            lab = f"{i.id}: {m.kind}" + (f" on {m.option}" if m.option else "")
            cv2.putText(V, lab, (p0[0], max(14, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    def head(img, text):
        bar = np.full((34, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        return np.vstack([bar, img])
    legend = "red = pen ink, orange = faint ink only, light blue = blank's print, boxes = marks (unboxed red is ignored)"
    out = np.vstack([head(W, "The page, lined up with the blank"), np.full((12, W.shape[1], 3), 255, np.uint8),
                     head(V, "What the tool sees: " + legend)])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), out)

    # how well the page sits on the blank here
    dx, dy, resp = _offset(cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY), cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    off = (dx * dx + dy * dy) ** 0.5
    lines.append(f"  lining up around this row: the page's print sits {off:.1f} px from the blank's "
                 f"(dx {dx:+.1f}, dy {dy:+.1f}); ink within {s.template_dilate_px} px of the print is "
                 f"removed as print" + ("  <-- more than that: print may be left over" if off > s.template_dilate_px - 2 else ""))
    for i in show:
        rd = choices.get(i.id)
        if rd is None:
            continue
        tag = "  >>" if i.id == item_id else "    "
        lines.append(f"{tag} {i.id}: read as {rd.state}" + (f" '{rd.selected}'" if rd.selected else "")
                     + (f" ({', '.join(rd.options)})" if getattr(rd, 'options', ()) else "")
                     + (f" - {rd.note}" if rd.note else ""))
        for m in rd.marks:
            if m.kind == "speck":
                continue
            mx0, my0, mx1, my1 = m.bbox
            sub = added[max(0, my0 - y0):max(0, my1 - y0), max(0, mx0 - x0):max(0, mx1 - x0)]
            npx = int(sub.sum())
            enc = ", ".join(f"{o} {v:.2f}" for o, v in sorted(m.enclosure.items()) if v > 0)
            lines.append(f"{tag}   - {m.kind}" + (f" on {m.option}" if m.option else "")
                         + f", confidence {m.confidence:.2f}, box {m.bbox}, {npx} ink px"
                         + (f", surrounds: {enc}" if enc else "")
                         + (f" - {m.note}" if m.note else ""))
        dr = details.get(i.id)
        if dr is not None:
            lines.append(f"{tag}   details: {dr.state}" + (f" (may run on from {dr.continuation_of})"
                                                        if dr.continuation_of else ""))
    lines.append(f"  picture: {out_png}")
    return lines


def explain(inputs: list[str], item_ids: list[str], out_dir: str,
            templates: Optional[list[str]] = None, page: Optional[int] = None) -> int:
    from .api import default_templates
    tpls = [load_template(Path(t)) for t in (templates or default_templates())]
    aud = Auditor(tpls, verbose=False)
    out = Path(out_dir)
    found = 0
    for f in iter_input_files([Path(p) for p in inputs]):
        prev = None
        for pg in load_pages(f):
            if page and pg.page_index + 1 != page:
                continue
            res = aud.analyse_page(pg, prefer=prev)
            if res.runtime is None:
                print(f"{f.name} page {pg.page_index + 1}: not recognised as a page of the form")
                continue
            prev = res.runtime.template
            for iid in item_ids:
                png = out / f"{f.stem}_p{pg.page_index + 1}_{iid.replace('.', '_')}.png"
                lines = explain_page(res, iid, png)
                if lines:
                    found += 1
                    print(f"\n{f.name}")
                    print("\n".join(lines))
    if not found:
        print(f"None of {', '.join(item_ids)} found on a recognised page.")
    print("\nThis shows the ink reading only (no handwriting reader); it is what decides "
          "which option is marked.")
    return 0 if found else 1
