"""All tunable knobs in one place.

Every value can be overridden in a template YAML under `settings:` (applies to the
whole form) or under an individual item's `overrides:` (applies to that field only).

Ink ratios are "fraction of visible ROI pixels that contain NEW ink", i.e. ink that
is present in the filled page but not in the blank template.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Settings:
    # ---- Input rendering -------------------------------------------------
    render_dpi: int = 200                 # PDF pages are rasterised at this DPI
    draw_pdf_annotations: bool = False    # False = hide PDF annotation layers (sticky
                                          # notes, callout boxes) before auditing

    # ---- Alignment (filled page -> blank template) -----------------------
    align_max_dim: int = 1000             # feature detection runs on images this size
    align_blur: float = 1.2               # Gaussian blur before SIFT - suppresses screen moiré
    sift_features: int = 8000
    ratio_test: float = 0.8
    ransac_reproj_px: float = 3.0         # at align_max_dim scale
    min_inliers: int = 40                 # below this the page is "unrecognised"
    ecc_refine: bool = True               # sub-pixel refinement after SIFT
    layout_min_recall: float = 0.5        # share of the template's table rules that must be
                                          # found on the page
    layout_max_extra: int = 3             # more unexplained table rules than this => a
                                          # different form / version (extra or moved rows)
    local_refine: bool = True             # snap table rules to the template (lens distortion)
    local_refine_radius: int = 22         # px search either side of each template rule
    local_refine_max_px: float = 40.0     # reject corrections larger than this

    # ---- Ink extraction --------------------------------------------------
    ink_contrast: float = 50.0            # how much darker than the local paper ink must be
    ink_blur: float = 1.5                 # blur (px) before measuring - cancels screen grids
    ink_bg_kernel: int = 41               # local paper colour = median over this window (odd)
    dark_max: int = 190                   # pixels brighter than this are never ink
    template_dilate_px: int = 7           # tolerance for residual misalignment
    recover_halo: bool = False            # experimental: give back pen ink in that margin next to
                                          # pen strokes (better tick/cross shapes, more false marks)
    print_core_px: int = 2                # margin kept around printed strokes when recovering
    min_component_px: int = 15            # drop specks smaller than this
    sliver_px: int = 3                    # inside a box, ignore marks this thin in either
                                          # direction (fragments of borders / box outlines)
    roi_inset_frac: float = 0.06          # shrink each box to avoid its cell borders
    strip_line_frac: float = 0.035        # remove straight H/V runs longer than this fraction
                                          # of page width (border slivers, box outlines);
                                          # 0 disables

    # ---- Coloured overlay masking (e.g. blue "Region" callouts) ----------
    mask_overlays: bool = True
    overlay_hue_min: int = 95             # OpenCV hue range 0..180 (blue ~ 120)
    overlay_hue_max: int = 135
    overlay_sat_min: int = 110
    overlay_val_min: int = 150            # keep high so dark blue pen ink survives
    min_visible_frac: float = 0.55        # less visible than this => "occluded"

    # ---- Presence decision (text / date / signature / details) -----------
    fill_lo: float = 0.004                # <= lo  -> blank
    fill_hi: float = 0.012                # >= hi  -> filled ; between -> REVIEW

    # ---- Choice decision (circled Yes / No / NA) -------------------------
    mark_lo: float = 0.006                # <= lo  -> unmarked
    mark_hi: float = 0.030                # >= hi  -> marked ; between -> faint
    mark_dominance: float = 2.0           # (legacy cell-ink method; unused by stroke analysis)

    # ---- Response marks (stroke analysis, see marks.py) -------------------
    mark_base_group_px: int = 15          # first pass: join stroke pieces this close
    mark_group_px: int = 15               # then join leftover fragments this close (printed text
                                          # removed from the scan cuts gaps into strokes)
    ring_bridge_px: int = 50              # max gap between arcs of one broken circle
    spill_merge_ratio: float = 0.35       # across a row line, merge a piece into its neighbour
                                          # only if it is this much smaller (a spilled circle)
    mark_min_ink: int = 60                # smaller stroke groups are specks
    circle_coverage: float = 0.5          # share of directions around its centre a ring covers
    circle_ring: float = 0.7              # share of its ink lying near its outline
    enclose_min: float = 0.5              # share of a printed word inside the circle
    strike_rule_band_px: int = 4          # cross-out test ignores ink this close to table rules
    strike_erase_px: int = 9              # width of the crossing strokes removed before
                                          # testing for a circle around a crossed-out word
    cancel_ring_coverage: float = 0.5     # a crossed-out word only counts as cancelled when a
                                          # circle covers this share of directions around it
    cancel_mark_means: str = "withdrawn"  # circled-then-crossed-out option: withdrawn (default:
                                          # the other circled option counts) | selected | review

    # ---- NA checkboxes (items with sub-items and a "[ ] NA" box) -------------
    checkbox_ink_lo: int = 30             # pen ink pixels in/around the box: <= lo -> not ticked
    checkbox_ink_hi: int = 80             # >= hi -> ticked ; between -> REVIEW
    na_checkbox_rules: bool = False       # False: the NA boxes are only recorded; an item
                                          # with sub-items must be complete once any part of
                                          # it is filled in.
                                          # True: it must be complete unless its NA box is
                                          # ticked, and the template's cross-checks apply (NA
                                          # ticked while the linked answer is Yes -> flagged).
                                          # A template whose form has the cross-checked items
                                          # turns this on itself.

    # ---- Details column / free-text extent --------------------------------
    details_overflow_px: int = 120        # how far right of the column writing is still collected
    text_join_x: int = 21                 # join letters into words / lines (px)
    text_join_y: int = 5
    overflow_note_min_px: int = 150       # only report overflow for blobs at least this big
    min_text_blob_px: int = 150           # one blob of writing this big = the field is filled

    # ---- Faint marks: a second, more sensitive ink pass ---------------------
    faint_check: bool = True              # a box / answer the normal pass sees as empty but
                                          # this pass sees marks in -> REVIEW (not FAIL/PASS)
    faint_ink_contrast: float = 25.0      # (normal pass: ink_contrast)
    faint_dark_max: int = 225             # (normal pass: dark_max)
    edge_margin_px: int = 7               # ignore ink this close to the edge of the photographed
                                          # area (the faint pass uses faint_edge_margin_px)
    faint_edge_margin_px: int = 35

    # ---- Writing outside the boxes -----------------------------------------
    outside_box_px: int = 60              # writing within this distance of a box, and in no
                                          # other box, is "beside" it (~0.3 in at 200 dpi)

    # ---- Azure Document Intelligence, page mode (--ocr azure) ----------------
    azure_model: str = "prebuilt-layout"
    azure_pages_per_request: int = 2      # images ("pages" to Azure) per request: the free
                                          # tier (F0) analyses only the first 2 of a request
    azure_pages_per_sheet: int = 3        # form pages tiled side by side on one image, so a
                                          # whole form goes in ONE request (3 pages -> 2 + 1);
                                          # 1 = one form page per image
    azure_max_request_mb: float = 4.0     # F0 limit (S0: 500)
    azure_page_scale: float = 1.5         # pages are sent straightened, at 1.5x the template
                                          # resolution (~300 dpi)
    azure_accept_conf: float = 0.8        # writing the reader finds in a box the ink check
                                          # sees as empty counts as filled from this confidence
    printed_word_density: float = 0.04    # a found word lying on this much of the form's own
                                          # print is a printed word, not writing
    pen_word_min_px: int = 20             # pen ink under a word that confirms it is writing

    # ---- Reading values (OCR) ---------------------------------------------
    ocr_scale: float = 2.0                # crops are taken from the original at 2x template res
    ocr_min_conf: float = 0.6             # below this a reading goes to REVIEW
    ocr_fail_conf: float = 0.85           # a format violation is FAIL only when read this surely
    date_order: str = "DMY"               # numeric dates: 14-10-26 = 14 Oct
    time_strict: bool = False             # True: HH:MM must have two-digit hours (09:05)
    check_date_format: bool = False       # True: a date that can't be read as a date -> REVIEW
                                          # (False: dates only need to be present; the value
                                          # read is still reported, ISO-normalised if possible)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    def merged(self, overrides: dict | None) -> "Settings":
        if not overrides:
            return self
        if "cancel_mark_means" in overrides and overrides["cancel_mark_means"] not in (
                "withdrawn", "selected", "review"):
            raise ValueError("cancel_mark_means must be withdrawn | selected | review")
        unknown = set(overrides) - self.field_names()
        if unknown:
            raise ValueError(f"Unknown setting(s): {sorted(unknown)}")
        return replace(self, **overrides)
