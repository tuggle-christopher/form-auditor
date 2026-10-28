"""CLI.

  python -m form_auditor audit  INPUT [INPUT ...] -o audit_out [--ocr azure]
  python -m form_auditor preview   templates/form.yaml -o preview.png
  python -m form_auditor render    blank_form.pdf -o templates/
  python -m form_auditor calibrate templates/form.yaml   (desktop only - needs a GUI)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibrate import calibrate, preview, render_template
from .api import run_audit


def _cmd_audit(a) -> int:
    overrides = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()
    if a.na_checkbox_rules:
        overrides["na_checkbox_rules"] = True
    if a.azure_pages_per_request:
        overrides["azure_pages_per_request"] = a.azure_pages_per_request
    res = run_audit(a.inputs, a.out, templates=a.template, ocr=a.ocr, settings=overrides,
                    env_file=a.env_file, azure_endpoint=a.azure_endpoint,
                    azure_api_version=a.azure_api_version, ocr_model=a.ocr_model,
                    ocr_mock_file=a.ocr_mock_file, cancel_means=a.cancel_means,
                    show_pass=a.show_pass, debug=a.debug,
                    skip_unrecognised=a.skip_unrecognised,
                    keep_pdf_annotations=a.keep_pdf_annotations, dpi=a.dpi,
                    verbose=not a.quiet)
    if res["stopped"]:
        print(f"\nSTOPPED: {res['stopped']}", file=sys.stderr)
        return 2
    return 1 if (res["failing"] and a.strict) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="form_auditor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    au = sub.add_parser("audit", help="audit filled forms (PDFs, images, or folders)")
    au.add_argument("inputs", nargs="+")
    au.add_argument("-t", "--template", nargs="+", default=None,
                    help="template YAML(s); default: every template in templates/. With "
                         "several (e.g. an old and a new version of the form) each page is "
                         "matched to the one it fits best")
    au.add_argument("-o", "--out", default="audit_out",
                    help="output folder (local, /dbfs/..., abfss://, wasbs://, dbfs:/ or "
                         "https://<account>.blob.core.windows.net/...)")
    au.add_argument("--env-file", default=None,
                    help=".env file with AZURE_DI_ENDPOINT / AZURE_DI_KEY etc. "
                         "(default: ./.env or the project's .env)")
    au.add_argument("--na-checkbox-rules", action="store_true",
                    help="turn on the NA-checkbox rules (items with sub-items required unless "
                         "their NA box is ticked; cross-checks apply). A template built from a "
                         "form that has the cross-checked items turns them on itself")
    au.add_argument("--dpi", type=int, default=None, help="PDF render DPI (default from template)")
    au.add_argument("--show-pass", action="store_true", help="also draw passing fields in green")
    au.add_argument("--debug", action="store_true", help="write aligned/ink-mask images for tuning")
    au.add_argument("--keep-pdf-annotations", action="store_true",
                    help="render PDF annotation layers (hidden by default)")
    au.add_argument("--ocr", default="none",
                    choices=["none", "azure", "azure-read", "claude", "tesseract", "mock", "mock-layout"],
                    help="handwriting reader: azure (Azure AI Document Intelligence Layout on whole "
                         "pages, combined with the ink check; needs AZURE_DI_ENDPOINT + "
                         "AZURE_DI_KEY), azure-read (Azure Read on crops of the filled boxes), "
                         "claude (Anthropic API), tesseract (offline), mock / mock-layout "
                         "(JSON answers, tests), none (default)")
    au.add_argument("--azure-pages-per-request", type=int, default=None,
                    help="pages per Azure request (default 2 = the free tier's limit; on the "
                         "paid tier e.g. 20)")
    au.add_argument("--azure-endpoint", default=None,
                    help="Document Intelligence endpoint (default: AZURE_DI_ENDPOINT)")
    au.add_argument("--azure-api-version", default="2024-11-30",
                    help="Document Intelligence REST API version (default 2024-11-30)")
    au.add_argument("--ocr-model", default=None, help="model for --ocr claude (default claude-sonnet-5-5)")
    au.add_argument("--ocr-mock-file", default=None, help="JSON answers for --ocr mock")
    au.add_argument("--cancel-means", choices=["withdrawn", "selected", "review"], default=None,
                    help="an option circled then crossed out: withdrawn (default - the other "
                         "circled option counts), selected, or review")
    au.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a setting, e.g. --set fill_hi=0.02 (repeatable)")
    au.add_argument("--strict", action="store_true", help="exit code 1 if any page fails")
    au.add_argument("--skip-unrecognised", action="store_true",
                    help="leave pages that match no template out of the annotated PDF")
    au.add_argument("-q", "--quiet", action="store_true")
    au.set_defaults(func=_cmd_audit)

    r = sub.add_parser("render", help="rasterise a BLANK form into template PNGs")
    r.add_argument("blank")
    r.add_argument("-o", "--out", default="templates")
    r.add_argument("--dpi", type=int, default=200)
    r.set_defaults(func=lambda a: render_template(Path(a.blank), Path(a.out), a.dpi) and 0)

    c = sub.add_parser("calibrate", help="draw field boxes on the blank template (GUI)")
    c.add_argument("yaml")
    c.add_argument("--only", help="only boxes whose description starts with this, e.g. B3")
    c.add_argument("--redo", action="store_true", help="redraw boxes that already exist")
    c.set_defaults(func=lambda a: calibrate(Path(a.yaml), a.only, a.redo) or 0)

    v = sub.add_parser("preview", help="draw all template boxes on the blank page")
    v.add_argument("yaml")
    v.add_argument("-o", "--out", default="template_preview.png")
    v.set_defaults(func=lambda a: preview(Path(a.yaml), Path(a.out)) and 0)

    a = p.parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
