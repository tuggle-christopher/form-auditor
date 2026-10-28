"""Python entry point - the same as the `audit` command, callable from a notebook.

    from form_auditor.api import run_audit
    result = run_audit(["abfss://forms@myacct.dfs.core.windows.net/incoming/"],
                       "abfss://forms@myacct.dfs.core.windows.net/audited/",
                       ocr="azure")
    result["summary"]          # one dict per page
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .env import PROJECT_ROOT, load_env
from .io_utils import iter_input_files
from .ocr import FatalOcrError, make_engine
from .pipeline import Auditor
from .report import FINDING_COLUMNS, RULE_SUMMARY_COLUMNS, SUMMARY_COLUMNS, rule_summary, write_csv
from .settings import Settings
from .storage import local_output_dir, stage_inputs, upload
from .template import load_template

TEMPLATE_DIR = PROJECT_ROOT / "templates"


def default_templates() -> list[Path]:
    """Every template in templates/ (YAML files with pages - not the form config). With
    an old and a new version both there, each page is matched to the one it fits."""
    from .template import read_yaml
    out = []
    for p in sorted(TEMPLATE_DIR.glob("*.yaml")):
        try:
            if (read_yaml(p) or {}).get("pages"):
                out.append(p)
        except Exception:
            continue
    if not out:
        raise FileNotFoundError(
            f"No templates in {TEMPLATE_DIR}. Build one from the blank form: "
            f"python tools/build_template_from_pdf.py <blank form PDF>  (see templates/README.md)")
    return out


def parse_setting(key: str, value) -> object:
    """'fill_hi', '0.02' -> 0.02 (typed like the default)."""
    if key not in Settings.field_names():
        raise ValueError(f"Unknown setting: {key}")
    default = getattr(Settings(), key)
    if not isinstance(value, str):
        return value
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return type(default)(value)


def run_audit(inputs: list[str | Path] | str | Path,
              out: str | Path = "audit_out",
              templates: Optional[list[str | Path]] = None,
              ocr: str = "none",
              settings: Optional[dict] = None,
              *,
              env_file: Optional[str | Path] = None,
              azure_endpoint: Optional[str] = None,
              azure_api_version: str = "2024-11-30",
              ocr_model: Optional[str] = None,
              ocr_mock_file: Optional[str] = None,
              cancel_means: Optional[str] = None,
              show_pass: bool = False,
              debug: bool = False,
              skip_unrecognised: bool = False,
              keep_pdf_annotations: bool = False,
              dpi: Optional[int] = None,
              verbose: bool = True) -> dict:
    """Audit files / folders (local paths, /dbfs/... paths, abfss://, wasbs://, dbfs:/ or
    https://<account>.blob.core.windows.net/... URLs) and write the reports to `out`
    (local or remote, same schemes).

    `settings` overrides values of form_auditor/settings.py, e.g.
    {"na_checkbox_rules": True, "date_order": "MDY"}.

    Returns {"out", "pages", "failing", "review", "unrecognised", "findings", "summary",
    "extracted", "stopped"}.
    """
    load_env(env_file)
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    overrides = {k: parse_setting(k, v) for k, v in (settings or {}).items()}
    if cancel_means:
        overrides["cancel_mark_means"] = cancel_means
    base = Settings().merged(overrides)
    tpls = [load_template(t, base) for t in (templates or default_templates())]
    for t in tpls:                                   # caller's settings win over the YAML's
        t.settings = t.settings.merged(overrides)

    engine = make_engine(ocr, ocr_model, ocr_mock_file, azure_endpoint, azure_api_version,
                         tpls[0].settings.azure_model)
    auditor = Auditor(tpls, verbose=verbose, ocr=engine)

    tmp = Path(tempfile.mkdtemp(prefix="form_auditor_"))
    stopped = ""
    try:
        files = iter_input_files(stage_inputs(list(inputs), tmp))
        out_local, out_remote = local_output_dir(out, tmp)
        rows_all, summary_all, records_all = [], [], []
        for f in files:
            try:
                rows, summ, recs = auditor.audit_file(
                    f, out_local, show_pass=show_pass, debug=debug,
                    skip_unrecognised=skip_unrecognised, dpi=dpi,
                    draw_pdf_annotations=keep_pdf_annotations or None)
            except FatalOcrError as e:
                stopped = str(e)
                break
            except Exception as e:                       # keep the batch going
                if verbose:
                    print(f"  ERROR {f}: {e}")
                summ = [{"file": f.name, "page": "", "page_status": "ERROR", "note": str(e)}]
                rows, recs = [], []
            rows_all += rows
            summary_all += summ
            records_all += recs

        write_csv(out_local / "all_findings.csv", rows_all, FINDING_COLUMNS)
        write_csv(out_local / "batch_summary.csv", summary_all, SUMMARY_COLUMNS)
        write_csv(out_local / "extracted_data.csv", records_all)
        rules, inactive = {}, {}
        for t in tpls:
            rules.update(t.rules)
            used = t.referenced_rules()
            for ref in t.rules:
                if ref not in used:
                    inactive.setdefault(ref, "not checked - nothing on this version of the form "
                                             "uses this rule")
            for r in t.document_rules + [r for p in t.pages for r in p.rules]:
                req = r.get("requires_setting")
                if req and not getattr(t.settings, req):
                    for ref in r.get("rule_ref", []):
                        inactive[ref] = f"not checked - switched off ({req}=false)"
        write_csv(out_local / "rule_summary.csv", rule_summary(rows_all, rules, inactive),
                  RULE_SUMMARY_COLUMNS)
        if out_remote:
            upload(out_local, out_remote)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    result = {
        "out": str(out),
        "pages": len(summary_all),
        "failing": sum(s.get("page_status") in ("FAIL", "ERROR") for s in summary_all),
        "review": sum(s.get("page_status") == "REVIEW" for s in summary_all),
        "unrecognised": sum(s.get("page_status") == "UNRECOGNISED" for s in summary_all),
        "findings": rows_all, "summary": summary_all, "extracted": records_all,
        "stopped": stopped,
    }
    if verbose:
        print(f"\n{result['pages']} page(s): {result['failing']} failing, {result['review']} need "
              f"review, {result['unrecognised']} not recognised as a known form page. "
              f"Reports in {out}/" + (f"\nSTOPPED: {stopped}" if stopped else ""))
    return result
