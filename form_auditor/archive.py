"""A record of every run, kept for yourself: all the CSV reports of the run, complete (the
document report with every column, including the ones left out of the copy people get),
copied to a folder of their own in blob storage (or any folder).

Where to: the `archive` argument of run_audit / `--archive`, else the environment variable
FORM_AUDITOR_ARCHIVE (e.g. in .env, or a Databricks App secret). Any destination storage.py
understands:

    https://<account>.blob.core.windows.net/<container>/<folder>   (azure-storage-blob, with
        AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_SAS_TOKEN or AZURE_STORAGE_KEY)
    abfss://<container>@<account>.dfs.core.windows.net/<folder>     (Databricks clusters)
    /Volumes/..., /dbfs/..., or a local folder

Each run goes to <archive>/<yyyy>/<mm>/<yyyymmdd-hhmmss>_<id>/ :
    run_info.csv           when, how many files / documents, results, reader used, where from
    document_report.csv    every column
    all_findings.csv, batch_summary.csv, extracted_data.csv, rule_summary.csv
    <file>/<file>_audit.csv  per input file
Only CSVs: no forms, PDFs or pictures are archived.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .report import write_csv
from .storage import is_remote, upload

ENV_VAR = "FORM_AUDITOR_ARCHIVE"


def archive_destination(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get(ENV_VAR, "")).strip()


def archive_run(out_local: Path, documents: list[dict], doc_cols: list[str], dest: str,
                info: dict, now: Optional[datetime] = None) -> str:
    """Copy this run's CSVs (with the full document report) to a new folder under `dest`.
    Returns where they went."""
    now = now or datetime.now().astimezone()
    run_id = f"{now:%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:6]}"
    rel = f"{now:%Y}/{now:%m}/{run_id}"
    tmp = Path(tempfile.mkdtemp(prefix="form_auditor_archive_"))
    try:
        stage = tmp / run_id
        stage.mkdir()
        for f in sorted(Path(out_local).rglob("*.csv")):
            target = stage / f.relative_to(out_local)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        write_csv(stage / "document_report.csv", documents, doc_cols)       # every column
        rows = [{"key": "run_id", "value": run_id}, {"key": "time", "value": now.isoformat(timespec="seconds")}]
        rows += [{"key": k, "value": v} for k, v in info.items()]
        write_csv(stage / "run_info.csv", rows, ["key", "value"])
        target = dest.rstrip("/") + "/" + rel
        if is_remote(dest):
            upload(stage, target)
        else:
            Path(target).mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, target, dirs_exist_ok=True)
        return target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
