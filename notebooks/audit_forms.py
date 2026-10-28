# Databricks notebook source
# MAGIC %md
# MAGIC # Audit checklist forms
# MAGIC
# MAGIC Audits every PDF / image in **Input** and writes to **Output**:
# MAGIC - an annotated copy of each file (problems boxed and numbered);
# MAGIC - CSV reports: `all_findings.csv`, `batch_summary.csv`, `extracted_data.csv`, `rule_summary.csv`.
# MAGIC
# MAGIC **Paths** can be:
# MAGIC - `https://<account>.blob.core.windows.net/<container>/<folder>`, using the storage credentials in `.env`;
# MAGIC - `abfss://<container>@<account>.dfs.core.windows.net/<folder>`, using the storage access configured on the cluster;
# MAGIC - a mounted path such as `/dbfs/mnt/...`.
# MAGIC
# MAGIC **Setup (once):**
# MAGIC 1. Put the project folder in the workspace, e.g. as a Git folder.
# MAGIC 2. Next to `requirements.txt`, create a file named `.env` from `.env.example`, containing the Azure Document Intelligence endpoint and key and, for `https://` paths, the storage connection string.
# MAGIC 3. Upload your template files (`form.yaml`, `form_p*.png`, `form_rules.csv`) into the project's `templates/` folder. They are not in git.
# MAGIC
# MAGIC If the next cell can't find `../requirements.txt`, give the full path, e.g. `%pip install -r /Workspace/Users/<you>/form_auditor/requirements.txt`.
# MAGIC
# MAGIC Runs on classic compute (single node is enough); about 5–10 s per page, plus the handwriting reading.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("input", "https://<account>.blob.core.windows.net/<container>/incoming/", "Input (file or folder)")
dbutils.widgets.text("output", "https://<account>.blob.core.windows.net/<container>/audited/", "Output folder")
dbutils.widgets.dropdown("ocr", "azure", ["azure", "azure-read", "none"], "Read handwriting")
dbutils.widgets.text("pages_per_request", "2", "Images per Azure request (free tier: 2; 3 form pages per image)")
dbutils.widgets.dropdown("show_pass", "no", ["no", "yes"], "Also box passing checks")
dbutils.widgets.text("project", "", "Project folder (blank = parent of this notebook)")

# COMMAND ----------

import os
import sys
from datetime import datetime

# the project folder = the parent of this notebook's folder, unless given
project = dbutils.widgets.get("project").strip() or os.path.dirname(os.getcwd())
if not os.path.isdir(os.path.join(project, "form_auditor")):
    raise FileNotFoundError(f"form_auditor not found in {project} - set the 'Project folder' "
                            "widget, e.g. /Workspace/Users/<you>/form_auditor")
if project not in sys.path:
    sys.path.insert(0, project)

import cv2  # noqa: E402  (fails with "libGL.so.1" if the GUI build opencv-python is installed: %pip uninstall -y opencv-python)
from form_auditor.api import run_audit  # noqa: E402

inp = dbutils.widgets.get("input").strip()
out = dbutils.widgets.get("output").strip().rstrip("/") + "/" + datetime.now().strftime("%Y%m%d_%H%M%S")

env_file = os.path.join(project, ".env")
if not os.path.isfile(env_file):
    print(f"No .env file in {project} - keys must come from the cluster's environment variables")
    env_file = None

result = run_audit(
    inp, out,
    ocr=dbutils.widgets.get("ocr"),
    env_file=env_file,
    show_pass=dbutils.widgets.get("show_pass") == "yes",
    settings={"azure_pages_per_request": int(dbutils.widgets.get("pages_per_request") or 2)},
)

# COMMAND ----------

# MAGIC %md ## Results

# COMMAND ----------

import pandas as pd  # noqa: E402

if result["stopped"]:
    print("STOPPED:", result["stopped"])
print(f"Reports written to {result['out']}")
display(pd.DataFrame(result["summary"]))

# COMMAND ----------

findings = pd.DataFrame(result["findings"])
if len(findings):
    display(findings[findings["status"].isin(["FAIL", "REVIEW"])][
        ["file", "page", "marker", "section", "item_id", "check", "status", "observed", "reason",
         "value_read", "rule_ref"]])
