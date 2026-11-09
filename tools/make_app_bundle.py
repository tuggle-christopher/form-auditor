"""Copy just what the web app needs into a clean folder, to deploy as a Databricks App.

    python tools/make_app_bundle.py /Workspace/Users/<you>/form-auditor-app

Copies app.py, app.yaml, the form_auditor package, your templates (the blank form's
template, page images and rules) and the app's look (.streamlit/config.toml, branding/,
static/), and writes the app's requirements.txt (the project's, plus Streamlit). Leaves out notebooks, tests, inputs, outputs and .env, so no forms or keys
end up in the app, and checks every file is under the 10 MB limit for app files.
Run it again after changing the code or the templates, then redeploy.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10 * 1024 * 1024
TEMPLATE_TYPES = {".yaml", ".yml", ".png", ".csv", ".json"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", help="folder to create / refresh (its contents are replaced)")
    a = ap.parse_args(argv)
    dst = Path(a.target)
    if dst.resolve() == ROOT or ROOT in dst.resolve().parents:
        raise SystemExit("Choose a folder outside the project.")
    tpl = ROOT / "templates"
    if not any(tpl.glob("*.yaml")):
        raise SystemExit("No template in templates/ - build it first (tools/build_template_from_pdf.py).")
    if dst.exists():
        for p in ("app.py", "app.yaml", "requirements.txt", "form_auditor", "templates",
                  ".streamlit", "branding", "static"):
            q = dst / p
            shutil.rmtree(q) if q.is_dir() else q.unlink(missing_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "app.py", dst / "app.py")
    shutil.copy2(ROOT / "app.yaml", dst / "app.yaml")
    shutil.copytree(ROOT / "form_auditor", dst / "form_auditor",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst / "templates").mkdir()
    for p in tpl.iterdir():
        if p.is_file() and p.suffix.lower() in TEMPLATE_TYPES:
            shutil.copy2(p, dst / "templates" / p.name)
    # the look: theme (your config.toml, else the example), branding and static files
    (dst / ".streamlit").mkdir()
    theme = ROOT / ".streamlit" / "config.toml"
    shutil.copy2(theme if theme.is_file() else ROOT / ".streamlit" / "config.example.toml",
                 dst / ".streamlit" / "config.toml")
    if not theme.is_file():
        print("  no .streamlit/config.toml - using the example theme")
    shutil.copytree(ROOT / "branding", dst / "branding")
    if not (ROOT / "branding" / "branding.yaml").is_file():
        print("  no branding/branding.yaml - using the example wording, no logo")
    if (ROOT / "static").is_dir():
        shutil.copytree(ROOT / "static", dst / "static")
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").rstrip()
    (dst / "requirements.txt").write_text(reqs + "\n\n# the web app\nstreamlit>=1.40\n", encoding="utf-8")

    big = [p for p in dst.rglob("*") if p.is_file() and p.stat().st_size > LIMIT]
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"App bundle ready in {dst} ({n} files, templates: "
          f"{', '.join(sorted(p.name for p in (dst / 'templates').glob('*.yaml')))})")
    if big:
        print("Too big for a Databricks App (10 MB per file): " + ", ".join(str(p) for p in big))
        return 1
    print("Deploy it: Compute > Apps > your app > Deploy, source code path = this folder, "
          f"or: databricks apps deploy <app name> --source-code-path {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
