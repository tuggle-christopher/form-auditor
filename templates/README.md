# templates/

This folder holds everything specific to your form. It is **kept out of git** (see
`.gitignore`); only this README is committed.

| file | what it is |
|---|---|
| `form_config.yaml` | your settings for the template builder (start from `tools/form_config.example.yaml`) |
| `form.yaml` | the template: where every box, answer cell and sign-off is, and which rules apply |
| `form_p1.png`, `form_p2.png`, … | the blank pages that scans are matched against |
| `form_rules.csv` | the rule texts quoted in the reports |
| `FORM_NOTES.md` | what is checked on your form (optional, for people) |
| `form_v2.yaml`, `form_v2_p*.png` | a second version of the form, if you have one |

To build a template from the blank form PDF (keep the PDF in `forms/`, also git-ignored):

```bash
python tools/build_template_from_pdf.py forms/blank.pdf
```

The rules CSV and the config are written by you, not by the builder, so keep a copy of
them somewhere safe, along with the blank PDF.

On Databricks, upload these files into this folder of the workspace copy of the project.
Workspace files that git ignores stay in the workspace and are never committed.
