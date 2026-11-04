# tools/

`build_template_from_pdf.py` builds a template (YAML + one image per page) straight from
the blank form PDF, using your form config (`templates/form_config.yaml`; the example is
`form_config.example.yaml` here):

```bash
python tools/build_template_from_pdf.py forms/blank.pdf                      # -> templates/form.yaml
python tools/build_template_from_pdf.py forms/blank_v2.pdf --out templates/form_v2.yaml --version 2
```

It derives everything else from the PDF itself:

- **Cells** come from the ruled lines of the rendered page.
- **Sections** come from the "Section X – Title" headings.
- **Answer rows** come from the printed `Yes  No [NA]` words in the PDF text layer,
  including words split over two lines ("Ye/s"). The exact box of each printed word is
  stored and used to decide which word a circle surrounds. NA is set to fail on rows
  where it isn't printed.
- **Fields from printed markers:** a "… (HH:MM):" line becomes a time field, a
  "______ °C" line a number field, and a "☐ NA" box on a heading row a checkbox.
  Sub-items such as 5.1–5.5 are grouped into a block with that checkbox.
- **Header fields** are found by the printed labels named in the config. **Sign-offs**
  come from "Sign … Date" rows.
- **Rule references** link each field to the rule ids in the rules CSV, so every finding
  cites its rule.
- **From the config:**
  - which answers are flagged and when the details must be empty;
  - which rows the flag rules cover;
  - the cross-checks, added only when the form has both items. The NA-checkbox rules are
    then switched on.

After writing the template it:

1. compares its items with every other template in the same folder (added / removed);
2. audits the blank PDF against it (every page recognised, no box reading as filled);
3. lists rule ids missing from, or unused in, the rules CSV.

Check the boxes visually:

```bash
python -m form_auditor preview templates/form.yaml -o preview.png
```

## Comparing with "just ask an AI"

`ai_audit.py` gives an AI model the same job as the auditor: the page images, the same
checklist (every item of your template, by the same item ids and check names) and the same
rule texts (your rules CSV). It asks for a verdict on every check (what it sees, PASS /
FAIL / REVIEW, why, and a box on the page) and writes it in the auditor's format:

- `ai_findings.csv`, laid out like `all_findings.csv`;
- `<file>_ai_audited.pdf`, the pages with the AI's numbered boxes (red FAIL, orange REVIEW);
- the AI's raw answer.

Pages are grouped into forms the same way as the auditor (the auditor's page matcher picks
which part of the checklist to send; no AI call). `--no-identify` takes them in order.

```bash
# Azure OpenAI (AZURE_OPENAI_ENDPOINT / _KEY / _DEPLOYMENT in .env)
python tools/ai_audit.py input/form1.pdf input/form2.pdf -o ai_out

# A chat window (Copilot, Claude...): writes the prompt and the page images per file
python tools/ai_audit.py input/form1.pdf input/form2.pdf -o ai_out --provider manual
#   attach <file>_image*.jpg in order to a new chat, paste <file>_prompt.txt,
#   save the JSON reply as ai_out/<file>/<file>_ai_answer.json, then:
python tools/ai_audit.py input/form1.pdf input/form2.pdf -o ai_out --import-answers

# Claude through the Anthropic API (ANTHROPIC_API_KEY in .env)
python tools/ai_audit.py input/form1.pdf -o ai_out --provider anthropic
```

For a chat, `--provider manual --attach-pdf` writes just the prompt, to paste in with the
PDF itself attached (no page images). Fine for Claude; if Copilot seems to miss the
handwriting, leave `--attach-pdf` off and attach the page images instead.

`--halves` sends each page as two overlapping halves (more detail per word, twice the images).

Then line the two up:

```bash
python -m form_auditor audit input/form1.pdf input/form2.pdf -o audit_out --ocr azure
python tools/compare_results.py audit_out ai_out -o comparison.csv
```

`comparison.csv` has one row per check with both verdicts side by side, disagreements
first, and two empty columns (`correct`, `notes`) to fill in by looking at the form.
`comparison_summary.csv` counts, per file, the agreements and what each one flagged alone.
SKIP counts as PASS. Both outputs hold values read from your forms and are git-ignored.
