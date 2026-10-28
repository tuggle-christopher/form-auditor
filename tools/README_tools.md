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
