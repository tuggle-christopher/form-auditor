# form_auditor

Audits scanned or photographed paper checklist forms against your rules: forms with
header fields, sections of Yes / No / NA questions with a details column, items with
sub-items and NA boxes, and sign-offs. It works on PDFs of any length and on images. For
each page it:

- checks every section;
- reads the handwriting;
- draws numbered boxes around every problem;
- writes CSV reports that say what was checked, where, why, and which rule applies.

It runs on a laptop or on Databricks classic compute, and reads and writes local folders or Azure Blob Storage.

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in the Azure Document Intelligence endpoint + key

# once: build the template from your BLANK form (see "Setting up a form")
python tools/build_template_from_pdf.py forms/blank.pdf

python -m form_auditor audit scans/ -o audit_out --ocr azure
```

Inputs can be files or folders: `.pdf`, `.jpg`, `.png`, `.tif` and multi-page TIFFs. A
page takes roughly 5–10 s, plus the handwriting reading.

## Running on Databricks

1. Put this project folder in the workspace, for example as a Git folder.
2. Next to `requirements.txt`, create `.env` from `.env.example`. It holds the Azure Document
   Intelligence endpoint and key and, for `https://` blob paths, a storage connection string or SAS token.
   `.env` is git-ignored, so it has to be created in the workspace itself. Anyone who can read
   the folder can read the key, so keep the folder's permissions tight.
3. Open `notebooks/audit_forms.py` on a classic cluster (single node is enough). Fill in
   the **Input** and **Output** widgets and run all cells.

Paths can be any of:

| path | access |
|---|---|
| `https://<account>.blob.core.windows.net/<container>/<folder>` | credentials from `.env` (`AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_SAS_TOKEN` or `AZURE_STORAGE_KEY`) |
| `abfss://<container>@<account>.dfs.core.windows.net/<folder>` (also `wasbs://`, `dbfs:/`) | the storage access already configured on the cluster (copied with `dbutils.fs`) |
| `/dbfs/mnt/...`, `/Volumes/...` or a local folder | used directly |

Input files are copied to the driver's local disk, audited, and the results are copied
back. Each notebook run writes to a new time-stamped subfolder of the output. Nothing is
written to tables.

From your own code (a notebook or a job), the same thing is:

```python
import sys; sys.path.insert(0, "/Workspace/Users/<you>/form_auditor")
from form_auditor.api import run_audit
result = run_audit("https://acct.blob.core.windows.net/forms/incoming/",
                   "https://acct.blob.core.windows.net/forms/audited/run1",
                   ocr="azure", env_file="/Workspace/Users/<you>/form_auditor/.env")
result["summary"]        # one dict per page; result["findings"]: one per check
```

The project uses the headless OpenCV build (`opencv-python-headless`). If `import cv2`
fails with `libGL.so.1`, the GUI build is also installed: run `%pip uninstall -y opencv-python`.

## What is checked

What the auditor checks comes from your form, through three files in `templates/`. All
three are kept out of git:

- **`form.yaml`**, the template, built from the blank PDF. It records:
  - every box, answer row, checkbox and sign-off on each page;
  - which answers are flagged;
  - when a details column must stay empty;
  - which items have sub-items (and an NA box);
  - which items are cross-checked against each other.
- **`form_config.yaml`**, your settings for the builder: the form id, the header fields,
  which answers are flagged, and the cross-checks.
- **`form_rules.csv`**, the wording of each rule, quoted in the reports.

`templates/FORM_NOTES.md` (also local) describes what is checked on your form.

The kinds of checks:

| check | what the auditor does |
|---|---|
| Header fields | Must be filled in. The value is read and returned. "NA" can be allowed or forbidden per field (needs the handwriting reader). |
| Answer rows | Exactly one **printed** option (Yes / No / NA) must be marked. NA written in, or circled in an empty cell, is flagged. |
| Flagged answers | e.g. any **No** → FAIL, optionally only on the first rows of a section. The details read are returned. |
| Details column | Writing next to an answer that must have empty details (e.g. Yes / NA) → FAIL. |
| Items with sub-items | e.g. 5.1–5.5 under item 5. With the NA-checkbox rules on, all parts are required unless the item's NA box is ticked. With them off, an item that is partly filled in must be complete. Time fields need HH:MM; number fields need a value. |
| Cross-checks | e.g. "the NA box of item X must not be ticked when answer Y is Yes". |
| Sign-offs | Signature and date must be present. Dates only need to be present; the value is returned (ISO when possible). |

`rule_summary.csv` lists, for every rule in the CSV, how often it was checked and failed,
and marks the rules nothing on this version of the form uses as "not checked".

### Verdicts

| status | meaning |
|---|---|
| **FAIL** | a rule is broken: this is the "flag" from the rules |
| **REVIEW** | the machine is not sure (two answers, a mark hidden under an annotation, a faint mark, unreadable handwriting...). A person must look. Nothing uncertain is ever passed silently. |
| **PASS** | the rule is satisfied |
| **SKIP** | the check depends on something unreadable (e.g. the details rule when the answer is unclear), or is recorded only |
| **UNRECOGNISED** (page) | the page is not a page of a known form version (e.g. a new version before its template exists, or other paperwork). It is not audited and not counted as a failure. |

### How the answers are read

- **Circle:** the printed word it is drawn around is the answer. If a circle overlaps Yes
  and No, the word it is centred on wins (with a note). A mark surrounding both words
  equally → REVIEW.
- **Tick or X instead of a circle:** that cell's option is chosen, with a note.
- **Circled and then crossed out (cancel):** the option is withdrawn.
  - If another option is circled, that one is the answer. For example, Yes circled and
    crossed out plus No circled gives **No**, with the note "Yes was cancelled".
  - If the crossed-out option was the only mark, the row has no valid answer → REVIEW.
  - Change this with `--cancel-means selected` (still counts) or `--cancel-means review`.
- **Two answers:** REVIEW.
- **A circle cut where it crosses printed text or a row line:** the pieces are re-joined.
- **NA checkboxes:** any pen mark in or on the box counts as ticked, a small faint mark →
  REVIEW.

Details handwriting:

- Lines of writing go to the row that holds most of their ink, so writing that
  overlaps into the next row counts for the row it was written for.
- Writing that runs past the right edge is included.
- Stray marks beside the form are ignored.

## How a box is judged empty or filled

Two independent methods look at every box, and neither can call a box blank on its own
when the other sees something.

1. **The ink check** (always on, no cloud). The page is aligned to the blank form, the
   printed form is subtracted, and the pen ink left in each box is measured. It runs twice:
   - **Normal pass:** clear pen strokes, including signatures, scribbles, a small "NA",
     and heavy or thick writing.
   - **Faint pass:** much more sensitive, for light pen or pencil. What only the faint
     pass sees is never decided by it alone: it turns a would-be FAIL ("blank", "no
     answer") into a REVIEW with what it saw, e.g. "no clear mark, only a faint circle on
     Yes". On all the sample photos, the blank form and the test scans it raised no false
     alarms.
2. **The handwriting reader** (`--ocr azure`, Azure Document Intelligence Layout). The
   straightened pages are sent to Azure, and every word comes back with its exact position
   on the form. Printed words of the form are recognised and ignored.

The two are combined like this:

| ink check | reader | result |
|---|---|---|
| ink in the box | anything | **filled** (a signature needs no reading) |
| nothing | reads writing with confidence ≥ 0.8 | **filled**, noted as light writing; the value is read |
| nothing / faint only | reads something, unsure | **REVIEW**, with what it read |
| faint marks only | nothing | **REVIEW** |
| nothing | nothing | **blank** |

Writing that doesn't stay in its box:

- **Runs out of a box** (past the edge, over a line, into the next cell): it stays with
  the box it starts in, and the whole value is read. A line that sags over a row line
  belongs to the row holding most of it, so the row below doesn't look filled.
- **Written next to an empty box instead of in it:** REVIEW, "box is empty but there is
  writing right next to it", with both boxed on the page. Leftovers of printed lines and
  the edge of the photographed page are not counted as writing.

## Reading handwriting (`--ocr`)

| engine | use |
|---|---|
| `azure` | Azure AI Document Intelligence **Layout** on whole pages, combined with the ink check as above. Set `AZURE_DI_ENDPOINT` and `AZURE_DI_KEY` in `.env`; both are on the resource's **Keys and Endpoint** page. |
| `azure-read` | Azure **Read** on crops of the boxes the ink check found filled. Cheaper, but it can't help with faint writing or writing beside a box. |
| `none` (default) | No reading. The ink check (both passes) and the writing-beside-a-box check still run. Checks on a value (e.g. "NA" not allowed in a box) need a reader. |
| `claude` | Claude vision through the Anthropic API (`ANTHROPIC_API_KEY`), if that is ever allowed. |
| `tesseract` | Offline, but **not suitable for handwriting**: it read 0 of 8 sample values. |
| `mock`, `mock-layout` | Canned answers from JSON, for tests. |

**One request per PDF, within the free tier (F0).** The free tier analyses only the first
2 pages (images) of a request, up to 4 MB, with 1 request and 1 result check per second.
To read a whole form in one request anyway:

- The straightened pages of a PDF are **tiled onto 2 images**. A 3-page form goes as pages
  1 + 2 side by side, and page 3 on the second image.
- Both images go in **one** greyscale multi-page TIFF (about 2–3 MB at 300 dpi).
- Every word that comes back is mapped to the form page and spot it lies on.
- Up to **6 form pages** (3 per image) go in one request at full resolution. Only longer
  PDFs need more requests. Images stay within the service's 10,000 px limit, and the
  request is compressed further if it would exceed 4 MB.
- If the service doesn't accept the TIFF, the same images are sent as a PDF instead.
- Azure counts each image as one page, so a 3-page form uses 2 pages of the free tier's
  monthly allowance (see Azure's pricing page). When the allowance runs out, Azure refuses
  requests and the run stops with a message saying so.
- Requests and result checks are paced to 1 per second, and "too many requests" answers
  are retried.
- If the service returns fewer images than were sent, or a request fails, those pages are
  still audited with the ink check. Their filled-in values go to REVIEW as "could not be
  transcribed", and the log says why.
- On the paid tier (S0) the same settings work. `--azure-pages-per-request` (images per
  request) and `azure_pages_per_sheet` (form pages per image, default 3; 1 = no tiling)
  can be raised or lowered.

A reading below `ocr_min_conf` (0.6) goes to REVIEW with the value read. "NA" read with
high confidence in a box where it isn't allowed is FAIL; with lower confidence it goes to
REVIEW. A bad key or endpoint stops the run immediately rather than producing hundreds of
REVIEWs.

The request format, batching, pacing and coordinate handling were checked against a local
stand-in that behaves like the free tier, but not against a real Azure resource yet. Run
a few known forms first and compare.

## Outputs

```
<out>/
  all_findings.csv        every check on every page (one row per check)
  extracted_data.csv      one row per form: every value read, every answer, signed / blank
  rule_summary.csv        per rule: checks, fails, reviews, and rules not checked on this form version
  batch_summary.csv       one row per page: status, counts, alignment quality
  <file>/<file>_audited.pdf       the pages with numbered red (FAIL) / orange (REVIEW) boxes
  <file>/<file>_pN_audited.png|jpg
  <file>/<file>_audit.csv | _extracted.csv | _summary.csv
```

The columns of `all_findings.csv`:

| group | columns |
|---|---|
| Where | `file, page, marker` (the number drawn on the page), `section, item_id, item_label` |
| What was checked | `check, status, observed, expected, reason` |
| What was read | `value_read` (as written), `value_normalized` (ISO date, `NA`, ...), `mark_type`, `notes` |
| Which rule | `rule_ref`, `rule_instruction` (from the rules CSV) |
| Evidence | `confidence, ocr_confidence, ink_ratio, visible_ratio, marks` |
| Location | `bbox_*` (pixels on the original page), `align_method` |

Pages are grouped into forms (page 1 + 2 + 3) in `extracted_data.csv` and in the
`form_no` column.

## Options

| flag | |
|---|---|
| `-t TEMPLATE [...]` | template YAML(s). Default: every template in `templates/`. With several (old and new version), each page is matched to the one it fits best. |
| `--ocr azure/azure-read/none/...` | handwriting reader |
| `--azure-pages-per-request N` | images per Azure request: 2 (free tier, default). Each image holds up to `azure_pages_per_sheet` (3) form pages, so a PDF of up to 6 pages is one request |
| `--env-file PATH` | `.env` to load (default: `./.env`, then the project's `.env`) |
| `--na-checkbox-rules` | turn on the NA-checkbox rules (sub-items required unless NA is ticked; cross-checks). A template whose form has the cross-checked items turns them on itself. |
| `--cancel-means withdrawn/selected/review` | how a circled-then-crossed-out option counts |
| `--skip-unrecognised` | leave non-form pages out of the annotated PDF |
| `--show-pass` | also outline passing fields in green |
| `--debug` | write the aligned page and ink maps, for tuning |
| `--set KEY=VALUE` | override any setting in `form_auditor/settings.py` (e.g. `--set date_order=MDY`) |
| `--strict` | exit code 1 when any page fails |

## Setting up a form

1. Put the **blank** form PDF in `forms/` (git-ignored), e.g. `forms/blank.pdf`.
2. Copy `tools/form_config.example.yaml` to `templates/form_config.yaml` and fill it in:
   the form id, the header fields (the printed label of each field), which answers are
   flagged, and any cross-checks.
3. Put your rule texts in `templates/form_rules.csv`, with the columns `label`,
   `category` and `instruction`. The rule ids follow the builder's convention: for
   section X, `X-1` is "answer required", `X-2` "flagged answer", `X-3` "details must be
   blank", `X-4` "sign-off", and `X-<n>` is item n with sub-items. `G-1` covers "only
   printed options" and `G-2` "how answers are read"; header fields cite the ids given in
   the config.
4. Build:

   ```bash
   python tools/build_template_from_pdf.py forms/blank.pdf
   ```

   This writes `templates/form.yaml` and one `form_pN.png` per page. It then checks the
   result:
   - **Self-check:** the blank PDF is audited against the new template. Every page must
     be recognised, and no box may read as filled on a blank form; a box that does
     probably covers printed text.
   - **Rule ids:** missing or unused ones are listed.
   - **Comparison:** items added or removed compared with any other template in
     `templates/` are listed.
5. Look at the boxes: `python -m form_auditor preview templates/form.yaml -o preview.png`.

## Updating to a new version of the form

When a new blank PDF arrives:

1. Save it as e.g. `forms/blank_v2.pdf`.
2. Update `templates/form_config.yaml` only if something there changed: a header label
   was reworded, a flagged-rows limit or cross-check is new or different. Cross-checks
   whose items don't exist on a form are skipped, so a config can describe both versions.
3. Build it **next to** the old template, so forms of both versions can be audited while
   old ones are still coming in:

   ```bash
   python tools/build_template_from_pdf.py forms/blank_v2.pdf --out templates/form_v2.yaml --version 2
   ```

4. Read the builder's report:
   - the self-check must say every page is recognised, with 0 problems;
   - check that the items added and removed compared with `form.yaml` are what changed
     on the paper (e.g. new rows in a section);
   - rule ids in the template but missing from `form_rules.csv` need adding to the CSV.
5. Check the boxes with `preview`. Then audit a few filled forms of the new version that
   you've checked by hand, and compare.
6. Update `form_rules.csv` and `templates/FORM_NOTES.md` if the rules' wording changed.

Every template in `templates/` is loaded by default, and each page is matched to the
version it fits best (a file's pages are kept on one version). Once old forms stop
arriving, delete the old template's files (`form.yaml`, `form_p*.png`).

On Databricks, upload the new template files to the workspace's `templates/` folder.
They aren't in git.

## Keeping sensitive files out of git

`.gitignore` keeps these out of the repository:

- blank and filled PDFs, scans and photos;
- the `templates/` folder (form images, wording, config, rules, notes) and `local_tests/`;
- audit outputs and the CSVs they contain;
- `.env` (keys).

Before a first push, check what would go up:

```bash
git add -A
git status            # only code, docs, templates/README.md and the example config
```

If something sensitive was committed before, `git rm --cached <file>` stops tracking it.
It is still in the history, though, and a file already pushed to GitHub needs the history
rewritten (e.g. with `git filter-repo`).

## How it works

1. **Page matching.** Each page is matched to the blank template using SIFT features, a
   homography and ECC refinement. The layout is then checked by counting table rules, so
   another version of the form is rejected rather than misread.
2. **Lens correction.** The photographed table lines are snapped onto the template's;
   phone photos come out within about 1 px.
3. **Ink isolation.** Pen ink is anything clearly darker than the local paper, with the
   printed form removed. Blue annotation callouts count as "hidden" and go to REVIEW.
4. **Reading.** Strokes become marks (circle, tick, cross, cancel) and text lines, the
   rules are applied, and handwriting is cropped from the full-resolution page and read.

Results are deterministic: the same file always gives the same report.

## Tests

```bash
python examples/synthetic_demo.py      # a made-up form, end to end, with known answers
python examples/unit_checks.py         # .env, blob / abfss staging, the Azure Layout client (free-tier
                                       # stand-in: tiling, pacing, retries), template validation
                                       # (needs a template in templates/)
```

Tests that use your own template live in `local_tests/`, which is git-ignored because it
names your form's items:

- `edge_cases.py` fills complete forms with every situation in the rules (228 checks).
- `reader_fusion_test.py` covers messy writing: faint writing, writing out of or beside
  boxes, a faint circle, a heavy signature, a sagging line.
- `big_circles_test.py` covers big circles: ones that cross row lines into the rows
  above and below, ones cut by a line, a circle whose edge passes between Yes and No, and
  a tight circle around NA.

Both are written for the current form, so after a new version, update the item ids they
use.

## Known limitations

- **Mark-type names** (`mark_type`) are best effort. A tick drawn over a word is sometimes
  called "partial-circle". The chosen answer is what decides pass/fail, and it was correct in every test.
- **Cancellations** are found from straight strokes across a circled word. They were
  tested on synthetic forms only, since none of the sample photos has one. A scribbled-out
  (blacked-out) option is not specifically recognised.
- **Ink exactly on a printed line or letter** can't be told apart from the print. Broken
  circles are re-joined, and extreme cases go to REVIEW.
- **Content is not verified**: the tool doesn't know whether a reference number is the right one.
- **Storage and Azure:** the blob, `abfss://` and Document Intelligence code was tested
  against stand-ins for Azure and `dbutils`, not real resources. In particular, compare
  the reading of tiled pages against `azure_pages_per_sheet=1` on a few forms. Tiling
  keeps full resolution, so the reading should be the same, but it is unverified on the
  real service.
- **Extremely light writing** (barely visible on the scan) is beyond the ink check. Only
  the reader can find it, so use `--ocr azure`.
- **`calibrate`** (drawing boxes by hand) needs a desktop GUI and doesn't run on Databricks.
  It is not normally needed.
