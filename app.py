"""Form auditor web app: upload filled forms, get the marked-up PDFs and the reports back.

Runs as a Databricks App (see app.yaml and README, "Running it as a web app"), or on a
laptop:

    streamlit run app.py

The look comes from .streamlit/config.toml (colours, fonts) and branding/branding.yaml
(title, logo, company name, footer); see the .example files next to them.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from form_auditor.webapp import audit_uploads, load_branding, load_guide, reader_configured

ROOT = Path(__file__).resolve().parent
B = load_branding(ROOT)
st.set_page_config(page_title=B["app_title"], page_icon=B["logo_small"] or B["logo"] or None,
                   layout="wide")
if B["logo"] and hasattr(st, "logo"):
    st.logo(B["logo"], icon_image=B["logo_small"] or None)
if B["css"]:
    st.markdown(f"<style>{Path(B['css']).read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

STATUS_LABEL = {"FAIL": "FAIL", "ERROR": "ERROR", "REVIEW": "Needs review",
                "UNRECOGNISED": "Not recognised", "NOT RECOGNISED": "Not recognised", "PASS": "PASS"}

GUIDE = load_guide(ROOT)


def how_to_use_body():
    st.markdown(GUIDE)
    if st.button("Got it", type="primary"):
        st.rerun()


# "How to use": a pop-up (opened once when someone opens the app, and from the sidebar)
if hasattr(st, "dialog"):
    how_to_use = st.dialog("How to use", width="large")(how_to_use_body)
else:                                                  # older Streamlit: no pop-ups
    def how_to_use():
        with st.expander("How to use", expanded=True):
            st.markdown(GUIDE)

if B["guide_on_open"] and not st.session_state.get("guide_shown"):
    st.session_state["guide_shown"] = True
    how_to_use()

st.title(B["app_title"])
if B["subtitle"]:
    st.write(B["subtitle"])
st.caption("**1** Add your scanned forms  →  **2** Check forms  →  **3** See what was found and download")

with st.sidebar:
    if B["company_name"]:
        st.subheader(B["company_name"])
    if st.button("How to use", icon=":material/help:", use_container_width=True):
        how_to_use()
    st.header("Options")
    show_pass = st.checkbox("Also outline checks that passed", value=False,
                            help="Draws a light box around everything that passed too, not "
                                 "only the problems, on the marked-up PDFs.")
    st.divider()
    if B["footer"]:
        st.caption(B["footer"])
    if B["support_contact"]:
        st.caption(B["support_contact"])

# The handwriting reader is always used: it reads the Section A values for the report, the
# printed form version (Document ID line) and gives a second opinion on faint writing. Without
# it the app does not check forms at all.
has_reader = reader_configured()
if not has_reader:
    st.error("The handwriting reader (Azure Document Intelligence) is not set up for this app, "
             "so forms can't be checked. Add the endpoint and key as the app's resources "
             "(adi-endpoint and adi-key - see README, \"Running it as a web app\") and redeploy.")

uploads = st.file_uploader("Scanned forms (PDF) - several at once is fine", type=["pdf"],
                           accept_multiple_files=True)

if st.button("Check forms", type="primary", disabled=not (uploads and has_reader)):
    n = len(uploads)
    with st.status(f"Checking {n} file{'s' if n != 1 else ''} - about 10-15 seconds per "
                   f"3-page form...", expanded=False) as box:
        try:
            st.session_state["result"] = audit_uploads(
                [(u.name, u.getvalue()) for u in uploads],
                use_reader=True, show_pass=show_pass)
            box.update(label="Done", state="complete")
        except Exception as e:                           # shown, not hidden
            st.session_state.pop("result", None)
            box.update(label="The check failed", state="error")
            st.error(f"Something went wrong: {e}")

res = st.session_state.get("result")
if res is not None:
    if res.stopped:
        st.error(f"Stopped early: {res.stopped}")
    if res.archive_error:
        st.warning(f"The results are below, but the record of this run could not be saved to "
                   f"the archive: {res.archive_error}")

    # forms whose version is not one this tool knows: said first, plainly
    for d in res.documents:
        vs = str(d.get("version_status", ""))
        if vs.startswith(("NOT RECOGNISED", "FORM NOT RECOGNISED")):
            label = d["file"] + (f" (form {d['form_no']})" if str(d.get("form_no", "")) not in ("", "1") else "")
            read = f" - version read: {d['version_read']}" if d.get("version_read") else ""
            why = ("Its pages don't match any form version this tool knows."
                   if vs.startswith("FORM NOT RECOGNISED") else vs.split(" - ", 1)[-1])
            st.error(f"**{label}: this form version is not recognised**{read}. {why}")
        elif vs.startswith("NOT READ"):
            st.warning(f"**{d['file']}**: the form version could not be read - check it by hand.")

    t = res.totals
    c = st.columns(5)
    c[0].metric("Files", t["files"])
    c[1].metric("Fail", t["fail"])
    c[2].metric("Need review", t["review"])
    c[3].metric("Not recognised", t["unrecognised"])
    c[4].metric("Pass", t["pass"])

    st.subheader("Report")
    if res.documents:
        cols = res.document_columns or list(res.documents[0])
        st.dataframe([{c: d.get(c, "") for c in cols} for d in res.documents],
                     hide_index=True, use_container_width=True)
    d1, d2, d3 = st.columns(3)
    if res.document_report.is_file():
        d1.download_button("Download the report (.csv)", data=res.document_report.read_bytes(),
                           file_name="document_report.csv", mime="text/csv", type="primary",
                           help="One row per form: its result, Section A details and how many "
                                "problems in each section. Opens in Excel.")
    d2.download_button("Download everything (PDFs + reports, .zip)",
                       data=res.zip_path.read_bytes(), file_name="form_audit_results.zip",
                       mime="application/zip",
                       help="The marked-up PDFs and all the reports, in one zip file.")
    if res.all_findings.is_file():
        d3.download_button("Download every check (all_findings.csv)",
                           data=res.all_findings.read_bytes(), file_name="all_findings.csv",
                           mime="text/csv",
                           help="Every single check on every page, passed or not, with where "
                                "and why. For looking into a result in detail.")

    st.subheader("Files")
    st.caption("Open a file to see its problems. In its marked-up PDF each problem is boxed and "
               "numbered; the number matches the # in the list.")
    for i, f in enumerate(res.files):
        with st.expander(f"{f.name} - {STATUS_LABEL.get(f.status, f.status)}",
                         expanded=f.status in ("FAIL", "ERROR", "REVIEW") and len(res.files) <= 3):
            cols = st.columns(2)
            if f.pdf is not None:
                cols[0].download_button("Marked-up PDF", data=f.pdf.read_bytes(),
                                        file_name=f.pdf.name, mime="application/pdf", key=f"pdf{i}")
            if f.csv is not None:
                cols[1].download_button("Checks for this file (.csv)", data=f.csv.read_bytes(),
                                        file_name=f.csv.name, mime="text/csv", key=f"csv{i}")
            if f.note:
                st.caption(f.note)
            if f.problems:
                st.dataframe([{"#": r.get("marker", ""), "Page": r.get("page", ""),
                               "Item": r.get("item_id", ""), "Check": r.get("check", ""),
                               "Result": r.get("status", ""), "Seen": r.get("observed", ""),
                               "Why": r.get("reason", "")} for r in f.problems],
                             hide_index=True, use_container_width=True)
            for p in f.previews:
                st.image(str(p), use_container_width=True)
