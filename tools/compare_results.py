"""Compare the form auditor's verdicts with the AI's, check by check.

    python tools/compare_results.py audit_out ai_out -o comparison.csv

Reads audit_out/all_findings.csv (the auditor) and ai_out/ai_findings.csv
(tools/ai_audit.py) and writes one row per check with both verdicts side by side,
disagreements first. Two empty columns are left for you: `correct` (auditor / ai / both /
neither) and `notes`, filled in by looking at the form. A summary per file is printed and
written next to it (comparison_summary.csv).

SKIP counts as agreeing with PASS (the check did not apply / nothing to flag).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

COLUMNS = ["file", "form_no", "section", "item_id", "check", "item_label", "agree",
           "auditor_status", "ai_status", "auditor_observed", "ai_observed",
           "auditor_value", "ai_value", "auditor_reason", "ai_reason",
           "auditor_marker", "ai_marker", "correct", "notes"]


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def norm(status: str) -> str:
    s = (status or "").upper()
    return "PASS" if s in ("PASS", "SKIP") else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("auditor", help="the auditor's output folder (or its all_findings.csv)")
    ap.add_argument("ai", help="tools/ai_audit.py's output folder (or its ai_findings.csv)")
    ap.add_argument("-o", "--out", default="comparison.csv")
    a = ap.parse_args(argv)
    pa, pb = Path(a.auditor), Path(a.ai)
    aud = read(pa / "all_findings.csv" if pa.is_dir() else pa)
    ai = read(pb / "ai_findings.csv" if pb.is_dir() else pb)

    def key(r):
        return (r["file"], str(r.get("form_no") or "1"), r["item_id"], r["check"])
    A = {key(r): r for r in aud if r.get("item_id")}
    B = {key(r): r for r in ai if r.get("item_id")}
    files = sorted({k[0] for k in B})                 # compare the files the AI was given
    rows = []
    for k in sorted(set(A) | set(B)):
        if k[0] not in files:
            continue
        x, y = A.get(k, {}), B.get(k, {})
        sa, sb = x.get("status", "(not checked)"), y.get("status", "(not checked)")
        rows.append({"file": k[0], "form_no": k[1], "item_id": k[2], "check": k[3],
                     "section": x.get("section") or y.get("section", ""),
                     "item_label": x.get("item_label") or y.get("item_label", ""),
                     "agree": "yes" if norm(sa) == norm(sb) else "NO",
                     "auditor_status": sa, "ai_status": sb,
                     "auditor_observed": x.get("observed", ""), "ai_observed": y.get("observed", ""),
                     "auditor_value": x.get("value_normalized") or x.get("value_read", ""),
                     "ai_value": y.get("value_read", ""),
                     "auditor_reason": x.get("reason", ""), "ai_reason": y.get("reason", ""),
                     "auditor_marker": x.get("marker", ""), "ai_marker": y.get("marker", ""),
                     "correct": "", "notes": ""})
    rows.sort(key=lambda r: (r["agree"] == "yes", r["file"], r["form_no"]))
    out = Path(a.out)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for f in files:
        mine = [r for r in rows if r["file"] == f]
        summary.append({"file": f, "checks": len(mine),
                        "agree": sum(r["agree"] == "yes" for r in mine),
                        "disagree": sum(r["agree"] != "yes" for r in mine),
                        "auditor_fail": sum(r["auditor_status"] == "FAIL" for r in mine),
                        "auditor_review": sum(r["auditor_status"] == "REVIEW" for r in mine),
                        "ai_fail": sum(r["ai_status"] == "FAIL" for r in mine),
                        "ai_review": sum(r["ai_status"] == "REVIEW" for r in mine),
                        "ai_unanswered": sum(r["ai_status"] in ("MISSING", "(not checked)") for r in mine),
                        "flagged_by_auditor_only": sum(r["auditor_status"] in ("FAIL", "REVIEW")
                                                       and r["ai_status"] not in ("FAIL", "REVIEW") for r in mine),
                        "flagged_by_ai_only": sum(r["ai_status"] in ("FAIL", "REVIEW")
                                                  and r["auditor_status"] not in ("FAIL", "REVIEW") for r in mine)})
    with open(out.with_name(out.stem + "_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]) if summary else ["file"])
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['file']}: {s['checks']} checks, agree on {s['agree']}, disagree on {s['disagree']} "
              f"(flagged only by the auditor: {s['flagged_by_auditor_only']}, only by the AI: "
              f"{s['flagged_by_ai_only']}, AI gave no answer: {s['ai_unanswered']})")
    print(f"\nWrote {out} (disagreements first - fill in 'correct' by looking at the form)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
