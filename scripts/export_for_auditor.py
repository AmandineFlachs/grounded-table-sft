"""Flatten our reasoning-trace JSONL into a per-example CSV for dataset-auditor.

`../dataset-auditor` audits FLAT tabular datasets (one row per record) for *health*
- missing data, out-of-range values, duplicate keys, near-duplicates, PII. Our data
is nested JSONL (a whole table + question + trace per example), so we project each
example onto one summary row the auditor's checks can read. This is an independent
data-health second opinion, complementary to `trace_validator` (which checks task
correctness, not dataset health).

    python scripts/export_for_auditor.py data/processed/realtable.v0_1_0.jsonl results/audit/realtable.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import base_table_id, content_sig, load_jsonl  # noqa: E402

COLUMNS = [
    "example_id", "base_table_id", "table_id", "domain", "question_type", "split",
    "trace_source", "difficulty", "question", "answer_label",
    "n_table_rows", "n_table_cols", "n_trace_steps", "n_cites", "answer_n_rows",
    "table_content_sig",
]


def row_for(r: dict) -> dict:
    md = r.get("metadata") or {}
    t = r["table"]
    ga = r.get("gold_answer") or {}
    return {
        "example_id": md.get("example_id", r.get("table_id", "")),
        "base_table_id": base_table_id(r.get("table_id", "")),
        "table_id": r.get("table_id", ""),
        "domain": r.get("domain", ""),
        "question_type": r.get("question_type", ""),
        "split": r.get("split", ""),
        "trace_source": r.get("trace_source", ""),
        "difficulty": md.get("difficulty", ""),
        "question": r.get("question", ""),
        "answer_label": ga.get("label", ""),
        "n_table_rows": len(t.get("rows", [])),
        "n_table_cols": len(t.get("headers", [])),
        "n_trace_steps": len(r.get("trace_steps", [])),
        "n_cites": len(r.get("evidence_cells", [])),
        "answer_n_rows": len(ga.get("rows", [])),
        "table_content_sig": content_sig(r),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/export_for_auditor.py <in.jsonl> <out.csv>")
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    recs = load_jsonl(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in recs:
            w.writerow(row_for(r))
    print(f"wrote {len(recs)} rows x {len(COLUMNS)} cols -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
