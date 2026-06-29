"""Groundedness diagnostic (P3.7): WHY do engine-correct answers still get an invalid TRACE?

The executor (P3.6) fixed the ANSWER (dev 54.7%->95.8%), but full trace validity is only ~70%.
From results/p3_4b_exec_dev.json (no GPU): of 182 engine-correct answers, 50 have an invalid
trace, and 47 of those fail purely on GROUNDEDNESS (a bad cell citation). The saved results do
NOT keep the raw trace or the offending cite, so this script re-runs (deterministic greedy ->
reproduces the eval verdict) just those 47 cases, captures the raw output + validator
failed_checks, and labels EACH failing citation:

  - row_oob          : cited (row,col) outside the table
  - colname_mismatch : col_name doesn't match the header at that col (rare; col is backfilled)
  - rounding         : numeric value within ~rel-tol of the true cell but > validator's 1e-6
  - type_format      : type/format mismatch (e.g. "92%"/"0.92"/"$1,234" vs the numeric cell)
  - wrong_row        : the cited value is correct for the SAME column at a DIFFERENT row
  - not_in_column    : value appears nowhere in that column (hallucinated / derived)

Mirrors the validator's cells_exist rule (src/trace_validator.py:119) exactly so labels and
verdicts agree. Modeled on scripts/error_analysis.py.

Run from project root (needs the GPU / WSL .venv-train):
    python scripts/diag_groundedness.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src.llm_renderer import render               # noqa: E402
from src.local_infer import LocalGenerator        # noqa: E402
from src.schema import Example                    # noqa: E402
from src.splits import load_jsonl                  # noqa: E402
from src.trace_validator import validate           # noqa: E402
from src.table_utils import render_markdown        # noqa: E402

DEV = "data/processed/eval_dev.v0_1_0.jsonl"
EXEC_RESULTS = "results/p3_4b_exec_dev.json"
ADAPTER = "models/qwen3-4b-sft-exec"
MODEL = "Qwen/Qwen3-4B"
OUT = "results/p3_4b_groundedness_diag.json"

TOL = 1e-6  # validator's float tolerance (src/trace_validator.py:_value_eq)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def classify_cite(table, c) -> str | None:
    """Return a failure label for one citation, or None if it grounds correctly.
    Mirrors validator cells_exist, then sub-classifies the value mismatch."""
    n_rows, n_cols = len(table.rows), len(table.headers)
    if not (0 <= c.row < n_rows and 0 <= c.col < n_cols):
        return "row_oob"
    if table.headers[c.col] != c.col_name:
        return "colname_mismatch"
    actual = table.cell(c.row, c.col)
    val = c.value
    # numeric match within validator tolerance -> grounded
    if _is_num(actual) and _is_num(val) and abs(actual - val) <= TOL:
        return None
    if actual == val:
        return None
    # --- it's a genuine mismatch: sub-classify ---
    # wrong_row: same column, value correct for a DIFFERENT row
    for r in range(n_rows):
        if r == c.row:
            continue
        other = table.cell(r, c.col)
        if (_is_num(other) and _is_num(val) and abs(other - val) <= TOL) or other == val:
            return "wrong_row"
    # rounding: both numeric, close in relative terms but beyond 1e-6
    if _is_num(actual) and _is_num(val):
        denom = max(1.0, abs(actual))
        if abs(actual - val) / denom <= 5e-3:
            return "rounding"
        return "not_in_column"
    # type/format: one side numeric, other a string (percent/currency/thousands formatting)
    if _is_num(actual) != _is_num(val):
        return "type_format"
    return "not_in_column"


def main() -> int:
    det = json.load(open(EXEC_RESULTS, encoding="utf-8"))["details"]
    target = {d["example_id"] for d in det if d.get("engine_ok") and not d.get("grounded")}
    print(f"target engine-correct-but-ungrounded ids: {len(target)}", flush=True)

    recs = load_jsonl(Path(DEV))
    sources = [Example.model_validate(r) for r in recs]

    def _id(ex):
        return ex.metadata.get("example_id", ex.table_id)

    def _qt(ex):
        return ex.question_type.value if hasattr(ex.question_type, "value") else str(ex.question_type)

    sel = [ex for ex in sources if _id(ex) in target]
    print(f"matched sources: {len(sel)}", flush=True)

    gen = LocalGenerator(model_id=MODEL, adapter=ADAPTER, max_new_tokens=1536)

    cite_hist: dict[str, int] = {}
    case_hist: dict[str, int] = {}   # cases whose failure-set CONTAINS this label
    reproduced_ungrounded = 0
    rows = []
    for i, ex in enumerate(sel):
        out = render(ex, generate_fn=gen.generate)
        rec = {"id": _id(ex), "type": _qt(ex), "question": ex.question,
               "table_markdown": render_markdown(ex.table), "parsed": out.ok, "raw": out.raw}
        if out.ok:
            try:
                res = validate(out.example)
                grounded = bool(res.checks.get("cells_exist", False))
                rec["failed"] = res.failed_checks()
                rec["grounded"] = grounded
                bad = []
                for s in out.example.trace_steps:
                    for c in s.cites:
                        label = classify_cite(ex.table, c)
                        if label is not None:
                            bad.append({"row": c.row, "col_name": c.col_name, "value": c.value,
                                        "actual": ex.table.cell(c.row, c.col) if 0 <= c.row < len(ex.table.rows)
                                        and 0 <= c.col < len(ex.table.headers) else None,
                                        "label": label})
                            cite_hist[label] = cite_hist.get(label, 0) + 1
                rec["bad_cites"] = bad
                for lab in {b["label"] for b in bad}:
                    case_hist[lab] = case_hist.get(lab, 0) + 1
                if not grounded:
                    reproduced_ungrounded += 1
            except Exception as e:  # noqa: BLE001
                rec["failed"] = [f"validator_error: {type(e).__name__}: {e}"]
        rows.append(rec)
        print(f"  {i+1}/{len(sel)}  {_id(ex)}  bad_cites={len(rec.get('bad_cites', []))}", flush=True)

    summary = {"n_target": len(sel), "reproduced_ungrounded": reproduced_ungrounded,
               "cite_label_hist": dict(sorted(cite_hist.items(), key=lambda kv: -kv[1])),
               "case_label_hist": dict(sorted(case_hist.items(), key=lambda kv: -kv[1]))}
    Path(OUT).write_text(json.dumps({**summary, "cases": rows}, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nreproduced ungrounded: {reproduced_ungrounded}/{len(sel)}", flush=True)
    print(f"cite-level labels : {summary['cite_label_hist']}", flush=True)
    print(f"case-level labels : {summary['case_label_hist']}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
