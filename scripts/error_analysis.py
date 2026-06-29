"""Error analysis (P3.6): regenerate the 4B grounded-but-answer-wrong dev cases with raw
text captured, so we can classify the failure MECHANISM (cell misread / wrong logic /
trace-right-but-label-wrong). Greedy decode is deterministic -> reproduces the eval verdict.

Result 2026-06-24: all 47 cases are ARITHMETIC-EXECUTION errors (false numeric comparisons +
dominance-logic), 0 cell-misreads, 0 procedure-ignorance -> the residual is not data/perception.

Run from project root:  python scripts/error_analysis.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.getcwd())

from src.llm_renderer import render, build_prompt  # noqa: E402
from src.local_infer import LocalGenerator         # noqa: E402
from src.schema import Example                     # noqa: E402
from src.splits import load_jsonl                   # noqa: E402
from src.trace_validator import validate            # noqa: E402
from src.table_utils import render_markdown         # noqa: E402
from pathlib import Path

DEV = "data/processed/eval_dev.v0_1_0.jsonl"
RESULTS = "results/p3_sft_4b_dev.json"
ADAPTER = "models/qwen3-4b-sft"
MODEL = "Qwen/Qwen3-4B"
OUT = "results/p3_4b_error_analysis.json"

# 1. ids the eval flagged grounded & parsed & NOT answer_correct
det = json.load(open(RESULTS, encoding="utf-8"))["details"]
target = {d["example_id"] for d in det
          if d.get("grounded") and d.get("parsed") and not d.get("answer_correct")}
print(f"target grounded-but-wrong ids: {len(target)}", flush=True)

# 2. dev sources filtered to those ids
recs = load_jsonl(Path(DEV))
sources = [Example.model_validate(r) for r in recs]
def _id(ex): return ex.metadata.get("example_id", ex.table_id)
def _qt(ex): return ex.question_type.value if hasattr(ex.question_type, "value") else str(ex.question_type)
sel = [ex for ex in sources if _id(ex) in target]
print(f"matched sources: {len(sel)}", flush=True)

# 3. load 4B + adapter
gen = LocalGenerator(model_id=MODEL, adapter=ADAPTER, max_new_tokens=1536)

rows = []
reproduced = 0
for i, ex in enumerate(sel):
    out = render(ex, generate_fn=gen.generate)
    rec = {"id": _id(ex), "type": _qt(ex), "question": ex.question,
           "table_markdown": render_markdown(ex.table),
           "gold": {"label": ex.gold_answer.label, "rows": ex.gold_answer.rows,
                    "metrics": list(ex.gold_answer.metrics)},
           "parsed": out.ok, "raw": out.raw}
    if out.ok:
        try:
            res = validate(out.example)
            fa = out.example.gold_answer  # reconstructed = MODEL's answer
            rec["model_answer"] = {"label": fa.label, "rows": fa.rows, "metrics": list(fa.metrics)}
            rec["model_trace"] = [{"kind": s.kind, "description": s.description,
                                   "cites": [{"row": c.row, "col_name": c.col_name, "value": c.value} for c in s.cites]}
                                  for s in out.example.trace_steps]
            rec["failed"] = res.failed_checks()
            rec["answer_correct"] = bool(res.checks.get("answer_correct", False))
            rec["grounded"] = bool(res.checks.get("cells_exist", False))
            if rec["grounded"] and not rec["answer_correct"]:
                reproduced += 1
        except Exception as e:  # noqa: BLE001
            rec["failed"] = [f"validator_error: {type(e).__name__}: {e}"]
    rows.append(rec)
    print(f"  {i+1}/{len(sel)}  {_id(ex)}", flush=True)

Path(OUT).write_text(json.dumps({"n": len(rows), "reproduced_grounded_wrong": reproduced,
                                 "cases": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nreproduced grounded-but-wrong: {reproduced}/{len(sel)}", flush=True)
print(f"wrote {OUT}", flush=True)
