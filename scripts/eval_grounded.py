"""Step A - grounded traces by CONSTRUCTION, measured on the locked-test outputs (no GPU).

The executor solved the ANSWER by not trusting the model's arithmetic. The trace citations have
the same disease (the model writes wrong numbers), so we apply the same fix: replace the model's
hand-typed citations with the cells the ENGINE actually read (src/executor.evidence_for) - grounded
by construction. This re-scores the SAVED locked-test outputs (results/p3_4b_exec_TEST.json), so it
needs no model re-run.

Reports BOTH numbers honestly:
  * model's UNAIDED citation faithfulness (the trace cites it typed itself) - unchanged ~71%,
  * the SYSTEM's grounded-by-construction rate (engine-read evidence) - the new number.
Never conflates them: the trace prose stays model-authored; the cited evidence becomes
system-constructed.

    python scripts/eval_grounded.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.executor import compute_answer, evidence_for     # noqa: E402
from src.llm_renderer import extract_operation            # noqa: E402
from src.schema import Example, TraceStep                 # noqa: E402
from src.splits import load_jsonl                          # noqa: E402
from src.table_utils import entity_universe               # noqa: E402
from src.trace_validator import validate                   # noqa: E402

TEST = "data/processed/eval_test.v0_1_0.jsonl"
RESULTS = "results/p3_4b_exec_TEST.json"
OUT = "results/p3_4b_grounded_constructed.json"


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100*p:.1f}% [{100*max(0,c-h):.1f}-{100*min(1,c+h):.1f}]"


def main() -> int:
    det = json.loads(Path(RESULTS).read_text(encoding="utf-8"))["details"]
    src = {}
    for r in load_jsonl(Path(TEST)):
        ex = Example.model_validate(r)
        src[ex.metadata.get("example_id", ex.table_id)] = ex

    n = len(det)
    agg = Counter()
    by_type = defaultdict(lambda: Counter())
    type_n = Counter()
    rows = []
    for d in det:
        eid, qt = d["example_id"], d["type"]
        type_n[qt] += 1
        ex = src.get(eid)
        model_grounded = bool(d.get("grounded"))
        agg["model_grounded"] += int(model_grounded)
        by_type[qt]["model_grounded"] += int(model_grounded)

        grounded_c = valid_c = False
        if ex is not None and d.get("raw"):
            op = extract_operation(d["raw"])
            if isinstance(op, dict):
                try:
                    op2 = {**op, "rows": entity_universe(ex.table)}
                    ev = evidence_for(ex.table, op2)
                    eng = compute_answer(ex.table, op2)
                    conclude = TraceStep(index=0, kind="conclude",
                                         description="Answer computed from the cited cells.", cites=ev)
                    sysex = ex.model_copy(update={
                        "gold_answer": eng, "trace_steps": [conclude],
                        "evidence_cells": ev, "trace_source": "programmatic",
                    })
                    vr = validate(sysex)
                    grounded_c = bool(ev) and bool(vr.checks.get("cells_exist"))
                    valid_c = vr.valid
                except Exception:  # noqa: BLE001
                    pass
        agg["grounded_constructed"] += int(grounded_c)
        agg["valid_constructed"] += int(valid_c)
        by_type[qt]["grounded_constructed"] += int(grounded_c)
        rows.append({"example_id": eid, "type": qt,
                     "model_grounded": model_grounded,
                     "grounded_constructed": grounded_c, "valid_constructed": valid_c})

    print(f"\n=== GROUNDED-BY-CONSTRUCTION re-score on the locked test (n={n}, no GPU) ===")
    print(f"  model's OWN citations (unaided)   : {wilson(agg['model_grounded'], n)}")
    print(f"  SYSTEM grounded by construction   : {wilson(agg['grounded_constructed'], n)}")
    print(f"  SYSTEM valid trace (incl. grounded): {wilson(agg['valid_constructed'], n)}")
    print("\n  by type (model-grounded -> system-grounded):")
    for t in sorted(type_n):
        print(f"    {t:<22} {by_type[t]['model_grounded']}/{type_n[t]} -> "
              f"{by_type[t]['grounded_constructed']}/{type_n[t]}")

    Path(OUT).write_text(json.dumps({
        "n": n, "agg": dict(agg), "type_n": dict(type_n),
        "by_type": {t: dict(by_type[t]) for t in by_type},
        "note": "grounded_constructed = trace evidence built from the cells the engine read "
                "(src/executor.evidence_for), re-scored on saved locked-test outputs; "
                "model_grounded = the model's own hand-typed citations (unchanged).",
        "details": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
