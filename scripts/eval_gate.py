"""Step B - out-of-vocabulary SAFETY GATE, measured on saved outputs (no GPU).

The executor fails CONFIDENTLY out-of-distribution: on the OOD extremum slice the model emits a
*trained* op type (best_under_constraint) for a question that has no constraint, so the engine
computes a wrong answer 0/12 - while the model's OWN answer is right 100%. A naive "is the op type
known?" gate can't catch this (the type IS known). The principled gate: check the QUESTION actually
contains the signal the emitted op type REQUIRES; if not, the operation is fabricated, so DON'T
trust the engine - fall back to the model's own answer ("do no harm": never worse than the model).

Gate signals (deployable - from question text + op type, no gold):
  * best_under_constraint / threshold_filter  REQUIRE threshold language ("while", "under",
    "at least", "at most", "below", "exceed", "less/greater than", ...). Absent -> fabricated.
  * tradeoff_summary  REQUIRES trade-off language ("pareto", "trade-off", "optimal"). Absent -> susp.

Re-scores SAVED outputs (in-distribution + OOD) so it needs no model re-run. Reports gated answer
accuracy vs engine-only vs model-only, and how often the gate fires on each set.

    python scripts/eval_gate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo import gate_fires                     # noqa: E402  (canonical gate, shared with the demo)
from src.llm_renderer import extract_operation     # noqa: E402
from src.schema import Example                      # noqa: E402
from src.splits import load_jsonl                    # noqa: E402

IN_DIST = ("results/p3_4b_exec_TEST.json", "data/processed/eval_test.v0_1_0.jsonl")
OOD = ("results/p3_4b_exec_TEST_verified.json", "data/processed/realtable_eval_verified.v0_1_0.jsonl")
OUT = "results/p3_4b_gate.json"


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100*p:.1f}% [{100*max(0,c-h):.1f}-{100*min(1,c+h):.1f}]"


def score(results_path: str, source_path: str, label: str) -> dict:
    det = json.loads(Path(results_path).read_text(encoding="utf-8"))["details"]
    src = {}
    for r in load_jsonl(Path(source_path)):
        ex = Example.model_validate(r)
        src[ex.metadata.get("example_id", ex.table_id)] = ex.question
    n = len(det)
    engine = model = gated = fires = 0
    for d in det:
        q = src.get(d["example_id"], "")
        op = extract_operation(d["raw"]) if d.get("raw") else None
        fire = gate_fires(op, q)
        eok, mok = bool(d.get("engine_ok")), bool(d.get("model_ok"))
        engine += int(eok)
        model += int(mok)
        gated += int(mok if fire else eok)
        fires += int(fire)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  engine-only answer   : {wilson(engine, n)}")
    print(f"  model-only answer    : {wilson(model, n)}")
    print(f"  GATED (fall back when op unsupported): {wilson(gated, n)}")
    print(f"  gate fired on        : {fires}/{n} ({100*fires/n:.0f}%)")
    return {"label": label, "n": n, "engine": engine, "model": model, "gated": gated, "fires": fires}


def main() -> int:
    out = [score(*IN_DIST, "IN-DISTRIBUTION locked test"),
           score(*OOD, "OUT-OF-DISTRIBUTION extremum anchor")]
    Path(OUT).write_text(json.dumps({
        "results": out,
        "note": "gate falls back to the model's own answer when the emitted op type's required "
                "question signal (threshold / trade-off language) is absent; do-no-harm safety gate; "
                "heuristic validated on a small OOD set.",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
