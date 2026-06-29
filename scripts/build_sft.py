"""Build the chat-format SFT training file from the clean silver seed (P3.2).

First slice = SILVER ONLY. Drops the tables the frozen split marks as
``exclude_from_training`` (the verified-eval overlap), and hard-asserts that no
training table is an eval table (id + content level) so a future edit can't leak.
Every record is round-tripped through ``reconstruct_example`` + ``validate`` and the
build FAILS if any completion does not re-validate - a converter bug must never ship
corrupt labels into training.

    python scripts/build_sft.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.executor import compute_answer  # noqa: E402
from src.llm_renderer import extract_json, reconstruct_example  # noqa: E402
from src.schema import Example  # noqa: E402
from src.sft_format import completion_str, to_sft_record  # noqa: E402
from src.table_utils import entity_universe  # noqa: E402
from src.splits import (  # noqa: E402
    ROOT,
    assert_no_leak_content,
    assert_no_leak_ids,
    base_table_id,
    load_jsonl,
    load_manifest,
)
from src.trace_validator import validate  # noqa: E402

SILVER = ROOT / "data" / "processed" / "realtable_silver_clean.v0_1_0.jsonl"
OUT = ROOT / "data" / "processed" / "sft_train.v0_1_0.jsonl"


def main() -> int:
    manifest = load_manifest()
    exclude = set(manifest["exclude_from_training"])
    eval_tables = set(manifest["eval_tables_dev"]) | set(manifest["eval_tables_test"])

    raw = load_jsonl(SILVER)
    examples = [Example.model_validate(r) for r in raw]

    kept, dropped, eval_recs_for_check = [], [], []
    for ex in examples:
        bt = base_table_id(ex.table_id)
        if bt in exclude:
            dropped.append(ex)
            continue
        kept.append(ex)

    # --- leakage guards (defense in depth; freeze_splits already enforced these) ---
    train_tables = {base_table_id(ex.table_id) for ex in kept}
    assert_no_leak_ids(train_tables, eval_tables, "sft/eval")
    # content-level: compare kept training records vs the frozen eval sets
    for name in ("eval_dev.v0_1_0.jsonl", "eval_test.v0_1_0.jsonl"):
        eval_recs_for_check += load_jsonl(ROOT / "data" / "processed" / name)
    assert_no_leak_content(
        [ex.model_dump() for ex in kept], eval_recs_for_check, "sft/eval"
    )

    # --- convert + round-trip correctness gate ---
    records, failures = [], []
    for ex in kept:
        rec = to_sft_record(ex)
        parsed = extract_json(rec["messages"][1]["content"])
        rebuilt = reconstruct_example(ex, parsed)
        res = validate(rebuilt)
        if not res.valid:
            failures.append((rec["example_id"], res.failed_checks()))
            continue
        # defense-in-depth (P3.6): the emitted name-only operation + the deterministic entity
        # universe must independently reproduce gold (executor's predicates != validator's).
        op = parsed.get("operation")
        try:
            computed = compute_answer(ex.table, {**op, "rows": entity_universe(ex.table)})
            op_ok = sorted(computed.rows) == sorted(ex.gold_answer.rows) and (
                op.get("type") != "tradeoff_summary"
                or set(computed.metrics) == set(ex.gold_answer.metrics)
            )
        except Exception as e:  # noqa: BLE001
            failures.append((rec["example_id"], [f"operation_exec: {type(e).__name__}: {e}"]))
            continue
        if not op_ok:
            failures.append((rec["example_id"], [f"operation_exec: rows {computed.rows} != gold {ex.gold_answer.rows}"]))
            continue
        records.append(rec)

    if failures:
        print(f"ROUND-TRIP FAILURES: {len(failures)} (converter bug - not writing output)")
        for eid, checks in failures[:10]:
            print(f"  {eid}: {checks}")
        return 1

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type = Counter(r["question_type"] for r in records)
    print("Built SFT training file (P3.2) - silver only, first slice")
    print(f"  source silver examples : {len(examples)}")
    print(f"  dropped (eval overlap) : {len(dropped)}  tables={sorted(exclude)}")
    print(f"  written SFT records    : {len(records)}")
    print(f"  by type                : {dict(by_type)}")
    print(f"  round-trip validation  : {len(records)}/{len(kept)} valid (100% required)")
    print(f"  leakage (id+content)   : PASS")
    print(f"\n  out : {OUT.relative_to(ROOT)}")
    # show one record's shape (truncated) for a human sanity glance
    sample = records[0]
    print(f"\n  sample example_id: {sample['example_id']}")
    print(f"  sample completion: {sample['messages'][1]['content'][:240]} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
