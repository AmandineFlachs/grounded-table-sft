"""P3.2 converter: SFT completions must round-trip back to valid examples.

Uses the programmatic real-table dataset (always present, gold-by-construction) so
the test is deterministic and independent of the silver seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_renderer import build_prompt, extract_json, reconstruct_example
from src.schema import Example
from src.sft_format import completion_dict, completion_str, to_sft_record
from src.trace_validator import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "realtable.v0_1_0.jsonl"


def _sample(n=20):
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Example.model_validate(json.loads(line)))
            if len(rows) >= n:
                break
    return rows


@pytest.mark.skipif(not DATA.exists(), reason="realtable dataset not built")
def test_completion_dict_shape():
    ex = _sample(1)[0]
    cd = completion_dict(ex)
    assert set(cd) == {"trace_steps", "final_answer", "operation"}
    assert set(cd["final_answer"]) == {"label", "rows", "metrics"}
    for step in cd["trace_steps"]:
        assert set(step) == {"kind", "description", "cites"}
        for c in step["cites"]:
            # contract: row/col_name/value only - no index, no col
            assert set(c) == {"row", "col_name", "value"}
    # operation is name-only: carries a type but never row indices or index fields
    op = cd["operation"]
    assert op.get("type")
    assert "rows" not in op and "col" not in op and "c1" not in op


@pytest.mark.skipif(not DATA.exists(), reason="realtable dataset not built")
def test_roundtrip_revalidates():
    for ex in _sample(20):
        rec = to_sft_record(ex)
        # user turn must be exactly the render prompt (train==inference)
        assert rec["messages"][0]["content"] == build_prompt(ex)
        parsed = extract_json(rec["messages"][1]["content"])
        rebuilt = reconstruct_example(ex, parsed)
        res = validate(rebuilt)
        assert res.valid, f"{ex.table_id}: {res.failed_checks()}"


@pytest.mark.skipif(not DATA.exists(), reason="realtable dataset not built")
def test_completion_is_compact_json():
    ex = _sample(1)[0]
    s = completion_str(ex)
    assert "\n" not in s  # single line
    json.loads(s)  # parseable
