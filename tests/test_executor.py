"""Proof that the deterministic executor reproduces the gold answer (P3.6).

(1) Hand-built unit cases per op type, incl. edge cases (empty result, ties, non-numeric
    cells, total-row exclusion via the entity universe).
(2) Dataset round-trip: for EVERY clean-silver / dev / verified example, the operation
    encoded in metadata['spec'] must recompute the stored gold answer. This is the
    foundation the "answer by construction" path rests on - if it isn't 100%, the engine
    (or a spec) is wrong and nothing downstream is trustworthy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.executor import compute_answer
from src.schema import Example, Table
from src.sft_format import operation_dict
from src.table_utils import entity_universe

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def _t() -> Table:
    # rows: A,B,C entities then a Total row (index 3) that the universe must exclude.
    return Table(
        headers=["name", "x", "y"],
        rows=[["A", 10, 5], ["B", 8, 9], ["C", 12, 2], ["Total", 30, 16]],
        column_types=["categorical", "numeric", "numeric"],
    )


def test_best_under_constraint_basic():
    # lowest x while y >= 4, entities only -> A(10) and B(8) qualify (C y=2 fails) -> B
    op = {"type": "best_under_constraint", "target": "x", "target_dir": "lower",
          "constraint": "y", "op": "gte", "threshold": 4, "rows": [0, 1, 2]}
    assert compute_answer(_t(), op).rows == [1]


def test_best_under_constraint_empty():
    op = {"type": "best_under_constraint", "target": "x", "target_dir": "lower",
          "constraint": "y", "op": "gte", "threshold": 999, "rows": [0, 1, 2]}
    ans = compute_answer(_t(), op)
    assert ans.rows == [] and ans.label == "none"


def test_threshold_filter_multi_condition():
    # x < 11 AND y >= 5 -> A(10,5) yes, B(8,9) yes, C(12,2) no
    op = {"type": "threshold_filter", "rows": [0, 1, 2],
          "conditions": [{"metric": "x", "op": "lt", "T": 11}, {"metric": "y", "op": "gte", "T": 5}]}
    assert compute_answer(_t(), op).rows == [0, 1]


def test_tradeoff_frontier_and_metrics():
    # maximize x and y over A(10,5) B(8,9) C(12,2): C dominates none on both; frontier = A,B,C?
    # A(10,5),B(8,9),C(12,2): C has max x, B has max y, A dominated by? A(10,5) vs C(12,2):x C>A,y A>C -> nondom.
    op = {"type": "tradeoff_summary", "m1": "x", "m2": "y", "d1": "higher", "d2": "higher", "rows": [0, 1, 2]}
    ans = compute_answer(_t(), op)
    assert ans.rows == [0, 1, 2]
    assert set(ans.metrics) == {"x", "y"}


def test_total_row_excluded_by_universe():
    # Without restricting rows, the Total row (x=30) would never win 'lowest x', but a
    # 'highest x' extremum would wrongly pick it; the universe must gate that.
    op = {"type": "extremum", "target": "x", "target_dir": "higher", "rows": [0, 1, 2]}
    assert compute_answer(_t(), op).rows == [2]  # C, not Total


def test_evidence_for_is_grounded_and_relevant():
    # Evidence by construction (Step A): every cited cell must really exist & match the
    # table, and cover the question's required metric columns for the answer row.
    from src.executor import evidence_for
    t = _t()
    op = {"type": "best_under_constraint", "target": "x", "target_dir": "lower",
          "constraint": "y", "op": "gte", "threshold": 4, "rows": [0, 1, 2]}
    ev = evidence_for(t, op)
    assert ev, "evidence should not be empty"
    for c in ev:                                  # grounded by construction
        assert t.headers[c.col] == c.col_name
        assert t.cell(c.row, c.col) == c.value
    assert {"x", "y"} <= {c.col_name for c in ev}  # relevant columns covered
    assert any(c.row == 1 for c in ev)             # the winner (B) is cited


def test_evidence_for_tradeoff_covers_frontier_metrics():
    from src.executor import evidence_for
    t = _t()
    op = {"type": "tradeoff_summary", "m1": "x", "m2": "y", "d1": "higher", "d2": "higher",
          "rows": [0, 1, 2]}
    ev = evidence_for(t, op)
    assert {c.col_name for c in ev} == {"x", "y"}
    assert all(t.cell(c.row, c.col) == c.value for c in ev)


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        compute_answer(_t(), {"type": "nope"})


def test_canon_ge_equals_gte():
    base = {"type": "threshold_filter", "rows": [0, 1, 2]}
    ge = {**base, "conditions": [{"metric": "y", "op": "ge", "T": 5}]}
    gte = {**base, "conditions": [{"metric": "y", "op": "gte", "T": 5}]}
    assert compute_answer(_t(), ge).rows == compute_answer(_t(), gte).rows


def test_unknown_threshold_op_rejected():
    # a malformed op must raise, NOT silently default to gte and score a lucky pass
    op = {"type": "threshold_filter", "rows": [0, 1, 2],
          "conditions": [{"metric": "y", "op": "le", "T": 5}]}
    with pytest.raises(ValueError):
        compute_answer(_t(), op)


def test_unknown_direction_rejected():
    with pytest.raises(ValueError):
        compute_answer(_t(), {"type": "extremum", "target": "x", "target_dir": "max", "rows": [0, 1, 2]})


@pytest.mark.parametrize("fname", [
    "realtable_silver_clean.v0_1_0.jsonl",
    "eval_dev.v0_1_0.jsonl",
    "realtable_eval_verified.v0_1_0.jsonl",
])
def test_dataset_roundtrip_reproduces_gold(fname):
    fp = PROC / fname
    if not fp.exists():
        pytest.skip(f"{fname} not present")
    examples = [Example.model_validate_json(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert examples, "no examples loaded"
    mismatches = []
    for ex in examples:
        spec = ex.metadata.get("spec")
        if not isinstance(spec, dict) or "type" not in spec:
            mismatches.append((ex.metadata.get("example_id", ex.table_id), "no spec"))
            continue
        got = compute_answer(ex.table, spec)
        gold = ex.gold_answer
        if sorted(got.rows) != sorted(gold.rows):
            mismatches.append((ex.metadata.get("example_id"), f"rows {got.rows} != {gold.rows}"))
        elif spec["type"] == "tradeoff_summary" and set(got.metrics) != set(gold.metrics):
            mismatches.append((ex.metadata.get("example_id"), f"metrics {got.metrics} != {gold.metrics}"))
    assert not mismatches, f"{len(mismatches)}/{len(examples)} mismatched: {mismatches[:8]}"


def _load(fname):
    fp = PROC / fname
    if not fp.exists():
        pytest.skip(f"{fname} not present")
    return [Example.model_validate_json(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("fname", [
    "realtable_silver_clean.v0_1_0.jsonl", "eval_dev.v0_1_0.jsonl", "realtable_eval_verified.v0_1_0.jsonl",
])
def test_entity_universe_reproduces_spec_rows(fname):
    # The deterministic eval-time universe must equal the gold spec['rows'] (which was built
    # by the same _is_total rule) - otherwise the inference-path universe diverges from gold.
    bad = []
    for ex in _load(fname):
        rows = ex.metadata.get("spec", {}).get("rows")
        if rows is not None and sorted(entity_universe(ex.table)) != sorted(rows):
            bad.append(ex.metadata.get("example_id"))
    assert not bad, f"{len(bad)} universe mismatches: {bad[:8]}"


@pytest.mark.parametrize("fname", [
    "realtable_silver_clean.v0_1_0.jsonl", "eval_dev.v0_1_0.jsonl", "realtable_eval_verified.v0_1_0.jsonl",
])
def test_eval_path_normalized_operation_reproduces_gold(fname):
    # The FULL eval path: the NAME-ONLY operation_dict (what the model is trained to emit) +
    # the deterministically-supplied entity universe must reproduce the gold answer. This proves
    # the path we actually score (not just the spec-fed engine).
    mismatches = []
    for ex in _load(fname):
        op = operation_dict(ex)
        if op.get("type") is None:
            continue
        op = {**op, "rows": entity_universe(ex.table)}
        got = compute_answer(ex.table, op)
        if sorted(got.rows) != sorted(ex.gold_answer.rows):
            mismatches.append((ex.metadata.get("example_id"), f"{got.rows} != {ex.gold_answer.rows}"))
        elif op["type"] == "tradeoff_summary" and set(got.metrics) != set(ex.gold_answer.metrics):
            mismatches.append((ex.metadata.get("example_id"), "metrics"))
    assert not mismatches, f"{len(mismatches)} eval-path mismatches: {mismatches[:8]}"
