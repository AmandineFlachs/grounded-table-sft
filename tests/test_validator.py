"""Validator unit tests.

Two halves:
  1. A freshly built dataset must be 100% valid (programmatic traces are
     grounded by construction).
  2. Deliberately-broken examples MUST fail the right check - otherwise the
     validator isn't actually guarding anything.
"""

from __future__ import annotations

import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_builder import build_dataset  # noqa: E402
from src.schema import Example, QuestionType  # noqa: E402
from src.trace_validator import validate, validate_dict  # noqa: E402


def _build(n=60, seed=0):
    examples, invalid = build_dataset(n=n, seed=seed)
    return examples, invalid


def test_built_dataset_is_fully_valid():
    examples, invalid = _build()
    assert invalid == [], f"unexpected invalid examples: {[r.example_id for r in invalid]}"
    assert len(examples) == 60
    for ex in examples:
        assert validate(ex).valid


def test_all_three_types_present():
    examples, _ = _build()
    types = {e.question_type for e in examples}
    assert QuestionType.BEST_UNDER_CONSTRAINT in types
    assert QuestionType.THRESHOLD_FILTER in types
    assert QuestionType.TRADEOFF_SUMMARY in types


def test_hard_cases_present():
    examples, _ = _build()
    hard = [e for e in examples if e.metadata.get("difficulty") == "hard"]
    assert hard, "expected some hard/edge-case examples"


def _first(examples, qtype) -> Example:
    return next(e for e in examples if e.question_type == qtype)


def test_corrupted_cell_value_fails_cells_exist():
    examples, _ = _build()
    ex = copy.deepcopy(examples[0])
    # Corrupt the value of the first cited cell.
    ex.trace_steps[0].cites[0].value = "DEFINITELY_WRONG"
    res = validate(ex)
    assert not res.valid
    assert "cells_exist" in res.failed_checks()


def test_out_of_range_cell_fails():
    examples, _ = _build()
    ex = copy.deepcopy(examples[0])
    ex.evidence_cells[0].row = 9999
    res = validate(ex)
    assert not res.valid
    assert "cells_exist" in res.failed_checks()


def test_wrong_answer_rows_fail_answer_correct():
    examples, _ = _build()
    ex = copy.deepcopy(_first(examples, QuestionType.TRADEOFF_SUMMARY))
    # Flip the frontier to something certainly wrong (all rows reversed/extended).
    all_rows = list(range(len(ex.table.rows)))
    ex.gold_answer.rows = [r for r in all_rows if r not in ex.gold_answer.rows] or [0]
    # ensure it actually differs from the true frontier
    res = validate(ex)
    assert not res.valid
    assert "answer_correct" in res.failed_checks()


def test_missing_difficulty_tag_fails():
    examples, _ = _build()
    ex = copy.deepcopy(examples[0])
    ex.metadata.pop("difficulty", None)
    res = validate(ex)
    assert not res.valid
    assert "difficulty_tag_present" in res.failed_checks()


def test_missing_spec_fails_answer_correct():
    examples, _ = _build()
    ex = copy.deepcopy(examples[0])
    ex.metadata.pop("spec", None)
    res = validate(ex)
    assert not res.valid
    assert "answer_correct" in res.failed_checks()


def test_schema_invalid_dict_fails():
    res = validate_dict({"not": "an example"})
    assert not res.valid
    assert "schema_valid" in res.failed_checks()


def test_empty_result_label_consistency():
    examples, _ = _build()
    # Find (or skip) an empty-result example and break its label.
    empties = [e for e in examples if not e.gold_answer.rows]
    if not empties:
        pytest.skip("no empty-result example in this sample")
    ex = copy.deepcopy(empties[0])
    ex.gold_answer.label = "SOMETHING"
    res = validate(ex)
    assert not res.valid
    assert "empty_result_consistency" in res.failed_checks()
