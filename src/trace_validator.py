"""Rule-based groundedness & correctness validation.

Runs on EVERY example before it is written to the dataset. The validator
**independently recomputes** the answer from the structured ``spec`` in
``metadata`` (it does not trust the stored gold/trace), then checks that every
cited cell exists and matches, that comparisons hold, and that trade-off
frontiers are exactly the non-dominated set.

This is the quality gate that will later discard bad LLM-generated traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from .schema import Example

EPS = 1e-9

CHECK_NAMES = [
    "schema_valid",
    "cells_exist",
    "answer_correct",
    "numeric_comparisons_correct",
    "threshold_satisfaction",
    "relevance",
    "pareto_frontier_correct",
    "empty_result_consistency",
    "difficulty_tag_present",
]


@dataclass
class ValidationResult:
    valid: bool
    checks: dict[str, bool]
    issues: list[str] = field(default_factory=list)

    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]


# --------------------------------------------------------------------------- #
# independent predicate re-implementations (not shared with the generator)
# --------------------------------------------------------------------------- #
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _passes(value: float, op: str, threshold: float) -> bool:
    return value < threshold if op == "lt" else value >= threshold


def _num_passes(value, op: str, threshold: float) -> bool:
    """Like ``_passes`` but a non-numeric cell (e.g. an empty real-table cell)
    can never satisfy a numeric threshold."""
    return _is_num(value) and _passes(value, op, threshold)


def _value_eq(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 1e-6
    return a == b


def _optimum_rows(rows: list[int], col: int, table, direction: str) -> list[int]:
    vals = [(r, table.cell(r, col)) for r in rows]
    best = max(v for _, v in vals) if direction == "higher" else min(v for _, v in vals)
    return sorted(r for r, v in vals if abs(v - best) <= EPS)


def _frontier(table, c1: int, c2: int, d1: str, d2: str, universe=None) -> list[int]:
    def obj(v, d):
        return v if d == "higher" else -v
    rows = range(len(table.rows)) if universe is None else universe
    # Only rows numeric in both axes are comparable on the frontier.
    eligible = [r for r in rows
                if _is_num(table.cell(r, c1)) and _is_num(table.cell(r, c2))]
    pts = [(r, obj(table.cell(r, c1), d1), obj(table.cell(r, c2), d2)) for r in eligible]
    out = []
    for ri, oi1, oi2 in pts:
        dominated = any(
            rj != ri and oj1 >= oi1 - EPS and oj2 >= oi2 - EPS and (oj1 > oi1 + EPS or oj2 > oi2 + EPS)
            for rj, oj1, oj2 in pts
        )
        if not dominated:
            out.append(ri)
    return sorted(out)


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def validate_dict(raw: dict) -> ValidationResult:
    """Validate a raw dict: first parse against the schema, then run checks."""
    try:
        ex = Example.model_validate(raw)
    except ValidationError as e:
        checks = {name: False for name in CHECK_NAMES}
        return ValidationResult(valid=False, checks=checks,
                                issues=[f"schema_valid: {e.error_count()} schema error(s)"])
    return validate(ex)


def validate(ex: Example) -> ValidationResult:
    checks: dict[str, bool] = {name: True for name in CHECK_NAMES}
    issues: list[str] = []

    def fail(check: str, msg: str) -> None:
        checks[check] = False
        issues.append(f"{check}: {msg}")

    table = ex.table
    n_rows, n_cols = len(table.rows), len(table.headers)

    # --- cells_exist ---
    all_cells = [c for s in ex.trace_steps for c in s.cites] + list(ex.evidence_cells)
    for c in all_cells:
        if not (0 <= c.row < n_rows and 0 <= c.col < n_cols):
            fail("cells_exist", f"cell ({c.row},{c.col}) out of range")
            continue
        if table.headers[c.col] != c.col_name:
            fail("cells_exist", f"col_name '{c.col_name}' != header '{table.headers[c.col]}' at col {c.col}")
        if not _value_eq(table.cell(c.row, c.col), c.value):
            fail("cells_exist", f"value {c.value!r} != table[{c.row}][{c.col}]={table.cell(c.row, c.col)!r}")

    spec = ex.metadata.get("spec")
    if not isinstance(spec, dict) or "type" not in spec:
        fail("answer_correct", "missing structured spec in metadata; cannot recompute")
    else:
        _check_answer(ex, spec, fail)

    # --- relevance: evidence must cite the question's metric columns ---
    cited_cols = {c.col_name for c in ex.evidence_cells}
    required = _required_metrics(spec) if isinstance(spec, dict) else set()
    # An empty best_under_constraint never reaches the target metric (no row
    # satisfies the constraint), so the target column is not genuinely required.
    if isinstance(spec, dict) and spec.get("type") == "best_under_constraint" and not ex.gold_answer.rows:
        required.discard(spec.get("target"))
    missing = required - cited_cols
    if missing:
        fail("relevance", f"evidence does not cite required metric column(s): {sorted(missing)}")

    # --- empty_result_consistency ---
    if not ex.gold_answer.rows:
        if ex.gold_answer.label != "none":
            fail("empty_result_consistency", f"empty answer but label is '{ex.gold_answer.label}'")
        if ex.trace_steps[-1].kind != "conclude":
            fail("empty_result_consistency", "no concluding step for empty result")

    # --- difficulty_tag_present ---
    md = ex.metadata
    if md.get("difficulty") not in ("easy", "hard") or not isinstance(md.get("edge_cases"), list):
        fail("difficulty_tag_present", "metadata missing difficulty/edge_cases tags")

    return ValidationResult(valid=all(checks.values()), checks=checks, issues=issues)


def _required_metrics(spec: dict) -> set[str]:
    t = spec.get("type")
    if t == "best_under_constraint":
        return {spec["target"], spec["constraint"]}
    if t == "threshold_filter":
        return {c["metric"] for c in spec["conditions"]}
    if t == "tradeoff_summary":
        return {spec["m1"], spec["m2"]}
    if t == "extremum":
        return {spec["target"]}
    return set()


def _check_answer(ex: Example, spec: dict, fail) -> None:
    table = ex.table
    gold = ex.gold_answer
    t = spec["type"]

    # The universe of entity rows a question ranges over. Real-table specs record
    # it (e.g. to exclude total/subtotal rows); synthetic specs omit it -> all rows.
    universe = spec.get("rows")
    if universe is None:
        universe = list(range(len(table.rows)))
    else:
        universe = [r for r in universe if 0 <= r < len(table.rows)]

    if t == "best_under_constraint":
        cc = table.col_index(spec["constraint"])
        tc = table.col_index(spec["target"])
        # A candidate must satisfy the (numeric) constraint AND have a numeric
        # target value to be rankable - real tables can have empty cells.
        survivors = [r for r in universe
                     if _is_num(table.cell(r, tc))
                     and _num_passes(table.cell(r, cc), spec["op"], spec["threshold"])]
        if not survivors:
            expected: list[int] = []
        else:
            expected = _optimum_rows(survivors, tc, table, spec["target_dir"])
        if sorted(gold.rows) != expected:
            fail("answer_correct", f"gold rows {sorted(gold.rows)} != recomputed {expected}")
        # threshold satisfaction: winners are within survivors
        if any(r not in survivors for r in gold.rows):
            fail("threshold_satisfaction", "a gold winner does not satisfy the constraint")
        # numeric comparison: each winner is truly optimal among survivors
        if survivors and gold.rows:
            best = table.cell(gold.rows[0], tc)
            better = [r for r in survivors
                      if (table.cell(r, tc) > best + EPS) == (spec["target_dir"] == "higher")
                      and abs(table.cell(r, tc) - best) > EPS]
            if better:
                fail("numeric_comparisons_correct", "a survivor is better than the chosen winner")

    elif t == "threshold_filter":
        conds = spec["conditions"]

        def _row_satisfies(r: int) -> bool:
            return all(_num_passes(table.cell(r, table.col_index(c["metric"])), c["op"], c["T"])
                       for c in conds)

        matches = [r for r in universe if _row_satisfies(r)]
        if sorted(gold.rows) != sorted(matches):
            fail("answer_correct", f"gold rows {sorted(gold.rows)} != recomputed {sorted(matches)}")
        for r in gold.rows:
            if not _row_satisfies(r):
                fail("threshold_satisfaction", f"row {r} does not satisfy all conditions")
        for r in universe:
            if _row_satisfies(r) and r not in gold.rows:
                fail("numeric_comparisons_correct", f"row {r} satisfies conditions but is missing from the answer")

    elif t == "extremum":
        # Single-column argmax/argmin over rows (no constraint). Used for the
        # TAT-QA-gold-anchored eval slice adapted from TAT-QA's native superlatives.
        tc = table.col_index(spec["target"])
        cand = [r for r in universe if _is_num(table.cell(r, tc))]
        expected = _optimum_rows(cand, tc, table, spec["target_dir"]) if cand else []
        if sorted(gold.rows) != expected:
            fail("answer_correct", f"gold rows {sorted(gold.rows)} != recomputed {expected}")
        if cand and gold.rows:
            best = table.cell(gold.rows[0], tc)
            better = [r for r in cand
                      if (table.cell(r, tc) > best + EPS) == (spec["target_dir"] == "higher")
                      and abs(table.cell(r, tc) - best) > EPS]
            if better:
                fail("numeric_comparisons_correct", "a row is more extreme than the chosen one")

    elif t == "tradeoff_summary":
        front = _frontier(table, spec["c1"], spec["c2"], spec["d1"], spec["d2"], universe)
        if sorted(gold.rows) != front:
            fail("answer_correct", f"gold frontier {sorted(gold.rows)} != recomputed {front}")
            fail("pareto_frontier_correct", "cited frontier is not the non-dominated set")
        # Metric order is cosmetic (the question's two axes); compare as a set.
        if set(gold.metrics) != {spec["m1"], spec["m2"]}:
            fail("answer_correct", f"gold metrics {gold.metrics} != {{{spec['m1']}, {spec['m2']}}}")
    else:
        fail("answer_correct", f"unknown spec type '{t}'")
