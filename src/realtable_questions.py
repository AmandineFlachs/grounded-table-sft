"""Generate our grounded selection questions over *real* (ingested) tables.

Phase 2 / P2.3. Strategy (A): we take real tables (from TAT-QA, ingested into the
canonical ``Table`` schema) and generate our own ``best_under_constraint`` /
``threshold_filter`` / ``tradeoff_summary`` questions over them - the native
TAT-QA questions (mostly arithmetic) are deferred to Phase 3.

Direction is handled by **explicit-direction phrasing** (methodology decision,
2026-06-20): the question states "highest"/"lowest" and the comparison operator,
so the gold answer is a pure mechanical argmax/argmin/threshold/frontier with no
domain "goodness" judgment. Gold is therefore grounded-by-construction and the
existing ``trace_validator`` recomputes it from the same ``spec`` format used for
synthetic tables - so it independently re-checks every example we emit here.

TAT-QA tables are *transposed* vs synthetic ones (rows = line items, columns =
periods), which yields two orientations, both produced by the same builders:
  * ``across_rows``    : entities = line items, metrics = period columns
                         ("which line item had the highest value in 2019?")
  * ``across_columns`` : the table transposed, so entities = periods, metrics =
                         line items ("in which year was Revenue highest?")
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .ingest.tatqa import IngestedTable, TATQA_ATTRIBUTION
from .question_templates import (
    _cells,
    _choose_threshold,
    _dedup,
    _name,
    _optimum,
    _pareto_frontier,
    _passes,
)
from .schema import Example, GoldAnswer, QuestionType, TraceStep
from .table_utils import _fmt as num
from .table_utils import infer_column_types, is_total as _is_total
from .schema import Table

Orientation = Literal["across_rows", "across_columns"]
Mode = Literal["normal", "empty", "near_threshold"]  # no forced "tie": never mutate real data
Direction = Literal["higher", "lower"]

# Total/subtotal exclusion (the entity universe) is defined ONCE in
# table_utils.is_total (imported above as _is_total) so question generation and the
# executor's eval-time universe share the same detector. See methodology log j.


@dataclass
class RealTable:
    """An ingested table prepared for question generation in one orientation."""

    table: Table
    table_id: str
    metric_cols: list[str]
    entity_noun: str
    orientation: Orientation
    source_uid: str
    confidence: str
    # The entity rows that questions range over. Total/subtotal rows are excluded
    # for across_rows (they make argmax trivial); all rows for across_columns
    # (periods are never totals). Stored into each spec so the validator
    # recomputes over the SAME universe.
    entity_rows: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# orientation adapters
# --------------------------------------------------------------------------- #
def as_across_rows(it: IngestedTable) -> RealTable:
    """Entities = line items (rows), metrics = period columns. Used as-is.
    Total/subtotal rows are excluded from the entity universe."""
    t = it.table
    entity_rows = [r for r in range(len(t.rows)) if not _is_total(str(t.cell(r, 0)))]
    return RealTable(
        table=t,
        table_id=f"{it.table_id}_rows",
        metric_cols=list(it.metric_cols),
        entity_noun="line item",
        orientation="across_rows",
        source_uid=it.source_uid,
        confidence=it.confidence,
        entity_rows=entity_rows,
    )


def transpose(it: IngestedTable) -> Optional[RealTable]:
    """Transpose so entities = periods (rows), metrics = line items (columns).

    Returns ``None`` if the result has no numeric columns to query.
    """
    t = it.table
    n_rows = len(t.rows)
    if n_rows == 0 or len(t.headers) < 2:
        return None

    # New col 0 = the original column headers (periods); new columns = line items.
    new_headers = [t.headers[0]] + [str(t.cell(r, 0)) for r in range(n_rows)]
    new_rows: list[list] = []
    for c in range(1, len(t.headers)):
        new_rows.append([t.headers[c]] + [t.cell(r, c) for r in range(n_rows)])

    # Dedup headers (line-item labels can repeat).
    seen: dict[str, int] = {}
    uniq: list[str] = []
    for h in new_headers:
        name = h if h.strip() else "period"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        uniq.append(name)
    new_headers = uniq

    column_types = infer_column_types(new_headers, new_rows)
    if column_types:
        column_types[0] = "categorical"
    table = Table(headers=new_headers, rows=new_rows, column_types=column_types)
    metric_cols = [h for c, h in enumerate(new_headers) if column_types[c] == "numeric"]
    if not metric_cols:
        return None

    return RealTable(
        table=table,
        table_id=f"{it.table_id}_cols",
        metric_cols=metric_cols,
        entity_noun="period",
        orientation="across_columns",
        source_uid=it.source_uid,
        confidence=it.confidence,
        # A source "Total" COLUMN becomes a row after transpose - exclude it too.
        entity_rows=[r for r in range(len(new_rows)) if not _is_total(str(table.cell(r, 0)))],
    )


# --------------------------------------------------------------------------- #
# phrasing helpers (explicit-direction)
# --------------------------------------------------------------------------- #
def _ref(rt: RealTable, metric: str) -> str:
    """Noun phrase for a metric: 'value in 2019' (period cols) or 'Revenue'."""
    return f"value in {metric}" if rt.orientation == "across_rows" else metric


def _constraint_phrase(rt: RealTable, metric: str, direction: Direction, T: float) -> str:
    ref = _ref(rt, metric)
    return f"{ref} under {num(T)}" if direction == "lower" else f"{ref} of at least {num(T)}"


def _entities(rt: RealTable, rows: list[int]) -> str:
    return ", ".join(_name(rt.table, r) for r in rows) if rows else f"no {rt.entity_noun}"


def _has_excluded_totals(rt: RealTable) -> bool:
    """True when the oriented table contains total/subtotal rows (which the gold
    universe excludes). Drives whether to state the convention in the question."""
    return any(_is_total(str(rt.table.cell(r, 0))) for r in range(len(rt.table.rows)))


def _totals_note(rt: RealTable) -> str:
    """Make the gold's universe explicit in the question text: our gold excludes
    total/subtotal rows, but a model reading the raw table would naturally count
    them (measured: 100% of answer ambiguity traces to totals). Stated only when
    such rows exist, so questions stay natural when they don't."""
    return " Exclude any total or subtotal rows." if _has_excluded_totals(rt) else ""


def _rt_example(rt: RealTable, qtype: QuestionType, mode: str, question: str,
                gold: GoldAnswer, steps: list[TraceStep], edge: Optional[str],
                spec: dict) -> Example:
    evidence = _dedup([c for s in steps for c in s.cites])
    return Example(
        table_id=rt.table_id,
        domain="finance",
        table=rt.table,
        question=question,
        question_type=qtype,
        gold_answer=gold,
        trace_steps=steps,
        evidence_cells=evidence,
        trace_source="programmatic",
        metadata={
            "difficulty": "hard" if edge else "easy",
            "edge_cases": [edge] if edge else [],
            "example_id": f"{rt.table_id}_{qtype.value}_{mode}",
            "spec": spec,
            "orientation": rt.orientation,
            "confidence": rt.confidence,
            "source": {
                "dataset": "TAT-QA",
                "uid": rt.source_uid,
                "attribution": TATQA_ATTRIBUTION,
            },
        },
    )


# --------------------------------------------------------------------------- #
# 1. best_under_constraint
# --------------------------------------------------------------------------- #
def build_best_under_constraint(rt: RealTable, rng: random.Random,
                                mode: Mode = "normal") -> Optional[Example]:
    table = rt.table
    if len(rt.metric_cols) < 2:
        return None
    target, constraint = rng.sample(rt.metric_cols, 2)
    td: Direction = rng.choice(["higher", "lower"])
    cd: Direction = rng.choice(["higher", "lower"])
    col_t, col_c = table.col_index(target), table.col_index(constraint)
    rows = [r for r in rt.entity_rows
            if isinstance(table.cell(r, col_t), (int, float))
            and isinstance(table.cell(r, col_c), (int, float))]
    if len(rows) < 2:
        return None

    op_c_mode: Mode = mode
    T, op_c = _choose_threshold([table.cell(r, col_c) for r in rows], cd, op_c_mode, rng)
    survivors = [r for r in rows if _passes(table.cell(r, col_c), op_c, T)]
    spec = {"type": "best_under_constraint", "target": target, "target_dir": td,
            "constraint": constraint, "constraint_dir": cd, "op": op_c, "threshold": T,
            "rows": list(rt.entity_rows)}

    best_word = "highest" if td == "higher" else "lowest"
    question = (f"Which {rt.entity_noun} has the {best_word} {_ref(rt, target)} while keeping "
                f"{_constraint_phrase(rt, constraint, cd, T)}?" + _totals_note(rt))

    steps: list[TraceStep] = [TraceStep(
        index=0, kind="filter",
        description=("Apply the constraint " + _constraint_phrase(rt, constraint, cd, T) + ": "
                     + (", ".join(f"{_name(table, r)} ({num(table.cell(r, col_c))})" for r in survivors)
                        if survivors else f"no {rt.entity_noun} qualifies") + "."),
        cites=_cells(table, rows, col_c),
    )]

    if not survivors:
        steps.append(TraceStep(
            index=1, kind="conclude",
            description=f"No {rt.entity_noun} satisfies {_constraint_phrase(rt, constraint, cd, T)}, "
                        f"so there is no answer.",
            cites=_cells(table, rows, col_c),
        ))
        return _rt_example(rt, QuestionType.BEST_UNDER_CONSTRAINT, mode, question,
                           GoldAnswer(label="none", rows=[]), steps, "empty", spec)

    winners, best_val = _optimum(survivors, col_t, table, td)
    steps.append(TraceStep(
        index=1, kind="select",
        description=(f"Among the qualifying {rt.entity_noun}s, the {best_word} {_ref(rt, target)} "
                     f"is {num(best_val)} (" + ", ".join(_name(table, r) for r in winners) + ")."),
        cites=_cells(table, survivors, col_t),
    ))
    if len(winners) == 1:
        label = _name(table, winners[0])
        concl = f"{label} is the answer: {best_word} {_ref(rt, target)} ({num(best_val)})."
        edge: Optional[str] = "near_threshold" if mode == "near_threshold" else None
    else:
        label = "tie: " + ", ".join(_name(table, r) for r in winners)
        concl = (f"There is a tie for the {best_word} {_ref(rt, target)} ({num(best_val)}): "
                 + ", ".join(_name(table, r) for r in winners) + ".")
        edge = "argmax_tie"
    steps.append(TraceStep(index=2, kind="conclude", description=concl,
                           cites=_dedup(_cells(table, winners, col_t) + _cells(table, winners, col_c))))
    gold = GoldAnswer(label=label, rows=sorted(winners))
    return _rt_example(rt, QuestionType.BEST_UNDER_CONSTRAINT, mode, question, gold, steps, edge, spec)


# --------------------------------------------------------------------------- #
# 2. threshold_filter
# --------------------------------------------------------------------------- #
def build_threshold_filter(rt: RealTable, rng: random.Random,
                           mode: Mode = "normal") -> Optional[Example]:
    table = rt.table
    if not rt.metric_cols:
        return None
    k = min(len(rt.metric_cols), rng.choice([1, 2]))
    metrics = rng.sample(rt.metric_cols, k)
    rows = list(rt.entity_rows)

    conds = []
    for idx, m in enumerate(metrics):
        d: Direction = rng.choice(["higher", "lower"])
        col = table.col_index(m)
        vals = [table.cell(r, col) for r in rows if isinstance(table.cell(r, col), (int, float))]
        if len(vals) < 2:
            return None
        cmode: Mode = mode if idx == 0 else "normal"
        T, op = _choose_threshold(vals, d, cmode, rng)
        conds.append({"metric": m, "col": col, "dir": d, "op": op, "T": T})

    def passes_all(r: int) -> bool:
        for c in conds:
            v = table.cell(r, c["col"])
            if not isinstance(v, (int, float)) or not _passes(v, c["op"], c["T"]):
                return False
        return True

    matches = [r for r in rows if passes_all(r)]
    cond_text = " and ".join(_constraint_phrase(rt, c["metric"], c["dir"], c["T"]) for c in conds)
    question = f"Which {rt.entity_noun}s have {cond_text}?" + _totals_note(rt)

    steps: list[TraceStep] = []
    for i, c in enumerate(conds):
        passing = [r for r in rows if isinstance(table.cell(r, c["col"]), (int, float))
                   and _passes(table.cell(r, c["col"]), c["op"], c["T"])]
        steps.append(TraceStep(
            index=i, kind="filter",
            description=(f"Condition {_constraint_phrase(rt, c['metric'], c['dir'], c['T'])} keeps: "
                        + (", ".join(_name(table, r) for r in passing) if passing
                           else f"no {rt.entity_noun}") + "."),
            cites=_cells(table, rows, c["col"]),
        ))

    if matches:
        label = ", ".join(_name(table, r) for r in matches)
        concl = "All conditions are satisfied by: " + label + "."
    else:
        label = "none"
        concl = f"No {rt.entity_noun} satisfies all conditions."
    concl_cells = _dedup([cell for c in conds for cell in _cells(table, matches, c["col"])])
    steps.append(TraceStep(index=len(conds), kind="conclude", description=concl, cites=concl_cells))

    edge = "empty" if not matches else ("near_threshold" if mode == "near_threshold" else None)
    gold = GoldAnswer(label=label, rows=sorted(matches))
    spec = {"type": "threshold_filter",
            "conditions": [{"metric": c["metric"], "dir": c["dir"], "op": c["op"],
                            "T": c["T"], "col": c["col"]} for c in conds],
            "rows": list(rt.entity_rows)}
    return _rt_example(rt, QuestionType.THRESHOLD_FILTER, mode, question, gold, steps, edge, spec)


# --------------------------------------------------------------------------- #
# 3. tradeoff_summary (Pareto frontier, maximize-both framing)
# --------------------------------------------------------------------------- #
def build_tradeoff_summary(rt: RealTable, rng: random.Random,
                           mode: Mode = "normal") -> Optional[Example]:
    table = rt.table
    if len(rt.metric_cols) < 2:
        return None
    m1, m2 = rng.sample(rt.metric_cols, 2)
    c1, c2 = table.col_index(m1), table.col_index(m2)
    # Restrict to entity rows numeric in both axes; Pareto needs a clean comparison.
    rows = [r for r in rt.entity_rows
            if isinstance(table.cell(r, c1), (int, float))
            and isinstance(table.cell(r, c2), (int, float))]
    if len(rows) < 2:
        return None
    # Explicit framing: maximize both axes (no domain "goodness" assumed).
    d1: Direction = "higher"
    d2: Direction = "higher"

    # Frontier over the eligible rows only.
    def obj(v: float, d: Direction) -> float:
        return v if d == "higher" else -v
    pts = [(r, obj(table.cell(r, c1), d1), obj(table.cell(r, c2), d2)) for r in rows]
    frontier = sorted(
        ri for ri, oi1, oi2 in pts
        if not any(rj != ri and oj1 >= oi1 - 1e-9 and oj2 >= oi2 - 1e-9
                   and (oj1 > oi1 + 1e-9 or oj2 > oi2 + 1e-9)
                   for rj, oj1, oj2 in pts)
    )

    question = (f"Among these {rt.entity_noun}s, which are Pareto-optimal when maximizing both "
                f"{_ref(rt, m1)} and {_ref(rt, m2)}?" + _totals_note(rt))

    def pair(r: int) -> str:
        return f"{_name(table, r)} ({m1}={num(table.cell(r, c1))}, {m2}={num(table.cell(r, c2))})"

    steps = [
        TraceStep(index=0, kind="aggregate",
                  description=f"Read {_ref(rt, m1)} and {_ref(rt, m2)} for every {rt.entity_noun} "
                              f"(higher is better on both).",
                  cites=_dedup(_cells(table, rows, c1) + _cells(table, rows, c2))),
        TraceStep(index=1, kind="select",
                  description=("Non-dominated (Pareto-optimal) - no other "
                               f"{rt.entity_noun} is at least as high in both and strictly higher "
                               "in one: " + "; ".join(pair(r) for r in frontier) + "."),
                  cites=_dedup(_cells(table, frontier, c1) + _cells(table, frontier, c2))),
    ]
    dominated = [r for r in rows if r not in frontier]
    if len(frontier) == 1:
        concl = f"{_name(table, frontier[0])} dominates on both, so there is no real trade-off."
        edge: Optional[str] = "single_dominator"
    elif not dominated:
        concl = f"Every {rt.entity_noun} is Pareto-optimal: each trades {_ref(rt, m1)} against {_ref(rt, m2)}."
        edge = "no_domination"
    else:
        concl = (f"The frontier ({', '.join(_name(table, r) for r in frontier)}) trades "
                 f"{_ref(rt, m1)} against {_ref(rt, m2)}; "
                 f"{', '.join(_name(table, r) for r in dominated)} are lower on both.")
        edge = None
    steps.append(TraceStep(index=2, kind="conclude", description=concl,
                           cites=_dedup(_cells(table, frontier, c1) + _cells(table, frontier, c2))))

    gold = GoldAnswer(label=f"{m1} vs {m2}", rows=sorted(frontier), metrics=[m1, m2])
    spec = {"type": "tradeoff_summary", "m1": m1, "m2": m2, "d1": d1, "d2": d2, "c1": c1, "c2": c2,
            "rows": list(rt.entity_rows)}
    return _rt_example(rt, QuestionType.TRADEOFF_SUMMARY, mode, question, gold, steps, edge, spec)


BUILDERS: dict[QuestionType, Callable[..., Optional[Example]]] = {
    QuestionType.BEST_UNDER_CONSTRAINT: build_best_under_constraint,
    QuestionType.THRESHOLD_FILTER: build_threshold_filter,
    QuestionType.TRADEOFF_SUMMARY: build_tradeoff_summary,
}


def build_extremum_from_rt(rt: Optional[RealTable], target: str, direction: Direction,
                           question: str, native_gold) -> Optional[Example]:
    """Materialize a single-column extremum as a validated Example with HUMAN
    gold (the trusted eval slice). ``target`` is a column of ``rt.table``; the
    answer is the row (entity) holding the extreme value. Orientation-agnostic:
    pass an across_columns RealTable for "which period" answers, or an
    across_rows one for "which line item" answers."""
    if rt is None or target not in rt.table.headers:
        return None
    table = rt.table
    col = table.col_index(target)
    rows = [r for r in rt.entity_rows if isinstance(table.cell(r, col), (int, float))]
    if len(rows) < 2:
        return None

    best_word = "highest" if direction == "higher" else "lowest"
    items = [(r, table.cell(r, col)) for r in rows]
    best = max(v for _, v in items) if direction == "higher" else min(v for _, v in items)
    winners = sorted(r for r, v in items if abs(v - best) <= 1e-9)

    steps = [
        TraceStep(index=0, kind="aggregate",
                  description=f"Read {target} across every {rt.entity_noun}: "
                              + ", ".join(f"{_name(table, r)} ({num(table.cell(r, col))})" for r in rows) + ".",
                  cites=_cells(table, rows, col)),
        TraceStep(index=1, kind="conclude",
                  description=f"The {best_word} {target} is {num(best)} "
                              + "(" + ", ".join(_name(table, r) for r in winners) + ").",
                  cites=_cells(table, winners, col)),
    ]
    gold = GoldAnswer(label=", ".join(_name(table, r) for r in winners), rows=winners)
    spec = {"type": "extremum", "target": target, "target_dir": direction,
            "rows": list(rt.entity_rows)}

    ex = _rt_example(rt, QuestionType.RANK_MODELS, "verified", question, gold, steps,
                     "argmax_tie" if len(winners) > 1 else None, spec)
    # Record the human provenance that makes this a *trusted* eval example.
    ex.metadata["verified_by"] = "tatqa_native_gold"
    ex.metadata["native_gold"] = native_gold
    ex.split = "eval"
    return ex


def build_extremum_example(it: IngestedTable, line_item: str, direction: Direction,
                           question: str, native_gold) -> Optional[Example]:
    """"In which year was <line item> largest?" - answer is a period, so build on
    the transposed table (rows = periods, columns = line items)."""
    return build_extremum_from_rt(transpose(it), line_item, direction, question, native_gold)


def build_extremum_example_rows(it: IngestedTable, period: str, direction: Direction,
                                question: str, native_gold) -> Optional[Example]:
    """"Which line item had the largest value in <period>?" - answer is a line
    item, so build on the table as-is (rows = line items, columns = periods)."""
    return build_extremum_from_rt(as_across_rows(it), period, direction, question, native_gold)


def orientations(it: IngestedTable) -> list[RealTable]:
    """Both queryable orientations for an ingested table (skips empty transposes)."""
    out = [as_across_rows(it)]
    tp = transpose(it)
    if tp is not None:
        out.append(tp)
    return out
