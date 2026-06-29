"""Programmatic question + gold-answer + grounded-trace generation.

Design rule: **gold answers are always derived by applying the predicate to the
data**, never hand-asserted. The validator independently recomputes the same
quantities, so a programmatic example is grounded and correct by construction.

Edge-case ``mode`` only influences how thresholds are *chosen* (to force empty
results, near-threshold boundaries, or argmax ties); the answer is still derived.
"""

from __future__ import annotations

import random
from typing import Literal, Optional

from .schema import CellRef, Example, GoldAnswer, QuestionType, TraceStep
from .synthetic_generator import Direction, GeneratedTable
from .table_utils import _fmt as num
from .table_utils import make_cell_ref

EPS = 1e-9
Mode = Literal["normal", "empty", "near_threshold", "tie"]


# --------------------------------------------------------------------------- #
# small numeric / predicate helpers
# --------------------------------------------------------------------------- #
def _op_for(direction: Direction) -> str:
    return "lt" if direction == "lower" else "ge"


def _passes(value: float, op: str, threshold: float) -> bool:
    return value < threshold if op == "lt" else value >= threshold


def _phrase(metric: str, direction: Direction, threshold: float) -> str:
    if direction == "lower":
        return f"{metric} under {num(threshold)}"
    return f"{metric} of at least {num(threshold)}"


def _choose_threshold(values: list[float], direction: Direction, mode: Mode,
                      rng: random.Random) -> tuple[float, str]:
    """Pick a threshold T (and comparison op) for a constraint on this column."""
    op = _op_for(direction)
    uniq = sorted(set(values))
    spread = (max(values) - min(values)) or 1.0
    margin = max(spread * 0.1, 0.01)
    small = max(spread * 0.04, 0.005)

    if mode == "empty":
        T = min(values) - margin if direction == "lower" else max(values) + margin
    elif mode == "near_threshold":
        bv = rng.choice(uniq)
        T = bv + small if direction == "lower" else bv - small
    else:  # normal subset (also the fallback for "tie")
        if len(uniq) >= 2:
            k = rng.randint(1, len(uniq) - 1)
            T = (uniq[k - 1] + uniq[k]) / 2.0
        else:
            T = (min(values) + margin) if direction == "lower" else (max(values) - margin)

    T = round(T, 6)
    while any(abs(T - v) <= EPS for v in values):  # keep T off any exact cell value
        T = round(T + 1e-6, 6)
    return T, op


def _optimum(rows: list[int], col: int, table, direction: Direction) -> tuple[list[int], float]:
    items = [(r, table.cell(r, col)) for r in rows]
    best = max(v for _, v in items) if direction == "higher" else min(v for _, v in items)
    winners = [r for r, v in items if abs(v - best) <= EPS]
    return winners, best


def _pareto_frontier(table, c1: int, c2: int, d1: Direction, d2: Direction) -> list[int]:
    def obj(v: float, d: Direction) -> float:
        return v if d == "higher" else -v

    pts = [(r, obj(table.cell(r, c1), d1), obj(table.cell(r, c2), d2)) for r in range(len(table.rows))]
    frontier = []
    for ri, oi1, oi2 in pts:
        dominated = any(
            rj != ri and oj1 >= oi1 - EPS and oj2 >= oi2 - EPS and (oj1 > oi1 + EPS or oj2 > oi2 + EPS)
            for rj, oj1, oj2 in pts
        )
        if not dominated:
            frontier.append(ri)
    return frontier


def _name(table, row: int) -> str:
    return str(table.cell(row, 0))


def _cells(table, rows: list[int], col: int) -> list[CellRef]:
    return [make_cell_ref(table, r, col) for r in rows]


def _dedup(cells: list[CellRef]) -> list[CellRef]:
    seen, out = set(), []
    for c in cells:
        key = (c.row, c.col)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _example(gen: GeneratedTable, qtype: QuestionType, suffix: str, question: str,
             gold: GoldAnswer, steps: list[TraceStep], edge: Optional[str],
             spec: dict) -> Example:
    evidence = _dedup([c for s in steps for c in s.cites])
    difficulty = "hard" if edge else "easy"
    return Example(
        table_id=gen.table_id,
        domain="ml_benchmark",
        table=gen.table,
        question=question,
        question_type=qtype,
        gold_answer=gold,
        trace_steps=steps,
        evidence_cells=evidence,
        trace_source="programmatic",
        metadata={
            "difficulty": difficulty,
            "edge_cases": [edge] if edge else [],
            "example_id": f"{gen.table_id}_{qtype.value}_{suffix}",
            # Structured question definition - lets the validator independently
            # recompute the answer from the table rather than parsing prose.
            "spec": spec,
        },
    )


# --------------------------------------------------------------------------- #
# 1. winner selection - best_under_constraint
# --------------------------------------------------------------------------- #
def build_best_under_constraint(gen: GeneratedTable, rng: random.Random,
                                mode: Mode = "normal") -> Optional[Example]:
    table = gen.table
    if len(gen.metric_cols) < 2:
        return None
    target, constraint = rng.sample(gen.metric_cols, 2)
    td, cd = gen.directions[target], gen.directions[constraint]
    col_t, col_c = table.col_index(target), table.col_index(constraint)
    rows = list(range(len(table.rows)))

    thr_mode: Mode = "normal" if mode == "tie" else mode
    T, op_c = _choose_threshold([table.cell(r, col_c) for r in rows], cd, thr_mode, rng)
    survivors = [r for r in rows if _passes(table.cell(r, col_c), op_c, T)]
    spec = {"type": "best_under_constraint", "target": target, "target_dir": td,
            "constraint": constraint, "constraint_dir": cd, "op": op_c, "threshold": T}

    # Optionally force an argmax tie among survivors on the target metric.
    edge: Optional[str] = None
    if mode == "tie" and len(survivors) >= 2:
        _, best_val = _optimum(survivors, col_t, table, td)
        other = next(r for r in survivors if table.cell(r, col_t) != best_val) \
            if any(table.cell(r, col_t) != best_val for r in survivors) else survivors[1]
        table.rows[other][col_t] = best_val  # mutate our own synthetic data

    best_word = "highest" if td == "higher" else "lowest"
    question = (f"Which model has the {best_word} {target} while keeping "
                f"{_phrase(constraint, cd, T)}?")

    steps: list[TraceStep] = []
    steps.append(TraceStep(
        index=0, kind="filter",
        description=("Apply the constraint " + _phrase(constraint, cd, T) + ": "
                     + (", ".join(f"{_name(table, r)} ({num(table.cell(r, col_c))})" for r in survivors)
                        if survivors else "no model qualifies") + "."),
        cites=_cells(table, rows, col_c),
    ))

    if not survivors:
        steps.append(TraceStep(
            index=1, kind="conclude",
            description=f"No model satisfies {_phrase(constraint, cd, T)}, so there is no answer.",
            cites=_cells(table, rows, col_c),
        ))
        gold = GoldAnswer(label="none", rows=[])
        edge = "empty"
        return _example(gen, QuestionType.BEST_UNDER_CONSTRAINT, mode, question, gold, steps, edge, spec)

    winners, best_val = _optimum(survivors, col_t, table, td)
    steps.append(TraceStep(
        index=1, kind="select",
        description=(f"Among the qualifying models, the {best_word} {target} is "
                     f"{num(best_val)} ("
                     + ", ".join(_name(table, r) for r in winners) + ")."),
        cites=_cells(table, survivors, col_t),
    ))
    if len(winners) == 1:
        label = _name(table, winners[0])
        concl = f"{label} is the answer: {best_word} {target} ({num(best_val)}) among qualifying models."
    else:
        label = "tie: " + ", ".join(_name(table, r) for r in winners)
        concl = ("There is a tie for the " + best_word + f" {target} ({num(best_val)}): "
                 + ", ".join(_name(table, r) for r in winners) + ".")
        edge = "argmax_tie"
    steps.append(TraceStep(
        index=2, kind="conclude", description=concl,
        cites=_dedup(_cells(table, winners, col_t) + _cells(table, winners, col_c)),
    ))
    if edge is None and mode == "near_threshold":
        edge = "near_threshold"
    gold = GoldAnswer(label=label, rows=sorted(winners))
    return _example(gen, QuestionType.BEST_UNDER_CONSTRAINT, mode, question, gold, steps, edge, spec)


# --------------------------------------------------------------------------- #
# 2. constraint filtering - threshold_filter
# --------------------------------------------------------------------------- #
def build_threshold_filter(gen: GeneratedTable, rng: random.Random,
                           mode: Mode = "normal") -> Optional[Example]:
    table = gen.table
    if not gen.metric_cols:
        return None
    k = min(len(gen.metric_cols), rng.choice([1, 2]))
    metrics = rng.sample(gen.metric_cols, k)
    rows = list(range(len(table.rows)))

    conds = []
    for idx, m in enumerate(metrics):
        d = gen.directions[m]
        col = table.col_index(m)
        cmode: Mode = mode if idx == 0 else "normal"
        T, op = _choose_threshold([table.cell(r, col) for r in rows], d, cmode, rng)
        conds.append({"metric": m, "col": col, "dir": d, "op": op, "T": T})

    matches = [r for r in rows
               if all(_passes(table.cell(r, c["col"]), c["op"], c["T"]) for c in conds)]

    cond_text = " and ".join(_phrase(c["metric"], c["dir"], c["T"]) for c in conds)
    question = f"Which models have {cond_text}?"

    steps: list[TraceStep] = []
    for i, c in enumerate(conds):
        passing = [r for r in rows if _passes(table.cell(r, c["col"]), c["op"], c["T"])]
        steps.append(TraceStep(
            index=i, kind="filter",
            description=(f"Condition {_phrase(c['metric'], c['dir'], c['T'])} keeps: "
                        + (", ".join(_name(table, r) for r in passing) if passing else "no model") + "."),
            cites=_cells(table, rows, c["col"]),
        ))

    if matches:
        label = ", ".join(_name(table, r) for r in matches)
        concl = "Models satisfying all conditions: " + label + "."
    else:
        label = "none"
        concl = "No model satisfies all conditions."
    concl_cells = _dedup([cell for c in conds for cell in _cells(table, matches, c["col"])])
    steps.append(TraceStep(index=len(conds), kind="conclude", description=concl, cites=concl_cells))

    edge = "empty" if not matches else ("near_threshold" if mode == "near_threshold" else None)
    gold = GoldAnswer(label=label, rows=sorted(matches))
    spec = {"type": "threshold_filter",
            "conditions": [{"metric": c["metric"], "dir": c["dir"], "op": c["op"],
                            "T": c["T"], "col": c["col"]} for c in conds]}
    return _example(gen, QuestionType.THRESHOLD_FILTER, mode, question, gold, steps, edge, spec)


# --------------------------------------------------------------------------- #
# 3. trade-off explanation - tradeoff_summary (Pareto frontier)
# --------------------------------------------------------------------------- #
def build_tradeoff_summary(gen: GeneratedTable, rng: random.Random,
                           mode: Mode = "normal") -> Optional[Example]:
    table = gen.table
    if len(gen.metric_cols) < 2:
        return None
    m1, m2 = rng.sample(gen.metric_cols, 2)
    d1, d2 = gen.directions[m1], gen.directions[m2]
    c1, c2 = table.col_index(m1), table.col_index(m2)
    rows = list(range(len(table.rows)))

    frontier = _pareto_frontier(table, c1, c2, d1, d2)
    question = f"What is the trade-off between {m1} and {m2} across these models?"

    def pair(r: int) -> str:
        return f"{_name(table, r)} ({m1}={num(table.cell(r, c1))}, {m2}={num(table.cell(r, c2))})"

    steps = [
        TraceStep(index=0, kind="aggregate",
                  description=f"Compare {m1} ({d1} is better) and {m2} ({d2} is better) for every model.",
                  cites=_dedup(_cells(table, rows, c1) + _cells(table, rows, c2))),
        TraceStep(index=1, kind="select",
                  description=("Non-dominated (Pareto-optimal) models - no other model is at least "
                               f"as good in both {m1} and {m2} and strictly better in one: "
                               + "; ".join(pair(r) for r in frontier) + "."),
                  cites=_dedup(_cells(table, frontier, c1) + _cells(table, frontier, c2))),
    ]
    dominated = [r for r in rows if r not in frontier]
    if len(frontier) == 1:
        concl = (f"{_name(table, frontier[0])} dominates on both metrics, so there is no real "
                 f"trade-off here.")
        edge = "single_dominator"
    elif not dominated:
        concl = (f"Every model is Pareto-optimal: each trades {m1} against {m2} with no dominated option.")
        edge = "no_domination"
    else:
        concl = (f"The trade-off lies on the frontier ({', '.join(_name(table, r) for r in frontier)}): "
                 f"gaining {m1} costs {m2} and vice-versa; dominated models "
                 f"({', '.join(_name(table, r) for r in dominated)}) are worse on both.")
        edge = None
    steps.append(TraceStep(index=2, kind="conclude", description=concl,
                           cites=_dedup(_cells(table, frontier, c1) + _cells(table, frontier, c2))))

    gold = GoldAnswer(label=f"{m1} vs {m2}", rows=sorted(frontier), metrics=[m1, m2])
    spec = {"type": "tradeoff_summary", "m1": m1, "m2": m2, "d1": d1, "d2": d2, "c1": c1, "c2": c2}
    return _example(gen, QuestionType.TRADEOFF_SUMMARY, mode, question, gold, steps, edge, spec)


BUILDERS = {
    QuestionType.BEST_UNDER_CONSTRAINT: build_best_under_constraint,
    QuestionType.THRESHOLD_FILTER: build_threshold_filter,
    QuestionType.TRADEOFF_SUMMARY: build_tradeoff_summary,
}
