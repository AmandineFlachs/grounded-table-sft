"""Deterministic answer engine - "answer by construction" (P3.6).

Given a table and a structured OPERATION (op type + column *names* + thresholds +
directions + the entity-row universe), compute the answer arithmetically. The model's
job becomes COMPREHENSION (which op / columns / threshold / which rows are entities);
the arithmetic - max/min/threshold/dominance, the P3.5 error class - is executed by
this proven code, extending "grounded by construction" to "answer by construction".

DESIGN: this engine is INDEPENDENT of ``trace_validator`` on purpose (its own predicate
re-implementations, no shared helpers). The validator must remain a separate check on
this engine's output - sharing code would make "executor output passes validator"
circular. Same philosophy as the validator's independence from the generator.

The OPERATION dict uses column NAMES (not indices) so it is robust to reordering and is
exactly what a model can emit by reading the question:
  best_under_constraint : {type, target, target_dir(higher|lower), constraint, op(lt|gte), threshold, rows?}
  threshold_filter      : {type, conditions:[{metric, op(lt|gte), T}], rows?}
  tradeoff_summary      : {type, m1, m2, d1(higher|lower), d2, rows?}
  extremum              : {type, target, target_dir(higher|lower), rows?}   # OOD verified slice
``rows`` (optional) is the entity universe (e.g. excluding total/subtotal rows); absent => all rows.
"""
from __future__ import annotations

from .schema import CellRef, GoldAnswer, Table

EPS = 1e-9

_OPS = {"lt", "gte"}
_DIRS = {"higher", "lower"}


def _canon_op(op: str) -> str:
    """Canonicalize the threshold operator to the prompt alphabet and REJECT anything
    else - a malformed op must score an honest 0, never a lucky default-to-gte pass."""
    op = "gte" if op == "ge" else op
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r} (expected one of {sorted(_OPS)})")
    return op


def _canon_dir(d: str) -> str:
    if d not in _DIRS:
        raise ValueError(f"unknown direction {d!r} (expected one of {sorted(_DIRS)})")
    return d


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _passes(value, op: str, threshold: float) -> bool:
    # op is pre-canonicalized to {lt, gte} by the caller.
    return _is_num(value) and (value < threshold if op == "lt" else value >= threshold)


def _universe(table: Table, op: dict) -> list[int]:
    n = len(table.rows)
    rows = op.get("rows")
    if rows is None:
        return list(range(n))
    return [r for r in rows if 0 <= r < n]


def _label_for(table: Table, rows: list[int]) -> str:
    """Human-readable label from the row-label column ('none' when empty)."""
    if not rows:
        return "none"
    src = table.row_labels if table.row_labels else [r[0] for r in table.rows]
    return ", ".join(str(src[r]) for r in rows)


def _optimum(table: Table, rows: list[int], col: int, direction: str) -> list[int]:
    vals = [(r, table.cell(r, col)) for r in rows]
    best = max(v for _, v in vals) if direction == "higher" else min(v for _, v in vals)
    return sorted(r for r, v in vals if abs(v - best) <= EPS)


def _frontier(table: Table, c1: int, c2: int, d1: str, d2: str, universe: list[int]) -> list[int]:
    def obj(v, d):
        return v if d == "higher" else -v

    elig = [r for r in universe if _is_num(table.cell(r, c1)) and _is_num(table.cell(r, c2))]
    pts = [(r, obj(table.cell(r, c1), d1), obj(table.cell(r, c2), d2)) for r in elig]
    out = []
    for ri, o1, o2 in pts:
        dominated = any(
            rj != ri and oj1 >= o1 - EPS and oj2 >= o2 - EPS and (oj1 > o1 + EPS or oj2 > o2 + EPS)
            for rj, oj1, oj2 in pts
        )
        if not dominated:
            out.append(ri)
    return sorted(out)


def compute_answer(table: Table, op: dict) -> GoldAnswer:
    """Compute the gold answer for ``op`` over ``table``. Raises on an unknown op type or
    a column name not in the table (a real model error we want surfaced, not hidden)."""
    t = op["type"]
    universe = _universe(table, op)

    if t == "best_under_constraint":
        cop, td = _canon_op(op["op"]), _canon_dir(op["target_dir"])
        tc = table.col_index(op["target"])
        cc = table.col_index(op["constraint"])
        survivors = [
            r for r in universe
            if _is_num(table.cell(r, tc)) and _passes(table.cell(r, cc), cop, op["threshold"])
        ]
        rows = _optimum(table, survivors, tc, td) if survivors else []
        return GoldAnswer(label=_label_for(table, rows), rows=rows, metrics=[])

    if t == "threshold_filter":
        conds = [{"metric": c["metric"], "op": _canon_op(c["op"]), "T": c["T"]} for c in op["conditions"]]

        def ok(r: int) -> bool:
            return all(_passes(table.cell(r, table.col_index(c["metric"])), c["op"], c["T"]) for c in conds)

        rows = sorted(r for r in universe if ok(r))
        return GoldAnswer(label=_label_for(table, rows), rows=rows, metrics=[])

    if t == "tradeoff_summary":
        d1, d2 = _canon_dir(op["d1"]), _canon_dir(op["d2"])
        c1, c2 = table.col_index(op["m1"]), table.col_index(op["m2"])
        rows = _frontier(table, c1, c2, d1, d2, universe)
        return GoldAnswer(label=f"{op['m1']} vs {op['m2']}", rows=rows, metrics=[op["m1"], op["m2"]])

    if t == "extremum":
        td = _canon_dir(op["target_dir"])
        tc = table.col_index(op["target"])
        cand = [r for r in universe if _is_num(table.cell(r, tc))]
        rows = _optimum(table, cand, tc, td) if cand else []
        return GoldAnswer(label=_label_for(table, rows), rows=rows, metrics=[])

    raise ValueError(f"unknown op type {t!r}")


# --------------------------------------------------------------------------- #
# evidence by construction (P3.10 / Step A)
# --------------------------------------------------------------------------- #
def _cellref(table: Table, row: int, col_name: str) -> CellRef:
    col = table.col_index(col_name)
    return CellRef(row=row, col=col, col_name=col_name, value=table.cell(row, col))


def evidence_for(table: Table, op: dict) -> list[CellRef]:
    """The cells the engine actually READ to justify its answer - grounded by
    construction (every CellRef is read straight from the table, so it always exists
    and matches). Covers the question's required metric columns for the answer rows,
    and for an empty/filter answer the tested cells of the entity universe (the
    justification that nothing qualifies). This is the trustworthy replacement for the
    model's hand-typed citations, mirroring how the executor replaced its arithmetic."""
    t = op["type"]
    ans = compute_answer(table, op)
    rows = ans.rows
    universe = _universe(table, op)
    out: list[CellRef] = []
    seen: set[tuple[int, str]] = set()

    def add(r: int, name: str) -> None:
        key = (r, name)
        if key in seen or not (0 <= r < len(table.rows)):
            return
        seen.add(key)
        try:
            out.append(_cellref(table, r, name))
        except (ValueError, IndexError):
            pass

    if t == "best_under_constraint":
        if rows:
            for r in rows:
                add(r, op["target"]); add(r, op["constraint"])
        else:
            for r in universe:
                add(r, op["constraint"])
    elif t == "threshold_filter":
        metrics = [c["metric"] for c in op["conditions"]]
        for r in (rows or universe):
            for m in metrics:
                add(r, m)
    elif t == "tradeoff_summary":
        for r in rows:
            add(r, op["m1"]); add(r, op["m2"])
    elif t == "extremum":
        for r in (rows or universe):
            add(r, op["target"])
    return out
