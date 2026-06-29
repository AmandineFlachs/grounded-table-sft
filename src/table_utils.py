"""Table helpers: column-type inference, markdown rendering, cell-ref construction."""

from __future__ import annotations

import re
from typing import Sequence

from .schema import CellRef, CellValue, ColumnType, Table


def infer_column_types(headers: Sequence[str], rows: Sequence[Sequence[CellValue]]) -> list[ColumnType]:
    """Infer a type per column.

    - ``numeric``     : every non-empty cell is an int/float (and not bool).
    - ``categorical`` : string column with few distinct values relative to rows.
    - ``text``        : string column with high cardinality.
    """
    n_cols = len(headers)
    types: list[ColumnType] = []
    n_rows = len(rows)
    for c in range(n_cols):
        col_vals = [rows[r][c] for r in range(n_rows)]
        non_null = [v for v in col_vals if v is not None and v != ""]
        if non_null and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            types.append("numeric")
            continue
        distinct = len({str(v) for v in non_null})
        # Heuristic: a string column is "categorical" if it repeats values.
        if non_null and distinct <= max(2, len(non_null) // 2):
            types.append("categorical")
        else:
            types.append("text")
    return types


def make_cell_ref(table: Table, row: int, col: int) -> CellRef:
    """Build a fully-populated CellRef (index + name + value) for one cell."""
    return CellRef(
        row=row,
        col=col,
        col_name=table.headers[col],
        value=table.cell(row, col),
    )


def row_label(table: Table, row: int) -> str:
    """Human-facing label for a row: explicit row_label, else first categorical
    column value, else ``row <i>``."""
    if table.row_labels is not None:
        return table.row_labels[row]
    for c, t in enumerate(table.column_types):
        if t == "categorical":
            return str(table.cell(row, c))
    return f"row {row}"


def numeric_columns(table: Table) -> list[int]:
    return [c for c, t in enumerate(table.column_types) if t == "numeric"]


# Total/subtotal rows trivially win an argmax / dominate a frontier ("pick Total"), so
# they are excluded from the entity universe (methodology log j). This is the SINGLE
# definition shared by question generation (realtable_questions) and the executor's
# eval-time universe, so the inference-time universe matches the gold spec["rows"].
_TOTAL_RE = re.compile(r"\b(total|subtotal|aggregate)\b", re.I)


def is_total(label) -> bool:
    return bool(_TOTAL_RE.search(str(label)))


def entity_universe(table: Table) -> list[int]:
    """Entity rows a question ranges over: rows whose column-0 label is not a
    total/subtotal/aggregate. Same rule (col 0 of the presented table) used in both
    orientations at generation time, so this reproduces spec['rows'] at inference."""
    return [r for r in range(len(table.rows)) if not is_total(str(table.cell(r, 0)))]


def render_markdown(table: Table) -> str:
    """Render the table as GitHub-flavored markdown (what the model sees)."""
    headers = list(table.headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in table.rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _fmt(v: CellValue) -> str:
    if isinstance(v, float):
        # Trim trailing zeros but keep it readable.
        return f"{v:g}"
    return str(v)
