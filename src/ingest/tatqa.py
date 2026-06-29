"""TAT-QA -> canonical ``Table`` ingestion (Phase 2, P2.2).

TAT-QA (Zhu et al., ACL 2021; CC BY 4.0) ships each context as
``{table: {uid, table: List[List[str]]}, paragraphs, questions}`` where the
grid is a 2-D array of raw *strings*: currency/percent/comma-formatted numbers,
empty cells, multi-row headers, and merged super-headers represented as the
value sitting in one column with the spanned columns left blank.

This module turns that messy grid into our schema, mirroring the synthetic
generator's conventions so downstream stages are source-agnostic:
  * column 0 is the label column (forced ``categorical``);
  * remaining numeric columns are the "metrics";
  * ``row_labels`` is left ``None`` (the label column serves that role).

The P2.1 probe established the realities this code is built around: col 0 is a
reliable label column (~98.6%), the header band is 0-4 leading rows (usually 2),
and ~87% of non-empty body cells are numeric once symbols are stripped.

Strategy (A): we ingest the *tables* (filtered to ``answer_from == "table"``
contexts) and generate our own selection questions over them later (P2.3); the
native arithmetic questions are deferred to Phase 3.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..schema import CellValue, Table
from ..table_utils import infer_column_types

# Attribution carried into every ingested table's metadata (CC BY 4.0 duty).
TATQA_ATTRIBUTION = (
    "TAT-QA (Zhu et al., ACL 2021), CC BY 4.0 - "
    "https://github.com/NExTplusplus/TAT-QA"
)

# A cell is numeric if, after stripping $ , % and whitespace, what's left is a
# (optionally parenthesised-negative) number.
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
# Missing-value markers: lone dashes (incl. unicode), the U+FFFD mojibake that
# em-dashes decode to in this dataset, and n/a-style tokens.
_MISSING_CORE = re.compile(r"^[\-‒–—―�\s*]*$")
_MISSING_WORD = re.compile(r"^(n/?a|n\.a\.?|nm|nil|--+)$", re.I)


def parse_cell(raw: str) -> CellValue:
    """Parse one raw TAT-QA cell into a number when possible, else the string.

    Handles ``$``, thousands commas, ``%`` and accounting negatives ``(123)``.
    Missing-value markers (``-``, ``—``, ``- - %``, ``n/a``, mojibake ``�``) and
    empty/whitespace cells normalize to ``""`` so they don't poison numeric
    column-type inference.
    """
    s = (raw or "").strip()
    if s == "":
        return ""
    # Strip currency/percent first, so accounting negatives written as "$ (2,235)"
    # (dollar before the paren) are still detected as negative.
    core = s.replace("$", "").replace("%", "").strip()
    if _MISSING_CORE.match(core) or _MISSING_WORD.match(core):
        return ""
    neg = core.startswith("(") and core.endswith(")")
    body = (core[1:-1] if neg else core).replace(",", "").strip()
    if _NUM_RE.match(body):
        val = float(body)
        if neg:
            val = -val
        # Keep ints as ints for clean rendering/citation.
        return int(val) if val.is_integer() else val
    return s


def _is_numeric(v: CellValue) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_YEAR_PREFIX = re.compile(r"^(19|20)\d{2}\b")


def _is_year_int(v: CellValue) -> bool:
    """A bare 4-digit year (1900-2099) parsed as a number. Years parse as numbers
    but are *labels*, not data magnitudes - distinguishing them is what lets us
    catch sub-header rows (e.g. ``2019 | 2018 | Actual | Comp.``) that leaked into
    the body."""
    return _is_numeric(v) and float(v).is_integer() and 1900 <= v <= 2099


def _is_year_header(row: list[str]) -> bool:
    """True if every non-empty value cell (cols 1+) *starts with* a 4-digit year
    (e.g. ``2019``, ``2018 (4)``, ``2019 £m``) - a header band that happens to
    be numeric. Catches the minority of tables whose header row carries a
    non-empty label/units-note in col 0."""
    vals = [c.strip() for c in row[1:] if (c or "").strip() != ""]
    if not vals:
        return False
    return all(_YEAR_PREFIX.match(c) for c in vals)


def split_header_body(grid: list[list[str]]) -> tuple[list[list[str]], list[list[str]], int]:
    """Split a raw grid into (header_rows, body_rows, data_start_index).

    Primary signal (P2.1 probe): TAT-QA *data* rows carry a non-empty label in
    col 0, while header rows (years, units notes, merged super-headers) leave
    col 0 blank. So data starts at the first row with a non-empty col 0. A
    year-header guard then folds in the residual case where the header row's
    col 0 is itself a label (e.g. "Metric | 2019 | 2018").
    """
    n = len(grid)
    start = n
    for i, row in enumerate(grid):
        label = (row[0] if row else "").strip()
        has_values = any((c or "").strip() != "" for c in row[1:])
        # A data row needs both a label AND at least one value; a title/section
        # row (label, empty values) belongs to the header band, not the body.
        if label != "" and has_values:
            start = i
            break
    # Header guard: fold leading "header-like" rows whose col 0 is itself a
    # label/units-note into the header band - numeric year rows (2019, 2018) and
    # text/date rows (April 27, 2019; 2019 £m) that sit above the numeric data.
    while start < n:
        later_numeric = any(_row_has_numeric(grid[j]) for j in range(start + 1, n))
        if _looks_like_header(grid[start], later_numeric):
            start += 1
        else:
            break
    return grid[:start], grid[start:], start


def _row_has_numeric(row: list[str]) -> bool:
    return any(_is_numeric(parse_cell(c)) for c in row[1:])


def _looks_like_header(row: list[str], later_numeric: bool) -> bool:
    """A row is header-like if it's a year header (numeric years), or it carries
    NO real data magnitudes - only years and/or text - while a later row carries
    the numeric data, or its value cells are mostly non-numeric."""
    vals = [c for c in row[1:] if (c or "").strip() != ""]
    if not vals:
        return False
    if _is_year_header(row):
        return True
    parsed = [parse_cell(c) for c in vals]
    # A sub-header that leaked into the body (e.g. "(in millions) | 2019 | 2018 |
    # Actual | Comp.") has zero real magnitudes - only years (which parse as
    # numbers) and header text - with real data below. The plain numeric-fraction
    # test misses it because years count as numeric.
    real_mag = sum(1 for v in parsed if _is_numeric(v) and not _is_year_int(v))
    num_years = sum(1 for v in parsed if _is_year_int(v))
    num_nonnum = sum(1 for v in parsed if not _is_numeric(v))
    # No real magnitudes, only years/text, with real data below. Require >=2 years
    # (or a year + header text) so a single value that merely falls in 1900-2099
    # (e.g. an amount of 2047) doesn't trip it.
    if real_mag == 0 and later_numeric and (num_years >= 2 or (num_years >= 1 and num_nonnum >= 1)):
        return True
    numeric = sum(1 for v in parsed if _is_numeric(v))
    return numeric / len(vals) < 0.5 and later_numeric


def _compose_headers(header_rows: list[list[str]], n_cols: int) -> tuple[list[str], list[str]]:
    """Build one header per column from the header band.

    A header row that fills <=1 of its value columns is a *merged super-header*
    (e.g. "Years Ended September 30," spanning the year columns); it is set
    aside (returned separately) rather than smeared across a single column.
    Per-column names come from the remaining "dense" header rows, joined.
    """
    super_headers: list[str] = []
    dense_rows: list[list[str]] = []
    for row in header_rows:
        value_cells = [c for c in row[1:] if (c or "").strip() != ""]
        if len(value_cells) <= 1 and len(row) > 2:
            # Spanning/super-header (or a stray note) - keep for context only.
            super_headers.extend(c.strip() for c in row if (c or "").strip())
        else:
            dense_rows.append(row)

    headers: list[str] = []
    for c in range(n_cols):
        parts = [
            r[c].strip()
            for r in dense_rows
            if c < len(r) and (r[c] or "").strip() != ""
        ]
        headers.append(" ".join(parts))

    # Ensure non-empty, unique headers (col 0 often blank in the source).
    seen: dict[str, int] = {}
    out: list[str] = []
    for c, h in enumerate(headers):
        name = h if h else ("category" if c == 0 else f"col_{c}")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out, super_headers


@dataclass
class IngestedTable:
    """An ingested real table plus the semantics downstream stages rely on.

    Mirrors ``synthetic_generator.GeneratedTable`` so question generation (P2.3)
    is source-agnostic. ``directions`` is intentionally empty: real financial
    tables carry no inherent higher/lower-better semantics, so the question
    layer assigns direction per question (a P2.3 concern).
    """

    table: Table
    table_id: str
    source_uid: str
    name_col: str
    metric_cols: list[str]
    confidence: str  # "high" | "low"
    notes: list[str] = field(default_factory=list)
    super_headers: list[str] = field(default_factory=list)
    directions: dict[str, str] = field(default_factory=dict)


def ingest_context(ctx: dict, table_index: int) -> IngestedTable:
    """Convert one raw TAT-QA context into an ``IngestedTable``."""
    raw = ctx["table"]
    uid = raw.get("uid", f"ctx{table_index}")
    grid = raw["table"]
    n_cols = max((len(r) for r in grid), default=0)
    notes: list[str] = []

    header_rows, body_rows, _ = split_header_body(grid)
    headers, super_headers = _compose_headers(header_rows, n_cols)
    if not header_rows:
        notes.append("no header rows detected; synthesized column names")

    # Parse body rows, rectangularising ragged rows and dropping section rows
    # (a label with all value cells empty carries no data for our task).
    rows: list[list[CellValue]] = []
    row_labels: list[str] = []
    dropped_sections = 0
    ragged = 0
    for r in body_rows:
        if len(r) != n_cols:
            ragged += 1
            r = (list(r) + [""] * n_cols)[:n_cols]
        parsed = [parse_cell(c) for c in r]
        label_empty = str(parsed[0]).strip() == ""
        has_real = any(_is_numeric(v) and not _is_year_int(v) for v in parsed[1:])
        # Drop blank rows, label-less section sub-headers (no numeric values, e.g.
        # ['', 'Consolidated ... Data:', '', '']), and label-less rows carrying only
        # YEARS - a leaked mid-table period header, e.g. ['', 2018, 2018, 2018].
        # (Year-only label-less rows have no real magnitude; a real total row like
        # ['', 785.3, 826] does, so it is kept.)
        if all(v == "" for v in parsed[1:]) or (label_empty and not has_real):
            dropped_sections += 1
            continue
        rows.append(parsed)
        row_labels.append(str(parsed[0]))
    if ragged:
        notes.append(f"{ragged} ragged row(s) padded/truncated to {n_cols} cols")
    if dropped_sections:
        notes.append(f"{dropped_sections} section/blank row(s) dropped")

    # Disambiguate duplicate row labels: multi-section tables flatten to repeated
    # labels (e.g. two "Leasehold" rows, or one line item per period block), which
    # makes a cited/answered row ambiguous. Suffix the 2nd+ occurrence so every
    # entity is uniquely identifiable. (Cosmetic for unique tables; a real fix for
    # the rest.)
    seen_lbl: dict[str, int] = {}
    disambiguated = 0
    for i, row in enumerate(rows):
        key = str(row[0]).strip()
        if key == "":
            continue
        seen_lbl[key] = seen_lbl.get(key, 0) + 1
        if seen_lbl[key] > 1:
            new = f"{row[0]} ({seen_lbl[key]})"
            rows[i][0] = new
            row_labels[i] = new
            disambiguated += 1
    if disambiguated:
        notes.append(f"{disambiguated} duplicate row-label(s) disambiguated")

    column_types = infer_column_types(headers, rows)
    if column_types:
        column_types[0] = "categorical"  # col 0 is the label column by convention

    table = Table(headers=headers, rows=rows, column_types=column_types)

    metric_cols = [h for c, h in enumerate(headers) if column_types[c] == "numeric"]

    # Confidence: we trust the parse when there's a real header, a real body,
    # at least one numeric column, and no structural surprises.
    confidence = "high"
    if not header_rows or len(rows) < 2 or not metric_cols or ragged:
        confidence = "low"

    return IngestedTable(
        table=table,
        table_id=f"tatqa_{table_index:04d}",
        source_uid=uid,
        name_col=headers[0] if headers else "category",
        metric_cols=metric_cols,
        confidence=confidence,
        notes=notes,
        super_headers=super_headers,
    )


def load_contexts(path: str | Path) -> list[dict]:
    """Load a raw TAT-QA split file (a JSON list of contexts)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table_only_contexts(contexts: Iterable[dict]) -> list[dict]:
    """Keep contexts that have at least one ``answer_from == "table"`` question.

    Under strategy (A) we generate our own questions, so we only require that
    the *table* is one humans answered from the table alone (a cleanliness proxy).
    """
    out = []
    for c in contexts:
        if any(q.get("answer_from") == "table" for q in c.get("questions", [])):
            out.append(c)
    return out


def ingest_file(
    path: str | Path,
    *,
    table_only: bool = True,
    limit: Optional[int] = None,
) -> list[IngestedTable]:
    """Ingest a whole TAT-QA split into ``IngestedTable`` objects."""
    contexts = load_contexts(path)
    if table_only:
        contexts = table_only_contexts(contexts)
    if limit is not None:
        contexts = contexts[:limit]
    return [ingest_context(c, i) for i, c in enumerate(contexts)]
