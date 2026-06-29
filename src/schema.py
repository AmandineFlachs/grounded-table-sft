"""Canonical data schema for grounded table-reasoning examples (Pydantic v2).

One ``Example`` = one table + one question + its grounded answer and reasoning trace.
Every field is machine-validated so malformed records cannot reach training/eval.

See ``docs/methodology.html`` §3 for the annotated rationale behind these shapes.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A table cell value. Kept deliberately narrow: numbers or short strings.
CellValue = Union[int, float, str]

ColumnType = Literal["numeric", "categorical", "text"]
StepKind = Literal["filter", "compare", "aggregate", "select", "conclude"]
TraceSource = Literal["programmatic", "llm"]
Split = Literal["train", "eval"]


class QuestionType(str, Enum):
    """The five eventual task types. v0 implements the first three."""

    BEST_UNDER_CONSTRAINT = "best_under_constraint"  # winner selection
    THRESHOLD_FILTER = "threshold_filter"            # constraint filtering
    TRADEOFF_SUMMARY = "tradeoff_summary"            # trade-off explanation
    COMPARE_ROWS = "compare_rows"                    # (deferred)
    RANK_MODELS = "rank_models"                      # (deferred)


class CellRef(BaseModel):
    """One cited table cell - the unit of evidence.

    Carries BOTH the numeric coordinates (for exact validation) and the
    semantic ``col_name`` + ``value`` (for human-readable, reorder-robust
    references). See methodology decision #2.
    """

    model_config = ConfigDict(extra="forbid")

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    col_name: str
    value: CellValue


class Table(BaseModel):
    """Normalized internal table representation."""

    model_config = ConfigDict(extra="forbid")

    headers: list[str]
    rows: list[list[CellValue]]
    row_labels: Optional[list[str]] = None
    column_types: list[ColumnType]

    @model_validator(mode="after")
    def _check_shape(self) -> "Table":
        n_cols = len(self.headers)
        if n_cols == 0:
            raise ValueError("table must have at least one column")
        if len(self.column_types) != n_cols:
            raise ValueError(
                f"column_types length ({len(self.column_types)}) != headers ({n_cols})"
            )
        for i, r in enumerate(self.rows):
            if len(r) != n_cols:
                raise ValueError(f"row {i} has {len(r)} cells, expected {n_cols}")
        if self.row_labels is not None and len(self.row_labels) != len(self.rows):
            raise ValueError("row_labels length must match number of rows")
        return self

    def cell(self, row: int, col: int) -> CellValue:
        return self.rows[row][col]

    def col_index(self, name: str) -> int:
        return self.headers.index(name)


class TraceStep(BaseModel):
    """One reasoning step: a natural-language ``description`` PLUS structured
    ``cites`` linking it to specific cells (methodology decision #1)."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0)
    kind: StepKind
    description: str
    cites: list[CellRef] = Field(default_factory=list)


class GoldAnswer(BaseModel):
    """Structured gold answer, general enough for all v0 task types.

    - winner selection : ``label`` = winning model, ``rows`` = [winner row]
    - constraint filter: ``label`` = matched names (or "none"), ``rows`` = matches
    - trade-off        : ``label`` = "<m1> vs <m2>", ``rows`` = Pareto frontier,
                         ``metrics`` = the two conflicting metric column names
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    rows: list[int] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class Example(BaseModel):
    """The full training/eval record."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    domain: str
    table: Table
    question: str
    question_type: QuestionType
    gold_answer: GoldAnswer
    trace_steps: list[TraceStep]
    evidence_cells: list[CellRef] = Field(default_factory=list)
    trace_source: TraceSource = "programmatic"
    split: Split = "train"
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _trace_not_empty(self) -> "Example":
        if not self.trace_steps:
            raise ValueError("trace_steps must not be empty")
        if self.trace_steps[-1].kind != "conclude":
            raise ValueError("the final trace step must be of kind 'conclude'")
        return self
