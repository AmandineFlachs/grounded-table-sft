"""Seeded synthetic ML-benchmark table generator.

Produces small, plausible benchmark tables (models x metrics) together with the
*semantic* information later stages need: which column is the model name, and
for each metric whether higher or lower is better. All randomness flows through
an injected ``random.Random`` so generation is fully reproducible from a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import random

from .schema import Table
from .table_utils import infer_column_types

Direction = Literal["higher", "lower"]

# Metric registry: range + decimal precision + optimization direction.
METRIC_SPECS: dict[str, dict] = {
    "accuracy":   {"direction": "higher", "low": 0.70, "high": 0.99, "decimals": 3},
    "robustness": {"direction": "higher", "low": 0.40, "high": 0.95, "decimals": 2},
    "latency_ms": {"direction": "lower",  "low": 5,    "high": 200,  "decimals": 0},
    "cost":       {"direction": "lower",  "low": 0.5,  "high": 20.0, "decimals": 2},
    "memory_gb":  {"direction": "lower",  "low": 1.0,  "high": 40.0, "decimals": 1},
}

MODEL_NAMES = [
    "FalconLite", "NimbusQA", "OrcaMini", "SableNet", "TerraLM", "VividBERT",
    "ZenithGPT", "AtlasMoE", "CobraXL", "DeltaServe", "EmberTron", "GeoLlama",
]

NAME_COLUMN = "model"


def default_config() -> dict:
    return {
        "min_rows": 3,
        "max_rows": 8,
        "min_metrics": 2,   # -> 3 columns incl. the model name
        "max_metrics": 5,   # -> 6 columns incl. the model name
    }


@dataclass
class GeneratedTable:
    """A synthetic table plus the semantics downstream stages rely on."""

    table: Table
    table_id: str
    name_col: str
    metric_cols: list[str]
    directions: dict[str, Direction] = field(default_factory=dict)


def _sample_value(rng: random.Random, spec: dict) -> float | int:
    v = rng.uniform(spec["low"], spec["high"])
    dec = spec["decimals"]
    if dec == 0:
        return int(round(v))
    return round(v, dec)


def generate_table(rng: random.Random, table_index: int, config: dict | None = None) -> GeneratedTable:
    cfg = {**default_config(), **(config or {})}
    n_rows = rng.randint(cfg["min_rows"], cfg["max_rows"])
    n_metrics = rng.randint(cfg["min_metrics"], cfg["max_metrics"])

    metric_cols = rng.sample(list(METRIC_SPECS.keys()), k=n_metrics)
    names = rng.sample(MODEL_NAMES, k=n_rows)

    headers = [NAME_COLUMN] + metric_cols
    rows: list[list] = []
    for i in range(n_rows):
        row: list = [names[i]]
        for m in metric_cols:
            row.append(_sample_value(rng, METRIC_SPECS[m]))
        rows.append(row)

    column_types = infer_column_types(headers, rows)
    # The name column is always categorical regardless of cardinality.
    column_types[0] = "categorical"

    table = Table(headers=headers, rows=rows, column_types=column_types)
    directions: dict[str, Direction] = {m: METRIC_SPECS[m]["direction"] for m in metric_cols}

    return GeneratedTable(
        table=table,
        table_id=f"tbl_{table_index:04d}",
        name_col=NAME_COLUMN,
        metric_cols=metric_cols,
        directions=directions,
    )
