"""Assemble validated examples into a JSONL dataset.

Pipeline per record: generate a fresh table -> build one question/answer/trace
-> validate -> (if valid) assign a train/eval split -> collect. One example per
table keeps ``table_id``s disjoint, so the train/eval split has no leakage and
edge-case table mutations (e.g. forced ties) never contaminate other examples.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import cycle

from . import GENERATOR_VERSION
from .question_templates import BUILDERS, Mode
from .schema import Example, QuestionType
from .synthetic_generator import default_config, generate_table
from .table_utils import render_markdown
from .trace_validator import ValidationResult, validate

TYPE_ORDER = [
    QuestionType.BEST_UNDER_CONSTRAINT,
    QuestionType.THRESHOLD_FILTER,
    QuestionType.TRADEOFF_SUMMARY,
]

HARD_MODES = {
    QuestionType.BEST_UNDER_CONSTRAINT: ["empty", "near_threshold", "tie"],
    QuestionType.THRESHOLD_FILTER: ["empty", "near_threshold"],
    QuestionType.TRADEOFF_SUMMARY: ["normal"],  # trade-off edges arise from the data, not a mode
}


@dataclass
class InvalidReport:
    example_id: str
    result: ValidationResult


def build_dataset(n: int, seed: int, config: dict | None = None
                  ) -> tuple[list[Example], list[InvalidReport]]:
    cfg = {**default_config(),
           "hard_case_fraction": 0.35, "eval_fraction": 0.2,
           **(config or {})}
    rng = random.Random(seed)
    examples: list[Example] = []
    invalid: list[InvalidReport] = []
    type_cycle = cycle(TYPE_ORDER)
    table_index = 0

    # Generous attempt cap so a few un-constructible draws cannot loop forever.
    max_attempts = n * 20
    attempts = 0
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        qtype = next(type_cycle)
        gen = generate_table(rng, table_index, cfg)
        table_index += 1

        mode: Mode = "normal"
        if rng.random() < cfg["hard_case_fraction"]:
            mode = rng.choice(HARD_MODES[qtype])  # type: ignore[assignment]

        ex = BUILDERS[qtype](gen, rng, mode)
        if ex is None:
            continue

        ex.metadata["seed"] = seed
        ex.metadata["generator_version"] = GENERATOR_VERSION
        ex.metadata["table_index"] = table_index - 1
        ex.split = "eval" if rng.random() < cfg["eval_fraction"] else "train"

        result = validate(ex)
        if result.valid:
            examples.append(ex)
        else:
            invalid.append(InvalidReport(ex.metadata.get("example_id", "?"), result))

    return examples, invalid


def write_jsonl(examples: list[Example], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Supervised-fine-tuning view (used by the eval stub now, training later)
# --------------------------------------------------------------------------- #
RESPONSE_INSTRUCTIONS = (
    "Answer using only the table. Respond with a JSON object containing "
    '"trace_steps" (each with "kind", "description", and "cites") and '
    '"final_answer" (with "label" and "rows"). Cite the exact cells you use.'
)


def to_prompt(ex: Example) -> str:
    return (f"{render_markdown(ex.table)}\n\n"
            f"Question: {ex.question}\n\n{RESPONSE_INSTRUCTIONS}")


def to_target(ex: Example) -> dict:
    return {
        "trace_steps": [s.model_dump(mode="json") for s in ex.trace_steps],
        "final_answer": ex.gold_answer.model_dump(mode="json"),
    }
