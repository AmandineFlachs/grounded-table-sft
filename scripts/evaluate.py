"""Evaluation harness (stub).

Computes the v0 target metrics over a dataset. With no model predictions
supplied it scores the gold set against itself - a sanity baseline that should
be perfect. The structure is ready to later accept a predictions file.

Usage:
    python scripts/evaluate.py data/processed/dataset.v0_1_0.jsonl
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_builder import read_jsonl  # noqa: E402
from src.trace_validator import validate_dict  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate grounded table-reasoning examples.")
    p.add_argument("dataset", type=str, help="path to dataset JSONL")
    p.add_argument("--split", type=str, default=None, choices=[None, "train", "eval"],
                   help="restrict to one split")
    args = p.parse_args()

    records = read_jsonl(args.dataset)
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    if not records:
        print("No records to evaluate.", file=sys.stderr)
        return 1

    n = len(records)
    json_valid = 0
    answer_ok = 0
    trace_ok = 0
    grounded_total = 0.0
    failures: collections.Counter = collections.Counter()

    for r in records:
        res = validate_dict(r)
        # JSON/schema validity
        if res.checks.get("schema_valid", False):
            json_valid += 1
        # Trace correctness = whole-record validity (the gold-vs-gold signal)
        if res.valid:
            trace_ok += 1
        # Answer accuracy
        if res.checks.get("answer_correct", False):
            answer_ok += 1
        # Groundedness = cells that exist & match (1.0 if the cells_exist check passed)
        grounded_total += 1.0 if res.checks.get("cells_exist", False) else 0.0
        for fc in res.failed_checks():
            failures[fc] += 1

    def pct(x: int) -> str:
        return f"{100.0 * x / n:5.1f}%"

    print(f"Evaluated {n} records from {args.dataset}"
          + (f" (split={args.split})" if args.split else ""))
    print(f"  JSON validity rate    : {pct(json_valid)} ({json_valid}/{n})")
    print(f"  Answer accuracy       : {pct(answer_ok)} ({answer_ok}/{n})")
    print(f"  Groundedness score    : {100.0 * grounded_total / n:5.1f}%")
    print(f"  Trace correctness     : {pct(trace_ok)} ({trace_ok}/{n})")
    if failures:
        print(f"  Failure categories    : {dict(failures)}")
    else:
        print("  Failure categories    : none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
