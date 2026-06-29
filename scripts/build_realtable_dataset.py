"""P2.3: build a real-table dataset from ingested TAT-QA tables.

For each high-confidence ingested table, in both orientations, generate our
selection questions (explicit-direction phrasing), validate each with the SAME
trace_validator used for synthetic data, and write the valid ones to JSONL.

The headline number is the validation pass rate: because gold is derived by
construction and the validator independently recomputes it, a high pass rate
confirms the real-table pipeline is sound; any failures are real findings.

Usage:
    python scripts/build_realtable_dataset.py --path data/raw/tatqa/tatqa_dataset_dev.json --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import GENERATOR_VERSION  # noqa: E402
from src.dataset_builder import write_jsonl  # noqa: E402
from src.ingest.tatqa import ingest_file  # noqa: E402
from src.realtable_questions import BUILDERS, Mode, orientations  # noqa: E402
from src.schema import QuestionType  # noqa: E402
from src.trace_validator import validate  # noqa: E402

TYPE_ORDER = [
    QuestionType.BEST_UNDER_CONSTRAINT,
    QuestionType.THRESHOLD_FILTER,
    QuestionType.TRADEOFF_SUMMARY,
]
HARD_MODES: dict[QuestionType, list[Mode]] = {
    QuestionType.BEST_UNDER_CONSTRAINT: ["empty", "near_threshold"],
    QuestionType.THRESHOLD_FILTER: ["empty", "near_threshold"],
    QuestionType.TRADEOFF_SUMMARY: ["normal"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(ROOT / "data" / "raw" / "tatqa" / "tatqa_dataset_dev.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-fraction", type=float, default=0.35)
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "realtable.v0_1_0.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ingested = [t for t in ingest_file(args.path, table_only=True, limit=args.limit)
                if t.confidence == "high"]

    valid, invalid = [], []
    attempted = 0
    for it in ingested:
        for rt in orientations(it):
            for qtype in TYPE_ORDER:
                mode: Mode = "normal"
                if rng.random() < args.hard_fraction:
                    mode = rng.choice(HARD_MODES[qtype])
                ex = BUILDERS[qtype](rt, rng, mode)
                if ex is None:
                    continue
                attempted += 1
                ex.metadata["seed"] = args.seed
                ex.metadata["generator_version"] = GENERATOR_VERSION
                ex.split = "eval" if rng.random() < args.eval_fraction else "train"
                res = validate(ex)
                if res.valid:
                    valid.append(ex)
                else:
                    invalid.append((ex.metadata.get("example_id", "?"), res))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(valid, args.out)

    print("=" * 70)
    print(f"high-confidence tables : {len(ingested)}")
    print(f"questions attempted    : {attempted}")
    print(f"valid                  : {len(valid)} "
          f"({100*len(valid)/max(attempted,1):.1f}%)")
    print(f"invalid                : {len(invalid)}")
    print(f"  by type      : {dict(Counter(e.question_type.value for e in valid))}")
    print(f"  by orientation: {dict(Counter(e.metadata['orientation'] for e in valid))}")
    print(f"  by split     : {dict(Counter(e.split for e in valid))}")
    edges = Counter(ec for e in valid for ec in e.metadata.get('edge_cases', []))
    print(f"  edge cases   : {dict(edges)}")
    print(f"written -> {args.out}")

    if invalid:
        fails = Counter(c for _, r in invalid for c in r.failed_checks())
        print()
        print(f"FAILURE check histogram: {dict(fails)}")
        for ex_id, r in invalid[:5]:
            print(f"  - {ex_id}: {r.issues[:2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
