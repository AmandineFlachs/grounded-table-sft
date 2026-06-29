"""Generate -> validate -> write the JSONL dataset.

Usage:
    python scripts/build_dataset.py --n 40 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import GENERATOR_VERSION  # noqa: E402
from src.dataset_builder import build_dataset, write_jsonl  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Build the grounded table-reasoning dataset.")
    p.add_argument("--n", type=int, default=40, help="number of examples to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="output JSONL path")
    args = p.parse_args()

    out = args.out or str(
        ROOT / "data" / "processed" / f"dataset.v{GENERATOR_VERSION.replace('.', '_')}.jsonl"
    )
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)

    examples, invalid = build_dataset(n=args.n, seed=args.seed)

    if invalid:
        print(f"BUILD FAILED: {len(invalid)} example(s) did not pass validation:", file=sys.stderr)
        for rep in invalid[:10]:
            print(f"  - {rep.example_id}: {', '.join(rep.result.failed_checks())}", file=sys.stderr)
            for issue in rep.result.issues[:3]:
                print(f"      {issue}", file=sys.stderr)
        return 1

    write_jsonl(examples, out)

    by_type = collections.Counter(e.question_type.value for e in examples)
    by_diff = collections.Counter(e.metadata.get("difficulty") for e in examples)
    by_split = collections.Counter(e.split for e in examples)
    edges = collections.Counter(ec for e in examples for ec in e.metadata.get("edge_cases", []))

    print(f"Wrote {len(examples)} validated examples -> {out}")
    print(f"  validator pass rate: 100% ({len(examples)}/{len(examples)})")
    print(f"  by type      : {dict(by_type)}")
    print(f"  by difficulty: {dict(by_diff)}")
    print(f"  by split     : {dict(by_split)}")
    print(f"  edge cases   : {dict(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
