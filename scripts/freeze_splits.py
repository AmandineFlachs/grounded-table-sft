"""Freeze a leakage-free, table-level train/eval split (P3.0.5).

WHY. The stored per-example ``split`` field is leaky: it was assigned per example,
so 166/224 source tables appear in both train and eval. This script ignores that
field and re-splits by SOURCE table, then writes a *versioned manifest* that is the
single source of truth for every downstream step (build_sft, train_sft, eval_model)
and for any future data we add. Two eval tiers are produced:

  * PRIMARY (in-distribution by task, held-out by table): programmatic examples on
    eval tables, covering the three TRAINED task types. Partitioned BY TABLE into a
    ``dev`` slice (used for all iteration) and a ``test`` slice (scored exactly once
    at the very end) so iterating never silently overfits the eval.
  * SECONDARY: the 12 TAT-QA-gold-anchored examples (extremum / rank_models) - an
    OUT-OF-DISTRIBUTION probe (a task type not in training). The few tables they
    share with the silver training set are excluded from TRAINING (recorded here) so
    the probe stays clean.

Leakage is asserted at both ID and CONTENT level (orientation-invariant cell hash).
Read-only over the datasets; writes only the manifest + the two eval JSONL files.

    python scripts/freeze_splits.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import (  # noqa: E402
    MANIFEST,
    ROOT,
    assert_no_leak_content,
    assert_no_leak_ids,
    base_table_id,
    content_sig,
    load_jsonl,
)

PROG = ROOT / "data" / "processed" / "realtable.v0_1_0.jsonl"
SILVER = ROOT / "data" / "processed" / "realtable_silver_clean.v0_1_0.jsonl"
VERIFIED = ROOT / "data" / "processed" / "realtable_eval_verified.v0_1_0.jsonl"

DEV_OUT = ROOT / "data" / "processed" / "eval_dev.v0_1_0.jsonl"
TEST_OUT = ROOT / "data" / "processed" / "eval_test.v0_1_0.jsonl"

# How many non-training source tables to reserve for evaluation, and how to split
# that reserve into a (larger) locked test set and a (smaller) dev set. Table-level
# splitting balances task types automatically (each table yields all three types).
EVAL_RESERVE = 80
TEST_TABLES = 45  # remainder of the reserve -> dev
SEED = 0
TRAINED_TYPES = {"best_under_constraint", "threshold_filter", "tradeoff_summary"}


def _by_type(recs):
    return dict(Counter(r.get("question_type") for r in recs))


def _write(path: Path, recs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    prog = load_jsonl(PROG)
    silver = load_jsonl(SILVER)
    verified = load_jsonl(VERIFIED)

    # --- source-table sets ---------------------------------------------------
    training_tables = {base_table_id(r["table_id"]) for r in silver}
    verified_tables = {base_table_id(r["table_id"]) for r in verified}
    all_prog_tables = {base_table_id(r["table_id"]) for r in prog}
    non_training = sorted(all_prog_tables - training_tables)

    # Tables the silver training set shares with the verified probe: drop these
    # from TRAINING (build_sft reads this list) to keep the OOD probe clean.
    exclude_from_training = sorted(training_tables & verified_tables)

    # --- reserve + deterministic dev/test partition (by table) ---------------
    rng = random.Random(SEED)
    shuffled = non_training[:]
    rng.shuffle(shuffled)
    reserve = shuffled[: min(EVAL_RESERVE, len(shuffled))]
    test_tables = set(reserve[:TEST_TABLES])
    dev_tables = set(reserve[TEST_TABLES:])
    future_train_pool = sorted(set(non_training) - test_tables - dev_tables)

    # --- materialize eval sets (trained task types only) ---------------------
    def pick(tables):
        return [
            r
            for r in prog
            if base_table_id(r["table_id"]) in tables
            and r.get("question_type") in TRAINED_TYPES
        ]

    dev_recs = pick(dev_tables)
    test_recs = pick(test_tables)

    # --- leakage assertions (id + content) -----------------------------------
    assert_no_leak_ids(training_tables, test_tables | dev_tables, "silver/eval")
    assert_no_leak_content(silver, dev_recs + test_recs, "silver/eval")
    # dev and test must not share a table either
    assert_no_leak_ids(dev_tables, test_tables, "dev/test")

    # --- manifest ------------------------------------------------------------
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v0_1_0",
        "note": "Table-level split. Ignores the leaky per-example `split` field. "
        "base_table_id strips the _(rows|cols) orientation suffix.",
        "seed": SEED,
        "counts": {
            "programmatic_examples": len(prog),
            "source_tables_total": len(all_prog_tables),
            "training_tables": len(training_tables),
            "non_training_tables": len(non_training),
            "eval_reserve": len(reserve),
            "dev_tables": len(dev_tables),
            "test_tables": len(test_tables),
            "future_train_pool_tables": len(future_train_pool),
            "dev_examples": len(dev_recs),
            "test_examples": len(test_recs),
        },
        "training_tables": sorted(training_tables),
        "exclude_from_training": exclude_from_training,
        "eval_tables_dev": sorted(dev_tables),
        "eval_tables_test": sorted(test_tables),
        "future_train_pool": future_train_pool,
        "verified_tables": sorted(verified_tables),
        "leakage_checks": {"id_level": "passed", "content_level": "passed"},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write(DEV_OUT, dev_recs)
    _write(TEST_OUT, test_recs)

    # --- report --------------------------------------------------------------
    print("Frozen table-level split (P3.0.5)")
    print(f"  source tables total      : {len(all_prog_tables)}")
    print(f"  training tables (silver) : {len(training_tables)}")
    print(f"  excluded from TRAINING   : {len(exclude_from_training)} {exclude_from_training}")
    print(f"  non-training tables      : {len(non_training)}")
    print(f"  eval reserve             : {len(reserve)}  -> test {len(test_tables)} / dev {len(dev_tables)}")
    print(f"  future train pool        : {len(future_train_pool)} tables (for P3.5 scale-up)")
    print(f"  DEV  examples : {len(dev_recs):4d}  by type: {_by_type(dev_recs)}")
    print(f"  TEST examples : {len(test_recs):4d}  by type: {_by_type(test_recs)}")
    print("  leakage: id-level PASS, content-level PASS, dev/test disjoint PASS")
    print(f"\n  manifest : {MANIFEST.relative_to(ROOT)}")
    print(f"  dev set  : {DEV_OUT.relative_to(ROOT)}")
    print(f"  test set : {TEST_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
