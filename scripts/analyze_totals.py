"""P2.5 diagnosis (Part B) - how prevalent is the totals-convention ambiguity?

The silver run produced one false accept (tatqa_0000_rows_threshold_filter_normal):
all 3 samples agreed, all grounded, yet WRONG against gold, because the model
counted total/subtotal rows as "line items". Our gold deliberately excludes totals
from the entity universe (spec["rows"]); the prompt never states that convention,
so the silver gate (agreement + groundedness) cannot catch it.

This script quantifies the risk WITHOUT any LLM: for every generated example it
recomputes the answer twice - over the gold universe (totals excluded) and over
ALL rows (totals included, the naive reading) - and counts, per question type, how
often the two diverge. A divergence is exactly a question where a naive "totals are
entities" reader gets a wrong-per-gold answer.

Read-only: reads data/processed/realtable.v0_1_0.jsonl and writes nothing.

It REUSES the validator's recompute helpers (so the "expected" answer is computed
by the same code that gates the dataset) and realtable_questions._is_total.

    python scripts/analyze_totals.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_builder import read_jsonl  # noqa: E402
from src.realtable_questions import _is_total  # noqa: E402
from src.schema import Example  # noqa: E402
from src.trace_validator import _frontier, _is_num, _num_passes, _optimum_rows  # noqa: E402

DATASET = ROOT / "data" / "processed" / "realtable.v0_1_0.jsonl"


def _expected_rows(table, spec: dict, universe: list[int]) -> list[int]:
    """Recompute the gold rows for a spec over a given row universe.

    Mirrors trace_validator._check_answer's recompute branches exactly, but
    returns the expected rows instead of calling a fail() callback. Reuses the
    validator's own predicate helpers so the maths is identical to the gate.
    """
    t = spec["type"]
    if t == "best_under_constraint":
        cc = table.col_index(spec["constraint"])
        tc = table.col_index(spec["target"])
        survivors = [r for r in universe
                     if _is_num(table.cell(r, tc))
                     and _num_passes(table.cell(r, cc), spec["op"], spec["threshold"])]
        return _optimum_rows(survivors, tc, table, spec["target_dir"]) if survivors else []
    if t == "threshold_filter":
        conds = spec["conditions"]
        return sorted(r for r in universe
                      if all(_num_passes(table.cell(r, table.col_index(c["metric"])), c["op"], c["T"])
                             for c in conds))
    if t == "extremum":
        tc = table.col_index(spec["target"])
        cand = [r for r in universe if _is_num(table.cell(r, tc))]
        return _optimum_rows(cand, tc, table, spec["target_dir"]) if cand else []
    if t == "tradeoff_summary":
        return _frontier(table, spec["c1"], spec["c2"], spec["d1"], spec["d2"], universe)
    raise ValueError(f"unknown spec type {t!r}")


def _total_rows(table) -> set[int]:
    """Rows whose label (col 0) is a total/subtotal - holds in both orientations
    (across_rows: line-item labels; across_columns: period labels incl. 'Total')."""
    return {r for r in range(len(table.rows)) if _is_total(table.cell(r, 0))}


def main() -> int:
    records = read_jsonl(str(DATASET))
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    integrity_mismatch = []  # exp_gold should equal the stored gold (sanity)
    examples_divergent = []  # ids where a naive reader diverges

    for raw in records:
        ex = Example.model_validate(raw)
        spec = ex.metadata.get("spec")
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        t = spec["type"]
        table = ex.table
        n = len(table.rows)

        gold_universe = [r for r in spec.get("rows", range(n)) if 0 <= r < n]
        full_universe = list(range(n))
        totals = _total_rows(table)

        exp_gold = sorted(_expected_rows(table, spec, gold_universe))
        exp_full = sorted(_expected_rows(table, spec, full_universe))

        by_type[t]["examples"] += 1
        by_type[t]["with_total_rows"] += int(bool(totals))

        # sanity: recompute over the gold universe must match the stored gold
        if exp_gold != sorted(ex.gold_answer.rows):
            integrity_mismatch.append(ex.metadata.get("example_id"))

        if exp_gold != exp_full:
            by_type[t]["diverged"] += 1
            added = set(exp_full) - set(exp_gold)
            # every added row should be a total (the only universe difference)
            if added and added <= totals:
                by_type[t]["diverged_due_to_total"] += 1
            examples_divergent.append((ex.metadata.get("example_id"), t,
                                       exp_gold, exp_full, sorted(added & totals)))

    # ---- report ----
    print(f"totals-convention prevalence over {len(records)} examples "
          f"({DATASET.name})\n")
    hdr = f"{'type':<22}{'examples':>9}{'w/totals':>9}{'diverged':>9}{'%':>6}{'by_total':>9}"
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(int)
    for t in sorted(by_type):
        s = by_type[t]
        pct = 100 * s["diverged"] / max(s["examples"], 1)
        print(f"{t:<22}{s['examples']:>9}{s['with_total_rows']:>9}"
              f"{s['diverged']:>9}{pct:>5.0f}%{s['diverged_due_to_total']:>9}")
        for k, v in s.items():
            tot[k] += v
    pct = 100 * tot["diverged"] / max(tot["examples"], 1)
    print("-" * len(hdr))
    print(f"{'ALL':<22}{tot['examples']:>9}{tot['with_total_rows']:>9}"
          f"{tot['diverged']:>9}{pct:>5.0f}%{tot['diverged_due_to_total']:>9}")

    print(f"\ndivergent examples (naive 'include totals' reader gets a wrong answer): "
          f"{len(examples_divergent)}")
    print("first 12:")
    for eid, t, eg, ef, tot_added in examples_divergent[:12]:
        print(f"  {eid}\n      gold={eg}  naive={ef}  total-rows-wrongly-added={tot_added}")

    # sanity-check the metric: the observed false accept must be flagged divergent
    target = "tatqa_0000_rows_threshold_filter_normal"
    hit = [d for d in examples_divergent if d[0] == target]
    print(f"\nsanity: observed false-accept '{target}' divergent? "
          f"{'YES' if hit else 'NO'}")

    if integrity_mismatch:
        print(f"\n[!] integrity: {len(integrity_mismatch)} examples where recompute over the "
              f"gold universe != stored gold (unexpected): {integrity_mismatch[:5]}")
    else:
        print("\nintegrity: recompute over gold universe == stored gold for all examples (OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
