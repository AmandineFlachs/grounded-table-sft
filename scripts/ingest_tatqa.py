"""P2.2 round-trip harness: ingest TAT-QA tables and report how cleanly they
land in our ``Table`` schema. No question generation yet - this just proves the
ingestion before anything is built on top.

Usage:
    python scripts/ingest_tatqa.py --path data/raw/tatqa/tatqa_dataset_dev.json --show 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.tatqa import ingest_file  # noqa: E402
from src.table_utils import render_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(ROOT / "data" / "raw" / "tatqa" / "tatqa_dataset_dev.json"))
    ap.add_argument("--limit", type=int, default=None, help="max contexts to ingest")
    ap.add_argument("--show", type=int, default=3, help="how many tables to print in full")
    args = ap.parse_args()

    tables = ingest_file(args.path, table_only=True, limit=args.limit)

    conf = Counter(t.confidence for t in tables)
    n_metric = Counter(len(t.metric_cols) for t in tables)
    notes = Counter(n for t in tables for n in t.notes)
    no_metric = [t for t in tables if not t.metric_cols]

    print("=" * 70)
    print(f"ingested tables       : {len(tables)}")
    print(f"confidence            : {dict(conf)}  ({100*conf['high']/max(len(tables),1):.1f}% high)")
    print(f"numeric-column counts : {dict(sorted(n_metric.items()))}")
    print(f"tables w/ 0 metrics   : {len(no_metric)}")
    print(f"note histogram        : {dict(notes)}")
    print()

    # Spot-check a few high-confidence tables in full.
    shown = [t for t in tables if t.confidence == "high"][: args.show]
    for t in shown:
        print("-" * 70)
        print(f"{t.table_id}  (source uid={t.source_uid}, confidence={t.confidence})")
        if t.super_headers:
            print(f"  super-headers: {t.super_headers}")
        print(f"  name_col={t.name_col!r}  metric_cols={t.metric_cols}")
        print(f"  column_types={t.table.column_types}")
        if t.notes:
            print(f"  notes: {t.notes}")
        print(render_markdown(t.table))
        print()

    # And one low-confidence table, to see what's hard.
    low = [t for t in tables if t.confidence == "low"]
    if low:
        t = low[0]
        print("-" * 70)
        print(f"LOW-CONFIDENCE sample: {t.table_id} (uid={t.source_uid})  notes={t.notes}")
        print(render_markdown(t.table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
