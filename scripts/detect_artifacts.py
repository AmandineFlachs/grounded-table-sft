"""Data-quality artifact detector / quality gate (P3.10).

Scans a dataset JSONL (or freshly-ingested TAT-QA tables) for the two artifacts the
independent anchors surfaced (methodology log w/x):

  * HEADER-LEAK rows  - a sub-header that leaked into the body: a data row whose value
    cells carry NO real magnitude, only YEARS (which parse as numbers) and/or header
    text, e.g. ``(in millions) | 2019 | 2018 | Actual | Comp.`` Requires >=2 year-like
    values so a lone amount that merely falls in 1900-2099 (e.g. 2047) isn't flagged.
  * DUPLICATE row labels - multi-section tables flattened to repeated labels (two
    "Leasehold" rows), making a cited/answered row ambiguous.

Both are now fixed at ingestion (src/ingest/tatqa.py); this is the regression gate.
Exit code 1 if any artifact is found (so it can guard a build).

    python scripts/detect_artifacts.py data/processed/realtable.v0_2_0.jsonl
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import Example          # noqa: E402
from src.splits import load_jsonl       # noqa: E402


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_year(v) -> bool:
    return _is_num(v) and float(v).is_integer() and 1900 <= v <= 2099


def header_leak_rows(table) -> list[int]:
    out = []
    for ri, row in enumerate(table.rows):
        vals = [v for v in row[1:] if not (isinstance(v, str) and v.strip() == "")]
        if not vals:
            continue
        real_mag = sum(1 for v in vals if _is_num(v) and not _is_year(v))
        years = sum(1 for v in vals if _is_year(v))
        if real_mag == 0 and years >= 2:
            out.append(ri)
    return out


def duplicate_labels(table) -> dict[str, int]:
    labels = [str(r[0]).strip() for r in table.rows]
    return {l: labels.count(l) for l in set(labels) if l != "" and labels.count(l) > 1}


def main(argv: list[str]) -> int:
    path = argv[0] if argv else "data/processed/realtable.v0_1_0.jsonl"
    recs = load_jsonl(Path(path))
    tables = {}
    for r in recs:
        ex = Example.model_validate(r)
        tables[ex.table_id] = ex.table  # dedupe by table

    hl = {tid: header_leak_rows(t) for tid, t in tables.items() if header_leak_rows(t)}
    dl = {tid: duplicate_labels(t) for tid, t in tables.items() if duplicate_labels(t)}
    n = len(tables)
    print(f"=== artifact scan: {path}  ({n} unique tables) ===")
    print(f"  HEADER-LEAK tables    : {len(hl)}  ({100*len(hl)/n:.1f}%)")
    for tid, rows in list(hl.items())[:10]:
        print(f"      {tid}  rows {rows}")
    print(f"  DUPLICATE-LABEL tables: {len(dl)}  ({100*len(dl)/n:.1f}%)")
    for tid, d in list(dl.items())[:10]:
        print(f"      {tid}  {d}")
    bad = len(hl) + len(dl)
    print(f"\n  {'CLEAN - no artifacts' if bad == 0 else f'{bad} table(s) with artifacts'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
