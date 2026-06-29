"""Table-level split utilities for leakage-free train/eval separation.

The stored per-example ``split`` field in ``realtable.v0_1_0.jsonl`` was assigned
per EXAMPLE, so the same source table straddles train and eval (data exploration
found 166/224 source tables on both sides). For training we therefore re-split by
SOURCE table: two examples sharing a table - even across the row/col orientation
suffix - must never land on opposite sides of the split.

Used by ``scripts/freeze_splits.py`` (writes the manifest) and
``scripts/build_sft.py`` / ``scripts/train_sft.py`` (assert that no training
table is also an eval table, at both id and content level).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "eval_tables.v0_1_0.json"

_ORIENT = re.compile(r"_(rows|cols)$")


def base_table_id(table_id: str) -> str:
    """Collapse a ``table_id`` to its underlying source table (drop orientation).

    ``tatqa_0000_rows`` and ``tatqa_0000_cols`` are two orientations of the same
    real table and must be treated as one unit for leakage purposes.
    """
    return _ORIENT.sub("", str(table_id))


def content_sig(rec: dict) -> str:
    """Orientation-invariant content signature of a record's table.

    Hash of the sorted multiset of all cell values plus the sorted headers. Two
    tables holding the same data hash equal regardless of transpose, so this
    catches near-duplicate tables that differ only by id - a leak that an
    id-based split alone would miss.
    """
    t = rec["table"]
    vals = [str(c).strip() for row in t["rows"] for c in row]
    vals.append("||".join(sorted(str(h).strip() for h in t["headers"])))
    return hashlib.md5("|".join(sorted(vals)).encode()).hexdigest()


def load_jsonl(path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_no_leak_ids(train_tables, eval_tables, label: str = "train/eval") -> None:
    """Raise if any source table is on both sides (id-level)."""
    inter = set(train_tables) & set(eval_tables)
    if inter:
        raise AssertionError(
            f"LEAKAGE [{label}] id-level: {len(inter)} table(s) in both train and "
            f"eval: {sorted(inter)[:10]}"
        )


def assert_no_leak_content(train_recs, eval_recs, label: str = "train/eval") -> None:
    """Raise if any training table is a content-duplicate of an eval table."""
    train_sigs = {content_sig(r) for r in train_recs}
    inter = [r for r in eval_recs if content_sig(r) in train_sigs]
    if inter:
        ids = sorted({base_table_id(r["table_id"]) for r in inter})
        raise AssertionError(
            f"LEAKAGE [{label}] content-level: {len(inter)} eval example(s) are "
            f"content-duplicates of training tables: {ids[:10]}"
        )
