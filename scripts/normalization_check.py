"""Normalization null-result check (P3.7): is the trace-groundedness gap a measurement
artifact that a principled value-canonicalizer would erase, or is it genuine?

Reads the groundedness diagnostic (results/p3_4b_groundedness_diag.json) - no GPU, no model
re-run - and applies a PRINCIPLED canonicalizer to every bad citation: percent / currency /
thousands-separators / numeric-strings are matched to their numeric cell, but an EMPTY string
is NOT a number (empty != 0), and rounding (0.92 cited for 0.923) is a real value error, not a
format equivalence. Reports how many bad cites / cases a canonicalizer would legitimately
reclaim. Result (2026-06-26): 1/112 cites, 1/36 cases -> the gap is GENUINE, so the strict
validator is retained unchanged (no metric loosening before the locked test).

Run from project root:  python scripts/normalization_check.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DIAG = "results/p3_4b_groundedness_diag.json"
OUT = "results/p3_4b_normalization_check.json"


def canon_num(v):
    """Canonicalize a value to a float if it denotes a number, else None. Empty string -> None
    (empty is not zero). Handles $/£/€/¥, thousands commas, %, and accounting (parens) negatives."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if s == "":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = re.sub(r"[\$£€¥,]", "", s)
    s = re.sub(r"(RMB|USD|GBP|EUR)", "", s, flags=re.I).strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if pct:
        x /= 100.0
    if neg:
        x = -x
    return x


def fmt_equiv(val, actual) -> bool:
    """True iff val and actual are the SAME value modulo surface format (not modulo rounding)."""
    a, b = canon_num(val), canon_num(actual)
    if a is not None and b is not None:
        return abs(a - b) <= 1e-6
    if isinstance(val, str) and isinstance(actual, str):
        return val.strip() == actual.strip()
    return False


def main() -> int:
    d = json.loads(Path(DIAG).read_text(encoding="utf-8"))
    parsed_ungrounded = [c for c in d["cases"] if c.get("parsed") and not c.get("grounded")]

    total_bad = reclaimable_cites = reclaim_cases = 0
    examples = []
    for c in parsed_ungrounded:
        bad = c.get("bad_cites", [])
        if not bad:
            continue
        all_fmt = True
        for b in bad:
            total_bad += 1
            if fmt_equiv(b["value"], b["actual"]):
                reclaimable_cites += 1
                examples.append({"id": c["id"], "value": b["value"], "actual": b["actual"]})
            else:
                all_fmt = False
        if all_fmt:
            reclaim_cases += 1

    summary = {
        "source": DIAG,
        "n_parsed_ungrounded_cases": len(parsed_ungrounded),
        "total_bad_cites": total_bad,
        "format_equivalent_cites": reclaimable_cites,
        "cases_fully_reclaimed_by_normalization": reclaim_cases,
        "reclaimable_examples": examples,
        "conclusion": ("groundedness gap is GENUINE, not a formatting artifact; "
                       "strict validator retained unchanged"),
    }
    Path(OUT).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
