"""Show silver-seed progress at a glance (read-only).

Reads the runner's report + silver file and prints how far the seed has got:
attempted / accepted / gold-correct rate, a per-type breakdown, and when the
files were last updated. Safe to run anytime, including while a run is going -
it only reads the checkpointed files.

    python scripts/seed_status.py
    python scripts/seed_status.py --target 250
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "p2_5_silver_report.json"
SILVER = ROOT / "data" / "processed" / "realtable_silver.v0_1_0.jsonl"


def _age(p: Path) -> str:
    if not p.exists():
        return "missing"
    secs = time.time() - p.stat().st_mtime
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs/60:.0f} min ago"
    return f"{secs/3600:.1f} h ago"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250)
    args = ap.parse_args()

    if not REPORT.exists():
        print("No report yet - no run has checkpointed. (A run in progress may not have "
              "written its first checkpoint.)")
        return 0

    details = json.loads(REPORT.read_text(encoding="utf-8")).get("details", [])
    accepted = [d for d in details if d.get("accept")]
    correct = [d for d in accepted if d.get("gold_correct")]
    silver_n = sum(1 for _ in open(SILVER, encoding="utf-8")) if SILVER.exists() else 0

    done = len(details)
    bar_w = 28
    filled = int(bar_w * min(done, args.target) / max(args.target, 1))
    bar = "#" * filled + "-" * (bar_w - filled)
    pct = 100 * done / max(args.target, 1)

    print(f"silver seed progress   [{bar}] {done}/{args.target} ({pct:.0f}%)")
    print(f"  attempted (recorded) : {done}")
    print(f"  accepted as silver   : {len(accepted)}   (file has {silver_n} records)")
    rate = 100 * len(correct) / max(len(accepted), 1)
    print(f"  of accepted, correct : {len(correct)}/{max(len(accepted),1)} ({rate:.0f}%)")

    by = defaultdict(lambda: [0, 0])  # type -> [attempted, accepted]
    for d in details:
        by[d["type"]][0] += 1
        by[d["type"]][1] += int(bool(d.get("accept")))
    print("  by type (accepted/attempted):")
    for t in sorted(by):
        a, acc = by[t]
        print(f"      {t:<22} {acc}/{a}")

    print(f"\n  report updated : {_age(REPORT)}")
    print(f"  silver updated : {_age(SILVER)}")
    if done < args.target:
        print(f"  -> {args.target - done} to go; re-run silver_render to continue.")
    else:
        print("  -> target reached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
