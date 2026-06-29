"""Phase 1.5 driver: render LLM traces over a sample of synthetic examples,
validate each against known truth, and report the pass rate.

Usage:
    python scripts/render_traces.py --n 6 --dataset data/processed/dataset.v0_1_0.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_builder import read_jsonl  # noqa: E402
from src.llm_renderer import render  # noqa: E402
from src.schema import Example  # noqa: E402
from src.trace_validator import validate  # noqa: E402


def stratified(records: list[dict], n: int) -> list[dict]:
    """Pick up to n records, round-robin across question types, preferring a mix
    of hard and easy."""
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r)
    for v in by_type.values():
        v.sort(key=lambda r: 0 if r["metadata"].get("difficulty") == "hard" else 1)
    out, queues = [], list(by_type.values())
    i = 0
    while len(out) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
        if i > n * len(queues) + 10:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Render + validate LLM traces (Phase 1.5).")
    p.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "dataset.v0_1_0.jsonl"))
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "rendered.v0_1_0.jsonl"))
    p.add_argument("--report", default=str(ROOT / "results" / "phase1_5_render_report.json"))
    args = p.parse_args()

    records = read_jsonl(args.dataset)
    sample = stratified(records, args.n)
    sources = [Example.model_validate(r) for r in sample]

    rendered: list[Example] = []
    failures: collections.Counter = collections.Counter()
    rows_report = []
    n_parse_ok = n_valid = n_answer_ok = n_grounded = 0

    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sources, 1):
        sid = src.metadata.get("example_id")
        out = render(src, timeout=args.timeout)
        line = {"id": sid, "type": src.question_type.value,
                "difficulty": src.metadata.get("difficulty")}
        if not out.ok:
            line.update({"parse_ok": False, "valid": False, "error": out.error})
            failures["render/parse"] += 1
            print(f"[{i}/{len(sources)}] {sid}: RENDER/PARSE FAIL - {out.error}")
        else:
            n_parse_ok += 1
            res = validate(out.example)
            rendered.append(out.example)
            if res.checks.get("answer_correct"):
                n_answer_ok += 1
            if res.checks.get("cells_exist"):
                n_grounded += 1
            line.update({"parse_ok": True, "valid": res.valid, "failed": res.failed_checks()})
            if res.valid:
                n_valid += 1
                print(f"[{i}/{len(sources)}] {sid}: VALID")
            else:
                for fc in res.failed_checks():
                    failures[fc] += 1
                print(f"[{i}/{len(sources)}] {sid}: INVALID - {res.failed_checks()}")
                line["issues"] = res.issues
                line["raw"] = out.raw[:1500]
        rows_report.append(line)

    n = len(sources)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in rendered:
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")
    summary = {
        "n": n, "parse_ok": n_parse_ok, "valid": n_valid,
        "answer_correct": n_answer_ok, "grounded": n_grounded,
        "failure_histogram": dict(failures), "rows": rows_report,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def pct(x):
        return f"{100.0 * x / n:5.1f}%" if n else "n/a"

    print("\n=== Phase 1.5 render summary ===")
    print(f"  examples            : {n}")
    print(f"  parsed to schema    : {pct(n_parse_ok)} ({n_parse_ok}/{n})")
    print(f"  fully valid         : {pct(n_valid)} ({n_valid}/{n})")
    print(f"  answer correct      : {pct(n_answer_ok)} ({n_answer_ok}/{n})")
    print(f"  grounded (cells ok) : {pct(n_grounded)} ({n_grounded}/{n})")
    print(f"  failure histogram   : {dict(failures)}")
    print(f"  rendered -> {args.out}")
    print(f"  report   -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
