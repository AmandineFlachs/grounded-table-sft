"""P2.5 - silver-train loop: LLM-render real-table questions, keep the trustworthy.

For each question we render N independent traces via headless ``claude -p`` (Max
plan). A rendering is accepted as "silver" when the N samples AGREE on the final
answer AND every sample is grounded (cited cells exist & match). The accepted
trace is stored with ``trace_source="llm"``.

Because these generated questions also have programmatic gold, the pilot measures
the question that decides whether this method is trustworthy on the user's own
UNLABELED tables: among answers that agree + ground, how many are actually
gold-correct? High agreement→correctness is the licence to use the loop where we
can't check.

Resumable + window-aware (the Max usage window caps a run; see methodology log (m)):
each invocation attempts up to ``--n`` questions (size to one window), APPENDS to the
silver file, skips questions already attempted, and stops early on the spent-window
signature (consecutive ``cli:`` infra failures) so a re-run continues next window.
Repeat until ``--target`` is reached.

    python scripts/silver_render.py --n 25 --target 250 --samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_builder import read_jsonl, write_jsonl  # noqa: E402
from src.llm_renderer import render  # noqa: E402
from src.schema import Example  # noqa: E402
from src.trace_validator import validate  # noqa: E402


def _answer_key(ex: Example) -> tuple:
    """The grounded answer identity used for cross-sample agreement."""
    return (tuple(sorted(ex.gold_answer.rows)), ex.gold_answer.label.strip().lower())


def _stratified(examples: list[Example], n: int) -> list[Example]:
    by_type: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_type[ex.question_type.value].append(ex)
    out, i = [], 0
    types = sorted(by_type)
    while len(out) < n and any(by_type.values()):
        t = types[i % len(types)]
        if by_type[t]:
            out.append(by_type[t].pop(0))
        i += 1
    return out[:n]


def _render_question(src: Example, samples: int, cli: str, timeout: int,
                     t0: float, start_call: int) -> dict:
    """Render one question N times, evaluate agreement+grounding, and decide accept.

    Returns a dict with the per-question detail, the accepted Example (or None),
    how many claude calls it made, and whether it failed in a *wall* way (every
    sample died with a ``cli:`` error - the spent-usage-window signature, distinct
    from a content/parse failure)."""
    renders = []
    for i in range(samples):
        r = render(src, cli=cli, timeout=timeout)
        renders.append(r)
        tag = "ok" if (r.ok and r.example is not None) else f"FAIL {r.error}"
        print(f"    call #{start_call + i + 1} at +{time.time() - t0:5.0f}s  {tag}")

    ok = [r for r in renders if r.ok and r.example is not None]
    errors = [r.error for r in renders if not (r.ok and r.example is not None) and r.error]
    parsed_rate = len(ok)
    results = [(r, validate(r.example)) for r in ok]
    grounded = [r for r, v in results if v.checks.get("cells_exist")]
    keys = {_answer_key(r.example) for r in ok}
    agree = len(ok) == samples and len(keys) == 1
    all_grounded = len(grounded) == samples
    accept = agree and all_grounded

    accepted_ex = None
    gold_correct = None
    if accept:
        v0 = next(v for r, v in results if r is ok[0])
        gold_correct = v0.checks.get("answer_correct", False)
        accepted_ex = ok[0].example
        accepted_ex.metadata["silver"] = {"samples": samples, "agreement": "unanimous"}

    # A "wall" failure = nothing parsed AND every error is an infra (cli:) error.
    # Content/parse failures are genuine outcomes; a spent usage window is not.
    wall_failed = parsed_rate == 0 and bool(errors) and all(e.startswith("cli:") for e in errors)

    detail = {"id": src.metadata.get("example_id", src.table_id),
              "type": src.question_type.value, "samples": samples,
              "parsed": parsed_rate, "agree": agree, "grounded_all": all_grounded,
              "accept": accept, "gold_correct": gold_correct,
              "errors": errors, "first_error": errors[0] if errors else None}
    return {"detail": detail, "accepted": accepted_ex, "calls": samples,
            "wall_failed": wall_failed}


def _persist(out: str, report: str, accepted: list[Example], all_details: list[dict]) -> None:
    """Atomically write the silver file + report (temp file then os.replace), so an
    abrupt shutdown mid-write can never leave a half-written/corrupt file. Called
    periodically (checkpointing) and at the end - making unattended runs that may
    be killed at any moment safe to resume."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = f"{out}.tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        for ex in accepted:
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n")
    os.replace(tmp_out, out)
    tmp_rep = f"{report}.tmp"
    Path(tmp_rep).write_text(json.dumps({"details": all_details}, indent=2), encoding="utf-8")
    os.replace(tmp_rep, report)


def _summarize(details: list[dict], target: int) -> None:
    n = max(len(details), 1)
    acc = [d for d in details if d["accept"]]
    correct = [d for d in acc if d["gold_correct"]]
    print("\n=== P2.5 silver seed summary (cumulative) ===")
    print(f"  attempted (recorded) : {len(details)}  (target {target})")
    print(f"  parsed all N         : {sum(d['parsed'] == d.get('samples', 3) for d in details)}/{n}")
    print(f"  agreed (unanimous)   : {sum(d['agree'] for d in details)}/{n}")
    print(f"  grounded all N       : {sum(d['grounded_all'] for d in details)}/{n}")
    print(f"  ACCEPTED as silver   : {len(acc)}/{n}")
    pct = 100 * len(correct) / max(len(acc), 1)
    print(f"  of accepted, gold-correct: {len(correct)}/{max(len(acc),1)} ({pct:.0f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "realtable.v0_1_0.jsonl"))
    ap.add_argument("--n", type=int, default=25,
                    help="max questions to attempt THIS run (size to one usage window)")
    ap.add_argument("--target", type=int, default=250,
                    help="overall seed size; resume across windows until reached")
    ap.add_argument("--split", default="train", help="dataset split to draw from (no eval leakage)")
    ap.add_argument("--samples", type=int, default=3, help="independent renders per question (N)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--cli", default="claude")
    ap.add_argument("--wall-stop", type=int, default=2,
                    help="consecutive infra-failed questions before stopping (spent window)")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="persist progress every N recorded questions (shutdown-safe)")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "realtable_silver.v0_1_0.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "results" / "p2_5_silver_report.json"))
    args = ap.parse_args()

    # Candidate pool: chosen split only, in a STABLE deterministic order so resume
    # always continues where the last window stopped.
    examples = [Example.model_validate(r) for r in read_jsonl(args.dataset)]
    pool = [e for e in examples if e.split == args.split]
    ordered = _stratified(pool, len(pool))

    # Resume: prior progress = the report's details (attempted ids) + the silver file.
    prior_details: list[dict] = []
    if Path(args.report).exists():
        try:
            prior_details = json.loads(Path(args.report).read_text(encoding="utf-8")).get("details", [])
        except (ValueError, OSError):
            prior_details = []
    attempted = {d["id"] for d in prior_details}
    prior_accepted: list[Example] = []
    if Path(args.out).exists():
        prior_accepted = [Example.model_validate(r) for r in read_jsonl(args.out)]

    remaining = [e for e in ordered if e.metadata.get("example_id") not in attempted]
    to_target = max(0, args.target - len(attempted))
    this_run = remaining[:min(args.n, to_target)]

    print(f"silver seed: pool={len(pool)} ({args.split})  attempted={len(attempted)}/{args.target}  "
          f"this run={len(this_run)} questions x {args.samples} = {len(this_run) * args.samples} calls")
    if not this_run:
        msg = "target reached" if to_target == 0 else "nothing left in pool"
        print(f"  -> {msg}; nothing to do.")
        _summarize(prior_details, args.target)
        return 0

    # Running merged state, persisted incrementally so a shutdown loses ≤ one
    # checkpoint interval. Dedup accepted by example_id.
    all_details = list(prior_details)
    merged: dict[str, Example] = {e.metadata.get("example_id", e.table_id): e for e in prior_accepted}
    new_count = 0
    since_ckpt = 0

    t0 = time.time()
    call = 0
    consec_wall = 0
    stopped_early = False
    for qi, src in enumerate(this_run):
        res = _render_question(src, args.samples, args.cli, args.timeout, t0, call)
        call += res["calls"]
        eid = res["detail"]["id"]

        if res["wall_failed"]:
            # Do NOT record -> stays in the pool to retry next window.
            consec_wall += 1
            print(f"[{qi+1}/{len(this_run)}] !! {eid}  INFRA FAIL ({res['detail']['first_error']}) "
                  f"[{consec_wall}/{args.wall_stop}]")
            if consec_wall >= args.wall_stop:
                stopped_early = True
                print(f"\n  WALL: {consec_wall} consecutive infra failures - usage window likely "
                      f"exhausted. Saving progress and stopping; re-run to continue next window.")
                break
            continue

        consec_wall = 0
        all_details.append(res["detail"])
        if res["accepted"] is not None:
            merged[eid] = res["accepted"]
        new_count += 1
        since_ckpt += 1
        d = res["detail"]
        flag = "OK" if d["accept"] else "--"
        print(f"[{qi+1}/{len(this_run)}] {flag} {eid}  parsed={d['parsed']}/{args.samples} "
              f"agree={d['agree']} grounded={d['grounded_all']} gold_correct={d['gold_correct']}")

        if since_ckpt >= args.checkpoint_every:
            _persist(args.out, args.report, list(merged.values()), all_details)
            since_ckpt = 0
            print(f"    .. checkpoint: {len(all_details)} recorded, {len(merged)} silver")

    # Final persist (covers the tail since the last checkpoint and wall-stops).
    _persist(args.out, args.report, list(merged.values()), all_details)

    print(f"\n  this run: +{new_count} attempted, {len(merged) - len(prior_accepted)} new accepted"
          f"{' (stopped early at wall)' if stopped_early else ''}")
    _summarize(all_details, args.target)
    print(f"  accepted -> {args.out}")
    print(f"  report   -> {args.report}")
    if len(all_details) < args.target and len(all_details) < len(pool):
        print(f"  re-run to continue: {len(all_details)}/{args.target} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
