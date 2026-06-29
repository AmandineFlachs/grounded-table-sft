"""Evaluate the executor ("answer by construction") system (P3.6 Step 4).

Runs a trained model that emits trace + final_answer + a structured `operation`, then
scores the answer the DETERMINISTIC engine computes from that operation (compute_answer)
against gold - not the model's own arithmetic. Keeps eval_model.py (the baseline scorer)
untouched.

Reports the full rigor suite:
  - executor answer accuracy STRICT (engine-error = 0) and WITH-FALLBACK (model final_answer)
  - valid / grounded (validator run on the engine-substituted example)
  - op-extraction: whole-op exact-match + per-field accuracy, by type
  - model-vs-engine 2x2 + exact McNemar vs the model's own final_answer
  - paired McNemar vs a prior baseline run (join on example_id)
  - ablations: gold-op ceiling (~100% sanity) and all-rows-universe (sizes the universe's role)
  - Wilson 95% CIs on every rate; per-type breakdown

    python scripts/eval_executor.py --adapter models/qwen3-4b-sft-exec
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.executor import compute_answer  # noqa: E402
from src.llm_renderer import build_prompt, extract_json, extract_operation, render  # noqa: E402
from src.local_infer import LocalGenerator  # noqa: E402
from src.schema import Example, GoldAnswer  # noqa: E402
from src.sft_format import operation_dict  # noqa: E402
from src.splits import ROOT, load_jsonl  # noqa: E402
from src.table_utils import entity_universe  # noqa: E402
from src.trace_validator import validate  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "0.0 [0.0-0.0]"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100*p:.1f} [{100*max(0,c-h):.1f}-{100*min(1,c+h):.1f}]"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _numeq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def _canon(o):
    return "gte" if o == "ge" else o


def cmp_ops(gold: dict, model) -> tuple[dict, bool]:
    if not isinstance(model, dict):
        return {"type": False}, False
    t = gold["type"]
    f = {"type": model.get("type") == t}
    if t == "best_under_constraint":
        f["target"] = model.get("target") == gold["target"]
        f["target_dir"] = model.get("target_dir") == gold["target_dir"]
        f["constraint"] = model.get("constraint") == gold["constraint"]
        f["op"] = _canon(str(model.get("op"))) == gold["op"]
        f["threshold"] = _numeq(model.get("threshold"), gold["threshold"])
    elif t == "threshold_filter":
        g = sorted(gold["conditions"], key=lambda c: c["metric"])
        m = sorted((model.get("conditions") or []), key=lambda c: str(c.get("metric")))
        ok = len(g) == len(m)
        for gc, mc in zip(g, m):
            ok = ok and mc.get("metric") == gc["metric"] and _canon(str(mc.get("op"))) == gc["op"] and _numeq(mc.get("T"), gc["T"])
        f["conditions"] = ok
    elif t == "tradeoff_summary":
        f["metrics"] = {model.get("m1"), model.get("m2")} == {gold["m1"], gold["m2"]}
        f["dirs"] = model.get("d1") == gold["d1"] and model.get("d2") == gold["d2"]
    return f, all(f.values())


def _ans_ok(rows, metrics, ex: Example, is_tradeoff: bool) -> bool:
    if sorted(rows) != sorted(ex.gold_answer.rows):
        return False
    return (set(metrics) == set(ex.gold_answer.metrics)) if is_tradeoff else True


def _engine(table, op, rows_universe):
    """Run the engine on op with a given universe; return (GoldAnswer|None, errored)."""
    try:
        return compute_answer(table, {**op, "rows": rows_universe}), False
    except Exception:  # noqa: BLE001
        return None, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/eval_dev.v0_1_0.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="models/qwen3-4b-sft-exec")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--baseline", default="results/p3_sft_4b_dev.json",
                    help="prior run (no executor) to pair on example_id for McNemar")
    ap.add_argument("--out", default="results/p3_4b_exec_dev.json")
    args = ap.parse_args()

    recs = load_jsonl(Path(args.dataset))
    if args.limit:
        recs = recs[: args.limit]
    sources = [Example.model_validate(r) for r in recs]
    n = len(sources)

    print(f"Loading {args.model} + adapter {args.adapter} ...", flush=True)
    gen = LocalGenerator(model_id=args.model, adapter=args.adapter, max_new_tokens=args.max_new_tokens)
    print(f"Scoring {n} examples from {args.dataset}\n", flush=True)

    agg = Counter()
    field_ok, field_tot = defaultdict(Counter), defaultdict(Counter)
    by_type = defaultdict(lambda: Counter())   # type -> counters
    type_n = Counter()
    details = []
    t0 = time.time()

    for i, ex in enumerate(sources):
        eid = ex.metadata.get("example_id", ex.table_id)
        qt = ex.question_type.value
        type_n[qt] += 1
        is_tr = qt == "tradeoff_summary"
        gold_op = operation_dict(ex)
        uni = entity_universe(ex.table)

        out = render(ex, generate_fn=gen.generate)
        # Instrumentation only (no verdict effect): keep the raw output + reconstruction
        # error so a single scored run is self-documenting - lets us classify non-parsed
        # cases post-hoc (e.g. missing `conclude` step vs JSON malformation vs null cite)
        # without a second pass over the held-out set.
        rec = {"example_id": eid, "type": qt, "parsed": out.ok,
               "render_error": out.error, "raw": out.raw}

        # gold-op ceiling (no model) - sanity, should be ~100%
        ce, _ = _engine(ex.table, gold_op, uni)
        agg["ceiling_ok"] += int(ce is not None and _ans_ok(ce.rows, ce.metrics, ex, is_tr))

        model_op = extract_operation(out.raw) if out.raw else None
        rec["op_present"] = isinstance(model_op, dict)
        agg["parsed"] += int(out.ok)
        agg["op_present"] += int(rec["op_present"])

        # op-extraction
        fields, whole = cmp_ops(gold_op, model_op)
        rec["whole_op"] = whole
        agg["whole_op"] += int(whole)
        by_type[qt]["whole_op"] += int(whole)
        for k, v in fields.items():
            field_tot[qt][k] += 1
            field_ok[qt][k] += int(v)

        # engine answer (deterministic universe), strict + fallback
        eng, eng_err = _engine(ex.table, model_op, uni) if rec["op_present"] else (None, True)
        eng_ok = bool(eng is not None and _ans_ok(eng.rows, eng.metrics, ex, is_tr))
        rec["engine_ok"], rec["engine_err"] = eng_ok, eng_err
        agg["engine_strict"] += int(eng_ok)
        by_type[qt]["engine"] += int(eng_ok)

        # all-rows universe ablation
        eng_all, _ = _engine(ex.table, model_op, list(range(len(ex.table.rows)))) if rec["op_present"] else (None, True)
        agg["engine_allrows"] += int(eng_all is not None and _ans_ok(eng_all.rows, eng_all.metrics, ex, is_tr))

        # model's own final_answer
        model_ok = False
        if out.ok:
            try:
                fa = extract_json(out.raw).get("final_answer") or {}
                model_ok = _ans_ok([int(r) for r in fa.get("rows", [])], fa.get("metrics", []), ex, is_tr)
            except Exception:  # noqa: BLE001
                model_ok = False
        rec["model_ok"] = model_ok
        agg["model_ans"] += int(model_ok)
        agg["fallback"] += int(eng_ok if not eng_err else model_ok)
        agg["fallback_used"] += int(eng_err)

        # valid/grounded via validator on the ENGINE-substituted example
        valid = grounded = False
        if out.ok and eng is not None:
            sub = out.example.model_copy()
            sub.gold_answer = GoldAnswer(label=eng.label, rows=list(eng.rows), metrics=list(eng.metrics))
            try:
                vr = validate(sub)
                valid, grounded = vr.valid, bool(vr.checks.get("cells_exist", False))
            except Exception:  # noqa: BLE001
                pass
        rec["valid"], rec["grounded"] = valid, grounded
        agg["valid"] += int(valid)
        agg["grounded"] += int(grounded)

        # model-vs-engine 2x2 cells
        agg[f"mve_{int(model_ok)}{int(eng_ok)}"] += 1

        details.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    # paired McNemar vs prior baseline (join on example_id)
    base_pair = None
    bp = Path(args.baseline)
    if bp.exists():
        base = {d["example_id"]: bool(d.get("answer_correct")) for d in json.loads(bp.read_text())["details"]}
        b = c = both = neither = 0
        for d in details:
            if d["example_id"] in base:
                be, ee = base[d["example_id"]], d["engine_ok"]
                b += int(be and not ee); c += int(ee and not be)
                both += int(be and ee); neither += int(not be and not ee)
        base_pair = {"baseline_only": b, "exec_only": c, "both": both, "neither": neither,
                     "mcnemar_p": mcnemar_exact(b, c)}

    pc = lambda k: wilson(agg[k], n)
    print(f"\n=== EXECUTOR EVAL - {args.adapter} on {args.dataset} (n={n}) ===")
    print(f"  gold-op CEILING (sanity)     : {pc('ceiling_ok')}")
    print(f"  parse rate                   : {pc('parsed')}")
    print(f"  operation present            : {pc('op_present')}")
    print(f"  whole-op exact-match         : {pc('whole_op')}")
    print(f"  ENGINE answer (strict)       : {pc('engine_strict')}")
    print(f"  engine answer (with fallback): {pc('fallback')}  (fallback used {agg['fallback_used']})")
    print(f"  engine answer (ALL-ROWS abl.): {pc('engine_allrows')}")
    print(f"  model's own final_answer     : {pc('model_ans')}")
    print(f"  VALID (engine-substituted)   : {pc('valid')}")
    print(f"  grounded                     : {pc('grounded')}")
    print(f"\n  model-vs-engine 2x2 (model,engine): "
          f"both={agg['mve_11']} engine_only={agg['mve_01']} model_only={agg['mve_10']} neither={agg['mve_00']}")
    print(f"  exact McNemar (model vs engine) p = {mcnemar_exact(agg['mve_10'], agg['mve_01']):.4g}")
    if base_pair:
        print(f"  vs baseline {bp.name}: exec_only={base_pair['exec_only']} baseline_only={base_pair['baseline_only']} "
              f"both={base_pair['both']} neither={base_pair['neither']}  McNemar p={base_pair['mcnemar_p']:.4g}")
    print("\n  by type (whole-op / engine / n):")
    for t in sorted(type_n):
        print(f"    {t:<22} whole-op {by_type[t]['whole_op']}/{type_n[t]}  engine {by_type[t]['engine']}/{type_n[t]}")
    print("\n  per-field op accuracy by type:")
    for t in sorted(field_tot):
        print(f"    {t:<22} " + "  ".join(f"{k}={field_ok[t][k]}/{field_tot[t][k]}" for k in field_tot[t]))
    print(f"\n  wall {time.time()-t0:.0f}s")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "model": args.model, "adapter": args.adapter, "dataset": args.dataset, "n": n,
        "agg": dict(agg), "by_type": {t: dict(by_type[t]) for t in by_type},
        "type_n": dict(type_n),
        "field_ok": {t: dict(field_ok[t]) for t in field_ok},
        "field_tot": {t: dict(field_tot[t]) for t in field_tot},
        "baseline_pair": base_pair, "details": details,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {outp.relative_to(ROOT) if outp.is_absolute() else outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
