"""(a) Validator-gated best-of-N probe on a dev subset (P3.6).

Measures, on a stratified 45-example dev subset (15/type, fixed seed):
  - valid@1        : greedy fully-valid rate (baseline)
  - oracle pass@N  : fraction where ANY of N sampled traces is fully valid
                     (DIAGNOSTIC: is the correct answer in the model's distribution?)
  - majority vote  : gold-free self-consistency - most common final answer among
                     parsed samples; is it answer-correct?  (DEPLOYABLE)
  - grounded-gated : majority vote restricted to schema_valid+grounded samples (DEPLOYABLE)

Greedy + N=8 @ T=0.8 top_p=0.95. bf16 base + LoRA (matches eval_model). Run from project root.
Result 2026-06-24 (Qwen3-4B): greedy ans 55.6%, oracle pass@8 ans 88.9%, majority 57.8% -> no
cheap decoding win; the right answer is usually a MINORITY sample. Motivates the executor path.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.getcwd())

from src.llm_renderer import build_prompt, extract_json, reconstruct_example  # noqa: E402
from src.local_infer import LocalGenerator                                     # noqa: E402
from src.schema import Example                                                 # noqa: E402
from src.splits import load_jsonl                                              # noqa: E402
from src.trace_validator import validate                                       # noqa: E402
from pathlib import Path

DEV = "data/processed/eval_dev.v0_1_0.jsonl"
MODEL, ADAPTER = "Qwen/Qwen3-4B", "models/qwen3-4b-sft"
OUT = "results/p3_4b_best_of_n.json"
N, TEMP, TOP_P, PER_TYPE, SEED = 8, 0.8, 0.95, 15, 0

def _id(ex): return ex.metadata.get("example_id", ex.table_id)
def _qt(ex): return ex.question_type.value if hasattr(ex.question_type, "value") else str(ex.question_type)

# ---- stratified subset (deterministic) ----
recs = load_jsonl(Path(DEV))
src = [Example.model_validate(r) for r in recs]
import random
rng = random.Random(SEED)
bytype = defaultdict(list)
for ex in src:
    bytype[_qt(ex)].append(ex)
subset = []
for t in sorted(bytype):
    pool = sorted(bytype[t], key=_id)
    rng.shuffle(pool)
    subset += pool[:PER_TYPE]
print(f"subset: {len(subset)} ({ {t: min(PER_TYPE, len(bytype[t])) for t in bytype} })", flush=True)

gen = LocalGenerator(model_id=MODEL, adapter=ADAPTER, max_new_tokens=1536)
import torch

def verdict(source, raw):
    """raw text -> dict of per-sample checks (gold-free + oracle), plus answer key."""
    d = {"parsed": False, "schema_valid": False, "grounded": False,
         "valid": False, "answer_correct": False, "key": None}
    try:
        ex = reconstruct_example(source, extract_json(raw))
    except Exception:
        return d
    d["parsed"] = True
    try:
        res = validate(ex)
    except Exception:
        return d
    fa = ex.gold_answer
    d.update(schema_valid=bool(res.checks.get("schema_valid")),
             grounded=bool(res.checks.get("cells_exist")),
             valid=bool(res.valid),
             answer_correct=bool(res.checks.get("answer_correct")),
             key=json.dumps([fa.label, sorted(fa.rows), sorted(fa.metrics)], ensure_ascii=False))
    return d

def sample_n(prompt, n):
    text = gen._format(prompt)
    inp = gen.tokenizer(text, return_tensors="pt").to(gen.model.device)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = gen.model.generate(**inp, max_new_tokens=1536, do_sample=True,
                                 temperature=TEMP, top_p=TOP_P, num_return_sequences=n,
                                 pad_token_id=gen.tokenizer.eos_token_id)
    plen = inp["input_ids"].shape[1]
    return [gen.tokenizer.decode(o[plen:], skip_special_tokens=True).strip() for o in out]

cases = []
for i, ex in enumerate(subset):
    prompt = build_prompt(ex)
    greedy = verdict(ex, gen.generate(prompt))                       # @1 baseline
    samples = [verdict(ex, r) for r in sample_n(prompt, N)]          # best-of-N pool
    parsed = [s for s in samples if s["parsed"]]
    grounded_ok = [s for s in parsed if s["schema_valid"] and s["grounded"]]
    def vote(pool):
        if not pool: return None
        c = Counter(s["key"] for s in pool)
        topkey = c.most_common(1)[0][0]
        rep = next(s for s in pool if s["key"] == topkey)
        return rep["answer_correct"]
    cases.append({
        "id": _id(ex), "type": _qt(ex),
        "greedy_valid": greedy["valid"], "greedy_answer": greedy["answer_correct"],
        "n_valid": sum(s["valid"] for s in samples),
        "n_answer_correct": sum(s["answer_correct"] for s in samples),
        "oracle_passN": any(s["valid"] for s in samples),
        "oracle_passN_answer": any(s["answer_correct"] for s in samples),
        "majority_answer_correct": vote(parsed),
        "grounded_majority_answer_correct": vote(grounded_ok),
    })
    print(f"  {i+1}/{len(subset)} {_id(ex)}  greedyV={greedy['valid']} nValid={cases[-1]['n_valid']}/{N} passN={cases[-1]['oracle_passN']}", flush=True)

n = len(cases)
def freq(pred): return sum(1 for c in cases if pred(c))
summary = {
    "n": n, "N": N, "temp": TEMP,
    "valid@1_greedy": freq(lambda c: c["greedy_valid"]),
    "answer@1_greedy": freq(lambda c: c["greedy_answer"]),
    "oracle_passN_valid": freq(lambda c: c["oracle_passN"]),
    "oracle_passN_answer": freq(lambda c: c["oracle_passN_answer"]),
    "majority_answer_correct": freq(lambda c: c["majority_answer_correct"]),
    "grounded_majority_answer_correct": freq(lambda c: c["grounded_majority_answer_correct"]),
}
Path(OUT).write_text(json.dumps({"summary": summary, "cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n=== SUMMARY (counts out of %d) ===" % n, flush=True)
for k, v in summary.items():
    if k not in ("n", "N", "temp"):
        print(f"  {k:34} {v:3d}  ({100.0*v/n:.1f}%)", flush=True)
print(f"wrote {OUT}", flush=True)
