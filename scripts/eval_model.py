"""Score a local model's grounded traces against gold via the validator (P3.1/P3.4).

Reuses the render -> reconstruct_example -> validate pipeline with a LocalGenerator
in place of headless Claude, and prints the same metric block as
``scripts/evaluate.py`` plus a per-type breakdown. Rates are over ALL examples
(an unparseable output scores 0 on every metric) so the baseline is honest.

    # baseline (base model):
    python scripts/eval_model.py data/processed/eval_dev.v0_1_0.jsonl
    # trained model:
    python scripts/eval_model.py data/processed/eval_dev.v0_1_0.jsonl --adapter models/qwen3-1.7b-sft
    # quick smoke:
    python scripts/eval_model.py data/processed/eval_dev.v0_1_0.jsonl --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm_renderer import render  # noqa: E402
from src.local_infer import DEFAULT_MODEL, LocalGenerator  # noqa: E402
from src.schema import Example  # noqa: E402
from src.splits import ROOT, load_jsonl  # noqa: E402
from src.trace_validator import validate  # noqa: E402


def _qt(ex: Example) -> str:
    return ex.question_type.value if hasattr(ex.question_type, "value") else str(ex.question_type)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (trained model)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--out", default=None, help="optional JSON path for per-example results")
    args = ap.parse_args()

    recs = load_jsonl(Path(args.dataset))
    if args.limit:
        recs = recs[: args.limit]
    sources = [Example.model_validate(r) for r in recs]
    n = len(sources)

    tag = f"{args.model}" + (f" + adapter {args.adapter}" if args.adapter else " (base)")
    print(f"Loading {tag} ...", flush=True)
    gen = LocalGenerator(model_id=args.model, adapter=args.adapter, max_new_tokens=args.max_new_tokens)
    print(f"Scoring {n} examples from {args.dataset}\n", flush=True)

    m = Counter()  # parsed, schema_valid, answer_correct, grounded, valid
    by_type = defaultdict(lambda: [0, 0])  # type -> [count, answer_correct]
    fail = Counter()
    details = []
    t0 = time.time()

    for i, ex in enumerate(sources):
        out = render(ex, generate_fn=gen.generate)
        qt = _qt(ex)
        by_type[qt][0] += 1
        rec = {"example_id": ex.metadata.get("example_id", ex.table_id), "type": qt, "parsed": out.ok}
        if out.ok:
            m["parsed"] += 1
            # Untrusted model output can carry out-of-range indices etc. that make the
            # validator (built for trusted gold) raise. A crash must not kill the run -
            # an un-validatable answer is simply invalid.
            try:
                res = validate(out.example)
            except Exception as e:  # noqa: BLE001
                rec.update(valid=False, answer_correct=False, grounded=False,
                           failed=["validator_error"], error=f"validate: {type(e).__name__}: {e}")
                fail["validator_error"] += 1
            else:
                ac = bool(res.checks.get("answer_correct", False))
                rec.update(
                    valid=res.valid,
                    answer_correct=ac,
                    grounded=bool(res.checks.get("cells_exist", False)),
                    failed=res.failed_checks(),
                )
                m["schema_valid"] += int(bool(res.checks.get("schema_valid", False)))
                m["answer_correct"] += int(ac)
                m["grounded"] += int(bool(res.checks.get("cells_exist", False)))
                m["valid"] += int(res.valid)
                by_type[qt][1] += int(ac)
                for c in res.failed_checks():
                    fail[c] += 1
        else:
            rec["error"] = out.error
            fail["unparseable"] += 1
        details.append(rec)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}   ({time.time() - t0:.0f}s)", flush=True)

    def pct(k):
        return 100.0 * m[k] / n if n else 0.0

    print(f"\nResults - {tag}")
    print(f"  examples              : {n}")
    print(f"  parse rate            : {pct('parsed'):.1f}% ({m['parsed']}/{n})")
    print(f"  JSON validity rate    : {pct('schema_valid'):.1f}% ({m['schema_valid']}/{n})")
    print(f"  Answer accuracy       : {pct('answer_correct'):.1f}% ({m['answer_correct']}/{n})")
    print(f"  Groundedness score    : {pct('grounded'):.1f}% ({m['grounded']}/{n})")
    print(f"  Trace correctness     : {pct('valid'):.1f}% ({m['valid']}/{n})")
    print("  Answer accuracy by type:")
    for t in sorted(by_type):
        cnt, ok = by_type[t]
        print(f"      {t:<22} {ok}/{cnt} ({100.0 * ok / cnt if cnt else 0:.0f}%)")
    print(f"  Failure categories    : {dict(fail) if fail else 'none'}")
    print(f"  wall time             : {time.time() - t0:.0f}s")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "adapter": args.adapter,
                    "dataset": args.dataset,
                    "n": n,
                    "metrics": {k: m[k] for k in ("parsed", "schema_valid", "answer_correct", "grounded", "valid")},
                    "by_type": {t: by_type[t] for t in by_type},
                    "failures": dict(fail),
                    "details": details,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n  per-example results -> {outp.relative_to(ROOT) if outp.is_absolute() else outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
