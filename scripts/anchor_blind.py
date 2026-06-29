"""Independent in-distribution ground-truth anchor (P3.8) - break the circularity.

The 95.7% locked-test number is scored against SPEC-DERIVED gold computed by code that shares
the engine's arithmetic, so it measures comprehension-consistency, not correctness vs an outside
oracle. This builds a NON-CIRCULAR check: a blind annotator (no spec, no gold, no engine) derives
the answer from the RAW table + the natural-language question alone, and we compare to the stored
gold. Agreement = evidence the gold is genuinely correct (not just self-consistent); each
DISAGREEMENT is a candidate gold error to be human-adjudicated.

Two subcommands:
  build  - stratified random sample of in-distribution eval_test examples -> blind cards
           (id/gold/spec stripped) + a private key (card_id -> example_id, gold, engine_ok).
  score  - given the annotator's answers (card_id -> [selected row labels]), compare to gold,
           report agreement + Wilson CI per type, and list every disagreement verbatim.

    python scripts/anchor_blind.py build --n 36 --seed 0
    python scripts/anchor_blind.py score --answers results/anchor/answers.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import Example                 # noqa: E402
from src.splits import load_jsonl              # noqa: E402
from src.table_utils import render_markdown    # noqa: E402

TEST = "data/processed/eval_test.v0_1_0.jsonl"
ENGINE_RESULTS = "results/p3_4b_exec_TEST.json"
CARDS = "results/anchor/cards.json"
KEY = "results/anchor/key.json"
REPORT = "results/anchor/report.json"

TRAINED = ["best_under_constraint", "threshold_filter", "tradeoff_summary"]


def _row_labels(ex: Example) -> list[str]:
    t = ex.table
    return list(t.row_labels) if t.row_labels else [str(r[0]) for r in t.rows]


def _gold_label_set(ex: Example) -> set[str]:
    """The gold answer expressed as a SET of selected row labels (or {'none'})."""
    if not ex.gold_answer.rows:
        return {"none"}
    labels = _row_labels(ex)
    return {_norm(labels[r]) for r in ex.gold_answer.rows}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .,:;")


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100*p:.1f}% [{100*max(0,c-h):.1f}-{100*min(1,c+h):.1f}]"


def build(args) -> int:
    recs = load_jsonl(Path(TEST))
    exs = [Example.model_validate(r) for r in recs]
    by_type = defaultdict(list)
    for ex in exs:
        qt = ex.question_type.value
        if qt in TRAINED:
            by_type[qt].append(ex)

    engine_ok = {}
    ep = Path(ENGINE_RESULTS)
    if ep.exists():
        for d in json.loads(ep.read_text(encoding="utf-8"))["details"]:
            engine_ok[d["example_id"]] = bool(d.get("engine_ok"))

    rng = random.Random(args.seed)
    per = args.n // len(TRAINED)
    cards, key = [], []
    cid = 0
    for qt in TRAINED:
        pool = sorted(by_type[qt], key=lambda e: e.metadata.get("example_id", e.table_id))
        chosen = rng.sample(pool, min(per, len(pool)))
        for ex in chosen:
            eid = ex.metadata.get("example_id", ex.table_id)
            labels = _row_labels(ex)
            cards.append({
                "card_id": cid,
                "type": qt,                       # kept: the annotator IS told the task family
                "question": ex.question,
                "table_markdown": render_markdown(ex.table),
                "row_labels": labels,             # the choices; answer must be a subset (or "none")
            })
            key.append({
                "card_id": cid,
                "example_id": eid,
                "gold_label": ex.gold_answer.label,
                "gold_rows": list(ex.gold_answer.rows),
                "gold_label_set": sorted(_gold_label_set(ex)),
                "engine_ok": engine_ok.get(eid),
            })
            cid += 1

    Path(CARDS).parent.mkdir(parents=True, exist_ok=True)
    Path(CARDS).write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(KEY).write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cards)} blind cards -> {CARDS}")
    print(f"wrote private key            -> {KEY}")
    print(f"by type: { {qt: sum(c['type']==qt for c in cards) for qt in TRAINED} }")
    return 0


def score(args) -> int:
    cards = {c["card_id"]: c for c in json.loads(Path(CARDS).read_text(encoding="utf-8"))}
    key = {k["card_id"]: k for k in json.loads(Path(KEY).read_text(encoding="utf-8"))}
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    # answers: { "<card_id>": ["label", ...] }  (use ["none"] for no-match)

    agree_by = defaultdict(lambda: [0, 0])  # type -> [agree, total]
    disagreements = []
    for cid_s, picked in answers.items():
        cid = int(cid_s)
        k, c = key[cid], cards[cid]
        oracle_set = {_norm(x) for x in (picked or [])} or {"none"}
        if oracle_set == {"none"} or picked == []:
            oracle_set = {"none"}
        gold_set = set(k["gold_label_set"])
        ok = oracle_set == gold_set
        agree_by[c["type"]][0] += int(ok)
        agree_by[c["type"]][1] += 1
        if not ok:
            disagreements.append({
                "card_id": cid, "example_id": k["example_id"], "type": c["type"],
                "question": c["question"], "oracle": sorted(oracle_set),
                "gold": sorted(gold_set), "engine_ok": k["engine_ok"],
            })

    tot_a = sum(v[0] for v in agree_by.values())
    tot_n = sum(v[1] for v in agree_by.values())
    print(f"\n=== INDEPENDENT-LLM BLIND CROSS-CHECK vs stored gold (n={tot_n}) ===")
    print(f"  OVERALL agreement: {wilson(tot_a, tot_n)}")
    for qt in TRAINED:
        a, n = agree_by[qt]
        print(f"    {qt:<22} {wilson(a, n)}  ({a}/{n})")
    print(f"\n  disagreements (candidate gold errors -> human adjudication): {len(disagreements)}")
    for d in disagreements:
        print(f"    [{d['type']}] card {d['card_id']} ({d['example_id']}) "
              f"oracle={d['oracle']} gold={d['gold']} engine_ok={d['engine_ok']}")
        print(f"       Q: {d['question']}")

    Path(REPORT).write_text(json.dumps({
        "n": tot_n, "agree": tot_a,
        "by_type": {qt: {"agree": agree_by[qt][0], "n": agree_by[qt][1]} for qt in TRAINED},
        "disagreements": disagreements,
        "note": "oracle = blind independent-LLM annotation from raw table + question only; "
                "NOT human-verified; disagreements pending human adjudication",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {REPORT}")
    return 0


def score_human(args) -> int:
    """Score BLIND HUMAN annotations (from annotate_build.py's annotate.html export).
    Compares by ROW INDEX vs gold_rows (exact, duplicate-label-safe) and surfaces ambiguous
    flags + notes. Honest provenance: human, blind."""
    key = {k["card_id"]: k for k in json.loads(Path(args.key).read_text(encoding="utf-8"))}
    ans = json.loads(Path(args.answers).read_text(encoding="utf-8"))

    agree_by = defaultdict(lambda: [0, 0])
    disagreements, ambiguous = [], []
    for cid_s, a in ans.items():
        cid = int(cid_s)
        if cid not in key:
            continue
        k = key[cid]
        if a.get("ambiguous"):
            ambiguous.append({"card_id": cid, "example_id": k["example_id"], "note": a.get("note", "")})
            continue
        human_rows = set() if a.get("none") else set(a.get("rows", []))
        gold_rows = set(k["gold_rows"])
        ok = human_rows == gold_rows
        agree_by["all"][0] += int(ok)
        agree_by["all"][1] += 1
        if not ok:
            disagreements.append({"card_id": cid, "example_id": k["example_id"],
                                  "human_rows": sorted(human_rows), "gold_rows": sorted(gold_rows),
                                  "engine_ok": k["engine_ok"], "note": a.get("note", "")})

    a, n = agree_by["all"]
    print(f"\n=== BLIND HUMAN ANCHOR vs stored gold (scored n={n}, ambiguous-skipped={len(ambiguous)}) ===")
    print(f"  agreement (by row index): {wilson(a, n)}  ({a}/{n})")
    print(f"\n  disagreements (candidate gold errors -> adjudicate): {len(disagreements)}")
    for d in disagreements:
        print(f"    card {d['card_id']} ({d['example_id']}) human_rows={d['human_rows']} "
              f"gold_rows={d['gold_rows']} engine_ok={d['engine_ok']}  note={d['note']!r}")
    if ambiguous:
        print(f"\n  flagged AMBIGUOUS / ill-posed (question quality): {len(ambiguous)}")
        for x in ambiguous:
            print(f"    card {x['card_id']} ({x['example_id']})  note={x['note']!r}")

    out = Path("results/anchor/report_human.json")
    out.write_text(json.dumps({
        "scored": n, "agree": a, "ambiguous_skipped": len(ambiguous),
        "disagreements": disagreements, "ambiguous": ambiguous,
        "note": "oracle = BLIND HUMAN annotation from raw table + question only (annotate.html); "
                "compared by row index; honest human anchor",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=36)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=build)
    s = sub.add_parser("score")
    s.add_argument("--answers", required=True)
    s.set_defaults(func=score)
    h = sub.add_parser("score-human")
    h.add_argument("--answers", required=True)
    h.add_argument("--key", default="results/anchor/key_human.json")
    h.set_defaults(func=score_human)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
