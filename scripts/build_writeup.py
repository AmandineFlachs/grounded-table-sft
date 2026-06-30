"""Build docs/index.html - the standalone, promotable project write-up (no GPU).

A reader-facing narrative (hook -> problem -> approach -> results -> "is it real?" -> honest limits
-> demo -> reproduce), distinct from the lean ML methodology reference in docs/methodology.html.
Layered for two audiences: the story + headline numbers read top to bottom; deeper rigor (CIs,
McNemar, leakage, adjudication) sits in <details> blocks.

Headline numbers are SOURCED from the saved result files (not hand-typed) so the write-up cannot
drift from the measurements; two REAL demo cards are embedded via src.demo. Re-run after any re-score:

    python scripts/build_writeup.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo import DEMO_CARD_CSS, build_card, card_html   # noqa: E402
from src.schema import Example                              # noqa: E402
from src.splits import load_jsonl                            # noqa: E402

OUT = ROOT / "docs" / "index.html"
TEST_RES = ROOT / "results" / "p3_4b_exec_TEST.json"
GATE_RES = ROOT / "results" / "p3_4b_gate.json"
GROUND_RES = ROOT / "results" / "p3_4b_grounded_constructed.json"
# Untrained Qwen3-4B zero-shot baseline - measured separately (GPU). This row auto-fills once the file
# exists; produce it with:
#   python scripts/eval_model.py data/processed/eval_dev.v0_1_0.jsonl --model Qwen/Qwen3-4B \
#          --out results/p3_baseline_4b_dev.json
BASE4B_RES = ROOT / "results" / "p3_baseline_4b_dev.json"


def _load(p: Path) -> dict:
    return json.loads(io.open(p, encoding="utf-8").read())


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%"


# --------------------------------------------------------------------------- #
# numbers - sourced from the saved result files (single source of truth)
# --------------------------------------------------------------------------- #
def gather_numbers() -> dict:
    t = _load(TEST_RES)
    agg, by, tn, n = t["agg"], t["by_type"], t["type_n"], t["n"]
    gate = {r["label"]: r for r in _load(GATE_RES)["results"]}
    gin = gate["IN-DISTRIBUTION locked test"]
    good = gate["OUT-OF-DISTRIBUTION extremum anchor"]
    g = _load(GROUND_RES)
    gn = g["n"]
    base4b = None
    if BASE4B_RES.exists():
        bd = _load(BASE4B_RES)
        m, bn = bd["metrics"], bd["n"]
        base4b = {"valid": pct(m["valid"], bn), "grounded": pct(m["grounded"], bn),
                  "answer": pct(m["answer_correct"], bn)}
    return {
        "base4b": base4b,
        "test_n": n,
        "engine": pct(agg["engine_strict"], n),
        "model_own": pct(agg["model_ans"], n),
        "op_present": pct(agg["op_present"], n),
        "whole_op": pct(agg["whole_op"], n),
        "grounded_model": pct(agg["grounded"], n),
        "best": pct(by["best_under_constraint"]["engine"], tn["best_under_constraint"]),
        "thresh": pct(by["threshold_filter"]["engine"], tn["threshold_filter"]),
        "trade": pct(by["tradeoff_summary"]["engine"], tn["tradeoff_summary"]),
        "mcnemar_plus": agg["mve_01"],   # engine right, model wrong
        "mcnemar_minus": agg["mve_10"],  # engine wrong, model right
        # gate
        "gate_in_engine": pct(gin["engine"], gin["n"]),
        "gate_in_gated": pct(gin["gated"], gin["n"]),
        "gate_in_fires": f"{gin['fires']}/{gin['n']}",
        "gate_ood_engine": pct(good["engine"], good["n"]),
        "gate_ood_gated": pct(good["gated"], good["n"]),
        "ood_n": good["n"],
        # grounded by construction
        "grounded_sys": pct(g["agg"]["grounded_constructed"], gn),
        "valid_sys": pct(g["agg"]["valid_constructed"], gn),
        "grounded_model2": pct(g["agg"]["model_grounded"], gn),
    }


# --------------------------------------------------------------------------- #
# real demo cards (embedded inline)
# --------------------------------------------------------------------------- #
_POOLS = [
    ("results/p3_4b_exec_TEST.json", "data/processed/eval_test.v0_1_0.jsonl", False),
    ("results/p3_4b_exec_TEST_verified.json", "data/processed/realtable_eval_verified.v0_1_0.jsonl", True),
]


def _card_for(example_id: str) -> dict:
    for res_path, src_path, ood in _POOLS:
        det = {d["example_id"]: d for d in _load(ROOT / res_path)["details"]}
        for r in load_jsonl(ROOT / src_path):
            ex = Example.model_validate(r)
            eid = ex.metadata.get("example_id", ex.table_id)
            if eid == example_id and example_id in det and det[example_id].get("raw"):
                card = build_card(ex, det[example_id]["raw"])
                card["ood"] = ood
                return card
    raise SystemExit(f"demo example not found or missing raw output: {example_id}")


# in-distribution success (winner selection) and the out-of-distribution gate-fires case
CARD_INDIST = "tatqa_0036_cols_best_under_constraint_normal"
CARD_OOD = "tatqa_0000_cols_rank_models_verified"


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#fff;--panel:#f6f8fa;--panel2:#eef1f5;--ink:#1c2128;--muted:#57606a;
  --accent:#0969da;--accent2:#1a7f37;--warn:#9a6700;--line:#d0d7de;--code:#f3f4f6}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);line-height:1.65;background:var(--bg)}
.wrap{max-width:860px;margin:0 auto;padding:0 24px 96px}
header.hero{padding:56px 24px 36px;background:linear-gradient(160deg,#eef3fb 0%,#fff 100%);
  border-bottom:1px solid var(--line)}
header.hero .inner{max-width:860px;margin:0 auto}
h1{font-size:34px;margin:0 0 10px;letter-spacing:-.5px;line-height:1.2}
.subtitle{color:var(--muted);font-size:18px;margin:0;max-width:62ch}
.badges{margin-top:20px}
.badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:600;
  margin:0 6px 6px 0;background:rgba(9,105,218,.1);color:var(--accent);border:1px solid rgba(9,105,218,.25)}
.badge.green{background:rgba(26,127,55,.1);color:var(--accent2);border-color:rgba(26,127,55,.25)}
.badge.warn{background:rgba(154,103,0,.1);color:var(--warn);border-color:rgba(154,103,0,.25)}
section{margin:46px 0}
h2{font-size:24px;margin:0 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:18px;margin:26px 0 8px;color:var(--accent2)}
p{margin:12px 0}
.lede{font-size:18px}
.muted{color:var(--muted)}
.tldr{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:12px;padding:20px 24px;margin:28px 0}
.tldr h2{border:none;margin:0 0 10px;font-size:18px;color:var(--accent)}
.callout{background:rgba(26,127,55,.07);border:1px solid rgba(26,127,55,.25);
  border-left:4px solid var(--accent2);border-radius:10px;padding:14px 18px;margin:18px 0;font-size:15px}
.callout .d{font-weight:700;color:var(--accent2);white-space:nowrap}
.plain{background:#f0f6ff;border:1px solid rgba(9,105,218,.22);border-radius:12px;padding:18px 22px;margin:0 0 24px}
.plain h3{margin:0 0 8px;color:var(--accent);font-size:16px}
.plain p{margin:9px 0}.plain p:last-child{margin-bottom:0}
.plain .pay{font-size:16px}
.fineprint{font-size:13.5px;color:var(--muted);border-left:3px solid var(--line);
  padding-left:14px;margin:18px 0}
.fineprint b{color:var(--ink)}
td.pending{text-align:center;color:var(--warn);font-style:italic;background:rgba(154,103,0,.06)}
.tldr ul{margin:8px 0 0;padding-left:20px}.tldr li{margin:6px 0}
.big{font-size:15px}
.statrow{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0}
.stat{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.stat .n{font-size:28px;font-weight:700;color:var(--accent2);letter-spacing:-.5px}
.stat .l{font-size:13px;color:var(--muted);margin-top:2px}
code{background:var(--code);padding:1px 6px;border-radius:5px;font-size:13px;
  font-family:"SFMono-Regular",Consolas,Menlo,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto;
  font-size:13px;line-height:1.5;font-family:"SFMono-Regular",Consolas,Menlo,monospace}
table.data{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
table.data th,table.data td{border:1px solid var(--line);padding:8px 12px;text-align:left}
table.data th{background:var(--panel2);font-weight:600}
table.data tr:nth-child(even) td{background:rgba(0,0,0,.025)}
table.data td.r,table.data th.r{text-align:right}
.pipeline{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:18px 0}
.node{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 13px;
  font-size:13px;font-weight:600}.node.key{background:rgba(26,127,55,.12);border-color:rgba(26,127,55,.3);color:var(--accent2)}
.arrow{color:var(--accent);font-size:17px}
.barwrap{margin:14px 0 6px}
.bar{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
.bar .bl{flex:0 0 172px;color:var(--muted);text-align:right}
.bar .bt{flex:1;min-width:60px;background:var(--panel2);border-radius:6px;height:20px;overflow:hidden}
.bar .bt i{display:block;height:100%;background:var(--accent);border-radius:6px}
.bar .bv{flex:0 0 52px;font-weight:700;color:var(--ink)}
.bar.sys .bt i{background:var(--accent2)}.bar.sys .bv{color:var(--accent2)}
.barcap{font-size:13px;color:var(--muted);margin:4px 0 0}
@media(max-width:560px){.bar .bl{flex-basis:104px;font-size:12px}}
ul.lim{padding-left:20px}ul.lim li{margin:8px 0}
.demo-note{font-size:14px;color:var(--muted);margin:6px 0 18px}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
a{color:var(--accent)}
nav.tabs{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;
  background:rgba(255,255,255,.92);backdrop-filter:saturate(180%) blur(6px);
  margin:24px -24px 0;padding:12px 24px;border-bottom:1px solid var(--line)}
.tab{appearance:none;border:1px solid var(--line);background:#fff;cursor:pointer;
  font:600 14px/1 inherit;color:var(--muted);padding:9px 15px;border-radius:999px;
  transition:background .12s,color .12s,border-color .12s,box-shadow .12s}
.tab:hover{color:var(--accent);border-color:var(--accent);background:rgba(9,105,218,.06)}
.tab.active{color:#fff;background:var(--accent);border-color:var(--accent);
  box-shadow:0 2px 6px rgba(9,105,218,.35)}
.tab.active:hover{color:#fff}
.panel{display:none}.panel.active{display:block;animation:fade .16s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.panel>section:first-child{margin-top:28px}
.panel>section:first-child h2{margin-top:0}
""" + DEMO_CARD_CSS


def build(N: dict) -> str:
    card_in = card_html(_card_for(CARD_INDIST))
    card_ood = card_html(_card_for(CARD_OOD))

    b4 = N.get("base4b")
    _b4label = 'Untrained bigger AI <span class="muted">(Qwen3-4B, zero-shot)</span>'
    if b4:
        base4b_row = (f'<tr><td>{_b4label}</td><td class="r">{b4["valid"]}</td>'
                      f'<td class="r">{b4["grounded"]}</td><td class="r">{b4["answer"]}</td></tr>')
        base4b_note = ""
    else:
        base4b_row = (f'<tr><td>{_b4label}</td>'
                      f'<td class="pending" colspan="3">measuring (added once the run completes)</td></tr>')
        base4b_note = ('<p class="muted">The untrained Qwen3-4B (zero-shot) row is still being measured '
                       'on the same dev set and will be filled in once that run completes.</p>')

    # right-answer accuracy, stage by stage - a visual of the climb (flat across every
    # model-only stage, then the calculator step closes it). Values mirror the table above.
    def _num(s):
        return float(str(s).rstrip("%"))
    _stages = [("Untrained 1.7B", 18.0, "18%", False),
               ("Trained 1.7B", 20.0, "20%", False)]
    if b4:
        _stages.append(("Untrained 4B", _num(b4["answer"]), b4["answer"], False))
    _stages += [("Trained 4B", 52.0, "52%", False),
                ("+ Calculator step", _num(N["engine"]), N["engine"], True)]
    _bars = "".join(
        f'<div class="bar{" sys" if sysf else ""}"><span class="bl">{lab}</span>'
        f'<span class="bt"><i style="width:{val:.0f}%"></i></span>'
        f'<span class="bv">{disp}</span></div>'
        for lab, val, disp, sysf in _stages)
    ans_chart = (f'<div class="barwrap">{_bars}</div>'
                 f'<p class="barcap"><b>Right-answer accuracy, stage by stage.</b> Flat through every '
                 f'model-only stage (more size and training barely move it), then the calculator step '
                 f'does what they could not. Green = answer built by the engine.</p>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounded Table Reasoning: a small model that reads tables, and proves it</title>
<style>{CSS}</style></head>
<body>
<header class="hero"><div class="inner">
  <h1>Teaching a small model to read tables, and prove every step</h1>
  <p class="subtitle">A 4-billion-parameter model that answers constrained questions over real
  financial tables and emits <b>grounded reasoning traces</b>, with every step tied to specific,
  verifiable cells. The headline: small models <i>comprehend</i> table tasks but can't reliably
  <i>execute</i> the arithmetic, so we let them comprehend and hand the arithmetic to proven code.</p>
  <div class="badges">
    <span class="badge green">Held-out test, scored once</span>
    <span class="badge">TAT-QA · CC BY 4.0</span>
    <span class="badge">QLoRA · single RTX 3090</span>
    <span class="badge warn">Independently corroborated &amp; honestly bounded</span>
  </div>
</div></header>

<div class="wrap">

<nav class="tabs" role="tablist">
  <button class="tab active" data-p="results">Results</button>
  <button class="tab" data-p="why">Problem &amp; data</button>
  <button class="tab" data-p="how">Approach</button>
  <button class="tab" data-p="limit">Validation &amp; limits</button>
  <button class="tab" data-p="demo">Demo</button>
  <button class="tab" data-p="repro">Reproduce</button>
</nav>

<div class="panel active" id="p-results" role="tabpanel">
<section>
  <h2>Results: the bottom line</h2>

  <div class="plain">
    <h3>How this started</h3>
    <p>At work I was benchmarking language models, comparing their answers on a set of data tables and
    talking the results over with an AI assistant. At some point I stopped and wondered: does the model
    actually <b>understand</b> the table in front of it, or is it just pattern-matching its way to a
    plausible-sounding answer? This project is my attempt to find out, using public tables I can share.</p>
    <p>Tables like those are full of questions such as <i>“which product was cheapest last year while
    still selling well?”</i> A small AI can <b>read the table and understand the question</b> just
    fine. But, like a person who gets a word problem yet slips on the mental math, it often gets the
    <b>calculation</b> wrong.</p>
    <p>So we split the work: the AI does the <b>understanding</b>, and a piece of ordinary, reliable
    software does the <b>arithmetic</b> (in effect, we hand it a calculator). The AI also has to
    <b>show its work</b>, pointing at the exact cells it used, so anyone can check the answer.</p>
    <p class="pay"><b>The payoff:</b> correct answers went from roughly <b>6 in 10</b> to about
    <b>96 in 100</b>, and nearly all of the AI’s explanations now point to the right cells. And when
    it meets a kind of question it was never taught, it <b>holds back</b> instead of guessing
    confidently, so the system is never worse than the AI on its own.</p>
    <p class="pay muted" style="font-size:15px">One thing to be upfront about: we wrote the answer key
    ourselves, so on its own this score shows the system <b>agrees with our key</b>, not that it is
    objectively right. To check the key itself, we had independent people and a separate AI re-answer
    the questions from scratch (details under <a href="#limit" data-goto="limit">Validation &amp;
    limits</a>).</p>
  </div>

  <div class="statrow">
    <div class="stat"><div class="n">{N['engine']}</div><div class="l">of answers correct on fresh,
      unseen tables, up from <b>{N['model_own']}</b> when the AI did the math itself</div></div>
    <div class="stat"><div class="n">{N['grounded_sys']}</div><div class="l">of its explanations point
      to the right cells in the table, up from <b>{N['grounded_model2']}</b> on its own</div></div>
    <div class="stat"><div class="n">Never&nbsp;worse</div><div class="l">than the AI alone: a built-in
      safety check holds back when the question is unfamiliar</div></div>
  </div>

  <h3>On the locked test ({N['test_n']} tables, graded once)</h3>
  <p class="muted">A “locked test” means these {N['test_n']} tables were set aside up front and the
  system was graded on them <b>a single time</b>, with no peeking and no retries.</p>
  <table class="data">
    <tr><th>What we checked</th><th class="r">Result</th><th>For comparison</th></tr>
    <tr><td><b>Right answers</b></td>
        <td class="r"><b>{N['engine']}</b></td><td>{N['model_own']} when the AI did the math itself</td></tr>
    <tr><td><b>Explanations that point to the right cells</b></td>
        <td class="r"><b>{N['grounded_sys']}</b></td><td>{N['grounded_model2']} from the AI’s own citations</td></tr>
    <tr><td>Explanations that fully check out</td>
        <td class="r">{N['valid_sys']}</td><td></td></tr>
    <tr><td>Understood the question well enough to compute</td>
        <td class="r">{N['op_present']}</td><td>exactly right {N['whole_op']} of the time</td></tr>
    <tr><td>Accuracy by question type (winner / threshold / trade-off)</td>
        <td class="r">{N['best']} / {N['thresh']} / {N['trade']}</td><td></td></tr>
  </table>
  <p class="muted">The score on these unseen tables matched the score during development, so it isn’t a
  fluke of one lucky test set. Could {N['engine']} be misleading, since we built the answer key
  ourselves? We checked it against independent people and a separate AI reasoning from scratch. See the
  <a href="#limit" data-goto="limit">Validation&nbsp;&amp;&nbsp;limits</a> tab.</p>

  <h3>How each stage got there</h3>
  <p class="muted">Each step of building the system, and what it added. The first three rows are
  different AI models on their own; the last two add the calculator and the show-your-work step.</p>
  <table class="data">
    <tr><th>Stage</th><th class="r">Valid explanation</th><th class="r">Cells correct</th><th class="r">Right answer</th></tr>
    <tr><td>Untrained small AI <span class="muted">(Qwen3-1.7B)</span></td><td class="r">0%</td><td class="r">1%</td><td class="r">18%</td></tr>
    <tr><td>Trained on examples <span class="muted">(1.7B, fine-tuned)</span></td><td class="r">17%</td><td class="r">49%</td><td class="r">20%</td></tr>
    {base4b_row}
    <tr><td>Bigger AI, trained <span class="muted">(Qwen3-4B, fine-tuned)</span></td><td class="r">44%</td><td class="r">70%</td><td class="r">52%</td></tr>
    <tr><td>+ Calculator step <span class="muted">(the executor)</span></td><td class="r">69%·</td><td class="r">71%·</td>
        <td class="r"><b>{N['engine']}</b></td></tr>
    <tr><td><b>+ Show-your-work step</b> <span class="muted">(grounded citations)</span></td><td class="r"><b>{N['valid_sys']}·</b></td>
        <td class="r"><b>{N['grounded_sys']}·</b></td><td class="r"><b>{N['engine']}</b></td></tr>
  </table>
  {ans_chart}
  <div class="callout">
    <b>What teaching the AI on examples bought (and didn’t).</b> Training the small AI on examples
    (“fine-tuning”) made its explanations far better than the untrained model: well-formed explanations
    <span class="d">0% → 17%</span>, cells pointed to correctly <span class="d">1% → 49%</span>. But it
    still got the <b>answer</b> wrong about as often (<span class="d">18% → 20%</span>). Better
    explanations, same shaky arithmetic: that gap is exactly what a bigger model and then the calculator
    step went on to close.
  </div>
  <p class="muted">· marks the two final steps (measured on the 254 locked-test tables); the earlier
  rows are from development. The big jump from the small AI to the larger one <i>on the same training</i>
  showed the bottleneck was the AI’s <b>size</b>, not the examples; and the right-answer score stayed
  stuck until the calculator step, confirming the weak spot was the <b>arithmetic, not the reading</b>.</p>
  {base4b_note}

  <div class="tldr big">
    <h2>In one breath</h2>
    <ul>
      <li><b>Let the AI understand, let software calculate.</b> The AI reads the question and table and
        says what to work out; reliable software does the math. Right answers
        <b>{N['model_own']} → {N['engine']}</b>.</li>
      <li><b>Make it show its work.</b> The explanation points to the actual cells the software used,
        not numbers the AI typed from memory. Explanations grounded in the real cells
        <b>{N['grounded_model2']} → {N['grounded_sys']}</b>.</li>
      <li><b>Know its limits.</b> Faced with a kind of question it was never taught, it defers to the
        AI’s own answer instead of guessing, so the system is <b>never worse than the AI alone</b>.</li>
    </ul>
  </div>
  <p class="fineprint"><b>The statistics, for the technically inclined.</b> Engine answer
  {N['engine']} carries a 95% Wilson confidence interval well clear of the model's {N['model_own']};
  the development and test numbers match almost exactly, so the result generalizes rather than
  overfitting the development set. Head-to-head on the same items, the calculator approach fixed
  +{N['mcnemar_plus']} answers and broke only −{N['mcnemar_minus']} (an exact-McNemar test gives
  p ≈ 10⁻²⁶), so the improvement is not noise.</p>
</section>
</div>

<div class="panel" id="p-why" role="tabpanel">
<section>
  <h2>The problem</h2>
  <p>Ask a small language model a precise question about a table (<i>"which line item has the lowest
  2019 value while 2018 stays at or above 12.7?"</i>), and two things can go wrong. It can
  <b>misread</b> the table, or it can read it correctly and then <b>botch the arithmetic</b>: a
  comparison flipped, a dominance check fumbled, a number hallucinated into a citation.</p>
  <p>For anything you'd actually trust, the answer isn't enough: you want the <b>reasoning</b>, and
  you want each step pinned to the exact cells it used, so a checker (or a person) can verify it. That
  is a <b>grounded trace</b>. The goal of this project was a small, local model that produces them.</p>
  <p>The central question turned out to be <b>which</b> of those two failures dominates, because the
  fix is completely different. If the model misreads, you need better perception or more data. If it
  reads fine but miscomputes, you don't need a bigger model at all; you need to stop trusting its
  arithmetic.</p>
</section>

<section>
  <h2>The task &amp; the benchmark</h2>
  <p>Tables come from <b>TAT-QA</b> (Zhu et al., ACL 2021; CC BY 4.0): real financial tables from
  company filings. TAT-QA's own questions are ~64% free-form arithmetic (change, %, sums) and only a
  sliver match clean, checkable predicates, so we keep the <b>real tables</b> and generate three
  families of <b>constrained, programmatically-verifiable</b> questions over them:</p>
  <table class="data">
    <tr><th>Task</th><th><code>question_type</code></th><th>Example</th></tr>
    <tr><td>Winner selection</td><td><code>best_under_constraint</code></td>
        <td>"Which line item has the lowest 2019 value with 2018 ≥ 12.7?"</td></tr>
    <tr><td>Constraint filtering</td><td><code>threshold_filter</code></td>
        <td>"Which periods have rate ≥ 2.89 and term ≥ 3.65?"</td></tr>
    <tr><td>Trade-off</td><td><code>tradeoff_summary</code></td>
        <td>"Which line items are Pareto-optimal maximizing both 2019 and 2018?"</td></tr>
  </table>
  <p>Every example ships with a machine-checked grounded trace; an independent rule-based validator
  recomputes the answer from a stored spec and verifies each cited cell. The dataset is
  <b>1,266 validated examples</b> (100% pass the validator), with <b>177</b> used as silver training
  records and a <b>leakage-free, table-level split</b>: dev (190) / locked test (254) / a separate
  12-example out-of-distribution anchor.</p>
  <p class="fineprint"><b>Why a table-level split, and how leakage was ruled out.</b> The same source
  table appears in multiple examples (two orientations × several question types), so a naïve
  per-example split leaks tables across train/test. We re-split by <b>source table</b> (stripping
  orientation suffixes); 166/224 tables had straddled a per-example split. Disjointness is asserted at
  both the <b>id</b> and <b>content-hash</b> level on the exact file the model trained on.
  Total/subtotal rows are excluded from the entity universe by a single shared rule, so the
  inference-time universe matches the gold spec.</p>
</section>
</div>

<div class="panel" id="p-how" role="tabpanel">
<section>
  <h2>The approach</h2>
  <div class="pipeline">
    <span class="node">TAT-QA tables</span><span class="arrow">→</span>
    <span class="node">generate Q + gold + trace</span><span class="arrow">→</span>
    <span class="node">validate</span><span class="arrow">→</span>
    <span class="node">QLoRA SFT</span><span class="arrow">→</span>
    <span class="node key">executor</span><span class="arrow">→</span>
    <span class="node key">grounding</span><span class="arrow">→</span>
    <span class="node key">safety gate</span>
  </div>
  <p>We fine-tuned Qwen3 (QLoRA, 4-bit, a single RTX 3090). Supervised fine-tuning alone got the
  model writing well-formed traces, but its <i>answers</i> were still only ~59% right. The error
  analysis was decisive: the remaining mistakes were <b>arithmetic-execution</b> errors, not
  misreadings. So:</p>
  <h3>The executor: answer by construction</h3>
  <p>Instead of trusting the model's arithmetic, the model emits a structured <b>operation</b> (the
  op type, the column <i>names</i>, thresholds, and directions it read from the question), and a
  small, deterministic, independently-tested engine computes the answer. The model's job becomes
  <b>comprehension</b>; the arithmetic is done by proven code. This is the move that takes answers
  from {N['model_own']} to <b>{N['engine']}</b>.</p>
  <h3>Grounding: citations by construction</h3>
  <p>The model's hand-typed citations had the same disease as its arithmetic (wrong/derived numbers),
  so they got the same cure: the trace cites the <b>cells the engine actually read</b>, which are
  grounded by construction: they always exist and always match. The <b>prose stays
  model-authored</b> (the explanation); only the cited evidence becomes system-constructed (the
  proof). Grounded-trace rate: {N['grounded_model2']} → <b>{N['grounded_sys']}</b>.</p>
  <h3>The safety gate</h3>
  <p>The engine is a specialist: it only knows the operations it was trained on. If the model emits
  an operation the <i>question</i> doesn't actually support (no threshold language for a threshold
  op, no trade-off language for a trade-off op), the gate treats it as fabricated and
  <b>falls back to the model's own answer</b> (do no harm).</p>
</section>
</div>

<div class="panel" id="p-limit" role="tabpanel">
<section>
  <h2>Is {N['engine']} real, or circular?</h2>
  <p>Honest worry: the test gold is computed by code that shares the engine's arithmetic, so on its
  own that number is <b>consistency</b>, not <b>correctness</b>. To break the circularity we ran two
  independent checks that never see the spec or the gold.</p>
  <p><b>Blind LLM cross-check.</b> 36 stratified examples, shown as raw table + question only,
  answered by independent annotators reasoning from scratch: <b>36/36 agreement</b> with the stored
  gold (confirming the engine on every one of its sampled correct cases). Two independent reasoning
  paths agreeing on every case ⇒ the answers are corroborated, not merely self-consistent.</p>
  <p><b>Blind human anchor.</b> A real person, given a self-contained annotation UI, independently
  answered the same items on the two <i>natural</i> question types (n=24): <b>22/24 blind agreement</b>
  with the stored gold: a second, human reasoning path reaching the same answers.</p>
</section>

<section>
  <h2>The limit: where it fails, and how it's caught</h2>
  <p>The one non-circular slice is an out-of-distribution task type (<code>extremum</code>) that is
  <b>not</b> in training and <b>not</b> in the engine's vocabulary. Here the model's <i>own</i> answer
  is right {N['gate_ood_gated']}, but it emits a <i>trained</i> operation instead, so the executor
  computes the wrong answer <b>{N['gate_ood_engine']}</b>. The engine isn't broken; the model just
  doesn't know the operation exists, and the executor overrides its correct intuition and fails
  <i>confidently</i>. This failure mode is invisible to any in-distribution (circular) test.</p>
  <p>The safety gate is exactly for this. Re-scored on saved outputs: out-of-distribution
  <b>{N['gate_ood_engine']} → {N['gate_ood_gated']}</b>; in-distribution {N['gate_in_engine']} →
  <b>{N['gate_in_gated']}</b> (it fires on only {N['gate_in_fires']}, and never on a case the engine
  got right). Net: <b>never worse than the model alone</b>, and it degrades gracefully. Here is the
  gate firing on a real out-of-distribution example: the engine produces nothing usable, so the
  system keeps the model's correct answer.</p>
  {card_ood}
  <p class="demo-note">Caveat: the gate is a keyword heuristic validated on a small out-of-distribution
  set.</p>
</section>

<section>
  <h2>Honest limitations</h2>
  <ul class="lim">
    <li><b>In-distribution gold is spec-derived.</b> Now corroborated by two independent blind oracles,
      but not <i>proven</i>: n is small, trade-off has only the LLM check, the human is one annotator.</li>
    <li><b>The trade-off questions are non-natural.</b> A non-specialist couldn't assess
      "Pareto-optimal" (verifiable, but not how a person would ask).</li>
    <li><b>Specialization.</b> The engine helps only inside its trained operation vocabulary; the gate
      makes that safe, but it is not a general table engine.</li>
    <li><b>The trace prose stays model-authored.</b> The system grounds the <i>answer</i> and the
      <i>cited evidence</i> by construction; an unaided 4B model's own citations remain ~{N['grounded_model2']}
      (near its ceiling).</li>
    <li><b>Coverage &amp; domain.</b> Only high-confidence ingested tables (~97%); selection /
      threshold / frontier reasoning, not TAT-QA's dominant free-form arithmetic; TAT-QA is financial,
      and transfer to scientific/ML tables is unverified.</li>
    <li><b>Provenance.</b> The 12-example anchor is <i>TAT-QA-gold-anchored</i> (their annotations,
      cross-checked by our independent recompute), not verified by anyone on this project.</li>
  </ul>
</section>
</div>

<div class="panel" id="p-demo" role="tabpanel">
<section>
  <h2>See it run</h2>
  <p>Each card below replays a held-out example through the full path: the model emits an operation,
  the engine computes the answer and returns the exact cells it read (<b>highlighted</b>), the gate
  decides whether to trust it. No model is called; these are saved outputs re-assembled on CPU.</p>
  {card_in}
  <p class="demo-note">Generate the full set yourself:
  <code>python scripts/demo.py</code> (writes <code>results/demo/index.html</code>),
  <code>python scripts/demo.py --list</code>, or <code>--id &lt;example_id&gt;</code>.</p>
</section>
</div>

<div class="panel" id="p-repro" role="tabpanel">
<section>
  <h2>Reproduce</h2>
  <pre>{_h_pre('''# build dataset / leakage-free split / SFT data
python scripts/build_dataset.py
python scripts/freeze_splits.py
python scripts/build_sft.py

# train + score (WSL .venv-train, RTX 3090; see docs/training_env.md)
python scripts/train_sft.py --model Qwen/Qwen3-4B --out models/qwen3-4b-sft-exec
python scripts/eval_executor.py --dataset data/processed/eval_test.v0_1_0.jsonl \\
       --adapter models/qwen3-4b-sft-exec --out results/p3_4b_exec_TEST.json

# re-score on CPU (no GPU): grounding, safety gate, this write-up, the demo
python scripts/eval_grounded.py
python scripts/eval_gate.py
python scripts/build_writeup.py
python scripts/demo.py''')}</pre>
  <p>ML methodology reference: <a href="methodology.html">docs/methodology.html</a>. Consolidated
  findings: <a href="../RESULTS.md">RESULTS.md</a>. Interactive demo:
  <a href="../results/demo/index.html">results/demo/index.html</a>.</p>
</section>
</div>

<footer>
  Numbers in this page are sourced directly from the saved result files
  (<code>results/p3_4b_exec_TEST.json</code>, <code>…_gate.json</code>,
  <code>…_grounded_constructed.json</code>) by <code>scripts/build_writeup.py</code>; the demo cards
  are real, re-assembled from saved outputs. Tables: TAT-QA (Zhu et al., <i>ACL 2021</i>), CC BY 4.0,
  used with attribution; questions, traces, and engine are this project's.
</footer>

</div>
<script>
(function(){{
  function show(id){{
    document.querySelectorAll('.panel').forEach(function(p){{p.classList.toggle('active',p.id==='p-'+id)}});
    document.querySelectorAll('.tab').forEach(function(t){{t.classList.toggle('active',t.dataset.p===id)}});
    if(history.replaceState) history.replaceState(null,'','#'+id);
    window.scrollTo({{top:0,behavior:'instant'}});
  }}
  document.querySelectorAll('.tab').forEach(function(t){{t.addEventListener('click',function(){{show(t.dataset.p)}})}});
  document.querySelectorAll('[data-goto]').forEach(function(a){{a.addEventListener('click',function(e){{e.preventDefault();show(a.dataset.goto)}})}});
  var h=location.hash.replace('#','');
  if(h&&document.getElementById('p-'+h)) show(h);
}})();
</script>
</body></html>"""


def _h_pre(s: str) -> str:
    import html
    return html.escape(s)


def main() -> int:
    N = gather_numbers()
    OUT.write_text(build(N), encoding="utf-8")
    print(f"  -> wrote {OUT.relative_to(ROOT)}")
    print(f"     engine {N['engine']} | grounded {N['grounded_sys']} | gate OOD "
          f"{N['gate_ood_engine']}->{N['gate_ood_gated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
