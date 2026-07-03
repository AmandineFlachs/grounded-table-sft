"""Build docs/methodology.html - the lean ML methodology reference, on the shared design kit.

Companion to the reader-facing write-up (scripts/build_writeup.py). Content is a static technical
reference; styling comes entirely from src/theme.py (the "Field Notes" system, personal brand), so
this page stays visually consistent with docs/index.html and can't drift from the design.

    python scripts/build_methodology.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.theme import brand_font_link, theme_css  # noqa: E402

OUT = ROOT / "docs" / "methodology.html"

# page-specific chrome not worth putting in the shared skeleton (a plain-string, single-brace CSS)
EXTRA_CSS = (
    ".toc{font-family:var(--font-mono);font-size:12.5px;letter-spacing:.02em;color:var(--muted);"
    "margin:36px 0 4px;line-height:2.1}"
    ".toc a{color:var(--accent);text-decoration:none}.toc a:hover{text-decoration:underline}"
    ".toc b{color:var(--muted);font-weight:600;margin-right:2px}"
    ".lead{font-size:16px}"
)


def build() -> str:
    css = theme_css("personal")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounded Table Reasoning Traces — Methodology</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{brand_font_link('personal')}" rel="stylesheet">
<style>{css}{EXTRA_CSS}</style></head>
<body>
<div class="topbar"></div>

<header class="hero"><div class="texture"></div><div class="inner">
  <div class="masthead">
    <div class="byline">
      <span class="pic"><img src="assets/author.jpg" alt="Amandine Flachs"
        onerror="this.style.display='none';this.parentNode.classList.add('ph')"></span>
      <div>
        <div class="name">Amandine Flachs</div>
        <div class="links"><a href="https://github.com/AmandineFlachs">github.com/AmandineFlachs</a></div>
      </div>
    </div>
  </div>
  <div class="head">
    <h1>Grounded Table Reasoning Traces — Methodology</h1>
    <p class="subtitle">A technical reference for the ML approach: teaching a small model to answer
    constrained questions over real tables and emit reasoning traces grounded in specific cells, by
    routing its comprehension through a deterministic engine.</p>
    <div class="badges">
      <span class="badge">Qwen3-4B · QLoRA</span>
      <span class="badge">TAT-QA · CC BY 4.0</span>
      <span class="badge">Held-out test, scored once</span>
    </div>
    <p class="fineprint" style="margin-top:20px">The reader-facing narrative with a live demo is in the
    <a href="index.html"><b>write-up</b></a>; the consolidated, honestly-bounded results are in
    <a href="../RESULTS.md"><b>RESULTS.md</b></a>. This page is the lean ML reference.</p>
  </div>
</div></header>

<main>

<nav class="toc">
  <a href="#problem"><b>1</b>Problem</a> · <a href="#data"><b>2</b>Data &amp; task</a> ·
  <a href="#schema"><b>3</b>Schema</a> · <a href="#train"><b>4</b>Model &amp; training</a> ·
  <a href="#executor"><b>5</b>Executor</a> · <a href="#grounding"><b>6</b>Grounding &amp; gate</a> ·
  <a href="#eval"><b>7</b>Evaluation</a> · <a href="#results"><b>8</b>Results</a> ·
  <a href="#limits"><b>9</b>Limitations</a>
</nav>

<section id="problem">
  <h2 class="sec-h">1. Problem &amp; approach</h2>
  <p class="lead">Small language models <i>comprehend</i> table questions well but cannot reliably
  <i>execute</i> the arithmetic. The goal is <b>grounded reasoning traces</b>: every step tied to
  specific, verifiable cells, with every numeric claim provably true.</p>
  <p>The design is <b>neuro-symbolic</b>. The model's job is comprehension: it emits a structured
  <code>operation</code> (which op, which columns, which thresholds and directions, read from the
  question). A small deterministic engine computes the answer and returns the exact cells it read. So
  the model contributes understanding and explanation; proven code contributes the arithmetic and the
  grounding. A safety gate falls back to the model when the emitted operation is not supported by the
  question, keeping the system never worse than the model alone.</p>
</section>

<section id="data">
  <h2 class="sec-h">2. Data &amp; task</h2>
  <p>Tables come from <b>TAT-QA</b> (Zhu et al., ACL 2021; CC BY 4.0): real financial tables from
  company filings. TAT-QA's native questions are ~64% free-form arithmetic and only ~8% match clean
  selection/threshold predicates, so we keep the <b>real tables</b> and generate three families of
  <b>constrained, programmatically-verifiable</b> questions over them (both row- and column-orientations,
  via transpose; explicit-direction phrasing so the gold is a mechanical computation):</p>
  <table class="data">
    <thead><tr><th>Task</th><th>question_type</th><th>Gold</th></tr></thead>
    <tbody>
    <tr><td>Winner selection</td><td><code class="acc">best_under_constraint</code></td><td>argmax of a target column over rows passing a threshold constraint</td></tr>
    <tr><td>Constraint filtering</td><td><code class="acc">threshold_filter</code></td><td>the set of rows satisfying all conditions</td></tr>
    <tr><td>Trade-off</td><td><code class="acc">tradeoff_summary</code></td><td>the Pareto frontier across two metrics</td></tr>
    </tbody>
  </table>
  <p><b>Independent validator.</b> A rule-based validator recomputes the answer from a stored
  <code>spec</code> and verifies each cited cell (existence, value match, numeric-comparison and
  threshold correctness, relevance). It is deliberately kept <b>independent</b> of the answer engine
  (its own predicate re-implementations, no shared helpers) so that "engine output passes validator" is
  not circular. Total/subtotal rows are excluded from the entity universe by a single shared rule, so
  the inference-time universe matches the gold spec.</p>
  <p><b>Splits &amp; sizes.</b> <b>1,266</b> validated examples (100% pass the validator); <b>177</b>
  silver-rendered training records. The eval split is <b>leakage-free at the table level</b>: re-split
  by <i>source table</i> (orientation suffixes stripped; 166/224 tables had straddled a per-example
  split), asserted disjoint at both <code>id</code> and content-hash level — <b>dev 190 / locked test
  254</b>, plus a separate <b>12-example out-of-distribution anchor</b> (an <code>extremum</code> task
  type not in training).</p>
</section>

<section id="schema">
  <h2 class="sec-h">3. Output schema</h2>
  <p>The model emits one JSON object per example: <code>trace_steps[]</code> (each a short
  natural-language <code>description</code> + a structured <code>cites</code> list), a structured
  <code>final_answer</code>, and the <code>operation</code> the engine consumes. Citations carry both
  coordinates and semantics for exact, reorder-robust validation:</p>
  <table class="data">
    <thead><tr><th>Type</th><th>Shape</th></tr></thead>
    <tbody>
    <tr><td><code>CellRef</code></td><td><code>row:int, col:int, col_name:str, value</code> — the unit of evidence</td></tr>
    <tr><td><code>TraceStep</code></td><td><code>kind:"filter"|"compare"|"aggregate"|"select"|"conclude", description, cites:[CellRef]</code></td></tr>
    <tr><td><code>operation</code></td><td>op type + column <i>names</i> + thresholds + directions (e.g. <code>{{type, target, target_dir, constraint, op, threshold}}</code>)</td></tr>
    </tbody>
  </table>
</section>

<section id="train">
  <h2 class="sec-h">4. Model &amp; training</h2>
  <p><b>QLoRA SFT</b>, fully local on a single RTX 3090 (~11 min/run). Base <b>Qwen3-1.7B</b> first (to
  rehearse the converter → SFT → eval loop), then promoted to the <b>Qwen3-4B</b> target on identical
  data.</p>
  <ul class="lim">
    <li><b>Quantization:</b> 4-bit NF4, bfloat16 compute.</li>
    <li><b>LoRA:</b> r=16, α=32, dropout=0.05, <code>target_modules="all-linear"</code>, causal-LM.</li>
    <li><b>Optimization:</b> 3 epochs, lr 2e-4, batch 2 (+ gradient accumulation), bf16.</li>
    <li><b>Loss:</b> <b>completion-only</b> — the prompt is masked to <code>-100</code> with a
      deterministic, verified assistant-only mask (this was the decisive fix over full-sequence loss:
      valid-trace 0%→17% at 1.7B).</li>
  </ul>
  <p><b>Silver training labels.</b> Each question is rendered <b>N=3</b> times (headless Claude); only
  answers that <i>agree</i> and pass the Tier-1 groundedness + numeric-consistency checks are kept
  (high-precision, ~56% yield). The held-out eval set keeps the silver set honest.</p>
</section>

<section id="executor">
  <h2 class="sec-h">5. The executor — answer by construction</h2>
  <p>SFT alone left answers at ~59%; error analysis showed the residual was
  <b>arithmetic-execution</b> errors (flipped comparisons, fumbled dominance), not misreadings. So
  instead of trusting the model's arithmetic, the model emits the structured <code>operation</code> and
  a deterministic engine (<code>src/executor.py</code>) computes the answer: row filtering, argmax/min,
  multi-condition thresholds, and Pareto dominance, with explicit epsilon tolerance, tie handling, and
  total-row exclusion. Operations reference column <i>names</i>, so they are robust to reordering and
  are exactly what a model can read off the question.</p>
  <div class="callout">By design the engine is <b>independent of the validator</b> (separate predicate
  re-implementations). The validator remains a genuine external check on the engine's output; sharing
  code would make "executor output passes validator" circular — the same independence the validator
  keeps from the generator.</div>
</section>

<section id="grounding">
  <h2 class="sec-h">6. Grounding &amp; the safety gate</h2>
  <h3 class="h3a">Grounding by construction</h3>
  <p>The model's hand-typed citations have the same disease as its arithmetic (wrong/derived numbers),
  so they get the same cure: the trace cites the <b>cells the engine actually read</b>
  (<code>executor.evidence_for</code>) — grounded by construction (read straight from the table, so they
  always exist and match). The trace <i>prose</i> stays model-authored (the explanation); the cited
  evidence becomes system-constructed (the proof).</p>
  <h3 class="h3a">Out-of-vocabulary safety gate</h3>
  <p>The engine only helps inside its trained operation vocabulary. The gate checks the <i>question</i>
  carries the signal the emitted op type requires — threshold language for best/threshold ops,
  trade-off/Pareto language for tradeoff. If the signal is absent, the operation is fabricated, so the
  system falls back to the model's own answer. Deployable from question text + op type alone (no gold);
  a "do no harm" check, not merely an OOD patch.</p>
</section>

<section id="eval">
  <h2 class="sec-h">7. Evaluation methodology</h2>
  <p><b>Metrics</b> (over <i>all</i> examples — an unparseable output scores 0 on every metric): engine
  answer accuracy vs. the model's own answer, operation present / whole-op exact-match, trace
  groundedness, full valid-trace, and a per-type breakdown.</p>
  <p><b>Protocol.</b> A <code>dev</code> slice (190) for iteration; the <b>locked test (254) scored
  exactly once</b> under a pre-registered frozen config (Qwen3-4B + the SFT-exec adapter,
  greedy/deterministic, strict validator).</p>
  <p><b>Circularity and the anchors.</b> The in-distribution gold is spec-derived and the engine
  re-implements the same semantics, so an in-distribution score measures <b>operation-comprehension
  consistency</b> on unseen tables, not correctness against an outside oracle. Two <i>blind</i> anchors
  break the circularity: independent LLM annotators answering from the raw table + question alone agreed
  with stored gold <b>36/36</b>; a blind human oracle agreed <b>24/24</b> on the two natural types (0
  gold errors). The <b>OOD anchor</b> (12 <code>extremum</code> examples, a task type outside the
  operation vocabulary) is the only fully non-circular slice — and it is what exposed the executor's
  specialization limit (below).</p>
</section>

<section id="results">
  <h2 class="sec-h">8. Results &amp; key findings</h2>
  <p>Progression (trained-model stages on dev 190; the two system layers re-scored on the locked test
  254):</p>
  <table class="data">
    <thead><tr><th>Stage</th><th class="r">Valid trace</th><th class="r">Grounded</th><th class="r">Answer</th></tr></thead>
    <tbody>
    <tr><td>Base Qwen3-1.7B (zero-shot)</td><td class="r">0%</td><td class="r">1%</td><td class="r">18%</td></tr>
    <tr><td>+ QLoRA SFT (1.7B)</td><td class="r">17%</td><td class="r">49%</td><td class="r">20%</td></tr>
    <tr><td>Base Qwen3-4B (zero-shot)</td><td class="pending" colspan="3">pending (measured separately)</td></tr>
    <tr><td>+ QLoRA SFT (4B, same data)</td><td class="r">44%</td><td class="r">70%</td><td class="r">52%</td></tr>
    <tr><td>+ Executor (answer by construction)</td><td class="r">69%</td><td class="r">71%</td><td class="r hot">95.7%</td></tr>
    <tr class="keyrow"><td><b>+ Grounding (citations by construction)</b></td><td class="r"><b>94.9%</b></td><td class="r"><b>96.9%</b></td><td class="r hot">95.7%</td></tr>
    </tbody>
  </table>
  <p>On the <b>locked test (254)</b>: engine answer <b>95.7%</b> [92.4–97.6] vs. the model's own
  arithmetic 59.4% — a paired <b>+93 / −1</b> (exact-McNemar p ≈ 10⁻²⁶); by type 97.6 / 96.6 / 92.9%.
  Grounding lifts the trace-grounded rate <b>71.3% → 96.9%</b> (valid 68.9% → 94.9%). The gate takes OOD
  <b>0% → 100%</b> and in-distribution 95.7% → 96.1% (fires on 9/254, never on a case the engine got
  right). Full numbers and CIs: <a href="../RESULTS.md">RESULTS.md</a>.</p>
  <h3 class="h3a">Key findings</h3>
  <ul class="lim">
    <li><b>Capacity-bound, not data-bound.</b> The 1.7B→4B jump on <i>identical</i> data (valid 17%→44%,
      answer 20%→52%) located the bottleneck in model size, not training data.</li>
    <li><b>The residual is execution, not perception.</b> Grounded-but-wrong answers were
      arithmetic-execution errors — which is what motivated routing the arithmetic through code.</li>
    <li><b>The model's own citations are near a ceiling</b> (~71%), so grounding is solved by
      construction rather than by retraining a 4B model to cite better.</li>
    <li><b>The executor is a specialization, not a general engine.</b> On the OOD anchor the model's own
      answer is 100% but it emits a <i>trained</i> op type, so the executor scores 0/12 — hence the
      gate.</li>
  </ul>
</section>

<section id="limits">
  <h2 class="sec-h">9. Limitations</h2>
  <ul class="lim">
    <li><b>In-distribution gold is spec-derived</b> — corroborated by two blind oracles, but not proven:
      n is small, tradeoff has only the LLM check, the human is one annotator.</li>
    <li><b>Specialization</b> — the engine helps only inside its trained op vocabulary; the gate makes
      that safe but it is not a general table engine.</li>
    <li><b>Trace prose stays model-authored</b> — the system grounds the answer and the cited evidence;
      an unaided 4B model's own citations remain ~71%.</li>
    <li><b>Coverage &amp; domain</b> — only high-confidence ingested tables (~97%); selection / threshold
      / frontier reasoning, not TAT-QA's dominant free-form arithmetic; TAT-QA is financial, and transfer
      to scientific/ML tables is unverified.</li>
    <li><b>Non-natural trade-off questions</b> — "Pareto-optimal" is technical; a non-specialist human
      couldn't assess it (12/12 "I don't know").</li>
    <li><b>Provenance</b> — the 12-example OOD anchor is <i>TAT-QA-gold-anchored</i> (their annotations,
      cross-checked 16/16 by our recompute), not verified by anyone on this project.</li>
  </ul>
</section>

</main>

<footer><div class="inner">
  <p class="note">Grounded Table Reasoning Traces · <code>docs/methodology.html</code>. Companion
  documents: <a href="index.html">write-up</a> (narrative + demo) and
  <a href="../RESULTS.md">RESULTS.md</a> (full results). The development chronology lives in the git
  history.</p>
</div></footer>

</body></html>"""


def main() -> int:
    OUT.write_text(build(), encoding="utf-8")
    print(f"  -> wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
