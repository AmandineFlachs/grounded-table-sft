# Results - Grounded Table Reasoning Traces (Phase 3)

A consolidated, honestly-bounded summary of what the project established. The ML methodology
(data, training, executor, evaluation) is in [`docs/methodology.html`](docs/methodology.html); this is
the synthesis. For a single reader-facing narrative (promotable, with a live demo embedded), see
[`docs/writeup.html`](docs/writeup.html).

## TL;DR

A small model (Qwen3-4B, QLoRA) was taught to read text tables, answer three families of
constrained questions, and emit grounded reasoning traces. The central finding:

> **Small models comprehend table tasks well but cannot reliably *execute* the arithmetic.**
> Routing the model's *comprehension* (a structured operation it emits) through a deterministic
> engine - instead of trusting its own arithmetic - lifts held-out answer accuracy from
> **59% → 95.7%**. The result generalizes to unseen tables and is independently corroborated.
> The same move closes the project's actual thesis - **grounded traces**: citing the cells the
> engine read (not the model's hand-typed numbers) lifts the grounded-trace rate **71% → 97%**.
> The engine only helps **inside the operation vocabulary it was trained on** - on an unseen task
> type it would confidently fail - so a **safety gate** falls back to the model's own answer when
> the operation isn't supported by the question, making the system **never worse than the model**.

## The task

Three question types over real financial tables (TAT-QA, CC BY 4.0; we generate the questions,
[strategy notes](docs/methodology.html#data)):

| Task | `question_type` | Example |
|---|---|---|
| Winner selection | `best_under_constraint` | "Which line item has the lowest 2019 value with 2018 ≥ 12.7?" |
| Constraint filtering | `threshold_filter` | "Which periods have rate ≥ 2.89 and term ≥ 3.65?" |
| Trade-off | `tradeoff_summary` | "Which line items are Pareto-optimal maximizing both 2019 and 2018?" |

Every example carries a machine-checked grounded trace; a rule-based validator independently
recomputes the answer and verifies each cited cell. Data: 1,266 validated examples; 177
silver-rendered training records; a **leakage-free table-level split** (dev 190 / locked test 254 /
a separate 12-example out-of-distribution anchor), disjoint at id *and* content-hash level.

## The progression

Trained-model stages on the **dev** set (190); the two **system layers** (executor answer, grounded
citations) on the **locked test** (254), where the model's raw outputs were saved so they can be
re-scored on CPU - they match dev wherever both exist.

| Stage | Valid trace | Grounded | Answer |
|---|---|---|---|
| Base Qwen3-1.7B (zero-shot) | 0% | 1% | 18% |
| + QLoRA SFT (1.7B) | 17% | 49% | 20% |
| Base Qwen3-4B (zero-shot) | 0% | 0% | 27% |
| + QLoRA SFT (4B, same data) | 44% | 70% | 52% |
| + Executor - *answer* by construction | 69%· | 71%· | **96%** |
| **+ Grounded - *citations* by construction** | **95%·** | **97%·** | **96%** |

> **Note:** the **untrained Qwen3-4B (zero-shot)** baseline was backfilled on the dev set (190) after
> Phase 3 (it had not been measured then; only the 1.7B base was). It confirms the expected shape: the
> bigger base model answers better than the 1.7B base (18% → 27%) but still emits **0% valid / 0%
> grounded** traces. So SFT is what teaches the structured, grounded format, and the executor and
> grounding layers are what lift the answer and citations from there.

·locked test (254). The executor fixes the *answer* (the model's own arithmetic was 59%); grounding
fixes the *citations* (the model's own were 71%). The trace **prose stays model-authored** throughout
- only the answer and the cited evidence become system-constructed.

The 1.7B→4B jump on *identical* data showed the residual was **capacity-bound, not data-bound**.
Error analysis then showed the remaining wrong answers were **arithmetic-execution** errors
(false comparisons, dominance logic), not misreadings - which motivated the executor; the same
"don't trust the model's numbers, construct them from the cells the engine read" move then closed
the *citation* gap (last row).

## See it run - the demo

```bash
python scripts/demo.py            # featured cards (1 per type + 1 out-of-distribution) + writes the page
python scripts/demo.py --list     # every replayable example
python scripts/demo.py --id <example_id>
```

Each card replays a held-out example through the production path - model *operation* → engine answer
→ engine-read evidence cells (**highlighted**) → safety gate - from **saved** outputs, no GPU. The
shareable page lands at [`results/demo/index.html`](results/demo/index.html); the out-of-distribution
card shows the gate **firing** and falling back to the model's own (correct) answer.

## The executor

The model emits a structured `operation` (op type + column *names* + thresholds + directions);
a deterministic engine ([`src/executor.py`](src/executor.py)) computes the answer. The model's job
becomes *comprehension*; the arithmetic is done by proven code.

## Locked test (254 held-out tables, scored once)

| Metric | Result [95% CI] |
|---|---|
| **Engine answer** | **95.7%** [92.4–97.6] |
| Model's *own* answer (no engine) | 59.4% [53.3–65.3] |
| Operation present / exact-match | 97.2% / 87.4% |
| Trace groundedness | 71.3% [65.4–76.5] |

Engine vs the model's own arithmetic: **+93 / −1** (exact McNemar p ≈ 10⁻²⁶). By type:
best 97.6%, threshold 96.6%, tradeoff 92.9%. The dev→test numbers match almost exactly - the
result **generalizes**.

## Is 95.7% real, or circular? - the independent anchor

The test gold is computed by code that shares the engine's arithmetic, so on its own the number is
*consistency*, not correctness. To break the circularity we ran a **blind independent cross-check**
([`scripts/anchor_blind.py`](scripts/anchor_blind.py)): 36 stratified in-distribution examples
presented as raw table + question only (no spec, no gold) and answered by independent LLM
annotators reasoning from scratch.

> **36 / 36 agreement** with the stored gold (CI ≥ 90%), confirming the engine on 35/35 of its
> correct cases. Two independent reasoning paths agree on every sampled case → the in-distribution
> answers are **independently corroborated**, not merely self-consistent.

A **blind human anchor** (a self-contained annotation UI, [`scripts/annotate_build.py`](scripts/annotate_build.py)) then checked the same items with a real human oracle. On the two **natural**
question types (best_under_constraint + threshold_filter, n=24): blind agreement 22/24, and **both
disagreements adjudicated to human slips, not gold errors** (one induced by a duplicate row-label
artifact) → **0 gold errors found**. The annotator could **not** assess `tradeoff_summary`
(all 12 flagged "I don't know" - "Pareto-optimal" is a technical term), which is itself evidence
those questions are **non-natural**; tradeoff's independent check rests on the LLM anchor.

*Caveats:* the LLM oracle is not human; the human is one annotator and didn't cover tradeoff; n is small.

## The limit - an out-of-distribution anchor (the only non-circular slice)

On 12 `extremum` examples (a task type **not** in training and **not** in the operation
vocabulary): the model's *own* answer is **100%**, but it emits a *trained* op type instead of
`extremum`, so the executor computes the wrong answer **0/12** (the engine itself is fine - the
model just doesn't know the op exists).

> **The executor is a specialization within its trained operation vocabulary, not a general engine.**
> Outside that vocabulary it overrides the model's correct intuition and fails *confidently*. This
> failure mode is invisible to any in-distribution (circular) test - only the independent anchor
> revealed it.

**Fixed (Step B - safety gate, [`scripts/eval_gate.py`](scripts/eval_gate.py)):** a check that the
question carries the signal the emitted operation *requires* (threshold/trade-off language); if not,
the operation is fabricated, so fall back to the model's own answer. Re-scored on saved outputs:
OOD **0% → 100%**, in-distribution **95.7% → 96.1%** (fires on 4%, never on a case the engine got
right). The system is now **never worse than the model** and degrades gracefully. *Caveat:* keyword
heuristic, validated on a small OOD set.

## Honest limitations

- **In-distribution gold is spec-derived** - now corroborated by two independent blind oracles
  (LLM 36/36 all types; human 24/24-after-adjudication, natural types, 0 gold errors), but not
  *proven*: n is small, tradeoff has only the LLM check, the human is one annotator.
- **The `tradeoff_summary` questions are non-natural** - a non-specialist couldn't assess
  "Pareto-optimal" (12/12 "I don't know"). Verifiable, but not how a person would ask.
- **Data-quality artifacts** (surfaced by the anchors): duplicate row labels and leaked header rows
  in some ingested tables - not gold arithmetic errors, but they lower question quality and can
  trip even a human. **Fixed at the root** (`src/ingest/tatqa.py`, regression-tested); the cleaned
  `realtable.v0_2_0.jsonl` has 0 such artifacts. The scored locked test stays on `v0_1_0` (the
  data it was scored on); `v0_2_0` is for future work.
- **Trace groundedness** - the model's *own* citations are only ~71% (it writes wrong/derived
  numbers). **Closed by construction (Step A):** citing the cells the engine actually read lifts
  the *system's* grounded-trace rate to **96.9%** (valid 94.9%) on the locked test - the trace prose
  stays model-authored, the cited evidence becomes system-constructed. So the *system* meets the
  grounded-trace thesis; an unaided 4B model does not (near its citation ceiling).
- **Specialization** - no graceful degradation outside the trained op vocabulary.
- **Coverage** - only high-confidence ingested tables (~97%); selection/threshold/frontier
  reasoning, not TAT-QA's dominant arithmetic (change/%/sum).
- **Domain** - TAT-QA is financial; the eventual target (scientific/ML tables) is unverified.
- **Provenance** - the 12-example anchor is *TAT-QA-gold-anchored* (their annotations,
  cross-checked 16/16 by our recompute), **not** verified by anyone on this project.

## Reproduce

```bash
# build dataset / split / SFT data
python scripts/build_dataset.py            # synthetic sanity
python scripts/freeze_splits.py            # leakage-free table-level split
python scripts/build_sft.py                # SFT records (with operations)

# train + score (WSL .venv-train, RTX 3090; see docs/training_env.md)
python scripts/train_sft.py --model Qwen/Qwen3-4B --out models/qwen3-4b-sft-exec
python scripts/eval_executor.py --dataset data/processed/eval_test.v0_1_0.jsonl \
       --adapter models/qwen3-4b-sft-exec --out results/p3_4b_exec_TEST.json

# independent in-distribution anchor (CPU + blind LLM annotation)
python scripts/anchor_blind.py build --n 36 --seed 0
python scripts/anchor_blind.py score --answers results/anchor/answers.json
```

Key artifacts: `results/p3_4b_exec_TEST.json` (locked test),
`results/p3_4b_exec_TEST_verified.json` (OOD anchor), `results/anchor/report.json`
(independent in-distribution anchor), `results/p2_4_crosscheck.json` (TAT-QA cross-check).
