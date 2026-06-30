# Project: Grounded Table Reasoning Traces

<!--
  TASKS.md - project status & roadmap, also read by the Claude Code status line and the
  /project-status and /update-progress skills.

  PARSING RULES (keep this format so the tools work):
  - Project name : the "# Project:" line above.
  - A "step"     : any "- [ ]" (todo) or "- [x]" (done) checkbox line.
  - The status line's "current step" = the FIRST checkbox under "## Current Step".
  Keep exactly ONE checkbox under "## Current Step" at a time.
-->

## Plan
Teach a small, locally-trainable model to read text tables, answer constrained questions, and emit
reasoning traces grounded in specific cells - by routing the model's comprehension (a structured
operation it emits) through a deterministic engine that computes the answer and the cited evidence.
See [`docs/methodology.html`](docs/methodology.html) for the approach and [`RESULTS.md`](RESULTS.md)
for results.

## Current Step
- [ ] Final review/polish pass on the public docs, then publish (the human runs `git init` + commit +
  push + flip-public). The v0 milestone is complete; the progression table is now fully populated.

## Completed
- [x] **Untrained Qwen3-4B baseline** - the last missing progression cell, measured on dev (190):
  answer 27%, valid 0%, grounded 0%. A bigger base answers better than the 1.7B base (18% → 27%) but
  still emits no valid/grounded traces, so SFT teaches the format and the engine lifts the rest.
- [x] **v0 pipeline** - Pydantic schema, seeded synthetic table generator, programmatic
  grounded-by-construction traces, an independent rule-based validator, and an eval harness.
- [x] **Real tables** - TAT-QA ingestion (CC BY 4.0), three constrained question types in both
  orientations, 1,266 validated examples, and a leakage-free table-level split (dev 190 / locked
  test 254 / 12-example out-of-distribution anchor).
- [x] **Training** - QLoRA SFT, Qwen3-1.7B → Qwen3-4B, completion-only loss; silver-label loop
  (N=3 agreement + groundedness).
- [x] **Executor (answer by construction)** - the model emits a structured operation; a deterministic
  engine computes the answer. Dev answer accuracy 55% → 96%.
- [x] **Locked test scored once (254)** - engine 95.7%; independently corroborated by two blind
  anchors (LLM 36/36, human 24/24 on the natural types).
- [x] **Grounded traces by construction** - trace-grounded rate 71% → 97%.
- [x] **Out-of-vocabulary safety gate** - falls back to the model when the operation isn't supported
  by the question, so the system is never worse than the model alone.
- [x] **Demo, write-up, reproducibility pass** - `scripts/demo.py`, `docs/index.html`, and a
  fresh-clone CPU re-score path.

## Roadmap
- [ ] Extend the operation vocabulary (`extremum` / `compare_rows` / `rank_models`) and add native
  arithmetic question types (change / % / sum).
- [ ] Improve the model's *unaided* trace groundedness (~71%, near the 4B citation ceiling).
- [ ] Phase 4 - multimodal / image tables, and transfer to scientific/ML tables (unverified).
