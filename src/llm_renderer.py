"""Phase 1.5 - render natural-language traces with the LLM, then validate.

The LLM (Claude Code, headless ``claude -p`` - uses the Max subscription, not the
paid API) is asked to produce a grounded trace + final answer for a table+question
in our JSON schema. We reconstruct an ``Example`` that reuses the ORIGINAL table
and the structured ``spec`` (the known ground truth), but with the LLM's
``trace_steps`` and ``final_answer``. The validator then independently recomputes
the truth and checks the LLM's answer + citations against it.

This is applied first to SYNTHETIC tables (truth known) so we can measure how often
LLM-rendered traces survive validation before trusting the loop on real tables.

Backfill policy: the model cites by ``row`` + ``col_name`` (natural). We backfill
the numeric ``col`` index from ``col_name`` (mechanical), but we DO NOT backfill
``value`` - the model must supply it, so groundedness is genuinely tested.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from .schema import CellRef, Example, GoldAnswer, TraceStep
from .table_utils import render_markdown

PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "render_trace.md"


@dataclass
class RenderOutcome:
    source_id: str
    ok: bool                      # produced a schema-valid Example at all
    example: Optional[Example]    # the reconstructed LLM example (if parseable)
    raw: str                      # raw model output (for debugging failures)
    error: Optional[str] = None   # parse/exec error, if any


# --------------------------------------------------------------------------- #
# prompt + CLI invocation
# --------------------------------------------------------------------------- #
def build_prompt(ex: Example) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{TABLE}", render_markdown(ex.table)).replace("{QUESTION}", ex.question)


def call_claude(prompt: str, cli: str = "claude", timeout: int = 180) -> str:
    """Invoke headless Claude Code and return its text output.

    Runs in an isolated temp cwd so the renderer does not pick up project files,
    and feeds an empty stdin to avoid the CLI's stdin wait.
    """
    exe = shutil.which(cli) or cli
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [exe, "-p", prompt, "--output-format", "text"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",      # decode Claude's output as UTF-8, not the
            errors="replace",      # Windows locale (cp1252), which mangles JSON
            timeout=timeout,
            cwd=tmp,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict:
    """Extract the first balanced top-level JSON object from model output."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in output")


def extract_operation(raw: str) -> dict | None:
    """Pull the top-level 'operation' block from a model output, or None if absent or
    unparseable. Tolerant on purpose: the executor eval path falls back to the model's
    own final_answer when no usable operation is present."""
    try:
        op = extract_json(raw).get("operation")
        return op if isinstance(op, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _cite_from(table, raw_cite: dict) -> CellRef:
    """Build a CellRef from a model-provided citation. Backfills col index from
    col_name; keeps the model's value so it can be checked."""
    col_name = raw_cite["col_name"]
    col = table.col_index(col_name)  # raises if the column name is invented
    return CellRef(row=int(raw_cite["row"]), col=col, col_name=col_name, value=raw_cite["value"])


def reconstruct_example(source: Example, parsed: dict) -> Example:
    table = source.table
    steps = []
    for i, s in enumerate(parsed["trace_steps"]):
        cites = [_cite_from(table, c) for c in s.get("cites", [])]
        steps.append(TraceStep(index=i, kind=s["kind"], description=s["description"], cites=cites))

    fa = parsed["final_answer"]
    gold = GoldAnswer(label=fa.get("label", ""), rows=[int(r) for r in fa.get("rows", [])],
                      metrics=list(fa.get("metrics", [])))

    md = dict(source.metadata)
    md["example_id"] = md.get("example_id", source.table_id) + "_llm"
    md["rendered_from"] = source.metadata.get("example_id")

    return Example(
        table_id=source.table_id,
        domain=source.domain,
        table=table,
        question=source.question,
        question_type=source.question_type,
        gold_answer=gold,
        trace_steps=steps,
        evidence_cells=_dedup([c for s in steps for c in s.cites]),
        trace_source="llm",
        split=source.split,
        metadata=md,
    )


def _dedup(cells: list[CellRef]) -> list[CellRef]:
    seen, out = set(), []
    for c in cells:
        if (c.row, c.col) not in seen:
            seen.add((c.row, c.col))
            out.append(c)
    return out


def render(source: Example, generate_fn=None, cli: str = "claude", timeout: int = 180) -> RenderOutcome:
    """Render a trace for ``source`` and reconstruct an Example.

    ``generate_fn`` is a string->string function (prompt -> raw model output). It
    defaults to headless Claude (``call_claude``); pass a local model's
    ``LocalGenerator.generate`` to score an open model with the same pipeline.
    """
    sid = source.metadata.get("example_id", source.table_id)
    if generate_fn is None:
        generate_fn = lambda p: call_claude(p, cli=cli, timeout=timeout)  # noqa: E731
    try:
        raw = generate_fn(build_prompt(source))
    except Exception as e:  # noqa: BLE001
        return RenderOutcome(sid, ok=False, example=None, raw="", error=f"cli: {e}")
    try:
        parsed = extract_json(raw)
        ex = reconstruct_example(source, parsed)
        return RenderOutcome(sid, ok=True, example=ex, raw=raw)
    except Exception as e:  # noqa: BLE001
        return RenderOutcome(sid, ok=False, example=None, raw=raw, error=f"parse: {e}")
