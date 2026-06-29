"""Convert validated Examples into chat-format SFT records (P3.2).

The training PROMPT is exactly ``build_prompt(ex)`` (so training and inference share
one prompt), and the COMPLETION is the JSON object matching the
``prompts/render_trace.md`` output contract - the very shape that
``reconstruct_example()`` consumes. That symmetry gives a free correctness gate:
serialize → ``extract_json`` → ``reconstruct_example`` → ``validate`` must reproduce
a valid example, or the converter is teaching the model corrupt labels.

Contract details (from render_trace.md): trace steps carry ``kind`` /
``description`` / ``cites`` only (no ``index``, no ``col``); citations are
``row`` / ``col_name`` / ``value``; the answer object is ``final_answer`` with
``label`` / ``rows`` / ``metrics``.
"""

from __future__ import annotations

import json

from .llm_renderer import build_prompt
from .schema import Example


def operation_dict(ex: Example) -> dict:
    """Normalized, NAME-ONLY operation for the executor ('answer by construction').

    Built from metadata['spec'], dropping index/redundant fields (col, c1, c2,
    constraint_dir, per-condition dir). The entity-row universe is deliberately NOT
    included: it is a deterministic table property supplied at eval time via
    table_utils.entity_universe. The threshold operator is canonicalized to the prompt
    alphabet ('ge' -> 'gte')."""
    spec = ex.metadata.get("spec", {})
    t = spec.get("type")
    _op = lambda o: "gte" if o == "ge" else o  # noqa: E731
    if t == "best_under_constraint":
        return {"type": t, "target": spec["target"], "target_dir": spec["target_dir"],
                "constraint": spec["constraint"], "op": _op(spec["op"]), "threshold": spec["threshold"]}
    if t == "threshold_filter":
        return {"type": t,
                "conditions": [{"metric": c["metric"], "op": _op(c["op"]), "T": c["T"]}
                               for c in spec["conditions"]]}
    if t == "tradeoff_summary":
        return {"type": t, "m1": spec["m1"], "m2": spec["m2"], "d1": spec["d1"], "d2": spec["d2"]}
    if t == "extremum":
        return {"type": t, "target": spec["target"], "target_dir": spec["target_dir"]}
    return {"type": t}


def completion_dict(ex: Example) -> dict:
    """The target JSON object the model must learn to emit (render_trace.md contract)."""
    return {
        "trace_steps": [
            {
                "kind": s.kind,
                "description": s.description,
                "cites": [
                    {"row": c.row, "col_name": c.col_name, "value": c.value} for c in s.cites
                ],
            }
            for s in ex.trace_steps
        ],
        "final_answer": {
            "label": ex.gold_answer.label,
            "rows": list(ex.gold_answer.rows),
            "metrics": list(ex.gold_answer.metrics),
        },
        "operation": operation_dict(ex),
    }


def completion_str(ex: Example) -> str:
    """Compact single-line JSON - what the assistant turn should contain verbatim."""
    return json.dumps(completion_dict(ex), ensure_ascii=False)


def to_sft_record(ex: Example) -> dict:
    """Chat-format record: user turn = the render prompt, assistant turn = target JSON.

    The train script (P3.3) applies the model's chat template to ``messages``; the
    same ``build_prompt`` is fed as the user turn at inference, keeping train and
    eval prompts identical.
    """
    return {
        "example_id": ex.metadata.get("example_id", ex.table_id),
        "question_type": ex.question_type.value
        if hasattr(ex.question_type, "value")
        else str(ex.question_type),
        "messages": [
            {"role": "user", "content": build_prompt(ex)},
            {"role": "assistant", "content": completion_str(ex)},
        ],
    }
