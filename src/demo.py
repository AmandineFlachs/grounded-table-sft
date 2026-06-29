"""Demo assembly - turn one SAVED model output into a shown 'card' (no GPU, no model run).

This is the canonical, front-end-agnostic path that both demo surfaces reuse:

    scripts/demo.py        -> terminal replay
    results/demo/index.html -> shareable static page

A card walks the exact production pipeline on a stored example:

  1. the model's `operation` (comprehension it emitted)        -> parsed from the saved raw output
  2. the deterministic engine answer (src.executor.compute_answer)  -> arithmetic by proven code
  3. the engine-READ evidence cells (src.executor.evidence_for)     -> citations grounded by construction
  4. the out-of-vocabulary SAFETY GATE (gate_fires, below)          -> fall back to the model when the
     emitted op type isn't supported by the question (do-no-harm)

`gate_fires` lives HERE (not in scripts/eval_gate.py) so the gate has a single definition shared by
the Step-B re-score and the demo - change the heuristic in one place and both move together.
"""
from __future__ import annotations

import html
import re

from .executor import compute_answer, evidence_for
from .llm_renderer import extract_json
from .schema import Example, Table
from .table_utils import _fmt, entity_universe

# --------------------------------------------------------------------------- #
# the out-of-vocabulary safety gate (canonical home; imported by eval_gate.py)
# --------------------------------------------------------------------------- #
# best_under_constraint / threshold_filter REQUIRE threshold language; tradeoff_summary REQUIRES
# trade-off language. If the question lacks the signal the emitted op type needs, the operation is
# fabricated -> don't trust the engine, fall back to the model's own answer.
_THRESH = re.compile(r"\b(while|under|at least|at most|below|above|exceed|exceeds|"
                     r"less than|greater than|no more than|no less than|of at least|of at most)\b", re.I)
_TRADE = re.compile(r"\b(pareto|trade-?off|optimal|optimize|optimise|maxim|minim)\b", re.I)


def gate_fires(op: dict | None, question: str) -> bool:
    """True => the emitted operation is not supported by the question (fabricated) => fall back
    to the model's own answer. Deployable: uses only the question text + op type, never the gold."""
    if not isinstance(op, dict):
        return True  # no usable operation at all -> trust the model's own answer
    t = op.get("type")
    if t in ("best_under_constraint", "threshold_filter"):
        return _THRESH.search(question) is None
    if t == "tradeoff_summary":
        return _TRADE.search(question) is None
    return True  # unknown/extremum op type -> fall back


# --------------------------------------------------------------------------- #
# card assembly
# --------------------------------------------------------------------------- #
def _cells(refs) -> list[dict]:
    return [{"row": c.row, "col": c.col, "col_name": c.col_name, "value": c.value} for c in refs]


def _answer(ans) -> dict:
    return {"label": ans.label, "rows": list(ans.rows), "metrics": list(ans.metrics)}


def build_card(ex: Example, raw: str) -> dict:
    """Assemble one demo card from an Example and its SAVED raw model output.

    Never raises on bad model output: parse/engine failures are recorded on the card so the demo
    can show the gate catching them (that IS the story for the out-of-distribution slice)."""
    table = ex.table
    card: dict = {
        "example_id": ex.metadata.get("example_id", ex.table_id),
        "question": ex.question,
        "question_type": ex.question_type.value,
        "headers": list(table.headers),
        "rows": [[_fmt(v) for v in r] for r in table.rows],
        "row_labels": list(table.row_labels) if table.row_labels else None,
        "model_trace": [],
        "model_answer": None,
        "operation": None,
        "engine_answer": None,
        "engine_error": None,
        "evidence_cells": [],
        "gate_fired": True,
        "system_answer": None,
        "highlight": [],
    }

    # 1. parse the saved model output (prose trace + operation + the model's own final answer)
    try:
        parsed = extract_json(raw)
    except Exception:  # noqa: BLE001
        parsed = {}
    for s in parsed.get("trace_steps", []):
        card["model_trace"].append({
            "kind": s.get("kind", ""),
            "description": s.get("description", ""),
            "cites": [{"row": c.get("row"), "col_name": c.get("col_name"), "value": c.get("value")}
                      for c in s.get("cites", []) if isinstance(c, dict)],
        })
    fa = parsed.get("final_answer") or {}
    model_rows = [int(r) for r in fa.get("rows", []) if isinstance(r, (int, float))]
    card["model_answer"] = {"label": fa.get("label", ""), "rows": model_rows}
    op = parsed.get("operation")
    card["operation"] = op if isinstance(op, dict) else None

    # 2 + 3. engine answer and engine-read evidence (grounded by construction), over the entity
    # universe (totals excluded) so the inference-time universe matches the gold spec.
    if isinstance(op, dict):
        op2 = {**op, "rows": entity_universe(table)}
        try:
            eng = compute_answer(table, op2)
            ev = evidence_for(table, op2)
            card["engine_answer"] = _answer(eng)
            card["evidence_cells"] = _cells(ev)
        except Exception as exc:  # noqa: BLE001  (unknown op type, invented column, ...)
            card["engine_error"] = f"{type(exc).__name__}: {exc}"

    # 4. safety gate: when the op isn't supported by the question, fall back to the model's answer
    fired = gate_fires(op, ex.question)
    card["gate_fired"] = fired
    if not fired and card["engine_answer"] is not None:
        card["system_answer"] = {**card["engine_answer"], "source": "engine"}
        card["highlight"] = [[c["row"], c["col"]] for c in card["evidence_cells"]]
    else:
        # gate fired (or engine unusable): trust the model. Highlight the cells the model cited.
        card["system_answer"] = {"label": card["model_answer"]["label"],
                                 "rows": card["model_answer"]["rows"], "metrics": [], "source": "model"}
        hl = []
        for s in card["model_trace"]:
            for c in s["cites"]:
                col = _col_index_safe(table, c.get("col_name"))
                if col is not None and isinstance(c.get("row"), int):
                    hl.append([c["row"], col])
        card["highlight"] = hl

    card["gold_answer"] = _answer(ex.gold_answer)
    return card


def _col_index_safe(table: Table, name) -> int | None:
    try:
        return table.col_index(name)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# shared HTML rendering of a card (used by scripts/demo.py and scripts/build_writeup.py)
# --------------------------------------------------------------------------- #
def _h(s) -> str:
    return html.escape(str(s))


def fmt_answer(a: dict | None) -> str:
    if not a:
        return "-"
    rows = ",".join(map(str, a.get("rows", [])))
    return f"{a.get('label', '')}  (rows [{rows}])"


def card_table_html(card: dict) -> str:
    hl = {tuple(p) for p in card["highlight"]}
    th = "".join(f"<th>{_h(h)}</th>" for h in card["headers"])
    trs = []
    for ri, row in enumerate(card["rows"]):
        tds = []
        for ci, v in enumerate(row):
            cls = ' class="hl"' if (ri, ci) in hl else ""
            tds.append(f"<td{cls}>{_h(v)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def card_html(card: dict) -> str:
    """Render one assembled card as a self-contained <section class=card> block. Styling comes from
    DEMO_CARD_CSS, included by whichever page embeds the card."""
    tag = "OOD" if card["ood"] else "in-dist"
    trace = "".join(
        f"<li><span class=kind>{_h(s['kind'])}</span> {_h(s['description'])}</li>"
        for s in card["model_trace"])
    ev = "".join(f"<li><code>{_h(c['col_name'])}</code>[row {c['row']}] = <b>{_h(c['value'])}</b></li>"
                 for c in card["evidence_cells"]) or "<li><i>none</i></li>"
    sys_a = card["system_answer"]
    if card["gate_fired"]:
        gate = ('<span class="badge fire">GATE FIRED</span> op not supported by the question → '
                'fall back to the model’s own answer')
    else:
        gate = ('<span class="badge pass">GATE PASS</span> op supported by the question → '
                'trust the engine')
    return f"""
  <section class=card>
    <header><span class="tag {tag.lower().replace('-', '')}">{_h(tag)}</span>
      <span class=type>{_h(card['question_type'])}</span>
      <span class=eid>{_h(card['example_id'])}</span></header>
    <p class=q>{_h(card['question'])}</p>
    {card_table_html(card)}
    <div class=cols>
      <div><h4>Model trace <small>(model-authored prose)</small></h4><ol class=trace>{trace}</ol></div>
      <div><h4>Grounded evidence <small>(cells the engine read, by construction)</small></h4>
           <ul class=ev>{ev}</ul></div>
    </div>
    <table class=ans>
      <tr><td>Engine answer</td><td>{_h(fmt_answer(card['engine_answer']))}</td></tr>
      <tr><td>Model’s own</td><td>{_h(fmt_answer(card['model_answer']))}</td></tr>
      <tr class=gate><td>Gate</td><td>{gate}</td></tr>
      <tr class=sys><td>System answer</td><td><b>{_h(fmt_answer(sys_a))}</b> <small>[{_h(sys_a['source'])}]</small></td></tr>
      <tr><td>Gold</td><td>{_h(fmt_answer(card['gold_answer']))}</td></tr>
    </table>
  </section>"""


DEMO_CARD_CSS = """
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 22px;
  box-shadow:0 1px 2px rgba(0,0,0,.03)}
.card header{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.card .tag{font-size:11px;font-weight:700;letter-spacing:.04em;padding:2px 8px;border-radius:999px;color:#fff;background:#495057}
.card .tag.ood{background:#c92a2a}.card .tag.indist{background:#1864ab}
.card .type{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}
.card .eid{margin-left:auto;font-size:11px;color:#aeb6c0;font-family:ui-monospace,monospace}
.card .q{font-weight:600;margin:.2em 0 14px}
.card table{border-collapse:collapse;font-size:13px;width:100%;margin:0}
.card th,.card td{border:1px solid var(--line);padding:5px 9px;text-align:right}
.card th:first-child,.card td:first-child{text-align:left}
.card thead th{background:#f1f3f5;font-weight:600}
.card td.hl{background:#fff3bf;box-shadow:inset 0 0 0 2px #f2cb05;font-weight:700}
.card .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:16px 0}
@media(max-width:640px){.card .cols{grid-template-columns:1fr}}
.card h4{margin:0 0 6px;font-size:13px}.card h4 small{color:var(--muted);font-weight:400}
.card ol.trace,.card ul.ev{margin:0;padding-left:18px}.card ol.trace li,.card ul.ev li{margin:.2em 0}
.card .kind{display:inline-block;font:11px ui-monospace,monospace;background:#eef1f4;color:#495057;
  padding:0 6px;border-radius:4px;margin-right:4px}
.card table.ans{margin-top:6px}.card table.ans td:first-child{color:var(--muted);width:130px;text-align:left}
.card table.ans td{border:none;border-top:1px solid var(--line);text-align:left;padding:6px 9px}
.card tr.sys td{background:#f1f8f2}.card tr.gate td{font-size:13px}
.card .badge{font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;color:#fff;margin-right:6px}
.card .badge.pass{background:#2b8a3e}.card .badge.fire{background:#c92a2a}
"""
