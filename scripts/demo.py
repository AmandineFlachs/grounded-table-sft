"""Demo - replay a saved example as a shown 'card': table + grounded trace + cited cells + answer.

No GPU, no model call: it reuses the SAVED locked-test outputs and the production assembly path
(src.demo.build_card -> executor answer + engine-read evidence + safety gate). Two front-ends from
one assembler: this terminal view and a self-contained results/demo/index.html.

    python scripts/demo.py                 # featured set (1 per type + 1 OOD gate-fires) + write HTML
    python scripts/demo.py --list          # every available example_id, with flags
    python scripts/demo.py --id <eid>      # one specific card to the terminal
    python scripts/demo.py --no-html       # skip writing the HTML page

The HTML page is the shareable showcase; the terminal view is for quick reproducible inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo import build_card, card_html                  # noqa: E402
from src.schema import Example                              # noqa: E402
from src.splits import load_jsonl                            # noqa: E402
from src.theme import brand_font_link, theme_css            # noqa: E402

IN_DIST = ("results/p3_4b_exec_TEST.json", "data/processed/eval_test.v0_1_0.jsonl")
OOD = ("results/p3_4b_exec_TEST_verified.json", "data/processed/realtable_eval_verified.v0_1_0.jsonl")
HTML_OUT = "results/demo/index.html"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_pool(results_path: str, source_path: str, ood: bool) -> list[dict]:
    """Return ordered records: {detail, ex, ood} for every saved example with raw output."""
    det = json.loads((ROOT / results_path).read_text(encoding="utf-8"))["details"]
    src = {}
    for r in load_jsonl(ROOT / source_path):
        ex = Example.model_validate(r)
        src[ex.metadata.get("example_id", ex.table_id)] = ex
    out = []
    for d in det:
        ex = src.get(d["example_id"])
        if ex is not None and d.get("raw"):
            out.append({"detail": d, "ex": ex, "ood": ood})
    return out


def _all_records() -> list[dict]:
    return _load_pool(*IN_DIST, ood=False) + _load_pool(*OOD, ood=True)


def _make_card(rec: dict) -> dict:
    card = build_card(rec["ex"], rec["detail"]["raw"])
    card["ood"] = rec["ood"]
    card["engine_ok"] = bool(rec["detail"].get("engine_ok"))
    card["model_ok"] = bool(rec["detail"].get("model_ok"))
    card["grounded"] = bool(rec["detail"].get("grounded"))
    return card


def _featured(records: list[dict]) -> list[dict]:
    """One clean in-distribution card per trained type (engine-correct, grounded, gate passes),
    plus one out-of-distribution card where the gate fires and saves the answer."""
    chosen, seen_types = [], set()
    for rec in records:
        if rec["ood"]:
            continue
        d = rec["detail"]
        t = d["type"]
        if t not in seen_types and d.get("engine_ok") and d.get("grounded"):
            card = _make_card(rec)
            ea = card["engine_answer"]
            if not card["gate_fired"] and ea and ea["rows"]:  # prefer a non-empty answer to showcase
                chosen.append(card)
                seen_types.add(t)
    for rec in records:           # one OOD card (the gate-fires story)
        if rec["ood"]:
            chosen.append(_make_card(rec))
            break
    return chosen


# --------------------------------------------------------------------------- #
# terminal rendering
# --------------------------------------------------------------------------- #
def _grid(card: dict) -> str:
    hl = {tuple(p) for p in card["highlight"]}
    headers = card["headers"]
    body = card["rows"]
    cols = list(zip(*([headers] + body))) if body else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(cells, ri):
        out = []
        for ci, v in enumerate(cells):
            s = str(v).ljust(widths[ci])
            if ri is not None and (ri, ci) in hl:
                s = f"[{str(v)}]".ljust(widths[ci] + 2)
            out.append(s)
        return " | ".join(out)

    lines = [fmt_row(headers, None), "-+-".join("-" * w for w in widths)]
    for ri, row in enumerate(body):
        lines.append(fmt_row(row, ri))
    return "\n".join(lines)


def _ans(a: dict | None) -> str:
    if not a:
        return "-"
    rows = ",".join(map(str, a.get("rows", [])))
    return f"{a.get('label','')}  (rows [{rows}])"


def print_card(card: dict) -> None:
    bar = "=" * 78
    tag = "OUT-OF-DISTRIBUTION" if card["ood"] else "in-distribution"
    print(f"\n{bar}\n CARD  {card['example_id']}   [{tag}]\n type: {card['question_type']}\n{'-'*78}")
    print(f" Q: {card['question']}\n")
    print(_grid(card))
    print("\n MODEL TRACE (model-authored prose):")
    for i, s in enumerate(card["model_trace"], 1):
        print(f"   {i}. [{s['kind']}] {s['description']}")
    print("\n GROUNDED EVIDENCE (cells the engine read - grounded by construction):")
    if card["evidence_cells"]:
        for c in card["evidence_cells"]:
            print(f"   - {c['col_name']}[row {c['row']}] = {c['value']}")
    else:
        print("   (none - engine produced no usable evidence)")
        if card["engine_error"]:
            print(f"   engine error: {card['engine_error']}")
    print()
    print(f" ENGINE answer : {_ans(card['engine_answer'])}")
    print(f" MODEL's own   : {_ans(card['model_answer'])}")
    sys_a = card["system_answer"]
    if card["gate_fired"]:
        print(f" GATE          : FIRED - op unsupported by the question → fall back to the MODEL")
    else:
        print(f" GATE          : pass - op supported by the question → trust the ENGINE")
    print(f" SYSTEM answer : {_ans(sys_a)}   [source: {sys_a['source']}]")
    print(f" GOLD          : {_ans(card['gold_answer'])}")


# --------------------------------------------------------------------------- #
# HTML page (self-contained, shareable) - card rendering lives in src.demo
# --------------------------------------------------------------------------- #
# Shared "Field Notes" design system (src/theme.py) - the standalone demo page uses the same
# skeleton + personal-brand tokens as docs/index.html, so its cards render fully themed.
_PAGE_CSS = theme_css("personal")


def write_html(cards: list[dict], out_path: Path) -> None:
    body = "".join(card_html(c) for c in cards)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounded Table Reasoning — demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{brand_font_link('personal')}" rel="stylesheet">
<style>{_PAGE_CSS}</style></head>
<body>
<div class="topbar"></div>
<header class="hero"><div class="inner"><div class="head">
  <h1>Grounded Table Reasoning — demo</h1>
  <p class="subtitle">Each card replays a held-out example through the system: the model emits an
  <i>operation</i> (its comprehension); a deterministic engine computes the answer and returns the
  exact cells it read (<b>highlighted</b> — citations grounded by construction); a safety gate falls
  back to the model’s own answer when the question doesn’t support the emitted operation. No model
  is called here — these are saved locked-test outputs, re-assembled on CPU.</p>
</div></div></header>
<main>
{body}
</main>
<footer><div class="inner"><p class="note">Generated by <code>scripts/demo.py</code> from saved
outputs (<code>results/p3_4b_exec_TEST.json</code>, <code>…_verified.json</code>). See
<code>RESULTS.md</code> for the held-out numbers.</p></div></footer>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="show one example_id")
    ap.add_argument("--list", action="store_true", help="list available example_ids")
    ap.add_argument("--html", default=HTML_OUT, help=f"HTML output path (default {HTML_OUT})")
    ap.add_argument("--no-html", action="store_true", help="do not write the HTML page")
    args = ap.parse_args()

    try:  # Windows consoles default to cp1252; the cards use unicode glyphs (→ - […])
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    records = _all_records()

    if args.list:
        print(f"{'example_id':<52} {'type':<22} eng grd gate")
        for rec in records:
            c = _make_card(rec)
            gate = "FIRE" if c["gate_fired"] else "pass"
            print(f"{c['example_id']:<52} {c['question_type']:<22} "
                  f"{'Y' if c['engine_ok'] else '.':>3} {'Y' if c['grounded'] else '.':>3} {gate:>4}")
        print(f"\n{len(records)} examples ({sum(r['ood'] for r in records)} out-of-distribution)")
        return 0

    if args.id:
        for rec in records:
            if rec["ex"].metadata.get("example_id", rec["ex"].table_id) == args.id:
                print_card(_make_card(rec))
                return 0
        print(f"no example with id {args.id!r} (try --list)", file=sys.stderr)
        return 1

    cards = _featured(records)
    for c in cards:
        print_card(c)
    if not args.no_html:
        out = Path(args.html)
        if not out.is_absolute():
            out = ROOT / out
        write_html(cards, out)
        print(f"\n  -> wrote {out.relative_to(ROOT)}  ({len(cards)} cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
