"""P2.4 cross-check: do OUR recomputed answers agree with TAT-QA's HUMAN gold?

TAT-QA's selection-style native questions carry human gold answers. We
independently recompute the same extremum from our *ingested* table and compare.
High agreement is direct evidence our ingestion + argmax matches humans on real
tables - the confidence we want before trusting the dataset and the silver loop.

Two complementary methods (a question is handled by at most one):
  * derivation-chain (primary, NL-free): TAT-QA's ``derivation`` for a comparison
    question is a chain like ``918>914>814`` or ``56<63<66`` - the actual compared
    cell values. We locate that line in our ingested table (a row across periods,
    or a column across line items), recompute the extreme, and check the label
    matches gold. Robust and orientation-aware; no text parsing.
  * NL superlative (fallback): "in which year was X largest?" parsed from text,
    for comparison questions whose derivation isn't a clean chain.

Usage:
    python scripts/crosscheck_tatqa.py --emit-eval data/processed/realtable_eval_verified.v0_1_0.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_builder import write_jsonl  # noqa: E402
from src.ingest.tatqa import ingest_context, load_contexts, parse_cell  # noqa: E402
from src.realtable_questions import build_extremum_example  # noqa: E402
from src.trace_validator import validate  # noqa: E402

HIGHER = {"largest", "highest", "greatest", "most", "maximum", "max", "biggest", "longest", "top"}
LOWER = {"lowest", "smallest", "least", "minimum", "min", "fewest", "shortest"}
_SUPER = "|".join(sorted(HIGHER | LOWER, key=len, reverse=True))

_PAT_A = re.compile(
    r"in which (?:year|period|fiscal year)\b.*?\b(?:was|were|did|is|are|has|had)\b\s+"
    r"(?:the\s+)?(?:amount\s+(?:for|of)\s+|value\s+(?:for|of)\s+)?"
    r"(?P<ent>.+?)\s+(?:the\s+|at\s+(?:its|their)\s+)?(?P<dir>" + _SUPER + r")\b",
    re.I,
)
_PAT_B = re.compile(
    r"which (?:year|period|fiscal year)\b\s+(?:has|had|saw|recorded|reported)\s+"
    r"(?:the\s+)?(?P<dir>" + _SUPER + r")\s+(?P<ent>.+?)\s*\??$",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _year(s):
    m = re.search(r"(19|20)\d{2}", str(s))
    return m.group(0) if m else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.01 + 0.005 * max(abs(a), abs(b))


# Total/subtotal rows trivially win an argmax over components; humans exclude
# them when asking "which component/segment is largest", and so do we.
_TOTAL_RE = re.compile(r"\b(total|subtotal|aggregate|grand\s+total)\b", re.I)


def _is_total(label: str) -> bool:
    return bool(_TOTAL_RE.search(str(label)))


# --------------------------------------------------------------------------- #
# method 1: derivation-chain (NL-free)
# --------------------------------------------------------------------------- #
def _chain_values(deriv: str):
    """A pure comparison chain (only > or only <) -> its values, else None.

    The operator only reflects TAT-QA's sort order, NOT which extreme the
    question asks for (e.g. a descending chain can answer a "lowest" question),
    so we use the chain only to LOCATE the compared cells and take direction
    from the question text instead."""
    if not deriv or (">" in deriv and "<" in deriv):
        return None
    op = ">" if ">" in deriv else ("<" if "<" in deriv else None)
    if op is None:
        return None
    vals = []
    for part in deriv.split(op):
        v = parse_cell(part.strip())
        if not isinstance(v, (int, float)):
            return None
        vals.append(v)
    return vals if len(vals) >= 2 else None


def _question_direction(question: str):
    """'higher'/'lower' from the question's superlative word, else None."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words & HIGHER:
        return "higher"
    if words & LOWER:
        return "lower"
    return None


def _line_contains(cells: list[float], vals: list[float]) -> bool:
    """Every chain value matches a distinct cell (greedy, with tolerance)."""
    pool = list(cells)
    for v in vals:
        hit = next((i for i, c in enumerate(pool) if _close(c, v)), None)
        if hit is None:
            return False
        pool.pop(hit)
    return True


def _match_cells(table, fixed: int, axis: str, indices: list[int], vals: list[float]):
    """Return the indices (within ``indices``) whose cells match the chain values
    - i.e. exactly the cells humans compared. ``axis='row'`` varies the column
    (fixed row), ``axis='col'`` varies the row (fixed column). None if not all
    chain values can be matched to distinct cells."""
    def cell(i):
        return table.cell(fixed, i) if axis == "row" else table.cell(i, fixed)
    out, used = [], set()
    for v in vals:
        hit = next((i for i in indices if i not in used and isinstance(cell(i), (int, float))
                    and _close(cell(i), v)), None)
        if hit is None:
            return None
        used.add(hit)
        out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# method 2: NL superlative
# --------------------------------------------------------------------------- #
def _parse_nl(question: str):
    for pat in (_PAT_A, _PAT_B):
        m = pat.search(question)
        if m:
            ent = _norm(re.sub(r"\b(amount|value)\b", "", m.group("ent"), flags=re.I))
            return ent, ("higher" if m.group("dir").lower() in HIGHER else "lower")
    return None, None


def _match_row(ent: str, labels: list[str]):
    norm = [_norm(x) for x in labels]
    exact = [i for i, n in enumerate(norm) if n == ent]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    cand = difflib.get_close_matches(ent, norm, n=1, cutoff=0.85)
    if not cand:
        return None
    i = norm.index(cand[0])
    return None if norm.count(norm[i]) > 1 else i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(ROOT / "data" / "raw" / "tatqa" / "tatqa_dataset_dev.json"))
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--emit-eval", default=None)
    ap.add_argument("--out", default=None, help="write a JSON summary of the agreement stats")
    args = ap.parse_args()

    contexts = load_contexts(args.path)
    stats = Counter()
    by_method = Counter()
    disagreements, eval_examples = [], []

    for ci, ctx in enumerate(contexts):
        it = ingest_context(ctx, ci)
        if it.confidence != "high":
            continue
        t = it.table
        labels = [str(t.cell(r, 0)) for r in range(len(t.rows))]

        for q in ctx.get("questions", []):
            if q.get("answer_from") not in ("table", "table-text"):
                continue
            gold = q.get("answer")
            gold_list = gold if isinstance(gold, list) else [gold]

            ex = None
            our = None
            agree = False
            method = None

            vals = _chain_values(q.get("derivation", ""))
            direction = _question_direction(q["question"]) if vals else None
            if vals and direction:
                period_answer = all(_year(g) is not None for g in gold_list) and bool(gold_list)
                pick = max if direction == "higher" else min
                if period_answer:
                    # chain spans one row across period columns; require a unique row
                    rmatch = [r for r in range(len(t.rows))
                              if _line_contains([t.cell(r, c) for c in range(1, len(t.headers))
                                                 if isinstance(t.cell(r, c), (int, float))], vals)]
                    if len(rmatch) == 1:
                        method = "derivation"
                        idx = rmatch[0]
                        num_cols = [c for c in range(1, len(t.headers))
                                    if isinstance(t.cell(idx, c), (int, float))]
                        # recompute over ONLY the cells humans compared (the chain),
                        # so 2D cross-tabs / partial comparisons stay faithful
                        chain_cols = _match_cells(t, idx, "row", num_cols, vals)
                        if chain_cols:
                            bc = pick(chain_cols, key=lambda c: t.cell(idx, c))
                            our = _year(t.headers[bc])
                            agree = our is not None and our in {_year(g) for g in gold_list}
                            # emit only when the chain covers the whole row (clean 1D)
                            if agree and args.emit_eval and len(chain_cols) == len(num_cols):
                                ex = build_extremum_example(it, labels[idx], direction,
                                                            q["question"], gold)
                        else:
                            method = None
                    else:
                        stats["ambiguous"] += 1
                else:
                    # chain spans one numeric column across NON-total entity rows
                    nz = [r for r in range(len(t.rows)) if not _is_total(labels[r])]
                    cmatch = [c for c in range(1, len(t.headers)) if t.column_types[c] == "numeric"
                              and _line_contains([t.cell(r, c) for r in nz
                                                  if isinstance(t.cell(r, c), (int, float))], vals)]
                    if len(cmatch) == 1:
                        method = "derivation"
                        c = cmatch[0]
                        chain_rows = _match_cells(t, c, "col", nz, vals)
                        if chain_rows:
                            br = pick(chain_rows, key=lambda r: t.cell(r, c))
                            our = _norm(labels[br])
                            golds = {_norm(g) for g in gold_list}
                            agree = any(our == g or (g and (our in g or g in our)) for g in golds)
                            # entity answers are cross-checked but not emitted (totals
                            # exclusion isn't modelled by the generic extremum builder)
                        else:
                            method = None
                    else:
                        stats["ambiguous"] += 1

            if method is None:  # fallback: NL superlative (period answers)
                ent, direction = _parse_nl(q["question"])
                if ent is None:
                    continue
                gold_years = {_year(g) for g in gold_list}
                gold_years.discard(None)
                period_cols = [c for c in range(len(t.headers))
                               if t.column_types[c] == "numeric" and _year(t.headers[c])]
                if not gold_years or not period_cols:
                    continue
                ri = _match_row(ent, labels)
                if ri is None:
                    stats["row_unmatched"] += 1
                    continue
                vals = [(c, t.cell(ri, c)) for c in period_cols if isinstance(t.cell(ri, c), (int, float))]
                if not vals:
                    stats["row_unmatched"] += 1
                    continue
                method = "nl"
                bc = (max if direction == "higher" else min)(vals, key=lambda cv: cv[1])[0]
                our = _year(t.headers[bc])
                agree = our in gold_years
                if agree and args.emit_eval:
                    ex = build_extremum_example(it, labels[ri], direction, q["question"], gold)

            if method is None:
                continue
            stats["checked"] += 1
            by_method[method] += 1
            if agree:
                stats["agree"] += 1
                if ex is not None and validate(ex).valid:
                    eval_examples.append(ex)
                    stats["eval_emitted"] += 1
            else:
                stats["disagree"] += 1
                disagreements.append({"q": q["question"], "ours": our, "gold": gold_list,
                                      "method": method, "uid": it.source_uid})

    print("=" * 70)
    print(f"cross-checked          : {stats['checked']}  (by method: {dict(by_method)})")
    print(f"  rows unmatched (NL)  : {stats['row_unmatched']}")
    if stats["checked"]:
        print(f"  AGREE with human gold: {stats['agree']} "
              f"({100*stats['agree']/stats['checked']:.1f}%)")
        print(f"  disagree             : {stats['disagree']}")
    if disagreements:
        print("\ndisagreements (sample):")
        for r in disagreements[: args.show]:
            print(f"  [{r['method']}] ours={r['ours']} gold={r['gold']} | {r['q']}")
    if args.emit_eval:
        Path(args.emit_eval).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(eval_examples, args.emit_eval)
        print(f"\nTAT-QA-gold-anchored eval examples written: {stats['eval_emitted']} -> {args.emit_eval}")
    if args.out:
        summary = {"checked": stats["checked"], "agree": stats["agree"],
                   "disagree": stats["disagree"],
                   "agree_pct": (100 * stats["agree"] / stats["checked"]) if stats["checked"] else 0.0,
                   "by_method": dict(by_method),
                   "note": "OUR recomputed extrema vs TAT-QA's native human gold; "
                           "third-party annotation, not project-verified"}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
