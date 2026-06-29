"""Build a BLIND human-annotation interface for the in-distribution ground-truth benchmark (P3.9).

Draws a stratified random sample of in-distribution test examples and writes a SELF-CONTAINED
`results/anchor/annotate.html` with the cards embedded (raw table + question + row choices, but
NO spec / gold / engine answer - blind by construction). Open it in a browser, answer each card,
and export `answers_human.json`; score it with `scripts/anchor_blind.py score-human`.

Default sample = the SAME 36 cards the blind LLM annotated (seed 0, n 36) so we get a clean
human-vs-LLM-vs-gold three-way on identical items. Override with --n / --seed.

    python scripts/annotate_build.py --n 36 --seed 0
    # open results/anchor/annotate.html, annotate, export answers_human.json
    python scripts/anchor_blind.py score-human --answers results/anchor/answers_human.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import Example                 # noqa: E402
from src.splits import load_jsonl              # noqa: E402
from scripts.anchor_blind import (             # noqa: E402
    TEST, ENGINE_RESULTS, TRAINED, _row_labels, _gold_label_set,
)

HTML_OUT = "results/anchor/annotate.html"
KEY_OUT = "results/anchor/key_human.json"


def sample(n: int, seed: int) -> list[Example]:
    recs = load_jsonl(Path(TEST))
    by_type = defaultdict(list)
    for r in recs:
        ex = Example.model_validate(r)
        if ex.question_type.value in TRAINED:
            by_type[ex.question_type.value].append(ex)
    rng = random.Random(seed)
    per = n // len(TRAINED)
    out = []
    for qt in TRAINED:
        pool = sorted(by_type[qt], key=lambda e: e.metadata.get("example_id", e.table_id))
        out.extend(rng.sample(pool, min(per, len(pool))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine_ok = {}
    if Path(ENGINE_RESULTS).exists():
        for d in json.loads(Path(ENGINE_RESULTS).read_text(encoding="utf-8"))["details"]:
            engine_ok[d["example_id"]] = bool(d.get("engine_ok"))

    exs = sample(args.n, args.seed)
    cards, key = [], []
    for cid, ex in enumerate(exs):
        eid = ex.metadata.get("example_id", ex.table_id)
        labels = _row_labels(ex)
        cards.append({                          # BLIND card - no gold, no engine answer
            "card_id": cid,
            "type": ex.question_type.value,
            "question": ex.question,
            "headers": list(ex.table.headers),
            "rows": [[("" if c is None else c) for c in row] for row in ex.table.rows],
            "row_labels": labels,
        })
        key.append({
            "card_id": cid, "example_id": eid,
            "gold_rows": list(ex.gold_answer.rows),
            "gold_label_set": sorted(_gold_label_set(ex)),
            "engine_ok": engine_ok.get(eid),
        })

    Path(KEY_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(KEY_OUT).write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__CARDS_JSON__", json.dumps(cards, ensure_ascii=False))
    Path(HTML_OUT).write_text(html, encoding="utf-8")
    print(f"wrote {len(cards)} blind cards -> {HTML_OUT}")
    print(f"wrote private key            -> {KEY_OUT}")
    print(f"by type: { {qt: sum(c['type']==qt for c in cards) for qt in TRAINED} }")
    print(f"\nOpen {HTML_OUT} in a browser, annotate, export answers_human.json, then:")
    print("  python scripts/anchor_blind.py score-human --answers results/anchor/answers_human.json")
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind table-answer annotation</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --ink:#e6e8ec; --mut:#9aa3b2; --acc:#4f9cff; --ok:#3fb950; --warn:#d29922; --line:#2a2f3a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  header { position:sticky; top:0; background:#0c0e12; border-bottom:1px solid var(--line); padding:10px 16px; z-index:5; }
  .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin-top:8px; }
  .bar > i { display:block; height:100%; background:var(--acc); width:0; transition:width .2s; }
  .wrap { max-width:980px; margin:0 auto; padding:18px 16px 120px; }
  .note { color:var(--mut); font-size:13px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }
  .q { font-size:17px; font-weight:600; margin:4px 0 14px; }
  .tag { display:inline-block; font-size:11px; color:var(--mut); border:1px solid var(--line); border-radius:20px; padding:1px 9px; margin-right:8px; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 4px; display:block; overflow-x:auto; }
  th,td { border:1px solid var(--line); padding:5px 9px; text-align:right; white-space:nowrap; }
  th { background:#0c0e12; color:var(--mut); position:sticky; top:0; }
  td.lbl, th.lbl { text-align:left; }
  tr.sel td { background:rgba(79,156,255,.12); }
  .pick { width:34px; text-align:center; }
  .controls { margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  button { background:#222733; color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:8px 14px; cursor:pointer; font-size:14px; }
  button:hover { border-color:var(--acc); }
  button.primary { background:var(--acc); border-color:var(--acc); color:#04101f; font-weight:600; }
  button.toggle.on { background:var(--warn); border-color:var(--warn); color:#1a1300; }
  button.none.on { background:#2d333b; border-color:var(--mut); }
  textarea { width:100%; background:#0c0e12; color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:8px; margin-top:10px; font:13px system-ui; }
  .nav { display:flex; justify-content:space-between; margin-top:16px; }
  .grid { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
  .grid b { width:26px; height:24px; display:grid; place-items:center; border:1px solid var(--line); border-radius:6px; font-size:11px; cursor:pointer; color:var(--mut); }
  .grid b.done { background:var(--ok); color:#04210b; border-color:var(--ok); }
  .grid b.amb { background:var(--warn); color:#1a1300; border-color:var(--warn); }
  .grid b.cur { outline:2px solid var(--acc); }
  footer { position:fixed; bottom:0; left:0; right:0; background:#0c0e12; border-top:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; align-items:center; justify-content:center; }
</style></head>
<body>
<header>
  <b>Blind table-answer annotation</b> &nbsp;<span class="note">Answer each from the table + question ONLY. There are no "right answers" shown - you are the independent oracle.</span>
  <div class="bar"><i id="prog"></i></div>
</header>
<div class="wrap">
  <p class="note">Tick the row(s) that answer the question. Use <b>None satisfy</b> if no row qualifies, or <b>Ambiguous / ill-posed</b> if the question can't be answered as written. Your work auto-saves in this browser; click <b>Export</b> when done.</p>
  <div id="view"></div>
  <div class="grid" id="grid"></div>
</div>
<footer>
  <button id="prev">&larr; Prev</button>
  <span class="note" id="count"></span>
  <button id="next" class="primary">Next &rarr;</button>
  <button id="export">Export answers</button>
</footer>
<script>
const CARDS = __CARDS_JSON__;
const KEY = "tbl_anno_v1";
let A = JSON.parse(localStorage.getItem(KEY) || "{}");   // card_id -> {rows:[], none, ambiguous, note}
let i = 0;
const $ = s => document.querySelector(s);
function get(c){ return A[c.card_id] || (A[c.card_id]={rows:[],none:false,ambiguous:false,note:""}); }
function answered(c){ const a=A[c.card_id]; return a && (a.rows.length||a.none||a.ambiguous); }
function save(){ localStorage.setItem(KEY, JSON.stringify(A)); render(); }

function render(){
  const c = CARDS[i], a = get(c);
  const hdr = c.headers.map(h=>`<th class="${h===c.headers[0]?'lbl':''}">${esc(h)}</th>`).join("");
  const body = c.rows.map((row,ri)=>{
    const cells = row.map((v,ci)=>`<td class="${ci===0?'lbl':''}">${esc(v)}</td>`).join("");
    const on = a.rows.includes(ri);
    return `<tr class="${on?'sel':''}"><td class="pick"><input type="checkbox" data-row="${ri}" ${on?'checked':''}></td>${cells}</tr>`;
  }).join("");
  $("#view").innerHTML = `
    <div class="card">
      <div><span class="tag">card ${i+1} / ${CARDS.length}</span><span class="tag">${c.type}</span></div>
      <div class="q">${esc(c.question)}</div>
      <table><thead><tr><th class="pick"></th>${hdr}</tr></thead><tbody>${body}</tbody></table>
      <div class="controls">
        <button class="none toggle ${a.none?'on':''}" id="bnone">None satisfy</button>
        <button class="toggle ${a.ambiguous?'on':''}" id="bamb">Ambiguous / ill-posed</button>
      </div>
      <textarea id="note" rows="2" placeholder="optional note (e.g. why ambiguous, or a tie)">${esc(a.note||"")}</textarea>
    </div>`;
  $("#prog").style.width = (100*CARDS.filter(answered).length/CARDS.length)+"%";
  $("#count").textContent = CARDS.filter(answered).length + " / " + CARDS.length + " answered";
  $("#grid").innerHTML = CARDS.map((c,k)=>{
    const a=A[c.card_id], cls=[]; if(a&&a.ambiguous)cls.push('amb'); else if(answered(c))cls.push('done'); if(k===i)cls.push('cur');
    return `<b class="${cls.join(' ')}" data-k="${k}">${k+1}</b>`;}).join("");

  $("#view").querySelectorAll('input[data-row]').forEach(cb=>cb.onchange=e=>{
    const r=+e.target.dataset.row, a=get(c);
    if(e.target.checked){ if(!a.rows.includes(r))a.rows.push(r); a.none=false; }
    else a.rows=a.rows.filter(x=>x!==r);
    save();
  });
  $("#bnone").onclick=()=>{ const a=get(c); a.none=!a.none; if(a.none)a.rows=[]; save(); };
  $("#bamb").onclick=()=>{ const a=get(c); a.ambiguous=!a.ambiguous; save(); };
  $("#note").oninput=e=>{ get(c).note=e.target.value; localStorage.setItem(KEY,JSON.stringify(A)); };
  $("#grid").querySelectorAll('b').forEach(b=>b.onclick=()=>{ i=+b.dataset.k; render(); });
}
function esc(s){ return String(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }
$("#prev").onclick=()=>{ if(i>0){i--;render();} };
$("#next").onclick=()=>{ if(i<CARDS.length-1){i++;render();} };
$("#export").onclick=()=>{
  const out={}; CARDS.forEach(c=>{ const a=A[c.card_id]||{rows:[],none:false,ambiguous:false,note:""};
    out[c.card_id]={rows:a.rows.slice().sort((x,y)=>x-y), labels:a.rows.map(r=>c.row_labels[r]),
                    none:!!a.none, ambiguous:!!a.ambiguous, note:a.note||""}; });
  const blob=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob), link=document.createElement("a");
  link.href=url; link.download="answers_human.json"; link.click(); URL.revokeObjectURL(url);
};
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
