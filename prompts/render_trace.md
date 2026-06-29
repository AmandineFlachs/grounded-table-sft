You are an expert at reading tables and reasoning about them with rigor.

You will be given a markdown table and a question about it. Produce a **grounded,
step-by-step reasoning trace** and a final answer, using ONLY the data in the table.

Output a SINGLE JSON object and nothing else - no prose, no markdown fences, no commentary.

## Output schema

```
{
  "trace_steps": [
    {
      "kind": "<one of: filter | compare | aggregate | select | conclude>",
      "description": "<one short sentence of natural-language reasoning>",
      "cites": [
        {"row": <0-based row index>, "col_name": "<exact column header>", "value": <exact cell value>}
      ]
    }
  ],
  "final_answer": {
    "label": "<see rules below>",
    "rows": [<0-based row indices supporting the answer>],
    "metrics": [<only for trade-off questions: the two metric column names>]
  },
  "operation": "<the structured intent of the question - see Operation below>"
}
```

## Operation

After reasoning, also emit a compact `operation` object capturing the question's intent
using EXACT column-header names (a deterministic engine will execute it). Use the shape for
the question's type:

- winner selection: `{"type":"best_under_constraint","target":"<col>","target_dir":"higher|lower","constraint":"<col>","op":"lt|gte","threshold":<number>}`
- constraint filtering: `{"type":"threshold_filter","conditions":[{"metric":"<col>","op":"lt|gte","T":<number>}]}`
- trade-off: `{"type":"tradeoff_summary","m1":"<col>","m2":"<col>","d1":"higher|lower","d2":"higher|lower"}`

Rules for `operation`: `target_dir`/`d1`/`d2` follow the question's stated direction
("highest"→`higher`, "lowest"→`lower`); `op` is `lt` for "under/below/less than" and `gte`
for "at least/at or above"; do NOT list rows - total/subtotal exclusion is handled
automatically. Column names must match the table headers exactly.

## Rules

- **Cite real cells.** Every `cites` entry must reference an actual cell: a correct 0-based
  `row`, the exact `col_name` as it appears in the header, and the exact `value` from that cell.
- **Ground every step.** Each reasoning step should cite the cells it depends on.
- The **last step must have `kind` = "conclude"** and state the answer.
- **Row indices are 0-based** (the first data row is row 0).
- **Answer label:**
  - Winner selection: the winning model's name. If there is an exact tie, use `"tie: NameA, NameB"`.
  - Constraint filtering: a comma-separated list of matching model names, or `"none"` if no row matches.
  - Trade-off: `"<metric1> vs <metric2>"`, with `rows` = the Pareto-optimal (non-dominated) models and
    `metrics` = the two metric names.
- If **no row satisfies** the constraints, set `label` to `"none"` and `rows` to `[]`, and the
  concluding step must say so explicitly.
- For trade-offs, "best" depends on the metric: higher is better for accuracy/robustness; lower is
  better for latency/cost/memory.

## Task

{TABLE}

Question: {QUESTION}

Respond with the JSON object only.
