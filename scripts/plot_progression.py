"""Publication-style progression figure for the README / docs (docs/assets/progression.png).

A grouped bar chart of the three headline metrics (valid trace / cell grounding /
answer accuracy) across the six pipeline stages. The model-only stages are flat;
the deterministic executor is what closes the answer-accuracy gap.

Numbers are pulled from the saved result files via build_writeup.gather_numbers, so
this figure cannot drift from the write-up. The rendered PNG is committed, so a fresh
clone shows it without regenerating. Regenerating needs matplotlib (an optional dev
dependency, deliberately NOT in the minimal requirements.txt):

    pip install matplotlib
    python scripts/plot_progression.py                 # -> docs/assets/progression.png
    python scripts/plot_progression.py --out some/other/path.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_writeup import gather_numbers  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "assets" / "progression.png"


def _f(s) -> float:
    return float(str(s).rstrip("%"))


def stages():
    N = gather_numbers()
    b4 = N["base4b"]
    eng = _f(N["engine"])
    # (label, valid, grounding, answer). Historical rows mirror the write-up stage table;
    # the 4B-zero-shot / executor / grounded-citations rows are sourced from result files.
    return [
        ("Qwen3-1.7B\nzero-shot",   0.0,  1.0,  18.0),
        ("Qwen3-1.7B\nSFT",        17.0, 49.0,  20.0),
        ("Qwen3-4B\nzero-shot",    _f(b4["valid"]), _f(b4["grounded"]), _f(b4["answer"])),
        ("Qwen3-4B\nSFT",          44.0, 70.0,  52.0),
        ("+ symbolic\nexecutor",   69.0, 71.0,  eng),
        ("+ grounded\ncitations",  _f(N["valid_sys"]), _f(N["grounded_sys"]), eng),
    ]


# metric styling: answer accuracy (the hero metric) in green, the two structural
# metrics in a cool blue ramp.
METRICS = [
    ("Valid trace",          "#a7c4e2"),
    ("Cell grounding",       "#3f7cc0"),
    ("Answer accuracy (EM)", "#1a7f37"),
]

INK = "#1c2128"
MUTED = "#57606a"


def fmt(v: float) -> str:
    return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"


def render(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = stages()
    labels = [s[0] for s in data]
    series = np.array([[s[1], s[2], s[3]] for s in data])  # rows=stage, cols=metric

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": "#c8ccd1",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    fig, ax = plt.subplots(figsize=(11.6, 5.6), dpi=200)
    x = np.arange(len(labels))
    w = 0.26

    # shade the region where the deterministic executor is switched on
    ax.axvspan(3.5, len(labels) - 0.4, color="#1a7f37", alpha=0.055, zorder=0)
    ax.axvline(3.5, color="#c8ccd1", lw=1.0, ls=(0, (4, 3)), zorder=1)

    for j, (name, color) in enumerate(METRICS):
        offs = (j - 1) * w
        ax.bar(x + offs, series[:, j], width=w, label=name, color=color,
               edgecolor="white", linewidth=0.6, zorder=3)
        for xi, v in zip(x + offs, series[:, j]):
            ax.text(xi, v + 1.4, fmt(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK if j == 2 else MUTED,
                    fontweight="bold" if j == 2 else "normal", zorder=4)

    ax.text(1.5, 104, "model only", ha="center", va="bottom", fontsize=11,
            color=MUTED, style="italic")
    ax.text(4.5, 104, "+ deterministic executor", ha="center",
            va="bottom", fontsize=11, color="#1a7f37", style="italic", fontweight="bold")

    # annotate the executor's answer-accuracy jump, kept in open space above stage 4
    x5 = 4 + w
    ax.text(3.18, 87, "executor closes\nthe answer gap", ha="left", va="center",
            fontsize=10.5, color="#1a7f37", fontweight="bold", zorder=6)
    arr = FancyArrowPatch((4.0, 88), (x5 - 0.02, 95), arrowstyle="-|>", mutation_scale=16,
                          lw=1.8, color="#1a7f37", zorder=5,
                          connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arr)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", color="#e6e8eb", lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.set_title("Grounded table reasoning: metrics by pipeline stage",
                 fontsize=16, fontweight="bold", pad=34, loc="left", color=INK)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#d0d7de",
              fontsize=10.5, ncol=3, bbox_to_anchor=(0.0, 1.045), borderaxespad=0)

    fig.text(0.125, 0.005,
             "Answer accuracy = exact match. Model-only stages are flat across scale (1.7B->4B) "
             "and fine-tuning; the deterministic executor supplies the arithmetic. "
             "Metrics mirror the write-up's stage table.",
             fontsize=8.5, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"  -> wrote {out.relative_to(ROOT)}  ({series.shape[0]} stages x {series.shape[1]} metrics)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the pipeline-stage progression figure.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG path")
    render(ap.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
