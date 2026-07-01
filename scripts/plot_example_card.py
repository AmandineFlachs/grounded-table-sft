"""Render one worked demo card to docs/assets/example.png for the README.

A single in-distribution success, end to end: question + table (with the cells the
engine actually read highlighted) -> the model's structured operation -> the engine's
answer -> grounded evidence -> the safety-gate decision -> the system answer vs gold.
This is the same card the demo produces (python scripts/demo.py); it is rendered to a
committed PNG so the README can show a concrete example instead of only describing one.

Needs headless Chrome/Edge + Pillow (optional dev deps, deliberately NOT in the minimal
requirements.txt); the PNG is committed, so a fresh clone shows it without rendering.

    python scripts/plot_example_card.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo import DEMO_CARD_CSS, build_card, card_html   # noqa: E402
from src.schema import Example                              # noqa: E402
from src.splits import load_jsonl                            # noqa: E402

OUT = ROOT / "docs" / "assets" / "example.png"
# the featured in-distribution success (same card the write-up embeds)
EXAMPLE_ID = "tatqa_0036_cols_best_under_constraint_normal"
RES = ROOT / "results" / "p3_4b_exec_TEST.json"
SRC = ROOT / "data" / "processed" / "eval_test.v0_1_0.jsonl"

_CHROMES = [
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]


def _chrome() -> str:
    for c in _CHROMES:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("msedge")
    if not found:
        raise SystemExit("No Chrome/Edge found for headless rendering.")
    return found


def _card_page() -> str:
    det = {d["example_id"]: d for d in json.loads(RES.read_text(encoding="utf-8"))["details"]}
    for r in load_jsonl(SRC):
        ex = Example.model_validate(r)
        eid = ex.metadata.get("example_id", ex.table_id)
        if eid == EXAMPLE_ID:
            card = build_card(ex, det[EXAMPLE_ID]["raw"])
            card["ood"] = False
            return ("<!doctype html><meta charset=utf-8><style>"
                    ":root{--line:#dfe4ea;--muted:#6b7686}"
                    "body{margin:0;padding:24px;background:#fff;"
                    "font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;width:720px}"
                    f"{DEMO_CARD_CSS}</style>{card_html(card)}")
    raise SystemExit(f"example not found: {EXAMPLE_ID}")


def _trim(im: Image.Image, pad: int = 24) -> Image.Image:
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im.convert("RGB"), bg).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    return im


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = _card_page()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "card.html"
        page.write_text(html, encoding="utf-8")
        raw = Path(tmp) / "raw.png"
        subprocess.run([_chrome(), "--headless", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--virtual-time-budget=2000",
                        "--window-size=780,1400", f"--screenshot={raw}", page.as_uri()],
                       capture_output=True, timeout=120)
        _trim(Image.open(raw)).save(OUT)
    print(f"  -> wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
