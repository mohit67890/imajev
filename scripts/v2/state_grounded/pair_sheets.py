"""Side-by-side PNG sheets of reference/target pairs, for eyeballing edit quality.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/pair_sheets.py \
        data/decision-v2/pairs_grounded/records.jsonl reports/v2-datasets/pairs-examples --n 6

Each sheet is one pair: reference (image 1) on the left, target (image 2) on the right, at up to
720 px per side, with the edit kind and the question's answer underneath.  `--grid` instead writes
contact sheets of many pairs (used to review every probe pair by eye).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pair_images(row):
    if "images" in row:
        return row["images"][0]["image"], row["images"][1]["image"]
    return row["reference"]["image"], row["target"]["image"]


def caption(row):
    kind = row.get("edit_kind") or (row.get("meta") or {}).get("kind")
    q = row.get("request", {}).get("fields", [{}])[0].get("question", "")
    gold = row.get("target", row.get("gold"))
    if isinstance(gold, dict):
        gold = ""
    return f"edit: {kind}   answer: {gold}", q[:150]


def sheet(row, side=720):
    a, b = (Image.open(ROOT / p).convert("RGB") for p in pair_images(row))
    for im in (a, b):
        im.thumbnail((side, side))
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width + 30, h + 70), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 30, 0))
    d = ImageDraw.Draw(out)
    top, q = caption(row)
    d.text((6, h + 8), "image 1 (reference)", fill="black")
    d.text((a.width + 36, h + 8), "image 2 (target)", fill="black")
    d.text((6, h + 28), top, fill="black")
    d.text((6, h + 46), q, fill=(80, 80, 80))
    return out


def grid(rows, path, cols=2, side=380):
    cells = []
    for r in rows:
        a, b = (Image.open(ROOT / p).convert("RGB") for p in pair_images(r))
        for im in (a, b):
            im.thumbnail((side, side))
        cells.append((a, b, caption(r)[0], r.get("key") or r.get("id")))
    w = cols * (2 * side + 40)
    rows_n = (len(cells) + cols - 1) // cols
    out = Image.new("RGB", (w, rows_n * (side + 40)), "white")
    d = ImageDraw.Draw(out)
    for i, (a, b, text, key) in enumerate(cells):
        x, y = (i % cols) * (2 * side + 40), (i // cols) * (side + 40)
        out.paste(a, (x, y))
        out.paste(b, (x + side + 10, y))
        d.text((x + 4, y + side + 4), f"#{i} {text}", fill="black")
        d.text((x + 4, y + side + 20), str(key)[:60], fill=(90, 90, 90))
    out.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--kinds", default="")
    a = ap.parse_args()
    rows = load(a.records)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.grid:
        for k in range(0, len(rows), 8):
            grid(rows[k:k + 8], a.out_dir / f"grid-{k // 8:02d}.png")
        print(f"{(len(rows) + 7) // 8} grids -> {a.out_dir}")
        sys.exit(0)
    rng = random.Random(a.seed)
    wanted = [k for k in a.kinds.split(",") if k]
    picked = []
    for kind in wanted or [None] * a.n:
        pool = [r for r in rows if kind is None or r.get("edit_kind") == kind]
        pool = [r for r in pool if r not in picked]
        if pool:
            picked.append(rng.choice(pool))
    for i, r in enumerate(picked[:a.n]):
        sheet(r).save(a.out_dir / f"pair-{i + 1:02d}-{r.get('edit_kind')}.png")
    print(f"{len(picked[:a.n])} sheets -> {a.out_dir}")
