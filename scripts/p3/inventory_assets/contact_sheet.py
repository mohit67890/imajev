"""Visual-QA contact sheets for the inventory / safety candidates (reports/phase3/inventory-safety-samples/).

    .venv/bin/python scripts/p3/inventory_assets/contact_sheet.py data/p3/candidates/I-inventory.jsonl inventory --n 24
Writes <name>-overview.jpg (all samples as thumbnails) and <name>-detail-<k>.jpg (4 items per sheet, large enough to
count), plus <name>-samples.jsonl with the sampled ids.
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inventory_assets.common import ROOT, font  # noqa: E402

OUT = ROOT / "reports" / "phase3" / "inventory-safety-samples"


def caption(r) -> list[str]:
    f = r["field"]
    lines = [f"{r['id']}  [{r['family']}, d{r['difficulty']}]", f"Q: {f['question']}"]
    if f["type"] == "choice":
        lines.append("opts: " + ", ".join(f"{o['key']}={o['text']}" for o in f["options"]))
    elif f["type"] == "score":
        lines.append(f"levels 0..{len(f['levels']) - 1}")
    st = r["state"]
    for k, v in st.items():
        if k in ("photo", "plan", "view"):
            continue
        s = json.dumps(v)
        lines.append(f"{k}: {s[:260]}")
    lines.append(f"GOLD: {r['gold']!r}" + (f"  ({r['unknown_reason']})" if r["gold"] is None else ""))
    out = []
    for ln in lines:
        out += textwrap.wrap(ln, 118) or [""]
    return out


def tile(r, w):
    im = Image.open(ROOT / "data" / r["images"][0]).convert("RGB")
    im.thumbnail((w, w * 0.8))
    cap = caption(r)
    f = font("sans", 13)
    th = im.size[1] + 8 + 16 * len(cap)
    t = Image.new("RGB", (w, th), (255, 255, 255))
    t.paste(im, (0, 0))
    d = ImageDraw.Draw(t)
    y = im.size[1] + 4
    for ln in cap:
        d.text((4, y), ln, font=f, fill=(180, 0, 0) if ln.startswith("GOLD") else (20, 20, 20))
        y += 16
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--filter", default=None, help="only rows whose dataset equals this")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path) if l.strip()]
    if a.filter:
        rows = [r for r in rows if r["dataset"] == a.filter]
    rng = random.Random(a.seed)
    sample = rng.sample(rows, min(a.n, len(rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}-samples.jsonl").write_text("".join(json.dumps({"id": r["id"], "image": r["images"][0]}) + "\n"
                                                         for r in sample))
    # overview
    tw = 320
    cols = 6
    thumbs = []
    for r in sample:
        im = Image.open(ROOT / "data" / r["images"][0]).convert("RGB")
        im.thumbnail((tw - 6, tw - 6))
        thumbs.append(im)
    rows_n = (len(thumbs) + cols - 1) // cols
    ov = Image.new("RGB", (cols * tw, rows_n * tw), (240, 240, 240))
    for i, im in enumerate(thumbs):
        ov.paste(im, ((i % cols) * tw + 3, (i // cols) * tw + 3))
    ov.save(OUT / f"{a.name}-overview.jpg", quality=85)
    # details: 4 per sheet
    for k in range(0, len(sample), 4):
        ts = [tile(r, 720) for r in sample[k:k + 4]]
        h = max(t.size[1] for t in ts)
        sheet = Image.new("RGB", (1460, 2 * h + 10), (200, 200, 200))
        for i, t in enumerate(ts):
            sheet.paste(t, ((i % 2) * 740, (i // 2) * (h + 10)))
        sheet.save(OUT / f"{a.name}-detail-{k // 4 + 1}.jpg", quality=88)
    print(f"{len(sample)} samples -> {OUT}")


if __name__ == "__main__":
    main()
