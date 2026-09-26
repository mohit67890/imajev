"""Contact sheets for visual QA of generated image items (gen_geometry.py / gen_screens.py).

    .venv/bin/python scripts/p3/screen_templates/contact_sheet.py --in rows.jsonl --out-dir reports/phase3/geometry-screens-samples \
        --name geometry --n 24 --per-sheet 6 [--seed 0]

Each tile: the image (scaled to the tile width) with the row id, family/task, state, question, options (gold marked
with *) printed underneath. Sheets are PNG, n/per-sheet of them.
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(row, W=640, text_h=250):
    im = Image.open(ROOT / "data" / row["images"][0]).convert("RGB")
    s = min(W / im.width, 560 / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    t = Image.new("RGB", (W, im.height + text_h), "white")
    t.paste(im, ((W - im.width) // 2, 0))
    d = ImageDraw.Draw(t)
    f, fb = font(15), font(16)
    y = im.height + 6
    pv = row["provenance"]
    task = pv.get("task") or pv.get("template")
    d.text((8, y), f"{row['id']}  [{row['family']} / {task}]  d{row['difficulty']}", fill="#555", font=f)
    y += 20
    st = json.dumps(row["state"], ensure_ascii=False)
    for line in textwrap.wrap("state: " + st, 78)[:3]:
        d.text((8, y), line, fill="#335", font=f)
        y += 18
    for line in textwrap.wrap("Q: " + row["field"]["question"], 74)[:3]:
        d.text((8, y), line, fill="#000", font=fb)
        y += 20
    fld = row["field"]
    if fld["type"] == "choice":
        opts = [("* " if o["key"] == row["gold"] else "  ") + o["text"] for o in fld["options"]]
        txt = " | ".join(opts)
    elif fld["type"] == "score":
        txt = f"score gold level {row['gold']} = " + (fld["levels"][row["gold"]]["description"] if row["gold"] is not None else "-")
    else:
        txt = f"noul gold = {row['gold']}"
    if row["gold"] is None:
        txt += f"   GOLD: unknown ({row['unknown_reason']})"
    for line in textwrap.wrap(txt, 80)[:4]:
        d.text((8, y), line, fill="#a00" if row["gold"] is None else "#060", font=f)
        y += 18
    d.rectangle([0, 0, W - 1, t.height - 1], outline="#bbb")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter", default=None, help="substring the family or task must contain")
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.inp for l in open(p) if l.strip()]
    if a.filter:
        rows = [r for r in rows if any(a.filter in str(x) for x in (r["family"], r["provenance"].get("task"), r["provenance"].get("template"), r["provenance"].get("screen_template")))]
    sample = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = 2
    for si in range(0, len(sample), a.per_sheet):
        tiles = [tile(r) for r in sample[si:si + a.per_sheet]]
        rows_n = (len(tiles) + cols - 1) // cols
        hs = [max(t.height for t in tiles[i * cols:(i + 1) * cols]) for i in range(rows_n)]
        sheet = Image.new("RGB", (cols * 650, sum(hs) + 10 * rows_n), "#eee")
        y = 0
        for i in range(rows_n):
            for j, t in enumerate(tiles[i * cols:(i + 1) * cols]):
                sheet.paste(t, (j * 650 + 5, y + 5))
            y += hs[i] + 10
        p = out / f"{a.name}-{si // a.per_sheet + 1:02d}.png"
        sheet.save(p)
        print(p)
    (out / f"{a.name}-ids.json").write_text(json.dumps([r["id"] for r in sample], indent=0) + "\n")


if __name__ == "__main__":
    main()
