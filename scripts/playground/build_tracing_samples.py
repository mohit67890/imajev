"""Draw the sample tracings the tracing page offers and the checker verifies.

For every character in static/tracing/letters.json: a good tracing (the skeleton with hand-like wobble), a
half-finished one, the wrong character, and a scribble. Rendered in the page's pen style (white canvas, dark
blue round strokes) so a sample is what a child's drawing sends. The guide is never drawn: imajev sees only
the strokes. Deterministic (seeded).

    .venv/bin/python scripts/playground/build_tracing_samples.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent / "static/tracing"
SIZE = 512
PEN = (37, 56, 140)
WIDTH = 22
WRONG = {"A": "H", "C": "O", "E": "L", "H": "A", "L": "E", "O": "C", "T": "7", "1": "7", "4": "A", "7": "1"}


def points(stroke, n=60):
    if "arc" in stroke:
        cx, cy, rx, ry, a0, a1 = stroke["arc"]
        return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                 cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
    pts = stroke["line"]
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        out += [(x0 + (x1 - x0) * i / 20, y0 + (y1 - y0) * i / 20) for i in range(20)]
    return out + [tuple(pts[-1])]


def wobble(pts, rng, amount=0.012):
    phase, drift = rng.uniform(0, 6.3), rng.uniform(-0.01, 0.01)
    return [(x + amount * math.sin(i / 5 + phase) + drift, y + amount * math.cos(i / 7 + phase) - drift) for i, (x, y) in enumerate(pts)]


def render(strokes, path):
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)
    for pts in strokes:
        xy = [(x * SIZE, y * SIZE) for x, y in pts]
        draw.line(xy, fill=PEN, width=WIDTH, joint="curve")
        for x, y in (xy[0], xy[-1]):
            draw.ellipse((x - WIDTH / 2, y - WIDTH / 2, x + WIDTH / 2, y + WIDTH / 2), fill=PEN)
    image.save(path)


def main():
    letters = {k: v for k, v in json.loads((HERE / "letters.json").read_text()).items() if not k.startswith("_")}
    out = HERE / "samples"
    out.mkdir(exist_ok=True)
    index = []
    for char, strokes in letters.items():
        rng = random.Random(f"imajev-tracing-{char}")
        full = [wobble(points(s), rng) for s in strokes]
        if len(full) > 1:
            half = full[:1]
        else:  # one-stroke characters: stop half way
            half = [full[0][: len(full[0]) // 2]]
        wrong = [wobble(points(s), rng) for s in letters[WRONG[char]]]
        scribble, x, y = [], 0.5, 0.5
        for i in range(90):
            x = min(0.85, max(0.15, x + rng.uniform(-0.09, 0.09)))
            y = min(0.85, max(0.15, y + rng.uniform(-0.09, 0.09)))
            scribble.append((x, y))
        for kind, drawing, truth in [("good", full, char), ("half", half, char), ("wrong", wrong, WRONG[char]), ("scribble", [scribble], None)]:
            name = f"{char}-{kind}.png"
            render(drawing, out / name)
            index.append({"file": f"samples/{name}", "task": char, "kind": kind, "drawn": truth})
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"{len(index)} samples")


if __name__ == "__main__":
    main()
