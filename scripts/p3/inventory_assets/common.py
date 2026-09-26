"""Shared helpers for the inventory and safety generators (scripts/p3/gen_inventory.py, scripts/p3/gen_safety.py).

- deterministic rng (the same sha256-keyed scheme as gen_image_joint.rng_for)
- fonts (macOS system fonts; only rendered into our own images, never redistributed)
- ABO product cut-outs: white background -> alpha matte by a border flood fill (+ strict-white interior holes), eroded
  1 px and feathered so no white halo remains; photos that do not sit cleanly on white are rejected
- scene post-processing: mild affine (rotation / shear / scale), lighting gradient, colour cast, noise, JPEG
- image output: <= 1280 px, stored under data/p3/images/<family>/<sha256>.jpg, never an ImajevBench hash
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
CUTOUT_CACHE = ROOT / "data" / "p3" / "images" / "inventory" / ".cutouts"
MAX_SIDE = 1280


def rng_for(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ fonts
SUPP = Path("/System/Library/Fonts/Supplemental")
FONT_FILES = {
    "sans": SUPP / "Arial.ttf", "sans_bold": SUPP / "Arial Bold.ttf", "narrow": SUPP / "Arial Narrow.ttf",
    "narrow_bold": SUPP / "Arial Narrow Bold.ttf", "verdana": SUPP / "Verdana.ttf", "verdana_bold": SUPP / "Verdana Bold.ttf",
    "tahoma_bold": SUPP / "Tahoma Bold.ttf", "din": SUPP / "DIN Alternate Bold.ttf", "din_cond": SUPP / "DIN Condensed Bold.ttf",
    "mono": SUPP / "Courier New Bold.ttf", "impact": SUPP / "Impact.ttf", "trebuchet_bold": SUPP / "Trebuchet MS Bold.ttf",
}


@lru_cache(maxsize=512)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_FILES[name]), max(6, int(size)))


def text_size(draw: ImageDraw.ImageDraw, txt: str, f) -> tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=f)
    return x1 - x0, y1 - y0


def fit_font(draw, txt: str, name: str, max_w: float, max_h: float, start: int = 40):
    s = int(start)
    while s > 7:
        f = font(name, s)
        w, h = text_size(draw, txt, f)
        if w <= max_w and h <= max_h:
            return f
        s -= 1
    return font(name, 7)


def text_center(draw, cx, cy, txt, f, fill):
    x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=f)
    draw.text((cx - (x0 + x1) / 2, cy - (y0 + y1) / 2), txt, font=f, fill=fill)


# ------------------------------------------------------------------------------------------------ cut-outs
def _cutout_array(path: Path):
    """RGBA uint8 array of the product on transparent background, or (None, reason)."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((420, 420), Image.LANCZOS)
    a = np.asarray(im).astype(np.int16)
    h, w, _ = a.shape
    ring = np.zeros((h, w), dtype=bool)
    ring[:3, :] = ring[-3:, :] = True
    ring[:, :3] = ring[:, -3:] = True
    rp = a[ring]
    light = rp[(rp.min(axis=1) >= 200)]
    if len(light) < 0.6 * len(rp):
        return None, "background_not_light_plain"
    bgc = np.median(light, axis=0)
    if bgc.min() < 205 or bgc.max() - bgc.min() > 12:
        return None, "background_not_light_plain"
    dist = np.abs(a - bgc[None, None, :]).max(axis=2)
    # a tight studio crop may let the product touch an edge briefly (a shoe tip); a long contact means it is cut off
    sides = [dist[0, :], dist[-1, :], dist[:, 0], dist[:, -1]]
    if max((sd > 16).mean() for sd in sides) > 0.22 or (dist[ring] <= 16).mean() < 0.9:
        return None, "border_not_plain"
    whiteish = dist <= 13
    lab, _ = ndimage.label(whiteish)
    border_labels = np.unique(lab[ring & whiteish])
    bg = np.isin(lab, border_labels[border_labels > 0])
    # strict-white interior holes (handle loops etc.): only large ones, the colour filter keeps white products out
    strict = (dist <= 4) & ~bg
    hl, hn = ndimage.label(strict)
    if hn:
        sizes = ndimage.sum(np.ones_like(hl), hl, index=np.arange(1, hn + 1))
        for i, s in enumerate(sizes, 1):
            if s >= 0.004 * h * w:
                bg |= hl == i
    fg = ~bg
    fg = ndimage.binary_opening(fg, iterations=1)
    fl, fn = ndimage.label(fg)
    if not fn:
        return None, "empty"
    sizes = ndimage.sum(np.ones_like(fl), fl, index=np.arange(1, fn + 1))
    big = int(np.argmax(sizes)) + 1
    tot = float(sizes.sum())
    if sizes[big - 1] / tot < 0.94:
        return None, "several_parts"          # dimension arrows, badges, multiple products, text
    fg = fl == big
    frac = fg.mean()
    if not 0.10 <= frac <= 0.82:
        return None, "fg_fraction"
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # fill of the object's bbox must not be a thin line / sliver
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if min(bw, bh) < 0.15 * max(bw, bh):
        return None, "sliver"
    if bh / bw > 2.3:
        return None, "too_tall"                # model shots (a person wearing the product), tall boots
    edge = ndimage.binary_erosion(fg, iterations=2) & ~ndimage.binary_erosion(fg, iterations=4)
    if (a.min(axis=2)[edge] >= 226).mean() > 0.18:
        return None, "white_edge"              # white parts of the product merge with the background (soles, rims)
    # erode 1 px (drops the white-blended rim) and feather
    core = ndimage.binary_erosion(fg, iterations=1)
    alpha = Image.fromarray((core * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    alpha = np.asarray(alpha).astype(np.float32)
    alpha[~fg] = 0
    rgba = np.dstack([a.astype(np.uint8), alpha.clip(0, 255).astype(np.uint8)])
    rgba = rgba[y0:y1 + 1, x0:x1 + 1]
    return rgba, None


def cutout(image_rel: str, sha: str):
    """PIL RGBA cut-out of an ABO photo (cached as PNG), or None when the photo is not a clean white-background shot."""
    CUTOUT_CACHE.mkdir(parents=True, exist_ok=True)
    p = CUTOUT_CACHE / f"{sha}.png"
    bad = CUTOUT_CACHE / f"{sha}.reject"
    if p.is_file():
        return Image.open(p).convert("RGBA")
    if bad.is_file():
        return None
    arr, why = _cutout_array(ROOT / "data" / image_rel)
    if arr is None:
        bad.write_text(why)
        return None
    im = Image.fromarray(arr, "RGBA")
    im.save(p)
    return im


def reject_reason(sha: str) -> str | None:
    bad = CUTOUT_CACHE / f"{sha}.reject"
    return bad.read_text() if bad.is_file() else None


def paste_item(canvas: Image.Image, item: Image.Image, box, shadow=True, flip=False, bright=1.0, anchor="bottom"):
    """Fit `item` (RGBA) inside box=(x0,y0,x1,y1) keeping aspect, bottom-centred; returns the pasted bbox."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    iw, ih = item.size
    s = min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    it = item.resize((nw, nh), Image.LANCZOS)
    if flip:
        it = it.transpose(Image.FLIP_LEFT_RIGHT)
    if bright != 1.0:
        rgb = np.asarray(it.convert("RGB")).astype(np.float32) * bright
        it = Image.fromarray(np.dstack([rgb.clip(0, 255).astype(np.uint8), np.asarray(it)[..., 3]]), "RGBA")
    px = int(x0 + (bw - nw) / 2)
    py = int(y1 - nh) if anchor == "bottom" else int(y0 + (bh - nh) / 2)
    if shadow:
        sh = Image.new("L", canvas.size, 0)
        d = ImageDraw.Draw(sh)
        d.ellipse([px + nw * 0.08, y1 - max(3, nh * 0.045), px + nw * 0.92, y1 + max(3, nh * 0.045)], fill=110)
        sh = sh.filter(ImageFilter.GaussianBlur(max(2, nw * 0.04)))
        canvas.paste(Image.new("RGB", canvas.size, (20, 20, 20)), (0, 0), sh)
    canvas.paste(it, (px, py), it)
    return (px, py, px + nw, py + nh)


# ------------------------------------------------------------------------------------------------ post-processing
def affine_params(rng, strength=1.0) -> dict:
    return {"rot": round(rng.uniform(-2.2, 2.2) * strength, 3), "shear": round(rng.uniform(-0.035, 0.035) * strength, 4),
            "scale": round(rng.uniform(0.985, 1.03), 4), "light_dir": rng.choice(["left", "right", "top", "none"]),
            "light_amt": round(rng.uniform(0.0, 0.22), 3), "cast": [round(rng.uniform(-10, 10), 1) for _ in range(3)],
            "noise": round(rng.uniform(0.0, 5.0), 2), "blur": round(rng.choice([0, 0, 0, 0.4, 0.6]), 2),
            "jpeg": rng.choice([82, 86, 90])}


def forward_matrix(size, prm):
    """2x3 forward matrix (canvas -> output coordinates) about the image centre."""
    w, h = size
    cx, cy = w / 2, h / 2
    th = math.radians(prm["rot"])
    c, s = math.cos(th) * prm["scale"], math.sin(th) * prm["scale"]
    sh = prm["shear"]
    # M = R*S*Shear
    a, b = c, -s + c * sh
    d, e = s, c + s * sh
    return np.array([[a, b, cx - a * cx - b * cy], [d, e, cy - d * cx - e * cy]])


def map_box(M, box):
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=float).T
    q = M @ pts
    return (float(q[0].min()), float(q[1].min()), float(q[0].max()), float(q[1].max()))


def apply_post(img: Image.Image, prm: dict, rng_seed: str, fill=(40, 40, 40)) -> tuple[Image.Image, np.ndarray]:
    M = forward_matrix(img.size, prm)
    A = np.vstack([M, [0, 0, 1]])
    inv = np.linalg.inv(A)
    out = img.transform(img.size, Image.AFFINE, tuple(inv[:2].flatten()), resample=Image.BICUBIC, fillcolor=fill)
    arr = np.asarray(out).astype(np.float32)
    h, w, _ = arr.shape
    if prm["light_dir"] != "none" and prm["light_amt"] > 0:
        if prm["light_dir"] in ("left", "right"):
            g = np.linspace(1 + prm["light_amt"] / 2, 1 - prm["light_amt"] / 2, w)
            g = g if prm["light_dir"] == "left" else g[::-1]
            arr *= g[None, :, None]
        else:
            g = np.linspace(1 + prm["light_amt"] / 2, 1 - prm["light_amt"] / 2, h)
            arr *= g[:, None, None]
    arr += np.array(prm["cast"])[None, None, :]
    if prm["noise"] > 0:
        nrng = np.random.default_rng(int(hashlib.sha256(rng_seed.encode()).hexdigest()[:8], 16))
        arr += nrng.normal(0, prm["noise"], arr.shape)
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if prm["blur"]:
        out = out.filter(ImageFilter.GaussianBlur(prm["blur"]))
    return out, M


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def store_image(data: bytes, subdir: Path, bench: set[str]) -> tuple[str, str] | None:
    """Write bytes as <sha>.jpg under subdir; returns (data-relative path, sha) or None when it is an ImajevBench image."""
    sha = sha256_bytes(data)
    if sha in bench:
        return None
    subdir.mkdir(parents=True, exist_ok=True)
    p = subdir / f"{sha}.jpg"
    if not p.exists():
        p.write_bytes(data)
    try:
        return str(p.relative_to(ROOT / "data")), sha
    except ValueError:             # tests write to a temp dir
        return str(p), sha


def dumps_spec(spec) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def is_multi(item: Image.Image) -> bool:
    """True when a cut-out looks like a set (two planters, a pair of shoes, stacked baskets): after an erosion of ~3% of
    its size the mask falls into two or more large parts. Such products are never used for counting questions."""
    a = np.asarray(item)[..., 3] > 128
    k = max(2, int(round(0.03 * max(a.shape))))
    e = ndimage.binary_erosion(a, iterations=k)
    lab, n = ndimage.label(e)
    if n < 2:
        return False
    sizes = np.sort(ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)))[::-1]
    return sizes[1] >= 0.12 * sizes.sum()
