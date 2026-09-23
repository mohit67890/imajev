"""Controlled, plausible edits for the reference/target image pairs.

Pure image functions: each takes a PIL image (and, for the paste, a second one) plus a seeded
`random.Random`, and returns `(edited image, meta)`.  Nothing here reads or writes a file, and no
model is involved -- which is the point: the label of every pair is known because we made the edit.

Two rules run through all of it:

1. **The edit must not announce itself.**  Every paste, fill and recolour goes through a feathered
   mask, and the fill patch is taken from the same photograph, so lighting, grain and colour
   temperature already agree.  A hard rectangle would let a model answer "which one is the target"
   without looking at the content, and then the family would measure nothing.
2. **The edit must be one thing.**  A pair differs in exactly one respect, so "what changed" has
   one right answer.  The irrelevant edits (brightness, contrast, a small crop, a mirror, a tiny
   object in a corner) are the deliberate exception: they change the pixels everywhere but change
   nothing a person checking the scene would call a change.

`variance_map` is the small piece of machinery the rest rests on: a coarse local-detail map, used
to find a patch that probably holds an object (high detail) and a patch that is probably plain
background (low detail).  It is a heuristic, not a segmenter, and the pair README says so.
"""
from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageStat

RELEVANT_KINDS = ("add_object", "remove_object", "recolour_region", "hide_label")
PASTE_KINDS = ("add_object", "remove_object", "corner_object")   # the edits that need a donor
IRRELEVANT_KINDS = ("brightness", "contrast", "small_crop", "mirror", "corner_object")
EDIT_KINDS = RELEVANT_KINDS + IRRELEVANT_KINDS

# What each edit falsifies in the reference record used by `which_state_field_now_wrong`.
FALSIFIES = {
    "add_object": "no_new_items",
    "remove_object": "all_items_still_there",
    "recolour_region": "colours_as_before",
    "hide_label": "nothing_covered_or_blurred",
}

# Named colours with a representative RGB, for saying in words what colour a region is.
NAMED_RGB = {
    "red": (190, 40, 40), "orange": (225, 130, 35), "yellow": (225, 205, 60),
    "green": (70, 140, 70), "blue": (55, 95, 180), "purple": (120, 70, 160),
    "pink": (225, 150, 175), "brown": (125, 85, 55), "black": (30, 30, 30),
    "white": (240, 240, 240), "grey": (135, 135, 135), "beige": (215, 200, 170),
    "turquoise": (60, 170, 170), "navy": (30, 45, 95), "cream": (240, 230, 200),
    "burgundy": (110, 30, 45),
}
GRID = 32           # the variance map is GRID x GRID cells


def colour_name(rgb) -> str:
    """The nearest of `NAMED_RGB`, weighted the way human colour naming leans."""
    r, g, b = [float(x) for x in rgb[:3]]
    best, best_d = "grey", None
    for name, (nr, ng, nb) in NAMED_RGB.items():
        d = 2.0 * (r - nr) ** 2 + 3.0 * (g - ng) ** 2 + 1.0 * (b - nb) ** 2
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


def region_colour(img: Image.Image, box) -> str:
    return colour_name(ImageStat.Stat(img.crop(box)).mean)


def variance_map(img: Image.Image):
    """-> a GRID x GRID list of local detail scores (higher = more going on there)."""
    small = img.convert("L").resize((GRID * 2, GRID * 2), Image.BILINEAR)
    blurred = small.filter(ImageFilter.GaussianBlur(2))
    detail = ImageChops.difference(small, blurred).resize((GRID, GRID), Image.BOX)
    pixels = list(detail.getdata())
    return [pixels[y * GRID:(y + 1) * GRID] for y in range(GRID)]


def _cells_sorted(vmap, reverse):
    cells = [(vmap[y][x], x, y) for y in range(GRID) for x in range(GRID)]
    cells.sort(reverse=reverse)
    return cells


def pick_region(img, rng, *, busy: bool, frac=(0.18, 0.34), aspect=1.0, central=False, top=40):
    """A box (l, t, r, b) over a busy part of the picture (`busy=True`) or a plain one.

    `top` keeps the choice among the most/least detailed cells rather than always the single most
    extreme one, so two photographs of the same kind do not get the same edit in the same place.
    """
    w, h = img.size
    side = rng.uniform(*frac) * min(w, h)
    bw, bh = int(side * max(1.0, aspect)), int(side / max(1.0, 1.0 / aspect) if aspect < 1 else side)
    bw, bh = max(24, min(bw, w - 8)), max(24, min(bh, h - 8))
    vmap = variance_map(img)
    cells = _cells_sorted(vmap, reverse=busy)
    if central:
        cells = [c for c in cells if GRID // 5 <= c[1] < GRID - GRID // 5
                 and GRID // 5 <= c[2] < GRID - GRID // 5] or cells
    pick = cells[rng.randrange(min(top, len(cells)))]
    cx = int((pick[1] + 0.5) * w / GRID)
    cy = int((pick[2] + 0.5) * h / GRID)
    left = max(0, min(w - bw, cx - bw // 2))
    top_ = max(0, min(h - bh, cy - bh // 2))
    return (left, top_, left + bw, top_ + bh)


def feather(size, rng, shape="ellipse", radius=None):
    """A soft-edged mask: the edit fades into the photograph instead of cutting into it."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    inset = max(2, int(min(w, h) * 0.10))
    if shape == "ellipse":
        draw.ellipse((inset, inset, w - inset, h - inset), fill=255)
    else:
        r = max(3, int(min(w, h) * 0.18))
        draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=r, fill=255)
    blur = radius if radius is not None else max(2.0, min(w, h) * 0.09)
    return mask.filter(ImageFilter.GaussianBlur(blur))


def _match_luminance(patch: Image.Image, target_mean):
    """Scale a pasted patch so it sits in the same light as the place it is going."""
    mean = ImageStat.Stat(patch.convert("L")).mean[0] or 1.0
    factor = max(0.55, min(1.8, (target_mean or mean) / mean))
    return ImageEnhance.Brightness(patch).enhance(factor)


# --------------------------------------------------------------------------- helpers for edits
def is_colour_photo(img, threshold=28.0) -> bool:
    """False for black-and-white / sepia photographs: a hue rotation or a colour paste on those is
    an artefact, not an edit.  PD12M holds a great many historical monochrome prints."""
    sat = ImageStat.Stat(img.convert("HSV").split()[1].resize((64, 64))).mean[0]
    return sat >= threshold


def cutout(donor, max_side=900):
    """-> an RGBA cut-out of the donor's subject, or None when the donor is not suitable.

    Only donors photographed against a near-uniform backdrop qualify (museum specimens, product
    shots): the backdrop colour is read from the border, and the mask is every pixel clearly away
    from it, cleaned up and feathered.  The object is then cropped to its own bounding box, so what
    is pasted is the object's outline -- not a square or an ellipse of somebody else's photograph.
    """
    d = donor.convert("RGB")
    d.thumbnail((max_side, max_side))
    w, h = d.size
    edge = max(4, int(min(w, h) * 0.04))
    strips = [d.crop((0, 0, w, edge)), d.crop((0, h - edge, w, h)),
              d.crop((0, 0, edge, h)), d.crop((w - edge, 0, w, h))]
    means, stds = [], []
    for strip in strips:
        st = ImageStat.Stat(strip)
        means.append(st.mean[:3])
        stds.append(sum(st.stddev[:3]) / 3)
    if max(stds) > 14:
        return None                     # a busy border: no clean backdrop to cut against
    bg = tuple(sum(m[i] for m in means) / 4 for i in range(3))
    if max(abs(means[k][i] - bg[i]) for k in range(4) for i in range(3)) > 18:
        return None                     # the four edges disagree: not one backdrop
    diff = ImageChops.difference(d, Image.new("RGB", d.size, tuple(int(c) for c in bg)))
    dist = diff.convert("L").point(lambda v: 255 if v > 38 else 0)
    dist = dist.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(3))
    bbox = dist.getbbox()
    if not bbox:
        return None
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    area = sum(1 for v in dist.crop(bbox).getdata() if v) / max(1, bw * bh)
    if bw * bh < 0.04 * w * h or bw * bh > 0.80 * w * h or area < 0.25 or area > 0.90:
        return None                     # too small, fills the frame, or mostly holes
    mask = dist.crop(bbox).filter(ImageFilter.GaussianBlur(1.6))
    obj = d.crop(bbox).convert("RGBA")
    obj.putalpha(mask)
    return obj


def _ring_is_plain(vmap, cx, cy, r, limit):
    ring = [vmap[y][x] for y in range(max(0, cy - r), min(GRID, cy + r + 1))
            for x in range(max(0, cx - r), min(GRID, cx + r + 1))
            if max(abs(x - cx), abs(y - cy)) == r]
    return ring and sum(ring) / len(ring) <= limit


# --------------------------------------------------------------------------- the four real edits
def add_object(img, donor, rng):
    """Paste a cut-out of another CC0 photograph's subject into a plain part of this one."""
    obj = donor if (donor is not None and donor.mode == "RGBA") else (
        cutout(donor) if donor is not None else None)
    if obj is None:
        return None
    out = img.copy()
    w, h = out.size
    target = int(min(w, h) * rng.uniform(0.16, 0.28))
    ratio = target / max(obj.size)
    obj = obj.resize((max(8, int(obj.width * ratio)), max(8, int(obj.height * ratio))),
                     Image.LANCZOS)
    # rest it on a plain area in the lower two-thirds, clear of the frame edge: things stand on
    # the ground, and an object jammed against the border looks pasted
    vmap = variance_map(out)
    cells = [(vmap[cy][cx], cx, cy) for cy in range(GRID // 3, GRID - 3)
             for cx in range(4, GRID - 4)]
    cells.sort()
    _, cx, cy = cells[rng.randrange(min(60, len(cells)))]
    x = int((cx + 0.5) * w / GRID) - obj.width // 2
    y = int((cy + 0.5) * h / GRID) - obj.height // 2
    x = max(int(w * 0.03), min(w - obj.width - int(w * 0.03), x))
    y = max(int(h * 0.03), min(h - obj.height - int(h * 0.03), y))
    here = ImageStat.Stat(out.crop((x, y, x + obj.width, y + obj.height)).convert("L")).mean[0]
    rgb = _match_luminance(obj.convert("RGB"), 0.5 * here + 0.5 * ImageStat.Stat(
        obj.convert("L"), obj.split()[3]).mean[0])
    # a soft contact shadow so the object sits on the scene instead of floating over it
    shadow = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    shadow.putalpha(obj.split()[3].point(lambda v: int(v * 0.35)).filter(
        ImageFilter.GaussianBlur(max(2, obj.width // 25))))
    out = out.convert("RGBA")
    off = max(2, obj.width // 40)
    out.alpha_composite(shadow, (min(w - obj.width, x + off), min(h - obj.height, y + off)))
    rgb.putalpha(obj.split()[3])
    out.alpha_composite(rgb, (x, y))
    return out.convert("RGB"), {"kind": "add_object",
                                "box": [x, y, x + obj.width, y + obj.height]}


def remove_object(img, donor, rng):
    """'Something was removed', made without any inpainting at all.

    Filling a hole convincingly needs a segmenter and an inpainting model; a patch copied from
    next door leaves ghosts (an early version of this function turned a grazing horse's head into
    a smear of legs).  So the pair is built the other way round: an object is pasted into the photo
    exactly as `add_object` does, and the EDITED image becomes the reference while the untouched
    original becomes the target.  The target then has no artefact whatsoever, and the answer --
    something in image 1 is missing from image 2 -- is still known by construction.
    -> (image to use as the REFERENCE, meta with `swap: True`).
    """
    res = add_object(img, donor, rng)
    if res is None:
        return None
    edited, meta = res
    return edited, {**meta, "kind": "remove_object", "swap": True}


def _saturated_box(img, rng, frac=(0.14, 0.26)):
    """A central box over the most strongly coloured part of the picture."""
    w, h = img.size
    sat = img.convert("HSV").split()[1].resize((GRID, GRID), Image.BOX)
    vmap = variance_map(img)
    # coloured AND in focus: a saturated but blurred background is not "an object"
    cells = [(sat.getpixel((x, y)) * (1 + vmap[y][x]), x, y)
             for y in range(GRID // 6, GRID - GRID // 6)
             for x in range(GRID // 6, GRID - GRID // 6)]
    cells.sort(reverse=True)
    pick = cells[rng.randrange(min(12, len(cells)))]
    side = int(rng.uniform(*frac) * min(w, h))
    cx, cy = int((pick[1] + 0.5) * w / GRID), int((pick[2] + 0.5) * h / GRID)
    left, top = max(0, min(w - side, cx - side // 2)), max(0, min(h - side, cy - side // 2))
    return (left, top, left + side, top + side)


def recolour_region(img, rng, min_shift=0.28):
    """Change the colour of ONE whole object, leaving shading, texture and everything else alone.

    The object is found by growing a region of similar hue out from a coloured, in-focus seed
    point (a flood fill over a hue/saturation mask), and it must sit entirely inside a search box:
    an object that runs off the box is part of something bigger, and recolouring a piece of it
    reads as a stain rather than as "the object is a different colour".
    -> None when no such object is found, or when its colour does not change *name*.
    """
    w, h = img.size
    seed_box = _saturated_box(img, rng, frac=(0.02, 0.03))
    sx, sy = (seed_box[0] + seed_box[2]) // 2, (seed_box[1] + seed_box[3]) // 2
    side = int(min(w, h) * rng.uniform(0.40, 0.55))
    box = (max(0, min(w - side, sx - side // 2)), max(0, min(h - side, sy - side // 2)))
    box = (box[0], box[1], box[0] + side, box[1] + side)
    patch = img.crop(box).convert("RGB")
    hue, sat, val = patch.convert("HSV").split()
    px, py = sx - box[0], sy - box[1]
    seed_h = sorted(hue.crop((px - 3, py - 3, px + 4, py + 4)).getdata())[24]
    near = hue.point(lambda v, c=seed_h: 255 if min(abs(v - c), 256 - abs(v - c)) <= 20 else 0)
    coloured = sat.point(lambda v: 255 if v > 60 else 0)
    mask = ImageChops.multiply(near, coloured).filter(ImageFilter.MedianFilter(5))
    if mask.getpixel((px, py)) != 255:
        return None
    ImageDraw.floodfill(mask, (px, py), 128)
    component = mask.point(lambda v: 255 if v == 128 else 0)
    bbox = component.getbbox()
    if not bbox:
        return None
    frac = sum(1 for v in component.getdata() if v) / (side * side)
    margin = 3
    touches = sum([bbox[0] <= margin, bbox[1] <= margin, bbox[2] >= side - margin,
                   bbox[3] >= side - margin])
    if frac < 0.012 or frac > 0.6 or touches >= 1:
        return None
    before = colour_name(ImageStat.Stat(patch, component).mean)
    shift = rng.uniform(min_shift, 1.0 - min_shift)
    offset = int(shift * 255) % 256
    turned = Image.merge("HSV", (hue.point(lambda v, o=offset: (v + o) % 256), sat, val)).convert("RGB")
    soft = component.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.2))
    out = img.copy()
    out.paste(turned, box[:2], soft)
    after = colour_name(ImageStat.Stat(out.crop(box), component).mean)
    if after == before:
        return None
    obj_box = [box[0] + bbox[0], box[1] + bbox[1], box[0] + bbox[2], box[1] + bbox[3]]
    return out, {"kind": "recolour_region", "box": obj_box, "colour_before": before,
                 "colour_after": after, "hue_shift": round(shift, 3),
                 "object_fraction_of_search_box": round(frac, 3)}


def hide_label(img, rng):
    """Blur out a detailed label-shaped patch, or cover it with a strip of plain tape/paper."""
    out = img.copy()
    box = pick_region(out, rng, busy=True, frac=(0.12, 0.20), aspect=2.4, central=True)
    bw, bh = box[2] - box[0], box[3] - box[1]
    if rng.random() < 0.80:          # mostly a blur; a taped-over strip is the rarer kind
        patch = out.crop(box).filter(ImageFilter.GaussianBlur(max(4.0, bh * 0.22)))
        out.paste(patch, box[:2], feather((bw, bh), rng, "rounded", radius=max(2.5, bh * 0.10)))
        how = "blurred"
    else:
        # a strip of masking tape / a paper sticker: near-white or pale grey, lit like the scene,
        # with a faint shadow and a touch of grain so it reads as a physical thing on the item
        here = ImageStat.Stat(out.crop(box).convert("L")).mean[0]
        tone = int(max(150, min(245, here * 0.4 + rng.uniform(150, 190))))
        tint = rng.choice([(0, 0, 0), (6, 4, -4), (-3, 0, 4)])
        patch = Image.new("RGB", (bw, bh), tuple(max(0, min(255, tone + t)) for t in tint))
        noise = Image.effect_noise((bw, bh), 6).convert("RGB")
        patch = Image.blend(patch, noise, 0.06)
        shadow = Image.new("RGBA", (bw, bh), (0, 0, 0, 70))
        m = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(m).rounded_rectangle((1, 1, bw - 2, bh - 2), radius=max(2, bh // 10), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(1.0))
        base = out.convert("RGBA")
        shadow.putalpha(m.point(lambda v: v * 70 // 255).filter(ImageFilter.GaussianBlur(3)))
        base.alpha_composite(shadow, (min(base.width - bw, box[0] + 3), min(base.height - bh, box[1] + 3)))
        tape = patch.convert("RGBA")
        tape.putalpha(m)
        base.alpha_composite(tape, box[:2])
        out = base.convert("RGB")
        how = "covered"
    return out, {"kind": "hide_label", "box": list(box), "how": how}


# --------------------------------------------------------------------------- irrelevant edits
def irrelevant(img, rng, kind=None, donor=None):
    """A change that alters every pixel and yet changes nothing about the scene."""
    kind = kind or rng.choice([k for k in IRRELEVANT_KINDS if k != "corner_object" or donor])
    w, h = img.size
    if kind == "brightness":
        factor = rng.choice([rng.uniform(0.80, 0.90), rng.uniform(1.10, 1.22)])
        return ImageEnhance.Brightness(img).enhance(factor), \
            {"kind": kind, "factor": round(factor, 3)}
    if kind == "contrast":
        factor = rng.choice([rng.uniform(0.80, 0.90), rng.uniform(1.12, 1.28)])
        return ImageEnhance.Contrast(img).enhance(factor), \
            {"kind": kind, "factor": round(factor, 3)}
    if kind == "small_crop":
        pad = rng.uniform(0.02, 0.06)
        box = (int(w * pad), int(h * pad), int(w * (1 - pad)), int(h * (1 - pad)))
        return img.crop(box).resize((w, h), Image.LANCZOS), \
            {"kind": kind, "crop_fraction": round(pad, 3)}
    if kind == "mirror":
        return img.transpose(Image.FLIP_LEFT_RIGHT), {"kind": kind}
    # corner_object: something small and unrelated, tucked into a corner
    obj = donor if (donor is not None and donor.mode == "RGBA") else (
        cutout(donor) if donor is not None else None)
    if obj is None:
        return None
    out = img.convert("RGBA")
    side = int(min(w, h) * rng.uniform(0.05, 0.075))
    ratio = side / max(obj.size)
    obj = obj.resize((max(6, int(obj.width * ratio)), max(6, int(obj.height * ratio))),
                     Image.LANCZOS)
    margin = int(min(w, h) * 0.02)
    corner = rng.randrange(4)
    x = margin if corner in (0, 2) else w - obj.width - margin
    y = margin if corner in (0, 1) else h - obj.height - margin
    out.alpha_composite(obj, (x, y))
    return out.convert("RGB"), {"kind": kind, "box": [x, y, x + obj.width, y + obj.height],
                                "corner": corner}


def change_is_visible(before, after, box, min_mean=16.0, min_area=0.006):
    """A relevant edit nobody can see is not a fair question: require the edited box to be a
    reasonable size and to differ, on average, by a clearly visible amount."""
    w, h = before.size
    l, t, r, b = box
    if (r - l) * (b - t) < min_area * w * h:
        return False
    # per-channel, not luminance: a hue rotation keeps brightness but moves the channels a lot
    diff = ImageChops.difference(before.crop(box).convert("RGB"), after.crop(box).convert("RGB"))
    return max(ImageStat.Stat(diff).mean[:3]) >= min_mean


def apply_edit(img, kind, rng, donor=None):
    """-> (edited image, meta) or None when the edit could not be made honestly (including when
    a relevant edit would be too small or too faint to see)."""
    res = _apply(img, kind, rng, donor)
    if res is None or kind not in RELEVANT_KINDS:
        return res
    edited, meta = res
    if not change_is_visible(img, edited, tuple(meta["box"])):
        return None
    return res


def _apply(img, kind, rng, donor=None):
    """-> (edited image, meta) or None when the edit could not be made honestly."""
    if kind == "add_object":
        return add_object(img, donor, rng) if donor is not None else None
    if kind == "remove_object":
        return remove_object(img, donor, rng) if donor is not None else None
    if kind == "recolour_region":
        return recolour_region(img, rng) if is_colour_photo(img) else None
    if kind == "hide_label":
        return hide_label(img, rng)
    return irrelevant(img, rng, kind=kind, donor=donor)
