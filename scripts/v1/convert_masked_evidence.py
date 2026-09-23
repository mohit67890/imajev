"""masked_evidence -> decision-v1 records: evidence that is *hidden*, in an otherwise normal scene.

Why this source exists
----------------------
A model trained on decision-v1 learned to answer "unknown" only for the kinds of unanswerability it
had seen (absent objects, unreadable phone photos, mismatched reference images).  On held-out
TUBench-style yes/no questions that are unanswerable because the evidence is *occluded or outside
the frame* it abstained on 1 of 225.  This source supplies that missing case.

What it builds
--------------
Every record is a well-posed question about ONE annotated Visual Genome object, over a photograph
that has been given exactly one visual treatment (an opaque/destructive patch, or a crop):

* HIDDEN items (~45%): the treatment falls on the questioned object - target null,
  `abstention_cause` = "insufficient_evidence".
* CONTROL items (~55%): the *same* treatments, the same size distribution, but applied to a region
  that does not touch the questioned object's padded box - target = the annotated answer.

The point of the controls is that a grey rectangle must not by itself mean "unknown".  Every patch
style and the crop treatment appears in hidden and control items at the same rate, with the same
field-type mix, the same statement/question mix and the same source mix, so the only way to tell
the two apart is to look at *where* the treatment landed relative to the thing being asked about.

Referents
---------
Two question pools, both keyed to a Visual Genome object box:
1. `vg_attributes`-style attribute questions (colour / material / shape / state) built from
   `scripts/v1/vg_attribute_lexicon.py`;
2. GQA `train_balanced`/`val_balanced` questions whose `annotations` reference exactly one object id
   (that id is a Visual Genome object id), restricted to question types that ask about that one
   object's attribute, material, activity or category.

In both pools the object's primary name must occur exactly once in the image's object list and must
not be a word or substring of any other object name there, so the question has a single referent and
hiding that object really removes the evidence.  Box area must be 2-40% of the image.

Deterministic from SEED; re-runnable (images are content-addressed, orphans are pruned).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import CACHE_ROOT, DATA_ROOT, VG_PROC, load_exclusions, stable_rank, vg_image_data
from vg_attribute_lexicon import build_index
from vision_decision.contracts import Request

SOURCE = "masked_evidence"
SEED = 20260922
LICENSE = "CC-BY-4.0"

FIT_TARGET = 32_000
DEV_TARGET = 960                      # 3% of train
TEST_TARGET = 900
HIDDEN_SHARE = 0.45
BOOLEAN_SHARE = 0.55                  # no honest ordinal exists here - see README
STATEMENT_SHARE = 0.34                # share of *boolean* items phrased as a statement
BIG_OPTIONS_RATE = 0.10               # share of choice items with 13-25 options

MIN_AREA, MAX_AREA = 0.02, 0.40       # questioned object's box, as a fraction of the image
MIN_SIDE_PX = 24
PAD_LO, PAD_HI = 0.10, 0.20           # padding added to each side of the box before covering it
CROP_MIN_AREA = 0.40                  # a crop must keep >= 40% of the area ...
CROP_MIN_SHORT = 200                  # ... and >= 200 px on the short side

PATCH_STYLES = ["flat_grey", "flat_black", "flat_white", "mean_colour",
                "gaussian_blur", "pixelate", "noise"]
TREATMENTS = PATCH_STYLES + ["crop"]
# crop is rarer than the patches because hidden crops need the object near an edge
TREATMENT_SHARE = {t: 0.88 / len(PATCH_STYLES) for t in PATCH_STYLES} | {"crop": 0.12}
# blur and pixelation leave the region's average colour behind, so they are never used for a
# question whose answer is a colour - in hidden *or* control items, so the rates stay matched.
COLOUR_BLIND_TREATMENTS = {"gaussian_blur", "pixelate"}

FAMILY_SHARE = {"color": 0.34, "state": 0.22, "material": 0.17, "shape": 0.07, "gqa": 0.20}

OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
CACHE = CACHE_ROOT / SOURCE
ATTR_ZIP = CACHE_ROOT / "vg_attributes/attributes.json.zip"
GQA_INSTR = CACHE_ROOT / "gqa/hf"

INDEX, GROUPS = build_index()

# ----------------------------------------------------------------- wording

TEMPLATES = {
    "color": [
        "What color {be} the {obj}?",
        "What is the color of the {obj}?",
        "The {obj} {be} what color?",
        "Which color best describes the {obj} in this photo?",
        "In this image, what color {be} the {obj}?",
        "Which of these is the colour of the {obj}?",
        "Pick the colour of the {obj}.",
        "What colour would you call the {obj}?",
    ],
    "material": [
        "What {be} the {obj} made of?",
        "What material {be} the {obj}?",
        "Which material {be} the {obj} made from?",
        "The {obj} {appears} to be made of what?",
        "What {does} the {obj} seem to be made out of?",
        "Judging by the photo, what {be} the {obj} made from?",
        "Which of these materials is used for the {obj}?",
        "Identify the material of the {obj}.",
    ],
    "shape": [
        "What shape {be} the {obj}?",
        "What is the shape of the {obj}?",
        "Which shape best describes the {obj}?",
        "The {obj} {has} what shape?",
        "How would you describe the shape of the {obj}?",
        "In this picture, what shape {be} the {obj}?",
        "Which of these shapes matches the {obj}?",
        "Pick the shape of the {obj}.",
    ],
    "state": [
        "Which of these best describes the {obj} in this image?",
        "What is the state of the {obj}?",
        "In this photo, the {obj} {be} which of the following?",
        "How would you describe the {obj} here?",
        "Which option describes the {obj} shown in the picture?",
        "The {obj} in this image {be} what?",
        "Which of these is true of the {obj}?",
        "Pick the option that fits the {obj}.",
    ],
}

BOOL_QUESTIONS = [
    "{Be} the {obj} {val}?",
    "{Be} {that} {obj} {val}?",
    "{Does} the {obj} look {val}?",
    "Would you say the {obj} {be} {val}?",
    "In this photo, {be} the {obj} {val}?",
    "Looking at this picture, {be} the {obj} {val}?",
    "{Be} the {obj} in this image {val}?",
    "Can you confirm the {obj} {be} {val}?",
]
BOOL_STATEMENTS = [
    "The {obj} {be} {val}.",
    "{That} {obj} {be} {val}.",
    "The {obj} in this photo {be} {val}.",
    "{This} {obj} {looks} {val}.",
    "In this image, the {obj} {be} {val}.",
    "The {obj} here {be} {val}.",
    "The {obj} shown {be} {val}.",
]
MADE_OF_QUESTIONS = ["{Be} the {obj} made of {val}?", "{Be} {that} {obj} made of {val}?"]
MADE_OF_STATEMENTS = ["The {obj} {be} made of {val}.", "{That} {obj} {be} made of {val}."]

SINGULAR_S = {"grass", "glass", "bus", "dress", "cross", "moss", "tennis", "class", "gas",
              "compass", "mattress", "asparagus", "cactus", "iris", "lens", "chess", "harness",
              "bass", "brass", "boss", "mouse", "house", "horse", "nose", "vase", "hose",
              "purse", "cheese", "blouse", "suitcase", "briefcase", "fence", "police"}


def plural(name: str) -> bool:
    head = name.split()[-1]
    if head in SINGULAR_S or head.endswith(("ss", "us", "is", "se", "ce")):
        return False
    return head.endswith("s")


def fill(template: str, obj: str, val: str | None = None) -> str:
    p = plural(obj)
    words = {"obj": obj, "val": val or "",
             "be": "are" if p else "is", "Be": "Are" if p else "Is",
             "does": "do" if p else "does", "Does": "Do" if p else "Does",
             "that": "those" if p else "that", "That": "Those" if p else "That",
             "this": "these" if p else "this", "This": "These" if p else "This",
             "looks": "look" if p else "looks", "appears": "appear" if p else "appears",
             "has": "have" if p else "has"}
    return template.format(**words)
NOUN_MATERIALS = {"metal", "glass", "plastic", "brick", "concrete", "stone", "marble", "leather",
                  "paper", "cardboard", "rubber", "wicker", "cloth", "denim", "wool", "straw",
                  "bamboo", "clay", "canvas", "mesh", "ceramic", "tile"}

COLOUR_WORDS = set(GROUPS[("color", "color")]) | {"colour", "colored", "coloured"}

# Values that a human could confuse with one another: never used as a FALSE boolean claim
# against each other.  (Two values also count as confusable when they share a word.)
CONFUSABLE = [
    {"white", "cream", "ivory", "beige", "light gray", "silver", "gray"},
    {"gray", "silver", "black", "dark gray"},
    {"brown", "tan", "beige", "khaki", "bronze", "copper", "gold", "light brown", "dark brown",
     "maroon", "olive"},
    {"blue", "navy", "teal", "turquoise", "light blue", "dark blue"},
    {"green", "olive", "teal", "turquoise", "light green", "dark green"},
    {"red", "maroon", "burgundy", "orange", "pink", "dark red", "peach"},
    {"purple", "lavender", "maroon", "burgundy", "pink"},
    {"yellow", "gold", "cream", "orange", "peach", "khaki"},
    {"black", "dark brown", "dark blue", "dark green", "navy", "dark gray", "dark red"},
    {"multicolored", "black and white", "red and white", "blue and white", "green and white"},
    {"stone", "marble", "concrete", "brick", "clay", "ceramic", "tile"},
    {"wooden", "bamboo", "straw", "wicker", "cardboard", "paper"},
    {"cloth", "denim", "wool", "canvas", "leather", "carpeted", "mesh"},
    {"plastic", "rubber", "glass"},
    {"round", "oval", "curved", "cylindrical", "arched", "wavy"},
    {"square", "triangular", "diamond shaped", "hexagonal", "octagonal", "star shaped",
     "heart shaped"},
    {"flat", "straight", "slanted"},
    {"bent", "curved", "twisted", "wavy"},
    {"dirty", "dusty", "muddy"},
    {"sitting", "kneeling", "crouching", "leaning", "bending"},
    {"standing", "leaning", "walking", "perched", "landed"},
    {"walking", "running", "jumping"},
    {"laying", "sleeping", "resting"},
    {"flying", "floating", "jumping"},
    {"cooked", "burnt", "sliced"},
    {"raw", "unripe", "ripe"},
    {"full", "half full"},
    {"melted", "melting", "frozen"},
    {"young", "baby"},
    {"old", "adult"},
    {"eating", "drinking", "grazing"},
    {"playing", "swinging", "waving", "smiling", "talking"},
    {"skiing", "snowboarding", "skating", "skateboarding", "surfing", "climbing"},
    {"wet", "damp"},
    {"open", "closed"},          # a false "open/closed" claim is fine, but keep them apart from
]                                 # the unrelated state dimensions; same-dimension pairs are kept


# Mass / "stuff" nouns: their annotated box covers a fraction of their real extent, so covering
# the box leaves the same stuff visible elsewhere and a "hidden" item would still be answerable.
# Excluded from hidden AND control items, so the exclusion cannot leak the label.
STUFF_HEADS = {
    "sky", "cloud", "clouds", "sun", "sunlight", "light", "lighting", "shadow", "shadows",
    "reflection", "reflections", "background", "foreground", "air", "smoke", "fog", "steam",
    "grass", "lawn", "weeds", "vegetation", "foliage", "leaves", "leafs", "bushes", "bush",
    "shrubs", "hedge", "hedges", "trees", "plants", "flowers", "moss", "ivy",
    "water", "ocean", "sea", "lake", "river", "pond", "waves", "wave", "surf", "foam", "snow",
    "ice", "sand", "dirt", "mud", "gravel", "rocks", "stones", "pebbles", "soil",
    "ground", "floor", "flooring", "carpet", "carpeting", "rug", "ceiling", "wall", "walls",
    "wallpaper", "tile", "tiles", "tiling", "brick", "bricks", "siding", "paneling",
    "street", "road", "roadway", "highway", "sidewalk", "pavement", "asphalt", "concrete",
    "pathway", "path", "field", "beach", "shore", "hill", "hills", "mountain", "mountains",
    "forest", "woods", "grassland", "pasture", "meadow", "yard", "garden", "fence", "fencing",
    "railing", "railings", "hair", "fur", "skin", "wool", "beard", "mane",
    "people", "crowd", "letters", "letter", "writing", "text", "words", "word", "lines", "line",
    "stripes", "spots", "design", "pattern", "edge", "side", "top", "bottom", "part", "area",
    "surface", "corner", "piece", "section", "row",
}


def dim_hits(text: str) -> Counter:
    """(family, dimension) buckets named by the words of `text` - used to drop self-answering items.

    Visual Genome object names often carry an attribute ("blue sky", "metal poles"), and GQA
    referring expressions do the same.  Asking for that very attribute would let a model answer
    from the question alone, which would silently turn a hidden-evidence item into an easy one.
    """
    words = [w for w in "".join(ch if ch.isalnum() or ch == " " else " " for ch in text.lower()).split()]
    hits: Counter = Counter()
    i = 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        hit = INDEX.get(two)
        if hit:
            hits[(hit[0], hit[1])] += 1
            i += 2
            continue
        hit = INDEX.get(words[i])
        if hit:
            hits[(hit[0], hit[1])] += 1
        i += 1
    return hits


def confusable(a: str, b: str) -> bool:
    if set(a.split()) & set(b.split()):
        return True
    return any(a in grp and b in grp for grp in CONFUSABLE)


# ----------------------------------------------------------------- quotas

class Quota:
    """Greedy 'least-filled cell first' scheduler - keeps observed shares close to the targets."""

    def __init__(self, targets: dict):
        self.targets = {k: v for k, v in targets.items() if v > 0}
        self.count: Counter = Counter()

    def pick(self, feasible, salt: str):
        best = None
        for k in feasible:
            t = self.targets.get(k)
            if not t:
                continue
            key = (round((self.count[k] + 1) / t, 9), stable_rank(SEED, salt, k))
            if best is None or key < best[0]:
                best = (key, k)
        if best is None:
            return None
        self.count[best[1]] += 1
        return best[1]

    def charge(self, k):
        self.count[k] += 1

    def refund(self, k):
        self.count[k] -= 1


# ----------------------------------------------------------------- geometry

def pad_box(box, pad, W, H):
    x, y, w, h = box
    dx, dy = pad * w, pad * h
    return (max(0, int(math.floor(x - dx))), max(0, int(math.floor(y - dy))),
            min(W, int(math.ceil(x + w + dx))), min(H, int(math.ceil(y + h + dy))))


def crop_ok(rect, W, H):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    return w > 0 and h > 0 and (w * h) >= CROP_MIN_AREA * W * H and min(w, h) >= CROP_MIN_SHORT


def hidden_crops(box, W, H):
    """Crops that put the object's box fully outside the frame."""
    x, y, w, h = box
    bx0, by0, bx1, by1 = int(math.floor(x)), int(math.floor(y)), int(math.ceil(x + w)), int(math.ceil(y + h))
    out = []
    for name, rect in (("cut_left", (bx1, 0, W, H)), ("cut_right", (0, 0, bx0, H)),
                       ("cut_top", (0, by1, W, H)), ("cut_bottom", (0, 0, W, by0))):
        if crop_ok(rect, W, H):
            removed = 1.0 - ((rect[2] - rect[0]) * (rect[3] - rect[1])) / (W * H)
            out.append((name, rect, removed))
    return out


def control_crops(pbox, W, H, want):
    """Crops that keep the padded box fully inside, removing about `want` of the area."""
    px0, py0, px1, py1 = pbox
    out = []
    for name, free, axis in (("cut_left", px0 / W, "x0"), ("cut_right", (W - px1) / W, "x1"),
                             ("cut_top", py0 / H, "y0"), ("cut_bottom", (H - py1) / H, "y1")):
        f = min(want, max(0.0, free - 0.01))
        if f < 0.10:
            continue
        if axis == "x0":
            rect = (int(round(f * W)), 0, W, H)
        elif axis == "x1":
            rect = (0, 0, W - int(round(f * W)), H)
        elif axis == "y0":
            rect = (0, int(round(f * H)), W, H)
        else:
            rect = (0, 0, W, H - int(round(f * H)))
        if crop_ok(rect, W, H):
            removed = 1.0 - ((rect[2] - rect[0]) * (rect[3] - rect[1])) / (W * H)
            out.append((name, rect, removed))
    return out


def control_patch(pbox, W, H, rng):
    """A rectangle the size of the padded box that does not intersect the padded box at all."""
    px0, py0, px1, py1 = pbox
    pw, ph = px1 - px0, py1 - py0
    for shrink in (1.0, 0.85, 0.7, 0.55):
        w, h = max(16, int(pw * shrink)), max(16, int(ph * shrink))
        strips = []
        if px0 >= w:
            strips.append(("left", (0, px0 - w), (0, max(0, H - h))))
        if W - px1 >= w:
            strips.append(("right", (px1, W - w), (0, max(0, H - h))))
        if py0 >= h:
            strips.append(("top", (0, max(0, W - w)), (0, py0 - h)))
        if H - py1 >= h:
            strips.append(("bottom", (0, max(0, W - w)), (py1, H - h)))
        strips = [s for s in strips if s[1][0] <= s[1][1] and s[2][0] <= s[2][1]]
        if not strips:
            continue
        _, xr, yr = strips[rng.randrange(len(strips))]
        x = rng.randint(xr[0], xr[1])
        y = rng.randint(yr[0], yr[1])
        rect = (x, y, x + w, y + h)
        if rect[2] <= px0 or rect[0] >= px1 or rect[3] <= py0 or rect[1] >= py1:
            return rect
    return None


# ----------------------------------------------------------------- image treatment

def apply_patch(img: Image.Image, rect, style: str, seed: int) -> Image.Image:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    img = img.copy()
    if style in ("flat_grey", "flat_black", "flat_white"):
        fill = {"flat_grey": (128, 128, 128), "flat_black": (0, 0, 0),
                "flat_white": (255, 255, 255)}[style]
        img.paste(Image.new("RGB", (w, h), fill), (x0, y0))
    elif style == "mean_colour":
        arr = np.asarray(img, dtype=np.float64)
        total = arr.sum(axis=(0, 1)) - arr[y0:y1, x0:x1].sum(axis=(0, 1))
        n = arr.shape[0] * arr.shape[1] - w * h
        mean = tuple(int(round(v / n)) for v in total) if n > 0 else (128, 128, 128)
        img.paste(Image.new("RGB", (w, h), mean), (x0, y0))
    elif style == "gaussian_blur":
        r = max(14.0, 0.55 * min(w, h))
        m = int(math.ceil(r * 2))
        W, H = img.size
        cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
        cx1, cy1 = min(W, x1 + m), min(H, y1 + m)
        ctx = img.crop((cx0, cy0, cx1, cy1)).filter(ImageFilter.GaussianBlur(r))
        img.paste(ctx.crop((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)), (x0, y0))
    elif style == "pixelate":
        cell = max(8, max(w, h) // 3)
        nx, ny = max(1, w // cell), max(1, h // cell)
        region = img.crop(rect).resize((nx, ny), Image.BOX).resize((w, h), Image.NEAREST)
        img.paste(region, (x0, y0))
    elif style == "noise":
        rng = np.random.default_rng(seed)
        img.paste(Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)), (x0, y0))
    else:
        raise ValueError(style)
    return img


def render(src: Path, spec: dict) -> tuple[bytes, int, int] | None:
    try:
        with Image.open(src) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            out = im.crop(tuple(spec["rect"])) if spec["kind"] == "crop" \
                else apply_patch(im, tuple(spec["rect"]), spec["style"], spec["seed"])
            buf = io.BytesIO()
            out.save(buf, "JPEG", quality=90)
            return buf.getvalue(), out.width, out.height
    except Exception:
        return None


def store(data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = IMAGES / f"{sha}.jpg"
    if not path.exists():
        tmp = IMAGES / f".{sha}.tmp"
        tmp.write_bytes(data)
        tmp.replace(path)
    return sha


# ----------------------------------------------------------------- inputs

def partitions() -> tuple[set[int], set[int]]:
    """VG image ids that the donor sources put in train/dev, and the ones they put in test."""
    fit, test = set(), set()
    for src in ("vg_attributes", "gqa"):
        for line in (DATA_ROOT / src / "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            (test if r["partition"] == "test" else fit).add(int(r["source_group"]))
    return fit - test, test


def vg_objects(keep: set[int]) -> dict[int, dict]:
    """image_id -> {object_id: (name, box, typed attribute values, blocked values)} (filtered)."""
    with zipfile.ZipFile(ATTR_ZIP) as z:
        vg = json.loads(z.read("attributes.json"))
    dims = {r["image_id"]: (r["width"], r["height"]) for r in vg_image_data()}
    out = {}
    for im in vg:
        iid = im["image_id"]
        if iid not in keep:
            continue
        W, H = dims.get(iid, (0, 0))
        if not W or not H:
            continue
        objs = []
        for o in im["attributes"]:
            names = [str(n).strip().lower() for n in (o.get("names") or []) if str(n).strip()]
            if not names:
                continue
            name = names[0]
            if not (1 <= len(name.split()) <= 3) or len(name) > 30 or not name.replace(" ", "").isalpha():
                continue
            objs.append((name, o))
        counts = Counter(n for n, _ in objs)
        syn_counts = Counter(s for _, o in objs for s in set(o.get("synsets") or []))
        kept = {}
        for name, o in objs:
            if counts[name] != 1 or name.split()[-1] in STUFF_HEADS:
                continue
            if any(other != name and (name in other.split() or name in other) for other in counts):
                continue
            syns = set(o.get("synsets") or [])
            # a second annotated instance of the same *concept* ("tree" and "palm tree") would
            # leave the evidence visible after the first one is covered
            if not syns or any(syn_counts[s] != 1 for s in syns):
                continue
            if o["w"] < MIN_SIDE_PX or o["h"] < MIN_SIDE_PX:
                continue
            if not (MIN_AREA <= o["w"] * o["h"] / (W * H) <= MAX_AREA):
                continue
            typed = defaultdict(set)
            for a in (o.get("attributes") or []):
                hit = INDEX.get(str(a).strip().lower())
                if hit:
                    typed[(hit[0], hit[1])].add(hit[2])
            blocked = {g for gs in typed.values() for g in gs}
            kept[o["object_id"]] = {"name": name, "box": (o["x"], o["y"], o["w"], o["h"]),
                                    "typed": {k: next(iter(v)) for k, v in typed.items() if len(v) == 1},
                                    "blocked": blocked, "name_dims": set(dim_hits(name))}
        if kept:
            out[iid] = {"vg_size": (W, H), "objects": kept}
    return out


GQA_TYPES = {"verifyAttr", "verifyAttrC", "verifyAttrs", "verifyAttrsC", "verifyAttrK",
             "verifyAttrKC", "chooseAttr", "material", "materialChoose", "activity",
             "activityChoose", "category", "typeChoose", "state", "stateChoose", "company"}


def gqa_rows(keep_by_image: dict) -> tuple[list[dict], dict]:
    """Single-object GQA questions whose object survives the referent filter, plus answer pools."""
    scopes = {k: defaultdict(Counter) for k in ("local", "glob", "detailed", "sem")}
    rows = []
    for sub, split in (("train_balanced_instructions", "train"), ("val_balanced_instructions", "val")):
        for shard in sorted((GQA_INSTR / sub).glob("*.parquet")):
            d = pq.read_table(shard, columns=["id", "imageId", "question", "answer", "annotations",
                                              "types", "groups"]).to_pydict()
            for i in range(len(d["id"])):
                a, ty, g = d["answer"][i], d["types"][i], d["groups"][i] or {}
                key = {"local": g.get("local") or "", "glob": g.get("global") or "",
                       "detailed": (ty or {}).get("detailed") or "", "sem": (ty or {}).get("semantic") or ""}
                if a not in ("yes", "no") and a and len(a) <= 128:
                    for k, v in key.items():
                        if v:
                            scopes[k][v][a] += 1
                iid = d["imageId"][i]
                if not iid.isdigit() or int(iid) not in keep_by_image:
                    continue
                if key["detailed"] not in GQA_TYPES:
                    continue
                ann = d["annotations"][i] or {}
                oids = {x["value"] for k in ("question", "answer", "fullAnswer")
                        for x in (ann.get(k) or [])}
                if len(oids) != 1:
                    continue
                oid = next(iter(oids))
                if not oid.isdigit() or int(oid) not in keep_by_image[int(iid)]["objects"]:
                    continue
                q = d["question"][i]
                if not q or len(q) > 300:
                    continue
                rows.append({"qid": d["id"][i], "img": int(iid), "oid": int(oid), "q": q, "a": a,
                             "split": split, **key})
    pools = {k: {g: [a for a, _ in c.most_common(60)] for g, c in v.items()} for k, v in scopes.items()}
    return rows, pools


def gqa_distractors(row: dict, pools: dict) -> list[str]:
    gold = row["a"]
    gold_words = set(gold.split())
    seen, out = {gold}, []
    if " or " in row["q"]:
        tail = row["q"].rstrip("?").split(" or ")[-1].strip().strip(",")
        if tail and tail != gold and 0 < len(tail) <= 128:
            out.append(tail)
            seen.add(tail)
    for key in ("local", "glob", "detailed", "sem"):
        for cand in pools[key].get(row[key], ()):
            if cand in seen or not 0 < len(cand) <= 128:
                continue
            if cand in gold or gold in cand or (gold_words & set(cand.split())):
                continue
            seen.add(cand)
            out.append(cand)
    return out


# ----------------------------------------------------------------- candidates

def build_candidates(objects: dict, grows: list[dict], pools: dict) -> dict[int, list[dict]]:
    """image_id -> list of question candidates (each tied to one object box)."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for iid, info in objects.items():
        for oid, o in info["objects"].items():
            for (family, dim), gold in o["typed"].items():
                if (family, dim) in o["name_dims"]:
                    continue                       # the object's own name gives the answer away
                siblings = [g for g in GROUPS[(family, dim)] if g != gold and g not in o["blocked"]]
                if not siblings:
                    continue
                clear = [g for g in siblings if not confusable(g, gold)]
                by_image[iid].append({
                    "kind": "vg", "img": iid, "oid": oid, "name": o["name"], "box": o["box"],
                    "family": family, "dim": dim, "gold": gold, "siblings": siblings,
                    "clear": clear, "colour": family == "color",
                    "key": f"{oid}-{dim}"})
    for r in grows:
        o = objects[r["img"]]["objects"][r["oid"]]
        yesno = r["a"] in ("yes", "no")
        hits = dim_hits(r["q"])
        if hits and max(hits.values()) > 1:
            continue                               # "Are the brown tiles brown?" answers itself
        if not yesno and f" {r['a'].lower()} " in f" {r['q'].lower()} ".replace("?", " "):
            continue                               # the question already contains the answer
        options = [] if yesno else gqa_distractors(r, pools)
        if not yesno and len(options) < 1:
            continue
        colour = (r["detailed"].endswith("C") or "color" in r["q"].lower()
                  or "colour" in r["q"].lower() or r["a"] in COLOUR_WORDS)
        by_image[r["img"]].append({
            "kind": "gqa", "img": r["img"], "oid": r["oid"], "name": o["name"], "box": o["box"],
            "family": "gqa", "dim": r["detailed"], "gold": r["a"], "question": r["q"],
            "options": options, "yesno": yesno, "colour": colour, "qid": r["qid"],
            "key": f"{r['oid']}-{r['qid']}"})
    for v in by_image.values():
        v.sort(key=lambda c: stable_rank(SEED, "cand", c["img"], c["key"]))
    return by_image


def field_types(c: dict) -> list[str]:
    if c["kind"] == "gqa":
        return ["boolean"] if c["yesno"] else ["choice"]
    out = ["choice"]
    if c["clear"]:
        out.append("boolean")
    return out


# ----------------------------------------------------------------- planning

def plan_item(c: dict, side: str, treatment: str, proc: tuple[int, int], vg_size: tuple[int, int],
              rng: random.Random):
    """Turn (candidate, side, treatment) into a concrete image spec, or None if infeasible."""
    W, H = proc
    sx, sy = W / vg_size[0], H / vg_size[1]
    x, y, w, h = c["box"]
    box = (x * sx, y * sy, w * sx, h * sy)
    pad = PAD_LO + (PAD_HI - PAD_LO) * rng.random()
    pbox = pad_box(box, pad, W, H)
    if pbox[2] - pbox[0] < 8 or pbox[3] - pbox[1] < 8:
        return None
    # a hidden crop must remove the whole object, so it tends to cut more; aim the control crops
    # at a slightly larger share of the frame to keep the two distributions comparable
    want = (0.15 if side == "hidden" else 0.25) + 0.30 * rng.random()
    if treatment == "crop":
        opts = hidden_crops(box, W, H) if side == "hidden" else control_crops(pbox, W, H, want)
        if not opts:
            return None
        opts.sort(key=lambda o: (abs(o[2] - want), o[0]))
        name, rect, removed = opts[0]
        return {"kind": "crop", "style": "crop", "rect": list(rect), "seed": 0,
                "side_note": name, "removed": removed, "pbox": list(pbox)}
    rect = pbox if side == "hidden" else control_patch(pbox, W, H, rng)
    if rect is None:
        return None
    return {"kind": "patch", "style": treatment, "rect": list(rect),
            "seed": int(stable_rank(SEED, "noise", c["img"], c["key"], side)[:8], 16),
            "side_note": "on_object" if side == "hidden" else "off_object",
            "removed": ((rect[2] - rect[0]) * (rect[3] - rect[1])) / (W * H), "pbox": list(pbox)}


def feasible_treatments(c: dict, side: str, proc, vg_size, salt: str) -> list[str]:
    ok = []
    for t in TREATMENTS:
        if t in COLOUR_BLIND_TREATMENTS and c["colour"]:
            continue
        rng = random.Random(stable_rank(SEED, salt, c["key"], side, t))
        if plan_item(c, side, t, proc, vg_size, rng) is not None:
            ok.append(t)
    return ok


# ----------------------------------------------------------------- record building

def build_record(c, side, treatment, spec, image, partition, quotas, salt) -> dict | None:
    rng = random.Random(stable_rank(SEED, "rec", c["key"], side))
    hidden = side == "hidden"
    ftype = quotas["field"].pick(field_types(c), salt)
    if ftype is None:
        return None
    statement = False
    if ftype == "boolean":
        if c["kind"] == "gqa":
            quotas["phrasing"].charge("question")
            quotas["claim"].charge("true" if c["gold"] == "yes" else "false")
            question = c["question"]
            claim_true = c["gold"] == "yes"
            answer = c["gold"]
        else:
            statement = quotas["phrasing"].pick(["statement", "question"], salt) == "statement"
            claim_true = quotas["claim"].pick(["true", "false"], salt) == "true"
            val = c["gold"] if claim_true else c["clear"][rng.randrange(len(c["clear"]))]
            noun = c["family"] == "material" and val in NOUN_MATERIALS and rng.random() < 0.5
            pool = (MADE_OF_STATEMENTS if noun else BOOL_STATEMENTS) if statement else \
                   (MADE_OF_QUESTIONS if noun else BOOL_QUESTIONS)
            question = fill(pool[rng.randrange(len(pool))], c["name"], val)
            answer = "yes" if claim_true else "no"
        field = {"id": "answer", "type": "boolean", "question": question}
        target = None if hidden else claim_true
        source_answer = answer
    else:
        if c["kind"] == "gqa":
            question, gold, pool = c["question"], c["gold"], list(c["options"])
        else:
            tpls = TEMPLATES[c["family"]]
            question = fill(tpls[rng.randrange(len(tpls))], c["name"])
            gold, pool = c["gold"], list(c["siblings"])
        want = rng.randint(13, 25) if rng.random() < BIG_OPTIONS_RATE else rng.randint(2, 12)
        values = pool[:max(1, min(want - 1, len(pool)))] + [gold]
        values = list(dict.fromkeys(values))
        if len(values) < 2:
            return None
        rng.shuffle(values)
        field = {"id": "answer", "type": "choice", "question": question,
                 "options": [{"value": v} for v in values]}
        target = None if hidden else gold
        source_answer = gold
    request = {"request_id": f"masked-{c['img']}-{c['key']}-{side[0]}"[:128], "state": {},
               "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{c['img']}-{c['key']}-{side[0]}", "source": SOURCE,
            "source_split": "train" if partition != "test" else "val",
            "source_group": str(c["img"]), "family": c["family"], "license": LICENSE,
            "images": [image], "request": request, "target": target,
            "abstention_cause": "insufficient_evidence" if hidden else None,
            "source_answer": source_answer, "partition": partition,
            "hidden_evidence": hidden, "treatment": treatment,
            "treatment_detail": spec["side_note"],
            "treated_area_fraction": round(spec["removed"], 4),
            "object_name": c["name"], "referent_source": c["kind"], "question_kind": c["dim"],
            "phrasing": "statement" if statement else "question"}


# ----------------------------------------------------------------- selection

def select(pool_images: list[int], cands: dict, objects: dict, proc_size: dict, total: int):
    """Walk images in hash order: pairs (1 hidden + 1 control, different objects) first, then
    control-only singles, until the hidden/control split matches HIDDEN_SHARE."""
    n_hidden = round(total * HIDDEN_SHARE)
    n_control = total - n_hidden
    quotas = {}
    for side, n in (("hidden", n_hidden), ("control", n_control)):
        quotas[side] = {
            "treatment": Quota({t: n * s for t, s in TREATMENT_SHARE.items()}),
            "family": Quota({f: n * s for f, s in FAMILY_SHARE.items()}),
            "field": Quota({"boolean": n * BOOLEAN_SHARE, "choice": n * (1 - BOOLEAN_SHARE)}),
            "phrasing": Quota({"statement": n * BOOLEAN_SHARE * STATEMENT_SHARE,
                               "question": n * BOOLEAN_SHARE * (1 - STATEMENT_SHARE)}),
            "claim": Quota({"true": n * BOOLEAN_SHARE * 0.5, "false": n * BOOLEAN_SHARE * 0.5}),
        }
    order = sorted(pool_images, key=lambda i: stable_rank(SEED, "order", i))
    plans: list[tuple] = []
    made = {"hidden": 0, "control": 0}

    def emit(iid, side, used_oids):
        """Returns (candidate, side, treatment, spec, charged) or None; `charged` allows a refund."""
        proc = proc_size[iid]
        vg_size = objects[iid]["vg_size"]
        options = [c for c in cands[iid] if c["oid"] not in used_oids]
        if not options:
            return None
        feas = {c["key"]: feasible_treatments(c, side, proc, vg_size, "feas") for c in options}
        union = {t for ts in feas.values() for t in ts}
        t = quotas[side]["treatment"].pick([x for x in TREATMENTS if x in union], f"t{iid}")
        if t is None:
            return None
        pick_from = [c for c in options if t in feas[c["key"]]]
        fam = quotas[side]["family"].pick(sorted({c["family"] for c in pick_from}), f"f{iid}")
        if fam is None:
            fam = pick_from[0]["family"]
            quotas[side]["family"].charge(fam)
        c = next(x for x in pick_from if x["family"] == fam)
        rng = random.Random(stable_rank(SEED, "feas", c["key"], side, t))
        spec = plan_item(c, side, t, proc, vg_size, rng)
        if spec is None:
            quotas[side]["treatment"].refund(t)
            quotas[side]["family"].refund(fam)
            return None
        made[side] += 1
        return (c, side, t, spec, (t, fam))

    def refund(got):
        c, side, t, spec, (ct, cf) = got
        quotas[side]["treatment"].refund(ct)
        quotas[side]["family"].refund(cf)
        made[side] -= 1

    # phase 1: pairs (one hidden + one control, different objects), until the hidden quota is full
    for iid in order:
        if made["hidden"] >= n_hidden:
            break
        if len({c["oid"] for c in cands[iid]}) < 2:
            continue
        # alternate which side chooses its object first: whoever picks first gets the rarer
        # question kinds (GQA questions cluster on one object per image), and a systematic
        # first-pick advantage would make the question *kind* predict hidden vs control.
        first = "hidden" if stable_rank(SEED, "which", iid) < "8" else "control"
        second = "control" if first == "hidden" else "hidden"
        got1 = emit(iid, first, set())
        if got1 is None:
            continue
        got2 = emit(iid, second, {got1[0]["oid"]})
        if got2 is None:
            refund(got1)
            continue
        plans.extend([got1[:4], got2[:4]])
    used = {p[0]["img"] for p in plans}
    # phase 2: control-only singles on fresh images
    for iid in order:
        if made["control"] >= n_control:
            break
        if iid in used:
            continue
        got = emit(iid, "control", set())
        if got is not None:
            plans.append(got[:4])
    return plans, quotas


# ----------------------------------------------------------------- contact sheet

def contact_sheet(records: list[dict], path: Path):
    picks = []
    for side in (True, False):
        per_t = defaultdict(list)
        for r in sorted(records, key=lambda r: r["id"]):
            if r["hidden_evidence"] is side:
                per_t[r["treatment"]].append(r)
        side_picks, rnd = [], 0
        while len(side_picks) < 12 and rnd < 6:
            for t in TREATMENTS:
                if len(side_picks) >= 12:
                    break
                if len(per_t[t]) > rnd:
                    side_picks.append(per_t[t][rnd])
            rnd += 1
        picks.extend(side_picks)
    picks = picks[:24]
    cols, cw, ih, cap = 4, 340, 250, 96
    rows = math.ceil(len(picks) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * (ih + cap)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 12)
    except Exception:
        font = bold = ImageFont.load_default()
    for i, r in enumerate(picks):
        cx, cy = (i % cols) * cw, (i // cols) * (ih + cap)
        with Image.open(r["images"][0]["image"]) as im:
            im = im.copy()
            im.thumbnail((cw - 12, ih - 12))
        sheet.paste(im, (cx + (cw - im.width) // 2, cy + 6))
        head = ("HIDDEN -> unknown" if r["hidden_evidence"] else "CONTROL -> answerable")
        draw.text((cx + 6, cy + ih), f"{head}  [{r['treatment']}]", (150, 0, 0) if r["hidden_evidence"]
                  else (0, 110, 0), font=bold)
        q = r["request"]["fields"][0]["question"]
        line, y = "", cy + ih + 16
        for word in q.split():
            if draw.textlength(line + " " + word, font=font) > cw - 12:
                draw.text((cx + 6, y), line, (20, 20, 20), font=font)
                y += 14
                line = word
            else:
                line = (line + " " + word).strip()
        draw.text((cx + 6, y), line, (20, 20, 20), font=font)
        tgt = "unknown" if r["target"] is None else str(r["target"])
        draw.text((cx + 6, cy + ih + cap - 16), f"target: {tgt}   (answer: {r['source_answer']})",
                  (60, 60, 60), font=font)
    sheet.save(path, "PNG")


# ----------------------------------------------------------------- main

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--smoke":   # debug only: tiny, non-canonical build
        FIT_TARGET, TEST_TARGET = int(sys.argv[2]), int(sys.argv[3])
        DEV_TARGET = max(1, FIT_TARGET // 33)
    IMAGES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()

    fit_ids, test_ids = partitions()
    cached = {int(p.stem) for p in VG_PROC.glob("*.jpg")}
    fit_ids = (fit_ids & cached) - excl["vg"]
    test_ids = test_ids & cached
    print(f"donor images: fit={len(fit_ids)} test={len(test_ids)}", flush=True)

    objects = vg_objects(fit_ids | test_ids)
    print(f"images with usable objects: {len(objects)}", flush=True)

    grows, pools = gqa_rows(objects)
    print(f"gqa single-object questions: {len(grows)}", flush=True)

    cands = build_candidates(objects, grows, pools)
    print(f"images with candidates: {len(cands)} candidates: {sum(len(v) for v in cands.values())}",
          flush=True)

    size_cache = CACHE / "proc_sizes.json"
    proc_size: dict[int, tuple[int, int]] = {}
    if size_cache.exists():
        proc_size = {int(k): tuple(v) for k, v in json.loads(size_cache.read_text()).items()}
    missing = [i for i in cands if i not in proc_size]
    if missing:
        print(f"measuring {len(missing)} images...", flush=True)

        def size_of(i):
            try:
                with Image.open(VG_PROC / f"{i}.jpg") as im:
                    return i, im.size
            except Exception:
                return i, None
        with ThreadPoolExecutor(16) as ex:
            for i, s in ex.map(size_of, missing):
                if s:
                    proc_size[i] = s
        size_cache.write_text(json.dumps({str(k): list(v) for k, v in sorted(proc_size.items())}))

    # a proc image must be a uniform rescale of the Visual Genome frame, or the boxes do not map
    usable = []
    for iid in cands:
        if iid not in proc_size:
            continue
        W, H = proc_size[iid]
        vw, vh = objects[iid]["vg_size"]
        if abs((W / vw) / (H / vh) - 1) > 0.02 or W < CROP_MIN_SHORT or H < CROP_MIN_SHORT:
            continue
        usable.append(iid)
    fit_pool = [i for i in usable if i in fit_ids]
    test_pool = [i for i in usable if i in test_ids]
    print(f"usable images: fit={len(fit_pool)} test={len(test_pool)}", flush=True)

    all_plans, all_quotas = [], {}
    for name, pool, total in (("fit", fit_pool, FIT_TARGET), ("test", test_pool, TEST_TARGET)):
        plans, quotas = select(pool, cands, objects, proc_size, total)
        print(f"{name}: planned {len(plans)} items "
              f"(hidden={sum(1 for p in plans if p[1] == 'hidden')})", flush=True)
        all_plans.append((name, plans, quotas))
        all_quotas[name] = quotas

    print("rendering treated images...", flush=True)
    records = []
    for name, plans, quotas in all_plans:
        jobs = [(c, side, t, spec) for c, side, t, spec in plans]

        def work(job):
            c, side, t, spec = job
            return render(VG_PROC / f"{c['img']}.jpg", spec)
        done = 0
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(work, jobs))
        for (c, side, t, spec), got in zip(jobs, results):
            done += 1
            if done % 5000 == 0:
                print(f"  {name} {done}/{len(jobs)}", flush=True)
            if got is None:
                continue
            data, w, h = got
            if w * h > 1_000_000:
                continue
            sha = store(data)
            partition = "test" if name == "test" else "train"
            if partition != "test" and sha in excl["sha256"]:
                continue
            image = {"image": str(IMAGES / f"{sha}.jpg"), "sha256": sha, "width": w, "height": h}
            rec = build_record(c, side, t, spec, image, partition, quotas[side],
                               f"{name}-{c['img']}")
            if rec:
                records.append(rec)

    fit_recs = [r for r in records if r["partition"] != "test"]
    groups = sorted({r["source_group"] for r in fit_recs}, key=lambda g: stable_rank(SEED, "dev", g))
    per_group = Counter(r["source_group"] for r in fit_recs)
    dev_groups, acc = set(), 0
    for g in groups:
        if acc >= DEV_TARGET:
            break
        dev_groups.add(g)
        acc += per_group[g]
    for r in fit_recs:
        if r["source_group"] in dev_groups:
            r["partition"] = "dev"

    records.sort(key=lambda r: r["id"])
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))

    keep = {Path(r["images"][0]["image"]).name for r in records}
    for f in IMAGES.iterdir():
        if f.name not in keep:
            f.unlink()

    contact_sheet(records, OUT / "examples.png")

    cross = Counter((r["treatment"], "hidden" if r["hidden_evidence"] else "control") for r in records)
    summary = {
        "records": len(records),
        "partition": dict(Counter(r["partition"] for r in records)),
        "hidden": sum(1 for r in records if r["hidden_evidence"]),
        "family": dict(Counter(r["family"] for r in records)),
        "field": dict(Counter(r["request"]["fields"][0]["type"] for r in records)),
        "target": dict(Counter("unknown" if r["target"] is None else
                               (str(r["target"]).lower() if isinstance(r["target"], bool) else "choice")
                               for r in records)),
        "phrasing": dict(Counter(r["phrasing"] for r in records
                                 if r["request"]["fields"][0]["type"] == "boolean")),
        "referent_source": dict(Counter(r["referent_source"] for r in records)),
        "treatment_x_side": {f"{t}|{s}": n for (t, s), n in sorted(cross.items())},
        "options": dict(sorted(Counter(len(r["request"]["fields"][0]["options"]) for r in records
                                       if r["request"]["fields"][0]["type"] == "choice").items())),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
