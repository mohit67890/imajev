"""Phase-3 Stage 0-I: composited inventory scenes (families `inventory_*`, source I, dataset `inventory_synthetic`).

We build shelf / cooler / pallet-rack / bin-wall / pegboard scenes ourselves and place every product, label and carton,
so the gold is exact by construction (docs/phase-3-plan.md, image branch). Products are cut out of our licensed ABO
listing photos (Amazon Berkeley Objects, CC BY 4.0, train partition only; white background -> alpha matte, see
inventory_assets/common.py) or drawn procedurally (cooler bottles and cans, cartons, bins, shelving, labels, barcodes).

Question families (field type):
  inventory_count        (choice)  units of one product on one shelf / in one location
  inventory_total        (choice)  units of one product on the whole unit
  inventory_colour_count (choice)  cooler: containers of one colour on one shelf
  inventory_out_of_stock (choice / noul)  label up, slot empty
  inventory_misplaced    (choice)  a slot holds a product other than the one its label (planogram) names
  inventory_price_tag    (choice)  a shelf label price differs from the price file in the state (joint)
  inventory_reorder      (choice / noul)  shelf units + backroom <= reorder point (joint image + state)
  inventory_locate       (choice)  which shelf / row holds product X
  inventory_compare      (choice)  which product has the most units on display
  inventory_audit        (score)   how many slots on a shelf break the planogram (empty, wrong product, wrong price)
  inventory_not_visible  (noul / choice, always unknown)  asks for something no photo can settle (stock behind the
                                   front row, sell-by dates, backroom counts that are not in the state)

Unknowns (~15%): an occluder (stock cart, box stack, promo banner, pillar) covers part of the unit, or the frame is
cropped through a column, and the question depends on what is hidden; or the question names something no photo can
show; or a "which X" question has no such X (false_premise). The gold is settled only when every alternative content
of the hidden slots (empty, one more unit, another product, another label SKU / price) gives the same answer
(`settle`); tests/test_p3_inventory_safety.py re-derives it from provenance.spec.

    .venv/bin/python scripts/p3/gen_inventory.py [--count 4000] [--seed p3-inventory-v1] [--out ...]
    .venv/bin/python scripts/p3/gen_inventory.py --heldout --count 250          # fresh seed, held-out products/wording
    .venv/bin/python scripts/p3/gen_inventory.py --variant-of parents.jsonl --per-parent 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402
from convert_common import option_key  # noqa: E402
from inventory_assets.common import (affine_params, apply_post, cutout, encode_jpeg, fit_font, font, is_multi,  # noqa: E402
                                     map_box, paste_item, rng_for, store_image, text_center, text_size)

SEED = "p3-inventory-v1"
HELDOUT_SEED = "p3-inventory-heldout-fresh-v1"
IMG_DIR = ROOT / "data" / "p3" / "images" / "inventory"
OUT = ROOT / "data" / "p3" / "candidates" / "I-inventory.jsonl"
HELDOUT_OUT = ROOT / "data" / "p3" / "pool" / "heldout-fresh-inventory.jsonl"
ABO_EVIDENCE = "data/decision-v2/licenses/state_grounded/abo-LICENSE-CC-BY-4.0.txt"
ABO_CREDIT = "Amazon Berkeley Objects (Collins et al., CVPR 2022), https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
HELDOUT_PRODUCT_MOD = 10          # ABO items with sha(item) % 10 == 0 are held-out-only (never in a training scene)

# ABO product type -> (group, singular noun, plural noun, countable). One product per group in a scene, so a product is
# always identified by what it is (the colour only helps). Footwear and types often sold as sets (cups, jars, plates,
# hangers, bulbs, baskets, candle holders) are not countable: one photo may show two or more units.
SHELF_TYPES = {
    "handbag": ("bag", "handbag", "handbags", True), "tote bag": ("bag", "tote bag", "tote bags", True),
    "backpack": ("backpack", "backpack", "backpacks", True), "wallet": ("wallet", "wallet", "wallets", True),
    "cosmetic case": ("cosmetic", "cosmetic case", "cosmetic cases", True),
    "lamp": ("lamp", "table lamp", "table lamps", True), "clock": ("clock", "wall clock", "wall clocks", True),
    "jar": ("jar", "jar", "jars", False), "drinking cup": ("cup", "cup", "cups", False),
    "bottle": ("bottle", "bottle", "bottles", True), "planter": ("pot", "planter", "planters", True),
    "vase": ("pot", "vase", "vases", True), "frying pan": ("pan", "frying pan", "frying pans", True),
    "dutch oven": ("oven", "casserole pot", "casserole pots", True), "hat": ("hat", "hat", "hats", True),
    "headphones": ("headphones", "pair of headphones", "pairs of headphones", True),
    "padlock": ("padlock", "padlock", "padlocks", True), "phone case": ("phonecase", "phone case", "phone cases", True),
    "tablet case": ("phonecase", "tablet case", "tablet cases", True), "eyeglasses": ("glasses", "pair of glasses", "pairs of glasses", True),
    "light bulb": ("bulb", "light bulb", "light bulbs", False), "basket": ("basket", "basket", "baskets", False),
    "pillow": ("pillow", "cushion", "cushions", True), "sports ball": ("ball", "ball", "balls", True),
    "trash can": ("bin", "waste bin", "waste bins", True), "electric fan": ("fan", "electric fan", "electric fans", True),
    "blender": ("blender", "blender", "blenders", True), "loudspeaker": ("speaker", "speaker", "speakers", True),
    "toy figure": ("toy", "toy figure", "toy figures", True), "umbrella": ("umbrella", "umbrella", "umbrellas", True),
    "jewellery box": ("jbox", "jewellery box", "jewellery boxes", True), "candle holder": ("candle", "candle holder", "candle holders", False),
    "ring binder": ("binder", "ring binder", "ring binders", False), "plate": ("plate", "plate", "plates", False),
    "bowl": ("plate", "bowl", "bowls", False), "electric kettle": ("kettle", "kettle", "kettles", True),
    "suitcase": ("case", "suitcase", "suitcases", True), "luggage": ("case", "travel bag", "travel bags", True),
    "shoe": ("footwear", "shoe", "shoes", False), "boot": ("footwear", "boot", "boots", False),
    "sandal": ("footwear", "sandal", "sandals", False), "athletic shoe": ("footwear", "trainer", "trainers", False),
    "stool": ("stool", "stool", "stools", True), "chair": ("chair", "chair", "chairs", True),
    "ottoman": ("ottoman", "footstool", "footstools", True), "clothes hanger": ("hanger", "hanger", "hangers", False),
}
# rough relative heights on a shelf (1.0 = the full slot height), so a padlock is not the size of a lamp
TYPE_SCALE = {"padlock": 0.42, "light bulb": 0.5, "phone case": 0.62, "tablet case": 0.75, "wallet": 0.5, "drinking cup": 0.5,
              "jar": 0.58, "clock": 0.78, "lamp": 1.0, "frying pan": 0.7, "dutch oven": 0.62, "hat": 0.62, "headphones": 0.66,
              "handbag": 0.88, "tote bag": 0.9, "backpack": 1.0, "planter": 0.72, "vase": 0.85, "bottle": 0.8, "basket": 0.72,
              "pillow": 0.85, "sports ball": 0.55, "trash can": 0.85, "electric fan": 1.0, "blender": 0.95, "loudspeaker": 0.75,
              "toy figure": 0.6, "umbrella": 1.0, "jewellery box": 0.48, "candle holder": 0.6, "ring binder": 0.8,
              "plate": 0.62, "bowl": 0.45, "electric kettle": 0.78, "shoe": 0.55, "boot": 0.9, "sandal": 0.5,
              "athletic shoe": 0.55, "cosmetic case": 0.55, "eyeglasses": 0.4, "clothes hanger": 0.6}
PEG_GROUPS = {"phonecase", "wallet", "glasses", "headphones", "padlock", "cosmetic", "hat", "bulb", "binder", "toy", "jbox",
              "hanger", "cup"}
BIG_GROUPS = {"stool", "chair", "ottoman", "case", "fan"}          # cartons / pallet rack only

# ------------------------------------------------------------------------------------------------ wording banks
# Every bank is split: the last third is reserved for the held-out file (13-gram leakage rule), the rest trains.


def pick(rng, bank, split):
    k = max(1, len(bank) // 3)
    part = bank[-k:] if split == "heldout" else bank[:-k]
    return rng.choice(part)


PHOTO_LINES = [
    "{kind} photo from store {store}, aisle {aisle}, taken on the {time} gap scan.",
    "Picture of {kindl} {aisle}-{bay} at site {store}, shot by the {time} replenishment walk.",
    "Store {store} / aisle {aisle} / bay {bay}: {kindl} photographed at {time}.",
    "Image uploaded by the associate at location {store} ({kindl}, aisle {aisle}), {time}.",
    "Aisle {aisle}, bay {bay} of branch {store}; {kindl} captured {time} for the stock check.",
    "Branch {store} stock photo, {kindl} in aisle {aisle} bay {bay}, time stamp {time}.",
]
KIND_NAMES = {"retail_shelf": ("Shelf", "the shelf unit"), "cooler": ("Cooler", "the drinks cooler"),
              "pallet_rack": ("Rack", "the pallet rack"), "bin_wall": ("Bin wall", "the bin wall"),
              "pegboard": ("Pegboard", "the pegboard display")}
REGION_NOUN = {"retail_shelf": "shelf", "cooler": "shelf", "pallet_rack": "level", "bin_wall": "row", "pegboard": "row"}
UNIT_NOUN = {"retail_shelf": "shelf unit", "cooler": "cooler", "pallet_rack": "rack", "bin_wall": "bin wall",
             "pegboard": "pegboard"}

Q_COUNT = [
    "How many {plural} are on {region}?",
    "Count the {plural} standing on {region}. How many are there?",
    "What is the number of {plural} on {region} of the {unit}?",
    "On {region}, how many {plural} can be seen?",
    "How many {plural} does {region} hold?",
    "Looking only at {region}, what is the count of {plural}?",
]
Q_COUNT_CARTON = [
    "How many cartons are stored in location {loc}?",
    "What is the carton count in rack location {loc}?",
    "Location {loc} holds how many cartons?",
    "How many cartons sit in bay position {loc}?",
    "Count the cartons in {loc}. How many are there?",
    "For rack slot {loc}, how many cartons are present?",
]
Q_TOTAL = [
    "How many {plural} are there on the whole {unit}?",
    "Across every {rnoun} of the {unit}, how many {plural} are displayed?",
    "In total, how many {plural} does this {unit} hold?",
    "What is the total number of {plural} visible on the {unit}?",
    "Adding up all the {rnoun}s, how many {plural} are on display?",
    "Summed over the entire {unit}, what is the number of {plural}?",
]
Q_COLOUR = [
    "How many {colour} {container}s are on {region}?",
    "Count the {colour} {container}s on {region}.",
    "On {region} of the cooler, how many {container}s are {colour}?",
    "What number of {colour} {container}s stand on {region}?",
    "How many {container}s with a {colour} body are on {region}?",
    "Looking at {region}, how many {colour} {container}s are there?",
]
Q_OOS = [
    "Which SKU has a shelf label but nothing on display above or in its slot?",
    "Which product is out of stock here (its label is up but its slot is empty)?",
    "Which labelled slot is empty?",
    "Which SKU's space on the {unit} has no stock in it?",
    "Find the empty slot: which SKU is it labelled for?",
    "The gap scan looks for labels with no stock. Which SKU is the gap?",
]
Q_OOS_NOUL = [
    "Is SKU {sku} out of stock on this {unit} (label present, slot empty)?",
    "Is the slot labelled {sku} empty?",
    "Does SKU {sku} have a label with no stock in its space?",
    "Would the gap scan report SKU {sku} as a gap?",
    "Is there nothing in the space whose label reads {sku}?",
    "Is SKU {sku} a gap on this {unit}?",
]
Q_MISPLACED = [
    "Which label has a different product in its slot than the planogram lists for that SKU?",
    "Which SKU's slot holds the wrong product?",
    "One slot contains stock that does not match its label. Which SKU label is it?",
    "Which labelled slot is holding a product that belongs to another SKU?",
    "Which SKU label sits under or on stock of a different item?",
    "Where is stock misplaced? Give the SKU on that slot's label.",
]
Q_PRICE = [
    "Which shelf label shows a price that differs from the price file?",
    "Which SKU's printed tag price does not match the price file?",
    "Checking tags against the file, which SKU has the wrong price on its label?",
    "Which label needs reprinting because its price disagrees with the price file?",
    "For which SKU is the price on the label not the file price?",
    "Which tag in the photo carries an out-of-date price according to the file?",
]
Q_REORDER = [
    "Which SKU has to be reordered under the rule?",
    "Under the reorder rule, which SKU should be ordered now?",
    "Which product triggers a reorder according to the rule and the counts?",
    "Applying the reorder rule to what is on display, which SKU must be reordered?",
    "Which SKU falls to or below its reorder point once shelf and backroom are added up?",
    "The rule flags one SKU for ordering. Which one?",
]
Q_REORDER_NOUL = [
    "Does SKU {sku} need to be reordered under the rule?",
    "Should SKU {sku} be ordered now according to the rule?",
    "Is SKU {sku} at or below its reorder point once displayed and backroom units are counted?",
    "Under the reorder rule, is an order due for SKU {sku}?",
    "Does the rule trigger a reorder of SKU {sku}?",
    "Must SKU {sku} be put on the order list?",
]
REORDER_RULES = [
    "Reorder a SKU when the units on display plus its backroom units are at or below its reorder point.",
    "A SKU is reordered once displayed stock + backroom stock <= reorder point.",
    "Order more of an item if what is on the {unit} and in the backroom together does not exceed its reorder point.",
    "Place an order for any SKU whose on-display count plus backroom count is no more than its reorder point.",
    "Replenishment rule: total of visible units and backroom units at or under the reorder point means reorder.",
    "If (units you can see on the {unit}) + (backroom units) is less than or equal to the reorder point, reorder.",
]
Q_LOCATE = [
    "Which {rnoun} holds the {desc}?",
    "On which {rnoun} of the {unit} is the {desc}?",
    "Where on the {unit} is the {desc}: which {rnoun}?",
    "The {desc} is on which {rnoun}?",
    "Which {rnoun} would you go to for the {desc}?",
    "Find the {desc}. Which {rnoun} is it on?",
]
Q_COMPARE = [
    "Which of these products has the most units on display?",
    "Of the listed products, which one has the largest number of units on the {unit}?",
    "Which product is stocked most heavily on the {unit}?",
    "Which item has more units showing than any of the others listed?",
    "Among these, which product has the highest unit count on display?",
    "Which listed product is there the most of on the {unit}?",
]
Q_AUDIT = [
    "How many slots on {region} break the planogram (empty, wrong product, or price not matching the file)?",
    "Audit {region}: how many slots fail (empty, holding the wrong product, or wrong tag price)?",
    "On {region}, count the slots that are empty, hold stock of another SKU, or show a price different from the file.",
    "How many slots on {region} would the compliance audit flag (gap, misplaced stock, or tag price off the file)?",
    "Score {region}: number of slots that are not compliant with the planogram and price file.",
    "What is the number of non-compliant slots on {region} (empty / wrong item / price mismatch)?",
]
Q_HIDDEN = [
    ("How many {plural} are stored behind the front row on {region}?", "count"),
    ("Are the {plural} on {region} still within their sell-by date?", "noul"),
    ("Are any of the {plural} on {region} damaged on the side facing the wall?", "noul"),
    ("How many {plural} are waiting in the backroom?", "count"),
    ("Were the {plural} on {region} restocked this morning?", "noul"),
    ("Is the stock of {plural} on {region} from the latest delivery batch?", "noul"),
]
AUDIT_RULES = [
    "A slot fails when its space is empty, when it holds a product other than its label's SKU, or when its tag price is not the file price.",
    "Non-compliant slot = gap, or stock that belongs to a different SKU, or a tag price that differs from the price file.",
    "Flag a slot if nothing is in it, if the wrong item sits in it, or if its printed price is off the file.",
    "Compliance: every slot must be stocked with its own SKU and carry the file price; anything else counts as one failed slot.",
    "Each slot is checked for three faults (empty, wrong product, price off file); a slot with any fault counts once.",
    "Count a slot as failing on any of: no stock, another SKU's stock, label price different from the file.",
]


# ------------------------------------------------------------------------------------------------ products
OTHER_TRAIN = [ROOT / "data" / "p3" / "candidates" / n for n in ("I-joint.jsonl", "C-images.jsonl")]


def other_train_abo() -> tuple[set, set]:
    """ABO listing ids and photo hashes that other training candidates already use; held-out scenes never use them."""
    import re
    items, shas = set(), set()
    for p in OTHER_TRAIN:
        if not p.is_file():
            continue
        for line in open(p):
            r = json.loads(line)
            pv = r.get("provenance") or {}
            ups = pv.get("upstream_ids") or []
            ups = ups if isinstance(ups, list) else [ups]
            for u in ups + [pv.get("upstream_id") or ""]:
                items |= set(re.findall(r"\b[A-Z0-9]{10}\b", str(u)))
            for im in r.get("images") or []:
                shas.add(Path(im).stem)
    return items, shas


def item_bucket(item: str) -> int:
    return int(hashlib.sha256(("inv-split\0" + item).encode()).hexdigest()[:8], 16) % HELDOUT_PRODUCT_MOD


_PRODUCTS = None


def load_products() -> list[dict]:
    """ABO listing photos (train partition, never an ImajevBench hash) that cut out cleanly, as scene products."""
    global _PRODUCTS
    if _PRODUCTS is not None:
        return _PRODUCTS
    abo = abo_photos()
    used_items, used_shas = other_train_abo()
    out = []
    for p in abo:
        f = p["facts"]
        t = f["product_type"]
        if t not in SHELF_TYPES or f.get("colour") in {"white", "ivory", "off-white", "silver", "transparent"}:
            continue
        if f.get("colour") == "multicoloured":
            f = dict(f, colour=None)
        c = cutout(p["image"], p["sha256"])
        if c is None:
            continue
        g, sing, plur, countable = SHELF_TYPES[t]
        colour = f.get("colour")
        desc = f"{colour} {sing}" if colour else sing
        out.append({"pid": f"abo:{p['item']}", "kind": "abo", "item": p["item"], "image": p["image"], "sha256": p["sha256"],
                    "type": t, "group": g, "noun": sing, "plural": plur, "colour": colour, "desc": desc,
                    "plural_desc": f"{colour} {plur}" if colour else plur, "countable": countable and not is_multi(c),
                    "aspect": round(c.size[0] / c.size[1], 3),
                    "split": ("heldout" if p["item"] not in used_items and p["sha256"] not in used_shas else "none")
                    if item_bucket(p["item"]) == 0 else "train"})
    _PRODUCTS = out
    return out


def abo_photos() -> list[dict]:
    """One photo per ABO listing whose records are all in the train partition (the gen_image_joint rule, without its
    per-type cap and attribute minimum), never an ImajevBench image or an excluded hash."""
    import gen_image_joint as J
    meta = json.loads(J.ABO_META.read_text())
    if J.ABO_META_V2.is_file():
        meta.update(json.loads(J.ABO_META_V2.read_text()))
    bench = J.bench_image_shas()
    banned = set(json.loads(J.EXCLUSIONS.read_text()).get("sha256", [])) if J.EXCLUSIONS.is_file() else set()
    by_item, non_train = {}, set()
    for line in open(J.ABO_RECORDS):
        row = json.loads(line)
        m = J.ABO_ID.match(row["id"].split(":", 1)[1])
        if not m:
            continue
        item = m.group(1)
        if row["partition"] != "train":
            non_train.add(item)
            continue
        by_item.setdefault(item, row["images"][0])
    out = []
    for item, image in sorted(by_item.items()):
        if item in non_train or item not in meta or image["sha256"] in banned or image["sha256"] in bench:
            continue
        if not (ROOT / image["image"]).is_file():
            continue
        a = meta[item]["attrs"]
        out.append({"item": item, "image": J.data_rel(image["image"]), "sha256": image["sha256"],
                    "facts": {"product_type": a.get("product_type"), "colour": a.get("color")}})
    return out


def abo_image(p: dict) -> Image.Image:
    return cutout(p["image"], p["sha256"])


# procedurally drawn drinks (cooler)
DRINK_COLOURS = {"red": (200, 32, 38), "blue": (30, 80, 190), "green": (30, 150, 70), "yellow": (240, 200, 30),
                 "orange": (240, 120, 20), "purple": (120, 50, 160), "black": (25, 25, 28), "white": (236, 236, 232),
                 "pink": (235, 110, 170), "brown": (120, 70, 35)}
DRINK_BRANDS = ["Kiro", "Vela", "Quorn", "Tamba", "Brio", "Odessa", "Zeffa", "Lumo", "Sorba", "Fenni", "Marra", "Tovo",
                "Wexo", "Ulmi", "Raska", "Pelo", "Nibo", "Corra", "Desso", "Gaula"]
DRINK_KINDS = ["cola", "lemonade", "sparkling water", "iced tea", "orange soda", "energy drink", "ginger ale", "tonic",
               "mango juice", "cold brew"]
CONTAINERS = {"can": ("can", "cans"), "bottle": ("bottle", "bottles")}


def drink_products(rng, n, split, exclude=()) -> list[dict]:
    """Up to n drinks with pairwise different colours (so a colour names one product)."""
    out = []
    cols = [c for c in DRINK_COLOURS if c not in exclude]
    rng.shuffle(cols)
    brands = DRINK_BRANDS[:]
    rng.shuffle(brands)
    for i, colour in enumerate(cols[:n]):
        cont = rng.choice(list(CONTAINERS))
        brand, kind = brands[i], rng.choice(DRINK_KINDS)
        size = "330 ml" if cont == "can" else rng.choice(["500 ml", "750 ml"])
        out.append({"pid": f"drink:{brand}-{kind}-{colour}-{cont}".replace(" ", "_"), "kind": "drink", "brand": brand,
                    "flavour": kind, "container": cont, "colour": colour, "size": size, "countable": True,
                    "noun": f"{brand} {kind} {cont}", "plural": f"{brand} {kind} {CONTAINERS[cont][1]}",
                    "desc": f"{brand} {kind} {size} {cont} ({colour})", "plural_desc": f"{brand} {kind} {CONTAINERS[cont][1]}",
                    "group": f"drink-{colour}", "aspect": 0.42 if cont == "can" else 0.3, "split": split})
    return out


# ------------------------------------------------------------------------------------------------ drawing primitives
def lighten(c, k):
    return tuple(int(min(255, x + (255 - x) * k)) for x in c)


def darken(c, k):
    return tuple(int(x * (1 - k)) for x in c)


def draw_barcode(d, box, rng, fg=(20, 20, 20)):
    x0, y0, x1, y1 = box
    x = x0
    while x < x1 - 1:
        w = rng.choice((1, 1, 1, 2, 2, 3))
        if rng.random() < 0.55:
            d.rectangle([x, y0, min(x1, x + w - 1), y1], fill=fg)
        x += w


def draw_drink(size, prod, rng) -> Image.Image:
    """An RGBA drawing of a can or bottle in the product's colour, with a brand band."""
    w, h = size
    S = 3
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = DRINK_COLOURS[prod["colour"]]
    W_, H_ = w * S, h * S
    if prod["container"] == "can":
        d.rounded_rectangle([0, int(H_ * 0.04), W_ - 1, H_ - 1], radius=int(W_ * 0.12), fill=col)
        d.rectangle([int(W_ * 0.06), 0, int(W_ * 0.94), int(H_ * 0.07)], fill=(185, 188, 192))
        d.rectangle([int(W_ * 0.02), H_ - int(H_ * 0.05), int(W_ * 0.98), H_ - 1], fill=(170, 172, 176))
        band = (int(H_ * 0.38), int(H_ * 0.62))
    else:
        neck_w = int(W_ * 0.34)
        cx = W_ // 2
        d.rectangle([cx - neck_w // 2 - 2, 0, cx + neck_w // 2 + 2, int(H_ * 0.06)], fill=darken(col, 0.35))
        d.rectangle([cx - neck_w // 2, int(H_ * 0.05), cx + neck_w // 2, int(H_ * 0.3)], fill=col)
        d.polygon([(cx - neck_w // 2, int(H_ * 0.28)), (cx + neck_w // 2, int(H_ * 0.28)), (W_ - 1, int(H_ * 0.42)),
                   (0, int(H_ * 0.42))], fill=col)
        d.rounded_rectangle([0, int(H_ * 0.40), W_ - 1, H_ - 1], radius=int(W_ * 0.15), fill=col)
        band = (int(H_ * 0.55), int(H_ * 0.78))
    band_col = {"white": (200, 30, 30), "yellow": (30, 30, 30)}.get(prod["colour"], (250, 250, 248))
    d.rectangle([0, band[0], W_ - 1, band[1]], fill=band_col)
    txt_col = {"white": (255, 255, 255), "yellow": (250, 210, 40)}.get(prod["colour"], col)
    f = fit_font(d, prod["brand"].upper(), "impact", W_ * 0.86, (band[1] - band[0]) * 0.8, start=60)
    text_center(d, W_ / 2, (band[0] + band[1]) / 2, prod["brand"].upper(), f, txt_col)
    # highlight stripe
    hl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).rectangle([int(W_ * 0.18), int(H_ * 0.1), int(W_ * 0.3), H_ - int(H_ * 0.06)], fill=(255, 255, 255, 60))
    im = Image.alpha_composite(im, hl)
    return im.resize((w, h), Image.LANCZOS)


def draw_label(d, box, lab, style, rng, show_price=True, loc=None):
    """Shelf-edge label: short name, SKU (bold), price (large), barcode stripes."""
    x0, y0, x1, y1 = box
    bgc = style.get("label_bg", (252, 252, 248))
    d.rectangle(box, fill=bgc, outline=(90, 90, 90))
    w, h = x1 - x0, y1 - y0
    ink = (20, 20, 20)
    left_w = w * (0.56 if show_price else 0.95)
    line1 = lab.get("short", "")
    top = loc if loc else line1
    f1 = fit_font(d, top, "narrow", left_w - 6, h * 0.3, start=int(h * 0.3))
    d.text((x0 + 4, y0 + 2), top, font=f1, fill=(60, 60, 60) if not loc else ink)
    f2 = fit_font(d, f"SKU {lab['sku']}", "narrow_bold", left_w - 6, h * 0.36, start=int(h * 0.38))
    d.text((x0 + 4, y0 + h * 0.34), f"SKU {lab['sku']}", font=f2, fill=ink)
    draw_barcode(d, (x0 + 4, int(y0 + h * 0.76), int(x0 + left_w * 0.85), y1 - 3), rng)
    if show_price:
        pr = f"{lab['price']:.2f}"
        f3 = fit_font(d, pr, "din", w - left_w - 6, h * 0.62, start=int(h * 0.62))
        text_center(d, x0 + left_w + (w - left_w) / 2, y0 + h / 2, pr, f3, ink)


COLOUR_CODE = {"black": "BLK", "blue": "BLU", "red": "RED", "pink": "PNK", "grey": "GRY", "brown": "BRN", "green": "GRN",
               "beige": "BGE", "gold": "GLD", "yellow": "YEL", "purple": "PUR", "orange": "ORG", "navy blue": "NVY",
               "tan": "TAN", "bronze": "BRZ", "turquoise": "TRQ", "burgundy": "BUR", "light blue": "LBL"}


def short_name(prod) -> str:
    if prod["kind"] == "drink":
        return f"{prod['brand']} {prod['flavour']}".upper()[:18]
    n = prod["type"].upper()
    c = COLOUR_CODE.get(prod.get("colour") or "", "")
    return f"{n[:15]} {c}".strip()


# ------------------------------------------------------------------------------------------------ scene layout
ROW_NAMES = {2: ["top", "bottom"], 3: ["top", "middle", "bottom"], 4: ["top", "second", "third", "bottom"],
             5: ["top", "second", "third", "fourth", "bottom"]}


def region_name(style, r, R):
    if style == "bin_wall":
        return f"row {chr(65 + r)}"
    if style == "pegboard":
        return ["top row", "bottom row"][r] if R == 2 else "row"
    if style == "pallet_rack":
        return f"level {R - r}"          # level 1 = floor
    nm = ROW_NAMES[R][r]
    return {"top": "top shelf", "bottom": "bottom shelf", "middle": "middle shelf"}.get(nm, f"{nm} shelf from the top")


def region_options(style, R):
    return [region_name(style, r, R) for r in range(R)]


def slot_loc(style, aisle, r, c, R):
    if style == "pallet_rack":
        return f"{aisle:02d}-{c + 1:02d}-{R - r}"
    if style == "bin_wall":
        return f"{chr(65 + r)}{c + 1}"
    return None


def make_layout(rng, style, W, H, split):
    """Rows, columns, slot capacity and the geometry of every slot (canvas coordinates)."""
    mx, my = int(W * 0.06), int(H * 0.06)
    if style == "retail_shelf":
        R, C = rng.choice([(3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (3, 2), (4, 2), (5, 2)])
    elif style == "cooler":
        R, C = rng.choice([(4, 4), (4, 3), (5, 3), (3, 4), (4, 2), (5, 4)])
    elif style == "pallet_rack":
        R, C = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    elif style == "bin_wall":
        R, C = rng.choice([(3, 3), (3, 4), (4, 4), (4, 3), (3, 5)])
    else:
        R, C = rng.choice([(2, 4), (2, 5), (2, 3), (1, 5), (1, 6)])
    x0, x1 = mx + int(W * 0.03), W - mx - int(W * 0.03)
    y0, y1 = my + int(H * 0.08), H - my
    rh = (y1 - y0) / R
    cw = (x1 - x0) / C
    rail = max(40, min(52, int(rh * 0.22)))
    slots = []
    for r in range(R):
        for c in range(C):
            sx0, sx1 = x0 + c * cw, x0 + (c + 1) * cw
            ry0, ry1 = y0 + r * rh, y0 + (r + 1) * rh
            if style in ("retail_shelf", "cooler"):
                item = (sx0 + 6, ry0 + 10, sx1 - 6, ry1 - rail - 8)
                lw = min(cw * 0.78, 170 if style == "retail_shelf" else 140)
                label = (sx0 + 8, ry1 - rail + 3, sx0 + 8 + lw, ry1 - 4)
            elif style == "pallet_rack":
                beam = max(40, int(rh * 0.14))
                item = (sx0 + 14, ry0 + 12, sx1 - 14, ry1 - beam - 2)
                lw = min(cw * 0.5, 170)
                label = ((sx0 + sx1) / 2 - lw / 2, ry1 - beam + 4, (sx0 + sx1) / 2 + lw / 2, ry1 - 5)
            elif style == "bin_wall":
                item = (sx0 + 10, ry0 + 12, sx1 - 10, ry1 - 12)
                lw = min(cw * 0.7, 130)
                lh = max(34, int(rh * 0.22))
                label = ((sx0 + sx1) / 2 - lw / 2, ry1 - lh - 10, (sx0 + sx1) / 2 + lw / 2, ry1 - 12)
            else:
                lw = min(cw * 0.72, 130)
                lh = max(36, int(rh * 0.16))
                label = ((sx0 + sx1) / 2 - lw / 2, ry0 + 20, (sx0 + sx1) / 2 + lw / 2, ry0 + 20 + lh)
                item = (sx0 + 10, ry0 + 20 + lh + 12, sx1 - 10, ry1 - 10)
            slots.append({"id": f"r{r}c{c}", "row": r, "col": c, "item_box": [round(v) for v in item],
                          "label_box": [round(v) for v in label]})
    ib = slots[0]["item_box"]
    iw, ih = ib[2] - ib[0], ib[3] - ib[1]
    if style == "retail_shelf":
        cap = max(2, min(6, int(iw / (ih * 0.7))))
    elif style == "cooler":
        cap = max(2, min(6, int(iw / (ih * 0.36))))
    elif style == "pallet_rack":
        cap = 6 if iw / ih > 1.1 else 4
    elif style == "bin_wall":
        cap = max(1, min(3, int(iw / (ih * 0.5))))
    else:
        cap = max(2, min(4, int(ih / max(60, iw * 0.55))))
    return {"R": R, "C": C, "cap": cap, "bounds": [x0, y0, x1, y1], "rail": rail, "slots": slots}


# ------------------------------------------------------------------------------------------------ scene content
COOLER_EXCLUDE = [{"black", "brown"}, {"white"}, {"black", "blue"}]      # low contrast against each cooler palette


def choose_products(rng, style, n, split, palette=0):
    if style == "cooler":
        return drink_products(rng, n, split, exclude=COOLER_EXCLUDE[palette % 3])
    pool = [p for p in load_products() if p["split"] == split]
    if style == "pegboard":
        pool = [p for p in pool if p["group"] in PEG_GROUPS]
    elif style in ("retail_shelf", "bin_wall"):
        pool = [p for p in pool if p["group"] not in BIG_GROUPS]
    by_group = {}
    for p in pool:
        by_group.setdefault(p["group"], []).append(p)
    groups = sorted(by_group)
    # phone cases dominate ABO; give every group the same chance
    rng.shuffle(groups)
    out = []
    for g in groups[:n]:
        out.append(rng.choice(sorted(by_group[g], key=lambda p: p["pid"])))
    return out


def build_scene(rng, style, split, want: dict) -> dict:
    """Scene spec: layout, products, labels, what sits in every slot, occluder / crop, post-processing params."""
    W = rng.choice([1024, 1152, 1280, 1280])
    H = rng.choice([720, 768, 800, 864, 900, 960])
    lay = make_layout(rng, style, W, H, split)
    slots = lay["slots"]
    n_slots = len(slots)
    palette = rng.randrange(1000)
    prods = choose_products(rng, style, n_slots + 1, split, palette=palette)
    if len(prods) < min(n_slots, 6) + 1:
        return None
    # the last product is the stray one (can show up as misplaced stock); the others fill the slots, cycling when there
    # are fewer products than slots (a SKU may then have two labelled spaces, as in real planograms)
    extra, base = prods[-1], prods[:-1]
    store, aisle, bay = rng.randint(101, 989), rng.randint(1, 38), rng.randint(1, 12)
    used_skus = set()

    def new_sku():
        while True:
            s = str(rng.randint(10000, 99999)) if style != "pallet_rack" else f"{rng.randint(100000, 999999)}"
            if s not in used_skus:
                used_skus.add(s)
                return s
    products = {}
    slot_prod = {}
    order_ = list(range(n_slots))
    rng.shuffle(order_)
    for i, s in zip(order_, slots):
        slot_prod[s["id"]] = base[i % len(base)]
    # one SKU per product; a product in two slots keeps one SKU (planogram lists it once)
    sku_of = {}
    for s in slots:
        p = slot_prod[s["id"]]
        if p["pid"] not in sku_of:
            sku_of[p["pid"]] = new_sku()
            products[p["pid"]] = p
    if extra["pid"] not in products:
        sku_of[extra["pid"]] = new_sku()
        products[extra["pid"]] = extra        # the stray product that can appear as misplaced stock
    price_file = {}
    for pid, sku in sku_of.items():
        price_file[sku] = round(rng.choice([rng.uniform(1.5, 9.9), rng.uniform(10, 49), rng.uniform(50, 180)]), 2)
        price_file[sku] = math.floor(price_file[sku]) + rng.choice([0.99, 0.49, 0.29, 0.79, 0.0, 0.5])
    cap = lay["cap"]
    content = {}
    for s in slots:
        p = slot_prod[s["id"]]
        n = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
        content[s["id"]] = {"sku": sku_of[p["pid"]], "price": price_file[sku_of[p["pid"]]], "pid": p["pid"],
                            "n": min(n, cap), "short": short_name(p)}
    ids = [s["id"] for s in slots]
    # anomalies: gaps, misplaced stock, wrong tag prices (0-2 each, chosen by the question sampler through `want`)
    n_gap = want.get("gaps", rng.choice([0, 1, 1, 2]))
    n_mis = want.get("misplaced", rng.choice([0, 0, 1, 1]))
    n_price = want.get("price", rng.choice([0, 1, 1, 2])) if style not in ("pallet_rack", "bin_wall") else 0
    free = ids[:]
    rng.shuffle(free)
    for sid in free[:n_gap]:
        content[sid]["n"] = 0
    rest = free[n_gap:]
    if n_mis and rest:
        for sid in rest[:n_mis]:
            content[sid]["pid"] = extra["pid"] if rng.random() < 0.5 or len(rest) < 3 else content[rest[-1]]["pid"]
            if content[sid]["pid"] == products_pid_of(content, sid, sku_of):
                content[sid]["pid"] = extra["pid"]
            content[sid]["n"] = max(1, min(content[sid]["n"], 3))
    rng.shuffle(ids)
    for sid in ids[:n_price]:
        c = content[sid]
        delta = rng.choice([1.0, 2.0, -1.0, 0.5, 5.0, 10.0, -0.5])
        c["price"] = round(max(0.49, c["price"] + delta), 2)
        if c["price"] == price_file[c["sku"]]:
            c["price"] = round(c["price"] + 1.0, 2)
    spec = {"style": style, "W": W, "H": H, "R": lay["R"], "C": lay["C"], "cap": cap, "rail": lay["rail"],
            "bounds": lay["bounds"], "store": store, "aisle": aisle, "bay": bay, "split": split,
            "slots": [dict(s, **content[s["id"]], loc=slot_loc(style, aisle, s["row"], s["col"], lay["R"])) for s in slots],
            "products": {pid: public_product(p) for pid, p in products.items()},
            "planogram": {sku: pid for pid, sku in sku_of.items()}, "price_file": price_file,
            "palette": palette, "post": affine_params(rng), "occluder": None, "crop": None}
    return spec


def products_pid_of(content, sid, sku_of):
    inv = {v: k for k, v in sku_of.items()}
    return inv[content[sid]["sku"]]


def public_product(p) -> dict:
    keep = ("pid", "kind", "item", "image", "sha256", "type", "group", "noun", "plural", "colour", "desc", "plural_desc",
            "countable", "brand", "flavour", "container", "size", "aspect", "split")
    return {k: p[k] for k in keep if k in p}


# ------------------------------------------------------------------------------------------------ occlusion / crop
def add_occluder(rng, spec, target_col=None, target_row=None):
    """A floor-standing (cart / box stack / pillar) or hanging (banner) occluder over part of one column."""
    slots = spec["slots"]
    C, R = spec["C"], spec["R"]
    col = target_col if target_col is not None else rng.randrange(C)
    row = target_row if target_row is not None else rng.randrange(R)
    col_slots = [s for s in slots if s["col"] == col]
    x_lo = min(s["item_box"][0] for s in col_slots)
    x_hi = max(s["item_box"][2] for s in col_slots)
    w = x_hi - x_lo
    frac = rng.uniform(0.55, 0.95)
    ox0 = x_lo + rng.uniform(0, 1 - frac) * w
    ox1 = ox0 + frac * w
    tgt = next(s for s in col_slots if s["row"] == row)
    ib = tgt["item_box"]
    kind = rng.choice(["cart", "boxes", "pillar", "banner"]) if spec["style"] != "pegboard" else rng.choice(["banner", "boxes"])
    if kind == "banner":
        oy0 = spec["bounds"][1] - 30
        oy1 = ib[1] + (ib[3] - ib[1]) * rng.uniform(0.55, 0.95)
    elif kind == "pillar":
        oy0, oy1 = 0, spec["H"]
        ox1 = ox0 + max(60, min(frac * w, w * 0.6))
    else:
        oy0 = ib[1] + (ib[3] - ib[1]) * rng.uniform(0.05, 0.45)
        oy1 = spec["H"]
    spec["occluder"] = {"kind": kind, "box": [round(ox0), round(oy0), round(ox1), round(oy1)],
                        "text": rng.choice(["NEW LINE", "SALE", "2 FOR 1", "SPECIAL OFFER", "BACK TO SCHOOL", "PRICE DROP",
                                            "CLEARANCE", "SEASONAL"]),
                        "colour": rng.choice([(200, 40, 40), (30, 90, 180), (240, 180, 0), (30, 140, 90), (120, 60, 160)])}


def add_crop(rng, spec):
    """Crop the output through one outer column (or the bottom row) so part of the unit is out of frame."""
    W, H = spec["W"], spec["H"]
    side = rng.choice(["right", "left", "bottom"])
    if side in ("right", "left"):
        col = spec["C"] - 1 if side == "right" else 0
        s = next(s for s in spec["slots"] if s["col"] == col)
        x0, x1 = s["item_box"][0], s["item_box"][2]
        cut = x0 + (x1 - x0) * rng.uniform(0.3, 0.7)
        spec["crop"] = [0, 0, round(cut), H] if side == "right" else [round(cut), 0, W, H]
    else:
        s = next(s for s in spec["slots"] if s["row"] == spec["R"] - 1)
        y0, y1 = s["item_box"][1], s["item_box"][3]
        cut = y0 + (y1 - y0) * rng.uniform(0.3, 0.6)
        spec["crop"] = [0, 0, W, round(cut)]


def overlap(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def inside(box, win, tol=1.0) -> bool:
    return box[0] >= win[0] - tol and box[1] >= win[1] - tol and box[2] <= win[2] + tol and box[3] <= win[3] + tol


def compute_hidden(spec, M=None):
    """items_hidden / label_hidden per slot, from the occluder (canvas coords) and the crop (output coords)."""
    from inventory_assets.common import forward_matrix
    if M is None:
        M = forward_matrix((spec["W"], spec["H"]), spec["post"])
    win = spec["crop"] or [0, 0, spec["W"], spec["H"]]
    win = [win[0] + 2, win[1] + 2, win[2] - 2, win[3] - 2]
    for s in spec["slots"]:
        ih = lh = False
        oc = spec["occluder"]
        if oc and overlap(oc["box"], s["item_box"]) > 0:
            ih = True
        if oc and overlap(oc["box"], s["label_box"]) > 0:
            lh = True
        if not inside(map_box(M, s["item_box"]), win):
            ih = True
        if not inside(map_box(M, s["label_box"]), win):
            lh = True
        s["items_hidden"], s["label_hidden"] = ih, lh


# ------------------------------------------------------------------------------------------------ rendering
PALETTES = {
    "retail_shelf": [((236, 232, 224), (70, 74, 80), (215, 218, 222)), ((222, 230, 236), (40, 44, 52), (232, 234, 236)),
                     ((244, 239, 230), (120, 86, 52), (201, 170, 128)), ((60, 64, 72), (30, 30, 34), (200, 204, 210)),
                     ((230, 226, 240), (90, 90, 100), (245, 245, 245))],
    "cooler": [((28, 30, 36), (55, 58, 66), (200, 205, 212)), ((238, 238, 240), (150, 152, 158), (210, 214, 220)),
               ((20, 60, 110), (40, 44, 52), (190, 196, 204))],
    "pallet_rack": [((196, 198, 200), (230, 110, 20), (30, 70, 150)), ((180, 176, 170), (240, 150, 20), (40, 90, 60)),
                    ((210, 212, 206), (220, 60, 40), (50, 60, 80))],
    "bin_wall": [((222, 224, 226), (40, 90, 170), (120, 124, 130)), ((236, 232, 222), (220, 60, 50), (90, 90, 96)),
                 ((200, 206, 212), (240, 180, 20), (70, 74, 80)), ((226, 230, 226), (40, 140, 80), (100, 104, 110))],
    "pegboard": [((196, 160, 116), (120, 120, 126), (80, 80, 84)), ((230, 230, 232), (150, 150, 156), (70, 70, 74)),
                 ((60, 62, 66), (170, 170, 176), (220, 220, 224))],
}


def render_scene(spec) -> Image.Image:
    style = spec["style"]
    W, H = spec["W"], spec["H"]
    rng = rng_for("render", json.dumps(spec["slots"], sort_keys=True)[:4000], spec["palette"])
    pal = PALETTES[style][spec["palette"] % len(PALETTES[style])]
    img = Image.new("RGB", (W, H), pal[0])
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = spec["bounds"]
    R, C = spec["R"], spec["C"]
    rh = (y1 - y0) / R
    # --- background and structure
    if style == "retail_shelf":
        wall, frame, board = pal
        d.rectangle([0, 0, W, H], fill=wall)
        d.rectangle([x0 - 22, y0 - 40, x1 + 22, H], fill=lighten(frame, 0.75))
        for yy in range(int(y0 - 30), H, 16):           # back panel perforation
            for xx in range(int(x0 - 10), int(x1 + 10), 16):
                d.point((xx, yy), fill=lighten(frame, 0.55))
        d.rectangle([x0 - 26, y0 - 40, x0 - 12, H], fill=frame)
        d.rectangle([x1 + 12, y0 - 40, x1 + 26, H], fill=frame)
        hdr = f"AISLE {spec['aisle']}"
        f = font("din", 26)
        d.rectangle([x0 + (x1 - x0) * 0.35, y0 - 78, x0 + (x1 - x0) * 0.65, y0 - 44], fill=frame)
        text_center(d, (x0 + x1) / 2, y0 - 61, hdr, f, (250, 250, 250))
        for r in range(R):
            yb = y0 + (r + 1) * rh
            d.rectangle([x0 - 12, yb - spec["rail"] - 6, x1 + 12, yb - spec["rail"]], fill=darken(board, 0.15))
            d.rectangle([x0 - 12, yb - spec["rail"], x1 + 12, yb], fill=board)
            d.line([x0 - 12, yb, x1 + 12, yb], fill=darken(board, 0.4), width=2)
    elif style == "cooler":
        body, frame, rack = pal
        d.rectangle([0, 0, W, H], fill=(72, 74, 80))
        d.rectangle([x0 - 30, y0 - 60, x1 + 30, H - 6], fill=frame)
        d.rectangle([x0 - 14, y0 - 16, x1 + 14, y1 + 6], fill=body)
        f = font("din", 24)
        text_center(d, (x0 + x1) / 2, y0 - 38, "COLD DRINKS", f, (240, 240, 240))
        for r in range(R):
            yb = y0 + (r + 1) * rh
            d.rectangle([x0 - 14, yb - spec["rail"], x1 + 14, yb], fill=rack)
            d.line([x0 - 14, yb - spec["rail"], x1 + 14, yb - spec["rail"]], fill=darken(rack, 0.3), width=2)
    elif style == "pallet_rack":
        floor, beam, upr = pal
        d.rectangle([0, 0, W, H], fill=(150, 152, 156))
        d.rectangle([0, int(H * 0.9), W, H], fill=floor)
        cw = (x1 - x0) / C
        for r in range(R):
            yb = y0 + (r + 1) * rh
            bh = max(40, int(rh * 0.14))
            d.rectangle([x0 - 10, yb - bh, x1 + 10, yb - bh + 10], fill=darken(beam, 0.2))
            d.rectangle([x0 - 10, yb - bh, x1 + 10, yb], fill=beam)
        for c in range(C + 1):
            xx = x0 + c * cw
            d.rectangle([xx - 11, y0 - 30, xx + 11, H], fill=upr)
            for yy in range(int(y0 - 20), H, 22):
                d.rectangle([xx - 3, yy, xx + 3, yy + 8], fill=darken(upr, 0.45))
    elif style == "bin_wall":
        wall, bincol, frame = pal
        d.rectangle([0, 0, W, H], fill=wall)
        d.rectangle([x0 - 16, y0 - 16, x1 + 16, y1 + 12], fill=lighten(frame, 0.6))
        for r in range(R + 1):
            yy = y0 + r * rh
            d.rectangle([x0 - 16, yy - 6, x1 + 16, yy], fill=frame)
    else:
        board, hook, lab = pal
        d.rectangle([0, 0, W, H], fill=(235, 235, 235) if board[0] < 100 else (80, 82, 88))
        d.rectangle([x0 - 20, y0 - 30, x1 + 20, y1 + 10], fill=board)
        for yy in range(int(y0 - 20), int(y1), 18):
            for xx in range(int(x0 - 10), int(x1 + 10), 18):
                d.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], fill=darken(board, 0.45))
    # --- items
    for s in spec["slots"]:
        draw_slot(img, d, spec, s, pal, rng)
    # --- labels
    for s in spec["slots"]:
        lab = {"sku": s["sku"], "price": s["price"], "short": s["short"]}
        if style == "pegboard":
            lb = s["label_box"]
            cx = (lb[0] + lb[2]) / 2
            d.line([cx, lb[1] - 16, cx, lb[1]], fill=pal[1], width=3)
            d.ellipse([cx - 5, lb[1] - 20, cx + 5, lb[1] - 10], fill=pal[1])
        draw_label(d, s["label_box"], lab, {"label_bg": (252, 250, 240) if style != "cooler" else (250, 246, 200)}, rng,
                   show_price=style not in ("pallet_rack", "bin_wall"), loc=s.get("loc"))
    # --- occluder
    oc = spec["occluder"]
    if oc:
        draw_occluder(img, oc, spec, rng)
    return img


def draw_slot(img, d, spec, s, pal, rng):
    style = spec["style"]
    ib = s["item_box"]
    n = s["n"]
    cap = spec["cap"]
    prod = spec["products"][s["pid"]]
    bx0, by0, bx1, by1 = ib
    if style == "bin_wall":
        # the bin: a coloured box with a lower front lip; items stand inside, visible above the lip
        bincol = pal[1]
        lip_top = by1 - (by1 - by0) * 0.34
        d.rectangle([bx0 + 6, by0 + (by1 - by0) * 0.25, bx1 - 6, by1], fill=lighten(darken(bincol, 0.25), 0.35))
        d.polygon([(bx0 - 4, lip_top), (bx1 + 4, lip_top), (bx1, by1), (bx0, by1)], fill=bincol)
    if n == 0:
        return
    if style == "pallet_rack":
        draw_cartons(img, d, spec, s, prod, rng)
        return
    if style == "pegboard":
        slot_h = (by1 - by0) / cap
        for k in range(n):
            box = (bx0 + 6, by0 + k * slot_h + 3, bx1 - 6, by0 + (k + 1) * slot_h - 3)
            paste_item(img, abo_image(prod), box, shadow=False, anchor="center")
        return
    if style == "bin_wall":
        lip_top = by1 - (by1 - by0) * 0.34
        wslot = (bx1 - bx0 - 8) / cap
        for k in range(n):
            sc = min(1.0, TYPE_SCALE.get(prod["type"], 0.8) + 0.25)
            bot = lip_top + (by1 - lip_top) * 0.12
            top = by0 + (bot - by0) * (1 - sc) + 6
            box = (bx0 + 4 + k * wslot + 2, top, bx0 + 4 + (k + 1) * wslot - 2, bot)
            paste_item(img, abo_image(prod), box, shadow=False)
        bincol = pal[1]
        d.polygon([(bx0 - 4, lip_top), (bx1 + 4, lip_top), (bx1, by1), (bx0, by1)], fill=bincol)
        d.line([(bx0 - 4, lip_top), (bx1 + 4, lip_top)], fill=lighten(bincol, 0.3), width=3)
        return
    wslot = (bx1 - bx0) / cap
    for k in range(n):
        box = (bx0 + k * wslot + 3, by0 + (by1 - by0) * 0.08, bx0 + (k + 1) * wslot - 3, by1)
        bw, bh = box[2] - box[0], box[3] - box[1]
        if prod["kind"] == "drink":
            ih = bh * (0.97 if prod["container"] == "bottle" else 0.58)
            iw = min(bw * 0.9, ih * (0.3 if prod["container"] == "bottle" else 0.56))
            ih = iw / (0.3 if prod["container"] == "bottle" else 0.56)
            dim = draw_drink((max(8, int(iw)), max(8, int(ih))), prod, rng)
            paste_item(img, dim, (box[0], box[3] - ih, box[2], box[3]), shadow=True)
        else:
            sc = TYPE_SCALE.get(prod["type"], 0.8)
            paste_item(img, abo_image(prod), (box[0], box[3] - bh * sc, box[2], box[3]), shadow=True,
                       flip=(k % 2 == 1 and prod["group"] not in ("phonecase",)))


def draw_cartons(img, d, spec, s, prod, rng):
    bx0, by0, bx1, by1 = s["item_box"]
    cap = spec["cap"]
    cols = 3 if cap >= 6 else 2
    rows = math.ceil(cap / cols)
    cw, ch = (bx1 - bx0) / cols, (by1 - by0) / rows
    kraft = [(196, 160, 112), (184, 146, 98), (205, 172, 125), (175, 140, 95)][spec["palette"] % 4]
    sku = s["sku"]
    for k in range(s["n"]):
        r, c = divmod(k, cols)
        x0 = bx0 + c * cw + 3
        y1 = by1 - r * ch
        y0 = y1 - ch + 4
        x1 = x0 + cw - 6
        d.rectangle([x0, y0, x1, y1], fill=kraft, outline=darken(kraft, 0.35), width=2)
        d.rectangle([x0 + (x1 - x0) * 0.46, y0, x0 + (x1 - x0) * 0.54, y1], fill=lighten(kraft, 0.18))
        lw, lh = (x1 - x0) * 0.62, (y1 - y0) * 0.5
        lx0, ly0 = x0 + (x1 - x0 - lw) / 2, y0 + (y1 - y0) * 0.36
        d.rectangle([lx0, ly0, lx0 + lw, ly0 + lh], fill=(250, 250, 250), outline=(80, 80, 80))
        f = fit_font(d, sku, "narrow_bold", lw * 0.62, lh * 0.45, start=int(lh * 0.45))
        d.text((lx0 + 3, ly0 + 2), sku, font=f, fill=(15, 15, 15))
        draw_barcode(d, (int(lx0 + 3), int(ly0 + lh * 0.62), int(lx0 + lw * 0.6), int(ly0 + lh - 3)), rng)
        pim = abo_image(spec["products"][s["pid"]])
        paste_item(img, pim, (lx0 + lw * 0.64, ly0 + 3, lx0 + lw - 3, ly0 + lh - 3), shadow=False, anchor="center")


def draw_occluder(img, oc, spec, rng):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = oc["box"]
    if oc["kind"] == "cart":
        d.rectangle([x0, y0, x1, y1 - 30], fill=(150, 154, 160), outline=(60, 62, 66), width=3)
        for k in range(3):
            yy = y0 + (y1 - 30 - y0) * (k + 1) / 4
            d.line([x0, yy, x1, yy], fill=(80, 82, 86), width=3)
        bw = (x1 - x0) / 2
        for k in range(2):
            d.rectangle([x0 + 6 + k * bw, y0 + 8, x0 + (k + 1) * bw - 6, y0 + (y1 - y0) * 0.3], fill=(190, 150, 100),
                        outline=(120, 90, 60), width=2)
        for xx in (x0 + 14, x1 - 14):
            d.ellipse([xx - 12, y1 - 28, xx + 12, y1 - 4], fill=(30, 30, 30))
    elif oc["kind"] == "boxes":
        n = max(2, int((y1 - y0) / 110))
        bh = (y1 - y0) / n
        for k in range(n):
            sh = rng.uniform(-8, 8)
            d.rectangle([x0 + sh, y0 + k * bh, x1 + sh * 0.5, y0 + (k + 1) * bh], fill=(192, 152, 104),
                        outline=(110, 82, 54), width=3)
            d.rectangle([x0 + sh + (x1 - x0) * 0.45, y0 + k * bh, x0 + sh + (x1 - x0) * 0.55, y0 + (k + 1) * bh],
                        fill=(210, 180, 140))
    elif oc["kind"] == "pillar":
        d.rectangle([x0, y0, x1, y1], fill=(176, 176, 172))
        d.rectangle([x0, y0, x0 + (x1 - x0) * 0.25, y1], fill=(196, 196, 192))
        d.rectangle([x1 - (x1 - x0) * 0.15, y0, x1, y1], fill=(150, 150, 146))
        d.rectangle([x0, y1 - 120, x1, y1 - 70], fill=(230, 190, 20))
    else:
        col = tuple(oc["colour"])
        d.rectangle([x0, y0, x1, y1], fill=col, outline=darken(col, 0.4), width=3)
        f = fit_font(d, oc["text"], "impact", (x1 - x0) * 0.85, min(80, (y1 - y0) * 0.3), start=80)
        text_center(d, (x0 + x1) / 2, (y0 + y1) / 2, oc["text"], f, (255, 255, 255))


def render_final(spec) -> Image.Image:
    img = render_scene(spec)
    out, M = apply_post(img, spec["post"], f"inv-noise-{spec['palette']}-{spec['store']}", fill=(60, 60, 60))
    if spec["crop"]:
        out = out.crop(tuple(spec["crop"]))
    return out


# ------------------------------------------------------------------------------------------------ gold derivation
def world_of(spec) -> list[dict]:
    return [{"id": s["id"], "row": s["row"], "col": s["col"], "sku": s["sku"], "price": s["price"], "pid": s["pid"],
             "n": s["n"]} for s in spec["slots"]]


def alt_worlds(spec) -> list[list[dict]]:
    """Single-slot alternatives for every hidden slot: what could be behind the occluder / beyond the crop."""
    base = world_of(spec)
    pids = sorted(spec["products"])
    skus = sorted(spec["planogram"])
    out = []
    for i, s in enumerate(spec["slots"]):
        if s["items_hidden"]:
            alts = [{"n": 0}, {"n": s["n"] + 1}, {"n": max(1, s["n"] - 1)}]
            alts += [{"pid": p, "n": max(1, s["n"])} for p in pids if p != s["pid"]]
            for a in alts:
                w = copy.deepcopy(base)
                w[i].update(a)
                out.append(w)
        if s["label_hidden"]:
            alts = [{"price": round(spec["price_file"][s["sku"]] + 1.0, 2)}, {"price": spec["price_file"][s["sku"]]}]
            alts += [{"sku": k} for k in skus if k != s["sku"]]
            for a in alts:
                w = copy.deepcopy(base)
                w[i].update(a)
                out.append(w)
    return out


def answer(spec, world, q):
    """The answer to question q in one world; None = the question is ill-posed in that world (a tie, two answers)."""
    k = q["kind"]
    plan = spec["planogram"]
    if k in ("count", "colour_count", "total"):
        reg = q.get("region")
        tot = 0
        for s in world:
            if reg is not None and s["row"] != reg:
                continue
            if q.get("slot") is not None and s["id"] != q["slot"]:
                continue
            if k == "colour_count":
                if spec["products"][s["pid"]]["colour"] == q["colour"]:
                    tot += s["n"]
            elif q.get("slot") is not None or s["pid"] == q["pid"]:
                tot += s["n"]
        return tot
    if k == "oos_which":
        xs = sorted({s["sku"] for s in world if s["n"] == 0})
        return "NONE" if not xs else (xs[0] if len(xs) == 1 else None)
    if k == "oos_noul":
        return any(s["sku"] == q["sku"] and s["n"] == 0 for s in world)
    if k == "misplaced":
        xs = sorted({s["sku"] for s in world if s["n"] > 0 and s["pid"] != plan[s["sku"]]})
        return "NONE" if not xs else (xs[0] if len(xs) == 1 else None)
    if k == "price":
        xs = sorted({s["sku"] for s in world if abs(s["price"] - spec["price_file"][s["sku"]]) > 1e-6})
        return "NONE" if not xs else (xs[0] if len(xs) == 1 else None)
    if k in ("reorder_which", "reorder_noul"):
        stock = q["stock"]
        need = {}
        for sku, st in stock.items():
            shown = sum(s["n"] for s in world if s["pid"] == plan[sku])
            need[sku] = shown + st["backroom"] <= st["reorder_point"]
        if k == "reorder_noul":
            return need[q["sku"]]
        xs = sorted(x for x, v in need.items() if v)
        return "NONE" if not xs else (xs[0] if len(xs) == 1 else None)
    if k == "locate":
        rows = sorted({s["row"] for s in world if s["pid"] == q["pid"] and s["n"] > 0})
        return "NONE" if not rows else (rows[0] if len(rows) == 1 else None)
    if k == "compare":
        tot = {p: sum(s["n"] for s in world if s["pid"] == p) for p in q["pids"]}
        best = max(tot.values())
        xs = [p for p, v in tot.items() if v == best]
        return xs[0] if len(xs) == 1 else None
    if k == "audit":
        bad = 0
        for s in world:
            if s["row"] != q["region"]:
                continue
            if s["n"] == 0 or s["pid"] != plan[s["sku"]] or abs(s["price"] - spec["price_file"][s["sku"]]) > 1e-6:
                bad += 1
        return bad
    raise KeyError(k)


def has_witness(spec, q) -> bool:
    """True when some alternative content of the hidden part changes the answer (the unknown is undecidable)."""
    truth = answer(spec, world_of(spec), q)
    return any(answer(spec, w, q) != truth for w in alt_worlds(spec))


def settle(spec, q):
    """(gold answer, unknown_reason). Hidden-attribute questions are unknown by construction."""
    if q["kind"] == "hidden":
        return None, "insufficient_evidence"
    truth = answer(spec, world_of(spec), q)
    if truth is None:
        return None, "ill_posed"
    for w in alt_worlds(spec):
        if answer(spec, w, q) != truth:
            return None, "insufficient_evidence"
    if truth == "NONE":
        return None, "false_premise"
    return truth, None


# ------------------------------------------------------------------------------------------------ questions
def ordinal_region(spec, r):
    nm = region_name(spec["style"], r, spec["R"])
    return nm if spec["style"] in ("bin_wall", "pallet_rack") else "the " + nm


def count_options(rng, gold_n, lo=0):
    width = 5
    start = max(lo, gold_n - rng.randint(0, width - 1))
    return [str(v) for v in range(start, start + width)]


def sku_options(spec, with_desc=True):
    opts = []
    for sku in sorted(spec["planogram"], key=lambda s: spec["_sku_order"].index(s)):
        p = spec["products"][spec["planogram"][sku]]
        if sku not in {s["sku"] for s in spec["slots"]}:
            continue
        opts.append((f"sku_{sku}", f"SKU {sku}", p["desc"] if with_desc else ""))
    return opts


TEMPLATE_KINDS = ["count", "total", "colour_count", "oos_which", "oos_noul", "misplaced", "price", "reorder_which",
                  "reorder_noul", "locate", "compare", "audit", "hidden"]
FAMILY = {"count": "inventory_count", "total": "inventory_total", "colour_count": "inventory_colour_count",
          "oos_which": "inventory_out_of_stock", "oos_noul": "inventory_out_of_stock", "misplaced": "inventory_misplaced",
          "price": "inventory_price_tag", "reorder_which": "inventory_reorder", "reorder_noul": "inventory_reorder",
          "locate": "inventory_locate", "compare": "inventory_compare", "audit": "inventory_audit",
          "hidden": "inventory_not_visible"}
BASE_DIFF = {"count": 3, "total": 4, "colour_count": 3, "oos_which": 3, "oos_noul": 2, "misplaced": 4, "price": 3,
             "reorder_which": 5, "reorder_noul": 5, "locate": 2, "compare": 4, "audit": 5, "hidden": 3}
STYLE_KINDS = {
    "retail_shelf": ["count", "total", "oos_which", "oos_noul", "misplaced", "price", "reorder_which", "reorder_noul",
                     "locate", "compare", "audit", "hidden"],
    "cooler": ["count", "colour_count", "total", "oos_which", "oos_noul", "price", "reorder_which", "reorder_noul",
               "locate", "compare", "audit", "hidden"],
    "pallet_rack": ["count", "oos_which", "oos_noul", "misplaced", "reorder_which", "reorder_noul", "hidden"],
    "bin_wall": ["count", "total", "oos_which", "oos_noul", "misplaced", "reorder_noul", "locate", "hidden"],
    "pegboard": ["count", "total", "oos_which", "oos_noul", "price", "reorder_noul", "locate", "compare", "hidden"],
}
KIND_WEIGHT = {"count": 3, "total": 1.5, "colour_count": 2, "oos_which": 1.2, "oos_noul": 0.9, "misplaced": 2.3,
               "price": 2.0, "reorder_which": 0.8, "reorder_noul": 0.8, "locate": 1.5, "compare": 1.2, "audit": 1.2,
               "hidden": 0.45}


def sample_question(rng, spec, kind, split):
    """Question params for `kind` on this scene, or None when the scene cannot carry it."""
    style = spec["style"]
    R = spec["R"]
    prods = spec["products"]
    plan = spec["planogram"]
    slot_pids = sorted({s["pid"] for s in spec["slots"] if s["n"] > 0 or True})
    countable = [p for p in slot_pids if prods[p]["countable"]]
    q = {"kind": kind}
    if kind == "count":
        if style == "pallet_rack":
            s = rng.choice(spec["slots"])
            q.update(slot=s["id"])
            return q
        if not countable:
            return None
        pid = rng.choice(countable)
        rows = sorted({s["row"] for s in spec["slots"] if s["pid"] == pid})
        q.update(pid=pid, region=rng.choice(rows) if rng.random() < 0.85 else rng.randrange(R))
        return q
    if kind == "total":
        if not countable:
            return None
        q.update(pid=rng.choice(countable))
        return q
    if kind == "colour_count":
        cols = sorted({prods[s["pid"]]["colour"] for s in spec["slots"]})
        q.update(colour=rng.choice(cols), region=rng.randrange(R))
        return q
    if kind in ("oos_which", "misplaced", "price"):
        return q
    if kind == "oos_noul":
        s = rng.choice([s for s in spec["slots"] if s["n"] == 0] or spec["slots"]) if rng.random() < 0.5 else rng.choice(spec["slots"])
        q.update(sku=s["sku"])
        return q
    if kind in ("reorder_which", "reorder_noul"):
        labelled = {s["sku"] for s in spec["slots"]}
        skus = sorted(k for k in plan if k in labelled and (prods[plan[k]]["countable"] or style == "pallet_rack"))
        if len(skus) < 2:
            return None
        on_unit = {sku: sum(s["n"] for s in spec["slots"] if s["pid"] == plan[sku]) for sku in skus}
        target = rng.choice(skus)
        stock = {}
        for sku in skus:
            rp = rng.randint(3, 14)
            if kind == "reorder_which":
                need = sku == target
            else:
                need = rng.random() < 0.5 if sku == target else rng.random() < 0.3
            gap = rp - on_unit[sku]
            if need:
                back = max(0, gap - rng.randint(0, 2))
                if on_unit[sku] + back > rp:
                    rp = on_unit[sku] + back
            else:
                back = max(0, gap + rng.randint(1, 3))
                if on_unit[sku] + back <= rp:
                    back = rp - on_unit[sku] + 1
            stock[sku] = {"backroom": back, "reorder_point": rp}
        q.update(stock=stock)
        if kind == "reorder_noul":
            q.update(sku=target)
        return q
    if kind == "locate":
        if R < 2:
            return None
        cands = [p for p in slot_pids if sum(1 for s in spec["slots"] if s["pid"] == p and s["n"] > 0) >= 1]
        if not cands:
            return None
        q.update(pid=rng.choice(cands))
        return q
    if kind == "compare":
        cands = [p for p in countable]
        if len(cands) < 3:
            return None
        q.update(pids=sorted(rng.sample(cands, rng.choice([3, 3, 4])) if len(cands) >= 4 else cands[:3]))
        return q
    if kind == "audit":
        q.update(region=rng.randrange(R))
        return q
    if kind == "hidden":
        pool = countable or slot_pids
        pid = rng.choice(pool)
        rows = sorted({s["row"] for s in spec["slots"] if s["pid"] == pid})
        hk = max(1, len(Q_HIDDEN) // 3)
        hs = range(len(Q_HIDDEN) - hk, len(Q_HIDDEN)) if split == "heldout" else range(len(Q_HIDDEN) - hk)
        q.update(pid=pid, region=rows[0], hidden_idx=rng.choice(list(hs)))
        return q
    raise KeyError(kind)


def gold_key(spec, q, gold):
    """The row's gold (option key / bool / level) for a settled answer; None stays None."""
    if gold is None:
        return None
    k = q["kind"]
    if k in ("count", "total", "colour_count"):
        return f"n_{gold}"
    if k in ("oos_which", "misplaced", "price", "reorder_which"):
        return f"sku_{gold}"
    if k == "locate":
        return option_key(region_options(spec["style"], spec["R"])[gold])
    if k == "compare":
        return f"sku_{next(s for s, p in spec['planogram'].items() if p == gold)}"
    return gold


def render_question(rng, spec, q, gold, split):
    """(state, field, gold key) for the question."""
    style = spec["style"]
    prods = spec["products"]
    plan = spec["planogram"]
    kind_title, kind_lower = KIND_NAMES[style]
    unit = UNIT_NOUN[style]
    rnoun = REGION_NOUN[style]
    t = pick(rng, [f"{h:02d}:{m:02d}" for h in range(6, 22) for m in (0, 10, 20, 30, 40, 50)], "train")
    photo = pick(rng, PHOTO_LINES, split).format(kind=kind_title, kindl=kind_lower, store=spec["store"], aisle=spec["aisle"],
                                                  bay=spec["bay"], time=t)
    order = spec["_sku_order"]
    planogram = [{"sku": sku, "product": prods[plan[sku]]["desc"]} for sku in order]
    state = {"photo": photo, "planogram": planogram}
    k = q["kind"]
    region = ordinal_region(spec, q["region"]) if q.get("region") is not None else None
    fmt = dict(unit=unit, rnoun=rnoun, region=region)
    field, gkey = None, None
    if k in ("count", "total"):
        if q.get("slot"):
            s = next(s for s in spec["slots"] if s["id"] == q["slot"])
            question = pick(rng, Q_COUNT_CARTON, split).format(loc=s["loc"])
        else:
            p = prods[q["pid"]]
            fmt.update(plural=p["plural_desc"])
            question = pick(rng, Q_COUNT if k == "count" else Q_TOTAL, split).format(**fmt)
        n_truth = answer(spec, world_of(spec), q)
        opts = count_options(rng, n_truth)
        field = {"type": "choice", "question": question,
                 "options": [{"key": f"n_{v}", "text": v, "description": ""} for v in opts]}
        gkey = None if gold is None else f"n_{gold}"
    elif k == "colour_count":
        cont = rng.choice(["container", "drink"])
        fmt.update(colour=q["colour"], container=cont)
        question = pick(rng, Q_COLOUR, split).format(**fmt)
        n_truth = answer(spec, world_of(spec), q)
        opts = count_options(rng, n_truth)
        field = {"type": "choice", "question": question,
                 "options": [{"key": f"n_{v}", "text": v, "description": ""} for v in opts]}
        gkey = None if gold is None else f"n_{gold}"
    elif k in ("oos_which", "misplaced", "price", "reorder_which"):
        bank = {"oos_which": Q_OOS, "misplaced": Q_MISPLACED, "price": Q_PRICE, "reorder_which": Q_REORDER}[k]
        question = pick(rng, bank, split).format(**fmt)
        opts = sku_options(spec)
        field = {"type": "choice", "question": question,
                 "options": [{"key": a, "text": b, "description": c} for a, b, c in opts]}
        gkey = None if gold is None else f"sku_{gold}"
        if k == "price":
            state["price_file"] = {sku: f"{spec['price_file'][sku]:.2f}" for sku in order}
        if k == "reorder_which":
            state["rule"] = pick(rng, REORDER_RULES, split).format(unit=unit)
            state["stock_file"] = [{"sku": sku, **q["stock"][sku]} for sku in order if sku in q["stock"]]
    elif k == "oos_noul":
        question = pick(rng, Q_OOS_NOUL, split).format(sku=q["sku"], **fmt)
        field = {"type": "noul", "question": question}
        gkey = gold
    elif k == "reorder_noul":
        question = pick(rng, Q_REORDER_NOUL, split).format(sku=q["sku"], **fmt)
        field = {"type": "noul", "question": question}
        gkey = gold
        state["rule"] = pick(rng, REORDER_RULES, split).format(unit=unit)
        state["stock_file"] = [{"sku": sku, **q["stock"][sku]} for sku in order if sku in q["stock"]]
    elif k == "locate":
        p = prods[q["pid"]]
        fmt.update(desc=p["desc"])
        question = pick(rng, Q_LOCATE, split).format(**fmt)
        regs = region_options(style, spec["R"])
        field = {"type": "choice", "question": question,
                 "options": [{"key": option_key(r_), "text": r_, "description": ""} for r_ in regs]}
        gkey = None if gold is None else option_key(regs[gold])
    elif k == "compare":
        question = pick(rng, Q_COMPARE, split).format(**fmt)
        opts = []
        for pid in q["pids"]:
            sku = next(s for s, p in plan.items() if p == pid)
            opts.append({"key": f"sku_{sku}", "text": prods[pid]["desc"], "description": f"SKU {sku}"})
        field = {"type": "choice", "question": question, "options": opts}
        gkey = None if gold is None else f"sku_{next(s for s, p in plan.items() if p == gold)}"
    elif k == "audit":
        question = pick(rng, Q_AUDIT, split).format(**fmt)
        n = sum(1 for s in spec["slots"] if s["row"] == q["region"])
        field = {"type": "score", "question": question,
                 "levels": [{"value": i, "description": f"{i} failing slot{'s' if i != 1 else ''}"} for i in range(n + 1)]}
        gkey = gold
        state["price_file"] = {sku: f"{spec['price_file'][sku]:.2f}" for sku in order}
        state["rule"] = pick(rng, AUDIT_RULES, split)
    elif k == "hidden":
        p = prods[q["pid"]]
        fmt.update(plural=p["plural_desc"])
        tmpl, typ = Q_HIDDEN[q["hidden_idx"]]
        question = tmpl.format(**fmt)
        if typ == "count":
            opts = [str(v) for v in range(0, 5)]
            field = {"type": "choice", "question": question,
                     "options": [{"key": f"n_{v}", "text": v, "description": ""} for v in opts]}
        else:
            field = {"type": "noul", "question": question}
        gkey = None
    assert gkey == gold_key(spec, q, gold)
    return state, field, gkey


# ------------------------------------------------------------------------------------------------ build
STYLE_MIX = {"retail_shelf": 0.32, "cooler": 0.20, "pallet_rack": 0.16, "bin_wall": 0.16, "pegboard": 0.16}


def scene_questions(rng, spec, split, n_q, want_unknown: bool):
    """Pick up to n_q questions (distinct kinds) for a scene. Returns [(q, gold, reason)]."""
    kinds = STYLE_KINDS[spec["style"]][:]
    out, used = [], set()
    for _ in range(40):
        if len(out) >= n_q:
            break
        ks = [k for k in kinds if k not in used]
        if not ks:
            break
        k = rng.choices(ks, weights=[KIND_WEIGHT[x] for x in ks])[0]
        if k == "hidden" and not want_unknown:
            continue
        q = sample_question(rng, spec, k, split)
        if q is None:
            used.add(k)
            continue
        gold, reason = settle(spec, q)
        if reason == "ill_posed":
            continue
        if (gold is None) != want_unknown and len(out) == 0 and rng.random() < 0.85:
            continue
        if gold is None and not want_unknown:
            continue
        used.add(k)
        out.append((q, gold, reason))
    return out


def make_item(spec, q, gold, reason, img_rel, img_sha, idx, prefix, split, rng, parent=None, variant=None):
    state, field, gkey = render_question(rng, spec, q, gold, split)
    k = q["kind"]
    diff = BASE_DIFF[k]
    if k in ("count", "total", "colour_count") and gold is not None and gold >= 5:
        diff = min(5, diff + 1)
    if spec["occluder"] or spec["crop"]:
        diff = min(5, diff + (1 if gold is None else 0))
    diff = max(2, diff)
    abo = [p for p in spec["products"].values() if p["kind"] == "abo"]
    lic = "CC-BY-4.0" if abo else "generated"
    rid = f"{prefix}-{idx:06d}" if variant is None else f"{parent}-v{variant}"
    return {
        "id": rid, "source": "I", "dataset": "inventory_synthetic", "family": FAMILY[k], "difficulty": diff,
        "state": state, "images": [img_rel], "field": field, "gold": gkey,
        "unknown_reason": None if gkey is not None else reason, "gold_kind": "constructed", "parent_id": parent,
        "provenance": {
            "licence": lic, "image_licences": [lic],
            "licence_evidence": [ABO_EVIDENCE] if abo else [],
            "composite": "our own render (PIL) with ABO product photos cut out of their white background" if abo
                         else "our own render (PIL), procedurally drawn products",
            "photo_attribution": [{"upstream_dataset": "abo", "item_id": p["item"], "image_sha256": p["sha256"],
                                   "licence": "CC-BY-4.0", "credit": ABO_CREDIT} for p in abo],
            "upstream_dataset": "abo" if abo else None,
            "upstream_ids": sorted(p["item"] for p in abo) if abo else None,
            "upstream_split": "train" if abo else "generated", "split": split,
            "image_sha256": [img_sha], "scene_style": spec["style"], "question": q, "spec": spec_public(spec),
            "generator": "scripts/p3/gen_inventory.py", "seed": spec["_seed"],
        },
    }


def spec_public(spec) -> dict:
    return {k: v for k, v in spec.items() if not k.startswith("_") or k in ("_sku_order", "_seed")}


def gen_scene(seed: str, i: int, split: str, want_unknown: bool, style: str | None = None, force_kind: str | None = None):
    rng = rng_for(seed, "scene", i)
    style = style or rng.choices(list(STYLE_MIX), weights=list(STYLE_MIX.values()))[0]
    want = {}
    if force_kind in ("oos_which",):
        want["gaps"] = rng.choice([0, 1, 1, 1])
    for _ in range(12):
        spec = build_scene(rng, style, split, want)
        if spec is not None:
            break
    else:
        return None, None
    spec["_seed"] = f"{seed}/scene/{i}"
    order = sorted(spec["planogram"])
    rng.shuffle(order)
    spec["_sku_order"] = order
    r = rng.random()
    if want_unknown and r < 0.75 or (not want_unknown and r < 0.18):
        if rng.random() < 0.6:
            add_occluder(rng, spec)
        else:
            add_crop(rng, spec)
    compute_hidden(spec)
    return spec, rng


def build(count: int, seed: str, split: str, unknown_share=0.15, per_scene=2, img_dir=None):
    bench = __import__("gen_image_joint").bench_image_shas()
    img_dir = img_dir or (IMG_DIR if split == "train" else IMG_DIR / "heldout")
    prefix = "p3-I-inventory" if split == "train" else "p3-hf1-I-inventory"
    rows, stats = [], Counter()
    i = 0
    n_unknown = 0
    while len(rows) < count and i < count * 6:
        want_unknown = n_unknown < unknown_share * (len(rows) + per_scene)
        spec, rng = gen_scene(seed, i, split, want_unknown)
        i += 1
        if spec is None:
            stats["scene_failed"] += 1
            continue
        qs = scene_questions(rng, spec, split, per_scene, want_unknown)
        if not qs:
            stats["no_question"] += 1
            continue
        img = render_final(spec)
        data = encode_jpeg(img, spec["post"]["jpeg"])
        st = store_image(data, img_dir, bench)
        if st is None:
            stats["bench_sha"] += 1
            continue
        rel, sha = st
        for q, gold, reason in qs:
            if len(rows) >= count:
                break
            row = make_item(spec, q, gold, reason, rel, sha, len(rows), prefix, split, rng)
            n_unknown += row["gold"] is None
            rows.append(row)
        stats["scenes"] += 1
    return rows, stats


def build_variants(parents_path: str, per_parent: int):
    """Constructed variants of existing inventory items: same question kind and scene style, a fresh scene keyed by the
    parent id (the pattern, not the item). Variants inherit the parent's split through parent_id."""
    bench = __import__("gen_image_joint").bench_image_shas()
    rows = []
    for par in (json.loads(l) for l in open(parents_path) if l.strip()):
        if par.get("dataset") != "inventory_synthetic":
            continue
        pv = par["provenance"]
        kind, style = pv["question"]["kind"], pv["scene_style"]
        split = pv.get("split", "train")
        for v in range(per_parent):
            for attempt in range(20):
                spec, rng = gen_scene(f"variant/{par['id']}/{v}", attempt, split, par["gold"] is None, style=style,
                                      force_kind=kind)
                if spec is None:
                    continue
                q = sample_question(rng, spec, kind, split)
                if q is None:
                    continue
                gold, reason = settle(spec, q)
                if reason == "ill_posed" or (gold is None) != (par["gold"] is None):
                    continue
                img = render_final(spec)
                st = store_image(encode_jpeg(img, spec["post"]["jpeg"]),
                                 IMG_DIR if split == "train" else IMG_DIR / "heldout", bench)
                if st is None:
                    continue
                rows.append(make_item(spec, q, gold, reason, st[0], st[1], 0, "", split, rng, parent=par["id"], variant=v))
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--heldout", action="store_true", help="fresh seed, held-out-only products and wordings")
    ap.add_argument("--variant-of", default=None, help="JSONL of parent inventory items to make constructed variants of")
    ap.add_argument("--per-parent", type=int, default=2)
    a = ap.parse_args()
    if a.variant_of:
        rows = build_variants(a.variant_of, a.per_parent)
        out = a.out or str(ROOT / "data" / "p3" / "candidates" / "I-inventory-variants.jsonl")
        print(json.dumps({"rows": write(out, rows), "out": out}))
        return
    split = "heldout" if a.heldout else "train"
    seed = a.seed or (HELDOUT_SEED if a.heldout else SEED)
    out = a.out or str(HELDOUT_OUT if a.heldout else OUT)
    rows, stats = build(a.count, seed, split)
    n = write(out, rows)
    print(json.dumps({"rows": n, "out": out, "stats": dict(stats), "unknown": sum(r["gold"] is None for r in rows),
                      "by_family": dict(Counter(r["family"] for r in rows)),
                      "by_style": dict(Counter(r["provenance"]["scene_style"] for r in rows)),
                      "by_reason": dict(Counter(r["unknown_reason"] for r in rows)),
                      "difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
                      "images": len({r["images"][0] for r in rows})}, indent=1))


if __name__ == "__main__":
    main()
