"""Amazon Berkeley Objects -> decision-v1 records.

ABO is catalogue photography with seller-supplied structured metadata: a product
type, a colour with a controlled `standardized_values` vocabulary, material,
pattern, shape, finish and style, plus a main image and several alternate shots of
the same listing. That last part is what makes ABO the only source in the product
group that can carry two-image same-product questions honestly: two photos of one
`item_id` are the same product by construction, and two photos of two `item_id`s
under one `product_type` are a hard negative by construction.

Design decisions that the spec forces and that are worth stating:

* Only values a photograph can settle are used. `scripts/v1/abo_taxonomy.py` holds
  the curated maps; anything not in them is dropped rather than guessed. That
  throws away most of `style` (5,327 free-text values), all of `fabric_type`
  (fibre percentages are not visible), and the department-style product types
  (HOME, GROCERY, ACCESSORY ...) that name a shelf rather than a thing.
* A listing with two different values for an attribute is dropped, not resolved.
* Options are built per example from the curated pool, hard distractors first
  (same coarse group / same product family), with confusable values suppressed so
  exactly one option is defensible.
* `source_group` is the product identity (brand + model, else item_id), so colour
  variants of one model never straddle the split. Two-image records pair only
  within one partition pool, so a reference image cannot leak across the split.
* ABO supports exactly two abstention causes honestly: `not_listed` (the gold is
  removed from the options) and `mismatched_reference` (the target photo shows a
  different product than the reference). `insufficient_evidence` is NOT emitted:
  proving an attribute is unreadable from a given shot needs image inspection this
  converter does not do, and the research doc's warning stands -- a missing
  catalogue field is not the same as not-determinable-from-image.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_abo.py
"""
import csv, gzip, hashlib, io, json, random, re, sys, tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageOps
from vision_decision.contracts import Request
import abo_taxonomy as T

SOURCE = "abo"
SEED = 20260922
CACHE = Path(".cache/datasets/v1/abo")
OUT = Path("data/decision-v1/abo")
IMG_DIR = OUT / "images"
LICENSE = "CC-BY-4.0"
TARGET = 60_000
TWO_IMAGE_SHARE = 0.25
NOT_LISTED_RATE = 0.18
MISMATCH_RATE = 0.40
DEV_FRACTION = 0.03
TEST_GROUP_PERMILLE = 45          # ~4.5% of product groups reserved for `test`
TEST_CAP = 1_500                  # spec cap per source; 12 families -> 125 each
TEST_PER_FAMILY = 125
BIG_OPTION_RATE = 0.10            # share of choice records with 13-25 options
MAX_PER_TYPE = 2_500              # keeps CELLULAR_PHONE_CASE (44% of ABO) in check
MAX_PER_VALUE = 3_000             # per attribute value, same reason

# ---------------------------------------------------------------- question text
CATEGORY_TEMPLATES = [
    "What kind of product is shown in this photo?",
    "Which of these best describes the item in the picture?",
    "This is a product listing photo. What is the product?",
    "Pick the category that fits the object shown.",
    "What is being sold in this image?",
    "Identify the type of product photographed here.",
]
ATTR_TEMPLATES = {
    "color": [
        "What colour is this product?",
        "What is the main colour of the item shown?",
        "Which colour best describes the product in the photo?",
        "Looking at the picture, what colour is this item?",
        "The product above is listed in one colour. Which one?",
        "Choose the colour of the object shown.",
    ],
    "material": [
        "What material is this product made of?",
        "Which material is the item in the photo made from?",
        "What is the main material of the product shown?",
        "Judging from the picture, what is this made of?",
        "Pick the material used for the item above.",
        "The product is made from one main material. Which?",
    ],
    "pattern": [
        "What pattern does this product have?",
        "Which pattern is shown on the item?",
        "Describe the pattern on the product in the photo.",
        "Looking at the picture, what patterning does this item carry?",
        "Pick the pattern that matches the product shown.",
    ],
    "item_shape": [
        "What shape is this product?",
        "Which shape best describes the item shown?",
        "Looking at the photo, what is the shape of this product?",
        "Pick the shape of the object in the picture.",
        "The item above has one overall shape. Which?",
    ],
    "finish": [
        "What finish does this product have?",
        "Which surface finish is shown on the item?",
        "Looking at the picture, how is this product finished?",
        "Pick the finish of the object shown.",
        "The surface of the item above has one finish. Which?",
    ],
    "style": [
        "What style of {family} is this?",
        "Which {family} style is shown in the photo?",
        "Looking at the picture, what kind of {family} is this?",
        "Pick the style that matches the {family} shown.",
        "The {family} above has one recognisable style. Which?",
    ],
}
BOOL_TEMPLATES = {
    "color": ["Is this product {v}?", "Is the item in the photo {v}?",
              "Would you describe the colour of this product as {v}?",
              "Looking at the picture, is this {v}?",
              "Is {v} the colour of the item shown?"],
    "material": ["Is this product made of {v}?", "Is the item in the photo made from {v}?",
                 "Would you say this is made of {v}?",
                 "Looking at the picture, is the material {v}?",
                 "Is {v} what this product is made from?"],
    "pattern": ["Does this product have a {v} pattern?",
                "Is the item in the photo {v}?",
                "Would you describe the patterning here as {v}?",
                "Looking at the picture, is this {v}?",
                "Is the pattern on this product {v}?"],
    "item_shape": ["Is this product {v}?", "Is the item in the photo {v}?",
                   "Would you describe the shape of this product as {v}?",
                   "Looking at the picture, is this {v} in shape?",
                   "Is {v} the shape of the item shown?"],
    "category": ["Is this a {v}?", "Does this photo show a {v}?",
                 "Is the product in the picture a {v}?",
                 "Would you call the item shown a {v}?",
                 "Is a {v} what is being sold here?"],
}
SAME_PRODUCT_TEMPLATES = [
    "Do both photos show the same product?",
    "The first photo is a reference image. Does the second photo show that same product?",
    "Are these two photos of the same product?",
    "Photo 1 is the reference. Is the item in photo 2 the same item?",
    "Is the product in the second photo the same one as in the first?",
    "Compare the two photos: is it one and the same product?",
]
REFERENCE_TEMPLATES = {
    "color": [
        "The first photo is a reference image of a product. Using the second photo, what colour is that product?",
        "Photo 1 shows the reference product. Judging from photo 2, what colour is it?",
        "Take the item in the first photo as the reference. According to the second photo, what colour is it?",
        "The reference product appears in photo 1. From photo 2, what colour does it have?",
        "Image 1 is the reference; image 2 is the photo under review. What colour is the reference product there?",
    ],
    "material": [
        "The first photo is a reference image of a product. Using the second photo, what is that product made of?",
        "Photo 1 shows the reference product. Judging from photo 2, what material is it?",
        "Take the item in the first photo as the reference. According to the second photo, what is it made from?",
        "The reference product appears in photo 1. From photo 2, what material does it use?",
        "Image 1 is the reference; image 2 is the photo under review. What material is the reference product made of?",
    ],
    "category": [
        "The first photo is a reference image of a product. Using the second photo, what kind of product is it?",
        "Photo 1 shows the reference product. Judging from photo 2, what sort of product is it?",
        "Take the item in the first photo as the reference. According to the second photo, what is it?",
        "The reference product appears in photo 1. From photo 2, what type of product is it?",
        "Image 1 is the reference; image 2 is the photo under review. What kind of product is the reference item?",
    ],
}

ATTR_SPEC = {  # family -> (listing key, value map, pool, confusable groups)
    "color": ("color", T.COLOR_MAP, T.COLOR_POOL, T.COLOR_CONFUSABLE),
    "material": ("material", T.MATERIAL_MAP, T.MATERIAL_POOL, T.MATERIAL_CONFUSABLE),
    "pattern": ("pattern", T.PATTERN_MAP, T.PATTERN_POOL, T.PATTERN_CONFUSABLE),
    "item_shape": ("item_shape", T.SHAPE_MAP, T.SHAPE_POOL, T.SHAPE_CONFUSABLE),
    "finish": ("finish_type", T.FINISH_MAP, T.FINISH_POOL, T.FINISH_CONFUSABLE),
    "style": ("style", T.STYLE_MAP, T.STYLE_POOL, T.STYLE_CONFUSABLE),
}
QUOTAS = {"product_category": 18_000, "color": 9_800, "material": 6_200,
          "pattern": 1_200, "item_shape": 900, "finish": 400, "style": 2_000,
          "attribute_boolean": 8_000}
TWO_IMAGE_QUOTAS = {"same_product": 8_000, "reference_color": 3_000,
                    "reference_material": 2_000, "reference_category": 2_000}


def english(entries):
    return [e for e in entries or [] if str(e.get("language_tag", "")).startswith("en")]


def attribute(row, key, vmap):
    """The single curated value for this attribute, or None (absent/ambiguous/unusable)."""
    raw = set()
    for e in english(row.get(key, [])):
        if key == "color":
            raw |= {str(s).strip().lower() for s in (e.get("standardized_values") or [])}
        else:
            raw.add(str(e.get("value", "")).strip().lower())
    mapped = {vmap[v] for v in raw if v in vmap}
    if len(mapped) != 1 or len(raw - set(vmap)) > 0 and len(raw) > 1:
        # one unambiguous curated value, and no competing uncurated value
        return None
    return mapped.pop() if len(mapped) == 1 else None


def confusable_with(value, groups):
    out = set()
    for g in groups:
        if value in g:
            out |= g
    return out


def token_subset(a, b):
    """'shoe' vs 'athletic shoe' — one label contains the other, so both could be right."""
    x, y = set(a.split()), set(b.split())
    return x <= y or y <= x


def make_options(gold, near, far, rng, cause, banned=()):
    """gold + hard-then-easy distractors, count varied, order shuffled."""
    banned = set(banned)

    def ok(v):
        return v != gold and v not in banned and not token_subset(v, gold)

    big = rng.random() < BIG_OPTION_RATE
    total = rng.randint(13, 25) if big else rng.randint(2, 12)
    pool = [v for v in near if ok(v)]
    rng.shuffle(pool)
    rest = [v for v in far if ok(v) and v not in set(pool)]
    rng.shuffle(rest)
    pool += rest
    want = total if cause == "not_listed" else total - 1
    chosen = pool[:want]
    if len(chosen) < (2 if cause == "not_listed" else 1):
        return None
    values = chosen if cause == "not_listed" else chosen + [gold]
    if len(values) < 2:
        return None
    rng.shuffle(values)
    return values


def build_choice(rng, qid, family, question, gold, near, far, source_group,
                 images, partition, cause=None, extra_family=None, banned=()):
    values = make_options(gold, near, far, rng, cause, banned)
    if values is None:
        return None
    field = {"id": "answer", "type": "choice", "question": question,
             "options": [{"value": v} for v in values]}
    request = {"request_id": qid[:128], "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{qid}", "source": SOURCE, "source_split": "all",
            "source_group": source_group, "family": extra_family or family,
            "license": LICENSE, "images": images, "request": request,
            "target": None if cause else gold, "abstention_cause": cause,
            "source_answer": gold, "partition": partition}


def build_boolean(rng, qid, family, question, target, source_group, images,
                  partition, source_answer):
    field = {"id": "answer", "type": "boolean", "question": question}
    request = {"request_id": qid[:128], "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{qid}", "source": SOURCE, "source_split": "all",
            "source_group": source_group, "family": family, "license": LICENSE,
            "images": images, "request": request, "target": target,
            "abstention_cause": None, "source_answer": source_answer,
            "partition": partition}


# ------------------------------------------------------------------ load inputs
def load_listings():
    listings = {}
    with tarfile.open(CACHE / "abo-listings.tar") as tar:
        for member in tar:
            if not member.name.endswith(".json.gz"):
                continue
            for line in gzip.GzipFile(fileobj=tar.extractfile(member)):
                row = json.loads(line)
                item = row["item_id"]
                # one listing per item_id: prefer the marketplace with English metadata
                if item in listings and not english(row.get("item_name", [])):
                    continue
                if item in listings and not english(listings[item].get("item_name", [])):
                    listings[item] = row
                elif item not in listings:
                    listings[item] = row
    return listings


def main():
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    image_meta = {}
    with gzip.open(CACHE / "abo-images.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            image_meta[r["image_id"]] = r["path"]
    listings = load_listings()
    stats["listings"] = len(listings)

    # ---- one record per listing with the facts the converter needs
    products = []
    for item, row in listings.items():
        pt = row.get("product_type", [{}])[0].get("value")
        label_group = T.PRODUCT_TYPES.get(pt)
        images = [i for i in ([row.get("main_image_id")] + list(row.get("other_image_id") or []))
                  if i and i in image_meta]
        if not images:
            continue
        attrs = {f: attribute(row, key, vmap) for f, (key, vmap, _, _) in ATTR_SPEC.items()}
        brand = (english(row.get("brand", [])) or [{}])[0].get("value", "")
        model = (row.get("model_number") or [{}])[0].get("value", "")
        group = f"{pt}|{brand}|{model}".lower() if model else f"item|{item}"
        products.append({"item": item, "pt": pt, "label": label_group[0] if label_group else None,
                         "cgroup": label_group[1] if label_group else None,
                         "images": images, "attrs": attrs, "group": group})
    stats["products"] = len(products)

    # ---- partition pools, by product group so variants never straddle the split
    def pool_of(group):
        h = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16) % 1000
        return "test" if h < TEST_GROUP_PERMILLE else "fit"

    for p in products:
        p["pool"] = pool_of(p["group"])
    rng.shuffle(products)

    # ---- indexes for distractors and for two-image negatives
    by_cgroup = defaultdict(list)
    labels_by_cgroup = defaultdict(set)
    all_labels = sorted({p["label"] for p in products if p["label"]})
    for p in products:
        if p["label"]:
            by_cgroup[p["cgroup"]].append(p)
            labels_by_cgroup[p["cgroup"]].add(p["label"])
    by_type_pool = defaultdict(list)
    for p in products:
        if p["label"]:
            by_type_pool[(p["pt"], p["pool"])].append(p)
    # which materials/patterns actually occur inside a coarse group -> hard distractors
    attr_by_cgroup = {f: defaultdict(Counter) for f in ATTR_SPEC}
    for p in products:
        for f, v in p["attrs"].items():
            if v and p["cgroup"]:
                attr_by_cgroup[f][p["cgroup"]][v] += 1

    used_image = set()      # an image answers at most one question
    ref_uses = Counter()    # a reference image is reused sparingly
    records = []
    family_counts = Counter()
    type_counts = Counter()
    value_counts = Counter()

    def partition_for(pool, rng):
        if pool == "test":
            return "test"
        return "dev" if rng.random() < DEV_FRACTION else "train"

    def take_image(p, idx=None):
        for img in (p["images"] if idx is None else [p["images"][idx]]):
            if img not in used_image:
                used_image.add(img)
                return img
        return None

    # Two-image records are built first: they need two free photos of one listing,
    # and the single-image families would otherwise have eaten the spares.
    two_image(products, by_type_pool, rng, records, family_counts, used_image,
              ref_uses, partition_for, all_labels, labels_by_cgroup, attr_by_cgroup)

    # ---------------------------------------------------------- single image
    order = [f for f in QUOTAS]
    for family in order:
        quota = QUOTAS[family]
        test_quota = min(TEST_PER_FAMILY, quota)
        made = Counter()
        for p in products:
            pool = p["pool"]
            if made["test" if pool == "test" else "fit"] >= (test_quota if pool == "test" else quota - test_quota):
                if made["test"] >= test_quota and made["fit"] >= quota - test_quota:
                    break
                continue
            if family == "product_category":
                if not p["label"] or type_counts[p["pt"]] >= MAX_PER_TYPE:
                    continue
                gold = p["label"]
                near = sorted(labels_by_cgroup[p["cgroup"]] - {gold})
                far = [l for l in all_labels if l != gold and l not in labels_by_cgroup[p["cgroup"]]]
                img = take_image(p)
                if img is None:
                    continue
                cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
                part = partition_for(pool, rng)
                rec = build_choice(rng, f"cat-{p['item']}-{img}", family,
                                   rng.choice(CATEGORY_TEMPLATES), gold, near, far,
                                   p["group"], [img], part, cause,
                                   banned=confusable_with(gold, T.PRODUCT_CONFUSABLE))
                if rec is None:
                    used_image.discard(img); continue
                type_counts[p["pt"]] += 1
            elif family == "attribute_boolean":
                rec, img = make_bool(p, rng, partition_for(pool, rng), take_image, value_counts)
                if rec is None:
                    continue
            else:
                key, vmap, vpool, conf = ATTR_SPEC[family]
                gold = p["attrs"].get(family)
                if not gold or value_counts[(family, gold)] >= MAX_PER_VALUE:
                    continue
                if family == "style":
                    scope = next((s for s in T.STYLE_SCOPE.values() if gold in s), None)
                    if scope is None or p["cgroup"] not in ("footwear", "bags"):
                        continue
                    near = sorted(scope - confusable_with(gold, conf))
                    far = []
                    fam_word = "shoe" if p["cgroup"] == "footwear" else "bag"
                    question = rng.choice(ATTR_TEMPLATES["style"]).format(family=fam_word)
                else:
                    banned = confusable_with(gold, conf)
                    near = [v for v in attr_by_cgroup[family].get(p["cgroup"], {}) if v not in banned]
                    far = [v for v in vpool if v not in banned and v not in set(near)]
                    question = rng.choice(ATTR_TEMPLATES[family])
                img = take_image(p)
                if img is None:
                    continue
                cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
                rec = build_choice(rng, f"{family}-{p['item']}-{img}", family, question,
                                   gold, near, far, p["group"], [img],
                                   partition_for(pool, rng), cause)
                if rec is None:
                    used_image.discard(img); continue
                value_counts[(family, gold)] += 1
            made["test" if pool == "test" else "fit"] += 1
            records.append(rec)
            family_counts[family] += 1
        stats[f"family:{family}"] = family_counts[family]

    # ---------------------------------------------------------- materialise images
    wanted = {im for r in records for im in r["_images"]} if records and "_images" in records[0] else \
             {im for r in records for im in r["images"]}
    written = write_images(wanted, image_meta)
    final = []
    for r in records:
        imgs = []
        ok = True
        for im in r["images"]:
            info = written.get(im)
            if info is None:
                ok = False; break
            imgs.append(info)
        if not ok:
            stats["dropped_missing_image"] += 1
            continue
        r["images"] = imgs
        final.append(r)

    final.sort(key=lambda r: r["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in final))
    boolean = [r for r in final if r["request"]["fields"][0]["type"] == "boolean"]
    summary = {"records": len(final), "seed": SEED,
               "partitions": dict(Counter(r["partition"] for r in final)),
               "families": dict(Counter(r["family"] for r in final)),
               "abstention": dict(Counter(str(r["abstention_cause"]) for r in final)),
               "abstention_share": round(sum(r["target"] is None for r in final) / max(1, len(final)), 4),
               "boolean_balance": dict(Counter(str(r["target"]) for r in boolean)),
               "boolean_balance_by_family": {f: dict(Counter(str(r["target"]) for r in boolean if r["family"] == f))
                                             for f in sorted({r["family"] for r in boolean})},
               "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in final)),
               "option_counts": dict(sorted(Counter(len(r["request"]["fields"][0].get("options", []))
                                                    for r in final if "options" in r["request"]["fields"][0]).items())),
               "two_image": sum(len(r["images"]) == 2 for r in final),
               "images": len(written), "stats": dict(stats)}
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def make_bool(p, rng, partition, take_image, value_counts):
    """A balanced yes/no attribute question; 'no' only where metadata proves the
    negative (the listing carries exactly one curated value for that attribute)."""
    choices = [f for f in ("color", "material", "pattern", "item_shape") if p["attrs"].get(f)]
    if p["label"]:
        choices.append("category")
    if not choices:
        return None, None
    family = rng.choice(choices)
    truth = rng.random() < 0.5
    if family == "category":
        gold = p["label"]
        if truth:
            value = gold
        else:
            banned = confusable_with(gold, T.PRODUCT_CONFUSABLE)
            pool = [l for l, _ in T.PRODUCT_TYPES.values()
                    if l != gold and l not in banned and not token_subset(l, gold)]
            if not pool:
                return None, None
            value = rng.choice(pool)
    else:
        _, vmap, vpool, conf = ATTR_SPEC[family]
        gold = p["attrs"][family]
        if truth:
            value = gold
        else:
            banned = confusable_with(gold, conf)
            options = [v for v in vpool if v not in banned and not token_subset(v, gold)]
            if not options:
                return None, None
            value = rng.choice(options)
    img = take_image(p)
    if img is None:
        return None, None
    question = rng.choice(BOOL_TEMPLATES[family]).format(v=value)
    rec = build_boolean(rng, f"bool-{family}-{p['item']}-{img}", "attribute_boolean",
                        question, truth, p["group"], [img], partition, f"{family}={gold}")
    return rec, img


def two_image(products, by_type_pool, rng, records, family_counts, used_image,
              ref_uses, partition_for, all_labels, labels_by_cgroup, attr_by_cgroup):
    by_pool = defaultdict(list)
    for p in products:
        if p["label"]:
            by_pool[p["pool"]].append(p)

    def pick_negative(p, hard):
        """A different product, same pool. hard = same product_type."""
        for _ in range(25):
            if hard:
                cand = by_type_pool.get((p["pt"], p["pool"]))
                if not cand or len(cand) < 2:
                    return None
                q = cand[rng.randrange(len(cand))]
            else:
                cand = by_pool[p["pool"]]
                q = cand[rng.randrange(len(cand))]
                if q["pt"] == p["pt"]:
                    continue
            if q["group"] != p["group"]:
                return q
        return None

    def reference_image(p):
        """A photo of p that is not being consumed as an answer image."""
        opts = [i for i in p["images"] if ref_uses[i] < 3]
        return opts[0] if opts else None

    # --- same-product boolean, balanced true/false, negatives 60% hard
    quota = TWO_IMAGE_QUOTAS["same_product"]
    test_quota = min(TEST_PER_FAMILY, quota)
    made = Counter()
    for p in products:
        if made["test"] + made["fit"] >= quota:
            break
        if not p["label"]:
            continue
        pool = p["pool"]
        bucket = "test" if pool == "test" else "fit"
        if made[bucket] >= (test_quota if bucket == "test" else quota - test_quota):
            continue
        positive = (made["pos"] <= made["neg"])
        ref = reference_image(p)
        if ref is None:
            continue
        if positive:
            target_src = p
        else:
            target_src = pick_negative(p, rng.random() < 0.6)
            if target_src is None:
                continue
        tgt = next((i for i in target_src["images"] if i not in used_image and i != ref), None)
        if tgt is None:
            continue
        used_image.add(tgt); ref_uses[ref] += 1
        rec = build_boolean(rng, f"same-{p['item']}-{tgt}", "same_product",
                            rng.choice(SAME_PRODUCT_TEMPLATES), positive, p["group"],
                            [ref, tgt], partition_for(pool, rng),
                            "same_item_id" if positive else "different_item_id")
        records.append(rec); family_counts["same_product"] += 1
        made[bucket] += 1; made["pos" if positive else "neg"] += 1

    # --- reference-conditioned attribute questions, with mismatched_reference
    for family, attr in (("reference_color", "color"), ("reference_material", "material"),
                         ("reference_category", "category")):
        quota = TWO_IMAGE_QUOTAS[family]
        test_quota = min(TEST_PER_FAMILY, quota)
        made = Counter()
        for p in products:
            if made["test"] + made["fit"] >= quota:
                break
            gold = p["label"] if attr == "category" else p["attrs"].get(attr)
            if not gold:
                continue
            pool = p["pool"]
            bucket = "test" if pool == "test" else "fit"
            if made[bucket] >= (test_quota if bucket == "test" else quota - test_quota):
                continue
            ref = reference_image(p)
            if ref is None:
                continue
            mismatch = rng.random() < MISMATCH_RATE
            if mismatch:
                other = pick_negative(p, rng.random() < 0.6)
                if other is None:
                    continue
                tgt = next((i for i in other["images"] if i not in used_image and i != ref), None)
            else:
                tgt = next((i for i in p["images"] if i not in used_image and i != ref), None)
            if tgt is None:
                continue
            if attr == "category":
                near = sorted(labels_by_cgroup[p["cgroup"]] - {gold})
                far = [l for l in all_labels if l != gold and l not in labels_by_cgroup[p["cgroup"]]]
                banned = confusable_with(gold, T.PRODUCT_CONFUSABLE)
            else:
                _, _, vpool, conf = ATTR_SPEC[attr]
                banned = confusable_with(gold, conf)
                near = [v for v in attr_by_cgroup[attr].get(p["cgroup"], {}) if v not in banned]
                far = [v for v in vpool if v not in banned and v not in set(near)]
            cause = "mismatched_reference" if mismatch else (
                "not_listed" if rng.random() < NOT_LISTED_RATE else None)
            rec = build_choice(rng, f"{family}-{p['item']}-{tgt}", family,
                               rng.choice(REFERENCE_TEMPLATES[attr]), gold, near, far,
                               p["group"], [ref, tgt], partition_for(pool, rng), cause,
                               extra_family=family, banned=banned)
            if rec is None:
                continue
            used_image.add(tgt); ref_uses[ref] += 1
            records.append(rec); family_counts[family] += 1
            made[bucket] += 1


def write_images(wanted, image_meta):
    """Stream the small-image tar once, writing content-addressed RGB JPEGs."""
    paths = {}
    for img in wanted:
        p = image_meta.get(img)
        if p:
            paths.setdefault(f"images/small/{p}", []).append(img)
    written = {}
    with tarfile.open(CACHE / "abo-images-small.tar") as tar:
        for member in tar:
            ids = paths.get(member.name)
            if not ids:
                continue
            data = tar.extractfile(member).read()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    if im.width * im.height > 1_000_000:
                        s = (1_000_000 / (im.width * im.height)) ** 0.5
                        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                                       Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=90)
                    blob = buf.getvalue()
                    w, h = im.size
            except Exception:
                continue
            sha = hashlib.sha256(blob).hexdigest()
            dest = IMG_DIR / f"{sha}.jpg"
            if not dest.exists():
                dest.write_bytes(blob)
            for i in ids:
                written[i] = {"image": str(dest), "sha256": sha, "width": w, "height": h}
    return written


if __name__ == "__main__":
    main()
