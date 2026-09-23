"""Fashion200K (via `Marqo/fashion200k`) -> decision-v1 records.

Two label sources, deliberately kept apart:

* `category1` (5 coarse classes) and `category2` (31 classes) — the dataset's own
  clean taxonomy, used directly for apparel category choice questions.
* `category3` — the *original* Fashion200K product title ("green seamed a-line
  dress"). Fashion200K was built by mining attribute words out of exactly these
  titles, so a single colour / fabric / pattern / length / silhouette / neckline
  word in the title is a usable attribute label. A curated lexicon per slot does
  the extraction; a title with zero or more than one word in a slot is skipped for
  that slot rather than guessed.

The `text` column is NOT used anywhere. It is a model-generated caption added by
Marqo, not human annotation, and the spec's licence/provenance discipline treats a
machine label as evidence of nothing.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_fashion200k.py
"""
import hashlib, io, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from vision_decision.contracts import Request

SOURCE = "fashion200k"
SEED = 20260922
CACHE = Path(".cache/datasets/v1/fashion200k")
OUT = Path("data/decision-v1/fashion200k")
IMG_DIR = OUT / "images"
LICENSE = "Apache-2.0 (annotations); image rights unclear (crawled from Lyst)"
NOT_LISTED_RATE = 0.18
DEV_FRACTION = 0.03
TEST_GROUP_PERMILLE = 60
TEST_PER_FAMILY = 150          # 10 families -> the spec's 1,500-per-source cap
BIG_OPTION_RATE = 0.10

# --- attribute lexicons, one canonical value per surface form -----------------
COLOR = {w: w for w in ["black", "white", "blue", "red", "green", "pink", "purple",
                        "yellow", "orange", "brown", "beige", "gray", "gold",
                        "silver", "navy", "burgundy", "ivory", "khaki", "coral",
                        "teal", "turquoise", "mint", "lavender", "cream", "tan",
                        "olive", "maroon", "charcoal", "nude", "multicolor"]}
COLOR.update({"grey": "gray", "multicolour": "multicolor", "multi-color": "multicolor",
              "off-white": "ivory", "camel": "tan", "wine": "burgundy"})
COLOR_CONFUSABLE = [{"white", "ivory", "cream", "beige", "nude", "tan", "khaki"},
                    {"gray", "silver", "charcoal"}, {"blue", "navy", "teal", "turquoise"},
                    {"red", "burgundy", "maroon", "coral"}, {"green", "olive", "mint"},
                    {"purple", "lavender"}, {"gold"}, {"multicolor"}]

FABRIC = {w: w for w in ["lace", "silk", "satin", "chiffon", "denim", "leather",
                         "velvet", "tulle", "cotton", "linen", "wool", "cashmere",
                         "jersey", "crepe", "tweed", "corduroy", "suede", "mesh",
                         "sequin", "jacquard", "faux-fur", "knit"]}
FABRIC.update({"sequined": "sequin", "knitted": "knit", "fur": "faux-fur"})
FABRIC_CONFUSABLE = [{"silk", "satin", "chiffon", "crepe"},
                     {"cotton", "linen", "jersey", "knit"},
                     {"wool", "cashmere", "tweed"},
                     {"leather", "suede"}, {"lace", "mesh", "tulle"}]

PATTERN = {w: w for w in ["floral", "striped", "plaid", "polka-dot", "paisley",
                          "checked", "camouflage", "geometric", "animal-print",
                          "leopard", "snakeskin", "tie-dye", "colorblock",
                          "embroidered", "beaded", "metallic", "printed", "solid"]}
PATTERN.update({"stripe": "striped", "stripes": "striped", "checkered": "checked",
                "polka": "polka-dot", "color-block": "colorblock",
                "colour-block": "colorblock", "print": "printed"})
PATTERN_CONFUSABLE = [{"printed", "floral", "paisley", "geometric", "tie-dye"},
                      {"striped", "plaid", "checked"},
                      {"animal-print", "leopard", "snakeskin"},
                      {"embroidered", "beaded", "metallic"},
                      {"solid", "colorblock"}]

LENGTH = {"maxi": "maxi", "midi": "midi", "mini": "mini", "knee-length": "knee-length",
          "floor-length": "maxi", "ankle-length": "maxi", "tea-length": "midi"}
LENGTH_CONFUSABLE = [{"maxi", "midi"}, {"mini", "knee-length"}]

SILHOUETTE = {w: w for w in ["a-line", "sheath", "shift", "bodycon", "wrap",
                             "pencil", "skater", "swing", "shirtdress", "peplum",
                             "trapeze", "empire", "ballgown", "mermaid", "tunic"]}
SILHOUETTE_CONFUSABLE = [{"sheath", "shift", "bodycon", "pencil"},
                         {"a-line", "skater", "swing", "trapeze"},
                         {"ballgown", "mermaid", "empire"}]

NECKLINE = {w: w for w in ["v-neck", "halter", "strapless", "off-the-shoulder",
                           "boat-neck", "scoop-neck", "crew-neck", "turtleneck",
                           "cowl-neck", "square-neck", "one-shoulder", "sweetheart"]}
NECKLINE.update({"boatneck": "boat-neck", "off-shoulder": "off-the-shoulder"})
NECKLINE_CONFUSABLE = [{"scoop-neck", "crew-neck", "boat-neck", "square-neck"},
                       {"strapless", "sweetheart"}, {"turtleneck", "cowl-neck"}]

SLEEVE = {"sleeveless": "sleeveless", "long-sleeve": "long sleeve",
          "long-sleeved": "long sleeve", "short-sleeve": "short sleeve",
          "short-sleeved": "short sleeve", "cap-sleeve": "cap sleeve",
          "three-quarter": "three-quarter sleeve", "puff-sleeve": "puff sleeve",
          "bell-sleeve": "bell sleeve"}
SLEEVE_CONFUSABLE = [{"short sleeve", "cap sleeve", "three-quarter sleeve"},
                     {"puff sleeve", "bell sleeve"}]

SLOTS = {
    "color": (COLOR, COLOR_CONFUSABLE),
    "fabric": (FABRIC, FABRIC_CONFUSABLE),
    "pattern": (PATTERN, PATTERN_CONFUSABLE),
    "length": (LENGTH, LENGTH_CONFUSABLE),
    "silhouette": (SILHOUETTE, SILHOUETTE_CONFUSABLE),
    "neckline": (NECKLINE, NECKLINE_CONFUSABLE),
    "sleeve": (SLEEVE, SLEEVE_CONFUSABLE),
}

TEMPLATES = {
    "apparel_category": [
        "What kind of garment is shown in this photo?",
        "Which category does this clothing item belong to?",
        "This is a fashion catalogue photo. What is the item?",
        "Pick the category that fits the garment shown.",
        "What sort of clothing is being modelled here?",
        "Identify the type of apparel in the picture.",
    ],
    "apparel_supercategory": [
        "Which broad clothing category does this item fall into?",
        "At the coarsest level, what kind of clothing is this?",
        "Is this photo of a dress, a jacket, or something else? Pick one.",
        "Choose the general apparel group for the item shown.",
        "What family of garment is pictured?",
    ],
    "color": [
        "What colour is this garment?",
        "What is the main colour of the item shown?",
        "Which colour best describes the clothing in the photo?",
        "Looking at the picture, what colour is this piece?",
        "Pick the colour of the garment above.",
    ],
    "fabric": [
        "What fabric is this garment made of?",
        "Which material does the item in the photo appear to be?",
        "Looking at the picture, what is this piece made from?",
        "Pick the fabric used for the garment shown.",
        "The item above is made from one main fabric. Which?",
    ],
    "pattern": [
        "What pattern does this garment carry?",
        "Which pattern is shown on the clothing?",
        "Describe the patterning on the item in the photo.",
        "Looking at the picture, how is this piece patterned?",
        "Pick the pattern that matches the garment shown.",
    ],
    "length": [
        "How long is this garment?",
        "Which length best describes the item shown?",
        "Looking at the photo, what length is this piece cut to?",
        "Pick the hem length of the garment above.",
        "The item above is cut to one length. Which?",
    ],
    "silhouette": [
        "What silhouette does this garment have?",
        "Which cut is shown in the photo?",
        "Looking at the picture, what shape is this piece cut to?",
        "Pick the silhouette of the garment shown.",
        "The item above has one recognisable cut. Which?",
    ],
    "neckline": [
        "What neckline does this garment have?",
        "Which neckline is shown in the photo?",
        "Looking at the picture, how is the neck of this piece cut?",
        "Pick the neckline of the garment shown.",
        "The item above has one neckline. Which?",
    ],
    "sleeve": [
        "What sleeves does this garment have?",
        "Which sleeve style is shown in the photo?",
        "Looking at the picture, how are the sleeves cut?",
        "Pick the sleeve style of the garment shown.",
        "The item above has one sleeve style. Which?",
    ],
}
BOOL_TEMPLATES = {
    "color": ["Is this garment {v}?", "Is the item in the photo {v}?",
              "Would you describe the colour of this piece as {v}?",
              "Looking at the picture, is this {v}?",
              "Is {v} the colour of the garment shown?"],
    "fabric": ["Is this garment made of {v}?", "Is the item in the photo {v}?",
               "Would you say this piece is {v}?",
               "Looking at the picture, is the fabric {v}?",
               "Is {v} what this garment is made from?"],
    "pattern": ["Does this garment have a {v} pattern?",
                "Is the item in the photo {v}?",
                "Would you describe the patterning here as {v}?",
                "Looking at the picture, is this piece {v}?",
                "Is the pattern on this garment {v}?"],
    "length": ["Is this garment {v} length?", "Is the item in the photo {v}?",
               "Would you call the length of this piece {v}?",
               "Looking at the picture, is this cut {v}?",
               "Is {v} the length of the garment shown?"],
    "silhouette": ["Is this garment a {v} cut?", "Is the item in the photo {v}?",
                   "Would you describe the silhouette here as {v}?",
                   "Looking at the picture, is this a {v} shape?",
                   "Is the cut of this garment {v}?"],
    # category2 labels are plural noun phrases ("cocktail dresses"), so the
    # phrasings wrap them rather than putting them behind an article.
    "category": ["Does this item belong to the category '{v}'?",
                 "Is this garment listed under '{v}'?",
                 "Would you file the item shown under '{v}'?",
                 "Is '{v}' the right category for this photo?",
                 "Looking at the picture, is this one of the '{v}'?"],
}
# category2 pairs a catalogue photo does not reliably separate
CATEGORY_CONFUSABLE = [
    {"cocktail dresses", "prom and formal dresses", "gowns",
     "casual and day dresses"},
    {"maxi and long dresses", "gowns"},
    {"mini and short dresses", "cocktail dresses"},
    {"blazers and suit jackets", "casual jackets"},
    {"padded and down jackets", "casual jackets"},
]

QUOTAS = {"apparel_category": 7_000, "apparel_supercategory": 2_000, "color": 4_000,
          "fabric": 2_000, "pattern": 1_500, "length": 1_200, "silhouette": 1_000,
          "neckline": 600, "sleeve": 600, "attribute_boolean": 2_000}


def token_subset(a, b):
    x, y = set(re.split(r"[ \-]", a)), set(re.split(r"[ \-]", b))
    return x <= y or y <= x


def confusable_with(value, groups):
    out = set()
    for g in groups:
        if value in g:
            out |= g
    return out


def make_options(gold, near, far, rng, cause, banned=()):
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


def extract(title, lexicon):
    """The single lexicon value in this title, or None (absent or ambiguous)."""
    words = set(re.findall(r"[a-z0-9\-']+", (title or "").lower()))
    hits = {lexicon[w] for w in words if w in lexicon}
    return hits.pop() if len(hits) == 1 else None


def main():
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("data-*.parquet"))
    if not shards:
        raise SystemExit("no fashion200k shards under .cache/datasets/v1/fashion200k")

    rows = []
    for s in shards:
        t = pq.read_table(s, columns=["category1", "category2", "category3", "item_ID"])
        for i, r in enumerate(t.to_pylist()):
            r["shard"] = s.name
            r["row"] = i
            rows.append(r)
    stats = Counter(rows=len(rows), shards=len(shards))

    items = []
    for r in rows:
        title = r["category3"] or ""
        attrs = {slot: extract(title, lex) for slot, (lex, _) in SLOTS.items()}
        items.append({"c1": r["category1"], "c2": r["category2"], "title": title,
                      "item": r["item_ID"], "group": r["item_ID"].rsplit("_", 1)[0],
                      "attrs": attrs, "shard": r["shard"], "row": r["row"]})
    rng.shuffle(items)
    for it in items:
        h = int(hashlib.sha256(f"{SEED}:{it['group']}".encode()).hexdigest()[:8], 16) % 1000
        it["pool"] = "test" if h < TEST_GROUP_PERMILLE else "fit"

    c2_by_c1 = defaultdict(set)
    for it in items:
        c2_by_c1[it["c1"]].add(it["c2"])
    all_c2 = sorted({it["c2"] for it in items})
    all_c1 = sorted({it["c1"] for it in items})
    slot_by_c1 = {s: defaultdict(Counter) for s in SLOTS}
    for it in items:
        for s, v in it["attrs"].items():
            if v:
                slot_by_c1[s][it["c1"]][v] += 1
    stats["distinct_category2"] = len(all_c2)
    stats["distinct_category1"] = len(all_c1)
    stats["products"] = len({it["group"] for it in items})

    used = set()
    records = []

    def partition_for(pool):
        return "test" if pool == "test" else ("dev" if rng.random() < DEV_FRACTION else "train")

    def emit_choice(it, family, question, gold, near, far, cause, banned=()):
        values = make_options(gold, near, far, rng, cause, banned)
        if values is None:
            return None
        field = {"id": "answer", "type": "choice", "question": question,
                 "options": [{"value": v} for v in values]}
        request = {"request_id": f"{family}-{it['item']}"[:128], "state": {}, "fields": [field]}
        Request.model_validate(request)
        return {"id": f"{SOURCE}:{family}-{it['item']}", "source": SOURCE,
                "source_split": "data", "source_group": it["group"], "family": family,
                "license": LICENSE, "images": [(it["shard"], it["row"])],
                "request": request, "target": None if cause else gold,
                "abstention_cause": cause, "source_answer": gold,
                "partition": partition_for(it["pool"])}

    for family, quota in QUOTAS.items():
        test_quota = min(TEST_PER_FAMILY, quota)
        made = Counter()
        for it in items:
            if made["test"] >= test_quota and made["fit"] >= quota - test_quota:
                break
            bucket = "test" if it["pool"] == "test" else "fit"
            if made[bucket] >= (test_quota if bucket == "test" else quota - test_quota):
                continue
            if it["item"] in used:
                continue
            if family == "apparel_category":
                gold = it["c2"]
                near = sorted(c2_by_c1[it["c1"]] - {gold})
                far = [c for c in all_c2 if c != gold and c not in c2_by_c1[it["c1"]]]
                rec = emit_choice(it, family, rng.choice(TEMPLATES[family]), gold, near, far,
                                  "not_listed" if rng.random() < NOT_LISTED_RATE else None,
                                  confusable_with(gold, CATEGORY_CONFUSABLE))
            elif family == "apparel_supercategory":
                gold = it["c1"]
                rec = emit_choice(it, family, rng.choice(TEMPLATES[family]), gold, [],
                                  [c for c in all_c1 if c != gold],
                                  "not_listed" if rng.random() < NOT_LISTED_RATE else None)
            elif family == "attribute_boolean":
                rec = emit_bool(it, rng, partition_for(it["pool"]), all_c2)
            else:
                gold = it["attrs"].get(family)
                if not gold:
                    continue
                lex, conf = SLOTS[family]
                banned = confusable_with(gold, conf)
                near = [v for v in slot_by_c1[family][it["c1"]] if v not in banned]
                far = [v for v in sorted(set(lex.values())) if v not in banned and v not in set(near)]
                rec = emit_choice(it, family, rng.choice(TEMPLATES[family]), gold, near, far,
                                  "not_listed" if rng.random() < NOT_LISTED_RATE else None,
                                  banned)
            if rec is None:
                continue
            used.add(it["item"])
            records.append(rec)
            made[bucket] += 1
        stats[f"family:{family}"] = made["test"] + made["fit"]

    written = write_images({r["images"][0] for r in records})
    final = []
    for r in records:
        info = written.get(r["images"][0])
        if info is None:
            stats["dropped_missing_image"] += 1
            continue
        r["images"] = [info]
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
               "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in final)),
               "option_counts": dict(sorted(Counter(len(r["request"]["fields"][0].get("options", []))
                                                    for r in final if "options" in r["request"]["fields"][0]).items())),
               "images": len(written), "stats": dict(stats)}
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def emit_bool(it, rng, partition, all_c2):
    choices = [s for s in ("color", "fabric", "pattern", "length", "silhouette")
               if it["attrs"].get(s)] + ["category"]
    family = rng.choice(choices)
    truth = rng.random() < 0.5
    if family == "category":
        gold = it["c2"]
        if truth:
            value = gold
        else:
            banned = confusable_with(gold, CATEGORY_CONFUSABLE)
            pool = [c for c in all_c2
                    if c != gold and c not in banned and not token_subset(c, gold)]
            if not pool:
                return None
            value = rng.choice(pool)
    else:
        lex, conf = SLOTS[family]
        gold = it["attrs"][family]
        if truth:
            value = gold
        else:
            banned = confusable_with(gold, conf)
            pool = [v for v in sorted(set(lex.values()))
                    if v not in banned and not token_subset(v, gold)]
            if not pool:
                return None
            value = rng.choice(pool)
    field = {"id": "answer", "type": "boolean",
             "question": rng.choice(BOOL_TEMPLATES[family]).format(v=value)}
    request = {"request_id": f"bool-{it['item']}"[:128], "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:bool-{it['item']}", "source": SOURCE, "source_split": "data",
            "source_group": it["group"], "family": "attribute_boolean", "license": LICENSE,
            "images": [(it["shard"], it["row"])], "request": request, "target": truth,
            "abstention_cause": None, "source_answer": f"{family}={gold}",
            "partition": partition}


def write_images(wanted):
    by_shard = defaultdict(set)
    for shard, row in wanted:
        by_shard[shard].add(row)
    written = {}
    for shard, want_rows in by_shard.items():
        offset = 0
        pf = pq.ParquetFile(CACHE / shard)
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            hits = [r for r in range(offset, offset + n) if r in want_rows]
            if hits:
                col = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
                for r in hits:
                    blob = col[r - offset]["bytes"]
                    try:
                        with Image.open(io.BytesIO(blob)) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            if im.width * im.height > 1_000_000:
                                s = (1_000_000 / (im.width * im.height)) ** 0.5
                                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                                               Image.LANCZOS)
                            buf = io.BytesIO()
                            im.save(buf, "JPEG", quality=90)
                            data = buf.getvalue()
                            w, h = im.size
                    except Exception:
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    dest = IMG_DIR / f"{sha}.jpg"
                    if not dest.exists():
                        dest.write_bytes(data)
                    written[(shard, r)] = {"image": str(dest), "sha256": sha,
                                           "width": w, "height": h}
            offset += n
    return written


if __name__ == "__main__":
    main()
