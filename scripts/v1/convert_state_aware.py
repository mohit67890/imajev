"""state_aware -> decision-v1 records whose `request.state` is load-bearing.

Every other source in decision-v1 ships `state: {}`. Jev's real API hands the model a
structured JSON `state` and a question whose wording *references* state keys in
backticks -- either a top-level key whose value is substituted in (`subject`,
`attribute`) or a dotted path (`listing.color`). A model trained on 520k empty-state
records has never had to read one. This source is the fix.

Nothing is downloaded and no new image is encoded: every record reuses an image an
earlier converter already wrote, hard-linked into `data/decision-v1/state_aware/images/`
under the same sha256 name, and inherits that record's `partition`, `license` and
(namespaced) `source_group`, so the train/dev/test division of the mixture is unchanged.

Four families:

* `listing_verification` (ABO listings) -- a realistic listing object built from the real
  seller metadata for the photographed product; boolean "does the photo match
  `listing.<field>`" (false = exactly one referenced field swapped for a clearly
  different, non-confusable value of the same attribute), a choice "which listed field
  does the photo contradict" with `none of them` as an ordinary option, abstentions that
  reference an absent field (`false_premise`) or a field no photograph can settle
  (`insufficient_evidence`), and a two-image reference/received variant.
* `state_scoped_question` (vg_attributes, tdiuc, vqav2) -- the question's parameters move
  out of the sentence and into the state under seventeen different key layouts.
* `query_relevance` (sqid_esci) -- the search query moves into the state; the 4-level ESCI
  ordinal is re-indexed from 0 the way Jev score criteria are.
* `candidate_labels_in_state` (abo, marqo_gs, fashion200k) -- the candidate set is also in
  the state, and one record in five carries a seller's claim that is wrong half the time.

Serving-shape details copied from real Jev requests: about 15% of instructions are built
as an object ({"question": ..., "note": ...}) and stored flattened through
`vision_decision.jev_api.flatten_instructions`, about 20% carry judging rules that are
true to how the label was derived, ordinal levels are numbered from 0 with short
descriptions, and field ids are meaningful snake_case names.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_state_aware.py
"""
import gzip
import json
import os
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import abo_taxonomy as T
from vision_decision.contracts import Request
from vision_decision.jev_api import flatten_instructions

SOURCE = "state_aware"
SEED = 20260922
DATA = Path("data/decision-v1")
OUT = DATA / SOURCE
IMG_DIR = OUT / "images"
CACHE = Path(".cache/datasets/v1/state_aware")
ABO_LISTINGS = Path(".cache/datasets/v1/abo/abo-listings.tar")

FIT_QUOTA = {"listing_verification": 22_000, "state_scoped_question": 10_000,
             "query_relevance": 6_000, "candidate_labels_in_state": 7_000}
TEST_QUOTA = {f: 300 for f in FIT_QUOTA}
STATEMENT_RATE = 0.40     # share of listing booleans phrased as statements
TWO_IMAGE_SHARE = 0.15
ABSTAIN_SHARE = 0.15      # of listing_verification, split false_premise/insufficient
CHOICE_SHARE = 0.25
PRIOR_RATE = 0.20         # share of candidate_labels records carrying a seller claim
HIDDEN_NOISE_RATE = 0.20  # unreferenced not-photo-judgeable field kept in the state
OBJECT_FORM_RATE = 0.15   # instructions built as a Jev instruction object
RUBRIC_RATE = 0.20        # instructions carrying explicit judging rules

# ---------------------------------------------------------------- listing vocabulary
ALL_LABELS = sorted({lab for lab, _ in T.PRODUCT_TYPES.values()})
LABEL_GROUP = {lab: grp for lab, grp in T.PRODUCT_TYPES.values()}
CHECKABLE = {                       # state key -> (listing key, value map, pool, confusable)
    "product_type": (None, None, ALL_LABELS, T.PRODUCT_CONFUSABLE),
    "color": ("color", T.COLOR_MAP, T.COLOR_POOL, T.COLOR_CONFUSABLE),
    "material": ("material", T.MATERIAL_MAP, T.MATERIAL_POOL, T.MATERIAL_CONFUSABLE),
    "pattern": ("pattern", T.PATTERN_MAP, T.PATTERN_POOL, T.PATTERN_CONFUSABLE),
    "shape": ("item_shape", T.SHAPE_MAP, T.SHAPE_POOL, T.SHAPE_CONFUSABLE),
    "finish": ("finish_type", T.FINISH_MAP, T.FINISH_POOL, T.FINISH_CONFUSABLE),
}
NOUN = {"product_type": "product type", "color": "colour", "material": "material",
        "pattern": "pattern", "shape": "shape", "finish": "finish",
        "fabric_type": "fibre composition", "item_weight": "weight",
        "model_number": "model number", "model_year": "model year",
        "item_dimensions": "dimensions"}
# Fields the ABO README/taxonomy names as invisible in a catalogue photo ("fibre
# percentages, model names, pack counts"; fabric_type is "dropped entirely" there).
HIDDEN = ["fabric_type", "item_weight", "model_number", "model_year", "item_dimensions"]

# ---------------------------------------------------------------- question wording
BOOL_Q = [
    "Does the photo match `{p}`?",
    "Is the product in the photo consistent with `{p}`?",
    "Looking at the picture, does the item match `{p}`?",
    "Does the item shown agree with the value recorded at `{p}`?",
    "Is `{p}` correct for the product in this photo?",
    "Judging from the photo alone, is the listing's `{p}` accurate?",
    "Does this photograph support `{p}`?",
]
BOOL_S = [
    "The product in the photo matches `{p}`.",
    "The product's {n} matches `{p}`.",
    "The item shown is consistent with `{p}`.",
    "`{p}` correctly describes the product in this photo.",
    "The photo agrees with the value stored at `{p}`.",
    "What the picture shows matches the listing's `{p}`.",
    "The {n} of the photographed item is the one recorded at `{p}`.",
]
CHOICE_Q = [
    "Which field of `{r}` does the photo contradict?",
    "Which listed field of `{r}` is contradicted by the photo?",
    "Compare the photo with `{r}`. Which field does not match?",
    "One of these fields may be wrong. Using the photo, which field of `{r}` is incorrect?",
    "Check the photograph against `{r}` and name the field that is wrong.",
    "Which entry in `{r}` does the picture disagree with?",
    "Reading `{r}` against this photo, which field is mistaken?",
]
NONE_OPTION = "none of them"
TWO_IMAGE_Q = [
    "The first photo is the listing image. Does the second photo show the item described by `{r}`?",
    "Photo 1 is the catalogue shot. Is the item in photo 2 the product described by `{r}`?",
    "Image 1 is the listing photo and image 2 is what arrived. Does it match `{r}`?",
    "Photo 1 shows the listing; photo 2 shows the item received. Is it the product in `{r}`?",
    "Compare the second photo with the listing in `{r}`, whose own photo is image 1. Same product?",
    "The product described by `{r}` appears in photo 1. Is the item in photo 2 that same product?",
]
TWO_IMAGE_S = [
    "The second photo shows the product described by `{r}` (photo 1 is its listing image).",
    "Photo 2 is the item that arrived; it is the product recorded in `{r}`.",
    "The item in image 2 is the same product as the listing in `{r}` shown in image 1.",
]
ATTR_Q = [
    "What is the `{a}` of the `{o}` in this photo?",
    "Looking at the image, report the `{a}` of the `{o}`.",
    "The state names one object and one property. What is the `{a}` of the `{o}`?",
    "Which value describes the `{a}` of the `{o}` shown here?",
    "Inspect the photo and give the `{a}` of the `{o}`.",
    "For the object named at `{o}`, what is its `{a}`?",
    "Read `{o}` and `{a}` from the state, then answer for this photo.",
]
COUNT_Q = [
    "How many `{o}` are visible in this photo?",
    "Count the `{o}` in the image.",
    "The state names what to count. How many `{o}` are there?",
    "Looking at the picture, how many `{o}` can you see?",
    "Give the number of `{o}` shown in this photo.",
    "For the target named at `{o}`, how many appear in the image?",
]
POS_Q = [
    "Which object is `{rel}` the `{o}` in this photo?",
    "The state gives a relation and an anchor. What is `{rel}` the `{o}`?",
    "Looking at the image, what sits `{rel}` the `{o}`?",
    "Name the thing that is `{rel}` the `{o}` in this picture.",
    "Using `{rel}` and `{o}` from the state, what do you see there?",
    "What object appears `{rel}` the `{o}` in this photo?",
]
ESCI_Q = [
    "How well does the product in this photo match `{q}`?",
    "A shopper ran the search recorded at `{q}`. How well does this product answer it?",
    "Rate how well the pictured product matches the query at `{q}`.",
    "Given the search stored at `{q}`, how relevant is the item in this photo?",
    "Judge the relevance of the product shown to `{q}`.",
    "Someone searched for `{q}`. How good a result is the product in this photo?",
]
LABEL_Q = [
    "Which of the `{c}` best describes the item in this photo?",
    "Pick the entry from `{c}` that matches the product shown.",
    "The state lists the allowed answers at `{c}`. Which one is the item in the picture?",
    "Using only the values in `{c}`, what is the object photographed here?",
    "Choose the label from `{c}` that fits this photo.",
    "Which value of `{c}` names the product in this image?",
]
PRIOR_Q = [
    "The seller listed this as `{s}`. Using `{c}`, what does the photo actually show?",
    "`{s}` is the seller's own claim. Judge the photo instead: which entry of `{c}` is it?",
    "Ignore the claim at `{s}` if the picture disagrees. Which of the `{c}` is shown?",
    "The listing says `{s}`. Going by the photograph alone, which value of `{c}` is right?",
    "A seller typed `{s}`. What does the photo show, choosing from `{c}`?",
    "Check `{s}` against the picture and answer with the matching entry of `{c}`.",
]

# ------------------------------------------------- judging rules, true to the labels
RUBRIC = {
    "product_type": ["Name what the item is, not what it is used for or sold with.",
                     "Judge the product itself; packaging, props and accessories do not count."],
    "color": ["Judge the main colour of the product itself; ignore the background, "
              "packaging and small trim.",
              "A single listed colour is expected, so pick the dominant one."],
    "material": ["Judge the main material of the item; ignore packaging and accessories.",
                 "Trim, fittings and linings do not decide the answer."],
    "pattern": ["Judge the patterning of the product itself, not of the backdrop."],
    "shape": ["Judge the overall outline of the item, not of any detail on it."],
    "finish": ["Judge the surface finish of the product, not the lighting of the shot."],
    "listing_bool": ["Only the field named in the question matters; other listed fields "
                     "may be wrong without changing your answer.",
                     "Use the photograph as the evidence; the listing text is the claim "
                     "being checked."],
    "listing_choice": ["At most one listed field is wrong; it is also possible that none is.",
                       "Fields that are not listed as options are not your concern."],
    "two_image": ["Judge whether it is the same product, not whether the two photographs "
                  "were taken the same way.",
                  "A different colour, size or model of the same kind of product is a "
                  "different product."],
    "scoped_attr": ["Judge only the object named in the state, not the rest of the scene."],
    "esci": ["Grade the product against every requirement in the query; a product that "
             "meets only some of them is not an exact match.",
             "Judge the product in the photo, not the seller or the price."],
    "labels": ["Answer from the photograph; a claim stored in the state is not evidence.",
               "Pick the closest listed label; do not invent one."],
}
CONTEXT = {
    "listing": [("note", "Other listed fields may be inaccurate."),
                ("scope", "the product in the photo, not the background or packaging"),
                ("focus", "the single field named in the question")],
    "listing_choice": [("scope", "the fields offered as options"),
                       ("note", "It is possible that no listed field is wrong.")],
    "two_image": [("note", "Photo 1 is the listing image, photo 2 is the item received."),
                  ("scope", "the product, not the photography")],
    "scoped": [("scope", "the single object named in the state"),
               ("note", "Answer for this photograph only.")],
    "esci": [("scope", "the product shown in the photo"),
             ("note", "Relevance is judged against the query text as typed.")],
    "labels": [("note", "The photograph outranks any claim stored in the state."),
               ("scope", "the item shown, not its packaging")],
}

# ---------------------------------------------------------------- state layouts
# `top` marks layouts whose backticked reference is a bare top-level key whose value is
# substituted into the sentence, as in Jev's own requests.
ATTR_SHAPES = [
    (True, lambda o, a: ({"target_object": o, "attribute": a}, "target_object", "attribute")),
    (False, lambda o, a: ({"query": {"object": o, "property": a}}, "query.object", "query.property")),
    (False, lambda o, a: ({"subject": {"name": o}, "ask": {"attribute": a}},
                          "subject.name", "ask.attribute")),
    (False, lambda o, a: ({"inspect": {"item": o, "aspect": a}}, "inspect.item", "inspect.aspect")),
    (True, lambda o, a: ({"object_name": o, "dimension": a, "scope": "one object"},
                         "object_name", "dimension")),
    (False, lambda o, a: ({"task": {"target": o, "attribute": a}, "request": {"locale": "en"}},
                          "task.target", "task.attribute")),
    (True, lambda o, a: ({"focus": o, "measure": a}, "focus", "measure")),
    (False, lambda o, a: ({"annotation": {"object": {"name": o}, "field": a}},
                          "annotation.object.name", "annotation.field")),
    (True, lambda o, a: ({"item": o, "attribute": a, "notes": {"source": "annotator"}},
                         "item", "attribute")),
    (False, lambda o, a: ({"scene": {"target": {"label": o}}, "property": a},
                          "scene.target.label", "property")),
    (True, lambda o, a: ({"subject": o, "property": a}, "subject", "property")),
]
COUNT_SHAPES = [
    (True, lambda o: ({"count_target": o}, "count_target")),
    (False, lambda o: ({"count": {"target": o}}, "count.target")),
    (False, lambda o: ({"tally": {"object": o, "unit": "instances"}}, "tally.object")),
    (True, lambda o: ({"target_object": o, "task": "count"}, "target_object")),
]
POS_SHAPES = [
    (True, lambda o, r: ({"anchor": o, "relation": r}, "anchor", "relation")),
    (False, lambda o, r: ({"spatial": {"anchor": o, "relation": r}},
                          "spatial.anchor", "spatial.relation")),
    (True, lambda o, r: ({"reference_object": o, "direction": r, "scope": "one object"},
                         "reference_object", "direction")),
]
LABEL_SHAPES = [
    (True, lambda labels: ({"candidate_labels": labels}, "candidate_labels", None)),
    (False, lambda labels: ({"taxonomy": {"candidate_labels": labels, "version": "v1"}},
                            "taxonomy.candidate_labels", "taxonomy")),
    (True, lambda labels: ({"candidate_labels": labels, "locale": "en"},
                           "candidate_labels", None)),
    (False, lambda labels: ({"classification": {"candidate_labels": labels}},
                            "classification.candidate_labels", "classification")),
]
ESCI_SHORT = {0: "Irrelevant", 1: "Complement", 2: "Substitute", 3: "Exact"}


def token_subset(a, b):
    x, y = set(re.split(r"[ \-]", a)), set(re.split(r"[ \-]", b))
    return x <= y or y <= x


def confusable_with(value, groups):
    out = set()
    for g in groups:
        if value in g:
            out |= g
    return out


def swap_value(key, gold, r):
    """A clearly different value of the same attribute: never a confusable shade,
    never a synonym or a super/sub-string of the gold."""
    _, _, pool, conf = CHECKABLE[key]
    banned = confusable_with(gold, conf)
    if key == "product_type":
        grp = LABEL_GROUP.get(gold)
        cands = [v for v in pool if v != gold and v not in banned
                 and LABEL_GROUP.get(v) != grp and not token_subset(v, gold)]
    else:
        cands = [v for v in pool if v != gold and v not in banned and not token_subset(v, gold)]
    return r.choice(cands) if cands else None


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "q_" + s
    return s[:56]


# ---------------------------------------------------------------- shared build state
rng = random.Random(SEED)
records = []
used_sha = set()
ref_uses = Counter()
group_bucket = {}
stats = Counter()


def instruct(r, text, context=(), rubric=()):
    """Jev instructions: optionally with judging rules, optionally as an object that is
    flattened exactly the way the server flattens it."""
    if rubric and r.random() < RUBRIC_RATE:
        rules = list(rubric)
        r.shuffle(rules)
        text = text + " " + " ".join(rules[:r.randint(1, min(3, len(rules)))])
        stats["style:rubric"] += 1
    if context and r.random() < OBJECT_FORM_RATE:
        extras = list(context)
        r.shuffle(extras)
        obj = {"question": text}
        for key, val in extras[:r.randint(1, min(2, len(extras)))]:
            obj[key] = val
        text = flatten_instructions(obj)
        stats["style:object_form"] += 1
    return text


def phrase(r, statements, questions, rate=STATEMENT_RATE):
    statement = r.random() < rate
    stats["phrasing:statement" if statement else "phrasing:question"] += 1
    return r.choice(statements if statement else questions)


def bucket_of(partition):
    return "test" if partition == "test" else "fit"


def claim_group(group, partition):
    """A source_group must never straddle test and train/dev."""
    return group_bucket.setdefault(group, bucket_of(partition)) == bucket_of(partition)


def adopt(info):
    """Hard-link an image an earlier converter already wrote, keeping its sha name."""
    sha = info["sha256"]
    dest = IMG_DIR / f"{sha}.jpg"
    if not dest.exists():
        os.link(info["image"], dest)
    return {"image": str(dest), "sha256": sha, "width": info["width"], "height": info["height"]}


def emit(qid, family, origin, state, field, target, cause, source_answer, images,
         partition, group, source_split, license_):
    request = {"request_id": qid[:128], "state": state, "fields": [field]}
    Request.model_validate(request)
    records.append({"id": f"{SOURCE}:{qid}", "source": SOURCE, "source_split": source_split,
                    "source_group": group, "family": family, "license": license_,
                    "images": images, "request": request, "target": target,
                    "abstention_cause": cause, "source_answer": source_answer,
                    "partition": partition, "origin": origin})


def read_records(name):
    return [json.loads(line) for line in (DATA / name / "records.jsonl").read_text().splitlines()]


# ===================================================================== ABO metadata
ID_ITEM = re.compile(
    r"^(?:cat|color|material|pattern|item_shape|finish|style|same"
    r"|reference_color|reference_material|reference_category"
    r"|bool-(?:color|material|pattern|item_shape|category))-([A-Z0-9]{10})-")


def english(entries):
    return [e for e in entries or [] if str(e.get("language_tag", "")).startswith("en")]


def curated(row, key, vmap):
    raw = set()
    for e in english(row.get(key, [])):
        if key == "color":
            raw |= {str(s).strip().lower() for s in (e.get("standardized_values") or [])}
        else:
            raw.add(str(e.get("value", "")).strip().lower())
    mapped = {vmap[v] for v in raw if v in vmap}
    if len(mapped) != 1 or (len(raw - set(vmap)) > 0 and len(raw) > 1):
        return None
    return mapped.pop()


def strip_brand(title, brand):
    t = title.replace(" ", " ").strip()
    if brand:
        for piece in sorted({brand, brand.split(" - ")[-1]}, key=len, reverse=True):
            if piece:
                t = re.sub(re.escape(piece), "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -–—,:|")
    return t[:180].strip()


def hidden_fields(row):
    """Values a photograph cannot settle, taken verbatim from the listing."""
    out = {}
    fab = (english(row.get("fabric_type", [])) or [{}])[0].get("value")
    if fab and re.search(r"\d", str(fab)):        # fibre percentages only
        out["fabric_type"] = str(fab)[:80]
    w = (row.get("item_weight") or [{}])[0]
    if w.get("value") is not None and w.get("unit"):
        out["item_weight"] = f"{w['value']} {w['unit']}"[:40]
    mn = (row.get("model_number") or [{}])[0].get("value")
    if mn and len(str(mn)) <= 40:
        out["model_number"] = str(mn)
    my = (row.get("model_year") or [{}])[0].get("value")
    if isinstance(my, (int, float)):
        out["model_year"] = int(my)
    dims = row.get("item_dimensions") or {}
    parts = [f"{dims[k]['value']:g} {dims[k]['unit']}" for k in ("length", "width", "height")
             if isinstance(dims.get(k), dict) and dims[k].get("value") is not None]
    if len(parts) == 3:
        out["item_dimensions"] = " x ".join(parts)[:60]
    return out


def load_abo_meta(wanted):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "abo_listing_meta.json"
    if cache.exists():
        return json.loads(cache.read_text())
    meta = {}
    with tarfile.open(ABO_LISTINGS) as tar:
        for member in tar:
            if not member.name.endswith(".json.gz"):
                continue
            for line in gzip.GzipFile(fileobj=tar.extractfile(member)):
                row = json.loads(line)
                item = row["item_id"]
                if item not in wanted or item in meta:
                    continue
                name = (english(row.get("item_name", [])) or [{}])[0]
                title = name.get("value")
                pt = (row.get("product_type") or [{}])[0].get("value")
                label = T.PRODUCT_TYPES.get(pt)
                if not title or not label:
                    continue
                attrs = {"product_type": label[0]}
                for key, (lk, vmap, _, _) in CHECKABLE.items():
                    if lk:
                        v = curated(row, lk, vmap)
                        if v:
                            attrs[key] = v
                brand = (english(row.get("brand", [])) or [{}])[0].get("value", "")
                meta[item] = {"title": strip_brand(title, brand), "pt": pt,
                              "cgroup": label[1], "attrs": attrs,
                              "hidden": hidden_fields(row), "sku": item,
                              "locale": name.get("language_tag", "en_US"),
                              "marketplace": row.get("domain_name", "amazon.com")}
    cache.write_text(json.dumps(meta, sort_keys=True))
    return meta


# ===================================================================== family 1
def pick_fields(r, m, n_max=3):
    """1..n_max photo-checkable fields the listing really carries. A real listing nearly
    always names the product type, but it is left out of 40% of the states that can spare
    it: the product title gives the type away in words, so a state carrying both would let
    the model answer a product_type question from the text alone."""
    have = [k for k in CHECKABLE if k in m["attrs"]]
    if not have:
        return []
    r.shuffle(have)
    if "product_type" in have:
        have.remove("product_type")
        if not have or r.random() >= 0.40:
            have = ["product_type"] + have
    return have[:r.randint(1, min(n_max, len(have)))]


def normalise(text):
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip() + " "


def title_safe(m, guard_keys, hidden_key):
    """A title may only join the state when it cannot answer *this* question by itself.
    `guard_keys` are the fields the question could be answered with (the referenced field
    for a boolean, every option for a choice, none for the two-image and absent-field
    forms): no value of their vocabularies, and no not-photo-judgeable value the question
    names, may appear in the title, and `product_type` — which a product title always
    spells out in words — may not be among them. The whole vocabulary is checked rather
    than the gold, so a swapped record is no likelier to lack a title than a truthful one."""
    if not m["title"] or "product_type" in guard_keys:
        return False
    hay = normalise(m["title"])
    values = {v for k in guard_keys for v in CHECKABLE[k][2]}
    if hidden_key:
        values.add(str(m["hidden"][hidden_key]))
    return not any(normalise(v) in hay for v in values)


def listing_values(r, m, keys, swap_key=None):
    """Field/value pairs for the state, with at most one field swapped for a clearly
    different value of the same attribute."""
    fields, swapped = [], None
    for k in keys:
        gold = m["attrs"][k]
        if k == swap_key:
            cand = swap_value(k, gold, r)
            if cand is None:
                return None
            fields.append((k, cand))
            swapped = (k, gold, cand)
        else:
            fields.append((k, gold))
    if swap_key and swapped is None:
        return None
    return {"fields": fields, "swapped": swapped}


def build_listing_state(r, m, values, hidden_key=None, drop_key=None, noise_hidden=None,
                        allow_flat=True, guard_keys=None):
    """A realistic listing state: 2-6 leaf entries, sometimes nested, sometimes flat with
    short top-level keys, sometimes with unrelated fields.
    Returns (state, {field: reference}, root reference or None, title_used)."""
    shape = r.choice([0, 1, 2, 3, 4, 5, 6, 6, 7, 7] if allow_flat else [0, 1, 2, 3, 4, 5])
    root, sub = {0: ("listing", None), 1: ("listing", "attributes"), 2: ("listing", None),
                 3: ("listing", None), 4: ("catalog.listing", "specs"),
                 5: ("order.listing", "attributes"),
                 6: (None, None), 7: (None, None)}[shape]
    keys = [k for k, _ in values["fields"]]
    hk = hidden_key or noise_hidden
    n = len(keys) + (1 if hk else 0) + (1 if shape == 5 else 0)
    if n > 6 and noise_hidden and not hidden_key:      # unreferenced noise goes first
        hk, n = None, n - 1
    title_ok = title_safe(m, keys if guard_keys is None else guard_keys, hk) and n < 6

    inner, paths = {}, {}

    def ref(key, nested=False):
        if root is None:
            return key
        return f"{root}.{sub}.{key}" if (nested and sub) else f"{root}.{key}"

    if title_ok:
        inner["item" if shape == 7 else "title"] = m["title"]
        n += 1
    holder = inner if sub is None else inner.setdefault(sub, {})
    for key, val in values["fields"]:
        holder[key] = val
        paths[key] = ref(key, nested=True)
    if hk:
        inner[hk] = m["hidden"][hk]
        paths[hk] = ref(hk)
    if drop_key:
        paths[drop_key] = ref(drop_key, nested=True)

    top = {}
    if shape == 5:
        top["order_id"] = f"{m['sku']}-1"
    want = max(0, 2 - n) + (1 if (n < 5 and r.random() < 0.45) else 0)
    for key, val in [("sku", m["sku"]), ("locale", m["locale"]),
                     ("marketplace", m["marketplace"])][:want]:
        (top if shape in (2, 5) else inner)[key] = val
        n += 1

    if shape == 4:
        state = {"catalog": {"listing": inner}}
    elif shape == 5:
        state = {"order": dict(top, listing=inner)}
    elif root is None:
        state = inner
    else:
        state = dict(top, listing=inner)
    stats["shape:flat" if root is None else "shape:nested"] += 1
    return state, paths, root, title_ok


def make_bool(item, m, info, part, grp, split, lic):
    truth = stats["listing_true"] <= stats["listing_false"]
    keys = pick_fields(rng, m)
    if not keys:
        return False
    ref = rng.choice(keys)
    vals = listing_values(rng, m, keys, swap_key=None if truth else ref)
    if vals is None:
        return False
    noise = rng.choice(sorted(m["hidden"])) if (m["hidden"] and rng.random() < HIDDEN_NOISE_RATE) else None
    state, paths, root, title_ok = build_listing_state(rng, m, vals, noise_hidden=noise,
                                                       guard_keys=[ref])
    text = phrase(rng, BOOL_S, BOOL_Q)
    question = instruct(rng, text.format(p=paths[ref], n=NOUN[ref]),
                        CONTEXT["listing"], RUBRIC["listing_bool"] + RUBRIC.get(ref, []))
    field = {"id": slug(f"matches_{ref}"), "type": "boolean", "question": question}
    gold = m["attrs"][ref]
    shown = dict(vals["fields"])[ref]
    used_sha.add(info["sha256"])
    stats["listing_true" if truth else "listing_false"] += 1
    stats["title_used" if title_ok else "title_dropped"] += 1
    emit(f"lv-bool-{item}-{info['sha256'][:10]}", "listing_verification", "abo", state,
         field, truth, None, f"{ref}={gold}" if truth else f"{ref}={gold}, state says {shown}",
         [adopt(info)], part, grp, split, lic)
    return True


def make_choice(item, m, info, part, grp, split, lic):
    keys = pick_fields(rng, m, n_max=4)
    if not keys:
        return False
    swap = rng.random() < 0.55
    ref = rng.choice(keys) if swap else None
    vals = listing_values(rng, m, keys, swap_key=ref)
    if vals is None:
        return False
    noise = rng.choice(sorted(m["hidden"])) if (m["hidden"] and rng.random() < HIDDEN_NOISE_RATE) else None
    state, paths, root, title_ok = build_listing_state(rng, m, vals, noise_hidden=noise,
                                                       allow_flat=False)
    options = [paths[k] for k, _ in vals["fields"]] + [NONE_OPTION]
    if len(options) < 2:
        return False
    rng.shuffle(options)
    question = instruct(rng, rng.choice(CHOICE_Q).format(r=root),
                        CONTEXT["listing_choice"], RUBRIC["listing_choice"])
    field = {"id": "contradicted_field", "type": "choice", "question": question,
             "options": [{"value": o} for o in options]}
    used_sha.add(info["sha256"])
    stats["choice_swapped" if ref else "choice_none"] += 1
    stats["title_used" if title_ok else "title_dropped"] += 1
    emit(f"lv-choice-{item}-{info['sha256'][:10]}", "listing_verification", "abo", state,
         field, paths[ref] if ref else NONE_OPTION, None,
         f"{ref}={m['attrs'][ref]} swapped" if ref else "no field swapped",
         [adopt(info)], part, grp, split, lic)
    return True


def make_hidden_bool(item, m, info, part, grp, split, lic):
    """The referenced field is in the state, but no photograph can settle it."""
    if not m["hidden"]:
        return False
    hk = rng.choice(sorted(m["hidden"]))
    keys = pick_fields(rng, m)
    vals = listing_values(rng, m, keys)
    if vals is None:
        return False
    state, paths, root, title_ok = build_listing_state(rng, m, vals, hidden_key=hk,
                                                       guard_keys=[])
    text = phrase(rng, BOOL_S, BOOL_Q)
    question = instruct(rng, text.format(p=paths[hk], n=NOUN[hk]),
                        CONTEXT["listing"], RUBRIC["listing_bool"])
    field = {"id": slug(f"matches_{hk}"), "type": "boolean", "question": question}
    used_sha.add(info["sha256"])
    stats["title_used" if title_ok else "title_dropped"] += 1
    emit(f"lv-ie-{item}-{info['sha256'][:10]}", "listing_verification", "abo", state, field,
         None, "insufficient_evidence", f"{hk}={m['hidden'][hk]} (not visible in a photograph)",
         [adopt(info)], part, grp, split, lic)
    return True


def make_absent_bool(item, m, info, part, grp, split, lic):
    """The referenced path is not in the state at all."""
    keys = pick_fields(rng, m)
    vals = listing_values(rng, m, keys)
    if vals is None:
        return False
    missing = [k for k in CHECKABLE if k not in keys]
    if not missing:
        return False
    mk = rng.choice(missing)
    state, paths, root, title_ok = build_listing_state(rng, m, vals, drop_key=mk,
                                                       guard_keys=[])
    text = phrase(rng, BOOL_S, BOOL_Q)
    question = instruct(rng, text.format(p=paths[mk], n=NOUN[mk]),
                        CONTEXT["listing"], RUBRIC["listing_bool"])
    field = {"id": slug(f"matches_{mk}"), "type": "boolean", "question": question}
    used_sha.add(info["sha256"])
    stats["title_used" if title_ok else "title_dropped"] += 1
    emit(f"lv-fp-{item}-{info['sha256'][:10]}", "listing_verification", "abo", state, field,
         None, "false_premise", f"{mk} is absent from the state",
         [adopt(info)], part, grp, split, lic)
    return True


def make_two_image(item, m, bucket, info, part, grp, split, lic, free, by_pt, meta, take):
    truth = stats["two_true"] <= stats["two_false"]
    ref_sha = info["sha256"]
    second = None
    if truth:
        got = take(item, bucket, skip={ref_sha})
        if got:
            second = got[1]
    else:
        peers = by_pt.get((m["pt"], bucket), [])
        for _ in range(20):
            if not peers:
                break
            other = peers[rng.randrange(len(peers))]
            if other == item:
                continue
            got = take(other, bucket, skip={ref_sha})
            if got and got[3] != grp:
                second = got[1]
                break
    if second is None:
        return False
    keys = pick_fields(rng, m)
    vals = listing_values(rng, m, keys)
    if vals is None:
        return False
    state, paths, root, title_ok = build_listing_state(rng, m, vals, allow_flat=False,
                                                       guard_keys=[])
    text = phrase(rng, TWO_IMAGE_S, TWO_IMAGE_Q, rate=0.25)
    question = instruct(rng, text.format(r=root), CONTEXT["two_image"], RUBRIC["two_image"])
    field = {"id": "received_matches_listing", "type": "boolean", "question": question}
    used_sha.add(second["sha256"])
    ref_uses[ref_sha] += 1
    if ref_uses[ref_sha] >= 3:
        used_sha.add(ref_sha)
    stats["two_true" if truth else "two_false"] += 1
    stats["title_used" if title_ok else "title_dropped"] += 1
    emit(f"lv-two-{item}-{second['sha256'][:10]}", "listing_verification", "abo", state, field,
         truth, None, "same_item_id" if truth else "different_item_id, same product_type",
         [adopt(info), adopt(second)], part, grp, split, lic)
    return True


def listing_verification(abo_rows, meta):
    items = defaultdict(list)
    for rec in abo_rows:
        hit = ID_ITEM.match(rec["id"].split(":", 1)[1])
        if hit and hit.group(1) in meta:
            items[hit.group(1)].append(rec)
    order = sorted(items)
    rng.shuffle(order)

    free = {}
    for item in order:
        seen, lst = set(), []
        for rec in items[item]:
            info = rec["images"][0]
            if info["sha256"] in seen:
                continue
            seen.add(info["sha256"])
            lst.append((info, rec["partition"], f"abo:{rec['source_group']}",
                        rec["source_split"], rec["license"]))
        free[item] = lst
    by_pt = defaultdict(list)
    for item in order:
        for _, part, _, _, _ in free[item]:
            by_pt[(meta[item]["pt"], bucket_of(part))].append(item)

    def take(item, bucket, skip=()):
        for info, part, grp, split, lic in free[item]:
            if bucket_of(part) != bucket or info["sha256"] in used_sha or info["sha256"] in skip:
                continue
            return part, info, bucket, grp, split, lic
        return None

    for bucket, total in (("fit", FIT_QUOTA["listing_verification"]),
                          ("test", TEST_QUOTA["listing_verification"])):
        n_two = round(total * TWO_IMAGE_SHARE)
        n_ab = round(total * ABSTAIN_SHARE / 2)
        n_choice = round(total * CHOICE_SHARE)
        kinds = [("two_image", n_two), ("insufficient", n_ab), ("false_premise", n_ab),
                 ("choice", n_choice), ("bool", total - n_two - 2 * n_ab - n_choice)]
        for kind, quota in kinds:
            done = 0
            for item in order:
                if done >= quota:
                    break
                m = meta[item]
                if not m["attrs"]:
                    continue
                got = take(item, bucket)
                if got is None:
                    continue
                part, info, _, grp, split, lic = got
                if not claim_group(grp, part):
                    continue
                if kind == "two_image":
                    ok = make_two_image(item, m, bucket, info, part, grp, split, lic,
                                        free, by_pt, meta, take)
                elif kind == "insufficient":
                    ok = make_hidden_bool(item, m, info, part, grp, split, lic)
                elif kind == "false_premise":
                    ok = make_absent_bool(item, m, info, part, grp, split, lic)
                elif kind == "choice":
                    ok = make_choice(item, m, info, part, grp, split, lic)
                else:
                    ok = make_bool(item, m, info, part, grp, split, lic)
                if ok:
                    done += 1
            stats[f"listing:{bucket}:{kind}"] = done


# ===================================================================== family 2
DIM_NOUN = {"color": "colour", "material": "material", "shape": "shape",
            "posture": "posture", "doing": "activity", "openness": "open/closed state",
            "age": "apparent age", "vehicle_motion": "motion state",
            "power": "power state", "wetness": "wetness", "fullness": "fullness",
            "cleanliness": "cleanliness", "air_state": "inflation state",
            "cooked": "cooked state", "temperature_state": "temperature",
            "ripeness": "ripeness"}
TDIUC_PAT = {
    "colour": re.compile(r"^What colou?r (?:is|are) (?:the )?(.{1,40}?)\?$", re.I),
    "material": re.compile(r"^What (?:is|are) the (.{1,40}?) made of\?$", re.I),
    "shape": re.compile(r"^What shape (?:is|are) (?:the )?(.{1,40}?)\?$", re.I),
}
COUNT_PAT = re.compile(
    r"^How many (.{1,40}?) (?:are there|are visible|are shown|are in the (?:picture|photo|image))\?$",
    re.I)
POS_PAT = re.compile(
    r"^What (?:is|are) (to the left of|to the right of|behind|in front of) (?:the )?(.{1,40}?)\?$",
    re.I)


def scoped_candidates():
    """(origin, record, kind, object, attribute/relation) for everything parseable."""
    out = []
    for rec in read_records("vg_attributes"):
        out.append(("vg_attributes", rec, "attr", rec["object_name"],
                    DIM_NOUN.get(rec["attribute_dimension"], rec["attribute_dimension"])))
    for rec in read_records("tdiuc"):
        q = rec["request"]["fields"][0]["question"]
        if rec["family"] in ("color", "attribute"):
            for noun, pat in TDIUC_PAT.items():
                hit = pat.match(q)
                if hit:
                    out.append(("tdiuc", rec, "attr", hit.group(1).strip(), noun))
                    break
        elif rec["family"] == "counting":
            hit = COUNT_PAT.match(q)
            if hit:
                out.append(("tdiuc", rec, "count", hit.group(1).strip(), None))
        elif rec["family"] == "positional_reasoning":
            hit = POS_PAT.match(q)
            if hit and rec["request"]["fields"][0]["type"] == "choice":
                out.append(("tdiuc", rec, "pos", hit.group(2).strip(), hit.group(1).strip()))
    for rec in read_records("vqav2"):
        if rec["family"] != "color":
            continue
        hit = TDIUC_PAT["colour"].match(rec["request"]["fields"][0]["question"])
        if hit:
            out.append(("vqav2", rec, "attr", hit.group(1).strip(), "colour"))
    return out


def state_scoped(cands):
    rng.shuffle(cands)
    done = Counter()
    for origin, rec, kind, obj, attr in cands:
        bucket = bucket_of(rec["partition"])
        quota = (TEST_QUOTA if bucket == "test" else FIT_QUOTA)["state_scoped_question"]
        if done[bucket] >= quota:
            continue
        info = rec["images"][0]
        if info["sha256"] in used_sha or not obj or len(obj) > 60:
            continue
        group = (f"coco:{rec['source_group']}" if origin in ("tdiuc", "vqav2")
                 else f"{origin}:{rec['source_group']}")
        if not claim_group(group, rec["partition"]):
            continue
        src = rec["request"]["fields"][0]
        if kind == "attr":
            top, fn = ATTR_SHAPES[rng.randrange(len(ATTR_SHAPES))]
            state, op, ap = fn(obj, attr)
            text = rng.choice(ATTR_Q).format(o=op, a=ap)
            fid = slug(f"{attr}_of_target")
            rubric = RUBRIC["scoped_attr"] + (RUBRIC["color"][:1] if attr == "colour" else [])
        elif kind == "count":
            top, fn = COUNT_SHAPES[rng.randrange(len(COUNT_SHAPES))]
            state, op = fn(obj)
            text = rng.choice(COUNT_Q).format(o=op)
            fid = "visible_count"
            rubric = ()
        else:
            top, fn = POS_SHAPES[rng.randrange(len(POS_SHAPES))]
            state, op, rp = fn(obj, attr)
            text = rng.choice(POS_Q).format(o=op, rel=rp)
            fid = "object_at_relation"
            rubric = RUBRIC["scoped_attr"]
        question = instruct(rng, text, CONTEXT["scoped"], rubric)
        field = {"id": fid, "type": src["type"], "question": question}
        target = rec["target"]
        if src["type"] == "choice":
            opts = list(src["options"])
            rng.shuffle(opts)
            field["options"] = opts
        else:
            # Jev score criteria are ordered labels numbered from 0.
            levels, index = [], {}
            for i, lv in enumerate(src["levels"]):
                levels.append({"value": i, "description": lv["description"]})
                index[lv["value"]] = i
            field["levels"] = levels
            target = None if target is None else index[target]
            stats["style:ordinal_reindexed"] += 1
        used_sha.add(info["sha256"])
        done[bucket] += 1
        stats[f"scoped:{origin}:{kind}"] += 1
        stats["ref:top_level" if top else "ref:dotted"] += 1
        emit(f"ssq-{origin}-{rec['id'].split(':', 1)[1]}"[:120], "state_scoped_question",
             origin, state, field, target, rec["abstention_cause"], rec["source_answer"],
             [adopt(info)], rec["partition"], group, rec["source_split"], rec["license"])
    return done


# ===================================================================== family 3
QUERY_IN_Q = re.compile(r'"(.+)"')


def query_relevance():
    rows = read_records("sqid_esci")
    rng.shuffle(rows)
    done = Counter()
    for rec in rows:
        bucket = bucket_of(rec["partition"])
        quota = (TEST_QUOTA if bucket == "test" else FIT_QUOTA)["query_relevance"]
        if done[bucket] >= quota:
            continue
        hit = QUERY_IN_Q.search(rec["request"]["fields"][0]["question"])
        if not hit:
            continue
        query = hit.group(1).strip()
        info = rec["images"][0]
        if not query or len(query) > 200 or info["sha256"] in used_sha:
            continue
        group = f"sqid_esci:{rec['source_group']}"
        if not claim_group(group, rec["partition"]):
            continue
        pid = rec["source_group"]
        shape = rng.randrange(6)
        if shape == 0:
            state, qp, top = {"search": {"query": query, "locale": "us"}}, "search.query", False
        elif shape == 1:
            state, qp, top = ({"search": {"query": query, "locale": "us",
                                          "marketplace": "amazon.com"}}, "search.query", False)
        elif shape == 2:
            state, qp, top = ({"search": {"query": query}, "product": {"sku": pid}},
                              "search.query", False)
        elif shape == 3:
            state, qp, top = ({"session": {"search": {"query": query, "locale": "us"}},
                               "product": {"sku": pid}}, "session.search.query", False)
        elif shape == 4:
            state, qp, top = {"query": query, "locale": "us"}, "query", True
        else:
            state, qp, top = {"search_query": query, "sku": pid}, "search_query", True
        question = instruct(rng, rng.choice(ESCI_Q).format(q=qp), CONTEXT["esci"], RUBRIC["esci"])
        src = rec["request"]["fields"][0]
        short = rng.random() < 0.5
        levels, index = [], {}
        for i, lv in enumerate(src["levels"]):
            levels.append({"value": i, "description": ESCI_SHORT[lv["value"]] if short
                           else lv["description"]})
            index[lv["value"]] = i
        field = {"id": "query_relevance", "type": "ordinal", "question": question,
                 "levels": levels}
        used_sha.add(info["sha256"])
        done[bucket] += 1
        stats["ref:top_level" if top else "ref:dotted"] += 1
        stats["style:ordinal_reindexed"] += 1
        stats["esci_short_labels" if short else "esci_long_labels"] += 1
        emit(f"qr-{pid}", "query_relevance", "sqid_esci", state, field,
             None if rec["target"] is None else index[rec["target"]],
             rec["abstention_cause"], rec["source_answer"], [adopt(info)],
             rec["partition"], group, rec["source_split"], rec["license"])
    return done


# ===================================================================== family 4
def candidate_labels():
    pool = []
    for origin, families in (("marqo_gs", {"product_category"}),
                             ("fashion200k", {"apparel_category", "apparel_supercategory"}),
                             ("abo", {"product_category"})):
        for rec in read_records(origin):
            if rec["family"] in families and rec["request"]["fields"][0]["type"] == "choice":
                pool.append((origin, rec))
    rng.shuffle(pool)
    done = Counter()
    for origin, rec in pool:
        bucket = bucket_of(rec["partition"])
        quota = (TEST_QUOTA if bucket == "test" else FIT_QUOTA)["candidate_labels_in_state"]
        if done[bucket] >= quota:
            continue
        info = rec["images"][0]
        if info["sha256"] in used_sha:
            continue
        group = f"{origin}:{rec['source_group']}"
        if not claim_group(group, rec["partition"]):
            continue
        labels = [o["value"] for o in rec["request"]["fields"][0]["options"]]
        if not 2 <= len(labels) <= 25:
            continue
        state_labels = list(labels)
        rng.shuffle(state_labels)
        top, fn = LABEL_SHAPES[rng.randrange(len(LABEL_SHAPES))]
        state, cpath, holder = fn(state_labels)
        prior = rec["target"] is not None and rng.random() < PRIOR_RATE
        if prior:
            correct = stats["prior_right"] <= stats["prior_wrong"]
            others = [v for v in labels if v != rec["target"]]
            if not correct and not others:
                continue
            claim = rec["target"] if correct else rng.choice(others)
            if holder is None:
                state["seller_claimed_type"] = claim
                spath = "seller_claimed_type"
            else:
                state[holder]["seller_claimed_type"] = claim
                spath = f"{holder}.seller_claimed_type"
            text = rng.choice(PRIOR_Q).format(c=cpath, s=spath)
            stats["prior_right" if correct else "prior_wrong"] += 1
            fid = "actual_product_label"
        else:
            text = rng.choice(LABEL_Q).format(c=cpath)
            fid = "product_label"
        question = instruct(rng, text, CONTEXT["labels"], RUBRIC["labels"])
        opts = [{"value": v} for v in labels]
        rng.shuffle(opts)
        field = {"id": fid, "type": "choice", "question": question, "options": opts}
        used_sha.add(info["sha256"])
        done[bucket] += 1
        stats[f"labels:{origin}"] += 1
        stats["ref:top_level" if top else "ref:dotted"] += 1
        emit(f"cl-{origin}-{rec['id'].split(':', 1)[1]}"[:120], "candidate_labels_in_state",
             origin, state, field, rec["target"], rec["abstention_cause"],
             rec["source_answer"], [adopt(info)], rec["partition"], group,
             rec["source_split"], rec["license"])
    return done


# ===================================================================== main
BACKTICK = re.compile(r"`([^`]+)`")


def reference_style(question):
    refs = BACKTICK.findall(question)
    if not refs:
        return "none"
    return "dotted" if any("." in r for r in refs) else "top_level"


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    abo_rows = read_records("abo")
    wanted = {hit.group(1) for hit in
              (ID_ITEM.match(r["id"].split(":", 1)[1]) for r in abo_rows) if hit}
    print(f"abo items referenced by records: {len(wanted)}", flush=True)
    meta = load_abo_meta(wanted)
    print(f"abo listings with title + curated product type: {len(meta)}", flush=True)

    listing_verification(abo_rows, meta)
    print(f"after listing_verification: {len(records)}", flush=True)
    state_scoped(scoped_candidates())
    print(f"after state_scoped_question: {len(records)}", flush=True)
    query_relevance()
    candidate_labels()

    records.sort(key=lambda r: r["id"])
    keep = {im["sha256"] for r in records for im in r["images"]}
    for path in IMG_DIR.glob("*.jpg"):          # links left by an earlier run
        if path.stem not in keep:
            path.unlink()
            stats["images_unlinked"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))

    boolean = [r for r in records if r["request"]["fields"][0]["type"] == "boolean"]
    styles = Counter(reference_style(r["request"]["fields"][0]["question"]) for r in records)
    summary = {
        "records": len(records), "seed": SEED,
        "partitions": dict(Counter(r["partition"] for r in records)),
        "families": dict(Counter(r["family"] for r in records)),
        "family_partition": {f: dict(Counter(r["partition"] for r in records if r["family"] == f))
                             for f in sorted({r["family"] for r in records})},
        "family_field_type": {f: dict(Counter(r["request"]["fields"][0]["type"]
                                              for r in records if r["family"] == f))
                              for f in sorted({r["family"] for r in records})},
        "origins": dict(Counter(r["origin"] for r in records)),
        "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in records)),
        "field_ids": dict(Counter(r["request"]["fields"][0]["id"] for r in records)),
        "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
        "abstention_share": round(sum(r["target"] is None for r in records) / max(1, len(records)), 4),
        "boolean_balance": dict(Counter(str(r["target"]) for r in boolean)),
        "boolean_balance_by_family": {
            f: dict(Counter(str(r["target"]) for r in boolean if r["family"] == f))
            for f in sorted({r["family"] for r in boolean})},
        "reference_style": dict(styles),
        "two_image": sum(len(r["images"]) == 2 for r in records),
        "images": len({im["sha256"] for r in records for im in r["images"]}),
        "state_bytes_max": max(len(json.dumps(r["request"]["state"]).encode()) for r in records),
        "option_counts": dict(sorted(Counter(
            len(r["request"]["fields"][0].get("options", r["request"]["fields"][0].get("levels", [])))
            for r in records).items())),
        "stats": dict(sorted(stats.items())),
    }
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
