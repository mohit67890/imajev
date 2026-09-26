"""Phase-3 Stage 0-I: joint image + record rule generator (family `image_joint_rule`, source I).

A licensed photo whose attributes are KNOWN from its source annotations, plus a generated record and a rule with a
threshold, an exception or two steps. The gold follows from the record and the photo's known attributes, so it is
constructed, not guessed (docs/phase-3-plan.md rev 4, item 2 and Stage 0 source I).

Photo facts used (only fields the source annotates reliably):
- ABO listing photos (data/decision-v1/abo, CC BY 4.0): curated product type (+ its catalogue group), colour (never
  "multicoloured"/"transparent"), material collapsed to visually distinct groups (wood, metal, glass, ...), shape
  collapsed to round / rectangular, pattern solid / patterned. Only listings whose v1 records are all train.
- Defect photos (data/decision-v1/defects: VisA + DAGM CC BY 4.0, BTAD CC BY-SA 4.0): normal / anomalous, and for VisA
  the single annotated defect kind. Only the train-partition categories.

Rule kinds (provenance.rule_kind): threshold_rule, rule_exception, multi_step_rule, two_image (pairs of photos with known
attributes: two ABO listings, two units of one defect category). ~15% unknown variants (gold null, insufficient_evidence):
the decisive record field is removed, the lookup row for the photo's value is missing, or the rule turns on an attribute
no photo can show (weight, origin, an X-ray result) and nothing else settles it. A rule on a hidden attribute that the
rest of the rule already settles keeps its gold (these are emitted as ordinary items).

Every row stores provenance.template / params / photo_facts, so tests/test_p3_images.py re-derives the gold from the
source labels (`decide`).

    .venv/bin/python scripts/p3/gen_image_joint.py [--n 15000] [--out data/p3/candidates/I-joint.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402

SEED = "p3-image-joint-v1"
OUT = ROOT / "data" / "p3" / "candidates" / "I-joint.jsonl"
ABO_META = ROOT / ".cache" / "datasets" / "v1" / "state_aware" / "abo_listing_meta.json"
ABO_META_V2 = ROOT / ".cache" / "datasets" / "v2" / "state_grounded" / "abo_listing_meta.json"
ABO_RECORDS = ROOT / "data" / "decision-v1" / "abo" / "records.jsonl"
DEFECT_RECORDS = ROOT / "data" / "decision-v1" / "defects" / "records.jsonl"
EXCLUSIONS = ROOT / "data" / "decision-v1" / "exclusions.json"
BENCH_DIR = ROOT / "data" / "imajev-bench"
DECONTAM_META = ROOT / "data" / "p3" / "decontam-index" / "meta.json"

LICENCE_EVIDENCE = {
    ("abo", "CC-BY-4.0"): "data/decision-v2/licenses/state_grounded/abo-LICENSE-CC-BY-4.0.txt",
    ("defects", "CC-BY-4.0"): "data/provenance/licenses/v1-image/defects/CC-BY-4.0.md",
    ("defects", "CC-BY-SA-4.0"): "data/provenance/licenses/v1-image/defects/CC-BY-SA-4.0.md",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


# ------------------------------------------------------------------------------------------------ shared helpers
def rng_for(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_BENCH: set[str] | None = None


def bench_image_shas() -> set[str]:
    """sha256 of every image file under data/imajev-bench (all versions, thumbnails, pilots, private sets) plus the
    image hashes in the decontamination index. Stricter than the decontam filter on purpose."""
    global _BENCH
    if _BENCH is None:
        out = set()
        for p in BENCH_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                out.add(sha256_file(p))
        if DECONTAM_META.is_file():
            out |= set(json.loads(DECONTAM_META.read_text()).get("images", {}))
        _BENCH = out
    return _BENCH


def data_rel(path: str) -> str:
    """Candidate image paths are relative to data/."""
    return path[5:] if path.startswith("data/") else path


def plural(noun: str) -> str:
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def join_or(xs: list[str]) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def ref(rng, prefix: str) -> str:
    return f"{prefix}-{rng.randint(10000, 99999)}"


def money(x: float) -> str:
    return f"{x:.2f}"


# ------------------------------------------------------------------------------------------------ photo facts
COLOUR_CONFUSABLE = {
    "black": {"grey", "navy blue", "brown"}, "grey": {"black", "silver", "white", "beige"},
    "silver": {"grey", "white"}, "white": {"ivory", "off-white", "beige", "grey", "silver"},
    "ivory": {"white", "off-white", "beige"}, "off-white": {"white", "ivory", "beige"},
    "beige": {"ivory", "off-white", "white", "tan", "brown", "grey"}, "brown": {"tan", "bronze", "beige", "black"},
    "tan": {"brown", "beige"}, "gold": {"yellow", "bronze", "tan"}, "yellow": {"gold", "orange"},
    "bronze": {"gold", "brown"}, "blue": {"navy blue", "light blue", "turquoise"}, "navy blue": {"blue", "black"},
    "light blue": {"blue", "turquoise"}, "turquoise": {"blue", "green", "light blue"}, "green": {"turquoise"},
    "red": {"pink", "burgundy", "orange"}, "pink": {"red", "purple"}, "burgundy": {"red", "purple"},
    "orange": {"red", "yellow"}, "purple": {"pink", "burgundy"},
}
RULE_COLOURS = ["black", "white", "grey", "brown", "blue", "red", "pink", "green", "beige", "silver", "gold", "yellow",
                "purple", "orange"]
MATERIAL_GROUP = {
    "wood": "wood", "engineered wood": "wood", "bamboo": "wood",
    "metal": "metal", "stainless steel": "metal", "steel": "metal", "aluminium": "metal", "iron": "metal",
    "brass": "metal", "glass": "glass", "leather": "leather", "faux leather": "leather", "suede": "leather",
    "fabric": "fabric", "cotton": "fabric", "polyester": "fabric", "wool": "fabric", "velvet": "fabric",
    "linen": "fabric", "canvas": "fabric", "nylon": "fabric", "ceramic": "ceramic", "stoneware": "ceramic",
    "porcelain": "ceramic", "stone": "stone", "marble": "stone", "silicone": "plastic", "plastic": "plastic",
    "polypropylene": "plastic", "rubber": "plastic",
}
MATERIAL_PHRASE = {"wood": "wood", "metal": "metal", "glass": "glass", "leather": "leather or leather-look material",
                   "fabric": "fabric", "ceramic": "ceramic", "stone": "stone", "plastic": "plastic or silicone"}
MATERIAL_CONFUSABLE = {"leather": {"plastic", "fabric"}, "plastic": {"leather", "metal", "glass"},
                       "metal": {"plastic"}, "ceramic": {"stone", "glass"}, "stone": {"ceramic"},
                       "glass": {"plastic", "ceramic"}, "fabric": {"leather"}, "wood": set()}
SHAPE_GROUP = {"round": "round", "oval": "round", "drum": "round", "cylindrical": "round",
               "rectangular": "rectangular", "runner": "rectangular", "square": "rectangular"}
PATTERN_GROUP = {"solid": "solid", "geometric": "patterned", "floral": "patterned", "striped": "patterned",
                 "plaid": "patterned", "colour block": "patterned", "moroccan": "patterned",
                 "animal print": "patterned", "trellis": "patterned", "ikat": "patterned", "paisley": "patterned",
                 "medallion": "patterned", "checked": "patterned", "chevron": "patterned", "polka dot": "patterned"}
ABO_ID = re.compile(
    r"^(?:cat|color|material|pattern|item_shape|finish|style|same|reference_color|reference_material"
    r"|reference_category|bool-(?:color|material|pattern|item_shape|category))-([A-Z0-9]{10})-")
ABO_PER_TYPE = 700

DEFECT_OBJECT = {
    "visa:candle": "candle", "visa:cashew": "cashew nut", "visa:fryum": "fryum snack", "visa:macaroni1": "macaroni piece",
    "visa:macaroni2": "macaroni piece", "visa:pcb1": "circuit board", "visa:pcb2": "circuit board",
    "visa:pcb3": "circuit board", "visa:pcb4": "circuit board", "dagm:1": "textured panel", "dagm:2": "textured panel",
    "dagm:3": "textured panel", "dagm:4": "textured panel", "btad:01": "machined part", "btad:02": "machined part",
}

# Attributes no product photo can show (the "photo cannot show" unknowns).
HIDDEN_ABO = [("weighs more than 2 kg", "weight"), ("was made in Portugal", "country of origin"),
              ("carries a five-year warranty", "warranty"), ("was bought with a gift card", "payment method"),
              ("is covered by the extended-care plan", "care plan"), ("was restocked this month", "restock date")]
HIDDEN_DEFECT = [("failed the internal X-ray scan", "X-ray result"), ("is from a supplier batch under recall", "batch recall"),
                 ("failed the seal-pressure test", "pressure test"), ("was weighed under the minimum", "weight check")]


def load_abo_photos(bench: set[str]) -> list[dict]:
    meta_path = ABO_META_V2 if ABO_META_V2.is_file() else ABO_META
    meta = json.loads(meta_path.read_text())
    banned = set(json.loads(EXCLUSIONS.read_text()).get("sha256", [])) if EXCLUSIONS.is_file() else set()
    by_item, non_train = {}, set()
    for line in open(ABO_RECORDS):
        row = json.loads(line)
        m = ABO_ID.match(row["id"].split(":", 1)[1])
        if not m:
            continue
        item = m.group(1)
        if row["partition"] != "train":
            non_train.add(item)
            continue
        by_item.setdefault(item, row["images"][0])
    photos = []
    for item, image in sorted(by_item.items()):
        if item in non_train or item not in meta or image["sha256"] in banned or image["sha256"] in bench:
            continue
        if not (ROOT / image["image"]).is_file():
            continue
        facts = abo_facts(meta[item])
        if len(facts) < 3:        # product type + group + at least one visual attribute
            continue
        photos.append({"source": "abo", "item": item, "image": data_rel(image["image"]), "sha256": image["sha256"],
                       "licence": "CC-BY-4.0", "facts": facts})
    per_type = defaultdict(list)
    for p in photos:
        per_type[p["facts"]["product_type"]].append(p)
    out = []
    for t in sorted(per_type):
        rows = per_type[t]
        rng_for(SEED, "abo-cap", t).shuffle(rows)
        out += rows[:ABO_PER_TYPE]
    return out


def abo_facts(info: dict) -> dict:
    a = info["attrs"]
    f = {"product_type": a["product_type"], "cgroup": info["cgroup"]}
    c = a.get("color")
    if c and c not in ("multicoloured", "transparent"):
        f["colour"] = c
    if a.get("material") in MATERIAL_GROUP:
        f["material"] = MATERIAL_GROUP[a["material"]]
    if a.get("shape") in SHAPE_GROUP:
        f["shape"] = SHAPE_GROUP[a["shape"]]
    if a.get("pattern") in PATTERN_GROUP:
        f["pattern"] = PATTERN_GROUP[a["pattern"]]
    return f


def load_defect_photos(bench: set[str]) -> list[dict]:
    kinds = {}
    for line in open(DEFECT_RECORDS):
        r = json.loads(line)
        if r["family"] == "defect_type" and r["partition"] == "train" and r["target"]:
            kinds[r["images"][0]["sha256"]] = r["target"]
    photos, seen = [], set()
    for line in open(DEFECT_RECORDS):
        r = json.loads(line)
        if r["family"] != "defect_presence" or r["partition"] != "train" or r["source_group"] not in DEFECT_OBJECT:
            continue
        im = r["images"][0]
        if im["sha256"] in seen or im["sha256"] in bench or not (ROOT / im["image"]).is_file():
            continue
        seen.add(im["sha256"])
        defective = r["source_answer"] != "normal"
        facts = {"category": r["source_group"], "object": DEFECT_OBJECT[r["source_group"]], "defective": defective}
        if defective and im["sha256"] in kinds:
            facts["defect_kind"] = kinds[im["sha256"]]
        photos.append({"source": "defects", "item": r["id"], "image": data_rel(im["image"]), "sha256": im["sha256"],
                       "licence": r["license"], "facts": facts})
    return photos


def defect_vocab(photos) -> dict[str, list[str]]:
    v = defaultdict(set)
    for p in photos:
        k = p["facts"].get("defect_kind")
        if k:
            v[p["facts"]["category"]].add(k)
    return {c: sorted(s) for c, s in v.items()}


# ------------------------------------------------------------------------------------------------ predicates
TYPE_GROUP: dict[str, str] = {}      # ABO product type -> catalogue group (filled by build / load_context)


def pred_cond(pred: dict) -> str:
    """A verb phrase the pictured item satisfies or not, e.g. 'is black or red'."""
    if pred.get("hidden"):
        return pred["hidden"]
    a, vals = pred["attr"], pred["values"]
    if a == "colour":
        return f"is {join_or(vals)}"
    if a == "material":
        return f"is made of {join_or([MATERIAL_PHRASE[v] for v in vals])}"
    if a == "product_type":
        q = [f'"{v}"' for v in vals]
        return f"is of the product type {q[0]}" if len(q) == 1 else f"is of one of the product types {join_or(q)}"
    if a == "shape":
        return "is round or oval" if vals[0] == "round" else "is rectangular or square"
    if a == "pattern":
        return "has a printed or woven pattern" if vals[0] == "patterned" else "is plain, with no pattern"
    raise KeyError(a)


def pred_true(pred: dict, facts: dict):
    if pred.get("hidden"):
        return None
    return facts[pred["attr"]] in pred["values"]


def pred_clear(pred: dict, facts: dict) -> bool:
    """False when the photo does NOT meet the predicate but sits close to a listed value (grey vs black, boot vs shoe),
    so that a careful reader could honestly disagree with the label."""
    if pred.get("hidden"):
        return True
    a = pred["attr"]
    if a not in facts:
        return False
    v = facts[a]
    if v in pred["values"]:
        return True
    if a == "colour":
        return not any(v in COLOUR_CONFUSABLE.get(x, set()) or x in COLOUR_CONFUSABLE.get(v, set()) for x in pred["values"])
    if a == "material":
        return not any(x in MATERIAL_CONFUSABLE.get(v, set()) or v in MATERIAL_CONFUSABLE.get(x, set()) for x in pred["values"])
    if a == "product_type":
        return all(TYPE_GROUP.get(x) != facts.get("cgroup") for x in pred["values"])
    return True


def make_pred(rng, facts: dict, want: bool, attrs=None) -> dict | None:
    """A predicate on an attribute the photo's source annotates; `want` = whether the photo satisfies it."""
    avail = [a for a in (attrs or ("colour", "colour", "material", "product_type", "shape", "pattern")) if a in facts]
    if not avail:
        return None
    a = rng.choice(avail)
    v = facts[a]
    if a == "colour":
        pool = [c for c in RULE_COLOURS if c != v and c not in COLOUR_CONFUSABLE.get(v, set())]
        if want:
            vals = [v] + (rng.sample(pool, 1) if rng.random() < 0.5 else [])
        else:
            vals = rng.sample(pool, rng.choice((1, 2)))
    elif a == "material":
        pool = [m for m in MATERIAL_PHRASE if m != v and m not in MATERIAL_CONFUSABLE.get(v, set())]
        vals = [v] if want else [rng.choice(pool)]
    elif a == "product_type":
        other = [t for t, g in TYPE_GROUP.items() if g != facts["cgroup"]]
        vals = [v] if want else rng.sample(other, rng.choice((1, 2)))
    elif a == "shape":
        vals = [v] if want else [{"round": "rectangular", "rectangular": "round"}[v]]
    else:
        vals = [v] if want else [{"solid": "patterned", "patterned": "solid"}[v]]
    rng.shuffle(vals)
    return {"attr": a, "values": vals}


def hidden_pred(rng, bank) -> dict:
    return {"hidden": rng.choice(bank)[0]}


# ------------------------------------------------------------------------------------------------ templates
# sample(rng, photos, ctx) -> params | None; render(params) -> (state, field); decide(params, facts_list, assume) -> gold.
# `assume` maps a hidden predicate's index to a bool; params["_withheld"] lists the record fields (or lookup rows) the
# state leaves out. decide() below enumerates both to tell a settled gold from an unknown.

def noul(q):
    return {"type": "noul", "question": q}


def choice(question, opts):
    return {"type": "choice", "question": question,
            "options": [{"key": k, "text": t, "description": d} for k, t, d in opts]}


def W(p) -> set:
    return set(p.get("_withheld") or ())


def P(p, i, fl, assume):
    pred = p["preds"][i]
    if pred.get("hidden"):
        return assume.get(i)
    return pred_true(pred, fl[pred.get("photo", 0) or 0])


# -- ABO 1: return window with a clearance exception (rule_exception, noul)
def t_return_sample(rng, photos, ctx):
    f = photos[0]["facts"]
    pred = hidden_pred(rng, HIDDEN_ABO) if ctx.get("hidden") else make_pred(rng, f, rng.random() < 0.6)
    if pred is None:
        return None
    w = rng.choice((14, 21, 30, 45, 60))
    return {"preds": [pred], "window": w, "days": rng.randint(1, w) if rng.random() < 0.9 else rng.randint(w + 1, w + 40),
            "clearance": rng.random() < 0.85, "tier": rng.choice(("Gold", "Plus", "Premier", "Circle")),
            "member": rng.random() < 0.15, "order": ref(rng, "ORD"), "rma": ref(rng, "RMA")}


def t_return_render(p):
    c, tier, w = pred_cond(p["preds"][0]), p["tier"], p["window"]
    rule = rng_for(p["rma"]).choice([
        f"Returns are accepted up to {w} days after delivery. A clearance-sale purchase is final sale when the item {c}, unless the customer holds {tier} membership.",
        f"We take items back within {w} days of delivery. Exception: if the item {c} and it was bought in the clearance sale, it cannot be returned, except by {tier} members.",
        f"Return window: {w} days from delivery. Clearance purchases are excluded from returns when the item {c}; {tier} members are exempt from that exclusion.",
    ])
    req = {"reference": p["rma"], "order": p["order"]}
    if "days" not in W(p):
        req["days_since_delivery"] = p["days"]
    if "clearance" not in W(p):
        req["bought_in_clearance_sale"] = p["clearance"]
    if "member" not in W(p):
        req["membership"] = tier if p["member"] else "none"
    q = rng_for(p["rma"], "q").choice(["Should this return be accepted under the policy?",
                                       "Does the return policy allow this return?",
                                       "Can the pictured item be taken back under this policy?"])
    return {"policy": rule, "return_request": req, "photo": "the item being returned"}, noul(q)


def t_return_decide(p, fl, assume):
    if p["days"] > p["window"]:
        return False
    return not (P(p, 0, fl, assume) and p["clearance"] and not p["member"])


# -- ABO 2: free-delivery threshold that depends on the item (threshold_rule, noul)
def t_delivery_sample(rng, photos, ctx):
    pred = make_pred(rng, photos[0]["facts"], rng.random() < 0.5)
    if pred is None:
        return None
    lo = rng.choice((25, 30, 40, 50, 60))
    hi = lo + rng.choice((20, 30, 40, 50))
    qty = rng.choice((1, 1, 2, 3, 4))
    r = rng.random()
    total = rng.uniform(lo + 1, hi - 1) if r < 0.7 else (rng.uniform(5, lo - 1) if r < 0.85 else rng.uniform(hi + 1, hi + 60))
    return {"preds": [pred], "low": lo, "high": hi, "qty": qty, "unit_price": round(total / qty, 2),
            "cheap_side": rng.random() < 0.5, "order": ref(rng, "ORD")}


def t_delivery_render(p):
    lo, hi, c = p["low"], p["high"], pred_cond(p["preds"][0])
    if p["cheap_side"]:
        rule = f"Delivery is free when the order value (unit price x quantity) is at least {lo} if the item {c}, and at least {hi} otherwise."
    else:
        rule = f"Delivery is free once the order value (unit price x quantity) reaches {hi} if the item {c}; any other item needs only {lo}."
    order = {"reference": p["order"], "unit_price": money(p["unit_price"])}
    if "qty" not in W(p):
        order["quantity"] = p["qty"]
    q = rng_for(p["order"]).choice(["Does this order get free delivery?", "Is delivery free for this order under the rule?",
                                    "Will the customer pay nothing for delivery on this order?"])
    return {"rule": rule, "order": order, "photo": "the item ordered"}, noul(q)


def t_delivery_decide(p, fl, assume):
    pv = P(p, 0, fl, assume)
    thr = (p["low"] if pv else p["high"]) if p["cheap_side"] else (p["high"] if pv else p["low"])
    return round(p["unit_price"] * p["qty"], 2) >= thr


# -- ABO 3: handling fee looked up by product type, then a budget band (multi_step_rule, choice)
def t_handling_sample(rng, photos, ctx):
    f = photos[0]["facts"]
    others = sorted(t for t, g in TYPE_GROUP.items() if g != f["cgroup"])
    types = rng.sample(others, rng.randint(3, 5)) + [f["product_type"]]
    rng.shuffle(types)
    fees = {t: rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 15)) for t in types}
    qty = rng.randint(2, 12)
    total = fees[f["product_type"]] * qty
    budget = max(5, int(total * rng.choice((0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5))))
    return {"preds": [], "fees": fees, "qty": qty, "budget": budget, "margin": rng.choice((10, 20, 25)), "po": ref(rng, "PO")}


def t_handling_render(p):
    fees = {t: v for t, v in p["fees"].items() if t not in W(p)}
    rule = (f"Handling cost = the per-unit fee for the pictured item's product type x the quantity. Book the shipment when "
            f"the handling cost is within the budget; when it is over the budget by at most {p['margin']}%, book it only "
            f"with a manager's sign-off; otherwise decline it.")
    q = rng_for(p["po"]).choice(["What should happen to this shipment?", "Which action does the handling rule give?",
                                 "How should the shipment of the pictured item be handled?"])
    field = choice(q, [("book", "book the shipment", "handling cost within budget"),
                       ("sign_off", "book with manager sign-off", f"over budget by no more than {p['margin']}%"),
                       ("decline", "decline the shipment", f"over budget by more than {p['margin']}%")])
    ship = {"reference": p["po"], "handling_budget": p["budget"]}
    if "qty" not in W(p):
        ship["quantity"] = p["qty"]
    return {"rule": rule, "handling_fee_per_unit": fees, "shipment": ship, "photo": "one unit of the item shipped"}, field


def t_handling_decide(p, fl, assume):
    t = fl[0]["product_type"]
    if t in W(p):
        return None
    total = p["fees"][t] * p["qty"]
    if total <= p["budget"]:
        return "book"
    return "sign_off" if total <= p["budget"] * (1 + p["margin"] / 100) else "decline"


# -- ABO 4: priority from two photo attributes and one record field (multi_step_rule, score 0-3)
def t_priority_sample(rng, photos, ctx):
    f = photos[0]["facts"]
    attrs = [a for a in ("colour", "material", "product_type", "shape", "pattern") if a in f]
    if len(attrs) < 2:
        return None
    a1, a2 = rng.sample(attrs, 2)
    preds = [make_pred(rng, f, rng.random() < 0.5, attrs=[a1]), make_pred(rng, f, rng.random() < 0.5, attrs=[a2])]
    if ctx.get("hidden"):
        preds[rng.randrange(2)] = hidden_pred(rng, HIDDEN_ABO)
    n = rng.choice((3, 5, 7, 10))
    return {"preds": preds, "wait_limit": n, "days_waiting": rng.randint(0, 2 * n), "ticket": ref(rng, "TCK")}


def t_priority_render(p):
    a, b = (pred_cond(x) for x in p["preds"])
    rule = (f"Priority starts at 0. Add 1 if the pictured item {a}. Add 1 if it {b}. "
            f"Add 1 if the request has waited more than {p['wait_limit']} days.")
    req = {"ticket": p["ticket"]}
    if "days_waiting" not in W(p):
        req["days_waiting"] = p["days_waiting"]
    q = rng_for(p["ticket"]).choice(["What priority does this request get?", "Which priority level does the rule assign?"])
    field = {"type": "score", "question": q,
             "levels": [{"value": i, "description": d} for i, d in
                        enumerate(("0: no condition met", "1: one condition met", "2: two conditions met", "3: all three conditions met"))]}
    return {"rule": rule, "request": req, "photo": "the item the request is about"}, field


def t_priority_decide(p, fl, assume):
    return int(P(p, 0, fl, assume)) + int(P(p, 1, fl, assume)) + int(p["days_waiting"] > p["wait_limit"])


# -- ABO 5: pallet freight (AND) or delivery signature (OR) with a quantity threshold (threshold_rule, noul)
def t_freight_sample(rng, photos, ctx):
    pred = hidden_pred(rng, HIDDEN_ABO) if ctx.get("hidden") else make_pred(rng, photos[0]["facts"], rng.random() < 0.5)
    if pred is None:
        return None
    mode = rng.choice(("and", "or"))
    q = rng.choice((3, 4, 5, 6, 10))
    qty = rng.choice((q - 1, q, q, q + 2, q + 5, 1)) if mode == "and" else rng.choice((1, 2, q - 1, q - 1, q, q + 3))
    return {"preds": [pred], "mode": mode, "limit": q, "qty": max(1, qty), "order": ref(rng, "ORD")}


def t_freight_render(p):
    c = pred_cond(p["preds"][0])
    if p["mode"] == "and":
        rule = f"An order goes by pallet freight when the item ordered {c} and the quantity is {p['limit']} or more; every other order goes by parcel."
        q = "Will this order go by pallet freight?"
    else:
        rule = f"A signature on delivery is required when the item ordered {c}, or when the quantity is {p['limit']} or more."
        q = "Is a signature on delivery required for this order?"
    order = {"reference": p["order"]}
    if "qty" not in W(p):
        order["quantity"] = p["qty"]
    return {"rule": rule, "order": order, "photo": "the item ordered"}, noul(q)


def t_freight_decide(p, fl, assume):
    big = p["qty"] >= p["limit"]
    if p["mode"] == "and":
        return bool(big and P(p, 0, fl, assume))
    return bool(big or P(p, 0, fl, assume))


# -- ABO 6: two offers, one photo each (two_image, choice)
def t_offers_sample(rng, photos, ctx):
    pred = make_pred(rng, rng.choice(photos)["facts"], rng.random() < 0.6, attrs=[ctx["pair_attr"]])
    if pred is None:
        return None
    pred["photo"] = "all"
    cap = rng.choice((40, 60, 80, 100, 150))
    p1, p2 = (round(rng.uniform(cap * 0.5, cap * 1.2), 2) for _ in range(2))
    return {"preds": [pred], "cap": cap, "price_1": p1, "price_2": p2, "req": ref(rng, "REQ")}


def t_offers_render(p):
    c = pred_cond(p["preds"][0])
    rule = (f"Buy from the cheaper of the offers whose photo shows an item that {c}, provided that offer costs at most "
            f"{p['cap']}. If no offer qualifies, buy nothing.")
    offers = {}
    for i in (1, 2):
        o = {"seller": f"seller {chr(64 + i)}"}
        if f"price_{i}" not in W(p):
            o["price"] = money(p[f"price_{i}"])
        offers[f"offer_{i}"] = o
    field = choice(rng_for(p["req"]).choice(["Which offer should be bought?", "Which offer does the rule pick?"]),
                   [("offer_1", "offer 1 (Image 1)", "the item in the first photo"),
                    ("offer_2", "offer 2 (Image 2)", "the item in the second photo"),
                    ("neither", "buy nothing", "no offer qualifies under the rule")])
    return {"rule": rule, "request": p["req"], **offers, "photos": "Image 1 is offer 1, Image 2 is offer 2"}, field


def t_offers_decide(p, fl, assume):
    ok = [i for i in (1, 2) if pred_true(p["preds"][0], fl[i - 1]) and p[f"price_{i}"] <= p["cap"]]
    if not ok:
        return "neither"
    if len(ok) == 2 and p["price_1"] == p["price_2"]:
        return None
    return f"offer_{min(ok, key=lambda i: p[f'price_{i}'])}"


# -- DEF 1: lot sampling threshold (threshold_rule, noul)
def t_lot_sample(rng, photos, ctx):
    t = rng.choice((2, 3, 4, 5))
    k = t - 1 if rng.random() < 0.6 else rng.randint(0, t + 1)
    return {"preds": [], "limit": t, "found": k, "sampled": rng.randint(t + 3, 30), "lot": ref(rng, "LOT"),
            "ask_accept": rng.random() < 0.4, "object": photos[0]["facts"]["object"]}


def t_lot_render(p):
    obj = p.get("object", "unit")
    rule = (f"Sampling plan: the lot is rejected once {p['limit']} or more of the sampled units show a visible defect. "
            f"The photo is the last unit sampled; `defective_so_far` counts the units sampled before it.")
    lot = {"lot": p["lot"], "product": obj, "units_sampled": p["sampled"]}
    if "found" not in W(p):
        lot["defective_so_far"] = p["found"]
    q = "Is the lot accepted?" if p["ask_accept"] else "Is the lot rejected?"
    return {"rule": rule, "inspection": lot, "photo": "the last sampled unit"}, noul(q)


def t_lot_decide(p, fl, assume):
    rej = p["found"] + int(fl[0]["defective"]) >= p["limit"]
    return (not rej) if p["ask_accept"] else rej


# -- DEF 2: ship / rework / scrap with an exception (rule_exception, choice)
def t_rework_sample(rng, photos, ctx):
    lines = [f"line {c}" for c in "ABCDEFG"]
    rl = sorted(rng.sample(lines, rng.randint(1, 3)))
    line = rng.choice(rl) if rng.random() < 0.6 else rng.choice(lines)
    preds = [hidden_pred(rng, HIDDEN_DEFECT)] if ctx.get("hidden") else []
    return {"preds": preds, "rework_lines": rl, "line": line, "reworked": rng.random() < 0.35, "unit": ref(rng, "SN"),
            "object": photos[0]["facts"]["object"]}


def t_rework_render(p):
    extra = f" A unit that {p['preds'][0]['hidden']} is scrapped whatever the photo shows." if p["preds"] else ""
    rule = ("A unit with no visible defect ships. A unit with a visible defect is scrapped, except that it goes to rework "
            "when it has never been reworked and its line is on the rework list." + extra)
    unit = {"serial": p["unit"], "product": p["object"]}
    if "line" not in W(p):
        unit["line"] = p["line"]
    if "reworked" not in W(p):
        unit["reworked_before"] = p["reworked"]
    field = choice(rng_for(p["unit"]).choice(["Where does the pictured unit go?", "What happens to this unit under the rule?"]),
                   [("ship", "ship it", "no visible defect"), ("rework", "send it to rework", "defective, but the exception applies"),
                    ("scrap", "scrap it", "defective, and the exception does not apply")])
    return {"rule": rule, "rework_list": p["rework_lines"], "unit": unit, "photo": "the unit"}, field


def t_rework_decide(p, fl, assume):
    if p["preds"] and P(p, 0, fl, assume):
        return "scrap"
    if not fl[0]["defective"]:
        return "ship"
    return "rework" if (not p["reworked"] and p["line"] in p["rework_lines"]) else "scrap"


# -- DEF 3: severity points by defect kind, then a running-total threshold (multi_step_rule, noul); VisA only
def t_points_sample(rng, photos, ctx):
    f = photos[0]["facts"]
    vocab = ctx["vocab"].get(f["category"])
    if not vocab or len(vocab) < 2 or (f["defective"] and "defect_kind" not in f):
        return None
    pts = {k: rng.choice((1, 2, 3, 4, 5)) for k in vocab}
    lim = rng.choice((6, 8, 10, 12))
    mine = pts.get(f.get("defect_kind"), 0)
    base = lim - mine if rng.random() < 0.5 else lim - rng.randint(1, 5)
    return {"preds": [], "points": pts, "limit": lim, "total": max(0, base), "shift": ref(rng, "SHIFT"), "object": f["object"]}


def t_points_render(p):
    pts = {k: v for k, v in p["points"].items() if k not in W(p)}
    rule = (f"Each inspected {p['object']} adds the points listed for its visible defect kind (0 when it shows no visible "
            f"defect) to the shift's running total. The line halts once the total reaches {p['limit']}.")
    shift = {"shift": p["shift"]}
    if "total" not in W(p):
        shift["points_so_far"] = p["total"]
    q = rng_for(p["shift"]).choice(["Does the line halt after the pictured unit?", "Will this unit bring the line to a halt?"])
    return {"rule": rule, "points_per_defect_kind": pts, "shift": shift, "photo": "the unit just inspected"}, noul(q)


def t_points_decide(p, fl, assume):
    k = fl[0].get("defect_kind")
    if fl[0]["defective"]:
        if k in W(p):
            return None
        add = p["points"][k]
    else:
        add = 0
    return p["total"] + add >= p["limit"]


# -- DEF 4: two units of one category (two_image, choice)
def t_units_sample(rng, photos, ctx):
    preds = [hidden_pred(rng, HIDDEN_DEFECT)] if ctx.get("hidden") else []
    return {"preds": preds, "w1": rng.random() < 0.75, "w2": rng.random() < 0.75, "batch": ref(rng, "BATCH"),
            "object": photos[0]["facts"]["object"]}


def t_units_render(p):
    extra = f" A unit that {p['preds'][0]['hidden']} is held back." if p["preds"] else ""
    rule = f"A unit ships when its photo shows no visible defect and it passed the weight check.{extra}"
    units = {}
    for i in (1, 2):
        u = {"serial": f"{p['batch']}-{i}", "product": p["object"]}
        if f"w{i}" not in W(p):
            u["weight_check"] = "pass" if p[f"w{i}"] else "fail"
        units[f"unit_{i}"] = u
    field = choice(rng_for(p["batch"]).choice(["Which of the two units ship?", "Which units does the rule let ship?"]),
                   [("unit_1", "only unit 1 (Image 1)", ""), ("unit_2", "only unit 2 (Image 2)", ""),
                    ("both", "both units", ""), ("neither", "neither unit", "")])
    return {"rule": rule, **units, "photos": "Image 1 is unit 1, Image 2 is unit 2"}, field


def t_units_decide(p, fl, assume):
    ships = []
    for i in (1, 2):
        ok = (not fl[i - 1]["defective"]) and p[f"w{i}"]
        if p["preds"]:
            ok = ok and not assume.get(f"u{i}", False)
        ships.append(bool(ok))
    return {(True, True): "both", (True, False): "unit_1", (False, True): "unit_2", (False, False): "neither"}[tuple(ships)]


TEMPLATES = {
    # name: (domain, n_photos, rule_kind, difficulty, sample, render, decide, withholdable record fields -> alternatives)
    "abo_return_exception": ("abo", 1, "rule_exception", 4, t_return_sample, t_return_render, t_return_decide,
                             {"days": [1, 400], "clearance": [True, False], "member": [True, False]}),
    "abo_delivery_threshold": ("abo", 1, "threshold_rule", 3, t_delivery_sample, t_delivery_render, t_delivery_decide,
                               {"qty": [1, 2, 3, 5, 10, 50]}),
    "abo_handling_lookup": ("abo", 1, "multi_step_rule", 5, t_handling_sample, t_handling_render, t_handling_decide,
                            {"qty": [1, 2, 5, 10, 50], "__row__": None}),
    "abo_priority_score": ("abo", 1, "multi_step_rule", 5, t_priority_sample, t_priority_render, t_priority_decide,
                           {"days_waiting": [0, 1000]}),
    "abo_freight_signature": ("abo", 1, "threshold_rule", 3, t_freight_sample, t_freight_render, t_freight_decide,
                              {"qty": [1, 1000]}),
    "abo_two_offers": ("abo", 2, "two_image", 5, t_offers_sample, t_offers_render, t_offers_decide,
                       {"price_1": [0.5, 100000.0], "price_2": [0.5, 100000.0]}),
    "def_lot_threshold": ("defects", 1, "threshold_rule", 3, t_lot_sample, t_lot_render, t_lot_decide,
                          {"found": [0, 99]}),
    "def_rework_exception": ("defects", 1, "rule_exception", 4, t_rework_sample, t_rework_render, t_rework_decide,
                             {"line": None, "reworked": [True, False]}),
    "def_severity_points": ("defects", 1, "multi_step_rule", 5, t_points_sample, t_points_render, t_points_decide,
                            {"total": [0, 999], "__row__": None}),
    "def_two_units": ("defects", 2, "two_image", 4, t_units_sample, t_units_render, t_units_decide,
                      {"w1": [True, False], "w2": [True, False]}),
}
HIDDEN_OK = {"abo_return_exception", "abo_priority_score", "abo_freight_signature", "def_rework_exception", "def_two_units"}
# Share of the base (non-variant) items per template.
MIX = {"abo_return_exception": 0.11, "abo_delivery_threshold": 0.10, "abo_handling_lookup": 0.10,
       "abo_priority_score": 0.09, "abo_freight_signature": 0.09, "abo_two_offers": 0.12,
       "def_lot_threshold": 0.10, "def_rework_exception": 0.10, "def_severity_points": 0.08, "def_two_units": 0.11}


def alt_values(name: str, p: dict, field: str) -> list:
    if field == "line":
        out = [f"line {x}" for x in "ABCDEFG" if f"line {x}" not in p["rework_lines"]]
        return [p["rework_lines"][0], out[0]]
    return list(TEMPLATES[name][7][field])


def assumptions(name: str, p: dict) -> list[dict]:
    """Every truth assignment for the hidden predicates (none -> [{}])."""
    if not any(pr.get("hidden") for pr in p.get("preds", [])):
        return [{}]
    keys = ["u1", "u2"] if name == "def_two_units" else [i for i, pr in enumerate(p["preds"]) if pr.get("hidden")]
    out = [{}]
    for k in keys:
        out = [{**a, k: v} for a in out for v in (True, False)]
    return out


def decide(name: str, p: dict, facts_list: list[dict]):
    """(gold, unknown_reason). The gold is settled when every value a withheld field could take, and every truth value
    of a hidden attribute, gives the same answer; otherwise the item is unknown (insufficient_evidence)."""
    fn = TEMPLATES[name][6]
    fields = [w for w in W(p) if w in TEMPLATES[name][7] and not w.startswith("__")]
    variants = [p]
    for w in fields:
        variants = [dict(v, **{w: a}) for v in variants for a in alt_values(name, p, w) + [p[w]]]
    outcomes = {fn(v, facts_list, a) for v in variants for a in assumptions(name, p)}
    if None in outcomes or len(outcomes) != 1:
        return None, "insufficient_evidence"
    return outcomes.pop(), None


def flipped_facts(name: str, p: dict, fl: list[dict]) -> list[list[dict]]:
    """Counterfactual photo facts: each photo's decisive attribute changed (for photo_decisive)."""
    alts = []
    for i, f in enumerate(fl):
        if "defective" in f:
            g = dict(f, defective=not f["defective"])
            g.pop("defect_kind", None)
            if g["defective"] and name == "def_severity_points":
                for k in p["points"]:
                    alts.append([*fl[:i], dict(g, defect_kind=k), *fl[i + 1:]])
                continue
            alts.append([*fl[:i], g, *fl[i + 1:]])
            continue
        if name == "abo_handling_lookup":
            for t in p["fees"]:
                if t != f["product_type"]:
                    alts.append([*fl[:i], dict(f, product_type=t), *fl[i + 1:]])
            continue
        for pr in p.get("preds", []):
            if pr.get("hidden") or pr["attr"] not in f:
                continue
            if pr.get("photo") not in ("all", None, 0, i) and not (pr.get("photo") is None and i == 0):
                continue
            new = "__other__" if f[pr["attr"]] in pr["values"] else pr["values"][0]
            alts.append([*fl[:i], dict(f, **{pr["attr"]: new}), *fl[i + 1:]])
    return alts


def photo_decisive(name: str, p: dict, fl: list[dict], gold) -> bool:
    """True when some change to what the photo shows would change the settled gold."""
    fn = TEMPLATES[name][6]
    for alt in flipped_facts(name, p, fl):
        for a in assumptions(name, p):
            try:
                if fn(p, alt, a) != gold:
                    return True
            except KeyError:
                continue
    return False


# ------------------------------------------------------------------------------------------------ build
def load_context():
    """Photos + lookup context; also fills TYPE_GROUP. Shared with tests/test_p3_images.py."""
    bench = bench_image_shas()
    abo = load_abo_photos(bench)
    dfx = load_defect_photos(bench)
    TYPE_GROUP.clear()
    TYPE_GROUP.update({p["facts"]["product_type"]: p["facts"]["cgroup"] for p in abo})
    return abo, dfx, {"vocab": defect_vocab(dfx)}


def preds_clear(p: dict, photos: list[dict]) -> bool:
    for pr in p.get("preds", []):
        if pr.get("hidden"):
            continue
        idx = range(len(photos)) if pr.get("photo") == "all" else [pr.get("photo", 0) or 0]
        if not all(pred_clear(pr, photos[i]["facts"]) for i in idx):
            return False
    return True


def build(n_total: int = 15000, unknown_share: float = 0.15) -> list[dict]:
    abo, dfx, ctx_base = load_context()
    by_cat = defaultdict(list)
    for p in dfx:
        by_cat[p["facts"]["category"]].append(p)
    abo_by_attr = defaultdict(list)
    for p in abo:
        for a in ("colour", "material", "product_type", "shape", "pattern"):
            if a in p["facts"]:
                abo_by_attr[a].append(p)
    points_cats = sorted(c for c in by_cat if len(ctx_base["vocab"].get(c, ())) >= 2)

    n_settled = n_total - int(round(n_total * unknown_share))     # base items with a settled gold
    use = Counter()
    rows = []
    for name, share in MIX.items():
        domain, nph, kind, diff, sample, render, dec, alts = TEMPLATES[name]
        want = int(round(n_settled * share))
        rng = rng_for(SEED, name)
        made, tries, gold_ct = 0, 0, Counter()
        n_vals = {"def_two_units": 4, "abo_priority_score": 4, "abo_two_offers": 3, "abo_handling_lookup": 3,
                  "def_rework_exception": 3}.get(name, 2)
        while made < want and tries < want * 60:
            tries += 1
            ctx = dict(ctx_base, hidden=name in HIDDEN_OK and rng.random() < 0.15)
            if domain == "abo":
                if nph == 2:
                    attr = rng.choice(["colour", "colour", "material", "product_type", "pattern", "shape"])
                    photos = rng.sample(abo_by_attr[attr], 2)
                    ctx["pair_attr"] = attr
                else:
                    photos = [rng.choice(abo)]
            elif nph == 2:
                photos = rng.sample(by_cat[rng.choice(sorted(by_cat))], 2)
            elif name == "def_severity_points":
                photos = [rng.choice(by_cat[rng.choice(points_cats)])]
            else:
                photos = [rng.choice(dfx)]
            cap = 3 if domain == "abo" else 8
            if any(use[ph["sha256"]] >= cap for ph in photos) or len({ph["sha256"] for ph in photos}) < nph:
                continue
            p = sample(rng, photos, ctx)
            if p is None or not preds_clear(p, photos):
                continue
            fl = [ph["facts"] for ph in photos]
            gold, unk = decide(name, p, fl)
            if gold is None and not ctx["hidden"]:
                continue            # a tie or an otherwise ill-posed draw
            if gold is not None and not ctx["hidden"] and not photo_decisive(name, p, fl, gold) and rng.random() < 0.6:
                continue            # keep most items ones where the photo matters
            if gold is not None:
                if gold_ct[str(gold)] > (made + 20) / n_vals * 1.4:
                    continue
                gold_ct[str(gold)] += 1
            for ph in photos:
                use[ph["sha256"]] += 1
            rows.append(make_row(name, p, photos, gold, unk, len(rows)))
            made += gold is not None        # hidden-attribute unknowns ride along, they do not count

    # unknown variants: withhold a decisive record field, or the lookup row the photo's value needs
    rng = rng_for(SEED, "unknowns")
    need = max(0, n_total - len(rows))
    parents = [r for r in rows if r["gold"] is not None]
    rng.shuffle(parents)
    photo_by_sha = {p["sha256"]: p for p in abo + dfx}
    made = 0
    for par in parents:
        if made >= need:
            break
        name = par["provenance"]["template"]
        p = par["provenance"]["params"]
        photos = [photo_by_sha[s] for s in par["provenance"]["image_sha256"]]
        fl = [ph["facts"] for ph in photos]
        options = [f for f in TEMPLATES[name][7] if not f.startswith("__")]
        rng.shuffle(options)
        if "__row__" in TEMPLATES[name][7]:
            options.insert(0 if rng.random() < 0.6 else len(options), "__row__")
        for f in options:
            if f == "__row__":
                key = fl[0]["product_type"] if name == "abo_handling_lookup" else fl[0].get("defect_kind")
                if key is None:
                    continue
                q = dict(p, _withheld=[key])
            else:
                q = dict(p, _withheld=[f])
            g, u = decide(name, q, fl)
            if g is None:
                rows.append(make_row(name, q, photos, None, u, len(rows), parent=par["id"]))
                made += 1
                break
    # remaining budget: settled variants are useful too (the withheld field did not matter) but are not added here;
    # the pool stays at base + unknown variants.
    return rows


def make_row(name, p, photos, gold, unk, idx, parent=None) -> dict:
    domain, nph, kind, diff, *_ = TEMPLATES[name]
    state, field = TEMPLATES[name][5](p)
    lic = sorted({ph["licence"] for ph in photos})
    licence = "CC-BY-SA-4.0" if "CC-BY-SA-4.0" in lic else lic[0]
    d = diff
    if name == "abo_delivery_threshold" and p["qty"] > 1:
        d = 4
    if name == "def_two_units" and p.get("preds"):
        d = 5
    if gold is None:
        d = min(5, d + 1)
    hidden = any(pr.get("hidden") for pr in p.get("preds", []))
    return {
        "id": f"p3-I-joint-{idx:06d}",
        "source": "I",
        "dataset": "image_joint",
        "family": "image_joint_rule",
        "difficulty": d,
        "state": state,
        "images": [ph["image"] for ph in photos],
        "field": field,
        "gold": gold,
        "unknown_reason": unk,
        "gold_kind": "constructed",
        "parent_id": parent,
        "provenance": {
            "licence": licence,
            "image_licences": [ph["licence"] for ph in photos],
            "licence_evidence": sorted({LICENCE_EVIDENCE[(ph["source"], ph["licence"])] for ph in photos}),
            "photo_source": domain,
            "upstream_dataset": "abo" if domain == "abo" else "visa/dagm/btad",
            "upstream_ids": [ph["item"] for ph in photos],
            "image_sha256": [ph["sha256"] for ph in photos],
            "upstream_split": "train",
            "template": name,
            "rule_kind": kind,
            "params": p,
            "photo_facts": [ph["facts"] for ph in photos],
            "hidden_attribute": hidden,
            "photo_decisive": None if gold is None else photo_decisive(name, p, [ph["facts"] for ph in photos], gold),
            "unknown_construction": (None if gold is not None else
                                     ("withheld:" + ",".join(map(str, p["_withheld"])) if p.get("_withheld")
                                      else "hidden_attribute")),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rows = build(a.n)
    n = write(a.out, rows)
    c = Counter((r["provenance"]["template"], r["gold"] is None) for r in rows)
    print(json.dumps({"rows": n, "unknown": sum(r["gold"] is None for r in rows),
                      "by_template": {f"{k[0]}{' (unknown)' if k[1] else ''}": v for k, v in sorted(c.items())}}, indent=1))


if __name__ == "__main__":
    main()
