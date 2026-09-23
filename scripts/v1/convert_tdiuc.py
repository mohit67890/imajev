"""decision-v1 converter: TDIUC -> 12 typed question families over COCO images.

Field types
  * `counting`                      -> ordinal, a window of consecutive integers around the count
                                       (`not_listed` = the true count falls outside the window).
  * yes/no answers (object_presence and the yes/no half of scene/sentiment/utility) -> boolean,
    balanced true/false; these stay answerable.
  * everything else                 -> choice, distractors from the frequent training answers of the
                                       same TDIUC question type.
  * `absurd`                        -> false_premise abstentions. Two shapes:
      - choice: the original absurd question with options borrowed from the family its template
        belongs to (colour question -> colour options, "what material" -> attribute options, ...).
      - boolean: an absurd "What colour/material/shape is the X?" rewritten as a boolean that
        presupposes the absent X ("Is the sofa red?"), with >= 5 phrasings per attribute. Only
        *attribute-of-a-definite-object* rewrites are used: "Is there a sofa?" would be honestly
        answerable "no" and is never produced.

The absurd category is dominated by colour questions, so each template family is capped at a share
of the absurd budget (docs/training-mixture-research.md flags the "colour question -> absurd"
shortcut).

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_tdiuc.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v1")

import _coco_shared as C  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402

SOURCE = "tdiuc"
SEED = 20260922
ARCHIVE = Path(".cache/datasets/tdiuc/TDIUC.zip")
ROOT = Path("data/decision-v1") / SOURCE
LICENSE = ("CC-BY-4.0 (declared in the TDIUC annotation JSON `licence` field, inherited from "
           "VQA/COCO); images: COCO/Flickr terms, no image licence granted")

MAX_PER_IMAGE = 3
DEV_FRACTION = 0.03
TEST_PER_FAMILY = 115
NOT_LISTED_RATE = 0.06
BIG_OPTION_RATE = 0.10
POOL = 100

TRAIN_TARGETS = {
    "object_presence": 2200, "counting": 2200, "color": 2200, "object_recognition": 2200,
    "scene_recognition": 2200, "positional_reasoning": 2000, "sport_recognition": 2000,
    "attribute": 2000, "activity_recognition": 2000, "sentiment_understanding": 1100,
    "utility_affordance": 250, "absurd": 2000, "absurd_boolean": 1200,
}
ABSURD_SHARE_CAP = 0.35          # per template family, of the absurd choice budget

# absurd question template -> the family whose answer pool supplies its options
ABSURD_TEMPLATES = [
    (re.compile(r"^what colou?r"), "color"),
    (re.compile(r"^what material"), "attribute"),
    (re.compile(r"^what shape"), "attribute"),
    (re.compile(r"^what (is|are) the .* made (of|from)"), "attribute"),
    (re.compile(r"^what animal"), "object_recognition"),
    (re.compile(r"^what (sport|game)"), "sport_recognition"),
    (re.compile(r"^what activity"), "activity_recognition"),
    (re.compile(r"^what (kind|type|sort) of"), "object_recognition"),
    (re.compile(r"^what (food|furniture|vehicle|fruit)"), "object_recognition"),
    (re.compile(r"^what is (behind|in front of|under|above|on top of|to the (left|right))"),
     "positional_reasoning"),
    (re.compile(r"^what (is|are) the (man|woman|boy|girl|person|people|men|women|kids?|children)"
                r" doing"), "activity_recognition"),
]

# absurd -> boolean rewrites. Each entry: attribute, regex over the question, value pool, phrasings.
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple",
          "pink", "gray", "silver", "gold", "beige"]
MATERIALS = ["wood", "metal", "plastic", "glass", "brick", "concrete", "leather", "stone",
             "cloth", "paper"]
SHAPES = ["round", "square", "rectangular", "triangular", "oval", "curved"]
REWRITES = {
    "color": (re.compile(r"^what colou?r (is|are) (the [a-z][a-z' ]{1,40}?)\s*\?*$"), COLORS, [
        "{Be} {obj} {v}?",
        "Is the color of {obj} {v}?",
        "Would you describe {obj} as {v}?",
        "{Do} {obj} look {v}?",
        "{Be} {obj} {v} in this photo?",
        "In this picture, {be} {obj} {v}?",
    ]),
    "material": (re.compile(r"^what material (is|are) (the [a-z][a-z' ]{1,40}?)"
                            r"(?: made (?:of|from))?\s*\?*$"), MATERIALS, [
        "{Be} {obj} made of {v}?",
        "{Be} {obj} {v}?",
        "{Do} {obj} appear to be made of {v}?",
        "Is the material of {obj} {v}?",
        "{Be} {obj} constructed from {v}?",
        "Would you say {obj} {be} made of {v}?",
    ]),
    "shape": (re.compile(r"^what shape (is|are) (the [a-z][a-z' ]{1,40}?)\s*\?*$"), SHAPES, [
        "{Be} {obj} {v}?",
        "Is the shape of {obj} {v}?",
        "{Do} {obj} look {v}?",
        "Would you call the shape of {obj} {v}?",
        "{Be} {obj} {v} in shape?",
        "In this photo, {be} {obj} {v}?",
    ]),
}
TRAILING_PREP = re.compile(r"\b(on|of|in|at|with|for|to|from|by|under|over|near|behind|"
                           r"beside|above|inside)$")
REWRITE_TARGETS = {"color": 500, "material": 400, "shape": 300}
BAD_OBJECT = re.compile(r"^the (it|he|she|they|this|that|there|picture|photo|image)\b")

NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
                "eighteen": 18, "nineteen": 19, "twenty": 20}
WORD_OF = {v: k for k, v in NUMBER_WORDS.items()}
WORD_OF[0] = "none"

STOP = {"a", "an", "the", "of", "and", "or", "to", "in", "on", "is", "are", "it"}
# TDIUC's small answer vocabularies contain outright synonyms ("happy"/"happiness"/"joy",
# "racket"/"racquet"); two options must never mean the same thing.
SYNONYM_GROUPS = [
    {"happy", "happiness", "joy", "joyful", "smiling", "smile", "cheerful", "glad", "excited"},
    {"sad", "sadness", "unhappy", "crying", "upset"},
    {"racket", "racquet"}, {"sofa", "couch"}, {"tv", "television"}, {"bike", "bicycle"},
    {"plane", "airplane", "aeroplane", "jet"}, {"gray", "grey"},
    {"metal", "tin", "steel", "aluminum", "iron"}, {"wood", "wooden", "logs", "lumber", "timber"},
    {"photo", "picture", "image"}, {"trash", "garbage", "rubbish"}, {"car", "automobile", "auto"},
    {"child", "kid", "children", "kids", "boy", "girl"},
    {"round", "circle", "circular", "oval", "spherical"},
    {"square", "rectangle", "rectangular"}, {"indoor", "indoors", "inside"},
    {"outdoor", "outdoors", "outside"}, {"couch", "loveseat"},
]
CANON = {w: sorted(g)[0] for g in SYNONYM_GROUPS for w in g}


def tokens(text: str) -> frozenset[str]:
    out = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t not in CANON and len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        t = CANON.get(t, t)
        if t not in STOP:
            out.add(t)
    return frozenset(out)


def clashes(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return True
    if ta & tb or a in b or b in a:
        return True
    # "happy" / "happiness", "circle" / "circular": one stem is a prefix of the other
    return any(len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x))
               for x in ta for y in tb)


def absurd_family(question: str) -> str | None:
    q = question.lower().strip()
    for pattern, family in ABSURD_TEMPLATES:
        if pattern.search(q):
            return family
    return None


def rewrite_kind(question: str) -> tuple[str, str, str] | None:
    """-> (attribute kind, object phrase, 'is'|'are'), or None when no clean rewrite exists."""
    q = " ".join(question.lower().split())
    for kind, (pattern, _values, _phrasings) in REWRITES.items():
        m = pattern.match(q)
        if m:
            be, obj = m.group(1), m.group(2).strip()
            if BAD_OBJECT.match(obj) or len(obj.split()) > 5 or TRAILING_PREP.search(obj):
                return None
            return kind, obj, be
    return None


# --------------------------------------------------------------------------- load
def load(split: str) -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as z:
        ann = json.loads(z.read(f"TDIUC/Annotations/mscoco_{split}2014_annotations.json"))
        text = {q["question_id"]: q["question"] for q in
                json.loads(z.read(f"TDIUC/Questions/OpenEnded_mscoco_{split}2014_questions.json")
                           )["questions"]}
    rows = []
    for a in ann["annotations"]:
        if a["image_id"] >= 80_000_000:      # Visual Genome half of TDIUC: no stable public URL
            continue
        answer = a["answers"][0]["answer"]
        if answer == "grey":
            answer = "gray"
        rows.append(dict(qid=a["question_id"], image_id=a["image_id"], split=f"{split}2014",
                         family=a["question_type"], question=text[a["question_id"]],
                         answer=answer))
    return rows


def pool_key(family: str, question: str) -> str:
    """TDIUC's `attribute` type mixes material and shape questions; keep their pools apart so a
    material question is not offered "triangle" as a distractor."""
    if family == "attribute":
        q = question.lower()
        if "shape" in q:
            return "attribute:shape"
        if "material" in q or "made of" in q or "made from" in q:
            return "attribute:material"
    return family


def build_pools(rows: list[dict]) -> dict[str, list[str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["family"] in ("absurd", "counting") or r["answer"] in ("yes", "no"):
            continue
        counts[pool_key(r["family"], r["question"])][r["answer"]] += 1
    return {f: [a for a, n in c.most_common(POOL) if n >= 3 and len(a) <= 128]
            for f, c in counts.items()}


# --------------------------------------------------------------------------- fields
def option_total(rng: random.Random) -> int:
    return rng.randint(13, 25) if rng.random() < BIG_OPTION_RATE else rng.randint(2, 12)


def choice_field(question: str, gold: str | None, pool: list[str], rng: random.Random):
    """gold None means the question has no right answer at all (absurd)."""
    picked: list[str] = []
    taken: list[str] = [] if gold is None else [gold]
    want = option_total(rng) - (0 if gold is None else 1)
    order = list(pool)
    rng.shuffle(order)
    for cand in order:
        if len(picked) >= want:
            break
        if any(clashes(cand, t) for t in taken):
            continue
        picked.append(cand)
        taken.append(cand)
    values = picked + ([] if gold is None else [gold])
    if len(values) < 2:
        return None
    rng.shuffle(values)
    return dict(id="answer", type="choice", question=question,
                options=[dict(value=v) for v in values])


def ordinal_field(question: str, count: int, rng: random.Random, hide: bool):
    size = rng.randint(2, 7)
    if hide:
        below = count - size
        if below >= 0 and (rng.random() < 0.5 or count > 12):
            lo = below
        else:
            lo = count + 1
    else:
        lo = rng.randint(max(0, count - size + 1), count)
    lo = max(0, lo)
    levels = [dict(value=lo + i, description=WORD_OF.get(lo + i, str(lo + i)))
              for i in range(size)]
    if not hide and count not in [x["value"] for x in levels]:
        return None
    if hide and count in [x["value"] for x in levels]:
        return None
    return dict(id="answer", type="ordinal", question=question, levels=levels)


# --------------------------------------------------------------------------- main
def main() -> None:
    rng = random.Random(SEED)
    excluded = C.exclusions()
    ledger = C.load_ledger()
    train_rows, val_rows = load("train"), load("val")
    pools = build_pools(train_rows)
    print(f"loaded train {len(train_rows)} val {len(val_rows)}; pools "
          f"{ {k: len(v) for k, v in pools.items()} }", flush=True)

    def build(r: dict, partition: str, bucket: str, state: dict) -> dict | None:
        family, answer, question = r["family"], r["answer"], r["question"]
        cause = None
        if bucket == "absurd_boolean":
            kind_obj = rewrite_kind(question)
            if kind_obj is None:
                return None
            kind, obj, be = kind_obj
            _pattern, values, phrasings = REWRITES[kind]
            value = rng.choice(values)
            text = rng.choice(phrasings).format(obj=obj, v=value, be=be, Be=be.capitalize(),
                                                Do="Does" if be == "is" else "Do")
            field = dict(id="answer", type="boolean", question=text[0].upper() + text[1:])
            target, cause = None, "false_premise"
            source_answer = "doesnotapply"
        elif family == "absurd":
            pool_family = absurd_family(question)
            if pool_family == "attribute":
                pool_family = ("attribute:shape" if "shape" in question.lower()
                               else "attribute:material")
            if pool_family is None or pool_family not in pools:
                return None
            field = choice_field(question, None, pools[pool_family], rng)
            if field is None:
                return None
            target, cause, source_answer = None, "false_premise", answer
        elif family == "counting":
            if answer not in NUMBER_WORDS:
                return None
            count = NUMBER_WORDS[answer]
            hide = rng.random() < NOT_LISTED_RATE
            field = ordinal_field(question, count, rng, hide)
            if field is None:
                return None
            target, cause = (None, "not_listed") if hide else (count, None)
            source_answer = answer
        elif answer in ("yes", "no"):
            field = dict(id="answer", type="boolean", question=question)
            target, source_answer = answer == "yes", answer
        else:
            pool = pools.get(pool_key(family, question))
            if not pool or answer not in pool:
                return None
            hide = rng.random() < NOT_LISTED_RATE
            field = choice_field(question, None if hide else answer,
                                 [x for x in pool if not clashes(x, answer)], rng)
            if field is None:
                return None
            target, cause = (None, "not_listed") if hide else (answer, None)
            source_answer = answer
        request = dict(request_id=f"tdiuc-{r['split']}-{r['qid']}"
                                  + ("-b" if bucket == "absurd_boolean" else ""),
                       state={}, fields=[field])
        Request.model_validate(request)
        return dict(id=f"{SOURCE}:{r['split']}-{r['qid']}"
                       + ("-b" if bucket == "absurd_boolean" else ""),
                    source=SOURCE, source_split=r["split"], source_group=str(r["image_id"]),
                    family=bucket, license=LICENSE, images=[], request=request, target=target,
                    abstention_cause=cause, source_answer=source_answer, partition=partition,
                    _image_id=r["image_id"], _coco_split=r["split"])

    def harvest(rows, quotas, side, partition, scale):
        by_image = defaultdict(list)
        for r in rows:
            if side == "fit" and r["image_id"] in excluded:
                continue
            by_image[r["image_id"]].append(r)
        # images already in the shared cache first, so this source adds almost no new downloads
        order = sorted(by_image, key=lambda i: (
            0 if C.cached(f"{by_image[i][0]['split']}/{i}") is not None else 1,
            hashlib.sha256(f"{SOURCE}-{side}:{i}".encode()).hexdigest()))
        state = dict(absurd_templates=Counter(), bool_balance=Counter(), rewrites=Counter())
        absurd_cap = max(1, int(quotas.get("absurd", 0) * ABSURD_SHARE_CAP))
        out = []
        for image_id in order:
            if not any(v > 0 for v in quotas.values()):
                break
            if not C.available(ledger, image_id, side):
                continue
            cands = sorted(by_image[image_id], key=lambda r: r["qid"])
            rng.shuffle(cands)
            taken = 0
            for r in cands:
                if taken >= MAX_PER_IMAGE:
                    break
                bucket = r["family"]
                if bucket == "absurd":
                    kind_obj = rewrite_kind(r["question"])
                    can_bool = (kind_obj is not None and quotas.get("absurd_boolean", 0) > 0
                                and state["rewrites"][kind_obj[0]]
                                < REWRITE_TARGETS[kind_obj[0]] * scale)
                    if can_bool and (quotas.get("absurd", 0) <= 0 or rng.random() < 0.5):
                        bucket = "absurd_boolean"
                if quotas.get(bucket, 0) <= 0:
                    continue
                fam = None
                if bucket == "absurd":
                    fam = absurd_family(r["question"])
                    if fam is None or state["absurd_templates"][fam] >= absurd_cap:
                        continue
                rec = build(r, partition, bucket, state)
                if rec is None:
                    continue
                f = rec["request"]["fields"][0]
                if f["type"] == "boolean" and rec["target"] is not None:
                    key = (bucket, rec["target"])
                    if state["bool_balance"][key] > state["bool_balance"][(bucket,
                                                                          not rec["target"])]:
                        continue
                    state["bool_balance"][key] += 1
                if bucket == "absurd_boolean":
                    state["rewrites"][rewrite_kind(r["question"])[0]] += 1
                if fam is not None:
                    state["absurd_templates"][fam] += 1
                quotas[bucket] -= 1
                C.claim(ledger, image_id, side)
                out.append(rec)
                taken += 1
        return out

    fit = harvest(train_rows, dict(TRAIN_TARGETS), "fit", "train", 1.0)
    test = harvest(val_rows, {k: TEST_PER_FAMILY for k in TRAIN_TARGETS}, "test", "test",
                   3 * TEST_PER_FAMILY / TRAIN_TARGETS["absurd_boolean"])

    fit_images = sorted({r["_image_id"] for r in fit},
                        key=lambda i: hashlib.sha256(f"{SOURCE}-dev:{i}".encode()).hexdigest())
    dev = set(fit_images[: max(1, round(len(fit_images) * DEV_FRACTION))])
    for r in fit:
        if r["_image_id"] in dev:
            r["partition"] = "dev"
    records = fit + test
    print(f"planned {len(records)} ({len(fit)} fit / {len(test)} test) over "
          f"{len({r['_image_id'] for r in records})} images", flush=True)
    if "--plan" in sys.argv:
        print(json.dumps(dict(families=dict(Counter(r["family"] for r in records)),
                              partitions=dict(Counter(r["partition"] for r in records)),
                              abstention=dict(Counter(str(r["abstention_cause"])
                                                      for r in records)),
                              types=dict(Counter(r["request"]["fields"][0]["type"]
                                                 for r in records)),
                              new_images=sum(C.cached(f"{r['_coco_split']}/{r['_image_id']}")
                                             is None for r in records)), indent=2))
        return
    C.save_ledger(ledger)

    keys = [f"{r['_coco_split']}/{r['_image_id']}" for r in records]
    print(f"fetching {sum(C.cached(k) is None for k in set(keys))} new COCO images ...", flush=True)
    meta = C.ensure(keys, workers=64, label=SOURCE)
    banned = C.banned_sha()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "images").mkdir(exist_ok=True)
    kept = []
    for r in records:
        key = f"{r['_coco_split']}/{r['_image_id']}"
        rec = meta.get(key)
        if rec is None:
            continue
        if rec["sha256"] in banned and r["partition"] != "test":
            continue
        r["images"] = [C.link_into(ROOT, key, rec)]
        r.pop("_image_id"), r.pop("_coco_split")
        kept.append(r)
    C.write_records(SOURCE, kept)
    print(json.dumps(dict(records=len(kept),
                          partitions=dict(Counter(r["partition"] for r in kept)),
                          families=dict(Counter(r["family"] for r in kept)),
                          abstention=dict(Counter(str(r["abstention_cause"]) for r in kept)),
                          types=dict(Counter(r["request"]["fields"][0]["type"] for r in kept)),
                          images=len({r["images"][0]["sha256"] for r in kept})), indent=2))


if __name__ == "__main__":
    main()
