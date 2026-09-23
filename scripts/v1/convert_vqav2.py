"""decision-v1 converter: VQAv2 -> COCO-image general VQA decision records.

Rules applied (see docs/decision-v1-dataset-spec.md):
  * Only questions whose majority answer has >= 6/10 human agreement are kept, so the label is clean.
  * `answer_type == "yes/no"`  -> boolean field, balanced 50/50 before abstentions.
  * `answer_type == "number"`  -> skipped entirely (counting is covered by another source).
  * `answer_type == "other"`   -> choice field whose distractors are frequent *training* answers to
    the same VQA `question_type` (the "what color is the ...", "what sport is ...", "what kind of ..."
    prefix families), filtered so no distractor is a synonym / substring / token-sharing variant of
    the gold answer.
  * `not_listed` abstentions: the gold option is withheld from the option set.
  * train/dev come from COCO train2014, test from COCO val2014; images in
    data/decision-v1/exclusions.json never enter train/dev.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_vqav2.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v1")

import _coco_shared as C  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402

SOURCE = "vqav2"
SEED = 20260922
RAW = Path(".cache/datasets/v1/vqav2")
ROOT = Path("data/decision-v1") / SOURCE
LICENSE = "CC-BY-4.0 (VQA v2 annotations); images: COCO/Flickr terms, no image licence granted"

TRAIN_TOTAL = 100_000
BOOLEAN_SHARE = 0.40
MAX_PER_IMAGE = 3
MAX_PER_FAMILY_PER_IMAGE = 2
DEV_FRACTION = 0.03
TEST_PER_FAMILY = 300
TEST_CAP = 1_500
NOT_LISTED_RATE = 0.28          # of choice records -> ~17% abstentions over the whole source
MIN_AGREEMENT = 6               # of 10 human answers
MIN_GOLD_COUNT = 5              # gold must be an established answer for its question family
MIN_POOL = 12                   # distinct distractor candidates required
POOL_SIZE = 150
BIG_OPTION_RATE = 0.10

STOP = {"a", "an", "the", "of", "and", "or", "is", "are", "in", "on", "at", "to", "it", "its",
        "this", "that", "there", "his", "her", "their", "with", "for", "be", "by", "from"}
# Tokens that mean the same thing; collapsed so two options never say the same thing twice.
SYNONYM = {
    "grey": "gray", "gray": "gray",
    "tv": "television", "television": "television",
    "bike": "bicycle", "bicycle": "bicycle", "cycle": "bicycle",
    "plane": "airplane", "airplane": "airplane", "aeroplane": "airplane", "jet": "airplane",
    "couch": "sofa", "sofa": "sofa",
    "photo": "picture", "picture": "picture", "photograph": "picture",
    "cellphone": "phone", "phone": "phone", "mobile": "phone",
    "hotdog": "hotdog", "hotdogs": "hotdog",
    "doughnut": "donut", "donut": "donut", "doughnuts": "donut", "donuts": "donut",
    "kid": "child", "child": "child", "kids": "child", "children": "child",
    "auto": "car", "car": "car", "cars": "car", "automobile": "car",
    "bathroom": "bathroom", "restroom": "bathroom", "washroom": "bathroom",
    "sneaker": "shoe", "shoe": "shoe", "sneakers": "shoe", "shoes": "shoe",
    "veggies": "vegetable", "vegetables": "vegetable", "vegetable": "vegetable",
    "soda": "soda", "pop": "soda", "cola": "soda",
    "rubbish": "trash", "garbage": "trash", "trash": "trash",
    "sunglasses": "glasses", "glasses": "glasses", "eyeglasses": "glasses",
}
NUMBERISH = re.compile(r"^\d+(\.\d+)?$")


def tokens(text: str) -> frozenset[str]:
    out = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        t = SYNONYM.get(t, t)
        if t not in STOP:
            out.add(t)
    return frozenset(out)


def clashes(a: str, b: str, ta: frozenset[str], tb: frozenset[str]) -> bool:
    """True when two answer strings are too close to sit in the same option set."""
    if not ta or not tb:
        return True
    if ta & tb:
        return True
    return a in b or b in a


def family_of(question_type: str, answer_type: str) -> str:
    if answer_type == "yes/no":
        return "yes_no"
    if "color" in question_type:
        return "color"
    if question_type in ("what kind of", "what type of", "what brand", "what is the brand"):
        return "object_type"
    if question_type.startswith("where") or question_type == "what room is":
        return "scene_location"
    return "general_choice"


def pool_key(question_type: str, question: str) -> str:
    if question_type != "none of the above":
        return question_type
    words = re.findall(r"[a-z']+", question.lower())[:2]
    return "prefix:" + " ".join(words)


# --------------------------------------------------------------------------- load
def load(split: str) -> list[dict]:
    qs = json.loads((RAW / f"v2_OpenEnded_mscoco_{split}_questions.json").read_text())["questions"]
    ann = json.loads((RAW / f"v2_mscoco_{split}_annotations.json").read_text())["annotations"]
    text = {q["question_id"]: q["question"] for q in qs}
    rows = []
    for a in ann:
        if a["answer_type"] == "number":
            continue
        counts = Counter(x["answer"] for x in a["answers"])
        gold, n = counts.most_common(1)[0]
        if n < MIN_AGREEMENT or gold != a["multiple_choice_answer"]:
            continue
        if a["answer_type"] == "yes/no" and gold not in ("yes", "no"):
            continue
        if a["answer_type"] == "other" and (len(gold) > 128 or NUMBERISH.match(gold)):
            continue
        q = text[a["question_id"]]
        rows.append(dict(qid=a["question_id"], image_id=a["image_id"], question=q,
                         question_type=a["question_type"], answer_type=a["answer_type"],
                         gold=gold, agreement=n, split=split, alts=sorted(counts),
                         family=family_of(a["question_type"], a["answer_type"]),
                         pool_key=pool_key(a["question_type"], q)))
    return rows


# --------------------------------------------------------------------------- options
def build_pools(rows: list[dict]) -> dict[str, list[str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["answer_type"] == "other":
            counts[r["pool_key"]][r["gold"]] += 1
    pools = {}
    for key, c in counts.items():
        ranked = [a for a, n in c.most_common(POOL_SIZE) if n >= 3 and not NUMBERISH.match(a)
                  and len(a) <= 128]
        if len(ranked) >= MIN_POOL:
            pools[key] = ranked
    return pools


def option_total(rng: random.Random) -> int:
    return rng.randint(13, 25) if rng.random() < BIG_OPTION_RATE else rng.randint(2, 12)


def choose_options(gold: str, alts: list[str], pool: list[str], rng: random.Random,
                   hide_gold: bool):
    """Return (option values, ok). No option is a synonym of another, and no distractor matches
    *any* of the ten human answers - otherwise a withheld-gold item would still have a right
    option on the list."""
    gold_tok = tokens(gold)
    forbidden = [(a, tokens(a)) for a in alts]
    picked: list[str] = []
    taken: list[tuple[str, frozenset[str]]] = [] if hide_gold else [(gold, gold_tok)]
    want = option_total(rng) - (0 if hide_gold else 1)
    order = list(pool)
    rng.shuffle(order)
    for cand in order:
        if len(picked) >= want:
            break
        ct = tokens(cand)
        if any(clashes(a, cand, at, ct) for a, at in forbidden):
            continue
        if any(clashes(cand, t, ct, tt) for t, tt in taken):
            continue
        picked.append(cand)
        taken.append((cand, ct))
    values = picked + ([] if hide_gold else [gold])
    if len(values) < 2:
        return None, False
    rng.shuffle(values)
    return values, True


def make_record(r: dict, pools: dict, rng: random.Random, partition: str) -> dict | None:
    if r["answer_type"] == "yes/no":
        field = dict(id="answer", type="boolean", question=r["question"])
        target, cause = r["gold"] == "yes", None
    else:
        pool = pools.get(r["pool_key"])
        if not pool:
            return None
        counts = POOL_COUNTS[r["pool_key"]]
        if counts.get(r["gold"], 0) < MIN_GOLD_COUNT:
            return None
        hide = rng.random() < NOT_LISTED_RATE
        values, ok = choose_options(r["gold"], r["alts"], pool, rng, hide)
        if not ok:
            return None
        field = dict(id="answer", type="choice", question=r["question"],
                     options=[dict(value=v) for v in values])
        target, cause = (None, "not_listed") if hide else (r["gold"], None)
    request = dict(request_id=f"vqav2-{r['split']}-{r['qid']}", state={}, fields=[field])
    Request.model_validate(request)
    return dict(id=f"{SOURCE}:{r['qid']}", source=SOURCE, source_split=r["split"],
                source_group=str(r["image_id"]), family=r["family"], license=LICENSE,
                images=[], request=request, target=target, abstention_cause=cause,
                source_answer=r["gold"], partition=partition, _image_id=r["image_id"],
                _coco_split=r["split"])


# --------------------------------------------------------------------------- selection
def image_order(image_ids, salt: str) -> list[int]:
    return sorted(image_ids, key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())


def main() -> None:
    rng = random.Random(SEED)
    excluded = C.exclusions()
    ledger = C.load_ledger()

    print("loading annotations ...", flush=True)
    train_rows = load("train2014")
    val_rows = load("val2014")
    global POOL_COUNTS
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in train_rows:
        if r["answer_type"] == "other":
            counts[r["pool_key"]][r["gold"]] += 1
    POOL_COUNTS = counts
    pools = build_pools(train_rows)
    print(f"  train eligible {len(train_rows)}, val eligible {len(val_rows)}, pools {len(pools)}",
          flush=True)

    # ---- test partition first: it claims its val2014 images in the shared ledger.
    by_image_val = defaultdict(list)
    for r in val_rows:
        by_image_val[r["image_id"]].append(r)
    test_quota = {f: TEST_PER_FAMILY for f in ("yes_no", "color", "object_type",
                                               "scene_location", "general_choice")}
    test_bool = Counter()
    test: list[dict] = []
    for image_id in image_order(by_image_val, f"{SOURCE}-test"):
        if len(test) >= TEST_CAP:
            break
        if not C.available(ledger, image_id, "test"):
            continue
        cands = sorted(by_image_val[image_id], key=lambda r: r["qid"])
        rng.shuffle(cands)
        for r in cands:
            if test_quota.get(r["family"], 0) <= 0:
                continue
            if r["family"] == "yes_no" and test_bool[r["gold"]] >= TEST_PER_FAMILY // 2:
                continue
            rec = make_record(r, pools, rng, "test")
            if rec is None:
                continue
            test_quota[r["family"]] -= 1
            if r["family"] == "yes_no":
                test_bool[r["gold"]] += 1
            test.append(rec)
            C.claim(ledger, image_id, "test")
            break

    # ---- train/dev: up to MAX_PER_IMAGE questions per COCO train2014 image, quota-driven.
    by_image_train = defaultdict(list)
    for r in train_rows:
        if r["image_id"] in excluded:
            continue
        by_image_train[r["image_id"]].append(r)
    bool_quota = {"yes": int(TRAIN_TOTAL * BOOLEAN_SHARE / 2),
                  "no": int(TRAIN_TOTAL * BOOLEAN_SHARE / 2)}
    choice_quota = TRAIN_TOTAL - sum(bool_quota.values())
    fit: list[dict] = []
    fit_order = sorted(by_image_train,
                       key=lambda i: (-min(len(by_image_train[i]), MAX_PER_IMAGE),
                                      hashlib.sha256(f"{SOURCE}-fit:{i}".encode()).hexdigest()))
    for image_id in fit_order:
        if choice_quota <= 0 and not any(bool_quota.values()):
            break
        if not C.available(ledger, image_id, "fit"):
            continue
        cands = sorted(by_image_train[image_id], key=lambda r: r["qid"])
        rng.shuffle(cands)
        # spread the questions taken from one image over different question families
        used_families: Counter = Counter()
        taken = 0
        for r in cands:
            if taken >= MAX_PER_IMAGE:
                break
            if used_families[r["family"]] >= MAX_PER_FAMILY_PER_IMAGE:
                continue
            if r["family"] == "yes_no":
                if bool_quota[r["gold"]] <= 0:
                    continue
            elif choice_quota <= 0:
                continue
            rec = make_record(r, pools, rng, "train")
            if rec is None:
                continue
            if r["family"] == "yes_no":
                bool_quota[r["gold"]] -= 1
            else:
                choice_quota -= 1
            used_families[r["family"]] += 1
            fit.append(rec)
            C.claim(ledger, image_id, "fit")
            taken += 1
    if "--plan" not in sys.argv:
        C.save_ledger(ledger)
    print(f"  planned train/dev {len(fit)} over {len({r['_image_id'] for r in fit})} images, "
          f"test {len(test)}", flush=True)

    # dev is a 3% slice taken by image so an image is not split across train and dev
    fit_images = image_order({r["_image_id"] for r in fit}, f"{SOURCE}-dev")
    dev_images = set(fit_images[: max(1, round(len(fit_images) * DEV_FRACTION))])
    for r in fit:
        if r["_image_id"] in dev_images:
            r["partition"] = "dev"

    records = fit + test
    if "--plan" in sys.argv:
        print(json.dumps(dict(records=len(records),
                              partitions=dict(Counter(r["partition"] for r in records)),
                              families=dict(Counter(r["family"] for r in records)),
                              abstention=dict(Counter(str(r["abstention_cause"]) for r in records)),
                              booleans=dict(Counter(str(r["target"]) for r in records
                                                    if r["request"]["fields"][0]["type"] == "boolean")),
                              images=len({r["_image_id"] for r in records}),
                              options=dict(sorted(Counter(len(r["request"]["fields"][0].get("options", []))
                                                          for r in records).items()))), indent=2))
        return

    # ---- images
    keys = [f"{r['_coco_split']}/{r['_image_id']}" for r in records]
    print(f"fetching {len(set(keys))} COCO images ...", flush=True)
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
                          booleans=dict(Counter(str(r["target"]) for r in kept
                                                if r["request"]["fields"][0]["type"] == "boolean")),
                          images=len({r["images"][0]["sha256"] for r in kept})), indent=2))


if __name__ == "__main__":
    main()
