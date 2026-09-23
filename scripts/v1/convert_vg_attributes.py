"""Visual Genome attribute annotations -> decision-v1 `choice` records.

One question per image about one *unambiguously named* object (its name occurs exactly once in
that image's object list, and is not a sub-phrase of any other object's name there), asking for
one attribute of it.  Distractors are sibling values of the same attribute dimension - only
values that cannot simultaneously be true of the object - ordered hardest first by how often
they co-occur with that object name elsewhere in Visual Genome.

Attribute typing and the question templates live in scripts/v1/vg_attribute_lexicon.py.
"""
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import CACHE_ROOT, DATA_ROOT, VG_PROC, load_exclusions, stable_rank, store_cached
from vg_attribute_lexicon import TEMPLATES, build_index
from vision_decision.contracts import Request

SOURCE = "vg_attributes"
SEED = 20260922
LICENSE = "CC-BY-4.0"
TRAIN, DEV, TEST_PER_FAMILY = 28_500, 855, 300
NOT_LISTED_RATE = 0.17
FAMILY_SHARE = {"color": 0.40, "state": 0.30, "material": 0.20, "shape": 0.10}

OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
ATTR_ZIP = CACHE_ROOT / "vg_attributes/attributes.json.zip"
INDEX, GROUPS = build_index()


def gqa_image_split() -> tuple[set[int], set[int]]:
    """Reuse GQA's own train/val image partition so the two sources agree on what is held out."""
    def ids(sub):
        out = set()
        for shard in sorted((CACHE_ROOT / "gqa/hf" / sub).glob("*.parquet")):
            out |= {int(x) for x in pq.read_table(shard, columns=["imageId"])["imageId"].to_pylist()}
        return out
    return ids("train_balanced_instructions"), ids("val_balanced_instructions")


def candidates(images: list[dict]):
    """(image_id, object_id, name, family, dimension, gold_group, blocked_groups)"""
    for im in images:
        objs = []
        for o in im["attributes"]:
            names = [str(n).strip().lower() for n in (o.get("names") or []) if str(n).strip()]
            if not names:
                continue
            name = names[0]
            if not (1 <= len(name.split()) <= 3) or len(name) > 30 or not name.replace(" ", "").isalpha():
                continue
            objs.append((name, o))
        name_count = Counter(n for n, _ in objs)
        for name, o in objs:
            if name_count[name] != 1:
                continue
            if any(other != name and (name in other.split() or name in other) for other in name_count):
                continue
            typed = defaultdict(set)
            for a in (o.get("attributes") or []):
                hit = INDEX.get(str(a).strip().lower())
                if hit:
                    typed[(hit[0], hit[1])].add(hit[2])
            blocked = {g for gs in typed.values() for g in gs}
            for (family, dim), vals in typed.items():
                if len(vals) != 1:
                    continue                      # conflicting values on the same dimension
                yield (im["image_id"], o["object_id"], name, family, dim, next(iter(vals)), blocked)


def make_record(c, split, partition, rng, image, cooc, dim_freq):
    image_id, object_id, name, family, dim, gold, blocked = c
    siblings = [g for g in GROUPS[(family, dim)] if g != gold and g not in blocked]
    siblings.sort(key=lambda g: (-cooc.get((name, g), 0), -dim_freq.get((dim, g), 0), g))
    cause = "not_listed" if (len(siblings) >= 2 and rng.random() < NOT_LISTED_RATE) else None
    want = rng.randint(13, 25) if rng.random() < 0.10 else rng.randint(2, 12)
    take = want if cause else want - 1
    values = siblings[:max(1, min(take, len(siblings)))]
    if not cause:
        values = values + [gold]
    if len(values) < 2:
        return None
    rng.shuffle(values)
    question = rng.choice(TEMPLATES[family]).format(obj=name)
    field = {"id": "answer", "type": "choice", "question": question,
             "options": [{"value": v} for v in values]}
    request = {"request_id": f"vgattr-{split}-{object_id}", "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{image_id}-{object_id}-{dim}", "source": SOURCE, "source_split": split,
            "source_group": str(image_id), "family": family, "license": LICENSE,
            "images": [image], "request": request, "target": None if cause else gold,
            "abstention_cause": cause, "source_answer": gold, "partition": partition,
            "attribute_dimension": dim, "object_name": name}


def select(cands_by_image: dict, quota: int, caps: dict) -> list:
    order = sorted(cands_by_image, key=lambda i: stable_rank(SEED, "img", i))
    fam_counts: Counter = Counter()
    picked = []
    for image_id in order:
        if len(picked) >= quota:
            break
        options = sorted(cands_by_image[image_id], key=lambda c: stable_rank(SEED, c[1], c[4]))
        best = None
        for c in options:
            if fam_counts[c[3]] < caps.get(c[3], 0):
                best = c
                break
        if best is None:
            continue
        fam_counts[best[3]] += 1
        picked.append(best)
    return picked


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()
    rng = random.Random(SEED)
    cached = {int(p.stem) for p in VG_PROC.glob("*.jpg")}
    gqa_train, gqa_val = gqa_image_split()
    print(f"gqa train images={len(gqa_train)} val images={len(gqa_val)} cached={len(cached)}", flush=True)

    with zipfile.ZipFile(ATTR_ZIP) as z:
        vg = json.loads(z.read("attributes.json"))
    print(f"vg images with attributes: {len(vg)}", flush=True)

    all_cands = list(candidates(vg))
    cooc = Counter((c[2], c[5]) for c in all_cands)
    dim_freq = Counter((c[4], c[5]) for c in all_cands)
    print(f"candidates={len(all_cands)} families={Counter(c[3] for c in all_cands)}", flush=True)

    fit_by, test_by = defaultdict(list), defaultdict(list)
    for c in all_cands:
        img = c[0]
        if img not in cached:
            continue
        if img in gqa_train and img not in excl["vg"]:
            fit_by[img].append(c)
        elif img in gqa_val:
            test_by[img].append(c)
    print(f"fit images={len(fit_by)} test images={len(test_by)}", flush=True)

    total = TRAIN + DEV
    fit = select(fit_by, total, {f: round(total * s) for f, s in FAMILY_SHARE.items()})
    test = select(test_by, TEST_PER_FAMILY * len(FAMILY_SHARE),
                  {f: TEST_PER_FAMILY for f in FAMILY_SHARE})
    print(f"selected fit={len(fit)} ({Counter(c[3] for c in fit)}) test={len(test)}", flush=True)
    assert not ({c[0] for c in fit} & {c[0] for c in test})

    images = {}
    for c in fit + test:
        if c[0] not in images:
            images[c[0]] = store_cached(IMAGES, VG_PROC / f"{c[0]}.jpg")

    records = []
    for cands, split, partition in ((fit, "train", "train"), (test, "train", "test")):
        for c in cands:
            image = images.get(c[0])
            if image is None:
                continue
            if partition != "test" and image["sha256"] in excl["sha256"]:
                continue
            rec = make_record(c, split, partition, rng, image, cooc, dim_freq)
            if rec:
                records.append(rec)

    fit_recs = [r for r in records if r["partition"] == "train"]
    dev_groups = set(sorted({r["source_group"] for r in fit_recs},
                            key=lambda g: stable_rank(SEED, "dev", g))[:DEV])
    for r in fit_recs:
        if r["source_group"] in dev_groups:
            r["partition"] = "dev"
    records.sort(key=lambda r: r["id"])
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps({"records": len(records),
                      "partition": dict(Counter(r["partition"] for r in records)),
                      "family": dict(Counter(r["family"] for r in records)),
                      "dimension": dict(Counter(r["attribute_dimension"] for r in records)),
                      "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
                      "options": dict(sorted(Counter(len(r["request"]["fields"][0]["options"])
                                                     for r in records).items()))}, indent=2))
