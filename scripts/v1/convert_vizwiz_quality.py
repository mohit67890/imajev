"""VizWiz-QualityIssues -> decision-v1 `vizwiz_quality`: can this photo be used at all?

Each image carries five annotators' votes on six flaw types (blur, framing, obstruction,
too dark, too bright, rotation) plus an "unrecognizable" vote. Only high-agreement labels are
converted: a flaw is present at >=4/5 votes and absent at 0/5; the ambiguous middle is dropped.

  image_flaw       boolean per flaw, balanced, with inverted phrasings so "yes" is not a prior
  image_flaw       choice - which single flaw is the problem (option deletion -> not_listed)
  image_usability  ordinal - how many of the five viewers could make out what the photo shows

Images are shared with VizWiz-VQA, so this converter downloads nothing: it reuses
.cache/datasets/v1/vizwiz/{train,val}-images fetched by convert_vizwiz.py.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import store_image, write_jsonl  # noqa: E402

SOURCE = "vizwiz_quality"
SEED = 20260923
LICENSE = "CC-BY-4.0"
ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip"
CACHE = Path(".cache/datasets/v1/vizwiz_quality")
IMAGE_CACHE = Path(".cache/datasets/v1/vizwiz")
OUT = Path("data/decision-v1/vizwiz_quality")
IMAGES = OUT / "images"

PRESENT, ABSENT = 4, 0          # votes out of five
DEV_FRACTION = 0.03
NOT_LISTED_RATE = 0.25          # applied to the choice and ordinal records only
TEST_PER_FAMILY = 300

FLAW_NAME = {"BLR": "blur", "FRM": "bad framing", "OBS": "something blocking the camera",
             "DRK": "too dark", "BRT": "too bright", "ROT": "rotated"}
# (template, inverted) - an inverted phrasing flips the target, so "yes" carries no prior.
TEMPLATES = {
    "BLR": [("Is this photo blurry?", False),
            ("Is this picture out of focus?", False),
            ("Is the image too blurred to inspect?", False),
            ("Is this photo sharp and in focus?", True),
            ("Is this picture free of motion blur or focus blur?", True)],
    "FRM": [("Is the subject of this photo badly framed or cut off?", False),
            ("Is part of what the photographer wanted to show missing from the frame?", False),
            ("Does this picture cut off the thing it is meant to show?", False),
            ("Is the subject fully inside the frame of this photo?", True),
            ("Is this photo framed well enough to show the whole subject?", True)],
    "OBS": [("Is something blocking the camera in this photo, such as a finger over the lens?", False),
            ("Is part of this picture covered by an obstruction in front of the lens?", False),
            ("Is the view in this photo obstructed by something right in front of the camera?", False),
            ("Is the camera's view in this photo unobstructed?", True),
            ("Is this picture free of any obstruction over the lens?", True)],
    "DRK": [("Is this photo too dark to see properly?", False),
            ("Was this picture taken with too little light?", False),
            ("Is this image underexposed?", False),
            ("Is this photo bright enough to see what it shows?", True),
            ("Is the exposure of this picture adequate rather than too dark?", True)],
    "BRT": [("Is this photo too bright or overexposed?", False),
            ("Is this picture washed out by too much light?", False),
            ("Is this image blown out by glare or flash?", False),
            ("Is this photo free of overexposure?", True),
            ("Is the lighting in this picture reasonable rather than too bright?", True)],
    "ROT": [("Is this photo rotated so that the subject is not upright?", False),
            ("Would this picture need to be turned to view it the right way up?", False),
            ("Is the content of this image sideways or upside down?", False),
            ("Is this photo already the right way up?", True),
            ("Is the subject of this picture upright as it stands?", True)],
}
CHOICE_TEMPLATES = [
    "What is the main problem with this photo?",
    "This picture was rejected by a quality check. Which of these is the reason?",
    "Which of these quality problems does this photo have?",
    "A viewer could not use this picture. What is wrong with it?",
    "Which flaw best describes what is wrong with this image?",
]
ORDINAL_TEMPLATES = [
    "How usable is this photo - could a viewer make out what it shows?",
    "Rate how usable this picture is for telling what it shows.",
    "Five people were asked whether they could tell what this photo shows. How usable is it?",
    "How well can the content of this photo be made out?",
    "Judge this image's usability: can a viewer tell what is in it?",
]
LEVELS = [
    {"value": 0, "description": "unusable - no viewer could tell what the photo shows"},
    {"value": 1, "description": "poor - most viewers could not tell what it shows"},
    {"value": 2, "description": "borderline - some viewers could tell, some could not"},
    {"value": 3, "description": "usable - every viewer could tell what it shows"},
]
# how many of five annotators marked the image unrecognizable -> level
VOTE_TO_LEVEL = {5: 0, 4: 1, 3: 1, 2: 2, 1: 2, 0: 3}
# boolean quota per flaw, per side (positive and negative get the same count)
BOOLEAN_QUOTA = {"BLR": 900, "FRM": 900, "ROT": 400, "DRK": 300, "BRT": 200, "OBS": 200}
CHOICE_QUOTA, ORDINAL_QUOTA, FALSE_PREMISE_QUOTA = 2000, 2500, 900


def hkey(*parts):
    return hashlib.sha256((":".join(str(p) for p in parts)).encode()).hexdigest()


def order(rows, tag):
    return sorted(rows, key=lambda r: hkey(SEED, tag, r["image"]))


def load(split):
    with zipfile.ZipFile(CACHE / "annotations.zip") as z:
        return json.loads(z.read(f"{split}.json"))


def record(rid, split, image, family, field, target, cause, answer, partition, distribution=None):
    row = {"id": rid, "source": SOURCE, "source_split": split, "source_group": image,
           "family": family, "license": LICENSE, "images": None,
           "request": {"request_id": rid[:128], "state": {}, "fields": [field]},
           "target": target, "abstention_cause": cause, "source_answer": answer,
           "partition": partition, "_image": image}
    if distribution:
        row["target_distribution"] = distribution
    return row


def build(rows, available, split, partition_of, quotas, choice_quota, ordinal_quota,
          false_premise_quota, rng):
    used = Counter()          # at most two questions per image
    out = []

    rows = [r for r in rows if r["image"] in available]
    for flaw, quota in quotas.items():
        for want_present in (True, False):
            pool = [r for r in order(rows, "b" + flaw)
                    if (r["flaws"][flaw] >= PRESENT) == want_present
                    and (r["flaws"][flaw] >= PRESENT or r["flaws"][flaw] == ABSENT)]
            taken = 0
            for r in pool:
                if taken >= quota:
                    break
                if used[r["image"]] >= 1:
                    continue
                text, inverted = TEMPLATES[flaw][int(hkey(SEED, "t", flaw, r["image"]), 16) % len(TEMPLATES[flaw])]
                field = {"id": "answer", "type": "boolean", "question": text}
                out.append(record(f"{SOURCE}:{Path(r['image']).stem}:flaw_{flaw.lower()}", split, r["image"],
                                  "image_flaw", field, want_present != inverted, None,
                                  f"{flaw}={r['flaws'][flaw]}/5", partition_of(r["image"])))
                used[r["image"]] += 1
                taken += 1

    core = list(FLAW_NAME)
    singles = [r for r in order(rows, "c") if sum(r["flaws"][f] >= PRESENT for f in core) == 1]
    taken = 0
    for r in singles:
        if taken >= choice_quota:
            break
        if used[r["image"]] >= 2:
            continue
        gold_code = next(f for f in core if r["flaws"][f] >= PRESENT)
        gold = FLAW_NAME[gold_code]
        others = [FLAW_NAME[f] for f in core if f != gold_code]
        rng.shuffle(others)
        cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
        n = rng.randint(2, 6)
        values = others[: n if cause else n - 1] + ([] if cause else [gold])
        if len(values) < 2:
            continue
        rng.shuffle(values)
        field = {"id": "answer", "type": "choice",
                 "question": CHOICE_TEMPLATES[int(hkey(SEED, "q", r["image"]), 16) % len(CHOICE_TEMPLATES)],
                 "options": [{"value": v} for v in values]}
        out.append(record(f"{SOURCE}:{Path(r['image']).stem}:main_flaw", split, r["image"], "image_flaw",
                          field, None if cause else gold, cause, gold, partition_of(r["image"])))
        used[r["image"]] += 1
        taken += 1

    # Asking "what is wrong with this photo?" of an image five annotators marked flawless is a
    # false premise the annotation proves: no flaw vote at all, and NON carried by >=4 of 5.
    clean = [r for r in order(rows, "fp")
             if r["flaws"]["NON"] >= PRESENT and all(r["flaws"][f] == 0 for f in core)]
    taken = 0
    for r in clean:
        if taken >= false_premise_quota:
            break
        if used[r["image"]] >= 2:
            continue
        values = [FLAW_NAME[f] for f in core]
        rng.shuffle(values)
        values = values[: rng.randint(2, 6)]
        field = {"id": "answer", "type": "choice",
                 "question": CHOICE_TEMPLATES[int(hkey(SEED, "fq", r["image"]), 16) % len(CHOICE_TEMPLATES)],
                 "options": [{"value": v} for v in values]}
        out.append(record(f"{SOURCE}:{Path(r['image']).stem}:no_flaw", split, r["image"], "image_flaw",
                          field, None, "false_premise", f"NON={r['flaws']['NON']}/5",
                          partition_of(r["image"])))
        used[r["image"]] += 1
        taken += 1

    by_level = defaultdict(list)
    for r in order(rows, "o"):
        by_level[VOTE_TO_LEVEL[r["unrecognizable"]]].append(r)
    per_level = max(1, ordinal_quota // len(LEVELS))
    taken = 0
    for level in sorted(by_level):
        picked = 0
        for r in by_level[level]:
            if picked >= per_level or taken >= ordinal_quota:
                break
            if used[r["image"]] >= 2:
                continue
            cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
            if cause:
                windows = [w for w in ([0, 1], [1, 2], [2, 3], [0, 1, 2], [1, 2, 3])
                           if level not in w]
                window = windows[rng.randrange(len(windows))]
            else:
                lo = max(0, min(level - rng.randint(0, 2), len(LEVELS) - 2))
                hi = min(len(LEVELS) - 1, max(level + rng.randint(0, 2), lo + 1))
                window = list(range(lo, hi + 1))
            field = {"id": "answer", "type": "ordinal",
                     "question": ORDINAL_TEMPLATES[int(hkey(SEED, "r", r["image"]), 16) % len(ORDINAL_TEMPLATES)],
                     "levels": [LEVELS[i] for i in window]}
            out.append(record(f"{SOURCE}:{Path(r['image']).stem}:usability", split, r["image"],
                              "image_usability", field, None if cause else level, cause,
                              f"unrecognizable={r['unrecognizable']}/5", partition_of(r["image"])))
            used[r["image"]] += 1
            picked += 1
            taken += 1
    return out


def main():
    if not (CACHE / "annotations.zip").exists():
        raise SystemExit(f"download {ANNOTATIONS_URL} to {CACHE/'annotations.zip'} first")
    available = {split: {p.name for p in (IMAGE_CACHE / f"{split}-images").glob("*.jpg")}
                 for split in ("train", "val")}
    if not available["train"]:
        raise SystemExit("run scripts/v1/convert_vizwiz.py first: this converter reuses its images")

    rng = random.Random(SEED)
    dev_cut = lambda image: "dev" if int(hkey(SEED, "p", image), 16) % 1000 < DEV_FRACTION * 1000 else "train"
    rows = build(load("train"), available["train"], "train", dev_cut,
                 BOOLEAN_QUOTA, CHOICE_QUOTA, ORDINAL_QUOTA, FALSE_PREMISE_QUOTA, rng)
    test_quota = {f: max(20, TEST_PER_FAMILY // (2 * len(BOOLEAN_QUOTA))) for f in BOOLEAN_QUOTA}
    rows += build(load("val"), available["val"], "val", lambda image: "test",
                  test_quota, TEST_PER_FAMILY // 2, TEST_PER_FAMILY, TEST_PER_FAMILY // 4, rng)

    cache, final = {}, []
    for r in rows:
        name = r.pop("_image")
        split = "train" if r["source_split"] == "train" else "val"
        if name not in cache:
            cache[name] = store_image((IMAGE_CACHE / f"{split}-images" / name).read_bytes(), IMAGES)
        r["images"] = [cache[name]]
        final.append(r)
    final.sort(key=lambda r: r["id"])
    write_jsonl(OUT / "records.jsonl", final)

    summary = dict(records=len(final), images=len(cache),
                   partitions=dict(Counter(r["partition"] for r in final)),
                   families=dict(Counter(r["family"] for r in final)),
                   field_types=dict(Counter(r["request"]["fields"][0]["type"] for r in final)),
                   abstention=dict(Counter(str(r["abstention_cause"]) for r in final)),
                   boolean_targets=dict(Counter(str(r["target"]) for r in final
                                                if r["request"]["fields"][0]["type"] == "boolean")))
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
