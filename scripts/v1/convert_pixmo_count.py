"""PixMo-Count -> decision-v1 `ordinal` counting records.

Same ordinal shape as convert_tallyqa.py: a window of 2..7 consecutive integers around the gold
count, sometimes placed so the gold is outside it (-> `not_listed`).  Questions are templated
from the `label` field with six phrasings.  Images are fetched one by one from the Flickr URLs
the dataset ships; dead URLs are skipped.

⚠️ Train labels are machine-generated (Detic object detector) - see README.md, which also carries
the licence stop-flag for this source.
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import (CACHE_ROOT, DATA_ROOT, fetch_normalised_many, load_exclusions,
                             number_word, set_fetch_deadline, stable_rank, store_cached)
from vision_decision.contracts import Request

SOURCE = "pixmo_count"
SEED = 20260922
LICENSE = "ODC-BY-1.0"
TRAIN, DEV, TEST = 15_000, 450, 300
ATTEMPT_MULTIPLIER = 1.15       # over-fetch to absorb dead Flickr URLs (~2% measured)
NOT_LISTED_RATE = 0.17
MAX_LEVEL = 25
PEOPLE_SHARE_CAP = 0.40
FETCH_BUDGET_SECONDS = float(os.environ.get("PIXMO_FETCH_BUDGET", 900))  # Flickr throttles; see README

OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
SRC = CACHE_ROOT / "pixmo_count/hf/data"
PROC = CACHE_ROOT / "pixmo_count/proc"          # normalised bytes, so a re-run costs no bandwidth

TEMPLATES = [
    "How many {label} are in this image?",
    "How many {label} can you see in the photo?",
    "Count the {label} in this picture.",
    "What is the number of {label} visible here?",
    "In this photograph, how many {label} are there?",
    "How many {label} does this image show?",
]


def levels_for(gold: int, rng: random.Random) -> tuple[list[int], bool]:
    k = rng.randint(2, 7)
    if rng.random() < NOT_LISTED_RATE:
        starts = [gold + 1 + rng.randint(0, 3)]
        if gold - k - 3 >= 0:
            starts.append(max(0, gold - k - rng.randint(0, 3)))
        start = rng.choice(starts)
        if start + k - 1 > MAX_LEVEL:
            start = max(0, gold - k - rng.randint(0, 3))
            if start + k - 1 >= gold:
                start = max(0, gold - k + 1)
                return list(range(start, start + k)), True
        return list(range(start, start + k)), False
    start = max(0, gold - rng.randint(0, k - 1))
    if start + k - 1 > MAX_LEVEL:
        start = MAX_LEVEL - k + 1
    return list(range(start, start + k)), True


def load(split: str) -> list[dict]:
    t = pq.read_table(SRC / f"{split}-00000-of-00001.parquet",
                      columns=["image_url", "image_sha256", "count", "label"])
    return t.to_pylist()


def select(rows: list[dict], quota: int) -> list[dict]:
    """Deterministic, one record per image, with the dominant 'people' label capped."""
    seen, picked, people = set(), [], 0
    for r in sorted(rows, key=lambda r: stable_rank(SEED, r["image_sha256"], r["label"])):
        if len(picked) >= quota:
            break
        if r["image_sha256"] in seen:
            continue
        if r["label"] == "people" and people >= PEOPLE_SHARE_CAP * quota:
            continue
        seen.add(r["image_sha256"])
        people += r["label"] == "people"
        picked.append(r)
    return picked


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()
    rng = random.Random(SEED)

    test_rows = load("test")
    test_pool = select(test_rows, round(TEST * 1.6))
    # PixMo's train and test splits share a handful of images (identical image_sha256),
    # so the test images are removed from the fit pool before selection.
    held_out = {r["image_sha256"] for r in test_rows}
    fit_pool = select([r for r in load("train") if r["image_sha256"] not in held_out],
                      round((TRAIN + DEV) * ATTEMPT_MULTIPLIER))
    print(f"attempting fit={len(fit_pool)} test={len(test_pool)} urls", flush=True)

    records = []
    PROC.mkdir(parents=True, exist_ok=True)
    set_fetch_deadline(FETCH_BUDGET_SECONDS)
    for pool, partition, quota in ((test_pool, "test", TEST), (fit_pool, "train", TRAIN + DEV)):
        made = []
        missing = [r for r in pool if not (PROC / f"{r['image_sha256']}.jpg").exists()]
        print(f"  {partition}: {len(pool)} urls, {len(pool) - len(missing)} already cached", flush=True)
        for sha, got in fetch_normalised_many([(r["image_sha256"], r["image_url"]) for r in missing],
                                              workers=6, log_every=500):
            if got is not None:
                (PROC / f"{sha}.jpg").write_bytes(got[0])
        for r in pool:
            sha = r["image_sha256"]
            cached = PROC / f"{sha}.jpg"
            if not cached.exists():
                continue
            image = store_cached(IMAGES, cached)
            if image is None:
                continue
            if partition != "test" and (image["sha256"] in excl["sha256"] or sha in excl["sha256"]):
                continue
            gold = int(r["count"])
            levels, present = levels_for(gold, rng)
            if (gold in levels) != present:
                continue
            words = rng.random() < 0.6
            field = {"id": "answer", "type": "ordinal",
                     "question": rng.choice(TEMPLATES).format(label=r["label"]),
                     "levels": [{"value": v, "description": number_word(v) if words else str(v)}
                                for v in levels]}
            request = {"request_id": f"pixmocount-{partition}-{sha[:24]}", "state": {}, "fields": [field]}
            Request.model_validate(request)
            made.append({"id": f"{SOURCE}:{sha[:24]}", "source": SOURCE,
                         "source_split": "test" if partition == "test" else "train",
                         "source_group": sha, "family": "counting", "license": LICENSE,
                         "images": [image], "request": request,
                         "target": gold if present else None,
                         "abstention_cause": None if present else "not_listed",
                         "source_answer": str(gold), "partition": partition,
                         "label": r["label"], "labels_are_machine_generated": partition != "test"})
        made.sort(key=lambda r: stable_rank(SEED, "trim", r["id"]))
        print(f"  {partition}: fetched+usable={len(made)} keeping {min(quota, len(made))}", flush=True)
        records += made[:quota]

    fit = [r for r in records if r["partition"] != "test"]
    dev_n = min(DEV, max(10, round(0.03 * len(fit))))   # 3% of whatever the fetch actually yielded
    dev_groups = set(sorted({r["source_group"] for r in fit},
                            key=lambda g: stable_rank(SEED, "dev", g))[:dev_n])
    for r in fit:
        if r["source_group"] in dev_groups:
            r["partition"] = "dev"
    records.sort(key=lambda r: r["id"])
    keep = {Path(r["images"][0]["image"]).name for r in records}
    for f in IMAGES.iterdir():
        if f.name not in keep:
            f.unlink()
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps({"records": len(records),
                      "partition": dict(Counter(r["partition"] for r in records)),
                      "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
                      "labels": Counter(r["label"] for r in records).most_common(8),
                      "levels": dict(sorted(Counter(len(r["request"]["fields"][0]["levels"])
                                                    for r in records).items()))}, indent=2))
