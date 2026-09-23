"""VSR (Visual Spatial Reasoning) -> decision-v1 `boolean` records.

Each VSR item is a caption asserting a spatial relation between two objects in a COCO 2017
photo, plus a human true/false judgement.  It becomes the boolean question
"Is it true that: <caption>?" with target true/false, kept 50/50 inside every partition.
Families are the relation categories of the VSR paper (Adjacency, Directional, Orientation,
Projective, Proximity, Topological, Unallocated).
"""
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import (CACHE_ROOT, DATA_ROOT, fetch_normalised_many, load_exclusions,
                             stable_rank, store_normalised)
from vision_decision.contracts import Request

SOURCE = "vsr"
SEED = 20260922
LICENSE = "Apache-2.0"
TEST_TOTAL, TEST_PER_FAMILY, DEV_FRACTION = 1_500, 300, 0.03
MAX_PER_IMAGE = 3               # VSR is small: the spec allows up to 3 questions per image
OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
SRC = CACHE_ROOT / "vsr/random"

CATEGORY = {
    "adjacency": ["adjacent to", "alongside", "at the side of", "at the right side of",
                  "at the left side of", "attached to", "at the back of", "ahead of", "against",
                  "at the edge of"],
    "directional": ["off", "past", "toward", "down", "away from", "along", "around", "into",
                    "across", "across from", "down from", "through"],
    "orientation": ["facing", "facing away from", "parallel to", "perpendicular to", "congruent"],
    "projective": ["on top of", "beneath", "beside", "behind", "left of", "right of", "under",
                   "in front of", "below", "above", "over", "in the middle of"],
    "proximity": ["by", "close to", "near", "far from", "far away from"],
    "topological": ["connected to", "detached from", "has as a part", "part of", "contains",
                    "within", "at", "on", "in", "with", "surrounding", "among", "consists of",
                    "out of", "between", "inside", "outside", "touching"],
    "unallocated": ["beyond", "next to", "opposite to", "enclosed by"],
}
REL2FAM = {r: f for f, rs in CATEGORY.items() for r in rs}


def load() -> list[dict]:
    rows = []
    for split in ("train", "dev", "test"):
        for line in (SRC / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            r["vsr_split"] = split
            r["coco_id"] = int(Path(r["image"]).stem)
            r["family"] = REL2FAM.get(r["relation"], "unallocated")
            r["votes_true"] = len(ast.literal_eval(r["vote_true_validator_id"]))
            r["votes_false"] = len(ast.literal_eval(r["vote_false_validator_id"]))
            rows.append(r)
    return rows


def balance(rows: list[dict]) -> list[dict]:
    """Keep true/false at 50/50 by dropping the surplus deterministically."""
    pos = sorted([r for r in rows if r["label"] == 1], key=lambda r: stable_rank(SEED, "b", r["caption"]))
    neg = sorted([r for r in rows if r["label"] == 0], key=lambda r: stable_rank(SEED, "b", r["caption"]))
    n = min(len(pos), len(neg))
    return pos[:n] + neg[:n]


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()
    rng = random.Random(SEED)
    rows = load()
    print(f"vsr rows={len(rows)} images={len({r['coco_id'] for r in rows})} "
          f"families={Counter(r['family'] for r in rows)}", flush=True)

    by_image = defaultdict(list)
    for r in rows:
        by_image[r["coco_id"]].append(r)
    for v in by_image.values():
        v.sort(key=lambda r: stable_rank(SEED, r["caption"]))

    # VSR's own random splits share images, so partitions are assigned per image.  Images that
    # occur only in VSR's held-out test split are used first for our test partition.
    test_only = {i for i, v in by_image.items() if all(r["vsr_split"] == "test" for r in v)}
    test_images, taken = [], 0
    for i in sorted(test_only, key=lambda i: stable_rank(SEED, "t", i)):
        if taken >= TEST_TOTAL:
            break
        test_images.append(i)
        taken += min(MAX_PER_IMAGE, len(by_image[i]))
    test_images = set(test_images)

    fit_images = [i for i in by_image if i not in test_images and i not in excl["coco"]]
    fit_images.sort(key=lambda i: stable_rank(SEED, "f", i))
    dev_cut = round(len(fit_images) * DEV_FRACTION)
    dev_images = set(fit_images[:dev_cut])

    selected = []
    for image_id, group in by_image.items():
        if image_id not in test_images and image_id not in set(fit_images):
            continue
        partition = "test" if image_id in test_images else ("dev" if image_id in dev_images else "train")
        for r in group[:MAX_PER_IMAGE]:
            selected.append(dict(r, partition=partition))

    # family cap on test, 50/50 balance inside every partition
    kept = []
    fam_counts: Counter = Counter()
    for r in sorted([x for x in selected if x["partition"] == "test"],
                    key=lambda r: stable_rank(SEED, "cap", r["caption"])):
        if fam_counts[r["family"]] < TEST_PER_FAMILY:
            fam_counts[r["family"]] += 1
            kept.append(r)
    for part in ("train", "dev"):
        kept += [x for x in selected if x["partition"] == part]
    final = []
    for part in ("train", "dev", "test"):
        final += balance([r for r in kept if r["partition"] == part])
    print(f"selected={len(final)} {Counter(r['partition'] for r in final)}", flush=True)

    need = sorted({(r["coco_id"], "train2017" if "train2017" in r["image_link"] else "val2017")
                   for r in final})
    print(f"fetching {len(need)} COCO 2017 images...", flush=True)
    images = {}
    for key, got in fetch_normalised_many([((c, s), "https://s3.amazonaws.com/images.cocodataset.org/"
                                            f"{s}/{c:012d}.jpg") for c, s in need], log_every=1000):
        images[key[0]] = store_normalised(IMAGES, *got) if got else None

    records = []
    for r in final:
        image = images.get(r["coco_id"])
        if image is None:
            continue
        if r["partition"] != "test" and image["sha256"] in excl["sha256"]:
            continue
        field = {"id": "answer", "type": "boolean",
                 "question": f"Is it true that: {r['caption'].rstrip('.')}?"}
        request = {"request_id": f"vsr-{r['vsr_split']}-{stable_rank(r['caption'])[:16]}",
                   "state": {}, "fields": [field]}
        Request.model_validate(request)
        records.append({"id": f"{SOURCE}:{r['coco_id']}-{stable_rank(r['caption'])[:12]}",
                        "source": SOURCE, "source_split": r["vsr_split"],
                        "source_group": str(r["coco_id"]), "family": r["family"], "license": LICENSE,
                        "images": [image], "request": request, "target": r["label"] == 1,
                        "abstention_cause": None, "source_answer": "true" if r["label"] else "false",
                        "partition": r["partition"], "relation": r["relation"],
                        "validator_votes": [r["votes_true"], r["votes_false"]]})

    records.sort(key=lambda r: r["id"])
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps({"records": len(records),
                      "partition": dict(Counter(r["partition"] for r in records)),
                      "family": dict(Counter(r["family"] for r in records)),
                      "targets": dict(Counter(str(r["target"]) for r in records)),
                      "by_partition_target": {f"{r}": n for r, n in sorted(Counter(
                          (x["partition"], str(x["target"])) for x in records).items())}}, indent=2))
