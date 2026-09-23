"""TallyQA -> decision-v1 `ordinal` counting records.

Each question keeps its original wording; the answer becomes an ordinal field whose levels are
a window of 2..7 consecutive integers positioned at random around the gold count.  On ~17% of
records the window is placed entirely above (or below) the gold count, which makes the honest
answer `not_listed`.

Images are COCO 2014 (fetched from the public S3 bucket) and Visual Genome (taken from the
shared normalised cache built by scripts/v1/build_vg_image_cache.py).
"""
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import (CACHE_ROOT, DATA_ROOT, VG_PROC, coco_url, fetch_many, load_exclusions,
                             number_word, stable_rank, store_cached, store_normalised, normalise,
                             vg_image_data)
from vision_decision.contracts import Request

SOURCE = "tallyqa"
SEED = 20260922
LICENSE = "Apache-2.0"          # TallyQA annotations; images are COCO / Visual Genome, see README
TRAIN, DEV, TEST_PER_FAMILY = 38_500, 1_155, 300
NOT_LISTED_RATE = 0.17
MAX_LEVEL = 20
VALUE_SHARE_CAP = 0.16          # flattens TallyQA's heavy 0-3 answer prior
COMPLEX_TARGET = 0.30           # share of "complex" (attribute/relation) counting questions
COCO_SHARE = 0.55

OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
ZIP = CACHE_ROOT / "tallyqa/tallyqa.zip"
COCO_RAW = CACHE_ROOT / "tallyqa/coco_raw"

SIMPLE_SOURCES = {"imported_genome", "imported_vqa", "tdiuc_templates"}


def family_of(row: dict) -> str:
    """test.json ships `issimple`; train.json does not, so data_source is the documented proxy."""
    if "issimple" in row:
        return "counting_simple" if row["issimple"] else "counting_complex"
    return "counting_simple" if row["data_source"] in SIMPLE_SOURCES else "counting_complex"


def levels_for(gold: int, rng: random.Random) -> tuple[list[int], bool]:
    """Window of consecutive ints; returns (levels, gold_present)."""
    k = rng.randint(2, 7)
    if rng.random() < NOT_LISTED_RATE:
        starts = [gold + 1 + rng.randint(0, 3)]
        if gold - k - 3 >= 0:
            starts.append(max(0, gold - k - rng.randint(0, 3)))
        start = rng.choice(starts)
        if start + k - 1 > MAX_LEVEL:
            start = max(0, gold - k - rng.randint(0, 3))
            if start + k - 1 >= gold:
                return list(range(max(0, gold - k + 1), max(0, gold - k + 1) + k)), True
        return list(range(start, start + k)), False
    start = max(0, gold - rng.randint(0, k - 1))
    if start + k - 1 > MAX_LEVEL:
        start = MAX_LEVEL - k + 1
    return list(range(start, start + k)), True


def make_record(row: dict, split: str, partition: str, rng: random.Random, image: dict) -> dict | None:
    gold = int(row["answer"])
    levels, present = levels_for(gold, rng)
    if not (2 <= len(levels) <= 7) or levels != sorted(set(levels)) or levels[0] < 0:
        return None
    if present and gold not in levels:
        return None
    if not present and gold in levels:
        return None
    words = rng.random() < 0.6
    field = {"id": "answer", "type": "ordinal", "question": row["question"],
             "levels": [{"value": v, "description": number_word(v) if words else str(v)} for v in levels]}
    request = {"request_id": f"tallyqa-{split}-{row['question_id']}", "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{split}-{row['question_id']}", "source": SOURCE, "source_split": split,
            "source_group": row["group"], "family": family_of(row), "license": LICENSE,
            "images": [image], "request": request,
            "target": gold if present else None,
            "abstention_cause": None if present else "not_listed",
            "source_answer": str(gold), "partition": partition,
            "data_source": row["data_source"]}


def annotate(rows: list[dict], coco_of: dict) -> list[dict]:
    """Attach a canonical image identity (a VG photo that is also a COCO photo shares its group)."""
    out = []
    for r in rows:
        if r["image"].startswith("VG_"):
            vg = r["image_id"] - 90_000_000
            cid = coco_of.get(vg)
            r = dict(r, kind="vg", vg=vg, coco=int(cid) if cid else None,
                     group=f"coco:{int(cid)}" if cid else f"vg:{vg}")
        else:
            r = dict(r, kind="coco", vg=None, coco=r["image_id"],
                     group=f"coco:{r['image_id']}", split2014=r["image"].split("/")[0][:-4])
        out.append(r)
    return out


def choose(rows: list[dict], quota: int, excl, want_coco: int | None, family_cap: dict | None,
           available: set | None) -> list[dict]:
    """One question per image; the gold-count prior and the simple/complex mix are both steered.

    TallyQA's answers pile up on 0-3, which would make "guess the smallest level" a winning
    heuristic, so no gold value may exceed VALUE_SHARE_CAP of the quota.  The cap is relaxed in
    later passes only if the quota cannot otherwise be met.
    """
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["group"]].append(r)
    for v in by_image.values():
        v.sort(key=lambda r: stable_rank(SEED, r["question_id"]))
    order = sorted(by_image, key=lambda g: stable_rank(SEED, "img", g))
    picked, value_counts, fam_counts, coco_taken, used = [], Counter(), Counter(), 0, set()

    def usable(r):
        if r["kind"] == "vg" and available is not None and r["vg"] not in available:
            return False
        if family_cap is not None and fam_counts[family_of(r)] >= family_cap.get(family_of(r), 0):
            return False
        return True

    for share in (VALUE_SHARE_CAP, 0.25, 1.0):
        cap = max(20, round(share * quota))
        for group in order:
            if len(picked) >= quota:
                return picked
            if group in used:
                continue
            cands = by_image[group]
            if excl is not None and any((r["coco"] is not None and r["coco"] in excl["coco"])
                                        or (r["vg"] is not None and r["vg"] in excl["vg"])
                                        for r in cands):
                used.add(group)
                continue
            fits = [r for r in cands if usable(r) and value_counts[int(r["answer"])] < cap]
            if not fits:
                continue
            want_complex = fam_counts["counting_complex"] < COMPLEX_TARGET * max(1, len(picked))
            preferred = [r for r in fits
                         if (family_of(r) == "counting_complex") == want_complex
                         and (want_coco is None
                              or (r["kind"] == "coco") == (coco_taken < want_coco))]
            best = (preferred or fits)[0]
            used.add(group)
            picked.append(best)
            value_counts[int(best["answer"])] += 1
            fam_counts[family_of(best)] += 1
            coco_taken += best["kind"] == "coco"
    return picked


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()
    rng = random.Random(SEED)
    coco_of = {r["image_id"]: r.get("coco_id") for r in vg_image_data()}
    cached = {int(p.stem) for p in VG_PROC.glob("*.jpg")}

    with zipfile.ZipFile(ZIP) as z:
        train_rows = annotate(json.loads(z.read("train.json")), coco_of)
        test_rows = annotate(json.loads(z.read("test.json")), coco_of)
    print(f"train={len(train_rows)} test={len(test_rows)}", flush=True)

    fit = choose(train_rows, TRAIN + DEV, excl, want_coco=round((TRAIN + DEV) * COCO_SHARE),
                 family_cap=None, available=cached)
    test = choose(test_rows, TEST_PER_FAMILY * 2, None, want_coco=None,
                  family_cap={"counting_simple": TEST_PER_FAMILY, "counting_complex": TEST_PER_FAMILY},
                  available=cached)
    assert not ({r["group"] for r in fit} & {r["group"] for r in test}), "image group spans splits"
    print(f"selected fit={len(fit)} test={len(test)} "
          f"coco={sum(r['kind'] == 'coco' for r in fit)}", flush=True)

    # ---- images
    need_vg = {r["vg"] for r in fit + test if r["kind"] == "vg"}
    need_coco = {(r["coco"], r["split2014"]) for r in fit + test if r["kind"] == "coco"}
    images: dict = {}
    for vg in need_vg:
        images[("vg", vg)] = store_cached(IMAGES, VG_PROC / f"{vg}.jpg")
    print(f"vg images stored: {sum(v is not None for v in images.values())}", flush=True)
    COCO_RAW.mkdir(parents=True, exist_ok=True)
    missing = [(c, s) for c, s in sorted(need_coco) if not (COCO_RAW / f"{s}_{c}.jpg").exists()]
    print(f"COCO images needed={len(need_coco)} cached={len(need_coco) - len(missing)}", flush=True)
    for (c, s), blob in fetch_many([((c, s), coco_url(c, s)) for c, s in missing]):
        if blob:
            (COCO_RAW / f"{s}_{c}.jpg").write_bytes(blob)
    for c, s in sorted(need_coco):
        raw = COCO_RAW / f"{s}_{c}.jpg"
        got = normalise(raw.read_bytes()) if raw.exists() else None
        images[("coco", c)] = store_normalised(IMAGES, *got) if got else None

    records = []
    for rows, split, part in ((fit, "train", None), (test, "test", "test")):
        for r in rows:
            image = images.get(("vg", r["vg"])) if r["kind"] == "vg" else images.get(("coco", r["coco"]))
            if image is None:
                continue
            partition = part or "train"
            if partition != "test" and image["sha256"] in excl["sha256"]:
                continue
            rec = make_record(r, split, partition, rng, image)
            if rec:
                records.append(rec)

    fit_recs = [r for r in records if r["partition"] != "test"]
    dev_groups, acc = set(), 0
    for g in sorted({r["source_group"] for r in fit_recs}, key=lambda g: stable_rank(SEED, "dev", g)):
        if acc >= DEV:
            break
        dev_groups.add(g)
        acc += 1
    for r in fit_recs:
        if r["source_group"] in dev_groups:
            r["partition"] = "dev"
    records.sort(key=lambda r: r["id"])
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(json.dumps({"records": len(records),
                      "partition": dict(Counter(r["partition"] for r in records)),
                      "family": dict(Counter(r["family"] for r in records)),
                      "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
                      "levels": dict(sorted(Counter(len(r["request"]["fields"][0]["levels"])
                                                    for r in records).items())),
                      "gold": dict(sorted(Counter(r["source_answer"] for r in records).items(),
                                          key=lambda kv: int(kv[0])))}, indent=2))
