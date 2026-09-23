"""GQA train_balanced -> decision-v1 records.

Yes/no questions become `boolean` fields (kept 50/50); every other question becomes a
`choice` field whose distractors come from GQA's own question grouping: the most frequent
other answers inside the same `groups.local` bucket (e.g. `10q-shirt_color`), then
`groups.global`, then `types.detailed`, then the semantic family.  GQA's published
Validity/Plausibility scopes were *not* reachable at a usable speed - see README.md.

Images are the Visual Genome photographs embedded in the `lmms-lab-encoder/GQA` parquet
mirror, normalised once by scripts/v1/build_vg_image_cache.py.
"""
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")
from common_vg_group import CACHE_ROOT, DATA_ROOT, VG_PROC, load_exclusions, stable_rank, store_cached
from vision_decision.contracts import Request

SOURCE = "gqa"
SEED = 20260922
LICENSE = "CC-BY-4.0"
TRAIN, DEV, TEST_PER_FAMILY = 80_000, 2_400, 300
CHOICE_TARGET = 0.60          # keeps not-listed at ~28% of choice -> ~17% of the source
NOT_LISTED_RATE = 0.28
MAX_PER_IMAGE = 2

FAMILY = {"rel": "relation", "attr": "attribute", "obj": "object_presence",
          "cat": "category", "global": "global_scene"}
OUT = DATA_ROOT / SOURCE
IMAGES = OUT / "images"
INSTR = CACHE_ROOT / "gqa/hf"


def load(split_dir: str) -> list[dict]:
    rows = []
    for shard in sorted((INSTR / split_dir).glob("*.parquet")):
        t = pq.read_table(shard, columns=["id", "imageId", "question", "answer", "groups", "types"])
        d = t.to_pydict()
        for qid, img, q, a, g, ty in zip(d["id"], d["imageId"], d["question"], d["answer"],
                                         d["groups"], d["types"]):
            rows.append({"qid": qid, "img": img, "q": q, "a": a,
                         "local": (g or {}).get("local") or "", "glob": (g or {}).get("global") or "",
                         "detailed": (ty or {}).get("detailed") or "",
                         "sem": (ty or {}).get("semantic") or ""})
    return rows


def build_pools(rows: list[dict]) -> dict:
    """Answer frequency, most frequent first, at four widening scopes."""
    scopes = {"local": defaultdict(Counter), "glob": defaultdict(Counter),
              "detailed": defaultdict(Counter), "sem": defaultdict(Counter)}
    for r in rows:
        a = r["a"]
        if a in ("yes", "no") or not a or len(a) > 128:
            continue
        for key in scopes:
            if r[key]:
                scopes[key][r[key]][a] += 1
    return {k: {g: [a for a, _ in c.most_common(60)] for g, c in v.items()} for k, v in scopes.items()}


def distractors(row: dict, pools: dict) -> list[str]:
    gold = row["a"]
    gold_words = set(gold.split())
    seen, out = {gold}, []
    # the alternative a "choose" question names in its own text is the hardest negative there is
    if " or " in row["q"]:
        tail = row["q"].rstrip("?").split(" or ")[-1].strip().strip(",")
        if tail and tail != gold and len(tail) <= 128:
            out.append(tail)
            seen.add(tail)
    for key in ("local", "glob", "detailed", "sem"):
        for cand in pools[key].get(row[key], ()):
            if cand in seen or not 0 < len(cand) <= 128:
                continue
            if cand in gold or gold in cand or (gold_words & set(cand.split())):
                continue
            seen.add(cand)
            out.append(cand)
    return out


def make_record(row: dict, split: str, partition: str, rng: random.Random, image: dict) -> dict | None:
    family = FAMILY.get(row["sem"], "other")
    if row["a"] in ("yes", "no"):
        field = {"id": "answer", "type": "boolean", "question": row["q"]}
        target, cause = row["a"] == "yes", None
    else:
        pool = distractors(row, pools)
        if len(pool) < 2:
            return None
        cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
        want = rng.randint(13, 25) if rng.random() < 0.10 else rng.randint(2, 12)
        take = want if cause else want - 1
        values = pool[:max(1, min(take, len(pool)))]
        if not cause:
            values = values + [row["a"]]
        if len(values) < 2:
            return None
        rng.shuffle(values)
        field = {"id": "answer", "type": "choice", "question": row["q"],
                 "options": [{"value": v} for v in values]}
        target = None if cause else row["a"]
    request = {"request_id": f"gqa-{split}-{row['qid']}", "state": {}, "fields": [field]}
    Request.model_validate(request)
    return {"id": f"{SOURCE}:{split.split('_')[0]}-{row['qid']}", "source": SOURCE, "source_split": split,
            "source_group": str(row["img"]), "family": family, "license": LICENSE,
            "images": [image], "request": request, "target": target,
            "abstention_cause": cause, "source_answer": row["a"], "partition": partition}


def pick(rows_by_image: dict, order: list, quota: int, rng: random.Random, max_per_image: int,
         family_cap: dict | None = None) -> list[dict]:
    """Deterministic selection steering the choice/boolean mix and the yes/no balance."""
    counts = {"choice": 0, "boolean": 0}
    yn = {"yes": 0, "no": 0}
    fam_counts: Counter = Counter()
    taken_groups = defaultdict(set)
    taken_qids: set[str] = set()
    picked: list[dict] = []
    for round_index in range(max_per_image):
        for img in order:
            if len(picked) >= quota:
                return picked
            used = taken_groups[img]
            if len(used) > round_index:
                continue
            want_choice = counts["choice"] < CHOICE_TARGET * max(1, sum(counts.values()))
            want_yes = yn["yes"] <= yn["no"]
            best = None
            for r in rows_by_image[img]:
                if r["qid"] in taken_qids or (r["local"] and r["local"] in used):
                    continue
                if family_cap is not None and fam_counts[FAMILY.get(r["sem"], "other")] >= family_cap.get(
                        FAMILY.get(r["sem"], "other"), 0):
                    continue
                is_choice = r["a"] not in ("yes", "no")
                score = (is_choice == want_choice)
                if not is_choice:
                    score = score and ((r["a"] == "yes") == want_yes)
                if score:
                    best = r
                    break
                if best is None:
                    best = r
            if best is None:
                continue
            used.add(best["local"] or best["qid"])
            taken_qids.add(best["qid"])
            picked.append(best)
            if best["a"] in ("yes", "no"):
                counts["boolean"] += 1
                yn[best["a"]] += 1
            else:
                counts["choice"] += 1
            fam_counts[FAMILY.get(best["sem"], "other")] += 1
    return picked


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()
    rng = random.Random(SEED)

    print("loading instructions...", flush=True)
    train_rows = load("train_balanced_instructions")
    val_rows = load("val_balanced_instructions")
    pools = build_pools(train_rows + val_rows)
    cached = {p.stem for p in VG_PROC.glob("*.jpg")}
    print(f"train={len(train_rows)} val={len(val_rows)} cached_images={len(cached)}", flush=True)

    fit_rows = [r for r in train_rows if r["img"] in cached and int(r["img"]) not in excl["vg"]]
    test_rows = [r for r in val_rows if r["img"] in cached]
    assert not ({r["img"] for r in fit_rows} & {r["img"] for r in test_rows}), "train/val image overlap"

    def group(rows):
        by = defaultdict(list)
        for r in rows:
            by[r["img"]].append(r)
        for v in by.values():
            v.sort(key=lambda r: stable_rank(SEED, r["qid"]))
        return by

    fit_by, test_by = group(fit_rows), group(test_rows)
    fit_order = sorted(fit_by, key=lambda i: stable_rank(SEED, "img", i))
    test_order = sorted(test_by, key=lambda i: stable_rank(SEED, "timg", i))

    selected_fit = pick(fit_by, fit_order, TRAIN + DEV + 7_000, rng, MAX_PER_IMAGE)
    selected_test = pick(test_by, test_order, TEST_PER_FAMILY * len(FAMILY), rng, 1,
                         family_cap={f: TEST_PER_FAMILY for f in FAMILY.values()} | {"other": 0})
    print(f"selected fit={len(selected_fit)} test={len(selected_test)}", flush=True)

    per_image = Counter(r["img"] for r in selected_fit)
    dev_images, acc = set(), 0
    for img in sorted(per_image, key=lambda i: stable_rank(SEED, "dev", i)):
        if acc >= DEV * 1.6:
            break
        dev_images.add(img)
        acc += per_image[img]
    records, need_images = [], {}
    for rows, split, part_fn in ((selected_fit, "train_balanced", lambda r: "dev" if r["img"] in dev_images else "train"),
                                 (selected_test, "val_balanced", lambda r: "test")):
        for r in rows:
            need_images.setdefault(r["img"], None)
            records.append((r, split, part_fn(r)))

    print(f"storing {len(need_images)} images...", flush=True)
    for n, vg_id in enumerate(need_images):
        need_images[vg_id] = store_cached(IMAGES, VG_PROC / f"{vg_id}.jpg")
        if n % 10000 == 0:
            print(f"  {n}", flush=True)

    out, dev_count = [], 0
    for r, split, partition in records:
        image = need_images[r["img"]]
        if image is None:
            continue
        if partition == "dev":
            dev_count += 1
        rec = make_record(r, split, partition, rng, image)
        if rec is None:
            continue
        if partition != "test" and rec["images"][0]["sha256"] in excl["sha256"]:
            continue
        out.append(rec)

    # trim to quota deterministically, keeping every test record
    fit = [r for r in out if r["partition"] != "test"]
    test = [r for r in out if r["partition"] == "test"]
    fit.sort(key=lambda r: stable_rank(SEED, "trim", r["id"]))
    train_recs = [r for r in fit if r["partition"] == "train"][:TRAIN]
    dev_recs = [r for r in fit if r["partition"] == "dev"][:DEV]
    out = train_recs + dev_recs + test
    out.sort(key=lambda r: r["id"])

    keep = {Path(r["images"][0]["image"]).name for r in out}     # drop images the trim orphaned
    for f in IMAGES.iterdir():
        if f.name not in keep:
            f.unlink()
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    print(json.dumps({"records": len(out),
                      "partition": dict(Counter(r["partition"] for r in out)),
                      "family": dict(Counter(r["family"] for r in out)),
                      "field": dict(Counter(r["request"]["fields"][0]["type"] for r in out)),
                      "abstention": dict(Counter(str(r["abstention_cause"]) for r in out)),
                      "bool_targets": dict(Counter(str(r["target"]) for r in out
                                                   if r["request"]["fields"][0]["type"] == "boolean"))},
                     indent=2))
