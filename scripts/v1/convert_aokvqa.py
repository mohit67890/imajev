"""decision-v1 converter: A-OKVQA -> knowledge/commonsense choice records.

A-OKVQA is natively 4-way multiple choice with `correct_choice_idx`, so the human-authored wrong
answers are used verbatim as the hard distractors. Extra distractors are drawn from the choice sets
of *other* questions that share the same question prefix, which varies the option count without
softening the item. Abstentions withhold the gold choice (`not_listed`).

Annotations: aokvqa_v1p0.tar.gz from the AllenAI bucket (Apache-2.0).
Images: the COCO JPEGs embedded in `HuggingFaceM4/A-OKVQA` parquet, joined on `question_id`, so no
second COCO download is needed.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_aokvqa.py
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

import pyarrow.parquet as pq  # noqa: E402

import _coco_shared as C  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402

SOURCE = "aokvqa"
SEED = 20260922
RAW = Path(".cache/datasets/v1/aokvqa")
ROOT = Path("data/decision-v1") / SOURCE
LICENSE = "Apache-2.0 (A-OKVQA annotations); images: COCO/Flickr terms, no image licence granted"
FAMILY = "knowledge_commonsense"

DEV_FRACTION = 0.03
NOT_LISTED_RATE = 0.17
FEWER_NATIVE_RATE = 0.15     # sometimes show only 1-2 of the 3 human distractors
BIG_OPTION_RATE = 0.10
PARQUET = ["data/train-00000-of-00002-c1d24de3bacb5e0c.parquet",
           "data/train-00001-of-00002-6b4f3abe2dc385d0.parquet",
           "data/validation-00000-of-00001-b2bd0de231b6326a.parquet"]


def norm(text: str) -> frozenset[str]:
    out = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t not in {"a", "an", "the", "of", "and", "or", "to", "in", "on", "is", "are", "it"}:
            out.add(t)
    return frozenset(out)


def same(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    return bool(na) and na == nb


def clashes(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return True
    return bool(na & nb) or a in b or b in a


def prefix_of(question: str) -> str:
    return " ".join(re.findall(r"[a-z']+", question.lower())[:2])


def load_images() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name in PARQUET:
        table = pq.read_table(RAW / name, columns=["question_id", "image"])
        for qid, im in zip(table.column("question_id").to_pylist(),
                           table.column("image").to_pylist()):
            out[qid] = im["bytes"]
    return out


def main() -> None:
    rng = random.Random(SEED)
    excluded = C.exclusions()
    ledger = C.load_ledger()
    train = json.loads((RAW / "aokvqa_v1p0_train.json").read_text())
    val = json.loads((RAW / "aokvqa_v1p0_val.json").read_text())

    # distractor pools: every choice string any other question offered, keyed by question prefix
    by_prefix: dict[str, list[str]] = defaultdict(list)
    global_pool: list[str] = []
    for r in train:
        for c in (x.strip() for x in r["choices"]):
            if 1 <= len(c) <= 128:
                by_prefix[prefix_of(r["question"])].append(c)
                global_pool.append(c)
    by_prefix = {k: [a for a, _ in Counter(v).most_common(400)] for k, v in by_prefix.items()}
    global_pool = [a for a, _ in Counter(global_pool).most_common(3000)]

    def build(r: dict, partition: str) -> dict | None:
        gold = r["choices"][r["correct_choice_idx"]].strip()
        if not 1 <= len(gold) <= 128:
            return None
        direct = r.get("direct_answers", [])
        natives = [c.strip() for i, c in enumerate(r["choices"])
                   if i != r["correct_choice_idx"] and 1 <= len(c.strip()) <= 128
                   and not any(same(c, d) for d in direct)]
        rng.shuffle(natives)
        keep = rng.randint(1, 2) if rng.random() < FEWER_NATIVE_RATE else 3
        natives = natives[:keep]
        hide = rng.random() < NOT_LISTED_RATE
        extra_want = rng.randint(9, 21) if rng.random() < BIG_OPTION_RATE else rng.randint(0, 8)

        chosen = list(natives)
        blocked = [gold] + list(direct) + list(r["choices"])
        pool = list(by_prefix.get(prefix_of(r["question"]), [])) + global_pool
        rng.shuffle(pool)
        for cand in pool:
            if len(chosen) - len(natives) >= extra_want:
                break
            if any(clashes(cand, b) for b in blocked):
                continue
            if any(clashes(cand, c) for c in chosen):
                continue
            chosen.append(cand)
        values = chosen + ([] if hide else [gold])
        values = list(dict.fromkeys(values))[:25]
        if len(values) < 2:
            return None
        rng.shuffle(values)
        field = dict(id="answer", type="choice", question=r["question"],
                     options=[dict(value=v) for v in values])
        request = dict(request_id=f"aokvqa-{r['split']}-{r['question_id']}", state={},
                       fields=[field])
        Request.model_validate(request)
        return dict(id=f"{SOURCE}:{r['question_id']}", source=SOURCE, source_split=r["split"],
                    source_group=str(r["image_id"]), family=FAMILY, license=LICENSE,
                    images=[], request=request,
                    target=None if hide else gold,
                    abstention_cause="not_listed" if hide else None,
                    source_answer=gold, partition=partition, _qid=r["question_id"],
                    _image_id=r["image_id"])

    records: list[dict] = []
    dropped = Counter()
    for r in sorted(val, key=lambda x: x["question_id"]):
        if not C.available(ledger, r["image_id"], "test"):
            dropped["val_ledger_conflict"] += 1
            continue
        rec = build(r, "test")
        if rec is None:
            dropped["val_unbuildable"] += 1
            continue
        C.claim(ledger, r["image_id"], "test")
        records.append(rec)
    for r in sorted(train, key=lambda x: x["question_id"]):
        if r["image_id"] in excluded:
            dropped["train_excluded_image"] += 1
            continue
        if not C.available(ledger, r["image_id"], "fit"):
            dropped["train_ledger_conflict"] += 1
            continue
        rec = build(r, "train")
        if rec is None:
            dropped["train_unbuildable"] += 1
            continue
        C.claim(ledger, r["image_id"], "fit")
        records.append(rec)

    fit_images = sorted({r["_image_id"] for r in records if r["partition"] != "test"},
                        key=lambda i: hashlib.sha256(f"{SOURCE}-dev:{i}".encode()).hexdigest())
    dev = set(fit_images[: max(1, round(len(fit_images) * DEV_FRACTION))])
    for r in records:
        if r["partition"] != "test" and r["_image_id"] in dev:
            r["partition"] = "dev"

    print(f"planned {len(records)} records, dropped {dict(dropped)}", flush=True)
    if "--plan" in sys.argv:
        print(json.dumps(dict(partitions=dict(Counter(r["partition"] for r in records)),
                              abstention=dict(Counter(str(r["abstention_cause"]) for r in records)),
                              options=dict(sorted(Counter(len(r["request"]["fields"][0]["options"])
                                                          for r in records).items()))), indent=2))
        return
    C.save_ledger(ledger)

    print("decoding parquet images ...", flush=True)
    blobs = load_images()
    banned = C.banned_sha()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "images").mkdir(exist_ok=True)
    kept = []
    for r in records:
        raw = blobs.get(r["_qid"])
        if raw is None:
            continue
        key = None
        for split in ("train2014", "val2014"):
            if C.cached(f"{split}/{r['_image_id']}") is not None:
                key = f"{split}/{r['_image_id']}"
                break
        meta = C.cached(key) if key else None
        if meta is None:
            key = f"coco2017-aokvqa/{r['_image_id']}"
            meta = C.put(key, raw)
        if meta["sha256"] in banned and r["partition"] != "test":
            continue
        r["images"] = [C.link_into(ROOT, key, meta)]
        r.pop("_qid"), r.pop("_image_id")
        kept.append(r)

    C.write_records(SOURCE, kept)
    print(json.dumps(dict(records=len(kept),
                          partitions=dict(Counter(r["partition"] for r in kept)),
                          abstention=dict(Counter(str(r["abstention_cause"]) for r in kept)),
                          images=len({r["images"][0]["sha256"] for r in kept})), indent=2))


if __name__ == "__main__":
    main()
