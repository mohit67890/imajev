"""textvqa: TextVQA v0.5.1 -> scene-text decision records.

Questions come from the official textvqa.org release; the option sets are built the hard way the
spec asks for: the distractors are *other strings that actually appear in the same photograph*,
taken from the official Rosetta OCR release (single tokens plus 2- and 3-token runs in reading
order, so multi-word answers meet multi-word decoys), topped up with gold answers of other
questions when one image does not carry enough text.

Only questions where at least 5 of the 10 annotators agree on the same answer are kept, so the
gold option is not one person's guess. Images are stored at the 1 MP cap, which leaves TextVQA's
~1024px OpenImages photos essentially untouched and the printed text legible.

Families: `scene_text_read` when the answer is literally printed in the image (it matches an OCR
run), `scene_text_reason` when it is not and the model has to combine what it reads with what it
sees. train/dev come from the train split, test from val.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_textvqa.py
"""
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter

sys.path.insert(0, "scripts/v1")
from _common_textrich import (CACHE, ImageStore, choice_field, excluded_sha256,  # noqa: E402
                              make_record, norm, option_budget, write)

SOURCE = "textvqa"
SEED = 20260922
TARGET_TRAIN = 20000
DEV_FRACTION = 0.03
TEST_PER_FAMILY = 300
NOT_LISTED_RATE = 0.17
MIN_AGREEMENT = 5
# TextVQA lets annotators answer "unanswerable"; when 5+ of them agree on it that is a
# human-labelled insufficient-evidence item, not an option value. The other convention answer is a
# comment on the question rather than an answer, so it is dropped.
UNANSWERABLE = "unanswerable"
NOT_AN_ANSWER = "answering does not require reading text in the image"
LICENSE = "CC-BY-4.0 (TextVQA annotations, per the authors' dataset card); images are OpenImages"

RAW = CACHE / SOURCE


def load(split):
    qs = json.loads((RAW / f"TextVQA_0.5.1_{split}.json").read_text())["data"]
    ocr = json.loads((RAW / f"TextVQA_Rosetta_OCR_v0.2_{split}.json").read_text())["data"]
    tokens = {r["image_id"]: [str(t) for t in (r.get("ocr_tokens") or []) if str(t).strip()] for r in ocr}
    return qs, tokens


def runs(tokens):
    """Single OCR tokens plus 2- and 3-token runs in reading order."""
    out = []
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            s = " ".join(tokens[i:i + n]).strip()
            if s and len(s) <= 128:
                out.append(s)
    return out


def gold_answer(answers):
    counts = Counter(norm(a) for a in answers if str(a).strip())
    if not counts:
        return None
    best, n = counts.most_common(1)[0]
    if n < MIN_AGREEMENT or not best:
        return None
    # return the most common surface form of that normalised answer
    forms = Counter(str(a).strip() for a in answers if norm(a) == best)
    text = forms.most_common(1)[0][0]
    return text if len(text) <= 128 else None


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    banned = excluded_sha256()
    zf = zipfile.ZipFile(RAW / "train_val_images.zip")
    members = set(zf.namelist())

    data = {s: load(s) for s in ("train", "val")}
    # answer pool for top-up distractors, from the training split only
    pool_counts = Counter()
    for q in data["train"][0]:
        g = gold_answer(q["answers"])
        if g:
            pool_counts[g] += 1
    answer_pool = [a for a, _ in pool_counts.most_common(4000)
                   if norm(a) not in (UNANSWERABLE, NOT_AN_ANSWER)]

    def build(split, quota_by_family, max_per_image=1):
        qs, tokens = data[split]
        qs = sorted(qs, key=lambda q: hashlib.sha256(f'{SEED}:{q["question_id"]}'.encode()).hexdigest())
        used_images, taken, out, seen = Counter(), Counter(), [], set()
        # One question per image first; extra passes only if the family quotas are still short.
        for limit in range(1, max_per_image + 1):
            if all(taken[f] >= n for f, n in quota_by_family.items()):
                break
            out += pass_over(qs, tokens, split, quota_by_family, used_images, taken, seen, limit)
        return out

    def pass_over(qs, tokens, split, quota_by_family, used_images, taken, seen, limit):
        out = []
        for q in qs:
            if all(taken[f] >= n for f, n in quota_by_family.items()):
                break
            image_id = q["image_id"]
            if q["question_id"] in seen or used_images[image_id] >= limit:
                continue
            gold = gold_answer(q["answers"])
            if not gold or norm(gold) == NOT_AN_ANSWER:
                continue
            unanswerable = norm(gold) == UNANSWERABLE
            ocr = runs(tokens.get(image_id, []))
            family = ("scene_text_unanswerable" if unanswerable else
                      "scene_text_read" if any(norm(r) == norm(gold) for r in ocr) else
                      "scene_text_reason")
            if taken[family] >= quota_by_family.get(family, 0):
                continue
            name = f"train_images/{image_id}.jpg"
            if name not in members:
                continue
            distractors = [r for r in ocr] + rng.sample(answer_pool, min(60, len(answer_pool)))
            abstain = unanswerable or rng.random() < NOT_LISTED_RATE
            built = choice_field(rng, str(q["question"]).strip(), None if unanswerable else gold,
                                 distractors, abstain=abstain, budget=option_budget(rng))
            if built is None:
                continue
            field, target, cause = built
            if unanswerable:
                cause = "insufficient_evidence"
            img = store.add(zf.read(name), image_id)
            if img["sha256"] in banned and split == "train":
                continue
            used_images[image_id] += 1
            seen.add(q["question_id"])
            taken[family] += 1
            out.append((family, make_record(
                source=SOURCE, uid=f'{split}-{q["question_id"]}', source_split=split,
                source_group=image_id, family=family, license=LICENSE, images=[img], field=field,
                target=target, cause=cause, source_answer=gold, partition="train")))
        return out

    fit = build("train", {"scene_text_read": int(TARGET_TRAIN * 0.63),
                          "scene_text_reason": int(TARGET_TRAIN * 0.35),
                          "scene_text_unanswerable": 10_000}, max_per_image=2)
    test = build("val", {"scene_text_read": TEST_PER_FAMILY, "scene_text_reason": TEST_PER_FAMILY,
                         "scene_text_unanswerable": TEST_PER_FAMILY})

    # dev is carved out by image, so a source_group never straddles train and dev either
    records = []
    for _, rec in fit:
        h = int(hashlib.sha256(f'dev:{SEED}:{rec["source_group"]}'.encode()).hexdigest()[:8], 16)
        rec["partition"] = "dev" if h % 1000 < DEV_FRACTION * 1000 else "train"
        records.append(rec)
    for _, rec in test:
        rec["partition"] = "test"
        records.append(rec)
    write(SOURCE, records)


if __name__ == "__main__":
    main()
