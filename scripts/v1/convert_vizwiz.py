"""VizWiz-VQA -> decision-v1 `vizwiz`: honest abstention from real, human-labelled unanswerables.

Three record families, all from photographs taken by blind photographers:
  answerability      boolean "can this question be answered from the image", balanced true/false
  vqa                the crowd's question, answered (choice, or boolean for yes/no questions)
  unanswerable_vqa   the same question where VizWiz marks it unanswerable -> target null,
                     abstention_cause=insufficient_evidence, as a choice AND (for yes/no
                     questions) as a boolean. This is the signal decision-v0 never had.

train/dev come from the TRAIN split only; test from VAL. Images are pulled out of the 11.3 GB
train.zip by HTTP byte range, one coalesced run per group of neighbouring members.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RemoteZip, compatible, option_count, store_image, write_jsonl  # noqa: E402

SOURCE = "vizwiz"
SEED = 20260922
LICENSE = "CC-BY-4.0"
ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
IMAGE_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/{split}.zip"
CACHE = Path(".cache/datasets/v1/vizwiz")
OUT = Path("data/decision-v1/vizwiz")
IMAGES = OUT / "images"

N_UNANSWERABLE_TRAIN = 5532          # every unanswerable train item
N_ANSWERABLE_TRAIN = 5532            # matched 1:1 so the answerability boolean is balanced
N_TEST_PER_FAMILY = 300
DEV_FRACTION = 0.03
NOT_LISTED_RATE = 0.15
AGREEMENT = 5                        # >=5 of 10 crowd answers agree

UNKNOWN_ANSWERS = {"unanswerable", "unsuitable", "unsuitable image", "unsuitable  image",
                   "unanswerable image", "none", "no idea", "unknown", "can't tell", "cannot tell", ""}
YES_NO = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|am|"
                    r"isn't|aren't|doesn't|don't|it|this|there|that)\b", re.I)

# >= 5 phrasings so the model cannot key on one string (spec rule 3)
ANSWERABILITY_TEMPLATES = [
    "Can this question be answered from the image: '{q}'?",
    "Does this photo show enough to answer the question '{q}'?",
    "Is the question '{q}' answerable from this picture?",
    "Looking only at this image, could you answer '{q}'?",
    "Is there enough visible here to answer: '{q}'?",
    "Could someone answer the question '{q}' using this photo alone?",
]


def hkey(*parts):
    return hashlib.sha256((":".join(str(p) for p in parts)).encode()).hexdigest()


def question_prefix(q, n=2):
    words = re.sub(r"[^a-z' ]", " ", q.lower()).split()
    return " ".join(words[:n])


def modal_answer(row):
    counts = Counter(a["answer"].strip().lower() for a in row["answers"])
    value, n = counts.most_common(1)[0]
    return value, n


def load_split(split):
    with zipfile.ZipFile(CACHE / "Annotations.zip") as z:
        return json.loads(z.read(f"{split}.json"))


def clean_answerable(row):
    """Answerable, the crowd agrees, and the agreed answer is a real answer (not a hedge)."""
    if row["answerable"] != 1:
        return None
    value, n = modal_answer(row)
    if n < AGREEMENT or value in UNKNOWN_ANSWERS or not 1 <= len(value) <= 128:
        return None
    return value


STOP = set("a an the is are was were this that these those what whats which who whose how much many of in "
           "on at it to you your my i me do does did can could tell please picture image photo thing there "
           "here and or for with be am has have had says say name kind type s t".split())


def content_words(question):
    return [w for w in re.sub(r"[^a-z0-9' ]", " ", question.lower()).split()
            if w not in STOP and len(w) > 2]


def build_pools(train_rows):
    """Distractors come from VizWiz's own answers to questions of the same shape or subject."""
    by_prefix2, by_prefix1, by_type, by_word = (defaultdict(Counter), defaultdict(Counter),
                                                defaultdict(Counter), defaultdict(Counter))
    for row in train_rows:
        value = clean_answerable(row)
        if value is None:
            continue
        by_prefix2[question_prefix(row["question"], 2)][value] += 1
        by_prefix1[question_prefix(row["question"], 1)][value] += 1
        by_type[row["answer_type"]][value] += 1
        for word in set(content_words(row["question"])):
            by_word[word][value] += 1
    rank = lambda c: [v for v, _ in c.most_common(80)]
    return ({k: rank(v) for k, v in by_prefix2.items()},
            {k: rank(v) for k, v in by_prefix1.items()},
            {k: rank(v) for k, v in by_type.items()},
            {k: v for k, v in by_word.items() if 3 <= sum(v.values()) <= 3000})


def pool_for(question, answer_type, pools, minimum=8):
    """Ordered candidate answers: subject-matched first, then question-shape, then answer type."""
    p2, p1, ptype, pword = pools
    scored = Counter()
    for word in set(content_words(question)):
        counter = pword.get(word)
        if not counter:
            continue
        for value, n in counter.most_common(25):
            scored[value] += 1 + n / (1 + sum(counter.values()))
    pool = [v for v, _ in scored.most_common(40)]
    for candidate in (p2.get(question_prefix(question, 2), []), p1.get(question_prefix(question, 1), [])):
        if len(candidate) >= minimum:
            pool += candidate
            break
    pool += ptype.get(answer_type) or ptype["other"]
    if not YES_NO.match(question.strip()):   # "yes"/"no" are not plausible answers to a wh-question
        pool = [v for v in pool if v not in ("yes", "no")]
    return list(dict.fromkeys(pool))


def choice_field(rng, question, gold, pool, cause):
    """gold=None means nothing in the option set is right (abstention by insufficient evidence)."""
    wanted = option_count(rng)
    candidates = [x for x in pool if gold is None or compatible(gold, x)]
    head = candidates[:max(25, wanted * 3)]   # keep distractors near the top of the relevance order
    rng.shuffle(head)
    values = head[: wanted if gold is None or cause == "not_listed" else wanted - 1]
    if gold is not None and cause is None:
        values.append(gold)
    values = list(dict.fromkeys(values))
    if len(values) < 2:
        return None
    rng.shuffle(values)
    return {"id": "answer", "type": "choice", "question": question,
            "options": [{"value": v} for v in values]}


def record(rid, split, group, family, field, target, cause, answer, partition):
    return {"id": rid, "source": SOURCE, "source_split": split, "source_group": group,
            "family": family, "license": LICENSE, "images": None,
            "request": {"request_id": rid[:128], "state": {}, "fields": [field]},
            "target": target, "abstention_cause": cause, "source_answer": answer,
            "partition": partition, "_image": group}


def select(rows, pools, rng_seed, n_answerability_each, n_content_answerable,
           n_content_unanswerable, split, partition_of):
    """Deterministically choose items and emit the record skeletons (images attached later)."""
    rng = random.Random(rng_seed)
    unanswerable = sorted((r for r in rows if r["answerable"] == 0),
                          key=lambda r: hkey(SEED, "u", r["image"]))
    answerable = sorted((r for r in rows if clean_answerable(r) is not None),
                        key=lambda r: hkey(SEED, "a", r["image"]))
    unanswerable = unanswerable[:n_content_unanswerable]
    answerable = answerable[:n_content_answerable]

    # Balance the yes/no content booleans before they are emitted (spec rule 5).
    yn = [r for r in answerable if r["answer_type"] == "yes/no" and clean_answerable(r) in ("yes", "no")]
    by_value = defaultdict(list)
    for r in yn:
        by_value[clean_answerable(r)].append(r)
    keep = min(len(by_value["yes"]), len(by_value["no"]))
    boolean_ok = {r["image"] for v in ("yes", "no") for r in by_value[v][:keep]}

    out = []
    for kind, pool_rows in (("a", answerable[:n_answerability_each]), ("u", unanswerable[:n_answerability_each])):
        for r in pool_rows:
            template = ANSWERABILITY_TEMPLATES[int(hkey(SEED, "t", r["image"]), 16) % len(ANSWERABILITY_TEMPLATES)]
            field = {"id": "answer", "type": "boolean",
                     "question": template.format(q=r["question"].strip())}
            out.append(record(f"{SOURCE}:{Path(r['image']).stem}:answerability", split, r["image"],
                              "answerability", field, kind == "a", None,
                              "answerable" if kind == "a" else "unanswerable", partition_of(r["image"])))

    for r in answerable:
        gold = clean_answerable(r)
        q = r["question"].strip()
        if r["image"] in boolean_ok:
            field = {"id": "answer", "type": "boolean", "question": q}
            out.append(record(f"{SOURCE}:{Path(r['image']).stem}:vqa", split, r["image"], "vqa",
                              field, gold == "yes", None, gold, partition_of(r["image"])))
            continue
        cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
        field = choice_field(rng, q, gold, pool_for(q, r["answer_type"], pools), cause)
        if field is None:
            continue
        out.append(record(f"{SOURCE}:{Path(r['image']).stem}:vqa", split, r["image"], "vqa",
                          field, None if cause else gold, cause, gold, partition_of(r["image"])))

    for r in unanswerable:
        q = r["question"].strip()
        yes_no = bool(YES_NO.match(q))
        pool = ["yes", "no"] if yes_no else pool_for(q, "other", pools)
        field = choice_field(rng, q, None, pool, "insufficient_evidence")
        if field is not None:
            out.append(record(f"{SOURCE}:{Path(r['image']).stem}:unanswerable_choice", split, r["image"],
                              "unanswerable_vqa", field, None, "insufficient_evidence",
                              "unanswerable", partition_of(r["image"])))
        if yes_no:
            field = {"id": "answer", "type": "boolean", "question": q}
            out.append(record(f"{SOURCE}:{Path(r['image']).stem}:unanswerable_boolean", split, r["image"],
                              "unanswerable_vqa", field, None, "insufficient_evidence",
                              "unanswerable", partition_of(r["image"])))
    return out


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "Annotations.zip").exists():
        raise SystemExit(f"download {ANNOTATIONS_URL} to {CACHE/'Annotations.zip'} first")
    train_rows, val_rows = load_split("train"), load_split("val")
    pools = build_pools(train_rows)

    excluded = set(json.loads(Path("data/decision-v1/exclusions.json").read_text())["vizwiz_images"])
    val_rows = [r for r in val_rows if r["image"] not in excluded]

    dev_cut = lambda image: "dev" if int(hkey(SEED, "p", image), 16) % 1000 < DEV_FRACTION * 1000 else "train"
    rows = select(train_rows, pools, SEED, N_ANSWERABLE_TRAIN, N_ANSWERABLE_TRAIN,
                  N_UNANSWERABLE_TRAIN, "train", dev_cut)
    rows += select(val_rows, pools, SEED + 1, N_TEST_PER_FAMILY // 2, N_TEST_PER_FAMILY,
                   N_TEST_PER_FAMILY, "val", lambda image: "test")

    needed = defaultdict(set)
    for r in rows:
        needed["train" if r["source_split"] == "train" else "val"].add(r["_image"])
    for split, names in sorted(needed.items()):
        dest = CACHE / f"{split}-images"
        missing = [n for n in sorted(names) if not (dest / n).exists()]
        if not missing:
            print(f"{split}: {len(names)} images already cached", flush=True)
            continue
        url = IMAGE_URL.format(split=split)
        with RemoteZip(url, CACHE / split) as z:
            members = [f"{split}/{n}" for n in missing]
            unknown = [m for m in members if m not in z.infos]
            if unknown:
                raise SystemExit(f"members missing from {url}: {unknown[:3]}")
            if split == "train":
                # 54% of train.zip is wanted here and more by the vizwiz_quality converter, while
                # the origin charges ~12 s of latency per request: coalesced runs over the whole
                # archive cost the same wall clock as per-member ranges and are reused downstream.
                members = sorted(n for n in z.infos if n.lower().endswith(".jpg"))
            print(f"{split}: fetching {len(members)} of {len(z.infos)} members", flush=True)
            z.fetch(members, dest, gap=(8 << 20) if split == "train" else (1 << 20),
                    progress=lambda d, t, b: print(f"  {split} {d}/{t} members, {b/1e9:.2f} GB", flush=True))
            (CACHE / f"{split}-receipt.json").write_text(json.dumps(
                dict(url=url, etag=z.etag, archive_bytes=z.size, transferred_bytes=z.transferred,
                     members=len(members)), indent=2) + "\n")

    cache_by_sha, final = {}, []
    for r in rows:
        name = r.pop("_image")
        split = "train" if r["source_split"] == "train" else "val"
        if name not in cache_by_sha:
            cache_by_sha[name] = store_image((CACHE / f"{split}-images" / name).read_bytes(), IMAGES)
        r["images"] = [cache_by_sha[name]]
        final.append(r)

    final.sort(key=lambda r: r["id"])
    write_jsonl(OUT / "records.jsonl", final)
    summary = dict(records=len(final), images=len(cache_by_sha),
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
