"""dude: DUDE single-page documents -> decision records, including its native unanswerable ones.

DUDE's "not-answerable" label is only true against the **whole** document, so for a single-image
model it is only honest on documents that *are* one page. This converter therefore keeps the 1,372
single-page documents only, which is what the research doc advises; every kept question is one
whose evidence — or whose proven absence of evidence — lives on the one page the model is shown.

The release ships rendered page images and three OCR versions, so nothing is re-rendered here. The
distractors are other strings from the same page, read from the Amazon Textract OCR (whole text
lines plus 1-5 word runs inside a line) and shape-matched to the gold, so a numeric answer meets
numeric decoys and a textual answer meets textual ones; the answers of other questions top up pages
that are too sparse.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_dude.py
"""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/v1")
from _common_textrich import (CACHE, ImageStore, choice_field, make_record,  # noqa: E402
                              option_budget, write)

SOURCE = "dude"
SEED = 20260922
DEV_FRACTION = 0.03
TEST_PER_FAMILY = 300
NOT_LISTED_RATE = 0.06  # DUDE's own not-answerable share tops the total abstention to ~17%
LICENSE = "CC-BY-4.0 (DUDE annotations and document collection, per the dataset card)"

RAW = CACHE / SOURCE
BIN = RAW / "DUDE_train-val-test_binaries"
FAMILY = {"extractive": "document_extractive", "abstractive": "document_abstractive",
          "list/extractive": "document_list", "list/abstractive": "document_list",
          "not-answerable": "document_unanswerable"}
NUMERIC = re.compile(r"\d")


def shape(text: str) -> str:
    return "numeric" if NUMERIC.search(text) else "text"


def page_index():
    """docId -> (split, number of page images). Page count comes from the shipped renders."""
    out = {}
    for split in ("train", "val", "test"):
        for f in (BIN / "images" / split).iterdir():
            if f.suffix.lower() != ".jpg":
                continue
            doc = f.stem.rsplit("_", 1)[0]
            split_, n = out.get(doc, (split, 0))
            out[doc] = (split_, n + 1)
    return out


def ocr_lines(doc_id: str):
    path = BIN / "OCR" / "Amazon" / f"{doc_id}_original.json"
    if not path.is_file():
        return []
    try:
        pages = json.loads(path.read_text())
    except Exception:
        return []
    lines = []
    for page in pages[:1]:  # single-page documents only
        for block in page.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                text = " ".join(str(block.get("Text", "")).split())
                if text:
                    lines.append(text)
    return lines


def like(rng, candidates, words: int):
    """Shuffle, then prefer candidates of a similar length to the gold answer.

    Without this the OCR runs are dominated by one- and two-word fragments ("in", "and the"),
    which are not plausible answers and make the gold obvious by its shape alone.
    """
    out = list(candidates)
    rng.shuffle(out)
    out.sort(key=lambda c: abs(len(c.split()) - words))
    return out


def spans(lines):
    """Candidate answer-shaped strings: whole OCR lines plus 1-5 word runs inside a line."""
    out = []
    for line in lines:
        if 1 <= len(line) <= 128:
            out.append(line)
        words = line.split()
        for n in (1, 2, 3, 4, 5):
            for i in range(len(words) - n + 1):
                s = " ".join(words[i:i + n])
                if 1 <= len(s) <= 128:
                    out.append(s)
    return out


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    gt = json.loads((RAW / "DUDE_gt.json").read_text())["data"]
    pages = page_index()
    single = {d for d, (_, n) in pages.items() if n == 1}

    by_doc = {}
    for q in gt:
        if q.get("data_split") not in ("train", "val") or q["docId"] not in single:
            continue  # test answers are withheld by the competition
        by_doc.setdefault(q["docId"], []).append(q)
    print(f"{len(pages)} documents, {len(single)} single-page, "
          f"{sum(len(v) for v in by_doc.values())} train/val questions on them")

    pool = Counter()
    for q in gt:
        if q.get("data_split") != "train":
            continue
        for a in (q.get("answers") or []):
            a = str(a).strip()
            if 0 < len(a) <= 128:
                pool[a] += 1
    answer_pool = [a for a, _ in pool.most_common(6000)]

    fit, test, taken_test = [], [], Counter()
    for doc_id in sorted(by_doc, key=lambda d: hashlib.sha256(f"{SEED}:{d}".encode()).hexdigest()):
        split = pages[doc_id][0]
        image = BIN / "images" / split / f"{doc_id}_0.jpg"
        if not image.is_file():
            continue
        candidates = spans(ocr_lines(doc_id))
        img = store.add(image.read_bytes(), doc_id)
        for q in sorted(by_doc[doc_id], key=lambda x: x["questionId"]):
            answers = [str(a).strip() for a in (q.get("answers") or []) if str(a).strip()]
            family = FAMILY.get(q.get("answer_type") or "")
            if family is None:
                continue
            question = str(q["question"]).strip()[:2000]
            if family == "document_unanswerable":
                if answers:
                    continue
                decoys = like(rng, [c for c in candidates if len(c) >= 2], 3)
                built = choice_field(rng, question, None, decoys + answer_pool,
                                     abstain=True, budget=option_budget(rng))
                if built is None:
                    continue
                field, target = built[0], built[1]
                cause, gold = "insufficient_evidence", "not-answerable"
            else:
                gold = next((a for a in answers if len(a) <= 128), None)
                if not gold:
                    continue
                want, words = shape(gold), len(gold.split())
                decoys = like(rng, [c for c in candidates if shape(c) == want], words)
                decoys += like(rng, [a for a in answer_pool if shape(a) == want], words)
                built = choice_field(rng, question, gold, decoys,
                                     abstain=rng.random() < NOT_LISTED_RATE, budget=option_budget(rng))
                if built is None:
                    continue
                field, target, cause = built
            rec = make_record(
                source=SOURCE, uid=q["questionId"], source_split=split, source_group=doc_id,
                family=family, license=LICENSE, images=[img], field=field, target=target,
                cause=cause, source_answer=gold, partition="train")
            if split == "val":
                if taken_test[family] >= TEST_PER_FAMILY:
                    continue
                taken_test[family] += 1
                rec["partition"] = "test"
                test.append(rec)
            else:
                fit.append(rec)

    records = []
    for rec in fit:
        h = int(hashlib.sha256(f'dev:{SEED}:{rec["source_group"]}'.encode()).hexdigest()[:8], 16)
        rec["partition"] = "dev" if h % 1000 < DEV_FRACTION * 1000 else "train"
        records.append(rec)
    records += test
    write(SOURCE, records)


if __name__ == "__main__":
    main()
