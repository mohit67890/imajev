"""heldout_countqa: CountQA -> ordinal counting records, every record partition "test".

CountQA is the only counting set in the survey whose *images* are cleanly licensed: the authors
photographed them themselves and released the set CC BY 4.0. It is test-only by design and hard
(best reported model 42.9%), so it is a clean held-out probe for the ordinal field.

Each question becomes a window of consecutive integers (2-7 levels). About 17% of records put the
true count outside the window, which is an honest `not_listed` abstention: the right answer is not
among the offered levels. Windows are sampled so the gold is not systematically central and so
"pick the smallest / middle level" is not a winning heuristic.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_heldout_countqa.py
"""
import glob
import hashlib
import random
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "scripts/v1")
from _common_textrich import (CACHE, ImageStore, count_levels, count_window,  # noqa: E402
                              make_record, ordinal_field, write)

SOURCE = "heldout_countqa"
SEED = 20260922
CAP = 1500
MAX_PER_IMAGE = 3
NOT_LISTED_RATE = 0.17
LICENSE = "CC-BY-4.0 (annotations and images; images were captured by the CountQA authors)"


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    items = []
    for path in sorted(glob.glob(str(CACHE / SOURCE / "data" / "*.parquet"))):
        table = pq.read_table(path)
        for i, row in enumerate(table.to_pylist()):
            key = hashlib.sha256(row["image"]["bytes"]).hexdigest()
            for q, a in zip(row["questions"], row["answers"]):
                items.append({"key": key, "row": row, "question": str(q).strip(),
                              "answer": str(a).strip(), "categories": row.get("categories") or [],
                              "focused": bool(row.get("is_focused"))})
    # deterministic order independent of shard order
    items.sort(key=lambda it: hashlib.sha256(f'{SEED}:{it["key"]}:{it["question"]}'.encode()).hexdigest())

    records, per_image, seen = [], {}, set()
    for it in items:
        if len(records) >= CAP:
            break
        if not it["answer"].isdigit():
            continue
        gold = int(it["answer"])
        if gold > 20:
            continue  # outside any 7-level window we can offer honestly
        if per_image.get(it["key"], 0) >= MAX_PER_IMAGE:
            continue
        uid = hashlib.sha256(f'{it["key"]}:{it["question"]}'.encode()).hexdigest()[:20]
        if uid in seen:
            continue
        abstain = rng.random() < NOT_LISTED_RATE
        window = count_window(rng, gold, lo=0, hi=20, outside=abstain)
        if window is None:
            window = count_window(rng, gold, lo=0, hi=20, outside=False)
            abstain = False
            if window is None:
                continue
        field, target, cause = ordinal_field(it["question"], count_levels(window), gold, abstain=abstain)
        img = store.add(it["row"]["image"]["bytes"], it["key"])
        seen.add(uid)
        per_image[it["key"]] = per_image.get(it["key"], 0) + 1
        family = "counting_focused" if it["focused"] else "counting_cluttered"
        records.append(make_record(
            source=SOURCE, uid=uid, source_split="test", source_group=img["sha256"], family=family,
            license=LICENSE, images=[img], field=field, target=target, cause=cause,
            source_answer=it["answer"], partition="test"))
    write(SOURCE, records)


if __name__ == "__main__":
    main()
