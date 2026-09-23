"""heldout_pairs: MuirBench two-image items -> decision-v1 records, every record partition "test".

MuirBench is the only multi-image MCQ benchmark in the survey with a stated commercial-compatible
licence (CC BY 4.0), which is why it is used here rather than NLVR2 (Cornell's terms restrict the
images to non-commercial research) or spot-the-diff (its images stay under the VIRAT agreement and
"may not be relicensed").

Only items whose `image_list` holds exactly two images and whose options are text (not `<image>`
placeholders) can become a decision record, so image-retrieval style tasks are dropped. MuirBench
pairs every item with an unanswerable twin via `counterpart_idx` whose gold is the literal option
"None of the choices provided"; those twins are honest `not_listed` abstentions and a fifth of the
records are drawn from them, matching the spec's 15-20% band.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_heldout_pairs.py
"""
import glob
import hashlib
import random
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, "scripts/v1")
from _common_textrich import CACHE, ImageStore, make_record, native_choice, write  # noqa: E402

SOURCE = "heldout_pairs"
SEED = 20260922
CAP = 1500
NONE_OPTION = "None of the choices provided"
ABSTAIN_SHARE = 0.20
LICENSE = "CC-BY-4.0 (MuirBench annotations); source images per the upstream datasets MuirBench draws on"


ORDER_NOTE = "The first image is the reference and the second image is the one to compare against it."


def rewrite(question: str) -> str:
    """Turn MuirBench's `<image>` placeholders into wording that names the two images in order.

    Most items park the placeholders at the end ("...in the images? <image> <image>"), where a
    literal substitution reads as gibberish; those are stripped and replaced by one sentence that
    states the order. Placeholders used inside a sentence are substituted in place.
    """
    head, _, tail = question.rpartition("?")
    if head and "<image>" not in head:
        return re.sub(r"\s+", " ", f"{head}? {ORDER_NOTE}").strip()[:2000]
    names = ["the first image", "the second image"]
    out, i = [], 0
    for part in re.split(r"(<image>)", question):
        if part == "<image>":
            out.append(names[i] if i < len(names) else "the image")
            i += 1
        else:
            out.append(part)
    q = re.sub(r"\s+", " ", "".join(out)).strip()
    if i == 0:
        q = f"{q} {ORDER_NOTE}"
    return q[:2000]


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    rows = []
    for path in sorted(glob.glob(str(CACHE / SOURCE / "data" / "*.parquet"))):
        rows += pq.read_table(path).to_pylist()
    by_idx = {str(r["idx"]): r for r in rows}

    def usable(r):
        return (len(r["image_list"]) == 2 and not any("<image>" in o for o in r["options"])
                and 2 <= len(r["options"]) <= 25)

    def gold_of(r):
        pos = ord(str(r["answer"]).strip().upper()[:1]) - 65
        return r["options"][pos] if 0 <= pos < len(r["options"]) else None

    answerable = [r for r in rows if usable(r) and gold_of(r) not in (None, NONE_OPTION)]
    answerable.sort(key=lambda r: hashlib.sha256(f'{SEED}:{r["idx"]}'.encode()).hexdigest())
    answerable = answerable[:CAP]

    # Pull in each selected item's unanswerable twin for a fifth of the set.
    want_abstain = round(len(answerable) * ABSTAIN_SHARE / (1 - ABSTAIN_SHARE))
    twins = []
    for r in answerable:
        if len(twins) >= want_abstain:
            break
        t = by_idx.get(str(r.get("counterpart_idx")))
        if t and usable(t) and gold_of(t) == NONE_OPTION:
            twins.append(t)

    records = []
    for r in answerable + twins:
        gold = gold_of(r)
        abstain = gold == NONE_OPTION
        options = [o for o in r["options"] if o != NONE_OPTION]
        built = native_choice(rng, rewrite(str(r["question"])), options, None if abstain else gold)
        if not built:
            continue
        field, target = built
        images = [store.add(im["bytes"], hashlib.sha256(im["bytes"]).hexdigest()) for im in r["image_list"]]
        if len({i["sha256"] for i in images}) != 2:
            continue  # the "pair" is one image twice
        group = "|".join(sorted(i["sha256"] for i in images))
        records.append(make_record(
            source=SOURCE, uid=f'muirbench-{r["idx"]}', source_split="test", source_group=group,
            family=f'pair_{str(r["task"]).lower().replace(" ", "_")}', license=LICENSE, images=images,
            field=field, target=target, cause="not_listed" if abstain else None,
            source_answer=gold, partition="test"))
    write(SOURCE, records)


if __name__ == "__main__":
    main()
