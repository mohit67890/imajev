"""heldout_fashionpedia: Fashionpedia attribute questions -> records, every record partition "test".

Fashionpedia is deliberately held out of training so its 294-attribute expert ontology stays an
*unseen option vocabulary*: the distractors here are sibling attributes from the same ontology
group (all 25 neckline types, all 18 textile patterns, ...), so the option sets share no vocabulary
with the product attributes trained elsewhere in the mixture.

Two things this converter takes seriously:

* **Per-image licence filtering.** Fashionpedia's own statement is that it does not own the
  photographs, and the COCO-style JSON carries a `license` id per image. Only images under
  CC BY 2.0, CC BY-SA 2.0, public domain, Unsplash, Pexels, freestocks or Burst are kept; the
  NonCommercial and NoDerivatives images (the majority of the val split) are dropped.
* **Unambiguous reference.** A question naming a garment type is only asked when the image holds
  exactly one instance of that type, and only when the instance carries exactly one attribute from
  the group being asked about, so there is exactly one right answer.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_heldout_fashionpedia.py
"""
import ast
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, "scripts/v1")
from _common_textrich import (CACHE, ImageStore, choice_field, make_record,  # noqa: E402
                              option_budget, write)

SOURCE = "heldout_fashionpedia"
SEED = 20260922
CAP = 1500
MAX_PER_IMAGE = 3
NOT_LISTED_RATE = 0.17
MIN_AREA_FRACTION = 0.01
# Fashionpedia licence ids that permit commercial use (see `licenses` in the annotation JSON).
COMMERCIAL_LICENSES = {0, 1, 6, 7, 8, 9, 10}
LICENSE = ("CC-BY-4.0 (Fashionpedia annotations and ontology); images restricted to the per-image "
           "CC-BY / CC-BY-SA / public-domain / Unsplash / Pexels / freestocks / Burst licences")

RAW = CACHE / SOURCE
GROUP_PHRASE = {
    "nickname": "specific style of {cat}",
    "silhouette": "silhouette of the {cat}",
    "neckline type": "neckline of the {cat}",
    "textile finishing, manufacturing techniques": "textile finish or manufacturing technique used on the {cat}",
    "textile pattern": "pattern on the {cat}'s fabric",
    "length": "length of the {cat}",
    "opening type": "opening of the {cat}",
    "non-textile material type": "non-textile material the {cat} is made of",
    "waistline": "waistline of the {cat}",
    "animal": "animal the {cat}'s material comes from",
    "leather": "leather type used for the {cat}",
}
TEMPLATES = [
    "What is the {phrase}?",
    "Which option describes the {phrase}?",
    "From this photo, identify the {phrase}.",
    "The image shows a {cat}. What is the {phrase}?",
    "Judging from this image, what is the {phrase}?",
    "Pick the {phrase} shown here.",
]


def short(name: str) -> str:
    return name.split(",")[0].strip()


def suffix(name: str) -> str:
    m = re.search(r"\(([^)]*)\)\s*$", name)
    return m.group(1).strip() if m else ""


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    ann = json.loads((RAW / "instances_attributes_val2020.json").read_text())
    zf = zipfile.ZipFile(RAW / "val_test2020.zip")
    members = set(zf.namelist())

    cats = {int(c["id"]): c for c in ann["categories"]}
    attrs = {int(a["id"]): a for a in ann["attributes"]}
    by_group = defaultdict(list)
    for a in attrs.values():
        by_group[a["supercategory"]].append(a)

    images = {int(im["id"]): im for im in ann["images"] if im.get("license") in COMMERCIAL_LICENSES}
    per_image_cats = defaultdict(Counter)
    rows = []
    for a in ann["annotations"]:
        image_id = int(a["image_id"])
        if image_id not in images:
            continue
        ids = ast.literal_eval(str(a.get("attribute_ids") or "[]"))
        if not ids:
            continue
        rows.append({"image_id": image_id, "category_id": int(a["category_id"]),
                     "attribute_ids": [int(x) for x in ids],
                     "bbox": ast.literal_eval(str(a["bbox"])), "ann_id": str(a["id"])})
        per_image_cats[image_id][int(a["category_id"])] += 1

    candidates = []
    for r in rows:
        im = images[r["image_id"]]
        if per_image_cats[r["image_id"]][r["category_id"]] != 1:
            continue  # the garment type would not identify one garment in this photo
        w, h = r["bbox"][2], r["bbox"][3]
        if w * h < MIN_AREA_FRACTION * im["width"] * im["height"]:
            continue
        present = [attrs[i] for i in r["attribute_ids"] if i in attrs]
        groups = Counter(a["supercategory"] for a in present)
        for a in present:
            if groups[a["supercategory"]] != 1:
                continue  # more than one attribute from this group applies: no single right answer
            candidates.append((r, im, a, {x["name"] for x in present}))

    candidates.sort(key=lambda c: hashlib.sha256(
        f'{SEED}:{c[0]["ann_id"]}:{c[2]["id"]}'.encode()).hexdigest())

    records, per_image = [], Counter()
    for r, im, attr, present_names in candidates:
        if len(records) >= CAP:
            break
        if per_image[r["image_id"]] >= MAX_PER_IMAGE:
            continue
        name = f'test/{im["file_name"]}'
        if name not in members:
            continue
        group = attr["supercategory"]
        gold = attr["name"]
        siblings = [s["name"] for s in by_group[group] if s["name"] not in present_names]
        if group == "nickname" and suffix(gold):
            same = [s for s in siblings if suffix(s) == suffix(gold)]
            if len(same) >= 3:
                siblings = same
        rng.shuffle(siblings)
        if len(siblings) < 2:
            continue
        cat = short(cats[r["category_id"]]["name"])
        phrase = GROUP_PHRASE[group].format(cat=cat)
        question = rng.choice(TEMPLATES).format(phrase=phrase, cat=cat)
        abstain = rng.random() < NOT_LISTED_RATE
        built = choice_field(rng, question, gold, siblings, abstain=abstain,
                             budget=min(option_budget(rng), len(siblings) + 1))
        if built is None:
            continue
        field, target, cause = built
        img = store.add(zf.read(name), im["file_name"])
        per_image[r["image_id"]] += 1
        records.append(make_record(
            source=SOURCE, uid=f'{r["ann_id"]}-{attr["id"]}', source_split="val",
            source_group=img["sha256"], family=f'attribute_{group.split(",")[0].replace(" ", "_")}',
            license=LICENSE, images=[img], field=field, target=target, cause=cause,
            source_answer=gold, partition="test"))
    write(SOURCE, records)


if __name__ == "__main__":
    main()
