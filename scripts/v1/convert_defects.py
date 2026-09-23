"""VisA + DAGM 2007 + BTAD -> decision-v1 `defects`: visual inspection under a clean licence.

  defect_presence      boolean "does this <object> show a defect?", balanced, inverted phrasings
  defect_type          choice over the object's own VisA anomaly taxonomy; option deletion
                       gives `not_listed`, and asking it of a VisA image the annotation proves
                       normal gives `false_premise`
  reference_comparison two images - images[0] a normal reference of the category, images[1] the
                       inspected item. Swapping in a reference from another category makes the
                       question's claim about the reference false: `mismatched_reference`

Splits are by CATEGORY, not by image, so every test category is unseen at training time.
MVTec is deliberately absent: CC BY-NC-SA forbids commercial use.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import store_image, write_jsonl  # noqa: E402

SOURCE = "defects"
SEED = 20260924
CACHE = Path(".cache/datasets/v1/defects")
OUT = Path("data/decision-v1/defects")
IMAGES = OUT / "images"
DEV_FRACTION = 0.03
NOT_LISTED_RATE = 0.15
MISMATCH_RATE = 0.22

VISA_TAR = CACHE / "VisA_20220922.tar"
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
DAGM_URL = "https://zenodo.org/records/12750201"
BTAD_URL = "https://avires.dimi.uniud.it/papers/btad/btad.zip"
DAGM_CLASSES = [1, 2, 3, 4, 5, 6]

VISA_OBJECTS = {
    "candle": "candle", "capsules": "sheet of capsules", "cashew": "cashew nut",
    "chewinggum": "stick of chewing gum", "fryum": "fryum snack",
    "macaroni1": "piece of macaroni", "macaroni2": "piece of macaroni",
    "pcb1": "printed circuit board", "pcb2": "printed circuit board",
    "pcb3": "printed circuit board", "pcb4": "printed circuit board",
    "pipe_fryum": "pipe-shaped fryum snack",
}
# held out so the test partition contains categories the model never trained on
TEST_CATEGORIES = {"visa:capsules", "visa:chewinggum", "visa:pipe_fryum", "dagm:5", "dagm:6", "btad:03"}

BOOLEAN_TEMPLATES = [
    ("Does this {obj} show a defect?", False),
    ("Is there anything wrong with this {obj}?", False),
    ("Would this {obj} fail a visual quality inspection?", False),
    ("Does this {obj} look damaged or faulty?", False),
    ("Is this {obj} free of visible defects?", True),
    ("Does this {obj} look acceptable, with nothing wrong with it?", True),
]
TYPE_TEMPLATES = [
    "What kind of defect does this {obj} show?",
    "Which defect is visible on this {obj}?",
    "This {obj} failed inspection. What is the fault?",
    "Identify the defect on this {obj}.",
    "What is wrong with this {obj}?",
]
PAIR_TEMPLATES = [
    ("The first image shows a normal {obj}. Does the second image show a {obj} with a defect?", False),
    ("Image 1 is a defect-free reference {obj}. Compared with it, is the {obj} in image 2 faulty?", False),
    ("The reference (first image) is an acceptable {obj}. Does the inspected item in the second image "
     "deviate from it?", False),
    ("Given the reference {obj} in the first image, does the {obj} in the second image have a fault?", False),
    ("First image: a reference {obj} known to be good. Second image: the item under inspection. "
     "Does the inspected {obj} match the reference?", True),
]


def hkey(*parts):
    return hashlib.sha256((":".join(str(p) for p in parts)).encode()).hexdigest()


# --------------------------------------------------------------------------- sources
def visa_items(excluded):
    """(category, key, normal?, defect types) from each object's image_anno.csv."""
    items = []
    with tarfile.open(VISA_TAR) as tar:
        for obj in VISA_OBJECTS:
            rows = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(f"{obj}/image_anno.csv"))))
            for r in rows:
                label = r["label"].strip()
                types = [] if label == "normal" else [t.strip() for t in label.split(",") if t.strip()]
                items.append(dict(dataset="visa", category=f"visa:{obj}", object=VISA_OBJECTS[obj],
                                  key=r["image"], normal=(label == "normal"), types=types,
                                  excluded=r["image"] in excluded, source_split="visa_all"))
    return items


def dagm_items():
    items = []
    for n in DAGM_CLASSES:
        path = CACHE / f"Class{n}.zip"
        if not path.exists():
            raise SystemExit(f"missing {path} - download it from {DAGM_URL}")
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            labelled = {}
            for label_file in [m for m in names if m.lower().endswith("labels.txt")]:
                split = "train" if "/train/" in label_file.lower() else "test"
                for line in z.read(label_file).decode("utf8", "replace").splitlines():
                    parts = [p.strip() for p in re.split(r"[\t;,]+|\s{2,}|\s", line.strip()) if p.strip()]
                    image = next((p for p in parts if p.lower().endswith(".png")
                                  and "label" not in p.lower()), None)
                    if not image:
                        continue
                    mask = next((p for p in parts if p.lower().endswith(".png") and "label" in p.lower()), None)
                    labelled[(split, image)] = bool(mask)
            for member in names:
                base = Path(member).name
                if not base.lower().endswith(".png") or "label" in member.lower():
                    continue
                split = "train" if "/train/" in member.lower() else "test"
                if (split, base) not in labelled:
                    continue
                items.append(dict(dataset="dagm", category=f"dagm:{n}", object="textured surface",
                                  key=f"Class{n}.zip::{member}", normal=not labelled[(split, base)],
                                  types=[], excluded=False, source_split=f"dagm_{split}"))
    return items


def btad_items():
    path = CACHE / "btad.zip"
    if not path.exists():
        raise SystemExit(f"missing {path} - download it from {BTAD_URL}")
    items = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            low = member.lower()
            if not low.endswith((".bmp", ".png", ".jpg")) or "ground_truth" in low:
                continue
            parts = [p for p in member.split("/") if p]
            product = next((p for p in parts if re.fullmatch(r"0?\d", p)), None)
            if product is None or len(parts) < 3:
                continue
            normal = parts[-2].lower() in ("ok", "good")
            if parts[-2].lower() not in ("ok", "good", "ko"):
                continue
            items.append(dict(dataset="btad", category=f"btad:{product}", object="manufactured part",
                              key=f"btad.zip::{member}", normal=normal, types=[], excluded=False,
                              source_split="btad_" + ("train" if "/train/" in low else "test")))
    return items


# --------------------------------------------------------------------------- bytes
class Readers:
    def __init__(self):
        self.tar = tarfile.open(VISA_TAR)
        self.zips = {}

    def read(self, item):
        if item["dataset"] == "visa":
            return self.tar.extractfile(item["key"]).read()
        archive, member = item["key"].split("::", 1)
        if archive not in self.zips:
            self.zips[archive] = zipfile.ZipFile(CACHE / archive)
        return self.zips[archive].read(member)

    def close(self):
        self.tar.close()
        for z in self.zips.values():
            z.close()


# --------------------------------------------------------------------------- records
def main():
    excluded = set()
    for p in Path("data/manifests").glob("*visa*.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = r.get("source_id") or r.get("archive_member")
            if key:
                excluded.add(key)

    items = visa_items(excluded) + dagm_items() + btad_items()
    by_category = defaultdict(list)
    for it in items:
        by_category[it["category"]].append(it)
    print({c: (len(v), sum(x["normal"] for x in v)) for c, v in sorted(by_category.items())})

    rng = random.Random(SEED)
    partition_of = lambda cat, key: ("test" if cat in TEST_CATEGORIES else
                                     "dev" if int(hkey(SEED, "p", key), 16) % 1000 < DEV_FRACTION * 1000
                                     else "train")

    # per-category quotas: balanced boolean pairs, sized by how many anomalies each category has
    rows = []
    references = {}
    for category, group in sorted(by_category.items()):
        test = category in TEST_CATEGORIES
        usable = [it for it in group if not (it["excluded"] and not test)]
        normals = sorted([it for it in usable if it["normal"]], key=lambda it: hkey(SEED, "n", it["key"]))
        anomalies = sorted([it for it in usable if not it["normal"]], key=lambda it: hkey(SEED, "a", it["key"]))
        if len(normals) < 12 or len(anomalies) < 8:
            continue
        references[category] = normals[:4]            # reference pool, never used as an inspected item
        normals = normals[4:]
        cap = 120 if test else (200 if group[0]["dataset"] == "visa" else 150)
        anomalies = anomalies[:cap]
        # one normal per anomaly keeps the boolean 50/50; the surplus only feeds false-premise items
        rows.append((category, normals[: len(anomalies)], normals[len(anomalies): len(anomalies) * 2],
                     anomalies))

    out = []
    pair_pool = defaultdict(list)

    def boolean(it, partition, tag):
        text, inverted = BOOLEAN_TEMPLATES[int(hkey(SEED, "bt", it["key"]), 16) % len(BOOLEAN_TEMPLATES)]
        field = {"id": "answer", "type": "boolean", "question": text.format(obj=it["object"])}
        return dict(id=f"{SOURCE}:{tag}", source=SOURCE, source_split=it["source_split"],
                    source_group=it["category"], family="defect_presence", license=LICENSE_OF[it["dataset"]],
                    images=[it], request={"request_id": f"{SOURCE}-{tag}"[:128], "state": {}, "fields": [field]},
                    target=(not it["normal"]) != inverted, abstention_cause=None,
                    source_answer="normal" if it["normal"] else ",".join(it["types"]) or "anomaly",
                    partition=partition)

    for category, normals, surplus, anomalies in rows:
        taxonomy = sorted({t for it in by_category[category] for t in it["types"]})
        for it in anomalies + normals:
            partition = partition_of(category, it["key"])
            out.append(boolean(it, partition, f"{it['key']}:presence"))
            pair_pool[category].append(it)
        if len(taxonomy) >= 3:
            singles = [it for it in anomalies if len(it["types"]) == 1]
            for it in singles:
                gold = it["types"][0]
                others = [t for t in taxonomy if t != gold]
                rng.shuffle(others)
                cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
                n = min(len(others) + (0 if cause else 1), rng.randint(2, max(3, len(taxonomy))))
                values = others[: n if cause else n - 1] + ([] if cause else [gold])
                if len(values) < 2:
                    continue
                rng.shuffle(values)
                text = TYPE_TEMPLATES[int(hkey(SEED, "tt", it["key"]), 16) % len(TYPE_TEMPLATES)]
                field = {"id": "answer", "type": "choice", "question": text.format(obj=it["object"]),
                         "options": [{"value": v} for v in values]}
                out.append(dict(id=f"{SOURCE}:{it['key']}:type", source=SOURCE,
                                source_split=it["source_split"], source_group=category, family="defect_type",
                                license=LICENSE_OF[it["dataset"]], images=[it],
                                request={"request_id": f"{SOURCE}-{it['key']}-type"[:128], "state": {},
                                         "fields": [field]},
                                target=None if cause else gold, abstention_cause=cause,
                                source_answer=gold, partition=partition_of(category, it["key"])))
            # the same question on an image the annotation proves normal: the premise is false
            for it in surplus[: max(1, len(singles) // 2)]:
                options = list(taxonomy)
                rng.shuffle(options)
                options = options[: rng.randint(2, max(3, min(8, len(taxonomy))))]
                if len(options) < 2:
                    continue
                text = TYPE_TEMPLATES[int(hkey(SEED, "ft", it["key"]), 16) % len(TYPE_TEMPLATES)]
                field = {"id": "answer", "type": "choice", "question": text.format(obj=it["object"]),
                         "options": [{"value": v} for v in options]}
                out.append(dict(id=f"{SOURCE}:{it['key']}:no_defect", source=SOURCE,
                                source_split=it["source_split"], source_group=category,
                                family="defect_type", license=LICENSE_OF[it["dataset"]], images=[it],
                                request={"request_id": f"{SOURCE}-{it['key']}-nodef"[:128], "state": {},
                                         "fields": [field]},
                                target=None, abstention_cause="false_premise",
                                source_answer="normal", partition=partition_of(category, it["key"])))

    # two-image reference/target records, references drawn from the same partition only
    by_partition = defaultdict(list)
    for category in references:
        by_partition["test" if category in TEST_CATEGORIES else "fit"].append(category)
    for category, pool in sorted(pair_pool.items()):
        side = "test" if category in TEST_CATEGORIES else "fit"
        siblings = [c for c in by_partition[side] if c != category]
        for it in pool:
            if rng.random() > 0.90:
                continue
            partition = partition_of(category, it["key"])
            mismatch = bool(siblings) and rng.random() < MISMATCH_RATE
            source_cat = rng.choice(siblings) if mismatch else category
            reference = references[source_cat][int(hkey(SEED, "r", it["key"]), 16) % len(references[source_cat])]
            text, inverted = PAIR_TEMPLATES[int(hkey(SEED, "pt", it["key"]), 16) % len(PAIR_TEMPLATES)]
            field = {"id": "answer", "type": "boolean", "question": text.format(obj=it["object"])}
            target = None if mismatch else ((not it["normal"]) != inverted)
            out.append(dict(id=f"{SOURCE}:{it['key']}:pair", source=SOURCE, source_split=it["source_split"],
                            source_group=category, family="reference_comparison",
                            license=LICENSE_OF[it["dataset"]], images=[reference, it],
                            request={"request_id": f"{SOURCE}-{it['key']}-pair"[:128], "state": {},
                                     "fields": [field]},
                            target=target,
                            abstention_cause="mismatched_reference" if mismatch else None,
                            source_answer=("reference is a " + reference["object"]) if mismatch else
                                          ("normal" if it["normal"] else "anomaly"),
                            partition=partition))

    # spec: about 300 test records per family, capped at 1,500 per source
    kept, per_family = [], Counter()
    for r in sorted(out, key=lambda r: hkey(SEED, "cap", r["id"])):
        if r["partition"] == "test":
            if per_family[r["family"]] >= 300:
                continue
            per_family[r["family"]] += 1
        kept.append(r)
    out = kept

    readers = Readers()
    stored = {}
    final = []
    for r in out:
        images = []
        for it in r["images"]:
            if it["key"] not in stored:
                stored[it["key"]] = store_image(readers.read(it), IMAGES)
            images.append(stored[it["key"]])
        r["images"] = images
        final.append(r)
    readers.close()
    final.sort(key=lambda r: r["id"])
    write_jsonl(OUT / "records.jsonl", final)

    summary = dict(records=len(final), images=len(stored),
                   partitions=dict(Counter(r["partition"] for r in final)),
                   families=dict(Counter(r["family"] for r in final)),
                   field_types=dict(Counter(r["request"]["fields"][0]["type"] for r in final)),
                   abstention=dict(Counter(str(r["abstention_cause"]) for r in final)),
                   licenses=dict(Counter(r["license"] for r in final)),
                   two_image=sum(len(r["images"]) == 2 for r in final),
                   boolean_targets=dict(Counter(str(r["target"]) for r in final
                                                if r["request"]["fields"][0]["type"] == "boolean")))
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


LICENSE_OF = {"visa": "CC-BY-4.0", "dagm": "CC-BY-4.0", "btad": "CC-BY-SA-4.0"}

if __name__ == "__main__":
    main()
