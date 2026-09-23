"""Open Images V7 TEST split -> `data/decision-v2/openimages_v2/images/`.

Why the test split only: decision-v1 already used Open Images photographs through TextVQA, whose
image ids come from the Open Images *train* and *validation* splits (see
`data/decision-v1/textvqa/README.md`), and the 2026-09 evaluation pilot
(`scripts/prepare_pilot_openimages.py`) pulled from *validation*.  Every id this script can emit is
checked against both sets, so no photograph the model has already seen can enter decision-v2.

Phases (both resumable):

  select    read `test-images-with-rotation.csv` (per-photo licence + author) and
            `oidv7-test-annotations-human-imagelabels.csv` (human-verified image-level labels),
            drop anything previously seen or needing rotation, and pick ~N images by *label
            diversity*: images are bucketed by their rarest verified-present label and taken
            round-robin from the rarest bucket outwards, so the tail of the ontology is as well
            represented as `Person` is.
  download  fetch each photo from the CVDF public mirror (`open-images-dataset.s3.amazonaws.com`),
            re-encode through the v1 image store (RGB JPEG q90, <= 1 MP), and write
            `manifest.jsonl` (url + sha256) and `attribution.jsonl` (Flickr author, licence, page).

The labels are used for TWO things only: choosing a diverse subset, and supplying present/absent
object candidates for the question templates.  They are never written as a target -- every answer
in decision-v2 comes from the 9B teacher.  Confidence==0 rows are human-verified *absences*, which
is what makes an honest "is there a <thing> in this photo?" distractor possible.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, append_jsonl, get_with_backoff, read_jsonl, repo_relative, session, store_image

SOURCE = "openimages_v2"
RAW_DIR = RAW / SOURCE
IMG_DIR = OUT / SOURCE / "images"
S3 = "https://open-images-dataset.s3.amazonaws.com/test/{}.jpg"
EXPECTED_LICENSE = "https://creativecommons.org/licenses/by/2.0/"

# Ontology entries that describe the picture rather than a thing in it, or that are too abstract to
# ask about; excluded from the diversity buckets and from the candidate pools.
ABSTRACT = {
    "Image", "Photograph", "Photography", "Font", "Text", "Screenshot", "Line", "Pattern",
    "Circle", "Rectangle", "Square", "Triangle", "Colorfulness", "Azure", "Aqua", "Magenta",
    "Art", "Illustration", "Drawing", "Sketch", "Painting", "Graphics", "Design", "Product",
    "Brand", "Logo", "Advertising", "Poster", "Banner", "Screen", "Display device", "Space",
    "Atmosphere", "Landscape", "Nature", "Natural landscape", "Natural environment", "Ecoregion",
    "Sky", "Horizon", "Light", "Shadow", "Darkness", "Night", "Daytime", "Morning", "Evening",
    "Reflection", "Symmetry", "Line art", "Monochrome", "Monochrome photography",
    "Black-and-white", "Stock photography", "Portrait", "Portrait photography", "Snapshot",
    "Selfie", "Close-up", "Macro photography", "Still life photography", "Event", "Fun",
    "Leisure", "Recreation", "Tourism", "Vacation", "Holiday", "Fashion", "Beauty", "Smile",
    "Happiness", "Sitting", "Standing", "Walking", "Jumping", "Games", "Sports", "Team",
    "Community", "Crowd", "Public space", "Urban area", "Human settlement", "City", "Town",
    "Neighbourhood", "Metropolitan area", "Downtown", "Road surface", "Asphalt", "Material",
    "Material property", "Wood", "Metal", "Plastic", "Glass", "Paper", "Textile", "Composite material",
}


# Body parts are boxable, extremely common in Open Images, and make poor question subjects
# ("what colour is a human body?").  People themselves (Person, Man, Woman, Boy, Girl) stay.
BODY_PARTS = {
    "Human body", "Human head", "Human face", "Human hair", "Human eye", "Human ear", "Human nose",
    "Human mouth", "Human arm", "Human hand", "Human leg", "Human foot", "Human beard",
    "Skull", "Human", "Person",
}


def load_previously_seen() -> set[str]:
    """Open Images ids that decision-v1 or an earlier evaluation already used."""
    seen: set[str] = set()
    for split in ("train", "val"):
        path = Path(".cache/datasets/v1/textvqa") / f"TextVQA_0.5.1_{split}.json"
        if path.is_file():
            seen |= {q["image_id"] for q in json.loads(path.read_text())["data"]}
    pilot = Path("data/manifests/pilot-openimages.jsonl")
    if pilot.is_file():
        for line in pilot.open():
            if line.strip():
                row = json.loads(line)
                for key in ("image_id", "source_id", "source_group", "id"):
                    value = row.get(key)
                    if isinstance(value, str):
                        seen.add(value.split(":")[-1])
    return seen


def run_select(target: int, per_label_cap: int, seed: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = {}
    with (RAW_DIR / "oidv7-class-descriptions.csv").open() as f:
        for row in csv.DictReader(f):
            names[row["LabelName"]] = row["DisplayName"]

    meta: dict[str, dict] = {}
    licenses = Counter()
    skipped = Counter()
    with (RAW_DIR / "test-images-with-rotation.csv").open() as f:
        for row in csv.DictReader(f):
            licenses[row["License"]] += 1
            if row["License"] != EXPECTED_LICENSE:
                skipped["license_not_cc_by_2_0"] += 1
                continue
            if row["Rotation"] not in ("", "0.0", "0"):
                skipped["needs_rotation"] += 1
                continue
            meta[row["ImageID"]] = row
    print(f"licence values in the test manifest: {dict(licenses)}", flush=True)

    seen_before = load_previously_seen()
    overlap = set(meta) & seen_before
    for image_id in overlap:
        del meta[image_id]
    skipped["previously_seen"] = len(overlap)
    print(f"{len(meta)} candidate test photos after filters {dict(skipped)}", flush=True)

    present: dict[str, set[str]] = defaultdict(set)
    absent: dict[str, set[str]] = defaultdict(set)
    with (RAW_DIR / "oidv7-test-annotations-human-imagelabels.csv").open() as f:
        for row in csv.DictReader(f):
            image_id = row["ImageID"]
            if image_id not in meta:
                continue
            name = names.get(row["LabelName"])
            if not name or name in ABSTRACT:
                continue
            (present if row["Confidence"] == "1" else absent)[image_id].add(name)

    usable = [i for i in meta if present.get(i)]
    freq = Counter(name for i in usable for name in present[i])
    print(f"{len(usable)} photos carry at least one concrete verified label; {len(freq)} distinct labels", flush=True)

    # Bucket each image by its rarest verified-present label, then take round-robin from the rarest
    # bucket outwards.  This is the whole diversity mechanism, and it is deterministic given `seed`.
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for image_id in usable:
        rarest = min(present[image_id], key=lambda n: (freq[n], n))
        buckets[rarest].append(image_id)
    for value in buckets.values():
        rng.shuffle(value)
    order = sorted(buckets, key=lambda n: (freq[n], n))

    chosen: list[str] = []
    taken = Counter()
    round_index = 0
    while len(chosen) < target:
        progressed = False
        for name in order:
            if len(chosen) >= target:
                break
            if taken[name] > round_index or taken[name] >= per_label_cap:
                continue
            pool = buckets[name]
            if len(pool) <= taken[name]:
                continue
            chosen.append(pool[taken[name]])
            taken[name] += 1
            progressed = True
        if not progressed:
            break
        round_index += 1

    rows = []
    for image_id in chosen:
        row = meta[image_id]
        rows.append({
            "image_id": image_id,
            "url": S3.format(image_id),
            "flickr_landing_url": row["OriginalLandingURL"],
            "author": row["Author"] or "(not stated)",
            "author_profile": row["AuthorProfileURL"],
            "title": row["Title"],
            "license_url": row["License"],
            "spdx": "CC-BY-2.0",
            "original_md5": row["OriginalMD5"],
            "labels_present": sorted(present[image_id]),
            "labels_absent": sorted(absent.get(image_id, ())),
            "bucket_label": min(present[image_id], key=lambda n: (freq[n], n)),
        })
    (RAW_DIR / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {
        "target": target, "selected": len(rows), "distinct_bucket_labels": len(taken),
        "distinct_present_labels": len({n for r in rows for n in r["labels_present"]}),
        "photos_with_verified_absences": sum(1 for r in rows if r["labels_absent"]),
        "filters": dict(skipped), "labels_in_test_split": len(freq),
        "candidates_after_filters": len(meta), "usable_candidates": len(usable),
    }
    (RAW_DIR / "selection_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


def run_annotate() -> None:
    """Add `labels_present_concrete` / `labels_absent_concrete` to an existing selection.

    The 2,461 image-level labels in the test split are a fine diversity signal, but many of them are
    colours ("Black"), events ("Academic conference") or qualities rather than things you can point
    at, and a question like "what colour is a black?" is nonsense.  The 601 *boxable* classes are by
    construction things with a bounding box, i.e. concrete objects, so the question templates draw
    their present/absent candidates from that subset.  The diversity buckets keep the full set:
    this step only changes what may be ASKED about, never which photos were chosen (the selection
    and the downloaded images are untouched).
    """
    boxable = {r["DisplayName"] for r in csv.DictReader((RAW_DIR / "oidv7-class-descriptions-boxable.csv").open())}
    boxable -= ABSTRACT | BODY_PARTS
    rows = read_jsonl(RAW_DIR / "selection.jsonl")
    for row in rows:
        row["labels_present_concrete"] = sorted(set(row["labels_present"]) & boxable)
        row["labels_absent_concrete"] = sorted(set(row["labels_absent"]) & boxable)
    (RAW_DIR / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {
        "boxable_classes_used": len(boxable),
        "photos": len(rows),
        "with_concrete_present": sum(1 for r in rows if r["labels_present_concrete"]),
        "with_concrete_absent": sum(1 for r in rows if r["labels_absent_concrete"]),
        "distinct_concrete_present": len({n for r in rows for n in r["labels_present_concrete"]}),
    }
    path = RAW_DIR / "concrete_label_stats.json"
    path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def run_download(workers: int, limit: int | None) -> None:
    rows = read_jsonl(RAW_DIR / "selection.jsonl")
    manifest_path = RAW_DIR / "manifest.jsonl"
    attrib_path = OUT / SOURCE / "attribution.jsonl"
    done = {r["image_id"] for r in read_jsonl(manifest_path)}
    todo = [r for r in rows if r["image_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(rows)} selected, {len(done)} already stored, {len(todo)} to fetch", flush=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sess = session(pool=workers * 2)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}

    def one(row):
        try:
            r = get_with_backoff(sess, row["url"], timeout=(20, 180))
            entry = store_image(r.content, IMG_DIR)
        except Exception as exc:
            with lock:
                counters["fail"] += 1
            return None, f"{row['image_id']}: {exc}"
        with lock:
            counters["ok"] += 1
        return ({
            "image_id": row["image_id"], "url": row["url"], "sha256": entry["sha256"],
            "image": repo_relative(entry["image"]), "width": entry["width"], "height": entry["height"],
            "bytes": len(r.content), "spdx": row["spdx"],
        }, {
            "sha256": entry["sha256"], "image_id": row["image_id"], "author": row["author"],
            "author_profile": row["author_profile"], "title": row["title"],
            "license": "CC BY 2.0", "spdx": row["spdx"], "license_url": row["license_url"],
            "flickr_landing_url": row["flickr_landing_url"], "source_url": row["url"],
        }), None

    batch_m, batch_a = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (payload, err) in enumerate(pool.map(one, todo), 1):
            if err:
                if counters["fail"] <= 20:
                    print(f"  ! {err}", flush=True)
                continue
            batch_m.append(payload[0])
            batch_a.append(payload[1])
            if len(batch_m) >= 250:
                append_jsonl(manifest_path, batch_m)
                append_jsonl(attrib_path, batch_a)
                batch_m, batch_a = [], []
                print(f"  {i}/{len(todo)} ok={counters['ok']} fail={counters['fail']}", flush=True)
    if batch_m:
        append_jsonl(manifest_path, batch_m)
        append_jsonl(attrib_path, batch_a)
    print(f"stored ok={counters['ok']} fail={counters['fail']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "annotate", "download"])
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--per-label-cap", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    if args.phase == "select":
        run_select(args.target, args.per_label_cap, args.seed)
    elif args.phase == "annotate":
        run_annotate()
    else:
        run_download(args.workers, args.limit)
