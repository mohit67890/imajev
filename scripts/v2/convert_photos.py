"""decision-v2 photo converter: `commons_photos`, `openimages_v2` and `pd12m` -> candidate decisions.

Every row this writes is a *question without an answer*: `target: null`, `abstention_cause: null`,
`pseudo_label: "pending"`.  The 9B teacher fills those in later (`scripts/v2/pseudo_label.py`), so
nothing here may depend on knowing the right answer.  What the converter does decide is the shape
of the decision: which family, which wording, which options, which state, and which partition.

Run:
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/convert_photos.py commons_photos
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/convert_photos.py openimages_v2
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/convert_photos.py pd12m

Deterministic from `--seed`; re-running rewrites `records.jsonl` from the stored manifests without
touching the network or the images.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, ROOT, read_jsonl, repo_relative, stable_partition, verified_license, write_jsonl
from v2.templates import photo as T

PARTITION_SEED = "decision-v2-photo"
DECISIONS_PER_PHOTO = 3
TWO_IMAGE_SHARE = 0.10
UNKNOWN_SHARE = 0.20
SAME_CATEGORY_PAIR_SHARE = 0.60  # the rest are cross-category, so the "same subject?" boolean is not all-yes
ABSENT_OBJECT_SHARE = 0.40       # object_present questions asked about something believed NOT to be there

# Family weights per source.  `scene_type` is Commons-only: Commons files carry a curator-chosen
# subject category, so a plausible scene option set can be built; Open Images' 2,461 object labels
# do not map to a scene without guessing, and guessing would quietly turn most Open Images
# scene questions into `unknown`.
SINGLE_FAMILY_WEIGHTS = {
    "commons_photos": {
        "scene_type": 1.4, "main_object": 1.2, "object_present": 1.4, "count_bucket": 1.0,
        "photo_quality": 0.8, "colour_of_x": 1.0, "text_visible": 0.8, "material": 1.0,
        "indoor_outdoor": 1.0,
    },
    "openimages_v2": {
        "main_object": 1.4, "object_present": 1.6, "count_bucket": 1.2, "photo_quality": 0.8,
        "colour_of_x": 1.2, "text_visible": 0.8, "material": 1.0, "indoor_outdoor": 1.0,
    },
    # pd12m gets `scene_type` back: every one of its 73 caption buckets has an authored scene list
    # in `photo.CATEGORY_SCENES`, so a plausible scene option set exists (which is exactly what
    # Open Images' bare object ontology could not give).
    "pd12m": {
        "scene_type": 1.2, "main_object": 1.2, "object_present": 1.6, "count_bucket": 1.0,
        "photo_quality": 0.8, "colour_of_x": 1.2, "text_visible": 0.8, "material": 1.0,
        "indoor_outdoor": 1.0,
    },
}
TWO_IMAGE_WEIGHTS = {"two_image_same_subject": 0.55, "two_image_which": 0.45}

# Per-source overrides.  A source not listed here keeps the module-level default, so adding
# `pd12m` changed nothing for `commons_photos` or `openimages_v2`.
# pd12m sits BELOW the module default because it also reaches an unknown construction by itself:
# about a fifth of its captions name nothing in the object lexicon, and `main_object` /
# `colour_of_x` then cannot list the right answer whatever the caller asked for.  0.18 + that
# fallback lands inside the spec's 15-25% band; see the source README for the measured figure.
UNKNOWN_SHARE_BY_SOURCE = {"pd12m": 0.18}
# Higher than the 0.40 default: a PD12M caption gives a usable "probably not in this photo" pool
# for every photo, which Commons cannot do at all.
ABSENT_OBJECT_SHARE_BY_SOURCE = {"pd12m": 0.50}
# State kinds. `None` means "use photo.make_state's own weights" (25/30/25/20).  pd12m tilts
# towards `contradictory` -- the state describes a DIFFERENT scene from the photograph -- because
# the brief asks this source to press contradicting-state questions harder.  The `empty` weight is
# left at 25 so the ~40% free-text-state share the spec asks for is unchanged.
STATE_KIND_WEIGHTS = {"pd12m": {"empty": 25, "aligned": 22, "irrelevant": 23, "contradictory": 30}}

LICENSES = {
    "commons_photos": {
        spdx: f"data/decision-v2/licenses/commons_photos/{spdx}.deed.html"
        for spdx in ("CC0-1.0", "CC-BY-3.0", "CC-BY-4.0")
    },
    "openimages_v2": {"CC-BY-2.0": "data/decision-v2/licenses/openimages_v2/CC-BY-2.0.deed.html"},
    # PD12M's per-row licence column is a MIX of CC0-1.0 and Public Domain Mark 1.0; only CC0-1.0
    # is fetched (scripts/v2/fetch_pd12m.py:KEEP_LICENSE), so only CC0-1.0 can appear here.
    "pd12m": {"CC0-1.0": "data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"},
}
ANNOTATION_LICENSE = {
    "openimages_v2": ("CC-BY-4.0", "data/decision-v2/licenses/openimages_v2/CC-BY-4.0.deed.html"),
}
# NB: never the bare string "test" -- `scripts/v1_text/audit_mixture.py` reads
# `source_split == "test"` as "this is a held-out family that must sit in the test partition".
SOURCE_SPLIT = {"commons_photos": "crawl", "openimages_v2": "openimages_v7_test", "pd12m": "crawl"}

MAX_PRESENT = 8
MAX_ABSENT = 12


def banned_sha() -> set[str]:
    path = ROOT / "data" / "decision-v1" / "exclusions.json"
    return set(json.loads(path.read_text())["sha256"]) if path.is_file() else set()


def load_photos(source: str) -> list[dict]:
    """-> one dict per stored photo, deduplicated by sha256, with the inputs the templates want."""
    manifest = read_jsonl(RAW / source / "manifest.jsonl")
    photos, seen_sha = [], set()
    if source == "commons_photos":
        for row in manifest:
            if row["sha256"] in seen_sha:
                continue
            seen_sha.add(row["sha256"])
            label = row.get("category_label", "")
            subject = T.CATEGORY_OBJECT.get(label)
            photos.append({
                "key": str(row["pageid"]),
                "category": label,
                # Commons has no per-file object annotation: the only claim available is the
                # category a human filed the file under.  Used for plausible options only.
                "present": [subject] if subject else [],
                "absent": [],
                "title": (row.get("title") or "").removeprefix("File:").rsplit(".", 1)[0][:120],
                "spdx": row["spdx"],
                "image": {"image": repo_relative(row["image"]), "sha256": row["sha256"],
                          "width": row["width"], "height": row["height"]},
            })
    elif source == "pd12m":
        labels = {r["pd12m_id"]: r for r in read_jsonl(RAW / source / "selection.jsonl")}
        for row in manifest:
            if row["sha256"] in seen_sha:
                continue
            seen_sha.add(row["sha256"])
            extra = labels.get(row["pd12m_id"], {})
            photos.append({
                "key": row["pd12m_id"],
                "category": row.get("bucket_label") or extra.get("bucket_label", ""),
                # A PD12M caption is machine-written and unverified, so `present` is a CLAIM and
                # `absent` is "the caption did not mention it, and it is not typical of this
                # bucket" -- weaker than Open Images' human-verified Confidence==0. The teacher
                # settles every one of them.
                "present": list(extra.get("objects_present_claimed", []))[:MAX_PRESENT],
                "absent": list(extra.get("objects_absent_claimed", []))[:MAX_ABSENT],
                # DELIBERATELY EMPTY. `photo.make_state` puts `title` into the `aligned` state as a
                # caption, and PD12M's only title is its caption. Letting it through would hand the
                # teacher the answer in words; the teacher must read the pixels. The caption is
                # kept in attribution.jsonl and drives template choice only.
                "title": "",
                "spdx": row["spdx"],
                "image": {"image": repo_relative(row["image"]), "sha256": row["sha256"],
                          "width": row["width"], "height": row["height"]},
            })
    else:
        labels = {r["image_id"]: r for r in read_jsonl(RAW / source / "selection.jsonl")}
        for row in manifest:
            if row["sha256"] in seen_sha:
                continue
            seen_sha.add(row["sha256"])
            extra = labels.get(row["image_id"], {})
            photos.append({
                "key": row["image_id"],
                "category": extra.get("bucket_label", ""),
                # concrete (boxable) labels only: a question may only ask about a thing you could
                # put a box round, never about "Black" or "Academic conference"
                "present": list(extra.get("labels_present_concrete", []))[:MAX_PRESENT],
                "absent": list(extra.get("labels_absent_concrete", []))[:MAX_ABSENT],
                "title": (extra.get("title") or "")[:120],
                "spdx": row["spdx"],
                "image": {"image": repo_relative(row["image"]), "sha256": row["sha256"],
                          "width": row["width"], "height": row["height"]},
            })
    return photos


def weighted(rng, weights: dict) -> str:
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def build(source: str, seed: int, limit: int | None = None) -> dict:
    licence_cache = {spdx: verified_license(Path(rel), spdx) for spdx, rel in LICENSES[source].items()}
    annotation = None
    if source in ANNOTATION_LICENSE:
        spdx, rel = ANNOTATION_LICENSE[source]
        annotation = verified_license(Path(rel), spdx)

    banned = banned_sha()
    photos = load_photos(source)
    dropped_banned = sum(1 for p in photos if p["image"]["sha256"] in banned)
    photos = [p for p in photos if p["image"]["sha256"] not in banned]
    if limit:
        photos = photos[:limit]
    for p in photos:
        p["group"] = f"{source}:{p['key']}"
        p["partition"] = stable_partition(p["group"], seed=PARTITION_SEED)

    # pools used to find a partner for a two-image record; a partner is always in the SAME
    # partition, so no photo can reach across the split through a pair
    by_partition: dict[str, list[int]] = defaultdict(list)
    by_partition_category: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(photos):
        by_partition[p["partition"]].append(i)
        by_partition_category[(p["partition"], p["category"])].append(i)

    single_weights = SINGLE_FAMILY_WEIGHTS[source]
    unknown_families = {f: w for f, w in single_weights.items() if f in T.UNKNOWN_CAPABLE}
    unknown_share = UNKNOWN_SHARE_BY_SOURCE.get(source, UNKNOWN_SHARE)
    absent_share = ABSENT_OBJECT_SHARE_BY_SOURCE.get(source, ABSENT_OBJECT_SHARE)
    state_weights = STATE_KIND_WEIGHTS.get(source)
    rows, stats = [], Counter()

    for i, p in enumerate(photos):
        rng = random.Random(f"{seed}\0{source}\0{p['key']}")
        used_templates = set()
        for k in range(DECISIONS_PER_PHOTO):
            two_image = rng.random() < TWO_IMAGE_SHARE
            unknown = (not two_image) and rng.random() < unknown_share
            partner = None
            if two_image:
                same_category = rng.random() < SAME_CATEGORY_PAIR_SHARE
                pool = by_partition_category[(p["partition"], p["category"])] if same_category \
                    else by_partition[p["partition"]]
                if len(pool) < 2:
                    pool = by_partition[p["partition"]]
                if len(pool) < 2:
                    two_image = False
                else:
                    for _ in range(8):
                        j = pool[rng.randrange(len(pool))]
                        if j != i and (same_category or photos[j]["category"] != p["category"]):
                            partner = photos[j]
                            break
                    if partner is None:
                        two_image = False

            inputs = {"id": p["key"], "category": p["category"], "present": p["present"],
                      "absent": p["absent"], "title": p["title"]}
            if two_image and partner is not None:
                inputs["second"] = {"category": partner["category"], "present": partner["present"],
                                    "absent": partner["absent"], "title": partner["title"]}
                inputs["same_category"] = partner["category"] == p["category"]
                family = weighted(rng, TWO_IMAGE_WEIGHTS)
                fields, template_id = T.make(family, inputs, rng)
                images = [p["image"], partner["image"]]
                group = f"{p['group']}+{partner['group']}"
            else:
                if unknown and unknown_families:
                    family = weighted(rng, unknown_families)
                else:
                    unknown = False
                    family = weighted(rng, single_weights)
                if family == "object_present":
                    fields, template_id = T.make_object_present(
                        inputs, rng, about_absent=rng.random() < absent_share)
                elif family == "count_bucket":
                    fields, template_id = T.make_count_bucket(
                        inputs, rng, about_absent=rng.random() < absent_share)
                else:
                    fields, template_id = T.make(family, inputs, rng, unknown=unknown)
                images = [p["image"]]
                group = p["group"]
            if template_id in used_templates:  # never ask one photo the same worded question twice
                fields, template_id = T.make(family, inputs, rng, unknown=unknown)
            used_templates.add(template_id)

            kind = None if state_weights is None else rng.choices(
                sorted(state_weights), weights=[state_weights[k] for k in sorted(state_weights)])[0]
            state, state_kind = T.make_state(inputs, rng, kind=kind)
            record = {
                "id": f"{source}:{p['key']}#{k}",
                "source": source,
                "source_group": group,
                "partition": p["partition"],
                "family": family,
                "source_split": SOURCE_SPLIT[source],
                "license": licence_cache[p["spdx"]],
                "images": images,
                "request": {
                    "schema_version": "1.0",
                    "request_id": f"{source}-{p['key']}-{k}"[:128],
                    "state": state,
                    "fields": fields,
                },
                "target": None,
                "abstention_cause": None,
                "source_answer": None,
                "pseudo_label": "pending",
                "template_id": template_id,
                "state_kind": state_kind,
                # read back from the template id: a family falls back to an unknown construction
                # on its own when a photo has no usable present label
                "unknown_construction": T.is_unknown_construction(family, template_id),
                "category_label": p["category"],
            }
            if len(images) == 2 and partner is not None:
                record["pair_same_category"] = partner["category"] == p["category"]
                record["partner_category_label"] = partner["category"]
            if p["partition"] == "test":
                record["pseudo_label_test"] = True
            if annotation is not None:
                record["annotation_license"] = annotation
            rows.append(record)
            stats[family] += 1
            stats[f"state:{state_kind}"] += 1
            stats["two_image"] += len(images) == 2
            stats["unknown_construction"] += record["unknown_construction"]

    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", rows)
    summary = {
        "source": source,
        "photos": len(photos),
        "photos_dropped_for_exclusion_list": dropped_banned,
        "records": len(rows),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "families": {f: stats[f] for f in sorted(single_weights) + sorted(TWO_IMAGE_WEIGHTS) if stats[f]},
        "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in rows)),
        "two_image_records": stats["two_image"],
        "two_image_share": round(stats["two_image"] / max(1, len(rows)), 4),
        "two_image_same_category": sum(bool(r.get("pair_same_category")) for r in rows),
        "unknown_construction": stats["unknown_construction"],
        "unknown_construction_share": round(stats["unknown_construction"] / max(1, len(rows)), 4),
        "state_kinds": {k.split(":", 1)[1]: v for k, v in sorted(stats.items()) if k.startswith("state:")},
        "state_is_string": sum(isinstance(r["request"]["state"], str) for r in rows),
        "state_string_share": round(sum(isinstance(r["request"]["state"], str) for r in rows) / max(1, len(rows)), 4),
        "distinct_templates": len({r["template_id"] for r in rows}),
        "licenses": dict(Counter(r["license"]["spdx"] for r in rows)),
        "options_with_criteria": sum(
            any("description" in o for o in r["request"]["fields"][0].get("options", []))
            for r in rows),
        # Jev-style "option -> what it means" criteria, counting the boolean yes/no criteria too
        "records_with_criteria": sum(
            any("description" in o for o in r["request"]["fields"][0].get("options", []))
            or "yes_description" in r["request"]["fields"][0]
            for r in rows),
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(SINGLE_FAMILY_WEIGHTS))
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.source, args.seed, args.limit)
