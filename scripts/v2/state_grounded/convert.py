"""decision-v2 `state_grounded` converter: photos already on disk -> state-grounded candidates.

Every row this writes is a *question without an answer*: `target: null`, `abstention_cause: null`,
`pseudo_label: "pending"`.  The 9B teacher fills those in later (`scripts/v2/pseudo_label.py`).

Nothing here downloads, resizes or stores an image.  Every record points at an image file an
earlier converter already wrote:

    data/decision-v2/{openimages_v2,pd12m,commons_photos}/images/<sha>.jpg
    data/decision-v1/abo/images/<sha>.jpg

and copies that source's verified licence object.  `source_group` is the photo id, so a photo
never spans partitions, and the partition seed is the one `scripts/v2/convert_photos.py` uses, so
a photo that also carries `photo.py` questions lands in the SAME partition in both sources.

Run:
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/convert.py
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_image_records.py state_grounded
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from v2.common import OUT, RAW, ROOT, read_jsonl, repo_relative, stable_partition, verified_license, write_jsonl
from v2.state_grounded import templates as T
from v2.templates import photo as P

SOURCE = "state_grounded"
# The seed `scripts/v2/convert_photos.py` uses, so a photo's partition is the same in both sources.
PARTITION_SEED = "decision-v2-photo"
SEED = 20260923

# Target decisions per family.  ~60,000 in all; the paired family is 10% of the rows (5% of the
# photos, two records each).
FAMILY_TARGET = {
    "claim_supported": 20_000,
    "which_field_conflicts": 12_000,
    "count_matches": 8_000,
    "label_text_matches": 6_000,
    "condition_report": 8_000,
    "instruction_changes_answer": 6_000,      # 3,000 pairs
}
MAX_DECISIONS_PER_PHOTO = 4

# Intent mix per family.  An intent is what the converter AIMS at, never a label: the teacher
# settles every record.  `claim_supported` is aimed at the 35/35/30 the brief asks for.
INTENT_WEIGHTS = {
    "claim_supported": {"true": 35, "false": 35, "unverifiable": 30},
    "which_field_conflicts": {"one_conflict": 70, "none": 30},
    "count_matches": {"match": 34, "fewer": 40, "more": 26},
    "label_text_matches": {"word": 45, "code": 55},
    "condition_report": {"reported": 100},
}

# Which sources can honestly carry which family.
FAMILY_SOURCES = {
    "claim_supported": {"abo": 3.0, "openimages_v2": 2.2, "pd12m": 2.0, "commons_photos": 0.6},
    "which_field_conflicts": {"abo": 4.0, "openimages_v2": 1.6, "pd12m": 1.4, "commons_photos": 0.2},
    "count_matches": {"abo": 0.5, "openimages_v2": 2.4, "pd12m": 2.2, "commons_photos": 0.4},
    "label_text_matches": {"abo": 2.0, "openimages_v2": 1.6, "pd12m": 1.4, "commons_photos": 1.0},
    "condition_report": {"abo": 3.0, "openimages_v2": 1.4, "pd12m": 1.4, "commons_photos": 0.0},
    "instruction_changes_answer": {"abo": 0.0, "openimages_v2": 3.0, "pd12m": 2.0,
                                   "commons_photos": 0.0},
}

LICENSES = {
    "openimages_v2": ("CC-BY-2.0", "data/decision-v2/licenses/openimages_v2/CC-BY-2.0.deed.html"),
    "pd12m": ("CC0-1.0", "data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"),
    "abo": ("CC-BY-4.0", "data/decision-v2/licenses/state_grounded/abo-LICENSE-CC-BY-4.0.txt"),
}
COMMONS_LICENSES = {spdx: f"data/decision-v2/licenses/commons_photos/{spdx}.deed.html"
                    for spdx in ("CC0-1.0", "CC-BY-3.0", "CC-BY-4.0")}
ANNOTATION_LICENSE = ("CC-BY-4.0", "data/decision-v2/licenses/openimages_v2/CC-BY-4.0.deed.html")
# Never the bare string "test": `scripts/v1_text/audit_mixture.py` reads that as "held-out family".
SOURCE_SPLIT = {"abo": "abo_train_photos", "openimages_v2": "openimages_v7_test",
                "pd12m": "crawl", "commons_photos": "crawl"}

# Diversity caps.  ABO is dominated by phone cases (11,596 of 24,859 listings with attributes);
# without a cap half this source would be one product type.
ABO_PER_PRODUCT_TYPE = 400
PHOTO_CAP = {"abo": 13_000, "openimages_v2": 20_000, "pd12m": 16_000, "commons_photos": 4_000}

ABO_META_CACHE = ROOT / ".cache" / "datasets" / "v1" / "state_aware" / "abo_listing_meta.json"
ABO_META_CACHE_V2 = ROOT / ".cache" / "datasets" / "v2" / "state_grounded" / "abo_listing_meta.json"
ABO_LISTINGS_TAR = ROOT / "data" / "decision-v1-text" / "raw" / "abo" / "abo-listings.tar"
ABO_ID = re.compile(
    r"^(?:cat|color|material|pattern|item_shape|finish|style|same"
    r"|reference_color|reference_material|reference_category"
    r"|bool-(?:color|material|pattern|item_shape|category))-([A-Z0-9]{10})-")

# ABO's `attrs` keys, renamed to the state keys the templates use.
ABO_ATTR_KEYS = {"color": "colour", "material": "material", "product_type": "product_type",
                 "pattern": "pattern", "shape": "shape", "finish": "finish"}
# Colour values a photograph cannot sensibly confirm or deny for a whole product.
ABO_WEAK_COLOURS = {"multicoloured", "transparent"}


def banned_sha() -> set[str]:
    path = ROOT / "data" / "decision-v1" / "exclusions.json"
    return set(json.loads(path.read_text())["sha256"]) if path.is_file() else set()


# --------------------------------------------------------------------------- ABO metadata
def load_abo_meta() -> dict:
    """The ABO listing metadata `scripts/v1/convert_state_aware.py` already extracted.

    Falls back to reading `abo-listings.tar` (which is in the repo) when that cache is absent, so
    this converter does not depend on a hidden cache directory surviving.
    """
    for cache in (ABO_META_CACHE_V2, ABO_META_CACHE):
        if cache.is_file():
            return json.loads(cache.read_text())
    sys.path.insert(0, str(ROOT / "scripts" / "v1"))
    import abo_taxonomy as AT  # noqa: E402  (only reached when the v1 cache is missing)

    def english(entries):
        return [e for e in entries or [] if str(e.get("language_tag", "")).startswith("en")]

    def curated(row, key, vmap):
        raw = set()
        for e in english(row.get(key, [])):
            if key == "color":
                raw |= {str(s).strip().lower() for s in (e.get("standardized_values") or [])}
            raw.add(str(e.get("value", "")).strip().lower())
        hits = {vmap[v] for v in raw if v in vmap}
        return next(iter(hits)) if len(hits) == 1 else None

    meta = {}
    with tarfile.open(ABO_LISTINGS_TAR) as tar:
        for member in tar:
            if not member.name.endswith(".json.gz"):
                continue
            for line in gzip.GzipFile(fileobj=tar.extractfile(member)):
                row = json.loads(line)
                item = row["item_id"]
                if item in meta:
                    continue
                pt = (row.get("product_type") or [{}])[0].get("value")
                label = AT.PRODUCT_TYPES.get(pt)
                name = (english(row.get("item_name", [])) or [{}])[0]
                if not label or not name.get("value"):
                    continue
                attrs = {"product_type": label[0]}
                for key, vmap in (("color", AT.COLOR_MAP), ("material", AT.MATERIAL_MAP),
                                  ("pattern", AT.PATTERN_MAP), ("item_shape", AT.SHAPE_MAP),
                                  ("finish_type", AT.FINISH_MAP)):
                    value = curated(row, key, vmap)
                    if value:
                        attrs[{"item_shape": "shape", "finish_type": "finish",
                               "color": "color"}.get(key, key)] = value
                meta[item] = {"title": name["value"], "pt": pt, "cgroup": label[1],
                              "attrs": attrs, "hidden": {}}
    ABO_META_CACHE_V2.parent.mkdir(parents=True, exist_ok=True)
    ABO_META_CACHE_V2.write_text(json.dumps(meta, sort_keys=True))
    return meta


def load_abo_photos(rng, banned) -> list[dict]:
    """One photo per ABO listing, from the images `data/decision-v1/abo` already stored.

    Only listings whose v1 records are ALL in the v1 train partition are used: the v1 dev/test ABO
    photos are part of the v1 image exam and must not reappear as v2 training rows.
    """
    meta = load_abo_meta()
    spdx, rel = LICENSES["abo"]
    licence = verified_license(Path(rel), spdx)
    by_item: dict[str, dict] = {}
    excluded_non_train: set[str] = set()
    for row in read_jsonl(ROOT / "data" / "decision-v1" / "abo" / "records.jsonl"):
        hit = ABO_ID.match(row["id"].split(":", 1)[1])
        if not hit:
            continue
        item = hit.group(1)
        if row["partition"] != "train":
            excluded_non_train.add(item)
            continue
        image = row["images"][0]
        if image["sha256"] in banned:
            continue
        by_item.setdefault(item, image)

    groups = defaultdict(list)
    for item, image in by_item.items():
        if item in excluded_non_train or item not in meta:
            continue
        info = meta[item]
        attrs = {ABO_ATTR_KEYS[k]: v for k, v in info["attrs"].items() if k in ABO_ATTR_KEYS}
        if attrs.get("colour") in ABO_WEAK_COLOURS:
            attrs.pop("colour")
        if not attrs:
            continue
        groups[info["cgroup"] + "\0" + info["attrs"].get("product_type", "")].append(
            (item, image, info, attrs))

    peers_by_group = defaultdict(set)
    for key, rows in groups.items():
        cgroup = key.split("\0")[0]
        for _, _, info, _ in rows[:1]:
            peers_by_group[cgroup].add(info["attrs"].get("product_type", ""))

    photos = []
    for key in sorted(groups):
        rows = groups[key]
        rng.shuffle(rows)
        for item, image, info, attrs in rows[:ABO_PER_PRODUCT_TYPE]:
            cgroup = key.split("\0")[0]
            peers = sorted(peers_by_group[cgroup] - {attrs.get("product_type")})
            subject = attrs.get("product_type") or "the product"
            photos.append({
                "source": "abo", "key": item,
                "image": {"image": repo_relative(image["image"]), "sha256": image["sha256"],
                          "width": image["width"], "height": image["height"]},
                "licence": licence, "spdx": spdx,
                "inputs": {
                    "id": item, "source": "abo", "subject": subject,
                    "present": [subject], "absent": [],
                    "attrs": attrs, "hidden": info.get("hidden") or {},
                    "peers": peers[:40],
                    # The listing title is DELIBERATELY not passed on. An ABO title spells the
                    # product type out in words ("IGI Certified Platinum Princess Diamond Stud
                    # Earrings"), and several attributes with it, so a state carrying the title
                    # would let a model answer a product-type or material question from the text
                    # alone -- exactly the shortcut this source exists to close.
                    "title": "",
                    # ABO is a retail catalogue: the photographs are of new goods, so a damage
                    # report about them has a checkable answer ("no visible damage").
                    "condition_is_new": True,
                },
                "category": attrs.get("product_type", ""),
            })
    return photos


# --------------------------------------------------------------------------- v2 image sources
def load_v2_photos(source: str, banned) -> list[dict]:
    manifest = read_jsonl(RAW / source / "manifest.jsonl")
    if source == "commons_photos":
        selection = {}
    else:
        key = "image_id" if source == "openimages_v2" else "pd12m_id"
        selection = {r[key]: r for r in read_jsonl(RAW / source / "selection.jsonl")}
    if source == "commons_photos":
        licences = {spdx: verified_license(Path(rel), spdx)
                    for spdx, rel in COMMONS_LICENSES.items()}
    else:
        spdx, rel = LICENSES[source]
        licences = {spdx: verified_license(Path(rel), spdx)}

    photos, seen = [], set()
    for row in manifest:
        sha = row["sha256"]
        if sha in seen or sha in banned:
            continue
        seen.add(sha)
        image = {"image": repo_relative(row["image"]), "sha256": sha,
                 "width": row["width"], "height": row["height"]}
        if source == "openimages_v2":
            extra = selection.get(row["image_id"], {})
            present = [_phrase(x) for x in extra.get("labels_present_concrete", [])][:8]
            absent = [_phrase(x) for x in extra.get("labels_absent_concrete", [])][:12]
            key_value, category = row["image_id"], extra.get("bucket_label", "")
            title = ""
        elif source == "pd12m":
            extra = selection.get(row["pd12m_id"], {})
            present = list(extra.get("objects_present_claimed", []))[:8]
            absent = list(extra.get("objects_absent_claimed", []))[:12]
            key_value, category = row["pd12m_id"], extra.get("bucket_label", "")
            title = ""          # PD12M's only title IS its caption; it never enters a state
        else:
            # Commons has no per-file object annotation.  The only thing available is the category
            # a curator filed the file under, which `scripts/v2/templates/photo.py` already treats
            # as a CLAIM rather than a label; its `CATEGORY_ASSOCIATED` table is reused here so the
            # "probably not in this photo" pool does not name something typical of the category.
            key_value = str(row["pageid"])
            category = row.get("category_label", "")
            subject_phrase = P.CATEGORY_OBJECT.get(category)
            present = [subject_phrase] if subject_phrase else []
            withheld = set(P.CATEGORY_ASSOCIATED.get(category, ())) | set(present)
            absent = [x for x in P.COMMON_OBJECTS if x not in withheld][:40]
            random.Random(f"commons\0{key_value}").shuffle(absent)
            absent = absent[:12]
            title = ""          # a Commons filename often names the subject; keep it out
        subject = present[0] if present else (category.replace("_", " ") or "the main subject")
        photos.append({
            "source": source, "key": key_value, "image": image,
            "licence": licences[row["spdx"]], "spdx": row["spdx"],
            "inputs": {"id": key_value, "source": source, "subject": subject,
                       "present": present, "absent": absent, "attrs": {}, "hidden": {},
                       "peers": [], "title": title, "condition_is_new": False},
            "category": category,
        })
    return photos


def _phrase(label: str) -> str:
    """An Open Images label ('Coffee table') as it reads in a claim ('a coffee table')."""
    text = str(label).strip()
    if not text:
        return text
    if text[:1].isupper() and not text.isupper():
        text = text[0].lower() + text[1:]
    return text


# --------------------------------------------------------------------------- build
def weighted_choice(rng, weights: dict):
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


PROBE_MANIFESTS = [ROOT / "data" / "manifests" / "decision-v2-state-probe.jsonl",
                   ROOT / "data" / "manifests" / "decision-v2-pairs-probe.jsonl"]


def probe_groups() -> set[str]:
    """Photos the human-derived probes use.  Most are in the held-out partition anyway; the few
    that are not (Open Images 'Stop sign' / 'Rust' photos, which are too rare to be picky about)
    are kept out of this source altogether so the probe never tests a photo it trained on."""
    groups = set()
    for path in PROBE_MANIFESTS:
        for row in read_jsonl(path):
            groups.add(row["source_group"])
    return groups


def build(seed: int = SEED, scale: float = 1.0, out_dir: Path | None = None) -> dict:
    rng = random.Random(seed)
    banned = banned_sha()
    held_for_probe = probe_groups()
    photos: list[dict] = []
    per_source = Counter()
    for source in ("abo", "openimages_v2", "pd12m", "commons_photos"):
        found = load_abo_photos(random.Random(f"{seed}\0abo"), banned) if source == "abo" \
            else load_v2_photos(source, banned)
        pool = random.Random(f"{seed}\0{source}\0pool")
        pool.shuffle(found)
        found = found[:PHOTO_CAP[source]]
        for p in found:
            p["group"] = f"{p['source']}:{p['key']}"
            p["partition"] = stable_partition(p["group"], seed=PARTITION_SEED)
        found = [p for p in found if p["group"] not in held_for_probe]
        photos += found
        per_source[source] = len(found)

    by_source = defaultdict(list)
    for p in photos:
        by_source[p["source"]].append(p)

    used = Counter()          # photo group -> decisions already emitted for that photo
    used_templates = defaultdict(set)
    rows: list[dict] = []
    stats = Counter()
    annotation = verified_license(Path(ANNOTATION_LICENSE[1]), ANNOTATION_LICENSE[0])

    def emit(photo, family, made, index, pair_id=None):
        group = photo["group"]
        record = {
            "id": f"{SOURCE}:{photo['source']}-{photo['key']}#{index}",
            "source": SOURCE,
            "source_group": group,
            "partition": photo["partition"],
            "family": family,
            "source_split": SOURCE_SPLIT[photo["source"]],
            "license": photo["licence"],
            "images": [photo["image"]],
            "request": {
                "schema_version": "1.0",
                "request_id": f"{SOURCE}-{photo['source']}-{photo['key']}-{index}"[:128],
                "state": made["state"],
                "fields": made["fields"],
            },
            "target": None,
            "abstention_cause": None,
            "source_answer": None,
            "pseudo_label": "pending",
            "template_id": made["template_id"],
            "photo_source": photo["source"],
            "state_kind": made["state_kind"],
            "state_path": made["state_path"],
            "claim_intent": made["intent"],
            "claim_basis": made["basis"],
            "category_label": photo["category"],
        }
        if pair_id:
            record["pair_id"] = pair_id
            record["pair_role"] = made["pair_role"]
        if photo["partition"] == "test":
            record["pseudo_label_test"] = True
        if photo["source"] == "openimages_v2":
            record["annotation_license"] = annotation
        rows.append(record)
        used[group] += 1
        stats[family] += 1
        stats[f"intent:{family}:{made['intent']}"] += 1
        stats[f"photo_source:{photo['source']}"] += 1

    # ---------------------------------------------------------------- the paired family first:
    # it needs the richest photos (something verified present AND two things verified absent).
    target_pairs = int(FAMILY_TARGET["instruction_changes_answer"] * scale) // 2
    pair_pool = [p for src in ("openimages_v2", "pd12m")
                 for p in by_source[src]
                 if p["inputs"]["present"] and len(p["inputs"]["absent"]) >= 2]
    random.Random(f"{seed}\0pairs").shuffle(pair_pool)
    made_pairs = 0
    for photo in pair_pool:
        if made_pairs >= target_pairs:
            break
        r = random.Random(f"{seed}\0pair\0{photo['group']}")
        pair = T.make_instruction_pair(photo["inputs"], r)
        if not pair:
            continue
        pair_id = f"{SOURCE}:pair:{photo['source']}-{photo['key']}"
        for k, made in enumerate(pair):
            emit(photo, "instruction_changes_answer", made, f"pair{k}", pair_id=pair_id)
        made_pairs += 1
        stats["pairs"] += 1

    # ---------------------------------------------------------------- the five single families
    # Most constrained family first: `which_field_conflicts` needs either ABO structured
    # attributes or three human-verified labels, and every photo it spends is one the flexible
    # families could have used instead.
    for family in ("which_field_conflicts", "count_matches", "condition_report",
                   "claim_supported", "label_text_matches"):
        target = int(FAMILY_TARGET[family] * scale)
        weights = {s: w for s, w in FAMILY_SOURCES[family].items() if w > 0 and by_source[s]}
        order: list[dict] = []
        for src, weight in sorted(weights.items()):
            pool = list(by_source[src])
            random.Random(f"{seed}\0{family}\0{src}").shuffle(pool)
            share = weight / sum(weights.values())
            order.append((src, pool, share))
        # interleave the per-source pools in proportion to their weight
        cursors = {src: 0 for src, _, _ in order}
        pools = {src: pool for src, pool, _ in order}
        shares = {src: share for src, _, share in order}
        picker = random.Random(f"{seed}\0{family}\0pick")
        done = 0
        while done < target and any(v > 0 for v in shares.values()):
            src = weighted_choice(picker, shares)
            pool, i = pools[src], cursors[src]
            picked = None
            while i < len(pool):
                candidate = pool[i]
                i += 1
                if used[candidate["group"]] < MAX_DECISIONS_PER_PHOTO:
                    picked = candidate
                    break
            cursors[src] = i
            if picked is None:          # this source has nothing left for this family
                shares[src] = 0.0
                continue
            r = random.Random(f"{seed}\0{family}\0{picked['group']}\0{used[picked['group']]}")
            intent = weighted_choice(r, INTENT_WEIGHTS[family])
            made = T.make(family, picked["inputs"], r, intent=intent)
            if made is None:
                continue                # this photo cannot carry this family honestly
            if made["template_id"] in used_templates[picked["group"]]:
                made = T.make(family, picked["inputs"], r, intent=intent)
                if made is None or made["template_id"] in used_templates[picked["group"]]:
                    continue
            used_templates[picked["group"]].add(made["template_id"])
            emit(picked, family, made, f"{family[:4]}{used[picked['group']]}")
            done += 1
        stats[f"shortfall:{family}"] = target - done

    rows.sort(key=lambda r: r["id"])
    out = out_dir or (OUT / SOURCE)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", rows)

    first = lambda r: r["request"]["fields"][0]
    summary = {
        "source": SOURCE,
        "seed": seed,
        "photos_available": dict(per_source),
        "photos_held_out_for_probes": len(held_for_probe),
        "photos_used": len({r["source_group"] for r in rows}),
        "records": len(rows),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "families": {f: stats[f] for f in T.FAMILIES if stats[f]},
        "photo_sources": {k.split(":", 1)[1]: v for k, v in sorted(stats.items())
                          if k.startswith("photo_source:")},
        "field_types": dict(Counter(first(r)["type"] for r in rows)),
        "intents": {k[len("intent:"):]: v for k, v in sorted(stats.items())
                    if k.startswith("intent:")},
        "pairs": stats["pairs"],
        "paired_records": stats["instruction_changes_answer"],
        "paired_share": round(stats["instruction_changes_answer"] / max(1, len(rows)), 4),
        "state_kinds": dict(Counter(r["state_kind"] for r in rows)),
        "state_is_string": sum(isinstance(r["request"]["state"], str) for r in rows),
        "state_string_share": round(
            sum(isinstance(r["request"]["state"], str) for r in rows) / max(1, len(rows)), 4),
        "distinct_templates": len({r["template_id"] for r in rows}),
        "distinct_questions": len({first(r)["question"] for r in rows}),
        "option_counts": dict(sorted(Counter(
            len(first(r).get("options", first(r).get("levels", []))) for r in rows).items())),
        "records_with_criteria": sum(
            any("description" in o for o in first(r).get("options", []))
            or "yes_description" in first(r) for r in rows),
        "licenses": dict(Counter(r["license"]["spdx"] for r in rows)),
        "decisions_per_photo": dict(sorted(Counter(
            Counter(r["source_group"] for r in rows).values()).items())),
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the family targets to build (for a quick smoke run)")
    args = ap.parse_args()
    build(args.seed, args.scale)
