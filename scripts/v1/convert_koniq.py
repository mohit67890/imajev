"""KonIQ-10k -> decision-v1 `koniq`: ordinal image quality with the human rating histogram.

KonIQ ships, per image, the counts of how many of ~120 crowd workers picked each level of the
native ACR scale (bad / poor / fair / good / excellent). That histogram becomes
`target_distribution` - a real soft target, not a synthetic one - while `target` is the rounded
MOS level. Level windows vary per example; a window that excludes the gold level yields
`not_listed`, but only when the excluded levels hold less than a quarter of the rating mass, so
the abstention label is never a coin flip dressed up as a fact.

LIVE-in-the-Wild is deliberately NOT converted: it stays the held-out ordinal transfer probe.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import store_image, write_jsonl  # noqa: E402

SOURCE = "koniq"
SEED = 20260925
LICENSE = "CC-BY-4.0 (dataset); per-image Creative Commons terms inherited from YFCC100M"
CACHE = Path(".cache/datasets/v1/koniq")
OUT = Path("data/decision-v1/koniq")
IMAGES = OUT / "images"
SCORES_URL = "https://darus.uni-stuttgart.de/api/access/datafile/72288"
IMAGES_URL = "https://darus.uni-stuttgart.de/api/access/datafile/72286"

TARGET_TOTAL = 8000
TEST_RECORDS = 300
DEV_FRACTION = 0.03
NOT_LISTED_RATE = 0.16
MASS_GUARD = 0.25          # never call a level "not listed" if the offered window holds >=75% of votes
PER_LEVEL_CAP = {1: 400, 2: 1400, 3: 3200, 4: 3200, 5: 400}

LEVELS = [
    {"value": 1, "description": "bad"},
    {"value": 2, "description": "poor"},
    {"value": 3, "description": "fair"},
    {"value": 4, "description": "good"},
    {"value": 5, "description": "excellent"},
]
TEMPLATES = [
    "How would you rate the visual quality of this photo?",
    "Rate the technical picture quality of this image.",
    "On this scale, how good does this picture look?",
    "What quality rating would most viewers give this photo?",
    "Judge the picture quality of this image on the scale given.",
    "How good is the image quality here?",
]


def hkey(*parts):
    return hashlib.sha256((":".join(str(p) for p in parts)).encode()).hexdigest()


def load_scores():
    path = CACHE / "koniq10k_scores_and_distributions.tab"
    if not path.exists():
        raise SystemExit(f"download {SCORES_URL} to {path} first")
    rows = []
    for r in csv.DictReader(path.open(), delimiter="\t"):
        name = r["image_name"].strip().strip('"')
        counts = [int(r[f"c{i}"]) for i in range(1, 6)]
        total = sum(counts)
        if total <= 0:
            continue
        mos = float(r["MOS"])
        rows.append(dict(name=name, counts=counts, total=total, mos=mos,
                         level=min(5, max(1, round(mos)))))
    return rows


def window_for(rng, level, counts, total):
    """A per-example window of consecutive levels; sometimes one that excludes the gold."""
    if rng.random() < NOT_LISTED_RATE:
        candidates = []
        for lo in range(1, 6):
            for hi in range(lo + 1, 6):
                if lo <= level <= hi:
                    continue
                mass = sum(counts[i - 1] for i in range(lo, hi + 1)) / total
                if mass < MASS_GUARD:
                    continue
                candidates.append((lo, hi))
        if candidates:
            lo, hi = candidates[rng.randrange(len(candidates))]
            return list(range(lo, hi + 1)), "not_listed"
    if rng.random() < 0.6:
        return [1, 2, 3, 4, 5], None
    lo = max(1, level - rng.randint(0, 2))
    hi = min(5, max(level + rng.randint(0, 2), lo + 1))
    return list(range(lo, hi + 1)), None


def main():
    archive = CACHE / "koniq10k_512x384.zip"
    if not archive.exists():
        raise SystemExit(f"download {IMAGES_URL} to {archive} first")
    rows = load_scores()
    rng = random.Random(SEED)

    with zipfile.ZipFile(archive) as z:
        members = {Path(n).name: n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        rows = [r for r in rows if r["name"] in members]

        # KonIQ ships no official split: partition deterministically by image id and say so.
        def partition_of(name):
            h = int(hkey(SEED, "p", name), 16) % 10000
            if h < TEST_RECORDS * 10000 // max(1, TARGET_TOTAL):
                return "test"
            return "dev" if h % 1000 < DEV_FRACTION * 1000 else "train"

        by_level = defaultdict(list)
        for r in sorted(rows, key=lambda r: hkey(SEED, "o", r["name"])):
            by_level[r["level"]].append(r)
        selected = []
        for level, cap in PER_LEVEL_CAP.items():
            selected += by_level[level][:cap]
        selected = sorted(selected, key=lambda r: hkey(SEED, "s", r["name"]))[:TARGET_TOTAL]

        out = []
        for r in selected:
            window, cause = window_for(rng, r["level"], r["counts"], r["total"])
            field = {"id": "answer", "type": "ordinal",
                     "question": TEMPLATES[int(hkey(SEED, "t", r["name"]), 16) % len(TEMPLATES)],
                     "levels": [LEVELS[i - 1] for i in window]}
            record = {"id": f"{SOURCE}:{Path(r['name']).stem}", "source": SOURCE,
                      "source_split": "koniq10k", "source_group": r["name"], "family": "image_quality",
                      "license": LICENSE, "images": [store_image(z.read(members[r["name"]]), IMAGES)],
                      "request": {"request_id": f"{SOURCE}-{Path(r['name']).stem}"[:128], "state": {},
                                  "fields": [field]},
                      "target": None if cause else r["level"], "abstention_cause": cause,
                      "source_answer": f"MOS={r['mos']:.3f} over {r['total']} ratings",
                      "partition": partition_of(r["name"])}
            if cause is None:
                mass = sum(r["counts"][i - 1] for i in window)
                if mass > 0:
                    record["target_distribution"] = {str(i): round(r["counts"][i - 1] / mass, 6)
                                                     for i in window}
            out.append(record)
            if len(out) % 1000 == 0:
                print(f"  {len(out)}/{len(selected)}", flush=True)

    out.sort(key=lambda r: r["id"])
    write_jsonl(OUT / "records.jsonl", out)
    summary = dict(records=len(out), images=len({r["images"][0]["sha256"] for r in out}),
                   partitions=dict(Counter(r["partition"] for r in out)),
                   families=dict(Counter(r["family"] for r in out)),
                   field_types=dict(Counter(r["request"]["fields"][0]["type"] for r in out)),
                   abstention=dict(Counter(str(r["abstention_cause"]) for r in out)),
                   targets=dict(Counter(str(r["target"]) for r in out)),
                   with_distribution=sum("target_distribution" in r for r in out),
                   level_window_sizes=dict(sorted(Counter(len(r["request"]["fields"][0]["levels"])
                                                          for r in out).items())))
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
