"""Phase-3 Stage 0 source C (images): our own image-decision training pools as candidates, with their original labels.

Reads data/manifests/decision-v2.1-4b.jsonl (which contains every v1.1 image row, the v2 9B-labelled photo rows, the v2.1
state-grounded, grounded-pair and natural-pair rows) and tags rows that the phase-2c image replay
(data/decision-p2c/image-replay/replay-{fresh,delta}.jsonl) also used. Stage 1 mines these for the ones the shipped model
still fails (docs/phase-3-plan.md rev 4, Stage 0 source I (1) "mine the shipped 4B's failures on our own image pools").

Only rows we may use commercially are kept, and the licence is decided by the PHOTO, not by the row's annotation
licence: every image must come from a pool whose photos carry a recorded permissive licence.

| pool (path under data/)                        | photo licence                         |
|------------------------------------------------|---------------------------------------|
| decision-v1/abo/images                         | CC-BY-4.0 (ABO in-bucket licence)     |
| decision-v1/defects/images                     | CC-BY-4.0 (VisA, DAGM) / CC-BY-SA-4.0 (BTAD), per image |
| decision-v1/vizwiz/images, vizwiz_quality/images | CC-BY-4.0 (VizWiz)                 |
| decision-v2/pd12m/images                       | CC0-1.0 (PD12M)                       |
| decision-v2/pairs_grounded/images_edited       | CC0-1.0 (our PIL edits of PD12M photos) |
| decision-v2/commons_photos/images              | CC0-1.0 / CC-BY-3.0 / CC-BY-4.0, per file (Commons manifest) |

Excluded on purpose: COCO / Flickr / Visual Genome photos (vqav2, tdiuc, bool_abstain, aokvqa, unkvqa, tallyqa, vsr, gqa,
vg_attributes, masked_evidence: annotation licences only, photos not relicensed), crawled retail photos (fashion200k,
marqo_gs, sqid_esci), KonIQ (per-image YFCC terms, some NC), DUDE (crawled documents), and Open Images / TextVQA
(photos CC BY 2.0) EXCEPT the Open Images **state-grounded** rows (owner decision 2026-09-25, after CC-BY-2.0 joined the
allow-list): those are appended in full (train partition) after the 40k balanced sample, with the per-file Flickr
attribution from data/decision-v2/openimages_v2/attribution.jsonl. No other Open Images row is used. Only the train
partition is used; any image whose sha256 is an ImajevBench image is dropped.

    .venv/bin/python scripts/p3/convert_own_images.py [--n 40000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_image_joint import bench_image_shas, rng_for  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "decision-v2.1-4b.jsonl"
REPLAY = {"p2c-replay-fresh": ROOT / "data" / "decision-p2c" / "image-replay" / "replay-fresh.jsonl",
          "p2c-replay-delta": ROOT / "data" / "decision-p2c" / "image-replay" / "replay-delta.jsonl"}
OUT = ROOT / "data" / "p3" / "candidates" / "C-images.jsonl"
SEED = "p3-own-images-v1"

POOLS = {  # data-relative dir -> (pool, licence or None = per image, evidence)
    "decision-v1/abo/images/": ("abo", "CC-BY-4.0", "data/decision-v2/licenses/state_grounded/abo-LICENSE-CC-BY-4.0.txt"),
    "decision-v1/defects/images/": ("defects", None, None),
    "decision-v1/vizwiz/images/": ("vizwiz", "CC-BY-4.0", "data/provenance/licenses/v1-image/vizwiz/CC-BY-4.0.md"),
    "decision-v1/vizwiz_quality/images/": ("vizwiz_quality", "CC-BY-4.0", "data/provenance/licenses/v1-image/vizwiz/CC-BY-4.0.md"),
    "decision-v2/pd12m/images/": ("pd12m", "CC0-1.0", "data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"),
    "decision-v2/pairs_grounded/images_edited/": ("pd12m_edited", "CC0-1.0", "data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"),
    "decision-v2/commons_photos/images/": ("commons_photos", None, None),
}
OPENIMAGES_DIR = "decision-v2/openimages_v2/images/"
OPENIMAGES_EVIDENCE = "data/decision-v2/licenses/openimages_v2/CC-BY-2.0.deed.html"
OPENIMAGES_ATTRIBUTION = ROOT / "data" / "decision-v2" / "openimages_v2" / "attribution.jsonl"
OPENIMAGES_SOURCES = {"state_grounded"}      # the only datasets whose Open Images rows are used
DEFECT_EVIDENCE = {"CC-BY-4.0": "data/provenance/licenses/v1-image/defects/CC-BY-4.0.md",
                   "CC-BY-SA-4.0": "data/provenance/licenses/v1-image/defects/CC-BY-SA-4.0.md"}
COMMONS_EVIDENCE = "data/decision-v2/licenses/commons_photos/{}.deed.html"
ABSTENTION_MAP = {"not_listed": "not_listed", "insufficient_evidence": "insufficient_evidence",
                  "false_premise": "false_premise", "mismatched_reference": "mismatched_reference",
                  "teacher_unknown": "insufficient_evidence"}
# Most restrictive wins (multi-image rows, and a text row's own licence when it carries an irrelevant photo).
LICENCE_RANK = ["CC0-1.0", "MIT", "BSD-3-Clause", "Apache-2.0", "CC-BY-2.0", "CC-BY-3.0", "CC-BY-4.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0"]


class ImageLicences:
    """Resolve an image path (any copy, e.g. the phase-2c replay copy) to its pool file and photo licence."""

    def __init__(self):
        self.by_sha: dict[str, tuple[str, str, str, str]] = {}    # sha -> (rel path, pool, licence, evidence)
        per_image = {}
        for line in open(ROOT / "data" / "decision-v1" / "defects" / "records.jsonl"):
            r = json.loads(line)
            for im in r["images"]:
                per_image[im["sha256"]] = r["license"]
        for line in open(ROOT / "data" / "decision-v2-raw" / "commons_photos" / "manifest.jsonl"):
            r = json.loads(line)
            per_image[r["sha256"]] = r["spdx"]
        for rel, (pool, lic, ev) in POOLS.items():
            d = ROOT / "data" / rel
            for f in d.iterdir():
                sha = f.stem
                if len(sha) != 64:
                    continue
                l = lic or per_image.get(sha)
                if l is None:
                    continue
                e = ev or (DEFECT_EVIDENCE[l] if pool == "defects" else COMMONS_EVIDENCE.format(l))
                self.by_sha.setdefault(sha, (rel + f.name, pool, l, e))
        # Open Images (CC BY 2.0): resolvable, but eligible_rows keeps them only for OPENIMAGES_SOURCES
        self.attribution = {}
        for line in open(OPENIMAGES_ATTRIBUTION):
            a = json.loads(line)
            self.attribution[a["sha256"]] = {k: a.get(k) for k in ("image_id", "author", "author_profile", "title", "license",
                                                                     "license_url", "flickr_landing_url", "source_url")}
        for line in open(ROOT / "data" / "decision-v2-raw" / "openimages_v2" / "manifest.jsonl"):
            r = json.loads(line)
            if r.get("spdx") == "CC-BY-2.0" and (ROOT / r["image"]).is_file():
                self.by_sha.setdefault(r["sha256"], (OPENIMAGES_DIR + Path(r["image"]).name, "openimages_v2", "CC-BY-2.0",
                                                     OPENIMAGES_EVIDENCE))

    def resolve(self, image: dict):
        sha = image.get("sha256") or Path(image["image"]).stem
        return self.by_sha.get(sha)


def convert_field(fl: dict):
    """Manifest field -> candidate field + a gold mapper; None when the field cannot be expressed."""
    t = fl["type"]
    q = fl.get("question") or ""
    if len(q) < 8:
        return None
    if t == "boolean":
        f = {"type": "noul", "question": q}
        for k in ("yes_description", "no_description"):
            if fl.get(k):
                f[k] = fl[k]
        return f, (lambda v: v if isinstance(v, bool) else "__bad__")
    if t == "choice":
        taken, opts, keymap = set(), [], {}
        for o in fl.get("options") or []:
            val = str(o["value"])
            if val.strip().lower() in ("unknown", "__unknown__"):
                return None
            k = option_key(val, taken)
            keymap[val] = k
            opt = {"key": k, "text": val}
            if o.get("description"):
                opt["description"] = o["description"]
            opts.append(opt)
        if not 2 <= len(opts) <= 254:
            return None
        return {"type": "choice", "question": q, "options": opts}, (lambda v: keymap.get(str(v), "__bad__"))
    if t == "ordinal":
        levels = sorted(fl.get("levels") or [], key=lambda l: l["value"])
        if not 2 <= len(levels) <= 10:
            return None
        idx = {l["value"]: i for i, l in enumerate(levels)}
        f = {"type": "score", "question": q,
             "levels": [{"value": i, "description": str(l.get("description") or l["value"])} for i, l in enumerate(levels)]}
        return f, (lambda v: idx.get(v, "__bad__"))
    return None


def difficulty(row: dict, n_images: int, gold) -> int:
    d = 2
    st = row["request"].get("state")
    if st:
        d += 1
    if n_images > 1:
        d += 1
    if gold is None:
        d += 1
    return max(1, min(5, d))


def load_replay_ids() -> dict[str, list[str]]:
    out = defaultdict(list)
    for name, path in REPLAY.items():
        if path.is_file():
            for line in open(path):
                out[json.loads(line)["id"]].append(name)
    return out


def eligible_rows(stats: Counter):
    lic = ImageLicences()
    bench = bench_image_shas()
    replay = load_replay_ids()
    with open(MANIFEST) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("images"):
                continue
            stats["image_rows"] += 1
            if r.get("partition") != "train":
                stats["drop_not_train"] += 1
                continue
            res = [lic.resolve(im) for im in r["images"]]
            if any(x is None for x in res):
                stats["drop_photo_licence"] += 1
                continue
            is_oi = any(x[1] == "openimages_v2" for x in res)
            if is_oi and r["source"] not in OPENIMAGES_SOURCES:
                stats["drop_photo_licence"] += 1        # Open Images outside state_grounded: not used (owner decision)
                continue
            shas = [Path(x[0]).stem for x in res]
            if any(s in bench for s in shas):
                stats["drop_bench_image"] += 1
                continue
            fields = r["request"]["fields"]
            if "targets" in r:
                golds = [(f, r["targets"].get(f["id"]), (r.get("abstention_causes") or {}).get(f["id"])) for f in fields
                         if f["id"] in r["targets"]]
            else:
                golds = [(fields[0], r.get("target"), r.get("abstention_cause"))] if len(fields) == 1 else []
            for fi, (fl, target, cause) in enumerate(golds):
                conv = convert_field(fl)
                if conv is None:
                    stats["drop_field_shape"] += 1
                    continue
                field, mapper = conv
                if target is None:
                    if cause not in ABSTENTION_MAP:
                        stats["drop_no_label"] += 1
                        continue
                    gold, unk = None, ABSTENTION_MAP[cause]
                else:
                    gold, unk = mapper(target), None
                    if gold == "__bad__":
                        stats["drop_gold_not_in_field"] += 1
                        continue
                yield {"row": r, "field": field, "gold": gold, "unk": unk, "cause": cause, "res": res,
                       "suffix": f"::field={fl['id']}" if len(golds) > 1 else "", "replay": replay.get(r["id"], []),
                       "openimages": is_oi,
                       "attribution": [lic.attribution.get(Path(x[0]).stem) for x in res] if is_oi else None}


def to_candidate(e: dict, idx: int) -> dict:
    r, res = e["row"], e["res"]
    lics = [x[2] for x in res]
    row_lic = r.get("license")
    text_lic = row_lic.get("spdx") if isinstance(row_lic, dict) else row_lic
    licence = max(lics + ([text_lic] if text_lic in LICENCE_RANK else []), key=LICENCE_RANK.index)
    state = r["request"].get("state")
    label_origin = r.get("pseudo_label") or "source_annotation"
    return {
        "id": f"p3-C-img-{idx:06d}",
        "source": "C",
        "dataset": r["source"],
        "family": r["family"],
        "difficulty": difficulty(r, len(res), e["gold"]),
        "state": state if state is not None else {},
        "images": [x[0] for x in res],
        "field": e["field"],
        "gold": e["gold"],
        "unknown_reason": e["unk"],
        "gold_kind": "constructed" if label_origin == "construction" else "dataset",
        "parent_id": None,
        "provenance": {
            "licence": licence,
            "photo_licence": max(lics, key=LICENCE_RANK.index),
            "image_licences": lics,
            "licence_evidence": sorted({x[3] for x in res}),
            "image_pools": [x[1] for x in res],
            "text_licence": text_lic,
            "upstream_dataset": r["source"],
            "upstream_id": r["id"] + e["suffix"],
            "upstream_split": "train",
            "source_group": r.get("source_group"),
            "image_sha256": [Path(x[0]).stem for x in res],
            "label_origin": label_origin,
            "original_abstention_cause": e["cause"],
            "teacher_confidence": r.get("teacher_confidence"),
            "image_role": r.get("image_role"),
            "template_id": r.get("template_id"),
            "pools": ["decision-v2.1-4b"] + e["replay"],
            **({"photo_attribution": e["attribution"],
                "annotation_licence": "CC-BY-4.0 (Open Images V7 human-verified image labels, Google LLC)",
                "licence_note": "Open Images lists its photos as CC BY 2.0 and disclaims per-image warranties; "
                                "attribution per photo from data/decision-v2/openimages_v2/attribution.jsonl"}
               if e.get("openimages") else {}),
        },
    }


def balance_key(e: dict) -> str:
    """Text decisions that carry an irrelevant photo (image_role == "irrelevant") share ONE bucket: they are
    the irrelevance gate's replay, not image decisions, and must not take a family share each."""
    return "irrelevant_image" if e["row"].get("image_role") == "irrelevant" else e["row"]["family"]


def balanced_sample(entries: list[dict], n: int) -> list[dict]:
    """Water-fill over original families (every family gets min(its size, cap)); inside a family, round-robin over
    datasets, so one source cannot fill a shared family (e.g. `material`)."""
    fam = defaultdict(list)
    for e in entries:
        fam[balance_key(e)].append(e)
    sizes = {f: len(v) for f, v in fam.items()}
    lo, hi = 0, max(sizes.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(s, mid) for s in sizes.values()) <= n:
            lo = mid
        else:
            hi = mid - 1
    cap = lo
    out = []
    for f in sorted(fam):
        by_ds = defaultdict(list)
        for e in fam[f]:
            by_ds[e["row"]["source"]].append(e)
        for ds in by_ds:
            by_ds[ds].sort(key=lambda e: e["row"]["id"] + e["suffix"])
            rng_for(SEED, f, ds).shuffle(by_ds[ds])
        take, i = [], 0
        order = sorted(by_ds)
        while len(take) < min(cap, sizes[f]):
            for ds in order:
                if i < len(by_ds[ds]) and len(take) < cap:
                    take.append(by_ds[ds][i])
            i += 1
        out += take
    # top up to n with the leftovers of the largest families, deterministically
    if len(out) < n:
        chosen = {id(e) for e in out}
        rest = [e for e in entries if id(e) not in chosen]
        rng_for(SEED, "topup").shuffle(rest)
        out += rest[: n - len(out)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--summary", default=None, help="optional path for a JSON summary")
    a = ap.parse_args()
    stats = Counter()
    every = list(eligible_rows(stats))
    entries = [e for e in every if not e["openimages"]]
    oi = sorted((e for e in every if e["openimages"]), key=lambda e: e["row"]["id"] + e["suffix"])
    stats["eligible_fields"] = len(entries)
    stats["openimages_state_grounded_fields"] = len(oi)
    picked = balanced_sample(entries, a.n)
    picked.sort(key=lambda e: (e["row"]["family"], e["row"]["id"] + e["suffix"]))
    # The 40k balanced sample is unchanged; the Open Images state-grounded rows are appended after it, all of them.
    rows = [to_candidate(e, i) for i, e in enumerate(picked + oi)]
    n = write(a.out, rows)
    summary = {"written": n, "stats": dict(stats),
               "eligible_by_dataset": dict(Counter(e["row"]["source"] for e in entries).most_common()),
               "by_family": dict(Counter(r["family"] for r in rows).most_common()),
               "by_dataset": dict(Counter(r["dataset"] for r in rows).most_common()),
               "by_licence": dict(Counter(r["provenance"]["licence"] for r in rows).most_common()),
               "unknown": sum(r["gold"] is None for r in rows),
               "two_image": sum(len(r["images"]) > 1 for r in rows),
               "in_p2c_replay": sum(len(r["provenance"]["pools"]) > 1 for r in rows),
               "openimages_state_grounded": sum("openimages_v2" in r["provenance"]["image_pools"] for r in rows)}
    if a.summary:
        Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
