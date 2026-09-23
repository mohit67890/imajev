"""Reference-vs-target two-image pairs for decision-v2.

Two sources come out of this script:

`pairs_grounded` -- COMPOSITED pairs, labelled by construction (no teacher).
    image 1 (reference) = a CC0 PD12M photograph already on disk, used by path, untouched;
    image 2 (target)    = the same photograph with ONE controlled edit (`edits.py`), written as a
                          new content-addressed JPEG under `data/decision-v2/pairs_grounded/
                          images_edited/`.
    70% of targets carry a relevant edit (an object added / removed, one object recoloured,
    something covered or blurred); 30% carry an irrelevant one (brightness, contrast, a small
    crop, a mirror, a tiny unrelated object in a corner) whose right answer is "no relevant
    change".  Because we made the edit, the answer is known: `target` is filled in and
    `pseudo_label` is "construction".
    Families: `what_changed` (choice), `target_still_matches_reference` (boolean),
    `which_state_field_now_wrong` (choice over the fields of a reference record in the state).

`pairs_natural` -- NATURAL pairs for the teacher (`target: null`, `pseudo_label: "pending"`).
    Two views of the same ABO listing, a different listing of the same product type, or two
    PD12M photographs from the same subject bucket; "same item?" and "which attribute differs?".

Why not inside `data/decision-v2/pd12m/`: the brief asked for `images_edited/` beside the source's
`images/`, but a GPU labelling run is reading `data/decision-v2/pd12m/` right now, so nothing is
written there.  Everything lives under `data/decision-v2/pairs_grounded/` and every edited image's
derivation (reference file, donor file and crop, edit parameters) is in `attribution.jsonl`.

Run:
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/build_pairs.py
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_image_records.py pairs_grounded
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_image_records.py pairs_natural
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from PIL import Image

from v2.common import OUT, RAW, ROOT, read_jsonl, repo_relative, stable_partition, verified_license, write_jsonl
from v2.state_grounded import edits as E
from v2.state_grounded import templates as T
from v2.templates import photo as P

SEED = 20260924
PARTITION_SEED = "decision-v2-photo"          # same as convert_photos / state_grounded
COMPOSITE = "pairs_grounded"
NATURAL = "pairs_natural"
EDIT_DIR = OUT / COMPOSITE / "images_edited"
N_COMPOSITE = 20_000
N_PROBE = 60
JPEG_QUALITY = 90

# 70% relevant / 30% irrelevant edits
# Planned weights, not final shares: a recolour succeeds on only ~1 photo in 4 (it needs one
# whole, coloured, in-focus object), and a failed edit falls back to another RELEVANT kind, so
# recolour is over-planned to land near its share.  The final mix is in build_summary.json.
EDIT_WEIGHTS = {"add_object": 18, "remove_object": 17, "recolour_region": 40, "hide_label": 12,
                "brightness": 7, "contrast": 7, "small_crop": 7, "mirror": 7, "corner_object": 7}
FAMILY_WEIGHTS = {"what_changed": 45, "target_still_matches_reference": 30,
                  "which_state_field_now_wrong": 25}

CHANGE_OPTIONS = {
    "add_object": ("an object was added", "image 2 shows something that is not in image 1"),
    "remove_object": ("an object was removed", "something in image 1 is missing from image 2"),
    "recolour_region": ("an object changed colour", "the same object is there, in a different colour"),
    "hide_label": ("something was covered or blurred out",
                   "part of image 1 is hidden, blurred or taped over in image 2"),
    "irrelevant": ("nothing relevant changed",
                   "only lighting, framing or mirroring differ -- or nothing at all"),
}
RULES = [
    "ignore changes of brightness, contrast, framing and left-right mirroring, and anything tiny at the very edge of the frame",
    "lighting, a slightly different crop, a mirrored shot or a speck in a corner do not count as changes",
    "only a change to what is in the scene counts; exposure, framing, mirroring and edge specks do not",
    "treat the two photos as matching unless an item was added, removed, recoloured or hidden",
]

WHAT_CHANGED_Q = [
    "Image 1 is the reference, image 2 the target. Following `{p}`, what changed?",
    "Compare image 2 with image 1 under the rules at `{p}`. What is different?",
    "Using `{p}` to decide what counts, what changed between the two photographs?",
    "`{p}` says what to ignore. What, if anything, changed from image 1 to image 2?",
    "Check the target (image 2) against the reference (image 1), applying `{p}`. What changed?",
    "Somebody has to log the change between these two photos. Going by `{p}`, which is it?",
    "Read `{p}`, then compare the photographs. What kind of change is there?",
    "What is the relevant difference between image 1 and image 2, as `{p}` defines relevant?",
    "Following the rules at `{p}`, classify the change from the reference to the target.",
    "Image 2 was taken after image 1. Under `{p}`, what happened in between?",
]
STILL_MATCHES_Q = [
    "Does image 2 still match the reference image 1, under the rules at `{p}`?",
    "Applying `{p}`, is the target (image 2) unchanged from the reference (image 1)?",
    "Going by `{p}`, would you say image 2 still shows the scene exactly as image 1 does?",
    "Answer yes or no: under `{p}`, nothing relevant differs between image 1 and image 2.",
    "Using `{p}` to decide what counts, do the two photographs still agree?",
    "Is image 2 an acceptable match for image 1, following `{p}`?",
    "Read `{p}`. Is the target photo still the same as the reference?",
    "Following `{p}`, can image 2 be signed off as unchanged from image 1?",
    "Under the rules in `{p}`, does the target still match the reference?",
    "Image 1 is the reference. Does image 2 still match it, as `{p}` defines a match?",
]
FIELD_WRONG_Q = [
    "`{r}` was true of the reference (image 1). Which of its fields is no longer true of image 2?",
    "Image 1 is the photo `{r}` was written from. Which field of `{r}` does image 2 now contradict?",
    "Compare image 2 with image 1. Which entry of `{r}` has stopped being true?",
    "Every field of `{r}` held for image 1. Which one fails for image 2?",
    "Check `{r}` against the new photo (image 2), using image 1 as the reference. Which field is now wrong?",
    "Somebody must update `{r}` after image 2 was taken. Which field needs changing?",
    "Which field of `{r}` does the target photo (image 2) prove wrong?",
    "Read `{r}`, compare the two photographs, and name the field that no longer holds.",
    "Going from image 1 to image 2, which recorded field of `{r}` became false?",
    "Point to the field of `{r}` that image 2 contradicts.",
]
FIELD_TEXT = {
    "all_items_still_there": "yes, everything in the reference is still in the frame",
    "no_new_items": "yes, nothing has been added",
    "colours_as_before": "yes, every object keeps its colour",
    "nothing_covered_or_blurred": "yes, no part of the scene is hidden or blurred",
}
RECORD_ROOTS = ["inspection", "site_check", "condition_record", "audit"]

# --------------------------------------------------------------------------- photos
ABO_SINGLE = re.compile(
    r"^(?:cat|color|material|pattern|item_shape|finish|style"
    r"|bool-(?:color|material|pattern|item_shape|category))-([A-Z0-9]{10})-")


def pd12m_photos():
    selection = {r["pd12m_id"]: r for r in read_jsonl(RAW / "pd12m" / "selection.jsonl")}
    spdx, rel = "CC0-1.0", "data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"
    licence = verified_license(Path(rel), spdx)
    out, seen = [], set()
    for row in read_jsonl(RAW / "pd12m" / "manifest.jsonl"):
        if row["sha256"] in seen or row["spdx"] != "CC0-1.0" or not (ROOT / row["image"]).is_file():
            continue
        seen.add(row["sha256"])
        extra = selection.get(row["pd12m_id"], {})
        group = f"pd12m:{row['pd12m_id']}"
        out.append({"key": row["pd12m_id"], "group": group,
                    "partition": stable_partition(group, seed=PARTITION_SEED),
                    "bucket": extra.get("bucket_label", ""), "licence": licence,
                    "image": {"image": repo_relative(row["image"]), "sha256": row["sha256"],
                              "width": row["width"], "height": row["height"]}})
    return out


# --------------------------------------------------------------------------- composite worker
def _open(rel):
    return Image.open(ROOT / rel).convert("RGB")


def make_target(job):
    """Worker: -> result dict, or {"failed": ...}.  Deterministic from the job's seed."""
    rng = random.Random(job["seed"])
    ref = _open(job["reference"]["image"])
    if not E.is_colour_photo(ref) and job["kind"] in ("recolour_region",):
        return {"failed": "monochrome", **job}
    donor = None
    if job.get("donor"):
        donor = E.cutout(_open(job["donor"]["image"]))
        if donor is None:
            return {"failed": "donor", **job}
    res = None
    attempts = 12 if job["kind"] == "recolour_region" else 5
    for attempt in range(attempts):  # a different region each try, same kind of edit
        res = E.apply_edit(ref, job["kind"], random.Random(f"{job['seed']}\0{attempt}"), donor=donor)
        if res is not None:
            break
    if res is None:
        return {"failed": "edit", **job}
    edited, meta = res
    swap = bool(meta.get("swap"))
    buf = io.BytesIO()
    edited.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    path = EDIT_DIR / f"{sha}.jpg"
    if not path.exists():
        path.write_bytes(data)
    new = {"image": repo_relative(path), "sha256": sha, "width": edited.width,
           "height": edited.height}
    if swap:     # remove_object: the pasted-into photo is the reference, the original the target
        return {**job, "meta": meta, "reference": new, "target": job["reference"],
                "original": job["reference"]}
    return {**job, "meta": meta, "target": new, "original": job["reference"]}


def plan_jobs(refs, donors_by_partition, n, seed, tag):
    """One job per reference: an edit kind, and a donor (same partition) when the edit pastes."""
    jobs = []
    rng = random.Random(f"{seed}\0{tag}")
    kinds = sorted(EDIT_WEIGHTS)
    for ref in refs[:n]:
        kind = rng.choices(kinds, weights=[EDIT_WEIGHTS[k] for k in kinds])[0]
        job = {"key": ref["key"], "group": ref["group"], "partition": ref["partition"],
               "bucket": ref["bucket"], "reference": ref["image"], "kind": kind,
               "seed": f"{seed}\0{tag}\0{ref['key']}\0{kind}"}
        if kind in E.PASTE_KINDS:
            pool = donors_by_partition[ref["partition"]]
            donor = pool[rng.randrange(len(pool))]
            job["donor"] = {"key": donor["key"], "image": donor["image"]["image"],
                            "sha256": donor["image"]["sha256"]}
        jobs.append(job)
    return jobs


def run_jobs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(make_target, jobs, chunksize=16))


def fallback_kinds(kind, seed):
    """When an edit cannot be made honestly on this photo, try another of the same polarity, in a
    per-photo random order so no single kind soaks up every failure."""
    pool = list(E.RELEVANT_KINDS) if kind in E.RELEVANT_KINDS else \
        ["brightness", "contrast", "small_crop", "mirror"]
    pool = [k for k in pool if k != kind]
    random.Random(seed).shuffle(pool)
    return pool


# --------------------------------------------------------------------------- questions
def _rules_state(rng, extra=None):
    rule = rng.choice(RULES)
    shape = rng.randrange(4)
    if shape == 0:
        state, path = {"rules": rule}, "rules"
    elif shape == 1:
        state, path = {"comparison": {"reference": "image 1", "target": "image 2", "ignore": rule}}, \
            "comparison.ignore"
    elif shape == 2:
        state, path = {"task": {"id": T._ref(rng), "what_counts": rule}}, "task.what_counts"
    else:
        state, path = {"policy": {"name": "change check", "rule": rule}}, "policy.rule"
    if extra:
        state.update(extra)
    return state, path


def build_question(res, family, rng):
    """-> (request fields, state, gold, template_id)."""
    kind = res["meta"]["kind"]
    relevant = kind in E.RELEVANT_KINDS
    change = kind if relevant else "irrelevant"
    if family == "what_changed":
        state, path = _rules_state(rng)
        i, template = T._pick(rng, WHAT_CHANGED_Q)
        criteria = {v: d for v, d in CHANGE_OPTIONS.values()}
        field = T._field("choice", T._fmt(template, path),
                         options=T._options([v for v, _ in CHANGE_OPTIONS.values()], rng, criteria))
        return [field], state, CHANGE_OPTIONS[change][0], f"what_changed/{i:02d}"
    if family == "target_still_matches_reference":
        state, path = _rules_state(rng)
        i, template = T._pick(rng, STILL_MATCHES_Q)
        field = T._maybe_bool_criteria(
            T._field("boolean", T._fmt(template, path)), rng,
            ["nothing relevant differs between the two photographs"],
            ["an item was added, removed, recoloured or hidden in image 2"])
        return [field], state, (not relevant), f"target_still_matches_reference/{i:02d}"
    # which_state_field_now_wrong: 3-5 fields true of the reference, at most one falsified
    names = list(E.FALSIFIES.values())
    wrong = E.FALSIFIES.get(kind)
    rng.shuffle(names)
    keep = names[:rng.randint(3, 4)]
    if wrong and wrong not in keep:
        keep[-1] = wrong
    root = rng.choice(RECORD_ROOTS)
    record = {}
    subject = P.CATEGORY_OBJECT.get(res.get("bucket", ""))
    if subject and rng.random() < 0.6:
        record["subject"] = subject                 # context only, never an option
    for k in keep:
        record[k] = FIELD_TEXT[k]
    if rng.random() < 0.4:
        record["checked_by"] = rng.choice(T.PEOPLE)
    state = {root: record}
    rule_state, rule_path = _rules_state(rng)
    state.update(rule_state)
    i, template = T._pick(rng, FIELD_WRONG_Q)
    values = [f"{root}.{k}" for k in keep] + [T.NONE_OF_THESE]
    criteria = {f"{root}.{k}": f"image 2 shows that '{k.replace('_', ' ')}' no longer holds"
                for k in keep}
    criteria[T.NONE_OF_THESE] = f"every field still holds for image 2, as `{rule_path}` defines a change"
    field = T._field("choice", T._fmt(template.replace("{r}", "{p}"), root) +
                     f" Changes that `{rule_path}` says to ignore do not count.",
                     options=T._options(values, rng, criteria))
    gold = f"{root}.{wrong}" if wrong else T.NONE_OF_THESE
    return [field], state, gold, f"which_state_field_now_wrong/{i:02d}"


def composite_record(res, family, index, licence, rng, source=COMPOSITE):
    fields, state, gold, tid = build_question(res, family, rng)
    rec = {
        "id": f"{source}:{res['key']}#{index}",
        "source": source,
        "source_group": res["group"],
        "partition": res["partition"],
        "family": family,
        "source_split": "crawl",
        "license": licence,
        "images": [res["reference"], res["target"]],
        "request": {"schema_version": "1.0",
                    "request_id": f"{source}-{res['key']}-{index}"[:128],
                    "state": state, "fields": fields},
        "target": gold,
        "abstention_cause": None,
        "source_answer": f"edit={res['meta']['kind']}",
        "pseudo_label": "construction",
        "template_id": tid,
        "edit_kind": res["meta"]["kind"],
        "edit_relevant": res["meta"]["kind"] in E.RELEVANT_KINDS,
        "edit_meta": res["meta"],
    }
    if res.get("donor"):
        rec["donor"] = res["donor"]
    return rec


def choose_family(res, rng):
    fams = dict(FAMILY_WEIGHTS)
    if res["meta"]["kind"] == "corner_object":
        fams.pop("what_changed")     # an added speck would make "an object was added" arguable
    keys = sorted(fams)
    return rng.choices(keys, weights=[fams[k] for k in keys])[0]


# --------------------------------------------------------------------------- natural pairs
SAME_ITEM_Q = [
    "Image 1 is the listing photo and image 2 the item that arrived. Is it the same item?",
    "Do these two photographs show the very same product?",
    "Is the item in image 2 the same item as in image 1 (not just the same kind of thing)?",
    "Answer yes or no: image 1 and image 2 show one and the same item.",
    "Compare the two photos. Same item, or two different items?",
    "Somebody claims image 2 shows the item from image 1. Is that right?",
    "Is image 2 another view of the exact item in image 1?",
    "Would you accept image 2 as a photo of the same item as image 1?",
]
ATTR_DIFF_Q = [
    "Which attribute differs between the item in image 1 and the item in image 2?",
    "Compare the two items. In what respect do they differ?",
    "Image 1 is the reference. What is different about the item in image 2?",
    "Pick the attribute that is not the same in the two photographs.",
    "What distinguishes the item in image 2 from the one in image 1?",
    "Somebody has to note how image 2's item differs from image 1's. Which attribute?",
    "Looking at both photos, which property of the item changed?",
    "Which of these differs between the two pictured items?",
]
ATTR_OPTIONS = [("colour", "the items differ in colour"), ("material", "they are made of different materials"),
                ("shape", "their shape or outline differs"),
                ("none -- they match", "no visible difference in colour, material or shape")]


def natural_pairs(seed, n_abo=6000, n_pd=4000):
    rng = random.Random(f"{seed}\0natural")
    abo_lic = verified_license(Path("data/decision-v2/licenses/state_grounded/abo-LICENSE-CC-BY-4.0.txt"), "CC-BY-4.0")
    pd_lic = verified_license(Path("data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"), "CC0-1.0")
    meta = json.loads((ROOT / ".cache/datasets/v1/state_aware/abo_listing_meta.json").read_text()) \
        if (ROOT / ".cache/datasets/v1/state_aware/abo_listing_meta.json").is_file() else {}
    images, nontrain = defaultdict(dict), set()
    for row in read_jsonl(ROOT / "data" / "decision-v1" / "abo" / "records.jsonl"):
        hit = ABO_SINGLE.match(row["id"].split(":", 1)[1])
        if not hit:
            continue
        item = hit.group(1)
        if row["partition"] != "train":
            nontrain.add(item)
            continue
        im = row["images"][0]
        images[item][im["sha256"]] = im
    items = sorted(k for k, v in images.items() if len(v) >= 2 and k not in nontrain)
    rng.shuffle(items)
    by_type_part = defaultdict(list)
    part_of = {}
    for item in items:
        group = f"abo:{item}"
        part_of[item] = stable_partition(group, seed=PARTITION_SEED)
        pt = (meta.get(item) or {}).get("pt", "")
        by_type_part[(pt, part_of[item])].append(item)
    rows = []
    for item in items:
        if len(rows) >= n_abo:
            break
        views = sorted(images[item].values(), key=lambda x: x["sha256"])
        same = rng.random() < 0.5
        if same:
            first, second = views[0], views[1]
            group = f"abo:{item}"
            partner = None
        else:
            pt = (meta.get(item) or {}).get("pt", "")
            pool = [x for x in by_type_part[(pt, part_of[item])] if x != item]
            if not pool:
                continue
            partner = rng.choice(pool)
            first = views[0]
            second = sorted(images[partner].values(), key=lambda x: x["sha256"])[0]
            group = f"abo:{item}+abo:{partner}"
        family = "natural_same_item" if rng.random() < 0.6 else "natural_attribute_differs"
        rows.append(natural_record(item, [first, second], group, part_of[item], family, rng,
                                   abo_lic, "abo_train_photos", len(rows), partner=partner,
                                   same=same))
    # PD12M: two photographs from the same subject bucket (always different photos)
    photos = pd12m_photos()
    rng.shuffle(photos)
    by_bucket = defaultdict(list)
    for p in photos:
        by_bucket[(p["bucket"], p["partition"])].append(p)
    count = 0
    for p in photos:
        if count >= n_pd:
            break
        pool = [q for q in by_bucket[(p["bucket"], p["partition"])] if q["key"] != p["key"]]
        if not pool:
            continue
        q = rng.choice(pool)
        family = "natural_same_item" if rng.random() < 0.5 else "natural_attribute_differs"
        rows.append(natural_record(p["key"], [p["image"], q["image"]],
                                   f"{p['group']}+{q['group']}", p["partition"], family, rng,
                                   pd_lic, "crawl", len(rows), partner=q["key"], same=False,
                                   prefix="pd12m"))
        count += 1
    return rows


def natural_record(key, images, group, partition, family, rng, licence, split, index,
                   partner=None, same=None, prefix="abo"):
    if family == "natural_same_item":
        i, q = T._pick(rng, SAME_ITEM_Q)
        field = T._maybe_bool_criteria(T._field("boolean", q), rng,
                                       ["both photographs show one and the same physical item"],
                                       ["the photographs show two different items, even if alike"])
        tid = f"natural_same_item/{i:02d}"
        state = {} if rng.random() < 0.5 else {"return": {"reference": T._ref(rng),
                                                          "claim": "image 2 is the item from image 1"}}
    else:
        i, q = T._pick(rng, ATTR_DIFF_Q)
        field = T._field("choice", q, options=T._options([v for v, _ in ATTR_OPTIONS], rng,
                                                          dict(ATTR_OPTIONS)))
        tid = f"natural_attribute_differs/{i:02d}"
        state = {} if rng.random() < 0.6 else {"task": {"id": T._ref(rng),
                                                        "note": "the reference is image 1"}}
    rec = {
        "id": f"{NATURAL}:{prefix}-{key}#{index}", "source": NATURAL, "source_group": group,
        "partition": partition, "family": family, "source_split": split, "license": licence,
        "images": images,
        "request": {"schema_version": "1.0", "request_id": f"{NATURAL}-{prefix}-{key}-{index}"[:128],
                    "state": state, "fields": [field]},
        "target": None, "abstention_cause": None, "source_answer": None,
        "pseudo_label": "pending", "template_id": tid, "photo_source": prefix,
        "pair_kind": "same_listing_two_views" if same else
        ("same_product_type" if prefix == "abo" else "same_bucket"),
    }
    if partner:
        rec["partner"] = partner
    if partition == "test":
        rec["pseudo_label_test"] = True
    return rec


# --------------------------------------------------------------------------- main
def main(seed=SEED, n=N_COMPOSITE, workers=None, probe_only=False):
    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    photos = pd12m_photos()
    random.Random(f"{seed}\0photos").shuffle(photos)

    # donors: photographs on a plain backdrop (museum objects, product shots), found by trying
    # the cut-out on a slice of the pool.  A donor is never also a reference.
    donor_scan = photos[:12_000]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(_is_donor, [p["image"]["image"] for p in donor_scan], chunksize=32))
    donors = [p for p, good in zip(donor_scan, ok) if good]
    donor_keys = {d["key"] for d in donors}
    donors_by_partition = defaultdict(list)
    for d in donors:
        donors_by_partition[d["partition"]].append(d)
    for part in ("train", "dev", "test"):
        if not donors_by_partition[part]:
            donors_by_partition[part] = donors
    print(f"donors: {len(donors)} ({dict(Counter(d['partition'] for d in donors))})", flush=True)

    refs = [p for p in photos[12_000:] if p["key"] not in donor_keys]
    # colour photographs only: every edit kind must be possible on every reference
    with ProcessPoolExecutor(max_workers=workers) as pool:
        colour = list(pool.map(_is_colour, [p["image"]["image"] for p in refs[:int(n * 1.6) + 400]],
                               chunksize=32))
    refs = [p for p, c in zip(refs, colour) if c]
    probe_refs = [p for p in refs if p["partition"] == "test"][:N_PROBE * 3]
    probe_keys = {p["key"] for p in probe_refs}
    refs = [p for p in refs if p["key"] not in probe_keys]
    print(f"colour references: {len(refs)} (+{len(probe_refs)} probe candidates)", flush=True)

    summary = {}
    if not probe_only:
        results = composite_pass(refs, donors_by_partition, n, seed, "main", workers)
        rows, attribution = [], []
        licence = refs[0]["licence"]
        for k, res in enumerate(results):
            rng = random.Random(f"{seed}\0q\0{res['key']}")
            family = choose_family(res, rng)
            rows.append(composite_record(res, family, 0, licence, rng))
            attribution.append(attribution_row(res))
        rows.sort(key=lambda r: r["id"])
        write_jsonl(OUT / COMPOSITE / "records.jsonl", rows)
        write_jsonl(OUT / COMPOSITE / "attribution.jsonl", attribution)
        summary[COMPOSITE] = summarise(rows)
        natural = natural_pairs(seed)
        natural.sort(key=lambda r: r["id"])
        write_jsonl(OUT / NATURAL / "records.jsonl", natural)
        summary[NATURAL] = summarise(natural)

    # probe candidates: built exactly the same way, from held-out references never in records
    probe = composite_pass(probe_refs, donors_by_partition, len(probe_refs), seed, "probe", workers,
                           balanced=True)
    write_jsonl(OUT / COMPOSITE / "probe_candidates.jsonl", probe)
    summary["probe_candidates"] = len(probe)
    (OUT / COMPOSITE / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _is_donor(rel):
    try:
        return E.cutout(_open(rel)) is not None
    except Exception:
        return False


def _is_colour(rel):
    try:
        return E.is_colour_photo(_open(rel))
    except Exception:
        return False


def composite_pass(refs, donors_by_partition, n, seed, tag, workers, balanced=False):
    jobs = plan_jobs(refs, donors_by_partition, len(refs) if balanced else n, seed, tag)
    if balanced:   # the probe wants every edit kind represented evenly
        kinds = sorted(EDIT_WEIGHTS)
        for j, job in enumerate(jobs):
            job["kind"] = kinds[j % len(kinds)]
            job["seed"] = f"{seed}\0{tag}\0{job['key']}\0{job['kind']}"
            if job["kind"] in E.PASTE_KINDS and "donor" not in job:
                pool = donors_by_partition[job["partition"]]
                d = pool[random.Random(job["seed"]).randrange(len(pool))]
                job["donor"] = {"key": d["key"], "image": d["image"]["image"],
                                "sha256": d["image"]["sha256"]}
            elif job["kind"] not in E.PASTE_KINDS:
                job.pop("donor", None)
    done = run_jobs(jobs, workers)
    good = [r for r in done if "failed" not in r]
    failed = [r for r in done if "failed" in r]
    first_failures = len(failed)
    for round_ in range(2):
        retry = []
        for r in failed:
            alts = fallback_kinds(r["kind"], r["seed"])
            alt = alts[round_ % len(alts)]
            job = {k: v for k, v in r.items() if k != "failed"}
            job["kind"] = alt
            job["seed"] = f"{job['seed']}\0{alt}"
            if alt in E.PASTE_KINDS:
                pool = donors_by_partition[job["partition"]]
                d = pool[random.Random(job["seed"]).randrange(len(pool))]
                job["donor"] = {"key": d["key"], "image": d["image"]["image"],
                                "sha256": d["image"]["sha256"]}
            else:
                job.pop("donor", None)
            retry.append(job)
        second = run_jobs(retry, workers) if retry else []
        good += [r for r in second if "failed" not in r]
        failed = [r for r in second if "failed" in r]
    done, second = [None] * first_failures, failed
    print(f"[{tag}] {len(good)} targets ({first_failures} first-try failures, "
          f"{len(failed)} dropped)", flush=True)
    return good[:n]


def attribution_row(res):
    """One line per edited image: what it was derived from, and how."""
    swap = bool(res["meta"].get("swap"))
    edited = res["reference"] if swap else res["target"]
    row = {"edited_image": edited["image"], "edited_sha256": edited["sha256"],
           "edited_role": "reference (image 1)" if swap else "target (image 2)",
           "original_image": res["original"]["image"], "original_sha256": res["original"]["sha256"],
           "original_pd12m_id": res["key"], "original_licence": "CC0-1.0",
           "edit": res["meta"], "tool": "scripts/v2/state_grounded/edits.py",
           "note": "derivative of a CC0-1.0 photograph (PD12M); CC0 places no condition on "
                   "derivatives, and this derivative is likewise released under CC0-1.0"}
    if res.get("donor"):
        row.update({"donor_pd12m_id": res["donor"]["key"], "donor_image": res["donor"]["image"],
                    "donor_sha256": res["donor"]["sha256"], "donor_licence": "CC0-1.0",
                    "donor_use": "subject cut out against its plain backdrop (edits.cutout)"})
    return row


def summarise(rows):
    first = lambda r: r["request"]["fields"][0]
    return {
        "records": len(rows),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "families": dict(Counter(r["family"] for r in rows)),
        "edit_kinds": dict(Counter(r.get("edit_kind") for r in rows if r.get("edit_kind"))),
        "relevant_share": round(sum(bool(r.get("edit_relevant")) for r in rows) / max(1, len(rows)), 4),
        "targets": dict(Counter(str(r["target"]) for r in rows).most_common(12)),
        "field_types": dict(Counter(first(r)["type"] for r in rows)),
        "pair_kinds": dict(Counter(r.get("pair_kind") for r in rows if r.get("pair_kind"))),
        "distinct_templates": len({r["template_id"] for r in rows}),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_COMPOSITE)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--probe-only", action="store_true")
    a = ap.parse_args()
    main(a.seed, a.n, a.workers, a.probe_only)
