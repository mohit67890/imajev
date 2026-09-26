"""Phase-3 Stage 0-I: safety-inspection items (families `safety_*`, source I).

Two parts, both with gold that needs no teacher:

(a) Open Images photos (dataset `openimages_safety`, gold_kind "dataset"). Only the 20k Open Images V7 test-split photos
    already on disk under our licence rule (CC BY 2.0, per-photo Flickr attribution in
    data/decision-v2/openimages_v2/attribution.jsonl; annotations CC BY 4.0). Questions are answerable from the
    human-verified annotations only:
      safety_presence   (noul)   verified-positive image-level label (with a non-depiction box when the class is boxable)
                                 vs verified-negative label (and no positive / box of the class or any sub-class)
      safety_count      (choice) number of boxes of one class, only when every box of that class is IsGroupOf=0,
                                 IsDepiction=0, not tiny, and no box of a parent / child class is present
      safety_position   (choice) left / middle / right third for a single-instance class (box centre clear of the edges)
      safety_checklist  (noul)   an inspection rule over two verified classes ("flag if X or Y is visible")
      safety_headcount  (noul)   a minimum-count rule over complete box counts
      safety_not_visible (noul, unknown) asks for a property no photo can settle (rating, certification, date)
    Never a person-helmet association or anything else the boxes do not give.

(b) Rendered inspection scenes (dataset `safety_synthetic`, gold_kind "constructed"): our own PIL drawings of floor plans
    and stylised warehouse views with blocked exits, spill markers, empty extinguisher stations, obstructed forklift
    lanes, PPE on figures (hard hat / vest / goggles), missing guardrail sections, and hazard signs checked against a
    checklist in the state (joint). ~15% unknown: part of the plan / site out of frame, a worker behind a stack of
    boxes, an illegible sign; or a "which X" question with no such X (false_premise). `settle` proves every unknown by
    an alternative content of the hidden part that changes the answer.

    .venv/bin/python scripts/p3/gen_safety.py [--count 3000] [--seed p3-safety-v1]
    .venv/bin/python scripts/p3/gen_safety.py --heldout --count 250
    .venv/bin/python scripts/p3/gen_safety.py --variant-of parents.jsonl --per-parent 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402
from convert_common import option_key  # noqa: E402
from inventory_assets.common import (affine_params, apply_post, encode_jpeg, fit_font, font, map_box,  # noqa: E402
                                     rng_for, sha256_bytes, store_image, text_center, forward_matrix)

SEED = "p3-safety-v1"
HELDOUT_SEED = "p3-safety-heldout-fresh-v1"
IMG_DIR = ROOT / "data" / "p3" / "images" / "safety"
OUT = ROOT / "data" / "p3" / "candidates" / "I-safety.jsonl"
HELDOUT_OUT = ROOT / "data" / "p3" / "pool" / "heldout-fresh-safety.jsonl"
OI_RAW = ROOT / "data" / "decision-v2-raw" / "openimages_v2"
OI_MANIFEST = OI_RAW / "manifest.jsonl"
OI_ATTR = ROOT / "data" / "decision-v2" / "openimages_v2" / "attribution.jsonl"
OI_LABELS = OI_RAW / "oidv7-test-annotations-human-imagelabels.csv"
OI_BOXES = OI_RAW / "test-annotations-bbox.csv"
OI_CLASSES = OI_RAW / "oidv7-class-descriptions.csv"
OI_SOURCES = ROOT / "data" / "p3" / "raw" / "openimages" / "SOURCES.json"
OI_EVIDENCE = "data/decision-v2/licenses/openimages_v2/CC-BY-2.0.deed.html"
OI_ANNOT_EVIDENCE = "data/decision-v2/licenses/openimages_v2/CC-BY-4.0.deed.html"
C_IMAGES = ROOT / "data" / "p3" / "candidates" / "C-images.jsonl"
OI_URLS = {
    "labels": "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv",
    "boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
}


def pick(rng, bank, split):
    k = max(1, len(bank) // 3)
    return rng.choice(bank[-k:] if split == "heldout" else bank[:-k])


# ================================================================================================= (a) Open Images
# class -> (singular phrase, plural, boxable-countable)
OI_CLASS = {
    "Helmet": ("a helmet", "helmets", True), "Ladder": ("a ladder", "ladders", True),
    "Traffic sign": ("a traffic sign", "traffic signs", True), "Stop sign": ("a stop sign", "stop signs", True),
    "Fire hydrant": ("a fire hydrant", "fire hydrants", True), "Stairs": ("a stairway", "stairways", False),
    "Door": ("a door", "doors", True), "Glove": ("a glove", "gloves", True), "Goggles": ("goggles", "pairs of goggles", True),
    "Person": ("a person", "people", True), "Car": ("a car", "cars", True), "Truck": ("a truck", "trucks", True),
    "Bicycle": ("a bicycle", "bicycles", True), "Motorcycle": ("a motorcycle", "motorcycles", True),
    "Bus": ("a bus", "buses", True), "Window": ("a window", "windows", True), "Chair": ("a chair", "chairs", True),
    "Table": ("a table", "tables", True), "Box": ("a box or carton", "boxes", True), "Barrel": ("a barrel", "barrels", True),
    "Street light": ("a street light", "street lights", True), "Wheelchair": ("a wheelchair", "wheelchairs", True),
    "Boot": ("a boot", "boots", True), "Sunglasses": ("sunglasses", "pairs of sunglasses", True),
    "Bench": ("a bench", "benches", True), "Fire extinguisher": ("a fire extinguisher", "fire extinguishers", False),
    "Traffic cone": ("a traffic cone", "traffic cones", False), "Handrail": ("a handrail", "handrails", False),
    "Scaffolding": ("scaffolding", "scaffolds", False), "Fire": ("fire or flames", "fires", False),
    "Ambulance": ("an ambulance", "ambulances", True),
}
# "Vehicle", "Building" and "Tool" are left out: their verified negatives follow a narrower reading than everyday English
# (a boat is not a "Vehicle" in the labels), so a yes/no question on them would be ambiguous.
# child classes: a verified negative of the parent is dropped when a child is present, counts skip related boxes
OI_CHILDREN = {
    "Person": ["Man", "Woman", "Boy", "Girl", "Human face", "Human body", "Human head"],
    "Helmet": ["Bicycle helmet", "Football helmet"],
    "Vehicle": ["Car", "Truck", "Bus", "Motorcycle", "Ambulance", "Van", "Taxi", "Limousine", "Land vehicle", "Tank"],
    "Car": ["Taxi", "Limousine", "Van"], "Tool": ["Hammer", "Drill", "Screwdriver", "Chisel", "Wrench", "Saw", "Chainsaw"],
    "Building": ["House", "Skyscraper", "Tower", "Office building", "Castle"], "Door": ["Door handle"],
    "Boot": [], "Glove": ["Baseball glove"], "Table": ["Kitchen & dining room table", "Coffee table", "Desk"],
    "Chair": [], "Box": [], "Window": ["Window blind"],
}
OI_PARENTS = {"Man": ["Person"], "Woman": ["Person"], "Boy": ["Person"], "Girl": ["Person"],
              "Bicycle helmet": ["Helmet"], "Football helmet": ["Helmet"], "Car": ["Vehicle"], "Truck": ["Vehicle"],
              "Bus": ["Vehicle"], "Motorcycle": ["Vehicle"], "Ambulance": ["Vehicle"], "Taxi": ["Car", "Vehicle"],
              "Van": ["Car", "Vehicle"]}
MIN_BOX_AREA = 0.002
MAX_COUNT = 8

OI_PHOTO_LINES = [
    "Inspection photo {ref} from the {site} walk-round.",
    "Photo {ref}, uploaded by the {site} safety officer.",
    "Site record {ref}: picture taken during the {site} audit.",
    "Evidence photo {ref} attached to the {site} inspection report.",
    "Field photo {ref} from the {site} survey.",
    "Picture {ref} logged for the {site} check.",
]
SITES = ["depot", "car park", "north gate", "loading yard", "street works", "warehouse", "campus", "harbour", "plant room",
         "canteen", "workshop", "retail park", "station forecourt", "office block", "construction site", "garden centre"]
Q_PRESENCE = [
    "Is there {a} anywhere in this photo?",
    "Does the photo show {a}?",
    "Can {a} be seen in the picture?",
    "Is {a} visible in this inspection photo?",
    "Does this picture contain {a}?",
    "Would the inspector find {a} in this image?",
]
Q_OI_COUNT = [
    "How many {plural} are in the photo?",
    "Count the {plural} visible in this picture.",
    "What is the number of {plural} shown?",
    "How many {plural} can be seen in this inspection photo?",
    "In this image, how many {plural} are there?",
    "What count of {plural} does the photo show?",
]
Q_POSITION = [
    "In which part of the photo is the {noun}: left, middle or right?",
    "Where does the {noun} appear across the picture (left third, middle, right third)?",
    "Is the {noun} on the left, in the middle or on the right of the image?",
    "Which third of the photo, from left to right, holds the {noun}?",
    "Horizontally, where in the frame is the {noun}?",
    "Looking across the image, which side is the {noun} on?",
]
OI_RULES = [
    "Flag the photo when it shows {a} or {b}; otherwise mark it clear.",
    "The photo must be escalated if {a} or {b} is visible in it.",
    "Escalate on sight of either {a} or {b}; a photo with neither is clear.",
    "Rule: a photo that contains {a}, or contains {b}, goes to the review queue.",
    "Anything showing {a} or {b} is sent for review; the rest is signed off.",
    "Mark the photo for follow-up if you can see {a} or {b} in it.",
]
Q_RULE = [
    "Does this photo have to be flagged under the rule?",
    "Under the rule, is this photo escalated?",
    "Does the rule send this photo for review?",
    "Is this photo flagged according to the rule?",
    "Must the inspector escalate this photo?",
    "Does this picture go to the review queue?",
]
HEAD_RULES = [
    "The photo is accepted as evidence only if it shows at least {k} {plural}.",
    "Evidence rule: at least {k} {plural} must be visible, otherwise the photo is rejected.",
    "A photo counts for the tally when {k} or more {plural} appear in it.",
    "Reject the photo unless it shows {k} {plural} or more.",
    "Accept only photos with a minimum of {k} {plural} in view.",
    "The count check passes when the picture contains no fewer than {k} {plural}.",
]
Q_HEAD = [
    "Does this photo pass the rule?",
    "Is the photo accepted under the evidence rule?",
    "Does the picture meet the minimum in the rule?",
    "Under the rule, is this photo accepted?",
    "Does the count check pass for this photo?",
    "Is this photo good enough under the rule?",
]
Q_OI_HIDDEN = [
    "Is the {noun} in this photo certified to the current safety standard?",
    "Was the {noun} in this photo inspected within the last 30 days?",
    "Is the {noun} shown here rated for outdoor use at night?",
    "Has the {noun} in the picture been reported as faulty before?",
    "Does the {noun} in this photo belong to the site owner rather than a contractor?",
    "Was this photo of the {noun} taken during the morning shift?",
]


def oi_bucket(image_id: str) -> int:
    return int(hashlib.sha256(("safety-split\0" + image_id).encode()).hexdigest()[:8], 16) % 10


def write_sources():
    """Record the Open Images annotation files used (already on disk from the v2 collection)."""
    OI_SOURCES.parent.mkdir(parents=True, exist_ok=True)
    out = {"dataset": "Open Images V7 (test split)", "photos": "20k CC BY 2.0 photos already on disk "
           "(data/decision-v2/openimages_v2/images, Flickr attribution per photo in attribution.jsonl)",
           "annotation_licence": "CC BY 4.0 (Google LLC)", "photo_licence": "CC BY 2.0 (per-photo Flickr author)",
           "download_page": "https://storage.googleapis.com/openimages/web/download_v7.html", "files": {}}
    for k, p in (("labels", OI_LABELS), ("boxes", OI_BOXES), ("classes", OI_CLASSES)):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out["files"][k] = {"path": str(p.relative_to(ROOT)), "source_url": OI_URLS[k], "sha256": h.hexdigest(),
                           "bytes": p.stat().st_size,
                           "version": "V7 human-verified image-level labels" if k == "labels" else
                           ("V5 test boxes (unchanged in V6/V7)" if k == "boxes" else "V7 class descriptions")}
    OI_SOURCES.write_text(json.dumps(out, indent=1) + "\n")


_OI = None


def load_oi():
    """Per-photo facts: verified present / absent classes and complete box lists."""
    global _OI
    if _OI is not None:
        return _OI
    import gen_image_joint as J
    bench = J.bench_image_shas()
    names = {}
    for row in csv.reader(open(OI_CLASSES)):
        if len(row) >= 2:
            names[row[0]] = row[1]
    want = set(OI_CLASS) | {c for v in OI_CHILDREN.values() for c in v} | set(OI_PARENTS)
    mids = {m for m, n in names.items() if n in want}
    photos = {}
    for l in open(OI_MANIFEST):
        r = json.loads(l)
        if r.get("spdx") != "CC-BY-2.0" or r["sha256"] in bench or not (ROOT / r["image"]).is_file():
            continue
        photos[r["image_id"]] = {"image_id": r["image_id"], "image": r["image"][5:], "sha256": r["sha256"],
                                 "width": r["width"], "height": r["height"], "pos": set(), "neg": set(),
                                 "boxes": defaultdict(list)}
    attr = {}
    for l in open(OI_ATTR):
        a = json.loads(l)
        attr[a["sha256"]] = {k: a.get(k) for k in ("image_id", "author", "author_profile", "title", "license",
                                                    "license_url", "flickr_landing_url", "source_url")}
    with open(OI_LABELS) as fh:
        rd = csv.reader(fh)
        next(rd)
        for iid, src, lab, conf in rd:
            if lab in mids and iid in photos:
                (photos[iid]["pos"] if conf == "1" else photos[iid]["neg"]).add(names[lab])
    with open(OI_BOXES) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            iid, lab = row[0], row[2]
            if lab in mids and iid in photos:
                x0, x1, y0, y1 = map(float, row[4:8])
                photos[iid]["boxes"][names[lab]].append(
                    {"box": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)], "occluded": row[8] == "1",
                     "truncated": row[9] == "1", "group": row[10] == "1", "depiction": row[11] == "1"})
    used_c = set()
    if C_IMAGES.is_file():
        for l in open(C_IMAGES):
            for im in json.loads(l)["images"]:
                if "openimages" in im:
                    used_c.add(Path(im).stem)
    for p in photos.values():
        p["attribution"] = attr.get(p["sha256"])
        p["in_c_images"] = p["sha256"] in used_c
        p["split"] = "heldout" if (oi_bucket(p["image_id"]) == 0 and not p["in_c_images"]) else (
            "train" if oi_bucket(p["image_id"]) != 0 else "none")
    _OI = {k: v for k, v in photos.items() if v["attribution"]}
    return _OI


def related(cls):
    return set(OI_CHILDREN.get(cls, [])) | set(OI_PARENTS.get(cls, []))


def oi_presence(p, cls):
    """True / False when the annotations settle it, else None."""
    boxes = p["boxes"].get(cls, [])
    kids = OI_CHILDREN.get(cls, [])
    if cls in p["pos"] or boxes:
        if boxes and all(b["depiction"] for b in boxes):
            return None                       # only a picture of the object (poster, toy, drawing)
        return True
    if cls in p["neg"]:
        if any(k in p["pos"] or p["boxes"].get(k) for k in kids):
            return None
        return False
    return None


def oi_count(p, cls):
    """Complete box count or None."""
    if not OI_CLASS[cls][2]:
        return None
    boxes = p["boxes"].get(cls, [])
    if any(p["boxes"].get(r) for r in related(cls)):
        return None
    if not boxes:
        return 0 if oi_presence(p, cls) is False else None
    if any(b["group"] or b["depiction"] for b in boxes):
        return None
    if any((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) < MIN_BOX_AREA for b in boxes):
        return None
    for i, a in enumerate(boxes):          # near-duplicate boxes (double annotation) make the count unreliable
        for b in boxes[i + 1:]:
            if iou(a["box"], b["box"]) > 0.7:
                return None
    n = len(boxes)
    return n if n <= MAX_COUNT else None


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def oi_position(p, cls):
    n = oi_count(p, cls)
    if n != 1:
        return None
    b = p["boxes"][cls][0]["box"]
    cx = (b[0] + b[2]) / 2
    w = b[2] - b[0]
    if w > 0.3:
        return None
    if b[2] < 0.36:
        return "left"
    if b[0] > 0.64:
        return "right"
    if b[0] > 0.3 and b[2] < 0.7 and 0.4 <= cx <= 0.6:
        return "middle"
    return None


def oi_facts(p) -> dict:
    """The annotation facts one item relies on (stored in provenance so tests re-derive the gold)."""
    return {"pos": sorted(p["pos"]), "neg": sorted(p["neg"]),
            "boxes": {k: v for k, v in sorted(p["boxes"].items())}}


def oi_decide(facts, q):
    """Gold from stored annotation facts (used by build and by the tests)."""
    p = {"pos": set(facts["pos"]), "neg": set(facts["neg"]), "boxes": defaultdict(list, facts["boxes"])}
    k = q["kind"]
    if k == "presence":
        return oi_presence(p, q["cls"])
    if k == "count":
        return oi_count(p, q["cls"])
    if k == "position":
        return oi_position(p, q["cls"])
    if k == "rule":
        a, b = oi_presence(p, q["a"]), oi_presence(p, q["b"])
        if a is True or b is True:
            return True
        if a is False and b is False:
            return False
        return None
    if k == "headcount":
        n = oi_count(p, q["cls"])
        return None if n is None else n >= q["k"]
    if k == "hidden":
        return None
    raise KeyError(k)


def oi_items(rng, split, n_target, start_idx):
    photos = [p for p in load_oi().values() if p["split"] == split]
    photos.sort(key=lambda p: p["image_id"])
    rng.shuffle(photos)
    rows, per_kind = [], Counter()
    classes = sorted(OI_CLASS)
    for p in photos:
        if len(rows) >= n_target:
            break
        cands = []
        for c in classes:
            pr = oi_presence(p, c)
            if pr is not None:
                cands.append(("presence", {"cls": c}, pr))
            n = oi_count(p, c)
            if n is not None and n >= 1:
                cands.append(("count", {"cls": c}, n))
                if n >= 1 and c in ("Person", "Car", "Chair", "Window", "Door", "Helmet", "Bicycle", "Truck", "Traffic sign"):
                    kk = max(2, n + rng.choice([-1, 0, 1, 2]))
                    cands.append(("headcount", {"cls": c, "k": kk}, n >= kk))
            pos = oi_position(p, c)
            if pos is not None:
                cands.append(("position", {"cls": c}, pos))
        pres = [(c, g) for k, prm, g in cands if k == "presence" for c in [prm["cls"]]]
        pairs = [(x, y) for i_, x in enumerate(pres) for y in pres[i_ + 1:]
                 if y[0] not in related(x[0]) and not ({x[0], y[0]} & {"Vehicle", "Building"})]
        if pairs:
            (a, ga), (b, gb) = rng.choice(pairs)
            cands.append(("rule", {"a": a, "b": b}, ga or gb))
        present = [prm["cls"] for k, prm, g in cands if k == "presence" and g is True and prm["cls"] not in ("Person", "Building")]
        if present and rng.random() < 0.45:
            hk = max(1, len(Q_OI_HIDDEN) // 3)
            hs = range(len(Q_OI_HIDDEN) - hk, len(Q_OI_HIDDEN)) if split == "heldout" else range(len(Q_OI_HIDDEN) - hk)
            cands.append(("hidden", {"cls": rng.choice(present), "h": rng.choice(list(hs))}, None))
        if not cands:
            continue
        # balance: prefer rarer kinds, safety classes, and "no" answers for presence
        def weight(c):
            k, prm, g = c
            w = {"presence": 1.0, "count": 1.6, "position": 1.4, "rule": 1.5, "headcount": 1.2, "hidden": 1.2}[k]
            w /= (1 + per_kind[k] / 150)
            cls = prm.get("cls") or prm.get("a")
            if cls in ("Helmet", "Ladder", "Fire hydrant", "Traffic cone", "Fire extinguisher", "Glove", "Goggles", "Stairs",
                       "Handrail", "Scaffolding", "Barrel", "Box", "Traffic sign", "Stop sign", "Door", "Fire"):
                w *= 2.5
            if cls in ("Person", "Building", "Car", "Window", "Table", "Vehicle", "Chair"):
                w *= 0.3 if k == "presence" else 0.7
            return w
        chosen = []
        for _ in range(2 if rng.random() < 0.35 else 1):
            if not cands:
                break
            c = rng.choices(cands, weights=[weight(x) for x in cands])[0]
            cands = [x for x in cands if x[0] != c[0]]
            chosen.append(c)
        for k, prm, g in chosen:
            q = dict(prm, kind=k)
            facts = oi_facts(p)
            assert oi_decide(facts, q) == g
            rows.append(oi_row(rng, p, q, g, facts, split, start_idx + len(rows)))
            per_kind[k] += 1
    return rows


def oi_row(rng, p, q, gold, facts, split, idx):
    k = q["kind"]
    ref = f"IMG-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJK')}"
    state = {"photo": pick(rng, OI_PHOTO_LINES, split).format(ref=ref, site=rng.choice(SITES))}
    diff = {"presence": 2, "count": 3, "position": 3, "rule": 4, "headcount": 4, "hidden": 3}[k]
    fam = {"presence": "safety_presence", "count": "safety_count", "position": "safety_position",
           "rule": "safety_checklist", "headcount": "safety_headcount", "hidden": "safety_not_visible"}[k]
    if k == "presence":
        field = {"type": "noul", "question": pick(rng, Q_PRESENCE, split).format(a=OI_CLASS[q["cls"]][0])}
        g = gold
    elif k == "count":
        start = max(0, gold - rng.randint(0, 3))
        opts = [str(v) for v in range(start, start + 5)]
        field = {"type": "choice", "question": pick(rng, Q_OI_COUNT, split).format(plural=OI_CLASS[q["cls"]][1]),
                 "options": [{"key": f"n_{v}", "text": v, "description": ""} for v in opts]}
        g = f"n_{gold}"
        diff = 3 if gold <= 3 else 4
    elif k == "position":
        noun = OI_CLASS[q["cls"]][0].split(" ", 1)[1] if " " in OI_CLASS[q["cls"]][0] else OI_CLASS[q["cls"]][0]
        field = {"type": "choice", "question": pick(rng, Q_POSITION, split).format(noun=noun),
                 "options": [{"key": "left", "text": "left third", "description": ""},
                             {"key": "middle", "text": "middle", "description": ""},
                             {"key": "right", "text": "right third", "description": ""}]}
        g = gold
    elif k == "rule":
        state["rule"] = pick(rng, OI_RULES, split).format(a=OI_CLASS[q["a"]][0], b=OI_CLASS[q["b"]][0])
        field = {"type": "noul", "question": pick(rng, Q_RULE, split)}
        g = gold
    elif k == "headcount":
        state["rule"] = pick(rng, HEAD_RULES, split).format(k=q["k"], plural=OI_CLASS[q["cls"]][1])
        field = {"type": "noul", "question": pick(rng, Q_HEAD, split)}
        g = gold
    else:
        noun = OI_CLASS[q["cls"]][0].split(" ", 1)[-1]
        field = {"type": "noul", "question": Q_OI_HIDDEN[q["h"]].format(noun=noun)}
        g = None
    assert g == oi_gold_key(q, gold)
    a = p["attribution"]
    return {
        "id": f"{'p3-I-safety' if split == 'train' else 'p3-hf1-I-safety'}-{idx:06d}", "source": "I",
        "dataset": "openimages_safety", "family": fam, "difficulty": diff, "state": state, "images": [p["image"]],
        "field": field, "gold": g, "unknown_reason": None if g is not None else "insufficient_evidence",
        "gold_kind": "dataset" if g is not None else "constructed", "parent_id": None,
        "provenance": {
            "licence": "CC-BY-2.0", "image_licences": ["CC-BY-2.0"], "annotation_licence": "CC-BY-4.0",
            "licence_evidence": [OI_EVIDENCE, OI_ANNOT_EVIDENCE],
            "photo_attribution": [dict(a, licence="CC-BY-2.0")],
            "upstream_dataset": "openimages_v7", "upstream_id": p["image_id"], "upstream_split": "test", "split": split,
            "image_sha256": [p["sha256"]], "gold_source": "human-verified Open Images V7 image-level labels and boxes"
            if g is not None else "unknown by construction (property no photo can show)",
            "annotation_files": str(OI_SOURCES.relative_to(ROOT)), "question": q, "oi_facts": facts,
            "generator": "scripts/p3/gen_safety.py",
        },
    }


# ================================================================================================= (b) rendered scenes
S = 2       # supersampling


def lighten(c, k):
    return tuple(int(min(255, x + (255 - x) * k)) for x in c)


def darken(c, k):
    return tuple(int(x * (1 - k)) for x in c)


SIGNS = {  # name -> (header colour, text colour, header text, body text, glyph)
    "hard_hat": ((20, 80, 170), (255, 255, 255), "MANDATORY", "HARD HAT AREA", "hat"),
    "eye": ((20, 80, 170), (255, 255, 255), "MANDATORY", "EYE PROTECTION", "eye"),
    "hearing": ((20, 80, 170), (255, 255, 255), "MANDATORY", "HEARING PROTECTION", "ear"),
    "voltage": ((245, 200, 0), (0, 0, 0), "WARNING", "HIGH VOLTAGE", "bolt"),
    "forklift": ((245, 200, 0), (0, 0, 0), "WARNING", "FORKLIFT TRAFFIC", "fork"),
    "corrosive": ((245, 200, 0), (0, 0, 0), "WARNING", "CORROSIVE SUBSTANCES", "drop"),
    "no_smoking": ((200, 30, 30), (255, 255, 255), "PROHIBITED", "NO SMOKING", "cig"),
    "staff_only": ((200, 30, 30), (255, 255, 255), "DANGER", "AUTHORISED STAFF ONLY", "hand"),
    "fire_exit": ((20, 140, 60), (255, 255, 255), "SAFE CONDITION", "FIRE EXIT KEEP CLEAR", "run"),
}
SIGN_TEXT = {k: v[3].lower() for k, v in SIGNS.items()}
SIGN_TEXT.update({"fire_exit": "fire exit keep clear", "staff_only": "authorised staff only"})


def draw_sign(d, box, kind, legible=True, s=S):
    x0, y0, x1, y1 = box
    hc, tc, head, body, glyph = SIGNS[kind]
    d.rectangle(box, fill=(250, 250, 248), outline=(40, 40, 40), width=2 * s)
    hh = (y1 - y0) * 0.3
    d.rectangle([x0, y0, x1, y0 + hh], fill=hc)
    f = fit_font(d, head, "sans_bold", (x1 - x0) * 0.85, hh * 0.7, start=int(hh * 0.7))
    text_center(d, (x0 + x1) / 2, y0 + hh / 2, head, f, tc)
    gx = x0 + (x1 - x0) * 0.2
    gy = y0 + hh + (y1 - y0 - hh) / 2
    r = min((x1 - x0) * 0.14, (y1 - y0 - hh) * 0.38)
    if hc[0] > 200 and hc[1] > 150:
        d.polygon([(gx, gy - r), (gx + r, gy + r * 0.8), (gx - r, gy + r * 0.8)], fill=hc, outline=(0, 0, 0))
    else:
        d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=hc)
    words = body.split()
    lines = [" ".join(words[: (len(words) + 1) // 2]), " ".join(words[(len(words) + 1) // 2:])] if len(words) > 1 else words
    tw = (x1 - x0) * 0.62
    lh = (y1 - y0 - hh) / (len(lines) + 0.6)
    f = fit_font(d, max(lines, key=len), "sans_bold", tw, lh * 0.85, start=int(lh))
    for i, ln in enumerate(lines):
        text_center(d, x0 + (x1 - x0) * 0.66, y0 + hh + lh * (i + 0.8), ln, f, (20, 20, 20))


def draw_worker(d, cx, feet_y, h, w, s=S):
    """A flat stylised worker; w = {"hard_hat","vest","goggles","shirt","skin","hair","hat_colour","vest_colour"}."""
    head_r = h * 0.075
    head_cy = feet_y - h + head_r * 1.2
    body_top = head_cy + head_r * 1.2
    body_bot = feet_y - h * 0.44
    bw = h * 0.2
    skin, shirt, hair = tuple(w["skin"]), tuple(w["shirt"]), tuple(w["hair"])
    trousers = tuple(w["trousers"])
    # legs
    d.rectangle([cx - bw * 0.45, body_bot, cx - bw * 0.06, feet_y - h * 0.02], fill=trousers)
    d.rectangle([cx + bw * 0.06, body_bot, cx + bw * 0.45, feet_y - h * 0.02], fill=trousers)
    d.rectangle([cx - bw * 0.55, feet_y - h * 0.035, cx - bw * 0.02, feet_y], fill=(30, 30, 30))
    d.rectangle([cx + bw * 0.02, feet_y - h * 0.035, cx + bw * 0.55, feet_y], fill=(30, 30, 30))
    # arms
    d.rectangle([cx - bw * 0.78, body_top + h * 0.02, cx - bw * 0.52, body_bot + h * 0.04], fill=shirt)
    d.rectangle([cx + bw * 0.52, body_top + h * 0.02, cx + bw * 0.78, body_bot + h * 0.04], fill=shirt)
    d.ellipse([cx - bw * 0.8, body_bot + h * 0.02, cx - bw * 0.5, body_bot + h * 0.07], fill=skin)
    d.ellipse([cx + bw * 0.5, body_bot + h * 0.02, cx + bw * 0.8, body_bot + h * 0.07], fill=skin)
    # torso
    d.rounded_rectangle([cx - bw * 0.55, body_top, cx + bw * 0.55, body_bot], radius=int(bw * 0.18), fill=shirt)
    if w["vest"]:
        vc = tuple(w["vest_colour"])
        d.polygon([(cx - bw * 0.55, body_top + h * 0.01), (cx - bw * 0.12, body_top + h * 0.01), (cx - bw * 0.08, body_bot),
                   (cx - bw * 0.55, body_bot)], fill=vc)
        d.polygon([(cx + bw * 0.55, body_top + h * 0.01), (cx + bw * 0.12, body_top + h * 0.01), (cx + bw * 0.08, body_bot),
                   (cx + bw * 0.55, body_bot)], fill=vc)
        for f_ in (0.55, 0.8):
            yy = body_top + (body_bot - body_top) * f_
            d.rectangle([cx - bw * 0.55, yy, cx - bw * 0.1, yy + h * 0.018], fill=(215, 215, 220))
            d.rectangle([cx + bw * 0.1, yy, cx + bw * 0.55, yy + h * 0.018], fill=(215, 215, 220))
    # neck + head
    d.rectangle([cx - head_r * 0.35, head_cy + head_r * 0.8, cx + head_r * 0.35, body_top + 2], fill=skin)
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=skin)
    if w["hard_hat"]:
        hc = tuple(w["hat_colour"])
        d.pieslice([cx - head_r * 1.12, head_cy - head_r * 1.45, cx + head_r * 1.12, head_cy + head_r * 0.55], 180, 360, fill=hc)
        d.rectangle([cx - head_r * 1.4, head_cy - head_r * 0.48, cx + head_r * 1.4, head_cy - head_r * 0.3], fill=darken(hc, 0.15))
        d.rectangle([cx - head_r * 0.12, head_cy - head_r * 1.42, cx + head_r * 0.12, head_cy - head_r * 0.5], fill=darken(hc, 0.2))
    else:
        d.chord([cx - head_r * 1.02, head_cy - head_r * 1.05, cx + head_r * 1.02, head_cy + head_r * 0.6], 180, 360, fill=hair)
    if w["goggles"]:
        gy = head_cy + head_r * 0.05
        d.rectangle([cx - head_r * 1.02, gy - head_r * 0.12, cx + head_r * 1.02, gy + head_r * 0.12], fill=(30, 30, 30))
        d.rounded_rectangle([cx - head_r * 0.85, gy - head_r * 0.3, cx - head_r * 0.08, gy + head_r * 0.3],
                            radius=int(head_r * 0.12), fill=(60, 150, 210), outline=(20, 20, 20), width=s)
        d.rounded_rectangle([cx + head_r * 0.08, gy - head_r * 0.3, cx + head_r * 0.85, gy + head_r * 0.3],
                            radius=int(head_r * 0.12), fill=(60, 150, 210), outline=(20, 20, 20), width=s)
    else:
        for ex in (-0.38, 0.38):
            d.ellipse([cx + head_r * ex - head_r * 0.09, head_cy - head_r * 0.05, cx + head_r * ex + head_r * 0.09,
                       head_cy + head_r * 0.13], fill=(30, 30, 30))
    return {"head": [cx - head_r * 1.45, head_cy - head_r * 1.5, cx + head_r * 1.45, head_cy + head_r * 1.1],
            "body": [cx - bw * 0.8, body_top, cx + bw * 0.8, body_bot + h * 0.07],
            "all": [cx - bw * 0.8, head_cy - head_r * 1.5, cx + bw * 0.8, feet_y]}


SKINS = [(241, 204, 170), (224, 172, 130), (190, 130, 90), (141, 85, 56), (100, 62, 40), (250, 220, 195)]
HAIRS = [(30, 24, 20), (70, 45, 25), (95, 60, 30), (15, 15, 15)]      # dark only: never mistaken for a hard hat
SHIRTS = [(40, 60, 110), (90, 90, 96), (130, 30, 30), (40, 100, 60), (60, 60, 64), (150, 120, 90), (20, 20, 24), (90, 60, 120)]
HAT_COLOURS = [(250, 210, 0), (245, 245, 245), (240, 120, 0), (30, 90, 200), (220, 40, 40)]
VEST_COLOURS = [(255, 140, 0), (190, 240, 20), (255, 90, 20)]


# ------------------------------------------------------------------------------------------------ site view
def build_site(rng, split):
    W = rng.choice([1152, 1280, 1280])
    H = rng.choice([768, 800, 864, 900])
    n_w = rng.randint(3, 6)
    zones = ["A", "B"] if rng.random() < 0.7 else ["A", "B", "C"]
    workers = []
    xs = sorted(rng.sample(range(0, 1000), n_w))
    span_x0, span_x1 = W * 0.09, W * 0.91
    gap = (span_x1 - span_x0) / n_w
    bounds_x = [W * 0.05 + k * W * 0.9 / len(zones) for k in range(1, len(zones))]
    for i in range(n_w):
        cx = span_x0 + gap * (i + 0.5) + rng.uniform(-gap * 0.12, gap * 0.12)
        for bx in bounds_x:                   # never stand on a zone line: the zone must be unambiguous
            if abs(cx - bx) < W * 0.055:
                cx = bx + (W * 0.055 if cx >= bx else -W * 0.055)
        zi = min(len(zones) - 1, int((cx - W * 0.05) / (W * 0.9) * len(zones)))
        workers.append({"id": f"W{i + 1}", "cx": round(cx), "zone": zones[zi],
                        "hard_hat": rng.random() < 0.7, "vest": rng.random() < 0.68, "goggles": rng.random() < 0.45,
                        "skin": rng.choice(SKINS), "hair": rng.choice(HAIRS), "shirt": rng.choice(SHIRTS),
                        "trousers": rng.choice([(40, 44, 60), (60, 60, 60), (30, 30, 34), (80, 70, 55)]),
                        "hat_colour": rng.choice(HAT_COLOURS), "vest_colour": rng.choice(VEST_COLOURS),
                        "h": round(H * rng.uniform(0.34, 0.4))})
    tags = [w["id"] for w in workers]
    rng.shuffle(tags)
    for w, t in zip(workers, tags):   # tag order is shuffled so W1 is not always leftmost
        w["id"] = t
    n_sec = rng.choice([3, 4, 5])
    sections = [{"id": f"S{i + 1}", "present": True} for i in range(n_sec)]
    for s in rng.sample(sections, rng.choice([0, 1, 1, 1, 2])):
        s["present"] = False
    n_doors = rng.choice([2, 3, 3, 4])
    sign_kinds = rng.sample(sorted(SIGNS), n_doors)
    doors = [{"id": f"D{i + 1}", "sign": sign_kinds[i], "legible": True} for i in range(n_doors)]
    spec = {"scene": "site_view", "W": W, "H": H, "zones": zones, "workers": workers, "sections": sections, "doors": doors,
            "palette": rng.randrange(1000), "post": affine_params(rng, 0.7), "occluders": [], "crop": None,
            "ref": f"{rng.choice(['WH', 'DC', 'YD', 'PL'])}-{rng.randint(100, 999)}", "split": split}
    return spec


def layout_site(spec):
    """Geometry (canvas coords, 1x) of the site view elements."""
    W, H = spec["W"], spec["H"]
    wall_bot = H * 0.52
    mez_y = H * 0.2
    geo = {"wall_bot": wall_bot, "mez_y": mez_y}
    n_sec = len(spec["sections"])
    x0, x1 = W * 0.05, W * 0.95
    sw = (x1 - x0) / n_sec
    geo["sections"] = [[x0 + i * sw, mez_y - H * 0.08, x0 + (i + 1) * sw, mez_y] for i in range(n_sec)]
    nd = len(spec["doors"])
    dw = W * 0.1
    gapd = (W * 0.9) / nd
    geo["doors"] = []
    for i in range(nd):
        cx = W * 0.05 + gapd * (i + 0.5)
        geo["doors"].append({"door": [cx - dw / 2, wall_bot - H * 0.19, cx + dw / 2, wall_bot],
                             "sign": [cx - W * 0.075, mez_y + H * 0.04, cx + W * 0.075, mez_y + H * 0.04 + H * 0.075]})
    geo["workers"] = {}
    for w in spec["workers"]:
        feet = H * 0.93 - (hash_int(w["id"]) % 5) * H * 0.012
        geo["workers"][w["id"]] = {"feet": feet}
    return geo


def hash_int(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def render_site(spec):
    W, H = spec["W"], spec["H"]
    geo = layout_site(spec)
    img = Image.new("RGB", (W * S, H * S), (200, 200, 196))
    d = ImageDraw.Draw(img)
    pal = spec["palette"]
    wall = [(214, 212, 204), (190, 198, 206), (222, 218, 210), (176, 180, 186)][pal % 4]
    floor = [(150, 150, 146), (120, 124, 126), (165, 160, 150)][pal % 3]
    d.rectangle([0, 0, W * S, geo["wall_bot"] * S], fill=wall)
    d.rectangle([0, geo["wall_bot"] * S, W * S, H * S], fill=floor)
    # zone floor markings
    nz = len(spec["zones"])
    zc = [(250, 210, 0), (60, 160, 230), (240, 110, 60)]
    for i, z in enumerate(spec["zones"]):
        zx0, zx1 = W * 0.05 + i * W * 0.9 / nz, W * 0.05 + (i + 1) * W * 0.9 / nz
        d.rectangle([zx0 * S, geo["wall_bot"] * S + 8 * S, zx1 * S - 6 * S, H * S - 6 * S], outline=zc[i], width=5 * S)
        f = font("din", 30 * S)
        d.text(((zx0 + 16) * S, (H - 50) * S), f"ZONE {z}", font=f, fill=zc[i])
    # mezzanine
    my = geo["mez_y"]
    d.rectangle([0, my * S, W * S, (my + H * 0.03) * S], fill=(90, 94, 100))
    for i, (sec, box) in enumerate(zip(spec["sections"], geo["sections"])):
        bx0, by0, bx1, by1 = [v * S for v in box]
        d.rectangle([bx0, by0, bx0 + 6 * S, by1], fill=(240, 190, 0))
        if sec["present"]:
            for f_ in (0.05, 0.5):
                d.rectangle([bx0, by0 + (by1 - by0) * f_, bx1, by0 + (by1 - by0) * f_ + 6 * S], fill=(240, 190, 0))
            for k in range(1, 4):
                xx = bx0 + (bx1 - bx0) * k / 4
                d.rectangle([xx - 2 * S, by0, xx + 2 * S, by1], fill=(240, 190, 0))
        f = font("din", 18 * S)
        d.rectangle([bx0 + 10 * S, by1 + 6 * S, bx0 + 52 * S, by1 + 30 * S], fill=(30, 30, 30))
        text_center(d, bx0 + 31 * S, by1 + 18 * S, sec["id"], f, (255, 255, 255))
    d.rectangle([W * S - 6 * S, (my - H * 0.08) * S, W * S, (my + H * 0.03) * S], fill=(240, 190, 0))
    # racks in the background between doors
    for k in range(6):
        rx = (W * 0.02 + k * W * 0.17) * S
        d.rectangle([rx, (my + H * 0.17) * S, rx + W * 0.012 * S, geo["wall_bot"] * S], fill=(40, 80, 150))
    # doors + signs
    for dr, g in zip(spec["doors"], geo["doors"]):
        x0, y0, x1, y1 = [v * S for v in g["door"]]
        d.rectangle([x0 - 6 * S, y0 - 6 * S, x1 + 6 * S, y1], fill=(70, 72, 78))
        d.rectangle([x0, y0, x1, y1], fill=(120, 130, 140))
        d.line([(x0 + x1) / 2, y0, (x0 + x1) / 2, y1], fill=(70, 72, 78), width=3 * S)
        f = font("din", 24 * S)
        d.rectangle([(x0 + x1) / 2 - 24 * S, y0 + (y1 - y0) * 0.35, (x0 + x1) / 2 + 24 * S, y0 + (y1 - y0) * 0.35 + 30 * S],
                    fill=(250, 250, 250))
        text_center(d, (x0 + x1) / 2, y0 + (y1 - y0) * 0.35 + 15 * S, dr["id"], f, (20, 20, 20))
        sb = [v * S for v in g["sign"]]
        sign_img = Image.new("RGB", (int(sb[2] - sb[0]), int(sb[3] - sb[1])), (255, 255, 255))
        draw_sign(ImageDraw.Draw(sign_img), [0, 0, sign_img.size[0] - 1, sign_img.size[1] - 1], dr["sign"])
        if not dr["legible"]:
            sign_img = sign_img.filter(ImageFilter.GaussianBlur(9 * S))
            dd = ImageDraw.Draw(sign_img)
            for k in range(5):
                dd.ellipse([sign_img.size[0] * (0.1 + 0.18 * k), sign_img.size[1] * 0.25, sign_img.size[0] * (0.35 + 0.18 * k),
                            sign_img.size[1] * 0.95], fill=(120, 110, 95))
        img.paste(sign_img, (int(sb[0]), int(sb[1])))
        d.line([(sb[0] + sb[2]) / 2, sb[3], (x0 + x1) / 2, y0 - 6 * S], fill=(80, 80, 80), width=2 * S)
    # workers
    for w in sorted(spec["workers"], key=lambda w: geo["workers"][w["id"]]["feet"]):
        feet = geo["workers"][w["id"]]["feet"]
        boxes = draw_worker(d, w["cx"] * S, feet * S, w["h"] * S, w)
        hb = boxes["head"]
        f = font("din", 22 * S)
        tx, ty = w["cx"] * S, hb[1] - 20 * S
        d.rounded_rectangle([tx - 26 * S, ty - 15 * S, tx + 26 * S, ty + 15 * S], radius=6 * S, fill=(255, 255, 255),
                            outline=(20, 20, 20), width=2 * S)
        text_center(d, tx, ty, w["id"], f, (20, 20, 20))
    for oc in spec["occluders"]:
        x0, y0, x1, y1 = [v * S for v in oc["box"]]
        n = max(2, int((y1 - y0) / (95 * S)))
        bh = (y1 - y0) / n
        for k in range(n):
            d.rectangle([x0, y0 + k * bh, x1, y0 + (k + 1) * bh], fill=(192, 152, 104), outline=(110, 82, 54), width=3 * S)
            d.rectangle([x0 + (x1 - x0) * 0.44, y0 + k * bh, x0 + (x1 - x0) * 0.56, y0 + (k + 1) * bh], fill=(212, 182, 140))
        d.rectangle([x0 - 8 * S, y1 - 16 * S, x1 + 8 * S, y1], fill=(150, 110, 60))
    return img.resize((W, H), Image.LANCZOS)


def worker_boxes(spec):
    """Head / body / whole boxes per worker in canvas coords (mirrors draw_worker)."""
    geo = layout_site(spec)
    out = {}
    for w in spec["workers"]:
        feet = geo["workers"][w["id"]]["feet"]
        h = w["h"]
        cx = w["cx"]
        head_r = h * 0.075
        head_cy = feet - h + head_r * 1.2
        body_top = head_cy + head_r * 1.2
        body_bot = feet - h * 0.44
        bw = h * 0.2
        out[w["id"]] = {"head": [cx - head_r * 1.45, head_cy - head_r * 1.5 - 36, cx + head_r * 1.45, head_cy + head_r * 1.1],
                        "body": [cx - bw * 0.8, body_top, cx + bw * 0.8, body_bot + h * 0.07],
                        "all": [cx - bw * 0.8, head_cy - head_r * 1.5 - 36, cx + bw * 0.8, feet]}
    return out


def site_hidden(spec):
    M = forward_matrix((spec["W"], spec["H"]), spec["post"])
    win = spec["crop"] or [0, 0, spec["W"], spec["H"]]
    win = [win[0] + 2, win[1] + 2, win[2] - 2, win[3] - 2]
    geo = layout_site(spec)
    wb = worker_boxes(spec)
    for w in spec["workers"]:
        b = wb[w["id"]]
        head_h = any(ovl(o["box"], b["head"]) > 0 for o in spec["occluders"]) or not inside(map_box(M, b["head"]), win)
        body_h = any(ovl(o["box"], b["body"]) > 0 for o in spec["occluders"]) or not inside(map_box(M, b["body"]), win)
        w["head_hidden"], w["body_hidden"] = head_h, body_h
    for s, box in zip(spec["sections"], geo["sections"]):
        s["hidden"] = not inside(map_box(M, box), win)
    for dr, g in zip(spec["doors"], geo["doors"]):
        dr["hidden"] = (not dr["legible"]) or not inside(map_box(M, g["sign"]), win)


def ovl(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def inside(box, win, tol=1.0):
    return box[0] >= win[0] - tol and box[1] >= win[1] - tol and box[2] <= win[2] + tol and box[3] <= win[3] + tol


# ------------------------------------------------------------------------------------------------ floor plan
def build_plan(rng, split):
    W = rng.choice([1152, 1280, 1280])
    H = rng.choice([800, 864, 900, 960])
    cols, rows = rng.choice([(2, 2), (3, 2), (2, 2), (3, 2)])
    zone_names = [chr(65 + i) for i in range(cols * rows)]
    x0, y0, x1, y1 = W * 0.06, H * 0.1, W * 0.94, H * 0.9
    zw, zh = (x1 - x0) / cols, (y1 - y0) / rows
    zones = []
    for r in range(rows):
        for c in range(cols):
            zones.append({"id": zone_names[r * cols + c], "box": [round(x0 + c * zw), round(y0 + r * zh),
                                                                  round(x0 + (c + 1) * zw), round(y0 + (r + 1) * zh)]})
    occupied = [[z["box"][2] - 116, z["box"][3] - 58, z["box"][2] - 2, z["box"][3] - 22] for z in zones]   # zone labels

    def free(r, m=14):
        return all(r[2] + m <= o[0] or o[2] + m <= r[0] or r[3] + m <= o[1] or o[3] + m <= r[1] for o in occupied)
    # forklift lanes first (along zone boundaries)
    lanes = []
    n_lanes = rng.choice([1, 2, 2])
    for i in range(n_lanes):
        if i == 0:
            ly = y0 + zh
            ln = {"id": f"L{i + 1}", "orient": "h", "pos": round(ly), "lo": round(x0 + 20), "hi": round(x1 - 20)}
        else:
            lx = x0 + zw * rng.randint(1, cols - 1)
            ln = {"id": f"L{i + 1}", "orient": "v", "pos": round(lx), "lo": round(y0 + 20), "hi": round(y1 - 20)}
        ln.update(obstructed=rng.random() < 0.4, block_at=round(rng.uniform(0.25, 0.75), 3))
        lanes.append(ln)
        occupied.append([ln["lo"] - 10, ln["pos"] - 40, ln["hi"] + 10, ln["pos"] + 40] if ln["orient"] == "h" else
                        [ln["pos"] - 40, ln["lo"] - 10, ln["pos"] + 40, ln["hi"] + 10])
    # exits on the outer wall (gap + sign + the area in front of it)
    exits = []
    walls = ["top", "bottom", "left", "right"]
    n_ex = rng.choice([3, 4, 4, 5])
    for i in range(n_ex):
        wall = walls[i % 4] if i < 4 else rng.choice(walls)
        for _ in range(60):
            if wall in ("top", "bottom"):
                ex, ey = rng.uniform(x0 + 90, x1 - 190), (y0 if wall == "top" else y1)
            else:
                ex, ey = (x0 if wall == "left" else x1), rng.uniform(y0 + 110, y1 - 90)
            e = {"id": f"E{len(exits) + 1}", "wall": wall, "x": round(ex), "y": round(ey), "blocked": False}
            gap, sign, (bx, by) = exit_geom(e)
            r = [min(gap[0], sign[0], bx - 45), min(gap[1], sign[1], by - 45), max(gap[2], sign[2], bx + 45),
                 max(gap[3], sign[3], by + 45)]
            if free(r):
                occupied.append(r)
                exits.append(e)
                break
    for e in rng.sample(exits, min(len(exits), rng.choice([0, 1, 1, 1, 2]))):
        e["blocked"] = True
    # extinguisher stations near the top of each zone, spills anywhere inside, all apart from each other
    stations, spills = [], []
    k = 0
    for z in zones:
        bx0, by0, bx1, by1 = z["box"]
        for _ in range(rng.choice([1, 1, 2])):
            for _ in range(60):
                x, y = rng.uniform(bx0 + 45, bx1 - 45), rng.uniform(by0 + 40, by0 + 110)
                r = [x - 34, y - 30, x + 34, y + 46]
                if free(r):
                    occupied.append(r)
                    k += 1
                    stations.append({"id": f"FE-{k}", "zone": z["id"], "x": round(x), "y": round(y),
                                     "stocked": rng.random() < 0.75})
                    break
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
            rad = rng.randint(16, 24)
            for _ in range(60):
                x, y = rng.uniform(bx0 + 50, bx1 - 50), rng.uniform(by0 + 60, by1 - 70)
                r = [x - rad * 1.5 - 6, y - rad * 1.3 - 6, x + rad * 1.5 + 6, y + rad + 6]
                if free(r, 24):
                    occupied.append(r)
                    spills.append({"zone": z["id"], "x": round(x), "y": round(y), "r": rad})
                    break
    for i, sp in enumerate(spills):
        sp["id"] = f"sp{i}"
    return {"scene": "floor_plan", "W": W, "H": H, "bounds": [round(x0), round(y0), round(x1), round(y1)],
            "cols": cols, "rows": rows, "zones": zones, "exits": exits, "stations": stations, "spills": spills,
            "lanes": lanes, "palette": rng.randrange(1000), "post": affine_params(rng, 0.6), "crop": None,
            "ref": f"PLAN-{rng.randint(100, 999)}{rng.choice('ABCDEF')}", "split": split}


def exit_geom(e):
    """(gap rect, sign rect, block centre) of an exit in canvas coords; the sign sits inside the building next to the gap."""
    ex, ey = e["x"], e["y"]
    if e["wall"] in ("top", "bottom"):
        gap = [ex - 30, ey - 5, ex + 30, ey + 5]
        sy = ey + 10 if e["wall"] == "top" else ey - 34
        sign = [ex + 36, sy, ex + 124, sy + 24]
        block = (ex, ey + 30 if e["wall"] == "top" else ey - 30)
    else:
        gap = [ex - 5, ey - 30, ex + 5, ey + 30]
        sx = ex + 10 if e["wall"] == "left" else ex - 98
        sign = [sx, ey - 64, sx + 88, ey - 40]
        block = (ex + 30 if e["wall"] == "left" else ex - 30, ey)
    return gap, sign, block


def plan_boxes(spec):
    """Element boxes (canvas coords) used for visibility."""
    out = {}
    for e in spec["exits"]:
        gap, sign, (bx, by) = exit_geom(e)
        out[e["id"]] = [min(gap[0], sign[0], bx - 45) - 4, min(gap[1], sign[1], by - 45) - 4,
                        max(gap[2], sign[2], bx + 45) + 4, max(gap[3], sign[3], by + 45) + 4]
    for s in spec["stations"]:
        out[s["id"]] = [s["x"] - 30, s["y"] - 26, s["x"] + 30, s["y"] + 40]
    for s in spec["spills"]:
        out[s["id"]] = [s["x"] - s["r"] - 20, s["y"] - s["r"] - 20, s["x"] + s["r"] + 20, s["y"] + s["r"] + 20]
    for ln in spec["lanes"]:
        if ln["orient"] == "h":
            out[ln["id"]] = [ln["lo"], ln["pos"] - 30, ln["hi"], ln["pos"] + 30]
        else:
            out[ln["id"]] = [ln["pos"] - 30, ln["lo"], ln["pos"] + 30, ln["hi"]]
    for z in spec["zones"]:
        out["zone:" + z["id"]] = z["box"]
    return out


def plan_hidden(spec):
    M = forward_matrix((spec["W"], spec["H"]), spec["post"])
    win = spec["crop"] or [0, 0, spec["W"], spec["H"]]
    win = [win[0] + 2, win[1] + 2, win[2] - 2, win[3] - 2]
    bx = plan_boxes(spec)
    for coll in ("exits", "stations", "lanes"):
        for e in spec[coll]:
            e["hidden"] = not inside(map_box(M, bx[e["id"]]), win)
    for z in spec["zones"]:
        z["partial"] = not inside(map_box(M, z["box"]), win)
    for s in spec["spills"]:
        s["hidden"] = not inside(map_box(M, bx[s["id"]]), win)


def render_plan(spec):
    W, H = spec["W"], spec["H"]
    img = Image.new("RGB", (W * S, H * S), (250, 250, 247))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [v * S for v in spec["bounds"]]
    f_title = font("din", 26 * S)
    d.text((x0, (spec["bounds"][1] - 48) * S), f"FLOOR PLAN {spec['ref']}  -  EMERGENCY & HAZARD MARKUP", font=f_title,
           fill=(30, 30, 30))
    # grid
    for gx in range(int(x0), int(x1), 40 * S):
        d.line([gx, y0, gx, y1], fill=(232, 236, 240), width=S)
    for gy in range(int(y0), int(y1), 40 * S):
        d.line([x0, gy, x1, gy], fill=(232, 236, 240), width=S)
    # lanes
    for ln in spec["lanes"]:
        if ln["orient"] == "h":
            box = [ln["lo"] * S, (ln["pos"] - 26) * S, ln["hi"] * S, (ln["pos"] + 26) * S]
        else:
            box = [(ln["pos"] - 26) * S, ln["lo"] * S, (ln["pos"] + 26) * S, ln["hi"] * S]
        d.rectangle(box, fill=(255, 246, 200))
        for side in (0, 1):
            if ln["orient"] == "h":
                yy = box[1] if side == 0 else box[3]
                for xx in range(int(box[0]), int(box[2]), 28 * S):
                    d.line([xx, yy, xx + 16 * S, yy], fill=(230, 170, 0), width=4 * S)
            else:
                xx = box[0] if side == 0 else box[2]
                for yy in range(int(box[1]), int(box[3]), 28 * S):
                    d.line([xx, yy, xx, yy + 16 * S], fill=(230, 170, 0), width=4 * S)
        f = font("din", 18 * S)
        lx, ly = (box[0] + 30 * S, (box[1] + box[3]) / 2) if ln["orient"] == "h" else ((box[0] + box[2]) / 2, box[1] + 30 * S)
        d.rectangle([lx - 20 * S, ly - 13 * S, lx + 20 * S, ly + 13 * S], fill=(230, 170, 0))
        text_center(d, lx, ly, ln["id"], f, (20, 20, 20))
        if ln["obstructed"]:
            t = ln["block_at"]
            if ln["orient"] == "h":
                cx, cy = (ln["lo"] + (ln["hi"] - ln["lo"]) * t) * S, ln["pos"] * S
            else:
                cx, cy = ln["pos"] * S, (ln["lo"] + (ln["hi"] - ln["lo"]) * t) * S
            for dx, dy in ((-14, -12), (6, -10), (-4, 8)):
                d.rectangle([cx + dx * S - 12 * S, cy + dy * S - 12 * S, cx + dx * S + 12 * S, cy + dy * S + 12 * S],
                            fill=(170, 120, 60), outline=(90, 60, 30), width=2 * S)
    # zones (walls)
    for z in spec["zones"]:
        bx0, by0, bx1, by1 = [v * S for v in z["box"]]
        d.rectangle([bx0, by0, bx1, by1], outline=(40, 40, 40), width=3 * S)
        f = font("din", 34 * S)
        d.text((bx1 - 110 * S, by1 - 52 * S), f"ZONE {z['id']}", font=font("din", 22 * S), fill=(90, 90, 100))
    d.rectangle([x0, y0, x1, y1], outline=(20, 20, 20), width=7 * S)
    # exits: gap in outer wall + green sign
    for e in spec["exits"]:
        gap, sign, (bx, by) = exit_geom(e)
        d.rectangle([v * S for v in gap], fill=(250, 250, 247))
        sbox = [v * S for v in sign]
        d.rectangle(sbox, fill=(20, 140, 60))
        text_center(d, (sbox[0] + sbox[2]) / 2, (sbox[1] + sbox[3]) / 2, f"EXIT {e['id']}", font("din", 15 * S),
                    (255, 255, 255))
        if e["blocked"]:
            cx, cy = bx * S, by * S
            for dx, dy in ((-16, -10), (6, -12), (-6, 8), (14, 6)):
                d.rectangle([cx + dx * S - 12 * S, cy + dy * S - 11 * S, cx + dx * S + 12 * S, cy + dy * S + 11 * S],
                            fill=(170, 120, 60), outline=(90, 60, 30), width=2 * S)
    # stations
    for st in spec["stations"]:
        cx, cy = st["x"] * S, st["y"] * S
        d.rectangle([cx - 20 * S, cy - 20 * S, cx + 20 * S, cy + 20 * S], outline=(210, 30, 30), width=3 * S,
                    fill=(255, 235, 235))
        if st["stocked"]:
            d.rounded_rectangle([cx - 7 * S, cy - 12 * S, cx + 7 * S, cy + 16 * S], radius=5 * S, fill=(210, 30, 30))
            d.rectangle([cx - 3 * S, cy - 17 * S, cx + 3 * S, cy - 11 * S], fill=(40, 40, 40))
            d.line([cx + 3 * S, cy - 15 * S, cx + 12 * S, cy - 8 * S], fill=(40, 40, 40), width=2 * S)
        text_center(d, cx, cy + 32 * S, st["id"], font("din", 15 * S), (180, 20, 20))
    # spills
    for sp in spec["spills"]:
        cx, cy, r = sp["x"] * S, sp["y"] * S, sp["r"] * S
        d.ellipse([cx - r * 1.3, cy - r * 0.8, cx + r * 1.2, cy + r * 0.9], fill=(120, 180, 235))
        d.ellipse([cx - r * 0.4, cy - r * 1.1, cx + r * 1.4, cy + r * 0.4], fill=(120, 180, 235))
        d.polygon([(cx, cy - 16 * S), (cx + 15 * S, cy + 11 * S), (cx - 15 * S, cy + 11 * S)], fill=(250, 210, 0),
                  outline=(30, 30, 30))
        text_center(d, cx, cy + 3 * S, "!", font("sans_bold", 16 * S), (20, 20, 20))
    # legend
    lx, ly = x0, (spec["bounds"][3] + 14) * S
    f = font("sans", 15 * S)
    items = [("exit", "Exit (boxes in front = blocked)"), ("st", "Extinguisher station (empty = no extinguisher)"),
             ("sp", "Marked spill"), ("ln", "Forklift lane")]
    for kind, txt in items:
        if kind == "exit":
            d.rectangle([lx, ly, lx + 26 * S, ly + 16 * S], fill=(20, 140, 60))
        elif kind == "st":
            d.rectangle([lx, ly, lx + 16 * S, ly + 16 * S], outline=(210, 30, 30), width=2 * S)
        elif kind == "sp":
            d.ellipse([lx, ly, lx + 22 * S, ly + 16 * S], fill=(120, 180, 235))
        else:
            d.rectangle([lx, ly, lx + 26 * S, ly + 16 * S], fill=(255, 246, 200), outline=(230, 170, 0), width=2 * S)
        d.text((lx + 32 * S, ly - 1 * S), txt, font=f, fill=(40, 40, 40))
        lx += (40 + 8.2 * len(txt)) * S
    return img.resize((W, H), Image.LANCZOS)


# ------------------------------------------------------------------------------------------------ gold for rendered scenes
REQUIRE = {"hard_hat": "a hard hat", "vest": "a high-visibility vest", "goggles": "safety goggles"}


def world_site(spec):
    return {"workers": {w["id"]: {k: w[k] for k in ("hard_hat", "vest", "goggles", "zone")} for w in spec["workers"]},
            "sections": {s["id"]: s["present"] for s in spec["sections"]},
            "doors": {dr["id"]: dr["sign"] for dr in spec["doors"]}}


def alt_site(spec):
    base = world_site(spec)
    out = []
    for w in spec["workers"]:
        attrs = (["hard_hat", "goggles"] if w["head_hidden"] else []) + (["vest"] if w["body_hidden"] else [])
        for a in attrs:
            x = copy.deepcopy(base)
            x["workers"][w["id"]][a] = not x["workers"][w["id"]][a]
            out.append(x)
    for s in spec["sections"]:
        if s["hidden"]:
            x = copy.deepcopy(base)
            x["sections"][s["id"]] = not x["sections"][s["id"]]
            out.append(x)
    for dr in spec["doors"]:
        if dr["hidden"]:
            for k in sorted(SIGNS):
                if k != dr["sign"]:
                    x = copy.deepcopy(base)
                    x["doors"][dr["id"]] = k
                    out.append(x)
    return out


def world_plan(spec):
    return {"exits": {e["id"]: e["blocked"] for e in spec["exits"]},
            "stations": {s["id"]: (s["zone"], s["stocked"]) for s in spec["stations"]},
            "spills": [s["zone"] for s in spec["spills"]],
            "lanes": {ln["id"]: ln["obstructed"] for ln in spec["lanes"]},
            "exit_zone": {e["id"]: plan_exit_zone(spec, e) for e in spec["exits"]}}


def plan_exit_zone(spec, e):
    for z in spec["zones"]:
        b = z["box"]
        if b[0] - 2 <= e["x"] <= b[2] + 2 and b[1] - 2 <= e["y"] <= b[3] + 2:
            return z["id"]
    return None


def alt_plan(spec):
    base = world_plan(spec)
    out = []
    for e in spec["exits"]:
        if e["hidden"]:
            x = copy.deepcopy(base)
            x["exits"][e["id"]] = not x["exits"][e["id"]]
            out.append(x)
    for s in spec["stations"]:
        if s["hidden"]:
            x = copy.deepcopy(base)
            x["stations"][s["id"]] = (s["zone"], not s["stocked"])
            out.append(x)
    for ln in spec["lanes"]:
        if ln["hidden"]:
            x = copy.deepcopy(base)
            x["lanes"][ln["id"]] = not x["lanes"][ln["id"]]
            out.append(x)
    for z in spec["zones"]:
        if z["partial"]:
            x = copy.deepcopy(base)
            x["spills"] = x["spills"] + [z["id"]]          # one more spill in the part out of frame
            out.append(x)
            hid = [s for s in spec["spills"] if s["zone"] == z["id"] and s["hidden"]]
            if hid:
                x = copy.deepcopy(base)
                x["spills"].remove(z["id"])
                out.append(x)
    return out


def single(xs):
    xs = sorted(xs)
    return "NONE" if not xs else (xs[0] if len(xs) == 1 else None)


def answer_site(spec, wd, q):
    k = q["kind"]
    W_ = wd["workers"]
    if k == "ppe_which":
        return single(i for i, w in W_.items() if not w[q["attr"]])
    if k == "ppe_count":
        return sum(1 for w in W_.values() if w[q["attr"]] == q["has"])
    if k == "ppe_noul":
        return W_[q["worker"]][q["attr"]]
    if k == "ppe_rule":
        req = q["rules"]
        return single(i for i, w in W_.items() if any(not w[a] for a in req[w["zone"]]))
    if k == "ppe_rule_noul":
        w = W_[q["worker"]]
        return all(w[a] for a in q["rules"][w["zone"]])
    if k == "guardrail_which":
        return single(i for i, p in wd["sections"].items() if not p)
    if k == "guardrail_noul":
        return wd["sections"][q["section"]]
    if k == "sign_which":
        return single(i for i, s in wd["doors"].items() if s != q["checklist"][i])
    if k == "sign_noul":
        return wd["doors"][q["door"]] == q["checklist"][q["door"]]
    raise KeyError(k)


def answer_plan(spec, wd, q):
    k = q["kind"]
    if k == "exit_which":
        return single(i for i, b in wd["exits"].items() if b)
    if k == "exit_noul":
        return wd["exits"][q["exit"]]
    if k == "station_which":
        return single(i for i, (z, s) in wd["stations"].items() if not s)
    if k == "station_noul":
        return wd["stations"][q["station"]][1]
    if k == "spill_count":
        return sum(1 for z in wd["spills"] if z == q["zone"])
    if k == "lane_which":
        return single(i for i, b in wd["lanes"].items() if b)
    if k == "zone_score":
        z = q["zone"]
        n = sum(1 for zz in wd["spills"] if zz == z)
        n += sum(1 for i, (zz, s) in wd["stations"].items() if zz == z and not s)
        n += sum(1 for i, b in wd["exits"].items() if b and wd["exit_zone"][i] == z)
        return n
    raise KeyError(k)


def settle_scene(spec, q):
    if spec["scene"] == "site_view":
        base, alts, fn = world_site(spec), alt_site(spec), answer_site
    else:
        base, alts, fn = world_plan(spec), alt_plan(spec), answer_plan
    truth = fn(spec, base, q)
    if truth is None:
        return None, "ill_posed"
    for w in alts:
        if fn(spec, w, q) != truth:
            return None, "insufficient_evidence"
    if truth == "NONE":
        return None, "false_premise"
    return truth, None


# ------------------------------------------------------------------------------------------------ rendered questions
SITE_LINES = [
    "Inspection view {ref} of the warehouse floor, round {n}.",
    "Camera still {ref} from the site walk, check number {n}.",
    "Site view {ref} captured for safety round {n}.",
    "Warehouse snapshot {ref} (inspection {n}).",
    "Frame {ref} from the floor camera used in audit {n}.",
    "Illustrated site view {ref} for toolbox talk {n}.",
]
PLAN_LINES = [
    "Marked-up floor plan {ref} from fire-safety walk {n}.",
    "Plan {ref} annotated during hazard survey {n}.",
    "Emergency layout {ref}, markup from inspection {n}.",
    "Floor plan {ref} with hazards marked for review {n}.",
    "Annotated plan {ref} returned by the marshal, round {n}.",
    "Hazard map {ref} for audit {n}.",
]
Q_PPE_WHICH = [
    "Which worker is not wearing {item}?",
    "Which tagged worker is missing {item}?",
    "Who on the floor has no {item_bare}?",
    "Which worker should be stopped for not having {item} on?",
    "Pick the worker without {item}.",
    "Which worker lacks {item}?",
]
Q_PPE_COUNT_HAS = [
    "How many workers are wearing {item}?",
    "Count the workers who have {item} on.",
    "How many of the people on the floor wear {item}?",
    "What number of workers can be seen with {item}?",
    "How many tagged workers are in {item}?",
    "Of the workers shown, how many wear {item}?",
]
Q_PPE_COUNT_NOT = [
    "How many workers are not wearing {item}?",
    "Count the workers without {item}.",
    "How many people on the floor are missing {item}?",
    "What number of workers have no {item_bare}?",
    "How many tagged workers go without {item}?",
    "Of the workers shown, how many lack {item}?",
]
Q_PPE_NOUL = [
    "Is worker {w} wearing {item}?",
    "Does {w} have {item} on?",
    "Can you see {item} on worker {w}?",
    "Has worker {w} put on {item}?",
    "Is {item} being worn by {w}?",
    "Does worker {w} wear {item}?",
]
PPE_RULE_LINES = [
    "PPE rules by zone: {rules}. A worker must wear everything required in the zone they stand in.",
    "Zone PPE requirements: {rules}. Anyone in a zone without all of its items is in breach.",
    "Required equipment per floor zone: {rules}.",
    "Site rule card: {rules}; a person in a zone must carry all of that zone's PPE.",
    "PPE matrix: {rules}. Missing any listed item in your zone is a breach.",
    "Zone requirements (all items mandatory): {rules}.",
]
Q_PPE_RULE = [
    "Which worker breaks the PPE rule for their zone?",
    "Who is in breach of the zone PPE requirements?",
    "Which worker is missing equipment their zone requires?",
    "Under the zone rules, which worker is not compliant?",
    "Which tagged worker would the inspector write up?",
    "Which worker does not meet the PPE matrix?",
]
Q_PPE_RULE_NOUL = [
    "Is worker {w} compliant with the PPE rule for the zone they are in?",
    "Does {w} wear everything their zone requires?",
    "Does worker {w} meet the zone PPE requirements?",
    "Is {w} fully equipped for their zone?",
    "Would worker {w} pass the zone PPE check?",
    "Is worker {w} within the PPE rules?",
]
Q_GUARD_WHICH = [
    "Which mezzanine edge section has no guardrail?",
    "Which labelled section of the upper-level edge is left unguarded?",
    "Where is the guardrail missing along the mezzanine (section label)?",
    "Which edge section is open with no rails?",
    "Name the mezzanine section without a guardrail.",
    "Which section of the platform edge lacks its rails?",
]
Q_GUARD_NOUL = [
    "Is mezzanine section {s} protected by a guardrail?",
    "Does section {s} of the upper edge have its guardrail?",
    "Is there a guardrail along section {s}?",
    "Is edge section {s} guarded?",
    "Has section {s} got its rails in place?",
    "Is the mezzanine edge at {s} railed?",
]
SIGN_CHECK_LINES = [
    "Signage checklist (sign required above each door): {items}.",
    "Door sign register: {items}.",
    "Required signs per the site plan: {items}.",
    "Each door must carry this sign: {items}.",
    "Signs expected at the doors: {items}.",
    "Checklist of door signage: {items}.",
]
Q_SIGN_WHICH = [
    "Which door's sign does not match the checklist?",
    "Above which door is the wrong sign posted?",
    "Which door carries a sign other than the one the checklist requires?",
    "Where does the signage differ from the register? Give the door.",
    "Which door needs its sign replaced to match the checklist?",
    "Which door fails the signage check?",
]
Q_SIGN_NOUL = [
    "Does door {d} show the sign the checklist requires?",
    "Is the sign above {d} the required one?",
    "Does the signage at door {d} match the checklist?",
    "Is door {d} correctly signed per the checklist?",
    "Would door {d} pass the signage check?",
    "Is the checklist sign posted above {d}?",
]
Q_EXIT_WHICH = [
    "Which emergency exit is blocked?",
    "Which exit on the plan has something in front of it?",
    "Which exit is obstructed according to the markup?",
    "Name the exit that is not clear.",
    "Which exit route is blocked at the door?",
    "Which exit does the marshal have to clear?",
]
Q_EXIT_NOUL = [
    "Is exit {e} blocked?",
    "Is there an obstruction in front of exit {e}?",
    "Is exit {e} clear?",
    "Does exit {e} have something stacked in front of it?",
    "Is exit {e} obstructed on the plan?",
    "Can exit {e} be used as it is marked?",
]
Q_STATION_WHICH = [
    "Which extinguisher station is empty?",
    "Which fire extinguisher station has no extinguisher in it?",
    "At which station is the extinguisher missing?",
    "Which station needs an extinguisher put back?",
    "Name the empty extinguisher point.",
    "Which extinguisher point is unstocked?",
]
Q_STATION_NOUL = [
    "Is station {s} stocked with an extinguisher?",
    "Does station {s} hold its extinguisher?",
    "Is there an extinguisher at {s}?",
    "Is extinguisher point {s} stocked?",
    "Has {s} got an extinguisher in place?",
    "Is station {s} in order (extinguisher present)?",
]
Q_SPILL = [
    "How many marked spills are in Zone {z}?",
    "Count the spill markers inside Zone {z}.",
    "How many spills does the plan show in Zone {z}?",
    "What is the number of marked spills in Zone {z}?",
    "In Zone {z}, how many spills are flagged?",
    "How many spill markers fall inside Zone {z}?",
]
Q_LANE = [
    "Which forklift lane is obstructed?",
    "Which lane has goods blocking it?",
    "Which forklift route is not clear?",
    "Name the forklift lane with an obstruction.",
    "Which lane on the plan is blocked?",
    "Which forklift lane needs clearing?",
]
ZONE_RULES = [
    "Zone problem count = marked spills + empty extinguisher stations + blocked exits located in that zone.",
    "Each spill, each station without an extinguisher and each blocked exit inside a zone counts as one problem for that zone.",
    "Add up, for the zone, its marked spills, its unstocked extinguisher points and its obstructed exits.",
    "Problems per zone: every spill marker, every empty extinguisher station and every blocked exit in the zone.",
    "Score a zone by counting spills, empty stations and blocked exits inside it.",
    "One point per marked spill, per empty extinguisher station and per blocked exit in the zone.",
]
Q_ZONE = [
    "How many problems does Zone {z} have under the rule?",
    "What is Zone {z}'s problem count?",
    "Score Zone {z} using the rule.",
    "Under the rule, how many problems are in Zone {z}?",
    "What problem count does the rule give Zone {z}?",
    "How many problems are counted for Zone {z}?",
]
REQ_PHRASE = {"hard_hat": "hard hat", "vest": "high-visibility vest", "goggles": "safety goggles"}


def scene_question(rng, spec, kind):
    q = {"kind": kind}
    if spec["scene"] == "site_view":
        ws = spec["workers"]
        if kind in ("ppe_which", "ppe_count"):
            q["attr"] = rng.choice(["hard_hat", "hard_hat", "vest", "goggles"])
            if kind == "ppe_count":
                q["has"] = rng.random() < 0.5
        elif kind == "ppe_noul":
            q.update(attr=rng.choice(["hard_hat", "vest", "goggles"]), worker=rng.choice(ws)["id"])
        elif kind in ("ppe_rule", "ppe_rule_noul"):
            rules = {}
            for z in spec["zones"]:
                rules[z] = sorted(rng.sample(["hard_hat", "vest", "goggles"], rng.choice([1, 1, 2, 2, 3])))
            q["rules"] = rules
            if kind == "ppe_rule_noul":
                q["worker"] = rng.choice(ws)["id"]
        elif kind == "guardrail_noul":
            q["section"] = rng.choice(spec["sections"])["id"]
        elif kind in ("sign_which", "sign_noul"):
            chk = {dr["id"]: dr["sign"] for dr in spec["doors"]}
            if rng.random() < 0.75:
                dr = rng.choice(spec["doors"])
                chk[dr["id"]] = rng.choice([k for k in sorted(SIGNS) if k != dr["sign"]])
            q["checklist"] = chk
            if kind == "sign_noul":
                q["door"] = rng.choice(spec["doors"])["id"]
    else:
        if kind == "exit_noul":
            q["exit"] = rng.choice(spec["exits"])["id"]
        elif kind == "station_noul":
            q["station"] = rng.choice(spec["stations"])["id"]
        elif kind in ("spill_count", "zone_score"):
            q["zone"] = rng.choice(spec["zones"])["id"]
        elif kind == "lane_which" and len(spec["lanes"]) < 2:
            return None
    return q


SITE_KINDS = {"ppe_which": 1.4, "ppe_count": 1.4, "ppe_noul": 1.0, "ppe_rule": 1.2, "ppe_rule_noul": 0.9,
              "guardrail_which": 0.9, "guardrail_noul": 0.6, "sign_which": 1.0, "sign_noul": 0.8}
PLAN_KINDS = {"exit_which": 1.0, "exit_noul": 0.8, "station_which": 1.0, "station_noul": 0.7, "spill_count": 1.2,
              "lane_which": 0.8, "zone_score": 1.0}
SCENE_FAMILY = {"ppe_which": "safety_ppe", "ppe_count": "safety_ppe", "ppe_noul": "safety_ppe",
                "ppe_rule": "safety_ppe_rule", "ppe_rule_noul": "safety_ppe_rule", "guardrail_which": "safety_guardrail",
                "guardrail_noul": "safety_guardrail", "sign_which": "safety_sign_check", "sign_noul": "safety_sign_check",
                "exit_which": "safety_blocked_exit", "exit_noul": "safety_blocked_exit",
                "station_which": "safety_extinguisher", "station_noul": "safety_extinguisher",
                "spill_count": "safety_spill", "lane_which": "safety_forklift_lane", "zone_score": "safety_zone_score"}
SCENE_DIFF = {"ppe_which": 3, "ppe_count": 3, "ppe_noul": 2, "ppe_rule": 5, "ppe_rule_noul": 4, "guardrail_which": 3,
              "guardrail_noul": 2, "sign_which": 4, "sign_noul": 3, "exit_which": 3, "exit_noul": 2, "station_which": 3,
              "station_noul": 2, "spill_count": 3, "lane_which": 3, "zone_score": 5}


def scene_row(rng, spec, q, gold, reason, img_rel, sha, split, rid, parent=None):
    k = q["kind"]
    n = rng.randint(2, 99)
    if spec["scene"] == "site_view":
        state = {"view": pick(rng, SITE_LINES, split).format(ref=spec["ref"], n=n),
                 "workers_on_shift": sorted(w["id"] for w in spec["workers"])}
    else:
        state = {"plan": pick(rng, PLAN_LINES, split).format(ref=spec["ref"], n=n)}
    field = None
    g = gold
    if k in ("ppe_which", "ppe_count", "ppe_noul"):
        item = {"hard_hat": "a hard hat", "vest": "a high-visibility vest", "goggles": "safety goggles"}[q["attr"]]
        bare = REQ_PHRASE[q["attr"]]
        if k == "ppe_which":
            field = {"type": "choice", "question": pick(rng, Q_PPE_WHICH, split).format(item=item, item_bare=bare),
                     "options": [{"key": w.lower(), "text": w, "description": ""} for w in state["workers_on_shift"]]}
            g = None if gold is None else gold.lower()
        elif k == "ppe_count":
            bank = Q_PPE_COUNT_HAS if q["has"] else Q_PPE_COUNT_NOT
            truth = answer_site(spec, world_site(spec), q)
            nw = len(spec["workers"])
            field = {"type": "choice", "question": pick(rng, bank, split).format(item=item, item_bare=bare),
                     "options": [{"key": f"n_{v}", "text": str(v), "description": ""} for v in range(0, nw + 1)]}
            g = None if gold is None else f"n_{gold}"
        else:
            field = {"type": "noul", "question": pick(rng, Q_PPE_NOUL, split).format(w=q["worker"], item=item)}
    elif k in ("ppe_rule", "ppe_rule_noul"):
        rules = "; ".join(f"Zone {z}: " + " + ".join(REQ_PHRASE[a] for a in q["rules"][z]) for z in spec["zones"])
        state["ppe_rules"] = pick(rng, PPE_RULE_LINES, split).format(rules=rules)
        if k == "ppe_rule":
            field = {"type": "choice", "question": pick(rng, Q_PPE_RULE, split),
                     "options": [{"key": w.lower(), "text": w, "description": ""} for w in state["workers_on_shift"]]}
            g = None if gold is None else gold.lower()
        else:
            field = {"type": "noul", "question": pick(rng, Q_PPE_RULE_NOUL, split).format(w=q["worker"])}
    elif k == "guardrail_which":
        field = {"type": "choice", "question": pick(rng, Q_GUARD_WHICH, split),
                 "options": [{"key": s["id"].lower(), "text": s["id"], "description": ""} for s in spec["sections"]]}
        g = None if gold is None else gold.lower()
    elif k == "guardrail_noul":
        field = {"type": "noul", "question": pick(rng, Q_GUARD_NOUL, split).format(s=q["section"])}
    elif k in ("sign_which", "sign_noul"):
        items = "; ".join(f"{d_}: {SIGN_TEXT[s_].upper()}" for d_, s_ in sorted(q["checklist"].items()))
        state["signage_checklist"] = pick(rng, SIGN_CHECK_LINES, split).format(items=items)
        if k == "sign_which":
            field = {"type": "choice", "question": pick(rng, Q_SIGN_WHICH, split),
                     "options": [{"key": dr["id"].lower(), "text": f"door {dr['id']}", "description": ""} for dr in spec["doors"]]}
            g = None if gold is None else gold.lower()
        else:
            field = {"type": "noul", "question": pick(rng, Q_SIGN_NOUL, split).format(d=q["door"])}
    elif k in ("exit_which", "exit_noul"):
        if k == "exit_which":
            field = {"type": "choice", "question": pick(rng, Q_EXIT_WHICH, split),
                     "options": [{"key": e["id"].lower(), "text": f"exit {e['id']}", "description": ""} for e in spec["exits"]]}
            g = None if gold is None else gold.lower()
        else:
            field = {"type": "noul", "question": pick(rng, Q_EXIT_NOUL, split).format(e=q["exit"])}
            if "clear" in field["question"] or "Can exit" in field["question"]:
                q["negated"] = True            # the wording asks whether the exit is usable, not whether it is blocked
                g = None if gold is None else (not gold)
    elif k in ("station_which", "station_noul"):
        if k == "station_which":
            field = {"type": "choice", "question": pick(rng, Q_STATION_WHICH, split),
                     "options": [{"key": option_key(s["id"]), "text": s["id"], "description": ""} for s in spec["stations"]]}
            g = None if gold is None else option_key(gold)
        else:
            field = {"type": "noul", "question": pick(rng, Q_STATION_NOUL, split).format(s=q["station"])}
    elif k == "spill_count":
        truth = answer_plan(spec, world_plan(spec), q)
        start = max(0, truth - rng.randint(0, 3))
        field = {"type": "choice", "question": pick(rng, Q_SPILL, split).format(z=q["zone"]),
                 "options": [{"key": f"n_{v}", "text": str(v), "description": ""} for v in range(start, start + 5)]}
        g = None if gold is None else f"n_{gold}"
    elif k == "lane_which":
        field = {"type": "choice", "question": pick(rng, Q_LANE, split),
                 "options": [{"key": ln["id"].lower(), "text": f"lane {ln['id']}", "description": ""} for ln in spec["lanes"]]}
        g = None if gold is None else gold.lower()
    elif k == "zone_score":
        state["rule"] = pick(rng, ZONE_RULES, split)
        truth = answer_plan(spec, world_plan(spec), q)
        top = max(5, truth + 1)
        field = {"type": "score", "question": pick(rng, Q_ZONE, split).format(z=q["zone"]),
                 "levels": [{"value": i, "description": f"{i} problem{'s' if i != 1 else ''}"} for i in range(min(10, top + 1))]}
    assert g == scene_gold_key(spec, q, gold)
    diff = SCENE_DIFF[k] + (1 if gold is None and reason == "insufficient_evidence" else 0)
    return {
        "id": rid, "source": "I", "dataset": "safety_synthetic", "family": SCENE_FAMILY[k], "difficulty": max(2, min(5, diff)),
        "state": state, "images": [img_rel], "field": field, "gold": g,
        "unknown_reason": None if g is not None else reason, "gold_kind": "constructed", "parent_id": parent,
        "provenance": {"licence": "generated", "image_licences": ["generated"],
                       "renderer": "our own PIL drawing (scripts/p3/gen_safety.py), no third-party content",
                       "upstream_split": "generated", "split": split, "image_sha256": [sha], "scene": spec["scene"],
                       "question": q, "spec": spec, "generator": "scripts/p3/gen_safety.py", "seed": spec["_seed"]},
    }


def scene_gold_key(spec, q, gold):
    """Gold as the row stores it (lower-case option keys, negated noul wordings)."""
    if gold is None:
        return None
    k = q["kind"]
    if k in ("ppe_which", "ppe_rule", "guardrail_which", "sign_which", "exit_which", "lane_which"):
        return gold.lower()
    if k == "station_which":
        return option_key(gold)
    if k in ("ppe_count", "spill_count"):
        return f"n_{gold}"
    if k == "exit_noul" and q.get("negated"):
        return not gold
    return gold


def scene_has_witness(spec, q) -> bool:
    if spec["scene"] == "site_view":
        base, alts, fn = world_site(spec), alt_site(spec), answer_site
    else:
        base, alts, fn = world_plan(spec), alt_plan(spec), answer_plan
    truth = fn(spec, base, q)
    return any(fn(spec, w, q) != truth for w in alts)


def oi_gold_key(q, gold):
    if gold is None:
        return None
    return f"n_{gold}" if q["kind"] == "count" else gold


def gen_rendered_scene(seed, i, split, want_unknown, scene=None):
    rng = rng_for(seed, "scene", i)
    scene = scene or ("site_view" if rng.random() < 0.55 else "floor_plan")
    if scene == "site_view":
        spec = build_site(rng, split)
        if want_unknown or rng.random() < 0.12:
            mode = rng.choice(["occluder", "occluder", "illegible", "crop"])
            if mode == "occluder":
                w = rng.choice(spec["workers"])
                wb = worker_boxes(spec)[w["id"]]
                full = rng.random() < 0.45
                top = wb["head"][1] - 10 if full else wb["body"][1] + rng.uniform(0, 12)
                spec["occluders"].append({"box": [round(wb["all"][0] - 16), round(top), round(wb["all"][2] + 16),
                                                  round(wb["all"][3] + 4)]})
            elif mode == "illegible":
                rng.choice(spec["doors"])["legible"] = False
            else:
                side = rng.choice(["left", "right"])
                geo = layout_site(spec)
                box = geo["sections"][0] if side == "left" else geo["sections"][-1]
                cut = box[0] + (box[2] - box[0]) * rng.uniform(0.3, 0.7)
                spec["crop"] = [round(cut), 0, spec["W"], spec["H"]] if side == "left" else [0, 0, round(cut), spec["H"]]
        site_hidden(spec)
    else:
        spec = build_plan(rng, split)
        if want_unknown or rng.random() < 0.12:
            side = rng.choice(["left", "right", "bottom"])
            z = rng.choice([z for z in spec["zones"]])
            x0, y0, x1, y1 = spec["bounds"]
            if side == "right":
                spec["crop"] = [0, 0, round(x0 + (x1 - x0) * rng.uniform(0.72, 0.88)), spec["H"]]
            elif side == "left":
                spec["crop"] = [round(x0 + (x1 - x0) * rng.uniform(0.12, 0.28)), 0, spec["W"], spec["H"]]
            else:
                spec["crop"] = [0, 0, spec["W"], round(y0 + (y1 - y0) * rng.uniform(0.72, 0.86))]
        plan_hidden(spec)
    spec["_seed"] = f"{seed}/scene/{i}"
    return spec, rng


def render_rendered(spec):
    img = render_site(spec) if spec["scene"] == "site_view" else render_plan(spec)
    out, _ = apply_post(img, spec["post"], f"safety-{spec['ref']}-{spec['palette']}",
                        fill=(200, 200, 196) if spec["scene"] == "site_view" else (250, 250, 247))
    if spec["crop"]:
        out = out.crop(tuple(spec["crop"]))
    return out


def rendered_items(seed, split, count, start_idx, unknown_share=0.165, per_scene=2, img_dir=None):
    bench = __import__("gen_image_joint").bench_image_shas()
    img_dir = img_dir or (IMG_DIR if split == "train" else IMG_DIR / "heldout")
    prefix = "p3-I-safety" if split == "train" else "p3-hf1-I-safety"
    rows, stats = [], Counter()
    i, n_unk = 0, 0
    while len(rows) < count and i < count * 8:
        want_unknown = n_unk < unknown_share * (len(rows) + per_scene)
        spec, rng = gen_rendered_scene(seed, i, split, want_unknown)
        i += 1
        kinds = dict(SITE_KINDS if spec["scene"] == "site_view" else PLAN_KINDS)
        picked = []
        for _ in range(30):
            if len(picked) >= per_scene or not kinds:
                break
            k = rng.choices(list(kinds), weights=list(kinds.values()))[0]
            q = scene_question(rng, spec, k)
            if q is None:
                kinds.pop(k)
                continue
            gold, reason = settle_scene(spec, q)
            if reason == "ill_posed":
                continue
            if (gold is None) != want_unknown and not picked and rng.random() < 0.85:
                continue
            if gold is None and not want_unknown:
                continue
            kinds.pop(k)
            picked.append((q, gold, reason))
        if not picked:
            stats["no_question"] += 1
            continue
        img = render_rendered(spec)
        st = store_image(encode_jpeg(img, spec["post"]["jpeg"]), img_dir, bench)
        if st is None:
            stats["bench_sha"] += 1
            continue
        for q, gold, reason in picked:
            if len(rows) >= count:
                break
            row = scene_row(rng, spec, q, gold, reason, st[0], st[1], split, f"{prefix}-{start_idx + len(rows):06d}")
            n_unk += row["gold"] is None
            rows.append(row)
        stats["scenes"] += 1
    return rows, stats


def build(count, seed, split, oi_share=0.33, img_dir=None):
    rng = rng_for(seed, "openimages")
    n_oi = int(round(count * oi_share))
    oi = oi_items(rng, split, n_oi, 0)
    rend, stats = rendered_items(seed, split, count - len(oi), len(oi), img_dir=img_dir)
    stats["openimages"] = len(oi)
    return oi + rend, stats


def build_variants(parents_path, per_parent):
    """Constructed variants of rendered-scene parents: same question kind and scene type, fresh scene keyed by the
    parent id. Open Images parents have no variant path (the photo and its labels are fixed)."""
    bench = __import__("gen_image_joint").bench_image_shas()
    rows = []
    for par in (json.loads(l) for l in open(parents_path) if l.strip()):
        if par.get("dataset") != "safety_synthetic":
            continue
        pv = par["provenance"]
        kind, scene = pv["question"]["kind"], pv["scene"]
        split = pv.get("split", "train")
        for v in range(per_parent):
            for attempt in range(30):
                spec, rng = gen_rendered_scene(f"variant/{par['id']}/{v}", attempt, split, par["gold"] is None, scene=scene)
                q = scene_question(rng, spec, kind)
                if q is None:
                    continue
                gold, reason = settle_scene(spec, q)
                if reason == "ill_posed" or (gold is None) != (par["gold"] is None):
                    continue
                st = store_image(encode_jpeg(render_rendered(spec), spec["post"]["jpeg"]),
                                 IMG_DIR if split == "train" else IMG_DIR / "heldout", bench)
                if st is None:
                    continue
                rows.append(scene_row(rng, spec, q, gold, reason, st[0], st[1], split, f"{par['id']}-v{v}", parent=par["id"]))
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--oi-share", type=float, default=0.33)
    ap.add_argument("--variant-of", default=None)
    ap.add_argument("--per-parent", type=int, default=2)
    a = ap.parse_args()
    if a.variant_of:
        rows = build_variants(a.variant_of, a.per_parent)
        out = a.out or str(ROOT / "data" / "p3" / "candidates" / "I-safety-variants.jsonl")
        print(json.dumps({"rows": write(out, rows), "out": out}))
        return
    write_sources()
    split = "heldout" if a.heldout else "train"
    seed = a.seed or (HELDOUT_SEED if a.heldout else SEED)
    out = a.out or str(HELDOUT_OUT if a.heldout else OUT)
    rows, stats = build(a.count, seed, split, a.oi_share)
    n = write(out, rows)
    print(json.dumps({"rows": n, "out": out, "stats": dict(stats), "unknown": sum(r["gold"] is None for r in rows),
                      "by_family": dict(Counter(r["family"] for r in rows)),
                      "by_dataset": dict(Counter(r["dataset"] for r in rows)),
                      "by_reason": dict(Counter(r["unknown_reason"] for r in rows)),
                      "difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
                      "images": len({r["images"][0] for r in rows})}, indent=1))


if __name__ == "__main__":
    main()
