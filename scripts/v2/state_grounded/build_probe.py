"""Build `data/manifests/decision-v2-state-probe.jsonl` -- 200 HUMAN-LABELLED probe decisions.

No teacher touches this file.  Every gold answer follows from metadata a human already verified:

  * Open Images V7 **human-verified image-level labels**: `Confidence == 1` means a person
    confirmed the thing is in the photograph, `Confidence == 0` means a person confirmed it is
    not.  `data/decision-v2-raw/openimages_v2/selection.jsonl` carries both lists per photo.
  * ABO **structured listing fields**: the curated `product_type`, `color` and `material` of the
    listing the photograph belongs to (`scripts/v1/abo_taxonomy.py` vocabularies), plus the fields
    the ABO README names as invisible in a catalogue photo (model number, weight, dimensions).

The questions are built with the SAME templates the training rows use
(`scripts/v2/state_grounded/templates.py`), so the probe measures the families rather than a
separate style of wording.  Each row carries a `rationale` saying which verified fact makes the
gold answer right, and the whole file is `partition: "test"`, `heldout_family: true`.

Run:
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/build_probe.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from v2.common import RAW, ROOT, read_jsonl, repo_relative, stable_partition, verified_license
from v2.state_grounded import convert as C
from v2.state_grounded import templates as T

OUT_PATH = ROOT / "data" / "manifests" / "decision-v2-state-probe.jsonl"
SEED = 4471

QUOTA = {
    "claim_supported": 33,
    "which_field_conflicts": 34,
    "count_matches": 33,
    "label_text_matches": 33,
    "condition_report": 33,
    "instruction_changes_answer": 34,     # 17 pairs
}

# Open Images labels that are about printed text being present.  A photo where a person verified
# one of these is ABSENT cannot show the serial the state asks for.
TEXT_LABELS_ABSENT = ["Signage", "Label", "Sign", "Street sign", "Billboard", "Poster",
                      "Vehicle registration plate", "Number", "Electronic signage", "Neon sign",
                      "Traffic sign", "Newspaper", "Book"]
# A stop sign carries one word, and it is the same word everywhere Open Images photographs one.
STOP_SIGN = "Stop sign"
# The one condition Open Images verifies often enough to use.
RUST = "Rust"

# ABO colours/materials that read unambiguously in a catalogue photograph.  `multicoloured`,
# `transparent` and fibre-composition materials are left out of the probe on purpose.
PROBE_COLOURS = {"black", "white", "red", "blue", "green", "brown", "grey", "pink", "beige",
                 "gold", "silver", "yellow", "orange", "purple", "ivory", "cream"}
PROBE_MATERIALS = {"wood", "metal", "plastic", "glass", "leather", "fabric", "ceramic", "stone",
                   "canvas", "silicone", "steel", "stainless steel", "rubber", "paper", "marble",
                   "suede", "faux leather", "wicker", "bamboo", "concrete"}


def load_openimages():
    """-> [{photo dict with the raw verified label lists attached}] (only photos on disk)."""
    manifest = {r["image_id"]: r for r in read_jsonl(RAW / "openimages_v2" / "manifest.jsonl")}
    spdx, rel = C.LICENSES["openimages_v2"]
    licence = verified_license(Path(rel), spdx)
    annotation = verified_license(Path(C.ANNOTATION_LICENSE[1]), C.ANNOTATION_LICENSE[0])
    out = []
    for row in read_jsonl(RAW / "openimages_v2" / "selection.jsonl"):
        m = manifest.get(row["image_id"])
        if not m or not (ROOT / m["image"]).is_file():
            continue
        group = f"openimages_v2:{row['image_id']}"
        out.append({
            "source": "openimages_v2", "key": row["image_id"],
            "partition": stable_partition(group, seed=C.PARTITION_SEED),
            "image": {"image": repo_relative(m["image"]), "sha256": m["sha256"],
                      "width": m["width"], "height": m["height"]},
            "licence": licence, "annotation_license": annotation,
            "verified_present": list(row.get("labels_present", [])),
            "verified_absent": list(row.get("labels_absent", [])),
            "present": [C._phrase(x) for x in row.get("labels_present_concrete", [])],
            "absent": [C._phrase(x) for x in row.get("labels_absent_concrete", [])],
            "bucket": row.get("bucket_label", ""),
        })
    return out


def load_abo():
    """ABO photographs from the v1 TEST partition only.

    Those images were held out of every v1/v1.1 training run (they are the v1 ABO exam), and
    `convert.py` never uses a non-train ABO listing, so no probe photo is a training photo of the
    model being probed or of the state-grounded source.
    """
    meta = C.load_abo_meta()
    spdx, rel = C.LICENSES["abo"]
    licence = verified_license(Path(rel), spdx)
    items, not_test = {}, set()
    for rec in read_jsonl(ROOT / "data" / "decision-v1" / "abo" / "records.jsonl"):
        hit = C.ABO_ID.match(rec["id"].split(":", 1)[1])
        if not hit or rec["id"].split(":", 1)[1].startswith(("same", "reference_")):
            continue
        item = hit.group(1)
        if rec["partition"] != "test":
            not_test.add(item)
            continue
        items.setdefault(item, rec["images"][0])
    keep = []
    for item, image in sorted(items.items()):
        if item in not_test or item not in meta or not (ROOT / image["image"]).is_file():
            continue
        info = meta[item]
        attrs = {C.ABO_ATTR_KEYS[k]: v for k, v in info["attrs"].items() if k in C.ABO_ATTR_KEYS}
        if attrs.get("colour") not in PROBE_COLOURS:
            attrs.pop("colour", None)
        if attrs.get("material") not in PROBE_MATERIALS:
            attrs.pop("material", None)
        for k in ("pattern", "shape", "finish"):
            attrs.pop(k, None)
        if not attrs.get("product_type"):
            continue
        keep.append({
            "source": "abo", "key": item, "partition": "test",
            "image": {"image": repo_relative(image["image"]), "sha256": image["sha256"],
                      "width": image["width"], "height": image["height"]},
            "licence": licence,
            "inputs": {"id": item, "source": "abo", "subject": attrs["product_type"],
                       "present": [attrs["product_type"]], "absent": [], "attrs": attrs,
                       "hidden": info.get("hidden") or {}, "peers": [], "title": "",
                       "condition_is_new": True},
        })
    return keep


def row(photo, family, made, gold, rationale, index, pair_id=None, pair_role=None,
        derived_from=""):
    record = {
        "id": f"state-probe:{family}:{photo['source']}-{photo['key']}#{index}",
        "source": "state_grounded_probe",
        "source_group": f"{photo['source']}:{photo['key']}",
        "partition": "test",
        "heldout_family": True,
        "family": family,
        "probe_family": family,
        "source_split": C.SOURCE_SPLIT[photo["source"]],
        "license": photo["licence"],
        "images": [photo["image"]],
        "request": {
            "schema_version": "1.0",
            "request_id": f"state-probe-{family}-{photo['source']}-{photo['key']}-{index}"[:128],
            "state": made["state"],
            "fields": made["fields"],
        },
        "target": None if gold == "unknown" else gold,
        "abstention_cause": "insufficient_evidence" if gold == "unknown" else None,
        "gold": gold,
        "pseudo_label": "human-derived-v2-state-probe",
        "template_id": made["template_id"],
        "photo_source": photo["source"],
        "state_kind": made["state_kind"],
        "state_path": made["state_path"],
        "claim_intent": made["intent"],
        "claim_basis": made["basis"],
        "rationale": rationale,
        "derived_from": derived_from,
        "photo_partition": photo.get("partition"),
    }
    if pair_id:
        record["pair_id"] = pair_id
        record["pair_role"] = pair_role
    if photo["source"] == "openimages_v2":
        record["annotation_license"] = photo["annotation_license"]
    return record


# --------------------------------------------------------------------------- families
def build_claim_supported(oi, abo, rng, quota):
    """Verified-present object -> true; verified-absent object -> false; a field no photograph can
    show -> unknown."""
    rows = []
    want_true = quota // 3 + quota % 3
    want_false = quota // 3
    want_unknown = quota // 3

    for photo in oi:
        if len(rows) >= want_true:
            break
        hits = [C._phrase(x) for x in photo["present"]]
        if not hits:
            continue
        thing = hits[0]
        r = random.Random(f"{SEED}\0claim-true\0{photo['key']}")
        made = T.make_claim_supported(
            {"id": photo["key"], "subject": thing, "present": [thing], "absent": [],
             "attrs": {}, "hidden": {}, "peers": []}, r, intent="true", kind="object")
        rows.append(row(photo, "claim_supported", made, True,
                        f"A human verified the Open Images label {photo['present'][0]!r} as "
                        f"present in this photograph (Confidence == 1), and the claim in the "
                        f"state asserts exactly that.",
                        len(rows), derived_from="openimages human-verified label, Confidence==1"))

    n = 0
    for photo in oi:
        if n >= want_false:
            break
        if not photo["absent"]:
            continue
        thing = C._phrase(photo["absent"][0])
        r = random.Random(f"{SEED}\0claim-false\0{photo['key']}")
        made = T.make_claim_supported(
            {"id": photo["key"], "subject": thing, "present": [], "absent": [thing],
             "attrs": {}, "hidden": {}, "peers": []}, r, intent="false", kind="object")
        rows.append(row(photo, "claim_supported", made, False,
                        f"A human verified the Open Images label {photo['absent'][0]!r} as NOT "
                        f"present in this photograph (Confidence == 0); the state claims it is "
                        f"there.", len(rows),
                        derived_from="openimages human-verified label, Confidence==0"))
        n += 1

    n = 0
    for photo in abo:
        if n >= want_unknown:
            break
        hidden = photo["inputs"].get("hidden") or {}
        usable = {k: v for k, v in hidden.items()
                  if k in ("model_number", "item_weight", "item_dimensions", "model_year")}
        if not usable:
            continue
        r = random.Random(f"{SEED}\0claim-unknown\0{photo['key']}")
        made = T.make_claim_supported(
            {**photo["inputs"], "hidden": usable}, r, intent="unverifiable")
        field = made["basis"].get("field", "an off-photo attribute")
        rows.append(row(photo, "claim_supported", made, "unknown",
                        f"The claim is about the listing's {field}, which the ABO README lists "
                        f"among the fields not visible in a catalogue photograph; no photograph "
                        f"can confirm or deny it, so the honest answer is unknown.", len(rows),
                        derived_from="ABO structured field the README calls invisible in a photo"))
        n += 1
    return rows


NEAR_MATERIALS = [{"leather", "faux leather", "suede"}, {"fabric", "canvas", "polyester", "cotton"},
                  {"plastic", "silicone", "rubber", "polypropylene"},
                  {"metal", "steel", "stainless steel", "aluminium", "iron"},
                  {"stone", "marble", "concrete", "ceramic"}, {"wood", "engineered wood", "bamboo"}]
COLOUR_ALIAS = {"ivory": "cream", "off-white": "white", "tan": "brown", "navy blue": "navy",
                "light blue": "blue", "bronze": "brown", "silver": "grey", "gold": "yellow"}


# Items rejected after looking at the photograph (23 Sept 2026): the ABO metadata was right about
# the listing but the photo cannot settle it, or the metadata disagrees with what is visible.
EYEBALL_REJECT = {
    "abo:B07TG4THF1": "printed phone case: leather vs canvas is not visible in the print",
    "abo:B07VLHKN8B": "round table top reads as stone/ceramic; listing says glass",
    "abo:B07B4D8982": "sofa: listed wood frame is not visible, legs look like metal",
    "abo:B07NC2Z7CR": "boots are visibly black; listing colour says blue",
    "abo:B07MCBKLXJ": "headboard photo is a fabric close-up; listing material says metal",
    "abo:B07MF1TZFT": "pillow reads as grey-green; listing colour says blue",
    "abo:B07C4HBW3Z": "towel holder is visibly chrome/brass metal; listing says black plastic",
    "abo:B07B78RCTK": "stool is wood and black metal; listing material says stone",
}
CASES = {"phone case", "tablet case", "laptop case"}


def clearly_different(basis) -> bool:
    """A probe swap must be unmistakable in a photograph: no black -> navy, no leather -> suede."""
    from v2.state_grounded.edits import NAMED_RGB
    field, old, new = basis["swapped"], str(basis["swapped_from"]), str(basis["swapped_to"])
    if field == "colour":
        a = NAMED_RGB.get(COLOUR_ALIAS.get(old, old))
        b = NAMED_RGB.get(COLOUR_ALIAS.get(new, new))
        if not a or not b:
            return False
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 >= 150
    if field == "material":
        return not any(old in grp and new in grp for grp in NEAR_MATERIALS)
    return True


def build_which_field_conflicts(abo, rng, quota):
    """An ABO listing state built from the real curated fields, with exactly one swapped (gold =
    that field's path) or none swapped (gold = `none of these`)."""
    rows = []
    want_conflict = round(quota * 0.65)
    rich = [p for p in abo if len(p["inputs"]["attrs"]) >= 2]
    for photo in rich:
        if len(rows) >= quota:
            break
        want = "one_conflict" if len(rows) < want_conflict else "none"
        r = random.Random(f"{SEED}\0conflict\0{want}\0{photo['key']}")
        made = T.make_which_field_conflicts(photo["inputs"], r, intent=want)
        if made is None or made["intent"] != want:
            continue
        if f"abo:{photo['key']}" in EYEBALL_REJECT:
            continue
        if want == "one_conflict" and not clearly_different(made["basis"]):
            continue
        if made["basis"]["swapped"] == "material" and \
                photo["inputs"]["attrs"].get("product_type") in CASES:
            continue            # a printed case shows its print, not what it is made of
        basis = made["basis"]
        if want == "one_conflict":
            gold = basis["swapped_path"]
            rationale = (f"The listing's curated ABO {basis['swapped']} is "
                         f"{basis['swapped_from']!r}; the state records {basis['swapped_to']!r} "
                         f"instead. Every other field in the state is the listing's real curated "
                         f"value, so the photograph can only contradict this one.")
        else:
            gold = T.NONE_OF_THESE
            rationale = ("Every field in the state is the listing's real curated ABO value, so "
                         "the photograph contradicts none of them.")
        rows.append(row(photo, "which_field_conflicts", made, gold, rationale, len(rows),
                        derived_from="ABO curated structured fields (product_type/color/material)"))
    return rows


def build_count_matches(oi, rng, quota):
    """Verified-absent thing with an expected count >= 1 -> fewer; verified-present thing with an
    expected count of 0 -> more.  Neither needs the exact number of anything."""
    rows = []
    half = quota // 2 + quota % 2
    for photo in oi:
        if len(rows) >= half:
            break
        pool = [C._phrase(x) for x in photo["absent"] if T.is_countable(C._phrase(x))]
        if not pool:
            continue
        thing = pool[0]
        r = random.Random(f"{SEED}\0count-fewer\0{photo['key']}")
        made = T.make_count_matches(
            {"id": photo["key"], "subject": thing, "present": [], "absent": [thing]},
            r, intent="fewer")
        if made is None:
            continue
        n = made["basis"]["expected"]
        is_bool = made["fields"][0]["type"] == "boolean"
        gold = False if is_bool else 0
        rows.append(row(photo, "count_matches", made, gold,
                        f"A human verified that this photograph does NOT contain "
                        f"{photo['absent'][0]!r} (Confidence == 0), so the photograph shows zero "
                        f"of them against the {n} the state expects: fewer, and certainly not a "
                        f"match.", len(rows),
                        derived_from="openimages human-verified label, Confidence==0"))
    n_more = 0
    for photo in oi:
        if n_more >= quota - half:
            break
        pool = [C._phrase(x) for x in photo["present"] if T.is_countable(C._phrase(x))]
        if not pool:
            continue
        thing = pool[0]
        r = random.Random(f"{SEED}\0count-more\0{photo['key']}")
        made = T.make_count_matches(
            {"id": photo["key"], "subject": thing, "present": [thing], "absent": []},
            r, intent="more")
        if made is None or made["basis"]["expected"] != 0:
            continue
        is_bool = made["fields"][0]["type"] == "boolean"
        gold = False if is_bool else 2
        rows.append(row(photo, "count_matches", made, gold,
                        f"A human verified {photo['present'][0]!r} as present in this photograph "
                        f"(Confidence == 1), so there is at least one, against the 0 the state "
                        f"expects: more.", len(rows),
                        derived_from="openimages human-verified label, Confidence==1"))
        n_more += 1
    return rows


def build_label_text_matches(oi, rng, quota):
    """A photo where a human verified that signage / labels / numbers are NOT present cannot show
    the serial the state expects.  Stop signs carry one word, and it is always the same word."""
    rows = []
    want_yes = quota // 4
    want_no = quota - want_yes
    for photo in oi:
        if len(rows) >= want_no:
            break
        absent = [x for x in TEXT_LABELS_ABSENT if x in photo["verified_absent"]]
        if not absent:
            continue
        r = random.Random(f"{SEED}\0label-no\0{photo['key']}")
        made = T.make_label_text_matches({"id": photo["key"]}, r, intent="code")
        rows.append(row(photo, "label_text_matches", made, False,
                        f"A human verified that {', '.join(repr(x) for x in absent[:3])} are NOT "
                        f"in this photograph (Confidence == 0), so there is no label, sign or "
                        f"number in it that could carry the serial the state expects.", len(rows),
                        derived_from="openimages human-verified label, Confidence==0"))
    n = 0
    for photo in oi:
        if n >= want_yes:
            break
        if STOP_SIGN not in photo["verified_present"]:
            continue
        r = random.Random(f"{SEED}\0label-yes\0{photo['key']}")
        made = T.make_label_text_matches({"id": photo["key"]}, r, intent="word")
        made = json.loads(json.dumps(made))
        made["state"], made["state_path"] = T._label_state(r, "STOP")
        i, template = T._pick(r, T.LABEL_QUESTIONS)
        made["fields"] = [T._field("boolean", T._fmt(template, made["state_path"]))]
        made["template_id"] = f"label_text_matches/{i:02d}/word"
        made["basis"] = {"kind": "text", "stated": "STOP", "text_kind": "word"}
        rows.append(row(photo, "label_text_matches", made, True,
                        "A human verified the Open Images label 'Stop sign' as present in this "
                        "photograph (Confidence == 1); a stop sign carries the word STOP, which "
                        "is the text the state expects. Checked by eye as well.", len(rows),
                        derived_from="openimages human-verified label, Confidence==1 + eyeball"))
        n += 1
    return rows


def build_condition_report(oi, abo, rng, quota):
    """ABO photographs are catalogue shots of new goods: a damage report about them is not borne
    out.  Open Images verifies `Rust` on a small number of photos, which gives the other two
    answers.  The rust photos are taken first because there are so few of them; ABO tops the
    family up to quota.
    """
    rows = []
    n = 0
    for photo in oi:
        if n >= min(10, quota // 3):
            break
        if RUST not in photo["verified_present"]:
            continue
        item = next((C._phrase(x) for x in photo["present"]), None)
        if item is None:
            continue
        r = random.Random(f"{SEED}\0cond-rust\0{photo['key']}")
        reported = (n % 2 == 0)
        damage = "rust or corrosion" if reported else "a shattered screen or glass"
        made = T.make_condition_report({"attrs": {}, "present": [item], "subject": item}, r)
        if made is None:
            continue
        made["state"], made["state_path"] = T._condition_state(r, damage, item=T._a(item))
        i, template = T._pick(r, T.CONDITION_QUESTIONS)
        pairs = list(T.CONDITION_OPTIONS)
        made["fields"] = [T._field("choice", T._fmt(template, made["state_path"]),
                                   options=T._options([v for v, _ in pairs], r,
                                                      {v: d for v, d in pairs}))]
        made["template_id"] = f"condition_report/{i:02d}/3/reported"
        made["basis"] = {"kind": "condition", "stated": damage}
        if reported:
            gold = "the reported condition is visible"
            why = ("A human verified the Open Images label 'Rust' as present in this photograph "
                   "(Confidence == 1), and the report describes rust or corrosion.")
        else:
            gold = "a different condition is visible"
            why = ("A human verified 'Rust' as present in this photograph (Confidence == 1), so "
                   "the item is visibly in poor condition -- but the report describes a "
                   "shattered screen, which is not what is there.")
        rows.append(row(photo, "condition_report", made, gold, why, len(rows),
                        derived_from="openimages human-verified label, Confidence==1"))
        n += 1

    for photo in abo:
        if len(rows) >= quota:
            break
        r = random.Random(f"{SEED}\0cond-clean\0{photo['key']}")
        made = T.make_condition_report(photo["inputs"], r)
        if made is None:
            continue
        rows.append(row(photo, "condition_report", made, "no visible damage",
                        f"This is an ABO retail catalogue photograph of a new "
                        f"{photo['inputs']['attrs'].get('product_type')} offered for sale; the "
                        f"damage the report describes ({made['basis']['stated']!r}) is not in the "
                        f"picture. The assumption that ABO catalogue shots show undamaged new "
                        f"goods was spot-checked by eye on a sample of these images.", len(rows),
                        derived_from="ABO catalogue photograph of new goods"))
    return rows


def build_instruction_pairs(oi, rng, quota):
    """The `catalogue_switch` construction: list A holds something a human verified is in the
    photograph, list B holds only things a human verified are not."""
    rows = []
    pairs = quota // 2
    made_pairs = 0
    for photo in oi:
        if made_pairs >= pairs:
            break
        present = [C._phrase(x) for x in photo["present"]]
        absent = [C._phrase(x) for x in photo["absent"]]
        if not present or len(absent) < 3:
            continue
        r = random.Random(f"{SEED}\0pair\0{photo['key']}")
        pair = T.make_instruction_pair(
            {"id": photo["key"], "subject": present[0], "present": present[:1],
             "absent": absent, "attrs": {}}, r, construction="catalogue_switch")
        if not pair or pair[0]["basis"]["kind"] != "catalogue_switch":
            continue
        pair_id = f"state-probe:pair:{photo['key']}"
        hit = T._a(pair[0]["basis"]["hit"])
        for made in pair:
            role = made["basis"]["role"]
            if role == "a":
                gold, why = hit, (
                    f"The checklist in the state includes {hit!r}, which a human verified is in "
                    f"this photograph (Confidence == 1); every other entry on the list was "
                    f"verified absent (Confidence == 0).")
            else:
                gold, why = T.CATALOGUE_NONE, (
                    "Every entry on this checklist was human-verified as NOT in the photograph "
                    "(Confidence == 0), so the honest answer is 'none of them'. The photograph "
                    "and the question are identical to the other half of the pair; only the "
                    "state changed.")
            rows.append(row(photo, "instruction_changes_answer", made, gold, why,
                            f"{made_pairs}{role}", pair_id=pair_id, pair_role=role,
                            derived_from="openimages human-verified labels, both polarities"))
        made_pairs += 1
    return rows


def main(seed=SEED):
    rng = random.Random(seed)
    oi = load_openimages()
    abo = load_abo()
    random.Random(f"{seed}\0oi").shuffle(oi)
    random.Random(f"{seed}\0abo").shuffle(abo)
    # held-out photographs first: an Open Images photo whose `decision-v2-photo` partition is
    # `test` is never a training row of any v2 source.  Only the scarce 'Stop sign' and 'Rust'
    # photos reach further, and those are flagged and kept out of the state-grounded rows.
    oi.sort(key=lambda p: p["partition"] != "test")
    # For the probe a swapped product type comes from a DIFFERENT catalogue group (a phone case
    # listed as a sandal, not as a tablet case): the gold must be beyond argument.  Training rows
    # keep the harder same-group swaps.
    peers = defaultdict(set)
    meta = C.load_abo_meta()
    for p in abo:
        peers[meta[p["key"]]["cgroup"]].add(p["inputs"]["attrs"]["product_type"])
    for p in abo:
        grp = meta[p["key"]]["cgroup"]
        others = sorted({t for g, ts in peers.items() if g != grp for t in ts})
        random.Random(f"{seed}\0peers\0{p['key']}").shuffle(others)
        p["inputs"]["peers"] = others[:30]
    # at most three photos per product type, so phone cases do not fill the ABO half
    per_type, capped = Counter(), []
    for p in abo:
        t = p["inputs"]["attrs"]["product_type"]
        if per_type[t] < 3:
            per_type[t] += 1
            capped.append(p)
    abo = capped

    rows = []
    rows += build_claim_supported(oi, abo, rng, QUOTA["claim_supported"])
    rows += build_which_field_conflicts(abo, rng, QUOTA["which_field_conflicts"])
    rows += build_count_matches(oi, rng, QUOTA["count_matches"])
    rows += build_label_text_matches(oi, rng, QUOTA["label_text_matches"])
    rows += build_condition_report(oi, abo, rng, QUOTA["condition_report"])
    rows += build_instruction_pairs(oi, rng, QUOTA["instruction_changes_answer"])

    seen = set()
    unique = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        unique.append(r)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n")

    summary = {
        "path": str(OUT_PATH.relative_to(ROOT)),
        "items": len(unique),
        "families": dict(Counter(r["probe_family"] for r in unique)),
        "photo_sources": dict(Counter(r["photo_source"] for r in unique)),
        "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in unique)),
        "gold": dict(Counter(str(r["gold"]) for r in unique)),
        "pairs": len({r["pair_id"] for r in unique if r.get("pair_id")}),
        "distinct_photos": len({r["source_group"] for r in unique}),
        "photos_outside_test_partition": sorted({r["source_group"] for r in unique
                                                 if r["photo_partition"] != "test"}),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    main(ap.parse_args().seed)
