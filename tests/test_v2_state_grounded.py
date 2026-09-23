"""Tests for the state-grounded families, the reference/target pairs, and the human-derived probes.

Run with:  PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v2_state_grounded.py
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decision_data import expand_fields, render                  # noqa: E402
from v2.common import verified_license                           # noqa: E402
from v2.state_grounded import build_pairs as BP                  # noqa: E402
from v2.state_grounded import edits as E                         # noqa: E402
from v2.state_grounded import templates as T                     # noqa: E402
from vision_decision.contracts import UNKNOWN, Request           # noqa: E402

RECORDS = ROOT / "data" / "decision-v2" / "state_grounded" / "records.jsonl"
PAIRS = ROOT / "data" / "decision-v2" / "pairs_grounded" / "records.jsonl"
NATURAL = ROOT / "data" / "decision-v2" / "pairs_natural" / "records.jsonl"
PROBE = ROOT / "data" / "manifests" / "decision-v2-state-probe.jsonl"
PAIRS_PROBE = ROOT / "data" / "manifests" / "decision-v2-pairs-probe.jsonl"

BANKS = {
    "CLAIM_QUESTIONS": T.CLAIM_QUESTIONS, "CONFLICT_QUESTIONS": T.CONFLICT_QUESTIONS,
    "COUNT_BOOL_QUESTIONS": T.COUNT_BOOL_QUESTIONS, "COUNT_ORDINAL_QUESTIONS": T.COUNT_ORDINAL_QUESTIONS,
    "COUNT_BAND_QUESTIONS": T.COUNT_BAND_QUESTIONS, "LABEL_QUESTIONS": T.LABEL_QUESTIONS,
    "CONDITION_QUESTIONS": T.CONDITION_QUESTIONS, "PAIR_SUBJECT_QUESTIONS": T.PAIR_SUBJECT_QUESTIONS,
    "PAIR_CATALOGUE_QUESTIONS": T.PAIR_CATALOGUE_QUESTIONS,
    "OBJECT_CLAIM_FORMS": T.OBJECT_CLAIM_FORMS, "ATTRIBUTE_CLAIM_FORMS": T.ATTRIBUTE_CLAIM_FORMS,
    "COUNT_CLAIM_FORMS": T.COUNT_CLAIM_FORMS, "TEXT_CLAIM_FORMS": T.TEXT_CLAIM_FORMS,
    "CONDITION_CLAIM_FORMS": T.CONDITION_CLAIM_FORMS, "CATEGORY_CLAIM_FORMS": T.CATEGORY_CLAIM_FORMS,
    "UNVERIFIABLE_CLAIM_FORMS": T.UNVERIFIABLE_CLAIM_FORMS,
    "WHAT_CHANGED_Q": BP.WHAT_CHANGED_Q, "STILL_MATCHES_Q": BP.STILL_MATCHES_Q,
    "FIELD_WRONG_Q": BP.FIELD_WRONG_Q, "SAME_ITEM_Q": BP.SAME_ITEM_Q, "ATTR_DIFF_Q": BP.ATTR_DIFF_Q,
}

ABO_INPUTS = {"id": "B000TEST01", "source": "abo", "subject": "chair", "present": ["chair"],
              "absent": [], "attrs": {"product_type": "chair", "colour": "black", "material": "wood"},
              "hidden": {"model_number": "TX-1", "item_weight": "4 pounds"},
              "peers": ["stool", "sofa", "bench"], "title": "", "condition_is_new": True}
OI_INPUTS = {"id": "oi1", "source": "openimages_v2", "subject": "dog",
             "present": ["dog", "car", "bicycle", "tree"], "absent": ["cat", "horse", "bus", "boat"],
             "attrs": {}, "hidden": {}, "peers": [], "title": ""}


def _rows(path, cap=None):
    if not path.is_file():
        pytest.skip(f"{path.relative_to(ROOT)} has not been built")
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if cap and len(out) >= cap:
                    break
    return out


# --------------------------------------------------------------------------- templates
@pytest.mark.parametrize("name", sorted(BANKS))
def test_every_bank_has_at_least_eight_paraphrases(name):
    bank = BANKS[name]
    assert len(bank) >= 8, f"{name} has {len(bank)}"
    assert len(set(bank)) == len(bank), f"{name} repeats a paraphrase"


@pytest.mark.parametrize("family", sorted(T.MAKERS))
@pytest.mark.parametrize("inputs", [ABO_INPUTS, OI_INPUTS])
def test_templates_are_pure_and_contract_valid(family, inputs):
    for seed in range(40):
        a = T.make(family, json.loads(json.dumps(inputs)), random.Random(seed))
        b = T.make(family, json.loads(json.dumps(inputs)), random.Random(seed))
        assert a == b
        if a is None:
            continue
        Request.model_validate({"request_id": "t", "state": a["state"], "fields": a["fields"]})


def test_instruction_pairs_share_photo_wording_and_differ_only_in_state():
    for seed in range(60):
        pair = T.make_instruction_pair(OI_INPUTS, random.Random(seed))
        assert pair and len(pair) == 2
        a, b = pair
        assert a["state"] != b["state"]
        assert {a["pair_role"], b["pair_role"]} == {"a", "b"}
        # same question template (the quoted path may differ with the state shape)
        assert a["template_id"].rsplit("/", 1)[0] == b["template_id"].rsplit("/", 1)[0]


def test_conflict_record_swaps_exactly_one_field_or_none():
    for seed in range(80):
        for intent in ("one_conflict", "none"):
            made = T.make_which_field_conflicts(ABO_INPUTS, random.Random(seed), intent=intent)
            options = [o["value"] for o in made["fields"][0]["options"]]
            assert T.NONE_OF_THESE in options
            assert made["basis"]["swapped_path"] in options
            if made["intent"] == "none":
                assert made["basis"]["swapped"] is None
            else:
                assert made["basis"]["swapped_from"] != made["basis"]["swapped_to"]


def test_mass_nouns_never_get_an_article_or_a_count():
    assert T._a("clothing") == "clothing"
    assert T._plural("clothing") == "clothing"
    assert not T.is_countable("food")
    assert T._a("chair") == "a chair" and T._plural("a chair") == "chairs"


# --------------------------------------------------------------------------- converted records
def test_record_count_and_family_mix():
    rows = _rows(RECORDS)
    assert 55_000 <= len(rows) <= 65_000
    fams = Counter(r["family"] for r in rows)
    assert set(fams) == set(T.FAMILIES)
    paired = fams["instruction_changes_answer"] / len(rows)
    assert 0.08 <= paired <= 0.12


def test_claim_supported_truth_false_unknown_shares():
    rows = [r for r in _rows(RECORDS) if r["family"] == "claim_supported"]
    shares = {k: v / len(rows) for k, v in Counter(r["claim_intent"] for r in rows).items()}
    assert abs(shares["true"] - 0.35) <= 0.05
    assert abs(shares["false"] - 0.35) <= 0.05
    assert abs(shares["unverifiable"] - 0.30) <= 0.05


def test_criteria_on_about_thirty_percent():
    rows = _rows(RECORDS)
    with_criteria = sum(any("description" in o for o in r["request"]["fields"][0].get("options", []))
                        or "yes_description" in r["request"]["fields"][0] for r in rows)
    ordinal = sum(r["request"]["fields"][0]["type"] == "ordinal" for r in rows)
    share = (with_criteria) / (len(rows) - ordinal)
    assert 0.20 <= share <= 0.40, share


def test_pairs_are_intact():
    rows = [r for r in _rows(RECORDS) if r.get("pair_id")]
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    assert by_pair
    for pid, members in by_pair.items():
        assert sorted(m["pair_role"] for m in members) == ["a", "b"], pid
        a, b = members
        assert a["images"] == b["images"]
        assert a["source_group"] == b["source_group"] and a["partition"] == b["partition"]
        assert a["request"]["state"] != b["request"]["state"]


def test_pending_shape_and_partitions_by_photo():
    parts = defaultdict(set)
    for r in _rows(RECORDS):
        assert r["target"] is None and r["abstention_cause"] is None
        assert r["pseudo_label"] == "pending"
        assert (r["partition"] == "test") == bool(r.get("pseudo_label_test"))
        parts[r["source_group"]].add(r["partition"])
    assert all(len(p) == 1 for p in parts.values())


def test_no_caption_or_title_leaks_into_the_state():
    captions = {}
    sel = ROOT / "data" / "decision-v2-raw" / "pd12m" / "selection.jsonl"
    if sel.is_file():
        for line in sel.open():
            row = json.loads(line)
            captions[row["pd12m_id"]] = row["caption"]
    meta_path = ROOT / ".cache" / "datasets" / "v1" / "state_aware" / "abo_listing_meta.json"
    titles = {k: v["title"] for k, v in json.loads(meta_path.read_text()).items()} \
        if meta_path.is_file() else {}
    checked = 0
    for r in _rows(RECORDS):
        text = T.state_text(r["request"]["state"])
        key = r["source_group"].split(":", 1)[1]
        if r["photo_source"] == "pd12m" and key in captions:
            assert not T.shares_long_ngram(text, captions[key]), r["id"]
            checked += 1
        if r["photo_source"] == "abo" and key in titles and len(titles[key]) > 12:
            assert titles[key].lower() not in text.lower(), r["id"]
            assert not T.shares_long_ngram(text, titles[key]), r["id"]
            checked += 1
    assert checked > 1000


def test_licence_objects_verify():
    seen = {}
    for r in _rows(RECORDS) + _rows(PROBE):
        lic = r["license"]
        key = (lic["evidence"], lic["spdx"])
        if key not in seen:
            seen[key] = verified_license(Path(lic["evidence"]), lic["spdx"])
        assert seen[key] == lic
        assert not Path(lic["evidence"]).is_absolute()
    assert len(seen) >= 4


def test_images_are_existing_files_by_repo_path():
    rows = _rows(RECORDS)
    for r in rows[::500]:
        for im in r["images"]:
            assert not Path(im["image"]).is_absolute()
            assert im["image"].startswith(("data/decision-v2/", "data/decision-v1/abo/"))
            assert (ROOT / im["image"]).is_file()


def test_records_render_through_the_training_path():
    rng = random.Random(5)
    for r in _rows(RECORDS)[::60]:
        for item in expand_fields(json.loads(json.dumps(r))):
            header, choices, texts, index = render(item, rng)
            assert choices[index][0] == UNKNOWN
            assert len(texts) == len(choices)


# --------------------------------------------------------------------------- probe
def test_probe_is_human_derived_and_heldout():
    rows = _rows(PROBE)
    assert len(rows) == 200
    fams = Counter(r["probe_family"] for r in rows)
    assert set(fams) == set(T.FAMILIES) and min(fams.values()) >= 30
    for r in rows:
        assert r["partition"] == "test" and r["heldout_family"] is True
        assert r["rationale"] and len(r["rationale"]) > 40
        assert r["pseudo_label"].startswith("human-derived")
        f = r["request"]["fields"][0]
        Request.model_validate(r["request"])
        if r["gold"] == "unknown":
            continue
        if f["type"] == "boolean":
            assert isinstance(r["gold"], bool)
        elif f["type"] == "choice":
            assert r["gold"] in [o["value"] for o in f["options"]]
        else:
            assert r["gold"] in [lv["value"] for lv in f["levels"]]


def test_probe_pairs_change_gold_when_state_changes():
    pairs = defaultdict(list)
    for r in _rows(PROBE):
        if r.get("pair_id"):
            pairs[r["pair_id"]].append(r)
    assert len(pairs) >= 15
    for members in pairs.values():
        a, b = sorted(members, key=lambda r: r["pair_role"])
        assert a["images"] == b["images"]
        assert a["gold"] != b["gold"]
        assert a["request"]["state"] != b["request"]["state"]


def test_probe_photos_never_appear_in_the_state_grounded_rows():
    probe = _rows(PROBE)
    probe_groups = {r["source_group"] for r in probe}
    used = {r["source_group"] for r in _rows(RECORDS)}
    assert not (probe_groups & used)
    # ABO probe photos come from the v1 TEST partition, never seen by a v1/v1.1 run
    abo = [r for r in probe if r["photo_source"] == "abo"]
    assert abo and all(r["photo_partition"] == "test" for r in abo)
    outside = [r for r in probe if r["photo_partition"] != "test"]
    assert len(outside) <= 12, "only the rare Stop sign / Rust photos may come from outside test"


# --------------------------------------------------------------------------- composited pairs
def test_edit_kinds_and_falsified_fields_line_up():
    assert set(E.FALSIFIES) == set(E.RELEVANT_KINDS)
    assert set(E.FALSIFIES.values()) == set(BP.FIELD_TEXT)


def test_composited_pairs_labels_follow_the_edit():
    rows = _rows(PAIRS)
    assert len(rows) >= 15_000
    irrelevant = sum(not r["edit_relevant"] for r in rows) / len(rows)
    assert 0.22 <= irrelevant <= 0.36, irrelevant
    for r in rows:
        kind, fam, gold = r["edit_kind"], r["family"], r["target"]
        assert len(r["images"]) == 2 and r["pseudo_label"] == "construction"
        field = r["request"]["fields"][0]
        if fam == "what_changed":
            assert kind != "corner_object"
            want = BP.CHANGE_OPTIONS[kind if kind in E.RELEVANT_KINDS else "irrelevant"][0]
            assert gold == want
            assert gold in [o["value"] for o in field["options"]]
        elif fam == "target_still_matches_reference":
            assert gold is (kind not in E.RELEVANT_KINDS)
        else:
            options = [o["value"] for o in field["options"]]
            assert gold in options
            if kind in E.RELEVANT_KINDS:
                assert gold.endswith("." + E.FALSIFIES[kind])
            else:
                assert gold == T.NONE_OF_THESE


def test_composited_pairs_images_and_attribution():
    rows = _rows(PAIRS)
    attribution = _rows(ROOT / "data" / "decision-v2" / "pairs_grounded" / "attribution.jsonl")
    edited = {a["edited_sha256"] for a in attribution}
    for r in rows[::200]:
        ref, tgt = r["images"]
        assert (ROOT / ref["image"]).is_file() and (ROOT / tgt["image"]).is_file()
        new = ref if r["edit_meta"].get("swap") else tgt
        assert new["image"].startswith("data/decision-v2/pairs_grounded/images_edited/")
        assert new["sha256"] in edited
        assert r["license"]["spdx"] == "CC0-1.0"


def test_natural_pairs_are_pending_and_partition_safe():
    rows = _rows(NATURAL)
    assert len(rows) >= 8_000
    for r in rows:
        assert r["target"] is None and r["pseudo_label"] == "pending" and len(r["images"]) == 2


def test_pairs_probe_is_eyeballed_and_balanced():
    rows = _rows(PAIRS_PROBE)
    assert len(rows) == 60
    assert all(r["partition"] == "test" and r["heldout_family"] for r in rows)
    assert all(r.get("eyeballed") is True for r in rows)
    assert len({r["probe_family"] for r in rows}) == 3
