"""decision-v2 photo templates and photo converter.

Nothing here touches the network or a model.  The converter tests build a synthetic manifest in a
temporary directory and point the converter at it, so they check the record shape, the partitioning
and the mix (two-image share, unknown-construction share, state kinds) without needing the real
45,000 photos to be on disk.

    PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v2_photo_templates.py -q
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from v1_text.common import stable_partition  # noqa: E402
from v2 import convert_photos as C  # noqa: E402
from v2.templates import photo as T  # noqa: E402
from vision_decision.contracts import UNKNOWN, Request  # noqa: E402

INPUTS = {
    "id": "photo-1",
    "category": "kitchen",
    "present": ["Coffee table", "Chair", "Window"],
    "absent": ["Dog", "Piano", "Bicycle"],
    "title": "A kitchen in Bristol",
    "second": {"category": "kitchen", "present": ["Sink"], "absent": ["Horse"], "title": "Another kitchen"},
    "same_category": True,
}

PARAPHRASE_BANKS = [
    T.SCENE_QUESTIONS, T.MAIN_OBJECT_QUESTIONS, T.OBJECT_PRESENT_QUESTIONS, T.COUNT_QUESTIONS,
    T.QUALITY_QUESTIONS, T.COLOUR_QUESTIONS, T.TEXT_VISIBLE_QUESTIONS, T.MATERIAL_QUESTIONS,
    T.MATERIAL_OF_X_QUESTIONS, T.INDOOR_QUESTIONS, T.SAME_SUBJECT_QUESTIONS, T.WHICH_IMAGE_QUESTIONS,
]


# --------------------------------------------------------------------------- paraphrases
def test_every_paraphrase_bank_has_at_least_eight_distinct_wordings():
    for bank in PARAPHRASE_BANKS:
        assert len(bank) >= 8, bank[:1]
        assert len(set(bank)) == len(bank), "paraphrases must be distinct"
        assert all(q.strip() and len(q) <= 2000 for q in bank)


def test_each_family_actually_uses_at_least_eight_wordings():
    rng = random.Random(0)
    for family in T.FAMILIES:
        questions = set()
        for _ in range(400):
            fields, _ = T.make(family, INPUTS, rng)
            questions.add(fields[0]["question"])
        assert len(questions) >= 8, (family, len(questions))


def test_template_id_names_the_family_and_the_wording():
    rng = random.Random(1)
    for family in T.FAMILIES:
        _, template_id = T.make(family, INPUTS, rng)
        head, index = template_id.split("/")[0], template_id.split("/")[1]
        assert head == family
        assert index.isdigit()


# --------------------------------------------------------------------------- option hygiene
def _all_fields(n=600, seed=2):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for family in T.FAMILIES:
            for unknown in (False, True):
                fields, template_id = T.make(family, INPUTS, rng, unknown=unknown)
                out.append((family, template_id, fields[0]))
    return out


def test_fields_validate_against_the_request_contract():
    rng = random.Random(3)
    for family, _, field in _all_fields(n=40):
        state, _ = T.make_state(INPUTS, rng)
        Request.model_validate({"schema_version": "1.0", "request_id": "r", "state": state, "fields": [field]})


def test_options_are_plausible_unique_and_never_the_reserved_unknown():
    for family, template_id, field in _all_fields(n=60):
        if field["type"] != "choice":
            continue
        values = [o["value"] for o in field["options"]]
        assert 2 <= len(values) <= 25, (template_id, len(values))
        assert len(set(values)) == len(values), template_id
        assert len({v.strip().lower() for v in values}) == len(values), template_id
        assert UNKNOWN not in values
        for value in values:
            assert value.strip() == value and 0 < len(value) <= 128
        # every option is drawn from the family's declared bank, so no distractor is nonsense
        bank = {"scene_type": set(T.SCENES), "colour_of_x": set(T.COLOURS),
                "material": set(T.MATERIALS),
                "indoor_outdoor": {v for v, _ in T.INDOOR_OPTIONS},
                "two_image_which": {v for v, _ in T.WHICH_OPTIONS}}.get(family)
        if bank is not None:
            assert set(values) <= bank, (template_id, set(values) - bank)
        if family == "main_object":
            pool = set(T.COMMON_OBJECTS) | {T._thing_phrase(x) for x in INPUTS["present"] + INPUTS["absent"]}
            assert set(values) <= pool, (template_id, set(values) - pool)


def test_ordinal_levels_are_ascending_and_described():
    for family, template_id, field in _all_fields(n=40):
        if field["type"] != "ordinal":
            continue
        values = [l["value"] for l in field["levels"]]
        assert values == sorted(set(values)) and 2 <= len(values) <= 10, template_id
        assert all(l["description"].strip() for l in field["levels"])


def test_about_a_third_of_questions_carry_option_criteria():
    fields = [f for _, _, f in _all_fields(n=120) if f["type"] in ("choice", "boolean")]
    described = sum(
        any("description" in o for o in f.get("options", [])) or "yes_description" in f
        for f in fields)
    assert 0.15 <= described / len(fields) <= 0.55, described / len(fields)


# --------------------------------------------------------------------------- state
def test_state_kinds_cover_empty_irrelevant_and_a_contradicting_scene():
    rng = random.Random(4)
    kinds = Counter()
    strings = 0
    for _ in range(4000):
        state, kind = T.make_state(INPUTS, rng)
        kinds[kind] += 1
        strings += isinstance(state, str)
    assert set(kinds) == set(T.STATE_KINDS)
    assert all(v > 200 for v in kinds.values())
    assert 0.33 <= strings / 4000 <= 0.47, strings / 4000  # the spec asks for about 40% string states
    # the contradicting state must really describe a different scene, not the photo
    state, _ = T.make_state(INPUTS, rng, kind="contradictory", as_string=False)
    assert state["record"]["claimed_scene"] in {s for s, _, _ in T.OTHER_SCENES}
    assert "kitchen" not in json.dumps(state).lower()


def test_empty_state_is_an_empty_object():
    rng = random.Random(5)
    state, kind = T.make_state(INPUTS, rng, kind="empty")
    assert kind == "empty" and state == {}


# --------------------------------------------------------------------------- the converter
def _fixture(tmp_path, monkeypatch, source, n=400):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    (raw / source).mkdir(parents=True)
    out.mkdir()
    categories = sorted(T.CATEGORY_OBJECT)[:12]
    manifest, selection = [], []
    for i in range(n):
        sha = f"{i:064x}"
        manifest.append({
            "pageid": i, "image_id": f"img{i:012d}", "title": f"File:Thing {i}.jpg",
            "category_label": categories[i % len(categories)],
            "sha256": sha, "image": f"data/decision-v2/{source}/images/{sha}.jpg",
            "width": 800, "height": 600, "bytes": 1234,
            "spdx": "CC-BY-4.0" if source == "commons_photos" else "CC-BY-2.0",
            "url": "https://example.invalid/x.jpg",
        })
        selection.append({
            "image_id": f"img{i:012d}", "bucket_label": categories[i % len(categories)],
            "labels_present": ["Chair", "Table"], "labels_absent": ["Dog", "Piano"],
            "labels_present_concrete": ["Chair", "Table"], "labels_absent_concrete": ["Dog", "Piano"],
            "title": f"Thing {i}",
        })
    (raw / source / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (raw / source / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selection))
    monkeypatch.setattr(C, "RAW", raw)
    monkeypatch.setattr(C, "OUT", out)
    return out


@pytest.mark.parametrize("source", ["commons_photos", "openimages_v2"])
def test_converter_shape_and_mix(tmp_path, monkeypatch, source):
    out = _fixture(tmp_path, monkeypatch, source)
    summary = C.build(source, seed=7)
    rows = [json.loads(line) for line in (out / source / "records.jsonl").read_text().splitlines()]

    assert summary["records"] == summary["photos"] * C.DECISIONS_PER_PHOTO
    for r in rows:
        assert r["target"] is None and r["abstention_cause"] is None
        assert r["pseudo_label"] == "pending"
        assert r["template_id"].split("/")[0] == r["family"]
        assert r["license"]["spdx"] in C.LICENSES[source]
        assert r["partition"] == stable_partition(r["source_group"].split("+")[0], seed=C.PARTITION_SEED)
        assert (r["partition"] == "test") == bool(r.get("pseudo_label_test"))
        Request.model_validate(r["request"])

    assert 0.15 <= summary["unknown_construction_share"] <= 0.25
    assert 0.06 <= summary["two_image_share"] <= 0.14
    assert 0.33 <= summary["state_string_share"] <= 0.47
    assert set(summary["families"]) >= set(C.SINGLE_FAMILY_WEIGHTS[source])
    assert summary["distinct_templates"] >= 80


@pytest.mark.parametrize("source", ["commons_photos", "openimages_v2"])
def test_two_image_records_pair_inside_one_partition(tmp_path, monkeypatch, source):
    out = _fixture(tmp_path, monkeypatch, source)
    C.build(source, seed=11)
    rows = [json.loads(line) for line in (out / source / "records.jsonl").read_text().splitlines()]
    pairs = [r for r in rows if len(r["images"]) == 2]
    assert pairs, "the converter must emit two-image records"
    for r in pairs:
        a, b = r["source_group"].split("+")
        assert a != b, "a photo may not be paired with itself"
        assert stable_partition(a, seed=C.PARTITION_SEED) == stable_partition(b, seed=C.PARTITION_SEED) == r["partition"]
        assert r["images"][0]["sha256"] != r["images"][1]["sha256"]
        assert r["family"] in T.TWO_IMAGE_FAMILIES
        assert isinstance(r["pair_same_category"], bool)
    same_category = sum(r["pair_same_category"] for r in pairs)
    # mostly same-category pairs, but not all of them: an all-same-subject pool would make the
    # "do these show the same kind of thing?" boolean answer yes every time
    assert 0.40 <= same_category / len(pairs) <= 0.80, same_category / len(pairs)

    # a two-image record must say which image is which
    assert all("image 1" in json.dumps(r["request"]).lower() or "image 2" in json.dumps(r["request"]).lower()
               or "two" in r["request"]["fields"][0]["question"].lower()
               or "both" in json.dumps(r["request"]).lower()
               for r in pairs)


def test_converter_is_deterministic(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch, "openimages_v2", n=120)
    C.build("openimages_v2", seed=3)
    first = (out / "openimages_v2" / "records.jsonl").read_text()
    C.build("openimages_v2", seed=3)
    assert (out / "openimages_v2" / "records.jsonl").read_text() == first


def test_no_source_group_spans_test_and_fit(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch, "commons_photos")
    C.build("commons_photos", seed=13)
    rows = [json.loads(line) for line in (out / "commons_photos" / "records.jsonl").read_text().splitlines()]
    seen = {}
    for r in rows:
        side = "test" if r["partition"] == "test" else "fit"
        assert seen.setdefault(r["source_group"], side) == side
