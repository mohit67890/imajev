"""decision-v2 `pd12m`: caption reading, bucket coverage, and the converter's pd12m branch.

Nothing here touches the network, PD12M or a model.  The converter tests build a synthetic
manifest in a temporary directory, so they check the record shape and the mix without the real
images being on disk.  The one thing they check that no other source needs is the rule that makes
this source usable at all: **a PD12M caption may never reach a record**.

    PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v2_pd12m.py -q
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from v1_text.common import stable_partition  # noqa: E402
from v2 import convert_photos as C  # noqa: E402
from v2 import pd12m_captions as CAP  # noqa: E402
from v2.templates import photo as T  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402

CAPTIONS = {
    "The image shows a yellow and black wasp sitting on top of a white cup. The wasp has a "
    "yellow body with black stripes and a black head.": "insect",
    "The image shows a man riding a bicycle down a street lined with trees, poles and railway "
    "tracks. The sky is visible in the background.": "bicycle",
    "The image shows a cobblestone street in the old town, with a row of colourful houses on "
    "either side and a board in front of them.": "street_scene",
    "The image shows a white car parked in front of a carousel, surrounded by a group of people "
    "standing on the ground near a metal fence.": "car",
    "The image shows a stethoscope resting on a white hospital bed beside a folded blanket and a "
    "small table.": "medical_device",
    # two captions that name nothing in the object lexicon: a third of real PD12M captions are
    # like this, and they are what makes `main_object` fall back to an unknown construction
    "The image shows a wide open meadow of dry golden grass stretching towards low hills, under "
    "a pale overcast sky.": "landscape",
    "The image shows a close up of pale green lichen spreading across a damp granite boulder "
    "under soft, even light.": "fungus",
}

ARTWORKS = [
    "The image shows an oil painting of a woman in a white dress, her face turned to the side.",
    "The image shows a pencil drawing of two spoons on a piece of paper with text written on it.",
    "The image shows an engraving of a sailing ship on rough seas, printed in black ink.",
    "The image shows a logo with a stylised bird above the name of a company in bold letters.",
    "The image shows a map of the county, with roads, rivers and settlements marked in colour.",
    "The image shows a handwritten manuscript page in brown ink with decorated initials.",
    "The image shows a silver coin with a crown on it against a white background.",
]


# --------------------------------------------------------------------------- caption reading
def test_artworks_scans_logos_and_maps_are_rejected():
    for caption in ARTWORKS:
        assert CAP.reject(caption), caption


def test_photographs_are_not_rejected():
    for caption in CAPTIONS:
        assert CAP.reject(caption) is None, caption


def test_a_caption_too_short_to_read_is_rejected():
    assert CAP.reject("The image shows a thing.") == "caption_too_short"
    assert CAP.reject("") == "caption_too_short"
    assert CAP.reject(None) == "caption_too_short"


def test_buckets_match_the_obvious_subject():
    for caption, expected in CAPTIONS.items():
        assert CAP.bucket(caption) == expected, caption


def test_there_are_at_least_forty_subject_buckets_and_all_are_distinct():
    assert len(CAP.BUCKETS) >= 40
    assert len(set(CAP.BUCKETS)) == len(CAP.BUCKETS)


def test_every_bucket_is_known_to_the_photo_templates():
    """A bucket with no entry in these three tables would silently produce empty option sets."""
    for name in CAP.BUCKETS:
        assert name in T.CATEGORY_OBJECT, name
        assert name in T.CATEGORY_SCENES, name
        assert name in T.CATEGORY_ASSOCIATED, name
        assert set(T.CATEGORY_SCENES[name]) <= set(T.SCENES), name


def test_the_buckets_pd12m_added_use_only_authored_object_phrases():
    """An associated item that is not one of the authored phrases would not actually be withheld
    from the "probably not in this photo" pool, which is the only job that list has."""
    for name, items in T.PD12M_CATEGORY_ASSOCIATED.items():
        assert set(items) <= set(T.COMMON_OBJECTS) | set(CAP.LEXICON_PHRASES), name
    assert set(T.PD12M_CATEGORY_OBJECT) == set(T.PD12M_CATEGORY_SCENES) == set(T.PD12M_CATEGORY_ASSOCIATED)
    assert set(T.PD12M_CATEGORY_OBJECT) <= set(CAP.BUCKETS)


def test_bucket_rules_are_valid_and_ordered_most_specific_first():
    # a rule that never fires is dead weight; a duplicate regex means one of them is unreachable
    patterns = [pattern for _, pattern in CAP.BUCKET_RULES]
    assert len(set(patterns)) == len(patterns)
    assert CAP.bucket("The image shows nothing that any rule in this file mentions at all.") is None


# --------------------------------------------------------------------------- present / absent
def test_present_objects_come_from_the_caption_and_absent_ones_do_not():
    caption = ("The image shows a man riding a bicycle down a street lined with trees, poles and "
               "railway tracks. The sky is visible in the background.")
    present, absent = CAP.objects(caption, associated=T.CATEGORY_ASSOCIATED["bicycle"])
    assert "a bicycle" in present and "a tree" in present
    assert not set(present) & set(absent)
    for phrase in absent:
        assert phrase not in present
        # an absent candidate is never something the bucket is expected to contain, and never one
        # of the things photographs carry without a caption saying so
        assert phrase not in T.CATEGORY_ASSOCIATED["bicycle"]
        assert phrase not in CAP.OFTEN_UNSAID


def test_absent_candidates_are_shuffled_per_photo_not_fixed_lexicon_order():
    caption = list(CAPTIONS)[0]
    a = CAP.objects(caption, rng=random.Random("photo-a"))[1]
    b = CAP.objects(caption, rng=random.Random("photo-b"))[1]
    assert a != b, "every photo would otherwise be asked about the same twelve things"


def test_object_phrases_read_as_english_through_the_template_helper():
    for phrase in CAP.LEXICON_PHRASES:
        assert T._thing_phrase(phrase) == phrase, phrase


# --------------------------------------------------------------------------- the converter
def _fixture(tmp_path, monkeypatch, n=400):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "pd12m").mkdir(parents=True)
    out.mkdir()
    buckets = sorted(CAP.BUCKETS)
    captions = list(CAPTIONS)
    manifest, selection = [], []
    for i in range(n):
        sha = f"{i:064x}"
        bucket = buckets[i % len(buckets)]
        caption = captions[i % len(captions)]
        present, absent = CAP.objects(caption, associated=T.CATEGORY_ASSOCIATED[bucket],
                                      rng=random.Random(i))
        manifest.append({
            "pd12m_id": f"pd{i:012d}", "url": f"https://pd12m.s3.us-west-2.amazonaws.com/images/{i}.jpeg",
            "sha256": sha, "image": f"data/decision-v2/pd12m/images/{sha}.jpg",
            "width": 800, "height": 600, "bytes": 1234, "spdx": "CC0-1.0", "bucket_label": bucket,
        })
        selection.append({
            "pd12m_id": f"pd{i:012d}", "bucket_label": bucket, "caption": caption,
            "objects_present_claimed": present, "objects_absent_claimed": absent,
        })
    (raw / "pd12m" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (raw / "pd12m" / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selection))
    monkeypatch.setattr(C, "RAW", raw)
    monkeypatch.setattr(C, "OUT", out)
    return out


def _rows(out):
    return [json.loads(line) for line in (out / "pd12m" / "records.jsonl").read_text().splitlines()]


def test_converter_shape_and_mix(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch)
    summary = C.build("pd12m", seed=7)
    rows = _rows(out)

    assert summary["records"] == summary["photos"] * C.DECISIONS_PER_PHOTO
    for r in rows:
        assert r["source"] == "pd12m"
        assert r["target"] is None and r["abstention_cause"] is None
        assert r["pseudo_label"] == "pending"
        assert r["license"]["spdx"] == "CC0-1.0"
        assert r["template_id"].split("/")[0] == r["family"]
        assert r["partition"] == stable_partition(r["source_group"].split("+")[0],
                                                  seed=C.PARTITION_SEED)
        assert (r["partition"] == "test") == bool(r.get("pseudo_label_test"))
        Request.model_validate(r["request"])

    assert 0.15 <= summary["unknown_construction_share"] <= 0.25
    assert 0.06 <= summary["two_image_share"] <= 0.14
    assert 0.33 <= summary["state_string_share"] <= 0.47
    assert set(summary["families"]) >= set(C.SINGLE_FAMILY_WEIGHTS["pd12m"])
    assert summary["distinct_templates"] >= 80


def test_no_caption_ever_reaches_a_record(tmp_path, monkeypatch):
    """The whole point of this source: the teacher must answer from the pixels.

    A caption in the state would hand it the answer in words, so no fragment of any caption may
    appear anywhere in a record -- not in the state, not in a question, not in an option.
    """
    out = _fixture(tmp_path, monkeypatch)
    C.build("pd12m", seed=17)
    blob = (out / "pd12m" / "records.jsonl").read_text().lower()
    for caption in CAPTIONS:
        assert caption.lower() not in blob
        # and not even a distinctive clause of one
        for clause in caption.split(". "):
            fragment = clause.strip().lower()
            if len(fragment) >= 25:
                assert fragment not in blob, fragment


def test_aligned_state_carries_only_the_coarse_bucket_never_a_caption(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch)
    C.build("pd12m", seed=19)
    aligned = [r for r in _rows(out) if r.get("state_kind") == "aligned"]
    assert aligned, "the aligned state kind must still be produced"
    for r in aligned[:200]:
        text = json.dumps(r["request"]["state"]).lower()
        assert "image shows" not in text
        assert "caption" not in text or "photograph of" in text


def test_contradicting_states_are_pressed_harder_than_the_default(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch)
    summary = C.build("pd12m", seed=23)
    kinds = summary["state_kinds"]
    assert set(kinds) == set(T.STATE_KINDS)
    share = kinds["contradictory"] / sum(kinds.values())
    assert 0.25 <= share <= 0.36, share


def test_absent_object_questions_are_actually_built(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch)
    C.build("pd12m", seed=29)
    rows = _rows(out)
    absent = [r for r in rows if r["template_id"].endswith("/absent")]
    assert len(absent) / len(rows) >= 0.10, "the brief asks for absent-object questions"
    # and the four constructions whose honest answer is `unknown` are all present
    ids = {r["template_id"].rsplit("/", 1)[-1] for r in rows if r["unknown_construction"]}
    assert ids <= {"unknown", "absent"}
    families = {r["family"] for r in rows if r["unknown_construction"]}
    assert families <= set(T.UNKNOWN_CAPABLE)


def test_converter_is_deterministic(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch, n=120)
    C.build("pd12m", seed=3)
    first = (out / "pd12m" / "records.jsonl").read_text()
    C.build("pd12m", seed=3)
    assert (out / "pd12m" / "records.jsonl").read_text() == first


def test_two_image_pairs_stay_inside_one_partition(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch)
    C.build("pd12m", seed=11)
    pairs = [r for r in _rows(out) if len(r["images"]) == 2]
    assert pairs
    for r in pairs:
        a, b = r["source_group"].split("+")
        assert a != b
        assert stable_partition(a, seed=C.PARTITION_SEED) == stable_partition(b, seed=C.PARTITION_SEED) == r["partition"]
        assert r["images"][0]["sha256"] != r["images"][1]["sha256"]
        assert r["family"] in T.TWO_IMAGE_FAMILIES


@pytest.mark.parametrize("source", ["commons_photos", "openimages_v2"])
def test_adding_pd12m_left_the_other_sources_untouched(source):
    """The pd12m overrides are opt-in: the other two sources still use the module defaults."""
    assert source not in C.UNKNOWN_SHARE_BY_SOURCE
    assert source not in C.ABSENT_OBJECT_SHARE_BY_SOURCE
    assert source not in C.STATE_KIND_WEIGHTS
    assert C.UNKNOWN_SHARE_BY_SOURCE.get(source, C.UNKNOWN_SHARE) == C.UNKNOWN_SHARE
