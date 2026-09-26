"""Phase-3 image branch: candidate validity, gold re-checks, unknowns, image paths, licences.

Covers data/p3/candidates/I-joint.jsonl (scripts/p3/gen_image_joint.py), C-images.jsonl (convert_own_images.py) and
I-gui.jsonl (gen_gui.py). Skips a file's tests when it has not been generated.

    .venv/bin/python -m pytest tests/test_p3_images.py -q
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))

import candidate  # noqa: E402
import gen_image_joint as J  # noqa: E402

CAND = ROOT / "data" / "p3" / "candidates"
FILES = {"joint": CAND / "I-joint.jsonl", "own": CAND / "C-images.jsonl", "gui": CAND / "I-gui.jsonl"}
ID_RE = __import__("re").compile(r'"id": "([^"]+)"')
PHOTO_LICENCES = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "CC-BY-3.0", "CC-BY-2.0"}


def rows(name):
    p = FILES[name]
    if not p.is_file():
        pytest.skip(f"{p.name} not generated")
    return list(candidate.read(p))


@pytest.fixture(scope="module")
def joint():
    return rows("joint")


@pytest.fixture(scope="module")
def own():
    return rows("own")


@pytest.fixture(scope="module")
def gui():
    return rows("gui")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ all three files
@pytest.mark.parametrize("name", ["joint", "own", "gui"])
def test_valid_unique_and_images_exist(name):
    rs = rows(name)
    ids = [r["id"] for r in rs]
    assert len(set(ids)) == len(ids)
    seen = set()
    for r in rs:
        assert candidate.validate(r) == [], r["id"]
        assert r["images"], r["id"]
        for im in r["images"]:
            assert isinstance(im, str) and not im.startswith("data/") and not im.startswith("/")
            if im not in seen:
                assert (ROOT / "data" / im).is_file(), im
                seen.add(im)


@pytest.mark.parametrize("name", ["joint", "own", "gui"])
def test_licence_recorded(name):
    for r in rows(name):
        pv = r["provenance"]
        if name == "gui":
            assert pv["licence"] == "generated"
        else:
            assert set(pv["image_licences"]) <= PHOTO_LICENCES, r["id"]
            assert pv["licence_evidence"] and all((ROOT / e).is_file() for e in pv["licence_evidence"]), r["id"]


@pytest.mark.parametrize("name", ["joint", "own", "gui"])
def test_no_imajevbench_image(name):
    bench = J.bench_image_shas()
    rs = rows(name)
    for r in rs:
        assert not set(r["provenance"]["image_sha256"]) & bench, r["id"]
    # the recorded sha256 is the file's content (sample)
    for r in random.Random(0).sample(rs, min(300, len(rs))):
        for im, s in zip(r["images"], r["provenance"]["image_sha256"]):
            assert sha(ROOT / "data" / im) == s, im


# ------------------------------------------------------------------------------------------------ joint rule generator
def test_joint_mix(joint):
    assert 14000 <= len(joint) <= 16000
    assert {r["family"] for r in joint} == {"image_joint_rule"}
    assert all(3 <= r["difficulty"] <= 5 for r in joint)
    kinds = Counter(r["provenance"]["rule_kind"] for r in joint)
    assert set(kinds) == {"threshold_rule", "rule_exception", "multi_step_rule", "two_image"}
    unk = sum(r["gold"] is None for r in joint) / len(joint)
    assert 0.12 <= unk <= 0.18
    assert sum(len(r["images"]) == 2 for r in joint) > 1500
    assert all(r["gold_kind"] == "constructed" for r in joint)


def test_joint_gold_rederived_from_source_labels(joint):
    """Reload the photo labels from the source annotations (not from the row) and re-derive every sampled gold."""
    abo, dfx, _ = J.load_context()
    facts = {p["sha256"]: p["facts"] for p in abo + dfx}
    sample = random.Random(1).sample(joint, 1500)
    for r in sample:
        pv = r["provenance"]
        fl = [facts[s] for s in pv["image_sha256"]]
        assert fl == pv["photo_facts"], r["id"]
        gold, unk = J.decide(pv["template"], pv["params"], fl)
        assert (gold, unk) == (r["gold"], r["unknown_reason"]), r["id"]


def test_joint_unknowns_are_real(joint):
    by_id = {r["id"]: r for r in joint}
    abo, dfx, _ = J.load_context()
    facts = {p["sha256"]: p["facts"] for p in abo + dfx}
    n_var = 0
    for r in joint:
        if r["gold"] is not None:
            continue
        pv = r["provenance"]
        assert r["unknown_reason"] == "insufficient_evidence"
        if r["parent_id"]:
            n_var += 1
            par = by_id[r["parent_id"]]
            assert par["gold"] is not None and par["images"] == r["images"]
            w = pv["params"]["_withheld"][0]
            # the withheld field (or lookup row) is really absent from what the model reads
            text = json.dumps(r["state"])
            fl = [facts[s] for s in pv["image_sha256"]]
            if w in J.TEMPLATES[pv["template"]][7]:
                assert json.dumps(par["state"]) != text
            else:
                assert f'"{w}"' not in text and f'"{w}"' in json.dumps(par["state"])
            # and the parent's gold follows once it is put back
            assert J.decide(pv["template"], par["provenance"]["params"], fl)[0] == par["gold"]
        else:
            assert pv["hidden_attribute"] and pv["unknown_construction"] == "hidden_attribute"
    assert n_var > 1000


def test_joint_state_does_not_name_the_photo_attribute(joint):
    """The record never states the attribute the rule tests on the photo (colour / material / type)."""
    for r in random.Random(2).sample(joint, 2000):
        pv = r["provenance"]
        st = r["state"]
        record = {k: v for k, v in st.items() if k not in ("rule", "policy", "handling_fee_per_unit", "points_per_defect_kind",
                                                             "rework_list", "photo", "photos")}
        text = json.dumps(record).lower()
        for f in pv["photo_facts"]:
            for key in ("colour", "material", "product_type"):
                if key in f:
                    assert f'"{f[key]}"' not in text, (r["id"], key)
            if "defective" in f:
                assert "defect" not in text or "defective_so_far" in text, r["id"]


def test_joint_uses_train_photos_only(joint):
    for r in joint:
        assert r["provenance"]["upstream_split"] == "train"


# ------------------------------------------------------------------------------------------------ own pools
def test_own_pool_shape(own):
    oi = [r for r in own if "openimages_v2" in r["provenance"]["image_pools"]]
    base = [r for r in own if "openimages_v2" not in r["provenance"]["image_pools"]]
    assert 38000 <= len(base) <= 40000
    assert 9000 <= len(oi) <= 10000
    assert all(r["source"] == "C" for r in own)
    fam = Counter(r["family"] for r in base)
    assert max(fam.values()) <= 1500          # family balance of the sampled 40k: no family dominates
    # Open Images: ONLY the state-grounded rows, CC BY 2.0 with a per-photo Flickr attribution
    for r in oi:
        pv = r["provenance"]
        assert r["dataset"] == "state_grounded" and pv["upstream_id"].startswith("state_grounded:"), r["id"]
        assert pv["photo_licence"] == "CC-BY-2.0" and pv["licence"] == "CC-BY-2.0"
        assert all(a and a["author"] and a["flickr_landing_url"] for a in pv["photo_attribution"]), r["id"]
    assert not any(im.startswith("decision-v2/openimages_v2/") for r in base for im in r["images"])
    assert sum(len(r["images"]) == 2 for r in own) > 3000
    allowed_pools = ("decision-v1/abo/images/", "decision-v1/defects/images/", "decision-v1/vizwiz/images/",
                     "decision-v1/vizwiz_quality/images/", "decision-v2/pd12m/images/",
                     "decision-v2/pairs_grounded/images_edited/", "decision-v2/commons_photos/images/",
                     "decision-v2/openimages_v2/images/")
    for r in own:
        assert all(im.startswith(allowed_pools) for im in r["images"]), r["id"]
        assert r["provenance"]["upstream_id"] and r["provenance"]["upstream_split"] == "train"


def test_own_gold_matches_manifest(own):
    sample = random.Random(3).sample(own, 800)
    want = {r["provenance"]["upstream_id"].split("::field=")[0]: r for r in sample}
    found = {}
    with open(ROOT / "data" / "manifests" / "decision-v2.1-4b.jsonl") as fh:
        for line in fh:
            if any(i in want for i in ID_RE.findall(line)):
                m = json.loads(line)
                if m["id"] in want:
                    found[m["id"]] = m
    assert len(found) == len(want)
    for mid, r in want.items():
        m = found[mid]
        uid = r["provenance"]["upstream_id"]
        fid = uid.split("::field=")[1] if "::field=" in uid else None
        if fid is None and "targets" in m:          # a one-field record in the multi-field format
            fid = next(iter(m["targets"]))
        target = m["targets"][fid] if fid else m.get("target")
        fl = next(f for f in m["request"]["fields"] if fid is None or f["id"] == fid)
        if target is None:
            assert r["gold"] is None and r["unknown_reason"]
        elif fl["type"] == "boolean":
            assert r["gold"] is target
        elif fl["type"] == "choice":
            text = {o["key"]: o["text"] for o in r["field"]["options"]}[r["gold"]]
            assert text == str(target)
        else:
            levels = sorted(l["value"] for l in fl["levels"])
            assert levels[r["gold"]] == target
        assert r["field"]["question"] == fl["question"]


# ------------------------------------------------------------------------------------------------ GUI screens
def _gui_gold(r):
    m, t = r["provenance"]["screen_meta"], r["provenance"]["task"]
    if t == "form_ready":
        return not m["invalid"] and (not m["has_terms"] or m["terms_on"])
    if t == "form_invalid_field":
        return m["labels"][m["invalid"][0]] if m["invalid"] else None
    if t == "toolbar_goal":
        return m["correct"] if m["mode"] == "present" else None
    if t == "toolbar_available":
        return m["mode"] == "present"
    if t == "settings_which":
        return m["target"] if m["present"] else None
    if t == "settings_state":
        return m["state"][m["target"]] if m["present"] else None
    if t == "dialog_button":
        return m["gold"]
    if t == "cart_free_shipping":
        return round(sum(l[1] * l[2] for l in m["lines"]), 2) >= m["threshold"]
    if t == "cart_over_limit":
        over = [l[0] for l in m["lines"] if l[3] and l[2] > l[3]]
        assert len(over) <= 1
        return over[0] if over else None
    raise KeyError(t)


def test_gui_gold_rederived(gui):
    assert 4500 <= len(gui) <= 5500
    for r in gui:
        g = _gui_gold(r)
        if r["field"]["type"] == "choice":
            text = {o["key"]: o["text"] for o in r["field"]["options"]}
            got = text[r["gold"]] if r["gold"] is not None else None
            assert got == g, r["id"]
            assert [o["text"] for o in r["field"]["options"]] == [str(x) for x in (
                r["provenance"]["screen_meta"]["labels"])], r["id"]
        else:
            assert r["gold"] == g, r["id"]
        assert (r["gold"] is None) == (r["unknown_reason"] is not None)


def test_gui_form_values_follow_their_rules(gui):
    """Recheck the invalid field against the printed rule text with independent checks (sample of form screens)."""
    import re
    checks = {
        "Digits only, exactly 10": lambda v: bool(re.fullmatch(r"\d{10}", v)),
        "5 digits": lambda v: bool(re.fullmatch(r"\d{5}", v)),
        "DD/MM/YYYY": lambda v: bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", v)),
        "Between 1 and 8": lambda v: v.isdigit() and 1 <= int(v) <= 8,
        "Up to 500.00": lambda v: float(v) <= 500,
        "4-16 letters or digits, no spaces": lambda v: bool(re.fullmatch(r"[A-Za-z0-9]{4,16}", v)),
        "Format: name@domain": lambda v: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v)),
        "8 characters: capital letters and digits": lambda v: bool(re.fullmatch(r"[A-Z0-9]{8}", v)),
    }
    n = 0
    for r in gui:
        if r["provenance"]["task"] != "form_ready":
            continue
        m = r["provenance"]["screen_meta"]
        for i, (rule, v) in enumerate(zip(m["rules"], m["values"])):
            ok = v != "" if i in m["required"] else True
            if v and rule in checks:
                ok = ok and checks[rule](v)
            if rule in checks or not rule:
                assert ok == (i not in m["invalid"]), (r["id"], rule, v)
                n += 1
    assert n > 1000


def test_gui_images_are_small_renders(gui):
    from PIL import Image
    for im in {r["images"][0] for r in random.Random(4).sample(gui, 200)}:
        w, h = Image.open(ROOT / "data" / im).size
        assert w <= 1280 and h <= 1000
        assert im.startswith("p3/images/gui/")


def test_gui_unknown_share(gui):
    unk = sum(r["gold"] is None for r in gui) / len(gui)
    assert 0.08 <= unk <= 0.2
