"""Trap gold re-derivation (scripts/p3/gen_traps.py, fix 2026-09-26) and the --gold-fixes path of build_manifest / make_extra_queue.

The bug: a trap of an UNKNOWN parent inherited gold null even when the transformed question is decidable, e.g. the negated score
question "is the band something other than Band 3?" when the missing engagement type only leaves Bands 0 and 1 possible. Every trap
gold is now re-derived by enumerating the parent's undetermined facts; these tests check that against an independent mapping.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p3"))
sys.path.insert(0, str(ROOT / "tests"))

import gen_policy as gp  # noqa: E402
import gen_traps as gt  # noqa: E402
from candidate import validate  # noqa: E402

FAMS = gp.FAMILY_NAMES
N_PER_FAM = {"long_policy": 80, "rule_exception": 160, "judge_hard": 120, "agent_action": 160, "long_input": 40, "multi_label": 30}


def module_for(row):
    for fam in FAMS:
        mod = gp.MODULES[fam]
        if row["family"] in getattr(mod, "FAMILIES", (fam,)):
            return fam, mod
    raise KeyError(row["family"])


@pytest.fixture(scope="module")
def parents():
    return [r for fam in FAMS for i in range(N_PER_FAM[fam]) for r in gp.gen_one(fam, 321, i)]


# ------------------------------------------------------------------------------------------------ the reported example
def test_reported_example_negated_band_is_decidable():
    """p3-trap-negation-p3-pol-rexc-s0-002901-u: engagement type unrecorded -> band 0 or 1; "other than Band 3?" is YES."""
    parent = next(r for r in gp.gen_one("rule_exception", 0, 2901) if r["id"] == "p3-pol-rexc-s0-002901-u")
    assert parent["gold"] is None and parent["provenance"]["spec"]["record"]["tier"] is None
    raw, _, _ = gt.parent_worlds(parent)
    assert set(raw) == {0, 1}                                   # the unrecorded tier leaves Band 0 or Band 1
    trap = gt.make_trap(parent, "negation", seed=0)
    assert trap["id"] == "p3-trap-negation-p3-pol-rexc-s0-002901-u"
    assert "Band 3" in trap["field"]["question"] and trap["provenance"]["trap_info"]["level"] == 3
    assert trap["gold"] is True and trap["unknown_reason"] is None
    assert trap["provenance"]["gold_derivation"]["method"] == "enumerated"
    assert validate(trap) == []


def test_negated_level_inside_the_possible_range_stays_unknown():
    parent = next(r for r in gp.gen_one("rule_exception", 0, 2901) if r["id"] == "p3-pol-rexc-s0-002901-u")
    trap = gt.make_trap(parent, "negation", seed=0)
    for L in (0, 1):                                            # a level some completion reaches: undetermined
        t = json.loads(json.dumps(trap))
        t["provenance"]["trap_info"]["level"] = L
        assert gt.rederive_gold(parent, t)[0] is None
    t["provenance"]["trap_info"]["level"] = 2
    assert gt.rederive_gold(parent, t)[0] is True


# ------------------------------------------------------------------------------------------------ independent enumeration
def _completions(parent):
    """Parent answers of every completion, computed here from the family worlds() + the test-side value mapping (mirrors
    tests/test_p3_gen_policy.py::recheck_matches), premise-violating completions ("invalid") ignored."""
    fam, mod = module_for(parent)
    spec = parent["provenance"]["spec"]
    ov = spec.get("opt_values")
    if fam == "long_policy" and spec.get("missing"):
        traced, tov = gt.regen_trace(parent)
        raw, ov = mod.worlds(spec, traced), ov or tov
    else:
        raw = mod.worlds(spec)
    f = parent["field"]
    out = []
    for v in raw:
        if v == "invalid":
            continue
        if f["type"] in ("noul", "score"):
            out.append(v if (isinstance(v, bool) if f["type"] == "noul" else isinstance(v, int) and not isinstance(v, bool)) else "?")
        elif ov:
            keys = [k for k, cv in ov.items() if json.dumps(cv, sort_keys=True) == json.dumps(v, sort_keys=True)]
            marker = v is None or (isinstance(v, str) and v.startswith("__") and v not in ("__not_an_option__", "__none__"))
            out.append(keys[0] if len(keys) == 1 else "?" if marker else "-none-")      # a real value no option states: none
        elif isinstance(v, int) and not isinstance(v, bool):
            out.append(f["options"][v]["key"])
        elif v == "equal":
            out.append(f["options"][2]["key"])
        elif isinstance(v, str) and not v.startswith("__"):
            hit = [o["key"] for o in f["options"] if o["text"].strip().lower() == v.strip().lower()]
            out.append(hit[0] if hit else "-none-")
        else:
            out.append("-none-" if v in ("__not_an_option__", "__none__") else "?")
    return out


def _expected(parent, trap):
    kind, info = trap["provenance"]["trap"], trap["provenance"]["trap_info"]
    comps = _completions(parent)
    pf, tf = parent["field"], trap["field"]
    per = []
    for a in comps:
        if kind == "negation" and pf["type"] == "choice":
            if a == "?":
                per.append("?")
                continue
            t1, t2 = (o["text"].split(": ", 1)[1].strip().lower() for o in tf["options"][:2])
            at = next((o["text"].strip().lower() for o in pf["options"] if o["key"] == a), None)
            per.append(tf["options"][1]["key"] if at == t1 else tf["options"][0]["key"] if at == t2 else tf["options"][3]["key"])
        elif a in ("?", "-none-"):
            per.append("?")
        elif kind == "negation" and pf["type"] == "score":
            per.append(a != info["level"])
        elif kind == "negation":
            per.append(not a)
        else:
            per.append(a)
    if per and "?" not in per and len({json.dumps(x) for x in per}) == 1:
        return per[0]
    return None


def test_every_unknown_parent_really_has_several_completions(parents):
    n = 0
    for p in parents:
        if p["gold"] is None:
            comps = [c for c in _completions(p) if c != "?"]
            assert len({json.dumps(c) for c in comps}) >= 2 or "?" in _completions(p), p["id"]
            n += 1
        else:
            assert set(map(json.dumps, _completions(p))) == {json.dumps(p["gold"])}, p["id"]
    assert n > 60


def test_property_trap_gold_equals_enumeration(parents):
    """Over many traps of every kind, answerable and unknown parents: gold == the enumerated completions' unique answer."""
    seen = Counter()
    decided_from_unknown = Counter()
    for p in parents:
        for kind in gt.KINDS:
            t = gt.make_trap(p, kind, seed=5)
            if t is None:
                continue
            assert validate(t) == []
            exp = _expected(p, t)
            assert json.dumps(t["gold"]) == json.dumps(exp), (t["id"], t["gold"], exp)
            assert (t["gold"] is None) == (t["unknown_reason"] is not None)
            seen[(kind, p["gold"] is None)] += 1
            if p["gold"] is None and t["gold"] is not None:
                decided_from_unknown[(kind, p["field"]["type"])] += 1
            if p["gold"] is not None:                                  # answerable parents: unchanged by construction
                if kind in ("bury", "near_miss", "qualifier"):
                    assert t["gold"] == p["gold"]
                elif p["field"]["type"] == "noul":
                    assert t["gold"] is (not p["gold"])
                elif p["field"]["type"] == "score":
                    assert t["gold"] is (t["provenance"]["trap_info"]["level"] != p["gold"])
            elif kind != "negation":
                assert t["gold"] is None                               # an added option / notice cannot decide an unknown item
            elif p["field"]["type"] == "noul":
                assert t["gold"] is None
    assert sum(seen.values()) > 1500 and all(seen[(k, u)] > 5 for k in gt.KINDS for u in (False, True) if k != "qualifier")
    assert sum(decided_from_unknown.values()) > 0, decided_from_unknown       # the bug class occurs and is now labelled


def test_foreign_rows_without_solver_never_get_a_risky_unknown_negation():
    row = {"id": "p3-reason-9", "source": "A", "dataset": "gen_reasoning", "family": "logic_chain", "difficulty": 4,
           "state": "Some facts.", "images": [], "field": {"type": "score", "question": "How many?",
                                                           "levels": [{"value": i, "description": f"{i}"} for i in range(4)]},
           "gold": None, "unknown_reason": "insufficient_evidence", "gold_kind": "constructed", "parent_id": None,
           "provenance": {"licence": "generated"}}
    assert gt.make_trap(row, "negation") is None
    b = gt.make_trap(row, "bury")
    assert b is not None and b["gold"] is None and b["provenance"]["gold_derivation"]["method"] == "construction"


# ------------------------------------------------------------------------------------------------ the regenerated D file
D_FILE = ROOT / "data/p3/candidates/D-traps.jsonl"


@pytest.mark.skipif(not D_FILE.exists(), reason="no D-traps.jsonl")
def test_d_file_sample_gold_equals_enumeration():
    """500 stored D rows (every unknown-parent negation + a stratified rest): parent regenerated from its seed/index, gold
    re-derived independently here."""
    rows = [json.loads(l) for l in D_FILE.open()]
    pick = [r for r in rows if re.search(r"-u\d*$", r["parent_id"]) and r["provenance"]["trap"] == "negation"][:300]
    rest = [r for r in rows if r not in pick]
    pick += rest[:: max(1, len(rest) // 200)][:200]
    assert len(pick) >= 400
    for t in pick:
        prov = t["provenance"]
        p = gt._regen_policy_row(t["parent_id"], t["family"], prov["seed"], prov["index"])
        assert p is not None, t["id"]
        assert json.dumps(p["provenance"]["spec"], sort_keys=True) == json.dumps(prov["spec"], sort_keys=True)
        assert json.dumps(t["gold"]) == json.dumps(_expected(p, t)), t["id"]
    ex = next(r for r in rows if r["id"] == "p3-trap-negation-p3-pol-rexc-s0-002901-u")
    assert ex["gold"] is True


# ------------------------------------------------------------------------------------------------ fixes file consumers
def _fixes(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_and_apply_gold_fixes(tmp_path):
    p = _fixes(tmp_path / "f.jsonl", [
        {"id": "a", "old_gold": None, "new_gold": True, "new_unknown_reason": None, "where": "x", "action": "relabel"},
        {"id": "a", "old_gold": None, "new_gold": True, "new_unknown_reason": None, "where": "y", "action": "relabel"},
        {"id": "b", "old_gold": None, "new_gold": "k", "new_unknown_reason": None, "where": "t#kept", "action": "drop"}])
    rl, dr, n = gt.load_gold_fixes(p)
    assert rl == {"a": (True, None)} and dr == {"b"} and n == {"relabel_ids": 1, "drop_ids": 1}
    row = {"id": "a", "gold": None, "unknown_reason": "insufficient_evidence",
           "label": {"target": None, "probs": None, "rationale": None, "target_kind": "constructed", "review": None}}
    assert gt.apply_gold_fix(row, rl) and row["gold"] is True and row["unknown_reason"] is None and row["label"]["target"] is True
    teacher = {"id": "a", "gold": None, "label": {"target": "false", "target_kind": "teacher"}}
    gt.apply_gold_fix(teacher, rl)
    assert teacher["label"]["target"] == "false"                   # a teacher label is never rewritten
    assert gt.label_contradicts("false", True) and not gt.label_contradicts("true", True)
    assert not gt.label_contradicts("__unknown__", None) and gt.label_contradicts(None, "k")


def test_make_extra_queue_applies_gold_fixes_to_direct_leftovers(tmp_path):
    import make_extra_queue as X
    from test_p3_labeltrain import azure_fixture, read_jsonl, xargs
    azure_fixture(tmp_path)
    fx = _fixes(tmp_path / "fixes.jsonl", [
        {"id": "it-6", "old_gold": "b", "new_gold": "c", "new_unknown_reason": None, "where": "q", "action": "drop"},
        {"id": "it-9", "old_gold": "b", "new_gold": None, "new_unknown_reason": "insufficient_evidence", "where": "q",
         "action": "relabel"}])
    assert X.main(xargs(tmp_path, "extra", "--gold-fixes", str(fx))) == 0
    direct = {r["id"]: r for r in read_jsonl(tmp_path / "labeltrain/direct-constructed-leftovers.jsonl")}
    assert "it-6" not in direct and "it-12" in direct and direct["it-12"]["gold"] == "b"
    assert direct["it-9"]["gold"] is None and direct["it-9"]["label"]["target"] is None
    assert direct["it-9"]["unknown_reason"] == "insufficient_evidence"
    s = json.loads((tmp_path / "labeltrain/extra-summary.json").read_text())
    assert s["gold_fixes"] == {"direct_dropped": 1, "direct_relabelled": 1}


def test_build_manifest_applies_gold_fixes(tmp_path):
    import assembly_common as ac
    import assembly_manifest as am
    import azure_teacher as T
    import build_manifest as bm
    from test_p3_assembly import _fixture_pool
    from test_p3_labeltrain import teacher_dir, write_jsonl
    _fixture_pool(tmp_path)
    rows = [r for r in ac.read_jsonl(tmp_path / "shards" / "pool-00.jsonl") if not r["pool"]["heldout_flagged_bucket"]]
    text = [r for r in rows if not r.get("images")][:260]
    imgs = [r for r in rows if r.get("images")][:80]

    def kept_row(r):
        labs = am.candidate_labels(r)
        gold = am.to_label(r, r["gold"])
        probs = {l: (0.9 if l == gold else 0.1 / (len(labs) - 1)) for l in labs}
        return {"id": r["id"], "keep": True, "rule": "A", "origin": "mine", "item": r,
                "label": T.label_for(r, {"keep": True, "target": "teacher_dist", "target_probs": probs}, ["Rationale."])}
    kept = text[:200] + imgs
    teacher_dir(tmp_path / "teacher", [{"id": r["id"], "keep": True} for r in kept], [kept_row(r) for r in kept])
    # direct leftover rows (constructed) taken from the remaining text rows
    left = []
    for r in text[200:230]:
        c = {k: v for k, v in r.items() if k != "pool"}
        c["gold_kind"] = "constructed"
        c["label"] = {"target": c["gold"], "probs": None, "rationale": None, "target_kind": "constructed", "review": None}
        left.append(c)
    write_jsonl(tmp_path / "leftovers.jsonl", left)
    drop_id = text[0]["id"]                                         # a teacher-kept row whose kept label contradicts
    contra = text[1]                                                # relabel contradicting the kept teacher label -> dropped too
    agree = text[2]                                                 # relabel that agrees with the teacher label -> kept
    lr = left[0]
    other = next(o["key"] for o in lr["field"]["options"] if o["key"] != lr["gold"]) if lr["field"]["type"] == "choice" else None
    new_left_gold = other if other is not None else None
    fx = _fixes(tmp_path / "fixes.jsonl", [
        {"id": drop_id, "old_gold": None, "new_gold": True, "where": "data/p3/teacher/results.jsonl#kept", "action": "drop"},
        {"id": contra["id"], "old_gold": contra["gold"], "new_gold": "__changed__", "where": "pool", "action": "relabel"},
        {"id": agree["id"], "old_gold": None, "new_gold": agree["gold"], "where": "pool", "action": "relabel"},
        {"id": lr["id"], "old_gold": lr["gold"], "new_gold": new_left_gold, "new_unknown_reason": "insufficient_evidence",
         "where": "leftover", "action": "relabel"}])
    args = ["--teacher-dirs", str(tmp_path / "teacher"), "--direct-leftovers", str(tmp_path / "leftovers.jsonl"),
            "--pool-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--total", "100000",
            "--replay-p2", str(tmp_path / "replay-text.jsonl"), "--replay-eikos", "", "--replay-v21", "",
            "--replay-images", str(tmp_path / "replay-img.jsonl"), "--publish", "decision-p3", "--publish-dir", str(tmp_path / "man")]
    assert bm.main(args + ["--gold-fixes", str(fx)]) in (0, 1)
    rep = json.loads((tmp_path / "out/manifest-report.json").read_text())
    g = rep["gold_fixes"]
    assert g["relabel_ids_in_file"] == 3 and g["drop_ids_in_file"] == 1
    assert g["verified_dropped"] == 1 and g["verified_dropped_teacher_label_contradicts_new_gold"] == 1
    assert g["verified_relabelled"] == 2 and g["pool_rows_relabelled"] >= 3
    labelled = {r["id"] for r in ac.read_jsonl(tmp_path / "man/decision-p3-labelled.jsonl")}
    assert drop_id not in labelled and contra["id"] not in labelled
    # without the flag nothing is dropped
    assert bm.main(args + ["--out-dir", str(tmp_path / "out2")]) in (0, 1)
    rep2 = json.loads((tmp_path / "out2/manifest-report.json").read_text())
    assert rep2["gold_fixes"]["files"] == [] and "verified_dropped" not in rep2["gold_fixes"]
