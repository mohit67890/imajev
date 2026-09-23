import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from imajev_bench.annotations import merge_reviews
from imajev_bench.assemble import assemble
from imajev_bench.cli import read_jsonl
from imajev_bench.lint import ABSTENTION_CUE, lint
from imajev_bench.runner import digest
from imajev_bench.schema import model_payload, validate_records
from imajev_bench.scenes import BUILDERS, answer, plan_scenes, render
from imajev_bench.textgen import text_items
from imajev_bench.triage import audit_report, packet_sha256, plan_construction

spec = importlib.util.spec_from_file_location("build_v2", Path(__file__).resolve().parents[1] / "scripts/imajev_bench/build_v2.py")
build_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_v2)


def _domain(field):
    if field["type"] == "boolean":
        return {True, False}
    if field["type"] == "choice":
        return {o["value"] for o in field["options"]}
    return {level["value"] for level in field["levels"]}


def test_every_question_answer_is_in_domain_and_wording_never_cues_unknown():
    scenes = plan_scenes(240, seed=7)
    assert Counter(s.kind for s in scenes) == Counter({k: 240 // len(BUILDERS) for k in BUILDERS}) if 240 % len(BUILDERS) == 0 else True
    assert abs(Counter(s.generator for s in scenes)["flare"] - 120) <= len(BUILDERS)
    unknowns = 0
    for scene in scenes:
        assert 2 <= len(scene.questions) <= 3
        for qi, q in enumerate(scene.questions):
            track, family, state, field = render(q["template"], q["params"])
            text = field["question"] + json.dumps(state)
            assert not ABSTENTION_CUE.search(text), text
            images = [scene.facts] + ([v.facts for v in scene.variants] if qi == 0 else [])
            for facts in images:
                gold = answer(q["template"], facts, q["params"])
                unknowns += gold is None
                assert gold is None or gold in _domain(field), (scene.kind, q["template"], gold, field)
    assert unknowns > 0


def test_hidden_values_make_answers_unknown_and_edits_change_them():
    scenes = [s for s in plan_scenes(80, seed=3) if s.kind in ("menu", "timetable", "sign", "nutrition")]
    for scene in scenes:
        q = scene.questions[0]
        base = answer(q["template"], scene.facts, q["params"])
        assert base is not None
        for variant in scene.variants:
            gold = answer(q["template"], variant.facts, q["params"])
            if variant.name == "cover" and scene.kind == "sign":
                # Hidden hours matter only when the stay exceeds the limit on a restricted day.
                over = q["params"]["stay_minutes"] > scene.facts["limit_hours"] * 60
                assert (gold is None) == over and (over or gold is True), (scene.id, gold)
            elif variant.name == "cover":
                assert gold is None, (scene.kind, variant.name)
            if variant.name == "same":
                assert gold == base


def test_timetable_hidden_time_only_matters_when_it_could_be_next():
    facts = {"route": 42, "times": ["07:00", None, "08:00"]}
    assert answer("timetable_next", facts, {"now": "07:01"}) is None
    assert answer("timetable_next", facts, {"now": "06:30"}) == "07:00"


def test_text_items_are_computed_and_balanced():
    items = text_items(90, seed=2)
    assert Counter(i["family"] for i in items) == Counter({"numerical_reconciliation": 30, "policy_precedence": 30,
                                                          "missing_conflicting_evidence": 30})
    assert 5 <= sum(i["draft_gold"] is None for i in items) <= 40
    for item in items:
        assert item["construction"]["truth"] == item["draft_gold"]
        assert not ABSTENTION_CUE.search(item["field"]["question"] + json.dumps(item["state"]).replace("authoritative_source", ""))


def fake_build(tmp_path, scenes=16, text=6):
    out = tmp_path / "build"

    class Args:
        pass
    args = Args()
    args.out, args.scenes, args.text, args.seed, args.kinds, args.flare_share = out, scenes, text, 5, None, 0.5
    build_v2.cmd_plan(args)
    plan = json.loads((out / "plan.json").read_text())
    for i, raw in enumerate(plan["scenes"]):
        folder = out / "scenes" / raw["id"]
        folder.mkdir(parents=True)
        names = ["base"] + [v["name"] for v in raw["variants"] if not (i % 5 == 0 and v["name"] == "same")]
        for n, name in enumerate(names):
            Image.new("RGB", (48, 32), ((i * 37) % 255, n * 60, 90)).save(folder / f"{name}.jpg")
        (folder / "result.json").write_text(json.dumps({"scene": raw["id"], "generator": raw["generator"], "model": "fake-model",
                                                        "images": {name: f"{name}.jpg" for name in names},
                                                        "attempts": {}, "estimated_usd": 0}))
    args.dataset, args.version, args.split_salt, args.dev, args.calibration, args.pairs = "t", "0", "salt", 0.3, 0.15, True
    build_v2.cmd_items(args)
    return out


def test_items_phase_builds_valid_assembled_dataset(tmp_path):
    out = fake_build(tmp_path)
    receipt = assemble(json.loads((out / "spec.json").read_text()), out, tmp_path / "dataset")
    records = read_jsonl(tmp_path / "dataset" / "records.jsonl")
    assert receipt["records"] == len(records) > 40
    per_image = Counter(i["sha256"] for r in records for i in r["images"])
    assert max(per_image.values()) <= 3
    sets = [r for r in records if "contrast" in r["provenance"]]
    assert sets and all(r["provenance"]["construction"]["truth"] == r["gold"] for r in records)
    report = {c["check"]: c for c in lint(records, tmp_path / "dataset")["checks"]}
    assert report["contrast_sets"]["status"] != "fail" and report["abstention_cues"]["status"] == "pass"
    assert report["split_isolation"]["status"] == "pass" and report["provenance"]["status"] == "pass"


def test_construction_route_promotes_and_audits(tmp_path):
    out = fake_build(tmp_path)
    assemble(json.loads((out / "spec.json").read_text()), out, tmp_path / "dataset")
    records = read_jsonl(tmp_path / "dataset" / "records.jsonl")
    planned, summary = plan_construction(records, audit_share=0.2, seed=1)
    planned = validate_records(planned, tmp_path / "dataset")
    audited = [r for r in planned if r["provenance"]["audit_sample"]]
    assert summary["audit_sample"] == len(audited) > 0 and summary["promoted"] == len(planned) - len(audited)
    validate_records([r for r in planned if not r["provenance"]["audit_sample"]], tmp_path / "dataset", require_reviewed=True)
    wrong = audited[0]["id"]
    flipped = {r["id"]: r["gold"] for r in audited}
    flipped[wrong] = None if audited[0]["gold"] is not None else next(iter(_domain(audited[0]["request"]["fields"][0])))
    export = {"format_version": "0.0.1", "protocol": "imajev-bench-blind-review-v1", "reviewer_id": "human-a",
              "purpose": "independent_review", "input_sha256": packet_sha256(audited), "flags": [],
              "reviews": [{"id": r["id"], "reviewer_id": "human-a", "value": flipped[r["id"]], "evidence": "looked",
                           "input_sha256": digest(model_payload(r))} for r in audited]}
    merged = merge_reviews(planned, [export], tmp_path / "dataset")
    by_id = {r["id"]: r for r in merged}
    assert by_id[wrong]["annotation_status"] == "draft" and by_id[wrong]["provenance"]["requires_second_review"]
    assert all(by_id[r["id"]]["annotation_status"] == "reviewed" for r in audited if r["id"] != wrong)
    assert audit_report(merged)["audited"] == len(audited) - 1
