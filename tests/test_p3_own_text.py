"""scripts/p3/convert_own_text.py (phase-3 Stage 0 source C, text): mapping, split rules and selection on small in-memory fixtures."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p3"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import convert_own_text as cot  # noqa: E402
from candidate import validate  # noqa: E402


def rec(rid="p2_teacher:doc-1:0", field=None, target=True, cause=None, partition="train", source="p2_teacher", domain="finance_ops",
        spdx="Apache-2.0", **kw):
    field = field or {"id": "decision", "type": "boolean", "question": "Is the invoice overdue on the review date?"}
    r = {"id": rid, "source": source, "source_split": "teacher", "source_group": rid.split(":")[1], "partition": partition,
         "family": "date_number_trap", "domain": domain, "license": {"spdx": spdx, "evidence": "x"}, "images": [],
         "request": {"schema_version": "1.0", "request_id": "r", "state": "INVOICE 12 ... due 2024-03-01", "fields": [field]},
         "target": target, "abstention_cause": cause}
    r.update(kw)
    return r


def build_c(r, **kw):
    args = dict(group="p2-teacher", dataset="decision-p2/teacher", family=r["family"], record=r, field=r["request"]["fields"][0],
                target=r["target"], cause=r["abstention_cause"], licence="own", manifest="decision-p2", base_difficulty=3)
    args.update(kw)
    return cot.candidate(**args)


def test_boolean_maps_to_noul_with_record_provenance():
    c = build_c(rec(field={"id": "decision", "type": "boolean", "question": "Is the invoice overdue?", "yes_description": "y",
                           "no_description": "n"}))
    assert validate(c) == []
    assert c["source"] == "C" and c["field"]["type"] == "noul" and c["gold"] is True and c["gold_kind"] == "dataset"
    assert c["field"]["yes_description"] == "y"
    pv = c["provenance"]
    assert pv["manifest"] == "decision-p2" and pv["record_id"] == "p2_teacher:doc-1:0" and pv["partition"] == "train"
    assert pv["upstream_split"] == "teacher" and pv["licence"] == "own" and pv["source_licence"]["spdx"] == "Apache-2.0"


def test_choice_keeps_option_values_as_keys_and_target_as_gold():
    f = {"id": "decision", "type": "choice", "question": "Which net payable amount is correct?",
         "options": [{"value": "7_298_00", "description": "net of depreciation"}, {"value": "8_900_00", "description": "gross"}]}
    c = build_c(rec(field=f, target="7_298_00"))
    assert validate(c) == []
    assert [o["key"] for o in c["field"]["options"]] == ["7_298_00", "8_900_00"] and c["gold"] == "7_298_00"
    assert c["field"]["options"][0]["description"] == "net of depreciation"


def test_ordinal_levels_reindexed_zero_based_and_values_kept():
    f = {"id": "decision", "type": "ordinal", "question": "Which priority level applies to the case?",
         "levels": [{"value": 1, "description": "routine"}, {"value": 2, "description": "supervisor"}, {"value": 3, "description": "halt"}]}
    c = build_c(rec(field=f, target=3))
    assert validate(c) == []
    assert [l["value"] for l in c["field"]["levels"]] == [0, 1, 2] and c["gold"] == 2
    assert c["provenance"]["level_values"] == [1, 2, 3]


def test_unknown_target_gets_null_gold_and_reason():
    c = build_c(rec(target=None, cause="not_listed"))
    assert validate(c) == [] and c["gold"] is None and c["unknown_reason"] == "not_listed"
    c = build_c(rec(target=None, cause="teacher_unknown"))
    assert validate(c) == [] and c["unknown_reason"] == "insufficient_evidence"
    assert c["provenance"]["abstention_cause_original"] == "teacher_unknown"


def test_target_probs_kept_in_provenance_with_string_keys():
    c = build_c(rec(), target_probs=cot.probs_dict({True: 0.9, False: 0.05, "unknown": 0.05}))
    assert c["provenance"]["target_probs"] == {"true": 0.9, "false": 0.05, "unknown": 0.05}
    json.dumps(c)


def test_candidate_round_trips_through_the_trainer_renderer():
    from decision_data import render
    f = {"id": "decision", "type": "ordinal", "question": "Which priority level applies to the case?",
         "levels": [{"value": 1, "description": "routine"}, {"value": 2, "description": "supervisor"}, {"value": 3, "description": "halt"}]}
    r = rec(field=f, target=2)
    c = build_c(r)
    lv = c["provenance"]["level_values"]
    back = {"id": c["id"], "target": lv[c["gold"]], "request": {"schema_version": "1.0", "request_id": "x", "state": c["state"], "fields": [
        {"id": "decision", "type": "ordinal", "question": c["field"]["question"],
         "levels": [{"value": lv[l["value"]], "description": l["description"]} for l in c["field"]["levels"]]}]}}
    assert render(back)[3] == render(r)[3] == 1


def test_ids_are_stable_and_distinct_per_field():
    a, b = build_c(rec()), build_c(rec())
    assert a["id"] == b["id"]
    two = rec(field={"id": "a", "type": "boolean", "question": "Is the first thing true?"})
    two["request"]["fields"].append({"id": "b", "type": "boolean", "question": "Is the second thing true?"})
    ca = build_c(two, field=two["request"]["fields"][0])
    cb = build_c(two, field=two["request"]["fields"][1])
    assert ca["id"] != cb["id"] and cb["provenance"]["field_id"] == "b"


def test_long_documents_are_harder():
    short = build_c(rec())
    long_r = rec()
    long_r["request"]["state"] = "word " * 3000
    assert build_c(long_r)["difficulty"] > short["difficulty"]
    assert 1 <= short["difficulty"] <= 5


def _write(tmp_path, name, rows):
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_p2_keeps_only_train_non_holdout_and_reads_replay_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cot, "MANIFESTS", tmp_path)
    p2 = [rec("p2_teacher:a:0"), rec("p2_teacher:b:0", partition="dev"), rec("p2_teacher:c:0", domain="telecom"),
          rec("p2_gsm8k:train-1", source="p2_gsm8k", spdx="MIT", domain=None, source_group="gsm8k:train-1"),
          rec("p2_strategyqa:q1", source="p2_strategyqa", spdx="MIT", domain=None, source_group="strategyqa:q1")]
    p2b = [rec("p2_teacher:a:0", replay=True), rec("p2_teacher:d:0", partition="calibration"), rec("p2_teacher:e:0")]
    _write(tmp_path, "decision-p2", p2)
    _write(tmp_path, "decision-p2b", p2b)
    b = _write(tmp_path, "B", [{"provenance": {"upstream_id": "q1"}}])
    teacher, human = cot.load_p2(b_strategyqa=b)
    assert sorted(c["provenance"]["record_id"] for c in teacher) == ["p2_teacher:a:0", "p2_teacher:e:0"]
    assert [c["provenance"]["record_id"] for c in human] == ["p2_gsm8k:train-1"]
    h = human[0]
    assert h["provenance"]["licence"] == "MIT" and h["provenance"]["upstream_dataset"] == "gsm8k" and h["provenance"]["upstream_id"] == "train-1"
    assert all(validate(c) == [] for c in teacher + human)


def test_load_eikos_keeps_attribution_and_probs(tmp_path):
    r = rec("eikos_decisions:core:x-1", source="eikos_decisions", spdx="CC-BY-4.0", family="eikos.judge",
            field={"id": "decision", "type": "choice", "question": "Which answer is correct here?",
                   "options": [{"value": "a", "description": "A"}, {"value": "b", "description": "B"}]}, target="b")
    r["license"]["attribution"] = "Contains data from Eikos Decisions (CC BY 4.0)"
    r.update(target_probs=[0.1, 0.9], target_probs_keys=["a", "b"], provenance={"dataset": "caiovicentino1/eikos-decisions", "difficulty": "hard"})
    dev = dict(r, id="eikos_decisions:core:x-2", partition="dev")
    out = cot.load_eikos(_write(tmp_path, "eikos", [r, dev]))
    assert len(out) == 1
    c = out[0]
    assert validate(c) == [] and c["provenance"]["licence"] == "CC-BY-4.0"
    assert "Eikos" in c["provenance"]["attribution"] and c["provenance"]["target_probs"] == {"a": 0.1, "b": 0.9}


def test_load_v21_expands_multi_field_text_rows_and_skips_images_and_excluded(tmp_path):
    multi = rec("stackexchange:x:0", source="stackexchange", spdx="CC-BY-SA-4.0", family="multi_question", target=None)
    multi["request"]["fields"] = [{"id": "p", "type": "boolean", "question": "Does it contain a phone number?"},
                                  {"id": "s", "type": "ordinal", "question": "How urgent is this request?",
                                   "levels": [{"value": 0, "description": "low"}, {"value": 1, "description": "high"}]}]
    multi.update(targets={"p": False, "s": None}, target_distributions={"p": {"true": 0.1, "false": 0.9, "unknown": 0}},
                 abstention_causes={"s": "teacher_unknown"}, field_families={"p": "pii_present", "s": "severity_urgency"})
    img = rec("abo:img", source="abo", images=[{"image": "x.jpg"}])
    bank = rec("banking77:train-1", source="banking77")
    test = rec("fever:1", source="fever", partition="test")
    out = cot.load_v21(_write(tmp_path, "v21", [multi, img, bank, test]))
    assert sorted(c["family"] for c in out) == ["pii_present", "severity_urgency"]
    s = next(c for c in out if c["family"] == "severity_urgency")
    assert s["gold"] is None and s["unknown_reason"] == "insufficient_evidence" and validate(s) == []
    p = next(c for c in out if c["family"] == "pii_present")
    assert p["gold"] is False and p["provenance"]["target_probs"]["false"] == 0.9


def test_load_prog_matches_assemble_partition_and_holdout(tmp_path):
    from v1_text.common import stable_partition
    w = {"doc_id": None, "batch": "prog-numeric", "plan": {"domain": "logistics", "templates": ["numeric_reconciliation.x.choice0"]},
         "output": {"document": "Invoice: 3 pallets at $10.", "questions": [
             {"family": "numeric_reconciliation", "type": "choice", "question": "What is the invoice total?",
              "options": [{"text": "$30.00", "description": "$30.00"}, {"text": "$20.00", "description": "$20.00"},
                          {"text": "$13.00", "description": "$13.00"}],
              "levels": None, "intended": "$30.00", "unknown_reason": None, "justification": "3 x 10"}]}}
    docs = []
    for i in range(40):
        d = json.loads(json.dumps(w))
        d["doc_id"] = f"prog-numeric-logistics-{i:04d}"
        docs.append(d)
    hold = json.loads(json.dumps(w))
    hold["doc_id"], hold["plan"]["domain"] = "prog-numeric-telecom-0", "telecom"
    out = cot.load_prog(_write(tmp_path, "prog", docs + [hold]))
    want = {d["doc_id"] for d in docs if stable_partition(d["doc_id"], seed="decision-p2") == "train"}
    assert {c["provenance"]["group_id"] for c in out} == want
    c = out[0]
    assert validate(c) == [] and c["gold_kind"] == "constructed" and c["provenance"]["licence"] == "own"
    gold_opt = next(o for o in c["field"]["options"] if o["key"] == c["gold"])
    assert gold_opt["description"] == "$30.00"


def test_allocate_water_fills_by_weight_and_hits_quota():
    alloc = cot.allocate({"hard": 1000, "easy": 1000, "tiny": 10}, 610, {"hard": 2.0, "easy": 1.0})
    assert sum(alloc.values()) == 610 and alloc["tiny"] == 10
    assert alloc["hard"] == pytest.approx(2 * alloc["easy"], abs=2)
    assert cot.allocate({"a": 5}, 100, {}) == {"a": 5}


def test_select_prefers_unknown_up_to_cap_and_is_deterministic():
    rows = []
    for i in range(100):
        c = build_c(rec(f"p2_teacher:d{i}:0", target=None if i < 50 else True, cause="insufficient_evidence" if i < 50 else None))
        rows.append(c)
    got = cot.select(rows, 20, unknown_cap=0.35)
    assert len(got) == 20 and sum(c["gold"] is None for c in got) == 7
    assert [c["id"] for c in got] == [c["id"] for c in cot.select(list(reversed(rows)), 20, unknown_cap=0.35)]
