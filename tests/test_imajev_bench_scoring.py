import math

from imajev_bench.scoring import UNKNOWN, candidate_keys, score


def record(item_id, group, track="text", family="policy", gold=True, kind="boolean"):
    if kind == "boolean":
        field = {"id": "decision", "question": "Q?", "type": "boolean"}
    elif kind == "ordinal":
        field = {"id": "decision", "question": "Q?", "type": "ordinal",
                 "levels": [{"value": 1, "description": "low"}, {"value": 2, "description": "high"}]}
    else:
        field = {"id": "decision", "question": "Q?", "type": "choice",
                 "options": [{"value": "a"}, {"value": "b"}]}
    return {"id": item_id, "group_id": group, "track": track, "family": family,
            "gold": gold, "request": {"fields": [field]}, "split": "test", "annotation_status": "approved"}


def prediction(value, probabilities=None, latency=10, cost=0.01):
    result = {"status": "answered", "value": value, "latency_ms": latency, "cost_usd": cost}
    if probabilities is not None:
        result["probabilities"] = probabilities
    return result


def test_candidate_keys_are_canonical():
    assert candidate_keys(record("b", "g")) == ["true", "false", UNKNOWN]
    assert candidate_keys(record("o", "g", gold=1, kind="ordinal")) == ["1", "2", UNKNOWN]
    assert candidate_keys(record("c", "g", gold="a", kind="choice")) == ["a", "b", UNKNOWN]


def test_perfect_three_track_submission():
    records = [record("t", "gt", "text"), record("v", "gv", "visual"), record("j", "gj", "joint")]
    probabilities = {"true": 1.0, "false": 0.0, UNKNOWN: 0.0}
    report = score(records, {item["id"]: prediction(True, probabilities) for item in records}, 20, 4)
    assert report["capability"]["overall_score"] == 100
    assert report["groups"]["accuracy"] == 1
    assert report["probability_quality"]["brier"] == 0
    assert report["probability_quality"]["eligible_for_complete_reliability"] is True
    assert report["efficiency"]["cost_usd"]["per_1000_requests"] == 10


def test_majority_baseline_exposed_by_balanced_metrics():
    records = [record(str(i), str(i), gold=(i < 3)) for i in range(4)]
    report = score(records, {item["id"]: prediction(True) for item in records}, 0)
    raw = report["capability"]["all_records"]
    assert raw["accuracy"] == 0.75
    assert raw["class_balanced_accuracy"] == 0.5
    assert math.isclose(raw["macro_f1"], 3 / 7)
    assert report["capability"]["overall_score"] is None


def test_unknown_gold_is_correct_abstention_and_separate_selective_population():
    records = [record("u1", "g1", gold=None), record("u2", "g2", gold=None)]
    predictions = {
        "u1": {"status": "abstained", "value": None,
               "probabilities": {"true": 0, "false": 0, UNKNOWN: 1}},
        "u2": prediction(False, {"true": 0, "false": 1, UNKNOWN: 0}),
    }
    report = score(records, predictions, 10)
    assert report["capability"]["all_records"]["accuracy"] == 0.5
    assert report["selective"]["answerable_total"] == 0
    assert report["selective"]["unknown_gold_total"] == 2
    assert report["selective"]["substantive_answers_on_unknown_gold"] == 1


def test_missing_errors_and_typed_bool_int_mismatch_count_wrong():
    records = [record("missing", "g"), record("error", "g"), record("typed", "g")]
    predictions = {"error": {"status": "error"}, "typed": prediction(1)}
    report = score(records, predictions, 0)
    assert report["capability"]["all_records"]["correct"] == 0
    assert report["submission"]["missing_predictions"] == 1
    assert report["submission"]["error_predictions"] == 1
    assert report["efficiency"]["latency_ms"]["missing"] == 2


def test_invalid_and_missing_probability_vectors_are_not_fabricated():
    records = [record("valid", "g1"), record("invalid", "g2"), record("missing", "g3")]
    predictions = {
        "valid": prediction(True, {"true": .8, "false": .1, UNKNOWN: .1}),
        "invalid": prediction(True, {"true": .8, "false": .2}),
        "missing": prediction(True),
    }
    quality = score(records, predictions, 0)["probability_quality"]
    assert (quality["valid"], quality["invalid"], quality["missing"]) == (1, 1, 1)
    assert quality["coverage"] == 1 / 3
    assert quality["eligible_for_complete_reliability"] is False
    assert math.isclose(quality["brier"], .06)


def test_family_macro_and_group_accuracy_do_not_weight_large_units_more():
    records = [record("a1", "group-a", family="large"), record("a2", "group-a", family="large"),
               record("b", "group-b", family="small")]
    predictions = {"a1": prediction(True), "a2": prediction(True), "b": prediction(False)}
    report = score(records, predictions, 0)
    assert report["capability"]["tracks"]["text"]["raw"]["accuracy"] == 2 / 3
    assert report["capability"]["tracks"]["text"]["family_macro_accuracy"] == 0.5
    assert report["capability"]["tracks"]["text"]["score"] == 50
    assert report["groups"] == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_abstentions_are_not_accepted_in_selective_metrics():
    records = [record("a", "g1"), record("d", "g2")]
    predictions = {
        "a": prediction(True, {"true": .9, "false": .05, UNKNOWN: .05}),
        "d": {"status": "abstained", "value": None,
              "probabilities": {"true": .1, "false": .1, UNKNOWN: .8}},
    }
    selective = score(records, predictions, 0)["selective"]
    assert selective["answered"] == 1
    assert selective["achieved_coverage"] == 0.5
    assert selective["achieved_error"] == 0
    assert selective["aurc"] is None
    assert selective["partial_aurc"] == 0


def test_selective_confidence_uses_selected_answer_and_aggregates_ties():
    records = [record("wrong", "g1", gold=False), record("right", "g2", gold=True),
               record("low", "g3", gold=True)]
    predictions = {
        # Selected true has confidence .2 even though unknown is larger.
        "wrong": prediction(True, {"true": .2, "false": .1, UNKNOWN: .7}),
        "right": prediction(True, {"true": .2, "false": .1, UNKNOWN: .7}),
        "low": prediction(True, {"true": .1, "false": 0, UNKNOWN: .9}),
    }
    selective = score(records, predictions, 0)["selective"]
    assert [point["answered"] for point in selective["curve"]] == [0, 2, 3]
    assert selective["curve"][1]["threshold"] == .2
    assert selective["curve"][1]["risk"] == .5
    assert selective["aurc"] == selective["partial_aurc"]


def test_error_with_probabilities_is_invalid_and_extra_ids_rejected():
    records = [record("x", "g")]
    error = {"status": "error", "probabilities": {"true": 1, "false": 0, UNKNOWN: 0}}
    quality = score(records, {"x": error}, 0)["probability_quality"]
    assert quality["invalid"] == 1
    assert quality["eligible_for_complete_reliability"] is False
    try:
        score(records, {"x": prediction(True), "extra": prediction(True)}, 0)
    except ValueError as exc:
        assert "unknown record IDs" in str(exc)
    else:
        raise AssertionError("extra prediction ID was accepted")


def test_class_metrics_keep_boolean_and_string_labels_distinct():
    bool_false = record("bf", "g1", gold=False)
    bool_true = record("bt", "g2", gold=True)
    string_false = record("sf", "g3", gold="false", kind="choice")
    string_false["request"]["fields"][0]["options"] = [{"value": "false"}, {"value": "other"}]
    report = score(
        [bool_false, bool_true, string_false],
        {"bf": prediction(False), "bt": prediction("false"), "sf": prediction("false")},
        0,
    )["capability"]["all_records"]
    assert report["classes_present"] == ["bool:false", "bool:true", "str:false"]
    assert math.isclose(report["macro_f1"], (1 + 0 + 2 / 3) / 3)


def test_probability_vector_requires_status_value_consistency():
    records = [record("answered-null", "g1"), record("abstained-value", "g2")]
    vector = {"true": .4, "false": .3, UNKNOWN: .3}
    predictions = {
        "answered-null": {"status": "answered", "value": None, "probabilities": vector},
        "abstained-value": {"status": "abstained", "value": False, "probabilities": vector},
    }
    quality = score(records, predictions, 0)["probability_quality"]
    assert quality["invalid"] == 2
    assert quality["valid"] == 0


def test_wrong_type_and_out_of_domain_answers_invalidate_vectors():
    records = [record("bool-string", "g1"), record("garbage", "g2", gold="a", kind="choice")]
    vector_bool = {"true": .7, "false": .2, UNKNOWN: .1}
    vector_choice = {"a": .7, "b": .2, UNKNOWN: .1}
    predictions = {
        "bool-string": {"status": "answered", "value": "true", "probabilities": vector_bool},
        "garbage": {"status": "answered", "value": "garbage", "probabilities": vector_choice},
    }
    report = score(records, predictions, 0)
    assert report["capability"]["all_records"]["correct"] == 0
    assert report["probability_quality"]["invalid"] == 2
    assert report["submission"]["malformed_predictions"] == 2


def test_missing_abstention_value_and_nonmapping_prediction_are_malformed_not_crashes():
    records = [record("missing-value", "g1", gold=None), record("not-object", "g2")]
    predictions = {
        "missing-value": {"status": "abstained", "probabilities": {"true": 0, "false": 0, UNKNOWN: 1}},
        "not-object": "garbage",
    }
    report = score(records, predictions, 0)
    assert report["capability"]["all_records"]["correct"] == 0
    assert report["submission"]["malformed_predictions"] == 2
    assert report["probability_quality"]["invalid"] == 1
    assert report["probability_quality"]["missing"] == 1


def test_ece_is_explicitly_distribution_argmax_ece():
    item = record("x", "g")
    quality = score([item], {"x": prediction(True, {"true": .8, "false": .1, UNKNOWN: .1})}, 0)["probability_quality"]
    assert quality["ece"] == quality["distribution_argmax_ece"]
    assert "argmax" in quality["ece_definition"]
