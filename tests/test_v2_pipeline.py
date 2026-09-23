"""decision-v2 pseudo-labelling and assembly.

The fast tests drive the labeller with a stub scorer, so the keep rule, the rotated-order averaging
and the record assembly are checked without a model.  The end-to-end smoke (40 synthetic candidates,
the 2B teacher on Apple MPS, then an assembled+audited mini manifest) is opt-in:

    IMAJEV_V2_SMOKE=1 PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_v2_pipeline.py -k smoke

Both smokes need the fixtures from ``scripts/v2/make_smoke_candidates.py``; the labelling smoke
builds them itself, the blend smoke skips if they are missing.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decision_data import expand_fields, render  # noqa: E402
from vision_decision.calibration import TemperatureCalibrator  # noqa: E402
from v2.assemble_v2 import assemble, flag_v2, regularization, statistics  # noqa: E402
from v2.pseudo_label import (average_orders, blend, blend_records, expand_candidates,  # noqa: E402
                             has_not_listed, keep_decision, label_records, label_vector, option_keys,
                             presented, read_candidates, realign, should_blend, values_by_key)

SMOKE = ROOT / "data/decision-v2-smoke"


# ------------------------------------------------------------------ fixtures

def choice_field(values=("a", "b"), fid="answer"):
    return {"id": fid, "type": "choice", "question": "Which one?", "options": [{"value": v} for v in values]}


def boolean_field(fid="answer"):
    return {"id": fid, "type": "boolean", "question": "Is it blue?"}


def ordinal_field(fid="rating"):
    return {"id": fid, "type": "ordinal", "question": "Rate it.",
            "levels": [{"value": v, "description": f"level {v}"} for v in (1, 2, 3)]}


def record(fields, rid="src:1"):
    return {"id": rid, "source": "src", "source_group": "g1", "partition": "train", "family": "fam",
            "source_split": "crawl", "license": {}, "images": [],
            "request": {"schema_version": "1.0", "request_id": rid.replace(":", "-"), "state": {},
                        "fields": list(fields)},
            "target": None, "abstention_cause": None, "pseudo_label": "pending"}


def choices_of(item):
    return render(item)[1]


def to_presented(canonical, offset, scope="options"):
    """Served-order values -> the order the model actually sees (the inverse of pseudo_label.realign)."""
    n = len(canonical)
    span = n if scope == "candidates" else n - 1
    out = [canonical[(position + offset) % span] for position in range(span)]
    return out + ([] if span == n else [canonical[span]])


def confident(index, sharpness=8.0, scope="options"):
    """A stub scorer putting all its mass on the served candidate ``index``."""
    def score(item, choices, texts, offset):
        canonical = [sharpness if i == index else 0.0 for i in range(len(choices))]
        return to_presented(canonical, offset, scope)
    return score


# ------------------------------------------------------------- keys / values

def test_option_keys_and_values_for_every_decision_type():
    text = record([choice_field(("red", "green"))])
    assert option_keys(choices_of(text)) == ["red", "green", "unknown"]
    assert values_by_key(choices_of(text)) == {"red": "red", "green": "green", "unknown": None}

    flag = record([boolean_field()])
    assert option_keys(choices_of(flag)) == ["true", "false", "unknown"]
    values = values_by_key(choices_of(flag))
    assert values["true"] is True and values["false"] is False and values["unknown"] is None

    rating = record([ordinal_field()])
    assert option_keys(choices_of(rating)) == ["1", "2", "3", "unknown"]
    assert values_by_key(choices_of(rating))["2"] == 2 and not isinstance(values_by_key(choices_of(rating))["2"], bool)


def test_unknown_key_falls_back_to_the_reserved_identifier_when_an_option_shadows_it():
    shadow = record([choice_field(("unknown", "other"))])
    assert option_keys(choices_of(shadow)) == ["unknown", "other", "__unknown__"]


# --------------------------------------------------------- order realignment

def test_default_rotation_moves_the_options_and_leaves_unknown_last():
    item = record([choice_field(("a", "b", "c"))])
    choices = choices_of(item)
    rotated, texts = presented(choices, ["A", "B", "C", "U"], 1)
    assert [c[0] for c in rotated] == ["b", "c", "a", "__unknown__"] and texts == ["B", "C", "A", "U"]
    served, _ = presented(choices, ["A", "B", "C", "U"], 0)
    assert [c[0] for c in served] == ["a", "b", "c", "__unknown__"]


def test_candidate_scope_rotation_also_moves_unknown():
    choices = choices_of(record([choice_field(("a", "b"))]))
    rotated, texts = presented(choices, ["A", "B", "U"], 1, "candidates")
    assert [c[0] for c in rotated] == ["b", "__unknown__", "a"] and texts == ["B", "U", "A"]


@pytest.mark.parametrize("scope", ["options", "candidates"])
def test_realign_is_the_inverse_of_the_presentation_order(scope):
    canonical = [0.1, 0.2, 0.3, 0.4]
    for offset in (0, 1, 2):
        assert realign(to_presented(canonical, offset, scope), offset, scope) == canonical


@pytest.mark.parametrize("scope", ["options", "candidates"])
def test_rotated_order_is_realigned_onto_the_served_option_keys(scope):
    item = record([choice_field(("a", "b"))])
    choices = choices_of(item)
    # The same belief expressed in each presentation order must average back to that belief.
    logits = {offset: to_presented([3.0, 0.0, 0.0], offset, scope) for offset in (0, 1)}
    distribution, argmaxes, agreement = average_orders(choices, logits, None, scope)
    assert argmaxes == {"served": "a", "rotated": "a"}
    assert math.isclose(agreement, 1.0, abs_tol=1e-9)
    assert math.isclose(distribution["a"], math.exp(3) / (math.exp(3) + 2), rel_tol=1e-9)
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12)


def test_position_bias_shows_up_as_disagreement_not_as_a_realignment_bug():
    item = record([choice_field(("a", "b"))])
    choices = choices_of(item)
    # Both orders always prefer whatever is printed first: a in the served order, b in the rotated one.
    distribution, argmaxes, agreement = average_orders(choices, {0: [3.0, 0.0, 0.0], 1: [3.0, 0.0, 0.0]}, None)
    assert argmaxes == {"served": "a", "rotated": "b"}  # the rotated order prints b first
    assert agreement < 0.5
    assert keep_decision(distribution, argmaxes, "unknown")[:2] == (False, "order_disagreement")


def test_temperature_is_applied_per_bucket():
    item = record([choice_field(("a", "b"))])
    choices = choices_of(item)
    logits = {offset: to_presented([3.0, 0.0, 0.0], offset) for offset in (0, 1)}
    hot, _, _ = average_orders(choices, logits, 3.0)
    cold, _, _ = average_orders(choices, logits, 1.0)
    assert hot["a"] < cold["a"]


# ------------------------------------------------------------------ keep rule

@pytest.mark.parametrize("distribution,argmaxes,expected", [
    ({"a": 0.7, "b": 0.2, "unknown": 0.1}, {"served": "a", "rotated": "a"}, (True, None)),
    ({"a": 0.59, "b": 0.3, "unknown": 0.11}, {"served": "a", "rotated": "a"}, (False, "low_confidence")),
    ({"a": 0.6, "b": 0.3, "unknown": 0.1}, {"served": "a", "rotated": "a"}, (True, None)),
    ({"a": 0.9, "b": 0.05, "unknown": 0.05}, {"served": "a", "rotated": "b"}, (False, "order_disagreement")),
    ({"a": 0.25, "b": 0.2, "unknown": 0.55}, {"served": "unknown", "rotated": "unknown"}, (True, None)),
    ({"a": 0.3, "b": 0.25, "unknown": 0.45}, {"served": "unknown", "rotated": "unknown"},
     (False, "low_confidence_unknown")),
    ({"a": 0.2, "b": 0.1, "unknown": 0.7}, {"served": "unknown", "rotated": "a"}, (False, "order_disagreement")),
])
def test_keep_rule(distribution, argmaxes, expected):
    assert keep_decision(distribution, argmaxes, "unknown")[:2] == expected


# -------------------------------------------------------- labelling / assembly

def label(records, score, calibrator=None, teacher="teacher-test"):
    return label_records(records, score, calibrator, teacher)


def test_single_field_label_round_trips_through_render():
    rows = [record([choice_field(("a", "b"))])]
    labelled, dropped, report = label(rows, confident(0))
    assert not dropped and report["kept_decisions"] == 1 and report["unknown_share"] == 0.0
    out = labelled[0]
    assert out["target"] == "a" and out["abstention_cause"] is None and out["pseudo_label"] == "teacher-test"
    assert set(out["target_distribution"]) == {"a", "b", "unknown"}
    assert out["teacher_confidence"] > 0.99 and out["teacher_agreement"] > 0.99
    _, choices, _, weights = render(out)
    assert isinstance(weights, list) and math.isclose(sum(weights), 1.0, abs_tol=1e-9)
    assert weights[[c[0] for c in choices].index("a")] > 0.99


def test_boolean_and_ordinal_targets_keep_their_python_type_and_render():
    labelled, _, _ = label([record([boolean_field()], "src:bool")], confident(0))
    assert labelled[0]["target"] is True
    assert set(labelled[0]["target_distribution"]) == {"true", "false", "unknown"}
    render(labelled[0])

    labelled, _, _ = label([record([ordinal_field()], "src:ord")], confident(1))
    assert labelled[0]["target"] == 2 and not isinstance(labelled[0]["target"], bool)
    assert set(labelled[0]["target_distribution"]) == {"1", "2", "3", "unknown"}
    render(labelled[0])


def test_unknown_argmax_becomes_a_null_target_with_the_teacher_cause():
    rows = [record([choice_field(("a", "b"))])]
    labelled, _, report = label(rows, confident(2))
    assert labelled[0]["target"] is None
    assert labelled[0]["abstention_cause"] == "teacher_unknown"
    assert report["unknown_share"] == 1.0
    _, choices, _, weights = render(labelled[0])
    assert weights[-1] > 0.99  # unknown is the last candidate


def test_render_rejects_a_distribution_key_the_candidate_list_does_not_offer():
    labelled, _, _ = label([record([choice_field(("a", "b"))])], confident(0))
    labelled[0]["target_distribution"]["bogus"] = 0.0
    with pytest.raises(ValueError, match="unknown keys"):
        render(labelled[0])


def test_multi_field_partial_keep_drops_only_the_failing_field():
    rows = [record([choice_field(("a", "b"), "first"), choice_field(("c", "d"), "second")], "src:multi")]

    def score(item, choices, texts, offset):
        n = len(choices)
        if item["request"]["fields"][0]["id"] == "first":
            return to_presented([8.0] + [0.0] * (n - 1), offset)
        return [8.0 if position == 0 else 0.0 for position in range(n)]  # pure position bias

    labelled, dropped, report = label(rows, score)
    assert len(labelled) == 1 and len(dropped) == 1
    assert dropped[0]["field_id"] == "second" and dropped[0]["reason"] == "order_disagreement"
    assert dropped[0]["record_id"] == "src:multi" and dropped[0]["id"] == "src:multi:second"
    kept = labelled[0]
    assert [f["id"] for f in kept["request"]["fields"]] == ["first"]
    assert kept["target"] == "a" and "targets" not in kept and "target_distributions" not in kept
    assert report["by_family"]["fam"] == {"kept": 1, "dropped": 1}
    render(kept)


def test_multi_field_full_keep_writes_the_expand_fields_target_maps():
    rows = [record([choice_field(("a", "b"), "first"), boolean_field("second"),
                    ordinal_field("third")], "src:multi3")]
    labelled, dropped, _ = label(rows, confident(0))
    assert not dropped
    kept = labelled[0]
    assert kept["target"] is None and kept["targets"] == {"first": "a", "second": True, "third": 1}
    assert set(kept["target_distributions"]) == {"first", "second", "third"}
    assert kept["abstention_causes"] == {"first": None, "second": None, "third": None}
    items = expand_fields(kept)
    assert [i["id"] for i in items] == ["src:multi3:first", "src:multi3:second", "src:multi3:third"]
    for item in items:
        assert isinstance(render(item)[3], list)


def test_record_is_dropped_when_no_field_passes():
    rows = [record([choice_field(("a", "b"), "first"), choice_field(("c", "d"), "second")], "src:none")]

    def biased(item, choices, texts, offset):
        return [8.0 if position == 0 else 0.0 for position in range(len(choices))]

    labelled, dropped, report = label(rows, biased)
    assert labelled == [] and len(dropped) == 2
    assert report["dropped_records"] == 1 and report["kept_records"] == 0


def test_a_malformed_candidate_is_dropped_loudly_instead_of_killing_the_run():
    broken = record([{"id": "answer", "type": "choice", "question": "?", "options": [{"value": "a"}]}], "src:bad")
    rows = [broken, record([choice_field()], "src:good")]
    labelled, dropped, report = label(rows, confident(0))
    assert [r["id"] for r in labelled] == ["src:good"]
    assert dropped[0]["reason"] == "invalid_candidate" and "ValidationError" in dropped[0]["error"]
    assert report["dropped_by_reason"]["invalid_candidate"] == 1


def test_shards_partition_the_input_exactly(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"id": f"r{i}"}) + "\n" for i in range(37)))
    for shards in (1, 4, 5):
        seen = [r["id"] for shard in range(shards) for r in read_candidates([path], shard, shards)]
        assert sorted(seen, key=lambda x: int(x[1:])) == [f"r{i}" for i in range(37)]
    # --limit strides first, then shards, exactly as the evaluator does
    limited = [r["id"] for shard in range(2) for r in read_candidates([path], shard, 2, limit=10)]
    assert len(limited) == 10 and len(set(limited)) == 10


def test_shard_paths_do_not_collide():
    from v2.pseudo_label import shard_path
    assert str(shard_path("out/labelled.jsonl", 0, 1)) == "out/labelled.jsonl"
    assert str(shard_path("out/labelled.jsonl", 2, 4)) == "out/labelled.shard2of4.jsonl"


def test_calibrator_temperature_is_looked_up_by_type_and_option_count():
    calibrator = TemperatureCalibrator.from_dict({"schema_version": "1.0", "calibration_version": "t",
                                                  "temperatures": {"choice:2": 50.0}, "counts": {"choice:2": 10}})
    rows = [record([choice_field(("a", "b"))])]
    labelled, dropped, _ = label(rows, confident(0, sharpness=4.0), calibrator)
    assert labelled == [] and dropped[0]["reason"] == "low_confidence"  # flattened past the 0.6 gate
    labelled, dropped, _ = label(rows, confident(0, sharpness=4.0), None)
    assert labelled and labelled[0]["teacher_fields"]["answer"]["temperature"] is None


def test_expand_candidates_does_not_need_a_target_map():
    rows = record([choice_field(("a", "b"), "first"), boolean_field("second")], "src:pair")
    items = expand_candidates(rows)
    assert [fid for fid, _ in items] == ["first", "second"]
    assert [item["id"] for _, item in items] == ["src:pair:first", "src:pair:second"]
    assert all(item["target"] is None and len(item["request"]["fields"]) == 1 for _, item in items)
    assert expand_candidates(record([choice_field()]))[0][1]["id"] == "src:1"


# ------------------------------------------------------------------ assembly

def test_assembly_flags_test_partition_v2_rows_only():
    v1 = [dict(record([choice_field()], "v1:1"), partition="train")]
    v2 = [dict(record([choice_field()], "v2:train"), partition="train"),
          dict(record([choice_field()], "v2:test"), partition="test"),
          dict(record([choice_field()], "v2:dev"), partition="dev")]
    rows = assemble(v1, v2)
    assert rows[0] == v1[0] and "pseudo_label_test" not in rows[0]
    by_id = {r["id"]: r for r in rows}
    assert by_id["v2:test"]["heldout_family"] is True and by_id["v2:test"]["pseudo_label_test"] is True
    for rid in ("v2:train", "v2:dev"):
        assert "heldout_family" not in by_id[rid] and "pseudo_label_test" not in by_id[rid]
    assert flag_v2(v2[0]) is not v2[0]  # never mutates its input


def test_assembly_rejects_an_id_collision_with_v1():
    v1 = [record([choice_field()], "shared:1")]
    with pytest.raises(ValueError, match="collides"):
        assemble(v1, [record([choice_field()], "shared:1")])


def test_statistics_count_unknown_decisions_across_single_and_multi_field_rows():
    single = dict(record([choice_field()], "v2:a"), target=None, pseudo_label="teacher-x")
    answered = dict(record([choice_field()], "v2:b"), target="a", pseudo_label="teacher-x")
    multi = dict(record([choice_field(fid="first"), boolean_field("second")], "v2:c"),
                 targets={"first": "a", "second": None}, pseudo_label="teacher-x")
    stats = statistics([], [single, answered, multi])
    assert stats["v2_rows"] == 3 and stats["v2_decisions"] == 4
    assert stats["v2_unknown_decisions"] == 2 and stats["v2_pseudo_labels"] == {"teacher-x": 3}


# --------------------------------------------------- self-distillation blend

def labelled(fields, rid="src:1", **extra):
    row = record(fields, rid)
    row.pop("pseudo_label")
    row.update(extra)
    return row


def blend_of(records, alpha, score=None, **kwargs):
    kwargs.setdefault("partitions", ("train",))
    kwargs.setdefault("rotations", 1)
    return blend_records(records, score or confident(0), None, "2b", alpha, **kwargs)


def test_blend_maths():
    assert blend([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0) == [0.0, 1.0, 0.0]
    assert blend([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0) == [1.0, 0.0, 0.0]
    assert blend([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5) == [0.5, 0.5, 0.0]
    assert blend([0.6, 0.3, 0.1], [0.0, 0.0, 1.0], 0.25) == pytest.approx([0.15, 0.075, 0.775])
    # unnormalised operands still come out as a distribution
    assert sum(blend([2.0, 0.0], [0.0, 2.0], 0.5)) == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_blend_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        blend([1.0, 0.0], [0.0, 1.0], alpha)


def test_blend_rejects_mismatched_operands():
    with pytest.raises(ValueError, match="same candidates"):
        blend([1.0, 0.0], [0.0, 1.0, 0.0], 0.5)


def test_label_vector_from_a_hard_index_and_from_an_existing_soft_distribution():
    choices = choices_of(record([choice_field(("a", "b"))]))
    assert label_vector(choices, 1) == [0.0, 1.0, 0.0]
    assert label_vector(choices, [0.2, 0.5, 0.3]) == [0.2, 0.5, 0.3]
    with pytest.raises(ValueError, match="candidate list"):
        label_vector(choices, [0.5, 0.5])


def test_blend_writes_a_renderable_mixture_and_leaves_the_hard_label_alone():
    rows = [labelled([choice_field(("a", "b"))], target="b", abstention_cause=None,
                     image_role="relevant", group="trained_source")]
    out, notes, report = blend_of(rows, 0.5)
    assert notes == [] and report["blended_records"] == 1 and report["passthrough_records"] == 0
    row = out[0]
    assert row["target"] == "b" and row["image_role"] == "relevant" and row["group"] == "trained_source"
    assert row["regularizer"] == "base-2b" and row["regularizer_alpha"] == 0.5
    # confident(0) puts ~all model mass on "a"; the label is "b".
    assert row["target_distribution"]["a"] == pytest.approx(0.5, abs=1e-3)
    assert row["target_distribution"]["b"] == pytest.approx(0.5, abs=1e-3)
    assert report["base_agreement_rate"] == 0.0
    weights = render(row)[3]
    assert isinstance(weights, list) and math.isclose(sum(weights), 1.0, abs_tol=1e-9)
    assert rows[0].get("target_distribution") is None  # the input record is never mutated


def test_blend_alpha_one_is_the_model_and_alpha_zero_is_the_label():
    rows = [labelled([choice_field(("a", "b"))], target="b")]
    model_only = blend_of(rows, 1.0)[0][0]["target_distribution"]
    assert model_only["a"] > 0.99 and model_only["b"] < 0.01

    def explode(*_):
        raise AssertionError("alpha 0 must not score")

    label_only, _, report = blend_of(rows, 0.0, score=explode)
    assert label_only[0]["target_distribution"] == {"a": 0.0, "b": 1.0, "unknown": 0.0}
    assert report["base_agreement_rate"] is None and report["mean_model_confidence"] is None
    render(label_only[0])


def test_blend_of_an_unknown_target_puts_the_label_mass_on_unknown():
    rows = [labelled([choice_field(("a", "b"))], target=None, abstention_cause="insufficient_evidence")]
    row = blend_of(rows, 0.0)[0][0]
    assert row["target_distribution"] == {"a": 0.0, "b": 0.0, "unknown": 1.0}
    assert row["target"] is None and row["abstention_cause"] == "insufficient_evidence"


def test_blend_rekeys_an_existing_soft_target_onto_the_served_option_keys():
    # v1.1 rows spell booleans "True" and unknown "__unknown__"; render's aliases resolve both.
    rows = [labelled([boolean_field()], target=True,
                     target_distribution={"True": 0.8, "__unknown__": 0.2})]
    row = blend_of(rows, 0.0)[0][0]
    assert row["target_distribution"] == {"true": 0.8, "false": 0.0, "unknown": 0.2}
    assert row["regularizer_fields"]["answer"]["label_was_soft"] is True
    render(row)


def test_blend_handles_ordinal_levels_and_keeps_their_keys_stringified():
    rows = [labelled([ordinal_field()], target=2, target_distribution={"1": 0.1, "2": 0.6, "3": 0.3})]
    row = blend_of(rows, 0.5)[0][0]
    assert set(row["target_distribution"]) == {"1", "2", "3", "unknown"}
    assert row["target"] == 2
    assert row["target_distribution"]["1"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.1, abs=1e-3)
    render(row)


def test_blend_of_a_multi_field_record_keeps_targets_and_writes_the_distribution_map():
    rows = [labelled([choice_field(("a", "b"), "first"), boolean_field("second")], "src:multi",
                     targets={"first": "a", "second": False},
                     target_distributions={"second": {"true": 0.25, "false": 0.75}})]
    out, _, report = blend_of(rows, 0.5)
    row = out[0]
    assert row["targets"] == {"first": "a", "second": False} and row["target"] is None
    assert set(row["target_distributions"]) == {"first", "second"}
    assert row["target_distributions"]["second"]["false"] == pytest.approx(0.5 * 0.75, abs=1e-3)
    assert report["blended_decisions"] == 2
    for item in expand_fields(row):
        assert isinstance(render(item)[3], list)


def test_blend_passes_through_other_partitions_and_not_listed_rows_untouched():
    train = labelled([choice_field()], "src:train", target="a", partition="train")
    test = labelled([choice_field()], "src:test", target="a", partition="test")
    skip = labelled([choice_field()], "src:skip", target="a", abstention_cause="not_listed")
    out, notes, report = blend_of([train, test, skip], 0.5)
    by_id = {r["id"]: r for r in out}
    assert len(out) == 3 and report["blended_records"] == 1 and report["passthrough_records"] == 2
    assert report["passthrough_by_reason"] == {"not_listed": 1, "partition": 1}
    assert {n["reason"] for n in notes} == {"not_listed", "partition"}
    for rid in ("src:test", "src:skip"):
        assert "target_distribution" not in by_id[rid] and "regularizer" not in by_id[rid]
    assert "regularizer" in by_id["src:train"]


def test_blend_not_listed_override_and_all_partitions():
    skip = labelled([choice_field()], "src:skip", target="a", abstention_cause="not_listed",
                    partition="test")
    out, _, report = blend_of([skip], 0.5, partitions=None, blend_not_listed=True)
    assert report["blended_records"] == 1 and "regularizer" in out[0]


def test_not_listed_detection_covers_the_multi_field_cause_map():
    assert has_not_listed({"abstention_cause": "not_listed"})
    assert has_not_listed({"abstention_causes": {"a": None, "b": "not_listed"}})
    assert not has_not_listed({"abstention_cause": None, "abstention_causes": {"a": None}})
    assert should_blend({"partition": "dev"}) == (False, "partition")
    assert should_blend({"partition": "train"}) == (True, None)


def test_blend_passes_through_a_record_it_cannot_expand():
    broken = labelled([choice_field(fid="first"), choice_field(fid="second")], "src:broken", target="a")
    out, notes, report = blend_of([broken], 0.5)
    assert len(out) == 1 and "regularizer" not in out[0]
    assert notes[0]["reason"] == "invalid_record" and report["passthrough_records"] == 1


def test_assembler_reports_the_regularizer_alpha():
    v1 = [dict(record([choice_field()], "v1:1"), partition="train", regularizer="base-9b",
               regularizer_alpha=0.3, target="a")]
    v2 = [dict(record([choice_field()], "v2:1"), partition="train", target="a")]
    stats = statistics(v1, v2)
    assert stats["v1_regularization"] == {"regularizer_alpha": 0.3, "regularizer": "base-9b",
                                          "regularized_rows": 1}
    assert stats["v2_regularization"]["regularizer_alpha"] is None
    assert regularization([])["regularized_rows"] == 0


# --------------------------------------------------------------- MPS smoke

def run(*command, env=None):
    environment = dict(os.environ, PYTHONPATH="src:scripts", HF_HUB_OFFLINE="1", **(env or {}))
    done = subprocess.run([str(ROOT / ".venv/bin/python"), *command], cwd=ROOT, env=environment,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return done.stdout


@pytest.mark.skipif(os.environ.get("IMAJEV_V2_SMOKE") != "1", reason="set IMAJEV_V2_SMOKE=1 (loads the 2B on MPS)")
def test_end_to_end_mps_smoke(tmp_path):
    """40 synthetic candidates -> 2B teacher on MPS -> assembled, audited mini manifest."""
    manifests = [SMOKE / "smoke_text/records.jsonl", SMOKE / "smoke_image/records.jsonl"]
    if not all(p.is_file() for p in manifests):
        run("scripts/v2/make_smoke_candidates.py")
    output = tmp_path / "labelled.jsonl"
    run("scripts/v2/pseudo_label.py", "--manifest", *[str(p) for p in manifests],
        "--model", "artifacts/model.json", "--adapter", "reports/decision-v1.1/runs/h100x4/best",
        "--calibration", "reports/decision-v1.1/calibration-v1.1.json", "--output", str(output),
        "--device", "mps", "--teacher-name", "2b-v1.1-smoke")
    labelled = [json.loads(x) for x in output.open() if x.strip()]
    summary = json.loads(Path(str(output) + ".summary.json").read_text())
    dropped = [json.loads(x) for x in Path(str(output) + ".dropped.jsonl").open() if x.strip()]
    assert summary["candidate_records"] == 40
    assert summary["kept_decisions"] + summary["dropped_decisions"] == summary["candidate_decisions"]
    assert len(dropped) == summary["dropped_decisions"] and labelled

    for row in labelled:
        assert row["pseudo_label"] == "teacher-2b-v1.1-smoke"
        assert 0.0 <= row["teacher_confidence"] <= 1.0 and 0.0 <= row["teacher_agreement"] <= 1.0
        for item in expand_fields(row):
            weights = render(item)[3]
            assert isinstance(weights, list) and math.isclose(sum(weights), 1.0, abs_tol=1e-9)

    manifest = tmp_path / "decision-v2-smoke.jsonl"
    report = tmp_path / "mixture-audit.json"
    run("scripts/v2/assemble_v2.py", "--labelled", str(output), "--output", str(manifest),
        "--report", str(report), "--v1-limit", "50")
    audit = json.loads(report.read_text())
    assert audit["ok"], audit["errors"][:10]
    assert audit["v1_rows"] == 50 and audit["v2_rows"] == len(labelled)
    rows = [json.loads(x) for x in manifest.open() if x.strip()]
    assert len(rows) == 50 + len(labelled)
    for row in rows[50:]:
        assert (row.get("pseudo_label_test") is True) == (row["partition"] == "test")


@pytest.mark.skipif(os.environ.get("IMAJEV_V2_SMOKE") != "1", reason="set IMAJEV_V2_SMOKE=1 (loads the 2B on MPS)")
def test_end_to_end_mps_blend_smoke(tmp_path):
    """10 already-labelled v1.1 training rows -> base 2B self-distillation blend -> audited mixture."""
    fixture = SMOKE / "v1.1-train-10.jsonl"
    labelled = SMOKE / "labelled/smoke-2b.jsonl"
    if not fixture.is_file() or not labelled.is_file():
        pytest.skip("run make_smoke_candidates.py and the labelling smoke first")
    source = [json.loads(x) for x in fixture.open() if x.strip()]
    blended_path = tmp_path / "blended.jsonl"
    run("scripts/v2/pseudo_label.py", "--manifest", str(fixture), "--model", "artifacts/model.json",
        "--adapter", "none", "--blend", "0.5", "--base-name", "2b-base",
        "--output", str(blended_path), "--device", "mps")
    rows = [json.loads(x) for x in blended_path.open() if x.strip()]
    summary = json.loads(Path(str(blended_path) + ".summary.json").read_text())
    assert summary["mode"] == "blend" and summary["readout"] == "base_lm_head" and summary["adapter"] is None
    assert summary["calibration_version"] is None  # calibration is optional in base mode
    assert len(rows) == len(source) == summary["records"]  # nothing is ever dropped
    assert summary["blended_records"] + summary["passthrough_records"] == len(rows)
    assert summary["passthrough_by_reason"].get("not_listed") == 1

    by_id = {r["id"]: r for r in source}
    for row in rows:
        before = by_id[row["id"]]
        assert row.get("target") == before.get("target") and row.get("targets") == before.get("targets")
        assert ("target" in row) == ("target" in before)  # absent stays absent
        for key in ("image_role", "group", "license", "partition", "source_group", "abstention_cause"):
            assert row.get(key) == before.get(key)
        if "regularizer" not in row:
            continue
        assert row["regularizer"] == "base-2b-base" and row["regularizer_alpha"] == 0.5
        for item in expand_fields(row):
            weights = render(item)[3]
            assert isinstance(weights, list) and math.isclose(sum(weights), 1.0, abs_tol=1e-9)

    report = tmp_path / "blend-audit.json"
    run("scripts/v2/assemble_v2.py", "--v1", str(blended_path), "--labelled", str(labelled),
        "--output", str(tmp_path / "decision-v2-blend.jsonl"), "--report", str(report))
    audit = json.loads(report.read_text())
    assert audit["ok"], audit["errors"][:10]
    assert audit["v1_regularization"] == {"regularizer_alpha": 0.5, "regularizer": "base-2b-base",
                                          "regularized_rows": summary["blended_records"]}
