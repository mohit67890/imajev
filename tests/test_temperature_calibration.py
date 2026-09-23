import math

import pytest

from vision_decision.calibration import (
    TemperatureCalibrator, calibration_key, fit_temperature, option_count_bucket, softmax,
)
from vision_decision.contracts import UNKNOWN, Result


def test_bucket_boundaries_and_aliases():
    assert [option_count_bucket(n) for n in (2, 3, 5, 6, 10, 11, 25, 26, 254)] == [
        "2", "3-5", "3-5", "6-10", "6-10", "11-25", "11-25", "26-254", "26-254"]
    assert calibration_key("noul", 2) == calibration_key("boolean", 2) == "boolean:2"
    assert calibration_key("score", 5) == "ordinal:3-5"
    for bad in (1, 255, True):
        with pytest.raises(ValueError):
            option_count_bucket(bad)


def test_fit_temperature_reduces_heldout_nll_for_overconfident_logits():
    # Identical confident predictions with only 75% accuracy should be softened.
    samples = [([5.0, 0.0], 0)] * 3 + [([5.0, 0.0], 1)]
    temperature = fit_temperature(samples)
    assert temperature > 1
    raw = sum(-math.log(softmax(logits)[target]) for logits, target in samples)
    fitted = sum(-math.log(softmax(logits, temperature)[target]) for logits, target in samples)
    assert fitted < raw


def test_fit_rejects_non_calibration_and_false_cardinality():
    base = {"decision_type": "choice", "option_count": 2,
            "logits": [1.0, 0.0, -1.0], "target_index": 0}
    with pytest.raises(ValueError, match="held-out"):
        TemperatureCalibrator.fit([{**base, "partition": "test"}])
    with pytest.raises(ValueError, match=r"len\(logits\)"):
        TemperatureCalibrator.fit([{**base, "partition": "calibration", "option_count": 3}])


def test_unknown_is_calibrated_and_unseen_bucket_is_identity():
    calibrator = TemperatureCalibrator("test-v1", {"choice:2": 2.0}, {"choice:2": 8})
    logits = {"a": 1.0, "b": 0.0, UNKNOWN: 3.0}
    scores, version = calibrator.calibrate_scores(logits, "choice", 2)
    assert version == "test-v1" and max(scores, key=scores.get) == UNKNOWN
    assert scores[UNKNOWN] < softmax(list(logits.values()))[-1]

    unseen = {"a": 1.0, "b": 0.0, "c": -2.0, UNKNOWN: 3.0}
    scores, version = calibrator.calibrate_scores(unseen, "choice", 3)
    assert version is None
    assert list(scores.values()) == pytest.approx(softmax(list(unseen.values())))


def test_calibrate_result_preserves_engine_tie_break_and_unknown():
    calibrator = TemperatureCalibrator("test-v1", {"choice:2": 1.7}, {"choice:2": 4})
    # Engine chose b under its token-ID tie break. Calibration must not redo argmax by dict order.
    result = Result(status="answered", value="b", scores={"a": .45, "b": .45, UNKNOWN: .1},
                    raw_logits={"a": 1.0, "b": 1.0, UNKNOWN: 0.0})
    calibrated = calibrator.calibrate_result(result, "choice", 2)
    assert calibrated.value == "b" and calibrated.status == "answered"
    assert calibrated.calibration_version == "test-v1"
    assert calibrated.score_semantics == "calibrated_normalized_scores"

    abstained = Result(status="abstained", value=None, reason="insufficient_evidence",
                       scores={"a": .2, "b": .2, UNKNOWN: .6},
                       raw_logits={"a": 0.0, "b": 0.0, UNKNOWN: 2.0})
    calibrated_unknown = calibrator.calibrate_result(abstained, "choice", 2)
    assert calibrated_unknown.status == "abstained" and calibrated_unknown.value is None
    assert calibrated_unknown.reason == "insufficient_evidence"


def test_artifact_round_trip_and_validation(tmp_path):
    rows = [{"partition": "calibration", "decision_type": "choice", "option_count": 2,
             "logits": [2.0, 0.0, -1.0], "target_index": target} for target in (0, 0, 1)]
    fitted = TemperatureCalibrator.fit(rows)
    path = tmp_path / "temperature.json"
    path.write_text(__import__("json").dumps(fitted.to_dict()))
    assert TemperatureCalibrator.load(path) == fitted
    with pytest.raises(ValueError, match="schema"):
        TemperatureCalibrator.from_dict({"schema_version": "2.0"})
    with pytest.raises(ValueError, match="Invalid temperature"):
        TemperatureCalibrator("bad", {"choice:2": "hot"}, {"choice:2": 1})


def test_unknown_offset_moves_abstention_boundary_for_text_only():
    from vision_decision.calibration import fit_unknown_offset_and_temperature
    # A model that puts too much mass on unknown: targets are mostly the first option.
    samples = [([1.0, 0.0, 1.2], 0)] * 8 + [([1.0, 0.0, 1.2], 2)] * 2
    offset, temperature = fit_unknown_offset_and_temperature(samples)
    assert offset < 0 and temperature > 0
    rows = [{"partition": "calibration", "decision_type": "choice", "option_count": 2, "logits": l, "target_index": t}
            for l, t in samples]
    fitted = TemperatureCalibrator.fit(rows, version="t", unknown_offsets=True)
    assert fitted.unknown_offsets["choice:2"] == offset
    raw = {"a": 1.0, "b": 0.0, UNKNOWN: 1.2}
    engine = Result(status="abstained", value=None, reason="insufficient_evidence",
                    scores={"a": 0.3, "b": 0.1, UNKNOWN: 0.6}, raw_logits=raw)
    text = fitted.calibrate_result(engine, "choice", 2)
    assert text.status == "answered" and text.value == "a" and text.reason is None
    with_image = fitted.calibrate_result(engine, "choice", 2, image=True)
    assert with_image.status == "abstained" and with_image.value is None
    # round trip keeps the offsets (schema 1.1) and a 1.0 artifact still loads
    again = TemperatureCalibrator.from_dict(fitted.to_dict())
    assert again.unknown_offsets == fitted.unknown_offsets and fitted.to_dict()["schema_version"] == "1.1"
    legacy = TemperatureCalibrator.from_dict({"schema_version": "1.0", "calibration_version": "x",
                                              "temperatures": {"choice:2": 1.0}, "counts": {"choice:2": 1}})
    assert legacy.unknown_offset("choice", 2) == 0.0


def test_unknown_offset_recovers_typed_values():
    cal = TemperatureCalibrator("v", {"boolean:2": 1.0, "ordinal:3-5": 1.0}, {"boolean:2": 1, "ordinal:3-5": 1},
                                {"boolean:2": -5.0, "ordinal:3-5": -5.0})
    engine = Result(status="abstained", value=None, reason="insufficient_evidence",
                    scores={"true": 0.2, "false": 0.3, UNKNOWN: 0.5}, raw_logits={"true": 0.0, "false": 0.5, UNKNOWN: 1.0})
    out = cal.calibrate_result(engine, "boolean", 2)
    assert out.value is False and out.status == "answered"
    engine = Result(status="abstained", value=None, reason="insufficient_evidence",
                    scores={"1": 0.2, "2": 0.2, "3": 0.3, UNKNOWN: 0.3},
                    raw_logits={"1": 0.0, "2": 0.0, "3": 0.5, UNKNOWN: 0.6})
    out = cal.calibrate_result(engine, "ordinal", 3)
    assert out.value == 3 and isinstance(out.value, int)
