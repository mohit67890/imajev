"""Finite supported logits and temperatures retain a normalized distribution."""
import json
import math

import pytest

from vision_decision.calibration import TemperatureCalibrator, softmax
from vision_decision.contracts import UNKNOWN
from vision_decision.scoring import result_from_logits


@pytest.mark.parametrize("logits, temperature, expected", [
    ([1000.0, 1001.0, -1000.0], 1e-308, [0.0, 1.0, 0.0]),
    ([-1001.0, -1000.0, -1002.0], 1e-308, [0.0, 1.0, 0.0]),
    ([1e308, 1e308, 1e308], 0.5, [1 / 3, 1 / 3, 1 / 3]),
    ([1e308, -1e308, 0.0], 0.5, [1.0, 0.0, 0.0]),
])
def test_loaded_calibration_handles_finite_scale(tmp_path, logits, temperature, expected):
    original = result_from_logits([("red", ""), ("blue", ""), (UNKNOWN, "")], logits)
    artifact = tmp_path / "calibration.json"
    calibrator = TemperatureCalibrator("finite-scale", {"choice:2": temperature}, {"choice:2": 1})
    artifact.write_text(json.dumps(calibrator.to_dict()))
    loaded = TemperatureCalibrator.load(artifact)

    result = loaded.calibrate_result(original, "choice", 2)

    assert list(result.scores.values()) == pytest.approx(expected)
    assert result.status == original.status
    assert result.value == original.value
    assert result.raw_logits == original.raw_logits
    assert result.calibration_version == "finite-scale"
    assert result.score_semantics == "calibrated_normalized_scores"


def test_calibrated_distribution_is_invariant_to_common_logit_shift():
    assert softmax([1000.0, 1001.0, 1002.0], 1e-308) == softmax([0.0, 1.0, 2.0], 1e-308)


@pytest.mark.parametrize("logits", [[math.nan, 0.0], [math.inf, 0.0], [-math.inf, 0.0]])
def test_nonfinite_original_logits_remain_invalid(logits):
    with pytest.raises(ValueError, match="finite"):
        softmax(logits, 0.5)


def test_ordinary_temperature_retains_original_probabilities():
    expected = [math.exp(-1) / (1 + math.exp(-1)), 1 / (1 + math.exp(-1))]
    assert softmax([0.0, 2.0], 2.0) == pytest.approx(expected)


def test_large_temperature_does_not_saturate_opposite_extreme_logits():
    expected = [1 / (1 + math.exp(-2)), math.exp(-2) / (1 + math.exp(-2))]
    assert softmax([1e308, -1e308], 1e308) == pytest.approx(expected)
