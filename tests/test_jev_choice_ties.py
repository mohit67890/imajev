import pytest

from imajev_bench.runner import decode_jev
from vision_decision.calibration import TemperatureCalibrator
from vision_decision.jev_api import to_request, to_response
from vision_decision.scoring import candidates, result_from_logits


@pytest.mark.parametrize("calibrated", [False, True])
def test_choice_response_preserves_the_engine_selection_for_tied_options(calibrated):
    request = to_request({"questions": {"colour": {
        "type": "choice", "instructions": "Which colour?", "criteria": {"red": None, "blue": None},
    }}})
    result = result_from_logits(candidates(request.fields[0]), [1., 1., 0.], token_ids=[90, 40, 100])
    if calibrated:
        result = TemperatureCalibrator("test", {"choice:2": 2.0}, {"choice:2": 1}).calibrate_result(result, "choice", 2)
    assert result.value == "blue"
    answer = to_response(request, [result])["answers"]["colour"]
    assert answer["choice"] == result.value
    assert answer["probabilities"]["red"] == answer["probabilities"]["blue"]
    decoded = decode_jev(to_response(request, [result]), {"request": request.model_dump()})
    assert decoded["status"] == "answered"
    assert decoded["value"] == result.value


def test_abstained_choice_still_reports_the_top_known_option():
    request = to_request({"questions": {"colour": {
        "type": "choice", "instructions": "Which colour?", "criteria": {"red": None, "blue": None},
    }}})
    result = result_from_logits(candidates(request.fields[0]), [0., 1., 2.])
    answer = to_response(request, [result])["answers"]["colour"]
    assert answer["abstained"] is True
    assert answer["choice"] == "blue"
    decoded = decode_jev(to_response(request, [result]), {"request": request.model_dump()})
    assert decoded["status"] == "abstained" and decoded["value"] is None


def test_declared_choice_with_strictly_lower_probability_is_rejected():
    payload = {"request": {"fields": [{"id": "colour", "type": "choice",
                                      "options": [{"value": "red"}, {"value": "blue"}]}]}}
    response = {"answers": {"colour": {"type": "choice", "choice": "blue", "abstained": False,
                "unknown_probability": 0., "probabilities": {"red": .5 + 1e-12, "blue": .5 - 1e-12}}}}
    with pytest.raises(ValueError, match="conflicts"):
        decode_jev(response, payload)
