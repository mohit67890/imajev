import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_decision.cli import main
from vision_decision.contracts import UNKNOWN
from vision_decision.scoring import candidates, result_from_logits


@pytest.mark.parametrize("command", ["predict", "decide"])
@pytest.mark.parametrize("state, image, expected_abstained, expected_temperature", [
    ({}, False, False, 2.0),
    ({}, True, True, 1.0),
    ({"note": "evidence"}, True, True, 2.0),
])
def test_cli_calibration_uses_the_request_modality(
        tmp_path, monkeypatch, capsys, command, state, image, expected_abstained, expected_temperature):
    request = tmp_path / "request.json"
    field = {"id": "colour", "type": "choice", "question": "Which colour?",
             "options": [{"value": "red"}, {"value": "blue"}]}
    payload = ({"request_id": "test", "state": state, "fields": [field]} if command == "predict" else
               {"state": state, "questions": {"colour": {"type": "choice", "instructions": "Which colour?",
                                                         "criteria": {"red": None, "blue": None}}}})
    request.write_text(json.dumps(payload))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"schema_version": "1.2", "calibration_version": "test",
        "temperatures": {"choice:2": 2.0}, "counts": {"choice:2": 1},
        "unknown_offsets": {"choice:2": -5.0}, "photo_only_temperatures": {"choice:2": 1.0},
        "photo_only_counts": {"choice:2": 1}}))

    class Backend:
        bundle = {"repo": "stub", "revision": "stub"}
        load_seconds = 0.0
        max_options = 254

        def __init__(self, *args, **kwargs):
            pass

        def score_request(self, images, fields, state, rotations):
            return [result_from_logits(candidates(f), [1., 0., 2.]) for f in fields], {"questions": [{}]}

    monkeypatch.setitem(sys.modules, "vision_decision.backend", SimpleNamespace(MLXDirect=Backend))
    argv = ["vd", command, "--request", str(request), "--calibration", str(calibration)]
    if image:
        path = tmp_path / "image.png"
        Image.new("RGB", (2, 2)).save(path)
        argv += ["--image", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    answer = output["results" if command == "predict" else "answers"]["colour"]
    abstained = answer["status"] == "abstained" if command == "predict" else answer["abstained"]
    assert abstained is expected_abstained
    from vision_decision.calibration import softmax
    logits = [1., 0., 2. if image else -3.]
    unknown = answer["scores"][UNKNOWN] if command == "predict" else answer["unknown_probability"]
    assert unknown == pytest.approx(softmax(logits, expected_temperature)[-1])
