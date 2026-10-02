import json
import sys
from types import SimpleNamespace

import pytest

from vision_decision.backend import MLXDirect
from vision_decision.cli import main
from vision_decision.scoring import labels_for_count


@pytest.mark.parametrize("codes, options, valid", [
    (255, 254, True), (256, 254, True), (256, 255, True),
    (255, 255, False), (255, 256, False), (256, 256, False),
])
def test_decide_respects_the_loaded_readout_capacity(tmp_path, monkeypatch, capsys, codes, options, valid):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"questions": {"pick": {
        "type": "choice", "instructions": "Which option?",
        "criteria": {f"option_{i}": None for i in range(options)},
    }}}))
    scoring_calls = []

    class Backend(MLXDirect):
        """Use the real prompt, capacity and scoring paths with a synthetic readout."""

        def __init__(self, *args, **kwargs):
            self.codes = codes
            self._codebook = list(zip(labels_for_count(codes, limit=codes), range(codes)))
            self.prompt_layout = "standard"
            self.bundle = {"repo": "stub"}

        def _shared_prefix_logits(self, images, prompts):
            scoring_calls.append(len(prompts))
            return [([1.] + [0.] * (len(labels) - 1), {}) for _, labels in prompts], {}

    monkeypatch.setitem(sys.modules, "vision_decision.backend", SimpleNamespace(MLXDirect=Backend))
    monkeypatch.setattr(sys, "argv", ["vd", "decide", "--request", str(path), "--adapter", "stub"])
    assert main() == (0 if valid else 2)
    captured = capsys.readouterr()
    if valid:
        answer = json.loads(captured.out)["answers"]["pick"]
        assert answer["choice"] == "option_0"
        assert len(answer["probabilities"]) == options
        assert scoring_calls == [1]
    else:
        assert json.loads(captured.err)["error"] in {"ValueError", "ValidationError"}
        assert not scoring_calls
