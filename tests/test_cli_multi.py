import json
import sys
from types import SimpleNamespace

from vision_decision.cli import main
from vision_decision.scoring import candidates, result_from_logits


def test_decide_folds_multi_results_into_the_public_question(tmp_path, monkeypatch, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"questions": {"topics": {
        "type": "multi", "instructions": "Which topics apply?",
        "criteria": {"billing": None, "bug": None},
    }}}))

    class Backend:
        bundle = {"repo": "stub"}
        max_options = 254

        def __init__(self, *args, **kwargs):
            pass

        def score_request(self, images, fields, state, rotations):
            return [result_from_logits(candidates(f), logits) for f, logits in
                    zip(fields, ([3., 0., -3.], [0., 3., -3.]))], {"questions": []}

    monkeypatch.setitem(sys.modules, "vision_decision.backend", SimpleNamespace(MLXDirect=Backend))
    monkeypatch.setattr(sys, "argv", ["vd", "decide", "--request", str(request)])
    assert main() == 0
    answers = json.loads(capsys.readouterr().out)["answers"]
    assert list(answers) == ["topics"]
    assert answers["topics"]["type"] == "multi"
    assert answers["topics"]["labels"] == ["billing"]
