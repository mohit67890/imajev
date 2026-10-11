import json
import sys

import pytest
from fastapi.testclient import TestClient

from playground.server import create_app
from vision_decision.cli import main
from vision_decision.jev_api import flatten_description, to_request


@pytest.mark.parametrize("criteria", [[], ["yes"], "yes", 1, True, False])
def test_noul_rejects_non_object_criteria(criteria):
    payload = {"questions": {"q": {"type": "noul", "instructions": "Does it apply?", "criteria": criteria}}}
    with pytest.raises(ValueError, match="criteria object"):
        to_request(payload)


def test_malformed_noul_criteria_is_an_invalid_request_before_scoring():
    class Backend:
        model = "stub"

        def score(self, *args, **kwargs):
            raise AssertionError("Invalid requests must not reach the backend")

    response = TestClient(create_app(Backend(), examples=[]), raise_server_exceptions=False).post(
        "/v1/systemone", json={"questions": {"q": {
            "type": "noul", "instructions": "Does it apply?", "criteria": ["yes"],
        }}})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


@pytest.mark.parametrize("criteria", [None, {}, {"true": "It applies", "false": "It does not apply"}])
def test_noul_accepts_optional_criteria_objects(criteria):
    request = to_request({"questions": {"q": {
        "type": "noul", "instructions": "Does it apply?", "criteria": criteria,
    }}})
    assert request.fields[0].type == "boolean"


@pytest.mark.parametrize("description", [True, False, 1, 1.5])
def test_description_rejects_unsupported_scalar_types(description):
    with pytest.raises(ValueError, match="description"):
        flatten_description(description)


def test_invalid_description_returns_a_structured_cli_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"questions": {"q": {
        "type": "choice", "instructions": "Which?", "criteria": {"a": 1, "b": None},
    }}}))
    monkeypatch.setattr(sys, "argv", ["vd", "decide", "--request", str(path)])
    assert main() == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "ValueError"
    assert "description" in error["message"]
