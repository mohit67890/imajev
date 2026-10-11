"""Think-if-unsure: the request option, the trigger rule, and the torch backend's think-then-decide flow (stub model)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fastapi.testclient import TestClient  # noqa: E402

from playground import server  # noqa: E402
from vision_decision.jev_api import to_request  # noqa: E402
from vision_decision.scoring import candidates, result_from_logits  # noqa: E402
from vision_decision.thinking import ThinkingPolicy, unsureness  # noqa: E402

REQUEST = {"state": {"note": "a red ball"},
           "questions": {"colour": {"type": "choice", "instructions": "Which colour?", "criteria": {"red": None, "blue": None}}}}


def result(logits):
    field = to_request(REQUEST).fields[0]
    return result_from_logits(candidates(field), logits)  # choices: red, blue, unknown (listed order, then unknown)


# ---------------------------------------------------------------------------------------------- policy

def test_policy_defaults_and_overrides():
    base = ThinkingPolicy()
    assert base.mode == "off" and not base.active
    auto = base.with_request({"mode": "auto", "max_tokens": 128, "threshold": 0.6})
    assert (auto.mode, auto.max_tokens, auto.threshold, auto.return_thought) == ("auto", 128, 0.6, False)
    assert base.with_request(None) is base


@pytest.mark.parametrize("bad", [{"mode": "sometimes"}, {"max_tokens": 0}, {"max_tokens": 5000}, {"max_tokens": 1.5},
                                 {"threshold": 1.2}, {"threshold": True}, {"unknown_threshold": -0.1},
                                 {"return_thought": "yes"}, {"budget": 10}, "auto"])
def test_policy_rejects_bad_values(bad):
    with pytest.raises((ValueError, TypeError)):
        ThinkingPolicy().with_request(bad)


def test_auto_thinks_only_when_unsure():
    auto = ThinkingPolicy(mode="auto", threshold=0.7)
    confident, unsure = result([0.0, 3.0, -9.0]), result([0.0, 0.2, -9.0])
    assert unsureness(confident)[0] > 0.9 and not auto.should_think(confident)
    assert unsureness(unsure)[0] < 0.6 and auto.should_think(unsure)
    assert ThinkingPolicy(mode="always").should_think(confident) and not ThinkingPolicy().should_think(unsure)


def test_unknown_mass_trigger_is_optional():
    unknown_heavy = result([0.0, 4.0, 3.5])  # confident among real options, but a lot of unknown mass
    assert not ThinkingPolicy(mode="auto", threshold=0.7).should_think(unknown_heavy)
    assert ThinkingPolicy(mode="auto", threshold=0.7, unknown_threshold=0.2).should_think(unknown_heavy)


# ------------------------------------------------------------------------------------------------ HTTP

class ThinkingBackend:
    name, model, adapter, load_seconds, supports_thinking = "torch", "imajev-4b", "x", 0.0, True

    def __init__(self):
        self.policies = []

    def score(self, images, request, thinking=None):
        self.policies.append(thinking)
        results = [result_from_logits(candidates(f), [2.0] + [0.0] * (len(candidates(f)) - 1)) for f in request.fields]
        usage = {"prefill_ms": 0.0, "questions_ms": 1.0, "input_tokens": 10}
        if thinking is not None:
            usage["thinking"] = {"mode": thinking.mode, "max_tokens": thinking.max_tokens, "thought": []}
        return results, usage


class PlainBackend(ThinkingBackend):
    supports_thinking = False

    def score(self, images, request):
        return super().score(images, request)


def post(app, body):
    return TestClient(app, raise_server_exceptions=False).post("/v1/systemone", json=body)


def test_request_without_thinking_keeps_the_single_pass():
    backend = ThinkingBackend()
    assert post(server.create_app(backend, examples=[]), REQUEST).status_code == 200
    assert backend.policies == [None]


def test_request_thinking_reaches_the_backend_and_usage():
    backend = ThinkingBackend()
    response = post(server.create_app(backend, examples=[]), {**REQUEST, "thinking": {"mode": "auto", "max_tokens": 64}})
    assert response.status_code == 200, response.text
    assert backend.policies[0].mode == "auto" and backend.policies[0].max_tokens == 64
    assert response.json()["usage"]["thinking"]["max_tokens"] == 64


def test_server_default_applies_and_a_request_can_switch_it_off():
    backend = ThinkingBackend()
    app = server.create_app(backend, examples=[], thinking=ThinkingPolicy(mode="auto", threshold=0.5))
    post(app, REQUEST)
    post(app, {**REQUEST, "thinking": {"mode": "off"}})
    assert backend.policies[0].mode == "auto" and backend.policies[0].threshold == 0.5 and backend.policies[1] is None


@pytest.mark.parametrize("bad", [{"mode": "maybe"}, {"max_tokens": -1}, "yes", {"extra": 1}])
def test_bad_thinking_is_422(bad):
    response = post(server.create_app(ThinkingBackend(), examples=[]), {**REQUEST, "thinking": bad})
    assert response.status_code == 422 and response.json()["error"] == "invalid_request"


def test_backend_without_thinking_refuses_it():
    response = post(server.create_app(PlainBackend(), examples=[]), {**REQUEST, "thinking": {"mode": "always"}})
    assert response.status_code == 422 and "not available" in response.json()["detail"]


# ------------------------------------------------------------------------------ torch backend flow (stub)

class StubEngine:
    """Single pass is a near tie between blue and red; after a thought, red is clear. Records what it was asked."""

    def __init__(self, single, after):
        self.single, self.after, self.thought_calls = single, after, []
        self.processor = type("P", (), {"tokenizer": type("T", (), {"decode": staticmethod(lambda ids: "thought text")})()})()

    def labels(self, count, n_images):
        return ["A", "B", "C"][:count]

    def prepare(self, images, prompt, labels):
        return prompt, {"input_ids": _Ids(5)}, [11, 12, 13]

    def candidate_logits(self, inputs, token_ids):
        return _Logits(self.after if inputs.get("after") else self.single)

    def generate_thought(self, images, prompt, max_tokens):
        self.thought_calls.append((prompt, max_tokens))
        return [7] * min(max_tokens, 3), True

    def inputs_after_thought(self, images, prompt, labels, thought, closed):
        return {"input_ids": _Ids(9), "after": True}, [11, 12, 13]


class _Ids:
    def __init__(self, n):
        self.shape = (1, n)


class _Logits(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def torch_backend(engine, rotations=1):
    import contextlib
    backend = server.TorchBackend.__new__(server.TorchBackend)
    backend.engine, backend.rotations, backend.fast, backend.prompt_layout = engine, rotations, False, "standard"
    backend.torch = type("Torch", (), {"no_grad": staticmethod(contextlib.nullcontext), "inference_mode": staticmethod(contextlib.nullcontext)})
    return backend


def test_torch_backend_thinks_only_on_the_unsure_question_and_decides_after_the_thought():
    engine = StubEngine(single=[0.0, 0.05, -9.0], after=[3.0, 0.0, -9.0])  # blue by a hair, then red clearly
    request = to_request(REQUEST)
    results, usage = torch_backend(engine).score([], request, thinking=ThinkingPolicy(mode="auto", threshold=0.7, max_tokens=64))
    assert results[0].value == "red"
    assert usage["thinking"]["thought"][0]["question"] == "colour" and usage["thinking"]["thought"][0]["thought_tokens"] == 3
    assert engine.thought_calls and engine.thought_calls[0][1] == 64
    assert "text" not in usage["thinking"]["thought"][0]


def test_torch_backend_skips_thinking_when_confident_and_returns_the_thought_on_request():
    confident = StubEngine(single=[0.0, 4.0, -9.0], after=[3.0, 0.0, -9.0])
    results, usage = torch_backend(confident).score([], to_request(REQUEST), thinking=ThinkingPolicy(mode="auto"))
    assert results[0].value == "blue" and usage["thinking"]["thought"] == [] and not confident.thought_calls
    always = StubEngine(single=[0.0, 4.0, -9.0], after=[3.0, 0.0, -9.0])
    _, usage = torch_backend(always).score([], to_request(REQUEST), thinking=ThinkingPolicy(mode="always", return_thought=True))
    assert usage["thinking"]["thought"][0]["text"] == "thought text"


def test_thinking_needs_a_single_order_server():
    with pytest.raises(server.PlaygroundError):
        torch_backend(StubEngine([0, 0, 0], [0, 0, 0]), rotations=4).score([], to_request(REQUEST), thinking=ThinkingPolicy(mode="always"))


def test_a_failed_thought_keeps_the_single_pass_answer():
    class Broken:
        def think(self, images, prompt, max_tokens):
            raise RuntimeError("prompt longer than the thought engine's context")
    engine = StubEngine(single=[0.0, 0.05, -9.0], after=[3.0, 0.0, -9.0])
    backend = torch_backend(engine)
    backend.thoughts = Broken()
    results, usage = backend.score([], to_request(REQUEST), thinking=ThinkingPolicy(mode="always"))
    assert results[0].value == "blue"  # the single-pass answer, not an error
    note = usage["thinking"]["thought"][0]
    assert note["thought_tokens"] == 0 and "RuntimeError" in note["error"]


@pytest.mark.parametrize("stage", ["prepare", "score"])
def test_failed_post_thought_decision_keeps_the_single_pass(stage):
    class BrokenAfterThought(StubEngine):
        def inputs_after_thought(self, *args):
            if stage == "prepare":
                raise ValueError("Processed thought exceeds the token limit")
            return super().inputs_after_thought(*args)

        def candidate_logits(self, inputs, token_ids):
            if stage == "score" and inputs.get("after"):
                raise RuntimeError("post-thought scoring failed")
            return super().candidate_logits(inputs, token_ids)

    backend = torch_backend(BrokenAfterThought(single=[0.0, 0.05, -9.0], after=[3.0, 0.0, -9.0]))
    results, usage = backend.score([], to_request(REQUEST), thinking=ThinkingPolicy(mode="always"))
    assert results[0].value == "blue"
    note = usage["thinking"]["thought"][0]
    assert note["thought_tokens"] == 3 and note["closed"] is True
    assert "error" in note
