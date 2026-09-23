"""Playground HTTP API tests: request parsing, validation errors and response shape.

The model backend is mocked throughout — nothing here loads MLX or torch.
"""
import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fastapi.testclient import TestClient  # noqa: E402

from playground import server  # noqa: E402
from playground.examples import EXAMPLES, load_examples  # noqa: E402
from vision_decision.scoring import candidates, result_from_logits  # noqa: E402


def png_bytes(size=(32, 24), color=(10, 60, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(size=(40, 30), color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def data_url(blob, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(blob).decode()


REQUEST = {
    "state": {"listing": {"color": "blue", "product_type": "shirt"}},
    "questions": {
        "color_matches": {"type": "noul", "instructions": "The garment matches `listing.color`."},
        "product_type": {"type": "choice", "instructions": "Which category?",
                         "criteria": {"shirt": None, "shoe": None, "bag": None, "other": None}},
        "usability": {"type": "score", "instructions": "How usable is the photo?",
                      "criteria": ["unusable", "acceptable", "excellent"]},
    },
}


class FakeBackend:
    """Deterministic stand-in: the first candidate wins unless `abstain` is set."""

    name = "fake"
    model = "imajev-v1"
    adapter = "reports/decision-v1/runs/h100x4-full/best"
    load_seconds = 1.25

    def __init__(self, abstain=False, fail=None, delay=0.0):
        self.abstain, self.fail, self.delay = abstain, fail, delay
        self.calls = []
        self.concurrent, self.max_concurrent = 0, 0
        self._guard = threading.Lock()

    def score(self, images, request):
        with self._guard:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError(self.fail)
            self.calls.append((len(images), [f.id for f in request.fields]))
            results = []
            for field in request.fields:
                choices = candidates(field)
                logits = [0.0] * len(choices)
                logits[-1 if self.abstain else 0] = 4.0
                results.append(result_from_logits(choices, logits))
            return results, {"prefill_ms": 12.5, "questions_ms": 30.0, "input_tokens": 512}
        finally:
            with self._guard:
                self.concurrent -= 1


def make_client(**kwargs):
    backend = FakeBackend(**kwargs)
    app = server.create_app(backend, examples=load_examples())
    return TestClient(app, raise_server_exceptions=False), backend


@pytest.fixture
def client():
    made, _ = make_client()
    return made


# ------------------------------------------------------------------------------- models / static

def test_models_endpoint_reports_the_loaded_backend(client):
    body = client.get("/v1/models").json()
    assert body == {"model": "imajev-v1", "adapter": "reports/decision-v1/runs/h100x4-full/best",
                    "backend": "fake", "loaded": True, "load_seconds": 1.25}


def test_root_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


# ----------------------------------------------------------------------------------- multipart

def test_multipart_request_returns_jev_shaped_answers(client):
    response = client.post("/v1/systemone", data={"request": json.dumps(REQUEST)},
                           files=[("image", ("a.png", png_bytes(), "image/png"))])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "imajev-v1"
    assert set(body["answers"]) == {"color_matches", "product_type", "usability"}
    noul, choice, score = (body["answers"][k] for k in ("color_matches", "product_type", "usability"))
    assert noul["type"] == "noul" and 0.0 <= noul["noul"] <= 1.0
    assert choice["type"] == "choice" and choice["choice"] == "shirt"
    assert set(choice["probabilities"]) == {"shirt", "shoe", "bag", "other"}
    assert 0.0 <= choice["confidence"] <= 1.0
    assert score["type"] == "score" and score["legend"] == {"0": "unusable", "1": "acceptable", "2": "excellent"}
    for answer in body["answers"].values():
        assert "unknown_probability" in answer and answer["abstained"] is False
    usage = body["usage"]
    assert set(usage) == {"prefill_ms", "questions_ms", "total_ms", "input_tokens", "images"}
    assert usage["total_ms"] >= 0 and usage["input_tokens"] == 512
    assert usage["images"] == [{"sha256": usage["images"][0]["sha256"], "width": 32, "height": 24}]


def test_multipart_accepts_a_reference_target_pair(client):
    response = client.post("/v1/systemone", data={"request": json.dumps(REQUEST)},
                           files=[("image", ("ref.png", png_bytes(), "image/png")),
                                  ("image", ("target.jpg", jpeg_bytes(), "image/jpeg"))])
    assert response.status_code == 200, response.text
    assert len(response.json()["usage"]["images"]) == 2


def test_multipart_ignores_the_optional_model_field(client):
    payload = {**REQUEST, "model": "jev-1"}
    response = client.post("/v1/systemone", data={"request": json.dumps(payload)},
                           files=[("image", ("a.png", png_bytes(), "image/png"))])
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "imajev-v1"


# ---------------------------------------------------------------------------------------- json

def test_json_request_with_data_urls(client):
    response = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes())]})
    assert response.status_code == 200, response.text
    assert len(response.json()["answers"]) == 3


def test_json_and_multipart_agree(client):
    multipart = client.post("/v1/systemone", data={"request": json.dumps(REQUEST)},
                            files=[("image", ("a.png", png_bytes(), "image/png"))]).json()
    as_json = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes())]}).json()
    assert multipart["answers"] == as_json["answers"]
    assert multipart["usage"]["images"] == as_json["usage"]["images"]


def test_abstention_is_reported():
    client, _ = make_client(abstain=True)
    body = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes())]}).json()
    assert all(answer["abstained"] for answer in body["answers"].values())
    assert body["answers"]["product_type"]["unknown_probability"] > 0.5


# -------------------------------------------------------------------------------------- errors

@pytest.mark.parametrize("payload, needle", [
    ({"questions": {}}, "questions"),
    ({"questions": {"a": {"type": "mystery", "instructions": "?"}}}, "unsupported type"),
    ({"questions": {"a": {"type": "choice", "instructions": "?"}}}, "criteria"),
    ({"questions": {"a": {"type": "score", "instructions": "?", "criteria": ["only one"]}}}, "at least 2"),
    ({"questions": {"a": {"type": "choice", "instructions": "?",
                          "criteria": {str(i): None for i in range(255)}}}}, "at most 254"),
    ({"questions": {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(9)}}, "at most 8"),
    ({"questions": {"a": {"type": "noul"}}}, "instructions"),
])
def test_invalid_requests_are_422(client, payload, needle):
    response = client.post("/v1/systemone", json={**payload, "images": [data_url(png_bytes())]})
    assert response.status_code == 422, response.text
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert needle.lower() in body["detail"].lower()


def test_bad_json_in_multipart_request_field_is_422(client):
    response = client.post("/v1/systemone", data={"request": "{not json"},
                           files=[("image", ("a.png", png_bytes(), "image/png"))])
    assert response.status_code == 422
    assert response.json()["error"] == "bad_json"


def test_bad_json_body_is_422(client):
    response = client.post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"] == "bad_json"


def test_missing_request_field_is_422(client):
    response = client.post("/v1/systemone", files=[("image", ("a.png", png_bytes(), "image/png"))])
    assert response.status_code == 422
    assert "request" in response.json()["detail"]


def test_three_images_is_422(client):
    response = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes())] * 3})
    assert response.status_code == 422
    assert "at most two images" in response.json()["detail"]
    form = client.post("/v1/systemone", data={"request": json.dumps(REQUEST)},
                       files=[("image", (f"{i}.png", png_bytes(), "image/png")) for i in range(3)])
    assert form.status_code == 422


# ------------------------------------------------------------------------------- text-only (no image)

def test_zero_images_is_a_text_only_request(client):
    """No image is not an error: the model answers from state and the questions alone."""
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "imajev-v1" and body["usage"]["images"] == []
    assert set(body["answers"]) == set(REQUEST["questions"])
    assert body["answers"]["product_type"]["choice"] == "shirt"


def test_zero_images_reaches_the_backend_with_no_image():
    made, backend = make_client()
    assert made.post("/v1/systemone", json=REQUEST).status_code == 200
    assert backend.calls == [(0, ["color_matches", "product_type", "usability"])]
    # The UI posts FormData; with no image attached that is a multipart body of one field.
    boundary = "xxBOUNDARYxx"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n"
            f"{json.dumps(REQUEST)}\r\n--{boundary}--\r\n").encode()
    empty = made.post("/v1/systemone", content=body,
                      headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert empty.status_code == 200, empty.text
    assert empty.json()["usage"]["images"] == []


def test_text_only_requests_are_still_validated(client):
    response = client.post("/v1/systemone", json={"questions": {"a": {"type": "mystery", "instructions": "?"}}})
    assert response.status_code == 422
    assert "unsupported type" in response.json()["detail"]


def test_unsupported_content_type_is_422(client):
    response = client.post("/v1/systemone", content=json.dumps(REQUEST).encode(),
                           headers={"content-type": "text/plain"})
    assert response.status_code == 422
    assert response.json()["error"] == "bad_request"


def test_broken_data_url_is_422(client):
    response = client.post("/v1/systemone", json={**REQUEST, "images": ["https://example.com/a.png"]})
    assert response.status_code == 422
    assert response.json()["error"] == "bad_image"
    bad_base64 = client.post("/v1/systemone", json={**REQUEST, "images": ["data:image/png;base64,!!!!"]})
    assert bad_base64.status_code == 422


def test_unsupported_image_format_is_422(client):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="BMP")
    response = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(buffer.getvalue(), "image/bmp")]})
    assert response.status_code == 422
    assert "JPEG" in response.json()["detail"]


def test_oversized_image_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 128)
    response = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes((200, 200)))]})
    assert response.status_code == 413
    assert response.json()["error"] == "image_too_large"


def test_backend_failure_is_500_with_a_message():
    client, _ = make_client(fail="metal kernel exploded")
    response = client.post("/v1/systemone", json={**REQUEST, "images": [data_url(png_bytes())]})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "RuntimeError" and "metal kernel exploded" in body["detail"]
    assert "answers" not in body


# ------------------------------------------------------------------------------ lock / examples

def test_requests_are_served_one_at_a_time():
    client, backend = make_client(delay=0.05)
    body = {**REQUEST, "images": [data_url(png_bytes())]}
    errors = []

    def call():
        try:
            assert client.post("/v1/systemone", json=body).status_code == 200
        except Exception as exc:  # pragma: no cover - surfaced through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert backend.max_concurrent == 1
    assert len(backend.calls) == 4


def test_examples_are_listed_with_existing_images(client):
    items = client.get("/examples").json()
    assert len(items) == len(EXAMPLES), "every shipped example should have its images in the checkout"
    for index, example in enumerate(items):
        assert example["index"] == index
        assert {"state", "questions"} <= set(example["request"])
        assert 0 <= len(example["images"]) <= 2
        for position, image in enumerate(example["images"]):
            assert image["url"].split("?")[0] == f"/examples/{index}/image/{position}"
            assert (ROOT / image["path"]).is_file()
            assert image["role"] == ("reference" if position == 0 else "target")


def test_example_images_are_served(client):
    response = client.get("/examples/0/image/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(response.content)).format == "JPEG"


def test_unknown_example_image_is_404(client):
    assert client.get("/examples/99/image/0").status_code == 404
    assert client.get("/examples/0/image/7").status_code == 404


def test_text_only_examples_are_shipped(client):
    text_only = [e for e in client.get("/examples").json() if not e["images"]]
    assert {e["name"] for e in text_only} == {"Text-only: support ticket triage", "Text-only: resume screening"}


def test_every_example_is_a_valid_request(client):
    for example in client.get("/examples").json():
        response = client.post("/v1/systemone", json={
            **example["request"],
            "images": [data_url(Path(ROOT / image["path"]).read_bytes(), "image/jpeg")
                       for image in example["images"]],
        })
        assert response.status_code == 200, (example["name"], response.text)
        assert set(response.json()["answers"]) == set(example["request"]["questions"])


def test_calibration_is_applied_to_http_results():
    from vision_decision.calibration import TemperatureCalibrator
    calibration = TemperatureCalibrator('http-test', {'boolean:2': 2.}, {'boolean:2': 10})
    backend = FakeBackend()
    app = server.create_app(backend, examples=[], calibration=calibration)
    client = TestClient(app)
    response = client.post('/v1/systemone', json={'state': 'A billing ticket.', 'questions': {
        'q': {'type': 'noul', 'instructions': 'Is this a billing issue?'}}})
    assert response.status_code == 200
    answer = response.json()['answers']['q']
    assert answer['calibration_version'] == 'http-test'
    assert .8 < answer['noul'] < .9
