"""Local playground HTTP API for the imajev decision model — see docs/playground-spec.md ("HTTP API").

Run it directly:
    .venv/bin/python scripts/playground/server.py --backend auto
The model loads once at startup and stays resident; a lock keeps requests strictly sequential
because there is only one GPU.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import mimetypes
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.requests import Request as HttpRequest  # noqa: E402

from vision_decision.images import MAX_BYTES, load_image_bytes  # noqa: E402
from vision_decision.jev_api import to_request, to_response  # noqa: E402
from vision_decision.scoring import compile_question, result_from_logits  # noqa: E402

try:  # package import (PYTHONPATH=scripts) or plain script run
    from .examples import load_examples
except ImportError:  # pragma: no cover - exercised only when run as a script
    from examples import load_examples

# Public model names. The base checkpoint behind the adapter is an implementation detail and is
# deliberately not part of the API or the UI; `/v1/models` still reports the adapter directory.
MODEL_NAME = "imajev-v1.1"
BASE_MODEL_NAME = "imajev-v1.1-base"

MLX_ADAPTER = ROOT / "reports/decision-v1/runs/h100x4-full/best-mlx"
TORCH_ADAPTER = ROOT / "reports/decision-v1/runs/h100x4-full/best"
BUNDLE = ROOT / "artifacts/model.json"
STATIC = Path(__file__).resolve().parent / "static"
DATA_URL = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)

log = logging.getLogger("playground")


class PlaygroundError(Exception):
    """A request the server refuses; `status` is the HTTP code and `detail` the human message."""

    def __init__(self, status, error, detail):
        super().__init__(detail)
        self.status, self.error, self.detail = status, error, detail


def _image_status(message):
    """20 MiB / 20 Mpixel limits are 413; everything else about an image is a 422."""
    return 413 if "exceeds 20 MiB" in message or "20 million decoded pixels" in message else 422


# --------------------------------------------------------------------------------------- backends

class MLXBackend:
    """MLX path: one image prefill shared by every question of the request."""

    name = "mlx"

    def __init__(self, bundle=BUNDLE, adapter=None, rotations=1):
        from vision_decision.backend import MLXDirect
        self.engine = MLXDirect(str(bundle), adapter=None if adapter is None else str(adapter))
        self.adapter = None if adapter is None else str(adapter)
        self.model = MODEL_NAME if adapter else BASE_MODEL_NAME
        self.load_seconds = self.engine.load_seconds
        self.rotations = rotations

    def score(self, images, request):
        # One image is passed on its own, a pair as a list; [] is a text-only request.
        target = images[0] if len(images) == 1 else list(images)
        results, run = self.engine.score_request(target, request.fields, request.state, self.rotations)
        passes = [row for detail in run.get("questions", []) for row in detail["rotations"]]
        return results, {
            "prefill_ms": round(run.get("prefill_seconds", 0.0) * 1000, 1),
            "questions_ms": round(sum(row["forward_seconds"] for row in passes) * 1000, 1),
            "input_tokens": max((row["input_tokens"] for row in passes), default=0),
        }


class TorchBackend:
    """PyTorch fallback: one full forward per question, never a batch (left padding NaNs on MPS)."""

    name = "torch"

    def __init__(self, bundle=BUNDLE, adapter=None, device=None):
        import torch
        from torch_decision import TorchDecision
        self.torch = torch
        if device is None:  # CUDA on a pod or Space, Metal on a Mac, CPU otherwise
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.bundle = json.loads(Path(bundle).read_text())
        if not Path(self.bundle["path"]).is_dir():
            raise ValueError("Local model snapshot is missing; run scripts/download_model.py")
        start = perf_counter()
        self.engine = TorchDecision(self.bundle["path"], device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        if adapter is not None:
            from peft import PeftModel
            self.engine.model = PeftModel.from_pretrained(self.engine.model, str(adapter)).eval()
            self.engine.enable_readout(adapter, trainable=False)
        self.adapter = None if adapter is None else str(adapter)
        self.model = MODEL_NAME if adapter else BASE_MODEL_NAME
        self.load_seconds = perf_counter() - start

    def score(self, images, request):
        results, seconds, tokens = [], 0.0, 0
        for field in request.fields:
            header, choices, texts = compile_question(field, request.state)
            labels = self.engine.labels(len(choices), len(images))
            prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
            start = perf_counter()
            with self.torch.no_grad():
                _, inputs, token_ids = self.engine.prepare(images, prompt, labels)
                logits = [float(x) for x in self.engine.candidate_logits(inputs, token_ids).cpu().tolist()]
            seconds += perf_counter() - start
            tokens = max(tokens, int(inputs["input_ids"].shape[-1]))
            results.append(result_from_logits(choices, logits, token_ids=token_ids))
        # No shared prefill on this path: the whole cost is reported per question.
        return results, {"prefill_ms": 0.0, "questions_ms": round(seconds * 1000, 1), "input_tokens": tokens}


def build_backend(kind, adapter=None, no_adapter=False, bundle=BUNDLE, rotations=1):
    """`auto` prefers MLX with the converted adapter and falls back to torch + the PEFT adapter."""
    if kind == "auto":
        kind = "mlx" if MLX_ADAPTER.is_dir() else "torch"
    default = MLX_ADAPTER if kind == "mlx" else TORCH_ADAPTER
    if no_adapter:
        chosen = None
    elif adapter is not None:
        chosen = Path(adapter)
    else:
        chosen = default if default.is_dir() else None
        if chosen is None:
            log.warning("adapter %s is missing; serving the base model", default)
    if chosen is not None and not Path(chosen).is_dir():
        raise ValueError(f"Adapter directory {chosen} does not exist")
    if kind == "mlx":
        return MLXBackend(bundle, chosen, rotations=rotations)
    if kind == "torch":
        return TorchBackend(bundle, chosen)
    raise ValueError(f"Unknown backend {kind!r}")


# ----------------------------------------------------------------------------------- request body

def decode_data_url(value, position):
    if not isinstance(value, str) or not value:
        raise PlaygroundError(422, "bad_image", f"images[{position}] must be a data URL string")
    match = DATA_URL.match(value.strip())
    if match is None:
        raise PlaygroundError(422, "bad_image", f"images[{position}] is not a data: URL")
    if not match.group("b64"):
        raise PlaygroundError(422, "bad_image", f"images[{position}] must be base64-encoded (data:...;base64,...)")
    try:
        return base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError):
        raise PlaygroundError(422, "bad_image", f"images[{position}] is not valid base64")


async def read_payload(http_request):
    """-> (payload dict without images, [image bytes]) for multipart or JSON encodings."""
    content_type = (http_request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type == "multipart/form-data":
        try:
            form = await http_request.form()
        except Exception as exc:  # malformed multipart body
            raise PlaygroundError(422, "bad_request", f"Malformed multipart body: {exc}")
        raw = form.get("request")
        if raw is None:
            raise PlaygroundError(422, "bad_request", "Missing the 'request' form field")
        if not isinstance(raw, str):
            raw = (await raw.read()).decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlaygroundError(422, "bad_json", f"The 'request' field is not valid JSON: {exc}")
        blobs = []
        for key in ("image", "images", "image[]"):
            for upload in form.getlist(key):
                if isinstance(upload, str):
                    raise PlaygroundError(422, "bad_image", f"Form field {key!r} must be an uploaded file")
                blobs.append(await upload.read())
    elif content_type == "application/json":
        body = await http_request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PlaygroundError(422, "bad_json", f"Request body is not valid JSON: {exc}")
        if not isinstance(payload, dict):
            raise PlaygroundError(422, "bad_request", "Request body must be a JSON object")
        images = payload.get("images", [])
        if not isinstance(images, list):
            raise PlaygroundError(422, "bad_image", "'images' must be a list of data URLs")
        blobs = [decode_data_url(value, i) for i, value in enumerate(images)]
    else:
        raise PlaygroundError(422, "bad_request",
                              "Use multipart/form-data (field 'request' + files 'image') or application/json")
    if not isinstance(payload, dict):
        raise PlaygroundError(422, "bad_request", "The request must be a JSON object")
    payload = {k: v for k, v in payload.items() if k not in ("images", "model")}
    return payload, blobs


def decode_images(blobs):
    """Zero images is a text-only request; one or two go to the image model. Three is a 422."""
    if len(blobs) > 2:
        raise PlaygroundError(422, "bad_image",
                              f"Provide at most two images (first = reference, second = target); got {len(blobs)}")
    loaded = []
    for position, data in enumerate(blobs):
        if len(data) > MAX_BYTES:
            raise PlaygroundError(413, "image_too_large", f"Image {position} exceeds 20 MiB")
        try:
            loaded.append(load_image_bytes(data))
        except ValueError as exc:
            raise PlaygroundError(_image_status(str(exc)), "bad_image", f"Image {position}: {exc}")
        except OSError as exc:
            raise PlaygroundError(422, "bad_image", f"Image {position} could not be decoded: {exc}")
    return loaded


def compile_payload(payload):
    try:
        return to_request(payload, request_id="playground")
    except ValidationError as exc:
        raise PlaygroundError(422, "invalid_request", "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()) or str(exc))
    except (ValueError, TypeError) as exc:
        raise PlaygroundError(422, "invalid_request", str(exc))


# ------------------------------------------------------------------------------------------- app

def create_app(backend, examples=None, static=STATIC, calibration=None):
    """`backend` needs .name, .model, .adapter, .load_seconds and .score(images, request)."""
    app = FastAPI(title="imajev playground", docs_url="/docs", redoc_url=None)
    app.state.backend = backend
    app.state.examples = load_examples() if examples is None else examples
    app.state.lock = threading.Lock()

    @app.exception_handler(PlaygroundError)
    async def playground_error(request, exc):
        return JSONResponse({"error": exc.error, "detail": exc.detail}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": "http_error", "detail": exc.detail}, status_code=exc.status_code)

    @app.get("/v1/models")
    def models():
        return {"model": backend.model, "adapter": backend.adapter, "backend": backend.name,
                "loaded": True, "load_seconds": round(backend.load_seconds, 3)}

    @app.get("/examples")
    def examples_index():
        return app.state.examples

    @app.get("/examples/{example}/image/{index}")
    def example_image(example: int, index: int):
        items = app.state.examples
        if not 0 <= example < len(items) or not 0 <= index < len(items[example]["images"]):
            raise PlaygroundError(404, "not_found", "No such example image")
        path = ROOT / items[example]["images"][index]["path"]
        if not path.is_file():
            raise PlaygroundError(404, "not_found", "Example image is missing from this checkout")
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media, headers={"Cache-Control": "no-cache"})

    @app.post("/v1/systemone")
    async def systemone(http_request: HttpRequest):
        started = perf_counter()
        payload, blobs = await read_payload(http_request)
        loaded = decode_images(blobs)
        request = compile_payload(payload)
        images = [image for image, _ in loaded]   # [] is a text-only request; the model handles it

        def run():
            with app.state.lock:  # one GPU: strictly sequential
                return backend.score(images, request)

        try:
            results, usage = await run_in_threadpool(run)
        except PlaygroundError:
            raise
        except Exception as exc:
            log.exception("scoring failed")
            raise PlaygroundError(500, type(exc).__name__, str(exc))
        if calibration is not None:
            results = [calibration.calibrate_result(result, field.type, len(result.scores) - 1, image=bool(images))
                       for field, result in zip(request.fields, results)]
        body = to_response(request, results, model=backend.model)
        total_ms = round((perf_counter() - started) * 1000, 1)
        body["usage"] = {**usage, "total_ms": total_ms,
                         "images": [{k: meta[k] for k in ("sha256", "width", "height")} for _, meta in loaded]}
        abstained = sum(1 for answer in body["answers"].values() if answer["abstained"])
        log.info("%s  images=%d questions=%d total_ms=%.1f abstained=%d",
                 datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 len(images), len(request.fields), total_ms, abstained)
        return body

    static = Path(static)
    if static.is_dir():
        app.mount("/", StaticFiles(directory=str(static), html=True), name="static")
    else:
        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return ("<!doctype html><meta charset=utf-8><title>imajev playground</title>"
                    "<p>Playground UI not built yet — the API is at <code>POST /v1/systemone</code>.</p>")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="playground-server", description=__doc__)
    parser.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    parser.add_argument("--adapter", help="override the adapter directory")
    parser.add_argument("--no-adapter", action="store_true", help="serve the base model")
    parser.add_argument("--model-bundle", default=str(BUNDLE))
    parser.add_argument("--rotations", type=int, default=1, help="candidate orders averaged per question (MLX)")
    parser.add_argument("--calibration", help="held-out temperature calibration artifact")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    backend = build_backend(args.backend, args.adapter, args.no_adapter, Path(args.model_bundle), args.rotations)
    log.info("backend=%s model=%s adapter=%s load_seconds=%.1f",
             backend.name, backend.model, backend.adapter, backend.load_seconds)
    calibration = None
    if args.calibration:
        from vision_decision.calibration import TemperatureCalibrator
        calibration = TemperatureCalibrator.load(args.calibration)
    import uvicorn
    uvicorn.run(create_app(backend, calibration=calibration), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
