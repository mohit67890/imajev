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
from vision_decision.jev_api import MAX_OPTIONS, to_request_with_plan, to_response  # noqa: E402
from vision_decision.thinking import ThinkingPolicy  # noqa: E402
from vision_decision.scoring import (check_prompt_layout, check_readout_codes, combine_rotations, compile_question,  # noqa: E402
                                    cyclic_offsets, result_from_logits, rotate)

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

    def __init__(self, bundle=BUNDLE, adapter=None, rotations=1, max_input_tokens=4096, readout_codes=None, prompt_layout=None):
        from vision_decision.backend import MLXDirect
        self.engine = MLXDirect(str(bundle), adapter=None if adapter is None else str(adapter),
                                max_input_tokens=max_input_tokens, readout_codes=readout_codes, prompt_layout=prompt_layout)
        self.readout_codes, self.prompt_layout = self.engine.codes, self.engine.prompt_layout
        self.max_options = self.engine.max_options
        _warn_layout(self.engine.trained_prompt_layout, self.prompt_layout)
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
    supports_thinking = True

    def __init__(self, bundle=BUNDLE, adapter=None, device=None, rotations=1, max_input_tokens=4096, readout_codes=None,
                 prompt_layout=None, fast=False, graph_lengths=None, merge_lora=False, float32=False,
                 question_microbatch=8, result_cache_size=4096, result_cache_path=None, shared_prefix=True):
        from torch_caches import ResultCache, artifact_namespace
        from torch_prefix_cache import PrefixScorer
        import torch
        from torch_decision import GRAPH_LENGTHS, TorchDecision
        self.torch = torch
        self.rotations = max(1, int(rotations))
        self.fast = bool(fast)
        if device is None:  # CUDA on a pod or Space, Metal on a Mac, CPU otherwise
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.bundle = json.loads(Path(bundle).read_text())
        if not Path(self.bundle["path"]).is_dir():
            raise ValueError("Local model snapshot is missing; run scripts/download_model.py")
        start = perf_counter()
        dtype = torch.bfloat16 if device == "cuda" and not float32 else torch.float32
        # --merge-lora loads in float32 so the merge is formed at full precision, then rounds once to the serving dtype.
        # In bf16 that rounding still moves near-tie decisions (and calibration), so it is opt-in; --fast alone keeps
        # the LoRA unmerged and computes exactly the benchmarked function.
        merge = bool(merge_lora) and adapter is not None
        # The float32 merge runs on the CPU (no float32 copy on the GPU); the rounded weights then move to the device.
        self.engine = TorchDecision(self.bundle["path"], "cpu" if merge else device, dtype=torch.float32 if merge else dtype,
                                   max_length=int(max_input_tokens))
        if adapter is not None:
            if merge:
                self.engine.merge_adapter(adapter, dtype)
                self.engine.model.to(device); self.engine.device = device
            else:
                from peft import PeftModel
                self.engine.model = PeftModel.from_pretrained(self.engine.model, str(adapter)).eval()
            self.engine.enable_readout(adapter, trainable=False, codes=readout_codes)
        self.graph_lengths = []
        if self.fast and device == "cuda":
            lengths = [n for n in (graph_lengths or GRAPH_LENGTHS) if n <= int(max_input_tokens)]
            try:
                self.graph_lengths = self.engine.capture_graphs(lengths).lengths
            except Exception:  # the eager path computes the same function
                log.exception("CUDA graph capture failed; serving the eager path")
                self.engine.graphs = None
        if self.engine.readout is None and readout_codes is not None:  # base model / legacy adapter: LM-head rows
            self.engine.codes = check_readout_codes(readout_codes)
        trained = self.engine.prompt_layout  # the adapter's recorded layout (standard when absent)
        self.prompt_layout = trained if prompt_layout is None else check_prompt_layout(prompt_layout)
        _warn_layout(trained, self.prompt_layout)
        self.readout_codes, self.max_options = self.engine.codes, self.engine.max_options
        self.adapter = None if adapter is None else str(adapter)
        self.model = MODEL_NAME if adapter else BASE_MODEL_NAME
        self.merge_lora = bool(merge_lora) and adapter is not None
        self.question_microbatch = max(1, int(question_microbatch))
        self.shared_prefix_requested = bool(shared_prefix)
        self._prefix_scorer = PrefixScorer(self.engine, fast=self.fast, microbatch=self.question_microbatch)
        self._prefix_scorer.enabled = self.shared_prefix_requested
        self.result_cache_namespace = artifact_namespace(bundle, Path(adapter) if adapter is not None else None)
        from vision_decision.contracts import Result
        self.result_cache = ResultCache(int(result_cache_size), path=result_cache_path, result_type=Result)
        self.load_seconds = perf_counter() - start

    def score(self, images, request, thinking=None):
        """One forward per question and per presentation order; with rotations > 1 the per-candidate log-probabilities
        are averaged over cyclic option orders exactly as the MLX path does (combine_rotations), which drops the engine's
        vocabulary tie break; with rotations == 1 the single pass keeps it.

        thinking (a ThinkingPolicy, None = off): a question whose single-pass answer the policy calls unsure gets a greedy
        thought of at most max_tokens, and its decision is read again right after the thought (docs/think-if-unsure-plan.md)."""
        if thinking is not None and thinking.active and self.rotations != 1:
            raise PlaygroundError(422, "invalid_request", "thinking needs a server running --rotations 1")
        if thinking is not None and thinking.active:
            # Thinking writes follow-up tokens per question: never serve stored answers.
            return self._score_uncached(images, request, thinking=thinking)
        from torch_caches import cache_context_digest, field_cache_key
        cache = getattr(self, "result_cache", None)
        if cache is None or cache.capacity <= 0:
            return self._score_uncached(images, request, thinking=thinking)
        context = cache_context_digest(namespace=getattr(self, "result_cache_namespace", None), model=self.model,
                                       adapter=self.adapter, rotations=self.rotations, fast=self.fast,
                                       merge_lora=getattr(self, "merge_lora", False),
                                       shared_prefix=getattr(self, "shared_prefix_requested", False),
                                       microbatch=getattr(self, "question_microbatch", 8),
                                       prompt_layout=getattr(self, "prompt_layout", None),
                                       readout_codes=getattr(self, "readout_codes", None),
                                       state=request.state, images=images)
        keys = [field_cache_key(context, field) for field in request.fields]
        results, miss_idx, miss_fields = [None] * len(request.fields), [], []
        for i, (key, field) in enumerate(zip(keys, request.fields)):
            hit = cache.get(key)
            if hit is None:
                miss_idx.append(i); miss_fields.append(field)
            else:
                results[i] = hit
        if miss_fields:
            fresh, usage = self._score_uncached(images, request.model_copy(update={"fields": miss_fields}),
                                              thinking=thinking)
            for i, result in zip(miss_idx, fresh):
                results[i] = result
                cache.put(keys[i], result)
        else:
            usage = {"prefill_ms": 0.0, "questions_ms": 0.0, "input_tokens": 0, "rotations": self.rotations}
        usage = dict(usage)
        if miss_fields and getattr(self, "_prefix_cache_metadata", None):
            usage["prefix_cache"] = dict(self._prefix_cache_metadata)
        usage.update(cache_hits=len(request.fields) - len(miss_fields), cache_misses=len(miss_fields),
                     cache_entries=len(cache), cache_capacity=cache.capacity, cache_persistent=cache.persistent)
        self._prefix_cache_metadata = None
        return results, usage

    def _score_uncached(self, images, request, thinking=None):
        """Model forward path (no result cache); thinking still applies per question."""
        if thinking is not None and thinking.active and self.rotations != 1:
            raise PlaygroundError(422, "invalid_request", "thinking needs a server running --rotations 1")
        results, seconds, tokens, thoughts, compiled = [], 0.0, 0, [], []
        for field in request.fields:
            header, choices, texts = compile_question(field, request.state, getattr(self, "prompt_layout", "standard"))
            labels = self.engine.labels(len(choices), len(images))
            compiled.append((field, header, choices, texts, labels))
        use_batch = (
            (thinking is None or not thinking.active)
            and len(compiled) > 1
            and getattr(getattr(self.engine, "device", None), "type", getattr(self.engine, "device", "")) == "cuda"
        )
        if use_batch:
            try:
                results = self._score_batched(images, compiled)
                tokens = max(tokens, self._last_batch_tokens)
            except Exception:
                log.exception("batched torch scoring failed; retrying per question")
                use_batch = False
                results = []
                self._batch_seconds = 0.0
        if not use_batch:
            for field, header, choices, texts, labels in compiled:
                start = perf_counter()
                passes = []
                for offset in cyclic_offsets(len(choices), self.rotations):
                    prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, rotate(texts, offset)))
                    if getattr(self, "fast", False):
                        with self.torch.inference_mode():
                            _, inputs, token_ids = self.engine.prepare_fast(images, prompt, labels)
                            logits = [float(x) for x in self.engine.candidate_logits_fast(inputs, token_ids).cpu().tolist()]
                    else:
                        with self.torch.no_grad():
                            _, inputs, token_ids = self.engine.prepare(images, prompt, labels)
                            logits = [float(x) for x in self.engine.candidate_logits(inputs, token_ids).cpu().tolist()]
                    tokens = max(tokens, int(inputs["input_ids"].shape[-1]))
                    passes.append((offset, logits, token_ids))
                seconds += perf_counter() - start
                if len(passes) == 1:
                    result = result_from_logits(choices, passes[0][1], token_ids=passes[0][2])
                else:
                    result = combine_rotations(choices, [(offset, logits) for offset, logits, _ in passes])
                if thinking is not None and thinking.should_think(result):
                    start = perf_counter()
                    thought_result, note = self._think(images, prompt, choices, labels, thinking)
                    result = thought_result or result
                    note["think_ms"] = round((perf_counter() - start) * 1000, 1)
                    thoughts.append({"question": field.id, **note})
                results.append(result)
        # No shared prefill on this path: the whole cost is reported per question.
        usage = {"prefill_ms": 0.0, "questions_ms": round((seconds + getattr(self, "_batch_seconds", 0.0)) * 1000, 1), "input_tokens": tokens, "rotations": self.rotations}
        if thinking is not None and thinking.active:
            usage["thinking"] = {"mode": thinking.mode, "max_tokens": thinking.max_tokens, "thought": thoughts}
        self._batch_seconds = 0.0
        return results, usage

    def _score_batched(self, images, compiled):
        """Physical batch dispatcher: chunked to <= microbatch, prefix-KV when validated.

        Splits large logical panels into <= question_microbatch chunks (bounded
        VRAM), then per chunk tries the shared-prefix KV path (parity-gated per
        evidence mode, serial fallback) so the vision/evidence prefix runs once.
        """
        microbatch = max(1, int(getattr(self, "question_microbatch", 8) or 8))
        if len(compiled) > microbatch:
            combined, total_seconds, max_tokens = [], 0.0, 0
            for start in range(0, len(compiled), microbatch):
                combined.extend(self._score_chunk(images, compiled[start:start + microbatch]))
                total_seconds += float(getattr(self, "_batch_seconds", 0.0) or 0.0)
                max_tokens = max(max_tokens, int(getattr(self, "_last_batch_tokens", 0) or 0))
            self._batch_seconds, self._last_batch_tokens = total_seconds, max_tokens
            return combined
        return self._score_chunk(images, compiled)

    def _score_chunk(self, images, compiled):
        """One <= microbatch chunk: prefix-KV when requested+enabled, else serial batch."""
        scorer = getattr(self, "_prefix_scorer", None)
        if scorer is not None and getattr(self, "shared_prefix_requested", False):
            try:
                out = scorer.maybe_validate_and_score(images, compiled, self.rotations, self._score_serial_batch)
            except Exception:
                log.exception("prefix-KV scoring failed; falling back to serial batch")
                out = self._score_serial_batch(images, compiled)
            self._prefix_cache_metadata = dict(getattr(scorer, "metadata", None) or {})
            return out
        return self._score_serial_batch(images, compiled)

    def _score_serial_batch(self, images, compiled):
        """One forward per rotation offset across all questions (CUDA only).

        Groups questions by rotation offset, renders each (prompt, labels) pair,
        collates into a single left-padded batch via the engine, and reads
        per-question logits from one candidate_logits_batch call. Falls back to
        serial on any alignment error (collate asserts the decision suffix).
        """
        from time import perf_counter as _pc
        start = _pc()
        max_rots = max(len(cyclic_offsets(len(choices), self.rotations)) for _, _, choices, _, _ in compiled)
        per_q = [[] for _ in compiled]
        max_tokens = 0
        fast = getattr(self, "fast", False)
        for r in range(max_rots):
            examples, owners = [], []
            for qi, (field, header, choices, texts, labels) in enumerate(compiled):
                offsets = cyclic_offsets(len(choices), self.rotations)
                if r >= len(offsets):
                    continue
                offset = offsets[r]
                prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, rotate(texts, offset)))
                rendered, imgs, tids = self.engine.render_example(images, prompt, labels)
                examples.append((rendered, imgs, tids, None))
                owners.append((qi, offset, tids))
            if fast:
                with self.torch.inference_mode():
                    inputs, tids_list, _ = self.engine.collate_fast(examples)
                    batch_logits = self.engine.candidate_logits_batch_fast(inputs, tids_list)
            else:
                with self.torch.no_grad():
                    inputs, tids_list, _ = self.engine.collate(examples)
                    batch_logits = self.engine.candidate_logits_batch(inputs, tids_list)
            max_tokens = max(max_tokens, int(inputs["input_ids"].shape[-1]))
            for (qi, offset, _), logits_t in zip(owners, batch_logits):
                per_q[qi].append((offset, [float(x) for x in logits_t.cpu().tolist()]))
        self._last_batch_tokens = max_tokens
        self._batch_seconds = _pc() - start
        out = []
        for qi, (field, header, choices, texts, labels) in enumerate(compiled):
            passes = per_q[qi]
            if len(passes) == 1:
                out.append(result_from_logits(choices, passes[0][1]))
            else:
                out.append(combine_rotations(choices, [(offset, logits) for offset, logits in passes]))
        return out

    def _think(self, images, prompt, choices, labels, thinking):
        """-> (the Result read after a greedy thought, a usage note). `prompt` is the rotations == 1 single-pass prompt."""
        engine = getattr(self, "thoughts", None)  # a VLLMThoughts client (--think-engine vllm), else transformers generate
        try:
            tokens, closed = engine.think(images, prompt, thinking.max_tokens)[:2] if engine else self.engine.generate_thought(images, prompt, thinking.max_tokens)
        except Exception as exc:  # no thought (e.g. a prompt longer than the thought engine's context): keep the single pass
            log.warning("thought failed, keeping the single-pass answer: %s", exc)
            return None, {"thought_tokens": 0, "closed": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        inputs, token_ids = self.engine.inputs_after_thought(images, prompt, labels, tokens, closed)
        with self.torch.inference_mode():
            scored = self.engine.candidate_logits_fast(inputs, token_ids) if getattr(self, "fast", False) else self.engine.candidate_logits(inputs, token_ids)
            logits = [float(x) for x in scored.cpu().tolist()]
        note = {"thought_tokens": len(tokens), "closed": closed}
        if thinking.return_thought:
            note["text"] = self.engine.processor.tokenizer.decode(tokens)
        return result_from_logits(choices, logits, token_ids=token_ids), note


def _warn_layout(trained, served):
    if trained != served:
        log.warning("serving prompt layout %r but the adapter was trained with %r", served, trained)


def build_backend(kind, adapter=None, no_adapter=False, bundle=BUNDLE, rotations=1, max_input_tokens=4096,
                  readout_codes=None, prompt_layout=None, fast=False, merge_lora=False, float32=False,
                  question_microbatch=8, result_cache_size=4096, result_cache_path=None, shared_prefix=True):
    """`auto` prefers MLX with the converted adapter and falls back to torch + the PEFT adapter.

    readout_codes None / prompt_layout None follow the adapter (its readout rows; its decision_readout.json layout)."""
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
        return MLXBackend(bundle, chosen, rotations=rotations, max_input_tokens=max_input_tokens,
                          readout_codes=readout_codes, prompt_layout=prompt_layout)
    if kind == "torch":
        return TorchBackend(bundle, chosen, rotations=rotations, max_input_tokens=max_input_tokens,
                            readout_codes=readout_codes, prompt_layout=prompt_layout, fast=fast, merge_lora=merge_lora,
                            float32=float32, question_microbatch=question_microbatch,
                            result_cache_size=result_cache_size, result_cache_path=result_cache_path,
                            shared_prefix=shared_prefix)
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


IMAGE_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")


def extract_state_images(payload):
    """Image JevBench-style requests carry the photo as a data:image URI inside the state instead of the `images` field.
    Pull every such URI out of the state (a string, a dict value, a list item or a messages[].content), in reading order,
    replace it with "[image N]" and return the data URLs. The state keeps its structure; nothing else is touched."""
    found = []

    def take(value):
        if not isinstance(value, str) or "data:image/" not in value:
            return value
        def repl(match):
            found.append(match.group(0)); return f"[image {len(found)}]"
        return IMAGE_DATA_URL.sub(repl, value)

    def walk(node):
        if isinstance(node, str): return take(node)
        if isinstance(node, list): return [walk(x) for x in node]
        if isinstance(node, dict): return {k: walk(v) for k, v in node.items()}
        return node

    if "state" in payload and payload["state"] is not None:
        payload["state"] = walk(payload["state"])
    return found


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
        if isinstance(payload, dict):
            blobs += [decode_data_url(value, i) for i, value in enumerate(extract_state_images(payload))]
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
        images = list(images) + extract_state_images(payload)   # data:image URIs embedded in the state count as images too
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


def compile_payload(payload, max_options=MAX_OPTIONS):
    """-> (internal Request, multi plan). max_options is the loaded readout's limit (254, or 255 with 256 codes)."""
    try:
        return to_request_with_plan(payload, request_id="playground", max_options=max_options)
    except ValidationError as exc:
        raise PlaygroundError(422, "invalid_request", "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()) or str(exc))
    except (ValueError, TypeError) as exc:
        raise PlaygroundError(422, "invalid_request", str(exc))


# ------------------------------------------------------------------------------------------- app

def create_app(backend, examples=None, static=STATIC, calibration=None, thinking=None):
    """`backend` needs .name, .model, .adapter, .load_seconds and .score(images, request); a backend with
    .supports_thinking also takes .score(images, request, thinking=ThinkingPolicy). `thinking` = the server's default policy
    (off unless the server was started with --thinking); a request's "thinking" object overrides it."""
    thinking = thinking or ThinkingPolicy()
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
        body = {"model": backend.model, "adapter": backend.adapter, "backend": backend.name,
                "loaded": True, "load_seconds": round(backend.load_seconds, 3)}
        for extra in ("readout_codes", "prompt_layout", "max_options"):
            if hasattr(backend, extra):
                body[extra] = getattr(backend, extra)
        return body

    @app.get("/v1/aimino-capabilities")
    def aimino_capabilities():
        scorer = getattr(backend, "_prefix_scorer", None)
        cache = getattr(backend, "result_cache", None)
        requested = bool(getattr(backend, "shared_prefix_requested", False))
        enabled = bool(scorer.enabled) if scorer is not None else False
        return {
            "logical_max_questions": int(getattr(backend, "logical_max_questions",
                                                getattr(__import__("vision_decision.jev_api", fromlist=["MAX_QUESTIONS"]),
                                                         "MAX_QUESTIONS"))),
            "question_microbatch": int(getattr(backend, "question_microbatch", 8)),
            "result_cache_size": int(getattr(cache, "capacity", 0)) if cache is not None else 0,
            "result_cache_persistent": bool(cache is not None and cache.persistent),
            "shared_prefix_requested": requested,
            "shared_prefix": enabled,
            "shared_prefix_enabled": enabled,
            "shared_prefix_validated": bool(getattr(scorer, "validated", False)) if scorer is not None else False,
            "shared_prefix_validated_text": bool(getattr(scorer, "validated_text", False)) if scorer is not None else False,
            "shared_prefix_validated_visual": bool(getattr(scorer, "validated_visual", False)) if scorer is not None else False,
            "shared_prefix_error": getattr(scorer, "error", None),
            # Compatibility aliases for the first overlay revision.
            "shared_vision_requested": requested,
            "shared_vision": enabled,
            "shared_vision_validated": bool(getattr(scorer, "validated", False)) if scorer is not None else False,
            "fast": bool(getattr(backend, "fast", False)),
            "rotations": int(getattr(backend, "rotations", 1)),
        }


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
        request, plan = compile_payload(payload, getattr(backend, "max_options", MAX_OPTIONS))
        images = [image for image, _ in loaded]   # [] is a text-only request; the model handles it
        try:
            policy = thinking.with_request(payload.get("thinking"))
        except (TypeError, ValueError) as exc:
            raise PlaygroundError(422, "invalid_request", str(exc))
        if policy.active and not getattr(backend, "supports_thinking", False):
            raise PlaygroundError(422, "invalid_request", f"thinking is not available on the {backend.name} backend")

        def run():
            with app.state.lock:  # one GPU: strictly sequential
                return backend.score(images, request, thinking=policy) if policy.active else backend.score(images, request)

        try:
            results, usage = await run_in_threadpool(run)
        except PlaygroundError:
            raise
        except Exception as exc:
            log.exception("scoring failed")
            raise PlaygroundError(500, type(exc).__name__, str(exc))
        if calibration is not None:
            photo_only = bool(images) and not request.state   # images with no record: schema-1.2 files carry their own temperature
            results = [calibration.calibrate_result(result, field.type, len(result.scores) - 1, image=bool(images), photo_only=photo_only)
                       for field, result in zip(request.fields, results)]
        body = to_response(request, results, model=backend.model, plan=plan)
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
    parser.add_argument("--rotations", type=int, default=1, help="candidate orders averaged per question (MLX and torch)")
    parser.add_argument("--calibration", help="held-out temperature calibration artifact")
    parser.add_argument("--max-input-tokens", type=int, default=4096,
                        help="refuse requests longer than this many processed tokens (training used <= 4096)")
    parser.add_argument("--model-name", help="public model name reported by the API and the UI (e.g. imajev-2b)")
    parser.add_argument("--readout-codes", type=int, choices=(255, 256),
                        help="decision readout size; default = the adapter's. 256 extends a 255-code adapter by one "
                             "LM-head-initialised code so choice questions may have 255 options")
    parser.add_argument("--prompt-layout", choices=("auto", "standard", "compact"), default="auto",
                        help="prompt layout; auto = the layout recorded in the adapter (standard when absent)")
    parser.add_argument("--fast", action="store_true",
                        help="torch only: tokenize once and, on CUDA, replay recorded graphs of the language model "
                             "(captured at load, about a minute); same decisions as without it")
    parser.add_argument("--merge-lora", action="store_true",
                        help="torch only: fold the LoRA into the weights at load (a little faster; bf16 rounding can move "
                             "near-tie decisions)")
    parser.add_argument("--thinking", choices=("off", "auto", "always"), default="off",
                        help="default thinking mode (torch only); a request's \"thinking\" object overrides it. auto thinks only "
                             "when the single-pass answer is unsure (docs/think-if-unsure-plan.md)")
    parser.add_argument("--think-max-tokens", type=int, default=256, help="default thought budget in tokens")
    parser.add_argument("--think-threshold", type=float, default=0.7,
                        help="auto: think when the top option's share of the non-unknown mass is below this")
    parser.add_argument("--think-unknown-threshold", type=float, default=None,
                        help="auto: also think when the unknown mass is above this (default: not used)")
    parser.add_argument("--think-engine", choices=("hf", "vllm"), default="hf",
                        help="who writes thoughts: transformers generate in this process (hf), or a vLLM OpenAI server running "
                             "the same merged weights (vllm; see scripts/export_merged.py)")
    parser.add_argument("--think-url", default="http://127.0.0.1:8000", help="--think-engine vllm: the vLLM server")
    parser.add_argument("--think-model", default="imajev-4b", help="--think-engine vllm: its served model name")
    parser.add_argument("--float32", action="store_true",
                        help="torch only: serve in float32 on CUDA too (a slow reference for checking bf16 serving paths)")
    parser.add_argument("--logical-max-questions", type=int, default=32,
                        help="torch only: Jev question ceiling for this server (at most 64)")
    parser.add_argument("--question-microbatch", type=int, default=8,
                        help="torch only: physical questions per batch chunk (bounded VRAM)")
    parser.add_argument("--result-cache-size", type=int, default=4096,
                        help="torch only: exact per-question result cache entries (0 = off)")
    parser.add_argument("--result-cache-path", default=None,
                        help="torch only: optional SQLite path for result reuse across restarts")
    parser.add_argument("--shared-prefix", dest="shared_prefix", action="store_true", default=True,
                        help="torch only: reuse the multimodal transformer prefix/KV across unique questions")
    parser.add_argument("--no-shared-prefix", dest="shared_prefix", action="store_false",
                        help="torch only: disable the shared-prefix KV path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.logical_max_questions < 1 or args.logical_max_questions > 64:
        parser.error("--logical-max-questions must be in 1..64")
    if args.question_microbatch < 1 or args.question_microbatch > args.logical_max_questions:
        parser.error("--question-microbatch must be in 1..logical-max-questions")
    if args.result_cache_size < 0 or args.result_cache_size > 100_000:
        parser.error("--result-cache-size must be in 0..100000")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    from vision_decision import jev_api
    jev_api.MAX_QUESTIONS = args.logical_max_questions
    backend = build_backend(args.backend, args.adapter, args.no_adapter, Path(args.model_bundle), args.rotations,
                            args.max_input_tokens, readout_codes=args.readout_codes,
                            prompt_layout=None if args.prompt_layout == "auto" else args.prompt_layout, fast=args.fast,
                            merge_lora=args.merge_lora, float32=args.float32,
                            question_microbatch=args.question_microbatch, result_cache_size=args.result_cache_size,
                            result_cache_path=args.result_cache_path, shared_prefix=args.shared_prefix)
    backend.logical_max_questions = args.logical_max_questions
    if args.model_name:
        backend.model = args.model_name
    log.info("backend=%s model=%s adapter=%s load_seconds=%.1f readout_codes=%s prompt_layout=%s",
             backend.name, backend.model, backend.adapter, backend.load_seconds, backend.readout_codes, backend.prompt_layout)
    calibration = None
    if args.calibration:
        from vision_decision.calibration import TemperatureCalibrator
        calibration = TemperatureCalibrator.load(args.calibration)
    import uvicorn
    thinking = ThinkingPolicy(mode=args.thinking, max_tokens=args.think_max_tokens, threshold=args.think_threshold,
                              unknown_threshold=args.think_unknown_threshold)
    if thinking.active and not getattr(backend, "supports_thinking", False):
        parser.error(f"--thinking needs the torch backend (this server runs {backend.name})")
    if args.think_engine == "vllm":
        if not getattr(backend, "supports_thinking", False):
            parser.error("--think-engine vllm needs the torch backend")
        from thought_engine import VLLMThoughts
        backend.thoughts = VLLMThoughts(args.think_url, args.think_model, backend.engine.processor.tokenizer)
    uvicorn.run(create_app(backend, calibration=calibration, thinking=thinking), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
