"""Serial execution with immutable manifests and a gold-free adapter boundary."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import random
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .schema import model_payload


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoring_digest(records):
    """Hash of everything scoring depends on: ids, splits, model inputs and gold. Review bookkeeping in provenance
    can change without invalidating a run; a corrected gold label does invalidate it."""
    return digest([{"id": r["id"], "split": r["split"], "payload": model_payload(r), "gold": r["gold"]} for r in records])


def inputs_digest(records):
    """Hash of the model inputs only (ids, splits, payloads; no gold). A run made on a public release whose test gold
    is withheld binds to this, so the label holders can verify and score it against the full records."""
    return digest([{"id": r["id"], "split": r["split"], "payload": model_payload(r)} for r in records])


def run_hashes(records):
    """Dataset hashes every run manifest carries."""
    return {"records_sha256": digest(records), "scoring_sha256": scoring_digest(records),
            "inputs_sha256": inputs_digest(records), "gold_withheld": any(r.get("gold_withheld") for r in records)}


def domain(field):
    if field["type"] == "boolean":
        return [("true", True), ("false", False)]
    if field["type"] == "ordinal":
        return [(str(x["value"]), x["value"]) for x in field["levels"]]
    return [(x["value"], x["value"]) for x in field["options"]]


def baseline(payload, mode, rng):
    """Controls use the model payload only; there is deliberately no gold argument."""
    choices = domain(payload["request"]["fields"][0]) + [("__unknown__", None)]
    index = len(choices) - 1 if mode == "unknown" else (0 if mode == "first" else rng.randrange(len(choices)))
    key, value = choices[index]
    probabilities = {k: (1 / len(choices) if mode == "random" else float(k == key)) for k, _ in choices}
    return {"status": "abstained" if value is None else "answered", "value": value,
            "probabilities": probabilities, "confidence_source": "synthetic_control"}


def jev_payload(payload, root):
    """Map ordinal values to Jev's zero-based positions; decode restores values."""
    request = payload["request"]
    field = request["fields"][0]
    question = {"instructions": field["question"]}
    if field["type"] == "boolean":
        question.update(type="noul", criteria={"true": field.get("yes_description"), "false": field.get("no_description")})
    elif field["type"] == "choice":
        question.update(type="choice", criteria={x["value"]: x.get("description") for x in field["options"]})
    else:
        question.update(type="score", criteria=[x["description"] for x in field["levels"]])
    images = []
    for asset in payload["images"]:
        path = (root / asset["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Image escapes dataset root")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != asset["sha256"]:
            raise ValueError("Image changed since validation")
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError("Unsupported image format")
        images.append(f"data:{mime};base64," + base64.b64encode(blob).decode())
    return {"state": request["state"], "questions": {field["id"]: question}, "images": images}


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Expected finite probability")
    return float(value)


def decode_jev(response, payload):
    """imajev extension decoder; not a generic confidence interpretation of Jev."""
    field = payload["request"]["fields"][0]
    answer = response["answers"][field["id"]]
    expected_type = {"boolean": "noul", "choice": "choice", "ordinal": "score"}[field["type"]]
    if answer.get("type") != expected_type:
        raise ValueError("Response type does not match requested field")
    # Reconstruct the full distribution, reversing the local server's unknown mapping.
    unknown = _number(answer["unknown_probability"])
    if not 0 <= unknown <= 1 or type(answer.get("abstained")) is not bool:
        raise ValueError("Invalid imajev abstention metadata")
    choices = domain(field)
    if field["type"] == "boolean":
        p_true = _number(answer["noul"]) - unknown / 2
        probabilities = {"true": p_true, "false": 1 - unknown - p_true}
    else:
        raw = answer["probabilities"]
        wire_keys = [str(i) for i in range(len(choices))] if field["type"] == "ordinal" else [k for k, _ in choices]
        if set(raw) != set(wire_keys):
            raise ValueError("Response candidate keys mismatch")
        probabilities = {key: _number(raw[wire]) * (1 - unknown) for (key, _), wire in zip(choices, wire_keys)}
    probabilities["__unknown__"] = unknown
    if any(p < -1e-8 or p > 1 + 1e-8 for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-6:
        raise ValueError("Invalid probability distribution")
    # Clamp floating point reconstruction noise only, never malformed model values.
    probabilities = {k: min(1., max(0., p)) for k, p in probabilities.items()}
    value = None if answer["abstained"] else max(choices, key=lambda x: probabilities[x[0]])[1]
    if field["type"] == "choice":
        declared = answer.get("choice")
        if declared not in dict(choices) or (not answer["abstained"] and declared != value):
            raise ValueError("Declared choice conflicts with candidate probabilities")
    if field["type"] == "ordinal":
        expected_legend = {str(i): item["description"] for i, item in enumerate(field["levels"])}
        if answer.get("legend") != expected_legend:
            raise ValueError("Ordinal legend differs from requested levels")
        expected_score = sum(i * _number(answer["probabilities"][str(i)]) for i in range(len(choices)))
        if abs(_number(answer["score"]) - expected_score) > 1e-6:
            raise ValueError("Ordinal expected score conflicts with probabilities")
    return {"status": "abstained" if value is None else "answered", "value": value,
            "probabilities": probabilities, "confidence_source": "imajev_reconstructed_distribution",
            "model": response.get("model"), "calibration_version": answer.get("calibration_version")}


def run(records, root, output, adapter="first", endpoint=None, timeout=60., seed=0):
    if adapter not in ("first", "random", "unknown", "imajev-http"):
        raise ValueError("Unknown adapter")
    if not records or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Nonempty records and positive finite timeout required")
    if adapter == "imajev-http" and not endpoint:
        raise ValueError("HTTP adapter requires an explicit endpoint")
    output = Path(output)
    # New directory per run: no accidental overwrite, stale resume or mixed model output.
    output.mkdir(parents=True, exist_ok=False)
    config = {"format_version": "0.0.1", "adapter": adapter, "endpoint": endpoint,
              "timeout_seconds": timeout, "seed": seed, **run_hashes(records),
              "record_count": len(records), "reviewed": all(x["annotation_status"] == "reviewed" for x in records),
              "started_at": datetime.now(timezone.utc).isoformat(), "concurrency": 1,
              "cost_basis": "not_measured", "retries": 0,
              "python_version": platform.python_version(),
              "source_sha256": {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
              "decision_rule": "server abstention flag; substantive argmax; ordinal indices restored" if endpoint else "control",
              "warning": "Draft data and synthetic controls are not ranked model results."}
    (output / "manifest.json").write_bytes(canonical_bytes(config) + b"\n")
    rng = random.Random(seed)
    with (output / "predictions.jsonl").open("x") as predictions, (output / "raw.jsonl").open("x") as raw_file:
        for record in records:
            payload = model_payload(record)
            start = time.perf_counter()
            raw = None
            wire_hash = None
            try:
                if adapter == "imajev-http":
                    wire = jev_payload(payload, Path(root))
                    wire_hash = digest(wire)
                    req = urllib.request.Request(endpoint, data=canonical_bytes(wire), headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        raw = response.read(16 * 1024 * 1024 + 1)
                    if len(raw) > 16 * 1024 * 1024:
                        raise ValueError("Response exceeds 16 MiB")
                    result = decode_jev(json.loads(raw), payload)
                else:
                    result = baseline(payload, adapter, rng)
            except Exception as exc:
                result = {"status": "error", "value": None, "error_type": type(exc).__name__, "error": str(exc)}
            elapsed = (time.perf_counter() - start) * 1000
            result.update(id=record["id"], latency_ms=elapsed)
            predictions.write(json.dumps(result, allow_nan=False) + "\n")
            predictions.flush()
            raw_file.write(json.dumps({"id": record["id"], "payload_sha256": digest(payload),
                                       "wire_sha256": wire_hash,
                                       "response": raw.decode("utf-8", errors="replace") if raw else None}) + "\n")
            raw_file.flush()
    completion = {"status": "complete", "completed_count": len(records),
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": file_digest(output / "manifest.json"),
                  "predictions_sha256": file_digest(output / "predictions.jsonl"),
                  "raw_sha256": file_digest(output / "raw.jsonl")}
    (output / "completion.json").write_bytes(canonical_bytes(completion) + b"\n")
    return output / "predictions.jsonl"


def verify_run(records, predictions_path):
    """Require finalized artifacts to bind replay scores to exact inputs/settings."""
    path = Path(predictions_path)
    manifest_path, completion_path = path.parent / "manifest.json", path.parent / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise ValueError("Scoring requires a completed run manifest and completion receipt")
    manifest, completion = json.loads(manifest_path.read_text()), json.loads(completion_path.read_text())
    if completion.get("status") != "complete" or completion.get("completed_count") != len(records):
        raise ValueError("Incomplete run or dataset count mismatch")
    if manifest.get("scoring_sha256") != scoring_digest(records) and manifest.get("records_sha256") != digest(records):
        # A run made on gold-withheld public records binds to the model inputs only.
        if not (manifest.get("gold_withheld") and manifest.get("inputs_sha256") == inputs_digest(records)):
            raise ValueError("Run dataset hash does not match selected records")
    for key, artifact in (("manifest_sha256", manifest_path), ("predictions_sha256", path), ("raw_sha256", path.parent / "raw.jsonl")):
        if not artifact.is_file() or completion.get(key) != file_digest(artifact):
            raise ValueError(f"Run artifact integrity mismatch: {artifact.name}")
    return {"manifest": manifest, "completion": completion,
            "completion_sha256": file_digest(completion_path), "scorer_source_sha256": file_digest(Path(__file__).parent / "scoring.py")}
