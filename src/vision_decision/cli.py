import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from .contracts import Request
from .images import load_image

def hardware():
    info = {"architecture": platform.machine(), "os": platform.mac_ver()[0], "python": platform.python_version()}
    if sys.platform == "darwin":
        # Whitelist fields; never persist serial numbers or hardware UUIDs.
        raw = subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"], text=True)
        details = json.loads(raw)["SPHardwareDataType"][0]
        info.update({k: details[k] for k in ("chip_type", "physical_memory", "machine_model") if k in details})
    info["packages"] = {}
    for package in ("mlx", "mlx-vlm", "transformers", "Pillow", "pydantic"):
        try: info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    return info

def calibrate_results(results, fields, path):
    if path is None:
        return results
    from .calibration import TemperatureCalibrator
    calibrator = TemperatureCalibrator.load(path)
    return [calibrator.calibrate_result(result, field.type, len(result.scores) - 1)
            for field, result in zip(fields, results)]

def main():
    parser = argparse.ArgumentParser(prog="vd")
    sub = parser.add_subparsers(dest="command", required=True)
    hw = sub.add_parser("inspect-hardware")
    hw.add_argument("--output")
    val = sub.add_parser("validate-schema")
    val.add_argument("--input", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--image", help="optional image; omit for a text-only request")
    pred.add_argument("--request", required=True)
    pred.add_argument("--model-bundle", default="artifacts/model.json")
    pred.add_argument("--rotations", type=int, default=4, help="candidate orders averaged per field")
    pred.add_argument("--output")
    dec = sub.add_parser("decide", help="Jev-style request: state + typed questions (noul / choice / score) with optional images")
    dec.add_argument("--image", action="append", default=[], help="repeat for a reference/target pair (first = reference)")
    dec.add_argument("--request", required=True)
    dec.add_argument("--model-bundle", default="artifacts/model.json")
    dec.add_argument("--adapter")
    dec.add_argument("--rotations", type=int, default=1)
    dec.add_argument("--output")
    for command in (pred, dec):
        command.add_argument("--calibration", help="held-out temperature calibration artifact")
    args = parser.parse_args()
    try:
        if args.command == "inspect-hardware":
            output = hardware()
        elif args.command == "validate-schema":
            request = Request.model_validate_json(Path(args.input).read_text())
            output = {"valid": True, "fields": len(request.fields), "mode": request.execution.mode}
        elif args.command == "decide":
            from .jev_api import to_request, to_response
            request = to_request(json.loads(Path(args.request).read_text()))
            if len(args.image) > 2:
                raise ValueError("Provide at most two images")
            loaded = [load_image(path) for path in args.image]
            from .backend import MLXDirect
            backend = MLXDirect(args.model_bundle, adapter=args.adapter)
            images = [image for image, _ in loaded]
            scored, run = backend.score_request(images[0] if len(images) == 1 else images, request.fields, request.state, args.rotations)
            scored = calibrate_results(scored, request.fields, args.calibration)
            run.pop("questions")
            output = {**to_response(request, scored, model=backend.bundle["repo"] + ("+adapter" if args.adapter else "")),
                      "images": [meta for _, meta in loaded], "usage": run}
        else:
            request = Request.model_validate_json(Path(args.request).read_text())
            image, image_meta = load_image(args.image) if args.image else (None, None)
            from .backend import MLXDirect
            backend = MLXDirect(args.model_bundle)
            scored, run = backend.score_request(image, request.fields, request.state, args.rotations)
            scored = calibrate_results(scored, request.fields, args.calibration)
            details = run.pop("questions")
            results = {field.id: result.model_dump() for field, result in zip(request.fields, scored)}
            timings = {field.id: detail for field, detail in zip(request.fields, details)}
            output = {"schema_version": "1.0", "request_id": request.request_id, "results": results,
                      "model_bundle": {k: backend.bundle[k] for k in ("repo", "revision")}, "image": image_meta,
                      "execution": {"mode": "inspect", "automation_allowed": False,
                                    "image_prefill_count": int(image is not None), "external_fallback_used": False, **run,
                                    "model_load_seconds": backend.load_seconds, "fields": timings}}
        text = json.dumps(output, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            Path(args.output).write_text(text)
        else: print(text, end="")
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
