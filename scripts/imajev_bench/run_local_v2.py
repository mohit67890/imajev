"""Run one local model through the v2 harness (prompt profile, full rotations, ablations, timing).

GPU use must be coordinated: the run refuses to start without --gpu-coordinated, and the
harness marks every case whose timing overlapped another busy process.

Example:
  .venv/bin/python scripts/imajev_bench/run_local_v2.py --model qwen9b --records data/imajev-bench/v2-pilot/records.jsonl \
      --split dev --profile benchmark-neutral --condition full --output reports/imajev-bench-v2/qwen9b-neutral-full \
      --gpu-coordinated
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imajev_bench.cli import read_jsonl  # noqa: E402
from imajev_bench.harness import CONDITIONS, PROFILES, MLXBackend, run_local  # noqa: E402
from imajev_bench.schema import validate_records  # noqa: E402

MODELS = {
    "qwen2b": ("artifacts/model.json", None, "backend"),
    "imajev2b": ("artifacts/model.json", "reports/decision-v1.1/runs/h100x4/best-mlx", "backend"),
    "imajev2b-v21": ("artifacts/model.json", "reports/decision-v2.1/runs/h100x4/last-step1361-mlx", "backend"),
    "qwen9b": ("artifacts/model-qwen9b.json", None, "backend"),
    "imajev9b": ("artifacts/model-qwen9b.json", "reports/decision-v1.1-9b/runs/h200x4/best-mlx", "backend"),
    "gemma": ("artifacts/model-gemma.json", None, "gemma_backend"),
    "smolvlm": ("artifacts/model-smolvlm2.json", None, "smolvlm_backend"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", choices=("dev", "calibration", "test"), required=True)
    parser.add_argument("--profile", choices=PROFILES, default="benchmark-neutral")
    parser.add_argument("--condition", choices=CONDITIONS, default="full")
    parser.add_argument("--rotations", default="full", help="'full' or a number of evenly spaced orders")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--limit", type=int, help="Score only the first N selected records (smoke tests)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-coordinated", action="store_true", help="Confirm no other GPU workload is scheduled")
    args = parser.parse_args(argv)
    if not args.gpu_coordinated:
        parser.error("Coordinate GPU use first, then pass --gpu-coordinated")
    os.chdir(ROOT)
    root = args.root or args.records.parent
    records = [r for r in validate_records(read_jsonl(args.records), root, require_reviewed=not args.allow_draft)
               if r["split"] == args.split][:args.limit]
    bundle, adapter, module = MODELS[args.model]
    if module == "backend":
        from vision_decision.backend import MLXDirect as Direct
    elif module == "gemma_backend":
        from vision_decision.gemma_backend import GemmaDirect as Direct
    else:
        from vision_decision.smolvlm_backend import SmolVLMDirect as Direct
    direct = Direct(bundle, adapter=adapter) if adapter else Direct(bundle)
    if adapter and direct.readout is None:
        raise SystemExit("Tuned model loaded without its trained decision readout")
    identity = {"name": args.model, **json.loads(Path(bundle).read_text())}
    sources = sorted((ROOT / "src/imajev_bench").glob("*.py")) + sorted((ROOT / "src/vision_decision").glob("*.py")) \
        + [Path(__file__).resolve()]
    if adapter:
        sources += sorted(p for p in (ROOT / adapter).iterdir() if p.suffix == ".json")
    rotations = args.rotations if args.rotations == "full" else int(args.rotations)
    print(run_local(records, root, args.output, MLXBackend(direct, identity), profile=args.profile,
                    condition=args.condition, rotations=rotations, warmup=args.warmup, repeats=args.repeats,
                    sources=sources, command=sys.argv))


if __name__ == "__main__":
    main()
