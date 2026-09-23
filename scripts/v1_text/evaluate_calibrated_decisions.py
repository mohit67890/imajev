"""Report raw/calibrated gates and optional paired family-bootstrap comparisons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vision_decision.calibration import TemperatureCalibrator
try:  # Supports both direct execution and `python -m scripts.v1_text...`.
    from .calibration_evaluation import (
        evaluate, paired_family_bootstrap, paired_irrelevance_delta, release_acceptance_gates,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI smoke tests
    from calibration_evaluation import (
        evaluate, paired_family_bootstrap, paired_irrelevance_delta, release_acceptance_gates,
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--paired-baseline", type=Path,
                        help="JSONL with id/family/correct, aligned to prediction evaluation rows")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system-metrics", type=Path,
                        help="JSON with image accuracies, irrelevance delta, latency, and baseline statuses")
    parser.add_argument("--irrelevance-predictions", type=Path,
                        help="JSONL paired control rows with pair_id, control_variant, and correct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    report = evaluate(rows, TemperatureCalibrator.load(args.calibration))
    paired_control = None
    if args.irrelevance_predictions:
        paired_control = paired_irrelevance_delta(read_jsonl(args.irrelevance_predictions),
                                                   iterations=args.bootstrap_iterations, seed=args.seed)
        report["paired_irrelevance"] = paired_control
    if args.system_metrics:
        system = json.loads(args.system_metrics.read_text())
        if paired_control is not None:
            supplied = system.get("irrelevant_image_delta")
            if supplied is not None and not math.isclose(supplied, paired_control["irrelevant_image_delta"], abs_tol=1e-12):
                raise ValueError("system irrelevant_image_delta disagrees with paired controls")
            system["irrelevant_image_delta"] = paired_control["irrelevant_image_delta"]
        report["release_acceptance"] = release_acceptance_gates(report["calibrated"], system)
        if "external_baselines" in system:
            report["external_baselines"] = system["external_baselines"]
    if args.paired_baseline:
        report["paired_family_bootstrap"] = paired_family_bootstrap(
            read_jsonl(args.paired_baseline), rows, iterations=args.bootstrap_iterations, seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
