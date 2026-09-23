"""Fit per-type/count-bucket scalar temperatures from a JSONL calibration fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_decision.calibration import TemperatureCalibrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL rows with partition, decision_type, option_count, logits, target_index")
    parser.add_argument("output", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--unknown-offsets", action="store_true",
                        help="also fit a per-bucket additive offset on the unknown logit (applied to text-only requests)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    artifact = TemperatureCalibrator.fit(rows, version=args.version, unknown_offsets=args.unknown_offsets).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "version": artifact["calibration_version"],
                      "buckets": artifact["counts"], "temperatures": artifact["temperatures"],
                      "unknown_offsets": artifact.get("unknown_offsets")}, sort_keys=True))


if __name__ == "__main__":
    main()
