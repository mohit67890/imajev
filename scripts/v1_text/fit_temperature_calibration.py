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
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    artifact = TemperatureCalibrator.fit(rows, version=args.version).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "version": artifact["calibration_version"],
                      "buckets": artifact["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
