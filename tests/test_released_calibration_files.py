"""The calibration.json files shipped with the released adapters must load in the server and cover every bucket.

A missing bucket is served uncalibrated (T=1) and a missing `counts` field makes the server refuse to start, so both are checked
on the actual artefacts in `results/imajev-1.0/` (and on their sources in a development checkout).
"""
from pathlib import Path

import pytest

from vision_decision.calibration import TemperatureCalibrator, calibration_key

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(ROOT.glob("reports/decision-p2b/calibration-p2b-*-final.json")) + sorted(ROOT.glob("results/imajev-1.0/calibration-imajev-*.json"))
# every (type, option count) the request contract allows: yes/no; choice over 2-254 options; score over 2-10 levels
REQUESTS = [("boolean", 2)] + [("choice", n) for n in (2, 3, 5, 6, 10, 11, 25, 26, 254)] + [("ordinal", n) for n in (2, 3, 5, 6, 10)]


@pytest.mark.skipif(not FILES, reason="no released calibration files in this checkout")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_released_calibration_loads_and_covers_every_bucket(path):
    calibrator = TemperatureCalibrator.load(str(path))
    for decision_type, option_count in REQUESTS:
        temperature = calibrator.temperature(decision_type, option_count)
        assert temperature is not None, f"{calibration_key(decision_type, option_count)} would be served uncalibrated"
        assert temperature > 0
