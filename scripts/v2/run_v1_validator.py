"""Run the UNMODIFIED v1 gate `scripts/v1/validate_records.py` against a decision-v2 image source.

The v1 script is hard-wired to `data/decision-v1/<source>/`, so this builds a throw-away directory
whose `data/decision-v1/<name>/records.jsonl` is a copy of the v2 records and whose
`data/decision-v2` is a symlink to the real one (that is where the records' image paths point).
Nothing in the repo is written except the report this saves next to the source.

The point is to record *exactly* what the v1 gate says about spec-shaped v2 rows rather than to
claim it passes: a `pending` row has `target: null` AND `abstention_cause: null`, which v1 reads as
a contradiction.  `scripts/v2/validate_image_records.py` is the ported gate that these rows do pass.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/run_v1_validator.py openimages_v2
"""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, ROOT


def main(source: str) -> int:
    records = OUT / source / "records.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data" / "decision-v1" / source).mkdir(parents=True)
        (tmp / "src").symlink_to(ROOT / "src")
        (tmp / "data" / "decision-v2").symlink_to(OUT)
        shutil.copy(ROOT / "data" / "decision-v1" / "exclusions.json", tmp / "data" / "decision-v1" / "exclusions.json")
        shutil.copy(records, tmp / "data" / "decision-v1" / source / "records.jsonl")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v1" / "validate_records.py"), source],
            cwd=tmp, capture_output=True, text=True)
        out = json.loads((tmp / "data" / "decision-v1" / source / "validation.json").read_text())
    kinds = collections.Counter(e.split(": ", 1)[1] for e in out["first_errors"])
    report = {
        "ran": "scripts/v1/validate_records.py (unmodified), against a copy of "
               f"data/decision-v2/{source}/records.jsonl",
        "exit_code": proc.returncode,
        "records": out["summary"]["records"],
        "errors": out["summary"]["errors"],
        "error_kinds_in_the_first_50": dict(kinds),
        "reading": "The error count equals the record count, and every error v1 saved (it saves the "
                   "first 50) is the same one: a decision-v2 candidate is unanswered by "
                   "construction (target null, abstention_cause null, pseudo_label 'pending'), "
                   "which v1 reads as a contradiction. v1 does not save the full error list, so "
                   "this alone does not prove nothing else failed; the ported gate "
                   "scripts/v2/validate_image_records.py runs the same checks plus the licence "
                   "receipts and reports 0 errors, and that is the claim to rely on.",
        "v2_gate": f"scripts/v2/validate_image_records.py {source}",
    }
    path = OUT / source / "v1_validator_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
