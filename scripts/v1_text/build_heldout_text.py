"""Assemble the evaluation-only held-out TEXT panel for decision-v1.1.

Two outputs, because the two sources have different license standing:

* ``data/manifests/decision-v1.1-heldout-text.jsonl`` -- typed-decisions test split,
  carrying verified Apache-2.0 evidence, audited by ``v1_text.audit_mixture``.
* ``data/manifests/decision-v1.1-heldout-text-unlicensed.jsonl`` -- SST-5 test split,
  ``license: null`` plus a ``license_note``, because no retained SST file grants one.
  Nothing here may enter training or be redistributed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import write_jsonl
from .convert_sst5 import convert as convert_sst5
from .convert_typed_decisions import convert as convert_typed_decisions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decision-v1-text"
TYPED_SOURCE = DATA / "raw/typed-decisions/test.parquet"
TYPED_LICENSE = Path("data/decision-v1-text/licenses/typed_decisions/README.md")
LICENSED_OUTPUT = ROOT / "data/manifests/decision-v1.1-heldout-text.jsonl"
UNLICENSED_OUTPUT = ROOT / "data/manifests/decision-v1.1-heldout-text-unlicensed.jsonl"
SOURCE_NAME = "typed_decisions_test"


def typed_decisions_panel() -> list[dict]:
    """Typed-decisions official test split, renamed so the audit sees a held-out source."""
    rows = convert_typed_decisions(TYPED_SOURCE, TYPED_LICENSE, "Apache-2.0")
    out = []
    for row in rows:
        if row["source"] != "typed_decisions" or row["source_split"] != "test":
            raise ValueError(f"{row['id']}: unexpected source/split for a held-out panel")
        row["id"] = row["id"].replace("typed_decisions:", f"{SOURCE_NAME}:", 1)
        row["source"] = SOURCE_NAME
        row["source_group"] = f"{SOURCE_NAME}:{row['source_group']}"
        row["heldout_family"] = True
        out.append(row)
    return out


def check(rows: list[dict], expect_license: bool) -> None:
    if not rows:
        raise ValueError("empty panel")
    for row in rows:
        if row.get("partition") != "test":
            raise ValueError(f"{row['id']}: partition is not test")
        if row.get("heldout_family") is not True:
            raise ValueError(f"{row['id']}: heldout_family is not true")
        if row.get("images"):
            raise ValueError(f"{row['id']}: the text panel must carry no images")
        if expect_license and not isinstance(row.get("license"), dict):
            raise ValueError(f"{row['id']}: licensed panel needs license evidence")
        if not expect_license and (row.get("license") is not None or not row.get("license_note")):
            raise ValueError(f"{row['id']}: unlicensed panel needs license null and a license_note")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate ids in panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--licensed-output", type=Path, default=LICENSED_OUTPUT)
    parser.add_argument("--unlicensed-output", type=Path, default=UNLICENSED_OUTPUT)
    args = parser.parse_args()

    typed = typed_decisions_panel()
    check(typed, expect_license=True)
    write_jsonl(args.licensed_output, typed)

    sst5 = convert_sst5()
    check(sst5, expect_license=False)
    write_jsonl(args.unlicensed_output, sst5)

    summary = {
        "licensed": {"output": str(args.licensed_output.relative_to(ROOT)), "records": len(typed),
                     "by_source": dict(sorted(Counter(r["source"] for r in typed).items())),
                     "by_family": dict(sorted(Counter(r["family"] for r in typed).items())),
                     "decisions": sum(len(r["request"]["fields"]) for r in typed)},
        "unlicensed": {"output": str(args.unlicensed_output.relative_to(ROOT)), "records": len(sst5),
                       "by_source": dict(sorted(Counter(r["source"] for r in sst5).items())),
                       "by_level": dict(sorted(Counter(r["target"] for r in sst5).items())),
                       "decisions": sum(len(r["request"]["fields"]) for r in sst5)},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
