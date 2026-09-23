"""Does `pd12m` share any stored image with another decision-v2 image source?

PD12M's biggest contributor is Wikimedia Commons, and `commons_photos` crawled Commons directly,
so the two sources could in principle hold the same photograph twice -- which would let one photo
sit on both sides of a train/test split through two different `source_group`s.

Every v2 image source re-encodes through the same `scripts/v1/_common.py:store_image` (RGB JPEG
q90, <= 1 MP, deterministic), so an identical ORIGINAL lands on an identical sha256 and shows up
here.  The same photograph re-encoded differently upstream would NOT, and the README says so.

Reads only; writes `data/decision-v2/pd12m/cross_source_overlap.json`.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/pd12m_overlap_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT  # noqa: E402

SOURCE = "pd12m"
OTHERS = ("commons_photos", "openimages_v2")


def shas(source: str) -> set[str]:
    d = OUT / source / "images"
    return {p.stem for p in d.glob("*.jpg")} if d.is_dir() else set()


def main() -> int:
    mine = shas(SOURCE)
    report = {"pd12m_images": len(mine), "per_source": {}, "shared_sha256": 0,
              "other_images": 0, "examples": []}
    shared: set[str] = set()
    for other in OTHERS:
        theirs = shas(other)
        both = mine & theirs
        shared |= both
        report["per_source"][other] = {"images": len(theirs), "shared_sha256": len(both)}
        report["other_images"] += len(theirs)
    report["shared_sha256"] = len(shared)
    report["examples"] = sorted(shared)[:10]
    report["note"] = ("byte-identical originals only; both sources re-encode with the same "
                      "deterministic store, so a shared sha256 means the same original bytes")
    (OUT / SOURCE / "cross_source_overlap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
