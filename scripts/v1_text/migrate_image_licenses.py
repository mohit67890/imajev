#!/usr/bin/env python3
"""Build a strict image-source license mapping from locally retained evidence.

This intentionally admits only sources whose primary publisher grants a
commercially usable license over the downloaded work. Dataset names, converter
constants, papers, and Hub metadata are not license evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/decision-v1.jsonl"
PROVENANCE = ROOT / "data/provenance"
REPORT = ROOT / "reports/v1.1-datasets/image-license-migration.json"
EVIDENCE = PROVENANCE / "licenses/vizwiz/vqa-dataset-page.html"
RECEIPT = EVIDENCE.with_name(EVIDENCE.name + ".receipt.json")
SOURCE_PAGE = PROVENANCE / "vizwiz-vqa.html"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def license_object(evidence: Path, receipt: Path, expected_spdx: str) -> dict:
    meta = json.loads(receipt.read_text())
    digest = sha256(evidence)
    required = {
        "evidence_sha256": digest,
        "spdx": expected_spdx,
        "commercial_use_reviewed": True,
    }
    if any(meta.get(key) != value for key, value in required.items()):
        raise ValueError(f"invalid strict license receipt: {receipt}")
    return {
        "spdx": expected_spdx,
        "evidence": str(evidence.relative_to(ROOT)),
        "evidence_sha256": digest,
        "receipt": str(receipt.relative_to(ROOT)),
        "scope": meta["scope"],
        "attribution": meta["attribution"],
    }


def counts() -> tuple[Counter, dict[str, set]]:
    records, groups = Counter(), {}
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            source = row["source"]
            records[source] += 1
            groups.setdefault(source, set()).add(row["source_group"])
    return records, groups


def build_report() -> dict:
    records, groups = counts()
    vizwiz_license = license_object(EVIDENCE, RECEIPT, "CC-BY-4.0")
    admitted = {
        "vizwiz": {
            "record_count": records["vizwiz"],
            "source_group_count": len(groups["vizwiz"]),
            "license": vizwiz_license,
            "derivation": (
                "Questions, VQA annotations, and images are downloaded from links on the retained "
                "official VizWiz-VQA page; its page-level CC BY 4.0 grant applies to that work."
            ),
        }
    }
    excluded = {
        "vizwiz_quality": {
            "record_count": records["vizwiz_quality"],
            "source_group_count": len(groups["vizwiz_quality"]),
            "reason": (
                "The reused VizWiz-VQA image bytes have a CC BY 4.0 grant, but the official "
                "VizWiz-QualityIssues page does not state a license for its distinct flaw and "
                "unrecognizability annotations. The record as a whole is therefore not admitted."
            ),
            "reviewed_primary_sources": [
                "https://vizwiz.org/tasks-and-datasets/image-quality-issues/"
            ],
        },
        "gqa": {
            "record_count": records["gqa"],
            "source_group_count": len(groups["gqa"]),
            "reason": (
                "The official GQA site says its images come from COCO and Flickr but supplies no "
                "dataset-wide image license grant. No per-image upstream license mapping is retained."
            ),
            "reviewed_primary_sources": [
                "https://cs.stanford.edu/people/dorarad/gqa/",
                "https://github.com/dorarad/gqa",
            ],
        },
        "vg_attributes": {
            "record_count": records["vg_attributes"],
            "source_group_count": len(groups["vg_attributes"]),
            "reason": (
                "The official Visual Genome site and API describe and distribute images and annotations "
                "but provide no retained license grant covering the image bytes. An annotation license "
                "would not establish rights to the underlying Flickr images."
            ),
            "reviewed_primary_sources": [
                "https://visualgenome.org/home",
                "https://visualgenome.org/api/v0/api_endpoint_reference",
            ],
        },
    }
    admitted_records = sum(x["record_count"] for x in admitted.values())
    return {
        "schema_version": 1,
        "policy": (
            "Admit only a primary source grant whose scope covers the actual distributed image work; "
            "repository code licenses, papers, converter constants, and metadata tags do not qualify."
        ),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "mapping_by_source": admitted,
        "excluded": excluded,
        "summary": {
            "admitted_sources": len(admitted),
            "admitted_records": admitted_records,
            "admitted_source_groups_sum": sum(x["source_group_count"] for x in admitted.values()),
            "target_records": 300000,
            "target_met": admitted_records >= 300000,
            "shortfall_records": max(0, 300000 - admitted_records),
            "note": "Source-group counts overlap because VizWiz task families reuse images.",
        },
    }


def main() -> None:
    if not SOURCE_PAGE.is_file():
        raise FileNotFoundError(SOURCE_PAGE)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    if not EVIDENCE.exists():
        EVIDENCE.write_bytes(SOURCE_PAGE.read_bytes())
    html = EVIDENCE.read_text(errors="replace")
    if "Creative Commons Attribution 4.0 International License" not in html:
        raise ValueError("retained official page does not contain the reviewed grant")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
