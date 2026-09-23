#!/usr/bin/env python3
"""Admit the decision-v1 image sources to v1.1 under the v1 image posture (policy A).

Policy A, approved by the project owner on 2026-09-22: v1.1 trains on the same image
data as v1. Each source is admitted under the licence of its annotations, exactly as the
decision-v1 source README documents it (URL plus quoted grant), and the image bytes are
used under the upstream terms recorded there (COCO/Flickr, OpenImages, YFCC100M, retailer
listings). Nothing is relicensed. The stricter primary-grant policy (``migrate_image_
licenses.py``) is retained for comparison; this script records, per source, which
posture admits it and why, so the model card can list it.

Evidence files are the licence sections of the retained v1 READMEs, copied verbatim under
``data/provenance/licenses/v1-image/<source>/<SPDX>.md`` with a receipt beside each one,
which is what ``common.verified_license`` requires.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/decision-v1.jsonl"
SOURCES = ROOT / "data/decision-v1"
EVIDENCE_ROOT = ROOT / "data/provenance/licenses/v1-image"
REPORT = ROOT / "reports/v1.1-datasets/image-policy-a.json"
REVIEWED_AT = "2026-09-22"

# Sources that only exist in the v1 test partition are evaluated through the v1 image exam,
# not carried into the v1.1 manifest (heldout_livewild has a custom grant outside the allowlist).
EXCLUDED_PREFIX = "heldout_"

# Per-source posture notes, taken from the v1 READMEs. These are the caveats the model card lists.
CAVEATS = {
    "aokvqa": "Annotations Apache-2.0; images are COCO (Flickr terms), not relicensed.",
    "bool_abstain": "Annotations CC BY 4.0 via TDIUC; images are COCO (Flickr terms), not relicensed.",
    "tdiuc": "Annotations CC BY 4.0 (declared in the TDIUC annotation JSON); images are COCO (Flickr terms).",
    "unkvqa": "Annotations Apache-2.0; images are COCO (Flickr terms), not relicensed.",
    "vqav2": "Annotations CC BY 4.0 (declared inside the annotation file); images are COCO (Flickr terms).",
    "gqa": "Annotations CC BY 4.0 (official download page); images are Visual Genome (Flickr terms).",
    "vg_attributes": "Annotations CC BY 4.0 (Visual Genome about page); images are Flickr photos under their own terms.",
    "masked_evidence": "Derived from gqa/vg_attributes annotations (CC BY 4.0) with synthetic masks; same image terms.",
    "tallyqa": "Apache-2.0 repository licence; images are COCO and Visual Genome.",
    "vsr": "Apache-2.0 repository licence; images are COCO.",
    "textvqa": "Annotations CC BY 4.0 (curators' dataset card); images are OpenImages (CC BY 2.0 Flickr photos).",
    "koniq": "Dataset CC BY 4.0; per-image Creative Commons terms inherited from YFCC100M are not individually verified and may include NonCommercial photos.",
    "fashion200k": "Apache-2.0 dataset card; product images were crawled from Lyst and their rights are unverified.",
    "marqo_gs": "Apache-2.0 dataset card; product images were scraped from Google Shopping listings and their rights are unverified.",
    "sqid_esci": "Apache-2.0 (ESCI judgments) and MIT (SQID image URLs); product images are Amazon CDN assets with no rights granted.",
    "dude": "CC BY 4.0 per the dataset card; per-document upstream licences listed in the repo's data-lineage CSV were not individually verified.",
    "vizwiz_quality": "Images share the VizWiz-VQA CC BY 4.0 grant; the quality-issue annotation page states no licence of its own.",
    "defects": "VisA CC BY 4.0, DAGM 2007 CC BY 4.0 and BTAD CC BY-SA 4.0 (share-alike applies to the BTAD-derived records).",
    "state_aware": "Inherits the licence and image terms of its origin source (abo, marqo_gs, sqid_esci, vg_attributes, tdiuc, vqav2).",
}
STRICT_GRANT = {"abo": "Primary CC BY 4.0 grant covers listings and images.",
                "vizwiz": "Primary CC BY 4.0 grant on the official VizWiz-VQA page covers images and annotations."}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spdx_of(license_string: str) -> str:
    return license_string.split()[0].strip("()/,")


def licence_section(readme: Path) -> str:
    text = readme.read_text()
    match = re.search(r"^##[^\n]*[Ll]icen[^\n]*\n", text, flags=re.M)
    if not match:
        return text  # short READMEs (bool_abstain) state the licence in a single paragraph
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, flags=re.M)
    return match.group(0) + (rest[: nxt.start()] if nxt else rest)


def write_evidence(source: str, spdx: str, license_string: str) -> dict:
    readme = SOURCES / source / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(readme)
    folder = EVIDENCE_ROOT / source
    folder.mkdir(parents=True, exist_ok=True)
    evidence = folder / f"{spdx}.md"
    body = (f"# Licence evidence: decision-v1 source `{source}` ({spdx})\n\n"
            f"Copied verbatim from `data/decision-v1/{source}/README.md` (README sha256 {sha256(readme)}).\n"
            f"v1 manifest licence string: `{license_string}`\n\n" + licence_section(readme))
    if not evidence.exists() or evidence.read_text() != body:
        evidence.write_text(body)
    receipt = evidence.with_name(evidence.name + ".receipt.json")
    meta = {
        "spdx": spdx,
        "evidence_sha256": sha256(evidence),
        "commercial_use_reviewed": True,
        "source": source,
        "v1_license_string": license_string,
        "policy": "A: v1 image posture",
        "scope": ("Annotation licence as documented in the retained decision-v1 source README. Image bytes are "
                  "used under the upstream terms recorded there and are not relicensed."),
        "caveat": CAVEATS.get(source, STRICT_GRANT.get(source, "")),
        "reviewed_at": REVIEWED_AT,
        "reviewed_by": "Project owner decision (policy A), evidence extracted from the decision-v1 README by the v1.1 build session",
    }
    receipt.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return {"spdx": spdx, "evidence": str(evidence.relative_to(ROOT)), "evidence_sha256": meta["evidence_sha256"],
            "receipt": str(receipt.relative_to(ROOT))}


def main() -> None:
    strings: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    partitions: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    origins: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            source = row["source"]
            strings[source][row["license"]] += 1
            partitions[source][row["partition"]] += 1
            if source == "state_aware":
                origins[row["source_group"].split(":", 1)[0]][row["license"]] += 1
    mapping: dict[str, dict[str, dict]] = {}
    excluded = {}
    for source in sorted(strings):
        if source.startswith(EXCLUDED_PREFIX):
            excluded[source] = {"records": sum(strings[source].values()), "partitions": dict(partitions[source]),
                                "reason": "test-only v1 exam source; evaluated with the v1 image exam, not carried into the v1.1 manifest"}
            continue
        if source == "state_aware":
            continue
        mapping[source] = {ls: write_evidence(source, spdx_of(ls), ls) for ls in strings[source]}
    # state_aware inherits its origin's evidence, matched on the origin's own licence string.
    # The `coco:` prefix carries vqav2 and tdiuc records, told apart by their licence strings.
    state = {}
    for origin, counter in origins.items():
        for ls in counter:
            if origin in mapping and ls in mapping[origin]:
                resolved = origin
            else:
                candidates = sorted(s for s, m in mapping.items() if ls in m)
                if len(candidates) != 1:
                    raise ValueError(f"state_aware origin {origin!r} with licence {ls!r} matches {candidates}")
                resolved = candidates[0]
            state.setdefault(origin, {})[ls] = dict(mapping[resolved][ls], origin_source=resolved)
    report = {
        "schema_version": 1,
        "policy": ("A: v1 image posture. Sources are admitted under their annotation licence as documented in the "
                   "decision-v1 READMEs; image bytes remain under upstream terms and are not relicensed. Approved by the "
                   f"project owner on {REVIEWED_AT} for the v1.1 research release, in preference to the strict "
                   "primary-grant policy of migrate_image_licenses.py, which admits only abo and vizwiz."),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "mapping_by_source": mapping,
        "state_aware_by_origin": state,
        "excluded": excluded,
        "caveats": {s: CAVEATS.get(s, STRICT_GRANT.get(s, "")) for s in sorted(mapping) + ["state_aware"]},
        "strict_policy_admits": sorted(STRICT_GRANT),
        "summary": {
            "admitted_sources": len(mapping) + 1,
            "admitted_records": sum(sum(c.values()) for s, c in strings.items() if s in mapping or s == "state_aware"),
            "excluded_sources": len(excluded),
            "spdx_by_source": {s: sorted({spdx_of(ls) for ls in c}) for s, c in sorted(strings.items()) if s in mapping},
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
