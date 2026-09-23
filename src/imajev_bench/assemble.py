"""Assemble benchmark records from an authoring spec with cluster-level splits.

Authors write sources (licensed images with a source_cluster naming the physical scene,
product, document or template) and items (question, typed field, draft label, evidence).
The assembler hashes and copies assets under content names, links every item to its
evidence cluster, assigns whole clusters to splits deterministically, enforces reuse caps
and validates the result. Labels stay `draft` until two human reviews are imported.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import validate_records
from .stats import evidence_clusters

SPLITS = ("dev", "calibration", "test")
ITEM_KEYS = {"id", "track", "family", "images", "state", "field", "draft_gold", "evidence", "contrast",
             "judgement_dependent", "difficulty", "answerability", "construction"}


def _split_for(anchor: str, salt: str, fractions: dict[str, float]) -> str:
    point = int(hashlib.sha256(f"{salt}\0{anchor}".encode()).hexdigest()[:12], 16) / 16 ** 12
    edge = 0.0
    for split in SPLITS:
        edge += fractions.get(split, 0.0)
        if point < edge:
            return split
    return SPLITS[-1]


def assemble(spec: dict[str, Any], spec_dir: Path, output: Path) -> dict[str, Any]:
    spec_dir, output = Path(spec_dir), Path(output)
    fractions = spec.get("split_fractions", {"dev": 0.2, "calibration": 0.1, "test": 0.7})
    if set(fractions) - set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split_fractions must use dev/calibration/test and sum to 1")
    cap = int(spec.get("max_records_per_image", 3))
    salt = spec.get("split_salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("split_salt is required so splits are reproducible and deliberate")

    sources = {}
    for source in spec["sources"]:
        if source["id"] in sources:
            raise ValueError(f"duplicate source {source['id']!r}")
        if not source.get("source_cluster"):
            raise ValueError(f"source {source['id']!r} needs a source_cluster (scene/product/document/template)")
        provenance = source.get("provenance") or {}
        if source.get("synthetic"):
            if not provenance.get("generator"):
                raise ValueError(f"synthetic source {source['id']!r} needs provenance.generator")
        elif not all(provenance.get(k) for k in ("source", "source_page", "creator", "license")):
            raise ValueError(f"source {source['id']!r} needs source, source_page, creator and license")
        sources[source["id"]] = source

    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    copied = {}
    records = []
    for item in spec["items"]:
        unknown_keys = set(item) - ITEM_KEYS
        if unknown_keys:
            raise ValueError(f"item {item.get('id')!r} has unknown keys {sorted(unknown_keys)}")
        refs, clusters, image_sources = [], [], []
        for source_id in item.get("images", []):
            source = sources[source_id]
            if source_id not in copied:
                path = spec_dir / source["path"]
                blob = path.read_bytes()
                sha = hashlib.sha256(blob).hexdigest()
                name = f"assets/{sha[:16]}{path.suffix.lower()}"
                shutil.copyfile(path, output / name)
                copied[source_id] = {"path": name, "sha256": sha}
            refs.append(copied[source_id])
            clusters.append(source["source_cluster"])
            image_sources.append({"source_id": source_id, "synthetic": bool(source.get("synthetic")),
                                  **(source.get("provenance") or {})})
        synthetic = [s["synthetic"] for s in image_sources]
        provenance = {"source_clusters": sorted(set(clusters)), "image_sources": image_sources,
                      "synthetic": all(synthetic) if synthetic else None,
                      "draft_evidence": item.get("evidence"), "answerability": item.get("answerability"),
                      "judgement_dependent": bool(item.get("judgement_dependent", False)),
                      "difficulty": item.get("difficulty"),
                      "label_origin": "author draft; requires two independent human reviews",
                      "review_protocol": "imajev-bench-blind-review-v1"}
        if synthetic and all(synthetic):
            provenance["generator"] = image_sources[0].get("generator")
        if item.get("contrast"):
            provenance["contrast"] = item["contrast"]
        if item.get("construction"):
            provenance["construction"] = item["construction"]
        # group_id is provisional (one per item) until evidence clusters are known.
        records.append({"id": item["id"], "group_id": item["id"], "track": item["track"], "family": item["family"],
                        "split": "dev", "images": refs,
                        "request": {"schema_version": "1.0", "request_id": item["id"], "state": item.get("state", {}),
                                    "fields": [item["field"]], "execution": {"mode": "inspect", "allow_external_fallback": False}},
                        "gold": item.get("draft_gold"), "annotation_status": "draft", "provenance": provenance})

    per_image = Counter(ref["sha256"] for r in records for ref in r["images"])
    over = {k[:12]: v for k, v in per_image.items() if v > cap}
    if over:
        shutil.rmtree(output)
        raise ValueError(f"images exceed max_records_per_image={cap}: {over}")

    components = evidence_clusters(records)
    members = defaultdict(list)
    for r in records:
        members[components[r["id"]]].append(r)
    for rows in members.values():
        anchors = sorted({c for r in rows for c in r["provenance"]["source_clusters"]}
                         | {r["provenance"]["contrast"]["set_id"] for r in rows if "contrast" in r["provenance"]}
                         | ({rows[0]["id"]} if not any(r["provenance"]["source_clusters"] for r in rows) else set()))
        anchor = anchors[0]
        split = _split_for(anchor, salt, fractions)
        group = "grp-" + hashlib.sha256(anchor.encode()).hexdigest()[:12]
        for r in rows:
            r["group_id"], r["split"] = group, split

    try:
        validated = validate_records(records, output)
    except ValueError:
        shutil.rmtree(output)
        raise
    with (output / "records.jsonl").open("x") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    receipt = {"dataset": spec.get("dataset"), "version": spec.get("version"), "records": len(validated),
               "unique_images": len(per_image), "evidence_clusters": len(members),
               "split_records": dict(Counter(r["split"] for r in validated)),
               "split_clusters": dict(Counter(rows[0]["split"] for rows in members.values())),
               "split_salt": salt, "split_fractions": fractions, "max_records_per_image": cap,
               "records_sha256": hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()}
    (output / "assembly.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
