"""Model-assisted review triage. Models route work; humans decide every label.

Route per record:
- single: model pre-labels from >= MIN_CONSENSUS_PROVIDERS providers agree unanimously, the item
  is not judgement-dependent and not in the audit sample. One blind human review; if it agrees,
  the record is consensus-verified, otherwise it gets a second review and adjudication.
- double: everything else. Two blind human reviews, adjudicated on disagreement.

A seeded audit sample of whole evidence clusters among consensus items takes the double route
so the consensus error rate can be measured and published.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from typing import Any

from .runner import digest
from .schema import MIN_CONSENSUS_PROVIDERS, model_payload
from .stats import evidence_clusters


def consensus(record: dict, min_providers: int = MIN_CONSENSUS_PROVIDERS) -> tuple[bool, Any]:
    """(True, value) when current-input pre-labels from enough providers agree exactly."""
    expected = digest(model_payload(record))
    labels = [x for x in record["provenance"].get("model_prelabels", []) if x.get("input_sha256") == expected]
    if len({x["provider"] for x in labels}) < min_providers:
        return False, None
    values = {(type(x["value"]).__name__, json.dumps(x["value"])) for x in labels}
    if len(values) != 1:
        return False, None
    return True, labels[0]["value"]


def plan_reviews(records: list[dict], audit_share: float = 0.15, seed: int = 0,
                 min_providers: int = MIN_CONSENSUS_PROVIDERS) -> tuple[list[dict], dict]:
    if not 0 <= audit_share <= 1:
        raise ValueError("audit_share must be between 0 and 1")
    result = copy.deepcopy(records)
    clusters = evidence_clusters(result)
    eligible = defaultdict(list)
    for record in result:
        agreed, _ = consensus(record, min_providers)
        if agreed and not record["provenance"].get("judgement_dependent"):
            eligible[clusters[record["id"]]].append(record)
    cluster_ids = sorted(eligible)
    rng = random.Random(hashlib.sha256(f"audit\0{seed}".encode()).digest())
    audited = set(rng.sample(cluster_ids, round(len(cluster_ids) * audit_share))) if cluster_ids else set()
    counts = defaultdict(int)
    for record in result:
        provenance = record["provenance"]
        cluster = clusters[record["id"]]
        in_consensus = any(record is r for r in eligible.get(cluster, []))
        provenance["audit_sample"] = in_consensus and cluster in audited
        provenance["review_plan"] = "single" if in_consensus and not provenance["audit_sample"] else "double"
        counts[provenance["review_plan"]] += 1
        counts["audit"] += provenance["audit_sample"]
    summary = {"records": len(result), "single": counts["single"], "double": counts["double"],
               "audit_sample": counts["audit"], "audit_share": audit_share, "seed": seed,
               "min_providers": min_providers,
               "double_packet_sha256": packet_sha256([r for r in result if r["provenance"]["review_plan"] == "double"])}
    return result, summary


def plan_construction(records: list[dict], audit_share: float = 0.15, seed: int = 0) -> tuple[list[dict], dict]:
    """Promote constructed records; route a seeded audit sample of whole clusters to one blind human.

    Records without provenance.construction are left untouched for the review routes above.
    """
    result = copy.deepcopy(records)
    clusters = evidence_clusters(result)
    constructed = [r for r in result if isinstance(r["provenance"].get("construction"), dict)]
    cluster_ids = sorted({clusters[r["id"]] for r in constructed})
    rng = random.Random(hashlib.sha256(f"construction-audit\0{seed}".encode()).digest())
    audited = set(rng.sample(cluster_ids, round(len(cluster_ids) * audit_share))) if cluster_ids else set()
    counts = defaultdict(int)
    for record in constructed:
        provenance = record["provenance"]
        provenance["audit_sample"] = (clusters[record["id"]] in audited or bool(provenance.get("judgement_dependent"))
                                      or bool(provenance["construction"].get("occlusion_check")))
        if provenance["audit_sample"]:
            provenance["review_plan"] = "single"
            record["annotation_status"] = "draft"
            counts["audit"] += 1
        else:
            provenance["review_route"] = "construction_verified"
            record["annotation_status"] = "reviewed"
            counts["promoted"] += 1
    return result, {"constructed": len(constructed), "promoted": counts["promoted"], "audit_sample": counts["audit"],
                    "audit_share": audit_share, "seed": seed,
                    "audit_packet_sha256": packet_sha256([r for r in result if r["provenance"].get("audit_sample")])}


def packet_sha256(records: list[dict]) -> str:
    """The input hash a blind review packet built from exactly these records exports."""
    hashes = sorted((r["id"], digest(model_payload(r))) for r in records)
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def audit_report(records: list[dict], min_providers: int = MIN_CONSENSUS_PROVIDERS) -> dict:
    """Consensus error rate on audited records whose final label came from two humans."""
    audited = [r for r in records if r["provenance"].get("audit_sample") and r["annotation_status"] == "reviewed"]
    errors = []
    for record in audited:
        construction = record["provenance"].get("construction")
        if isinstance(construction, dict) and "truth" in construction:
            agreed, value = True, construction["truth"]
        else:
            agreed, value = consensus(record, min_providers)
        if agreed and (type(value) is not type(record["gold"]) or value != record["gold"]):
            errors.append(record["id"])
    n = len(audited)
    rate = len(errors) / n if n else None
    upper = None
    if n:
        z = 1.959964
        centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
        upper = centre + half
    return {"audited": n, "consensus_errors": len(errors), "error_rate": rate, "wilson_upper_95": upper,
            "error_ids": errors[:50]}
