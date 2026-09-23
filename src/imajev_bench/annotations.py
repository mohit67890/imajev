"""Import explicit human judgments; consensus replaces provisional symbolic labels."""
import copy
import hashlib
import json
from .runner import digest
from .schema import MIN_CONSENSUS_PROVIDERS, model_payload, validate_records


def merge_reviews(records, exports, root):
    result = copy.deepcopy(records)
    by_id = {r["id"]: r for r in result}
    seen = set()
    for export in exports:
        if export.get("annotator_type") == "model":
            raise ValueError("Model annotations must remain separate from human-review promotion")
        if export.get("format_version") != "0.0.1" or export.get("purpose") not in ("independent_review", "adjudication"):
            raise ValueError("Unrecognized review export")
        if "protocol" in export:
            if export["protocol"] != "imajev-bench-blind-review-v1":
                raise ValueError("Unknown independent-review protocol")
            hashes = sorted((r["id"], digest(model_payload(r))) for r in records)
            expected = hashlib.sha256(json.dumps(hashes, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            # Second-reviewer packets cover only the records a triage plan routed to double review.
            double = sorted((r["id"], digest(model_payload(r))) for r in records
                            if r["provenance"].get("review_plan") == "double")
            subset = hashlib.sha256(json.dumps(double, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            audit = sorted((r["id"], digest(model_payload(r))) for r in records if r["provenance"].get("audit_sample"))
            audit_hash = hashlib.sha256(json.dumps(audit, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            allowed = {expected} | ({subset} if double else set()) | ({audit_hash} if audit else set())
            if export.get("input_sha256") not in allowed:
                raise ValueError("Review packet does not match this dataset input hash")
            reviewer = export.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip():
                raise ValueError("Review packet requires one trimmed reviewer ID")
            if any(r.get("reviewer_id") != reviewer for r in export.get("reviews", []) + export.get("flags", [])):
                raise ValueError("Packet judgments must belong to the declared reviewer")
        for review in export.get("reviews", []):
            record = by_id.get(review.get("id"))
            if record is None:
                raise ValueError("Review refers to unknown record")
            reviewer = review.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip():
                raise ValueError("Reviewer ID must be nonempty without surrounding whitespace")
            if review.get("input_sha256") != digest(model_payload(record)):
                raise ValueError("Review input hash does not match record")
            if "value" not in review or not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
                raise ValueError("Review requires value and evidence")
            key = (record["id"], reviewer)
            existing = record["provenance"].setdefault("reviews", [])
            if key in seen or any(r["reviewer_id"] == reviewer for r in existing):
                raise ValueError("Duplicate reviewer judgment; resolve revisions explicitly")
            seen.add(key)
            judgment = {k: review[k] for k in ("reviewer_id", "value", "evidence", "input_sha256")}
            if export["purpose"] == "adjudication":
                if len(existing) < 2 or "adjudication" in record["provenance"]:
                    raise ValueError("Adjudication requires two prior reviews and no existing adjudication")
                if len({(type(r["value"]), r["value"]) for r in existing}) < 2:
                    raise ValueError("Adjudication requires a reviewer disagreement")
                record["provenance"]["adjudication"] = judgment
            else:
                if "adjudication" in record["provenance"]:
                    raise ValueError("Cannot add reviews to an adjudicated record; create an explicit revision")
                existing.append(judgment)
        for flag in export.get("flags", []):
            record = by_id.get(flag.get("id"))
            if record is None:
                raise ValueError("Flag refers to unknown record")
            reviewer = flag.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip():
                raise ValueError("Flag requires a trimmed reviewer ID")
            if flag.get("input_sha256") != digest(model_payload(record)):
                raise ValueError("Flag input hash does not match record")
            if not isinstance(flag.get("reason"), str) or not flag["reason"].strip():
                raise ValueError("Flag requires a nonblank reason")
            flags = record["provenance"].setdefault("review_flags", [])
            if any(f["reviewer_id"] == reviewer for f in flags):
                raise ValueError("Duplicate reviewer flag; resolve revisions explicitly")
            flags.append({k: flag[k] for k in ("reviewer_id", "input_sha256", "reason")})
    quarantined_groups = {r["group_id"] for r in result if r["provenance"].get("review_flags")}
    for record in result:
        reviews = record["provenance"].get("reviews", [])
        # Validate every individual imported decision against the existing typed domain.
        for review in reviews:
            check = copy.deepcopy(record)
            check["gold"] = review["value"]
            check["annotation_status"] = "draft"
            validate_records([check], root)
        if record["group_id"] in quarantined_groups:
            record["annotation_status"] = "draft"
            record["provenance"]["quarantined"] = True
            record["provenance"]["quarantine_reason"] = "An independent reviewer flagged this group; explicit dataset revision required."
            continue
        adjudication = record["provenance"].get("adjudication")
        if adjudication:
            record["provenance"].setdefault("provisional_gold", record["gold"])
            record["gold"] = adjudication["value"]
            record["annotation_status"] = "reviewed"
            record["provenance"].pop("requires_adjudication", None)
        elif (len(reviews) == 1 and record["provenance"].get("review_plan") == "single"
              and isinstance(record["provenance"].get("construction"), dict)):
            truth = record["provenance"]["construction"].get("truth")
            human = reviews[0]["value"]
            if type(human) is type(truth) and human == truth:
                record["annotation_status"] = "reviewed"
                record["provenance"]["review_route"] = "construction_verified"
                record["provenance"].pop("requires_second_review", None)
            else:
                record["annotation_status"] = "draft"
                record["provenance"]["review_plan"] = "double"
                record["provenance"]["requires_second_review"] = True
        elif len(reviews) == 1 and record["provenance"].get("review_plan") == "single":
            from .triage import consensus
            agreed, value = consensus(record, MIN_CONSENSUS_PROVIDERS)
            human = reviews[0]["value"]
            if agreed and type(human) is type(value) and human == value:
                record["provenance"].setdefault("provisional_gold", record["gold"])
                record["gold"] = value
                record["annotation_status"] = "reviewed"
                record["provenance"]["review_route"] = "consensus_verified"
                record["provenance"].pop("requires_second_review", None)
            else:
                # Human disagrees with the models (or consensus changed): escalate to double review.
                record["annotation_status"] = "draft"
                record["provenance"]["review_plan"] = "double"
                record["provenance"]["requires_second_review"] = True
        elif len(reviews) >= 2:
            value = reviews[0]["value"]
            if all(type(r["value"]) is type(value) and r["value"] == value for r in reviews):
                record["provenance"].setdefault("provisional_gold", record["gold"])
                record["gold"] = value
                record["annotation_status"] = "reviewed"
                record["provenance"]["review_route"] = "double_human"
                record["provenance"].pop("requires_second_review", None)
            else:
                record["annotation_status"] = "draft"
                record["provenance"]["requires_adjudication"] = True
    return validate_records(result, root)


def import_prelabels(records, export):
    """Attach a model pre-label export (see api_models.prelabel_export) without touching reviews or gold."""
    if export.get("purpose") != "model_prelabel" or export.get("annotator_type") != "model":
        raise ValueError("Not a model pre-label export")
    provider, model = export.get("provider"), export.get("model")
    if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
        raise ValueError("Pre-label export requires provider and model")
    result = copy.deepcopy(records)
    by_id = {r["id"]: r for r in result}
    for label in export.get("labels", []):
        record = by_id.get(label.get("id"))
        if record is None:
            raise ValueError("Pre-label refers to unknown record")
        if label.get("input_sha256") != digest(model_payload(record)):
            raise ValueError(f"Pre-label input hash does not match record {record['id']!r}")
        existing = record["provenance"].setdefault("model_prelabels", [])
        if any(x["provider"] == provider for x in existing):
            raise ValueError(f"Record {record['id']!r} already has a pre-label from provider {provider!r}")
        existing.append({"provider": provider, "model": model, "value": label["value"],
                         "evidence": label.get("evidence"), "input_sha256": label["input_sha256"],
                         "run_manifest_sha256": export.get("run_manifest_sha256")})
    return result
