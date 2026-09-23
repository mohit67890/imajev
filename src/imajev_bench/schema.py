"""Strict records and integrity checks for imajev-bench datasets."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from vision_decision.contracts import Request


Identifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
Family = Annotated[StrictStr, Field(min_length=1, max_length=128)]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Gold = StrictBool | StrictInt | StrictStr | None


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ImageRef(_StrictModel):
    path: Annotated[StrictStr, Field(min_length=1, max_length=1024)]
    sha256: Sha256


class BenchmarkRecord(_StrictModel):
    id: Identifier
    group_id: Identifier
    track: Literal["text", "visual", "joint"]
    family: Family
    split: Literal["dev", "calibration", "test"]
    images: Annotated[list[ImageRef], Field(max_length=2)]
    request: Request
    gold: Gold
    annotation_status: Literal["draft", "reviewed"]
    provenance: dict[str, Any]


def _model_input_sha256(record: BenchmarkRecord) -> str:
    payload = {
        "request": record.request.model_dump(mode="json"),
        "images": [image.model_dump(mode="json") for image in record.images],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _target_error(record: BenchmarkRecord, value: object, *, label: str = "gold") -> str | None:
    """Return a useful error when a label is outside the single field's domain."""
    if len(record.request.fields) != 1:
        return "request must contain exactly one decision field"
    if value is None:
        return None

    field = record.request.fields[0]
    if field.type == "boolean":
        if type(value) is not bool:
            return f"{label} must be a strict boolean or null"
    elif field.type == "choice":
        choices = {option.value for option in field.options}
        if type(value) is not str or value not in choices:
            return f"{label} must be one of {sorted(choices)!r} or null"
    elif field.type == "ordinal":
        levels = {level.value for level in field.levels}
        if type(value) is not int or value not in levels:
            return f"{label} must be one of {sorted(levels)!r} or null"
    return None


def _review_error(record: BenchmarkRecord) -> str | None:
    if record.annotation_status != "reviewed":
        return None
    if record.provenance.get("review_flags") or record.provenance.get("quarantined"):
        return "flagged or quarantined records cannot be marked reviewed"
    reviews = record.provenance.get("reviews")
    if record.provenance.get("review_route") == "consensus_verified":
        return _consensus_error(record, reviews)
    if record.provenance.get("review_route") == "construction_verified":
        return _construction_error(record, reviews)
    if not isinstance(reviews, list) or len(reviews) < 2:
        return "reviewed records require provenance.reviews with at least two reviews"

    expected_input_sha256 = _model_input_sha256(record)
    reviewer_ids: list[str] = []
    review_values: list[object] = []
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            return f"provenance.reviews[{index}] must be an object"
        reviewer_id = review.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip() or reviewer_id != reviewer_id.strip():
            return f"provenance.reviews[{index}].reviewer_id must be a trimmed non-empty string"
        evidence = review.get("evidence")
        if "value" not in review or not isinstance(evidence, str) or not evidence.strip():
            return f"provenance.reviews[{index}] requires value and non-empty string evidence"
        if review.get("input_sha256") != expected_input_sha256:
            return f"provenance.reviews[{index}].input_sha256 must match the reviewed model input"
        target_error = _target_error(record, review["value"], label=f"review {index} value")
        if target_error:
            return target_error
        reviewer_ids.append(reviewer_id)
        review_values.append(review["value"])

    if len(set(reviewer_ids)) != len(reviewer_ids):
        return "reviewer_id values must be distinct"
    if len(set((type(value), value) for value in review_values)) == 1:
        if type(review_values[0]) is not type(record.gold) or review_values[0] != record.gold:
            return "agreed review value must equal gold"
        return None

    adjudication = record.provenance.get("adjudication")
    if not isinstance(adjudication, dict):
        return "disagreeing reviews require provenance.adjudication"
    adjudicator = adjudication.get("reviewer_id")
    if (
        not isinstance(adjudicator, str)
        or not adjudicator.strip()
        or adjudicator != adjudicator.strip()
        or adjudicator in reviewer_ids
    ):
        return "adjudication requires a distinct trimmed non-empty reviewer_id"
    evidence = adjudication.get("evidence")
    if "value" not in adjudication or not isinstance(evidence, str) or not evidence.strip():
        return "adjudication requires value and non-empty string evidence"
    if adjudication.get("input_sha256") != expected_input_sha256:
        return "adjudication input_sha256 must match the reviewed model input"
    target_error = _target_error(record, adjudication["value"], label="adjudication value")
    if target_error:
        return target_error
    if type(adjudication["value"]) is not type(record.gold) or adjudication["value"] != record.gold:
        return "adjudication value must equal gold"
    return None


MIN_CONSENSUS_PROVIDERS = 2


def _consensus_error(record: BenchmarkRecord, reviews: object) -> str | None:
    """Triage route: unanimous model pre-labels from distinct providers plus one agreeing blind human.

    Model pre-labels never count as reviews. Audit-sample and judgement-dependent items must
    take the double-human route instead.
    """
    provenance = record.provenance
    if provenance.get("audit_sample") or provenance.get("judgement_dependent"):
        return "audit-sample and judgement-dependent records require two human reviews"
    expected = _model_input_sha256(record)
    prelabels = provenance.get("model_prelabels")
    if not isinstance(prelabels, list) or not prelabels:
        return "consensus route requires provenance.model_prelabels"
    providers, values = set(), set()
    for index, label in enumerate(prelabels):
        if not isinstance(label, dict) or not isinstance(label.get("provider"), str) or "value" not in label:
            return f"provenance.model_prelabels[{index}] requires provider and value"
        if label.get("input_sha256") != expected:
            return f"provenance.model_prelabels[{index}].input_sha256 must match the model input"
        providers.add(label["provider"])
        values.add((type(label["value"]), label["value"]))
    if len(providers) < MIN_CONSENSUS_PROVIDERS or len(values) != 1:
        return f"consensus route requires unanimous pre-labels from at least {MIN_CONSENSUS_PROVIDERS} providers"
    if not isinstance(reviews, list) or len(reviews) != 1 or not isinstance(reviews[0], dict):
        return "consensus route requires exactly one human review"
    review = reviews[0]
    reviewer_id = review.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip() or reviewer_id != reviewer_id.strip():
        return "provenance.reviews[0].reviewer_id must be a trimmed non-empty string"
    if "value" not in review or not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
        return "provenance.reviews[0] requires value and non-empty string evidence"
    if review.get("input_sha256") != expected:
        return "provenance.reviews[0].input_sha256 must match the reviewed model input"
    (consensus_type, consensus), = values
    if type(review["value"]) is not consensus_type or review["value"] != consensus:
        return "the human review must agree with the model consensus"
    if type(record.gold) is not consensus_type or record.gold != consensus:
        return "consensus-verified value must equal gold"
    return _target_error(record, record.gold)


def _construction_error(record: BenchmarkRecord, reviews: object) -> str | None:
    """Answer known by construction: generated from a spec (images checked by two model families) or computed.

    Audit-sample records additionally need one human review that agrees with the constructed answer.
    """
    construction = record.provenance.get("construction")
    if not isinstance(construction, dict) or construction.get("truth_source") not in ("generator_spec", "programmatic"):
        return "construction route requires provenance.construction with a truth_source"
    if "truth" not in construction:
        return "construction route requires provenance.construction.truth"
    truth = construction["truth"]
    if type(truth) is not type(record.gold) or truth != record.gold:
        return "constructed truth must equal gold"
    if record.images and construction.get("image_checks") != "passed":
        return "image records on the construction route require passed image checks"
    if record.provenance.get("judgement_dependent"):
        return "judgement-dependent records cannot use the construction route"
    if record.provenance.get("audit_sample"):
        expected = _model_input_sha256(record)
        if not isinstance(reviews, list) or not reviews:
            return "audit-sample construction records require a human review"
        for review in reviews:
            if not isinstance(review, dict) or review.get("input_sha256") != expected or "value" not in review:
                return "construction audit reviews must bind to the model input and carry a value"
            if type(review["value"]) is not type(record.gold) or review["value"] != record.gold:
                return "construction audit reviews must agree with gold"
    return _target_error(record, record.gold)


def _safe_asset(root: Path, image: ImageRef) -> tuple[Path, str]:
    raw = image.path
    pure = PurePosixPath(raw)
    if "\\" in raw or pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"invalid relative image path {raw!r}")
    try:
        asset = (root / Path(*pure.parts)).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"image asset does not exist: {raw!r}") from exc
    if not asset.is_relative_to(root) or not asset.is_file():
        raise ValueError(f"image asset escapes dataset root or is not a file: {raw!r}")
    digest = hashlib.sha256()
    with asset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != image.sha256:
        raise ValueError(f"sha256 mismatch for image {raw!r}: expected {image.sha256}, got {actual}")
    return asset, actual


def validate_records(
    records: list[dict[str, Any]], root: Path, require_reviewed: bool = False
) -> list[dict[str, Any]]:
    """Validate records plus filesystem and cross-record dataset invariants.

    ``root`` is the dataset root against which every image path is resolved. The
    returned dictionaries are canonical, deep copies produced by Pydantic.
    """
    if not isinstance(root, Path):
        raise ValueError("root must be a pathlib.Path")
    try:
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"dataset root does not exist: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    if not records:
        raise ValueError("records must contain at least one record")

    validated: list[BenchmarkRecord] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    image_splits: dict[tuple[str, str], str] = {}

    for index, raw in enumerate(records):
        try:
            record = BenchmarkRecord.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"record {index} failed schema validation: {exc}") from exc
        if record.id in seen_ids:
            raise ValueError(f"duplicate record id {record.id!r}")
        seen_ids.add(record.id)
        if record.request.request_id != record.id:
            raise ValueError(f"record {record.id!r}: request.request_id must equal record id")
        if len(record.request.fields) != 1:
            raise ValueError(f"record {record.id!r}: request must contain exactly one decision field")
        target_error = _target_error(record, record.gold)
        if target_error:
            raise ValueError(f"record {record.id!r}: {target_error}")

        image_count = len(record.images)
        if record.track == "text" and image_count:
            raise ValueError(f"record {record.id!r}: text track requires zero images")
        if record.track in ("visual", "joint") and image_count == 0:
            raise ValueError(f"record {record.id!r}: {record.track} track requires at least one image")
        if require_reviewed and record.annotation_status != "reviewed":
            raise ValueError(f"record {record.id!r}: reviewed annotation is required")
        review_error = _review_error(record)
        if review_error:
            raise ValueError(f"record {record.id!r}: {review_error}")

        previous_group_split = group_splits.setdefault(record.group_id, record.split)
        if previous_group_split != record.split:
            raise ValueError(f"group {record.group_id!r} leaks across splits")

        for image in record.images:
            _asset, digest = _safe_asset(resolved_root, image)
            for identity in (("path", image.path), ("sha256", digest)):
                previous_image_split = image_splits.setdefault(identity, record.split)
                if previous_image_split != record.split:
                    raise ValueError(f"image {identity[1]!r} leaks across splits")
        validated.append(record)

    return [record.model_dump(mode="json") for record in validated]


def model_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the complete model input and no benchmark-only label metadata."""
    validated = BenchmarkRecord.model_validate(record)
    return copy.deepcopy(
        {
            "request": validated.request.model_dump(mode="json"),
            "images": [image.model_dump(mode="json") for image in validated.images],
        }
    )
