import hashlib
import json

import pytest

from imajev_bench.schema import model_payload, validate_records


def _request(record_id="item-1", field=None):
    return {
        "schema_version": "1.0",
        "request_id": record_id,
        "state": {"rule": "Accept only intact items."},
        "fields": [field or {"id": "accept", "type": "boolean", "question": "Accept?"}],
        "execution": {"mode": "inspect", "allow_external_fallback": False},
    }


def _record(**changes):
    data = {
        "id": "item-1",
        "group_id": "group-1",
        "track": "text",
        "family": "policy",
        "split": "dev",
        "images": [],
        "request": _request(),
        "gold": True,
        "annotation_status": "draft",
        "provenance": {"source": "test"},
    }
    data.update(changes)
    return data


def _image(tmp_path, name="assets/image.bin", content=b"pixels"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def _input_sha256(record):
    payload = model_payload(record)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _review(record, reviewer_id, value, evidence):
    return {
        "reviewer_id": reviewer_id,
        "value": value,
        "evidence": evidence,
        "input_sha256": _input_sha256(record),
    }


def test_validates_and_returns_plain_detached_dicts(tmp_path):
    source = _record()
    result = validate_records([source], tmp_path)
    assert result[0]["id"] == source["id"] and result[0]["gold"] is True
    assert type(result[0]) is dict and type(result[0]["request"]) is dict
    source["request"]["state"]["rule"] = "mutated"
    assert result[0]["request"]["state"]["rule"] == "Accept only intact items."


def test_model_payload_contains_only_model_inputs_and_is_detached():
    source = _record(gold=False, provenance={"secret": "label source"})
    payload = model_payload(source)
    assert set(payload) == {"request", "images"}
    assert "gold" not in repr(payload) and "secret" not in repr(payload)
    payload["request"]["state"]["rule"] = "changed"
    assert source["request"]["state"]["rule"] == "Accept only intact items."


@pytest.mark.parametrize("gold", [1, 0, "true", 1.0])
def test_boolean_gold_is_strict(gold, tmp_path):
    with pytest.raises(ValueError, match="strict boolean|schema validation"):
        validate_records([_record(gold=gold)], tmp_path)


def test_choice_and_ordinal_gold_must_be_in_declared_domain(tmp_path):
    choice = {"id": "kind", "type": "choice", "question": "Kind?", "options": [{"value": "cup"}, {"value": "bowl"}]}
    with pytest.raises(ValueError, match="must be one of"):
        validate_records([_record(request=_request(field=choice), gold="plate")], tmp_path)
    ordinal = {"id": "wear", "type": "ordinal", "question": "Wear?", "levels": [{"value": 1, "description": "low"}, {"value": 3, "description": "high"}]}
    assert validate_records([_record(request=_request(field=ordinal), gold=3)], tmp_path)[0]["gold"] == 3
    with pytest.raises(ValueError, match="must be one of"):
        validate_records([_record(request=_request(field=ordinal), gold=2)], tmp_path)


def test_null_is_the_unknown_gold_for_every_field(tmp_path):
    assert validate_records([_record(gold=None)], tmp_path)[0]["gold"] is None


def test_track_image_requirements_hash_and_containment(tmp_path):
    image = _image(tmp_path)
    visual = _record(track="visual", images=[image])
    assert validate_records([visual], tmp_path)[0]["images"] == [image]
    with pytest.raises(ValueError, match="text track requires zero"):
        validate_records([_record(images=[image])], tmp_path)
    with pytest.raises(ValueError, match="requires at least one"):
        validate_records([_record(track="joint")], tmp_path)
    bad = {**image, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_records([_record(track="visual", images=[bad])], tmp_path)
    with pytest.raises(ValueError, match="invalid relative image path"):
        validate_records([_record(track="visual", images=[{"path": "../outside", "sha256": "0" * 64}])], tmp_path)


def test_rejects_duplicate_ids_and_group_or_image_cross_split_leakage(tmp_path):
    image = _image(tmp_path)
    first = _record(track="visual", images=[image])
    with pytest.raises(ValueError, match="duplicate record id"):
        validate_records([first, first], tmp_path)
    second = _record(id="item-2", request=_request("item-2"), split="test", track="visual", images=[image])
    with pytest.raises(ValueError, match="group .* leaks"):
        validate_records([first, second], tmp_path)
    second["group_id"] = "group-2"
    with pytest.raises(ValueError, match="image .* leaks"):
        validate_records([first, second], tmp_path)


def test_same_image_can_be_reused_within_one_split(tmp_path):
    image = _image(tmp_path)
    records = [
        _record(track="visual", images=[image]),
        _record(id="item-2", request=_request("item-2"), group_id="group-2", track="visual", images=[image]),
    ]
    assert len(validate_records(records, tmp_path)) == 2


def test_request_id_must_match_record_id_and_one_field_is_required(tmp_path):
    with pytest.raises(ValueError, match="request.request_id"):
        validate_records([_record(request=_request("different"))], tmp_path)
    request = _request()
    request["fields"].append({"id": "other", "type": "boolean", "question": "Other?"})
    with pytest.raises(ValueError, match="exactly one"):
        validate_records([_record(request=request)], tmp_path)


def test_reviewed_requires_two_independent_agreeing_reviews(tmp_path):
    reviewed = _record(annotation_status="reviewed")
    reviewed["provenance"] = {"reviews": [
        _review(reviewed, "human-a", True, "Rule applies."),
        _review(reviewed, "human-b", True, "No damage stated."),
    ]}
    assert validate_records([reviewed], tmp_path, require_reviewed=True)[0]["annotation_status"] == "reviewed"
    with pytest.raises(ValueError, match="distinct"):
        duplicated = _record(annotation_status="reviewed")
        duplicated["provenance"] = {"reviews": [
            _review(duplicated, "same", True, "a"),
            _review(duplicated, "same", True, "b"),
        ]}
        validate_records([duplicated], tmp_path)
    with pytest.raises(ValueError, match="reviewed annotation is required"):
        validate_records([_record()], tmp_path, require_reviewed=True)


@pytest.mark.parametrize(
    "review_patch, message",
    [
        ({"reviewer_id": " human-a"}, "trimmed"),
        ({"evidence": "   "}, "string evidence"),
        ({"evidence": ["not", "text"]}, "string evidence"),
    ],
)
def test_review_identity_and_evidence_are_strict(review_patch, message, tmp_path):
    record = _record(annotation_status="reviewed")
    reviews = [
        _review(record, "human-a", True, "Rule applies."),
        _review(record, "human-b", True, "Evidence applies."),
    ]
    reviews[0].update(review_patch)
    record["provenance"] = {"reviews": reviews}
    with pytest.raises(ValueError, match=message):
        validate_records([record], tmp_path)


def test_review_disagreement_requires_distinct_adjudicator_resolving_gold(tmp_path):
    disputed = _record(annotation_status="reviewed")
    reviews = [
        _review(disputed, "human-a", True, "visible evidence"),
        _review(disputed, "human-b", False, "policy interpretation"),
    ]
    disputed["provenance"] = {"reviews": reviews}
    with pytest.raises(ValueError, match="adjudication"):
        validate_records([disputed], tmp_path)
    disputed["provenance"]["adjudication"] = _review(
        disputed, "human-c", True, "Clause explicitly permits it."
    )
    assert validate_records([disputed], tmp_path)[0]["gold"] is True


def test_review_input_hash_must_match_current_model_input(tmp_path):
    reviewed = _record(annotation_status="reviewed")
    reviewed["provenance"] = {"reviews": [
        _review(reviewed, "human-a", True, "Rule applies."),
        _review(reviewed, "human-b", True, "No damage stated."),
    ]}
    reviewed["request"]["state"]["rule"] = "Accept every item."
    with pytest.raises(ValueError, match="input_sha256 must match"):
        validate_records([reviewed], tmp_path)


def test_extra_fields_and_non_dict_provenance_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Extra inputs"):
        validate_records([{**_record(), "answer": True}], tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        validate_records([_record(provenance=[])], tmp_path)


def test_empty_dataset_is_rejected_even_when_review_is_required(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        validate_records([], tmp_path, require_reviewed=True)
