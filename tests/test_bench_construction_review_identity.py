import copy

import pytest

from imajev_bench.runner import digest
from imajev_bench.schema import model_payload, validate_records
from test_imajev_bench_runner import record


def audited_record():
    item = record()
    item["annotation_status"] = "reviewed"
    item["provenance"] = {"review_route": "construction_verified", "audit_sample": True,
                          "construction": {"truth_source": "programmatic", "truth": True},
                          "reviews": [{"reviewer_id": "human", "value": True, "evidence": "Count independently checked.",
                                       "input_sha256": digest(model_payload(item))}]}
    return item


@pytest.mark.parametrize("key,value", [("reviewer_id", None), ("reviewer_id", ""),
                                       ("reviewer_id", " human"), ("evidence", None),
                                       ("evidence", "  "), ("evidence", ["not text"])])
def test_construction_audit_requires_identified_human_and_evidence(tmp_path, key, value):
    item = audited_record()
    item["provenance"]["reviews"][0][key] = value
    with pytest.raises(ValueError, match="reviewer_id|evidence"):
        validate_records([item], tmp_path, require_reviewed=True)


def test_construction_audit_rejects_duplicate_human_reviews(tmp_path):
    item = audited_record()
    item["provenance"]["reviews"].append(copy.deepcopy(item["provenance"]["reviews"][0]))
    with pytest.raises(ValueError, match="distinct"):
        validate_records([item], tmp_path, require_reviewed=True)


def test_construction_audit_accepts_identified_human_review(tmp_path):
    assert validate_records([audited_record()], tmp_path, require_reviewed=True)[0]["gold"] is True
