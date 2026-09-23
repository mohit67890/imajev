import json

import pytest

from imajev_bench.runner import baseline, decode_jev, jev_payload, run
from imajev_bench.review import build_review
from imajev_bench.cli import read_jsonl
from imajev_bench.annotations import merge_reviews
from imajev_bench.runner import digest, verify_run
from imajev_bench.schema import model_payload


def record():
    return {"id": "one", "group_id": "group", "track": "text", "family": "policy", "split": "dev",
            "images": [], "request": {"request_id": "one", "state": "Visible count is 3", "fields": [
                {"id": "decision", "type": "boolean", "question": "Is count greater than 2?"}]},
            "gold": True, "annotation_status": "draft", "provenance": {"secret_gold_evidence": "three"}}


def test_controls_do_not_receive_gold_and_refuse_overwrite(tmp_path):
    out = tmp_path / "run"
    path = run([record()], tmp_path, out)
    prediction = read_jsonl(path)[0]
    assert prediction["status"] == "answered"
    assert prediction["value"] is True
    assert "cost_usd" not in prediction
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["reviewed"] is False
    assert verify_run([record()], path)["completion"]["status"] == "complete"
    changed = record()
    changed["gold"] = False
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run([changed], path)
    path.write_text(path.read_text().replace('"value": true', '"value": false'))
    with pytest.raises(ValueError, match="integrity"):
        verify_run([record()], path)
    with pytest.raises(FileExistsError):
        run([record()], tmp_path, out)


def test_boolean_unknown_mass_reconstruction():
    response = {"answers": {"decision": {"type": "noul", "noul": .6, "unknown_probability": .2, "abstained": False}}}
    result = decode_jev(response, {"request": record()["request"]})
    assert result["probabilities"] == pytest.approx({"true": .5, "false": .3, "__unknown__": .2})
    assert result["value"] is True
    response["answers"]["decision"]["noul"] = 1
    with pytest.raises(ValueError):
        decode_jev(response, {"request": record()["request"]})


def test_ordinal_indices_restore_values_not_expected_score(tmp_path):
    request = {"request_id": "ordinal", "state": {}, "fields": [{"id": "level", "type": "ordinal", "question": "Select level", "levels": [
        {"value": 10, "description": "low"}, {"value": 20, "description": "high"}]}]}
    payload = {"request": request, "images": []}
    wire = jev_payload(payload, tmp_path)
    assert wire["questions"]["level"]["criteria"] == ["low", "high"]
    result = decode_jev({"answers": {"level": {"type": "score", "legend": {"0": "low", "1": "high"}, "probabilities": {"0": .2, "1": .8}, "unknown_probability": .1, "abstained": False, "score": .8}}}, payload)
    assert result["value"] == 20
    assert result["probabilities"] == pytest.approx({"10": .18, "20": .72, "__unknown__": .1})


def test_declared_choice_contradiction_is_error():
    payload = {"request": {"fields": [{"id": "pick", "type": "choice", "options": [{"value": "a"}, {"value": "b"}]}]}}
    response = {"answers": {"pick": {"type": "choice", "choice": "b", "probabilities": {"a": .8, "b": .2}, "unknown_probability": .1, "abstained": False}}}
    with pytest.raises(ValueError, match="conflicts"):
        decode_jev(response, payload)


def test_review_hides_gold_and_generation_metadata(tmp_path):
    item = record()
    item["request"]["state"] = "</script><script>alert('x')</script>"
    path = build_review([item], tmp_path, tmp_path / "review.html")
    html = path.read_text()
    assert '"gold"' not in html
    assert "secret_gold_evidence" not in html
    assert "</script><script>alert" not in html
    assert "\\u003c/script>" in html


def test_jsonl_rejects_nan_and_blank_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    for content in ('{"x":NaN}\n', '\n', '[]\n'):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_jsonl(path)


def export_review(item, reviewer, value):
    return {"format_version": "0.0.1", "purpose": "independent_review", "reviews": [{
        "id": item["id"], "reviewer_id": reviewer, "input_sha256": digest(model_payload(item)),
        "value": value, "evidence": "Count and threshold independently checked."}]}


def test_review_consensus_can_correct_draft_gold(tmp_path):
    item = record()
    result = merge_reviews([item], [export_review(item, "a", False), export_review(item, "b", False)], tmp_path)
    assert result[0]["gold"] is False
    assert result[0]["annotation_status"] == "reviewed"
    assert result[0]["provenance"]["provisional_gold"] is True
    assert item["gold"] is True


def test_disagreement_remains_draft_and_duplicate_review_rejected(tmp_path):
    item = record()
    a, b = export_review(item, "a", True), export_review(item, "b", False)
    result = merge_reviews([item], [a, b], tmp_path)
    assert result[0]["annotation_status"] == "draft"
    assert result[0]["provenance"]["requires_adjudication"] is True
    with pytest.raises(ValueError, match="Duplicate"):
        merge_reviews([item], [a, a], tmp_path)
    a["reviews"][0]["input_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        merge_reviews([item], [a], tmp_path)


def test_distinct_third_adjudicator_resolves_disagreement(tmp_path):
    item = record()
    first = merge_reviews([item], [export_review(item, "a", True), export_review(item, "b", False)], tmp_path)
    third = export_review(item, "c", False)
    third["purpose"] = "adjudication"
    resolved = merge_reviews(first, [third], tmp_path)
    assert resolved[0]["annotation_status"] == "reviewed"
    assert resolved[0]["gold"] is False
    assert "requires_adjudication" not in resolved[0]["provenance"]
