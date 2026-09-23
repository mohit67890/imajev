import base64
import json
import re

import pytest

from imajev_bench.review import PROTOCOL, build_review


def _record(item_id, group="group", state="state"):
    return {"id": item_id, "group_id": group, "track": "text", "family": "policy", "split": "dev",
            "images": [], "request": {"request_id": item_id, "state": state, "fields": [
                {"id": "decision", "type": "boolean", "question": "Question?"}]},
            "gold": True, "annotation_status": "draft", "provenance": {}}


def _embedded(html, name):
    match = re.search(rf"const {name}=JSON\.parse\([^']*'([^']+)'", html)
    assert match
    return json.loads(base64.b64decode(match.group(1)))


def test_packet_is_reproducible_per_reviewer_and_separates_groups(tmp_path):
    records = [_record(f"a{i}", "a") for i in range(3)] + [_record(f"b{i}", "b") for i in range(3)]
    first = build_review(records, tmp_path, tmp_path / "a.html", reviewer_id="reviewer-a", seed=7)
    second = build_review(records, tmp_path, tmp_path / "a2.html", reviewer_id="reviewer-a", seed=7)
    rows = _embedded(first.read_text(), "records")
    assert [r["id"] for r in rows] == [r["id"] for r in _embedded(second.read_text(), "records")]
    assert all(left["id"][0] != right["id"][0] for left, right in zip(rows, rows[1:]))
    assert all("group_id" not in row for row in rows)
    other = build_review(records, tmp_path, tmp_path / "b.html", reviewer_id="reviewer-b", seed=7)
    assert [r["id"] for r in rows] != [r["id"] for r in _embedded(other.read_text(), "records")]


def test_packet_contract_and_content_are_not_executable_source(tmp_path):
    hostile = "</script><script>globalThis.pwned=true</script>"
    path = build_review([_record("opaque-id", state=hostile)], tmp_path, tmp_path / "review.html", reviewer_id="human-1")
    html = path.read_text()
    config = _embedded(html, "config")
    assert config["protocol"] == PROTOCOL
    assert config["format_version"] == "0.0.1"
    assert len(config["dataset_sha256"]) == 64
    assert hostile not in html
    assert "localStorage" in html and "config.dataset_sha256" in html
    assert "flags:Object.values(flags)" in html
    assert "Item ${packetIndex}" in html
    assert "row.id} ·" not in html and "row.group_id" not in html
    assert '<label for="reviewer">Reviewer ID</label>' in html
    assert "Reset local draft" in html


def test_reviewer_id_validation_and_legacy_signature(tmp_path):
    build_review([_record("one")], tmp_path, tmp_path / "legacy.html")
    with pytest.raises(ValueError, match="trimmed"):
        build_review([_record("one")], tmp_path, tmp_path / "bad.html", reviewer_id=" x")
