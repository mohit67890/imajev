import pytest

from imajev_bench.schema import validate_records
from test_imajev_bench_v2 import text_record


@pytest.mark.parametrize("first_link,second_link", [
    ({"source_cluster": "scene"}, {"source_cluster": "scene"}),
    ({"source_cluster": "scene"}, {"source_clusters": ["other", "scene"]}),
    ({"contrast": {"set_id": "s", "role": "original"}},
     {"contrast": {"set_id": "s", "role": "variant", "relation": "change"}}),
])
def test_validation_rejects_shared_evidence_across_splits(tmp_path, first_link, second_link):
    first, second = text_record("first"), text_record("second", split="test")
    first["provenance"].update(first_link)
    second["provenance"].update(second_link)
    with pytest.raises(ValueError, match="evidence cluster .* leaks across splits"):
        validate_records([first, second], tmp_path)
    second["split"] = "dev"
    assert len(validate_records([first, second], tmp_path)) == 2


def test_validation_checks_transitively_linked_evidence(tmp_path):
    first, bridge, last = text_record("first"), text_record("bridge"), text_record("last", split="test")
    first["provenance"]["source_cluster"] = "scene-a"
    bridge["provenance"]["source_clusters"] = ["scene-a", "scene-b"]
    last["provenance"]["source_cluster"] = "scene-b"
    with pytest.raises(ValueError, match="evidence cluster .* leaks across splits"):
        validate_records([first, bridge, last], tmp_path)


def test_validation_allows_unrelated_sources_in_different_splits(tmp_path):
    first, second = text_record("first"), text_record("second", split="test")
    first["provenance"]["source_cluster"] = "scene-a"
    second["provenance"]["source_cluster"] = "scene-b"
    assert len(validate_records([first, second], tmp_path)) == 2
