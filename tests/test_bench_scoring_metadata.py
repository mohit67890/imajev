import copy

import pytest

from imajev_bench.runner import run, verify_run
from test_imajev_bench_runner import record


@pytest.mark.parametrize("key,value", [("track", "joint"), ("family", "changed"), ("group_id", "changed")])
def test_replay_rejects_changed_scoring_strata(tmp_path, key, value):
    original = record()
    path = run([original], tmp_path, tmp_path / "run")
    changed = copy.deepcopy(original)
    changed[key] = value
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run([changed], path)


@pytest.mark.parametrize("key,value", [("source_cluster", "scene"), ("source_clusters", ["scene"]),
                                       ("contrast", {"set_id": "s", "role": "original"})])
def test_replay_rejects_changed_evidence_cluster_metadata(tmp_path, key, value):
    original = record()
    path = run([original], tmp_path, tmp_path / "run")
    changed = copy.deepcopy(original)
    changed["provenance"][key] = value
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run([changed], path)


def test_replay_allows_review_bookkeeping_changes(tmp_path):
    original = record()
    path = run([original], tmp_path, tmp_path / "run")
    changed = copy.deepcopy(original)
    changed["provenance"]["notes"] = "Reviewed again without changing inputs or labels"
    assert verify_run([changed], path)["completion"]["status"] == "complete"


@pytest.mark.parametrize("key,value", [("track", "joint"), ("family", "changed"), ("group_id", "changed")])
def test_public_run_binding_rejects_changed_retained_scoring_strata(tmp_path, key, value):
    from test_imajev_bench_public import full, public
    original = full()
    path = run([public(original)], tmp_path, tmp_path / "run")
    changed = copy.deepcopy(original)
    changed[key] = value
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run([changed], path)
