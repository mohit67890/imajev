import pytest

from imajev_bench.scoring import score
from test_imajev_bench_public import full, public


def test_scorer_refuses_withheld_labels_instead_of_treating_them_as_unknown():
    record = public(full())
    with pytest.raises(ValueError, match="gold withheld"):
        score([record], {record["id"]: {"status": "abstained", "value": None}}, bootstrap_samples=0)


def test_scorer_refuses_mixed_withheld_and_full_labels():
    records = [full("available", gold=None), public(full("withheld"))]
    with pytest.raises(ValueError, match="gold withheld"):
        score(records, {}, bootstrap_samples=0)


def test_scorer_accepts_real_unknown_labels_without_withheld_flag():
    record = full(gold=None)
    report = score([record], {record["id"]: {"status": "abstained", "value": None}}, bootstrap_samples=0)
    assert report["capability"]["all_records"]["correct"] == 1
