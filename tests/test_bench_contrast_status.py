import pytest

from imajev_bench.stats import contrast_metrics


def records():
    return [
        {"id": "original", "gold": None, "provenance": {"contrast": {"set_id": "s", "role": "original"}}},
        {"id": "variant", "gold": None, "provenance": {"contrast": {"set_id": "s", "role": "variant", "relation": "same"}}},
    ]


@pytest.mark.parametrize("prediction", [None, {"status": "error", "value": None},
                                       {"status": "abstained"}, {"status": "answered", "value": None},
                                       {"status": "abstained", "value": True}])
def test_unknown_contrast_labels_require_explicit_valid_abstention(prediction):
    predictions = {} if prediction is None else {"original": prediction, "variant": prediction}
    report = contrast_metrics(records(), predictions)
    assert report["all_correct"] == 0
    assert report["same_pairs_correct"] == 0
    assert report["same_unchanged"] == 0
    assert report["constant_sets"] == 0


def test_unknown_contrast_labels_accept_explicit_abstention_and_raw_values():
    for prediction in ({"status": "abstained", "value": None}, None):
        report = contrast_metrics(records(), {"original": prediction, "variant": prediction})
        assert report["all_correct"] == 1
        assert report["same_pairs_correct"] == 1
