import pytest

from imajev_bench.runner import decode_jev


def payload():
    return {"request": {"fields": [{"id": "pick", "type": "choice", "options": [{"value": "a"}, {"value": "b"}]}]}}


def response(probabilities, unknown=1.0):
    return {"answers": {"pick": {"type": "choice", "choice": "a", "probabilities": probabilities,
                                  "unknown_probability": unknown, "abstained": True}}}


@pytest.mark.parametrize("probabilities", [{"a": -2.0, "b": 3.0}, {"a": .2, "b": .2},
                                           {"a": 0.0, "b": 0.0}, {"a": 5.0, "b": 0.0}])
def test_unknown_mass_does_not_hide_invalid_conditional_probabilities(probabilities):
    with pytest.raises(ValueError, match="probability distribution"):
        decode_jev(response(probabilities), payload())


def test_all_unknown_accepts_valid_conditional_distribution():
    result = decode_jev(response({"a": .8, "b": .2}), payload())
    assert result["status"] == "abstained"
    assert result["probabilities"] == {"a": 0.0, "b": 0.0, "__unknown__": 1.0}
