"""Deterministic, dependency-free scoring for imajev-bench submissions."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping

UNKNOWN = "__unknown__"
TRACKS = ("text", "visual", "joint")
NLL_FLOOR = 1e-15
ECE_BINS = 10


def candidate_keys(record: Mapping[str, Any]) -> list[str]:
    """Return the canonical probability keys for a single-field record."""
    request = record.get("request")
    if hasattr(request, "model_dump"):
        request = request.model_dump()
    if not isinstance(request, Mapping):
        raise ValueError("record request must be a mapping or typed Request")
    fields = request.get("fields")
    if not isinstance(fields, list) or len(fields) != 1 or not isinstance(fields[0], Mapping):
        raise ValueError("benchmark records must contain exactly one decision field")
    field = fields[0]
    kind = field.get("type")
    if kind == "boolean":
        keys = ["true", "false"]
    elif kind == "choice":
        options = field.get("options")
        if not isinstance(options, list):
            raise ValueError("choice field has no options")
        keys = [str(option["value"]) for option in options]
    elif kind == "ordinal":
        levels = field.get("levels")
        if not isinstance(levels, list):
            raise ValueError("ordinal field has no levels")
        keys = [str(level["value"]) for level in levels]
    else:
        raise ValueError(f"unsupported decision field type: {kind!r}")
    if UNKNOWN in keys or len(keys) != len(set(keys)):
        raise ValueError("candidate keys must be unique and may not use __unknown__")
    return keys + [UNKNOWN]


def _gold_key(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _class_identity(value: Any) -> tuple[str, str]:
    """Keep equal-looking values from different typed contracts distinct."""
    if value is None:
        return ("unknown", UNKNOWN)
    if type(value) is bool:
        return ("bool", "true" if value else "false")
    if type(value) is int:
        return ("int", str(value))
    if type(value) is str:
        return ("str", value)
    return (type(value).__name__, repr(value))


def _typed_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _correct(record: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> bool:
    if not isinstance(prediction, Mapping):
        return False
    gold = record.get("gold")
    status = prediction.get("status")
    if gold is None:
        return status == "abstained" and "value" in prediction and prediction["value"] is None
    return status == "answered" and "value" in prediction and _typed_equal(prediction["value"], gold)


def _value_in_domain(record: Mapping[str, Any], value: Any) -> bool:
    request = record.get("request")
    if hasattr(request, "model_dump"):
        request = request.model_dump()
    if not isinstance(request, Mapping) or not isinstance(request.get("fields"), list) or len(request["fields"]) != 1:
        return False
    field = request["fields"][0]
    kind = field.get("type")
    if kind == "boolean":
        return type(value) is bool
    if kind == "ordinal":
        return type(value) is int and any(type(level.get("value")) is int and level["value"] == value
                                          for level in field.get("levels", []))
    if kind == "choice":
        return type(value) is str and any(type(option.get("value")) is str and option["value"] == value
                                          for option in field.get("options", []))
    return False


def _prediction_malformed(record: Mapping[str, Any], prediction: Any) -> bool:
    if not isinstance(prediction, Mapping):
        return True
    status = prediction.get("status")
    if status == "error":
        return False
    if status == "answered":
        return "value" not in prediction or not _value_in_domain(record, prediction["value"])
    if status == "abstained":
        return "value" not in prediction or prediction["value"] is not None
    return True


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    labels = sorted({_class_identity(row["record"].get("gold")) for row in rows})
    recalls, f1s = [], []
    for label in labels:
        tp = sum(row["correct"] and _class_identity(row["record"].get("gold")) == label for row in rows)
        fn = sum((not row["correct"]) and _class_identity(row["record"].get("gold")) == label for row in rows)
        fp = sum(
            row["prediction"] is not None
            and _prediction_class(row["prediction"]) == label
            and _class_identity(row["record"].get("gold")) != label
            for row in rows
        )
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "class_balanced_accuracy": _mean(recalls),
        "macro_f1": _mean(f1s),
        "classes_present": [f"{kind}:{value}" for kind, value in labels],
    }


def _prediction_key(prediction: Mapping[str, Any]) -> str | None:
    if not isinstance(prediction, Mapping):
        return None
    if prediction.get("status") == "abstained":
        return UNKNOWN
    if prediction.get("status") == "answered":
        return _gold_key(prediction.get("value"))
    return None


def _prediction_class(prediction: Mapping[str, Any]) -> tuple[str, str] | None:
    if not isinstance(prediction, Mapping):
        return None
    if prediction.get("status") == "abstained" and prediction.get("value") is None:
        return _class_identity(None)
    if prediction.get("status") == "answered":
        return _class_identity(prediction.get("value"))
    return None


def _valid_vector(record: Mapping[str, Any], prediction: Mapping[str, Any] | None) -> tuple[str, dict[str, float] | None]:
    if not isinstance(prediction, Mapping) or "probabilities" not in prediction or prediction.get("probabilities") is None:
        return "missing", None
    if _prediction_malformed(record, prediction) or prediction.get("status") == "error":
        return "invalid", None
    probabilities = prediction.get("probabilities")
    expected = candidate_keys(record)
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(expected):
        return "invalid", None
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        return "invalid", None
    vector = {key: float(probabilities[key]) for key in expected}
    if not math.isclose(sum(vector.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        return "invalid", None
    return "valid", vector


def _probability_metrics(rows: list[dict[str, Any]], breakdowns: bool = True) -> dict[str, Any]:
    valid, missing, invalid = [], 0, 0
    for row in rows:
        state, vector = _valid_vector(row["record"], row["prediction"])
        if state == "missing":
            missing += 1
        elif state == "invalid":
            invalid += 1
        else:
            valid.append((row, vector))
    briers, nlls, bins = [], [], [[] for _ in range(ECE_BINS)]
    by_option_count: dict[int, list[float]] = defaultdict(list)
    for row, vector in valid:
        gold = _gold_key(row["record"].get("gold"))
        brier = sum((probability - (key == gold)) ** 2 for key, probability in vector.items())
        briers.append(brier)
        nlls.append(-math.log(max(vector[gold], NLL_FLOOR)))
        confidence = max(vector.values())
        predicted = max(vector, key=lambda key: vector[key])
        index = min(int(confidence * ECE_BINS), ECE_BINS - 1)
        bins[index].append((confidence, predicted == gold))
        by_option_count[len(vector)].append(brier)
    bin_report, ece = [], 0.0
    for index, entries in enumerate(bins):
        confidence = _mean(item[0] for item in entries)
        accuracy = _mean(float(item[1]) for item in entries)
        count = len(entries)
        if count:
            ece += count / len(valid) * abs(accuracy - confidence)
        bin_report.append({"lower": index / ECE_BINS, "upper": (index + 1) / ECE_BINS,
                           "count": count, "accuracy": accuracy, "mean_confidence": confidence})
    report = {
        "eligible_for_complete_reliability": bool(rows) and len(valid) == len(rows),
        "total": len(rows), "valid": len(valid), "missing": missing, "invalid": invalid,
        "coverage": len(valid) / len(rows) if rows else None,
        "brier": _mean(briers), "nll": _mean(nlls), "nll_floor": NLL_FLOOR,
        "ece": ece if valid else None,
        "ece_definition": "Fixed 10-bin ECE using the probability-vector argmax as the predicted class and its maximum probability as confidence; independent of the submitted answer/abstention policy.",
        "distribution_argmax_ece": ece if valid else None, "ece_bins": bin_report,
        "by_option_count": {str(key): {"count": len(values), "brier": _mean(values)}
                            for key, values in sorted(by_option_count.items())},
    }
    if breakdowns:
        tracks: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            tracks[str(row["record"].get("track"))][str(row["record"].get("family"))].append(row)
        report["tracks"] = {}
        for track, families in sorted(tracks.items()):
            family_reports = {name: _probability_metrics(items, False) for name, items in sorted(families.items())}
            report["tracks"][track] = {
                "families": family_reports,
                "family_macro_brier": _mean(item["brier"] for item in family_reports.values() if item["brier"] is not None),
                "family_macro_nll": _mean(item["nll"] for item in family_reports.values() if item["nll"] is not None),
                "family_macro_ece": _mean(item["ece"] for item in family_reports.values() if item["ece"] is not None),
                "complete_family_coverage": all(item["eligible_for_complete_reliability"] for item in family_reports.values()),
            }
    return report


def _selective(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["record"].get("gold") is not None]
    accepted = []
    for row in answerable:
        prediction = row["prediction"]
        if not isinstance(prediction, Mapping) or prediction.get("status") != "answered" or _prediction_malformed(row["record"], prediction):
            continue
        state, vector = _valid_vector(row["record"], prediction)
        selected_key = _prediction_key(prediction)
        confidence = vector[selected_key] if state == "valid" and selected_key in vector else None
        accepted.append((confidence, row["correct"]))
    ranked = sorted((item for item in accepted if item[0] is not None), key=lambda item: item[0], reverse=True)
    curve = [{"answered": 0, "coverage": 0.0, "risk": None}]
    errors = index = 0
    partial_aurc = 0.0
    while index < len(ranked):
        confidence = ranked[index][0]
        end = index
        while end < len(ranked) and ranked[end][0] == confidence:
            errors += not ranked[end][1]
            end += 1
        risk = errors / end
        coverage = end / len(answerable) if answerable else 0.0
        partial_aurc += risk * (end - index) / len(answerable) if answerable else 0.0
        curve.append({"answered": end, "coverage": coverage, "risk": risk, "threshold": confidence})
        index = end
    operational_errors = sum(
        row["record"].get("gold") is None
        and isinstance(row["prediction"], Mapping)
        and row["prediction"].get("status") == "answered"
        for row in rows
    )
    return {
        "convention": "Only status=answered is accepted. Confidence is the probability assigned to the selected answer. Equal-confidence answers enter as one threshold group. The denominator is all answerable items; abstentions, errors, and missing outputs reduce coverage. Full AURC is reported only at complete confidence-ranked coverage; partial_AURC is the observed area through achieved ranked coverage.",
        "answerable_total": len(answerable), "answered": len(accepted),
        "confidence_ranked_answered": len(ranked),
        "achieved_coverage": len(accepted) / len(answerable) if answerable else None,
        "achieved_error": (sum(not item[1] for item in accepted) / len(accepted)) if accepted else None,
        "partial_aurc": partial_aurc if answerable else None,
        "aurc": partial_aurc if answerable and len(ranked) == len(answerable) else None, "curve": curve,
        "unknown_gold_total": len(rows) - len(answerable),
        "substantive_answers_on_unknown_gold": operational_errors,
    }


def _bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["record"].get("group_id"))].append(row)
    group_rows = list(groups.values())
    if not group_rows or samples <= 0:
        return {"method": "group bootstrap", "samples": samples, "low": None, "high": None}
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        draw = [rng.choice(group_rows) for _ in group_rows]
        flat = [row for group in draw for row in group]
        estimates.append(sum(row["correct"] for row in flat) / len(flat))
    return {"method": "percentile group bootstrap", "level": 0.95, "samples": samples,
            "low": _quantile(estimates, 0.025), "high": _quantile(estimates, 0.975)}


def score(records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]],
          bootstrap_samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Score records and predictions and return a JSON-serializable report."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must map record IDs to prediction objects")
    ids = [record.get("id") for record in records]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("record IDs must be unique strings")
    extra_ids = set(predictions) - set(ids)
    if extra_ids:
        raise ValueError(f"predictions contain unknown record IDs: {sorted(extra_ids)!r}")
    rows = [{"record": record, "prediction": predictions.get(record["id"]),
             "correct": _correct(record, predictions.get(record["id"]))} for record in records]

    families: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[str(row["record"].get("track"))][str(row["record"].get("family"))].append(row)
    track_report = {}
    for track, family_map in sorted(families.items()):
        family_report = {family: _classification(items) for family, items in sorted(family_map.items())}
        track_report[track] = {
            "families": family_report,
            "family_macro_accuracy": _mean(item["accuracy"] for item in family_report.values()),
            "score": 100 * _mean(item["accuracy"] for item in family_report.values()),
            "raw": _classification([row for items in family_map.values() for row in items]),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["record"].get("group_id"))].append(row)
    groups_correct = sum(all(row["correct"] for row in items) for items in grouped.values())
    known_predictions = [prediction for prediction in (predictions.get(item) for item in ids)
                         if isinstance(prediction, Mapping)]
    latencies = [float(prediction["latency_ms"]) for prediction in known_predictions
                 if type(prediction.get("latency_ms")) in (int, float) and math.isfinite(prediction["latency_ms"])
                 and prediction["latency_ms"] >= 0]
    costs = [float(prediction["cost_usd"]) for prediction in known_predictions
             if type(prediction.get("cost_usd")) in (int, float) and math.isfinite(prediction["cost_usd"])
             and prediction["cost_usd"] >= 0]
    all_tracks = set(track_report) == set(TRACKS)
    return {
        "capability": {
            "overall_score": _mean(track_report[track]["score"] for track in TRACKS) if all_tracks else None,
            "overall_requires_complete_tracks": list(TRACKS), "tracks": track_report,
            "all_records": _classification(rows),
        },
        "groups": {"correct": groups_correct, "total": len(grouped),
                   "accuracy": groups_correct / len(grouped) if grouped else None},
        "accuracy_ci": {"estimand": "raw request-level accuracy, resampling whole groups",
                        **_bootstrap(rows, bootstrap_samples, seed)},
        "probability_quality": _probability_metrics(rows),
        "selective": _selective(rows),
        "efficiency": {
            "requested": len(records),
            "latency_ms": {"observed": len(latencies), "missing": len(records) - len(latencies),
                           "p50": _quantile(latencies, 0.5), "p95": _quantile(latencies, 0.95)},
            "cost_usd": {"observed": len(costs), "missing": len(records) - len(costs),
                         "total_observed": sum(costs) if costs else None,
                         "per_1000_requests": (sum(costs) * 1000 / len(records)) if costs and len(costs) == len(records) else None},
        },
        "submission": {"missing_predictions": sum(predictions.get(item) is None for item in ids),
                       "error_predictions": sum(isinstance(predictions.get(item), Mapping)
                                                and predictions[item].get("status") == "error" for item in ids),
                       "malformed_predictions": sum(predictions.get(item) is not None
                                                    and _prediction_malformed(row["record"], predictions.get(item))
                                                    for item, row in zip(ids, rows)),
                       "extra_prediction_ids": []},
    }
