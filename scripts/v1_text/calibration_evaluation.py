"""Raw/calibrated metrics, v1.1 acceptance gates, and paired family bootstrap."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from vision_decision.calibration import TemperatureCalibrator, softmax
from vision_decision.contracts import UNKNOWN


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    if len(confidences) != len(correct) or bins < 1:
        raise ValueError("ECE inputs must have equal length and at least one bin")
    if not confidences:
        return math.nan
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(confidences)
                   if low <= value <= high and (index == bins - 1 or value < high)]
        if members:
            confidence = sum(confidences[i] for i in members) / len(members)
            accuracy = sum(correct[i] for i in members) / len(members)
            error += len(members) / len(confidences) * abs(confidence - accuracy)
    return error


def _row_outcome(row: Mapping[str, Any], probabilities: Sequence[float], logits: Sequence[float]) -> dict[str, Any]:
    labels = row.get("labels")
    if labels is None:
        labels = [str(index) for index in range(len(probabilities))]
    if len(labels) != len(probabilities) or UNKNOWN not in labels or len(set(labels)) != len(labels):
        raise ValueError("labels must align with logits and include __unknown__")
    target = row["target_index"]
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(labels):
        raise ValueError("Invalid target_index")
    predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
    kind = {"noul": "boolean", "score": "ordinal"}.get(row["decision_type"], row["decision_type"])
    score_error = None
    expected_score = None
    if kind == "ordinal" and labels[target] != UNKNOWN:
        known_indices = [index for index, label in enumerate(labels) if label != UNKNOWN]
        conditional = softmax([logits[index] for index in known_indices])
        known = [(float(labels[index]), probability) for index, probability in zip(known_indices, conditional)]
        # Match the score API: expectation conditional on known levels. This is measured even when
        # unknown wins, so abstention cannot make difficult score cases disappear from MAE.
        expected_score = sum(level * probability for level, probability in known)
        score_error = abs(expected_score - float(labels[target]))
    return {"correct": predicted == target, "confidence": probabilities[predicted],
            "predicted_unknown": labels[predicted] == UNKNOWN, "score_error": score_error,
            "expected_score": expected_score,
            "nll": -math.log(max(probabilities[target], 1e-300))}


def metrics(rows: Iterable[Mapping[str, Any]], calibrator: TemperatureCalibrator | None = None,
            *, auto_accept_threshold: float = 0.95) -> dict[str, Any]:
    rows = list(rows)
    if any(row.get("partition") != "test" for row in rows):
        raise ValueError("Release evaluation accepts only partition='test' rows")
    if any(row.get("heldout_family") is not True for row in rows):
        raise ValueError("Release evaluation requires heldout_family=true on every row")
    outcomes = []
    calibrated_rows, missing_buckets = 0, set()
    for row in rows:
        logits = [float(value) for value in row["logits"]]
        option_count = row["option_count"]
        if isinstance(option_count, bool) or not isinstance(option_count, int) or len(logits) != option_count + 1:
            raise ValueError("option_count must be an integer equal to len(logits) - 1")
        temperature = None if calibrator is None else calibrator.temperature(str(row["decision_type"]), option_count)
        if calibrator is not None:
            if temperature is None:
                from vision_decision.calibration import calibration_key
                missing_buckets.add(calibration_key(str(row["decision_type"]), option_count))
            else:
                calibrated_rows += 1
        outcomes.append((_row_outcome(row, softmax(logits, temperature or 1.0), logits), row))
    if not outcomes:
        raise ValueError("Cannot evaluate an empty decision set")

    def summary(selected: list[tuple[dict[str, Any], Mapping[str, Any]]]) -> dict[str, Any] | None:
        if not selected:
            return None
        values = [outcome for outcome, _ in selected]
        accepted = [value for value in values
                    if not value["predicted_unknown"] and value["confidence"] >= auto_accept_threshold]
        errors = [value["score_error"] for value in values if value["score_error"] is not None]
        return {
            "n": len(values),
            "accuracy": sum(value["correct"] for value in values) / len(values),
            "nll": sum(value["nll"] for value in values) / len(values),
            "ece": expected_calibration_error([value["confidence"] for value in values],
                                                [value["correct"] for value in values]),
            "score_mae": sum(errors) / len(errors) if errors else None,
            "auto_accept_coverage": len(accepted) / len(values),
            "auto_accept_precision": (sum(value["correct"] for value in accepted) / len(accepted)
                                      if accepted else None),
        }

    by_type = {}
    for kind in ("choice", "boolean", "ordinal"):
        aliases = {kind, {"boolean": "noul", "ordinal": "score"}.get(kind, kind)}
        by_type[kind] = summary([(value, row) for value, row in outcomes if row["decision_type"] in aliases])
    result = {"overall": summary(outcomes), "by_type": by_type}
    if calibrator is not None:
        result["calibration"] = {"rows": len(rows), "calibrated_rows": calibrated_rows,
                                 "coverage": calibrated_rows / len(rows),
                                 "missing_buckets": sorted(missing_buckets)}
    return result


def acceptance_gates(calibrated: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the text gates from docs/v1.1-text-layer-plan.md."""
    types = calibrated["by_type"]
    classification = [types[kind] for kind in ("choice", "boolean") if types.get(kind)]
    classification_n = sum(item["n"] for item in classification)
    classification_accuracy = (sum(item["accuracy"] * item["n"] for item in classification) / classification_n
                               if classification_n else None)
    score_mae = types.get("ordinal", {}).get("score_mae") if types.get("ordinal") else None
    overall = calibrated["overall"]
    checks = {
        "calibration_bucket_coverage": calibrated.get("calibration", {}).get("coverage") == 1.0,
        "choice_noul_accuracy": classification_accuracy is not None and classification_accuracy >= 0.75,
        "score_mae": score_mae is not None and score_mae <= 0.6,
        "ece": overall["ece"] <= 0.06,
        "auto_accept_coverage": overall["auto_accept_coverage"] >= 0.40,
        "auto_accept_precision": (overall["auto_accept_precision"] is not None
                                  and overall["auto_accept_precision"] >= 0.95),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "observed": {"choice_noul_accuracy": classification_accuracy, "score_mae": score_mae,
                         "ece": overall["ece"], "auto_accept_coverage": overall["auto_accept_coverage"],
                         "auto_accept_precision": overall["auto_accept_precision"]}}


def release_acceptance_gates(calibrated: Mapping[str, Any], system_metrics: Mapping[str, float]) -> dict[str, Any]:
    """Evaluate every measurable release gate in the v1.1 plan in one auditable result.

    ``irrelevant_image_delta`` is signed accuracy minus paired text-only accuracy; latency is the
    Mac MLX three-question text-only measurement in milliseconds.
    """
    text = acceptance_gates(calibrated)
    required = {"image_heldout_accuracy", "image_trained_source_accuracy",
                "irrelevant_image_delta", "text_latency_ms", "external_baselines"}
    missing = required - set(system_metrics)
    if missing:
        raise ValueError(f"Missing system metrics: {sorted(missing)}")
    for name in ("image_heldout_accuracy", "image_trained_source_accuracy"):
        value = system_metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    delta, latency = system_metrics["irrelevant_image_delta"], system_metrics["text_latency_ms"]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or not -1 <= delta <= 1:
        raise ValueError("irrelevant_image_delta must be finite and in [-1, 1]")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError("text_latency_ms must be finite and non-negative")
    baselines = system_metrics["external_baselines"]
    valid_baselines = (isinstance(baselines, Mapping)
                       and all(baselines.get(name, {}).get("status") == "completed"
                               for name in ("our_base", "our_v1_1", "laya"))
                       and baselines.get("jev", {}).get("status") in {"completed", "unavailable_no_key"})
    system_checks = {
        "image_heldout_no_regression": system_metrics["image_heldout_accuracy"] >= 0.498 - 0.015,
        "image_trained_source_no_regression": system_metrics["image_trained_source_accuracy"] >= 0.858 - 0.01,
        "image_irrelevance_parity": abs(system_metrics["irrelevant_image_delta"]) <= 0.02,
        "text_latency": system_metrics["text_latency_ms"] <= 120.0,
        "external_baselines_reported": valid_baselines,
    }
    checks = {**text["checks"], **system_checks}
    return {"passed": all(checks.values()), "checks": checks,
            "observed": {**text["observed"], **dict(system_metrics)}}


def evaluate(rows: Sequence[Mapping[str, Any]], calibrator: TemperatureCalibrator) -> dict[str, Any]:
    raw, calibrated = metrics(rows), metrics(rows, calibrator)
    return {"raw": raw, "calibrated": calibrated, "acceptance": acceptance_gates(calibrated)}


def paired_family_bootstrap(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *,
    statistic: Callable[[Sequence[Mapping[str, Any]]], float] | None = None,
    iterations: int = 10_000, seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a paired metric delta by resampling whole held-out families."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    left_by_id, right_by_id = {row["id"]: row for row in left}, {row["id"]: row for row in right}
    if set(left_by_id) != set(right_by_id) or len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise ValueError("Paired bootstrap requires the same unique decision IDs")
    families: dict[str, list[str]] = defaultdict(list)
    for identifier, row in left_by_id.items():
        if row.get("family") != right_by_id[identifier].get("family"):
            raise ValueError("Paired rows must have matching families")
        families[str(row["family"])].append(identifier)
    names = sorted(families)
    if not names:
        raise ValueError("At least one family is required")

    def accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
        def correct(row: Mapping[str, Any]) -> bool:
            if "correct" in row:
                return bool(row["correct"])
            predicted = max(range(len(row["logits"])), key=lambda index: row["logits"][index])
            return predicted == row["target_index"]
        return sum(correct(row) for row in rows) / len(rows)

    stat = statistic or accuracy
    observed = stat(right) - stat(left)
    rng, deltas = random.Random(seed), []
    for _ in range(iterations):
        sampled = [rng.choice(names) for _ in names]
        ids = [identifier for family in sampled for identifier in families[family]]
        deltas.append(stat([right_by_id[i] for i in ids]) - stat([left_by_id[i] for i in ids]))
    deltas.sort()
    quantile = lambda p: deltas[min(len(deltas) - 1, max(0, int(p * len(deltas))))]
    return {"metric": "accuracy" if statistic is None else getattr(statistic, "__name__", "custom"),
            "delta_right_minus_left": observed, "ci95": [quantile(0.025), quantile(0.975)],
            "iterations": iterations, "families": len(names), "seed": seed}


def paired_irrelevance_delta(rows: Sequence[Mapping[str, Any]], *, iterations: int = 10_000,
                              seed: int = 0) -> dict[str, Any]:
    """Estimate irrelevant-image minus text-only accuracy on complete matched test pairs."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Control decision IDs must be unique")
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("partition") != "test":
            raise ValueError("Irrelevance evaluation accepts only partition='test' rows")
        pair_id, variant = row.get("pair_id"), row.get("control_variant")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every control row needs a non-empty pair_id")
        if variant not in {"text_only", "irrelevant_image"}:
            raise ValueError("control_variant must be text_only or irrelevant_image")
        if variant in pairs[pair_id]:
            raise ValueError(f"Duplicate {variant} row for pair {pair_id!r}")
        if not isinstance(row.get("correct"), bool):
            raise ValueError("Every control row needs a boolean correct value")
        pairs[pair_id][variant] = row
    if not pairs or any(set(pair) != {"text_only", "irrelevant_image"} for pair in pairs.values()):
        raise ValueError("Every pair_id needs one text_only and one irrelevant_image row")
    pair_deltas = [int(pair["irrelevant_image"]["correct"]) - int(pair["text_only"]["correct"])
                   for pair in pairs.values()]
    observed = sum(pair_deltas) / len(pair_deltas)
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(pair_deltas) for _ in pair_deltas) / len(pair_deltas)
                  for _ in range(iterations))
    quantile = lambda p: boot[min(len(boot) - 1, max(0, int(p * len(boot))))]
    return {"irrelevant_image_delta": observed, "ci95": [quantile(.025), quantile(.975)],
            "pairs": len(pair_deltas), "iterations": iterations, "seed": seed,
            "text_only_accuracy": sum(pair["text_only"]["correct"] for pair in pairs.values()) / len(pairs),
            "irrelevant_image_accuracy": sum(pair["irrelevant_image"]["correct"] for pair in pairs.values()) / len(pairs)}
