"""Temperature calibration for typed decision logits.

Artifacts contain one positive scalar temperature for each decision-type / option-count
bucket.  Option counts exclude the always-present ``__unknown__`` candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import UNKNOWN, Result

BUCKETS = ((2, 2), (3, 5), (6, 10), (11, 25), (26, 254))


def option_count_bucket(option_count: int) -> str:
    if isinstance(option_count, bool) or not isinstance(option_count, int) or option_count < 2:
        raise ValueError("option_count must be an integer >= 2")
    for low, high in BUCKETS:
        if option_count <= high:
            return str(low) if low == high else f"{low}-{high}"
    raise ValueError("option_count plus unknown exceeds the 255-option readout")


def calibration_key(decision_type: str, option_count: int) -> str:
    if decision_type not in {"choice", "boolean", "ordinal", "noul", "score"}:
        raise ValueError(f"Unsupported decision type {decision_type!r}")
    canonical = {"noul": "boolean", "score": "ordinal"}.get(decision_type, decision_type)
    return f"{canonical}:{option_count_bucket(option_count)}"


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected logits and a finite positive temperature")
    scaled = [float(value) / temperature for value in logits]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("Logits must be finite")
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def _nll(log_temperature: float, samples: Sequence[tuple[Sequence[float], int]], unknown_offset: float = 0.0) -> float:
    temperature = math.exp(log_temperature)
    loss = 0.0
    for logits, target in samples:
        probabilities = softmax(_offset(logits, unknown_offset), temperature)
        loss -= math.log(max(probabilities[target], 1e-300))
    return loss / len(samples)


def _offset(logits: Sequence[float], unknown_offset: float) -> list[float]:
    """Add ``unknown_offset`` to the unknown logit, which is always the last candidate."""
    values = [float(value) for value in logits]
    if unknown_offset:
        values[-1] += unknown_offset
    return values


def fit_unknown_offset_and_temperature(samples: Iterable[tuple[Sequence[float], int]], *,
                                       offsets: Sequence[float] = tuple(x / 10 for x in range(-40, 21)),
                                       max_temperature: float = 100.0) -> tuple[float, float]:
    """Grid-search an additive unknown-logit offset, refitting the temperature at each point; returns (offset, temperature).

    The offset moves the abstention boundary (it can change the argmax); the temperature then only reshapes confidence.
    Both are chosen by held-out NLL on the calibration fold, so a model that abstains too often gets a negative offset.
    """
    checked = [(tuple(float(v) for v in logits), target) for logits, target in samples]
    if not checked:
        raise ValueError("Cannot fit an offset without calibration samples")
    best: tuple[float, float, float] | None = None
    for offset in offsets:
        temperature = fit_temperature([(_offset(logits, offset), target) for logits, target in checked],
                                      max_temperature=max_temperature)
        loss = _nll(math.log(temperature), checked, offset)
        if best is None or loss < best[0] - 1e-9:
            best = (loss, offset, temperature)
    assert best is not None
    return best[1], best[2]


def fit_temperature(samples: Iterable[tuple[Sequence[float], int]], *, max_temperature: float = 100.0) -> float:
    """Fit a scalar temperature by bounded one-dimensional held-out NLL minimization."""
    checked: list[tuple[tuple[float, ...], int]] = []
    for logits, target in samples:
        values = tuple(float(value) for value in logits)
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("Each sample needs at least two finite logits")
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(values):
            raise ValueError("Target index is outside its candidate logits")
        checked.append((values, target))
    if not checked:
        raise ValueError("Cannot fit a temperature without calibration samples")
    if not math.isfinite(max_temperature) or max_temperature <= 1:
        raise ValueError("max_temperature must be finite and > 1")

    # Golden-section search in log space. Bounds prevent a separable fold from yielding 0 or infinity.
    lo, hi = math.log(1e-2), math.log(max_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f_left, f_right = _nll(left, checked), _nll(right, checked)
    for _ in range(96):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = _nll(left, checked)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = _nll(right, checked)
    candidates = ((0.0, _nll(0.0, checked)), (left, f_left), (right, f_right))
    return math.exp(min(candidates, key=lambda item: item[1])[0])


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Per-bucket temperatures, plus optional per-bucket additive offsets on the unknown logit.

    Offsets are fitted on the text-only calibration fold and are applied to text-only requests only
    (``image=False``); requests carrying images keep the raw abstention boundary.
    """
    version: str
    temperatures: Mapping[str, float]
    counts: Mapping[str, int]
    unknown_offsets: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.version or not isinstance(self.version, str):
            raise ValueError("Calibration version must be a non-empty string")
        if set(self.counts) != set(self.temperatures):
            raise ValueError("Temperature and count buckets must match")
        if self.unknown_offsets is not None:
            if set(self.unknown_offsets) - set(self.temperatures):
                raise ValueError("Unknown offsets must only name temperature buckets")
            for value in self.unknown_offsets.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Invalid unknown offset artifact")
        for key, value in self.temperatures.items():
            if (not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("Invalid temperature artifact")
            if (isinstance(self.counts[key], bool) or not isinstance(self.counts[key], int)
                    or self.counts[key] < 1):
                raise ValueError("Calibration bucket counts must be positive integers")

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, Any]], *, version: str | None = None,
            unknown_offsets: bool = False) -> "TemperatureCalibrator":
        grouped: dict[str, list[tuple[Sequence[float], int]]] = {}
        for row in rows:
            if row.get("partition") != "calibration":
                raise ValueError("Only held-out calibration rows may fit temperatures")
            option_count = row["option_count"]
            if isinstance(option_count, bool) or not isinstance(option_count, int):
                raise ValueError("option_count must be an integer")
            if len(row["logits"]) != option_count + 1:
                raise ValueError("option_count must equal len(logits) - 1 for unknown")
            key = calibration_key(str(row["decision_type"]), option_count)
            grouped.setdefault(key, []).append((row["logits"], row["target_index"]))
        if not grouped:
            raise ValueError("No calibration rows supplied")
        offsets: dict[str, float] | None = None
        if unknown_offsets:
            fitted = {key: fit_unknown_offset_and_temperature(values) for key, values in sorted(grouped.items())}
            offsets = {key: offset for key, (offset, _) in fitted.items()}
            temperatures = {key: temperature for key, (_, temperature) in fitted.items()}
        else:
            temperatures = {key: fit_temperature(values) for key, values in sorted(grouped.items())}
        counts = {key: len(values) for key, values in sorted(grouped.items())}
        payload = {"temperatures": temperatures, "counts": counts, "unknown_offsets": offsets}
        resolved_version = version or "temp-v1-" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return cls(resolved_version, temperatures, counts, offsets)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemperatureCalibrator":
        if value.get("schema_version") not in ("1.0", "1.1"):
            raise ValueError("Unsupported calibration artifact schema")
        offsets = value.get("unknown_offsets")
        return cls(str(value["calibration_version"]), dict(value["temperatures"]), dict(value["counts"]),
                   None if offsets is None else dict(offsets))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema_version": "1.1" if self.unknown_offsets is not None else "1.0",
                   "calibration_version": self.version,
                   "temperatures": dict(self.temperatures), "counts": dict(self.counts)}
        if self.unknown_offsets is not None:
            payload["unknown_offsets"] = dict(self.unknown_offsets)
        return payload

    def temperature(self, decision_type: str, option_count: int) -> float | None:
        return self.temperatures.get(calibration_key(decision_type, option_count))

    def unknown_offset(self, decision_type: str, option_count: int, *, image: bool = False) -> float:
        """Offset on the unknown logit for text-only requests; zero for requests with images or unseen buckets."""
        if image or not self.unknown_offsets:
            return 0.0
        return float(self.unknown_offsets.get(calibration_key(decision_type, option_count), 0.0))

    def calibrate_scores(self, raw_logits: Mapping[str, float], decision_type: str,
                         option_count: int, *, image: bool = False) -> tuple[dict[str, float], str | None]:
        """Return identity softmax/None for an unseen bucket, including unknown safely."""
        if UNKNOWN not in raw_logits:
            raise ValueError(f"Raw logits must include {UNKNOWN}")
        if len(raw_logits) != option_count + 1:
            raise ValueError("option_count must equal len(raw_logits) - 1 for unknown")
        temperature = self.temperature(decision_type, option_count)
        offset = self.unknown_offset(decision_type, option_count, image=image)
        keys = list(raw_logits)
        values = [raw_logits[key] + (offset if key == UNKNOWN else 0.0) for key in keys]
        probabilities = softmax(values, temperature or 1.0)
        return dict(zip(keys, probabilities)), self.version if temperature is not None else None

    def calibrate_result(self, result: Result, decision_type: str, option_count: int, *, image: bool = False) -> Result:
        """Rebuild a Result from calibrated logits; unknown participates like every other class.

        With a non-zero unknown offset (text-only requests) the argmax can move, so the value is re-derived
        from the calibrated scores; otherwise the engine's selected value (with its tie break) is kept.
        """
        scores, version = self.calibrate_scores(result.raw_logits, decision_type, option_count, image=image)
        status, value, reason = result.status, result.value, result.reason
        if self.unknown_offset(decision_type, option_count, image=image):
            current = UNKNOWN if value is None else _score_key(value)
            top = max(scores, key=lambda key: scores[key])
            if top != current and scores[top] > scores.get(current, float("-inf")):
                if top == UNKNOWN:
                    status, value, reason = "abstained", None, "insufficient_evidence"
                else:
                    status, value, reason = "answered", _typed_value(top, decision_type), None
        return Result(status=status, value=value,
                      scores=scores, raw_logits=dict(result.raw_logits),
                      score_semantics="calibrated_normalized_scores" if version else "uncalibrated_normalized_scores",
                      calibration_version=version,
                      reason=reason)


def _score_key(value: Any) -> str:
    """The scores/raw_logits key for a typed value (mirrors scoring.key)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _typed_value(key: str, decision_type: str) -> Any:
    """Recover a typed value from a scores key when the offset moves the argmax."""
    if decision_type in ("boolean", "noul"):
        if key not in ("true", "false"):
            raise ValueError(f"Unexpected boolean key {key!r}")
        return key == "true"
    if decision_type in ("ordinal", "score"):
        return int(key)
    return key
