from __future__ import annotations

import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

# Serialized state limit (compact UTF-8 JSON). Raised from 32 KB to 128 KB in phase 3 (~32k tokens) so long documents can be
# trained and served; the processed-token limit (--max-input-tokens) remains the real guard.
MAX_STATE_BYTES = 131072

UNKNOWN = "__unknown__"
MAX_INTERNAL_FIELDS = 64
Text = Annotated[StrictStr, Field(min_length=1, max_length=2000)]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class Option(StrictModel):
    value: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    description: Text | None = None  # a bare label is a valid option, as in Jev's choice type

    @model_validator(mode="after")
    def reserved(self):
        if self.value == UNKNOWN:
            raise ValueError("__unknown__ is reserved")
        return self

class Level(StrictModel):
    value: StrictInt
    description: Text

class CommonField(StrictModel):
    id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
    question: Text

class ChoiceField(CommonField):
    type: Literal["choice"]
    # +unknown: 254 options fill the shipped 255-code readout; 255 need the extended 256-code readout.
    # The serving limit for the loaded adapter (254 or 255) is enforced by jev_api.to_request(max_options=...).
    options: Annotated[list[Option], Field(min_length=2, max_length=255)]

    @model_validator(mode="after")
    def unique(self):
        if len({o.value for o in self.options}) != len(self.options):
            raise ValueError("Choice values must be unique")
        return self

class BooleanField(CommonField):
    type: Literal["boolean"]
    # Optional criteria for each answer, as in Jev's noul type; the question may also be a statement.
    yes_description: Text | None = None
    no_description: Text | None = None

class OrdinalField(CommonField):
    type: Literal["ordinal"]
    levels: Annotated[list[Level], Field(min_length=2, max_length=10)]  # Jev's score type allows 2-10 levels

    @model_validator(mode="after")
    def ordered(self):
        values = [x.value for x in self.levels]
        if values != sorted(set(values)):
            raise ValueError("Ordinal levels must be unique and ascending")
        return self

DecisionField = Annotated[ChoiceField | BooleanField | OrdinalField, Field(discriminator="type")]

class Execution(StrictModel):
    mode: Literal["inspect"] = "inspect"
    allow_external_fallback: Literal[False] = False

class Request(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    state: dict | StrictStr = Field(default_factory=dict)
    # A public request has at most 8 questions (jev_api.to_request); a `multi` question fans out to one
    # boolean field per label, so the internal request may carry more fields.
    fields: Annotated[list[DecisionField], Field(min_length=1, max_length=MAX_INTERNAL_FIELDS)]
    execution: Execution = Field(default_factory=Execution)

    @model_validator(mode="after")
    def bounds(self):
        if len({f.id for f in self.fields}) != len(self.fields):
            raise ValueError("Field IDs must be unique")
        def check(value, depth=0):
            if depth > 8:
                raise ValueError("State nesting exceeds 8")
            if isinstance(value, dict):
                if not all(isinstance(k, str) for k in value):
                    raise ValueError("State keys must be strings")
                for v in value.values(): check(v, depth + 1)
            elif isinstance(value, list):
                for v in value: check(v, depth + 1)
            elif value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("State must contain JSON values")
        check(self.state)
        if len(json.dumps(self.state, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError(f"State exceeds {MAX_STATE_BYTES} bytes")
        return self

class Result(StrictModel):
    status: Literal["answered", "abstained"]
    value: StrictStr | StrictInt | bool | None
    scores: dict[str, float]
    raw_logits: dict[str, float]
    score_semantics: Literal["uncalibrated_normalized_scores", "calibrated_normalized_scores"] = "uncalibrated_normalized_scores"
    calibration_version: StrictStr | None = None
    reason: Literal["insufficient_evidence"] | None = None

    @model_validator(mode="after")
    def consistent(self):
        import math
        if (self.score_semantics == "calibrated_normalized_scores") != bool(self.calibration_version):
            raise ValueError("Calibrated scores require a non-empty calibration version")
        if not self.scores or set(self.scores) != set(self.raw_logits) or UNKNOWN not in self.scores:
            raise ValueError("Score/logit keys must match and include unknown")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in self.scores.values()):
            raise ValueError("Invalid normalized scores")
        if not all(math.isfinite(v) for v in self.raw_logits.values()):
            raise ValueError("Invalid logits")
        if not math.isclose(sum(self.scores.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Scores must sum to one")
        if self.status == "abstained":
            if self.value is not None or self.reason is None:
                raise ValueError("Abstention requires null value and reason")
        elif self.value is None or self.reason is not None:
            raise ValueError("Answered result requires value and no reason")
        return self
