"""Jev-style request and response shapes on top of the internal contract.

Request:  {"state": {...}, "questions": {"<id>": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}}
  instructions: a string, or an object such as {"question": "...", "today": "..."} (flattened to "key: value" lines);
                backticks refer to state keys or dotted paths, e.g. `subject`, `listing.color`
  noul   criteria: optional {"true": "...", "false": "..."}; instructions may be a question or a statement
  choice criteria: {"<option>": "<description>" | null, ...}
  score  criteria: ["<lowest level>", ..., "<highest level>"]  (2-10 ordered levels, scored from 0)
  multi  criteria: {"<label>": "<description>" | null, ...}  (1-32 labels; "pick all that apply"), optional "threshold"
         (default 0.5). Serving only: fans out to one noul per label (multi_label_question), all scored in the same
         request, so the image and state prefill is shared.
Response: {"answers": {"<id>": {...}}}; every answer also reports `unknown_probability` and `abstained`,
which Jev does not have: an image can simply fail to show what a question needs.
  multi answer: {"type": "multi", "labels": [selected, in criteria order], "probabilities": {label: p}, "threshold": t,
                 "unknown_probabilities": {label: u}, "abstained": every label abstained}; p is the label's noul value
                 (P(yes) + 0.5 x P(unknown)) and a label is selected when p > t (strictly, so a fully unknown label is not).
"""
import json
import math
from functools import lru_cache
from typing import Annotated
from pydantic import Field, TypeAdapter
from .contracts import UNKNOWN, Request

MAX_QUESTIONS = 8
MAX_OPTIONS = 254          # shipped 255-code readout; the extended 256-code readout serves 255
MAX_MULTI_LABELS = 32
DEFAULT_MULTI_THRESHOLD = 0.5

def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def flatten_instructions(instructions):
    """Jev accepts instructions (and criteria descriptions) as a string, an object or an array.

    Objects become one "key: value" line per entry, in order; arrays one line per item. Training data
    uses this same function, so the model sees structured instructions exactly as they are served.
    """
    if isinstance(instructions, str) and instructions:
        return instructions
    if isinstance(instructions, dict) and instructions:
        return "\n".join(f"{k}: {_text(v)}" for k, v in instructions.items())
    if isinstance(instructions, list) and instructions:
        return "\n".join(_text(v) for v in instructions)
    raise ValueError("'instructions' must be a non-empty string, object or array")

def flatten_description(description):
    """Criteria descriptions may be null, a string, or structured ({"what": ..., "not_for": ..., "examples": [...]})."""
    if description is None or isinstance(description, str):
        return description or None
    if isinstance(description, dict):
        return "; ".join(f"{k}: {_text(v)}" for k, v in description.items()) or None
    return "; ".join(_text(v) for v in description) or None

def concentration(probabilities):
    """Jev's confidence for choice and score: (n * p_max - 1) / (n - 1); 1.0 on one option, 0.0 when uniform."""
    n = len(probabilities)
    return 1.0 if n < 2 else max(0.0, (n * max(probabilities.values()) - 1) / (n - 1))

@lru_cache(maxsize=4)
def _option_limit(max_options):
    """Options + unknown must fit the loaded readout: 254 options with 255 codes, 255 with the extended 256."""
    if max_options not in (254, 255):
        raise ValueError("max_options must be 254 (255-code readout) or 255 (256-code readout)")
    return TypeAdapter(Annotated[list, Field(max_length=max_options)])

def multi_label_question(instructions, label, description=None):
    """The noul question one `multi` label becomes. Training data for multi-label families must use this
    exact wording (per-label noul rows), so the model sees at training time what the server sends."""
    detail = f" ({description})" if description else ""
    return f"{instructions}\nLabel: {label}{detail}\nDoes this label apply?"

def to_request(payload, request_id="jev-request", max_options=MAX_OPTIONS):
    """The internal Request for a Jev-style payload (multi questions expanded into noul fields)."""
    return to_request_with_plan(payload, request_id, max_options)[0]

def to_request_with_plan(payload, request_id="jev-request", max_options=MAX_OPTIONS):
    """-> (Request, plan). plan maps each `multi` question name to its labels, internal field ids and threshold;
    pass it to to_response so the per-label nouls are folded back into one multi answer."""
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), dict) or not payload["questions"]:
        raise ValueError("Expected an object with a non-empty 'questions' object")
    if len(payload["questions"]) > MAX_QUESTIONS:
        raise ValueError(f"A request has at most {MAX_QUESTIONS} questions; got {len(payload['questions'])}")
    fields, plan = [], {}
    names = set(payload["questions"])
    for name, q in payload["questions"].items():
        if not isinstance(q, dict) or "instructions" not in q:
            raise ValueError(f"Question {name!r} needs 'instructions'")
        q = {**q, "instructions": flatten_instructions(q["instructions"])}
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "multi":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"Multi question {name!r} needs a criteria object of label -> description")
            if len(criteria) > MAX_MULTI_LABELS:
                raise ValueError(f"Multi question {name!r} has {len(criteria)} labels; at most {MAX_MULTI_LABELS}")
            threshold = q.get("threshold", DEFAULT_MULTI_THRESHOLD)
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
                    or not 0 <= threshold <= 1):
                raise ValueError(f"Multi question {name!r}: threshold must be a number in [0, 1]")
            entries = []
            for index, (label, description) in enumerate(criteria.items()):
                field_id = f"{name}__{index}"
                if len(field_id) > 64:
                    raise ValueError(f"Multi question name {name!r} is too long (at most {62 - len(str(index))} characters here)")
                if field_id in names:
                    raise ValueError(f"Multi question {name!r} collides with question {field_id!r}")
                fields.append({"id": field_id, "type": "boolean",
                               "question": multi_label_question(q["instructions"], label, flatten_description(description))})
                entries.append((label, field_id))
            plan[name] = {"type": "multi", "labels": entries, "threshold": float(threshold)}
        elif kind == "noul":
            criteria = criteria or {}
            fields.append({"id": name, "type": "boolean", "question": q["instructions"],
                           "yes_description": flatten_description(criteria.get("true")), "no_description": flatten_description(criteria.get("false"))})
        elif kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError(f"Choice question {name!r} needs a criteria object of option -> description")
            _option_limit(max_options).validate_python(list(criteria))  # pydantic ValidationError, as the contract raised
            fields.append({"id": name, "type": "choice", "question": q["instructions"],
                           "options": [{"value": k, "description": flatten_description(v)} for k, v in criteria.items()]})
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError(f"Score question {name!r} needs an ordered criteria list")
            fields.append({"id": name, "type": "ordinal", "question": q["instructions"],
                           "levels": [{"value": i, "description": flatten_description(text)} for i, text in enumerate(criteria)]})
        else:
            raise ValueError(f"Question {name!r} has unsupported type {kind!r}; use noul, choice, score or multi")
    state = payload.get("state", {})  # Jev allows a string or an array too; keep it addressable under one key
    request = Request.model_validate({"request_id": request_id, "state": state if isinstance(state, (dict, str)) else {"state": state}, "fields": fields})
    return request, plan

def _known(result):
    """Probabilities over the real answers, renormalized without the abstention mass."""
    known = {k: v for k, v in result.scores.items() if k != UNKNOWN}
    total = sum(known.values())
    return {k: (v / total if total > 0 else 1.0 / len(known)) for k, v in known.items()}, result.scores[UNKNOWN]

def to_response(request, results, model="imajev", plan=None):
    """Jev's answer shapes, plus `unknown_probability` / `abstained`.

    Jev has no built-in abstention: an undeterminable noul sits near 0.5 and an uncertain choice or score has low
    confidence. We keep that behaviour: the unknown mass pulls noul toward 0.5 and scales confidence down.
    `plan` (from to_request_with_plan) folds each multi question's per-label nouls into one multi answer.
    """
    if len(results) != len(request.fields):
        raise ValueError("Expected one result per request field")
    answers = {}
    for field, result in zip(request.fields, results):
        known, unknown = _known(result)
        common = {"unknown_probability": unknown, "abstained": result.status == "abstained"}
        if result.calibration_version is not None:
            common["calibration_version"] = result.calibration_version
        if field.type == "boolean":
            answers[field.id] = {"type": "noul", "noul": result.scores["true"] + 0.5 * unknown, **common}
        elif field.type == "choice":
            answers[field.id] = {"type": "choice", "choice": max(known, key=known.get), "probabilities": known,
                                 "confidence": concentration(known) * (1 - unknown), **common}
        else:
            answers[field.id] = {"type": "score", "score": sum(float(k) * v for k, v in known.items()),
                                 "legend": {str(l.value): l.description for l in field.levels}, "probabilities": known,
                                 "confidence": concentration(known) * (1 - unknown), **common}
    for name, entry in (plan or {}).items():
        probabilities, unknowns, abstained = {}, {}, []
        for label, field_id in entry["labels"]:
            part = answers.pop(field_id)
            probabilities[label], unknowns[label] = part["noul"], part["unknown_probability"]
            abstained.append(part["abstained"])
        threshold = entry["threshold"]
        answers[name] = {"type": "multi", "labels": [label for label, p in probabilities.items() if p > threshold + 1e-9],  # rounding cannot select a pure-unknown label
                         "probabilities": probabilities, "threshold": threshold,
                         "unknown_probabilities": unknowns, "abstained": all(abstained)}
    owner = {field_id: name for name, entry in (plan or {}).items() for _, field_id in entry["labels"]}
    order = dict.fromkeys(owner.get(f.id, f.id) for f in request.fields)  # question order as requested
    return {"model": model, "answers": {name: answers[name] for name in order}}
