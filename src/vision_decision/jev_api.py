"""Jev-style request and response shapes on top of the internal contract.

Request:  {"state": {...}, "questions": {"<id>": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}}
  instructions: a string, or an object such as {"question": "...", "today": "..."} (flattened to "key: value" lines);
                backticks refer to state keys or dotted paths, e.g. `subject`, `listing.color`
  noul   criteria: optional {"true": "...", "false": "..."}; instructions may be a question or a statement
  choice criteria: {"<option>": "<description>" | null, ...}
  score  criteria: ["<lowest level>", ..., "<highest level>"]  (2-10 ordered levels, scored from 0)
Response: {"answers": {"<id>": {...}}}; every answer also reports `unknown_probability` and `abstained`,
which Jev does not have: an image can simply fail to show what a question needs.
"""
import json
from .contracts import UNKNOWN, Request

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

def to_request(payload, request_id="jev-request"):
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), dict) or not payload["questions"]:
        raise ValueError("Expected an object with a non-empty 'questions' object")
    fields = []
    for name, q in payload["questions"].items():
        if not isinstance(q, dict) or "instructions" not in q:
            raise ValueError(f"Question {name!r} needs 'instructions'")
        q = {**q, "instructions": flatten_instructions(q["instructions"])}
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "noul":
            criteria = criteria or {}
            fields.append({"id": name, "type": "boolean", "question": q["instructions"],
                           "yes_description": flatten_description(criteria.get("true")), "no_description": flatten_description(criteria.get("false"))})
        elif kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError(f"Choice question {name!r} needs a criteria object of option -> description")
            fields.append({"id": name, "type": "choice", "question": q["instructions"],
                           "options": [{"value": k, "description": flatten_description(v)} for k, v in criteria.items()]})
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError(f"Score question {name!r} needs an ordered criteria list")
            fields.append({"id": name, "type": "ordinal", "question": q["instructions"],
                           "levels": [{"value": i, "description": flatten_description(text)} for i, text in enumerate(criteria)]})
        else:
            raise ValueError(f"Question {name!r} has unsupported type {kind!r}; use noul, choice or score")
    state = payload.get("state", {})  # Jev allows a string or an array too; keep it addressable under one key
    return Request.model_validate({"request_id": request_id, "state": state if isinstance(state, (dict, str)) else {"state": state}, "fields": fields})

def _known(result):
    """Probabilities over the real answers, renormalized without the abstention mass."""
    known = {k: v for k, v in result.scores.items() if k != UNKNOWN}
    total = sum(known.values())
    return {k: (v / total if total > 0 else 1.0 / len(known)) for k, v in known.items()}, result.scores[UNKNOWN]

def to_response(request, results, model="imajev"):
    """Jev's answer shapes, plus `unknown_probability` / `abstained`.

    Jev has no built-in abstention: an undeterminable noul sits near 0.5 and an uncertain choice or score has low
    confidence. We keep that behaviour: the unknown mass pulls noul toward 0.5 and scales confidence down.
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
    return {"model": model, "answers": answers}
