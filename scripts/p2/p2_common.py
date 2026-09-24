"""Shared pieces for the decision-p2 generators: OpenAI-compatible client, writer/answerer JSON schemas, option-key rebuild,
answer parsing, and the JevBench contamination lint. Everything here runs without a model (tests use fake clients)."""
from __future__ import annotations
import asyncio, hashlib, json, re, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any, Iterable
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
WRITER_MODEL = {"repo": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "served": "qwen3.6-27b", "family": "qwen"}
ANSWERERS = {
    "qwen": {"repo": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "served": "qwen3.6-27b", "family": "qwen", "extra": {}},
    "gptoss": {"repo": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "served": "gpt-oss-20b", "family": "openai-oss",
               "extra": {"reasoning_effort": "medium"}},
    # third answerer added for phase 2b (2026-09-24): frozen Qwen3.6-35B-A3B with thinking scored 97.3 on JevBench hard / 100 on original
    "qwen35": {"repo": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "served": "qwen3.6-35b-a3b", "family": "qwen-moe", "extra": {}},
}
UNKNOWN_REASONS = ("insufficient_evidence", "false_premise", "not_listed", "mismatched_reference")


# ------------------------------------------------------------------------------------------------- writer output schema
class WOption(BaseModel):
    text: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr = Field(min_length=1, max_length=400)


class WLevel(BaseModel):
    value: StrictInt
    description: StrictStr = Field(min_length=1, max_length=400)


class WQuestion(BaseModel):
    family: StrictStr
    type: StrictStr
    question: StrictStr = Field(min_length=8, max_length=1500)
    options: list[WOption] | None = None
    levels: list[WLevel] | None = None
    intended: StrictBool | StrictInt | StrictStr | None = None
    unknown_reason: StrictStr | None = None
    justification: StrictStr = Field(min_length=1, max_length=800)

    @field_validator("type")
    @classmethod
    def _type(cls, v):
        if v not in ("noul", "choice", "score"):
            raise ValueError("type must be noul, choice or score")
        return v

    @model_validator(mode="after")
    def _shape(self):
        if self.type == "choice":
            if not self.options or not 3 <= len(self.options) <= 12:
                raise ValueError("choice needs 3-12 options")
            texts = [o.text.strip().lower() for o in self.options]
            if len(set(texts)) != len(texts):
                raise ValueError("duplicate option texts")
            if self.intended is not None and (not isinstance(self.intended, str) or self.intended.strip().lower() not in texts):
                raise ValueError("intended must be one of the option texts")
        elif self.type == "score":
            if not self.levels or not 2 <= len(self.levels) <= 10:
                raise ValueError("score needs 2-10 levels")
            vals = [l.value for l in self.levels]
            if vals != sorted(vals) or len(set(vals)) != len(vals):
                raise ValueError("levels must be strictly increasing")
            if self.intended is not None and (isinstance(self.intended, bool) or not isinstance(self.intended, int) or self.intended not in vals):
                raise ValueError("intended must be one of the level values")
        else:
            if self.intended is not None and not isinstance(self.intended, bool):
                raise ValueError("noul intended must be a boolean")
        if (self.intended is None) != (self.unknown_reason is not None):
            raise ValueError("unknown_reason exactly when intended is null")
        if self.unknown_reason is not None and self.unknown_reason not in UNKNOWN_REASONS:
            raise ValueError("bad unknown_reason")
        return self


class WriterOutput(BaseModel):
    document: StrictStr | dict[str, Any]
    document_kind: StrictStr | None = None
    questions: list[WQuestion] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _doc(self):
        text = self.document if isinstance(self.document, str) else json.dumps(self.document)
        words = len(text.split())
        if not 120 <= words <= 1600:
            raise ValueError(f"document length {words} words outside 120-1600")
        if len(json.dumps(self.document).encode()) > 32768:
            raise ValueError("document exceeds the 32 KB state limit")
        return self


# ------------------------------------------------------------------------------------------------- option keys / fields
def option_key(text: str, taken: set[str] | None = None) -> str:
    """Rebuild an option key from its text: lowercase snake_case, ASCII, <= 48 chars, unique within a field, never 'unknown'."""
    key = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")[:48] or "option"
    if key in ("unknown", "__unknown__"):
        key = "option_" + key
    if taken is not None:
        base, n = key, 2
        while key in taken:
            suffix = f"_{n}"; key = base[:48 - len(suffix)] + suffix; n += 1
        taken.add(key)
    return key


def to_field(q: WQuestion, field_id: str = "decision") -> tuple[dict, Any]:
    """Our request field + the intended target in our value space (bool / option key / int / None)."""
    if q.type == "noul":
        return {"id": field_id, "type": "boolean", "question": q.question}, q.intended
    if q.type == "choice":
        taken: set[str] = set(); options = []; intended = None
        for o in q.options:
            key = option_key(o.text, taken); options.append({"value": key, "description": o.description})
            if q.intended is not None and o.text.strip().lower() == q.intended.strip().lower():
                intended = key
        return {"id": field_id, "type": "choice", "question": q.question, "options": options}, intended
    levels = [{"value": l.value, "description": l.description} for l in q.levels]
    return {"id": field_id, "type": "ordinal", "question": q.question, "levels": levels}, q.intended


# ------------------------------------------------------------------------------------------------- answer parsing
def parse_answer(raw: str, field: dict) -> tuple[Any, float | None, str | None]:
    """(value in our value space or None for unknown, confidence, error). Unparseable -> error string."""
    try:
        obj = json.loads(_extract_json(raw))
    except Exception as exc:
        return None, None, f"json: {type(exc).__name__}"
    if not isinstance(obj, dict) or "answer" not in obj:
        return None, None, "no answer key"
    a = obj["answer"]; conf = obj.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 <= conf <= 1 else None
    if isinstance(a, str) and a.strip().lower() in ("unknown", "__unknown__", "null", "none"):
        return None, conf, None
    if field["type"] == "boolean":
        if isinstance(a, bool): return a, conf, None
        if isinstance(a, str) and a.strip().lower() in ("true", "yes"): return True, conf, None
        if isinstance(a, str) and a.strip().lower() in ("false", "no"): return False, conf, None
        return None, conf, f"bad boolean {a!r}"
    if field["type"] == "choice":
        keys = [o["value"] for o in field["options"]]
        if isinstance(a, str):
            k = a.strip(); k2 = option_key(k)
            if k in keys: return k, conf, None
            if k2 in keys: return k2, conf, None
            for o in field["options"]:  # answered with the description text
                if o["description"].strip().lower() == k.lower(): return o["value"], conf, None
        return None, conf, f"bad option {a!r}"
    vals = [l["value"] for l in field["levels"]]
    if isinstance(a, bool): return None, conf, "bool for ordinal"
    if isinstance(a, int) and a in vals: return a, conf, None
    if isinstance(a, str) and a.strip().lstrip("-").isdigit() and int(a) in vals: return int(a), conf, None
    return None, conf, f"bad level {a!r}"


# ------------------------------------------------------------------------------------------------- soft targets (phase 2c)
# A question's LABELS are the strings a distribution is keyed by, identical to the trainer's `target_probs` aliases
# (scripts/decision_data.py:distribution_weights): "true"/"false" for booleans, option keys for choices, the integer as a string
# for ordinal levels, and "unknown" (always present, always last) for abstention.
UNKNOWN_LABEL = "unknown"
_UNKNOWN_ALIASES = ("unknown", "__unknown__", "null", "none")


def field_labels(field: dict, with_unknown: bool = True) -> list[str]:
    if field["type"] == "boolean":
        labels = ["true", "false"]
    elif field["type"] == "choice":
        labels = [str(o["value"]) for o in field["options"]]
    else:
        labels = [str(l["value"]) for l in field["levels"]]
    return labels + [UNKNOWN_LABEL] if with_unknown else labels


def value_to_label(value) -> str:
    """Our value space (bool / option key / int / None) -> label."""
    if value is None:
        return UNKNOWN_LABEL
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def label_to_value(field: dict, label: str):
    """Label -> our value space; "unknown" -> None."""
    if label == UNKNOWN_LABEL:
        return None
    if field["type"] == "boolean":
        return label == "true"
    if field["type"] == "ordinal":
        return int(label)
    return label


def normalise_label(field: dict, key) -> str | None:
    """Map a key as a model or an upstream file writes it onto one of the field's labels (None when unrecognised)."""
    labels = field_labels(field, with_unknown=False)
    k = str(key).strip()
    if k.lower() in _UNKNOWN_ALIASES:
        return UNKNOWN_LABEL
    if field["type"] == "boolean":
        return {"true": "true", "yes": "true", "false": "false", "no": "false"}.get(k.lower())
    if k in labels:
        return k
    if field["type"] == "choice":
        k2 = option_key(k)
        return k2 if k2 in labels else None
    if k.lstrip("-").isdigit() and str(int(k)) in labels:
        return str(int(k))
    return None


def project_probs(probs, field: dict, strict: bool = False) -> tuple[dict | None, str | None]:
    """Renormalise a {key: probability} mapping over the field's labels + "unknown" (missing labels count as 0; the result
    always carries every label, "unknown" included). Unrecognised keys are ignored (strict=False, model output) or an error
    (strict=True, stored data). Returns (probs, None) or (None, error)."""
    import math
    if not isinstance(probs, dict):
        return None, "probabilities is not an object"
    out = {lab: 0.0 for lab in field_labels(field)}
    for k, v in probs.items():
        lab = normalise_label(field, k)
        if lab is None:
            if strict:
                return None, f"unrecognised label {k!r}"
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return None, f"bad probability for {k!r}: {v!r}"
        out[lab] += float(v)
    total = sum(out.values())
    if total <= 0:
        return None, "probabilities sum to zero"
    return {k: round(v / total, 6) for k, v in out.items()}, None


def argmax_label(probs: dict, field: dict) -> str:
    """First maximum in label order (real options first, unknown last), as the trainer resolves ties."""
    labels = field_labels(field)
    return max(labels, key=lambda lab: (probs.get(lab, 0.0), -labels.index(lab)))


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


# ------------------------------------------------------------------------------------------------- OpenAI-compatible client
class ChatClient:
    """Minimal OpenAI-compatible chat client (vLLM). `complete(messages, **kw)` -> content string. Retries transient errors."""

    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 600.0, opener=urllib.request.urlopen, sleep=time.sleep):
        self.base_url, self.model, self.api_key, self.timeout, self.opener, self.sleep = base_url.rstrip("/"), model, api_key, timeout, opener, sleep

    def complete(self, messages: list[dict], temperature: float = 0.9, max_tokens: int = 6000, json_mode: bool = True, extra: dict | None = None,
                 response_format: dict | None = None) -> str:
        """`response_format` (e.g. a json_schema object) overrides the plain JSON mode when given."""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format is not None:
            body["response_format"] = response_format
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if extra:
            body.update(extra)
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(6):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                         headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
            try:
                with self.opener(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                msg = payload["choices"][0]["message"]
                return msg.get("content") or ""
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last = exc; self.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"chat completion failed after retries: {last}")


def run_parallel(tasks: Iterable, worker, workers: int = 16):
    """Run `worker(task)` over tasks with a thread pool; yields (task, result_or_exception) in completion order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                yield t, fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded per task, never fatal
                yield t, exc


# ------------------------------------------------------------------------------------------------- contamination lint
_TOKEN = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from _strings(v)
    elif isinstance(value, list):
        for v in value: yield from _strings(v)


def reference_ngrams(paths: Iterable[Path], n: int = 8) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = line
            for s in _strings(obj):
                grams |= ngrams(s, n)
    return grams


def contamination_count(record_text: str, reference: set[tuple[str, ...]], n: int = 8) -> int:
    return len(ngrams(record_text, n) & reference)


def jevbench_public_files() -> list[Path]:
    base = ROOT / ".cache/external/jevbench/datasets/public"
    return sorted(base.glob("*.jsonl")) if base.is_dir() else []


def doc_id(domain_id: str, seed: int) -> str:
    return f"{domain_id}-{hashlib.sha256(f'p2\0{domain_id}\0{seed}'.encode()).hexdigest()[:12]}"


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

def normalise_writer_output(obj: dict) -> dict:
    """Coerce the shapes the writer naturally emits into the WriterOutput schema (strings for options, bare
    integers or {"score": n} for levels, numeric strings for intended values, raw text descriptions)."""
    if not isinstance(obj, dict):
        return obj
    for q in obj.get("questions") or []:
        if not isinstance(q, dict):
            continue
        opts = q.get("options")
        if isinstance(opts, dict):
            opts = [{"text": str(k), "description": str(v)} for k, v in opts.items()]
        if isinstance(opts, list):
            fixed = []
            for o in opts:
                if isinstance(o, str):
                    fixed.append({"text": o.strip()[:120], "description": o.strip()[:400] or "option"})
                elif isinstance(o, dict):
                    text = o.get("text") or o.get("option") or o.get("label") or o.get("value") or o.get("name") or ""
                    desc = o.get("description") or o.get("meaning") or o.get("desc") or str(text)
                    fixed.append({"text": str(text).strip()[:120], "description": str(desc).strip()[:400] or str(text)})
                else:
                    fixed.append({"text": str(o)[:120], "description": str(o)[:400]})
            q["options"] = fixed
        lv = q.get("levels")
        if isinstance(lv, dict):
            lv = [{"value": k, "description": v} for k, v in lv.items()]
        if isinstance(lv, list):
            fixed = []
            for i, l in enumerate(lv):
                if isinstance(l, bool):
                    continue
                if isinstance(l, (int, float)) or (isinstance(l, str) and l.strip().lstrip("-").isdigit()):
                    fixed.append({"value": int(l), "description": f"level {int(l)}"})
                elif isinstance(l, dict):
                    v = l.get("value", l.get("score", l.get("level", l.get("rank", i))))
                    try:
                        v = int(str(v).strip())
                    except (TypeError, ValueError):
                        v = i
                    desc = l.get("description") or l.get("label") or l.get("meaning") or l.get("text") or f"level {v}"
                    fixed.append({"value": v, "description": str(desc).strip()[:400] or f"level {v}"})
            q["levels"] = fixed
        it = q.get("intended")
        if q.get("type") == "score" and isinstance(it, str) and it.strip().lstrip("-").isdigit():
            q["intended"] = int(it.strip())
        if q.get("type") == "noul" and isinstance(it, str) and it.strip().lower() in ("true", "false", "yes", "no"):
            q["intended"] = it.strip().lower() in ("true", "yes")
        if q.get("type") == "choice" and isinstance(it, dict):
            q["intended"] = it.get("text") or it.get("option") or it.get("label") or it.get("value")
        if isinstance(q.get("justification"), list):
            q["justification"] = " ".join(str(x) for x in q["justification"])[:800]
        if q.get("justification") in (None, ""):
            q["justification"] = "see document"
    return obj


def loads_lenient(raw: str):
    """JSON with control characters inside strings (the writer emits raw newlines) and fenced blocks."""
    text = _extract_json(raw) if "_extract_json" in globals() else raw
    return json.loads(text, strict=False)
