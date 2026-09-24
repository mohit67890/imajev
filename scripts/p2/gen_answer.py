"""Stage 2 of decision-p2: an ANSWERER model answers every writer question independently (document + question only).

    OPENAI_BASE_URL=... python scripts/p2/gen_answer.py --answerer qwen  --writer data/decision-p2/teacher/writer.jsonl --out data/decision-p2/teacher/answers-qwen.jsonl
    OPENAI_BASE_URL=... python scripts/p2/gen_answer.py --answerer gptoss --writer ... --out data/decision-p2/teacher/answers-gptoss.jsonl

Phase 2c soft targets (`--mode distribution`, docs/phase-2c-plan.md): the answerer (normally `qwen35`, Qwen3.6-35B-A3B with thinking
on; serve it with vLLM's reasoning parser so `content` is the JSON only) returns a probability for EVERY label of the question,
`unknown` included, plus a rationale of at most two sentences, constrained by a JSON schema (response_format json_schema):

    OPENAI_BASE_URL=... python scripts/p2/gen_answer.py --mode distribution --answerer qwen35 --writer ... --out .../answers-p2c-relabel-qwen35.jsonl

Distribution rows: {doc_id, qi, answerer, model, repo, revision, mode: "distribution", value: argmax label in our value space (None for
unknown), probs: {label: p} renormalised over the labels + "unknown", rationale, raw, parse_error}. Labels are the trainer's
`target_probs` keys: "true"/"false", option keys, ordinal integers as strings, "unknown".

Resumable by (doc_id, question index). Records the raw reply, the parsed value and the parse error, never the intended answer.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import (ANSWERERS, UNKNOWN_LABEL, ChatClient, WQuestion, append_jsonl, argmax_label, field_labels, label_to_value, loads_lenient,
                       parse_answer, project_probs, read_jsonl, run_parallel, to_field)
from families import ANSWERER_SYSTEM, answerer_prompt

DEFAULT_MAX_TOKENS = {"answer": 8000, "distribution": 8000}
RATIONALE_MAX_CHARS = 400

DISTRIBUTION_SYSTEM = (
    "You estimate how likely each possible answer to one typed question about a document is to be correct. Use only the document. "
    "Work it out carefully, then give a probability for EVERY listed label, including \"unknown\" (the document does not determine the "
    "answer, its premise is false, or no listed option is correct). The probabilities sum to 1 and state your real uncertainty: put "
    "almost all of the mass on one label when the document settles the question, and spread it only as far as the document leaves "
    "the answer open. Then give a rationale of at most two sentences naming the evidence that decides it. Reply with a single JSON "
    "object {\"probabilities\": {<label>: <number>, ...}, \"rationale\": <string>} and nothing else."
)


def questions(writer_rows: list[dict]):
    for r in writer_rows:
        for qi, q in enumerate(r["output"]["questions"]):
            field, _ = to_field(WQuestion.model_validate(q), field_id="decision")
            yield {"doc_id": r["doc_id"], "qi": qi, "state": r["output"]["document"], "field": field}


def distribution_prompt(state, field: dict) -> str:
    doc = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    unknown = f"- {UNKNOWN_LABEL}: the document does not determine the answer"
    if field["type"] == "boolean":
        lines = ["- true: yes", "- false: no", unknown]
    elif field["type"] == "choice":
        lines = [f"- {o['value']}: {o['description']}" for o in field["options"]] + [unknown]
    else:
        lines = [f"- {l['value']}: {l['description']}" for l in field["levels"]] + [unknown]
    return (f"DOCUMENT:\n{doc}\n\nQUESTION: {field['question']}\nLabels (give a probability for each, keyed exactly as written):\n"
            + "\n".join(lines) + "\nJSON only.")


def distribution_schema(field: dict) -> dict:
    """response_format for vLLM structured outputs: every label required, nothing else allowed."""
    labels = field_labels(field)
    return {"type": "json_schema", "json_schema": {"name": "decision_distribution", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["probabilities", "rationale"],
        "properties": {
            "probabilities": {"type": "object", "additionalProperties": False, "required": labels,
                              "properties": {lab: {"type": "number", "minimum": 0, "maximum": 1} for lab in labels}},
            "rationale": {"type": "string", "maxLength": RATIONALE_MAX_CHARS}}}}}


def _two_sentences(text: str) -> str:
    text = " ".join(str(text).split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:2])[:RATIONALE_MAX_CHARS].strip()


def parse_distribution(raw: str, field: dict) -> tuple[object, dict | None, str | None, str | None]:
    """(value, probs, rationale, error). probs are renormalised over the field's labels + unknown (missing labels -> 0,
    unrecognised keys ignored); value is the argmax label in our value space (None for unknown or on error)."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    try:
        obj = loads_lenient(text)
    except Exception as exc:
        return None, None, None, f"json: {type(exc).__name__}"
    if not isinstance(obj, dict) or "probabilities" not in obj:
        return None, None, None, "no probabilities key"
    probs, err = project_probs(obj["probabilities"], field)
    rationale = obj.get("rationale")
    rationale = _two_sentences(rationale) if isinstance(rationale, str) and rationale.strip() else None
    if err:
        return None, None, rationale, err
    return label_to_value(field, argmax_label(probs, field)), probs, rationale, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", choices=sorted(ANSWERERS), required=True)
    ap.add_argument("--mode", choices=("answer", "distribution"), default="answer",
                    help="answer: one JSON answer + confidence (phase 2/2b); distribution: probabilities over every label + rationale (phase 2c)")
    ap.add_argument("--writer", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=None, help="default 8000 in both modes")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")); ap.add_argument("--model")
    ap.add_argument("--shard", default="0/1", help="i/n: answer every n-th remaining question starting at i")
    ap.add_argument("--done-from", type=Path, nargs="*", default=[], help="extra output files whose (doc, qi) count as done")
    a = ap.parse_args(argv)
    max_tokens = a.max_tokens or DEFAULT_MAX_TOKENS[a.mode]
    spec = ANSWERERS[a.answerer]; model = a.model or spec["served"]
    done = {(r["doc_id"], r["qi"]) for r in read_jsonl(a.out)}
    for extra in a.done_from: done |= {(r["doc_id"], r["qi"]) for r in read_jsonl(extra)}
    todo = [q for q in questions(read_jsonl(a.writer)) if (q["doc_id"], q["qi"]) not in done]
    si, sn = (int(x) for x in a.shard.split("/")); todo = todo[si::sn]
    print(f"answerer {a.answerer} ({a.mode}): {len(done)} done, {len(todo)} to answer", flush=True)
    client = ChatClient(a.base_url, model); started = time.time(); n = 0
    extra = dict(spec["extra"])
    if a.mode == "distribution" and spec["family"].startswith("qwen"):
        extra.setdefault("chat_template_kwargs", {"enable_thinking": True})  # thinking on; the reasoning parser keeps it out of content
    base = {"answerer": a.answerer, "model": model, "repo": spec["repo"], "revision": spec["revision"]}

    def work(q):
        if a.mode == "distribution":
            messages = [{"role": "system", "content": DISTRIBUTION_SYSTEM}, {"role": "user", "content": distribution_prompt(q["state"], q["field"])}]
            raw = client.complete(messages, temperature=0.0, max_tokens=max_tokens, extra=extra, response_format=distribution_schema(q["field"]))
            value, probs, rationale, err = parse_distribution(raw, q["field"])
            return {"doc_id": q["doc_id"], "qi": q["qi"], **base, "mode": "distribution", "value": value, "probs": probs, "rationale": rationale,
                    "raw": raw[:4000], "parse_error": err}
        messages = [{"role": "system", "content": ANSWERER_SYSTEM}, {"role": "user", "content": answerer_prompt(q["state"], q["field"])}]
        raw = client.complete(messages, temperature=0.0, max_tokens=max_tokens, json_mode=True, extra=extra)
        value, conf, err = parse_answer(raw, q["field"])
        return {"doc_id": q["doc_id"], "qi": q["qi"], **base, "raw": raw[:4000], "value": value, "confidence": conf, "parse_error": err}

    for q, result in run_parallel(todo, work, workers=a.workers):
        n += 1
        if isinstance(result, Exception):
            if a.mode == "distribution":
                row = {"doc_id": q["doc_id"], "qi": q["qi"], **base, "mode": "distribution", "value": None, "probs": None, "rationale": None,
                       "raw": "", "parse_error": f"exception: {result}"}
            else:
                row = {"doc_id": q["doc_id"], "qi": q["qi"], "answerer": a.answerer, "model": model, "raw": "", "value": None,
                       "confidence": None, "parse_error": f"exception: {result}"}
            append_jsonl(a.out, row)
        else:
            append_jsonl(a.out, result)
        if n % 200 == 0:
            rate = n / max(time.time() - started, 1e-6)
            print(f"answerer {a.answerer}: {n}/{len(todo)}, {rate*60:.0f}/min, eta {(len(todo)-n)/max(rate,1e-6)/60:.0f} min", flush=True)
    print(f"answerer {a.answerer} done: {n} answered -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
