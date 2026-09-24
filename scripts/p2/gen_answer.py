"""Stage 2 of decision-p2: an ANSWERER model answers every writer question independently (document + question only).

    OPENAI_BASE_URL=... python scripts/p2/gen_answer.py --answerer qwen  --writer data/decision-p2/teacher/writer.jsonl --out data/decision-p2/teacher/answers-qwen.jsonl
    OPENAI_BASE_URL=... python scripts/p2/gen_answer.py --answerer gptoss --writer ... --out data/decision-p2/teacher/answers-gptoss.jsonl

Resumable by (doc_id, question index). Records the raw reply, the parsed value and the parse error, never the intended answer.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import ANSWERERS, ChatClient, WQuestion, append_jsonl, parse_answer, read_jsonl, run_parallel, to_field
from families import ANSWERER_SYSTEM, answerer_prompt


def questions(writer_rows: list[dict]):
    for r in writer_rows:
        for qi, q in enumerate(r["output"]["questions"]):
            field, _ = to_field(WQuestion.model_validate(q), field_id="decision")
            yield {"doc_id": r["doc_id"], "qi": qi, "state": r["output"]["document"], "field": field}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", choices=sorted(ANSWERERS), required=True)
    ap.add_argument("--writer", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")); ap.add_argument("--model")
    ap.add_argument("--shard", default="0/1", help="i/n: answer every n-th remaining question starting at i")
    ap.add_argument("--done-from", type=Path, nargs="*", default=[], help="extra output files whose (doc, qi) count as done")
    a = ap.parse_args(argv)
    spec = ANSWERERS[a.answerer]; model = a.model or spec["served"]
    done = {(r["doc_id"], r["qi"]) for r in read_jsonl(a.out)}
    for extra in a.done_from: done |= {(r["doc_id"], r["qi"]) for r in read_jsonl(extra)}
    todo = [q for q in questions(read_jsonl(a.writer)) if (q["doc_id"], q["qi"]) not in done]
    si, sn = (int(x) for x in a.shard.split("/")); todo = todo[si::sn]
    print(f"answerer {a.answerer}: {len(done)} done, {len(todo)} to answer", flush=True)
    client = ChatClient(a.base_url, model); started = time.time(); n = 0

    def work(q):
        messages = [{"role": "system", "content": ANSWERER_SYSTEM}, {"role": "user", "content": answerer_prompt(q["state"], q["field"])}]
        raw = client.complete(messages, temperature=0.0, max_tokens=a.max_tokens, json_mode=True, extra=spec["extra"])
        value, conf, err = parse_answer(raw, q["field"])
        return {"doc_id": q["doc_id"], "qi": q["qi"], "answerer": a.answerer, "model": model, "repo": spec["repo"], "revision": spec["revision"],
                "raw": raw[:4000], "value": value, "confidence": conf, "parse_error": err}

    for q, result in run_parallel(todo, work, workers=a.workers):
        n += 1
        if isinstance(result, Exception):
            append_jsonl(a.out, {"doc_id": q["doc_id"], "qi": q["qi"], "answerer": a.answerer, "model": model, "raw": "", "value": None,
                                 "confidence": None, "parse_error": f"exception: {result}"})
        else:
            append_jsonl(a.out, result)
        if n % 200 == 0:
            rate = n / max(time.time() - started, 1e-6)
            print(f"answerer {a.answerer}: {n}/{len(todo)}, {rate*60:.0f}/min, eta {(len(todo)-n)/max(rate,1e-6)/60:.0f} min", flush=True)
    print(f"answerer {a.answerer} done: {n} answered -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
