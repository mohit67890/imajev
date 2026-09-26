#!/usr/bin/env python3
"""A local stand-in for the Azure Foundry OpenAI-compatible endpoint, for tests and dry runs of the phase-3 streaming tools.

Mimics what matters from the measured deployment: POST <base>/chat/completions, `api-key` header (401 otherwise, and the 401 body
ECHOES the key it received, so tests prove the callers redact it), `chat_template_kwargs` rejected with 400, response_format
json_schema honoured, thinking returned in `reasoning_content`, usage tokens. Fault injection: the first N requests get 429
(with Retry-After) or 500. Every request body is recorded (never the key) for assertions.

    .venv/bin/python scripts/p3/mock_azure.py --port 8901 --key-file /path/to/fake-key --policy gold
    .venv/bin/python scripts/p3/azure_teacher.py run --base-url http://127.0.0.1:8901/openai/v1 --key-file /path/to/fake-key ...

Policies for distribution requests (response_format name "decision_distribution"):
    first    0.9 on the first label                       unknown  0.9 on "unknown"
    gold     0.9 on the label the test registered for that question text (answers={question: label}), else first
    custom   a callable(body, labels) -> {label: p} set on the server object
    realistic  a dry-run teacher: looks the item's gold up from --gold-from candidate/queue files (keyed by document + question),
             answers it with probability --accuracy (else another label), with a peaked but not one-hot distribution; a share of
             responses are unparseable (--parse-error-rate) or truncated (--truncation-rate: finish_reason "length", empty
             content); completion tokens are log-normal around --completion-tokens. Deterministic per (prompt, seed).
Writer requests ("decision_variant") get a mechanical variant: digits shifted by one, words reversed, answer = the verified one
(or "unknown" for an unanswerable-variant task). Review requests ("review_verdict", scripts/p3/kimi_review.py) answer the gold
(realistic policy, with probability --accuracy) or the first label, and flag "wrong_answer" when they differ from the gold.

    # dry run of the whole streamed session (scripts/p3 + reports/phase3/dry-run.md)
    .venv/bin/python scripts/p3/mock_azure.py --port 8901 --key-file $S/key --policy realistic \
        --gold-from "$S/pool/shards/*.jsonl" --gold-from "$S/stream/queue*.jsonl" --latency 0.05
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockAzure:
    def __init__(self, key: str, host: str = "127.0.0.1", port: int = 0, policy: str = "first", completion_tokens: int = 1800,
                 fail_429: int = 0, fail_500: int = 0, latency: float = 0.0, gold_from: list[str] | None = None,
                 accuracy: float = 0.9, parse_error_rate: float = 0.0, truncation_rate: float = 0.0,
                 served: list[str] | None = None):
        # key None = a LOCAL vLLM server (label-then-train pod): GET /v1/models lists `served`, chat_template_kwargs is accepted,
        # and a request that carries ANY credential header (api-key / Authorization) is refused with 400, so tests prove the pod
        # clients send none
        self.key, self.policy, self.completion_tokens = key, policy, completion_tokens
        self.served = list(served or ["teacher"])
        self.credential_headers_seen = 0
        self.fail_429, self.fail_500, self.latency = fail_429, fail_500, latency
        self.gold_from, self.accuracy = list(gold_from or []), accuracy
        self.parse_error_rate, self.truncation_rate = parse_error_rate, truncation_rate
        self.gold_map: dict[tuple, str] = {}
        self._gold_mtimes: dict[str, float] = {}
        self.stats = {"parse_error": 0, "truncated": 0, "gold_hit": 0, "gold_miss": 0}
        self.answers: dict[str, str] = {}
        self.custom = None
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.n = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silent
                pass

            def _send(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if outer.key is None and self.path.rstrip("/").endswith("/models"):
                    return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in outer.served]})
                return self._send(404, {"error": {"message": f"no route {self.path}"}})

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": f"no route {self.path}"}})
                if outer.key is None:
                    if self.headers.get("api-key") is not None or self.headers.get("authorization") is not None:
                        with outer.lock:
                            outer.credential_headers_seen += 1
                        return self._send(400, {"error": {"message": "credential header sent to a local server"}})
                    if body.get("model") not in outer.served:
                        return self._send(404, {"error": {"message": f"model {body.get('model')} not served"}})
                else:
                    got = self.headers.get("api-key")
                    if got != outer.key:
                        # like a careless upstream: echoes what it received
                        return self._send(401, {"error": {"message": f"invalid api-key {got}"}})
                with outer.lock:
                    outer.n += 1
                    n = outer.n
                    outer.requests.append(body)
                if outer.latency:
                    time.sleep(outer.latency)
                if n <= outer.fail_429:
                    return self._send(429, {"error": {"message": "rate limited"}}, {"retry-after": "0"})
                if n <= outer.fail_429 + outer.fail_500:
                    return self._send(500, {"error": {"message": "server error"}})
                if "chat_template_kwargs" in body and outer.key is not None:
                    return self._send(400, {"error": {"message": "chat_template_kwargs is not supported"}})
                return self._send(200, outer.respond(body))

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/openai/v1"

    @property
    def vllm_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "MockAzure":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ---------------------------------------------------------------------------------------------- responses
    @staticmethod
    def user_text(body: dict) -> str:
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
        return content

    # ---------------------------------------------------------------------------------------------- realistic policy
    @staticmethod
    def doc_key(text: str) -> tuple | None:
        m = re.search(r"DOCUMENT:\n(.*?)\n\nQUESTION: (.*?)\n", text, re.S)
        if not m:
            return None
        doc = re.sub(r"\n\n\(\d+ image\(s\) attached;.*?\)$", "", m.group(1), flags=re.S)     # kimi_review's image note
        return hashlib.sha1(doc.strip().encode()).hexdigest(), m.group(2).strip()

    def load_gold(self) -> None:
        """(re)load the gold map from --gold-from files whose mtime changed (queues grow during a run)."""
        for pat in self.gold_from:
            for path in glob.glob(pat):
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                if self._gold_mtimes.get(path) == mt:
                    continue
                self._gold_mtimes[path] = mt
                with open(path) as fh:
                    for line in fh:
                        if not line.strip().endswith("}"):
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row = row.get("item", row)
                        if "field" not in row or "state" not in row:
                            continue
                        doc = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False, indent=1)
                        g = row.get("gold")
                        lab = "unknown" if g is None else ("true" if g else "false") if isinstance(g, bool) else str(g)
                        self.gold_map[(hashlib.sha1(doc.strip().encode()).hexdigest(), row["field"]["question"].strip())] = lab

    def gold_of(self, text: str) -> str | None:
        k = self.doc_key(text)
        if k is None:
            return None
        if k not in self.gold_map:
            with self.lock:
                self.load_gold()
        g = self.gold_map.get(k)
        self.stats["gold_hit" if g else "gold_miss"] += 1
        return g

    def rng(self, body: dict, salt: str = "") -> random.Random:
        text = self.user_text(body)
        return random.Random(hashlib.sha1(f"{salt}|{body.get('seed')}|{body['messages'][0].get('content', '')[:40]}|{text}".encode()).hexdigest())

    def realistic(self, body: dict, labels: list[str]) -> dict:
        r = self.rng(body)
        gold = self.gold_of(self.user_text(body))
        if gold not in labels:
            gold = r.choice(labels)
        target = gold if r.random() < self.accuracy else r.choice([l for l in labels if l != gold] or labels)
        top = r.uniform(0.62, 0.97)
        rest = [l for l in labels if l != target]
        w = [r.random() for _ in rest]
        z = sum(w) or 1.0
        probs = {l: round((1 - top) * x / z, 4) for l, x in zip(rest, w)}
        probs[target] = round(top, 4)
        return probs

    def distribution(self, body: dict, labels: list[str]) -> dict:
        if self.custom is not None:
            return self.custom(body, labels)
        if self.policy == "realistic":
            return self.realistic(body, labels)
        text = self.user_text(body)
        target = labels[0]
        if self.policy == "unknown":
            target = "unknown"
        elif self.policy == "gold":
            q = re.search(r"QUESTION: (.*)", text)
            question = q.group(1).strip() if q else ""
            target = self.answers.get(question, labels[0])
        rest = [l for l in labels if l != target]
        probs = {l: 0.1 / max(1, len(rest)) for l in rest}
        probs[target] = 0.9
        return probs

    def variant(self, body: dict) -> dict:
        text = self.user_text(body)
        doc = re.search(r"ORIGINAL DOCUMENT:\n(.*?)\n\nQUESTION: (.*?)\n", text, re.S)
        state, question = (doc.group(1), doc.group(2)) if doc else ("", "")
        # numbers shifted and the word order reversed, so the variant shares no 13-gram with its parent
        shifted = " ".join(reversed(re.sub(r"\d+", lambda m: str(int(m.group(0)) + 1), state).split())) + " (variant)"
        gold = re.search(r"VERIFIED ANSWER: (.*)", text).group(1).strip()
        unknown = "unknown #" in text
        out = {"state": shifted, "question": question, "answer": "unknown" if unknown else gold,
               "change": "shifted every number by one"}
        opts = re.search(r"OPTIONS:\n((?:- .*\n?)+)", text)
        if opts:
            out["options"] = [{"key": m.group(1), "text": m.group(2)} for m in re.finditer(r"- (\S+): ([^\n(]*)", opts.group(1))]
        return out

    def review(self, body: dict, labels: list[str]) -> dict:
        r = self.rng(body)
        gold = self.gold_of(self.user_text(body)) if self.policy == "realistic" else None
        if gold not in labels:
            gold = labels[0]
        ans = gold if r.random() < self.accuracy else r.choice([l for l in labels if l != gold] or labels)
        issues = ["none"] if ans == gold else ["wrong_answer"]
        if ans == gold and r.random() < 0.03:
            issues = ["ambiguous"]
        return {"answer": ans, "confidence": round(r.uniform(0.6, 0.99), 3), "issues": issues, "note": "mock review"}

    def respond(self, body: dict) -> dict:
        rf = (body.get("response_format") or {}).get("json_schema") or {}
        schema = rf.get("schema") or {}
        finish = "stop"
        tokens = self.completion_tokens
        if self.policy == "realistic":
            r = self.rng(body, "tokens")
            tokens = int(min(body.get("max_tokens") or 12000, max(200, r.lognormvariate(math.log(self.completion_tokens) - 0.18, 0.6))))
        if rf.get("name") == "decision_distribution":
            labels = schema["properties"]["probabilities"]["required"]
            content = json.dumps({"probabilities": self.distribution(body, labels), "rationale": "The document states it."})
            if self.policy == "realistic":
                x = self.rng(body, "fault").random()
                if x < self.truncation_rate:
                    finish, content, tokens = "length", "", int(body.get("max_tokens") or 12000)
                    self.stats["truncated"] += 1
                elif x < self.truncation_rate + self.parse_error_rate:
                    content = 'The answer is {"probabilities": oops'
                    self.stats["parse_error"] += 1
        elif rf.get("name") == "decision_variant":
            content = json.dumps(self.variant(body))
        elif rf.get("name") == "review_verdict":
            content = json.dumps(self.review(body, schema["properties"]["answer"]["enum"]))
        else:
            content = json.dumps({"ok": True})
        prompt_chars = sum(len(json.dumps(m["content"])) for m in body.get("messages", []))
        return {"id": f"mock-{self.n}", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": content, "reasoning_content": "thinking " * 20}}],
                "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": tokens,
                          "total_tokens": prompt_chars // 4 + tokens}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--key-file", default=None, help="a FAKE key the clients will send (never the real one)")
    ap.add_argument("--vllm", action="store_true", help="act as a LOCAL vLLM server: no key accepted, GET /v1/models, "
                    "chat_template_kwargs allowed (label-then-train pod dry runs)")
    ap.add_argument("--served", action="append", default=None, help="--vllm: served model name(s) (default teacher)")
    ap.add_argument("--policy", choices=("first", "unknown", "gold", "realistic"), default="first")
    ap.add_argument("--completion-tokens", type=int, default=1800)
    ap.add_argument("--fail-429", type=int, default=0)
    ap.add_argument("--latency", type=float, default=0.0)
    ap.add_argument("--gold-from", action="append", default=[], help="realistic: glob of candidate / queue JSONL (repeatable)")
    ap.add_argument("--accuracy", type=float, default=0.9, help="realistic: share of answers equal to the gold")
    ap.add_argument("--parse-error-rate", type=float, default=0.01)
    ap.add_argument("--truncation-rate", type=float, default=0.015)
    a = ap.parse_args(argv)
    if not a.vllm and not a.key_file:
        ap.error("--key-file is required (or --vllm for a keyless local server)")
    key = None if a.vllm else open(a.key_file).read().strip()
    m = MockAzure(key, port=a.port, policy=a.policy, completion_tokens=a.completion_tokens, fail_429=a.fail_429,
                  latency=a.latency, gold_from=a.gold_from, accuracy=a.accuracy, parse_error_rate=a.parse_error_rate,
                  truncation_rate=a.truncation_rate, served=a.served).start()
    print(f"mock {'vllm' if a.vllm else 'azure'} at {m.vllm_url if a.vllm else m.base_url}", flush=True)
    try:
        m.thread.join()
    except KeyboardInterrupt:
        m.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
