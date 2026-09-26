#!/usr/bin/env python3
"""Phase-3 Stage 2 teacher runner on Azure Foundry (docs/phase-3-plan.md: Stage 2 verification table, "Streamed mining + teacher",
"Measured: Azure teacher test").

Runs ON THE MAC so the key stays local. Reads queue JSONL files (written by stream_coordinator.py and variants.py), asks the
Qwen3.6-35B-A3B deployment(s) for a full distribution over every label + unknown (scripts/p2/gen_answer.py prompts and schema),
adds a SECOND independent solve (paraphrased system prompt, temperature 0.6, another seed) only where the gold is not guaranteed
(gold_kind != "constructed"), applies the keep rules, and appends everything:

    <out>/solves.jsonl      every solve, ok or not (usage tokens, per-solve cost estimate, parse errors, truncations) - append-only
    <out>/results.jsonl     one verdict per item: keep / drop, rule, target, target_probs, `label`, the item itself - append-only
    <out>/instances.jsonl   instance ledger: {t, pool, instances} events; instance-hours x $7.91/h = the spend the cap reads
    <out>/status.json       live status (the coordinator reads it; state "cap_reached" stops its dispatch)
    <out>/pilot-gate.json   the automatic pilot gate's metrics and decision (below)
    <out>/kept.jsonl        `finalize`: kept TRAINING items (origin mine / variant) after the 25%-of-A cap on hard-gold labels
    <out>/heldout-results.jsonl  `finalize`: every verdict on a held-out item (origin "heldout": the heldout-flagged bucket; the
                            coordinator routes those rows here so they get an evaluation label, never a training one)

Kept rows carry the manifest label contract (scripts/p3/assembly_manifest.py), so build_manifest.py / kimi_review.py read them as is:
    "label": {"target": <candidate-space answer: option key | true/false | level int | null = unknown>,
              "probs": {<label>: p} | null (hard gold), "rationale": str | null, "target_kind": "teacher" | "gold", "review": null}
Image items (constructed or mined) use the 35B distribution + agreement with the constructed gold / stored label; there is no
9B distribution step.

Automatic pilot gate (on by default): once --gate-items results exist (default 1,000), it measures parse-error rate, truncation
rate (finish_reason "length"), keep rate per source, teacher agreement on constructed-gold items, mean completion tokens and $/item.
If parse errors > 2%, truncation > 3% or constructed-gold agreement < 60% (all configurable), it PAUSES dispatch (in-flight requests
finish, nothing new is sent; the instances keep billing), writes <out>/PILOT_GATE_ALERT.txt and a status line, and waits for the
resume flag <out>/RESUME_PILOT (`azure_teacher.py resume`). --no-pilot-gate turns it off.

Endpoint: POST https://<azure-foundry-endpoint>/openai/v1/chat/completions, header api-key (read from
~/.imajev-azure-key, never printed, logged or written), model = deployment name, NO chat_template_kwargs (rejected; thinking is
always on), response_format json_schema. Measured: ~2,100 output tok/s and ~4,200 items/h per H100 instance at 64 concurrent.

    # pilot: one deployment with one instance, 1k items, hard cap $20
    .venv/bin/python scripts/p3/azure_teacher.py instances --set 8          # when the portal shows the 8 instances Succeeded
    .venv/bin/python scripts/p3/azure_teacher.py run --deployment imajev-teacher:8 --follow      # hard cap $145, pilot gate on
    .venv/bin/python scripts/p3/azure_teacher.py resume                     # only after reading PILOT_GATE_ALERT.txt
    .venv/bin/python scripts/p3/azure_teacher.py instances --set 0          # after deleting the deployment in the portal
    .venv/bin/python scripts/p3/azure_teacher.py finalize                   # kept.jsonl + heldout-results.jsonl + summary

Crash-safe and idempotent: a restart skips items with a verdict, re-runs only missing solves, and repairs a torn last line.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import queue
import random
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (ROOT / "scripts/p2", HERE):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from gen_answer import DISTRIBUTION_SYSTEM, distribution_prompt, distribution_schema, parse_distribution  # noqa: E402
from p2_common import UNKNOWN_LABEL, argmax_label, field_labels  # noqa: E402

ENDPOINT = "https://<azure-foundry-endpoint>/openai/v1"
KEY_FILE = Path(os.path.expanduser("~/.imajev-azure-key"))
USD_PER_INSTANCE_HOUR = 7.91
TOK_PER_S = 2100.0           # measured output tok/s per instance at 64 concurrent requests
PREFILL_WEIGHT = 0.05        # a prompt token costs ~1/20 of a generated token of instance time (estimate only)
PER_INSTANCE_CONCURRENCY = 64
MAX_USD = 145.0              # owner decision 2026-09-25: the whole teacher run, pilot included
GATE = {"items": 1000, "max_parse_errors": 0.02, "max_truncation": 0.03, "min_constructed_keep": 0.60, "min_constructed_n": 30}
OUT = ROOT / "data/p3/teacher"
QUEUES = [ROOT / "data/p3/stream/queue.jsonl", ROOT / "data/p3/stream/queue-variants.jsonl"]

# The second solve gets a paraphrase of DISTRIBUTION_SYSTEM (same contract, different wording) so it is not a copy of solve 1.
SECOND_SYSTEM = (
    "Read the document and decide, for one typed question about it, how probable each candidate answer is. Rely on the document "
    "alone. Reason it through, then assign a probability to EVERY label listed, \"unknown\" included (choose \"unknown\" when the "
    "document leaves the answer undetermined, the question rests on a false premise, or none of the listed options is right). "
    "Make the probabilities add up to 1 and reflect how sure you actually are: concentrate nearly all the mass on one label when "
    "the document decides it, and spread it only as much as the document truly leaves open. Finish with a rationale of no more "
    "than two sentences citing the deciding evidence. Output exactly one JSON object {\"probabilities\": {<label>: <number>, ...}, "
    "\"rationale\": <string>} and nothing else."
)
SOLVE_SETTINGS = {1: {"system": "primary", "temperature": 0.0, "seed": 1},
                  2: {"system": "paraphrase", "temperature": 0.6, "seed": 2}}


# ------------------------------------------------------------------------------------------------ key handling

def load_key(path: Path | str | None = None) -> str:
    p = Path(path) if path else Path(os.environ.get("IMAJEV_AZURE_KEY_FILE") or KEY_FILE)
    key = p.read_text().strip()
    if not key:
        raise SystemExit(f"empty key file {p}")
    return key


class Redactor:
    """Every string that leaves this process through a log or a file goes through here."""

    def __init__(self, secret: str):
        self.secret = secret

    def __call__(self, text) -> str:
        text = str(text)
        return text.replace(self.secret, "<key>") if self.secret else text


# ------------------------------------------------------------------------------------------------ item -> prompt

def to_field(row: dict) -> dict:
    """Candidate field -> the gen_answer / p2 field shape (boolean / choice / ordinal)."""
    f = row["field"]
    if f["type"] == "noul":
        return {"id": "decision", "type": "boolean", "question": f["question"]}
    if f["type"] == "choice":
        opts = [{"value": o["key"], "description": (f"{o['text']}: {o['description']}" if o.get("description") else o.get("text") or o["key"])}
                for o in f["options"]]
        return {"id": "decision", "type": "choice", "question": f["question"], "options": opts}
    return {"id": "decision", "type": "ordinal", "question": f["question"],
            "levels": [{"value": l["value"], "description": l["description"]} for l in f["levels"]]}


def gold_label(row: dict) -> str | None:
    """The item's gold (or, for gold_kind "none", its claimed answer, e.g. a writer variant's intended answer) as a label."""
    g = row.get("gold")
    if g is None:
        return UNKNOWN_LABEL
    if isinstance(g, bool):
        return "true" if g else "false"
    return str(g)


def image_part(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError(f"image over 20 MiB: {path.name}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def build_messages(row: dict, solve: int, data_root: Path) -> list[dict]:
    field = to_field(row)
    system = DISTRIBUTION_SYSTEM if SOLVE_SETTINGS[solve]["system"] == "primary" else SECOND_SYSTEM
    text = distribution_prompt(row["state"], field)
    images = row.get("images") or []
    if images:
        note = ("The attached image(s) are part of the document (first image = reference, second = target when there are two).\n\n"
                if len(images) > 1 else "The attached image is part of the document.\n\n")
        content = [image_part(data_root / rel) for rel in images] + [{"type": "text", "text": note + text}]
    else:
        content = text
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def request_body(row: dict, solve: int, deployment: str, data_root: Path, max_tokens: int, attempt: int = 0) -> dict:
    s = SOLVE_SETTINGS[solve]
    return {"model": deployment, "messages": build_messages(row, solve, data_root), "temperature": s["temperature"],
            "seed": s["seed"] + 100 * attempt, "max_tokens": max_tokens, "response_format": distribution_schema(to_field(row))}


# ------------------------------------------------------------------------------------------------ HTTP client

class TransientError(Exception):
    pass


class AzureClient:
    """POST chat completions with retries/backoff on 429, 5xx, timeouts and connection errors (Retry-After honoured)."""

    def __init__(self, base_url: str, key: str, timeout: float = 900.0, max_attempts: int = 8, sleep=time.sleep):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key, self.timeout, self.max_attempts, self.sleep = key, timeout, max_attempts, sleep
        self.redact = Redactor(key)
        self.stats = Counter()

    def post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(self.url, data=data, headers={"api-key": self.key, "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                msg = self.redact(f"HTTP {e.code} " + e.read()[:300].decode(errors="ignore"))
                if e.code == 429 or e.code >= 500:
                    self.stats[f"http_{e.code}"] += 1
                    retry_after = e.headers.get("retry-after") if e.headers else None
                    try:
                        wait = float(retry_after) if retry_after else None
                    except ValueError:
                        wait = None
                    last = msg
                    self.sleep(min(120.0, wait if wait is not None else (2 ** attempt) * (0.5 + random.random())))
                    continue
                raise RuntimeError(msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self.stats["conn_error"] += 1
                last = self.redact(f"{type(e).__name__}: {str(e)[:200]}")
                self.sleep(min(120.0, (2 ** attempt) * (0.5 + random.random())))
                continue
        raise TransientError(f"gave up after {self.max_attempts} attempts: {last}")


# ------------------------------------------------------------------------------------------------ cost ledger

class Ledger:
    """Instance-hour spend from an append-only event log: {t, pool, instances}. Instances bill while they exist, whether or not
    this runner is up, so the integral runs from each event to the next (or to now)."""

    def __init__(self, path: Path, usd_per_instance_hour: float = USD_PER_INSTANCE_HOUR, clock=time.time):
        self.path, self.rate, self.clock = path, usd_per_instance_hour, clock

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip() and l.endswith("}")]

    def current(self, pool: str = "teacher") -> int | None:
        ev = [e for e in self.events() if e.get("pool", "teacher") == pool]
        return ev[-1]["instances"] if ev else None

    def record(self, instances: int, pool: str = "teacher", t: float | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"t": self.clock() if t is None else t, "pool": pool, "instances": int(instances)}) + "\n")

    def spend(self, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        total = 0.0
        by_pool = defaultdict(list)
        for e in self.events():
            by_pool[e.get("pool", "teacher")].append(e)
        for evs in by_pool.values():
            evs.sort(key=lambda e: e["t"])
            for a, b in zip(evs, evs[1:] + [{"t": now}]):
                total += a["instances"] * max(0.0, min(b["t"], now) - a["t"]) / 3600 * self.rate
        return total


def est_usd(usage: dict | None, tok_per_s: float = TOK_PER_S, rate: float = USD_PER_INSTANCE_HOUR) -> float:
    if not usage:
        return 0.0
    work = (usage.get("completion_tokens") or 0) + PREFILL_WEIGHT * (usage.get("prompt_tokens") or 0)
    return work * rate / (tok_per_s * 3600)


# ------------------------------------------------------------------------------------------------ keep rules

def rule_for(q: dict) -> str:
    """Which row of the plan's verification table applies to a queue row."""
    item, variant = q["item"], q.get("variant") or {}
    if variant.get("kind") == "unknown":
        return "unknown_variant"
    kind = item.get("gold_kind")
    if item.get("images"):
        return "image_constructed" if kind == "constructed" else "image_mined"
    if kind == "constructed":
        return {"A": "A", "D": "D"}.get(item["source"], "constructed")
    if variant.get("kind") == "perturb":
        return "writer_variant"
    if item["source"] == "B" and kind == "dataset":
        return "B"
    return "C"


def needs_two(q: dict) -> bool:
    return q["item"].get("gold_kind") != "constructed"


def mean_probs(solves: list[dict]) -> dict:
    labs = list(solves[0]["probs"])
    m = {k: sum(s["probs"].get(k, 0.0) for s in solves) / len(solves) for k in labs}
    z = sum(m.values()) or 1.0
    return {k: round(v / z, 6) for k, v in m.items()}


def first_solve_passes(q: dict, s1: dict) -> bool:
    """Whether solve 1 alone already fails the item's rule (then no second solve is paid for)."""
    rule, field, gold = rule_for(q), to_field(q["item"]), gold_label(q["item"])
    if not s1.get("ok"):
        return False
    lab = argmax_label(s1["probs"], field)
    if rule == "unknown_variant":
        return s1["probs"].get(UNKNOWN_LABEL, 0.0) > 0.5
    if rule in ("B", "writer_variant", "image_mined") or (rule == "C" and gold is not None):
        return lab == gold
    return True


def keep_decision(q: dict, solves: list[dict]) -> dict:
    """Apply the plan's Stage-2 verification table. `solves` are this item's ok-or-failed solves, in solve order."""
    item, rule = q["item"], rule_for(q)
    field, gold = to_field(item), gold_label(item)
    good = [s for s in solves if s.get("ok")]
    labels = [argmax_label(s["probs"], field) for s in good]
    out = {"rule": rule, "gold_label": gold, "labels": labels, "keep": False, "target": None, "target_probs": None,
           "hard_label": None, "review_first": False, "reason": None}
    if not good or len(good) < len(solves):
        out["reason"] = "solve_failed"
        return out
    s1 = good[0]
    if rule in ("A", "D", "constructed", "image_constructed"):
        if labels[0] == gold:
            out.update(keep=True, target="hard_gold+dist" if rule == "image_constructed" else "teacher_dist",
                       target_probs=s1["probs"], hard_label=gold if rule == "image_constructed" else None, reason="teacher_matches_gold")
        elif rule == "A":
            # teacher wrong but the gold is certain: a hard label, capped at 25% of A at finalize; review first as a generator bug
            out.update(keep=True, target="hard_gold", target_probs={l: float(l == gold) for l in field_labels(field)},
                       hard_label=gold, review_first=True, reason="hard_gold_teacher_disagrees")
        else:
            out["reason"] = "teacher_disagrees_with_constructed_gold"
        return out
    need = 2 if needs_two(q) else 1
    if rule == "unknown_variant":
        # programmatic unknown variants (constructed) take one solve, writer-made ones two; keep only when every solve puts
        # most of its mass on unknown
        if not all(s["probs"].get(UNKNOWN_LABEL, 0.0) > 0.5 for s in good):
            out["reason"] = "teacher_not_unknown"
        elif len(good) < need:
            out["reason"] = "missing_second_solve"
        else:
            out.update(keep=True, target="teacher_dist", target_probs=mean_probs(good), reason="teacher_mass_on_unknown")
        return out
    if len(good) < 2:
        out["reason"] = "first_solve_failed_rule" if not first_solve_passes(q, s1) else "missing_second_solve"
        return out
    if labels[0] != labels[1]:
        out["reason"] = "solves_disagree"
        return out
    agreed = labels[0]
    if rule in ("B", "writer_variant", "image_mined") and agreed != gold:
        out["reason"] = "solves_disagree_with_gold"
        return out
    if rule == "C" and gold is not None and agreed != gold:
        out["reason"] = "gold_conflict"
        return out
    out.update(keep=True, target="teacher_dist", target_probs=mean_probs(good), reason="two_solves_agree")
    if rule == "image_mined":
        # owner decision: no 9B labelling pass; the target is the 35B distribution, kept only because both solves agree
        # with the stored (human / 9B) label
        out.update(target="hard_gold+dist", hard_label=gold)
    return out


def label_for(item: dict, verdict: dict, rationales: list) -> dict | None:
    """The manifest label contract (scripts/p3/assembly_manifest.py) for a kept verdict; None for a dropped one.
    Every keep rule requires the teacher's argmax (or, for hard gold, the constructed answer) to equal the item's gold, so the
    candidate-space target is the item's gold (null = unknown)."""
    if not verdict.get("keep"):
        return None
    if verdict.get("target") == "hard_gold":
        return {"target": item.get("gold"), "probs": None, "rationale": None, "target_kind": "gold", "review": None}
    rationale = next((r for r in rationales or [] if isinstance(r, str) and r.strip()), None)
    return {"target": item.get("gold"), "probs": verdict.get("target_probs"), "rationale": rationale, "target_kind": "teacher",
            "review": None}


# ------------------------------------------------------------------------------------------------ JSONL helpers

def repair_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
        fh.truncate(data.rfind(b"\n") + 1)


def read_complete(path: Path, start: int = 0) -> tuple[list[dict], int]:
    """Complete lines from byte offset `start`; returns (rows, new offset). A partial trailing line is left for next time."""
    if not path.exists():
        return [], start
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read()
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows, start + end


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ runner

def parse_deployments(specs: list[str]) -> list[tuple[str, int]]:
    out = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if part:
                name, _, n = part.partition(":")
                out.append((name, int(n or 1)))
    return out


PAUSED = "paused_pilot_gate"


def ist(t: float | None = None) -> str:
    """Wall-clock time in IST (UTC+5:30), whatever the machine's zone: the owner reads every time in IST."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime((time.time() if t is None else t) + 19800))
CONSTRUCTED_RULES = ("A", "D", "constructed", "image_constructed")


class PilotGate:
    """Automatic pilot gate: evaluated once, when --gate-items results exist; a failed gate pauses dispatch until the owner
    creates <out>/RESUME_PILOT (`azure_teacher.py resume`). The decision persists in <out>/pilot-gate.json across restarts."""

    def __init__(self, out: Path, a):
        self.out = out
        self.path, self.alert_path, self.flag = out / "pilot-gate.json", out / "PILOT_GATE_ALERT.txt", out / "RESUME_PILOT"
        self.th = {k: getattr(a, f"gate_{k}", v) for k, v in GATE.items()}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"decision": "pending"}

    def paused(self) -> bool:
        return self.state.get("decision") == "paused"

    def summary(self) -> str:
        d = self.state.get("decision")
        return f"pending ({self.th['items']} items)" if d == "pending" else d

    def metrics(self, runner) -> dict:
        results = sorted(runner.results.values(), key=lambda r: r.get("t", 0))
        ids = {r["id"] for r in results}
        solves = [x for x in read_complete(runner.solves_path)[0] if x["id"] in ids]
        responses = sum(x.get("responses", 1 if x.get("usage") else 0) for x in solves)
        trunc = sum(x.get("truncated", int(x.get("finish") == "length")) for x in solves)
        perr = sum(x.get("parse_errors", int(bool(x.get("parse_error")) and x.get("finish") != "length")) for x in solves)
        comp = [(x.get("usage") or {}).get("completion_tokens") or 0 for x in solves if x.get("usage")]
        by_src = defaultdict(lambda: [0, 0])
        for r in results:
            by_src[r["item"].get("source")][0] += 1
            by_src[r["item"].get("source")][1] += bool(r["keep"])
        cons = [r for r in results if r.get("rule") in CONSTRUCTED_RULES]
        agree = sum(1 for r in cons if r.get("reason") == "teacher_matches_gold")
        spend = runner.spend()
        return {"items": len(results), "solves": len(solves), "responses": responses,
                "parse_error_rate": round(perr / max(1, responses), 4), "truncation_rate": round(trunc / max(1, responses), 4),
                "keep_rate_by_source": {k: round(v[1] / v[0], 4) for k, v in sorted(by_src.items(), key=str)},
                "keep_rate": round(sum(bool(r["keep"]) for r in results) / max(1, len(results)), 4),
                "constructed_items": len(cons), "constructed_keep_rate": round(agree / len(cons), 4) if cons else None,
                "mean_completion_tokens": round(sum(comp) / max(1, len(comp))),
                "usd_per_item": round(spend / max(1, len(results)), 4),
                "token_estimate_usd_per_item": round(runner.token_usd / max(1, len(results)), 4),
                "spend_usd": round(spend, 2)}

    def evaluate(self, m: dict) -> list[str]:
        why = []
        if m["parse_error_rate"] > self.th["max_parse_errors"]:
            why.append(f"parse errors {m['parse_error_rate']:.1%} > {self.th['max_parse_errors']:.0%}")
        if m["truncation_rate"] > self.th["max_truncation"]:
            why.append(f"truncation {m['truncation_rate']:.1%} > {self.th['max_truncation']:.0%} (raise --max-tokens?)")
        if m["constructed_keep_rate"] is not None and m["constructed_items"] >= self.th["min_constructed_n"] \
                and m["constructed_keep_rate"] < self.th["min_constructed_keep"]:
            why.append(f"teacher agrees with constructed gold on {m['constructed_keep_rate']:.1%} < {self.th['min_constructed_keep']:.0%}"
                       " (generator bug or prompt problem: check review_first rows)")
        return why

    def _save(self):
        write_atomic(self.path, self.state)

    def step(self, runner, state: str) -> str:
        d = self.state.get("decision")
        if d == "paused":
            if self.flag.exists():
                used = self.out / f"RESUME_PILOT.used-{int(time.time())}"
                os.replace(self.flag, used)
                self.state.update(decision="resumed", resumed_at=time.time())
                self._save()
                with open(self.alert_path, "a") as fh:
                    fh.write(f"\nRESUMED by the owner at {ist()} IST (flag {used.name}).\n")
                print("PILOT GATE: resume flag found; dispatch resumes.", flush=True)
                return "running" if state == PAUSED else state
            return PAUSED if state in ("running", PAUSED) else state
        if d != "pending" or len(runner.results) < self.th["items"]:
            return state
        m = self.metrics(runner)
        why = self.evaluate(m)
        self.state = {"decision": "paused" if why else "pass", "t": time.time(), "metrics": m, "thresholds": self.th, "why": why}
        self._save()
        if not why:
            print(f"PILOT GATE PASS after {m['items']} items: {json.dumps(m)}", flush=True)
            return state
        note = runner.pause_note() if hasattr(runner, "pause_note") else (
                "Nothing new is sent to Azure. In-flight requests finish. THE INSTANCES KEEP BILLING while paused "
                f"({runner.total_instances} x ${runner.a.usd_per_instance_hour}/h).\n"
                "Decide now:\n"
                "  - fix and continue: `.venv/bin/python scripts/p3/azure_teacher.py resume` (the runner picks it up within seconds);\n"
                "  - or stop: Ctrl-C the runner, delete the deployment in the portal, `azure_teacher.py instances --set 0`.\n")
        text = ("PILOT GATE: DISPATCH PAUSED\n"
                f"time: {ist()} IST\n"
                f"after {m['items']} items: " + "; ".join(why) + "\n\n" + note + "\n"
                "metrics:\n" + json.dumps(m, indent=1) + "\n")
        self.alert_path.write_text(text)
        print(text, flush=True)
        return PAUSED if state == "running" else state


class Runner:
    # hooks for other OpenAI-compatible backends (scripts/p3/pod_teacher.py: local vLLM servers on the label-then-train pod, no key,
    # no Azure instance ledger); the Azure behaviour below is unchanged
    RECORD_INSTANCES = True

    def make_client(self, key: str, sleep):
        return AzureClient(self.a.base_url, key, timeout=self.a.timeout, max_attempts=self.a.max_attempts, sleep=sleep)

    def make_ledger(self, clock):
        return Ledger(self.out / "instances.jsonl", self.a.usd_per_instance_hour, clock)

    def dispatch_ready(self, deployment: str) -> bool:
        """Whether a worker of `deployment` may take a task now (a local server that is down takes none)."""
        return True

    def cost_text(self, s: dict) -> str:
        return (f"spend ${s['spend_usd']:.2f} / ${s['max_usd']:.0f} (inst-h ${s['instance_hour_usd']:.2f}, "
                f"tok-est ${s['token_estimate_usd']:.2f})")

    def __init__(self, a, key: str, clock=time.time, sleep=time.sleep):
        self.a, self.clock = a, clock
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.redact = Redactor(key)
        self.client = self.make_client(key, sleep)
        self.deployments = parse_deployments(a.deployment)
        self.total_instances = sum(n for _, n in self.deployments)
        self.ledger = self.make_ledger(clock)
        if self.RECORD_INSTANCES and self.ledger.current() != self.total_instances:
            start = a.billing_start if a.billing_start is not None and self.ledger.current() is None else None
            self.ledger.record(self.total_instances, t=start)
        self.solves_path, self.results_path = self.out / "solves.jsonl", self.out / "results.jsonl"
        for p in (self.solves_path, self.results_path):
            repair_tail(p)
        self.solves = defaultdict(dict)            # id -> {solve_idx: solve row (the ok one, else the latest failure)}
        self.fail_counts = Counter()
        past = read_complete(self.solves_path)[0]
        for s in past:
            if s.get("ok") or s["solve"] not in self.solves[s["id"]]:
                self.solves[s["id"]][s["solve"]] = s
            if not s.get("ok"):
                self.fail_counts[(s["id"], s["solve"])] += 1
        self.results = {r["id"]: r for r in read_complete(self.results_path)[0]}
        self.items: dict[str, dict] = {}
        self.offsets = {Path(p): 0 for p in a.queue}
        self.pending: list[tuple[str, int]] = []          # (id, solve) waiting for dispatch, second solves first
        self.inflight: set[tuple[str, int]] = set()
        self.tasks: queue.Queue = queue.Queue()
        self.done_q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.new_items = 0
        self.counts = Counter()
        self.tokens = Counter()
        self.token_usd = sum(s.get("est_usd", 0.0) for s in past)     # every attempt ever recorded, failed ones included
        self.started = clock()
        self.state = "running"
        self.fatal = None
        self.gate = PilotGate(self.out, a) if getattr(a, "pilot_gate", True) else None
        if self.gate is not None and self.gate.paused():
            self.state = PAUSED
        self.sw = open(self.solves_path, "a")
        self.rw = open(self.results_path, "a")

    # -- bookkeeping
    def spend(self) -> float:
        return max(self.ledger.spend(), self.token_usd)

    def cap_reached(self) -> bool:
        return self.spend() >= self.a.max_usd

    def _append(self, fh, row):
        fh.write(self.redact(json.dumps(row, ensure_ascii=False)) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def poll_queues(self) -> None:
        for path in list(self.offsets):
            rows, self.offsets[path] = read_complete(path, self.offsets[path])
            for q in rows:
                qid = q["id"]
                if qid in self.items or qid in self.results:
                    continue
                self.items[qid] = q
                have = self.solves.get(qid, {})
                if 1 in have and have[1].get("ok"):
                    if needs_two(q) and first_solve_passes(q, have[1]):
                        if 2 in have and have[2].get("ok"):
                            self.finish(qid)
                        else:
                            self.pending.insert(0, (qid, 2))
                    else:
                        self.finish(qid)
                else:
                    self.pending.append((qid, 1))

    def finish(self, qid: str) -> None:
        q = self.items[qid]
        solves = [self.solves[qid][k] for k in sorted(self.solves.get(qid, {}))]
        verdict = keep_decision(q, solves)
        rationales = [s.get("rationale") for s in solves]
        row = {"id": qid, **verdict, "label": label_for(q["item"], verdict, rationales),
               "est_usd": round(sum(s.get("est_usd", 0.0) for s in solves), 6),
               "solve_models": [s.get("deployment") for s in solves], "rationales": rationales,
               "origin": q.get("origin", "mine"), "variant": q.get("variant"), "mine": q.get("mine"), "item": q["item"],
               "t": self.clock()}
        self._append(self.rw, row)
        self.results[qid] = row
        self.counts["kept" if verdict["keep"] else "dropped"] += 1
        self.counts[f"rule:{verdict['rule']}:{'keep' if verdict['keep'] else verdict['reason']}"] += 1

    def dispatch(self) -> None:
        capacity = self.a.per_instance_concurrency * self.total_instances
        while self.pending and len(self.inflight) < capacity and not self.stop.is_set():
            if self.cap_reached():
                self.state = "cap_reached"
                return
            qid, solve = self.pending[0]
            if solve == 1 and self.a.max_items is not None and self.new_items >= self.a.max_items:
                # only second solves of already-started items may still go out
                second = next((i for i, (_, s) in enumerate(self.pending) if s == 2), None)
                if second is None:
                    return
                qid, solve = self.pending.pop(second)
            else:
                self.pending.pop(0)
            if solve == 1:
                self.new_items += 1
            self.inflight.add((qid, solve))
            self.tasks.put((qid, solve))

    def worker(self, deployment: str) -> None:
        data_root = Path(self.a.data_root)
        while not self.stop.is_set():
            if not self.dispatch_ready(deployment):
                time.sleep(0.5)
                continue
            try:
                qid, solve = self.tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            self.done_q.put(self.solve_one(qid, solve, deployment, data_root))

    def solve_one(self, qid: str, solve: int, deployment: str, data_root: Path) -> dict:
        q = self.items[qid]
        t0 = time.time()
        row = {"id": qid, "solve": solve, "deployment": deployment, **SOLVE_SETTINGS[solve], "ok": False}
        row.update(responses=0, parse_errors=0, truncated=0)
        try:
            for attempt in range(2):          # one retry with a new seed on a parse error or truncation
                body = request_body(q["item"], solve, deployment, data_root, self.a.max_tokens, attempt)
                resp = self.client.post(body)
                choice = resp["choices"][0]
                msg = choice.get("message") or {}
                raw = msg.get("content") or ""
                value, probs, rationale, err = parse_distribution(raw, to_field(q["item"]))
                usage = resp.get("usage") or {}
                # per HTTP-200 response: a truncation (finish_reason "length") and a parse error of a finished answer
                # are counted apart, for the pilot gate
                row["responses"] += 1
                if choice.get("finish_reason") == "length":
                    row["truncated"] += 1
                elif probs is None:
                    row["parse_errors"] += 1
                row.update(seed=body["seed"], finish=choice.get("finish_reason"), usage=usage, attempt=attempt,
                           reasoning_chars=len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
                           probs=probs, rationale=rationale, parse_error=err, raw=None if probs else raw[:1500],
                           est_usd=round(row.get("est_usd", 0.0) + est_usd(usage, self.a.tok_per_s, self.a.usd_per_instance_hour), 8))
                if probs is not None:
                    row.update(ok=True, label=argmax_label(probs, to_field(q["item"])))
                    break
        except Exception as exc:  # noqa: BLE001 - recorded on the solve row, redacted
            row["error"] = self.redact(f"{type(exc).__name__}: {str(exc)[:300]}")
        row["secs"] = round(time.time() - t0, 2)
        return row

    def collect(self) -> None:
        while True:
            try:
                row = self.done_q.get_nowait()
            except queue.Empty:
                return
            qid, solve = row["id"], row["solve"]
            row["t"] = self.clock()
            self._append(self.sw, row)
            self.inflight.discard((qid, solve))
            self.token_usd += row.get("est_usd", 0.0)
            u = row.get("usage") or {}
            self.tokens["completion"] += u.get("completion_tokens") or 0
            self.tokens["prompt"] += u.get("prompt_tokens") or 0
            self.counts["solves_ok" if row["ok"] else "solves_failed"] += 1
            self.counts[f"solve{solve}_done"] += 1
            if row["ok"] or solve not in self.solves[qid]:
                self.solves[qid][solve] = row
            q = self.items[qid]
            if not row["ok"]:
                err = row.get("error") or ""
                if any(f"HTTP {c}" in err for c in (401, 403, 404)):
                    # wrong key, deployment name or endpoint: stop instead of dropping the whole queue as failed
                    self.state = "fatal"
                    self.fatal = err
                    self.pending.insert(0, (qid, solve))
                    continue
                self.fail_counts[(qid, solve)] += 1
                if self.fail_counts[(qid, solve)] < self.a.solve_retries and "HTTP 4" not in (row.get("error") or ""):
                    self.pending.append((qid, solve))     # a transient failure: try again later
                    continue
                self.finish(qid)
                continue
            if solve == 1 and needs_two(q) and first_solve_passes(q, row):
                self.pending.insert(0, (qid, 2))
            else:
                self.finish(qid)

    def status(self, final=False) -> dict:
        el = max(1e-6, self.clock() - self.started)
        s = {"state": self.state, "t": self.clock(), "spend_usd": round(self.spend(), 2),
             "instance_hour_usd": round(self.ledger.spend(), 2), "token_estimate_usd": round(self.token_usd, 2),
             "max_usd": self.a.max_usd, "instances": self.total_instances, "items_seen": len(self.items),
             "results": len(self.results), "pending": len(self.pending), "inflight": len(self.inflight),
             "kept": sum(1 for r in self.results.values() if r["keep"]),
             "dropped": sum(1 for r in self.results.values() if not r["keep"]),
             "completion_tok_per_s": round(self.tokens["completion"] / el),
             "items_per_hour": round((self.counts["kept"] + self.counts["dropped"]) * 3600 / el),
             "http": dict(self.client.stats), "by_rule": {k[5:]: v for k, v in sorted(self.counts.items()) if k.startswith("rule:")},
             "second_solves": sum(1 for v in self.solves.values() if 2 in v),
             "by_origin": dict(Counter(r.get("origin", "mine") for r in self.results.values())),
             "pilot_gate": self.gate.summary() if self.gate is not None else "off"}
        write_atomic(self.out / "status.json", s)
        line = (f"[teacher {time.strftime('%H:%M:%S')}] {s['state']} results {s['results']} (kept {s['kept']}, dropped "
                f"{s['dropped']}) pending {s['pending']} inflight {s['inflight']} | {s['completion_tok_per_s']} tok/s "
                f"{s['items_per_hour']} items/h | " + self.cost_text(s) + (f" http {s['http']}" if s["http"] else "")
                + (f" | PILOT GATE PAUSED: read {self.gate.alert_path}, then `azure_teacher.py resume`"
                   if s["state"] == PAUSED else f" | pilot gate {self.gate.summary()}" if self.gate is not None else ""))
        print(self.redact(line), flush=True)
        return s

    def closed(self) -> bool:
        """Every queue has a <queue>.closed marker (the coordinator and variants.py write them when they are finished)."""
        return all((p.parent / (p.name + ".closed")).exists() for p in self.offsets)

    def run(self) -> dict:
        threads = []
        for name, n in self.deployments:
            for _ in range(n * self.a.per_instance_concurrency):
                t = threading.Thread(target=self.worker, args=(name,), daemon=True)
                t.start()
                threads.append(t)
        last_status = last_poll = 0.0
        try:
            while True:
                now = time.time()
                if now - last_poll >= (self.a.poll_secs if self.items else 1.0):
                    self.poll_queues()
                    last_poll = now
                self.collect()
                if self.gate is not None:
                    self.state = self.gate.step(self, self.state)
                if self.state == "running":
                    self.dispatch()
                if self.cap_reached() and self.state in ("running", PAUSED):
                    self.state = "cap_reached"
                if now - last_status >= self.a.status_secs:
                    self.status()
                    last_status = now
                idle = not self.pending and not self.inflight
                limit_hit = (self.a.max_items is not None and self.new_items >= self.a.max_items
                             and not self.inflight and not any(s == 2 for _, s in self.pending))
                if self.state == "fatal" and not self.inflight:
                    print(f"FATAL: {self.fatal} - check the deployment name, endpoint and key file; nothing was dropped.",
                          flush=True)
                    break
                if self.state == "cap_reached" and not self.inflight:
                    print("COST CAP REACHED: dispatch stopped. Delete the Azure deployments in the Foundry portal now, then run "
                          "`azure_teacher.py instances --set 0`.", flush=True)
                    break
                if limit_hit:
                    self.state = "max_items_reached"
                    break
                if idle and self.state != PAUSED and (not self.a.follow or self.closed()):
                    self.poll_queues()
                    if not self.pending:
                        self.state = "done"
                        break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.state = "interrupted"
        finally:
            self.stop.set()
            self.collect()
            s = self.status(final=True)
            self.sw.close()
            self.rw.close()
        return s


# ------------------------------------------------------------------------------------------------ finalize

def finalize(out: Path, hard_gold_share: float = 0.25) -> dict:
    results = {r["id"]: r for r in read_complete(out / "results.jsonl")[0]}
    for r in results.values():            # rows written before the label contract existed
        if "label" not in r and "item" in r:
            r["label"] = label_for(r["item"], r, r.get("rationales") or [])
    heldout = [r for r in results.values() if r.get("origin") == "heldout"]
    tmp = out / "heldout-results.jsonl.tmp"
    with open(tmp, "w") as fh:
        for r in heldout:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out / "heldout-results.jsonl")
    results = {k: r for k, r in results.items() if r.get("origin") != "heldout"}
    kept = [r for r in results.values() if r["keep"]]
    a_match = [r for r in kept if r["rule"] == "A" and r["target"] != "hard_gold"]
    a_hard = sorted((r for r in kept if r["rule"] == "A" and r["target"] == "hard_gold"),
                    key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
    allow = int(hard_gold_share / (1 - hard_gold_share) * len(a_match))    # hard <= 25% of all kept A
    capped = {r["id"] for r in a_hard[allow:]}
    rows = [r for r in kept if r["id"] not in capped]
    tmp = out / "kept.jsonl.tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out / "kept.jsonl")
    summary = {"results": len(results), "kept": len(rows), "hard_gold_A_kept": len(a_hard) - len(capped),
               "hard_gold_A_capped": len(capped), "by_rule": dict(Counter(r["rule"] for r in rows)),
               "by_source": dict(Counter(r["item"]["source"] for r in rows)),
               "review_first": sum(1 for r in rows if r.get("review_first")),
               "drop_reasons": dict(Counter(r["reason"] for r in results.values() if not r["keep"])),
               "heldout_results": len(heldout), "heldout_kept": sum(1 for r in heldout if r["keep"])}
    write_atomic(out / "kept-summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ CLI

def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stream the queue through the teacher")
    r.add_argument("--deployment", action="append", required=True,
                   help="deployment name[:instances]; repeat or comma-separate for several deployments")
    r.add_argument("--queue", action="append", default=None, help="queue JSONL (default: data/p3/stream/queue*.jsonl)")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--base-url", default=os.environ.get("IMAJEV_AZURE_BASE_URL", ENDPOINT))
    r.add_argument("--key-file", default=None, help="default ~/.imajev-azure-key (or $IMAJEV_AZURE_KEY_FILE)")
    r.add_argument("--data-root", default=str(ROOT / "data"))
    r.add_argument("--max-usd", type=float, default=MAX_USD,
                   help=f"HARD credit cap for the whole teacher run: dispatch stops once spend reaches it (default {MAX_USD:.0f})")
    r.add_argument("--usd-per-instance-hour", type=float, default=USD_PER_INSTANCE_HOUR)
    r.add_argument("--tok-per-s", type=float, default=TOK_PER_S, help="per-instance throughput for the per-item cost estimate")
    r.add_argument("--billing-start", type=float, default=None,
                   help="epoch seconds the instances started billing (first run only; default now)")
    r.add_argument("--per-instance-concurrency", type=int, default=PER_INSTANCE_CONCURRENCY)
    r.add_argument("--max-tokens", type=int, default=12000)
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--max-attempts", type=int, default=8, help="HTTP attempts per request (429/5xx/connection)")
    r.add_argument("--solve-retries", type=int, default=3, help="dispatches per solve before the item is dropped")
    r.add_argument("--max-items", type=int, default=None, help="new items this run (pilot: 1000)")
    r.add_argument("--follow", action="store_true", help="keep polling the queues until every <queue>.closed exists")
    r.add_argument("--poll-secs", type=float, default=15.0)
    r.add_argument("--status-secs", type=float, default=15.0)
    r.add_argument("--no-pilot-gate", dest="pilot_gate", action="store_false", help="turn the automatic pilot gate off")
    r.add_argument("--gate-items", type=int, default=GATE["items"], help="results before the gate is evaluated")
    r.add_argument("--gate-max-parse-errors", type=float, default=GATE["max_parse_errors"])
    r.add_argument("--gate-max-truncation", type=float, default=GATE["max_truncation"])
    r.add_argument("--gate-min-constructed-keep", type=float, default=GATE["min_constructed_keep"],
                   help="minimum share of constructed-gold items where the teacher's argmax equals the gold")
    r.add_argument("--gate-min-constructed-n", type=int, default=GATE["min_constructed_n"],
                   help="the constructed-gold rule applies only with at least this many constructed items")
    g = sub.add_parser("resume", help="release a pilot-gate pause (writes <out>/RESUME_PILOT)")
    g.add_argument("--out", default=str(OUT))
    i = sub.add_parser("instances", help="record an instance-count change (the cost ledger)")
    i.add_argument("--set", type=int, required=True)
    i.add_argument("--pool", default="teacher", help="teacher | writer")
    i.add_argument("--out", default=str(OUT))
    i.add_argument("--at", type=float, default=None, help="epoch seconds of the change (default now)")
    f = sub.add_parser("finalize", help="kept.jsonl with the 25%-of-A hard-gold cap")
    f.add_argument("--out", default=str(OUT))
    s = sub.add_parser("spend", help="print the ledger spend")
    s.add_argument("--out", default=str(OUT))
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "instances":
        Ledger(Path(a.out) / "instances.jsonl").record(a.set, a.pool, a.at)
        print(f"recorded {a.pool} instances = {a.set}; spend so far ${Ledger(Path(a.out) / 'instances.jsonl').spend():.2f}")
        return 0
    if a.cmd == "spend":
        print(f"${Ledger(Path(a.out) / 'instances.jsonl').spend():.2f}")
        return 0
    if a.cmd == "finalize":
        print(json.dumps(finalize(Path(a.out)), indent=1))
        return 0
    if a.cmd == "resume":
        gate = Path(a.out) / "pilot-gate.json"
        st = json.loads(gate.read_text()) if gate.exists() else {}
        if st.get("decision") != "paused":
            print(f"pilot gate is {st.get('decision', 'not evaluated yet')}; nothing to resume")
            return 0
        (Path(a.out) / "RESUME_PILOT").write_text(json.dumps({"t": time.time()}) + "\n")
        print("resume flag written; the running teacher resumes dispatch within a few seconds")
        return 0
    a.queue = [Path(p) for p in (a.queue or QUEUES)]
    runner = Runner(a, load_key(a.key_file))
    def on_term(*_):
        raise KeyboardInterrupt
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    s = runner.run()
    return 0 if s["state"] in ("done", "max_items_reached", "interrupted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
