#!/usr/bin/env python3
"""Phase-3 Stage-2 variants of verified failures + hard unknown variants (docs/phase-3-plan.md, Stage 2 items 4 and 5).

Parents are the teacher-KEPT mined items in data/p3/teacher/results.jsonl (origin "mine"). Per parent:
  * 2 variants (same family and difficulty, the pattern not the item):
      - source A / D / I generated (gen_reasoning, gen_policy, gen_traps, image_joint): the generators' variant paths, fresh rng
        keyed by the parent id, gold constructed -> written straight to data/p3/variants/constructed.jsonl (no teacher cost);
      - source B / C text: the Qwen3.6-27B writer on Azure perturbs numbers / entities / conditions / chain order and states the
        new answer -> queued for the teacher (data/p3/stream/queue-variants.jsonl), kept under the same rules as their parent
        (two solves that agree with each other and with the writer's answer).
  * hard unknown variants for ~15% of parents (deterministic by parent id): the decisive fact removed - programmatically for
    A / D / image_joint, by the writer for B / C text. ALL unknown variants go to the teacher and are kept only when the teacher
    puts most of its mass on unknown (azure_teacher.keep_decision, rule "unknown_variant").
  * a variant whose 13-gram overlap with its parent is > 70% is dropped (no near-copies); unknown variants are exempt (they are
    the parent minus one fact by design).
  * leakage: every variant inherits its parent's split (row["split"]; "heldout" when the parent id is in --heldout-ids).
Not supported (counted, reported): GUI screens (no generator variant path) and mined image items (the writer is text-only).
Only origin "mine" results are parents: held-out items (origin "heldout") and variants never get variants.

NO-WRITER MODE (owner decision 2026-09-25: no 27B writer this run): --no-writer, or simply no --writer-deployment. B/C text
parents are recorded as `skipped_no_writer` in parents.jsonl (so --follow/--stop-when-closed terminates), their B/C variants and
writer-made unknown variants are not made; A/D/I-generated variants and programmatic unknown variants are made as usual.

Constructed variants (data/p3/variants/constructed.jsonl) carry the manifest label contract with target_kind "constructed"
(hard constructed gold, no teacher involved; build_manifest.py does not count them against the 25% hard-gold cap of A).

    .venv/bin/python scripts/p3/variants.py --no-writer --follow --stop-when-closed --close                 # this run
    .venv/bin/python scripts/p3/variants.py --writer-deployment qwen36-27b-fp8:1 --max-usd 140 --follow   # with a writer
    # label-then-train pod: a LOCAL writer (vLLM, no key), parents from the pod teacher + Azure's kept B/C parents
    python scripts/p3/variants.py --writer-base-url http://127.0.0.1:8200/v1 --writer-model writer --writer-no-think \
        --results data/p3/teacher-pod/results.jsonl --results data/p3/stream-pod/writer-parents.jsonl --teacher-out data/p3/teacher-pod \
        --out data/p3/variants-pod --queue data/p3/stream-pod/queue-variants.jsonl --upstream-queues <pod queues + writer-parents> \
        --writer-done-marker <markers>/WRITER_DONE --follow --stop-when-closed --close

Resumable: parents already in data/p3/variants/parents.jsonl are skipped; every output is append-only and id-deduplicated.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT / "scripts/p2"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from azure_teacher import (ENDPOINT, AzureClient, Ledger, Redactor, load_key, parse_deployments, read_complete,  # noqa: E402
                           repair_tail)
from candidate import validate  # noqa: E402
from p2_common import loads_lenient, ngrams  # noqa: E402

OUT = ROOT / "data/p3/variants"
OVERLAP_MAX = 0.70
UNKNOWN_SHARE = 0.15
PER_PARENT = 2

WRITER_SYSTEM = (
    "You write training variants of typed decision questions. You receive one item: a document (the state), a question, its "
    "answer options and the verified answer. Write ONE new item that tests the same reasoning pattern but is a different "
    "instance, as instructed. Keep the question type, the answer format and the difficulty. The new document must be "
    "self-contained and realistic, and the answer you give must follow from it exactly; work it out carefully before answering. "
    "Reply with a single JSON object and nothing else."
)
PERTURB_BRIEF = (
    "Make a VARIANT: change the numbers, names, entities, dates or conditions (for example flip a condition, change a threshold, "
    "reorder the steps of a reasoning chain) so the item is new and its answer must be re-derived; the answer may or may not "
    "change. Do not copy sentences from the original. The document must still determine the answer: \"answer\" must not be unknown."
)
UNKNOWN_BRIEF = (
    "Make an UNANSWERABLE version: rewrite the document so that the ONE fact that decides the answer is missing (removed, not "
    "contradicted), while everything else stays plausible and the question still reads naturally. A careful reader must not be "
    "able to determine the answer from the new document. Set \"answer\" to \"unknown\"."
)


# ------------------------------------------------------------------------------------------------ helpers

def text_of(row: dict) -> str:
    st = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], sort_keys=True)
    f = row["field"]
    opts = " ".join(f"{o.get('text', '')} {o.get('description') or ''}" for o in f.get("options") or [])
    return f"{st} {f['question']} {opts}"


def overlap(variant: dict, parent: dict, n: int = 13) -> float:
    v, p = ngrams(text_of(variant), n), ngrams(text_of(parent), n)
    return len(v & p) / len(v) if v else 0.0


def wants_unknown(parent_id: str, share: float = UNKNOWN_SHARE) -> bool:
    return int(hashlib.sha1(f"unknown:{parent_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def origin(row: dict) -> str:
    gen = str((row.get("provenance") or {}).get("generator") or "")
    if gen.startswith("scripts/p3/gen_reasoning.py"):
        return "gen_reasoning"
    if row.get("dataset") == "gen_policy" or gen == "scripts/p3/gen_policy.py":
        return "gen_policy"
    if row.get("dataset") == "gen_traps":
        return "gen_traps"
    if row.get("dataset") == "image_joint":
        return "image_joint"
    if row.get("images"):
        return "image_other"
    return "writer"


def finish_row(row: dict, parent: dict, vid: str, kind: str, method: str, split: str) -> dict:
    row = dict(row)
    prov = dict(row.get("provenance") or {})
    prov.update({"generator_id": row.get("id"), "variant_kind": kind, "variant_method": method, "variant_parent": parent["id"],
                 "split": split})
    row.update(id=vid, parent_id=parent["id"], provenance=prov, split=split)
    return row


# ------------------------------------------------------------------------------------------------ programmatic paths

class Programmatic:
    def __init__(self, work: Path, pool_glob: str):
        self.work, self.pool_glob = work, pool_glob
        self._pool_index: dict[str, dict] | None = None
        self._pool_offsets: dict[str, tuple] | None = None
        self._image_ctx = None

    # -- source A: gen_reasoning
    def reasoning(self, parent: dict, n: int, unknown: bool) -> list[dict]:
        import gen_reasoning as gr
        m = gr.ID_RE.match(parent["id"])
        if not m:
            return []
        kind, d, seed = m["kind"], int(m["d"]), m["seed"]
        if unknown:
            if parent.get("gold") is None:
                return []
            it = gr.gen_one(kind, d, parent["id"], "unknown-variant", 1, tries=60, need_unk=True)
            return [gr.make_row(it, kind, d, seed, parent["id"] + "-u", parent=parent["id"], unknown=True)]
        return gr.variants_of(parent["id"], n)

    # -- source A: gen_policy
    def policy(self, parent: dict, n: int, unknown: bool) -> list[dict]:
        import gen_policy as gp
        src = dict(parent)
        if unknown:
            if parent.get("gold") is None:
                return []
            src.update(gold=None, unknown_reason="insufficient_evidence")
            n = 1
        tmp = self.work / f"tmp-{os.getpid()}-{hashlib.sha1(parent['id'].encode()).hexdigest()[:10]}.jsonl"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(src, ensure_ascii=False) + "\n")
        try:
            rows = gp.variants_of(tmp, n)
        finally:
            tmp.unlink(missing_ok=True)
        # an unknown parent's regenerated variants are unknown too (variants_of keeps the parent's role)
        return [r for r in rows if r.get("gold") is None] if unknown else rows

    # -- source D: a trap of a fresh variant of the trap's parent (same transform)
    def pool_row(self, rid: str) -> dict | None:
        if self._pool_index is None:
            self._pool_index = {}
        if rid in self._pool_index:
            return self._pool_index[rid]
        if self._pool_offsets is None:
            # one pass over the shards: id -> (file, byte offset); every later lookup is a seek (the old per-miss scan of ~1 GB
            # of shards made each trap parent cost seconds)
            self._pool_offsets = {}
            for path in sorted(glob.glob(self.pool_glob)):
                with open(path, "rb") as fh:
                    off = 0
                    for line in fh:
                        if line.startswith(b'{"id": "'):
                            end = line.find(b'"', 8)
                            if end > 8:
                                self._pool_offsets.setdefault(line[8:end].decode(), (path, off))
                        off += len(line)
        hit = self._pool_offsets.get(rid)
        if hit is not None:
            with open(hit[0], "rb") as fh:
                fh.seek(hit[1])
                row = json.loads(fh.readline())
            if row.get("id") == rid:
                self._pool_index[rid] = row
                return row
        if rid not in self._pool_offsets and self._pool_offsets:
            self._pool_index[rid] = None
            return None
        for path in sorted(glob.glob(self.pool_glob)):
            with open(path) as fh:
                for line in fh:
                    if rid in line:
                        row = json.loads(line)
                        if row["id"] == rid:
                            self._pool_index[rid] = row
                            return row
        self._pool_index[rid] = None
        return None

    def traps(self, parent: dict, n: int, unknown: bool) -> list[dict]:
        import gen_traps as gt
        base = self.pool_row(parent.get("parent_id") or "")
        kind = (parent.get("provenance") or {}).get("trap")
        if base is None or kind not in gt.TRANSFORMS:
            return []
        fresh = self.dispatch(base, n + 2, unknown)          # a few spare parents: a transform may not apply to one
        out = []
        for k, b in enumerate(fresh):
            t = gt.make_trap(b, kind, seed=k)
            if t is not None:
                out.append(t)
            if len(out) >= (1 if unknown else n):
                break
        return out

    # -- source I: joint image rule, same photos, fresh record/rule parameters (or a withheld decisive field)
    def image_joint(self, parent: dict, n: int, unknown: bool) -> list[dict]:
        import gen_image_joint as gij
        if self._image_ctx is None:
            abo, dfx, ctx = gij.load_context()
            self._image_ctx = ({p["sha256"]: p for p in abo + dfx}, ctx)
        by_sha, ctx_base = self._image_ctx
        prov = parent["provenance"]
        name = prov["template"]
        photos = [by_sha.get(s) for s in prov["image_sha256"]]
        if any(p is None for p in photos):
            return []
        fl = [p["facts"] for p in photos]
        domain, nph, kind, diff, sample, render, dec, alts = gij.TEMPLATES[name]
        if unknown:
            if parent.get("gold") is None:
                return []
            p = prov["params"]
            fields = [f for f in alts if not f.startswith("__")]
            rng = random.Random(f"unknown:{parent['id']}")
            rng.shuffle(fields)
            for f in fields:
                q = dict(p, _withheld=[f])
                g, u = gij.decide(name, q, fl)
                if g is None:
                    return [gij.make_row(name, q, photos, None, u, 0, parent=parent["id"])]
            return []
        out = []
        for k in range(n):
            rng = random.Random(f"variant:{parent['id']}:{k}")
            for _ in range(200):
                ctx = dict(ctx_base, hidden=False)
                if domain == "abo" and nph == 2:
                    shared = [a for a in ("colour", "material", "product_type", "pattern", "shape") if all(a in f for f in fl)]
                    if not shared:
                        break
                    ctx["pair_attr"] = rng.choice(shared)
                p = sample(rng, photos, ctx)
                if p is None or not gij.preds_clear(p, photos):
                    continue
                g, u = gij.decide(name, p, fl)
                if g is None or p == prov["params"]:
                    continue
                out.append(gij.make_row(name, p, photos, g, None, k, parent=parent["id"]))
                break
        return out

    def dispatch(self, parent: dict, n: int, unknown: bool) -> list[dict]:
        o = origin(parent)
        fn = {"gen_reasoning": self.reasoning, "gen_policy": self.policy, "gen_traps": self.traps,
              "image_joint": self.image_joint}.get(o)
        if fn is None:
            return []
        try:
            return fn(parent, n, unknown)
        except Exception as exc:  # noqa: BLE001 - a generator that cannot vary this parent: counted, not fatal
            print(f"variant generator failed for {parent['id']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            return []


# ------------------------------------------------------------------------------------------------ writer path (B / C text)

def writer_schema(parent: dict) -> dict:
    t = parent["field"]["type"]
    props = {"state": {"type": "string"}, "question": {"type": "string"}, "change": {"type": "string", "maxLength": 300},
             "answer": {"type": "string"}}
    req = ["state", "question", "answer", "change"]
    if t == "choice":
        props["options"] = {"type": "array", "minItems": 2, "maxItems": 26, "items": {
            "type": "object", "additionalProperties": False, "required": ["key", "text"],
            "properties": {"key": {"type": "string"}, "text": {"type": "string"}}}}
        req.append("options")
    return {"type": "json_schema", "json_schema": {"name": "decision_variant", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": req, "properties": props}}}


def writer_prompt(parent: dict, kind: str, k: int) -> str:
    f = parent["field"]
    state = parent["state"] if isinstance(parent["state"], str) else json.dumps(parent["state"], ensure_ascii=False, indent=1)
    if f["type"] == "choice":
        fmt = ("options: a list of {key, text}; keys are short snake_case ids; \"answer\" is one option key or \"unknown\".\n"
               "OPTIONS:\n" + "\n".join(f"- {o['key']}: {o['text']}" + (f" ({o['description']})" if o.get("description") else "")
                                         for o in f["options"]))
    elif f["type"] == "noul":
        fmt = "This is a yes/no question: \"answer\" is \"true\", \"false\" or \"unknown\"."
    else:
        fmt = ("Scored question, levels unchanged: \"answer\" is a level value or \"unknown\".\nLEVELS:\n"
               + "\n".join(f"- {l['value']}: {l['description']}" for l in f["levels"]))
    gold = parent.get("gold")
    gold_s = "unknown" if gold is None else str(gold).lower() if isinstance(gold, bool) else str(gold)
    brief = UNKNOWN_BRIEF if kind == "unknown" else PERTURB_BRIEF
    return (f"ORIGINAL DOCUMENT:\n{state}\n\nQUESTION: {f['question']}\n{fmt}\nVERIFIED ANSWER: {gold_s}\n\n"
            f"TASK ({kind} #{k}): {brief}\nReturn JSON: state (the new document as text), question, "
            + ("options, " if f["type"] == "choice" else "") + "answer, change (one sentence: what you changed).")


def writer_row(parent: dict, obj: dict, kind: str, vid: str, deployment: str, split: str) -> tuple[dict | None, str | None]:
    f = copy.deepcopy(parent["field"])
    f["question"] = str(obj.get("question") or "").strip()
    ans = str(obj.get("answer") or "").strip()
    # the writer prompt shows the parent's "VERIFIED ANSWER"; a variant that copies it into its state/question/options leaks
    # the label (seen on the 2026-09-26 pod: 127 of 19,890 writer rows)
    if "VERIFIED ANSWER" in json.dumps(obj, ensure_ascii=False).upper():
        return None, "leaked VERIFIED ANSWER"
    if f["type"] == "choice":
        opts = []
        for o in obj.get("options") or []:
            key = str(o.get("key") or "").strip()
            opts.append({"key": key, "text": str(o.get("text") or "").strip()})
        f["options"] = opts
    if ans.lower() == "unknown":
        gold, reason = None, "insufficient_evidence"
    elif f["type"] == "noul":
        if ans.lower() not in ("true", "false"):
            return None, f"bad noul answer {ans!r}"
        gold, reason = ans.lower() == "true", None
    elif f["type"] == "score":
        try:
            gold, reason = int(ans), None
        except ValueError:
            return None, f"bad score answer {ans!r}"
    else:
        gold, reason = ans, None
    if kind == "unknown" and gold is not None:
        return None, "unknown variant with a known answer"
    if kind == "perturb" and gold is None:
        return None, "perturb variant answered unknown"
    state = obj.get("state") or ""
    if isinstance(parent["state"], dict):
        try:
            parsed = json.loads(state)
            state = parsed if isinstance(parsed, dict) else state
        except (json.JSONDecodeError, TypeError):
            pass
    prov = dict(parent.get("provenance") or {})
    prov.update({"variant_kind": kind, "variant_method": "writer", "writer_deployment": deployment,
                 "writer_change": str(obj.get("change") or "")[:300], "variant_parent": parent["id"], "split": split})
    row = {**{k: v for k, v in parent.items() if k not in ("split",)}, "id": vid, "state": state, "field": f, "gold": gold,
           "unknown_reason": reason, "gold_kind": "none", "parent_id": parent["id"], "provenance": prov, "split": split}
    errs = validate(row)
    return (None, "; ".join(errs)) if errs else (row, None)


# ------------------------------------------------------------------------------------------------ driver

class Variants:
    def __init__(self, a, client: AzureClient | None):
        if not getattr(a, "results", None):
            a.results = [str(ROOT / "data/p3/teacher/results.jsonl")]
        elif isinstance(a.results, (str, Path)):
            a.results = [a.results]
        self.a, self.client = a, client
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.paths = {k: self.out / f"{k}.jsonl" for k in ("constructed", "writer-raw", "rejects", "parents")}
        self._alock, self._plock = threading.Lock(), threading.Lock()   # appends / writer-result handling (overlapping rounds)
        self._round_pool = ThreadPoolExecutor(2)   # at most two writer rounds in flight
        self.queue = Path(a.queue)
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        for p in list(self.paths.values()) + [self.queue]:
            repair_tail(p)
        self.done_parents = {r["parent_id"] for r in read_complete(self.paths["parents"])[0]}
        self.ids = {r["id"] for r in read_complete(self.paths["constructed"])[0]} | {r["id"] for r in read_complete(self.queue)[0]}
        self.prog = Programmatic(self.out, a.pool_shards)
        self.heldout = set()
        for p in a.heldout_ids or []:
            if Path(p).exists():
                for line in Path(p).read_text().splitlines():
                    if line.strip():
                        self.heldout.add(json.loads(line)["id"] if line.strip().startswith("{") else line.strip())
        self.ledger = Ledger(Path(a.teacher_out) / "instances.jsonl")
        self.counts = Counter()
        self.redact = Redactor(client.key if client else "")

    def append(self, key_or_path, row: dict) -> None:
        path = self.paths.get(key_or_path, key_or_path) if isinstance(key_or_path, str) else key_or_path
        with self._alock, open(path, "a") as fh:
            fh.write(self.redact(json.dumps(row, ensure_ascii=False)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def split_of(self, parent: dict) -> str:
        if parent["id"] in self.heldout:
            return "heldout"
        return parent.get("split") or (parent.get("provenance") or {}).get("split") or "train"

    def budget_left(self) -> bool:
        if self.a.max_usd is not None and self.ledger.spend() >= self.a.max_usd:
            return False
        try:
            st = json.loads((Path(self.a.teacher_out) / "status.json").read_text())
            return st.get("state") not in ("cap_reached", "fatal")
        except (OSError, json.JSONDecodeError):
            return True

    def parents(self) -> list[dict]:
        out, seen = [], set()
        for path in self.a.results:
            for r in read_complete(Path(path))[0]:
                if r.get("keep") and r.get("origin", "mine") == "mine" and r["id"] not in self.done_parents and r["id"] not in seen:
                    seen.add(r["id"])
                    out.append(r["item"])
        return out

    def writer_done(self) -> bool:
        """No writer parent can still appear: every upstream queue is closed, the teacher has a verdict for every upstream row,
        and no parent is waiting. (The variants queue's own rows never make parents, so they are not waited for: the writer's
        GPU can go back to the teacher while those rows are still being labelled.)"""
        if not all((Path(p).parent / (Path(p).name + ".closed")).exists() for p in self.a.upstream_queues):
            return False
        # rows that already are verdicts (the pod's writer-parents files: Azure-kept B/C parents) need no teacher verdict
        want = {r["id"] for q in self.a.upstream_queues for r in read_complete(Path(q))[0] if "keep" not in r}
        have = {r["id"] for r in read_complete(Path(self.a.teacher_out) / "results.jsonl")[0]}
        return want <= have and not self.parents()

    def queue_row(self, row: dict, kind: str, method: str, parent: dict) -> None:
        if row["id"] in self.ids:
            return
        self.append(self.queue, {"id": row["id"], "origin": "variant", "queued_at": time.time(),
                                 "variant": {"kind": kind, "method": method, "parent_id": parent["id"]}, "item": row})
        self.ids.add(row["id"])
        self.counts[f"queued:{kind}:{method}"] += 1

    def accept(self, row: dict, parent: dict, kind: str, method: str) -> bool:
        errs = validate(row)
        if errs:
            self.append("rejects", {"id": row.get("id"), "parent_id": parent["id"], "reason": "; ".join(errs)})
            self.counts["reject:invalid"] += 1
            return False
        if kind != "unknown":
            ov = overlap(row, parent)
            if ov > OVERLAP_MAX:
                self.append("rejects", {"id": row["id"], "parent_id": parent["id"], "reason": f"13-gram overlap {ov:.2f}"})
                self.counts["reject:overlap"] += 1
                return False
        if kind == "unknown" or method == "writer":
            self.queue_row(row, kind, method, parent)
        elif row["id"] not in self.ids:
            row = dict(row, label={"target": row.get("gold"), "probs": None, "rationale": None, "target_kind": "constructed",
                                   "review": None})
            self.append("constructed", row)
            self.ids.add(row["id"])
            self.counts["constructed"] += 1
        return True

    def programmatic(self, parent: dict) -> dict:
        split, made = self.split_of(parent), Counter()
        for k, row in enumerate(self.prog.dispatch(parent, PER_PARENT, False)[:PER_PARENT], start=1):
            if self.accept(finish_row(row, parent, f"{parent['id']}-var{k}", "regen", "programmatic", split), parent, "regen",
                           "programmatic"):
                made["regen"] += 1
        if wants_unknown(parent["id"], self.a.unknown_share):
            for row in self.prog.dispatch(parent, 1, True)[:1]:
                if self.accept(finish_row(row, parent, f"{parent['id']}-unk1", "unknown", "programmatic", split), parent,
                               "unknown", "programmatic"):
                    made["unknown"] += 1
        return made

    def writer_up(self, wait_s: float | None = None) -> bool:
        """A local writer (LocalClient) must answer /models before a round is sent (a round sent to a dead server would mark its
        parents done with writer errors). Azure clients have no health check: always True."""
        healthy = getattr(self.client, "healthy", None)
        if healthy is None:
            return True
        deadline = time.time() + (self.a.writer_wait if wait_s is None else wait_s)
        while True:
            if healthy(self.a.writer_name):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(5)

    def writer_job(self, parent: dict, kind: str, k: int):
        body = {"model": self.a.writer_name, "temperature": 0.8, "seed": 1000 + k, "max_tokens": self.a.writer_max_tokens,
                "messages": [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": writer_prompt(parent, kind, k)}],
                "response_format": writer_schema(parent)}
        if getattr(self.a, "writer_no_think", False):
            # local vLLM only (Azure rejects chat_template_kwargs); the teacher re-verifies every writer variant anyway
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = self.client.post(body)
            raw = (resp["choices"][0].get("message") or {}).get("content") or ""
            return raw, resp.get("usage"), None
        except Exception as exc:  # noqa: BLE001
            return "", None, self.redact(f"{type(exc).__name__}: {str(exc)[:300]}")

    def writer(self, parents: list[dict]) -> Counter:
        jobs = []
        for parent in parents:
            jobs += [(parent, "perturb", k) for k in range(1, PER_PARENT + 1)]
            if wants_unknown(parent["id"], self.a.unknown_share):
                jobs.append((parent, "unknown", 1))
        made = Counter()
        with ThreadPoolExecutor(self.a.writer_concurrency) as ex:
            futs = {ex.submit(self.writer_job, *j): j for j in jobs}
            # consumed as they complete (not in submission order): one slow call no longer holds back the rows, and the
            # teacher gets writer variants as soon as they exist
            for fut in as_completed(futs):
                parent, kind, k = futs[fut]
                raw, usage, err = fut.result()
                with self._plock:
                    self._one_writer_result(parent, kind, k, raw, usage, err, made)
        return made

    def _finish_round(self, fut, chunk) -> None:
        fut.result()
        for parent in chunk:
            self.append("parents", {"parent_id": parent["id"], "method": "writer", "t": time.time()})
            self.done_parents.add(parent["id"])

    def _one_writer_result(self, parent, kind, k, raw, usage, err, made) -> None:
        vid = f"{parent['id']}-w{'u' if kind == 'unknown' else 'v'}{k}"
        self.append("writer-raw", {"id": vid, "parent_id": parent["id"], "kind": kind, "raw": raw[:6000], "usage": usage,
                                   "error": err, "t": time.time()})
        if err:
            self.counts["writer_error"] += 1
            return
        try:
            obj = loads_lenient(raw)
        except Exception:  # noqa: BLE001
            self.counts["reject:writer_json"] += 1
            return
        row, why = writer_row(parent, obj, kind, vid, self.a.writer_name, self.split_of(parent))
        if row is None:
            self.append("rejects", {"id": vid, "parent_id": parent["id"], "reason": why})
            self.counts["reject:writer_row"] += 1
            return
        if self.accept(row, parent, kind, "writer"):
            made[kind] += 1

    def pass_once(self) -> Counter:
        parents = self.parents()
        writer_parents = []
        for parent in parents:
            o = origin(parent)
            if o in ("gen_reasoning", "gen_policy", "gen_traps", "image_joint"):
                made = self.programmatic(parent)
                self.append("parents", {"parent_id": parent["id"], "method": "programmatic", "origin": o, **made, "t": time.time()})
                self.done_parents.add(parent["id"])
            elif o == "writer" and parent.get("source") in ("B", "C"):
                writer_parents.append(parent)
            else:
                self.counts[f"unsupported:{o}"] += 1
                self.append("parents", {"parent_id": parent["id"], "method": "none", "origin": o, "t": time.time()})
                self.done_parents.add(parent["id"])
        if writer_parents and self.client is not None:
            prev = None
            for i in range(0, len(writer_parents), self.a.writer_batch):
                if not self.budget_left():
                    self.counts["stopped_budget"] += 1
                    print("variants: credit cap reached or teacher stopped; writer paused", flush=True)
                    break
                if not self.writer_up():
                    self.counts["writer_down_pass_skipped"] += 1
                    print("variants: writer server not answering; these parents wait for the next pass", flush=True)
                    break
                chunk = writer_parents[i:i + self.a.writer_batch]
                # two rounds overlap: this chunk starts now, the previous chunk's slow tail finishes meanwhile, then its
                # parents are marked done (a round no longer waits for its slowest call with the writers idle)
                fut = self._round_pool.submit(self.writer, chunk)
                if prev is not None:
                    self._finish_round(*prev)
                prev = (fut, chunk)
            if prev is not None:
                self._finish_round(*prev)
                prev = None
        elif writer_parents:
            # no writer this run: record the parents as handled so --follow / --stop-when-closed can finish
            self.counts["skipped_no_writer"] += len(writer_parents)
            for parent in writer_parents:
                self.append("parents", {"parent_id": parent["id"], "method": "skipped_no_writer", "origin": "writer",
                                        "t": time.time()})
                self.done_parents.add(parent["id"])
        print(f"variants: {dict(sorted(self.counts.items()))}", flush=True)
        return self.counts

    def run(self) -> Counter:
        marker = Path(self.a.writer_done_marker) if getattr(self.a, "writer_done_marker", None) else None
        while True:
            self.pass_once()
            if marker is not None and not marker.exists() and self.writer_done():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"t": time.time(), "counts": dict(self.counts)}) + "\n")
                print(f"variants: writer done (no parent can still appear): wrote {marker}", flush=True)
            if not self.a.follow:
                break
            if self.a.stop_when_closed and all((Path(p).parent / (Path(p).name + ".closed")).exists() for p in self.a.upstream_queues):
                teacher = {}
                try:
                    teacher = json.loads((Path(self.a.teacher_out) / "status.json").read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                # idle = the teacher has a verdict for every row of every queue (so no kept parent can still appear), or it
                # stopped for good; a momentary pending == inflight == 0 between two queue polls is not enough
                rows = len({r.get("id") for q in list(self.a.upstream_queues) + [self.queue] for r in read_complete(Path(q))[0]
                            if "keep" not in r})     # distinct ids (a restarted run can append a row twice); verdict rows
                                                     # (writer-parents files) need no teacher verdict
                idle = teacher.get("pending") == 0 and teacher.get("inflight") == 0 and (teacher.get("results") or 0) >= rows
                if (idle or teacher.get("state") in ("cap_reached", "fatal")) and not self.parents():
                    break
            time.sleep(self.a.interval)
        if self.a.close:
            (self.queue.parent / (self.queue.name + ".closed")).write_text(json.dumps({"t": time.time()}) + "\n")
        return self.counts


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", action="append", default=None,
                    help="teacher results (parents = kept, origin mine); repeatable (default data/p3/teacher/results.jsonl)")
    ap.add_argument("--teacher-out", default=str(ROOT / "data/p3/teacher"), help="for status.json and the instance ledger")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--queue", default=str(ROOT / "data/p3/stream/queue-variants.jsonl"))
    ap.add_argument("--pool-shards", default=str(ROOT / "data/p3/pool/shards/*.jsonl"), help="to look up a trap's parent")
    ap.add_argument("--heldout-ids", action="append", help="ids in the held-out set (variants inherit the heldout split)")
    ap.add_argument("--unknown-share", type=float, default=UNKNOWN_SHARE)
    ap.add_argument("--writer-deployment", default=None, help="Azure deployment name[:instances] of the 27B writer")
    ap.add_argument("--no-writer", action="store_true", help="programmatic variants only (B/C parents wait for a later pass)")
    ap.add_argument("--base-url", default=os.environ.get("IMAJEV_AZURE_BASE_URL", ENDPOINT))
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--writer-base-url", default=None,
                    help="a LOCAL OpenAI-compatible writer (vLLM on the pod): no key is read or sent; overrides --writer-deployment")
    ap.add_argument("--writer-model", default="writer", help="--writer-base-url: the served model name")
    ap.add_argument("--writer-no-think", action="store_true",
                    help="--writer-base-url: chat_template_kwargs enable_thinking=false (faster; the teacher re-verifies)")
    ap.add_argument("--writer-wait", type=float, default=600.0,
                    help="--writer-base-url: seconds to wait for the writer server before a round (then retry next pass)")
    ap.add_argument("--writer-done-marker", default=None,
                    help="write this file once no writer parent can still appear (the pod then gives the writer GPU to the teacher)")
    ap.add_argument("--writer-concurrency", type=int, default=None, help="default 64 x writer instances")
    ap.add_argument("--writer-batch", type=int, default=200, help="parents per writer round (budget checked between rounds)")
    ap.add_argument("--writer-max-tokens", type=int, default=None,
                    help="default 4096 with --writer-no-think (pod p99 3.8k tokens; longer = runaway text that stalls a round), else 12000")
    ap.add_argument("--max-usd", type=float, default=None, help="stop writer calls once the shared ledger spend reaches this")
    ap.add_argument("--follow", action="store_true", help="repeat every --interval seconds")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--stop-when-closed", action="store_true",
                    help="with --follow: exit once the upstream queues are closed and the teacher is idle")
    ap.add_argument("--upstream-queues", nargs="*", default=[str(ROOT / "data/p3/stream/queue.jsonl")])
    ap.add_argument("--close", action="store_true", help="write <queue>.closed at exit so the teacher can finish")
    return ap


class WriterPool:
    """Several local writer servers (LocalClient each). Each request goes to the healthy server with the fewest requests in
    flight from this process, so a slower server gets less work and no server idles while another still holds a backlog."""

    def __init__(self, clients: list, name: str):
        self.clients, self.name, self.key, self._lock = clients, name, "", threading.Lock()
        self.inflight = [0] * len(clients)

    def healthy(self, route: str) -> bool:
        return any(c.healthy(route) for c in self.clients)

    def post(self, body: dict) -> dict:
        ok = [i for i, c in enumerate(self.clients) if c.healthy(self.name)] or list(range(len(self.clients)))
        with self._lock:
            i = min(ok, key=lambda j: self.inflight[j])
            self.inflight[i] += 1
        try:
            return self.clients[i].post(body)
        finally:
            with self._lock:
                self.inflight[i] -= 1


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.writer_max_tokens is None:
        a.writer_max_tokens = 4096 if getattr(a, "writer_no_think", False) else 12000
    a.results = a.results or [str(ROOT / "data/p3/teacher/results.jsonl")]
    client = None
    a.writer_name = None
    if a.writer_base_url and not a.no_writer:
        from local_backend import LocalClient
        a.writer_name = a.writer_model
        # extra writer servers: comma-separated --writer-base-url, plus <out>/writer-urls.txt (one URL per line; read at start, so
        # the lead can add servers on idle teacher GPUs and restart variants.py without changing the pod's launch line)
        urls = [u for u in a.writer_base_url.split(",") if u]
        extra = Path(a.out) / "writer-urls.txt"
        if extra.exists():
            urls += [u.strip() for u in extra.read_text().splitlines() if u.strip() and u.strip() not in urls]
        clients = [LocalClient({a.writer_model: u}, a.writer_model) for u in urls]
        client = clients[0] if len(clients) == 1 else WriterPool(clients, a.writer_model)
        a.writer_concurrency = a.writer_concurrency or 64 * len(clients)
        if len(clients) > 1:
            a.writer_batch = a.writer_batch * len(clients)
            print(f"variants: {len(clients)} writer servers {urls}, concurrency {a.writer_concurrency}, batch {a.writer_batch}", flush=True)
        Variants(a, client).run()
        return 0
    if not a.writer_deployment and not a.no_writer:
        print("variants: no --writer-deployment given: running in no-writer mode (B/C parents skipped)", flush=True)
    if a.writer_deployment and not a.no_writer:
        (name, n), = parse_deployments([a.writer_deployment])
        a.writer_name = name
        a.writer_concurrency = a.writer_concurrency or 64 * n
        client = AzureClient(a.base_url, load_key(a.key_file))
    a.writer_concurrency = a.writer_concurrency or 64
    Variants(a, client).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
