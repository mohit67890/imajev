#!/usr/bin/env python3
"""Phase-3 teacher on the label-then-train pod: azure_teacher.py's runner against LOCAL vLLM servers (reports/phase3/LABELTRAIN-RUNBOOK.md).

Same prompts, JSON schema, keep rules, second-solve policy, pilot gate, finalize and file formats as scripts/p3/azure_teacher.py
(it IS that runner, through its backend hooks). What differs:
  * backend: one OpenAI-compatible base URL per server (--server gpu0=http://127.0.0.1:8100/v1 ...), the served model name
    (--model), NO api-key / Authorization header (there is no key on the pod), no chat_template_kwargs (thinking stays on, as on
    Azure);
  * a worker only takes tasks while its server answers /v1/models, so a server that is still loading, restarting or being switched
    from writer to teacher (GPU 7) drops nothing;
  * NO Azure cost ledger: no instances.jsonl, no $ cap. --pod-usd-per-hour (optional) turns elapsed pod time into an info-only
    "pod $" figure on the status line;
  * its own output directory (default data/p3/teacher-pod/), so data/p3/teacher/ (Azure) is never touched;
  * every solve and result row records "teacher_backend": "pod-fp8" and "teacher_model" (repo@revision).

    python scripts/p3/pod_teacher.py run --server gpu0=http://127.0.0.1:8100/v1 --server gpu1=http://127.0.0.1:8101/v1 \
        --model teacher --queue data/p3/stream-pod/queue-extra.jsonl --queue data/p3/stream-pod/queue-leftover.jsonl \
        --queue data/p3/stream-pod/queue-variants.jsonl --follow
    python scripts/p3/pod_teacher.py finalize            # = azure_teacher.py finalize --out data/p3/teacher-pod
    python scripts/p3/pod_teacher.py resume              # release a pilot-gate pause (writes data/p3/teacher-pod/RESUME_PILOT)
    python scripts/p3/pod_teacher.py compare --out data/p3/teacher-pod-consistency   # FP8 pod vs Azure bf16 on shared items
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (ROOT / "scripts/p2", HERE):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import azure_teacher as T  # noqa: E402
from local_backend import LocalClient, parse_servers  # noqa: E402

OUT = ROOT / "data/p3/teacher-pod"
STREAM = ROOT / "data/p3/stream-pod"
QUEUES = [STREAM / "queue-extra.jsonl", STREAM / "queue-leftover.jsonl", STREAM / "queue-variants.jsonl"]
BACKEND = "pod-fp8"
TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8@95a723d08a9490559dae23d0cff1d9466213d989"
PER_SERVER_CONCURRENCY = 96


class PodLedger:
    """Info only: elapsed wall time since the first run in this directory x --pod-usd-per-hour. Never a cap, never Azure."""

    def __init__(self, out: Path, usd_per_hour: float, clock=time.time):
        self.path, self.rate, self.clock = out / "pod-clock.json", usd_per_hour, clock
        if not self.path.exists():
            T.write_atomic(self.path, {"t0": clock(), "note": "label-then-train pod: first pod_teacher.py run (info only)"})
        self.t0 = json.loads(self.path.read_text())["t0"]

    def spend(self, now: float | None = None) -> float:
        return max(0.0, ((self.clock() if now is None else now) - self.t0) / 3600 * self.rate)

    def current(self, pool: str = "teacher"):
        return None

    def record(self, *a, **k) -> None:          # never writes an instance ledger
        return None


class PodRunner(T.Runner):
    RECORD_INSTANCES = False

    def make_client(self, key: str, sleep):
        return LocalClient(self.routes, self.a.model, timeout=self.a.timeout, max_attempts=self.a.max_attempts, sleep=sleep,
                           health_ttl=self.a.health_ttl)

    def make_ledger(self, clock):
        return PodLedger(self.out, self.a.pod_usd_per_hour, clock)

    def __init__(self, a, clock=time.time, sleep=time.sleep):
        self.routes = parse_servers(a.server)
        a.deployment = [f"{name}:1" for name in self.routes]       # one "instance" per server
        a.per_instance_concurrency = a.per_server_concurrency
        a.max_usd = 0.0                                              # unused: cap_reached() is always False here
        a.usd_per_instance_hour = a.pod_usd_per_hour / max(1, len(self.routes))
        super().__init__(a, "", clock=clock, sleep=sleep)
        self.token_usd = 0.0

    # -- no Azure cap on the pod
    def cap_reached(self) -> bool:
        return False

    def spend(self) -> float:
        return self.ledger.spend()

    def dispatch_ready(self, deployment: str) -> bool:
        return self.client.healthy(deployment)

    def cost_text(self, s: dict) -> str:
        up = sum(1 for r in self.routes if self.client.healthy(r))
        return f"servers up {up}/{len(self.routes)} | pod ${s['spend_usd']:.2f} (info, {self.a.pod_usd_per_hour:.2f}/h)"

    def pause_note(self) -> str:
        return ("Nothing new is sent to the local teacher servers; in-flight requests finish. THE POD KEEPS BILLING while paused "
                f"({self.a.pod_usd_per_hour:.2f} $/h as configured; ~$21.5-28/h for 8xH100).\n"
                "Decide now (lead, via ssh):\n"
                f"  - continue: `cd  && venv/bin/python scripts/p3/pod_teacher.py resume --out {self.out}`;\n"
                "  - or fix and continue: stop the runner (kill its pid in labeltrain/pids/teacher-py.pid; the orchestrator "
                "restarts it, resumable), then resume;\n"
                "  - or stop: decide with the owner; kill the orchestrator and the runner (pids in labeltrain/pids/).\n")

    def _append(self, fh, row):
        row.setdefault("teacher_backend", BACKEND)
        row.setdefault("teacher_model", self.a.teacher_model)
        super()._append(fh, row)

    def status(self, final=False) -> dict:
        s = super().status(final)
        s.update(teacher_backend=BACKEND, teacher_model=self.a.teacher_model,
                 servers={r: self.client.healthy(r) for r in self.routes})
        T.write_atomic(self.out / "status.json", s)
        return s


# ------------------------------------------------------------------------------------------------ FP8 vs bf16 consistency
def compare(out: Path, queues: list[Path]) -> dict:
    """Pod verdicts on a sample Azure already labelled (queue rows carry the Azure verdict under "azure"). Written to
    <out>/consistency.json; never read by build_manifest (the sample stays in its own directory)."""
    results = {r["id"]: r for r in T.read_complete(out / "results.jsonl")[0]}
    qrows = []
    for p in queues:
        qrows += T.read_complete(Path(p))[0]
    by_id = {q["id"]: q for q in qrows}
    n = agree1 = agree_keep = both_keep = tv_n = 0
    tv_sum = 0.0
    per_rule = Counter()
    per_rule_agree = Counter()
    for qid, r in results.items():
        az = (by_id.get(qid) or {}).get("azure")
        if not az:
            continue
        n += 1
        pod1 = (r.get("labels") or [None])[0]
        az1 = (az.get("labels") or [None])[0]
        agree1 += pod1 is not None and pod1 == az1
        agree_keep += bool(r.get("keep")) == bool(az.get("keep"))
        per_rule[r.get("rule")] += 1
        per_rule_agree[r.get("rule")] += pod1 is not None and pod1 == az1
        if r.get("keep") and az.get("keep") and r.get("target_probs") and az.get("target_probs"):
            both_keep += 1
            labs = set(r["target_probs"]) | set(az["target_probs"])
            tv_sum += 0.5 * sum(abs(r["target_probs"].get(k, 0.0) - az["target_probs"].get(k, 0.0)) for k in labs)
            tv_n += 1
    rep = {"items": n, "solve1_argmax_agreement": round(agree1 / n, 4) if n else None,
           "keep_verdict_agreement": round(agree_keep / n, 4) if n else None,
           "both_kept": both_keep, "mean_tv_distance_kept": round(tv_sum / tv_n, 4) if tv_n else None,
           "by_rule": {k: {"n": v, "solve1_agreement": round(per_rule_agree[k] / v, 4)} for k, v in sorted(per_rule.items(), key=str)},
           "t": time.time()}
    T.write_atomic(out / "consistency.json", rep)
    return rep


# ------------------------------------------------------------------------------------------------ CLI
def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stream the pod queues through the local teacher servers")
    r.add_argument("--server", action="append", required=True, help="NAME=BASE_URL (e.g. gpu0=http://127.0.0.1:8100/v1); repeat")
    r.add_argument("--model", default="teacher", help="the served model name (vllm --served-model-name)")
    r.add_argument("--teacher-model", default=TEACHER_MODEL, help="repo@revision recorded on every row")
    r.add_argument("--queue", action="append", default=None, help="queue JSONL (default: data/p3/stream-pod/queue-*.jsonl)")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--data-root", default=str(ROOT / "data"))
    r.add_argument("--per-server-concurrency", type=int, default=PER_SERVER_CONCURRENCY)
    r.add_argument("--pod-usd-per-hour", type=float, default=0.0, help="info only: the pod's $/h for the status line")
    r.add_argument("--health-ttl", type=float, default=5.0)
    r.add_argument("--max-tokens", type=int, default=16000, help="owner 2026-09-26: 16k (Azure truncated ~4%% at 12k on later items)")
    r.add_argument("--timeout", type=float, default=1800.0)
    r.add_argument("--max-attempts", type=int, default=8)
    r.add_argument("--solve-retries", type=int, default=3)
    r.add_argument("--max-items", type=int, default=None)
    r.add_argument("--follow", action="store_true")
    r.add_argument("--poll-secs", type=float, default=15.0)
    r.add_argument("--status-secs", type=float, default=15.0)
    r.add_argument("--tok-per-s", type=float, default=T.TOK_PER_S)
    r.add_argument("--billing-start", type=float, default=None)
    r.add_argument("--no-pilot-gate", dest="pilot_gate", action="store_false")
    r.add_argument("--gate-items", type=int, default=T.GATE["items"])
    r.add_argument("--gate-max-parse-errors", type=float, default=T.GATE["max_parse_errors"])
    r.add_argument("--gate-max-truncation", type=float, default=T.GATE["max_truncation"])
    r.add_argument("--gate-min-constructed-keep", type=float, default=T.GATE["min_constructed_keep"])
    r.add_argument("--gate-min-constructed-n", type=int, default=T.GATE["min_constructed_n"])
    for name in ("finalize", "resume", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT))
        if name == "compare":
            p.add_argument("--queue", action="append", default=None,
                           help="consistency queue(s) (default data/p3/stream-pod/queue-consistency.jsonl)")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "finalize":
        print(json.dumps(T.finalize(Path(a.out)), indent=1))
        return 0
    if a.cmd == "resume":
        return T.main(["resume", "--out", a.out])
    if a.cmd == "compare":
        print(json.dumps(compare(Path(a.out), a.queue or [STREAM / "queue-consistency.jsonl"]), indent=1))
        return 0
    a.queue = [Path(p) for p in (a.queue or QUEUES)]
    runner = PodRunner(a)

    def on_term(*_):
        raise KeyboardInterrupt
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    s = runner.run()
    return 0 if s["state"] in ("done", "max_items_reached", "interrupted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
