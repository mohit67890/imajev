#!/usr/bin/env python3
"""Phase-3 streamed session coordinator (docs/phase-3-plan.md, "Streamed mining + teacher"). Runs ON THE MAC.

Every --interval seconds it
  1. pulls the mining workers' flagged outputs from the pod over ssh (rsync via an ssh alias; the Azure key never leaves the Mac),
     or, for a dry run, rsyncs them from a local directory (--pull-from-dir, same filters),
  2. reads the new complete lines of every <shard>.flagged.jsonl,
  3. orders them most-confidently-wrong first and applies the per-family quotas (data/p3/pool/quotas.json, re-read every cycle,
     so the assembly agent or the owner can change them mid-run) and the global cap; a family stops feeding once its quota is full,
  4. appends the admitted items to the teacher queue (data/p3/stream/queue.jsonl), which azure_teacher.py `run --follow` consumes,
  5. prints a status line (pod progress, queue per family vs quota, teacher spend) and stops feeding when the teacher reports
     state "cap_reached" or "fatal".
When every pool shard has a .done marker and all flagged rows are consumed, it writes <queue>.closed.

Held-out rows (leakage rule, docs/phase-3-plan.md Stage 1 item 4):
  * heldout-fresh ids (data/p3/pool/heldout-fresh*.jsonl, plus any --exclude-ids file) are never queued;
  * a flagged row whose pool.heldout_flagged_bucket is set belongs to the flagged held-out half. It is NOT a training item: it is
    queued with origin "heldout" (outside the family quotas, capped by heldout_flagged_cap / --heldout-cap) so the teacher gives
    it an EVALUATION label; azure_teacher.py `finalize` writes those verdicts to heldout-results.jsonl (never kept.jsonl), variants.py
    ignores them, and `assemble_pool.py take-flagged` + build_manifest.py turn them into the flagged held-out manifest.

quotas.json (data/p3/pool/quotas.json from scripts/p3/assembly_quotas.py, or any of the older shapes):
    {"cap_total_teacher_calls": 70000, "families": {"<quota_key>": {"quota": 4146, ...}, ...}, "heldout_flagged_cap": 4000}
    {"global_cap": 70000, "default_family_quota": 3000, "families": {"long_policy": 9000, ...}}   or {"quotas": {...}} or flat
The quota key of a row is pool.quota_key (the family, prefixed "img:" for image rows), else its family. Family quotas count
queued first solves of mined items. The global cap counts ESTIMATED TEACHER CALLS: queued items (mined + held-out) + second solves
the teacher has made (status.json) + rows in the variants queue. --global-cap overrides the file's cap. Items waiting on a full
quota are not written anywhere: raising the quota admits them on the next cycle. The Azure runner's $ cap is the final guard.

    .venv/bin/python scripts/p3/stream_coordinator.py --ssh imajev-mine --remote-dir data/p3/mine
    .venv/bin/python scripts/p3/stream_coordinator.py --pull-from-dir <scratch>/pod/mine --local-dir <scratch>/pulled   # dry run
    .venv/bin/python scripts/p3/stream_coordinator.py --no-pull --local-dir data/p3/mine --once     # local / test

Crash-safe: the queue is append-only with ids; a restart rebuilds the per-family counts from it and never queues an id twice.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from azure_teacher import read_complete, repair_tail, write_atomic  # noqa: E402

DEFAULT_GLOBAL_CAP = 70000
DEFAULT_HELDOUT_CAP = 4000
META_KEYS = ("global_cap", "default_family_quota", "cap_total_teacher_calls", "heldout_flagged_cap", "first_solve_total",
             "first_solve_text", "first_solve_image", "expected_second_solves", "expected_bc_variant_calls", "expected_total_calls")


def load_quotas(path: Path) -> tuple[dict, int | None, int | None]:
    """-> (quota key -> quota, default quota or None = unlimited, global cap on teacher calls or None)."""
    if not path.exists():
        return {}, None, None
    obj = json.loads(path.read_text())
    fams = obj.get("families") or obj.get("quotas")
    if fams is None:
        fams = {k: v for k, v in obj.items() if isinstance(v, (int, float)) and k not in META_KEYS}
    fams = {k: int(v["quota"] if isinstance(v, dict) else v) for k, v in fams.items()
            if not isinstance(v, dict) or v.get("teacher", True)}
    default = obj.get("default_family_quota")
    cap = obj.get("global_cap") or obj.get("cap_total_teacher_calls")
    return fams, (int(default) if default is not None else None), (int(cap) if cap else None)


def heldout_cap(path: Path) -> int | None:
    try:
        v = json.loads(path.read_text()).get("heldout_flagged_cap")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return int(v) if v is not None else None


def quota_key(item: dict) -> str:
    return (item.get("pool") or {}).get("quota_key") or item.get("family")


def in_heldout_bucket(item: dict) -> bool:
    return bool((item.get("pool") or {}).get("heldout_flagged_bucket"))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.endswith(b"\n") and line.strip())


def pull(ssh: str | None, remote_dir: str, local_dir: Path, scores: bool = False, runner=subprocess.run) -> bool:
    """rsync the mining outputs (flagged rows, .done markers, progress heartbeats; scores on request). ssh None = remote_dir is a
    local directory (dry run: exercises the same filters without ssh)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    inc = ["--include=*.flagged.jsonl", "--include=*.done", "--include=progress/", "--include=progress/*.json"]
    if scores:
        inc.append("--include=*.scores.jsonl")
    src = f"{ssh}:{remote_dir.rstrip('/')}/" if ssh else f"{remote_dir.rstrip('/')}/"
    transport = ["-e", "ssh -o BatchMode=yes -o ConnectTimeout=20"] if ssh else []
    cmd = ["rsync", "-az", "--timeout=60", *transport, *inc, "--exclude=*", src, f"{str(local_dir).rstrip('/')}/"]
    r = runner(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"pull failed (rc {r.returncode}): {(r.stderr or '').strip()[:300]}", flush=True)
    return r.returncode == 0


class Coordinator:
    def __init__(self, a):
        self.a = a
        self.local = Path(a.local_dir)
        self.queue = Path(a.queue)
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        repair_tail(self.queue)
        self.queued: set[str] = set()
        self.heldout_queued: set[str] = set()
        self.by_family: Counter = Counter()
        for q in read_complete(self.queue)[0]:
            if q.get("origin", "mine") == "mine" and q["id"] not in self.queued:
                self.queued.add(q["id"])
                self.by_family[quota_key(q["item"])] += 1
            elif q.get("origin") == "heldout":
                self.heldout_queued.add(q["id"])
        self.offsets: dict[Path, int] = {}
        self.seen: set[str] = set(self.queued) | self.heldout_queued
        self.waiting: dict[str, dict] = {}
        self.heldout_waiting: dict[str, dict] = {}
        self.excluded: set[str] = set()
        self._excl_mtimes: dict[str, float] = {}
        self.excluded_seen = 0
        self.flagged_total = 0
        self.stopped_reason = None

    def exclude_files(self) -> list[str]:
        files = list(self.a.exclude_ids or [])
        if not self.a.no_default_exclude:
            files += sorted(glob.glob(str(Path(self.a.pool_shards).parent.parent / "heldout-fresh*.jsonl")))
        return files

    def load_excluded(self) -> None:
        """Held-out-fresh ids (never sent to the teacher); files are re-read only when their mtime changes."""
        for p in self.exclude_files():
            path = Path(p)
            try:
                mt = path.stat().st_mtime
            except OSError:
                continue
            if self._excl_mtimes.get(p) == mt:
                continue
            self._excl_mtimes[p] = mt
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.excluded.add(json.loads(line)["id"] if line.startswith("{") else line)

    def read_new(self) -> list[dict]:
        new = []
        for path in sorted(self.local.glob("*.flagged.jsonl")):
            rows, self.offsets[path] = read_complete(path, self.offsets.get(path, 0))
            for r in rows:
                if r["id"] in self.seen:
                    continue
                self.seen.add(r["id"])
                self.flagged_total += 1
                new.append(r)
        return new

    def teacher_state(self) -> dict:
        try:
            return json.loads(Path(self.a.teacher_status).read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def cycle(self) -> dict:
        if not self.a.no_pull:
            pull(self.a.ssh, self.a.pull_from_dir or self.a.remote_dir, self.local, scores=False)
        self.load_excluded()
        for r in self.read_new():
            if r["id"] in self.excluded:
                self.excluded_seen += 1
                continue
            if in_heldout_bucket(r["item"]):
                self.heldout_waiting[r["id"]] = r
            else:
                self.waiting[r["id"]] = r
        fams, default, file_cap = load_quotas(Path(self.a.quotas))
        cap = self.a.global_cap if self.a.global_cap is not None else file_cap if file_cap is not None else DEFAULT_GLOBAL_CAP
        hcap = self.a.heldout_cap if self.a.heldout_cap is not None else heldout_cap(Path(self.a.quotas))
        hcap = DEFAULT_HELDOUT_CAP if hcap is None else hcap
        teacher = self.teacher_state()
        variant_rows = count_lines(Path(self.a.variants_queue))

        def calls() -> int:
            return len(self.queued) + len(self.heldout_queued) + int(teacher.get("second_solves") or 0) + variant_rows
        admitted = admitted_heldout = 0
        blocked = Counter()
        if teacher.get("state") in ("cap_reached", "fatal"):
            self.stopped_reason = f"teacher {teacher['state']}"
        else:
            self.stopped_reason = None
            ordered = sorted(self.waiting.values(), key=lambda r: (-r.get("priority", 0.0), r["id"]))
            with open(self.queue, "a") as fh:
                # the flagged held-out half first (arrival order, no priority: an unbiased slice), outside the family quotas
                for r in sorted(self.heldout_waiting.values(), key=lambda r: r["id"]):
                    if len(self.heldout_queued) >= hcap or calls() >= cap:
                        blocked["heldout_cap"] += 1
                        continue
                    row = {"id": r["id"], "origin": "heldout", "queued_at": time.time(), "priority": r.get("priority"),
                           "shard": r.get("shard"), "item": r["item"], "mine": r.get("mine")}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.heldout_queued.add(r["id"])
                    del self.heldout_waiting[r["id"]]
                    admitted_heldout += 1
                for r in ordered:
                    fam = quota_key(r["item"])
                    quota = fams.get(fam, default)
                    if calls() >= cap:
                        blocked["global_cap"] += 1
                        continue
                    if quota is not None and self.by_family[fam] >= quota:
                        blocked[f"quota:{fam}"] += 1
                        continue
                    row = {"id": r["id"], "origin": "mine", "queued_at": time.time(), "priority": r.get("priority"),
                           "shard": r.get("shard"), "item": r["item"], "mine": r.get("mine")}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.queued.add(r["id"])
                    self.by_family[fam] += 1
                    del self.waiting[r["id"]]
                    admitted += 1
                fh.flush()
                os.fsync(fh.fileno())
        done_markers = len(list(self.local.glob("*.done")))
        expected = self.a.expected_shards if self.a.expected_shards is not None else len(glob.glob(self.a.pool_shards))
        progress = []
        for p in (self.local / "progress").glob("*.json"):
            try:
                progress.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        mining_done = expected > 0 and done_markers >= expected
        status = {"t": time.time(), "shards_done": done_markers, "shards_expected": expected, "mining_done": mining_done,
                  "pod_items": sum(p.get("items", 0) for p in progress),
                  "pod_items_per_hour": sum(p.get("items_per_hour", 0) for p in progress if not p.get("finished")),
                  "flagged_seen": self.flagged_total, "queued": len(self.queued), "global_cap": cap, "admitted_now": admitted,
                  "estimated_calls": calls(), "variant_queue_rows": variant_rows,
                  "heldout_queued": len(self.heldout_queued), "heldout_cap": hcap, "heldout_admitted_now": admitted_heldout,
                  "heldout_waiting": len(self.heldout_waiting), "heldout_fresh_ids_skipped": self.excluded_seen,
                  "waiting": len(self.waiting), "blocked": dict(blocked), "stopped": self.stopped_reason,
                  "by_family": {f: {"queued": n, "quota": fams.get(f, default)} for f, n in sorted(self.by_family.items(), key=str)},
                  "teacher": {k: teacher.get(k) for k in ("state", "results", "kept", "pending", "inflight", "spend_usd", "max_usd")}}
        write_atomic(Path(self.a.status), status)
        full = [f for f, v in status["by_family"].items() if v["quota"] is not None and v["queued"] >= v["quota"]]
        t = status["teacher"]
        print(f"[coord {time.strftime('%H:%M:%S')}] shards {done_markers}/{expected} pod {status['pod_items']} items "
              f"({status['pod_items_per_hour']}/h) | flagged {self.flagged_total} queued {len(self.queued)} (+{admitted}) "
              f"heldout {len(self.heldout_queued)}/{hcap} calls~{calls()}/{cap} "
              f"waiting {len(self.waiting)} full families {len(full)} | teacher {t.get('state')} results {t.get('results')} "
              f"kept {t.get('kept')} spend ${t.get('spend_usd')}/{t.get('max_usd')}"
              + (f" | STOPPED: {self.stopped_reason}" if self.stopped_reason else ""), flush=True)
        if mining_done and not self.a.no_close:
            closed = self.queue.parent / (self.queue.name + ".closed")
            if not closed.exists():
                # every .done marker is written after its shard's flagged file is complete, and this cycle's pull fetched
                # both, so everything mined has been read; items still waiting are blocked by a full quota or the cap
                closed.write_text(json.dumps({"t": time.time(), "queued": len(self.queued)}) + "\n")
                print(f"mining complete: wrote {closed.name} ({len(self.queued)} queued, {len(self.waiting)} left waiting)", flush=True)
        return status

    def run(self) -> dict:
        while True:
            status = self.cycle()
            if self.a.once:
                return status
            if (self.queue.parent / (self.queue.name + ".closed")).exists():
                if self.a.pull_scores_at_end and not self.a.no_pull:
                    print("pulling score files (all items) for analysis ...", flush=True)
                    pull(self.a.ssh, self.a.pull_from_dir or self.a.remote_dir, self.local, scores=True)
                return status
            time.sleep(self.a.interval)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ssh", help="ssh alias of the mining pod (from ~/.ssh/config)")
    ap.add_argument("--remote-dir", default="data/p3/mine")
    ap.add_argument("--no-pull", action="store_true", help="read --local-dir as is (tests, or mining output already local)")
    ap.add_argument("--pull-from-dir", help="dry run: rsync from this LOCAL directory (the fake pod's mine dir) instead of --ssh")
    ap.add_argument("--local-dir", default=str(ROOT / "data/p3/mine-pulled"))
    ap.add_argument("--queue", default=str(ROOT / "data/p3/stream/queue.jsonl"))
    ap.add_argument("--status", default=str(ROOT / "data/p3/stream/coordinator-status.json"))
    ap.add_argument("--quotas", default=str(ROOT / "data/p3/pool/quotas.json"))
    ap.add_argument("--global-cap", type=int, default=None, help=f"items to queue in total (default: quotas.json, else {DEFAULT_GLOBAL_CAP})")
    ap.add_argument("--teacher-status", default=str(ROOT / "data/p3/teacher/status.json"))
    ap.add_argument("--pool-shards", default=str(ROOT / "data/p3/pool/shards/*.jsonl"), help="to count the expected shards")
    ap.add_argument("--expected-shards", type=int, default=None)
    ap.add_argument("--exclude-ids", action="append", help="extra id files never sent to the teacher (re-read when they change); "
                    "the pool's heldout-fresh*.jsonl files are always added unless --no-default-exclude")
    ap.add_argument("--no-default-exclude", action="store_true", help="do not add <pool>/heldout-fresh*.jsonl (tests)")
    ap.add_argument("--heldout-cap", type=int, default=None,
                    help=f"flagged held-out items to label (default quotas.json heldout_flagged_cap, else {DEFAULT_HELDOUT_CAP})")
    ap.add_argument("--variants-queue", default=str(ROOT / "data/p3/stream/queue-variants.jsonl"), help="counted in the call cap")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-close", action="store_true", help="never write <queue>.closed")
    ap.add_argument("--pull-scores-at-end", action="store_true", help="also pull every *.scores.jsonl once mining is done")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if not a.no_pull and not a.ssh and not a.pull_from_dir:
        raise SystemExit("--ssh ALIAS is required unless --no-pull or --pull-from-dir")
    Coordinator(a).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
