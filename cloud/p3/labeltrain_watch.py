#!/usr/bin/env python3
"""Helpers for cloud/p3/bootstrap_labeltrain.sh (the label-then-train pod; reports/phase3/LABELTRAIN-RUNBOOK.md). Stdlib only.

    labeltrain_watch.py inbox     --inbox labeltrain/inbox --stream <P3>/stream-pod
        Installs the Azure leftovers the lead scp'd (queue-leftover.jsonl.gz, writer-parents-late.jsonl.gz, SHA256SUMS): verifies
        every sha256, writes each file atomically (tmp + rename, so the running teacher never reads half a file) and closes both
        queues (<file>.closed). A file NO_LEFTOVERS instead closes both as empty. Exit 0 = installed (now or earlier), 3 = nothing
        yet, 2 = a checksum mismatch (nothing installed).
    labeltrain_watch.py snapshot  --p3 <P3> --log labeltrain/throughput.log
        One IST status line (teacher results / kept / pending / items per hour / completion tok/s / servers up, variants and
        writer counts, pilot gate) appended to the log and a JSON line to throughput.jsonl next to it; prints the line.
        Exit 5 when the pilot gate is paused (the orchestrator writes the PILOT_GATE_PAUSED marker).
    labeltrain_watch.py gpu-free  [--max-mib 2048]        exit 0 when every GPU uses less than --max-mib (nvidia-smi)
    labeltrain_watch.py summary   --p3 <P3>               the LABEL_DONE summary (JSON)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

INSTALL = ("queue-leftover.jsonl", "writer-parents-late.jsonl")


def ist(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime((time.time() if t is None else t) + 19800)) + " IST"


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p, "rb") as fh:
        return sum(1 for line in fh if line.endswith(b"\n") and line.strip())


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def inbox(a) -> int:
    box, stream = Path(a.inbox), Path(a.stream)
    stream.mkdir(parents=True, exist_ok=True)
    if all((stream / f"{n}.closed").exists() for n in INSTALL):
        return 0
    if (box / "NO_LEFTOVERS").exists():
        for n in INSTALL:
            if not (stream / n).exists():
                (stream / n).write_text("")
            (stream / f"{n}.closed").write_text(json.dumps({"t": time.time(), "rows": count_lines(stream / n),
                                                            "why": "NO_LEFTOVERS"}) + "\n")
        print(f"{ist()} inbox: NO_LEFTOVERS -> both leftover queues closed empty", flush=True)
        return 0
    sums = box / "SHA256SUMS"
    if not sums.exists() or not all((box / f"{n}.gz").exists() for n in INSTALL):
        return 3
    want = {}
    for line in sums.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            want[name.strip()] = h
    for n in INSTALL:
        got = hashlib.sha256((box / f"{n}.gz").read_bytes()).hexdigest()
        if want.get(f"{n}.gz") != got:
            print(f"{ist()} inbox: sha256 mismatch for {n}.gz ({got} != {want.get(n + '.gz')}): nothing installed "
                  "(an scp still running? it is retried)", flush=True)
            return 2
    for n in INSTALL:
        tmp = stream / f".{n}.tmp"
        with gzip.open(box / f"{n}.gz", "rb") as src, open(tmp, "wb") as dst:
            while True:
                block = src.read(1 << 20)
                if not block:
                    break
                dst.write(block)
        os.replace(tmp, stream / n)
        (stream / f"{n}.closed").write_text(json.dumps({"t": time.time(), "rows": count_lines(stream / n)}) + "\n")
    print(f"{ist()} inbox: installed " + ", ".join(f"{n} ({count_lines(stream / n)} rows)" for n in INSTALL), flush=True)
    return 0


def snapshot(a) -> int:
    p3 = Path(a.p3)
    t = load(p3 / "teacher-pod/status.json")
    gate = load(p3 / "teacher-pod/pilot-gate.json").get("decision", "pending")
    var_rows = count_lines(p3 / "stream-pod/queue-variants.jsonl")
    constructed = count_lines(p3 / "variants-pod/constructed.jsonl")
    parents = Counter()
    pp = p3 / "variants-pod/parents.jsonl"
    if pp.exists():
        for line in open(pp):
            try:
                parents[json.loads(line).get("method")] += 1
            except json.JSONDecodeError:
                pass
    cons = load(p3 / "teacher-pod-consistency/consistency.json")
    servers = t.get("servers") or {}
    row = {"t": time.time(), "ist": ist(), "state": t.get("state"), "results": t.get("results"), "kept": t.get("kept"),
           "pending": t.get("pending"), "inflight": t.get("inflight"), "items_per_hour": t.get("items_per_hour"),
           "completion_tok_per_s": t.get("completion_tok_per_s"), "second_solves": t.get("second_solves"),
           "servers_up": sum(1 for v in servers.values() if v), "servers": len(servers), "by_origin": t.get("by_origin"),
           "variant_queue_rows": var_rows, "constructed_variants": constructed, "parents": dict(parents), "pilot_gate": gate,
           "consistency": {k: cons.get(k) for k in ("items", "solve1_argmax_agreement", "keep_verdict_agreement")} if cons else None}
    line = (f"{row['ist']} teacher {row['state']} results {row['results']} kept {row['kept']} pending {row['pending']} "
            f"inflight {row['inflight']} | {row['items_per_hour']} items/h {row['completion_tok_per_s']} tok/s | servers "
            f"{row['servers_up']}/{row['servers']} | variants queued {var_rows} constructed {constructed} parents {dict(parents)} "
            f"| gate {gate}" + (f" | FP8-vs-bf16 {row['consistency']}" if cons else ""))
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")
    with open(log.with_suffix(".jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(line, flush=True)
    return 5 if gate == "paused" else 0


def gpu_free(a) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"nvidia-smi failed: {e}", flush=True)
        return 1
    used = {}
    for line in out.strip().splitlines():
        i, m = [x.strip() for x in line.split(",")]
        used[int(i)] = int(float(m))
    busy = {i: m for i, m in used.items() if m >= a.max_mib}
    print(f"{ist()} GPU memory used (MiB): {used}" + (f"; BUSY {busy}" if busy else "; all free"), flush=True)
    return 0 if used and not busy else 1


def summary(a) -> int:
    p3 = Path(a.p3)
    t = load(p3 / "teacher-pod/status.json")
    ks = load(p3 / "teacher-pod/kept-summary.json")
    rep = {"ist": ist(), "teacher_state": t.get("state"), "results": t.get("results"), "kept_training_rows": ks.get("kept"),
           "heldout_results": ks.get("heldout_results"), "by_origin": t.get("by_origin"), "second_solves": t.get("second_solves"),
           "items_per_hour_last_run": t.get("items_per_hour"), "completion_tok_per_s_last_run": t.get("completion_tok_per_s"),
           "variant_queue_rows": count_lines(p3 / "stream-pod/queue-variants.jsonl"),
           "constructed_variants": count_lines(p3 / "variants-pod/constructed.jsonl"),
           "consistency": load(p3 / "teacher-pod-consistency/consistency.json") or None,
           "queues": {p.name: count_lines(p) for p in sorted((p3 / "stream-pod").glob("*.jsonl"))}}
    print(json.dumps(rep, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inbox")
    i.add_argument("--inbox", required=True)
    i.add_argument("--stream", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--p3", required=True)
    s.add_argument("--log", required=True)
    g = sub.add_parser("gpu-free")
    g.add_argument("--max-mib", type=int, default=2048)
    m = sub.add_parser("summary")
    m.add_argument("--p3", required=True)
    a = ap.parse_args(argv)
    return {"inbox": inbox, "snapshot": snapshot, "gpu-free": gpu_free, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
