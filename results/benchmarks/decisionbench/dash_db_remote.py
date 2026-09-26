#!/usr/bin/env python3
"""Runs ON the pod: one clean snapshot of a DecisionBench run (called by dash_db.sh over ssh). Stages, progress, ETA, resources."""
import glob, json, os, re, subprocess, time
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30)); B="\033[1m"; D="\033[2m"; G="\033[32m"; Y="\033[33m"; R="\033[31m"; C="\033[36m"; Z="\033[0m"
L0 = "db/run.log"; USD = float(os.environ.get("POD_USD_PER_HOUR", "27.92"))
log = open(L0).read().splitlines() if os.path.exists(L0) else []
if any(l.startswith("RESTART") for l in log):
    i = max(i for i, l in enumerate(log) if l.startswith("RESTART")); log = log[i:]
text = "\n".join(log)
def has(m): return m in text
def fmt_t(sec): sec = int(max(0, sec)); return f"{sec//3600}h{(sec%3600)//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"
def bar(frac, w=34): n = int(max(0, min(1, frac)) * w); return "█" * n + "░" * (w - n)
start = time.time()
for l in log:   # first UTC timestamp line written by the runner (date -u)
    mt = re.match(r"[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d UTC \d{4}", l)
    if mt:
        start = datetime.strptime(mt.group(0), "%a %b %d %H:%M:%S UTC %Y").replace(tzinfo=timezone.utc).timestamp(); break
now = time.time(); el = now - start
ngpu = int(subprocess.run("nvidia-smi -L | grep -c '^GPU'", shell=True, capture_output=True, text=True).stdout.strip() or 8)
m = re.search(r"GPUs (\d+) x (\d+) servers = (\d+)", text); nserv = int(m.group(3)) if m else 24
gl = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
utils = [int(x.split(",")[0]) for x in gl if x.strip()]; mems = [int(x.split(",")[1]) // 1024 for x in gl if x.strip()]
serv_up = int(subprocess.run("ps -eo args | grep -c '^python scripts/playground/server.py --backend torch'", shell=True, capture_output=True, text=True).stdout.strip() or 0)
harness = int(subprocess.run("pgrep -f 'run-system-one-http' | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 0)
served = sum(int(subprocess.run(f"grep -c 'questions=' {f}", shell=True, capture_output=True, text=True).stdout.strip() or 0) for f in glob.glob("db/server*.log"))
# ---- stages
setup_done = has("SERVERS_READY"); pre_total = sum(1 for _ in open("precheck.jsonl")) if os.path.exists("precheck.jsonl") else 0
pre_done = has("PRECHECK_DONE"); pre_failed = has("FAILED precheck"); smoke_done = has("SMOKE_DONE"); all_done = has("ALL_DONE"); failed = bool(re.search(r"FAILED (harness|fla|precheck)", text))
prog = [l for l in open("db/full-summary.json").read().splitlines() if l.startswith("DECISION_BENCH_PROGRESS")] if os.path.exists("db/full-summary.json") else []
full_n = int(re.search(r"completed=(\d+)", prog[-1]).group(1)) if prog else 0; full_rps = float(re.search(r"rows_per_second=([\d.]+)", prog[-1]).group(1)) if prog else 0.0
full_total = 23900
smoke_n = 0
if os.path.exists("db/smoke/raw.jsonl"): smoke_n = sum(1 for _ in open("db/smoke/raw.jsonl"))
setup_min = re.search(r"setup (\d+) min", text)
stages = []
stages.append(("setup", "done" if setup_done else "running", f"{setup_min.group(1)} min" if setup_min else fmt_t(el)))
if pre_total:
    if pre_failed: stages.append(("pre-check", "failed", f"{pre_total} long requests at 64k"))
    elif pre_done:
        pm = re.search(r"PRECHECK (\d+) requests, (\d+) errors, (\d+)s wall, latency median ([\d.]+)s max ([\d.]+)s", text)
        stages.append(("pre-check", "done", f"{pm.group(1)} requests, {pm.group(2)} errors, {fmt_t(int(pm.group(3)))}, median {pm.group(4)}s" if pm else "done"))
    elif setup_done: stages.append(("pre-check", "running", f"{min(served, pre_total)}/{pre_total} answered"))
    else: stages.append(("pre-check", "pending", f"{pre_total} requests"))
stages.append(("smoke", "done" if smoke_done else ("running" if (pre_done or not pre_total) and setup_done and not smoke_done and harness else "pending"), f"{smoke_n} rows" if smoke_n else "100 rows"))
stages.append(("full run", "done" if all_done else ("running" if prog else "pending"), f"{full_n:,}/{full_total:,}"))
stages.append(("pack", "done" if all_done else "pending", "result tarball"))
icon = {"done": f"{G}✓{Z}", "running": f"{C}▶{Z}", "pending": f"{D}·{Z}", "failed": f"{R}✗{Z}"}
# ---- print
print(f"{B}DecisionBench 1.0 · imajev-4b phase 3{Z}   {datetime.now(IST).strftime('%H:%M:%S')} IST   elapsed {fmt_t(el)}   ~${el/3600*USD:.0f} at ${USD}/h   pod {os.uname().nodename}")
print("  " + "   ".join(f"{icon[s]} {B}{n}{Z} {D}{d}{Z}" for n, s, d in stages) + (f"   {R}{B}FAILED{Z}" if failed else "") + (f"   {G}{B}ALL DONE{Z}" if all_done else ""))
cur = next((s for s in stages if s[1] == "running"), None)
print()
if cur and cur[0] == "pre-check":
    n = min(served, pre_total); print(f"{B}now: pre-check{Z}  [{bar(n/pre_total)}] {n}/{pre_total}   the {pre_total} rows that hit the 32k limit last time, replayed at 64k; any error stops the run")
    print(f"{D}      these are the heaviest rows in the suite (35-50k tokens × 4 rotations, 10-40 s each); the tail drains on a few servers, so idle GPUs here are expected{Z}")
elif cur and cur[0] == "smoke":
    since = max(0, served - (pre_total if pre_total else 0))
    print(f"{B}now: smoke run{Z}  [{bar(min(1, since/100))}] {min(since,100)}/100 rows answered by the servers   (the harness validates the endpoint on the suite's first 100 rows, then starts the full run)")
    print(f"{D}      the harness writes its own files in batches, so smoke/raw.jsonl stays empty until the pass ends; server-side completions are counted above{Z}")
elif cur and cur[0] == "full run":
    left = (full_total - full_n) / full_rps if full_rps else 0
    print(f"{B}now: full run{Z}  [{bar(full_n/full_total)}] {full_n:,}/{full_total:,} ({100*full_n/full_total:.0f}%)   {full_rps:.1f} rows/s cumulative   ~{fmt_t(left)} left")
    errs = 0
    if os.path.exists("db/full/raw.jsonl"):
        errs = int(subprocess.run("grep -c '\"status\": \"error\"' db/full/raw.jsonl", shell=True, capture_output=True, text=True).stdout.strip() or 0)
    since = max(0, served - (pre_total if pre_total else 0) - 100)
    print(f"{D}      error rows so far: {errs} (each counts as a miss)   server-side completions this stage: {since:,} (the harness counter above updates every 100 rows){Z}")
elif all_done:
    s = {}
    try: s = json.loads("\n".join(l for l in open("db/full-summary.json").read().splitlines() if not l.startswith("DECISION_BENCH_PROGRESS")))
    except Exception: pass
    if s: print(f"{B}result{Z}  primary {100*s.get('benchmark_accuracy_counting_unsupported_as_incorrect',0):.2f}%   scored rows {s.get('successful_rows')}   errors {s.get('error_rows')}   unsupported {s.get('ineligible_rows',0)}   coverage {100*s.get('coverage',0):.2f}%")
elif not setup_done: print(f"{B}now: setup{Z}  venv, base model download, {nserv} servers loading (about 3-5 min)")
print()
busy = sum(1 for u in utils if u >= 50)
print(f"{B}resources{Z}  GPUs busy {busy}/{ngpu} (avg util {sum(utils)//max(1,len(utils))}%, mem {min(mems) if mems else 0}-{max(mems) if mems else 0} GB)   servers {serv_up}/{nserv}   harness {'running' if harness else 'idle'}   requests served {served:,}")
last = [l for l in log if l.strip() and not re.search(r"HF_TOKEN|Generating eval split|FAILED line|Terminated|^\s*$|Fetching|Downloading|nginx:", l)][-3:]
print(f"{B}last events{Z}  " + f"\n{'':<13}".join(l[:150] for l in last))
