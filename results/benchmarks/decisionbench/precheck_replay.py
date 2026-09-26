#!/usr/bin/env python3
"""Replay saved SystemOne requests against the pod's endpoint with N in flight; exit 1 if any request errors.
    python3 precheck_replay.py <requests.jsonl> <base_url> <concurrency>"""
import json, sys, time, urllib.request, urllib.error, statistics
from concurrent.futures import ThreadPoolExecutor
path, base, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows=[json.loads(l) for l in open(path) if l.strip()]
def one(r):
    t=time.time(); body=json.dumps(r["request"]).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(base+"/v1/systemone", data=body, headers={"Content-Type":"application/json"}), timeout=900) as f: f.read()
        return (r["row_id"], None, time.time()-t)
    except urllib.error.HTTPError as e: return (r["row_id"], f"HTTP {e.code}: {e.read()[:160]!r}", time.time()-t)
    except Exception as e: return (r["row_id"], repr(e)[:160], time.time()-t)
t0=time.time()
with ThreadPoolExecutor(conc) as ex: res=list(ex.map(one, rows))
errs=[x for x in res if x[1]]; lat=[x[2] for x in res if not x[1]]
print(f"PRECHECK {len(rows)} requests, {len(errs)} errors, {time.time()-t0:.0f}s wall, latency median {statistics.median(lat) if lat else 0:.1f}s max {max(lat) if lat else 0:.1f}s")
for x in errs[:5]: print("  ", x[0], x[1])
sys.exit(1 if errs else 0)
