"""Resumable parallel range downloader.

Shared bandwidth makes a single S3 stream crawl; several ranged connections get a
fairer share. Usage: parallel_fetch.py <url> <dest> [workers]
"""
import os, sys, threading, time
import requests

url, dest = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
total = int(requests.head(url, timeout=60).headers["Content-Length"])
have = os.path.getsize(dest) if os.path.exists(dest) else 0
if have >= total:
    print(f"complete {have}"); sys.exit(0)
with open(dest, "r+b" if have else "wb") as fh:
    fh.truncate(total)

CHUNK = 32 * 1024 * 1024
jobs = [(s, min(s + CHUNK, total) - 1) for s in range(have - have % CHUNK if have else 0, total, CHUNK)]
if jobs and have:  # first job restarts at the already-fetched boundary
    jobs[0] = (have, jobs[0][1])
    if jobs[0][0] > jobs[0][1]:
        jobs.pop(0)
lock = threading.Lock(); done = [0]; start = time.time()


def run(i):
    fh = open(dest, "r+b")
    while True:
        with lock:
            if not jobs: break
            a, b = jobs.pop(0)
        for attempt in range(6):
            try:
                r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=180, stream=True)
                r.raise_for_status()
                buf = r.content
                if len(buf) != b - a + 1: raise IOError("short read")
                fh.seek(a); fh.write(buf)
                break
            except Exception as e:
                if attempt == 5: raise
                time.sleep(2 * (attempt + 1))
        with lock:
            done[0] += b - a + 1
            el = time.time() - start
            print(f"{(have + done[0]) / total:6.1%} {(have + done[0]) / 1e6:9.0f}MB {done[0] / el / 1e6:5.1f}MB/s", flush=True)
    fh.close()


threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(workers)]
for t in threads: t.start()
for t in threads: t.join()
print("complete", os.path.getsize(dest))
