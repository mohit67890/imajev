"""Resumable parallel range downloader.

Usage: parallel_fetch.py <url> <dest> [workers]. Incomplete bytes live in
<dest>.part with a progress journal; only a completed download replaces <dest>.
"""
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CHUNK = 32 * 1024 * 1024


def download(url, dest, workers=12, chunk=CHUNK, attempts=6):
    if workers < 1 or chunk < 1 or attempts < 1:
        raise ValueError("workers, chunk and attempts must be positive")
    dest = Path(dest)
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    if total < 0:
        raise ValueError("negative Content-Length")
    if dest.exists() and dest.stat().st_size == total:
        print(f"complete {total}")
        return
    part = dest.with_name(dest.name + ".part")
    journal = dest.with_name(dest.name + ".part.json")
    etag = response.headers.get("ETag")
    modified = response.headers.get("Last-Modified")
    identity = {"url": url, "total": total, "chunk": chunk,
                "etag": etag, "last_modified": modified}
    starts = list(range(0, total, chunk))
    done = set()
    if part.exists() and part.stat().st_size == total and journal.exists():
        state = json.loads(journal.read_text())
        if all(state.get(k) == v for k, v in identity.items()):
            done = set(state.get("done", []))
            if not done.issubset(starts):
                raise ValueError("invalid download progress")
        else:
            part.unlink()
    if not part.exists() or not journal.exists():
        # Preserve the old downloader's contiguous-prefix resume convention.
        have = dest.stat().st_size if dest.exists() else 0
        if 0 < have < total:
            shutil.copyfile(dest, part)
            done = {s for s in starts if s + chunk <= have}
        else:
            part.write_bytes(b"")
    with part.open("r+b") as output:
        output.truncate(total)
    lock = threading.Lock()

    def save_progress():
        temp = journal.with_name(journal.name + ".tmp")
        temp.write_text(json.dumps({**identity, "done": sorted(done)}) + "\n")
        temp.replace(journal)

    save_progress()

    def fetch(start):
        end = min(start + chunk, total) - 1
        headers = {"Range": f"bytes={start}-{end}"}
        if etag:
            headers["If-Match"] = etag
        elif modified:
            headers["If-Unmodified-Since"] = modified
        for attempt in range(attempts):
            try:
                with requests.get(url, headers=headers, timeout=180, stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise IOError("server ignored byte range")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise IOError("server returned a different byte range")
                    data = response.content
                    if len(data) != end - start + 1:
                        raise IOError("short read")
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        with lock:
            with part.open("r+b") as output:
                output.seek(start)
                output.write(data)
                output.flush()
            done.add(start)
            save_progress()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, start) for start in starts if start not in done]
        for future in futures:
            future.result()  # worker errors must make the command fail
    part.replace(dest)
    journal.unlink()
    print("complete", total)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        raise SystemExit("Usage: parallel_fetch.py <url> <dest> [workers]")
    download(args[0], args[1], int(args[2]) if len(args) == 3 else 12)


if __name__ == "__main__":
    main()
