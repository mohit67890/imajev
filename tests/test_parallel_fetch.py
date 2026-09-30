"""Range downloader failures and resumes through real local HTTP and the CLI."""
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v1/parallel_fetch.py"


@contextmanager
def range_server(data):
    state = {"fail": True, "ranges": [], "data": data, "etag": "v1", "wrong_range": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["data"])))
            self.send_header("ETag", state["etag"])
            self.end_headers()

        def do_GET(self):
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            state["ranges"].append((start, end))
            failing = state["fail"]
            if (failing(start, end) if callable(failing) else failing):
                self.send_error(503)
                return
            if self.headers.get("If-Match", state["etag"]) != state["etag"]:
                self.send_error(412)
                return
            chunk = state["data"][start:end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            offset = 1 if state["wrong_range"] else 0
            self.send_header("Content-Range", f"bytes {start + offset}-{end + offset}/{len(state['data'])}")
            self.end_headers()
            self.wfile.write(chunk)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/archive", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_cli(url, dest, workers=2):
    # Remove retry delays while retaining real network requests and CLI exit semantics.
    launcher = "import runpy,sys,time;time.sleep=lambda _:None;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    return subprocess.run([sys.executable, "-c", launcher, str(SCRIPT), url, str(dest), str(workers)],
                          capture_output=True, text=True, timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_worker_failure_does_not_publish_a_completed_file(tmp_path):
    data = b"complete archive bytes"
    dest = tmp_path / "archive.bin"
    with range_server(data) as (url, state):
        failed = run_cli(url, dest)
        assert failed.returncode != 0, failed.stdout + failed.stderr
        assert not dest.exists()
        state["fail"] = False
        succeeded = run_cli(url, dest)
        assert succeeded.returncode == 0, succeeded.stdout + succeeded.stderr
        assert dest.read_bytes() == data


def test_zero_workers_cannot_report_completion(tmp_path):
    with range_server(b"archive bytes") as (url, _):
        response = run_cli(url, tmp_path / "archive.bin", workers=0)
        assert response.returncode != 0, response.stdout + response.stderr
        assert not (tmp_path / "archive.bin").exists()


def test_resume_retries_only_unfinished_chunks(tmp_path):
    import json
    import pytest
    from v1.parallel_fetch import download
    data = b"abcdefghijkl"
    dest = tmp_path / "archive.bin"
    with range_server(data) as (url, state):
        state["fail"] = lambda start, end: start == 4
        with pytest.raises(Exception):
            download(url, dest, workers=1, chunk=4, attempts=1)
        assert not dest.exists()
        progress = json.loads((tmp_path / "archive.bin.part.json").read_text())
        assert progress["done"] == [0, 8]
        state["ranges"].clear()
        state["fail"] = False
        download(url, dest, workers=1, chunk=4, attempts=1)
        assert state["ranges"] == [(4, 7)]
        assert dest.read_bytes() == data
        assert not (tmp_path / "archive.bin.part.json").exists()


def test_unjournaled_preallocation_is_not_trusted(tmp_path):
    from v1.parallel_fetch import download
    data = b"archive bytes"
    dest = tmp_path / "archive.bin"
    (tmp_path / "archive.bin.part").write_bytes(b"\0" * len(data))
    with range_server(data) as (url, state):
        state["fail"] = False
        download(url, dest, workers=1, chunk=4, attempts=1)
        assert dest.read_bytes() == data


def test_changed_etag_discards_partial_bytes(tmp_path):
    import pytest
    from v1.parallel_fetch import download
    dest = tmp_path / "archive.bin"
    with range_server(b"abcdefghijkl") as (url, state):
        state["fail"] = lambda start, end: start == 4
        with pytest.raises(Exception):
            download(url, dest, workers=1, chunk=4, attempts=1)
        state.update(fail=False, data=b"ABCDEFGHIJKL", etag="v2", ranges=[])
        download(url, dest, workers=1, chunk=4, attempts=1)
        assert state["ranges"] == [(0, 3), (4, 7), (8, 11)]
        assert dest.read_bytes() == b"ABCDEFGHIJKL"


def test_wrong_content_range_cannot_publish_bytes(tmp_path):
    import pytest
    from v1.parallel_fetch import download
    dest = tmp_path / "archive.bin"
    with range_server(b"archive bytes") as (url, state):
        state.update(fail=False, wrong_range=True)
        with pytest.raises(OSError, match="different byte range"):
            download(url, dest, workers=1, attempts=1)
        assert not dest.exists()
