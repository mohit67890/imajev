"""Remote archive cache identity, exercised against real ZIP bytes and HTTP ranges."""
import io
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from v1._common import RemoteZip


def archive(name, data):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zipped:
        zipped.writestr(name, data)
    return out.getvalue()


@contextmanager
def archives_server(archives):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(archives[self.path])))
            self.end_headers()

        def do_GET(self):
            data = archives[self.path]
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            chunk = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.end_headers()
            self.wfile.write(chunk)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_same_size_archives_without_etags_do_not_share_a_directory(tmp_path):
    first = archive("first.bin", b"one")
    second = archive("other.bin", b"two")
    assert len(first) == len(second)
    with archives_server({"/first.zip": first, "/other.zip": second}) as base:
        with RemoteZip(base + "/first.zip", tmp_path / "cache") as zipped:
            assert set(zipped.infos) == {"first.bin"}
        with RemoteZip(base + "/other.zip", tmp_path / "cache") as zipped:
            assert set(zipped.infos) == {"other.bin"}
            assert zipped.fetch(["other.bin"], tmp_path / "members") == 1
            assert (tmp_path / "members/other.bin").read_bytes() == b"two"
