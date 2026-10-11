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


def test_same_size_corruption_is_refetched_and_valid_bytes_are_reused(tmp_path):
    data = b"valid member bytes"
    zipped_bytes = archive("image.bin", data)
    with archives_server({"/source.zip": zipped_bytes}) as base:
        with RemoteZip(base + "/source.zip", tmp_path / "cache") as zipped:
            dest = tmp_path / "members"
            assert zipped.fetch(["image.bin"], dest) == 1
            member = dest / "image.bin"
            assert member.read_bytes() == data
            member.write_bytes(b"x" * len(data))
            assert zipped.fetch(["image.bin"], dest) == 1
            assert member.read_bytes() == data
            assert zipped.fetch(["image.bin"], dest) == 0
