"""Shared helpers for the decision-v1 converters in this directory.

Two jobs:
  * `store_image` — the spec's content-addressed image store (RGB JPEG q90, EXIF applied,
    downscaled with LANCZOS so width*height <= 1,000,000, never upscaled).
  * `RemoteZip` — read selected members out of a remote ZIP over HTTP byte ranges, so an
    11 GB archive costs only the bytes of the members actually wanted.
Nothing here runs a model or touches anything outside the caller's own source directory.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

MAX_PIXELS = 1_000_000
Image.MAX_IMAGE_PIXELS = 200_000_000


# --------------------------------------------------------------------------- images
def encode_image(raw: bytes) -> tuple[bytes, int, int]:
    """Decode arbitrary image bytes -> (jpeg bytes, width, height) under the storage cap."""
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), w, h


def store_image(raw: bytes, out_dir: Path) -> dict:
    """Write the re-encoded image under its own sha256 and return the record's image entry."""
    data, w, h = encode_image(raw)
    digest = hashlib.sha256(data).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{digest}.jpg"
    if not dest.exists() or dest.stat().st_size != len(data):
        dest.write_bytes(data)
    return {"image": dest.as_posix(), "sha256": digest, "width": w, "height": h}


# --------------------------------------------------------------------------- remote zip
class _Tail(io.RawIOBase):
    """Minimal seekable reader used only to parse a remote ZIP's central directory."""

    def __init__(self, url, session, size, etag):
        self.url, self.session, self.size, self.etag, self.pos = url, session, size, etag, 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, size=-1):
        n = self.size - self.pos if size is None or size < 0 else min(size, self.size - self.pos)
        if n <= 0:
            return b""
        data = _range_get(self.session, self.url, self.pos, self.pos + n - 1, self.etag)
        self.pos += n
        self.transferred += n
        return data


def _range_get(session, url, start, end, etag=None, tries=4):
    headers = {"Range": f"bytes={start}-{end}"}
    if etag:
        headers["If-Match"] = etag
    last = None
    for attempt in range(tries):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as r:
                if r.status_code != 206:
                    raise ValueError(f"server ignored byte range: {r.status_code}")
                blocks, got = [], 0
                want = end - start + 1
                for b in r.iter_content(1 << 20):
                    got += len(b)
                    if got > want:
                        raise ValueError("range exceeded requested bytes")
                    blocks.append(b)
                if got != want:
                    raise ValueError("truncated range")
                return b"".join(blocks)
        except Exception as exc:  # transient network faults on a multi-hour fetch
            last = exc
    raise RuntimeError(f"range {start}-{end} of {url} failed: {last}")


class RemoteZip:
    """Central directory over HTTP range; members fetched in coalesced runs by a small pool."""

    def __init__(self, url, cache_dir: Path):
        self.url = url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.etag = head.headers.get("ETag")
        self.transferred = 0
        self.infos = self._directory()

    def _directory(self) -> dict:
        cache = self.cache_dir / "central-directory.json"
        if cache.exists():
            meta = json.loads(cache.read_text())
            if meta.get("etag") == self.etag and meta.get("size") == self.size:
                return meta["members"]
        tail = _Tail(self.url, self.session, self.size, self.etag)
        with zipfile.ZipFile(tail) as z:
            members = {
                i.filename: dict(offset=i.header_offset, compress_size=i.compress_size,
                                 file_size=i.file_size, compress_type=i.compress_type, crc=i.CRC)
                for i in z.infolist() if not i.is_dir()
            }
        self.transferred += tail.transferred
        cache.write_text(json.dumps(dict(url=self.url, etag=self.etag, size=self.size,
                                         members=members)) + "\n")
        return members

    def _extract(self, blob: bytes, base: int, info: dict) -> bytes:
        o = info["offset"] - base
        if blob[o:o + 4] != b"PK\x03\x04":
            raise ValueError("bad local file header")
        name_len, extra_len = struct.unpack("<HH", blob[o + 26:o + 30])
        start = o + 30 + name_len + extra_len
        raw = blob[start:start + info["compress_size"]]
        if len(raw) != info["compress_size"]:
            raise ValueError("short member payload")
        data = zlib.decompressobj(-15).decompress(raw) if info["compress_type"] == 8 else raw
        if len(data) != info["file_size"] or zlib.crc32(data) != info["crc"]:
            raise ValueError("member failed CRC/size check")
        return data

    def fetch(self, names, dest: Path, workers=12, gap=4 << 20, span=48 << 20, slack=4096, progress=None):
        """Download the named members into `dest` (skipping ones already cached intact).

        The origin adds a large fixed latency per request, so neighbouring members are
        coalesced into runs of up to `span` bytes and fetched by a small thread pool.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        todo = []
        for n in names:
            info = self.infos[n]
            p = dest / Path(n).name
            if p.exists() and p.stat().st_size == info["file_size"]:
                continue
            todo.append((n, info, p))
        todo.sort(key=lambda t: t[1]["offset"])
        runs, cur = [], []
        for item in todo:
            if cur:
                last = cur[-1][1]
                end = last["offset"] + last["compress_size"] + 4096
                reach = item[1]["offset"] + item[1]["compress_size"] - cur[0][1]["offset"]
                if item[1]["offset"] - end > gap or reach > span:
                    runs.append(cur)
                    cur = []
            cur.append(item)
        if cur:
            runs.append(cur)

        done = [0]

        def run(batch):
            start = batch[0][1]["offset"]
            last = batch[-1][1]
            end = min(self.size - 1, last["offset"] + last["compress_size"] + slack + 4096)
            blob = _range_get(self.session, self.url, start, end, self.etag)
            for name, info, path in batch:
                path.write_bytes(self._extract(blob, start, info))
            return len(blob), len(batch)

        if runs:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for nbytes, count in pool.map(run, runs):
                    self.transferred += nbytes
                    done[0] += count
                    if progress and done[0] % 1000 < count:
                        progress(done[0], len(todo), self.transferred)
        return len(todo)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- option sets
def option_count(rng):
    """Vary the option count: mostly 2-12, about 10% with 13-25 (spec rule 1)."""
    return rng.randint(13, 25) if rng.random() < 0.10 else rng.randint(2, 12)


def compatible(gold: str, other: str) -> bool:
    """Reject distractors that are synonyms/substrings/word-overlaps of the gold answer."""
    g, o = gold.strip().lower(), other.strip().lower()
    if not o or g == o or g in o or o in g:
        return False
    return not (set(g.split()) & set(o.split()))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
