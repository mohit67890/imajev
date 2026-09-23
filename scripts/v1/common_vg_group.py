"""Shared helpers for the Visual-Genome-family decision-v1 converters.

Used by convert_gqa.py, convert_tallyqa.py, convert_vg_attributes.py,
convert_pixmo_count.py and convert_vsr.py.  Nothing here is source specific:
image normalisation (spec: RGB JPEG q90, EXIF applied, width*height <= 1e6),
the leakage exclusion sets (with the Visual Genome <-> COCO id mapping), and a
small threaded HTTP fetcher.
"""
from __future__ import annotations

import hashlib
import io
import random
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

MAX_PIXELS = 1_000_000
DATA_ROOT = Path("data/decision-v1")
CACHE_ROOT = Path(".cache/datasets/v1")
VG_COMMON = CACHE_ROOT / "vg_common"          # shared VG image cache (this agent's five sources)
VG_PROC = VG_COMMON / "proc"                  # <vg_image_id>.jpg, already normalised
IMAGE_DATA_ZIP = Path(".cache/datasets/tdiuc/image_data-uw.json.zip")

Image.MAX_IMAGE_PIXELS = 400_000_000


# ---------------------------------------------------------------- images

def normalise(raw: bytes) -> tuple[bytes, int, int] | None:
    """Decode arbitrary image bytes into the storage form the spec mandates."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if w < 8 or h < 8:
                return None
            if w * h > MAX_PIXELS:                      # never upscale
                scale = (MAX_PIXELS / (w * h)) ** 0.5
                im = im.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return buf.getvalue(), im.width, im.height
    except Exception:
        return None


def store_normalised(out_dir: Path, data: bytes, width: int, height: int) -> dict:
    """Write already-normalised bytes content-addressed and return the images[] entry."""
    sha = hashlib.sha256(data).hexdigest()
    path = out_dir / f"{sha}.jpg"
    if not path.exists():
        tmp = out_dir / f".{sha}.tmp"
        tmp.write_bytes(data)
        tmp.replace(path)
    return {"image": str(path), "sha256": sha, "width": width, "height": height}


def store_cached(out_dir: Path, path: Path) -> dict | None:
    """Store an already-normalised cache file without a second JPEG round-trip."""
    try:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        return None
    return store_normalised(out_dir, data, w, h)


def store_raw(out_dir: Path, raw: bytes) -> dict | None:
    got = normalise(raw)
    if got is None:
        return None
    return store_normalised(out_dir, *got)


# ---------------------------------------------------------------- leakage

def load_exclusions() -> dict:
    """COCO ids, VG ids and sha256s that must never reach train/dev, cross-mapped by coco_id."""
    excl = json.loads((DATA_ROOT / "exclusions.json").read_text())
    coco = set(excl["coco_image_ids"])
    vg = set(excl["visual_genome_image_ids"])
    for rec in vg_image_data():
        cid = rec.get("coco_id")
        if cid is None:
            continue
        if rec["image_id"] in vg:
            coco.add(int(cid))
        if int(cid) in coco:
            vg.add(rec["image_id"])
    return {"coco": coco, "vg": vg, "sha256": set(excl["sha256"])}


_IMAGE_DATA: list | None = None


def vg_image_data() -> list:
    """Visual Genome image_data.json (image_id, url, coco_id, width, height)."""
    global _IMAGE_DATA
    if _IMAGE_DATA is None:
        with zipfile.ZipFile(IMAGE_DATA_ZIP) as z:
            _IMAGE_DATA = json.loads(z.read("image_data.json"))
    return _IMAGE_DATA


def vg_urls() -> dict[int, str]:
    return {r["image_id"]: r["url"] for r in vg_image_data()}


# ---------------------------------------------------------------- fetching

_local = threading.local()
_deadline: float | None = None


def set_fetch_deadline(seconds: float | None) -> None:
    """Bound the total wall-clock a converter may spend fetching.

    PixMo-Count's only host (Flickr) answers 429 to a sustained stream and does not recover
    within a session, so an unbounded fetch loop would never finish.  Past the deadline `fetch`
    returns None immediately, the converter completes with whatever is cached, and re-running it
    later tops the cache up.
    """
    global _deadline
    _deadline = None if seconds is None else time.time() + seconds


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (decision-v1 dataset build)"
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.s = s
    return _local.s


def fetch(url: str, attempts: int = 4, timeout: int = 40, throttled_attempts: int = 14) -> bytes | None:
    """HTTP GET with backoff.

    429 is treated as "come back in a moment", not as a dead URL.  Flickr - PixMo-Count's only
    host - serves a token bucket: measured over 240 requests it answers ~33% with 200 and the
    rest with 429 at any concurrency from 4 to 24 threads, so a short-retry policy would record
    two thirds of a live dataset as link rot.  The wait stays short on purpose: long exponential
    sleeps put every worker to sleep at once and throughput collapses to ~1/s.
    """
    throttles = 0
    i = 0
    while i < attempts and throttles < throttled_attempts:
        if _deadline is not None and time.time() > _deadline:
            return None
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (403, 404, 410):
                return None
            if r.status_code == 429:
                throttles += 1
                time.sleep(0.4 + random.random() * 1.6)
                continue
        except Exception:
            pass
        i += 1
        time.sleep(1.5 * i)
    return None


def fetch_many(jobs, workers: int = 48, log_every: int = 2000):
    """jobs: iterable of (key, url). Yields (key, bytes|None) as they complete."""
    jobs = list(jobs)
    done = 0
    start = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for key, blob in zip((k for k, _ in jobs), ex.map(lambda ju: fetch(ju[1]), jobs)):
            done += 1
            if log_every and done % log_every == 0:
                rate = done / max(1e-9, time.time() - start)
                print(f"    fetched {done}/{len(jobs)}  {rate:.1f}/s", flush=True)
            yield key, blob


def fetch_normalised_many(jobs, workers: int = 32, log_every: int = 2000):
    """Like fetch_many, but decode/resize/re-encode happen inside the pool.

    Flickr-sized sources spend far more time in Pillow than on the wire; doing that work in the
    main thread caps throughput at a couple of images a second and buffers raw bytes in memory.
    Yields (key, (jpeg_bytes, width, height) | None) in job order.
    """
    jobs = list(jobs)
    done = 0
    start = time.time()

    def work(ju):
        blob = fetch(ju[1])
        return normalise(blob) if blob else None

    with ThreadPoolExecutor(workers) as ex:
        for key, got in zip((k for k, _ in jobs), ex.map(work, jobs)):
            done += 1
            if log_every and done % log_every == 0:
                print(f"    processed {done}/{len(jobs)}  "
                      f"{done / max(1e-9, time.time() - start):.1f}/s", flush=True)
            yield key, got


# ---------------------------------------------------------------- misc

def stable_rank(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def coco_url(image_id: int, split: str) -> str:
    return (f"https://s3.amazonaws.com/images.cocodataset.org/{split}2014/"
            f"COCO_{split}2014_{image_id:012d}.jpg")


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty"]


def number_word(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)
