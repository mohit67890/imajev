"""Shared COCO-image plumbing for the COCO-image general-VQA group (vqav2, aokvqa, tdiuc).

Images are fetched one JPEG at a time from the public COCO S3 bucket (never the 13 GB zips),
normalised once (EXIF applied, RGB, <=1 MP, JPEG q90) into a content-addressed cache under
`.cache/datasets/v1/coco-shared/`, and then hard-linked into each source's own `images/`
directory so every source directory is self-contained.

A small ledger records, per COCO image id, whether that image was first used for `test` or for
`train`/`dev`, so an image never ends up on both sides of the split across the three sources.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

SHARED = Path(".cache/datasets/v1/coco-shared")
PROC = SHARED / "proc"          # proc/<split>/<image_id>.jpg  (normalised bytes)
META = SHARED / "meta"          # meta/<split>/<image_id>.json (sha256/width/height)
LEDGER = SHARED / "partition-ledger.json"
MAX_PIXELS = 1_000_000
BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"

_lock = threading.Lock()


# --------------------------------------------------------------------------- images
def coco_url(split: str, image_id: int) -> str:
    """`split` is one of train2014/val2014/train2017/val2017."""
    if split.endswith("2014"):
        return f"{BUCKET}/{split}/COCO_{split}_{image_id:012d}.jpg"
    return f"{BUCKET}/{split}/{image_id:012d}.jpg"


def normalise(raw: bytes) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            while w * h > MAX_PIXELS:  # guard the rounding edge
                w, h = max(1, w - 1), max(1, h - 1)
            im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), w, h


def _paths(key: str) -> tuple[Path, Path]:
    split, image_id = key.split("/")
    return PROC / split / f"{image_id}.jpg", META / split / f"{image_id}.json"


def _store(key: str, raw: bytes) -> dict:
    jpg, meta = _paths(key)
    data, w, h = normalise(raw)
    rec = dict(sha256=hashlib.sha256(data).hexdigest(), width=w, height=h, bytes=len(data))
    jpg.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    tmp = jpg.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, jpg)
    meta.write_text(json.dumps(rec))
    return rec


def put(key: str, raw: bytes) -> dict:
    """Normalise and store bytes that did not come from the COCO bucket (e.g. an HF parquet)."""
    have = cached(key)
    return have if have else _store(key, raw)


def cached(key: str) -> dict | None:
    jpg, meta = _paths(key)
    if jpg.is_file() and meta.is_file():
        try:
            return json.loads(meta.read_text())
        except Exception:
            return None
    return None


_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "imajev-decision-v1/1.0"
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        _local.s = s
    return s


def _fetch_one(key: str, retries: int = 4) -> tuple[str, dict | None]:
    have = cached(key)
    if have:
        return key, have
    split, image_id = key.split("/")
    url = coco_url(split, int(image_id))
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=60)
            if r.status_code == 404:
                return key, None
            r.raise_for_status()
            return key, _store(key, r.content)
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return key, None


def ensure(keys, workers: int = 64, label: str = "") -> dict[str, dict]:
    """keys are '<split>/<image_id>'. Returns the metadata of every image that could be fetched."""
    keys = list(dict.fromkeys(keys))
    todo = [k for k in keys if cached(k) is None]
    out = {k: m for k in keys if (m := cached(k)) is not None}
    if todo:
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, rec in pool.map(_fetch_one, todo):
                done += 1
                if rec is not None:
                    out[key] = rec
                if done % 2000 == 0:
                    mb = sum(v["bytes"] for v in out.values()) / 2**20
                    print(f"  [{label}] {done}/{len(todo)} fetched, {mb:.0f} MiB, "
                          f"{done / max(1e-9, time.time() - t0):.1f}/s", flush=True)
    return out


def link_into(source_dir: Path, key: str, rec: dict) -> dict:
    """Hard-link (copy on failure) the normalised JPEG into a source's own images/ directory."""
    dest = source_dir / "images" / f"{rec['sha256']}.jpg"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = _paths(key)[0]
        try:
            os.link(src, dest)
        except OSError:
            dest.write_bytes(src.read_bytes())
    return dict(image=str(dest), sha256=rec["sha256"], width=rec["width"], height=rec["height"])


# --------------------------------------------------------------------------- ledger
def load_ledger() -> dict[str, str]:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text())
    return {}


def save_ledger(ledger: dict[str, str]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = LEDGER.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, sort_keys=True))
        os.replace(tmp, LEDGER)


def available(ledger: dict[str, str], image_id: int, side: str) -> bool:
    """side is 'test' or 'fit'. False when an earlier source already committed this COCO image
    to the other side of the split."""
    return ledger.get(str(image_id), side) == side


def claim(ledger: dict[str, str], image_id: int, side: str) -> bool:
    """Record that this image is actually used on `side`. Only call it once a record is kept."""
    if not available(ledger, image_id, side):
        return False
    ledger[str(image_id)] = side
    return True


# --------------------------------------------------------------------------- misc
def exclusions() -> set[int]:
    d = json.loads(Path("data/decision-v1/exclusions.json").read_text())
    return set(d["coco_image_ids"])


def banned_sha() -> set[str]:
    d = json.loads(Path("data/decision-v1/exclusions.json").read_text())
    return set(d["sha256"])


def write_records(source: str, records: list[dict]) -> Path:
    root = Path("data/decision-v1") / source
    root.mkdir(parents=True, exist_ok=True)
    path = root / "records.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return path
