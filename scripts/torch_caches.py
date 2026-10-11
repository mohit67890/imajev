"""Exact per-question result cache for the torch serving path (stdlib only).

One entry per (shared evidence context, exact typed question). The context covers
the artifact namespace, the serving config and the image bytes, so any weight,
adapter, prompt or evidence change invalidates. Thinking-active requests bypass.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

RESULT_CACHE_SCHEMA = "aimino-imajev-result-cache-v2"


def _clone_result(value):
    copier = getattr(value, "model_copy", None)
    return copier(deep=True) if copier is not None else copy.deepcopy(value)


def _image_digest(image):
    digest = hashlib.sha256()
    digest.update(str(getattr(image, "mode", "")).encode())
    digest.update(repr(getattr(image, "size", None)).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def artifact_namespace(bundle, adapter=None):
    """Cheap startup fingerprint used to invalidate persistent cache entries."""
    digest = hashlib.sha256(RESULT_CACHE_SCHEMA.encode())
    paths = [Path(bundle)]
    if adapter is not None and Path(adapter).is_dir():
        paths.extend(sorted(Path(adapter).rglob("*")))
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(str(path).encode("utf-8", "surrogateescape"))
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        if path.suffix.lower() in {".json", ".txt"} and stat.st_size <= 2_000_000:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_context_digest(*, namespace, model, adapter, rotations, fast, merge_lora,
                         shared_prefix, microbatch, prompt_layout, readout_codes, state, images):
    """Shared evidence context for one request: config, namespace, state, image bytes."""
    payload = {
        "cache_schema": RESULT_CACHE_SCHEMA,
        "artifact_namespace": namespace,
        "model": model,
        "adapter": adapter,
        "rotations": rotations,
        "fast": bool(fast),
        "merge_lora": bool(merge_lora),
        "shared_prefix": bool(shared_prefix),
        "question_microbatch": microbatch,
        "prompt_layout": prompt_layout,
        "readout_codes": readout_codes,
        "state": state,
        "images": [_image_digest(image) for image in images],
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).digest()


def field_cache_key(context_digest, field):
    field_bytes = json.dumps(field.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(context_digest)
    digest.update(b"\0")
    digest.update(field_bytes)
    return digest.hexdigest()


class ResultCache:
    """Bounded in-memory LRU with optional SQLite backing."""

    def __init__(self, capacity, path=None, result_type=None):
        self.capacity = int(capacity)
        self._items = OrderedDict()
        self.path = None if not path else Path(path).expanduser().resolve()
        self.result_type = result_type
        self._db = None
        if self.capacity > 0 and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS result_cache ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, last_access REAL NOT NULL)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS result_cache_lru ON result_cache(last_access)")
            self._db.commit()
            self._prune_disk()

    @property
    def persistent(self):
        return self._db is not None

    def _remember(self, key, value):
        self._items.pop(key, None)
        self._items[key] = _clone_result(value)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def _prune_disk(self):
        if self._db is None or self.capacity <= 0:
            return
        self._db.execute(
            "DELETE FROM result_cache WHERE cache_key IN ("
            "SELECT cache_key FROM result_cache ORDER BY last_access DESC LIMIT -1 OFFSET ?)",
            (self.capacity,),
        )
        self._db.commit()

    def get(self, key):
        if self.capacity <= 0:
            return None
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return _clone_result(value)
        if self._db is None or self.result_type is None:
            return None
        row = self._db.execute(
            "SELECT payload FROM result_cache WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            value = self.result_type.model_validate(json.loads(row[0]))
        except Exception:
            # Never trust an incompatible/corrupt persisted judgment.
            self._db.execute("DELETE FROM result_cache WHERE cache_key = ?", (key,))
            self._db.commit()
            return None
        self._db.execute(
            "UPDATE result_cache SET last_access = ? WHERE cache_key = ?", (time.time(), key))
        self._db.commit()
        self._remember(key, value)
        return _clone_result(value)

    def put(self, key, value):
        if self.capacity <= 0:
            return
        self._remember(key, value)
        if self._db is None:
            return
        dumper = getattr(value, "model_dump", None)
        if not callable(dumper):
            return
        payload = json.dumps(dumper(mode="json"), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        self._db.execute(
            "INSERT INTO result_cache(cache_key, payload, last_access) VALUES(?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, last_access=excluded.last_access",
            (key, payload, time.time()),
        )
        self._db.commit()
        self._prune_disk()

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __len__(self):
        return len(self._items)
