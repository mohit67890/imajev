"""Shared helpers for the text-rich / native-MC / held-out converters in decision-v1.

Every converter that imports this stays deterministic from its own seed: nothing here
consults the clock, the filesystem ordering, or an unseeded RNG.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from vision_decision.contracts import Request

ROOT = Path("data/decision-v1")
CACHE = Path(".cache/datasets/v1")
MAX_PIXELS = 1_000_000
JPEG_QUALITY = 90


# --------------------------------------------------------------------------- images
class ImageStore:
    """Content-addressed RGB JPEG store, <=1 MP, EXIF orientation applied."""

    def __init__(self, source: str):
        self.dir = ROOT / source / "images"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._by_key: dict[str, dict] = {}

    def add(self, data, key: str) -> dict:
        """`data` is raw bytes or a PIL image; `key` dedupes repeat calls for one source image."""
        hit = self._by_key.get(key)
        if hit is not None:
            return hit
        img = Image.open(io.BytesIO(data)) if isinstance(data, (bytes, bytearray)) else data
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        raw = buf.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        path = self.dir / f"{sha}.jpg"
        if not path.exists():
            path.write_bytes(raw)
        rec = {"image": str(path), "sha256": sha, "width": img.width, "height": img.height}
        self._by_key[key] = rec
        return rec


# --------------------------------------------------------------------------- options
_WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def conflicts(gold: str, candidate: str) -> bool:
    """True when `candidate` is a synonym/substring variant of `gold` and so not a clean wrong answer."""
    a, b = norm(gold), norm(candidate)
    if not b or not a:
        return True
    if a == b or a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    return bool(wa & wb) and (wa <= wb or wb <= wa)


def clean_pool(gold, pool, limit: int | None = None) -> list[str]:
    """Distractor candidates, deduped on normalised text, with gold-conflicting entries dropped.

    `gold=None` means the item has no correct option at all (the source itself says the question is
    unanswerable), so there is nothing to protect against and every candidate is a valid wrong one.
    """
    out, seen = [], set() if gold is None else {norm(gold)}
    for c in pool:
        c = str(c).strip()
        if not c or len(c) > 128 or c == "__unknown__":
            continue
        n = norm(c)
        if n in seen or (gold is not None and conflicts(gold, c)):
            continue
        seen.add(n)
        out.append(c)
        if limit and len(out) >= limit:
            break
    return out


def option_budget(rng) -> int:
    """Total option-set size: mostly 2-12, about 10% in the 13-25 band (spec rule 1)."""
    return rng.randint(13, 25) if rng.random() < 0.10 else rng.randint(2, 12)


def choice_field(rng, question, gold, distractors, *, abstain: bool, budget: int | None = None,
                 keep_all: bool = False, descriptions: dict | None = None):
    """Build one choice field. Returns (field, target, cause) or None when no valid set exists.

    abstain=True removes the gold option entirely -> `not_listed`.
    keep_all=True uses exactly the supplied distractors (native MC: do not resample the option count).
    gold=None means the source says nothing on the list is right; pass abstain=True with it and
    record the cause the source proves (e.g. `insufficient_evidence`) rather than `not_listed`.
    """
    if gold is None and not abstain:
        raise ValueError("an item with no gold answer must be an abstention item")
    pool = clean_pool(gold, distractors)
    if keep_all:
        values = list(pool) if abstain else [gold, *pool]
    else:
        n = budget if budget is not None else option_budget(rng)
        take = n if abstain else n - 1
        values = pool[:max(0, take)]
        if not abstain:
            values.append(gold)
    if len(values) < 2 or len(values) > 25:
        return None
    rng.shuffle(values)
    opts = []
    for v in values:
        o = {"value": v}
        if descriptions and descriptions.get(v):
            o["description"] = descriptions[v]
        opts.append(o)
    field = {"id": "answer", "type": "choice", "question": question, "options": opts}
    return field, (None if abstain else gold), ("not_listed" if abstain else None)


def native_choice(rng, question, options, gold):
    """Native human-written option set kept verbatim (no resampling, no synonym pruning).

    `gold` is the correct option's text, or None when the item is an abstention item.
    Returns (field, target) or None when the option set is unusable.
    """
    values, seen = [], set()
    for o in options:
        o = str(o).strip()
        if not o or len(o) > 128 or o == "__unknown__" or norm(o) in seen:
            continue
        seen.add(norm(o))
        values.append(o)
    if not 2 <= len(values) <= 25:
        return None
    if gold is not None and gold not in values:
        return None
    rng.shuffle(values)
    return {"id": "answer", "type": "choice", "question": question,
            "options": [{"value": v} for v in values]}, gold


def ordinal_field(question, levels, gold, *, abstain: bool):
    """levels: list of (value, description) ascending. abstain=True means gold is outside the window."""
    field = {"id": "answer", "type": "ordinal", "question": question,
             "levels": [{"value": v, "description": d} for v, d in levels]}
    if abstain:
        return field, None, "not_listed"
    return field, gold, None


def count_window(rng, gold: int, *, lo: int = 0, hi: int = 20, width: int | None = None,
                 outside: bool = False):
    """A window of consecutive integers (2..7 levels). outside=True puts gold outside it."""
    width = width or rng.randint(3, 7)
    if outside:
        # place the window entirely above or below gold, staying inside [lo, hi]
        options = []
        if gold - width >= lo:
            options.append(gold - width)
        if gold + 1 + width - 1 <= hi:
            options.append(gold + 1)
        if not options:
            return None
        start = rng.choice(options)
    else:
        start = gold - rng.randint(0, width - 1)
    start = max(lo, min(start, hi - width + 1))
    window = list(range(start, start + width))
    if outside and gold in window:
        return None
    if not outside and gold not in window:
        return None
    return window


def count_levels(window):
    def label(n):
        return "none" if n == 0 else ("exactly one" if n == 1 else f"exactly {n}")
    return [(n, label(n)) for n in window]


# --------------------------------------------------------------------------- records
def make_record(*, source, uid, source_split, source_group, family, license, images, field,
                target, cause, source_answer, partition, request_id=None, target_distribution=None):
    request = {"request_id": (request_id or f"{source}-{uid}")[:128], "state": {}, "fields": [field]}
    Request.model_validate(request)
    rec = {
        "id": f"{source}:{uid}", "source": source, "source_split": source_split,
        "source_group": str(source_group), "family": family, "license": license,
        "images": images, "request": request, "target": target, "abstention_cause": cause,
        "source_answer": str(source_answer)[:2000], "partition": partition,
    }
    if target_distribution:
        rec["target_distribution"] = target_distribution
    return rec


def write(source: str, records: list[dict]) -> dict:
    out = ROOT / source
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary = {
        "source": source, "records": len(records),
        "partitions": dict(Counter(r["partition"] for r in records)),
        "families": dict(Counter(r["family"] for r in records)),
        "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in records)),
        "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
        "option_counts": dict(sorted(Counter(
            len(r["request"]["fields"][0].get("options", r["request"]["fields"][0].get("levels", [])))
            for r in records).items())),
        "images": len({i["sha256"] for r in records for i in r["images"]}),
        "two_image": sum(len(r["images"]) == 2 for r in records),
    }
    (out / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def excluded_sha256() -> set[str]:
    return set(json.loads((ROOT / "exclusions.json").read_text())["sha256"])
