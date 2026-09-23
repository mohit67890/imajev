"""Shared helpers for the decision-v2 (pseudo-label) image converters.

Deliberately thin: everything that already exists in v1 is imported rather than
re-implemented, so the v2 sources land in exactly the same shape as v1.

  * image store        -> `scripts/v1/_common.py:store_image` (RGB JPEG q90, <= 1 MP, EXIF applied)
  * partitioning       -> `scripts/v1_text/common.py:stable_partition` (dev 5 %, test 10 %)
  * licence receipts   -> `verified_license` below, a copy of the v1_text check with ONE change,
                          documented in `V2_COMMERCIAL_ALLOWLIST`.

Nothing here runs a model or touches anything outside `data/decision-v2*`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from v1._common import encode_image, store_image  # noqa: F401  (re-exported)
from v1_text.common import COMMERCIAL_ALLOWLIST, stable_partition, write_jsonl  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "decision-v2-raw"
OUT = ROOT / "data" / "decision-v2"

# The v1 allowlist plus CC-BY-2.0. Open Images ships only Flickr photos licensed CC BY 2.0;
# 2.0 is an attribution-only licence with no NC/ND/SA term, i.e. commercially usable on the
# same terms as 3.0/4.0, but it is not in the v1 constant and this file does not edit v1.
# Every use of it is called out in the source README and in the receipt's `notes`.
V2_COMMERCIAL_ALLOWLIST = COMMERCIAL_ALLOWLIST | {"CC-BY-2.0"}

USER_AGENT = (
    "imajev-decision-v2-collector/1.0 "
    "(research dataset collection for a non-commercial model card; crypto.valuations@gmail.com)"
)


def verified_license(evidence: Path, expected_spdx: str) -> dict:
    """`scripts/v1_text/common.py:verified_license`, widened to `V2_COMMERCIAL_ALLOWLIST`.

    The receipt must still name the exact evidence file, its sha256, the SPDX id and an
    explicit commercial-use review; a Hub tag or a dataset card alone is not enough.
    """
    evidence = Path(evidence)
    receipt = evidence.with_name(evidence.name + ".receipt.json")
    located = evidence if evidence.is_absolute() else ROOT / evidence
    located_receipt = located.with_name(located.name + ".receipt.json")
    if not located.is_file() or not located_receipt.is_file():
        raise ValueError(f"license evidence and its .receipt.json are required ({evidence})")
    meta = json.loads(located_receipt.read_text())
    digest = hashlib.sha256(located.read_bytes()).hexdigest()
    if meta.get("evidence_sha256") != digest or meta.get("spdx") != expected_spdx or meta.get("commercial_use_reviewed") is not True:
        raise ValueError(f"license receipt does not verify this evidence/SPDX/commercial-use review ({evidence})")
    if expected_spdx not in V2_COMMERCIAL_ALLOWLIST:
        raise ValueError(f"license {expected_spdx} is not in the commercial allowlist")
    return {
        "spdx": expected_spdx,
        "evidence": str(evidence),
        "evidence_sha256": digest,
        "receipt": str(receipt),
    }


def repo_relative(path) -> str:
    """Repo-relative POSIX path. Manifests and records must travel to a pod, so they may never
    carry an absolute path from this laptop."""
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def session(pool: int = 16) -> requests.Session:
    """A session that identifies itself, as the public servers we read from ask us to."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get_with_backoff(sess: requests.Session, url: str, *, params=None, tries: int = 5, timeout=(20, 120), stream=False):
    """GET with exponential backoff; 429/503 are honoured with a wait rather than a retry storm."""
    delay = 1.0
    last = None
    for attempt in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code in (429, 503):
                wait = float(r.headers.get("Retry-After", delay))
                time.sleep(min(wait, 30.0))
                delay = min(delay * 2, 30.0)
                last = RuntimeError(f"{r.status_code} from {url}")
                continue
            r.raise_for_status()
            return r
        except Exception as exc:  # transient network faults over a multi-hour fetch
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def append_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def read_jsonl(path: Path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]
