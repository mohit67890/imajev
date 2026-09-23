"""PD12M (Spawning, `Spawning/PD12M`) -> `data/decision-v2/pd12m/images/`.

PD12M is 12.4M public-domain / CC0 image-caption pairs, re-hosted by Spawning on their own S3
bucket so that the original institutions are not crawled.  Three phases, each resumable:

  meta      download the metadata parquet shards (id, url, caption, width, height, mime_type,
            hash, license, source) into `data/decision-v2-raw/pd12m/metadata/`.
  select    read those shards, keep only rows whose per-row `license` is exactly
            `https://creativecommons.org/publicdomain/zero/1.0/`, drop anything the caption shows
            to be an artwork / scan / engraving / logo / map, size-filter to 512-4000 px on the
            long side, put each survivor in one of the 73 subject buckets
            (`scripts/v2/pd12m_captions.py`) and take them round-robin from the rarest bucket
            outwards under a per-bucket cap.  Writes `selection.jsonl` in that round-robin order,
            so a download that stops early is still balanced across buckets.
  download   fetch each image, re-encode through the v1 image store (RGB JPEG q90, <= 1 MP), and
            write `manifest.jsonl` + `manifest.json` (url + sha256) and `attribution.jsonl`
            (PD12M id, source URL, caption, licence, contributing institution).  Stops at
            `--deadline-minutes` of wall clock whatever happens.

Licence, in one line: **the images are CC0 only after the per-row filter** -- PD12M's own licence
column is a mix of CC0-1.0 and Public Domain Mark 1.0, and only CC0-1.0 is in
`scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST`.  `data/decision-v2/licenses/pd12m/` holds the
evidence and `data/decision-v2/pd12m/README.md` quotes it.

The caption is used to choose and shape the question and for nothing else.  It is never written
into a record's state (the converter passes `title=""` for this source) and never becomes a target
-- every answer in decision-v2 comes from the 9B teacher.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pyarrow.parquet as pq

from v2 import pd12m_captions as CAP
from v2.common import (OUT, RAW, append_jsonl, get_with_backoff, read_jsonl, repo_relative,
                       session, store_image)
from v2.templates import photo as T

SOURCE = "pd12m"
RAW_DIR = RAW / SOURCE
META_DIR = RAW_DIR / "metadata"
IMG_DIR = OUT / SOURCE / "images"

REPO = "https://huggingface.co/datasets/Spawning/PD12M"
SHARD_URL = REPO + "/resolve/main/metadata/pd12m.{:03d}.parquet"
SHARDS = 125  # metadata/pd12m.000.parquet .. metadata/pd12m.124.parquet

CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
PDM_URL = "https://creativecommons.org/publicdomain/mark/1.0/"
KEEP_LICENSE = {CC0_URL: "CC0-1.0"}          # the ONLY per-row licence value admitted
KEEP_MIME = {"image/jpeg", "image/png"}
COLUMNS = ["id", "url", "caption", "width", "height", "mime_type", "hash", "license", "source"]


# --------------------------------------------------------------------------- phase: meta
def run_meta(shards: list[int]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    sess = session(pool=8)
    for n in shards:
        dest = META_DIR / f"pd12m.{n:03d}.parquet"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        r = get_with_backoff(sess, SHARD_URL.format(n), timeout=(20, 600))
        dest.write_bytes(r.content)
        print(f"  {dest.name} {len(r.content)/1e6:.1f} MB", flush=True)
    print(f"{len(list(META_DIR.glob('*.parquet')))} metadata shards on disk", flush=True)


# --------------------------------------------------------------------------- phase: select
def run_select(target: int, per_bucket_cap: int, reservoir: int, seed: int,
               min_long: int, max_long: int, min_short: int) -> None:
    shards = sorted(META_DIR.glob("pd12m.*.parquet"))
    if not shards:
        raise SystemExit(f"no metadata shards in {META_DIR}; run the `meta` phase first")

    rng = random.Random(seed)
    licence_census: Counter = Counter()
    source_census: Counter = Counter()
    dropped: Counter = Counter()
    reject_reasons: Counter = Counter()
    pools: dict[str, list[dict]] = defaultdict(list)
    seen_counts: Counter = Counter()      # per-bucket count of everything seen, for reservoir R
    seen_hash: set[str] = set()
    rows_read = 0

    for shard in shards:
        table = pq.read_table(shard, columns=COLUMNS)
        ids = table.column("id").to_pylist()
        urls = table.column("url").to_pylist()
        captions = table.column("caption").to_pylist()
        widths = table.column("width").to_pylist()
        heights = table.column("height").to_pylist()
        mimes = table.column("mime_type").to_pylist()
        hashes = table.column("hash").to_pylist()
        licences = table.column("license").to_pylist()
        sources = table.column("source").to_pylist()
        rows_read += len(ids)
        for i in range(len(ids)):
            licence_census[str(licences[i])] += 1
            if licences[i] not in KEEP_LICENSE:
                dropped["licence_not_cc0"] += 1
                continue
            if mimes[i] not in KEEP_MIME:
                dropped["mime_type"] += 1
                continue
            w, h = widths[i] or 0, heights[i] or 0
            if not (min_long <= max(w, h) <= max_long) or min(w, h) < min_short:
                dropped["size_out_of_range"] += 1
                continue
            if hashes[i] in seen_hash:
                dropped["duplicate_upstream_hash"] += 1
                continue
            why = CAP.reject(captions[i])
            if why:
                dropped["caption_rejected"] += 1
                reject_reasons[why] += 1
                continue
            name = CAP.bucket(captions[i])
            if name is None:
                dropped["no_bucket"] += 1
                continue
            seen_hash.add(hashes[i])
            source_census[str(sources[i])] += 1
            row = {
                "pd12m_id": ids[i], "url": urls[i], "caption": captions[i],
                "orig_width": w, "orig_height": h, "mime_type": mimes[i],
                "upstream_md5": hashes[i], "license_url": licences[i],
                "spdx": KEEP_LICENSE[licences[i]], "contributor": sources[i] or "(not stated)",
                "bucket_label": name,
            }
            # reservoir sampling per bucket, so a bucket's members are spread over every shard
            # instead of being whatever the first shards happened to contain
            seen_counts[name] += 1
            pool = pools[name]
            if len(pool) < reservoir:
                pool.append(row)
            else:
                j = rng.randrange(seen_counts[name])
                if j < reservoir:
                    pool[j] = row
        print(f"  {shard.name}: {rows_read:,} rows read, "
              f"{sum(len(p) for p in pools.values()):,} held, {len(pools)} buckets", flush=True)

    for pool in pools.values():
        rng.shuffle(pool)
    # round-robin from the rarest bucket outwards, exactly as `fetch_openimages.py` does, so the
    # thin buckets (medical, damaged, laboratory) are exhausted before the fat ones are topped up
    order = sorted(pools, key=lambda n: (len(pools[n]), n))
    chosen: list[dict] = []
    taken: Counter = Counter()
    round_index = 0
    while len(chosen) < target:
        progressed = False
        for name in order:
            if len(chosen) >= target:
                break
            if taken[name] > round_index or taken[name] >= per_bucket_cap:
                continue
            if len(pools[name]) <= taken[name]:
                continue
            chosen.append(pools[name][taken[name]])
            taken[name] += 1
            progressed = True
        if not progressed:
            break
        round_index += 1

    # objects: what the caption claims is in the frame, and what it is silent about
    for row in chosen:
        seeded = random.Random(f"{seed}\0{row['pd12m_id']}")
        present, absent = CAP.objects(
            row["caption"], associated=T.CATEGORY_ASSOCIATED.get(row["bucket_label"], ()),
            rng=seeded)
        row["objects_present_claimed"] = present
        row["objects_absent_claimed"] = absent

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen))
    stats = {
        "target": target, "selected": len(chosen),
        "metadata_shards_read": len(shards), "rows_read": rows_read,
        "distinct_buckets": len(taken), "buckets_available": len(pools),
        "per_bucket_cap": per_bucket_cap, "reservoir_per_bucket": reservoir,
        "size_filter": {"min_long_side": min_long, "max_long_side": max_long,
                        "min_short_side": min_short},
        "dropped": dict(dropped.most_common()),
        "caption_reject_reasons": dict(reject_reasons.most_common()),
        "licence_census": dict(licence_census.most_common()),
        "bucket_distribution": dict(sorted(taken.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_contributors": dict(Counter(r["contributor"] for r in chosen).most_common(25)),
        "with_claimed_present_objects": sum(1 for r in chosen if r["objects_present_claimed"]),
        "mean_claimed_present": round(
            sum(len(r["objects_present_claimed"]) for r in chosen) / max(1, len(chosen)), 2),
    }
    (RAW_DIR / "selection_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    census = {
        "note": "per-row `license` values seen across the metadata shards actually read; the "
                "admission filter keeps only CC0-1.0",
        "shards_read": [s.name for s in shards],
        "rows_read": rows_read,
        "values": dict(licence_census.most_common()),
        "kept_value": CC0_URL,
        "public_domain_mark_value": PDM_URL,
    }
    (OUT / SOURCE).mkdir(parents=True, exist_ok=True)
    (OUT / "licenses" / SOURCE / "pd12m-license-census.json").write_text(
        json.dumps(census, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "bucket_distribution"}, indent=2))
    print(f"buckets: {len(taken)}; smallest {min(taken.values())}, largest {max(taken.values())}")


# --------------------------------------------------------------------------- phase: download
def run_download(workers: int, limit: int | None, deadline_minutes: float) -> None:
    rows = read_jsonl(RAW_DIR / "selection.jsonl")
    manifest_path = RAW_DIR / "manifest.jsonl"
    attrib_path = OUT / SOURCE / "attribution.jsonl"
    done = {r["pd12m_id"] for r in read_jsonl(manifest_path)}
    todo = [r for r in rows if r["pd12m_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(rows)} selected, {len(done)} already stored, {len(todo)} to fetch; "
          f"deadline {deadline_minutes} min", flush=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sess = session(pool=workers * 2)
    lock = threading.Lock()
    counters = Counter()
    started = time.monotonic()
    deadline = started + deadline_minutes * 60.0
    stop = threading.Event()

    def one(row):
        if stop.is_set() or time.monotonic() > deadline:
            stop.set()
            with lock:
                counters["skipped_past_deadline"] += 1
            return None, None
        try:
            r = get_with_backoff(sess, row["url"], tries=3, timeout=(20, 120))
            entry = store_image(r.content, IMG_DIR)
        except Exception as exc:
            with lock:
                counters["fail"] += 1
            return None, f"{row['pd12m_id']}: {exc}"
        with lock:
            counters["ok"] += 1
            counters["bytes"] += len(r.content)
        return ({
            "pd12m_id": row["pd12m_id"], "url": row["url"], "sha256": entry["sha256"],
            "image": repo_relative(entry["image"]), "width": entry["width"],
            "height": entry["height"], "bytes": len(r.content), "spdx": row["spdx"],
            "bucket_label": row["bucket_label"],
        }, {
            "sha256": entry["sha256"], "pd12m_id": row["pd12m_id"], "source_url": row["url"],
            "caption": row["caption"], "license": "CC0 1.0 Universal (public domain dedication)",
            "spdx": row["spdx"], "license_url": row["license_url"],
            "contributing_institution": row["contributor"],
            "dataset": "Spawning/PD12M", "dataset_url": REPO,
            "original_width": row["orig_width"], "original_height": row["orig_height"],
            "attribution_required": False,
        }), None

    batch_m, batch_a = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (payload, err) in enumerate(pool.map(one, todo), 1):
            if err:
                if counters["fail"] <= 20:
                    print(f"  ! {err}", flush=True)
                continue
            if payload is None:
                continue
            batch_m.append(payload[0])
            batch_a.append(payload[1])
            if len(batch_m) >= 250:
                append_jsonl(manifest_path, batch_m)
                append_jsonl(attrib_path, batch_a)
                batch_m, batch_a = [], []
                rate = counters["ok"] / max(1e-9, time.monotonic() - started)
                print(f"  {i}/{len(todo)} ok={counters['ok']} fail={counters['fail']} "
                      f"{rate:.1f}/s {counters['bytes']/1e9:.1f} GB "
                      f"{(time.monotonic()-started)/60:.1f} min", flush=True)
    if batch_m:
        append_jsonl(manifest_path, batch_m)
        append_jsonl(attrib_path, batch_a)

    write_manifest_json(started, deadline_minutes, counters)


def write_manifest_json(started: float | None = None, deadline_minutes: float | None = None,
                        counters: Counter | None = None) -> None:
    """The raw `manifest.json` the spec asks for: every stored image's URL and sha256."""
    rows = read_jsonl(RAW_DIR / "manifest.jsonl")
    seen, entries = set(), []
    for r in rows:
        if r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        entries.append({"pd12m_id": r["pd12m_id"], "url": r["url"], "sha256": r["sha256"],
                        "image": r["image"], "width": r["width"], "height": r["height"],
                        "bytes": r["bytes"], "bucket_label": r["bucket_label"],
                        "spdx": r["spdx"]})
    payload = {
        "source": SOURCE,
        "dataset": "Spawning/PD12M",
        "dataset_url": REPO,
        "image_host": "https://pd12m.s3.us-west-2.amazonaws.com/images/",
        "stored": len(entries),
        "manifest_rows": len(rows),
        "stored_bytes_original": sum(e["bytes"] for e in entries),
        "buckets": dict(Counter(e["bucket_label"] for e in entries).most_common()),
        "download": {
            "wall_clock_minutes": round((time.monotonic() - started) / 60.0, 2) if started else None,
            "deadline_minutes": deadline_minutes,
            "ok": counters["ok"] if counters else None,
            "fail": counters["fail"] if counters else None,
            "skipped_past_deadline": counters["skipped_past_deadline"] if counters else None,
        },
        "images": entries,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"manifest.json: {len(entries)} images, "
          f"{payload['stored_bytes_original']/1e9:.1f} GB fetched", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["meta", "select", "download", "manifest"])
    ap.add_argument("--shards", type=int, default=42, help="meta: how many shards, spread evenly")
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--per-bucket-cap", type=int, default=2000)
    ap.add_argument("--reservoir", type=int, default=3000)
    ap.add_argument("--min-long-side", type=int, default=512)
    ap.add_argument("--max-long-side", type=int, default=4000)
    ap.add_argument("--min-short-side", type=int, default=320)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deadline-minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    if args.phase == "meta":
        step = max(1, SHARDS // args.shards)
        run_meta(list(range(0, SHARDS, step))[:args.shards])
    elif args.phase == "select":
        run_select(args.target, args.per_bucket_cap, args.reservoir, args.seed,
                   args.min_long_side, args.max_long_side, args.min_short_side)
    elif args.phase == "download":
        run_download(args.workers, args.limit, args.deadline_minutes)
    else:
        write_manifest_json()
