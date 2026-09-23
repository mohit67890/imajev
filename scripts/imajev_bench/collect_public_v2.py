"""Collect recent CC0/CC BY Wikimedia Commons photos for imajev-bench v2, excluding imajev training data.

Phases (each writes new files only):
  search    Commons full-text search per query; keep files whose own extmetadata licence is CC0,
            CC BY 3.0 or CC BY 4.0, uploaded and (when recorded) taken on/after --uploaded-after,
            at least --min-side px, and NOT in the exclusion inventory: page IDs harvested for imajev
            training (data/decision-v2-raw/commons_photos/metadata.jsonl), any photo by a
            photographer whose work was harvested (same shoot = same scene risk), and v1 benchmark
            sources. Writes candidates.jsonl.
  sheet     candidates.html: thumbnails with checkboxes; "Export" downloads accepted.json. A human
            picks photos that can carry benchmark questions (legible text, countable objects).
  download  For accepted IDs: fetch the full file, correct orientation, strip metadata, cap the long
            side at 2048 px, reject near-duplicates of imajev training images (dHash), and write
            images plus sources.json entries for `imajev_bench assemble`.

Commons API etiquette: serial requests, maxlag=5, descriptive User-Agent, backoff on 429/503.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "imajev-bench-v2-collector/0.1 (research benchmark; contact via project maintainer)"
HOSTS = {"commons.wikimedia.org", "upload.wikimedia.org"}
LICENSES = {"CC0": "CC0-1.0", "CC0 1.0": "CC0-1.0", "CC BY 3.0": "CC-BY-3.0", "CC BY 4.0": "CC-BY-4.0"}
TITLE_REJECT = re.compile(r"(coat of arms|\bflag of\b|\blogo\b|\bmap\b|\bdiagram\b|\bchart\b|\bicon\b|\bscreenshot\b|"
                          r"\bportrait\b|\bselfie\b|\bpeople\b|\bcrowd\b)", re.I)
TRAINING_METADATA = ROOT / "data/decision-v2-raw/commons_photos/metadata.jsonl"
TRAINING_IMAGES = ROOT / "data/decision-v2/commons_photos/images"
V1_REGISTRIES = sorted((ROOT / "scripts/imajev_bench").glob("fresh_sources*.json"))

# Scenes that can carry text-reading, counting, state, spatial and rule questions.
DEFAULT_QUERIES = [
    "menu board prices", "price list cafe", "notice board", "bus timetable", "opening hours sign",
    "parking sign times", "shop shelf price labels", "nutrition facts label", "product packaging label",
    "whiteboard schedule", "recycling bins", "vending machine", "parcel shipping label", "gauge dial",
    "control panel buttons", "museum label", "instructions sign", "departure board", "kitchen jars labels",
    "bookshelf spines", "market stall prices", "fuel prices sign", "warning sign text", "elevator buttons",
    "street name sign", "water meter", "thermostat display", "spice rack", "tool wall", "bicycle rack",
]


def plain(value, limit=400):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", plain(text).lower()).strip("-")[:60] or "unknown"


def http_get(url, opener=urllib.request.urlopen, sleep=time.sleep, retries=5):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS:
        raise ValueError(f"Refusing non-Commons URL {url}")
    for attempt in range(retries + 1):
        try:
            with opener(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        sleep(min(60.0, 2.0 ** attempt))


def api(params, **kwargs):
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2", "maxlag": "5"})
    data = json.loads(http_get(f"{API}?{query}", **kwargs))
    if "error" in data:
        raise RuntimeError(f"Commons API error: {data['error'].get('info')}")
    return data


def exclusion_inventory(training_metadata=TRAINING_METADATA, registries=V1_REGISTRIES):
    pageids, titles, artists = set(), set(), set()
    if Path(training_metadata).is_file():
        for line in Path(training_metadata).read_text().splitlines():
            row = json.loads(line)
            pageids.add(int(row["pageid"]))
            titles.add(row["title"])
            if row.get("artist"):
                artists.add(slug(row["artist"]))
    for registry in registries:
        titles.update(re.findall(r"File:[^\"|\]]+?\.(?:jpe?g|png|webp)", Path(registry).read_text(), re.I))
    return {"pageids": pageids, "titles": titles, "artists": artists}


def _date(value):
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", plain(value))
    return match.group(0) if match else None


def classify(page, cutoff, min_side, exclude):
    """Return (candidate, None) or (None, rejection reason)."""
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})
    title = page.get("title", "")
    license_name = plain(meta.get("LicenseShortName", {}).get("value"))
    if license_name not in LICENSES:
        return None, "license"
    if plain(meta.get("Restrictions", {}).get("value")):
        return None, "restrictions"
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None, "mime"
    if min(info.get("width", 0), info.get("height", 0)) < min_side:
        return None, "too_small"
    if TITLE_REJECT.search(title) or TITLE_REJECT.search(plain(meta.get("ImageDescription", {}).get("value"))):
        return None, "subject"
    uploaded = (info.get("timestamp") or "")[:10]
    taken = _date(meta.get("DateTimeOriginal", {}).get("value"))
    if not uploaded or uploaded < cutoff or (taken and taken < cutoff):
        return None, "too_old"
    artist = plain(meta.get("Artist", {}).get("value")) or info.get("user", "")
    if page.get("pageid") in exclude["pageids"] or title in exclude["titles"]:
        return None, "in_training_or_v1"
    if slug(artist) in exclude["artists"]:
        return None, "photographer_in_training"
    return {"pageid": page["pageid"], "title": title,
            "file_page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "thumb_url": info.get("thumburl"), "original_url": info.get("url"), "mime": info.get("mime"),
            "width": info.get("width"), "height": info.get("height"), "artist": artist, "uploader": info.get("user"),
            "license": license_name, "spdx": LICENSES[license_name],
            "license_url": plain(meta.get("LicenseUrl", {}).get("value")), "uploaded": uploaded, "taken": taken,
            "description": plain(meta.get("ImageDescription", {}).get("value"))}, None


def run_search(queries, cutoff, per_query, min_side, output, exclude, **http):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {}
    with output.open("x") as handle:
        for query in queries:
            params = {"action": "query", "generator": "search", "gsrnamespace": "6", "gsrlimit": "50",
                      "gsrsearch": f"{query} filetype:bitmap", "gsrsort": "create_timestamp_desc",
                      "prop": "imageinfo", "iiprop": "url|timestamp|user|extmetadata|size|mime", "iiurlwidth": "480"}
            kept, offset = 0, 0
            while kept < per_query and offset < 500:
                data = api({**params, "gsroffset": str(offset)}, **http)
                pages = data.get("query", {}).get("pages", [])
                if not pages:
                    break
                for page in pages:
                    if page.get("pageid") in seen:
                        continue
                    seen.add(page.get("pageid"))
                    row, reason = classify(page, cutoff, min_side, exclude)
                    stats[reason or "kept"] = stats.get(reason or "kept", 0) + 1
                    if row and kept < per_query:
                        handle.write(json.dumps({**row, "query": query}) + "\n")
                        kept += 1
                if "continue" not in data:
                    break
                offset = int(data["continue"].get("gsroffset", offset + 50))
                http.get("sleep", time.sleep)(0.2)
    return stats


def write_sheet(candidates, output):
    rows = [json.loads(line) for line in Path(candidates).read_text().splitlines()]
    cards = "".join(
        f'<label class="card"><input type="checkbox" value="{r["pageid"]}"><img loading="lazy" src="{html.escape(r["thumb_url"] or "")}">'
        f'<span>{html.escape(r["title"][5:80])}<br>{html.escape(r["query"])} · {r["width"]}×{r["height"]} · {r["uploaded"]}'
        f' · {html.escape(r["license"])}</span><a href="{html.escape(r["file_page"])}" target="_blank">source</a></label>'
        for r in rows)
    Path(output).write_text(f"""<!doctype html><meta charset="utf-8"><title>imajev-bench v2 candidates</title>
<style>body{{font:14px system-ui;margin:16px;background:#f6f6f6}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:6px}}
.card img{{width:100%;height:180px;object-fit:contain;background:#eee}}.card:has(input:checked){{outline:3px solid #2a7}}
header{{position:sticky;top:0;background:#f6f6f6;padding:8px 0}}</style>
<header><b>{len(rows)} candidates.</b> Tick photos that can carry questions (legible text, countable objects, clear states; no identifiable people or personal data). <button id="export">Export accepted.json</button> <span id="count"></span></header>
<div class="grid">{cards}</div>
<script>const boxes=[...document.querySelectorAll('input')];const upd=()=>document.getElementById('count').textContent=boxes.filter(b=>b.checked).length+' selected';
boxes.forEach(b=>b.onchange=upd);document.getElementById('export').onclick=()=>{{const ids=boxes.filter(b=>b.checked).map(b=>+b.value);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({{accepted:ids}},null,1)],{{type:'application/json'}}));a.download='accepted.json';a.click()}};upd();</script>""")
    return len(rows)


def normalise(blob, max_side=2048):
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(blob)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=92, optimize=True)  # no EXIF/metadata written
        return out.getvalue(), image.size


def training_hashes(cache: Path, images_dir=TRAINING_IMAGES):
    from imajev_bench.lint import dhash
    if cache.is_file():
        return {int(k, 16) for k in json.loads(cache.read_text())}
    values = [dhash(p) for p in sorted(Path(images_dir).glob("*.jpg"))] if Path(images_dir).is_dir() else []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([f"{v:016x}" for v in values]))
    return set(values)


def run_download(candidates, accepted, output_dir, hash_cache, max_bits=6, **http):
    from imajev_bench.lint import dhash
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=False)
    wanted = set(json.loads(Path(accepted).read_text())["accepted"])
    rows = [r for r in (json.loads(line) for line in Path(candidates).read_text().splitlines()) if r["pageid"] in wanted]
    known = training_hashes(Path(hash_cache))
    sources, rejected = [], []
    for row in rows:
        blob = http_get(row["original_url"], **{k: v for k, v in http.items() if k in ("opener", "sleep")})
        if len(blob) > 40 * 1024 * 1024:
            rejected.append({"pageid": row["pageid"], "reason": "too_large"})
            continue
        data, size = normalise(blob)
        sha = hashlib.sha256(data).hexdigest()
        path = output_dir / "images" / f"{sha[:16]}.jpg"
        path.write_bytes(data)
        value = dhash(path)
        if any(bin(value ^ other).count("1") <= max_bits for other in known):
            path.unlink()
            rejected.append({"pageid": row["pageid"], "reason": "near_duplicate_of_training"})
            continue
        day = row.get("taken") or row["uploaded"]
        sources.append({"id": f"commons-{row['pageid']}", "path": f"images/{path.name}",
                        # Same photographer on the same day is treated as one scene.
                        "source_cluster": f"commons-{slug(row['artist'])}-{day}",
                        "provenance": {"source": "Wikimedia Commons", "source_page": row["file_page"],
                                       "creator": row["artist"], "license": row["license"], "spdx": row["spdx"],
                                       "license_url": row["license_url"], "uploaded": row["uploaded"],
                                       "taken": row.get("taken"), "original_sha256": hashlib.sha256(blob).hexdigest(),
                                       "normalised": f"exif-transposed, metadata stripped, JPEG q92, {size[0]}x{size[1]}",
                                       "retrieved_at": datetime.now(timezone.utc).isoformat()}})
    (output_dir / "sources.json").write_text(json.dumps(sources, indent=1) + "\n")
    (output_dir / "download-receipt.json").write_text(json.dumps(
        {"accepted": len(wanted), "downloaded": len(sources), "rejected": rejected,
         "training_hashes_checked": len(known), "near_duplicate_bits": max_bits}, indent=1) + "\n")
    return len(sources), rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)
    s = sub.add_parser("search")
    s.add_argument("--uploaded-after", required=True, help="YYYY-MM-DD; after the newest evaluated model's cutoff")
    s.add_argument("--per-query", type=int, default=40)
    s.add_argument("--min-side", type=int, default=1000)
    s.add_argument("--queries", type=Path, help="JSON list of search phrases (default: built-in list)")
    s.add_argument("--output", type=Path, required=True)
    h = sub.add_parser("sheet")
    h.add_argument("--candidates", type=Path, required=True)
    h.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("download")
    d.add_argument("--candidates", type=Path, required=True)
    d.add_argument("--accepted", type=Path, required=True)
    d.add_argument("--output-dir", type=Path, required=True)
    d.add_argument("--hash-cache", type=Path, default=ROOT / "data/imajev-bench/v2-sources/training-dhash.json")
    args = parser.parse_args(argv)
    if args.phase == "search":
        queries = json.loads(args.queries.read_text()) if args.queries else DEFAULT_QUERIES
        stats = run_search(queries, args.uploaded_after, args.per_query, args.min_side, args.output, exclusion_inventory())
        print(json.dumps(stats))
    elif args.phase == "sheet":
        print(write_sheet(args.candidates, args.output), "candidates ->", args.output)
    else:
        count, rejected = run_download(args.candidates, args.accepted, args.output_dir, args.hash_cache)
        print(json.dumps({"downloaded": count, "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
