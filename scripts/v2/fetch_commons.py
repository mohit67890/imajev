"""Wikimedia Commons -> `data/decision-v2/commons_photos/images/` (+ attribution sidecar).

Two phases, both resumable and both safe to re-run:

  metadata  for each of 46 subject seeds, ask the Commons search API for files that are BOTH in the
            subject's category tree (`deepcategory:`) and in one of the three licence maintenance
            categories (`CC-BY-4.0`, `CC-BY-3.0`, `CC-Zero`); then verify every candidate against
            the API's own `imageinfo` -> `extmetadata` block and keep ONLY files whose
            `LicenseShortName` is CC0, CC BY 3.0 or CC BY 4.0 *and* whose machine-readable
            `License` tag agrees.  CC BY-SA, CC BY 2.0/2.5, GFDL, PD-*, any NC/ND variant, anything
            with a non-empty `Restrictions` field and anything unknown are dropped and counted.
            The membership query is only a way of finding candidates; the admission decision is
            always made per file from `extmetadata`.
            `deepcategory:` silently returns almost nothing for very large category trees, so each
            seed also carries a plain full-text fallback query, used only when the tree query
            cannot fill the seed's quota.
  download  fetch the 1024 px rendering of each kept file, re-encode through the v1 image store
            (RGB JPEG q90, <= 1 MP) and write `manifest.jsonl` (url + sha256) and
            `attribution.jsonl` (author, licence, file page) beside it.

Politeness: one serial metadata request at a time with `maxlag=5`, a small sleep between them, a
descriptive User-Agent, at most 6 concurrent thumbnail downloads, exponential backoff on 429/503.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, append_jsonl, get_with_backoff, read_jsonl, repo_relative, session, store_image

API = "https://commons.wikimedia.org/w/api.php"
SOURCE = "commons_photos"
RAW_DIR = RAW / SOURCE
IMG_DIR = OUT / SOURCE / "images"

# LicenseShortName -> SPDX. Nothing else is admitted; in particular CC BY-SA, CC BY 2.0/2.5,
# GFDL, "Public domain", PD-*, and any NC/ND variant are refused.
LICENSE_MAP = {
    "CC0": "CC0-1.0",
    "CC0 1.0": "CC0-1.0",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY 4.0": "CC-BY-4.0",
}
MACHINE_TAGS = {"CC0-1.0": {"cc0"}, "CC-BY-3.0": {"cc-by-3.0"}, "CC-BY-4.0": {"cc-by-4.0"}}

# Not photographs of a scene/object: Commons mixes these into otherwise photographic categories.
TITLE_REJECT = re.compile(
    r"(coat of arms|\bflag of\b|\blogo\b|\bmap of\b|\bdiagram\b|\bchart\b|\bicon\b|\bblazon\b|"
    r"\bcrest\b|\bemblem\b|\bseal of\b|\bgraph\b|\bplot of\b|\bscreenshot\b|\bstamp of\b)",
    re.I,
)
ALLOWED_MIME = {"image/jpeg", "image/png"}

# 46 seed categories.  The label is what the converter uses as the photo's coarse scene/object
# category (for pairing two-image records and for building plausible distractors); it is NEVER a
# target -- the teacher decides every answer.
SEEDS = [
    ("street_scene", "Category:Street scenes", "street"),
    ("kitchen", "Category:Kitchens", "kitchen"),
    ("car", "Category:Automobiles", "car"),
    ("bicycle", "Category:Bicycles", "bicycle"),
    ("bus", "Category:Buses", "bus"),
    ("train", "Category:Trains", "train"),
    ("boat", "Category:Boats", "boat"),
    ("aircraft", "Category:Aircraft", "aircraft"),
    ("dog", "Category:Dogs", "dog"),
    ("cat", "Category:Cats", "cat"),
    ("bird", "Category:Birds", "bird"),
    ("farm_animal", "Category:Livestock", "cattle"),
    ("prepared_food", "Category:Food", "food"),
    ("vegetable", "Category:Vegetables", "vegetable"),
    ("fruit", "Category:Fruits", "fruit"),
    ("beverage", "Category:Drinks", "drink"),
    ("hand_tool", "Category:Hand tools", "tool"),
    ("power_tool", "Category:Power tools", "drill"),
    ("sign", "Category:Signs", "sign"),
    ("road_sign", "Category:Road signs", "roadsign"),
    ("document", "Category:Documents", "document"),
    ("storefront", "Category:Shop fronts", "storefront"),
    ("supermarket", "Category:Supermarkets", "supermarket"),
    ("restaurant_interior", "Category:Restaurant interiors", "restaurant"),
    ("clothing", "Category:Clothing", "clothing"),
    ("footwear", "Category:Shoes", "shoes"),
    ("electronics", "Category:Consumer electronics", "electronics"),
    ("computer_hardware", "Category:Computer hardware", "computer"),
    ("furniture", "Category:Furniture", "furniture"),
    ("living_room", "Category:Living rooms", "livingroom"),
    ("bedroom", "Category:Bedrooms", "bedroom"),
    ("bathroom", "Category:Bathrooms", "bathroom"),
    ("office_interior", "Category:Offices", "office"),
    ("sport", "Category:Sports", "sport"),
    ("medical_device", "Category:Medical equipment", "medical"),
    ("laboratory", "Category:Laboratories", "laboratory"),
    ("damaged_item", "Category:Damage", "damaged"),
    ("construction_site", "Category:Construction sites", "construction"),
    ("machine", "Category:Machines", "machine"),
    ("musical_instrument", "Category:Musical instruments", "instrument"),
    ("kitchenware", "Category:Kitchenware", "kitchenware"),
    ("toy", "Category:Toys", "toy"),
    ("houseplant", "Category:Houseplants", "houseplant"),
    ("book", "Category:Books", "book"),
    ("packaging", "Category:Packaging", "packaging"),
    ("waste_container", "Category:Waste containers", "bin"),
]

LICENSE_CATEGORIES = ("CC-BY-4.0", "CC-BY-3.0", "CC-Zero")
SEARCH_PAGE = 50
MAX_OFFSET = 3000
TAG = re.compile(r"<[^>]+>")


def plain(html: str | None, limit: int = 300) -> str:
    if not html:
        return ""
    text = TAG.sub(" ", html)
    text = (
        text.replace("&amp;", "&").replace("&quot;", '"').replace("&#039;", "'")
        .replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", text).strip()[:limit]


def api(sess, params, sleep=0.10):
    base = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5"}
    for _ in range(6):
        r = get_with_backoff(sess, API, params={**base, **params})
        data = r.json()
        if "error" in data and data["error"].get("code") == "maxlag":
            time.sleep(5)
            continue
        time.sleep(sleep)
        return data
    raise RuntimeError("Commons API stayed lagged")


def classify(info: dict) -> tuple[str | None, str]:
    """-> (spdx, reason).  spdx is None when the file must be skipped."""
    if info.get("mime") not in ALLOWED_MIME:
        return None, "mime"
    w, h = info.get("width", 0) or 0, info.get("height", 0) or 0
    if w < 400 or h < 400 or w * h < 240_000:
        return None, "too_small"
    meta = info.get("extmetadata", {})
    short = plain(meta.get("LicenseShortName", {}).get("value"), 60)
    if short not in LICENSE_MAP:
        return None, f"license:{short or 'missing'}"
    spdx = LICENSE_MAP[short]
    machine = (meta.get("License", {}).get("value") or "").strip().lower()
    if machine not in MACHINE_TAGS[spdx]:
        return None, f"license_tag_mismatch:{short}/{machine or 'missing'}"
    if plain(meta.get("Restrictions", {}).get("value"), 200):
        return None, "restrictions"
    return spdx, "ok"


def search_titles(sess, query: str, want: int, stats: dict) -> list[str]:
    titles, offset = [], 0
    while len(titles) < want and offset < MAX_OFFSET:
        try:
            data = api(sess, {"list": "search", "srsearch": query, "srnamespace": "6",
                              "srlimit": str(SEARCH_PAGE), "sroffset": str(offset)})
        except Exception as exc:
            stats["search_errors"] = stats.get("search_errors", 0) + 1
            print(f"  ! search {query[:60]}: {exc}", flush=True)
            break
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        titles += [h["title"] for h in hits]
        offset += len(hits)
        if "continue" not in data:
            break
    return titles[:want]


def title_chunks(titles: list[str], max_titles: int = 50, max_chars: int = 5000):
    """Commons takes up to 50 titles per request, but the request is a GET: a batch of long
    filenames (Hong Kong and Shenzhen photographs routinely run past 200 characters) overruns the
    URI limit and the whole batch is lost to a 414.  Chunk on both count and length."""
    chunk, size = [], 0
    for title in titles:
        cost = len(title) * 3 + 3  # worst-case percent-encoding plus the separator
        if chunk and (len(chunk) >= max_titles or size + cost > max_chars):
            yield chunk
            chunk, size = [], 0
        chunk.append(title)
        size += cost
    if chunk:
        yield chunk


def file_info(sess, titles: list[str]) -> dict:
    """Batch `imageinfo` -> {title: imageinfo}."""
    out = {}
    for chunk in title_chunks(titles):
        try:
            data = api(sess, {
                "titles": "|".join(chunk), "prop": "imageinfo",
                "iiprop": "extmetadata|url|size|mime", "iiurlwidth": "1024",
                "iiextmetadatafilter": "License|LicenseShortName|UsageTerms|LicenseUrl|Artist|Credit|Restrictions|ImageDescription",
            })
        except Exception as exc:  # one over-long or rejected batch must not lose the whole seed
            print(f"  ! imageinfo batch of {len(chunk)} skipped: {str(exc)[:120]}", flush=True)
            continue
        for page in data.get("query", {}).get("pages", []):
            infos = page.get("imageinfo") or []
            if infos:
                out[page["title"]] = (page.get("pageid"), infos[0])
    return out


def harvest(sess, label: str, category: str, terms: str, target: int, seen: set[int], stats: dict) -> list[dict]:
    """Candidate queries in order: the category tree per licence, then a full-text fallback."""
    name = category.split(":", 1)[1]
    queries = [f'deepcategory:"{name}" incategory:"{lic}" filetype:bitmap' for lic in LICENSE_CATEGORIES]
    queries += [f'incategory:"{lic}" intitle:{terms} filetype:bitmap' for lic in LICENSE_CATEGORIES]
    queries += [f'incategory:"{lic}" {terms} filetype:bitmap' for lic in LICENSE_CATEGORIES]
    kept: list[dict] = []
    seen_titles: set[str] = set()
    for query in queries:
        if len(kept) >= target:
            break
        want = min(target - len(kept), max(target // 2, 200))
        # licence categories are not perfectly clean, so ask for headroom and verify every file
        titles = [t for t in search_titles(sess, query, int(want * 1.6) + 40, stats) if t not in seen_titles]
        seen_titles |= set(titles)
        if not titles:
            continue
        stats["searched"] = stats.get("searched", 0) + len(titles)
        try:
            infos = file_info(sess, titles)
        except Exception as exc:
            stats["info_errors"] = stats.get("info_errors", 0) + 1
            print(f"  ! imageinfo {label}: {exc}", flush=True)
            continue
        for title in titles:
            if len(kept) >= target:
                break
            if title not in infos:
                continue
            pageid, info = infos[title]
            if pageid is None or pageid in seen:
                continue
            if TITLE_REJECT.search(title):
                stats["skip_title"] = stats.get("skip_title", 0) + 1
                continue
            spdx, reason = classify(info)
            if spdx is None:
                key = reason.split(":")[0]
                stats[f"skip_{key}"] = stats.get(f"skip_{key}", 0) + 1
                if reason.startswith("license:"):
                    stats.setdefault("license_seen", {})
                    seen_name = reason.split(":", 1)[1]
                    stats["license_seen"][seen_name] = stats["license_seen"].get(seen_name, 0) + 1
                continue
            seen.add(pageid)
            meta = info.get("extmetadata", {})
            kept.append({
                "pageid": pageid,
                "title": title,
                "category_label": label,
                "seed_category": category,
                "query": query,
                "spdx": spdx,
                "license_short_name": plain(meta.get("LicenseShortName", {}).get("value"), 60),
                "license_url": plain(meta.get("LicenseUrl", {}).get("value"), 200),
                "usage_terms": plain(meta.get("UsageTerms", {}).get("value"), 200),
                "artist": plain(meta.get("Artist", {}).get("value")) or "(not stated on the file page)",
                "credit": plain(meta.get("Credit", {}).get("value"), 200),
                "description": plain(meta.get("ImageDescription", {}).get("value"), 400),
                "file_page": info.get("descriptionurl", ""),
                "thumb_url": (info.get("thumburl") or "").split("?")[0],
                "original_url": (info.get("url") or "").split("?")[0],
                "orig_width": info.get("width"),
                "orig_height": info.get("height"),
            })
    return kept


def run_metadata(per_seed: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / "metadata.jsonl"
    done_path = RAW_DIR / "metadata_progress.json"
    progress = json.loads(done_path.read_text()) if done_path.is_file() else {"seeds": {}, "stats": {}}
    seen = {row["pageid"] for row in read_jsonl(meta_path)}
    sess = session()
    for label, category, terms in SEEDS:
        have = progress["seeds"].get(label, 0)
        if have >= per_seed:
            continue
        stats: dict = {}
        t0 = time.time()
        kept = harvest(sess, label, category, terms, per_seed - have, seen, stats)
        append_jsonl(meta_path, kept)
        progress["seeds"][label] = have + len(kept)
        progress["stats"][label] = stats
        done_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"{label:22s} +{len(kept):5d} (total {progress['seeds'][label]}) in {time.time()-t0:5.0f}s", flush=True)
    print(f"metadata rows: {len(read_jsonl(meta_path))}", flush=True)


# Why the URL the API hands back is used verbatim, and not rewritten to a smaller width:
# upload.wikimedia.org no longer renders an arbitrary thumbnail width on demand.  Only widths that
# already exist for that file are served; everything else is a 400.  Probing one file gave
# 500 px OK, 640 px 400, 800 px 400, 960 px OK, 1024 px 400, 1280 px OK -- i.e. the set is
# per-file, not a global list.  The metadata phase asks for `iiurlwidth=1024`, MediaWiki rounds up
# to the bucket it has (usually 1280 px, about 260-380 kB), and that URL is what works.  Rewriting
# it cost three failed requests per file before the fallback, which was four times slower, not
# faster.  Storage still applies the 1 MP cap, and the training/teacher pixel budget is 400,000 px.
def fetch_url(row: dict) -> str:
    return row["thumb_url"]


def run_download(workers: int, limit: int | None) -> None:
    meta_path = RAW_DIR / "metadata.jsonl"
    manifest_path = RAW_DIR / "manifest.jsonl"
    attrib_path = OUT / SOURCE / "attribution.jsonl"
    rows = read_jsonl(meta_path)
    done = {row["pageid"] for row in read_jsonl(manifest_path)}
    todo = [r for r in rows if r["pageid"] not in done and r["thumb_url"]]
    if limit:
        todo = todo[:limit]
    print(f"{len(rows)} known, {len(done)} already stored, {len(todo)} to fetch", flush=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sess = session(pool=workers * 2)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}

    def one(row):
        try:
            url = fetch_url(row)
            r = get_with_backoff(sess, url, timeout=(20, 180))
            entry = store_image(r.content, IMG_DIR)
        except Exception as exc:
            with lock:
                counters["fail"] += 1
            return None, f"{row['pageid']}: {exc}"
        manifest = {
            "pageid": row["pageid"], "title": row["title"], "category_label": row["category_label"],
            "url": url, "api_thumb_url": row["thumb_url"], "original_url": row["original_url"],
            "sha256": entry["sha256"],
            "image": repo_relative(entry["image"]), "width": entry["width"], "height": entry["height"],
            "bytes": len(r.content), "spdx": row["spdx"],
        }
        attribution = {
            "sha256": entry["sha256"], "title": row["title"], "author": row["artist"],
            "license": row["license_short_name"], "spdx": row["spdx"], "license_url": row["license_url"],
            "file_page": row["file_page"], "source_url": url, "credit": row["credit"],
        }
        with lock:
            counters["ok"] += 1
        return (manifest, attribution), None

    batch_m, batch_a = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (payload, err) in enumerate(pool.map(one, todo), 1):
            if err:
                if counters["fail"] <= 20:
                    print(f"  ! {err}", flush=True)
                continue
            batch_m.append(payload[0])
            batch_a.append(payload[1])
            if len(batch_m) >= 200:
                append_jsonl(manifest_path, batch_m)
                append_jsonl(attrib_path, batch_a)
                batch_m, batch_a = [], []
                print(f"  {i}/{len(todo)} ok={counters['ok']} fail={counters['fail']}", flush=True)
    if batch_m:
        append_jsonl(manifest_path, batch_m)
        append_jsonl(attrib_path, batch_a)
    print(f"stored ok={counters['ok']} fail={counters['fail']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["metadata", "download"])
    ap.add_argument("--per-seed", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.phase == "metadata":
        run_metadata(args.per_seed)
    else:
        run_download(args.workers, args.limit)
