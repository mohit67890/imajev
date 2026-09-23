"""Shopping Queries Image Dataset (SQID) + Amazon ESCI -> decision-v1 ordinal records.

ESCI is the one *native* ordinal signal in the product group, and it grades a
product judgment rather than a perceptual one: given a real Amazon search query and
a product, a human annotator said the product is Exact, Substitute, Complement or
Irrelevant. Mapped to ascending levels that is
`0 Irrelevant < 1 Complement < 2 Substitute < 3 Exact`, which is exactly the
`ordinal` field's shape.

Two repositories are joined:
* `amazon-science/esci-data` (Apache-2.0) — the judgments, via the GitHub LFS media
  endpoint, `shopping_queries_dataset_examples.parquet`.
* `crossingminds/shopping-queries-image-dataset` (MIT) — `product_id -> image_url`.
  The images themselves are Amazon CDN assets fetched here; SQID grants no rights in
  them.

Only `product_locale == 'us'` (English) rows are used, which is also the only locale
SQID covers. Dead URLs, the "no image available" placeholder and the known default
digital-video artwork are skipped.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_sqid_esci.py
"""
import hashlib, io, json, random, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")
import pandas as pd
import requests
from PIL import Image, ImageOps
from vision_decision.contracts import Request

SOURCE = "sqid_esci"
SEED = 20260922
CACHE = Path(".cache/datasets/v1/sqid_esci")
RAW = CACHE / "images"
OUT = Path("data/decision-v1/sqid_esci")
IMG_DIR = OUT / "images"
LICENSE = "Apache-2.0 (ESCI judgments) / MIT (SQID image URLs); product images are Amazon CDN assets, rights not granted"
TARGET_PER_LEVEL = 2_500
CANDIDATES_PER_LEVEL = 3_400     # headroom for dead URLs
NOT_LISTED_RATE = 0.17
DEV_FRACTION = 0.03
TEST_GROUP_PERMILLE = 150        # single family, so the test slice is taken here
TEST_CAP = 300                   # ~300 per family, one family
WORKERS = 16
# 446 product ids point at this placeholder; the SQID card names it explicitly.
PLACEHOLDER = "https://m.media-amazon.com/images/G/01/digital/video/web/Default_Background_Art_LTR._SX1080_FMjpg_.jpg"

LABELS = {"I": 0, "C": 1, "S": 2, "E": 3}
LEVEL_TEXT = {
    0: "Irrelevant — the product does not match the query at all",
    1: "Complement — the product does not match the query, but would be used together with something that does",
    2: "Substitute — the product does not fully match the query, but could be bought instead of something that does",
    3: "Exact — the product matches everything the query asks for",
}
TEMPLATES = [
    "How well does this product match the search query \"{q}\"?",
    "A shopper searched for \"{q}\". How well does the product in the photo match that search?",
    "Rate how well the pictured product answers the search query \"{q}\".",
    "Someone typed \"{q}\" into a shopping search. How good a result is this product?",
    "Judge the relevance of the product shown to the query \"{q}\".",
    "Given the search \"{q}\", how relevant is the item in this photo?",
]


def levels_for(window):
    return [{"value": v, "description": LEVEL_TEXT[v]} for v in window]


def pick_window(gold, rng):
    """Full 0-3 scale, or a consecutive sub-window that EXCLUDES gold (-> not_listed)."""
    if rng.random() >= NOT_LISTED_RATE:
        return [0, 1, 2, 3], None
    options = [w for w in ([0, 1], [0, 1, 2], [1, 2], [1, 2, 3], [2, 3])
               if gold not in w]
    if not options:
        return [0, 1, 2, 3], None
    return rng.choice(options), "not_listed"


def load_pairs():
    ex = pd.read_parquet(CACHE / "esci_examples.parquet")
    ex = ex[(ex.small_version == 1) & (ex.split == "test") & (ex.product_locale == "us")]
    urls = pd.concat([pd.read_parquet(CACHE / "product_image_urls.parquet"),
                      pd.read_parquet(CACHE / "supp_product_image_urls.parquet")])
    urls = urls.dropna(subset=["image_url"])
    urls = urls[urls.image_url.str.startswith("http") & (urls.image_url != PLACEHOLDER)]
    url_of = dict(zip(urls.product_id, urls.image_url))
    ex = ex[ex.product_id.isin(url_of)]
    return ex, url_of


def fetch_all(candidates, url_of):
    """One image per product_id, cached on disk so re-runs do not refetch."""
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [p for p in candidates if not (RAW / f"{p}.jpg").exists()
            and not (RAW / f"{p}.dead").exists()]
    lock = threading.Lock()
    done = [0]
    session = threading.local()

    def get(pid):
        s = getattr(session, "s", None)
        if s is None:
            s = session.s = requests.Session()
            s.headers["User-Agent"] = "Mozilla/5.0 (decision-v1 dataset build)"
        try:
            r = s.get(url_of[pid], timeout=30)
            if r.status_code != 200 or len(r.content) < 1200:
                raise IOError(str(r.status_code))
            with Image.open(io.BytesIO(r.content)) as im:
                im.load()
            (RAW / f"{pid}.jpg").write_bytes(r.content)
        except Exception:
            (RAW / f"{pid}.dead").write_bytes(b"")
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  fetched {done[0]}/{len(todo)}", flush=True)

    if todo:
        print(f"fetching {len(todo)} product images", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(get, todo))
    return {p for p in candidates if (RAW / f"{p}.jpg").exists()}


def store(pid):
    """Re-encode to a content-addressed RGB JPEG under the 1 MP cap."""
    try:
        with Image.open(RAW / f"{pid}.jpg") as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.width * im.height > 1_000_000:
                s = (1_000_000 / (im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                               Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            data = buf.getvalue()
            w, h = im.size
    except Exception:
        return None
    sha = hashlib.sha256(data).hexdigest()
    dest = IMG_DIR / f"{sha}.jpg"
    if not dest.exists():
        dest.write_bytes(data)
    return {"image": str(dest), "sha256": sha, "width": w, "height": h}


def main():
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ex, url_of = load_pairs()
    stats = Counter(judgments=len(ex), products_with_url=len(url_of))

    # one judgment per product, sampled so the four ESCI levels stay balanced
    by_product = defaultdict(list)
    for q, pid, lab in zip(ex["query"], ex.product_id, ex.esci_label):
        by_product[pid].append((q, lab))
    by_label = defaultdict(list)
    for pid, js in by_product.items():
        q, lab = js[int(hashlib.sha256(f"{SEED}:{pid}".encode()).hexdigest()[:8], 16) % len(js)]
        by_label[lab].append((pid, q))
    for lab in by_label:
        by_label[lab].sort()
        rng.shuffle(by_label[lab])
        stats[f"products_label_{lab}"] = len(by_label[lab])

    candidates, label_of, query_of = [], {}, {}
    for lab, items in by_label.items():
        for pid, q in items[:CANDIDATES_PER_LEVEL]:
            candidates.append(pid)
            label_of[pid] = lab
            query_of[pid] = q
    alive = fetch_all(candidates, url_of)
    stats["fetch_attempted"] = len(candidates)
    stats["fetch_ok"] = len(alive)
    stats["fetch_dead"] = len(candidates) - len(alive)

    keep = []
    per = Counter()
    for pid in candidates:
        if pid not in alive:
            continue
        lab = label_of[pid]
        if per[lab] >= TARGET_PER_LEVEL:
            continue
        per[lab] += 1
        keep.append(pid)

    records = []
    test_used = 0
    for pid in sorted(keep):
        gold = LABELS[label_of[pid]]
        query = str(query_of[pid]).strip()
        if not query or len(query) > 300:
            continue
        window, cause = pick_window(gold, rng)
        field = {"id": "answer", "type": "ordinal",
                 "question": rng.choice(TEMPLATES).format(q=query),
                 "levels": levels_for(window)}
        request = {"request_id": f"esci-{pid}"[:128], "state": {}, "fields": [field]}
        Request.model_validate(request)
        h = int(hashlib.sha256(f"{SEED}:part:{pid}".encode()).hexdigest()[:8], 16) % 1000
        if h < TEST_GROUP_PERMILLE and test_used < TEST_CAP:
            partition = "test"
            test_used += 1
        else:
            partition = "dev" if rng.random() < DEV_FRACTION else "train"
        info = store(pid)
        if info is None:
            continue
        records.append({"id": f"{SOURCE}:{pid}", "source": SOURCE,
                        "source_split": "esci_test_small_us", "source_group": str(pid),
                        "family": "query_product_relevance", "license": LICENSE,
                        "images": [info], "request": request,
                        "target": None if cause else gold, "abstention_cause": cause,
                        "source_answer": label_of[pid], "partition": partition})

    records.sort(key=lambda r: r["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary = {"records": len(records), "seed": SEED,
               "partitions": dict(Counter(r["partition"] for r in records)),
               "families": dict(Counter(r["family"] for r in records)),
               "esci_labels": dict(Counter(r["source_answer"] for r in records)),
               "targets": dict(Counter(str(r["target"]) for r in records)),
               "abstention": dict(Counter(str(r["abstention_cause"]) for r in records)),
               "abstention_share": round(sum(r["target"] is None for r in records) / max(1, len(records)), 4),
               "level_counts": dict(sorted(Counter(len(r["request"]["fields"][0]["levels"])
                                                   for r in records).items())),
               "stats": dict(stats)}
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
