"""Marqo-GS-10M -> decision-v1 product category choice records.

The research doc expects "~2.4k leaf categories"; the published parquet has no
category column at all. Its columns are `query, product_id, position, title,
pair_id, score_linear, score_reciprocal, no_score, query_id, image`, and the
"categories" are the **search queries** — long-tail natural-language phrases
("Breathable full slips for summer", "Fashion Forward Trendsetting Wrap Designs",
"Climbing sale") that no single photograph determines. Using a query as a gold
category label would be manufacturing an answer the source does not support.

What the source does support is the **title**, which states what the product is:
"Ugg Women's Shearling Earmuffs - Black". `scripts/v1/marqo_gs_taxonomy.py` holds a
curated phrase lexicon; the longest phrase wins, overlapping matches are suppressed,
and a title is used only when **exactly one** canonical category survives. A title
naming two things ("Pajama Set with Slippers") is dropped rather than resolved.

Download is budgeted: every 7th `in_domain` shard, so the sample spans the whole
index range (the shards are ordered by query id, so consecutive shards share
queries) without pulling the 84.9 GB the split actually weighs.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_marqo_gs.py
"""
import hashlib, io, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from vision_decision.contracts import Request
import marqo_gs_taxonomy as T

SOURCE = "marqo_gs"
SEED = 20260922
CACHE = Path(".cache/datasets/v1/marqo_gs")
OUT = Path("data/decision-v1/marqo_gs")
IMG_DIR = OUT / "images"
LICENSE = "Apache-2.0 (dataset card); images scraped from Google Shopping listings, upstream rights unverified"
QUOTA = 40_000
MAX_PER_LABEL = 900        # 359 labels; keeps 'boots'/'jacket' from swamping the set
NOT_LISTED_RATE = 0.18
DEV_FRACTION = 0.03
TEST_GROUP_PERMILLE = 20
TEST_CAP = 300             # one family
BIG_OPTION_RATE = 0.10

TEMPLATES = [
    "What kind of product is shown in this photo?",
    "Which of these best describes the item in the picture?",
    "This is a shopping listing photo. What is the product?",
    "Pick the category that fits the object shown.",
    "What is being sold in this image?",
    "Identify the type of product photographed here.",
    "Looking at the picture, which product category is this?",
]
MAXN = max(len(p.split()) for p in T.LEXICON)


def categorise(title):
    """The one curated category this title names, or None."""
    words = re.findall(r"[a-z]+", (title or "").lower())
    spans = []
    for n in range(MAXN, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if phrase in T.LEXICON:
                spans.append((i, i + n, T.LEXICON[phrase]))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    taken, occupied = [], set()
    for a, b, value in spans:
        if occupied & set(range(a, b)):
            continue
        occupied |= set(range(a, b))
        taken.append(value)
    if len({v[0] for v in taken}) != 1:
        return None
    return taken[0]


def token_subset(a, b):
    x, y = set(a.split()), set(b.split())
    return x <= y or y <= x


def confusable_with(value):
    out = set()
    for g in T.CONFUSABLE:
        if value in g:
            out |= g
    return out


def make_options(gold, near, far, rng, cause):
    banned = confusable_with(gold)

    def ok(v):
        return v != gold and v not in banned and not token_subset(v, gold)

    total = rng.randint(13, 25) if rng.random() < BIG_OPTION_RATE else rng.randint(2, 12)
    pool = [v for v in near if ok(v)]
    rng.shuffle(pool)
    rest = [v for v in far if ok(v) and v not in set(pool)]
    rng.shuffle(rest)
    pool += rest
    want = total if cause == "not_listed" else total - 1
    chosen = pool[:want]
    if len(chosen) < (2 if cause == "not_listed" else 1):
        return None
    values = chosen if cause == "not_listed" else chosen + [gold]
    if len(values) < 2:
        return None
    rng.shuffle(values)
    return values


def main():
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("in_domain-*.parquet"),
                    key=lambda p: int(p.stem.split("-")[1]))
    if not shards:
        raise SystemExit("no marqo_gs shards under .cache/datasets/v1/marqo_gs")
    stats = Counter(shards=len(shards),
                    download_bytes=sum(p.stat().st_size for p in shards))

    items, seen_product = [], set()
    for s in shards:
        table = pq.read_table(s, columns=["title", "product_id", "query"])
        titles = table.column("title").to_pylist()
        pids = table.column("product_id").to_pylist()
        queries = table.column("query").to_pylist()
        for row, (title, pid, query) in enumerate(zip(titles, pids, queries)):
            stats["rows"] += 1
            if pid in seen_product:      # one question per product, and per image
                continue
            cat = categorise(title)
            if cat is None:
                continue
            seen_product.add(pid)
            items.append({"shard": s.name, "row": row, "pid": str(pid),
                          "label": cat[0], "group": cat[1], "query": query})
    stats["products"] = len(seen_product)
    stats["categorised"] = len(items)
    rng.shuffle(items)

    labels_by_group = defaultdict(set)
    for it in items:
        labels_by_group[it["group"]].add(it["label"])
    all_labels = sorted({it["label"] for it in items})
    stats["distinct_labels"] = len(all_labels)

    records, per_label = [], Counter()
    test_used = 0
    for it in items:
        if len(records) >= QUOTA:
            break
        if per_label[it["label"]] >= MAX_PER_LABEL:
            continue
        gold = it["label"]
        near = sorted(labels_by_group[it["group"]] - {gold})
        far = [l for l in all_labels if l != gold and l not in labels_by_group[it["group"]]]
        cause = "not_listed" if rng.random() < NOT_LISTED_RATE else None
        values = make_options(gold, near, far, rng, cause)
        if values is None:
            continue
        field = {"id": "answer", "type": "choice", "question": rng.choice(TEMPLATES),
                 "options": [{"value": v} for v in values]}
        request = {"request_id": f"marqogs-{it['pid']}"[:128], "state": {}, "fields": [field]}
        Request.model_validate(request)
        h = int(hashlib.sha256(f"{SEED}:{it['pid']}".encode()).hexdigest()[:8], 16) % 1000
        if h < TEST_GROUP_PERMILLE and test_used < TEST_CAP:
            partition = "test"
            test_used += 1
        else:
            partition = "dev" if rng.random() < DEV_FRACTION else "train"
        per_label[gold] += 1
        records.append({"id": f"{SOURCE}:{it['pid']}", "source": SOURCE,
                        "source_split": "in_domain", "source_group": it["pid"],
                        "family": "product_category", "license": LICENSE,
                        "images": [(it["shard"], it["row"])], "request": request,
                        "target": None if cause else gold, "abstention_cause": cause,
                        "source_answer": gold, "partition": partition})

    written = write_images({r["images"][0] for r in records})
    final = []
    for r in records:
        info = written.get(r["images"][0])
        if info is None:
            stats["dropped_missing_image"] += 1
            continue
        r["images"] = [info]
        final.append(r)
    final.sort(key=lambda r: r["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in final))
    summary = {"records": len(final), "seed": SEED,
               "partitions": dict(Counter(r["partition"] for r in final)),
               "families": dict(Counter(r["family"] for r in final)),
               "abstention": dict(Counter(str(r["abstention_cause"]) for r in final)),
               "abstention_share": round(sum(r["target"] is None for r in final) / max(1, len(final)), 4),
               "field_types": dict(Counter(r["request"]["fields"][0]["type"] for r in final)),
               "option_counts": dict(sorted(Counter(len(r["request"]["fields"][0]["options"])
                                                    for r in final).items())),
               "distinct_gold": len({r["source_answer"] for r in final}),
               "top_labels": Counter(r["source_answer"] for r in final).most_common(15),
               "images": len(written), "stats": dict(stats)}
    (OUT / "conversion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def write_images(wanted):
    by_shard = defaultdict(set)
    for shard, row in wanted:
        by_shard[shard].add(row)
    written = {}
    for shard in sorted(by_shard):
        want_rows = by_shard[shard]
        pf = pq.ParquetFile(CACHE / shard)
        offset = 0
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            hits = [r for r in range(offset, offset + n) if r in want_rows]
            if hits:
                col = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
                for r in hits:
                    cell = col[r - offset]
                    blob = cell["bytes"] if cell else None
                    if not blob:
                        continue
                    try:
                        with Image.open(io.BytesIO(blob)) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            if im.width * im.height > 1_000_000:
                                s = (1_000_000 / (im.width * im.height)) ** 0.5
                                im = im.resize((max(1, int(im.width * s)),
                                                max(1, int(im.height * s))), Image.LANCZOS)
                            buf = io.BytesIO()
                            im.save(buf, "JPEG", quality=90)
                            data = buf.getvalue()
                            w, h = im.size
                    except Exception:
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    dest = IMG_DIR / f"{sha}.jpg"
                    if not dest.exists():
                        dest.write_bytes(data)
                    written[(shard, r)] = {"image": str(dest), "sha256": sha,
                                           "width": w, "height": h}
            offset += n
    return written


if __name__ == "__main__":
    main()
