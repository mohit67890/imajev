"""Acquire Stack Exchange question + accepted-answer passages for decision-v2 (TEXT source).

Upstream: `donfu/oa-stackexchange` on the Hugging Face Hub, a redistribution of the official
Stack Exchange data dump (https://archive.org/details/stackexchange) restricted to threads that
have an accepted answer.  Only the parquet row groups that hold the sites we want are fetched,
using HTTP range reads against the Hub, so this pulls tens of megabytes rather than the 3.7 GB
of the full dataset.

Output
  data/decision-v2-raw/stackexchange/passages.jsonl   one question+accepted answer per line
  data/decision-v2-raw/stackexchange/manifest.json    URLs, revision, row groups read, sha256

Nothing here runs a model, and nothing outside data/decision-v2-raw/stackexchange is written.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "decision-v2-raw" / "stackexchange"
REPO = "donfu/oa-stackexchange"
REVISION = "ae54b00b7c688866e86acf4605034ed2bded75f3"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/"

# 32 deliberately non-programming, topically diverse sites.  The twelve named in the brief
# (cooking, diy, travel, personalfinance=money, law, gardening, fitness, workplace, parenting,
# pets, photo, security) are all here; the rest widen the topic spread.
SITES = [
    "cooking", "diy", "travel", "money", "law", "gardening", "fitness", "workplace",
    "parenting", "pets", "photo", "security", "outdoors", "bicycles", "music", "movies",
    "health", "interpersonal", "woodworking", "boardgames", "academia", "writers",
    "mechanics", "ux", "graphicdesign", "homebrew", "sustainability", "vegetarianism",
    "martialarts", "lifehacks", "expatriates", "sports",
]
PER_SITE = 2000           # cap; small sites simply contribute everything they have
TOTAL_TARGET = 40000

TRUNCATED = re.compile(r"[<\[]\s*$")


def shard_names(fs) -> list[str]:
    import urllib.request

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as r:
        meta = json.load(r)
    names = sorted(
        s["rfilename"].split("/")[-1] for s in meta["siblings"] if s["rfilename"].endswith(".parquet")
    )
    if not names:
        raise RuntimeError("no parquet shards listed for " + REPO)
    return names


def plan(fs, names: list[str]) -> dict[str, list[int]]:
    """Row groups worth reading, from the parquet footer statistics only (one range read each).

    The shards are sorted by SOURCE, so a row group's (min, max) statistics say exactly which
    sites can appear inside it.  A group is read when either bound names a wanted site.
    """
    wanted = {f"stackexchange-{s}" for s in SITES}
    chosen: dict[str, list[int]] = {}
    for name in names:
        with fs.open(BASE + name) as f:
            md = pq.ParquetFile(f).metadata
            col = md.schema.names.index("SOURCE")
            keep = []
            for g in range(md.num_row_groups):
                st = md.row_group(g).column(col).statistics
                if st.min in wanted or st.max in wanted:
                    keep.append(g)
        if keep:
            chosen[name] = keep
        print(f"  {name}: {len(keep)}/{md.num_row_groups} row groups", file=sys.stderr)
    return chosen


def clean(row: dict) -> dict | None:
    instruction = (row.get("INSTRUCTION") or "").strip()
    answer = (row.get("RESPONSE") or "").strip()
    if "\n" not in instruction:
        return None
    title, body = instruction.split("\n", 1)
    title, body = title.strip(), body.strip()
    if not (15 <= len(title) <= 200) or not (80 <= len(body) <= 3500):
        return None
    if not (40 <= len(answer) <= 3500):
        return None
    if TRUNCATED.search(body) or TRUNCATED.search(answer):
        return None          # the upstream HTML stripper truncates a few posts mid-link
    meta = row.get("METADATA") or {}
    tags = [t.strip() for t in str(meta.get("tags") or "").split(",") if t.strip()]
    site = str(row["SOURCE"]).replace("stackexchange-", "")
    ident = hashlib.sha256(f"{site}\0{title}\0{body[:400]}".encode()).hexdigest()[:20]
    return {
        "id": f"{site}-{ident}",
        "site": site,
        "title": title,
        "body": body,
        "answer": answer,
        "tags": tags,
        "question_score": int(meta.get("question_score") or 0),
        "answer_score": int(meta.get("answer_score") or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-site", type=int, default=PER_SITE)
    ap.add_argument("--target", type=int, default=TOTAL_TARGET)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("https")
    names = shard_names(fs)
    print(f"scanning {len(names)} shard footers", file=sys.stderr)
    chosen = plan(fs, names)

    kept: list[dict] = []
    counts: collections.Counter = collections.Counter()
    seen: set[str] = set()
    read_groups = {}
    for name, groups in sorted(chosen.items()):
        if len(kept) >= args.target and all(counts[s] >= args.per_site for s in SITES if counts[s]):
            break
        used = []
        with fs.open(BASE + name) as f:
            pf = pq.ParquetFile(f)
            for g in groups:
                # skip a group whose only possible sites are already full
                st = pf.metadata.row_group(g).column(pf.metadata.schema.names.index("SOURCE")).statistics
                bounds = {st.min.replace("stackexchange-", ""), st.max.replace("stackexchange-", "")}
                if bounds and all(counts[b] >= args.per_site for b in bounds if b in SITES) and not (bounds - set(SITES)):
                    continue
                used.append(g)
                for row in pf.read_row_group(g).to_pylist():
                    site = str(row["SOURCE"]).replace("stackexchange-", "")
                    if site not in SITES or counts[site] >= args.per_site:
                        continue
                    item = clean(row)
                    if item is None or item["id"] in seen:
                        continue
                    seen.add(item["id"])
                    kept.append(item)
                    counts[site] += 1
        read_groups[name] = used
        print(f"  read {name}: {len(used)} groups, total kept {len(kept)}", file=sys.stderr)

    kept.sort(key=lambda r: r["id"])
    out = RAW / "passages.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in kept))
    manifest = {
        "source": "stackexchange",
        "upstream_repo": f"https://huggingface.co/datasets/{REPO}",
        "upstream_revision": REVISION,
        "upstream_of_upstream": "https://archive.org/details/stackexchange (official Stack Exchange data dump)",
        "read_row_groups": read_groups,
        "urls": [BASE + n for n in sorted(read_groups)],
        "sites": dict(sorted(counts.items())),
        "passages": len(kept),
        "output": {
            "path": str(out.relative_to(ROOT)),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "bytes": out.stat().st_size,
        },
        "note": "Row groups are fetched with HTTP range reads; whole-shard sha256 values are not "
                "computed because whole shards are never downloaded.",
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passages": len(kept), "sites": len(counts)}, indent=2))


if __name__ == "__main__":
    main()
