"""Acquire English Wikipedia lead paragraphs for decision-v2 (TEXT source).

Upstream: `wikimedia/wikipedia`, config `20231101.en` (the 1 November 2023 dump, prepared and
published by the Wikimedia Foundation itself).  Only a spread of parquet row groups is read,
via HTTP range requests, so this costs ~100 MB rather than the ~20 GB of the full config.

Topic spread comes from sampling row groups evenly across several shards: the shards are ordered
by page id, so an even sweep covers articles created across the whole life of the project rather
than one alphabetical or topical block.

Output
  data/decision-v2-raw/wikipedia_paragraphs/passages.jsonl
  data/decision-v2-raw/wikipedia_paragraphs/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "decision-v2-raw" / "wikipedia_paragraphs"
REPO = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
CONFIG = "20231101.en"
DUMP_DATE = "2023-11-01"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{CONFIG}/"
SHARDS = [0, 6, 13, 20, 27, 34, 40]      # even sweep over the 41 shards
GROUPS_PER_SHARD = 9                      # 1000 articles each, spread inside the shard
TARGET = 30000

SKIP_TITLE = re.compile(r"(?i)^(list of|index of|timeline of|glossary of|outline of)\b|\(disambiguation\)$")
BAD_LEAD = re.compile(r"(?i)^(redirect|see also)\b")


def lead_paragraph(text: str) -> str | None:
    """The article's opening prose: the first paragraph, extended until it carries some substance."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) < 2:
        return None
    lead = parts[0]
    for extra in parts[1:3]:
        if len(lead) >= 350:
            break
        if "\n" in extra or len(extra) < 40:      # a section heading block, not prose
            break
        lead = lead + " " + extra
    if "\n" in lead:
        return None
    return lead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("https")

    kept: list[dict] = []
    read_groups: dict[str, list[int]] = {}
    for shard in SHARDS:
        name = f"train-{shard:05d}-of-00041.parquet"
        with fs.open(BASE + name) as f:
            pf = pq.ParquetFile(f)
            total = pf.metadata.num_row_groups
            step = max(1, total // GROUPS_PER_SHARD)
            groups = [min(total - 1, g * step) for g in range(GROUPS_PER_SHARD)]
            read_groups[name] = groups
            for g in groups:
                for row in pf.read_row_group(g, columns=["id", "url", "title", "text"]).to_pylist():
                    title = (row["title"] or "").strip()
                    if not title or SKIP_TITLE.search(title):
                        continue
                    lead = lead_paragraph(row["text"] or "")
                    if lead is None or BAD_LEAD.match(lead):
                        continue
                    if not (280 <= len(lead) <= 1800):
                        continue
                    kept.append({
                        "id": f"page{row['id']}",
                        "page_id": str(row["id"]),
                        "title": title,
                        "url": row["url"],
                        "lead": lead,
                    })
        print(f"  {name}: {len(groups)} groups, total kept {len(kept)}", file=sys.stderr)
        if len(kept) >= args.target:
            break

    # Deterministic thinning to the quota, keeping the sweep across shards balanced.
    kept.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    kept = kept[: args.target]
    kept.sort(key=lambda r: int(r["page_id"]))

    out = RAW / "passages.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in kept))
    manifest = {
        "source": "wikipedia_paragraphs",
        "upstream_repo": f"https://huggingface.co/datasets/{REPO}",
        "upstream_revision": REVISION,
        "config": CONFIG,
        "dump_date": DUMP_DATE,
        "read_row_groups": read_groups,
        "urls": [BASE + n for n in sorted(read_groups)],
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
    print(json.dumps({"passages": len(kept)}, indent=2))


if __name__ == "__main__":
    main()
