"""Write the `pd12m` licence evidence receipts.

Separate from `scripts/v2/write_license_receipts.py` only so the two v2 collection efforts do not
edit one file at the same time; the receipt shape, the `flatten` helper, the reviewer and the
review date are imported from it, so a pd12m receipt is byte-for-byte the same kind of object as a
commons_photos or openimages_v2 one.

The licence position this establishes, in short:

  * IMAGES   PD12M's per-row `license` column is a MIX of CC0-1.0 and Public Domain Mark 1.0.
             Only `https://creativecommons.org/publicdomain/zero/1.0/` is admitted
             (`scripts/v2/fetch_pd12m.py:KEEP_LICENSE`), because CC0-1.0 is in
             `scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST` and PDM-1.0 is not an SPDX id in it.
             The receipt below is the operative one: it is the licence object every record carries.
  * METADATA The captions, dimensions and licence column are the *dataset*, which Spawning licenses
             under CDLA-Permissive-2.0 -- NOT an allowlisted SPDX id.  No record carries it,
             because no caption is ever written into a record: captions only pick the template and
             seed its option set, and the stored copy in `attribution.jsonl` is provenance.  It is
             recorded here as supporting evidence, with its grant quoted, and the source README
             says so in as many words.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/write_license_receipts_pd12m.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.write_license_receipts import CC0_GRANT, L, REVIEWED_AT, REVIEWER, flatten  # noqa: E402

D = L / "pd12m"
CARD = "https://huggingface.co/datasets/Spawning/PD12M"

CC0_SCOPE = ("the PD12M images admitted to this source: rows whose per-row metadata `license` "
             "field is exactly https://creativecommons.org/publicdomain/zero/1.0/ . Rows marked "
             "Public Domain Mark 1.0, or with no licence value, are not downloaded.")
CC0_NOTES = (
    "PD12M's image licence is a MIX. Its own dataset card says 'PD12M consists of entirely public "
    "domain and CC0 licensed images', and its Datasheet says 'In all cases, we only collected "
    "images with metadata indicating a Public Domain Mark or CC0 license.' -- i.e. two different "
    "statuses. decision-v2 therefore filters on the per-row `license` column and keeps ONLY "
    "CC0-1.0 (scripts/v2/fetch_pd12m.py:KEEP_LICENSE); the census of the column over every "
    "metadata shard actually read is in pd12m-license-census.json. Spawning disclaims a "
    "guarantee: 'the sources of the original images, and by extension, Spawning, cannot fully "
    "guarantee that no copyrighted material appears in the dataset.' The dataset (captions and "
    "metadata) is separately licensed CDLA-Permissive-2.0, which is NOT in the commercial "
    "allowlist; no caption is written into any record, so no record carries that licence. See "
    "data/decision-v2/pd12m/README.md."
)

RECEIPTS = [
    ("CC0-1.0.deed.html", "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/",
     CC0_GRANT, CC0_SCOPE, CC0_NOTES),
]

PROVENANCE = [
    ("CC0-1.0.legalcode.txt", "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
     "Full CC0 1.0 Universal legal code, the operative text behind the deed.",
     "Affirmer hereby overtly, fully, permanently, irrevocably and unconditionally waives, "
     "abandons, and surrenders all of Affirmer's Copyright and Related Rights"),
    ("pd12m-hf-dataset-card.md", CARD + "/raw/main/README.md",
     "The Hugging Face dataset card for Spawning/PD12M: the primary statement of what the images "
     "are and what the dataset is licensed under. Quoted verbatim for the IMAGES: 'PD12M consists "
     "of entirely public domain and CC0 licensed images, with automated recaptioning of image "
     "data, and quality and safety filtering.' And for the METADATA/captions: 'The dataset is "
     "licensed under the [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/).' The card also "
     "documents the per-row field this source filters on: '`license`: The URL of the image "
     "license.'",
     "PD12M consists of entirely public domain and CC0 licensed images, with automated "
     "recaptioning of image data, and quality and safety filtering."),
    ("pd12m-hf-dataset-card.md.metadata-licence-quote.json", CARD + "/raw/main/README.md",
     "Second quote from the same dataset card, recorded separately because it is about the "
     "METADATA rather than the images.", None),
    ("pd12m-datasheet.txt", CARD + "/blob/main/Datasheet.pdf",
     "Text extracted from Spawning's PD12M Datasheet (the PDF itself is stored alongside as "
     "pd12m-datasheet.pdf). This is the primary-source statement that the image licences are a "
     "MIX, which is why this source filters per row.",
     "In all cases, we only collected images with metadata indicating a Public Domain Mark or "
     "CC0 license."),
    ("pd12m-datasheet.pdf", CARD + "/resolve/main/Datasheet.pdf",
     "Spawning's PD12M Datasheet as published (binary; see pd12m-datasheet.txt for the text the "
     "quotes were checked against).", None),
    ("cdla-permissive-2.0.html", "https://cdla.dev/permissive-2-0/",
     "Community Data License Agreement - Permissive 2.0, the licence Spawning puts on the PD12M "
     "dataset (captions and metadata). Recorded as supporting evidence only: CDLA-Permissive-2.0 "
     "is NOT in scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST and no decision-v2 record carries "
     "it, because captions are never written into a record.",
     "A Data Recipient may use, modify, and share the Data made available by Data Provider(s) "
     "under this agreement if that Data Recipient follows the terms of this agreement."),
    ("pd12m-arxiv-abstract.html", "https://arxiv.org/abs/2410.23144",
     "The PD12M paper (Meyer, Padgett, Miller, Exline, 2024), abstract page. Quoted: 'a dataset "
     "of 12.4 million high-quality public domain and CC0-licensed images with synthetic "
     "captions'.",
     "public domain and CC0-licensed images with synthetic captions"),
    ("pd12m-tutorial-images.md", CARD + "/raw/main/tutorials/images.md",
     "Spawning's own instructions for fetching the image files, i.e. the sanctioned access path "
     "this collector uses (the pd12m S3 bucket, one GET per image).", None),
    ("pd12m-tutorial-metadata.md", CARD + "/raw/main/tutorials/metadata.md",
     "Spawning's own instructions for reading the metadata parquet shards.", None),
    ("pd12m-license-census.json", CARD + "/tree/main/metadata",
     "Our own census of the per-row `license` column over every metadata shard actually read, "
     "written by scripts/v2/fetch_pd12m.py. This is the evidence for the CC0-only filter.", None),
]

NORM = lambda s: re.sub(r"[‐-―−]", "-", re.sub(r"[‘’]", "'", s))


def main() -> int:
    problems = []
    D.mkdir(parents=True, exist_ok=True)

    # the second dataset-card quote, about the metadata, stored as its own small evidence file
    meta_quote = ("The dataset is licensed under the "
                  "[CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/).")
    card = D / "pd12m-hf-dataset-card.md"
    (D / "pd12m-hf-dataset-card.md.metadata-licence-quote.json").write_text(json.dumps({
        "quote": meta_quote,
        "about": "the PD12M dataset: captions, dimensions and the licence column",
        "spdx_like": "CDLA-Permissive-2.0",
        "in_commercial_allowlist": False,
        "quoted_from": "pd12m-hf-dataset-card.md",
        "quoted_from_sha256": hashlib.sha256(card.read_bytes()).hexdigest() if card.is_file() else None,
        "source_url": CARD + "/raw/main/README.md",
        "found_in_stored_card": meta_quote in card.read_text() if card.is_file() else None,
        "note": "Recorded because it differs from the IMAGE licence. No decision-v2 record carries "
                "this licence: captions are used only to choose and shape questions and are kept "
                "in attribution.jsonl as provenance, never in a record's state or target.",
    }, indent=2) + "\n")

    for rel, spdx, url, quote, scope, notes in RECEIPTS:
        path = D / rel
        if not path.is_file():
            problems.append(f"missing evidence {rel}")
            continue
        verified = NORM(quote) in NORM(flatten(path))
        if not verified:
            problems.append(f"quoted grant not found in {rel}")
        path.with_name(path.name + ".receipt.json").write_text(json.dumps({
            "spdx": spdx,
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "commercial_use_reviewed": bool(verified),
            "source_url": url,
            "reviewed_at": REVIEWED_AT,
            "reviewed_by": REVIEWER,
            "scope": scope,
            "quoted_grant": quote,
            "notes": notes,
        }, indent=2) + "\n")
        print(f"receipt {rel} spdx={spdx} commercial_use_reviewed={verified}")

    for rel, url, note, quote in PROVENANCE:
        path = D / rel
        if not path.is_file():
            problems.append(f"missing provenance {rel}")
            continue
        found = None
        if quote is not None:
            found = NORM(quote) in NORM(flatten(path))
            if not found:
                problems.append(f"quote not found in {rel}")
        payload = {
            "source_url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "fetched_at": REVIEWED_AT,
            "fetched_by": REVIEWER,
            "note": note,
            "kind": "supporting evidence, not a licence receipt",
        }
        if quote is not None:
            payload["quote"] = quote
            payload["quote_found_in_stored_file"] = found
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"provenance {rel}" + ("" if quote is None else f" quote_found={found}"))

    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
        return 1
    print("all pd12m receipts written and their grants verified against the stored files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
