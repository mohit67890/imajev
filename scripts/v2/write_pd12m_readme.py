"""Write `data/decision-v2/pd12m/README.md` from the artefacts, so no number is typed by hand.

A sibling of `scripts/v2/write_source_readme.py` -- it imports that file's `table`,
`licence_block` and `common_sections` so the two READMEs stay the same shape -- kept separate only
so the two v2 collection efforts never edit one file at the same time.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/write_pd12m_readme.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2 import pd12m_captions as CAP  # noqa: E402
from v2.common import OUT, RAW, read_jsonl  # noqa: E402
from v2.write_source_readme import common_sections, licence_block, table  # noqa: E402

SOURCE = "pd12m"


def build() -> str:
    out = OUT / SOURCE
    summary = json.loads((out / "conversion_summary.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    stats = json.loads((RAW / SOURCE / "selection_stats.json").read_text())
    manifest = json.loads((RAW / SOURCE / "manifest.json").read_text())
    census = json.loads((OUT / "licenses" / SOURCE / "pd12m-license-census.json").read_text())
    attribution = read_jsonl(out / "attribution.jsonl")
    overlap = out / "cross_source_overlap.json"
    overlap_note = ""
    if overlap.is_file():
        o = json.loads(overlap.read_text())
        overlap_note = (
            f"\n* **Overlap with the other v2 image sources is only partly ruled out.** "
            f"{stats['top_contributors'].get('Wikimedia Commons', 0):,} of the selected photos "
            f"name Wikimedia Commons as their contributing institution, and `commons_photos` "
            f"crawled Commons directly. Every v2 image source re-encodes through the same "
            f"deterministic `scripts/v1/_common.py:store_image`, so an identical ORIGINAL lands on "
            f"an identical sha256; comparing the image stores "
            f"(`scripts/v2/pd12m_overlap_check.py`, `cross_source_overlap.json`) found "
            f"**{o['shared_sha256']} shared sha256** between this source's {o['pd12m_images']:,} "
            f"images and the {o['other_images']:,} images of "
            f"{', '.join(f'`{k}`' for k in o['per_source'])}. That detects byte-identical "
            f"originals only; the same photograph re-encoded differently upstream would not be "
            f"caught.")

    v1path = out / "v1_validator_report.json"
    v1report = (
        f"**{json.loads(v1path.read_text())['errors']:,} errors for "
        f"{json.loads(v1path.read_text())['records']:,} records**, every saved one of the single "
        f"kind `target None <=> abstention_cause set`."
        if v1path.is_file() else "(not run)")

    families, partitions, small, large = common_sections(SOURCE, summary, validation)
    stored_mb = sum(p.stat().st_size for p in (out / "images").glob("*.jpg")) / 1e6
    state_kinds = table(sorted(summary["state_kinds"].items(), key=lambda kv: -kv[1]),
                        ["state", "records"])
    licences = table(sorted(summary["licenses"].items()), ["SPDX", "records"])

    stored_buckets = Counter(manifest["buckets"])
    bucket_rows = [(b, n, f"{n / max(1, manifest['stored']):.1%}")
                   for b, n in sorted(stored_buckets.items(), key=lambda kv: (-kv[1], kv[0]))]
    bucket_table = table(bucket_rows, ["subject bucket", "photos", "share"])
    contributors = table(list(stats["top_contributors"].items())[:12],
                         ["contributing institution (PD12M `source`)", "selected photos"])
    dropped = table([(k, f"{v:,}") for k, v in stats["dropped"].items()],
                    ["filter", "metadata rows dropped"])
    rejects = table([(k, f"{v:,}") for k, v in stats["caption_reject_reasons"].items()],
                    ["caption reject reason", "rows"])
    census_table = table([("(no licence value)" if k in ("None", "null", "") else k,
                           f"{v:,}", f"{v / max(1, census['rows_read']):.1%}")
                          for k, v in census["values"].items()],
                         ["per-row `license` value", "rows", "share"])
    no_licence = sum(v for k, v in census["values"].items() if k in ("None", "null", ""))
    download = manifest["download"]

    head = f"""# pd12m — PD12M (Spawning), CC0 rows only (decision-v2 candidates)

Photographs from **PD12M** (Spawning's 12.4M public-domain / CC0 image–caption dataset,
`Spawning/PD12M` on Hugging Face), collected on 2026-09-23 and turned into **unanswered** decision
candidates: `target: null`, `abstention_cause: null`, `pseudo_label: "pending"`. The 9B teacher
labels them later; nothing here claims an answer.

PD12M is the bulk image source the revised spec asks for
(`docs/decision-v2-pseudolabel-spec.md`, "Image sources, revised 23 Sept"): a clean licence by
construction and a caption per image that lets the templates build present **and absent**
questions, which a bare category crawl cannot.

## The caption rule

Every question here is shaped by PD12M's synthetic caption, and **no record contains one**. The
caption picks the subject bucket, seeds the option set, and decides which objects an
absent-object question may name. It is then dropped: `scripts/v2/convert_photos.py` passes
`title=""` for this source precisely so that `photo.make_state` cannot put it into the `aligned`
state, and `tests/test_v2_pd12m.py:test_no_caption_ever_reaches_a_record` fails the build if any
25-character clause of a caption turns up anywhere in `records.jsonl`. The teacher must read the
pixels. The captions are kept in `attribution.jsonl` as provenance only.

## What was collected

{manifest['stored']:,} distinct photographs ({stored_mb:,.0f} MB stored, re-encoded to RGB JPEG q90
under the 1 MP cap), fetched one GET at a time from Spawning's own image host
`https://pd12m.s3.us-west-2.amazonaws.com/images/` — the access path PD12M's own tutorial
documents, and the reason the dataset exists in that form ("Images in PD12M are also hosted on
dedicated cloud storage, separate from the original image hosts, to avoid placing an undue burden
on those hosts"). {download['ok'] or 0:,} downloads succeeded and {download['fail'] or 0:,} failed in
{download['wall_clock_minutes']} minutes of wall clock, inside the brief's
{download['deadline_minutes']:.0f}-minute cap, so {download['skipped_past_deadline'] or 0:,} selected
photos were left unfetched when the clock ran out. Content-addressing then collapsed those
downloads to **{manifest['stored']:,} distinct images**
({manifest['manifest_rows'] - manifest['stored']:,} PD12M ids turned out to be byte-identical to
another). {manifest['stored_bytes_original'] / 1e9:.1f} GB was fetched over the wire and
{stored_mb / 1000:.1f} GB kept after the 1 MP re-encode.

Per-photo attribution — PD12M id, the image URL, the caption, the licence and the contributing
institution — is in `attribution.jsonl` ({len(attribution):,} rows). The raw manifest with URL +
sha256 per photo is `data/decision-v2-raw/pd12m/manifest.json` (and the resumable
`manifest.jsonl` beside it).

CC0 requires no attribution; the institution and caption are kept because provenance is worth more
than the licence strictly demands.

## Licence: images CC0, captions CDLA — and they are not the same

PD12M's **image** licence is a **mix**, and this is the single most important fact about this
source. Spawning's own Datasheet says, verbatim:

> In all cases, we only collected images with metadata indicating a Public Domain Mark or CC0
> license.

while the Hugging Face dataset card says, verbatim:

> PD12M consists of entirely public domain and CC0 licensed images, with automated recaptioning of
> image data, and quality and safety filtering.

Two different statuses, so the dataset card's blanket sentence is not enough on its own. The
admission decision is therefore made **per row**, on the metadata `license` column that the card
documents as "`license`: The URL of the image license." Only
`https://creativecommons.org/publicdomain/zero/1.0/` is downloaded
(`scripts/v2/fetch_pd12m.py:KEEP_LICENSE`); Public Domain Mark 1.0 rows are **not**, because
PDM-1.0 is not an SPDX id in `scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST`, and neither are the
{no_licence:,} rows with no licence value.

Census of that column over the {census['rows_read']:,} metadata rows actually read
(`data/decision-v2/licenses/pd12m/pd12m-license-census.json`):

{census_table}

The **metadata**, which includes the captions, is licensed differently. The dataset card says:

> The dataset is licensed under the [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/).

**CDLA-Permissive-2.0 is not in the commercial allowlist**, and no record carries it — because no
caption is written into a record. The only stored copies of PD12M metadata are
`attribution.jsonl` and the raw selection files, which are provenance, not training data. The
CDLA grant is on file anyway (`cdla-permissive-2.0.html`): "A Data Recipient may use, modify, and
share the Data made available by Data Provider(s) under this agreement if that Data Recipient
follows the terms of this agreement."

## How ~60,000 photographs were chosen (not at random)

{stats['rows_read']:,} metadata rows were read from {stats['metadata_shards_read']} of PD12M's 125
parquet shards, spread evenly across the dataset rather than taken from the front. Every row had
to pass, in this order:

{dropped}

The caption filter is what keeps this source to **photographs**. PD12M is drawn largely from
Wikimedia Commons, iNaturalist and OpenGLAM museum collections, so paintings, engravings, scanned
pages, coats of arms, maps and coins are common; a caption that says so is dropped:

{rejects}

Each survivor was then put in **one of {stats['buckets_available']} subject buckets** by the
earliest subject word in its caption — one regex, leftmost-first, so a caption that says "a wasp
sitting on top of a white cup" is filed under `insect`, not `kitchenware`
(`scripts/v2/pd12m_captions.py`). Buckets were held in a per-bucket reservoir sample of
{stats['reservoir_per_bucket']:,} (so a bucket's members come from every shard, not the first ones)
and then taken **round-robin from the rarest bucket outwards**, at most
{stats['per_bucket_cap']:,} per bucket. `selection.jsonl` is written in that round-robin order, so a
download stopped by the clock is still balanced across buckets.

Size filter: {stats['size_filter']['min_long_side']}–{stats['size_filter']['max_long_side']} px on
the long side and at least {stats['size_filter']['min_short_side']} px on the short side.

### Bucket distribution of the photographs actually stored

{stats['distinct_buckets']} buckets were selected from; {len(stored_buckets)} survived the download.

{bucket_table}

### Where the photographs come from

{contributors}

## Present and absent objects (both are claims, neither is a label)

{stats['with_claimed_present_objects']:,} of the {stats['selected']:,} selected photos have at
least one object the caption names ({stats['mean_claimed_present']} on average), drawn from a fixed
lexicon of {len(CAP.LEXICON_PHRASES)} everyday things so that a question reads as English.

* **present** — the caption said so. The caption is machine-written, so this can be wrong.
* **absent** — the caption did **not** say so, the thing is not in the bucket's typical-contents
  list (`photo.CATEGORY_ASSOCIATED`), and it is not one of the things photographs carry without a
  caption mentioning them (`pd12m_captions.OFTEN_UNSAID`: windows, doors, fences, trees, chairs …).

This is **weaker** than Open Images' human-verified `Confidence == 0` absences, and deliberately
so: it costs nothing when it is wrong, because the teacher decides every answer. A mistaken
"absent" simply produces a question the teacher answers with a real colour or a "yes" instead of
`unknown`.
"""

    body = f"""
## Counts

{partitions}

{families}

Field types: {json.dumps(summary['field_types'])}. Distinct question templates used:
{summary['distinct_templates']:,}. Records that carry Jev-style option criteria (an option or
yes/no answer with a short description of what it means): {summary['records_with_criteria']:,}
({summary['records_with_criteria'] / summary['records']:.0%}).

Option-set sizes: {small:,} records with 2–12 options/levels, {large:,} with 13–25
({large / max(1, small + large):.0%}); decision-v1 asks for about 10% in the 13–25 band, and only
`main_object` and `colour_of_x` can reach it, because the other families' option sets are fixed to
small natural pools (scene_type 6–10, twelve materials in all, indoor/outdoor 4, which-image 4,
and 4–5 levels for the ordinal families).

Two-image records: {summary['two_image_records']:,} ({summary['two_image_share']:.1%} of records),
of which {summary.get('two_image_same_category', 0):,} pair two photos from the **same subject
bucket** (the brief's "~10% two-image pairs within the same bucket"); the rest cross buckets on
purpose, so the "do these show the same kind of thing?" boolean is not answered yes every time. A
pair is always drawn from within one partition, so no photo reaches across the split through a
pair.

Constructions built so that the honest answer is `unknown`:
{summary['unknown_construction']:,} ({summary['unknown_construction_share']:.1%}) — inside the
spec's 15–25% band. Only four constructions count: `scene_type/unknown` and `main_object/unknown`
(nothing that fits is listed → `not_listed`), and `colour_of_x/absent` and `material/absent` (the
question asks for the colour or the material of something the caption does not place in the frame
→ `false_premise`). `object_present/absent`, `count_bucket/absent` and `two_image_which` about a
missing thing are deliberately **not** counted: "no", "none at all" and "neither image" are
honest, listed answers. They are built all the same, and often —
`ABSENT_OBJECT_SHARE_BY_SOURCE["pd12m"]` is 0.50, against 0.40 for the other two sources.

State: {summary['state_is_string']:,} records ({summary['state_string_share']:.0%}) carry a
free-text state, the rest a JSON object.

{state_kinds}

`contradictory` — a state that describes a **different scene** from the photograph — is pressed
harder here than in the other sources ({summary['state_kinds']['contradictory'] / summary['records']:.0%}
of records against a 20% default), because the brief asks this source for contradicting-state
questions. `aligned` describes the same photo but, for pd12m, carries **only the coarse bucket
word** (`"filed_under": "market stall"`, `"caption": "photograph of market stall"` — the template's
own fallback wording) and never PD12M's caption. `irrelevant` is unrelated machine state,
`empty` is `{{}}`.

## Licence receipts

{licences}

{licence_block(SOURCE)}

## Exclusions

* Every metadata row whose `license` is not exactly `https://creativecommons.org/publicdomain/zero/1.0/`
  — including all Public Domain Mark 1.0 rows and all rows with a null licence.
* Every row whose caption shows it to be an artwork, a scan, a manuscript page, a logo, a map, a
  coin or a poster (table above). We want photographs.
* Every row outside {stats['size_filter']['min_long_side']}–{stats['size_filter']['max_long_side']} px
  on the long side, or under {stats['size_filter']['min_short_side']} px on the short side.
* Every row whose MIME type is not `image/jpeg` or `image/png`.
* Duplicates by PD12M's own MD5, then again by the sha256 of our re-encoded copy.
* `data/decision-v1/exclusions.json`: {summary['photos_dropped_for_exclusion_list']} photos whose
  sha256 is on the ban list of images used by an earlier evaluation.

## What is NOT verified

* **The per-photo licence is asserted by PD12M, not verified by us at the originating
  institution.** Spawning says so itself: "While we believe that this is the most comprehensive
  attempt to create a public domain only image-text dataset to date, the sources of the original
  images, and by extension, Spawning, cannot fully guarantee that no copyrighted material appears
  in the dataset." Our filter is on PD12M's licence column, which is a claim about a claim.
* **The captions are machine-written and unchecked.** Every `present` object, every subject bucket
  and every artwork rejection rests on one. A wrong caption costs a question, never a label: the
  teacher answers from the image.
* **The absent-object claims are inferred from a caption's silence**, not from a human-verified
  absence. See above.
* **CDLA-Permissive-2.0 (the metadata/caption licence) is not in the commercial allowlist.** It is
  not used as any record's licence; the captions never enter a record. If a future step wanted to
  train on PD12M captions as text, that would need its own licence decision.{overlap_note}

## Validation

`PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_image_records.py pd12m` →
**{validation['summary']['errors']} errors** (`validation.json`).

`scripts/v2/validate_v2_records.py` is **not** the gate for this source, and running it here is a
mistake worth recording rather than hiding: it is the decision-v2 **text** gate, it asserts
`images == []`, and it partitions with the seed `"decision-v2-text"` rather than the photo
converter's `"decision-v2-photo"`. Run against these rows it reports **228,009 errors over 179,997
records**, of two structural kinds only — `text records must carry no images` and `partition does
not match stable_partition(source_group)` — both of which are statements about which gate was run,
not about the data. (It also overwrites `validation.json`, so the image gate must be re-run after
it.) The image gate is `validate_image_records.py`, which is `scripts/v1/validate_records.py`
ported check for check, with the licence-receipt check added; see the header of each file.

`scripts/v2/run_v1_validator.py pd12m` runs the **unmodified** v1 gate against a copy of these
records in a throw-away directory, and saves what it says to `v1_validator_report.json`:
{v1report}

The v1 gate cannot pass on these rows: it enforces `target is None <=> abstention_cause is not
None`, which a `pending` candidate row breaks by construction.

## Provenance

* collection: `scripts/v2/fetch_pd12m.py` (phases `meta`, `select`, `download`; seed 20260923)
* caption reading: `scripts/v2/pd12m_captions.py`
* converter: `scripts/v2/convert_photos.py` (seed 20260923)
* templates: `scripts/v2/templates/photo.py` (the `PD12M_CATEGORY_*` tables add this source's
  buckets; the other two sources never emit them)
* licence receipts: `scripts/v2/write_license_receipts_pd12m.py`
* raw manifests: `data/decision-v2-raw/pd12m/` (`metadata/`, `selection.jsonl`, `manifest.jsonl`,
  `manifest.json`)
* tests: `tests/test_v2_pd12m.py`
"""
    return head + body


if __name__ == "__main__":
    text = build()
    (OUT / SOURCE / "README.md").write_text(text)
    print(f"wrote data/decision-v2/{SOURCE}/README.md ({len(text)} chars)")
