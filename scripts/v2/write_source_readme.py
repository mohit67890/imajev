"""Write `data/decision-v2/<source>/README.md` from the artefacts, so no number is typed by hand.

Everything numeric comes from `conversion_summary.json`, `validation.json`, the raw manifests and
the licence receipts; the prose is fixed text that states what was done and what was NOT verified.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/write_source_readme.py commons_photos
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, read_jsonl
from v2.templates import photo as T

REVIEWED_AT = "2026-09-23"
REVIEWER = "Claude v2 collection agent"


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def licence_block(source):
    lines = []
    for receipt in sorted((OUT / "licenses" / source).glob("*.receipt.json")):
        meta = json.loads(receipt.read_text())
        lines.append(
            f"* **{meta['spdx']}** — evidence `{receipt.name[:-len('.receipt.json')]}` "
            f"(<{meta['source_url']}>), sha256 `{meta['evidence_sha256'][:16]}…`, "
            f"`commercial_use_reviewed: {str(meta['commercial_use_reviewed']).lower()}`, "
            f"reviewed {meta['reviewed_at']} by \"{meta['reviewed_by']}\".\n"
            f"  Scope: {meta['scope']}.\n"
            f"  Quoted grant: \"{meta['quoted_grant']}\"\n"
            f"  Note: {meta['notes']}")
    for prov in sorted((OUT / "licenses" / source).glob("*.provenance.json")):
        meta = json.loads(prov.read_text())
        quote = f" Quote found in the stored file: `{meta['quote_found_in_stored_file']}`." if "quote" in meta else ""
        lines.append(f"* supporting: `{prov.name[:-len('.provenance.json')]}` (<{meta['source_url']}>), "
                     f"sha256 `{meta['sha256'][:16]}…`. {meta['note']}{quote}")
    return "\n".join(lines)


def common_sections(source, summary, validation):
    part = summary["partitions"]
    fam = summary["families"]
    total = summary["records"]
    families = table(
        [(f, n, f"{n/total:.1%}") for f, n in sorted(fam.items(), key=lambda kv: -kv[1])],
        ["family", "records", "share"])
    partitions = table([(p, part.get(p, 0), f"{part.get(p,0)/total:.1%}") for p in ("train", "dev", "test")],
                       ["partition", "records", "share"])
    opts = validation["summary"]["option_counts"]
    choice_like = {int(k): v for k, v in opts.items() if int(k) > 0}
    small = sum(v for k, v in choice_like.items() if k <= 12)
    large = sum(v for k, v in choice_like.items() if k > 12)
    return families, partitions, small, large


def build(source: str) -> str:
    out = OUT / source
    v1path = out / "v1_validator_report.json"
    v1report = (
        f"**{json.loads(v1path.read_text())['errors']:,} errors for "
        f"{json.loads(v1path.read_text())['records']:,} records**, every saved one of the single "
        f"kind `target None <=> abstention_cause set`."
        if v1path.is_file() else "(not run)")
    summary = json.loads((out / "conversion_summary.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    attribution = read_jsonl(out / "attribution.jsonl")
    families, partitions, small, large = common_sections(source, summary, validation)
    images = len({r["sha256"] for r in read_jsonl(RAW / source / "manifest.jsonl")})
    metadata_rows = len(read_jsonl(RAW / source / "metadata.jsonl")) if (RAW / source / "metadata.jsonl").is_file() else 0
    stored_mb = sum(p.stat().st_size for p in (out / "images").glob("*.jpg")) / 1e6
    state_kinds = table(sorted(summary["state_kinds"].items(), key=lambda kv: -kv[1]),
                        ["state", "records"])
    licences = table(sorted(summary["licenses"].items()), ["SPDX", "records"])

    if source == "commons_photos":
        progress = json.loads((RAW / source / "metadata_progress.json").read_text())
        seeds = table(sorted(((k, v) for k, v in progress["seeds"].items()), key=lambda kv: -kv[1]),
                      ["category label", "files kept"])
        refused = collections.Counter()
        for stats in progress["stats"].values():
            for name, n in (stats.get("license_seen") or {}).items():
                refused[name] += n
        refused_table = table(refused.most_common(15), ["licence seen on a candidate file", "files refused"])
        skips = collections.Counter()
        for stats in progress["stats"].values():
            for k, v in stats.items():
                if k.startswith("skip_") and isinstance(v, int):
                    skips[k] += v
        skip_table = table(sorted(skips.items(), key=lambda kv: -kv[1]), ["reason a candidate was dropped", "files"])
        head = f"""# commons_photos — Wikimedia Commons photographs (decision-v2 candidates)

Photographs the model has never seen, collected from Wikimedia Commons on {REVIEWED_AT} and turned
into **unanswered** decision candidates: `target: null`, `abstention_cause: null`,
`pseudo_label: "pending"`. The 9B teacher labels them later; nothing in this directory claims an
answer.

## What was collected

{images:,} distinct photographs ({stored_mb:,.0f} MB stored, re-encoded to RGB JPEG q90 under the
1 MP cap by `scripts/v1/_common.py:store_image`), across {len(progress['seeds'])} subject
categories. Per-file attribution — author, licence, licence URL and the Commons file page — is in
`attribution.jsonl` ({len(attribution):,} rows), which is what the model card needs.

{seeds}

## How files were chosen, and how the licence was decided

Candidates come from the Commons search API, asking for files that are in a subject's category tree
(`deepcategory:`) *and* in one of the three licence maintenance categories (`CC-BY-4.0`,
`CC-BY-3.0`, `CC-Zero`); a plain `intitle:` query is the fallback for subjects whose tree is too
large for `deepcategory:` to expand. **Category membership only finds candidates. The admission
decision is made per file** from that file's own `imageinfo` → `extmetadata`, and a file is kept
only when

* `LicenseShortName` is exactly `CC0`, `CC BY 3.0` or `CC BY 4.0`, **and**
* the machine-readable `License` tag agrees (`cc0`, `cc-by-3.0`, `cc-by-4.0`), **and**
* `Restrictions` is empty, **and**
* the file is a JPEG or PNG of at least 400×400 and 0.24 MP.

Everything else is refused. Licences actually seen on refused candidates:

{refused_table}

Other reasons a candidate was dropped:

{skip_table}

A sample of the verbatim API licence blocks behind these decisions is stored at
`data/decision-v2/licenses/commons_photos/commons-api-license-samples.json`.
"""
        extra_exclusions = """* **CC BY-SA (any version), GFDL, PD-*/"Public domain", CC BY 2.0/2.5 and every NC or ND
  variant are excluded**, per the spec's allowlist. Share-alike is excluded even though it is in
  `COMMERCIAL_ALLOWLIST`, because the brief for this source asked for CC0/CC-BY only.
* Files whose `Restrictions` field is non-empty (trademark, personality rights, insignia) are
  excluded even when the copyright licence is fine.
* Titles that look like non-photographic media (coats of arms, flags, logos, maps, diagrams,
  charts, screenshots) are excluded.
"""
        caveats = f"""* **Commons has no per-file object annotation.** The only subject claim available is the category
  a human filed the file under, so `category_label` is a curator's claim, not a verified label. It
  is used only to make an option set plausible and to pair two-image records; it is never a target.
* Because of that, the "thing that is probably not in the photo" used by the
  `colour_of_x/absent`, `material/absent`, `object_present/absent` and `count_bucket/absent`
  constructions is **plausible but unverified** for this source (objects strongly associated with
  the category are held out — see `CATEGORY_ASSOCIATED` in `scripts/v2/templates/photo.py` — but
  the photo is not inspected). The teacher decides the answer, so a wrong guess costs a dropped
  candidate, not a wrong label.
* The search API ranks by relevance, so a subject's files are not a uniform sample of that
  category.
* **The category label is noisy.** Commons category trees wander: a `deepcategory:` walk of
  "Shoes" reaches costume and portrait categories, so a Louis XIV portrait can arrive labelled
  `footwear`. Nothing downstream treats the label as truth -- it only shapes the option set and
  chooses a pairing partner, and the teacher answers from the photograph -- but a two-image
  "same kind of subject?" pair built from one noisy label is a harder, sometimes wrong-looking
  pair, and `scene_type` options for a mislabelled photo may not contain the right scene (the
  honest answer is then `unknown`, which the teacher can give).
* **This source is short of its target.** The brief asked for about 25,000 photographs; collection
  was stopped early by the run's time budget. {images:,} of the {metadata_rows:,} licence-screened
  candidates in `data/decision-v2-raw/commons_photos/metadata.jsonl` were downloaded. The fetcher
  is resumable and idempotent -- `PYTHONPATH=src:scripts .venv/bin/python scripts/v2/fetch_commons.py
  download --workers 10` picks up exactly where it stopped, and re-running `convert_photos.py`
  afterwards extends the source. Throughput was about 160-240 photographs a minute at 10 workers;
  upload.wikimedia.org starts returning 429 at 14."""
    else:
        stats = json.loads((RAW / source / "selection_stats.json").read_text())
        spot_path = OUT / "licenses" / source / "flickr-license-spotcheck.json"
        spot = json.loads(spot_path.read_text()) if spot_path.is_file() else None
        spot_text = (
            f"{spot['checked']} sampled photos' Flickr pages were fetched and the Creative Commons "
            f"licence they link was recorded: {json.dumps(spot['verdicts'])} "
            f"(`data/decision-v2/licenses/openimages_v2/flickr-license-spotcheck.json`). This is a "
            f"sample, **not** a per-photo verification."
            if spot else "No Flickr spot-check was run.")
        head = f"""# openimages_v2 — Open Images V7, TEST split only (decision-v2 candidates)

Photographs the model has never seen, taken from the Open Images V7 **test** split on {REVIEWED_AT}
and turned into **unanswered** decision candidates: `target: null`, `abstention_cause: null`,
`pseudo_label: "pending"`. The 9B teacher labels them later; nothing here claims an answer.

## Why the test split, and how leakage was ruled out

decision-v1 already used Open Images photographs through TextVQA, whose image ids are Open Images
**train** and **validation** ids (`data/decision-v1/textvqa/README.md`), and the earlier evaluation
pilot (`scripts/prepare_pilot_openimages.py`) pulled from **validation**. Both id sets were loaded
and intersected with the test manifest: **{stats['filters'].get('previously_seen', 0)} overlapping
ids** (TextVQA's 25,119 distinct image ids are all train/val; the evaluation pilot's ids are all
validation). Records also pass the `data/decision-v1/exclusions.json` sha256 ban list, which the
validator re-checks.

## What was collected

{images:,} distinct photographs ({stored_mb:,.0f} MB stored, re-encoded to RGB JPEG q90 under the
1 MP cap), downloaded from the CVDF public mirror
`https://open-images-dataset.s3.amazonaws.com/test/<id>.jpg`. Per-photo attribution — Flickr author,
author profile, title, licence and the Flickr photo page — is in `attribution.jsonl`
({len(attribution):,} rows). The download manifest with URL + sha256 per photo is
`data/decision-v2-raw/openimages_v2/manifest.jsonl`.

## Label diversity (selection only — labels are never targets)

{stats['candidates_after_filters']:,} test photos survived the licence/rotation/leakage filters and
{stats['usable_candidates']:,} carry at least one concrete human-verified image-level label
({stats['labels_in_test_split']:,} distinct labels after dropping abstract ontology entries such as
"Photograph", "Pattern" or "Daytime"). Each photo was bucketed by its **rarest** verified-present
label and photos were taken round-robin from the rarest bucket outwards, at most 25 per label.
The result spans **{stats['distinct_bucket_labels']:,} distinct bucket labels** and
{stats['distinct_present_labels']:,} distinct present labels;
{stats['photos_with_verified_absences']:,} of the selected photos also carry human-verified
**absences** (`Confidence == 0`), which is what makes an honest "is there a X?" distractor possible.

Image-level labels are used for exactly two things: choosing this diverse subset, and supplying
present/absent object candidates to the question templates. **No label is ever written as a
target.**
"""
        extra_exclusions = f"""* Photos needing rotation ({stats['filters'].get('needs_rotation', 0):,}) are excluded rather than
  silently rotated.
* Photos with no concrete verified label are excluded (they cannot seed a candidate pool).
* Every image id used by TextVQA (Open Images train+val) and by the evaluation pilot (validation)
  is excluded; the intersection with the test split was {stats['filters'].get('previously_seen', 0)}.
"""
        caveats = f"""* **The per-photo licence is asserted by the dataset, not verified by us for every photo.** Open
  Images' own words: "Note: while we tried to identify images that are licensed under a Creative
  Commons Attribution license, we make no representations or warranties regarding the license
  status of each image and you should verify the license for each image yourself."
  {spot_text}
* **CC-BY-2.0 is not in `scripts/v1_text/common.py:COMMERCIAL_ALLOWLIST`.** decision-v2 widens the
  allowlist to include it (`scripts/v2/common.py:V2_COMMERCIAL_ALLOWLIST`) because CC BY 2.0 is
  attribution-only, with no NC, ND or SA term — the deed grants use "for any purpose, even
  commercially". This is a deliberate, documented deviation from the v2 spec's allowlist.
* (Verified, for contrast) all {stats['candidates_after_filters'] + stats['filters'].get('needs_rotation', 0):,}
  rows of the Open Images test manifest carry the same licence URL,
  `https://creativecommons.org/licenses/by/2.0/`; no row carried anything else. That is a fact
  about the dataset's own manifest, which is a weaker claim than checking each photo at Flickr."""

    body = f"""
## Counts

{partitions}

{families}

Field types: {json.dumps(summary['field_types'])}. Distinct question templates used:
{summary['distinct_templates']:,}. Records that carry Jev-style option criteria (an option or yes/no answer with a short description
of what it means): {summary['records_with_criteria']:,}
({summary['records_with_criteria']/summary['records']:.0%}).

Option-set sizes: {small:,} records with 2–12 options/levels, {large:,} with 13–25
({large/max(1, small+large):.0%}). decision-v1 asks for about 10% in the 13–25 band; here only
`main_object` and `colour_of_x` can reach it, because the brief fixes the other families' option
sets to small natural pools (scene_type 6–10, materials 12 in total, indoor/outdoor 4,
which-image 4, and the ordinal families 4–5 levels).

Two-image records: {summary['two_image_records']:,} ({summary['two_image_share']:.1%} of records),
of which {summary.get('two_image_same_category', 0):,} pair two photos of the same category. A pair
is always drawn from within one partition, so no photo reaches across the split through a pair.

Constructions built so that the honest answer is `unknown`:
{summary['unknown_construction']:,} ({summary['unknown_construction_share']:.1%}). Only four
constructions count here — `scene_type/unknown` and `main_object/unknown` (nothing that fits is
listed → `not_listed`), and `colour_of_x/absent` and `material/absent` (the question names
something that is not in the photo → `false_premise`). `object_present/absent`,
`count_bucket/absent` and `two_image_which` about a missing thing are deliberately **not** counted:
"no", "none at all" and "neither image" are honest, listed answers.

State: {summary['state_is_string']:,} records ({summary['state_string_share']:.0%}) carry a
free-text state, the rest a JSON object.

{state_kinds}

`aligned` describes the same photo, `irrelevant` is unrelated machine state, `contradictory`
describes a **different scene** than the photograph (so a question answered from the state alone
contradicts the image), `empty` is `{{}}`.

## Licence

{licences}

{licence_block(source)}

## Exclusions

{extra_exclusions}
## What is NOT verified

{caveats}

## Validation

`PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_image_records.py {source}` →
**{validation['summary']['errors']} errors** (`validation.json`).

`scripts/v2/run_v1_validator.py {source}` runs the **unmodified** v1 gate against a copy of these
records in a throw-away directory, and saves what it says to `v1_validator_report.json`:
{v1report}

The v1 gate cannot pass on these rows: it is
hard-wired to `data/decision-v1/<source>/` and enforces `target is None <=> abstention_cause is not
None`, which a `pending` candidate row breaks by construction. `scripts/v2/validate_image_records.py`
is that script ported check for check, with the root changed and that one rule replaced by "a
pending row must have a null target *and* a null cause"; it additionally verifies every row's
licence receipt. See the header of that file.

## Provenance

* converter: `scripts/v2/convert_photos.py` (seed 20260923)
* templates: `scripts/v2/templates/photo.py`
* collection: `scripts/v2/{'fetch_commons.py' if source == 'commons_photos' else 'fetch_openimages.py'}`
* raw manifests: `data/decision-v2-raw/{source}/`
* tests: `tests/test_v2_photo_templates.py`
"""
    return head + body


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["commons_photos", "openimages_v2"])
    args = ap.parse_args()
    text = build(args.source)
    (OUT / args.source / "README.md").write_text(text)
    print(f"wrote data/decision-v2/{args.source}/README.md ({len(text)} chars)")
