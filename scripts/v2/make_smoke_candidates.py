"""Build a small synthetic decision-v2 candidate set under data/decision-v2-smoke/.

The real candidate sources are being collected separately; this set exists so the pseudo-labeller,
the assembler and the audit can be exercised end to end on a laptop.  It reuses *questions and
images that already exist in v1.1* with their targets stripped (``target: null``,
``pseudo_label: "pending"``), so it is a shape fixture, never a new dataset:

- 20 text-only candidates from v1.1 test text rows (choice/boolean/ordinal, small and large option
  sets, two multi-question requests);
- 20 image candidates from ``data/decision-v1/abo`` test rows.  Images are re-encoded through
  ``decision_data.load_image`` into ``data/decision-v2-smoke/smoke_image/images/<sha>.jpg`` (a copy
  with its own content hash, not a link), so the fixture never shares an image hash with v1.1.

Licence evidence is the *same reviewed evidence file and receipt* as the v1 source, copied into
``data/decision-v2-smoke/licenses/<source>/<upstream>/`` so the repo-relative v2 licence layout is
exercised.
Nothing under data/decision-v1*, data/manifests or reports/ is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from v1_text.common import stable_partition, verified_license

from decision_data import load_image

ROOT = Path(__file__).resolve().parents[2]
HOME = Path("data/decision-v2-smoke")
SEED = "decision-v2-smoke"
TEXT_SOURCES = [("fever", 4), ("boolq", 4), ("snli", 4), ("esci", 4), ("banking77", 4)]


def harvest(v1_path, wanted, images):
    """First ``n`` v1.1 test rows per source, either text-only or image rows."""
    need = dict(wanted)
    found = {s: [] for s in need}
    with Path(v1_path).open() as handle:
        for line in handle:
            if ('"images": []' in line) == images:
                continue
            row = json.loads(line)
            source = row.get("source")
            if row.get("partition") != "test" or source not in need or len(found[source]) >= need[source]:
                continue
            found[source].append(row)
            if all(len(v) >= need[k] for k, v in found.items()):
                break
    missing = [k for k, v in found.items() if len(v) < need[k]]
    if missing:
        raise SystemExit(f"not enough v1.1 test rows for {missing}")
    return found


def licence(source, origin, upstream):
    """Copy the reviewed evidence + receipt into the v2 layout and verify it there.

    One directory per upstream source: the v1 evidence files share filenames (LICENSE.txt), so a
    flat directory would silently overwrite one grant with another.
    """
    evidence = Path(origin["evidence"])
    target_dir = HOME / "licenses" / source / upstream
    (ROOT / target_dir).mkdir(parents=True, exist_ok=True)
    for name in (evidence.name, evidence.name + ".receipt.json"):
        shutil.copyfile(ROOT / evidence.parent / name, ROOT / target_dir / name)
    return verified_license(target_dir / evidence.name, origin["spdx"])


def store_image(source, path, pixels=1_000_000):
    """Re-encode through the v2 pixel budget; the copy is addressed by its own hash."""
    directory = ROOT / HOME / source / "images"
    directory.mkdir(parents=True, exist_ok=True)
    image = load_image(ROOT / path, pixels)
    staging = directory / "staging.jpg"
    image.save(staging, "JPEG", quality=95)
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    final = directory / f"{digest}.jpg"
    staging.replace(final)
    return {"image": str(HOME / source / "images" / f"{digest}.jpg"), "sha256": digest,
            "width": image.width, "height": image.height}


def candidate(source, index, family, template, fields, state, lic, images=(), group=None):
    key = f"{source}-{index:03d}"
    return {"id": f"{source}:{key}", "source": source, "source_group": group or key,
            "partition": stable_partition(group or key, SEED), "family": family, "source_split": "crawl",
            "license": lic, "images": list(images),
            "request": {"schema_version": "1.0", "request_id": key, "state": state, "fields": fields},
            "target": None, "abstention_cause": None, "pseudo_label": "pending", "template_id": template}


SUFFICIENCY = {"id": "sufficient", "type": "boolean",
               "question": "Is the supplied evidence enough to settle the question above?",
               "yes_description": "the evidence decides it", "no_description": "the evidence leaves it open"}
QUALITY = {"id": "quality", "type": "ordinal", "question": "Rate how clearly the product photo shows the item.",
           "levels": [{"value": 1, "description": "the item is not usable in this photo"},
                      {"value": 2, "description": "partly visible or obscured"},
                      {"value": 3, "description": "the item is fully and clearly visible"}]}
OFF_TOPIC = {"id": "answer", "type": "choice",
             "question": "What is the expiry date printed on the packaging of this item?",
             "options": [{"value": "2026-01-01"}, {"value": "2027-06-30"}, {"value": "no expiry is printed"}]}


def text_candidates(rows):
    out = []
    licences = {}
    index = 0
    for source, _ in TEXT_SOURCES:
        for row in rows[source]:
            licences.setdefault(source, licence("smoke_text", row["license"], source))
            field = json.loads(json.dumps(row["request"]["fields"][0]))
            state = json.loads(json.dumps(row["request"].get("state", {})))
            index += 1
            if index in (3, 9):  # two multi-question requests
                second = json.loads(json.dumps(SUFFICIENCY))
                out.append(candidate("smoke_text", index, f"{source}_pair", f"{source}+sufficiency",
                                     [field, second], state, licences[source]))
            else:
                out.append(candidate("smoke_text", index, source, source, [field], state, licences[source]))
    return out


def image_candidates(rows):
    out = []
    lic = None
    index = 0
    for row in rows["abo"]:
        lic = lic or licence("smoke_image", row["license"], row["source"])
        image = store_image("smoke_image", row["images"][0]["image"])
        field = json.loads(json.dumps(row["request"]["fields"][0]))
        index += 1
        if index % 7 == 0:  # constructed unanswerable question: the honest answer is unknown
            out.append(candidate("smoke_image", index, "abo_unanswerable", "off_topic",
                                 [json.loads(json.dumps(OFF_TOPIC))], {}, lic, [image]))
        elif index % 5 == 0:  # multi-question: the source question plus a constructed ordinal
            out.append(candidate("smoke_image", index, "abo_pair", "abo+quality",
                                 [field, json.loads(json.dumps(QUALITY))], {}, lic, [image]))
        elif index % 4 == 0:
            out.append(candidate("smoke_image", index, "abo_quality", "quality",
                                 [json.loads(json.dumps(QUALITY))], {}, lic, [image]))
        else:
            state = {"listing": {"asin": row.get("source_group", "unknown")}} if index % 3 == 0 else {}
            out.append(candidate("smoke_image", index, row["family"], row["source"], [field], state, lic, [image]))
    return out


def blend_fixture(v1_path, count, destination):
    """A small, deliberately varied slice of v1.1 *training* rows as the --blend smoke input.

    They already carry target / targets (and sometimes target_distribution), which is exactly what
    the self-distillation pass consumes.  The quotas make sure the smoke sees every shape that
    behaves differently: an image row, a text row, a row that already has a soft distribution, a
    multi-question row, and a ``not_listed`` row (which blending must pass through untouched,
    because the trainer's augmenter appends an option its soft target could not name).  Nothing is
    stripped and nothing is written back to v1.1.
    """
    def kind(row):
        if row.get("abstention_cause") == "not_listed" or "not_listed" in (row.get("abstention_causes") or {}).values():
            return "not_listed"
        if len(row["request"]["fields"]) > 1:
            return "multi_field"
        if row.get("target_distribution"):
            return "soft_target"
        return "image" if row.get("images") else "text"

    quota = {"image": 3, "text": 2, "soft_target": 2, "multi_field": 2, "not_listed": 1}
    scale = max(1, round(count / sum(quota.values())))
    quota = {k: v * scale for k, v in quota.items()}
    found = {k: [] for k in quota}
    with Path(v1_path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition") != "train":
                continue
            bucket = kind(row)
            if len(found[bucket]) < quota[bucket]:
                found[bucket].append(row)
            if all(len(found[k]) >= quota[k] for k in quota):
                break
    rows = [r for k in quota for r in found[k]][:count]
    path = ROOT / destination
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n" for r in rows))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", default="data/manifests/decision-v1.1.jsonl")
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--blend-rows", type=int, default=10,
                   help="also write that many v1.1 training rows for the --blend smoke")
    a = p.parse_args(argv)
    import os
    os.chdir(ROOT)
    text = text_candidates(harvest(a.v1, TEXT_SOURCES, images=False))
    images = image_candidates(harvest(a.v1, [("abo", a.images)], images=True))
    for source, rows in (("smoke_text", text), ("smoke_image", images)):
        path = ROOT / HOME / source / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n" for r in rows))
        print(source, len(rows), "records", sum(len(r["request"]["fields"]) for r in rows), "decisions",
              {x: sum(1 for r in rows if r["partition"] == x) for x in ("train", "dev", "test")})
    blended = blend_fixture(a.v1, a.blend_rows, HOME / f"v1.1-train-{a.blend_rows}.jsonl") if a.blend_rows else []
    if blended:
        print("blend fixture", len(blended), "v1.1 training rows",
              sum(len(r["request"]["fields"]) for r in blended), "decisions")
    return text, images


if __name__ == "__main__":
    main()
