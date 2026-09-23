"""Validation gate for decision-v2 TEXT sources.

`scripts/v1/validate_records.py` cannot validate these rows.  It is hard-wired to
`data/decision-v1/<source>/records.jsonl` and, run against a copy of a v2 text file, rejects
every row for reasons that are about v1's image records rather than about this data:

  * `missing source_answer`                 -- a v1 field the v2 schema drops (there is no
                                               human answer: the teacher supplies it later);
  * `need 1-2 images`                       -- text records carry `images: []`;
  * `target None <=> abstention_cause set`  -- v2 rows are `target: null` with
                                               `pseudo_label: "pending"`, not abstentions;
  * `assert len(req.fields) == 1`           -- v2 allows 2-5 field multi-question requests.

So, as the brief instructs, this gate runs `scripts/decision_data.py:expand_fields` followed by
`render` over EVERY row -- the same code path training and serving use -- plus the structural
checks that still apply (ids, partitions, licence receipts, option hygiene, group leakage).

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/validate_v2_records.py stackexchange
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decision_data import expand_fields, render          # noqa: E402
from v1_text.common import stable_partition              # noqa: E402
from v2.common import verified_license                   # noqa: E402
from vision_decision.contracts import UNKNOWN, Request   # noqa: E402

PARTITIONS = {"train", "dev", "test"}
PARTITION_SEED = "decision-v2-text"


def validate(source: str) -> dict:
    root = ROOT / "data" / "decision-v2" / source
    rows = [json.loads(line) for line in (root / "records.jsonl").open() if line.strip()]
    errors: list[str] = []
    ids: set[str] = set()
    groups: dict[str, set[str]] = {}
    families: collections.Counter = collections.Counter()
    field_types: collections.Counter = collections.Counter()
    option_counts: collections.Counter = collections.Counter()
    partitions: collections.Counter = collections.Counter()
    variants: collections.Counter = collections.Counter()
    licences: collections.Counter = collections.Counter()
    fields_per_record: collections.Counter = collections.Counter()
    decisions = unknown_built = 0
    rng = random.Random("decision-v2-validate")

    for n, r in enumerate(rows):
        def bad(msg: str) -> None:
            errors.append(f'{r.get("id", n)}: {msg}')

        missing = [k for k in ("id", "source", "source_group", "partition", "family", "source_split",
                               "license", "images", "request", "target", "abstention_cause",
                               "pseudo_label", "template_id") if k not in r]
        if missing:
            bad(f"missing {', '.join(missing)}")
            continue
        if r["id"] in ids:
            bad("duplicate id")
        ids.add(r["id"])
        if r["source"] != source:
            bad("source field does not match the directory")
        if r["partition"] not in PARTITIONS:
            bad("bad partition")
        if r["partition"] != stable_partition(r["source_group"], seed=PARTITION_SEED):
            bad("partition does not match stable_partition(source_group)")
        if r["images"]:
            bad("text records must carry no images")
        if r["target"] is not None or r["abstention_cause"] is not None:
            bad("candidate rows must be target null / abstention_cause null until the teacher runs")
        if r["pseudo_label"] != "pending":
            bad("pseudo_label must be 'pending' before teacher labelling")
        if (r["partition"] == "test") != bool(r.get("pseudo_label_test")):
            bad("test rows must carry pseudo_label_test true, others must not")
        if not r["template_id"]:
            bad("empty template_id")

        try:
            licence = verified_license(Path(r["license"]["evidence"]), r["license"]["spdx"])
            if licence != r["license"]:
                bad("licence object does not match a fresh verification of its evidence")
            licences[r["license"]["spdx"]] += 1
        except Exception as exc:
            bad(f"licence: {exc}")

        groups.setdefault(r["source_group"], set()).add("test" if r["partition"] == "test" else "fit")
        partitions[r["partition"]] += 1
        variants[r.get("state_variant", "?")] += 1
        families[r["family"]] += 1

        try:
            request = Request.model_validate(r["request"])
        except Exception as exc:
            bad(f"request: {exc}")
            continue
        fields_per_record[len(request.fields)] += 1
        if not 1 <= len(request.fields) <= 5:
            bad("2-5 fields for multi-question requests, 1 otherwise")

        for f in r["request"]["fields"]:
            field_types[f["type"]] += 1
            items = f.get("options") or f.get("levels") or []
            option_counts[len(items)] += 1
            if f["type"] == "choice":
                values = [str(o["value"]) for o in items]
                if len(set(values)) != len(values):
                    bad(f"duplicate option values in {f['id']}")
                if any(v == UNKNOWN for v in values):
                    bad("__unknown__ is added by the loader, never listed")
                if any(a != b and a in b for a in values for b in values):
                    bad(f"one option value contains another in {f['id']}")
            decisions += 1
        unknown_built += sum(1 for v in (r.get("unknown_by_construction") or {}).values() if v)

        # The training/serving path itself: expand a multi-question request into one item per
        # field, then render each item exactly as the trainer would.
        probe = json.loads(json.dumps(r))
        if len(probe["request"]["fields"]) > 1:
            probe["targets"] = {f["id"]: None for f in probe["request"]["fields"]}
        try:
            for item in expand_fields(probe):
                header, choices, texts, index = render(item, rng)
                if choices[index][0] != UNKNOWN:
                    bad("pending rows must render with unknown as the target index")
                if not header or len(texts) != len(choices):
                    bad("render produced an inconsistent prompt")
        except Exception as exc:
            bad(f"expand_fields/render: {exc}")

    leak = [g for g, s in groups.items() if len(s) > 1]
    if leak:
        errors.append(f"{len(leak)} source_groups appear in both test and train/dev, e.g. {leak[:3]}")

    summary = {
        "source": source,
        "records": len(rows),
        "decisions": decisions,
        "source_groups": len(groups),
        "partitions": dict(partitions),
        "families": dict(sorted(families.items())),
        "field_types": dict(sorted(field_types.items())),
        "fields_per_record": dict(sorted(fields_per_record.items())),
        "option_counts": dict(sorted(option_counts.items())),
        "state_variants": dict(sorted(variants.items())),
        "licenses": dict(sorted(licences.items())),
        "unknown_by_construction": unknown_built,
        "unknown_share": round(unknown_built / max(1, decisions), 4),
        "errors": len(errors),
    }
    (root / "validation.json").write_text(
        json.dumps({"summary": summary, "validator": "scripts/v2/validate_v2_records.py",
                    "first_errors": errors[:50]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        print("\n".join(errors[:20]))
    return summary if not errors else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()
    ok = True
    for source in args.sources:
        ok = bool(validate(source)) and ok
    if not ok:
        sys.exit(1)
    print("VALID")


if __name__ == "__main__":
    main()
