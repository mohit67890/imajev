"""Build ``data/manifests/decision-v2-reasoning-dev.jsonl``.

A human-labelled, text-only development set used ONLY for checkpoint selection during the
v2 run (see ``docs/decision-v2-pseudolabel-spec.md``, "Training recipe for v2", step 3).
Nothing in it is ever trained on.

Two parts:

* ``typed_decisions_devsel`` -- the typed-decisions TRAIN parquet, converted with the v1
  converter and renamed so it can never be confused with the v1.1 training rows or with
  the evaluation-only ``typed_decisions_test`` panel. Apache-2.0, evidence already retained
  under ``data/decision-v1-text/licenses/typed_decisions/``.
* ``reasoning_authored`` -- 240 original decisions written for this repository (CC0-1.0),
  30 in each of eight JevBench-style reasoning families. No JevBench file was read to make
  them; only the public family names are reused.

Every row carries ``partition: "dev"`` and ``dev_role: "reasoning"``. ``heldout_family`` is
FALSE: ``scripts/v1_text/audit_mixture.py`` fails any row with ``heldout_family: true``
outside the test partition, and this set must stay in dev because the trainer reads it as a
dev signal. The held-out property is carried by ``dev_role`` and by the source names.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from decision_data import expand_fields, render
from v1_text.common import write_jsonl
from v1_text.convert_typed_decisions import convert as convert_typed_decisions
from v2.common import verified_license
from v2.reasoning_dev import all_items

ROOT = Path(__file__).resolve().parents[2]
TYPED_SOURCE = ROOT / "data/decision-v1-text/raw/typed-decisions/train.parquet"
TYPED_LICENSE = Path("data/decision-v1-text/licenses/typed_decisions/README.md")
AUTHORED_LICENSE = Path("data/decision-v2/licenses/reasoning_authored/CC0.txt")
OUTPUT = ROOT / "data/manifests/decision-v2-reasoning-dev.jsonl"
TYPED_SOURCE_NAME = "typed_decisions_devsel"
AUTHORED_SOURCE_NAME = "reasoning_authored"
DEV_ROLE = "reasoning"


def typed_rows() -> list[dict]:
    """Typed-decisions train split, renamed and pinned to the dev partition."""
    rows = convert_typed_decisions(TYPED_SOURCE, TYPED_LICENSE, "Apache-2.0")
    out = []
    for row in rows:
        if row["source"] != "typed_decisions" or row["source_split"] != "train":
            raise ValueError(f"{row['id']}: unexpected source/split for the dev-selection split")
        row["id"] = row["id"].replace("typed_decisions:", f"{TYPED_SOURCE_NAME}:", 1)
        row["source"] = TYPED_SOURCE_NAME
        row["source_group"] = f"{TYPED_SOURCE_NAME}:{row['source_group']}"
        row["partition"] = "dev"
        row["heldout_family"] = False
        row["dev_role"] = DEV_ROLE
        row["label_source"] = "human"
        row["pseudo_label"] = None
        row["template_id"] = "typed_decisions/published_questions"
        out.append(row)
    return out


def authored_rows() -> list[dict]:
    """The 240 hand-written reasoning items."""
    license_info = verified_license(AUTHORED_LICENSE, "CC0-1.0")
    out = []
    for item in all_items():
        key = item["key"]
        field = json.loads(json.dumps(item["field"]))
        row = {
            "id": f"{AUTHORED_SOURCE_NAME}:{item['family']}:{key}",
            "source": AUTHORED_SOURCE_NAME,
            "source_group": f"{AUTHORED_SOURCE_NAME}:{key}",
            "source_split": "authored",
            "partition": "dev",
            "family": item["family"],
            "heldout_family": False,
            "dev_role": DEV_ROLE,
            "domain": item["domain"],
            "license": license_info,
            "images": [],
            "request": {"schema_version": "1.0",
                        "request_id": f"{AUTHORED_SOURCE_NAME}-{key}",
                        "state": item["state"],
                        "fields": [field]},
            "target": item["target"],
            "abstention_cause": item["abstention_cause"],
            "rationale": item["rationale"],
            "label_source": "authored",
            "pseudo_label": None,
            "template_id": f"authored/{item['family']}",
        }
        out.append(row)
    return out


def check(rows: list[dict]) -> None:
    """Fail closed on anything the manifest promises: schema, labels, partition, roles."""
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"{row['id']}: duplicate id")
        seen.add(row["id"])
        if row["partition"] != "dev":
            raise ValueError(f"{row['id']}: partition is not dev")
        if row["dev_role"] != DEV_ROLE:
            raise ValueError(f"{row['id']}: dev_role is not {DEV_ROLE}")
        if row["heldout_family"] is not False:
            raise ValueError(f"{row['id']}: heldout_family must be false inside the dev partition")
        if row["images"]:
            raise ValueError(f"{row['id']}: this set is text only")
        if not isinstance(row.get("license"), dict):
            raise ValueError(f"{row['id']}: missing license evidence")
        for item in expand_fields(row):
            render(item)
        if row["source"] == AUTHORED_SOURCE_NAME:
            state = row["request"]["state"]
            size = len(state) if isinstance(state, str) else len(json.dumps(state))
            if not 300 <= size <= 2500:
                raise ValueError(f"{row['id']}: state length {size} outside 300-2500")
            if not row.get("rationale"):
                raise ValueError(f"{row['id']}: authored rows need a rationale")
            if (row["target"] is None) != bool(row["abstention_cause"]):
                raise ValueError(f"{row['id']}: unknown target and abstention cause disagree")


def summarise(rows: list[dict]) -> dict:
    authored = [r for r in rows if r["source"] == AUTHORED_SOURCE_NAME]
    types = Counter(r["request"]["fields"][0]["type"] for r in authored)
    states = Counter("object" if isinstance(r["request"]["state"], dict) else "string" for r in authored)
    unknown = sum(1 for r in authored if r["target"] is None)
    return {
        "output": str(OUTPUT.relative_to(ROOT)),
        "records": len(rows),
        "decisions": sum(len(r["request"]["fields"]) for r in rows),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "authored": {
            "records": len(authored),
            "by_family": dict(sorted(Counter(r["family"] for r in authored).items())),
            "by_type": dict(sorted(types.items())),
            "by_domain": dict(sorted(Counter(r["domain"] for r in authored).items())),
            "state_form": dict(sorted(states.items())),
            "unknown": unknown,
            "unknown_share": round(unknown / len(authored), 4) if authored else 0.0,
        },
    }


def build() -> list[dict]:
    rows = typed_rows() + authored_rows()
    check(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = build()
    write_jsonl(args.output, rows)
    print(json.dumps(summarise(rows), indent=2))


if __name__ == "__main__":
    main()
