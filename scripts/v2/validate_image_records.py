"""Gate for the decision-v2 IMAGE sources: `python scripts/v2/validate_image_records.py <source>`.

(The text sources have their own gate, `scripts/v2/validate_v2_records.py`, because they carry no
images and may use multi-question requests.)

This is `scripts/v1/validate_records.py` ported to decision-v2, check for check.  Two things had to
change, and nothing else did:

  1. the root directory is `data/decision-v2/`;
  2. a v1 row must satisfy `target is None  <=>  abstention_cause is not None`.  A v2 candidate row
     is unanswered *by construction* -- `target: null`, `abstention_cause: null`,
     `pseudo_label: "pending"` -- so that rule is replaced by: a `pending` row must have a null
     target AND a null abstention cause, and any row whose `pseudo_label` is not `pending` must
     satisfy the original v1 rule.

Because of (2) the v1 script itself cannot pass on spec-shaped v2 rows; running it on this data
reports one error per row, all of them `target None <=> abstention_cause set`.  That is recorded in
each source README rather than papered over.

On top of the v1 checks this one also verifies the licence receipt of every row
(`scripts/v2/common.py:verified_license`) and the extra v2 fields.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image

from v2.common import ROOT, verified_license
from vision_decision.contracts import Request

# v1's five causes plus the one the teacher writes (`scripts/v2/pseudo_label.py`), so this gate can
# also be run over a source after it has been labelled, not only over pending candidates.
CAUSES = {None, "not_listed", "false_premise", "insufficient_evidence", "mismatched_reference",
          "teacher_unknown"}
PARTITIONS = {"train", "dev", "test"}
REQUIRED = ("id", "source", "source_split", "source_group", "family", "license", "images",
            "request", "target", "abstention_cause", "partition", "pseudo_label", "template_id")


def main(source: str) -> int:
    root = ROOT / "data" / "decision-v2" / source
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line.strip()]
    excl = json.loads((ROOT / "data" / "decision-v1" / "exclusions.json").read_text())
    banned = set(excl["sha256"])
    errors, ids, groups, licences = [], set(), {}, {}

    for n, r in enumerate(rows):
        def bad(msg):
            errors.append(f'{r.get("id", n)}: {msg}')

        missing = [k for k in REQUIRED if k not in r]
        if missing:
            bad(f"missing {', '.join(missing)}")
            continue
        if r["id"] in ids:
            bad("duplicate id")
        ids.add(r["id"])
        if r["partition"] not in PARTITIONS:
            bad("bad partition")
        if r["abstention_cause"] not in CAUSES:
            bad("bad abstention_cause")
        if r["pseudo_label"] == "pending":
            if r["target"] is not None or r["abstention_cause"] is not None:
                bad("a pending row must carry target null and abstention_cause null")
            if (r["partition"] == "test") != bool(r.get("pseudo_label_test")):
                bad("test rows must be flagged pseudo_label_test, and only those")
        elif (r["target"] is None) != (r["abstention_cause"] is not None):
            bad("target None <=> abstention_cause set")
        groups.setdefault(r["source_group"], set()).add("test" if r["partition"] == "test" else "fit")

        licence = r["license"]
        if not isinstance(licence, dict):
            bad("license must be the decision-v2 licence object")
        else:
            key = (licence.get("evidence"), licence.get("spdx"))
            if key not in licences:
                try:
                    licences[key] = verified_license(Path(licence["evidence"]), licence["spdx"])
                except Exception as exc:
                    licences[key] = None
                    bad(f"licence: {exc}")
            resolved = licences.get(key)
            if resolved and resolved != licence:
                bad("licence object does not match the verified receipt")

        try:
            req = Request.model_validate(r["request"])
            assert len(req.fields) == 1, "exactly one field per record"
            f = req.fields[0]
            if r["target"] is not None:
                if f.type == "boolean":
                    assert isinstance(r["target"], bool), "boolean target must be true/false"
                elif f.type == "choice":
                    assert r["target"] in [o.value for o in f.options], "target not among options"
                else:
                    assert r["target"] in [l.value for l in f.levels], "target not among levels"
            if f.type == "choice":
                values = [o.value for o in f.options]
                assert len(set(values)) == len(values), "duplicate options"
                assert 2 <= len(values) <= 25, "choice fields carry 2-25 options"
        except Exception as exc:
            bad(f"request/target: {exc}")

        if not 1 <= len(r["images"]) <= 2:
            bad("need 1-2 images")
        for im in r["images"]:
            p = ROOT / im["image"]
            if not p.is_file():
                bad(f"missing image {im['image']}")
                continue
            if im["sha256"] in banned and r["partition"] != "test":
                bad("image was used by an earlier evaluation")
            if n % 200 == 0:  # spot-check bytes and decodability
                if hashlib.sha256(p.read_bytes()).hexdigest() != im["sha256"]:
                    bad("sha256 mismatch")
                with Image.open(p) as x:
                    x.load()
                    if x.width * x.height > 1_050_000:
                        bad("image larger than the 1 MP storage cap")

    leak = [g for g, s in groups.items() if len(s) > 1]
    if leak:
        errors.append(f"{len(leak)} source_groups appear in both test and train/dev, e.g. {leak[:3]}")

    first = lambda r: r["request"]["fields"][0]
    summary = dict(
        source=source, records=len(rows),
        partitions=dict(Counter(r["partition"] for r in rows)),
        families=dict(Counter(r["family"] for r in rows)),
        field_types=dict(Counter(first(r)["type"] for r in rows)),
        pseudo_label=dict(Counter(r["pseudo_label"] for r in rows)),
        abstention=dict(Counter(str(r["abstention_cause"]) for r in rows)),
        option_counts=dict(sorted(Counter(
            len(first(r).get("options", first(r).get("levels", []))) for r in rows).items())),
        two_image=sum(len(r["images"]) == 2 for r in rows),
        unknown_construction=sum(bool(r.get("unknown_construction")) for r in rows),
        state_string=sum(isinstance(r["request"]["state"], str) for r in rows),
        licenses=dict(Counter(r["license"]["spdx"] for r in rows if isinstance(r["license"], dict))),
        distinct_templates=len({r.get("template_id") for r in rows}),
        errors=len(errors),
    )
    (root / "validation.json").write_text(json.dumps(dict(summary=summary, first_errors=errors[:50]), indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        print("\n".join(errors[:20]))
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
