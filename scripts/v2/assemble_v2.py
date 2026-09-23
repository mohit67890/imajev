"""Assemble data/manifests/decision-v2.jsonl = every v1.1 row, unchanged, + teacher-labelled v2 rows.

v1.1 rows are copied byte-equivalent (same dicts, re-serialised); nothing under data/decision-v1*
is touched.  v2 rows whose partition is ``test`` become held-out exam rows for the new sources:
they get ``heldout_family: true`` (so audit_mixture refuses to see them outside the test partition)
and ``pseudo_label_test: true`` (so no accuracy claim against human labels is ever made from them).

The assembled manifest is audited with ``v1_text.audit_mixture.audit`` — which verifies the licence
receipts (repo-relative evidence paths), partition/group integrity, image presence, and that every
row, including the soft pseudo-labels, renders — and the report is written to
reports/v2-datasets/mixture-audit.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from v1_text.audit_mixture import audit
from v1_text.common import write_jsonl

ROOT = Path(__file__).resolve().parents[2]
V1 = "data/manifests/decision-v1.1.jsonl"
OUT = "data/manifests/decision-v2.jsonl"
REPORT = "reports/v2-datasets/mixture-audit.json"


def read_limited(path, limit=0, stride=1):
    """Stream a manifest, keeping every ``stride``-th row and at most ``limit`` of them."""
    rows = []
    with Path(path).open() as handle:
        for n, line in enumerate(handle):
            if not line.strip() or n % stride:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def flag_v2(row):
    """Test-partition v2 rows are exam rows for the new sources, never an accuracy claim."""
    out = dict(row)
    if out.get("partition") == "test":
        out["heldout_family"] = True
        out["pseudo_label_test"] = True
    return out


def assemble(v1_rows, v2_rows):
    seen = {r["id"] for r in v1_rows}
    flagged = []
    for row in v2_rows:
        if row["id"] in seen:
            raise ValueError(f"{row['id']}: v2 row collides with a v1.1 id")
        seen.add(row["id"])
        flagged.append(flag_v2(row))
    return list(v1_rows) + flagged


def decisions(rows):
    return sum(len(r["request"]["fields"]) for r in rows)


def regularization(rows):
    """What self-distillation blend, if any, the rows on either side of the mixture carry."""
    alphas = sorted({r["regularizer_alpha"] for r in rows if "regularizer_alpha" in r})
    names = sorted({r["regularizer"] for r in rows if "regularizer" in r})
    return {"regularizer_alpha": (alphas[0] if len(alphas) == 1 else alphas or None),
            "regularizer": (names[0] if len(names) == 1 else names or None),
            "regularized_rows": sum(1 for r in rows if "regularizer_alpha" in r)}


def statistics(v1_rows, v2_rows):
    partitions = collections.Counter(r.get("partition") for r in v2_rows)
    labels = collections.Counter(r.get("pseudo_label") for r in v2_rows)
    sources = collections.Counter(r.get("source") for r in v2_rows)
    unknown = sum(1 for r in v2_rows for value in
                  ([r.get("target")] if len(r["request"]["fields"]) == 1 else list((r.get("targets") or {}).values()))
                  if value is None)
    total = decisions(v2_rows)
    return {"v1_regularization": regularization(v1_rows), "v2_regularization": regularization(v2_rows),
            "v1_rows": len(v1_rows), "v1_decisions": decisions(v1_rows),
            "v2_rows": len(v2_rows), "v2_decisions": total,
            "v2_partitions": dict(sorted(partitions.items(), key=lambda x: str(x[0]))),
            "v2_pseudo_labels": dict(sorted(labels.items(), key=lambda x: str(x[0]))),
            "v2_sources": dict(sorted(sources.items(), key=lambda x: str(x[0]))),
            "v2_heldout_rows": sum(1 for r in v2_rows if r.get("pseudo_label_test")),
            "v2_unknown_decisions": unknown,
            "v2_unknown_share": (unknown / total) if total else 0.0}


def build(labelled, v1_path=V1, output=OUT, report=REPORT, v1_limit=0, v2_limit=0, v1_stride=1):
    os.chdir(ROOT)  # audit_mixture resolves image paths relative to the working directory
    v1_rows = read_limited(v1_path, v1_limit, v1_stride)
    v2_rows = [r for path in labelled for r in read_limited(path)]
    if v2_limit:
        v2_rows = v2_rows[:v2_limit]
    rows = assemble(v1_rows, v2_rows)
    result = audit(rows)
    result.update(statistics(v1_rows, [r for r in rows[len(v1_rows):]]))
    result["inputs"] = {"v1": str(v1_path), "v2": [str(p) for p in labelled],
                        "v1_limit": v1_limit, "v1_stride": v1_stride, "v2_limit": v2_limit}
    result["status"] = "audited" if result["ok"] else "audit_failed"
    destination = Path(output)
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if result["ok"]:
        write_jsonl(destination, rows)
        result["manifest"] = str(destination)
        result["manifest_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        result["records"] = len(rows)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labelled", nargs="+", required=True, help="teacher-labelled jsonl shards")
    p.add_argument("--v1", default=V1,
                   help="v1.1 rows to carry over; point it at a blended manifest from "
                        "pseudo_label.py --blend to build decision-v2 on self-distilled v1.1 rows")
    p.add_argument("--output", default=OUT)
    p.add_argument("--report", default=REPORT)
    p.add_argument("--v1-limit", type=int, default=0, help="smoke builds only: keep at most N v1.1 rows")
    p.add_argument("--v1-stride", type=int, default=1, help="smoke builds only: keep every N-th v1.1 row")
    p.add_argument("--v2-limit", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = build(args.labelled, args.v1, args.output, args.report, args.v1_limit, args.v2_limit, args.v1_stride)
    summary = {k: v for k, v in result.items() if k not in ("counts", "licenses", "errors")}
    summary["error_count"] = len(result["errors"])
    summary["errors"] = result["errors"][:20]
    print(json.dumps(summary, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
