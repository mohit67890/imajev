"""Build a bounded, subject-stratified MMLU evaluation set (never training data)."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from v1_text.common import verified_license, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decision-v1-text"
RAW = DATA / "raw/mmlu/extracted/data/test"
OUTPUT = DATA / "converted/mmlu-heldout.jsonl"
REPORT = ROOT / "reports/v1.1-datasets/mmlu-heldout-conversion.json"


def rank(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def evaluation_license() -> dict:
    registry = json.loads((DATA / "source-registry.json").read_text())
    entry = next((x for x in registry["sources"] if x["source"] == "mmlu"), None)
    if entry is None:
        raise ValueError("MMLU is absent from the source registry")
    if entry.get("training_admitted") or entry.get("training_approved"):
        raise ValueError("MMLU must remain excluded from training")
    blocker = str(entry.get("training_blocker", "")).lower()
    if "held-out evaluation" not in blocker:
        raise ValueError("registry does not explicitly identify MMLU as held-out evaluation data")
    license_meta = entry.get("license", {})
    if license_meta.get("status") != "verified_primary_license" or license_meta.get("spdx") != "MIT":
        raise ValueError("MMLU lacks verified primary MIT license metadata")
    return verified_license(ROOT / license_meta["path"], "MIT")


def load_subjects() -> dict[str, list[list[str]]]:
    subjects = {}
    for path in sorted(RAW.glob("*_test.csv")):
        subject = path.name.removesuffix("_test.csv")
        with path.open(newline="") as stream:
            rows = [row for row in csv.reader(stream) if row]
        for row in rows:
            if len(row) != 6 or row[5] not in "ABCD":
                raise ValueError(f"malformed MMLU row in {path}")
        subjects[subject] = rows
    if not subjects:
        raise ValueError(f"no MMLU test files found under {RAW}")
    return subjects


def select_stratified(subjects: dict[str, list[list[str]]], limit: int = 1000):
    names = sorted(subjects)
    if limit < len(names):
        raise ValueError("limit must permit at least one example per subject")
    ordered = {s: sorted(subjects[s], key=lambda r: rank(s, *r)) for s in names}
    chosen = []
    # Round-robin selection keeps subject counts within one while data remains.
    depth = 0
    while len(chosen) < limit:
        added = False
        for subject in names:
            if depth < len(ordered[subject]) and len(chosen) < limit:
                chosen.append((subject, ordered[subject][depth])); added = True
        if not added:
            break
        depth += 1
    return chosen


def convert(limit: int = 1000) -> list[dict]:
    license_info = evaluation_license()
    rows = []
    for subject, raw in select_stratified(load_subjects(), limit):
        question, *tail = raw
        choices, answer_letter = tail[:4], tail[4]
        # The public contract caps option values at 128 characters and requires
        # uniqueness. MMLU answers may be long or duplicated, so use the official
        # answer letters as values and retain the verbatim answer text as description.
        answer = answer_letter
        digest = rank(subject, *raw)
        rows.append({
            "id": f"mmlu:test:{subject}:{digest[:16]}",
            "source": "mmlu",
            "source_split": "test",
            "source_group": f"mmlu:{digest}",
            "family": "mmlu_heldout",
            "heldout_family": True,
            "subject": subject,
            "license": license_info,
            "images": [],
            "request": {"schema_version": "1.0", "request_id": f"mmlu-{digest[:16]}",
                        "state": question,
                        "fields": [{"id": "answer", "type": "choice",
                                    "question": "Select the correct answer.",
                                    "options": [{"value": letter, "description": value}
                                                for letter, value in zip("ABCD", choices)]}]},
            "target": answer,
            "abstention_cause": None,
            "partition": "test",
            "source_answer": answer_letter,
        })
    return rows


def main() -> None:
    rows = convert()
    write_jsonl(OUTPUT, rows)
    counts = Counter(r["subject"] for r in rows)
    report = {"output": str(OUTPUT.relative_to(ROOT)), "records": len(rows),
              "partition": "test", "training_records": 0,
              "subjects": len(counts), "min_per_subject": min(counts.values()),
              "max_per_subject": max(counts.values()), "by_subject": dict(sorted(counts.items())),
              "selection": "deterministic round-robin over hash-ranked official test rows",
              "admission": "evaluation_only"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
