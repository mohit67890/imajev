"""Convert approved screening and rubric datasets into decision-v1-text records."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import itertools
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

try:
    from .common import fit_partition, verified_license, write_jsonl
except ImportError:  # direct script execution
    from common import fit_partition, verified_license, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/decision-v1-text/raw"
LICENSES = ROOT / "data/decision-v1-text/licenses"
OUTPUT = ROOT / "data/decision-v1-text/converted/screening-rubric.jsonl"
REPORT = ROOT / "reports/v1.1-datasets/screening-rubric-conversion.json"

PHRASINGS = {
    "toxicity": [
        "Does this text contain toxic language?", "Should this text be flagged as toxic?",
        "Is the text toxic to readers?", "Does the text meet the toxicity criterion?",
        "Would moderators classify this text as toxic?", "Is toxicity present in this text?",
    ],
    "safety": [
        "Does this text require content-safety review?", "Should a safety moderator review this text?",
        "Is this text unsafe or caution-worthy?", "Does the text trigger a content-safety concern?",
        "Would this content need a safety intervention?", "Should this text be flagged for safety review?",
    ],
    "category": [
        "Does this text contain {label}?", "Should this text be flagged for {label}?",
        "Is {label} present in this text?", "Does the text match the {label} category?",
        "Would a moderator label this as {label}?", "Is this an instance of {label}?",
    ],
    "rubric": [
        "Rate the response's {label} using the rubric.", "What {label} score does this response earn?",
        "Score this response for {label}.", "How strong is the response's {label}?",
        "Choose the rubric level for response {label}.", "Assess the response's {label} from 0 to 4.",
    ],
}

CIVIL_ATTRIBUTES = {
    "severe_toxicity": "severe toxicity", "obscene": "obscene content", "threat": "threats",
    "insult": "insults", "identity_attack": "identity attacks", "sexual_explicit": "sexually explicit content",
}
OPENAI_LABELS = {
    "S": "sexual content", "H": "hate", "V": "violence", "HR": "harassment",
    "SH": "self-harm", "S3": "sexual content involving minors", "H2": "threatening hate",
    "V2": "graphic violence",
}
RUBRICS = {
    "helpfulness": ["not helpful", "slightly helpful", "moderately helpful", "very helpful", "fully helpful"],
    "correctness": ["incorrect", "mostly incorrect", "partly correct", "mostly correct", "fully correct"],
    "coherence": ["incoherent", "mostly incoherent", "partly coherent", "mostly coherent", "fully coherent"],
    "complexity": ["far too simple", "somewhat too simple", "appropriate complexity", "somewhat too complex", "far too complex"],
}


def rank(value: str) -> str:
    return hashlib.sha256(("screening-rubric-v1\0" + value).encode()).hexdigest()


def phrase(kind: str, group: str, field: str, label: str | None = None) -> str:
    choices = PHRASINGS[kind]
    text = choices[int(rank(f"{group}:{field}")[:8], 16) % len(choices)]
    return text.format(label=label) if label else text


def clipped(value: str, limit: int = 24000) -> str:
    data = str(value).encode("utf-8")
    return data[:limit].decode("utf-8", errors="ignore") if len(data) > limit else str(value)


def boolean_field(field_id: str, question: str, yes: str, no: str) -> dict:
    return {"id": field_id, "type": "boolean", "question": question,
            "yes_description": yes, "no_description": no}


def record(source: str, source_split: str, group: str, family: str, state, fields: list[dict],
           targets: dict, distributions: dict, license_info: dict, partition: str,
           abstention_causes: dict | None = None) -> dict:
    rid = f"{source}:{group}"
    return {"id": rid, "source": source, "source_split": source_split, "source_group": group,
            "family": family, "heldout_family": False, "license": license_info, "images": [],
            "request": {"schema_version": "1.0", "request_id": rid[:128], "state": state, "fields": fields,
                        "execution": {"mode": "inspect", "allow_external_fallback": False}},
            "targets": targets, "target_distributions": distributions,
            "abstention_causes": abstention_causes or {},
            "partition": partition, "source_answer": targets}


def selected(rows: Iterable[dict], limit: int) -> list[dict]:
    return heapq.nsmallest(limit, rows, key=lambda row: rank(str(row["_group"])))


def unique_groups(rows: Iterable[dict], excluded: set[str]):
    seen = set(excluded)
    for row in rows:
        group = row["_group"]
        if group not in seen:
            seen.add(group)
            yield row


def parquet_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def civil_source_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            row["_group"] = hashlib.sha256(str(row["text"]).encode()).hexdigest()[:24]
            yield row


def civil_comments() -> list[dict]:
    license_info = verified_license(LICENSES / "civil_comments/LICENSE_LEGALCODE.txt", "CC0-1.0")
    paths = sorted((RAW / "civil_comments/data/data").glob("*.parquet"))
    output = []
    by_split = {split: [path for path in paths if path.name.startswith(split + "-")]
                for split in ("train", "validation", "test")}
    used_groups: set[str] = set()
    # Held-out source splits take precedence when upstream contains duplicate text.
    for official in ("test", "validation", "train"):
        split_paths = by_split[official]
        limit = 14500 if official == "train" else 1000
        raw = itertools.chain.from_iterable(civil_source_rows(path) for path in split_paths)
        chosen = selected(unique_groups(raw, used_groups), limit)
        used_groups.update(row["_group"] for row in chosen)
        for row in chosen:
            group = row["_group"]
            partition = fit_partition(group) if official == "train" else "dev" if official == "validation" else "test"
            secondary = sorted(CIVIL_ATTRIBUTES)[int(rank(group)[:8], 16) % len(CIVIL_ATTRIBUTES)]
            fields, targets, distributions = [], {}, {}
            for field_id, label in (("toxicity", "toxicity"), (secondary, CIVIL_ATTRIBUTES[secondary])):
                probability = min(1.0, max(0.0, float(row[field_id])))
                fields.append(boolean_field(field_id, phrase("toxicity" if field_id == "toxicity" else "category", group, field_id, label),
                                            f"annotators identify {label}", f"annotators do not identify {label}"))
                targets[field_id] = probability >= 0.5
                distributions[field_id] = {"true": probability, "false": 1.0 - probability}
            output.append(record("civil_comments", official, group, "toxicity_screening", clipped(row["text"]),
                                 fields, targets, distributions, license_info, partition))
    return output


def aegis() -> list[dict]:
    license_info = verified_license(LICENSES / "nvidia_aegis/LICENSE_LEGALCODE.txt", "CC-BY-4.0")
    output = []
    paths = sorted((RAW / "nvidia_aegis/data").glob("*.parquet"), key=lambda path: "_test_" not in path.name)
    used_groups = set()
    for path in paths:
        official = "test" if "_test_" in path.name else "train"
        for row in parquet_rows(path):
            group = str(row["id"])
            if group in used_groups:
                continue
            used_groups.add(group)
            labels = [row[f"labels_{i}"] for i in range(5) if row.get(f"labels_{i}")]
            unsafe_votes = sum(label != "Safe" for label in labels)
            probability = unsafe_votes / len(labels)
            field = boolean_field("needs_safety_review", phrase("safety", group, "needs_safety_review"),
                                  "unsafe or needs caution", "safe")
            partition = fit_partition(group) if official == "train" else "test"
            output.append(record("nvidia_aegis", official, group, "content_safety", clipped(row["text"]),
                                 [field], {"needs_safety_review": probability >= 0.5},
                                 {"needs_safety_review": {"true": probability, "false": 1 - probability}},
                                 license_info, partition))
    return output


def openai_moderation() -> list[dict]:
    license_info = verified_license(LICENSES / "openai_moderation/LICENSE", "MIT")
    path = RAW / "openai_moderation/data/samples-1680.jsonl.gz"
    output = []
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seen_groups = set()
    for index, row in enumerate(rows):
        group = hashlib.sha256(row["prompt"].encode()).hexdigest()[:24]
        if group in seen_groups:
            continue
        seen_groups.add(group)
        fields, targets = [], {}
        for code, label in OPENAI_LABELS.items():
            if code not in row:
                continue
            fields.append(boolean_field(code.lower(), phrase("category", group, code, label),
                                        f"the text contains {label}", f"the text does not contain {label}"))
            targets[code.lower()] = bool(row[code])
        output.append(record("openai_moderation", "evaluation_release", group, "moderation_taxonomy",
                             clipped(row["prompt"]), fields, targets, {}, license_info, fit_partition(group)))
    return output


def helpsteer2() -> list[dict]:
    license_info = verified_license(LICENSES / "helpsteer2/LICENSE_LEGALCODE.txt", "CC-BY-4.0")
    output = []
    used_groups = set()
    for official in ("validation", "train"):
        path = RAW / f"helpsteer2/data/{official}.jsonl.gz"
        with gzip.open(path, "rt") as handle:
            raw = [json.loads(line) for line in handle if line.strip()]
        for row in raw:
            row["_group"] = hashlib.sha256((row["prompt"] + "\0" + row["response"]).encode()).hexdigest()[:24]
        raw = [row for row in raw if row["_group"] not in used_groups]
        if official == "train":
            raw = selected(unique_groups(raw, used_groups), 5000)
        used_groups.update(row["_group"] for row in raw)
        for row in raw:
            group = row["_group"]
            fields, targets = [], {}
            for attribute, descriptions in RUBRICS.items():
                fields.append({"id": attribute, "type": "ordinal",
                               "question": phrase("rubric", group, attribute, attribute),
                               "levels": [{"value": value, "description": description}
                                          for value, description in enumerate(descriptions)]})
                targets[attribute] = int(row[attribute])
            state = {"prompt": clipped(row["prompt"], 8000), "response": clipped(row["response"], 22000)}
            partition = fit_partition(group) if official == "train" else "dev"
            output.append(record("helpsteer2", official, group, "response_rubric", state, fields, targets, {},
                                 license_info, partition))
    return output


def build() -> list[dict]:
    return civil_comments() + aegis() + openai_moderation() + helpsteer2()


def conversion_report(rows: list[dict]) -> dict:
    decisions = Counter()
    records = Counter()
    families = Counter()
    phrase_counts = Counter()
    for row in rows:
        key = (row["source"], row["partition"])
        records[key] += 1
        decisions[key] += len(row["request"]["fields"])
        families[row["family"]] += len(row["request"]["fields"])
        for field in row["request"]["fields"]:
            phrase_counts[(row["source"], field["question"])] += 1
    return {"schema_version": "1.0", "records": len(rows),
            "decisions": sum(len(row["request"]["fields"]) for row in rows),
            "records_by_source_partition": {f"{s}/{p}": n for (s, p), n in sorted(records.items())},
            "decisions_by_source_partition": {f"{s}/{p}": n for (s, p), n in sorted(decisions.items())},
            "decisions_by_family": dict(sorted(families.items())),
            "distinct_question_phrasings_by_source": {source: len({question for (s, question) in phrase_counts if s == source})
                                                       for source in sorted({s for s, _ in phrase_counts})},
            "train_decisions": sum(n for (source, partition), n in decisions.items() if partition == "train"),
            "output": str(OUTPUT.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    rows = build()
    write_jsonl(args.output, rows)
    report = conversion_report(rows)
    report["output"] = str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": report["records"], "decisions": report["decisions"],
                      "train_decisions": report["train_decisions"], "output": report["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
