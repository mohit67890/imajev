"""Convert the Stanford Sentiment Treebank test split to 5-level ordinal decisions.

Evaluation only. The retained SST archive ships a README but no license grant, so
this converter refuses to attach a fabricated license: callers either pass verified
evidence explicitly or accept ``license: null`` plus a recorded ``license_note``.

Sentence-level items only. Each test sentence from ``datasetSentences.txt`` is looked
up in ``dictionary.txt`` to recover its phrase id, and its sentiment value is read from
``sentiment_labels.txt``. The 5 fine-grained classes come from the cut-offs the archive
README states: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0].
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import verified_license, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/decision-v1-text"
RAW = DATA / "raw/sst5/extracted/stanfordSentimentTreebank"
OUTPUT = DATA / "converted/sst5-heldout.jsonl"

# README.txt: "you can recover the 5 classes by mapping the positivity probability
# using the following cut-offs: [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
# for very negative, negative, neutral, positive, very positive, respectively."
LEVELS = [
    {"value": 1, "description": "Very negative: the review is harshly critical of the film."},
    {"value": 2, "description": "Negative: the review is unfavourable, without being scathing."},
    {"value": 3, "description": "Neutral: the review is mixed, descriptive, or does not lean either way."},
    {"value": 4, "description": "Positive: the review is favourable, without being effusive."},
    {"value": 5, "description": "Very positive: the review is enthusiastically complimentary."},
]
QUESTION = "How positive is the sentiment this excerpt expresses about the film?"
UNLICENSED_NOTE = (
    "No license grant was found in any retained SST-1.0 file: the archive README.txt "
    "carries only a citation request, and data/decision-v1-text/source-registry.json "
    "records sst5 as quarantined_no_primary_license_file_identified with spdx null. "
    "Evaluation-only; never admit to training or redistribution."
)

SPLIT_CODES = {1: "train", 2: "test", 3: "dev"}


def sentiment_level(value: float) -> int:
    """Map an SST positivity probability onto its 1-5 fine-grained level."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"sentiment value {value} outside [0, 1]")
    if value <= 0.2:
        return 1
    if value <= 0.4:
        return 2
    if value <= 0.6:
        return 3
    if value <= 0.8:
        return 4
    return 5


def repair(text: str) -> str:
    """Undo the latin-1 mis-encoding in datasetSentences.txt where it applies."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def readable(text: str) -> str:
    """Penn-Treebank bracket escapes back to literal brackets for the served state."""
    return (text.replace("-LRB-", "(").replace("-RRB-", ")")
                .replace("-LSB-", "[").replace("-RSB-", "]")
                .replace("-LCB-", "{").replace("-RCB-", "}"))


def _lookup_variants(sentence: str):
    """Dictionary spellings to try, in order, for one datasetSentences.txt line."""
    seen = []
    for candidate in (sentence, repair(sentence)):
        for value in (candidate, readable(candidate)):
            if value not in seen:
                seen.append(value)
    return seen


def load(raw: Path = RAW):
    raw = Path(raw)
    phrase_ids = {}
    for line in (raw / "dictionary.txt").read_bytes().decode("utf-8").split("\n"):
        if not line.strip():
            continue
        phrase, ident = line.rsplit("|", 1)
        phrase_ids[phrase] = int(ident)
    values = {}
    for line in (raw / "sentiment_labels.txt").read_text().strip().split("\n")[1:]:
        ident, value = line.split("|")
        values[int(ident)] = float(value)
    splits = {}
    for line in (raw / "datasetSplit.txt").read_text().strip().split("\n")[1:]:
        index, code = line.split(",")
        splits[int(index)] = SPLIT_CODES[int(code)]
    sentences = {}
    for line in (raw / "datasetSentences.txt").read_bytes().decode("utf-8").split("\n")[1:]:
        if not line.strip():
            continue
        index, sentence = line.split("\t", 1)
        sentences[int(index)] = sentence.strip()
    return sentences, splits, phrase_ids, values


def convert(raw: Path = RAW, split: str = "test", license_evidence: Path | None = None,
            spdx: str | None = None, source_name: str = "sst5") -> list[dict]:
    license_info = None
    if license_evidence is not None:
        if not spdx:
            raise ValueError("license evidence requires an expected SPDX identifier")
        license_info = verified_license(Path(license_evidence), spdx)
    sentences, splits, phrase_ids, values = load(raw)
    rows, unresolved = [], []
    for index in sorted(sentences):
        if splits.get(index) != split:
            continue
        sentence = sentences[index]
        phrase_id = next((phrase_ids[v] for v in _lookup_variants(sentence) if v in phrase_ids), None)
        if phrase_id is None:
            unresolved.append(index)
            continue
        value = values[phrase_id]
        text = readable(repair(sentence))
        record = {
            "id": f"{source_name}:{split}:{index}",
            "source": source_name,
            "source_split": split,
            "source_group": f"{source_name}:sentence:{index}",
            "family": f"{source_name}_sentiment",
            "heldout_family": True,
            "license": license_info,
            "images": [],
            "request": {
                "schema_version": "1.0",
                "request_id": f"{source_name}-{split}-{index}",
                "state": {"review": text},
                "fields": [{"id": "sentiment", "type": "ordinal",
                            "question": QUESTION, "levels": [dict(x) for x in LEVELS]}],
            },
            "target": sentiment_level(value),
            "abstention_cause": None,
            "partition": "test" if split == "test" else ("dev" if split == "dev" else "train"),
            "source_answer": value,
            "sst_phrase_id": phrase_id,
        }
        if license_info is None:
            record["license_note"] = UNLICENSED_NOTE
        rows.append(record)
    if unresolved:
        raise ValueError(f"{len(unresolved)} SST sentences could not be resolved to a phrase id")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--license-evidence", type=Path)
    parser.add_argument("--spdx")
    args = parser.parse_args()
    rows = convert(args.raw, args.split, args.license_evidence, args.spdx)
    write_jsonl(args.output, rows)
    print(json.dumps({"records": len(rows), "output": str(args.output),
                      "licensed": rows[0]["license"] is not None if rows else None,
                      "levels": dict(sorted(Counter(r["target"] for r in rows).items()))}))


if __name__ == "__main__":
    main()
