"""Phase-3 candidate item: one typed decision question with a known or checkable answer (docs/phase-3-plan.md, Stage 0).

Every Stage-0 source (programmatic generators, public-dataset converters, our own pools, trap items, the image branch)
writes JSONL rows of this shape. Mining (Stage 1), the teacher (Stage 2) and the manifest builder (Stage 3) read only this.

    {"id": "p3-finqa-000123",               # unique across the whole pool; stable across re-runs
     "source": "B",                          # A generators | B public | C own pools | D traps | I image branch
     "dataset": "finqa",                     # generator family name or public dataset id
     "family": "table_arithmetic",           # decision family used for balancing and reporting
     "difficulty": 4,                        # 1-5; generators set it, converters estimate it (hops, steps, depth)
     "state": "<text>" | {...},              # what the model reads; images referenced in "images"
     "images": [],                           # relative paths under data/ (image branch only)
     "field": {"type": "choice" | "noul" | "score", "question": "...",
               "options": [{"key": "...", "text": "...", "description": "..."}],   # choice: 2-255 options (255 = 256-code lane only)
               "levels": [{"value": 0, "description": "..."}]},                     # score: 2-10 levels, 0-based increasing
     "gold": "<option key>" | true | false | <level int> | null,   # null = the answer is unknown (unknown_reason set)
     "unknown_reason": null | "insufficient_evidence" | "false_premise" | "not_listed" | "mismatched_reference",
     "gold_kind": "constructed" | "dataset" | "none",   # constructed = provably correct by the generator
     "parent_id": null | "<id>",             # variants and unknown/trap variants point at their parent (split inheritance)
     "provenance": {"licence": "MIT", "upstream_split": "train", "upstream_id": "...", "url": "..."}}
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCES = ("A", "B", "C", "D", "I")
TYPES = ("choice", "noul", "score")
UNKNOWN_REASONS = ("insufficient_evidence", "false_premise", "not_listed", "mismatched_reference")
GOLD_KINDS = ("constructed", "dataset", "none")
PERMISSIVE_LICENCES = ("MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-2.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0", "CC0-1.0", "BSD-3-Clause",
                       "own", "generated")
# Options + unknown must fit the readout: 254 options on the shipped 255-code readout, 255 on the gated 256-code readout
# (docs/phase-3-plan.md item 5; `--readout-codes 256`). The pool allows 255 so the `large_choice` family can train code 256;
# scripts/p3/build_manifest.py drops 255-option rows from every 255-code lane.
MAX_OPTIONS = 255


def validate(row: dict) -> list[str]:
    """Return a list of problems (empty = valid)."""
    errs = []
    for k in ("id", "source", "dataset", "family", "difficulty", "state", "field", "gold_kind", "provenance"):
        if k not in row:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if row["source"] not in SOURCES:
        errs.append("bad source")
    if not (isinstance(row["difficulty"], int) and 1 <= row["difficulty"] <= 5):
        errs.append("difficulty must be int 1-5")
    if row["gold_kind"] not in GOLD_KINDS:
        errs.append("bad gold_kind")
    lic = (row.get("provenance") or {}).get("licence")
    if lic not in PERMISSIVE_LICENCES:
        errs.append(f"licence {lic!r} not in the permissive allow-list")
    f = row["field"]
    t = f.get("type")
    if t not in TYPES:
        return errs + ["bad field type"]
    if not isinstance(f.get("question"), str) or len(f["question"]) < 8:
        errs.append("question too short")
    gold, unk = row.get("gold"), row.get("unknown_reason")
    if (gold is None) != (unk is not None):
        errs.append("unknown_reason exactly when gold is null")
    if unk is not None and unk not in UNKNOWN_REASONS:
        errs.append("bad unknown_reason")
    if t == "choice":
        opts = f.get("options") or []
        keys = [o.get("key") for o in opts]
        if not 2 <= len(opts) <= MAX_OPTIONS:
            errs.append(f"choice needs 2-{MAX_OPTIONS} options")
        if len(set(keys)) != len(keys) or any(not k or k == "unknown" for k in keys):
            errs.append("option keys must be unique, non-empty, not 'unknown'")
        if gold is not None and gold not in keys:
            errs.append("gold must be an option key")
    elif t == "score":
        vals = [l.get("value") for l in f.get("levels") or []]
        if not 2 <= len(vals) <= 10 or vals != list(range(len(vals))):
            errs.append("score levels must be 0..n-1, 2-10 of them")
        if gold is not None and (isinstance(gold, bool) or gold not in vals):
            errs.append("gold must be a level value")
    else:
        if gold is not None and not isinstance(gold, bool):
            errs.append("noul gold must be a boolean")
    return errs


def read(path: str | Path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            errs = validate(r)
            if errs:
                raise ValueError(f"{r.get('id')}: {errs}")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
