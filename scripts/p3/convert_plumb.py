"""crh225/plumb-decisions (Apache-2.0, https://huggingface.co/datasets/crh225/plumb-decisions) *train* -> phase-3 candidates.

Only data/train.jsonl (5,014 questions) is read; data/test.jsonl (131) is never downloaded or used. Every question is its own
candidate with the full document as state. The dataset's own ``criteria`` define the options (choice: key -> description;
score: ordered level descriptions; noul: what true and false mean, which is written into the question because some items
define ``true`` as the negation of the question's wording). ``teacher_probs`` (two thinking solutions, or the exact
distribution for probability questions) is kept in provenance for soft targets.

    .venv/bin/python scripts/p3/convert_plumb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from candidate import write  # noqa: E402
from convert_common import MAX_STATE_TOKENS, approx_tokens, option_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/plumb/train.jsonl"
OUT = ROOT / "data/p3/candidates/B-plumb.jsonl"
URL = "https://huggingface.co/datasets/crh225/plumb-decisions"
LICENCE = "Apache-2.0"
FAMILY = {"judge": "judge_hard", "routing": "routing_hard"}   # our JevBench-hard family names; the rest are identical
LENGTH_DIFFICULTY = {"medium": 3, "long": 4, "very": 5}


def difficulty(length: str | None) -> int:
    first = (length or "").split(" ")[0]
    return LENGTH_DIFFICULTY.get(first, 3)


def convert_row(r: dict, idx: int) -> tuple[dict | None, str]:
    q = r["question"]
    t = q["type"]
    crit = q.get("criteria")
    exp = r["expected"]
    text = q["instructions"].strip()
    if t == "choice":
        if not isinstance(crit, dict) or exp not in crit:
            return None, "bad_choice"
        taken: set[str] = set()
        keymap = {k: option_key(k, taken) for k in crit}
        field = {"type": "choice", "question": text,
                 "options": [{"key": keymap[k], "text": k.replace("_", " "), "description": v} for k, v in crit.items()]}
        gold = keymap[exp]
        probs = {keymap[k]: v for k, v in (r.get("teacher_probs") or {}).items() if k in keymap}
    elif t == "noul":
        if exp not in ("true", "false"):
            return None, "bad_noul"
        if isinstance(crit, dict) and crit.get("true") and crit.get("false"):
            text = f"{text}\nAnswer true if: {crit['true'].strip()}\nAnswer false if: {crit['false'].strip()}"
        field = {"type": "noul", "question": text}
        gold = exp == "true"
        probs = dict(r.get("teacher_probs") or {})
    elif t == "score":
        if not isinstance(crit, list) or not str(exp).isdigit() or not 0 <= int(exp) < len(crit):
            return None, "bad_score"
        field = {"type": "score", "question": text, "levels": [{"value": i, "description": d} for i, d in enumerate(crit)]}
        gold = int(exp)
        probs = dict(r.get("teacher_probs") or {})
    else:
        return None, "bad_type"
    if approx_tokens(r["state"]) > MAX_STATE_TOKENS:
        return None, "too_long"
    return {"id": f"p3-plumb-{idx:06d}", "source": "B", "dataset": "plumb", "family": FAMILY.get(r["family"], r["family"]),
            "difficulty": difficulty(r.get("length")), "state": r["state"], "images": [], "field": field, "gold": gold,
            "unknown_reason": None, "gold_kind": "dataset", "parent_id": None,
            "provenance": {"licence": LICENCE, "upstream_dataset": "plumb", "upstream_split": "train", "upstream_id": r["id"],
                           "url": URL, "group_id": r["id"].rsplit("-q", 1)[0], "domain": r.get("domain"),
                           "length": r.get("length"), "upstream_family": r["family"], "teacher_probs": probs,
                           "explanation": r.get("explanation"), "author": "Qwen3.8-27B (open weights, Apache-2.0)"}}, "ok"


def convert(raw: Path = RAW, out: Path = OUT) -> dict:
    assert "test" not in raw.name, "plumb test split is never used"
    rows, why = [], {}
    for line in open(raw):
        r = json.loads(line)
        if not str(r.get("id", "")).startswith("train-"):
            why["not_train"] = why.get("not_train", 0) + 1
            continue
        row, reason = convert_row(r, len(rows))
        why[reason] = why.get(reason, 0) + 1
        if row:
            rows.append(row)
    n = write(out, rows)
    return {"written": n, "reasons": why}


if __name__ == "__main__":
    print(json.dumps(convert(), indent=1))
