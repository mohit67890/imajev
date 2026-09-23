"""Generate the reasoning-dev README and the human review sheet from the built manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/decision-v2-reasoning-dev.jsonl"
OUT_DIR = ROOT / "data/decision-v2/reasoning-dev"

FAMILY_ORDER = ["long_policy", "multi_hop", "temporal_numeric", "ambiguous", "tradeoff", "trap",
                "probability", "routing"]
FAMILY_BLURB = {
    "long_policy": "a policy of 8 to 15 numbered rules in the state, one case, and a permission or outcome question",
    "multi_hop": "two or three facts in the state that must be chained before the answer follows",
    "temporal_numeric": "dates, durations, counts and thresholds that need arithmetic or ordering",
    "ambiguous": "the state underdetermines the question, so the honest answer is unknown",
    "tradeoff": "pick the option that best satisfies the stated weighted priorities",
    "trap": "a salient distractor fact that must be ignored",
    "probability": "read an ordinal likelihood level off the stated evidence",
    "routing": "choose the right handler from 5 to 8 options, one of which is 'other'",
}


def answer_text(row: dict) -> str:
    field = row["request"]["fields"][0]
    if row["target"] is None:
        return "unknown"
    if field["type"] == "boolean":
        return "yes" if row["target"] else "no"
    if field["type"] == "ordinal":
        description = next((l["description"] for l in field["levels"] if l["value"] == row["target"]), "")
        return f"{row['target']} ({description})"
    return str(row["target"])


def options_text(row: dict) -> str:
    field = row["request"]["fields"][0]
    if field["type"] == "boolean":
        return "yes / no / unknown"
    if field["type"] == "ordinal":
        return " / ".join(str(l["value"]) for l in field["levels"]) + " / unknown"
    return " / ".join(o["value"] for o in field["options"]) + " / unknown"


def readme(rows: list[dict]) -> str:
    authored = [r for r in rows if r["source"] == "reasoning_authored"]
    typed = [r for r in rows if r["source"] == "typed_decisions_devsel"]
    by_family = defaultdict(list)
    for row in authored:
        by_family[row["family"]].append(row)
    types = Counter(r["request"]["fields"][0]["type"] for r in authored)
    states = Counter("object" if isinstance(r["request"]["state"], dict) else "string" for r in authored)
    domains = Counter(r["domain"] for r in authored)
    unknown = sum(1 for r in authored if r["target"] is None)

    lines = [
        "# decision-v2 reasoning dev set (checkpoint selection only)",
        "",
        "`data/manifests/decision-v2-reasoning-dev.jsonl` -- a human-labelled, text-only development set.",
        "It exists for one job: the second dev signal in the v2 training recipe",
        "(`docs/decision-v2-pseudolabel-spec.md`, *Training recipe for v2*, step 3), where the best checkpoint is",
        "the best mean of the usual dev loss and accuracy on a reasoning-style held-out set.",
        "",
        "**Never train on this file.** Every row carries `partition: \"dev\"` and `dev_role: \"reasoning\"`.",
        "Nothing here is pseudo-labelled: `pseudo_label` is `null` on every row and `label_source` says where the",
        "label came from.",
        "",
        f"| | records | decisions |",
        "|---|---:|---:|",
        f"| `typed_decisions_devsel` | {len(typed)} | {sum(len(r['request']['fields']) for r in typed)} |",
        f"| `reasoning_authored` | {len(authored)} | {len(authored)} |",
        f"| **total** | **{len(rows)}** | **{sum(len(r['request']['fields']) for r in rows)}** |",
        "",
        "## `typed_decisions_devsel` -- typed-decisions TRAIN split, for model selection only",
        "",
        "The 1,200 rows of `data/decision-v1-text/raw/typed-decisions/train.parquet` (5 questions each), converted",
        "with the existing `scripts/v1_text/convert_typed_decisions.py` logic and then renamed to",
        "`typed_decisions_devsel` and pinned to `partition: \"dev\"`.",
        "",
        "This split is used for **model selection only**. It is not training data, it is not an accuracy claim",
        "against humans, and it is kept apart from two things it must never be confused with: the v1.1 training",
        "rows, and the evaluation-only `typed_decisions_test` panel in",
        "`data/manifests/decision-v1.1-heldout-text.jsonl`. Licence: Apache-2.0 on the strength of the dataset card",
        "already retained as evidence at `data/decision-v1-text/licenses/typed_decisions/README.md` (+ receipt).",
        "That receipt's scope is `evaluation_only`; checkpoint selection is an evaluation use, not a training one.",
        "",
        "## `reasoning_authored` -- 240 original decisions",
        "",
        "Written for this repository, by hand, in the same record schema. No JevBench file was opened to make",
        "them (`.cache/external/jevbench/datasets` was never read); only the public *names* of the hard families",
        "are reused, so that the dev signal covers the same kinds of reasoning the v1.1 model lost.",
        "",
        "Licence: **CC0-1.0**, original work. Evidence `data/decision-v2/licenses/reasoning_authored/CC0.txt`",
        "(the CC0 1.0 legal code fetched from creativecommons.org) with `CC0.txt.receipt.json` beside it, so the",
        "same `verified_license` check that guards the collected v2 sources also guards this one.",
        "",
        "### Counts by family",
        "",
        "| family | records | boolean | choice | ordinal | unknown | what it tests |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for family in FAMILY_ORDER:
        group = by_family[family]
        t = Counter(r["request"]["fields"][0]["type"] for r in group)
        u = sum(1 for r in group if r["target"] is None)
        lines.append(f"| `{family}` | {len(group)} | {t['boolean']} | {t['choice']} | {t['ordinal']} | {u} | "
                     f"{FAMILY_BLURB[family]} |")
    lines += [
        f"| **total** | **{len(authored)}** | **{types['boolean']}** | **{types['choice']}** | "
        f"**{types['ordinal']}** | **{unknown}** | |",
        "",
        "### Counts by decision type and state form",
        "",
        "| | count | share |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(types.items()):
        lines.append(f"| type `{name}` | {count} | {count / len(authored):.1%} |")
    for name, count in sorted(states.items()):
        lines.append(f"| state as JSON {name} | {count} | {count / len(authored):.1%} |")
    lines.append(f"| honest answer is `unknown` | {unknown} | {unknown / len(authored):.1%} |")
    lines += [
        "",
        "### Domains",
        "",
        "| domain | records |",
        "|---|---:|",
    ]
    for name, count in sorted(domains.items()):
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "State lengths are between 300 and 2,500 characters. Choice questions carry 3 to 8 options; ordinal",
        "questions carry 3 to 5 levels; the reserved `unknown` is added by the loader and is never listed.",
        "",
        "## How to review the authored items",
        "",
        "`authored_review.md` in this directory lists every authored item with its id, family, domain, question,",
        "the option set, the label and a one-sentence rationale. To review:",
        "",
        "1. Read the rationale **after** deciding the answer yourself from the state, not before. The rationale is",
        "   the author's derivation, so reading it first tells you nothing about whether the item is sound.",
        "2. Open the full state for anything you disagree with:",
        "   `PYTHONPATH=src:scripts .venv/bin/python -c \"import json;[print(json.dumps(r,indent=2)) for r in",
        "   map(json.loads, open('data/manifests/decision-v2-reasoning-dev.jsonl')) if r['id']=='<id>']\"`",
        "3. Flag an item if any of these is true: two options are defensible; the label depends on a fact the state",
        "   does not give (unless the family is `ambiguous`, where that is the point); the rationale does not match",
        "   the state; arithmetic is wrong; or the distractor in a `trap` item is doing real work rather than being",
        "   a distractor.",
        "4. Fix items in `scripts/v2/reasoning_dev/<family>.py`, then rebuild and re-audit:",
        "",
        "```sh",
        "PYTHONPATH=src:scripts .venv/bin/python -m v2.build_reasoning_dev",
        "PYTHONPATH=src:scripts .venv/bin/python -m v2.write_reasoning_dev_docs",
        "PYTHONPATH=src:scripts .venv/bin/python -m v1_text.audit_mixture \\",
        "    data/manifests/decision-v2-reasoning-dev.jsonl",
        "PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v2_reasoning_dev.py -q",
        "```",
        "",
        "## `heldout_family` is false, on purpose",
        "",
        "`scripts/v1_text/audit_mixture.py` fails any row with `heldout_family: true` that is not in the `test`",
        "partition. These rows must stay in `dev`, because the trainer reads them as a dev signal every 100 steps,",
        "so `heldout_family` is `false` and the held-out property is carried by `dev_role: \"reasoning\"` and by the",
        "source names instead. Nothing in this file is in any training mixture.",
        "",
        "## Build",
        "",
        "```sh",
        "PYTHONPATH=src:scripts .venv/bin/python -m v2.build_reasoning_dev",
        "```",
        "",
        "Sources: `scripts/v2/build_reasoning_dev.py` and `scripts/v2/reasoning_dev/`.",
        "",
    ]
    return "\n".join(lines)


def review_sheet(rows: list[dict]) -> str:
    authored = [r for r in rows if r["source"] == "reasoning_authored"]
    by_family = defaultdict(list)
    for row in authored:
        by_family[row["family"]].append(row)
    lines = [
        "# Review sheet: `reasoning_authored` (240 items)",
        "",
        "Every hand-written item in `data/manifests/decision-v2-reasoning-dev.jsonl`, for a human pass.",
        "Decide the answer from the state first; the rationale is the author's own derivation and is here to be",
        "checked, not to be trusted. Review guidance is in `README.md` in this directory.",
        "",
        "States are abbreviated out of this sheet; read the full state from the manifest by id.",
        "",
    ]
    for family in FAMILY_ORDER:
        group = by_family[family]
        lines += [f"## `{family}` ({len(group)} items)", ""]
        for row in group:
            field = row["request"]["fields"][0]
            lines += [
                f"### `{row['id']}`",
                "",
                f"- **domain**: {row['domain']} &nbsp;&nbsp; **type**: {field['type']}"
                f" &nbsp;&nbsp; **state**: {'JSON object' if isinstance(row['request']['state'], dict) else 'string'}"
                f", {len(row['request']['state']) if isinstance(row['request']['state'], str) else len(json.dumps(row['request']['state']))} chars",
                f"- **question**: {field['question']}",
                f"- **options**: {options_text(row)}",
                f"- **label**: **{answer_text(row)}**"
                + (f"  (abstention cause: `{row['abstention_cause']}`)" if row["abstention_cause"] else ""),
                f"- **rationale**: {row['rationale']}",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.md").write_text(readme(rows))
    (args.out_dir / "authored_review.md").write_text(review_sheet(rows))
    print(json.dumps({"readme": str((args.out_dir / "README.md").relative_to(ROOT)),
                      "review": str((args.out_dir / "authored_review.md").relative_to(ROOT)),
                      "items_listed": sum(1 for r in rows if r["source"] == "reasoning_authored")}, indent=2))


if __name__ == "__main__":
    main()
