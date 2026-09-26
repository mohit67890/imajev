"""Tokens per decision for the standard and compact prompt layouts (phase-3 item 14). Tokenizer only, no model.

A decision = one question, one pass (no rotations), text only: the full Qwen chat-rendered prompt up to and including
the decision position, exactly as MLXDirect / TorchDecision feed it.

    .venv/bin/python scripts/p3/measure_prompt_layout.py      # writes reports/phase3/compact-layout.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_readout256 import CANDIDATES, JEVBENCH, JEVSTYLE_DEV, candidate_request  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402
from vision_decision.jev_api import to_request  # noqa: E402
from vision_decision.scoring import PROMPT_LAYOUTS, compile_prompt  # noqa: E402


def items(per_candidate_file=25, seed=0):
    import random
    out = []
    for tier in ("easy", "original", "hard"):
        for row in (json.loads(x) for x in (JEVBENCH / f"{tier}.jsonl").open() if x.strip()):
            out.append((f"jevbench/{tier}", to_request({"state": row["state"], "questions": {"d": row["question"]}})))
    for row in (json.loads(x) for x in JEVSTYLE_DEV.open() if x.strip()):
        out.append(("p2b-jevstyle-dev", Request.model_validate(row["request"])))
    for path in sorted(CANDIDATES.glob("B-*.jsonl")):
        rows = [json.loads(x) for x in path.open() if x.strip()]
        for row in random.Random(f"{seed}:{path.name}").sample(rows, min(per_candidate_file, len(rows))):
            out.append((f"candidates/{path.stem}", candidate_request(row)))
    return out


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(json.loads((ROOT / "artifacts/model-qwen4b-local.json").read_text())["path"],
                                              local_files_only=True)

    def count(prompt):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                 tokenize=False, enable_thinking=False)
        assert rendered.endswith("<think>\n\n</think>\n\n")
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    rows = []
    for group, request in items():
        field = request.fields[0]
        rows.append({"set": group, **{layout: count(compile_prompt(field, request.state, layout)[0]) for layout in PROMPT_LAYOUTS}})

    def summary(selected):
        s, c = [r["standard"] for r in selected], [r["compact"] for r in selected]
        return {"n": len(selected), "standard_mean": statistics.mean(s), "compact_mean": statistics.mean(c),
                "standard_median": statistics.median(s), "compact_median": statistics.median(c),
                "saved_mean": statistics.mean(a - b for a, b in zip(s, c)),
                "reduction": 1 - sum(c) / sum(s)}

    groups = list(dict.fromkeys(r["set"] for r in rows))
    report = {"all": summary(rows), "jevbench_all_tiers": summary([r for r in rows if r["set"].startswith("jevbench/")]),
              "sets": {g: summary([r for r in rows if r["set"] == g]) for g in groups}}
    out = ROOT / "reports/phase3/compact-layout.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name, value in [("all", report["all"]), ("jevbench (all tiers)", report["jevbench_all_tiers"])] + list(report["sets"].items()):
        print(f"{name:28s} n={value['n']:4d}  standard {value['standard_mean']:7.1f}  compact {value['compact_mean']:7.1f}  "
              f"saved {value['saved_mean']:5.1f}  ({value['reduction']:.1%})")


if __name__ == "__main__":
    main()
