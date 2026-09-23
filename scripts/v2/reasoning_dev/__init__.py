"""Hand-authored reasoning items for the decision-v2 checkpoint-selection dev set.

Original work (CC0-1.0). Nothing here is derived from JevBench or any other public
benchmark: the families are named after the JevBench *hard* family names only so the
dev signal covers the same kinds of reasoning, and every state, question, option set
and label was written for this repository.

Each item is a plain dict:

    {"key": "lp01", "domain": "retail", "state": {...} | "...",
     "field": {"id": ..., "type": "boolean|choice|ordinal", "question": ..., ...},
     "target": True | False | "option value" | 3 | None,
     "abstention_cause": "insufficient_evidence" | "not_listed" | None,
     "rationale": "one sentence for the human reviewer"}
"""
from __future__ import annotations

from . import (ambiguous, long_policy, multi_hop, probability, routing,
               temporal_numeric, tradeoff, trap)

MODULES = {
    "long_policy": long_policy,
    "multi_hop": multi_hop,
    "temporal_numeric": temporal_numeric,
    "ambiguous": ambiguous,
    "tradeoff": tradeoff,
    "trap": trap,
    "probability": probability,
    "routing": routing,
}


def all_items() -> list[dict]:
    out = []
    for family, module in MODULES.items():
        for item in module.ITEMS:
            out.append(dict(item, family=family))
    return out
