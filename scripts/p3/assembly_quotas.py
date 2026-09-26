"""Per-family streaming quotas for the phase-3 teacher (docs/phase-3-plan.md: "Streamed mining + teacher", reconciliation).

The pool is streamed through mining in family-balanced shards; every flagged item goes to the Azure teacher until its family's
quota is filled. The quotas are an upper bound per family, not a target: a family that flags fewer items simply leaves budget
unused. Rules (all parameters are written to quotas.json):

1. Overall Azure cap ~70k teacher calls (~$130-140 at the measured ~530 solves/$). A call is a first solve, a second solve
   (gold not guaranteed: own pools C and the teacher-written plumb documents need two agreeing thinking solves; constructed
   and public-dataset golds need one solve that must match) or the re-verification of a B/C variant (written by the 27B,
   re-verified under the same keep rule, so two calls when its parent needed two). A/D/I-generated variants are free
   (constructed gold). The first-solve total F is the largest value whose expected load
       F + sum_f q_f * (second_share_f + bc_share_f * keep_rate * bc_variants * (1 + second_share_f))
   fits the cap.
2. F is split text / image (image_share of F): images are labelled within the same cap (the 35B text side; the 9B image
   teacher runs on our own GPU and is not an Azure call).
3. Inside each modality every family takes a RATE of its pool rows: q_f = pool_rows_f * min(expected_flag_rate,
   lambda * w_f), with lambda set so the quotas sum to the modality budget. A weight-3 family therefore accepts its whole
   expected flagged stream (the plan's 25-35% flag estimate, upper end), lighter families a share of it proportional to
   their weight, and a family's quota grows with its size (one big generator family is not squeezed by forty small ones).
   Weights: 3.0 for the three families plumb attacked and our shipped 4B misses most (long_policy,
   judge_hard, temporal_numeric, and the own-pool families of the same kind); 2.0 for the joint image+record rule family
   (the ImajevBench families phase 2c lost) and long-document policy rules; 1.5 for the other weak areas from the
   benchmarks (multi-step numeric, logic, multi-hop, ordinal scoring, probability, agent/GUI actions, routing, traps,
   long inputs, multi-label, grounded image rules); 1.0 default; 0.5 for easy classification (emotion, intent, sentiment,
   NLI, moderation, templated Eikos rules).
4. large_choice (255 options) gets no teacher quota: its gold is constructed, the shipped model cannot even serve 255 options
   (so mining is meaningless), and every row goes straight to the 256-code lanes.
"""
from __future__ import annotations

import collections
import math

VERSION = "p3-quotas-v1"
CAP_TOTAL = 70_000
IMAGE_SHARE = 0.36
FLAG_RATE = 0.35
KEEP_RATE = 0.70
BC_VARIANTS = 1          # B/C variants budgeted per kept parent (the plan allows 2; the cap decides)

PRIORITY = {  # weight 3.0
    "long_policy", "judge_hard", "temporal_numeric", "eikos.temporal_numeric", "temporal_arithmetic", "temporal_ordering",
    "date_number_trap", "eikos.judge", "judge_answer", "response_rubric", "adequacy_of_answer", "rubric",
}
W2 = {"image_joint_rule", "policy_exception", "rule_precedence"}
W15 = {
    "table_arithmetic", "table_text_arithmetic", "table_arithmetic_yesno", "logic", "multi_hop", "multi_hop_qa",
    "multi_hop_unanswerable", "implicit_multi_hop", "implicit_multihop", "ordinal", "probability", "probability_exact",
    "probability_estimate", "agent_action", "agent_action_choice", "routing_hard", "rule_exception", "long_input", "constraint",
    "multi_step_lookup", "numeric_reconciliation", "numerical_reconciliation", "numerical_reasoning", "eikos.finance_exact",
    "eikos.trade_exact", "trap", "adversarial", "ambiguous", "ambiguity", "tradeoff", "multi_label", "injected_instruction",
    "misleading_note", "gui_action", "condition_report", "which_field_conflicts", "which_state_field_now_wrong",
    "claim_supported", "photo_listing_contradiction", "listing_verification", "target_still_matches_reference",
    "insufficient_evidence", "truthfulness_trap", "arithmetic_word_problem", "fact_combination", "numeric_comparison",
}
W05_PREFIX = ("emotion", "eikos.rules_")
W05 = {"intent", "intent_routing", "sentiment", "topic", "snli", "nli", "entity_type", "pii_present", "toxicity_screening",
       "content_safety", "moderation_taxonomy", "commonsense", "reading_commonsense", "science_reasoning", "stance"}
NO_TEACHER = {"large_choice"}


def weight(family: str) -> float:
    if family in NO_TEACHER:
        return 0.0
    if family in PRIORITY:
        return 3.0
    if family in W2:
        return 2.0
    if family in W15:
        return 1.5
    if family in W05 or family.startswith(W05_PREFIX):
        return 0.5
    return 1.0


def quota_key(row: dict) -> str:
    """The key the streamer counts against: the family, prefixed for image rows (image and text budgets are separate)."""
    return ("img:" if row.get("images") else "") + row["family"]


def needs_second_solve(row: dict) -> bool:
    """Two agreeing thinking solves when the gold is not guaranteed: own-pool text (C) and teacher-written plumb documents."""
    if row.get("images") or row.get("gold_kind") == "constructed":
        return False
    return row["source"] == "C" or row.get("dataset") == "plumb"


def family_stats(rows) -> dict[str, dict]:
    st: dict[str, dict] = {}
    for r in rows:
        k = quota_key(r)
        s = st.setdefault(k, {"family": r["family"], "modality": "image" if r.get("images") else "text", "pool_rows": 0,
                              "second": 0, "bc": 0, "sources": collections.Counter()})
        s["pool_rows"] += 1
        s["second"] += needs_second_solve(r)
        s["bc"] += (r["source"] in ("B", "C") and not r.get("images"))
        s["sources"][r["source"]] += 1
    return st


def water_fill(avail: dict[str, int], weights: dict[str, float], budget: int, size: dict[str, int] | None = None) -> dict[str, int]:
    """Integer quotas q_k = min(avail_k, lam * w_k * size_k) summing to min(budget, sum avail over w > 0) (largest-remainder
    rounding). With size = pool rows this is the rate rule min(flag_rate, lam' * w) * rows; without it, plain water-filling."""
    size = size or {k: 1 for k in avail}
    keys = [k for k in avail if weights.get(k, 0) > 0 and avail[k] > 0]
    total_avail = sum(avail[k] for k in keys)
    budget = min(budget, total_avail)
    out = {k: 0 for k in avail}
    if budget <= 0:
        return out
    lo, hi = 0.0, max(avail[k] / (weights[k] * size[k]) for k in keys) + 1
    for _ in range(200):
        lam = (lo + hi) / 2
        if sum(min(avail[k], lam * weights[k] * size[k]) for k in keys) < budget:
            lo = lam
        else:
            hi = lam
    raw = {k: min(avail[k], hi * weights[k] * size[k]) for k in keys}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    left = budget - sum(base.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - base[k]), k)):
        if left <= 0:
            break
        if base[k] < avail[k]:
            base[k] += 1
            left -= 1
    out.update(base)
    return out


def compute(rows, cap_total: int = CAP_TOTAL, image_share: float = IMAGE_SHARE, flag_rate: float = FLAG_RATE,
            keep_rate: float = KEEP_RATE, bc_variants: int = BC_VARIANTS) -> dict:
    st = family_stats(rows)
    avail = {k: int(math.ceil(s["pool_rows"] * flag_rate)) for k, s in st.items()}
    w = {k: weight(s["family"]) for k, s in st.items()}

    def plan(first_total: int):
        img_b = int(round(first_total * image_share))
        q = {}
        size = {k: s["pool_rows"] for k, s in st.items()}
        q.update(water_fill({k: avail[k] for k, s in st.items() if s["modality"] == "image"}, w, img_b, size))
        q.update(water_fill({k: avail[k] for k, s in st.items() if s["modality"] == "text"}, w, first_total - img_b, size))
        load = 0.0
        for k, s in st.items():
            sec = s["second"] / s["pool_rows"]
            bc = s["bc"] / s["pool_rows"]
            load += q[k] * (1 + sec + bc * keep_rate * bc_variants * (1 + sec))
        return q, load

    lo, hi = 0, cap_total
    while lo < hi:                      # largest first-solve total whose expected load fits the cap
        mid = (lo + hi + 1) // 2
        if plan(mid)[1] <= cap_total:
            lo = mid
        else:
            hi = mid - 1
    q, load = plan(lo)
    fams = {}
    for k, s in sorted(st.items()):
        sec = s["second"] / s["pool_rows"]
        bc = s["bc"] / s["pool_rows"]
        fams[k] = {"family": s["family"], "modality": s["modality"], "quota": q[k], "pool_rows": s["pool_rows"],
                   "available_at_flag_rate": avail[k], "weight": w[k], "teacher": w[k] > 0,
                   "rate": round(q[k] / s["pool_rows"], 3),
                   "second_solve_share": round(sec, 3), "bc_text_share": round(bc, 3),
                   "expected_calls": round(q[k] * (1 + sec + bc * keep_rate * bc_variants * (1 + sec)), 1),
                   "sources": dict(sorted(s["sources"].items()))}
    first = sum(q.values())
    return {
        "version": VERSION,
        "cap_total_teacher_calls": cap_total,
        "first_solve_total": first,
        "first_solve_text": sum(v["quota"] for v in fams.values() if v["modality"] == "text"),
        "first_solve_image": sum(v["quota"] for v in fams.values() if v["modality"] == "image"),
        "expected_second_solves": round(sum(q[k] * st[k]["second"] / st[k]["pool_rows"] for k in st)),
        "expected_bc_variant_calls": round(sum(q[k] * st[k]["bc"] / st[k]["pool_rows"] * keep_rate * bc_variants *
                                               (1 + st[k]["second"] / st[k]["pool_rows"]) for k in st)),
        "expected_total_calls": round(load),
        "parameters": {"image_share": image_share, "expected_flag_rate": flag_rate, "keep_rate": keep_rate,
                       "bc_variants_per_kept_parent": bc_variants},
        "streaming_rules": [
            "quota_key = family, prefixed 'img:' for image rows; shard rows carry it as pool.quota_key",
            "a family stops feeding the teacher once its quota of FIRST solves is filled; second solves and B/C variant "
            "re-verifications of accepted items do not count against the family quota but do count against cap_total",
            "stop everything at cap_total teacher calls (the Azure runner's hard cost cap is the final guard)",
            "rows in the heldout-flagged bucket (pool.heldout_flagged_bucket) are never sent for training labels; they are "
            "collected by `assemble_pool.py take-flagged`",
            "families with teacher=false (large_choice) are not in the shards at all",
        ],
        "why": __doc__.split("Rules (all parameters are written to quotas.json):", 1)[1].strip(),
        "families": fams,
    }
