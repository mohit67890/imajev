# Phase-3 label review (Kimi-K2.5 + owner)

Generated 2026-09-26 05:34 by `scripts/p3/review_report.py`. Method: docs/phase-3-plan.md, "Review: open-weight model + small human slice"; formulas in the script docstring.

## Summary

- Sampled 2400 kept items; Kimi reviewed 2373 (unreviewed/errored: 27), cost $26.01.
- Owner reviewed 220 of 220 queued items (167 Kimi disagreements, 60 random).
- Kimi precision on disagreements (owner-confirmed errors / checked): 20.4%; owner error rate on random items Kimi agreed with (Kimi misses): 1.9% (n=53).
- Owner and Kimi concordance on random items: 90.0%. Verdicts changed after seeing Kimi: 0.
- **Sources over 5% estimated errors: C, D.** Watch (only with Kimi misses): I.
- Dev slice: 785 items (185 human-verified core + 600 Kimi-agreed) → `data/p3/review/dev-slice.jsonl`. Never train on these.

Small groups have wide intervals: a group with n < 50 moves ±5 points on one or two items. Treat its status as a prompt to look, not a verdict.

## Error rates

### By source

| group | n | Kimi disagrees | Kimi rate | owner checked | confirmed | precision | confirmed floor | **est. error** | + Kimi misses | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | 713 | 87 | 12.2% | 33 | 2 | 6.1% (own) | 0.3% | **0.7%** | 2.4% | ok |
| B | 324 | 48 | 14.8% | 32 | 5 | 15.6% (own) | 1.5% | **2.3%** | 3.9% | ok |
| C | 719 | 139 | 19.3% | 36 | 11 | 30.6% (own) | 1.7% | **5.9%** | 7.4% | FLAGGED |
| D | 133 | 39 | 29.3% | 32 | 9 | 28.1% (own) | 6.8% | **8.2%** | 9.6% | FLAGGED |
| I | 484 | 107 | 22.1% | 34 | 7 | 20.6% (own) | 1.4% | **4.6%** | 6.0% | watch |

### By item kind (base / variant / unknown)

| group | n | Kimi disagrees | Kimi rate | owner checked | confirmed | precision | confirmed floor | **est. error** | + Kimi misses | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 1093 | 135 | 12.4% | 42 | 11 | 26.2% (own) | 1.0% | **3.2%** | 4.9% | ok |
| unknown | 230 | 97 | 42.2% | 30 | 5 | 16.7% (own) | 2.2% | **7.0%** | 8.1% | FLAGGED |
| unknown_variant | 404 | 143 | 35.4% | 72 | 15 | 20.8% (own) | 4.0% | **7.4%** | 8.6% | FLAGGED |
| variant | 646 | 45 | 7.0% | 23 | 3 | 13.0% (own) | 0.5% | **0.9%** | 2.7% | ok |

### Image vs text

| group | n | Kimi disagrees | Kimi rate | owner checked | confirmed | precision | confirmed floor | **est. error** | + Kimi misses | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image | 697 | 181 | 26.0% | 52 | 12 | 23.1% (own) | 1.7% | **6.0%** | 7.4% | FLAGGED |
| text | 1676 | 239 | 14.3% | 115 | 22 | 19.1% (own) | 1.4% | **2.7%** | 4.3% | ok |

### By source / family

| group | n | Kimi disagrees | Kimi rate | owner checked | confirmed | precision | confirmed floor | **est. error** | + Kimi misses | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A / agent_action | 45 | 10 | 22.2% | 3 | 1 | 33.3% (own) | 2.2% | **7.4%** | 8.9% | FLAGGED |
| A / constraint | 49 | 10 | 20.4% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.5% | ok |
| A / judge_hard | 55 | 15 | 27.3% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.4% | ok |
| A / logic | 55 | 2 | 3.6% | 1 | 0 | 20.4% (pooled) | 0.0% | **0.7%** | 2.6% | ok |
| A / long_input | 34 | 7 | 20.6% | 3 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.5% | ok |
| A / long_policy | 80 | 8 | 10.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.0%** | 3.7% | ok |
| A / multi_hop | 46 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| A / multi_label | 34 | 11 | 32.4% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.3% | ok |
| A / ordinal | 59 | 9 | 15.3% | 4 | 1 | 25.0% (own) | 1.7% | **3.8%** | 5.4% | watch |
| A / probability | 49 | 2 | 4.1% | 2 | 0 | 20.4% (pooled) | 0.0% | **0.8%** | 2.6% | ok |
| A / routing_hard | 39 | 4 | 10.3% | 2 | 0 | 20.4% (pooled) | 0.0% | **2.1%** | 3.8% | ok |
| A / rule_exception | 64 | 6 | 9.4% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.7% | ok |
| A / table_arithmetic | 50 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| A / temporal_numeric | 54 | 3 | 5.6% | 2 | 0 | 20.4% (pooled) | 0.0% | **1.1%** | 2.9% | ok |
| B / adversarial | 11 | 2 | 18.2% | 2 | 0 | 20.4% (pooled) | 0.0% | **3.7%** | 5.2% | watch |
| B / agent_action_choice | 24 | 4 | 16.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |
| B / ambiguous | 14 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / extraction | 11 | 1 | 9.1% | 1 | 0 | 20.4% (pooled) | 0.0% | **1.9%** | 3.6% | ok |
| B / implicit_multi_hop | 14 | 3 | 21.4% | 1 | 0 | 20.4% (pooled) | 0.0% | **4.4%** | 5.8% | watch |
| B / judge_hard | 13 | 1 | 7.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **1.6%** | 3.3% | ok |
| B / long_policy | 15 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / multi_hop | 17 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / multi_hop_qa | 27 | 7 | 25.9% | 4 | 1 | 25.0% (own) | 3.7% | **6.5%** | 7.9% | FLAGGED |
| B / multi_hop_unanswerable | 37 | 7 | 18.9% | 6 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.5% | ok |
| B / probability | 18 | 10 | 55.6% | 9 | 0 | 0.0% (own) | 0.0% | **0.0%** | 0.8% | ok |
| B / routing_hard | 11 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / rubric | 13 | 2 | 15.4% | 1 | 0 | 20.4% (pooled) | 0.0% | **3.1%** | 4.7% | ok |
| B / table_arithmetic | 27 | 5 | 18.5% | 3 | 2 | 66.7% (own) | 7.4% | **12.3%** | 13.9% | FLAGGED |
| B / table_arithmetic_yesno | 10 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / table_text_arithmetic | 24 | 3 | 12.5% | 1 | 1 | 20.4% (pooled) | 4.2% | **2.5%** | 4.2% | ok |
| B / temporal_numeric | 16 | 2 | 12.5% | 1 | 0 | 20.4% (pooled) | 0.0% | **2.5%** | 4.2% | ok |
| B / tradeoff | 9 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| B / trap | 13 | 1 | 7.7% | 1 | 1 | 20.4% (pooled) | 7.7% | **1.6%** | 3.3% | ok |
| C / adequacy_of_answer | 1 | 1 | 100.0% | 1 | 1 | 20.4% (pooled) | 100.0% | **20.4%** | 20.4% | FLAGGED |
| C / ambiguity | 5 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / answerability | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / arithmetic_word_problem | 12 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / attribute_boolean | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / boolq | 9 | 1 | 11.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.3%** | 3.9% | ok |
| C / candidate_labels_in_state | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / claim_supported | 7 | 2 | 28.6% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| C / color | 6 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / colour_of_x | 6 | 3 | 50.0% | 2 | 1 | 20.4% (pooled) | 16.7% | **10.2%** | 11.1% | FLAGGED |
| C / condition_report | 3 | 1 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / contains_claim | 9 | 1 | 11.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.3%** | 3.9% | ok |
| C / content_safety | 8 | 6 | 75.0% | 3 | 2 | 66.7% (own) | 25.0% | **50.0%** | 50.5% | FLAGGED |
| C / contradiction | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / count_bucket | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / count_matches | 4 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / date_number_trap | 13 | 1 | 7.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **1.6%** | 3.3% | ok |
| C / defect_presence | 3 | 2 | 66.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **13.6%** | 14.2% | FLAGGED |
| C / defect_type | 6 | 4 | 66.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **13.6%** | 14.2% | FLAGGED |
| C / eikos.finance_exact | 19 | 1 | 5.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **1.1%** | 2.9% | ok |
| C / eikos.judge | 14 | 1 | 7.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **1.5%** | 3.2% | ok |
| C / eikos.rules_book | 15 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.rules_permit | 14 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.rules_route | 14 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.rules_score | 13 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.rules_tier | 15 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.temporal_numeric | 18 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / eikos.trade_exact | 16 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / emotion | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / emotion_curiosity | 2 | 1 | 50.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| C / emotion_desire | 1 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / entity_type | 9 | 1 | 11.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.3%** | 3.9% | ok |
| C / esci_structured | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / extraction | 2 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / fever | 5 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / finish | 6 | 2 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / image_flaw | 3 | 1 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / image_usability | 6 | 2 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / indoor_outdoor | 4 | 1 | 25.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.1%** | 6.5% | FLAGGED |
| C / insufficient_evidence | 8 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / intent | 10 | 4 | 40.0% | 2 | 0 | 20.4% (pooled) | 0.0% | **8.1%** | 9.3% | FLAGGED |
| C / intent_routing | 8 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / item_shape | 6 | 1 | 16.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |
| C / judge_answer | 7 | 1 | 14.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| C / label_text_matches | 4 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / listing_verification | 3 | 1 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / main_object | 6 | 3 | 50.0% | 1 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| C / material | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / misleading_note | 6 | 1 | 16.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |
| C / moderation_taxonomy | 1 | 1 | 100.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **20.4%** | 20.4% | FLAGGED |
| C / multi_step_lookup | 12 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / natural_attribute_differs | 6 | 4 | 66.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **13.6%** | 14.2% | FLAGGED |
| C / natural_same_item | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / next_action_routing | 11 | 3 | 27.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.6%** | 6.9% | FLAGGED |
| C / nli | 6 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / numeric_comparison | 11 | 4 | 36.4% | 1 | 0 | 20.4% (pooled) | 0.0% | **7.4%** | 8.6% | FLAGGED |
| C / numeric_reconciliation | 26 | 4 | 15.4% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.1%** | 4.7% | ok |
| C / numerical_reasoning | 11 | 1 | 9.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **1.9%** | 3.6% | ok |
| C / numerical_reconciliation | 13 | 3 | 23.1% | 2 | 1 | 20.4% (pooled) | 7.7% | **4.7%** | 6.1% | watch |
| C / object_present | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / pattern | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / photo_quality | 6 | 4 | 66.7% | 2 | 2 | 20.4% (pooled) | 33.3% | **13.6%** | 14.2% | FLAGGED |
| C / pii_present | 1 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / policy_exception | 10 | 2 | 20.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **4.1%** | 5.6% | watch |
| C / probability_estimate | 15 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / probability_exact | 26 | 2 | 7.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **1.6%** | 3.3% | ok |
| C / product_category | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / reference_category | 6 | 2 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / reference_color | 5 | 3 | 60.0% | 1 | 0 | 20.4% (pooled) | 0.0% | **12.2%** | 13.0% | FLAGGED |
| C / reference_comparison | 3 | 3 | 100.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **20.4%** | 20.4% | FLAGGED |
| C / reference_material | 6 | 1 | 16.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |
| C / response_rubric | 8 | 6 | 75.0% | 2 | 1 | 20.4% (pooled) | 12.5% | **15.3%** | 15.7% | FLAGGED |
| C / routing | 8 | 1 | 12.5% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.5%** | 4.2% | ok |
| C / rubric | 5 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / rule_precedence | 6 | 2 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / same_product | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / scene_type | 6 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / sentiment | 5 | 5 | 100.0% | 2 | 0 | 20.4% (pooled) | 0.0% | **20.4%** | 20.4% | FLAGGED |
| C / severity_urgency | 5 | 2 | 40.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **8.1%** | 9.3% | FLAGGED |
| C / snli | 9 | 6 | 66.7% | 1 | 1 | 20.4% (pooled) | 11.1% | **13.6%** | 14.2% | FLAGGED |
| C / squad2 | 2 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / stance | 16 | 4 | 25.0% | 2 | 0 | 20.4% (pooled) | 6.2% | **5.1%** | 6.5% | FLAGGED |
| C / style | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / target_still_matches_reference | 3 | 1 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| C / temporal_arithmetic | 29 | 1 | 3.4% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.7%** | 2.5% | ok |
| C / temporal_ordering | 16 | 2 | 12.5% | 2 | 1 | 20.4% (pooled) | 6.2% | **2.5%** | 4.2% | ok |
| C / text_visible | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / topic | 5 | 1 | 20.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **4.1%** | 5.6% | watch |
| C / toxicity_screening | 1 | 1 | 100.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **20.4%** | 20.4% | FLAGGED |
| C / tradeoff | 7 | 2 | 28.6% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| C / two_image_same_subject | 3 | 1 | 33.3% | 1 | 1 | 20.4% (pooled) | 33.3% | **6.8%** | 8.0% | FLAGGED |
| C / two_image_which | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / unanswerable_vqa | 3 | 3 | 100.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **20.4%** | 20.4% | FLAGGED |
| C / vqa | 6 | 5 | 83.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **17.0%** | 17.3% | FLAGGED |
| C / what_changed | 3 | 2 | 66.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **13.6%** | 14.2% | FLAGGED |
| C / which_field_conflicts | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| C / which_state_field_now_wrong | 3 | 2 | 66.7% | 1 | 0 | 20.4% (pooled) | 0.0% | **13.6%** | 14.2% | FLAGGED |
| D / agent_action | 18 | 6 | 33.3% | 5 | 5 | 100.0% (own) | 27.8% | **33.3%** | 34.6% | FLAGGED |
| D / judge_hard | 22 | 6 | 27.3% | 3 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.4% | ok |
| D / long_input | 18 | 7 | 38.9% | 6 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.2% | ok |
| D / long_policy | 19 | 3 | 15.8% | 3 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.6% | ok |
| D / multi_label | 15 | 4 | 26.7% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.4% | ok |
| D / routing_hard | 22 | 8 | 36.4% | 7 | 4 | 57.1% (own) | 18.2% | **20.8%** | 22.0% | FLAGGED |
| D / rule_exception | 19 | 5 | 26.3% | 4 | 0 | 0.0% (own) | 0.0% | **0.0%** | 1.4% | ok |
| I / chart_claim | 7 | 2 | 28.6% | 1 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| I / chart_compare | 12 | 2 | 16.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |
| I / chart_extremum | 8 | 3 | 37.5% | 2 | 0 | 20.4% (pooled) | 0.0% | **7.6%** | 8.8% | FLAGGED |
| I / chart_readoff | 9 | 1 | 11.1% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.3%** | 3.9% | ok |
| I / chart_score | 11 | 1 | 9.1% | 1 | 0 | 20.4% (pooled) | 0.0% | **1.9%** | 3.6% | ok |
| I / chart_threshold | 8 | 1 | 12.5% | 1 | 0 | 20.4% (pooled) | 0.0% | **2.5%** | 4.2% | ok |
| I / chart_trend | 7 | 2 | 28.6% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| I / docimg_arithmetic | 11 | 2 | 18.2% | 2 | 2 | 20.4% (pooled) | 18.2% | **3.7%** | 5.2% | watch |
| I / docimg_category | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / docimg_date | 8 | 1 | 12.5% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.5%** | 4.2% | ok |
| I / docimg_extract | 14 | 2 | 14.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| I / docimg_form | 8 | 2 | 25.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.1%** | 6.5% | FLAGGED |
| I / docimg_policy | 8 | 3 | 37.5% | 1 | 0 | 20.4% (pooled) | 0.0% | **7.6%** | 8.8% | FLAGGED |
| I / docimg_timetable | 7 | 3 | 42.9% | 1 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / geometry_angle | 9 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / geometry_area | 7 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / geometry_circle | 7 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / geometry_classify | 7 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / geometry_coord | 4 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / geometry_length | 7 | 3 | 42.9% | 1 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / geometry_polygon | 7 | 2 | 28.6% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| I / geometry_relation | 7 | 1 | 14.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| I / gui_action | 12 | 5 | 41.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **8.5%** | 9.6% | FLAGGED |
| I / image_joint_rule | 73 | 10 | 13.7% | 2 | 1 | 20.4% (pooled) | 1.4% | **2.8%** | 4.4% | ok |
| I / inventory_audit | 6 | 3 | 50.0% | 2 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| I / inventory_colour_count | 6 | 3 | 50.0% | 1 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| I / inventory_compare | 6 | 4 | 66.7% | 3 | 0 | 0.0% (own) | 0.0% | **0.0%** | 0.6% | ok |
| I / inventory_count | 10 | 3 | 30.0% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.1%** | 7.4% | FLAGGED |
| I / inventory_locate | 7 | 2 | 28.6% | 1 | 1 | 20.4% (pooled) | 14.3% | **5.8%** | 7.2% | FLAGGED |
| I / inventory_misplaced | 7 | 3 | 42.9% | 1 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / inventory_not_visible | 3 | 1 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| I / inventory_out_of_stock | 8 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / inventory_price_tag | 7 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / inventory_reorder | 8 | 3 | 37.5% | 1 | 1 | 20.4% (pooled) | 12.5% | **7.6%** | 8.8% | FLAGGED |
| I / inventory_total | 7 | 4 | 57.1% | 1 | 1 | 20.4% (pooled) | 14.3% | **11.6%** | 12.4% | FLAGGED |
| I / mobile_locate | 11 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / mobile_next | 4 | 2 | 50.0% | 1 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| I / mobile_permission | 7 | 2 | 28.6% | 1 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| I / mobile_possible | 4 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / mobile_tap | 7 | 1 | 14.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| I / pro_ground | 8 | 4 | 50.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **10.2%** | 11.1% | FLAGGED |
| I / pro_locate | 7 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / pro_menu | 7 | 1 | 14.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| I / safety_blocked_exit | 7 | 2 | 28.6% | 0 | 0 | 20.4% (pooled) | 0.0% | **5.8%** | 7.2% | FLAGGED |
| I / safety_checklist | 4 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_count | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_extinguisher | 7 | 1 | 14.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **2.9%** | 4.5% | ok |
| I / safety_forklift_lane | 6 | 4 | 66.7% | 3 | 1 | 33.3% (own) | 16.7% | **22.2%** | 22.9% | FLAGGED |
| I / safety_guardrail | 6 | 2 | 33.3% | 1 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| I / safety_headcount | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_not_visible | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_position | 3 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_ppe | 7 | 3 | 42.9% | 0 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / safety_ppe_rule | 7 | 3 | 42.9% | 1 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / safety_presence | 5 | 0 | 0.0% | 0 | 0 | 20.4% (pooled) | 0.0% | **0.0%** | 1.9% | ok |
| I / safety_sign_check | 7 | 3 | 42.9% | 0 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / safety_spill | 7 | 3 | 42.9% | 2 | 0 | 20.4% (pooled) | 0.0% | **8.7%** | 9.8% | FLAGGED |
| I / safety_zone_score | 6 | 2 | 33.3% | 0 | 0 | 20.4% (pooled) | 0.0% | **6.8%** | 8.0% | FLAGGED |
| I / screen_locate | 12 | 2 | 16.7% | 0 | 0 | 20.4% (pooled) | 0.0% | **3.4%** | 5.0% | ok |

