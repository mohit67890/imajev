# Phase 3 final comparison (pick: r2-s000291)

OWNER OVERRIDE (2026-09-26 15:40 IST): r2-s000291 picked although it fails c.unknown_correct_rate (11/14 unknown-gold p2b rows, bound 14/14, zero tolerance; misses at 0.83/0.60/0.51 confidence; false-abstention 0.5% on 421 answerable rows). Every other gate passes (ImajevBench 83.9, visual 109, joint 100, state 76.5, pairs 96.7, irrelevance 80.0). Best by score (81.67 = mean(fresh 87.2, human 76.2)). Soup fallback stopped by the owner (w75/w50/w25 evals killed, not evaluated).

| checkpoint | fresh | human | hard_single_raw | hard_single_cal | hard_ece_single_cal | pooled_ece_single_cal | hard_rot4_cal | joint | gates | fastdec_macro | db_full_equiv | db_ordinal | db_reasoning | charts | docimg | inventory | safety | geometry | screens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 68.4 | 48.2 | 69.4 | 69.4 | 0.109 | 0.038 | 70.3 | 97 | True | 59.0 | 78.2 | 49.1 | 67.1 | 84.0 | 76.0 | 54.8 | 72.0 | 66.4 | 72.8 |
| r1-s000660 | 83.7 | 73.0 | 71.2 | 71.2 | 0.083 | 0.051 | 68.5 | 97 | False | 60.0 | 78.9 | 54.5 | 76.1 | 94.7 | 88.3 | 65.2 | 89.2 | 83.6 | 92.0 |
| r1-s001320 | 86.1 | 73.5 | 71.2 | 71.2 | 0.093 | 0.054 | 68.5 | 96 | False | 59.4 | 79.0 | 57.0 | 76.1 | 94.0 | 90.7 | 68.8 | 90.8 | 87.2 | 95.2 |
| r1-s001980 | 87.0 | 75.3 | 68.5 | 68.5 | 0.107 | 0.054 | 68.5 | 95 | False | 60.2 | 78.8 | 58.4 | 79.3 | 95.0 | 92.0 | 71.2 | 93.2 | 88.8 | 95.2 |
| r1-s002640 | 87.1 | 76.2 | 69.4 | 69.4 | 0.096 | 0.047 | 67.6 | 95 | False | 60.3 | 78.3 | 58.8 | 78.2 | 95.3 | 90.3 | 69.6 | 92.8 | 88.8 | 95.2 |
| r2-s000099 | 86.9 | 75.5 | 71.2 | 71.2 | 0.084 | 0.053 | 72.1 | 100 | False | 60.4 | 78.7 | 60.6 | 79.6 | 95.7 | 89.7 | 69.2 | 92.0 | 87.6 | 94.8 |
| r2-s000198 | 87.2 | 76.1 | 69.4 | 69.4 | 0.122 | 0.065 | 72.1 | 100 | False | 60.4 | 78.9 | 60.6 | 80.7 | 96.0 | 89.7 | 68.4 | 92.8 | 88.8 | 96.0 |
| **r2-s000291** | 87.2 | 76.2 | 71.2 | 71.2 | 0.082 | 0.046 | 72.1 | 100 | False | 60.4 | 78.5 | 61.3 | 80.4 | 95.7 | 90.3 | 67.6 | 92.4 | 88.4 | 95.6 |

## Targets (picked vs shipped 1.0, same pod, same protocol)

| target | value | rule | met |
|---|---:|---|---|
| JevBench public hard, single pass + cal | 71.2 | >= 73 (rot4 measured as fallback) | NO |
| JevBench public hard, rot4 + cal (fallback) | 72.1 | >= 73 | NO |
| Held-out fresh-seed vs shipped | 18.8 | +8 pts | yes |
| DecisionBench 3k (full-suite equivalent) vs shipped | 0.3 | +2 pts (tracking-only; read by a person) | NO |
| DecisionBench ordinal vs shipped | 12.2 | +5 pts | yes |
| DecisionBench reasoning vs shipped | 13.2 | +5 pts | yes |
| All gates PASS without a soup | False | phase-2c rules | NO |
| ImajevBench joint track | 100 | >= 99/122 | yes |
| Hard ECE after calibration (single) | 0.082 | <= 0.08 | NO |
| Pooled public ECE (single, cal) | 0.046 | <= 0.03 | NO |

DecisionBench columns are tracking-only (never read by the selector). Ship only if every target rule holds; otherwise keep 1.0.
