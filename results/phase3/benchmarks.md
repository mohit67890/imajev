# Phase-3 benchmark results (all checkpoints, same pod, same protocol)

Every number below was produced by `scripts/p3/eval_checkpoint.py` on pod the phase-3 pod (8xH100) on 2026-09-26 and verified for completeness by
`scripts/p3/collect_benchmarks.py` (row counts per panel). Soup rows are absent: the soup fallback was stopped by the owner and never evaluated.

## External benchmarks

| checkpoint | ImajevBench acc | visual | joint | text | JevBench easy | original | hard (single raw) | hard (single cal) | hard ECE cal | pooled ECE cal | hard rot4 raw | hard rot4 cal | fast-decisions macro | DecisionBench 3k acc | full-suite equiv | ordinal | reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 82.4 | 107/120 | 97/122 | 26/37 | 100.0 | 98.6 | 69.4 | 69.4 | 0.109 | 0.038 | 70.3 | 70.3 | 59.0 | 72.2 | 78.2 | 49.1 | 67.1 |
| r1-s000660 | 82.4 | 109/120 | 97/122 | 24/37 | 100.0 | 98.6 | 71.2 | 71.2 | 0.083 | 0.051 | 68.5 | 68.5 | 60.0 | 73.4 | 78.9 | 54.5 | 76.1 |
| r1-s001320 | 81.7 | 109/120 | 96/122 | 23/37 | 100.0 | 98.6 | 71.2 | 71.2 | 0.093 | 0.054 | 68.5 | 68.5 | 59.4 | 73.6 | 79.0 | 57.0 | 76.1 |
| r1-s001980 | 82.4 | 109/120 | 95/122 | 26/37 | 100.0 | 98.6 | 68.5 | 68.5 | 0.107 | 0.054 | 68.5 | 68.5 | 60.2 | 73.4 | 78.8 | 58.4 | 79.3 |
| r1-s002640 | 81.7 | 108/120 | 95/122 | 25/37 | 100.0 | 98.6 | 69.4 | 69.4 | 0.096 | 0.047 | 67.6 | 67.6 | 60.3 | 72.9 | 78.3 | 58.8 | 78.2 |
| r2-s000099 | 83.5 | 109/120 | 100/122 | 24/37 | 100.0 | 98.6 | 71.2 | 71.2 | 0.084 | 0.053 | 72.1 | 72.1 | 60.4 | 73.1 | 78.7 | 60.6 | 79.6 |
| r2-s000198 | 84.2 | 110/120 | 100/122 | 25/37 | 100.0 | 98.6 | 69.4 | 69.4 | 0.122 | 0.065 | 72.1 | 72.1 | 60.4 | 73.2 | 78.9 | 60.6 | 80.7 |
| r2-s000291 | 83.9 | 109/120 | 100/122 | 25/37 | 100.0 | 98.6 | 71.2 | 71.2 | 0.082 | 0.046 | 72.1 | 72.1 | 60.4 | 72.9 | 78.5 | 61.3 | 80.4 |

n: ImajevBench 279 items (visual 120, joint 122, text 37); JevBench public 231 (easy 48, original 72, hard 111); fast-decisions dev 1,700 rows / 2,900 heads; DecisionBench 3k stratified subset (tracking only, never used for selection or training).

## Our held-out panels

| checkpoint | fresh (4,297) | flagged (2,813) | human (785) | authored dev (150) | charts (300) | docimg (300) | inventory (250) | safety (250) | geometry (250) | screens (250) | p2b test (435) | state probe (200) | pairs probe (60) | irrelevance (2,823) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 68.4 | 23.9 | 48.2 | 64.7 | 84.0 | 76.0 | 54.8 | 72.0 | 66.4 | 72.8 | 85.7 | 71.5 | 98.3 | 80.2 |
| r1-s000660 | 83.7 | 69.3 | 73.0 | 72.0 | 94.7 | 88.3 | 65.2 | 89.2 | 83.6 | 92.0 | 81.6 | 73.0 | 96.7 | 79.3 |
| r1-s001320 | 86.1 | 75.3 | 73.5 | 80.0 | 94.0 | 90.7 | 68.8 | 90.8 | 87.2 | 95.2 | 83.9 | 76.5 | 98.3 | 80.8 |
| r1-s001980 | 87.0 | 77.2 | 75.3 | 73.3 | 95.0 | 92.0 | 71.2 | 93.2 | 88.8 | 95.2 | 84.1 | 77.0 | 96.7 | 80.2 |
| r1-s002640 | 87.1 | 78.3 | 76.2 | 76.7 | 95.3 | 90.3 | 69.6 | 92.8 | 88.8 | 95.2 | 84.6 | 75.5 | 96.7 | 80.3 |
| r2-s000099 | 86.9 | 78.0 | 75.5 | 77.3 | 95.7 | 89.7 | 69.2 | 92.0 | 87.6 | 94.8 | 84.4 | 76.0 | 96.7 | 80.4 |
| r2-s000198 | 87.2 | 78.6 | 76.1 | 78.7 | 96.0 | 89.7 | 68.4 | 92.8 | 88.8 | 96.0 | 84.1 | 75.5 | 96.7 | 80.2 |
| r2-s000291 | 87.2 | 79.0 | 76.2 | 79.3 | 95.7 | 90.3 | 67.6 | 92.4 | 88.4 | 95.6 | 84.1 | 76.5 | 96.7 | 80.0 |

## Gates (phase-2c rules, 4b bounds)

| checkpoint | shippable | image gate | failing checks |
|---|---|---|---|
| shipped | True | True | none |
| r1-s000660 | False | True | c.unknown_correct_rate 64.29 >= 100.0 |
| r1-s001320 | False | False | a.joint_items 96 >= 97; c.unknown_correct_rate 78.57 >= 100.0 |
| r1-s001980 | False | False | a.joint_items 95 >= 97; c.unknown_correct_rate 92.86 >= 100.0 |
| r1-s002640 | False | False | a.joint_items 95 >= 97; c.unknown_correct_rate 92.86 >= 100.0 |
| r2-s000099 | False | True | c.unknown_correct_rate 64.29 >= 100.0 |
| r2-s000198 | False | True | c.unknown_correct_rate 78.57 >= 100.0 |
| r2-s000291 | False | True | c.unknown_correct_rate 78.57 >= 100.0 |

## Typed S1-Bench (our conversion of the 220 English items; 212 kept; NOT an S1-Bench score)

| checkpoint | acc | abstain % | mean conf | ECE | reasoning | knowledge | analysis | instruction following |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | 98.6 | 0.0 | 98.4 | 0.006 | 96.7 | 100.0 | 100.0 | 96.2 |
| r1-s000660 | 99.1 | 0.0 | 97.1 | 0.020 | 98.3 | 100.0 | 100.0 | 96.2 |
| r1-s001320 | 99.1 | 0.0 | 97.2 | 0.024 | 98.3 | 100.0 | 100.0 | 96.2 |
| r1-s001980 | 99.5 | 0.0 | 98.6 | 0.010 | 100.0 | 100.0 | 100.0 | 96.2 |
| r1-s002640 | 99.1 | 0.0 | 98.4 | 0.011 | 98.3 | 100.0 | 100.0 | 96.2 |
| r2-s000099 | 99.1 | 0.0 | 97.7 | 0.020 | 98.3 | 100.0 | 100.0 | 96.2 |
| r2-s000198 | 99.1 | 0.0 | 97.6 | 0.020 | 98.3 | 100.0 | 100.0 | 96.2 |
| r2-s000291 | 99.1 | 0.0 | 97.7 | 0.019 | 98.3 | 100.0 | 100.0 | 96.2 |
