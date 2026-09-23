# Reasoning dev set (authored 240 items) — baselines, 23 Sept 2026

Scored on the Mac (MLX, single order, no calibration). Abstention credited when the label is unknown.

| Model | All | ambiguous | long_policy | multi_hop | probability | routing | temporal_numeric | tradeoff | trap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B base | 33.3% | 17/30 | 6/30 | 7/30 | 6/30 | 12/30 | 4/30 | 15/30 | 13/30 |
| 2B v1.1 | 46.7% | 4/30 | 13/30 | 19/30 | 8/30 | 26/30 | 5/30 | 13/30 | 24/30 |
| 9B base | **68.8%** | 25/30 | 20/30 | 23/30 | 11/30 | 29/30 | 14/30 | 15/30 | 28/30 |
| 9B v1.1 | 63.7% | 12/30 | 18/30 | 28/30 | 13/30 | 27/30 | 10/30 | 17/30 | 28/30 |

Reading: the set detects the 9B's forgetting (−5 overall, concentrated in `ambiguous`: the fine-tuned models stop abstaining on
underdetermined states, 25 → 12 and 17 → 4) and in temporal/numeric. It is milder than JevBench hard (−22), so use it as one of two
selection signals (`--select mean_accuracy`), not as a JevBench proxy. Both fine-tuned models gain on routing, trap and multi-hop
in this set, i.e. our format training helps where reasoning is shallow and hurts where the honest answer is "unknown".
