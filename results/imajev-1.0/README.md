# decision-p2b: last hard top-up before launch (7×H100 pod `ei3vnzgkax4r3m`, 2026-09-24)

## Frozen Qwen3.6-35B-A3B on public JevBench (`check35b/`)

Interface: structured generation through vLLM 0.30 (bf16, one H100) with the jevbench `openai_compat` adapter
(JSON-schema output, temperature 0). Thinking off = `enable_thinking: false` in the chat template, 4,096-token budget.
Thinking on = reasoning parser `qwen3`, 16,000-token completion budget, 900 s timeout (the adapter's default 4,096 budget
was exhausted by reasoning on ~20% of hard items; the patched adapter reads `OPENAI_COMPAT_MAX_TOKENS`).

| Mode | hard (111) | original (72) | easy (48) |
|---|---|---|---|
| thinking off | 61.3 / ECE 0.329 | 88.9 / 0.098 | 100 / 0.002 |
| thinking on | **97.3** / ECE 0.045 | 100 / 0.004 | 100 / 0.001 |

Thinking-on hard by family: adversarial 6/6, ambiguous 7/7, judge_hard 17/17, long_policy 18/19, multi_hop 17/18,
probability 10/10, routing_hard 5/5, temporal_numeric 14/15, tradeoff 6/6, trap 8/8 (108/111; 1 request failed).

Reading: single-pass (the imajev serving mode) the frozen 35B-A3B is at our trained 4B's level (62), so it is not a
fourth tier without training. With reasoning it is above every published JevBench hard number (Eikos-27B 82.9,
xor 77.5, reflex 76.6, jevk5 73.9) — a different interface (seconds of latency, thousands of tokens per decision), and a
very strong grader of exactly this kind of question. Consequence: it was added as a THIRD answerer for phase-2b
(unanimous agreement across Qwen3.6-27B thinking, gpt-oss-20b and Qwen3.6-35B-A3B thinking). Adding a unanimous
answerer can only remove rows, never mislabel them; the two-answerer records are recoverable by re-running the
assembler on the first two answer files.

## Generation (writers)
Two Qwen3.6-27B replicas (TP=4 and TP=2), thinking on for the hard families, off for the unknown families.
Accepted documents: hard 2,166 (1,445 + 721; 234 rejected), unknown 533 (of 900 attempted) → 2,699 documents.

## mojev (MoLeMo-Lab/mojev, 0.85B, MIT) on public JevBench — same `typesafe` interface as imajev (`mojev/` on the pod)
Served with `mojev serve` (its TypeSafe-compatible `/v1/systemone`), one H100, 2026-09-24. hard **33.3** / ECE 0.268, original 63.9 / 0.171,
easy 91.7 / 0.066. Its reported 93.23% accuracy is on its own synthetic mojev-mix test split. No unknown/abstain, image support untested.

## Phase-2b assembly (three answerers, unanimous)
8,097 questions from 2,699 documents → 4,852 kept (train 4,114 / test 486 / dev 252); 1,883 dropped by disagreement, 1,362 by the
near-duplicate lint (454 documents), 148 unknown rows kept (3.1%, below the 15% target — caveat carried into training).
Manifest `decision-p2b.jsonl`: 9,066 rows (train 7,812 incl. 30% phase-2 replay, calibration 622 on held-out domains
telecom/hospitality/nonprofit_grants, test 435, dev 197; unknown 196) + `decision-p2b-jevstyle-dev.jsonl` (150 authored items).

## cua-ai/cua-s1-4b-0.2 on public JevBench (off-domain row, `cua-s1-v2/` on the pod)
A computer-use specialist: text LoRA (r16/α32 on q/k/v/o/gate/up/down) on Qwen3.5-4B that scores (element, action) options
for a screen state; its authors state it is not for use outside that task. Its adapter keys use the text-only layout
(`model.layers.N`); the multimodal Qwen3.5-4B class names them `model.language_model.layers.N`, so PEFT and vLLM both attach
zero adapters silently (first run: identical to base on 111/111 hard items). With the keys remapped (PEFT check: max |Δlogit|
7.3), served with vLLM LoRA, generic JevBench prompt, thinking off, JSON-schema output, one H100, 2026-09-24:
hard **52.3** / ECE 0.456, original 88.9 / 0.111, easy 100 / 0.001. Control, untouched Qwen3.5-4B base under identical settings:
hard 48.6 / 0.448, original 80.6 / 0.181, easy 100 / 0.005. Read as "a GUI-action LoRA transfers a little, stays far below the
trained single-pass tiers and is badly over-confident", not as a ranking of the model at its own task.

## Calibration fit, 4B phase 2b (`calibration-p2b-4b.json`, schema 1.1, fitted on the 622-row held-out-domain fold)
Temperatures: boolean:2 1.80, choice:3-5 1.39, choice:6-10 2.12 (n=8), ordinal:3-5 1.42, ordinal:6-10 1.44 (n=9) — the model is
over-confident on hard typed questions, in line with the raw JevBench hard ECE 0.215 → 0.163 with the pod's single temperature (1.52).
Unknown offsets from this fold are NOT shippable as fitted (choice:3-5 −2.4, boolean:2 −0.9, ordinal:3-5 +0.7, two buckets at the −4 floor):
the fold has 3.1% unknown targets, so NLL pushes offsets to extremes exactly as on the v2.1 fold. Policy (as decided for v2.1):
ship temperatures from this fold; keep offsets bounded and only in buckets with unknown targets, decided after the MMLU /
irrelevance off-distribution check in the morning. Small buckets (n<20) inherit the neighbouring bucket's temperature.

## Phase-2b training results (7×H100, 2026-09-24; `pod/train-out-p2b/`; 2 epochs, lr 2e-5 from the phase-2 best adapters)

| Panel | 2B (p2 → p2b) | 4B (p2 → p2b) | 9B (p2 → p2b) |
|---|---|---|---|
| JevBench hard, raw | 55.0 → 56.8 (ECE 0.234 → 0.187) | 62.2 → **67.6** (0.224 → 0.215) | 67.6 → 68.5 (0.207 → 0.236) |
| JevBench hard, best serving variant | 56.8 cal (ECE 0.184) | 67.6 cal (ECE **0.163**) | **69.4** rot4+cal (ECE **0.170**) |
| JevBench original / easy | 91.7 / 100 | 98.6 / 100 | **100** / 100 |
| ImajevBench (279) | 68.5 → **70.3** | 80.6 → **82.4** | 81.4 → **82.8** |
| phase-2b test (435) | 77.7 | 85.1 | 86.0 |
| phase-2 test (regression, 2,983) | 66.4 → 65.5 | 77.4 → 77.7 | 79.7 → 80.1 |
| reasoning dev (6,240) | 64.5 → **58.9** | 67.8 → 66.6 | 69.2 → 67.4 |
| state probe / pairs probe | 69.5 / 100 | 68.5 / 98.3 | 69.5 / 80.0 |
| selected checkpoint | best = step 80 of 220 | best = last (266) | best ≈ step 350 of 365 |

2B: best/last differ (last hard 55.0 raw, 59.5 rot4 — noise on 111 items). Dev curves in `pod/train-out-p2b/<size>/dev-curve.jsonl`.

Reading: the hard-tier data moved the 4B most (+5.4 hard, +1.8 ImajevBench, calibration better), the 9B less (+1.8 hard with
rotations+calibration, +1.4 ImajevBench, text track 21→29/37, original 100), the 2B least (+1.8 hard) at the cost of a 5.6-point drop on
the broad reasoning dev set. All three miss largely the same hard items (4B∩9B = 25 of ~35 misses; 18 of those are JevK5 misses too);
the family we are consistently behind JevK5 on is judge_hard (10/17 vs 13/17), which no writer family covers. Next round: soft-target
distillation from Qwen3.6-35B-A3B (thinking) + a judge-style family + calibration term.

Same-protocol competitor rows (our runs on this pod, `reports/benchmarks/`): JevK5 hard 73.9 / ECE 0.073, original 97.2, easy 100.

## Calibration decision (2026-09-24 ~05:25 IST): temperatures fitted on the authored JevBench-style dev set
Why: the model is over-confident on hard items (4B raw: 72/111 items above 0.9 confidence, 56 correct; JevK5 36/34, Hopper 19/18).
The pod's single T (1.52, fitted on the 85%-accuracy held-out-domain fold) under-corrects for the ~68%-accuracy hard regime.
Re-temperaturing the raw JevBench-hard predictions offline: 4B ECE 0.215 → 0.164 (T 1.52) → 0.088 (T 2.30, NLL-optimal on the
150-item authored dev set); 9B 0.236 → 0.181 → 0.104 raw (rotations lower it further). Shipped files `calibration-p2b-<size>-final.json`
(schema 1.1, one T for every bucket: 2B/4B/9B fitted per size, offsets 0 until the MMLU/irrelevance check). Long-term fix: soft-target
distillation + Brier term so the model hedges on its own (competitors' mean top-probability ~0.7 vs ours 0.88).
