# Phase 2c (Eikos-style soft-target delta fine-tunes) — results log

Pod the phase-2c pod (8×H100, $27.92/h), 2026-09-24. Recipe: `docs/phase-2c-plan.md` (+amendments), runbook
`docs/phase-2c-runbook.md`, audit `docs/phase-2c-audit.md`. Training mixture `data/manifests/decision-p2c.jsonl`
(39,515 rows: phase-2b teacher rows relabelled with Qwen3.6-35B-A3B distributions, 9,880 new hard/judge/programmatic
rows, Eikos strict slice 10,570, image replay 5k; train unknown share 4.9%). Delta lanes start from the phase-2b
adapters; 2 epochs; `--soft-targets --soft-weight 1.0 --rationale-weight 0.3 --rationale-max-tokens 192 --permute-options`.

Ship gates (`cloud/p2c_gates_reference.json`): ImajevBench public ≥ phase-2b − 1 (4B 81.4), visual/joint item counts,
probes within tolerance, correct-unknown rate, false abstention, irrelevance. Bars: Eikos-4B same-protocol
JevBench hard 73.9 / ECE 0.054; JevK5 73.9. Eikos's own published table (their harness, 2026-09): 4B final 72.1 hard / 91.7 original / ECE 0.049; 27B final 82.9 / 100 / 0.051; Jev reference 73.0 / 98.6 (official board). Our run of their 4B is 2 items above their own number, so harnesses agree within noise; the material gap to Eikos-4B is ECE, not accuracy.

## 4B (trained 09:26–10:52 UTC, 747 steps, OOM at step 625 → fresh-process resume at the same budget)

Dev curve: phase-2c dev 0.908 → 0.928 (best, step 260) → 0.922 (last); judge-dev 29/29 from step ~100 on.
Best checkpoint selected on the text dev sets, i.e. before most of the image replay had been seen.

| candidate | JevBench hard raw (ECE) | original | easy | ImajevBench public (joint/text/visual) | private-1 hidden | gates |
|---|---|---|---|---|---|---|
| phase-2b 4B (reference) | 67.6 (cal 0.088) | | | 82.4 (99/26/105) | 84.7 (68/23/80), ECE 0.076 | — |
| 2c best (step 260) | **71.2** (0.165) | 98.6 | 100 | 79.6 (92/24/106) | 83.2 (66/21/81), ECE 0.053 | FAIL a.imajevbench_acc, a.joint_items; all others PASS |
| 2c last (step 747) | 70.3 (0.162) | 100 | 100 | 79.9 (94/25/104) | | (not gated) |
| soup50 = ½ p2b + ½ 2c-best (LoRA + readout averaged) | 69.4 (0.164) | 98.6 | 100 | **82.4 (97/26/107)** | 84.2 (67/23/80), ECE 0.056 | image gates PASS → **4B ship candidate** |
| soup75 = ¼ p2b + ¾ 2c-best | 69.4 (0.142) | 98.6 | 100 | 80.6 (94/25/106) | | FAIL a.imajevbench_acc |

Where 2c-best lost on ImajevBench vs phase-2b: joint threshold_rule 48→46, rule_exception 20→18, multi_step_rule
6→4, text numerical_reconciliation 6→4; visual counting +1, spatial_counting +1, comparison −1 (14 joint items lost,
8 gained). Unknown gold 14/14 correct; false abstention 0.24%; irrelevance 80.7 (v2.1 73.8); state probe 71.0 (ref 72.0);
pairs 98.3; reasoning dev 67.2 (phase-2b 66.6, v2.1 67.8).

Calibration: the pod fits T on the phase-2c calibration fold (teacher rows, hard ×3): 2c-best T=1.045 → hard ECE
0.159 — the soft-target model is calibrated on its own distribution but over-confident on JevBench hard. Authored-dev
(150 items, phase-2b policy) fits: 2c-best T=1.29, last 1.50, soup50 1.72. Offline re-temperature of the raw hard
responses: 2c-best ECE 0.165 → 0.148 (T 1.3) → 0.071 (1.6) → 0.069 (2.0); soup50 0.164 → 0.104 (1.6) → 0.091 (2.0)
→ 0.078 (2.6). Policy: ship the authored-dev T (no fitting on JevBench), report the resulting ECE.
Served check (server `--calibration` with the authored-dev file, T=1.72): soup50 hard 69.4 / ECE 0.109, original 98.6 / 0.093.
Rotation averaging (`--rotations 4`, the phase-2b 9B serving setting): soup50 rot4 hard 70.3 / ECE 0.171 raw; rot4+cal **70.3 / 0.116** (original 98.6 / 0.024 raw) → 4B serving configuration = soup50 + rot4 + authored-dev T.

Raw responses: pod `p2/train-out-p2c/4b-delta/jevbench-*` (pulled to `reports/decision-p2c/pod/` after the run).

4B decision (11:20 UTC): interpolation sweep w∈{0, 0.5, 0.75, 1} on the 2c-best adapter gives hard 67.6 / 69.4 / 69.4 / 71.2 and ImajevBench 82.4 / 82.4 / 80.6 / 79.6; the differences on hard are 2 items apiece (111 items). soup50 is the only candidate above phase-2b on hard that passes every gate.

## Calibration objective (12:00 UTC, owner: "no further runs — if we can make calibration better then do it now")
Post-hoc only, on the authored dev predictions already on disk. The NLL-optimal single T on the 150 authored items under-corrects
the hard regime for every candidate; fitting the same items for ECE (mean of 5/10/15/20-bin ECE, no JevBench data) gives a larger T.

| candidate | NLL-fit T → hard ECE (offline) | Brier-fit T → hard ECE | ECE-fit T → hard ECE | authored-dev acc / hard acc |
|---|---|---|---|---|
| 4B soup50 raw | 1.70 → 0.110 | 1.95 → 0.100 | 2.05 → 0.074 | 64.7 / 69.4 |
| 4B soup50 rot4 (offline approx.) | 1.70 → 0.111 | 1.95 → 0.121 | 2.05 → 0.094 | 64.7 / 70.3 |
| 2B soup50 rot4 (offline approx.) | 1.65 → 0.124 | 1.95 → 0.126 | 1.90 → 0.096 (smoothed fit: 2.20) | 58.0 / 60.4 |
| 4B 2c-best raw | 1.30 → 0.148 | 1.30 → 0.148 | 1.65 → 0.082 | 64.0 / 71.2 |

Hard ECE on 111 items moves by ±0.02 between neighbouring T values, so single points are noisy; the direction (T ≈ 2 rather than
1.7) is consistent across candidates. New policy: **ECE-fit T on the authored dev** — 4B soup50 T 2.05, 2B soup50 T 2.20
(`reports/decision-p2c/calibration-p2c-{4b,2b}-soup50-final.json`, schema 1.1, the fit note records both temperatures). Served
verification (variants `cal2` / `rot4cal2`, pod GPUs 4-7 while the 9B trained, 12:05 UTC) — per-tier and pooled 10-bin ECE over the 231 public items:

| soup50, served | hard | original | easy | pooled (231) |
|---|---|---|---|---|
| 4B NLL-fit T 1.72, rot4 | 0.116 | 0.096 | 0.009 | **0.026** |
| 4B ECE-fit T 2.05, rot4 | 0.091 | 0.132 | 0.021 | 0.048 |
| 4B NLL-fit T 1.72, no rotations | 0.109 | 0.093 | 0.009 | 0.038 |
| 4B ECE-fit T 2.05, no rotations | 0.073 | 0.131 | 0.020 | 0.052 |
| 2B NLL-fit T 1.65, rot4 | 0.123 | 0.091 | 0.040 | **0.025** |
| 2B ECE-fit T 2.20, rot4 | 0.090 | 0.170 | 0.088 | 0.075 |

**Decision: keep the NLL-fit temperatures.** The official board's Calibration axis is `100·(1 − ECE/0.5)` on one pooled ECE
(`composite_v13/v14.py`), and the larger T trades hard-tier ECE for under-confidence on the original/easy tiers, which the pooled
metric punishes. The ECE-fit files are kept as `calibration-p2c-{4b,2b}-soup50-ecefit-alternative.json`. Fixing hard without
touching the easy tiers needs a non-linear confidence map (the server schema is temperature-only) → post-launch item.

## 2B (trained on the side lane 10:03–10:52 UTC, 620 steps; evaluation started 11:20 UTC on GPUs 4-6 via `eval2b_side.sh`, the main script's own eval functions)

| candidate | JevBench hard raw (cal) | original | easy | ImajevBench public (joint/text/visual) | gates |
|---|---|---|---|---|---|
| phase-2b 2B (reference) | 56.8 | | | 70.3 (83/—/95) | — |
| 2c best | 55.9, ECE 0.209 (cal 0.161) | 90.3 | 100 | 71.0 (80/19/99) | FAIL a.joint_items (80 < 81); all others PASS |
| 2c best, private-1 hidden | | | | **74.8 (59/24/68)** vs phase-2b 70.8 (55/22/66), ECE 0.094 vs 0.101 | hidden split: +8 items, joint +4 |
| 2c last | 56.8, ECE 0.233 (cal 0.168) | 88.9 | 100 | | |
| soup50 = ½ p2b + ½ 2c-best | **58.6** (0.176) | 91.7 | 100 | **71.7 (82/19/99)** | **SHIPPABLE, all gates pass** (state 68.5, pairs 100, unknown 12/14 = 85.7 as phase-2b, false abstention 1.19%, irrelevance 68.9) |
| soup50, private-1 hidden | | | | **74.3 (58/25/67)** vs phase-2b 70.8, ECE 0.089 | |

2B panels: reasoning dev 62.7 (phase-2b 58.9, v2.1 64.5 → most of the phase-2b dip recovered), state probe 69.0 (69.5), pairs 100 (100), irrelevance 69.8, phase-2b test 76.8, phase-2 test 67.1, judge-dev 79.3.

2B decision (revised 11:35 UTC): hard −1 item; public images +2 items (joint −3 / visual +4, the joint gate misses by one item); hidden split +8 items (joint +4, text +2, visual +2). The public joint miss looks like noise against the hidden split. 2B soup50 passes every gate and beats phase-2b on every tier → **2B ship candidate = soup50**.
Authored-dev T: 2c-best 1.76, soup50 1.65. Served variants for soup50: cal 58.6 / ECE 0.075; rot4 60.4 / 0.207; rot4+cal **60.4 / 0.123** (original 93.1 / 0.091, easy 100) → 2B serving configuration = soup50 + rot4 + authored-dev T (phase-2b 2B: 56.8 / 0.163).

## 9B (trained 10:52–12:31 UTC on GPUs 0-3, 1,018 steps; evaluated 12:31– on all 8 GPUs: main script on 4-7, side orchestrator on 0-3)

Best selection score 0.947 (mean of phase-2c dev + judge-dev). Authored-dev: 2c-best 72.7% (NLL T 1.43), last 72.0% (T 1.64), soup50 73.3% (T 1.75).

| candidate | JevBench hard raw (cal) | original | easy | ImajevBench public (joint/text/visual) | private-1 hidden | gates |
|---|---|---|---|---|---|---|
| phase-2b 9B (reference) | 68.5 raw / 69.4 rot4+cal (ECE 0.104) | 100 | | 82.8 | 84.2 (67/23/80), ECE 0.108 | — |
| 2c best | 66.7 (0.161); cal T 1.43 → 66.7 / 0.123 | 100 | 100 | 82.4 (96/29/105) | **85.1 (69/23/80)**, ECE 0.046 | SHIPPABLE, all gates pass (state 75.5, pairs 90, unknown 14/14, false abst. 0.48%, irrelevance 84.0) |
| 2c last | 68.5 (0.168); cal 68.5 / 0.114 | 100 | 100 | | | |
| soup50 = ½ p2b + ½ 2c-best | 69.4 (0.187); cal 69.4 / 0.090; rot4 69.4 / 0.174; rot4+cal 69.4 / 0.092 | 100 | 100 | 82.1 (96/29/104) | 84.7 (67/24/80), ECE 0.079 | SHIPPABLE, all gates pass |

Panels (2c-best): phase-2c test 85.5, judge-dev 28/29, phase-2b test 87.4, phase-2 test 80.3, reasoning dev 68.9 (phase-2b 67.4, v2.1 69.2 → recovered), state probe 75.5, pairs 90.0 (phase-2b 80.0), irrelevance 84.0.
9B decision (12:47 UTC): the pure 2c checkpoint does not gain on hard (66.7 < phase-2b 68.5) although it passes every gate and is best on the hidden split; **soup50 ships**: hard 69.4 / ECE 0.092 (rot4+cal; phase-2b 69.4 / 0.104), all gates, hidden 84.7 (phase-2b 84.2), public images 82.1 (82.8, −2 items, within gate). All three sizes therefore ship the phase-2c soup50 recipe.
soup50 gates (12:41 UTC): **SHIPPABLE, all pass** — ImajevBench 82.1 ≥ 81.8, visual 104 ≥ 104, joint 96 ≥ 94, state 73.5, pairs 90.0, unknown 14/14, false abstention 0.24%, irrelevance 84.0.

MLX parity (Mac, staged `imajev-release/hf/imajev-4b/mlx`): MLX 4b-soup50 ImajevBench: 229/279 = 82.1% tracks {'joint': (96, 122), 'text': (26, 37), 'visual': (107, 120)} (torch pod run 230/279 = 82.4%); argmax agreement 274/279 = 98.2%, mean |Δp| 0.006, max 0.22 → PASS.
MLX parity (Mac, staged `imajev-release/hf/imajev-2b/mlx`): MLX 2b-soup50 ImajevBench: 195/279 = 69.9% tracks {'joint': (81, 122), 'text': (18, 37), 'visual': (96, 120)} (torch pod run 200/279 = 71.7%); argmax agreement 271/279 = 97.1%, mean |Δp| 0.0074, max 0.20 → PASS.

## Pod closed (12:55 UTC)
the phase-2c pod terminated after ALL_DONE (12:46 UTC); uptime 6 h 20 min at $27.92/h ≈ $177. Everything is under `reports/decision-p2c/`:
`pod/<size>-delta/` (JevBench runs incl. raw responses, ImajevBench runs, calibration fits, gates, dev curves, best adapters in `train-best/`),
`pod/<size>-soup50/gates.json`, `pod/adapters/<size>-soup50/`, `pod/private1/<run>/score`, `pod/p2c-train-results.tgz` (the script's own
pack, 698 MB, incl. every eval panel's predictions), `pod/run.log`, `mlx-parity/<size>-soup50/`, `calibration-p2c-<size>-soup50-final.json`
(+ `-ecefit-alternative.json` for 4B/2B). No credential files remained on the pod (checked before termination).

## Final ship set (all three sizes = phase-2c soup50 + 4 rotations + authored-dev NLL-fit T), staged in `imajev-release/hf/imajev-<size>/`
| size | JevBench hard / ECE | original | ImajevBench public | private-1 hidden | phase-2b (previous) |
|---|---|---|---|---|---|
| 2B | 60.4 / 0.123 | 93.1 | 71.7 | 74.3 | 56.8 / 0.163, 70.3, 70.8 |
| 4B | 70.3 / 0.116 | 98.6 | 82.4 | 84.2 | 67.6 / 0.088, 82.4, 84.7 |
| 9B | 69.4 / 0.092 | 100 | 82.1 | 84.7 | 69.4 / 0.104, 82.8, 84.2 |
MLX parity (Mac, staged `imajev-release/hf/imajev-9b/mlx`): MLX 9b-soup50 ImajevBench: 230/279 = 82.4% tracks {'joint': (96, 122), 'text': (29, 37), 'visual': (105, 120)} (torch pod run 229/279 = 82.1%); argmax agreement 277/279 = 99.3%, mean |Δp| 0.0057, max 0.29 → PASS.
