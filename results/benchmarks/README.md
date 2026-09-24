# Benchmark registry — every JevBench and ImajevBench run kept for the launch report

Rule: every run keeps its `results.jsonl`, `manifest.json` (settings, served model id, adapter, token budget), `summary.json`
and the raw responses, so any number in the launch material can be traced to a file. Interfaces are never pooled: direct
option scoring (imajev's single-pass mode), structured generation (JSON schema through an OpenAI-compatible server), and
reasoning-on generation are separate rows.

## JevBench public (hard 111 / original 72 / easy 48)

| System | Interface | hard | original | easy | Date | Files |
|---|---|---|---|---|---|---|
| imajev-2b phase 2 (best) | typesafe server, raw | 55.0 / ECE 0.234 | 91.7 | 100 | 2026-09-24 | `reports/decision-p2/pod/train-out/2b/jevbench-{best,last}/` |
| imajev-4b phase 2 (best / last) | typesafe server, raw | 62.2 / 63.1 | 98.6 | 100 | 2026-09-23 | `reports/decision-p2/pod/train-out/4b/jevbench-*/` |
| imajev-9b phase 2 (best / last) | typesafe server, raw | 67.6 / 66.7 | 98.6 | 100 | 2026-09-23 | `reports/decision-p2/pod/train-out/9b/jevbench-*/` |
| imajev-2b phase 2b (best; rot4+cal, authored-dev T 1.61) | typesafe server | 56.8 / 0.163 | | | 2026-09-24 | `reports/decision-p2b/pod/train-out-p2b/2b/jevbench-<tag>-<variant>/`; `reports/decision-p2b/README.md` |
| imajev-4b phase 2b (best; cal, authored-dev T 2.32) | typesafe server | 67.6 / 0.088 | | | 2026-09-24 | `reports/decision-p2b/pod/train-out-p2b/4b/jevbench-<tag>-<variant>/` |
| imajev-9b phase 2b (step 260; rot4+cal, authored-dev T 2.19) | typesafe server | 69.4 / 0.104 | | | 2026-09-24 | `reports/decision-p2b/pod/train-out-p2b/9b/jevbench-<tag>-<variant>/` |
| imajev-4b phase 2c best (step 260, soft-target delta from 2b; raw / pod-fold cal T 1.045) — FAILS image gate (ImajevBench 79.6) | typesafe server | 71.2 / 0.165 (cal 0.159) | 98.6 | 100 | 2026-09-24 | `reports/decision-p2c/pod/4b-delta/jevbench-best-{raw,cal}/` |
| imajev-4b phase 2c last (step 747) | typesafe server | 70.3 / 0.162 (cal 0.126) | 100 | 100 | 2026-09-24 | `reports/decision-p2c/pod/4b-delta/jevbench-last-{raw,cal}/` |
| **imajev-4b phase 2c soup50** (½ phase-2b + ½ 2c-best adapters; raw / cal T 1.72 authored-dev / rot4 / rot4+cal) — ship candidate, all gates pass, ImajevBench 82.4 | typesafe server | 69.4 / 0.164; cal 69.4 / 0.109; rot4 70.3 / 0.171; **rot4+cal 70.3 / 0.116** | 98.6 | 100 | 2026-09-24 | `reports/decision-p2c/pod/4b-delta/jevbench-soup50-{raw,cal,rot4,rot4cal}/`; adapter `reports/decision-p2c/pod/adapters/4b-soup50/` |
| imajev-4b phase 2c soup75 (¼ + ¾) — fails image gate (80.6) | typesafe server | 69.4 / 0.142 | 98.6 | 100 | 2026-09-24 | `reports/decision-p2c/pod/4b-delta/jevbench-soup75-raw/` |
| imajev-2b phase 2c best / last (raw; pod-fold cal T 1.31 / 1.43) — best fails joint gate by 1 item | typesafe server | 55.9 / 0.209 (cal 0.161); last 56.8 / 0.233 (cal 0.168) | 90.3 / 88.9 | 100 | 2026-09-24 | `reports/decision-p2c/pod/2b-delta/jevbench-{best,last}-{raw,cal}/` |
| **imajev-2b phase 2c soup50** (raw / cal T 1.65 authored-dev / rot4 / rot4+cal) — ship candidate, all gates pass, ImajevBench 71.7 | typesafe server | 58.6 / 0.176; cal 58.6 / 0.075; rot4 60.4 / 0.207; **rot4+cal 60.4 / 0.123** | 91.7 (rot4 93.1) | 100 | 2026-09-24 | `reports/decision-p2c/pod/2b-delta/jevbench-soup50-{raw,cal,rot4,rot4cal}/`; adapter `reports/decision-p2c/pod/adapters/2b-soup50/` |
| imajev-9b phase 2c best / last (raw; pod-fold cal T 1.02 / …) — best passes all gates (ImajevBench 82.4) but hard below phase-2b | typesafe server | 66.7 / 0.161 (cal 0.123); last 68.5 / 0.168 (cal 0.114) | 100 | 100 | 2026-09-24 | `reports/decision-p2c/pod/9b-delta/jevbench-{best,last}-{raw,cal}/` |
| **imajev-9b phase 2c soup50** (raw / cal T 1.75 authored-dev / rot4 / rot4+cal) — ship candidate, all gates pass, ImajevBench 82.1, private-1 84.7 | typesafe server | 69.4 / 0.187; cal 69.4 / 0.090; rot4 69.4 / 0.174; **rot4+cal 69.4 / 0.092** | 100 | 100 | 2026-09-24 | `reports/decision-p2c/pod/9b-delta/jevbench-soup50-{raw,cal,rot4,rot4cal}/`; adapter `reports/decision-p2c/pod/adapters/9b-soup50/` |
| Qwen3.6-35B-A3B frozen, thinking off | vLLM + openai_compat, JSON schema | 61.3 / 0.329 | 88.9 | 100 | 2026-09-24 | `jevbench/qwen3.6-35b-a3b-frozen/nothink/` |
| Qwen3.6-35B-A3B frozen, thinking on (16k budget) | vLLM + openai_compat, reasoning parser | **97.3** / 0.045 | 100 | 100 | 2026-09-24 | `jevbench/qwen3.6-35b-a3b-frozen/think/` |
| mojev 0.85B (MoLeMo-Lab) | its own TypeSafe-compatible server, typesafe adapter | 33.3 / 0.268 | 63.9 | 91.7 | 2026-09-24 | `jevbench/mojev-0.85b/` |
| cua-s1-4b-0.2 text LoRA (off-domain: GUI-action specialist) | vLLM LoRA on Qwen3.5-4B + openai_compat, thinking off; adapter keys remapped `model.layers`→`model.language_model.layers` (first run without the remap was a silent no-op, identical to base on 111/111; kept in `jevbench/cua-s1-4b-0.2-noop/`) | 52.3 / 0.456 | 88.9 / 0.111 | 100 / 0.001 | 2026-09-24 | `jevbench/cua-s1-4b-0.2/` |
| Qwen3.5-4B base, structured generation, thinking off | vLLM + openai_compat, JSON schema | 48.6 / 0.448 | 80.6 / 0.181 | 100 / 0.005 | 2026-09-24 | `jevbench/qwen3.5-4b-base-generation/` |
| JevK5 v0.2.0 (alibiserikbay/JevK5, Qwen3.5-4B merged LoRA) — OUR run, same protocol as imajev (its `jevk5-serve`, typesafe adapter, serial, native probs) | typesafe server | **73.9** / 0.073 | 97.2 / 0.138 | 100 / 0.039 | 2026-09-24 | `jevbench/jevk5-v0.2.0/` |
| Hopper (HopitAI/hopper LoRA on Qwen3.5-4B, shipped calibration map) — OUR run, same protocol (its `hopper-serve`, typesafe adapter, serial, native probs) | typesafe server | 67.6 / **0.050** | 95.8 / 0.103 | 100 / 0.034 | 2026-09-24 | `jevbench/hopper/` |
| Winnow-12B Q8 | not measured: its server rejects the harness's model name ("Unknown model; use Winnow-12B or the configured alias"); needs `--model Winnow-12B` in the run; pod closed before a retry | | | | 2026-09-24 | — |
| Published by others (not our runs) | their reports | jevk5 73.9, reflex-27B 76.6, xor 35B-A3B 77.5, Eikos-27B 82.9 (self-reported) | | | | see `imajev-release/writing/` competitor notes |

## ImajevBench v2.0-lite (279-item test split) — `../../../imajev-release/bench/LEADERBOARD.md` holds the full table with CIs, tracks, controls and the paired cluster tests
Direct option scoring: imajev-9b 81.0 (v1.1) → 81.4 (phase 2); Qwen3.5-9B base 76.7; Qwen3.5-4B base 70.6 → imajev-4b phase 2 80.6; imajev-2b 63.1 (v2.1) → 68.5 (phase 2); Qwen3.5-2B base 60.2.
Structured generation: Gemini 3.1 Pro 99.6, GPT-5.6 Luna 99.3, GPT-5.4 98.9, Gemini 3.8 Flash 98.2, Grok 4.3 91.4, Qwen3.5-9B 74.6, Qwen3.5-4B 67.4, Gemma 4 E4B-it 60.2, Qwen3.5-2B 59.9, Gemma 4 E2B-it 58.8.
Phase 2b (pod, torch direct scoring, full rotations): imajev-2b 70.3, imajev-4b 82.4, imajev-9b 82.8 — `reports/decision-p2b/pod/train-out-p2b/<size>/imajevbench-best/`.
Phase 2c (2026-09-24, same backend): imajev-4b 2c-best 79.6 (joint 92/122, text 24/37, visual 106/120; FAILS gate), 2c-last 79.9, soup50 **82.4** (97/26/107, all gates pass), soup75 80.6; imajev-2b 2c-best 71.0 (80/19/99; joint gate miss by 1), soup50 **71.7** (82/19/99, all gates pass); imajev-9b 2c-best 82.4 (96/29/105, all gates pass), soup50 **82.1** (96/29/104, all gates pass) — `reports/decision-p2c/pod/<size>-delta/imajevbench-<tag>/`, gates `reports/decision-p2c/pod/<run>/gates.json`.
Hidden split private-1 (202 items, aggregates only, GPU 7 runner, same backend): phase-2b 2B 70.8 / 4B 84.7 / 9B 84.2; phase-2c 4B 2c-best 83.2, 4B soup50 84.2, 2B 2c-best 74.8, 2B soup50 74.3, 9B 2c-best 85.1, 9B soup50 84.7 — `reports/decision-p2c/pod/private1/<run>/score`.
Controls: imajev-2b no-image (abstains 258/279), no-state 51.6, family-majority 35.8, image-blind best on contrast sets 66.7%.
Run files: `imajev-release/bench/records/`, per-run predictions and manifests under `imajev-release/bench/` and `reports/imajev-bench/` (H100 bench pod pull); phase-2 rows under `reports/decision-p2/pod/train-out/<size>/imajevbench-best/`.
Pending rows: Gemma 4 E2B/E4B and SmolVLM2-2.2B **direct option scoring** (MLX-only backends; Mac runs were stopped to avoid crashes) — marked pending on the leaderboard; mojev and cua-s1 on ImajevBench need a TypeSafe-protocol provider in the harness (follow-up).

## fastino/fast-decisions (DEV split only: 17 domains × 100 rows, 2,900 heads; the 300/domain test split behind their board is held out)
| System | Setup | Domain avg | Pooled heads | Date | Files |
|---|---|---|---|---|---|
| imajev-4b 1.0 (phase 2c soup50) | MLX playground server, rot4, raw; one generic prompt (`Which {task} label fits this text?`, choice over full label list); multi-label heads = one noul per label, set = noul > 0.5 (top label if empty); exact set match | 59.3 | 58.8 ±1.8 (1704/2900) | 2026-09-25 | `results/benchmarks/fast-decisions/imajev-4b/` (public repo) |
Not comparable to their board (test split: GLiNER2.5-Decide 60.2, JevK5 57.6, SemIf 4B 56.4). Weak spots: multi-label heads (product_area 12, genres 26, aspects 42), ticket urgency 29, support_topic 38, paper_field 41.

## DecisionBench 1.0 (Hanno-Labs/decision-bench, 23,900 rows, official harness `run-system-one-http`)
| System | Setup | Primary acc | Scored-row acc | Coverage | ECE | Date | Files |
|---|---|---|---|---|---|---|---|
| imajev-4b 1.0 (HF rev 712891d1) | torch server @ public 4e623a8, rot4 + shipped calibration, `--max-input-tokens 32768`, `--max-candidates 254`; 1×H100 | **77.5** | 79.3 | 97.75% (537 × 255-candidate rows unsupported, 0 errors) | **0.024** | 2026-09-25 | `decisionbench/` (+ raw run in `decisionbench-run/`, not exported) |
Rank on 2026-09-25: 3rd of 55 (after Bosun v3.1 1.7B 84.9 / 0.6B 81.2; above Winnow-12B 76.7, Jev 1.13 72.0); lowest ECE of all records. Upstream-dataset overlap disclosed in `decisionbench/contamination-check.md` (banking77 → RouteFinancial scored 87.1%, below RouteGeneralAssistant 96.7%).

## Reading notes for the launch report
- Frozen 35B-A3B with reasoning at 97.3 hard shows the hard split is largely a reasoning problem; single-pass small models (ours) are a different interface with milliseconds of latency — state both.
- mojev and cua-s1 are the two open "same-interface" small models found so far; neither has abstention or image grounding.

## The OFFICIAL JevBench board (v1.4.1, benchmarkheaven.com/jev-models) — how our numbers map
The official score is not hard-split accuracy. It is the harmonic mean of four 0–100 axes — Intelligence (chance-corrected,
tiers hard 30% / standard 28% / judge 28% / easy 14%, blended 80/20 with 308 fresh SEALED decisions, minus a public-to-sealed
gap penalty), Calibration (hard-tier ECE + distribution fidelity), Speed (p50/p95 latency, self-hosted ×2 + 0.15 s), Cost
($/1,000 decisions) — with (axis/50)² gates below 50 on Intelligence, Speed and Cost. Method: `.cache/external/jevbench/docs/METHOD-v1.4.md`,
code `jevbench/composite_v14.py`; per-system artifact `results/v1.4/jevbench-v1.4-results.json`.

Official MEASURED tier accuracies (their runs; hard = 220 items incl. 109 held out; ours below is the 111 public only):

| System | Score | Intel | Calib | Speed | Cost | hard | standard | judge | easy | sealed 308 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 (API) | 63.3 | 53 | 76 | 83 | 52 | 74.1 | 99.0 | 94.5 | 100 | 36.7 |
| JevK5 v0.2.0 | 62.0 | 49 | 75 | 91 | 60 | 70.0 | 95.8 | 94.5 | 100 | 33.1 |
| Hopper | 59.4 | 48 | 79 | 87 | 59 | 65.0 | 96.9 | 83.6 | 100 | 34.1 |
| Winnow-12B Q8 | 55.6 | 48 | 65 | 82 | 53 | 70.9 | 96.9 | 91.1 | 100 | 33.1 |
| reflex 4B | 54.0 | 47 | 70 | 68 | 60 | 63.2 | 94.8 | 97.3 | 100 | 28.2 |
| djev | 52.2 | 47 | 55 | 91 | 58 | 69.5 | 97.9 | 93.2 | 100 | 29.9 |
| SemIf (Qwen3.5-4B) | 47.7 | 44 | 67 | 84 | 59 | 59.5 | 97.9 | 95.2 | 100 | 26.3 |
| local-jev Qwen3.5-4B | 46.8 | 44 | 73 | 75 | 56 | 60.5 | 95.8 | 95.9 | 100 | 26.0 |
| decider-35b-a3b | 41.2 | 47 | 65 | 81 | 45 | 65.5 | 96.9 | 91.1 | 100 | 31.5 |
| kev 4B | 36.1 | 42 | 40 | 76 | 62 | 42.3 | 91.7 | 85.6 | 100 | 22.4 |
| **imajev-4b phase 2b (ours, public items only)** | est. 54–61 | est. 47–50 | est. 55–62 | est. 86 | est. 60 | **67.6** (111 public) | 98.6 (72 public) | not public | 100 (48 public) | unmeasured |

Notes: the "hard 73.9 / 76.6 / 77.5 / 82.9" figures quoted earlier for jevk5 / reflex-27B / xor / Eikos come from those projects'
own READMEs on other splits or self-runs; the official board measures jevk5 at 70.0 and reflex 4B at 63.2, and xor / Eikos are
not on the board. Every system collapses to 22–37% on the sealed 308 and pays the gap penalty; that, not the public hard split,
now decides rank. Entry: the maintainers run entrants themselves on their RunPod GPUs at the author's request (GitHub issues on
fstandhartinger/jevbench), pinned Apache-2.0 package + the author's server, serially; self-operated endpoints no longer receive
held-out items. Post-launch action: request a run of imajev-2b/4b/9b with the playground server (`/v1/systemone`).

## To benchmark next (same protocol, next pod): CLM-8B (Contrastive-LM/CLM, Apache-2.0)
Dual-encoder "System One" (frozen Qwen3-8B + 20M-param state/action projection heads, bidirectional InfoNCE; 60M Nemotron Q&A pairs +
30M Gemini-generated hard negatives + 1M agent trajectories). Serves `/v1/systemone` via `clm-serve`; option embeddings cacheable →
claims up to 9× lower latency than Jev at parity on computer-use / gaming / tool-calling; no JevBench or Decision Index numbers, no
images (roadmap). Head `Contrastive-LM/CLM-v0.1-8B` (75 MB). Expect strong easy/standard and Retrieval-style tasks, weaker hard
(multi-hop / numeric / traps need state–option interaction a dual encoder cannot model). Run: hard/original/easy + ImajevBench text track.
Provenance note for our datasheet: CLM's synthetic negatives come from a paid API (Gemini 2.5 Flash-Lite); imajev's data does not.

## To benchmark next: Eikos-4B and Eikos-27B (caiovicentino/eikos; HF caiovicentino1/Eikos-{4B,27B} + FP8/INT4/MLX; MIT deltas on Qwen3.5)
Self-reported (their harness, vLLM 0.30, batching + prefix cache): JevBench public original/hard **4B 91.7 / 72.1**, **27B 100 / 82.9**;
Jev in their table 98.6 / 73.0. Hard-split ECE 0.049 (4B) / 0.051 (27B); at ≥90% confidence 34.7% of decisions taken with 2.5% error (4B).
Long-context: 88.3 (27B) / 74.2 (4B) with the decision hidden in 64k tokens. No images. Serving: `bash serve_vllm.sh <MODEL_DIR> 8001`
then `python serve.py --model <MODEL_DIR> --vllm-url http://<host> --port 8000` (TypeSafe `/v1/systemone`). Official board: pending.
Recipe (worth adopting): LoRA r64, 1 epoch, lr 1e-4; SOFT cross-entropy on option-LETTER logits with option PERMUTATION during training
(order invariance learned, no rotations at inference); auxiliary rationale loss (0.3); PT↔EN view-consistency KL (0.5); programmatic
exact-answer families (probability, temporal, financial, trade rules, entity sentiment); teachers Qwen3.8-27B ("blind labeling") and
GLM-5.3-Flash for PT/policy items; 8-gram decontamination vs JevBench. Dataset `caiovicentino1/eikos-decisions` (CC-BY-4.0; 21.2k train /
1.22k val / 1.19k held-out core + 1.58k long-context + 1.62k PT/EN views; 28 families; fields incl. `target_probs`, `teacher_probs`,
`rationale`). Caveat for OUR use of that dataset: GLM-5.3-Flash rows are API-labelled (our no-paid-API-outputs rule) → usable only after
filtering to Qwen3.8-27B-labelled and programmatic rows, with the CC-BY attribution. Run both sizes same-protocol on the next pod.
