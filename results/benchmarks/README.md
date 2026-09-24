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
| imajev-{2b,4b,9b} phase 2b (best/last × raw, rot4, cal, rot4cal) | typesafe server | pending (running) | | | 2026-09-24 | `reports/decision-p2b/pod/train-out-p2b/<size>/jevbench-<tag>-<variant>/` (after pull) |
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
Controls: imajev-2b no-image (abstains 258/279), no-state 51.6, family-majority 35.8, image-blind best on contrast sets 66.7%.
Run files: `imajev-release/bench/records/`, per-run predictions and manifests under `imajev-release/bench/` and `reports/imajev-bench/` (H100 bench pod pull); phase-2 rows under `reports/decision-p2/pod/train-out/<size>/imajevbench-best/`.
Pending rows: Gemma 4 E2B/E4B and SmolVLM2-2.2B **direct option scoring** (MLX-only backends; Mac runs were stopped to avoid crashes) — marked pending on the leaderboard; mojev and cua-s1 on ImajevBench need a TypeSafe-protocol provider in the harness (follow-up).

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
