# imajev v1.1 — results (22 Sept 2026)

Qwen3.5-2B + LoRA (r16/α32, language layers) + 255-code decision readout, initialised from the v1 adapter and trained one epoch
on `data/manifests/decision-v1.1.jsonl` (sha256 `6948c2dce8867d08…`): 504,000 training decisions = 200,000 text (15 licence-verified
sources) + 300,000 image (stratified subsample of all 21 trainable v1 image sources, policy A) + 4,000 photo-vs-listing contradictions.
4× H100 SXM (RunPod Secure, $13.96/h), 2,495 optimiser steps at ~118 examples/s, 9 SIGSEGV resumes from 20-step checkpoints, no data
skipped. Final checkpoint (step 2,495) is the best: dev loss 1.21 → 0.404 (83.25% on 400 dev cases).

Adapter: `reports/decision-v1.1/runs/h100x4/best` (PEFT + `decision_readout.safetensors`); MLX copy in `best-mlx`.
Temperatures: `reports/decision-v1.1/calibration-v1.1.json` (fitted on the 25,217-decision text calibration fold; buckets by type × option count).

## v1.1 test set (35,528 decisions: 19,232 image + 16,296 text), paired base vs v1.1

| Metric | Base 2B | v1.1 |
|---|---:|---:|
| Accuracy (abstention credited when it is the target) | 51.5% | **87.3%** |
| Accuracy on answerable cases | 50.1% | 87.0% |
| False abstention on answerable cases | 20.9% | 1.6% |
| Correct abstention: false premise | 224/699 | 667/699 |
| Correct abstention: insufficient evidence | 1821/2454 | 2297/2454 |
| Correct abstention: mismatched reference | 26/273 | 271/273 |
| Correct abstention: not listed | 1347/2458 | 1993/2458 |
| Brier | 0.243 | 0.079 |
| ECE (5 bins, raw confidence) | 0.155 | **0.016** |
| Auto-accept at raw confidence ≥ 0.95 | – | 61.2% of cases at 98.6% precision |

Paired: 13,782 fixed, 1,050 broken (McNemar p < 1e-4). Full per-source table: `eval/v1.1-report/report.md`.

By kind: image 61.2% → 86.1%; text 40.0% → 88.7%. Text sources (base → v1.1): banking77 2.0 → 91.7, clinc150 0.9 → 91.2,
massive 1.3 → 86.1, civil_comments 35.4 → 96.5, goemotions 18.2 → 83.9, nvidia_aegis 8.9 → 87.8, snli 39.1 → 90.0, boolq 54.1 → 88.0,
fever 76.7 → 97.2, squad2 90.7 → 99.7, esci 31.8 → 65.3, **MMLU (never trained) 33.6 → 53.5**.

## Held-out panels (never trained on)

| Panel | n | Base 2B | v1.1 | Notes |
|---|---:|---:|---:|---|
| v1 image exam, held-out sources (5 sources) | 4,989 | 39.4% | 50.7% | v1 adapter: 49.8% — no regression from adding text |
| v1 image exam, trained sources | 19,232 | 57.6% | 86.0% | v1 adapter: 85.8% |
| v1 exam held-out abstention targets | 1,622 | – | 900 correct | v1: 849 |
| typed-decisions test (Jev-style workflows, 5 questions each) | 2,000 | 37.3% (ECE 0.140) | 58.1% (ECE 0.032; calibrated 0.039) | v1: 55.8%; Laya fine-tuned 76.6%; paired-family bootstrap Δ +0.21 [0.14, 0.27] |
| SST-5 test (5-level ordinal, unlicensed, eval-only) | 2,210 | 21.0% (ECE 0.179) | 47.5% (ECE 0.026) | chance 20% |
| MMLU-1000 text-only | 1,000 | 33.6% | 53.9% | 4 options + unknown |
| MMLU-1000 with an irrelevant image attached | 1,000 | – | 49.8% | Δ −4.1 pts [−6.2, −2.1]: the model still leaks a little attention to an unrelated image |
| Irrelevance panel, relevant image tasks (14 families) | 823 | 41–88% | 77–100% | e.g. unanswerable VQA 67.7 → 96.1, reference colour 35.3 → 100 |

Release gates (`eval/calibrated-heldout-typed.json`, on typed-decisions): ECE and auto-accept precision pass; auto-accept coverage (5.1%),
choice/noul accuracy (63%) and score MAE (0.67) do not meet the gate thresholds set for a Jev-parity release. Those thresholds were written for
a 9B-class result; the 2B is the latency tier.

## What changed vs v1

- Text-only requests (no image) are first-class; images optional, up to two.
- 255 options via the trained readout (v1: 25, single letters); 32 KB states; 4,096-token requests.
- Multi-question requests (up to 8 fields) trained jointly.
- Calibrated temperatures per (type, option count).
- Image behaviour unchanged within noise (86.0 vs 85.8 trained; 50.7 vs 49.8 held-out).

## Known gaps

- Held-out image sources still ~51%: the image ceiling is the recipe/model size, not data (v1 → v1.1 flat).
- typed-decisions 58%: zero-shot on unseen workflow schemas; a 9B teacher and distillation are the next lever.
- Irrelevant-image penalty of ~4 points on MMLU.
- Data-licence posture is v1's (annotation licences; images under upstream terms); see `reports/v1.1-datasets/image-policy-a.json`.
