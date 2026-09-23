# imajev v1.1 — 9B variant (23 Sept 2026)

Qwen3.5-9B + LoRA (r16/α32, language layers) + 255-code decision readout, trained from scratch (no v1 adapter) for one epoch on the
same manifest as the 2B (`data/manifests/decision-v1.1.jsonl`, sha256 `6948c2dce8867d08…`, 504,000 training decisions), lr 2e-4,
token budget 7,500 (no checkpointing, pad to 64), 4× H200 SXM (RunPod Secure, $18.36/h), 3,508 steps at ~55 examples/s, ~$60.
Best checkpoint step 3,100: dev loss 0.987 → 0.371 (2B: 1.21 → 0.404).

Adapter: `reports/decision-v1.1-9b/runs/h200x4/best` (161 MB + readout); MLX copy in `best-mlx`; temperatures `calibration-9b.json`.

Crash note: 8 SIGSEGVs in the first 980 steps, zero in the remaining 2,528 after removing the trainer's periodic
`faulthandler.dump_traceback_later` timer. That timer was the cause of the "random" segfaults in all three runs.

## Same test, three models (35,528 decisions of the v1.1 test set)

| | Base 2B | 2B v1.1 | Base 9B | **9B v1.1** |
|---|---:|---:|---:|---:|
| Accuracy (abstention credited) | 51.5% | 87.3% | 74.7% | **89.0%** |
| Image decisions (19,232) | 61.2% | 86.1% | – | 87.1% |
| Text decisions (16,296) | 40.0% | 88.7% | – | 91.2% |
| MMLU in test (never trained) | 33.6% | 53.5% | – | **73.7%** |
| False abstention on answerable | 20.9% | 1.6% | 5.0% | 1.4% |
| ECE (5 bins, raw) | 0.155 | 0.016 | 0.074 | **0.007** |
| Brier | 0.243 | 0.079 | 0.158 | 0.068 |
| Auto-accept at ≥0.95 | – | 61.2% @ 98.6% | – | 64.2% @ 98.8% |

Per-source: `eval/9b-report/report.md`. Largest in-distribution gains: aokvqa +6, sqid_esci +6, boolq +3.5, snli +3; marqo_gs −3, esci −1.6 (noise-level on 300–600 cases).

## Held-out panels (never trained on)

| Panel | n | 2B v1.1 | **9B v1.1** | Notes |
|---|---:|---:|---:|---|
| v1 image exam, held-out sources | 4,989 | 50.7% | **55.9%** | first movement on this number in four runs (v1 49.8, v1s2 50.2) |
| · heldout_abstention (MM-UPD/TUBench) | 1,500 | 65.2% | 73.3% | |
| · heldout_fashionpedia | 983 | 47.7% | 58.1% | |
| · heldout_countqa | 1,167 | 35.6% | 39.9% | counting stays hard |
| · heldout_livewild (quality) | 1,162 | 51.6% | 52.2% | |
| · heldout_pairs (two images) | 177 | 37.9% | 26.6% | worse; two-image training data is thin |
| v1 image exam, trained sources | 19,232 | 86.0% | 87.1% | |
| typed-decisions test | 2,000 | 58.1% (ECE 0.032) | **66.2%** (ECE 0.045) | untrained 9B already 62.0%; Laya fine-tuned 76.6% |
| SST-5 test | 2,210 | 47.5% | 51.0% | |
| MMLU-1000 text-only | 1,000 | 53.9% | **73.8%** | |
| MMLU-1000 with an irrelevant image | 1,000 | 49.8% (−4.1) | 72.1% (−1.7) | the 9B ignores an unrelated image much better |
| Irrelevance panel, all | 2,823 | 64.3% (ECE 0.082) | 79.6% (ECE 0.017) | |

## JevBench public splits (local MLX serving, calibrated)

| Split | 2B v1.1 | 9B v1.1 | Leaderboard context (hard-220) |
|---|---:|---:|---|
| original (72) | 88.9% (ECE 0.080) | **98.6%** (ECE 0.072) | |
| easy (48) | 100% | 100% | |
| hard, public 111 | 43.2% (ECE 0.254) | 42.3% (ECE 0.313) | Jev 74.1, SemIf-4B 59.5, jev-local-9B 59.1, decider-2b 47.3, kev-0.6B 40.0 |
| p50 latency (Mac Studio) | 79 ms | 298 ms | |

Hard by family (9B): routing_hard 5/5, adversarial 5/6, judge_hard 11/17, trap 5/8, probability 6/10, long_policy 5/19, multi_hop 5/18,
tradeoff 2/6, ambiguous 2/7, temporal_numeric 1/15, score-type 1/6. Identical shape to the 2B: the hard split is multi-step rule
application, and a single-pass decision readout does not acquire that from 2B → 9B scale. Note that `jev-local (Qwen3.5-9B)` on the
leaderboard reaches 59% hard with a generate-then-parse prompt on the same base model, so the gap is the readout recipe, not the weights.

## Reading

- Scale buys generalisation where the task is recognition and judgement: +5 on held-out image sources, +20 on MMLU, +8 on typed
  workflows, +10 on JevBench original, and a much smaller irrelevant-image penalty. Calibration is better still (ECE 0.007).
- Scale buys nothing on multi-hop reasoning items in this recipe. Getting those needs either a reasoning pass before the readout or
  distilling from a model that reasons.
- Two-image comparison got worse; it is under-represented in the mixture (17k of 504k) and was never a strength.
- Latency: ~300 ms per 3-question text request on the Mac vs 80 ms for the 2B. The 9B is the quality tier and a teacher for
  distillation into the 2B (soft targets on the same 504k decisions), not the default.

## Addendum (23 Sept): fine-tuning erased the 9B's reasoning on JevBench hard

Untrained base models through the same readout (no adapter, MLX, uncalibrated), JevBench public splits:

| Model | hard (111) | original (72) | our test set: image / text |
|---|---:|---:|---:|
| Qwen3.5-2B base, readout | 45.9% | 58.3% | 61.2% / 40.0% |
| **2B v1.1 (tuned)** | 43.2% | 88.9% | 86.1% / 88.7% |
| Qwen3.5-9B base, readout | **64.9%** (ECE 0.149) | 91.7% | 70.0% / 80.1% |
| **9B v1.1 (tuned)** | 42.3% | 98.6% | 87.1% / 91.2% |
| 9B base + tuned, probabilities averaged | 54.1% | 97.2% | – |

Per family on hard, base → tuned 9B: multi_hop 13 → 5 of 18, temporal_numeric 7 → 1 of 15, ambiguous 7 → 2 of 7, long_policy 9 → 5 of 19,
trap 8 → 5 of 8; judge_hard 9 → 11 and probability 5 → 6 improved. The base 9B would sit around 4th on the leaderboard's hard axis
(djev 69.5, Winnow-12B 70.9, SemIf 59.5) with zero training. Our mixture (short, pattern-style decisions, one epoch, lr 2e-4, LoRA on all
language layers) buys +17/+11 points on our own image/text test and abstention behaviour, and pays with ~22 points of multi-step reasoning.
Data: `jevbench-base9b/`, `../decision-v1.1/jevbench-base2b/`.

Implications for the next run: regularise toward the base (blend targets with the base model's own distributions, i.e. self-distillation),
train far shorter (the 9B's dev loss had most of its gain by step ~1,200 of 3,508), restrict LoRA (lower rank, attention-only), and add a
reasoning-style dev set for early stopping. At inference, averaging base and tuned distributions recovers half the loss at 2× compute.
