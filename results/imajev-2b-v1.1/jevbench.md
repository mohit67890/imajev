# JevBench (public splits) — imajev v1.1 2B

Run 22 Sept 2026 with the official harness (`fstandhartinger/jevbench`, MIT; cloned at `.cache/external/jevbench`, run with the
stdlib-only `.venv-jevbench`), adapter `typesafe` pointed at the local playground (`http://<host>/v1/systemone`, MLX on the
Mac Studio, v1.1 adapter + readout, v1.1 temperature calibration). No code changes to the harness; results in `original/`, `hard/`, `easy/`
(`results.jsonl`, `summary.json`, raw responses). 231/231 requests answered, schema validity 1.0, cost $0.

| Split | n | Accuracy | Brier | ECE | p50 latency | Notes |
|---|---:|---:|---:|---:|---:|---|
| original (6 families) | 72 | **88.9%** | 0.199 | 0.080 | 79 ms | paraphrase-pair agreement 88.9% |
| hard, public part | 111 | **43.2%** | 0.714 | 0.254 | 339 ms | leaderboard hard numbers are on 220 items (111 public + 109 private) |
| easy | 48 | 100% | 0.000 | 0.005 | 81 ms | |

Original by family: routing 12/12, ordinal 12/12, extraction 11/12, intent 11/12, policy 10/12, adequacy 8/12.
Hard by family: routing_hard 5/5, trap 6/8, probability 6/10, judge_hard 10/17, adversarial 3/6, long_policy 7/19, multi_hop 5/18,
tradeoff 2/6, temporal_numeric 3/15, ambiguous 1/7. By type on hard: choice 27/67, noul 21/38, score 0/6.

## Where that sits (JevBench v1.2 leaderboard, hard-220 accuracy / ECE on hard)

| System | Params | Hard acc | ECE hard | JevBench score |
|---|---|---:|---:|---:|
| Jev 1.13.0 (TypeSafe, closed) | ? | 74.1% | 0.061 | 74.4 |
| djev (Maisa) | ~gemma | 69.5% | 0.175 | 73.0 |
| SemIf / OpenJev | Qwen3.5-4B | 59.5% | 0.121 | 73.1 |
| jev-local | Qwen3.5-9B | 59.1% | 0.146 | 61.8 |
| system-one-open | Gemma 4 E2B LoRA | 49.1% | 0.257 | 66.6 |
| decider-2b (Mapika) | 2B | 47.3% | 0.322 | 61.7 |
| decision-machine-1 | ? | 46.8% | 0.158 | 68.3 |
| **imajev v1.1 (public 111 only)** | **Qwen3.5-2B** | **43.2%** | **0.254** | not computed (needs the private split) |
| kev 0.6B | 0.6B | 40.0% | 0.269 | 62.5 |
| Laya | ModernBERT-large | 34.1% | 0.206 | 54.4 |

Reading: on the ordinary splits the 2B is at ceiling (89% original, 100% easy, 80 ms). On the hard split it lands with the other
2B-class systems and above Laya and kev, well below Jev and the 4B+ open systems. The composite JevBench score cannot be computed
without the private 109 hard items; on speed and cost it would score near the top (80 ms local, $0).

## What the hard split exposes

- **Overconfidence off-distribution**: ECE 0.25 and Brier 0.71 on hard vs 0.016 on our own test. The v1.1 temperatures were fitted on
  in-distribution text; hard reasoning items need a different (higher) temperature or a confidence head that knows the difference.
- **Reasoning families**: multi-hop (28%), temporal/numeric (20%), long policy (37%), ambiguous (14%). These need chain-of-evidence,
  not a bigger readout; Jev itself only gets 27% on temporal/numeric.
- **Ordinal on hard: 0/6.** Worth a look at the raw responses before the 9B run.
- Two families are already Jev-level: routing_hard 5/5 and trap 6/8.

## Reproduce

```sh
cd .cache/external/jevbench
TYPESAFE_API_KEY=local ../../.venv-jevbench/bin/python -m jevbench.cli run --tasks datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://<host> --model imajev-v1.1 --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 \
  --results ../../reports/decision-v1.1/jevbench/hard/results.jsonl --raw-dir ../../reports/decision-v1.1/jevbench/hard/raw \
  --manifest ../../reports/decision-v1.1/jevbench/hard/manifest.json
../../.venv-jevbench/bin/python -m jevbench.cli summarize --tasks datasets/public/hard.jsonl \
  --results ../../reports/decision-v1.1/jevbench/hard/results.jsonl --public-export ../../reports/decision-v1.1/jevbench/hard/summary.json
```

## Option-order averaging (pijev-style), 4 cyclic rotations — `reports/decision-v1.1/jevbench-rot4/`

pijev (TypeLLM/pijev, Apache-2.0) wraps Jev's API and averages probabilities over several option orderings; imajev has the same
mechanism built in (`--rotations`). Measured on the same public splits, single pass vs 4 rotations:

| Split | Accuracy 1× → 4× | ECE 1× → 4× | Brier 1× → 4× | p50 latency 1× → 4× |
|---|---:|---:|---:|---:|
| hard (111) | 43.2% → 47.7% | 0.254 → 0.218 | 0.714 → 0.717 | 339 → 466 ms |
| original (72) | 88.9% → 87.5% | 0.080 → 0.039 | 0.199 → 0.192 | 79 → 233 ms |

Averaging buys accuracy and calibration where the model is unsure (hard) and mostly calibration where it is already confident, at
3–4× the compute. It is an inference-time option, not a substitute for a better model.

The six hard ordinal misses are all multi-step rule application (counting roster rest violations, applying bursary bands, licence
shortfalls, severity runbooks with amendments); probability mass is spread across levels rather than confidently wrong.
