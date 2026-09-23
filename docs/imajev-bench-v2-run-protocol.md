# imajevBench v2.0-lite: local-model evaluation protocol (23 Sept 2026)

This is the plan for the model runs that complete the public leaderboard. It follows the
pre-registration (`docs/imajev-bench-v2-preregistration.md`) and changes nothing about the items,
splits or metrics. Every row on the leaderboard states its interface, its settings and where it ran.

## What is scored

`data/imajev-bench/v2-lite-v1/records-audited.jsonl`, test split: 279 items (37 text, 120 visual,
122 joint), 89 evidence clusters, 21 Unknown references, 1 quarantined item excluded from scoring.
`verify_run` checks every run's `scoring_sha256` against these records before scoring.

## Interfaces (ranked separately, never pooled)

1. **Direct option scoring.** The model scores every allowed answer plus Unknown at the decision
   position, in every cyclic rotation of the options; the rotation-averaged distribution is the
   answer. This is the interface imajev serves. Backends: MLX (Mac) and PyTorch (CUDA). Both
   backends use the same prompt compilation and the same verified single-token label boundaries.
2. **Structured generation.** The model writes JSON that matches the answer schema (prompt
   `api-v2`), the same interface the frontier API rows used. Local models are served through an
   OpenAI-compatible server (vLLM) and called with `imajev_bench api-run --provider openai` with
   `OPENAI_BASE_URL` set; the manifest records the base URL and the served model id. Reasoning
   setting: the model's default, stated per row.

## Models

| Model | Size | Direct option scoring | Structured generation | Where |
|---|---|---|---|---|
| imajev-2b (v2.1, step 1,361) | 2B | yes (MLX done: 63.4%; PyTorch rerun for backend parity) | n/a (single-pass model) | Mac + pod |
| imajev-9b (v1.1) | 9B | yes | n/a | pod |
| Qwen3.5-2B base | 2B | yes | yes | pod |
| Qwen3.5-4B base | 4B | yes | yes | pod |
| Qwen3.5-9B base | 9B | yes | yes | pod |
| Gemma 4 E2B-it | 2B-effective | yes (MLX, Gemma-native template) | yes | Mac (direct) + pod (generation) |
| Gemma 4 E4B-it | 4B-effective | yes (MLX) | yes | Mac (direct) + pod (generation) |
| SmolVLM2-2.2B-Instruct | 2.2B | yes (MLX) | no (no reliable JSON mode) | Mac |

Excluded and why: kev (text-only, cannot take the image tracks); frontier APIs (already scored,
saturated); Qwen3.5-9B with thinking enabled (a different reasoning setting; may be added later as
its own row).

## Controls (imajev-2b unless stated)

- `no_image`: done (abstains on 258/279); reported as the image-necessity diagnostic.
- `no_state`: run on the pod; reported as the state-necessity diagnostic.
- Constant baselines: family majority and image-blind best, computed from the records (CPU).

## Correctness gates before any row is published

1. **Backend parity.** imajev-2b scored on MLX and on PyTorch must agree on ≥ 95% of test-item
   argmaxes and on accuracy within 2 points; otherwise the PyTorch rows are withheld until the
   difference is explained.
2. **Hash check.** Every run passes `verify_run`.
3. **Disclosure.** Interface, reasoning setting, rotations, backend, device, dtype, single-shot
   timing where repeats = 1, and the served model id for generation rows.
4. **Exclusions.** Only the pre-registered rule: the quarantined item, listed in the errata.
5. **Hypotheses.** H1 (imajev-9b vs Qwen3.5-9B) and H2 (imajev-2b vs Qwen3.5-2B) are reported as
   confirmatory paired cluster tests; every other comparison is labelled exploratory.
6. **Timing.** Pod runs are single-shot; the leaderboard says so. Contaminated Mac timings are
   excluded and counted.

## Hardware and cost

One H100 SXM 80 GB (RunPod secure, $2.69/h, disk 200 GB): bootstrap ~10 min (weights for five base
models ≈ 60 GB, adapters, vLLM), direct-scoring runs ~20 min, generation runs ~20 min, plus the
final-checkpoint calibration-fold and MMLU pass for the unknown-offset refit ~6 min. About one
hour, about $3. The Mac runs Gemma E2B, Gemma E4B and SmolVLM2 direct scoring sequentially with
the existing MLX backends (about 25 min each) at no cost.

## Corrections to the overview

- The canonical file is `records-audited.jsonl`; the pod bundle uses it.
- The hidden test set described in overview §8 (`data/imajev-bench/private-1/`) does not exist on
  disk yet; the overview must not claim it until it is built and its hash published.
