# imajev-4b on DecisionBench 1.0

Run of the released `mohit67890/imajev-4b` adapter on the full DecisionBench 1.0 suite
(`Hanno-Labs/decision-bench` @ `b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443`, 23,900 rows) through the
benchmark's own, unmodified `run-system-one-http` runner, 2026-09-24 20:08–23:36 UTC.
Contamination check: [`contamination-check.md`](contamination-check.md).

## Results

| | imajev-4b |
|---|---|
| Primary accuracy (unsupported rows count as misses) | **77.5%** |
| Accuracy on scored rows | 79.3% |
| Coverage | 23,363 / 23,900 (97.75%); 537 unsupported (255 candidates), 0 errors |
| Expected calibration error (15 bins) | **0.024** |
| Mean negative log-likelihood | 0.663 |
| Mean latency per row (1 × H100, 3 server processes, concurrency 3) | 1.59 s |

Against the reviewed records in `Hanno-Labs/decision-bench-results` on 2026-09-25 (55 records incl. this one), this is third by primary accuracy after the two Bosun v3.1 models and the lowest calibration error of any record.

### Per task (accuracy on scored rows; reference records for comparison)

| Task | Rows scored | imajev-4b | Jev 1.13 | Winnow-12B | Bosun v3.1 1.7B |
|---|---:|---:|---:|---:|---:|
| agent-call-retention | 100 | 84.0 | 50.0 | 81.0 | 92.0 |
| agent-readiness-assessment | 100 | 90.0 | 94.0 | 93.0 | 96.0 |
| agent-result-retention | 100 | 93.0 | 52.0 | 100.0 | 100.0 |
| agent-skill-ranking | 100 | 69.0 | 69.0 | 69.0 | 69.0 |
| bounded_value | 2,223 | 93.9 | 95.6 | 94.5 | 96.3 |
| browser-action-selection | 100 | 85.0 | 83.0 | 77.0 | 84.0 |
| browser-target-selection | 100 | 47.0 | 50.0 | 50.0 | 42.0 |
| canonical_entity | 1,971 | 100.0 | 100.0 | 99.5 | 99.9 |
| claim-evidence-verification | 100 | 69.0 | 73.0 | 72.0 | 71.0 |
| code-change-risk | 100 | 59.0 | 62.0 | 62.0 | 70.0 |
| codebase-result-ranking | 100 | 62.0 | 60.0 | 57.0 | 60.0 |
| contains_threat | 1,124 | 98.1 | 50.9 | 99.3 | 99.8 |
| database-row-classification | 100 | 48.0 | 49.0 | 46.0 | 48.0 |
| dom-ad-detection | 100 | 100.0 | 45.0 | 95.0 | 99.0 |
| drone-tactical-action | 100 | 79.0 | 74.0 | 71.0 | 84.0 |
| finqa-numerical-reasoning | 400 | 68.0 | 85.5 | 82.5 | 14.0 |
| folio-logical-inference | 200 | 61.0 | 78.5 | 68.5 | 51.5 |
| game-goal-selection | 100 | 70.0 | 72.0 | 64.0 | 75.0 |
| graph-edge-selection | 100 | 72.0 | 72.0 | 73.0 | 75.0 |
| home-alert-triage | 100 | 85.0 | 93.0 | 84.0 | 93.0 |
| musique-evidence-sufficiency | 400 | 64.0 | 55.8 | 56.8 | 56.0 |
| musique-multihop-answer | 200 | 83.0 | 86.0 | 74.5 | 46.5 |
| patent_section | 2,223 | 56.1 | 56.6 | 46.0 | 68.4 |
| platformer-control-selection | 100 | 50.0 | 53.0 | 43.0 | 59.0 |
| relevance | 2,222 | 87.9 | 52.7 | 87.5 | 92.1 |
| relevance_score | 2,222 | 62.3 | 55.7 | 56.1 | 72.8 |
| robot-skill-selection | 100 | 53.0 | 62.0 | 52.0 | 64.0 |
| route | 2,222 | 91.8 | 84.8 | 75.5 | 90.9 |
| runtime-outcome-verification | 100 | 46.0 | 49.0 | 43.0 | 57.0 |
| search-plan-routing | 100 | 92.0 | 90.0 | 90.0 | 90.0 |
| semantic-line-match | 100 | 84.0 | 54.0 | 86.0 | 88.0 |
| social-post-filtering | 100 | 80.0 | 81.0 | 79.0 | 80.0 |
| strategy-command-selection | 100 | 45.0 | 46.0 | 40.0 | 52.0 |
| tax-document-page-classification | 88 | 39.8 | 39.0 | 38.0 | 32.0 |
| tool-action-impact | 100 | 70.0 | 66.0 | 64.0 | 75.0 |
| tool-call-routing | 100 | 73.0 | 72.0 | 72.0 | 61.0 |
| tool-safety-gate | 100 | 95.0 | 57.0 | 96.0 | 96.0 |
| tool_route | 1,948 | 99.1 | 99.0 | 95.9 | 98.1 |
| toxicity_severity | 1,098 | 25.8 | 27.0 | 27.1 | 80.4 |
| trading-action-selection | 100 | 64.0 | 67.0 | 57.0 | 71.0 |
| workflow_decision | 2,222 | 81.9 | 72.9 | 84.5 | 93.7 |

## Data integrity

DecisionBench was not used in any way to build or tune imajev: no DecisionBench rows, task files or outputs were
used for training, checkpoint selection or calibration. The benchmark's data first reached our machines on
2026-09-25, after the last training run had finished, and nothing in our training pipeline references it.

Some DecisionBench tasks are built from public datasets that our training mixture also used. Any overlap comes only
from that shared public source; it is not a leak from DecisionBench. The affected tasks:

| DecisionBench task | Public source we also trained on | Rows whose upstream item is in our training data |
|---|---|---|
| RouteFinancial | banking77 (train split) | 1,030 of 1,145 |
| RouteGeneralAssistant | CLINC150 (train split) | 137 of 1,077 |
| RelevanceScore | Amazon ESCI | 54 of 2,222 |
| ContainsThreat / ToxicitySeverity | civil_comments (train split) | 8 / 11 |

MuSiQue tasks share some Wikipedia passage text with public QA datasets we trained on (SQuAD2, BoolQ, FEVER); no
MuSiQue question appears in our data. The other 36 tasks show no overlap. Method, manifests scanned and per-row
results: [`contamination-check.md`](contamination-check.md).

## What was evaluated

| | |
|---|---|
| Model | `mohit67890/imajev-4b` @ `712891d1192c6491441a19bdea254d8a6e1048d0` (Apache-2.0) |
| Adapter file | `adapter_model.safetensors` sha256 `d8d328f85dcda4459a2331c47aba2a89ec808210e712248afdb69e38b9e0c25d` (verified on the pod) |
| Base weights | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, unchanged |
| Parameters | 4,659,865,088 base (incl. vision tower, unused here) + 30,474,240 LoRA + a 255-code linear readout |
| Server | `github.com/mohit67890/imajev` @ `4e623a8a51aa319405a0f32e6b73a8ce73416800`, `scripts/playground/server.py` |
| Serving config | as released: torch backend, bf16, `--rotations 4` (four cyclic option orders averaged), shipped `calibration.json` (one temperature, 1.717), `--max-input-tokens 32768` |
| Harness | `Hanno-Labs/decision-bench` @ `47ea5a479e35fd5ac7fde5c72103143071d7d93f`, `decision-bench run-system-one-http`, unmodified; `--max-candidates 254`, `--concurrency 3` |
| Hardware | 1 × H100 SXM 80GB (RunPod), three server processes behind an nginx round-robin |

## Decision and probability contract

- `noul` → `answers.decision.noul` = P(true) (plus half of the model's explicit unknown mass); `choice` →
  `probabilities` over every candidate, summing to 1; `score` → `probabilities` keyed by level index.
- Probabilities come from one forward pass per option order: a trained readout over the candidate-code
  positions (no generated tokens), averaged over four cyclic option orders, then divided by one temperature fitted on
  150 of our own authored decision items (never on DecisionBench).
- Candidate limit 254 (one of the 255 readout codes is reserved for "unknown"). The 537 rows with 255 candidates
  are recorded by the harness as `UnsupportedSystemOneInput` and count as misses in the primary score.
- Input limit: 32,768 processed tokens (the server's default research limit is 4,096, the training length; the flag
  was added so long DecisionBench rows are scored instead of refused). Nothing is truncated.

## Harness metadata note

`run-system-one-http` writes XOR's serving identity into `summary.json` regardless of the endpoint
(`adapter: xor-serving-systemone-v1`, `probability_source: forward_reverse_option_letter_logprobs_calibrated_v1`,
`serving_bundle_sha256`, `inference_image`). We passed our own values where the CLI takes them
(`--model-repo`, `--model-revision`, `--serving-bundle-sha256` = adapter sha256, `--inference-image` = the pinned
server command). The adapter/probability-source strings in `summary.json` are the harness defaults and do not
describe imajev; the result record declares the correct ones.

## Reproduce

[`pod_decisionbench.sh`](pod_decisionbench.sh) does everything from public sources on a fresh
RunPod `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` H100: clone + install imajev at the pinned commit, download
the pinned base and adapter, start the servers, clone DecisionBench at the pinned commit, run the smoke subset and
then the full suite, tar the run directory.

Run log: the first launch stopped during setup (`pip install uv` refused by the system Python); the setup step was
fixed and the remaining steps resumed with the same servers. The full-suite run directory (`full/`) was written by a
single uninterrupted invocation.

## Files

| File | What |
|---|---|
| `summary.json`, `manifest.json` | the harness's run summary and manifest, unchanged (manifest hashes verified locally) |
| `result-record/` | the compact record produced by `decision-bench stage-result` (passes `validate-results` and the results repository's tests) |
| `pod-run.log` | the pod's run log |
| `contamination-check.md`, `contamination_check.py`, `contamination-results.json` | training-data overlap check |

The full row-level `raw.jsonl` (23,900 rows, 377 MB; sha256 in `manifest.json` and `result-record/DecisionBench.json`)
is kept off git and is available on request.

## Phase-3 version: rerun on any pod size (`pod_decisionbench_p3.sh`)
Same harness, same protocol, one script for 1–8 GPUs: it counts the GPUs, starts `SERVERS_PER_GPU` (3) torch servers per GPU behind one
nginx round-robin and runs the harness with `--concurrency` = total servers (1×H100 ≈ 3.5 h, 4×H100 ≈ 1 h, 8×H100 ≈ 30 min; ≈ $15 in all
cases). Differences from the 1.0 run: `--max-candidates 255` (the 256-code readout scores the 537 rows with 255 candidates that counted as
misses before), `--calibration calibration-rot4.json`, adapter sha256 computed on the pod and passed as `--serving-bundle-sha256`,
`run-meta.json` (GPUs, servers, setup / run seconds) in the tarball. Pins are passed as environment variables:
```bash
scp reports/benchmarks/decisionbench/pod_decisionbench_p3.sh <pod>:
ssh <pod> 'cd /workspace && IMAJEV_COMMIT=<public commit> ADAPTER_REV=<HF revision> setsid nohup bash pod_decisionbench_p3.sh > launch.log 2>&1 < /dev/null &'
# dry run before the HF upload: scp the staged adapter dir and add ADAPTER_LOCAL=adapters/imajev-4b (revision recorded as local-unpublished)
ssh <pod> 'grep -E "SERVERS_READY|SMOKE_DONE|ALL_DONE|FAILED" db/run.log'; scp <pod>:decisionbench-imajev-4b-p3.tgz reports/benchmarks/decisionbench-run/
```
The official submission (a second PR to `Hanno-Labs/decision-bench-results`, or an update of #68) needs the public HF revision, so the upload
comes first; the result record is built from `db/full/` exactly as for the 1.0 run.
