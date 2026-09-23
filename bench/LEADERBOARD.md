# ImajevBench v2.0-lite (preview) — leaderboard, test split (279 items, 89 evidence clusters, 21 Unknown references)

Preview: labels are construction-verified and model-audited (1 error in a 138-item blind audit); no human audit yet; all images are
AI-generated (see DATASHEET.md). Scores are accuracy with Unknown correct only when the reference is Unknown; errors count as wrong.
95% intervals are percentile bootstraps over evidence clusters. **The two interfaces are ranked separately and never pooled** (pre-registration):
direct option scoring and structured generation change scores by tens of points and answer different questions.
Models marked * also checked the images during construction, so accepted images were selected to be readable by them.

Runs of 23 Sept 2026. Local rows: one NVIDIA H100 80 GB (RunPod), single-shot timing (one warm-up record, one repeat, not the protocol's
three), so latency is indicative only. The imajev-2b MLX row is the earlier Mac Studio run (238/279 timings contaminated by a concurrent process).
Frontier rows: Azure OpenAI / Vertex AI; latency includes network time.

## A. Direct option scoring (single forward pass per rotation; the interface imajev serves)

Settings for every row: full option rotations (each option shown in every position), prompt profile `benchmark-neutral`, uncalibrated
probabilities, PyTorch bf16 on CUDA unless stated. "Correct Unknown" is abstentions on the 21 Unknown-reference items; "False abstention" is
abstentions on the 258 answerable items.

| Rank | System | Base model (revision) | Params | Accuracy (95% CI) | Text | Visual | Joint | Correct Unknown | False abstention | Contrast sets all correct | ECE / Brier | Latency p50 / p95 (single-shot) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | **imajev-9b** (v1.1 adapter + readout) | Qwen/Qwen3.5-9B (c2022362) | 9B | **226/279 (81.0%)** [0.75, 0.86] | 19/37 | 109/120 | 98/122 | 10/21 | 0/258 | 35/51 | 0.061 / 0.270 | 0.88 s / 1.47 s |
| 2 | Qwen3.5-9B base | Qwen/Qwen3.5-9B (c2022362) | 9B | 214/279 (76.7%) [0.70, 0.82] | 15/37 | 108/120 | 91/122 | 7/21 | 0/258 | 32/51 | 0.033 / 0.300 | 0.78 s / 1.28 s |
| 3 | Qwen3.5-4B base | Qwen/Qwen3.5-4B (851bf6e8) | 4B | 197/279 (70.6%) [0.64, 0.78] | 14/37 | 102/120 | 81/122 | 13/21 | 12/258 | 26/51 | 0.065 / 0.389 | 0.58 s / 0.98 s |
| 4 | **imajev-2b** (v2.1 step 1,361 adapter + readout) | Qwen/Qwen3.5-2B (15852e8c) | 2B | **176/279 (63.1%)** [0.56, 0.70] | 10/37 | 90/120 | 76/122 | 7/21 | 18/258 | 20/51 | 0.089 / 0.429 | 0.58 s / 0.98 s |
| 4' | imajev-2b, same adapter, Mac Studio MLX | Qwen/Qwen3.5-2B (15852e8c) | 2B | 177/279 (63.4%) [0.56, 0.70] | 9/37 | 91/120 | 77/122 | 7/21 | 19/258 | 21/51 | – | 4.1 s / 7.1 s (contaminated) |
| 5 | Qwen3.5-2B base | Qwen/Qwen3.5-2B (15852e8c) | 2B | 168/279 (60.2%) [0.53, 0.67] | 8/37 | 100/120 | 60/122 | 5/21 | 12/258 | 17/51 | 0.087 / 0.460 | 0.58 s / 0.97 s |

Backend parity (gate for publishing PyTorch rows): imajev-2b scored with MLX (Mac) and with PyTorch (H100) agrees on 274 of 279
answers (98.2%), 177 vs 176 correct. Gemma 4 E2B/E4B and SmolVLM2-2.2B direct-scoring rows are deferred: their harness backends are
MLX-only and the Mac runs were stopped; they will be added when a PyTorch path exists or the Mac runs are redone.

## B. Structured generation (JSON matching the answer schema, prompt `api-v2`)

Local rows are served by vLLM 0.30.0 on the same H100 (`--served-model-name` alias; the model path and revision are the ones listed),
`response_format` = strict JSON schema, the model's **default** reasoning setting, 8 parallel requests. Frontier rows as run earlier.

| Rank | System | Model (revision) | Setting | Accuracy (95% CI) | Text | Visual | Joint | Correct Unknown | False abstention | Errors | Contrast sets all correct | Latency p50 / p95 |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Gemini 3.1 Pro* | gemini-3.1-pro-preview | default thinking; Vertex AI | 278/279 (99.6%) [0.99, 1.00] | 37/37 | 120/120 | 121/122 | 20/21 | 0/258 | 0 | 50/51 | 8.0 s / 40.8 s |
| 2 | GPT-5.6 Luna* | gpt-5.6-luna | default; Azure OpenAI | 277/279 (99.3%) [0.98, 1.00] | 37/37 | 120/120 | 120/122 | 20/21 | 0/258 | 0 | 50/51 | 5.1 s / 24.4 s |
| 3 | GPT-5.4 | gpt-5.4 | reasoning medium; Azure OpenAI | 276/279 (98.9%) [0.98, 1.00] | 37/37 | 120/120 | 119/122 | 19/21 | 0/258 | 1 | 48/51 | 6.0 s / 9.7 s |
| 4 | Gemini 3.8 Flash | gemini-3.8-flash | default thinking; Vertex AI | 274/279 (98.2%) [0.96, 1.00] | 37/37 | 115/120 | 122/122 | 21/21 | 0/258 | 5 | 50/51 | 7.2 s / 79.8 s |
| 5 | Grok 4.3 | grok-4.3 | no response schema; default; Azure OpenAI | 255/279 (91.4%) [0.87, 0.95] | 24/37 | 113/120 | 118/122 | 17/21 | 11/258 | 10 | 45/51 | 7.7 s / 14.6 s |
| 6 | Qwen3.5-9B base | Qwen/Qwen3.5-9B (c2022362) | default; vLLM, JSON schema | 208/279 (74.6%) [0.69, 0.80] | 16/37 | 102/120 | 90/122 | 8/21 | 3/258 | 5 | 25/51 | 0.79 s / 3.57 s |
| 7 | Qwen3.5-4B base | Qwen/Qwen3.5-4B (851bf6e8) | default; vLLM, JSON schema | 188/279 (67.4%) [0.61, 0.73] | 12/37 | 101/120 | 75/122 | 9/21 | 13/258 | 6 | 21/51 | 0.68 s / 3.28 s |
| 8 | Gemma 4 E4B-it | google/gemma-4-E4B-it (ee0ef602) | default; vLLM, JSON schema | 168/279 (60.2%) [0.53, 0.68] | 14/37 | 85/120 | 69/122 | 4/21 | 20/258 | 4 | 19/51 | 0.43 s / 5.85 s |
| 9 | Qwen3.5-2B base | Qwen/Qwen3.5-2B (15852e8c) | default; vLLM, JSON schema | 167/279 (59.9%) [0.54, 0.66] | 9/37 | 93/120 | 65/122 | 2/21 | 9/258 | 6 | 11/51 | 0.56 s / 1.73 s |
| 10 | Gemma 4 E2B-it | google/gemma-4-E2B-it (3e22461f) | default; vLLM, JSON schema | 164/279 (58.8%) [0.52, 0.66] | 14/37 | 80/120 | 70/122 | 3/21 | 5/258 | 4 | 16/51 | 0.37 s / 2.50 s |

Errors are replies that were not valid JSON in the schema; they count as wrong. imajev models have no generation row: the adapter is a
single-pass readout, not a generator.

## Controls and constant baselines

| Control | Result | Reading |
| --- | --- | --- |
| imajev-2b, no image (MLX, Mac) | 21 answered / 258 abstained; 24/279 (8.6%) correct | image-necessity: without the photo the model abstains; 0.0% on the 227 answerable visual and joint items vs family-majority 37.4% |
| imajev-2b, no state (PyTorch, H100) | 144/279 (51.6%) [0.45, 0.58]; joint 46/122 (full: 76/122); false abstention 70/258 | state-necessity: joint items need the written situation; 30 joint items lost, abstentions quadruple |
| Family-majority constant answer | 100/279 (35.8%) | one answer per family |
| Image-blind best on contrast sets | 110/165 (66.7%) of contrast-set items; only 7/51 sets can be fully correct with one repeated answer | an image-blind system that repeats one answer per scene cannot pass the contrast-set metric |

## Pre-registered hypotheses (paired cluster sign-flip tests over the 89 evidence clusters, seed 0)

| Hypothesis | Comparison | Δ accuracy | Discordant clusters | p | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| H1 (confirmatory, two-sided) | imajev-9b vs Qwen3.5-9B base, direct scoring | +4.3 pts (226 vs 214) | 17 | 0.031 | confirmed |
| H2 (confirmatory, two-sided) | imajev-2b vs Qwen3.5-2B base, direct scoring | +2.9 pts (176 vs 168) | 46 | 0.572 | not significant |
| H3 (one-sided, contrast score) | 9B vs 2B imajev | 35/51 vs 20/51 sets; overall p = 0.0001 | 46 | 0.0001 | supported |

Everything else on this page is exploratory. Exploratory readings: (a) at every size, direct option scoring of the base Qwen model
equals or beats its own structured-generation row (2B 60.2 vs 59.9, 4B 70.6 vs 67.4, 9B 76.7 vs 74.6); (b) the imajev adapters gain on the
joint track (2B +16 items, 9B +7) and the 2B loses 10 visual-only items to abstention; (c) the same families are hard for every open model:
rule exceptions, date arithmetic, numerical reconciliation, policy precedence.

## Per-family accuracy, direct option scoring (test split)

| Family | n | imajev-9b | Qwen3.5-9B | Qwen3.5-4B | imajev-2b (H100) | imajev-2b (MLX) | Qwen3.5-2B | imajev-2b no state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answerability | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| attribute_state | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| comparison | 13 | 13 | 13 | 10 | 8 | 9 | 12 | 10 |
| counting | 34 | 28 | 27 | 24 | 18 | 19 | 21 | 18 |
| date_arithmetic | 6 | 3 | 1 | 1 | 1 | 1 | 0 | 0 |
| missing_conflicting_evidence | 8 | 4 | 4 | 4 | 4 | 3 | 1 | 4 |
| multi_clause_rule | 17 | 16 | 17 | 15 | 15 | 15 | 15 | 8 |
| multi_step_rule | 17 | 12 | 10 | 4 | 9 | 9 | 4 | 4 |
| numerical_reconciliation | 10 | 4 | 4 | 4 | 2 | 2 | 4 | 0 |
| policy_precedence | 13 | 8 | 6 | 5 | 3 | 3 | 3 | 2 |
| rule_exception | 36 | 18 | 15 | 20 | 13 | 14 | 17 | 14 |
| spatial_counting | 8 | 6 | 4 | 4 | 2 | 2 | 3 | 2 |
| spatial_relation | 11 | 8 | 10 | 10 | 8 | 7 | 10 | 8 |
| text_reading | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 |
| threshold_rule | 52 | 52 | 49 | 42 | 39 | 39 | 24 | 20 |

## Per-family accuracy, structured generation (test split)

| Family | n | Gemini 3.1 Pro | GPT-5.6 Luna | GPT-5.4 med | Gemini 3.8 Flash | Grok 4.3 | Qwen3.5-9B | Qwen3.5-4B | Gemma 4 E4B | Qwen3.5-2B | Gemma 4 E2B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answerability | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| attribute_state | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 6 | 5 |
| comparison | 13 | 13 | 13 | 13 | 13 | 12 | 12 | 12 | 12 | 12 | 9 |
| counting | 34 | 34 | 34 | 34 | 34 | 29 | 21 | 22 | 15 | 18 | 15 |
| date_arithmetic | 6 | 6 | 6 | 6 | 6 | 0 | 3 | 2 | 1 | 0 | 2 |
| missing_conflicting_evidence | 8 | 8 | 8 | 8 | 8 | 4 | 3 | 2 | 3 | 2 | 3 |
| multi_clause_rule | 17 | 17 | 17 | 17 | 17 | 17 | 16 | 15 | 15 | 13 | 14 |
| multi_step_rule | 17 | 17 | 17 | 17 | 17 | 17 | 10 | 9 | 7 | 8 | 3 |
| numerical_reconciliation | 10 | 10 | 10 | 10 | 10 | 9 | 4 | 2 | 4 | 4 | 4 |
| policy_precedence | 13 | 13 | 13 | 13 | 13 | 11 | 6 | 6 | 6 | 3 | 5 |
| rule_exception | 36 | 35 | 35 | 33 | 36 | 32 | 16 | 15 | 8 | 17 | 20 |
| spatial_counting | 8 | 8 | 8 | 8 | 7 | 8 | 6 | 4 | 3 | 4 | 2 |
| spatial_relation | 11 | 11 | 11 | 11 | 11 | 11 | 10 | 11 | 4 | 8 | 6 |
| text_reading | 43 | 43 | 43 | 43 | 39 | 42 | 42 | 42 | 40 | 40 | 38 |
| threshold_rule | 52 | 52 | 51 | 52 | 52 | 52 | 48 | 36 | 39 | 27 | 33 |

Sources: `h100-run/out/direct-*/score`, `h100-run/out/gen-*/score`, `h100-run/out/compare-H1|H2|H3x`, `eval-local/imajev2b-v21-full-test`,
`eval-local/imajev2b-v21-no_image-test`, `eval-summary/summary.md` (frontier). Every run carries a `scoring_sha256` verified against the records.
