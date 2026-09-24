# ImajevBench v2.0-lite (preview) — leaderboard, test split (279 items, 89 evidence clusters, 21 Unknown references)

Preview: labels are construction-verified and model-audited (1 error in a 138-item blind audit); no human audit yet; all images are
AI-generated (see DATASHEET.md). Scores are accuracy with Unknown correct only when the reference is Unknown; errors count as wrong.
95% intervals are percentile bootstraps over evidence clusters. **The two interfaces are ranked separately and never pooled** (pre-registration):
direct option scoring and structured generation change scores by tens of points and answer different questions.
Models marked * also checked the images during construction, so accepted images were selected to be readable by them.

Runs of 23 Sept 2026 (base models, earlier adapters, generation rows) and 24 Sept 2026 (released adapters and their pre-release checkpoints). Local rows: one NVIDIA H100 80 GB (RunPod), single-shot timing (one warm-up record, one repeat, not the protocol's
three), so latency is indicative only. The imajev-2b MLX row is the earlier Mac Studio run (238/279 timings contaminated by a concurrent process).
Frontier rows: Azure OpenAI / Vertex AI; latency includes network time.

**Scoring rules.**
- **Abstention.** Answering Unknown on an item that has an answer is **wrong**, and giving an answer on an item whose reference is
  Unknown is also **wrong**. Unknown is correct only when the reference is Unknown. Errors (unparseable or out-of-domain replies,
  timeouts) are wrong. The Correct Unknown and False abstention columns break this down; they do not change the accuracy.
- **Chance-corrected** = (accuracy − chance) / (1 − chance), × 100. Chance is a uniform guess over each item's allowed answers,
  Unknown included, averaged over the test split: **26.5%** (text 25.0%, visual 21.2%, joint 32.3%). 0 means no better than guessing.
  Allowed answers per item (Unknown included), counts on the test split:

  | Track | 3 answers | 4 | 5 | 6 |
  | --- | ---: | ---: | ---: | ---: |
  | text | 10 | 11 | 15 | 1 |
  | visual | 11 | 0 | 109 | 0 |
  | joint | 111 | 3 | 8 | 0 |

- **Calibration.** ECE (10 bins) and Brier cover direct-scoring rows only. Structured-generation rows return an answer without
  probabilities, so calibration does not apply to them.
- **Interval width.** With 279 items in 89 clusters, the 95% intervals are about ±6 points for rows scoring 60–80%, and about ±1 point
  near 99%. Treat a difference smaller than the intervals as unestablished. The paired cluster tests (pre-registered hypotheses below)
  decide comparisons, not the gap between point estimates.

## A. Direct option scoring (single forward pass per rotation; the interface imajev serves)

Settings for every row: full option rotations (each option shown in every position), prompt profile `benchmark-neutral`, uncalibrated
probabilities, PyTorch bf16 on CUDA unless stated. "Correct Unknown" is abstentions on the 21 Unknown-reference items; "False abstention" is
abstentions on the 258 answerable items.

| Rank | System | Base model (revision) | Params | Accuracy (95% CI) | Chance-corrected | Text | Visual | Joint | Correct Unknown | False abstention | Contrast sets all correct | ECE / Brier | Latency p50 / p95 (single-shot) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | **imajev-9b** (released) | Qwen/Qwen3.5-9B (c2022362) | 9B | **231/279 (82.8%)** [0.77, 0.88] | 76.6 | 29/37 | 106/120 | 96/122 | 15/21 | 2/258 | 35/51 | 0.127 / 0.282 | 0.55 s / 0.96 s |
| 2 | **imajev-4b** (released) | Qwen/Qwen3.5-4B (851bf6e8) | 4B | **230/279 (82.4%)** [0.77, 0.88] | 76.1 | 26/37 | 105/120 | 99/122 | 15/21 | 3/258 | 33/51 | 0.085 / 0.273 | 0.42 s / 0.69 s |
| – | imajev-9b, earlier version | Qwen/Qwen3.5-9B (c2022362) | 9B | 226/279 (81.0%) [0.75, 0.86] | 74.1 | 19/37 | 109/120 | 98/122 | 10/21 | 0/258 | 35/51 | 0.061 / 0.270 | 0.88 s / 1.47 s |
| 3 | Qwen3.5-9B base | Qwen/Qwen3.5-9B (c2022362) | 9B | 214/279 (76.7%) [0.70, 0.82] | 68.3 | 15/37 | 108/120 | 91/122 | 7/21 | 0/258 | 32/51 | 0.033 / 0.300 | 0.78 s / 1.28 s |
| 4 | Qwen3.5-4B base | Qwen/Qwen3.5-4B (851bf6e8) | 4B | 197/279 (70.6%) [0.64, 0.78] | 60.0 | 14/37 | 102/120 | 81/122 | 13/21 | 12/258 | 26/51 | 0.065 / 0.389 | 0.58 s / 0.98 s |
| 5 | **imajev-2b** (released) | Qwen/Qwen3.5-2B (15852e8c) | 2B | **196/279 (70.3%)** [0.64, 0.77] | 59.5 | 18/37 | 95/120 | 83/122 | 5/21 | 5/258 | 23/51 | 0.092 / 0.378 | 0.31 s / 0.50 s |
| 5' | imajev-2b, pre-release checkpoint (before the last part of the hard-question stage); not released | Qwen/Qwen3.5-2B (15852e8c) | 2B | 191/279 (68.5%) [0.61, 0.76] | 57.1 | 18/37 | 96/120 | 77/122 | 6/21 | 5/258 | 22/51 | 0.119 / 0.390 | 0.32 s / 0.53 s |
| – | imajev-2b, earlier version | Qwen/Qwen3.5-2B (15852e8c) | 2B | 176/279 (63.1%) [0.56, 0.70] | 49.7 | 10/37 | 90/120 | 76/122 | 7/21 | 18/258 | 20/51 | 0.089 / 0.429 | 0.58 s / 0.98 s |
| –' | imajev-2b, earlier version, same adapter, Mac Studio MLX | Qwen/Qwen3.5-2B (15852e8c) | 2B | 177/279 (63.4%) [0.56, 0.70] | 50.2 | 9/37 | 91/120 | 77/122 | 7/21 | 19/258 | 21/51 | – | 4.1 s / 7.1 s (contaminated) |
| 6 | Gemma 4 E4B-it | google/gemma-4-E4B-it (ee0ef602) | 4B (effective) | 176/279 (63.1%) [0.55, 0.72] | 17/37 | 95/120 | 64/122 | 7/21 | 35/258 | 26/51 | 0.252 / 0.582 | 0.93 s / 1.54 s (Mac Studio, MLX) |
| 7 | Qwen3.5-2B base | Qwen/Qwen3.5-2B (15852e8c) | 2B | 168/279 (60.2%) [0.53, 0.67] | 45.8 | 8/37 | 100/120 | 60/122 | 5/21 | 12/258 | 17/51 | 0.087 / 0.460 | 0.58 s / 0.97 s |
| 8 | SmolVLM2-2.2B-Instruct | HuggingFaceTB/SmolVLM2-2.2B-Instruct (482adb53) | 2.2B | 80/279 (28.7%) [0.22, 0.36] | 12/37 | 28/120 | 40/122 | 6/21 | **98/258** | 3/51 | 0.199 / 0.795 | 3.6 s / 6.0 s (Mac Studio, MLX) |

Rows for the released adapters and the 2B pre-release checkpoint (24 Sept 2026) come from the training pods' evaluation stage: same records, same scoring hash
(`7c9469d3…`), same settings (full rotations, `benchmark-neutral`, PyTorch bf16, one H100, single-shot), rescored locally with
`python -m imajev_bench score`. Pre-release checkpoints of the larger models (before the last part of the hard-question stage): imajev-4b 225/279 (80.6%), imajev-9b
227/279 (81.4%). The 9B and 4B released rows are statistically indistinguishable (paired test p = 1.0).

Backend parity (gate for publishing PyTorch rows): imajev-2b scored with MLX (Mac) and with PyTorch (H100) agrees on 274 of 279
answers (98.2%), 177 vs 176 correct. Gemma 4 E4B and SmolVLM2 rows (24 Sept 2026) come from the harness's MLX backends on a Mac Studio,
same records, settings and scoring hash; their latency is Mac latency and not comparable with the H100 rows. SmolVLM2 abstains on
98 of the 258 answerable items; on the 175 items it answers it is right 74 times (42%). **Gemma 4 E2B-it: run, withheld.** Its
direct-scoring run answered "False" on 127 of the 132 yes/no items (references: 67 true, 55 false, 10 Unknown), while the same model by
structured generation answers 51 true / 73 false, so we read it as a fault in our MLX Gemma backend on E2B, not a property of the model;
the row returns when the backend is fixed. Its structured-generation row (section B) stands. Exploratory paired tests: imajev-4b vs
Gemma 4 E4B +19.4 pts [+12.8, +26.1], p = 0.0001; imajev-2b vs Gemma 4 E4B +7.2 pts [+0.0, +14.5], p = 0.076.

## A2. Other open Jev-class decision models (their own single-pass interface)

Run on 24 Sept 2026 on one H100 through each model's own public inference API, on the public records (test gold withheld), then
scored with the same harness. Not pooled with section A: these models use their own prompt and head, no option rotations, and may
lack an abstain output.

| System | Model | Params | Accuracy (95% CI) | Text | Visual | Joint | Correct Unknown | False abstention | Contrast sets all correct | ECE / Brier | Latency p50 / p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Jev-Omni | akhilaaa3/Jev-Omni (c050d513), Gemma 4 12B + classifier head, `jev_omni.predict` | 12B | 219/279 (78.5%) [0.73, 0.84] | 22/37 | 101/120 | 96/122 | 0/21 (no abstain output) | 0/258 | 28/51 | 0.069 / 0.298 | 0.08 s / 0.37 s (one pass) |

Mapping used for Jev-Omni (its API takes a state string, a question and a list of option labels): yes/no → "Yes"/"No"; choice →
the option values (with descriptions when present); levels → "value: description"; the one image → `modality="image"`. The test split
has no two-image items. Runner: `scripts/bench/run_jevomni_bench.py` in the code repository. Exploratory paired tests: imajev-4b vs
Jev-Omni +3.9 pts [−2.4, +10.1], p = 0.27; imajev-9b vs Jev-Omni +4.3 pts, p = 0.24; imajev-2b vs Jev-Omni −8.2 pts, p = 0.017.
On the 258 answerable items Jev-Omni answers 219 correctly (imajev-4b 215, imajev-9b 216): the imajev lead comes from the Unknown
items. mojev and cua-s1 are text-only and not run here.

## B. Structured generation (JSON matching the answer schema, prompt `api-v2`)

Local rows are served by vLLM 0.30.0 on the same H100 (`--served-model-name` alias; the model path and revision are the ones listed),
`response_format` = strict JSON schema, the model's **default** reasoning setting, 8 parallel requests. Frontier rows as run earlier.

| Rank | System | Model (revision) | Setting | Accuracy (95% CI) | Chance-corrected | Text | Visual | Joint | Correct Unknown | False abstention | Errors | Contrast sets all correct | Latency p50 / p95 |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Gemini 3.1 Pro* | gemini-3.1-pro-preview | default thinking; Vertex AI | 278/279 (99.6%) [0.99, 1.00] | 99.5 | 37/37 | 120/120 | 121/122 | 20/21 | 0/258 | 0 | 50/51 | 8.0 s / 40.8 s |
| 2 | GPT-5.6 Luna* | gpt-5.6-luna | default; Azure OpenAI | 277/279 (99.3%) [0.98, 1.00] | 99.0 | 37/37 | 120/120 | 120/122 | 20/21 | 0/258 | 0 | 50/51 | 5.1 s / 24.4 s |
| 3 | GPT-5.4 | gpt-5.4 | reasoning medium; Azure OpenAI | 276/279 (98.9%) [0.98, 1.00] | 98.5 | 37/37 | 120/120 | 119/122 | 19/21 | 0/258 | 1 | 48/51 | 6.0 s / 9.7 s |
| 4 | Gemini 3.8 Flash | gemini-3.8-flash | default thinking; Vertex AI | 274/279 (98.2%) [0.96, 1.00] | 97.6 | 37/37 | 115/120 | 122/122 | 21/21 | 0/258 | 5 | 50/51 | 7.2 s / 79.8 s |
| 5 | Grok 4.3 | grok-4.3 | no response schema; default; Azure OpenAI | 255/279 (91.4%) [0.87, 0.95] | 88.3 | 24/37 | 113/120 | 118/122 | 17/21 | 11/258 | 10 | 45/51 | 7.7 s / 14.6 s |
| 6 | Qwen3.5-9B base | Qwen/Qwen3.5-9B (c2022362) | default; vLLM, JSON schema | 208/279 (74.6%) [0.69, 0.80] | 65.4 | 16/37 | 102/120 | 90/122 | 8/21 | 3/258 | 5 | 25/51 | 0.79 s / 3.57 s |
| 7 | Qwen3.5-4B base | Qwen/Qwen3.5-4B (851bf6e8) | default; vLLM, JSON schema | 188/279 (67.4%) [0.61, 0.73] | 55.6 | 12/37 | 101/120 | 75/122 | 9/21 | 13/258 | 6 | 21/51 | 0.68 s / 3.28 s |
| 8 | Gemma 4 E4B-it | google/gemma-4-E4B-it (ee0ef602) | default; vLLM, JSON schema | 168/279 (60.2%) [0.53, 0.68] | 45.8 | 14/37 | 85/120 | 69/122 | 4/21 | 20/258 | 4 | 19/51 | 0.43 s / 5.85 s |
| 9 | Qwen3.5-2B base | Qwen/Qwen3.5-2B (15852e8c) | default; vLLM, JSON schema | 167/279 (59.9%) [0.54, 0.66] | 45.3 | 9/37 | 93/120 | 65/122 | 2/21 | 9/258 | 6 | 11/51 | 0.56 s / 1.73 s |
| 10 | Gemma 4 E2B-it | google/gemma-4-E2B-it (3e22461f) | default; vLLM, JSON schema | 164/279 (58.8%) [0.52, 0.66] | 43.9 | 14/37 | 80/120 | 70/122 | 3/21 | 5/258 | 4 | 16/51 | 0.37 s / 2.50 s |

Errors are replies that were not valid JSON in the schema; they count as wrong. imajev models have no generation row: the adapter is a
single-pass readout, not a generator.

## Controls and constant baselines

| Control | Result | Reading |
| --- | --- | --- |
| earlier imajev-2b, no image (MLX, Mac) | 21 answered / 258 abstained; 24/279 (8.6%) correct | image-necessity: without the photo the model abstains; 0.0% on the 227 answerable visual and joint items vs family-majority 37.4% |
| earlier imajev-2b, no state (PyTorch, H100) | 144/279 (51.6%) [0.45, 0.58]; joint 46/122 (full: 76/122); false abstention 70/258 | state-necessity: joint items need the written situation; 30 joint items lost, abstentions quadruple |
| Family-majority constant answer | 100/279 (35.8%); chance-corrected 12.7 | one answer per family |
| Image-blind best on contrast sets | 110/165 (66.7%) of contrast-set items; only 7/51 sets can be fully correct with one repeated answer | an image-blind system that repeats one answer per scene cannot pass the contrast-set metric |

## Pre-registered hypotheses (paired cluster sign-flip tests over the 89 evidence clusters, seed 0)

| Hypothesis | Comparison | Δ accuracy | Discordant clusters | p | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| H1: imajev-9b vs its untuned base (confirmatory, two-sided) | **earlier imajev-9b the test was registered with** vs Qwen3.5-9B base, direct scoring | +4.3 pts (226 vs 214) | 17 | 0.031 | confirmed |
| H1 re-run on the released imajev-9b vs its untuned base | imajev-9b **released** vs Qwen3.5-9B base | +6.1 pts (231 vs 214), 95% CI [+0.0, +12.7] | 35 | 0.074 | **not significant at 0.05** |
| H2: imajev-2b vs its untuned base (confirmatory, two-sided) | **earlier imajev-2b the test was registered with** vs Qwen3.5-2B base, direct scoring | +2.9 pts (176 vs 168) | 46 | 0.572 | not significant |
| H2 re-run on the released imajev-2b vs its untuned base | imajev-2b released vs Qwen3.5-2B base | +10.0 pts (196 vs 168), 95% CI [+1.8, +17.8] | 49 | 0.019 | significant |
| H2 re-run on the imajev-2b pre-release checkpoint vs its untuned base | imajev-2b pre-release checkpoint vs Qwen3.5-2B base | +8.2 pts (191 vs 168), 95% CI [+0.0, +16.2] | 47 | 0.054 | not significant at 0.05 |
| H3: imajev-9b beats imajev-2b on contrast sets (one-sided, contrast score) | earlier imajev-9b vs earlier imajev-2b (the versions the test was registered with) | 35/51 vs 20/51 sets; overall p = 0.0001 | 46 | 0.0001 | supported |

The hypotheses were registered against the adapters released first (the earlier imajev-9b and imajev-2b). The adapters were then replaced by
versions trained further on hard typed questions; the tests were re-run on the same 89 clusters with the same seed and are reported beside the registered
results rather than instead of them. Re-testing after a model change is exploratory by construction. Further exploratory tests:
imajev-4b vs Qwen3.5-4B base +11.8 pts [+5.1, +18.5], p = 0.0008 (40 discordant clusters); imajev-9b vs the earlier imajev-9b
+1.8 pts, p = 0.618; imajev-9b vs imajev-4b (both released) +0.4 pts, p = 1.0; imajev-2b vs its pre-release checkpoint +1.8 pts, p = 0.273.

Everything else on this page is exploratory. Exploratory readings: (a) at every size, direct option scoring of the base Qwen model
equals or beats its own structured-generation row (2B 60.2 vs 59.9, 4B 70.6 vs 67.4, 9B 76.7 vs 74.6); (b) the imajev adapters gain on the
joint track (2B +16 items, 9B +7) and the 2B loses 10 visual-only items to abstention; (c) the same families are hard for every open model:
rule exceptions, date arithmetic, numerical reconciliation, policy precedence.

## Per-family accuracy, direct option scoring (test split)

imajev columns without a qualifier are the released models; "pre-release" and "earlier" are the rows of the same names in section A.

| Family | n | imajev-9b | imajev-4b | imajev-2b | imajev-2b pre-release | imajev-9b earlier | Qwen3.5-9B | Qwen3.5-4B | imajev-2b earlier (H100) | imajev-2b earlier (MLX) | Qwen3.5-2B | imajev-2b earlier, no state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answerability | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| attribute_state | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| comparison | 13 | 12 | 13 | 10 | 10 | 13 | 13 | 10 | 8 | 9 | 12 | 10 |
| counting | 34 | 27 | 24 | 22 | 22 | 28 | 27 | 24 | 18 | 19 | 21 | 18 |
| date_arithmetic | 6 | 5 | 5 | 5 | 5 | 3 | 1 | 1 | 1 | 1 | 0 | 0 |
| missing_conflicting_evidence | 8 | 8 | 8 | 2 | 3 | 4 | 4 | 4 | 4 | 3 | 1 | 4 |
| multi_clause_rule | 17 | 16 | 17 | 16 | 15 | 16 | 17 | 15 | 15 | 15 | 15 | 8 |
| multi_step_rule | 17 | 13 | 10 | 9 | 9 | 12 | 10 | 4 | 9 | 9 | 4 | 4 |
| numerical_reconciliation | 10 | 7 | 6 | 4 | 4 | 4 | 4 | 4 | 2 | 2 | 4 | 0 |
| policy_precedence | 13 | 9 | 7 | 7 | 6 | 8 | 6 | 5 | 3 | 3 | 3 | 2 |
| rule_exception | 36 | 15 | 20 | 14 | 10 | 18 | 15 | 20 | 13 | 14 | 17 | 14 |
| spatial_counting | 8 | 4 | 5 | 2 | 3 | 6 | 4 | 4 | 2 | 2 | 3 | 2 |
| spatial_relation | 11 | 9 | 9 | 7 | 7 | 8 | 10 | 10 | 8 | 7 | 10 | 8 |
| text_reading | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 | 43 |
| threshold_rule | 52 | 52 | 52 | 44 | 43 | 52 | 49 | 42 | 39 | 39 | 24 | 20 |

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

Sources: every run's predictions, manifests and scores are in the repository's results/ and bench/ folders; rows were rescored with `python -m imajev_bench score` and the paired tests re-run with `imajev_bench.stats.paired_cluster_test`. Every run carries a `scoring_sha256` verified against the records.
