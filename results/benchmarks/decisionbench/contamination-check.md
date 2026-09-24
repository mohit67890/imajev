# DecisionBench contamination check for imajev-4b 1.0

Date: 2026-09-25 (IST). Model: imajev-4b 1.0, the "phase-2c soup50" adapter, a 50/50 weight average of the phase-2b and phase-2c
adapters. Benchmark: `Hanno-Labs/decision-bench` at revision `b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443`, split `eval`
(23,900 rows, 43 tasks).
Script: `reports/benchmarks/decisionbench/contamination_check.py`. Raw output: `reports/benchmarks/decisionbench/contamination-results.json`.

## Conclusion

**No DecisionBench evaluation row was used in training. That statement is true. The data is still not clean at the source
level, and the submission should say so.** Our stage-1 text mixture contains the public training splits of four datasets that
DecisionBench also samples from. For four tasks, the upstream item behind a DecisionBench row was in our training data:

| DecisionBench task | Rows | Rows whose upstream item was in our `train` partition | How |
|---|---:|---:|---|
| **RouteFinancial** (banking77) | 1,145 | **1,030 (90.0%)**, all with the **same intent label** | DecisionBench paraphrases banking77 *train* utterances. Our stage-1 mixture trained on the whole banking77 train split (10,069 rows) as intent routing. 1,134 rows (99.0%) are in some partition of ours. |
| **RouteGeneralAssistant** (CLINC150) | 1,077 | **137 (12.7%)**, all with the same intent label | 134 of the 175 DecisionBench rows drawn from CLINC150 *train* (we trained on that split). 3 more are CLINC *test* utterances that also appear word for word in CLINC *train*. |
| **RelevanceScore** (ESCI) | 2,222 | **54 (2.4%)** exact (query, product) pairs, 43 with the same E/S/C/I label | Seen as *image* rows (`sqid_esci`, `state_aware`). Both are built on ESCI's `test_small_us` split, which is the split DecisionBench samples. Weaker overlap as well: 520 rows share their query with one of our training rows, and 815 rows share a product listing text of which > 50% of the 13-grams also appear in our ESCI-train rows. There the product is the same, but the query and label are different. |
| **ContainsThreat / ToxicitySeverity** (civil_comments) | 1,124 / 1,098 | **8 / 11** | Both use civil_comments *train*. We trained on a 13,219-row sample of that split. The overlap is what random sampling from the same 1.8M-comment split predicts. |

One more task shares source text with us without sharing items:

- **MuSiQue** (EvidenceSufficiency 400 rows, MultihopAnswer 200 rows). In 295 and 149 rows, at least one supporting passage is a
  near-copy (> 50% of its 13-grams) of a Wikipedia paragraph in our training data (SQuAD2, BoolQ, FEVER, Wikipedia paragraphs).
  MuSiQue builds its hops from SQuAD and other Wikipedia datasets. **No MuSiQue question matched**: 0 question-field hits. No row
  is a near-duplicate as a whole: the best is 60% of a row's 13-grams, and only 3 rows per task pass 50%.

Everything else is clean. The other 36 tasks have no near-duplicate row and no upstream-item overlap. They are PatentSection,
ToolRoute, BoundedValue, CanonicalEntity, Relevance (MS MARCO), WorkflowDecision (LegalBench), FinQA, FOLIO and the 27 synthetic
agent, game and code tasks. Their residual hits are shared boilerplate. Examples: "net cash provided
by operating activities" (FinQA vs Eikos TAT-QA), contract clauses quoted inside one HelpSteer2 prompt (WorkflowDecision), and
generic tax-form field labels.

**Suggested wording for the template.** "I am not aware of DecisionBench evaluation rows being used for training. Disclosure:
imajev-4b's training mixture includes the public train splits of banking77, CLINC150, civil_comments and ESCI (plus image rows
built on ESCI test_small_us). As a result, the source utterance and intent label behind 1,030 of the 1,145 RouteFinancial rows
(and 137 of the 1,077 RouteGeneralAssistant rows) were seen in training in their original, un-paraphrased form. The overlap for
RelevanceScore is 54 query–product pairs and for the civil_comments tasks 19 comments. RouteFinancial should be read as
contaminated. Method and per-task counts: <link to this file>."

## Why "no DecisionBench row was used" is supported

- **Timing.** The DecisionBench parquet reached this machine on 2026-09-25 01:16 IST (HF cache blob mtime). The last training
  stage (phase 2c) ran on pod the phase-2c pod on 2026-09-24. The pod was terminated at 12:55 UTC (18:25 IST). Every scanned
  training input predates the pod run: `gen_programmatic.py`, the Eikos slice and the image replay were last modified on
  2026-09-24 between 11:09 and 11:31 IST.
- **No reference in the pipeline.** No file under `scripts/`, `data/manifests/`, `docs/` or `src/` names DecisionBench or any of
  the GitHub repos its synthetic tasks come from. The only hit is the evaluation launcher `cloud/pod_decisionbench.sh`, written
  after training.
- **Text evidence.** Every near-duplicate state traces back to an upstream dataset item or passage, never to a DecisionBench
  construction:
  - ESCI product listings that the paraphrase kept;
  - Wikipedia passages in MuSiQue;
  - unchanged banking77, CLINC and civil_comments originals. Where a DecisionBench state equals one of our training texts exactly
    (banking77 11 rows, CLINC 14, civil_comments 8), DecisionBench's paraphrase left the upstream original unchanged.

  The 27 synthetic tasks and the DecisionBench-written parts of the other rows (instructions, reformatted states) have no
  near-duplicate.

## Method

1. **DecisionBench side.** The parquet was read at the pinned revision. As a cross-check, `get_benchmark(task_specs/decisionbench-dev.toml)`
   yields the same 23,900 row IDs with identical `state` values (0 mismatches). For every row two texts were indexed:
   - the **state**: every string value in `state_json`, each tokenised on its own;
   - for the 12 paraphrased tasks, the **upstream original** in `source_json.source.raw` (for example the banking77 utterance
     before paraphrase).
   Public task names come from `decision_bench/task_spec.py::_CANONICAL_TASKS` (task_name, domain, primitive).
2. **Normalisation.** The JevBench 8-gram lint's normalisation was reused (`scripts/p2/p2_common.py::ngrams`): lower-case,
   tokens `[a-z0-9]+`. n-grams never cross from one JSON field to the next. Hashing: blake2b 64-bit token IDs, then a 64-bit
   polynomial n-gram hash. The index held 2,256,448 distinct 13-grams, 2,433,643 distinct 8-grams, 21,100 short-field spans and
   38,427 exact keys.
3. **Training side.** Each file was streamed once, in 48 MB chunks across 10 worker processes (about 30 s in total, low memory).
   Every string value of each record's `request` was checked: state, questions and options. Rationale fields were checked too,
   and for writer files the whole `output`. Signals, per DecisionBench row:
   - **≥1 13-gram**: any of the row's 13-grams is found anywhere in the training data;
   - **row near-dup**: more than 50% of the row's distinct 13-grams are found;
   - **field near-dup**: more than 50% of the 13-grams of a single field are found (for example one MuSiQue passage or one ESCI
     listing);
   - **short field contained**: a field of 6–12 tokens, too short to have a 13-gram, appears verbatim as a contiguous token span;
   - **exact state**: the whole normalised state equals a training state or string field;
   - **≥ 2 8-grams**: the JevBench lint's rule, reported for sensitivity only. It fires on shared boilerplate.
   Each signal was counted twice: against `train`-partition rows only (the rows that produced gradient updates), and against any
   partition, which adds the dev, test and calibration rows used for checkpoint selection and temperature fitting.
4. **Item-level joins** (`targeted()` in the script). These were run for the four datasets we share with DecisionBench:
   - banking77, CLINC150 and civil_comments: normalised upstream text, with the intent label compared for the first two;
   - ESCI: `example_id` is mapped to `(query, product_id)` through the ESCI examples table we downloaded for stage 1. It is then
     matched against our text rows by (query, title) and against our image rows by (query, SKU), and the E/S/C/I label is compared.

## Manifests scanned

All three stage manifests match the `manifest_sha256` recorded in their training run's `config.json`.

| File | Role | Rows (train / other partitions) | sha256 |
|---|---|---|---|
| `data/manifests/decision-v2.1-4b.jsonl` | Stage 1–2 (v2.1 mixture). It contains the v1.1 text/image train set, the v2 teacher slice (StackExchange, Wikipedia, support/reviews, PD12M, Commons, Open Images) and the v2.1 grounded/pairs rows | 950,839 (808,869 / 23,309 cal, 44,123 dev, 74,538 test) | `1ea2eb63f391e2dd5e1657973dbb4babec8986049d051920c23c187942f39c90` |
| `data/manifests/decision-p2.jsonl` | Stage 3a (teacher + human half: ARC, CSQA, CosmosQA, DROP, GSM8K, MMLU-Pro, QASC, StrategyQA, TruthfulQA, WANLI) | 17,898 (14,112 / 803 dev, 2,983 test) | `98a39e69f29b9a547897b0642ac5e1b52fb14c36945afb00ccaeb3304a965d59` |
| `data/manifests/decision-p2b.jsonl` | Stage 3b (incl. the 30% replay) | 9,066 (7,812 / 622 cal, 197 dev, 435 test) | `085b42ff6529ac7dd44a27df20faa387bf8584b93304fb4da19fc515e8a9a8b9` |
| `data/external/eikos-decisions/records.jsonl` | Stage 3c: the Eikos CC-BY strict slice (includes GSM8K-train and TAT-QA judge rows) | 10,570 (10,152 / 418 dev) | `eb46bee90b480b789ce90c014a940eb36b130a9554d148f73a2339b976874e4f` |
| `data/decision-p2c/image-replay/replay-delta.jsonl` | Stage 3c image replay | 5,000 (5,000 / 0) | `c541ab76da59bd1c8f11f09b2c40334dd1c717b96a1306f6839146ab18af0b21` |
| `writer-prog-seed7.jsonl`, regenerated in the session scratchpad with `scripts/p2/gen_programmatic.py --docs 3000 --seed 7` (the pod's arguments; the generator is deterministic) | Stage 3c programmatic questions | 3,000 docs / 9,000 questions | `b7e27800cfd3e316ba1c577fb1e6e3d08ef4993878c85a05eef83ffc563bd6c8` |
| Local pre-run build of `decision-p2c.jsonl` (session scratchpad `…/969cc7ad…/scratchpad/p2c-audit/`) | Stage 3c: p2b rows as relabelled + Eikos + replay + 71 dry-run rows | 29,859 (25,823 / 4,036) | `031e35141845c822eefc1a81634f3a6bb725453484410b63e74365ca9d25c582` |
| `data/manifests/decision-v2-reasoning-dev.jsonl` | selection only (dev2 in v2 and 3a) | 1,440 | `ec89cff0050960fb759cae230a5d7570f5aa9e03d63e25b75893958dafd165d4` |
| `data/manifests/decision-v2.1-grounded-dev.jsonl` | selection only (4B v2.1 dev2) | 396 | `182eb08adfb75aec923f754d4ff8768d7b8758cff3542e6a9ff579a56a1fb672` |
| `data/manifests/decision-p2b-jevstyle-dev.jsonl` | selection only (3b dev2, authored-dev temperature fit) | 150 | `3c206b2156204b60e1872eb59dff341494f11799d2c7988767a55a593b5c853b` |
| `data/manifests/decision-v2.1-calibration.jsonl` | calibration only | 23,309 | `fe884c75165780b3c4ff4622a50590b67938135d2b09600f7f9734ec61c12626` |

Split facts from the scan (`decision-v2.1-4b`, `train` partition):
- banking77 10,069, CLINC150 13,219, civil_comments 13,219 and ESCI text 13,219 rows, all from the upstream `train` split. Their
  upstream test rows sit only in our `test` partition.
- `sqid_esci` 5,563 and `state_aware` 5,563 rows come from ESCI `test_small_us`.

## Name-level overlap with DecisionBench sources

| DecisionBench task | Upstream (per `source_json`) | DB split | Our use | Split-level overlap possible? | Measured |
|---|---|---|---|---|---|
| RouteFinancial | PolyAI/banking77 | train (1,127 items), test (18) | banking77 train, 10,069 rows, stage 1 | **Yes, same split** | 1,030 items, same label |
| RouteGeneralAssistant | clinc/clinc_oos | test (902), train (175) | CLINC150 train, 13,219 rows | **Yes** for the 175 train items; test items only through upstream duplicates | 137 items, same label |
| RelevanceScore | tasksource/esci | test (2,092 items) | ESCI train, 13,219 text rows; ESCI test_small_us, 11,126 image rows | **Yes** through the image rows; the text rows share products, not queries | 54 pairs (43 same label); 815 product texts |
| ContainsThreat, ToxicitySeverity | google/civil_comments | train | civil_comments train, 13,219-row sample | **Yes, same split**, small sample | 19 comments |
| MuSiQueEvidenceSufficiency, MuSiQueMultihopAnswer | dgslibisey/MuSiQue | validation | not used; SQuAD2 / BoolQ / FEVER / Wikipedia paragraphs share its source passages | Passages only | 0 questions; passage near-copies in 295 / 149 rows |
| FinQANumericalReasoning | dreamerdeo/finqa | validation | not used; Eikos TAT-QA rows are a different dataset built on similar annual-report text | No | 0 near-dups |
| FolioLogicalInference | tasksource/folio | validation | not used | No | 0 |
| Relevance | Tevatron/msmarco-passage | train | not used | No | 0 near-dups |
| WorkflowDecisionBoolean/Choice | nguha/legalbench | test | not used | No | 0 near-dups (the 12 13-gram hits are clause fragments quoted in one HelpSteer2 prompt) |
| PatentSection | big_patent | train | not used | No | 0 |
| ToolRoute | Team-ACE/ToolACE | train | not used | No | 0 |
| BoundedValue | nvidia/Nemotron-PII | train | not used | No | 0 near-dups |
| CanonicalEntity | zal-analytics-core/dutch-law-enforcement-entity-resolution-dataset | entity pairs | not used | No | 0 |
| 27 synthetic tasks | synthetic from public GitHub decision contracts and use-case observations (jev-* repos etc.) | — | none of these repos appear anywhere in the pipeline | No | 0 near-dups |

Names checked and absent from DecisionBench: typed-decisions, MMLU/MMLU-Pro, GSM8K, TAT-QA, Eikos, SQuAD2, BoolQ, FEVER, SNLI,
HelpSteer2, GoEmotions, MASSIVE, SNIPS, Aegis, OpenAI moderation, StackExchange and all image sources.

## Per-task text overlap

All columns except the last pair are measured on the DecisionBench **state**, against our **train** partition. "Any signal"
counts a row that has a 13-gram hit, a contained short field or an exact state. "Upstream original" is the pre-paraphrase text
from `source_json`, which exists for the 12 paraphrased tasks.

| Task | Rows | ≥1 13-gram (train) | Row near-dup >50% (train) | Field near-dup >50% (train) | Short field contained (train) | Exact state (train) | ≥2 8-grams (train) | Any signal, train / any partition | Upstream original: exact (train) | Upstream original: near-dup (train) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RelevanceScore | 2222 | 962 | 525 | 815 | 467 | 0 | 1193 | 1190 / 1384 | 17 | 909 |
| RouteFinancial | 1145 | 8 | 1 | 1 | 7 | 11 | 19 | 21 / 26 | 968 | 278 |
| MuSiQueEvidenceSufficiency | 400 | 334 | 3 | 295 | 79 | 0 | 385 | 343 / 360 | n/a | n/a |
| RouteGeneralAssistant | 1077 | 0 | 0 | 0 | 4 | 14 | 4 | 16 / 30 | 107 | 14 |
| MuSiQueMultihopAnswer | 200 | 169 | 3 | 149 | 34 | 0 | 187 | 170 / 177 | n/a | n/a |
| WorkflowDecisionBoolean | 2142 | 12 | 1 | 1 | 0 | 0 | 81 | 12 / 15 | 0 | 4 |
| FinQANumericalReasoning | 400 | 3 | 0 | 3 | 33 | 0 | 9 | 36 / 36 | n/a | n/a |
| BoundedValue | 2223 | 20 | 0 | 0 | 0 | 0 | 105 | 20 / 20 | 0 | 0 |
| ToxicitySeverity | 1098 | 5 | 5 | 5 | 0 | 5 | 9 | 6 / 8 | 9 | 8 |
| Relevance | 2222 | 3 | 0 | 0 | 2 | 0 | 15 | 4 / 6 | 0 | 0 |
| ContainsThreat | 1124 | 4 | 3 | 3 | 0 | 3 | 8 | 5 / 6 | 6 | 6 |
| TaxDocumentPageClassification | 100 | 1 | 0 | 0 | 15 | 0 | 1 | 16 / 16 | n/a | n/a |
| ClaimEvidenceVerification | 100 | 0 | 0 | 0 | 3 | 0 | 1 | 3 / 6 | n/a | n/a |
| HomeAlertTriage | 100 | 0 | 0 | 0 | 4 | 0 | 0 | 4 / 4 | n/a | n/a |
| SearchPlanRouting | 100 | 0 | 0 | 0 | 3 | 0 | 0 | 3 / 3 | n/a | n/a |
| FolioLogicalInference | 200 | 0 | 0 | 0 | 1 | 0 | 3 | 1 / 1 | n/a | n/a |
| TradingActionSelection | 100 | 0 | 0 | 0 | 1 | 0 | 0 | 1 / 1 | n/a | n/a |
| AgentCallRetention | 100 | 0 | 0 | 0 | 0 | 0 | 1 | 0 / 0 | n/a | n/a |
| AgentReadinessAssessment | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| AgentResultRetention | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| AgentSkillRanking | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| BrowserActionSelection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| BrowserTargetSelection | 100 | 0 | 0 | 0 | 0 | 0 | 1 | 0 / 0 | n/a | n/a |
| CanonicalEntity | 2222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| CodeChangeRisk | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| CodebaseResultRanking | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| DatabaseRowClassification | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| DomAdDetection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| DroneTacticalAction | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| GameGoalSelection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| GraphEdgeSelection | 100 | 0 | 0 | 0 | 0 | 0 | 1 | 0 / 0 | n/a | n/a |
| PatentSection | 2223 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| PlatformerControlSelection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| RobotSkillSelection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| RuntimeOutcomeVerification | 100 | 0 | 0 | 0 | 0 | 0 | 2 | 0 / 0 | n/a | n/a |
| SemanticLineMatch | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| SocialPostFiltering | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| StrategyCommandSelection | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| ToolActionImpact | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| ToolCallRouting | 100 | 0 | 0 | 0 | 0 | 0 | 1 | 0 / 0 | n/a | n/a |
| ToolRoute | 2222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| ToolSafetyGate | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | n/a | n/a |
| WorkflowDecisionChoice | 80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |

Reading the table:
- **RouteFinancial and RouteGeneralAssistant.** The DecisionBench states are paraphrases, so the state columns stay low. The
  upstream-original columns and the item join show the real overlap. The item join counts 1,030 and 137 rows; the text columns
  count fewer (968 and 107) because they require an exact token match after `[a-z0-9]` normalisation.
- **RelevanceScore.** The state hits are product-listing text. ESCI splits by query, not by product, so the same product appears
  under a different query in our ESCI-train rows. Only 54 rows match on the (query, product) pair.
- **Short-field hits in the synthetic tasks** (TaxDocument, ClaimEvidence, HomeAlert, SearchPlan, Trading). These are 6–12-token
  generic phrases such as "July 1, 2023 to September 30, 2023" or "No false positives in last 30 days" that also occur in
  teacher-written documents. No row is a near-duplicate.

### Strongest hits (row_id: training match)

- `routing-banking77-unable_to_verify_identity-00042`: the upstream utterance equals `banking77:train-3934` (64 of 64 13-grams).
  The DecisionBench paraphrase still shares 66% of its 13-grams with it.
- `routing-banking77-failed_transfer-00043`: the upstream utterance equals `banking77:train-7471`.
- `routing-clinc-pto_used-00030`: the state is "have i used half of my days off yet?", identical to `clinc150:train-6100`.
- `routing-clinc-what_is_your_name-00027`: the state is "what name should I use for you", identical to `clinc150:train-12027`.
- `scoring-esci-complement-01757`: the (query, product B00D8ZCH0U) pair is in `sqid_esci:B00D8ZCH0U`, with the same label C.
- `scoring-esci-substitute-01288`: the product listing is a 100% 13-gram match with `esci:text:5403`, which has a different query.
- `moderation-civil-comments-0001018` (ToxicitySeverity): the comment ("dude, you did a great job with your coverage of the Eight
  Bells record…") is identical to `civil_comments:4a5d232794fe16c1d4d48b85`.
- `musique-…3hop1__857_846_7870` (MultihopAnswer): 59% of the row's 13-grams come from SQuAD2/BoolQ Wikipedia paragraphs. The
  question does not match.

The five strongest rows for every task and signal are in `contamination-results.json` → `flagged_examples`. The full item lists
are under `targeted_item_checks.*.train_row_ids`.

## What could not be checked

1. **The final phase-2c manifest is not on disk.** The run's config records `manifest_sha256`
   `e2bc1f271cb1e73ae5517b4ce6edcb8cd61551f6f0328bc6f123c9a2739c390b` (39,515 rows). The pod was terminated, and
   `decision-p2c-teacher.tgz` was never pulled. GCS (`gs://<bucket>/decision-p2c/`) needs re-authentication, which
   was not attempted. The manifest's components were scanned instead:
   - the p2b rows (same texts; the relabel changes only the targets);
   - the exact Eikos slice;
   - the exact image replay;
   - the 9,000 programmatic questions, regenerated byte-for-byte from the deterministic generator with the pod's arguments. The
     generator file was last modified at 11:17 IST, before the run.

   **Not scanned:**
   - the judge- and unknown-family questions written by Qwen3.6-27B on the pod (the rest of the 9,880 new rows);
   - the Qwen3.6-35B rationale texts attached to about 25% of the text rows.

   These were model-written from domain prompts, with no dataset input. The pipeline's JevBench lint ran on them, but not a
   DecisionBench lint. We think they carry little risk, but that is not verified.
2. **Base model and teacher models.** The pretraining data of Qwen3.5-4B, and what the teacher models (Qwen3.6-27B/35B,
   gpt-oss-20b, the imajev-9b v1.1 teacher) memorised, cannot be inspected. Public sources such as banking77 and MuSiQue are
   likely in web-scale pretraining for any base model.
3. **Paraphrase-level overlap is not measured.** Beyond the four item joins, a DecisionBench row that is an LLM paraphrase of a
   text we trained on is found only if enough 13-grams survive. The item joins cover every paraphrased task whose upstream
   dataset we also used.
4. **Tokeniser scope.** `[a-z0-9]+` drops non-Latin scripts. DecisionBench is declared `eng-Latn`, and the ESCI item join uses a
   Unicode-aware key, so the effect should be small. It is still a limit of the n-gram columns.
5. **Images.** There was no image-level check; DecisionBench is text-only.
6. **Session paths.** Two inputs live in session scratchpads, not the repo: the local p2c pre-run build and the regenerated
   programmatic file. To re-run, regenerate the programmatic file with the command above and edit `SCAN` in the script.
