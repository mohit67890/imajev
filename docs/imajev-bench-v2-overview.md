# imajevBench v2.0-lite: what was built and how it works

Status: **public preview**, 2026-09-23. This page summarises the whole effort. Details live in the
linked files.

## 1. What imajevBench measures

imajevBench tests whether a model can make a **typed decision from an image plus a written
situation**:
- read a menu or price board, a shelf, a timetable, a parking sign, a thermostat, a parcel label,
  a nutrition panel or recycling bins;
- apply the rules in the question;
- return an answer of the requested type, such as yes/no, one option, or a number on a scale;
- answer **Unknown** when the evidence is not there.

It is built for the small, on-device models that imajev targets, and it runs frontier API models as
reference points.

**Each item has:**
- 0–2 images;
- a short state (the situation, and rules such as "parking allowed 2 h, except Sundays");
- one typed question;
- a reference answer, which may be `null`, meaning Unknown.

**Three tracks:**

| Track | Evidence needed | Items (lite-v1) |
| --- | --- | ---: |
| text | only the written state | 75 |
| visual | only the image | 223 |
| joint | image **and** state together | 235 |

**Families** (skills tested):
- text reading, counting and spatial counting, spatial relation, attribute state, comparison;
- threshold rules, rule exceptions, multi-clause and multi-step rules, policy precedence;
- numerical reconciliation, date arithmetic;
- answerability and missing or conflicting evidence.

The full list is in `docs/imajev-bench-v2-families.md`.

## 2. Why a v2 was needed (the v1 audit)

The independent audit of v1 is in `reports/imajev-bench-independent-review/review.md`. It found
that the 113-case v1 was fit only as a smoke test:
- many items could be answered from text alone;
- there were too few independent evidence clusters for confidence intervals;
- labels were written by the same model family that was being tested;
- timing was contaminated.

The v2 plan (`docs/imajev-bench-v2-plan.md`, requirements R1–R17) fixes each of these. v2.0-lite is
the low-cost version of that plan.

## 3. How the items are built

**Principle: answers come from construction, not from model votes.**

1. **Scene specification** (`src/imajev_bench/scenes.py`). A seeded program writes the facts of a
   scene, for example a menu with 6 dishes and prices, a sign with hours and exceptions, or 7 mugs,
   3 of them blue.
2. **Image generation** (`scripts/imajev_bench/build_v2.py images`). The spec is rendered into a
   photo-style prompt:
   - Generators: **gpt-image-2.5-flare** (Azure, about $0.011 per image, most images) and
     **Nano Banana 2** (`gemini-3.1-flash-image`, Vertex).
   - Prompts forbid logos, brands, phone frames and identifiable faces.
3. **Variants: contrast sets.** Each scene is edited into variants that each test one thing:
   - **change**: one fact changes, so the answer must change;
   - **cover**: a sticky note or hand hides the needed value, so the answer becomes **Unknown**;
   - **same**: a harmless change, so the answer must stay the same.

   The expected relation between variants is computed automatically.
4. **Two-model image check.** `gpt-5.6-luna` and `gemini-3.1-pro-preview` each confirm every
   stated fact. They also check hygiene: no frame or UI, no identifiable face, no brand. An image
   is kept only when **both** pass it. Scenes are never dropped because a model answered a
   question wrongly.
5. **Questions and answers** (`build_v2.py items`). Questions are generated from templates. The
   answer is computed from the spec, including the hidden-value logic for Unknown. Text-only items
   come from `src/imajev_bench/textgen.py`, with standard and hard families.
6. **Assemble, lint and split** (`imajev_bench assemble`, `imajev_bench lint`).
   - Items are grouped into **evidence clusters**: the union of image hash, group, source cluster
     and contrast set. Whole clusters go to dev, calibration or test, so no image leaks across
     splits.
   - Lint checks: abstention cues in the wording, label balance, image reuse ≤3, split isolation,
     provenance completeness, and near-duplicate images (dHash).
7. **Audit** (section 5).

**Label routes** (`src/imajev_bench/schema.py`):
- `construction_verified`: the answer is computed and the image facts are confirmed by two model
  families. A seeded audit sample and every judgement-dependent item are audited as well.
- `consensus_verified` and `double_human`: supported for future human-labelled items.

## 4. The dataset (lite-v1)

- **Location:** `data/imajev-bench/v2-lite-v1/records-audited.jsonl`. This is the canonical file.
- **Items:** 533 over 98 scenes and 317 AI-generated images. 45 items (8%) are Unknown.
- **Splits:** dev 173, calibration 81 and test 279 items, over 173 clusters (89 in test).
- **Image origin:** Flare made 385 image references and Nano Banana 2 made 73. **There are no real
  photographs.**

## 5. Label quality: the AI audit

- **Scope:** 138 items. That is a 15% cluster sample, every occlusion-based Unknown, and every
  flagged item.
- **Auditors:** Claude Opus 5.5 (four sub-agents looking at every image) and Kimi K2.5. They
  answered **blind**, without the reference answers. Neither family generated, checked or appears
  on the leaderboard.
- **Adjudication:** every disagreement was checked on the image.
- **Result: 1 label error in 138 (0.7%), with a Wilson 95% upper bound of 4.0%.** In that item,
  translucent tape left a "hidden" value readable. It is quarantined: kept but not scored, with its
  gold unchanged so existing runs still verify.
- **Labelling:** the release says "model-audited" rather than "human-audited". Audit data is stored
  per item in `provenance.model_audit`, and the files are in
  `reports/imajev-bench-v2-lite-v1/audit/`.

## 6. How a model is evaluated

**Two interfaces** (`docs/imajev-bench-v2-preregistration.md`):
- **Direct option scoring** (local models, `scripts/imajev_bench/run_local_v2.py`). The model
  scores every allowed answer, including Unknown. Options are shown in every rotation to remove
  position bias. Backends: MLX on Mac, or PyTorch on GPU.
- **Structured generation** (API models, `imajev_bench api-run`). The model returns JSON that
  matches a schema, with prompt version `api-v2`. Supported providers:
  - Azure OpenAI
  - OpenAI
  - Gemini
  - Vertex
  - any OpenAI-compatible endpoint (`OPENAI_BASE_URL`), which is how local models are run through
    vLLM; the manifest records the base URL and served model

**Required disclosure:** every result must state its interface and its reasoning setting. GPT-5.4
scored 76% with no reasoning and 93% with medium reasoning on the hard trial.

**Controls:**
- `no_image`: this checks that the images are needed.
- `no_state`.
- Constant-answer baselines: family majority and image-blind.

**Metrics** (`src/imajev_bench/stats.py`, `scoring.py`):
- Accuracy, where Unknown counts as correct only when the reference is Unknown, and errors count
  as wrong.
- A **cluster-bootstrap 95% interval**. It is suppressed below 10 clusters.
- Per-track and per-family accuracy.
- Correct-Unknown rate and false-abstention rate.
- **Contrast-set consistency**: every variant of a scene must be correct.
- Paired cluster sign-flip tests for comparing two models.
- Latency, as a secondary measure.

**Integrity:** every run records a `scoring_sha256` over item IDs, splits, model inputs and gold.
`verify_run` refuses to score a run that doesn't match the records. The API runner refuses to send
test items unless `--allow-test-exposure` is passed.

## 7. Results (test split, 279 items)

Interfaces are ranked separately (pre-registration). Local rows: one H100, 23–24 Sept 2026.
Chance-corrected accuracy (chance 26.5%) and ECE / Brier are in `LEADERBOARD.md`.

Direct option scoring (single pass, full option rotations):

| Model | Accuracy (95% CI) | Notes |
| --- | --- | --- |
| imajev-9b (phase-2c soup50, released) | 82.1% (76–88) | phase-2b adapter 82.8% (77–88) |
| imajev-4b (phase-2c soup50, released) | 82.4% (77–89) | phase-2b adapter 82.4% (77–88), indistinguishable from the released 9B (paired test p = 1.0) |
| imajev-9b (earlier v1.1 adapter) | 81.0% (75–86) | pre-registered H1 vs its base: p = 0.03, confirmed |
| Qwen3.5-9B base | 76.7% (70–82) | |
| Qwen3.5-4B base | 70.6% (64–78) | |
| imajev-2b (phase-2c soup50, released) | 71.7% (65–78) | phase-2b adapter 70.3% (64–77); phase-2 adapter 68.5% |
| imajev-2b (earlier v2.1 adapter) | 63.1% (56–70) | 63.4% on a Mac with MLX (98.2% answer agreement); pre-registered H2 vs its base: p = 0.57, not significant |
| Qwen3.5-2B base | 60.2% (53–67) | |
| imajev-2b, no image | abstains on 258/279 | image-necessity control, not ranked |
| imajev-2b, no state | 51.6% (45–58) | state-necessity control, not ranked; joint track 46/122 vs 76/122 with the state |

Structured generation (JSON matching the schema, prompt `api-v2`):

| Model | Accuracy (95% CI) | Notes |
| --- | --- | --- |
| Gemini 3.1 Pro* | 99.6% (99–100) | *checked images during construction |
| GPT-5.6 Luna* | 99.3% (98–100) | *checked images during construction |
| GPT-5.4 (reasoning medium) | 98.9% (98–100) | |
| Gemini 3.8 Flash | 98.2% (96–100) | |
| Grok 4.3 | 91.4% (87–95) | no response schema; 24 format errors; abstained on business-day items |
| Qwen3.5-9B base | 74.6% (69–80) | vLLM, JSON schema, default reasoning |
| Qwen3.5-4B base | 67.4% (61–73) | vLLM |
| Gemma 4 E4B | 60.2% (53–68) | vLLM |
| Qwen3.5-2B base | 59.9% (54–66) | vLLM |
| Gemma 4 E2B | 58.8% (52–66) | vLLM |

- **Frontier models are saturated.** v2.0-lite is a sanity check for them, not a separator.
- **Small models are far from the ceiling and separate by size and by training,** which is the
  intended use.
- **Direct option scoring beats structured generation for the same base model at every size**
  (2B 60.2 vs 59.9, 4B 70.6 vs 67.4, 9B 76.7 vs 74.6); this is exploratory.
- **The controls show both inputs are needed:** without the image imajev-2b abstains; without the
  state its joint-track score drops from 76 to 46 of 122.
- **No home advantage:** generator-family effects were not visible in the per-generator
  breakdowns. With 73 Nano Banana 2 items, this is weak evidence.
- **Deferred:** direct-scoring rows for Gemma 4 E2B/E4B and SmolVLM2 (their harness backends are
  MLX-only; the Mac runs were stopped).

Full tables: `reports/imajev-bench-v2-lite-v1/RESULTS.md` and
`imajev-release/bench/LEADERBOARD.md`.

## 8. Hidden test for submitted models

- **Status: built and audited (24 Sept 2026).** `data/imajev-bench/private-1/records-audited.jsonl`,
  sha256 `b5b3fa93da5ecb7289b73ef7e24b1ef6059140869b65bf1c62d6458817d115cf`.
  - 202 items, all in test, in 66 evidence clusters: 30 text, 84 visual, 88 joint; 15 Unknown.
  - Built from 36 new scenes (all Flare images) and 30 text items (24 of them hard), with its own seed
    and split salt. Never published.
  - The same blind AI audit as the public set: 41 items (15% cluster sample plus every cover
    variant), Claude Opus 5.5 and Kimi K2.5. **0 label errors in 41** (upper 95% bound about 8.6%).
    One boundary case (a stay ending exactly when no-parking starts) is noted.
- **Tier A (weights):** the team runs the model offline, so hidden items never leave team machines.
- **Tier B (API endpoint):** the items are sent to the endpoint, and exposure rules apply.
- **Reports:** `scripts/imajev_bench/evaluate_submission.py` publishes **aggregate metrics only**:
  - accuracy and interval, per-track accuracy, Unknown and abstention rates, contrast consistency;
  - the hash of the hidden set used.

  Item-level results stay private.
- **Rotation:** the set rotates at least every 6 months, or sooner when it has been exposed.

Full policy: `docs/imajev-bench-hidden-test.md`.

## 9. Known limitations (disclosed, not hidden)

- All images are AI-generated. The results say nothing direct about real photos.
- The checker models (Luna and Gemini 3.1 Pro) selected which images were kept, so they may be
  favoured on perception items. They are starred on the leaderboard.
- The audit was done by models. The adjudicator is the pipeline author's model family.
- **Unmet release gates:**
  - Statistical power: 89 test clusters, where about 99 are needed to detect a 10-point paired
    difference.
  - Construction lint: one adjudicated near-duplicate false positive.
- **Minor realism issues the auditors raised:**
  - some implausible nutrition numbers;
  - a few self-cancelling rules;
  - small people at image edges in 3 scenes;
  - weekday timetables that don't state the day.
- The hidden set is not built yet (§8); when it is, the first version will be Flare-only.
- Gemma 4 and SmolVLM2 have generation rows only (Gemma) or no row yet (SmolVLM2): their
  direct-scoring backends are MLX-only.

## 10. Cost

Estimates from list prices; bills are authoritative.

| Item | Est. cost |
| --- | ---: |
| Image generation | ~$11 |
| Image checking | ~$2.40 |
| Evaluations | ~$10–15 |
| Trials | ~$15 |
| Audit and hidden set | a few dollars |
| **Total** | **~$45–50** |

## 11. Reproducing or extending

```
python scripts/imajev_bench/build_v2.py plan   --out <build> --scenes N --text N --text-hard N --seed S
python scripts/imajev_bench/build_v2.py images --out <build> --generator flare
python scripts/imajev_bench/build_v2.py items  --out <build> --dataset <name>
PYTHONPATH=src python -m imajev_bench assemble / lint / promote-constructed / release-check
PYTHONPATH=src python -m imajev_bench api-run --provider azure-openai --model <deployment> --split dev
```

Keys go in `.env.local`, which is gitignored. Tests: `pytest tests/test_imajev_bench_*.py` (117
passing).

**Open items for the owner:**
- licences (proposed: CC0 for images, CC BY 4.0 for labels);
- a contact address;
- the public README.
