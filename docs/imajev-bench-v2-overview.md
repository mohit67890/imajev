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
  - any OpenAI-compatible endpoint

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

| Model | Accuracy (95% CI) | Notes |
| --- | --- | --- |
| Gemini 3.1 Pro* | 99.6% (99–100) | *checked images during construction |
| GPT-5.6 Luna* | 99.3% (98–100) | *checked images during construction |
| GPT-5.4 (reasoning medium) | 98.9% (98–100) | |
| Gemini 3.8 Flash | 98.2% (96–100) | |
| Grok 4.3 | 91.4% (87–95) | no response schema; 24 format errors; abstained on business-day items |
| imajev-2b (v2.1) | 63.4% (56–70) | local, direct option scoring, MLX |
| imajev-2b, no image | abstains on 258/279 | image-necessity control, not ranked |

- **Frontier models are saturated.** v2.0-lite is a sanity check for them, not a separator.
- **Small models are far from the ceiling,** which is the intended use.
- **The no-image control shows the images are needed.**
- **No home advantage:** generator-family effects were not visible in the per-generator
  breakdowns. With 73 Nano Banana 2 items, this is weak evidence.
- **Still running:** imajev-9b and the base Qwen models, on an A100.

Full tables: `reports/imajev-bench-v2-lite-v1/RESULTS.md` and
`imajev-release/bench/LEADERBOARD.md`.

## 8. Hidden test for submitted models

- **What it is:** the imajevBench team keeps a **private test set** that is never published:
  `data/imajev-bench/private-1/`. It has 36 new scenes plus 30 text items (24 of them hard),
  built with a different seed and split salt.
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
- The first hidden set is Flare-only.

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
