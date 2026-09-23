# imajev-bench v2: plan for a public benchmark

Status: 2026-09-23. The tooling and the synthetic build pipeline are built and tested. The first
public version is **v2.0-lite**, a low-cost preview built mostly with AI (see "v2.0-lite" below).
The full-size targets in §4 remain the goal for later versions.

## v2.0-lite: how the first public version is built

- **Size.** About 450 items (about 300 test, 90 dev, 60 calibration) from about 120–150
  independent scenes. It can detect differences of about 10 points or more.
- **Images.** AI-generated scenes, split between `gpt-image-2.5-flare` (Azure) and
  `gemini-3.1-flash-image` (Vertex) and reported per generator; plus programmatic text items; plus
  recent public CC photos where available.
- **Generator choice.** From the bake-off in `reports/imajev-bench-v2-bakeoff/run-1/SUMMARY.md`.
- **Labels.** Answers are **known by construction**:
  - Each scene is generated from a structured fact spec (`src/imajev_bench/scenes.py`), and every
    answer is computed from those facts, including after one-change edits.
  - `gpt-5.4` and `gemini-3.1-pro-preview` only check that each generated image shows every stated
    fact. Images fail on any doubt and are regenerated, at most 2 retries.
  - Models never vote on answers, and no item is dropped because a model answers it wrongly.
  - A seeded 15% cluster audit, plus any judgement-dependent item, gets one blind human check.
  - The audited error rate is published, and the release needs 2% or less.
- **Pipeline.** `scripts/imajev_bench/build_v2.py` runs plan → images → items, then:
  - `imajev_bench assemble` and `lint`;
  - `promote-constructed`, then the `review --plan audit` packet;
  - `import-reviews`, then `release-check --pilot`.
- **Cost.** About $50–100 of API spend plus a few hours of human audit.
- **Limits, disclosed in the datasheet.** Most images are synthetic, so real-world claims rest on
  the real-photo subset. Only large differences are detectable.

This plan builds on `docs/imajev-bench-proposal.md` and on the problems found in
`reports/imajev-bench-independent-review/review.md` (findings F1–F12 below). Every design rule
names the audit finding it fixes and the tool that enforces it, so the rules cannot drift.

## 1. What the benchmark claims

**imajev-bench measures typed decisions made from an image, a written rule, or both.** A system is
given the evidence and a rule, and returns one of: yes/no, a listed choice, a level, or Unknown.
Unknown means the supplied evidence does not determine the answer. The benchmark checks four things:

- whether the decision is correct;
- whether the system abstains exactly when it should;
- whether its answer changes when, and only when, the evidence changes;
- optionally, whether its probabilities are calibrated.

It makes no claims about:

- free-form generation quality;
- safety;
- production latency outside the stated timing protocol;
- any domain it does not sample.

A leaderboard entry states the model revision, precision, prompt profile, visual budget and
decoding interface.

## 2. Design rules learned from v1

| # | Rule | Fixes | Enforced by |
| --- | --- | --- | --- |
| R1 | Unknown must not be predictable from wording. Use the same instruction style on answerable and unanswerable items, and put "if not shown, answer unknown" clauses on answerable items too. | F1 | `lint: abstention_cues` (fail when the cue-rate gap exceeds 0.25) |
| R2 | No family may be solvable by one constant answer. | F1, F2 | `lint: label_balance` (fail when a family's majority answer exceeds 65%) |
| R3 | At most 3 records per image. Grow scenes before growing questions. | F3 | `lint: image_reuse`, `assemble` cap |
| R4 | Intended contrasts are declared as sets, and scored as sets: all-correct and change-pair correctness. "Invariant unchanged" is never a headline metric on its own. | F2 | `stats.contrast_metrics`, `lint: contrast_sets` |
| R5 | Contrast edits vary in direction and size. Originals are not always on the threshold. Some variants look relevant but preserve the answer. | F2 | `lint: contrast_sets` (constant-answer baseline and skewed-original checks) |
| R6 | Whole evidence clusters stay in one split (shared image, source scene, contrast set). | F3 | `lint: split_isolation`, `assemble` split assignment |
| R7 | Every interval and paired test resamples evidence clusters, never rows. | F3, F12 | `stats.cluster_bootstrap`, `stats.paired_cluster_test`, CLI `score`/`compare` |
| R8 | Every label has a defensible source, and each source is audited. The three routes are: (1) two blind humans with adjudication; (2) unanimous model pre-labels from two families plus one agreeing blind human; (3) an answer known by construction (from a generator spec with images fact-checked by two model families, or computed) plus a seeded human audit. Model labels never count as reviews, and no item is removed because a model got it wrong. | F9 | `schema` (`double_human`, `consensus_verified`, `construction_verified` routes), `triage`, `annotations`, `release: consensus_audit` |
| R9 | Items that depend on a perceptual judgement call (occlusion, colour naming) are flagged. They are capped at 10% of test and reported separately. | F8 | `provenance.judgement_dependent`, `release: judgement_dependent` |
| R10 | Image necessity is shown, not assumed. A no-image run of each open model must score at most the family-majority baseline plus 10 points on answerable visual and joint items. | F1 | `harness --condition no_image`, `release: image_necessity` |
| R11 | The harness's Unknown wording matches the benchmark's definition. The imajev served wording is a secondary profile. | F6 | `harness` profile `benchmark-neutral` (default) |
| R12 | Full answer-order rotations, so every option appears in every position. Rotation agreement is reported. | F6 | `harness` `rotations="full"`, `rotation_agreement` per case |
| R13 | Timing needs warm-up, 3 or more repeats, a per-case contention monitor and a per-case memory peak. Contaminated cases are excluded from latency tables. | F4, F5 | `harness` (`timing_contaminated`, `latency_repeats_ms`, `peak_memory_bytes`) |
| R14 | A run snapshots every project source file and `pip freeze`. | F10 | `harness.environment_receipt` |
| R15 | Every submission is scored by the same scorer. | F12 | local runs write `manifest.json`/`completion.json` that `imajev_bench score` verifies |
| R16 | The answer must not be quoted in the state while the distractors are absent. | F1 | `lint: answer_in_state` |
| R17 | No near-duplicate images across splits. Real sources carry licence and attribution. | F3 | `lint: near_duplicates`, `lint: provenance` |

## 3. Task design

**Tracks.** There are three tracks, each reported separately. The overall score is their unweighted
mean, and only when all three are complete.

- **text:** decisions from state or policy text only.
- **visual:** decisions from images only, where the state holds just a neutral instruction.
- **joint:** decisions that need both images and a written rule.

**Families.** Target 12, announced before collection. Each has at least 30 test clusters in its
track.

| Track | Family | Example decision |
| --- | --- | --- |
| visual | text reading | Which price is printed for item X? |
| visual | counting | How many separate containers are visible? |
| visual | attribute and state | Is the lid open? Which colour is the cap? |
| visual | spatial relation | Is the sign above or below the arrow? |
| visual | answerability | Is the year printed? The answer can be yes, with a year. |
| joint | threshold rule | Approve if the visible total is at most N. |
| joint | multi-clause rule | Approve only if clause A and clause B both hold. |
| joint | two-image comparison | Does image 1 have more items than image 2? |
| joint | rule precedence and exceptions | A later clause overrides an earlier one. |
| text | numerical reconciliation | Totals, dates and units. |
| text | policy precedence and ties | Priorities, exceptions, undefined ties that make the answer Unknown. |
| text | missing or conflicting evidence | Pending counts, disagreeing sources. |

**Answer types.** Boolean, choice (2–6 options) and ordinal (3–5 levels), roughly 35/40/25 per
track. The harness adds Unknown to every item. Items whose honest answer is "none of these" must
list an explicit `other` option, or be rejected at curation. Unknown never means "no option fits".

**Unknown share.** 15–25% per track, spread across families. No family is 100% Unknown.

**Contrast sets.** 25–35% of visual and joint clusters. A set has one original plus 1–3 variants,
each labelled `relation: change` or `relation: same`. Variants are of three kinds:

- **Image edits:** fresh photos of the same scene with an object added or removed, or re-rendered
  diagrams.
- **Rule edits:** a stricter or looser threshold, or an added clause.
- **Order controls:** swapped images with ordinal references updated.

The direction and size of each change are randomised. At least a third of variants are `same`, and
answer-preserving edits must be visually salient.

**Strata.** Real photos and synthetic renders are reported separately, and at least 70% of visual
and joint clusters are real photos. Difficulty (easy/standard/hard at 25/50/25) is set by rule and
evidence complexity, not by degrading the image.

## 4. Size

Sizes come from `stats.required_clusters`, assuming 25% discordance between two systems and a
within-cluster correlation of 0.3. Measured correlation on v1 was near 0, but 113 cases cannot
support that estimate.

| Goal | Items | Clusters |
| --- | ---: | ---: |
| Detect a 5-point pooled paired difference | ~1,250 test items | ~415 clusters at 3 items/cluster |
| Detect an 8-point difference within one track | ~480 test items | ~160 clusters per track |
| Pilot: detect a 10-point pooled difference | ~300 test items | ~100 clusters |

**Release v2.0 targets.** These are encoded as `RELEASE_TARGETS` in `release.py`.

| Split | Clusters per track | Items (all tracks) | Visibility |
| --- | ---: | ---: | --- |
| test | ≥ 170 | ~1,500 | Hidden labels |
| calibration | ≥ 35 | ~300 | Public; for fitting temperatures and thresholds only |
| dev | ≥ 35 | ~300 | Public, with labels |

That is about 2,100 items, about 720 clusters and about 600 unique photos.

**Pilot v2.0-pilot targets** (`PILOT_TARGETS`): about 450 items and 60 clusters per track, with test ≥ 35 per track.

## 5. Sourcing

1. **Primary source: newly commissioned photographs.** These are the only credible protection against
   pretraining exposure, which cannot be ruled out for Wikimedia images. Photographers sign a release
   granting CC BY 4.0 and confirming that no identifiable people or private documents appear. Each
   photographer shoots whole scenes, with 1–3 shots per scene, and each scene is one
   `source_cluster`.
2. **Secondary source: CC0 or CC BY photos.** These can come from Wikimedia Commons, Openverse or
   similar, restricted to images uploaded after the newest evaluated model's training cutoff where
   that can be determined. They are reported as the "public-source" stratum, and each receipt keeps
   the source page, creator, licence and download hash (see `acquire_fresh.py`).
3. **Synthetic renders.** These cover exact counts and boundaries, with generator and version recorded.
   Use at least 4 distinct renderers or layouts, and no fixed "irrelevant" edit.
4. **Preparation and screening.**
   - Normalise orientation and strip EXIF.
   - Blur incidental faces and licence plates.
   - Run the existing exposure inventory (`build_exposure_inventory.py`) against every imajev training
     manifest, and the near-duplicate check against v1 and the other splits.
5. **Retire v1 material.** All v1 data (real-pilot, interventions, harder, pilot-v0.x) becomes dev
   examples only. None of it goes into calibration or test.

## 6. Authoring

- **Authors write specs, not records.**
  - Each source has an `id`, `path`, `source_cluster` and `provenance`.
  - Each item has: `id`, `track`, `family`, `images`, `state`, `field`, `draft_gold`, `evidence`,
    `answerability`, and optionally `contrast`, `judgement_dependent` and `difficulty`.
  - `imajev_bench assemble --spec spec.json --output data/imajev-bench/v2-pilot` then copies the
    assets under content hashes, assigns whole clusters to splits with the spec's `split_salt`,
    enforces the per-image cap, and validates the result.
- **Item-writing rules.**
  - Use neutral wording.
  - The answer must depend on visible evidence, not world knowledge.
  - Distractors must be plausible and appear elsewhere in the scene where possible.
  - Place the correct option at a random position.
  - Avoid "fictional record" premises that join unrelated photos. The v1 harder cases invited
    "premise is false" abstentions.
- **Model assistance is allowed, with limits.** A model may draft distractors or paraphrases, but
  every such item is marked in provenance and reviewed by humans. Model review never counts as
  annotation.
- **Separate roles.** A different person authors, annotates and adjudicates each item.
- **Lint the draft.** Run `imajev_bench lint --allow-draft` and fix every `fail` before annotation
  starts.

## 7. Annotation

The full procedure is in `docs/imajev-bench-v2-annotation.md`. Its model-assisted triage (decided
2026-09-23) works like this:

- OpenAI and Gemini models pre-label every item blind.
- Items where they agree, and that are not judgement-dependent or audited, get one blind human
  label. The label is accepted only if the human agrees with the models.
- All other items follow steps 1–3 below.
- A 15% cluster audit of consensus items gets two humans, and the measured consensus error rate is
  published.
- Hidden-test images go to a provider only under no-training terms; otherwise the item gets two
  humans.
- The single-human route saves roughly a third of the human time: about 75 hours instead of 115.

The double-human procedure:

1. Two annotators per item, working independently from blind packets (`imajev_bench review`). The
   packets show the images, state, question and options, but no draft label, rationale or grouping.
2. Annotators record a value, an evidence note (the visible region or rule clause), and a flag for
   ambiguity. Any flagged item quarantines its group.
3. A third person adjudicates disagreements (`import-reviews` with purpose `adjudication`). Items
   that stay ambiguous are dropped or rewritten, not forced.
4. Quality targets are Cohen's kappa ≥ 0.70 overall, reported per family, and adjudication on
   15% of items or fewer. Families below kappa 0.6 are rewritten.
5. **Annotator pool.** At least 6 trained annotators with a written guide
   (`docs/imajev-bench-annotation.md`, updated for v2), a 30-item calibration round with feedback,
   and fair pay.
6. **Effort.**

   | Task | Estimate |
   | --- | --- |
   | Annotation | ~2,100 items × 2 reviews, plus ~15% adjudication: ~4,500 judgements at ~1.5 min each, about 115 person-hours |
   | Authoring | ~175 hours |
   | Photography | ~600 photos |

## 8. Evaluation protocol

**Two interfaces, reported in separate columns.**

1. **Direct-option scoring** for open-weights models (`scripts/imajev_bench/run_local_v2.py`).
   - Native template and processor.
   - Profile `benchmark-neutral`.
   - Full rotations.
   - Unknown appended as a candidate.
   - Candidate log-probabilities averaged across rotations.
   - Uncalibrated, unless calibration is fitted on the calibration split only.
   - FP32 candidate projection wherever the backend supports it.
2. **Structured generation** for API and closed models (the HTTP adapter in `runner.py`). The model
   returns JSON. Malformed or out-of-domain outputs count as errors, never as abstentions, and the
   parse-failure rate is reported.

**Required runs per model.**

- `full`, on all splits.
- `no_image`, on visual and joint items.
- `no_state`, on joint items.
- Optionally the `imajev-native` profile.

Visual budget:

- Report each model's native visual-token count.
- Where the processor allows it, also run a matched budget of about 1,000 visual tokens per image as
  a secondary column.
- A model whose token budget is fixed by its processor, such as Gemma's roughly 280, is labelled as
  such.

**Timing, reported only as a secondary column.**

- Idle machine, confirmed with `--gpu-coordinated`.
- 2 warm-up records and 3 repeats.
- Report the median of repeats, and p50/p95 over cases that were not contaminated.
- Report precision, and peak memory reset for each case.

## 9. Metrics and leaderboard

| Column | Definition |
| --- | --- |
| Track score | 100 × the unweighted mean of family accuracies. Unknown counts as correct only when gold is Unknown; errors score 0. Cluster-bootstrap 95% CI. |
| Contrast score | Share of contrast sets that are all-correct, plus change-pair accuracy. The constant-answer baseline is shown beside it. |
| Abstention | Correct-Unknown rate, false-abstention rate, and Unknown F1. |
| Image necessity | Full minus no-image accuracy on answerable visual and joint items. A diagnostic, not a ranked score. |
| Stability | Mean rotation agreement (direct-option interface only). |
| Probability quality | Brier, NLL and ECE for systems that submit distributions. Calibration fitted on the calibration split only. |
| Selective | Risk–coverage and AURC. Coverage at 5% error, with the threshold fixed on the calibration split. |
| Efficiency | p50/p95 latency and peak memory under the §8 protocol, with hardware and precision stated. |

**Ranking rule.** Models are ordered by the mean track score. Neighbours whose paired cluster
sign-flip test gives p ≥ 0.05 (`imajev_bench compare`) share a tie group, and no rank number is
printed inside a tie group.

**Baselines published with the release.**

- Controls: random, always-first, always-Unknown, family majority, and the lexical-cue heuristic.
- The no-image run of every open model.
- Open models: Qwen3.5-2B, Qwen3.5-9B, Gemma 4 E4B, and SmolVLM2. SmolVLM2 is included only if it
  passes a per-type format smoke check; otherwise it is labelled "interface-limited".
- imajev 1.1 in its 2B and 9B versions.
- At least one frontier API model, through the structured-generation interface.

## 10. Test integrity and submissions

- **Hidden test labels.** Test images and requests are public, and the labels are held by the
  maintainers. Submitters upload prediction JSONL, which is checked with `verify_run`-style receipts
  and then scored. The limit is 3 scored submissions per team per month.
- **Contamination canary.** Test records carry a canary GUID, and the datasheet asks model trainers
  to filter it out.
- **Refresh cycle.** Test labels are released after 12 months, or when v3 ships, and that split then
  retires to dev.
- **Conflict of interest.** The imajev developer maintains the benchmark. An external reviewer
  therefore:
  - curates or audits the hidden test;
  - signs off the release check;
  - pre-registers the analysis in this document before any imajev model sees the test split.
- **Versioning.** Semantic versions apply to data and scorer, and an errata file lists every label
  change. Scores are only compared within one data version.

## 11. Release checklist

Run the release check with the no-image runs:

`imajev_bench release-check --records <all splits>/records.jsonl --output reports/imajev-bench-v2/release.json --blind-runs <no_image runs...>`

Every gate must pass:

1. `human_review`: every record is reviewed.
2. `construction_lint`: no failing lint check (R1–R6, R16, R17).
3. `independent_clusters`: every split and track meets its cluster target.
4. `judgement_dependent`: 10% of test or fewer.
5. `annotator_agreement`: kappa ≥ 0.70.
6. `image_necessity`: no-image runs are at or below the family majority plus 10 points.
7. `statistical_power`: the test split has enough clusters for the target difference.

The release check also writes `DATASHEET.md`. Its TODOs (sources, consent, annotator pay,
intended use, maintenance) are completed by hand. Publication also needs a licence (CC BY 4.0 for
annotations; images keep their own licences), a takedown contact, the external reviewer's
sign-off, and a public baseline report generated by the scorer.

## 12. Phases

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| 0. Tooling | Linter, cluster statistics, harness, assembler, release gate (done; `tests/test_imajev_bench_v2.py`) | Tests pass |
| 1. Spec freeze | Final family list, annotation guide v2, pre-registered analysis, external reviewer named | Signed off |
| 2. Pilot collection | ~180 clusters, about 450 items, drafted and linted | Lint has no fails |
| 3. Pilot annotation | Two human reviews plus adjudication | Kappa ≥ 0.70; guide revised |
| 4. Pilot evaluation | All baselines, full and ablation runs, `release-check --pilot` | All pilot gates pass; families that fail image necessity are rewritten |
| 5. Scale-up | ~720 clusters, about 2,100 items; pilot items move to dev | Lint clean |
| 6. Annotation | Same protocol | Kappa ≥ 0.70 |
| 7. Release | `release-check` passes, datasheet, baseline report, submission process | External sign-off |

## 13. Decisions

Decided on 2026-09-23:

- **Photos.** A mix: about 60% commissioned (`docs/imajev-bench-v2-photo-brief.md`) and about 40%
  recent public CC0/CC BY photos, reported as separate strata.
- **Labels.** Model-assisted triage with human decisions (§7).
- **Interfaces.** Both direct option scoring and structured generation (`api_models.py`).

Still open:

1. The human annotators for the single-route and double-route reviews: who they are, and the
   budget.
2. The external reviewer.
3. Hidden-test scoring: a manual process run by the maintainers, or an evaluation server.
4. The exact OpenAI and Gemini model IDs for pre-labelling and evaluation. These are fixed when the
   runs are made.

## 14. Tooling map

| Command or module | Purpose |
| --- | --- |
| `python -m imajev_bench assemble --spec S --output D` | Spec to records, with cluster splits and the reuse cap |
| `python -m imajev_bench lint --records R --allow-draft --output O` | Construction and shortcut checks |
| `python -m imajev_bench review` / `import-reviews` | Blind human packets and review promotion (existing) |
| `scripts/imajev_bench/run_local_v2.py --model M --condition C --gpu-coordinated` | Direct-option runs (full, no_image, no_state) |
| `python -m imajev_bench score --predictions P --split S` | Scores with a cluster CI and contrast-set metrics |
| `python -m imajev_bench compare --predictions-a A --predictions-b B --split S` | Paired cluster sign-flip test and interval |
| `python -m imajev_bench release-check --records R --blind-runs ...` | Release gates and datasheet draft |
| `src/imajev_bench/stats.py` | Evidence clusters, bootstrap, paired test, contrast metrics, ICC, power |
| `python -m imajev_bench api-run --provider azure-openai\|vertex-gemini\|openai\|gemini --model ID [--purpose prelabel]` | Structured-generation runs and model pre-labels (Azure key in `.env.local`; Vertex via the gcloud login) |
| `python -m imajev_bench import-prelabels --runs ...` | Attach model pre-labels without touching reviews or gold |
| `python -m imajev_bench plan-reviews [--audit-share 0.15]` | Route items to single or double human review, with the audit sample |
| `python -m imajev_bench review --plan double` | Second-reviewer packet for double-route items |
