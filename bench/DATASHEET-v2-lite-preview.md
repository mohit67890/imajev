# Datasheet: v2-lite-v1

Generated from the release check (pilot-size targets, preview status) on 23 Sept 2026; the descriptive sections were completed by hand.

## Composition

| Split | Text | Visual | Joint |
| --- | ---: | ---: | ---: |
| dev | 25 | 69 | 79 |
| calibration | 13 | 34 | 34 |
| test | 37 | 120 | 122 |

Unknown references: 45 of 533.

## Release gates

- PASS `human_review`: Every scored record has a validated label route (see limitations for the audit method); 1 quarantined record(s) are excluded from scoring.
- FAIL `construction_lint`: Lint status fail: {'pass': 6, 'warn': 3, 'fail': 1}
- PASS `independent_clusters`: Clusters per split/track meet targets.
- PASS `judgement_dependent`: 12/279 test records are marked judgement-dependent (4%).
- PASS `annotator_agreement`: Not applicable: no record uses the two-human review route.
- PASS `consensus_audit`: 532 labels use the model-consensus or construction route; audited 138 (model audit) with 1 errors (upper 95% bound 0.039902703895898395).
- FAIL `image_necessity`: No no_image harness run supplied; image necessity is unverified.
- FAIL `statistical_power`: 89 test clusters; about 99 needed to detect a 10% paired difference (assumes 25% discordance, ICC 0.3).

## Known limitations

- Release gate not met: construction_lint (Lint status fail: {'pass': 6, 'warn': 3, 'fail': 1}). The one failing check is the near-duplicate image check (dHash). The flagged images were inspected: they are two different nutrition panels (one in test, one in calibration) with different values that share the generator's template layout, so no answer transfers across splits. Adjudicated as a false positive (`adjudications.json`); images of one scene type can look template-like.
- Release gate not met: image_necessity (No no_image harness run supplied; image necessity is unverified.) Measured on the test split for imajev-2b instead: see "Image necessity" below.
- Release gate not met: statistical_power (89 test clusters; about 99 needed to detect a 10% paired difference (assumes 25% discordance, ICC 0.3).)
- All 458 image references are AI-generated (gpt-image-2.5-flare: 385, gemini-3.1-flash-image: 73). This version contains no real photographs, so results say nothing direct about performance on real-world photos.
- Generated images were kept only when two checker models confirmed every stated fact (passed: every stated fact confirmed by gpt-5.6-luna and gemini-3.1-pro-preview; no frame, identifiable faces or brands). This acceptance step favours those checker models on perception questions, so flag them in any comparison.
- Label routes: construction_verified: 532; 1 quarantined (a label error found by the audit, kept with its original gold and excluded from scoring). Construction labels are computed from the scene or text specification; a seeded 15% cluster sample, every occlusion-based Unknown and every flagged item were audited by AI models (next item), and the audited error rate is reported with its upper bound.
- The audit of 138 items was done by AI models (claude-opus-5.5, kimi-k2.5), not by humans: blind answers from auditors outside the generator, checker and leaderboard roles, with every disagreement adjudicated by the pipeline author's model. Treat the audited error rate as a model audit.
- 45/533 references are Unknown (insufficient evidence).
- Scores depend on each model's interface and reasoning setting (direct option scoring vs structured generation; reasoning effort); every leaderboard entry must state both.

## Collection and licensing

- **Images.** All 458 image references are AI-generated for this benchmark from structured scene specifications
  (`src/imajev_bench/scenes.py`): 385 by `gpt-image-2.5-flare` (Azure OpenAI) and 73 by `gemini-3.1-flash-image` (Vertex AI).
  Every generated image was kept only when two checker models (`gpt-5.6-luna` and `gemini-3.1-pro-preview`) confirmed each stated
  fact and found no frame, identifiable face or brand; failures were regenerated at most twice. No photographs of people and no
  private documents appear. There are no real photographs in v2.0-lite; the real-photo strata planned in
  `docs/imajev-bench-v2-plan.md` §5 (commissioned CC BY 4.0 photos and recent public CC0/CC BY photos) are deferred to a later version.
- **Text items.** 75 text-only items authored programmatically from rule and state templates; no third-party text is reproduced.
- **Licences.** Records, questions, labels and scene specifications: CC BY 4.0. Generated images: released under CC0-1.0 as recorded
  in each record's `provenance.image_sources[].license`; the generator and its terms of use are recorded per image. Harness and scorer
  code: Apache-2.0 (in the `imajev` repository).
- **Consent.** Not applicable: no people, no user-contributed content.
- **Removal or correction contact.** Open an issue in the `imajev` GitHub repository (https://github.com/mohit67890/imajev) with the
  record id; errata are versioned (see Maintenance).
- **Provider terms.** Images were generated and checked through Azure OpenAI and Google Vertex AI under their standard terms; test-split
  images were sent to those providers during construction, so the test split is not blind to those two model families (disclosed in
  the leaderboard with a `*` on models that also checked images).

## Annotation

- **Label route.** Answers are known by construction: each is computed from the scene or text specification, including after
  one-change edits (`provenance.construction`). Models never vote on answers.
- **Audit (AI, not human).** 138 items (a seeded 15% cluster sample, every occlusion-based Unknown and every flagged item) were
  answered blind by Claude Opus 5.5 and Kimi K2.5, two model families outside the generator, checker and leaderboard roles. Every
  disagreement was adjudicated on the image by the pipeline author's model family. Result: 1 label error in 138 (0.7%, upper 95%
  bound 4.0%); the item is quarantined. Per-item audit data is in `provenance.model_audit`. A human spot-check of the same packet
  would upgrade this to a human audit; it has not been done.
- **Annotator pool, training, pay.** No human annotators or paid labellers were used in v2.0-lite. The `consensus_audit` gate
  (audited error rate, target ≤ 2%) is met by the model audit: 0.7%, upper bound 4.0%.
- **Adjudication rule.** A flagged or disagreeing item quarantines its contrast group; ambiguous items are dropped or rewritten, never
  forced (plan §7). Two likely label errors found in error triage (`adjudications-triage.json`) were in the audit sample; one was confirmed and quarantined, the other confirmed correct.
- **Conflict of interest.** The imajev developer authored and maintains this benchmark. The external reviewer required by plan §10
  is TBD by owner; until then, imajev results on this benchmark are self-reported.

## Intended use and limits

- **Use.** Evaluating typed decisions (yes/no, listed choice, level, or Unknown) made from a photo, a written rule, or both, by small
  and local models where differences are expected; frontier API models are near the ceiling here (about 99%) and this version does
  not separate them.
- **Not for.** Safety or deployment certification, free-form generation quality, latency claims outside the stated protocol, or any
  domain the scenes do not sample (menus, signs, price tags, nutrition panels, shelves, parking rules, invoices and similar).
- **Limits.** All images are synthetic, so results say nothing direct about real photographs; only differences of roughly 10 points or
  more are detectable (89 test clusters); labels are model-audited, not human-audited; every leaderboard entry must state interface and
  reasoning setting because they change scores by tens of points.
- **Scoring rule.** Unknown is correct only when the reference is Unknown: abstaining on an answerable item is wrong, answering an
  Unknown-reference item is wrong, and errors are wrong. The leaderboard also reports chance-corrected accuracy (chance = 26.5% on the
  test split, a uniform guess over each item's allowed answers including Unknown).
- **Training.** Do not train on the test split. Dev and calibration may be used for development and calibration fitting.

## Maintenance

- **Versioning.** Data and scorer carry semantic versions (this is `v2.0-lite`); scores are compared only within one data version.
  An `ERRATA.md` lists every label change with the record id and date.
- **Test labels.** Test images and requests are public; test labels are withheld in the public records (`gold_withheld: true`) and
  held by the maintainers. Submit a predictions JSONL produced by the harness (with its manifest and completion receipts) through a
  GitHub issue; up to 3 scored submissions per team per month. Test labels are released after 12 months or when v3 ships.
- **Hidden test.** The maintainers also hold a private test set that is never published (202 items, 66 evidence clusters, same
  families and construction, model-audited). Teams can send model weights (run offline by the maintainers) or an API endpoint; only
  aggregate metrics and the private set's hash are published. Policy: `docs/imajev-bench-hidden-test.md` in the code repository.
- **Contamination canary.** TBD by owner: the canary GUID field in plan §10 is not yet embedded in the v2.0-lite records.
- **Errata and takedown.** GitHub issues in the `imajev` repository; corrections ship as a new patch version.

## Image necessity (measured on the test split)

The `image_necessity` gate above reads "unverified" because the release check consumes a no-image run over all splits, and only the
test split was run before local evaluation was stopped. On the test split, imajev-2b without images answers 21 of 279 items and
abstains on 258; its accuracy on the 227 answerable visual and joint items is 0.0% against a family-majority constant of 37.4%, so
the gate condition (no-image accuracy at or below family majority + 10 points) holds for that model on that split
(measured on the earlier imajev-2b). The dev and calibration no-image runs and the other models' no-image controls
were not run in this preview (see Local-model runs: status).

## Local-model runs: status

- Completed on one H100 80 GB (RunPod, 23 Sept 2026), direct option scoring (PyTorch bf16, full rotations, single-shot timing):
  the earlier imajev-9b, Qwen3.5-9B base, Qwen3.5-4B base, the earlier imajev-2b (also on a Mac Studio with MLX, 98.2% answer agreement),
  Qwen3.5-2B base, and the imajev-2b `no_state` control. Structured generation through vLLM 0.30.0 (JSON schema, default reasoning):
  Qwen3.5-2B/4B/9B base, Gemma 4 E2B-it, Gemma 4 E4B-it.
- Completed on the training pods (one H100 per run, 24 Sept 2026), same settings and scoring hash: the released adapters
  imajev-4b (82.4%), imajev-9b (82.1%) and imajev-2b (71.7%), their previous versions (phase-2b: 82.4%, 82.8% and 70.3%), and the
  pre-release checkpoints from before the last part of the hard-question stage (imajev-2b 68.5%, and the 4B and 9B). The pre-registered
  tests H1 (imajev-9b vs its untuned base) and H2 (imajev-2b vs its untuned base) were re-run for each replaced adapter and are reported
  beside the registered results.
- Completed on the training pod (24 Sept 2026): the released adapters and their previous versions on the hidden test set (202 items);
  only aggregates are published (`LEADERBOARD.md`).
- Completed on a Mac Studio (MLX): imajev-2b `no_image` control, test split.
- Deferred: direct-scoring rows for Gemma 4 E2B/E4B and SmolVLM2-2.2B (MLX-only backends; Mac runs stopped); no-image controls for
  models other than imajev-2b; dev and calibration no-image runs.
- Results: `LEADERBOARD.md` in this folder.
