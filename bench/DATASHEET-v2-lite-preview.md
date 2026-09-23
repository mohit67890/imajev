# Datasheet: v2-lite-v1

Generated from the release check. Complete every TODO by hand.

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

- Release gate not met: construction_lint (Lint status fail: {'pass': 6, 'warn': 3, 'fail': 1})
- Release gate not met: image_necessity (No no_image harness run supplied; image necessity is unverified.)
- Release gate not met: statistical_power (89 test clusters; about 99 needed to detect a 10% paired difference (assumes 25% discordance, ICC 0.3).)
- All 458 image references are AI-generated (gpt-image-2.5-flare: 385, gemini-3.1-flash-image: 73). This version contains no real photographs, so results say nothing direct about performance on real-world photos.
- Generated images were kept only when two checker models confirmed every stated fact (passed: every stated fact confirmed by gpt-5.6-luna and gemini-3.1-pro-preview; no frame, identifiable faces or brands). This acceptance step favours those checker models on perception questions, so flag them in any comparison.
- Label routes: construction_verified: 532, unreviewed: 1. Construction labels are computed from the scene or text specification; a seeded sample and every occlusion-based Unknown are human-audited, and the audited error rate is reported with its upper bound.
- The audit of 138 items was done by AI models (claude-opus-5.5, kimi-k2.5), not by humans: blind answers from auditors outside the generator, checker and leaderboard roles, with every disagreement adjudicated by the pipeline author's model. Treat the audited error rate as a model audit.
- 45/533 references are Unknown (insufficient evidence).
- Scores depend on each model's interface and reasoning setting (direct option scoring vs structured generation; reasoning effort); every leaderboard entry must state both.

## Collection and licensing

TODO: sources, licenses, consent, removal contact.

## Annotation

TODO: annotator pool, training, pay, instructions, adjudication rule.

## Intended use and limits

TODO: typed decision evaluation only; not a safety or deployment certificate.

## Maintenance

TODO: versioning, errata process, hidden-test submission policy.
