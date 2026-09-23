# imajev-bench v2.0-lite (preview): first results

Status on 2026-09-23: a **preview**. The labels are not yet human-audited, and five release gates
are unmet (listed in `release/DATASHEET.md`). Numbers can change after the audit.

## Dataset

- **Items:** 533 (`data/imajev-bench/v2-lite-v1/records-final-v2.jsonl`).
  - 75 text items.
  - 223 visual items and 235 joint items, over 317 AI-generated images from 98 scenes.
  - 45 items (8%) have Unknown as the reference answer.
- **Splits:** dev 173, calibration 81, test 279. They are split by evidence cluster: 173 clusters in
  total, 89 of them in test.
- **Images:** all AI-generated. `gpt-image-2.5-flare` made 385 image references and
  `gemini-3.1-flash-image` made 73. There are no real photographs in this version.
- **Labels:** answers are known by construction: each is computed from the scene or text
  specification.
  - Generated images were kept only when `gpt-5.6-luna` and `gemini-3.1-pro-preview` both confirmed
    every stated fact.
  - 400 labels are promoted. 138 items wait for a blind human audit (a 15% cluster sample, every
    occlusion-based Unknown, and every flagged item).

## Results: full set (533 items)

All models use structured JSON generation with the prompt `api-v2`. The reasoning setting is part
of each entry. `*` marks a model that also checked the images during generation, so acceptance
favours it on perception.

| Model (setting) | Accuracy | Text | Visual | Joint | Correct Unknown | False abstention | Errors | Contrast sets all correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash (default thinking) | 527/533 (99%) | 75/75 | 218/223 | 234/235 | 45/45 | 1/488 | 5 | 94/96 |
| GPT-5.4 (reasoning medium) | 527/533 (99%) | 75/75 | 223/223 | 229/235 | 41/45 | 1/488 | 1 | 90/96 |
| GPT-5.6 Luna* (default) | 526/533 (99%) | 75/75 | 223/223 | 228/235 | 39/45 | 0/488 | 1 | 91/96 |
| Grok 4.3 (default, no response schema) | 483/533 (91%) | 46/75 | 211/223 | 226/235 | 40/45 | 21/488 | 24 | 82/96 |
| Gemini 3.1 Pro* (default thinking) | 525/533 (98%) | 73/75 | 222/223 | 230/235 | 41/45 | 1/488 | 3 | 91/96 |

**Gemini 3.1 Pro was rerun.** Vertex capacity for this preview model ran out partway through the
first run (HTTP 429 even on single requests), leaving 51 capacity errors. A retry job waited for
capacity and reran all its splits; the table shows that clean rerun. The error-containing first runs
are kept in `eval-aborted/`.

The 95% intervals, from a cluster bootstrap, are in `eval-summary/summary.md`.

## Robustness

| Model | Excluding the 23 flagged items | Flare images | Nano Banana 2 images | Text only |
| --- | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash | 505/510 (99.0%) | 380/385 | 72/73 | 75/75 |
| GPT-5.4 medium | 507/510 (99.4%) | 382/385 | 70/73 | 75/75 |
| GPT-5.6 Luna | 504/510 (98.8%) | 379/385 | 72/73 | 75/75 |
| Grok 4.3 | 469/510 (92.0%) | 369/385 | 68/73 | 46/75 |
| Gemini 3.1 Pro | 504/510 (98.8%) | 381/385 | 71/73 | 73/75 |

- **No visible home advantage.** OpenAI models do not score higher on OpenAI-generated images, and
  Gemini does not score higher on Google-generated images. With only 73 Nano Banana 2 image items,
  this is weak evidence either way.

## What this preview shows

1. **Frontier API models are near the ceiling.** Raw accuracy is about 99%. Their remaining errors
   sit in rules with exceptions, multi-step orders and Unknown handling:
   - GPT-5.4 medium and Luna miss about 4 of 68 rule-exception items each.
   - Covered sign hours are often treated as if the restriction applies.

   v2.0-lite does not separate frontier models, and it is disclosed as a sanity check for them.
2. **The intended use is small and local models.** These are the models doing typed decisions on
   device, where differences are expected. Those runs (Qwen 3.5 2B/9B, imajev 1.1 2B/9B, Gemma 4
   E4B), and the no-image baseline, are pending the owner's choice of hardware (Mac or A100).
3. **The reasoning setting matters a lot.** In the hard trial, GPT-5.4 scored 76% with Azure's
   default of no reasoning and 93% with medium reasoning. Every entry must state its setting.
4. **The labels look clean, but the audit is still needed.** Five strong models agree with the
   constructed answers on about 99% of items. Error triage still found two image defects the
   checkers missed:
   - a sticky note spilling onto a second price;
   - translucent tape.

   Both are flagged and in the audit packet.

## Error triage (`data/imajev-bench/v2-lite-v1/adjudications-triage.json`)

- **Likely label errors, from image defects:**
  - `scene-03-0054-cover_other-q0`: the note also covers the Toastie price, which the order needs.
  - `scene-03-0089-cover-q0`: the tape is see-through, and 29 g is readable.
- **Judgement calls:** 6 boundary items where a stay ends exactly when a no-parking period starts,
  plus 12 nutrition panels made internally inconsistent by the edit.
- **Premise mismatch:** 3 items say "on the shelf" while the mugs stand on a table.
- **Genuine model errors, kept:**
  - models assuming covered sign hours apply;
  - confusing "Soup" with "Soup of the Day";
  - arithmetic slips.
- **Interface issues, disclosed:**
  - Grok abstained on all 15 business-day items, reading "no outside knowledge" as forbidding
    calendar knowledge. v2.1 will state the weekday.
  - Grok ran without a response schema and had 24 format errors or timeouts. Errors are scored as
    wrong.

## Still to do before a public preview

1. **Blind human audit of 138 items:**
   `data/imajev-bench/v2-lite-v1/packets/audit-auditor-1-v2.html`.
2. **Local and open-model runs, plus the no-image baseline:** hardware decision pending.
3. **Publishing details:** licences (proposed: CC0 for the AI images, CC BY 4.0 for the labels), a
   contact address, a README, and the test-split policy.

## Cost so far (estimates from list prices; bills are authoritative)

| Item | Est. cost |
| --- | ---: |
| Image generation, full build | ~$11 |
| Image checking, full build | ~$2.40 |
| Evaluations | ~$10–15 |
| Trials and tests | ~$15 |
| **Total** | **~$40–45** |
