# imajev v2.1 — 2B, state-grounded and two-image data, continued from v2 (23 Sept 2026)

Qwen3.5-2B + the v1.1/v2 LoRA (r16/α32, language layers) + 255-code readout, continued from the v2 adapter on
`data/manifests/decision-v2.1.jsonl` (365,092 records; sha256 `2f904eca…`): a 150k replay of v2's training rows (+2,732 irrelevant-image
control twins restored), plus 71,630 new decisions on three sources built for the image-plus-state proposition:
`state_grounded` (45,999 kept; questions whose answer depends on reading the state against the photo, labelled by the calibrated 9B
teacher, 84% kept, 15% unknown), `pairs_natural` (8,469; same-item / attribute-differs on natural photo pairs, teacher-labelled) and
`pairs_grounded` (17,162; reference photo + edited target, labels by construction). New-source dev/test rows whose photos v2 had trained on
were moved to train (899) or dropped (7,743 rows) so that no image hash spans partitions. Recipe as v2: lr 5e-5, one epoch (1,361 steps,
zero crashes), best checkpoint by `mean_accuracy` over the v1.1 dev set and the reasoning dev set. 4× H100 SXM, ~1.3 h including
labelling and evaluation; pod8 cost $95.43 in total for v2 and v2.1.

Two checkpoints are kept: **step 50**, which the selector chose (v1.1 dev 87.0%, reasoning dev 48.0%, mean 0.675) and which saw ~8k of
the 232k training decisions, and **step 1,361**, the end of the epoch (87.0% / 43.75%). The pod panels ran on step 50; step 1,361 was
evaluated on the Mac. Adapters: `runs/h100x4/best-step50`, `runs/h100x4/last-step1361` (MLX copies alongside); temperatures
`calibration-v2.1.json` (fitted on step 50; 0.34–0.76, under-confident raw model as in v2). Per-panel numbers: `eval/v21-comparison.md`.

## Headline

| Panel | 2B v1.1 | 2B v2 | **2B v2.1 step 50** | 2B v2.1 step 1,361 | 9B v1.1 |
|---|---:|---:|---:|---:|---:|
| Two-image pairs probe (60, authored) | 68.3% | 65.0% | **81.7%** (Mac 83.3%) | 100% | 81.7% |
| · which state field is now wrong | 50% | 45% | **70%** | 100% | 70% |
| State-grounding probe (200, authored) | 60.5% | 69.5% | **68.5%** | 73.0% | 72.5% |
| · which_field_conflicts | 85% | 71% | **97%** | 100% | 97% |
| · condition_report | 70% | 88% | **94%** | 91% | 100% |
| · claim_supported | 61% | 52% | **64%** | 70% | 64% |
| · count_matches | 42% | 67% | **61%** | 64% | 67% |
| · instruction_changes_answer | 38% | 41% | **41%** | 38% | 35% |
| · label_text_matches | 67% | 100% | **55%** | 76% | 73% |
| v1 image exam, held-out sources (4,989) | 50.7% | 54.4% | **54.6%** | 53.6% | 55.9% |
| · heldout_pairs (177 real two-image cases) | 37.9% | 54.2% | **50.3%** | 51.4% | 26.6% |
| v1.1 test (35,528) | 87.3% | 86.1% | **86.3%** | – | 89.0% |
| MMLU in test (1,000) | 53.5% | 43.0% | **45.7%** | 45.2% (41.8% with an irrelevant image) | 73.7% |
| MMLU predicted-unknown rate | 6.4% | 29.3% | **24.6%** | 24.9% | 3.9% |
| typed-decisions (2,000) | 58.1% | 59.1% | **59.7%** | – | 66.2% |
| SST-5 (2,210) | 47.5% | 44.2% | **45.6%** | – | 51.0% |
| Irrelevance panel (2,823) | 64.3% | 55.5% | **56.7%** | – | 79.6% |
| Reasoning dev, authored 240 (pod) | 46.7% (Mac) | 42.1% | **44.6%** | – | 63.7% (Mac) |
| JevBench hard 111, raw | 43.2% | 44.1% | – | 43.2% | 42.3% |
| JevBench original 72, raw | 88.9% | 87.5% | – | 84.7% | 98.6% |
| New-source test rows, teacher agreement (v2's 41,533) | – | 96.4% | **96.2%** | – | – |
| v2.1-only test rows (state 1,837 / pairs_natural 327 / pairs_grounded 14) | – | – | **91.9 / 93.9 / 78.6%** | – | – |

## Reading

- **The two-image data worked, and quickly.** Fifty steps took the pairs probe from 65% to 82%, level with the 9B teacher, with the
  hardest family (which state field the target now contradicts) going 45% to 70%. The full epoch reaches 100% on that probe and 73% on the
  state probe, above the teacher. Caveat: both probes were authored from the same template families as the training data, so they measure
  how well the model absorbed the relation, not how it generalises. The independent two-image check is `heldout_pairs` (177 real cases from
  v1's held-out sources): 50.3% at step 50 against v2's 54.2%, i.e. no gain yet on natural pairs outside the templates.
- **The state probe did not move overall at step 50, but its shape changed.** Which-field-conflicts recovered from 71% to 97% and condition
  reports rose to 94%, while label-text-matches fell from 100% to 55% on 33 items. Instruction-changes-answer stays at ~40% for every model,
  including the 9B teacher (35%), whose labels for that family are therefore near-random; that family needs constructed labels, not a teacher.
- **Everything else is v2 within a point.** Held-out image sources 54.6%, v1.1 test 86.3%, typed 59.7%. MMLU recovered three points as the
  abstention rate eased from 29% to 25%; it remains eight points under v1.1 for the reason given in the v2 report (abstention prior).
  Reasoning is flat; JevBench hard 43.2% raw on the final checkpoint.
- **The selector rejected the run's own data.** `mean_accuracy` weighs the reasoning dev set equally with the v1.1 dev set, and nothing in
  v2.1's data is about reasoning, so the best-by-selector checkpoint is the one that had barely trained. For a data drop that targets a
  specific capability, select on a probe of that capability plus a regression guard, not on an unrelated dev set.

## Release view

For the image proposition (state against photo, reference against target), v2.1 step 1,361 is the strongest 2B we have: it is at or above
the 9B teacher on both probes and equals v2 elsewhere, at the cost of the same abstention prior v2 carries on knowledge questions. The
independent checks on step 1,361 came back flat against v2 and step 50: held-out image sources 53.6% (v2 54.4, step 50 54.6), real
two-image cases 51.4% (54.2 / 50.3), MMLU text-only 45.2% (42.8 / 45.9) and 41.8% with an irrelevant image (38.0 / 37.8), so the epoch
cost nothing outside the template families and bought the full absorption of the new relation. **Step 1,361 is the v2.1 release
candidate for the image proposition.** What remains before a release is a per-type unknown offset fitted on the calibration fold to bring
MMLU-style questions back toward v1.1 behaviour without touching image decisions, and a calibration that does not over-sharpen
off-distribution items.

## Files

- `eval/tuned-2b-v21/`, `eval/calibration-2b-v21/`: pod predictions on the decision-v2.1 test partition (79,239 rows) and calibration fold.
- `panels/{v1exam,typed,sst5,irrelevance,reasoning,state-probe,pairs-probe}-tuned/`: pod panels on step 50.
- `jevbench-raw-last/`: JevBench public splits on step 1,361 (local MLX serving, uncalibrated).
- `../v2-datasets/{state,pairs}-probe-2b-v2.1-{best,last}.json`, `…-9b-v1.1.json`: Mac probe runs.
- `../decision-v1-test/eval/v21-last-heldout/`, `../decision-v1.1-irrelevance-test/eval/v21-last-mmlu/`: Mac evaluations of step 1,361.
- `labelled/teacher-9b-v21.jsonl`, `mixture-audit-v2.1.json`, `runs/h100x4/run.combined.log`.
