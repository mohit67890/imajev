## v1.1 test set, identical 35,528 decisions

| | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 |
|---|---:|---:|---:|---:|
| Accuracy (abstention credited) | 87.3% | 89.0% | 86.1% | 86.3% |
| Image decisions | 86.1% | 87.1% | 84.7% | 85.0% |
| Text decisions | 88.7% | 91.2% | 87.8% | 87.8% |
| MMLU in test | 53.5% | 73.7% | 43.0% | 45.7% |
| MMLU predicted-unknown rate | 6.4% | 3.9% | 29.3% | 24.6% |
| False abstention on answerable | 1.6% | 1.4% | 3.6% | 3.0% |
| ECE raw (10 bins) | 0.016 | 0.007 | 0.151 | 0.134 |
| ECE v2.1 calibrated | | | | 0.032 |

Paired v2 → v2.1: 370 fixed, 317 broken.

## v2 new-source test rows shared with v2.1 (41,533): v2 96.4% → v2.1 96.2%

## v2.1-only test rows (2,178; state_grounded/pairs_natural = teacher agreement, pairs_grounded = constructed labels)

| Source | n | 2B v2.1 | abstains |
|---|---:|---:|---:|
| pairs_grounded | 14 | 78.6% | 0.0% |
| pairs_natural | 327 | 93.9% | 12.2% |
| state_grounded | 1837 | 91.9% | 13.2% |

## v1 image exam (identical cases)

| Panel | n | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 |
|---|---:|---:|---:|---:|---:|
| held-out sources | 4,989 | 50.7% | 55.9% | 54.4% | 54.6% |
| trained sources | 19,232 | 86.0% | 87.1% | 84.7% | 85.0% |
| · heldout_abstention | 1,500 | 65.2% | 73.3% | 72.0% | 71.5% |
| · heldout_countqa | 1,167 | 35.6% | 39.9% | 32.0% | 34.5% |
| · heldout_fashionpedia | 983 | 47.7% | 58.1% | 51.6% | 52.8% |
| · heldout_livewild | 1,162 | 51.6% | 52.2% | 56.6% | 55.0% |
| · heldout_pairs | 177 | 37.9% | 26.6% | 54.2% | 50.3% |

## Text panels and reasoning dev

| Panel | n | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 |
|---|---:|---:|---:|---:|---:|
| typed | 2000 | 58.1% | 66.2% | 59.1% | 59.7% |
| sst5 | 2210 | 47.5% | 51.0% | 44.2% | 45.6% |
| irrelevance | 2823 | 64.3% | 79.6% | 55.5% | 56.7% |
| · mmlu with irrelevant image | 2000 | 51.8% | 73.0% | 40.4% | 41.9% |
| reasoning dev on pod (2B v2): authored 240 / typed-devsel 6000 | | | | 42.1% / 55.6% |
| reasoning dev on pod (2B v2.1) | | | | | 44.6% / 56.5% |

## Probes

| Probe | 2B v1.1 | 9B v1.1 | 2B v2 | 2B v2.1 step 50 (pod) | 2B v2.1 step 50 (Mac) | 2B v2.1 final step 1361 (Mac) |
|---|---:|---:|---:|---:|---:|---:|
| state | 60.5% | 72.5% | 69.5% | 68.5% | 68.5% | pending |
| · claim_supported | 61% | 64% | 52% | 64% | 64% | pending |
| · condition_report | 70% | 100% | 88% | 94% | 94% | pending |
| · count_matches | 42% | 67% | 67% | 61% | 61% | pending |
| · instruction_changes_answer | 38% | 35% | 41% | 38% | 41% | pending |
| · label_text_matches | 67% | 73% | 100% | 58% | 55% | pending |
| · which_field_conflicts | 85% | 97% | 71% | 97% | 97% | pending |
| pairs | 68.3% | 81.7% | 65.0% | 81.7% | 83.3% | pending |
| · target_still_matches_reference | 75% | 80% | 75% | 80% | 85% | pending |
| · what_changed | 80% | 95% | 75% | 95% | 95% | pending |
| · which_state_field_now_wrong | 50% | 70% | 45% | 70% | 70% | pending |
