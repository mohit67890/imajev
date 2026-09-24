# imajev-4b v2.1 — Qwen3.5-4B, corrected single-run mixture (23 Sept 2026)

Qwen3.5-4B (rev `851bf6e8…`) + LoRA r16/α32 on the language projections (incl. DeltaNet in/out projections) + 255-code readout,
trained **from scratch in one run** on `data/manifests/decision-v2.1-4b.jsonl` (sha256 `1ea2eb63…`; 866,854 training decisions:
504,000 v1.1 rows with their original labels and **no base-model blend**, 296,482 9B-teacher rows with the `unknown` share capped at
15% of the slice, 66,372 v2.1 state-grounded and two-image rows minus the teacher-labelled `instruction_changes_answer` family; 17.2%
unknown targets overall). Half an epoch (2,595 steps), lr 1.5e-4, token budget 10,000 without activation checkpointing (probe: 114 GB
peak of 141), 4× H200 SXM, ~64 examples/s, ~68 min, zero crashes. Selection: `mean_accuracy` over the v1.1 dev set and a **grounded dev
slice** (396 new-source dev rows) → **step 1,900** (v1.1 dev 89.5%, grounded dev 96.7%); the final step 2,595 (88.5 / 96.0) is kept as
`last-step2595`. Pod `vo3jgs1nlchuyx`, $18.36/h; training + evaluation ≈ 2.4 h ≈ $45.

Adapters: `runs/h200x4/best-step1900`, `runs/h200x4/last-step2595`. Pod predictions and panels: `pod/`. Training log: `runs/h200x4/train-log.jsonl`.

## Headline (step 1,900 unless noted)

| Panel | 2B v1.1 | 2B v2.1 (final) | **4B v2.1** | 9B v1.1 |
|---|---:|---:|---:|---:|
| v1.1 test (35,528) | 87.3% | 86.3%* | **87.6%** | 89.0% |
| · image / text | 86.1 / 88.7 | 85.0 / 87.8* | **85.5 / 90.0** | 87.1 / 91.2 |
| · MMLU in test | 53.5% | 45.7%* | **65.9%** | 73.7% |
| · MMLU predicted-unknown | 6.4% | 24.6%* | **8.3%** | 3.9% |
| · false abstention on answerable | 1.6% | 3.0%* | **2.0%** | 1.4% |
| Untrained base, same test | 51.5% | – | 60.1% | 74.7% |
| v1 image exam, held-out sources (4,989) | 50.7% | 53.6% | **56.7%** | 55.9% |
| · heldout_abstention / fashionpedia / livewild / countqa | 65 / 48 / 52 / 36 | – | **75 / 56 / 55 / 38** | 73 / 58 / 52 / 40 |
| · real two-image cases (177) | 37.9% | 51.4% | **41.8%** | 26.6% |
| State-grounding probe (200) | 60.5% | 73.0% | **72.0%** (last 70.0) | 72.5% (teacher) |
| Two-image probe (60, templated) | 68.3% | 100% | **98.3%** | 81.7% |
| New-source test rows, teacher agreement | – | 96.2% | **97.3%** | – |
| typed-decisions (2,000) | 58.1% | 59.7%* | **66.0%** | 66.2% |
| SST-5 (2,210) | 47.5% | 45.6%* | **52.7%** | 51.0% |
| Irrelevance panel (2,823) | 64.3% | 56.7%* | **73.8%** | 79.6% |
| · MMLU with an irrelevant image | 49.8% | 41.9% | **65.2%** | 72.1% |
| Reasoning dev, authored 240 | 46.7% | 42.5% | **62.9%** | 63.7% |
| Reasoning dev, typed-decisions dev-selection (6,000) | – | 56.5%* | **64.9%** | – |

\* 2B v2.1 numbers marked with an asterisk are the step-50 pod panels; unmarked ones are the final checkpoint measured on the Mac.

## Reading

- **The corrected mixture worked.** Dropping the base-model blend and capping unknown targets removed the abstention prior: MMLU
  abstention 8% (2B v2.1: 25%), false abstention 2.0%, and the reasoning dev set at 62.9%, level with the 9B and twenty points above
  the 2B. The forgetting that hit the 2B and 9B did not happen here: reasoning was preserved by the data shape, not by a regulariser.
- **Image proposition at 9B level.** Held-out photo sources 56.7% (9B 55.9%), the state probe at the teacher's level (72.0%), the
  templated two-image probe at 98%. Real two-image cases from v1's held-out sources are the exception at 41.8%, below the 2B v2.1's
  51.4% (177 items; the 4B saw the same pairs data, so this is variance plus the 2B's extra epoch on pairs).
- **Knowledge and text panels sit between the 2B and 9B, closer to the 9B**: MMLU 65.9%, typed-decisions 66.0% (9B 66.2%), SST-5 52.7%
  (above the 9B), irrelevance 73.8%.
- **Cost of a from-scratch single run at 4B:** about $45 of pod time, one hour of training.

## Files

- `pod/eval-tuned`, `pod/eval-last`, `pod/eval-base`, `pod/eval-calibration` (with logits): test partition and calibration fold.
- `pod/{v1exam,typed,sst5,irrelevance,reasoning}-tuned`, `pod/{state,pairs}-probe-{tuned,last}`: panels.
- `mixture-audit.json`: the assembled mixture's audit and counts.
