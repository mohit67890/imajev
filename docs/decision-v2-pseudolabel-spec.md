# decision-v2: pseudo-labelled data from new sources (spec, 23 Sept 2026)

Goal: new, never-seen sources for the 2B, labelled by the 9B teacher (`reports/decision-v1.1-9b/runs/h200x4/best`, calibrated).
The pipeline is the v1 pipeline with the teacher replacing human labels. Everything here is additive: `data/decision-v1*` and the
v1.1 manifests are never modified.

## Layout
- `data/decision-v2/<source>/records.jsonl` — candidate records (schema below, `target: null`, `pseudo_label: "pending"`).
- `data/decision-v2/<source>/images/<sha256>.<ext>` — content-addressed, ≤ 1 MP (resize with `scripts/decision_data.load_image` logic), JPEG/PNG only.
- `data/decision-v2/<source>/README.md` — what, how many, licence evidence (URL + quoted grant), exclusions, question templates used.
- `data/decision-v2/licenses/<source>/<file>` + `<file>.receipt.json` — evidence + receipt as `scripts/v1_text/common.py:verified_license`
  requires (`spdx`, `evidence_sha256`, `commercial_use_reviewed: true`, `reviewed_at`, `reviewed_by`, `scope`, `source_url`). Only SPDX
  values in `COMMERCIAL_ALLOWLIST` (Apache-2.0, MIT, BSD, CC0-1.0, CC-BY-2.0/3.0/4.0, CC-BY-SA-3.0/4.0). No licence → source not admitted.
- `data/decision-v2/<source>/validation.json` — output of `scripts/v1/validate_records.py` (must have 0 errors).
- Raw downloads under `data/decision-v2-raw/<source>/` (git-ignored, keep URLs + sha256 in a `manifest.json`).

## Record schema (identical to v1, plus two fields)
```json
{"id": "<source>:<unique>", "source": "<source>", "source_group": "<group that must not split across partitions>",
 "partition": "train|dev|test", "family": "<question family>", "source_split": "<upstream split or 'crawl'>",
 "license": {"spdx": "...", "evidence": "data/decision-v2/licenses/<source>/<file>", "evidence_sha256": "...", "receipt": "...receipt.json"},
 "images": [{"image": "data/decision-v2/<source>/images/<sha>.jpg", "sha256": "<sha>", "width": 0, "height": 0}],
 "request": {"schema_version": "1.0", "request_id": "<unique>", "state": {} | "string",
             "fields": [{"id": "answer", "type": "choice|boolean|ordinal", "question": "...",
                         "options": [{"value": "..."}], "levels": ["..."]}]},
 "target": null, "abstention_cause": null, "pseudo_label": "pending", "template_id": "<which template made this>"}
```
- `partition`: by `source_group` hash with `scripts/v1_text/common.py:stable_partition` (dev 5%, test 10%). Test rows become held-out exam
  rows for the new sources; they are pseudo-labelled too but flagged `pseudo_label_test: true` (never used to claim accuracy against humans).
- Option sets: 2–12 options for choice, drawn from a template bank per family; always plausible distractors; the reserved `unknown`
  is added by the loader, do not list it. 15–25% of questions must be constructed so that the honest answer is `unknown` (object not
  present, attribute not visible, question about a different item than shown, state contradicting the image) — the teacher decides.
- Two-image records (~10% of image records): same object/product from two views, or two different items, with questions
  "same item?", "which image shows X", "does image 2 contradict the listing".
- State: ~40% string, ~60% object; some fields answerable from state alone, some from the image alone, some need both.
- Multi-question requests allowed (2–5 fields) using `targets`/`target_distributions` keyed by field id (see `decision_data.expand_fields`).

## Templates
`scripts/v2/templates/<family>.py`: pure functions `make(record_inputs, rng) -> list[field]`. Keep wording varied (≥ 8 paraphrases per
template), Jev-style (instructions in plain language; criteria as option → short description for ~30% of questions).

## Teacher labelling (`scripts/v2/pseudo_label.py`, GPU)
For every candidate decision: score with the 9B (calibrated) → full distribution over listed options + unknown, keyed by option value.
Score again with options rotated by one position. Keep the decision if argmax agrees across the two orders AND (top probability ≥ 0.6
OR argmax is `unknown` with probability ≥ 0.5). Write `target_distribution` = averaged distribution, `target` = its argmax
(`null` when `unknown`), `pseudo_label: "teacher-9b-v1.1"`, `teacher_confidence`. Drop the rest (keep counts in the source README).

## Audit
`scripts/v1_text/audit_mixture.py` on the assembled manifest (licence objects, partitions, images present, render OK) plus the token
audit. Then `scripts/v1_text/build_local_mixture.py`-style assembly into `data/manifests/decision-v2.jsonl` = v1.1 training rows +
pseudo-labelled rows (soft targets), with the v1.1 dev/calibration/test untouched and the new sources' test rows added as
`heldout_family: true`.

## Training recipe for v2 (added 23 Sept after the forgetting finding)
Fine-tuning on v1.1 erased the base models' multi-step reasoning (base Qwen3.5-9B through the readout: 64.9% on JevBench hard;
after our fine-tune: 42.3%; 2B: 45.9% → 43.2%) while adding format fit and abstention. v2 therefore trains with a regulariser:
1. `scripts/v2/pseudo_label.py --adapter none --blend 0.5` over the v1.1 training rows: every target becomes
   0.5 × base-model distribution + 0.5 × label (one-hot or the existing soft target). The base model is the same size as the student.
2. New-source candidates are labelled by the tuned 9B (calibrated) as above; their targets are the teacher distribution as is.
3. Student: start from the v1.1 adapter (which fixes the LoRA structure at rank 16 on all language projections; `--lora-targets`/`--rank`
   apply only to runs started from scratch); ≤ 0.4 epoch over v1.1-blended + v2 rows, lr 5e-5, 20-step checkpoints; dev-every 100 with TWO dev signals logged: the usual dev loss and accuracy
   on a reasoning-style held-out set (`data/manifests/decision-v2-reasoning-dev.jsonl`, to be built from typed-decisions train +
   JevBench-style multi-hop items we author; never from JevBench public files). Best checkpoint = `--select mean_accuracy` (trainer flags: `--dev2-version decision-v2-reasoning-dev --dev2-cases 400 --select mean_accuracy`).
4. Report: our test, v1 image exam, held-out panels, JevBench public splits, all against 2B v1.1, base 2B, 9B v1.1 and base 9B.

## Image sources, revised 23 Sept
Wikimedia Commons crawling is slow (~150 photos/min via the API) and category-noisy; it was capped at ~12.8k photos. **PD12M**
(Spawning; 12.4M CC0/public-domain image–caption pairs, hosted files, Hugging Face `Spawning/PD12M`) is the preferred bulk image
source from here on: clean licence by construction and captions that let templates build present/absent questions. Captions are
never placed in the state. ~60k images admitted for v2; scale further in v3.
