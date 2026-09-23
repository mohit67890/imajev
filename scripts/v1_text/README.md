# decision-v1-text data tools

Every converter is local-only and fail-closed on licensing. `LICENSE` evidence must have an adjacent
`LICENSE.receipt.json` with `evidence_sha256`, SPDX identifier, and `commercial_use_reviewed: true`.
A Hub license tag is not evidence.

Commands:

```
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_typed_decisions SOURCE LICENSE OUTPUT
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_classification SOURCE LICENSE OUTPUT --spdx Apache-2.0 --source-name NAME
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.build_controls TEXT.jsonl IMAGE.jsonl MIXED.jsonl
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.assemble_mixture --text TEXT.jsonl --images IMAGE.jsonl --output MIXED.jsonl
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.filter_synthetic_agreement RAW.jsonl FILTERED.jsonl
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.audit_mixture MIXED.jsonl --report reports/decision-v1-text/audit.json
```

`convert_typed_decisions` accepts the published LocalLLaMA Parquet schema (`state`, `questions`,
`gold`) and normalized JSON/JSONL. It preserves the upstream test split exactly; upstream train is
group-hashed into train/dev/calibration. `convert_classification` covers intent, sentiment, moderation,
and entailment exports that have one text and one categorical label. Use `--heldout` for entire held-out
families such as MASSIVE-test or SST-5.

Source-specific converters now operate directly on acquired, admitted source files:

```sh
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_routing_emotion
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_screening_rubric
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_claims_reading
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_structured
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.convert_mmlu_heldout
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.build_local_mixture
PYTHONPATH=src:scripts .venv/bin/python -m v1_text.audit_token_lengths \
  data/manifests/decision-v1.1-draft.jsonl --report reports/v1.1-datasets/token-length-audit.json
```

The converters write under `data/decision-v1-text/converted/` and retain upstream evaluation
splits, derive calibration only from training groups, and remove duplicate groups with held-out
precedence. The MMLU converter is explicitly evaluation-only. FEVER uses retrieved Wikipedia
evidence, SQuAD uses gold-independent candidate IDs, HelpSteer carries its real ordinal rubric,
and unannotated moderation categories are omitted rather than mislabeled as unknown.

The locally cached typed-decisions repository has a dataset-card `apache-2.0` tag but no `LICENSE` file
at commit `ea9306458d6e9563628369a3d1e72e362fb381d2`. It is therefore intentionally blocked from training
assembly. Existing zero-shot baseline evidence remains in `reports/text-decisions-v1/`.

`assemble_mixture` caps expanded training decisions at 200k text and 300k image by default using deterministic
source/family round-robin sampling; it retains dev, calibration, and test rows without applying a cap.
`build_local_mixture` also constructs grounded ABO photo/listing contradictions, exact-state
cross-fold deduplication, request-style variants, and four kinds of abstention controls. Its image
inputs are restricted to reviewed grants; it does not pad a shortfall with unverified sources.
Measured overlength exclusions are explicit in `reports/v1.1-datasets/overlength-exclusions.json`.
The resulting `decision-v1.1-draft` remains a partial mixture pending the full image/synthetic slices.

`generate_synthetic_local` makes a generation call followed by separate blind label calls and
records raw generations and failures. Its pilot outputs stay outside `converted/` and are not
training-admitted. Human review and stronger quality evidence are required before scaling.
`filter_synthetic_agreement` only filters precomputed labels; it does not itself call a model.
