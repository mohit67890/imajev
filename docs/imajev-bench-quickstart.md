# imajev-bench pilot v0

The initial implementation is a runnable evaluation and annotation harness with synthetic development candidates. It is not a released hidden benchmark. Draft labels come from symbolic generation, not human approval. Consult `generator_manifest.json` for the actual split and composition; the proposal's eventual release sizes are not claims about this pilot.

Run from the repository root using the existing virtual environment. The package uses the existing Pydantic/Pillow dependencies and stdlib; baseline execution loads no model weights.

## Validate and audit

```sh
PYTHONPATH=src .venv/bin/python -m imajev_bench validate --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft
PYTHONPATH=src .venv/bin/python -m imajev_bench audit --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --output reports/imajev-bench-pilot-v0.1/audit-new.json
```

Validation checks strict target types, domain membership, image hashes, path containment, duplicate IDs, shared images/groups across splits, and reviewed-label evidence. The audit reports repeated image content and identical inputs across declared groups. Those are conservative diagnostics: unique images still do not establish independent scenarios or unseen templates.

The CLI refuses draft records unless `--allow-draft` is explicit. An output file or run directory must be new. This avoids overwriting a prior run or mixing predictions from different inputs/settings. Scoring requires a completion receipt and verifies dataset, predictions, raw-response and manifest hashes. Interrupted or modified runs cannot silently produce an attributed score.

## Blind annotation

```sh
PYTHONPATH=src .venv/bin/python -m imajev_bench review --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --output reports/imajev-bench-pilot-v0.1/review-new.html
```

Open the generated HTML in a browser. It embeds the image assets, hides provisional gold and generator metadata, and exports a review JSON. Each judgment needs a reviewer ID, a typed decision and evidence. Work is kept in the tab only; export before closing. Two people should review independently using separate exports. A unique reviewer ID is an audit identifier, not machine verification that reviewers are independent humans.

Import both exports into a new records file:

```sh
PYTHONPATH=src .venv/bin/python -m imajev_bench import-reviews --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --reviews /path/reviewer-a.json /path/reviewer-b.json --output data/imajev-bench/pilot-v0.1/reviewed-records.jsonl
```

Inputs are hash-bound so stale reviews cannot approve a changed request or image. Agreement can correct the generator's provisional gold, which remains recorded in provenance. Disagreements stay draft. For those, a third independent reviewer exports in **Adjudication** mode; import that export against the records file containing the prior two reviews. An adjudicator cannot share either reviewer's ID. Revised judgments after adjudication require an explicit dataset revision rather than silently replacing a review.

## Reproducible control runs

```sh
PYTHONPATH=src .venv/bin/python -m imajev_bench run --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --split dev --adapter first --output reports/imajev-bench-pilot-v0.1/first-new
PYTHONPATH=src .venv/bin/python -m imajev_bench score --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --split dev --predictions reports/imajev-bench-pilot-v0.1/first-new/predictions.jsonl --output reports/imajev-bench-pilot-v0.1/first-new/summary.json
```

Controls are `first` (first substantive option), `random` (uniform sampling including unknown), and `unknown` (always abstain). They are harness checks, not model submissions. Their synthetic probability vectors are explicitly marked. Missing/error outputs count as incorrect; they do not disappear from the denominator. Extra or duplicate prediction IDs are rejected. Missing costs remain missing, not zero.

## imajev HTTP adapter

When an existing imajev playground server is running, replace the control adapter with:

```sh
PYTHONPATH=src .venv/bin/python -m imajev_bench run --records data/imajev-bench/pilot-v0.1/records.jsonl --allow-draft --split dev --adapter imajev-http --endpoint http://127.0.0.1:8765/v1/systemone --output reports/imajev-bench-pilot-v0.1/imajev-new
```

This adapter targets the project's imajev extensions, including `unknown_probability` and `abstained`. It is not a universal adapter for every TypeSafe-compatible server. Boolean probabilities are reconstructed from the local server's documented unknown-mass mapping; ordinal position probabilities are mapped back to the benchmark's actual level values. The substantive decision is argmax, not a fractional expected ordinal score. The abstention flag remains the server's decision. Every image is passed as bytes without its source filename. Raw responses, payload hashes, record hash, settings and elapsed time are logged serially with no retries.

Pin the server's checkpoint, calibration and processor configuration when running real comparisons. The runner records returned model/calibration identifiers but does not establish their identity independently. An endpoint alone is not a pinned model. No API cost is inferred from local execution; hardware costing remains a separate declared measurement.

## What the scores mean

- Capability macro-averages family accuracy within each track; overall capability requires all three tracks. These weights are separate from raw request accuracy.
- Group accuracy requires every request in a declared group to be correct. Exact content reuse across groups must be resolved before using group confidence intervals for a benchmark claim.
- Brier/NLL/ECE report vector validity and coverage, plus family/track breakdowns. ECE is the calibration of the distribution's top label; it can differ from server policy decisions.
- Selective curves rank the probability of the selected substantive answer. Ties enter together. Full AURC is absent unless all answerable items have confidence-ranked answers; partial area must always be read with coverage.
- The bootstrap interval explicitly estimates raw request accuracy by resampling whole groups. It is not an interval on the macro-weighted headline score. The CLI suppresses it when the audit detects shared exact content across declared groups.
- Costs and latency describe this run only. Control-loop timings are not model inference performance.

## Remaining release work

Independent annotation; real-photo/document/screenshot collection with provenance and redistribution permissions; fresh template and source holdouts; calibrated-threshold evaluation on a separate calibration set; permutation/ablation suites; common hardware/concurrency measurements; and independently curated hidden evaluation. Multi-field stress tests and arbitrary third-party VLM adapters are also outside this first implementation.

The benchmark's training/evaluation isolation is deliberate: these modules do not modify model weights or existing training datasets.
