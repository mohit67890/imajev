# decision-v2: teacher pseudo-labelling and assembly

Spec: `docs/decision-v2-pseudolabel-spec.md`. Everything here is additive — `data/decision-v1*`,
`data/manifests/decision-v1.1.jsonl` and `reports/` are read, never rewritten.

| file | what it does |
| --- | --- |
| `pseudo_label.py` | **label mode**: scores every candidate decision twice (served order + rotated by one), calibrates, averages, applies the spec keep rule, writes labelled rows + `<output>.dropped.jsonl` + `<output>.summary.json`. **blend mode** (`--blend ALPHA`): self-distillation over *already-labelled* rows — rewrites `target_distribution` as `ALPHA * model + (1-ALPHA) * existing target`, drops nothing |
| `assemble_v2.py` | `data/manifests/decision-v2.jsonl` = every v1.1 row unchanged + labelled v2 rows (test rows flagged `heldout_family` / `pseudo_label_test`), audited into `reports/v2-datasets/mixture-audit.json` |
| `make_smoke_candidates.py` | builds the 40-record synthetic candidate fixture under `data/decision-v2-smoke/` (v1.1 questions and ABO photos with their targets stripped) plus `v1.1-train-10.jsonl`, a varied slice of real v1.1 training rows for the blend smoke |

Every command below assumes the repo root as the working directory and:

```bash
export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1
PY=.venv/bin/python           # a pod uses its own interpreter
```

## What the labeller writes

Per kept decision:

- `target_distribution` — the averaged, calibrated distribution keyed by option value exactly as
  `decision_data.render` expects: the option's `str(value)` (`"true"`/`"false"` for booleans,
  `"1"`, `"2"`, … for ordinal levels) plus `"unknown"` for the abstention mass. If a listed option
  would itself be spelled `unknown`/`null`, the reserved `__unknown__` key is used instead.
- `target` — the argmax value with its original Python type (`str` / `bool` / `int`), or `null` when
  the argmax is unknown, in which case `abstention_cause: "teacher_unknown"`.
- `pseudo_label: "teacher-<name>"`, `teacher_confidence` (averaged top probability),
  `teacher_agreement` (distribution overlap between the two orders, `1 - ½·Σ|p_served − p_rotated|`;
  `1.0` = the two presentation orders agree exactly), and `teacher_fields` with the per-field
  detail (per-order argmax, temperature, calibration version, option count).
- Multi-question records keep only the fields that pass and use the
  `targets` / `target_distributions` / `abstention_causes` maps that `decision_data.expand_fields`
  consumes; a record with no passing field is dropped entirely.

Dropped decisions land in `<output>.dropped.jsonl` with `reason` ∈
`order_disagreement`, `low_confidence`, `low_confidence_unknown`, `invalid_candidate` (a candidate
that does not render — counted and skipped rather than killing a multi-hour run) and the full
distribution.

### `--rotate`: which list the second pass rotates

Rotation happens on the *presented* candidate lines, not inside `request.fields[0].options`:
boolean fields have no option list and ordinal levels must stay ascending inside the request.

- `--rotate options` (default, the spec's "options rotated by one position"): the listed options
  move by one and `unknown` stays on the last line, where training and serving always put it
  (`decision_data.render`: "unknown stays last as served").
- `--rotate candidates`: the whole list rotates, so rotating by one always promotes `unknown` to
  the first line. That is a much harsher probe of a different bias, and on the 2B smoke it costs
  most of the yield: keep rate **0.70** vs **0.89**, with 13 of the 14 drops being
  `order_disagreement` where the rotated pass flipped to `unknown` (every text family, all of
  `esci` and `banking77`). Worth running once on the 9B as a diagnostic; not what the dataset
  should be built from.

## `--adapter none`: scoring with the base model

`--adapter none` (or omitting `--adapter`) loads the bundle with **no LoRA and no trained 255-row
readout**, and scores through the engine's legacy lm_head path — `TorchDecision.candidate_logits`
falls back to `hidden @ lm_head.weight[token_ids].T` when `self.readout is None`. `--calibration`
is optional in that mode; without it every bucket uses `T = 1`. The summary records
`readout: "base_lm_head"` vs `"trained"` so a run can never silently be the wrong one.

The decision codebook is pinned to the no-image prompt at engine start (what `enable_readout` does
in adapter mode), so a base run uses the same candidate codes for image and text records.

## Blend mode: self-distillation against the base model

Fine-tuning erases the base model's reasoning (base Qwen3.5-9B through our readout: 64.9% on
JevBench hard; after the fine-tune: 42.3%). Blend mode regularises the training targets toward the
base model's own distributions.

For every field of an **already-labelled** record:

```
target_distribution = ALPHA * model_distribution + (1 - ALPHA) * existing_target
```

- `existing_target` is a one-hot over the listed options **plus unknown** (unknown when
  `target` is `null`), or the record's own soft distribution when it already has one. Both are
  resolved through `decision_data.render`, which is what makes v1.1's older key spellings
  (`True`, `__unknown__`, `null`, stringified ordinal levels) land on the right candidate; the
  output is re-keyed to the canonical `true` / `false` / `"3"` / `unknown` form.
- `target` / `targets` are **not** changed, and neither is anything else on the record —
  `image_role`, `group`, control flags, licences, partitions all pass through byte-identical.
- No keep rule, nothing is dropped: every input record is written out. Rows that are not blended
  (see below) are emitted exactly as they came in, with no `regularizer` fields, and listed in
  `<output>.passthrough.jsonl`.
- Blended rows gain `regularizer: "base-<name>"`, `regularizer_alpha`, and `regularizer_fields`
  (per field: the model's argmax and confidence, the label's argmax, whether the base agreed,
  whether the label was already soft). The summary reports `base_agreement_rate` — how often the
  base model already agrees with the human label — which is the number to watch when picking ALPHA.
- One served order is enough (`--rotations` defaults to 1 in blend mode); `--rotate` still applies
  if you ask for 2.

**Two rows are deliberately not blended.**

- `--blend-partitions` (default `train`): only training rows are regularised. Every other
  partition — dev, calibration, test — passes through untouched, so the evaluation folds keep
  their hard labels. Pass `all` to blend everything.
- `abstention_cause: "not_listed"` rows (41,093 of them in v1.1) pass through unless
  `--blend-not-listed`. The trainer calls `decision_data.with_noise_state` before `render`, which
  for those rows **appends** a "none of the above" option and moves the hard target onto it. A soft
  distribution written now has no key for an option that does not exist yet, so its mass would stay
  on `unknown` and contradict the target the trainer is about to set. Leaving them one-hot is
  correct; blending them is a silent label corruption.

```bash
# self-distil the v1.1 training rows toward the BASE 9B, alpha 0.5
$PY scripts/v2/pseudo_label.py \
  --manifest data/manifests/decision-v1.1.jsonl \
  --model artifacts/model-qwen9b.json --adapter none \
  --blend 0.5 --base-name 9b-base \
  --output data/decision-v2/blended/v1.1-base9b-a05.jsonl \
  --device cuda --token-budget 16000 --max-batch 32
```

Sharded over 4 GPUs exactly as in section 2 (`--shard $i/4`); each shard emits its own passthrough
rows, so concatenating the four shard files reproduces the whole manifest. Without `--limit` the
shard filter is applied while streaming, so a shard never holds all 862 MB of v1.1.

`--blend 0` is a no-op blend: it rewrites the one-hot targets in canonical key form and never loads
a model (useful as a plumbing check).

Then build decision-v2 on top of the blended rows instead of the raw ones:

```bash
$PY scripts/v2/assemble_v2.py \
  --v1 data/decision-v2/blended/v1.1-base9b-a05.jsonl \
  --labelled data/decision-v2/labelled/teacher-9b.jsonl \
  --output data/manifests/decision-v2.jsonl \
  --report reports/v2-datasets/mixture-audit.json
```

The audit report records `v1_regularization` / `v2_regularization`
(`regularizer`, `regularizer_alpha`, `regularized_rows`) on each side of the mixture.

### Measured throughput (for sizing the GPU pass)

Apple M-series MPS, 2B base, `--max-batch 1`, `--rotations 1`, 60 mixed v1.1 training rows
(38 image / 22 text, 84 blended decisions): **1.61 records/s, 2.25 decisions/s** (~445 ms per
decision, one forward pass each). The 10-row smoke gives 1.51 rec/s. Label mode costs twice that
per decision because it scores two rotations.

Work units for a full v1.1 blend pass (counted from `data/manifests/decision-v1.1.jsonl`):

| partition | rows | decisions |
| --- | --- | --- |
| train | 466,135 | 504,000 |
| dev | 24,006 | 30,026 |
| calibration | 23,309 | 25,217 |
| test | 34,528 | 35,528 |

With the defaults (`--blend-partitions train`, `not_listed` passed through: 37,285 rows) that is
**466,715 forward passes**, one per blended decision, 69% of them with an image (322,469 of the
466,135 train rows carry one). The remaining 81,843 rows in the other partitions cost nothing —
they stream straight through. MPS is the only number measured here; size the pod against the
batched evaluator's observed rate on the same hardware rather than extrapolating from 2.25 dec/s.

## 1. MPS smoke (laptop, 2B stand-in, no GPU)

Batched left padding NaNs on Apple MPS, so `--max-batch` is clamped to 1 automatically and the
one-at-a-time path is used (`docs/full-run-runbook.md`). Roughly 2 s per candidate record (two
forward passes) on an M-series laptop; the whole 40-record fixture takes about 70 s plus load.

```bash
# 40 synthetic candidates (20 text-only, 20 ABO image), targets stripped
$PY scripts/v2/make_smoke_candidates.py

$PY scripts/v2/pseudo_label.py \
  --manifest data/decision-v2-smoke/smoke_text/records.jsonl \
             data/decision-v2-smoke/smoke_image/records.jsonl \
  --model artifacts/model.json \
  --adapter reports/decision-v1.1/runs/h100x4/best \
  --calibration reports/decision-v1.1/calibration-v1.1.json \
  --output data/decision-v2-smoke/labelled/smoke-2b.jsonl \
  --device mps --teacher-name 2b-v1.1-smoke
```

The same command with `--limit 4` is the 30-second version. `--manifest` takes any number of
`records.jsonl` files, so real sources are added by listing
`data/decision-v2/<source>/records.jsonl` instead of the smoke ones.

Blend smoke (base 2B, no adapter, no calibration, 10 real v1.1 training rows — ~7 s):

```bash
$PY scripts/v2/pseudo_label.py \
  --manifest data/decision-v2-smoke/v1.1-train-10.jsonl \
  --model artifacts/model.json --adapter none \
  --blend 0.5 --base-name 2b-base \
  --output data/decision-v2-smoke/labelled/blend-2b-base.jsonl \
  --device mps

$PY scripts/v2/assemble_v2.py \
  --v1 data/decision-v2-smoke/labelled/blend-2b-base.jsonl \
  --labelled data/decision-v2-smoke/labelled/smoke-2b.jsonl \
  --output data/decision-v2-smoke/decision-v2-blend-smoke.jsonl \
  --report reports/v2-datasets/smoke-blend-audit.json
```

## 2. CUDA sharded run, 4 GPUs (9B teacher)

One shard per GPU, all four in parallel; `--shard i/n` splits by *candidate record*, so a
multi-question record is never split across shards. With `--shard i/n` and `n > 1` the labeller
writes `<output stem>.shard<i>of<n>.jsonl` (plus `.dropped.jsonl` / `.summary.json` per shard), so
the four processes can share one `--output`.

```bash
mkdir -p data/decision-v2/labelled
MANIFESTS=$(ls data/decision-v2/*/records.jsonl)

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PY scripts/v2/pseudo_label.py \
    --manifest $MANIFESTS \
    --model artifacts/model-qwen9b.json \
    --adapter reports/decision-v1.1-9b/runs/h200x4/best \
    --calibration reports/decision-v1.1-9b/calibration-9b.json \
    --output data/decision-v2/labelled/teacher-9b.jsonl \
    --device cuda --shard $i/4 --token-budget 16000 --max-batch 32 \
    --teacher-name 9b-v1.1 --rotate options \
    > data/decision-v2/labelled/shard$i.log 2>&1 &
done
wait

# one file, and one combined summary of the four shard summaries
cat data/decision-v2/labelled/teacher-9b.shard?of4.jsonl > data/decision-v2/labelled/teacher-9b.jsonl
cat data/decision-v2/labelled/teacher-9b.shard?of4.jsonl.dropped.jsonl \
  > data/decision-v2/labelled/teacher-9b.dropped.jsonl
$PY - <<'EOF'
import json,glob,collections
total=collections.Counter()
for p in sorted(glob.glob('data/decision-v2/labelled/teacher-9b.shard*of4.jsonl.summary.json')):
    s=json.load(open(p))
    for k in ('candidate_records','candidate_decisions','kept_records','dropped_records','kept_decisions','dropped_decisions','unknown_kept'):total[k]+=s[k]
    for k,v in s['dropped_by_reason'].items():total['reason:'+k]+=v
print(json.dumps(dict(total),indent=2))
EOF
```

Notes for the pod run:

- `--token-budget` is the padded-batch budget (longest × count), same meaning as in the trainer and
  the batched evaluator: 16000 on 80 GB cards, 24000 on 141 GB. `--max-batch` is only the cap.
- `--pixels` (default 400000) is the per-image budget; text-rich sources get at least 1 MP through
  `decision_data.pixel_budget`.
- `--limit N` takes an evenly spaced sample of candidate records — use it for a first-minutes check
  (`--limit 64 --shard 0/1`) before committing the whole run.
- Thresholds are the spec's: `--top-threshold 0.6 --unknown-threshold 0.5`. Changing them, or
  `--rotate`, changes the dataset, so record the values in the source README.

## 3. Assembly

```bash
$PY scripts/v2/assemble_v2.py \
  --labelled data/decision-v2/labelled/teacher-9b.jsonl \
  --output data/manifests/decision-v2.jsonl \
  --report reports/v2-datasets/mixture-audit.json
```

`--labelled` accepts several files, so the four shards can be passed directly instead of being
concatenated first. v2 rows whose `partition` is `test` get `heldout_family: true` and
`pseudo_label_test: true`; v1.1 rows are passed through untouched, and an id collision with v1.1 is
a hard error. Nothing is written if the audit fails — the report still is, with the errors.

Smoke build (mini manifest, does not touch `data/manifests/`):

```bash
$PY scripts/v2/assemble_v2.py \
  --labelled data/decision-v2-smoke/labelled/smoke-2b.jsonl \
  --output data/decision-v2-smoke/decision-v2-smoke.jsonl \
  --report reports/v2-datasets/smoke-mixture-audit.json \
  --v1-limit 50
```

`--v1-limit` / `--v1-stride` (and `--v2-limit`) are smoke-only knobs. A v1.1 subset can in principle
strand an `image_role: irrelevant` control whose relevant twin was not sampled, which the audit
reports as `irrelevant-control images never appear as relevant`; raise `--v1-limit` or drop the
limit for anything but a smoke.

## 4. Audit

`assemble_v2.py` already runs `v1_text.audit_mixture.audit` and writes the report; it exits non-zero
and prints the first 20 errors when the audit fails. To re-audit an existing manifest on its own:

```bash
$PY -m v1_text.audit_mixture data/manifests/decision-v2.jsonl \
  --report reports/v2-datasets/mixture-audit.json
# token audit, per the spec (tokenizes with artifacts/model.json; --report is required)
$PY -m v1_text.audit_token_lengths data/manifests/decision-v2.jsonl \
  --report reports/v2-datasets/token-audit.json
```

The audit is fail-closed on: licence receipts (repo-relative evidence paths under
`data/decision-v2/licenses/…` verify through `v1_text.common.verified_license`), duplicate ids,
groups and image hashes leaking across partitions, held-out families outside `test`, missing image
files, and — the part that matters for pseudo-labels — `render()` on every expanded field, which
raises on a `target_distribution` key the candidate list does not offer.

## 5. Tests

```bash
$PY -m pytest tests/test_v2_pipeline.py -q                     # fast, no model
IMAJEV_V2_SMOKE=1 $PY -m pytest tests/test_v2_pipeline.py -q   # + the 2B/MPS end-to-end smoke
```
