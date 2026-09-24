# Using `caiovicentino1/eikos-decisions` in imajev

Status 2026-09-24. Script: `scripts/p2/filter_eikos.py`. Outputs: `data/external/eikos-decisions/`. Upstream snapshot is pinned to
revision `26d9a680efdf5c05a0aa12d97231b0cc70898392` and checked against the dataset's own `MANIFEST.sha256` before anything is read.

```
.venv/bin/python scripts/p2/filter_eikos.py                               # strict (default) -> data/external/eikos-decisions/
.venv/bin/python scripts/p2/filter_eikos.py --teacher-policy writer-only  # what-if -> data/external/eikos-decisions-writer-only/
```

Re-running gives byte-identical `*.jsonl` files. Only `report.json` and the licence receipts carry a date.

## What survives (default `strict` policy)

| Step (in order) | core/train | core/validation |
|---|---:|---:|
| rows in | 21,150 | 1,221 |
| 1. held out by the authors (Spanish, or empty `used_in`) | 21,150 (0 dropped) | 1,221 (0 dropped) |
| 2. licence on `COMMERCIAL_ALLOWLIST` | 19,239 (−1,911 FinEntity, ODC-BY-1.0) | 1,138 (−83) |
| 3. teacher provenance (no GLM-5.3-Flash anywhere in the row) | 11,254 (−7,985) | 464 (−674) |
| 4. JevBench public 8-gram lint (≥ 2 shared 8-grams) | 10,152 (−1,102) | 418 (−46) |
| 5. exact state+question overlap with `heldout` | 10,152 (0) | 418 (0) |
| **kept** | **10,152** | **418** |

`long_context` (1,517 + 59 rows) and `views_pt_en` (1,619 rows) keep **0 rows** under `strict`. Every one of those rows carries
GLM-5.3-Flash labels, or embeds text GLM wrote.

In `records.jsonl`, the 10,570 surviving rows are converted to imajev's decision-record schema: 10,152 `train` and 418 `dev`.
By field type that is 6,578 boolean, 3,249 choice and 743 ordinal. Every record has `target_probs`.

Every survivor is a **programmatic** item: the Eikos code wrote the state and computed the answer exactly. Kept families:
- `judge`: 2,082 train rows. Of these, 1,145 are GSM8K-train math (MIT) and 937 are TAT-QA tables (CC-BY-4.0).
- `rules_book`: 1,899.
- `trade_exact`: 1,439.
- `rules_permit`: 1,045 of 1,571.
- `finance_exact`: 960.
- `rules_route`: 781.
- `rules_tier`: 749.
- `rules_score`: 717.
- `temporal_numeric`: 480 of 1,020. The programmatic half survives.

By language: 8,866 English and 1,286 Brazilian Portuguese in train. `report.json` has the per-family, teacher, language, licence,
label-source and question-type tables, for both input and kept rows.

The whole slice is rule application, exact finance and trade arithmetic, and answer verification. The heavy lint loss explains
the gaps. There is no probability family: all 576 `prog_prob` rows fail the JevBench lint. There are also no traps, routing,
extraction or policy items, because those are all GLM-labelled.

### What-if counts (no files written unless you run that policy)

| Policy | core/train | core/validation | long_context | views_pt_en | Meaning |
|---|---:|---:|---:|---:|---|
| `strict` (default) | 10,152 | 418 | 0 | 0 | Nothing that GLM-5.3-Flash wrote, labelled or selected |
| `writer-only` | 16,769 | 956 | 0 | 1,052 | Also keeps Qwen3.8-27B-written items. GLM's `teacher_probs` and `target_probs` are discarded, and the target becomes the Qwen writer's gold (hard), or the writer's exact distribution for `writer_exact_probs` rows |
| `open-weights` | 18,120 | 1,092 | 1,423 + 57 | 1,619 | Treats GLM-5.3-Flash as an allowed open model |

**Decision for the owner.** The phase-2c rule (`docs/phase-2c-plan.md`: "Eikos GLM rows excluded") treats GLM-5.3-Flash as an
API teacher, so `strict` is the default. Two facts cut the other way:
- `zai-org/GLM-5.3-Flash` is a public **MIT open-weight** model on the Hub.
- The card says the authors "run [it] on their own infrastructure".

If that is accepted, `open-weights` is admissible. `writer-only` is a middle ground. The labels come from Qwen, but GLM still
**selected** which items survived: the card says only items where GLM's argmax matched the writer's gold were kept. Do not treat
it as GLM-free.

## Per-row provenance fields

These are the upstream fields, as verified on the actual files. They differ from the brief in places:

| Field | Values | How we use it |
|---|---|---|
| `item_writer` | `program` 11,254 / `Qwen3.8-27B` 6,627 / `FinEntity (human annotators)` 1,911 / `GLM-5.3-Flash` 1,358 (core/train) | Who wrote state, question, options, `expected`, `rationale` |
| `label_source` | `program`, `exact_probability`, `teacher`, `writer_exact_probs`, `human` | Origin of `target_probs` |
| `teacher_probs` | non-null on **every** generated row (Qwen- and GLM-written alike); null on program/human rows | Always GLM-5.3-Flash's blind distribution |
| `upstream` | null, `GSM8K train (MIT)`, `TAT-QA (CC BY 4.0)`, `FinEntity (ODC-BY 1.0)` | Per-row licence (null = dataset CC-BY-4.0) |
| `used_in` | `[eikos-4b, eikos-27b]` on train/validation, `[]` on heldout | Authors' hold-out marker |
| `source` | `generated`, `prog_*:<generator>`, `real_finentity`, `long:*` | Kept as `template_id` |

**Correction to the brief.** The card does not say that Qwen was the teacher for most rows. **Every** `label_source=teacher` row
(9,602 in core) has GLM-5.3-Flash as its teacher. Qwen3.8-27B is the *writer* of most generated items. Qwen also did the
PT↔EN views translations. So in `strict`, Qwen-written rows are dropped along with GLM-written ones.

**Other provenance facts:**
- 75 early generated rows have no recorded writer. They are filed under GLM-5.3-Flash and are indistinguishable from the rest.
  Under `writer-only` this could let a few GLM-written rows pass as Qwen. Under `strict` it does not matter.
- The GSM8K judge rows contain candidate solutions sampled from Qwen3.5-0.8B (Apache-2.0). Their labels are exact grading against
  GSM8K answers. `provenance.models` records this model.

Each kept row in `train.jsonl` and `validation.jsonl` is the upstream row unchanged, plus an `_imajev` object with these fields:
`teacher`, `target_from`, `models`, `licence`, `teacher_stripped`.

## Licences and attribution

The per-row SPDX comes from `upstream`, checked against `COMMERCIAL_ALLOWLIST` in `scripts/v1_text/common.py`:
- The dataset's own rows are CC-BY-4.0. Kept.
- TAT-QA rows are CC-BY-4.0. Kept.
- GSM8K rows are MIT. Kept.
- FinEntity rows are **ODC-BY-1.0**. Dropped: 1,911 train and 83 validation rows. ODC-BY permits commercial use with attribution,
  so it is a candidate for the allow-list if the owner wants the human-labelled entity-sentiment rows. It is not on the list today.

`data/external/eikos-decisions/licenses/` holds the evidence and receipt files that `verified_license()` checks. Each evidence
file quotes the card's licence section. The receipts were written by the script, following the `assemble_p2.py` convention. The
owner should confirm them.

Attribution text (carried in every record's `license.attribution`; reproduce it in model cards and release notes):

> Contains data from "Eikos Decisions" by Caio Vicentino (Hugging Face: caiovicentino1),
> https://huggingface.co/datasets/caiovicentino1/eikos-decisions (revision 26d9a680efdf), licensed CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Rows were filtered and converted by the imajev project; changes: rows removed,
> fields renamed into imajev's record schema.

Also credit the upstream datasets for the kept judge rows:
- Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K train, MIT).
- Zhu et al. (2021), *TAT-QA* (CC BY 4.0).

## Record conversion (`records.jsonl`)

The conversion follows `data/decision-p2/teacher/records-p2b.jsonl`. The mapping is mechanical, with no judgement calls:
- `noul` → `boolean`. Options are always `yes`, `no`. `target` = `expected == "yes"`.
- `choice` → `choice`. Option `label` → `value` (already snake_case and unique), and `description` carries over.
- `score` → `ordinal`. Labels are integer strings, which become `levels[].value`.
- `id` = `eikos_decisions:<config>:<upstream id>`. `source_group` = the upstream id (a view or dossier maps to its original item).
- `family` = `eikos.<family>`, prefixed so it cannot collide with p2 family names. `domain` = `topic`.
- `partition`: upstream `train` → `train`, `validation` → `dev`. Upstream validation was never trained on, so it stays dev.
- `target_probs` is kept verbatim, aligned to the listed options or levels. `target_probs_keys` gives the order, which is
  `[true, false]` for boolean. There is **no `unknown` mass**. If the trainer's `--soft-targets` appends an `unknown` candidate, it
  must pad with 0. Upstream targets are label-smoothed: 0.95 on the answer for program rows. Six rows sum to 1 ± 1e-5.
- `rationale` (program template explanations, often empty) is kept as a top-level field for `--rationale-weight`.
- `provenance` holds the dataset, revision, writer, label source, teacher, models, upstream, language, difficulty and `used_in`.

## Heldout (`heldout.jsonl`)

The upstream `core/heldout` split is copied byte for byte: 1,191 rows. Evaluation only: no output of this script uses it for
training, and train/validation rows that exactly repeat a heldout state+question would be dropped (there are none). It contains:
- the entire `tradeoff` family (326 rows);
- the *healthcare administration* topic;
- all 489 Spanish rows (plus 557 English and 145 Brazilian Portuguese).

**Caveats for using it as an eval:**
- Its labels are GLM-5.3-Flash teacher distributions. Every row is `label_source=teacher` or `writer_exact_probs`, and every
  row has `teacher_probs`. That is acceptable for evaluation but is not a human gold.
- One row, `6aad26ef78f8b116` (Spanish `trap`), shares 7 8-grams with JevBench public: force-majeure boilerplate. Exclude it
  from any number reported next to JevBench.

## Caveats

- **The JevBench lint removes templated families.** All 576 `prog_prob` rows fail on the instruction "Give probabilities that
  reflect the evidence in the state", which appears verbatim in 6 JevBench **hard** items. The 526 dropped `rules_permit` rows
  fail on "only if ALL of the following hold: (a) the", which JevBench hard also uses. These are template-level overlaps: the
  Eikos probability family mirrors JevBench's hard probability items in both phrasing and structure. The card reports 0 flags
  because it uses a 0.15 overlap *fraction*, not our absolute count of 2 or more shared 8-grams. Keep the drop. If the
  probability skill is wanted, generate it with our own wording (`scripts/p2/gen_programmatic.py`).
- **Narrow skill coverage under `strict`.** The slice is rules, trade, finance and judge. It will lift exact-answer and
  verification skills, not traps, routing or extraction.
- **Eikos evaluation suites.** The kept rows come from the same generators as Eikos's own suites (`eval/suite_rules.jsonl`,
  `eval/suite_trade.jsonl`, different seeds). Our scores on those suites would be in-distribution. Do not report them as
  zero-shot. Its three held-out trade families (Incoterms, LC presentation, EU VAT) are absent from training upstream, so they
  are absent here too.
- **Label smoothing.** Program targets carry the authors' smoothing: 0.95 on the answer, with the rest spread. Rules-book and
  permit rows are near-hard labels, not calibrated uncertainty.
- **Rows the authors removed are gone.** About 2% of their training rows were dropped for PII-shaped values, credential-shaped
  strings, one LegalBench boilerplate and 3 label bugs, so this is not their exact training set.
- **Portuguese.** 1,286 pt-BR program rows survive. Our benchmarks are English, so weight them accordingly.
