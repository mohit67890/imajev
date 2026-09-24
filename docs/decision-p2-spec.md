# decision-p2: hard typed-decision data for the phase-2 delta fine-tune (spec, 23 Sept 2026)

## Why

Our adapters lose the base models' multi-step reasoning (base Qwen3.5-9B 64.9% on JevBench hard → 42.3% after v1.1; 2B 45.9 → 43.2)
because the v1/v2 mixtures are hundreds of thousands of short, pattern-style decisions. jevk5 (Qwen3.5-4B + LoRA r16, attention only,
two epochs, lr 3e-5) reaches 73.9% hard with **6,544 items**: 3,272 hard typed questions written by Qwen3.6-27B and kept only when the
teacher answered them the same way twice, plus 3,272 human-labelled items. The data, not the size, decided it. decision-p2 reproduces
that recipe with a wider and cleaner set, so that a short second phase can be applied to every imajev size (2B v2.1, 4B v2.1, 9B v1.1)
without touching the image data. Target: ≥ 12,000 kept teacher questions from ~6,000 documents, plus the licensed human-labelled half
(built separately from sources we already hold licences for).

## What the teacher half contains

- **Writer:** `Qwen/Qwen3.6-27B` (revision `6a9e13bd6fc8…`, Apache-2.0), thinking on, temperature 0.9, JSON mode. One call writes a
  realistic business document (200–900 words; ticket, log, policy, contract excerpt, diff, invoice, schedule, transcript, spec,
  report) and **three** hard typed questions with the intended answer and a one-line justification. Questions are Jev-typed: `noul`
  (boolean), `choice` (3–12 options, keys rebuilt from option text), `score` (ordinal, 2–10 integer levels with a legend).
- **Families (15):** jevk5's eleven (policy_exception, date_number_trap, multi_step_lookup, judge_answer, ambiguity, misleading_note,
  injected_instruction, rule_precedence, routing, extraction, rubric) plus numerical_reconciliation, temporal_ordering,
  insufficient_evidence and contradiction. Each has a hardness rule and ≥ 6 paraphrased briefs (`scripts/p2/families.py`).
  Three distinct families per document. Unknown-heavy families (ambiguity, insufficient_evidence, contradiction) are sampled so that
  15–25% of questions are meant to be `unknown`, with the abstention cause recorded (insufficient_evidence, false_premise, not_listed,
  mismatched_reference).
- **Domains (24):** `scripts/p2/domains.py` (e-commerce listings, logistics, insurance claims, HR policy, IT support, finance ops,
  healthcare admin, travel, contracts, education, real estate, hospitality, telecom, energy, government forms, manufacturing QA, retail
  returns, SaaS billing, security incidents, procurement, events, fleet, agriculture supply, nonprofit grants), round-robin over
  documents, each with 3 document kinds.
- **State shape:** 50% plain string, 50% JSON object (the document as an application would store it), fixed per document by the plan.
- **Answerers (two model families, independent, never shown the intended answer):** the writer model with a fresh sample, and
  `openai/gpt-oss-20b` (revision `6cee5e81ee83…`, Apache-2.0) at reasoning effort medium. Temperature 0, JSON `{"answer", "confidence"}`,
  `unknown` always offered.
- **Keep rule:** a question is kept only when **both** answerers' parsed values equal the writer's intended value (unknown equals
  unknown). All three answers and confidences are stored in `provenance`.
- **Contamination lint:** word 8-grams of every document and question against the JevBench public files
  (`.cache/external/jevbench/datasets/public/{easy,hard,original}.jsonl`); any record sharing ≥ 2 8-grams is dropped and counted.
  No JevBench item, no output of Jev and no paid API is used anywhere in the pipeline.
- **Partition:** by document with `scripts/v1_text/common.py:stable_partition(doc_id, seed="decision-p2")` (dev 5%, test 10%); the
  three questions of a document never split.
- **Licence:** teacher outputs released under Apache-2.0; `assemble_p2.py` writes `data/decision-p2/licenses/p2_teacher/
  TEACHER-OUTPUTS-Apache-2.0.md` + receipt (model ids, revisions, review note) and attaches the verified licence object to every record.
- **Record schema:** decision-v1 schema (`docs/decision-v2-pseudolabel-spec.md`), `source: "p2_teacher"`, `source_split: "teacher"`,
  `pseudo_label: "p2-teacher-agreed"`, `template_id: "p2.<family>.v1"`, `unknown_by_construction`, `state_variant`, `domain`,
  `provenance {writer, answerers, agreement, justification}`. Validates with `vision_decision.contracts.Request`.
- **Optional trainer fields (v1.1 recipe flags, `docs/full-run-runbook.md`):**
  - `target_probs`: `{option value: probability}` for the record's single field. Keys follow the same aliases as
    `target_distribution`: `true`/`false` for booleans, the integer as a string for ordinal levels, and `unknown`, `null` or
    `__unknown__` for unknown. Values are non-negative and are renormalised. Missing options count as 0, and unrecognised keys
    are an error. It is used only with `--soft-targets`. Gold for dev accuracy is `target` when present, else the argmax of
    `target_probs`. If a no-match augmentation relabels a row, the row falls back to its hard target. Multi-field records should
    keep per-field `target_distributions` instead.
  - `rationale`: a short plain-text reason (for example the writer's one-line justification). It is used only with
    `--rationale-weight`, which appends it as `" Because: <rationale>"` after the decision position, capped at
    `--rationale-max-tokens`. It is never shown at inference.

## What is excluded

JevBench items and paraphrases of them; Jev outputs; paid APIs; real companies, brands or people (prohibited in the writer prompt);
documents outside 120–1,600 words or above the 32 KB state limit; questions whose family/type/unknown flag differs from the plan;
answers that fail to parse; any question one answerer gets wrong.

## Pipeline (`scripts/p2/`)

| Step | Script | Notes |
|---|---|---|
| 1 | `gen_write.py --docs N --out writer.jsonl --workers 32` | plans are deterministic by index; resumable by doc id; rejects logged with reason and retried once |
| 2 | `gen_answer.py --answerer qwen\|gptoss --writer writer.jsonl --out answers-<a>.jsonl` | resumable by (doc, question); raw + parsed |
| 3 | `assemble_p2.py --writer … --answers … --out data/decision-p2/teacher/records.jsonl` | agreement filter, licence, partition, lint, `assembly-report.json`, README |
| pod | `pod_run_p2_gen.sh` | one H100: vLLM serves the writer (write + qwen answers), then gpt-oss-20b (answers), then assemble; DONE markers |

Tests: `tests/test_p2_generation.py` (fake clients; prompt rendering, schema validation, option keys, answer parsing, retries, agreement,
unknown share, partition, licence receipt, contamination lint, request validity; phase 2c: judge prompts and the judge CLI path,
programmatic schema/ids/determinism, independent answer recomputation for every kind, zero JevBench 8-gram overlap, assembly).

## Sizes, time and cost (one H100 SXM, vLLM, ~2k tokens/s aggregate)

| Stage | Tokens | Time |
|---|---:|---:|
| write 6,000 documents (~1.5k thinking + ~1.3k output each, ~10% rejects retried) | ~19M | ~2.6 h |
| answer 18,000 questions with the writer model (~0.7k each incl. thinking) | ~13M | ~1.8 h |
| answer 18,000 questions with gpt-oss-20b (medium reasoning, ~0.5k each) | ~9M | ~1.3 h |
| assemble | – | minutes |

About 5.5–6 h and $20 at $3.49/h. Expected yield after agreement and lint: 60–70% of 18,000 → 11,000–12,500 kept questions
(jevk5 kept ~50% with a single-teacher double answer). If the yield falls short of 12,000, rerun `gen_write.py --start 6000 --docs 2000`.

## Phase 2c: judge families and exact-answer generators (24 Sept 2026)

**Why.** After phase 2b our adapters still trail JevK5/Hopper on two JevBench tiers: `judge_hard` (grading whether, or how well,
a response satisfies a request under stated criteria: method, edge cases, units, format, retained facts) and `temporal_numeric`
/ `probability` (exact date, time-zone, duration and arithmetic answers). The phase-2 writer families touch both only indirectly
(`judge_answer`, `rubric`, `date_number_trap`), and LLM-written numeric items are the ones the two answerers most often disagree
on, so few survive the keep rule. Phase 2c adds (a) two writer families that make the judge setting the whole point of a question,
and (b) Eikos-style exact-answer families whose answers are computed by code, so correctness does not depend on the writer and the
answerer agreement filter only removes items the teacher models cannot solve. We studied the STRUCTURE of the public JevBench items
(state shapes, question types, option types, 2-10 score levels); no item text is reused, and every rule, brief and generated
document is checked against the JevBench 8-gram lint (tests assert zero shared 8-grams).

**Judge writer families** (`scripts/p2/families.py`, opt-in: reached only with `gen_write.py --families`, so default phase-2 plans
stay byte-identical and `FAMILY_IDS` keeps its 17 members; `ALL_FAMILY_IDS` lists all 19):

| Family | Type | Unknown rate | What the document holds / what decides it |
|---|---|---:|---|
| `judge_pairwise` | choice: Response A / Response B / Both equally or Neither is acceptable | 15% | a request, a rubric or policy with (usually) a ranking of criteria, and two plausible responses; the better one wins on the highest-ranked separating criterion (a unit, a dropped requirement, a stale fact, a format gate, a policy breach) while the other looks more polished; unknown when the rubric cannot separate them |
| `judge_rubric_score` | score, 2-9 levels copied from the rubric | 5% | a request, a response and a rubric with concrete level conditions (caps, deductions, cumulative checks); impression-based grading lands on an adjacent level |

Both families carry 8-9 briefs and a family-specific justification requirement ("name the rubric criterion that decides it and
the concrete difference; the answerers are checked against it"). The intended answer and justification fields are the existing
writer schema, so `gen_answer.py` / `assemble_p2.py` are unchanged. Run:
`gen_write.py --docs N --families judge_pairwise,judge_rubric_score --tag p2c-judge --out writer-p2c-judge.jsonl` (the third question
of each document is drawn from the default families).

**Programmatic exact-answer families** (`scripts/p2/gen_programmatic.py`, no model in the loop, seeded and deterministic):

| Family | Batch tag | Kinds (templates = kind x type x wording) |
|---|---|---|
| `temporal_arithmetic` | `prog-temporal` | business-day deadlines with closure days, calendar roll-over, time-zone ordering and elapsed time, UTC cut-offs, mixed-unit duration sums, overnight shifts, notice by post with deemed receipt, business-hour SLAs, month-based eligibility, weekday offsets, recurring schedules with moved occurrences, lateness bands (score), grace periods (13 kinds, 53 templates) |
| `probability_exact` | `prog-probability` | single draws, pairs and at-least-one without replacement, 2x2 conditional tables, Bayes with natural frequencies, independent both/any, constrained committee counts, ordered slots, expected cost, random assignment, clean-sample thresholds (noul), probability bands (score, 4-10 levels), binomial exactly-one (14 kinds, 50 templates) |
| `numeric_reconciliation` | `prog-numeric` | invoice totals (discount, tax, untaxed delivery), PO-vs-invoice line and amount, bank reconciliation, stacked discounts and voucher order, lb/kg manifests, proration, net from tax-inclusive price, budgets with pending items and transfers, FX with card fees, stock roll-forward variance, change bands (score), usage-based cost split, free-delivery thresholds (13 kinds, 50 templates) |

Each document is a realistic artifact in one of the 24 domains (domain vocabulary, fictional organisations and people, one of
six currencies, string or JSON-object state) with three questions of three different kinds of one family. Options are the correct
value plus the typical mistakes (calendar instead of business days, local time read as UTC, with instead of without replacement,
P(B|A) for P(A|B), tax on the undiscounted subtotal, additive discounts, the wrong allocation key ...), deduplicated by text,
option key and canonical value so exactly one option is right. About 3% of questions withhold one needed input ("pending",
"to be confirmed") and are intended `unknown` (insufficient_evidence). `plan.params[i]` stores the inputs, the canonical answer and
every option's canonical value; `tests/test_p2_generation.py` recomputes every kind independently (numpy business days, tz-aware
datetimes, brute-force enumeration of draws/assignments/panels, Decimal money) and checks unique ids, per-seed determinism and zero
JevBench 8-gram overlap. The generator itself regenerates any document that shares a single 8-gram with the public files.

    python scripts/p2/gen_programmatic.py --docs 3000 --seed prog-v1 --out data/decision-p2/teacher/writer-prog.jsonl
    python scripts/p2/gen_answer.py --answerer qwen35 --writer data/decision-p2/teacher/writer-prog.jsonl --out .../answers-prog-qwen35.jsonl   # (and a second answerer)
    python scripts/p2/assemble_p2.py --writer .../writer-prog.jsonl --answers ... --out .../records-prog.jsonl \
        --batch-filter 'prog-*' --dedupe-threshold 100000 --min-unknown-share 0

Templated documents share boilerplate by construction, so the within-batch near-duplicate lint (built for paraphrasing LLM
writers) would drop almost all of them: disable it with a high `--dedupe-threshold` for `prog-*` batches; diversity comes from the
template, entity and number randomisation instead. The keep rule is unchanged (both answerers must match the computed answer), so
records still say `p2-teacher-agreed`; the programmatic origin is visible in `provenance.writer.model == "programmatic"` and the
`prog-*` batch. The unknown share of these batches is low by design (exact-answer families); mix them with p2b-unknown when
building a manifest.

## What comes next (not in this spec)

The human-labelled half (MMLU-Pro, WANLI, MultiNLI, BoolQ, banking77, ARC, CommonsenseQA; ~3,000 items converted with our existing
`scripts/v1_text` converters and licences), the phase-2 trainer settings (LoRA continuation of each adapter, two epochs, lr 3e-5,
reasoning dev set + grounded dev as guard, every checkpoint kept), and the calibration refit on unseen-domain teacher questions.
