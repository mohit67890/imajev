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
unknown share, partition, licence receipt, contamination lint, request validity).

## Sizes, time and cost (one H100 SXM, vLLM, ~2k tokens/s aggregate)

| Stage | Tokens | Time |
|---|---:|---:|
| write 6,000 documents (~1.5k thinking + ~1.3k output each, ~10% rejects retried) | ~19M | ~2.6 h |
| answer 18,000 questions with the writer model (~0.7k each incl. thinking) | ~13M | ~1.8 h |
| answer 18,000 questions with gpt-oss-20b (medium reasoning, ~0.5k each) | ~9M | ~1.3 h |
| assemble | – | minutes |

About 5.5–6 h and $20 at $3.49/h. Expected yield after agreement and lint: 60–70% of 18,000 → 11,000–12,500 kept questions
(jevk5 kept ~50% with a single-teacher double answer). If the yield falls short of 12,000, rerun `gen_write.py --start 6000 --docs 2000`.

## What comes next (not in this spec)

The human-labelled half (MMLU-Pro, WANLI, MultiNLI, BoolQ, banking77, ARC, CommonsenseQA; ~3,000 items converted with our existing
`scripts/v1_text` converters and licences), the phase-2 trainer settings (LoRA continuation of each adapter, two epochs, lr 3e-5,
reasoning dev set + grounded dev as guard, every checkpoint kept), and the calibration refit on unseen-domain teacher questions.
