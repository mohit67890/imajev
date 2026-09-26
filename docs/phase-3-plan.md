# Phase 3: failure-driven hard data for imajev-4b (plan, 2026-09-25, revision 3)

> **Revision 3 (2026-09-25).** Consistency fixes, marked **[rev3]**: the image branch is carried through Stages 1, 2 and 2b; out-of-scope
> levers are marked as future; the mixture subsampling rule; the DecisionBench ship rule is stated plainly; per-type temperatures;
> the Goal line matches the rotation fallback. Measured Azure teacher numbers are in the addendum at the end. Revision 2 is kept in
> `docs/phase-3-plan.rev2-2026-09-25.md`. The shared item format is `scripts/p3/candidate.py`.

> **Revision 2 (2026-09-25 evening).** Changes from the review, all marked **[rev2]** below: calibration is fitted on the mined
> hard set; variants inherit their parent's split (no leakage); the held-out target is defined on fresh-seed items; the single-pass
> target keeps rotations as a measured fallback; time, GPU and budget estimates corrected (train on all 8 GPUs, ~$300 ceiling);
> a licence/disclosure section; the DecisionBench subset is tracking-only by construction; a new **image branch** (Stage 0-I / mixture
> 35%) so the hard lane passes the image gate without a soup; a "how to reach +6" section. The previous text is kept in
> `docs/phase-3-plan.v1-2026-09-25.md`.

**Scope: imajev-4b only** (owner, 2026-09-25). The 2B and 9B are out of scope for this phase.

## Everything in this phase (revision 4, owner 2026-09-25: "include everything; I don't want to run this further")

Phase 3 is the **last training run for imajev-4b**. Everything previously marked "deferred" or "future" that needs *training* is pulled
in here. What needs only *serving* code can be added later without another run. Risky changes are **gated inside this run** (pilot
ablations on the pod, then the release gates), so none of them can sink the run: if one fails its gate, the run continues without it.

| # | Item | Status | How |
|---|---|---|---|
| 1 | Hard text data (A generators, B public incl. agent datasets, C own pools, D traps) | **in** | Stages 0–2 as planned |
| 2 | Hard image branch (I) | **in** | mined image failures + photo/record/rule generator + image unknown variants; **plus generated GUI screenshots**: synthetic web/app screens we render ourselves (no third-party rights), with the target element or action as gold. Image items are labelled by **two image teachers**: the Azure 35B (multimodal) and our 9B; keep when they agree or the gold is constructed |
| 3 | Long inputs (4k–16k tokens) | **in** | 5–8% of the mixture: needle family, full TAT-QA reports, MuSiQue with all paragraphs, long agent histories; trainer `--max-length 16384` with length-bucketed batches (serving already allows 32k) |
| 4 | Multi-label ("pick all that apply") | **in, as data + serving** | multi-label families trained as per-label noul questions (no architecture change); the server gets a `multi` question type that fans out to per-label nouls and returns the set + per-label probabilities |
| 5 | 255-option limit | **in, gated** | Unknown is not a fixed slot: each question uses the first N codes for its options and the next code for unknown, so the limit is options + unknown ≤ 255 codes. The readout grows to **256 codes** by appending one code, initialised from its LM-head row exactly as the original 255 were. Questions with ≤ 254 options use the same codes as today, so outputs are identical; only 255-option questions touch the new code. **Gate 0: a bit-exact parity test** (256-code model = shipped model on our eval sets for ≤ 254 options) before any training. Then update every hard-coded 255/254 (MLX + torch backends, server option check, calibration buckets, readout shape check, release files) with tests. The pod pilot and the release gates decide; the fallback keeps 255 |
| 6 | Ordinal loss for scored questions | **in, gated** | `--ordinal-weight` (being built); pilot ablation decides W ∈ {0, 0.15} |
| 7 | Order invariance (single pass) | **in** | `--permute-options`; rot4 remains a serving option |
| 8 | Rationale training (enables a later "think then decide" serving mode) | **in** | `--rationale-weight 0.3 --rationale-max-tokens 192`; the think mode itself is serving-only, added later without retraining |
| 9 | Round 2 on newly failed items | **in (mandatory)** | same pod, labels cached; re-mine the pool with the round-1 checkpoint, one short extra epoch |
| 10 | Per-type temperatures | **in** | choice / noul / score fitted separately on the held-out hard set |
| 11 | rot4 quality mode, two-adapter ensemble | serving only | measured on the pod's eval lane; no training needed |
| 12 | 2B / 9B | **out** | owner scope: 4B only. Cached teacher labels keep a future 9B run possible, but it is not planned |
| 13 | Plumb lessons (field note below) | **in** | `crh225/plumb-decisions` train split (Apache-2.0, open-weight teacher; its 131-item test split stays out) added to source B after decontamination; a **long-document family** (1,500–2,800-word policies, contracts, logs, schedules); mining weighted toward long_policy, judge_hard and temporal_numeric |
| 14 | **Compact prompt layout** (~1,269 → ~700 tokens per decision) | **in, gated** | The model is trained on its prompt layout, so a shorter layout must be trained *in this run* or never. Train on the compact layout; pod pilot lane compares it with the current layout. Keep compact only if accuracy and calibration match or beat the current one; the server switches layout with the adapter |

**Reconciliation of revision 4 with the plumb field note:**
- **Pool size:** text candidates **~120k** (not 330k), plus the ~60k image branch.
- **Two thinking solves only where the gold is not guaranteed** (teacher-written documents, own pools, public items without exact gold), kept on agreement. Items with a constructed gold keep one solve that must match it, which is the stronger check.
  Teacher budget at the measured ~530 solves/$: ~50k flagged items + ~20k second solves ≈ **$130–140** of Azure credit.
- **Pod pilot lanes grow to 5:** baseline / +ordinal / +256 codes / +compact layout / +all winners. ~45 min.


**Pod sequence (one 8×H100 pod):**
1. setup;
2. **pilot ablations**: 4 lanes × 2 GPUs × ~200 steps (baseline / +ordinal / +256 readout / +both), ~40 min;
3. full training with the winners on all 8 GPUs, ~3.5 h (long inputs add ~40%);
4. round-2 re-mine + short epoch, ~1.5 h;
5. eval, gates and calibration, ~1.5 h.

**Money:**
- **RunPod:** ~8–9.5 h → **~$180–230**.
- **Azure credits (teacher):** ~$140, image items included within the 75k cap.

**Mac engineering:** 3.5–4.5 days (up from 2–3): readout-256 change + tests, `multi` server type, GUI-screen renderer, long-context data.


**Goal:** a big step on hard decisions without losing what we have. Mine what imajev-4b gets wrong, have a thinking teacher
solve only those items, verify the answers, and train on them. Aim to serve with a single pass; 4 rotations stay the measured fallback (see Targets). **[rev3]**

**Why this shape:** the benchmarks this week show where we lose.

| Where we lose | Evidence |
|---|---|
| Multi-step reasoning | DecisionBench FinQA 68.0 vs Jev 85.5 / Luna 97.8; FOLIO 61.0 vs 78.5 / 83.0; reasoning family −6.5 pp vs Jev |
| Scored (ordinal) questions | DecisionBench ordinal_scoring 40.3%, −4.8 pp vs Jev, −5.7 pp vs Luna |
| Agent and action choice | browser-target 47, robot-skill 53, strategy 45, platformer 50 (everyone is weak here: a chance to lead) |
| Speed and cost | JevBench speed axis ~79 with `--rotations 4` vs ~87 single pass; rotations quadruple compute |
| Hard tier plateau | JevBench public hard 67.6 → 70.3 across phases 2b/2c; the frozen 35B-A3B scores 61.3 thinking off, **97.3 thinking on** |

**What phase 2c already taught us:**
- The delta lane gained hard accuracy (71.2) but failed the image gate with 15% image replay. The 50/50 adapter soup recovered the images.
  This phase uses **30% image replay** and gates every checkpoint.
- The teacher labels, generators, assembler, soft-target / permutation / rationale trainer flags, gates and pod scripts all exist
  (`scripts/p2/*`, `cloud/pod_run_p2c_*`). This plan reuses them.

## Targets

These are the decision rules for shipping, not predictions.

| Metric | Now (shipped 4B) | Target |
|---|---|---|
| JevBench public hard, single pass + calibration | 69.4 (70.3 with rot4) | **≥ 73** single pass. **[rev2]** rot4 is measured on every candidate too; if single pass misses and rot4 passes, ship rot4 for the board run and document single pass as the fast mode (phase 2c: permutation training left +0.9 on the table for rot4). |
| Own held-out hard set (Stage 4) | measured in Stage 1 | **[rev2]** **+8 pts on the fresh-seed half** (never mined; baseline is the shipped 4B). The flagged half is reported but not a target: its baseline is near zero by construction. |
| DecisionBench 3k stratified dev subset | 77.5 full-suite equivalent | **≥ +2 pts**; ordinal ≥ +5; reasoning ≥ +5. **[rev3]** This is a *ship* rule read by a person at the end. The checkpoint selector never reads it (enforced in the pod script), and it is never used for training or tuning. |
| ImajevBench + probes + unknown + irrelevance gates | phase-2c gates | all PASS (unchanged rules) **without a soup** (the soup stays as the fallback only) **[rev2]** |
| ImajevBench joint track (image + state rules) | 97/122 | **[rev2]** ≥ 99/122 (the families phase 2c lost: threshold_rule, rule_exception, multi_step_rule) |
| Calibration (hard-tier ECE after T fit) | 0.109–0.116 | **[rev3]** One temperature **per answer type** (choice / noul / score), since a single T could not meet both ECE rules in phase 2c (pooled 0.026 → 0.048). **[rev2]** **≤ 0.08**: T is fitted on the Stage-1 held-out hard set (our own data at JevBench-hard difficulty), not on the 150 authored items, which under-correct (phase 2c: fitted 1.72 vs ~2.3 optimal). Pooled public ECE must not rise above 0.03. |

## Stage 0: build the candidate pool (Mac, ~1.5 days, $0)

Aim for about **330k candidate questions with known or checkable answers**. Every source is licence-checked (permissive only:
Apache/MIT/CC-BY/CC-BY-SA; no NC, no paid-API outputs) and decontaminated before it enters the pool.

| Source | ~Questions | Notes |
|---|---:|---|
| **A. Programmatic generators with a difficulty knob** (extend `scripts/p2/families.py` / `gen_programmatic.py`) | 120k | Exact gold by construction. New/extended families: multi-step table and financial arithmetic (2–6 ops), logic chains with negation and distractors (depth 2–6), multi-hop over a synthetic KB (2–5 hops), date/time/unit/currency reasoning, rule application with exceptions (policy → record), ranking/constraint puzzles, **ordinal rubric scoring** (graded evidence → level), **agent action choice** from synthetic state machines (grid, DOM-like page, inventory), long-input needle variants (decisive fact buried in 4–16k tokens). Levels 3–5 of each knob only. |
| **B. Public train splits, converted to our question format** | 120k | Candidates: FinQA, TAT-QA, ConvFinQA (numbers); ProofWriter (depth ≥ 3), FOLIO train, LogiQA (logic); MuSiQue train, HotpotQA hard, StrategyQA (multi-hop); ARC-Challenge train; WebShop, ALFWorld, Mind2Web train (actions). The licence check decides the final list: LogiQA 2.0 and ReClor are non-commercial as far as we know, so they are out unless the check says otherwise. **[rev2]** Round 1 converts only the four simplest (FinQA, TAT-QA, MuSiQue, StrategyQA); agent actions come from generators this round; FOLIO, HotpotQA and ARC are CC-BY-SA (see Licences and disclosure). |
| **C. Our existing pools** | 60k | Phase 2/2b/2c questions and v2.1 text rows. Already labelled; mining finds the ones still failing. |
| **I. Image branch [rev2]** | 60k | Hard image decisions, so the hard lane holds ImajevBench without a soup. (1) Mine the shipped 4B's failures on our own image pools (v1.1 human-labelled, v2 9B-labelled, v2.1 state-grounded and two-photo rows; never ImajevBench). (2) New joint-rule generator: an existing licensed photo + a record + a rule with a threshold / exception / two steps, gold by construction from the record and the photo's known attributes (the v2.1 state-grounded pipeline, extended with the three families ImajevBench showed as weak). (3) Unknown variants: remove the decisive attribute from the record. Teacher for the mined items: the 9B (as in v2) with the 35B verifying the text side; keep only when both agree. |
| **D. Trap and near-miss family** (new generator) | 30k | Built on top of A/B items. (1) A distractor option that almost fits: right entity, wrong number; right rule, wrong exception. (2) Wording that negates the obvious answer ("which order is NOT eligible…"). (3) The decisive fact buried mid-input among plausible but irrelevant facts. (4) Two options that differ only in a qualifier ("within 30 days" vs "within 30 business days"). Gold comes from the parent item, so it stays exact. |

**Decontamination, before mining.**
- Run the 13-gram + item-join checker from `reports/benchmarks/decisionbench/contamination_check.py` against JevBench (public + our
  authored dev), DecisionBench, fast-decisions dev, ImajevBench (public + private-1) and our held-out sets.
- For public datasets that benchmarks draw on (MuSiQue, FinQA, FOLIO, banking77, CLINC…), drop **every upstream item a benchmark row points to**
  (DecisionBench `source_json` ids and text).
- This uses the benchmarks only to *exclude* items, which is standard decontamination. No benchmark row, label or output is ever trained on.

## Stage 1: mine what imajev-4b gets wrong (pod, ~1.5 h on 8×H100) **[rev2: needle items are 10–30× slower per item; cap them at 10k in the mining pass]**

1. Run the shipped 4B once over the whole pool: single pass, raw, about 0.1 s per text item.
   **[rev3]** The pool is ~390k items (330k text + 60k image branch); image items cost 2–5× a text item, and needle items 10–30×.
   Mining runs on the owner's Mac (MLX, one server at a time) overnight, or on the pod at ~1.5–2 h on 8 GPUs if the Mac is too slow.
2. **Flag an item** if any of these holds:
   - the argmax is wrong (sources A/B/C with gold);
   - it is correct but raw p(gold) < 0.6 (**[rev2]** raw probabilities; expect this branch to be small because the raw model is over-confident);
   - option order flips the answer (a 2-order check on a 20% sample).
3. **Expected yield:** 25–35% flagged, i.e. **75–100k items**.
4. **[rev2] Held-out hard set** = two halves of equal size: (a) **fresh-seed items** from every generator family and a random 2% of the unflagged pool, never mined, baseline = the shipped 4B; (b) a 5% random slice of flagged items. Targets are set on (a). We never train or tune on either half, and **T is fitted on (b)** after training (Stage 4).
5. **[rev2] Leakage rule:** variants (Stage 2.5) inherit their parent's split; a parent in the held-out set has no variants in training. At assembly, the 13-gram checker runs held-out against train and fails the build on any hit.

## Stage 2: teacher labels only the hard items (pod, ~2–3 h on 8×H100)

1. **Pilot first:** 1,000 flagged items on 1×H100 (~20 min, < $2). It measures thinking length, throughput and verification pass rate,
   then fixes the full budget.
2. **Teacher:** Qwen3.6-35B-A3B, thinking on, `gen_answer.py --mode distribution`. One copy per GPU, FP8, vLLM batched.
   It returns a full distribution over the options plus `unknown`, and a ≤ 2-sentence rationale.
   Budget: ~80k items × ~2.5k tokens ≈ 200M tokens; at ~5k tokens/s per GPU × 8 that's **~1.5 h**.
3. **Verification.** Only verified items are kept; a wrong label is worse than no label.

   | Source | Keep when | Target used |
   |---|---|---|
   | A (programmatic) | teacher argmax = constructed gold | teacher distribution |
   | A (programmatic) | teacher wrong but gold is certain | hard gold label, capped at 25% of A (teaches beyond the teacher). **[rev2]** Review these first as generator bug reports: the 35B with thinking is at 97% on hard, so a disagreement is more often our generator than the teacher. |
   | B (public, labelled) | teacher argmax = dataset gold | teacher distribution |
   | C and anything unlabelled | 35B and a second open model (Qwen3.6-27B or gpt-oss-20b) agree | 35B distribution |
   | D (traps) | inherits the parent's rule; teacher argmax = constructed gold | teacher distribution |
   | **I (image branch) [rev3]** | constructed gold (joint-rule generator) or, for mined image items, the 9B's answer and the 35B's text-side answer agree | constructed gold → hard label + 9B distribution when they agree; mined → 9B distribution |

   Drop everything else.
4. **Hard `unknown` variants.** For ~15% of kept items, remove the decisive fact: programmatically for A, D and the image generator (drop the decisive record attribute), with the 27B writer for B/C. **[rev3]**
   Keep only when the teacher puts most of its mass on `unknown`. This holds the unknown-gold share at ≥ 15% and stops hard training
   from teaching the model to guess.
5. **Variants of verified failures** (so the model learns the pattern, not the item).
   - For each verified item, make 2 variants.
   - Source A/D/I-generated: same family and difficulty, new seed, at no teacher cost when the gold is constructed. **[rev3]**
   - Source B/C: the 27B writer perturbs it (different numbers or entities, a flipped condition, a reordered chain),
     then the teacher re-verifies under the same keep rules.
   - Cap at 2 per parent; drop a variant if its 13-gram overlap with the parent is > 70% (no near-copies).
   - This adds ~30–40k items, about 40% more teacher time.
6. **Expected result:** **75–95k verified hard items** (45–60k mined + 30–40k variants), plus ~10k unknown variants.

## Stage 2b: human-checked slice (owner or reviewer, ~3–4 h of review, $0)

- **What:** 400 kept items, stratified by source (A/B/C/D/I, variants, unknown variants) and by family; ~60 of them image items (the page shows the photo). **[rev3]**
- **Review page:** a local page (built by a new `scripts/p3/build_review.py`) shows the state, question, options, the kept answer
  and the teacher's rationale, with buttons *correct / wrong answer / ambiguous / bad question* and a note field.
- **What we learn:**
  - the real label error rate per source;
  - whether "unknown" variants are really unanswerable.
- **Rule:** if a source's error or ambiguity rate is > 5%, tighten its keep rule (e.g. require two-teacher agreement for it too)
  or drop it before training.
- **These 400 items become the clean human-verified dev set.** They are never trained on, and they are used for checkpoint
  selection alongside the held-out hard set.
- **Timing:** the pod pauses between Stage 2 and Stage 3 while this happens. Either stop the pod (the disk is kept) or,
  better, run the review on the pilot's output and a first mining shard ahead of time, so the pod never waits.

## Stage 3: train one 4B run (pod, ~2.5 h on all 8 GPUs) **[rev2]**

**Start point:** the shipped soup50 4B (LoRA + readout), delta lane. This keeps what works and adds hard skill on top.

**Mixture (~130k rows) [rev2]:**

| Share | Content |
|---:|---|
| 45% | verified hard text items (+ unknown variants). **[rev3]** Stage 2 yields more than this share holds (75–95k); subsample stratified by family × difficulty × source, keeping every unknown variant and every human-verified-family item, rather than growing the mixture. |
| 20% | text replay from the phase-2c mixture (keeps easy, standard and judge accuracy) |
| **35%** | images: ~20% **hard image branch** (Stage 0-I, verified) + ~15% image-decision replay. Phase 2c used ~13% replay only and failed the image gate; the goal here is that the hard lane passes the gate on its own. |

**Loss and flags:**
- `--soft-targets --soft-weight 1.0` (the teacher distribution carries the "reasoned" answer into one pass).
- **`--permute-options`**: train-time option shuffling, so inference needs **no rotations** (4× faster, better JevBench speed and cost).
- `--rationale-weight 0.3 --rationale-max-tokens 192`.
- **New: ordinal loss for score questions**: an expected-level / earth-mover term next to CE. A small trainer change with a test. **[rev2]** Weight 0.1–0.2, and a 200-step pilot ablation with and without it before the full run; a new loss term is the one thing here that can silently hurt everything else.

**Hyperparameters:** 2 epochs, lr 2e-5, as the phase-2c delta lane. Checkpoints every ~15% of steps; the eval lane gates each one. **[rev2]** One size only, so train on all 8 GPUs (phase 2c: 39.5k rows × 2 epochs took 1.4 h on 4 GPUs for the 4B; 130k rows with a third images is ~5 h on 4, ~2.5 h on 8); the eval lane then runs after training, not beside it.

## Stage 4: evaluate, gate, pick (pod eval lane, overlapping training; ~1 h tail)

- **Every checkpoint:**
  - own held-out hard set (Stage 1) + the human-verified slice (Stage 2b), the selection metrics;
  - JevBench public, single pass raw and calibrated;
  - ImajevBench + probes + unknown/false-abstention + irrelevance gates (`gates.json`);
  - DecisionBench **3k stratified dev subset**, internal tracking only (single pass, ~15 min per checkpoint);
  - fast-decisions dev as an off-distribution check.
- **Pick:** the best fresh-seed held-out checkpoint that passes every gate. **[rev2]** When two are within noise, prefer the later one (phase 2c's dev-selected step 260 had not yet absorbed the image replay). Selection reads only the held-out set and the human slice; the DecisionBench subset is written to the log, never read by the selector (enforced in the pod script).
- **If none passes the image gate:** soup it with the shipped adapter (the phase-2c fix), re-gate, and take the best passing soup weight.
- **Calibration:** **[rev2]** fit T on the flagged half of the held-out hard set (our own data, JevBench-hard difficulty), single pass and rot4 separately; report hard ECE, pooled public ECE and the authored-dev fit side by side. The authored-dev fit is kept only if the hard-set fit raises pooled public ECE above 0.03.
- **Ship only if** the target table's rules hold. Otherwise keep 1.0 and keep the data: it is reusable.

## Stage 5 (optional, cheap): round 2

Re-mine the Stage-0 pool with the new checkpoint. Items it newly fails are **already teacher-labelled**, so no new teacher cost.
One short extra epoch on them plus replay. This costs about 1.5 h of pod time and is decided after Stage 4.

## Budget and schedule

| Step | Where | Time | Cost |
|---|---|---|---|
| Stage 0: generators (incl. the image branch), 4 converters, licence + decontamination, trainer ordinal loss + tests, review page | Mac (agents) | **[rev2]** 2–3 days | $0 |
| Pilot (1k teacher items + 5k mining throughput test) | 1×H100 | ~30 min | ~$2 |
| Stages 1–4 (incl. variants) | 8×H100 (~$21.5–28/h) | **[rev2]** ~9–11 h (mining 1.5 h, teacher 3 h budgeted at half the plan's throughput, training 2.5 h on 8 GPUs, eval 1–1.5 h, transfers) | **[rev2] ~$220–300** |
| Stage 2b human review | owner / reviewer | ~3–4 h | $0 |
| Stage 5 (optional) | same pod | ~1.5 h | ~$35–45 |

**Efficiency rules:**
- Mine before labelling, so the teacher sees ~25% of the pool.
- Teacher labels are cached and reused across rounds and sizes.
- Pilot before the full pod.
- Code and data go to the bucket; nothing large goes from the Mac to the pod.
- One pod, stages pipelined. Terminate right after pulling.

## Open decisions for the owner

1. **Budget ceiling:** **[rev2]** $300 for Stages 1–4 (recommended), or $345 with Stage 5.
2. **Who reviews the 400-item human slice** (Stage 2b): you, or someone you trust. About 3–4 hours; it can start on pilot output before the pod.
3. **Agent-action public data** (Mind2Web / WebShop / ALFWorld): include if the licences pass (recommended), else generators only.
4. **Deferred to a later version** (architecture changes, not data):
   - a separate `unknown` head to lift the 254-option limit (+~2.2 pts on DecisionBench coverage);
   - a native multi-label question type;
   - long-context training beyond the needle family.


## Licences and disclosure **[rev2]**

- FOLIO, HotpotQA and ARC are CC-BY-SA 4.0. Weights trained on them are fine under our posture, but a published phase-3 *dataset*
  containing derived rows would be share-alike. Decision: publish the generators and the mined-item ids, not the converted SA rows,
  unless the owner wants the whole set under CC-BY-SA.
- Training on the train splits of FinQA, TAT-QA, MuSiQue and FOLIO is legitimate after the upstream-id exclusion, and it is
  disclosed on the model card and in the results log ("trained on the train splits of …; every DecisionBench upstream id excluded").
- The DecisionBench 3k subset is tracking-only: the selector cannot read it (see Stage 4).
- Nothing from JevBench, ImajevBench (public or private-1) or Jev/paid-API outputs enters the pool; the decontamination report is
  published with the run.

## How to reach +6 on the hard tier (70.3 → ~76) **[rev2]**

One data round buys +2 to +4. Six points means stacking levers, each measured on its own so we know what paid:

| Lever | Expected | Cost | Notes |
|---|---|---|---|
| Phase 3 verified hard data (this plan) | +2 to +4 | this plan | the same lever as phase 2c, done properly |
| Round 2 on newly failed items (Stage 5) | +0.5 to +1 | ~$40 | labels already cached |
| Serving with 4 rotations | +0.9 (measured) | latency ×3 | keep as the "quality mode"; the board can run it |
| Two-adapter ensemble at serving (average of two phase-3 checkpoints, or phase-3 + phase-2c) | +0.5 to +1.5 | latency ×2 | cheap to test on the pod's eval lane; soups already showed averaging is benign |
| Calibration on the hard set | 0 on accuracy, big on the board's Calibration axis | $0 | the surest board gain of all |
| **[rev3] Future, outside this 4B data phase.** A short "think then decide" mode: the model writes ≤ 64 tokens of rationale (it is already trained with the rationale loss) and the readout scores at the end of it | +2 to +5 | latency ×5–10; a different serving mode | the 35B with thinking is at 97.3: reasoning tokens are where the ceiling is. Offer it as `--think` alongside single pass; report both rows honestly |
| **[rev3] Future, outside the 4B-only scope.** A 9B phase-3 run on the same cached labels | +1 to +3 over the 4B | ~$120 | the 9B absorbs reasoning better and needed the soup less in phase 2c |

Realistic stack for the 4B: phase 3 (+3) + round 2 (+0.5) + rot4 (+0.9) + ensemble (+1) ≈ 75–76 in quality mode, ~73 single pass.
Past that, the next jump is the base model (a newer small dense base with a better vision tower), not more data.

## Images: more data instead of the soup **[rev2]**

The soup was needed in phase 2c because the hard mixture carried ~13% image rows, all replay of easy decisions, and the joint
image+state rule families lost 7 items. The fix in this plan is two-fold: (1) the **image branch** in Stage 0-I adds ~60k *hard*
image decisions targeted at exactly those families (threshold, exception, multi-step over a record and a photo), mined and
generated on our own licensed photos, and (2) the image share rises to ~35% of the mixture. The gate stays; the expectation is that
the hard checkpoint passes it directly, so what ships is a single trained adapter rather than an average. The soup remains only as
the documented fallback. If the image branch cannot be built in the Stage-0 window, run the plan with 30% replay and accept the soup;
do not skip the gate.


## Field note, 2026-09-25: plumb-4b and blink-4b **[rev2]**

Two text-only Qwen3.5-4B decision models appeared within a day of our launch. Both matter for this plan.

- **plumb-4b** (crh225, Apache-2.0, GGUF available): a *full fine-tune of JevK5 v0.2* on **5,014** teacher-written questions
  (Qwen3.8-27B writes a realistic 1,500–2,800-word document plus 2–3 questions, then solves each twice with thinking; kept only when both
  solutions match the intended answer), families weighted to JevBench's published categories and to the starting model's weak spots,
  hard-mined ("trained harder on the questions it got wrong or was unsure of"), replay from public human-labelled sets, one T (2.07) fitted
  on held-out teacher questions. Public hard 89/111 = **80.2** from JevK5's 73.9 (+8 items fixed, 1 broken, McNemar p = 0.039);
  long_policy 11 → 14, probability 7 → 10, temporal_numeric flat at 7/15. Their own held-out check picks checkpoints; public aggregates
  "informed later recipe decisions" (disclosed). The dataset (`crh225/plumb-decisions`, train 5,014 / test 131) is Apache-2.0 and
  written by an open-weight teacher, so it is admissible under our data posture.
- **blink-4b** (thegovind, weights non-commercial): vision cut, three-checkpoint soup, 80/111 = 72.1, hard ECE 0.067 with T = 1.0,
  699 input tokens per decision (ours: 1,269), p50 64 ms. Selected on JevBench public items and JevK5's hand-written hard set (disclosed).

Our shipped 4B per family on hard (rot4 + cal, 78/111): long_policy 11/19, multi_hop 14/18, judge_hard 10/17, temporal_numeric 7/15,
probability 7/10, trap 8/8, ambiguous 6/7, tradeoff 5/6, adversarial 5/6, routing_hard 5/5. The misses concentrate in long_policy,
judge_hard and temporal_numeric — the same three families plumb attacked.

**Changes to this plan from the note**
1. Quality over volume: plumb moved 8 items with 5k double-solved questions. Cut the Stage-0 text pool target from 330k candidates to
   ~120k and spend the saved teacher budget on **two independent thinking solves per kept item** (keep only on agreement), which is
   stronger than our single-solve + argmax check.
2. Add a **long-document family** (1,500–2,800-word policies, contracts, logs, schedules) to generator A and to the teacher-written
   items; our long_policy is 11/19 and it is the largest hard family.
3. Add `crh225/plumb-decisions` (train split only; its 131-item test split is kept out) to source B after the 8-gram/13-gram lint.
4. Weight the mining pass toward long_policy, judge_hard and temporal_numeric.
5. Prompt length: audit our JevBench text template against SemIf's layout; 1,269 vs 699 tokens per decision is a speed/cost gap that no
   training fixes.
6. A JevK5-initialised branch is *not* adopted: JevK5 is text-only, and our adapter must keep the vision path; the comparable move for us
   is the image branch already in Stage 0-I.

## Measured: Azure teacher test (2026-09-25, 12:21–12:58 IST) **[rev2 addendum]**

Teacher Stage 2 runs on Azure credits (owner: ~$200), not RunPod.

- **Deployment:** Foundry project `mcp-servers`, **GlobalManagedCompute**, model `azure-huggingface/qwen--qwen3.6-35b-a3b` v2 (bf16),
  template `qwen--qwen3-6-35b-a3b--256k-nvidia-h100`, **1 × H100 80GB per instance, $7.91/h**, quota up to 10 instances.
- **API:** OpenAI-compatible chat completions on the account endpoint, `api-key` header, model = deployment name.
  - `chat_template_kwargs` is rejected, so thinking is always on. That's what the teacher wants; the reasoning comes back separately.
  - `response_format` json_schema works.
- **Test:** real `gen_answer.py` distribution prompts on 472 phase-2b hard questions (`scripts/p3/azure_teacher_bench.py`, raw results
  in `reports/phase3/bench-*.jsonl`).
  - Output: **472/472 parsed, 0 truncated**; ~1,800 completion tokens on average (p90 2.7k, max 6.3k).
  - Throughput: **~2,100 output tok/s per instance** at 64 concurrent requests (128 gives no gain), i.e. **~4,200 items/h per instance, ~530 items per $**.
- **Sizing:** 8 instances ≈ 33.6k items/h at $63.28/h. **Teacher budget ≈ $140 → ~74k items in ~2.2 h**; 27B-FP8 second opinion ~$15;
  the rest is buffer. Stage 1 must therefore hand the teacher **≤ 75k items**.
  - A/D items and their variants need no teacher when the gold is constructed.
  - If mining flags more than that, prioritise by family balance and by confidence (most confidently wrong first).
- **Consequence for the budget table:** Stages 1–4 real money = RunPod training + eval only (~$80–110). Mining runs on the Mac.

## Streamed mining + teacher (owner, 2026-09-25) **[rev4]**

Stages 1 and 2 run as **one streamed session** instead of mining overnight and labelling later.

**Setup.** A small rented pod (4 GPUs, A100 or L40S class, ~2 h, ~$10–20) mines the pool with the shipped 4B, single pass. The Mac is not used.

**Flow.**
- Every flagged item is pushed to a queue.
- The Azure teacher (8 × H100 instances, credits) consumes the queue as items arrive.
- Second solves and variants go through the same queue.
- Throughput is sized so mining (~100k items/h on 3–4 GPUs, ~35% flagged) keeps the teacher (~34k items/h) busy.

**Prioritisation without a global pass.**
- The pool is fed in shuffled, family-balanced shards with per-family quotas: larger for long_policy, judge_hard and temporal_numeric.
- A family stops feeding the teacher once its quota is filled.
- The overall cap stays ~70k items, about $130–140 of credit.

**Order.**
1. Tools (mining script, queue, Azure runner with a hard cost cap, variant and unknown generators, review page) are built first.
2. Session start: 1 Azure instance + the pod → 1k-item pilot (~20 min); the owner starts reviewing the first ~100 items.
3. The owner scales Azure to 8 instances in the portal; streaming runs ~2–2.5 h.
4. Pull everything, terminate the pod; the owner deletes the Azure deployments.

**Review.** Review findings can drop a source's labels afterwards; every teacher output is kept separately.

## Review: open-weight model + small human slice (owner, 2026-09-25) **[rev4]**

This replaces Stage 2b's 400-item human review.

**1. Model review of ~2,000 kept items.**
- Reviewer: **Kimi-K2.5 Thinking**, an open-weight model already deployed on the owner's Azure account <azure-account> (deployment
  `Kimi-K2-Thinking`, GlobalStandard, pay per token on credits, ~$10–20). Its open-weight licence keeps us inside the "no paid-API
  outputs" rule; confirm the exact licence text before use. Closed frontier APIs are not used, because of that rule and their
  terms on building competing models.
- Sample: stratified by source (A/B/C/D/I, variants, unknown variants) and family, with ~300 image items (Kimi-K2.5 reads images;
  confirm on the deployment).
- For each item, Kimi solves it independently and flags *wrong answer / ambiguous / badly posed / not really unknown*.

**2. Owner review of ~100 items (~45–60 min).**
- Every item where Kimi disagrees with our kept label (up to ~60).
- About 40 random items, to calibrate Kimi's judgement.

**Rules.**
- Per-source error rates come from Kimi's reviews, corrected by the owner's check of Kimi's disagreements.
- A source above 5% confirmed errors gets a tighter keep rule or is dropped.
- The dev set becomes the ~100 human-verified items (the core) plus the Kimi-reviewed items that agree with our label. Never trained on.

## Model-side engineering results (2026-09-25) **[rev4]**

**Item 5 (256 codes).** Gate 0 passed bit-identical.
- 686/686 questions (all with ≤ 254 options; including the 231 JevBench public items and the 150 authored dev items) give identical
  logits and probabilities on MLX, max abs diff 0.0 (`reports/phase3/parity-readout256.md`).
- Switches: `--readout-codes 256` (trainer, server), default 255.
- **Still to do before the pilot lane can mean anything:**
  1. Include ~1–2k **255-option training items**. Row 256 starts at its LM-head init and is read only by 255-option questions.
     Build them by construction, e.g. entity matching over 255 candidates or a long catalogue pick.
  2. Rerun the parity check on the pod's CUDA torch path before the +256 lane.
  3. Update `scripts/p3/candidate.py` `MAX_OPTIONS` for that lane, and the release copies (Space, READMEs) at release time.

**Item 14 (compact prompt layout).** **Dropped from the pilot lanes.**
- The layout only saves 4.7% on JevBench hard (1,269 → 1,210 tokens), because the state itself is 85% of the tokens.
- The "1,269 vs 699" comparison in the field note mixed tiers: our mean over all JevBench tiers is 691.
- The code stays in, behind `--prompt-layout`, default standard. The pilot is now **4 lanes**: baseline / +ordinal / +256 codes / +both.
  **(2026-09-26, owner-approved: "+both" is replaced by "+rank64", the shipped r16 LoRA expanded to r64 with an identical output at
  step 0; the three factors are decided independently and combined: `reports/phase3/pod-runbook.md` "Pilot".)**

**Item 4 (`multi` type).**
- Built: fan-out to per-label nouls in one backend call, at most 32 labels per question.
- Training rows for multi-label families must use `jev_api.multi_label_question()` wording, so training matches serving.

## Pre-live decisions (lead, 2026-09-25 ~21:30 IST) **[rev4]**

Integration dry run: every step PASS (`reports/phase3/dry-run.md`); tests 792 + 1 new review test pass. Final runbook: `reports/phase3/LIVE-RUNBOOK.md`.

1. **Mixture size.** At manifest build, pass `--share-hard-image 0.15 --share-replay-image 0.20`. Images stay 35%, and cheap image
   replay fills the gap. The default 20/15 split would bind the total at ~64k rows because verified hard images are the scarcest
   share. The other shares are unchanged: 45% hard text, 20% text replay.
2. **Dev slice cap.** `review_report.py --max-kimi-dev 600` (the default). Every owner-verified item is kept, plus ≤ 600 Kimi-agreed
   items, stratified by source and family. Each dev item removes its group from training, so the uncapped slice would cost ~3–4k
   training rows.
3. **Quotas v2.** No 27B writer frees the ~13.5k planned B/C variant calls. Image quotas ×1.35 and text ×1.15, each capped at the
   family's expected flagged count: first solves 57.4k (text 35.9k, image 21.5k), ~63.6k calls ≈ $120, plus ~4k held-out labels
   ≈ $8. Teacher cap stays $145. Previous file: `data/p3/pool/quotas.v1.json`.
