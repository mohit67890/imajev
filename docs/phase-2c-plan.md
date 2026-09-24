# Phase 2c ("v1.1 recipe") — decided 2026-09-24 ~10:00 IST: run BEFORE launch

Goal: JevBench official top-3 odds ~50% (from ~25%), by moving every axis we control: hard tier (judge + exact-answer families),
calibration (soft targets), and no inference-time rotations (train-time option permutation). Evidence: Eikos-4B (same base, same
recipe) self-reports hard 72.1 / ECE 0.049. Budget: ~3 days, one 7×H100 pod for ~5 h (~$150).

## Day 1 (Mac, engineering) — running as three agents
1. `scripts/p2/filter_eikos.py` + `docs/eikos-decisions-usage.md`: CC-BY-4.0/MIT slice of `caiovicentino1/eikos-decisions` (`--teacher-policy open-weights`: GLM-5.3-Flash is open-weight MIT,
   run by the authors locally, so allowed), minus non-allow-listed licences (ODC-BY FinEntity pending owner decision), minus JevBench
   8-gram hits (1,148 rows incl. all `probability` items, which copy JevBench hard wording) → 18,120 train + 1,092 dev records with
   `target_probs`; heldout split kept as an extra eval. DONE.
2. Trainer (`scripts/train_decision_lora_torch.py`): `--soft-targets` (KL to `target_probs`), `--permute-options` (train-time option
   shuffling for choice questions), `--rationale-weight W` (auxiliary LM loss on a ≤64-token rationale after the decision position).
   Defaults off; tests.
3. `scripts/p2/families.py` + `scripts/p2/gen_programmatic.py`: judge_pairwise, judge_rubric_score (writer families) and
   temporal_arithmetic, probability_exact, numeric_reconciliation (deterministic exact-answer generators); tests incl. 8-gram lint.
4. STILL TO WRITE (after agent 3 finishes, to avoid conflicts in scripts/p2/): `gen_answer.py --mode distribution` — the 35B-A3B
   (thinking) returns a full probability distribution over the options (+unknown) via JSON schema, plus a ≤2-sentence rationale;
   `assemble_p2.py` keeps `target_probs` (agreement rule: argmax must match the intended answer; keep the 35B's distribution as the
   soft target) and `rationale`. Re-label ALL existing phase-2/2b questions this way so the whole mixture has soft targets.
5. Manifest builder for phase 2c: phase-2 + 2b records (re-labelled) + judge/programmatic new rows + filtered Eikos slice (attributed),
   held-out domains for calibration, dev = p2b dev ∪ jevstyle dev ∪ a new judge-style dev (never JevBench items).

## Day 2 (7×H100 pod, ~5 h)
- Generation: writer (27B) for judge families ~1,500 docs; programmatic generators (CPU) ~3,000 docs; 35B-A3B distribution+rationale
  labelling of everything (~25k questions) on 6 GPUs (~1.5 h); assembly + manifest.
- Training (two lanes as in `cloud/pod_run_p2b_train_pipelined.sh`): 4B, 9B, 2B from the phase-2b best adapters with
  `--soft-targets --permute-options --rationale-weight 0.3`, 2 epochs, lr 2e-5 (4B/9B) — rationale tokens ~double the tokens per example.
- Evals: full panel + JevBench raw (no rotations needed) + calibrated + ImajevBench; same-protocol Eikos-4B/27B and Winnow (`--model Winnow-12B`).
## Day 3 (Mac): temperature refits, off-distribution checks, cards, leaderboard rows, handoff; then launch + JevBench submission with v1.1 adapters.

Rules unchanged: no paid-API outputs (Eikos's GLM teacher is open-weight, so its rows are allowed), no JevBench items, one local MLX server at a time, price before pods,
terminate after pulling, pod-to-pod transfers, keep raw benchmark responses.

## Amendments after the Eikos code review (docs/eikos-analysis.md, 2026-09-24)
- TWO training lanes on the pod for the 4B, ship the better: (a) delta lane as planned (from phase-2b best, r16, lr 2e-5, 2 epochs,
  `--soft-targets --rationale-weight 0.3 --rationale-max-tokens 192`); (b) FRESH lane = Eikos hyperparameters on our readout: new LoRA
  r64 / α128 / dropout 0.05 on all linear layers incl. DeltaNet projections, lr 1e-4, weight decay 0, ~20 warm-up steps then cosine,
  1 epoch, pure soft CE (`--soft-weight 1.0`), rationale 0.3, gradient checkpointing, plus a 30–40% replay of image decisions (v1.1/v2.1
  rows) so ImajevBench does not regress. 9B: delta lane only (fresh 9B if the 4B fresh lane wins and time allows).
- Soft targets: 35B-A3B thinking, keep a row only when its argmax = intended answer, no sharpening; audit its mass on `unknown` when the
  gold is a real option. Eikos ships no `unknown` and abstains by threshold; we keep `unknown` as an option (our differentiator).
- Eikos 4B's comparable number is the SERIAL 72.1 (official runner), not the batched 73.9. Their 1 epoch ≈ 0.8 (step-count bug).
- Temperature: fit on held-out teacher rows with hard rows weighted; ship 1.0 if the fitted T < 1. No inference rotations.
- Probability family: our own phrasing (theirs copies JevBench hard); cap templated sources per family as Eikos does.
- Skip: PT↔EN view loss, JEPA, long-context dossiers, run averaging.

## Owner gates (2026-09-24): image accuracy must not regress; `unknown` must be retained
- Both 4B lanes (and the 9B/2B delta lanes) train with an image-decision replay: fresh lane 35%, delta lanes 15% (from decision-v2.1-4b
  train rows with images, original labels).
- Every record keeps `unknown` as an option; `target_probs` always carries an `unknown` key; the assembler audits the 35B's unknown mass on
  answerable rows (flag >0.2, drop if it flips the argmax); the manifest keeps ≥15% unknown-gold share where possible (warn otherwise).
- Ship gates per run (PASS/FAIL printed by the eval lane, `gates.json`): ImajevBench ≥ phase-2b − 1.0 pt with visual/joint tracks within
  2 items; state/pairs probes within 2 pts; correct-unknown rate on unknown-gold rows ≥ phase-2b and false abstention ≤ phase-2b + 2 pts;
  irrelevance test within 2 pts. A run failing any gate is NOT shippable regardless of its JevBench number.
