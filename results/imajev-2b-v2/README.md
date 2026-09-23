# imajev v2 — 2B, pseudo-labelled new sources + self-distillation (23 Sept 2026)

Qwen3.5-2B + the v1.1 LoRA (r16/α32, language layers) + 255-code readout, continued from the v1.1 adapter on
`data/manifests/decision-v2.jsonl` (940,153 records; sha256 `5cf6f388…`): the 504k v1.1 training decisions with targets blended
0.5 × base-2B distribution + 0.5 × label (self-distillation regulariser), plus ~416k decisions on six never-seen sources labelled by the
calibrated 9B v1.1 teacher (commons_photos, openimages_v2, pd12m, stackexchange, wikipedia_paragraphs, support_reviews; keep rule
top ≥ 0.6 or unknown ≥ 0.5 with argmax agreement across two option orders; 74–80% kept). Recipe: 0.4 epoch, lr 5e-5, 20-step
checkpoints, best checkpoint by `mean_accuracy` over the v1.1 dev set and a reasoning dev set (`decision-v2-reasoning-dev`, 400 cases).
4× H100 SXM (RunPod Secure, $13.96/h), 1,866 steps, **zero crashes** (faulthandler timer removed), best = step 1,100
(dev 86.8 → 89.2%, reasoning dev 42.7 → 46.3%). Labelling ~2.5 h on the same pod, training ~1.5 h, evals ~40 min.

Adapter: `reports/decision-v2/runs/h100x4/best` (MLX copy `best-mlx`); temperatures `calibration-v2.json` (fitted on the 25,217-decision
text calibration fold; **0.27–0.69**, i.e. the raw model is under-confident because it was trained on soft targets).
Per-panel numbers: `eval/v2-comparison.md` (from `scripts/research/compare_v2.py`); probes `../v2-datasets/*-2b-v2.json`.

## Headline

| Panel | 2B v1.1 | 9B v1.1 | **2B v2** | Δ vs v1.1 |
|---|---:|---:|---:|---:|
| v1.1 test (35,528 decisions) | 87.3% | 89.0% | **86.1%** | −1.2 |
| · image / text | 86.1 / 88.7 | 87.1 / 91.2 | **84.7 / 87.8** | −1.4 / −0.9 |
| · MMLU in test (never trained) | 53.5% | 73.7% | **43.0%** | −10.5 |
| · false abstention on answerable | 1.6% | 1.4% | **3.6%** | +2.0 |
| · ECE raw / calibrated | 0.016 | 0.007 | **0.151 / 0.033** | |
| · auto-accept ≥ 0.95 (calibrated) | 61% @ 98.6% | 64% @ 98.8% | **58% @ 95.0%** | |
| v1 image exam, held-out sources (4,989) | 50.7% | 55.9% | **54.4%** | **+3.7** |
| · heldout_pairs (two images, 177) | 37.9% | 26.6% | **54.2%** | **+16.3** |
| · heldout_abstention (1,500) | 65.2% | 73.3% | **72.0%** | +6.8 |
| · heldout_livewild (1,162) | 51.6% | 52.2% | **56.6%** | +5.0 |
| · heldout_fashionpedia (983) | 47.7% | 58.1% | **51.6%** | +3.9 |
| · heldout_countqa (1,167) | 35.6% | 39.9% | **32.0%** | −3.6 |
| v1 image exam, trained sources (19,232) | 86.0% | 87.1% | **84.7%** | −1.3 |
| State-grounding probe (200, authored) | 60.5% | – | **69.5%** | **+9.0** |
| Two-image pairs probe (60, authored) | 68.0% | – | **65.0%** | −3.0 (2 items) |
| New-source test rows, agreement with teacher (41,533) | – | (teacher) | **96.4%** | base 2B 66.9% |
| typed-decisions test (2,000) | 58.1% | 66.2% | **59.1%** | +1.0 |
| SST-5 (2,210) | 47.5% | 51.0% | **44.2%** | −3.3 |
| Irrelevance panel (2,823) | 64.3% | 79.6% | **55.5%** | −8.8 (MMLU-with-image 51.8 → 40.4) |
| Reasoning dev, authored 240 (Mac, MLX) | 46.7% | 63.7% | **42.5%** | −4.2 |
| JevBench hard 111, raw / calibrated | 43.2% (ECE 0.254) | 42.3% | **44.1%** (ECE 0.227 / 0.396) | +0.9 |
| JevBench original 72 | 88.9% | 98.6% | **87.5%** (ECE 0.080 cal.) | −1.4 |
| JevBench easy 48 | 100% | 100% | **100%** | |

Paired on the v1.1 test: 682 fixed, 1,094 broken. Per-source deltas are −0.1 to −3 everywhere except MMLU (−10.5), koniq (−5.7), esci (−4.5),
goemotions (+2.2), massive (+1.1).

## Reading

- **The image goal moved.** Held-out image sources +3.7 (54.4%, within 1.5 of the 9B), two-image pairs +16, quality (livewild) +5,
  unseen-source abstention +7, and the state-grounding probe +9. This is what the new photo sources and the teacher's `unknown` labels
  bought. Counting got worse (−3.6); nothing in the new data counts.
- **In-distribution text and knowledge questions moved the other way.** MMLU −10.5 and the irrelevance panel −8.8 are the same effect:
  v2 abstains far more on knowledge questions (MMLU predicted-unknown 6.4% → 29.3%; 101 of the 129 broken MMLU cases are abstentions).
  Cause: ~30% of the teacher-labelled rows are `unknown` by construction and the blended v1.1 targets carry the base 2B's unknown mass
  (base false abstention 20.9%). The student learned a stronger abstention prior. A post-hoc offset on the unknown logit
  (`scripts/research/unknown_offset_sweep.py`) recovers MMLU to 50–54% but costs 2–4 points on image decisions, so it is a genuine
  boundary shift, not a free fix.
- **Reasoning did not improve** (authored set −4, JevBench hard +0.9, within noise on 111 items). The regulariser held reasoning roughly
  where v1.1 left it (dev-set mean accuracy rose during training, the authored subset did not); it did not restore the base model's 45.9%.
- **Calibration changed character.** Raw probabilities are the soft blend the model was trained on (temperatures 0.3–0.7). After fitting,
  in-distribution ECE is 0.033, but the sharpened temperatures make off-distribution items over-confident (JevBench hard ECE 0.40 vs 0.23
  raw). For serving, use the raw model or a milder temperature; the v1.1-style 98.6%-precision auto-accept band is not available at 61% coverage.

## Was the 9B → 2B pseudo-labelling worth it?

Yes for the image proposition (held-out sources, two-image, abstention, state grounding all up; no other run moved these), no for text
knowledge (MMLU) and reasoning. The two are the same lever pulled in opposite directions: the teacher's abstention behaviour transferred
well and over-generalised to answerable knowledge questions. v2.1 (state-grounded and pairs data, continued from this adapter) inherits
the same prior; the fix for the abstention prior is a data-mix change (cap unknown rows at ~15%, drop the base-blend on answerable text
rows) or a per-type unknown offset fitted on the calibration fold, not more training of the same kind.

## Files

- `eval/tuned-2b-v2/`, `eval/base-2b-v2/`, `eval/calibration-2b-v2/`: pod predictions on the decision-v2 test partition (77,061 rows =
  v1.1 test + 41,533 new-source rows) and the calibration fold.
- `panels/{v1exam,typed,sst5,irrelevance,reasoning}-tuned/`: pod predictions on the held-out panels.
- `jevbench-raw/`, `jevbench/`: public splits without / with `calibration-v2.json` (local MLX serving, port 8766).
- `runs/h100x4/run.combined.log`: pod log (label, blend, assemble, train, eval).
