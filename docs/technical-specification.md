# imajev 1.0: technical specification

Everything a developer or reviewer needs to judge how the three imajev models were built: architecture, data, training, compute,
evaluation sets and the checks behind them. Every number comes from the manifests, run configurations, logs and billing records in
this repository (sources are listed in `results/` and in the per-stage reports). Where a fact could not be recovered it says so.

Terms: a **decision** is one typed question answered about one request; a **record** is one request, which can hold up to eight
decisions. Stages are numbered in the order the models were trained.

## 1. Models

| | imajev-2b | imajev-4b | imajev-9b |
|---|---|---|---|
| Base model (pinned revision) | Qwen/Qwen3.5-2B @15852e8c | Qwen/Qwen3.5-4B @851bf6e8 | Qwen/Qwen3.5-9B @c2022362 |
| Base licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Adapter | LoRA r=16, α=32, dropout 0, no bias | same | same |
| Shipped adapter (phase 2c, "soup50") | element-wise average, ½ + ½, of the stage-3 adapter and the stage-4 best checkpoint (LoRA A and B and the readout); sha256 4c8981a9… | same; d8d328f8… | same; 1a841bdc… |
| LoRA targets | every language-model projection: `q,k,v,o`, `gate,up,down`, and the DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; vision encoder frozen, no LoRA | same | same |
| LoRA parameters | 15,630,336 (300 tensors) | 30,474,240 (400) | 40,108,032 (400) |
| Decision readout | one bias-free linear layer, 255 × 2048, float32 | 255 × 2560 | 255 × 4096 |
| Readout parameters | 522,240 | 652,800 | 1,044,480 |
| Total trainable parameters | 16,152,576 | 31,127,040 | 41,152,512 |
| Adapter file (`adapter_model.safetensors`, F32) | 62.6 MB | 122.0 MB | 160.5 MB |
| Readout file | 2.1 MB | 2.6 MB | 4.2 MB |
| Base weights to download | 4.6 GB | 9.3 GB | 19.3 GB |
| Shipped calibration temperature | 1.646 | 1.717 | 1.748 |

**How an answer is produced.** Each supplied option is bound to one of 255 single-token codes. The prompt renders the images, the
state and the question with its option list, then a decision position. The readout maps the hidden state at that position to logits over
the option codes plus an `unknown` code; a softmax (after the calibration temperature) gives the probabilities. One prefill per request,
one forward pass per question, no text generation. With `--rotations N` the server averages N option orders; every released number was
measured with 4 rotations and the shipped temperature.

**Limits per request.** 0 to 2 images (resized to at most 400,000 pixels), a state string or JSON object up to 32 KB, 1 to 8
questions, 2 to 254 options per `choice`, 2 to 10 levels per `score`, at most 4,096 tokens (longer requests are refused, not truncated).
English only.

**Precision.** Base weights run in bfloat16; the LoRA and readout weights are stored in float32. The MLX copies under `mlx/` are
converted from the same PEFT files.

## 2. Training data

### Stage 1: image and text decisions (2B and 9B; also part of the 4B's first run)

504,000 training decisions in 466,135 records, from 36 licence-admitted sources:

- **200,000 decisions from 15 text sources.** 24,832 of them carry an unrelated photo as an irrelevance control.
- **300,000 decisions from 21 image sources.**
- **4,000 photo-vs-listing contradictions** (Amazon Berkeley Objects).

Held-out partitions of the same manifest: dev 30,026 decisions, calibration 25,217, test 35,528 (19,232 with an image, 16,296 text-only,
including the 1,000 MMLU evaluation questions).

| Text source | Decisions | Licence | | Image source | Decisions | Licence as recorded |
|---|---:|---|---|---|---:|---|
| civil_comments | 26,438 | CC0-1.0 | | abo | 38,413 (+4,000 contradictions) | CC-BY-4.0 (images and listings) |
| helpsteer2 | 20,092 | CC-BY-4.0 | | gqa | 25,926 | CC-BY-4.0 annotations; Visual Genome images |
| goemotions | 13,428 | Apache-2.0 | | tdiuc | 22,505 | CC-BY-4.0 annotations; COCO images |
| clinc150 | 13,219 | CC-BY-3.0 | | masked_evidence | 22,384 | derived from gqa / vg_attributes |
| esci | 13,219 | Apache-2.0 | | state_aware | 22,252 | inherits each origin source |
| fever | 13,219 | CC-BY-SA-3.0 | | vqav2 | 21,807 | CC-BY-4.0 annotations; COCO images |
| massive | 13,218 | CC-BY-4.0 | | vg_attributes | 19,535 | CC-BY-4.0 annotations; Flickr images |
| snips | 13,218 | CC0-1.0 | | fashion200k | 18,804 | Apache-2.0 annotations; image rights unverified |
| snli | 13,218 | CC-BY-SA-4.0 | | vizwiz | 15,073 | CC-BY-4.0 (images and annotations) |
| squad2 | 13,218 | CC-BY-SA-4.0 | | bool_abstain | 11,232 | CC-BY-4.0 annotations; COCO images |
| nvidia_aegis | 10,232 | CC-BY-4.0 | | textvqa | 11,423 | CC-BY-4.0 annotations; Open Images photos (CC-BY-2.0) |
| banking77 | 10,069 | CC-BY-4.0 | | tallyqa | 11,126 | Apache-2.0; COCO and Visual Genome images |
| abo (listing text) | 9,986 | CC-BY-4.0 | | unkvqa | 9,075 | Apache-2.0 annotations; COCO images |
| openai_moderation | 9,274 | MIT | | vizwiz_quality | 7,870 | CC-BY-4.0 (VizWiz grant) |
| boolq | 7,952 | CC-BY-SA-3.0 | | defects | 7,459 | CC-BY-4.0 (VisA, DAGM 2007); CC-BY-SA-4.0 (BTAD rows) |
| | | | | vsr | 7,033 | Apache-2.0; COCO images |
| | | | | dude | 5,830 | CC-BY-4.0 dataset card; per-document licences unverified |
| | | | | aokvqa | 5,564 | Apache-2.0 annotations; COCO images |
| | | | | koniq | 5,563 | CC-BY-4.0; per-image YFCC100M terms unverified |
| | | | | marqo_gs | 5,563 | Apache-2.0 card; image rights unverified |
| | | | | sqid_esci | 5,563 | Apache-2.0 / MIT; image rights not granted |

For 16 of the 21 image sources the recorded licence covers the annotations only and the images stay under their upstream terms; the
photos are not redistributed. For abo, vizwiz, vizwiz_quality and defects the grant covers the images too; dude is partly verified.
Machine-readable receipts: `reports/v1.1-datasets/mixture-audit-v1.1.json` and `image-policy-a.json`.

### Stage 2: new photo sources labelled by the 9B, then photo-vs-record and two-photo pairs (2B; part of the 4B's first run)

The calibrated 9B (after stage 1) labelled candidate decisions on sources the models had not seen. A label was kept only when the
answer was the same under two option orders and the top probability was at least 0.6 (or `unknown` at least 0.5).

| Pass | Labelled | Kept | Keep rate |
|---|---:|---:|---:|
| New photo and text sources | 366,967 | 271,036 | 73.9% |
| PD12M photos | 179,997 | 144,871 | 80.5% |
| Photo-vs-record and natural pairs | 70,000 | 59,398 | 84.9% |
| **Total** | **616,964** | **475,305** | **77.0%** |

New photo sources (rows kept, all partitions): PD12M 144,871 (CC0-1.0); Wikimedia Commons 36,639 (CC-BY-4.0 23,283, CC-BY-3.0 12,302,
CC0-1.0 1,054); Open Images 44,851 (CC-BY-2.0). New text sources: StackExchange 74,071 decisions (CC-BY-SA-4.0), Wikipedia paragraphs
62,819 (CC-BY-SA-4.0), support and review texts 52,656 (CC-BY-4.0, CC-BY-SA-4.0, MIT).

Photo-vs-record and two-photo decisions: 71,630 in total. 45,999 check a photo against a written record (teacher-labelled);
8,469 compare two natural photos (teacher-labelled); 17,162 compare a reference photo with an edited copy, labelled by construction.

For the 2B, stage-2 training blended each target 0.5 : 0.5 with the untuned 2B's own distribution, to limit forgetting.

### The 4B's first run

The 4B was trained once on a combined mixture instead of stages 1 and 2 in sequence: 866,854 training decisions (504,000 from stage 1
with their original labels; 296,482 stage-2 teacher decisions with `unknown` targets capped at 15%; 66,372 photo-vs-record and pair
decisions). 17.15% of its training targets are `unknown`. No base-model blend.

### Stage 3: hard typed questions (all three models)

Two rounds. Documents and questions were written by open-weight teacher models and kept only when independent answerers agreed.

| | Round 1 | Round 2 |
|---|---|---|
| Writer | Qwen3.6-27B @6a9e13bd | Qwen3.6-27B @6a9e13bd (reasoning on for hard families) |
| Answerers | Qwen3.6-27B (reasoning), gpt-oss-20b @6cee5e81 | the two, plus Qwen3.6-35B-A3B @995ad96e (reasoning) |
| Keep rule | both answerers agree with the intended answer | all three agree |
| Documents written | 4,462 | 2,699 |
| Questions / kept | 13,386 / 9,368 | 8,097 / 4,852 |
| Dropped | 4,013 disagreement, 5 JevBench overlap | 1,883 disagreement, 1,362 near-duplicates, 0 JevBench overlap |
| Human-written reasoning items added | 8,532 from 10 licensed sets (ARC-Challenge, CommonsenseQA, CosmosQA, DROP, GSM8K, QASC, StrategyQA, TruthfulQA, WANLI; MMLU-Pro 800 for evaluation only), at most 800 training rows per set plus up to 133 upstream-test rows | none new |
| Training records | 14,112 (of 17,898) | 7,812: 3,598 new + 4,214 replayed from round 1 |
| Share of `unknown` targets | 2.6% of kept teacher questions | 3.1% of new questions |

Round-2 held-out calibration fold: 622 records from three domains kept out of training (telecom, hospitality, nonprofit grants).
Stage 3 uses hard labels only.

### Stage 4 (phase 2c): soft targets, then a weight-space average (all three models)

One round, continuing from the stage-3 adapters. Targets are probability distributions rather than hard labels, and the shipped adapter
is an average of two adapters, not a single checkpoint.

| | Stage 4 |
|---|---|
| Writer | Qwen3.6-27B @6a9e13bd (judge-style and unknown-family documents); programmatic generators for exact-answer families |
| Answerer | Qwen3.6-35B-A3B @995ad96e (reasoning), returning a full probability distribution over the options plus `unknown` and a short rationale; a question is kept when the argmax matches the intended answer |
| Training rows | 39,515: the stage-3 teacher questions relabelled with the 35B's distributions; 9,880 new hard, judge-style and programmatic questions; 10,570 rows of `caiovicentino1/eikos-decisions` (CC-BY-4.0, strict slice: programmatic and human-annotated rows only, nothing written, labelled or selected by an API model, `docs/eikos-decisions-usage.md`); 5,000 replayed image decisions |
| Share of `unknown` targets | 4.9% |
| Objective | soft cross-entropy against the target distribution (`--soft-targets --soft-weight 1.0`), plus an auxiliary language-model loss on a rationale of at most 192 tokens after the decision position (`--rationale-weight 0.3 --rationale-max-tokens 192`), with option-order shuffling at training time (`--permute-options`) |
| Epochs, peak LR, batch | 2, 2e-5, 40 examples × 2 micro-batches per GPU on 4×H100 |
| Steps (kept / total) | 2B 350 / 620; 4B 260 / 747; 9B 510 / 1,018 |
| Checkpoint selection | best mean accuracy over the phase-2c dev set (600 rows) and the 29-item judge dev set |

**Why an average and not the checkpoint.** The stage-4 checkpoint on its own gains on JevBench hard (4B 71.2 raw, from 67.6) but loses
photo-plus-record items on ImajevBench (4B 79.6, from 82.4, failing the release gate below); the 9B checkpoint passes every gate but scores
below its stage-3 adapter on hard (66.7 raw, from 68.5). The element-wise 50/50 average of the stage-3 adapter and the stage-4 best
checkpoint (LoRA A and B and the readout) keeps the image scores and most of the hard-question gain. Sweep on the 4B, weight of the
stage-4 checkpoint 0 / 0.5 / 0.75 / 1: hard 67.6 / 69.4 / 69.4 / 71.2 raw, ImajevBench 82.4 / 82.4 / 80.6 / 79.6. The averages ship for all
three sizes (`reports/decision-p2c/README.md`).

**Release gates** (all pass for all three averages; `results/imajev-1.0/imajev-<size>/gates.json`): ImajevBench public accuracy at least
the stage-3 adapter's minus 1 point; visual and joint item counts within 2 of the stage-3 adapter; state and pairs probes within 2 points;
correct-unknown rate on the 14 unknown-gold rows of the stage-3 test split not below the stage-3 adapter (14/14 for the 4B and 9B, 12/14
for the 2B); false abstention on the 421 answerable rows at most 2 points above it (0.24% / 0.24% / 1.19% for the 4B / 9B / 2B);
irrelevance panel within 2 points (80.2 / 84.0 / 68.9).

**Calibration.** One temperature per size, fitted by negative log-likelihood on the 150-item JevBench-style dev set (section 5) with the
average's own logits: 2B 1.646, 4B 1.717, 9B 1.748, unknown offsets 0. The stage-4 calibration fold gives T ≈ 1 (the soft-target model
is calibrated on its own distribution) and does nothing on JevBench hard, so it is not used. Fitting the same 150 items for expected
calibration error instead gives a larger temperature (4B 2.05, 2B 2.20) that lowers the hard-tier ECE (4B 0.116 → 0.091, 2B 0.123 → 0.090)
but raises the ECE on the original and easy tiers and the pooled ECE over all 231 public items (4B 0.026 → 0.048, 2B 0.025 → 0.075), which
is what the official board's calibration axis scores; the likelihood fit ships and the ECE fit is kept as an alternative file. Served
with 4 rotations and the shipped temperature: JevBench hard 60.4 / 70.3 / 69.4 with hard-tier ECE 0.123 / 0.116 / 0.092 and pooled public
ECE 0.025 / 0.026 / 0.050 for the 2B / 4B / 9B.

**Compute.** One 8×H100 pod for 6 h 20 min (data generation, the three training runs, all evaluations, the averages, the hidden-split
runs), about $177. The 4B run hit an out-of-memory error at step 625 and resumed in a fresh process at the same budget.

No training data comes from JevBench, from Jev, or from any paid API.

## 3. Training

All stages used the same trainer (`scripts/train_decision_lora_torch.py`, PyTorch and PEFT, 4 GPUs):

- **Objective:** cross-entropy on float32 readout logits over the listed options, no label smoothing. Records that carry a target
  distribution (teacher or source soft labels) use soft cross-entropy; stage 4 adds the rationale loss and option permutation of section 2.
- **Optimizer:** AdamW, weight decay 0, betas (0.9, 0.999).
- **Schedule:** linear warm-up, then cosine decay to 10% of the peak rate.
- **Other settings:**
  - gradient clipping 1.0, gradients averaged per example;
  - seed 0;
  - each step is 2 micro-batches per GPU × 4 GPUs, each micro-batch at most 40 examples and capped by a padded-token budget;
  - gradient checkpointing only for the 9B in stage 3.
- **Checkpoint selection:** stages 1 and the first 2B run used minimum dev loss; later stages kept the checkpoint with the best mean
  accuracy over two dev sets (named in section 5); stage 4 then averaged the kept checkpoint 50/50 with its starting adapter.

| Model | Stage | Started from | Epochs | Peak LR | Steps (kept / total) | Hardware | Optimizer-step time |
|---|---|---|---|---|---|---|---|
| imajev-2b | initial run (594,214 records, earlier image mixture) | Qwen3.5-2B | 1 | 2e-4 | 2,614 / 2,614 | 4×H100 | 1.3 h |
| imajev-2b | 1 | initial run | 1 | 1e-4 | 2,495 / 2,495 | 4×H100 | 1.3 h |
| imajev-2b | 2, first pass | stage 1 | 0.4 | 5e-5 | 1,100 / 1,866 | 4×H100 | 1.0 h |
| imajev-2b | 2, pairs pass | first pass | 1 | 5e-5 | 1,361 / 1,361 (chosen by hand on probe results) | 4×H100 | not recorded |
| imajev-2b | 3, round 1 | stage 2 | 2 | 3e-5 | 250 / 256 | 4×H200 | 8 min |
| imajev-2b | 3, round 2 | round 1 | 2 | 2e-5 | 80 / 220 | 4×H100 | 9 min |
| imajev-2b | 4 (phase 2c) | stage 3, round 2 | 2 | 2e-5 | 350 / 620 | 4×H100 | 49 min wall |
| imajev-4b | first run | Qwen3.5-4B | 0.5 | 1.5e-4 | 1,900 / 2,595 | 4×H200 | 1.9 h (2 h 06 min wall) |
| imajev-4b | 3, round 1 | first run | 2 | 3e-5 | 250 / 303 | 4×H200 | 12 min |
| imajev-4b | 3, round 2 | round 1 | 2 | 2e-5 | 266 / 266 | 4×H100 | 22 min (27 min wall) |
| imajev-4b | 4 (phase 2c) | stage 3, round 2 | 2 | 2e-5 | 260 / 747 | 4×H100 | 1 h 26 min wall, incl. one out-of-memory resume |
| imajev-9b | 1 | Qwen3.5-9B | 1 | 2e-4 | 3,100 / 3,508 | 4×H200 | 2.6 h |
| imajev-9b | 3, round 1 | stage 1 | 2 | 3e-5 | 300 / 404 | 4×H200 | 22 min |
| imajev-9b | 3, round 2 | round 1 | 2 | 2e-5 | 260 / 365 | 4×H100 | 20 min (27 min wall) |
| imajev-9b | 4 (phase 2c) | stage 3, round 2 | 2 | 2e-5 | 510 / 1,018 | 4×H100 | 1 h 39 min wall |

Known incidents, none of which skipped data: the 9B's stage 1 crashed eight times in its first 980 steps (a periodic diagnostic timer
in the trainer; removed) and resumed from 20-step checkpoints; the 2B's initial and stage-1 runs had similar resumes.

## 4. Compute and cost

**About $676 of rented GPU time in total** (RunPod, 17 pods, 21 to 24 September 2026), every run included: data labelling and
generation, all training, all evaluations, the competitor benchmark runs and aborted pods. $499.07 of it through stage 3, and about $177
for stage 4 (6 h 20 min at $27.92/h).

| Use | Hardware | Cost |
|---|---|---:|
| 9B stage 1 | 4×H200 | $71.67 |
| 2B stage 2 (labelling, blending, training) | 4×H100 | $95.43 |
| 4B first run, stage-3 round-1 data generation, round-1 training of all three models, evaluations | 4×H200 | $129.36 |
| Stage-3 round-2 data generation, round-2 training, a frozen 35B reference check, competitor benchmark runs | 7×H100 | $99.88 |
| 2B initial run and stage 1 with their evaluations (three pods) | 4×H100 | $92.69 |
| ImajevBench base-model and generation rows | 1×H100 | $4.91 |
| Release calibration check | 1×H100 | $2.32 |
| Short-lived and aborted pods | various | $2.81 |
| Stage 4: data generation, training of all three models, evaluations, the averages, hidden-split runs | 8×H100 | ≈ $177 |

## 5. Evaluation sets

| Set | Size | What it is | How it is kept out of training |
|---|---|---|---|
| ImajevBench v2.0-lite test | 279 items, 89 evidence clusters | photo, state and photo+state typed decisions, 21 with an `unknown` answer; AI-generated images | built after training; after-the-fact check found no shared images or questions (byte hashes; no perceptual-hash check) |
| JevBench public splits | 111 hard / 72 original / 48 easy | text-only typed decisions | 8-gram overlap lint on all teacher data (5 questions dropped in round 1, 0 in round 2); re-check found no row sharing 2 or more 8-grams |
| MMLU-1000 | 1,000 questions, also sent with an unrelated photo | official MMLU test split, 17 to 18 per subject | registered as never-train; audit gate fails on any MMLU row outside test; 589 MMLU-Pro items matching it were removed from stage 3 |
| Irrelevance panel | 2,823 | the 2,000 MMLU rows plus 489 ABO and 334 VizWiz controls that score each donor photo with its own relevant question | donors from the test partition; no panel image in training |
| Typed-decisions test | 2,000 decisions (400 records × 5) | synthetic workflow decisions (LocalLLaMA/typed-decisions test split); gold is agreement with that dataset's teacher model | evaluation only (the upstream repo has no licence file) |
| SST-5 | 2,210 | official Stanford Sentiment Treebank test sentences | evaluation only; no licence grant found, not redistributed |
| Reasoning dev set | 6,240 decisions | 6,000 from the typed-decisions train split plus 240 hand-written items in 8 families | not trained on, but used to pick checkpoints in stage 2 and stage-3 round 1, so its scores are not independent of selection |
| JevBench-style dev set | 150 | template-generated items in 10 families, none from JevBench (one shared stock contract phrase in 8 items) | not trained on; used to pick round-2 checkpoints and to fit the shipped calibration temperatures |
| Phase-2c dev set / judge dev set | 600 / 29 | held-out phase-2c rows / judge-style items | not trained on; used to pick the stage-4 checkpoints |
| Phase-2c test | 2,122 | held-out phase-2c rows | not trained on (2B 75.6, 4B 83.3, 9B 85.5 for the stage-4 checkpoints) |
| ImajevBench private-1 | 202 items | hidden companion split of ImajevBench, aggregates only | never published item by item; 74.3 / 84.2 / 84.7% for the shipped 2B / 4B / 9B |
| State probe / pairs probe | 200 / 60 | photo-vs-record and two-photo questions built from the same templates as training | 10 of the 200 state-probe items reuse photos seen elsewhere in training; the pairs probe shares no images |
| Held-out photo sources | 4,989 (plus 177 real two-photo pairs) | MM-UPD, TUBench, CountQA, LIVE in the Wild, Fashionpedia, MuirBench | sources never used in training; no shared images |

## 6. Checks behind the data

- **JevBench:** every stage-3 teacher question was linted against the public JevBench files (62,915 reference 8-grams); a question
  sharing two or more 8-grams was dropped (5 in round 1, 0 in round 2). A re-run over both training manifests found no row at the
  threshold.
- **Evaluation panels:** an audit gate fails any build that places MMLU, SST-5, typed-decisions test or any upstream test row outside the
  test partition. A re-check over the training partitions of all six training manifests found no question or long state shared with
  MMLU-1000, typed-decisions test, SST-5 or the reasoning dev set.
- **ImajevBench:** no mechanism excludes it by construction; an after-the-fact comparison of all 923 benchmark images and every
  record's text against all training partitions found no overlap. Training images were re-encoded, so a byte-hash check would miss the
  same photo saved differently; no perceptual-hash check was run.
- **Stage 4:** the new and relabelled questions come from the same open-weight writer and answerer models as stage 3 and nothing from
  JevBench items or Jev outputs was used; the Eikos slice keeps only rows that no API model wrote, labelled or selected, under CC-BY-4.0
  with the attribution in `docs/eikos-decisions-usage.md`.
- **Licences:** every training source is admitted with a licence receipt; 16 image sources are admitted for their annotations only, with
  photos under upstream terms and not redistributed.
