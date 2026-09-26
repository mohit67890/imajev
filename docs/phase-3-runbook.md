# Phase 3 pod runbook (imajev-4b, the last training run)

Plan: `docs/phase-3-plan.md` (Stage 3 flags, Stage 4 eval / gates / pick / per-type calibration, Stage 5 round 2, the pod
sequence in "Everything in this phase", the 4 pilot lanes, the model-side engineering results). Streamed mining + Azure teacher:
`reports/phase3/streaming-runbook.md` (another agent's; this runbook only covers the pods' scripts).

Write every time you report in **IST (UTC+5:30)**. The pod scripts already log `HH:MM:SS IST`; RunPod and `date -u` show UTC.

| File | Where it runs | What it does |
|---|---|---|
| `scripts/p3/transfer.py` | Mac (build, push, delete), pods (fetch) | bundles with a hard benchmark check; GCS `gs://<bucket>/phase3/` (this run) or a private HF dataset repo |
| `cloud/p3/bootstrap_mine.sh` | small mining pod | setup + one `scripts/p3/mine.py` worker per GPU; `MINE_DONE` / `MINE_FAILED` |
| `cloud/p3/bootstrap_train.sh` | 8×H100 pod | setup (venv, Qwen3.5-4B, shipped adapter, bundles, public eval sets, DecisionBench subset), then starts the run |
| `cloud/p3/pod_run_train.sh` | 8×H100 pod | parity → pilot → train r1 → selection → round 2 → eval → pick (+ soup fallback) → table → `ALL_DONE` / `FAILED` |
| `scripts/p3/eval_checkpoint.py` | 8×H100 pod, one per GPU | all panels for one checkpoint (see "Eval lane") |
| `scripts/p3/make_decisionbench_subset.py` | 8×H100 pod (bootstrap) | the 3k stratified DecisionBench subset, tracking only |
| `scripts/p3/parity_readout256.py --backend torch` | 8×H100 pod | Gate 0 on the CUDA torch path |
| `cloud/p3/{pilot_decision,select_checkpoint,prep_manifests,plan_steps,ckpt_watch,round2_manifest,soup,final_table}.py` | 8×H100 pod | the rules, as small tested programs |
| `cloud/p3/lane_resume.py` | 8×H100 pod | which pilot lane the full run continues, exact / inexact / fallback (section "Lane resume") |
| `cloud/p3/eval_lane.py` | 8×H100 pod | the rolling eval lane, **only with `EVAL_DURING_TRAIN=1` (off by default)** |

Tests (no GPU, no network):
- `.venv/bin/python -m pytest tests/test_p3_pod.py -q`: 60 tests, ~12 s.
- `.venv/bin/python -m pytest tests/test_p3_resume.py -q`: 5 tests, ~50 s. They run the real trainer on the CPU with a tiny stand-in
  model, with torchrun and gloo for 2 and 4 ranks.

**Nothing in these scripts terminates a pod.** The owner or the lead terminates it after the pull, and deletes the private HF
repos or bucket objects afterwards.

---

## 0. Rules this runbook enforces

- **Nothing large goes from the Mac to a pod.** Bundles go Mac → GCS (`gs://<bucket>/phase3/`, owner decision
  2026-09-25) → pod. Only the small scripts (`bootstrap_*.sh`, `transfer.py`, `p3.env`) and the short-lived reader key go over scp.
  Both bootstraps default to `BACKEND=gcs`, that bucket and prefix `phase3`.
- **No benchmark text in any training bundle.** `transfer.py build` fails (exit 2, `BUNDLE_CHECK_FAILED`, nothing written) if a
  file is on a benchmark path (`data/p3/decontam-sources/`, JevBench datasets, DecisionBench, fast-decisions, ImajevBench incl.
  `private-1`, `reports/benchmarks/`, the authored JevBench-style dev set), has the sha256 of a pinned benchmark file
  (`cloud/p3/benchmark_pins.json`), or contains a string of 80 or more characters equal to a benchmark string. Strings are
  whitespace- and case-normalised, and JSON nested inside strings is parsed. The index is built from the local benchmark copies:
  ~142k strings, 465 file hashes. Measured on 2026-09-25: the code tree (648 files) and all of `data/p3/candidates-clean` (59k
  files, 8.8 GB, 66 s) pass, and a planted JevBench state or DecisionBench state fails.
- **The one exception is the gatekit.** The image gate needs the ImajevBench test gold, which is not public, and the calibration
  fallback needs the authored dev set. The `gatekit` preset may contain only
  `bench/records/records-eval.jsonl`, `bench/assets/*` and `manifests/decision-p2b-jevstyle-dev.jsonl`. Its check still fails on
  decontam-sources, `private-1`, JevBench, DecisionBench and fast-decisions content. It unpacks to `bench`, outside
  the training tree, and `fetch` refuses to extract it without `--gatekit`.
- **Public benchmarks are fetched on the pod** from pinned public origins and never bundled:
  - JevBench: git @ `2fa63fa`, public splits checked against the sha256 pins;
  - fast-decisions: HF, compared with the pins; the upstream revision is unpinned, so a difference is logged, not fatal;
  - DecisionBench: HF @ `b7c8107e`, parquet sha256 pinned.
- **Tokens are never written to files in the tree, logs or command lines.**
  - On the Mac, HF uses the cached login (`.venv/bin/hf auth login`).
  - On a pod, the token is scp'd to `hf-token`, exported as `HF_TOKEN` by the bootstrap (never echoed), read by
    `transfer.py --token-env HF_TOKEN`, and the file is deleted as soon as the downloads finish.
  - Use a **fine-grained read-only token** scoped to the two or three private repos, and revoke it after the pull.
- **Never wait on `ssh host pgrep ...`.** It matches its own remote shell and never fires. Poll the marker files instead
  (commands below).

## 1. Mac: build and push the bundles (no pod yet)

Inputs come from the other agents. Check each exists before building:
- the pool shards `data/p3/pool/shards/*.jsonl` (`scripts/p3/assemble_pool.py`);
- the Stage-3 trainer manifests: `scripts/p3/build_manifest.py ... --publish decision-p3` writes exactly the names below into
  `data/manifests/` (checked end to end in `reports/phase3/dry-run.md`), so `p3.env` needs no manifest overrides.

| env var (`p3.env`) | default manifest (`data/manifests/<name>.jsonl`) | content |
|---|---|---|
| `P3_MANIFEST` | `decision-p3` | train + dev partitions of the Stage-3 mixture (the 256-code lane); 255-option rows allowed (the pod splits them off for the 255-code lanes) |
| – | `decision-p3-lane255` | the builder's size-matched 255-code lane; `prep_manifests.py` uses it as `decision-p3-r255` (bundle it) |
| `P3_LABELLED` | `decision-p3-labelled` (falls back to `P3_MANIFEST`) | every verified teacher-labelled item (train partition), the round-2 source |
| `P3_HELDOUT_FRESH` | `decision-p3-heldout-fresh` | fresh-seed held-out half incl. GUI (the target), no 255-option rows |
| – | `decision-p3-heldout-fresh-large-choice` | fresh 255-option items (reported for the 256 lanes only) |
| `P3_HELDOUT_FLAGGED` | `decision-p3-heldout-flagged` | flagged held-out half (per-type T fit, reported) |
| `P3_HUMAN_DEV` | `decision-p3-human-dev` | the ~100 human-verified + Kimi-agreed dev slice |
| `P3_HELDOUT_PARTITION` | `test` | partition name of the rows in the three held-out manifests |

```bash
cd ""
.venv/bin/python -m pytest tests/test_p3_pod.py -q

# mining pod bundle: code + pool shards + the images they reference
.venv/bin/python scripts/p3/transfer.py build --preset mine --name p3-mine --candidates 'data/p3/pool/shards/*.jsonl'

# training pod bundle: code + Stage-3 manifests + held-out + gate panels (+ all referenced images) + the pool (round-2 re-mine)
.venv/bin/python scripts/p3/transfer.py build --preset train --name p3-train \
  --manifest decision-p3 --manifest decision-p3-lane255 --manifest decision-p3-labelled --manifest decision-p3-heldout-fresh-large-choice \
  --manifest decision-p3-heldout-fresh --manifest decision-p3-heldout-flagged --manifest decision-p3-human-dev \
  --manifest decision-p2b --manifest decision-v2-state-probe --manifest decision-v2-pairs-probe --manifest decision-v1.1-irrelevance-test \
  --candidates 'data/p3/pool/shards/*.jsonl'

# gate kit (ImajevBench eval gold + assets, authored dev set): the only benchmark bundle
.venv/bin/python scripts/p3/transfer.py build --preset gatekit --name p3-gatekit
```

- Add `--list` to any build to see the file list and the check result without writing anything.
- Output goes to `.cache/p3-bundles/<name>.tgz` (split into 8 GB parts when larger) plus `<name>.manifest.json` (file list,
  sha256, check report).
- Build time is about 1 minute per 10 GB of text to check, plus compression.
- The images and the pool text make `p3-train` roughly 9–12 GB.

**Push. Backend (a): private HF dataset repo (fallback only; this run uses GCS, backend (b)).** The owner's HF login is already on the Mac.
```bash
.venv/bin/python scripts/p3/transfer.py push --backend hf --name p3-mine      # prints e.g. mohit67890/imajev-p3-mine-20260926093000
.venv/bin/python scripts/p3/transfer.py push --backend hf --name p3-train
.venv/bin/python scripts/p3/transfer.py push --backend hf --name p3-gatekit
```
- Each repo is created **private**, and the push refuses to upload if the repo reports public.
- Upload speed is the owner's uplink. At ~5–10 MB/s, 12 GB takes 20–40 min, so start pushing well before the pod is rented.
- Record the three repo ids for `p3.env`.

**Push. Backend (b): GCS (this run).** The bucket is `gs://<bucket>/`, prefix `phase3` (the default). The owner is logged in
(`gcloud auth list` to confirm).
```bash
for b in p3-mine p3-train p3-gatekit; do .venv/bin/python scripts/p3/transfer.py push --backend gcs --bucket gs://<bucket>/ --name $b; done
```
Create a short-lived key for the bucket reader service account, as in phase 2c (`reader-key.json`), and revoke it after the
bootstraps finish.

## 2. Mining pod (Stage 1, streamed; 4 GPUs A100/L40S, ~$5–8/h)

- State the price, then create the pod: 4× A100 80GB or 4× L40S, secure cloud, image `runpod/pytorch` cu128 / torch 2.8, 150 GB
  disk, ssh alias `imajev-mine`.
- The pod does the setup (~15–20 min) and runs one `mine.py` worker per GPU over all shards. Workers claim shards with flock.
- Mining writes to `data/p3/mine`, which is where `stream_coordinator.py --remote-dir` pulls from.
- For the Mac-side coordinator and Azure teacher, follow `reports/phase3/LIVE-RUNBOOK.md` (steps 4–13); rules in `reports/phase3/streaming-runbook.md`.

```bash
cat > /tmp/p3.env <<'EOF'
BACKEND=gcs
GCS_BUCKET=gs://<bucket>/
GCS_PREFIX=phase3
MINE_BUNDLE=p3-mine
# MINE_ARGS="--batch-tokens 32768"      # e.g. for L40S with long items (mine.py flag)
EOF
scp cloud/p3/bootstrap_mine.sh scripts/p3/transfer.py /tmp/p3.env imajev-mine:
scp ~/.config/imajev/reader-key.json imajev-mine:reader-key.json      # short-lived; revoke after the fetch
ssh imajev-mine 'cd /workspace && setsid nohup bash bootstrap_mine.sh > mine-bootstrap.log 2>&1 < /dev/null & echo launched'
```
The scripts read exactly `p3.env` (the scp above puts it there).

Watch it (never `pgrep`):
```bash
ssh imajev-mine 'tail -n 5 mine-bootstrap.log; ls p3-mine/ | grep -E "DONE|FAILED"; ls data/p3/mine/*.done 2>/dev/null | wc -l'
ssh imajev-mine 'cat data/p3/mine/progress/*.json'          # per-worker heartbeat + counters
```
- **Done** means `p3-mine/MINE_DONE` exists: every worker exited 0 and every shard has its `.done` file.
  `MINE_FAILED` names the failed workers.
- After the coordinator's final pull, fetch `p3-mine-results.tgz` with scp or `rsync -az` (it holds the mining output
  and the logs).
- Then the owner or lead terminates the pod.
- Expected time is ~2–2.5 h including setup (the plan assumes ~100k items/h on 3–4 GPUs), so about **$12–20**.

## 3. Training pod (Stages 3–5, 8×H100 SXM, ~$21.5–28/h)

**Create:** state the price, then create the pod: 8× H100 SXM, secure cloud, image `runpod/pytorch` cu128 / torch 2.8, ≥ 250 GB
disk (the 4B, the bundles, ~12 checkpoint snapshots, eval outputs), ssh alias `imajev-p3`.

```bash
cat > /tmp/p3.env <<'EOF'
BACKEND=gcs
GCS_BUCKET=gs://<bucket>/
GCS_PREFIX=phase3
TRAIN_BUNDLE=p3-train
GATEKIT_BUNDLE=p3-gatekit
# manifest names if the builder used others (section 1 table), e.g. P3_MANIFEST=decision-p3
# optional knobs (defaults in cloud/p3/pod_run_train.sh): PILOT_STEPS=100 (PILOT_ACC is derived: 8 GPUs x ACC / 2 = 4) BATCH=40
#   TOKEN_BUDGET=16384 EPOCHS=2 LR=2e-5 CKPT_FRAC=0.25 (default: ~4 r1 snapshots) R2_EPOCHS=1 R2_LR=1e-5 R2_MAX_NEW=20000 R2_REPLAY=1.0
#   SOUP_WEIGHTS="0.75 0.5 0.25" CKPTING=1 RUN_AFTER=1 RESUME_LANE=1 (continue the pilot winner) EVAL_DURING_TRAIN=0 (see "Eval during training")
EOF
scp cloud/p3/bootstrap_train.sh scripts/p3/transfer.py /tmp/p3.env imajev-p3:
scp ~/.config/imajev/reader-key.json imajev-p3:reader-key.json      # short-lived; revoke after BOOTSTRAP_DONE
ssh imajev-p3 'cd /workspace && setsid nohup bash bootstrap_train.sh > bootstrap.log 2>&1 < /dev/null & echo launched'
```
- For the HF fallback instead: `BACKEND=hf TRAIN_REPO=... GATEKIT_REPO=...` and scp a fine-grained read token to `hf-token`.
- With `RUN_AFTER=1` (the default), the bootstrap starts `pod_run_train.sh` itself after `BOOTSTRAP_DONE`.
- To start it by hand:
  `ssh imajev-p3 'cd /workspace && setsid nohup bash cloud/p3/pod_run_train.sh >> p3/run/run.log 2>&1 < /dev/null &'`.
- Never run two instances. Rerunning after a crash resumes: stages with a `.DONE` marker are skipped, the trainer resumes from
  `last`, and eval panels already on disk are reused.

**Dashboard** (markers and stage lines only):
```bash
ssh imajev-p3 'tail -n 3 bootstrap.log; grep -hE " stage | done |winners|lane resume|RESUME_FALLBACK|pick|parity|pilot|FAILED|NOT_SHIPPABLE|SHIPPABLE|ALL_DONE|WARNING" p3/run/run.log | tail -n 40'
ssh imajev-p3 'ls p3/run/*.DONE p3/run/FAILED 2>/dev/null; tail -n 1 p3/run/run.log'
ssh imajev-p3 'tail -n 2 p3/run/r1/train.log | cut -c1-300'          # step / of / loss / examples per second
```
Waiters poll `tail -n 1 p3/run/run.log` for `ALL_DONE` or `FAILED`.

### Sequence and expected durations (8×H100, 2 epochs)

| # | Stage | GPUs | Expected | Marker / output |
|---|---|---|---|---|
| – | bootstrap | – | 20–35 min (pip, 9 GB model, ~12 GB of bundles from GCS, JevBench clone, DecisionBench subset) | `BOOTSTRAP_DONE` in `bootstrap.log` |
| 0–1 | checks + prep | CPU | 2–5 min | `checks.DONE`, `prep.DONE`, `manifests.json` |
| 2 | CUDA parity 255 vs 256 | 6 | ~10 min, overlapping lanes 1–2 | `parity/summary.json`, `parity/rc` (0 = PASS) |
| 3 | pilot, 4 lanes × 2 GPUs × **100 full-run steps** (8 micro-batches per step, as the full run): baseline / +ordinal / +256 codes / **+rank64** (2026-09-26: replaces "+both") | 0–7 | 40–60 min (unchanged: each lane trains the same 800 micro-batches as the old 200 half-size steps; rank64 starts at once on GPUs 4–5, codes256 after the parity check on 6–7) | `pilot/<lane>/train/{log.jsonl,last}`, **`pilot-decision.json`** (winners; `lane` = the lane to continue, or null when two or more factors win) |
| 4 | train r1 (2 epochs, all 8), **continuing the winning lane at step 100** | 0–7 | 3.4–4.9 h (~3,300 steps; the 100 banked steps and the skipped step-0 dev pass save ~6–12 min) | `lane-resume.json`, `r1/train/resumed_from.json`, `ckpts/r1-s*` every ~25% (`CKPT_FRAC=0.25`), `train-r1.DONE` |
| 5 | selection panels for the r1 snapshots | 0–7 | 10–20 min | `eval/<ckpt>/selection.json`, `r1-pick.json` |
| 6 | round 2: re-mine the labelled pool with the r1 pick + 1 short epoch | 0–7 | 1–1.5 h | `r2/mine/*.flagged.jsonl`, `r2/manifest-report.json`, `ckpts/r2-s*`, `round2.DONE` |
| 7 | eval: 4 r1 + 3 r2 snapshots + shipped 1.0 = 8 checkpoints on 8 GPUs, one wave | 0–7 | ~1–1.2 h (one checkpoint's panels, ~50–70 min) | `eval/<ckpt>/{summary,gates,selection}.json`, `tracking-only/<ckpt>/decisionbench.json` |
| 8 | pick (+ soup fallback) | 0–7 | 5 min (+ ~1 h if the soup fallback runs) | **`pick.json`** |
| 9 | final table + pack | CPU | 5 min | **`final-comparison.md`**, `picked-adapter/`, `p3-train-results.tgz`, `ALL_DONE` |

- Total is **~7.8–10.8 h** (was ~8–11 h), which at $21.5–28/h is **~$170–300**. Add ~1 h (~$25) if the soup fallback runs.
- The saving is small, and it is the most these two changes can give:
  - Only 1 of the 4 pilot lanes survives. The winner's 100 steps are ~3% of r1, so continuing it saves ~8 min of r1, plus the
    full run's step-0 dev pass (~1–2 min).
  - The eval lane is off (next section), because it would make the run longer.
- The plan's ceiling is $300 for Stages 1–4. The mining pod and Azure are separate.
- The largest uncertainty is stage 4. The trainer prints `of` (the total steps) on its step lines, and `r1/plan` in the log gives
  the step count before it starts. `resumed_from.json` shows the step the run continued from.

### Eval during training (`EVAL_DURING_TRAIN`, default 0 = off)

`EVAL_DURING_TRAIN=1` trains on 7 GPUs and keeps GPU 7 as a rolling eval lane (`cloud/p3/eval_lane.py`), from train-r1 until
stage 7:
- The lane takes, in order: shipped 1.0, then every intermediate r1 / r2 snapshot, and r1's last snapshot once `sel-r1.DONE` exists.
- It pauses while stage 5 runs. A checkpoint it is evaluating at that moment gets its selection panels from the lane.
- Stage 7 stops it and runs the final queue over what is left, on the other 7 GPUs, while the lane finishes its current checkpoint.
  Checkpoints that failed in the lane, or were cut off when it died, run again.
- The selector rules, the tracking-only DecisionBench path and the gates are the same code as without the lane.

It is off by default because on this pod it costs time and changes the recipe. Estimate (8×H100, 2 epochs):

| | today (off) | lane on |
|---|---|---|
| r1 training | T ≈ 3.4–4.9 h | T × 8/7: **+29–42 min** |
| round 2 (re-mine + short epoch) on 7 GPUs | 1–1.5 h | **+5–10 min** |
| final eval (stage 7) | one wave: 8 checkpoints on 8 GPUs ≈ one checkpoint's panels, 50–70 min | the leftover (r2's snapshots, and r1-last if the lane is still busy) is still one wave: **50–70 min, ~0 saved** |
| net | – | **~35–50 min slower** |

- **Why nothing is saved at the end:**
  - Stage 7 is already a single wave: 8 checkpoints on 8 GPUs, one per GPU.
  - Its length is the eval time of one checkpoint (~1 h), and the last r2 snapshot only exists when training ends. So stage 7
    can never be shorter than one checkpoint's eval, with or without the lane.
  - Even if the last checkpoints' panels were split across GPUs, the lane cannot pay off. GPU time is conserved: 1 of 8 GPUs
    during a T-long training costs T/7 of wall time and buys at most T/7 of eval at the end.
- **Why it would change the recipe:**
  - 7 × `ACC 1` gives 7 micro-batches per step instead of 8: a 12.5% smaller batch and 14% more optimizer steps, for r1 and r2.
  - The pilot resume becomes inexact (8 → 7 per step; ≤ 6 micro-batches repeated, see below).
- If the pod has more GPUs than training can use, or checkpoints outnumber GPUs (e.g. `CKPT_FRAC=0.1`), re-check the table.
  The flag is ready and tested.

### What each stage decides

**Parity (Gate 0 on CUDA)**
- `scripts/p3/parity_readout256.py --backend torch --device cuda` loads the shipped adapter twice, one load after the other:
  first with 255 codes, then with 256.
- It scores JevBench public (231), the authored dev set (150) and 5 synthetic wide-choice questions. Candidate files are
  included when they are present. Logits must be bit-identical.
- A 40-question determinism control tells kernel noise apart from a real difference.
- On FAIL, or when the manifest has no 255-option training rows, the two 256 lanes are **not run** and the run keeps 255 codes.
  The plan's fix, gathering rows before the matmul as MLX does, needs a trainer or engine change outside this run.

**Pilot** (`cloud/p3/pilot_decision.py`; rules in its docstring and tested)
- Lanes: baseline (W 0, 255, r16), +ordinal (W 0.15, 255, r16), +256 (W 0, 256, r16), **+rank64** (W 0, 255, the shipped r16 LoRA
  expanded to r64: trainer `--expand-lora-rank 64`, `scripts/lora_expand.py`; kept A rows / B columns, new A rows with the standard
  LoRA init, new B columns zero, alpha 32 → 128 so alpha/rank stays 2: the output is identical at step 0; tested on the real trainer
  in `tests/test_p3_rank_expand.py`). Every lane logs `--dev-slices` (image-row and text-row dev accuracy).
- Every lane uses:
  - the Stage-3 flags and `--max-length 16384`;
  - the same seed and start adapter;
  - dev = the ≤ 254-option dev of the manifest;
  - dev2 = the ordinal-only dev built as in `reports/phase3/ordinal-loss.md`.
- **Every lane is the full run's first `PILOT_STEPS` (100) steps for its switches.** It has the same manifest, seed, start
  adapter, epochs (so the same step count and learning-rate schedule), batch plan and micro-batches per step: 2 GPUs ×
  `PILOT_ACC` 4 = 8 GPUs × `ACC` 1.
  - The old pilot ran 200 steps of 4 micro-batches; the new one runs 100 steps of 8. Each lane trains the same 800 micro-batches
    on the same examples, so the pilot's cost and data exposure are unchanged.
  - `PILOT_STEPS=200` gives a 2× longer pilot (+~30 min) to bank ~15 min of r1: slower overall.
- Metrics are the mean of the dev passes at steps 75 and 100 (dev every 25). These are the same data positions as the old
  steps 150 and 200.
- Winner rules:
  - **ordinal** wins if score accuracy (dev2) is above baseline, overall dev accuracy is no more than 1.0 pt lower, and dev loss
    is no more than 2% higher;
  - **256** wins on parity + 255-option rows + no harm;
  - **rank64** wins on no harm, no image / text dev slice more than 2.0 pts below baseline, AND clearly better beyond noise
    (dev accuracy ≥ baseline + 1.5 pts with dev loss not higher, or dev loss ≤ 0.97 × baseline with accuracy held); otherwise r16;
  - the three factors are decided **independently** and the full run **combines** the winners. One winner (or none): r1 continues
    that lane (below). Two or more: no lane trained them together, so r1 restarts from the shipped adapter with the combined flags
    (`lane resume: FALLBACK … COMBINED`), which costs the pilot's 100 steps (~3% of r1).
  - r1 gets `--expand-lora-rank 64` when rank64 won; round 2 starts from the r1 pick and takes its rank from the adapter. The soup
    fallback zero-pads the shipped r16 factors to r64 (`cloud/p3/soup.py`); every loader (server, evaluator, `mine.py`: PEFT
    `from_pretrained`) and the MLX converter read the rank from `adapter_config.json`.
- A lane that crashed before step 200 loses.

**Train r1**
- Command: `torchrun --nproc_per_node 8 scripts/train_decision_lora_torch.py --init-adapter <shipped soup50> --epochs 2 --lr 2e-5
  --warmup 10 --soft-targets --soft-weight 1.0 --permute-options --rationale-weight 0.3 --rationale-max-tokens 192
  --max-length 16384 --ordinal-weight <W> --readout-codes <codes> --resume-from <O>/pilot/<winner>/train/last`.
- The last flag continues the winning lane (next block). Without it the command is the old one, which is the fallback.
- Batch settings: `--batch-size 40 --accumulate 1 --token-budget 16384`, gradient checkpointing on.
- `--dev-every` is ~25% of the steps (`CKPT_FRAC=0.25`). `cloud/p3/ckpt_watch.py` copies every dev checkpoint to `ckpts/r1-s<step>`, without the
  optimizer state.

**Lane resume** (`cloud/p3/lane_resume.py`, trainer `--resume-from`, default off in the trainer)
- The winner's `last/` (step 100) holds the adapter, readout, optimizer, scheduler and step. `lane_resume.py` checks:
  - the lane finished (`pilot-<lane>.DONE`);
  - its files are present;
  - its switches match the decision.
  It then prints `--resume-from <lane>/train/last` and records `lane-resume.json`.
- The trainer checks the lane's `config.json` against its own and refuses on any difference except world size, accumulation,
  dev / selection settings and the dev2 set.
- It loads the state, saves its own `last/` at step 100, writes `resumed_from.json`, and continues. It skips the step-0 dev pass.
  Its `best` starts empty, because the lane's used another dev size and rule.
- **Exactness.** The trainer's batch plan (`flat`, before balancing) depends only on the manifest, seed, epochs, token budget,
  batch size and pixels, not on world or accumulation. Balancing only reorders micro-batches inside one optimizer step.
  - After S lane steps, exactly `flat[:S × 8]` has been trained.
  - The 8-GPU run started at step S trains `flat[S × 8:]` in the same step groups, with the same schedule (same step count).
  - So no example is repeated or skipped (tolerance 0), and every optimizer step averages the same examples as an uninterrupted
    run. Floating-point summation order differs, as it does between any two world sizes.
  - `tests/test_p3_resume.py` proves this on the real trainer. A 2-rank × 4 lane resumed on 4 ranks × 1 trains the reference's
    micro-batches step by step, sees every row once in epoch 1, and ends on the reference weights (max difference < 2e-5).
- **Inexact mode** is used only with `EVAL_DURING_TRAIN=1` (7 × 1 = 7 per step vs the lane's 8). It needs `--resume-inexact`.
  - The run starts at `floor(800 / 7) = 114`.
  - 2 micro-batches (≤ per-step − 1 = 6 in general, ≤ 40 examples each) are trained twice. Nothing is skipped.
  - The schedule is re-derived for the 7-GPU step count. Tested as well.
- **Fallback (automatic).** The full run restarts from the shipped adapter exactly as before, and logs
  `lane resume: FALLBACK` or `RESUME_FALLBACK`, in either case:
  - `lane_resume.py` finds anything missing;
  - the resumed trainer fails before its first save (no `resumed_from.json`).
  The fallback wipes `r1/train` and writes `r1/resume-fallback.json`. A crash after a successful resume continues from the run's
  own `last/`, as before. `RESUME_LANE=0` turns the resume off.

**Selection** (`cloud/p3/select_checkpoint.py`)
- The score is the mean of the fresh-seed held-out accuracy and the human-slice accuracy.
- The pool is the checkpoints that pass every gate. Within √2 × SE of the best score, the **latest** checkpoint wins.
- It reads only `selection.json` and `gates.json`. It cannot read `tracking-only/`: its reads refuse such paths, and the test
  instruments `open()` to prove it.
- The round-1 pick for round 2 uses the same score without the gates (the gates have not run yet).

**Round 2** (mandatory)
- The labelled train items of `P3_LABELLED` are split into 32 shards and re-mined by `scripts/p3/mine.py` (8 workers) with the r1
  pick. A 256-code checkpoint is mined through a 255-row copy, which is exact for ≤ 254 options.
- `cloud/p3/round2_manifest.py` takes the flagged rows:
  - never-trained rows first, capped at 20k;
  - refuses any id, parent or group that appears in the held-out or human manifests;
  - adds an equal number of replay rows and the base dev partition.
- One epoch at lr 1e-5 from the r1 pick. Snapshots at ~34%, ~68% and 100%.
- No Azure calls: the labels are cached.
- Fewer than 200 new rows, or a mining failure, → `FAILED` (round 2 is mandatory; fix and rerun, and the finished stages are
  skipped).

**Eval lane** (`scripts/p3/eval_checkpoint.py`, one checkpoint per GPU, ports 8765+gpu)
- **Held-out:** fresh, flagged and human slice (batched evaluator; the max length is raised to 16k in-process).
- **Gate panels:** decision-p2b test (unknown / false abstention), state and pairs probes, irrelevance test, and ImajevBench v2.0-lite
  test. Gates use `cloud/p2c_gates_reference.json` (4b row, unchanged tolerances). The joint target ≥ 99/122 is reported beside
  them.
- **Calibration:** one T per type (choice / noul / score), fitted by NLL on the flagged half. Single pass and rot4 are fitted
  separately; rot4 uses the served rotation path on up to 1,200 flagged rows. A fitted T < 1 ships as 1.0. A type with fewer
  than 30 rows uses the pooled T. The fit is kept only if pooled public JevBench ECE (all tiers, re-tempered offline from the
  raw run) stays ≤ 0.03; otherwise the authored-dev single T is used. Both guards are in `summary.json`.
- **JevBench public:** single raw, single calibrated, rot4 raw, rot4 calibrated.
- **fast-decisions dev:** off-distribution check.
- **DecisionBench 3k subset:** single pass raw, **written only to `p3/tracking-only/<ckpt>/`**.
- **Shipped 1.0:** evaluated the same way, with its own `calibration.json`, as the baseline for every "vs shipped" row.

**Pick + soup fallback**
- If no checkpoint passes every gate, the best-scoring checkpoint is souped with the shipped adapter at W = 0.75 / 0.5 / 0.25
  (weight on the new checkpoint). A 256-row readout keeps row 256 from the new checkpoint.
- The soups are evaluated and picked by the same rule, and `gates.json` records `"soup": true`.
- If nothing passes even then: keep 1.0.

**Final table** (`final-comparison.md` / `.json`)
- The picked checkpoint against the shipped 1.0 on every target row of the plan:
  - hard ≥ 73 single + cal, with rot4 as the fallback;
  - fresh held-out +8;
  - DecisionBench (full-suite equivalent) +2, ordinal +5, reasoning +5;
  - gates without a soup; joint ≥ 99;
  - hard ECE ≤ 0.08; pooled ECE ≤ 0.03.
- The DecisionBench rows are the plan's ship rule, **read by a person here**. No code acts on them.

## 4. Pull, verify, clean up (owner or lead)

```bash
mkdir -p reports/phase3/pod && scp imajev-p3:p3-train-results.tgz reports/phase3/pod/
tar xzf reports/phase3/pod/p3-train-results.tgz -C reports/phase3/pod     # run/ (ckpts, eval, pilot, logs, final-comparison) + tracking-only/
ssh imajev-p3 'cat p3/run/picked-adapter/SHA256SUMS'; shasum -a 256 reports/phase3/pod/run/picked-adapter/*.safetensors
```
Then:
1. Terminate the pod: RunPod delete. Nothing in the scripts does it.
2. Delete the bundles:
   `.venv/bin/python scripts/p3/transfer.py delete --backend gcs --bucket gs://<bucket>/ --name <bundle>` for p3-train and
   p3-gatekit (and p3-mine if it is still there).
3. Revoke the reader key (`cloud/README-4b.md` step 7).
4. Record the times (IST) and the cost in `reports/benchmarks/README.md` and the phase-3 results log.

## 5. Open risks (read before renting)

1. **Manifest names and the id scheme**: `build_manifest.py --publish decision-p3` writes the default names. Record ids are the
   candidate ids, which is what round 2 matches against the pool (verified on the dry run: 272 pool candidates matched, round-2
   manifest built). Only pool rows are re-mined, not the constructed variants.
2. **Stage-4 duration**: ~50–70 min per checkpoint is an estimate (the DecisionBench 3k and fast-decisions dev are the largest
   panels). `CKPT_FRAC=0.25` (the default) gives ~4 r1 snapshots; the eval lane scales with the number of snapshots.
3. **Memory at 16k tokens**: gradient checkpointing is on by default (`CKPTING=1`, ~25% slower). The trainer halves a chunk on OOM
   and fails only if a single example does not fit.
4. **Parity on CUDA** is measured on the pod for the first time. A FAIL simply keeps 255 codes.
5. **The gatekit** carries the ImajevBench test gold to the pod (the only benchmark content that travels). It is a separate private
   bundle unpacked outside the training tree; delete its repo after the pull.
6. **fast-decisions** has no pinned upstream revision; a changed file is logged, and shipped and candidates are scored on the same
   files on the same pod.
7. **Pilot noise**: 400-row dev passes have ±2.5 pt noise; the rules average the last two passes and demand "no harm" rather than
   a significant gain for 256 codes.
8. **Lane resume**:
   - The resumed state is the lane's rank-0 copy, and the LoRA parameters are identical on every rank.
   - If the lane's `last/` is corrupt in a way the loader does not detect, the r1 run continues from bad weights. Its first
     dev pass (~25% in) would show it.
   - A lane attempt that crashed and retried runs past step 100 (`--max-steps` counts from the retry). `lane_resume.py` then
     uses the step it reached (still an exact prefix) and notes it; the pilot decision compares that lane at later steps, as
     before this change.
