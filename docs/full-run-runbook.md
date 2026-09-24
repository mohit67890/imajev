# Full-run runbook (decision-v1 on rented multi-GPU pods)

Every line here is a lesson from the 21–22 Sept shakedowns. Follow it in order.

## Before renting anything
1. `scripts/v1/assemble_manifest.py` then `scripts/v1/audit_manifest.py`: zero missing images, read the flags.
2. `scripts/v1/upload_sources.sh <sources>` and upload the manifest named by its hash. Pods pull from
   `gs://imajev-emoland-925e3`; uploading from the Mac straight to a pod runs at ~1 MB/s, to the bucket at ~10–40 MB/s.
3. Local end-to-end test on the tiny manifest (`decision-v1t`) after any change to the loader, training or evaluation:
   `--device mps --batch-size 1 --max-batch 1` (left-padded batches give NaN on Apple MPS; batching is CUDA-only).
4. Prices: the capacity tool reports the cheapest tier. Secure-cloud pods cost more (4x H200: $18.36/hr, not $14.36).
   Quote the `cost` field returned by pod creation, and check the balance covers the whole run with margin.

## Pod setup
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; system pip is locked (PEP 668): use a venv with `--system-site-packages`.
- **Do not install `causal-conv1d`.** With it compiled, `torch.distributed.run` with 2+ processes hung before the first forward (single process worked); without it, 2x A40 and 4x H200 trained fine. Cause unconfirmed; not worth pod time.
- **Hopper GPUs (H100/H200) need `tilelang`**: flash-linear-attention refuses its backward pass with the Triton that ships
  with torch 2.8 (known wrong gradients). A40/L40S do not show this, so a test on those proves nothing about Hopper.
- Launch long jobs with `setsid nohup ... < /dev/null &` from a script file. Do not `pkill -f` a pattern that also appears in
  your own ssh command line.
- Under `set -o pipefail`, `quiet_command | grep -v ...` exits non-zero when nothing matches. Redirect to a log file instead.
- Create the bucket reader key just before launch; the bootstrap deletes it from the pod after the download; revoke it in IAM
  when the run ends (`gcloud iam service-accounts keys delete`). Only the SYSTEM_MANAGED key should remain.
- macOS tar adds `._*` files: build tars with `COPYFILE_DISABLE=1` and delete `._*` on the pod.

## Training settings that worked
- 4 GPUs via `python -m torch.distributed.run` (the venv has no `torchrun` binary). Verified: 2 GPUs reproduce 1-GPU losses, 1.9x faster.
- Fixed batch 32/GPU ran out of memory on 141 GB cards (one long document pads the whole batch); batch 16 ran clean at ~80 examples/s.
- Now: length-bucketed micro-batches with `--token-budget` (padding waste 0.9% instead of >18%). Start at 16000 on 80 GB cards,
  24000 on 141 GB; `--batch-size` is only the cap. The out-of-memory guard skips a batch rather than killing the run: watch `skipped_oom_batches`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no gradient checkpointing on 80 GB+ cards, 16 loader workers per GPU.
- One epoch. The 3.6k-example run memorised after ~1.2 epochs; the 60k run was still improving at the end of its single pass.
- Learning rate 1.5e-4 was healthy at 64 examples/step. Bucketed steps carry ~160 examples on 4 GPUs: use 2e-4, warm-up 100 steps.
- First 10 minutes of any run: confirm examples/s, GPU utilisation near 100%, zero skipped batches, falling loss. Stop if not.

## Evaluation
- Unbatched evaluation took ~20 minutes on 4x H200 (about $6). Use the batched evaluator, or set `SKIP_EVAL=1` and evaluate the
  adapter afterwards on one cheap GPU.
- Always report base-vs-tuned on the same cases, split into trained sources (new images) and held-out sources, plus abstention by
  cause and by field type. In-distribution numbers alone overstate a general model.

## Known result to beat (60k-example shakedown, manifest decision-v1s)
Trained sources 56.9% -> 80.0% (ECE 0.149 -> 0.020; 43% auto-accepted at >=0.95 with 98.4% precision).
Held-out sources 39.5% -> 46.8%; held-out unanswerable yes/no (TUBench): 1/225 correct abstentions — the gap `masked_evidence` targets.

## Full v1 run (22 Sept, 4x H100 SXM, RunPod Secure $13.96/hr)
- 594k examples, one epoch, ~75 min of training at 127–139 examples/s with GPUs at ~100%; balanced length-bucketed batches, token budget 11,000, 2 micro-batches/step.
- **Random SIGSEGV crashes in training ranks** (11 in total, different ranks, no ECC errors) — a GPU-kernel fault, not data. Mitigation that worked:
  checkpoints every 20 steps (`--save-every 20`, dev every 100) and a retry loop that resumes from the last checkpoint. Never skip data.
  Before the next big run, test a newer flash-linear-attention/Triton on a short job.
- Result: dev 1.238 -> 0.313 (87.5%). Test: trained sources 57.6% -> 85.8% (ECE 0.003; 55% auto-accepted at >=0.95 with 98.9% precision);
  held-out sources 39.4% -> 49.8% — the same as the 60k fixed-data adapter (50.2%). More in-distribution data did not improve generalisation
  to unseen sources. Held-out unanswerable yes/no (TUBench style): 0/150. Adapter: reports/decision-v1/runs/h100x4-full/best.

## v1.1 run (22 Sept, 4x H100 SXM Secure $13.96/hr, manifest decision-v1.1 = 200k text + 300k image + 4k contradiction decisions)
- The length plan is an estimate: mixed text+image micro-batches came out up to 1.7x larger than planned (19k padded tokens against an
  11k budget) and this model needs ~4.7 GB per 1k padded tokens without gradient checkpointing, so anything above ~15k tokens overflows 80 GB.
  With data skipping forbidden the run aborted at step 2. Fix in the trainer: a planned micro-batch whose real padded size exceeds 1.1x the
  budget is split in halves, and an out-of-memory error halves the chunk and retries instead of skipping. Probe first:
  `probe_memory.py` (session scratchpad) reports actual vs planned tokens and peak memory for the heaviest batches on one GPU.
- `pkill -f <pattern>` over ssh kills your own session when the pattern appears in the ssh command line: put stop commands in a script file.
- Smoke recipe that validated the 4-GPU readout path: 30 steps, save-every 10, dev-every 15, eval limit 300 per shard (~12 min including the 3-minute manifest audit).
- **After a SIGSEGV, check `nvidia-smi --query-compute-apps` before resuming.** In the v1.1 run the crashed attempt left one sibling
  process holding 52 GB on GPU 0 and 16 orphaned loader workers; the resumed rank 0 then hit out-of-memory on every step and crawled.
  `cloud/pod_run_v1_1.sh` now kills stragglers and waits for the GPUs to be empty before every attempt. The v1 run's "random" crashes
  and slow resumes may have had the same cause.
- Outcome: 2,495 steps, ~118 ex/s, 9 segfault resumes (one every ~250 steps, shorter towards the end), ~$24 of pod time. Final step = best
  (dev 1.21 → 0.404). Test 87.3% (image 86.1%, text 88.7%), ECE 0.016; v1 image exam held-out 50.7% (v1 49.8%): text added at no image cost.
  Results: reports/decision-v1.1/README.md. A pod without a volume loses everything when the balance hits zero: keep pulling checkpoints.

## 9B run (22 Sept evening, 4x H200 SXM Secure $18.36/hr, same manifest as v1.1, from scratch)
- Memory: the 9B needs ~13 GB per 1k padded tokens on top of ~20 GB fixed; 7.9k tokens peaked at 109 GB on a 141 GB H200. Settings:
  token budget 7,500, no checkpointing, pad-multiple 64. 3,508 steps at ~2.5 s.
- Padding to multiples of 64 did NOT stop the SIGSEGVs (first crash at step 55). A Triton 3.8 environment failed at startup with an NCCL
  CUDA error (its torch reinstall swapped CUDA runtime wheels); reverted after one attempt.
- **Likely SIGSEGV cause found: `faulthandler.dump_traceback_later(240, repeat=True)` in the trainer.** 6 of 8 crashes landed within a
  minute after a periodic dump (chance ≈ 25% each). The dump thread walks all thread stacks without the GIL while Triton/tilelang kernels
  and loader threads are active. Disabled at step 980 of the 9B run; watch the crash rate afterwards to confirm.
- Outcome: 3,508 steps at ~55 ex/s; after removing the periodic faulthandler dump at step 980 there were **zero** crashes in 2,528 steps
  (8 in the 980 before). Best = step 3,100, dev 0.987 → 0.371. ~$60 of pod time. Test 89.0% (2B 87.3%), ECE 0.007; v1 image exam
  held-out sources 55.9% (2B 50.7%, the first movement in four runs); MMLU 73.8% (2B 53.9%); typed-decisions 66.2% (2B 58.1%);
  JevBench original 98.6% (2B 88.9%), hard 42% (2B 43%: the hard split does not yield to scale at 9B). Results: reports/decision-v1.1-9b/README.md.
- Copy panel manifests and held-out source tars to EVERY new pod; they are not part of the bootstrap objects.

## v2 run (23 Sept, 4x H100 SXM Secure $13.96/hr, user-created pod resized to 250 GB)
- Stages: preflight (batched vs unbatched labeller on 48 candidates: 0 flips, max prob diff 0.01) → 9B labelling of 367k candidates
  (4 shards, ~130 decision-orders/s total, 2.3 h; 74% kept) → base-2B regularisation pass over 504k v1.1 rows (~15 min) → PD12M second
  labelling pass (180k candidates, 1 h; 80% kept) → assembly (940k records, audit ok) → 2B student 0.4 epoch from the v1.1 adapter,
  lr 5e-5, ~105 ex/s, 1,866 steps, **zero segfaults** (faulthandler timer removed) → evals.
- Two dev signals: dev (v1.1 + v2 dev rows) 86.8 → 89–90%; reasoning dev 42.7 → 46.3% at the selected best (step 1,100), after an
  early dip to 37%. The v1.1 recipe had traded reasoning for fit; the regularised recipe gained on both.
- Lessons: the labeller must prepare batches in worker processes (added `--workers`); its ETA uses a cumulative rate and undershoots on the
  image-heavy tail; a watchdog's pgrep pattern must match the script it launches (a mismatch relaunched a second copy); pod_run scripts
  rotate their log, so monitors should grep the current file only.

## v2 outcome (23 Sept 2026, the pod, 4x H100)
- Stages: preflight → label (9B teacher, 4 GPUs × 1 shard, `--workers 16`, ~2.5 h incl. PD12M) → blend (base 2B over v1.1 rows) →
  assemble (940k records) → train 0.4 epoch from the v1.1 adapter, lr 5e-5, `--select mean_accuracy` with the reasoning dev set →
  eval. 1,866 steps, zero crashes. Best step 1,100. ~$25.
- Result: held-out image sources 50.7 → 54.4, two-image 37.9 → 54.2, state probe 60.5 → 69.5; v1.1 test 87.3 → 86.1 with MMLU
  53.5 → 43.0 from over-abstention (teacher rows ~30% unknown + base-blend unknown mass). Reasoning flat. Raw model under-confident
  (temperatures 0.3–0.7); fitted temperatures over-sharpen off-distribution. Report: reports/decision-v2/README.md.
- Lesson: cap `unknown` targets in pseudo-labelled data (~15%) and do not blend base distributions into answerable text rows; check
  predicted-unknown rate per panel before calling a run better.
- v2.1 chained automatically (`chain_v21.sh` waits for ALL_DONE): label state_grounded + pairs_natural → assemble_v21 (replay 150k v2
  rows) → train 1 epoch from v2 best → eval incl. state/pairs probes.
- v2.1 assembly failed its audit twice on the 150k replay subsample: (1) 2,683 irrelevant-image controls whose relevant twin was not
  sampled; (2) 403 image hashes spanning partitions because state_grounded / pairs_natural dev/test rows were built on ABO and
  state_aware photos that v2 trained on. `assemble_v21.py` now tops up the twins and repairs new rows by whole source_group to a
  fixpoint (photo trained on → group to train; photo in v2 dev/test/calibration → group dropped; photo shared by two new groups in
  different partitions → both to train). Outcome: +2,732 twins, 899 rows moved, 7,743 dropped, 232,092 train decisions, audit ok.
  Lesson: build new sources on photos that are not in any existing manifest, or partition them by the photo's existing partition.

## v2.1 outcome (23 Sept 2026, the pod, terminated; pod8 total $95.43 for v2 + v2.1)
- Label (9B teacher, 4×35k decisions, ~15 min: 84% of state_grounded and 91% of pairs_natural kept) → assemble (audit repairs above)
  → train 1 epoch from v2 best (1,361 steps, 0 crashes, ~40 min) → eval incl. state/pairs probes (~25 min).
- The mean_accuracy selector picked step 50 (reasoning dev 48.0% vs 43.75% at step 1,361): the released "best" saw ~8k of the
  232k new decisions. Keep the final checkpoint too (`last/`) and evaluate both; a selector that only weighs the reasoning dev
  set will always reject a run whose data is not about reasoning.
- Step 50 vs v2: pairs probe 65 → 82%, state probe flat, everything else within a point; MMLU 43.0 → 45.7 as abstention eased.
  Final checkpoint: state probe 73%, pairs probe 100% (in-distribution templates), JevBench hard 43.2% raw.
- Lesson: probes built from the same template families as the training data measure absorption, not generalisation; the
  independent checks are the v1 exam held-out sources (incl. heldout_pairs) and MMLU.
- Lesson: the pod's evaluation stage must cover every checkpoint that might be released (selector's best AND the final one), on every
  panel. Scoring an afterthought checkpoint on the Mac takes ~45 min for what the pod does in 3 min for under $1.

## Phase-2 / benchmark pods (23 Sept 2026): lessons
- vLLM 0.30 workers call `ninja` from the shell PATH; a venv-installed ninja is not enough. Launch with
  `env PATH=<vllm-venv>/bin:$PATH`, or the engine dies with `No such file or directory: 'ninja'` after loading.
- `snapshot_download(..., revision=<prefix>)` and offline resolution of a partial snapshot both fail on a pod; pass the
  full 40-char revision and resolve `$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>` directly.
- Never put a process name you are about to `pkill -f` on the same ssh command line: the kill matches the ssh session
  itself. Always ship a script file and run `bash that.sh`.
- The PyTorch server backend defaulted to `mps`; it now auto-selects CUDA / MPS / CPU. Test the CUDA path on a pod
  before a release, not on the Mac.
- `tar --exclude` must precede the paths on macOS bsdtar, or the whole `&&` chain silently stops at that step.
- The 4B from-scratch run on the corrected mixture (no base blend, unknown capped, grounded dev selection) removed the
  abstention prior and preserved reasoning (authored reasoning dev 62.9% vs the 2B's 42.5%): data shape, not a
  regulariser, is what protects reasoning.
- Phase-2 (24 Sept): run every assembled record through `decision_data.render` before writing a manifest; two teacher rows with
  JSON states nested > 8 levels crash-looped the trainer's DataLoader ("State nesting exceeds 8"). Pass `--batch-size 40
  --accumulate 2 --workers 16` to the trainer: without them micro-batches are single examples and the GPUs sit at ~50%.

### Lessons 2026-09-24 (phase-2b staging)
- **Upload speed differs per pod.** The Mac pushed 1.3 MB/s to the Montreal H200 pod but 180 KB/s to the 7×H100 pod, so an 825 MB
  bundle would have taken over four hours. Ship pod-to-pod instead: `ssh-keygen` on the source pod, append its public key to the
  target's `authorized_keys`, `scp -P <port>` between pods (1.8 GB in under two minutes). From the Mac send only the code delta,
  and rebuild data/adapters on the source pod with `rsync --copy-dest=<existing tree>` so unchanged files never cross the link.
  GCS works too when no pod already holds the files.
- **`ssh host 'pgrep -f <script>'` always matches.** The remote `bash -c` command line contains the pattern, so a waiter built on it
  never fires (the H200 follow-up sat unlaunched for ~90 min). Check for a DONE marker or `ALL_DONE` in the log instead, or
  filter with `grep -v pgrep`.
- ImajevBench needs `artifacts/model-<name>.json` for every base model; `model-qwen4b.json` was missing (4B panel failed once).
- **Third-party LoRA on Qwen3.5: check the key layout before believing a number.** Adapters trained on the text-only class use
  `model.layers.N`; the multimodal class (what PEFT's `AutoModelForImageTextToText` and vLLM load) names them
  `model.language_model.layers.N`. Both PEFT and vLLM then attach zero adapters *silently* (cua-s1-4b: identical to base on
  111/111 items, vLLM even JIT-compiled its LoRA kernels). Always run a control on the untouched base and an item-level
  identity check; remap keys with a safetensors rewrite when they differ (`cua_fix.py` pattern) and verify max |Δlogit| > 0.
- RunPod pods reserve ports 8001 and 8081 (a proxy answers 405 there); pick 8010/8020/8030/8090 for extra vLLM servers.
- jevbench `openai_compat` sends `max_tokens 4096`; thinking models exhaust it and the runner stops after 10 consecutive
  empty replies. The pod copy reads `OPENAI_COMPAT_MAX_TOKENS` / `OPENAI_COMPAT_TIMEOUT_S`; port that patch upstream.
