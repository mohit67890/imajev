# Phase 2c pod runbook (8×H100, 24 Sept 2026)

> **Pre-launch audit (24 Sept, `docs/phase-2c-audit.md`): do not create the pod until the bucket is fixed.** The uploaded
> `p2c-static.tgz` (454,831,210 bytes) has empty `data/manifests/` and `data/external/eikos-decisions/` and lacks
> `writer-p2b*`, `human/`, `licenses/` and the `data/decision-v2` probe images; the fixed bootstrap stops with `DATA_INCOMPLETE`
> right after the download. Rebuild and re-upload it, and re-upload `p2c-code.tgz` (the audited scripts are newer than the uploaded code).

Plan: `docs/phase-2c-plan.md` (with the amendments), `docs/eikos-analysis.md` §7, `docs/decision-p2-spec.md` (Phase 2c section).
Scripts: `cloud/bootstrap_p2c_h7.sh`, `cloud/pod_run_p2c_gen.sh`, `cloud/pod_run_p2c_train.sh`,
`cloud/setup_eikos.sh`, `scripts/p2/competitor_bench.sh`, `scripts/p2/make_gates_reference.py` + `cloud/p2c_gates_reference.json`.

## GPU map

| Stage | GPUs 0-3 | GPUs 4-5 | GPU 6 | GPU 7 (8-GPU pod) | CPU |
|---|---|---|---|---|---|
| bootstrap | – | – | – | – | HF models, bucket objects (static, image replay, code, irrelevance images), venvs, Eikos install |
| gen phase A | Qwen3.6-27B writer, TP=4 `:8000`: 1,500 judge docs (thinking on), then 1,500 unknown-family docs (thinking off, max_tokens 8000) | 35B-A3B TP=2 `:8040`: relabel shard 0/3 | 35B-A3B TP=1 `:8060`: relabel shard 1/3 | 35B-A3B TP=1 `:8090`: relabel shard 2/3 | `gen_programmatic.py --docs 3000 --seed 7` |
| gen phase B | 35B-A3B TP=4 `:8050`: new questions shards 0,1/4 + relabel remainder 0,1/4 | 35B-A3B TP=2 `:8040`: new 2/4 + relabel 2/4 | gpt-oss-20b TP=1 `:8020`: new questions (default mode) | 35B-A3B TP=1 `:8090`: new 3/4 + relabel 3/4 | – |
| assemble, manifest, pack | – | – | – | – | yes |
| train: smoke (blocking) | idle (waits for `SMOKE_OK`) | smoke off / on | smoke on+sw0 | Eikos-4B `:8110` (+ vLLM `:8111`) | |
| train lane | trainer, torchrun 4 procs, master port 29500: 4b-delta → 4b-fresh → 9b-delta → 2b-delta | | | | |
| eval lane | | per run: panels sharded over 4-7; JevBench best on 4 `:8765`, last on 5 `:8766` | panels; ImajevBench best | panels (once Eikos is done) | |

On a 7-GPU pod the scripts fall back to the old map (no `:8090`; shards 0/2,1/2 and 0-2/3; Eikos on GPU 6 in the eval lane).

The 27B only writes (owner decision): the new judge + unknown-family + programmatic questions are answered by the 35B-A3B in distribution mode (soft
targets + rationale) and gpt-oss-20b in the default mode; assembly keeps a question only when the 35B's argmax AND gpt-oss both equal the
intended answer, and takes the 35B's distribution as `target_probs` (`--soft-from qwen35`). The two writer batches (`p2c-judge`, `p2c-unknown`)
are assembled together with `--unknown-rule relaxed`; an unknown-gold question whose 35B argmax is not `unknown` keeps its hard target
(the assembler's keep-hard default). The unknown pass exists because the mixture was only ~5-8% unknown-gold (target ≥ 15%). If the 35B finishes its relabel shards before
the writer is done, the phase-A workers continue on the programmatic questions; phase B resumes everything with `--done-from`.
Ports 8001/8081 are RunPod's and are never used. Every stop kills only its own servers (recorded process group, or the exact vLLM/trainer
command line for that port or output directory).

## Expected durations (rough; $21.52/h for 8×H100)

| Stage | Estimate | Notes |
|---|---:|---|
| bootstrap | 30-45 min (+ wait for the code tgz) | ~175 GB of models from HF (3 in parallel), bucket objects, vLLM + training venvs |
| gen phase A | 1.5-2 h | bound by the 27B writer: 1,500 thinking judge docs (max_tokens 16,000), then ~20-30 min for 1,500 thinking-off unknown docs; relabel ~21.5k questions runs alongside |
| gen phase B | 50-70 min | ~18k new questions (4.5k judge + 4.5k unknown-family + 9k programmatic) + the relabel remainder on 6 GPUs of 35B; gpt-oss ~20-30 min on one GPU |
| assemble + manifest + pack | 10-15 min | |
| **generation total** | **~2.8-3.3 h** | |
| 4b-delta / 4b-fresh / 9b-delta / 2b-delta | ~1-1.5 h / ~1-1.5 h / ~1.5-2.5 h / ~0.5-0.75 h | rationale tokens roughly double tokens per row; the trainer prints the step count at start |
| eval per run | 40-60 min | overlaps the next run's training; the last run's eval is the tail |
| **training total** | **~5-7 h** | |

Total ~8.5-10.5 h, ~$210-255, which is more than the plan's ~5 h. To cut it: `RUNS="4b-delta 4b-fresh 9b-delta"` (skip the 2B, ~45 min), or run the
9B only after the 4B lanes are compared (`RUNS="4b-delta 4b-fresh"`, then `RUNS="9b-delta"`).

## Before the pod (Mac)

1. The other agent's interfaces must be in the tree: `gen_answer.py --mode distribution`, `assemble_p2.py --soft-from`, `build_p2c_manifest.py`
   (its image replay is pre-sampled into `data/decision-p2c/image-replay/replay-{delta,fresh}.jsonl` by `scripts/p2/make_image_replay.py`).
2. Unknown references for the gates (already filled on 24 Sept: 2B 85.71% of 14 / 4B 100% of 14 / 9B 92.86% of 14 correct-unknown;
   false abstention 0.95 / 0.24 / 0.24%): `python scripts/p2/make_gates_reference.py`
3. Bucket objects (nothing large goes Mac → pod; the 7×H100 host took 180 KB/s from the Mac last night), uploaded by the coordinator:
   - `gs://<bucket>/decision-p2c/p2c-static.tgz`: `adapters/{2b,4b,9b}-p2b-best`, `data/decision-p2/{teacher,human}`,
     `data/external/eikos-decisions`, `data/manifests/*`, `data/decision-v2` probe images, `bench/`, `external/jevbench/`
   - `gs://<bucket>/decision-p2c/p2c-image-replay.tgz`: `data/decision-p2c/image-replay/{replay-fresh.jsonl,replay-delta.jsonl,images/}`
   - `gs://<bucket>/decision-p2c/p2c-code.tgz`: the imajev tree, uploaded LAST. The bootstrap polls for it for up to 90 min
     (`CODE_WAIT_MIN`) after everything else is in place. Its code (src, scripts, cloud, artifacts) replaces the tree's; its data never
     overwrites the bucket data.
   - Already in the bucket: `decision-v1/abo.463dd86293a969c0.tar` and `decision-v1/vizwiz.c1d295ebe24db813.tar`, the images of the
     `decision-v1.1-irrelevance-test` panel (ship gate d), fetched too.
   - Build every Mac tarball without macOS metadata: `COPYFILE_DISABLE=1 tar --no-xattrs -czf <name>.tgz ...` (no AppleDouble `._*`
     members, no `LIBARCHIVE.xattr.*` pax keywords; GNU tar on the pod warns on those and exits non-zero). The bootstrap's `tar_x` tolerates
     exactly that warning class for archives already uploaded, but clean archives are the fix. Check before upload: `tar tzf <name>.tgz | grep -c '/\._'`
     is 0 and the listing shows every expected file (the first p2c-static.tgz had empty `data/manifests/`).
4. A short-lived key for the bucket reader service account (the pattern of `cloud/bootstrap_4b.sh`), saved as `$S/reader-key.json`.
   The bootstrap deletes it on the pod. Revoke it in GCP after `BOOTSTRAP_DONE`.
5. State the price, then create the pod: 8×H100 SXM ($21.52/h), secure cloud, ≥ 350 GB volume disk, image as last night (`runpod/pytorch` cu128/torch 2.8).

## Launch

```
# Mac: small files only
scp cloud/bootstrap_p2c_h7.sh <pod>:; scp ~/.cache/huggingface/token <pod>:hf-token; scp $S/reader-key.json <pod>:reader-key.json
ssh <pod> 'cd /workspace && setsid nohup bash bootstrap_p2c_h7.sh > bootstrap.log 2>&1 < /dev/null & echo launched'
# after BOOTSTRAP_DONE in bootstrap.log: generation, and training chained on the generation's ALL_DONE (both on the pod)
ssh <pod> 'mkdir -p p2/out/p2c p2/train-out-p2c && cd /workspace && setsid nohup bash cloud/pod_run_p2c_gen.sh >> p2/out/p2c/run.log 2>&1 < /dev/null & echo gen'
ssh <pod> 'cd /workspace && setsid nohup bash -c "until grep -q ^ALL_DONE p2/out/p2c/run.log; do sleep 60; done; bash cloud/pod_run_p2c_train.sh" >> p2/train-out-p2c/run.log 2>&1 < /dev/null & echo chained'
```
Never put `pkill -f <pattern>` in an ssh command line (it matches the ssh session itself); stop things with a script file on the pod.
Local waiters poll the logs for `ALL_DONE` / `BOOTSTRAP_DONE`, never `ssh pgrep -f`.

Dashboard: `ssh <pod> 'grep -hE " stage | done |FAILED|MISSING|NOT_SHIPPABLE|SMOKE_|ALL_DONE" p2/out/p2c/run.log p2/train-out-p2c/run.log | tail -n 40'`.
Resume after a crash: rerun the same script; stages with a `.DONE` marker are skipped, answer files resume by (doc, question), the writer by doc id,
the trainer from its last checkpoint.

## What to check

**Bootstrap** (`bootstrap.log`, `out/fetch-data.log`): `DATA_DONE`, no `MISSING IN BUCKET`, no `DATA_INCOMPLETE` (checked
right after the bucket download, before the models finish), the image check lines
(irrelevance test, probes and both replay files with 0 missing images), all six model snapshots, `Eikos-4B installed`, `BOOTSTRAP_DONE`.
Then revoke the reader key.

**Generation** (`p2/out/p2c/`):
- `write-judge.log` / `write-unknown.log` tails: accepted documents (~70-75% acceptance expected; 1,500 planned each). `programmatic.log`: 3,000 docs × 3 questions.
- `answer-*.log`: progress per shard; after round 1 the script re-asks parse errors once; the merge lines report answers / questions / missing / parse errors.
- `unknown-audit.txt`: the 35B's `unknown` mass on questions whose intended answer is a real option, per batch. The plan's threshold is a mean
  ≤ 0.05. Above that, stop before training and decide (cap or strip it in `build_p2c_manifest.py`): it would teach the model to hold back
  probability on JevBench, where `unknown` is never gold. The same table gives the 35B's argmax agreement with the intended answer.
- `assembly-counts.txt` + `assemble-{writer,prog}.log` + `assembly-report-{writer,prog}.json` (its `unknown_mass` block is the assembler's own
  audit): kept per batch (judge and unknown: 2-answerer unanimity, expect roughly 50-65% of ~4,500 each; `soft_disagreed_unknown_kept_hard`
  counts unknown-gold rows kept with a hard target; programmatic:
  computed answers, expect a high keep rate of ~9,000), `target_probs` on every kept row, rationale present, contamination drops (should be ~0),
  no duplicate ids.
- `manifest-check.txt` + `data/manifests/decision-p2c-report.json`: unknown-gold share (≥ 15% is the goal), rows and partitions of `decision-p2c` / `-fresh` / `-judge-dev`, image rows (~15% delta, ~35% fresh), soft-target
  coverage, unknown mass on answerable soft rows, sources (Eikos slice capped, programmatic ≤ 50%).

**Training** (`p2/train-out-p2c/`):
- `smoke/summary.txt`: `SMOKE_OK`. Step-0 dev accuracy is equal for flags off / on / on with `--soft-weight 0`, the off and sw0 losses match,
  every run logged 5 steps, and only the flag-on runs log a rationale loss. The on-vs-off loss difference is expected (soft CE vs hard CE on dev
  rows with `target_probs`). The smoke runs BEFORE the train lane; `SMOKE_MISMATCH` stops the script (no training) — fix it, or rerun with
  `SMOKE_IGNORE=1` after reading the logs.
- `INIT_ADAPTER_MISMATCH` (before training): a delta lane's init adapter is not r16/α32 on the default targets with a readout; the trainer
  would otherwise silently start from a fresh LoRA (PEFT's non-strict load).
- `EVAL_FAILED <run> <tag> <panel>`: a shard of the evaluator failed; no `predictions.jsonl` is written, so a rerun of the script redoes it.
- The first lines of `<run>/train.log`: steps, rows, the recipe string incl. soft targets / rationale / permutation; `4b-fresh` must show r64 / α128
  and the twelve target modules. LoRA dropout stays 0.0 because the trainer has no flag for it (the plan's 0.05 is not applied).
- `<run>/dev-curve.jsonl`: dev and judge-dev accuracy per checkpoint (select = mean accuracy).
- Temperature lines: `fitted T -> shipped T` (shipped ≥ 1.0 by rule; teacher rows only, hard rows ×3).
- `<run>/gates.json` + `gate ... PASS/FAIL` lines: (a) ImajevBench ≥ ref − 1 pt, visual/joint ≥ ref − 2 items; (b) state/pairs probes ≥ ref − 2 pts;
  (c) decision-p2b test correct-unknown ≥ ref and false abstention ≤ ref + 2 pts; (d) irrelevance test ≥ ref − 2 pts. A failure prints
  `<run> NOT_SHIPPABLE`, and the run is still fully evaluated. Gate (c) rests on 14 unknown-gold rows: for the 4B one miss (100% → 92.9%) fails it,
  so read a (c) failure together with the row-level predictions before deciding.
- `summary.txt`: p2c test, judge dev, JevBench hard raw / cal (best, last) with ECE, T, gates; Eikos-4B same-protocol hard line
  (compare with its serial 72.1, not 73.9).

## Pull and terminate

```
mkdir -p reports/decision-p2c/pod
scp <pod>:p2/decision-p2c-teacher.tgz <pod>:p2/p2c-train-results.tgz reports/decision-p2c/pod/
tar xzf reports/decision-p2c/pod/decision-p2c-teacher.tgz -C data     # teacher files, records-p2c(-writer), manifests decision-p2c* + report
tar xzf reports/decision-p2c/pod/p2c-train-results.tgz -C reports/decision-p2c/pod   # train-out-p2c (adapters best/last, evals, raw JevBench) + out/eikos4b
```
Check the adapters' sha256 on both sides (`sha256sum p2/train-out-p2c/*/train/best/adapter_model.safetensors` vs `shasum -a 256`),
then terminate the pod (RunPod delete) and confirm that the reader key is revoked (only the system-managed key remains). A pod without a volume
loses everything when the balance hits zero: pull the teacher archive as soon as the generation reports `ALL_DONE`.
