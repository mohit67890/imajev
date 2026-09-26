#!/usr/bin/env bash
# Phase 3 (imajev-4b, last training run): bootstrap ONE fresh 8xH100 pod (docs/phase-3-plan.md; runbook reports/phase3/pod-runbook.md).
# Layout (as the phase-2c pod):
#               code + training data (bundle p3-train: src scripts cloud artifacts + manifests + referenced images + pool)
#   venv              training / eval venv (transformers 5.17, peft 0.21, as p2c)
#   hf/hub/...        Qwen/Qwen3.5-4B @ 851bf6e (public, from Hugging Face on the pod)
#   adapters/4b-shipped   the shipped imajev-4b 1.0 (soup50) from its public HF repo @ 712891d, sha256-checked: start point + baseline
#   bench             ImajevBench v2.0-lite eval records (test gold) + assets (bundle p3-gatekit; outside the training tree)
#   evalsets/jevbench          JevBench harness, public git @ 2fa63fa (datasets/public pinned by sha256)
#   evalsets/fast-decisions    fastino/fast-decisions dev files (public HF; sha256 compared with the pins, logged)
#   p3/tracking-only/decisionbench-3k.jsonl   DecisionBench 3k stratified subset (built here; TRACKING ONLY)
# Nothing large travels from the Mac: bundles come from a private HF dataset repo or the GCS bucket (scripts/p3/transfer.py).
# Small files scp'd from the Mac beforehand:
#   bootstrap_train.sh (this file), transfer.py (scripts/p3/transfer.py), p3.env (NO secrets:
#   BACKEND=gcs (default; GCS_BUCKET default gs://<bucket>/, GCS_PREFIX default phase3) or hf (TRAIN_REPO / GATEKIT_REPO),
#   TRAIN_BUNDLE, GATEKIT_BUNDLE, optional run overrides)
#   secrets, deleted by this script: hf-token (HF read token; needed for the hf backend) and, for gcs, reader-key.json
# Launch: cd /workspace && setsid nohup bash bootstrap_train.sh > bootstrap.log 2>&1 < /dev/null &
# Markers in bootstrap.log: BOOTSTRAP_DONE | BOOTSTRAP_FAILED. With RUN_AFTER=1 (default) it then starts
# cloud/p3/pod_run_train.sh detached (log p3/run/run.log; ALL_DONE | FAILED). Nothing here terminates the pod.
set -uo pipefail; cd /workspace
log(){ echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST bootstrap: $*"; }
die(){ log "BOOTSTRAP_FAILED: $*"; rm -f hf-token reader-key.json; echo BOOTSTRAP_FAILED; exit 1; }
[ -f p3.env ] || die "missing p3.env"; set -a; . p3.env; set +a
BACKEND=${BACKEND:-gcs}; GCS_BUCKET=${GCS_BUCKET:-gs://<bucket>/}; GCS_PREFIX=${GCS_PREFIX:-phase3}; TRAIN_BUNDLE=${TRAIN_BUNDLE:-p3-train}; GATEKIT_BUNDLE=${GATEKIT_BUNDLE:-p3-gatekit}; RUN_AFTER=${RUN_AFTER:-1}
QWEN_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; SHIPPED_REPO=${SHIPPED_REPO:-mohit67890/imajev-4b}; SHIPPED_REV=${SHIPPED_REV:-712891d1192c6491441a19bdea254d8a6e1048d0}
SHIPPED_SHA=d8d328f85dcda4459a2331c47aba2a89ec808210e712248afdb69e38b9e0c25d
JEVBENCH_URL=${JEVBENCH_URL:-https://github.com/fstandhartinger/jevbench.git}; JEVBENCH_REV=${JEVBENCH_REV:-2fa63fa3226cb369795525ed011800f57dcbd894}
[ -f transfer.py ] || die "missing transfer.py"
mkdir -p  adapters bench evalsets p3/tracking-only p3/run fetch out
case $BACKEND in
  hf)  [ -s hf-token ] || die "missing hf-token (hf backend)"; : "${TRAIN_REPO:?TRAIN_REPO}"; : "${GATEKIT_REPO:?GATEKIT_REPO}"
       FETCH_ARGS=(--backend hf --token-env HF_TOKEN) ;;
  gcs) [ -s reader-key.json ] || die "missing reader-key.json (gcs backend)"; chmod 600 reader-key.json
       export GOOGLE_APPLICATION_CREDENTIALS=reader-key.json; : "${GCS_BUCKET:?GCS_BUCKET}"
       FETCH_ARGS=(--backend gcs --bucket $GCS_BUCKET --prefix $GCS_PREFIX) ;;
  *) die "BACKEND must be hf or gcs" ;;
esac
# the token goes into the environment only (never on a command line, never echoed)
if [ -s hf-token ]; then HF_TOKEN=$(cat hf-token); export HF_TOKEN; fi
export HF_HOME=hf

# 1. venv (as phase 2c) --------------------------------------------------------------------------------------------------------
if [ "${SKIP_VENV:-0}" != 1 ]; then
  python -m venv venv --system-site-packages && . venv/bin/activate || die "venv"
  pip install -q "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" "safetensors>=0.8" "pillow>=11" "pydantic>=2,<3" scipy scikit-learn \
    fastapi "uvicorn[standard]" python-multipart httpx pyarrow "huggingface_hub>=1.32" google-cloud-storage flash-linear-attention tilelang 2>&1 | tail -n 1
else . venv/bin/activate; fi
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" || die "python deps"

# 2. model + shipped adapter from Hugging Face (background) ------------------------------------------------------------------
python - > out/download-models.log 2>&1 <<PY &
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-4B", revision="$QWEN_REV", ignore_patterns=["*.md", ".gitattributes", "original/*"], max_workers=16), flush=True)
print(snapshot_download("$SHIPPED_REPO", revision="$SHIPPED_REV", local_dir="adapters/4b-shipped", ignore_patterns=["mlx/*", "assets/*"]), flush=True)
print("MODELS_DONE", flush=True)
PY
PM=$!

# 3. bundles (private store) ------------------------------------------------------------------------------------------------------
log "fetching $TRAIN_BUNDLE and $GATEKIT_BUNDLE via $BACKEND"
python transfer.py fetch "${FETCH_ARGS[@]}" ${TRAIN_REPO:+--repo $TRAIN_REPO} --name $TRAIN_BUNDLE --dest fetch --extract  \
  || die "fetch $TRAIN_BUNDLE"
python transfer.py fetch "${FETCH_ARGS[@]}" ${GATEKIT_REPO:+--repo $GATEKIT_REPO} --name $GATEKIT_BUNDLE --dest fetch --extract gatekit --gatekit \
  || die "fetch $GATEKIT_BUNDLE"
cp -a gatekit/bench/. bench/
mkdir -p data/manifests && cp gatekit/manifests/* data/manifests/ 2>/dev/null
rm -rf gatekit fetch

# 4. public eval sets, pinned ----------------------------------------------------------------------------------------------------
( set -e
  rm -rf evalsets/jevbench; git clone -q $JEVBENCH_URL evalsets/jevbench; cd evalsets/jevbench; git checkout -q $JEVBENCH_REV
  pip install -q -e . 2>&1 | tail -n 1 || echo "jevbench: pip install -e failed (runs from the repo root as in phase 2c)"
) > out/evalsets.log 2>&1 || die "jevbench clone (see out/evalsets.log)"
python - <<'PY' >> out/evalsets.log 2>&1 || die "benchmark pins (see out/evalsets.log)"
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
pins = json.load(open("cloud/p3/benchmark_pins.json"))["benchmarks"]
bad = [f for f, h in pins["jevbench-public"]["files"].items()
       if hashlib.sha256(Path(f"evalsets/jevbench/datasets/public/{f}").read_bytes()).hexdigest() != h]
print("jevbench public pins:", "OK" if not bad else f"MISMATCH {bad}")
if bad: sys.exit(1)
d = snapshot_download("fastino/fast-decisions", repo_type="dataset", allow_patterns=["*.jsonl"], local_dir="evalsets/fast-decisions")
fd = pins["fast-decisions-dev"]["files"]
mism = [f for f, h in fd.items() if not Path(d, f).exists() or hashlib.sha256(Path(d, f).read_bytes()).hexdigest() != h]
print(f"fast-decisions: {len(fd) - len(mism)}/{len(fd)} files match the pins" + (f"; differ: {mism} (revision unpinned upstream; shipped and candidates run on the same files)" if mism else ""))
PY
python scripts/p3/make_decisionbench_subset.py --out p3/tracking-only/decisionbench-3k.jsonl >> out/evalsets.log 2>&1 \
  || die "DecisionBench subset (see out/evalsets.log)"
mkdir -p .cache/external; ln -sfn evalsets/jevbench .cache/external/jevbench

wait $PM || { tail -n 20 out/download-models.log; die "model download"; }
M4=hf/hub/models--Qwen--Qwen3.5-4B/snapshots/$QWEN_REV; echo "$M4" > model_path
got=$(sha256sum adapters/4b-shipped/adapter_model.safetensors | cut -d' ' -f1)
[ "$got" = "$SHIPPED_SHA" ] || die "shipped adapter sha256 $got != $SHIPPED_SHA"
rm -f hf-token reader-key.json; unset HF_TOKEN; log "HF token / reader key removed from the pod (revoke the reader key; delete the private repos after the pull)"

# 5. checks ----------------------------------------------------------------------------------------------------------------------
fail=0; chk(){ [ -e "$1" ] || { log "MISSING $1"; fail=1; }; }
for f in scripts/train_decision_lora_torch.py scripts/evaluate_decision_model_torch.py scripts/playground/server.py scripts/imajev_bench/run_local_v2.py \
         scripts/p3/eval_checkpoint.py scripts/p3/parity_readout256.py scripts/p3/transfer.py cloud/p3/pod_run_train.sh cloud/p3/pilot_decision.py \
         cloud/p3/select_checkpoint.py cloud/p2c_gates_reference.json; do chk $f; done
V=${P3_MANIFEST:-decision-p3}
for m in $V ${P3_HELDOUT_FRESH:-decision-p3-heldout-fresh} ${P3_HELDOUT_FLAGGED:-decision-p3-heldout-flagged} ${P3_HUMAN_DEV:-decision-p3-human-dev} \
         decision-p2b decision-v2-state-probe decision-v2-pairs-probe decision-v1.1-irrelevance-test decision-p2b-jevstyle-dev; do
  chk data/manifests/$m.jsonl; done
chk bench/records/records-eval.jsonl; chk $M4/config.json; chk adapters/4b-shipped/decision_readout.safetensors
chk p3/tracking-only/decisionbench-3k.jsonl; chk evalsets/jevbench/datasets/public/hard.jsonl
[ -n "$(ls evalsets/fast-decisions/*.jsonl 2>/dev/null)" ] || { log "MISSING fast-decisions dev files"; fail=1; }
cd 
python scripts/p3/mine.py --help > out/mine-help.txt 2>&1 && log "scripts/p3/mine.py present" \
  || log "WARNING scripts/p3/mine.py missing or broken: round 2 will FAIL (see out/mine-help.txt)"
python - <<'PY' || fail=1
import json, os
root = ""
for name in ("decision-v1.1-irrelevance-test", "decision-v2-state-probe", "decision-v2-pairs-probe"):
    p = f"{root}/data/manifests/{name}.jsonl"
    if not os.path.exists(p): continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    imgs = [i["image"] for r in rows for i in r.get("images", [])]; miss = [x for x in imgs if not os.path.exists(os.path.join(root, x))]
    print(f"{name}: {len(imgs)} images, {len(miss)} missing"); assert not miss, f"{name}: image files missing"
PY
[ $fail = 0 ] || die "checks failed"
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c; df -h /workspace | tail -1
echo BOOTSTRAP_DONE
if [ "$RUN_AFTER" = 1 ]; then
  cd /workspace && setsid nohup bash cloud/p3/pod_run_train.sh >> p3/run/run.log 2>&1 < /dev/null &
  log "started cloud/p3/pod_run_train.sh (log p3/run/run.log)"
fi
