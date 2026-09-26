#!/usr/bin/env bash
# Phase 3 Stage 1 (streamed mining, docs/phase-3-plan.md "Streamed mining + teacher"): bootstrap the SMALL mining pod
# (4 GPUs, A100 / L40S class) and start one scripts/p3/mine.py worker per GPU over the pool shards with the shipped imajev-4b.
# mine.py does the flagging and writes append-only files; nothing on the pod talks to Azure: the Mac's stream_coordinator.py
# pulls the flagged rows over ssh/rsync and feeds the teacher. This script only sets the pod up, launches the workers and writes markers.
# Small files scp'd from the Mac first: bootstrap_mine.sh, transfer.py, p3.env (no secrets:
# BACKEND (default gcs), GCS_BUCKET (default gs://<bucket>/) / GCS_PREFIX (default phase3) or MINE_REPO (hf), MINE_BUNDLE,
# optional MINE_SHARDS / MINE_OUT / MINE_CMD / MINE_ARGS / NWORKERS), and ONE secret, deleted here right after the fetch:
# reader-key.json (gcs; a short-lived bucket reader key) or hf-token (hf).
# The shipped adapter and Qwen3.5-4B come from their public Hugging Face repos (no token needed), sha256-checked.
# Launch: cd /workspace && setsid nohup bash bootstrap_mine.sh > mine-bootstrap.log 2>&1 < /dev/null &
# mine.py writes where the Mac's stream_coordinator.py pulls from (reports/phase3/streaming-runbook.md): data/p3/mine
# (<shard>.scores.jsonl, <shard>.flagged.jsonl, <shard>.done, progress/). Workers claim shards themselves (flock), so there is one
# worker per GPU and no shard arithmetic here.
# Markers: p3-mine/WORKERS_RUNNING (every worker has scored its first chunk: the signal to deploy the Azure teacher),
# p3-mine/MINE_DONE (every worker exited 0 AND every shard has its .done) or p3-mine/MINE_FAILED;
# per worker worker-<i>.DONE / .FAILED (logs worker-<i>.log); p3-mine-results.tgz (mining output + logs).
# Nothing here terminates the pod.
set -uo pipefail; cd /workspace
R=p3-mine; mkdir -p $R/out  adapters fetch
log(){ echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST mine-bootstrap: $*"; }
die(){ log "MINE_FAILED: $*"; rm -f hf-token reader-key.json; touch $R/MINE_FAILED; echo MINE_FAILED; exit 1; }
rm -f $R/MINE_DONE $R/MINE_FAILED $R/WORKERS_RUNNING
[ -f p3.env ] || die "missing p3.env"; set -a; . p3.env; set +a
BACKEND=${BACKEND:-gcs}; MINE_BUNDLE=${MINE_BUNDLE:-p3-mine}; GCS_BUCKET=${GCS_BUCKET:-gs://<bucket>/}; GCS_PREFIX=${GCS_PREFIX:-phase3}
QWEN_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; SHIPPED_REPO=${SHIPPED_REPO:-mohit67890/imajev-4b}; SHIPPED_REV=${SHIPPED_REV:-712891d1192c6491441a19bdea254d8a6e1048d0}
SHIPPED_SHA=d8d328f85dcda4459a2331c47aba2a89ec808210e712248afdb69e38b9e0c25d
MINE_SHARDS=${MINE_SHARDS:-data/p3/pool/shards/*.jsonl}; MINE_OUT=${MINE_OUT:-data/p3/mine}
MINE_CMD=${MINE_CMD:-"python scripts/p3/mine.py --backend torch --shards '{shards}' --out {out} --model-path {model} --adapter {adapter} --worker-id gpu{gpu} --device cuda"}
MINE_ARGS=${MINE_ARGS:-}
case $BACKEND in
  hf)  [ -s hf-token ] || die "missing hf-token"; : "${MINE_REPO:?MINE_REPO}"; FETCH_ARGS=(--backend hf --token-env HF_TOKEN --repo $MINE_REPO) ;;
  gcs) [ -s reader-key.json ] || die "missing reader-key.json"; chmod 600 reader-key.json
       export GOOGLE_APPLICATION_CREDENTIALS=reader-key.json; : "${GCS_BUCKET:?GCS_BUCKET}"
       FETCH_ARGS=(--backend gcs --bucket $GCS_BUCKET --prefix $GCS_PREFIX) ;;
  *) die "BACKEND must be hf or gcs" ;;
esac
if [ -s hf-token ]; then HF_TOKEN=$(cat hf-token); export HF_TOKEN; fi
export HF_HOME=hf

if [ "${SKIP_VENV:-0}" != 1 ]; then
  python -m venv venv --system-site-packages && . venv/bin/activate || die venv
  pip install -q "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" "safetensors>=0.8" "pillow>=11" "pydantic>=2,<3" httpx pyarrow \
    "huggingface_hub>=1.32" google-cloud-storage flash-linear-attention tilelang 2>&1 | tail -n 1
else . venv/bin/activate; fi
python - > download-models.log 2>&1 <<PY &
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-4B", revision="$QWEN_REV", ignore_patterns=["*.md", ".gitattributes", "original/*"], max_workers=16), flush=True)
print(snapshot_download("$SHIPPED_REPO", revision="$SHIPPED_REV", local_dir="adapters/4b-shipped", ignore_patterns=["mlx/*", "assets/*"]), flush=True)
PY
PM=$!
python transfer.py fetch "${FETCH_ARGS[@]}" --name $MINE_BUNDLE --dest fetch --extract  || die "fetch $MINE_BUNDLE"
rm -rf fetch
wait $PM || { tail -n 20 download-models.log; die "model download"; }
M4=hf/hub/models--Qwen--Qwen3.5-4B/snapshots/$QWEN_REV; echo "$M4" > model_path
got=$(sha256sum adapters/4b-shipped/adapter_model.safetensors | cut -d' ' -f1); [ "$got" = "$SHIPPED_SHA" ] || die "shipped adapter sha256 $got"
rm -f hf-token reader-key.json; unset HF_TOKEN
cd ; export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NSH=$(ls $MINE_SHARDS 2>/dev/null | wc -l); [ "$NSH" -gt 0 ] || die "no pool shards match $MINE_SHARDS in the bundle"
python scripts/p3/mine.py --help > $R/mine-help.txt 2>&1 || die "scripts/p3/mine.py missing or broken (see $R/mine-help.txt)"
NGPU=$(nvidia-smi -L | grep -c '^GPU'); N=${NWORKERS:-$NGPU}; mkdir -p $MINE_OUT
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
log "BOOTSTRAP_DONE: starting $N mine.py workers on $NGPU GPUs over $NSH shards ($MINE_SHARDS) -> $MINE_OUT"
pids=()
for i in $(seq 0 $((N-1))); do
  AD=adapters/4b-shipped; cmd=${MINE_CMD//\{shards\}/$MINE_SHARDS}; cmd=${cmd//\{out\}/$MINE_OUT}; cmd=${cmd//\{model\}/$M4}
  cmd=${cmd//\{adapter\}/$AD}; cmd=${cmd//\{gpu\}/$i}
  log "worker $i on GPU $((i % NGPU)): $cmd $MINE_ARGS"
  ( CUDA_VISIBLE_DEVICES=$((i % NGPU)) bash -c "$cmd $MINE_ARGS" > $R/worker-$i.log 2>&1 && touch $R/worker-$i.DONE || touch $R/worker-$i.FAILED ) & pids+=($!)
done
# WORKERS_RUNNING: every worker's heartbeat shows scored items (mine.py progress/<worker>.json), or a worker already finished
( for k in $(seq 1 360); do
    n=$(python - "$MINE_OUT/progress" <<'PY'
import glob, json, sys
n = 0
for p in glob.glob(sys.argv[1] + "/gpu*.json"):
    try:
        s = json.load(open(p)); n += bool(s.get("items") or s.get("finished"))
    except Exception:
        pass
print(n)
PY
)
    if [ "$n" -ge "$N" ]; then touch $R/WORKERS_RUNNING; log "WORKERS_RUNNING: $n workers scoring (deploy the Azure teacher now)"; exit 0; fi
    ls $R/worker-*.FAILED > /dev/null 2>&1 && { log "a worker failed before WORKERS_RUNNING: $(ls $R/worker-*.FAILED | xargs -n1 basename | tr '\n' ' ')"; exit 1; }
    sleep 10; done; log "WARNING: workers not all running after 60 min" ) &
for p in "${pids[@]}"; do wait $p; done
NDONE=$(ls $MINE_OUT/*.done 2>/dev/null | wc -l)
if ls $R/worker-*.FAILED > /dev/null 2>&1; then log "workers failed: $(ls $R/worker-*.FAILED | xargs -n1 basename | tr '\n' ' ')"; touch $R/MINE_FAILED
elif [ "$NDONE" -lt "$NSH" ]; then log "only $NDONE of $NSH shards done"; touch $R/MINE_FAILED
else touch $R/MINE_DONE; fi
tar czf p3-mine-results.tgz -C /workspace p3-mine imajev/data/p3/mine && ls -la p3-mine-results.tgz
log "$( [ -f $R/MINE_DONE ] && echo MINE_DONE || echo MINE_FAILED )"
