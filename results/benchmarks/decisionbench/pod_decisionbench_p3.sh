#!/usr/bin/env bash
# DecisionBench 1.0 (Hanno-Labs/decision-bench, 23,900 rows) for the phase-3 imajev-4b on ONE pod with ANY number of GPUs.
# Every GPU runs SERVERS_PER_GPU torch servers (default 3) behind one nginx round-robin; the unmodified harness runs with
# concurrency = total servers, so wall time scales ~1/GPUs (1×H100 ≈ 3.5 h, 4×H100 ≈ 1 h, 8×H100 ≈ 30 min; ~$15 either way).
# Everything comes from pinned public sources unless ADAPTER_LOCAL points at a copied adapter dir (dry runs before the HF upload).
#   required:  IMAJEV_COMMIT=<public repo commit with the phase-3 server code>   ADAPTER_REV=<HF revision of mohit67890/imajev-4b>
#   optional:  ADAPTER_LOCAL=adapters/imajev-4b (skip the HF download)  SERVERS_PER_GPU=3  MAX_TOKENS=32768
#              MAX_CANDIDATES=255 (256-code readout: 255 options + unknown; the previous release needed 254)  CAL_FILE=calibration-rot4.json
#              DB_COMMIT=<decision-bench harness commit>  SMOKE=1
# Launch:  IMAJEV_COMMIT=… ADAPTER_REV=… setsid nohup bash pod_decisionbench_p3.sh > launch.log 2>&1 < /dev/null &
# Log db/run.log, markers SERVERS_READY / SMOKE_DONE / ALL_DONE (or FAILED). Result: decisionbench-imajev-4b-p3.tgz
set -uo pipefail
: "${IMAJEV_COMMIT:?set IMAJEV_COMMIT (public repo commit)}"
ADAPTER_LOCAL=${ADAPTER_LOCAL:-}; [ -n "$ADAPTER_LOCAL" ] || : "${ADAPTER_REV:?set ADAPTER_REV (HF revision) or ADAPTER_LOCAL}"
ADAPTER_REV=${ADAPTER_REV:-local-unpublished}
DB_COMMIT=${DB_COMMIT:-47ea5a479e35fd5ac7fde5c72103143071d7d93f}
MAX_TOKENS=${MAX_TOKENS:-32768}; SERVERS_PER_GPU=${SERVERS_PER_GPU:-3}; MAX_CANDIDATES=${MAX_CANDIDATES:-255}; CAL_FILE=${CAL_FILE:-calibration-rot4.json}; SMOKE=${SMOKE:-1}
mkdir -p db && exec > >(tee -a db/run.log) 2>&1
trap 'echo FAILED line $LINENO; echo FAILED > db/FAILED' ERR
set -e
export HF_HOME=hf
cd /workspace
T0=$(date +%s); date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NGPU=$(nvidia-smi -L | grep -c '^GPU'); SERVERS=$((NGPU*SERVERS_PER_GPU)); echo "GPUs $NGPU x $SERVERS_PER_GPU servers = $SERVERS (harness concurrency $SERVERS)"

# imajev server (public repo at the pinned commit)
git clone -q https://github.com/mohit67890/imajev && (cd imajev && git checkout -q $IMAJEV_COMMIT)
python -m venv venv --system-site-packages && . venv/bin/activate
pip install -q -e "imajev[serve,torch]" "transformers==5.17.0" "peft==0.21.0" 2>&1 | tail -2
(cd imajev && python scripts/download_model.py --model 4b)
if [ -n "$ADAPTER_LOCAL" ]; then AD=$ADAPTER_LOCAL; echo "adapter: local $AD (revision recorded as $ADAPTER_REV)"
else hf download mohit67890/imajev-4b --revision $ADAPTER_REV --local-dir adapters/imajev-4b --exclude "mlx/*" --exclude "assets/*" > /dev/null; AD=adapters/imajev-4b; fi
ADAPTER_SHA=$(sha256sum $AD/adapter_model.safetensors | cut -d' ' -f1); echo "adapter sha256 $ADAPTER_SHA"; [ -f $AD/$CAL_FILE ] || { echo "missing $AD/$CAL_FILE"; exit 1; }
python - "$AD" <<'PY'
import json, sys; from safetensors import safe_open
a=sys.argv[1]; c=json.load(open(f"{a}/adapter_config.json"))
with safe_open(f"{a}/decision_readout.safetensors","pt") as f: rows=f.get_slice("weight").get_shape()[0]
print(f"adapter: LoRA r{c['r']} alpha {c['lora_alpha']}, readout {rows} codes -> max candidates {rows-1}")
PY
cd imajev; idx=0
for g in $(seq 0 $((NGPU-1))); do for j in $(seq 1 $SERVERS_PER_GPU); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src:scripts nohup python scripts/playground/server.py --backend torch --model-bundle artifacts/model-qwen4b.json \
    --adapter $AD --calibration $AD/$CAL_FILE --rotations 4 --max-input-tokens $MAX_TOKENS --model-name imajev-4b \
    --host <host> --port $((8765+idx)) > db/server$idx.log 2>&1 &
  idx=$((idx+1)); done; done
cd /workspace; deactivate

# round-robin proxy over every server
apt-get install -y -qq nginx > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nginx > /dev/null)
{ echo "events { worker_connections 4096; } http { client_max_body_size 64m; proxy_read_timeout 900s; upstream s1 {"
  for i in $(seq 0 $((SERVERS-1))); do echo "server <host>:$((8765+i));"; done
  echo "} server { listen 8800; location / { proxy_pass http://s1; } } }"; } > /etc/nginx/nginx.conf
nginx -t && (nginx -s reload 2>/dev/null || nginx)

# DecisionBench harness (unmodified, pinned)
venv/bin/pip install -q uv && export PATH=venv/bin:$PATH
git clone -q https://github.com/Hanno-Labs/decision-bench && (cd decision-bench && git checkout -q $DB_COMMIT && uv sync -q)
for i in $(seq 0 $((SERVERS-1))); do until curl -sf http://<host>:$((8765+i))/v1/models > /dev/null; do sleep 5; done; done
T1=$(date +%s); echo SERVERS_READY "(setup $(( (T1-T0)/60 )) min)"; date -u
cd decision-bench
ARGS=(task_specs/decisionbench-dev.toml --base-url http://<host> --model imajev-4b
  --model-repo mohit67890/imajev-4b --model-revision $ADAPTER_REV
  --serving-bundle-sha256 $ADAPTER_SHA
  --inference-image "github.com/mohit67890/imajev@$IMAJEV_COMMIT scripts/playground/server.py --backend torch --rotations 4 --calibration $CAL_FILE --max-input-tokens $MAX_TOKENS"
  --concurrency $SERVERS --max-candidates $MAX_CANDIDATES)
if [ "$SMOKE" = 1 ]; then .venv/bin/decision-bench run-system-one-http "${ARGS[0]}" db/smoke "${ARGS[@]:1}" --smoke | tail -5; echo SMOKE_DONE; date -u; fi
.venv/bin/decision-bench run-system-one-http "${ARGS[0]}" db/full "${ARGS[@]:1}" > db/full-summary.json
T2=$(date +%s); date -u
python3 - <<PY
import json; s=json.load(open("db/full-summary.json")) if open("db/full-summary.json").read().strip() else {}
meta={"gpus": $NGPU, "servers": $SERVERS, "servers_per_gpu": $SERVERS_PER_GPU, "concurrency": $SERVERS, "max_candidates": $MAX_CANDIDATES, "max_input_tokens": $MAX_TOKENS,
      "calibration_file": "$CAL_FILE", "adapter_sha256": "$ADAPTER_SHA", "adapter_revision": "$ADAPTER_REV", "imajev_commit": "$IMAJEV_COMMIT", "db_commit": "$DB_COMMIT",
      "setup_seconds": $((T1-T0)), "run_seconds": $((T2-T1))}
json.dump(meta, open("db/run-meta.json","w"), indent=1); print(json.dumps(meta))
PY
cd /workspace && tar czf decisionbench-imajev-4b-p3.tgz db
echo ALL_DONE
