#!/usr/bin/env bash
# Restart the DecisionBench full run on a pod already bootstrapped by pod_decisionbench_p3.sh, after installing the fast DeltaNet kernels.
set -uo pipefail; cd /workspace; L=db/run.log
echo "RESTART $(date -u) : installing flash-linear-attention + tilelang, restarting servers and the full run (no resume in the harness)" >> $L
pkill -f "run-system-one-http" 2>/dev/null; pkill -f "scripts/playground/server.py" 2>/dev/null; sleep 5
mv db/full db/full-partial-slowpath 2>/dev/null; rm -f db/full-summary.json
. venv/bin/activate
pip install -q flash-linear-attention tilelang 2>&1 | tail -1; python -c "import fla, tilelang; print('fla', fla.__version__, 'tilelang ok')" >> $L 2>&1 || { echo "FAILED fla install" >> $L; exit 1; }
NGPU=$(nvidia-smi -L | grep -c '^GPU'); SERVERS_PER_GPU=${SERVERS_PER_GPU:-3}; SERVERS=$((NGPU*SERVERS_PER_GPU)); MAX_TOKENS=${MAX_TOKENS:-32768}
AD=adapters/imajev-4b; CAL_FILE=${CAL_FILE:-calibration-rot4.json}; ADAPTER_SHA=$(sha256sum $AD/adapter_model.safetensors | cut -d' ' -f1)
cd ; idx=0
for g in $(seq 0 $((NGPU-1))); do for j in $(seq 1 $SERVERS_PER_GPU); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src:scripts nohup python scripts/playground/server.py --backend torch --model-bundle artifacts/model-qwen4b.json \
    --adapter $AD --calibration $AD/$CAL_FILE --rotations 4 --max-input-tokens $MAX_TOKENS --model-name imajev-4b --host <host> --port $((8765+idx)) > db/server$idx.log 2>&1 &
  idx=$((idx+1)); done; done
cd /workspace; deactivate; export PATH=venv/bin:$PATH
for i in $(seq 0 $((SERVERS-1))); do until curl -sf http://<host>:$((8765+i))/v1/models > /dev/null; do sleep 5; done; done
T1=$(date +%s); echo "SERVERS_READY (restart, fla)" >> $L; date -u >> $L
cd decision-bench
.venv/bin/decision-bench run-system-one-http task_specs/decisionbench-dev.toml db/full --base-url http://<host> --model imajev-4b \
  --model-repo mohit67890/imajev-4b --model-revision ${ADAPTER_REV:-local-unpublished-p3-r2-s000291} --serving-bundle-sha256 $ADAPTER_SHA \
  --inference-image "github.com/mohit67890/imajev@${IMAJEV_COMMIT:-9b66a662dd52bbd5bfe0febf3792359f24db606f} scripts/playground/server.py --backend torch --rotations 4 --calibration $CAL_FILE --max-input-tokens $MAX_TOKENS" \
  --concurrency $SERVERS --max-candidates 255 > db/full-summary.json 2>> $L || { echo "FAILED harness" >> $L; exit 1; }
T2=$(date +%s); date -u >> $L
python3 - <<PY
import json
meta={"gpus": $NGPU, "servers": $SERVERS, "servers_per_gpu": $SERVERS_PER_GPU, "concurrency": $SERVERS, "max_candidates": 255, "max_input_tokens": $MAX_TOKENS,
      "calibration_file": "$CAL_FILE", "adapter_sha256": "$ADAPTER_SHA", "adapter_revision": "${ADAPTER_REV:-local-unpublished-p3-r2-s000291}",
      "imajev_commit": "${IMAJEV_COMMIT:-9b66a662dd52bbd5bfe0febf3792359f24db606f}", "run_seconds": $((T2-T1)), "note": "restarted with flash-linear-attention + tilelang; the first (slow-path) partial run is in db/full-partial-slowpath"}
json.dump(meta, open("db/run-meta.json","w"), indent=1)
PY
cd /workspace && tar czf decisionbench-imajev-4b-p3.tgz db && echo ALL_DONE >> $L && echo ALL_DONE
