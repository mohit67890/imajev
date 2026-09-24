#!/usr/bin/env bash
# DecisionBench 1.0 (Hanno-Labs/decision-bench, 23,900 rows) for imajev-4b on one H100.
# Everything comes from public pinned sources: imajev public repo, HF adapter revision, DecisionBench repo + dataset.
# Three torch servers share the GPU behind an nginx round-robin; the harness runs with concurrency 3.
# Log: db/run.log, marker ALL_DONE (or FAILED). Result tarball: decisionbench-imajev-4b.tgz
set -uo pipefail
IMAJEV_COMMIT=4e623a8a51aa319405a0f32e6b73a8ce73416800
ADAPTER_REV=712891d1192c6491441a19bdea254d8a6e1048d0
DB_COMMIT=47ea5a479e35fd5ac7fde5c72103143071d7d93f
MAX_TOKENS=${MAX_TOKENS:-32768}
SERVERS=${SERVERS:-3}
mkdir -p db && exec > >(tee -a db/run.log) 2>&1
trap 'echo FAILED line $LINENO' ERR
set -e
export HF_HOME=hf
cd /workspace
date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# imajev server
git clone -q https://github.com/mohit67890/imajev && (cd imajev && git checkout -q $IMAJEV_COMMIT)
python -m venv venv --system-site-packages && . venv/bin/activate
pip install -q -e "imajev[serve,torch]" "transformers==5.17.0" "peft==0.21.0" 2>&1 | tail -2
(cd imajev && python scripts/download_model.py --model 4b)
hf download mohit67890/imajev-4b --revision $ADAPTER_REV --local-dir adapters/imajev-4b --exclude "mlx/*" --exclude "assets/*" > /dev/null
sha256sum adapters/imajev-4b/adapter_model.safetensors
cd imajev
for i in $(seq 0 $((SERVERS-1))); do
  PYTHONPATH=src:scripts nohup python scripts/playground/server.py --backend torch --model-bundle artifacts/model-qwen4b.json \
    --adapter ../adapters/imajev-4b --calibration ../adapters/imajev-4b/calibration.json --rotations 4 \
    --max-input-tokens $MAX_TOKENS --model-name imajev-4b --host <host> --port $((8765+i)) > db/server$i.log 2>&1 &
done
cd /workspace; deactivate

# round-robin proxy
apt-get install -y -qq nginx > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nginx > /dev/null)
{ echo "events {} http { client_max_body_size 64m; proxy_read_timeout 900s; upstream s1 {"
  for i in $(seq 0 $((SERVERS-1))); do echo "server <host>:$((8765+i));"; done
  echo "} server { listen 8800; location / { proxy_pass http://s1; } } }"; } > /etc/nginx/nginx.conf
nginx -t && (nginx -s reload 2>/dev/null || nginx)

# DecisionBench
venv/bin/pip install -q uv && export PATH=venv/bin:$PATH
git clone -q https://github.com/Hanno-Labs/decision-bench && (cd decision-bench && git checkout -q $DB_COMMIT && uv sync -q)
for i in $(seq 0 $((SERVERS-1))); do
  until curl -sf http://<host>:$((8765+i))/v1/models > /dev/null; do sleep 5; done
done
echo SERVERS_READY; date -u
cd decision-bench
ARGS=(task_specs/decisionbench-dev.toml --base-url http://<host> --model imajev-4b
  --model-repo mohit67890/imajev-4b --model-revision $ADAPTER_REV
  --serving-bundle-sha256 d8d328f85dcda4459a2331c47aba2a89ec808210e712248afdb69e38b9e0c25d
  --inference-image "github.com/mohit67890/imajev@$IMAJEV_COMMIT scripts/playground/server.py --backend torch --rotations 4 --calibration --max-input-tokens $MAX_TOKENS"
  --concurrency $SERVERS --max-candidates 254)
.venv/bin/decision-bench run-system-one-http "${ARGS[0]}" db/smoke "${ARGS[@]:1}" --smoke | tail -5
echo SMOKE_DONE; date -u
.venv/bin/decision-bench run-system-one-http "${ARGS[0]}" db/full "${ARGS[@]:1}" > db/full-summary.json
date -u
cd /workspace && tar czf decisionbench-imajev-4b.tgz db
echo ALL_DONE
