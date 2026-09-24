#!/usr/bin/env bash
# Same protocol as our own jevbench() step: serve a TypeSafe-compatible /v1/systemone server on one GPU, run the public
# hard/original/easy items serially through the jevbench `typesafe` adapter (native probabilities, no rotations, no calibration
# of ours), summarize, print the same summary lines. Usage: competitor_bench.sh <name> <gpu> <port> <serve command...>
set -uo pipefail
NAME=$1; GPU=$2; PORT=$3; shift 3
P2=p2; R=$P2/out/$NAME; mkdir -p $R; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts TYPESAFE_API_KEY=local HF_HOME=hf
log(){ echo "$(date +%H:%M:%S) $NAME: $*"; }
log "serve on GPU $GPU :$PORT -> $*"
(CUDA_VISIBLE_DEVICES=$GPU HF_HOME=hf setsid nohup "$@" > $R/server.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:$PORT/v1/systemone -H "Content-Type: application/json" -H "Authorization: Bearer local" -d '{}'); echo "$code" | grep -qE "^(200|400|422)$" && break; sleep 5; done
log "server probe: $code"; [ "$code" != "000" ] || { tail -n 8 $R/server.log | cut -c1-160; log SERVE_FAILED; exit 1; }
( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split; [ -f $R/$split/summary.json ] && continue
    python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:$PORT --model $NAME --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 \
      --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1 || { tail -n 4 $R/$split/run.log | cut -c1-160; log "RUN_FAILED $split"; continue; }
    python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
    python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  $NAME jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3),'p50',round(d['latency']['p50_s'],3),'s')"; done )
pkill -f -- "--port $PORT" 2>/dev/null; log "${NAME^^}_BENCH_DONE"
