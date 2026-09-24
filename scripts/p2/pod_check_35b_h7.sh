#!/usr/bin/env bash
# Frozen Qwen3.6-35B-A3B on public JevBench (hard/original/easy) through vLLM on GPU 6 of the 7xH100 pod, structured
# generation via the jevbench openai_compat adapter, thinking off then thinking on. Decides whether a 35B-A3B phase-2b tier
# is worth training. Runs alongside pod_run_p2b_gen_h7.sh (which owns GPUs 0-5 and ports 8000/8010/8020); this owns :8030.
set -uo pipefail
P2=${P2:-p2}; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_COMPAT_API_KEY=local PATH=vllm-venv/bin:$PATH
O=$P2/out/p2b; R=$P2/out/check35b; mkdir -p $O $R; touch $O/check35b.STARTED
M=hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
log(){ echo "$(date +%H:%M:%S) $*"; }
stop(){ pkill -f "vllm.entrypoints.*--port 8030" 2>/dev/null || true; sleep 8; }
serve(){ # $1 log-tag, rest = extra vllm args
  local tag=$1; shift
  (CUDA_VISIBLE_DEVICES=6 vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model $M --served-model-name qwen3.6-35b-a3b --port 8030 \
     --max-model-len 16384 --max-num-seqs 48 --gpu-memory-utilization 0.95 "$@" > $R/vllm-$tag.log 2>&1 &)
  for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:8030/v1/models | grep -q qwen3.6-35b && return 0
    grep -qE "Traceback|Error" $R/vllm-$tag.log && { sleep 10; curl -s -m 2 http://127.0.0.1:8030/v1/models | grep -q qwen3.6-35b && return 0; break; }; sleep 5; done
  log "serve $tag failed"; tail -n 8 $R/vllm-$tag.log | cut -c1-200; stop; return 1; }
bench(){ # $1 mode label
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$1/$split; [ -f $R/$1/$split/summary.json ] && continue
      python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter openai_compat --endpoint http://127.0.0.1:8030/v1 --model qwen3.6-35b-a3b \
        --key-env OPENAI_COMPAT_API_KEY --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$1/$split/results.jsonl --raw-dir $R/$1/$split/raw \
        --manifest $R/$1/$split/manifest.json > $R/$1/$split/run.log 2>&1 || { tail -n 4 $R/$1/$split/run.log; log "RUN_FAILED $1 $split"; continue; }
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$1/$split/results.jsonl --public-export $R/$1/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$1/$split/summary.json'));print('  35b-a3b frozen $1 jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))"; done ) }
# thinking off: try the chat-template kwarg flag (vLLM >= 0.10), bf16 first, fp8 weights as the fallback for memory
log "serve 35B-A3B bf16, thinking off"
serve nothink --default-chat-template-kwargs '{"enable_thinking": false}' \
 || { log "bf16 failed, retry with fp8 weights"; serve nothink-fp8 --quantization fp8 --default-chat-template-kwargs '{"enable_thinking": false}' || { log CHECK35B_FAILED; touch $O/check35b.DONE; exit 1; }; }
Q=$(grep -l "quantization fp8\|fp8" $R/vllm-nothink-fp8.log 2>/dev/null | wc -l); log "weights: $([ $Q -gt 0 ] && echo fp8 || echo bf16)"
log "stage nothink"; bench nothink; stop
log "serve 35B-A3B, thinking on"
serve think --reasoning-parser qwen3 $([ $Q -gt 0 ] && echo --quantization fp8) || { log CHECK35B_THINK_FAILED; touch $O/check35b.DONE; exit 1; }
log "stage think"; bench think; stop
touch $O/check35b.DONE; log CHECK35B_DONE
