#!/usr/bin/env bash
# Faster answer stage: wait for the writer, then run BOTH answerers at once (Qwen3.6-27B TP=2 on GPUs 0-1, gpt-oss-20b TP=2 on GPUs 2-3),
# then assemble and pack. Replaces the sequential answer stages of pod_run_p2_gen.sh.
set -uo pipefail
P2=${P2:-p2}; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_API_KEY=local PATH=vllm-venv/bin:$PATH
O=$P2/out/p2; D=data/decision-p2/teacher; WORKERS=${WORKERS:-192}; MAXTOK=${MAXTOK:-5000}
log(){ echo "$(date +%H:%M:%S) $*"; }; skip(){ [ -f $O/$1.DONE ]; }
QW=hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
GO=hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
until skip write; do sleep 20; done; log "writer done: $(wc -l < $D/writer.jsonl) documents"
pkill -f pod_run_p2_gen.sh 2>/dev/null; pkill -f "vllm.entrypoints" 2>/dev/null; sleep 10
serve(){ local gpus=$1 path=$2 name=$3 port=$4; shift 4
  curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "$name" && { log "  $name already serving on $port"; return 0; }
  (CUDA_VISIBLE_DEVICES=$gpus vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model "$path" --served-model-name "$name" --port $port \
     --max-model-len 24576 --gpu-memory-utilization 0.92 --max-num-seqs 192 --tensor-parallel-size 2 "$@" > $O/vllm-$name-par.log 2>&1 &)
  for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "$name" && return 0; sleep 5; done
  log "FAILED vllm $name"; tail -n 20 $O/vllm-$name-par.log; return 1; }
if ! skip answer-qwen || ! skip answer-gptoss; then
  log "serve qwen on GPUs 0,1 and gpt-oss on GPUs 2,3"
  serve 0,1 $QW qwen3.6-27b 8000 --reasoning-parser qwen3 & PQ=$!
  serve 2,3 $GO gpt-oss-20b 8010 --reasoning-parser openai_gptoss & PG=$!
  wait $PQ || exit 1; wait $PG || exit 1
  log "stage answer (parallel)"
  ( skip answer-qwen || { python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer.jsonl --out $D/answers-qwen.jsonl --workers $WORKERS --max-tokens $MAXTOK --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 > $O/answer-qwen.log 2>&1 && touch $O/answer-qwen.DONE || log ANSWER_QWEN_FAILED; } ) &
  ( skip answer-gptoss || { python scripts/p2/gen_answer.py --answerer gptoss --writer $D/writer.jsonl --out $D/answers-gptoss.jsonl --workers $WORKERS --max-tokens $MAXTOK --model gpt-oss-20b --base-url http://127.0.0.1:8010/v1 > $O/answer-gptoss.log 2>&1 && touch $O/answer-gptoss.DONE || log ANSWER_GPTOSS_FAILED; } ) &
  wait; pkill -f "vllm.entrypoints" 2>/dev/null; sleep 5
  tail -n 1 $O/answer-qwen.log; tail -n 1 $O/answer-gptoss.log
  skip answer-qwen && skip answer-gptoss || { log "ANSWER_STAGE_FAILED"; exit 1; }; fi
if ! skip assemble; then log "stage assemble"
  python scripts/p2/assemble_p2.py --writer $D/writer.jsonl --answers $D/answers-qwen.jsonl $D/answers-gptoss.jsonl --out $D/records.jsonl \
     --reference $P2/jevbench-public/*.jsonl > $O/assemble.log 2>&1 || { tail -n 5 $O/assemble.log; log "ASSEMBLE_FAILED"; exit 1; }
  tail -n 3 $O/assemble.log; touch $O/assemble.DONE; fi
log pack; tar czf $P2/decision-p2-teacher.tgz -C data decision-p2 && ls -la $P2/decision-p2-teacher.tgz; echo ALL_DONE
