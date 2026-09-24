#!/usr/bin/env bash
# decision-p2 teacher generation on ONE H100 80 GB. Stages with DONE markers (rerun skips finished stages):
#   write (Qwen3.6-27B, thinking) → answer-qwen (same server) → answer-gptoss (gpt-oss-20b, reasoning medium) → assemble
# Expects:  (code), a venv with pydantic + huggingface_hub at venv, vLLM at vllm-venv,
# HF_HOME=hf with the two models snapshotted at the pinned revisions (bootstrap does `hf download`).
set -uo pipefail
P2=${P2:-p2}; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local
O=$P2/out/p2; D=data/decision-p2/teacher; mkdir -p $O $D
DOCS=${DOCS:-6000}; WORKERS=${WORKERS:-128}; TP=${TP:-1}
WRITER_REPO=${WRITER_REPO:-Qwen/Qwen3.6-27B}; WRITER_REV=${WRITER_REV:-6a9e13bd6fc8}
GPTOSS_REPO=${GPTOSS_REPO:-openai/gpt-oss-20b}; GPTOSS_REV=${GPTOSS_REV:-6cee5e81ee83}
skip(){ [ -f $O/$1.DONE ]; }; log(){ echo "$(date +%H:%M:%S) $*"; }
snap(){ python -c "from huggingface_hub import snapshot_download;print(snapshot_download('$1',revision='$2'))" 2>/dev/null | tail -1; }
serve(){ # $1 model path, $2 served name, rest = extra vllm args
  local path=$1 name=$2; shift 2
  (vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model "$path" --served-model-name "$name" --port 8000 \
     --max-model-len 24576 --gpu-memory-utilization 0.92 --max-num-seqs $((64*TP)) --tensor-parallel-size $TP "$@" > $O/vllm-$name.log 2>&1 &)
  for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q "$name" && return 0; sleep 5; done
  log "FAILED vllm $name"; tail -n 20 $O/vllm-$name.log; return 1; }
stop(){ pkill -f "vllm.entrypoints" || true; sleep 8; }

if ! skip write || ! skip answer-qwen; then
  W=$(snap $WRITER_REPO $WRITER_REV); log "serve writer $W"; serve "$W" qwen3.6-27b --reasoning-parser qwen3 || exit 1
  if ! skip write; then log "stage write: $DOCS documents"
    python scripts/p2/gen_write.py --docs $DOCS --out $D/writer.jsonl --workers $WORKERS --model qwen3.6-27b > $O/write.log 2>&1 || { tail -n 5 $O/write.log; log "WRITE_FAILED"; stop; exit 1; }
    tail -n 1 $O/write.log; touch $O/write.DONE; fi
  if ! skip answer-qwen; then log "stage answer-qwen"
    python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer.jsonl --out $D/answers-qwen.jsonl --workers $WORKERS --model qwen3.6-27b > $O/answer-qwen.log 2>&1 || { tail -n 5 $O/answer-qwen.log; log "ANSWER_QWEN_FAILED"; stop; exit 1; }
    tail -n 1 $O/answer-qwen.log; touch $O/answer-qwen.DONE; fi
  stop
fi

if ! skip answer-gptoss; then
  G=$(snap $GPTOSS_REPO $GPTOSS_REV); log "serve gpt-oss $G"; serve "$G" gpt-oss-20b --reasoning-parser openai_gptoss || exit 1
  log "stage answer-gptoss"
  python scripts/p2/gen_answer.py --answerer gptoss --writer $D/writer.jsonl --out $D/answers-gptoss.jsonl --workers $WORKERS --model gpt-oss-20b > $O/answer-gptoss.log 2>&1 || { tail -n 5 $O/answer-gptoss.log; log "ANSWER_GPTOSS_FAILED"; stop; exit 1; }
  tail -n 1 $O/answer-gptoss.log; touch $O/answer-gptoss.DONE; stop
fi

if ! skip assemble; then log "stage assemble"
  python scripts/p2/assemble_p2.py --writer $D/writer.jsonl --answers $D/answers-qwen.jsonl $D/answers-gptoss.jsonl --out $D/records.jsonl \
     --reference $P2/jevbench-public/*.jsonl > $O/assemble.log 2>&1 || { tail -n 5 $O/assemble.log; log "ASSEMBLE_FAILED"; exit 1; }
  tail -n 2 $O/assemble.log; touch $O/assemble.DONE; fi

log pack; tar czf $P2/decision-p2-teacher.tgz -C $P2/imajev/data decision-p2 && ls -la decision-p2-teacher.tgz; echo ALL_DONE
