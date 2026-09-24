#!/usr/bin/env bash
# When gpt-oss finishes, give the Qwen answerer all four GPUs: second 27B server on GPUs 2-3, two sharded answerers,
# merge, assemble, pack, then launch the fine-tunes. Owns the rest of the chain (the parallel stage script exits on the kill).
set -uo pipefail
P2=p2; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_API_KEY=local PATH=vllm-venv/bin:$PATH
O=$P2/out/p2; D=data/decision-p2/teacher; QW=hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
log(){ echo "$(date +%H:%M:%S) $*"; }
until [ -f $O/answer-gptoss.DONE ]; do sleep 20; done; log "gpt-oss done; starting second Qwen server on GPUs 2,3"
pkill -f "served-model-name gpt-oss-20b" 2>/dev/null; sleep 8
(CUDA_VISIBLE_DEVICES=2,3 vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model $QW --served-model-name qwen3.6-27b --port 8011 \
   --max-model-len 24576 --gpu-memory-utilization 0.92 --max-num-seqs 192 --tensor-parallel-size 2 --reasoning-parser qwen3 > $O/vllm-qwen-b.log 2>&1 &)
for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:8011/v1/models | grep -q qwen3.6-27b && break; sleep 5; done
curl -s -m 2 http://127.0.0.1:8011/v1/models | grep -q qwen3.6-27b || { log "SECOND_SERVER_FAILED"; exit 1; }
log "second server up; resharding the remaining Qwen questions"
pkill -f pod_run_p2_answer_parallel 2>/dev/null; pkill -f "gen_answer.py --answerer qwen" 2>/dev/null; sleep 5
python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer.jsonl --out $D/answers-qwen.jsonl   --shard 0/2 --workers 192 --max-tokens 5000 --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 --done-from $D/answers-qwen-b.jsonl > $O/answer-qwen.log 2>&1 &
python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer.jsonl --out $D/answers-qwen-b.jsonl --shard 1/2 --workers 192 --max-tokens 5000 --model qwen3.6-27b --base-url http://127.0.0.1:8011/v1 --done-from $D/answers-qwen.jsonl > $O/answer-qwen-b.log 2>&1 &
wait; tail -n 1 $O/answer-qwen.log; tail -n 1 $O/answer-qwen-b.log
cat $D/answers-qwen-b.jsonl >> $D/answers-qwen.jsonl 2>/dev/null; touch $O/answer-qwen.DONE
pkill -f "vllm.entrypoints" 2>/dev/null; sleep 5
log "stage assemble"
python scripts/p2/assemble_p2.py --writer $D/writer.jsonl --answers $D/answers-qwen.jsonl $D/answers-gptoss.jsonl --out $D/records.jsonl \
   --reference $P2/jevbench-public/*.jsonl > $O/assemble.log 2>&1 || { tail -n 5 $O/assemble.log; log "ASSEMBLE_FAILED"; exit 1; }
tail -n 3 $O/assemble.log; touch $O/assemble.DONE
tar czf $P2/decision-p2-teacher.tgz -C data decision-p2 && ls -la $P2/decision-p2-teacher.tgz
mkdir -p $P2/imajev/data/decision-p2/teacher && cp $D/records.jsonl $P2/imajev/data/decision-p2/teacher/records.jsonl
log "launching fine-tunes"; cd /workspace && setsid nohup env PATH=vllm-venv/bin:$PATH bash pod_run_p2_train.sh > p2/run_p2_train.log 2>&1 < /dev/null &
echo ALL_DONE
