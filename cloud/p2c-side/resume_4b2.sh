#!/usr/bin/env bash
export NCCL_NVLS_ENABLE=0 PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=p2/train-out-p2c; r=4b-delta; cd ; . venv/bin/activate
M4=hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RECIPE="--soft-targets --soft-weight 1.0 --rationale-weight 0.3 --rationale-max-tokens 192 --permute-options"
run(){ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29500 scripts/train_decision_lora_torch.py \
    --version decision-p2c --output $O/$r/train --model $M4 --init-adapter adapters/4b-p2b-best --epochs 2 --lr 2e-5 --warmup 10 \
    --save-every 10 --dev-cases 600 --dev-every 10 --dev2-version decision-p2c-judge-dev --dev2-cases 400 --select mean_accuracy \
    --pad-multiple 64 --token-budget $1 --max-length 4096 --batch-size 40 --accumulate 2 --workers 16 $RECIPE --no-checkpointing >> $O/$r/train.log 2>&1; }
echo "$(date +%H:%M:%S) side-lane: 4b resume, original budget 10000 from a fresh process" >> $O/run.log
if run 10000; then echo ok; else
  echo "$(date +%H:%M:%S) side-lane: 4b resume at 10000 failed again -> budget 8000 (config.json updated to match)" >> $O/run.log
  venv/bin/python - <<PY
import json; p="p2/train-out-p2c/4b-delta/train/config.json"; c=json.load(open(p)); c["token_budget"]=8000; json.dump(c,open(p,"w"),indent=1); print("config token_budget ->", c["token_budget"])
PY
  run 8000 || { echo "$(date +%H:%M:%S) side-lane: TRAIN_FAILED 4b-delta at 8000 too" >> $O/run.log; tail -n 3 $O/$r/train.log | cut -c1-160 >> $O/run.log; exit 1; }
fi
touch $O/train-$r.DONE; cp $O/$r/train/log.jsonl $O/$r/dev-curve.jsonl 2>/dev/null; echo "$(date +%H:%M:%S) side-lane: done train-4b-delta" >> $O/run.log
cd /workspace && RUNS="4b-delta 9b-delta 2b-delta" setsid nohup bash cloud/pod_run_p2c_train.sh >> $O/run.log 2>&1 < /dev/null &
echo "$(date +%H:%M:%S) side-lane: main script relaunched" >> $O/run.log
