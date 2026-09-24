#!/usr/bin/env bash
# 2b-delta on the idle eval lane (GPUs 4-7), identical flags to cloud/pod_run_p2c_train.sh's cfg line; the main script skips its own
# train-2b-delta stage when $O/train-2b-delta.DONE exists and proceeds to evaluate it.
export NCCL_NVLS_ENABLE=0 PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ; . venv/bin/activate
O=p2/train-out-p2c; r=2b-delta; M2=hf/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
RECIPE="--soft-targets --soft-weight 1.0 --rationale-weight 0.3 --rationale-max-tokens 192 --permute-options"
mkdir -p $O/$r; echo "$(date +%H:%M:%S) side-lane: train-2b-delta on GPUs 4-7 (identical flags)" >> $O/run.log
for attempt in 1 2 3; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 29601 scripts/train_decision_lora_torch.py \
    --version decision-p2c --output $O/$r/train --model $M2 --init-adapter adapters/2b-p2b-best --epochs 2 --lr 2e-5 --warmup 10 \
    --save-every 10 --dev-cases 600 --dev-every 10 --dev2-version decision-p2c-judge-dev --dev2-cases 400 --select mean_accuracy \
    --pad-multiple 64 --token-budget 12000 --max-length 4096 --batch-size 40 --accumulate 2 --workers 16 \
    $RECIPE --no-checkpointing >> $O/$r/train.log 2>&1 && { touch $O/train-$r.DONE; cp $O/$r/train/log.jsonl $O/$r/dev-curve.jsonl 2>/dev/null; echo "$(date +%H:%M:%S) side-lane: done train-2b-delta" >> $O/run.log; exit 0; }
  echo "$(date +%H:%M:%S) side-lane: train-2b-delta attempt $attempt failed (resume)" >> $O/run.log; sleep 10; done
echo "$(date +%H:%M:%S) side-lane: TRAIN_FAILED 2b-delta" >> $O/run.log; exit 1
