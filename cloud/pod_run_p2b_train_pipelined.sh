#!/usr/bin/env bash
# PIPELINED variant for the 7xH100 pod (derived 2026-09-24 from pod_run_p2b_train.sh): a TRAIN lane on GPUs 0-3 runs the sizes back to
# back while an EVAL lane on GPUs 4-6 evaluates each finished size (panels sharded over 3 GPUs; then JevBench best on GPU 4,
# JevBench last on GPU 5 and ImajevBench on GPU 6 in parallel). Same trainer flags, budgets and selection as the original.
# Phase-2b delta fine-tunes: 2B/4B/9B continued from their phase-2 best adapters on decision-p2b
# (all new p2b rows + 30% deterministic replay of phase-2 train rows; 2 epochs, lr 2e-5; selection = mean of p2b dev accuracy
# and the authored JevBench-style dev accuracy). Then, for BEST and LAST: JevBench public raw, 4-rotation averaging, and
# calibrated with ONE temperature fitted by NLL on the held-out-domain calibration rows (scripts/p2/fit_p2_temperature.py);
# decision-p2b test, decision-p2 test (regression), reasoning dev, state + pairs probes; ImajevBench on best.
# Runs from the training tree (: trainer, manifests, probe images); p2 tools and the rotation-aware server
# come from the phase-2 tree ($P2/imajev) — sync this repo's scripts/p2 and scripts/playground there first.
# Never run two instances on one pod. Stages carry DONE markers under p2/train-out-p2b.
set -uo pipefail
cd ; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P2=${P2:-p2}; O=$P2/train-out-p2b; mkdir -p $O; GPUS=4; TRAIN_GPUS=0,1,2,3; EVAL_GPUS=(4 5 6); SIZES=${SIZES:-"4b 9b 2b"}
MAN=data/manifests; D=data/decision-p2/teacher; TOOLS=$P2/imajev/scripts/p2; SERVER=${SERVER:-$P2/imajev/scripts/playground/server.py}
HOLDOUT=${HOLDOUT:-telecom,hospitality,nonprofit_grants}
log(){ echo "$(date +%H:%M:%S) $*"; }; skip(){ [ -f $O/$1.DONE ]; }
snap(){ d=hf/hub/models--${1//\//--}/snapshots/$2; [ -d "$d" ] && echo "$d" || echo ""; }
cleanup(){ pkill -f train_decision_lora_torch 2>/dev/null; sleep 3; }  # never kill evaluators or servers: another stage may own them

if ! skip manifest; then log "stage manifest"
  if [ ! -f $MAN/decision-p2b.jsonl ] || [ "${REBUILD_MANIFEST:-0}" = "1" ]; then
    python $TOOLS/build_p2b_manifest.py --p2 $MAN/decision-p2.jsonl --records $D/records-p2b.jsonl --batch-filter 'p2b-*' --replay 0.30 \
      --holdout-domains $HOLDOUT --out $MAN/decision-p2b.jsonl || { echo MANIFEST_FAILED; exit 1; }; fi
  [ -f $MAN/decision-p2b-jevstyle-dev.jsonl ] || python $TOOLS/author_jevstyle_dev.py --out $MAN/decision-p2b-jevstyle-dev.jsonl || { echo JEVSTYLE_FAILED; exit 1; }
  python - <<'PY' || { echo MANIFEST_CHECK_FAILED; exit 1; }
import json, collections
for name in ("decision-p2b", "decision-p2b-jevstyle-dev"):
    rows = [json.loads(l) for l in open(f"data/manifests/{name}.jsonl") if l.strip()]
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate ids"
    print(name, len(rows), dict(collections.Counter(r["partition"] for r in rows)), "unknown", sum(r.get("target") is None for r in rows))
PY
  touch $O/manifest.DONE; fi

declare -A MODEL ADAPTER BUDGET
MODEL[4b]=$(cat model_path); ADAPTER[4b]=$P2/train-out/4b/train/best; BUDGET[4b]=10000
MODEL[9b]=$(snap Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a); ADAPTER[9b]=$P2/train-out/9b/train/best; BUDGET[9b]=7500
MODEL[2b]=$(snap Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc); ADAPTER[2b]=$P2/train-out/2b/train/best; BUDGET[2b]=12000  # 2B added 2026-09-24 so all three tiers get the same phase-2b pass

evalp(){ # $1 size $2 ckpt-dir $3 tag $4 version $5 partition(optional)
  local out=$O/$1/eval-$3-$4${5:+-$5}; [ -f $out/predictions.jsonl ] && return
  local k=0 n=${#EVAL_GPUS[@]}; for g in "${EVAL_GPUS[@]}"; do CUDA_VISIBLE_DEVICES=$g python scripts/evaluate_decision_model_torch.py --version $4 --output $out --model ${MODEL[$1]} --adapter $2 ${5:+--partition $5} --shard $k/$n --token-budget 16000 > $out-$k.log 2>&1 & k=$((k+1)); done; wait
  cat $out/predictions-*.jsonl > $out/predictions.jsonl 2>/dev/null
  python - <<PY
import json; rows=[json.loads(l) for l in open("$out/predictions.jsonl")]; print(f"  $1 $3 $4${5:+ $5}: {len(rows)} predictions, acc {sum(r['correct'] for r in rows)/max(1,len(rows)):.3f}")
PY
}

fit_temp(){ # $1 size $2 ckpt-dir $3 tag -> $O/$1/calibration-p2b-$1-$3.json (single temperature, held-out domains)
  local out=$O/$1/calibration-p2b-$1-$3.json; [ -f $out ] && return
  evalp $1 $2 $3 decision-p2b calibration
  python $TOOLS/fit_p2_temperature.py --predictions $O/$1/eval-$3-decision-p2b-calibration/predictions.jsonl --manifest $MAN/decision-p2b.jsonl \
    --holdout-domains $HOLDOUT --version p2b-$1-$3 --out $out || log "FIT_TEMP_FAILED $1 $3"
  [ "$3" = "best" ] && cp $out $O/$1/calibration-p2b-$1.json; }

ibench(){ # $1 size $2 ckpt-dir $3 tag : ImajevBench v2.0-lite test split, direct option scoring (torch)
  local out=$O/$1/imajevbench-$3; [ -f $out/completion.json ] && return; rm -rf $out
  local name=qwen4b; [ "$1" = "9b" ] && name=imajev9b; [ "$1" = "2b" ] && name=imajev2b-v21
  CUDA_VISIBLE_DEVICES=${IB_GPU:-${EVAL_GPUS[2]}} python scripts/imajev_bench/run_local_v2.py --records bench/records/records-eval.jsonl --root bench --split test --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $name --adapter $2 --base-path ${MODEL[$1]} --output $out > $out.log 2>&1 || { tail -n 3 $out.log; log "FAILED imajevbench $1 $3"; return; }
  python -m imajev_bench score --records bench/records/records-eval.jsonl --root bench --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
  python - <<PY
import json
s=json.load(open("$out/score")); tr={t:(sum(f["correct"] for f in v["families"].values()),sum(f["total"] for f in v["families"].values())) for t,v in s["capability"]["tracks"].items()}
tot=sum(v[0] for v in tr.values()); n=sum(v[1] for v in tr.values()); print(f"  $1 $3 imajevbench: {tot}/{n} = {tot/n:.1%} tracks {tr}")
PY
}

jevbench(){ # $1 size $2 ckpt-dir $3 tag $4 variant(raw|rot4|cal|rot4cal) $5 gpu $6 port
  local R=$O/$1/jevbench-$3-$4 gpu=${5:-${EVAL_GPUS[0]}} port=${6:-8765}; [ -f $R/hard/summary.json ] && return; mkdir -p $R
  local extra=""; case $4 in rot4) extra="--rotations 4";; cal) extra="--calibration $O/$1/calibration-p2b-$1-$3.json";; rot4cal) extra="--rotations 4 --calibration $O/$1/calibration-p2b-$1-$3.json";; esac
  case $4 in cal|rot4cal) [ -f $O/$1/calibration-p2b-$1-$3.json ] || { log "  no calibration for $1 $3: skipping $4"; return; };; esac
  printf '{"repo": "local", "revision": "local", "path": "%s"}\n' "${MODEL[$1]}" > $R/bundle.json
  (CUDA_VISIBLE_DEVICES=$gpu python $SERVER --backend torch --model-bundle $R/bundle.json --adapter $2 --host 127.0.0.1 --port $port $extra > $R/server.log 2>&1 &)
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break; sleep 3; done
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split
      TYPESAFE_API_KEY=local python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:$port --model imajev-$1-p2b-$3-$4 --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  $1 $3 $4 jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))"; done )
  pkill -f "playground/server.py.*--port $port" || true; sleep 3; }

train_lane(){ for SIZE in $SIZES; do
  M=${MODEL[$SIZE]}; A=${ADAPTER[$SIZE]}; TB=${BUDGET[$SIZE]}; mkdir -p $O/$SIZE
  [ -d "$A" ] || { log "MISSING adapter $A"; touch $O/train-$SIZE.FAILED; continue; }; [ -d "$M" ] || { log "MISSING model $M"; touch $O/train-$SIZE.FAILED; continue; }
  if ! skip train-$SIZE; then log "stage train $SIZE from $A (budget $TB, epochs ${EPOCHS:-2}, lr ${LR:-2e-5})"; cleanup
    for attempt in 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m torch.distributed.run --nproc_per_node=$GPUS scripts/train_decision_lora_torch.py --version decision-p2b --output $O/$SIZE/train --model $M \
        --init-adapter $A --epochs ${EPOCHS:-2} --lr ${LR:-2e-5} --warmup 10 --save-every 10 --dev-cases 600 --dev-every 10 \
        --dev2-version decision-p2b-jevstyle-dev --dev2-cases 150 --select mean_accuracy --pad-multiple 64 --token-budget $TB \
        --max-length 4096 --batch-size ${BATCH:-40} --accumulate ${ACCUMULATE:-2} --workers 16 \
        $( [ "$SIZE" = "9b" ] && echo "" || echo "--no-checkpointing" ) >> $O/$SIZE/train.log 2>&1 && { touch $O/train-$SIZE.DONE; break; }
      log "  train $SIZE attempt $attempt failed (resume)"; cleanup; done
    skip train-$SIZE || { log "TRAIN_FAILED $SIZE"; touch $O/train-$SIZE.FAILED; continue; }
    cp $O/$SIZE/train/log.jsonl $O/$SIZE/dev-curve.jsonl 2>/dev/null; fi   # every checkpoint's dev / dev2 lines
done; }

eval_size(){ local SIZE=$1 C
  if ! skip eval-$SIZE; then log "stage eval $SIZE on GPUs ${EVAL_GPUS[*]} (train lane continues on $TRAIN_GPUS)"
    for tag in best last; do C=$O/$SIZE/train/$tag; [ -d $C ] || continue
      fit_temp $SIZE $C $tag
      evalp $SIZE $C $tag decision-p2b test; evalp $SIZE $C $tag decision-p2 test; evalp $SIZE $C $tag decision-p2b-jevstyle-dev dev
      if [ "$tag" = "best" ]; then evalp $SIZE $C $tag decision-v2-reasoning-dev dev; evalp $SIZE $C $tag decision-v2-state-probe; evalp $SIZE $C $tag decision-v2-pairs-probe; fi; done
    ( C=$O/$SIZE/train/best; [ -d $C ] && for variant in raw rot4 cal rot4cal; do jevbench $SIZE $C best $variant ${EVAL_GPUS[0]} 8765; done ) &
    ( C=$O/$SIZE/train/last; [ -d $C ] && for variant in raw rot4 cal rot4cal; do jevbench $SIZE $C last $variant ${EVAL_GPUS[1]} 8766; done ) &
    ( C=$O/$SIZE/train/best; [ -d $C ] && IB_GPU=${EVAL_GPUS[2]} ibench $SIZE $C best ) &
    wait; touch $O/eval-$SIZE.DONE; log "eval $SIZE done"; fi; }
eval_lane(){ until ! pgrep -f "mojev serve" >/dev/null; do sleep 20; done
  for SIZE in $SIZES; do until [ -f $O/train-$SIZE.DONE ] || [ -f $O/train-$SIZE.FAILED ]; do sleep 20; done
    [ -f $O/train-$SIZE.DONE ] && eval_size $SIZE; done; }
eval_lane & EL=$!
train_lane
wait $EL
log pack; cd $P2 && tar czf $P2/p2b-train-results.tgz --exclude='*/trainer.pt' --exclude='*/raw' train-out-p2b && ls -la $P2/p2b-train-results.tgz; echo ALL_DONE
