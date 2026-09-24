#!/usr/bin/env bash
# Phase-2 delta fine-tunes on the 4xH200 pod: 2B v2.1, 4B v2.1, 9B v1.1 adapters continued on decision-p2 (teacher + human).
# Runs from the 4B code tree (: trainer, manifests, probe images). Stages per size with DONE markers.
set -uo pipefail
cd ; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P2=p2; O=$P2/train-out; mkdir -p $O; GPUS=4; SIZES=${SIZES:-"2b 4b 9b"}
log(){ echo "$(date +%H:%M:%S) $*"; }; skip(){ [ -f $O/$1.DONE ]; }
snap(){ d=hf/hub/models--${1//\//--}/snapshots/$2; [ -d "$d" ] && echo "$d" || echo ""; }
cleanup(){ pkill -f train_decision_lora_torch 2>/dev/null; sleep 3; }  # never kill evaluators or servers: another stage may own them

if ! skip manifest; then log "stage manifest: merge teacher + human into decision-p2"
  python - <<'PY' || { echo MANIFEST_FAILED; exit 1; }
import json, collections
rows = []
for p in ("p2/imajev/data/decision-p2/teacher/records.jsonl", "p2/imajev/data/decision-p2/human/records.jsonl"):
    rows += [json.loads(l) for l in open(p)]
ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate ids"
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from decision_data import expand_fields, render
valid, dropped = [], collections.Counter()
for r in rows:
    try:
        for item in expand_fields(r): render(item)
        valid.append(r)
    except Exception as e:
        dropped[str(e).split("\n")[1][:60] if "\n" in str(e) else str(e)[:60]] += 1
print("contract validation dropped", sum(dropped.values()), dict(dropped)); rows = valid
with open("data/manifests/decision-p2.jsonl", "w") as w:
    for r in rows: w.write(json.dumps(r) + "\n")
c = collections.Counter((r["source"].split("_")[0] if r["source"].startswith("p2_") and r["source"] != "p2_teacher" else r["source"], r["partition"]) for r in rows)
print("decision-p2:", len(rows), "records;", dict(collections.Counter(r["partition"] for r in rows)), "; unknown targets:", sum(r.get("target") in (None, "unknown", "__unknown__") for r in rows))
PY
  touch $O/manifest.DONE; fi

declare -A MODEL ADAPTER BUDGET
MODEL[2b]=$(snap Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc); ADAPTER[2b]=adapters/imajev-2b-v21; BUDGET[2b]=12000
MODEL[4b]=$(cat model_path); ADAPTER[4b]=out/4b/train/best; BUDGET[4b]=10000
MODEL[9b]=$(snap Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a); ADAPTER[9b]=reports/decision-v1.1-9b/runs/h200x4/best; BUDGET[9b]=7500

evalp(){ # $1 size $2 ckpt-dir $3 tag $4 version $5 partition(optional)
  local out=$O/$1/eval-$3-$4; [ -f $out/predictions.jsonl ] && return
  for i in $(seq 0 $((GPUS-1))); do CUDA_VISIBLE_DEVICES=$i python scripts/evaluate_decision_model_torch.py --version $4 --output $out --model ${MODEL[$1]} --adapter $2 ${5:+--partition $5} --shard $i/$GPUS --token-budget 16000 > $out-$i.log 2>&1 & done; wait
  cat $out/predictions-*.jsonl > $out/predictions.jsonl 2>/dev/null; log "  $1 $3 $4: $(wc -l < $out/predictions.jsonl) predictions"; }

ibench(){ # $1 size $2 ckpt-dir $3 tag : ImajevBench v2.0-lite test split, direct option scoring (torch)
  local out=$O/$1/imajevbench-$3; [ -f $out/completion.json ] && return; rm -rf $out
  local name=imajev2b-v21; [ "$1" = "4b" ] && name=qwen4b; [ "$1" = "9b" ] && name=imajev9b
  CUDA_VISIBLE_DEVICES=0 python scripts/imajev_bench/run_local_v2.py --records bench/records/records-eval.jsonl --root bench --split test --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $name --adapter $2 --base-path ${MODEL[$1]} --output $out > $out.log 2>&1 || { tail -n 3 $out.log; log "FAILED imajevbench $1 $3"; return; }
  python -m imajev_bench score --records bench/records/records-eval.jsonl --root bench --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
  python - <<PY
import json
s=json.load(open("$out/score")); tr={t:(sum(f["correct"] for f in v["families"].values()),sum(f["total"] for f in v["families"].values())) for t,v in s["capability"]["tracks"].items()}
tot=sum(v[0] for v in tr.values()); n=sum(v[1] for v in tr.values()); print(f"  $1 $3 imajevbench: {tot}/{n} = {tot/n:.1%} tracks {tr}")
PY
}

jevbench(){ # $1 size $2 ckpt-dir $3 tag
  local R=$O/$1/jevbench-$3; [ -f $R/hard/summary.json ] && return; mkdir -p $R
  printf '{"repo": "local", "revision": "local", "path": "%s"}\n' "${MODEL[$1]}" > $R/bundle.json
  (python scripts/playground/server.py --backend torch --model-bundle $R/bundle.json --adapter $2 --host 127.0.0.1 --port 8765 > $R/server.log 2>&1 &)
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8765/v1/models >/dev/null && break; sleep 3; done
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split
      TYPESAFE_API_KEY=local python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:8765 --model imajev-$1-p2-$3 --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  $1 $3 jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))"; done )
  pkill -f "playground/server.py" || true; sleep 3; }

for SIZE in $SIZES; do
  M=${MODEL[$SIZE]}; A=${ADAPTER[$SIZE]}; TB=${BUDGET[$SIZE]}; mkdir -p $O/$SIZE
  [ -d "$A" ] || { log "MISSING adapter $A"; continue; }; [ -d "$M" ] || { log "MISSING model $M"; continue; }
  if ! skip train-$SIZE; then log "stage train $SIZE from $A (budget $TB)"; cleanup
    for attempt in 1 2 3 4; do
      python -m torch.distributed.run --nproc_per_node=$GPUS scripts/train_decision_lora_torch.py --version decision-p2 --output $O/$SIZE/train --model $M \
        --init-adapter $A --epochs ${EPOCHS:-2} --lr ${LR:-3e-5} --warmup 20 --save-every 20 --dev-cases 400 --dev-every 50 \
        --dev2-version decision-v2-reasoning-dev --dev2-cases 400 --select mean_accuracy --pad-multiple 64 --token-budget $TB \
        --max-length 4096 --batch-size ${BATCH:-40} --accumulate ${ACCUMULATE:-2} --workers 16 \
        $( [ "$SIZE" = "9b" ] && echo "" || echo "--no-checkpointing" ) >> $O/$SIZE/train.log 2>&1 && { touch $O/train-$SIZE.DONE; break; }
      log "  train $SIZE attempt $attempt failed (resume)"; cleanup; done
    skip train-$SIZE || { log "TRAIN_FAILED $SIZE"; continue; }; fi
  [ "${TRAIN_ONLY:-0}" = "1" ] && { log "train-only instance: leaving eval $SIZE to the main run"; continue; }
  if ! skip eval-$SIZE; then log "stage eval $SIZE"
    for tag in best last; do C=$O/$SIZE/train/$tag; [ -d $C ] || continue
      if [ "$tag" = "best" ]; then
        evalp $SIZE $C $tag decision-p2 test; evalp $SIZE $C $tag decision-v2-reasoning-dev dev; evalp $SIZE $C $tag decision-v1.1-heldout-text test
        evalp $SIZE $C $tag decision-v1.1-irrelevance-test test; evalp $SIZE $C $tag decision-v2-state-probe; evalp $SIZE $C $tag decision-v2-pairs-probe; fi
      jevbench $SIZE $C $tag; [ "$tag" = "best" ] && ibench $SIZE $C $tag; done
    touch $O/eval-$SIZE.DONE; fi
done
[ "${TRAIN_ONLY:-0}" = "1" ] && { echo TRAIN_ONLY_DONE; exit 0; }
log pack; cd $P2 && tar czf $P2/p2-train-results.tgz --exclude='*/trainer.pt' --exclude='*/raw' train-out && ls -la $P2/p2-train-results.tgz; echo ALL_DONE
