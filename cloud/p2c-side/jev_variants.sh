#!/usr/bin/env bash
# Side lane: JevBench public variants (cal = authored-dev T, rot4 = 4 cyclic rotations averaged, rot4cal) for one adapter, three GPUs in parallel.
# usage: jev_variants.sh SIZE TAG ADAPTER CALFILE G_cal G_rot4 G_rot4cal
SZ=$1; TAG=$2; C=$3; CAL=$4; GC=$5; GR=$6; GRC=$7
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1; cd ; . venv/bin/activate
P2=p2; O=$P2/train-out-p2c/$SZ-delta; L=$P2/train-out-p2c/run.log; SERVER=scripts/playground/server.py
case $SZ in 2b) HF=Qwen3.5-2B;; 4b) HF=Qwen3.5-4B;; 9b) HF=Qwen3.5-9B;; esac
M=$(ls -d hf/hub/models--Qwen--$HF/snapshots/*/ | head -n 1); M=${M%/}
one(){ local variant=$1 gpu=$2 port=$3; local R=$O/jevbench-$TAG-$variant; mkdir -p $R; local extra=()
  case $variant in cal) extra=(--calibration $CAL);; rot4) extra=(--rotations 4);; rot4cal) extra=(--rotations 4 --calibration $CAL);; esac
  printf '{"repo": "local", "revision": "local", "path": "%s"}\n' "$M" > $R/bundle.json
  CUDA_VISIBLE_DEVICES=$gpu setsid python $SERVER --backend torch --model-bundle $R/bundle.json --adapter $C --host 127.0.0.1 --port $port "${extra[@]}" > $R/server.log 2>&1 < /dev/null &
  local pg=$!; for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break; kill -0 $pg 2>/dev/null || break; sleep 3; done
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split
      TYPESAFE_API_KEY=local python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:$port --model imajev-$SZ-$TAG-$variant \
        --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  side-lane: $SZ $TAG $variant jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))" 2>/dev/null | tee -a $L \
        || echo "  side-lane: $SZ $TAG $variant jevbench $split: FAILED" | tee -a $L; done )
  kill -TERM -- -$pg 2>/dev/null; sleep 3; kill -KILL -- -$pg 2>/dev/null; }
one cal $GC 8771 & one rot4 $GR 8772 & one rot4cal $GRC 8773 & wait
echo "$(date +%H:%M:%S) side-lane: $SZ $TAG variants done" | tee -a $L
