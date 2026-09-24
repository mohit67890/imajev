#!/usr/bin/env bash
# Side lane, any size: adapter soup with weight W on the phase-2c best (1-W on phase-2b); ImajevBench public on GPU $G1 and JevBench raw on GPU $G2/port $PORT in parallel.
# usage: soup_gen.sh SIZE(2b|4b|9b) W G1 G2 PORT
SZ=$1; W=$2; G1=$3; G2=$4; PORT=$5; TAG=soup$(python3 -c "print(int(round($W*100)))")
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1; cd ; . venv/bin/activate
P2=p2; O=$P2/train-out-p2c/$SZ-delta; L=$P2/train-out-p2c/run.log
case $SZ in 2b) HF=Qwen3.5-2B; NAME=imajev2b-v21;; 4b) HF=Qwen3.5-4B; NAME=qwen4b;; 9b) HF=Qwen3.5-9B; NAME=imajev9b;; esac
M=$(ls -d hf/hub/models--Qwen--$HF/snapshots/*/ | head -n 1); M=${M%/}
A=adapters/$SZ-p2b-best; B=$O/train/best; C=adapters/$SZ-$TAG; mkdir -p $C; cp $A/adapter_config.json $A/decision_readout.json $A/README.md $C/ 2>/dev/null
python - $A $B $C $W <<'PY'
import sys, torch
from safetensors.torch import load_file, save_file
a, b, c, w = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
for f in ("adapter_model.safetensors", "decision_readout.safetensors"):
    x = load_file(f"{a}/{f}"); y = load_file(f"{b}/{f}"); assert x.keys() == y.keys(), (f, set(x) ^ set(y))
    save_file({k: ((1 - w) * x[k].float() + w * y[k].float()).to(x[k].dtype).contiguous() for k in x}, f"{c}/{f}", metadata={"format": "pt"})
print("soup written", c)
PY
( out=$O/imajevbench-$TAG; rm -rf $out
  CUDA_VISIBLE_DEVICES=$G1 python scripts/imajev_bench/run_local_v2.py --records bench/records/records-eval.jsonl --root bench --split test \
    --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $NAME --adapter $C \
    --base-path $M --output $out > $out.log 2>&1 || { echo "$(date +%H:%M:%S) side-lane: FAILED imajevbench $SZ $TAG" | tee -a $L; exit 1; }
  python -m imajev_bench score --records bench/records/records-eval.jsonl --root bench --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
  python - $out/score "$SZ $TAG" <<'PY' | tee -a $L
import json, sys
s = json.load(open(sys.argv[1])); tr = {t: (sum(f["correct"] for f in v["families"].values()), sum(f["total"] for f in v["families"].values())) for t, v in s["capability"]["tracks"].items()}
tot = sum(v[0] for v in tr.values()); n = sum(v[1] for v in tr.values()); print(f"  side-lane: {sys.argv[2]} imajevbench: {tot}/{n} = {tot/n:.1%} tracks {tr}")
PY
) &
( SERVER=scripts/playground/server.py; R=$O/jevbench-$TAG-raw; mkdir -p $R
  printf '{"repo": "local", "revision": "local", "path": "%s"}\n' "$M" > $R/bundle.json
  CUDA_VISIBLE_DEVICES=$G2 setsid python $SERVER --backend torch --model-bundle $R/bundle.json --adapter $C --host 127.0.0.1 --port $PORT > $R/server.log 2>&1 < /dev/null &
  pg=$!; for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null && break; kill -0 $pg 2>/dev/null || break; sleep 3; done
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split
      TYPESAFE_API_KEY=local python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:$PORT --model imajev-$SZ-$TAG-raw \
        --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  side-lane: $SZ $TAG raw jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))" 2>/dev/null | tee -a $L \
        || echo "  side-lane: $SZ $TAG raw jevbench $split: FAILED" | tee -a $L; done )
  kill -TERM -- -$pg 2>/dev/null; sleep 3; kill -KILL -- -$pg 2>/dev/null ) &
wait; echo "$(date +%H:%M:%S) side-lane: $SZ $TAG done" | tee -a $L
