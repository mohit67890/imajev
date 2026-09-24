#!/usr/bin/env bash
# Side lane: ImajevBench private-1 hidden split for one adapter. usage: priv_run.sh RUNNAME ADAPTER SIZE GPU
N=$1; AD=$2; SZ=$3; G=$4
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1; cd ; . venv/bin/activate
O=p2/train-out-p2c; R=bench-private-1; L=$O/private1.log
case $SZ in 2b) HF=Qwen3.5-2B; NAME=imajev2b-v21;; 4b) HF=Qwen3.5-4B; NAME=qwen4b;; 9b) HF=Qwen3.5-9B; NAME=imajev9b;; esac
M=$(ls -d hf/hub/models--Qwen--$HF/snapshots/*/ | head -n 1); M=${M%/}
out=$O/private1/$N; rm -rf $out; mkdir -p $O/private1
CUDA_VISIBLE_DEVICES=$G python scripts/imajev_bench/run_local_v2.py --records $R/records-audited.jsonl --root $R --split test --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $NAME --adapter $AD --base-path $M --output $out > $out.log 2>&1 || { echo "$(date +%H:%M:%S) private-1: FAILED $N" | tee -a $L; exit 1; }
python -m imajev_bench score --records $R/records-audited.jsonl --root $R --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
python - $out/score $N <<'PY' | tee -a $L
import json, sys; s=json.load(open(sys.argv[1])); tr={t:(sum(f["correct"] for f in v["families"].values()),sum(f["total"] for f in v["families"].values())) for t,v in s["capability"]["tracks"].items()}
tot=sum(v[0] for v in tr.values()); n=sum(v[1] for v in tr.values()); print(f"  {sys.argv[2]} imajevbench PRIVATE-1: {tot}/{n} = {tot/n:.1%} tracks {tr} ece {s.get('probability_quality',{}).get('ece')}")
PY
