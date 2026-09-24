#!/usr/bin/env bash
# Side lane: authored JevBench-style dev (150 items) panel + single-T NLL fit for one adapter. usage: fitT_gen.sh SIZE TAG ADAPTER GPU
SZ=$1; TAG=$2; AD=$3; G=$4
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1; cd ; . venv/bin/activate
O=p2/train-out-p2c/$SZ-delta; L=p2/train-out-p2c/run.log; V=decision-p2b-jevstyle-dev
case $SZ in 2b) HF=Qwen3.5-2B;; 4b) HF=Qwen3.5-4B;; 9b) HF=Qwen3.5-9B;; esac
M=$(ls -d hf/hub/models--Qwen--$HF/snapshots/*/ | head -n 1); M=${M%/}
out=$O/eval-$TAG-$V; rm -rf $out; mkdir -p $out
CUDA_VISIBLE_DEVICES=$G python scripts/evaluate_decision_model_torch.py --version $V --output $out --model $M --adapter $AD --partition dev --token-budget 16000 > $out.log 2>&1 || { echo "$(date +%H:%M:%S) side-lane: FAILED authored-dev panel $SZ $TAG" | tee -a $L; exit 1; }
python scripts/p2/fit_p2_temperature.py --predictions $out/predictions.jsonl --manifest data/manifests/$V.jsonl --version p2c-$SZ-$TAG-authored --out $O/calibration-p2c-$SZ-$TAG-authored.json > $out-fit.log 2>&1 \
  && python - $O/calibration-p2c-$SZ-$TAG-authored.json $out/predictions.jsonl "$SZ $TAG" <<'PY' | tee -a $L
import json, sys
p = json.load(open(sys.argv[1])); rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
print(f"  side-lane: {sys.argv[3]} authored-dev T={float(p['fit']['temperature']):.3f} ({len(rows)} rows, acc {sum(r['correct'] for r in rows)/len(rows):.3f})")
PY
