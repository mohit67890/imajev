#!/usr/bin/env bash
# ImajevBench hidden split (private-1, 202 test items, aggregates only) on GPU 7 for the phase-2b baselines and each finished
# phase-2c run. Waits for the tarball, then for the 2B side-lane training to free GPU 7, then for each run's public ImajevBench.
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1; cd ; . venv/bin/activate
O=p2/train-out-p2c; R=bench-private-1; L=$O/private1.log; log(){ echo "$(date +%H:%M:%S) private-1: $*" | tee -a $L; }
until [ -s private-1.tgz ] && tar tzf private-1.tgz >/dev/null 2>&1; do sleep 20; done
mkdir -p $R && tar xzf private-1.tgz -C bench-private-1 --strip-components=1 --warning=no-unknown-keyword 2>/dev/null; ls $R | tr '\n' ' '; echo
until [ -f $O/train-2b-delta.DONE ] || [ -f $O/train-2b-delta.FAILED ]; do sleep 30; done
M2=hf/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc; M4=hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; M9=hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
run(){ # name adapter model harness-model
  local out=$O/private1/$1; [ -f $out/completion.json ] && return 0; mkdir -p $O/private1
  CUDA_VISIBLE_DEVICES=7 python scripts/imajev_bench/run_local_v2.py --records $R/records-audited.jsonl --root $R --split test --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $4 --adapter $2 --base-path $3 --output $out > $out.log 2>&1 || { log "FAILED $1: $(tail -n 2 $out.log | cut -c1-120)"; return 1; }
  python -m imajev_bench score --records $R/records-audited.jsonl --root $R --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
  python - <<PY | tee -a $L
import json; s=json.load(open("$out/score")); tr={t:(sum(f["correct"] for f in v["families"].values()),sum(f["total"] for f in v["families"].values())) for t,v in s["capability"]["tracks"].items()}
tot=sum(v[0] for v in tr.values()); n=sum(v[1] for v in tr.values()); print(f"  $1 imajevbench PRIVATE-1: {tot}/{n} = {tot/n:.1%} tracks {tr} ece {s.get('probability_quality',{}).get('ece')}")
PY
}
log "baselines (phase-2b adapters)"; run 4b-p2b adapters/4b-p2b-best $M4 qwen4b; run 9b-p2b adapters/9b-p2b-best $M9 imajev9b; run 2b-p2b adapters/2b-p2b-best $M2 imajev2b-v21
for r in 4b-delta 9b-delta 2b-delta; do
  until [ -f $O/$r/imajevbench-best/completion.json ] || [ -f $O/train-$r.FAILED ] || grep -q "^ALL_DONE" $O/run.log; do sleep 60; done
  [ -d $O/$r/train/best ] || continue; case $r in 4b*) m=$M4; h=qwen4b;; 9b*) m=$M9; h=imajev9b;; 2b*) m=$M2; h=imajev2b-v21;; esac
  log "run $r"; run $r $O/$r/train/best $m $h; done
log PRIVATE1_DONE
