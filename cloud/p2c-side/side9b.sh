#!/usr/bin/env bash
# 9B side orchestrator (GPUs 0-3), armed before the 9B training ends. The main script evaluates 9b-delta best/last on GPUs 4-7.
# Phase 1: soup50 public ImajevBench (GPU 0) + JevBench raw (GPU 1) | private-1 for 9b-delta best (GPU 2) | authored-dev fits best/last/soup50 (GPU 3)
# Phase 2: soup50 JevBench cal/rot4/rot4cal (GPUs 0,1,3) | private-1 for soup50 (GPU 2)
# Phase 3: soup50 gate panels + gates (GPUs 0-3, via gates_soup9b.sh generated from the main script's functions)
O=p2/train-out-p2c; L=$O/run.log; log(){ echo "$(date +%H:%M:%S) side-lane: $*" | tee -a $L; }
until [ -f $O/train-9b-delta.DONE ] || [ -f $O/train-9b-delta.FAILED ]; do sleep 15; done
[ -f $O/train-9b-delta.DONE ] || { log "9B training FAILED; side orchestrator exiting"; exit 1; }
[ -d $O/9b-delta/train/best ] || { log "no 9b best checkpoint"; exit 1; }
log "9B side orchestrator: phase 1 on GPUs 0-3"
bash soup_gen.sh 9b 0.5 0 1 8775 > $O/soup50_9b.out 2>&1 &
p_soup=$!
bash priv_run.sh 9b-delta $O/9b-delta/train/best 9b 2 > $O/priv_9b.out 2>&1 &
p_priv=$!
( bash fitT_gen.sh 9b best $O/9b-delta/train/best 3; bash fitT_gen.sh 9b last $O/9b-delta/train/last 3
  until [ -s adapters/9b-soup50/decision_readout.safetensors ]; do sleep 5; done; sleep 5
  bash fitT_gen.sh 9b soup50 adapters/9b-soup50 3 ) > $O/fitT_9b.out 2>&1 &
p_fit=$!
wait $p_soup $p_priv $p_fit
log "9B side orchestrator: phase 2"
[ -f $O/9b-delta/calibration-p2c-9b-soup50-authored.json ] && bash jev_variants.sh 9b soup50 adapters/9b-soup50 $O/9b-delta/calibration-p2c-9b-soup50-authored.json 0 1 3 > $O/jev_variants_9b_soup50.out 2>&1 &
p_var=$!
bash priv_run.sh 9b-soup50 adapters/9b-soup50 9b 2 > $O/priv_9b_soup50.out 2>&1 &
p_priv2=$!
wait $p_var $p_priv2
log "9B side orchestrator: phase 3 (soup gates)"
F=cloud/pod_run_p2c_train.sh; T=gates_soup9b.sh
{ echo '#!/usr/bin/env bash'
  awk '/^export NCCL_NVLS_ENABLE/{p=1} /^if ! skip manifest-check/{p=0} p' $F | grep -v 'rm -f $O/gpu7-free'
  echo 'EIKOS_GPU=""; EVAL_GPUS=(0 1 2 3)'
  awk '/^eval_gpus\(\)/{p=1} /^smoke\(\)/{p=0} p' $F
  cat <<'TAIL'
r=9b-soup50; C=adapters/9b-soup50; cfg $r 9b "$M9" "" decision-p2c 0 0 0 0 0 0 0 0
mkdir -p $O/$r; rm -f $O/$r/imajevbench-best; ln -s $O/9b-delta/imajevbench-soup50 $O/$r/imajevbench-best
log "side-lane: gate panels for $r ($C)"
evalp $r $C best decision-p2b test; evalp $r $C best decision-v2-state-probe; evalp $r $C best decision-v2-pairs-probe; evalp $r $C best decision-v1.1-irrelevance-test
gates $r; log "side-lane: gates done for $r"
TAIL
} > $T
bash -n $T && bash $T >> $L 2>&1
log "9B side orchestrator: ALL SIDE LANES DONE"
