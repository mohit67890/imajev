#!/usr/bin/env bash
# Builds eval2b_side.sh from the main script's own config + eval functions, then evaluates 2b-delta on GPUs 4-6 now
# (same outputs/paths/DONE marker, so the main script skips eval-2b-delta later). Ports 8767/8768 (main script uses 8765/8766).
F=cloud/pod_run_p2c_train.sh; T=eval2b_side.sh
{ echo '#!/usr/bin/env bash'
  awk '/^export NCCL_NVLS_ENABLE/{p=1} /^if ! skip manifest-check/{p=0} p' $F | grep -v 'rm -f $O/gpu7-free'
  echo 'EIKOS_GPU=""; EVAL_GPUS=(4 5 6)'
  awk '/^eval_gpus\(\)/{p=1} /^smoke\(\)/{p=0} p' $F
  cat <<'TAIL'
eval_run_side(){ local r=$1 c
  skip eval-$r && { log "side-lane: eval-$r already done"; return; }; stage "eval-$r (side-lane, GPUs ${EVAL_GPUS[*]})"
  for tag in best last; do c=$O/$r/train/$tag; [ -d $c ] || continue
    fit_temp $r $c $tag
    evalp $r $c $tag decision-p2c test; evalp $r $c $tag $DEV2 dev
    if [ "$tag" = best ]; then
      evalp $r $c $tag decision-p2b test; evalp $r $c $tag decision-p2 test; evalp $r $c $tag decision-v2-reasoning-dev dev
      evalp $r $c $tag decision-v2-state-probe; evalp $r $c $tag decision-v2-pairs-probe; evalp $r $c $tag decision-v1.1-irrelevance-test; fi; done
  ( c=$O/$r/train/best; [ -d $c ] && for v in raw cal; do jevbench $r $c best $v ${EVAL_GPUS[0]} 8767; done ) &
  ( c=$O/$r/train/last; [ -d $c ] && for v in raw cal; do jevbench $r $c last $v ${EVAL_GPUS[1]} 8768; done ) &
  ( c=$O/$r/train/best; [ -d $c ] && IB_GPU=${EVAL_GPUS[2]} ibench $r $c best ) &
  wait; gates $r; touch $O/eval-$r.DONE; log "done eval-$r"; }
eval_run_side 2b-delta
TAIL
} > $T
bash -n $T || { echo SYNTAX_ERROR; exit 1; }
echo "lines: $(wc -l < $T)"; grep -nE "^(cfg 2b|EIKOS_GPU|eval_run_side 2b|RUNS=|rm -f)" $T | cut -c1-120
[ -f p2/train-out-p2c/train-2b-delta.DONE ] && [ -d p2/train-out-p2c/2b-delta/train/best ] || { echo NO_2B_TRAIN; exit 1; }
cd /workspace && setsid nohup bash $T >> p2/train-out-p2c/run.log 2>&1 < /dev/null &
sleep 3; echo LAUNCHED; tail -n 2 p2/train-out-p2c/run.log | cut -c1-160
