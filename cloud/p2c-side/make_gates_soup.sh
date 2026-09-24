#!/usr/bin/env bash
# Builds gates_soup.sh from the main script's config + eval functions: runs the gate panels (p2b test, state/pairs probes,
# irrelevance) for the soup adapters as pseudo-runs 4b-soup50 / 2b-soup50 (ImajevBench result symlinked), then the gates() check. GPUs 4-6.
F=cloud/pod_run_p2c_train.sh; T=gates_soup.sh
{ echo '#!/usr/bin/env bash'
  awk '/^export NCCL_NVLS_ENABLE/{p=1} /^if ! skip manifest-check/{p=0} p' $F | grep -v 'rm -f $O/gpu7-free'
  echo 'EIKOS_GPU=""; EVAL_GPUS=(4 5 6)'
  awk '/^eval_gpus\(\)/{p=1} /^smoke\(\)/{p=0} p' $F
  cat <<'TAIL'
soup_gates(){ local r=$1 sz=$2 base=$3 C=$4 src=$5   # pseudo-run name, size, base model, soup adapter, source run dir (for the ImajevBench result)
  cfg $r $sz "$base" "" decision-p2c 0 0 0 0 0 0 0 0
  mkdir -p $O/$r; [ -e $O/$r/imajevbench-best ] || ln -s $src/imajevbench-$r-tag $O/$r/imajevbench-best 2>/dev/null
  rm -f $O/$r/imajevbench-best; ln -s $O/$src/imajevbench-${r#*-} $O/$r/imajevbench-best
  log "side-lane: gate panels for $r ($C)"
  evalp $r $C best decision-p2b test; evalp $r $C best decision-v2-state-probe; evalp $r $C best decision-v2-pairs-probe; evalp $r $C best decision-v1.1-irrelevance-test
  gates $r; log "side-lane: gates done for $r"; }
soup_gates 4b-soup50 4b "$M4" adapters/4b-soup50 4b-delta
soup_gates 2b-soup50 2b "$M2" adapters/2b-soup50 2b-delta
TAIL
} > $T
bash -n $T || { echo SYNTAX_ERROR; exit 1; }
cd /workspace && setsid nohup bash $T >> p2/train-out-p2c/run.log 2>&1 < /dev/null &
sleep 2; echo LAUNCHED
