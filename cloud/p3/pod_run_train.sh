#!/usr/bin/env bash
# Phase 3 (imajev-4b, last training run) on ONE 8xH100 pod, after cloud/p3/bootstrap_train.sh (docs/phase-3-plan.md;
# runbook reports/phase3/pod-runbook.md). Stages carry DONE markers under $O (rerun = resume; the trainer resumes from `last`).
#   0 checks     manifests, init-adapter structure (r16 / alpha 32 / readout; PEFT's non-strict load would silently start fresh)
#   1 prep       <V>-r255 / <V>-c256 / <V>-ordinal-dev (cloud/p3/prep_manifests.py)
#   2 parity     readout 255 vs 256 on the CUDA torch path (scripts/p3/parity_readout256.py --backend torch), GPU 6, while the
#                255-code pilot lanes already run; FAIL (or no 255-option training rows) disables the 256 lanes automatically
#   3 pilot      4 lanes x 2 GPUs x PILOT_STEPS (100) steps: baseline / +ordinal (W 0.15) / +256 codes / +rank64 (the shipped r16
#                LoRA expanded to r64, identical output at step 0: --expand-lora-rank, scripts/lora_expand.py), the Stage-3 flags,
#                dev = <= 254-option dev of <V> (+ image / text slices, --dev-slices), dev2 = ordinal-only dev -> pilot-decision.json
#                (cloud/p3/pilot_decision.py: the three factors are decided independently vs baseline; the full run combines them).
#                Each lane IS the full run's first PILOT_STEPS steps for its switches: same data order, seed, epochs (schedule) and
#                micro-batches per step (2 GPUs x PILOT_ACC 4 = 8 GPUs x ACC 1)
#   4 train-r1   all 8 GPUs, CONTINUING the winning lane from its step PILOT_STEPS (cloud/p3/lane_resume.py, trainer --resume-from;
#                exact) when the winning combination equals one lane; COMBINED winners (2+ factors) or a failed resume restart from
#                the shipped soup50 adapter with the combined flags (logged "lane resume: FALLBACK" / RESUME_FALLBACK):
#                --soft-targets --soft-weight 1.0 --permute-options --rationale-weight 0.3
#                --rationale-max-tokens 192 --max-length 16384 + the pilot's --ordinal-weight / --readout-codes, lr 2e-5, 2 epochs;
#                a snapshot of every dev checkpoint (~every 25% of the steps) in $O/ckpts (cloud/p3/ckpt_watch.py)
#                EVAL_DURING_TRAIN=1 (default 0; runbook "Eval during training"): 7 GPUs train, 1 GPU evaluates snapshots as they
#                appear (cloud/p3/eval_lane.py) through stages 4-6; stage 7 then covers only what is left
#   5 sel-r1     selection panels (held-out fresh / flagged, human slice) for every r1 snapshot -> r1-pick.json (no gates yet)
#   6 round 2    re-mine the labelled pool with the r1 pick (scripts/p3/mine.py, one worker per GPU; labels cached, no Azure),
#                round-2 manifest (cloud/p3/round2_manifest.py), one short extra epoch from the r1 pick (snapshots ~every 34%)
#   7 eval       every snapshot + the shipped 1.0 (minus what the eval lane finished): scripts/p3/eval_checkpoint.py, one checkpoint per
#                GPU at a time (queue)
#   8 pick       cloud/p3/select_checkpoint.py (held-out + human slice + gates ONLY); if nothing passes: soups with the shipped
#                adapter at SOUP_WEIGHTS, evaluated and picked the same way (fallback only)
#   9 final      cloud/p3/final_table.py (the DecisionBench ship rule is read there by a person) + tarball, ALL_DONE
# Markers: $O/run.log ends with ALL_DONE, or FAILED (and $O/FAILED). Nothing here terminates the pod.
# Launch: cd /workspace && setsid nohup bash cloud/p3/pod_run_train.sh >> p3/run/run.log 2>&1 < /dev/null &
set -uo pipefail
cd  || exit 1; . venv/bin/activate
[ -f p3.env ] && { set -a; . p3.env; set +a; }
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=${O:-p3/run}; TR=${TRACKING_ROOT:-p3/tracking-only}; mkdir -p $O $O/ckpts $O/eval
V=${P3_MANIFEST:-decision-p3}; SOURCE=${P3_LABELLED:-$V-labelled}
FRESH=${P3_HELDOUT_FRESH:-decision-p3-heldout-fresh}; FLAGGED=${P3_HELDOUT_FLAGGED:-decision-p3-heldout-flagged}; HUMAN=${P3_HUMAN_DEV:-decision-p3-human-dev}
HELDOUT_PART=${P3_HELDOUT_PARTITION:-test}
M4=$(cat model_path 2>/dev/null || echo hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
SHIPPED=${SHIPPED_ADAPTER:-adapters/4b-shipped}
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
BATCH=${BATCH:-40}; TOKEN_BUDGET=${TOKEN_BUDGET:-16384}; ACC=${ACC:-1}
EPOCHS=${EPOCHS:-2}; LR=${LR:-2e-5}; WARMUP=${WARMUP:-10}; CKPT_FRAC=${CKPT_FRAC:-0.25}
R2_EPOCHS=${R2_EPOCHS:-1}; R2_LR=${R2_LR:-1e-5}; R2_MAX_NEW=${R2_MAX_NEW:-20000}; R2_REPLAY=${R2_REPLAY:-1.0}; R2_FRAC=${R2_FRAC:-0.34}
MINE_SHARDS=${MINE_SHARDS:-data/p3/pool/shards/*.jsonl}   # the Stage-0 pool (scripts/p3/assemble_pool.py), candidate rows
# scripts/p3/mine.py CLI (workers claim shards with flock; outputs <out>/<shard>.{scores,flagged}.jsonl + <shard>.done)
MINE_CMD=${MINE_CMD:-"python scripts/p3/mine.py --backend torch --shards '{shards}' --out {out} --model-path {model} --adapter {adapter} --worker-id r2-gpu{gpu} --device cuda"}
SOUP_WEIGHTS=${SOUP_WEIGHTS:-"0.75 0.5 0.25"}
GATES_REF=${GATES_REF:-cloud/p2c_gates_reference.json}
CKPTING=${CKPTING:-auto}   # auto: OFF on >=120 GB GPUs (H200: ~25-30% faster, same math), ON otherwise; an OOM turns it ON for good
if [ "$CKPTING" = auto ]; then
  GMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ "${GMEM:-0}" -ge 120000 ]; then CKPTING=0; else CKPTING=1; fi
fi
RECIPE="--soft-targets --soft-weight 1.0 --permute-options --rationale-weight 0.3 --rationale-max-tokens 192 --max-length 16384"
log(){ echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*"; }; stage(){ log "stage $1"; }; skip(){ [ -f $O/$1.DONE ]; }
die(){ log "FAILED: $*"; [ -d $O/lane ] && touch $O/lane/STOP; touch $O/FAILED; echo FAILED; exit 1; }   # the eval lane ends its current checkpoint, then exits
rm -f $O/FAILED
[ "$NGPU" -ge 8 ] || log "WARNING: $NGPU GPUs (planned for 8); lanes and eval workers scale down"
CK_FLAG=""; [ "$CKPTING" = 1 ] || CK_FLAG=--no-checkpointing
ckflag(){ [ -f $O/CKPTING_ON ] && echo "" || echo "$CK_FLAG"; }   # read at every attempt (lanes run in subshells)
oom_fallback(){ # log -> 0 when the failure was an OOM with checkpointing off: switch it on for every later attempt
  [ -n "$(ckflag)" ] && tail -n 80 "$1" | grep -qiE "out of memory|OutOfMemoryError" && { touch $O/CKPTING_ON; log "  OOM without checkpointing: gradient checkpointing ON from now on"; }; return 0; }
log "gradient checkpointing: $([ -n "$CK_FLAG" ] && echo "OFF (auto, GPU memory ${GMEM:-?} MiB; OOM -> ON)" || echo ON)"
csv(){ local IFS=,; echo "$*"; }
# GPUs: training uses TRAIN_GPUS; with EVAL_DURING_TRAIN=1 the last GPU is the eval lane from train-r1 until stage 7
EVAL_DURING_TRAIN=${EVAL_DURING_TRAIN:-0}; RESUME_LANE=${RESUME_LANE:-1}
ALL_GPUS=($(seq 0 $((NGPU-1)))); TRAIN_GPUS=("${ALL_GPUS[@]}"); LANE_GPU=${EVAL_LANE_GPU:-$((NGPU-1))}
if [ "$EVAL_DURING_TRAIN" = 1 ]; then
  if [ "$NGPU" -ge 2 ]; then TRAIN_GPUS=($(for g in "${ALL_GPUS[@]}"; do [ "$g" = "$LANE_GPU" ] || echo $g; done))
  else log "WARNING: EVAL_DURING_TRAIN=1 needs >= 2 GPUs; off"; EVAL_DURING_TRAIN=0; fi
fi
TRAIN_WORLD=${#TRAIN_GPUS[@]}
# pilot lanes (2 GPUs each) take the full run's micro-batches per step, so the winner's state IS the full run's at PILOT_STEPS
PER_STEP=$((TRAIN_WORLD*ACC)); PILOT_ACC=${PILOT_ACC:-$(( (PER_STEP+1)/2 ))}
RANK_X=${RANK_X:-64}; PILOT_STEPS=${PILOT_STEPS:-100}; PILOT_DEV_EVERY=${PILOT_DEV_EVERY:-$(( PILOT_STEPS/4 > 0 ? PILOT_STEPS/4 : 1 ))}
[ $((2*PILOT_ACC)) = $PER_STEP ] || log "  note: pilot lanes plan $((2*PILOT_ACC)) micro-batches per step, the full run $PER_STEP: the lane resume is inexact"

# ---------------------------------------------------------------------------------------------------------------- 0 checks
if ! skip checks; then stage checks
  for m in $V $FRESH $FLAGGED $HUMAN decision-p2b decision-v2-state-probe decision-v2-pairs-probe decision-v1.1-irrelevance-test decision-p2b-jevstyle-dev; do
    [ -s data/manifests/$m.jsonl ] || die "missing manifest data/manifests/$m.jsonl"; done
  python - "$SHIPPED" <<'PY' || die "INIT_ADAPTER_MISMATCH ($SHIPPED)"
import json, sys
from pathlib import Path
from torch_decision import TARGETS
a = Path(sys.argv[1]); c = json.loads((a / "adapter_config.json").read_text())
want = r'.*language_model.*\.(' + '|'.join(TARGETS) + ')'
missing = [f for f in ("adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json") if not (a / f).exists()]
assert c["r"] == 16 and c["lora_alpha"] == 32 and c["target_modules"] == want and not missing, (c["r"], c["lora_alpha"], c["target_modules"], missing)
print(f"  init adapter {a}: r16/alpha32, default targets, readout present")
PY
  python - $V $FRESH $FLAGGED $HUMAN <<'PY' || die "manifest check (held-out leakage or duplicates)"
import json, sys
V, *held = sys.argv[1:]
rows = [json.loads(l) for l in open(f"data/manifests/{V}.jsonl") if l.strip()]
ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate ids in the training manifest"
train_keys = set()
for r in rows:
    if r.get("partition") == "train":
        train_keys |= {r["id"], r["id"].split(":")[0]} | {str(r[k]) for k in ("candidate_id", "parent_id") if r.get(k)}
for h in held:
    hr = [json.loads(l) for l in open(f"data/manifests/{h}.jsonl") if l.strip()]
    hk = {x for r in hr for x in ({r["id"], r["id"].split(":")[0]} | {str(r[k]) for k in ("candidate_id", "parent_id") if r.get(k)})}
    hit = hk & train_keys
    print(f"  {h}: {len(hr)} rows; overlap with {V} train: {len(hit)}")
    assert not hit, f"{h} overlaps the training partition: {sorted(hit)[:5]}"
from collections import Counter
print(f"  {V}: {len(rows)} rows {dict(Counter(r.get('partition') for r in rows))}; image rows {sum(bool(r.get('images')) for r in rows)}; "
      f"target_probs {sum('target_probs' in r for r in rows)}; rationale {sum(bool(r.get('rationale')) for r in rows)}")
PY
  touch $O/checks.DONE; fi

# ---------------------------------------------------------------------------------------------------------------- 1 prep
if ! skip prep; then stage prep
  python cloud/p3/prep_manifests.py --version $V --out $O/manifests.json || die "prep_manifests"; touch $O/prep.DONE; fi
V255=$(python -c "import json;print(json.load(open('$O/manifests.json'))['version_255'])")
V256=$(python -c "import json;print(json.load(open('$O/manifests.json'))['version_256'])")
VORD=$(python -c "import json;print(json.load(open('$O/manifests.json'))['ordinal_dev'] or '')")
ROWS255=$(python -c "import json;print(json.load(open('$O/manifests.json'))['rows_255'])")
DEV2=${VORD:+--dev2-version $VORD --dev2-cases 400}
log "  manifests: 255-code lanes $V255, 256-code lanes $V256, ordinal dev ${VORD:-none}, 255-option train rows $ROWS255"

# ---------------------------------------------------------------------------------------------------------------- 2+3 parity + pilot
parity(){ skip parity && return 0; stage "parity (GPU $1)"; mkdir -p $O/parity
  printf '{"repo": "Qwen/Qwen3.5-4B", "revision": "local", "path": "%s"}\n' "$M4" > $O/parity/bundle.json
  CUDA_VISIBLE_DEVICES=$1 python scripts/p3/parity_readout256.py --backend torch --device cuda --adapter $SHIPPED --bundle $O/parity/bundle.json \
    --out $O/parity --candidates-dir data/p3/candidates-clean --jevbench evalsets/jevbench/datasets/public > $O/parity/run.log 2>&1
  local rc=$?; echo $rc > $O/parity/rc
  [ $rc = 0 ] && log "  parity: PASS (bit-identical on CUDA)" || log "  parity: FAIL rc $rc (256 lanes disabled; see $O/parity/run.log)"
  touch $O/parity.DONE; }
parity_ok(){ [ -f $O/parity/rc ] && [ "$(cat $O/parity/rc)" = 0 ]; }
lane(){ # lane gpus port W codes version [extra trainer flags, e.g. --expand-lora-rank 64]
  local l=$1 g=$2 port=$3 w=$4 c=$5 v=$6; shift 6; local xf=("$@"); mkdir -p $O/pilot/$l; skip pilot-$l && return 0
  for attempt in 1 2; do
    CUDA_VISIBLE_DEVICES=$g python -m torch.distributed.run --nproc_per_node=2 --master_port $port scripts/train_decision_lora_torch.py \
      --version $v --output $O/pilot/$l/train --model $M4 --init-adapter $SHIPPED --epochs $EPOCHS --lr $LR --warmup $WARMUP --seed 0 \
      --max-steps $PILOT_STEPS --dev-every $PILOT_DEV_EVERY --dev-cases 400 $DEV2 --select dev_loss --pad-multiple 64 --token-budget $TOKEN_BUDGET \
      --batch-size $BATCH --accumulate $PILOT_ACC --workers 8 $(ckflag) $RECIPE --ordinal-weight $w --readout-codes $c --dev-slices ${xf[@]+"${xf[@]}"} >> $O/pilot/$l/train.log 2>&1 \
      && { touch $O/pilot-$l.DONE; log "  pilot $l done"; return 0; }
    log "  pilot $l attempt $attempt failed"; tail -n 3 $O/pilot/$l/train.log | cut -c1-200; oom_fallback $O/pilot/$l/train.log; done
  touch $O/pilot-$l.FAILED; return 1; }
if ! skip pilot; then stage pilot
  if [ "$NGPU" -ge 8 ]; then
    parity 6 & PP=$!
    lane baseline 0,1 29520 0.0 255 $V255 & A=$!
    lane ordinal 2,3 29521 0.15 255 $V255 & B=$!
    lane rank64 4,5 29523 0.0 255 $V255 --expand-lora-rank $RANK_X & C=$!     # needs no parity: starts at once
    wait $PP
    if parity_ok && [ "$ROWS255" -gt 0 ]; then
      lane codes256 6,7 29522 0.0 256 $V256 & D=$!; wait $D
    else log "  256 lane skipped (parity $(cat $O/parity/rc 2>/dev/null), 255-option train rows $ROWS255)"; fi
    wait $A; wait $B; wait $C
  elif [ "$NGPU" -ge 6 ]; then  # 6 GPUs: three lanes at once, then the 256 lane
    lane baseline 0,1 29520 0.0 255 $V255 & A=$!
    lane ordinal 2,3 29521 0.15 255 $V255 & B=$!
    lane rank64 4,5 29523 0.0 255 $V255 --expand-lora-rank $RANK_X & C=$!
    wait $A; parity 0
    if parity_ok && [ "$ROWS255" -gt 0 ]; then lane codes256 0,1 29522 0.0 256 $V256 & D=$!; wait $D
    else log "  256 lane skipped (parity $(cat $O/parity/rc 2>/dev/null), 255-option train rows $ROWS255)"; fi
    wait $B; wait $C
  else  # fewer GPUs: two lanes at a time
    parity 0; lane baseline 0,1 29520 0.0 255 $V255 & A=$!; lane ordinal 2,3 29521 0.15 255 $V255; wait $A
    lane rank64 0,1 29523 0.0 255 $V255 --expand-lora-rank $RANK_X & A=$!
    if parity_ok && [ "$ROWS255" -gt 0 ]; then lane codes256 2,3 29522 0.0 256 $V256; fi; wait $A
  fi
  python cloud/p3/pilot_decision.py --pilot-dir $O/pilot --parity $O/parity/summary.json --manifest-info $O/manifests.json --steps $PILOT_STEPS \
    --out $O/pilot-decision.json || die "pilot decision"
  touch $O/pilot.DONE; fi
W=$(python -c "import json;print(json.load(open('$O/pilot-decision.json'))['ordinal_weight'])")
CODES=$(python -c "import json;print(json.load(open('$O/pilot-decision.json'))['readout_codes'])")
RANKX=$(python -c "import json;print(json.load(open('$O/pilot-decision.json')).get('expand_lora_rank') or 0)")
R1_XFLAGS=""; [ "$RANKX" -gt 0 ] && R1_XFLAGS="--expand-lora-rank $RANKX"   # r1 only: r2 starts from the r1 pick (already rank RANKX)
VFULL=$V255; [ "$CODES" = 256 ] && VFULL=$V256
log "  winners: --ordinal-weight $W --readout-codes $CODES ${R1_XFLAGS:-(rank 16)} (manifest $VFULL; lane $(python -c "import json;print(json.load(open('$O/pilot-decision.json')).get('lane') or 'none: COMBINED winners, r1 restarts from the shipped adapter')"))"

# ---------------------------------------------------------------------------------------------------------------- 4 / 6 training
cleanup_train(){ # only this run's trainer processes (pattern = its own output dir)
  pkill -f "train_decision_lora_torch.py --version [^ ]+ --output $1 " 2>/dev/null; sleep 5; }
train_run(){ # tag version init epochs lr warmup frac [trainer resume flags (--resume-from <lane>/last [--resume-inexact])]
  local tag=$1 v=$2 init=$3 ep=$4 lr=$5 wu=$6 frac=$7; shift 7; local resume=("$@"); local T=$O/$tag; mkdir -p $T
  skip train-$tag && return 0; stage "train-$tag"; rm -f $O/train-$tag.STOP
  [ -f $T/resume-fallback.json ] && resume=()   # an earlier attempt already fell back to the restart path: stay on it
  local plan; plan=$(python cloud/p3/plan_steps.py --version $v --epochs $ep --token-budget $TOKEN_BUDGET --batch-size $BATCH --accumulate $ACC \
    --world $TRAIN_WORLD --rationale-tokens 192 --frac $frac) || { log "  plan_steps failed"; return 1; }
  local every; every=$(python -c "import json,sys;print(json.loads(sys.argv[1])['dev_every'])" "$plan"); log "  $tag plan: $plan"
  python cloud/p3/ckpt_watch.py --train $T/train --ckpts $O/ckpts --prefix $tag --stop $O/train-$tag.STOP >> $T/watch.log 2>&1 & local WP=$!
  local ok=1
  for attempt in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$(csv "${TRAIN_GPUS[@]}") python -m torch.distributed.run --nproc_per_node=$TRAIN_WORLD --master_port 29500 scripts/train_decision_lora_torch.py \
      --version $v --output $T/train --model $M4 --init-adapter $init --epochs $ep --lr $lr --warmup $wu --seed 0 \
      --dev-every $every --dev-cases 600 $DEV2 --select mean_accuracy --pad-multiple 64 --token-budget $TOKEN_BUDGET \
      --batch-size $BATCH --accumulate $ACC --workers 16 $(ckflag) $RECIPE --ordinal-weight $W --readout-codes $CODES $( [ "$tag" = r1 ] && echo $R1_XFLAGS ) ${resume[@]+"${resume[@]}"} >> $T/train.log 2>&1 \
      && { ok=0; break; }
    log "  train $tag attempt $attempt failed (resume from last)"; tail -n 3 $T/train.log | cut -c1-200; oom_fallback $T/train.log; cleanup_train $T/train
    if [ ${#resume[@]} -gt 0 ] && [ ! -f $T/train/resumed_from.json ]; then   # the lane resume itself failed (nothing trained on it yet)
      log "  RESUME_FALLBACK: continuing the pilot lane failed before its first save; $tag restarts from $init (the pre-resume path)"
      echo "{\"fallback\": true, \"attempt\": $attempt, \"flags\": \"${resume[*]}\"}" > $T/resume-fallback.json
      rm -rf $T/train; resume=()
    fi; done
  touch $O/train-$tag.STOP; wait $WP
  [ $ok = 0 ] || { touch $O/train-$tag.FAILED; return 1; }
  ls -d $O/ckpts/$tag-s* > /dev/null 2>&1 || { log "  no snapshots for $tag"; return 1; }
  touch $O/train-$tag.DONE; log "done train-$tag: $(ls -d $O/ckpts/$tag-s* | xargs -n1 basename | tr '\n' ' ')"; }

ckpt_meta(){ # name dir -> "round step order"
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
name, d = sys.argv[1], Path(sys.argv[2])
if name == "shipped": print("0 0 0"); sys.exit()
m = json.loads((d / "ckpt.json").read_text()) if (d / "ckpt.json").exists() else {}
rnd = 2 if name.startswith(("r2", "soup-r2")) else 1
step, of = m.get("step") or 0, m.get("of") or 1
print(rnd, step, rnd + step / max(1, of))
PY
}
# the arguments every eval_checkpoint.py call shares (also handed to the eval lane)
EVAL_COMMON="--model $M4 --tracking-root $TR --gates-ref $GATES_REF --heldout-fresh $FRESH:$HELDOUT_PART --heldout-flagged $FLAGGED:$HELDOUT_PART --human-dev $HUMAN:$HELDOUT_PART"
eval_one(){ # name ckpt gpu panels [extra...]
  local n=$1 c=$2 g=$3 p=$4; shift 4; local meta; meta=($(ckpt_meta $n $c))
  python scripts/p3/eval_checkpoint.py --ckpt $c --name $n --out $O/eval/$n --gpu $g --port $((8765+g)) --panels $p \
    --round ${meta[0]} --step ${meta[1]} --order ${meta[2]} $EVAL_COMMON "$@" >> $O/eval/$n.$p.log 2>&1; }
eval_queue(){ # panels claims-tag gpus(csv) names... : one worker per GPU claims checkpoints (mkdir is atomic)
  local panels=$1 claims=$O/claims-$2 gpus=$3; shift 3; local names=("$@"); rm -rf $claims; mkdir -p $claims; local pids=() g
  [ ${#names[@]} -gt 0 ] || return 0
  for g in ${gpus//,/ }; do
    ( for n in "${names[@]}"; do
        mkdir $claims/$n 2>/dev/null || continue
        local c=$O/ckpts/$n extra=(); [ "$n" = shipped ] && { c=$SHIPPED; extra=(--fixed-calibration $SHIPPED/calibration.json); }
        [[ $n == soup-* ]] && extra=(--soup)
        log "  eval[$panels] $n on GPU $g"
        eval_one $n $c $g $panels "${extra[@]}" || log "  EVAL_FAILED $n ($panels; see $O/eval/$n.$panels.log)"
      done ) & pids+=($!); done
  for g in "${pids[@]}"; do wait $g; done; }

# ------------------------------------------------------------------------ eval lane (EVAL_DURING_TRAIN=1 only; cloud/p3/eval_lane.py)
lane_on(){ [ "$EVAL_DURING_TRAIN" = 1 ] && [ -n "${LANE_PID:-}" ]; }
if [ "$EVAL_DURING_TRAIN" = 1 ] && ! skip eval; then
  mkdir -p $O/lane
  python cloud/p3/eval_lane.py run --state $O/lane --ckpts $O/ckpts --eval-root $O/eval --gpu $LANE_GPU --shipped $SHIPPED \
    --final-ok r1=$O/sel-r1.DONE --parent-pid $$ --eval-args "$EVAL_COMMON" >> $O/lane/lane.log 2>&1 & LANE_PID=$!
  log "  eval lane on GPU $LANE_GPU (pid $LANE_PID); training on GPUs $(csv "${TRAIN_GPUS[@]}")"
fi

# ------------------------------------------------------------------------ continue the winning pilot lane (cloud/p3/lane_resume.py)
R1_RESUME=()
if [ "$RESUME_LANE" = 1 ] && ! skip train-r1; then
  flags=$(python cloud/p3/lane_resume.py --pilot-dir $O/pilot --decision $O/pilot-decision.json --steps $PILOT_STEPS --world $TRAIN_WORLD \
    --accumulate $ACC --out $O/lane-resume.json) || flags=""
  log "  $(python -c "import json;print(json.load(open('$O/lane-resume.json'))['summary'])" 2>/dev/null || echo 'lane resume: FALLBACK (lane_resume.py failed)')"
  read -ra R1_RESUME <<< "$flags"
fi
train_run r1 $VFULL $SHIPPED $EPOCHS $LR $WARMUP $CKPT_FRAC ${R1_RESUME[@]+"${R1_RESUME[@]}"} || die "train-r1"

# ---------------------------------------------------------------------------------------------------------------- 5 selection of r1
R1=($(ls -d $O/ckpts/r1-s* 2>/dev/null | xargs -n1 basename))
if ! skip sel-r1; then stage sel-r1
  SEL=("${R1[@]}")
  if lane_on; then   # the lane takes nothing new meanwhile; a checkpoint it is evaluating right now gets its selection panels from it
    for n in $(python cloud/p3/eval_lane.py pause --state $O/lane); do
      [[ " ${R1[*]} " == *" $n "* ]] || continue
      log "  sel-r1: waiting for the eval lane's selection panels of $n"
      while [ ! -s $O/eval/$n/selection.json ] && [ "$(python cloud/p3/eval_lane.py status --state $O/lane --name $n)" = running ] \
            && kill -0 $LANE_PID 2>/dev/null; do sleep 30; done
      [ "$(python cloud/p3/eval_lane.py status --state $O/lane --name $n)" = running ] && kill -0 $LANE_PID 2>/dev/null && SEL=($(for x in ${SEL[@]+"${SEL[@]}"}; do [ "$x" = "$n" ] || echo $x; done))
    done
  fi
  eval_queue selection selection $(csv "${TRAIN_GPUS[@]}") ${SEL[@]+"${SEL[@]}"}
  lane_on && python cloud/p3/eval_lane.py resume --state $O/lane
  python cloud/p3/select_checkpoint.py --eval-root $O/eval --names $(IFS=,; echo "${R1[*]}") --no-gates --out $O/r1-pick.json || die "r1 selection"
  touch $O/sel-r1.DONE; fi
R1BEST=$(python -c "import json;print(json.load(open('$O/r1-pick.json'))['pick'])"); log "  round-1 pick (selection only): $R1BEST"

# ---------------------------------------------------------------------------------------------------------------- 6 round 2
if ! skip round2; then stage round2
  ( set -e; mkdir -p $O/r2
    [ -s data/manifests/$SOURCE.jsonl ] || SOURCE=$V
    # the labelled train items of the source manifest, as candidate rows, split into 4 x NGPU shards (mine.py claims them)
    rm -rf $O/r2/pool; mkdir -p $O/r2/pool $O/r2/mine
    python - $SOURCE "$MINE_SHARDS" $O/r2/pool $((4*NGPU)) <<'PY'
import glob, json, sys
src, shards, outdir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
keys = set()
for l in open(f"data/manifests/{src}.jsonl"):
    if l.strip():
        r = json.loads(l)
        if r.get("partition") == "train":
            keys |= {r["id"], r["id"].split(":")[0]} | ({str(r["candidate_id"])} if r.get("candidate_id") else set())
rows = [l for f in sorted(glob.glob(shards)) for l in open(f) if l.strip()]
keep = [l for l in rows if json.loads(l)["id"] in keys]
print(f"  round-2 pool: {len(keep)} of {len(rows)} pool candidates are labelled train items of {src}")
if not keep: sys.exit("no pool candidate matches a labelled train id (check the id scheme of the manifest)")
for k in range(n):
    part = keep[k::n]
    if part:
        with open(f"{outdir}/r2pool-{k:03d}.jsonl", "w") as w:
            w.writelines(x if x.endswith("\n") else x + "\n" for x in part)
PY
    python scripts/p3/mine.py --help > /dev/null
    # mine.py loads a 255-code readout. A 256-code checkpoint is mined through a 255-row copy: the codebook is prefix-stable and the
    # pool has <= 254 options, so rows 1-255 give the same scores (the CUDA parity check above measured exactly this equivalence).
    AD=$O/ckpts/$R1BEST
    if [ "$CODES" = 256 ]; then AD=$O/r2/mine-adapter-255; rm -rf $AD; cp -a $O/ckpts/$R1BEST $AD
      python - $AD <<'PY'
import json, sys
from pathlib import Path
from safetensors.torch import load_file, save_file
a = Path(sys.argv[1]); w = load_file(str(a / "decision_readout.safetensors"))["weight"]
assert w.shape[0] == 256, w.shape
save_file({"weight": w[:255].contiguous()}, str(a / "decision_readout.safetensors"))
m = json.loads((a / "decision_readout.json").read_text()); m["codes"] = m["codes"][:255]
(a / "decision_readout.json").write_text(json.dumps(m, indent=2) + "\n"); print("  mining copy with a 255-row readout:", a)
PY
    fi
    pids=()
    for g in "${TRAIN_GPUS[@]}"; do
      cmd=${MINE_CMD//\{shards\}/$O/r2/pool/*.jsonl}; cmd=${cmd//\{model\}/$M4}; cmd=${cmd//\{gpu\}/$g}
      cmd=${cmd//\{adapter\}/$AD}; OUTD=$O/r2/mine; cmd=${cmd//\{out\}/$OUTD}
      echo "  mine[$g]: $cmd"
      CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > $O/r2/mine-$g.log 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    nshard=$(ls $O/r2/pool/*.jsonl | wc -l); ndone=$(ls $O/r2/mine/*.done 2>/dev/null | wc -l)
    echo "  re-mine: $ndone of $nshard shards done"; [ "$ndone" -eq "$nshard" ] || { echo "  re-mine incomplete (see $O/r2/mine-*.log)"; exit 1; }
    python cloud/p3/round2_manifest.py --mined "$O/r2/mine/*.flagged.jsonl" --source $SOURCE --base $VFULL --exclude $FRESH,$FLAGGED,$HUMAN \
      --out $V-r2 --max-new $R2_MAX_NEW --replay-ratio $R2_REPLAY --codes $CODES --report $O/r2/manifest-report.json
  ) >> $O/r2.log 2>&1 || { tail -n 20 $O/r2.log; die "round 2 re-mine / manifest (mandatory; see $O/r2.log)"; }
  grep "round-2 pool\|re-mine:" $O/r2.log | tail -n 2; cat $O/r2/manifest-report.json
  train_run r2 $V-r2 $O/ckpts/$R1BEST $R2_EPOCHS $R2_LR 5 $R2_FRAC || die "train-r2"
  touch $O/round2.DONE; fi

# ---------------------------------------------------------------------------------------------------------------- 7 eval
ALL=(shipped $(ls -d $O/ckpts/r1-s* $O/ckpts/r2-s* 2>/dev/null | xargs -n1 basename))
if ! skip eval; then stage "eval (${#ALL[@]} checkpoints on $NGPU GPUs)"
  if lane_on; then   # the lane stops taking checkpoints; the queue covers what it has not finished (and retries what failed there)
    RUN=$(python cloud/p3/eval_lane.py stop --state $O/lane)
    LEFT=($(python cloud/p3/eval_lane.py left --state $O/lane --names $(csv "${ALL[@]}")))
    log "  eval lane finished $(ls $O/lane/*.ok 2>/dev/null | wc -l) during training; still on: ${RUN:-none}; final queue: ${LEFT[*]:-nothing}"
    eval_queue all all $(csv "${TRAIN_GPUS[@]}") ${LEFT[@]+"${LEFT[@]}"} & EQ=$!
    wait $LANE_PID; wait $EQ
    AGAIN=($(python cloud/p3/eval_lane.py left --state $O/lane --lane-exited --names "$(csv $RUN)"))   # failed there, or the lane died
    [ ${#AGAIN[@]} -gt 0 ] && { log "  the lane's last checkpoint failed there: ${AGAIN[*]}"; eval_queue all again $(csv "${ALL_GPUS[@]}") ${AGAIN[@]+"${AGAIN[@]}"}; }
  else
    eval_queue all all $(csv "${ALL_GPUS[@]}") "${ALL[@]}"
  fi
  touch $O/eval.DONE; fi

# ---------------------------------------------------------------------------------------------------------------- 8 pick (+ soup fallback)
if ! skip pick; then stage pick
  NAMES=$(IFS=,; echo "${ALL[*]:1}")
  if python cloud/p3/select_checkpoint.py --eval-root $O/eval --names $NAMES --out $O/pick.json; then log "  $(python -c "import json;print(json.load(open('$O/pick.json'))['reason'])")"
  else
    BEST=$(python -c "import json;print(json.load(open('$O/pick.json'))['best_by_score'])")
    log "  no checkpoint passes every gate: soup fallback on $BEST with the shipped adapter (W = $SOUP_WEIGHTS)"
    SOUPS=()
    for w in $SOUP_WEIGHTS; do n=soup-$BEST-w$(python -c "print(int(round($w*100)))")
      [ -d $O/ckpts/$n ] || python cloud/p3/soup.py --shipped $SHIPPED --new $O/ckpts/$BEST --weight $w --out $O/ckpts/$n || die "soup $n"
      cp $O/ckpts/$BEST/ckpt.json $O/ckpts/$n/ 2>/dev/null; SOUPS+=($n); done
    eval_queue all soups $(csv "${ALL_GPUS[@]}") "${SOUPS[@]}"
    python cloud/p3/select_checkpoint.py --eval-root $O/eval --names $(IFS=,; echo "${SOUPS[*]}") --out $O/pick.json \
      && log "  soup pick: $(python -c "import json;print(json.load(open('$O/pick.json'))['reason'])")" \
      || log "  NO SHIPPABLE CHECKPOINT (soups included): keep 1.0 (plan: the data stays reusable)"
  fi
  touch $O/pick.DONE; fi

# ---------------------------------------------------------------------------------------------------------------- 9 final table + pack
stage final
python cloud/p3/final_table.py --eval-root $O/eval --tracking-root $TR --pick $O/pick.json --out $O/final-comparison > /dev/null || log "final table FAILED"
PICK=$(python -c "import json;print(json.load(open('$O/pick.json')).get('pick') or '')" 2>/dev/null)
[ -n "$PICK" ] && { rm -rf $O/picked-adapter; cp -a $O/ckpts/$PICK $O/picked-adapter; cp $O/eval/$PICK/calibration/single-chosen.json $O/picked-adapter/calibration.json 2>/dev/null
  cp $O/eval/$PICK/calibration/rot4-chosen.json $O/picked-adapter/calibration-rot4.json 2>/dev/null
  sha256sum $O/picked-adapter/*.safetensors | tee $O/picked-adapter/SHA256SUMS; }
cat $O/final-comparison.md 2>/dev/null
tar czf p3-train-results.tgz --warning=no-file-changed --exclude='*/trainer.pt' --exclude='*/train/best' --exclude='*/train/last' \
  --exclude='*/raw' -C p3 run tracking-only; rc=$?
[ $rc -le 1 ] && ls -la p3-train-results.tgz || die "pack"
echo ALL_DONE
