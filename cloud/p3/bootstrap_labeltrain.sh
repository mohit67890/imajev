#!/usr/bin/env bash
# Phase 3 LABEL-THEN-TRAIN on ONE 8xH100 pod (runbook reports/phase3/LABELTRAIN-RUNBOOK.md; plan docs/phase-3-plan.md).
#   A. LABEL  the pod serves the teacher Qwen/Qwen3.6-35B-A3B-FP8 (vLLM, one server per GPU, TP1, reasoning parser qwen3) and, on
#             WRITER_GPU (default 7; empty = no writer), the variant writer Qwen/Qwen3.6-27B-FP8, and labels an EXTRA queue that never
#             overlaps Azure's: the flagged rows the coordinator left waiting (queue-extra), later Azure's unlabelled leftovers (the
#             lead scp's them into labeltrain/inbox after Azure stops), and B/C writer variants + their re-verification.
#             scripts/p3/pod_teacher.py = azure_teacher.py's runner against the local servers (same prompts / schema / keep rules /
#             second solve / pilot gate; no key, no Azure ledger) -> data/p3/teacher-pod/. When the writer has no parent left
#             (WRITER_DONE) its GPU is switched to a teacher server.
#   B. (the Mac) review + manifests + train bundle, overlapping A.
#   C. TRAIN  after LABEL_DONE every vLLM server is stopped and GPU memory is checked free (GPUS_FREE, TRAIN_WAITING); the script
#             waits for labeltrain/TRAIN_BUNDLE_READY (+ a NEW short-lived reader key), then runs
#             cloud/p3/bootstrap_train.sh (SKIP_VENV=1, RUN_AFTER=1: fetch p3-train + p3-gatekit, eval sets, checks, then
#             cloud/p3/pod_run_train.sh: pilot lanes + resume, 2 epochs, CKPT_FRAC 0.25, eval, round 2 (labels cached, no teacher),
#             gates, final table) and waits for ALL_DONE / FAILED in p3/run/run.log.
# Layout:  (code + pool from p3-mine + label files from p3-label), venv (training venv, as
# bootstrap_train.sh), vllm-venv (vLLM only), hf (models), labeltrain (markers, logs, inbox).
# Small files scp'd from the Mac first: {bootstrap_labeltrain.sh, bootstrap_train.sh, transfer.py, p3.env} (no secrets)
# and ONE secret, deleted right after the bundle fetch: reader-key.json (gcs) or hf-token (hf).
# Launch: cd /workspace && setsid nohup bash bootstrap_labeltrain.sh >> labeltrain.log 2>&1 < /dev/null &
# Markers (labeltrain/): BUNDLES_FETCHED (revoke the reader key now), SERVERS_READY, WRITER_READY, CONSISTENCY_WARN,
# PILOT_GATE_PAUSED, LEFTOVERS_INSTALLED, WRITER_DONE, WRITER_SWITCHED, LABEL_DONE, GPUS_FREE, TRAIN_WAITING, TRAIN_STARTED,
# ALL_DONE | FAILED (the reason inside). Rerunning the same launch line resumes (stage files in labeltrain/stage/). Logs are in IST.
# Nothing here terminates the pod.
# Dry run / tests: LT_FAKE=1 with LT_WS, LT_CODE, LT_P3, LT_PY, LT_NGPU, LT_SERVE_CMD ("... {name} ... {port}"), LT_LABEL_DIR,
# LT_GPU_CHECK=0, TRAIN_CMD, RUN_LOG (scripts/p3/dryrun/labeltrain.sh). Paths must not contain spaces (the Mac dry run uses a
# space-free symlink to the repo).
set -uo pipefail
WS=${LT_WS:-/workspace}; mkdir -p $WS; cd $WS
M=$WS/labeltrain; mkdir -p $M/stage $M/pids $M/logs $M/inbox
export TZ=Asia/Kolkata                     # every python child prints IST
log(){ echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST labeltrain: $*"; }
mark(){ echo "$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S') IST ${2:-}" > $M/$1; log "MARKER $1 ${2:-}"; }
fail(){ log "FAILED: $*"; rm -f $WS/hf-token $WS/reader-key.json; echo "$(TZ=Asia/Kolkata date '+%H:%M:%S') IST $*" > $M/FAILED; echo FAILED; exit 1; }
done_(){ [ -f $M/stage/$1.DONE ]; }; stage_done(){ touch $M/stage/$1.DONE; }
[ -f $M/ALL_DONE ] && { log "ALL_DONE already"; exit 0; }
rm -f $M/FAILED
[ -f $WS/p3.env ] && { set -a; . $WS/p3.env; set +a; }
FAKE=${LT_FAKE:-0}
CODE=${LT_CODE:-$WS/imajev}; P3=${LT_P3:-$CODE/data/p3}; S=$P3/stream-pod; PY=${LT_PY:-$WS/venv/bin/python}
BACKEND=${BACKEND:-gcs}; GCS_BUCKET=${GCS_BUCKET:-gs://<bucket>/}; GCS_PREFIX=${GCS_PREFIX:-phase3}
MINE_BUNDLE=${MINE_BUNDLE:-p3-mine}; LABEL_BUNDLE=${LABEL_BUNDLE:-p3-label}
# pins (reports/phase3/LABELTRAIN-RUNBOOK.md "Pins"): vLLM 0.30.0 served Qwen3.6-35B-A3B (hybrid attention, reasoning parser qwen3) on
# our pods on 2026-09-23; the model card asks for vllm>=0.19.0
VLLM_VERSION=${VLLM_VERSION:-0.30.0}
TEACHER_REPO=${TEACHER_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}; TEACHER_REV=${TEACHER_REV:-95a723d08a9490559dae23d0cff1d9466213d989}
WRITER_REPO=${WRITER_REPO:-Qwen/Qwen3.6-27B-FP8}; WRITER_REV=${WRITER_REV:-e89b16ebf1988b3d6befa7de50abc2d76f26eb09}
QWEN_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; SHIPPED_REPO=${SHIPPED_REPO:-mohit67890/imajev-4b}; SHIPPED_REV=${SHIPPED_REV:-712891d1192c6491441a19bdea254d8a6e1048d0}
WRITER_GPU=${WRITER_GPU-last}; WRITER_THINK=${WRITER_THINK:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}; GPU_UTIL=${GPU_UTIL:-0.92}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}; PER_SERVER_CONC=${PER_SERVER_CONC:-96}
VLLM_EXTRA=${VLLM_EXTRA:-}; POD_USD_PER_HOUR=${POD_USD_PER_HOUR:-0}; CONSISTENCY=${CONSISTENCY:-1}; CONSISTENCY_MIN=${CONSISTENCY_MIN:-0.85}
RUN_LOG=${RUN_LOG:-$WS/p3/run/run.log}; TRAIN_CMD=${TRAIN_CMD:-}; LT_GPU_CHECK=${LT_GPU_CHECK:-1}
if [ "$FAKE" = 1 ]; then NGPU=${LT_NGPU:-3}; else NGPU=${LT_NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}; fi
[ "$NGPU" -ge 1 ] || fail "no GPUs"
[ "$WRITER_GPU" = last ] && WRITER_GPU=$((NGPU-1))   # the last GPU writes (7 on 8 GPUs, 5 on 6)
[ -n "$WRITER_GPU" ] && [ "$WRITER_GPU" -ge "$NGPU" ] && { log "WARNING: WRITER_GPU $WRITER_GPU >= $NGPU GPUs: no writer"; WRITER_GPU=""; }
T_PORT_BASE=${T_PORT_BASE:-8100}; TPORT(){ echo $((T_PORT_BASE + $1)); }; WPORT=${W_PORT:-8200}
TEACHER_EXTRA_ARGS=${TEACHER_EXTRA_ARGS:-}; VARIANTS_EXTRA_ARGS=${VARIANTS_EXTRA_ARGS:-}
TEACH_GPUS=(); for g in $(seq 0 $((NGPU-1))); do [ "$g" = "$WRITER_GPU" ] || TEACH_GPUS+=($g); done
SERVER_ARGS=(); for g in $(seq 0 $((NGPU-1))); do SERVER_ARGS+=(--server gpu$g=http://127.0.0.1:$(TPORT $g)/v1); done   # the writer GPU's route answers after the switch
export HF_HOME=$WS/hf PYTHONPATH=$CODE/src:$CODE/scripts TOKENIZERS_PARALLELISM=false
# DeepGEMM JIT needs nvcc >= 12.9; the cu12.8 image has none -> FP8 servers die at load. vLLM falls back to CUTLASS/Triton FP8.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0} VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}
HUB=$WS/hf/hub; TEACHER_PATH=$HUB/models--${TEACHER_REPO//\//--}/snapshots/$TEACHER_REV; WRITER_PATH=$HUB/models--${WRITER_REPO//\//--}/snapshots/$WRITER_REV
log "start: $NGPU GPUs, teacher GPUs ${TEACH_GPUS[*]}, writer GPU ${WRITER_GPU:-none} (think $WRITER_THINK), vllm $VLLM_VERSION, fake $FAKE"

# ------------------------------------------------------------------------------------------------ 1 venvs
if [ "$FAKE" != 1 ] && ! done_ venv; then
  python -m venv $WS/venv --system-site-packages && . $WS/venv/bin/activate || fail "venv"
  pip install -q "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" "safetensors>=0.8" "pillow>=11" "pydantic>=2,<3" scipy scikit-learn \
    fastapi "uvicorn[standard]" python-multipart httpx pyarrow "huggingface_hub>=1.32" google-cloud-storage flash-linear-attention tilelang 2>&1 | tail -n 1
  python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "training venv deps"
  python -m venv $WS/vllm-venv && $WS/vllm-venv/bin/pip install -q "vllm==$VLLM_VERSION" ninja packaging 2>&1 | tail -n 1
  got=$($WS/vllm-venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
  [ "$got" = "$VLLM_VERSION" ] || fail "vllm venv: got '$got', want $VLLM_VERSION"
  stage_done venv; log "venvs ready (vllm $got)"
fi
[ "$FAKE" = 1 ] || . $WS/venv/bin/activate
VLLM=${LT_VLLM:-$WS/vllm-venv/bin/vllm}

# ------------------------------------------------------------------------------------------------ 2 model downloads (background)
download(){ # tag repo rev [local_dir]
  done_ dl-$1 && return 0
  $PY - "$2" "$3" "${4:-}" >> $M/logs/download-$1.log 2>&1 <<'PY' && stage_done dl-$1 && log "downloaded $1 ($2@${3:0:8})"
import sys
from huggingface_hub import snapshot_download
repo, rev, local = sys.argv[1], sys.argv[2], sys.argv[3]
kw = {"local_dir": local} if local else {}
print(snapshot_download(repo, revision=rev, ignore_patterns=["*.md", ".gitattributes", "original/*", "mlx/*", "assets/*"],
                        max_workers=16, **kw), flush=True)
PY
}
if [ "$FAKE" != 1 ]; then
  ( download teacher $TEACHER_REPO $TEACHER_REV; [ -n "$WRITER_GPU" ] && download writer $WRITER_REPO $WRITER_REV
    download qwen4b Qwen/Qwen3.5-4B $QWEN_REV; download shipped $SHIPPED_REPO $SHIPPED_REV $WS/adapters/4b-shipped ) &
  DLPID=$!
fi

# ------------------------------------------------------------------------------------------------ 3 bundles (pool + label files)
if [ "$FAKE" != 1 ] && ! done_ fetch; then
  case $BACKEND in
    hf)  [ -s $WS/hf-token ] || fail "missing $WS/hf-token"; HF_TOKEN=$(cat $WS/hf-token); export HF_TOKEN
         FETCH_ARGS=(--backend hf --token-env HF_TOKEN); MREPO=(--repo ${MINE_REPO:?MINE_REPO}); LREPO=(--repo ${LABEL_REPO:?LABEL_REPO}) ;;
    gcs) [ -s $WS/reader-key.json ] || fail "missing $WS/reader-key.json"; chmod 600 $WS/reader-key.json
         export GOOGLE_APPLICATION_CREDENTIALS=$WS/reader-key.json; FETCH_ARGS=(--backend gcs --bucket $GCS_BUCKET --prefix $GCS_PREFIX)
         MREPO=(); LREPO=() ;;
    *) fail "BACKEND must be gcs or hf" ;;
  esac
  mkdir -p $CODE $WS/fetch
  $PY $WS/transfer.py fetch "${FETCH_ARGS[@]}" "${MREPO[@]}" --name $MINE_BUNDLE --dest $WS/fetch --extract $CODE || fail "fetch $MINE_BUNDLE"
  $PY $WS/transfer.py fetch "${FETCH_ARGS[@]}" "${LREPO[@]}" --name $LABEL_BUNDLE --dest $WS/fetch --extract $CODE || fail "fetch $LABEL_BUNDLE"
  rm -rf $WS/fetch; rm -f $WS/reader-key.json $WS/hf-token; unset GOOGLE_APPLICATION_CREDENTIALS HF_TOKEN
  stage_done fetch; mark BUNDLES_FETCHED "reader key deleted on the pod: revoke it on the Mac now"
fi

# ------------------------------------------------------------------------------------------------ 4 install the label files
LT_LABEL_DIR=${LT_LABEL_DIR:-$P3/labeltrain}
if ! done_ install; then
  [ -s $LT_LABEL_DIR/queue-extra.jsonl ] || fail "missing $LT_LABEL_DIR/queue-extra.jsonl (bundle $LABEL_BUNDLE)"
  mkdir -p $S $P3/teacher-pod $P3/variants-pod
  for f in queue-extra.jsonl writer-parents.jsonl queue-consistency.jsonl; do
    if [ ! -f $LT_LABEL_DIR/$f ]; then [ -f $S/$f ] || : > $S/$f
    elif [ ! -s $S/$f ]; then cp $LT_LABEL_DIR/$f $S/.$f.tmp && mv -f $S/.$f.tmp $S/$f || fail "install $f"; fi
    echo "{\"installed\": \"$f\"}" > $S/$f.closed; done
  $PY - "$LT_LABEL_DIR" "$CODE" <<'PY' || fail "gen-context install"
import json, shutil, sys
from pathlib import Path
src, code = Path(sys.argv[1]), Path(sys.argv[2])
m = src / "gen-context.json"
files = json.loads(m.read_text())["files"] if m.exists() else {}
for dest, rel in files.items():
    d = code / dest
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src / rel, d)
print(f"gen-context: {len(files)} files in place")
PY
  stage_done install
  log "installed: $(for f in queue-extra writer-parents queue-consistency; do echo -n "$f $(wc -l < $S/$f.jsonl) "; done)"
fi

# ------------------------------------------------------------------------------------------------ 5 servers
# setsid (Linux); a python shim where it is missing (the macOS dry run): the server becomes its own process group either way
if command -v setsid > /dev/null; then SETSID=(setsid); else SETSID=($PY -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])'); fi
healthy(){ curl -s -m 3 http://127.0.0.1:$1/v1/models 2>/dev/null | grep -q "\"$2\""; }
serve(){ # tag gpu port model_path served_name
  local tag=$1 g=$2 port=$3 path=$4 name=$5 cmd
  healthy $port $name && return 0
  if [ "$FAKE" = 1 ]; then cmd=${LT_SERVE_CMD//\{port\}/$port}; cmd=${cmd//\{name\}/$name}; read -ra cmd <<< "$cmd"
  else
    [ -d "$path" ] || { log "MISSING model $path"; return 1; }
    cmd=($VLLM serve $path --served-model-name $name --host 127.0.0.1 --port $port --max-model-len $MAX_MODEL_LEN
         --gpu-memory-utilization $GPU_UTIL --max-num-seqs $MAX_NUM_SEQS --reasoning-parser qwen3
         $VLLM_EXTRA)
    if [ "$name" = writer ]; then cmd+=(--limit-mm-per-prompt '{"image": 0, "video": 0}'); else cmd+=(--limit-mm-per-prompt '{"image": 2, "video": 0}'); fi
  fi
  PATH=$WS/vllm-venv/bin:$PATH CUDA_VISIBLE_DEVICES=$g "${SETSID[@]}" "${cmd[@]}" >> $M/logs/vllm-$tag.log 2>&1 < /dev/null &
  echo $! > $M/pids/$tag.pid; log "  serving $name on GPU $g :$port (pid $!, log logs/vllm-$tag.log)"; }
wait_up(){ # port name seconds tag
  local i; for i in $(seq 1 $(( $3 / 5 ))); do healthy $1 $2 && return 0
    [ -f $M/pids/$4.pid ] && ! kill -0 $(cat $M/pids/$4.pid) 2>/dev/null && { log "  $4 died while loading (tail logs/vllm-$4.log):"; tail -n 5 $M/logs/vllm-$4.log | cut -c1-200; return 1; }
    sleep 5; done; return 1; }
stop_server(){ # tag
  local f=$M/pids/$1.pid pg; [ -f $f ] || return 0; pg=$(cat $f)
  kill -TERM -- -$pg 2>/dev/null; for i in $(seq 1 30); do kill -0 -- -$pg 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$pg 2>/dev/null; rm -f $f; log "  stopped $1"; }
wait_dl(){ local i; for i in $(seq 1 1440); do done_ dl-$1 && return 0; [ "$FAKE" = 1 ] && return 0
  kill -0 ${DLPID:-0} 2>/dev/null || { done_ dl-$1 || return 1; }; sleep 5; done; return 1; }
if [ ! -f $M/LABEL_DONE ]; then
  wait_dl teacher || fail "teacher download (see logs/download-teacher.log)"
  g0=${TEACH_GPUS[0]}
  serve t$g0 $g0 $(TPORT $g0) $TEACHER_PATH teacher && wait_up $(TPORT $g0) teacher 2400 t$g0 || fail "first teacher server (GPU $g0) did not come up"
  for g in "${TEACH_GPUS[@]:1}"; do serve t$g $g $(TPORT $g) $TEACHER_PATH teacher; done
  up=1; for g in "${TEACH_GPUS[@]:1}"; do if wait_up $(TPORT $g) teacher 1200 t$g; then up=$((up+1)); else log "WARNING teacher GPU $g not up (the watchdog retries)"; fi; done
  mark SERVERS_READY "$up/${#TEACH_GPUS[@]} teacher servers (ports $(TPORT ${TEACH_GPUS[0]})..), model $TEACHER_REPO@${TEACHER_REV:0:8}, max-model-len $MAX_MODEL_LEN"
fi

# ------------------------------------------------------------------------------------------------ 6 label processes + monitor
TSTAT=$P3/teacher-pod/status.json
teacher_run(){ # the main pod teacher; rc -> stage/teacher.rc
  ( cd $CODE && $PY scripts/p3/pod_teacher.py run "${SERVER_ARGS[@]}" --model teacher --out $P3/teacher-pod --data-root $CODE/data \
      --queue $S/queue-extra.jsonl --queue $S/queue-leftover.jsonl --queue $S/queue-variants.jsonl --follow \
      --per-server-concurrency $PER_SERVER_CONC --pod-usd-per-hour $POD_USD_PER_HOUR --status-secs 30 $TEACHER_EXTRA_ARGS >> $M/logs/teacher.log 2>&1 &
    c=$!; echo $c > $M/pids/teacher-py.pid; wait $c; echo $? > $M/stage/teacher.rc ) & echo $! > $M/pids/teacher-run.pid; log "pod teacher started (log logs/teacher.log)"; }
consistency_run(){
  ( cd $CODE && $PY scripts/p3/pod_teacher.py run "${SERVER_ARGS[@]}" --model teacher --out $P3/teacher-pod-consistency --data-root $CODE/data \
      --queue $S/queue-consistency.jsonl --no-pilot-gate --per-server-concurrency 8 --status-secs 60 $TEACHER_EXTRA_ARGS >> $M/logs/consistency.log 2>&1
    $PY scripts/p3/pod_teacher.py compare --out $P3/teacher-pod-consistency --queue $S/queue-consistency.jsonl >> $M/logs/consistency.log 2>&1
    echo $? > $M/stage/consistency.rc ) & echo $! > $M/pids/consistency-run.pid; log "FP8-vs-bf16 consistency sample started (log logs/consistency.log)"; }
variants_run(){
  local w=(--no-writer)
  [ -n "$WRITER_GPU" ] && [ ! -f $M/WRITER_FAILED ] && { w=(--writer-base-url http://127.0.0.1:$WPORT/v1 --writer-model writer); [ "$WRITER_THINK" = 1 ] || w+=(--writer-no-think); }
  ( cd $CODE && $PY scripts/p3/variants.py "${w[@]}" --results $P3/teacher-pod/results.jsonl --results $S/writer-parents.jsonl \
      --results $S/writer-parents-late.jsonl --teacher-out $P3/teacher-pod --out $P3/variants-pod --queue $S/queue-variants.jsonl \
      --pool-shards "$CODE/data/p3/pool/shards/*.jsonl" --upstream-queues $S/queue-extra.jsonl $S/queue-leftover.jsonl \
      $S/writer-parents.jsonl $S/writer-parents-late.jsonl --writer-done-marker $M/WRITER_DONE --follow --interval ${VARIANTS_INTERVAL:-120} \
      --stop-when-closed --close $VARIANTS_EXTRA_ARGS >> $M/logs/variants.log 2>&1 &
    c=$!; echo $c > $M/pids/variants-py.pid; wait $c; echo $? > $M/stage/variants.rc ) & echo $! > $M/pids/variants-run.pid; log "variants started (${w[*]}; log logs/variants.log)"; }
alive(){ [ -f $M/pids/$1.pid ] && kill -0 $(cat $M/pids/$1.pid) 2>/dev/null; }
tstate(){ python3 -c "import json;print(json.load(open('$TSTAT')).get('state',''))" 2>/dev/null; }
restarts(){ cat $M/stage/restarts-$1 2>/dev/null || echo 0; }; bump(){ echo $(( $(restarts $1) + 1 )) > $M/stage/restarts-$1; restarts $1; }
rm -f $M/stage/restarts-*
writer_on(){ [ -n "$WRITER_GPU" ] && [ ! -f $M/WRITER_DONE ] && [ ! -f $M/WRITER_FAILED ]; }
start_writer(){ # non-blocking: starts the writer server once its weights are on disk
  writer_on || return 0; alive w && return 0; { done_ dl-writer || [ "$FAKE" = 1 ]; } || return 1
  serve w $WRITER_GPU $WPORT $WRITER_PATH writer; }
if [ ! -f $M/LABEL_DONE ]; then
  if [ "$CONSISTENCY" = 1 ] && [ ! -f $M/stage/consistency.rc ] && ! alive consistency-run && [ -s $S/queue-consistency.jsonl ]; then consistency_run; fi
  rm -f $M/stage/teacher.rc $M/stage/variants.rc
  alive teacher-run || teacher_run
  start_writer
  alive variants-run || variants_run          # programmatic parents start at once; writer rounds wait until the writer answers
  last_snap=0; cons_seen=0
  while true; do
    now=$(date +%s)
    # a) Azure leftovers from the lead (scp into $M/inbox)
    if [ ! -f $M/LEFTOVERS_INSTALLED ]; then
      $PY $CODE/cloud/p3/labeltrain_watch.py inbox --inbox $M/inbox --stream $S >> $M/logs/inbox.log 2>&1
      rc=$?; [ $rc = 0 ] && mark LEFTOVERS_INSTALLED "$(cat $S/queue-leftover.jsonl.closed 2>/dev/null)"
      [ $rc = 2 ] && log "inbox: checksum mismatch (retrying; see logs/inbox.log)"
    fi
    # a2) the writer answered at least once: WRITER_READY (checked before the switch below, so a writer that finishes within
    #     one loop turn is still recorded as ready)
    if [ -n "$WRITER_GPU" ] && [ ! -f $M/WRITER_READY ] && [ ! -f $M/WRITER_SWITCHED ] && [ ! -f $M/WRITER_FAILED ] && healthy $WPORT writer; then
      mark WRITER_READY "GPU $WRITER_GPU :$WPORT $WRITER_REPO@${WRITER_REV:0:8} think $WRITER_THINK"; fi
    # b) writer done -> its GPU becomes a teacher server
    if [ -n "$WRITER_GPU" ] && [ -f $M/WRITER_DONE ] && [ ! -f $M/WRITER_SWITCHED ]; then
      log "writer done: switching GPU $WRITER_GPU to a teacher server"; stop_server w; sleep 5
      serve t$WRITER_GPU $WRITER_GPU $(TPORT $WRITER_GPU) $TEACHER_PATH teacher && wait_up $(TPORT $WRITER_GPU) teacher 1800 t$WRITER_GPU \
        && mark WRITER_SWITCHED "teacher on GPU $WRITER_GPU :$(TPORT $WRITER_GPU)" || log "WARNING switch: teacher on GPU $WRITER_GPU not up (watchdog retries)"
      touch $M/WRITER_SWITCHED.tried
    fi
    # c) watchdog: dead servers are restarted (3 times each); a writer that cannot be served falls back to no-writer variants
    for g in "${TEACH_GPUS[@]}" $( [ -f $M/WRITER_SWITCHED.tried ] && echo $WRITER_GPU ); do
      if [ -f $M/pids/t$g.pid ] && ! alive t$g && [ "$(restarts t$g)" -lt 3 ]; then
        log "WATCHDOG teacher GPU $g died: restart $(bump t$g)/3"; rm -f $M/pids/t$g.pid
        serve t$g $g $(TPORT $g) $TEACHER_PATH teacher; fi; done
    if writer_on; then
      if healthy $WPORT writer; then [ -f $M/WRITER_READY ] || mark WRITER_READY "GPU $WRITER_GPU :$WPORT $WRITER_REPO@${WRITER_REV:0:8} think $WRITER_THINK"
      elif [ -f $M/pids/w.pid ] && ! alive w; then
        if [ "$(restarts w)" -lt 3 ]; then log "WATCHDOG writer died: restart $(bump w)/3"; rm -f $M/pids/w.pid; start_writer
        else mark WRITER_FAILED "the writer server failed 4 times (logs/vllm-w.log): B/C parents left get no writer variants"
          for f in variants-run variants-py; do [ -f $M/pids/$f.pid ] && kill $(cat $M/pids/$f.pid) 2>/dev/null; done; sleep 5
          rm -f $M/stage/variants.rc; touch $M/WRITER_DONE; variants_run; fi
      elif [ ! -f $M/pids/w.pid ]; then start_writer; fi
    fi
    # d) status / throughput every 5 min; pilot gate
    if [ $((now - last_snap)) -ge ${SNAP_SECS:-300} ]; then last_snap=$now
      $PY $CODE/cloud/p3/labeltrain_watch.py snapshot --p3 $P3 --log $M/throughput.log > /dev/null 2>&1
      if [ $? = 5 ]; then [ -f $M/PILOT_GATE_PAUSED ] || mark PILOT_GATE_PAUSED "read $P3/teacher-pod/PILOT_GATE_ALERT.txt; resume: pod_teacher.py resume"
      else rm -f $M/PILOT_GATE_PAUSED; fi
      tail -n 1 $M/throughput.log; fi
    # e) FP8-vs-bf16 consistency result
    if [ $cons_seen = 0 ] && [ -f $M/stage/consistency.rc ]; then cons_seen=1
      ag=$(python3 -c "import json;print(json.load(open('$P3/teacher-pod-consistency/consistency.json')).get('solve1_argmax_agreement') or 0)" 2>/dev/null || echo 0)
      log "FP8-vs-bf16 consistency: solve-1 argmax agreement $ag (min $CONSISTENCY_MIN); details teacher-pod-consistency/consistency.json"
      python3 -c "import sys;sys.exit(0 if float('$ag') >= float('$CONSISTENCY_MIN') else 1)" || mark CONSISTENCY_WARN "agreement $ag < $CONSISTENCY_MIN (labelling continues; lead decides)"; fi
    # f) the label processes
    if [ -f $M/stage/teacher.rc ]; then
      if [ "$(cat $M/stage/teacher.rc)" = 0 ] && [ "$(tstate)" = done ]; then :
      elif [ "$(restarts teacher)" -lt 3 ]; then
        log "pod teacher exited (rc $(cat $M/stage/teacher.rc), state $(tstate)): restart $(bump teacher)/3 (resumes)"; tail -n 3 $M/logs/teacher.log | cut -c1-240
        rm -f $M/stage/teacher.rc; teacher_run
      else fail "pod teacher failed 3 times (see $M/logs/teacher.log)"; fi
    fi
    if [ -f $M/stage/variants.rc ] && [ "$(cat $M/stage/variants.rc)" != 0 ]; then
      if [ "$(restarts variants)" -lt 3 ]; then
        log "variants exited rc $(cat $M/stage/variants.rc): restart $(bump variants)/3"; tail -n 3 $M/logs/variants.log | cut -c1-240
        rm -f $M/stage/variants.rc; variants_run
      else fail "variants failed 3 times (see $M/logs/variants.log)"; fi
    fi
    if [ -f $M/stage/teacher.rc ] && [ -f $M/stage/variants.rc ] && [ "$(cat $M/stage/variants.rc)" = 0 ]; then break; fi
    sleep ${LOOP_SECS:-30}
  done
  alive consistency-run && { log "waiting for the consistency sample to finish"; while alive consistency-run; do sleep 10; done; }
  # ---------------------------------------------------------------------------------------------- 7 finalize + pack
  (cd $CODE && $PY scripts/p3/pod_teacher.py finalize --out $P3/teacher-pod) > $M/logs/finalize.json 2>&1 || fail "finalize (see logs/finalize.json)"
  $PY $CODE/cloud/p3/labeltrain_watch.py snapshot --p3 $P3 --log $M/throughput.log > /dev/null 2>&1
  $PY $CODE/cloud/p3/labeltrain_watch.py summary --p3 $P3 > $M/label-summary.json 2>&1
  $PY - $WS/p3-label-results.tgz $P3 $M <<'PY' || fail "pack p3-label-results.tgz"
import sys, tarfile
from pathlib import Path
out, p3, m = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
with tarfile.open(out + ".tmp", "w:gz", compresslevel=6) as tar:
    for d in ("teacher-pod", "variants-pod", "stream-pod", "teacher-pod-consistency"):
        if (p3 / d).is_dir():
            tar.add(p3 / d, arcname=d)
    # pod markers + logs as labeltrain-pod/ (extracting into the Mac's data/p3 must not touch data/p3/labeltrain, the bundle input)
    tar.add(m, arcname="labeltrain-pod", filter=lambda ti: None if ti.name.startswith(("labeltrain-pod/inbox", "labeltrain-pod/pids")) else ti)
Path(out + ".tmp").rename(out)
print(f"packed {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")
PY
  mark LABEL_DONE "$(python3 -c "import json;d=json.load(open('$M/label-summary.json'));print('results',d['results'],'kept',d['kept_training_rows'],'heldout',d['heldout_results'])" 2>/dev/null); pull p3-label-results.tgz"
fi

# ------------------------------------------------------------------------------------------------ 8 stop every server, GPUs free
if [ ! -f $M/GPUS_FREE ]; then
  for f in $M/pids/t*.pid $M/pids/w.pid; do [ -f $f ] && stop_server $(basename $f .pid); done
  [ "$FAKE" = 1 ] || pkill -f "vllm serve $HUB" 2>/dev/null
  if [ "$LT_GPU_CHECK" = 1 ]; then
    ok=0; for i in $(seq 1 24); do $PY $CODE/cloud/p3/labeltrain_watch.py gpu-free >> $M/logs/gpu-free.log 2>&1 && { ok=1; break; }; sleep 5; done
    if [ $ok = 0 ]; then
      log "GPU memory still in use: killing the remaining compute processes"
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 10
      $PY $CODE/cloud/p3/labeltrain_watch.py gpu-free >> $M/logs/gpu-free.log 2>&1 || fail "GPU memory not freed after stopping vLLM (see logs/gpu-free.log)"
    fi
    tail -n 1 $M/logs/gpu-free.log
  fi
  mark GPUS_FREE "every vLLM server stopped"
fi
# ------------------------------------------------------------------------------------------------ 8b direct-image baseline pass
# A SHORT pass of the shipped 4B over the DIRECT constructed image items (charts, docimg, inventory, safety, geometry, screens: they go
# straight to training; this is a baseline measurement, NOT a filter) -> data/p3/mine-direct/{pool,out,summary.json},
# p3-mine-direct.tgz, marker DIRECT_BASELINE. It runs while the pod waits for the train bundle (GPUs free) when the candidate
# files came with p3-label, else right after the train bundle is fetched (build it with their --candidates), before training starts.
DIRECT_MINE=${DIRECT_MINE:-${CHART_MINE:-1}}; DIRECT_GROUPS=${DIRECT_GROUPS:-"charts docimg inventory safety geometry screens"}
MD=$P3/mine-direct; DIRECT_DIR=${DIRECT_DIR:-$CODE/data/p3/candidates-clean}
direct_baseline(){ # [final]: with "final", a missing file set is recorded as skipped
  [ "$DIRECT_MINE" = 1 ] || return 0; done_ direct && return 0
  local have=() grp f g p cmd pids=() t0
  for grp in $DIRECT_GROUPS; do f=$DIRECT_DIR/I-$grp.jsonl; [ -s $f ] && have+=($grp=$f); done
  if [ ${#have[@]} = 0 ]; then
    [ "${1:-}" = final ] && { log "direct baseline: no I-<group>.jsonl candidate files on the pod: skipped"; stage_done direct; }
    return 0; fi
  { wait_dl qwen4b && wait_dl shipped; } || log "WARNING direct baseline: 4B / shipped adapter download not finished"
  local M4=$HUB/models--Qwen--Qwen3.5-4B/snapshots/$QWEN_REV; rm -rf $MD/pool; mkdir -p $MD/pool $MD/out
  $PY - $MD $((4*NGPU)) "${have[@]}" <<'PY' || { log "WARNING direct baseline: split failed"; return 1; }
import json, sys
out, n, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows, group = [], {}
for spec in specs:
    grp, path = spec.split("=", 1)
    for l in open(path):
        if l.strip():
            rows.append(l if l.endswith("\n") else l + "\n"); group[json.loads(l)["id"]] = grp
for k in range(n):
    part = rows[k::n]
    if part:
        open(f"{out}/pool/direct-{k:03d}.jsonl", "w").writelines(part)
json.dump(group, open(f"{out}/groups.json", "w"))
print(f"direct baseline: {len(rows)} rows from {len(specs)} groups in {min(n, len(rows))} shards")
PY
  local tpl=${LT_DIRECT_MINE_CMD:-"$PY scripts/p3/mine.py --backend torch --shards '{shards}' --out {out} --model-path {model} --adapter {adapter} --worker-id direct-gpu{gpu} --device cuda"}
  t0=$(date +%s)
  for g in $(seq 0 $((NGPU-1))); do
    cmd=${tpl//\{shards\}/$MD/pool/*.jsonl}; cmd=${cmd//\{out\}/$MD/out}; cmd=${cmd//\{model\}/$M4}; cmd=${cmd//\{adapter\}/$WS/adapters/4b-shipped}; cmd=${cmd//\{gpu\}/$g}
    ( cd $CODE && CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" >> $M/logs/mine-direct-$g.log 2>&1 ) & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  $PY - $MD > $MD/summary.txt 2>&1 <<'PY'
import glob, json, sys
from collections import defaultdict
d = sys.argv[1]; group = json.load(open(f"{d}/groups.json"))
rows = [json.loads(l) for f in glob.glob(f"{d}/out/*.scores.jsonl") for l in open(f) if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
fam, grp = defaultdict(list), defaultdict(list)
for r in ok:
    fam[(group.get(r["id"], "?"), r.get("family"))].append(bool(r.get("correct")))
    grp[group.get(r["id"], "?")].append(bool(r.get("correct")))
acc = lambda v: round(100 * sum(v) / len(v), 2)
rep = {"rows": len(rows), "ok": len(ok), "shipped_4b_accuracy": {g: {"n": len(v), "acc": acc(v)} for g, v in sorted(grp.items())},
       "by_family": {f"{g}/{f}": {"n": len(v), "acc": acc(v)} for (g, f), v in sorted(fam.items(), key=str)},
       "flag_rate": round(sum(bool(r.get("flagged")) for r in ok) / max(1, len(ok)), 4),
       "note": "baseline of the shipped imajev-4b 1.0 (single pass, raw) on the direct constructed image items; NOT a filter"}
json.dump(rep, open(f"{d}/summary.json", "w"), indent=1)
print(" ".join(f"{g} {v['acc']}% (n {v['n']})" for g, v in rep["shipped_4b_accuracy"].items()), f"| {len(rows)} rows")
PY
  tar czf $WS/p3-mine-direct.tgz -C $P3 mine-direct/out mine-direct/summary.json mine-direct/groups.json 2>/dev/null
  stage_done direct; mark DIRECT_BASELINE "$(tail -n 1 $MD/summary.txt) in $(( $(date +%s) - t0 )) s; p3-mine-direct.tgz"; }
direct_baseline
[ -f $M/TRAIN_WAITING ] || mark TRAIN_WAITING "build + push p3-train / p3-gatekit, scp a NEW reader key, then touch $M/TRAIN_BUNDLE_READY"

# ------------------------------------------------------------------------------------------------ 9 wait for the train bundle
n=0; while true; do
  if [ -f $M/TRAIN_BUNDLE_READY ]; then
    if [ -n "$TRAIN_CMD" ] || [ -f $M/TRAIN_STARTED ] || grep -q BOOTSTRAP_DONE $WS/bootstrap.log 2>/dev/null || [ -s $WS/reader-key.json ] || [ -s $WS/hf-token ]; then break; fi
    [ $((n % 20)) = 0 ] && log "TRAIN_BUNDLE_READY is there but no $WS/reader-key.json (gcs) / hf-token (hf) yet: waiting"
  else [ $((n % 20)) = 0 ] && log "waiting for $M/TRAIN_BUNDLE_READY (the pod idles; GPUs free)"; fi
  n=$((n+1)); sleep ${WAIT_SECS:-30}; done

# ------------------------------------------------------------------------------------------------ 10 train (bootstrap_train.sh -> pod_run_train.sh)
if ! grep -q BOOTSTRAP_DONE $WS/bootstrap.log 2>/dev/null; then
  mark TRAIN_STARTED "bootstrap_train.sh (SKIP_VENV=1 RUN_AFTER=0), then the direct baseline if still due, then pod_run_train.sh"
  if [ -n "$TRAIN_CMD" ]; then bash -c "$TRAIN_CMD" >> $WS/bootstrap.log 2>&1
  else
    BT=$WS/bootstrap_train.sh; [ -f $BT ] || BT=$CODE/cloud/p3/bootstrap_train.sh
    wait ${DLPID:-0} 2>/dev/null
    SKIP_VENV=1 RUN_AFTER=0 bash $BT >> $WS/bootstrap.log 2>&1
  fi
  rm -f $WS/reader-key.json $WS/hf-token
  grep -q BOOTSTRAP_DONE $WS/bootstrap.log || fail "bootstrap_train.sh: $(grep -h BOOTSTRAP_FAILED $WS/bootstrap.log | tail -n 1) (fix, then rerun this script; it resumes here)"
  direct_baseline final                    # the candidate files may have come with the train bundle
  if [ -z "$TRAIN_CMD" ] && grep -q "started cloud/p3/pod_run_train.sh" $WS/bootstrap.log; then
    log "bootstrap_train.sh already started pod_run_train.sh (RUN_AFTER set in p3.env?): not starting a second one"
  elif [ -z "$TRAIN_CMD" ]; then
    mkdir -p $(dirname $RUN_LOG); (cd $WS && "${SETSID[@]}" nohup bash $CODE/cloud/p3/pod_run_train.sh >> $RUN_LOG 2>&1 < /dev/null &)
    log "started cloud/p3/pod_run_train.sh (log $RUN_LOG)"
  fi
elif ! tail -n 1 $RUN_LOG 2>/dev/null | grep -qE "ALL_DONE|FAILED" && ! pgrep -f "cloud/p3/pod_run_train.sh" > /dev/null; then
  log "bootstrap done earlier, pod_run_train.sh not running: restarting it (it resumes from its stage markers)"
  mkdir -p $(dirname $RUN_LOG); (cd $WS && "${SETSID[@]}" nohup bash $CODE/cloud/p3/pod_run_train.sh >> $RUN_LOG 2>&1 < /dev/null &)
fi
log "training running: watching $RUN_LOG"
while true; do
  last=$(tail -n 1 $RUN_LOG 2>/dev/null)
  case "$last" in *ALL_DONE*) mark ALL_DONE "pull p3-train-results.tgz + p3-label-results.tgz, then terminate"; exit 0 ;;
                  *FAILED*) fail "pod_run_train.sh: $(grep -h 'FAILED' $RUN_LOG | tail -n 1)" ;; esac
  sleep ${WAIT_SECS:-60}; done
