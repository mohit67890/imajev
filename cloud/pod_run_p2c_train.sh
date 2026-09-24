#!/usr/bin/env bash
# Phase-2c training + evaluation on the 8xH100 pod (7 also works) (docs/phase-2c-plan.md incl. amendments, docs/phase-2c-runbook.md).
# Order: manifest check -> SMOKE (GPUs 4-6, blocking; SMOKE_MISMATCH aborts before any training) -> pipelined like
# cloud/pod_run_p2b_train_pipelined.sh: a TRAIN lane on GPUs 0-3 runs the runs back to back while an EVAL lane on GPUs 4-6 (+7 once
# the Eikos run has freed it) evaluates each finished run. On an 8-GPU pod the Eikos-4B run uses GPU 7 from the start, in parallel with the
# smoke. Runs (RUNS env, default all four, in this order):
#   4b-delta  from adapters/4b-p2b-best (r16), decision-p2c, 2 epochs, lr 2e-5, warm-up 10, budget 10000, no checkpointing
#   4b-fresh  NO --init-adapter: new LoRA r64 / alpha 128 on every language-layer linear incl. the DeltaNet projections
#             (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj + attention + MLP), decision-p2c-fresh, 1 epoch, lr 1e-4,
#             warm-up 20 then cosine (trainer: to 10%), AdamW wd 0 (trainer default), gradient checkpointing ON, micro-batch 8 x 1 x 4 GPUs
#             (~32 items per step, as Eikos). LoRA dropout: the trainer has no flag and hard-codes 0.0 (scripts/torch_decision.py
#             add_lora), so the plan's 0.05 is NOT applied (trainer not modified by design).
#   9b-delta  from adapters/9b-p2b-best, decision-p2c, 2 epochs, lr 2e-5, warm-up 10, budget 7500, checkpointing ON
#   2b-delta  from adapters/2b-p2b-best, decision-p2c, 2 epochs, lr 2e-5, warm-up 10, budget 12000, no checkpointing
#   every run: --soft-targets --soft-weight 1.0 --rationale-weight 0.3 --rationale-max-tokens 192 --permute-options,
#              dev = <manifest> dev, dev2 = decision-p2c-judge-dev, --select mean_accuracy
# SMOKE (once, before the train lane starts): 5 optimizer steps (--max-steps 5) of the 4b-delta setup (decision-p2c, 4b-p2b-best),
#          flags OFF (GPU 4) vs flags ON (GPU 5) vs flags ON with --soft-weight 0 (GPU 6); prints the step-0 dev loss/accuracy of each.
#          Accuracy must match everywhere; OFF and ON+sw0 must match in loss (dev rows never get a rationale or a permutation; with
#          --soft-targets the dev loss of rows carrying target_probs is the soft CE, so ON differs from OFF by exactly that); ON must log a
#          rationale loss and OFF must not. SMOKE_OK -> training starts; SMOKE_MISMATCH -> the script stops (SMOKE_IGNORE=1 overrides).
# EVAL lane:
#   once   Eikos-4B same-protocol JevBench (scripts/p2/competitor_bench.sh eikos4b <gpu> 8110 ...) if cloud/setup_eikos.sh installed it:
#          GPU 7 in parallel with the smoke on 8-GPU pods, else GPU 6 in the eval lane before the first eval.
#   per run (best and last):
#          fit_temp  held-out-domain calibration partition of the run's manifest, TEACHER rows only (source p2_teacher, no prog-*
#                    batches, no Eikos, no images), hard rows (p2b-hard*, p2c-judge*) weighted 3x by duplication; if the fitted
#                    T < 1 the shipped file uses T = 1.0 (docs/eikos-analysis.md §7 #6)
#          panels    best: decision-p2c test, decision-p2b test, decision-p2 test, decision-v2-reasoning-dev dev, state + pairs probes,
#                    decision-p2c-judge-dev dev, decision-v1.1-irrelevance-test; last: decision-p2c test + judge dev
#          JevBench  raw + cal only (no rotations): best on GPU 4 :8765, last on GPU 5 :8766
#          ImajevBench best on GPU 6
#          gates     cloud/p2c_gates_reference.json -> $O/<run>/gates.json + PASS/FAIL lines; any FAIL -> "<run> NOT_SHIPPABLE"
#                    (a) ImajevBench acc >= ref - 1.0 pt AND visual >= ref - 2 items AND joint >= ref - 2 items
#                    (b) state probe >= ref - 2 pts, pairs probe >= ref - 2 pts
#                    (c) decision-p2b test: correct-unknown rate on unknown-gold rows >= ref; false abstention on answerable <= ref + 2 pts
#                    (d) decision-v1.1-irrelevance-test accuracy >= ref - 2 pts
# Ports: 8765/8766 (JevBench servers), 8110 + 8111 (Eikos). Torch master ports: 29500 (train lane), 29611-29613 (smoke).
# GPUs: 0-3 train lane; 4-6 smoke, then eval (panels sharded over 4-6, plus 7 once Eikos is done; JevBench best 4, last 5; ImajevBench 6);
#       7 Eikos-4B (8-GPU pods).
# Stop commands kill recorded process groups or exact per-run command-line patterns, never a broad `pkill -f`.
# Launch (from the pod): cd /workspace && setsid nohup bash cloud/pod_run_p2c_train.sh >> p2/train-out-p2c/run.log 2>&1 < /dev/null &
# Never run two instances on one pod. Stages carry DONE markers under p2/train-out-p2c (rerun = resume).
set -uo pipefail
cd  || exit 1; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P2=${P2:-p2}; O=$P2/train-out-p2c; mkdir -p $O; TRAIN_GPUS=0,1,2,3; NTRAIN=4; EVAL_GPUS=(4 5 6)
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); EIKOS_GPU=""; [ "$NGPU" -ge 8 ] && EIKOS_GPU=7   # GPU 7: Eikos first, then eval shards
rm -f $O/gpu7-free
RUNS=${RUNS:-"4b-delta 4b-fresh 9b-delta 2b-delta"}
MAN=data/manifests; TOOLS=$P2/imajev/scripts/p2; SERVER=${SERVER:-$P2/imajev/scripts/playground/server.py}
HOLDOUT=${HOLDOUT:-telecom,hospitality,nonprofit_grants}; GATES_REF=${GATES_REF:-cloud/p2c_gates_reference.json}
DEV2=decision-p2c-judge-dev
RECIPE="--soft-targets --soft-weight 1.0 --rationale-weight 0.3 --rationale-max-tokens 192 --permute-options"
ALL_LINEAR=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj
log(){ echo "$(date +%H:%M:%S) $*"; }; stage(){ log "stage $1"; }; skip(){ [ -f $O/$1.DONE ]; }
snap(){ d=hf/hub/models--${1//\//--}/snapshots/$2; [ -d "$d" ] && echo "$d" || echo ""; }

M4=$(snap Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a); M9=$(snap Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a)
M2=$(snap Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc)
declare -A SIZE MODEL INIT MANI EPOCHS LR WARMUP BUDGET EXTRA BATCH ACC DEVEVERY SAVEEVERY
cfg(){ # run size model init manifest epochs lr warmup budget batch accumulate dev-every save-every extra...
  local r=$1; SIZE[$r]=$2; MODEL[$r]=$3; INIT[$r]=$4; MANI[$r]=$5; EPOCHS[$r]=$6; LR[$r]=$7; WARMUP[$r]=$8; BUDGET[$r]=$9
  BATCH[$r]=${10}; ACC[$r]=${11}; DEVEVERY[$r]=${12}; SAVEEVERY[$r]=${13}; shift 13; EXTRA[$r]="$*"; }
cfg 4b-delta 4b "$M4" adapters/4b-p2b-best decision-p2c       2 2e-5 10 10000 40 2 10 10 --no-checkpointing
cfg 4b-fresh 4b "$M4" ""                              decision-p2c-fresh 1 1e-4 20 10000 ${FRESH_BATCH:-8} ${FRESH_ACCUMULATE:-1} 50 25 --rank 64 --alpha 128 --lora-targets $ALL_LINEAR
cfg 9b-delta 9b "$M9" adapters/9b-p2b-best decision-p2c       2 2e-5 10 7500  40 2 10 10
cfg 2b-delta 2b "$M2" adapters/2b-p2b-best decision-p2c       2 2e-5 10 12000 40 2 10 10 --no-checkpointing

if ! skip manifest-check; then stage manifest-check
  for m in decision-p2c decision-p2c-fresh $DEV2 decision-p2b decision-p2 decision-v2-reasoning-dev decision-v2-state-probe decision-v2-pairs-probe decision-v1.1-irrelevance-test; do
    [ -s $MAN/$m.jsonl ] || { log "MISSING manifest $MAN/$m.jsonl"; exit 1; }; done
  python - <<'PY' || { log MANIFEST_CHECK_FAILED; exit 1; }
import json, collections, os
for name in ("decision-p2c", "decision-p2c-fresh", "decision-p2c-judge-dev"):
    rows = [json.loads(l) for l in open(f"data/manifests/{name}.jsonl") if l.strip()]
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
    imgs = [i["image"] for r in rows for i in r.get("images", [])]
    missing = sum(not os.path.exists(os.path.join("", p)) for p in imgs[:2000])
    print(f"{name}: {len(rows)} rows {dict(collections.Counter(r['partition'] for r in rows))}; target_probs {sum('target_probs' in r for r in rows)}; "
          f"rationale {sum(bool(r.get('rationale')) for r in rows)}; image rows {sum(bool(r.get('images')) for r in rows)}; missing images in first 2000: {missing}")
    assert missing == 0, f"{name}: image files missing (did the bootstrap fetch the image data from the bucket?)"
PY
  touch $O/manifest-check.DONE; fi
# delta lanes: the trainer builds a fresh r16/alpha32 LoRA on the default targets and then loads the init adapter with PEFT's
# NON-strict set_peft_model_state_dict, so a structure mismatch would silently train from scratch. Refuse that here.
for r in $RUNS; do A=${INIT[$r]}; [ -n "$A" ] || continue
  python - "$A" <<'PY' || { log "INIT_ADAPTER_MISMATCH $r ($A)"; exit 1; }
import json, sys
from pathlib import Path
from torch_decision import TARGETS
a = Path(sys.argv[1]); c = json.loads((a / "adapter_config.json").read_text())
want = r'.*language_model.*\.(' + '|'.join(TARGETS) + ')'
missing = [f for f in ("adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json") if not (a / f).exists()]
assert c["r"] == 16 and c["lora_alpha"] == 32 and c["target_modules"] == want and not missing, (c["r"], c["lora_alpha"], c["target_modules"], missing)
print(f"  init adapter {a.name}: r16/alpha32, default targets, readout present")
PY
done

# ------------------------------------------------------------------------------------------------------------------------ train lane
cleanup_train(){ # only this run's trainer processes (pattern = its own output dir), then wait for GPUs 0-3 to be free of trainers
  local r=$1 p
  pkill -f "train_decision_lora_torch.py --version [^ ]+ --output $O/$r/train " 2>/dev/null; sleep 5
  for i in $(seq 1 30); do local left=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $TRAIN_GPUS 2>/dev/null); do
      tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "train_decision_lora_torch" && left="$left $p"; done
    [ -z "$left" ] && return 0; [ $i -eq 15 ] && kill -9 $left 2>/dev/null; sleep 3; done; }
train_lane(){ local r
  for r in $RUNS; do
    local M=${MODEL[$r]} A=${INIT[$r]}; mkdir -p $O/$r
    [ -n "$M" ] && [ -d "$M" ] || { log "MISSING model for $r"; touch $O/train-$r.FAILED; continue; }
    [ -z "$A" ] || [ -d "$A" ] || { log "MISSING adapter $A"; touch $O/train-$r.FAILED; continue; }
    skip train-$r && continue
    stage "train-$r"; log "  $r: ${MANI[$r]}, init ${A:-none (fresh LoRA)}, epochs ${EPOCHS[$r]}, lr ${LR[$r]}, warm-up ${WARMUP[$r]}, budget ${BUDGET[$r]}, batch ${BATCH[$r]}x${ACC[$r]}, extra: ${EXTRA[$r]:-none}"
    cleanup_train $r
    for attempt in 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m torch.distributed.run --nproc_per_node=$NTRAIN --master_port 29500 scripts/train_decision_lora_torch.py \
        --version ${MANI[$r]} --output $O/$r/train --model $M ${A:+--init-adapter $A} --epochs ${EPOCHS[$r]} --lr ${LR[$r]} --warmup ${WARMUP[$r]} \
        --save-every ${SAVEEVERY[$r]} --dev-cases 600 --dev-every ${DEVEVERY[$r]} --dev2-version $DEV2 --dev2-cases 400 --select mean_accuracy \
        --pad-multiple 64 --token-budget ${BUDGET[$r]} --max-length 4096 --batch-size ${BATCH[$r]} --accumulate ${ACC[$r]} --workers 16 \
        $RECIPE ${EXTRA[$r]} >> $O/$r/train.log 2>&1 && { touch $O/train-$r.DONE; break; }
      log "  train $r attempt $attempt failed (resume from last checkpoint)"; tail -n 3 $O/$r/train.log | cut -c1-200; cleanup_train $r; done
    skip train-$r || { log "TRAIN_FAILED $r"; touch $O/train-$r.FAILED; continue; }
    cp $O/$r/train/log.jsonl $O/$r/dev-curve.jsonl 2>/dev/null; log "done train-$r"
  done; }

# ------------------------------------------------------------------------------------------------------------------------- eval lane
eval_gpus(){ echo "${EVAL_GPUS[@]}"; [ -n "$EIKOS_GPU" ] && [ -f $O/gpu7-free ] && echo $EIKOS_GPU; }
evalp(){ # run ckpt tag version [partition]; shards over the eval GPUs; a failed shard leaves NO predictions.jsonl (rerun redoes it)
  local r=$1 c=$2 tag=$3 v=$4 part=${5:-}; local out=$O/$r/eval-$tag-$v${part:+-$part}; [ -s $out/predictions.jsonl ] && return 0
  local gpus=($(eval_gpus)) k=0 g p bad=0 pids=(); local n=${#gpus[@]}; mkdir -p $out; rm -f $out/predictions-*.jsonl
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python scripts/evaluate_decision_model_torch.py --version $v --output $out --model ${MODEL[$r]} --adapter $c ${part:+--partition $part} \
      --shard $k/$n --token-budget 16000 > $out-$k.log 2>&1 & pids+=($!); k=$((k+1)); done
  for p in "${pids[@]}"; do wait $p || bad=1; done
  [ "$(ls $out/predictions-*.jsonl 2>/dev/null | wc -l)" -eq $n ] || bad=1
  [ $bad = 0 ] || { log "EVAL_FAILED $r $tag $v${part:+ $part} (see $out-*.log)"; tail -n 2 $out-*.log | cut -c1-200; return 1; }
  cat $out/predictions-*.jsonl > $out/predictions.jsonl
  python - "$out/predictions.jsonl" "$r $tag $v${part:+ $part}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"  {sys.argv[2]}: {len(rows)} predictions, acc {sum(r['correct'] for r in rows)/max(1,len(rows)):.3f}")
PY
}

fit_temp(){ # run ckpt tag -> $O/<run>/calibration-p2c-<run>-<tag>.json (teacher rows, hard x3, clamp T >= 1)
  local r=$1 c=$2 tag=$3; local out=$O/$r/calibration-p2c-$r-$tag.json; [ -f $out ] && return
  evalp $r $c $tag ${MANI[$r]} calibration
  local pred=$O/$r/eval-$tag-${MANI[$r]}-calibration/predictions.jsonl
  python - $pred $MAN/${MANI[$r]}.jsonl $pred.teacher-hard3.jsonl <<'PY' || { log "FIT_TEMP_FILTER_FAILED $r $tag"; return; }
import json, sys
meta = {}
for l in open(sys.argv[2]):
    if l.strip():
        r = json.loads(l); meta[r["id"]] = (r.get("source"), r.get("batch") or "", bool(r.get("images")))
kept = hard = 0
with open(sys.argv[3], "w") as w:
    for l in open(sys.argv[1]):
        if not l.strip(): continue
        p = json.loads(l); m = meta.get(p["id"]) or meta.get(p["id"].rsplit(":", 1)[0])
        if not m or m[0] != "p2_teacher" or m[1].startswith("prog") or m[2]: continue
        n = 3 if m[1].startswith(("p2b-hard", "p2c-judge")) else 1; kept += 1; hard += n == 3
        for _ in range(n): w.write(l if l.endswith("\n") else l + "\n")
print(f"  calibration rows: {kept} teacher rows kept ({hard} hard rows weighted x3)")
PY
  python $TOOLS/fit_p2_temperature.py --predictions $pred.teacher-hard3.jsonl --manifest $MAN/${MANI[$r]}.jsonl --holdout-domains $HOLDOUT \
    --version p2c-$r-$tag --out $out || { log "FIT_TEMP_FAILED $r $tag"; return; }
  python - $out <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); t = float(p["fit"]["temperature"])
if t < 1.0:
    p["temperatures"] = {k: 1.0 for k in p["temperatures"]}; p["fit"]["fitted_temperature"] = t; p["fit"]["shipped_temperature"] = 1.0
    p["fit"]["rule"] += "; fitted T < 1 -> shipped T = 1.0 (phase-2c plan)"
    json.dump(p, open(sys.argv[1], "w"), indent=1)
print(f"  temperature: fitted {t:.3f} -> shipped {max(1.0, t):.3f}")
PY
  [ "$tag" = "best" ] && cp $out $O/$r/calibration-p2c-$r.json; }

ibench(){ # run ckpt tag : ImajevBench v2.0-lite test split, direct option scoring (torch)
  local r=$1 c=$2 tag=$3; local out=$O/$r/imajevbench-$tag; [ -f $out/completion.json ] && return; rm -rf $out
  local name=qwen4b; [ "${SIZE[$r]}" = "9b" ] && name=imajev9b; [ "${SIZE[$r]}" = "2b" ] && name=imajev2b-v21
  CUDA_VISIBLE_DEVICES=${IB_GPU:-${EVAL_GPUS[2]}} python scripts/imajev_bench/run_local_v2.py --records bench/records/records-eval.jsonl --root bench --split test \
    --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --model $name --adapter $c \
    --base-path ${MODEL[$r]} --output $out > $out.log 2>&1 || { tail -n 3 $out.log; log "FAILED imajevbench $r $tag"; return; }
  python -m imajev_bench score --records bench/records/records-eval.jsonl --root bench --split test --allow-draft --predictions $out/predictions.jsonl --output $out/score > $out-score.log 2>&1
  python - $out/score "$r $tag" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); tr = {t: (sum(f["correct"] for f in v["families"].values()), sum(f["total"] for f in v["families"].values())) for t, v in s["capability"]["tracks"].items()}
tot = sum(v[0] for v in tr.values()); n = sum(v[1] for v in tr.values()); print(f"  {sys.argv[2]} imajevbench: {tot}/{n} = {tot/n:.1%} tracks {tr}")
PY
}

jevbench(){ # run ckpt tag variant(raw|cal) gpu port
  local r=$1 c=$2 tag=$3 variant=$4 gpu=$5 port=$6; local R=$O/$r/jevbench-$tag-$variant; [ -f $R/hard/summary.json ] && return; mkdir -p $R
  local extra=(); if [ "$variant" = cal ]; then [ -f $O/$r/calibration-p2c-$r-$tag.json ] || { log "  no calibration for $r $tag: skipping cal"; return; }
    extra=(--calibration $O/$r/calibration-p2c-$r-$tag.json); fi
  printf '{"repo": "local", "revision": "local", "path": "%s"}\n' "${MODEL[$r]}" > $R/bundle.json
  CUDA_VISIBLE_DEVICES=$gpu setsid python $SERVER --backend torch --model-bundle $R/bundle.json --adapter $c --host 127.0.0.1 --port $port "${extra[@]}" > $R/server.log 2>&1 < /dev/null &
  local pg=$!
  for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break; kill -0 $pg 2>/dev/null || break; sleep 3; done
  ( cd $P2/external/jevbench && for split in hard original easy; do mkdir -p $R/$split
      TYPESAFE_API_KEY=local python -m jevbench.cli run --tasks datasets/public/$split.jsonl --adapter typesafe --endpoint http://127.0.0.1:$port --model imajev-$r-p2c-$tag-$variant \
        --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1 --results $R/$split/results.jsonl --raw-dir $R/$split/raw --manifest $R/$split/manifest.json > $R/$split/run.log 2>&1
      python -m jevbench.cli summarize --tasks datasets/public/$split.jsonl --results $R/$split/results.jsonl --public-export $R/$split/summary.json > /dev/null 2>&1
      python -c "import json;d=json.load(open('$R/$split/summary.json'));print('  $r $tag $variant jevbench $split: acc',round(d['accuracy'],3),'ece',round(d['ece']['ece'],3))" 2>/dev/null \
        || echo "  $r $tag $variant jevbench $split: FAILED (see $R/$split/run.log)"; done )
  kill -TERM -- -$pg 2>/dev/null; sleep 3; kill -KILL -- -$pg 2>/dev/null; }

gates(){ # run -> $O/<run>/gates.json, PASS/FAIL lines, NOT_SHIPPABLE marker
  local r=$1
  python - $r ${SIZE[$r]} $O/$r $GATES_REF <<'PY'
import json, sys, os
run, size, d, refp = sys.argv[1:5]
ref = json.load(open(refp)); tol = ref["tolerances"]; R = ref["sizes"][size]
UNK = (None, "unknown", "__unknown__", "null")
def preds(name):
    p = os.path.join(d, name, "predictions.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
def acc(rows): return 100.0 * sum(r["correct"] for r in rows) / len(rows) if rows else None
checks = []
def check(name, value, op, bound, detail=""):
    if value is None or bound is None:
        checks.append({"gate": name, "value": value, "bound": bound, "pass": False, "note": "missing measurement" if value is None else "missing reference"}); return
    v, b = round(value, 2), round(bound, 2)  # references are stored to 2 decimals
    ok = v >= b - 1e-9 if op == ">=" else v <= b + 1e-9
    checks.append({"gate": name, "value": round(value, 2), "op": op, "bound": round(bound, 2), "pass": ok, "note": detail})
# (a) ImajevBench
sp = os.path.join(d, "imajevbench-best", "score")
if os.path.exists(sp):
    s = json.load(open(sp)); tr = {t: (sum(f["correct"] for f in v["families"].values()), sum(f["total"] for f in v["families"].values())) for t, v in s["capability"]["tracks"].items()}
    tot = sum(v[0] for v in tr.values()); n = sum(v[1] for v in tr.values())
    check("a.imajevbench_acc", 100.0 * tot / n, ">=", R["imajevbench_acc"] - tol["imajevbench_acc_points"], f"{tot}/{n}")
    for t in ("visual", "joint"):
        check(f"a.{t}_items", tr.get(t, (None,))[0], ">=", R[t][0] - tol["track_items"], f"ref {R[t][0]}/{R[t][1]}")
else:
    for g in ("a.imajevbench_acc", "a.visual_items", "a.joint_items"): check(g, None, ">=", 0)
# (b) probes
check("b.state_probe", acc(preds("eval-best-decision-v2-state-probe") or []), ">=", R["state_probe"] - tol["probe_points"])
check("b.pairs_probe", acc(preds("eval-best-decision-v2-pairs-probe") or []), ">=", R["pairs_probe"] - tol["probe_points"])
# (c) unknown behaviour on decision-p2b test
rows = preds("eval-best-decision-p2b-test")
if rows:
    unk = [r for r in rows if r.get("target") in UNK]; ans = [r for r in rows if r.get("target") not in UNK]
    cu = 100.0 * sum(r.get("prediction") in UNK for r in unk) / len(unk) if unk else None
    fa = 100.0 * sum(r.get("prediction") in UNK for r in ans) / len(ans) if ans else None
    check("c.unknown_correct_rate", cu, ">=", R.get("unknown_correct_rate"), f"{len(unk)} unknown-gold rows")
    fr = R.get("false_abstention_rate")
    check("c.false_abstention_rate", fa, "<=", None if fr is None else fr + tol["false_abstention_points"], f"{len(ans)} answerable rows")
else:
    check("c.unknown_correct_rate", None, ">=", 0); check("c.false_abstention_rate", None, "<=", 0)
# (d) irrelevance
check("d.irrelevance", acc(preds("eval-best-decision-v1.1-irrelevance-test") or []), ">=", R["irrelevance"] - tol["irrelevance_points"])
ok = all(c["pass"] for c in checks)
json.dump({"run": run, "size": size, "shippable": ok, "checks": checks, "reference": R, "tolerances": tol}, open(os.path.join(d, "gates.json"), "w"), indent=1)
for c in checks:
    print(f"  gate {run} {c['gate']}: {'PASS' if c['pass'] else 'FAIL'} value {c['value']} {c.get('op', '')} {c['bound']} {c.get('note', '')}")
print(f"  {run} {'SHIPPABLE (all gates pass)' if ok else 'NOT_SHIPPABLE'}")
sys.exit(0 if ok else 3)
PY
  local rc=$?; if [ $rc -eq 0 ]; then rm -f $O/$r/NOT_SHIPPABLE; else touch $O/$r/NOT_SHIPPABLE; log "$r NOT_SHIPPABLE (gates; see $O/$r/gates.json)"; fi; }

smoke(){ skip smoke && return 0; stage smoke
  local r=4b-delta; local M=${MODEL[$r]} A=${INIT[$r]}; rm -rf $O/smoke; mkdir -p $O/smoke   # fresh dirs: never resume an old smoke
  [ -n "$M" ] && [ -d "$A" ] || { log "SMOKE_MISMATCH: 4B model or adapter missing ($M, $A)"; return 1; }
  local common="--version decision-p2c --model $M --init-adapter $A --max-steps 5 --epochs 1 --lr 2e-5 --warmup 10 --dev-cases 200 --dev-every 1000 --save-every 0 \
    --select dev_loss --pad-multiple 64 --token-budget 10000 --max-length 4096 --batch-size 8 --accumulate 1 --workers 4 --no-checkpointing"
  local k=0 spec name flags p pids=()
  for spec in "off|" "on|$RECIPE" "on-sw0|${RECIPE/--soft-weight 1.0/--soft-weight 0}"; do name=${spec%%|*}; flags=${spec#*|}
    CUDA_VISIBLE_DEVICES=${EVAL_GPUS[$k]} python -m torch.distributed.run --nproc_per_node=1 --master_port $((29611+k)) scripts/train_decision_lora_torch.py \
      --output $O/smoke/$name $common $flags > $O/smoke/$name.log 2>&1 & pids+=($!); k=$((k+1)); done
  for p in "${pids[@]}"; do wait $p; done
  python - $O/smoke <<'PY' | tee $O/smoke/summary.txt
import json, os, sys
d = sys.argv[1]; res = {}
for name in ("off", "on", "on-sw0"):
    p = os.path.join(d, name, "log.jsonl")
    lines = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    s0 = next((x for x in lines if x.get("step") == 0 and "dev_loss" in x), None)
    last = [x for x in lines if "loss" in x or "train_loss" in x]
    res[name] = s0
    print(f"  smoke {name:7s}: step-0 dev_loss {s0 and round(s0['dev_loss'], 5)} dev_accuracy {s0 and round(s0['dev_accuracy'], 4)}; last log line {json.dumps(lines[-1])[:220] if lines else 'NONE (see log)'}")
def rationale_steps(name):
    p = os.path.join(d, name, "log.jsonl")
    lines = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    steps = [x for x in lines if "of" in x]
    return len(steps), sum("rationale_loss" in x for x in steps)
st = {n: rationale_steps(n) for n in ("off", "on", "on-sw0")}
print(f"  smoke optimizer steps logged / with rationale_loss: {st}")
ok = (all(res.values()) and len({round(v["dev_accuracy"], 6) for v in res.values()}) == 1 and abs(res["off"]["dev_loss"] - res["on-sw0"]["dev_loss"]) < 1e-3
      and all(n == 5 for n, _ in st.values()) and st["off"][1] == 0 and st["on"][1] == 5 and st["on-sw0"][1] == 5)
if all(res.values()):
    print(f"  smoke dev-loss difference ON - OFF = {res['on']['dev_loss'] - res['off']['dev_loss']:+.5f} (soft CE vs hard CE on dev rows with target_probs); OFF vs ON+sw0 = {res['on-sw0']['dev_loss'] - res['off']['dev_loss']:+.6f}")
print("SMOKE_OK" if ok else "SMOKE_MISMATCH: step-0 dev metrics differ where they must match, a run did not log 5 steps, or the rationale loss is missing/present where it must not be -- inspect before training")
sys.exit(0 if ok else 3)
PY
  local rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && { touch $O/smoke.DONE; log "done smoke"; return 0; }
  log "SMOKE_MISMATCH (see $O/smoke/summary.txt and $O/smoke/*.log)"; return 1; }

eikos_bench(){ # gpu
  local gpu=$1
  if ! skip eikos-bench; then
    if [ -f eikos/.installed ]; then stage eikos-bench
      bash $P2/imajev/scripts/p2/competitor_bench.sh eikos4b $gpu 8110 bash cloud/setup_eikos.sh serve 8110 8111 > $O/eikos4b.log 2>&1
      pkill -f "vllm serve [^ ]+ --host 127.0.0.1 --port 8111 " 2>/dev/null; sleep 3
      grep -E "jevbench|FAILED" $O/eikos4b.log; touch $O/eikos-bench.DONE; log "done eikos-bench (GPU $gpu)"
    else log "Eikos-4B not installed (cloud/setup_eikos.sh install): skipping the same-protocol run"; fi
  fi
  [ "$gpu" = "$EIKOS_GPU" ] && touch $O/gpu7-free; return 0; }

eval_run(){ local r=$1 c
  skip eval-$r && return; stage "eval-$r"
  for tag in best last; do c=$O/$r/train/$tag; [ -d $c ] || continue
    fit_temp $r $c $tag
    evalp $r $c $tag decision-p2c test; evalp $r $c $tag $DEV2 dev
    if [ "$tag" = best ]; then
      evalp $r $c $tag decision-p2b test; evalp $r $c $tag decision-p2 test; evalp $r $c $tag decision-v2-reasoning-dev dev
      evalp $r $c $tag decision-v2-state-probe; evalp $r $c $tag decision-v2-pairs-probe; evalp $r $c $tag decision-v1.1-irrelevance-test; fi; done
  ( c=$O/$r/train/best; [ -d $c ] && for v in raw cal; do jevbench $r $c best $v ${EVAL_GPUS[0]} 8765; done ) &
  ( c=$O/$r/train/last; [ -d $c ] && for v in raw cal; do jevbench $r $c last $v ${EVAL_GPUS[1]} 8766; done ) &
  ( c=$O/$r/train/best; [ -d $c ] && IB_GPU=${EVAL_GPUS[2]} ibench $r $c best ) &
  wait; gates $r; touch $O/eval-$r.DONE; log "done eval-$r"; }
eval_lane(){ [ -z "$EIKOS_GPU" ] && eikos_bench ${EVAL_GPUS[2]}; local r
  for r in $RUNS; do until [ -f $O/train-$r.DONE ] || [ -f $O/train-$r.FAILED ]; do sleep 20; done
    [ -f $O/train-$r.DONE ] && eval_run $r; done; }

EK=""; [ -n "$EIKOS_GPU" ] && { eikos_bench $EIKOS_GPU & EK=$!; }   # GPU 7, alongside the smoke and the first training run
if ! smoke; then
  if [ "${SMOKE_IGNORE:-0}" = 1 ]; then log "SMOKE_IGNORE=1: continuing despite the smoke result"
  else log "SMOKE_MISMATCH: aborting before the train lane (fix, or rerun with SMOKE_IGNORE=1)"; [ -n "$EK" ] && wait $EK; exit 1; fi
fi
eval_lane & EL=$!
train_lane
wait $EL; [ -n "$EK" ] && wait $EK

stage summary
python - $O "$RUNS" <<'PY' | tee $O/summary.txt
import json, os, sys
O, runs = sys.argv[1], sys.argv[2].split()
def jb(r, tag, v, split="hard"):
    p = f"{O}/{r}/jevbench-{tag}-{v}/{split}/summary.json"
    if not os.path.exists(p): return "-"
    d = json.load(open(p)); return f"{100*d['accuracy']:.1f}/{d['ece']['ece']:.3f}"
def acc(r, name):
    p = f"{O}/{r}/{name}/predictions.jsonl"
    if not os.path.exists(p): return "-"
    rows = [json.loads(l) for l in open(p) if l.strip()]; return f"{100*sum(x['correct'] for x in rows)/max(1,len(rows)):.1f}"
print("run        | p2c test | judge dev | JB hard raw best (acc/ECE) | cal best | raw last | cal last | T best | gates")
for r in runs:
    cal = f"{O}/{r}/calibration-p2c-{r}-best.json"; t = json.load(open(cal))["fit"] if os.path.exists(cal) else {}
    g = f"{O}/{r}/gates.json"; gate = ("PASS" if json.load(open(g))["shippable"] else "NOT_SHIPPABLE") if os.path.exists(g) else "-"
    print(f"{r:10s} | {acc(r, 'eval-best-decision-p2c-test'):8s} | {acc(r, 'eval-best-decision-p2c-judge-dev-dev'):9s} | {jb(r, 'best', 'raw'):26s} | {jb(r, 'best', 'cal'):8s} | "
          f"{jb(r, 'last', 'raw'):8s} | {jb(r, 'last', 'cal'):8s} | {t.get('fitted_temperature', t.get('temperature', '-'))} | {gate}")
e = f"{os.path.dirname(O)}/out/eikos4b/hard/summary.json"
if os.path.exists(e): d = json.load(open(e)); print(f"Eikos-4B same protocol: hard {100*d['accuracy']:.1f} / ECE {d['ece']['ece']:.3f}")
PY
stage pack
tar czf $P2/p2c-train-results.tgz --warning=no-file-changed --exclude='*/trainer.pt' --exclude='smoke/*/best' --exclude='smoke/*/last' -C $P2 train-out-p2c $( [ -d $P2/out/eikos4b ] && echo out/eikos4b ); rc=$?
[ $rc -le 1 ] && ls -la $P2/p2c-train-results.tgz || log PACK_FAILED
echo ALL_DONE
