#!/usr/bin/env bash
# Phase-2c teacher generation + soft-target relabelling on the 8xH100 pod (7 GPUs also work: GPU 7's replica is then skipped) (docs/phase-2c-plan.md, docs/phase-2c-runbook.md).
# Stages (DONE markers + logs under p2/out/p2c; the main log has one "HH:MM:SS stage <name>" line per stage start and
# "HH:MM:SS done <name>" per finish, for the dashboard):
#   write-judge   Qwen3.6-27B WRITER only (TP=4, GPUs 0-3, :8000, thinking on, max_tokens 16000): 1,500 docs over
#                 judge_pairwise,judge_rubric_score, tag p2c-judge -> writer-p2c-judge.jsonl
#   write-unknown then, on the same writer replica, thinking OFF, max_tokens 8000: 1,500 docs over insufficient_evidence,ambiguity,
#                 contradiction, tag p2c-unknown -> writer-p2c-unknown.jsonl (the mixture is only ~5-8% unknown-gold; target >= 15%)
#   relabel       Qwen3.6-35B-A3B (thinking) --mode distribution over ALL phase-2 + 2b questions (writer.jsonl + writer-p2b.jsonl
#                 = writer-p2c-relabel.jsonl): phase A on GPUs 4-5 (TP=2 :8040) + GPU 6 (TP=1 :8060) + GPU 7 (TP=1 :8090, 8-GPU pods),
#                 sharded workers with --done-from; finished in phase B -> answers-p2c-relabel-qwen35.jsonl
#   programmatic  CPU: gen_programmatic.py --docs 3000 --seed 7 -> writer-prog.jsonl (runs during phase A; the phase-A 35B workers
#                 start on these questions if they finish the relabel before the writer is done)
#   answer-new    after the writer: the new judge + unknown + programmatic questions (writer-p2c-new.jsonl) answered by
#                 (a) 35B-A3B --mode distribution (soft targets + rationale; shards 0,1 on TP=4 :8050 GPUs 0-3, 2 on TP=2 :8040,
#                     3 on TP=1 :8090 GPU 7 when present)
#                 (b) gpt-oss-20b default mode (independent-family agreement check; TP=1 GPU 6 :8020)
#                 The relabel remainder runs on the same two 35B servers (3 shards, --done-from the phase-A files).
#                 The 27B does NOT answer (owner decision 2026-09-24: writer only).
#   assemble      writer batches p2c-judge + p2c-unknown: both answer files (2-answerer unanimity: 35B argmax and gpt-oss),
#                 --unknown-rule relaxed --soft-from qwen35 (unknown-gold rows keep a hard target when the 35B disagrees: the assembler's
#                 default keep-hard policy), normal near-duplicate lint (threshold 5 vs phase-2/2b writer docs + human states); prog batches: --dedupe-threshold 100000 --min-unknown-share 0
#                 --soft-from qwen35, appended -> records-p2c.jsonl. JevBench 8-gram contamination lint on for both.
#   manifest      build_p2c_manifest.py (image replay pre-sampled in data/decision-p2c/image-replay) -> data/manifests/decision-p2c.jsonl, decision-p2c-fresh.jsonl, decision-p2c-judge-dev.jsonl
#   pack          p2/decision-p2c-teacher.tgz, then ALL_DONE
# GPU map:
#   phase A  GPUs 0-3  27B writer TP=4 :8000 (judge, then unknown) | GPUs 4-5 35B TP=2 :8040 relabel 0/n | GPU 6 35B TP=1 :8060 relabel 1/n
#            | GPU 7 35B TP=1 :8090 relabel 2/3                                             (n = 3 with GPU 7, else 2)
#   phase B  GPUs 0-3  35B TP=4 :8050 (new + relabel shards 0,1/m) | GPUs 4-5 35B TP=2 :8040 (shard 2/m) | GPU 6 gpt-oss :8020
#            | GPU 7 35B TP=1 :8090 (shard 3/4)                                              (m = 4 with GPU 7, else 3)
#   assemble / manifest / pack: CPU only (all servers stopped)
# Ports: 8000/8020/8040/8050/8060/8090 only (8001/8081 are RunPod's). stop_port kills a server by its recorded process group (setsid) and
# falls back to the exact vLLM command-line pattern for that port; never a broad `pkill -f` that an ssh command line could match.
# Launch (from the pod, never inline over ssh):
#   mkdir -p p2/out/p2c && cd /workspace && setsid nohup bash cloud/pod_run_p2c_gen.sh \
#     >> p2/out/p2c/run.log 2>&1 < /dev/null &
# Rerun = resume: finished stages are skipped, answer files resume by (doc, question), the writer resumes by doc id.
set -uo pipefail
P2=${P2:-p2}; cd $P2/imajev || exit 1; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_API_KEY=local PATH=vllm-venv/bin:$PATH
O=$P2/out/p2c; D=data/decision-p2/teacher; MAN=data/manifests; mkdir -p $O $D $MAN
JUDGE_DOCS=${JUDGE_DOCS:-1500}; PROG_DOCS=${PROG_DOCS:-3000}; PROG_SEED=${PROG_SEED:-7}; WORKERS=${WORKERS:-192}; GPTOSS_MAXTOK=${GPTOSS_MAXTOK:-5000}
JUDGE_FAMILIES=judge_pairwise,judge_rubric_score; UNK_DOCS=${UNK_DOCS:-1500}; UNK_FAMILIES=insufficient_evidence,ambiguity,contradiction
HOLDOUT=${HOLDOUT:-telecom,hospitality,nonprofit_grants}; REPLAY=data/decision-p2c/image-replay
QW=hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
GO=hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
Q35=hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
REF=( $P2/jevbench-public/*.jsonl ); HUMAN=data/decision-p2/human/records.jsonl
MY_PORTS="8000 8020 8040 8050 8060 8090"
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); G7=0; [ "$NGPU" -ge 8 ] && G7=1   # GPU 7: one more 35B-A3B replica (TP=1)

log(){ echo "$(date +%H:%M:%S) $*"; }; stage(){ log "stage $1"; }; mark(){ touch $O/$1.DONE; log "done $1"; }; skip(){ [ -f $O/$1.DONE ]; }
existing(){ local f; for f in "$@"; do [ -s "$f" ] && echo "$f"; done; }   # non-empty files only (for --done-from)
nq(){ python -c "import json,sys;print(sum(len(json.loads(l)['output']['questions']) for l in open(sys.argv[1]) if l.strip()))" "$1"; }

serve(){ # gpus model-path served-name port tp max-num-seqs [extra vllm args]; records the process group in $O/vllm-<port>.pid
  local gpus=$1 path=$2 name=$3 port=$4 tp=$5 seqs=$6; shift 6
  if curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "\"$name\""; then log "  $name already serving on :$port"; return 0; fi
  [ -d "$path" ] || { log "MISSING model $path"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpus setsid vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model "$path" --served-model-name "$name" --port $port \
    --max-model-len 24576 --gpu-memory-utilization 0.92 --max-num-seqs $seqs --tensor-parallel-size $tp "$@" > $O/vllm-$name-$port.log 2>&1 < /dev/null &
  local pid=$!; echo $pid > $O/vllm-$port.pid
  for i in $(seq 1 240); do
    curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "\"$name\"" && { log "  $name up on :$port (GPUs $gpus, TP=$tp)"; return 0; }
    kill -0 $pid 2>/dev/null || break; sleep 5; done
  log "FAILED vllm $name :$port"; tail -n 20 $O/vllm-$name-$port.log; stop_port $port; return 1; }
stop_port(){ # only the server this script started on $1
  local port=$1 f=$O/vllm-$1.pid pg
  if [ -f $f ]; then pg=$(cat $f)
    kill -TERM -- -$pg 2>/dev/null; for i in $(seq 1 30); do kill -0 -- -$pg 2>/dev/null || break; sleep 2; done
    kill -KILL -- -$pg 2>/dev/null; rm -f $f; fi
  pkill -f "vllm.entrypoints.openai.api_server --model [^ ]+ --served-model-name [^ ]+ --port $port " 2>/dev/null
  for i in $(seq 1 30); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null || return 0; sleep 2; done; log "  WARNING :$port still answering"; }
stop_all(){ local p; for p in $MY_PORTS; do [ -f $O/vllm-$p.pid ] && stop_port $p; done; }
trap stop_all EXIT

ans(){ # answerer mode port shard workers writer out [done-from files...]; the python pid goes to $O/worker-<out>.pid
  local who=$1 mode=$2 port=$3 shard=$4 workers=$5 writer=$6 out=$7; shift 7
  local model maxtok name; name=$(basename $out .jsonl)
  case $who in qwen35) model=qwen3.6-35b-a3b; maxtok=8000;; gptoss) model=gpt-oss-20b; maxtok=$GPTOSS_MAXTOK;; *) log "unknown answerer $who"; return 1;; esac
  local df=() extra=() x; for x in $(existing "$@"); do [ "$x" != "$out" ] && df+=("$x"); done
  [ ${#df[@]} -gt 0 ] && extra=(--done-from "${df[@]}")
  python scripts/p2/gen_answer.py --answerer $who --mode $mode --writer $writer --out $out --workers $workers --max-tokens $maxtok --model $model \
    --base-url http://127.0.0.1:$port/v1 --shard $shard "${extra[@]}" >> $O/answer-$name.log 2>&1 &
  local pid=$!; echo $pid > $O/worker-$name.pid; wait $pid; local rc=$?; rm -f $O/worker-$name.pid; return $rc; }
kill_workers(){ local f; for f in $O/worker-*.pid; do [ -f $f ] && kill $(cat $f) 2>/dev/null; done; sleep 3; }

merge_answers(){ # out writer partial-files... : best row per (doc_id, qi), a parse-error row only when nothing better exists
  python - "$@" <<'PY'
import json, sys, os
out, writer, parts = sys.argv[1], sys.argv[2], [p for p in sys.argv[3:] if os.path.exists(p)]
best = {}
for p in parts:
    for l in open(p):
        try: r = json.loads(l)
        except ValueError: continue  # a line cut by a killed worker
        k = (r["doc_id"], r["qi"])
        if k not in best or (best[k].get("parse_error") is not None and r.get("parse_error") is None): best[k] = r
want = {(json.loads(l)["doc_id"], i) for l in open(writer) if l.strip() for i in range(len(json.loads(l)["output"]["questions"]))}
rows = [best[k] for k in sorted(best) if k in want]
tmp = out + ".tmp"
with open(tmp, "w") as w:
    for r in rows: w.write(json.dumps(r) + "\n")
os.replace(tmp, out)
errs = sum(r.get("parse_error") is not None for r in rows)
print(f"  merged {os.path.basename(out)}: {len(rows)} answers for {len(want)} questions ({len(want) - len(rows)} missing), parse errors {errs}")
PY
}
drop_errors(){ # remove parse-error rows (and lines cut by a killed worker) from partial files so the next round re-asks them
  python - "$@" <<'PY'
import json, sys, os
for f in sys.argv[1:]:
    if not os.path.exists(f): continue
    rows, bad = [], 0
    for l in open(f):
        if not l.strip(): continue
        try: rows.append(json.loads(l))
        except ValueError: bad += 1
    good = [r for r in rows if r.get("parse_error") is None]
    if len(good) == len(rows) and not bad: continue
    with open(f, "w") as w:
        for r in good: w.write(json.dumps(r) + "\n")
    print(f"  {os.path.basename(f)}: dropped {len(rows) - len(good)} parse-error rows and {bad} cut lines (re-asked next round)")
PY
}

for f in writer.jsonl records.jsonl records-p2b.jsonl; do [ -s $D/$f ] || { log "MISSING $D/$f"; exit 1; }; done
[ ${#REF[@]} -gt 0 ] && [ -s "${REF[0]}" ] || { log "MISSING JevBench public files under $P2/jevbench-public (contamination lint)"; exit 1; }
[ -s $D/writer-p2b.jsonl ] || cat $D/writer-p2b-hard.jsonl $D/writer-p2b-unknown.jsonl > $D/writer-p2b.jsonl
[ -s $D/writer-p2c-relabel.jsonl ] || cat $D/writer.jsonl $D/writer-p2b.jsonl > $D/writer-p2c-relabel.jsonl
log "relabel set: $(wc -l < $D/writer-p2c-relabel.jsonl) documents, $(nq $D/writer-p2c-relabel.jsonl) questions (phase 2 + 2b)"
RL_PARTS(){ ls $D/answers-p2c-relabel-qwen35-[ab]*.jsonl 2>/dev/null; }
NEW_PARTS(){ ls $D/answers-p2c-new-qwen35-[ab]*.jsonl 2>/dev/null; }

# ---- programmatic (CPU, background) ---------------------------------------------------------------------------------------------
PPROG=""
if ! skip programmatic; then stage programmatic
  ( rm -f $D/writer-prog.jsonl
    python scripts/p2/gen_programmatic.py --docs $PROG_DOCS --seed $PROG_SEED --out $D/writer-prog.jsonl > $O/programmatic.log 2>&1 \
      && { tail -n 2 $O/programmatic.log; mark programmatic; } || { tail -n 5 $O/programmatic.log; log PROGRAMMATIC_FAILED; } ) & PPROG=$!
fi

# ---- phase A: writer (0-3) || relabel (4-6) ------------------------------------------------------------------------------------
if ! skip write-judge || ! skip write-unknown; then
  stage relabel
  rm -f $O/phaseB.flag
  serve 0,1,2,3 $QW qwen3.6-27b 8000 4 384 --reasoning-parser qwen3 & PA=$!
  serve 4,5 $Q35 qwen3.6-35b-a3b 8040 2 128 --reasoning-parser qwen3 & PB=$!
  serve 6 $Q35 qwen3.6-35b-a3b 8060 1 64 --reasoning-parser qwen3 & PC=$!
  PD7=""; [ $G7 = 1 ] && { serve 7 $Q35 qwen3.6-35b-a3b 8090 1 64 --reasoning-parser qwen3 & PD7=$!; }
  wait $PA || { log WRITER_SERVE_FAILED; exit 1; }; wait $PB || log "  :8040 failed to start (relabel continues in phase B)"; wait $PC || log "  :8060 failed to start (relabel continues in phase B)"
  [ -n "$PD7" ] && { wait $PD7 || log "  :8090 failed to start (relabel continues in phase B)"; }
  writers(){ # judge (thinking on), then unknown (thinking off) on the same replica :8000
    if ! skip write-judge; then stage write-judge
      python scripts/p2/gen_write.py --docs $JUDGE_DOCS --start 0 --out $D/writer-p2c-judge.jsonl --workers $WORKERS --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 \
        --thinking --max-tokens 16000 --families $JUDGE_FAMILIES --tag p2c-judge > $O/write-judge.log 2>&1 && [ -s $D/writer-p2c-judge.jsonl ] \
        || { tail -n 3 $O/write-judge.log; log WRITE_JUDGE_FAILED; return 1; }
      tail -n 2 $O/write-judge.log; log "judge writer documents: $(wc -l < $D/writer-p2c-judge.jsonl)"; mark write-judge; fi
    if ! skip write-unknown; then stage write-unknown
      python scripts/p2/gen_write.py --docs $UNK_DOCS --start 0 --out $D/writer-p2c-unknown.jsonl --workers $WORKERS --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 \
        --max-tokens 8000 --families $UNK_FAMILIES --tag p2c-unknown > $O/write-unknown.log 2>&1 && [ -s $D/writer-p2c-unknown.jsonl ] \
        || { tail -n 3 $O/write-unknown.log; log WRITE_UNKNOWN_FAILED; return 1; }
      tail -n 2 $O/write-unknown.log; log "unknown writer documents: $(wc -l < $D/writer-p2c-unknown.jsonl)"; mark write-unknown; fi; }
  writers & WJ=$!
  chainA(){ # port shard workers idx: relabel shard, then (if time remains) programmatic questions
    local port=$1 shard=$2 workers=$3 idx=$4
    curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q qwen3.6-35b || return 0
    ans qwen35 distribution $port $shard $workers $D/writer-p2c-relabel.jsonl $D/answers-p2c-relabel-qwen35-a$idx.jsonl $(RL_PARTS)
    [ -f $O/phaseB.flag ] && return 0
    until [ -f $O/programmatic.DONE ] || [ -f $O/phaseB.flag ]; do sleep 20; done; [ -f $O/phaseB.flag ] && return 0
    log "  :$port finished its relabel shard; starting on programmatic questions"
    ans qwen35 distribution $port $shard $workers $D/writer-prog.jsonl $D/answers-p2c-new-qwen35-a$idx.jsonl $(NEW_PARTS); }
  if [ $G7 = 1 ]; then chainA 8040 0/3 128 0 & CA=$!; chainA 8060 1/3 64 1 & CB=$!; chainA 8090 2/3 64 2 & CC=$!
  else chainA 8040 0/2 128 0 & CA=$!; chainA 8060 1/2 64 1 & CB=$!; CC=""; fi
  wait $WJ; rc=$?
  [ $rc -eq 0 ] && skip write-judge && skip write-unknown || { log WRITE_FAILED; touch $O/phaseB.flag; kill_workers; drop_errors $(RL_PARTS) $(NEW_PARTS); exit 1; }
  touch $O/phaseB.flag; kill_workers; wait $CA $CB $CC 2>/dev/null; drop_errors $(RL_PARTS) $(NEW_PARTS)
  stop_port 8000; stop_port 8060   # 8040 (and 8090) stay up: phase B reuses them (serve() sees them already serving)
fi

# ---- phase B: new questions (35B distribution + gpt-oss) and the relabel remainder -----------------------------------------------
[ -n "$PPROG" ] && wait $PPROG; skip programmatic || { log "PROGRAMMATIC_FAILED (see $O/programmatic.log)"; exit 1; }
for f in writer-p2c-judge writer-p2c-unknown; do [ -s $D/$f.jsonl ] || { log "MISSING $D/$f.jsonl"; exit 1; }; done
cat $D/writer-p2c-judge.jsonl $D/writer-p2c-unknown.jsonl $D/writer-prog.jsonl > $D/writer-p2c-new.jsonl
log "new questions: $(nq $D/writer-p2c-judge.jsonl) judge + $(nq $D/writer-p2c-unknown.jsonl) unknown-family + $(nq $D/writer-prog.jsonl) programmatic"
if ! skip relabel || ! skip answer-new; then stage answer-new
  touch $O/phaseB.flag
  serve 0,1,2,3 $Q35 qwen3.6-35b-a3b 8050 4 256 --reasoning-parser qwen3 & PA=$!
  serve 4,5 $Q35 qwen3.6-35b-a3b 8040 2 128 --reasoning-parser qwen3 & PB=$!
  serve 6 $GO gpt-oss-20b 8020 1 128 --reasoning-parser openai_gptoss & PG=$!
  PD7=""; [ $G7 = 1 ] && { serve 7 $Q35 qwen3.6-35b-a3b 8090 1 64 --reasoning-parser qwen3 & PD7=$!; }
  wait $PA || { log SERVE_FAILED_8050; exit 1; }; wait $PB || { log SERVE_FAILED_8040; exit 1; }; wait $PG || { log SERVE_FAILED_8020; exit 1; }
  SHARDS="8050:0:128 8050:1:128 8040:2:128"; M=3   # port:shard:workers of the 35B distribution clients
  if [ -n "$PD7" ]; then if wait $PD7; then SHARDS="$SHARDS 8090:3:64"; M=4; else log "  :8090 failed to start (phase B continues on 3 shards)"; fi; fi
  round(){
    local pids=() spec port k w
    for spec in $SHARDS; do IFS=: read -r port k w <<< "$spec"
      skip relabel || { ans qwen35 distribution $port $k/$M $w $D/writer-p2c-relabel.jsonl $D/answers-p2c-relabel-qwen35-b$k.jsonl $(RL_PARTS) & pids+=($!); }
      skip answer-new || { ans qwen35 distribution $port $k/$M $w $D/writer-p2c-new.jsonl $D/answers-p2c-new-qwen35-b$k.jsonl $(NEW_PARTS) & pids+=($!); }
    done
    skip answer-new || { ans gptoss answer 8020 0/1 $WORKERS $D/writer-p2c-new.jsonl $D/answers-p2c-new-gptoss.jsonl & pids+=($!); }
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"; }
  round
  log "round 1 finished; re-asking parse errors once"
  drop_errors $(RL_PARTS) $(NEW_PARTS) $D/answers-p2c-new-gptoss.jsonl; round
  skip relabel || { merge_answers $D/answers-p2c-relabel-qwen35.jsonl $D/writer-p2c-relabel.jsonl $(RL_PARTS) && mark relabel; }
  if ! skip answer-new; then merge_answers $D/answers-p2c-new-qwen35.jsonl $D/writer-p2c-new.jsonl $(NEW_PARTS) \
      && merge_answers $D/answers-p2c-new-gptoss.jsonl.merged $D/writer-p2c-new.jsonl $D/answers-p2c-new-gptoss.jsonl \
      && mv $D/answers-p2c-new-gptoss.jsonl.merged $D/answers-p2c-new-gptoss.jsonl && mark answer-new; fi
  stop_port 8050; stop_port 8040; stop_port 8020; [ -f $O/vllm-8090.pid ] && stop_port 8090
  skip relabel && skip answer-new || { log ANSWER_STAGE_FAILED; exit 1; }
fi

log "unknown-mass audit (35B distribution on questions whose intended answer is a real option; plan threshold: mean <= 0.05)"
python - $D/writer-p2c-relabel.jsonl $D/answers-p2c-relabel-qwen35.jsonl $D/writer-p2c-new.jsonl $D/answers-p2c-new-qwen35.jsonl <<'PY' | tee $O/unknown-audit.txt
import json, sys, statistics
def load(writer, answers):
    intended = {}
    for l in open(writer):
        if not l.strip(): continue
        r = json.loads(l)
        for i, q in enumerate(r["output"]["questions"]): intended[(r["doc_id"], i)] = (r.get("batch") or "", q.get("intended"))
    rows = []
    for l in open(answers):
        try: rows.append(json.loads(l))
        except ValueError: pass
    return intended, rows
for name, (w, a) in {"relabel": sys.argv[1:3], "new": sys.argv[3:5]}.items():
    intended, rows = load(w, a)
    by = {}
    for r in rows:
        if r.get("parse_error") is not None or not isinstance(r.get("probs"), dict): continue
        batch, gold = intended.get((r["doc_id"], r["qi"]), ("?", None))
        if gold is None: continue
        by.setdefault(batch or "p2", []).append((float(r["probs"].get("unknown", 0.0)), str(r.get("value")).lower() == str(gold).lower()))
    for key, v in sorted(by.items()):
        m = [x for x, _ in v]; q = sorted(m)
        print(f"  {name:8s} {key:18s} n={len(v):6d} unknown mass mean {statistics.mean(m):.3f} p90 {q[int(.9*(len(q)-1))]:.3f} "
              f">0.05 {sum(x > .05 for x in m)/len(m):.1%} >0.20 {sum(x > .2 for x in m)/len(m):.1%} | argmax==intended {sum(ok for _, ok in v)/len(v):.1%}")
PY

if ! skip assemble; then stage assemble
  python scripts/p2/assemble_p2.py --writer $D/writer-p2c-new.jsonl --answers $D/answers-p2c-new-qwen35.jsonl $D/answers-p2c-new-gptoss.jsonl --soft-from qwen35 \
     --out $D/records-p2c-writer.jsonl --unknown-rule relaxed --batch-filter 'p2c-*' \
     --dedupe-against $D/writer.jsonl $D/writer-p2b.jsonl $HUMAN --dedupe-threshold 5 --reference "${REF[@]}" > $O/assemble-writer.log 2>&1 \
     || { tail -n 8 $O/assemble-writer.log; log ASSEMBLE_WRITER_FAILED; exit 1; }
  tail -n 6 $O/assemble-writer.log; [ -f $D/assembly-report.json ] && cp $D/assembly-report.json $O/assembly-report-writer.json
  python scripts/p2/assemble_p2.py --writer $D/writer-p2c-new.jsonl --answers $D/answers-p2c-new-qwen35.jsonl $D/answers-p2c-new-gptoss.jsonl --soft-from qwen35 \
     --out $D/records-p2c.jsonl --append-to $D/records-p2c-writer.jsonl --batch-filter 'prog-*' --dedupe-threshold 100000 --min-unknown-share 0 \
     --reference "${REF[@]}" > $O/assemble-prog.log 2>&1 || { tail -n 8 $O/assemble-prog.log; log ASSEMBLE_PROG_FAILED; exit 1; }
  tail -n 6 $O/assemble-prog.log; [ -f $D/assembly-report.json ] && cp $D/assembly-report.json $O/assembly-report-prog.json
  python - $D/records-p2c.jsonl <<'PY' | tee $O/assembly-counts.txt || { log ASSEMBLY_CHECK_FAILED; exit 1; }
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate ids in records-p2c"
unk = lambda r: r.get("target") in (None, "unknown", "__unknown__")
print(f"records-p2c: {len(rows)} rows; with target_probs {sum('target_probs' in r for r in rows)}; with rationale {sum(bool(r.get('rationale')) for r in rows)}; unknown-gold {sum(map(unk, rows))}")
for k, n in sorted(collections.Counter(r.get("batch") for r in rows).items()): print(f"  batch {k}: {n}")
for k, n in sorted(collections.Counter(r.get("family") for r in rows).items()): print(f"  family {k}: {n}")
print("  partitions", dict(collections.Counter(r.get("partition") for r in rows)))
PY
  mark assemble; fi

if ! skip manifest; then stage manifest
  python scripts/p2/build_p2c_manifest.py --relabel $D/answers-p2c-relabel-qwen35.jsonl --p2b-records $D/records-p2b.jsonl --p2c-records $D/records-p2c.jsonl \
     --eikos-records data/external/eikos-decisions/records.jsonl --image-replay-fresh $REPLAY/replay-fresh.jsonl \
     --image-replay-delta $REPLAY/replay-delta.jsonl --holdout-domains $HOLDOUT --reference "${REF[@]}" > $O/manifest.log 2>&1 || { tail -n 8 $O/manifest.log; log MANIFEST_FAILED; exit 1; }
  tail -n 15 $O/manifest.log
  python - <<'PY' | tee $O/manifest-check.txt || { log MANIFEST_CHECK_FAILED; exit 1; }
import json, collections, statistics
UNK = ("unknown", "null", "__unknown__")
for name in ("decision-p2c", "decision-p2c-fresh", "decision-p2c-judge-dev"):
    rows = [json.loads(l) for l in open(f"data/manifests/{name}.jsonl") if l.strip()]
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
    tp = [r for r in rows if isinstance(r.get("target_probs"), dict)]
    mass = [sum(float(r["target_probs"].get(k, 0)) for k in UNK) / max(1e-9, sum(map(float, r["target_probs"].values()))) for r in tp if r.get("target") not in (None,) + UNK]
    print(f"{name}: {len(rows)} rows {dict(collections.Counter(r.get('partition') for r in rows))}; image rows {sum(bool(r.get('images')) for r in rows)}; "
          f"target_probs {len(tp)}; rationale {sum(bool(r.get('rationale')) for r in rows)}; unknown-gold {sum(r.get('target') in (None,) + UNK for r in rows)}; "
          f"unknown mass on answerable soft rows mean {statistics.mean(mass) if mass else 0:.3f}")
    print("  sources", dict(collections.Counter(r.get("source") for r in rows).most_common(12)))
PY
  mark manifest; fi

stage pack
FILES=$(cd data && ls -d decision-p2/teacher/writer-p2c-judge.jsonl decision-p2/teacher/writer-p2c-unknown.jsonl decision-p2/teacher/writer-prog.jsonl \
  decision-p2/teacher/answers-p2c-relabel-qwen35.jsonl decision-p2/teacher/answers-p2c-new-qwen35.jsonl decision-p2/teacher/answers-p2c-new-gptoss.jsonl \
  decision-p2/teacher/records-p2c-writer.jsonl decision-p2/teacher/records-p2c.jsonl decision-p2/licenses/p2_teacher manifests/decision-p2c.jsonl \
  manifests/decision-p2c-fresh.jsonl manifests/decision-p2c-judge-dev.jsonl manifests/decision-p2c-report.json 2>/dev/null)
tar czf $P2/decision-p2c-teacher.tgz --warning=no-file-changed -C data $FILES -C $P2 --exclude='*.pid' out/p2c; rc=$?
[ $rc -le 1 ] && [ -s $P2/decision-p2c-teacher.tgz ] || { log PACK_FAILED; exit 1; }
ls -la $P2/decision-p2c-teacher.tgz; mark pack
echo ALL_DONE
