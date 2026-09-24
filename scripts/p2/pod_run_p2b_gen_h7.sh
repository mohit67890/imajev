#!/usr/bin/env bash
# Phase-2b teacher generation on the 7xH100 pod (ei3vnzgkax4r3m), derived from pod_run_p2b_gen.sh: two 27B writer replicas
# (A = TP=4 on GPUs 0-3 port 8000, B = TP=2 on GPUs 4-5 port 8010) split the document range; GPU 6 is left to
# pod_check_35b_h7.sh (frozen Qwen3.6-35B-A3B JevBench check) and then serves gpt-oss-20b (TP=1) for the answer stage.
#   write-hard    Qwen3.6-27B TP=4, THINKING ON, max_tokens 16000: --docs 2400 over the eight hard families
#                 (multi_step_lookup, date_number_trap, policy_exception, rule_precedence, numerical_reconciliation,
#                  temporal_ordering, probability_estimate, tradeoff), tag p2b-hard
#   write-unknown Qwen3.6-27B TP=4, thinking OFF: --docs 900 over insufficient_evidence, ambiguity, contradiction, tag p2b-unknown
#                 (thinking off because an unknown-family document only needs a fact left out or stated twice; the thinking-off
#                  writer already produced these at ~75% acceptance in phase 2, and thinking costs ~5k tokens per document)
#   answer        both answerers in parallel: Qwen3.6-27B TP=2 on GPUs 0-1 port 8000, gpt-oss-20b TP=2 on GPUs 2-3 port 8010
#   assemble      --unknown-rule relaxed --batch-filter 'p2b-*' --append-to <phase-2 records> + contract validation + JevBench
#                 8-gram contamination lint + near-duplicate lint against phase-2 writer documents and human states
#   manifest      data/manifests/decision-p2b.jsonl (new rows + 30% replay of phase-2 train rows, 3 domains held out for the
#                 temperature fit) and data/manifests/decision-p2b-jevstyle-dev.jsonl (authored, 150 items)
# Time at ~2.5k output tok/s aggregate on TP=4 (thinking): hard pass 2,400 docs x ~7.5k tokens (reasoning + a long document)
#   = 18M tokens ~ 2.0 h; unknown pass 900 x ~2.5k = 2.3M ~ 15 min; answerers ~7.5k questions x 2 in parallel at phase-2 rates
#   ~ 40 min; assemble + manifest < 5 min. Total ~3 h, ~$55 at $18.36/h. Expected yield: ~1,800 hard + ~650 unknown documents
#   -> ~7,400 questions -> ~5,000 kept (70% agreement; relaxed unknown rule lifts the unknown share to ~15-20%).
# Stages carry DONE markers under p2/out/p2b (rerun skips finished stages). Sync this repo's scripts/p2 and
# scripts/playground into p2/imajev before running. Chain: cloud/pod_run_p2b_train.sh after ALL_DONE.
set -uo pipefail
P2=${P2:-p2}; cd $P2/imajev; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 OPENAI_API_KEY=local PATH=vllm-venv/bin:$PATH
O=$P2/out/p2b; D=data/decision-p2/teacher; MAN=data/manifests; mkdir -p $O $D $MAN
HARD_DOCS=${HARD_DOCS:-2400}; UNK_DOCS=${UNK_DOCS:-900}; WORKERS=${WORKERS:-192}; ANS_MAXTOK=${ANS_MAXTOK:-5000}
HARD_FAMILIES=multi_step_lookup,date_number_trap,policy_exception,rule_precedence,numerical_reconciliation,temporal_ordering,probability_estimate,tradeoff
UNK_FAMILIES=insufficient_evidence,ambiguity,contradiction
HOLDOUT=${HOLDOUT:-telecom,hospitality,nonprofit_grants}
QW=hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
GO=hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
log(){ echo "$(date +%H:%M:%S) $*"; }; skip(){ [ -f $O/$1.DONE ]; }
serve(){ # $1 gpus $2 path $3 name $4 port $5 tp, rest = extra vllm args
  local gpus=$1 path=$2 name=$3 port=$4 tp=$5; shift 5
  curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "$name" && { log "  $name already serving on $port"; return 0; }
  (CUDA_VISIBLE_DEVICES=$gpus vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model "$path" --served-model-name "$name" --port $port \
     --max-model-len 24576 --gpu-memory-utilization 0.92 --max-num-seqs $((96*tp)) --tensor-parallel-size $tp "$@" > $O/vllm-$name-$port.log 2>&1 &)
  for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:$port/v1/models | grep -q "$name" && return 0; sleep 5; done
  log "FAILED vllm $name"; tail -n 20 $O/vllm-$name-$port.log; return 1; }
stop(){ pkill -f "vllm.entrypoints.*--port 80[012]0" 2>/dev/null || true; sleep 8; } # only this script's servers (the 35B check owns :8030)
[ -f $D/records.jsonl ] || { log "MISSING phase-2 records $D/records.jsonl"; exit 1; }

if ! skip write-hard || ! skip write-unknown; then
  HA=$((HARD_DOCS*2/3)); HB=$((HARD_DOCS-HA))
  log "serve writer replicas: A TP=4 (GPUs 0-3, :8000) and B TP=2 (GPUs 4-5, :8010)"
  serve 0,1,2,3 $QW qwen3.6-27b 8000 4 --reasoning-parser qwen3 & PA=$!
  serve 4,5 $QW qwen3.6-27b 8010 2 --reasoning-parser qwen3 & PB=$!
  wait $PA || exit 1; wait $PB || exit 1
  log "stage write: hard $HA docs on A + $HB docs on B (thinking on), then unknown $UNK_DOCS docs on B (thinking off)"
  ( skip write-hard || python scripts/p2/gen_write.py --docs $HA --start 0 --out $D/writer-p2b-hard-a.jsonl --workers $WORKERS --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 \
      --thinking --max-tokens 16000 --families $HARD_FAMILIES --tag p2b-hard > $O/write-hard-a.log 2>&1 && touch $O/write-hard-a.DONE ) &
  ( skip write-hard || python scripts/p2/gen_write.py --docs $HB --start $HA --out $D/writer-p2b-hard-b.jsonl --workers $((WORKERS/2)) --model qwen3.6-27b --base-url http://127.0.0.1:8010/v1 \
      --thinking --max-tokens 16000 --families $HARD_FAMILIES --tag p2b-hard > $O/write-hard-b.log 2>&1 && touch $O/write-hard-b.DONE
    skip write-unknown || python scripts/p2/gen_write.py --docs $UNK_DOCS --out $D/writer-p2b-unknown.jsonl --workers $((WORKERS/2)) --model qwen3.6-27b --base-url http://127.0.0.1:8010/v1 \
      --max-tokens 8000 --families $UNK_FAMILIES --tag p2b-unknown > $O/write-unknown.log 2>&1 && touch $O/write-unknown.DONE ) &
  wait
  if ! skip write-hard; then [ -f $O/write-hard-a.DONE ] && [ -f $O/write-hard-b.DONE ] || { tail -n 3 $O/write-hard-a.log $O/write-hard-b.log; log WRITE_HARD_FAILED; stop; exit 1; }
    cat $D/writer-p2b-hard-a.jsonl $D/writer-p2b-hard-b.jsonl > $D/writer-p2b-hard.jsonl; tail -n 1 $O/write-hard-a.log; tail -n 1 $O/write-hard-b.log; touch $O/write-hard.DONE; fi
  skip write-unknown || { tail -n 3 $O/write-unknown.log; log WRITE_UNKNOWN_FAILED; stop; exit 1; }; tail -n 1 $O/write-unknown.log
  stop; fi
cat $D/writer-p2b-hard.jsonl $D/writer-p2b-unknown.jsonl > $D/writer-p2b.jsonl; log "p2b writer documents: $(wc -l < $D/writer-p2b.jsonl)"

if ! skip answer-qwen || ! skip answer-gptoss; then
  if [ -f $O/check35b.STARTED ] && [ ! -f $O/check35b.DONE ]; then log "waiting for the 35B-A3B check to release GPU 6"
    for i in $(seq 1 120); do [ -f $O/check35b.DONE ] && break; sleep 30; done; fi
  GG=6; [ -f $O/check35b.STARTED ] && [ ! -f $O/check35b.DONE ] && { log "35B check still running: gpt-oss shares GPUs 4-5 instead"; GG=4,5; }
  log "serve qwen A TP=4 (GPUs 0-3, :8000), qwen B TP=2 (GPUs 4-5, :8010), gpt-oss on GPU $GG (:8020)"
  serve 0,1,2,3 $QW qwen3.6-27b 8000 4 --reasoning-parser qwen3 & PA=$!
  serve 4,5 $QW qwen3.6-27b 8010 2 --reasoning-parser qwen3 & PB=$!
  if [ "$GG" = "6" ]; then serve 6 $GO gpt-oss-20b 8020 1 --reasoning-parser openai_gptoss & PG=$!; else serve 4,5 $GO gpt-oss-20b 8020 2 --reasoning-parser openai_gptoss --gpu-memory-utilization 0.45 & PG=$!; fi
  wait $PA || exit 1; wait $PB || exit 1; wait $PG || exit 1
  log "stage answer (parallel; qwen shards 0,1 of 3 on A and 2 of 3 on B)"
  ( skip answer-qwen || { python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer-p2b.jsonl --out $D/answers-p2b-qwen-0.jsonl --shard 0/3 --workers $WORKERS --max-tokens $ANS_MAXTOK --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 > $O/answer-qwen-0.log 2>&1 && touch $O/answer-qwen-0.DONE; } ) &
  ( skip answer-qwen || { python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer-p2b.jsonl --out $D/answers-p2b-qwen-1.jsonl --shard 1/3 --workers $WORKERS --max-tokens $ANS_MAXTOK --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 > $O/answer-qwen-1.log 2>&1 && touch $O/answer-qwen-1.DONE; } ) &
  ( skip answer-qwen || { python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer-p2b.jsonl --out $D/answers-p2b-qwen-2.jsonl --shard 2/3 --workers $((WORKERS/2)) --max-tokens $ANS_MAXTOK --model qwen3.6-27b --base-url http://127.0.0.1:8010/v1 > $O/answer-qwen-2.log 2>&1 && touch $O/answer-qwen-2.DONE; } ) &
  ( skip answer-gptoss || { python scripts/p2/gen_answer.py --answerer gptoss --writer $D/writer-p2b.jsonl --out $D/answers-p2b-gptoss.jsonl --workers $WORKERS --max-tokens $ANS_MAXTOK --model gpt-oss-20b --base-url http://127.0.0.1:8020/v1 > $O/answer-gptoss.log 2>&1 && touch $O/answer-gptoss.DONE; } ) &
  wait; stop
  if ! skip answer-qwen; then [ -f $O/answer-qwen-0.DONE ] && [ -f $O/answer-qwen-1.DONE ] && [ -f $O/answer-qwen-2.DONE ] || { tail -n 3 $O/answer-qwen-*.log; log ANSWER_STAGE_FAILED; exit 1; }
    cat $D/answers-p2b-qwen-0.jsonl $D/answers-p2b-qwen-1.jsonl $D/answers-p2b-qwen-2.jsonl > $D/answers-p2b-qwen.jsonl; touch $O/answer-qwen.DONE; fi
  for i in 0 1 2; do tail -n 1 $O/answer-qwen-$i.log; done; tail -n 1 $O/answer-gptoss.log
  skip answer-qwen && skip answer-gptoss || { log ANSWER_STAGE_FAILED; exit 1; }; fi

if ! skip assemble; then log "stage assemble (relaxed unknown rule, appended to phase-2 records, near-duplicate lint)"
  python scripts/p2/assemble_p2.py --writer $D/writer-p2b.jsonl --answers $D/answers-p2b-qwen.jsonl $D/answers-p2b-gptoss.jsonl \
     --out $D/records-p2b.jsonl --append-to $D/records.jsonl --unknown-rule relaxed --batch-filter 'p2b-*' \
     --dedupe-against $D/writer.jsonl $P2/imajev/data/decision-p2/human/records.jsonl --dedupe-threshold 5 \
     --reference $P2/jevbench-public/*.jsonl > $O/assemble.log 2>&1 || { tail -n 5 $O/assemble.log; log ASSEMBLE_FAILED; exit 1; }
  tail -n 4 $O/assemble.log; touch $O/assemble.DONE; fi

if ! skip manifest; then log "stage manifest: decision-p2b (+30% phase-2 replay, holdout $HOLDOUT) and the authored JevBench-style dev"
  python scripts/p2/build_p2b_manifest.py --p2 $MAN/decision-p2.jsonl --records $D/records-p2b.jsonl --batch-filter 'p2b-*' --replay 0.30 \
     --holdout-domains $HOLDOUT --out $MAN/decision-p2b.jsonl > $O/manifest.log 2>&1 || { tail -n 5 $O/manifest.log; log MANIFEST_FAILED; exit 1; }
  python scripts/p2/author_jevstyle_dev.py --out $MAN/decision-p2b-jevstyle-dev.jsonl >> $O/manifest.log 2>&1 || { tail -n 5 $O/manifest.log; log JEVSTYLE_FAILED; exit 1; }
  cat $O/manifest.log; touch $O/manifest.DONE; fi

log pack; tar czf $P2/decision-p2b-teacher.tgz -C data decision-p2/teacher/writer-p2b-hard.jsonl decision-p2/teacher/writer-p2b-unknown.jsonl \
  decision-p2/teacher/answers-p2b-qwen.jsonl decision-p2/teacher/answers-p2b-gptoss.jsonl decision-p2/teacher/records-p2b.jsonl \
  manifests/decision-p2b.jsonl manifests/decision-p2b-jevstyle-dev.jsonl && ls -la $P2/decision-p2b-teacher.tgz; echo ALL_DONE
