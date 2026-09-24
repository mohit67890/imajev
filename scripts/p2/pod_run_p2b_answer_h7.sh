#!/usr/bin/env bash
# Phase-2b ANSWER + assemble + manifest on the 7xH100 pod with THREE answerers (unanimous agreement): Qwen3.6-27B thinking (TP=4, GPUs 0-3,
# :8000), gpt-oss-20b (TP=2, GPUs 4-5, :8020) and Qwen3.6-35B-A3B thinking (TP=1, GPU 6, :8030). Runs after pod_run_p2b_gen_h7.sh's
# write stages (writer-p2b.jsonl present). Derived from pod_run_p2b_gen_h7.sh: two 27B writer replicas
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
stop(){ pkill -f "vllm.entrypoints.*--port 80[023]0" 2>/dev/null || true; sleep 8; } # only this script's servers (the 35B check owns :8030)
[ -f $D/records.jsonl ] || { log "MISSING phase-2 records $D/records.jsonl"; exit 1; }

Q35=hf/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
[ -f $D/writer-p2b.jsonl ] || { log "MISSING $D/writer-p2b.jsonl (run the write stages first)"; exit 1; }
log "p2b writer documents: $(wc -l < $D/writer-p2b.jsonl)"
if ! skip answer-qwen || ! skip answer-gptoss || ! skip answer-qwen35; then
  log "serve qwen 27B TP=4 (GPUs 0-3, :8000), gpt-oss TP=2 (GPUs 4-5, :8020), qwen 35B-A3B TP=1 (GPU 6, :8030)"
  serve 0,1,2,3 $QW qwen3.6-27b 8000 4 --reasoning-parser qwen3 & PA=$!
  serve 4,5 $GO gpt-oss-20b 8020 2 --reasoning-parser openai_gptoss & PG=$!
  serve 6 $Q35 qwen3.6-35b-a3b 8030 1 --reasoning-parser qwen3 --max-num-seqs 64 & P3=$!
  wait $PA || exit 1; wait $PG || exit 1; wait $P3 || exit 1
  log "stage answer (three answerers in parallel)"
  ( skip answer-qwen || { python scripts/p2/gen_answer.py --answerer qwen --writer $D/writer-p2b.jsonl --out $D/answers-p2b-qwen.jsonl --workers $WORKERS --max-tokens $ANS_MAXTOK --model qwen3.6-27b --base-url http://127.0.0.1:8000/v1 > $O/answer-qwen.log 2>&1 && touch $O/answer-qwen.DONE; } ) &
  ( skip answer-gptoss || { python scripts/p2/gen_answer.py --answerer gptoss --writer $D/writer-p2b.jsonl --out $D/answers-p2b-gptoss.jsonl --workers $WORKERS --max-tokens $ANS_MAXTOK --model gpt-oss-20b --base-url http://127.0.0.1:8020/v1 > $O/answer-gptoss.log 2>&1 && touch $O/answer-gptoss.DONE; } ) &
  ( skip answer-qwen35 || { python scripts/p2/gen_answer.py --answerer qwen35 --writer $D/writer-p2b.jsonl --out $D/answers-p2b-qwen35.jsonl --workers 64 --max-tokens 8000 --model qwen3.6-35b-a3b --base-url http://127.0.0.1:8030/v1 > $O/answer-qwen35.log 2>&1 && touch $O/answer-qwen35.DONE; } ) &
  wait; stop; for f in answer-qwen answer-gptoss answer-qwen35; do tail -n 1 $O/$f.log; done
  skip answer-qwen && skip answer-gptoss && skip answer-qwen35 || { log ANSWER_STAGE_FAILED; exit 1; }; fi

if ! skip assemble; then log "stage assemble (relaxed unknown rule, appended to phase-2 records, near-duplicate lint)"
  python scripts/p2/assemble_p2.py --writer $D/writer-p2b.jsonl --answers $D/answers-p2b-qwen.jsonl $D/answers-p2b-gptoss.jsonl $D/answers-p2b-qwen35.jsonl \
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
  decision-p2/teacher/answers-p2b-qwen.jsonl decision-p2/teacher/answers-p2b-gptoss.jsonl decision-p2/teacher/answers-p2b-qwen35.jsonl decision-p2/teacher/records-p2b.jsonl \
  manifests/decision-p2b.jsonl manifests/decision-p2b-jevstyle-dev.jsonl && ls -la $P2/decision-p2b-teacher.tgz; echo ALL_DONE
