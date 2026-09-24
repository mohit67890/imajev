#!/usr/bin/env bash
# Live phase-2c dashboard (refreshes every 30 s)
O=p2/out/p2c; T=p2/train-out-p2c
while true; do clear; echo "imajev phase-2c on H100x8   pod clock $(date +%H:%M:%S) UTC  (IST = +5:30)"; echo
  echo "BOOTSTRAP"; grep -vE "Fetching|it/s\]|LIBARCHIVE" bootstrap.log 2>/dev/null | grep -E "bootstrap:|BOOTSTRAP_DONE|MISSING|FAILED" | tail -n 3 | cut -c1-140
  echo; echo "GENERATION  ($O/run.log)"; grep -E "stage |done |FAILED|MISSING|ALL_DONE|Traceback" $O/run.log 2>/dev/null | tail -n 5 | cut -c1-140
  for p in $O/*.log; do [ -f "$p" ] || continue; f=$(basename $p .log); case $f in run|vllm-*) continue;; esac
    line=$(tail -c 300 $p | tr '\r' '\n' | grep -v '^$' | tail -n 1 | cut -c1-110)
    echo "  $f: $line"; done
  echo "  live output rows (grow continuously):"
  for o in data/decision-p2/teacher/answers-p2c-*.jsonl data/decision-p2/teacher/writer-p2c-judge.jsonl data/decision-p2/teacher/writer-p2c-unknown.jsonl; do [ -f "$o" ] && printf "    %-42s %6d rows  (updated %s)\n" "$(basename $o)" "$(wc -l < $o)" "$(date -r $o +%H:%M:%S)"; done
  [ -f $O/unknown-audit.txt ] && { echo "  unknown audit:"; tail -n 3 $O/unknown-audit.txt | cut -c1-120 | sed 's/^/    /'; }
  echo; echo "TRAINING  ($T/run.log)"; grep -E "stage |SMOKE|jevbench|imajevbench|PASS|FAIL|NOT_SHIPPABLE|ALL_DONE|attempt|Traceback" $T/run.log 2>/dev/null | tail -n 8 | cut -c1-140 || echo "  not started (chained on generation ALL_DONE)"
  for r in 4b-delta 4b-fresh 9b-delta 2b-delta; do f=$T/$r/train.log; [ -f $f ] && echo "  $r: $(grep -h '"step"' $f | tail -n 1 | cut -c1-110)"; done
  echo; echo "GPUS"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | awk -F', ' '{printf "  gpu%s %4s %s\n",$1,$2,$3}'
  sleep 30; done
