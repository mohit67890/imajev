#!/usr/bin/env bash
# Runs ON the pod: one snapshot of the DecisionBench run (called by dash_db.sh over ssh).
B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; R=$'\033[31m'; Z=$'\033[0m'; L0=db/run.log
# after a RESTART marker only the lines after it count (the killed first attempt left a "FAILED line" trap message behind)
if grep -q "^RESTART" $L0; then L=/tmp/run-since-restart.log; sed -n "$(grep -n "^RESTART" $L0 | tail -n 1 | cut -d: -f1),\$p" $L0 > $L; else L=$L0; fi
echo "${B}DecisionBench 1.0 · imajev-4b phase 3${Z}   $(TZ=Asia/Kolkata date +%H:%M:%S) IST   run started $(TZ=Asia/Kolkata date -d @$(stat -c %W db/run.log 2>/dev/null || stat -c %Y db/run.log) +%H:%M) IST"
grep -q "^RESTART" $L0 && echo "${D}restarted $(grep "^RESTART" $L0 | tail -n 1 | cut -c9-33) with flash-linear-attention (first attempt: slow path, kept in db/full-partial-slowpath)${Z}"
line=""; for m in SERVERS_READY ALL_DONE; do if grep -q "$m" $L 2>/dev/null; then line+="  ${G}✓${Z}$m"; else line+="  ${D}·$m${Z}"; fi; done
grep -qE "FAILED (harness|fla)" $L 2>/dev/null && line+="  ${R}${B}FAILED${Z}"; echo "$line"
grep -hE "GPUs |adapter:|setup |fla " $L0 2>/dev/null | tail -n 4 | cut -c1-150
NS=$(grep -h "GPUs " $L0 2>/dev/null | tail -n 1 | grep -oE "= [0-9]+" | tr -dc 0-9); NS=${NS:-24}
if [ -s precheck.jsonl ] && ! grep -q "PRECHECK_DONE\|FAILED precheck" $L 2>/dev/null && grep -q "SERVERS_READY" $L 2>/dev/null; then
  tot=$(wc -l < precheck.jsonl | tr -d " "); done=$(cat db/server*.log 2>/dev/null | grep -c "questions=")
  echo "${B}pre-check${Z}  replaying the $tot requests that errored at 32k through all $NS servers at the 64k limit: $done answered so far (stops the run on any error)"
elif grep -q "PRECHECK_DONE" $L 2>/dev/null; then echo "${B}pre-check${Z}  $(grep -h "^PRECHECK " $L | tail -n 1)"; fi
prog=$(grep -h "DECISION_BENCH_PROGRESS" db/full-summary.json 2>/dev/null | tail -n 1)
if [ -n "$prog" ]; then
  n=$(echo "$prog" | sed -n 's/.*completed=\([0-9]*\).*/\1/p'); rps=$(echo "$prog" | sed -n 's/.*rows_per_second=\([0-9.]*\).*/\1/p')
  left=$(awk -v n="$n" -v r="$rps" 'BEGIN{ if (r>0) printf "%d", (23900-n)/r/60; else print 0 }')
  bar=$(( n*40/23900 )); printf "${B}full run${Z}  [%s%s] %5d / 23900 (%d%%)   %.0f rows/s (harness)   ~%s min left\n" "$(printf '█%.0s' $(seq 1 $bar))" "$(printf '░%.0s' $(seq 1 $((40-bar))))" "$n" $((n*100/23900)) "$rps" "$left"
  echo "${D}raw rows on disk: $(cat db/full/raw.jsonl 2>/dev/null | wc -l | tr -d ' ') (written in batches)${Z}"
else echo "${B}full run${Z}  not started (smoke run or setup)"; fi
echo "${B}gpus${Z}  $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "%3d%%/%dG ", $1, $2/1024}')"
echo "${B}servers${Z} $(pgrep -f 'scripts/playground/server.py --backend' | wc -l | tr -d ' ')/$NS up   ${B}harness${Z} $(pgrep -f 'run-system-one-http' | wc -l | tr -d ' ') process"
echo "${B}log${Z}"; grep -v '^\s*$' $L | grep -v "HF_TOKEN\|Generating eval split\|FAILED line\|Terminated" | tail -n 14 | cut -c1-170
grep -v "DECISION_BENCH_PROGRESS" db/full-summary.json 2>/dev/null | grep -q . && echo "${B}summary${Z} $(grep -v DECISION_BENCH_PROGRESS db/full-summary.json | tr -d '\n' | head -c 300)"
