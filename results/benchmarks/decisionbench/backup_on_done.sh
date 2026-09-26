#!/usr/bin/env bash
# Poll the pod every 2 min; on ALL_DONE pull the result tarball (+ run log, meta, dashboard), verify sha256, extract; print BACKUP_OK or BACKUP_FAILED.
H=${1:-imajev-db2}; OUT=${2:-reports/benchmarks/decisionbench-run}; NAME=${3:-decisionbench-imajev-4b-p3-full}
cd ""; mkdir -p "$OUT/$NAME"
for i in $(seq 1 90); do
  st=$(ssh -n -o ConnectTimeout=20 $H 'tail -n 1 db/run.log; grep -h DECISION_BENCH_PROGRESS db/full-summary.json 2>/dev/null | tail -n 1' 2>&1)
  echo "=== $(TZ=Asia/Kolkata date +%H:%M:%S) IST | $(echo "$st" | tr '\n' ' ' | cut -c1-160)"
  if echo "$st" | grep -q "ALL_DONE"; then
    R=$(ssh -n $H 'sha256sum decisionbench-imajev-4b-p3.tgz | cut -c1-64')
    scp -q $H:decisionbench-imajev-4b-p3.tgz "$OUT/$NAME.tgz" && L=$(shasum -a 256 "$OUT/$NAME.tgz" | cut -c1-64)
    scp -q $H:db/run.log $H:db/run-meta.json $H:launch.log "$OUT/$NAME/" 2>/dev/null
    echo "remote $R"; echo "local  $L"
    if [ "$R" = "$L" ]; then tar xzf "$OUT/$NAME.tgz" -C "$OUT/$NAME" && echo "BACKUP_OK $(TZ=Asia/Kolkata date +%H:%M:%S) IST $(du -sh "$OUT/$NAME" | cut -f1)"; else echo "BACKUP_FAILED sha mismatch"; fi
    exit 0
  fi
  if echo "$st" | grep -qE "FAILED (harness|precheck|fla)"; then echo "RUN_FAILED"; exit 1; fi
  sleep 120
done
echo "TIMEOUT"; exit 1
