#!/usr/bin/env bash
# Finish an imajev-bench v2 build: items -> assemble -> lint -> promote -> audit packet -> API evaluation -> release check.
# Usage: scripts/imajev_bench/finish_v2.sh <build-dir> <dataset-dir> <report-dir> <split-salt>
# Each step writes new files only; rerunning after a partial failure needs fresh output paths.
set -euo pipefail
BUILD=$1; DATA=$2; REPORT=$3; SALT=$4
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
PY=.venv/bin/python
bench() { $PY -m imajev_bench "$@"; }

$PY scripts/imajev_bench/build_v2.py items --out "$BUILD" --split-salt "$SALT" --dataset "$(basename "$DATA")"
bench assemble --spec "$BUILD/spec.json" --output "$DATA" > "$DATA.assembly.log"
bench lint --records "$DATA/records.jsonl" --allow-draft --output "$DATA/lint.json"
bench promote-constructed --records "$DATA/records.jsonl" --allow-draft --output "$DATA/records-promoted.jsonl"
mkdir -p "$DATA/packets"
bench review --records "$DATA/records-promoted.jsonl" --allow-draft --plan audit --reviewer-id auditor-1 \
  --output "$DATA/packets/audit-auditor-1.html"

# Evaluation roster. Reasoning setting is part of each entry's identity and is recorded in its run manifest.
mkdir -p "$REPORT/eval"
run() {  # provider model tag [extra args...]
  local provider=$1 model=$2 tag=$3; shift 3
  for split in dev calibration test; do
    grep -q "\"split\": \"$split\"" "$DATA/records-promoted.jsonl" || continue
    bench api-run --records "$DATA/records-promoted.jsonl" --allow-draft --split "$split" --allow-test-exposure \
      --provider "$provider" --model "$model" --workers 4 "$@" --output "$REPORT/eval/$tag-$split" > /dev/null 2>&1 \
      || echo "FAILED $tag $split"
  done
  echo "done $tag"
}
run azure-openai gpt-5.4 gpt-5.4-medium --reasoning-effort medium &
run azure-openai gpt-5.6-luna gpt-5.6-luna-default &
run azure-openai grok-4.3 grok-4.3-default --free-json &
run vertex-gemini gemini-3.1-pro-preview gemini-3.1-pro-default &
run vertex-gemini gemini-3.8-flash gemini-3.8-flash-default &
wait

$PY scripts/imajev_bench/summarize_eval.py --records "$DATA/records-promoted.jsonl" --runs "$REPORT/eval" \
  --checkers gpt-5.6-luna gemini-3.1-pro --output "$REPORT/eval-summary"
bench release-check --records "$DATA/records-promoted.jsonl" --pilot --preview --output "$REPORT/release-check.json"
echo FINISHED
