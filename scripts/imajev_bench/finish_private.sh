#!/usr/bin/env bash
# Finish a hidden (team-only) imajevBench set: wait for images, then items (all test) -> assemble -> lint -> promote.
# Usage: scripts/imajev_bench/finish_private.sh <build-dir> <dataset-dir> <build-log> <split-salt>
set -euo pipefail
BUILD=$1; DATA=$2; LOG=$3; SALT=$4
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
PY=.venv/bin/python
until grep -q PRIVATE-IMAGES-DONE "$LOG"; do sleep 60; done
$PY scripts/imajev_bench/build_v2.py items --out "$BUILD" --split-salt "$SALT" --dataset "$(basename "$DATA")" --dev 0 --calibration 0
$PY -m imajev_bench assemble --spec "$BUILD/spec.json" --output "$DATA" > "$DATA.assembly.log"
$PY -m imajev_bench lint --records "$DATA/records.jsonl" --allow-draft --output "$DATA/lint.json" || echo "LINT-NOT-PASSING"
$PY -m imajev_bench promote-constructed --records "$DATA/records.jsonl" --allow-draft --output "$DATA/records-promoted.jsonl"
echo PRIVATE-FINISHED
