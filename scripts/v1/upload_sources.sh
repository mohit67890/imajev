#!/usr/bin/env bash
# Stream each validated decision-v1 source to the bucket as one tar (no local copy). Skips sources already uploaded with the same record hash.
set -uo pipefail
B=gs://imajev-emoland-925e3/decision-v1; P=emoland-925e3; export COPYFILE_DISABLE=1
for src in "$@"; do
  d=data/decision-v1/$src
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$d/validation.json'))['summary']['errors']==0 else 1)" 2>/dev/null || { echo "$src: not validated, skipped"; continue; }
  h=$(shasum -a 256 "$d/records.jsonl" | cut -c1-16)
  if gcloud storage ls "$B/$src.$h.tar" --project $P >/dev/null 2>&1; then echo "$src: already uploaded ($h)"; continue; fi
  start=$(date +%s)
  tar -cf - "$d/records.jsonl" "$d/validation.json" "$d/README.md" "$d/images" 2>/dev/null | gcloud storage cp - "$B/$src.$h.tar" --project $P >/dev/null 2>&1 \
    && echo "$src: uploaded $(gcloud storage ls -l "$B/$src.$h.tar" --project $P | awk 'NR==1{printf "%.2f GB", $1/1e9}') in $(( $(date +%s)-start ))s" || echo "$src: UPLOAD FAILED"
done
