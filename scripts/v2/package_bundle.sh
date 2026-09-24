#!/usr/bin/env bash
# Pack the v2 pod bundle: candidate sources (records + images + licences), manifests (v1.1 for the blend, reasoning dev, panels),
# adapters (2B v1.1 student init, 9B teacher + calibration), licence evidence. Uploads to gs://<bucket>/decision-v2/.
set -euo pipefail; cd "$(dirname "$0")/../.."; export COPYFILE_DISABLE=1
B=gs://<bucket>/decision-v2; P=<gcp-project>; SOURCES=${SOURCES:-"commons_photos openimages_v2 pd12m stackexchange wikipedia_paragraphs support_reviews"}
S=${SCRATCH:-/tmp}/decision-v2.tar; items=(data/manifests/decision-v1.1.jsonl data/manifests/decision-v2-reasoning-dev.jsonl data/manifests/decision-v1-test.jsonl
  data/manifests/decision-v1.1-heldout-text.jsonl data/manifests/decision-v1.1-heldout-text-unlicensed.jsonl data/manifests/decision-v1.1-irrelevance-test.jsonl
  data/provenance/licenses data/decision-v1-text/licenses data/decision-v2/licenses reports/v1.1-datasets/image-policy-a.json
  reports/decision-v1.1/runs/h100x4/best reports/decision-v1.1-9b/runs/h200x4/best reports/decision-v1.1-9b/calibration-9b.json reports/decision-v1.1/calibration-v1.1.json)
for s in $SOURCES; do d=data/decision-v2/$s; [ -f $d/records.jsonl ] && items+=($d/records.jsonl $d/README.md) && [ -d $d/images ] && items+=($d/images) || echo "skip $s"; done
tar -cf "$S" --exclude='__pycache__' --exclude='trainer.pt' "${items[@]}"; h=$(shasum -a 256 "$S" | cut -c1-16); mv "$S" "${S%.tar}.$h.tar"
ls -l "${S%.tar}.$h.tar" | awk '{printf "bundle %.2f GB\n", $5/1e9}'; gcloud storage cp "${S%.tar}.$h.tar" "$B/decision-v2.$h.tar" --project $P >/dev/null && echo "uploaded decision-v2/decision-v2.$h.tar"
