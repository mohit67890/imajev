#!/bin/sh
set -eu

if [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; then
    exec "$@"
fi

case "$MODEL_SIZE" in
    2b) bundle=artifacts/model.json ;;
    4b) bundle=artifacts/model-qwen4b.json ;;
    9b) bundle=artifacts/model-qwen9b.json ;;
    *) printf 'MODEL_SIZE must be 2b, 4b or 9b\n' >&2; exit 2 ;;
esac

if [ ! -f "$bundle" ]; then
    python /app/scripts/download_model.py --model "$MODEL_SIZE"
fi

adapter="adapters/imajev-$MODEL_SIZE"
if [ ! -f "$adapter/adapter_config.json" ] \
    || [ ! -f "$adapter/adapter_model.safetensors" ] \
    || [ ! -f "$adapter/decision_readout.json" ] \
    || [ ! -f "$adapter/decision_readout.safetensors" ] \
    || [ ! -f "$adapter/calibration.json" ]; then
    hf download "mohit67890/imajev-$MODEL_SIZE" \
        --local-dir "$adapter" \
        --exclude 'mlx/*' --exclude '*.md' --exclude '.gitattributes'
fi

exec python /app/scripts/playground/server.py \
    --backend torch \
    --model-bundle "$bundle" \
    --adapter "$adapter" \
    --calibration "$adapter/calibration.json" \
    --model-name "imajev-$MODEL_SIZE" \
    --rotations "$ROTATIONS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --host 0.0.0.0 --port 8765 "$@"