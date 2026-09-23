#!/usr/bin/env bash
# Bootstrap a fresh RunPod pytorch image (cu128, torch 2.8) for the imajevBench + release test.
# Expects release-test-bundle.tgz and hf-token (scp'd). Disk: 200 GB.
set -euo pipefail; cd /workspace
tar xzf release-test-bundle.tgz && ls
python -m venv venv --system-site-packages && . venv/bin/activate
pip install -q "transformers==5.17.0" "peft==0.21.0" "safetensors>=0.8" "pillow>=11" "pydantic>=2,<3" "huggingface_hub>=1.32" fastapi "uvicorn[standard]" python-multipart accelerate
# vLLM in its own venv so its torch pin cannot disturb the scoring venv
python -m venv vllm-venv && vllm-venv/bin/pip install -q vllm 2>&1 | tail -n 2 || echo "VLLM_INSTALL_FAILED"
export HF_HOME=hf HF_TOKEN=$(cat hf-token); mkdir -p $HF_HOME adapters
cd imajev && for m in 2b 4b 9b gemma-e2b gemma-e4b; do python scripts/download_model.py --model $m || echo "DOWNLOAD_FAILED $m"; done; cd ..
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ("imajev-2b","imajev-9b"): print(snapshot_download(f"mohit67890/{r}", local_dir=f"adapters/{r}", ignore_patterns=["mlx/*"]))
PY
rm -f hf-token; unset HF_TOKEN; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /workspace | tail -1; echo BOOTSTRAP_DONE
