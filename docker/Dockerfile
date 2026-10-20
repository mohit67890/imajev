FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    MODEL_SIZE=2b \
    ROTATIONS=4 \
    MAX_INPUT_TOKENS=4096

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && useradd --uid 10001 --create-home imajev \
    && mkdir /data \
    && chown imajev:imajev /data

ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install '.[serve,torch]' \
    && python -m pip check \
    && python -c "from transformers import Qwen3_5ForConditionalGeneration; import torchvision"

COPY scripts/ scripts/
COPY examples/ examples/
COPY --chmod=755 docker/scripts/entrypoint.sh /usr/local/bin/imajev-entrypoint

USER imajev
WORKDIR /data
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=60m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/v1/models', timeout=3)" || exit 1

ENTRYPOINT ["imajev-entrypoint"]