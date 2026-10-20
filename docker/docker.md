# Docker deployment

The image serves the existing playground and HTTP API on port 8765 using PyTorch.
CPU is the default; NVIDIA CUDA is optional. Images and text are both supported.
Apple MLX is not supported in these Linux containers.

## Docker Compose

Run from the `docker/` directory on Linux with Docker Engine and Docker Compose v2 installed.
The compose build context is the repository root (`..`), so the Dockerfile can copy
`pyproject.toml`, `src/`, `scripts/`, and `examples/` from there.
The container serves the playground and API on port 8765, published on the host
as `${HOST_PORT:-8765}`. Set `HOST_PORT` to expose a different host port; the
internal port stays 8765.

These commands run in the foreground and stream container logs to the terminal.
Press Ctrl+C to stop the service. To keep it running in the background instead,
add `-d` and follow the logs with `docker compose logs -f imajev`.

Choose **one command** below. All variants serve on host port `HOST_PORT` (default 8765),
the trained adapter, and the shipped calibration. Stop the previous instance
before switching models. Explicit variables override any matching `.env` settings.

### CPU: 2B

Standard: four option orders averaged, no extra flags.

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='' docker compose up --build
```

Lower latency: one option order, no extra flags.

```sh
MODEL_SIZE=2b ROTATIONS=1 SERVER_FLAGS='' docker compose up --build
```

### CPU: 4B

Standard:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='' docker compose up --build
```

Lower latency:

```sh
MODEL_SIZE=4b ROTATIONS=1 SERVER_FLAGS='' docker compose up --build
```

### CPU: 9B

Standard (requires substantial RAM):

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='' docker compose up --build
```

Lower latency:

```sh
MODEL_SIZE=9b ROTATIONS=1 SERVER_FLAGS='' docker compose up --build
```

CPU already uses float32. CUDA graphs and optimized GPU kernels do not accelerate
CPU inference. Warnings about missing `causal_conv1d` and `flash-linear-attention`
mean that slower reference PyTorch implementations are used, not that startup failed.

### NVIDIA GPU: Prerequisites

The GPU host needs a compatible NVIDIA driver for CUDA 12.8 and the NVIDIA
Container Toolkit configured for Docker. The CUDA runtime is supplied by the
PyTorch wheels; no CUDA toolkit installation is needed on the host.

Both Compose files are required. The GPU file inherits the port mapping and the
other common settings from the CPU file; do not launch it on its own.

GPU variants below:

| Variant | Rotations | Extra flags | Tradeoff |
| --- | --- | --- | --- |
| Standard | 4 | None | Unmerged adapter, four option orders averaged |
| Lower latency | 1 | None | Less work, without option-order averaging |
| Fast | 4 | `--fast` | Same decision computation; CUDA graphs need startup time and extra VRAM |
| Merged | 4 | `--merge-lora` | No CUDA graph capture; bf16 rounding can change near-tie decisions |
| Fast + merged | 4 | `--fast --merge-lora` | Merged adapter; bf16 rounding can change near-tie decisions |
| Fast + merged, single order | 1 | `--fast --merge-lora` | Combines kernel optimizations and less work per question |
| Float32 reference | 4 | `--float32` | Slower; weights need about twice the VRAM of bf16 |
| Think if unsure | 1 | `--thinking auto` | Optional generated reasoning; adds latency and may change answers |

The fast variants do not install `causal-conv1d` or `flash-linear-attention`;
those optional libraries are not included in the current image.

### NVIDIA GPU: 2B

Standard:

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Lower latency:

```sh
MODEL_SIZE=2b ROTATIONS=1 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast, adapter unmerged:

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='--fast' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Merged adapter without the fast path:

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='--merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter:

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter, single option order:

```sh
MODEL_SIZE=2b ROTATIONS=1 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Float32 reference:

```sh
MODEL_SIZE=2b ROTATIONS=4 SERVER_FLAGS='--float32' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Think if unsure:

```sh
MODEL_SIZE=2b ROTATIONS=1 SERVER_FLAGS='--thinking auto' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### NVIDIA GPU: 4B

Standard:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Lower latency:

```sh
MODEL_SIZE=4b ROTATIONS=1 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast, adapter unmerged:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='--fast' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Merged adapter without the fast path:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='--merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter, single option order:

```sh
MODEL_SIZE=4b ROTATIONS=1 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Float32 reference:

```sh
MODEL_SIZE=4b ROTATIONS=4 SERVER_FLAGS='--float32' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Think if unsure:

```sh
MODEL_SIZE=4b ROTATIONS=1 SERVER_FLAGS='--thinking auto' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### NVIDIA GPU: 9B

Standard:

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Lower latency:

```sh
MODEL_SIZE=9b ROTATIONS=1 SERVER_FLAGS='' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast, adapter unmerged:

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='--fast' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Merged adapter without the fast path:

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='--merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter:

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Fast + merged adapter, single option order:

```sh
MODEL_SIZE=9b ROTATIONS=1 SERVER_FLAGS='--fast --merge-lora' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Float32 reference (requires substantial VRAM):

```sh
MODEL_SIZE=9b ROTATIONS=4 SERVER_FLAGS='--float32' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Think if unsure:

```sh
MODEL_SIZE=9b ROTATIONS=1 SERVER_FLAGS='--thinking auto' docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### Logs and Health

```sh
docker compose logs -f imajev
curl --fail http://127.0.0.1:${HOST_PORT:-8765}/v1/models
```

Open <http://127.0.0.1:${HOST_PORT:-8765}/> on the server, or `http://SERVER_IP:${HOST_PORT:-8765}/` remotely,
for the playground. The decision endpoint is
`POST /v1/systemone` (see the README quickstart for requests).

The first start requires Internet access: it downloads the pinned Qwen base
checkpoint, the published imajev PEFT adapter, and its calibration. Downloads and
generated bundle paths live in the `model-data` volume. Subsequent starts reuse
them without downloading again. The health check allows one hour for initial
downloads and model loading; adjust it for a slower connection.

## Configuration

Compose accepts these variables from the shell or a root `.env` file:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST_PORT` | `8765` | Host port published for the container's internal port 8765 |
| `MODEL_SIZE` | `2b` | `2b`, `4b`, or `9b` |
| `ROTATIONS` | `4` | Candidate orders averaged per question; `1` reduces latency |
| `MAX_INPUT_TOKENS` | `4096` | Maximum processed request length |
| `SERVER_FLAGS` | Empty | Additional server CLI flags, split by Compose into arguments |

The CPU backend loads float32 weights, so allow substantially more RAM than the
download size (roughly 8/16/36 GB for 2B/4B/9B weights alone, plus loading and
request overhead). CPU inference can be slow. CUDA uses bfloat16, requiring
roughly half as much weight memory, plus GPU memory for requests and activations.
Use a GPU supporting bfloat16, typically NVIDIA Ampere or newer.

The API has no authentication. The container listens internally on `0.0.0.0:8765`,
published on the host as `HOST_PORT` (default 8765). Restrict that host port with
your firewall and use an authenticated TLS reverse proxy for public access.

Pass additional server flags using `SERVER_FLAGS`, as shown above. To change
the request length limit, prefix a launch command with `MAX_INPUT_TOKENS=8192`,
for example. Longer requests require additional memory; the model was trained
with a 4096-token limit. Thinking requires `ROTATIONS=1`.

For background operation, replace `up --build` with `up -d --build` in any
command above. To rebuild the same variant from scratch, use its same variables
and Compose files with `build --no-cache`, then run its launch command again.

## Docker Without Compose

CPU:

```sh
docker build -f docker/Dockerfile -t imajev:cpu ..
docker run -d --name imajev --init --restart unless-stopped \
  -p "${HOST_PORT:-8765}:8765" -v imajev-model-data:/data imajev:cpu
```

NVIDIA GPU:

```sh
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -f docker/Dockerfile -t imajev:cuda ..
docker run -d --name imajev-gpu --init --restart unless-stopped --gpus '"device=0"' \
  -e MODEL_SIZE=4b -p "${HOST_PORT:-8765}:8765" \
  -v imajev-model-data:/data imajev:cuda
```

Only run one of these containers on that host port at a time. A command after
the image name overrides normal startup, for example `imajev:cpu python -V`.
The container runs as UID 10001; a bind-mounted `/data` must be writable by that
UID. Named volumes are initialized with the correct permissions automatically.

`docker compose down` keeps the cached models. Removing the model volume also
removes the downloaded weights. Model weights are not baked into the image, and
Hugging Face credentials must not be passed as image build arguments.