"""Download a pinned base checkpoint into the repository's local Hugging Face cache and write its bundle file.

    python scripts/download_model.py                       # Qwen/Qwen3.5-2B → artifacts/model.json
    python scripts/download_model.py --model 9b            # Qwen/Qwen3.5-9B → artifacts/model-qwen9b.json

The bundle records the pinned repo, revision and snapshot path that the backends load from.
"""
import argparse, json, os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

PINNED = {
    "2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "artifacts/model.json", 6),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "artifacts/model-qwen9b.json", 20),
}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=sorted(PINNED), default="2b"); a = ap.parse_args()
    repo, revision, bundle, cap_gib = PINNED[a.model]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    size = sum(s.size or 0 for s in info.siblings)
    if size > cap_gib * 1024**3: raise SystemExit(f"{repo} exceeds the {cap_gib} GiB cap: {size}")
    print(f"Downloading {repo}@{info.sha}: {size / 1024**3:.2f} GiB", flush=True)
    path = snapshot_download(repo, revision=info.sha, ignore_patterns=["*.md", ".gitattributes"])
    Path("artifacts").mkdir(exist_ok=True)
    Path(bundle).write_text(json.dumps({"repo": repo, "revision": info.sha, "path": path, "repository_bytes": size}, indent=2) + "\n")
    print(path)


if __name__ == "__main__": main()
