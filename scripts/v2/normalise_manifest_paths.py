"""Rewrite stored image paths in a source's raw manifest to repo-relative POSIX paths.

The fetchers now write relative paths; this fixes manifests written before that change, so nothing
that travels to a pod carries an absolute path from a laptop.  Idempotent.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/normalise_manifest_paths.py commons_photos
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, repo_relative


def main(source: str) -> int:
    changed = 0
    for path in ((RAW / source / "manifest.jsonl"), (OUT / source / "attribution.jsonl")):
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        for row in rows:
            for key in ("image", "path"):
                if key in row and isinstance(row[key], str):
                    fixed = repo_relative(row[key])
                    changed += fixed != row[key]
                    row[key] = fixed
        path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
        print(f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}: {len(rows)} rows")
    print(f"paths rewritten: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
