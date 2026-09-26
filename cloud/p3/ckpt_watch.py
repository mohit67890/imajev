"""Phase-3 pod: keep a copy of every dev checkpoint of a training run (the trainer itself keeps only `best` and `last`).

The trainer appends a dev entry to <train>/log.jsonl right AFTER it has saved `last` at that step, and the next save is one
dev interval (~15% of the run) away, so copying `last` when a new dev entry appears captures exactly that step. The copy
leaves out trainer.pt (optimizer state). Each snapshot gets ckpt.json with the step and the dev metrics.

    python cloud/p3/ckpt_watch.py --train <run>/train --ckpts <O>/ckpts --prefix r1 --stop <O>/train-r1.DONE [--once]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def dev_entries(log: Path) -> list[dict]:
    if not log.exists():
        return []
    out = []
    for line in log.read_text().split("\n"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if "dev_loss" in e and e.get("step", 0) > 0:
            out.append(e)
    return out


def snapshot(train: Path, ckpts: Path, prefix: str, entry: dict) -> Path | None:
    src = train / "last"
    dst = ckpts / f"{prefix}-s{entry['step']:06d}"
    if dst.exists():
        return dst
    if not (src / "adapter_model.safetensors").exists():
        return None
    tmp = ckpts / (dst.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("trainer.pt"))
    (tmp / "ckpt.json").write_text(json.dumps({"prefix": prefix, **{k: entry.get(k) for k in ("step", "of", "dev_loss", "dev_accuracy", "dev2_loss", "dev2_accuracy")}}) + "\n")
    tmp.rename(dst)
    return dst


def sweep(train: Path, ckpts: Path, prefix: str) -> list[Path]:
    ckpts.mkdir(parents=True, exist_ok=True)
    made = []
    entries = dev_entries(train / "log.jsonl")
    if entries:  # only the newest dev entry can still match `last`; older ones were captured on earlier sweeps
        d = snapshot(train, ckpts, prefix, entries[-1])
        if d:
            made.append(d)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, required=True); ap.add_argument("--ckpts", type=Path, required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--stop", type=Path, help="stop after this marker exists (final sweep first)")
    ap.add_argument("--interval", type=float, default=15.0); ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    while True:
        stopping = a.stop is not None and a.stop.exists()
        for d in sweep(a.train, a.ckpts, a.prefix):
            print(time.strftime("%H:%M:%S"), "snapshot", d.name, flush=True)
        if a.once or stopping:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
