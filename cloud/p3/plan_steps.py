"""Phase-3 pod: the optimizer-step count the trainer will plan, computed on the CPU with the trainer's own batch planner, so the
full run can checkpoint every ~15% (--dev-every = ceil(frac x steps)). Mirrors scripts/train_decision_lora_torch.py:
plans = batch_plan(train, pixels, token_budget, batch_size, seed, epoch, rationale_tokens) per epoch, trimmed to the
fractional epoch count, steps = ceil(microbatches / (accumulate x world)). The watcher (cloud/p3/ckpt_watch.py) keys on the
trainer's own `of` field anyway, so a small drift here only moves the snapshot points.

    python cloud/p3/plan_steps.py --version decision-p3 --epochs 2 --token-budget 16384 --batch-size 40 --accumulate 1 --world 8 --rationale-tokens 192
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path[:0] = ["src", "scripts"]


def steps_for(n_microbatches_per_epoch: list[int], epochs: float, accumulate: int, world: int) -> int:
    need = math.ceil(epochs)
    flat = sum(n_microbatches_per_epoch[:need])
    flat = round(flat * epochs / need)
    return math.ceil(flat / (accumulate * world))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True); ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=16384); ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--accumulate", type=int, default=1); ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pixels", type=int, default=400000)
    ap.add_argument("--rationale-tokens", type=int, default=192); ap.add_argument("--frac", type=float, default=0.15)
    a = ap.parse_args(argv)
    from decision_data import batch_plan, load_records
    train = load_records("train", a.version)
    per_epoch = [len(batch_plan(train, a.pixels, a.token_budget or 10**9, a.batch_size, a.seed, e, rationale_tokens=a.rationale_tokens))
                 for e in range(math.ceil(a.epochs))]
    steps = steps_for(per_epoch, a.epochs, a.accumulate, a.world)
    out = {"version": a.version, "train_items": len(train), "microbatches_per_epoch": per_epoch, "steps": steps,
           "dev_every": max(5, math.ceil(a.frac * steps))}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
