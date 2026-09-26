"""Phase-3 pod: continue the winning pilot lane into the full run (cloud/p3/pod_run_train.sh, stage 4).

The pilot lanes are the full run's first PILOT_STEPS optimizer steps for their switches: same manifest, seed, start adapter,
epochs (so the same step count and learning-rate schedule), batch plan and micro-batches per step (lane: 2 GPUs x PILOT_ACC;
full run: TRAIN_WORLD x ACC). This picks the winner's `last/` checkpoint and prints the trainer flags that resume it:

  exact     the same micro-batches per step: `--resume-from <lane>/train/last`. The full run then trains exactly the steps an
            uninterrupted run would train after step PILOT_STEPS, on the same micro-batches (tests/test_p3_resume.py).
  inexact   a different per-step count (only with EVAL_DURING_TRAIN=1: 7 training GPUs x 1 = 7 vs the lane's 8): adds
            `--resume-inexact`; the trainer starts at floor(consumed / per-step), so nothing is skipped and fewer than one step's
            micro-batches (<= per-step - 1) are trained twice; the schedule is re-derived for the full run's step count.
  fallback  anything missing or inconsistent, or COMBINED winners (two or more factors won; no lane trained them together): prints
            nothing, and the full run restarts from the shipped adapter with the combined flags (the old path).
The trainer itself re-checks the configs (a mismatch exits before its first save; the pod script then falls back as well).

    python cloud/p3/lane_resume.py --pilot-dir <O>/pilot --decision <O>/pilot-decision.json --steps 100 --world 8 --accumulate 1 --out <O>/lane-resume.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANES = {"baseline": (0.0, 255, 0), "ordinal": (0.15, 255, 0), "codes256": (0.0, 256, 0), "rank64": (0.0, 255, 64)}


def read_log(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().split("\n"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def plan(pilot_dir: Path, decision: dict, steps: int, world: int, accumulate: int) -> dict:
    """{'ok', 'lane', 'source', 'mode', 'flags', ...} or {'ok': False, 'reason'}."""
    switches = (decision.get("ordinal_weight"), decision.get("readout_codes"), int(decision.get("expand_lora_rank") or 0))
    lane = decision.get("lane") if "lane" in decision else next((k for k, v in LANES.items() if v == switches), None)
    res = {"ok": False, "lane": lane, "flags": []}

    def fail(why: str) -> dict:
        res["reason"] = why
        res["summary"] = f"lane resume: FALLBACK (full run restarts from the shipped adapter): {why}"
        return res

    if decision.get("combined"):
        return fail(f"COMBINED winners {decision.get('winners')}: no pilot lane trained them together")
    if lane not in LANES:
        return fail(f"no pilot lane matches the decision {decision.get('ordinal_weight')} / {decision.get('readout_codes')} / "
                    f"rank x{decision.get('expand_lora_rank') or 0}")
    w, codes, rx = LANES[lane]
    if (float(decision.get("ordinal_weight", -1)), int(decision.get("readout_codes", 0)), int(decision.get("expand_lora_rank") or 0)) != (w, codes, rx):
        return fail(f"decision lane {lane} disagrees with its switches")
    train = pilot_dir / lane / "train"; src = train / "last"
    if not (pilot_dir.parent / f"pilot-{lane}.DONE").exists():
        return fail(f"pilot lane {lane} did not finish (no pilot-{lane}.DONE)")
    for f in ("trainer.pt", "adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json"):
        if not (src / f).exists():
            return fail(f"{src / f} missing")
    if not (train / "config.json").exists():
        return fail(f"{train / 'config.json'} missing")
    cfg = json.loads((train / "config.json").read_text())
    if float(cfg.get("ordinal_weight", 0.0)) != w or int(cfg.get("readout_codes", 255)) != codes or int(cfg.get("expand_lora_rank", 0)) != rx:
        return fail(f"lane {lane} config has ordinal_weight {cfg.get('ordinal_weight', 0.0)} / readout_codes {cfg.get('readout_codes', 255)}"
                    f" / expand_lora_rank {cfg.get('expand_lora_rank', 0)}")
    dev = [e for e in read_log(train / "log.jsonl") if "dev_loss" in e and e.get("step", 0) > 0]
    if not dev:
        return fail(f"lane {lane} has no saved step")
    # `last` is saved right before its dev entry is logged, so the newest dev entry is its step. A lane retry can run past
    # PILOT_STEPS (--max-steps counts from the resume point); any saved step is still a prefix of the full run, so it is used.
    at = dev[-1]["step"]
    if at != steps:
        res["note"] = f"lane {lane} saved last at step {at}, not {steps}"
    src_per = int(cfg["world_size"]) * int(cfg["accumulate"]); per = world * accumulate
    consumed = at * src_per; start = consumed // per
    if start < 1:
        return fail(f"lane {lane} step {at} maps to full-run step {start}")
    res.update(ok=True, source=str(src), source_step=at, source_microbatches_per_step=src_per, microbatches_per_step=per,
               source_microbatches=consumed, start_step=start, repeated_microbatches=consumed - start * per, skipped_microbatches=0,
               mode="exact" if src_per == per else "inexact")
    res["flags"] = ["--resume-from", str(src)] + ([] if src_per == per else ["--resume-inexact"])
    res["summary"] = (f"lane resume: {res['mode'].upper()} from pilot lane {lane} at step {at} ({consumed} micro-batches; "
                      f"{src_per} -> {per} per step; full run starts at step {start}, {res['repeated_microbatches']} micro-batches repeated, 0 skipped)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", type=Path, required=True); ap.add_argument("--decision", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True, help="PILOT_STEPS: the step the lanes stopped at")
    ap.add_argument("--world", type=int, required=True, help="GPUs of the full run"); ap.add_argument("--accumulate", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        res = plan(a.pilot_dir, json.loads(a.decision.read_text()), a.steps, a.world, a.accumulate)
    except Exception as e:   # any surprise = the old path
        res = {"ok": False, "flags": [], "reason": repr(e), "summary": f"lane resume: FALLBACK (full run restarts from the shipped adapter): {e!r}"}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(" ".join(res["flags"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
