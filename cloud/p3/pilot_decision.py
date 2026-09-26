"""Phase-3 pod pilot: pick the Stage-3 winners from the four pilot lanes (docs/phase-3-plan.md; reports/phase3/ordinal-loss.md).

Each lane is the first PILOT_STEPS (default 100) optimizer steps of the full run for its switches: same data order, seed, schedule
and micro-batches per step (2 GPUs x accumulate 4 = 8 GPUs x 1), so the full run continues the winner (`lane` in the output;
cloud/p3/lane_resume.py). 100 full-size steps are the same 800 micro-batches the old 200 half-size pilot steps trained.

Lanes (identical except for the ONE switch under test; same manifest, seed, start adapter, flags, dev sets; every lane logs
--dev-slices, i.e. image-row and text-row dev accuracy):
  baseline   --ordinal-weight 0     --readout-codes 255   rank 16 (the shipped adapter's)
  ordinal    --ordinal-weight 0.15  --readout-codes 255   rank 16
  codes256   --ordinal-weight 0     --readout-codes 256   rank 16
  rank64     --ordinal-weight 0     --readout-codes 255   --expand-lora-rank 64 (identical output at step 0; scripts/lora_expand.py)
dev  = the Stage-3 manifest's dev partition restricted to <= 254 options (identical for every lane; plain CE in every lane); its
       slices are image rows and text rows (the dev partition holds only verified hard rows, so "text" = hard text)
dev2 = the ordinal-only dev manifest (score questions), so dev2_accuracy is score accuracy.

Metrics per lane: the mean of the last TAIL dev checkpoints (default 2: steps 75 and 100 with dev every 25), which halves the noise of one
400-row pass. Rules (ordinal-loss.md "Decision rule", plus the plan's gate for item 5):
  no_harm(lane)  dev_accuracy >= baseline - 1.0 pt  AND  dev_loss <= baseline x 1.02
  ordinal wins   no_harm(ordinal) AND dev2_accuracy(ordinal) > dev2_accuracy(baseline)
  256 wins       CUDA parity passed AND the manifest has 255-option training rows AND no_harm(codes256)
  rank64 wins    no_harm(rank64) AND no dev slice (image, text) more than 2.0 pts below baseline AND clearly better beyond noise:
                 dev accuracy >= baseline + 1.5 pts with dev loss not above baseline, OR dev loss <= baseline x 0.97 with dev
                 accuracy not below baseline. Otherwise r16 is kept (the shipped structure).
The three factors are decided INDEPENDENTLY against baseline and the full run COMBINES the winners. `lane` names the pilot lane
whose switches equal the combination (the full run continues it: cloud/p3/lane_resume.py); when two or more factors win there is
no such lane (`lane` null, `combined` true) and the full run restarts from the shipped adapter with the combined flags.
A lane that is missing, crashed or never logged a dev pass loses. The fallback is always W = 0, 255 codes, rank 16.

    python cloud/p3/pilot_decision.py --pilot-dir <O>/pilot --parity <O>/parity/summary.json --manifest-info <O>/manifests.json --out <O>/pilot-decision.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RANK_X = 64
LANES = {"baseline": {"ordinal_weight": 0.0, "readout_codes": 255, "expand_lora_rank": 0},
         "ordinal": {"ordinal_weight": 0.15, "readout_codes": 255, "expand_lora_rank": 0},
         "codes256": {"ordinal_weight": 0.0, "readout_codes": 256, "expand_lora_rank": 0},
         "rank64": {"ordinal_weight": 0.0, "readout_codes": 255, "expand_lora_rank": RANK_X}}
DEV_ACC_TOL = 0.01      # fraction (1.0 pt)
DEV_LOSS_REL = 0.02     # relative
RANK_ACC_GAIN = 0.015   # rank64 must beat baseline dev accuracy by 1.5 pts ...
RANK_LOSS_GAIN = 0.03   # ... or lower dev loss by 3% (without losing accuracy)
SLICE_TOL = 0.02        # and no image / text dev slice may drop by more than 2.0 pts (slices are smaller: noisier)
TAIL = 2


def lane_metrics(entries: list[dict], tail: int = TAIL) -> dict | None:
    """Mean dev metrics over the last `tail` dev checkpoints after step 0 (the log's dev entries)."""
    dev = [e for e in entries if "dev_loss" in e and e.get("step", 0) > 0]
    if not dev:
        return None
    last = dev[-tail:]
    mean = lambda k: sum(e[k] for e in last) / len(last) if all(k in e and e[k] is not None for e in last) else None
    slices = {}
    for name in sorted({k for e in last for k in (e.get("dev_slices") or {})}):
        vals = [(e.get("dev_slices") or {}).get(name, {}).get("accuracy") for e in last]
        if all(v is not None for v in vals):
            slices[name] = sum(vals) / len(vals)
    return {"dev_loss": mean("dev_loss"), "dev_accuracy": mean("dev_accuracy"), "dev2_accuracy": mean("dev2_accuracy"),
            "dev_slices": slices or None,
            "steps": [e["step"] for e in last], "last_step": dev[-1]["step"],
            "ordinal_loss_last": next((e.get("ordinal_loss") for e in reversed(entries) if e.get("ordinal_loss") is not None), None)}


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def no_harm(m: dict | None, base: dict) -> bool:
    return bool(m and m["dev_accuracy"] is not None and m["dev_loss"] is not None
                and m["dev_accuracy"] >= base["dev_accuracy"] - DEV_ACC_TOL and m["dev_loss"] <= base["dev_loss"] * (1 + DEV_LOSS_REL))


def better_dev2(m: dict | None, base: dict) -> bool:
    return bool(m and m.get("dev2_accuracy") is not None and base.get("dev2_accuracy") is not None and m["dev2_accuracy"] > base["dev2_accuracy"])


def mean_acc(m: dict) -> float:
    return (m["dev_accuracy"] + (m["dev2_accuracy"] if m.get("dev2_accuracy") is not None else m["dev_accuracy"])) / 2


def slices_no_harm(m: dict | None, base: dict) -> tuple[bool, str]:
    ms, bs = (m or {}).get("dev_slices") or {}, base.get("dev_slices") or {}
    common = sorted(set(ms) & set(bs))
    bad = [f"{k} {ms[k]:.3f} < {bs[k]:.3f} - {SLICE_TOL}" for k in common if ms[k] < bs[k] - SLICE_TOL]
    return not bad, ("; ".join(bad) if bad else ("slices " + ", ".join(f"{k} {ms[k]:.3f} vs {bs[k]:.3f}" for k in common) if common
                                                    else "no dev slices logged"))


def rank_better(m: dict | None, base: dict) -> tuple[bool, str]:
    """rank64 over r16: no harm, no slice regression, and clearly better beyond noise (else keep r16)."""
    if not no_harm(m, base):
        return False, "harm (or missing)"
    ok_s, why_s = slices_no_harm(m, base)
    if not ok_s:
        return False, f"slice regression: {why_s}"
    acc_gain, loss_rel = m["dev_accuracy"] - base["dev_accuracy"], m["dev_loss"] / base["dev_loss"] - 1
    clear = (acc_gain >= RANK_ACC_GAIN and loss_rel <= 0) or (loss_rel <= -RANK_LOSS_GAIN and acc_gain >= 0)
    return clear, f"dev acc {acc_gain:+.3f}, dev loss {loss_rel:+.1%} vs baseline; {why_s}" + ("" if clear else " (not beyond noise: keep r16)")


def decide(metrics: dict, parity_ok: bool, has_255_rows: bool) -> dict:
    """metrics: lane -> lane_metrics(...) or None. Returns the winners (combined), the lane to continue (or None) and the reasons."""
    reasons, base = [], metrics.get("baseline")
    out = {"ordinal_weight": 0.0, "readout_codes": 255, "expand_lora_rank": 0, "lora_rank": 16, "lanes": metrics,
           "parity_ok": parity_ok, "has_255_rows": has_255_rows,
           "rules": {"dev_acc_tol": DEV_ACC_TOL, "dev_loss_rel": DEV_LOSS_REL, "tail": TAIL, "rank_acc_gain": RANK_ACC_GAIN,
                     "rank_loss_gain": RANK_LOSS_GAIN, "slice_tol": SLICE_TOL}}
    if not base or base.get("dev_accuracy") is None or base.get("dev_loss") is None:
        out["reasons"] = ["baseline lane missing or without dev metrics: keep W = 0, 255 codes, rank 16"]
        out.update(lane="baseline", combined=False, winners=[])
        return out
    o, c, r = metrics.get("ordinal"), metrics.get("codes256"), metrics.get("rank64")
    ord_ok = no_harm(o, base) and better_dev2(o, base)
    reasons.append(f"ordinal: {'WIN' if ord_ok else 'lose'} (no_harm {no_harm(o, base)}, score dev "
                   f"{o and o.get('dev2_accuracy')} vs baseline {base.get('dev2_accuracy')})")
    if not parity_ok:
        codes_ok = False; reasons.append("codes256: lose (CUDA parity check failed or did not run: 256 lanes disabled)")
    elif not has_255_rows:
        codes_ok = False; reasons.append("codes256: lose (no 255-option training rows: row 256 would stay untrained)")
    else:
        codes_ok = no_harm(c, base); reasons.append(f"codes256: {'WIN' if codes_ok else 'lose'} (no_harm {codes_ok})")
    rank_ok, why = rank_better(r, base)
    reasons.append(f"rank64: {'WIN' if rank_ok else 'lose'} ({why})")
    winners = [k for k, ok in (("ordinal", ord_ok), ("codes256", codes_ok), ("rank64", rank_ok)) if ok]
    for k in winners:
        for key, val in LANES[k].items():
            if val != LANES["baseline"][key]:
                out[key] = val
    out["lora_rank"] = out["expand_lora_rank"] or 16
    out["winners"] = winners
    out["lane"] = lane_of(out)
    out["combined"] = out["lane"] is None
    if out["combined"]:
        reasons.append(f"combined winners {winners}: no single lane trained them together; the full run restarts from the shipped "
                       "adapter with the combined flags (no lane resume)")
    out["reasons"] = reasons
    return out


def lane_of(decision: dict) -> str | None:
    """The pilot lane that trained exactly the winning switches (the full run continues it; cloud/p3/lane_resume.py), else None."""
    want = {k: decision.get(k, LANES["baseline"][k]) for k in LANES["baseline"]}
    return next((k for k, v in LANES.items() if v == want), None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", type=Path, required=True, help="<lane>/train/log.jsonl per lane")
    ap.add_argument("--parity", type=Path, help="parity summary.json (scripts/p3/parity_readout256.py --backend torch)")
    ap.add_argument("--manifest-info", type=Path, help="cloud/p3/prep_manifests.py output (has_255_rows)")
    ap.add_argument("--steps", type=int, default=0, help="a lane whose last dev pass is before this step (crashed) counts as missing")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    metrics = {lane: lane_metrics(read_log(a.pilot_dir / lane / "train" / "log.jsonl")) for lane in LANES}
    metrics = {k: (m if m and m["last_step"] >= a.steps else None) for k, m in metrics.items()}
    parity_ok = False
    if a.parity and a.parity.exists():
        p = json.loads(a.parity.read_text())["parity"]; parity_ok = p["bit_identical_logits"] == p["questions"] > 0
    has_255 = bool(a.manifest_info and a.manifest_info.exists() and json.loads(a.manifest_info.read_text()).get("rows_255", 0) > 0)
    d = decide(metrics, parity_ok, has_255)
    a.out.write_text(json.dumps(d, indent=1) + "\n")
    print(f"pilot decision: ordinal_weight {d['ordinal_weight']} readout_codes {d['readout_codes']} lora rank {d['lora_rank']} "
          f"(lane {d['lane'] or 'none: combined winners, restart from the shipped adapter'})")
    for r in d["reasons"]:
        print("  " + r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
