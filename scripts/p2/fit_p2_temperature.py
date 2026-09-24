"""Fit ONE temperature for a phase-2b adapter by held-out NLL on the calibration rows of decision-p2b.

Rows come from `evaluate_decision_model_torch.py --version decision-p2b --partition calibration` (it writes `logits`,
`target_index`, `decision_type`, `option_count` per row). The calibration partition of decision-p2b holds every row of
the three held-out domains (`build_p2b_manifest.py --holdout-domains`), so the model never trained on that domain and
the temperature is an off-distribution estimate — the single-temperature recipe jevk5 uses, fitted on our own data.

The artifact is schema 1.0 (`TemperatureCalibrator`): the same temperature under every bucket key the server can ask
for, and the calibration row count under every key (the calibrator needs a positive count per key; the fit used the
pooled rows, which the `fit` block records).

Usage: python scripts/p2/fit_p2_temperature.py --predictions <eval dir>/predictions.jsonl --manifest data/manifests/decision-p2b.jsonl \
           --holdout-domains telecom,hospitality,nonprofit_grants --version p2b-4b --out <dir>/calibration-p2b-4b.json
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_decision.calibration import BUCKETS, TemperatureCalibrator, _nll, fit_temperature  # noqa: E402

BUCKET_KEYS = [f"{t}:{str(lo) if lo == hi else f'{lo}-{hi}'}" for t in ("boolean", "choice", "ordinal") for lo, hi in BUCKETS]


def select_rows(predictions, manifest=None, holdout_domains=()):
    """Rows whose record (joined by id, `:decision` suffix tolerated) is in a held-out domain; every row when no domains are given."""
    domains = {}
    if manifest is not None:
        for line in Path(manifest).open():
            if line.strip():
                r = json.loads(line); domains[r["id"]] = r.get("domain")
    out = []
    for row in predictions:
        if holdout_domains:
            rid = row["id"]; dom = domains.get(rid) or domains.get(rid.rsplit(":", 1)[0])
            if dom not in holdout_domains: continue
        out.append(row)
    return out


def fit_single_temperature(rows, version):
    samples = []
    for row in rows:
        if len(row["logits"]) != row["option_count"] + 1: raise ValueError(f"{row['id']}: option_count must equal len(logits) - 1")
        samples.append(([float(x) for x in row["logits"]], int(row["target_index"])))
    if not samples: raise ValueError("no calibration rows")
    t = fit_temperature(samples)
    import math
    before, after = _nll(0.0, samples), _nll(math.log(t), samples)
    cal = TemperatureCalibrator(version, {k: t for k in BUCKET_KEYS}, {k: len(samples) for k in BUCKET_KEYS}, None)
    payload = cal.to_dict(); payload["fit"] = {"rule": "single temperature by held-out NLL on pooled rows", "rows": len(samples), "temperature": t,
                                                 "nll_raw": before, "nll_calibrated": after, "decision_types": sorted({r["decision_type"] for r in rows})}
    return payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--predictions", type=Path, nargs="+", required=True); ap.add_argument("--manifest", type=Path)
    ap.add_argument("--holdout-domains", default="", help="comma list; rows outside these domains are ignored (default: use every row)")
    ap.add_argument("--version", required=True); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for p in args.predictions for l in Path(p).open() if l.strip()]
    doms = tuple(d for d in args.holdout_domains.split(",") if d)
    rows = select_rows(rows, args.manifest, doms)
    payload = fit_single_temperature(rows, args.version); payload["fit"]["holdout_domains"] = list(doms)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=1))
    TemperatureCalibrator.load(args.out)  # round-trip check
    print(f"{args.version}: T={payload['fit']['temperature']:.3f} on {payload['fit']['rows']} rows (nll {payload['fit']['nll_raw']:.4f} -> {payload['fit']['nll_calibrated']:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
