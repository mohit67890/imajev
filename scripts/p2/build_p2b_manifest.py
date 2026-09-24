"""Build data/manifests/decision-p2b.jsonl for the phase-2b delta fine-tunes.

  train        = every new p2b row (batch matching --batch-filter, partition train, domain not held out)
                 + a deterministic --replay share of the phase-2 TRAIN rows (sha256("p2b-replay\\0" + id) % 100 < share)
  dev          = new p2b rows in partition dev (domain not held out)  -> checkpoint selection, with --dev2 the authored set
  test         = new p2b rows in partition test (domain not held out)
  calibration  = every new p2b row whose domain is in --holdout-domains (all partitions): never trained on, used only
                 by fit_p2_temperature.py, so the single temperature is fitted on domains the adapter has not seen.

Every row is contract-validated (decision_data.render) and duplicate ids are refused. Phase-2 dev/test rows are not
carried over: they are still evaluated through the decision-p2 manifest.
"""
import argparse, collections, hashlib, json, sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_HOLDOUT = "telecom,hospitality,nonprofit_grants"


def replay_selected(rid, share, seed="p2b-replay"):
    return int(hashlib.sha256(f"{seed}\0{rid}".encode()).hexdigest()[:8], 16) % 100 < int(round(share * 100))


def build(p2_rows, records, batch_filter="p2b-*", replay=0.30, holdout=(), validate=True):
    from decision_data import expand_fields, render
    out, report = [], collections.Counter()
    def ok(r):
        if not validate: return True
        try:
            for item in expand_fields(r): render(item)
            return True
        except Exception:
            report["contract_dropped"] += 1; return False
    new = [r for r in records if fnmatch(str(r.get("batch", "")), batch_filter)]
    for r in new:
        r = dict(r)
        if r.get("domain") in holdout: r["partition"] = "calibration"; report["calibration"] += 1
        else: report[f"new_{r['partition']}"] += 1
        if ok(r): out.append(r)
    for r in p2_rows:
        if r.get("partition") != "train" or not replay_selected(r["id"], replay): continue
        r = dict(r, replay=True); report["replay_train"] += 1
        if ok(r): out.append(r)
    ids = collections.Counter(r["id"] for r in out); dup = [k for k, v in ids.items() if v > 1]
    if dup: raise SystemExit(f"duplicate ids in the p2b manifest: {dup[:5]}")
    return out, dict(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", type=Path, required=True, help="phase-2 manifest (decision-p2.jsonl): its train rows are replayed")
    ap.add_argument("--records", type=Path, nargs="+", required=True, help="assembled records files holding the p2b rows (batch field)")
    ap.add_argument("--batch-filter", default="p2b-*"); ap.add_argument("--replay", type=float, default=0.30)
    ap.add_argument("--holdout-domains", default=DEFAULT_HOLDOUT); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p2 = [json.loads(l) for l in args.p2.open() if l.strip()]
    recs = [json.loads(l) for p in args.records for l in Path(p).open() if l.strip()]
    hold = tuple(d for d in args.holdout_domains.split(",") if d)
    rows, report = build(p2, recs, args.batch_filter, args.replay, hold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    parts = collections.Counter(r["partition"] for r in rows); fams = collections.Counter(r.get("family") for r in rows if r["partition"] == "train" and not r.get("replay"))
    unk = sum(r.get("target") is None for r in rows if r["partition"] == "train")
    print(json.dumps({"out": str(args.out), "rows": len(rows), "partitions": dict(parts), "report": report, "holdout_domains": list(hold),
                      "new_train_families": dict(fams), "train_unknown_share": round(unk / max(1, parts["train"]), 4)}, indent=1))


if __name__ == "__main__":
    main()
