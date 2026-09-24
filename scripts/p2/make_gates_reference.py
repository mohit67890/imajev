"""Fill the unknown-behaviour references of cloud/p2c_gates_reference.json from the phase-2b decision-p2b test predictions.

For each size, reads <reports>/<size>/eval-best-decision-p2b-test/predictions.jsonl and computes, in percent:
  unknown_correct_rate   = share of unknown-gold rows whose prediction is unknown
  false_abstention_rate  = share of answerable rows (gold is a real option) whose prediction is unknown
plus the row counts. Every other field of the JSON is kept as is. Run on the Mac before building the phase-2c bundle:

    python scripts/p2/make_gates_reference.py            # defaults: reports/decision-p2b/pod/train-out-p2b, cloud/p2c_gates_reference.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = {None, "unknown", "__unknown__", "null"}


def is_unknown(value) -> bool:
    return value is None or (isinstance(value, str) and value in UNKNOWN)


def unknown_stats(rows: list[dict]) -> dict:
    unk = [r for r in rows if is_unknown(r.get("target"))]
    ans = [r for r in rows if not is_unknown(r.get("target"))]
    pct = lambda hits, n: round(100.0 * hits / n, 2) if n else None
    return {"unknown_correct_rate": pct(sum(is_unknown(r.get("prediction")) for r in unk), len(unk)),
            "false_abstention_rate": pct(sum(is_unknown(r.get("prediction")) for r in ans), len(ans)),
            "unknown_rows": len(unk), "answerable_rows": len(ans)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=ROOT / "reports/decision-p2b/pod/train-out-p2b")
    ap.add_argument("--reference", type=Path, default=ROOT / "cloud/p2c_gates_reference.json")
    ap.add_argument("--sizes", default="2b,4b,9b")
    a = ap.parse_args(argv)
    ref = json.loads(a.reference.read_text())
    for size in [s for s in a.sizes.split(",") if s]:
        path = a.reports / size / "eval-best-decision-p2b-test" / "predictions.jsonl"
        if not path.exists():
            print(f"{size}: MISSING {path} (left as is)"); continue
        rows = [json.loads(l) for l in path.open() if l.strip()]
        stats = unknown_stats(rows)
        ref.setdefault("sizes", {}).setdefault(size, {}).update(stats)
        print(f"{size}: {len(rows)} rows; correct-unknown {stats['unknown_correct_rate']}% of {stats['unknown_rows']} unknown-gold rows; "
              f"false abstention {stats['false_abstention_rate']}% of {stats['answerable_rows']} answerable rows")
    ref["unknown_reference_source"] = str(a.reports.relative_to(ROOT) if a.reports.is_relative_to(ROOT) else a.reports) + "/<size>/eval-best-decision-p2b-test/predictions.jsonl"
    a.reference.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {a.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
