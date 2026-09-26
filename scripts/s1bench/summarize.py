#!/usr/bin/env python3
"""Summarize typed-S1-Bench predictions (scripts/evaluate_decision_model_torch.py output) per checkpoint: accuracy overall / per
category / per field type, abstention rate, mean confidence, ECE (10 bins). Writes data/s1bench/summary.json + prints a table."""
import glob, json, collections, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; RES = REPO / "data/s1bench/results"
def ece(rows, bins=10):
    b = [[] for _ in range(bins)]
    for r in rows: b[min(bins - 1, int(r["confidence"] * bins))].append(r)
    n = len(rows); return sum(len(x) / n * abs(sum(r["correct"] for r in x) / len(x) - sum(r["confidence"] for r in x) / len(x)) for x in b if x)
out = {}
order = ["shipped"] + sorted(n for n in (os.path.basename(p)[:-len(".predictions.jsonl")] for p in glob.glob(str(RES / "*.predictions.jsonl"))) if n != "shipped")
for n in order:
    rows = [json.loads(l) for l in open(RES / f"{n}.predictions.jsonl") if l.strip()]
    cat = collections.defaultdict(list)
    for r in rows: cat[r["family"]].append(r["correct"])
    s = {"n": len(rows), "acc": 100 * sum(r["correct"] for r in rows) / len(rows), "abstain_rate": 100 * sum(r["prediction"] in (None, "unknown", "__unknown__") for r in rows) / len(rows),
         "mean_conf": 100 * sum(r["confidence"] for r in rows) / len(rows), "ece": ece(rows),
         "by_category": {k: 100 * sum(v) / len(v) for k, v in sorted(cat.items())},
         "by_type": {t: 100 * sum(r["correct"] for r in rows if r["field_type"] == t) / max(1, sum(r["field_type"] == t for r in rows)) for t in ("choice", "boolean")}}
    out[n] = s
(REPO / "data/s1bench/summary.json").write_text(json.dumps(out, indent=1))
cats = ["reasoning_question", "knowledge_question", "analysis_question", "instruction_following"]
print(f"{'ckpt':<11}{'acc':>6}{'abst':>6}{'conf':>6}{'ece':>7}  " + "".join(f"{c[:9]:>10}" for c in cats) + f"{'bool':>7}")
for n, s in out.items():
    print(f"{n:<11}{s['acc']:6.1f}{s['abstain_rate']:6.1f}{s['mean_conf']:6.1f}{s['ece']:7.3f}  " + "".join(f"{s['by_category'].get(c, 0):10.1f}" for c in cats) + f"{s['by_type']['boolean']:7.1f}")
