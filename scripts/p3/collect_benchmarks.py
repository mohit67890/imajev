#!/usr/bin/env python3
"""Collect and verify every benchmark result of a phase-3 run for release.

Reads <root>/run/eval/<ckpt>/{summary,gates,selection}.json, panels/*/predictions.jsonl, jevbench/*/result.json, imajevbench/score,
fastdec/result.json and <root>/tracking-only/<ckpt>/decisionbench.json; checks each panel's row count against the expected size;
writes <out>/benchmarks.json (everything, per checkpoint) and <out>/benchmarks.md (release tables) and lists anything missing.
    python scripts/p3/collect_benchmarks.py --root p3 --out p3/release   (pod)
    python scripts/p3/collect_benchmarks.py --root reports/phase3/train-results/p3 --out reports/phase3/train-results  (Mac, after extract)
"""
from __future__ import annotations
import argparse, json, os, glob
from pathlib import Path

EXPECT = {"heldout_fresh": 4297, "heldout_flagged": 2813, "human_dev": 785, "heldout_charts": 300, "heldout_docimg": 300, "heldout_inventory": 250,
          "heldout_safety": 250, "heldout_geometry": 250, "heldout_screens": 250, "p2b_test": 435, "state_probe": 200, "pairs_probe": 60,
          "irrelevance": 2823, "authored_dev": 150}
JEV = {"easy": 48, "original": 72, "hard": 111}; JEV_VARIANTS = ("single_raw", "single_cal", "rot4_raw", "rot4_cal")
IMAJEV_ROWS = 279; FASTDEC_ROWS = 1700; DB_ROWS = 3000

def jread(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def nrows(p):
    p = Path(p)
    return sum(1 for l in open(p) if l.strip()) if p.exists() else 0

def collect(root: Path) -> tuple[dict, list[str]]:
    out, missing = {}, []
    names = sorted(os.path.basename(d) for d in glob.glob(str(root / "run/eval/*")) if os.path.isdir(d))
    order = ["shipped"] + [n for n in names if n.startswith("r1")] + [n for n in names if n.startswith("r2")] + [n for n in names if n.startswith("soup")]
    for n in [x for x in order if x in names]:
        d = root / "run/eval" / n; s = jread(d / "summary.json"); g = jread(d / "gates.json"); sel = jread(d / "selection.json")
        if n.startswith("soup"):   # evaluations the owner stopped: recorded as not evaluated, never as numbers
            out[n] = {"status": "not evaluated (soup fallback stopped by the owner)", "summary_present": bool(s)}; continue
        rec = {"status": "ok" if s and g and sel else "incomplete", "selection": sel and {k: sel[k] for k in ("fresh", "flagged", "human") if k in sel},
               "panels": {}, "jevbench": {}, "gates": g and {"shippable": g["shippable"], "image_gate": g["image_gate"],
               "checks": [{k: c.get(k) for k in ("gate", "value", "op", "bound", "pass", "note")} for c in g["checks"]]}}
        if not s: missing.append(f"{n}: summary.json"); out[n] = rec; continue
        for key, exp in EXPECT.items():
            cnt = nrows(d / "panels" / key / "predictions.jsonl")
            acc = None
            if cnt:
                rows = [json.loads(l) for l in open(d / "panels" / key / "predictions.jsonl") if l.strip()]
                acc = 100.0 * sum(bool(r.get("correct")) for r in rows) / len(rows)
            rec["panels"][key] = {"acc": acc, "n": cnt, "expected": exp, "complete": cnt == exp}
            if cnt != exp: missing.append(f"{n}: panel {key} has {cnt} rows, expected {exp}")
        for v in JEV_VARIANTS:
            jv = (s.get("jevbench") or {}).get(v)
            if not jv: missing.append(f"{n}: jevbench {v}"); continue
            rec["jevbench"][v] = {t: {"acc": jv[t]["acc"], "ece": jv[t]["ece"], "n": jv[t].get("n")} for t in JEV if t in jv}
            rec["jevbench"][v]["pooled_ece"] = jv.get("pooled_ece")
            for t, exp in JEV.items():
                if t not in jv or jv[t].get("n") != exp: missing.append(f"{n}: jevbench {v} {t} n={jv.get(t, {}).get('n')} expected {exp}")
        tr = s.get("imajevbench"); ib_rows = nrows(d / "imajevbench/predictions.jsonl")
        rec["imajevbench"] = tr and {"acc": 100.0 * sum(v[0] for v in tr.values()) / sum(v[1] for v in tr.values()), "tracks": tr, "predictions": ib_rows}
        if not tr or ib_rows != IMAJEV_ROWS: missing.append(f"{n}: imajevbench (tracks {bool(tr)}, predictions {ib_rows}/{IMAJEV_ROWS})")
        fd = s.get("fastdec"); rec["fastdec"] = fd and {"macro": fd["macro"], "pooled_heads": fd["pooled_heads"], "rows": fd["rows"], "by_domain": fd.get("by_domain")}
        if not fd or fd.get("rows") != FASTDEC_ROWS: missing.append(f"{n}: fast-decisions ({fd and fd.get('rows')}/{FASTDEC_ROWS} rows)")
        db = jread(root / "tracking-only" / n / "decisionbench.json")
        rec["decisionbench_tracking_only"] = db and {k: db.get(k) for k in ("rows", "primary_accuracy", "macro_task_accuracy", "full_suite_equivalent", "ordinal", "reasoning", "errors", "unsupported", "by_primitive", "by_family")}
        if not db or db.get("rows") != DB_ROWS: missing.append(f"{n}: decisionbench 3k ({db and db.get('rows')}/{DB_ROWS} rows)")
        rec["direct_panels"] = s.get("direct_panels"); rec["calibration"] = {k: {"choice": v.get("choice"), "why": v.get("why")} for k, v in (s.get("calibration") or {}).items()}
        rec["calibration_files"] = sorted(os.path.basename(p) for p in glob.glob(str(d / "calibration/*.json")))
        out[n] = rec
    return out, missing

def md(out: dict, extra: dict | None) -> str:
    f = lambda v, d=1: "·" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
    names = [n for n, r in out.items() if r.get("status") == "ok"]
    L = ["# Phase-3 benchmark results (all checkpoints, same pod, same protocol)", "",
         "Every number below was produced by `scripts/p3/eval_checkpoint.py` on pod the phase-3 pod (8xH100) on 2026-09-26 and verified for completeness by",
         "`scripts/p3/collect_benchmarks.py` (row counts per panel). Soup rows are absent: the soup fallback was stopped by the owner and never evaluated.", "",
         "## External benchmarks", "",
         "| checkpoint | ImajevBench acc | visual | joint | text | JevBench easy | original | hard (single raw) | hard (single cal) | hard ECE cal | pooled ECE cal | hard rot4 raw | hard rot4 cal | fast-decisions macro | DecisionBench 3k acc | full-suite equiv | ordinal | reasoning |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        r = out[n]; ib = r["imajevbench"] or {}; tr = ib.get("tracks") or {}; j = r["jevbench"]; db = r["decisionbench_tracking_only"] or {}
        g = lambda v, t: (j.get(v) or {}).get(t, {})
        L.append(f"| {n} | {f(ib.get('acc'))} | {tr.get('visual', ['·'])[0]}/120 | {tr.get('joint', ['·'])[0]}/122 | {tr.get('text', ['·'])[0]}/37 | {f(g('single_raw','easy').get('acc'))} | {f(g('single_raw','original').get('acc'))} | "
                 f"{f(g('single_raw','hard').get('acc'))} | {f(g('single_cal','hard').get('acc'))} | {f(g('single_cal','hard').get('ece'),3)} | {f((j.get('single_cal') or {}).get('pooled_ece'),3)} | "
                 f"{f(g('rot4_raw','hard').get('acc'))} | {f(g('rot4_cal','hard').get('acc'))} | {f((r['fastdec'] or {}).get('macro'))} | {f(db.get('primary_accuracy'))} | {f(db.get('full_suite_equivalent'))} | {f(db.get('ordinal'))} | {f(db.get('reasoning'))} |")
    L += ["", "n: ImajevBench 279 items (visual 120, joint 122, text 37); JevBench public 231 (easy 48, original 72, hard 111); fast-decisions dev 1,700 rows / 2,900 heads; DecisionBench 3k stratified subset (tracking only, never used for selection or training).", "",
          "## Our held-out panels", "",
          "| checkpoint | fresh (4,297) | flagged (2,813) | human (785) | authored dev (150) | charts (300) | docimg (300) | inventory (250) | safety (250) | geometry (250) | screens (250) | p2b test (435) | state probe (200) | pairs probe (60) | irrelevance (2,823) |",
          "|---|" + "---:|" * 14]
    for n in names:
        p = out[n]["panels"]; a = lambda k: f(p.get(k, {}).get("acc"))
        L.append(f"| {n} | {a('heldout_fresh')} | {a('heldout_flagged')} | {a('human_dev')} | {a('authored_dev')} | {a('heldout_charts')} | {a('heldout_docimg')} | {a('heldout_inventory')} | {a('heldout_safety')} | {a('heldout_geometry')} | {a('heldout_screens')} | {a('p2b_test')} | {a('state_probe')} | {a('pairs_probe')} | {a('irrelevance')} |")
    L += ["", "## Gates (phase-2c rules, 4b bounds)", "", "| checkpoint | shippable | image gate | failing checks |", "|---|---|---|---|"]
    for n in names:
        g = out[n]["gates"]; fails = "; ".join(f"{c['gate']} {c['value']} {c.get('op','')} {c['bound']}" for c in g["checks"] if not c["pass"]) or "none"
        L.append(f"| {n} | {g['shippable']} | {g['image_gate']} | {fails} |")
    if extra and extra.get("s1bench"):
        L += ["", "## Typed S1-Bench (our conversion of the 220 English items; 212 kept; NOT an S1-Bench score)", "", "| checkpoint | acc | abstain % | mean conf | ECE | reasoning | knowledge | analysis | instruction following |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n, s in extra["s1bench"].items():
            c = s["by_category"]; L.append(f"| {n} | {s['acc']:.1f} | {s['abstain_rate']:.1f} | {s['mean_conf']:.1f} | {s['ece']:.3f} | {c.get('reasoning_question',0):.1f} | {c.get('knowledge_question',0):.1f} | {c.get('analysis_question',0):.1f} | {c.get('instruction_following',0):.1f} |")
    return "\n".join(L) + "\n"

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--s1bench", type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    out, missing = collect(a.root)
    extra = {"s1bench": json.loads(a.s1bench.read_text())} if a.s1bench and a.s1bench.exists() else None
    (a.out / "benchmarks.json").write_text(json.dumps({"checkpoints": out, "missing": missing, "s1bench_typed": extra and extra["s1bench"]}, indent=1))
    (a.out / "benchmarks.md").write_text(md(out, extra))
    print(f"checkpoints: {[n for n, r in out.items() if r.get('status') == 'ok']}")
    print("MISSING:" if missing else "COMPLETE: every panel, benchmark and tracking file present with the expected row counts")
    for m in missing: print("  ", m)
    return 1 if missing else 0

if __name__ == "__main__":
    raise SystemExit(main())
