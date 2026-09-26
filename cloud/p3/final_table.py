"""Phase-3 final comparison table (for a PERSON; the ship decision is read here, not by any code).

Reads every checkpoint's summary.json / selection.json / gates.json under --eval-root, the pick (pick.json) and, for the
DecisionBench ship rule, the tracking-only scores under --tracking-root. This script is not the selector: it runs after the
pick is written and never feeds back into it. Writes <out>.md and <out>.json with the plan's target rows (docs/phase-3-plan.md
"Targets") for the picked checkpoint against the shipped 1.0 measured the same way on the same pod.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIRECT_GROUPS = ("charts", "docimg", "inventory", "safety", "geometry", "screens")


def rj(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def g(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def row(eval_root: Path, tracking_root: Path, name: str) -> dict:
    s = rj(eval_root / name / "summary.json") or {}
    sel = rj(eval_root / name / "selection.json") or {}
    gates = rj(eval_root / name / "gates.json") or {}
    db = rj(tracking_root / name / "decisionbench.json") or {}
    jb = s.get("jevbench") or {}
    return {"name": name, "fresh": g(sel, "fresh", "acc"), "flagged": g(sel, "flagged", "acc"), "human": g(sel, "human", "acc"),
            "hard_single_raw": g(jb, "single_raw", "hard", "acc"), "hard_single_cal": g(jb, "single_cal", "hard", "acc"),
            "hard_ece_single_cal": g(jb, "single_cal", "hard", "ece"), "pooled_ece_single_cal": g(jb, "single_cal", "pooled_ece"),
            "hard_rot4_raw": g(jb, "rot4_raw", "hard", "acc"), "hard_rot4_cal": g(jb, "rot4_cal", "hard", "acc"),
            "hard_ece_rot4_cal": g(jb, "rot4_cal", "hard", "ece"), "pooled_ece_rot4_cal": g(jb, "rot4_cal", "pooled_ece"),
            "calibration": g(s, "calibration", "single", "choice"), "calibration_rot4": g(s, "calibration", "rot4", "choice"),
            "gates": gates.get("shippable"), "soup": gates.get("soup", False), "joint": g(gates, "targets", "joint_items", "value"),
            "imajevbench": s.get("imajevbench"), "fastdec_macro": g(s, "fastdec", "macro"),
            "db_full_equiv": db.get("full_suite_equivalent"), "db_primary": db.get("primary_accuracy"), "db_ordinal": db.get("ordinal"),
            "db_reasoning": db.get("reasoning"),
            **{grp: g(s, "direct_panels", grp, "acc") for grp in DIRECT_GROUPS}}


def delta(a, b):
    return None if a is None or b is None else a - b


def targets(p: dict, base: dict) -> list[dict]:
    t = []
    def add(name, value, ok, rule):
        t.append({"target": name, "value": value, "met": bool(ok) if ok is not None else None, "rule": rule})
    add("JevBench public hard, single pass + cal", p["hard_single_cal"], p["hard_single_cal"] is not None and p["hard_single_cal"] >= 73, ">= 73 (rot4 measured as fallback)")
    add("JevBench public hard, rot4 + cal (fallback)", p["hard_rot4_cal"], p["hard_rot4_cal"] is not None and p["hard_rot4_cal"] >= 73, ">= 73")
    d = delta(p["fresh"], base["fresh"]); add("Held-out fresh-seed vs shipped", d, d is not None and d >= 8, "+8 pts")
    d = delta(p["db_full_equiv"], base["db_full_equiv"]); add("DecisionBench 3k (full-suite equivalent) vs shipped", d, d is not None and d >= 2, "+2 pts (tracking-only; read by a person)")
    d = delta(p["db_ordinal"], base["db_ordinal"]); add("DecisionBench ordinal vs shipped", d, d is not None and d >= 5, "+5 pts")
    d = delta(p["db_reasoning"], base["db_reasoning"]); add("DecisionBench reasoning vs shipped", d, d is not None and d >= 5, "+5 pts")
    add("All gates PASS without a soup", p["gates"] and not p["soup"], bool(p["gates"]) and not p["soup"], "phase-2c rules")
    add("ImajevBench joint track", p["joint"], p["joint"] is not None and p["joint"] >= 99, ">= 99/122")
    add("Hard ECE after calibration (single)", p["hard_ece_single_cal"], p["hard_ece_single_cal"] is not None and p["hard_ece_single_cal"] <= 0.08, "<= 0.08")
    add("Pooled public ECE (single, cal)", p["pooled_ece_single_cal"], p["pooled_ece_single_cal"] is not None and p["pooled_ece_single_cal"] <= 0.03, "<= 0.03")
    return t


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", type=Path, required=True); ap.add_argument("--tracking-root", type=Path, required=True)
    ap.add_argument("--pick", type=Path, required=True); ap.add_argument("--shipped", default="shipped")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    pick = rj(a.pick) or {}
    names = sorted(d.name for d in a.eval_root.iterdir() if d.is_dir() and (d / "selection.json").exists())
    rows = {n: row(a.eval_root, a.tracking_root, n) for n in names}
    base = rows.get(a.shipped) or row(a.eval_root, a.tracking_root, a.shipped)
    picked = rows.get(pick.get("pick")) if pick.get("pick") else None
    res = {"pick": pick.get("pick"), "pick_reason": pick.get("reason"), "shipped": base, "rows": rows,
           "targets": targets(picked, base) if picked else None}
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    cols = [("fresh", 1), ("human", 1), ("hard_single_raw", 1), ("hard_single_cal", 1), ("hard_ece_single_cal", 3), ("pooled_ece_single_cal", 3),
            ("hard_rot4_cal", 1), ("joint", 0), ("gates", 0), ("fastdec_macro", 1), ("db_full_equiv", 1), ("db_ordinal", 1), ("db_reasoning", 1)]
    # direct constructed image groups' held-out panels (charts, docimg, inventory, safety, geometry, screens): report only
    cols += [(grp, 1) for grp in DIRECT_GROUPS if any(r.get(grp) is not None for r in rows.values())]
    lines = [f"# Phase 3 final comparison (pick: {res['pick']})", "", pick.get("reason") or "no pick (see pick.json)", "",
             "| checkpoint | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for n in [a.shipped] + [x for x in names if x != a.shipped]:
        r = rows.get(n)
        if r:
            lines.append(f"| {'**' + n + '**' if n == res['pick'] else n} | " + " | ".join(fmt(r[c], nd) for c, nd in cols) + " |")
    if res["targets"]:
        lines += ["", "## Targets (picked vs shipped 1.0, same pod, same protocol)", "", "| target | value | rule | met |", "|---|---:|---|---|"]
        for t in res["targets"]:
            lines.append(f"| {t['target']} | {fmt(t['value'], 3 if 'ECE' in t['target'] else 1)} | {t['rule']} | {'yes' if t['met'] else 'NO'} |")
    lines += ["", "DecisionBench columns are tracking-only (never read by the selector). Ship only if every target rule holds; otherwise keep 1.0."]
    a.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
