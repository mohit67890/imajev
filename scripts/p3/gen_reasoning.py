#!/usr/bin/env python3
"""Phase-3 source A, REASONING families: programmatic generators with a difficulty knob and constructed, provably correct gold
(docs/phase-3-plan.md, Stage 0 source A; "Everything in this phase" items 1 and 13).

    .venv/bin/python scripts/p3/gen_reasoning.py --seed r1 --count 30000 --out data/p3/candidates/A-reasoning.jsonl
    .venv/bin/python scripts/p3/gen_reasoning.py --variant-of p3-ar-tab_fx-d4-r1-000123 --variant-count 2 --out variants.jsonl

Families (module in scripts/p3/gen_reasoning_families/):
  table_arithmetic   table.py      statements/tables: FX batches, weighted averages, balance-sheet ratios, budget variance,
                                   inventory roll-forward, SaaS MRR bridge, break-even (2-6 ops, mistake-based distractors)
  temporal_numeric   table.py      income-statement growth / margins, CAGR, compounding;
                     temporal.py   business-day deadlines, leave counts, time zones, SLA clocks, shift pay, unit conversion,
                                   dated FX, tenure eligibility, recurring schedules
  logic              logic.py      quantified rules with negation, exceptions, disjunction and distractors (depth 2-6)
  multi_hop          multihop.py   synthetic fictional knowledge base, 2-5 hops, former-value and wrong-turn distractors
  probability        probability.py exact probability / expected value in our own phrasing
  constraint         constraint.py ordering / seating / ranking puzzles, two-attribute grids, budgeted allocation (unique solution)
  ordinal            ordinal.py    score questions (3-10 levels) from explicit rubric rules (~20% of the output)

Every family also emits UNKNOWN variants (~15%): the decisive fact removed, gold=null, unknown_reason insufficient_evidence,
parent_id = the answerable item; logic also has natural "cannot be determined" items (gold=null, no parent). Difficulty is
3-5 only (weights 0.2 / 0.4 / 0.4). provenance.spec holds the formal inputs so tests can re-solve the gold independently.
Deterministic by --seed.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate import validate, write  # noqa: E402
from gen_reasoning_families import constraint, logic, multihop, ordinal, probability, table, temporal  # noqa: E402
from gen_reasoning_families.common import GENERATOR, Skip, rng_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODULES = {"table": table, "temporal": temporal, "probability": probability, "logic": logic, "multihop": multihop,
           "constraint": constraint, "ordinal": ordinal}
KINDS: dict[str, tuple] = {}  # kind -> (fn, family, module name)
for mname, mod in MODULES.items():
    for k, fn in mod.KINDS.items():
        KINDS[k] = (fn, mod.KIND_FAMILY.get(k) or getattr(mod, "FAM", "table_arithmetic"), mname)
for k, (fn, fam, m) in list(KINDS.items()):
    if m == "table" and k not in table.KIND_FAMILY:
        KINDS[k] = (fn, "table_arithmetic", m)
FAMILIES = ["table_arithmetic", "temporal_numeric", "logic", "multi_hop", "probability", "constraint", "ordinal"]
FAMILY_KINDS = {f: sorted(k for k, v in KINDS.items() if v[1] == f) for f in FAMILIES}
DIFF_W = {3: 0.2, 4: 0.4, 5: 0.4}
UNKNOWN_SHARE = 0.15
ORDINAL_SHARE = 0.20
ID_RE = re.compile(r"^p3-ar-(?P<kind>[a-z0-9_]+?)-d(?P<d>[1-5])-(?P<seed>.+)-(?P<i>\d{6})(?P<suffix>(?:-u|-v\d+)*)$")


def targets(count: int) -> dict[str, int]:
    o = round(count * ORDINAL_SHARE)
    rest = count - o
    others = [f for f in FAMILIES if f != "ordinal"]
    t = {f: rest // len(others) for f in others}
    for f in others[: rest - sum(t.values())]:
        t[f] += 1
    t["ordinal"] = o
    return t


def make_row(it: dict, kind: str, d: int, seed: str, rid: str, parent: str | None = None, unknown: bool = False) -> dict:
    fn, fam, mname = KINDS[kind]
    if unknown:
        u = it["unk"]
        state, field, gold, reason, spec = u["state"], u.get("field", it["field"]), None, u["reason"], u["spec"]
    else:
        state, field, gold, reason, spec = it["state"], it["field"], it["gold"], it["unknown_reason"], it["spec"]
    return {"id": rid, "source": "A", "dataset": f"reasoning_{mname}", "family": it["family"], "difficulty": d, "state": state, "images": [],
            "field": field, "gold": gold, "unknown_reason": reason, "gold_kind": "constructed", "parent_id": parent,
            "provenance": {"licence": "generated", "generator": GENERATOR, "seed": seed, "kind": kind, "spec": spec}}


def gen_one(kind: str, d: int, *seed_parts, tries: int = 30, need_unk: bool = False) -> dict:
    fn = KINDS[kind][0]
    last = None
    for a in range(tries):
        try:
            it = fn(rng_for(*seed_parts, a), d)
        except Skip as e:
            last = e
            continue
        if need_unk and not it.get("unk"):
            continue
        return it
    raise Skip(f"{kind} d{d}: no item after {tries} tries ({last})")


def gen_family(args) -> list[dict]:
    fam, n, seed = args
    kinds = list(FAMILY_KINDS[fam])
    rng_order = rng_for(seed, fam, "order")
    rng_order.shuffle(kinds)
    rows, n_unk, i = [], 0, 0
    while len(rows) < n:
        kind = kinds[i % len(kinds)]
        r = rng_for(seed, fam, i, "difficulty")
        d = r.choices(list(DIFF_W), weights=list(DIFF_W.values()))[0]
        rid = f"p3-ar-{kind}-d{d}-{seed}-{i:06d}"
        i += 1
        try:
            it = gen_one(kind, d, seed, kind, d, i)
        except Skip:
            continue
        row = make_row(it, kind, d, seed, rid)
        rows.append(row)
        if row["gold"] is None:
            n_unk += 1
        if it.get("unk") and len(rows) < n and (n_unk + 1) / (len(rows) + 1) <= UNKNOWN_SHARE + 0.005:
            rows.append(make_row(it, kind, d, seed, rid + "-u", parent=rid, unknown=True))
            n_unk += 1
    return rows


def generate(seed: str, count: int, families=None, workers: int = 7) -> list[dict]:
    t = targets(count)
    fams = [f for f in FAMILIES if not families or f in families]
    jobs = [(f, t[f], seed) for f in fams]
    if workers > 1:
        with mp.Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(gen_family, jobs)
    else:
        parts = [gen_family(j) for j in jobs]
    return [r for p in parts for r in p]


def variants_of(rid: str, count: int = 2) -> list[dict]:
    """Fresh same-kind, same-difficulty items for a parent id (Stage 2 variants). An unknown parent (-u) yields unknown variants."""
    m = ID_RE.match(rid)
    if not m:
        raise SystemExit(f"not a gen_reasoning id: {rid}")
    kind, d, seed = m["kind"], int(m["d"]), m["seed"]
    if kind not in KINDS:
        raise SystemExit(f"unknown kind {kind}")
    want_unk = m["suffix"].endswith("-u")
    out = []
    for k in range(1, count + 1):
        it = gen_one(kind, d, rid, "variant", k, tries=60, need_unk=want_unk)
        vid = f"{rid}-v{k}"
        if want_unk:
            out.append(make_row(it, kind, d, seed, vid, parent=rid, unknown=True))
        else:
            out.append(make_row(it, kind, d, seed, vid, parent=rid))
    return out


def summary(rows: list[dict]) -> dict:
    s = {"n": len(rows), "by_family": collections.Counter(r["family"] for r in rows),
         "by_family_difficulty": collections.Counter((r["family"], r["difficulty"]) for r in rows),
         "by_family_type": collections.Counter((r["family"], r["field"]["type"]) for r in rows),
         "unknown_by_family": collections.Counter(r["family"] for r in rows if r["gold"] is None),
         "by_kind": collections.Counter(r["provenance"]["kind"] for r in rows)}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default="r1")
    ap.add_argument("--count", type=int, default=30000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--variant-of", default=None, help="parent id: write fresh same-family same-difficulty variants")
    ap.add_argument("--variant-count", type=int, default=2)
    a = ap.parse_args()
    if a.variant_of:
        rows = variants_of(a.variant_of, a.variant_count)
    else:
        rows = generate(a.seed, a.count, a.families, a.workers)
    if a.out:
        n = write(a.out, rows)
        print(f"wrote {n} rows to {a.out}", file=sys.stderr)
    else:
        for r in rows:
            errs = validate(r)
            if errs:
                raise ValueError(f"{r['id']}: {errs}")
            print(json.dumps(r, ensure_ascii=False))
    if not a.variant_of:
        s = summary(rows)
        print(json.dumps({"n": s["n"], "by_family": dict(s["by_family"]), "unknown_by_family": dict(s["unknown_by_family"])}), file=sys.stderr)


if __name__ == "__main__":
    main()
