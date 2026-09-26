#!/usr/bin/env python3
"""Phase-3 source-A programmatic generators for policy-style decisions (docs/phase-3-plan.md rev 4, items 3, 4, 13).

Families (gold constructed by code, licence "generated", difficulty 3-5, ~15% unknown variants with parent_id):
    long_policy      1,500-2,800-word policies, contracts, SOPs, terms, schedules and incident logs + a case (~10k)
    rule_exception   a record + a policy with thresholds, exceptions and two-step conditions (~8k)
    judge_hard       rubric-based judging of a candidate response with subtle violations (~5k)
    agent_action     grid worlds, accessibility trees, inventory/tool states (+ routing_hard: which queue/team/tool) (~6k)
    long_input       4k-16k-token logs / transcripts / record lists with the decisive fact buried mid-way (~2k)
    multi_label      several labels apply; one noul item per label sharing provenance.group_id (~4k)

    .venv/bin/python scripts/p3/gen_policy.py                              # all families, default counts
    .venv/bin/python scripts/p3/gen_policy.py --families long_policy --count 200 --seed 7 --out /tmp/x.jsonl
    .venv/bin/python scripts/p3/gen_policy.py --variant-of verified.jsonl --variants-per 2 --out data/p3/candidates/A-policy-var.jsonl

Deterministic: every item draws from its own rng keyed by (seed, family, index), so a row does not depend on --count or on the
worker count. Stage-2 variants (--variant-of) keep the parent's family, kind and difficulty with a fresh rng keyed by the parent id.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate import read, validate, write  # noqa: E402
import importlib  # noqa: E402

from gen_policy_families.common import item_rng  # noqa: E402

FAMILY_NAMES = ["long_policy", "rule_exception", "judge_hard", "agent_action", "long_input", "multi_label"]


class _Modules(dict):
    """family -> module, imported on first use (so one family can be developed and run without the others)."""
    def __missing__(self, fam):
        mod = importlib.import_module(f"gen_policy_families.{fam}")
        self[fam] = mod
        return mod

    def __iter__(self):
        return iter(FAMILY_NAMES)

    def items(self):
        return [(f, self[f]) for f in FAMILY_NAMES]


MODULES = _Modules()
SHORT = {"long_policy": "lpol", "rule_exception": "rexc", "judge_hard": "judg", "agent_action": "agnt", "long_input": "long", "multi_label": "mlab"}
DEFAULT_ROWS = {"long_policy": 10000, "rule_exception": 8000, "judge_hard": 5000, "agent_action": 6000, "long_input": 2000, "multi_label": 4000}
UNKNOWN_SHARE = 0.15
DIFF_WEIGHTS = {3: 0.3, 4: 0.4, 5: 0.3}
OUT = ROOT / "data/p3/candidates/A-policy.jsonl"


def _difficulty(rng) -> int:
    return rng.choices(list(DIFF_WEIGHTS), weights=list(DIFF_WEIGHTS.values()), k=1)[0]


def _rows_from_items(items: list[dict], fam: str, base_id: str, meta: dict, parent_override: str | None = None) -> list[dict]:
    """Turn a module's item list into candidate rows. Unknown children point at their parent; everything else at parent_override."""
    ids = []
    n_unk = 0
    for j, it in enumerate(items):
        if it.get("child_of") is not None:
            n_unk += 1
            ids.append(f"{ids[it['child_of']]}-u" if n_unk == 1 else f"{ids[it['child_of']]}-u{n_unk}")
        elif j == 0:
            ids.append(base_id)
        else:
            ids.append(f"{base_id}-{j}")
    rows = []
    for j, it in enumerate(items):
        prov = {"licence": "generated", "generator": "scripts/p3/gen_policy.py", "kind": it["kind"], "domain": it["domain"],
                **meta, "spec": it["spec"]}
        if it.get("hints"):
            prov["trap_hints"] = it["hints"]
        if it.get("group"):
            prov["group_id"] = f"{base_id}-g"
        parent = ids[it["child_of"]] if it.get("child_of") is not None else parent_override
        rows.append({"id": ids[j], "source": "A", "dataset": "gen_policy", "family": it.get("family", fam),
                     "difficulty": it.get("difficulty", meta["difficulty"]), "state": it["state"], "images": [], "field": it["field"],
                     "gold": it["gold"], "unknown_reason": it["unknown_reason"], "gold_kind": "constructed", "parent_id": parent,
                     "provenance": prov})
    return rows


def _unknown_p(mod) -> float:
    # the module may return several rows per call (multi_label); it then handles unknowns itself via want_unknown
    return getattr(mod, "UNKNOWN_P", UNKNOWN_SHARE / (1 - UNKNOWN_SHARE))


def gen_one(fam: str, seed: int, i: int) -> list[dict]:
    mod = MODULES[fam]
    for attempt in range(40):
        rng = item_rng(seed, fam, i, attempt)
        difficulty = _difficulty(rng)
        kinds = mod.KINDS
        kind = kinds[(i + attempt) % len(kinds)] if attempt < 20 else rng.choice(kinds)
        want_unknown = rng.random() < _unknown_p(mod)
        try:
            items = mod.generate(rng, difficulty, kind, want_unknown)
        except (ValueError, IndexError, KeyError, ZeroDivisionError):
            continue
        base = f"p3-pol-{SHORT[fam]}-s{seed}-{i:06d}"
        rows = _rows_from_items(items, fam, base, {"seed": seed, "index": i, "difficulty": difficulty})
        bad = [(r["id"], validate(r)) for r in rows if validate(r)]
        if bad:
            raise ValueError(f"{fam} {i}: invalid rows {bad[:2]}")
        return rows
    raise RuntimeError(f"{fam} {i}: generation failed 40 times")


def _gen_chunk(args):
    fam, seed, lo, hi = args
    out = []
    for i in range(lo, hi):
        out.extend(gen_one(fam, seed, i))
    return out


def _gen_chunk_calls(args):
    fam, seed, lo, hi = args
    return [gen_one(fam, seed, i) for i in range(lo, hi)]


def generate(fams: dict[str, int], seed: int, workers: int = 0, log=print) -> list[dict]:
    rows: list[dict] = []
    for fam, target in fams.items():
        t0 = time.time()
        # generate calls 0..n-1 in chunks until the row target is reached; truncate at a call boundary (deterministic for a
        # given seed and target, independent of the worker count)
        chunk = 25
        parts: list[list[dict]] = []
        have, lo = 0, 0
        pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
        try:
            while have < target:
                per = (have / lo) if lo else 1.2
                n_more = max(chunk, int((target - have) / max(per, 0.5)))
                tasks = [(fam, seed, a, min(lo + n_more, a + chunk)) for a in range(lo, lo + n_more, chunk)]
                res = pool.map(_gen_chunk_calls, tasks) if pool else [_gen_chunk_calls(t) for t in tasks]
                for calls in res:
                    for call_rows in calls:
                        if have >= target:
                            break
                        parts.append(call_rows)
                        have += len(call_rows)
                lo += n_more
        finally:
            if pool:
                pool.close()
        n_calls = len(parts)
        fam_rows = [r for p in parts for r in p]
        rows.extend(fam_rows)
        unk = sum(r["gold"] is None for r in fam_rows)
        log(f"{fam}: {len(fam_rows)} rows ({n_calls} calls, {unk / max(1, len(fam_rows)):.1%} unknown) in {time.time() - t0:.0f}s")
    return rows


def variants_of(path: str | Path, per: int = 2) -> list[dict]:
    """Stage-2 variants: same family, kind and difficulty; new rng keyed by the parent id; parent_id = the parent row id."""
    out = []
    for row in read(path):
        prov = row.get("provenance") or {}
        if row.get("dataset") != "gen_policy" or prov.get("generator") != "scripts/p3/gen_policy.py":
            continue
        fam_mod = next((f for f, m in MODULES.items() if row["family"] in getattr(m, "FAMILIES", (f,))), None)
        if fam_mod is None:
            continue
        mod = MODULES[fam_mod]
        want_unknown = row.get("gold") is None
        for k in range(per):
            for attempt in range(40):
                rng = item_rng("variant", row["id"], k, attempt)
                try:
                    items = mod.generate(rng, row["difficulty"], prov["kind"], want_unknown)
                except (ValueError, IndexError, KeyError, ZeroDivisionError):
                    continue
                # pick the item that plays the parent's role: the unknown child for unknown parents, the matching label otherwise
                cands = [j for j, it in enumerate(items) if (it["gold"] is None) == want_unknown]
                if not cands:
                    continue
                j = cands[0]
                if prov.get("group_id"):
                    j = cands[int(row["id"].rsplit("-", 1)[-1]) % len(cands)] if row["id"].rsplit("-", 1)[-1].isdigit() else cands[0]
                it = dict(items[j])
                it["child_of"] = None
                it["group"] = None
                r = _rows_from_items([it], fam_mod, f"{row['id']}-v{k}",
                                     {"seed": f"variant:{row['id']}:{k}", "index": k, "difficulty": row["difficulty"]},
                                     parent_override=row["id"])[0]
                r["difficulty"] = row["difficulty"]
                out.append(r)
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="+", default=list(FAMILY_NAMES), choices=list(FAMILY_NAMES))
    ap.add_argument("--count", type=int, default=None, help="rows per selected family (default: the plan's per-family targets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--variant-of", help="candidate JSONL of verified gen_policy parents; writes Stage-2 variants instead")
    ap.add_argument("--variants-per", type=int, default=2)
    a = ap.parse_args(argv)
    if a.variant_of:
        rows = variants_of(a.variant_of, a.variants_per)
        out = a.out if a.out != str(OUT) else str(ROOT / "data/p3/candidates/A-policy-variants.jsonl")
        print(f"wrote {write(out, rows)} variants -> {out}")
        return 0
    fams = {f: (a.count if a.count is not None else DEFAULT_ROWS[f]) for f in a.families}
    rows = generate(fams, a.seed, a.workers)
    ids = Counter(r["id"] for r in rows)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate ids: {dup[:5]}")
    n = write(a.out, rows)
    fam_c = Counter(r["family"] for r in rows)
    print(f"wrote {n} rows -> {a.out}; " + ", ".join(f"{k} {v}" for k, v in sorted(fam_c.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
