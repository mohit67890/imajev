"""Held-out leakage filter for gen_geometry.py / gen_screens.py (--heldout --train ...).

Uses the same link rules as `scripts/p3/decontam.py --leakage` (13-gram near-duplicate of state or question+options,
exact whole state, same upstream item, same id / parent family) and returns every link per held-out row. The GUI
exemption (reports/phase3/pool.md, "GUI") applies: a link of kind "state" only is allowed, because the states of these
families are short fixed strings ("a screenshot of a phone settings screen", "points on a coordinate grid") shared by
every item of a template, while the screenshots differ. Held-out rows with any other link are dropped (with their
parent / children, so a family never straddles the split); screenshots are never shared (fresh seed, and checked).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decontam as dc  # noqa: E402
import decontam_index as di  # noqa: E402


def links(held: list[dict], train_paths: list[Path]) -> dict[str, collections.Counter]:
    """held-out id -> Counter(link kind) over all train rows."""
    brs = []
    for r in held:
        st, q = dc.cand_units(r)
        brs.append(di.BenchRow("heldout", r["id"], state=st, qopts=q, upstream=dc.cand_upstream(r), images=[], family=r["id"]))
    ix = di.build(brs)
    tok = di.Tok()
    held_ids = {r["id"] for r in held}
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in train_paths:
        for r in di._jsonl(Path(p)):
            if r.get("id") in held_ids:
                out[r["id"]]["same_id"] += 1
            st, q = dc.cand_units(r)
            m = dc.match(ix, tok, st, q, dc.cand_upstream(r), [], dc.NEAR_DUP, field_rule=False)
            for reason, row, _ in m["drop"]:
                out[ix.row_ids[row]][reason] += 1
    return out


def filter_heldout(held: list[dict], train_paths: list[Path], train_images: set[str], target: int) -> tuple[list[dict], dict]:
    lk = links(held, train_paths)
    fam = {}
    for r in held:
        fam[r["id"]] = r["parent_id"] or r["id"]
    bad_fams = set()
    for r in held:
        kinds = set(lk.get(r["id"], {}))
        if kinds - {"state"} or r["images"][0] in train_images or Path(r["images"][0]).name in train_images:
            bad_fams.add(fam[r["id"]])
    kept = [r for r in held if fam[r["id"]] not in bad_fams]
    # keep whole families, in order, up to the target
    out, fams_taken = [], set()
    by_fam = collections.defaultdict(list)
    for r in kept:
        by_fam[fam[r["id"]]].append(r)
    for r in kept:
        f = fam[r["id"]]
        if f in fams_taken:
            continue
        if len(out) + len(by_fam[f]) > target:
            continue
        fams_taken.add(f)
        out += by_fam[f]
    stats = {"generated": len(held), "dropped_for_links": len(held) - len(kept), "kept": len(out),
             "link_kinds_dropped": dict(collections.Counter(k for r in held if fam[r["id"]] in bad_fams for k in lk.get(r["id"], {}))),
             "state_linked_kept": sum(1 for r in out if lk.get(r["id"]))}
    return out, stats


if __name__ == "__main__":
    print(json.dumps({"doc": __doc__}))
