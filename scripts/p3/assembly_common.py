"""Shared helpers for the phase-3 pool assembly (scripts/p3/assemble_pool.py) and manifest builder (scripts/p3/build_manifest.py).

- JSONL io, stable hashing (sha256-based, independent of PYTHONHASHSEED);
- split groups: a union-find over provenance.group_id / source_group / parent_group_id, the parent_id chain, GUI screen ids
  and upstream item keys (decontam.cand_upstream, e.g. a FinQA filing page or an ABO photo item). Held-out sets are drawn
  by whole group, and everything held out leaves the pool with its whole group (docs/phase-3-plan.md, Stage 1);
- a parallel link scanner that applies exactly the rules of `decontam.py --leakage` (same id / parent family, 13-gram
  near-duplicate of state or question+options, same whole state, same upstream item) but reports EVERY (train row,
  held-out row) link, not only the best-matching held-out row, so link costs are exact and the assembler can remove linked
  pool rows before the real `decontam.py --leakage` gate runs (any link it finds, this finds).
Images are deliberately not a group key: ABO photos are shared by thousands of rows across families (one 7,955-row
component), so image reuse is reported instead (`image_overlap`).
"""
from __future__ import annotations

import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decontam as dc  # noqa: E402
import decontam_index as di  # noqa: E402

CLEAN_DIR = ROOT / "data/p3/candidates-clean"
POOL_DIR = ROOT / "data/p3/pool"
MANIFEST_DIR = ROOT / "data/p3/manifests"
# every clean source file in the pool (drops-*/flagged-* are decontamination side files, never pool rows)
SOURCE_FILES = ("A-policy", "A-reasoning", "B-alfworld", "B-finqa", "B-musique", "B-plumb", "B-strategyqa", "B-tatqa",
                "C-images", "C-text", "D-traps", "I-gui", "I-joint")


# ------------------------------------------------------------------------------------------------ io + hashing
def read_jsonl(path) -> Iterator[dict]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()


def stable_unit(*parts) -> float:
    """Deterministic uniform number in [0, 1) from the parts."""
    return int(stable_hash(*parts)[:15], 16) / float(16 ** 15)


# ------------------------------------------------------------------------------------------------ groups
class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.p
        if x not in p:
            p[x] = x
            return x
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:          # deterministic representative: the smaller key
                ra, rb = rb, ra
            self.p[rb] = ra


def group_keys(row: dict) -> list[str]:
    """Keys that tie a row to others that must share its split (the row's own id first)."""
    pv = row.get("provenance") or {}
    keys = ["id:" + str(row["id"])]
    if row.get("parent_id"):
        keys.append("id:" + str(row["parent_id"]))
    for k in ("group_id", "source_group", "parent_group_id"):
        if pv.get(k):
            keys.append("g:" + str(pv[k]))
    if row.get("dataset") == "gui_synthetic" and pv.get("screen_id"):
        keys.append("gui:" + str(pv["screen_id"]))   # several questions over one rendered screen
    keys += ["u:" + u for u in dc.cand_upstream(row)]
    return keys


def assign_groups(rows: Iterable[dict]) -> dict[str, str]:
    """id -> group root (the smallest key of its union-find component; stable across runs)."""
    uf = UnionFind()
    ids = []
    for r in rows:
        ks = group_keys(r)
        ids.append(ks[0])
        for k in ks[1:]:
            uf.union(ks[0], k)
    return {i[3:]: uf.find(i) for i in ids}


# ------------------------------------------------------------------------------------------------ link scanner
_G: dict = {}


def bench_row(r: dict, family: str | None = None) -> di.BenchRow:
    st, q = dc.cand_units(r)
    return di.BenchRow("heldout", str(r["id"]), state=st, qopts=q, upstream=dc.cand_upstream(r), images=[], family=family)


def match_all(ix: di.Index, tok: di.Tok, state: list[str], qopts: list[str], ups: list[str], near_dup: float) -> list[tuple[str, int]]:
    """Every (rule, held-out row) link of one train row under decontam.match's rules (near_dup, state, upstream; no field
    rule, as in --leakage), not only the best-matching held-out row: a pool row similar to several held-out rows links to all."""
    out = []
    st_ids = [x for x in (tok.ids(s) for s in state) if x]
    q_ids = [x for x in (tok.ids(s) for s in qopts) if x]
    for ids in (st_ids, q_ids):
        g = di.grams(ids)
        if not len(g) or not len(ix.g_hash):
            continue
        lo = np.searchsorted(ix.g_hash, g, "left")
        hi = np.searchsorted(ix.g_hash, g, "right")
        hit = hi > lo
        if not hit.any():
            continue
        units = dc._expand(ix.g_unit, lo[hit], hi[hit])
        u, c = np.unique(units, return_counts=True)
        share = np.maximum(c / len(g), c / np.asarray(ix.unit_n[u]))
        for j in np.flatnonzero(share > near_dup):
            out.append(("near_dup", int(ix.unit_row[u[j]])))
    full = [t for x in st_ids for t in x]
    if len(full) >= di.MIN_STATE_TOKENS and len(ix.state_hash):
        h = np.uint64(di.seq_hash(full))
        lo, hi = np.searchsorted(ix.state_hash, h, "left"), np.searchsorted(ix.state_hash, h, "right")
        out += [("state", int(ix.state_row[k])) for k in range(lo, hi)]
    for key in ups:
        out += [("upstream", r) for r in ix.upstream.get(key, [])]
    return out


def _scan_init(near_dup):
    _G["tok"] = di.Tok()
    _G["near_dup"] = near_dup


def _scan_work(span: tuple[int, int]) -> list[tuple[int, str, str]]:
    ix, tok, fam_ids, lines = _G["ix"], _G["tok"], _G["held_family_ids"], _G["lines"]
    out = []
    for i in range(*span):
        r = json.loads(lines[i])
        rid = str(r.get("id"))
        if rid in fam_ids:
            out.append((i, "same_id", fam_ids[rid]))
        par = r.get("parent_id")
        if par and str(par) in fam_ids:
            out.append((i, "family", fam_ids[str(par)]))
        st, q = dc.cand_units(r)
        for reason, row in set(match_all(ix, tok, st, q, dc.cand_upstream(r), _G["near_dup"])):
            out.append((i, reason, ix.row_ids[row]))
    return out


def scan_links(train_lines: list, heldout_rows: list[dict], workers: int = 0, near_dup: float = dc.NEAR_DUP,
               batch: int = 400) -> list[tuple[int, str, str]]:
    """Every (train line index, reason, held-out id) link under the `decontam.py --leakage` rules.

    Family links: a train row whose id or parent_id is a held-out id (or a held-out row's parent). Held-out rows are indexed
    once; the train lines are scanned in forked workers (the index is inherited, not pickled)."""
    fam_ids = {}
    for h in heldout_rows:
        fam_ids[str(h["id"])] = str(h["id"])
        if h.get("parent_id"):
            fam_ids.setdefault(str(h["parent_id"]), str(h["id"]))
    _G["ix"] = di.build([bench_row(h) for h in heldout_rows])
    _G["held_family_ids"] = fam_ids
    _G["lines"] = train_lines
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    batches = [(s, min(s + batch, len(train_lines))) for s in range(0, len(train_lines), batch)]
    out: list[tuple[int, str, str]] = []
    if workers == 1 or len(batches) <= 1:
        _scan_init(near_dup)
        for b in batches:
            out += _scan_work(b)
        return out
    with mp.get_context("fork").Pool(workers, initializer=_scan_init, initargs=(near_dup,)) as pool:
        for res in pool.imap(_scan_work, batches, chunksize=1):
            out += res
    return out


def image_shas(row: dict) -> list[str]:
    pv = row.get("provenance") or {}
    s = pv.get("image_sha256")
    if isinstance(s, str):
        return [s]
    if isinstance(s, list):
        return [x for x in s if isinstance(x, str)]
    return [Path(p).stem for p in row.get("images") or [] if isinstance(p, str)]


def counter_table(counter: collections.Counter, head: tuple[str, str], sort_key=None) -> list[str]:
    L = [f"| {head[0]} | {head[1]} |", "|---|---:|"]
    items = sorted(counter.items(), key=sort_key) if sort_key else counter.most_common()
    L += [f"| {k} | {v:,} |" for k, v in items]
    return L
