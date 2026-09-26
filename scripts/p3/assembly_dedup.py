"""Cross-source near-duplicate removal for the phase-3 pool (scripts/p3/assemble_pool.py step a).

Same normalisation and rule as scripts/p3/decontam.py: lower-case [a-z0-9]+ tokens per string leaf, distinct 13-grams per text
unit (state leaves; question + option/level leaves), and two rows are near-duplicates when one unit shares more than 50% of
either side's distinct 13-grams (max(shared / n_a, shared / n_b) > 0.5). Only pairs from DIFFERENT source files, different
split groups and the same modality are compared; question+options overlap alone counts only for text rows without any
state 13-gram (a trap and its parent, or two questions over one filing, are intended siblings, not duplicates).

13-grams that occur in more than DF_MAX rows are boilerplate (a generator's fixed sentences, a template header); they are
ignored for pair finding, which can only under-count a template-only overlap, never invent one. The known case is the Eikos
judge rows derived from TAT-QA / GSM8K train items (C-text) against source B's TAT-QA conversion.

Which copy is kept: the public-licensed one (licence other than own/generated), then source B (public dataset with an upstream
id the decontam join understands), then a checkable gold (constructed/dataset), then the richer state (longer), then target
probabilities, then the smaller id.
"""
from __future__ import annotations

import collections
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decontam as dc  # noqa: E402
import decontam_index as di  # noqa: E402

DF_MAX = 200
_G: dict = {}


def _grams_work(span):
    lines, tok = _G["lines"], _G["tok"]
    out = []
    for i in range(*span):
        r = json.loads(lines[i])
        st, q = dc.cand_units(r)
        gs = di.grams([x for x in (tok.ids(s) for s in st) if x])
        gq = di.grams([x for x in (tok.ids(s) for s in q) if x])
        out.append((i, gs, gq))
    return out


def _init():
    _G["tok"] = di.Tok()


def row_grams(lines: list, workers: int = 0, batch: int = 500):
    """(state grams, qopts grams) per line, as lists of uint64 arrays."""
    _G["lines"] = lines
    n = len(lines)
    st: list = [None] * n
    qo: list = [None] * n
    spans = [(s, min(n, s + batch)) for s in range(0, n, batch)]
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    if workers == 1 or len(spans) <= 1:
        _init()
        res_iter = (r for sp in spans for r in _grams_work(sp))
        for i, gs, gq in res_iter:
            st[i], qo[i] = gs, gq
        return st, qo
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        for res in pool.imap_unordered(_grams_work, spans, chunksize=2):
            for i, gs, gq in res:
                st[i], qo[i] = gs, gq
    return st, qo


def keep_score(row: dict) -> tuple:
    pv = row.get("provenance") or {}
    public = pv.get("licence") not in ("own", "generated")
    state_len = len(row["state"]) if isinstance(row["state"], str) else len(json.dumps(row["state"], ensure_ascii=False))
    return (int(public), int(row.get("source") == "B"), int(row.get("gold_kind") in ("constructed", "dataset")), state_len,
            int(bool(pv.get("target_probs") or pv.get("teacher_probs"))), _neg_id(row["id"]))


def _neg_id(rid: str) -> tuple:
    return tuple(-ord(c) for c in rid)          # smaller id wins ties


def find_pairs(unit_grams: list, file_idx: np.ndarray, group_idx: np.ndarray, df_max: int = DF_MAX,
               near_dup: float = dc.NEAR_DUP) -> list[tuple[int, int, float]]:
    """Near-duplicate (a, b, share) pairs across files for one unit kind."""
    lens = np.array([0 if g is None else len(g) for g in unit_grams], dtype=np.int64)
    if lens.sum() == 0:
        return []
    H = np.concatenate([g for g in unit_grams if g is not None and len(g)])
    R = np.repeat(np.arange(len(unit_grams), dtype=np.int64), lens)
    o = np.argsort(H, kind="stable")
    H, R = H[o], R[o]
    F = file_idx[R]
    # hashes seen in >= 2 files, df <= df_max
    starts = np.flatnonzero(np.concatenate([[True], H[1:] != H[:-1]]))
    ends = np.concatenate([starts[1:], [len(H)]])
    df = ends - starts
    fmin = np.minimum.reduceat(F, starts)
    fmax = np.maximum.reduceat(F, starts)
    sel = (df >= 2) & (df <= df_max) & (fmin != fmax)
    counts: collections.Counter = collections.Counter()
    for s, e in zip(starts[sel], ends[sel]):
        rows = R[s:e]
        fs = F[s:e]
        gs = group_idx[rows]
        k = len(rows)
        for x in range(k):
            for y in range(x + 1, k):
                if fs[x] != fs[y] and gs[x] != gs[y]:
                    a, b = int(rows[x]), int(rows[y])
                    counts[(a, b) if a < b else (b, a)] += 1
    out = []
    for (a, b), c in counts.items():
        share = max(c / lens[a], c / lens[b])
        if share > near_dup:
            out.append((a, b, float(share)))
    return out


def cross_source_duplicates(rows: list[dict], lines: list, files: list[str], groups: list[str], workers: int = 0):
    """-> (loser indices, pair records for the report, Counter of 13-gram pairs not counted as duplicates and why).

    A pair counts when one state shares > 50% of either side's 13-grams (both rows text, or both with images), or when the
    question+options do and neither row has any state 13-gram (text rows only). A question/option template shared over
    different states (e.g. a trap and an unrelated generator item with the same option set) is not a duplicate; nor is a
    text row against an image row."""
    fnames = sorted(set(files))
    file_idx = np.array([fnames.index(f) for f in files], dtype=np.int32)
    gnames = {g: i for i, g in enumerate(sorted(set(groups)))}
    group_idx = np.array([gnames[g] for g in groups], dtype=np.int64)
    st, qo = row_grams(lines, workers)
    has_img = [bool(r.get("images")) for r in rows]
    no_state = [st[i] is None or len(st[i]) == 0 for i in range(len(rows))]
    pairs, skipped = [], collections.Counter()
    for kind, ug in (("state", st), ("qopts", qo)):
        for a, b, share in find_pairs(ug, file_idx, group_idx):
            if has_img[a] != has_img[b]:
                skipped["cross_modality"] += 1          # a text row and an image row are different decisions
            elif kind == "qopts" and not (no_state[a] and no_state[b] and not has_img[a]):
                skipped["qopts_only_template"] += 1     # same question/option template over different states
            else:
                pairs.append((a, b, kind, share))
    losers = set()
    records = []
    for a, b, kind, share in sorted(pairs, key=lambda p: -p[3]):
        wa, wb = keep_score(rows[a]), keep_score(rows[b])
        win, lose = (a, b) if wa >= wb else (b, a)
        records.append({"kept": rows[win]["id"], "kept_file": files[win], "dropped": rows[lose]["id"], "dropped_file": files[lose],
                        "unit": kind, "share": round(share, 3)})
        if win not in losers:
            losers.add(lose)
    return losers, records, skipped
