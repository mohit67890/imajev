#!/usr/bin/env python3
"""Phase-3 decontamination filter and leakage checker (docs/phase-3-plan.md: "Decontamination, before mining",
Stage 1 leakage rule, "Licences and disclosure").

Filter a candidate pool (scripts/p3/candidate.py rows) against every evaluation set we must never train on
(JevBench public + our JevBench-style dev, DecisionBench 1.0 eval, fast-decisions dev, ImajevBench public + private-1):

    .venv/bin/python scripts/p3/decontam.py --in data/p3/candidates/B-finqa.jsonl [more.jsonl ...] \
        --out-dir data/p3/candidates-clean/ --report reports/phase3/decontam-B.md

A candidate is DROPPED if any of:
  upstream   provenance.upstream_id (+ dataset, + split for index-like ids) is an upstream item a benchmark row points to
             (FinQA: the same filing page counts as the same item);
  near_dup   its state or its question+options shares > 50% of its 13-grams with one benchmark text unit, or covers
             > 50% of that unit's 13-grams;
  field      any single field of >= 6 tokens equals a benchmark field (normalised);
  state      its whole normalised state equals a benchmark state (or DecisionBench raw upstream text);
  image      one of its images has the sha256 of a benchmark image.
Rows with any 13-gram hit that are kept are FLAGGED (counted, listed in flagged.jsonl) for inspection.
Kept rows are written byte-for-byte unchanged to <out-dir>/<input name>. drops.jsonl / flagged.jsonl carry id, reason and
benchmark name (plus the public benchmark row id); nothing from the hidden set private-1 but its name is ever written.

Leakage check for assembled splits (fails with exit 1 on any held-out/train link):

    .venv/bin/python scripts/p3/decontam.py --leakage --train A.jsonl [...] --heldout B.jsonl [...] [--report R.md]

Links: same id or parent_id family, a 13-gram near-duplicate (state or question+options), the same whole state, or the
same upstream item.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decontam_index as di  # noqa: E402

ROOT = di.ROOT
NEAR_DUP = 0.5


# ---------------------------------------------------------------- candidate side
def cand_units(row: dict) -> tuple[list[str], list[str]]:
    """(state leaves, question+options leaves) of a phase-3 candidate row."""
    state = list(di.leaves(row.get("state")))
    f = row.get("field") or {}
    q = [f.get("question") or ""]
    for o in f.get("options") or []:
        if isinstance(o, dict):
            q += [str(o.get(k)) for k in ("key", "text", "description") if o.get(k)]
    for lv in f.get("levels") or []:
        if isinstance(lv, dict) and lv.get("description"):
            q.append(str(lv["description"]))
    return state, [s for s in q if s]


def cand_upstream(row: dict) -> list[str]:
    pv = row.get("provenance") or {}
    uid = pv.get("upstream_id")
    if uid is None:
        uid = pv.get("upstream_ids")
    if uid is None:
        return []
    return di.upstream_keys(pv.get("upstream_dataset") or row.get("dataset"), uid, pv.get("upstream_split"))


_IMG_CACHE: dict[str, str | None] = {}


def image_sha(rel: str) -> str | None:
    if rel in _IMG_CACHE:
        return _IMG_CACHE[rel]
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / "data" / rel if not rel.startswith("data/") else ROOT / rel
    v = di.sha256_file(p) if p.exists() else None
    _IMG_CACHE[rel] = v
    return v


def row_images(row: dict) -> list[str]:
    out = []
    for im in row.get("images") or []:
        if isinstance(im, dict):
            out.append(im.get("sha256") or (image_sha(im["path"]) if im.get("path") else None))
        elif isinstance(im, str):
            out.append(image_sha(im))
    return [x for x in out if x]


# ---------------------------------------------------------------- matching
def _expand(idx_arr: np.ndarray, l: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Concatenate idx_arr[l[i]:r[i]] for all i."""
    lens = r - l
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, dtype=idx_arr.dtype)
    starts = np.repeat(l - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    pos = starts + np.arange(tot)
    return np.asarray(idx_arr[pos])


def match(ix: di.Index, tok: di.Tok, state: list[str], qopts: list[str], ups: list[str], imgs: list[str],
          near_dup: float = NEAR_DUP, field_rule: bool = True, template_rows: int = 0) -> dict:
    """Return {"drop": [(reason, row, detail)], "flag": (row, share) | None, "hit_grams": int, "template": bool}."""
    drops = []
    best_flag = None
    hit_total = 0
    template = False
    st_ids = [x for x in (tok.ids(s) for s in state) if x]
    q_ids = [x for x in (tok.ids(s) for s in qopts) if x]
    # (b) 13-gram near-duplicates
    for kind, ids in (("state", st_ids), ("qopts", q_ids)):
        g = di.grams(ids)
        if not len(g) or not len(ix.g_hash):
            continue
        l = np.searchsorted(ix.g_hash, g, "left")
        r = np.searchsorted(ix.g_hash, g, "right")
        hit = r > l
        nh = int(hit.sum())
        if not nh:
            continue
        hit_total += nh
        units = _expand(ix.g_unit, l[hit], r[hit])
        u, c = np.unique(units, return_counts=True)
        share_c = c / len(g)
        share_b = c / np.asarray(ix.unit_n[u])
        share = np.maximum(share_c, share_b)
        j = int(np.argmax(share))
        row = int(ix.unit_row[u[j]])
        if best_flag is None or share[j] > best_flag[1]:
            best_flag = (row, float(share[j]))
        if share[j] > near_dup:
            drops.append(("near_dup", row, {"unit": kind, "bench_unit": di.KIND_NAMES[int(ix.unit_kind[u[j]])],
                                            "share_of_candidate": round(float(share_c[j]), 3),
                                            "share_of_benchmark": round(float(share_b[j]), 3)}))
    # (c) exact field
    if field_rule and len(ix.leaf_hash):
        for ids in st_ids + q_ids:
            if len(ids) < di.MIN_FIELD_TOKENS:
                continue
            h = np.uint64(di.seq_hash(ids))
            k = int(np.searchsorted(ix.leaf_hash, h))
            if k < len(ix.leaf_hash) and ix.leaf_hash[k] == h:
                nrows = int(ix.leaf_rows[k])
                if template_rows and nrows >= template_rows:
                    template = True
                    continue  # template string shared by many benchmark rows (see --template-rows)
                drops.append(("field", int(ix.leaf_row[k]), {"tokens": len(ids), "bench_rows_with_field": nrows}))
                break
    # (d) exact whole state
    full = [t for x in st_ids for t in x]
    if len(full) >= di.MIN_STATE_TOKENS and len(ix.state_hash):
        h = np.uint64(di.seq_hash(full))
        k = int(np.searchsorted(ix.state_hash, h))
        if k < len(ix.state_hash) and ix.state_hash[k] == h:
            drops.append(("state", int(ix.state_row[k]), {"tokens": len(full)}))
    # (a) upstream ids
    for key in ups:
        rows = ix.upstream.get(key)
        if rows:
            drops.append(("upstream", rows[0], {"key_kind": "page" if "|page|" in key else "id"}))
            break
    # (e) images
    for s in imgs:
        rows = ix.images.get(s)
        if rows:
            drops.append(("image", rows[0], {}))
            break
    return {"drop": drops, "flag": best_flag, "hit_grams": hit_total, "template": template}


# ---------------------------------------------------------------- filter workers
G: dict = {}


def _init(index_dir: str | None, near_dup: float, template_rows: int):
    G["ix"] = di.Index.load(Path(index_dir)) if index_dir else G["ix"]
    G["tok"] = di.Tok()
    G["near_dup"] = near_dup
    G["template_rows"] = template_rows


def _entry(ix: di.Index, reason: str, row: int, detail: dict) -> dict:
    b = ix.bench_of(row)
    e = {"reason": reason, "benchmark": b}
    rid = ix.public_row_id(row)
    if rid:
        e["benchmark_row"] = rid
    if b not in di.PRIVATE_BENCHES:
        e.update(detail)
    return e


def _work(lines: list[bytes]) -> list[tuple]:
    ix, tok = G["ix"], G["tok"]
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            out.append(("bad", None, None, None, None))
            continue
        st, q = cand_units(row)
        m = match(ix, tok, st, q, cand_upstream(row), row_images(row), G["near_dup"], True, G["template_rows"])
        meta = (row.get("id"), row.get("source"), row.get("dataset"))
        if m["drop"]:
            ents = [_entry(ix, *d) for d in m["drop"]]
            out.append(("drop", meta, ents, None, None))
        elif m["flag"] is not None:
            r, share = m["flag"]
            out.append(("flag", meta, None, {"benchmark": ix.bench_of(r), **({"benchmark_row": ix.public_row_id(r)}
                        if ix.public_row_id(r) else {}), "max_share": round(share, 3), "hit_grams": m["hit_grams"],
                        "template_field": m["template"]}, None))
        elif m["template"]:
            out.append(("flag", meta, None, {"benchmark": "template-field", "template_field": True}, None))
        else:
            out.append(("keep", meta, None, None, None))
    return out


def _batches(path: Path, size: int):
    buf = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            buf.append(line)
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def run_filter(args) -> int:
    t0 = time.time()
    ix = di.get_index(rebuild=args.rebuild_index)
    t_index = time.time() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drops_p = Path(args.drops) if args.drops else out_dir / "drops.jsonl"
    flags_p = drops_p.with_name(drops_p.name.replace("drops", "flagged")) if "drops" in drops_p.name else out_dir / "flagged.jsonl"
    stats = collections.Counter()
    by_reason = collections.Counter()           # (reason, benchmark)
    by_ds = collections.defaultdict(collections.Counter)  # (file, source, dataset) -> keep/drop/flag
    flag_bench = collections.Counter()
    files_summary = []
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(str(di.INDEX_DIR), args.near_dup, args.template_rows)) as pool, \
            open(drops_p, "w") as fd, open(flags_p, "w") as ff:
        for inp in args.inputs:
            inp = Path(inp)
            outp = out_dir / inp.name
            if outp.resolve() == inp.resolve():
                raise SystemExit(f"refusing to overwrite the input {inp}")
            t1 = time.time()
            c = collections.Counter()
            with open(outp, "wb") as fo:
                for lines, res in _bounded(pool, _batches(inp, args.batch), 4 * workers):
                    for line, (kind, meta, ents, flag, _) in zip(lines, res):
                        c[kind] += 1
                        if kind == "bad":
                            continue
                        rid, src, ds = meta
                        by_ds[(inp.name, src, ds)][kind] += 1
                        if kind == "drop":
                            fd.write(json.dumps({"id": rid, "file": inp.name, "reason": ents[0]["reason"],
                                                 "benchmark": ents[0]["benchmark"], "matches": ents}) + "\n")
                            for e in {(e["reason"], e["benchmark"]) for e in ents}:
                                by_reason[e] += 1
                        else:
                            fo.write(line if line.endswith(b"\n") else line + b"\n")
                            if kind == "flag":
                                ff.write(json.dumps({"id": rid, "file": inp.name, **flag}) + "\n")
                                flag_bench[flag["benchmark"]] += 1
            dt = time.time() - t1
            n = sum(c.values())
            files_summary.append({"file": str(inp), "out": str(outp), "rows": n, "kept": c["keep"] + c["flag"],
                                  "dropped": c["drop"], "flagged": c["flag"], "bad_json": c["bad"], "seconds": round(dt, 1),
                                  "rows_per_s": round(n / dt) if dt else None})
            stats.update(c)
            print(f"{inp}: {n:,} rows, kept {c['keep'] + c['flag']:,}, dropped {c['drop']:,}, flagged {c['flag']:,}, "
                  f"{dt:.1f}s ({n / max(dt, 1e-9):,.0f} rows/s)")
    total_s = time.time() - t0
    if args.report:
        write_report(Path(args.report), ix, args, files_summary, by_reason, by_ds, flag_bench, drops_p, flags_p, t_index,
                     total_s, workers)
        print(f"report: {args.report}")
    return 0


def _bounded(pool, batches, window: int):
    """Ordered (batch, result) pairs with at most `window` batches in flight (streams arbitrarily large inputs)."""
    q = collections.deque()
    for b in batches:
        q.append((b, pool.apply_async(_work, (b,))))
        if len(q) >= window:
            b0, r0 = q.popleft()
            yield b0, r0.get()
    while q:
        b0, r0 = q.popleft()
        yield b0, r0.get()


def write_report(path: Path, ix: di.Index, args, files, by_reason, by_ds, flag_bench, drops_p, flags_p, t_index, total_s,
                 workers):
    src = json.loads((di.SOURCES_DIR / "sources.json").read_text())
    rows = sum(f["rows"] for f in files)
    L = [f"# Phase-3 decontamination report{(' — ' + args.name) if args.name else ''}", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M', time.localtime())} (local) by `scripts/p3/decontam.py`. "
         "Benchmarks are used only to *exclude* candidates; no benchmark row, label or output enters the pool.", "",
         "## Result", "",
         "| input | rows | kept | dropped | flagged (kept) | seconds | rows/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in files:
        L.append(f"| `{Path(f['file']).name}` | {f['rows']:,} | {f['kept']:,} | {f['dropped']:,} | {f['flagged']:,} | "
                 f"{f['seconds']} | {f['rows_per_s'] or '-'} |")
    L += ["", f"Total wall time {total_s:.1f}s incl. index load/build {t_index:.1f}s, {workers} workers. "
              f"Drops: `{drops_p}`; flagged rows: `{flags_p}`.", "",
          "## Drops by rule and benchmark", "",
          "A row can match several rules; each (rule, benchmark) pair is counted once per row.", "",
          "| rule | benchmark | rows |", "|---|---|---:|"]
    for (r, b), n in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
        L.append(f"| {r} | {b} | {n:,} |")
    if not by_reason:
        L.append("| - | - | 0 |")
    L += ["", "## Flagged (kept, any 13-gram hit below the near-duplicate threshold)", "", "| benchmark | rows |", "|---|---:|"]
    for b, n in flag_bench.most_common():
        L.append(f"| {b} | {n:,} |")
    if not flag_bench:
        L.append("| - | 0 |")
    L += ["", "## By source and dataset", "", "| file | source | dataset | kept | dropped | flagged |", "|---|---|---|---:|---:|---:|"]
    for (fn, s, d), c in sorted(by_ds.items(), key=lambda x: (x[0][0], str(x[0][1]), str(x[0][2]))):
        L.append(f"| `{fn}` | {s} | {d} | {c['keep'] + c['flag']:,} | {c['drop']:,} | {c['flag']:,} |")
    L += ["", "## Rules", "",
          f"- **upstream**: `provenance.upstream_id` joined to the upstream ids in DecisionBench `source_json` "
          "(MuSiQue, FinQA and FOLIO ids; ESCI example ids; civil_comments / ToolACE / legalbench / big_patent / msmarco "
          "/ Nemotron-PII / entity-resolution row ids; dataset names normalised; purely numeric ids must also match the split; "
          "a FinQA filing page counts as one item).",
          f"- **near_dup**: 13-grams of normalised text (lower-case `[a-z0-9]+` tokens, per field); dropped when the "
          f"candidate's state or question+options shares > {args.near_dup:.0%} of its distinct 13-grams with one benchmark "
          "text unit (state, question+options, or DecisionBench raw upstream text), or covers > "
          f"{args.near_dup:.0%} of that unit's 13-grams.",
          f"- **field**: any single field of >= {di.MIN_FIELD_TOKENS} tokens equal to a benchmark field"
          + (f" (fields shared by >= {args.template_rows} benchmark rows (label names, task instructions) are templates: flagged, not dropped)"
             if args.template_rows else "") + ".",
          f"- **state**: whole normalised state (>= {di.MIN_STATE_TOKENS} tokens) equal to a benchmark state or raw upstream text.",
          "- **image**: an image sha256 equal to an ImajevBench image.",
          "- **flagged**: kept rows with at least one 13-gram shared with a benchmark, or an exact match only on a "
          "template field (listed for inspection; benchmark `template-field` in the table).", "",
          "## Benchmarks indexed", "", "| benchmark | rows indexed | origin | revision |", "|---|---:|---|---|"]
    for b, n in ix.meta["bench_rows"].items():
        s = src["sources"].get(b, {})
        L.append(f"| {b} | {n:,} | {s.get('origin', '')} | {s.get('revision') or ''} |")
    L += ["", f"Index: {ix.meta['n_gram_pairs']:,} (13-gram, text unit) pairs, {ix.meta['n_leaves']:,} fields, "
              f"{ix.meta['n_upstream_keys']:,} upstream keys, {ix.meta['n_images']} image hashes; built "
              f"{ix.meta.get('built_utc')}. Source copies and hashes: `data/p3/decontam-sources/sources.json`. "
              "The hidden ImajevBench set private-1 is indexed as hashes only; no row id or text of it appears here or in "
              "drops.jsonl.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


# ---------------------------------------------------------------- leakage
def _roots(paths) -> dict[str, str]:
    """id -> family root (follow parent_id chains across all given files)."""
    parent = {}
    for p in paths:
        for r in di._jsonl(Path(p)):
            parent[str(r.get("id"))] = str(r["parent_id"]) if r.get("parent_id") else None
    memo = {}

    def root(i):
        seen = []
        while True:
            if i in memo:
                res = memo[i]
                break
            seen.append(i)
            p = parent.get(i)
            if not p or p in seen:
                res = i
                break
            i = p
        for s in seen:
            memo[s] = res
        return res
    return {i: root(i) for i in parent}


def run_leakage(args) -> int:
    t0 = time.time()
    roots = _roots(list(args.train) + list(args.heldout))
    held_rows = []
    held_fams = set()
    held_ids = set()
    for p in args.heldout:
        for r in di._jsonl(Path(p)):
            st, q = cand_units(r)
            rid = str(r.get("id"))
            held_ids.add(rid)
            fam = roots.get(rid, rid)
            held_fams.add(fam)
            if r.get("parent_id"):
                held_fams.add(str(r["parent_id"]))
            held_rows.append(di.BenchRow("heldout", rid, state=st, qopts=q, upstream=cand_upstream(r), images=[],
                                         family=fam))
    ix = di.build(held_rows)
    tok = di.Tok()
    hits = collections.Counter()
    examples = []
    n = 0
    for p in args.train:
        for r in di._jsonl(Path(p)):
            n += 1
            rid = str(r.get("id"))
            fam = roots.get(rid, rid)
            links = []
            if rid in held_ids:
                links.append(("same_id", rid))
            elif fam in held_fams or (r.get("parent_id") and str(r["parent_id"]) in held_fams):
                links.append(("family", fam))
            st, q = cand_units(r)
            m = match(ix, tok, st, q, cand_upstream(r), [], args.near_dup, field_rule=False)
            for reason, row, _ in m["drop"]:
                links.append((reason, ix.row_ids[row]))
            for reason, other in links:
                hits[reason] += 1
                if len(examples) < 50:
                    examples.append({"train_id": rid, "file": Path(p).name, "link": reason, "heldout": other})
    ok = not hits
    msg = (f"leakage check: {n:,} train rows vs {len(held_rows):,} held-out rows ({len(held_fams):,} families): "
           + ("PASS" if ok else "FAIL " + ", ".join(f"{k}={v}" for k, v in hits.most_common())) + f" [{time.time() - t0:.1f}s]")
    print(msg)
    for e in examples[:20]:
        print("  ", json.dumps(e))
    if args.report:
        L = ["# Phase-3 leakage check", "", f"Train: {', '.join(f'`{x}`' for x in args.train)}",
             f"Held-out: {', '.join(f'`{x}`' for x in args.heldout)}", "", f"**{'PASS' if ok else 'FAIL'}** — {msg}", ""]
        if examples:
            L += ["| train id | link | held-out id |", "|---|---|---|"] + \
                 [f"| {e['train_id']} | {e['link']} | {e['heldout']} |" for e in examples]
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(L) + "\n")
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", help="candidate JSONL files to filter")
    ap.add_argument("--out-dir", help="where kept rows, drops.jsonl and flagged.jsonl go")
    ap.add_argument("--report", help="markdown report path (reports/phase3/decontam-<name>.md)")
    ap.add_argument("--name", help="label for the report title")
    ap.add_argument("--drops", help="drops file (default <out-dir>/drops.jsonl; flagged rows go next to it)")
    ap.add_argument("--near-dup", type=float, default=NEAR_DUP, help="13-gram overlap share above which a row is dropped")
    ap.add_argument("--template-rows", type=int, default=50,
                    help="exact-field rule: a benchmark field shared by >= this many benchmark rows is a template (label "
                         "names, task instructions), flagged not dropped (0 = drop on every field match)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--build-index", action="store_true", help="only (re)build the index and print its summary")
    ap.add_argument("--leakage", action="store_true", help="held-out vs train leakage check (exit 1 on any link)")
    ap.add_argument("--train", nargs="+", default=[])
    ap.add_argument("--heldout", nargs="+", default=[])
    args = ap.parse_args(argv)
    if args.leakage:
        if not args.train or not args.heldout:
            ap.error("--leakage needs --train and --heldout")
        return run_leakage(args)
    if args.build_index:
        ix = di.get_index(rebuild=args.rebuild_index)
        print(json.dumps({k: v for k, v in ix.meta.items() if k not in ("row_ids", "row_family", "upstream", "images")}, indent=1))
        return 0
    if not args.inputs or not args.out_dir:
        ap.error("--in and --out-dir are required")
    return run_filter(args)


if __name__ == "__main__":
    sys.exit(main())
