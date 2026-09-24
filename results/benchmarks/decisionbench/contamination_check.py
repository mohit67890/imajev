#!/usr/bin/env python3
"""DecisionBench (Hanno-Labs/decision-bench @ b7c8107e, split "eval") vs imajev-4b 1.0 training data: text overlap check.

Read-only. Streams every training file once and builds the n-gram index from DecisionBench, which is the smaller side.

    .venv/bin/python reports/benchmarks/decisionbench/contamination_check.py            # writes contamination-results.json

Normalisation is the one the JevBench 8-gram lint uses (`scripts/p2/p2_common.py::ngrams`): lower-case, tokens = `[a-z0-9]+`.
Every string leaf of a JSON value is tokenised on its own, so n-grams never span two fields (a sentinel token separates leaves).
n-grams are 64-bit polynomial hashes of 64-bit blake2b token ids (collision odds are negligible at these set sizes).

Per DecisionBench row, for the state text (the JSON `state` values) and, separately, for the raw upstream text kept in
`source_json.source.raw` (the pre-paraphrase original for the 12 paraphrased tasks):
  * 13-gram hits: distinct 13-grams of the row found in any training text; share of the row's 13-grams found (near-dup > 0.5);
  * 8-gram hits (the JevBench lint's n, flag at >= 2 shared 8-grams, as in the lint);
  * short-span containment: a state leaf of 6-12 tokens (too short for a 13-gram) found verbatim as a contiguous token span;
  * exact state duplicate: the row's whole normalised state (all leaves concatenated) equals a training state or string leaf.
Everything is counted twice: against training rows in the `train` partition only (what gradient updates saw), and against
any partition of the scanned files (dev/test/calibration included, i.e. also selection/calibration data).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DB_PARQUET = Path.home() / ".cache/huggingface/hub/datasets--Hanno-Labs--decision-bench/snapshots/" \
    "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443/data/eval-00000-of-00001.parquet"
PREV = Path(os.environ.get("CONTAM_PREV_DIR", "scratch/prev"))  # holds p2c-audit/decision-p2c.jsonl (local pre-run build)
SCRATCH = Path(os.environ.get("CONTAM_SCRATCH_DIR", "scratch"))  # holds writer-prog-seed7.jsonl and the DecisionBench checkout

# (label, path, role, text extractor). role: "train" = fed gradient updates of some imajev-4b 1.0 stage (per partition);
# "selection" = dev/calibration panels used only for checkpoint selection or temperature fitting.
SCAN = [
    ("decision-v2.1-4b", ROOT / "data/manifests/decision-v2.1-4b.jsonl", "train", "record"),
    ("decision-p2", ROOT / "data/manifests/decision-p2.jsonl", "train", "record"),
    ("decision-p2b", ROOT / "data/manifests/decision-p2b.jsonl", "train", "record"),
    ("eikos-strict-slice", ROOT / "data/external/eikos-decisions/records.jsonl", "train", "record"),
    ("p2c-image-replay-delta", ROOT / "data/decision-p2c/image-replay/replay-delta.jsonl", "train", "record"),
    ("p2c-programmatic-seed7 (regenerated)", SCRATCH / "writer-prog-seed7.jsonl", "train", "writer"),
    ("p2c-dry-build (local pre-run build)", PREV / "p2c-audit/decision-p2c.jsonl", "train", "record"),
    ("decision-v2-reasoning-dev", ROOT / "data/manifests/decision-v2-reasoning-dev.jsonl", "selection", "record"),
    ("decision-v2.1-grounded-dev", ROOT / "data/manifests/decision-v2.1-grounded-dev.jsonl", "selection", "record"),
    ("decision-p2b-jevstyle-dev", ROOT / "data/manifests/decision-p2b-jevstyle-dev.jsonl", "selection", "record"),
    ("decision-v2.1-calibration", ROOT / "data/manifests/decision-v2.1-calibration.jsonl", "selection", "record"),
]

TOKEN = re.compile(r"[a-z0-9]+")
P = np.uint64(0x9E3779B97F4A7C15)
SENT = np.uint64(0xFFFFFFFFFFFFFFC5)
N_BIG, N_LINT, SHORT = 13, 8, range(6, 13)

# Public task names (decision_bench/task_spec.py::_CANONICAL_TASKS), keyed by (task_name, domain, primitive).
PUBLIC = {}


def tok_ids(text: str, cache: dict) -> list[int]:
    out = []
    for t in TOKEN.findall(text.lower()):
        v = cache.get(t)
        if v is None:
            v = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            if v == int(SENT):
                v ^= 1
            cache[t] = v
        out.append(v)
    return out


def leaves(value, skip=()):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in skip:
                yield from leaves(v, skip)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v, skip)


def path_leaves(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from path_leaves(v, f"{path}.{k}")
    elif isinstance(value, list):
        for v in value:
            yield from path_leaves(v, f"{path}[]")


QUESTION_KEYS = ("question", "query", "conclusion", "tool_call", "text")


def seq_hash(ids) -> bytes:
    return hashlib.blake2b(np.asarray(ids, dtype=np.uint64).tobytes(), digest_size=12).digest()


def rolling(arr: np.ndarray, upto: int):
    """Yield (n, hashes of every n-gram) for n = 1..upto, where hashes[i] covers arr[i:i+n]."""
    h = arr.copy()
    yield 1, h
    with np.errstate(over="ignore"):
        for n in range(2, upto + 1):
            if len(arr) < n:
                return
            h = h[:-1] * P + arr[n - 1:]
            yield n, h


# ---------------------------------------------------------------- DecisionBench side
def load_db():
    t = pq.read_table(DB_PARQUET).to_pylist()
    cache: dict = {}
    rows = []
    for r in t:
        state = json.loads(r["state_json"])
        src = json.loads(r["source_json"] or "{}")
        raw = (src.get("source") or {}).get("raw") if isinstance(src.get("source"), dict) else None
        pl = [(p_, tok_ids(v, cache)) for p_, v in path_leaves(state)]
        pl = [(p_, x) for p_, x in pl if x]
        s_leaves = [x for _, x in pl]
        r_leaves = [x for x in (tok_ids(s, cache) for s in leaves(raw, skip=("stratum", "label", "category", "label_key",
                                                                               "source_label", "selected_tool")))
                    if len(x) >= 6] if raw is not None else []
        rows.append({
            "row_id": r["row_id"], "task_name": r["task_name"], "domain": r["domain"], "primitive": r["primitive"],
            "state_text": json.dumps(state, ensure_ascii=False), "s": s_leaves, "r": r_leaves,
            "s_paths": [p_ for p_, _ in pl], "raw": raw, "state": state, "src": src.get("source") if isinstance(src.get("source"), dict) else None,
            "raw_text": json.dumps(raw, ensure_ascii=False) if raw is not None else None,
        })
    return rows


def grams_of(leaf_lists, n):
    out = set()
    for ids in leaf_lists:
        if len(ids) >= n:
            a = np.asarray(ids, dtype=np.uint64)
            for k, h in rolling(a, n):
                if k == n:
                    out.update(h.tolist())
    return out


def build_index(rows):
    g13, g8 = set(), set()
    short = {L: set() for L in SHORT}
    exact = set()
    for row in rows:
        for kind in ("s", "r"):
            lv = row[kind]
            row[f"{kind}13"] = grams_of(lv, N_BIG)
            row[f"{kind}8"] = grams_of(lv, N_LINT)
            g13 |= row[f"{kind}13"]
            g8 |= row[f"{kind}8"]
            row[f"{kind}short"] = []
            for ids in lv:
                if len(ids) in SHORT:
                    a = np.asarray(ids, dtype=np.uint64)
                    h = [hh for k, hh in rolling(a, len(ids)) if k == len(ids)][0][0]
                    row[f"{kind}short"].append(int(h))
                    short[len(ids)].add(int(h))
            row[f"{kind}leaf13"] = [grams_of([ids], N_BIG) for ids in lv]
            row[f"{kind}leafshort"] = [(int([hh for k, hh in rolling(np.asarray(ids, dtype=np.uint64), len(ids))
                                             if k == len(ids)][0][0]) if len(ids) in SHORT else None) for ids in lv]
            full = [t for ids in lv for t in ids]
            row[f"{kind}exact"] = seq_hash(full) if len(full) >= 3 else None
            if row[f"{kind}exact"]:
                exact.add(row[f"{kind}exact"])
    return g13, g8, short, exact


# ---------------------------------------------------------------- targeted source-item checks
def norm(t) -> str:
    """Unicode-aware key for item joins (the n-gram tokeniser drops non-Latin scripts, which would merge e.g. Japanese
    ESCI queries into one empty key)."""
    return " ".join(re.findall(r"\w+", str(t or "").casefold()))


def targeted(rows):
    """Item-level joins for the upstream datasets DecisionBench and the v2.1-4b mixture share (banking77, CLINC150,
    civil_comments, ESCI): is the exact upstream item (and its label) in our training data?"""
    man = ROOT / "data/manifests/decision-v2.1-4b.jsonl"
    db = collections.defaultdict(dict)
    for row in rows:
        raw = row["raw"]
        if not isinstance(raw, dict):
            continue
        ds = (row["src"] or {}).get("dataset")
        if ds == "PolyAI/banking77":
            db["banking77"][norm(raw["text"])] = (row["row_id"], raw["category"].lower(), row["src"].get("split"))
        elif ds == "clinc/clinc_oos":
            db["clinc150"][norm(raw["text"])] = (row["row_id"], raw["category"].lower(), row["src"].get("split"))
        elif ds == "google/civil_comments":
            k_ = norm(raw["text"])
            prev = db["civil_comments"].get(k_)
            db["civil_comments"][k_] = ((prev[0] + "+" if prev else "") + row["row_id"], row["task_name"], row["src"].get("split"))
        elif ds == "tasksource/esci":
            ids = row["src"].get("source_ids") or []
            prev = db["esci"].get(int(ids[0]))
            db["esci"][int(ids[0])] = ((prev[0] + "+" if prev else "") + row["row_id"], norm(raw["state"]["query"]), norm(raw["state"]["product"].get("title")),
                                       raw.get("label"), row["src"].get("split"))
    # ESCI example_id -> product_id through the upstream examples table we downloaded for stage 1
    ex = pq.read_table(ROOT / "data/decision-v1-text/raw/esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
                       columns=["example_id", "query", "product_id", "esci_label", "split"]).to_pylist()
    esci_by_id = {e["example_id"]: e for e in ex if e["example_id"] in db["esci"]}
    esci_pairs_title = {(v[1], v[2]): k for k, v in db["esci"].items()}
    esci_pairs_sku = {(v[1], esci_by_id[k]["product_id"]): k for k, v in db["esci"].items() if k in esci_by_id}
    esci_query = collections.defaultdict(list)
    for k, v in db["esci"].items():
        esci_query[v[1]].append(k)
    res = {n: collections.Counter() for n in ("banking77", "clinc150", "civil_comments", "esci")}
    hits = {n: {} for n in res}
    esci_q_hits = collections.defaultdict(set)
    want = ('"source": "banking77"', '"source": "clinc150"', '"source": "civil_comments"', '"source": "esci"',
            '"source": "sqid_esci"', '"source": "state_aware"')
    with open(man) as f:
        for line in f:
            if not any(w in line for w in want):
                continue
            r = json.loads(line)
            src, part = r["source"], r["partition"]
            st = r["request"].get("state")
            if src in ("banking77", "clinc150", "civil_comments") and isinstance(st, (str, dict)):
                text = st if isinstance(st, str) else " ".join(leaves(st))
                v = db[src].get(norm(text))
                if v is not None:
                    tgt = r.get("target")
                    same = (str(tgt).lower() == v[1]) if src != "civil_comments" else None
                    hits[src].setdefault(v[0], []).append({"partition": part, "train_id": r["id"],
                                                           "upstream_split": r.get("source_split"), "same_label": same})
            elif src == "esci" or (src in ("sqid_esci", "state_aware") and "esci" in str(r.get("source_split"))):
                if not isinstance(st, dict):
                    st = {}
                if src == "esci":
                    q, title = norm(st.get("query")), norm((st.get("listing") or {}).get("product_title"))
                    k = esci_pairs_title.get((q, title))
                elif src == "sqid_esci":
                    m = re.search(r'query "(.*)"', r["request"]["fields"][0]["question"])
                    q = norm(m.group(1)) if m else ""
                    k = esci_pairs_sku.get((q, str(r.get("source_group"))))
                else:
                    q = norm(((st or {}).get("session") or {}).get("search", {}).get("query"))
                    k = esci_pairs_sku.get((q, str(((st or {}).get("product") or {}).get("sku"))))
                if not q:
                    continue
                if q in esci_query:
                    esci_q_hits[part].update(esci_query[q])
                if k is not None:
                    ours = {0: "i", 1: "c", 2: "s", 3: "e"}.get(r.get("target")) if isinstance(r.get("target"), int) else None
                    same = (ours == str(db["esci"][k][3] or "")[:1].lower()) if ours else None
                    hits["esci"].setdefault(db["esci"][k][0], []).append({"partition": part, "train_id": r["id"],
                                                                          "upstream_split": r.get("source_split"),
                                                                          "same_label": same})
    split_of = {n: {} for n in res}
    for n in res:
        for v in db[n].values():
            for rid in v[0].split("+"):
                split_of[n][rid] = v[-1]
    for n in res:
        for rid in list(hits[n]):
            if "+" in rid:
                lst = hits[n].pop(rid)
                for r_ in rid.split("+"):
                    hits[n][r_] = lst
    out = {}
    for n in res:
        h = hits[n]
        tr = {rid for rid, lst in h.items() if any(x["partition"] == "train" for x in lst)}
        anyp = set(h)
        lab = {rid for rid, lst in h.items() if any(x["partition"] == "train" and x["same_label"] for x in lst)}
        out[n] = {"db_rows": len(split_of[n]), "db_upstream_splits": dict(collections.Counter(v[-1] for v in db[n].values())),
                  "item_in_train_partition": len(tr), "item_in_any_partition": len(anyp),
                  "item_in_train_with_same_label": len(lab) if n in ("banking77", "clinc150", "esci") else None,
                  "item_in_train_by_db_upstream_split": dict(collections.Counter(split_of[n][rid] for rid in tr)),
                  "our_upstream_splits_for_train_hits": dict(collections.Counter(
                      x["upstream_split"] for lst in h.values() for x in lst if x["partition"] == "train")),
                  "train_row_ids": sorted(tr),
                  "examples": [{"row_id": rid, "matches": lst[:2]} for rid, lst in list(h.items())[:5]]}
    out["esci"]["query_seen_in_train_partition"] = len(esci_q_hits["train"])
    out["esci"]["query_seen_in_any_partition"] = len(set().union(*esci_q_hits.values())) if esci_q_hits else 0
    out["esci"]["db_example_ids_resolved_to_product_id"] = len(esci_by_id)
    return out


# ---------------------------------------------------------------- training side (workers)
G = {}


def record_texts(rec: dict, kind: str):
    """(state-ish leaves, all text leaves) for one training line."""
    if kind == "writer":
        out = rec.get("output") or {}
        st = [out.get("document")] if isinstance(out.get("document"), str) else list(leaves(out.get("document")))
        return st, list(leaves(out, skip=("document_kind",)))
    req = rec.get("request") or {}
    st = list(leaves(req.get("state")))
    allv = list(leaves(req, skip=("schema_version", "request_id", "id", "type")))
    for k, v in rec.items():
        if "rationale" in k:
            allv.extend(leaves(v))
    return st, allv


def scan_chunk(args):
    label, path, kind, start, end = args
    cache: dict = {}
    a13, a8 = G["a13"], G["a8"]
    ashort = G["ashort"]
    exact = G["exact"]
    hits = {"train": {"13": {}, "8": {}, "short": {}, "exact": {}}, "other": {"13": {}, "8": {}, "short": {}, "exact": {}}}
    stats = collections.Counter()
    splits = collections.Counter()
    rec_hit = collections.Counter()
    with open(path, "rb") as f:
        f.seek(start)
        if start:
            f.readline()
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            part = rec.get("partition", "train") if kind == "record" else "train"
            cls = "train" if part == "train" else "other"
            src = rec.get("source") or rec.get("batch") or "?"
            rid = rec.get("id") or rec.get("doc_id") or "?"
            stats[(part, "rows")] += 1
            splits[(src, str(rec.get("source_split")), part)] += 1
            st, allv = record_texts(rec, kind)
            attrib = (label, src, part, rid)
            H = hits[cls]
            seqs = [tok_ids(s, cache) for s in allv if isinstance(s, str)]
            seqs = [s for s in seqs if s]
            stats[(part, "tokens")] += sum(map(len, seqs))
            # exact: each leaf, and the whole state
            st_ids = [t for s in st if isinstance(s, str) for t in tok_ids(s, cache)]
            cands = [seq_hash(s) for s in seqs if len(s) >= 3]
            if len(st_ids) >= 3:
                cands.append(seq_hash(st_ids))
            any_hit = False
            for c in cands:
                if c in exact and c not in H["exact"]:
                    H["exact"][c] = attrib
                    any_hit = True
            if not seqs:
                continue
            arr = []
            for s in seqs:
                arr.extend(s)
                arr.append(int(SENT))
            arr = np.asarray(arr, dtype=np.uint64)
            for n, h in rolling(arr, N_BIG):
                if n == N_BIG:
                    ref, key = a13, "13"
                elif n == N_LINT:
                    ref, key = a8, "8"
                elif n in ashort:
                    ref, key = ashort[n], "short"
                else:
                    continue
                if not len(ref):
                    continue
                idx = np.searchsorted(ref, h)
                idx[idx >= len(ref)] = len(ref) - 1
                m = h[ref[idx] == h]
                if len(m):
                    d = H[key]
                    for x in m.tolist():
                        if x not in d:
                            d[x] = attrib
                    if key == "13":
                        any_hit = True
            if any_hit:
                rec_hit[(label, src, part)] += 1
    return label, hits, stats, splits, rec_hit


def chunks(path: Path, size: int):
    total = path.stat().st_size
    return [(s, min(s + size, total)) for s in range(0, total, size)]


def top_examples(examples, k):
    """The k strongest flagged rows per (scope, task, text kind): highest 13-gram share, then exact, then raw hit count."""
    groups = collections.defaultdict(list)
    for e in examples:
        groups[(e["scope"], e["task"], e["text"])].append(e)
    out = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda e: (-e["frac13"], -int(e["exact"]), -e["short"], -e["h13"]))
        out.extend(g[:k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk-mb", type=int, default=48)
    ap.add_argument("--out", default=str(OUT_DIR / "contamination-results.json"))
    a = ap.parse_args()
    t0 = time.time()

    sys.path.insert(0, str(SCRATCH / "decision-bench/src"))
    try:
        from decision_bench.task_spec import _CANONICAL_TASKS  # type: ignore
        for name, tn, fam, dom, prim in _CANONICAL_TASKS:
            PUBLIC[(tn, dom, prim.value if hasattr(prim, "value") else str(prim))] = name
    except Exception as e:  # pragma: no cover
        print("warning: task table not importable:", e, file=sys.stderr)

    rows = load_db()
    g13, g8, short, exact = build_index(rows)
    G["a13"] = np.array(sorted(g13), dtype=np.uint64)
    G["a8"] = np.array(sorted(g8), dtype=np.uint64)
    G["ashort"] = {L: np.array(sorted(v), dtype=np.uint64) for L, v in short.items()}
    G["exact"] = exact
    print(f"DB rows {len(rows)}; 13-grams {len(g13):,}; 8-grams {len(g8):,}; short spans "
          f"{sum(map(len, short.values())):,}; exact keys {len(exact):,}; {time.time()-t0:.0f}s", flush=True)

    manifests = []
    jobs = []
    for label, path, role, kind in SCAN:
        if not path.is_file():
            manifests.append({"label": label, "path": str(path), "role": role, "missing": True})
            continue
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                sha.update(blk)
        manifests.append({"label": label, "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
                          "role": role, "bytes": path.stat().st_size, "sha256": sha.hexdigest()})
        for s, e in chunks(path, a.chunk_mb << 20):
            jobs.append((label, str(path), kind, s, e))
    print(f"{len(jobs)} chunks; hashing done {time.time()-t0:.0f}s", flush=True)

    role_of = {m["label"]: m["role"] for m in manifests}
    merged = {c: {k: {} for k in ("13", "8", "short", "exact")} for c in ("train", "other")}
    stats = collections.defaultdict(collections.Counter)
    splits = collections.defaultdict(collections.Counter)
    rec_hit = collections.Counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for i, (label, hits, st, sp, rh) in enumerate(pool.imap_unordered(scan_chunk, jobs)):
            for c in hits:
                # selection-only files never count as "train", whatever their partition says
                tgt = "other" if role_of[label] == "selection" else c
                for k, d in hits[c].items():
                    for x, at in d.items():
                        merged[tgt][k].setdefault(x, at)
            stats[label].update(st)
            splits[label].update(sp)
            rec_hit.update(rh)
            if i % 10 == 0:
                print(f"  {i+1}/{len(jobs)} chunks {time.time()-t0:.0f}s", flush=True)

    for m in manifests:
        if m.get("missing"):
            continue
        s = stats[m["label"]]
        m["rows_by_partition"] = {p: n for (p, k), n in sorted(s.items()) if k == "rows"}
        m["rows"] = sum(m["rows_by_partition"].values())
        m["text_tokens"] = sum(n for (p, k), n in s.items() if k == "tokens")

    anyp = {k: {**merged["other"][k], **merged["train"][k]} for k in merged["train"]}

    def row_metrics(row, kind, H):
        g = row[f"{kind}13"]
        h13 = [x for x in g if x in H["13"]]
        h8 = [x for x in row[f"{kind}8"] if x in H["8"]]
        hs = [x for x in row[f"{kind}short"] if x in H["short"]]
        ex = row[f"{kind}exact"] is not None and row[f"{kind}exact"] in H["exact"]
        at = collections.Counter(H["13"][x][:3] for x in h13)
        at.update(H["short"][x][:3] for x in hs)
        if ex:
            at[H["exact"][row[f"{kind}exact"]][:3]] += 1
        leaf_nd = 0
        q_hit = False
        paths = row["s_paths"] if kind == "s" else [""] * len(row[f"{kind}leaf13"])
        for pth, lg, ls in zip(paths, row[f"{kind}leaf13"], row[f"{kind}leafshort"]):
            lh = sum(1 for x in lg if x in H["13"])
            if lg and lh / len(lg) > 0.5:
                leaf_nd += 1
            if pth.split(".")[-1] in QUESTION_KEYS and pth.count(".") == 1 and (lh > 0 or (ls is not None and ls in H["short"])):
                q_hit = True
        return {"leaf_nd": leaf_nd, "q_hit": q_hit, "n13": len(g), "h13": len(h13), "frac13": (len(h13) / len(g)) if g else 0.0, "h8": len(h8),
                "nshort": len(row[f"{kind}short"]), "hshort": len(hs), "exact": ex, "attrib": at,
                "example_ids": sorted({H["13"][x][3] for x in h13} | {H["short"][x][3] for x in hs}
                                      | ({H["exact"][row[f'{kind}exact']][3]} if ex else set()))[:3]}

    per_task = collections.OrderedDict()
    examples = []
    for row in rows:
        pub = PUBLIC.get((row["task_name"], row["domain"], row["primitive"]), row["task_name"])
        T = per_task.setdefault(pub, {"rows": 0, "rows_with_13gram_capacity": 0, "raw_rows": 0})
        T["rows"] += 1
        T["rows_with_13gram_capacity"] += bool(row["s13"])
        T["raw_rows"] += bool(row["r"])
        for scope, H in (("train", merged["train"]), ("any", anyp)):
            for kind in ("s", "r"):
                if kind == "r" and not row["r"]:
                    continue
                mt = row_metrics(row, kind, H)
                pre = f"{scope}_{'state' if kind == 's' else 'raw'}"
                T[f"{pre}_13any"] = T.get(f"{pre}_13any", 0) + (mt["h13"] > 0)
                T[f"{pre}_near_dup"] = T.get(f"{pre}_near_dup", 0) + (mt["frac13"] > 0.5)
                T[f"{pre}_8ge2"] = T.get(f"{pre}_8ge2", 0) + (mt["h8"] >= 2)
                T[f"{pre}_short_contained"] = T.get(f"{pre}_short_contained", 0) + (mt["hshort"] > 0)
                T[f"{pre}_exact"] = T.get(f"{pre}_exact", 0) + mt["exact"]
                T[f"{pre}_leaf_near_dup"] = T.get(f"{pre}_leaf_near_dup", 0) + (mt["leaf_nd"] > 0)
                if kind == "s":
                    T[f"{pre}_question_leaf_hit"] = T.get(f"{pre}_question_leaf_hit", 0) + mt["q_hit"]
                flagged = mt["h13"] > 0 or mt["hshort"] > 0 or mt["exact"]
                T[f"{pre}_flagged"] = T.get(f"{pre}_flagged", 0) + flagged
                a_ = T.setdefault(f"{pre}_sources", collections.Counter())
                for k_, v_ in mt["attrib"].items():
                    a_[" | ".join(k_)] += 1
                if flagged:
                    examples.append({"scope": scope, "text": "state" if kind == "s" else "raw_upstream", "task": pub,
                                     "row_id": row["row_id"], "h13": mt["h13"], "n13": mt["n13"],
                                     "frac13": round(mt["frac13"], 3), "h8": mt["h8"], "short": mt["hshort"],
                                     "exact": mt["exact"], "leaf_near_dup": mt["leaf_nd"], "sources": {" | ".join(k): v for k, v in mt["attrib"].items()},
                                     "train_ids": mt["example_ids"],
                                     "snippet": (row["state_text"] if kind == "s" else row["raw_text"])[:300]})
    for T in per_task.values():
        for k in list(T):
            if k.endswith("_sources"):
                T[k] = dict(T[k].most_common(8))

    out = {
        "benchmark": {"dataset": "Hanno-Labs/decision-bench", "revision": "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443",
                      "split": "eval", "rows": len(rows), "parquet": str(DB_PARQUET),
                      "distinct_13grams": len(g13), "distinct_8grams": len(g8)},
        "manifests": manifests,
        "matched_gram_counts": {c: {k: len(v) for k, v in merged[c].items()} for c in merged},
        "training_records_with_13gram_hit": {" | ".join(k): v for k, v in rec_hit.most_common()},
        "source_splits_of_named_overlap_sources": {
            lab: {" | ".join(k): v for k, v in sp.items()
                  if any(s in k[0].lower() for s in ("civil", "clinc", "esci", "banking", "msmarco", "folio", "musique",
                                                     "finqa", "patent", "legalbench", "toolace", "nemotron", "tat",
                                                     "gsm8k", "eikos"))}
            for lab, sp in splits.items()},
        "per_task": per_task,
        "targeted_item_checks": targeted(rows),
        "flagged_examples": top_examples(examples, 5),
        "seconds": round(time.time() - t0),
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {a.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
