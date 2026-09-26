"""Benchmark index for phase-3 decontamination (docs/phase-3-plan.md, "Decontamination, before mining").

Collects the text, upstream ids and image hashes of every evaluation set phase 3 must never train on, and stores them as
hashes only (no text) in data/p3/decontam-index/. Stable local copies of the public sources live in
data/p3/decontam-sources/ with sources.json (origin, revision, sha256, row counts). The ImajevBench hidden set private-1 is
read in place and never copied: only its hashes enter the index and nothing about its text is ever written out.

Normalisation is the one of reports/benchmarks/decisionbench/contamination_check.py (and the JevBench 8-gram lint):
lower-case, tokens = [a-z0-9]+, every string leaf tokenised on its own (n-grams never span two fields), token ids are
64-bit blake2b, n-gram ids are 64-bit polynomial hashes.

Per benchmark row we index up to three text units:
  state  - all string leaves of the state (the thing the model reads)
  qopts  - question / instruction plus option labels and descriptions
  raw    - DecisionBench's pre-paraphrase upstream text (source_json.source.raw), where present
and for each: the distinct 13-grams (with the unit they came from), every leaf of >= 6 tokens (exact-field match), and the
whole normalised state (exact-state match). Upstream ids come from DecisionBench `source_json`; image sha256 from
ImajevBench records.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES_DIR = ROOT / "data/p3/decontam-sources"
INDEX_DIR = ROOT / "data/p3/decontam-index"
SCRATCH = Path("/private/tmp/claude-501/-Volumes-SSD-9100-Development-imajev/0e2da7e0-1607-4801-b7a5-28f4760b900a/scratchpad")

N = 13                 # n-gram size for overlap
MIN_FIELD_TOKENS = 6   # exact-field rule
MIN_STATE_TOKENS = 3   # exact whole-state rule (as in the DecisionBench check)
KIND_STATE, KIND_QOPTS, KIND_RAW = 0, 1, 2
KIND_NAMES = ("state", "qopts", "raw")

TOKEN = re.compile(r"[a-z0-9]+")
P = np.uint64(0x9E3779B97F4A7C15)
SENT = 0xFFFFFFFFFFFFFFC5

DB_REPO, DB_REV = "Hanno-Labs/decision-bench", "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
DB_HF = Path.home() / f".cache/huggingface/hub/datasets--Hanno-Labs--decision-bench/snapshots/{DB_REV}/data/eval-00000-of-00001.parquet"
JEVBENCH_DIR = ROOT / ".cache/external/jevbench"
FASTDEC_SCRATCH = SCRATCH / "fastdec"
IMB_PUBLIC = ROOT / "data/imajev-bench/v2-lite-v1/records-audited.jsonl"
IMB_RELEASE = ROOT.parent / "imajev-release/bench/records/records-public.jsonl"
IMB_PRIVATE = ROOT / "data/imajev-bench/private-1/records-audited.jsonl"
JEVSTYLE_DEV = ROOT / "data/manifests/decision-p2b-jevstyle-dev.jsonl"

# Benchmark names used in reports and drops.jsonl.
B_JEV = "jevbench-public"
B_JEVDEV = "jevstyle-dev-ours"
B_DB = "decisionbench-1.0"
B_FD = "fast-decisions-dev"
B_IMB = "imajevbench-public"
B_PRIV = "imajevbench-private-1"
PRIVATE_BENCHES = frozenset({B_PRIV})


# ---------------------------------------------------------------- tokenisation and hashing
class Tok:
    """Token -> 64-bit id with a per-process cache."""

    def __init__(self):
        self.cache: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        out = []
        c = self.cache
        for t in TOKEN.findall(text.lower()):
            v = c.get(t)
            if v is None:
                v = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
                if v == SENT:
                    v ^= 1
                c[t] = v
            out.append(v)
        return out


def seq_hash(ids) -> int:
    """64-bit hash of a whole token sequence (exact-field / exact-state keys)."""
    return int.from_bytes(hashlib.blake2b(np.asarray(ids, dtype=np.uint64).tobytes(), digest_size=8).digest(), "little")


def grams(leaf_ids: list[list[int]], n: int = N) -> np.ndarray:
    """Distinct n-gram hashes of a unit; n-grams never span two leaves."""
    leaf_ids = [x for x in leaf_ids if len(x) >= n]
    if not leaf_ids:
        return np.empty(0, dtype=np.uint64)
    arr, sent = [], []
    for x in leaf_ids:
        arr.extend(x)
        arr.append(SENT)
    a = np.asarray(arr, dtype=np.uint64)
    is_s = (a == np.uint64(SENT)).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(is_s)])
    h = a.copy()
    with np.errstate(over="ignore"):
        for k in range(2, n + 1):
            h = h[:-1] * P + a[k - 1:]
    valid = (cs[n:] - cs[:-n]) == 0
    return np.unique(h[valid])


def leaves(value, skip=()) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in skip:
                yield from leaves(v, skip)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v, skip)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def ds_alias(name: str | None) -> str:
    """Dataset name -> join key: last path component, lower-case, alphanumerics only, a few aliases."""
    s = re.sub(r"[^a-z0-9]", "", str(name or "").split("/")[-1].lower())
    return {"clincoos": "clinc150", "clinc": "clinc150", "banking": "banking77", "finqadataset": "finqa",
            "musiqueans": "musique", "musiquefull": "musique", "shoppingqueriesdataset": "esci",
            "folio2": "folio", "foliov2": "folio"}.get(s, s)


def upstream_keys(dataset: str | None, uid, split: str | None = None) -> list[str]:
    """Join keys for one upstream item. Index-like (purely numeric) ids only join together with their split."""
    out = []
    a = ds_alias(dataset)
    ids = uid if isinstance(uid, list) else [uid]
    for i in ids:
        if i is None or str(i).strip() == "":
            continue
        i = str(i).strip()
        sp = str(split or "").split("/")[-1].lower()
        if i.isdigit():
            if sp:
                out.append(f"{a}|{sp}|{i}")
        else:
            out.append(f"{a}||{i}")
            if a == "finqa":  # FinQA: one filing page carries several questions -> the page is the upstream item
                page = re.sub(r"-\d+$", "", i)
                out.append(f"{a}|page|{page}")
    return out


# ---------------------------------------------------------------- benchmark rows
@dataclass
class BenchRow:
    bench: str
    row_id: str
    state: list[str] = field(default_factory=list)
    qopts: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    upstream: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # sha256 hex
    family: str | None = None  # leakage mode: parent family root


def _jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _request_row(bench: str, r: dict) -> BenchRow:
    """Rows in the imajev request format (ImajevBench records, our manifests)."""
    req = r.get("request") or {}
    q = []
    for f in req.get("fields") or []:
        q.extend(leaves({k: f.get(k) for k in ("question", "options", "levels", "yes_description", "no_description",
                                                 "criteria", "labels")}, skip=("id",)))
    imgs = [i.get("sha256") for i in r.get("images") or [] if isinstance(i, dict) and i.get("sha256")]
    return BenchRow(bench, str(r.get("id")), state=list(leaves(req.get("state"))), qopts=q, images=imgs)


def rows_jevbench(src_dir: Path) -> Iterator[BenchRow]:
    for p in sorted(src_dir.glob("*.jsonl")):
        for r in _jsonl(p):
            q = list(leaves(r.get("question"), skip=("type",))) + list(leaves(r.get("labels")))
            yield BenchRow(B_JEV, f"{p.stem}:{r['id']}", state=list(leaves(r.get("state"))), qopts=q)


def rows_jevstyle_dev(path: Path) -> Iterator[BenchRow]:
    for r in _jsonl(path):
        yield _request_row(B_JEVDEV, r)


def rows_decisionbench(parquet: Path) -> Iterator[BenchRow]:
    import pyarrow.parquet as pq
    for r in pq.read_table(parquet).to_pylist():
        state = json.loads(r["state_json"])
        cands = json.loads(r["candidates_json"] or "[]")
        src = json.loads(r["source_json"] or "{}")
        q = [r["instruction"] or ""] + [s for c in cands for s in leaves({"l": c.get("label"), "d": c.get("description")})]
        raw, ups = [], []
        s = src.get("source") if isinstance(src.get("source"), dict) else None
        if s is not None:
            if s.get("raw") is not None:
                raw = list(leaves(s["raw"], skip=("stratum", "label", "category", "label_key", "source_label",
                                                  "selected_tool", "description_sha256", "uid")))
            for i in s.get("source_ids") or []:
                ups += upstream_keys(s.get("dataset"), i, s.get("split"))
        elif src.get("source_id") is not None:
            ups += upstream_keys(src.get("repo"), src["source_id"], src.get("split"))
        yield BenchRow(B_DB, r["row_id"], state=list(leaves(state)), qopts=q, raw=raw, upstream=ups)


def rows_fastdec(src_dir: Path) -> Iterator[BenchRow]:
    for p in sorted(src_dir.glob("*.jsonl")):
        for i, r in enumerate(_jsonl(p)):
            q = []
            for c in (r.get("output") or {}).get("classifications") or []:
                q += [str(c.get("task") or "")] + [str(x) for x in c.get("labels") or []]
            st = list(leaves(r.get("input")))
            yield BenchRow(B_FD, f"{p.stem}:{i}", state=st, qopts=q)


def rows_imajevbench(path: Path, bench: str, seen: set | None = None) -> Iterator[BenchRow]:
    """`seen` skips rows whose text and images were already indexed (the release package repeats v2-lite-v1 under
    salted test ids)."""
    for r in _jsonl(path):
        br = _request_row(bench, r)
        if seen is not None:
            key = hashlib.sha256(json.dumps([br.state, br.qopts, br.images]).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
        if bench in PRIVATE_BENCHES:
            br.row_id = ""  # never carried anywhere
        yield br


# ---------------------------------------------------------------- sources: stable local copies
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def sync_sources(verbose: bool = True) -> dict:
    """Copy every public source into data/p3/decontam-sources/ (once) and (re)write sources.json."""
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": {}}

    def cp(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if not src.exists():
                raise FileNotFoundError(f"source missing and no local copy: {src}")
            shutil.copy2(src, dst)
            if verbose:
                print(f"copied {src} -> {dst.relative_to(ROOT)}")
        return dst

    # JevBench public
    jd = SOURCES_DIR / "jevbench"
    files = [cp(p, jd / p.name) for p in sorted((JEVBENCH_DIR / "datasets/public").glob("*.jsonl"))] if \
        (JEVBENCH_DIR / "datasets/public").exists() else sorted(jd.glob("*.jsonl"))
    if (JEVBENCH_DIR / "datasets/manifest.json").exists():
        cp(JEVBENCH_DIR / "datasets/manifest.json", jd / "manifest.json")
    rev = None
    head = JEVBENCH_DIR / ".git/HEAD"
    try:
        import subprocess
        rev = subprocess.run(["git", "-C", str(JEVBENCH_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    if rev is None and (jd / "REVISION").exists():
        rev = (jd / "REVISION").read_text().strip()
    if rev:
        (jd / "REVISION").write_text(rev + "\n")
    out["sources"][B_JEV] = {
        "origin": ".cache/external/jevbench/datasets/public (git clone of the JevBench repo)", "revision": rev,
        "note": "public splits only (original, easy, hard = 231 items); heldout/router/judge/easy-heldout are not public and not available locally",
        "files": {p.name: {"sha256": sha256_file(p), "rows": _count_lines(p)} for p in files}}
    # our JevBench-style dev
    d = cp(JEVSTYLE_DEV, SOURCES_DIR / "jevstyle-dev/decision-p2b-jevstyle-dev.jsonl")
    out["sources"][B_JEVDEV] = {"origin": "data/manifests/decision-p2b-jevstyle-dev.jsonl (scripts/p2/author_jevstyle_dev.py)",
                                "revision": "local", "files": {d.name: {"sha256": sha256_file(d), "rows": _count_lines(d)}}}
    # DecisionBench
    d = cp(DB_HF, SOURCES_DIR / "decision-bench/eval-00000-of-00001.parquet")
    import pyarrow.parquet as pq
    out["sources"][B_DB] = {"origin": f"hf://datasets/{DB_REPO} split eval", "revision": DB_REV,
                            "files": {d.name: {"sha256": sha256_file(d), "rows": pq.ParquetFile(d).metadata.num_rows}}}
    # fast-decisions dev
    fd = SOURCES_DIR / "fast-decisions"
    srcs = [p for p in sorted(FASTDEC_SCRATCH.glob("*.jsonl")) if not re.match(r"(preds|run|score)", p.name)] \
        if FASTDEC_SCRATCH.exists() else []
    for p in srcs:
        cp(p, fd / p.name)
    if (FASTDEC_SCRATCH / "README.md").exists():
        cp(FASTDEC_SCRATCH / "README.md", fd / "README.md")
    files = sorted(fd.glob("*.jsonl"))
    out["sources"][B_FD] = {"origin": "hf://datasets/fastino/fast-decisions dev split (17 domain files), copied from a scratch download",
                            "revision": "unrecorded (downloaded 2026-09-24/25; file sha256 below pin the content)",
                            "files": {p.name: {"sha256": sha256_file(p), "rows": _count_lines(p)} for p in files}}
    # ImajevBench public (all splits of v2-lite-v1 incl. gold) + the release package copy
    d = cp(IMB_PUBLIC, SOURCES_DIR / "imajev-bench/v2-lite-v1-records-audited.jsonl")
    files = {d.name: {"sha256": sha256_file(d), "rows": _count_lines(d)}}
    if IMB_RELEASE.exists() or (SOURCES_DIR / "imajev-bench/release-records-public.jsonl").exists():
        d2 = cp(IMB_RELEASE, SOURCES_DIR / "imajev-bench/release-records-public.jsonl")
        files[d2.name] = {"sha256": sha256_file(d2), "rows": _count_lines(d2)}
    out["sources"][B_IMB] = {"origin": "data/imajev-bench/v2-lite-v1/records-audited.jsonl (dev+calibration+test) and "
                                       "../imajev-release/bench/records/records-public.jsonl", "revision": "local", "files": files}
    # private-1: in place, never copied
    out["sources"][B_PRIV] = {"origin": "data/imajev-bench/private-1/records-audited.jsonl (read in place, NOT copied; "
                                        "hashes only in the index)", "revision": "local", "private": True,
                              "files": {IMB_PRIVATE.name: {"sha256": sha256_file(IMB_PRIVATE), "rows": _count_lines(IMB_PRIVATE)}}}
    (SOURCES_DIR / "sources.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def all_bench_rows() -> Iterator[BenchRow]:
    yield from rows_jevbench(SOURCES_DIR / "jevbench")
    yield from rows_jevstyle_dev(SOURCES_DIR / "jevstyle-dev/decision-p2b-jevstyle-dev.jsonl")
    yield from rows_decisionbench(SOURCES_DIR / "decision-bench/eval-00000-of-00001.parquet")
    yield from rows_fastdec(SOURCES_DIR / "fast-decisions")
    seen: set = set()
    for p in (SOURCES_DIR / "imajev-bench/v2-lite-v1-records-audited.jsonl", SOURCES_DIR / "imajev-bench/release-records-public.jsonl"):
        if p.exists():
            yield from rows_imajevbench(p, B_IMB, seen)
    assert seen, "no ImajevBench public records"
    yield from rows_imajevbench(IMB_PRIVATE, B_PRIV)


# ---------------------------------------------------------------- index
class Index:
    """Hash-only benchmark index. Arrays are sorted by key for np.searchsorted lookups."""

    def __init__(self, arrays: dict, meta: dict):
        self.a = arrays
        self.meta = meta
        self.g_hash, self.g_unit = arrays["g_hash"], arrays["g_unit"]
        self.unit_row, self.unit_kind, self.unit_n = arrays["unit_row"], arrays["unit_kind"], arrays["unit_n"]
        self.leaf_hash, self.leaf_row, self.leaf_rows = arrays["leaf_hash"], arrays["leaf_row"], arrays["leaf_nrows"]
        self.state_hash, self.state_row = arrays["state_hash"], arrays["state_row"]
        self.row_bench = arrays["row_bench"]
        self.benches: list[str] = meta["benches"]
        self.row_ids: list[str] = meta["row_ids"]
        self.row_family: list[str | None] = meta.get("row_family") or []
        self.upstream: dict[str, list[int]] = meta["upstream"]
        self.images: dict[str, list[int]] = meta["images"]
        self.private = {i for i, b in enumerate(self.benches) if b in PRIVATE_BENCHES}

    def bench_of(self, row: int) -> str:
        return self.benches[int(self.row_bench[row])]

    def public_row_id(self, row: int) -> str | None:
        return None if int(self.row_bench[row]) in self.private else self.row_ids[row]

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "index.npz", **self.a)
        (d / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, d: Path, mmap: bool = True) -> "Index":
        # np.savez is uncompressed: np.load with mmap_mode maps each member lazily
        z = np.load(d / "index.npz", mmap_mode="r" if mmap else None)
        arrays = {k: z[k] for k in z.files}
        return cls(arrays, json.loads((d / "meta.json").read_text()))


def build(rows: Iterable[BenchRow], extra_meta: dict | None = None) -> Index:
    tok = Tok()
    benches: list[str] = []
    bidx: dict[str, int] = {}
    row_bench, row_ids, row_family = [], [], []
    g_parts, gu_parts = [], []
    unit_row, unit_kind, unit_n = [], [], []
    leaf_h, leaf_r = [], []
    state_h, state_r = [], []
    upstream: dict[str, list[int]] = {}
    images: dict[str, list[int]] = {}
    for ri, br in enumerate(rows):
        if br.bench not in bidx:
            bidx[br.bench] = len(benches)
            benches.append(br.bench)
        row_bench.append(bidx[br.bench])
        row_ids.append("" if br.bench in PRIVATE_BENCHES else br.row_id)
        row_family.append(br.family)
        for kind, lv in ((KIND_STATE, br.state), (KIND_QOPTS, br.qopts), (KIND_RAW, br.raw)):
            ids = [tok.ids(s) for s in lv if isinstance(s, str)]
            ids = [x for x in ids if x]
            g = grams(ids)
            if len(g):
                u = len(unit_row)
                unit_row.append(ri); unit_kind.append(kind); unit_n.append(len(g))
                g_parts.append(g); gu_parts.append(np.full(len(g), u, dtype=np.int32))
            for x in ids:
                if len(x) >= MIN_FIELD_TOKENS:
                    leaf_h.append(seq_hash(x)); leaf_r.append(ri)
            if kind in (KIND_STATE, KIND_RAW):
                full = [t for x in ids for t in x]
                if len(full) >= MIN_STATE_TOKENS:
                    state_h.append(seq_hash(full)); state_r.append(ri)
        for k in br.upstream:
            upstream.setdefault(k, []).append(ri)
        for s in br.images:
            images.setdefault(s, []).append(ri)
    gh = np.concatenate(g_parts) if g_parts else np.empty(0, np.uint64)
    gu = np.concatenate(gu_parts) if gu_parts else np.empty(0, np.int32)
    o = np.lexsort((gu, gh))
    gh, gu = gh[o], gu[o]
    lh = np.asarray(leaf_h, dtype=np.uint64); lr = np.asarray(leaf_r, dtype=np.int32)
    o = np.lexsort((lr, lh)); lh, lr = lh[o], lr[o]
    keep = np.ones(len(lh), bool)
    if len(lh):
        keep[1:] = (lh[1:] != lh[:-1]) | (lr[1:] != lr[:-1])
    lh, lr = lh[keep], lr[keep]
    # rows per distinct leaf (template detection): count of rows sharing the same leaf hash
    if len(lh):
        uniq, start, cnt = np.unique(lh, return_index=True, return_counts=True)
        nrows = np.repeat(cnt, cnt).astype(np.int32)
    else:
        nrows = np.empty(0, np.int32)
    sh = np.asarray(state_h, dtype=np.uint64); sr = np.asarray(state_r, dtype=np.int32)
    o = np.argsort(sh, kind="stable"); sh, sr = sh[o], sr[o]
    arrays = {"g_hash": gh, "g_unit": gu, "unit_row": np.asarray(unit_row, np.int32),
              "unit_kind": np.asarray(unit_kind, np.int8), "unit_n": np.asarray(unit_n, np.int32),
              "leaf_hash": lh, "leaf_row": lr, "leaf_nrows": nrows, "state_hash": sh, "state_row": sr,
              "row_bench": np.asarray(row_bench, np.int16)}
    counts = {b: int(np.sum(arrays["row_bench"] == i)) for i, b in enumerate(benches)}
    meta = {"benches": benches, "row_ids": row_ids, "row_family": row_family, "upstream": upstream, "images": images,
            "bench_rows": counts, "n": N, "min_field_tokens": MIN_FIELD_TOKENS, "min_state_tokens": MIN_STATE_TOKENS,
            "n_units": len(unit_row), "n_gram_pairs": int(len(gh)), "n_leaves": int(len(lh)),
            "n_upstream_keys": len(upstream), "n_images": len(images), **(extra_meta or {})}
    return Index(arrays, meta)


def _fingerprint(sources: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(sources["sources"]):
        for fn, v in sorted(sources["sources"][name]["files"].items()):
            h.update(f"{name}/{fn}/{v['sha256']}".encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def get_index(rebuild: bool = False, verbose: bool = True) -> Index:
    """Load the cached index, (re)building it when the sources or this module changed."""
    sources = sync_sources(verbose=verbose)
    fp = _fingerprint(sources)
    meta_p = INDEX_DIR / "meta.json"
    if not rebuild and meta_p.exists():
        try:
            if json.loads(meta_p.read_text()).get("fingerprint") == fp:
                return Index.load(INDEX_DIR)
        except Exception:
            pass
    t0 = time.time()
    idx = build(all_bench_rows(), {"fingerprint": fp, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    idx.meta["build_seconds"] = round(time.time() - t0, 1)
    idx.save(INDEX_DIR)
    if verbose:
        print(f"built index in {idx.meta['build_seconds']}s: {idx.meta['bench_rows']}, {idx.meta['n_gram_pairs']:,} 13-gram "
              f"pairs, {idx.meta['n_leaves']:,} leaves, {idx.meta['n_upstream_keys']:,} upstream keys, {idx.meta['n_images']} images")
    return Index.load(INDEX_DIR)
