"""Phase-3 bundle transfer: build a checked bundle on the Mac, push it to a private store, fetch it on a pod, delete it after.

One interface, two backends chosen at run time (docs/phase-3-plan.md "Efficiency rules": nothing large goes Mac -> pod):
  hf   a PRIVATE Hugging Face dataset repo. The Mac uses the owner's cached login (`.venv/bin/hf auth login`); the pod reads
       the token from an environment variable (default HF_TOKEN), never from an argument, a file in the tree or a log line.
       The repo is created private, the bundle uploaded, fetched on the pod, and deleted with `delete` after the pull.
  gcs  a GCS bucket via `gcloud storage cp` (Mac: needs a fresh `gcloud auth login`; pod: gcloud if installed, else the
       google-cloud-storage client with GOOGLE_APPLICATION_CREDENTIALS pointing at a short-lived reader key).

Bundles are checked before they can be pushed. The check is HARD (the build fails, nothing is written):
  * no path under data/p3/decontam-sources/ or any other benchmark location (JevBench datasets, DecisionBench, fast-decisions,
    ImajevBench records/assets incl. the hidden split private-1, reports/benchmarks, the authored JevBench-style dev set);
  * no file whose sha256 equals a pinned benchmark file (cloud/p3/benchmark_pins.json);
  * no JSON/JSONL string (>= --min-chars after whitespace normalisation) that equals a benchmark string (states, instructions,
    options) read from the local benchmark sources -- this catches benchmark rows copied into any other file.
The one deliberate exception is the `gatekit` preset: the ImajevBench eval records (test gold) + assets and the authored dev
set, which the pod's gates need and which are not public. A gatekit may contain ONLY those files (allow-list), is unpacked
outside the training tree, and still fails on decontam-sources, JevBench, DecisionBench, fast-decisions and private-1.

    python scripts/p3/transfer.py build --preset train --name p3-train --manifest decision-p3 --manifest decision-p3-heldout-fresh ...
    python scripts/p3/transfer.py push  --backend hf  --name p3-train            # prints the private repo id
    python scripts/p3/transfer.py fetch --backend hf  --repo <id> --name p3-train --dest fetch --extract 
    python scripts/p3/transfer.py delete --backend hf --repo <id>
    python scripts/p3/transfer.py push  --backend gcs --bucket gs://<bucket>/ --name p3-mine   # this run: GCS, prefix phase3/
    python scripts/p3/transfer.py check PATH...                                    # the benchmark check on its own
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
# Repo root when run from scripts/p3/; when copied alone onto a pod (e.g. transfer.py) the parents do not exist,
# so fall back to IMAJEV_ROOT or the file's own folder (fetch/push/delete need no repo files; build runs on the Mac).
ROOT = Path(os.environ["IMAJEV_ROOT"]) if os.environ.get("IMAJEV_ROOT") else (_HERE.parents[2] if len(_HERE.parents) > 2 and (_HERE.parents[2] / "scripts").is_dir() else _HERE.parent)
PINS = ROOT / "cloud/p3/benchmark_pins.json"
OUT_DIR = ROOT / ".cache/p3-bundles"          # local build output (never inside a bundle: .cache is excluded)
PART_BYTES = 8 * 1024 ** 3                    # split archives into <= 8 GB parts (HF LFS files must stay < 50 GB)
TEXT_SUFFIXES = {".jsonl", ".json", ".txt", ".md", ".csv", ".tsv"}

# Paths that are benchmark text or benchmark copies. Matched against the bundle-relative (arcname) path AND the source path.
FORBIDDEN_PATTERNS = (
    "data/p3/decontam-sources", "*/decontam-sources/*", "decontam-sources/*",
    ".cache/*", "*/.cache/*",
    "*jevbench/datasets/*", "*/jevbench-public/*",
    "*decision-bench*", "data/*decisionbench*", "*tracking-only/*",
    "*fast-decisions*", "*fast_decisions*",
    "data/imajev-bench/*", "*imajev-bench/*", "bench/*", "*/private-1/*", "*private-1*",
    "reports/benchmarks/*",
    "*decision-p2b-jevstyle-dev*",
)
# Walking never descends into these (they are also forbidden, so an explicit --include of them still fails the check).
SKIP_DIRS = {".git", ".venv", "huggingface-cache", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".DS_Store", ".cache",
             ".cutouts"}
# Rebuildable caches that never travel, whatever the preset or --include (e.g. data/p3/images/inventory/.cutouts, 474 MB)
NEVER_BUNDLE = ("*/.cutouts/*", ".cutouts/*")
GATEKIT_ALLOWED = ("bench/records/records-eval.jsonl", "bench/assets/*", "manifests/decision-p2b-jevstyle-dev.jsonl",
                   "manifests/decision-p2b-jevstyle-dev-images.json", "GATEKIT.json")
GATEKIT_BENCH_KEYS = ("imajevbench-public", "jevstyle-dev-ours")   # the only benchmark content a gatekit may hold

CODE_PATHS = ("src", "scripts", "cloud", "artifacts", "pyproject.toml")

# Local benchmark sources (Mac) for the fingerprint check: key -> list of paths (files or dirs).
BENCH_SOURCES = {
    "jevbench-public": [".cache/external/jevbench/datasets/public", "data/p3/decontam-sources/jevbench"],
    "jevstyle-dev-ours": ["data/manifests/decision-p2b-jevstyle-dev.jsonl", "data/p3/decontam-sources/jevstyle-dev"],
    "decisionbench-1.0": ["data/p3/decontam-sources/decision-bench"],
    "fast-decisions-dev": ["data/p3/decontam-sources/fast-decisions"],
    "imajevbench-public": ["data/imajev-bench/v2-lite-v1/records-audited.jsonl", "data/p3/decontam-sources/imajev-bench"],
    "imajevbench-private-1": ["data/imajev-bench/private-1/records-audited.jsonl"],
}
DECISIONBENCH_HF = ("Hanno-Labs/decision-bench", "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443", "data/eval-00000-of-00001.parquet")


class BundleError(RuntimeError):
    """The bundle failed its hard check; nothing is written or pushed."""


def log(*a):
    print(time.strftime("%H:%M:%S"), "transfer:", *a, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def iter_strings(value, depth=0):
    """Every string inside a JSON value; strings that are themselves JSON objects/arrays are parsed and walked too."""
    if depth > 12:
        return
    if isinstance(value, str):
        yield value
        t = value.strip()
        if len(t) > 2 and t[0] in "[{" and t[-1] in "]}":
            try:
                inner = json.loads(t)
            except ValueError:
                return
            yield from iter_strings(inner, depth + 1)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from iter_strings(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            yield from iter_strings(v, depth + 1)


def json_values(path: Path):
    """JSON values of a .json / .jsonl file (unparseable lines are yielded as raw strings)."""
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            yield json.loads(text)
            return
        except ValueError:
            pass
    for line in text.split("\n"):
        if line.strip():
            try:
                yield json.loads(line)
            except ValueError:
                yield line


@dataclass
class BenchmarkIndex:
    """What counts as benchmark content: sha256 pins, fingerprints of long strings, and the key each came from."""
    min_chars: int = 80
    shas: dict = field(default_factory=dict)          # sha256 -> benchmark key
    prints: dict = field(default_factory=dict)        # sha1(normalised string) -> benchmark key
    sources: dict = field(default_factory=dict)       # key -> number of strings indexed

    def add_string(self, key: str, s: str):
        n = norm_text(s)
        if len(n) >= self.min_chars:
            self.prints.setdefault(hashlib.sha1(n.encode()).hexdigest(), key)
            self.sources[key] = self.sources.get(key, 0) + 1

    def add_file(self, key: str, path: Path):
        self.shas.setdefault(sha256_file(path), key)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            for col in table.column_names:
                if str(table.schema.field(col).type) in ("string", "large_string"):
                    for s in table.column(col).to_pylist():
                        if isinstance(s, str):
                            for x in iter_strings(s):
                                self.add_string(key, x)
        elif path.suffix in (".json", ".jsonl"):
            for v in json_values(path):
                for s in iter_strings(v):
                    self.add_string(key, s)
        elif path.suffix in (".jpg", ".jpeg", ".png", ".webp"):
            pass  # sha only

    def add_path(self, key: str, path: Path):
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.is_file() and not p.name.startswith("._"):
                    self.add_file(key, p)
        elif path.is_file():
            self.add_file(key, path)

    @classmethod
    def load(cls, root: Path = ROOT, min_chars: int = 80, pins: Path = PINS, require: bool = True, hf_cache: bool = True) -> "BenchmarkIndex":
        idx = cls(min_chars=min_chars)
        if pins.exists():
            p = json.loads(pins.read_text())
            for key, b in p.get("benchmarks", {}).items():
                for sha in b.get("files", {}).values():
                    idx.shas.setdefault(sha, key)
        found = 0
        for key, rels in BENCH_SOURCES.items():
            for rel in rels:
                path = (root / rel).resolve()
                if path.exists():
                    idx.add_path(key, path); found += 1
            if key == "imajevbench-public":   # the image assets of ImajevBench (sha only)
                for d in ("data/imajev-bench/v2-lite-v1/assets", "data/imajev-bench/private-1/assets"):
                    if (root / d).is_dir():
                        idx.add_path(key if "private" not in d else "imajevbench-private-1", root / d)
        if hf_cache and "decisionbench-1.0" not in idx.sources:  # the HF cache copy, if the decontam copy is absent
            try:
                from huggingface_hub import hf_hub_download
                repo, rev, fname = DECISIONBENCH_HF
                idx.add_file("decisionbench-1.0", Path(hf_hub_download(repo, fname, repo_type="dataset", revision=rev, local_files_only=True)))
                found += 1
            except Exception:
                pass
        if require and not idx.prints:
            raise BundleError("no benchmark sources found to fingerprint; refusing to certify a bundle blind "
                              "(run on the Mac where data/p3/decontam-sources exists, or pass --no-require-sources for tests)")
        return idx


def forbidden_path(rel: str) -> str | None:
    rel = rel.replace(os.sep, "/")
    while rel.startswith("./"):
        rel = rel[2:]
    for pat in FORBIDDEN_PATTERNS:
        if fnmatch.fnmatch(rel, pat) or ("*" not in pat and (rel == pat or rel.startswith(pat + "/"))):
            return pat
    return None


@dataclass
class Entry:
    src: Path
    arc: str


def check_entries(entries: list[Entry], index: BenchmarkIndex, gatekit: bool = False, root: Path = ROOT) -> dict:
    """The hard check. Returns a report; raises BundleError listing every problem."""
    problems, scanned_strings, hits = [], 0, {}
    allowed_keys = set(GATEKIT_BENCH_KEYS) if gatekit else set()
    for e in entries:
        try:
            src_rel = str(e.src.resolve().relative_to(root.resolve()))
        except ValueError:
            src_rel = str(e.src)
        if gatekit:
            if not any(fnmatch.fnmatch(e.arc, pat) for pat in GATEKIT_ALLOWED):
                problems.append(f"{e.arc}: not on the gatekit allow-list")
            for rel in (e.arc, src_rel):
                if "decontam-sources" in rel or "private-1" in rel:
                    problems.append(f"{e.arc}: forbidden in every bundle ({rel})")
        else:
            for rel in (e.arc, src_rel):
                pat = forbidden_path(rel)
                if pat:
                    problems.append(f"{e.arc}: benchmark path ({rel} matches {pat})"); break
        sha = sha256_file(e.src)
        key = index.shas.get(sha)
        if key and key not in allowed_keys:
            problems.append(f"{e.arc}: identical to a pinned benchmark file ({key})")
        if e.src.suffix in (".json", ".jsonl"):
            for v in json_values(e.src):
                for s in iter_strings(v):
                    n = norm_text(s)
                    if len(n) < index.min_chars:
                        continue
                    scanned_strings += 1
                    k = index.prints.get(hashlib.sha1(n.encode()).hexdigest())
                    if k and k not in allowed_keys:
                        hits.setdefault(e.arc, {}).setdefault(k, 0)
                        hits[e.arc][k] += 1
    for arc, by in hits.items():
        problems.append(f"{arc}: contains benchmark text {by}")
    report = {"files": len(entries), "bytes": sum(e.src.stat().st_size for e in entries), "strings_scanned": scanned_strings,
              "benchmark_strings_indexed": len(index.prints), "benchmark_shas_indexed": len(index.shas),
              "indexed_sources": dict(index.sources), "gatekit": gatekit, "problems": problems[:200], "status": "FAIL" if problems else "PASS"}
    if problems:
        raise BundleError(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems[:50]))
    return report


# ------------------------------------------------------------------------------------------------ collecting files
def walk(src: Path, arc_prefix: str, excludes: tuple[str, ...]) -> list[Entry]:
    out = []
    if src.is_file():
        return [Entry(src, arc_prefix)]
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("._"))
        for fn in sorted(filenames):
            if fn.startswith("._") or fn == ".DS_Store" or fn.endswith(".pyc"):
                continue
            p = Path(dirpath) / fn
            arc = (arc_prefix.rstrip("/") + "/" + str(p.relative_to(src))).lstrip("/")
            if any(fnmatch.fnmatch(arc, x) for x in excludes):
                continue
            out.append(Entry(p, arc))
    return out


def manifest_entries(name: str, root: Path = ROOT) -> list[Entry]:
    """A trainer manifest (data/manifests/<name>.jsonl), its -images.json index, and every image file its rows reference."""
    man = root / "data/manifests" / f"{name}.jsonl"
    if not man.exists():
        raise BundleError(f"manifest {man} does not exist")
    out = [Entry(man, f"data/manifests/{name}.jsonl")]
    rows = [json.loads(l) for l in man.read_text().split("\n") if l.strip()]
    idx = root / "data/manifests" / f"{name}-images.json"
    if idx.exists():
        out.append(Entry(idx, f"data/manifests/{name}-images.json"))
        images = json.loads(idx.read_text())["images"]
        rows = [dict(r, **images.get(r["id"], {})) for r in rows]
    paths = set()
    for r in rows:
        for im in r.get("images") or []:
            paths.add(im["image"])
        if "image" in r:
            paths.add(r["image"])
    missing = [p for p in sorted(paths) if not (root / p).exists()]
    if missing:
        raise BundleError(f"manifest {name}: {len(missing)} referenced images missing, e.g. {missing[:3]}")
    out += [Entry(root / p, p) for p in sorted(paths)]
    return out


def candidate_entries(pattern: str, root: Path = ROOT) -> list[Entry]:
    """Phase-3 candidate rows (scripts/p3/candidate.py; e.g. the mining pool shards) + every image they reference (relative to data/)."""
    import glob as _glob
    files = sorted(Path(p) for p in _glob.glob(str(root / pattern)))
    if not files:
        raise BundleError(f"no candidate files match {pattern}")
    out, images = [], set()
    for f in files:
        out.append(Entry(f, str(f.resolve().relative_to(root.resolve()))))
        for line in f.read_text().split("\n"):
            if line.strip():
                images.update(json.loads(line).get("images") or [])
    missing = [p for p in sorted(images) if not (root / "data" / p).exists()]
    if missing:
        raise BundleError(f"{pattern}: {len(missing)} referenced images missing under data/, e.g. {missing[:3]}")
    return out + [Entry(root / "data" / p, f"data/{p}") for p in sorted(images)]


def collect(preset: str, includes: list[str], manifests: list[str], excludes: tuple[str, ...], root: Path = ROOT,
            candidates: list[str] = ()) -> list[Entry]:
    entries: list[Entry] = []
    if preset in ("train", "mine", "code"):
        for rel in CODE_PATHS:
            if (root / rel).exists():
                entries += walk(root / rel, rel, excludes)
    if preset == "gatekit":
        rec = root / "data/imajev-bench/v2-lite-v1/records-audited.jsonl"
        entries.append(Entry(rec, "bench/records/records-eval.jsonl"))
        entries += walk(root / "data/imajev-bench/v2-lite-v1/assets", "bench/assets", excludes)
        dev = root / "data/manifests/decision-p2b-jevstyle-dev.jsonl"
        if dev.exists():
            entries.append(Entry(dev, "manifests/decision-p2b-jevstyle-dev.jsonl"))
    for m in manifests:
        entries += manifest_entries(m, root)
    for c in candidates:
        entries += candidate_entries(c, root)
    for inc in includes:  # SRC or SRC=ARCNAME, relative to the repo root
        src, _, arc = inc.partition("=")
        p = (root / src) if not os.path.isabs(src) else Path(src)
        if not p.exists():
            raise BundleError(f"include {src} does not exist")
        entries += walk(p, arc or src, excludes)
    entries = [e for e in entries if not any(fnmatch.fnmatch(e.arc, pat) for pat in NEVER_BUNDLE)]
    seen, uniq = {}, []
    for e in entries:
        if e.arc in seen:
            if seen[e.arc] != e.src.resolve():
                raise BundleError(f"two sources for {e.arc}: {seen[e.arc]} and {e.src}")
            continue
        seen[e.arc] = e.src.resolve(); uniq.append(e)
    return uniq


# ------------------------------------------------------------------------------------------------ build
def build(name: str, entries: list[Entry], index: BenchmarkIndex, out_dir: Path, gatekit: bool = False, compresslevel: int = 3,
          part_bytes: int = PART_BYTES, root: Path = ROOT) -> dict:
    report = check_entries(entries, index, gatekit=gatekit, root=root)  # raises before anything is written
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz = out_dir / f"{name}.tgz"
    for old in out_dir.glob(f"{name}.tgz*"):
        old.unlink()
    files = []
    with tarfile.open(tgz, "w:gz", compresslevel=compresslevel, format=tarfile.PAX_FORMAT) as tar:
        for e in entries:
            info = tar.gettarinfo(str(e.src), arcname=e.arc)
            if info.islnk():   # the same inode seen under a second path (data/decision-v1 image dirs are hard-link farms): store a plain copy
                info.type = tarfile.REGTYPE; info.linkname = ""; info.size = e.src.stat().st_size
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.pax_headers = {}   # no macOS xattrs / owners
            with open(e.src, "rb") as f:
                tar.addfile(info, f)
            files.append({"path": e.arc, "bytes": info.size})
        if gatekit:
            data = json.dumps({"gatekit": True, "files": len(entries)}).encode()
            ti = tarfile.TarInfo("GATEKIT.json"); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))
    total = tgz.stat().st_size
    parts = []
    if total > part_bytes:
        with open(tgz, "rb") as f:
            k = 0
            while True:
                block = f.read(part_bytes)
                if not block:
                    break
                pp = out_dir / f"{name}.tgz.part{k:03d}"; pp.write_bytes(block)
                parts.append({"file": pp.name, "sha256": hashlib.sha256(block).hexdigest(), "bytes": len(block)}); k += 1
        archive_sha = sha256_file(tgz); tgz.unlink()
    else:
        archive_sha = sha256_file(tgz)
        parts.append({"file": tgz.name, "sha256": archive_sha, "bytes": total})
    manifest = {"name": name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gatekit": gatekit,
                "archive_sha256": archive_sha, "archive_bytes": total, "parts": parts, "check": report, "files": files}
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"built {name}: {len(files)} files, {total / 1e9:.2f} GB in {len(parts)} part(s); check PASS")
    return manifest


def load_manifest(name: str, out_dir: Path) -> dict:
    m = json.loads((out_dir / f"{name}.manifest.json").read_text())
    if m.get("check", {}).get("status") != "PASS":
        raise BundleError(f"{name}: manifest has no passing check; rebuild it")
    for p in m["parts"]:
        f = out_dir / p["file"]
        if not f.exists() or f.stat().st_size != p["bytes"]:
            raise BundleError(f"{name}: part {p['file']} missing or wrong size; rebuild")
    return m


# ------------------------------------------------------------------------------------------------ backends
def run(cmd: list[str], dry: bool):
    log("$", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


class HFBackend:
    """Private HF dataset repo. Token: the cached login on the Mac; on a pod the env var named by token_env."""

    def __init__(self, token_env: str = "HF_TOKEN", dry: bool = False):
        self.token_env, self.dry = token_env, dry

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=os.environ.get(self.token_env) or None)   # None -> the cached login (never printed)

    def default_repo(self, name: str) -> str:
        user = "<owner>" if self.dry else self._api().whoami()["name"]
        return f"{user}/imajev-{name}-{time.strftime('%Y%m%d%H%M%S')}"

    def push(self, name: str, out_dir: Path, repo: str | None) -> str:
        m = load_manifest(name, out_dir)
        repo = repo or self.default_repo(name)
        log(f"push {name} -> hf dataset {repo} (private)")
        if self.dry:
            for p in m["parts"]:
                log(f"  [dry] upload {p['file']} ({p['bytes'] / 1e9:.2f} GB)")
            return repo
        api = self._api()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo, repo_type="dataset")
        if not getattr(info, "private", False):
            raise BundleError(f"{repo} is not private; refusing to upload")
        for p in m["parts"] + [{"file": f"{name}.manifest.json"}]:
            api.upload_file(path_or_fileobj=str(out_dir / p["file"]), path_in_repo=p["file"], repo_id=repo, repo_type="dataset",
                            commit_message=f"{name}: {p['file']}")
        log(f"pushed {name} to {repo}")
        return repo

    def fetch(self, name: str, repo: str, dest: Path) -> Path:
        from huggingface_hub import hf_hub_download
        token = os.environ.get(self.token_env)
        if not token:
            raise BundleError(f"env {self.token_env} is empty: export the HF read token in the pod shell (never on a command line)")
        dest.mkdir(parents=True, exist_ok=True)
        mpath = Path(hf_hub_download(repo, f"{name}.manifest.json", repo_type="dataset", token=token, local_dir=dest))
        m = json.loads(mpath.read_text())
        for p in m["parts"]:
            hf_hub_download(repo, p["file"], repo_type="dataset", token=token, local_dir=dest)
        return dest

    def delete(self, repo: str):
        log(f"delete hf dataset {repo}")
        if not self.dry:
            self._api().delete_repo(repo, repo_type="dataset")


class GCSBackend:
    def __init__(self, dry: bool = False):
        self.dry = dry

    @staticmethod
    def _url(bucket: str, prefix: str, name: str = "") -> str:
        b = bucket if bucket.startswith("gs://") else f"gs://{bucket}"
        return f"{b.rstrip('/')}/{prefix.strip('/')}/{name}".rstrip("/")

    def push(self, name: str, out_dir: Path, bucket: str, prefix: str) -> str:
        m = load_manifest(name, out_dir)
        for p in m["parts"] + [{"file": f"{name}.manifest.json"}]:
            run(["gcloud", "storage", "cp", str(out_dir / p["file"]), self._url(bucket, prefix, p["file"])], self.dry)
        return self._url(bucket, prefix)

    def fetch(self, name: str, bucket: str, prefix: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        files = [f"{name}.manifest.json"]
        def get(fn):
            if shutil.which("gcloud"):
                run(["gcloud", "storage", "cp", self._url(bucket, prefix, fn), str(dest / fn)], self.dry)
            else:
                from google.cloud import storage
                cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                client = storage.Client.from_service_account_json(cred) if cred else storage.Client()
                b = client.bucket(bucket.replace("gs://", "").split("/")[0])
                b.blob(f"{prefix.strip('/')}/{fn}").download_to_filename(str(dest / fn))
        get(files[0])
        m = json.loads((dest / files[0]).read_text())
        for p in m["parts"]:
            get(p["file"])
        return dest

    def delete(self, bucket: str, prefix: str, name: str):
        run(["gcloud", "storage", "rm", self._url(bucket, prefix, f"{name}.*")], self.dry)


def verify_and_extract(name: str, src_dir: Path, extract_to: Path | None, gatekit_ok: bool = False) -> dict:
    m = json.loads((src_dir / f"{name}.manifest.json").read_text())
    if m.get("check", {}).get("status") != "PASS":
        raise BundleError(f"{name}: the bundle's manifest has no passing check")
    if m.get("gatekit") and not gatekit_ok:
        raise BundleError(f"{name} is a gatekit (benchmark gold); extract it only with --gatekit into the gate directory")
    for p in m["parts"]:
        got = sha256_file(src_dir / p["file"])
        if got != p["sha256"]:
            raise BundleError(f"{name}: {p['file']} sha256 mismatch ({got} != {p['sha256']})")
    tgz = src_dir / f"{name}.tgz"
    if len(m["parts"]) > 1:
        with open(tgz, "wb") as w:
            for p in m["parts"]:
                with open(src_dir / p["file"], "rb") as r:
                    shutil.copyfileobj(r, w, 1 << 24)
                (src_dir / p["file"]).unlink()
    if sha256_file(tgz) != m["archive_sha256"]:
        raise BundleError(f"{name}: archive sha256 mismatch")
    if extract_to is not None:
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym():
                    raise BundleError(f"{name}: unsafe member {member.name}")
                if member.islnk() and (member.linkname.startswith("/") or ".." in Path(member.linkname).parts or not member.linkname):
                    raise BundleError(f"{name}: unsafe hard link {member.name} -> {member.linkname}")   # in-archive hard links are fine
            try:
                tar.extractall(extract_to, filter="data")
            except TypeError:   # Python without extraction filters
                tar.extractall(extract_to)
        log(f"extracted {name} ({len(m['files'])} files) into {extract_to}")
    return m


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="collect, check (hard) and archive a bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--preset", choices=("train", "mine", "code", "gatekit", "none"), default="train",
                   help="train/mine/code: the code tree (src scripts cloud artifacts pyproject.toml) + --manifest/--include; gatekit: ImajevBench eval + authored dev only")
    b.add_argument("--manifest", action="append", default=[], help="data/manifests/<name>.jsonl + its images (repeatable)")
    b.add_argument("--candidates", action="append", default=[], help="glob of candidate-row files (e.g. data/p3/pool/shards/*.jsonl) + their images (repeatable)")
    b.add_argument("--include", action="append", default=[], help="SRC[=ARCNAME] file or directory relative to the repo root (repeatable)")
    b.add_argument("--exclude", action="append", default=[], help="glob on the bundle path (repeatable)")
    b.add_argument("--out-dir", type=Path, default=OUT_DIR)
    b.add_argument("--min-chars", type=int, default=80)
    b.add_argument("--compresslevel", type=int, default=3)
    b.add_argument("--list", action="store_true", help="print the file list and the check result; write nothing")
    c = sub.add_parser("check", help="run the benchmark check on files or directories (bundle path = repo-relative path)")
    c.add_argument("paths", nargs="+")
    c.add_argument("--min-chars", type=int, default=80)
    for name in ("push", "fetch", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=("hf", "gcs"), required=True)
        p.add_argument("--name", required=name != "delete")
        p.add_argument("--repo", help="hf: dataset repo id (push: default <user>/imajev-<name>-<timestamp>)")
        p.add_argument("--bucket", help="gcs: bucket, e.g. gs://<bucket>/")
        p.add_argument("--prefix", default="phase3", help="gcs: object prefix (this run: phase3/)")
        p.add_argument("--token-env", default="HF_TOKEN", help="hf on a pod: NAME of the env var holding the token (the value is never printed)")
        p.add_argument("--dry-run", action="store_true")
        if name == "push":
            p.add_argument("--out-dir", type=Path, default=OUT_DIR)
        if name == "fetch":
            p.add_argument("--dest", type=Path, required=True, help="download directory")
            p.add_argument("--extract", type=Path, help="extract the verified archive here")
            p.add_argument("--gatekit", action="store_true", help="allow extracting a gatekit bundle")
            p.add_argument("--keep", action="store_true", help="keep the archive after extraction")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "build":
            entries = collect(a.preset, a.include, a.manifest, tuple(a.exclude), candidates=a.candidates)
            if not entries:
                raise BundleError("empty bundle")
            index = BenchmarkIndex.load(min_chars=a.min_chars)
            if a.list:
                rep = check_entries(entries, index, gatekit=a.preset == "gatekit")
                for e in entries:
                    print(e.arc)
                print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1))
                return 0
            build(a.name, entries, index, a.out_dir, gatekit=a.preset == "gatekit", compresslevel=a.compresslevel)
            return 0
        if a.cmd == "check":
            entries = []
            for p in a.paths:
                path = Path(p).resolve()
                try:
                    rel = str(path.relative_to(ROOT))
                except ValueError:
                    rel = path.name
                entries += walk(path, rel, ()) if path.is_dir() else [Entry(path, rel)]
            rep = check_entries(entries, BenchmarkIndex.load(min_chars=a.min_chars))
            print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1))
            return 0
        if a.backend == "hf":
            be = HFBackend(a.token_env, a.dry_run)
            if a.cmd == "push":
                repo = be.push(a.name, a.out_dir, a.repo); print(repo); return 0
            if a.cmd == "fetch":
                if not a.repo:
                    raise BundleError("--repo is required for fetch")
                be.fetch(a.name, a.repo, a.dest)
            if a.cmd == "delete":
                if not a.repo:
                    raise BundleError("--repo is required for delete")
                be.delete(a.repo); return 0
        else:
            if not a.bucket:
                raise BundleError("--bucket is required for the gcs backend")
            be = GCSBackend(a.dry_run)
            if a.cmd == "push":
                print(be.push(a.name, a.out_dir, a.bucket, a.prefix)); return 0
            if a.cmd == "fetch":
                be.fetch(a.name, a.bucket, a.prefix, a.dest)
            if a.cmd == "delete":
                be.delete(a.bucket, a.prefix, a.name or "*"); return 0
        if a.cmd == "fetch" and not a.dry_run:
            verify_and_extract(a.name, a.dest, a.extract, gatekit_ok=a.gatekit)
            if a.extract and not a.keep:
                for f in a.dest.glob(f"{a.name}.tgz*"):
                    f.unlink()
        return 0
    except BundleError as exc:
        print(f"BUNDLE_CHECK_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
