"""Pre-sample the image-decision replay slice for the phase-2c manifests (runs on the Mac, where the images live).

    python scripts/p2/make_image_replay.py --manifest data/manifests/decision-v2.1-4b.jsonl --n-fresh 12000 --n-delta 5000 --seed 7 \
        --out-dir data/decision-p2c/image-replay/

Picks TRAIN rows that carry at least one image, stratified by `source` (each source gets its share of the eligible rows, largest
remainder), deterministically: within a source, candidates are taken in sha256("<seed>\\0image-replay\\0<id>") order. A candidate is
skipped (and counted) when an image file is missing, the record fails the serving contract, or it hits the JevBench 8-gram lint.
`replay-delta.jsonl` is a subset of `replay-fresh.jsonl` (the first rows of each source's fresh selection, again proportional).
Records are copied unchanged (labels, targets, target_distribution(s)) except: image paths are rewritten to
`<out-dir>/images/<sha256>.<ext>` (files copied there once each) and `replay: "image-v2.1"` is added.
If the fresh slice's images exceed --max-bytes, n-fresh is reduced in 10% steps until they fit (reported).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
REPLAY_TAG = "image-v2.1"


def row_images(r: dict) -> list[dict]:
    return r.get("images") or ([{"image": r["image"], "sha256": r.get("sha256")}] if "image" in r else [])


def rank(seed, rid) -> str:
    return hashlib.sha256(f"{seed}\0image-replay\0{rid}".encode()).hexdigest()


def allocate(counts: dict, n: int) -> dict:
    """Largest-remainder proportional allocation of n over the keys of counts (never above a key's count)."""
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in counts}
    n = min(n, total)
    exact = {k: n * c / total for k, c in counts.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - out[k]), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def scan(manifest: Path, seed) -> dict:
    """source -> [(rank, byte offset)] for train rows with images, sorted by rank."""
    by_source = collections.defaultdict(list)
    with manifest.open("rb") as f:
        offset = 0
        for line in f:
            if b'"train"' in line:
                r = json.loads(line)
                if r.get("partition") == "train" and row_images(r):
                    by_source[r.get("source", "?")].append((rank(seed, r["id"]), offset))
            offset += len(line)
    for v in by_source.values():
        v.sort()
    return dict(by_source)


def make_checker(reference):
    from decision_data import expand_fields, render
    from p2_common import contamination_count
    def check(r):
        for p in (Path(im["image"]) for im in row_images(r)):
            if not (p if p.is_absolute() else ROOT / p).is_file():
                return "missing_image"
        try:
            for item in expand_fields(r): render(item)
        except Exception:
            return "contract"
        if reference:
            state = r["request"].get("state", "")
            text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
            if contamination_count(text, reference) + sum(contamination_count(f.get("question", ""), reference) for f in r["request"]["fields"]) >= 2:
                return "jevbench_lint"
        return None
    return check


def select(manifest: Path, by_source: dict, n: int, check, skipped: collections.Counter) -> dict:
    """source -> list of records (in rank order), n in total, stratified; skips failing candidates."""
    alloc = allocate({s: len(v) for s, v in by_source.items()}, n); out = {}
    with manifest.open("rb") as f:
        for s in sorted(by_source):
            got = []
            for _, off in by_source[s]:
                if len(got) >= alloc[s]:
                    break
                f.seek(off); r = json.loads(f.readline())
                why = check(r)
                if why:
                    skipped[why] += 1; continue
                got.append(r)
            out[s] = got
    return out


def image_bytes(rows) -> tuple[int, int]:
    seen = {}
    for r in rows:
        for im in row_images(r):
            p = Path(im["image"]); p = p if p.is_absolute() else ROOT / p
            seen[str(p)] = p.stat().st_size
    return len(seen), sum(seen.values())


def rewrite(r: dict, out_dir: Path, copied: dict) -> dict:
    r = json.loads(json.dumps(r)); ims = row_images(r); new = []
    for im in ims:
        src = Path(im["image"]); src_abs = src if src.is_absolute() else ROOT / src
        sha = im.get("sha256") or hashlib.sha256(src_abs.read_bytes()).hexdigest()
        dest = out_dir / "images" / f"{sha}{src.suffix.lower() or '.jpg'}"
        if str(dest) not in copied:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file():
                shutil.copyfile(src_abs, dest)
            copied[str(dest)] = dest.stat().st_size
        rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
        new.append(dict(im, image=str(rel), sha256=sha))
    if "images" in r and r["images"]:
        r["images"] = new
    else:
        r["image"] = new[0]["image"]; r["sha256"] = new[0]["sha256"]
    r["replay"] = REPLAY_TAG
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/decision-v2.1-4b.jsonl")
    ap.add_argument("--n-fresh", type=int, default=12000); ap.add_argument("--n-delta", type=int, default=5000)
    ap.add_argument("--seed", default="7"); ap.add_argument("--out-dir", type=Path, default=ROOT / "data/decision-p2c/image-replay")
    ap.add_argument("--max-bytes", type=float, default=6e9, help="cap on the fresh slice's image bytes; n-fresh shrinks in 10%% steps to fit")
    ap.add_argument("--reference", type=Path, nargs="*", help="JevBench public files (default: .cache/external/jevbench/datasets/public/*.jsonl)")
    a = ap.parse_args(argv)
    a.out_dir = a.out_dir.resolve()
    from p2_common import jevbench_public_files, reference_ngrams
    reference = reference_ngrams(a.reference if a.reference is not None else jevbench_public_files())
    check = make_checker(reference)
    by_source = scan(a.manifest, a.seed)
    n_fresh = a.n_fresh; skipped = collections.Counter()
    while True:
        skipped.clear()
        fresh = select(a.manifest, by_source, n_fresh, check, skipped)
        n_img, n_bytes = image_bytes([r for v in fresh.values() for r in v])
        if n_bytes <= a.max_bytes or n_fresh <= 1:
            break
        print(f"fresh slice {n_fresh} rows = {n_bytes/1e9:.2f} GB of images > {a.max_bytes/1e9:.2f} GB; reducing", flush=True)
        n_fresh = int(n_fresh * 0.9)
    n_delta = min(a.n_delta, n_fresh)
    dalloc = allocate({s: len(v) for s, v in fresh.items()}, n_delta)
    delta = {s: v[:dalloc[s]] for s, v in fresh.items()}
    a.out_dir.mkdir(parents=True, exist_ok=True); copied: dict = {}
    fresh_rows = [rewrite(r, a.out_dir, copied) for s in sorted(fresh) for r in fresh[s]]
    fresh_ids = {r["id"] for r in fresh_rows}
    delta_ids = {x["id"] for v in delta.values() for x in v}
    delta_rows = [r for r in fresh_rows if r["id"] in delta_ids]
    assert {r["id"] for r in delta_rows} <= fresh_ids
    for name, rows in (("replay-fresh.jsonl", fresh_rows), ("replay-delta.jsonl", delta_rows)):
        (a.out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    def size(rows):
        paths = {im["image"] for r in rows for im in row_images(r)}
        return {"rows": len(rows), "images": len(paths), "image_bytes": sum(copied[str(ROOT / p) if not Path(p).is_absolute() else p] for p in paths)}
    report = {"manifest": str(a.manifest), "seed": a.seed, "eligible_train_rows_with_images": sum(len(v) for v in by_source.values()),
              "n_fresh_requested": a.n_fresh, "n_fresh": len(fresh_rows), "n_delta": len(delta_rows), "skipped_candidates": dict(skipped),
              "fresh": size(fresh_rows), "delta": size(delta_rows),
              "fresh_by_source": {s: len(v) for s, v in sorted(fresh.items())}, "delta_by_source": {s: len(v) for s, v in sorted(delta.items())},
              "manifest_bytes": {n: (a.out_dir / n).stat().st_size for n in ("replay-fresh.jsonl", "replay-delta.jsonl")},
              "images_dir_bytes": sum(copied.values())}
    (a.out_dir / "replay-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("n_fresh", "n_delta", "fresh", "delta", "skipped_candidates", "manifest_bytes", "images_dir_bytes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
