"""Phase-3 round 2 (Stage 5, mandatory): the short-epoch manifest from what the round-1 best checkpoint still fails.

Inputs
  --mined    scripts/p3/mine.py outputs of the round-1 best checkpoint: <shard>.flagged.jsonl (every row is flagged), or any jsonl
             whose rows carry a true `flag` / `flagged`
  --source   the manifest holding every teacher-labelled, verified item (default <V>-labelled when it exists, else <V>): only its
             TRAIN partition is eligible, so no held-out or dev item can enter round 2 (labels are cached: no teacher calls)
  --base     the round-1 manifest <V> (replay rows are sampled from its train partition; its dev partition is copied as is)
  --exclude  held-out / human-slice manifests: any row whose id, parent id or group appears there is refused (belt and braces)
Rows are matched to mined ids by exact id, by `candidate_id`, or by the id before a ':field' suffix.
Output: data/manifests/<out>.jsonl with the newly failed rows (never-trained-on first, capped at --max-new), replay rows
(--replay-ratio x new, stratified as the base) and the base dev partition; report JSON on stdout / --report.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

MAN = Path("data/manifests")


def read_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]


def keys_of(row: dict) -> set[str]:
    k = {row["id"], row["id"].split(":")[0]}
    for f in ("candidate_id", "parent_id", "group"):
        if row.get(f):
            k.add(str(row[f]))
    return k


def flagged_ids(paths: list[str]) -> set[str]:
    out = set()
    for p in paths:
        all_flagged = str(p).endswith(".flagged.jsonl")
        for r in read_jsonl(p):
            if all_flagged or r.get("flag") or r.get("flagged"):
                out.add(str(r["id"]))
    return out


def max_options(row: dict) -> int:
    return max((len(f.get("options") or []) for f in row["request"]["fields"] if f["type"] == "choice"), default=0)


def build(mined: set[str], source: list[dict], base: list[dict], exclude: set[str], max_new: int, replay_ratio: float, codes: int, seed: int = 0):
    base_train_ids = {r["id"] for r in base if r.get("partition") == "train"}
    fits = (lambda r: max_options(r) <= codes - 1)
    cand, refused = [], 0
    for r in source:
        if r.get("partition") != "train" or not fits(r):
            continue
        k = keys_of(r)
        if k & exclude:
            refused += 1; continue
        if (r["id"] in mined) or (r.get("candidate_id") in mined) or (r["id"].split(":")[0] in mined):
            cand.append(r)
    rng = random.Random(seed)
    fresh = [r for r in cand if r["id"] not in base_train_ids]; again = [r for r in cand if r["id"] in base_train_ids]
    rng.shuffle(fresh); rng.shuffle(again)
    new = (fresh + again)[:max_new]
    new_ids = {x["id"] for x in new}
    pool = [r for r in base if r.get("partition") == "train" and fits(r) and not keys_of(r) & exclude and r["id"] not in new_ids]
    rng.shuffle(pool)
    replay = pool[: int(round(replay_ratio * len(new)))]
    dev = [r for r in base if r.get("partition") == "dev" and fits(r)]
    rep = {"flagged_ids": len(mined), "eligible_flagged_rows": len(cand), "never_trained": len(fresh), "retrained": len(again),
           "new_rows": len(new), "replay_rows": len(replay), "dev_rows": len(dev), "refused_heldout_overlap": refused,
           "image_rows": sum(bool(r.get("images") or r.get("image")) for r in new + replay)}
    return [dict(r, partition="train") for r in new + replay] + dev, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mined", required=True, help="glob of mine.py outputs"); ap.add_argument("--source", required=True)
    ap.add_argument("--base", required=True); ap.add_argument("--exclude", default="", help="comma list of manifest names")
    ap.add_argument("--out", required=True); ap.add_argument("--max-new", type=int, default=20000)
    ap.add_argument("--replay-ratio", type=float, default=1.0); ap.add_argument("--codes", type=int, choices=(255, 256), default=255)
    ap.add_argument("--min-new", type=int, default=200, help="fail (exit 3) when fewer newly failed rows are found")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(a.mined))
    if not paths:
        raise SystemExit(f"no mined outputs match {a.mined}")
    exclude = set()
    for m in filter(None, a.exclude.split(",")):
        p = MAN / f"{m}.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                exclude |= keys_of(r)
    rows, rep = build(flagged_ids(paths), read_jsonl(MAN / f"{a.source}.jsonl"), read_jsonl(MAN / f"{a.base}.jsonl"), exclude,
                      a.max_new, a.replay_ratio, a.codes)
    with (MAN / f"{a.out}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for src in (a.source, a.base):   # image index: the union of the source and base indexes
        idx = MAN / f"{src}-images.json"
        if idx.exists():
            out = MAN / f"{a.out}-images.json"
            cur = json.loads(out.read_text()) if out.exists() else {"images": {}}
            cur["images"].update(json.loads(idx.read_text())["images"]); out.write_text(json.dumps(cur))
    rep["out"] = a.out; rep["mined_files"] = len(paths)
    print(json.dumps(rep))
    if a.report:
        a.report.write_text(json.dumps(rep, indent=1) + "\n")
    return 0 if rep["new_rows"] >= a.min_new else 3


if __name__ == "__main__":
    raise SystemExit(main())
