"""Build the phase-3 DecisionBench 3k stratified dev subset (TRACKING ONLY; docs/phase-3-plan.md Stage 4 and Targets).

Source: Hugging Face dataset Hanno-Labs/decision-bench @ b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443, split eval
(data/eval-00000-of-00001.parquet, 23,900 rows, sha256 pinned). Built ONCE, on the pod, into a tracking-only directory.

Stratification is by task (`task_id`, 43 tasks). Allocation is equal per task with water-filling: every task gets
min(its rows, q) rows, q chosen so the total is --size (3,000 -> ~70 per task), so small tasks (100 rows: agent actions,
reasoning, ordinal) are measured with the same precision as the 2,222-row ones. Rows are sampled with a fixed seed.
Each row keeps its task's full-suite row count so a scorer can report a "full-suite equivalent" accuracy (tasks weighted by
their share of the 23,900 rows), which is the number comparable with the shipped 77.5.

Rules (enforced by cloud/p3/pod_run_train.sh and tested in tests/test_p3_pod.py):
  * never used for training, tuning or checkpoint selection; the selector never opens this directory;
  * scores are written only to <tracking-only>/<checkpoint>/decisionbench.json, read by a person via the final table.

    python scripts/p3/make_decisionbench_subset.py --out p3/tracking-only/decisionbench-3k.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

REPO = "Hanno-Labs/decision-bench"
REVISION = "b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443"
FILENAME = "data/eval-00000-of-00001.parquet"
SHA256 = "6c97d3f3b5f79ca8566d2ecdf2b0e0f78c20102898810cdd8b7439ac776893ca"
ROWS = 23900
COLUMNS = ("row_id", "task_id", "task_name", "primitive", "family", "domain", "candidate_count", "reasoning_required", "reasoning_type",
           "instruction", "state_json", "candidates_json", "gold_candidate_id", "gold_probabilities")


def allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    """Equal allocation per stratum with water-filling: min(count, q) per stratum, remainder to the largest strata (by name)."""
    if size >= sum(counts.values()):
        return dict(counts)
    alloc = {k: 0 for k in counts}
    remaining, open_keys = size, sorted(counts)
    while remaining > 0 and open_keys:
        q = remaining // len(open_keys)
        if q == 0:  # distribute the last few rows one each, largest strata first (ties by name)
            for k in sorted(open_keys, key=lambda k: (-(counts[k] - alloc[k]), k))[:remaining]:
                alloc[k] += 1
            break
        nxt = []
        for k in open_keys:
            take = min(q, counts[k] - alloc[k]); alloc[k] += take; remaining -= take
            if alloc[k] < counts[k]:
                nxt.append(k)
        open_keys = nxt
    return alloc


def stratified(rows: list[dict], size: int, seed: int = 0, key: str = "task_id") -> list[dict]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    alloc = allocate({k: len(v) for k, v in by.items()}, size)
    out = []
    for k in sorted(by):
        group = sorted(by[k], key=lambda r: r["row_id"])
        picked = random.Random(f"{seed}:{k}").sample(group, alloc[k])
        full = len(group)
        for r in sorted(picked, key=lambda r: r["row_id"]):
            out.append(dict(r, stratum=k, stratum_full_rows=full, stratum_sampled_rows=alloc[k]))
    return out


def load_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=list(COLUMNS)).to_pylist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="tracking-only jsonl path (its directory must be named tracking-only)")
    ap.add_argument("--size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parquet", type=Path, help="a local copy of the pinned parquet (default: download the pinned revision)")
    ap.add_argument("--token-env", default="HF_TOKEN", help="env var with an HF token (optional; the dataset is public)")
    a = ap.parse_args(argv)
    if "tracking-only" not in a.out.parts:
        ap.error("--out must live under a directory named tracking-only (the selector's exclusion rule keys on it)")
    if a.out.exists():
        print(f"{a.out} exists; not rebuilding (built once)"); return 0
    path = a.parquet
    if path is None:
        import os
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION, token=os.environ.get(a.token_env) or None))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != SHA256:
        raise SystemExit(f"DecisionBench parquet sha256 {h} != pinned {SHA256}")
    rows = load_parquet(path)
    assert len(rows) == ROWS, len(rows)
    sub = stratified(rows, a.size, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(a.out)
    meta = {"source": f"hf://datasets/{REPO}@{REVISION}/{FILENAME}", "sha256": SHA256, "rows_total": len(rows), "rows_subset": len(sub),
            "seed": a.seed, "stratum": "task_id", "allocation": dict(collections.Counter(r["stratum"] for r in sub)),
            "full_counts": dict(collections.Counter(r["task_id"] for r in rows)),
            "primitives": dict(collections.Counter(r["primitive"] for r in sub)), "families": dict(collections.Counter(r["family"] for r in sub)),
            "over_254_candidates": sum(r["candidate_count"] > 254 for r in sub),
            "rule": "tracking only: never trained on, never tuned on, never read by the checkpoint selector"}
    a.out.with_name(a.out.stem + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(sub)} rows over {len(meta['allocation'])} tasks to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
