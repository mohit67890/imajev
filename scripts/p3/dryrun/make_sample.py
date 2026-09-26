"""Dry-run sample pool (reports/phase3/dry-run.md): 3 real shards cut to ~500 rows each, with image, long and held-out-bucket
rows over-represented; the held-out files, large-choice rows and a quotas.json scaled to the sample are copied beside them."""
import hashlib, json, math, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]           # scripts/p3/dryrun/ -> repo root
SRC, DST = ROOT / "data/p3/pool", ROOT / "data/p3/dryrun/pool"
(DST / "shards").mkdir(parents=True, exist_ok=True)
h = lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
total = 0
for k in range(3):
    rows = [json.loads(l) for l in open(SRC / f"shards/pool-{k:02d}.jsonl")]
    bucket = [r for r in rows if r["pool"]["heldout_flagged_bucket"]][:45]
    img = [r for r in rows if r.get("images") and not r["pool"]["heldout_flagged_bucket"]]
    img = sorted(img, key=lambda r: h(r["id"]))[:150]
    longr = sorted((r for r in rows if len(json.dumps(r["state"])) > 20000 and not r.get("images")), key=lambda r: h(r["id"]))[:15]
    ids = {r["id"] for r in bucket + img + longr}
    rest = sorted((r for r in rows if r["id"] not in ids and not r.get("images")), key=lambda r: h(r["id"]))[:500 - len(ids)]
    keep = ids | {r["id"] for r in rest}
    out = [r for r in rows if r["id"] in keep]            # shard order kept (family-interleaved)
    if k == 2:   # 8 planted heldout-fresh rows: the coordinator must never queue them (the leakage rule's first line)
        fresh = [json.loads(l) for l in open(SRC / "heldout-fresh.jsonl")]
        fresh = sorted((r for r in fresh if not r.get("images")), key=lambda r: h(r["id"]))[:8]
        out += [dict(r, pool={"group": "planted:" + r["id"], "quota_key": r["family"], "shard": 2, "heldout_flagged_bucket": False})
                for r in fresh]
    with open(DST / f"shards/pool-{k:02d}.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total += len(out)
    print(f"pool-{k:02d}: {len(out)} rows, bucket {len(bucket)}, image {len(img)}, long {len(longr)}")
for name in ("heldout-fresh.jsonl", "heldout-fresh-gui.jsonl", "heldout-fresh-large-choice.jsonl", "large-choice.jsonl"):
    shutil.copyfile(SRC / name, DST / name)
q = json.loads((SRC / "quotas.json").read_text())
f = total / 209065
for v in q["families"].values():
    v["quota"] = max(1, math.ceil(v["quota"] * f * 3))        # x3: the sample over-represents images / long rows
q["cap_total_teacher_calls"] = 600
q["heldout_flagged_cap"] = 60
q["dry_run_note"] = f"scaled from data/p3/pool/quotas.json by {f:.5f} x 3 for {total} sample rows"
(DST / "quotas.json").write_text(json.dumps(q, indent=1) + "\n")
print("total", total, "quota sum", sum(v["quota"] for v in q["families"].values()))
