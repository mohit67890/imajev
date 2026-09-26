"""Phase-3 pod: derived manifests for the pilot and the full run (run from ; trainer manifests live in data/manifests).

  <V>-r255        <V> without rows that have a 255-option choice field (the 255-code lanes cannot hold them: 255 options + unknown
                  = 256 codes). Written only when <V> has such rows; otherwise the 255 lanes use <V> itself. When the manifest
                  builder published a size-matched 255 lane (<V>-lane255.jsonl, scripts/p3/build_manifest.py --publish), that file
                  is used as <V>-r255 instead (checked: no 255-option row, and the same dev partition as <V>).
  <V>-c256        <V> with its 255-option rows in TRAIN only (dev keeps <= 254 options), for the 256-code lanes, so every pilot
                  lane reads the identical dev set. Written only when <V> has such rows.
  <V>-ordinal-dev dev rows of <V> restricted to their score (ordinal) fields, as reports/phase3/ordinal-loss.md step 1, from the
                  <= 254-option dev rows, so the pilot's dev2_accuracy is score accuracy.
Prints and writes a JSON summary (--out): versions to use, rows_255, ordinal dev rows.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MAN = Path("data/manifests")


def max_options(row: dict) -> int:
    return max((len(f.get("options") or []) for f in row["request"]["fields"] if f["type"] == "choice"), default=0)


def ordinal_only(row: dict) -> dict | None:
    keep = [x for x in row["request"]["fields"] if x["type"] == "ordinal"]
    if not keep:
        return None
    r = json.loads(json.dumps(row)); ids = {x["id"] for x in keep}; r["request"]["fields"] = keep
    for k in ("targets", "target_distributions", "abstention_causes"):
        if k in r:
            r[k] = {i: v for i, v in r[k].items() if i in ids}
    return r


def copy_index(src: str, dst: str, man: Path):
    idx = man / f"{src}-images.json"
    if idx.exists():
        shutil.copyfile(idx, man / f"{dst}-images.json")


def prep(version: str, man: Path = MAN) -> dict:
    rows = [json.loads(l) for l in (man / f"{version}.jsonl").read_text().split("\n") if l.strip()]
    wide = [r for r in rows if max_options(r) >= 255]
    if any(max_options(r) > 255 for r in rows):
        raise SystemExit(f"{version}: a choice field has more than 255 options")
    v255 = v256 = version
    lane255 = man / f"{version}-lane255.jsonl"
    prebuilt = False
    if wide:
        v255, v256 = f"{version}-r255", f"{version}-c256"
        if lane255.exists():
            lrows = [json.loads(l) for l in lane255.read_text().split("\n") if l.strip()]
            if any(max_options(r) >= 255 for r in lrows):
                raise SystemExit(f"{lane255.name}: holds a 255-option row")
            dev_v = sorted(r["id"] for r in rows if r.get("partition") == "dev")
            dev_l = sorted(r["id"] for r in lrows if r.get("partition") == "dev")
            if dev_v != dev_l:
                raise SystemExit(f"{lane255.name}: dev partition differs from {version}'s")
            shutil.copyfile(lane255, man / f"{v255}.jsonl")
            prebuilt = True
        with (man / f"{v256}.jsonl").open("w") as g:
            if not prebuilt:
                f = (man / f"{v255}.jsonl").open("w")
            for r in rows:
                if max_options(r) < 255 and not prebuilt:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if max_options(r) < 255 or r.get("partition") == "train":
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not prebuilt:
                f.close()
        copy_index(version, v255, man); copy_index(version, v256, man)
    vord = f"{version}-ordinal-dev"; n_ord = 0
    with (man / f"{vord}.jsonl").open("w") as f:
        for r in rows:
            if r.get("partition") == "dev" and max_options(r) < 255:
                o = ordinal_only(r)
                if o:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n"); n_ord += 1
    copy_index(version, vord, man)
    parts = {}
    for r in rows:
        parts[r.get("partition")] = parts.get(r.get("partition"), 0) + 1
    return {"version": version, "rows": len(rows), "partitions": parts, "rows_255": sum(1 for r in wide if r.get("partition") == "train"),
            "version_255": v255, "version_256": v256, "lane255_prebuilt": prebuilt,
            "ordinal_dev": vord if n_ord else None, "ordinal_dev_rows": n_ord}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    info = prep(a.version)
    a.out.write_text(json.dumps(info, indent=1) + "\n"); print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
