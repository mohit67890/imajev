"""Assemble decision-v2.1.jsonl: every non-train row of decision-v2 + a replay slice of its train rows + ALL new v2.1 rows
(teacher-labelled state_grounded / pairs_natural, and pairs_grounded whose labels are known by construction). New-source test rows
are flagged heldout_family / pseudo_label_test (pairs_grounded test rows are human-verifiable, flagged construction_test)."""
from __future__ import annotations
import collections
import argparse, hashlib, json, os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts"))
from v1_text.audit_mixture import audit
from v1_text.common import write_jsonl

def load(p): return [json.loads(l) for l in Path(p).open() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/manifests/decision-v2.jsonl")
    ap.add_argument("--labelled", nargs="+", required=True, help="teacher outputs for state_grounded / pairs_natural")
    ap.add_argument("--constructed", nargs="+", default=["data/decision-v2/pairs_grounded/records.jsonl"], help="already-labelled rows")
    ap.add_argument("--replay", type=int, default=150000, help="v2 train rows kept as replay")
    ap.add_argument("--output", default="data/manifests/decision-v2.1.jsonl"); ap.add_argument("--report", required=True)
    ap.add_argument("--seed", default="decision-v2.1")
    a = ap.parse_args(); os.chdir(ROOT)
    v2 = load(a.v2); ids = {r["id"] for r in v2}
    train = [r for r in v2 if r["partition"] == "train"]; rest = [r for r in v2 if r["partition"] != "train"]
    train.sort(key=lambda r: hashlib.sha256(f"{a.seed}\0{r['id']}".encode()).hexdigest()); replay = train[:a.replay]
    # A replay subsample strands irrelevant-image controls whose relevant twin was not sampled: add the twins back.
    sha = lambda im: im.get("sha256", im.get("image", ""))
    relevant_by_hash = {}
    for r in train:
        if r.get("image_role", "relevant") == "relevant":
            for im in r.get("images", []): relevant_by_hash.setdefault(sha(im), r)
    replay_ids = {r["id"] for r in replay}; covered = {sha(im) for r in replay if r.get("image_role", "relevant") == "relevant" for im in r.get("images", [])}
    topped = 0
    for r in list(replay):
        if r.get("image_role") == "irrelevant":
            for im in r.get("images", []):
                k = sha(im); twin = relevant_by_hash.get(k)
                if k not in covered and twin is not None and twin["id"] not in replay_ids:
                    replay.append(twin); replay_ids.add(twin["id"]); topped += 1
                    for im2 in twin.get("images", []): covered.add(sha(im2))
    train_hashes = {sha(im) for r in train for im in r.get("images", [])}
    new = []
    for p in a.labelled + a.constructed:
        for r in load(p):
            if r["id"] in ids: raise SystemExit(f"id collision with v2: {r['id']}")
            r.setdefault("pseudo_label", "constructed" if p in a.constructed else r.get("pseudo_label"))
            new.append(r)
    # Image hashes must not span partitions. v2 rows are fixed; new rows are repaired by whole source_group until stable:
    # a photo v2 trained on pulls the new group into train; a photo from v2 dev/test/calibration drops the new group;
    # a photo shared by two new groups in different partitions pulls both into train.
    v2_part = {}
    for r in rest + replay:
        for im in r.get("images", []): v2_part.setdefault(sha(im), r["partition"])
    moved = dropped = 0; collide = set()
    for _ in range(20):
        part_by_hash = collections.defaultdict(set); groups_by_hash = collections.defaultdict(set)
        for r in new:
            for im in r.get("images", []): part_by_hash[sha(im)].add(r["partition"]); groups_by_hash[sha(im)].add(r["source_group"])
        to_train = set(); to_drop = set()
        for k, parts in part_by_hash.items():
            vp = v2_part.get(k)
            if vp == "train" and parts != {"train"}: to_train |= groups_by_hash[k]
            elif vp not in (None, "train"): to_drop |= groups_by_hash[k]
            elif vp is None and len(parts) > 1: to_train |= groups_by_hash[k]
        to_train -= to_drop
        if not to_train and not to_drop: break
        kept = []
        for r in new:
            if r["source_group"] in to_drop: dropped += 1; continue
            if r["source_group"] in to_train and r["partition"] != "train": r["partition"] = "train"; r["moved_to_train"] = "image shared across partitions"; moved += 1
            kept.append(r)
        new = kept; collide |= to_train | to_drop
    for r in new:
        if r["partition"] == "test":
            r["heldout_family"] = True; r["construction_test" if r.get("pseudo_label") == "constructed" else "pseudo_label_test"] = True
    rows = rest + replay + new
    result = audit(rows)
    result.update(v2_rows_total=len(v2), replay_train_rows=len(replay), replay_control_twins_added=topped, new_rows=len(new),
                  new_rows_moved_to_train=moved, new_rows_dropped=dropped, new_groups_repaired=len(collide),
                  new_decisions=sum(len(r["request"]["fields"]) for r in new),
                  new_by_source={s: sum(1 for r in new if r["source"] == s) for s in sorted({r["source"] for r in new})},
                  train_decisions=sum(len(r["request"]["fields"]) for r in rows if r["partition"] == "train"))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    if not result["ok"]:
        Path(a.report).write_text(json.dumps(result, indent=2)); print(json.dumps({"ok": False, "errors": result["errors"][:10], "n": len(result["errors"])})); raise SystemExit(1)
    write_jsonl(a.output, rows); result["manifest_sha256"] = hashlib.sha256(Path(a.output).read_bytes()).hexdigest(); result["records"] = len(rows)
    Path(a.report).write_text(json.dumps(result, indent=2)); print(json.dumps({k: v for k, v in result.items() if k not in ("counts", "licenses", "errors")}))
if __name__ == "__main__": main()
