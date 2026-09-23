"""Assemble decision-v2.1-4b.jsonl: the corrected single-run mixture for imajev-4b v2.1.

Takes cognisance of the v2 / v2.1 findings:
  * no base-model blend on any row (v2's under-confidence and abstention prior came from it);
  * teacher-labelled rows keep their soft targets but their `unknown` share is capped (default 15%);
  * the teacher's `instruction_changes_answer` labels are near-random (teacher 35% on the probe): dropped;
  * all v2.1 two-image rows kept (they worked); state-grounded rows kept;
  * evaluation partitions are decision-v2.1's (v1.1 test + v2 new-source test + v2.1 new-source test), and
    image-hash leaks between the FULL train set and the new-source dev/test rows are repaired by source_group to
    a fixpoint (v2.1 only repaired against its 150k replay);
  * a grounded dev slice (new-source dev rows) is written for checkpoint selection, so the selector cannot reject
    the capability being trained (the v2.1 selector picked step 50 because the reasoning dev set was its only guard).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts"))
from v1_text.audit_mixture import audit

TEACHER_SOURCES = {"commons_photos", "openimages_v2", "pd12m", "stackexchange", "wikipedia_paragraphs", "support_reviews"}
NEW_SOURCES = {"state_grounded", "pairs_natural", "pairs_grounded"}
UNK = (None, "unknown", "__unknown__")
sha = lambda im: im.get("sha256", im.get("image", ""))
def is_unknown(r):
    if r.get("target") in UNK and not r.get("targets"):
        td = r.get("target_distribution") or {}
        return (not td) or max(td, key=td.get) in ("unknown", "__unknown__")
    return False
def h(seed, s): return hashlib.sha256(f"{seed}\0{s}".encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11", default="data/manifests/decision-v1.1.jsonl")
    ap.add_argument("--v2", default="data/manifests/decision-v2.jsonl")
    ap.add_argument("--v21", default="data/manifests/decision-v2.1.jsonl")
    ap.add_argument("--unknown-cap", type=float, default=0.15, help="max unknown share inside the teacher slice")
    ap.add_argument("--drop-families", default="instruction_changes_answer")
    ap.add_argument("--output", default="data/manifests/decision-v2.1-4b.jsonl")
    ap.add_argument("--dev-output", default="data/manifests/decision-v2.1-grounded-dev.jsonl")
    ap.add_argument("--report", default="reports/decision-v2.1-4b/mixture-audit.json")
    ap.add_argument("--seed", default="decision-v2.1-4b"); ap.add_argument("--skip-audit", action="store_true")
    a = ap.parse_args(); os.chdir(ROOT); drop_fam = set(filter(None, a.drop_families.split(",")))
    counts = collections.Counter()
    def load(p, pred):
        for line in open(p):
            r = json.loads(line)
            if pred(r): yield r

    # 1. v1.1 training rows with their ORIGINAL targets (no blend).
    v11_train = list(load(a.v11, lambda r: r["partition"] == "train")); counts["v11_train_records"] = len(v11_train)
    v11_ids = {r["id"] for r in v11_train}
    # 2. Teacher-labelled v2 training rows, unknown share capped.
    teacher = list(load(a.v2, lambda r: r["partition"] == "train" and r["source"] in TEACHER_SOURCES and str(r.get("pseudo_label", "")).startswith("teacher")))
    assert not ({r["id"] for r in teacher} & v11_ids)
    unk = [r for r in teacher if is_unknown(r)]; ans = [r for r in teacher if not is_unknown(r)]
    keep_unk = min(len(unk), int(a.unknown_cap / (1 - a.unknown_cap) * len(ans)))
    unk.sort(key=lambda r: h(a.seed, r["id"])); teacher_kept = ans + unk[:keep_unk]
    counts.update(teacher_records=len(teacher), teacher_unknown=len(unk), teacher_unknown_kept=keep_unk, teacher_kept=len(teacher_kept))
    # 3. v2.1: evaluation partitions (all sources) + new-source rows (all partitions).
    v21 = list(load(a.v21, lambda r: True))
    rest = [r for r in v21 if r["partition"] != "train" and r["source"] not in NEW_SOURCES]
    new = [r for r in v21 if r["source"] in NEW_SOURCES]
    dropped_fam = [r for r in new if r.get("family") in drop_fam and str(r.get("pseudo_label", "")).startswith("teacher")]
    new = [r for r in new if r not in dropped_fam]; counts["new_dropped_family"] = len(dropped_fam)
    for r in new:
        r.pop("moved_to_train", None)
    train_fixed = v11_train + teacher_kept
    # 4. Partition repair of new rows against the FULL train set + evaluation rows (fixpoint, by source_group).
    fixed_part = {}
    for r in rest + train_fixed:
        for im in r.get("images", []): fixed_part.setdefault(sha(im), r["partition"])
    moved = dropped = 0
    for _ in range(20):
        part_by_hash = collections.defaultdict(set); groups_by_hash = collections.defaultdict(set)
        for r in new:
            for im in r.get("images", []): part_by_hash[sha(im)].add(r["partition"]); groups_by_hash[sha(im)].add(r["source_group"])
        to_train, to_drop = set(), set()
        for k, parts in part_by_hash.items():
            fp = fixed_part.get(k)
            if fp == "train" and parts != {"train"}: to_train |= groups_by_hash[k]
            elif fp not in (None, "train"): to_drop |= groups_by_hash[k]
            elif fp is None and len(parts) > 1: to_train |= groups_by_hash[k]
        to_train -= to_drop
        if not to_train and not to_drop: break
        kept = []
        for r in new:
            if r["source_group"] in to_drop: dropped += 1; continue
            if r["source_group"] in to_train and r["partition"] != "train": r["partition"] = "train"; r["moved_to_train"] = "image shared across partitions"; moved += 1
            kept.append(r)
        new = kept
    counts.update(new_records=len(new), new_moved_to_train=moved, new_dropped_leak=dropped)
    for r in new:
        r["heldout_family"] = r["partition"] == "test"
        if r["partition"] == "test": r["construction_test" if r.get("pseudo_label") == "construction" else "pseudo_label_test"] = True
    rows = rest + train_fixed + new
    train_rows = [r for r in rows if r["partition"] == "train"]
    dec = lambda rs: sum(len(r["request"]["fields"]) for r in rs)
    unknown_share = sum(is_unknown(r) for r in train_rows) / len(train_rows)
    summary = dict(records=len(rows), train_records=len(train_rows), train_decisions=dec(train_rows), train_unknown_share=round(unknown_share, 4),
                   by_slice={"v1.1": dec(v11_train), "teacher": dec(teacher_kept), "new": dec([r for r in new if r["partition"] == "train"])},
                   by_source_train=dict(collections.Counter(r["source"] for r in train_rows)), **counts)
    if not a.skip_audit:
        result = audit(rows); summary["audit_ok"] = result["ok"]; summary["audit_errors"] = result["errors"][:20]
        if not result["ok"]:
            Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(summary, indent=2))
            print(json.dumps({"ok": False, "errors": result["errors"][:10], "n": len(result["errors"])})); raise SystemExit(1)
    with open(a.output, "w") as w:
        for r in rows: w.write(json.dumps(r) + "\n")
    dev_rows = [r for r in new if r["partition"] == "dev"]
    with open(a.dev_output, "w") as w:
        for r in dev_rows: w.write(json.dumps(r) + "\n")
    summary.update(manifest_sha256=hashlib.sha256(Path(a.output).read_bytes()).hexdigest(), grounded_dev_records=len(dev_rows),
                   grounded_dev_by_source=dict(collections.Counter(r["source"] for r in dev_rows)))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_source_train", "audit_errors")}))
if __name__ == "__main__": main()
