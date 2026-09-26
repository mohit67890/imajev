"""Recover teacher-REJECTED rows whose constructed gold is provable (owner decision 2026-09-26, reports/phase3/trap-fix-and-unknown-audit.md).

The 35B teacher rejects ~85% of generated "unknown" items although the audit re-derived every one of them; it commits to an answer
instead of abstaining. Rows recovered as direct constructed rows (passed to build_manifest.py --direct-constructed):
  D traps (all, gold re-derived by gen_traps + trap-gold-fixes), gui_action (rendered, fully specified), image_joint_rule UNKNOWNS only
  (the answerable ones depend on noisy ABO attributes vs the photo, so the teacher filter stays for them).
Excluded: D choice negations with pair_overlap "partial" (ambiguous), ids in image-gold-fixes, any id the teacher kept elsewhere.
Re-run after Azure stops:  .venv/bin/python scripts/p3/recover_rejected.py
"""
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/p3"))
from candidate import validate  # noqa: E402

OUT = ROOT / "data/p3/fixes/direct-recovered-rejected.jsonl"


def main(results=("data/p3/teacher/results.jsonl",)):
    fix = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/p3/fixes/trap-gold-fixes.jsonl")}
    partial = set()
    for l in open(ROOT / "data/p3/candidates/D-traps.jsonl"):
        r = json.loads(l)
        if ((r.get("provenance") or {}).get("gold_derivation") or {}).get("pair_overlap") == "partial":
            partial.add(r["id"])
    imgfix = {json.loads(l)["id"] for l in open(ROOT / "data/p3/fixes/image-gold-fixes.jsonl")}
    kept, out, c = set(), {}, collections.Counter()
    for rp in results:
        for l in open(ROOT / rp):
            r = json.loads(l); it = r.get("item") or {}
            if r.get("keep"):
                kept.add(it.get("id")); continue
            if it.get("gold_kind") != "constructed":
                continue
            fam = it.get("family")
            if not (it.get("source") == "D" or fam == "gui_action" or (fam == "image_joint_rule" and it.get("gold") is None)):
                continue
            if it["id"] in partial or it["id"] in imgfix:
                c["excluded"] += 1; continue
            it = {k: v for k, v in it.items() if k not in ("label", "pool")}
            if it["id"] in fix:
                f = fix[it["id"]]; it["gold"] = f["new_gold"]
                it["unknown_reason"] = (f.get("new_unknown_reason") or "insufficient_evidence") if f["new_gold"] is None else None
                c["relabelled"] += 1
            if validate(it):
                c["invalid"] += 1; continue
            it.setdefault("provenance", {})["recovered"] = "teacher-rejected, constructed gold re-derived"
            out[it["id"]] = it
    out = [v for k, v in out.items() if k not in kept]
    for it in out:
        c[f'{it["source"]}/{it["family"] if it["source"] == "I" else "trap"}/{"unk" if it["gold"] is None else "ans"}'] += 1
    with open(OUT, "w") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(out)} rows -> {OUT.relative_to(ROOT)}  (partial D ids {len(partial)})"); print(dict(c))


if __name__ == "__main__":
    main()
