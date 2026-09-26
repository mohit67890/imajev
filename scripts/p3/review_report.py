"""Phase-3 review report: per-source label error rates from Kimi, corrected by the owner's verdicts, and the dev slice.

    .venv/bin/python scripts/p3/review_report.py
      -> reports/phase3/review.md, data/p3/review/dev-slice.jsonl

Maths, for each group g (source; also kind and image/text):
    n  = items Kimi reviewed,  d = items Kimi disagreed with (different answer or a flagged issue)
    kimi rate          = d / n
    precision p        = owner-confirmed errors / owner-reviewed disagreements in g; pooled over all groups when g has < 3 reviewed
    confirmed floor    = owner-confirmed errors in g / n          (hard lower bound: errors a human actually saw)
    est. error rate    = d * p / n                                 (Kimi's rate corrected by the owner's check; the plan's rule)
    + Kimi misses      = (d*p + (n-d)*m) / n, m = pooled owner error rate on random items Kimi agreed with
An owner "error" is wrong / ambiguous / bad_question (the plan's "error or ambiguity"). A group is FLAGGED when its est. error
rate > 5% (tighten its keep rule or drop it); "watch" when only the miss-adjusted rate crosses 5%.

Dev slice (never trained on): every owner-reviewed item the owner marked correct (the human-verified core, verified_by=owner)
+ every other Kimi-reviewed item where Kimi agreed with our label and flagged nothing (verified_by=kimi).
Each dev row is a candidate row + "label" in the manifest label contract (scripts/p3/assembly_manifest.py): target = our kept
label, probs = the kept distribution, review = {verdict: "correct", verified_by}. build_manifest.py --review-dev reads it, drops
these items (and their groups and variants) from training and publishes them as decision-p3-human-dev (partition test).

Drop list (data/p3/review/drop-ids.json, read by build_manifest.py --drop-ids): items the owner marked wrong / ambiguous /
bad_question, and items Kimi disputed that the owner did not check (a wrong label is worse than no label). They leave training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimi_review import REPO, REVIEW_DIR, read_jsonl, write_jsonl  # noqa: E402

THRESHOLD = 0.05
MIN_REVIEWED = 3
ERROR_VERDICTS = ("wrong", "ambiguous", "bad_question")


def latest(rows, ok=lambda r: True):
    out = {}
    for r in rows:
        if ok(r):
            out[r["id"]] = r
    return out


def compute(sample, kimi_rows, queue, owner_rows) -> dict:
    items = {it["id"]: it for it in sample}
    kimi = latest(kimi_rows, lambda r: bool(r.get("kimi")) and r["id"] in items)
    owner = latest(owner_rows)
    why = {q["id"]: q["why"] for q in queue}
    is_err = lambda i: owner[i]["verdict"] in ERROR_VERDICTS

    # pooled Kimi precision on disagreements, and miss rate on random items Kimi did not dispute
    dis_rev = [i for i in owner if i in kimi and kimi[i].get("disagree")]
    pooled_p = (sum(map(is_err, dis_rev)) / len(dis_rev)) if dis_rev else None
    miss_pool = [i for i in owner if i in kimi and not kimi[i].get("disagree") and why.get(i) == "random"]
    miss = (sum(map(is_err, miss_pool)) / len(miss_pool)) if miss_pool else None

    def group_stats(key):
        groups = defaultdict(list)
        for i, r in kimi.items():
            groups[key(items[i])].append(i)
        out = {}
        for g, ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
            n = len(ids)
            d_ids = [i for i in ids if kimi[i].get("disagree")]
            d = len(d_ids)
            rev = [i for i in d_ids if i in owner]
            c = sum(map(is_err, rev))
            c_all = sum(1 for i in ids if i in owner and is_err(i))
            wrong = sum(1 for i in ids if i in owner and owner[i]["verdict"] == "wrong")
            if len(rev) >= MIN_REVIEWED:
                p, p_src = c / len(rev), "own"
            elif pooled_p is not None:
                p, p_src = pooled_p, "pooled"
            else:
                p, p_src = 1.0, "none (owner has not reviewed yet: Kimi's rate taken as is)"
            est = d * p / n
            est_miss = (d * p + (n - d) * (miss or 0.0)) / n
            out[g] = {"n": n, "disagree": d, "kimi_rate": d / n, "owner_reviewed_disagree": len(rev), "owner_confirmed": c,
                      "precision": p, "precision_from": p_src, "confirmed_floor": c_all / n, "wrong_floor": wrong / n,
                      "est_rate": est, "est_rate_with_misses": est_miss,
                      "status": "FLAGGED" if est > THRESHOLD else ("watch" if est_miss > THRESHOLD else "ok")}
        return out

    rand = [i for i in owner if why.get(i) == "random" and i in kimi]
    agree_owner_kimi = sum(1 for i in rand if (not is_err(i)) == (not kimi[i].get("disagree")))
    return {
        "n_sample": len(items), "n_kimi": len(kimi), "n_owner": len(owner), "n_queue": len(queue),
        "pooled_precision": pooled_p, "n_disagree_reviewed": len(dis_rev), "miss_rate": miss, "n_miss_pool": len(miss_pool),
        "random_reviewed": len(rand), "random_owner_kimi_concord": (agree_owner_kimi / len(rand)) if rand else None,
        "changed_after_reveal": sum(1 for r in owner.values() if r.get("first_verdict", r["verdict"]) != r["verdict"]),
        "by_source": group_stats(lambda it: it["source"]),
        "by_kind": group_stats(lambda it: it["kind"]),
        "by_modality": group_stats(lambda it: "image" if it["image"] else "text"),
        "by_source_family": group_stats(lambda it: f"{it['source']} / {it['family']}"),
        "kimi_cost_usd": round(sum(r.get("cost_usd", 0) for r in kimi_rows), 2),
        "kimi_errors": sum(1 for i in items if i not in kimi),
    }


def dev_slice(sample, kimi_rows, owner_rows, max_kimi: int | None = 600) -> list[dict]:
    """max_kimi caps the Kimi-agreed part (every owner-verified item is kept): each dev item and its group leave training,
    so an uncapped slice of ~1,800 items would cost ~3-4k training rows. The cap is filled round-robin over source/family
    groups in a fixed hash order, so it stays stratified and deterministic."""
    items = {it["id"]: it for it in sample}
    kimi = latest(kimi_rows, lambda r: bool(r.get("kimi")) and r["id"] in items)
    owner = latest(owner_rows)
    out = []
    for i in sorted(items):
        it = items[i]
        if i in owner:
            if owner[i]["verdict"] != "correct":
                continue
            by = "owner"
        elif i in kimi and kimi[i].get("agree") and not kimi[i].get("disagree"):
            by = "kimi"
        else:
            continue
        row = {k: it[k] for k in ("id", "source", "dataset", "family", "difficulty", "state", "images", "field", "parent_id",
                                  "gold", "unknown_reason", "gold_kind", "our_label", "target_probs", "kind")}
        row["verified_by"] = by
        if i in kimi:
            row["kimi_answer"] = kimi[i]["kimi"]["answer"]
        row["label"] = {"target": None if it["our_label"] == "unknown" else it["our_label"], "probs": it.get("target_probs"),
                        "rationale": it.get("rationale"), "target_kind": it.get("target_kind") or ("teacher" if it.get("target_probs") else "gold"),
                        "review": {"verdict": "correct", "verified_by": by}}
        out.append(row)
    if max_kimi is not None:
        import hashlib
        kimi_rows_out = [r for r in out if r["verified_by"] == "kimi"]
        if len(kimi_rows_out) > max_kimi:
            groups = {}
            for r in sorted(kimi_rows_out, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
                groups.setdefault((r["source"], r["family"]), []).append(r)
            keep, order = set(), sorted(groups)
            while len(keep) < max_kimi:
                for g in order:
                    if groups[g] and len(keep) < max_kimi:
                        keep.add(groups[g].pop(0)["id"])
            out = [r for r in out if r["verified_by"] == "owner" or r["id"] in keep]
    return out


def drop_ids(sample, kimi_rows, owner_rows) -> dict:
    items = {it["id"] for it in sample}
    kimi = latest(kimi_rows, lambda r: bool(r.get("kimi")) and r["id"] in items)
    owner = latest(owner_rows)
    owner_err = sorted(i for i, r in owner.items() if r["verdict"] in ERROR_VERDICTS)
    unchecked = sorted(i for i, r in kimi.items() if r.get("disagree") and i not in owner)
    return {"owner_errors": owner_err, "kimi_disputed_unchecked": unchecked, "ids": sorted(set(owner_err) | set(unchecked))}


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _table(stats: dict, title: str) -> list[str]:
    lines = [f"### {title}", "", "| group | n | Kimi disagrees | Kimi rate | owner checked | confirmed | precision | confirmed floor | **est. error** | + Kimi misses | status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for g, s in stats.items():
        lines.append(f"| {g} | {s['n']} | {s['disagree']} | {_pct(s['kimi_rate'])} | {s['owner_reviewed_disagree']} | {s['owner_confirmed']} | "
                     f"{_pct(s['precision'])} ({s['precision_from'].split(' ')[0]}) | {_pct(s['confirmed_floor'])} | **{_pct(s['est_rate'])}** | "
                     f"{_pct(s['est_rate_with_misses'])} | {s['status']} |")
    return lines + [""]


def render(res: dict, n_dev: int, n_dev_owner: int) -> str:
    flagged = [g for g, s in res["by_source"].items() if s["status"] == "FLAGGED"]
    watch = [g for g, s in res["by_source"].items() if s["status"] == "watch"]
    L = ["# Phase-3 label review (Kimi-K2.5 + owner)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/p3/review_report.py`. Method: docs/phase-3-plan.md, "
         "\"Review: open-weight model + small human slice\"; formulas in the script docstring.", "",
         "## Summary", "",
         f"- Sampled {res['n_sample']} kept items; Kimi reviewed {res['n_kimi']} (unreviewed/errored: {res['kimi_errors']}), cost ${res['kimi_cost_usd']}.",
         f"- Owner reviewed {res['n_owner']} of {res['n_queue']} queued items ({res['n_disagree_reviewed']} Kimi disagreements, {res['random_reviewed']} random).",
         f"- Kimi precision on disagreements (owner-confirmed errors / checked): {_pct(res['pooled_precision'])}; "
         f"owner error rate on random items Kimi agreed with (Kimi misses): {_pct(res['miss_rate'])} (n={res['n_miss_pool']}).",
         f"- Owner and Kimi concordance on random items: {_pct(res['random_owner_kimi_concord'])}. "
         f"Verdicts changed after seeing Kimi: {res['changed_after_reveal']}.",
         f"- **Sources over {int(THRESHOLD * 100)}% estimated errors: {', '.join(flagged) if flagged else 'none'}.**"
         + (f" Watch (only with Kimi misses): {', '.join(watch)}." if watch else ""),
         f"- Dev slice: {n_dev} items ({n_dev_owner} human-verified core + {n_dev - n_dev_owner} Kimi-agreed) → `data/p3/review/dev-slice.jsonl`. Never train on these.",
         "", "Small groups have wide intervals: a group with n < 50 moves ±5 points on one or two items. Treat its status as a prompt to look, not a verdict.", "",
         "## Error rates", ""]
    L += _table(res["by_source"], "By source")
    L += _table(res["by_kind"], "By item kind (base / variant / unknown)")
    L += _table(res["by_modality"], "Image vs text")
    L += _table(res["by_source_family"], "By source / family")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=str(REVIEW_DIR / "sample.jsonl"))
    ap.add_argument("--kimi", default=str(REVIEW_DIR / "kimi.jsonl"))
    ap.add_argument("--queue", default=str(REVIEW_DIR / "owner-queue.jsonl"))
    ap.add_argument("--owner", default=str(REVIEW_DIR / "owner.jsonl"))
    ap.add_argument("--report", default=str(REPO / "reports/phase3/review.md"))
    ap.add_argument("--dev-out", default=str(REVIEW_DIR / "dev-slice.jsonl"))
    ap.add_argument("--drop-out", default=None, help="default: drop-ids.json next to --dev-out")
    ap.add_argument("--max-kimi-dev", type=int, default=600, help="cap on Kimi-agreed dev items (owner-verified items are always kept); 0 = none")
    args = ap.parse_args(argv)
    sample, kimi, queue, owner = (read_jsonl(p) for p in (args.sample, args.kimi, args.queue, args.owner))
    res = compute(sample, kimi, queue, owner)
    dev = dev_slice(sample, kimi, owner, max_kimi=args.max_kimi_dev if args.max_kimi_dev > 0 else None)
    write_jsonl(args.dev_out, dev)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render(res, len(dev), sum(r["verified_by"] == "owner" for r in dev)) + "\n")
    drops = drop_ids(sample, kimi, owner)
    args.drop_out = args.drop_out or str(Path(args.dev_out).with_name("drop-ids.json"))
    Path(args.drop_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.drop_out).write_text(json.dumps(drops, indent=1) + "\n")
    print(f"wrote {args.report} and {args.dev_out} ({len(dev)} items); {args.drop_out}: {len(drops['ids'])} ids leave training "
          f"({len(drops['owner_errors'])} owner-confirmed errors, {len(drops['kimi_disputed_unchecked'])} unchecked Kimi disputes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
