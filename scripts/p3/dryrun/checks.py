"""Dry-run assertions for reports/phase3/dry-run.md: one PASS/FAIL line (+ counts) per integration check.

    .venv/bin/python scripts/p3/dryrun/checks.py <step>        # steps: mine, mlx, session, gate, flagged, review, page, manifest, pod
Exit 0 when every check of the step passes."""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/p3/dryrun"
S = D / "session"
FAILS = []


def rd(p):
    p = Path(p)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def mine():
    sc = [r for f in glob.glob(str(D / "pod/mine/*.scores.jsonl")) for r in rd(f)]
    fl = [r for f in glob.glob(str(D / "pod/mine/*.flagged.jsonl")) for r in rd(f)]
    pool = sum(len(rd(f)) for f in glob.glob(str(D / "pool/shards/*.jsonl")))
    check("every pool row scored once", len(sc) == pool == len({r["id"] for r in sc}), f"{len(sc)} scored / {pool} rows")
    check("all scores status ok (long + image rows render)", all(r["status"] == "ok" for r in sc),
          str(collections.Counter(r["status"] for r in sc)))
    check("3 .done markers", len(glob.glob(str(D / "pod/mine/*.done"))) == 3)
    check("flag rows carry item + mine + priority", all({"item", "mine", "priority"} <= set(r) for r in fl),
          f"{len(fl)} flagged ({len(fl) / max(1, len(sc)):.0%}); flags {dict(collections.Counter(f for r in fl for f in r['mine']['flags']))}")
    check("image, long and bucket rows among the flagged", all(x > 0 for x in (
        sum(bool(r["item"].get("images")) for r in fl), sum(len(json.dumps(r["item"]["state"])) > 20000 for r in fl),
        sum(r["item"]["pool"]["heldout_flagged_bucket"] for r in fl))),
        f"image {sum(bool(r['item'].get('images')) for r in fl)}, long {sum(len(json.dumps(r['item']['state'])) > 20000 for r in fl)}, "
        f"bucket {sum(r['item']['pool']['heldout_flagged_bucket'] for r in fl)}")


def mlx():
    sc = [r for f in glob.glob(str(D / "mine-mlx/*.scores.jsonl")) for r in rd(f)]
    check("MLX smoke: real model scores", len(sc) > 0 and all(r["status"] == "ok" for r in sc),
          f"{len(sc)} items, {sum(r['has_images'] for r in sc)} image, {sum(r['flagged'] for r in sc)} flagged, "
          f"max {max([r.get('input_tokens', 0) for r in sc] or [0])} input tokens")
    check("MLX smoke under --limit writes no .done", not glob.glob(str(D / "mine-mlx/*.done")))


def session():
    q, qv = rd(S / "stream/queue.jsonl"), rd(S / "stream/queue-variants.jsonl")
    fresh = {r["id"] for f in glob.glob(str(D / "pool/heldout-fresh*.jsonl")) for r in rd(f)}
    st = json.loads((S / "stream/coordinator-status.json").read_text())
    check("no heldout-fresh id reaches the teacher", not any(r["id"] in fresh for r in q + qv),
          f"{st['heldout_fresh_ids_skipped']} planted fresh rows skipped")
    check("bucket rows only as origin heldout", all((r["origin"] == "heldout") == bool(r["item"]["pool"]["heldout_flagged_bucket"])
                                                    for r in q), str(collections.Counter(r["origin"] for r in q)))
    pr = [r["priority"] for r in q if r["origin"] == "mine"]
    check("most confidently wrong first", all(a >= b for a, b in zip(pr, pr[1:])), f"{pr[0]:.3f} .. {pr[-1]:.3f}")
    check("call cap and quotas applied", st["estimated_calls"] <= st["global_cap"] and bool(st["blocked"]),
          f"calls {st['estimated_calls']}/{st['global_cap']}, blocked {st['blocked']}, heldout {st['heldout_queued']}/{st['heldout_cap']}")
    check("queues closed", (S / "stream/queue.jsonl.closed").exists() and (S / "stream/queue-variants.jsonl.closed").exists())
    res = rd(S / "teacher/results.jsonl")
    check("every queued item has one verdict", len(res) == len({r["id"] for r in res}) == len(q) + len(qv),
          f"{len(res)} results: {dict(collections.Counter(r['origin'] for r in res))}")
    ts = json.loads((S / "teacher/status.json").read_text())
    gate = json.loads((S / "teacher/pilot-gate.json").read_text())
    check("teacher done, pilot gate PASS", ts["state"] == "done" and gate["decision"] == "pass",
          f"gate metrics: parse {gate['metrics']['parse_error_rate']:.1%}, trunc {gate['metrics']['truncation_rate']:.1%}, "
          f"constructed agree {gate['metrics']['constructed_keep_rate']:.1%}, {gate['metrics']['mean_completion_tokens']} tok, "
          f"${gate['metrics']['usd_per_item']}/item; keep by source {gate['metrics']['keep_rate_by_source']}")
    sv = rd(S / "teacher/solves.jsonl")
    check("8 simulated instances used", ts["instances"] == 8 and {s["deployment"] for s in sv} == {"imajev-teacher"},
          f"{len(sv)} solves, {sum(s.get('truncated', 0) for s in sv)} truncated, {sum(s.get('parse_errors', 0) for s in sv)} parse errors")
    check("no 9B dependency on image rows", not any("needs_9b_distribution" in r for r in res),
          str(collections.Counter(r["rule"] for r in res if r["item"].get("images"))))
    var = rd(S / "variants/parents.jsonl")
    cons = rd(S / "variants/constructed.jsonl")
    check("variants without a writer", any(p["method"] == "skipped_no_writer" for p in var) and len(cons) > 0
          and all(c["label"]["target_kind"] == "constructed" for c in cons),
          f"{len(cons)} constructed, {len(qv)} unknown variants to the teacher, parents {dict(collections.Counter(p['method'] for p in var))}")
    kept = rd(S / "teacher/kept.jsonl")
    check("finalize: kept rows carry the label contract, held-out apart",
          kept and all(set(r["label"]) == {"target", "probs", "rationale", "target_kind", "review"} for r in kept)
          and not any(r["origin"] == "heldout" for r in kept) and all(r["origin"] == "heldout" for r in rd(S / "teacher/heldout-results.jsonl")),
          f"kept {len(kept)} ({dict(collections.Counter(r['label']['target_kind'] for r in kept))}), "
          f"heldout verdicts {len(rd(S / 'teacher/heldout-results.jsonl'))}")
    led = rd(S / "teacher/instances.jsonl")
    check("ledger: set 8 ... set 0", [e["instances"] for e in led][0] == 8 and led[-1]["instances"] == 0)


def gate():
    G = D / "gate/teacher"
    st = json.loads((G / "pilot-gate.json").read_text())
    check("gate PAUSED on 6% parse errors, then resumed", st["decision"] == "resumed" and st["why"], "; ".join(st["why"]))
    check("alert file + resume flag consumed", (G / "PILOT_GATE_ALERT.txt").exists() and not (G / "RESUME_PILOT").exists()
          and bool(glob.glob(str(G / "RESUME_PILOT.used-*"))))
    ts = json.loads((G / "status.json").read_text())
    check("run completed after resume", ts["state"] == "done", f"{ts['results']} results")


def flagged():
    h = rd(D / "pool/heldout-flagged.jsonl")
    ex = json.loads((D / "pool/heldout-flagged-exclude.json").read_text())
    check("take-flagged collected the bucket rows", len(h) > 0 and all(r["pool"]["heldout_flagged_bucket"] for r in h),
          f"{len(h)} held-out flagged, {len(ex['exclude_train_ids'])} training rows excluded for links")


def review():
    R = D / "review"
    smp, k = rd(R / "sample.jsonl"), rd(R / "kimi.jsonl")
    check("Kimi sample fills the request", len(smp) == 150 and sum(i["image"] for i in smp) == 30,
          str(dict(collections.Counter((i["source"], i["kind"]) for i in smp))))
    check("Kimi review (mock) complete, resumable", len(k) == len({r["id"] for r in k}) == len(smp) and all(r.get("kimi") for r in k),
          f"{sum(bool(r.get('disagree')) for r in k)} disagreements")
    dev = rd(R / "dev-slice.jsonl")
    check("dev slice rows carry a label", dev and all(r["label"]["review"]["verdict"] == "correct" for r in dev),
          f"{len(dev)} rows {dict(collections.Counter(r['verified_by'] for r in dev))}; drop-ids "
          f"{len(json.loads((R / 'drop-ids.json').read_text())['ids'])}")


def manifest():
    rep = json.loads((D / "manifests/manifest-report.json").read_text())
    for lane in ("256", "255"):
        check(f"leakage gate lane {lane}", rep["gate"][lane]["exit"] == 0, rep["gate"][lane]["result"])
        g = rep["gate"].get(lane + "-gui")
        check(f"GUI gate lane {lane}: fixed-state links only", g is None or g["only_fixed_state_links"], g and g["result"])
        check(f"lane {lane} shares", rep["lanes"][lane]["shares_ok"], str(rep["lanes"][lane]["shares"]))
    pub = rep.get("published", {})
    check("pod names published", set(pub) == {f"decision-p3dry{s}" for s in ("", "-lane255", "-labelled", "-heldout-fresh",
                                                                               "-heldout-fresh-large-choice", "-heldout-flagged", "-human-dev")},
          ", ".join(f"{k}={v['rows']}" for k, v in pub.items()))
    check("dropped counts", True, json.dumps(rep["dropped"]))


def main():
    for step in sys.argv[1:]:
        print(f"== {step}")
        globals()[step]()
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
