#!/usr/bin/env python3
"""PASS/FAIL checks for scripts/p3/dryrun/labeltrain.sh (reports/phase3/labeltrain-dryrun.md).

    labeltrain_checks.py pod <D>        the pulled pod outputs: backend tags, no Azure ledger, coverage of extra + leftovers,
                                        writer variants re-verified, consistency sample, markers, throughput log
    labeltrain_checks.py manifest <D>   build_manifest from both teacher dirs: preference, backend counts, direct rows, gate
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def check(ok: bool, msg: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def pod(d: Path) -> int:
    pulled, lt = d / "pulled", d / "labeltrain"
    res = rows(pulled / "teacher-pod/results.jsonl")
    solves = rows(pulled / "teacher-pod/solves.jsonl")
    extra = {r["id"] for r in rows(lt / "queue-extra.jsonl")}
    left = {r["id"] for r in rows(lt / "queue-leftover.jsonl")}
    ids = {r["id"] for r in res}
    ok = True
    ok &= check(bool(res) and {r.get("teacher_backend") for r in res} == {"pod-fp8"},
                f"{len(res)} pod verdicts, every one teacher_backend pod-fp8 ({Counter(r.get('origin') for r in res)})")
    ok &= check({s.get("teacher_backend") for s in solves} == {"pod-fp8"}, f"{len(solves)} solves tagged pod-fp8")
    ok &= check(not (pulled / "teacher-pod/instances.jsonl").exists(), "no Azure instance ledger in teacher-pod/")
    ok &= check(extra <= ids, f"every extra-queue row judged ({len(extra)})")
    ok &= check(left <= ids, f"every Azure leftover judged ({len(left)})")
    errs = [s.get("error") or "" for s in solves if not s.get("ok")]
    ok &= check(not any("credential" in e or "HTTP 400" in e for e in errs),
                f"no request refused for a credential header ({len(errs)} failed solves: {Counter(e[:40] for e in errs).most_common(3)})")
    var = rows(pulled / "stream-pod/queue-variants.jsonl")
    wv = [v for v in var if (v.get("variant") or {}).get("method") == "writer"]
    wjudged = [r for r in res if (r.get("variant") or {}).get("method") == "writer"]
    ok &= check(bool(wv) and len(wjudged) == len(wv),
                f"writer variants queued {len(wv)} (parents: {len({v['variant']['parent_id'] for v in wv})}), all re-verified "
                f"(kept {sum(1 for r in wjudged if r['keep'])}; rules {dict(Counter(r['rule'] for r in wjudged))})")
    azure_parents = {r["id"] for r in rows(lt / "writer-parents.jsonl")} | {r["id"] for r in rows(lt / "writer-parents-late.jsonl")}
    got = {v["variant"]["parent_id"] for v in wv}
    ok &= check(bool(azure_parents & got), f"Azure-kept B/C parents got writer variants ({len(azure_parents & got)}/{len(azure_parents)})")
    prog = rows(pulled / "variants-pod/constructed.jsonl")
    ok &= check(True, f"programmatic variants of pod-kept A/D/I parents: {len(prog)} constructed; "
                      f"parents {dict(Counter(p.get('method') for p in rows(pulled / 'variants-pod/parents.jsonl')))}")
    cons = json.loads((pulled / "teacher-pod-consistency/consistency.json").read_text()) \
        if (pulled / "teacher-pod-consistency/consistency.json").exists() else {}
    ok &= check(cons.get("items", 0) > 0, f"FP8-vs-bf16 consistency sample: {cons.get('items')} items, solve-1 agreement "
                                          f"{cons.get('solve1_argmax_agreement')}, keep agreement {cons.get('keep_verdict_agreement')}")
    kept = rows(pulled / "teacher-pod/kept.jsonl")
    ok &= check(bool(kept) and (pulled / "teacher-pod/kept-summary.json").exists(), f"pod finalize: kept.jsonl {len(kept)} rows")
    mk = sorted(p.name for p in (pulled / "labeltrain-pod").iterdir() if p.is_file() and p.name.isupper())
    # the pack is written just before LABEL_DONE, so LABEL_DONE itself is checked on the pod (step 5), not in the pack
    ok &= check({"SERVERS_READY", "WRITER_READY", "LEFTOVERS_INSTALLED", "WRITER_DONE", "WRITER_SWITCHED"} <= set(mk),
                f"markers in the pack: {mk}")
    thr = (pulled / "labeltrain-pod/throughput.log").read_text().splitlines() if (pulled / "labeltrain-pod/throughput.log").exists() else []
    ok &= check(len(thr) >= 2 and "IST" in thr[-1], f"throughput log: {len(thr)} IST lines")
    gate = json.loads((pulled / "teacher-pod/pilot-gate.json").read_text()) if (pulled / "teacher-pod/pilot-gate.json").exists() else {}
    ok &= check(gate.get("decision") in ("pass", "resumed"), f"pilot gate on the pod: {gate.get('decision')} {gate.get('why') or ''}")
    return 0 if ok else 1


def manifest(d: Path) -> int:
    rep = json.loads((d / "manifests/manifest-report.json").read_text())
    ok = True
    st = rep["teacher_dirs"]["stats"]
    ok &= check(st.get("kept:azure-bf16", 0) > 0 and st.get("kept:pod-fp8", 0) > 0, f"kept rows from both teachers: {st}")
    lane = rep["lanes"]["256"]
    ok &= check({"azure-bf16", "pod-fp8"} <= set(lane["hard_by_teacher_backend"]),
                f"hard rows by backend (256 lane): {lane['hard_by_teacher_backend']}")
    ok &= check(lane["hard_direct_constructed"] > 0, f"direct chart/safety rows in training: {lane['hard_direct_constructed']} "
                                                     f"{lane.get('hard_direct_by_family')}")
    ok &= check(rep["dev_manifests"].get("heldout-charts", {}).get("rows") == 1, "held-out charts dev manifest (1 row)")
    by = lane.get("hard_direct_by_source") or {}
    ok &= check(by.get("direct-leftover", 0) > 0 and by.get("direct-image", 0) == rep["image_adjust"]["direct_image_rows"] > 0,
                f"direct rows by source (256 lane): {by}; every direct image row trained (--direct-must default)")
    ok &= check(rep["image_adjust"].get("image_share_final", 0) <= 0.40 + 1e-9,
                f"image share capped at 40%: {rep['image_adjust']}; final shares {rep['shares_final']}")
    ok &= check(all(v["exit"] == 0 for k, v in rep["gate"].items() if not k.endswith("-gui")),
                f"leakage gate: {({k: v['result'] for k, v in rep['gate'].items()})}")
    ok &= check(all(v["shares_ok"] for v in rep["lanes"].values()), f"shares {({k: v['shares'] for k, v in rep['lanes'].items()})}")
    ok &= check("decision-p3ltdry-heldout-charts" in rep.get("published", {}), f"published: {sorted(rep.get('published', {}))}")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd, d = sys.argv[1], Path(sys.argv[2])
    sys.exit({"pod": pod, "manifest": manifest}[cmd](d))
