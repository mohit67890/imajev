#!/usr/bin/env python3
"""Build the label-then-train pod's EXTRA teacher queue, which never overlaps Azure's (reports/phase3/LABELTRAIN-RUNBOOK.md).

Reads the live session's files READ-ONLY (data/p3/stream/queue*.jsonl, data/p3/teacher/{results,status}.jsonl, data/p3/mine-pulled)
and writes only under --out-dir (default data/p3/labeltrain/), which the label bundle carries to the pod.

    make_extra_queue.py extra        (now, while Azure still runs; the coordinator is finished)
        OWNER DECISION 2026-09-26: the waiting rows are SPLIT. Rows with CONSTRUCTED gold need no teacher and go straight to training:
        direct-constructed-leftovers.jsonl (candidate rows + a constructed label, provenance.direct_source "direct-leftover";
        build_manifest.py --direct-leftovers). Only rows that need a teacher (B/C text, mined images, anything not constructed) go
        into the pod's teacher queue.
        queue-extra.jsonl       the TEACHER part of every flagged row the coordinator left WAITING behind the family quotas / call cap: its admission
                                logic replayed (first occurrence of an id across the sorted *.flagged.jsonl files; heldout-fresh ids
                                excluded; heldout-flagged-bucket rows are held-out, not training: excluded unless
                                --include-heldout-waiting); minus every id in Azure's queues or results; ordered as the coordinator
                                orders (most confidently wrong first: -priority, id); origin "mine". Checked against
                                coordinator-status.json ("waiting" / "heldout_waiting"): a mismatch is reported (and fails with --strict).
        writer-parents.jsonl    Azure-KEPT mined B/C text parents so far (the Mac ran variants.py --no-writer: they got no variants);
                                verdict rows (keep, origin mine, item) that the pod's variants.py reads as extra parents for the writer
        queue-consistency.jsonl a small stratified sample of items Azure already judged, each with Azure's verdict under "azure": the
                                pod labels it into its OWN directory (teacher-pod-consistency/) to measure FP8-vs-bf16 agreement; never
                                a training source
        gen-context/ + gen-context.json   the files scripts/p3/gen_image_joint.py needs for image-joint variants on the pod (ABO /
                                defect records, ABO listing meta, exclusions), staged outside .cache so the bundle check can scan them
        extra-summary.json
    make_extra_queue.py leftovers    (after Azure has STOPPED: cap reached or drained; refuses while its status says running)
        queue-leftover.jsonl    every row of Azure's queue.jsonl + queue-variants.jsonl with no Azure result that needs a teacher (held-out
                                rows first, then mined by priority, then variants); must not share an id with queue-extra.jsonl. The
                                same split: mined constructed-gold rows -> direct-constructed-leftovers-late.jsonl (Mac only, for
                                build_manifest); held-out constructed-gold rows are not queued (their stored gold is their eval label in
                                build_manifest); variant rows (programmatic unknown variants) keep their teacher check
        writer-parents-late.jsonl   Azure-kept B/C parents that are not in writer-parents.jsonl yet
        inbox/                  both files gzipped + SHA256SUMS: scp them to the pod's labeltrain/inbox/ (the pod installs
                                them atomically and closes the two queues)
    make_extra_queue.py audit --pod-results data/p3/teacher-pod/results.jsonl
                                proves no id has a verdict from both Azure and the pod (build_manifest prefers Azure if one ever did)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT / "scripts/p2"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from azure_teacher import read_complete, write_atomic  # noqa: E402

P3 = ROOT / "data/p3"
OUT = P3 / "labeltrain"
GEN_CONTEXT = {  # destination (repo-relative, where gen_image_joint.py reads it) -> staged name
    ".cache/datasets/v1/state_aware/abo_listing_meta.json": "abo_listing_meta.json",
    "data/decision-v1/abo/records.jsonl": "abo-records.jsonl",
    "data/decision-v1/defects/records.jsonl": "defects-records.jsonl",
    "data/decision-v1/exclusions.json": "exclusions.json",
}
LIVE_STATES = ("running", "paused_pilot_gate")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonl_write(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def ids_of(path: Path) -> set[str]:
    return {r["id"] for r in read_complete(path)[0]} if path.exists() else set()


def quota_key(item: dict) -> str:
    return (item.get("pool") or {}).get("quota_key") or item.get("family")


def in_bucket(item: dict) -> bool:
    return bool((item.get("pool") or {}).get("heldout_flagged_bucket"))


def excluded_ids(pool_dir: Path, extra_files: list[str]) -> set[str]:
    """heldout-fresh ids, exactly as stream_coordinator.py reads them (+ any --exclude-ids files)."""
    out: set[str] = set()
    for p in sorted(glob.glob(str(pool_dir / "heldout-fresh*.jsonl"))) + list(extra_files or []):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.add(json.loads(line)["id"] if line.startswith("{") else line)
    return out


DIRECT_SOURCE = "direct-leftover"


def direct_row(item: dict) -> dict:
    """A constructed-gold candidate row as a direct training row: the manifest label contract with target_kind "constructed"."""
    c = {k: v for k, v in item.items() if k != "pool"}
    c["provenance"] = dict(c.get("provenance") or {}, direct_source=DIRECT_SOURCE)
    c["label"] = {"target": c.get("gold"), "probs": None, "rationale": None, "target_kind": "constructed", "review": None}
    return c


GOLD_FIXES: dict = {"relabel": {}, "drop": set(), "counts": Counter()}   # --gold-fixes (gen_traps.py re-derived constructed golds)


def load_fixes(paths) -> None:
    import gen_traps
    for p in paths or []:
        rl, dr, _ = gen_traps.load_gold_fixes(p)
        GOLD_FIXES["relabel"].update(rl)
        GOLD_FIXES["drop"] |= dr


def fixed_direct(item: dict) -> dict | None:
    """direct_row() with the gold fixes applied: None for a dropped id, the re-derived gold (and label target) for a relabelled one."""
    import gen_traps
    if item.get("id") in GOLD_FIXES["drop"]:
        GOLD_FIXES["counts"]["direct_dropped"] += 1
        return None
    row = direct_row(item)
    if gen_traps.apply_gold_fix(row, GOLD_FIXES["relabel"]):
        GOLD_FIXES["counts"]["direct_relabelled"] += 1
    return row


TEACHER_SOURCES: set = set()      # constructed-gold sources that still go to the teacher (--teacher-sources; default none)


def needs_teacher(item: dict) -> bool:
    return item.get("gold_kind") != "constructed" or item.get("source") in TEACHER_SOURCES


def is_writer_parent(item: dict) -> bool:
    import variants as V
    return V.origin(item) == "writer" and item.get("source") in ("B", "C")


def azure_state(status_path: Path) -> tuple[str | None, float]:
    try:
        s = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, 0.0
    return s.get("state"), float(s.get("t") or 0.0)


# ------------------------------------------------------------------------------------------------ extra
def build_extra(a) -> dict:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    azure_queue_ids = ids_of(Path(a.azure_queue))
    azure_var_ids = ids_of(Path(a.azure_variants_queue))
    results = read_complete(Path(a.azure_results))[0]
    azure_result_ids = {r["id"] for r in results}
    excl = excluded_ids(Path(a.pool_dir), a.exclude_ids)
    seen: set[str] = set()
    waiting, held_waiting = {}, {}
    fresh_skipped = dup = 0
    files = sorted(Path(a.mine_pulled).glob("*.flagged.jsonl"))
    for path in files:
        for r in read_complete(path)[0]:
            if r["id"] in seen:
                dup += 1
                continue
            seen.add(r["id"])
            if r["id"] in excl:
                fresh_skipped += 1
                continue
            if r["id"] in azure_queue_ids:
                continue
            (held_waiting if in_bucket(r["item"]) else waiting)[r["id"]] = r
    # replayed coordinator: queued = Azure queue ids; waiting = everything else. Cross-check with its final status.
    check = {}
    try:
        st = json.loads(Path(a.coordinator_status).read_text())
        check = {"coordinator_waiting": st.get("waiting"), "replayed_waiting": len(waiting),
                 "coordinator_heldout_waiting": st.get("heldout_waiting"), "replayed_heldout_waiting": len(held_waiting),
                 "coordinator_flagged_seen": st.get("flagged_seen"), "replayed_flagged_seen": len(seen)}
        check["match"] = (st.get("waiting") == len(waiting) and st.get("heldout_waiting") == len(held_waiting))
    except (OSError, json.JSONDecodeError):
        check = {"match": None, "note": "no coordinator-status.json to compare with"}
    rows, direct = [], []
    now = time.time()
    overlap = held_constructed = 0
    pick = list(waiting.values()) + (list(held_waiting.values()) if a.include_heldout_waiting else [])
    for r in sorted(pick, key=lambda r: (0 if in_bucket(r["item"]) else 1, -(r.get("priority") or 0.0), r["id"])):
        if r["id"] in azure_result_ids or r["id"] in azure_var_ids:
            overlap += 1
            continue
        if not needs_teacher(r["item"]) and a.split_constructed:
            if in_bucket(r["item"]):
                held_constructed += 1          # held-out: its constructed gold is its eval label; no teacher, not training
            else:
                d = fixed_direct(r["item"])
                if d is not None:
                    direct.append(d)
            continue
        rows.append({"id": r["id"], "origin": "heldout" if in_bucket(r["item"]) else "mine", "queued_at": now,
                     "priority": r.get("priority"), "shard": r.get("shard"), "item": r["item"], "mine": r.get("mine"),
                     "extra": "labeltrain"})
    n = jsonl_write(out / "queue-extra.jsonl", rows)
    (out / "queue-extra.jsonl.closed").write_text(json.dumps({"t": now, "rows": n}) + "\n")
    nd = jsonl_write(out / "direct-constructed-leftovers.jsonl", direct)
    # writer parents: Azure-kept mined B/C text parents so far
    wp = [{"id": r["id"], "keep": True, "origin": "mine", "rule": r.get("rule"), "teacher_backend": "azure-bf16", "item": r["item"]}
          for r in results if r.get("keep") and r.get("origin", "mine") == "mine" and is_writer_parent(r["item"])]
    nwp = jsonl_write(out / "writer-parents.jsonl", wp)
    # FP8-vs-bf16 consistency sample: stratified by rule, deterministic
    cons = consistency_sample([r for r in results if r.get("reason") != "solve_failed"], a.consistency_n)
    ncons = jsonl_write(out / "queue-consistency.jsonl", cons)
    (out / "queue-consistency.jsonl.closed").write_text(json.dumps({"t": now, "rows": ncons}) + "\n")
    gen = stage_gen_context(out) if a.stage_gen_context else {}
    fam = Counter(quota_key(r["item"]) for r in rows)
    summary = {"t": now, "built_ist": time.strftime("%Y-%m-%d %H:%M", time.gmtime(now + 19800)),
               "waiting_total": len(waiting) + (len(held_waiting) if a.include_heldout_waiting else 0),
               "direct_constructed_leftovers": nd, "direct_by_source": dict(Counter(r["source"] for r in direct)),
               "direct_image_rows": sum(1 for r in direct if r.get("images")), "heldout_constructed_not_queued": held_constructed,
               "queue_extra": n, "by_origin": dict(Counter(r["origin"] for r in rows)),
               "by_source": dict(Counter(r["item"]["source"] for r in rows)),
               "image_rows": sum(1 for r in rows if r["item"].get("images")),
               "writer_parent_rows_in_extra (B/C text)": sum(1 for r in rows if is_writer_parent(r["item"])),
               "constructed_gold_rows": sum(1 for r in rows if r["item"].get("gold_kind") == "constructed"),
               "top_families": dict(fam.most_common(25)), "families": len(fam),
               "heldout_waiting_excluded": 0 if a.include_heldout_waiting else len(held_waiting),
               "heldout_fresh_skipped": fresh_skipped, "duplicate_flagged_ids": dup, "dropped_already_azure": overlap,
               "writer_parents_from_azure": nwp, "consistency_sample": ncons, "gen_context": gen,
               "coordinator_check": check, "files": len(files), "gold_fixes": dict(GOLD_FIXES["counts"]),
               "azure_results_at_build": len(azure_result_ids), "azure_queue_rows": len(azure_queue_ids)}
    write_atomic(out / "extra-summary.json", summary)
    log(f"direct-constructed-leftovers.jsonl: {nd:,} rows (no teacher); queue-extra.jsonl (teacher): {n:,} rows "
        f"({summary['by_origin']}), writer-parents {nwp:,}, consistency {ncons}; "
        f"coordinator check {check}")
    if a.strict and check.get("match") is False:
        raise SystemExit("replayed waiting set does not match coordinator-status.json (see extra-summary.json)")
    return summary


def consistency_sample(results: list[dict], n: int) -> list[dict]:
    by_rule: dict[str, list] = {}
    for r in results:
        by_rule.setdefault(r.get("rule") or "?", []).append(r)
    order = {k: sorted(v, key=lambda r: hashlib.sha1(f"consistency:{r['id']}".encode()).hexdigest()) for k, v in by_rule.items()}
    total = sum(len(v) for v in order.values()) or 1
    picked = []
    for k, v in sorted(order.items()):
        take = max(min(len(v), 10), round(n * len(v) / total))
        picked += v[:take]
    picked = sorted(picked, key=lambda r: hashlib.sha1(f"consistency:{r['id']}".encode()).hexdigest())[:max(n, 0)]
    return [{"id": r["id"], "origin": "consistency", "variant": r.get("variant"), "item": r["item"],
             "azure": {k: r.get(k) for k in ("labels", "keep", "target_probs", "rule", "reason")}} for r in picked]


def stage_gen_context(out: Path) -> dict:
    d = out / "gen-context"
    d.mkdir(parents=True, exist_ok=True)
    got = {}
    for dest, name in GEN_CONTEXT.items():
        src = ROOT / dest
        if src.exists():
            shutil.copy2(src, d / name)
            got[dest] = f"gen-context/{name}"
    write_atomic(out / "gen-context.json", {"files": got, "note": "the pod copies each staged file to its destination "
                                                                  "(repo-relative) before variants.py starts"})
    return got


# ------------------------------------------------------------------------------------------------ leftovers
def build_leftovers(a) -> dict:
    out = Path(a.out_dir)
    state, t = azure_state(Path(a.azure_status))
    if state in LIVE_STATES and time.time() - t < a.stale_secs and not a.allow_running:
        raise SystemExit(f"Azure teacher status is '{state}' (updated {int(time.time() - t)} s ago): leftovers are built only after "
                         "it has stopped (cap reached / drained), or they could overlap. Wait, or --allow-running if you are sure.")
    vq = Path(a.azure_variants_queue)
    if vq.exists() and not (vq.parent / (vq.name + ".closed")).exists() and not a.allow_open_variants:
        raise SystemExit(f"{vq.name} has no .closed marker: the Mac's variants.py may still append programmatic unknown variants after "
                         "the teacher stopped. Wait until it exits (it closes the queue), or --allow-open-variants.")
    results = read_complete(Path(a.azure_results))[0]
    have = {r["id"] for r in results}
    extra_ids = ids_of(out / "queue-extra.jsonl")
    rows, seen, direct = [], set(), []
    held_constructed = 0
    for qpath in (Path(a.azure_queue), Path(a.azure_variants_queue)):
        for q in read_complete(qpath)[0]:
            if q["id"] in have or q["id"] in seen:
                continue
            seen.add(q["id"])
            origin = q.get("origin", "mine")
            if a.split_constructed and origin in ("mine", "heldout") and not needs_teacher(q["item"]):
                if origin == "heldout":
                    held_constructed += 1
                else:
                    d = fixed_direct(q["item"])
                    if d is not None:
                        direct.append(d)
                continue
            rows.append(dict(q, leftover_of="azure"))
    clash = [r["id"] for r in rows if r["id"] in extra_ids]
    if clash:
        raise SystemExit(f"{len(clash)} leftover ids are also in queue-extra.jsonl (e.g. {clash[:3]}): refusing")
    rank = {"heldout": 0, "mine": 1, "variant": 2}
    rows.sort(key=lambda r: (rank.get(r.get("origin", "mine"), 3), -(r.get("priority") or 0.0), r["id"]))
    n = jsonl_write(out / "queue-leftover.jsonl", rows)
    nd = jsonl_write(out / "direct-constructed-leftovers-late.jsonl", direct)
    early = ids_of(out / "writer-parents.jsonl")
    late = [{"id": r["id"], "keep": True, "origin": "mine", "rule": r.get("rule"), "teacher_backend": "azure-bf16", "item": r["item"]}
            for r in results if r.get("keep") and r.get("origin", "mine") == "mine" and is_writer_parent(r["item"])
            and r["id"] not in early]
    nl = jsonl_write(out / "writer-parents-late.jsonl", late)
    inbox = out / "inbox"
    shutil.rmtree(inbox, ignore_errors=True)
    inbox.mkdir(parents=True)
    sums = []
    for name in ("queue-leftover.jsonl", "writer-parents-late.jsonl"):
        gz = inbox / f"{name}.gz"
        with open(out / name, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        sums.append(f"{hashlib.sha256(gz.read_bytes()).hexdigest()}  {gz.name}")
    (inbox / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    summary = {"t": time.time(), "azure_state": state, "azure_results": len(have), "queue_leftover": n,
               "direct_constructed_leftovers_late": nd, "heldout_constructed_not_queued": held_constructed,
               "gold_fixes": dict(GOLD_FIXES["counts"]),
               "by_origin": dict(Counter(r.get("origin", "mine") for r in rows)),
               "image_rows": sum(1 for r in rows if r["item"].get("images")), "writer_parents_late": nl,
               "inbox_bytes": sum(p.stat().st_size for p in inbox.iterdir())}
    write_atomic(out / "leftover-summary.json", summary)
    log(f"queue-leftover.jsonl (teacher): {n:,} rows {summary['by_origin']}; direct-constructed-leftovers-late {nd:,} (no teacher); "
        f"writer-parents-late {nl:,}; inbox {inbox} "
        f"({summary['inbox_bytes'] / 1e6:.1f} MB)")
    return summary


# ------------------------------------------------------------------------------------------------ audit
def audit(a) -> dict:
    az = {r["id"] for r in read_complete(Path(a.azure_results))[0]}
    pod = {r["id"] for p in a.pod_results for r in read_complete(Path(p))[0]}
    both = sorted(az & pod)
    rep = {"azure_results": len(az), "pod_results": len(pod), "overlap": len(both), "examples": both[:10],
           "status": "PASS" if not both else "OVERLAP (build_manifest keeps the Azure verdict)"}
    print(json.dumps(rep, indent=1))
    return rep


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extra", "leftovers", "audit"):
        p = sub.add_parser(name)
        if name != "audit":
            p.add_argument("--no-split-constructed", dest="split_constructed", action="store_false",
                           help="queue constructed-gold rows for the teacher too (before the owner's 2026-09-26 decision)")
            p.add_argument("--gold-fixes", nargs="*", default=[],
                           help="gen_traps.py --gold-fixes JSONL(s): relabel / drop direct constructed leftovers by id")
            p.add_argument("--teacher-sources", default="",
                           help="comma list of sources whose constructed-gold rows still go to the teacher, e.g. D,I (on Azure the "
                                "teacher disagreed with constructed gold on 32%% of D and 27%% of constructed images; those were dropped)")
        p.add_argument("--out-dir", default=str(OUT))
        p.add_argument("--azure-queue", default=str(P3 / "stream/queue.jsonl"))
        p.add_argument("--azure-variants-queue", default=str(P3 / "stream/queue-variants.jsonl"))
        p.add_argument("--azure-results", default=str(P3 / "teacher/results.jsonl"))
        p.add_argument("--azure-status", default=str(P3 / "teacher/status.json"))
        if name == "extra":
            p.add_argument("--mine-pulled", default=str(P3 / "mine-pulled"))
            p.add_argument("--pool-dir", default=str(P3 / "pool"))
            p.add_argument("--exclude-ids", action="append", default=[])
            p.add_argument("--coordinator-status", default=str(P3 / "stream/coordinator-status.json"))
            p.add_argument("--include-heldout-waiting", action="store_true",
                           help="also queue the flagged held-out rows left behind heldout_flagged_cap (origin heldout; eval labels)")
            p.add_argument("--consistency-n", type=int, default=400)
            p.add_argument("--no-gen-context", dest="stage_gen_context", action="store_false")
            p.add_argument("--strict", action="store_true", help="fail when the replayed waiting set differs from the coordinator's")
        if name == "leftovers":
            p.add_argument("--allow-running", action="store_true")
            p.add_argument("--allow-open-variants", action="store_true",
                           help="build even though Azure's queue-variants.jsonl is not closed yet")
            p.add_argument("--stale-secs", type=float, default=600.0,
                           help="a 'running' status older than this counts as stopped (the runner died)")
        if name == "audit":
            p.add_argument("--pod-results", action="append", default=None)
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    TEACHER_SOURCES.clear()
    TEACHER_SOURCES.update(x.strip() for x in getattr(a, "teacher_sources", "").split(",") if x.strip())
    GOLD_FIXES["relabel"].clear()
    GOLD_FIXES["drop"].clear()
    GOLD_FIXES["counts"].clear()
    load_fixes(getattr(a, "gold_fixes", None))
    if GOLD_FIXES["relabel"] or GOLD_FIXES["drop"]:
        log(f"gold fixes {a.gold_fixes}: {len(GOLD_FIXES['relabel']):,} relabel ids, {len(GOLD_FIXES['drop']):,} drop ids")
    if a.cmd == "extra":
        build_extra(a)
    elif a.cmd == "leftovers":
        build_leftovers(a)
    else:
        a.pod_results = a.pod_results or [str(P3 / "teacher-pod/results.jsonl")]
        return 0 if audit(a)["overlap"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
