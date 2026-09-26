#!/usr/bin/env python3
"""Assemble the phase-3 candidate pool for streamed mining + teacher (docs/phase-3-plan.md rev 4: Stage 0 -> Stage 1).

    .venv/bin/python scripts/p3/assemble_pool.py                       # build everything under data/p3/pool/
    .venv/bin/python scripts/p3/assemble_pool.py take-flagged --mined data/p3/mine-pulled/*.flagged.jsonl   # after mining (hook)

Build steps (report: reports/phase3/pool.md, numbers: data/p3/pool/pool-manifest.json):
 a. merge every clean source (data/p3/candidates-clean/<SOURCE_FILES>.jsonl; drops-*/flagged-* are side files) and remove
    cross-source near-duplicates (scripts/p3/assembly_dedup.py: decontam's 13-gram normalisation, keep the public-licensed /
    source-B / richer copy; pairs in dedup-pairs.jsonl);
 b. add ~1,500 255-option items (family large_choice, scripts/p3/assembly_large_choice.py) -> data/p3/pool/large-choice.jsonl.
    They are NOT in the mining shards (the shipped 255-code model cannot serve 255 options) and need no teacher (constructed
    gold); scripts/p3/build_manifest.py puts them into the 256-code lanes only;
 c. held-out sets, split by whole group (scripts/p3/assembly_common.assign_groups):
      heldout-fresh.jsonl              fresh-seed generator items (reasoning, policy, traps, image_joint) + a ~2% slice of the B/C
                                       groups (<= 5% of the plumb / finqa / tatqa groups), never mined
      heldout-fresh-gui.jsonl          fresh-seed GUI screens (own file: fixed state strings, see assembly_heldout.py)
      heldout-fresh-large-choice.jsonl fresh-seed 255-option items (256-code lane evals only)
      heldout-flagged.jsonl            written later by `take-flagged` from the mining output (hook below)
    Everything held out leaves the pool with its whole group; pool rows that still link to a held-out row (13-gram / state /
    upstream / family) leave with their whole group; `decontam.py --leakage` then runs pool vs held-out and the build FAILS
    on any link;
 d. per-family streaming quotas (scripts/p3/assembly_quotas.py) -> quotas.json, and family-balanced shuffled shards
    shards/pool-NN.jsonl + pool-manifest.json (counts per source/family/difficulty/type, sha256 per shard).

Shard rows are candidate rows (scripts/p3/candidate.py) plus a "pool" object:
    {"group": <split group>, "quota_key": <family, 'img:' prefix for image rows>, "shard": k,
     "heldout_flagged_bucket": <bool: the group is in the 5% reserved for the flagged held-out half>}
A flagged row whose bucket is true goes to heldout-flagged (never to training); the builder drops every bucket group from
training. The bucket is a stable hash of the group, so the streamer decides it online without a global pass.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly_common as ac  # noqa: E402
import assembly_dedup as ad  # noqa: E402
import assembly_heldout as ah  # noqa: E402
import assembly_large_choice as alc  # noqa: E402
import assembly_quotas as aq  # noqa: E402
from candidate import validate  # noqa: E402

POOL = ac.POOL_DIR
REPORT = ac.ROOT / "reports/phase3/pool.md"
FLAGGED_SALT = "p3-heldout-flagged-v1"
FLAGGED_SHARE = 0.05
BC_SALT = "p3-heldout-bc-v1"
BC_OVERSAMPLE = 3            # B/C candidate groups are drawn at 3x the slice rate, then the lowest-link ones are kept

# fresh-seed selection targets (rows per family); oversampling factors are in build_heldout()
REASONING_TARGET = 110
POLICY_TARGET = 150          # per policy family, traps included (a trap shares its parent's group)
ROUTING_TARGET = 60
IMAGE_JOINT_TARGET = 400
GUI_TARGET = 400
LARGE_CHOICE_HELDOUT = 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ steps
def load_sources(large_choice: int, lc_seed: str):
    rows, lines, files = [], [], []
    inputs = {}
    for f in ac.SOURCE_FILES:
        p = ac.CLEAN_DIR / f"{f}.jsonl"
        inputs[f"{f}.jsonl"] = ac.sha256_file(p)
        with open(p, "rb") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                    lines.append(line)
                    files.append(f)
    lc = alc.generate(lc_seed, large_choice) if large_choice else []
    for r in lc:
        rows.append(r)
        lines.append((json.dumps(r, ensure_ascii=False) + "\n").encode())
        files.append("A-large-choice")
    return rows, lines, files, inputs


def check_rows(rows) -> None:
    bad = [(r.get("id"), e) for r in rows if (e := validate(r))]
    if bad:
        raise SystemExit(f"{len(bad)} invalid candidate rows, e.g. {bad[:5]}")
    ids = collections.Counter(r["id"] for r in rows)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate ids: {dup[:10]}")


_ROWS: list = []


def _trainer_work(span):
    import assembly_manifest as am
    return [(i, why) for i in range(*span) if (why := am.trainer_problem(_ROWS[i]))]


def trainer_check(rows: list[dict], workers: int = 0) -> dict[int, str]:
    """{row index: problem} for rows that cannot be rendered by the trainer / accepted by the serving contract."""
    import multiprocessing as mp
    import os
    global _ROWS
    _ROWS = rows
    spans = [(s, min(len(rows), s + 2000)) for s in range(0, len(rows), 2000)]
    out = {}
    with mp.get_context("fork").Pool(workers or max(1, min(12, (os.cpu_count() or 2) - 2))) as pool:
        for res in pool.imap_unordered(_trainer_work, spans):
            out.update(res)
    _ROWS = []
    return out


def build_heldout(tag: str, policy_seed: int, work: Path):
    """Oversampled fresh-seed candidates: (main rows, gui rows, large-choice rows, per-family targets)."""
    work.mkdir(parents=True, exist_ok=True)
    main = []
    main += ah.fresh_reasoning(tag, int(REASONING_TARGET * 1.6), work)
    pol, pol_file = ah.fresh_policy(policy_seed, tag, POLICY_TARGET * 3, work)
    main += pol
    main += ah.fresh_traps(policy_seed, tag, 1200, pol_file, work)
    main += ah.fresh_image_joint(tag, IMAGE_JOINT_TARGET * 10)
    gui = ah.fresh_gui(tag, int(GUI_TARGET * 1.5), POOL / "heldout-images" / "gui")
    lc = ah.fresh_large_choice(tag, int(LARGE_CHOICE_HELDOUT * 1.5))
    targets = {f: REASONING_TARGET for f in ah.REASONING_FAMILIES}
    targets.update({f: POLICY_TARGET for f in ah.POLICY_FAMILIES})
    targets.update({"routing_hard": ROUTING_TARGET, "image_joint_rule": IMAGE_JOINT_TARGET, "gui_action": GUI_TARGET,
                    "large_choice": LARGE_CHOICE_HELDOUT})
    import assembly_manifest as am
    for r in main + gui + lc:
        errs = validate(r)
        if errs:
            raise SystemExit(f"fresh held-out row {r['id']} invalid: {errs}")
    bad = {r["id"] for r in main + gui + lc if am.trainer_problem(r)}
    if bad:
        log(f"  {len(bad)} fresh candidates fail the trainer/serving contract (dropped), e.g. {sorted(bad)[:3]}")
    return [r for r in main if r["id"] not in bad], [r for r in gui if r["id"] not in bad], \
        [r for r in lc if r["id"] not in bad], targets


def write_shards(rows: list[dict], groups: dict[str, str], n_shards: int, salt: str = "p3-pool-shards-v1") -> list[dict]:
    """Family-balanced: each quota_key's rows (stable-hash order) are dealt round-robin over the shards from a per-family offset,
    then each shard is ordered by a stable hash (families interleaved). Returns per-shard summaries."""
    by_key: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_key[aq.quota_key(r)].append(r)
    shards: list[list] = [[] for _ in range(n_shards)]
    for k, rs in sorted(by_key.items()):
        rs.sort(key=lambda r: ac.stable_hash(salt, r["id"]))
        off = int(ac.stable_hash(salt, "offset", k)[:8], 16) % n_shards
        for j, r in enumerate(rs):
            shards[(off + j) % n_shards].append(r)
    sdir = POOL / "shards"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    out = []
    for s, rs in enumerate(shards):
        rs.sort(key=lambda r: ac.stable_hash(salt, "order", r["id"]))
        path = sdir / f"pool-{s:02d}.jsonl"
        for r in rs:
            g = groups[r["id"]]
            r["pool"] = {"group": g, "quota_key": aq.quota_key(r), "shard": s,
                         "heldout_flagged_bucket": ac.stable_unit(FLAGGED_SALT, g) < FLAGGED_SHARE}
        ac.write_jsonl(path, rs)
        out.append({"path": str(path.relative_to(ac.ROOT)), "rows": len(rs), "sha256": ac.sha256_file(path),
                    "by_source": dict(sorted(collections.Counter(r["source"] for r in rs).items())),
                    "image_rows": sum(bool(r.get("images")) for r in rs)})
    return out


def run_gate(train: list[Path], heldout: list[Path], report: Path) -> tuple[int, str]:
    cmd = [sys.executable, str(ac.ROOT / "scripts/p3/decontam.py"), "--leakage", "--train", *map(str, train), "--heldout",
           *map(str, heldout), "--report", str(report)]
    res = subprocess.run(cmd, cwd=ac.ROOT, capture_output=True, text=True)
    return res.returncode, (res.stdout.strip().splitlines() or [res.stderr.strip()])[0]


def counts(rows) -> dict:
    return {
        "rows": len(rows),
        "by_source": dict(sorted(collections.Counter(r["source"] for r in rows).items())),
        "by_dataset": dict(sorted(collections.Counter(r["dataset"] for r in rows).items())),
        "by_family": dict(sorted(collections.Counter(r["family"] for r in rows).items())),
        "by_source_family": dict(sorted(collections.Counter(f"{r['source']}:{r['family']}" for r in rows).items())),
        "by_difficulty": {str(k): v for k, v in sorted(collections.Counter(r["difficulty"] for r in rows).items())},
        "by_type": dict(sorted(collections.Counter(r["field"]["type"] for r in rows).items())),
        "unknown_gold": sum(r["gold"] is None for r in rows),
        "image_rows": sum(bool(r.get("images")) for r in rows),
    }


# ------------------------------------------------------------------------------------------------ build
def build(args) -> int:
    t0 = time.time()
    POOL.mkdir(parents=True, exist_ok=True)
    work = POOL / "work"
    log("loading clean sources")
    rows, lines, files, inputs = load_sources(args.large_choice, args.lc_seed)
    check_rows(rows)
    n_in = collections.Counter(files)
    file_of = {r["id"]: f for r, f in zip(rows, files)}
    log("trainer/serving contract check (every row through decision_data.render)")
    problems = trainer_check(rows, args.workers)
    file_of_idx = list(files)
    contract_drop = collections.Counter()
    contract_ids = []
    for i, why in problems.items():
        contract_drop[f"{files[i]}: {why[:70]}"] += 1
        contract_ids.append(rows[i]["id"])
    keep = [i for i in range(len(rows)) if i not in problems]
    rows, lines, files = [rows[i] for i in keep], [lines[i] for i in keep], [files[i] for i in keep]
    log(f"  dropped {len(problems):,}: {dict(contract_drop)}")
    log(f"  {len(rows):,} rows ({sum(v for k, v in n_in.items() if k != 'A-large-choice'):,} clean + "
        f"{n_in['A-large-choice']:,} large_choice), all valid")

    # a. cross-source near-duplicates
    log("cross-source near-duplicate scan")
    g0 = ac.assign_groups(rows)
    losers, pairs, skipped = ad.cross_source_duplicates(rows, lines, files, [g0[r["id"]] for r in rows], args.workers)
    ac.write_jsonl(POOL / "dedup-pairs.jsonl", pairs)
    dedup_by_file = collections.Counter(files[i] for i in losers)
    dedup_by_dataset = collections.Counter(rows[i]["dataset"] for i in losers)
    dedup_pairs_by = collections.Counter(f"{p['kept_file']} > {p['dropped_file']} ({p['unit']})" for p in pairs)
    keep = [i for i in range(len(rows)) if i not in losers]
    rows, lines, files = [rows[i] for i in keep], [lines[i] for i in keep], [files[i] for i in keep]
    log(f"  removed {len(losers):,} rows ({dict(dedup_by_dataset)}) from {len(pairs):,} pairs; not duplicates: {dict(skipped)}")

    groups = ac.assign_groups(rows)
    pool_groups = [groups[r["id"]] for r in rows]

    # c. held-out candidates
    log("fresh-seed held-out generation")
    main, gui, lcf, targets = build_heldout(args.tag, args.policy_seed, work)
    fresh = main + gui + lcf
    fg = ac.assign_groups(fresh)
    held_group = {r["id"]: "fresh:" + fg[r["id"]] for r in fresh}
    gui_ids = {r["id"] for r in gui}
    log(f"  generated {len(main):,} main + {len(gui):,} GUI + {len(lcf):,} large-choice candidates")
    bc_cand = ah.bc_candidate_groups(rows, groups, args.bc_share * BC_OVERSAMPLE, BC_SALT)
    cand_rows = [r for r in rows if groups[r["id"]] in bc_cand]
    for r in cand_rows:
        held_group[r["id"]] = groups[r["id"]]
    log(f"  B/C candidates: {len(bc_cand):,} groups, {len(cand_rows):,} rows")

    log("link costs of the candidates")
    alive = [g not in bc_cand for g in pool_groups]
    costs = ah.link_costs(lines, pool_groups, alive, fresh + cand_rows, held_group, gui_ids)
    chosen = ah.select_by_cost(fresh, held_group, costs, targets, args.tag)
    bc = ah.select_bc_groups(rows, groups, bc_cand, costs, args.bc_share, args.scarce_cap, BC_SALT, args.max_links)
    bc_rows = [r for r in cand_rows if groups[r["id"]] in bc]
    log(f"  selected {len(chosen):,} fresh rows (of {len(fresh):,}) and {len(bc):,} B/C groups ({len(bc_rows):,} rows) by link cost")
    held = chosen + bc_rows

    log("resolving held-out/pool links")
    removed, dropped, rounds = ah.resolve_links(lines, pool_groups, held, held_group, bc, gui_ids, args.max_links, log=log)
    held = [h for h in held if held_group[h["id"]] not in dropped]
    held_ids = {h["id"] for h in held}
    bc_final = {g for g in bc if g not in dropped}
    bc_discarded = bc - bc_final
    pool_rows = [r for r, g in zip(rows, pool_groups) if g not in removed and g not in bc]
    removed_rows = [r for r, g in zip(rows, pool_groups) if g in removed or g in bc_discarded]
    log(f"  held-out {len(held):,} rows; removed {len(removed_rows):,} linked pool rows ({len(removed):,} groups); "
        f"{len(pool_rows):,} pool rows left")

    chosen_ids = {r["id"] for r in chosen}
    for h in held:
        if h["id"] not in chosen_ids:
            h["provenance"]["heldout"] = "pool_slice"
    fresh_main = [h for h in held if h["id"] not in gui_ids and h["dataset"] != "large_choice"]
    fresh_gui = [h for h in held if h["id"] in gui_ids]
    fresh_lc = [h for h in held if h["dataset"] == "large_choice"]
    assert len(fresh_main) + len(fresh_gui) + len(fresh_lc) == len(held_ids)
    # the held-out GUI PNGs that were not selected are not needed
    keep_png = {Path(p).name for h in fresh_gui for p in h["images"]}
    for png in (POOL / "heldout-images" / "gui").glob("*.png"):
        if png.name not in keep_png:
            png.unlink()

    paths = {"fresh": POOL / "heldout-fresh.jsonl", "gui": POOL / "heldout-fresh-gui.jsonl",
             "lc": POOL / "heldout-fresh-large-choice.jsonl"}
    ac.write_jsonl(paths["fresh"], fresh_main)
    ac.write_jsonl(paths["gui"], fresh_gui)
    ac.write_jsonl(paths["lc"], fresh_lc)
    ac.write_jsonl(POOL / "removed-by-links.jsonl",
                   ({"id": r["id"], "source": r["source"], "dataset": r["dataset"], "family": r["family"],
                     "group": groups[r["id"]]} for r in removed_rows))

    # d. shards, large-choice file, quotas
    lc_pool = [r for r in pool_rows if r["dataset"] == "large_choice"]
    mine_rows = [r for r in pool_rows if r["dataset"] != "large_choice"]
    lc_path = POOL / "large-choice.jsonl"
    for r in lc_pool:
        r["pool"] = {"group": groups[r["id"]], "quota_key": aq.quota_key(r), "shard": None, "heldout_flagged_bucket": False}
    ac.write_jsonl(lc_path, lc_pool)
    log(f"writing {args.shards} shards")
    shard_info = write_shards(mine_rows, groups, args.shards)
    quotas = aq.compute(mine_rows, cap_total=args.cap_total, image_share=args.image_share)
    (POOL / "quotas.json").write_text(json.dumps(quotas, indent=1) + "\n")

    # gate
    log("leakage gate (decontam.py --leakage)")
    shard_paths = [ac.ROOT / s["path"] for s in shard_info]
    rc, msg = run_gate(shard_paths + [lc_path], [paths["fresh"], paths["lc"]], POOL / "leakage-gate.md")
    log(f"  main: {msg}")
    gate = {"main": {"exit": rc, "result": msg, "report": str((POOL / 'leakage-gate.md').relative_to(ac.ROOT))}}
    rc_g, msg_g = run_gate(shard_paths + [lc_path], [paths["gui"]], POOL / "leakage-gate-gui.md")
    kinds = dict(re.findall(r"(\w+)=(\d+)", msg_g.split("FAIL", 1)[1])) if "FAIL" in msg_g else {}
    gui_ok = rc_g == 0 or set(kinds) <= {"state"}
    gate["gui"] = {"exit": rc_g, "result": msg_g, "only_fixed_state_links": gui_ok,
                   "report": str((POOL / 'leakage-gate-gui.md').relative_to(ac.ROOT))}
    log(f"  GUI: {msg_g} (only fixed-state links: {gui_ok})")
    pool_png = {Path(p).name for r in mine_rows if r["dataset"] == "gui_synthetic" for p in r["images"]}
    gate["gui"]["shared_screenshots_with_pool"] = len(keep_png & pool_png)
    img_pool = collections.Counter(s for r in mine_rows for s in ac.image_shas(r))
    gate["image_reuse"] = {
        "heldout_fresh_image_rows": sum(bool(h.get("images")) for h in fresh_main),
        "heldout_fresh_rows_with_a_pool_photo": sum(any(img_pool[s] for s in ac.image_shas(h)) for h in fresh_main if h.get("images")),
        "note": "photos are reused across families by design (the photo pool is finite); the leakage rule is textual + "
                "upstream item. Fresh image_joint items were chosen for the fewest upstream (photo item) links.",
    }
    manifest = {
        "version": "p3-pool-v1",
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "command": "scripts/p3/assemble_pool.py " + " ".join(sys.argv[1:]),
        "inputs": inputs,
        "parameters": {k: v for k, v in vars(args).items() if k != "cmd"},
        "row_format": "candidate row (scripts/p3/candidate.py) + pool{group, quota_key, shard, heldout_flagged_bucket}",
        "counts": {"clean_input_rows": sum(v for k, v in n_in.items() if k != "A-large-choice"),
                   "large_choice_generated": n_in["A-large-choice"], "contract_dropped": len(problems),
                   "dedup_removed": len(losers),
                   "heldout_rows": len(held), "removed_by_links": len(removed_rows), "shard_rows": len(mine_rows),
                   "large_choice_pool_rows": len(lc_pool)},
        "contract_drops": {"by_file_reason": dict(contract_drop), "ids": contract_ids},
        "by_file": {f: {"input": n_in[f], "contract_dropped": sum(1 for i in problems if file_of_idx[i] == f),
                        "dedup_removed": dedup_by_file.get(f, 0),
                        "heldout_slice": sum(1 for h in held if file_of.get(h["id"]) == f),
                        "removed_by_links": sum(1 for r in removed_rows if file_of[r["id"]] == f),
                        "pool": sum(1 for r in pool_rows if file_of[r["id"]] == f)} for f in sorted(n_in)},
        "dedup": {"pairs": len(pairs), "removed_by_file": dict(dedup_by_file), "removed_by_dataset": dict(dedup_by_dataset),
                  "pairs_by_kind": dict(dedup_pairs_by), "not_counted": dict(skipped)},
        "heldout": {
            "fresh": {"path": str(paths["fresh"].relative_to(ac.ROOT)), "sha256": ac.sha256_file(paths["fresh"]), **counts(fresh_main),
                      "by_origin": dict(collections.Counter(h["provenance"]["heldout"] for h in fresh_main))},
            "gui": {"path": str(paths["gui"].relative_to(ac.ROOT)), "sha256": ac.sha256_file(paths["gui"]), **counts(fresh_gui),
                    "images_dir": "data/p3/pool/heldout-images/gui"},
            "large_choice": {"path": str(paths["lc"].relative_to(ac.ROOT)), "sha256": ac.sha256_file(paths["lc"]), **counts(fresh_lc)},
            "flagged": {"path": "data/p3/pool/heldout-flagged.jsonl", "status": "hook: `assemble_pool.py take-flagged` after mining",
                        "bucket_salt": FLAGGED_SALT, "bucket_share": FLAGGED_SHARE,
                        "shard_rows_in_bucket": sum(r["pool"]["heldout_flagged_bucket"] for r in mine_rows)},
            "selection": {"targets": targets, "fresh_candidates": len(fresh), "chosen_before_links": len(chosen),
                          "bc_candidate_groups": len(bc_cand), "bc_groups_selected": len(bc), "bc_groups_kept": len(bc_final),
                          "bc_groups_discarded": len(bc_discarded),
                          "bc_slice_by_dataset": dict(collections.Counter(h["dataset"] for h in held if h["id"] not in chosen_ids)),
                          "max_pool_links": args.max_links,
                          "link_rounds": rounds},
        },
        "removed_by_links": counts(removed_rows) if removed_rows else {"rows": 0},
        "pool": counts(mine_rows),
        "large_choice": {"path": str(lc_path.relative_to(ac.ROOT)), "sha256": ac.sha256_file(lc_path), **counts(lc_pool)},
        "shards": shard_info,
        "quotas": {"path": "data/p3/pool/quotas.json", "first_solve_total": quotas["first_solve_total"],
                   "expected_total_calls": quotas["expected_total_calls"], "cap_total": quotas["cap_total_teacher_calls"]},
        "gate": gate,
        "seconds": round(time.time() - t0, 1),
    }
    (POOL / "pool-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    if not args.no_report:
        import assembly_report
        assembly_report.write_pool_report(manifest, quotas, REPORT)
        log(f"report: {REPORT.relative_to(ac.ROOT)}")
    if rc != 0:
        log("LEAKAGE GATE FAILED")
        return 1
    if not gui_ok:
        log("GUI held-out has non-state links")
        return 1
    log(f"done in {time.time() - t0:.0f}s")
    return 0


# ------------------------------------------------------------------------------------------------ hook: heldout-flagged
def take_flagged(args) -> int:
    """After mining: flagged rows of the reserved bucket -> heldout-flagged.jsonl (+ heldout-flagged-gui.jsonl).

    Same rules as the fresh set: every non-bucket pool row that links to a held-out row is excluded from training
    (heldout-flagged-exclude.json, read by build_manifest.py); a held-out group that would exclude more than --max-links
    training rows is not held out (it is in the bucket, so it is never trained on either); GUI rows are exempt from the
    fixed-state rule and gated separately. Then `decontam.py --leakage` must PASS."""
    flagged = set()
    for p in args.mined:
        for r in ac.read_jsonl(p):
            if r.get("flagged") or (r.get("mine") or {}).get("flagged"):
                flagged.add(r["id"])
    held, train_lines, train_ids = [], [], []
    pool_dir = Path(args.pool_dir) if getattr(args, "pool_dir", None) else POOL
    for p in sorted((pool_dir / "shards").glob("pool-*.jsonl")):
        with open(p, "rb") as fh:
            for line in fh:
                r = json.loads(line)
                if r["pool"]["heldout_flagged_bucket"]:
                    if r["id"] in flagged:
                        r["provenance"]["heldout"] = "flagged"
                        held.append(r)
                else:
                    train_lines.append(line)
                    train_ids.append(r["id"])
    gui_ids = {r["id"] for r in held if r["dataset"] == "gui_synthetic"}
    hg = {r["id"]: r["pool"]["group"] for r in held}
    by_h: dict[str, set] = collections.defaultdict(set)
    if held:
        for i, reason, hid in ac.scan_links(train_lines, held):
            if not (reason == "state" and hid in gui_ids):
                by_h[hg[hid]].add(i)
    too_costly = {g for g, idx in by_h.items() if len(idx) > args.max_links}
    kept = [r for r in held if hg[r["id"]] not in too_costly]
    exclude = sorted({train_ids[i] for g, idx in by_h.items() if g not in too_costly for i in idx})
    out = Path(args.out)
    main_rows = [r for r in kept if r["id"] not in gui_ids]
    gui_rows = [r for r in kept if r["id"] in gui_ids]
    ac.write_jsonl(out, main_rows)
    gui_out = out.with_name(out.stem + "-gui.jsonl")
    ac.write_jsonl(gui_out, gui_rows)
    excl_path = out.with_name("heldout-flagged-exclude.json")
    excl_path.write_text(json.dumps({"exclude_train_ids": exclude, "max_links": args.max_links,
                                     "groups_not_held_out_too_costly": len(too_costly),
                                     "rows_not_held_out_too_costly": len(held) - len(kept)}, indent=1) + "\n")
    summary = {"flagged_ids": len(flagged), "bucket_flagged_rows": len(held), "heldout_flagged": len(main_rows),
               "heldout_flagged_gui": len(gui_rows), "not_held_out_too_costly": len(held) - len(kept),
               "train_rows_excluded_for_links": len(exclude),
               "by_family": dict(collections.Counter(r["family"] for r in kept).most_common())}
    print(json.dumps(summary, indent=1))
    rc = 0
    if kept:
        ex = set(exclude)
        tmp = out.with_name(out.stem + ".train-tmp.jsonl")
        ac.write_jsonl(tmp, (json.loads(l) for l, i in zip(train_lines, train_ids) if i not in ex))
        if main_rows:
            rc, msg = run_gate([tmp], [out], out.with_name("leakage-gate-flagged.md"))
            print(msg)
        if gui_rows:
            rc_g, msg_g = run_gate([tmp], [gui_out], out.with_name("leakage-gate-flagged-gui.md"))
            kinds = set(re.findall(r"(\w+)=\d+", msg_g.split("FAIL", 1)[1])) if "FAIL" in msg_g else set()
            print(msg_g, "(only fixed-state links)" if kinds <= {"state"} else "(NON-STATE LINKS)")
            rc = rc or (0 if kinds <= {"state"} else 1)
        tmp.unlink()
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    tf = sub.add_parser("take-flagged", help="after mining: collect heldout-flagged from the reserved bucket")
    tf.add_argument("--mined", nargs="+", required=True, help="mining output rows with id + flagged (or mine.flagged)")
    tf.add_argument("--pool-dir", default=str(POOL), help="the pool (shards/) whose heldout_flagged_bucket rows are collected")
    tf.add_argument("--out", default=None, help="default <pool-dir>/heldout-flagged.jsonl")
    tf.add_argument("--max-links", type=int, default=ah.MAX_POOL_LINKS)
    ap.add_argument("--tag", default="hf1", help="fresh-seed tag for the held-out generators")
    ap.add_argument("--policy-seed", type=int, default=91001, help="gen_policy/gen_traps seed for the held-out (pool used 0)")
    ap.add_argument("--large-choice", type=int, default=1500)
    ap.add_argument("--lc-seed", default="lc1")
    ap.add_argument("--bc-share", type=float, default=0.02)
    ap.add_argument("--scarce-cap", type=float, default=0.05)
    ap.add_argument("--max-links", type=int, default=ah.MAX_POOL_LINKS)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--cap-total", type=int, default=aq.CAP_TOTAL)
    ap.add_argument("--image-share", type=float, default=aq.IMAGE_SHARE)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "take-flagged":
        args.out = args.out or str(Path(args.pool_dir) / "heldout-flagged.jsonl")
        return take_flagged(args)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
