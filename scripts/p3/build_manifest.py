#!/usr/bin/env python3
"""Build the phase-3 training manifests from teacher-verified items (docs/phase-3-plan.md rev 4, Stage 3). Runs AFTER the teacher.

    .venv/bin/python scripts/p3/build_manifest.py --verified data/p3/teacher/kept-*.jsonl [--review-dev data/p3/review/dev.jsonl]
    .venv/bin/python scripts/p3/build_manifest.py --synthetic 20000 --out-dir <scratch>     # dry run on synthetic labels
    # label-then-train (reports/phase3/LABELTRAIN-RUNBOOK.md): kept rows of BOTH teachers, Azure preferred on any shared id
    .venv/bin/python scripts/p3/build_manifest.py --teacher-dirs data/p3/teacher data/p3/teacher-pod \
        --verified data/p3/variants/constructed.jsonl data/p3/variants-pod/constructed.jsonl ...

Teacher directories (--teacher-dirs, in order of preference): each holds results.jsonl + kept.jsonl (+ heldout-results.jsonl) from
azure_teacher.py / pod_teacher.py `finalize`. The FIRST directory with a verdict on an id decides it (Azure first: a pod row whose id
Azure already judged is ignored, kept or not); kept rows are deduplicated by id; held-out verdicts merge the same way and replace
--heldout-results. Every row keeps its teacher_backend ("azure-bf16" | "pod-fp8"; rows written before the field existed take their
directory's default), carried into the record as p3.teacher_backend and counted in the report.

Direct rows (--direct-constructed data/p3/candidates-clean/I-{charts,docimg,inventory,safety,geometry,screens}.jsonl): candidate rows
with CONSTRUCTED gold (or dataset gold with human-verified labels, e.g. the Open Images part of I-safety) go STRAIGHT to training
without a teacher: hard-gold target (label target_kind "constructed", resp. "gold" for dataset gold; source I, so never under the
25%-of-A hard-gold cap), counted in the hard-image share (image rows). The mixture rule is unchanged: images stay at their shares;
when the hard-image supply (teacher-kept + direct) exceeds the hard-image share it is subsampled stratified by family x difficulty x
source, always keeping every unknown-gold row. OWNER DECISION 2026-09-26: --direct-must is ON by default: every direct IMAGE-group
row is trained ON TOP of the hard-image share, and the image total (hard images + direct image rows + image replay) may rise to
--max-image-share (0.40): above it, image replay is reduced first, then the teacher-verified hard images; direct rows are never cut;
the text rows keep their counts (their shares scale down). The final shares are logged and are what shares_ok checks.
--no-direct-must restores the old rule (direct image rows compete inside the hard-image share).
Direct LEFTOVERS (--direct-leftovers data/p3/labeltrain/direct-constructed-leftovers*.jsonl, make_extra_queue.py): the coordinator's
waiting rows with constructed gold, straight to training without a teacher (owner, 2026-09-26): hard gold (target_kind
"constructed"), text rows in the hard-TEXT share and image rows in the hard-image share, stratified like the teacher-kept rows
(every unknown-gold row kept), recorded as p3.direct_source "direct-leftover" (direct image groups: "direct-image"). Split group: the
pool group if the row is in the pool, else provenance group_id / record_id, else its first image (questions on one chart / page /
photo share a group). Their held-out files data/p3/pool/heldout-fresh-<group>.jsonl (DIRECT_GROUPS) are held out like heldout-fresh
(exclusions, leakage gate) and become dev manifests p3-dev-heldout-<group> and, with --publish, <NAME>-heldout-<group> (partition
test; separate report panels in scripts/p3/eval_checkpoint.py, never read by the selector).

Inputs: the verified rows (label contract in scripts/p3/assembly_manifest.py), the pool (data/p3/pool: shards for the id join
and the heldout-flagged bucket, large-choice.jsonl, the held-out files, heldout-flagged-exclude.json), the review dev slice,
and the replay sources:
  text replay   the phase-2c text components: decision-p2b + decision-p2 (train, hold-out domains excluded, ids read once) and
                the Eikos CC-BY slice (train; target_probs lists -> label dicts), topped up from the text rows of
                decision-v2.1-4b (train; banking77 / CLINC150 / civil_comments / ESCI excluded as in source C);
  image replay  data/decision-p2c/image-replay/replay-fresh.jsonl (12,000; replay-delta is a subset of it), topped up from the
                image rows of decision-v2.1-4b (train, images present on disk) only when more is needed.

Mixture (train partition; defaults from the plan): 45% hard text (verified text items + unknown variants; the 256-code lane also
carries every large_choice row inside this share), 20% text replay, 20% hard images (verified image items), 15% image replay.
The total is min(--total, what each share's supply allows), so the shares hold exactly (checked to +-2%). Hard rows are
subsampled proportionally stratified by family x difficulty x source (the teacher quotas already weighted the supply), always
keeping every unknown variant, every review-verified item and (256 lane) every large_choice row; replay is tiered (phase-2c
components first, v2.1 only as top-up) and proportionally stratified by source x family inside a tier.

Exclusions, before selection: held-out rows (fresh, GUI, large-choice, flagged, review) and their groups; every row of a group
in the heldout-flagged bucket; ids in heldout-flagged-exclude.json; variants whose parent's group is excluded (variants inherit
the parent's split); replay rows whose record is a held-out/hard row's source record, or that share a photo with a held-out row;
any row that links to a held-out row under the leakage rules (13-gram, state, upstream, id/family; exact photo+question for
images); A rows kept on a hard constructed gold (teacher disagreed) above 25% of A.
Train/dev: ~3% of the verified hard groups go to partition dev (same manifest).
Held-out flagged half: its evaluation label is the teacher's verdict (--heldout-results, azure_teacher.py finalize): a kept row
takes the teacher label, a row the teacher rejected leaves the eval set (a wrong label is worse than none), and a row the teacher
never saw keeps its stored gold only when that gold is constructed or dataset gold. Without the file every row keeps its stored gold.

Outputs (data/p3/manifests/):
  p3-train-256.jsonl / p3-train-255.jsonl   lane manifests (train + dev partitions). The 255 lane has no row with 255 options;
                                            it takes other hard text rows instead of large_choice, so both lanes have the same size
  p3-dev-heldout-fresh.jsonl, p3-dev-heldout-fresh-gui.jsonl, p3-dev-heldout-fresh-large-choice.jsonl (256 only),
  p3-dev-heldout-flagged.jsonl, p3-dev-review.jsonl                           dev/eval manifests (partition dev), never trained on
  manifest-report.json / manifest-report.md
The trainer reads data/manifests/<version>.jsonl. `--publish decision-p3` writes the names the pod scripts expect
(cloud/p3/pod_run_train.sh, p3.env) into data/manifests/:
  decision-p3.jsonl                     the 256-code lane (train + dev); the pod derives -r255 / -c256 / -ordinal-dev from it
  decision-p3-lane255.jsonl             the size-matched 255-code lane; cloud/p3/prep_manifests.py uses it as <V>-r255
  decision-p3-labelled.jsonl            every verified hard row (the round-2 source; only its train partition is used)
  decision-p3-heldout-fresh.jsonl       fresh-seed held-out (+ GUI), partition "test", no 255-option rows (the target)
  decision-p3-heldout-fresh-large-choice.jsonl   fresh 255-option items, partition "test" (256-code lanes; report only)
  decision-p3-heldout-flagged.jsonl     flagged held-out half (+ GUI), partition "test" (per-type T fit, reported)
  decision-p3-human-dev.jsonl           the review dev slice (owner-verified core + Kimi-agreed), partition "test"
Final gate: `decontam.py --leakage` over every lane's train+dev rows vs every held-out file must PASS (GUI: fixed-state links only).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly_common as ac  # noqa: E402
import assembly_manifest as am  # noqa: E402
import gen_traps  # noqa: E402  (gold-fixes helpers)
from candidate import validate  # noqa: E402

ROOT = ac.ROOT
SHARES = {"hard_text": 0.45, "replay_text": 0.20, "hard_image": 0.20, "replay_image": 0.15}
HOLDOUT_DOMAINS = {"telecom", "hospitality", "nonprofit_grants"}
DB_OVERLAP = {"banking77", "clinc150", "civil_comments", "esci"}
A_GOLD_CAP = 0.25
DEV_SALT = "p3-dev-v1"
SEL_SALT = "p3-mixture-v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


DIR_BACKEND = {"teacher": "azure-bf16", "teacher-pod": "pod-fp8"}
DIRECT_GROUPS = ("charts", "docimg", "inventory", "safety", "geometry", "screens")   # direct constructed image generators
FIXED_STATE_GROUPS = ("geometry", "screens")   # template state strings repeat by design: gated like GUI (state-only links allowed)


def _read_rows(path: Path) -> list[dict]:
    """Complete JSONL rows (a torn last line of a file still being written is skipped)."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_teacher_dirs(dirs: list, require_kept: bool = True) -> tuple[list[dict], dict, dict]:
    """Kept rows and held-out verdicts of several teacher output directories, the earlier directory preferred.
    -> (kept rows, {id: held-out verdict row}, stats). Each row gets `teacher_backend` (its own field, else the directory's default)."""
    stats: collections.Counter = collections.Counter()
    verdict: dict[str, int] = {}
    kept: dict[str, dict] = {}
    held: dict[str, dict] = {}
    for i, d in enumerate(Path(x) for x in dirs):
        backend = DIR_BACKEND.get(d.name, d.name)
        res = d / "results.jsonl"
        for r in (_read_rows(res) if res.exists() else []):
            if r["id"] in verdict:
                if verdict[r["id"]] != i:
                    stats[f"verdict_in_{d.name}_ignored_earlier_dir_wins"] += 1
                continue
            verdict[r["id"]] = i
        kp = d / "kept.jsonl"
        if not kp.exists():
            if require_kept:
                raise SystemExit(f"{kp} is missing: run `azure_teacher.py finalize --out {d}` (pod: pod_teacher.py finalize) first")
            continue
        for r in _read_rows(kp):
            if verdict.get(r["id"], i) != i:
                stats[f"kept_in_{d.name}_but_earlier_dir_verdict_wins"] += 1
                continue
            if r["id"] in kept:
                stats["duplicate_kept_id"] += 1
                continue
            r["teacher_backend"] = r.get("teacher_backend") or backend
            kept[r["id"]] = r
            verdict.setdefault(r["id"], i)       # a kept row is a verdict even when results.jsonl was not shipped
            stats[f"kept:{r['teacher_backend']}"] += 1
        hp = d / "heldout-results.jsonl"
        for r in (_read_rows(hp) if hp.exists() else []):
            if verdict.get(r["id"], i) != i or r["id"] in held:
                stats["heldout_verdict_earlier_dir_wins"] += 1
                continue
            r["teacher_backend"] = r.get("teacher_backend") or backend
            held[r["id"]] = r
            verdict.setdefault(r["id"], i)
            stats[f"heldout:{r['teacher_backend']}"] += 1
    return list(kept.values()), held, dict(stats)


# ------------------------------------------------------------------------------------------------ pool side
class Pool:
    def __init__(self, pool_dir: Path):
        self.dir = pool_dir
        self.rows: dict[str, dict] = {}
        for p in sorted((pool_dir / "shards").glob("pool-*.jsonl")):
            for r in ac.read_jsonl(p):
                self.rows[r["id"]] = r
        self.large_choice = list(ac.read_jsonl(pool_dir / "large-choice.jsonl")) if (pool_dir / "large-choice.jsonl").exists() else []
        for r in self.large_choice:
            self.rows[r["id"]] = r
        self.heldout: dict[str, list[dict]] = {}
        for name in ("heldout-fresh", "heldout-fresh-gui", "heldout-fresh-large-choice", "heldout-flagged", "heldout-flagged-gui",
                     *(f"heldout-fresh-{g}" for g in DIRECT_GROUPS)):
            p = pool_dir / f"{name}.jsonl"
            self.heldout[name] = list(ac.read_jsonl(p)) if p.exists() else []
        self.bucket_groups = {r["pool"]["group"] for r in self.rows.values() if r["pool"].get("heldout_flagged_bucket")}
        ex = pool_dir / "heldout-flagged-exclude.json"
        self.flagged_exclude = set(json.loads(ex.read_text())["exclude_train_ids"]) if ex.exists() else set()

    def group(self, rid: str) -> str | None:
        r = self.rows.get(rid)
        return r["pool"]["group"] if r else None



def root_group(pool: Pool, row: dict, by_id: dict) -> str:
    """The split group of a verified row: its pool group, else (variants) the group of the first pool ancestor."""
    if row.get("_group"):
        return row["_group"]
    seen = set()
    cur = row
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        g = pool.group(cur["id"])
        if g:
            return g
        par = cur.get("parent_id")
        cur = pool.rows.get(par) or by_id.get(par) if par else None
    return "orphan:" + row["id"]


# ------------------------------------------------------------------------------------------------ replay loaders
def _expand(rec: dict) -> list[dict]:
    """One row per question (the trainer's view), each remembering its source record id."""
    from decision_data import expand_fields
    out = expand_fields(rec)
    for x in out:
        x["_record_id"] = rec["id"]
    return out


def load_replay_text(p2_paths: list[Path], eikos: Path | None, v21: Path | None, need_v21: bool) -> list[dict]:
    from p2_common import project_probs, value_to_label
    out, seen = [], set()
    for p in p2_paths:
        if not p.exists():
            continue
        for r in ac.read_jsonl(p):
            if r["id"] in seen or r.get("partition") != "train" or r.get("images") or r.get("domain") in HOLDOUT_DOMAINS:
                continue
            seen.add(r["id"])
            for x in _expand(r):
                x["replay_source"] = "p2c:p2"
                out.append(x)
    if eikos and eikos.exists():
        for r in ac.read_jsonl(eikos):
            if r.get("partition") != "train" or r.get("images"):
                continue
            tp = r.get("target_probs")
            if isinstance(tp, list):
                keys = r.pop("target_probs_keys")
                probs, err = project_probs({value_to_label(k): v for k, v in zip(keys, tp)}, r["request"]["fields"][0], strict=True)
                if err:
                    continue
                r["target_probs"] = probs
            for x in _expand(r):
                x["replay_source"] = "p2c:eikos"
                out.append(x)
    if need_v21 and v21 and v21.exists():
        with open(v21) as fh:
            for line in fh:
                if '"partition": "train"' not in line:
                    continue
                r = json.loads(line)
                if r.get("partition") != "train" or r.get("images") or str(r.get("source")).lower() in DB_OVERLAP:
                    continue
                for x in _expand(r):
                    x["replay_source"] = "v2.1"
                    out.append(x)
    return out


def load_replay_images(fresh: list[Path], v21: Path | None, need_extra: bool) -> list[dict]:
    out, seen = [], set()
    for p in fresh:
        if not p.exists():
            continue
        for r in ac.read_jsonl(p):
            if r["id"] in seen or r.get("partition", "train") != "train":
                continue
            seen.add(r["id"])
            for x in _expand(r):
                x["replay_source"] = "p2c:image-replay"
                out.append(x)
    if need_extra and v21 and v21.exists():
        with open(v21) as fh:
            for line in fh:
                if '"partition": "train"' not in line:
                    continue
                r = json.loads(line)
                if r["id"] in seen or r.get("partition") != "train" or not r.get("images"):
                    continue
                if str(r.get("source")).lower() in DB_OVERLAP:
                    continue
                if not all((ROOT / i["image"]).exists() for i in r["images"]):
                    continue
                for x in _expand(r):
                    x["replay_source"] = "v2.1-images"
                    out.append(x)
    return out


# ------------------------------------------------------------------------------------------------ synthetic teacher output
def synthetic_verified(pool: Pool, n: int, seed: str = "synthetic") -> list[dict]:
    """Synthetic kept items for a dry run / tests: pool rows (no bucket) stratified by quota key, labelled with a smoothed
    one-hot on the gold (+ a two-sentence rationale); ~10% of A/D/I picks also get a constructed variant (new id, parent_id)."""
    rng = random.Random(seed)
    rows = [r for r in pool.rows.values() if r.get("pool", {}).get("shard") is not None and not r["pool"]["heldout_flagged_bucket"]]
    picked = am.stratified_take(rows, n, lambda r: r["pool"]["quota_key"], lambda r: False, seed)
    out = []
    for r in picked:
        labs = am.candidate_labels(r)
        gold = am.to_label(r, r["gold"])
        probs = {l: (0.0 if l != gold else 0.85) for l in labs}
        rest = [l for l in labs if l != gold]
        for l in rest:
            probs[l] = 0.15 / len(rest)
        lab = {"target": r["gold"], "probs": probs, "rationale": f"Synthetic rationale for {r['id']}. It names the evidence.",
               "target_kind": "teacher", "review": None}
        row = {k: v for k, v in r.items() if k != "pool"}
        out.append({**row, "label": lab})
        if r["source"] in ("A", "D", "I") and rng.random() < 0.1:
            v = am.deepcopy(row)
            v["id"] = r["id"] + "-synv1"
            v["parent_id"] = r["id"]
            if isinstance(v["state"], str):
                v["state"] = v["state"] + "\n(variant)"
            out.append({**v, "label": dict(lab, rationale="Synthetic variant.")})
    return out


# ------------------------------------------------------------------------------------------------ gate
def run_gate(train: list[Path], heldout: list[Path], report: Path) -> tuple[int, str]:
    cmd = [sys.executable, str(ROOT / "scripts/p3/decontam.py"), "--leakage", "--train", *map(str, train), "--heldout",
           *map(str, heldout), "--report", str(report)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return res.returncode, (res.stdout.strip().splitlines() or [res.stderr.strip()])[0]


# ------------------------------------------------------------------------------------------------ publish (pod names)
def publish(name: str, man_dir: Path, lane_rows: dict, hard_rows: list, dev: dict) -> dict:
    """Write the manifests under the names cloud/p3/pod_run_train.sh reads (see the module docstring)."""
    man_dir.mkdir(parents=True, exist_ok=True)

    def test_part(rows):
        return [dict(r, partition="test") for r in rows]

    def n_opts(r):
        return len(r["request"]["fields"][0].get("options") or [])
    fresh = test_part(dev.get("heldout-fresh", []) + dev.get("heldout-fresh-gui", []))
    out = {
        name: lane_rows["256"],
        f"{name}-lane255": lane_rows["255"],
        f"{name}-labelled": [{k: v for k, v in x.items() if not k.startswith("_")} for x in hard_rows],
        f"{name}-heldout-fresh": [r for r in fresh if n_opts(r) <= 254],
        f"{name}-heldout-fresh-large-choice": test_part(dev.get("heldout-fresh-large-choice", [])) + [r for r in fresh if n_opts(r) > 254],
        f"{name}-heldout-flagged": test_part(dev.get("heldout-flagged", []) + dev.get("heldout-flagged-gui", [])),
        f"{name}-human-dev": test_part(dev.get("review", [])),
    }
    for key in (f"heldout-{g}" for g in DIRECT_GROUPS):        # direct constructed image groups' held-out (when present)
        if dev.get(key):
            out[f"{name}-{key}"] = test_part(dev[key])
    if any(n_opts(r) > 254 for r in out[f"{name}-lane255"]):
        raise SystemExit("publish: the 255 lane holds a 255-option row")
    info = {}
    for n, rows in out.items():
        path = man_dir / f"{n}.jsonl"
        ac.write_jsonl(path, rows)
        info[n] = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "rows": len(rows),
                   "sha256": ac.sha256_file(path)}
        log(f"published {path} ({len(rows):,} rows)")
    return info


# ------------------------------------------------------------------------------------------------ main build
def build(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"dropped": collections.Counter()}
    drop = stats["dropped"]
    pool = Pool(Path(args.pool_dir))
    log(f"pool: {len(pool.rows):,} rows, held-out files {({k: len(v) for k, v in pool.heldout.items()})}")
    # ---- gold fixes (gen_traps.py --gold-fixes): re-derived constructed golds, applied right after loading, before any mixing
    fix_relabel, fix_drop = {}, set()
    gf = collections.Counter()
    if getattr(args, "gold_fixes", None):
        for p in args.gold_fixes:
            rl, dr, _ = gen_traps.load_gold_fixes(p)
            fix_relabel.update(rl)
            fix_drop |= dr
        gf["relabel_ids_in_file"], gf["drop_ids_in_file"] = len(fix_relabel), len(fix_drop)
        for r in pool.rows.values():
            gf["pool_rows_relabelled"] += gen_traps.apply_gold_fix(r, fix_relabel)
        for rows in pool.heldout.values():
            for r in rows:
                gf["heldout_rows_relabelled"] += gen_traps.apply_gold_fix(r, fix_relabel)

    # ---- held-out side (everything never trained on)
    review = []
    if args.review_dev:
        for r in ac.read_jsonl(args.review_dev):
            base = r if "state" in r else {**{k: v for k, v in pool.rows[r["id"]].items() if k != "pool"}, **r}
            review.append(base)
    held_all = [h for v in pool.heldout.values() for h in v] + review
    gui_ids = {h["id"] for h in pool.heldout["heldout-fresh-gui"] + pool.heldout["heldout-flagged-gui"]
               + [x for g in FIXED_STATE_GROUPS for x in pool.heldout[f"heldout-fresh-{g}"]]}
    held_ids = {h["id"] for h in held_all}
    held_groups = {pool.group(h["id"]) for h in held_all if pool.group(h["id"])}
    held_record_ids = {(h.get("provenance") or {}).get("record_id") for h in held_all} - {None}
    held_shas = {s for h in held_all for s in ac.image_shas(h)}
    held_imgq = {am.image_question_key(h["images"], h["field"]["question"]) for h in held_all if h.get("images")}

    # ---- verified rows
    review_drop = set()
    for p in args.drop_ids or []:
        if Path(p).exists():
            review_drop |= set(json.loads(Path(p).read_text())["ids"])
    verified = []
    teacher_ids: set = set()
    held_merged, dir_stats = None, {}
    if args.synthetic:
        verified = synthetic_verified(pool, args.synthetic)
        log(f"synthetic teacher output: {len(verified):,} rows")
    if args.teacher_dirs:
        dir_rows, held_merged, dir_stats = load_teacher_dirs(args.teacher_dirs)
        for r in dir_rows:
            backend = r.get("teacher_backend")
            r = am.unwrap_verified(r)
            if r is None:
                drop["teacher_dropped_row_in_verified"] += 1
                continue
            r["_teacher_backend"] = backend
            teacher_ids.add(r["id"])
            verified.append(r)
        log(f"teacher dirs {args.teacher_dirs}: {len(dir_rows):,} kept rows; {dir_stats}")
    for p in args.verified or []:
        if not Path(p).exists():
            log(f"verified file {p} missing: skipped (e.g. no pod-kept A/D/I parents -> no pod constructed variants)")
            continue
        for r in ac.read_jsonl(p):
            backend = r.get("teacher_backend")
            r = am.unwrap_verified(r)
            if r is None:
                drop["teacher_dropped_row_in_verified"] += 1
                continue
            if r["id"] in teacher_ids:
                drop["verified_file_row_shadowed_by_teacher_dir"] += 1
                continue
            if backend:
                r["_teacher_backend"] = backend
            verified.append(r)
    n_direct = collections.Counter()
    for p, kind in [(x, "image") for x in args.direct_constructed or []] + [(x, "leftover") for x in args.direct_leftovers or []]:
        if not Path(p).exists():
            log(f"direct file {p} missing: skipped")
            continue
        for r in ac.read_jsonl(p):
            if r.get("gold_kind") not in ("constructed", "dataset"):
                drop["direct_not_constructed_or_dataset_gold"] += 1
                continue
            if r["id"] in teacher_ids:
                drop["direct_shadowed_by_teacher_dir"] += 1
                continue
            r = {k: v for k, v in r.items() if k != "pool"}
            r.setdefault("label", {"target": r.get("gold"), "probs": None, "rationale": None,
                                   "target_kind": "constructed" if r["gold_kind"] == "constructed" else "gold", "review": None})
            if pool.group(r["id"]) is None:
                prov = r.get("provenance") or {}
                r["_group"] = "direct:" + str(prov.get("group_id") or prov.get("record_id") or (r.get("images") or [r["id"]])[0])
            r["_direct"] = kind
            verified.append(r)
            n_direct[kind] += 1
    if n_direct:
        log(f"direct constructed rows: {dict(n_direct)} from {args.direct_constructed + args.direct_leftovers}")
    if fix_relabel or fix_drop:
        kept_v = []
        for r in verified:
            if r["id"] in fix_drop:
                gf["verified_dropped"] += 1
                continue
            if r["id"] in fix_relabel:
                lab = r.get("label") if isinstance(r.get("label"), dict) else None
                if lab is not None and lab.get("target_kind") not in ("constructed", "gold") and \
                        gen_traps.label_contradicts(lab.get("target"), fix_relabel[r["id"]][0]):
                    gf["verified_dropped_teacher_label_contradicts_new_gold"] += 1   # verdicts newer than the fixes file
                    continue
                gen_traps.apply_gold_fix(r, fix_relabel)
                gf["verified_relabelled"] += 1
            kept_v.append(r)
        verified = kept_v
        log(f"gold fixes {args.gold_fixes}: {dict(gf)}")
    stats["gold_fixes"] = gf
    by_id = {}
    for r in verified:
        if "state" not in r:
            base = pool.rows.get(r["id"])
            if base is None:
                drop["verified_id_not_in_pool"] += 1
                continue
            r.update({k: v for k, v in base.items() if k not in r and k != "pool"})
        by_id[r["id"]] = r
    hard_text, hard_image = [], []
    a_gold = []
    for r in by_id.values():
        r = {k: v for k, v in r.items() if k != "pool"}
        errs = validate({k: v for k, v in r.items() if k != "label"})
        if errs:
            drop["invalid_candidate"] += 1
            continue
        g = root_group(pool, r, by_id)
        if r["id"] in held_ids or g in held_groups:
            drop["heldout_or_heldout_group"] += 1
            continue
        if g in pool.bucket_groups:
            drop["heldout_flagged_bucket_group"] += 1
            continue
        if r["id"] in pool.flagged_exclude:
            drop["linked_to_heldout_flagged"] += 1
            continue
        if r["id"] in review_drop:
            drop["review_says_wrong_or_disputed"] += 1
            continue
        try:
            lab = am.parse_label(r)
            rec = am.candidate_to_record(r, lab, group=g, partition="train", mix="hard_image" if r.get("images") else "hard_text")
        except (ValueError, KeyError, TypeError) as e:
            drop[f"label_or_convert: {str(e)[:60]}"] += 1
            continue
        err = am.check_record(rec)
        if err:
            drop[f"render: {err[:60]}"] += 1
            continue
        if r.get("_teacher_backend"):
            rec["p3"]["teacher_backend"] = r["_teacher_backend"]
        if r.get("_direct"):
            rec["p3"]["direct_source"] = "direct-leftover" if r["_direct"] == "leftover" else "direct-image"
        rec["_cand"] = r
        rec["_review_ok"] = bool(lab.get("review") and lab["review"].get("verdict") == "correct")
        if r["source"] == "A" and lab["target_kind"] == "gold":
            a_gold.append(rec)
        (hard_image if r.get("images") else hard_text).append(rec)
    # A kept on a hard constructed gold: at most 25% of the A rows
    n_a = sum(1 for x in hard_text + hard_image if x["_cand"]["source"] == "A")
    cap = int(A_GOLD_CAP * n_a)
    if len(a_gold) > cap:
        over = {x["id"] for x in sorted(a_gold, key=lambda x: ac.stable_hash(SEL_SALT, "agold", x["id"]))[cap:]}
        drop["a_hard_gold_over_25pct"] += len(over)
        hard_text = [x for x in hard_text if x["id"] not in over]
        hard_image = [x for x in hard_image if x["id"] not in over]
    # large_choice: constructed gold, 256-code lanes only
    lc = []
    for r in pool.large_choice:
        c = {k: v for k, v in r.items() if k != "pool"}
        g = r["pool"]["group"]
        if c["id"] in held_ids or g in held_groups:
            continue
        rec = am.candidate_to_record(c, None, group=g, partition="train", mix="hard_text")
        rec["_cand"], rec["_review_ok"] = c, False
        err = am.check_record(rec, soft=False)
        if err:
            drop[f"large_choice render: {err[:60]}"] += 1
            continue
        lc.append(rec)
    log(f"verified: {len(hard_text):,} hard text, {len(hard_image):,} hard image, {len(lc):,} large_choice; dropped {dict(drop)}")

    # ---- dev split of verified hard rows (by group)
    for rec in hard_text + hard_image:
        if ac.stable_unit(DEV_SALT, rec["source_group"]) < args.dev_share:
            rec["partition"] = "dev"

    # ---- replay candidates
    hard_record_ids = {(x["_cand"].get("provenance") or {}).get("record_id") for x in hard_text + hard_image} - {None}
    n_hard_text_train = sum(x["partition"] == "train" for x in hard_text if x["p3"]["n_options"] <= 254)
    n_hard_img_train = sum(x["partition"] == "train" for x in hard_image)
    cap_total = min(args.total, n_hard_text_train / args.share_hard_text if args.share_hard_text else args.total,
                    n_hard_img_train / args.share_hard_image if args.share_hard_image else args.total)
    need_rt = int(round(cap_total * args.share_replay_text))
    need_ri = int(round(cap_total * args.share_replay_image))
    seen_ids: set = set()

    def dedupe(rows, kind):
        out = []
        for x in rows:
            if x["id"] in seen_ids:
                drop[f"replay_{kind}_duplicate_id"] += 1
                continue
            seen_ids.add(x["id"])
            out.append(x)
        return out

    def replay_ok(rec, kind) -> bool:
        if rec["_record_id"] in review_drop or rec.get("id") in review_drop:     # --drop-ids apply to replay rows too
            drop[f"replay_{kind}_in_drop_ids"] += 1
            return False
        if rec["_record_id"] in held_record_ids:
            drop[f"replay_{kind}_is_heldout_record"] += 1
            return False
        if rec["_record_id"] in hard_record_ids:
            drop[f"replay_{kind}_duplicates_hard_row"] += 1
            return False
        if kind == "image" and any(i.get("sha256") in held_shas for i in rec.get("images") or []):
            drop["replay_image_shares_heldout_photo"] += 1
            return False
        return True
    # p2c components first; the v2.1 top-up is read only when the filtered supply is short (1.2x headroom for the later
    # leakage filter)
    rtext = [r for r in dedupe(load_replay_text([Path(p) for p in args.replay_p2],
                                                Path(args.replay_eikos) if args.replay_eikos else None, None, False), "text")
             if replay_ok(r, "text")]
    top_text = len(rtext) < need_rt * 1.2 and bool(args.replay_v21)
    if top_text:
        rtext += [r for r in dedupe(load_replay_text([], None, Path(args.replay_v21), True), "text") if replay_ok(r, "text")]
    rimg = [r for r in dedupe(load_replay_images([Path(p) for p in args.replay_images], None, False), "image")
            if replay_ok(r, "image")]
    top_img = len(rimg) < need_ri * 1.2 and bool(args.replay_v21)
    if top_img:
        rimg += [r for r in dedupe(load_replay_images([], Path(args.replay_v21), True), "image") if replay_ok(r, "image")]
    log(f"replay candidates after exclusions: {len(rtext):,} text (v2.1 top-up {top_text}), {len(rimg):,} image "
        f"(v2.1 top-up {top_img})")
    for r in rtext:
        r["mix"] = "replay_text"
    for r in rimg:
        r["mix"] = "replay_image"

    # ---- leakage pre-filter: every candidate train row vs every held-out row
    cand_rows = hard_text + hard_image + lc + rtext + rimg
    lines = [json.dumps(x["_cand"] if "_cand" in x else am.record_as_candidate(x), ensure_ascii=False) for x in cand_rows]
    links = ac.scan_links(lines, held_all) if held_all else []
    bad = {i for i, reason, hid in links if not (reason == "state" and hid in gui_ids)}
    for i, x in enumerate(cand_rows):
        if x.get("images") and am.image_question_key([im["image"] for im in x["images"]], x["request"]["fields"][0]["question"]) in held_imgq:
            bad.add(i)
    for i in bad:
        drop[f"leakage_link_{cand_rows[i]['mix']}"] += 1
    keep = [x for i, x in enumerate(cand_rows) if i not in bad]
    hard_text = [x for x in keep if x["mix"] == "hard_text" and x["_cand"]["dataset"] != "large_choice"]
    hard_image = [x for x in keep if x["mix"] == "hard_image"]
    lc = [x for x in keep if x.get("_cand", {}).get("dataset") == "large_choice"]
    rtext = [x for x in keep if x["mix"] == "replay_text"]
    rimg = [x for x in keep if x["mix"] == "replay_image"]
    log(f"after leakage pre-filter: {len(hard_text):,} hard text, {len(hard_image):,} hard image, {len(lc):,} large_choice, "
        f"{len(rtext):,} replay text, {len(rimg):,} replay image ({len(bad):,} dropped)")

    # ---- mixture sizes (train partition)
    def direct_image(x):
        return args.direct_must and x.get("_cand", {}).get("_direct") == "image"
    ht255 = [x for x in hard_text if x["partition"] == "train" and x["p3"]["n_options"] <= 254]
    ht256 = [x for x in hard_text if x["partition"] == "train"]
    hi_all = [x for x in hard_image if x["partition"] == "train" and x["p3"]["n_options"] <= 254]
    hi256_all = [x for x in hard_image if x["partition"] == "train"]
    hi_dir, hi256_dir = [x for x in hi_all if direct_image(x)], [x for x in hi256_all if direct_image(x)]
    hi, hi256 = [x for x in hi_all if not direct_image(x)], [x for x in hi256_all if not direct_image(x)]
    supply = {"hard_text": len(ht255), "hard_image": len(hi), "replay_text": len(rtext), "replay_image": len(rimg)}
    shares = {"hard_text": args.share_hard_text, "replay_text": args.share_replay_text, "hard_image": args.share_hard_image,
              "replay_image": args.share_replay_image}
    total = int(min([args.total] + [supply[k] / s for k, s in shares.items() if s > 0]))
    n = {k: int(round(total * s)) for k, s in shares.items()}
    limiting = min((supply[k] / s, k) for k, s in shares.items() if s > 0)[1]
    log(f"mixture total {total:,} (limited by {limiting}): {n}")
    # direct image-group rows (--direct-must): all trained, on top of the hard-image share; the image total may rise to
    # --max-image-share, image replay is reduced first, then the teacher-verified hard images; direct rows are never cut
    n_dir = len(hi_dir)
    image_adjust = {"direct_image_rows": n_dir, "max_image_share": args.max_image_share, "replay_image_cut": 0, "hard_image_cut": 0}
    if n_dir:
        text_n = n["hard_text"] + n["replay_text"]
        allowed = int(args.max_image_share / (1 - args.max_image_share) * text_n)
        over = n_dir + n["hard_image"] + n["replay_image"] - allowed
        if over > 0:
            cut = min(over, n["replay_image"]); n["replay_image"] -= cut; image_adjust["replay_image_cut"] = cut; over -= cut
        if over > 0:
            cut = min(over, n["hard_image"]); n["hard_image"] -= cut; image_adjust["hard_image_cut"] = cut; over -= cut
        if over > 0:
            log(f"WARNING: {n_dir:,} direct image rows alone exceed a {args.max_image_share:.0%} image share; they are all kept")
        image_adjust["image_share_final"] = round((n_dir + n["hard_image"] + n["replay_image"]) /
                                                  (text_n + n_dir + n["hard_image"] + n["replay_image"]), 4)
        log(f"direct image rows {n_dir:,} on top of the hard-image share: {image_adjust}")
    targets_final = {**n, "hard_image": n["hard_image"] + n_dir}
    tf_total = sum(targets_final.values()) or 1
    shares_final = {k: round(v / tf_total, 4) for k, v in targets_final.items()}
    log(f"final mixture targets {targets_final} -> shares {shares_final}")

    def must(x):
        return x["target"] is None or x.get("_review_ok") or x.get("_cand", {}).get("dataset") == "large_choice"

    def strat_hard(x):
        c = x["_cand"]
        return (c["family"], c["difficulty"], c["source"])

    def strat_replay(x):
        return (x.get("replay_source"), x.get("family"))

    sel_rt = am.tiered_take([[x for x in rtext if x["replay_source"].startswith("p2c")],
                             [x for x in rtext if not x["replay_source"].startswith("p2c")]], n["replay_text"], strat_replay,
                            SEL_SALT + ":rt")
    sel_ri = am.tiered_take([[x for x in rimg if x["replay_source"].startswith("p2c")],
                             [x for x in rimg if not x["replay_source"].startswith("p2c")]], n["replay_image"], strat_replay,
                            SEL_SALT + ":ri")
    sel_hi_255 = hi_dir + am.stratified_take(hi, n["hard_image"], strat_hard, must, SEL_SALT + ":hi")
    sel_hi_256 = hi256_dir + am.stratified_take(hi256, n["hard_image"], strat_hard, must, SEL_SALT + ":hi")
    sel_ht_255 = am.stratified_take(ht255, n["hard_text"], strat_hard, must, SEL_SALT + ":ht")
    sel_ht_256 = am.stratified_take(lc + ht256, n["hard_text"], strat_hard, must, SEL_SALT + ":ht")
    dev_rows_255 = [x for x in hard_text + hard_image if x["partition"] == "dev" and x["p3"]["n_options"] <= 254]
    dev_rows_256 = [x for x in hard_text + hard_image if x["partition"] == "dev"]
    lanes = {"256": sel_ht_256 + sel_rt + sel_hi_256 + sel_ri + dev_rows_256,
             "255": sel_ht_255 + sel_rt + sel_hi_255 + sel_ri + dev_rows_255}

    # ---- write lanes
    report = {"version": "p3-manifest-v1", "built": time.strftime("%Y-%m-%d %H:%M"), "command": " ".join(sys.argv[1:]),
              "synthetic": bool(args.synthetic), "supply": supply, "total_train": total, "limited_by": limiting,
              "targets": targets_final, "shares_wanted": shares, "shares_final": shares_final, "image_adjust": image_adjust, "lanes": {}, "dropped": None, "dev_manifests": {}}
    leak_dir = out_dir / ".leakage"
    leak_dir.mkdir(exist_ok=True)
    lane_paths, lane_rows = {}, {}
    for lane, rows in lanes.items():
        rows = sorted(rows, key=lambda x: ac.stable_hash(SEL_SALT, "order", x["id"]))
        ids = collections.Counter(x["id"] for x in rows)
        if max(ids.values()) > 1:
            raise SystemExit(f"lane {lane}: duplicate ids {[i for i, c in ids.items() if c > 1][:5]}")
        if lane == "255" and any(x.get("p3", {}).get("n_options", 0) > 254 or
                                 len(x["request"]["fields"][0].get("options") or []) > 254 for x in rows):
            raise SystemExit("255 lane holds a 255-option row")
        path = out_dir / f"p3-train-{lane}.jsonl"
        clean = [{k: v for k, v in x.items() if not k.startswith("_")} for x in rows]
        ac.write_jsonl(path, clean)
        lane_rows[lane] = clean
        lane_paths[lane] = path
        tr = [x for x in rows if x["partition"] == "train"]
        mix = collections.Counter(x["mix"] for x in tr)
        got = {k: round(mix[k] / max(1, len(tr)), 4) for k in shares}
        report["lanes"][lane] = {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": ac.sha256_file(path),
            "rows": len(rows), "train": len(tr), "dev": len(rows) - len(tr), "shares": got,
            "shares_target": shares_final,
            "shares_ok": all(abs(got[k] - shares_final[k]) <= 0.02 for k in shares),
            "unknown_share_hard_text": round(sum(x["target"] is None for x in tr if x["mix"] == "hard_text") / max(1, mix["hard_text"]), 4),
            "with_target_probs": sum(bool(x.get("target_probs")) for x in tr),
            "with_rationale": sum(bool(x.get("rationale")) for x in tr),
            "options_255": sum(len(x["request"]["fields"][0].get("options") or []) == 255 for x in rows),
            "hard_by_source": dict(sorted(collections.Counter(x["p3"]["source"] for x in tr if x["mix"].startswith("hard")).items())),
            "hard_by_family": dict(collections.Counter(x["family"] for x in tr if x["mix"].startswith("hard")).most_common()),
            "hard_by_difficulty": dict(sorted(collections.Counter(str(x["p3"]["difficulty"]) for x in tr if x["mix"].startswith("hard")).items())),
            "replay_by_source": dict(collections.Counter(x.get("replay_source") for x in tr if x["mix"].startswith("replay")).most_common()),
            "hard_direct_constructed": sum(1 for x in tr if x["mix"].startswith("hard") and x.get("_cand", {}).get("_direct")),
            "hard_direct_by_source": dict(collections.Counter(x["p3"].get("direct_source") for x in tr if x["mix"].startswith("hard")
                                                              and x["p3"].get("direct_source"))),
            "hard_direct_by_family": dict(collections.Counter(x["family"] for x in tr if x["mix"].startswith("hard")
                                                              and x.get("_cand", {}).get("_direct")).most_common()),
            "hard_by_teacher_backend": dict(collections.Counter(x["p3"].get("teacher_backend") or "none (constructed / gold)"
                                                                for x in tr if x["mix"].startswith("hard")).most_common()),
            "image_dirs": sorted({str(Path(i["image"]).parent) for x in rows for i in x.get("images") or []}),
        }
        ac.write_jsonl(leak_dir / f"train-{lane}.jsonl", (x["_cand"] if "_cand" in x else am.record_as_candidate(x) for x in rows))
        log(f"lane {lane}: {len(rows):,} rows ({len(tr):,} train), shares {got}")

    # ---- dev / eval manifests
    dev_specs = [("heldout-fresh", pool.heldout["heldout-fresh"]), ("heldout-fresh-gui", pool.heldout["heldout-fresh-gui"]),
                 ("heldout-fresh-large-choice", pool.heldout["heldout-fresh-large-choice"]),
                 ("heldout-flagged", pool.heldout["heldout-flagged"]), ("heldout-flagged-gui", pool.heldout["heldout-flagged-gui"]),
                 *((f"heldout-{g}", pool.heldout[f"heldout-fresh-{g}"]) for g in DIRECT_GROUPS),
                 ("review", review)]
    held_res = None
    if held_merged is not None:
        held_res = held_merged
        log(f"held-out teacher verdicts from {args.teacher_dirs}: {len(held_res):,} "
            f"({sum(1 for r in held_res.values() if r.get('keep')):,} kept)")
    elif args.heldout_results and Path(args.heldout_results).exists():
        held_res = {r["id"]: r for r in ac.read_jsonl(args.heldout_results)}
        log(f"held-out teacher verdicts: {len(held_res):,} ({sum(1 for r in held_res.values() if r.get('keep')):,} kept)")
    held_files, gui_files = [], []
    published: dict[str, list] = {}
    for name, rows in dev_specs:
        if not rows:
            continue
        recs = []
        for c in rows:
            if c["id"] in fix_drop:
                gf["heldout_eval_rows_dropped"] += 1
                continue
            if name.startswith("heldout-flagged") and held_res is not None:
                hr = held_res.get(c["id"])
                if hr is not None and not hr.get("keep"):
                    drop["heldout_flagged_teacher_rejected"] += 1
                    continue
                if hr is not None:
                    if c["id"] in fix_relabel and (hr.get("label") or {}).get("target_kind") not in ("constructed", "gold") and \
                            gen_traps.label_contradicts((hr.get("label") or {}).get("target"), fix_relabel[c["id"]][0]):
                        gf["heldout_eval_rows_dropped_teacher_label_contradicts_new_gold"] += 1
                        continue
                    c = {**c, "label": hr.get("label")}
                elif c.get("gold_kind") not in ("constructed", "dataset"):
                    drop["heldout_flagged_unlabelled"] += 1
                    continue
            lab = am.parse_label(c) if (c.get("label") is not None or "probs" in c) else None
            cc = {k: v for k, v in c.items() if k not in ("label", "pool")}
            rec = am.candidate_to_record(cc, lab, group=pool.group(c["id"]) or "heldout:" + c["id"], partition="dev", mix=name)
            err = am.check_record(rec)
            if err:
                drop[f"dev {name} render: {err[:50]}"] += 1
                continue
            recs.append(rec)
        path = out_dir / f"p3-dev-{name}.jsonl"
        ac.write_jsonl(path, recs)
        published[name] = recs
        cpath = leak_dir / f"heldout-{name}.jsonl"
        ac.write_jsonl(cpath, ({k: v for k, v in c.items() if k not in ("label", "pool")} for c in rows))
        fixed_state = name.endswith("-gui") or name in {f"heldout-{g}" for g in FIXED_STATE_GROUPS}
        (gui_files if fixed_state else held_files).append(cpath)
        report["dev_manifests"][name] = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                                         "rows": len(recs), "sha256": ac.sha256_file(path),
                                         "options_255": sum(len(r["request"]["fields"][0].get("options") or []) == 255 for r in recs)}

    # ---- final gate
    gate = {}
    for lane, path in lane_paths.items():
        train_c = leak_dir / f"train-{lane}.jsonl"
        rc, msg = run_gate([train_c], held_files, out_dir / f"leakage-{lane}.md") if held_files else (0, "no held-out files")
        gate[lane] = {"exit": rc, "result": msg}
        log(f"gate lane {lane}: {msg}")
        if gui_files:
            rc_g, msg_g = run_gate([train_c], gui_files, out_dir / f"leakage-{lane}-gui.md")
            only_state = rc_g == 0 or set(re.findall(r"(\w+)=\d+", msg_g.split("FAIL", 1)[-1])) <= {"state"}
            gate[lane + "-gui"] = {"exit": rc_g, "result": msg_g, "only_fixed_state_links": only_state}
            log(f"gate lane {lane} GUI: {msg_g} (only fixed-state: {only_state})")
    report["gate"] = gate
    report["teacher_dirs"] = {"dirs": list(args.teacher_dirs or []), "stats": dir_stats}
    if args.publish:
        report["published"] = publish(args.publish, Path(args.publish_dir), lane_rows, hard_text + hard_image, published)
    report["dropped"] = dict(drop)
    report["gold_fixes"] = {"files": list(getattr(args, "gold_fixes", None) or []), **dict(gf)}
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "manifest-report.json").write_text(json.dumps(report, indent=1) + "\n")
    import assembly_report
    assembly_report.write_manifest_report(report, out_dir / "manifest-report.md")
    if not args.keep_leakage_files:
        for p in leak_dir.glob("*.jsonl"):
            p.unlink()
    ok = all(v["exit"] == 0 for k, v in gate.items() if not k.endswith("-gui")) and \
        all(v["only_fixed_state_links"] for k, v in gate.items() if k.endswith("-gui")) and \
        all(v["shares_ok"] for v in report["lanes"].values())
    log("OK" if ok else "FAILED (gate or shares)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verified", nargs="*", default=[], help="teacher-verified rows (label contract in assembly_manifest.py)")
    ap.add_argument("--direct-constructed", nargs="*", default=[],
                    help="candidate files with constructed (or human-verified dataset) gold that go straight to training, no teacher: "
                         "data/p3/candidates-clean/I-{charts,docimg,inventory,safety,geometry,screens}.jsonl")
    ap.add_argument("--direct-must", dest="direct_must", action="store_true", default=True,
                    help="(default ON, owner 2026-09-26) every direct image-group row is trained, on top of the hard-image share")
    ap.add_argument("--no-direct-must", dest="direct_must", action="store_false",
                    help="direct image-group rows compete inside the hard-image share (stratified by family, unknowns kept)")
    ap.add_argument("--max-image-share", type=float, default=0.40,
                    help="with --direct-must: the image total may rise to this share; image replay is cut first, then hard images")
    ap.add_argument("--direct-leftovers", nargs="*", default=[],
                    help="make_extra_queue.py direct-constructed-leftovers*.jsonl: constructed-gold waiting rows, no teacher")
    ap.add_argument("--gold-fixes", nargs="*", default=[],
                    help="gen_traps.py --gold-fixes JSONL(s) (data/p3/fixes/trap-gold-fixes.jsonl): relabel re-derived constructed golds "
                         "in the pool, held-out and verified rows, drop listed ids; applied after loading, before mixing")
    ap.add_argument("--teacher-dirs", nargs="*", default=[],
                    help="teacher output dirs in order of preference (e.g. data/p3/teacher data/p3/teacher-pod): kept.jsonl + "
                         "heldout-results.jsonl of each, first verdict per id wins; replaces --heldout-results")
    ap.add_argument("--synthetic", type=int, default=0, help="add N synthetic verified rows drawn from the pool (dry run/tests)")
    ap.add_argument("--review-dev", default=None, help="human/Kimi-reviewed dev slice (candidate rows or ids, + label)")
    ap.add_argument("--drop-ids", nargs="*", default=[], help="review_report.py drop-ids.json: ids that leave training")
    ap.add_argument("--pool-dir", default=str(ac.POOL_DIR))
    ap.add_argument("--out-dir", default=str(ac.MANIFEST_DIR))
    ap.add_argument("--total", type=int, default=130_000)
    ap.add_argument("--share-hard-text", type=float, default=SHARES["hard_text"])
    ap.add_argument("--share-replay-text", type=float, default=SHARES["replay_text"])
    ap.add_argument("--share-hard-image", type=float, default=SHARES["hard_image"])
    ap.add_argument("--share-replay-image", type=float, default=SHARES["replay_image"])
    ap.add_argument("--dev-share", type=float, default=0.03)
    ap.add_argument("--replay-p2", nargs="*", default=[str(ROOT / "data/manifests/decision-p2b.jsonl"),
                                                        str(ROOT / "data/manifests/decision-p2.jsonl")])
    ap.add_argument("--replay-eikos", default=str(ROOT / "data/external/eikos-decisions/records.jsonl"))
    ap.add_argument("--replay-v21", default=str(ROOT / "data/manifests/decision-v2.1-4b.jsonl"))
    ap.add_argument("--replay-images", nargs="*", default=[str(ROOT / "data/decision-p2c/image-replay/replay-fresh.jsonl")])
    ap.add_argument("--keep-leakage-files", action="store_true")
    ap.add_argument("--heldout-results", default=str(ROOT / "data/p3/teacher/heldout-results.jsonl"),
                    help="teacher verdicts on the flagged held-out half (azure_teacher.py finalize); '' = stored gold only")
    ap.add_argument("--publish", default=None, help="also write the pod-facing manifests as data/manifests/<NAME>*.jsonl "
                    "(the run uses decision-p3)")
    ap.add_argument("--publish-dir", default=str(ROOT / "data/manifests"))
    args = ap.parse_args(argv)
    if not args.verified and not args.synthetic and not args.teacher_dirs and not args.direct_constructed and not args.direct_leftovers:
        ap.error("give --teacher-dirs / --verified files (or --synthetic N for a dry run)")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
