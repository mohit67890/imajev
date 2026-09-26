"""Held-out sets for phase 3 (docs/phase-3-plan.md, Stage 1 items 4-5; used by scripts/p3/assemble_pool.py).

heldout-fresh: items made with FRESH SEEDS from every generator (never in the pool, never mined), family-balanced:
  gen_reasoning.py  --seed <tag>   --count N          7 reasoning families, subsampled to <= PER_FAMILY each (by group)
  gen_policy.py     --seed <int>   --count PER_FAMILY  6 policy families (agent_action also yields routing_hard)
  gen_traps.py      --in <the fresh policy file> --seed <int> --count TRAPS
  gen_image_joint   no seed flag: imported with SEED replaced, ids re-prefixed; OVERSAMPLED, then the items whose photos link
                    to the fewest pool rows are kept (a photo item is an upstream key, so a fresh item on a busy photo would
                    force many pool rows out)
  gen_gui           no seed flag: imported with SEED and IMG_DIR replaced (PNGs under data/p3/pool/heldout-images/gui/), ids
                    and screen ids re-prefixed. Written to its OWN file: every GUI row's state is one of five fixed strings
                    ("a screenshot of a form screen in a web app"), so the exact-whole-state rule of `decontam.py --leakage`
                    links every GUI item to every pool GUI item of the same screen kind, although the screenshots differ.
  assembly_large_choice  fresh seed, own file (255 options: 256-code lane only)
plus a ~2% slice of the B/C (public + own pools) GROUPS, drawn by whole group: candidates are drawn at a higher rate by a
stable hash, and per dataset the candidates that link to the fewest pool rows are kept (ALFWorld rooms and observation strings,
MuSiQue's shared Wikipedia passages and the fixed image-question states make many groups inseparable under the 13-gram/state
rules; those are never held out). Every fresh generator is oversampled and selected the same way (select_by_cost).

Link resolution: every held-out row is checked against the pool with the leakage rules (assembly_common.scan_links). A pool row
that links to a held-out item leaves the pool with its whole group; a held-out group that would force more than
MAX_POOL_LINKS pool rows out is dropped from the held-out set instead (fresh items are discarded, B/C groups go back to the
pool... see resolve_links: a dropped B/C group is discarded). This repeats until a scan finds no link; `decontam.py --leakage`
is then run as the gate.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly_common as ac  # noqa: E402
import assembly_large_choice as alc  # noqa: E402

PY = sys.executable
GEN = ac.ROOT / "scripts/p3"
REASONING_FAMILIES = ("table_arithmetic", "temporal_numeric", "logic", "multi_hop", "probability", "constraint", "ordinal")
POLICY_FAMILIES = ("long_policy", "rule_exception", "judge_hard", "agent_action", "long_input", "multi_label")
MAX_POOL_LINKS = 25


def _run(cmd: list[str]):
    # PYTHONHASHSEED pinned: gen_reasoning's multi_hop states depend on set iteration order (seen 2026-09-25: 110 of the
    # fresh rows differed between two runs with the same --seed), so the held-out file is only reproducible with a fixed hash seed
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    res = subprocess.run(cmd, cwd=ac.ROOT, capture_output=True, text=True, env=env)
    if res.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stderr[-3000:]}")
    return res


def _mark(rows: list[dict], tag: str, generator: str) -> list[dict]:
    for r in rows:
        r["provenance"]["heldout"] = "fresh"
        r["provenance"]["heldout_tag"] = tag
        r["provenance"]["heldout_generator"] = generator
    return rows


def subsample_by_group(rows: list[dict], per_family: int, salt: str) -> list[dict]:
    """<= per_family rows per family, taking whole parent/child groups in stable hash order."""
    groups = ac.assign_groups(rows)
    by_fam: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_fam[r["family"]][groups[r["id"]]].append(r)
    out = []
    for fam, gs in sorted(by_fam.items()):
        n = 0
        for g in sorted(gs, key=lambda g: ac.stable_hash(salt, g)):
            if n + len(gs[g]) > per_family and n:
                continue
            out += gs[g]
            n += len(gs[g])
            if n >= per_family:
                break
    return out


def fresh_reasoning(tag: str, per_family: int, work: Path) -> list[dict]:
    out = work / "fresh-reasoning.jsonl"
    _run([PY, str(GEN / "gen_reasoning.py"), "--seed", tag, "--count", str(int(per_family * 7 * 1.3)), "--out", str(out)])
    rows = list(ac.read_jsonl(out))
    return _mark(subsample_by_group(rows, per_family, tag + ":reasoning"), tag, "gen_reasoning.py")


def fresh_policy(seed: int, tag: str, per_family: int, work: Path) -> tuple[list[dict], Path]:
    out = work / "fresh-policy.jsonl"
    _run([PY, str(GEN / "gen_policy.py"), "--seed", str(seed), "--count", str(per_family), "--out", str(out), "--workers", "6"])
    return _mark(list(ac.read_jsonl(out)), tag, "gen_policy.py"), out


def fresh_traps(seed: int, tag: str, count: int, policy_file: Path, work: Path) -> list[dict]:
    out = work / "fresh-traps.jsonl"
    _run([PY, str(GEN / "gen_traps.py"), "--in", str(policy_file), "--out", str(out), "--count", str(count), "--seed", str(seed)])
    return _mark(list(ac.read_jsonl(out)), tag, "gen_traps.py")


def _reid(rows: list[dict], old_prefix: str, new_prefix: str) -> list[dict]:
    for r in rows:
        r["id"] = r["id"].replace(old_prefix, new_prefix, 1)
        if r.get("parent_id"):
            r["parent_id"] = r["parent_id"].replace(old_prefix, new_prefix, 1)
    return rows


def fresh_image_joint(tag: str, n: int) -> list[dict]:
    import gen_image_joint as gij
    old = gij.SEED
    gij.SEED = f"p3-image-joint-heldout-{tag}"
    try:
        rows = gij.build(n)
    finally:
        gij.SEED = old
    return _mark(_reid(rows, "p3-I-joint-", f"p3-{tag}-I-joint-"), tag, "gen_image_joint.py (SEED override)")


def fresh_gui(tag: str, screens: int, img_dir: Path) -> list[dict]:
    import gen_gui as gg
    old_seed, old_dir = gg.SEED, gg.IMG_DIR
    gg.SEED, gg.IMG_DIR = f"p3-gui-heldout-{tag}", img_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="p3gui-heldout-") as work:
            rows, _ = gg.build(screens, Path(work))
    finally:
        gg.SEED, gg.IMG_DIR = old_seed, old_dir
    rel = img_dir.relative_to(ac.ROOT / "data")
    for i, r in enumerate(rows):
        r["id"] = f"p3-{tag}-I-gui-{i:06d}"
        r["images"] = [str(rel / Path(p).name) for p in r["images"]]
        r["provenance"]["screen_id"] = f"{tag}-{r['provenance']['screen_id']}"
    return _mark(rows, tag, "gen_gui.py (SEED/IMG_DIR override)")


def fresh_large_choice(tag: str, count: int) -> list[dict]:
    return _mark(alc.generate(f"heldout-{tag}", count, id_prefix=f"p3-{tag}-lc"), tag, "assembly_large_choice.py")


def bc_candidate_groups(rows: list[dict], groups: dict[str, str], rate: float, salt: str) -> set[str]:
    """Whole groups whose rows are all source B or C, each drawn with probability `rate` (stable hash)."""
    srcs: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        srcs[groups[r["id"]]].add(r["source"])
    return {g for g, s in srcs.items() if s <= {"B", "C"} and ac.stable_unit(salt, g) < rate}


def select_bc_groups(rows: list[dict], groups: dict[str, str], candidates: set[str], costs: dict[str, int], share: float,
                     scarce_cap: float, salt: str, max_links: int = MAX_POOL_LINKS,
                     scarce=("plumb", "finqa", "tatqa")) -> set[str]:
    """Per dataset, up to round(share * its groups) candidate groups in order (link cost, stable hash), never above max_links;
    scarce datasets never above scarce_cap of their groups. A group belongs to the dataset of its first row."""
    ds_groups: dict[str, set] = collections.defaultdict(set)
    first: dict[str, str] = {}
    for r in rows:
        g = groups[r["id"]]
        first.setdefault(g, r["dataset"])
    for g, d in first.items():
        ds_groups[d].add(g)
    chosen = set()
    for d, gs in sorted(ds_groups.items()):
        want = int(round(min(share, scarce_cap) * len(gs))) if d in scarce else int(round(share * len(gs)))
        cands = sorted((g for g in gs if g in candidates and costs.get(g, 0) <= max_links),
                       key=lambda g: (costs.get(g, 0), ac.stable_hash(salt, g)))
        chosen |= set(cands[:want])
    return chosen


def _links(pool_lines, alive_idx, held, ignore_state: set[str]):
    links = ac.scan_links([pool_lines[i] for i in alive_idx], held)
    return [(alive_idx[i], r, h) for i, r, h in links if not (r == "state" and h in ignore_state)]


def link_costs(pool_lines: list, pool_groups: list[str], alive: list[bool], held: list[dict], held_group: dict[str, str],
               ignore_state: set[str] = frozenset()) -> dict[str, int]:
    """held-out group -> number of pool rows (whole groups) it would force out of the pool."""
    group_size = collections.Counter(g for g, a in zip(pool_groups, alive) if a)
    by_h: dict[str, set] = collections.defaultdict(set)
    for i, _, hid in _links(pool_lines, [i for i, a in enumerate(alive) if a], held, ignore_state):
        by_h[held_group[hid]].add(pool_groups[i])
    return {h: sum(group_size[g] for g in gs) for h, gs in by_h.items()}


def select_by_cost(held: list[dict], held_group: dict[str, str], costs: dict[str, int], targets: dict[str, int], salt: str,
                   max_links: int = MAX_POOL_LINKS) -> list[dict]:
    """Per family, whole groups in order (cost, stable hash) until the family's row target is met; groups above max_links are
    never taken. A group counts toward the family of its first row (a parent, its unknown twin and its traps share one)."""
    groups: dict[str, list] = collections.defaultdict(list)
    for h in held:
        groups[held_group[h["id"]]].append(h)
    by_fam: dict[str, list] = collections.defaultdict(list)
    for g, rs in groups.items():
        by_fam[rs[0]["family"]].append(g)
    out = []
    for fam, gs in sorted(by_fam.items()):
        want = targets.get(fam, 0)
        n = 0
        for g in sorted(gs, key=lambda g: (costs.get(g, 0), ac.stable_hash(salt, g))):
            if n >= want:
                break
            if costs.get(g, 0) > max_links:
                continue
            out += groups[g]
            n += len(groups[g])
    return out


def resolve_links(pool_lines: list, pool_groups: list[str], held: list[dict], held_group: dict[str, str],
                  in_pool: set[str], ignore_state: set[str] = frozenset(), max_links: int = MAX_POOL_LINKS,
                  max_rounds: int = 8, log=print):
    """Iterate scan -> (remove linked pool groups | drop held-out groups) until no link remains.

    Returns (removed pool group set, dropped held-out group set, per-round stats). `in_pool` are held-out groups whose rows
    are in pool_lines (the B/C slice): they are never scanned as pool rows, and a dropped one is DISCARDED (neither held out
    nor in the pool), so the pool only shrinks and the loop converges. Links of rows in `ignore_state` by the exact-whole-
    state rule are ignored (GUI: see above)."""
    group_size = collections.Counter(pool_groups)
    removed: set[str] = set()
    dropped: set[str] = set()
    rounds = []
    for rnd in range(max_rounds):
        alive_idx = [i for i, g in enumerate(pool_groups) if g not in removed and g not in in_pool]
        cur_held = [h for h in held if held_group[h["id"]] not in dropped]
        links = _links(pool_lines, alive_idx, cur_held, ignore_state)
        reasons = collections.Counter(r for _, r, _ in links)
        log(f"  link round {rnd + 1}: {len(links):,} links {dict(reasons)} over {len(alive_idx):,} pool rows, "
            f"{len(cur_held):,} held-out rows")
        rounds.append({"round": rnd + 1, "links": len(links), "by_reason": dict(reasons), "pool_rows_scanned": len(alive_idx),
                       "heldout_rows": len(cur_held)})
        if not links:
            break
        by_h: dict[str, set] = collections.defaultdict(set)
        for i, _, hid in links:
            by_h[held_group[hid]].add(pool_groups[i])
        for h, gs in by_h.items():
            if sum(group_size[g] for g in gs) > max_links:
                dropped.add(h)
        for h, gs in by_h.items():
            if h not in dropped:
                removed |= gs
        rounds[-1].update({"heldout_groups_dropped_total": len(dropped), "pool_groups_removed_total": len(removed),
                           "pool_rows_removed_total": sum(group_size[g] for g in removed)})
    else:
        raise RuntimeError("held-out/pool links remain after the last round")
    return removed, dropped, rounds
