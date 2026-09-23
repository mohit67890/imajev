"""Shortcut and construction-quality checks run before any model sees a dataset.

Each check returns pass, warn or fail with evidence. Release gating treats fail as blocking.
Thresholds are explicit in DEFAULT_GATES and recorded with every report.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .stats import constant_baseline_contrast, contrast_info, evidence_clusters

DEFAULT_GATES = {
    "cue_rate_gap_fail": 0.25,        # abstention-cue rate on Unknown gold minus rate on answerable gold
    "cue_rate_gap_warn": 0.10,
    "majority_baseline_fail": 0.65,   # best single answer per family (families with >= min_family_n)
    "majority_baseline_warn": 0.55,
    "min_family_n": 10,
    "unknown_share": (0.10, 0.35),
    "max_records_per_image": 3,
    "constant_contrast_warn": 0.60,
    "near_duplicate_bits": 6,
}

ABSTENTION_CUE = re.compile(
    r"\b(unknown|undetermined|not determined|cannot be determined|can't be determined|insufficient|"
    r"not (?:shown|visible|printed|readable|legible)|if no\b|if none\b|abstain)", re.IGNORECASE)
REAL_PROVENANCE = ("source", "source_page", "creator", "license")


def _field(record):
    return record["request"]["fields"][0]


def _options(field):
    if field["type"] == "boolean":
        return [True, False]
    if field["type"] == "choice":
        return [o["value"] for o in field["options"]]
    return [level["value"] for level in field["levels"]]


def _text(record):
    field = _field(record)
    return " ".join([field["question"], json.dumps(record["request"].get("state", {}), ensure_ascii=False)])


def _gold_key(value):
    return f"{type(value).__name__}:{value!r}"


def _result(check, status, detail, **evidence):
    return {"check": check, "status": status, "detail": detail, **evidence}


def check_abstention_cues(records, gates):
    """Unknown must not be predictable from wording such as 'return unknown if ...'."""
    unknown = [r for r in records if r["gold"] is None]
    answerable = [r for r in records if r["gold"] is not None]
    if len(unknown) < 3 or not answerable:
        return _result("abstention_cues", "pass", "Too few Unknown references to test.")
    cued = {r["id"] for r in records if ABSTENTION_CUE.search(_text(r))}
    rate_u = sum(r["id"] in cued for r in unknown) / len(unknown)
    rate_a = sum(r["id"] in cued for r in answerable) / len(answerable)
    gap = rate_u - rate_a
    status = "fail" if gap > gates["cue_rate_gap_fail"] else "warn" if gap > gates["cue_rate_gap_warn"] else "pass"
    cue_rule_correct = sum((r["id"] in cued) == (r["gold"] is None) for r in records)
    return _result("abstention_cues", status,
                   f"Cue on {rate_u:.0%} of Unknown vs {rate_a:.0%} of answerable items (gap {gap:+.2f}).",
                   cue_only_unknown_detection_accuracy=cue_rule_correct / len(records),
                   unknown_items_without_cue=[r["id"] for r in unknown if r["id"] not in cued][:20],
                   answerable_items_with_cue=[r["id"] for r in answerable if r["id"] in cued][:20])


def check_label_balance(records, gates):
    """Per family: best constant answer, boolean skew, and gold position skew for choices."""
    families = defaultdict(list)
    for r in records:
        families[(r["track"], r["family"])].append(r)
    rows, worst = [], "pass"
    for (track, family), items in sorted(families.items()):
        counts = Counter(_gold_key(r["gold"]) for r in items)
        majority = counts.most_common(1)[0][1] / len(items)
        positions = Counter()
        for r in items:
            options = _options(_field(r))
            positions["unknown" if r["gold"] is None else options.index(r["gold"])] += 1
        status = "pass"
        if len(items) >= gates["min_family_n"]:
            status = ("fail" if majority > gates["majority_baseline_fail"]
                      else "warn" if majority > gates["majority_baseline_warn"] else "pass")
        worst = max(worst, status, key=("pass", "warn", "fail").index)
        rows.append({"track": track, "family": family, "n": len(items), "majority_baseline": round(majority, 3),
                     "gold_counts": dict(counts), "gold_positions": {str(k): v for k, v in positions.items()},
                     "status": status})
    return _result("label_balance", worst, "Best constant answer per family (families below min_family_n only reported).",
                   families=rows)


def check_unknown_share(records, gates):
    low, high = gates["unknown_share"]
    share = sum(r["gold"] is None for r in records) / len(records)
    status = "pass" if low <= share <= high else "warn"
    return _result("unknown_share", status, f"Unknown references are {share:.0%} of records (target {low:.0%}-{high:.0%}).")


def check_image_reuse(records, gates):
    per_image = Counter(image["sha256"] for r in records for image in r["images"])
    over = {k[:12]: v for k, v in per_image.items() if v > gates["max_records_per_image"]}
    status = "fail" if over else "pass"
    return _result("image_reuse", status,
                   f"{len(per_image)} unique images; {len(over)} used more than {gates['max_records_per_image']} times.",
                   overused=over)


def check_sibling_pairs(records, gates):
    """Undeclared same-question siblings with opposite labels let one item reveal the other."""
    buckets = defaultdict(list)
    for r in records:
        if contrast_info(r) is None:
            buckets[(r["group_id"], r["family"], _field(r)["question"])].append(r)
    pairs = opposite = 0
    for items in buckets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs += 1
                opposite += _gold_key(items[i]["gold"]) != _gold_key(items[j]["gold"])
    status = "warn" if pairs >= 5 and opposite / pairs >= 0.9 else "pass"
    return _result("sibling_pairs", status,
                   f"{pairs} undeclared same-question sibling pairs; {opposite} have different labels. "
                   "Declare intended contrasts in provenance.contrast so they are scored as sets.")


def check_contrast_sets(records, gates):
    try:
        baseline = constant_baseline_contrast(records)
    except ValueError as exc:
        return _result("contrast_sets", "fail", str(exc))
    if not baseline["sets"]:
        return _result("contrast_sets", "pass", "No contrast sets declared.")
    sets = defaultdict(list)
    for r in records:
        c = contrast_info(r)
        if c:
            sets[c["set_id"]].append((r, c))
    problems, relations, original_golds = [], Counter(), defaultdict(Counter)
    for set_id, members in sets.items():
        originals = [r for r, c in members if c["role"] == "original"]
        if len(originals) != 1 or len(members) < 2:
            problems.append(f"{set_id}: needs one original and at least one variant")
            continue
        original = originals[0]
        original_golds[original["family"]][_gold_key(original["gold"])] += 1
        for r, c in members:
            if c["role"] != "variant":
                continue
            relations[c["relation"]] += 1
            same = _gold_key(r["gold"]) == _gold_key(original["gold"])
            if (c["relation"] == "same") != same:
                problems.append(f"{set_id}/{r['id']}: relation {c['relation']} contradicts the gold labels")
    skewed = {f: dict(c) for f, c in original_golds.items() if sum(c.values()) >= 6 and max(c.values()) / sum(c.values()) > 0.8}
    status = "fail" if problems else "warn" if (
        baseline["best_constant_row_accuracy"] > gates["constant_contrast_warn"] or skewed) else "pass"
    return _result("contrast_sets", status,
                   f"{baseline['sets']} sets; an image-blind constant answer scores "
                   f"{baseline['best_constant_row_accuracy']:.0%} of rows.",
                   relations=dict(relations), problems=problems[:20], skewed_original_labels=skewed)


def check_split_isolation(records, gates):
    clusters = evidence_clusters(records)
    splits = defaultdict(set)
    for r in records:
        splits[clusters[r["id"]]].add(r["split"])
    leaking = sorted(k for k, v in splits.items() if len(v) > 1)
    return _result("split_isolation", "fail" if leaking else "pass",
                   f"{len(splits)} evidence clusters; {len(leaking)} span more than one split.", leaking=leaking[:20])


def check_provenance(records, gates):
    missing = []
    for r in records:
        if not r["images"]:
            continue
        p = r.get("provenance") or {}
        if p.get("synthetic") is True and not p.get("image_sources"):
            if not p.get("generator"):
                missing.append((r["id"], ["generator"]))
            continue
        per_image = p.get("image_sources")
        if isinstance(per_image, list) and per_image:
            # One source entry per distinct image; each needs its own license and attribution.
            absent = sorted({key for source in per_image
                             for key in (("generator",) if source.get("synthetic") else REAL_PROVENANCE[1:])
                             if not source.get(key)})
            if len(per_image) < len({image["sha256"] for image in r["images"]}):
                absent.append("image_sources entry per image")
        else:
            absent = [key for key in REAL_PROVENANCE if not p.get(key)]
        if absent:
            missing.append((r["id"], absent))
    return _result("provenance", "fail" if missing else "pass",
                   f"{len(missing)} image records lack license/source provenance.", missing=missing[:20])


def check_answer_in_state(records, gates):
    """A choice answer quoted in the state while distractors are not is a text shortcut."""
    hits, eligible = [], 0
    for r in records:
        field = _field(r)
        if field["type"] != "choice" or r["gold"] is None:
            continue
        eligible += 1
        state = json.dumps(r["request"].get("state", {}), ensure_ascii=False).lower()
        present = [o for o in _options(field) if str(o).lower() in state]
        if present == [r["gold"]]:
            hits.append(r["id"])
    rate = len(hits) / eligible if eligible else 0.0
    status = "warn" if eligible >= 10 and rate > 0.3 else "pass"
    return _result("answer_in_state", status, f"{len(hits)}/{eligible} choice items quote only the gold option in state.",
                   items=hits[:20])


def dhash(path, size=8):
    from PIL import Image
    with Image.open(path) as image:
        pixels = image.convert("L").resize((size + 1, size)).tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = bits << 1 | (pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1])
    return bits


def check_near_duplicates(records, gates, root=None):
    if root is None:
        return _result("near_duplicates", "pass", "Skipped: no dataset root supplied.", skipped=True)
    try:
        import PIL  # noqa: F401
    except ImportError:
        return _result("near_duplicates", "warn", "Skipped: Pillow is not installed.", skipped=True)
    clusters = evidence_clusters(records)
    seen = {}
    for r in records:
        for image in r["images"]:
            seen.setdefault(image["sha256"], (dhash(Path(root) / image["path"]), clusters[r["id"]], r["split"]))
    items = list(seen.items())
    cross_cluster, cross_split = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ha, (a, ca, sa)), (hb, (b, cb, sb)) = items[i], items[j]
            if ca != cb and bin(a ^ b).count("1") <= gates["near_duplicate_bits"]:
                (cross_split if sa != sb else cross_cluster).append((ha[:12], hb[:12]))
    status = "fail" if cross_split else "warn" if cross_cluster else "pass"
    return _result("near_duplicates", status,
                   f"{len(cross_split)} near-duplicate pairs cross splits; {len(cross_cluster)} link separate clusters.",
                   cross_split=cross_split[:20], cross_cluster=cross_cluster[:20])


CHECKS = (check_abstention_cues, check_label_balance, check_unknown_share, check_image_reuse, check_sibling_pairs,
          check_contrast_sets, check_split_isolation, check_provenance, check_answer_in_state)


def lint(records: list[dict[str, Any]], root: Path | None = None, gates: dict | None = None) -> dict[str, Any]:
    gates = {**DEFAULT_GATES, **(gates or {})}
    results = [check(records, gates) for check in CHECKS] + [check_near_duplicates(records, gates, root)]
    counts = Counter(r["status"] for r in results)
    return {"gates": {k: list(v) if isinstance(v, tuple) else v for k, v in gates.items()},
            "status": "fail" if counts["fail"] else "warn" if counts["warn"] else "pass",
            "counts": dict(counts), "checks": results}
