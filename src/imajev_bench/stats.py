"""Evidence-cluster statistics: independence units, cluster intervals, paired tests, contrast sets.

Rows that share an image, a declared group, a source cluster or a contrast set are one
statistical unit. Every interval and paired test here resamples those units, never rows.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Any, Mapping


def _typed_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def contrast_info(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return provenance.contrast = {set_id, role: original|variant, relation: change|same} or None."""
    contrast = (record.get("provenance") or {}).get("contrast")
    if contrast is None:
        return None
    if not isinstance(contrast, Mapping) or not isinstance(contrast.get("set_id"), str) or not contrast["set_id"]:
        raise ValueError(f"record {record.get('id')!r}: provenance.contrast requires a set_id")
    role = contrast.get("role")
    if role == "original":
        if "relation" in contrast:
            raise ValueError(f"record {record.get('id')!r}: the contrast original has no relation")
    elif role == "variant":
        if contrast.get("relation") not in ("change", "same"):
            raise ValueError(f"record {record.get('id')!r}: contrast variants need relation change or same")
    else:
        raise ValueError(f"record {record.get('id')!r}: contrast role must be original or variant")
    return dict(contrast)


def evidence_clusters(records: list[Mapping[str, Any]]) -> dict[str, str]:
    """Map record ID to a connected-component ID over every declared or exact shared-evidence link."""
    parent: dict[tuple, tuple] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        parent[find(a)] = find(b)

    for record in records:
        node = ("record", record["id"])
        find(node)
        union(node, ("group", record["group_id"]))
        for image in record.get("images", []):
            union(node, ("image", image["sha256"]))
        provenance = record.get("provenance") or {}
        for source in [provenance.get("source_cluster"), *(provenance.get("source_clusters") or [])]:
            if source:
                union(node, ("source", source))
        contrast = contrast_info(record)
        if contrast:
            union(node, ("contrast", contrast["set_id"]))
    roots, clusters = {}, {}
    for record in records:
        root = find(("record", record["id"]))
        clusters[record["id"]] = roots.setdefault(root, f"cluster-{len(roots) + 1:04d}")
    return clusters


def _by_cluster(ids, clusters):
    grouped = defaultdict(list)
    for item in ids:
        grouped[clusters[item]].append(item)
    return [grouped[key] for key in sorted(grouped)]


def _percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap(correct: Mapping[str, bool], clusters: Mapping[str, str],
                      samples: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Percentile bootstrap of request-level accuracy that resamples whole evidence clusters."""
    ids = sorted(correct)
    units = _by_cluster(ids, clusters)
    estimate = sum(correct[i] for i in ids) / len(ids) if ids else None
    report = {"method": "percentile bootstrap over evidence clusters", "estimate": estimate,
              "records": len(ids), "clusters": len(units), "samples": samples, "low": None, "high": None}
    if len(units) < 10:
        report["suppressed_reason"] = "Fewer than 10 independent evidence clusters; an interval would be misleading."
        return report
    rng, draws = random.Random(seed), []
    for _ in range(samples):
        chosen = [unit for unit in (rng.choice(units) for _ in units)]
        flat = [i for unit in chosen for i in unit]
        draws.append(sum(correct[i] for i in flat) / len(flat))
    report.update(low=_percentile(draws, 0.025), high=_percentile(draws, 0.975))
    return report


def paired_cluster_test(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool],
                        clusters: Mapping[str, str], samples: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Paired accuracy difference A-B: cluster bootstrap interval plus a cluster sign-flip test.

    Under the null that the two systems are exchangeable, each cluster's summed difference is
    equally likely to have either sign. Up to 16 non-zero clusters are enumerated exactly.
    """
    if set(correct_a) != set(correct_b):
        raise ValueError("Paired comparison requires the same record IDs")
    ids = sorted(correct_a)
    units = _by_cluster(ids, clusters)
    diffs = [sum(int(correct_a[i]) - int(correct_b[i]) for i in unit) for unit in units]
    total = sum(diffs)
    observed = total / len(ids) if ids else 0.0
    nonzero = [d for d in diffs if d]
    if not nonzero:
        p_value, method = 1.0, "no discordant clusters"
    elif len(nonzero) <= 16:
        extreme = sum(abs(sum(s * d for s, d in zip(signs, nonzero))) >= abs(total) - 1e-12
                      for signs in itertools.product((1, -1), repeat=len(nonzero)))
        p_value, method = extreme / 2 ** len(nonzero), "exact cluster sign-flip"
    else:
        rng = random.Random(seed)
        extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= abs(total) - 1e-12
                      for _ in range(samples))
        p_value, method = (extreme + 1) / (samples + 1), "Monte Carlo cluster sign-flip"
    report = {"difference": observed, "records": len(ids), "clusters": len(units),
              "discordant_clusters": len(nonzero),
              "a_only_correct": sum(correct_a[i] and not correct_b[i] for i in ids),
              "b_only_correct": sum(correct_b[i] and not correct_a[i] for i in ids),
              "p_value": p_value, "test": method, "low": None, "high": None}
    if len(units) < 10:
        report["interval_suppressed_reason"] = "Fewer than 10 independent evidence clusters."
        return report
    rng, draws = random.Random(seed), []
    for _ in range(min(samples, 4000)):
        chosen = [rng.choice(range(len(units))) for _ in units]
        count = sum(len(units[k]) for k in chosen)
        draws.append(sum(diffs[k] for k in chosen) / count)
    report.update(low=_percentile(draws, 0.025), high=_percentile(draws, 0.975))
    return report


def contrast_metrics(records: list[Mapping[str, Any]], predictions: Mapping[str, Any]) -> dict[str, Any]:
    """Set-level robustness. Constant answers never earn change credit.

    predictions maps record ID to a prediction value (None for abstention) or to a result
    mapping with status/value.
    """
    sets: dict[str, dict[str, list]] = defaultdict(lambda: {"original": [], "variant": []})
    for record in records:
        contrast = contrast_info(record)
        if contrast:
            sets[contrast["set_id"]][contrast["role"]].append((record, contrast))

    def value(record):
        raw = predictions.get(record["id"])
        if isinstance(raw, Mapping):
            return None if raw.get("status") != "answered" else raw.get("value")
        return raw

    def correct(record):
        return _typed_equal(value(record), record["gold"])

    totals = dict(sets=0, all_correct=0, change_pairs=0, change_pairs_correct=0, change_detected=0,
                  same_pairs=0, same_pairs_correct=0, same_unchanged=0, constant_sets=0)
    per_set = []
    for set_id in sorted(sets):
        members = sets[set_id]
        if len(members["original"]) != 1 or not members["variant"]:
            raise ValueError(f"contrast set {set_id!r} needs exactly one original and at least one variant")
        (original, _), = members["original"]
        rows = [original] + [record for record, _ in members["variant"]]
        totals["sets"] += 1
        all_ok = all(correct(r) for r in rows)
        totals["all_correct"] += all_ok
        values = [value(r) for r in rows]
        constant = all(_typed_equal(v, values[0]) for v in values)
        totals["constant_sets"] += constant
        for record, contrast in members["variant"]:
            both = correct(original) and correct(record)
            moved = not _typed_equal(value(original), value(record))
            if contrast["relation"] == "change":
                totals["change_pairs"] += 1
                totals["change_pairs_correct"] += both
                totals["change_detected"] += moved
            else:
                totals["same_pairs"] += 1
                totals["same_pairs_correct"] += both
                totals["same_unchanged"] += not moved
        per_set.append({"set_id": set_id, "all_correct": all_ok, "constant": constant,
                        "pattern": "".join("1" if correct(r) else "0" for r in rows)})
    return {**totals, "per_set": per_set,
            "note": "Headline: all_correct and change_pairs_correct. same_unchanged alone rewards image-blind systems."}


def constant_baseline_contrast(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Best score an image-blind system could get per set by repeating one answer for every member."""
    sets = defaultdict(list)
    for record in records:
        contrast = contrast_info(record)
        if contrast:
            sets[contrast["set_id"]].append(record)
    best_rows = total_rows = all_correct = 0
    for rows in sets.values():
        golds = [r["gold"] for r in rows]
        counts = defaultdict(int)
        for gold in golds:
            counts[(type(gold).__name__, repr(gold))] += 1
        top = max(counts.values())
        best_rows += top
        total_rows += len(rows)
        all_correct += top == len(rows)
    return {"sets": len(sets), "rows": total_rows,
            "best_constant_row_accuracy": best_rows / total_rows if total_rows else None,
            "sets_solvable_by_constant": all_correct}


def intraclass_correlation(correct: Mapping[str, bool], clusters: Mapping[str, str]) -> float | None:
    """One-way ANOVA ICC estimate of correctness within clusters; None when undefined."""
    units = [[float(correct[i]) for i in unit] for unit in _by_cluster(sorted(correct), clusters)]
    k, n = len(units), sum(len(u) for u in units)
    if k < 2 or n <= k:
        return None
    grand = sum(sum(u) for u in units) / n
    between = sum(len(u) * (sum(u) / len(u) - grand) ** 2 for u in units) / (k - 1)
    within = sum((x - sum(u) / len(u)) ** 2 for u in units for x in u) / (n - k)
    m0 = (n - sum(len(u) ** 2 for u in units) / n) / (k - 1)
    denominator = between + (m0 - 1) * within
    return None if denominator <= 0 else max(0.0, (between - within) / denominator)


def required_clusters(delta: float, discordant_rate: float, items_per_cluster: float, icc: float,
                      alpha: float = 0.05, power: float = 0.8) -> dict[str, Any]:
    """Clusters needed to detect a paired accuracy difference `delta` (normal approximation).

    discordant_rate is the expected share of items where exactly one system is correct.
    """
    if not 0 < delta < 1 or not 0 < discordant_rate <= 1 or items_per_cluster < 1 or not 0 <= icc < 1:
        raise ValueError("Invalid power-analysis inputs")
    z = {0.05: 1.959964, 0.01: 2.575829}.get(alpha)
    zb = {0.8: 0.841621, 0.9: 1.281552}.get(power)
    if z is None or zb is None:
        raise ValueError("Supported alpha: 0.05/0.01; power: 0.8/0.9")
    variance = max(discordant_rate - delta ** 2, 1e-9)
    effective_items = (z + zb) ** 2 * variance / delta ** 2
    design_effect = 1 + (items_per_cluster - 1) * icc
    items = effective_items * design_effect
    return {"effective_items": math.ceil(effective_items), "design_effect": design_effect,
            "items": math.ceil(items), "clusters": math.ceil(items / items_per_cluster)}
