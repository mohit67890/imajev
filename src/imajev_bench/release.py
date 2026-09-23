"""Release gate: every condition a public imajev-bench version must meet, in one report."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .lint import lint
from .schema import validate_records
from .stats import evidence_clusters, required_clusters
from .triage import audit_report

RELEASE_TARGETS = {
    # Independent evidence clusters per track and split (not records). With <=3 items per cluster and
    # ICC 0.3, 170 test clusters per track (~510 overall) detect a 5-point pooled paired difference and
    # about 8 points within one track (docs/imajev-bench-v2-plan.md, "Size").
    "min_clusters": {"test": 170, "calibration": 35, "dev": 35},
    "max_judgement_dependent_share": 0.10,
    "min_kappa": 0.70,
    "blind_margin": 0.10,           # no-image accuracy on answerable visual/joint items <= majority + margin
    "detectable_difference": 0.05,  # paired accuracy difference the test split should detect
    "min_audited": 100,             # double-human audits of model-consensus items (triage route only)
    "max_consensus_error": 0.02,
}
PILOT_TARGETS = {**RELEASE_TARGETS, "min_clusters": {"test": 35, "calibration": 10, "dev": 15}, "detectable_difference": 0.10,
                 "min_audited": 30}


def _key(value):
    return f"{type(value).__name__}:{value!r}"


def annotator_agreement(records):
    """Raw agreement and Cohen's kappa between the first two human reviews of each record."""
    pairs = [(r["provenance"]["reviews"][0]["value"], r["provenance"]["reviews"][1]["value"])
             for r in records if len((r.get("provenance") or {}).get("reviews", [])) >= 2]
    if not pairs:
        return {"pairs": 0, "agreement": None, "kappa": None}
    observed = sum(_key(a) == _key(b) for a, b in pairs) / len(pairs)
    first, second = Counter(_key(a) for a, _ in pairs), Counter(_key(b) for _, b in pairs)
    expected = sum(first[k] * second[k] for k in first) / len(pairs) ** 2
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return {"pairs": len(pairs), "agreement": observed, "kappa": kappa,
            "adjudicated": sum("adjudication" in r["provenance"] for r in records)}


def blind_baseline(records, predictions_path):
    """Score a completed no_image run: answerable visual/joint accuracy against the full-evidence label."""
    rows = {json.loads(line)["id"]: json.loads(line) for line in Path(predictions_path).read_text().splitlines()}
    manifest = json.loads((Path(predictions_path).parent / "manifest.json").read_text())
    if manifest.get("condition") != "no_image":
        raise ValueError("Blind baseline requires a harness run with condition=no_image")
    items = [r for r in records if r["track"] in ("visual", "joint") and r["gold"] is not None]
    correct = sum(rows.get(r["id"], {}).get("status") == "answered"
                  and _key(rows[r["id"]].get("value")) == _key(r["gold"]) for r in items)
    families = defaultdict(Counter)
    for r in items:
        families[r["family"]][_key(r["gold"])] += 1
    majority = sum(c.most_common(1)[0][1] for c in families.values()) / len(items) if items else None
    return {"items": len(items), "blind_accuracy": correct / len(items) if items else None,
            "family_majority_accuracy": majority, "model": manifest.get("backend", {}).get("repo")}


def release_check(records: list[dict], root: Path, targets: dict | None = None,
                  blind_runs: list[Path] = ()) -> dict[str, Any]:
    targets = {**RELEASE_TARGETS, **(targets or {})}
    gates = []

    def gate(name, ok, detail, **extra):
        gates.append({"gate": name, "passed": bool(ok), "detail": detail, **extra})

    try:
        validate_records(records, Path(root), require_reviewed=True)
        gate("human_review", True, "Every record has a validated label route (see limitations for the audit method).")
    except ValueError as exc:
        gate("human_review", False, str(exc))

    lint_report = lint(records, Path(root))
    gate("construction_lint", lint_report["status"] != "fail",
         f"Lint status {lint_report['status']}: {lint_report['counts']}",
         failing=[c["check"] for c in lint_report["checks"] if c["status"] == "fail"])

    clusters = evidence_clusters(records)
    per = defaultdict(set)
    for r in records:
        per[(r["split"], r["track"])].add(clusters[r["id"]])
    shortfalls = {f"{split}/{track}": len(per[(split, track)])
                  for split, minimum in targets["min_clusters"].items() for track in ("text", "visual", "joint")
                  if len(per[(split, track)]) < minimum}
    gate("independent_clusters", not shortfalls,
         "Clusters per split/track meet targets." if not shortfalls else f"Below target {targets['min_clusters']}",
         counts={f"{s}/{t}": len(v) for (s, t), v in sorted(per.items())}, shortfalls=shortfalls)

    test = [r for r in records if r["split"] == "test"]
    judged = sum(bool(r["provenance"].get("judgement_dependent")) for r in test)
    share = judged / len(test) if test else 0.0
    gate("judgement_dependent", share <= targets["max_judgement_dependent_share"],
         f"{judged}/{len(test)} test records are marked judgement-dependent ({share:.0%}).")

    agreement = annotator_agreement(records)
    double_route = any(r["provenance"].get("review_plan") == "double" or r["provenance"].get("review_route") == "double_human"
                       for r in records)
    if agreement["pairs"] == 0 and not double_route:
        gate("annotator_agreement", True, "Not applicable: no record uses the two-human review route.", **agreement)
    else:
        gate("annotator_agreement", agreement["kappa"] is not None and agreement["kappa"] >= targets["min_kappa"],
             f"Cohen's kappa {agreement['kappa']} over {agreement['pairs']} double-reviewed records.", **agreement)

    consensus_labels = sum(r["provenance"].get("review_route") in ("consensus_verified", "construction_verified")
                           for r in records)
    if consensus_labels:
        audit = audit_report(records)
        gate("consensus_audit", audit["audited"] >= targets["min_audited"]
             and audit["error_rate"] is not None and audit["error_rate"] <= targets["max_consensus_error"],
             f"{consensus_labels} labels use the model-consensus or construction route; audited {audit['audited']} "
             f"({audit['audit_method']} audit) with {audit['consensus_errors']} errors "
             f"(upper 95% bound {audit['wilson_upper_95']}).", **audit)
    else:
        gate("consensus_audit", True, "No labels used the model-consensus or construction route.")

    blind = [blind_baseline(records, path) for path in blind_runs]
    if blind:
        worst = max(blind, key=lambda b: (b["blind_accuracy"] or 0) - (b["family_majority_accuracy"] or 0))
        gate("image_necessity", all((b["blind_accuracy"] or 0) <= (b["family_majority_accuracy"] or 0) + targets["blind_margin"]
                                    for b in blind),
             f"Worst no-image run: {worst['blind_accuracy']} vs family majority {worst['family_majority_accuracy']}.",
             runs=blind)
    else:
        gate("image_necessity", False, "No no_image harness run supplied; image necessity is unverified.")

    test_clusters = len({clusters[r["id"]] for r in test})
    size = len(test) / test_clusters if test_clusters else 1.0
    power = required_clusters(targets["detectable_difference"], discordant_rate=0.25,
                              items_per_cluster=max(size, 1.0), icc=0.3)
    gate("statistical_power", test_clusters >= power["clusters"],
         f"{test_clusters} test clusters; about {power['clusters']} needed to detect a "
         f"{targets['detectable_difference']:.0%} paired difference (assumes 25% discordance, ICC 0.3).", **power)

    return {"release_ready": all(g["passed"] for g in gates), "targets": targets, "gates": gates, "lint": lint_report,
            "limitations": limitations(records, gates)}


def limitations(records: list[dict], gates: list[dict]) -> list[str]:
    """Plain-language limitations derived from the data and the unmet gates, for the datasheet."""
    notes = [f"Release gate not met: {g['gate']} ({g['detail']})" for g in gates if not g["passed"]]
    image_records = [r for r in records if r["images"]]
    sources = [s for r in image_records for s in (r["provenance"].get("image_sources") or [])]
    synthetic = [s for s in sources if s.get("synthetic")]
    if sources:
        generators = Counter(s.get("generator", "unknown") for s in synthetic)
        mix = ", ".join(f"{g}: {n}" for g, n in generators.most_common())
        if len(synthetic) == len(sources):
            notes.append(f"All {len(sources)} image references are AI-generated ({mix}). This version contains no real "
                         "photographs, so results say nothing direct about performance on real-world photos.")
        elif synthetic:
            notes.append(f"{len(synthetic)}/{len(sources)} image references are AI-generated ({mix}); report results "
                         "separately for real and generated images, and base real-world claims on the real ones.")
        checkers = Counter(s.get("image_checks", "") for s in synthetic)
        if checkers:
            notes.append("Generated images were kept only when two checker models confirmed every stated fact "
                         f"({'; '.join(sorted(checkers))[:300]}). This acceptance step favours those checker models on "
                         "perception questions, so flag them in any comparison.")
    routes = Counter(r["provenance"].get("review_route", "unreviewed") for r in records)
    notes.append("Label routes: " + ", ".join(f"{k}: {v}" for k, v in routes.most_common()) + ". Construction labels are "
                 "computed from the scene or text specification; a seeded sample and every occlusion-based Unknown are "
                 "human-audited, and the audited error rate is reported with its upper bound.")
    model_audited = [r for r in records if r["provenance"].get("model_audit")]
    if model_audited:
        auditors = sorted({a for r in model_audited for a in r["provenance"]["model_audit"].get("auditors", [])})
        notes.append(f"The audit of {len(model_audited)} items was done by AI models ({', '.join(auditors)}), not by "
                     "humans: blind answers from auditors outside the generator, checker and leaderboard roles, with every "
                     "disagreement adjudicated by the pipeline author's model. Treat the audited error rate as a model "
                     "audit.")
    unknown = sum(r["gold"] is None for r in records)
    notes.append(f"{unknown}/{len(records)} references are Unknown (insufficient evidence).")
    notes.append("Scores depend on each model's interface and reasoning setting (direct option scoring vs structured "
                 "generation; reasoning effort); every leaderboard entry must state both.")
    return notes


def datasheet(report: dict, records: list[dict], name: str) -> str:
    """Draft datasheet; the authors must complete every TODO before publication."""
    counts = Counter((r["split"], r["track"]) for r in records)
    lines = [f"# Datasheet: {name}", "", "Generated from the release check. Complete every TODO by hand.", "",
             "## Composition", "", "| Split | Text | Visual | Joint |", "| --- | ---: | ---: | ---: |"]
    for split in ("dev", "calibration", "test"):
        lines.append(f"| {split} | " + " | ".join(str(counts[(split, t)]) for t in ("text", "visual", "joint")) + " |")
    lines += ["", f"Unknown references: {sum(r['gold'] is None for r in records)} of {len(records)}.", "",
              "## Release gates", ""]
    lines += [f"- {'PASS' if g['passed'] else 'FAIL'} `{g['gate']}`: {g['detail']}" for g in report["gates"]]
    lines += ["", "## Known limitations", ""] + [f"- {note}" for note in report.get("limitations", [])]
    lines += ["", "## Collection and licensing", "", "TODO: sources, licenses, consent, removal contact.", "",
              "## Annotation", "", "TODO: annotator pool, training, pay, instructions, adjudication rule.", "",
              "## Intended use and limits", "", "TODO: typed decision evaluation only; not a safety or deployment certificate.", "",
              "## Maintenance", "", "TODO: versioning, errata process, hidden-test submission policy.", ""]
    return "\n".join(lines)
