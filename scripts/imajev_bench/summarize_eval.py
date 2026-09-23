"""Summarise API/harness evaluation runs on one imajev-bench dataset (all splits pooled, per split listed).

Each run directory must be a completed, verifiable run (manifest/completion receipts). Scoring uses the
framework's own `score` and `stats.contrast_metrics`, so numbers match `imajev_bench score`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imajev_bench.cli import read_jsonl  # noqa: E402
from imajev_bench.runner import verify_run  # noqa: E402
from imajev_bench.schema import validate_records  # noqa: E402
from imajev_bench.scoring import _correct, score  # noqa: E402
from imajev_bench.stats import cluster_bootstrap, constant_baseline_contrast, contrast_metrics, evidence_clusters  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="Directory of <model>-<split> run folders")
    parser.add_argument("--checkers", nargs="*", default=["gpt-5.6-luna", "gemini-3.1-pro"],
                        help="Run-name prefixes of models that also checked the images (selection-bias flag)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = validate_records(read_jsonl(args.records), args.records.parent)
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    runs = defaultdict(dict)
    for folder in sorted(p for p in args.runs.iterdir() if (p / "completion.json").exists()):
        model, split = folder.name.rsplit("-", 1)
        verify_run(by_split[split], folder / "predictions.jsonl")
        runs[model].update({row["id"]: row for row in read_jsonl(folder / "predictions.jsonl")})
    clusters = evidence_clusters(records)
    lines = [f"# Evaluation summary: {args.records.parent.name}", "",
             f"{len(records)} records, {len(set(clusters.values()))} evidence clusters, "
             f"{sum(r['gold'] is None for r in records)} Unknown references. Models marked * also checked the images "
             "during generation, so accepted images were selected to be readable by them.", "",
             "| Model | Accuracy | Text | Visual | Joint | Correct Unknown | False abstention | Errors | Contrast sets all correct |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summary = {}
    for model, predictions in sorted(runs.items()):
        missing = [r["id"] for r in records if r["id"] not in predictions]
        if missing:
            lines.append(f"| {model} | incomplete ({len(missing)} missing) | | | | | | | |")
            continue
        report = score(records, predictions, 0)
        correct = {r["id"]: _correct(r, predictions[r["id"]]) for r in records}
        tracks = {}
        for track in ("text", "visual", "joint"):
            rows = [r for r in records if r["track"] == track]
            tracks[track] = (sum(correct[r["id"]] for r in rows), len(rows))
        unknown = [r for r in records if r["gold"] is None]
        answerable = [r for r in records if r["gold"] is not None]
        cu = sum(predictions[r["id"]].get("status") == "abstained" for r in unknown)
        fa = sum(predictions[r["id"]].get("status") == "abstained" for r in answerable)
        errors = sum(predictions[r["id"]].get("status") == "error" for r in records)
        contrast = contrast_metrics(records, predictions)
        ci = cluster_bootstrap(correct, clusters)
        total = sum(correct.values())
        flag = "*" if any(model.startswith(c) for c in args.checkers) else ""
        cell = lambda k: f"{tracks[k][0]}/{tracks[k][1]}"
        interval = f" [{ci['low']:.2f}, {ci['high']:.2f}]" if ci["low"] is not None else ""
        lines.append(f"| {model}{flag} | {total}/{len(records)} ({total / len(records):.0%}){interval} | {cell('text')} | "
                     f"{cell('visual')} | {cell('joint')} | {cu}/{len(unknown)} | {fa}/{len(answerable)} | {errors} | "
                     f"{contrast['all_correct']}/{contrast['sets']} |")
        summary[model] = {"correct": total, "n": len(records), "tracks": tracks, "correct_unknown": cu,
                          "false_abstention": fa, "errors": errors, "contrast": {k: v for k, v in contrast.items() if k != "per_set"},
                          "cluster_ci": ci, "family": {f: [sum(correct[r["id"]] for r in records if r["family"] == f),
                                                           sum(r["family"] == f for r in records)]
                                                       for f in sorted({r["family"] for r in records})}}
    # Robustness views: without flagged items, and per image generator (home-advantage check).
    flagged = {r["id"] for r in records if r["provenance"].get("judgement_dependent")}
    def generator(r):
        sources = r["provenance"].get("image_sources") or []
        return sources[0].get("generator", "real/unknown") if sources else "text-only"
    strata = sorted({generator(r) for r in records})
    lines += ["", f"Robustness: accuracy excluding the {len(flagged)} flagged items (judgement calls, image defects, premise "
              "mismatches), and by image generator.", "",
              "| Model | Excl. flagged | " + " | ".join(strata) + " |", "| --- | ---: | " + " | ".join("---:" for _ in strata) + " |"]
    for model in summary:
        predictions = runs[model]
        keep = [r for r in records if r["id"] not in flagged]
        ok = sum(_correct(r, predictions[r["id"]]) for r in keep)
        cells = []
        for stratum in strata:
            rows = [r for r in records if generator(r) == stratum]
            cells.append(f"{sum(_correct(r, predictions[r['id']]) for r in rows)}/{len(rows)}")
        lines.append(f"| {model} | {ok}/{len(keep)} ({ok / len(keep):.1%}) | " + " | ".join(cells) + " |")
        summary[model]["excluding_flagged"] = [ok, len(keep)]
    constant = constant_baseline_contrast(records)
    majority = sum(Counter(repr(r["gold"]) for r in records if r["family"] == f).most_common(1)[0][1]
                   for f in {r["family"] for r in records})
    lines += ["", f"Baselines: family-majority constant answer {majority}/{len(records)} ({majority / len(records):.0%}); "
              f"image-blind constant answer on contrast sets {constant['best_constant_row_accuracy']:.0%} of rows, "
              f"{constant['sets_solvable_by_constant']}/{constant['sets']} sets.", "",
              "Per-family accuracy:", ""]
    families = sorted({r["family"] for r in records})
    lines.append("| Family | n | " + " | ".join(summary) + " |")
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in summary) + " |")
    for family in families:
        n = sum(r["family"] == family for r in records)
        lines.append(f"| {family} | {n} | " + " | ".join(str(summary[m]["family"][family][0]) for m in summary) + " |")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
