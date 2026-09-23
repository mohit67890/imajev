"""CLI for validation, blind annotation, serial inference and replay scoring."""
import argparse
import json
from pathlib import Path

from .review import build_review
from .runner import run, verify_run
from .schema import validate_records
from .scoring import score
from .annotations import import_prelabels, merge_reviews
from .api_models import PROVIDERS, make_provider, prelabel_export, run_api
from .triage import plan_construction, plan_reviews
from .audit import audit
from .assemble import assemble
from .lint import lint
from .release import PILOT_TARGETS, datasheet, release_check
from .scoring import _correct
from .stats import cluster_bootstrap, contrast_metrics, evidence_clusters, paired_cluster_test


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row {line_number}")
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {x}")))
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assembler = sub.add_parser("assemble", help="Build records and assets from an authoring spec")
    assembler.add_argument("--spec", type=Path, required=True)
    assembler.add_argument("--output", type=Path, required=True)
    for name in ("validate", "audit", "review", "import-reviews", "run", "score", "lint", "release-check", "compare",
                 "api-run", "import-prelabels", "plan-reviews", "promote-constructed"):
        p = sub.add_parser(name)
        p.add_argument("--records", type=Path, required=True)
        p.add_argument("--root", type=Path, help="Asset root; defaults to records directory")
        p.add_argument("--allow-draft", action="store_true", help="Explicitly permit unreviewed diagnostic data")
        if name != "validate":
            p.add_argument("--output", type=Path, required=True)
        if name in ("run", "score", "compare", "api-run"):
            p.add_argument("--split", choices=("dev", "calibration", "test"), required=True)
        if name == "run":
            p.add_argument("--adapter", choices=("first", "random", "unknown", "imajev-http"), required=True)
            p.add_argument("--endpoint", help="Explicit full URL, e.g. http://127.0.0.1:8765/v1/systemone")
            p.add_argument("--timeout", type=float, default=60)
            p.add_argument("--seed", type=int, default=0)
        if name == "score":
            p.add_argument("--predictions", type=Path, required=True)
            p.add_argument("--bootstrap-samples", type=int, default=1000)
        if name == "api-run":
            p.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
            p.add_argument("--model", required=True, help="Exact provider model ID; recorded with the served model")
            p.add_argument("--purpose", choices=("evaluation", "prelabel"), default="evaluation")
            p.add_argument("--free-json", action="store_true", help="Instruct JSON without a response schema")
            p.add_argument("--workers", type=int, default=1, help="Parallel API requests (results stay in dataset order)")
            p.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"),
                           help="OpenAI/Azure reasoning effort; recorded in the run manifest")
            p.add_argument("--limit", type=int)
            p.add_argument("--allow-test-exposure", action="store_true",
                           help="Permit sending hidden-test images to this provider (needs no-training terms)")
        if name == "import-prelabels":
            p.add_argument("--runs", type=Path, nargs="+", required=True, help="Completed api-run directories")
        if name == "promote-constructed":
            p.add_argument("--audit-share", type=float, default=0.15)
            p.add_argument("--seed", type=int, default=0)
        if name == "plan-reviews":
            p.add_argument("--audit-share", type=float, default=0.15)
            p.add_argument("--seed", type=int, default=0)
        if name == "compare":
            p.add_argument("--predictions-a", type=Path, required=True)
            p.add_argument("--predictions-b", type=Path, required=True)
        if name == "release-check":
            p.add_argument("--blind-runs", type=Path, nargs="*", default=[], help="Completed harness runs with condition=no_image")
            p.add_argument("--pilot", action="store_true", help="Use pilot-size targets instead of public-release targets")
            p.add_argument("--preview", action="store_true",
                           help="Label the result a preview: unmet gates become datasheet limitations, not a verdict")
        if name == "import-reviews":
            p.add_argument("--reviews", type=Path, nargs="+", required=True)
        if name == "review":
            p.add_argument("--reviewer-id")
            p.add_argument("--plan", choices=("all", "double", "audit"), default="all",
                           help="'double': second-reviewer packet; 'audit': construction/consensus audit sample")
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        print(json.dumps(assemble(json.loads(args.spec.read_text()), args.spec.parent, args.output), indent=2))
        return
    root = args.root or args.records.parent
    if args.command == "release-check":
        # The gate itself reports unreviewed records; do not stop at the first draft.
        records = validate_records(read_jsonl(args.records), root)
        report = release_check(records, root, PILOT_TARGETS if args.pilot else None, args.blind_runs)
        report["status"] = ("preview" if args.preview else "release_ready" if report["release_ready"] else "not_ready")
        _write_json(args.output, report)
        with args.output.with_name("DATASHEET.md").open("x") as handle:
            handle.write(datasheet(report, records, args.records.parent.name))
        print(json.dumps({"status": report["status"], "release_ready": report["release_ready"],
                          "unmet_gates": [g["gate"] for g in report["gates"] if not g["passed"]]}))
        return
    records = validate_records(read_jsonl(args.records), root, require_reviewed=not args.allow_draft)
    if args.command == "lint":
        report = lint(records, root)
        _write_json(args.output, report)
        print(json.dumps({"status": report["status"], "counts": report["counts"]}))
        return
    if args.command == "validate":
        print(json.dumps({"records": len(records), "groups": len({r['group_id'] for r in records}),
                          "reviewed": all(r['annotation_status'] == 'reviewed' for r in records)}))
        return
    if args.command == "review":
        selected = records if args.plan == "all" else [
            r for r in records if (r["provenance"].get("review_plan") == "double" if args.plan == "double"
                                   else r["provenance"].get("audit_sample") and r["annotation_status"] != "reviewed")]
        if not selected:
            parser.error("No records selected for this packet")
        print(build_review(selected, root, args.output, reviewer_id=args.reviewer_id, seed=args.seed))
        return
    if args.command == "import-prelabels":
        for run_dir in args.runs:
            records = import_prelabels(records, prelabel_export(run_dir, records))
        _write_jsonl(args.output, validate_records(records, root))
        print(json.dumps({"output": str(args.output), "runs": len(args.runs)}))
        return
    if args.command == "promote-constructed":
        planned, summary = plan_construction(records, args.audit_share, args.seed)
        _write_jsonl(args.output, validate_records(planned, root))
        _write_json(args.output.with_suffix(".plan.json"), summary)
        print(json.dumps(summary))
        return
    if args.command == "plan-reviews":
        planned, summary = plan_reviews(records, args.audit_share, args.seed)
        _write_jsonl(args.output, validate_records(planned, root))
        _write_json(args.output.with_suffix(".plan.json"), summary)
        print(json.dumps(summary))
        return
    if args.command == "audit":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(audit(records), handle, indent=2)
            handle.write("\n")
        print(args.output)
        return
    if args.command == "import-reviews":
        exports = [json.loads(path.read_text()) for path in args.reviews]
        merged = merge_reviews(records, exports, root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            for row in merged:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "reviewed": sum(r["annotation_status"] == "reviewed" for r in merged), "draft": sum(r["annotation_status"] == "draft" for r in merged)}))
        return
    records = [r for r in records if r["split"] == args.split]
    if not records:
        parser.error("Selected split has no records")
    if any(r["provenance"].get("quarantined") or r["provenance"].get("review_flags") for r in records):
        parser.error("Flagged groups are quarantined; resolve them through a dataset revision before evaluation")
    if args.command == "run":
        print(run(records, root, args.output, args.adapter, args.endpoint, args.timeout, args.seed))
        return
    if args.command == "api-run":
        provider = make_provider(args.provider, args.model, reasoning_effort=args.reasoning_effort)
        print(run_api(records[:args.limit], root, args.output, provider, constrained=not args.free_json,
                      allow_test_exposure=args.allow_test_exposure, purpose=args.purpose, workers=args.workers))
        return
    if args.command == "compare":
        runs = []
        for path in (args.predictions_a, args.predictions_b):
            verify_run(records, path)
            rows = {row["id"]: row for row in read_jsonl(path)}
            runs.append({r["id"]: _correct(r, rows.get(r["id"])) for r in records})
        report = paired_cluster_test(runs[0], runs[1], evidence_clusters(records))
        report.update(a=str(args.predictions_a), b=str(args.predictions_b), split=args.split)
        _write_json(args.output, report)
        print(json.dumps({k: report[k] for k in ("difference", "low", "high", "p_value", "clusters")}))
        return
    attribution = verify_run(records, args.predictions)
    prediction_rows = read_jsonl(args.predictions)
    predictions = {}
    for row in prediction_rows:
        if not isinstance(row.get("id"), str) or row["id"] in predictions:
            raise ValueError("Prediction IDs must be present and unique")
        predictions[row["id"]] = row
    if set(predictions) - {r["id"] for r in records}:
        raise ValueError("Predictions include IDs outside the selected dataset split")
    dataset_audit = audit(records)
    content_reuse = dataset_audit["exact_content_connected_clusters"] < dataset_audit["declared_groups"]
    result = score(records, predictions, bootstrap_samples=0 if content_reuse else args.bootstrap_samples)
    if content_reuse:
        result["accuracy_ci"]["suppressed_reason"] = "Exact content is shared across declared groups; independent group resampling is not justified."
    clusters = evidence_clusters(records)
    result["cluster_accuracy_ci"] = cluster_bootstrap({r["id"]: _correct(r, predictions.get(r["id"])) for r in records},
                                                      clusters, samples=max(args.bootstrap_samples, 1))
    result["contrast_sets"] = contrast_metrics(records, predictions)
    result["dataset_status"] = "reviewed" if all(r["annotation_status"] == "reviewed" for r in records) else "DRAFT_DIAGNOSTIC_ONLY"
    result["split"] = args.split
    result["run_attribution"] = attribution
    result["dataset_audit"] = dataset_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
