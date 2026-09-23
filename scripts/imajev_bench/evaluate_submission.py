"""Score a submitted model on the imajevBench team's hidden test set and write an aggregate-only report.

Two submission tiers (recorded in every report):
  A  weights/container: the team runs the model offline with run_local_v2.py (or any harness run folder) and
     passes the finished run with --run. Hidden items never leave the team's machines.
  B  API endpoint: this script sends the hidden items to the submitter's endpoint (--provider/--model, or an
     OpenAI-compatible --base-url with the key in --key-env). The endpoint owner can see those items, so the
     report marks them exposed and the policy rotates exposed items out (docs/imajev-bench-hidden-test.md).

Outputs:
  <private-dir>/<submission>/      raw predictions and receipts. Team only; never share.
  <public-dir>/<submission>.json/.md  aggregate metrics only: no item IDs, questions, images or answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imajev_bench.api_models import OpenAIProvider, make_provider, run_api  # noqa: E402
from imajev_bench.cli import read_jsonl  # noqa: E402
from imajev_bench.runner import verify_run  # noqa: E402
from imajev_bench.schema import validate_records  # noqa: E402
from imajev_bench.scoring import _correct  # noqa: E402
from imajev_bench.stats import cluster_bootstrap, contrast_metrics, evidence_clusters  # noqa: E402


def aggregate(records, predictions):
    """Metrics that reveal nothing item-level."""
    scored = [r for r in records if not r["provenance"].get("quarantined")]
    correct = {r["id"]: _correct(r, predictions.get(r["id"])) for r in scored}
    tracks = {t: [r for r in scored if r["track"] == t] for t in ("text", "visual", "joint")}
    unknown = [r for r in scored if r["gold"] is None]
    answerable = [r for r in scored if r["gold"] is not None]
    status = lambda r: (predictions.get(r["id"]) or {}).get("status")
    ci = cluster_bootstrap(correct, evidence_clusters(scored))
    contrast = contrast_metrics(scored, predictions)
    return {"items": len(scored), "accuracy": sum(correct.values()) / len(scored),
            "accuracy_ci95": [ci["low"], ci["high"]], "evidence_clusters": ci["clusters"],
            "tracks": {t: (sum(correct[r["id"]] for r in rs) / len(rs) if rs else None) for t, rs in tracks.items()},
            "correct_unknown_rate": sum(status(r) == "abstained" for r in unknown) / len(unknown) if unknown else None,
            "false_abstention_rate": sum(status(r) == "abstained" for r in answerable) / len(answerable),
            "error_rate": sum(status(r) == "error" for r in scored) / len(scored),
            "contrast_sets_all_correct": contrast["all_correct"] / contrast["sets"] if contrast["sets"] else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="Hidden set records (team only)")
    parser.add_argument("--submission", required=True, help="Short name, e.g. acme-vision-7b-2026-10")
    parser.add_argument("--submitter", required=True, help="Organisation or person submitting")
    parser.add_argument("--run", type=Path, help="Tier A: finished harness run folder for the hidden set")
    parser.add_argument("--provider", help="Tier B: azure-openai | openai | gemini | vertex-gemini")
    parser.add_argument("--model", help="Tier B: provider model or deployment id")
    parser.add_argument("--base-url", help="Tier B: OpenAI-compatible endpoint base URL (…/v1)")
    parser.add_argument("--key-env", help="Tier B: environment variable holding the endpoint key")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--free-json", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--notes", default="", help="Settings the submitter declared (precision, prompt, budget)")
    parser.add_argument("--private-dir", type=Path, default=ROOT / "reports/imajev-bench-hidden/private")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "reports/imajev-bench-hidden/public")
    args = parser.parse_args(argv)

    records = validate_records(read_jsonl(args.records), args.records.parent)
    records_sha = hashlib.sha256(args.records.read_bytes()).hexdigest()
    private = args.private_dir / args.submission
    if args.run:
        tier, run_dir = "A (weights, run offline by the team)", args.run
        verify_run(records, run_dir / "predictions.jsonl")
    else:
        tier = "B (API endpoint; hidden items exposed to the endpoint owner)"
        if args.base_url:
            import os
            provider = OpenAIProvider(args.model, os.environ[args.key_env], base_url=args.base_url.rstrip("/"),
                                      reasoning_effort=args.reasoning_effort)
        else:
            provider = make_provider(args.provider, args.model, reasoning_effort=args.reasoning_effort)
        run_dir = private / "run"
        private.mkdir(parents=True, exist_ok=False)
        run_api(records, args.records.parent, run_dir, provider, constrained=not args.free_json,
                allow_test_exposure=True, purpose="hidden-test submission", workers=args.workers)
    predictions = {row["id"]: row for row in read_jsonl(run_dir / "predictions.jsonl")}
    metrics = aggregate(records, predictions)
    report = {"submission": args.submission, "submitter": args.submitter, "tier": tier,
              "evaluated_at": datetime.now(timezone.utc).isoformat(), "hidden_set": args.records.parent.name,
              "hidden_set_records_sha256": records_sha, "declared_settings": args.notes,
              "reasoning_effort": args.reasoning_effort or "provider default", "metrics": metrics,
              "disclosure": "Aggregate metrics only. Item-level results are kept privately by the imajevBench team."}
    args.public_dir.mkdir(parents=True, exist_ok=True)
    (args.public_dir / f"{args.submission}.json").write_text(json.dumps(report, indent=1) + "\n")
    m = metrics
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"
    (args.public_dir / f"{args.submission}.md").write_text("\n".join([
        f"# imajevBench hidden-test result: {args.submission}", "",
        f"Submitter: {args.submitter}. Tier {tier}. Hidden set `{report['hidden_set']}` "
        f"(records sha256 {records_sha[:12]}…). Evaluated {report['evaluated_at'][:10]}.", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| Accuracy | {pct(m['accuracy'])} (95% CI {pct(m['accuracy_ci95'][0])}–{pct(m['accuracy_ci95'][1])}, "
        f"{m['evidence_clusters']} clusters, {m['items']} items) |",
        f"| Text / Visual / Joint | {pct(m['tracks']['text'])} / {pct(m['tracks']['visual'])} / {pct(m['tracks']['joint'])} |",
        f"| Correct Unknown | {pct(m['correct_unknown_rate'])} |", f"| False abstention | {pct(m['false_abstention_rate'])} |",
        f"| Contrast sets all correct | {pct(m['contrast_sets_all_correct'])} |", f"| Errors (unparseable, timeouts) | {pct(m['error_rate'])} |",
        "", f"Declared settings: {args.notes or 'none'}; reasoning: {report['reasoning_effort']}.", "",
        report["disclosure"], ""]))
    args.private_dir.mkdir(parents=True, exist_ok=True)
    with (args.private_dir / "submissions.jsonl").open("a") as log:
        log.write(json.dumps({**report, "private_run": str(run_dir)}) + "\n")
    print(json.dumps({"public_report": str(args.public_dir / f"{args.submission}.md"), **metrics}, default=str))


if __name__ == "__main__":
    main()
