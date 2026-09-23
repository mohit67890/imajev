"""Merge blind AI-auditor answers with constructed labels; list disagreements; apply adjudications.

  compare  Read auditor outputs (Claude part files and a Kimi api-run), normalise each answer to the item's
           typed domain, and write disagreements.json: every audited item where any auditor differs from the
           constructed answer (or flagged a problem). Nothing is changed.
  apply    Read adjudications.json ({id: {"verdict": "confirmed"|"corrected"|"excluded", "value": ..., "note": ...}})
           and write a new records file: confirmed items become reviewed on the construction route with a
           disclosed provenance.model_audit; corrected items (label errors) and excluded items (defective) are
           quarantined with their original gold kept, so existing runs still verify. Items missing an adjudication
           stay draft.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imajev_bench.cli import read_jsonl  # noqa: E402
from imajev_bench.runner import digest  # noqa: E402
from imajev_bench.schema import model_payload, validate_records  # noqa: E402

MISSING = object()


def normalise(field, answer):
    """Map an auditor's answer onto the field's typed domain; unknown -> None; unmappable -> MISSING."""
    if answer is None or (isinstance(answer, str) and answer.strip().lower() in ("unknown", "none", "null")):
        return None
    if field["type"] == "boolean":
        text = str(answer).strip().lower()
        return True if text in ("true", "yes") else False if text in ("false", "no") else MISSING
    if field["type"] == "ordinal":
        levels = [level["value"] for level in field["levels"]]
        try:
            value = int(str(answer).strip())
        except ValueError:
            return MISSING
        return value if value in levels else MISSING
    options = [o["value"] for o in field["options"]]
    text = str(answer).strip()
    return text if text in options else MISSING


def same(a, b):
    return a is not MISSING and type(a) is type(b) and a == b


def load_auditors(audit_dir: Path, records: dict):
    answers = {rid: {} for rid in records}
    notes = {rid: {} for rid in records}
    for part in sorted(audit_dir.glob("claude-part-*.json")):
        for row in json.loads(part.read_text()):
            if row["id"] in records:
                field = records[row["id"]]["request"]["fields"][0]
                answers[row["id"]]["claude-opus-5.5"] = normalise(field, row.get("answer"))
                notes[row["id"]]["claude-opus-5.5"] = {"evidence": row.get("evidence"), "flags": row.get("flags") or [],
                                                       "confidence": row.get("confidence")}
    for run in sorted(audit_dir.glob("kimi-*")):
        if not (run / "completion.json").exists():
            continue
        for row in read_jsonl(run / "predictions.jsonl"):
            if row["id"] in records:
                value = MISSING if row["status"] == "error" else row["value"] if row["status"] == "answered" else None
                answers[row["id"]]["kimi-k2.5"] = value
                notes[row["id"]]["kimi-k2.5"] = {"evidence": row.get("evidence"), "error": row.get("error_type")}
    return answers, notes


def cmd_compare(args):
    records = {r["id"]: r for r in read_jsonl(args.records) if r["provenance"].get("audit_sample")}
    answers, notes = load_auditors(args.audit_dir, records)
    rows, stats = [], {"audited": len(records), "all_agree": 0, "missing_answers": 0}
    for rid, record in records.items():
        truth = record["provenance"]["construction"]["truth"]
        got = answers[rid]
        flags = [f for n in notes[rid].values() for f in n.get("flags", [])]
        missing = [a for a in ("claude-opus-5.5", "kimi-k2.5") if a not in got or got[a] is MISSING]
        stats["missing_answers"] += bool(missing)
        agree = all(same(v, truth) for v in got.values()) and not missing
        stats["all_agree"] += agree and not flags
        if not agree or flags or record["provenance"].get("judgement_dependent"):
            rows.append({"id": rid, "constructed": truth, "answers": {k: (None if v is None else "UNMAPPED" if v is MISSING else v)
                                                                       for k, v in got.items()},
                         "missing": missing, "flags": flags, "notes": notes[rid],
                         "judgement_note": record["provenance"].get("judgement_note"),
                         "question": record["request"]["fields"][0]["question"], "state": record["request"]["state"],
                         "images": [i["path"] for i in record["images"]]})
    out = args.audit_dir / "disagreements.json"
    out.write_text(json.dumps({"stats": stats, "items": rows}, indent=1, default=str))
    print(json.dumps({**stats, "to_adjudicate": len(rows)}))


def cmd_apply(args):
    records = read_jsonl(args.records)
    decisions = json.loads(args.adjudications.read_text())
    audited = {r["id"]: r for r in records if r["provenance"].get("audit_sample")}
    answers, _ = load_auditors(args.audit_dir, audited)
    counts = {"confirmed": 0, "corrected": 0, "excluded": 0, "pending": 0}
    for rid, record in audited.items():
        truth = record["provenance"]["construction"]["truth"]
        blind = {k: (None if v is None else "UNMAPPED" if v is MISSING else v) for k, v in answers[rid].items()}
        decision = decisions.get(rid)
        if decision is None:
            if blind and all(same(v, truth) for v in answers[rid].values()) and len(answers[rid]) == 2:
                decision = {"verdict": "confirmed", "value": truth, "note": "both blind auditors agree with the constructed answer"}
            else:
                counts["pending"] += 1
                continue
        p = record["provenance"]
        audit = {"method": "model", "auditors": sorted(blind), "blind_answers": blind, "verdict": decision["verdict"],
                 "adjudicated_by": "claude-opus-5.5 (pipeline author's model)", "note": decision.get("note"),
                 "input_sha256": digest(model_payload(record))}
        if decision["verdict"] in ("confirmed", "corrected"):
            audit["adjudicated_value"] = decision["value"]
        p["model_audit"] = audit
        counts[decision["verdict"]] += 1
        if decision["verdict"] == "confirmed":
            record["annotation_status"], p["review_route"] = "reviewed", "construction_verified"
        elif decision["verdict"] == "corrected":
            # Keep gold unchanged so existing runs still verify; quarantine the item and record the correction.
            p["quarantined"] = True
            p["quarantine_reason"] = f"model audit found the constructed label wrong; audited value {decision['value']!r}"
            record["annotation_status"] = "draft"
        else:
            p["quarantined"] = True
            p["quarantine_reason"] = decision.get("note") or "excluded by model audit"
            record["annotation_status"] = "draft"
    validated = validate_records(records, args.records.parent)
    with args.output.open("x") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--records", type=Path, required=True)
        p.add_argument("--audit-dir", type=Path, required=True)
        if name == "apply":
            p.add_argument("--adjudications", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"compare": cmd_compare, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
