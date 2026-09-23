"""Score a served model on the state-grounded PROBE set (human-derived gold, no teacher).

The probe is `data/manifests/decision-v2-state-probe.jsonl` (and, for the composited two-image
pairs, `data/manifests/decision-v2-pairs-probe.jsonl`).  Every gold answer there follows from
metadata a human verified or from an edit we made ourselves, so this script needs no model but the
one being measured.

It talks to the local playground, which serves the Jev request shape
(`scripts/playground/README.md`):

    POST http://127.0.0.1:<port>/v1/systemone
    {"state": ..., "questions": {"answer": {...}}, "images": ["data:image/jpeg;base64,..."]}

and reports:

  * accuracy per family and overall, counted against the human-derived gold;
  * the abstention rate, and how often abstaining was the right call;
  * for the paired families, the **answer-changed rate**: the share of pairs where the model's
    answer moved when only the STATE changed.  Two photographs identical, one question identical,
    two states -- a model that reads the state has to move, and a model that does not, cannot.

Run (the 2B v1.1 playground must already be serving on that port; this script never starts one):
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/probe.py --port 8765
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "decision-v2-state-probe.jsonl"
UNKNOWN = "unknown"


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def to_jev(field: dict) -> dict:
    """One internal field -> one Jev question (`noul` / `choice` / `score`)."""
    if field["type"] == "boolean":
        q = {"type": "noul", "instructions": field["question"]}
        criteria = {}
        if field.get("yes_description"):
            criteria["true"] = field["yes_description"]
        if field.get("no_description"):
            criteria["false"] = field["no_description"]
        if criteria:
            q["criteria"] = criteria
        return q
    if field["type"] == "choice":
        return {"type": "choice", "instructions": field["question"],
                "criteria": {o["value"]: o.get("description") for o in field["options"]}}
    return {"type": "score", "instructions": field["question"],
            "criteria": [level["description"] for level in field["levels"]]}


def predict(answer: dict, field: dict):
    """The model's answer in the same vocabulary as the gold, or `unknown` when it abstained."""
    if answer.get("abstained"):
        return UNKNOWN
    if answer["type"] == "noul":
        return bool(answer["noul"] > 0.5)
    if answer["type"] == "choice":
        return answer["choice"]
    probabilities = answer.get("probabilities") or {}
    if not probabilities:
        return int(round(answer["score"]))
    return int(max(probabilities, key=probabilities.get))


def ask(session, url, rowdata, timeout=180):
    payload = {
        "state": rowdata["request"]["state"],
        "questions": {"answer": to_jev(rowdata["request"]["fields"][0])},
        "images": [data_url(ROOT / im["image"]) for im in rowdata["images"]],
    }
    response = session.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code} {response.text[:300]}")
    return response.json()["answers"]["answer"]


def run(manifest: Path, port: int, limit: int | None, out: Path | None, label: str):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    url = f"http://127.0.0.1:{port}/v1/systemone"
    session = requests.Session()
    info = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=30).json()

    results = []
    started = time.time()
    for n, r in enumerate(rows, 1):
        field = r["request"]["fields"][0]
        answer = ask(session, url, r)
        got = predict(answer, field)
        gold = r["gold"]
        results.append({
            "id": r["id"], "family": r["probe_family"], "photo_source": r.get("photo_source"),
            "type": field["type"], "gold": gold, "predicted": got, "correct": got == gold,
            "abstained": bool(answer.get("abstained")),
            "unknown_probability": round(float(answer.get("unknown_probability", 0.0)), 4),
            "pair_id": r.get("pair_id"), "pair_role": r.get("pair_role"),
            "template_id": r.get("template_id"), "rationale": r.get("rationale"),
        })
        if n % 25 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    by_family = defaultdict(list)
    for x in results:
        by_family[x["family"]].append(x)
    families = {}
    for family, items in sorted(by_family.items()):
        correct = sum(x["correct"] for x in items)
        families[family] = {
            "n": len(items),
            "correct": correct,
            "accuracy": round(correct / len(items), 4),
            "abstained": sum(x["abstained"] for x in items),
            "abstained_correctly": sum(x["abstained"] and x["gold"] == UNKNOWN for x in items),
            "gold_unknown": sum(x["gold"] == UNKNOWN for x in items),
        }

    pairs = defaultdict(dict)
    for x in results:
        if x["pair_id"]:
            pairs[x["pair_id"]][x["pair_role"]] = x
    complete = [p for p in pairs.values() if len(p) == 2]
    changed = sum(1 for p in complete if p["a"]["predicted"] != p["b"]["predicted"])
    both = sum(1 for p in complete if p["a"]["correct"] and p["b"]["correct"])
    gold_differs = sum(1 for p in complete if p["a"]["gold"] != p["b"]["gold"])

    report = {
        "manifest": str(manifest.relative_to(ROOT)),
        "label": label,
        "served_model": info,
        "items": len(results),
        "overall_accuracy": round(sum(x["correct"] for x in results) / max(1, len(results)), 4),
        "abstention_rate": round(sum(x["abstained"] for x in results) / max(1, len(results)), 4),
        "families": families,
        "pairs": {
            "complete_pairs": len(complete),
            "pairs_whose_gold_differs": gold_differs,
            "answer_changed_when_state_changed": changed,
            "answer_changed_rate": round(changed / max(1, len(complete)), 4),
            "both_halves_correct": both,
            "both_halves_correct_rate": round(both / max(1, len(complete)), 4),
        },
        "confusion_by_gold": {
            str(g): dict(Counter(str(x["predicted"]) for x in results if str(x["gold"]) == g))
            for g in sorted({str(x["gold"]) for x in results})[:40]
        },
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_by_gold"}, indent=2))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"report": report, "results": results}, indent=2) + "\n")
        print(f"wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default="2b-v1.1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.manifest, args.port, args.limit, args.out, args.label)
