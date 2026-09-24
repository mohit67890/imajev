"""Run fastino/fast-decisions (dev split) against an imajev server with one fixed, generic prompt template.

Single-label head -> one `choice` question (criteria = the full label list, null descriptions).
Multi-label head  -> one `noul` per label; predicted set = labels with noul > 0.5, or the top label if none.
Scoring: exact set match per head (the dataset card's rule). Resumable: appends to OUT.
"""
import json, sys, glob, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://<host>/v1/systemone"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "preds-imajev-4b.jsonl")


def human(task):
    return task.replace("_", " ")


def build(row):
    questions, plan = {}, []
    for i, h in enumerate(row["output"]["classifications"]):
        if h["multi_label"]:
            ids = []
            for j, lab in enumerate(h["labels"]):
                qid = f"h{i}_l{j}"
                questions[qid] = {"type": "noul",
                                  "instructions": f"Does the {human(h['task'])} label \"{lab}\" apply to this text?"}
                ids.append((qid, lab))
            plan.append((h, ids))
        else:
            qid = f"h{i}"
            questions[qid] = {"type": "choice",
                              "instructions": f"Which {human(h['task'])} label fits this text?",
                              "criteria": {lab: None for lab in h["labels"]}}
            plan.append((h, qid))
    return {"model": "imajev", "state": row["input"], "questions": questions}, plan


def post(req):
    data = json.dumps(req).encode()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.load(f)


def call(req, per_request=8):
    """The server takes at most 8 questions per request; split and merge answers."""
    items = list(req["questions"].items())
    answers = {}
    for k in range(0, len(items), per_request):
        answers.update(post({**req, "questions": dict(items[k:k + per_request])})["answers"])
    return {"answers": answers}


def main():
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["key"] for l in open(OUT)}
    files = sorted(glob.glob(os.path.join(HERE, "*.jsonl")))
    files = [f for f in files if not os.path.basename(f).startswith("preds")]
    with open(OUT, "a") as out:
        for f in files:
            dom = os.path.basename(f)[:-6]
            for n, line in enumerate(open(f)):
                key = f"{dom}:{n}"
                if key in done:
                    continue
                row = json.loads(line)
                req, plan = build(row)
                t = time.time()
                resp = call(req)["answers"]
                heads = []
                for h, q in plan:
                    if isinstance(q, list):
                        probs = {lab: resp[qid]["noul"] for qid, lab in q}
                        pred = [l for l, p in probs.items() if p > 0.5] or [max(probs, key=probs.get)]
                        detail = probs
                    else:
                        pred = [resp[q]["choice"]]
                        detail = {"confidence": resp[q]["confidence"], "unknown": resp[q]["unknown_probability"]}
                    heads.append({"task": h["task"], "multi_label": h["multi_label"], "gold": h["true_label"],
                                  "pred": pred, "correct": sorted(pred) == sorted(h["true_label"]), "detail": detail})
                out.write(json.dumps({"key": key, "domain": dom, "seconds": round(time.time() - t, 3),
                                      "heads": heads}) + "\n")
                out.flush()
            print(dom, "done", flush=True)


if __name__ == "__main__":
    main()
