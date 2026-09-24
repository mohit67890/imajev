"""ImajevBench v2.0-lite test split through Jev-Omni's public predict() API (akhilaaa3/Jev-Omni, Gemma 4 12B).

Runs on the PUBLIC records (test gold withheld); scoring happens elsewhere against the gold file.
Mapping: boolean -> options ["Yes", "No"]; choice -> option values ("value: description" when a description exists);
ordinal -> "value: description" per level; state -> JSON string; the one image (if any) -> modality="image".
Jev-Omni has no abstain output, so every item is answered; `__unknown__` gets probability 0.
"""
import json, sys, time, importlib.util
from pathlib import Path

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SNAP = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("jev_omni", SNAP / "jev_omni.py"); jo = importlib.util.module_from_spec(spec); spec.loader.exec_module(jo)
records = [json.loads(l) for l in open(ROOT / "records/records-public.jsonl")]
records = [r for r in records if r["split"] == "test"]
t0 = time.time(); clf = jo.load_jev_omni(); load_s = time.time() - t0
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for r in records:
    f = r["request"]["fields"][0]; kind = f["type"]
    if kind == "boolean":
        labels = ["Yes", "No"]; values = [True, False]
        if f.get("yes_description") or f.get("no_description"):
            labels = [f"Yes: {f.get('yes_description') or ''}".rstrip(": "), f"No: {f.get('no_description') or ''}".rstrip(": ")]
    elif kind == "choice":
        values = [o["value"] for o in f["options"]]
        labels = [f"{o['value']}: {o['description']}" if o.get("description") else str(o["value"]) for o in f["options"]]
    else:
        values = [l["value"] for l in f["levels"]]
        labels = [f"{l['value']}: {l['description']}" if l.get("description") else str(l["value"]) for l in f["levels"]]
    state = r["request"]["state"]; state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    media = str(ROOT / r["images"][0]["path"]) if r["images"] else None
    t = time.time()
    res = clf.predict(state=state, question=f["question"], options=labels, media=media, modality="image" if media else "text")
    ms = (time.time() - t) * 1000
    key = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
    probs = {key(v): float(res["probabilities"][lab]) for v, lab in zip(values, labels)}; probs["__unknown__"] = 0.0
    rows.append({"id": r["id"], "status": "answered", "value": values[res["prediction_index"]], "probabilities": probs,
                 "confidence_source": "jev_omni_classifier_head", "latency_ms": ms})
with open(OUT / "predictions.jsonl", "w") as h:
    for row in rows: h.write(json.dumps(row) + "\n")
json.dump({"model": "akhilaaa3/Jev-Omni", "snapshot": SNAP.name, "base": jo.BASE_ID, "interface": "jev_omni.predict (public API)",
           "records": "records-public.jsonl (test gold withheld)", "n": len(rows), "load_seconds": load_s,
           "mapping": __doc__}, open(OUT / "manifest.json", "w"), indent=1)
print("done", len(rows), "items; load", round(load_s), "s")
