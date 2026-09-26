"""Throughput / format test of the Azure-hosted Qwen3.6-35B-A3B teacher with our real distribution prompts."""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
REPO = ""
sys.path.insert(0, REPO + "/scripts/p2")
from gen_answer import DISTRIBUTION_SYSTEM, distribution_prompt, distribution_schema, parse_distribution, questions
from p2_common import read_jsonl
key = open(os.path.expanduser("~/.imajev-azure-key")).read().strip()
URL = "https://<azure-foundry-endpoint>/openai/v1/chat/completions"
MODEL = "qwen--qwen36-35b-a3b"
N, CONC, MAXTOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
OUT = sys.argv[4]
qs = list(questions(read_jsonl(__import__("pathlib").Path(REPO + "/data/decision-p2/teacher/writer-p2b-hard.jsonl"))))
random.Random(7).shuffle(qs); qs = qs[:N]

def work(q):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": MAXTOK,
            "messages": [{"role": "system", "content": DISTRIBUTION_SYSTEM}, {"role": "user", "content": distribution_prompt(q["state"], q["field"])}],
            "response_format": distribution_schema(q["field"])}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                      headers={"api-key": key, "content-type": "application/json"}), timeout=900))
        m = r["choices"][0]["message"]; raw = m.get("content") or ""
        value, probs, rationale, err = parse_distribution(raw, q["field"])
        return {"ok": True, "secs": time.time() - t, "usage": r.get("usage"), "finish": r["choices"][0].get("finish_reason"),
                "reason_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""), "parse_error": err, "value": value,
                "field_type": q["field"]["type"], "doc_id": q["doc_id"], "qi": q["qi"]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "secs": time.time() - t, "err": f"HTTP {e.code} " + e.read()[:200].decode(errors="ignore").replace(key, "<key>")}
    except Exception as e:
        return {"ok": False, "secs": time.time() - t, "err": type(e).__name__ + " " + str(e)[:150]}

t0 = time.time()
with ThreadPoolExecutor(CONC) as ex, open(OUT, "w") as f:
    for i, res in enumerate(ex.map(work, qs)):
        f.write(json.dumps(res) + "\n"); f.flush()
wall = time.time() - t0
rows = [json.loads(l) for l in open(OUT)]
ok = [r for r in rows if r["ok"]]
comp = [r["usage"]["completion_tokens"] for r in ok if r.get("usage")]
print(f"N={N} conc={CONC} max_tokens={MAXTOK} wall={wall:.0f}s ok={len(ok)}/{len(rows)}")
if comp:
    comp.sort()
    print(f"completion tokens: mean {sum(comp)/len(comp):.0f} median {comp[len(comp)//2]} p90 {comp[int(.9*len(comp))]} max {comp[-1]}")
    print(f"aggregate throughput: {sum(comp)/wall:.0f} output tok/s  |  items/hour: {3600*len(ok)/wall:.0f}")
    print(f"finish=length (truncated): {sum(r['finish']=='length' for r in ok)}  parse errors: {sum(1 for r in ok if r['parse_error'])}")
errs = [r["err"] for r in rows if not r["ok"]]
if errs: print("errors:", errs[:3])
