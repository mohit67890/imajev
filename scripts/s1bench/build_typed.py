#!/usr/bin/env python3
"""S1-Bench (WYRipple/S1-Bench, arXiv 2504.10368) -> a TYPED conversion for imajev: our derivative, never an S1-Bench score.

Each English item becomes one decision request (state = the S1-Bench question; one field):
  * yes/no gold                 -> boolean field, target true/false
  * instruction_following       -> choice: 4 candidate responses, the dataset's gold response + 3 written NON-compliant ones;
                                   question "Which response follows the instruction?"
  * everything else             -> choice: the gold answer + 3 written distractors; question "What is the correct answer?"
Distractors are written by Kimi-K2.5 (Azure, the phase-3 review model); a second blind pass answers each finished item and the row
is FLAGGED (kept out of the manifest) when the blind answer is not the gold, so a distractor that is also right does not survive.
Options are shuffled with a fixed seed. Output: data/manifests/s1bench-typed-en.jsonl (partition test) + data/s1bench/report.json.
    .venv/bin/python scripts/s1bench/build_typed.py [--limit N] [--lang en] [--workers 8]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / ".cache/external/s1bench/data/S1-Bench.csv"
OUT_DIR = REPO / "data/s1bench"; MAN = REPO / "data/manifests"
URL = "https://<azure-foundry-endpoint>/openai/v1/chat/completions"; DEPLOYMENT = "Kimi-K2-Thinking"
KEY = Path("~/.imajev-azure-key").expanduser().read_text().strip()
YES = {"yes", "yes.", "true", "true."}; NO = {"no", "no.", "false", "false."}

WRITE = """You are building a multiple-choice version of a simple question for an evaluation set.

Question: {q}
Correct answer: {a}

Write exactly 3 DISTRACTORS: short answers of the same kind and length as the correct answer that are clearly WRONG for this
question. Rules: each distractor must be unambiguously incorrect (a careful expert would never accept it); none may be a
paraphrase, superset or subset of the correct answer; if the question names alternatives (e.g. "positive or negative"), one
distractor is the other alternative; keep the style of the correct answer (numbers stay numbers, names stay names).
Output ONLY JSON: {{"distractors": ["...", "...", "..."]}}"""

WRITE_IF = """You are building a multiple-choice version of an instruction-following task for an evaluation set.

Instruction: {q}
A response that follows it correctly: {a}

Write exactly 3 responses that look like reasonable attempts but each VIOLATE the instruction in one clear, checkable way
(wrong count, wrong length, a forbidden word, wrong format, missing a required element, ...). Each must be plainly non-compliant,
similar in length and style to the correct response, and not a trivial copy of it.
Output ONLY JSON: {{"distractors": ["...", "...", "..."]}}"""

COMPLY = """Instruction: {q}

Candidate responses:
{opts}

For EACH candidate, decide strictly whether it satisfies the instruction exactly as written (count words, digits, letters and
punctuation literally; a response that adds extra words, characters or separators does not comply).
Output ONLY JSON: {{"compliant": ["<letters of every candidate that complies>"]}}"""

VERIFY = """Answer this multiple-choice question. Exactly one option is correct.

{q}

{opts}

Output ONLY JSON: {{"answer": "<the letter>"}}"""

def call(prompt: str, max_tokens: int = 8000, temperature: float = 0.3) -> str:
    body = {"model": DEPLOYMENT, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(6):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "api-key": KEY})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.loads(r.read()); ch = j["choices"][0]; content = ch["message"].get("content") or ""
                if content.strip(): return content
                # thinking model: an empty answer means the reasoning ate the budget; give it more room and try again
                body["max_tokens"] = min(32000, body["max_tokens"] * 2); continue
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504): time.sleep(3 * (attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError): time.sleep(3 * (attempt + 1))
    raise RuntimeError("Azure call failed after retries")

def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0)) if m else {}

def clean(a: str) -> str:
    a = a.strip()
    return a[:-1] if a.endswith(".") and a.count(".") == 1 and not re.search(r"\d\.$", a) else a

def slug(s: str, i: int) -> str:
    return f"o{i}_" + re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:28]

def gold_response(answer: str) -> str:
    """Instruction-following golds read 'Any <constraint> would work (for example: X)' or 'for example: X or Y': use the example X."""
    m = re.search(r"for example:\s*(.+?)\s*\)?\s*$", answer, flags=re.S)
    if not m: return clean(answer)
    ex = m.group(1).strip().rstrip(")").strip()
    ex = re.sub(r"(\w+)\(or ([^)]+)\)", r"\2", ex)                     # 'five(or 5)' -> '5'
    ex = re.split(r"\.?\s*or\s+(?=[A-Z0-9\"'])", ex)[0].strip()   # 'X.or Y.' / 'X or Y' -> X
    return ex.rstrip(".") if ex.count(".") <= 1 and not re.search(r"\d\.$", ex) else ex

def build(row: dict, cache: dict) -> dict:
    q = row["question"].strip(); iid = f"s1b-en-{int(row['ID']):04d}"
    a = gold_response(row["answer"]) if row["category"] == "instruction_following" else clean(row["answer"])
    base = {"id": iid, "source": "s1bench_typed", "source_split": "s1bench", "partition": "test", "family": row["category"],
            "sub_category": row["sub_category"], "license": {"spdx": "MIT", "note": "S1-Bench (WYRipple/S1-Bench, MIT), typed conversion by imajev"},
            "images": [], "s1bench_id": row["ID"], "s1bench_answer": row["answer"]}
    if a.lower() in YES | NO:
        req = {"schema_version": "1.0", "request_id": iid, "state": q, "fields": [{"id": "decision", "type": "boolean", "question": "Is the answer to the question above yes?"}]}
        return {**base, "request": req, "target": a.lower() in YES, "abstention_cause": None, "kind": "boolean"}
    key = hashlib.sha1((q + "||" + a).encode()).hexdigest()
    prompt = (WRITE_IF if row["category"] == "instruction_following" else WRITE).format(q=q, a=a)
    if key not in cache:
        cache[key] = {"raw": call(prompt)}
    ds = [clean(str(x)) for x in parse_json(cache[key]["raw"]).get("distractors", [])][:3]
    ds = [d for d in ds if d and d.lower() != a.lower()]
    if len(ds) < 3 and key + "|r" not in cache:
        cache[key + "|r"] = {"raw": call(prompt + "\nYou MUST give three distinct distractors.", temperature=0.8)}
    if len(ds) < 3:
        more = [clean(str(x)) for x in parse_json(cache[key + "|r"]["raw"]).get("distractors", [])]
        ds = list(dict.fromkeys(ds + [d for d in more if d and d.lower() != a.lower()]))[:3]
    if len(ds) < 3: return {**base, "flag": f"only {len(ds)} distractors"}
    rng = random.Random(key); opts = [a] + ds; rng.shuffle(opts)
    options = [{"value": slug(o, i), "description": o} for i, o in enumerate(opts)]
    target = options[opts.index(a)]["value"]
    qtext = "Which response follows the instruction?" if row["category"] == "instruction_following" else "What is the correct answer?"
    # blind verification: does an independent answerer pick the gold from these options?
    vkey = key + "|v"
    if vkey not in cache:
        letters = "ABCD"; shown = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        cache[vkey] = {"raw": call(VERIFY.format(q=q if row["category"] != "instruction_following" else "Instruction: " + q + "\nWhich response follows the instruction?", opts=shown), max_tokens=1500, temperature=0.0)}
    ans = str(parse_json(cache[vkey]["raw"]).get("answer", "")).strip().upper()[:1]
    picked = opts["ABCD".index(ans)] if ans in "ABCD" and ans else None
    if row["category"] == "instruction_following":
        ckey = key + "|c"
        if ckey not in cache:
            shown = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts))
            cache[ckey] = {"raw": call(COMPLY.format(q=q, opts=shown), max_tokens=3000, temperature=0.0)}
        comp = sorted(str(x).strip().upper()[:1] for x in parse_json(cache[ckey]["raw"]).get("compliant", []))
        if comp != ["ABCD"[opts.index(a)]]:
            picked = None; row["_comply"] = comp
    out = {**base, "request": {"schema_version": "1.0", "request_id": iid, "state": q, "fields": [{"id": "decision", "type": "choice", "question": qtext, "options": options}]},
           "target": target, "abstention_cause": None, "kind": "choice", "verifier_pick": picked}
    if picked != a: out["flag"] = f"verifier picked {picked!r}, gold {a!r}" + (f"; compliant {row['_comply']}" if row.get("_comply") else "")
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default="en"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["language"] == a.lang]
    if a.limit: rows = rows[: a.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True); cpath = OUT_DIR / f"cache-{a.lang}.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    with ThreadPoolExecutor(a.workers) as ex:
        items = list(ex.map(lambda r: build(r, cache), rows))
    cpath.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    keep = [x for x in items if "flag" not in x]; flagged = [x for x in items if "flag" in x]
    man = MAN / f"s1bench-typed-{a.lang}.jsonl"
    with man.open("w") as f:
        for x in keep: f.write(json.dumps({k: v for k, v in x.items() if k not in ("verifier_pick",)}, ensure_ascii=False) + "\n")
    (OUT_DIR / f"flagged-{a.lang}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in flagged))
    from collections import Counter
    rep = {"lang": a.lang, "source_rows": len(rows), "kept": len(keep), "flagged": len(flagged), "kinds": dict(Counter(x["kind"] for x in keep)),
           "by_category": dict(Counter(x["family"] for x in keep)), "flag_reasons": [x["flag"] for x in flagged], "manifest": str(man.relative_to(REPO)),
           "protocol": "typed conversion (choice: gold + 3 Kimi-written distractors, blind-verified; yes/no -> boolean); NOT comparable to S1-Bench paper scores"}
    (OUT_DIR / f"report-{a.lang}.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
