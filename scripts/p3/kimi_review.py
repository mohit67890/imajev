"""Phase-3 model review of kept teacher items by Kimi-K2.5 Thinking (docs/phase-3-plan.md, "Review: open-weight model + small
human slice").

    sample   stratified ~2,000 kept items (source x kind x family, ~300 image items) -> data/p3/review/sample.jsonl
    review   Kimi solves each sampled item independently (never shown our label) -> data/p3/review/kimi.jsonl
             resumable (done ids are skipped, errors are retried), hard cost cap (--max-usd, default 25)

    .venv/bin/python scripts/p3/kimi_review.py sample        # default: data/p3/teacher/kept.jsonl + data/p3/variants/constructed.jsonl
    .venv/bin/python scripts/p3/kimi_review.py review --max-usd 25 --workers 16

Endpoint: the owner's Azure Foundry account, OpenAI-compatible chat completions, `api-key` header read from ~/.imajev-azure-key,
model = deployment name (`Kimi-K2-Thinking`, which serves Kimi-K2.5). The key is never printed, logged or written anywhere:
error bodies are scrubbed and output rows hold only the parsed verdict, usage and cost.

Kept-row adapter (see `adapt`). The primary form is the manifest label contract (scripts/p3/assembly_manifest.py) as written
by azure_teacher.py `finalize` (kept.jsonl: {"id", "keep", "item": <candidate row>, "label": {target, probs, rationale,
target_kind, review}, ...}) and by variants.py (constructed.jsonl: a candidate row with "label"). Older forms are still read: a
candidate row with {kept_label, target_probs, rationale}, {"item"|"candidate": <candidate row>, ...} with those fields beside it,
or {"id": ..., kept_label, ...} resolved against data/p3/candidates-clean. Rows with "keep"/"kept": false are skipped. Targets
are in the candidate value space (option key | true/false | level int | null/"unknown").
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
REVIEW_DIR = DATA / "p3" / "review"
URL = "https://<azure-foundry-endpoint>/openai/v1/chat/completions"
DEPLOYMENT = "Kimi-K2-Thinking"
KEY_FILE = Path("~/.imajev-azure-key").expanduser()
# Azure Foundry GlobalStandard, USD per 1M tokens (Kimi-K2.5: $0.60 in / $3.00 out; the Kimi-K2-Thinking listing is $0.60 / $2.50,
# so 3.00 is the conservative choice). Reasoning tokens are billed inside completion_tokens.
PRICE_IN, PRICE_OUT = 0.60, 3.00
UNKNOWN = "unknown"
ISSUES = ("wrong_answer", "ambiguous", "badly_posed", "not_really_unknown", "none")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


# ---------------------------------------------------------------- io

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn last line after a crash; the item is simply redone
    return out


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- labels and items

def labels(field: dict) -> list[str]:
    """Answer labels for a field in string form, without unknown."""
    t = field["type"]
    if t == "choice":
        return [str(o["key"]) for o in field["options"]]
    if t == "score":
        return [str(l["value"]) for l in field["levels"]]
    return ["true", "false"]


def to_label(value) -> str:
    """Candidate value space -> string label ('unknown' for null)."""
    if value is None or value == UNKNOWN:
        return UNKNOWN
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def kind_of(item: dict) -> str:
    if item["our_label"] == UNKNOWN:
        return "unknown_variant" if item.get("parent_id") else "unknown"
    return "variant" if item.get("parent_id") else "base"


def stratum(item: dict) -> tuple:
    return (item["source"], item["kind"], item["family"])


def _candidate_index(candidates_dir: Path, ids: set) -> dict:
    found = {}
    for f in sorted(candidates_dir.glob("*.jsonl")):
        if f.name.startswith(("drops", "flagged")):
            continue
        with open(f) as fh:
            for line in fh:
                m = re.match(r'\{"id": "([^"]+)"', line)
                if m and m.group(1) in ids:
                    found[m.group(1)] = json.loads(line)
    return found


def adapt(row: dict, cand: dict | None = None) -> dict | None:
    """One kept teacher row (+ its candidate row when the teacher row only names the id) -> normalised review item, or None."""
    if row.get("kept") is False or row.get("keep") is False:
        return None
    base = row.get("item") or row.get("candidate") or (row if "field" in row else cand)
    if not base or "field" not in base:
        return None
    lab = row.get("label") if isinstance(row.get("label"), dict) else None
    if "kept_label" in row:
        kept = row["kept_label"]
    elif lab is not None:
        kept = lab.get("target")
    elif "label" in row:
        kept = row["label"]
    else:
        kept = base.get("gold")
    probs = (lab or {}).get("probs") or row.get("target_probs") or row.get("probs")
    if isinstance(probs, list):  # parallel list form
        probs = dict(zip(row.get("target_probs_keys") or [], probs))
    item = {"id": base["id"], "source": base["source"], "dataset": base.get("dataset"), "family": base.get("family") or base.get("dataset") or "?",
            "difficulty": base.get("difficulty"), "state": base["state"], "images": list(base.get("images") or []),
            "field": base["field"], "parent_id": base.get("parent_id"), "gold": base.get("gold"),
            "unknown_reason": base.get("unknown_reason"), "gold_kind": base.get("gold_kind"),
            "our_label": to_label(kept),
            "target_probs": {str(k): v for k, v in (probs or {}).items()} or None,
            "rationale": (lab or {}).get("rationale") or row.get("rationale") or row.get("teacher_rationale")
                         or next((r for r in row.get("rationales") or [] if r), None),
            "target_kind": (lab or {}).get("target_kind")}
    if item["our_label"] != UNKNOWN and item["our_label"] not in labels(item["field"]):
        return None
    item["kind"] = kind_of(item)
    item["image"] = bool(item["images"])
    return item


def load_kept(patterns: list[str], candidates_dir: Path) -> list[dict]:
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            rows.extend(read_jsonl(f))
    need = {r["id"] for r in rows if "field" not in r and not r.get("item") and not r.get("candidate") and "id" in r}
    cands = _candidate_index(candidates_dir, need) if need else {}
    items, seen = [], set()
    for r in rows:
        it = adapt(r, cands.get(r.get("id")))
        if it and it["id"] not in seen:
            seen.add(it["id"])
            items.append(it)
    return items


# ---------------------------------------------------------------- stratified sampler

def _allocate(sizes: dict, total: int, floor: int) -> dict:
    """Split `total` across strata: every stratum gets min(size, floor), the rest goes in proportion to sqrt(size) (so small
    strata are over-represented relative to their share, big ones are not starved), capped at the stratum size."""
    total = min(total, sum(sizes.values()))
    # the floor shrinks until every stratum's floor fits the budget (many strata, small budget); the rest is then spread as usual
    while floor > 1 and sum(min(n, floor) for n in sizes.values()) > total:
        floor -= 1
    alloc = {k: min(n, floor) for k, n in sizes.items()}
    if sum(alloc.values()) > total:  # more strata than budget: the largest strata first, one each
        alloc = {k: 0 for k in sizes}
        for k in sorted(sizes, key=lambda k: (-sizes[k], str(k)))[:total]:
            alloc[k] = 1
        return alloc
    left = total - sum(alloc.values())
    while left > 0:
        open_ = {k: math.sqrt(n) for k, n in sizes.items() if alloc[k] < n}
        if not open_:
            break
        z = sum(open_.values())
        share = {k: left * w / z for k, w in open_.items()}
        gave = 0
        for k in sorted(open_, key=lambda k: (-(share[k] - int(share[k])), str(k))):
            add = min(int(share[k]), sizes[k] - alloc[k])
            alloc[k] += add
            gave += add
        if gave == 0:  # hand out the remainder one by one, largest fractional share first
            for k in sorted(open_, key=lambda k: (-share[k], str(k))):
                if left - gave <= 0:
                    break
                if alloc[k] < sizes[k]:
                    alloc[k] += 1
                    gave += 1
        left -= gave
    return alloc


def stratified_sample(items: list[dict], n: int = 2000, n_image: int = 300, floor: int = 5, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    out = []
    img = [it for it in items if it["image"]]
    txt = [it for it in items if not it["image"]]
    n_img = min(n_image, len(img))
    n_txt = min(n - n_img, len(txt))
    if n_txt < n - n_img:  # not enough text: give the rest to images
        n_img = min(len(img), n - n_txt)
    for pool, k in ((img, n_img), (txt, n_txt)):
        groups = defaultdict(list)
        for it in pool:
            groups[stratum(it)].append(it)
        alloc = _allocate({g: len(v) for g, v in groups.items()}, k, floor)
        for g in sorted(groups):
            members = sorted(groups[g], key=lambda it: it["id"])
            out.extend(rng.sample(members, alloc[g]))
    out.sort(key=lambda it: it["id"])
    return out


# ---------------------------------------------------------------- prompt

SYSTEM = (
    "You check one typed decision question about a document (and any attached images). Solve it yourself, carefully and using only "
    "the document and images. The label \"unknown\" is correct when the document does not determine the answer, its premise is "
    "false, or no listed option is correct. Then judge the question itself and list every issue that applies:\n"
    "- wrong_answer: you are confident one specific label is correct and the question is fine (use this only with a concrete answer);\n"
    "- ambiguous: two or more labels are defensible from the document;\n"
    "- badly_posed: the question or options are malformed, contradictory, or do not fit the document;\n"
    "- not_really_unknown: the question looks built to be unanswerable, but the document does settle it;\n"
    "- none: no issue.\n"
    "Reply with one JSON object {\"answer\": <label>, \"confidence\": <number 0-1 that your answer is correct>, \"issues\": [...], "
    "\"note\": <one short sentence naming the deciding evidence or the problem>} and nothing else."
)


def render_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def label_lines(field: dict) -> list[str]:
    t = field["type"]
    if t == "choice":
        lines = []
        for o in field["options"]:
            desc = f" ({o['description']})" if o.get("description") else ""
            lines.append(f"- {o['key']}: {o.get('text', o['key'])}{desc}")
    elif t == "score":
        lines = [f"- {l['value']}: {l.get('description', '')}" for l in field["levels"]]
    else:
        lines = ["- true: yes", "- false: no"]
    return lines + [f"- {UNKNOWN}: the document does not determine the answer"]


def user_prompt(item: dict) -> str:
    img = f"\n\n({len(item['images'])} image(s) attached; they are part of the document.)" if item["images"] else ""
    return (f"DOCUMENT:\n{render_state(item['state'])}{img}\n\nQUESTION: {item['field']['question']}\n"
            f"Labels (answer with one, keyed exactly as written):\n" + "\n".join(label_lines(item["field"])) + "\nJSON only.")


def image_part(rel: str, data_root: Path = DATA) -> dict:
    p = (data_root / rel).resolve()
    mime = MIME.get(p.suffix.lower(), "image/png")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()}}


def verdict_schema(field: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "review_verdict", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["answer", "confidence", "issues", "note"],
        "properties": {"answer": {"type": "string", "enum": labels(field) + [UNKNOWN]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "issues": {"type": "array", "items": {"type": "string", "enum": list(ISSUES)}},
                       "note": {"type": "string"}}}}}


def build_body(item: dict, model: str, max_tokens: int, use_schema: bool, data_root: Path = DATA) -> dict:
    text = user_prompt(item)
    content = [{"type": "text", "text": text}] + [image_part(r, data_root) for r in item["images"]] if item["images"] else text
    body = {"model": model, "max_tokens": max_tokens, "temperature": 1.0,  # Moonshot's recommended temperature for thinking mode
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    if use_schema:
        body["response_format"] = verdict_schema(item["field"])
    return body


def _first_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def parse_verdict(raw: str, field: dict) -> tuple[dict | None, str | None]:
    try:
        obj = _first_json(raw)
    except Exception as exc:
        return None, f"json: {type(exc).__name__}"
    if not isinstance(obj, dict):
        return None, "not an object"
    ans = obj.get("answer")
    ans = to_label(ans) if not isinstance(ans, str) else ans.strip()
    if ans.lower() in ("true", "false", UNKNOWN):
        ans = ans.lower()
    if ans not in labels(field) + [UNKNOWN]:
        return None, f"answer {ans!r} not a label"
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    if conf is not None and conf > 1:
        conf = conf / 100.0  # the deployment ignores numeric bounds (probe returned 100)
    issues = [i for i in (obj.get("issues") or []) if i in ISSUES] or ["none"]
    if len(issues) > 1 and "none" in issues:
        issues = [i for i in issues if i != "none"]
    return {"answer": ans, "confidence": None if conf is None else max(0.0, min(1.0, conf)), "issues": issues,
            "note": str(obj.get("note") or "")[:400]}, None


def compare(item: dict, verdict: dict) -> dict:
    """Kimi vs our kept label. `disagree` = different answer, or any flagged issue on the question."""
    agree = verdict["answer"] == item["our_label"]
    issues = set(verdict["issues"]) - {"none"}
    if item["our_label"] == UNKNOWN and verdict["answer"] != UNKNOWN:
        issues.add("not_really_unknown")
    return {"agree": agree, "flags": sorted(issues), "disagree": (not agree) or bool(issues - {"wrong_answer"})}


# ---------------------------------------------------------------- client

def cost_usd(usage: dict | None, price_in: float, price_out: float) -> float:
    u = usage or {}
    return (u.get("prompt_tokens", 0) * price_in + u.get("completion_tokens", 0) * price_out) / 1e6


class Client:
    def __init__(self, url: str, key: str, timeout: float = 900):
        self.url, self._key, self.timeout = url, key, timeout

    def scrub(self, s: str) -> str:
        return s.replace(self._key, "<key>") if self._key else s

    def post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"api-key": self._key, "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)


class Budget:
    """Hard spend cap: a request is only sent while spent + reserved-for-in-flight + its own reservation stays under the cap."""

    def __init__(self, cap: float, spent: float = 0.0):
        self.cap, self.spent, self.reserved, self.lock = cap, spent, 0.0, threading.Lock()

    def reserve(self, amount: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + amount > self.cap:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved: float, actual: float):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


def review_one(client: Client, item: dict, args, state: dict) -> dict:
    row = {"id": item["id"], "source": item["source"], "family": item["family"], "kind": item["kind"], "image": item["image"],
           "our_label": item["our_label"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    last_err, parse_fails = None, 0
    for attempt in range(args.retries + 1):
        use_schema = state["schema"]
        body = build_body(item, args.model, args.max_tokens, use_schema, Path(args.data_root))
        try:
            r = client.post(body)
        except urllib.error.HTTPError as e:
            msg = client.scrub(e.read()[:300].decode(errors="ignore"))
            last_err = f"HTTP {e.code} {msg}"
            if e.code == 400 and use_schema and "response_format" in msg:
                state["schema"] = False  # fall back to strict-JSON prompt + parse for the rest of the run
                continue
            if e.code in (408, 429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            break
        except Exception as e:  # network / timeout
            last_err = client.scrub(f"{type(e).__name__} {str(e)[:150]}")
            time.sleep(min(60, 2 ** attempt * 3))
            continue
        u = r.get("usage") or {}
        for k in usage_total:
            usage_total[k] += int(u.get(k, 0) or 0)
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        verdict, err = parse_verdict(msg.get("content") or "", item["field"])
        row.update(model=r.get("model"), finish=ch.get("finish_reason"), schema=use_schema,
                   reasoning_chars=len(msg.get("reasoning_content") or msg.get("reasoning") or ""))
        if verdict:
            row.update(kimi=verdict, **compare(item, verdict), error=None)
            break
        last_err = f"parse: {err} (finish={ch.get('finish_reason')})"
        parse_fails += 1
        if parse_fails >= 2:  # a second unparseable answer will not improve; leave it for a later run
            break
    if "kimi" not in row:
        row["error"] = last_err or "failed"
    row["usage"] = usage_total
    row["cost_usd"] = round(cost_usd(usage_total, args.price_in, args.price_out), 6)
    return row


def estimate_reserve(item: dict, args) -> float:
    prompt_tok = (len(SYSTEM) + len(user_prompt(item))) / 3.0 + 1600 * len(item["images"])
    return (prompt_tok * args.price_in + args.max_tokens * args.price_out) / 1e6


def run_review(args, client: Client | None = None) -> dict:
    sample = read_jsonl(args.sample)
    if not sample:
        sys.exit(f"no sample at {args.sample}; run `sample` first")
    prev = read_jsonl(args.out)
    spent = sum(r.get("cost_usd", 0) for r in prev)
    done = {r["id"] for r in prev if r.get("kimi")}
    todo = [it for it in sample if it["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if client is None:
        key = Path(args.key_file).expanduser().read_text().strip()
        client = Client(args.url, key)
    budget = Budget(args.max_usd, spent)
    state = {"schema": not args.no_schema}
    lock = threading.Lock()
    stats = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stopped = False
    with open(args.out, "a") as fh, ThreadPoolExecutor(args.workers) as ex:
        pending, queue = {}, list(todo)

        def submit():
            nonlocal stopped
            while queue and len(pending) < args.workers and not stopped:
                it = queue[0]
                res = estimate_reserve(it, args)
                if not budget.reserve(res):
                    stopped = True
                    break
                queue.pop(0)
                pending[ex.submit(review_one, client, it, args, state)] = res

        submit()
        while pending:
            fin, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                res = pending.pop(f)
                row = f.result()
                budget.settle(res, row["cost_usd"])
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                stats["ok" if row.get("kimi") else "error"] += 1
                stats["disagree"] += bool(row.get("disagree"))
            submit()
            n = stats["ok"] + stats["error"]
            if n and n % 50 == 0:
                print(f"{n}/{len(todo)} ok={stats['ok']} err={stats['error']} disagree={stats['disagree']} spent=${budget.spent:.2f}", flush=True)
    summary = {"reviewed_now": stats["ok"], "errors_now": stats["error"], "disagree_now": stats["disagree"],
               "spent_usd_total": round(budget.spent, 4), "cap_usd": args.max_usd, "stopped_by_cap": stopped,
               "remaining": len(todo) - stats["ok"] - stats["error"], "schema_mode": state["schema"]}
    print(json.dumps(summary))
    return summary


# ---------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--teacher", nargs="+", default=[str(DATA / "p3/teacher/kept.jsonl"), str(DATA / "p3/variants/constructed.jsonl")],
                   help="glob(s) of kept rows (label contract): teacher kept.jsonl and constructed variants")
    s.add_argument("--candidates", default=str(DATA / "p3/candidates-clean"))
    s.add_argument("--out", default=str(REVIEW_DIR / "sample.jsonl"))
    s.add_argument("--n", type=int, default=2000); s.add_argument("--n-image", type=int, default=300)
    s.add_argument("--floor", type=int, default=5); s.add_argument("--seed", type=int, default=13)
    s.add_argument("--force", action="store_true", help="overwrite an existing sample (breaks resume of kimi.jsonl)")
    r = sub.add_parser("review")
    r.add_argument("--sample", default=str(REVIEW_DIR / "sample.jsonl"))
    r.add_argument("--out", default=str(REVIEW_DIR / "kimi.jsonl"))
    r.add_argument("--url", default=URL); r.add_argument("--model", default=DEPLOYMENT)
    r.add_argument("--key-file", default=str(KEY_FILE))
    r.add_argument("--data-root", default=str(DATA))
    r.add_argument("--max-usd", type=float, default=25.0)
    r.add_argument("--price-in", type=float, default=PRICE_IN, help="USD per 1M prompt tokens")
    r.add_argument("--price-out", type=float, default=PRICE_OUT, help="USD per 1M completion tokens (incl. reasoning)")
    r.add_argument("--max-tokens", type=int, default=12000)
    r.add_argument("--workers", type=int, default=16); r.add_argument("--retries", type=int, default=3)
    r.add_argument("--limit", type=int, default=0); r.add_argument("--no-schema", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        if Path(args.out).exists() and not args.force:
            print(f"{args.out} exists (use --force to redraw)"); return 0
        items = load_kept(args.teacher, Path(args.candidates))
        smp = stratified_sample(items, args.n, args.n_image, args.floor, args.seed)
        write_jsonl(args.out, smp)
        c = Counter((it["source"], it["kind"]) for it in smp)
        print(json.dumps({"kept": len(items), "sampled": len(smp), "image": sum(it["image"] for it in smp),
                          "by_source_kind": {f"{a}/{b}": n for (a, b), n in sorted(c.items())}}))
        return 0
    run_review(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
