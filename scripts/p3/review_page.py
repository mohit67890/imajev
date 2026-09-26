"""Owner review page for the phase-3 Kimi review (docs/phase-3-plan.md, "Review: open-weight model + small human slice").

    .venv/bin/python scripts/p3/review_page.py            # http://127.0.0.1:8790

Queue (~100 items, built once and kept in data/p3/review/owner-queue.jsonl so resuming is stable):
every item where Kimi disagrees with our kept label (up to --n-disagree, spread across sources) + --n-random other reviewed items.

Anchoring: the page shows the state, question, options, OUR kept answer and the teacher's rationale. The owner picks
correct / wrong / ambiguous / bad question (+ optional note) FIRST; only the server's reply to that save carries Kimi's verdict.
The item endpoint never includes Kimi's verdict or why the item is in the queue until the owner has answered it.
Verdicts autosave to data/p3/review/owner.jsonl (append-only; the latest row per id wins; `after_reveal` marks changes made
after Kimi was shown). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimi_review import DATA, REVIEW_DIR, UNKNOWN, read_jsonl, render_state, write_jsonl  # noqa: E402

VERDICTS = ("correct", "wrong", "ambiguous", "bad_question")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


# ---------------------------------------------------------------- queue

def build_queue(kimi_rows: list[dict], n_disagree: int = 60, n_random: int = 40, seed: int = 29) -> list[dict]:
    """All disagreements up to n_disagree (round-robin across sources so one noisy source cannot fill the slice), then n_random
    drawn uniformly from the remaining reviewed items. The mixed list is shuffled so position does not reveal the reason."""
    rng = random.Random(seed)
    ok = {r["id"]: r for r in kimi_rows if r.get("kimi")}  # latest row per id
    rows = sorted(ok.values(), key=lambda r: r["id"])
    dis = [r for r in rows if r.get("disagree")]
    by_src = defaultdict(list)
    for r in dis:
        by_src[r["source"]].append(r)
    for v in by_src.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < n_disagree and any(by_src.values()):
        for s in sorted(by_src):
            if by_src[s] and len(picked) < n_disagree:
                picked.append(by_src[s].pop())
    chosen = {r["id"] for r in picked}
    rest = [r for r in rows if r["id"] not in chosen]
    rand = rng.sample(rest, min(n_random, len(rest)))
    queue = [{"id": r["id"], "why": "disagree"} for r in picked] + [{"id": r["id"], "why": "random"} for r in rand]
    rng.shuffle(queue)
    return queue


# ---------------------------------------------------------------- store

class Store:
    def __init__(self, sample_path, kimi_path, queue_path, owner_path, data_root=DATA, n_disagree=60, n_random=40, rebuild=False):
        self.items = {it["id"]: it for it in read_jsonl(sample_path)}
        self.kimi = {}
        for r in read_jsonl(kimi_path):
            if r.get("kimi"):
                self.kimi[r["id"]] = r
        queue_path = Path(queue_path)
        if rebuild or not queue_path.exists():
            write_jsonl(queue_path, build_queue(list(self.kimi.values()), n_disagree, n_random))
        self.queue = [q for q in read_jsonl(queue_path) if q["id"] in self.items and q["id"] in self.kimi]
        self.owner_path = Path(owner_path)
        self.owner = {}
        for r in read_jsonl(self.owner_path):
            self.owner[r["id"]] = r
        self.data_root = Path(data_root).resolve()
        self.lock = threading.Lock()

    def progress(self):
        done = sum(1 for q in self.queue if q["id"] in self.owner)
        nxt = next((i for i, q in enumerate(self.queue) if q["id"] not in self.owner), len(self.queue) - 1 if self.queue else 0)
        return {"total": len(self.queue), "done": done, "next": max(nxt, 0)}

    @staticmethod
    def _label_text(field, label):
        if label == UNKNOWN:
            return "unknown"
        if field["type"] == "choice":
            for o in field["options"]:
                if str(o["key"]) == label:
                    return o.get("text") or label
        if field["type"] == "score":
            for l in field["levels"]:
                if str(l["value"]) == label:
                    return f"{label} — {l.get('description', '')}"
        return {"true": "yes (true)", "false": "no (false)"}.get(label, label)

    def item_view(self, i: int) -> dict:
        """What the page may show BEFORE the owner answers: no Kimi verdict, no queue reason."""
        q = self.queue[i]
        it = self.items[q["id"]]
        f = it["field"]
        if f["type"] == "choice":
            opts = [{"key": str(o["key"]), "text": o.get("text") or str(o["key"]), "description": o.get("description") or ""} for o in f["options"]]
        elif f["type"] == "score":
            opts = [{"key": str(l["value"]), "text": str(l["value"]), "description": l.get("description", "")} for l in f["levels"]]
        else:
            opts = [{"key": "true", "text": "yes", "description": ""}, {"key": "false", "text": "no", "description": ""}]
        view = {"index": i, "id": it["id"], "source": it["source"], "family": it["family"], "kind": it["kind"], "type": f["type"],
                "state": render_state(it["state"]), "images": [f"/img?p={p}" for p in it["images"]], "question": f["question"],
                "options": opts, "our_label": it["our_label"], "our_text": self._label_text(f, it["our_label"]),
                "unknown_reason": it.get("unknown_reason"), "rationale": it.get("rationale") or "", "owner": None, "kimi": None}
        own = self.owner.get(it["id"])
        if own:  # already answered: safe to show everything
            view["owner"] = {"verdict": own["verdict"], "note": own.get("note", "")}
            view["kimi"] = self.kimi_view(it["id"])
        return view

    def kimi_view(self, item_id: str) -> dict:
        r = self.kimi[item_id]
        k = r["kimi"]
        f = self.items[item_id]["field"]
        return {"answer": k["answer"], "answer_text": self._label_text(f, k["answer"]), "confidence": k.get("confidence"),
                "issues": k.get("issues", []), "note": k.get("note", ""), "agree": r.get("agree"), "flags": r.get("flags", [])}

    def save(self, item_id: str, verdict: str, note: str) -> dict:
        if item_id not in {q["id"] for q in self.queue}:
            raise KeyError(item_id)
        if verdict not in VERDICTS:
            raise ValueError(verdict)
        with self.lock:
            prev = self.owner.get(item_id)
            row = {"id": item_id, "verdict": verdict, "note": note[:2000], "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "after_reveal": prev is not None, "first_verdict": prev["first_verdict"] if prev else verdict}
            self.owner_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.owner_path, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.owner[item_id] = row
        return {"kimi": self.kimi_view(item_id), "progress": self.progress()}

    def image_path(self, rel: str) -> Path | None:
        p = (self.data_root / rel).resolve()
        if self.data_root not in p.parents or not p.is_file() or p.suffix.lower() not in MIME:
            return None
        return p


# ---------------------------------------------------------------- http

def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/progress":
                return self._json(store.progress())
            if u.path == "/api/item":
                try:
                    i = int(qs.get("i", ["0"])[0])
                    return self._json(store.item_view(i))
                except (ValueError, IndexError):
                    return self._json({"error": "no such item"}, 404)
            if u.path == "/img":
                p = store.image_path(qs.get("p", [""])[0])
                if not p:
                    return self._json({"error": "not found"}, 404)
                return self._send(200, p.read_bytes(), MIME[p.suffix.lower()])
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/verdict":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                return self._json(store.save(body["id"], body["verdict"], body.get("note") or ""))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)

    return H


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label review</title>
<style>
:root{--bg:#fbfbfa;--fg:#1b1b1b;--muted:#6b6b6b;--line:#e2e2df;--panel:#ffffff;--soft:#f3f3f1;--sel:#1b1b1b;--selfg:#fbfbfa}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e6;--muted:#9a9a97;--line:#2e2e2c;--panel:#1d1d1c;--soft:#242423;--sel:#e8e8e6;--selfg:#161616}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 80px}
header{display:flex;align-items:baseline;gap:16px;margin-bottom:6px}
header h1{font-size:15px;font-weight:600;margin:0}
.meta{color:var(--muted);font-size:13px}
.bar{height:3px;background:var(--line);margin:8px 0 22px}.bar>div{height:100%;background:var(--fg);width:0}
h2{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:22px 0 8px}
.state{white-space:pre-wrap;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);border:1px solid var(--line);
 padding:14px;max-height:48vh;overflow:auto}
.imgs img{max-width:100%;border:1px solid var(--line);display:block;margin:0 0 10px;background:#fff}
.q{font-size:18px;line-height:1.45;margin:4px 0 8px}
ol.opts{margin:0;padding:0;list-style:none;border-top:1px solid var(--line)}
ol.opts li{padding:7px 0;border-bottom:1px solid var(--line);display:flex;gap:12px}
ol.opts li .k{color:var(--muted);font:12px ui-monospace,Menlo,monospace;min-width:9em}
ol.opts li.ours{font-weight:600}
ol.opts li.ours>span:last-child::after{content:"ours";font:11px ui-monospace,Menlo,monospace;border:1px solid var(--fg);padding:0 5px;margin-left:10px;vertical-align:1px}
.ours-box{border-left:3px solid var(--fg);padding:8px 12px;background:var(--soft)}
.ours-box .a{font-size:17px;font-weight:600}
.ours-box .r{color:var(--muted);margin-top:4px}
.verdicts{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
button{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);padding:8px 14px;cursor:pointer;border-radius:3px}
button:hover{border-color:var(--fg)}button.on{background:var(--sel);color:var(--selfg);border-color:var(--sel)}
button kbd{font:11px ui-monospace,Menlo,monospace;color:var(--muted);margin-right:6px}button.on kbd{color:var(--selfg)}
textarea{width:100%;font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);padding:8px;min-height:54px;margin-top:10px;border-radius:3px}
.kimi{display:none;border:1px solid var(--line);padding:12px 14px;background:var(--panel)}
.kimi.shown{display:block}
.kimi .a{font-weight:600}.kimi .n{color:var(--muted);margin-top:4px}
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:22px}
.hint{color:var(--muted);font-size:12px}
.saved{color:var(--muted);font-size:12px;min-height:1em}
</style></head><body><main>
<header><h1>Label review</h1><span class="meta" id="meta"></span><span class="meta" id="count" style="margin-left:auto"></span></header>
<div class="bar"><div id="bar"></div></div>
<div id="imgwrap"><h2>Images</h2><div class="imgs" id="imgs"></div></div>
<h2>State</h2><div class="state" id="state"></div>
<h2>Question</h2><div class="q" id="q"></div>
<ol class="opts" id="opts"></ol>
<h2>Our kept answer</h2><div class="ours-box"><div class="a" id="ours"></div><div class="r" id="rat"></div></div>
<h2>Your verdict</h2>
<div class="verdicts" id="verdicts">
 <button data-v="correct"><kbd>1</kbd>Correct</button>
 <button data-v="wrong"><kbd>2</kbd>Wrong answer</button>
 <button data-v="ambiguous"><kbd>3</kbd>Ambiguous</button>
 <button data-v="bad_question"><kbd>4</kbd>Bad question</button>
</div>
<textarea id="note" placeholder="Optional note (n to focus, Esc to leave)"></textarea>
<div class="saved" id="saved"></div>
<h2 id="kh" style="display:none">Kimi's verdict</h2>
<div class="kimi" id="kimi"></div>
<div class="nav"><button id="prev"><kbd>←</kbd>Previous</button>
<span class="hint">1–4 verdict · n note · ← → move · Enter next</span>
<button id="next">Next<kbd style="margin:0 0 0 6px">→</kbd></button></div>
</main>
<script>
let idx=0,total=0,cur=null;
const $=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function j(u,o){const r=await fetch(u,o);return r.json()}
function setBar(p){total=p.total;$("bar").style.width=(p.total?100*p.done/p.total:0)+"%";$("count").textContent=p.done+" / "+p.total+" reviewed"}
function showKimi(k){
 if(!k){$("kimi").className="kimi";$("kh").style.display="none";$("kimi").innerHTML="";return}
 $("kh").style.display="";$("kimi").className="kimi shown";
 const conf=k.confidence==null?"":" · confidence "+Math.round(100*k.confidence)+"%";
 const iss=(k.issues||[]).filter(x=>x!=="none");
 $("kimi").innerHTML='<div class="a">'+esc(k.answer_text)+(k.agree?" — agrees with ours":" — differs from ours")+'</div>'+
  '<div class="n">'+esc(conf.slice(3))+(iss.length?" · flags: "+esc(iss.join(", ")):"")+'</div>'+
  (k.note?'<div style="margin-top:6px">'+esc(k.note)+'</div>':"");
}
function markVerdict(v){document.querySelectorAll("#verdicts button").forEach(b=>b.classList.toggle("on",b.dataset.v===v))}
async function load(i){
 if(total&&(i<0||i>=total))return;
 cur=await j("/api/item?i="+i); if(cur.error)return; idx=i;
 $("meta").textContent="#"+(i+1)+" · source "+cur.source+" · "+cur.family+" · "+cur.type+(cur.kind!=="base"?" · "+cur.kind:"");
 $("imgwrap").style.display=cur.images.length?"":"none";
 $("imgs").innerHTML=cur.images.map(u=>'<img loading="lazy" src="'+esc(u)+'">').join("");
 $("state").textContent=cur.state;$("state").scrollTop=0;
 $("q").textContent=cur.question;
 $("opts").innerHTML=cur.options.map(o=>'<li class="'+(o.key===cur.our_label?"ours":"")+'"><span class="k">'+esc(o.key)+'</span><span>'+esc(o.text)+(o.description?' <span class="meta">— '+esc(o.description)+'</span>':"")+'</span></li>').join("");
 $("ours").textContent=cur.our_text+(cur.our_label==="unknown"&&cur.unknown_reason?"  ("+cur.unknown_reason.replace(/_/g," ")+")":"");
 $("rat").textContent=cur.rationale?"Teacher: "+cur.rationale:"";
 $("note").value=cur.owner?cur.owner.note:"";
 markVerdict(cur.owner?cur.owner.verdict:null);
 showKimi(cur.kimi);$("saved").textContent="";
 window.scrollTo(0,0);
}
async function verdict(v){
 if(!cur)return;
 const r=await j("/api/verdict",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({id:cur.id,verdict:v,note:$("note").value})});
 if(r.error){$("saved").textContent=r.error;return}
 cur.owner={verdict:v,note:$("note").value};cur.kimi=r.kimi;
 markVerdict(v);showKimi(r.kimi);setBar(r.progress);$("saved").textContent="Saved";
}
let noteTimer=null;
$("note").addEventListener("input",()=>{if(!cur||!cur.owner)return;clearTimeout(noteTimer);noteTimer=setTimeout(()=>verdict(cur.owner.verdict),700)});
document.querySelectorAll("#verdicts button").forEach(b=>b.onclick=()=>verdict(b.dataset.v));
$("prev").onclick=()=>load(idx-1);$("next").onclick=()=>load(idx+1);
document.addEventListener("keydown",e=>{
 if(e.target===$("note")){if(e.key==="Escape")$("note").blur();return}
 if(e.metaKey||e.ctrlKey||e.altKey)return;
 const m={"1":"correct","2":"wrong","3":"ambiguous","4":"bad_question"};
 if(m[e.key]){verdict(m[e.key]);e.preventDefault()}
 else if(e.key==="ArrowRight"||e.key==="Enter"){load(idx+1);e.preventDefault()}
 else if(e.key==="ArrowLeft"){load(idx-1);e.preventDefault()}
 else if(e.key==="n"){$("note").focus();e.preventDefault()}
});
(async()=>{const p=await j("/api/progress");setBar(p);if(p.total)load(p.next);else $("q").textContent="Nothing to review: run kimi_review.py review first."})();
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=str(REVIEW_DIR / "sample.jsonl"))
    ap.add_argument("--kimi", default=str(REVIEW_DIR / "kimi.jsonl"))
    ap.add_argument("--queue", default=str(REVIEW_DIR / "owner-queue.jsonl"))
    ap.add_argument("--owner", default=str(REVIEW_DIR / "owner.jsonl"))
    ap.add_argument("--data-root", default=str(DATA))
    ap.add_argument("--n-disagree", type=int, default=60); ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--rebuild-queue", action="store_true", help="redraw the queue (only before the owner has started)")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)
    store = Store(args.sample, args.kimi, args.queue, args.owner, args.data_root, args.n_disagree, args.n_random, args.rebuild_queue)
    p = store.progress()
    print(f"{p['done']}/{p['total']} reviewed · http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
