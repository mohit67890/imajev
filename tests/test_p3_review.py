"""Phase-3 review tooling: Kimi sampler/reviewer (mock endpoint), owner page anchoring, report maths."""
import argparse
import json
import sys
import threading
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/p3"))
import kimi_review as kr  # noqa: E402
import review_page as rp  # noqa: E402
import review_report as rr  # noqa: E402

SECRET = "sk-TEST-SENTINEL-9f8e7d6c5b4a"
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")  # header is enough for the tests


def cand(i, source="A", family="fam", parent=None, gold="b", images=(), ftype="choice"):
    field = {"type": "choice", "question": f"Which option fits item {i}?",
             "options": [{"key": "a", "text": "Alpha"}, {"key": "b", "text": "Beta"}, {"key": "c", "text": "Gamma"}]}
    if ftype == "noul":
        field = {"type": "noul", "question": f"Is item {i} allowed?"}
    return {"id": f"it-{i:05d}", "source": source, "dataset": "ds", "family": family, "difficulty": 3, "state": f"State text {i}.",
            "images": list(images), "field": field, "gold": gold, "unknown_reason": None if gold is not None else "insufficient_evidence",
            "gold_kind": "constructed", "parent_id": parent, "provenance": {"licence": "generated"}}


def kept(c, label="__gold__", rationale="Because the doc says so."):
    r = dict(c)
    r["kept_label"] = c["gold"] if label == "__gold__" else label
    r["target_probs"] = {"a": 0.05, "b": 0.9, "c": 0.03, "unknown": 0.02}
    r["rationale"] = rationale
    return r


def pool():
    rows, i = [], 0
    for src, fams in {"A": ["f1", "f2", "f3"], "B": ["g1", "g2"], "C": ["h1"], "D": ["t1"], "I": ["img1", "img2"]}.items():
        for fam in fams:
            n = {"f1": 400, "g1": 300}.get(fam, 60)
            for k in range(n):
                i += 1
                img = ("p3/images/gui/x.png",) if src == "I" or (src == "C" and k % 3 == 0) else ()
                parent = "it-00001" if k % 5 == 0 else None
                gold = None if k % 7 == 0 else "b"
                rows.append(kept(cand(i, src, fam, parent, gold, img)))
    return rows


# ---------------------------------------------------------------- adapter + sampler

def test_adapter_forms(tmp_path):
    c = cand(1)
    a = kr.adapt(kept(c))
    assert a["our_label"] == "b" and a["kind"] == "base" and a["rationale"]
    nested = kr.adapt({"item": c, "kept_label": None, "target_probs": {"unknown": 1.0}, "rationale": "r"})
    assert nested["our_label"] == "unknown" and nested["kind"] == "unknown"
    by_id = kr.adapt({"id": c["id"], "kept_label": "a"}, c)
    assert by_id["our_label"] == "a"
    assert kr.adapt({**kept(c), "kept": False}) is None
    assert kr.adapt(kept(c, label="zzz")) is None  # label not in the field
    noul = kr.adapt(kept(cand(2, ftype="noul", gold=True)))
    assert noul["our_label"] == "true"
    var = kr.adapt(kept(cand(3, parent="p", gold=None)))
    assert var["kind"] == "unknown_variant"
    # load_kept resolves id-only rows against the candidate files
    (tmp_path / "cands").mkdir()
    (tmp_path / "cands" / "A-x.jsonl").write_text(json.dumps(c) + "\n")
    (tmp_path / "t.jsonl").write_text(json.dumps({"id": c["id"], "kept_label": "c", "rationale": "x"}) + "\n")
    items = kr.load_kept([str(tmp_path / "t.jsonl")], tmp_path / "cands")
    assert [it["our_label"] for it in items] == ["c"]


def test_sampler_stratification():
    items = [kr.adapt(r) for r in pool()]
    s1 = kr.stratified_sample(items, n=300, n_image=40, floor=3, seed=1)
    s2 = kr.stratified_sample(items, n=300, n_image=40, floor=3, seed=1)
    assert [x["id"] for x in s1] == [x["id"] for x in s2]
    assert len(s1) == 300 and len({x["id"] for x in s1}) == 300
    assert sum(x["image"] for x in s1) == 40
    strata = Counter(kr.stratum(x) for x in items)
    got = Counter(kr.stratum(x) for x in s1)
    assert set(got) == set(strata)  # every stratum represented
    assert all(got[k] >= min(3, n) for k, n in strata.items())
    # sqrt allocation: the biggest text stratum is under-represented relative to its pool share, not starved
    big = ("A", "base", "f1")
    txt_pool = sum(n for k, n in strata.items() if not any(x["image"] for x in items if kr.stratum(x) == k))
    assert 0 < got[big] / 260 < strata[big] / txt_pool
    assert {x["kind"] for x in s1} >= {"base", "variant", "unknown", "unknown_variant"}


def test_allocate_exact_total_and_caps():
    a = kr._allocate({"x": 1000, "y": 10, "z": 2}, 100, 5)
    assert sum(a.values()) == 100 and a["z"] == 2 and a["y"] >= 5
    assert sum(kr._allocate({"x": 3, "y": 2}, 100, 5).values()) == 5


# ---------------------------------------------------------------- reviewer against a mock endpoint

class Mock:
    def __init__(self, mode="ok"):
        self.mode, self.bodies, self.keys = mode, [], []
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                mock.bodies.append(body)
                mock.keys.append(self.headers.get("api-key"))
                if mock.mode == "no_schema" and "response_format" in body:
                    return self._reply(400, {"error": {"message": "response_format json_schema not supported"}})
                if mock.mode == "leak":  # a hostile error body that echoes the key back
                    return self._reply(401, {"error": f"bad key {self.headers.get('api-key')}"})
                user = body["messages"][1]["content"]
                text = user if isinstance(user, str) else user[0]["text"]
                n = int(text.split("item ")[1].split("?")[0]) if "item " in text else 0
                ans = "b" if n % 4 else "a"  # every 4th item disagrees
                content = json.dumps({"answer": ans, "confidence": 90, "issues": ["none"], "note": "n"})
                self._reply(200, {"model": "Kimi-K2.5", "choices": [{"message": {"content": content, "reasoning_content": "think"},
                                                                     "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1000, "completion_tokens": 2000}})

            def _reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1/chat/completions"


def _args(tmp_path, url, **kw):
    a = dict(sample=str(tmp_path / "sample.jsonl"), out=str(tmp_path / "kimi.jsonl"), url=url, model="Kimi-K2-Thinking",
             key_file=str(tmp_path / "key"), data_root=str(tmp_path / "data"), max_usd=25.0, price_in=0.6, price_out=3.0,
             max_tokens=4000, workers=4, retries=1, limit=0, no_schema=False)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "key").write_text(SECRET + "\n")
    (tmp_path / "data/p3/images/gui").mkdir(parents=True)
    (tmp_path / "data/p3/images/gui/x.png").write_bytes(PNG)
    items = [kr.adapt(kept(cand(i, "AB"[i % 2], images=("p3/images/gui/x.png",) if i % 5 == 0 else ()))) for i in range(1, 21)]
    kr.write_jsonl(tmp_path / "sample.jsonl", items)
    return tmp_path, items


def test_review_mock_endpoint_resume_and_no_key_leak(setup, capsys):
    tmp_path, items = setup
    m = Mock()
    s = kr.run_review(_args(tmp_path, m.url))
    assert s["reviewed_now"] == 20 and s["errors_now"] == 0
    assert all(k == SECRET for k in m.keys)
    rows = kr.read_jsonl(tmp_path / "kimi.jsonl")
    assert sum(r["disagree"] for r in rows) == 5  # items 4, 8, 12, 16, 20
    assert all(r["kimi"]["confidence"] == 0.9 for r in rows)  # 90 -> 0.9
    assert abs(rows[0]["cost_usd"] - (1000 * 0.6 + 2000 * 3.0) / 1e6) < 1e-9
    # never shown our label, rationale or target: the prompt carries only state, question and labels
    for b in m.bodies:
        blob = json.dumps(b)
        assert "Because the doc says so" not in blob and "kept" not in blob and "0.9" not in blob
    img_bodies = [b for b in m.bodies if isinstance(b["messages"][1]["content"], list)]
    assert len(img_bodies) == 4 and img_bodies[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert all(b["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["a", "b", "c", "unknown"] for b in m.bodies)
    # resume: nothing is re-sent
    n = len(m.bodies)
    s2 = kr.run_review(_args(tmp_path, m.url))
    assert s2["reviewed_now"] == 0 and len(m.bodies) == n
    out = capsys.readouterr().out
    assert SECRET not in out
    assert SECRET not in (tmp_path / "kimi.jsonl").read_text()


def test_key_scrubbed_from_error_bodies(setup, capsys):
    tmp_path, _ = setup
    m = Mock("leak")
    s = kr.run_review(_args(tmp_path, m.url, limit=3))
    assert s["errors_now"] == 3
    text = (tmp_path / "kimi.jsonl").read_text()
    assert "401" in text and "<key>" in text and SECRET not in text
    assert SECRET not in capsys.readouterr().out


def test_cost_cap_stops_dispatch(setup):
    tmp_path, _ = setup
    m = Mock()
    # reservation per item ~ max_tokens * 3/1e6 = $0.012; a $0.03 cap admits at most 2 at a time and stops early
    s = kr.run_review(_args(tmp_path, m.url, max_usd=0.03, workers=1))
    assert s["stopped_by_cap"] and s["spent_usd_total"] <= 0.03
    assert 0 < s["reviewed_now"] < 20


def test_schema_fallback(setup):
    tmp_path, _ = setup
    m = Mock("no_schema")
    s = kr.run_review(_args(tmp_path, m.url, limit=3, workers=1))
    assert s["reviewed_now"] == 3 and s["schema_mode"] is False
    assert "response_format" in m.bodies[0] and "response_format" not in m.bodies[-1]


def test_parse_verdict_and_compare():
    f = cand(1)["field"]
    v, err = kr.parse_verdict('<think>x</think>```json\n{"answer":"unknown","confidence":0.4,"issues":["none","ambiguous"],"note":"n"}\n```', f)
    assert err is None and v["answer"] == "unknown" and v["issues"] == ["ambiguous"]
    assert kr.parse_verdict('{"answer":"zz"}', f)[0] is None
    it = kr.adapt(kept(cand(2, gold=None)))
    c = kr.compare(it, {"answer": "b", "issues": ["none"]})
    assert c["disagree"] and "not_really_unknown" in c["flags"]
    it2 = kr.adapt(kept(cand(3)))
    assert kr.compare(it2, {"answer": "b", "issues": ["ambiguous"]})["disagree"]
    assert not kr.compare(it2, {"answer": "b", "issues": ["none"]})["disagree"]


# ---------------------------------------------------------------- owner page

def _kimi_rows(items, disagree_every=4):
    rows = []
    for k, it in enumerate(items, 1):
        dis = k % disagree_every == 0
        rows.append({"id": it["id"], "source": it["source"], "family": it["family"], "kind": it["kind"], "image": it["image"],
                     "our_label": it["our_label"], "kimi": {"answer": "a" if dis else it["our_label"], "confidence": 0.8,
                                                            "issues": ["none"], "note": "KIMI-NOTE"},
                     "agree": not dis, "disagree": dis, "flags": [], "cost_usd": 0.01})
    return rows


def test_queue_composition():
    items = [kr.adapt(kept(cand(i, "ABC"[i % 3]))) for i in range(1, 401)]
    q = rp.build_queue(_kimi_rows(items), 60, 40)
    why = Counter(x["why"] for x in q)
    assert why == {"disagree": 60, "random": 40} and len({x["id"] for x in q}) == 100
    dis_ids = {x["id"] for x in q if x["why"] == "disagree"}
    srcs = Counter(next(it["source"] for it in items if it["id"] == i) for i in dis_ids)
    assert max(srcs.values()) - min(srcs.values()) <= 1  # round-robin across sources


def test_page_anchoring_order(setup):
    tmp_path, items = setup
    kr.write_jsonl(tmp_path / "kimi.jsonl", _kimi_rows(items))
    store = rp.Store(tmp_path / "sample.jsonl", tmp_path / "kimi.jsonl", tmp_path / "queue.jsonl", tmp_path / "owner.jsonl",
                     tmp_path / "data", 5, 5)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rp.make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: json.load(urllib.request.urlopen(base + p))

    html = urllib.request.urlopen(base + "/").read().decode()
    assert html.index('id="verdicts"') < html.index('id="kimi"') and ".kimi{display:none" in html
    view = get("/api/item?i=0")
    raw = json.dumps(view)
    assert view["kimi"] is None and "KIMI-NOTE" not in raw and "disagree" not in raw and "random" not in raw
    assert view["our_label"] == "b" and view["rationale"]
    req = urllib.request.Request(base + "/api/verdict", data=json.dumps({"id": view["id"], "verdict": "wrong", "note": "hm"}).encode(),
                                 headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    assert r["kimi"]["note"] == "KIMI-NOTE" and r["progress"]["done"] == 1
    assert get("/api/item?i=0")["kimi"]["note"] == "KIMI-NOTE"  # revealed only after answering
    assert get("/api/progress")["next"] == 1
    # a change after the reveal is recorded as such and keeps the first verdict
    req2 = urllib.request.Request(base + "/api/verdict", data=json.dumps({"id": view["id"], "verdict": "correct"}).encode(),
                                  headers={"content-type": "application/json"})
    urllib.request.urlopen(req2)
    last = kr.read_jsonl(tmp_path / "owner.jsonl")[-1]
    assert last["after_reveal"] and last["first_verdict"] == "wrong" and last["verdict"] == "correct"
    # resume: a fresh store picks up where the owner stopped
    store2 = rp.Store(tmp_path / "sample.jsonl", tmp_path / "kimi.jsonl", tmp_path / "queue.jsonl", tmp_path / "owner.jsonl", tmp_path / "data")
    assert store2.progress()["done"] == 1 and [q["id"] for q in store2.queue] == [q["id"] for q in store.queue]
    # images are served from data/ only
    img = [i for i, q in enumerate(store.queue) if store.items[q["id"]]["images"]]
    if img:
        assert urllib.request.urlopen(base + get(f"/api/item?i={img[0]}")["images"][0]).read() == PNG
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(base + "/img?p=../key")
    srv.shutdown()


# ---------------------------------------------------------------- report maths

def test_report_math(tmp_path):
    # source A: 100 items, 10 disagreements; owner checks 5 of them, confirms 2 -> p = 0.4, est = 10*0.4/100 = 4% (ok)
    # source B: 40 items, 8 disagreements; owner checks 4, confirms 3 -> p = 0.75, est = 8*0.75/40 = 15% (FLAGGED)
    # source C: 20 items, 2 disagreements, owner checks 1 (confirms 1) -> < 3 checked, pooled p = 6/10 = 0.6 -> est 6%
    # random items Kimi agreed with: 10 checked, 1 error -> miss rate 10%
    sample, kimi, queue, owner = [], [], [], []
    spec = {"A": (100, 10, 5, 2), "B": (40, 8, 4, 3), "C": (20, 2, 1, 1)}
    n = 0
    for src, (tot, dis, chk, conf) in spec.items():
        for k in range(tot):
            n += 1
            it = kr.adapt(kept(cand(n, src)))
            sample.append(it)
            d = k < dis
            kimi.append({"id": it["id"], "kimi": {"answer": "a" if d else "b", "issues": ["none"]}, "agree": not d, "disagree": d, "cost_usd": 0.01})
            if d and k < chk:
                queue.append({"id": it["id"], "why": "disagree"})
                owner.append({"id": it["id"], "verdict": "wrong" if k < conf else "correct"})
    rand = [s for s in sample if s["source"] == "A"][50:60]
    for j, it in enumerate(rand):
        queue.append({"id": it["id"], "why": "random"})
        owner.append({"id": it["id"], "verdict": "ambiguous" if j == 0 else "correct"})
    res = rr.compute(sample, kimi, queue, owner)
    A, B, C = (res["by_source"][s] for s in "ABC")
    assert res["pooled_precision"] == pytest.approx(6 / 10)
    assert res["miss_rate"] == pytest.approx(0.1)
    assert A["precision"] == pytest.approx(0.4) and A["est_rate"] == pytest.approx(0.04) and A["status"] == "watch"
    assert A["est_rate_with_misses"] == pytest.approx((10 * 0.4 + 90 * 0.1) / 100)
    assert A["confirmed_floor"] == pytest.approx(3 / 100)  # 2 confirmed disagreements + 1 random error
    assert B["est_rate"] == pytest.approx(0.15) and B["status"] == "FLAGGED"
    assert C["precision_from"] == "pooled" and C["est_rate"] == pytest.approx(2 * 0.6 / 20)
    dev = rr.dev_slice(sample, kimi, owner)
    by = Counter(r["verified_by"] for r in dev)
    owner_correct = sum(1 for o in owner if o["verdict"] == "correct")
    assert by["owner"] == owner_correct
    assert by["kimi"] == (160 - 20) - 10  # Kimi-agreed minus the 10 owner-reviewed random ones
    assert not any(r["id"] == o["id"] for r in dev for o in owner if o["verdict"] != "correct")
    md = rr.render(res, len(dev), by["owner"])
    assert "Sources over 5% estimated errors: B" in md
    # CLI end to end writes both outputs and no key
    for name, rows in (("sample", sample), ("kimi", kimi), ("queue", queue), ("owner", owner)):
        kr.write_jsonl(tmp_path / f"{name}.jsonl", rows)
    rr.main(["--sample", str(tmp_path / "sample.jsonl"), "--kimi", str(tmp_path / "kimi.jsonl"), "--queue", str(tmp_path / "queue.jsonl"),
             "--owner", str(tmp_path / "owner.jsonl"), "--report", str(tmp_path / "review.md"), "--dev-out", str(tmp_path / "dev.jsonl")])
    assert (tmp_path / "review.md").exists() and len(kr.read_jsonl(tmp_path / "dev.jsonl")) == len(dev)


# ---------------------------------------------------------------- integration with the teacher's label contract

def test_allocate_fills_the_budget_when_strata_times_floor_exceeds_it():
    sizes = {f"s{i}": 10 for i in range(83)}
    alloc = kr._allocate(sizes, 120, 5)
    assert sum(alloc.values()) == 120 and min(alloc.values()) >= 1
    assert sum(kr._allocate({f"s{i}": 3 for i in range(900)}, 2000, 5).values()) == 2000


def test_adapt_reads_teacher_kept_rows_and_constructed_variants():
    c = cand(1, gold=None)
    teacher_row = {"id": c["id"], "keep": True, "origin": "mine", "rationales": ["r1", None], "item": c,
                   "label": {"target": None, "probs": {"a": 0.1, "b": 0.0, "c": 0.0, "unknown": 0.9}, "rationale": "Nothing says.",
                             "target_kind": "teacher", "review": None}}
    it = kr.adapt(teacher_row)
    assert it["our_label"] == "unknown" and it["target_probs"]["unknown"] == 0.9 and it["rationale"] == "Nothing says."
    assert it["target_kind"] == "teacher"
    assert kr.adapt(dict(teacher_row, keep=False)) is None
    var = dict(cand(2, parent="it-00001"), label={"target": "c", "probs": None, "rationale": None, "target_kind": "constructed",
                                                  "review": None})
    v = kr.adapt(var)
    assert v["our_label"] == "c" and v["kind"] == "variant" and v["target_kind"] == "constructed"


def test_dev_slice_rows_carry_a_label_and_drop_ids_cover_errors():
    items = [kr.adapt(kept(cand(i))) for i in range(1, 7)]
    kimi = [{"id": it["id"], "kimi": {"answer": "b"}, "agree": True, "disagree": False} for it in items[:4]]
    kimi += [{"id": items[4]["id"], "kimi": {"answer": "a"}, "agree": False, "disagree": True},
             {"id": items[5]["id"], "kimi": {"answer": "a"}, "agree": False, "disagree": True}]
    owner = [{"id": items[0]["id"], "verdict": "correct"}, {"id": items[1]["id"], "verdict": "wrong"},
             {"id": items[4]["id"], "verdict": "correct"}]
    dev = {r["id"]: r for r in rr.dev_slice(items, kimi, owner)}
    assert set(dev) == {items[0]["id"], items[2]["id"], items[3]["id"], items[4]["id"]}
    lab = dev[items[0]["id"]]["label"]
    assert lab["target"] == "b" and lab["review"] == {"verdict": "correct", "verified_by": "owner"} and lab["probs"]
    assert dev[items[2]["id"]]["label"]["review"]["verified_by"] == "kimi"
    drops = rr.drop_ids(items, kimi, owner)
    assert drops["owner_errors"] == [items[1]["id"]] and drops["kimi_disputed_unchecked"] == [items[5]["id"]]


def test_dev_slice_caps_kimi_part_and_keeps_owner():
    import review_report as rr
    sample = [{"id": f"i{n}", "source": "A" if n % 2 else "B", "dataset": "d", "family": f"f{n % 3}", "difficulty": 3, "state": "s",
               "images": [], "field": {}, "parent_id": None, "gold": "a", "unknown_reason": None, "gold_kind": "constructed",
               "our_label": "a", "target_probs": None, "kind": "base"} for n in range(50)]
    kimi = [{"id": f"i{n}", "kimi": {"answer": "a"}, "agree": True, "disagree": False} for n in range(50)]
    owner = [{"id": "i1", "verdict": "correct"}, {"id": "i2", "verdict": "correct"}]
    dev = rr.dev_slice(sample, kimi, owner, max_kimi=10)
    assert sum(r["verified_by"] == "owner" for r in dev) == 2
    assert sum(r["verified_by"] == "kimi" for r in dev) == 10
    assert dev == rr.dev_slice(sample, kimi, owner, max_kimi=10)
    assert len(rr.dev_slice(sample, kimi, owner, max_kimi=None)) == 50
