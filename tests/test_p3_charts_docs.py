"""Phase-3 image branch: rendered charts (scripts/p3/gen_charts.py) and document images (scripts/p3/gen_docimages.py).

Validity, licences, image files and sizes, determinism, INDEPENDENT gold re-derivation from the stored render spec (the
solvers below are written again here, not imported), unknowns that really are undecidable, unambiguous options, and the
fresh-seed held-out files. Skips a file's tests when it has not been generated.

    .venv/bin/python -m pytest tests/test_p3_charts_docs.py -q
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))

import candidate  # noqa: E402

CAND = ROOT / "data" / "p3" / "candidates"
POOL = ROOT / "data" / "p3" / "pool"
FILES = {"charts": CAND / "I-charts.jsonl", "docimg": CAND / "I-docimg.jsonl",
         "charts_heldout": POOL / "heldout-fresh-charts.jsonl", "docimg_heldout": POOL / "heldout-fresh-docimg.jsonl"}
_CACHE: dict = {}


def rows(name):
    if name not in _CACHE:
        p = FILES[name]
        if not p.is_file():
            pytest.skip(f"{p.name} not generated")
        _CACHE[name] = list(candidate.read(p))
    return _CACHE[name]


def gold_text(r):
    if r["gold"] is None or r["field"]["type"] != "choice":
        return r["gold"]
    return {o["key"]: o["text"] for o in r["field"]["options"]}[r["gold"]]


ALL = ["charts", "docimg", "charts_heldout", "docimg_heldout"]


# ------------------------------------------------------------------------------------------------ shape of every file
@pytest.mark.parametrize("name", ALL)
def test_valid_unique_constructed(name):
    rs = rows(name)
    ids = [r["id"] for r in rs]
    assert len(set(ids)) == len(ids)
    fam_prefix = "chart_" if name.startswith("charts") else "docimg_"
    for r in rs:
        assert candidate.validate(r) == [], r["id"]
        assert r["source"] == "I" and r["gold_kind"] == "constructed"
        assert r["family"].startswith(fam_prefix)
        assert r["provenance"]["licence"] == "generated"
        assert 2 <= r["difficulty"] <= 5
        assert len(r["images"]) == 1
    by_id = {r["id"]: r for r in rs}
    for r in rs:
        if r["parent_id"]:
            par = by_id[r["parent_id"]]
            assert par["gold"] is not None and r["gold"] is None
            assert par["provenance"]["task"] == r["provenance"]["task"]


@pytest.mark.parametrize("name", ["charts", "docimg"])
def test_counts_mix_and_unknown_share(name):
    rs = rows(name)
    assert 4500 <= len(rs) <= 5500
    unk = sum(r["gold"] is None for r in rs) / len(rs)
    assert 0.12 <= unk <= 0.18
    diffs = Counter(r["difficulty"] for r in rs)
    assert set(diffs) == {2, 3, 4, 5}
    types = Counter(r["field"]["type"] for r in rs)
    assert set(types) == {"choice", "noul", "score"}
    noul = Counter(r["gold"] for r in rs if r["field"]["type"] == "noul" and r["gold"] is not None)
    assert 0.4 <= noul[True] / (noul[True] + noul[False]) <= 0.6
    assert len({r["provenance"]["task"] for r in rs}) >= 12


@pytest.mark.parametrize("name", ["charts_heldout", "docimg_heldout"])
def test_heldout_fresh(name):
    rs = rows(name)
    train = rows(name.split("_")[0])
    assert 250 <= len(rs) <= 400
    assert all(r["provenance"].get("heldout") == "fresh" and r["provenance"].get("heldout_tag") for r in rs)
    assert not {r["id"] for r in rs} & {r["id"] for r in train}
    assert not {r["images"][0] for r in rs} & {r["images"][0] for r in train}
    assert {r["provenance"]["seed"] for r in rs}.isdisjoint({r["provenance"]["seed"] for r in train})
    assert 0.08 <= sum(r["gold"] is None for r in rs) / len(rs) <= 0.22


@pytest.mark.parametrize("name", ALL)
def test_images_exist_sized_and_hashed(name):
    from PIL import Image
    rs = rows(name)
    sub = "charts" if name.startswith("charts") else "docimg"
    seen = set()
    for r in rs:
        im = r["images"][0]
        assert not im.startswith(("/", "data/")) and im.startswith(f"p3/images/{sub}/"), im
        if im not in seen:
            assert (ROOT / "data" / im).is_file(), im
            seen.add(im)
    for im in random.Random(4).sample(sorted(seen), min(300, len(seen))):
        p = ROOT / "data" / im
        with Image.open(p) as f:
            w, h = f.size
            assert f.format in ("PNG", "WEBP")
        assert w <= 1280 and h <= 2000, (im, w, h)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == Path(im).stem


@pytest.mark.parametrize("name", ["charts", "docimg"])
def test_no_imajevbench_image(name):
    import gen_image_joint as J
    bench = J.bench_image_shas()
    for r in rows(name):
        assert not set(r["provenance"]["image_sha256"]) & bench


@pytest.mark.parametrize("name", ALL)
def test_options_unambiguous(name):
    num = re.compile(r"^-?[\d,]+(\.\d+)?")
    for r in rows(name):
        f = r["field"]
        if f["type"] == "choice":
            texts = [o["text"] for o in f["options"]]
            norm = [re.sub(r"[\W_]+", " ", t.lower()).strip() for t in texts]
            assert len(set(norm)) == len(norm), (r["id"], texts)
            if r["provenance"]["task"] not in ("readoff", "diff", "share", "tt_duration"):
                continue
            vals = [float(num.match(t).group(0).replace(",", "")) for t in texts if num.match(t)]
            assert len(vals) == len(texts), (r["id"], texts)
            if len(vals) > 1:
                vs = sorted(vals)
                gaps = [b - a for a, b in zip(vs, vs[1:])]
                assert min(gaps) >= 0.01 * max(abs(v) for v in vs) - 1e-9, (r["id"], texts)
        elif f["type"] == "score":
            ds = [l["description"] for l in f["levels"]]
            assert len(set(ds)) == len(ds), r["id"]


@pytest.mark.parametrize("name", ALL)
def test_state_leaves_are_short(name):
    """No templated state text long enough to form a 13-gram (the leakage rule's unit)."""
    for r in rows(name):
        for v in (r["state"] or {}).values():
            assert len(re.findall(r"[a-z0-9]+", str(v).lower())) < 13, r["id"]


# ------------------------------------------------------------------------------------------------ determinism
def _strip(items):
    out = []
    for it in items:
        it = {k: v for k, v in it.items() if k not in ("parent",)}
        out.append(json.dumps(it, sort_keys=True, default=str))
    return out


def test_charts_plan_deterministic():
    import gen_charts as C
    a = C.plan("unit-test", 80)
    b = C.plan("unit-test", 80)
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)
    assert _strip(a[1]) == _strip(b[1])
    assert _strip(a[1]) != _strip(C.plan("unit-test-2", 80)[1])


def test_docimg_plan_deterministic():
    import gen_docimages as Dg
    a = Dg.plan("unit-test", 80)
    b = Dg.plan("unit-test", 80)
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)
    assert _strip(a[1]) == _strip(b[1])


@pytest.mark.parametrize("name", ["charts", "docimg"])
def test_page_rebuilds_from_stored_spec(name):
    """The stored spec alone rebuilds the exact page that was rendered."""
    import gen_charts as C
    import gen_docimages as Dg
    mod = C if name == "charts" else Dg
    for r in random.Random(5).sample(rows(name), 200):
        html = mod.render_html(r["provenance"]["spec"])
        assert hashlib.sha256(html.encode()).hexdigest() == r["provenance"]["html_sha256"], r["id"]


# ------------------------------------------------------------------------------------------------ charts: independent solver
def c_val(sp, s, c):
    if c not in sp["categories"]:
        raise LookupError("category")
    i = sp["categories"].index(c)
    if s is None:
        return sum(x["values"][i] for x in sp["series"])
    for x in sp["series"]:
        if x["name"] == s:
            return x["values"][i]
    raise LookupError("series")


def c_answer(sp, task, p, opts):
    say = lambda c: sp.get("spoken", {}).get(c, c)  # noqa: E731
    if task == "extremum":
        vals = [(c_val(sp, r["s"], r["c"]), lab) for lab, r in p["cands"]]
        tgt = max(v for v, _ in vals) if p["mode"] == "max" else min(v for v, _ in vals)
        win = [lab for v, lab in vals if v == tgt]
        return win[0] if len(win) == 1 else "TIE"
    if task == "threshold":
        v = c_val(sp, p["ref"]["s"], p["ref"]["c"])
        return v > p["T"] if p["op"] == ">" else v < p["T"]
    if task in ("readoff", "share"):
        v = c_val(sp, p["ref"]["s"], p["ref"]["c"])
        m = [o for o, x in zip(opts, p["opt_values"]) if math.isclose(x, v, abs_tol=1e-6)]
        return m[0] if m else "NONE"
    if task == "diff":
        v = c_val(sp, p["a"]["s"], p["a"]["c"]) - c_val(sp, p["b"]["s"], p["b"]["c"])
        m = [o for o, x in zip(opts, p["opt_values"]) if math.isclose(x, v, abs_tol=1e-6)]
        return m[0] if m else "NONE"
    if task == "ratio":
        b = c_val(sp, p["b"]["s"], p["b"]["c"])
        v = c_val(sp, p["a"]["s"], p["a"]["c"]) / b if b else -1
        m = [o for o, x in zip(opts, p["opt_values"]) if math.isclose(x, v, abs_tol=1e-6)]
        return m[0] if m else "NONE"
    if task == "rank":
        vals = [(c_val(sp, r["s"], r["c"]), lab) for lab, r in p["cands"]]
        if len({v for v, _ in vals}) < 3:
            return "TIE"
        return ", ".join(lab for v, lab in sorted(vals, reverse=True))
    if task == "trend":
        v = [c_val(sp, p["s"], c) for c in sp["categories"]]
        up = [b > a for a, b in zip(v, v[1:])]
        if any(a == b for a, b in zip(v, v[1:])):
            return "OTHER"
        turns = sum(x != y for x, y in zip(up, up[1:]))
        key = "rise" if all(up) else "fall" if not any(up) else ("peak" if up[0] else "valley") if turns == 1 else None
        return p["labels"][key] if key else "OTHER"
    if task == "crossover":
        a = [c_val(sp, p["a"], c) for c in sp["categories"]]
        b = [c_val(sp, p["b"], c) for c in sp["categories"]]
        if a[0] > b[0]:
            return "NONE"
        for i, (x, y) in enumerate(zip(a, b)):
            if x > y:
                return say(sp["categories"][i])
        return "NONE"
    if task == "claim":
        k = p["claim"]
        if k == "above_every":
            return all(c_val(sp, p["a"], c) > c_val(sp, p["b"], c) for c in sp["categories"])
        if k == "doubled":
            return c_val(sp, p["r2"]["s"], p["r2"]["c"]) >= 2 * c_val(sp, p["r1"]["s"], p["r1"]["c"])
        if k == "higher":
            return c_val(sp, p["r1"]["s"], p["r1"]["c"]) > c_val(sp, p["r2"]["s"], p["r2"]["c"])
        if k == "over_half":
            return c_val(sp, p["ref"]["s"], p["ref"]["c"]) > 50
    if task == "band":
        v = c_val(sp, p["ref"]["s"], p["ref"]["c"])
        return len([e for e in p["edges"] if v >= e])
    if task == "count":
        return len([r for r in p["refs"] if c_val(sp, r["s"], r["c"]) > p["T"]])
    if task == "relation":
        xs, ys = sp["series"][0]["values"], sp["series"][1]["values"]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        r = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return p["labels"]["positive"] if r > 0.7 else p["labels"]["negative"] if r < -0.7 else p["labels"]["none"] if abs(r) < 0.2 else "OTHER"
    raise AssertionError(task)


def c_worlds(sp):
    """What the picture leaves open: values under a sticky note, unlabelled series, an unnumbered axis."""
    ws = [sp]
    if sp.get("cover") is not None:
        out = []
        if sp["kind"] == "table":
            k = [s["name"] for s in sp["series"]].index(sp["cover"]["col"])
            i = sp["categories"].index(sp["cover"]["row"])
            big = 10 * max(max(s["values"]) for s in sp["series"])
            for v in (0.0, sp["series"][k]["values"][i], big):
                w = copy.deepcopy(sp)
                w["series"][k]["values"][i] = v
                out.append(w)
        else:
            i = sp["categories"].index(sp["cover"])
            for combo in itertools.product((0.0, sp["y_max"] / 2, sp["y_max"]), repeat=len(sp["series"])):
                w = copy.deepcopy(sp)
                for s, v in zip(w["series"], combo):
                    s["values"][i] = v
                out.append(w)
        ws = out
    if (len(sp["series"]) > 1 and sp["kind"] in ("grouped", "stacked", "line", "hbar") and not sp["show"]["legend"]
            and not sp["show"]["direct_labels"]):
        out = []
        for w in ws:
            for perm in itertools.permutations([s["name"] for s in sp["series"]]):
                w2 = copy.deepcopy(w)
                for s, nm in zip(w2["series"], perm):
                    s["name"] = nm
                out.append(w2)
        ws = out
    if sp["kind"] in ("vbar", "hbar", "grouped", "stacked", "line") and not sp["show"]["y_labels"] and not sp["show"]["data_labels"]:
        out = []
        for w in ws:
            for k in (0.5, 1.0, 2.0):
                w2 = copy.deepcopy(w)
                for s in w2["series"]:
                    s["values"] = [v * k for v in s["values"]]
                out.append(w2)
        ws = out
    return ws


def c_decide(r):
    pv, f = r["provenance"], r["field"]
    opts = [o["text"] for o in f.get("options", [])]
    answers = set()
    for w in c_worlds(pv["spec"]):
        try:
            answers.add(c_answer(w, pv["task"], pv["params"], opts))
        except LookupError:
            return "UNKNOWN"
    if len(answers) != 1:
        return "UNKNOWN"
    a = answers.pop()
    return "UNKNOWN" if a in ("TIE", "NONE", "OTHER") else a


@pytest.mark.parametrize("name", ["charts", "charts_heldout"])
def test_chart_gold_rederived(name):
    rs = rows(name)
    sample = rs if len(rs) < 2000 else random.Random(6).sample(rs, 2500)
    for r in sample:
        got = c_decide(r)
        want = "UNKNOWN" if r["gold"] is None else gold_text(r)
        assert got == want, (r["id"], got, want)


def test_chart_unknowns_are_undecidable():
    rs = rows("charts")
    by_id = {r["id"]: r for r in rs}
    how = Counter()
    for r in rs:
        if r["gold"] is not None:
            continue
        pv = r["provenance"]
        sp, p = pv["spec"], pv["params"]
        k = pv["unknown_construction"]
        how[k] += 1
        par = by_id[r["parent_id"]]
        assert r["field"] == {**par["field"], "question": r["field"]["question"]}
        if k == "absent_series":
            names = {s["name"] for s in sp["series"]}
            refs = json.dumps(p)
            assert any(f'"{a}"' in refs for a in sp["absent_series"]) and r["images"] == par["images"]
            assert any(a in r["field"]["question"] for a in sp["absent_series"] if a not in names)
        elif k == "outside_axis":
            assert any(o in r["field"]["question"] for o in sp["outside"]) and r["images"] == par["images"]
        else:
            # re-rendered chart: two readings consistent with the picture disagree
            ws = c_worlds(sp)
            opts = [o["text"] for o in r["field"].get("options", [])]
            ans = {str(c_answer(w, pv["task"], p, opts)) for w in ws}
            assert len(ans) > 1 or ans & {"TIE", "NONE", "OTHER"}, r["id"]
            assert r["images"] != par["images"]
            psp = par["provenance"]["spec"]
            if k == "covered":
                assert psp.get("cover") != sp["cover"] and sp["cover"] is not None
            if k == "no_legend":
                assert psp["show"]["legend"] or psp["show"]["direct_labels"]
                assert not sp["show"]["legend"] and not sp["show"]["direct_labels"]
            if k == "no_scale":
                assert psp["show"]["y_labels"] and not sp["show"]["y_labels"] and not sp["show"]["data_labels"]
    assert set(how) == {"covered", "absent_series", "outside_axis", "no_legend", "no_scale"}


def test_chart_readability_margins():
    """Answerable comparisons have a visible gap; exact reads sit on ticks when no value is printed."""
    for r in rows("charts"):
        if r["gold"] is None:
            continue
        pv = r["provenance"]
        sp, p, t = pv["spec"], pv["params"], pv["task"]
        printed = sp["kind"] == "table" or (sp["show"]["data_labels"] and sp["kind"] not in ("stacked", "dual"))
        top = sp.get("y2_max") if p.get("axis2") else sp["y_max"]
        if sp["kind"] in ("pie", "donut", "scatter", "table") or printed:
            continue
        g = 0.06 * top - 1e-9
        refs = [x[1] for x in p.get("cands", [])] + p.get("refs", []) + [p[k] for k in ("ref", "a", "b") if isinstance(p.get(k), dict)]
        if sp["kind"] == "stacked" and sp["show"]["data_labels"] and refs and all(x["s"] is None for x in refs):
            continue          # stack totals are printed on top of each stack
        if t == "extremum":
            vs = sorted((c_val(sp, r_["s"], r_["c"]) for _, r_ in p["cands"]), reverse=p["mode"] == "max")
            assert abs(vs[0] - vs[1]) >= g, r["id"]
        if t == "threshold":
            assert abs(c_val(sp, p["ref"]["s"], p["ref"]["c"]) - p["T"]) >= g, r["id"]
        if t in ("readoff",):
            step = sp.get("y2_step") if p.get("axis2") else sp["y_step"]
            v = c_val(sp, p["ref"]["s"], p["ref"]["c"])
            assert abs(v / step - round(v / step)) < 1e-9, r["id"]
            vs = sorted(p["opt_values"])
            assert min(b - a for a, b in zip(vs, vs[1:])) >= step - 1e-9, r["id"]
        if t == "count":
            assert all(abs(c_val(sp, x["s"], x["c"]) - p["T"]) >= g for x in p["refs"]), r["id"]


# ------------------------------------------------------------------------------------------------ documents: independent solver
def d_hidden(sp):
    h = set(sp.get("hide", {}))
    if sp.get("crop_at"):
        order = sp["section_order"]
        gone = set(order[order.index(sp["crop_at"]):])
        h |= {k for k, s in sp["sections"].items() if s in gone}
    return h


class Unknown(Exception):
    pass


def d_answer(sp, task, p):
    hid = d_hidden(sp)
    f = sp["f"]

    def g(k):
        if k not in f or k in hid:
            raise Unknown(k)
        return f[k]

    def n(k):
        return float(g(k))

    def day(x):
        return dt.date.fromisoformat(x)

    def mins(t):
        return int(t[:2]) * 60 + int(t[3:])
    if task == "arith_total":
        return abs(sum(n(k) for k in p["parts"]) - n(p["total"])) < 0.005
    if task in ("balance_check", "largest_debit", "count_over") and int(g("pages")) > 1:
        raise Unknown("pages")
    if task == "balance_check":
        return abs(n(p["open"]) + sum(n(k) for k in p["credits"]) - sum(n(k) for k in p["debits"]) - n(p["close"])) < 0.005
    if task == "change_ok":
        return abs(n(p["tendered"]) - n(p["total"]) - n(p["change"])) < 0.005
    if task == "line_calc":
        bad = [lab for lab, q, u, a in p["rows"] if abs(n(q) * n(u) - n(a)) >= 0.005]
        if len(bad) != 1:
            raise Unknown("none/multi")
        return bad[0]
    if task in ("max_item", "largest_debit"):
        vals = [(n(k), lab) for lab, k in p["rows"]]
        m = max(v for v, _ in vals)
        win = [lab for v, lab in vals if v == m]
        assert len(win) == 1
        return win[0]
    if task == "overdue":
        if "paid" in f and "paid" not in hid:
            return False
        due = day(g("due")) if "due" in f else day(g("issue")) + dt.timedelta(days=int(g("terms")))
        return day(p["today"]) > due
    if task in ("expired", "deadline_passed"):
        return day(p["today"]) > day(g(p["key"]))
    if task == "late_band":
        if "paid" in f and "paid" not in hid:
            return 0
        due = day(g("due")) if "due" in f else day(g("issue")) + dt.timedelta(days=int(g("terms")))
        late = (day(p["today"]) - due).days
        return 0 if late < 1 else 1 if late < 31 else 2 if late < 61 else 3
    if task == "count_items":
        return len([k for k in p["rows"] if n(k) > p["T"]])
    if task == "threshold_amt":
        amt, cur = n(p["amount"]), g(p["cur"])
        if cur != p["limit_cur"]:
            if cur not in (p.get("rates") or {}):
                raise Unknown("rate")
            amt *= p["rates"][cur]
        return amt > p["limit"]
    if task == "allowance":
        return n(p["total"]) <= p["per_person"] * p["people"]
    if task == "budget_ok":
        return n(p["total"]) <= p["budget"]
    if task == "overweight":
        return n(p["key"]) > p["limit"]
    if task == "lead_time":
        return day(p["today"]) + dt.timedelta(days=p["days"]) <= day(g(p["key"]))
    if task == "access":
        return g(p["tier"]) in p["allowed"] and day(p["today"]) <= day(g(p["until"]))
    if task == "deadline":
        return p["map"][g("deadline")]
    if task in ("currency", "dest_city", "signer", "tier", "ask", "payment"):
        v = g(p["key"])
        if v == "":
            raise Unknown("blank")
        return v
    if task == "doc_type":
        return {"invoice": "invoice", "receipt": "receipt", "po": "purchase order", "statement": "bank statement",
                "form": "application form", "label": "shipping label", "letter": "letter", "card": "membership card",
                "timetable": "timetable"}[sp["doc"]]
    if task == "tax_rate":
        return abs(n(p["tax"]) - round(n(p["sub"]) * p["rate"] / 100 + 1e-9, 2)) < 0.015
    if task == "signed":
        return g(p["key"]) != ""
    if task == "consent":
        return bool(g(p["key"]))
    if task == "blank_field":
        b = [lab for lab, k in p["fields"] if g(k) == ""]
        if len(b) != 1:
            raise Unknown("blank")
        return b[0]
    if task in ("ticked", "service"):
        on = [lab for lab, k in p["group"] if g(k)]
        if len(on) != 1:
            raise Unknown("ticked")
        return on[0]
    if task == "adult":
        b = g(p["key"])
        if not b:
            raise Unknown("dob")
        b, t = day(b), day(p["today"])
        return (t.year - b.year - ((t.month, t.day) < (b.month, b.day))) >= p["age"]
    if task == "international":
        return g(p["from"]) != g(p["to"])
    if task == "merchant":
        seen, blind = False, False
        for k in p["rows"]:
            if k in hid:
                blind = True
            elif p["name"] in f[k]:
                seen = True
        if seen:
            return True
        if blind or int(g(p["pages"])) > 1:
            raise Unknown("merchant")
        return False
    if task == "overdrawn":
        if any(n(k) < 0 for k in p["rows"]):
            return True
        if int(g("pages")) > 1:
            raise Unknown("pages")
        return False
    if task == "count_over":
        return len([k for k in p["rows"] if n(k) > p["T"]])
    if task == "tt_latest":
        ok = [(mins(g(f"tt_{p['frm']}_{c}")), g(f"tt_{p['frm']}_{c}")) for c in p["cols"]
              if "—" not in (g(f"tt_{p['frm']}_{c}"), g(f"tt_{p['to']}_{c}")) and mins(g(f"tt_{p['to']}_{c}")) <= p["by"]]
        if not ok:
            raise Unknown("none")
        return max(ok)[1]
    if task == "tt_stops":
        return g(f"tt_{p['row']}_{p['col']}") != "—"
    if task == "tt_count":
        return len([c for c in p["cols"] if g(f"tt_{p['row']}_{c}") != "—" and mins(g(f"tt_{p['row']}_{c}")) < p["before"]])
    if task == "tt_duration":
        a, b = g(f"tt_{p['frm']}_{p['col']}"), g(f"tt_{p['to']}_{p['col']}")
        return f"{mins(b) - mins(a)} min"
    raise AssertionError(task)


def d_decide(r):
    pv = r["provenance"]
    try:
        return d_answer(pv["spec"], pv["task"], pv["params"])
    except Unknown:
        return "UNKNOWN"


@pytest.mark.parametrize("name", ["docimg", "docimg_heldout"])
def test_doc_gold_rederived(name):
    for r in rows(name):
        got = d_decide(r)
        want = "UNKNOWN" if r["gold"] is None else gold_text(r)
        assert got == want, (r["id"], got, want)


def test_doc_unknowns_are_undecidable():
    rs = rows("docimg")
    by_id = {r["id"]: r for r in rs}
    how = Counter()
    for r in rs:
        if r["gold"] is None:
            pv = r["provenance"]
            sp, par = pv["spec"], by_id[r["parent_id"]]
            k = pv["unknown_construction"]
            how[k] += 1
            psp = par["provenance"]["spec"]
            read = set(par["provenance"]["fields_read"])
            if k == "covered":
                newly = set(sp["hide"]) - set(psp["hide"])
                assert newly and newly & read, r["id"]
            elif k == "cropped":
                assert sp["crop_at"] and d_hidden(sp) & read, r["id"]
                assert r["provenance"]["post"]["crop_h"] < par["provenance"]["post"]["crop_h"]
            elif k == "not_printed":
                assert (set(psp["f"]) - set(sp["f"])) & read, r["id"]
            elif k == "no_exchange_rate":
                assert pv["params"]["limit_cur"] != sp["f"]["cur"] and not pv["params"].get("rates")
                assert not any(x.startswith("rate_") for x in r["state"])
            elif k == "none_applies":
                assert pv["task"] in ("blank_field", "line_calc", "ticked", "service")
            elif k == "partial_statement":
                assert sp["f"]["pages"] > 1
            else:
                raise AssertionError(k)
    assert {"covered", "cropped", "not_printed", "no_exchange_rate", "none_applies", "partial_statement"} <= set(how)


def test_doc_render_proved_fields_visible():
    """Every row records the fields its answer read; the builder rejected pages where one of them was not in view."""
    for r in rows("docimg"):
        pv = r["provenance"]
        if r["gold"] is not None and pv["task"] != "doc_type":
            assert pv["fields_read"], r["id"]
        assert pv["post"]["crop_h"] > 100


def test_docs_no_government_id_look():
    for r in rows("docimg"):
        sp = r["provenance"]["spec"]
        if sp["doc"] == "card":
            assert sp["f"]["title"] in {"Gym membership", "Library card", "Staff badge", "Conference pass", "Club card"}
            assert not re.search(r"passport|national|driv|identity card|social security", json.dumps(sp["f"]).lower())
