"""Phase-3 image branch: geometry diagrams (gen_geometry.py) and screen grounding (gen_screens.py).

Validity, determinism, gold re-derivation from the stored spec (geometry recomputed by an independent solver written
here; screen golds recomputed from the stored element boxes, grid and marks), unknowns really undecidable, images exist,
held-out files disjoint from training. Skips a file's tests when it has not been generated.

    .venv/bin/python -m pytest tests/test_p3_geometry_screens.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
sys.path.insert(0, str(ROOT / "scripts" / "p3" / "geometry_templates"))
sys.path.insert(0, str(ROOT / "scripts" / "p3" / "screen_templates"))

import candidate  # noqa: E402

CAND = ROOT / "data" / "p3" / "candidates"
POOL = ROOT / "data" / "p3" / "pool"
FILES = {"geo": CAND / "I-geometry.jsonl", "scr": CAND / "I-screens.jsonl",
         "geo_ho": POOL / "heldout-fresh-geometry.jsonl", "scr_ho": POOL / "heldout-fresh-screens.jsonl"}
TOL = 0.05 + 1e-9


def rows(name):
    p = FILES[name]
    if not p.is_file():
        pytest.skip(f"{p.name} not generated")
    return list(candidate.read(p))


@pytest.fixture(scope="module")
def geo():
    return rows("geo")


@pytest.fixture(scope="module")
def scr():
    return rows("scr")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ common checks
@pytest.mark.parametrize("name", ["geo", "scr", "geo_ho", "scr_ho"])
def test_valid_unique_images_licence(name):
    from PIL import Image
    rs = rows(name)
    ids = [r["id"] for r in rs]
    assert len(set(ids)) == len(ids)
    sub = "geometry" if name.startswith("geo") else "screens"
    seen = set()
    for r in rs:
        assert candidate.validate(r) == [], r["id"]
        assert r["source"] == "I" and r["gold_kind"] == "constructed"
        assert r["provenance"]["licence"] == "generated"
        assert 2 <= r["difficulty"] <= 5
        assert len(r["images"]) == 1
        im = r["images"][0]
        assert im.startswith(f"p3/images/{sub}/"), im
        if name.endswith("_ho"):
            assert im.startswith(f"p3/images/{sub}/heldout/")
        if im not in seen:
            assert (ROOT / "data" / im).is_file(), im
            seen.add(im)
    for im in random.Random(0).sample(sorted(seen), min(150, len(seen))):
        p = ROOT / "data" / im
        assert sha(p) == Path(im).stem
        w, h = Image.open(p).size
        assert w <= 1280 and h <= 1280


@pytest.mark.parametrize("name", ["geo", "scr"])
def test_no_benchmark_image(name):
    import gen_image_joint as J
    bench = J.bench_image_shas()
    for r in rows(name):
        assert not set(r["provenance"]["image_sha256"]) & bench


@pytest.mark.parametrize("name", ["geo", "scr"])
def test_parent_links(name):
    rs = rows(name)
    by = {r["id"]: r for r in rs}
    for r in rs:
        if r["parent_id"]:
            assert r["parent_id"] in by and by[r["parent_id"]]["parent_id"] is None


# ------------------------------------------------------------------------------------------------ geometry: independent solver
POS = [("U", "UL"), ("U", "UR"), ("U", "LL"), ("U", "LR"), ("L", "UL"), ("L", "UR"), ("L", "LL"), ("L", "LR")]


def _rel(a, b):
    (i1, q1), (i2, q2) = POS[a], POS[b]
    if i1 == i2:
        return "vertical angles" if q1[0] != q2[0] and q1[1] != q2[1] else "a linear pair"
    if q1 == q2:
        return "corresponding angles"
    between = lambda i, q: (i == "U") == (q[0] == "L")  # noqa: E731  interior = between the two lines
    b1, b2 = between(i1, q1), between(i2, q2)
    if b1 != b2:
        return None
    kind = "interior" if b1 else "exterior"
    return (f"alternate {kind} angles" if q1[1] != q2[1] else f"same-side {kind} angles")


def geo_solve(sp):
    """Answer from ONLY the shown givens and the marks; None = undetermined. Areas / circumferences of circles in π units."""
    t, v = sp["t"], sp.get("vals", {})
    sh = set(sp["shown"])
    mk = sp.get("marks", {})
    if t == "tri_angle_sum":
        return 180 - sum(v[k] for k in sh) if len(sh) == 2 else None
    if t == "tri_exterior":
        if sp["kind"] == "ext":
            return v["A"] + v["B"] if {"A", "B"} <= sh else None
        return v["ext"] - v["A"] if {"A", "ext"} <= sh else None
    if t == "isosceles":
        if not mk["ticks"]:
            return None
        if sp["kind"] == "base":
            return (180 - v["apex"]) / 2 if "apex" in sh else None
        return 180 - 2 * v["base"] if "base" in sh else None
    if t == "parallel":
        a, b = sp["perm"][sp["gi"] - 1], sp["perm"][sp["aj"] - 1]
        if sp["kind"] == "relation":
            return _rel(a, b)
        if POS[a][0] != POS[b][0] and not mk["parallel"]:
            return None
        same = (POS[a][1] in ("UR", "LL")) == (POS[b][1] in ("UR", "LL"))
        if sp["kind"] == "equal":
            return same
        if "g" not in sh:
            return None
        return v["g"] if same else 180 - v["g"]
    if t == "lines_algebra":
        a, b, c, d = v["a"], v["b"], v["c"], v["d"]
        q0, q1 = sp["quads"]
        x = (d - b) / (a - c) if abs(q0 - q1) == 2 else (180 - b - d) / (a + c)
        return x if sp["kind"] == "x" else a * x + b
    if t == "pythagoras":
        if not mk["right"]:
            return None
        if sp["kind"] == "hyp":
            return math.sqrt(v["a"] ** 2 + v["b"] ** 2) if {"a", "b"} <= sh else None
        return math.sqrt(v["c"] ** 2 - v["a"] ** 2) if {"a", "c"} <= sh else None
    if t == "circle_measure":
        if not sh:
            return None
        r = v["r"] if "r" in sh else v["d"] / 2
        return {"area": r * r, "circ": 2 * r, "diam": 2 * r, "radius": r}[sp["ask"]]
    if t == "inscribed":
        if sp["kind"] == "thales":
            return 90
        if sp["kind"] == "insc":
            return v["cent"] / 2 if "cent" in sh else None
        return 2 * v["insc"] if "insc" in sh else None
    if t == "tangent":
        if sp["kind"] == "pt":
            return math.sqrt(v["d"] ** 2 - v["r"] ** 2) if {"r", "d"} <= sh else None
        if sp["kind"] == "op":
            return math.sqrt(v["r"] ** 2 + v["t"] ** 2) if {"r", "t"} <= sh else None
        return 90 - v["alpha"] if "alpha" in sh else None
    if t == "parallelogram":
        if not mk["parallel"]:
            return None
        if sp["kind"] == "angle":
            return None if "A" not in sh else (v["A"] if sp["ask"] == "C" else 180 - v["A"])
        if sp["kind"] == "area":
            return v["b"] * v["h"] if {"b", "h"} <= sh else None
        return 2 * (v["b"] + v["s"]) if {"b", "s"} <= sh else None
    if t == "regular_polygon":
        n = v["n"]
        return {"int": 180 - 360 / n, "ext": 360 / n, "sum": 180 * n - 360}[sp["kind"]]
    if t == "polygon_missing":
        n = sp["n"]
        return (n - 2) * 180 - sum(v[k] for k in sh) if len(sh) == n - 1 else None
    if t == "area_shape":
        if sp["shape"] == "rect":
            if not {"w", "h"} <= sh:
                return None
            return v["w"] * v["h"] if sp["kind"] == "area" else 2 * v["w"] + 2 * v["h"]
        if sp["shape"] == "tri":
            return 0.5 * v["b"] * v["h"] if {"b", "h"} <= sh else None
        return 0.5 * (v["a"] + v["b"]) * v["h"] if {"a", "b", "h"} <= sh else None
    if t == "coord":
        P = {k: tuple(p) for k, p in sp["pts"].items()}
        k = sp["kind"]
        if k == "dist":
            return math.dist(P["A"], P["B"])
        if k == "slope":
            return Fraction(P["B"][1] - P["A"][1], P["B"][0] - P["A"][0])
        if k == "mid":
            return ((P["A"][0] + P["B"][0]) / 2, (P["A"][1] + P["B"][1]) / 2)
        if k == "area":
            (a, b), (c, d), (e, f) = P["A"], P["B"], P["C"]
            return abs(a * (d - f) + c * (f - b) + e * (b - d)) / 2
        if k == "right":
            sq = sorted(math.dist(P[x], P[y]) ** 2 for x, y in (("A", "B"), ("B", "C"), ("A", "C")))
            return abs(sq[0] + sq[1] - sq[2]) < 1e-9
        if k == "para":
            A, B, C, D = (P[c] for c in "ABCD")
            return (A[0] + C[0], A[1] + C[1]) == (B[0] + D[0], B[1] + D[1])
        x, y = P["P"]
        return "Quadrant " + {(True, True): "I", (False, True): "II", (False, False): "III", (True, False): "IV"}[(x > 0, y > 0)]
    if t == "classify":
        if sp["kind"] == "iso":
            known = [v[k] for k in sh]
            if len(set(known)) < len(known):
                return True
            return False if len(known) == 3 else None
        if sp["kind"] == "right":
            if len(sh) < 3:
                return None
            a, b, c = sorted(v[k] for k in sh)
            return a * a + b * b == c * c
        known = [v[k] for k in sh]
        if len(known) == 1 and known[0] < 90:
            return None
        m = max(known + ([180 - sum(known)] if len(known) == 2 else []))
        return "acute" if m < 90 else "right" if m == 90 else "obtuse"
    if t == "similar":
        if not mk["parallel"] or not {"p", "q"} <= sh:
            return None
        ratio = (v["p"] + v["q"]) / v["p"]
        if sp["kind"] == "bc":
            return v["e"] * ratio if "e" in sh else None
        return v["bc"] / ratio if "bc" in sh else None
    if t == "count_sides":
        return v["n"] - 3
    raise KeyError(t)


def parse_opt(text):
    t = text.replace("−", "-")
    m = re.fullmatch(r"\((-?[\d.]+), (-?[\d.]+)\)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.fullmatch(r"(-?\d+)/(\d+)", t)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    m = re.match(r"(-?[\d.]+)", t)
    if m:
        return float(m.group(1))
    return text


def same_value(a, b):
    if isinstance(b, tuple):
        return isinstance(a, tuple) and all(abs(x - y) <= TOL for x, y in zip(a, b))
    if isinstance(b, Fraction):
        return isinstance(a, (Fraction, float)) and abs(float(a) - float(b)) < 1e-9
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return isinstance(a, (int, float, Fraction)) and not isinstance(a, tuple) and abs(float(a) - b) <= TOL
    return a == b


def test_geometry_mix(geo):
    assert 2800 <= len(geo) <= 3300
    fam = Counter(r["family"] for r in geo)
    assert all(f.startswith("geometry_") for f in fam) and len(fam) == 8
    assert len({r["provenance"]["template"] for r in geo}) == 17
    unk = sum(r["gold"] is None for r in geo) / len(geo)
    assert 0.12 <= unk <= 0.18
    assert set(Counter(r["difficulty"] for r in geo)) == {2, 3, 4, 5}
    assert {r["field"]["type"] for r in geo} == {"choice", "noul", "score"}
    assert sum("given" in r["state"] for r in geo) > 300


def _check_geo_row(r):
    sp = r["provenance"]["spec"]
    ans = geo_solve(sp)
    f = r["field"]
    if r["gold"] is None:
        assert ans is None, r["id"]
        assert r["unknown_reason"] == "insufficient_evidence"
        return
    assert ans is not None, r["id"]
    if f["type"] == "choice":
        opts = {o["key"]: o["text"] for o in f["options"]}
        hits = [k for k, txt in opts.items() if same_value(parse_opt(txt), ans)]
        assert hits == [r["gold"]], (r["id"], ans, opts, r["gold"])
    elif f["type"] == "score":
        assert r["gold"] == ans and f["levels"][ans]["description"] == f"{sp['vals']['n']} sides"
    else:
        assert r["gold"] is ans, r["id"]


def test_geometry_gold_rederived(geo):
    for r in geo:
        _check_geo_row(r)


def test_geometry_heldout_gold_rederived():
    for r in rows("geo_ho"):
        _check_geo_row(r)


def test_geometry_unknowns_are_undecidable(geo):
    """An unknown child drops exactly one given or mark of its answerable parent; filling it back decides the answer,
    and two different values of the dropped given (or both states of the dropped mark) give different answers."""
    by = {r["id"]: r for r in geo}
    n = 0
    for r in geo:
        if r["gold"] is not None:
            continue
        par = by[r["parent_id"]]
        assert par["gold"] is not None and par["provenance"]["template"] == r["provenance"]["template"]
        assert par["field"].get("options") == r["field"].get("options") and par["field"]["type"] == r["field"]["type"]
        cs, ps = r["provenance"]["spec"], par["provenance"]["spec"]
        dropped = set(ps["shown"]) - set(cs["shown"])
        marks = {k for k in ps.get("marks", {}) if ps["marks"][k] and not cs["marks"][k]}
        assert len(dropped) + len(marks) == 1, r["id"]
        assert geo_solve(ps) is not None
        if dropped:
            k = dropped.pop()
            others = [ps["vals"][x] for x in cs["shown"] if isinstance(ps["vals"].get(x), (int, float))]
            cands = [ps["vals"][k], ps["vals"][k] * 1.25 + 3, ps["vals"][k] / 2 + 1, 30, 45, 60, 75, 80, 90, 100] + others
            cands += [math.sqrt(x * x + y * y) for x in others for y in others] + [math.sqrt(abs(x * x - y * y)) for x in others for y in others]
            outs = set()
            for val in cands:
                alt = json.loads(json.dumps(ps))
                alt["vals"][k] = val
                try:
                    o = geo_solve(alt)
                except (ValueError, ZeroDivisionError):
                    continue
                outs.add(round(o, 6) if isinstance(o, float) else json.dumps(o, default=str))
            assert len(outs) >= 2, r["id"]
        n += 1
    assert n > 350


def test_geometry_state_hides_dropped_given(geo):
    import templates as T
    for r in geo:
        if r["gold"] is None:
            sp = r["provenance"]["spec"]
            st = json.dumps(r["state"], ensure_ascii=False)
            for k in {k for k, g in sp.get("gtext", {}).items() if g[2] is not None} - set(sp["shown"]):
                assert T.given_text(k, sp) not in st, r["id"]


def test_geometry_not_to_scale_rule(geo):
    for r in geo:
        pv = r["provenance"]
        t = pv["template"]
        if t in ("coord", "regular_polygon", "count_sides", "lines_algebra"):
            assert pv["drawn_to_scale"] and pv["figure_note"] is None
        else:
            assert pv["figure_note"] == "Figure not drawn to scale"


def test_geometry_deterministic(geo):
    import gen_geometry as G
    from gen_image_joint import rng_for
    seed = geo[0]["provenance"]["seed"]
    parents = [r for r in geo if r["parent_id"] is None][:60]
    got = []
    i = 0
    while len(got) < 60:
        g = G.make_item(rng_for(seed, "geo", i))
        i += 1
        if g:
            got.append(G._json_safe(g[0]["spec"]))
    assert [p["provenance"]["spec"] for p in parents] == got
    a = G.make_item(rng_for("x", 1))
    b = G.make_item(rng_for("x", 1))
    assert json.dumps([x["spec"] for x in a], default=str) == json.dumps([x["spec"] for x in b], default=str)


# ------------------------------------------------------------------------------------------------ screens
SCREEN_FAMILIES = {"screen_locate", "mobile_locate", "mobile_tap", "mobile_possible", "mobile_next", "mobile_permission",
                   "pro_locate", "pro_ground", "pro_menu"}
NAMED = [["top left", "top centre", "top right"], ["middle left", "centre", "middle right"], ["bottom left", "bottom centre", "bottom right"]]


def _inter(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _cell_label(r, c, style):
    return NAMED[r][c] if style == "named" else f"row {r + 1}, column {c + 1}" if style == "rowcol" else f"{'ABCD'[r]}{c + 1}"


def _check_screen_row(r):
    pv, sp, f = r["provenance"], r["provenance"]["spec"], r["field"]
    task = pv["task"]
    opts = {o["key"]: o["text"] for o in f.get("options") or []}
    gold_text = opts.get(r["gold"]) if f["type"] == "choice" else r["gold"]
    W, H = pv["image_size"]
    assert abs(W - pv["viewport"][0] * pv["dpr"]) < 1 and abs(H - pv["viewport"][1] * pv["dpr"]) < 1
    if task == "locate":
        g = sp["grid"]
        n = g["rows"]
        assert list(opts.values()) == [_cell_label(i, j, g["style"]) for i in range(n) for j in range(n)]
        if r["gold"] is None:
            if pv["unknown_construction"] == "occluded":
                b, d = sp["target_box"], sp["dialog_box"]
                assert not sp["target_visible"] and b[0] >= d[0] and b[1] >= d[1] and b[0] + b[2] <= d[0] + d[2] and b[1] + b[3] <= d[1] + d[3]
                assert r["unknown_reason"] == "insufficient_evidence"
            else:
                assert sp["target_desc"] not in sp["present_descs"] and r["unknown_reason"] == "not_listed"
            return
        b = sp["target_box"]
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        ci, ri = int(cx // (W / n)), int(cy // (H / n))
        x0, y0 = ci * W / n, ri * H / n
        assert x0 <= b[0] and b[0] + b[2] <= x0 + W / n and y0 <= b[1] and b[1] + b[3] <= y0 + H / n, r["id"]   # no straddling
        assert gold_text == _cell_label(ri, ci, g["style"]), r["id"]
        for bb in g["badges"]:
            assert not _inter(b, bb)
        return
    if task in ("ground", "ground_goal", "tap_box"):
        marks = sp["marks"]
        assert sorted(opts.values()) == sorted(m["letter"] for m in marks)
        for i, a in enumerate(marks):
            assert 0 <= a["badge"][0] and a["badge"][0] + a["badge"][2] <= W + 0.5 and 0 <= a["badge"][1] and a["badge"][1] + a["badge"][3] <= H + 0.5
            for b in marks[i + 1:]:
                assert not _inter(a["box"], b["box"]) and not _inter(a["badge"], b["badge"])
            for b in marks:
                if b is not a:
                    assert not _inter(a["badge"], b["box"])
            # the drawn box surrounds the element
            eb = a["el_box"]
            assert a["box"][0] <= eb[0] + 0.5 and a["box"][1] <= eb[1] + 0.5
        if r["gold"] is None:
            assert r["unknown_reason"] == "not_listed" and sp["target"] is None
            if sp["goal"]:
                assert sp["goal"] not in sp["present_goals"] or task == "ground"
            else:
                assert sp["target_desc"] not in sp["present_descs"]
            return
        m = next(m for m in marks if m["letter"] == gold_text)
        assert m["el"] == sp["target"] and m["el_box"] == sp["target_box"], r["id"]
        return
    if task in ("tap_label", "menu"):
        oe = sp["option_elements"]
        assert list(opts.values()) == list(oe)
        if r["gold"] is None:
            assert all(sp["goal"] not in gs for gs in sp["option_goals"].values())
            return
        assert oe[gold_text] == sp["target"], r["id"]
        return
    if task in ("possible", "menu_possible"):
        assert r["gold"] is (not sp["disabled"])
        return
    if task == "next":
        assert gold_text == sp["action"]
        return
    if task in ("perm_more", "perm_which"):
        import mobile
        need = dict((t, sorted(n)) for t, n in mobile.TASKS)[sp["task_text"]]
        assert sp["needed"] == need and r["state"]["task"] == sp["task_text"]
        extra = sorted(set(sp["requested"]) - set(need))
        if task == "perm_more":
            assert r["gold"] is bool(extra)
        else:
            assert list(opts.values()) == sp["requested"]
            assert (gold_text == extra[0]) if extra else (r["gold"] is None and r["unknown_reason"] == "false_premise")
        return
    raise AssertionError(task)


def test_screens_mix(scr):
    assert 4500 <= len(scr) <= 5700
    fam = Counter(r["family"] for r in scr)
    assert set(fam) == SCREEN_FAMILIES
    unk = sum(r["gold"] is None for r in scr) / len(scr)
    assert 0.10 <= unk <= 0.20
    assert {"not_listed", "insufficient_evidence", "false_premise"} <= {r["unknown_reason"] for r in scr if r["gold"] is None}
    assert set(Counter(r["difficulty"] for r in scr)) == {2, 3, 4, 5}
    plat = Counter(r["provenance"]["platform"] for r in scr)
    assert set(plat) == {"desktop", "mobile", "pro"} and min(plat.values()) > 900
    mob = {r["images"][0] for r in scr if r["provenance"]["platform"] == "mobile"}
    from PIL import Image
    for im in list(mob)[:20]:
        assert Image.open(ROOT / "data" / im).size == (585, 1266)


def test_screens_gold_rederived(scr):
    for r in scr:
        _check_screen_row(r)


def test_screens_heldout_gold_rederived():
    for r in rows("scr_ho"):
        _check_screen_row(r)


def test_screens_rebuild_from_seed(scr):
    """Re-running the template with the stored seed rebuilds the same screen: the target / option elements exist with
    the same description, and the rebuilt HTML renders the stored layout (checked by the renderer's own box report in
    the generator; here we check the registry)."""
    import gen_screens as S
    from gen_image_joint import rng_for
    for r in random.Random(5).sample(scr, 150):
        pv = r["provenance"]
        s1 = S.build_screen(rng_for(*pv["screen_seed"]))
        s2 = S.build_screen(rng_for(*pv["screen_seed"]))
        assert s1.html == s2.html and s1.template == pv["screen_template"]
        sp = pv["spec"]
        if sp.get("target"):
            assert s1.els[sp["target"]]["desc"] == sp.get("target_desc", s1.els[sp["target"]]["desc"])
        for m in sp.get("marks") or []:
            assert m["el"] in s1.els


def test_screens_questions_never_name_the_answer(scr):
    for r in scr:
        if r["field"]["type"] == "choice" and r["provenance"]["task"] in ("locate", "ground", "ground_goal", "tap_box"):
            q = r["field"]["question"]
            assert not re.search(r"\b(box|cell) [A-F]\d?\b", q)


# ------------------------------------------------------------------------------------------------ held-out
@pytest.mark.parametrize("train,ho", [("geo", "geo_ho"), ("scr", "scr_ho")])
def test_heldout_disjoint(train, ho):
    tr, h = rows(train), rows(ho)
    assert 200 <= len(h) <= 330
    assert all(r["provenance"]["heldout"] == "fresh" and "-hf1-" in r["id"] for r in h)
    assert not {r["images"][0] for r in tr} & {r["images"][0] for r in h}
    assert not {r["provenance"]["image_sha256"][0] for r in tr} & {r["provenance"]["image_sha256"][0] for r in h}
    if train == "geo":
        key = lambda r: json.dumps(r["provenance"]["spec"], sort_keys=True)  # noqa: E731
        assert not {key(r) for r in tr} & {key(r) for r in h}
    hid = {r["id"] for r in h}
    assert all(r["parent_id"] is None or r["parent_id"] in hid for r in h)


def test_leakage_gate_report():
    p = ROOT / "reports" / "phase3" / "leakage-gate-geometry-screens.md"
    if not p.is_file():
        pytest.skip("leakage report not written")
    txt = p.read_text()
    assert "only fixed-state links: yes" in txt
