"""Geometry templates for scripts/p3/gen_geometry.py.

Each template has
  sample(rng)            -> spec        (JSON-safe: every number the world holds, which givens are SHOWN, marks, the ask)
  build(spec, rng)       -> Built       (figure + question + state; rng only moves the drawing: rotation, jitter)
  solve(spec)            -> answer | None   (uses ONLY the shown givens and the marks; None = undetermined)
  choices(spec, rng)     -> option values incl. the answer (distractors are the common mistakes named in the plan)

spec["shown"] lists the givens the reader gets (on the figure or in the state: spec["where"][k] = "fig" | "state").
An unknown child is the parent spec with one decisive given (or mark) removed; solve() returns None for it.
Numbers: angles in degrees; lengths unitless or in spec["unit"]; answers are floats compared at 0.05 tolerance.
"""
from __future__ import annotations

import math
from fractions import Fraction

from fig import Fig, add, mul, rot, sub, unit

NAMES3 = ["ABC", "PQR", "XYZ", "DEF", "KLM", "RST", "UVW", "GHJ"]
NAMES4 = ["ABCD", "PQRS", "KLMN", "EFGH", "WXYZ", "JKLM"]
CANVAS = [(640, 480), (720, 540), (800, 600), (760, 520), (680, 510)]
NOTE = "Figure not drawn to scale"
TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41), (6, 8, 10), (9, 12, 15),
           (12, 16, 20), (15, 20, 25), (10, 24, 26), (12, 35, 37), (15, 36, 39), (16, 30, 34), (18, 24, 30), (21, 28, 35)]


# ------------------------------------------------------------------------------------------------ formatting
def is_int(v, tol=1e-9):
    return abs(v - round(v)) < tol


def num(v):
    if is_int(v):
        return str(int(round(v)))
    return f"{v:.1f}"


def deg(v):
    return f"{num(v)}°"


def length(v, unit=""):
    return f"{num(v)} {unit}".strip()


def pi_txt(k, unit=""):
    return f"{num(k)}π {unit}".strip()


def expr(a, b):
    if b == 0:
        return f"({a}x)°"
    return f"({a}x + {b})°" if b > 0 else f"({a}x − {-b})°"


def given_text(k, spec):
    """Text form of a given, e.g. 'm∠A = 52°' or 'AB = 7 cm'."""
    kind, label, v = spec["gtext"][k]
    u = spec.get("unit", "")
    if kind == "angle":
        return f"m∠{label} = {deg(v)}"
    return f"{label} = {length(v, u)}"


def state_for(spec, figure: str, extra: dict | None = None) -> dict:
    st = {"figure": figure}
    gs = [given_text(k, spec) for k in spec["shown"] if spec.get("where", {}).get(k) == "state"]
    if gs:
        st["given"] = "; ".join(gs)
    if extra:
        st.update(extra)
    return st


def on_fig(spec, k):
    return k in spec["shown"] and spec.get("where", {}).get(k, "fig") == "fig"


def pick_where(rng, keys, p_state=0.25):
    return {k: ("state" if rng.random() < p_state else "fig") for k in keys}


class Built(dict):
    pass


def built(fig, question, state, canvas, difficulty, note=True, fit=None, margin=None, levels=None):
    return Built(fig=fig, question=question, state=state, canvas=canvas, difficulty=difficulty, note=note, fit=fit,
                 margin=margin, levels=levels)


def orient(rng, pts: dict, big=False) -> dict:
    """Random rotation (and mirror) of a whole figure."""
    ang = rng.uniform(-math.pi, math.pi) if big else rng.uniform(-0.45, 0.45)
    mirror = rng.random() < 0.5
    out = {}
    for k, p in pts.items():
        q = (-p[0], p[1]) if mirror else p
        out[k] = rot(q, ang)
    return out


def centroid(ps):
    return (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))


def tri_from_angles(A, B):
    """Triangle with angle A at P0=(0,0), B at P1=(1,0), apex above."""
    C = 180 - A - B
    ac = math.sin(math.radians(B)) / math.sin(math.radians(C))
    return (0.0, 0.0), (1.0, 0.0), (ac * math.cos(math.radians(A)), ac * math.sin(math.radians(A)))


def tri_from_sides(a, b, c):
    """Vertices A, B, C with BC=a, CA=b, AB=c (A at origin, B on +x)."""
    x = (b * b + c * c - a * a) / (2 * c)
    y = math.sqrt(max(b * b - x * x, 1e-9))
    return (0.0, 0.0), (c, 0.0), (x, y)


def jitter_angles(rng, angs, amt=14):
    for _ in range(50):
        j = [a + rng.uniform(-amt, amt) for a in angs[:2]]
        if min(j) > 18 and 180 - sum(j) > 18:
            return j
    return angs[:2]


# ------------------------------------------------------------------------------------------------ T1 triangle angle sum
def s_tri_angle(rng):
    while True:
        A = rng.randint(25, 115)
        B = rng.randint(25, 130)
        if 180 - A - B >= 20:
            break
    names = rng.choice(NAMES3)
    ask = rng.randrange(3)
    shown = [f"a{i}" for i in range(3) if i != ask]
    return {"t": "tri_angle_sum", "names": names, "vals": {"a0": A, "a1": B, "a2": 180 - A - B}, "ask": f"a{ask}",
            "shown": shown, "where": pick_where(rng, shown), "x_label": rng.random() < 0.45,
            "gtext": {f"a{i}": ["angle", names[i], [A, B, 180 - A - B][i]] for i in range(3)}}


def b_tri_angle(spec, rng, child=False):
    v = spec["vals"]
    n = spec["names"]
    angs = [v["a0"], v["a1"], v["a2"]]
    draw = jitter_angles(rng, angs) if True else angs
    P = tri_from_angles(draw[0], draw[1])
    pts = orient(rng, {n[0]: P[0], n[1]: P[1], n[2]: P[2]})
    f = Fig()
    ps = [pts[c] for c in n]
    f.poly(ps)
    g = centroid(ps)
    for c in n:
        f.label(pts[c], c, away=g, dist=18)
    for i, c in enumerate(n):
        k = f"a{i}"
        o = [pts[x] for x in n if x != c]
        if on_fig(spec, k):
            f.angle(pts[c], o[0], o[1], deg(angs[i]))
        elif k == spec["ask"] and spec["x_label"]:
            f.angle(pts[c], o[0], o[1], "x°")
    ai = int(spec["ask"][1])
    q = "Find the value of x." if spec["x_label"] else f"Find m∠{n[ai]}."
    return built(f, q, state_for(spec, f"triangle {n}"), rng.choice(CANVAS), 2)


def v_tri_angle(spec):
    v = spec["vals"]
    return 180 - sum(v[k] for k in v if k != spec["ask"])


def c_tri_angle(spec, rng):
    v = spec["vals"]
    g = [v[k] for k in v if k != spec["ask"]]
    return [v_tri_angle(spec), g[0] + g[1], 180 - g[0], 180 - g[1], 360 - g[0] - g[1], 90 - g[0] / 2]


def solve_tri_angle(spec):
    if len(spec["shown"]) < 2:
        return None
    return 180 - sum(spec["vals"][k] for k in spec["shown"])


# ------------------------------------------------------------------------------------------------ T2 exterior angle
def s_tri_ext(rng):
    while True:
        A = rng.randint(25, 110)
        B = rng.randint(25, 110)
        if 180 - A - B >= 20:
            break
    names = rng.choice(NAMES4)
    kind = rng.choice(["ext", "rem"])
    shown = ["A", "B"] if kind == "ext" else ["A", "ext"]
    return {"t": "tri_exterior", "names": names, "kind": kind, "vals": {"A": A, "B": B, "ext": A + B},
            "ask": "ext" if kind == "ext" else "B", "shown": shown, "where": pick_where(rng, shown), "x_label": rng.random() < 0.4,
            "gtext": {"A": ["angle", names[0], A], "B": ["angle", names[1] if False else f"{names[0]}{names[1]}{names[2]}", B],
                      "ext": ["angle", f"{names[0]}{names[2]}{names[3]}", A + B]}}


def b_tri_ext(spec, rng, child=False):
    v = spec["vals"]
    n = spec["names"]  # apex n0, base n1 n2, extension point n3
    dA, dB = jitter_angles(rng, [v["A"], v["B"], 180 - v["A"] - v["B"]])
    # base B=(0,0), C=(1,0), apex from angle at B (dB) and at C (180-dA-dB)
    P = tri_from_angles(dB, 180 - dA - dB)
    Bp, Cp, Ap = P[0], P[1], P[2]
    Dp = (1.0 + 0.65, 0.0)
    pts = orient(rng, {"A": Ap, "B": Bp, "C": Cp, "D": Dp})
    f = Fig()
    f.poly([pts["A"], pts["B"], pts["C"]])
    f.seg(pts["C"], pts["D"])
    g = centroid([pts["A"], pts["B"], pts["C"]])
    f.label(pts["A"], n[0], away=g)
    f.label(pts["B"], n[1], away=g)
    f.label(pts["C"], n[2], away=add(pts["A"], mul(sub(pts["C"], pts["A"]), 0.5)), dist=20)
    f.label(pts["D"], n[3], away=pts["C"])
    xl = spec["x_label"]
    for k, (vtx, o1, o2) in {"A": ("A", "B", "C"), "B": ("B", "A", "C"), "ext": ("C", "A", "D")}.items():
        if on_fig(spec, k):
            f.angle(pts[vtx], pts[o1], pts[o2], deg(v[k]))
        elif k == spec["ask"] and xl:
            f.angle(pts[vtx], pts[o1], pts[o2], "x°")
    if spec["ask"] == "ext":
        q = "Find the value of x." if xl else f"Find m∠{n[0]}{n[2]}{n[3]}."
    else:
        q = "Find the value of x." if xl else f"Find m∠{n[0]}{n[1]}{n[2]}."
    st = state_for(spec, f"triangle {n[:3]}, side {n[1]}{n[2]} extended to {n[3]}")
    return built(f, q, st, rng.choice(CANVAS), 3)


def c_tri_ext(spec, rng):
    v = spec["vals"]
    if spec["kind"] == "ext":
        return [v["A"] + v["B"], 180 - v["A"] - v["B"], 180 - v["A"], 180 - v["B"], 360 - v["A"] - v["B"], abs(v["A"] - v["B"])]
    return [v["ext"] - v["A"], v["ext"] + v["A"], 180 - v["ext"], 180 - v["A"], v["A"], 180 - v["ext"] + v["A"]]


def solve_tri_ext(spec):
    v, s = spec["vals"], set(spec["shown"])
    if spec["kind"] == "ext":
        return v["A"] + v["B"] if {"A", "B"} <= s else None
    return v["ext"] - v["A"] if {"A", "ext"} <= s else None


# ------------------------------------------------------------------------------------------------ T3 isosceles
def s_isosceles(rng):
    apex = rng.randrange(20, 142, 2)
    kind = rng.choice(["base", "apex"])
    names = rng.choice(NAMES3)
    shown = ["apex"] if kind == "base" else ["base"]
    return {"t": "isosceles", "names": names, "kind": kind, "vals": {"apex": apex, "base": (180 - apex) // 2},
            "ask": "base" if kind == "base" else "apex", "shown": shown, "marks": {"ticks": True},
            "where": pick_where(rng, shown), "x_label": rng.random() < 0.4,
            "gtext": {"apex": ["angle", names[0], apex], "base": ["angle", names[1], (180 - apex) // 2]}}


def b_isosceles(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    ticks = spec["marks"]["ticks"]
    if ticks:
        b = v["base"]
        P = tri_from_angles(b, b)
    else:
        d = jitter_angles(rng, [v["base"], v["base"], v["apex"]], 16)
        while abs(d[0] - d[1]) < 8:
            d = jitter_angles(rng, [v["base"], v["base"], v["apex"]], 18)
        P = tri_from_angles(d[0], d[1])
    pts = orient(rng, {"B": P[0], "C": P[1], "A": P[2]})
    f = Fig()
    f.poly([pts["A"], pts["B"], pts["C"]])
    g = centroid(list(pts.values()))
    f.label(pts["A"], n[0], away=g)
    f.label(pts["B"], n[1], away=g)
    f.label(pts["C"], n[2], away=g)
    if ticks:
        f.tick(pts["A"], pts["B"])
        f.tick(pts["A"], pts["C"])
    xl = spec["x_label"]
    for k, (vtx, o1, o2) in {"apex": ("A", "B", "C"), "base": ("B", "A", "C")}.items():
        if on_fig(spec, k):
            f.angle(pts[vtx], pts[o1], pts[o2], deg(v[k]))
        elif k == spec["ask"] and xl:
            f.angle(pts[vtx], pts[o1], pts[o2], "x°")
    q = "Find the value of x." if xl else (f"Find m∠{n[1]}." if spec["ask"] == "base" else f"Find m∠{n[0]}.")
    return built(f, q, state_for(spec, f"triangle {n}"), rng.choice(CANVAS), 3)


def c_isosceles(spec, rng):
    v = spec["vals"]
    if spec["kind"] == "base":
        a = v["apex"]
        return [(180 - a) / 2, 180 - a, a / 2, 90 - a, a, 180 - 2 * a]
    b = v["base"]
    return [180 - 2 * b, 180 - b, b, 2 * b, 90 - b]


def solve_isosceles(spec):
    if not spec["marks"]["ticks"]:
        return None
    v = spec["vals"]
    if spec["kind"] == "base":
        return (180 - v["apex"]) / 2 if "apex" in spec["shown"] else None
    return 180 - 2 * v["base"] if "base" in spec["shown"] else None


# ------------------------------------------------------------------------------------------------ T4 parallel lines + transversal
POS = [("U", "UL"), ("U", "UR"), ("U", "LL"), ("U", "LR"), ("L", "UL"), ("L", "UR"), ("L", "LL"), ("L", "LR")]
RELATIONS = ["corresponding angles", "alternate interior angles", "alternate exterior angles", "same-side interior angles",
             "same-side exterior angles", "vertical angles", "a linear pair"]


def relation(p1, p2):
    (i1, q1), (i2, q2) = p1, p2
    if i1 == i2:
        return "vertical angles" if {q1, q2} in ({"UL", "LR"}, {"UR", "LL"}) else "a linear pair"
    if q1 == q2:
        return "corresponding angles"
    interior = lambda i, q: (i == "U" and q[0] == "L") or (i == "L" and q[0] == "U")  # noqa: E731
    side = lambda q: q[1]  # noqa: E731
    in1, in2 = interior(i1, q1), interior(i2, q2)
    if in1 and in2:
        return "alternate interior angles" if side(q1) != side(q2) else "same-side interior angles"
    if not in1 and not in2:
        return "alternate exterior angles" if side(q1) != side(q2) else "same-side exterior angles"
    return None


def s_parallel(rng, kind=None):
    kind = kind or rng.choices(["find", "relation", "equal"], weights=(45, 30, 25))[0]
    theta = rng.randint(35, 75)
    slope = rng.choice([1, -1])
    perm = list(range(8))
    shuffled = rng.random() < 0.5
    if shuffled:
        rng.shuffle(perm)
    # perm[k] = position index of the angle numbered k+1
    while True:
        if kind == "relation":
            want = rng.choice(RELATIONS)
            pairs = [(a, b) for a in range(8) for b in range(8) if a != b and relation(POS[perm[a]], POS[perm[b]]) == want]
            gi, aj = rng.choice(pairs)
        else:
            gi, aj = rng.sample(range(8), 2)
            if kind in ("find", "equal") and POS[perm[gi]][0] == POS[perm[aj]][0] and rng.random() < 0.7:
                continue  # mostly across the two intersections
        break
    val = lambda k: theta if (POS[perm[k]][1] in ("UR", "LL")) == (slope > 0) else 180 - theta  # noqa: E731
    shown = ["g"] if kind == "find" else []
    return {"t": "parallel", "kind": kind, "vals": {"theta": theta, "g": val(gi), "a": val(aj)}, "slope": slope, "perm": perm,
            "shuffled": shuffled, "gi": gi + 1, "aj": aj + 1, "marks": {"parallel": True}, "shown": shown,
            "where": pick_where(rng, shown, 0.35), "gtext": {"g": ["angle", str(gi + 1), val(gi)]}}


def b_parallel(spec, rng, child=False):
    th = math.radians(spec["vals"]["theta"])
    s = spec["slope"]
    par = spec["marks"]["parallel"]
    dx = 1 / math.tan(th)
    xu, xl = s * dx / 2, -s * dx / 2
    U, L = (xu, 1.0), (xl, 0.0)
    tdir = unit((xu - xl, 1.0))
    ext = 0.95
    T1, T2 = add(U, mul(tdir, ext)), sub(L, mul(tdir, ext))
    u1, u2 = (-2.3, 1.0), (2.3, 1.0)
    l1, l2 = (-2.3, 0.0), (2.3, 0.0)
    if not par:
        tilt = math.radians(rng.choice([-1, 1]) * rng.uniform(7, 11))
        l1, l2 = rot(l1, tilt, L), rot(l2, tilt, L)
    pts = orient(rng, {"U": U, "L": L, "T1": T1, "T2": T2, "u1": u1, "u2": u2, "l1": l1, "l2": l2})
    f = Fig()
    f.seg(pts["u1"], pts["u2"])
    f.seg(pts["l1"], pts["l2"])
    f.seg(pts["T1"], pts["T2"])
    if par:
        f.arrow(pts["u1"], pts["u2"], 1, at=0.88)
        f.arrow(pts["l1"], pts["l2"], 1, at=0.88)
    # quadrant bisectors at each intersection (screen-independent: use math directions, then the transform keeps angles)
    for k, pos_idx in enumerate(spec["perm"]):
        inter, quad = POS[pos_idx]
        c = pts[inter]
        right = unit(sub(pts["u2"], pts["u1"])) if inter == "U" else unit(sub(pts["l2"], pts["l1"]))
        up = unit(sub(pts["T1"], pts["T2"]))
        dirs = {"UR": (right, up), "UL": (up, mul(right, -1)), "LL": (mul(right, -1), mul(up, -1)), "LR": (mul(up, -1), right)}
        a, b = dirs[quad]
        bis = unit(add(a, b))
        f.label(add(c, mul(bis, 0.001)), str(k + 1), away=c, dist=24, size=19, italic=False, color="#1f5fbf")
    f.label(pts["u2"], "ℓ", away=pts["u1"], dist=14, size=22)
    f.label(pts["l2"], "m", away=pts["l1"], dist=14, size=22)
    f.label(pts["T1"], "t", away=pts["T2"], dist=14, size=22)
    kind = spec["kind"]
    g, a = spec["gi"], spec["aj"]
    extra_px = []
    if kind == "find" and on_fig(spec, "g"):
        extra_px.append(f"m∠{g} = {deg(spec['vals']['g'])}")
    if kind == "find":
        q = f"Find m∠{a}."
    elif kind == "relation":
        q = f"What kind of angle pair are ∠{g} and ∠{a}?"
    else:
        q = f"Is m∠{g} equal to m∠{a}?"
    st = state_for(spec, "lines ℓ and m cut by transversal t")
    b = built(f, q, st, rng.choice(CANVAS), 3 + (1 if spec["shuffled"] else 0), note=True)
    b["given_px"] = extra_px
    return b


def c_parallel(spec, rng):
    g = spec["vals"]["g"]
    ans = solve_parallel(dict(spec, marks={"parallel": True}))
    return [ans, 180 - ans, 90 - g if g < 90 else g - 90, g / 2, 360 - g, 180 - g / 2]


def _diag(pos_idx):
    return POS[pos_idx][1] in ("UR", "LL")


def solve_parallel(spec):
    p1, p2 = spec["perm"][spec["gi"] - 1], spec["perm"][spec["aj"] - 1]
    if spec["kind"] == "relation":
        return relation(POS[p1], POS[p2])
    same_inter = POS[p1][0] == POS[p2][0]
    if not spec["marks"]["parallel"] and not same_inter:
        return None
    eq = _diag(p1) == _diag(p2)
    if spec["kind"] == "equal":
        return eq
    if "g" not in spec["shown"]:
        return None
    g = spec["vals"]["g"]
    return g if eq else 180 - g


# ------------------------------------------------------------------------------------------------ T7 intersecting lines algebra
def s_lines_alg(rng):
    while True:
        x = rng.randint(4, 30)
        rel = rng.choice(["vertical", "linear"])
        a, c = rng.sample(range(2, 10), 2)
        V1 = rng.randint(30, 150)
        V2 = V1 if rel == "vertical" else 180 - V1
        b, d = V1 - a * x, V2 - c * x
        if abs(b) <= 120 and abs(d) <= 120 and V1 != 90:
            break
    q0 = rng.randrange(4)
    q1 = (q0 + 2) % 4 if rel == "vertical" else (q0 + rng.choice([1, 3])) % 4
    return {"t": "lines_algebra", "kind": rng.choices(["x", "angle"], weights=(70, 30))[0], "rel": rel, "quads": [q0, q1],
            "vals": {"x": x, "a": a, "b": b, "c": c, "d": d, "V1": V1, "V2": V2}, "shown": ["e1", "e2"], "where": {}}


def b_lines_alg(spec, rng, child=False):
    v = spec["vals"]
    q0, q1 = spec["quads"]
    # quadrant k lies between ray k and ray k+1; rays: 0 = +l1, 1 = +l2, 2 = -l1, 3 = -l2; angle between l1 and l2
    # is V1 for the quadrants of the first expression's parity
    ang_between = v["V1"] if q0 % 2 == 0 else 180 - v["V1"]
    base = rng.uniform(-0.5, 0.5)
    r0 = (math.cos(base), math.sin(base))
    r1 = (math.cos(base + math.radians(ang_between)), math.sin(base + math.radians(ang_between)))
    rays = [r0, r1, mul(r0, -1), mul(r1, -1)]
    Ln = 1.6
    f = Fig()
    f.seg(mul(rays[0], Ln), mul(rays[2], Ln))
    f.seg(mul(rays[1], Ln), mul(rays[3], Ln))
    f.dot((0.0, 0.0), 3)
    for qd, (a, b) in ((q0, (v["a"], v["b"])), (q1, (v["c"], v["d"]))):
        bis = unit(add(rays[qd], rays[(qd + 1) % 4]))
        f.label(mul(bis, 0.001), expr(a, b), away=(0.0, 0.0), dist=78, size=19, italic=False, color="#1f5fbf")
    f.extent += [(-1.9, -1.4), (1.9, 1.4)]
    q = "Find the value of x." if spec["kind"] == "x" else f"Find the measure of the angle marked {expr(v['a'], v['b'])}."
    return built(f, q, {"figure": "two intersecting lines"}, rng.choice(CANVAS), 4, note=False)


def c_lines_alg(spec, rng):
    v = spec["vals"]
    a, b, c, d = v["a"], v["b"], v["c"], v["d"]
    right = solve_lines_alg(spec)
    if spec["rel"] == "vertical":
        wrong_x = (180 - b - d) / (a + c)
    else:
        wrong_x = (d - b) / (a - c)
    if spec["kind"] == "x":
        return [right, wrong_x, v["V1"], right + 2, 180 / (a + c), abs(d - b)]
    return [right, a * wrong_x + b, v["x"], 180 - right, a * v["x"], right / 2]


def solve_lines_alg(spec):
    v = spec["vals"]
    a, b, c, d = v["a"], v["b"], v["c"], v["d"]
    q0, q1 = spec["quads"]
    vertical = (q0 - q1) % 4 == 2
    x = (d - b) / (a - c) if vertical else (180 - b - d) / (a + c)
    return x if spec["kind"] == "x" else a * x + b


# ------------------------------------------------------------------------------------------------ T8 Pythagoras
def s_pythag(rng):
    kind = rng.choice(["hyp", "leg"])
    if rng.random() < 0.6:
        a, b, c = rng.choice(TRIPLES)
        if rng.random() < 0.5:
            a, b = b, a
    else:
        while True:
            a, b = rng.randint(3, 20), rng.randint(3, 20)
            c = math.hypot(a, b)
            if not is_int(c) and abs(a - b) > 1:
                break
        if kind == "leg":
            c = math.ceil(c) + rng.randint(0, 3)
            b = math.sqrt(c * c - a * a)
    names = rng.choice(NAMES3)
    unit_ = rng.choice(["", "cm", "m", "in"])
    shown = ["a", "b"] if kind == "hyp" else ["a", "c"]
    ask = "c" if kind == "hyp" else "b"
    return {"t": "pythagoras", "kind": kind, "names": names, "unit": unit_, "vals": {"a": a, "b": b, "c": c}, "ask": ask,
            "shown": shown, "marks": {"right": True}, "where": pick_where(rng, shown), "x_label": rng.random() < 0.5,
            "gtext": {"a": ["side", names[1] + names[2], a], "b": ["side", names[0] + names[2], b], "c": ["side", names[0] + names[1], c]}}


def b_pythag(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    a, b = v["a"], v["b"]
    Cp, Bp, Ap = (0.0, 0.0), (a, 0.0), (0.0, b)
    if not spec["marks"]["right"]:
        ang = math.radians(90 + rng.choice([-1, 1]) * rng.uniform(13, 22))
        Ap = (b * math.cos(ang), b * math.sin(ang))
    pts = orient(rng, {"A": Ap, "B": Bp, "C": Cp}, big=rng.random() < 0.3)
    f = Fig()
    f.poly([pts["A"], pts["B"], pts["C"]])
    g = centroid(list(pts.values()))
    for k, c in zip("ABC", n):
        f.label(pts[k], c, away=g)
    if spec["marks"]["right"]:
        f.angle(pts["C"], pts["A"], pts["B"], right=True)
    sides = {"a": ("B", "C"), "b": ("A", "C"), "c": ("A", "B")}
    u = spec["unit"]
    for k, (p, q) in sides.items():
        m = mul(add(pts[p], pts[q]), 0.5)
        if on_fig(spec, k):
            f.label(m, length(v[k], u), away=g, dist=22, size=18, italic=False, color="#1f5fbf")
        elif k == spec["ask"] and spec["x_label"]:
            f.label(m, "x", away=g, dist=20, size=20)
    ask_name = spec["gtext"][spec["ask"]][1]
    rounding = not is_int(solve_pythag(dict(spec, marks={"right": True})))
    q = ("Find the value of x" if spec["x_label"] else f"Find {ask_name}") + (" to one decimal place." if rounding else ".")
    d = 3 if spec["kind"] == "hyp" else 4
    return built(f, q, state_for(spec, f"triangle {n}"), rng.choice(CANVAS), d)


def c_pythag(spec, rng):
    v = spec["vals"]
    a, b, c = v["a"], v["b"], v["c"]
    if spec["kind"] == "hyp":
        return [c, math.sqrt(abs(a * a - b * b)) if a != b else a + b + 1, a + b, a * a + b * b, (a + b) / 2 + 1]
    return [b, math.sqrt(c * c + a * a), c - a, c * c - a * a, (c + a) / 2]


def solve_pythag(spec):
    if not spec["marks"]["right"]:
        return None
    v, s = spec["vals"], set(spec["shown"])
    if spec["kind"] == "hyp":
        return math.hypot(v["a"], v["b"]) if {"a", "b"} <= s else None
    return math.sqrt(v["c"] ** 2 - v["a"] ** 2) if {"a", "c"} <= s else None


# ------------------------------------------------------------------------------------------------ T9 circle measures
def s_circle(rng):
    r = rng.randint(2, 15)
    given = rng.choice(["r", "d"])
    ask = rng.choice(["area", "circ", "diam" if given == "r" else "radius"])
    unit_ = rng.choice(["cm", "m", "in", ""])
    return {"t": "circle_measure", "given": given, "ask": ask, "unit": unit_, "vals": {"r": r, "d": 2 * r},
            "shown": [given], "where": pick_where(rng, [given], 0.3),
            "gtext": {"r": ["side", "OA", r], "d": ["side", "AB", 2 * r]}}


def b_circle(spec, rng, child=False):
    v = spec["vals"]
    f = Fig()
    f.circle((0.0, 0.0), 1.0)
    f.dot((0.0, 0.0))
    f.label((0.0, 0.0), "O", anchor_px=(-0.6, 0.8), dist=16)
    ang = rng.uniform(0, 2 * math.pi)
    A = (math.cos(ang), math.sin(ang))
    u = spec["unit"]
    if spec["given"] == "r":
        f.seg((0.0, 0.0), A)
        f.dot(A)
        f.label(A, "A", away=(0.0, 0.0))
        if on_fig(spec, "r"):
            f.label(mul(A, 0.5), length(v["r"], u), anchor_px=(-A[1], -A[0]), dist=16, size=18, italic=False, color="#1f5fbf")
    else:
        B = mul(A, -1)
        f.seg(B, A)
        f.dot(A)
        f.dot(B)
        f.label(A, "A", away=(0.0, 0.0))
        f.label(B, "B", away=(0.0, 0.0))
        if on_fig(spec, "d"):
            f.label(mul(A, 0.5), length(v["d"], u), anchor_px=(-A[1], -A[0]), dist=16, size=18, italic=False, color="#1f5fbf")
    q = {"area": "What is the area of the circle?", "circ": "What is the circumference of the circle?",
         "diam": "What is the diameter of the circle?", "radius": "What is the radius of the circle?"}[spec["ask"]]
    st = state_for(spec, "circle with centre O")
    d = 2 if spec["ask"] in ("diam", "radius") else 3
    return built(f, q, st, rng.choice(CANVAS), d)


def fmt_circle(spec):
    u = spec["unit"]
    if spec["ask"] == "area":
        return lambda k: pi_txt(k, (u + "²") if u else "")
    if spec["ask"] == "circ":
        return lambda k: pi_txt(k, u)
    return lambda k: length(k, u)


def c_circle(spec, rng):
    r = spec["vals"]["r"]
    d = 2 * r
    return {"area": [r * r, d * d, 2 * r, r, r * r / 2], "circ": [2 * r, r, r * r, 4 * r, d * d],
            "diam": [d, r / 2, r * r, 4 * r, r + 2], "radius": [r, 2 * d, d / 4, d * d, r + 3]}[spec["ask"]]


def solve_circle(spec):
    v = spec["vals"]
    if spec["given"] not in spec["shown"]:
        return None
    r = v["r"] if spec["given"] == "r" else v["d"] / 2
    return {"area": r * r, "circ": 2 * r, "diam": 2 * r, "radius": r}[spec["ask"]]   # area/circ in units of π


# ------------------------------------------------------------------------------------------------ T10 inscribed angle
def s_inscribed(rng):
    kind = rng.choices(["insc", "cent", "thales"], weights=(45, 35, 20))[0]
    theta = rng.randrange(40, 162, 2)
    names = rng.choice(["ABC", "PQR", "DEF", "JKL"])
    shown = {"insc": ["cent"], "cent": ["insc"], "thales": []}[kind]
    return {"t": "inscribed", "kind": kind, "names": names, "vals": {"cent": theta if kind != "thales" else 180,
                                                                     "insc": (theta if kind != "thales" else 180) // 2},
            "shown": shown, "where": pick_where(rng, shown, 0.3),
            "gtext": {"cent": ["angle", f"{names[0]}O{names[1]}", theta], "insc": ["angle", f"{names[0]}{names[2]}{names[1]}", theta // 2]}}


def b_inscribed(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    th = math.radians(v["cent"])
    a0 = rng.uniform(0, 2 * math.pi)
    A = (math.cos(a0), math.sin(a0))
    B = (math.cos(a0 + th), math.sin(a0 + th))
    cm = a0 + th + (2 * math.pi - th) * rng.uniform(0.35, 0.65)
    C = (math.cos(cm), math.sin(cm))
    O = (0.0, 0.0)
    f = Fig()
    f.circle(O, 1.0)
    for p in (A, B, C, O):
        f.dot(p)
    f.label(O, "O", anchor_px=(0.7, 0.7) if spec["kind"] != "thales" else (-A[1], A[0]), dist=16)
    for p, c in zip((A, B, C), n):
        f.label(p, c, away=O)
    f.seg(C, A)
    f.seg(C, B)
    if spec["kind"] == "thales":
        f.seg(A, B)
    else:
        f.seg(O, A)
        f.seg(O, B)
    if on_fig(spec, "cent"):
        f.angle(O, A, B, deg(v["cent"]), r=22)
    if on_fig(spec, "insc"):
        f.angle(C, A, B, deg(v["insc"]), r=30)
    q = {"insc": f"Find m∠{n[0]}{n[2]}{n[1]}.", "cent": f"Find m∠{n[0]}O{n[1]}.", "thales": f"Find m∠{n[0]}{n[2]}{n[1]}."}[spec["kind"]]
    extra = {"diameter": f"{n[0]}{n[1]} is a diameter"} if spec["kind"] == "thales" else None
    st = state_for(spec, "circle with centre O", extra)
    return built(f, q, st, rng.choice(CANVAS), 3 if spec["kind"] != "thales" else 3)


def c_inscribed(spec, rng):
    v = spec["vals"]
    if spec["kind"] == "insc":
        c = v["cent"]
        return [c / 2, c, 2 * c if 2 * c < 360 else 360 - c, 180 - c / 2, 90 - c / 2]
    if spec["kind"] == "cent":
        i = v["insc"]
        return [2 * i, i, i / 2, 180 - i, 180 - 2 * i if 180 - 2 * i > 0 else 360 - 2 * i]
    return [90, 45, 180, 60, 120]


def solve_inscribed(spec):
    v, s = spec["vals"], set(spec["shown"])
    if spec["kind"] == "thales":
        return 90
    if spec["kind"] == "insc":
        return v["cent"] / 2 if "cent" in s else None
    return 2 * v["insc"] if "insc" in s else None


# ------------------------------------------------------------------------------------------------ T11 tangent
def s_tangent(rng):
    kind = rng.choices(["pt", "op", "ang"], weights=(40, 30, 30))[0]
    r, t, d = rng.choice(TRIPLES)
    if rng.random() < 0.5:
        r, t = t, r
    alpha = rng.randint(20, 70)
    shown = {"pt": ["r", "d"], "op": ["r", "t"], "ang": ["alpha"]}[kind]
    return {"t": "tangent", "kind": kind, "unit": rng.choice(["", "cm", "m"]), "vals": {"r": r, "t": t, "d": d, "alpha": alpha},
            "shown": shown, "where": pick_where(rng, shown, 0.3),
            "gtext": {"r": ["side", "OT", r], "t": ["side", "PT", t], "d": ["side", "OP", d], "alpha": ["angle", "OPT", alpha]}}


def b_tangent(spec, rng, child=False):
    v = spec["vals"]
    if spec["kind"] == "ang":
        a = math.radians(v["alpha"])
        r, t = 1.0, 1.0 / math.tan(a)
    else:
        r, t = float(v["r"]), float(v["t"])
    O = (0.0, 0.0)
    T = (0.0, r)
    P = (t, r)
    pts = orient(rng, {"O": O, "T": T, "P": P, "X": (-0.35 * t, r)}, big=True)
    f = Fig()
    f.circle(pts["O"], r)
    for k in ("O", "T", "P"):
        f.dot(pts[k])
    f.seg(pts["O"], pts["T"])
    f.seg(pts["O"], pts["P"])
    f.seg(pts["X"], pts["P"])
    g = centroid([pts["O"], pts["T"], pts["P"]])
    f.label(pts["O"], "O", away=g, dist=18)
    f.label(pts["T"], "T", away=g, dist=18)
    f.label(pts["P"], "P", away=g, dist=18)
    u = spec["unit"]
    segs = {"r": ("O", "T"), "t": ("T", "P"), "d": ("O", "P")}
    for k, (p, q) in segs.items():
        if on_fig(spec, k):
            f.label(mul(add(pts[p], pts[q]), 0.5), length(v[k], u), away=g, dist=20, size=18, italic=False, color="#1f5fbf")
    if on_fig(spec, "alpha"):
        f.angle(pts["P"], pts["O"], pts["T"], deg(v["alpha"]), r=40)
    q = {"pt": "Find PT.", "op": "Find OP.", "ang": "Find m∠POT."}[spec["kind"]]
    st = state_for(spec, "circle with centre O", {"tangent": "PT is tangent to the circle at T"})
    return built(f, q, st, rng.choice(CANVAS), 4)


def c_tangent(spec, rng):
    v = spec["vals"]
    r, t, d, a = v["r"], v["t"], v["d"], v["alpha"]
    return {"pt": [t, d - r, math.hypot(d, r), d + r, d * d - r * r],
            "op": [d, r + t, abs(t - r) if t != r else t + 1, t * t + r * r, math.sqrt(abs(t * t - r * r)) + 0.5],
            "ang": [90 - a, 180 - a, a, 90 + a, 180 - 2 * a]}[spec["kind"]]


def solve_tangent(spec):
    v, s = spec["vals"], set(spec["shown"])
    k = spec["kind"]
    if k == "pt":
        return math.sqrt(v["d"] ** 2 - v["r"] ** 2) if {"r", "d"} <= s else None
    if k == "op":
        return math.hypot(v["r"], v["t"]) if {"r", "t"} <= s else None
    return 90 - v["alpha"] if "alpha" in s else None


# ------------------------------------------------------------------------------------------------ T12 parallelogram
def s_parallelogram(rng):
    kind = rng.choices(["angle", "area", "perim"], weights=(40, 35, 25))[0]
    names = rng.choice(NAMES4)
    if kind == "angle":
        a = rng.choice([x for x in range(40, 141) if abs(x - 90) > 8])
        ask = rng.choice(["B", "C", "D"])
        vals = {"A": a, "B": 180 - a, "C": a, "D": 180 - a}
        shown = ["A"]
        gt = {"A": ["angle", names[0], a]}
    else:
        while True:
            b = rng.randint(5, 20)
            h = rng.randint(3, 14)
            s = rng.randint(h + 1, h + 8)
            if s * math.cos(math.asin(h / s)) < b - 1:
                break
        vals = {"b": b, "h": h, "s": s}
        ask = kind
        shown = ["b", "h", "s"] if kind == "area" else ["b", "s"]
        gt = {"b": ["side", names[0] + names[1], b], "h": ["side", "h", h], "s": ["side", names[0] + names[3], s]}
    return {"t": "parallelogram", "kind": kind, "names": names, "unit": rng.choice(["", "cm", "m"]), "vals": vals, "ask": ask,
            "shown": shown, "marks": {"parallel": True}, "where": pick_where(rng, [k for k in shown if k != "h"]), "gtext": gt,
            "x_label": rng.random() < 0.4}


def b_parallelogram(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    par = spec["marks"]["parallel"]
    if spec["kind"] == "angle":
        a = math.radians(v["A"])
        b, s = rng.uniform(1.4, 2.2), rng.uniform(0.9, 1.3)
    else:
        a = math.asin(v["h"] / v["s"])
        b, s = float(v["b"]), float(v["s"])
    A, B = (0.0, 0.0), (b, 0.0)
    D = (s * math.cos(a), s * math.sin(a))
    C = add(B, D)
    if not par:
        C = add(C, (rng.choice([-1, 1]) * rng.uniform(0.25, 0.4) * b, rng.uniform(0.2, 0.35) * s))
    pts = orient(rng, {"A": A, "B": B, "C": C, "D": D, "F": (D[0], 0.0)})
    f = Fig()
    f.poly([pts[k] for k in "ABCD"])
    g = centroid([pts[k] for k in "ABCD"])
    for k, c in zip("ABCD", n):
        f.label(pts[k], c, away=g)
    if par:
        f.arrow(pts["A"], pts["B"], 1)
        f.arrow(pts["D"], pts["C"], 1)
        f.arrow(pts["A"], pts["D"], 2)
        f.arrow(pts["B"], pts["C"], 2)
    u = spec["unit"]
    if spec["kind"] == "angle":
        if on_fig(spec, "A"):
            f.angle(pts["A"], pts["B"], pts["D"], deg(v["A"]))
        if spec["x_label"]:
            k = spec["ask"]
            nb = {"B": ("A", "C"), "C": ("B", "D"), "D": ("A", "C")}[k]
            f.angle(pts[k], pts[nb[0]], pts[nb[1]], "x°")
        q = "Find the value of x." if spec["x_label"] else f"Find m∠{n['ABCD'.index(spec['ask'])]}."
        d = 3
    else:
        if on_fig(spec, "b"):
            fr = D[0] / b
            tb = 0.5 if abs(fr - 0.5) > 0.22 or "h" not in spec["shown"] else (0.22 if fr > 0.5 else 0.78)
            f.label(add(pts["A"], mul(sub(pts["B"], pts["A"]), tb)), length(v["b"], u), away=g, dist=20, size=18, italic=False, color="#1f5fbf")
        if on_fig(spec, "s"):
            f.label(mul(add(pts["A"], pts["D"]), 0.5), length(v["s"], u), away=g, dist=22, size=18, italic=False, color="#1f5fbf")
        if "h" in spec["shown"]:
            f.seg(pts["D"], pts["F"], dash=True, w=1.8)
            f.angle(pts["F"], pts["D"], pts["B"], right=True)
            f.label(mul(add(pts["D"], pts["F"]), 0.5), length(v["h"], u), anchor_px=sub(pts["B"], pts["A"]) if False else None,
                    away=pts["A"], dist=20, size=18, italic=False, color="#1f5fbf")
        q = f"Find the area of {n}." if spec["kind"] == "area" else f"Find the perimeter of {n}."
        d = 4 if spec["kind"] == "area" else 3
    return built(f, q, state_for(spec, f"quadrilateral {n}"), rng.choice(CANVAS), d)


def c_parallelogram(spec, rng):
    v = spec["vals"]
    if spec["kind"] == "angle":
        a = v["A"]
        right = solve_parallelogram(dict(spec, marks={"parallel": True}))
        return [right, 180 - right, 360 - a, abs(90 - a), a / 2]
    b, h, s = v["b"], v["h"], v["s"]
    if spec["kind"] == "area":
        return [b * h, b * s, b * h / 2, 2 * (b + s), s * h]
    return [2 * (b + s), b + s, b * s, 2 * b + s, 4 * b]


def solve_parallelogram(spec):
    v, sh = spec["vals"], set(spec["shown"])
    k = spec["kind"]
    if k == "angle":
        if not spec["marks"]["parallel"] or "A" not in sh:
            return None
        return v["A"] if spec["ask"] == "C" else 180 - v["A"]
    if k == "area":
        return v["b"] * v["h"] if spec["marks"]["parallel"] and {"b", "h"} <= sh else None
    return 2 * (v["b"] + v["s"]) if spec["marks"]["parallel"] and {"b", "s"} <= sh else None


# ------------------------------------------------------------------------------------------------ T13 regular polygon
POLY_NAMES = {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon", 9: "nonagon",
              10: "decagon", 11: "11-gon", 12: "12-gon"}


def s_regular(rng):
    n = rng.randint(5, 10)
    return {"t": "regular_polygon", "kind": rng.choice(["int", "ext", "sum"]), "vals": {"n": n}, "shown": ["n_by_figure"],
            "where": {}}


def b_regular(spec, rng, child=False):
    n = spec["vals"]["n"]
    a0 = rng.uniform(0, 2 * math.pi)
    ps = [(math.cos(a0 + 2 * math.pi * i / n), math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]
    f = Fig()
    f.poly(ps)
    for i in range(n):
        f.tick(ps[i], ps[(i + 1) % n])
    q = {"int": "What is the measure of each interior angle?", "ext": "What is the measure of each exterior angle?",
         "sum": "What is the sum of the interior angles?"}[spec["kind"]]
    return built(f, q, {"figure": "a regular polygon"}, rng.choice(CANVAS), 4, note=False)


def c_regular(spec, rng):
    n = spec["vals"]["n"]
    fi = lambda m: (m - 2) * 180 / m  # noqa: E731
    fe = lambda m: 360 / m  # noqa: E731
    fs = lambda m: (m - 2) * 180  # noqa: E731
    main = {"int": fi, "ext": fe, "sum": fs}[spec["kind"]]
    other = {"int": [fe(n), fs(n) / 10 if False else 180 - fi(n) / 2], "ext": [fi(n), 180 / n], "sum": [n * 180, fs(n) + 180]}[spec["kind"]]
    return [main(n), main(n - 1), main(n + 1)] + other


def solve_regular(spec):
    n = spec["vals"]["n"]
    return {"int": (n - 2) * 180 / n, "ext": 360 / n, "sum": (n - 2) * 180}[spec["kind"]]


# ------------------------------------------------------------------------------------------------ T14 polygon missing angle
def s_poly_missing(rng):
    n = rng.choice([4, 4, 5, 5, 6])
    total = (n - 2) * 180
    lo, hi = (55, 150) if n == 4 else (75, 165)
    while True:
        angs = [rng.randint(lo, hi) for _ in range(n - 1)]
        last = total - sum(angs)
        if lo <= last <= hi:
            break
    angs.append(last)
    rng.shuffle(angs)
    ask = rng.randrange(n)
    names = "ABCDEF"[:n] if rng.random() < 0.5 else "PQRSTU"[:n]
    shown = [f"a{i}" for i in range(n) if i != ask]
    return {"t": "polygon_missing", "names": names, "vals": {f"a{i}": angs[i] for i in range(n)}, "n": n, "ask": f"a{ask}",
            "shown": shown, "where": pick_where(rng, shown, 0.15),
            "gtext": {f"a{i}": ["angle", names[i], angs[i]] for i in range(n)}}


def b_poly_missing(spec, rng, child=False):
    n = spec["n"]
    while True:
        cuts = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
        gaps = [(cuts[(i + 1) % n] - cuts[i]) % (2 * math.pi) for i in range(n)]
        if min(gaps) > 2 * math.pi / n * 0.55:
            break
    ex, ey = rng.uniform(1.0, 1.5), 1.0
    ps = [(ex * math.cos(c), ey * math.sin(c)) for c in cuts]
    f = Fig()
    f.poly(ps)
    g = centroid(ps)
    names = spec["names"]
    for i in range(n):
        f.label(ps[i], names[i], away=g, dist=18)
        k = f"a{i}"
        pr, nx = ps[i - 1], ps[(i + 1) % n]
        if on_fig(spec, k):
            f.angle(ps[i], pr, nx, deg(spec["vals"][k]), r=24)
        elif k == spec["ask"]:
            f.angle(ps[i], pr, nx, "x°", r=24)
    kind = POLY_NAMES[n]
    return built(f, "Find the value of x.", state_for(spec, f"{kind} {names}"), rng.choice(CANVAS), 4)


def c_poly_missing(spec, rng):
    n = spec["n"]
    others = sum(v for k, v in spec["vals"].items() if k != spec["ask"])
    return [(n - 2) * 180 - others, (n - 3) * 180 - others, (n - 1) * 180 - others, 360 - others, n * 180 - others,
            180 - ((n - 2) * 180 - others)]


def solve_poly_missing(spec):
    n = spec["n"]
    if len(spec["shown"]) < n - 1:
        return None
    return (n - 2) * 180 - sum(spec["vals"][k] for k in spec["shown"])


# ------------------------------------------------------------------------------------------------ T15 area / perimeter
def s_area(rng):
    shape = rng.choices(["rect", "tri", "trap"], weights=(30, 35, 35))[0]
    unit_ = rng.choice(["", "cm", "m", "in"])
    if shape == "rect":
        w, h = rng.randint(3, 20), rng.randint(2, 15)
        while w == h:
            h = rng.randint(2, 15)
        kind = rng.choice(["area", "perim"])
        vals, shown = {"w": w, "h": h}, ["w", "h"]
    elif shape == "tri":
        b, h = rng.randint(4, 20), rng.randint(3, 16)
        foot = rng.randint(1, b - 1)
        s = math.hypot(foot, h)
        kind = "area"
        vals, shown = {"b": b, "h": h, "foot": foot, "s": round(s, 1)}, ["b", "h", "s"]
    else:
        a = rng.randint(8, 22)
        b = rng.randint(3, a - 2)
        h = rng.randint(3, 14)
        kind = "area"
        vals, shown = {"a": a, "b": b, "h": h}, ["a", "b", "h"]
    return {"t": "area_shape", "shape": shape, "kind": kind, "unit": unit_, "vals": vals, "shown": shown,
            "names": rng.choice(NAMES4) if shape != "tri" else rng.choice(NAMES3), "where": pick_where(rng, [k for k in shown if k != "h"], 0.2),
            "gtext": {"w": ["side", "width", vals.get("w")], "h": ["side", "height", vals.get("h")], "b": ["side", "base", vals.get("b")],
                      "s": ["side", "side", vals.get("s")], "a": ["side", "base", vals.get("a")]}}


def b_area(spec, rng, child=False):
    v, n, u = spec["vals"], spec["names"], spec["unit"]
    f = Fig()
    lab = lambda p, q, text, away, d=20: f.label(mul(add(p, q), 0.5), text, away=away, dist=d, size=18, italic=False, color="#1f5fbf")  # noqa: E731
    if spec["shape"] == "rect":
        w, h = v["w"], v["h"]
        P = {"A": (0.0, 0.0), "B": (w, 0.0), "C": (w, h), "D": (0.0, h)}
        pts = orient(rng, P)
        f.poly([pts[k] for k in "ABCD"])
        g = centroid(list(pts.values()))
        for k, c in zip("ABCD", n):
            f.label(pts[k], c, away=g)
        for k, (p, q) in {"A": ("B", "D"), "B": ("A", "C"), "C": ("B", "D"), "D": ("A", "C")}.items():
            f.angle(pts[k], pts[p], pts[q], right=True)
        if on_fig(spec, "w"):
            lab(pts["A"], pts["B"], length(w, u), g)
        if on_fig(spec, "h"):
            lab(pts["B"], pts["C"], length(h, u), g)
        fig_name = f"rectangle {n}"
        q = f"Find the area of {n}." if spec["kind"] == "area" else f"Find the perimeter of {n}."
        d = 2
    elif spec["shape"] == "tri":
        b, h, foot = v["b"], v["h"], v["foot"]
        P = {"A": (0.0, 0.0), "B": (float(b), 0.0), "C": (float(foot), float(h)), "F": (float(foot), 0.0)}
        pts = orient(rng, P)
        f.poly([pts[k] for k in "ABC"])
        g = centroid([pts[k] for k in "ABC"])
        for k, c in zip("ABC", n):
            f.label(pts[k], c, away=g)
        f.seg(pts["C"], pts["F"], dash=True, w=1.8)
        if "h" in spec["shown"]:
            f.angle(pts["F"], pts["C"], pts["B"], right=True)
            f.label(mul(add(pts["C"], pts["F"]), 0.5), length(h, u), away=pts["A"], dist=20, size=18, italic=False, color="#1f5fbf")
        if on_fig(spec, "b"):
            lab(pts["A"], pts["B"], length(b, u), g)
        if on_fig(spec, "s"):
            lab(pts["A"], pts["C"], length(v["s"], u), g, 24)
        fig_name = f"triangle {n}"
        q = f"Find the area of triangle {n}."
        d = 3
    else:
        a, b, h = v["a"], v["b"], v["h"]
        off = rng.uniform(0.1, 0.9) * (a - b)
        P = {"A": (0.0, 0.0), "B": (float(a), 0.0), "C": (off + b, float(h)), "D": (off, float(h)), "F": (off, 0.0)}
        pts = orient(rng, P)
        f.poly([pts[k] for k in "ABCD"])
        g = centroid([pts[k] for k in "ABCD"])
        for k, c in zip("ABCD", n):
            f.label(pts[k], c, away=g)
        f.arrow(pts["A"], pts["B"], 1)
        f.arrow(pts["D"], pts["C"], 1)
        f.seg(pts["D"], pts["F"], dash=True, w=1.8)
        if "h" in spec["shown"]:
            f.angle(pts["F"], pts["D"], pts["B"], right=True)
            f.label(mul(add(pts["D"], pts["F"]), 0.5), length(h, u), away=pts["A"], dist=20, size=18, italic=False, color="#1f5fbf")
        if on_fig(spec, "a"):
            lab(pts["A"], pts["B"], length(a, u), g)
        if on_fig(spec, "b"):
            lab(pts["D"], pts["C"], length(b, u), g)
        fig_name = f"trapezoid {n}"
        q = f"Find the area of trapezoid {n}."
        d = 4
    return built(f, q, state_for(spec, fig_name), rng.choice(CANVAS), d)


def c_area(spec, rng):
    v = spec["vals"]
    if spec["shape"] == "rect":
        w, h = v["w"], v["h"]
        return [w * h, 2 * (w + h), w + h, 2 * w * h, 2 * w + h] if spec["kind"] == "area" else [2 * (w + h), w * h, w + h, 2 * w + h, 4 * w]
    if spec["shape"] == "tri":
        b, h, s = v["b"], v["h"], v["s"]
        return [b * h / 2, b * h, b * s / 2, (b + h) / 2 * 2, b * s]
    a, b, h = v["a"], v["b"], v["h"]
    return [(a + b) * h / 2, (a + b) * h, a * b * h / 2, a * h, (a - b) * h / 2]


def solve_area(spec):
    v, s = spec["vals"], set(spec["shown"])
    if spec["shape"] == "rect":
        if not {"w", "h"} <= s:
            return None
        return v["w"] * v["h"] if spec["kind"] == "area" else 2 * (v["w"] + v["h"])
    if spec["shape"] == "tri":
        return v["b"] * v["h"] / 2 if {"b", "h"} <= s else None
    return (v["a"] + v["b"]) * v["h"] / 2 if {"a", "b", "h"} <= s else None


# ------------------------------------------------------------------------------------------------ T16 coordinate grid (to scale)
def s_coord(rng):
    kind = rng.choices(["dist", "slope", "mid", "area", "right", "para", "quad"], weights=(18, 16, 14, 16, 14, 10, 12))[0]
    signed = kind == "quad" or rng.random() < 0.5
    lo, hi = (-6, 6) if signed else (0, 10)
    R = lambda: (rng.randint(lo, hi), rng.randint(lo, hi))  # noqa: E731
    pts = {}
    if kind in ("dist", "slope", "mid"):
        while True:
            A, B = R(), R()
            dx, dy = B[0] - A[0], B[1] - A[1]
            if dx and dy and abs(dx) != abs(dy) and math.hypot(dx, dy) >= 3:
                break
        pts = {"A": A, "B": B}
    elif kind == "area":
        while True:
            A, B, C = R(), R(), R()
            ar = abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / 2
            if ar >= 4:
                break
        pts = {"A": A, "B": B, "C": C}
    elif kind == "right":
        want = rng.random() < 0.5
        while True:
            A = R()
            u = (rng.randint(-5, 5), rng.randint(-5, 5))
            if u == (0, 0):
                continue
            if want:
                k = rng.choice([1, 1, 2]) if max(abs(u[0]), abs(u[1])) <= 3 else 1
                w = (-u[1] * k, u[0] * k) if rng.random() < 0.5 else (u[1] * k, -u[0] * k)
            else:
                w = (rng.randint(-5, 5), rng.randint(-5, 5))
            B, C = (A[0] + u[0], A[1] + u[1]), (A[0] + w[0], A[1] + w[1])
            if not all(lo <= p[0] <= hi and lo <= p[1] <= hi for p in (B, C)):
                continue
            ar = abs(u[0] * w[1] - u[1] * w[0]) / 2
            if ar < 3:
                continue
            if coord_right((A, B, C)) != want:
                continue
            break
        names = ["A", "B", "C"]
        rng.shuffle(names)
        pts = dict(zip(names, (A, B, C)))
    elif kind == "para":
        want = rng.random() < 0.5
        while True:
            A, B, C = R(), R(), R()
            D = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])
            if not want:
                D = (D[0] + rng.choice([-1, 1]), D[1] + rng.choice([-1, 0, 1]))
            if not (lo <= D[0] <= hi and lo <= D[1] <= hi):
                continue
            ar = abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / 2
            if ar < 5 or not convex_order((A, B, C, D)):
                continue
            if coord_para((A, B, C, D)) != want:
                continue
            break
        pts = {"A": A, "B": B, "C": C, "D": D}
    else:
        while True:
            P = R()
            if P[0] and P[1]:
                break
        pts = {"P": P}
    return {"t": "coord", "kind": kind, "range": [lo, hi], "pts": {k: list(v) for k, v in pts.items()}, "shown": ["grid"], "where": {}}


def convex_order(ps):
    n = len(ps)
    s = []
    for i in range(n):
        a, b, c = ps[i], ps[(i + 1) % n], ps[(i + 2) % n]
        s.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(x > 0 for x in s) or all(x < 0 for x in s)


def coord_right(ps):
    for i in range(3):
        a, b, c = ps[i], ps[(i + 1) % 3], ps[(i + 2) % 3]
        if (b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]) == 0:
            return True
    return False


def coord_para(ps):
    A, B, C, D = ps
    return A[0] + C[0] == B[0] + D[0] and A[1] + C[1] == B[1] + D[1]


def b_coord(spec, rng, child=False):
    lo, hi = spec["range"]
    f = Fig()
    for i in range(lo, hi + 1):
        f.seg((i, lo - 0.4), (i, hi + 0.4), w=1, color="#d5d9e0")
        f.seg((lo - 0.4, i), (hi + 0.4, i), w=1, color="#d5d9e0")
    ax0 = 0 if lo < 0 else lo
    f.seg((lo - 0.5, ax0), (hi + 0.5, ax0), w=1.8, color="#333")
    f.seg((ax0, lo - 0.5), (ax0, hi + 0.5), w=1.8, color="#333")
    f.label((hi + 0.5, ax0), "x", anchor_px=(1, 0), dist=12, size=18)
    f.label((ax0, hi + 0.5), "y", anchor_px=(0, -1), dist=12, size=18)
    for i in range(lo, hi + 1):
        if i == 0 and lo < 0:
            f.label((0, 0), "0", anchor_px=(-1, 1), dist=12, size=13, italic=False, color="#555")
            continue
        f.label((i, ax0), str(i), anchor_px=(0, 1), dist=13, size=13, italic=False, color="#555")
        if not (i == lo and lo == 0):
            f.label((ax0, i), str(i), anchor_px=(-1, 0), dist=14, size=13, italic=False, color="#555")
    P = {k: tuple(v) for k, v in spec["pts"].items()}
    if spec["kind"] in ("area", "right"):
        ks = sorted(P)
        f.poly([P[k] for k in ks], w=2.2)
    elif spec["kind"] == "para":
        f.poly([P[k] for k in "ABCD"], w=2.2)
    elif spec["kind"] in ("dist", "slope", "mid"):
        f.seg(P["A"], P["B"], w=2.2)
    g = centroid(list(P.values()))
    for k, p in P.items():
        f.dot(p, 5)
        away = g if len(P) > 1 else (p[0] - 1, p[1] - 1)
        f.label(p, k, away=away if away != p else (p[0] - 1, p[1] - 1), dist=17, size=21, color="#b0261c")
    kind = spec["kind"]
    ans = solve_coord(spec)
    rnd = kind == "dist" and not is_int(ans)
    q = {"dist": "What is the distance from A to B" + (", to one decimal place?" if rnd else "?"),
         "slope": "What is the slope of line AB?", "mid": "What is the midpoint of segment AB?",
         "area": "What is the area of triangle ABC?", "right": "Is triangle ABC a right triangle?",
         "para": "Is ABCD a parallelogram?", "quad": "In which quadrant is point P?"}[kind]
    d = {"dist": 3, "slope": 3, "mid": 3, "area": 4, "right": 4, "para": 5, "quad": 2}[kind]
    W = rng.choice([620, 660, 700])
    fit = (lo - 0.9, lo - 0.9, hi + 0.9, hi + 0.9)
    return built(f, q, {"figure": "points on a coordinate grid"}, (W, W), d, note=False, fit=fit, margin=(20, 20, 20, 20))


def frac_txt(fr: Fraction):
    if fr.denominator == 1:
        return str(fr.numerator).replace("-", "−")
    return f"{'−' if fr < 0 else ''}{abs(fr.numerator)}/{fr.denominator}"


def pt_txt(p):
    return f"({num(p[0])}, {num(p[1])})".replace("-", "−")


def c_coord(spec, rng):
    P = {k: tuple(v) for k, v in spec["pts"].items()}
    k = spec["kind"]
    if k == "dist":
        dx, dy = abs(P["B"][0] - P["A"][0]), abs(P["B"][1] - P["A"][1])
        return [math.hypot(dx, dy), dx + dy, dx * dx + dy * dy, math.sqrt(abs(dx * dx - dy * dy)) if dx != dy else dx + 1, math.hypot(dx + 1, dy)]
    if k == "slope":
        dx, dy = P["B"][0] - P["A"][0], P["B"][1] - P["A"][1]
        s = Fraction(dy, dx)
        return [s, Fraction(dx, dy), -s, -Fraction(dx, dy), s + 1]
    if k == "mid":
        (x1, y1), (x2, y2) = P["A"], P["B"]
        m = ((x1 + x2) / 2, (y1 + y2) / 2)
        return [m, (m[1], m[0]), ((x2 - x1) / 2, (y2 - y1) / 2), (x1 + x2, y1 + y2), (m[0] + 1, m[1])]
    if k == "area":
        (x1, y1), (x2, y2), (x3, y3) = P["A"], P["B"], P["C"]
        a = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        bb = (max(x1, x2, x3) - min(x1, x2, x3)) * (max(y1, y2, y3) - min(y1, y2, y3))
        return [a, 2 * a, bb, a + 2, bb / 2 if bb / 2 != a else a + 1]
    if k == "quad":
        return ["Quadrant I", "Quadrant II", "Quadrant III", "Quadrant IV"]
    return None


def solve_coord(spec):
    P = {k: tuple(v) for k, v in spec["pts"].items()}
    k = spec["kind"]
    if k == "dist":
        return math.hypot(P["B"][0] - P["A"][0], P["B"][1] - P["A"][1])
    if k == "slope":
        return Fraction(P["B"][1] - P["A"][1], P["B"][0] - P["A"][0])
    if k == "mid":
        return ((P["A"][0] + P["B"][0]) / 2, (P["A"][1] + P["B"][1]) / 2)
    if k == "area":
        (x1, y1), (x2, y2), (x3, y3) = P["A"], P["B"], P["C"]
        return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
    if k == "right":
        return coord_right([P["A"], P["B"], P["C"]])
    if k == "para":
        return coord_para([P[c] for c in "ABCD"])
    x, y = P["P"]
    return "Quadrant " + ("I" if x > 0 and y > 0 else "II" if x < 0 < y else "III" if x < 0 and y < 0 else "IV")


# ------------------------------------------------------------------------------------------------ T17 classify triangles
def s_classify(rng):
    kind = rng.choices(["iso", "right", "type"], weights=(35, 30, 35))[0]
    names = rng.choice(NAMES3)
    if kind == "iso":
        while True:
            if rng.random() < 0.5:
                e, o = rng.randint(4, 15), rng.randint(3, 20)
                if e == o or o >= 2 * e:
                    continue
                sides = [e, e, o]
                rng.shuffle(sides)
            else:
                sides = sorted(rng.sample(range(3, 20), 3))
                if sides[0] + sides[1] <= sides[2]:
                    continue
                rng.shuffle(sides)
            break
        iso = len(set(sides)) < 3
        mode = rng.random()
        if iso and mode < 0.4:
            # show only the two equal sides
            i, j = [(i, j) for i in range(3) for j in range(i + 1, 3) if sides[i] == sides[j]][0]
            shown = [f"s{i}", f"s{j}"]
        else:
            shown = ["s0", "s1", "s2"]
        vals = {f"s{i}": sides[i] for i in range(3)}
        gt = {f"s{i}": ["side", names[(i + 1) % 3] + names[(i + 2) % 3], sides[i]] for i in range(3)}
    elif kind == "right":
        a, b, c = rng.choice(TRIPLES)
        if rng.random() < 0.5:
            c = c + rng.choice([-1, 1, 2])
            if c <= max(a, b):
                c = max(a, b) + 2
        sides = [a, b, c]
        rng.shuffle(sides)
        vals = {f"s{i}": sides[i] for i in range(3)}
        shown = ["s0", "s1", "s2"]
        gt = {f"s{i}": ["side", names[(i + 1) % 3] + names[(i + 2) % 3], sides[i]] for i in range(3)}
    else:
        while True:
            t = rng.choice(["acute", "right", "obtuse"])
            A = rng.randint(20, 80)
            if t == "right":
                angs = [90, A, 90 - A]
            elif t == "obtuse":
                big = rng.randint(95, 150)
                A = rng.randint(10, 180 - big - 10)
                angs = [big, A, 180 - big - A]
            else:
                angs = [rng.randint(50, 85), rng.randint(50, 85)]
                angs.append(180 - sum(angs))
                if not all(x < 90 for x in angs):
                    continue
            if min(angs) >= 10:
                break
        rng.shuffle(angs)
        ask = rng.randrange(3)
        vals = {f"a{i}": angs[i] for i in range(3)}
        shown = [f"a{i}" for i in range(3) if i != ask]
        gt = {f"a{i}": ["angle", names[i], angs[i]] for i in range(3)}
    return {"t": "classify", "kind": kind, "names": names, "unit": rng.choice(["", "cm"]), "vals": vals, "shown": shown,
            "where": pick_where(rng, shown, 0.25), "gtext": gt}


def b_classify(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    f = Fig()
    if spec["kind"] in ("iso", "right"):
        s = [v["s0"], v["s1"], v["s2"]]
        if child or spec["kind"] == "right":
            # not to scale: a generic scalene shape
            d = jitter_angles(rng, [60, 60, 60], 25)
            P = tri_from_angles(d[0], d[1])
            pts = {n[0]: P[0], n[1]: P[1], n[2]: P[2]}
        else:
            A, B, C = tri_from_sides(*s)
            pts = {n[0]: A, n[1]: B, n[2]: C}
        pts = orient(rng, pts)
        f.poly([pts[c] for c in n])
        g = centroid(list(pts.values()))
        for c in n:
            f.label(pts[c], c, away=g)
        for i in range(3):
            p, q = pts[n[(i + 1) % 3]], pts[n[(i + 2) % 3]]
            if on_fig(spec, f"s{i}"):
                f.label(mul(add(p, q), 0.5), length(s[i], spec["unit"]), away=g, dist=20, size=18, italic=False, color="#1f5fbf")
        q = f"Is triangle {n} isosceles?" if spec["kind"] == "iso" else f"Is triangle {n} a right triangle?"
        d = 3 if spec["kind"] == "iso" else 4
    else:
        angs = [v["a0"], v["a1"], v["a2"]]
        d_ = jitter_angles(rng, angs, 10)
        P = tri_from_angles(d_[0], d_[1])
        pts = orient(rng, {n[0]: P[0], n[1]: P[1], n[2]: P[2]})
        f.poly([pts[c] for c in n])
        g = centroid(list(pts.values()))
        for i, c in enumerate(n):
            f.label(pts[c], c, away=g)
            o = [pts[x] for x in n if x != c]
            if on_fig(spec, f"a{i}"):
                if angs[i] == 90 and False:
                    f.angle(pts[c], o[0], o[1], right=True)
                else:
                    f.angle(pts[c], o[0], o[1], deg(angs[i]))
        q = f"Is triangle {n} acute, right or obtuse?"
        d = 3
    return built(f, q, state_for(spec, f"triangle {n}"), rng.choice(CANVAS), d)


def solve_classify(spec):
    v, s = spec["vals"], spec["shown"]
    if spec["kind"] == "iso":
        known = [v[k] for k in s]
        if len(set(known)) < len(known):
            return True
        return False if len(known) == 3 else None
    if spec["kind"] == "right":
        if len(s) < 3:
            return None
        a, b, c = sorted(v[k] for k in s)
        return a * a + b * b == c * c
    known = [v[k] for k in s]
    if len(known) == 2:
        known.append(180 - sum(known))
    elif max(known) < 90:
        return None
    m = max(known)
    return "right" if m == 90 else "obtuse" if m > 90 else "acute"


# ------------------------------------------------------------------------------------------------ T18 similar triangles (DE || BC)
def s_similar(rng):
    p, q = rng.randint(2, 9), rng.randint(2, 9)
    m = rng.randint(1, 4)
    e = p * m
    bc = m * (p + q)
    kind = rng.choice(["bc", "de"])
    shown = ["p", "q", "e"] if kind == "bc" else ["p", "q", "bc"]
    names = rng.choice(["ABCDE", "PQRST", "AMNXY"])
    return {"t": "similar", "kind": kind, "names": names, "unit": rng.choice(["", "cm"]), "vals": {"p": p, "q": q, "e": e, "bc": bc},
            "shown": shown, "marks": {"parallel": True}, "where": pick_where(rng, shown, 0.2), "x_label": rng.random() < 0.4,
            "gtext": {"p": ["side", names[0] + names[3], p], "q": ["side", names[3] + names[1], q], "e": ["side", names[3] + names[4], e],
                      "bc": ["side", names[1] + names[2], bc]}}


def b_similar(spec, rng, child=False):
    v, n = spec["vals"], spec["names"]
    d = jitter_angles(rng, [60, 60, 60], 22)
    B, C, A = tri_from_angles(d[0], d[1])
    t = v["p"] / (v["p"] + v["q"])
    D = add(A, mul(sub(B, A), t))
    E = add(A, mul(sub(C, A), t))
    if not spec["marks"]["parallel"]:
        E = add(A, mul(sub(C, A), t + rng.choice([-1, 1]) * rng.uniform(0.12, 0.18)))
    pts = orient(rng, {"A": A, "B": B, "C": C, "D": D, "E": E})
    f = Fig()
    f.poly([pts["A"], pts["B"], pts["C"]])
    f.seg(pts["D"], pts["E"])
    g = centroid([pts["A"], pts["B"], pts["C"]])
    for k, c in zip("ABCDE", n):
        f.label(pts[k], c, away=g if k in "ABC" else mul(add(pts["D"], pts["E"]), 0.5) if False else g, dist=18)
    if spec["marks"]["parallel"]:
        f.arrow(pts["D"], pts["E"], 1)
        f.arrow(pts["B"], pts["C"], 1)
    u = spec["unit"]
    segs = {"p": ("A", "D"), "q": ("D", "B"), "e": ("D", "E"), "bc": ("B", "C")}
    ask = "bc" if spec["kind"] == "bc" else "e"
    inner = mul(add(pts["A"], pts["C"]), 0.5)
    for k, (a, b) in segs.items():
        m = mul(add(pts[a], pts[b]), 0.5)
        away = g if k != "e" else pts["A"]
        if k == "e":
            away = pts["A"]
        if on_fig(spec, k):
            f.label(m, length(v[k], u), away=away if k != "e" else inner, dist=16 if k == "e" else 20, size=17, italic=False, color="#1f5fbf")
        elif k == ask and spec["x_label"]:
            f.label(m, "x", away=away if k != "e" else inner, dist=16, size=20)
    q = "Find the value of x." if spec["x_label"] else (f"Find {n[1]}{n[2]}." if ask == "bc" else f"Find {n[3]}{n[4]}.")
    return built(f, q, state_for(spec, f"triangle {n[:3]} with segment {n[3]}{n[4]}"), rng.choice(CANVAS), 5)


def c_similar(spec, rng):
    v = spec["vals"]
    p, q, e, bc = v["p"], v["q"], v["e"], v["bc"]
    if spec["kind"] == "bc":
        return [bc, e * q / p, e * (p + q) / q, e + q, e * p / q]
    return [e, bc * q / (p + q), bc * p / q, bc - q, bc / 2]


def solve_similar(spec):
    if not spec["marks"]["parallel"]:
        return None
    v, s = spec["vals"], set(spec["shown"])
    if not {"p", "q"} <= s:
        return None
    if spec["kind"] == "bc":
        return v["e"] * (v["p"] + v["q"]) / v["p"] if "e" in s else None
    return v["bc"] * v["p"] / (v["p"] + v["q"]) if "bc" in s else None


# ------------------------------------------------------------------------------------------------ T19 count sides (score)
def s_count(rng):
    n = rng.randint(3, 12)
    return {"t": "count_sides", "vals": {"n": n}, "regular": rng.random() < 0.5, "shown": ["figure"], "where": {}}


def b_count(spec, rng, child=False):
    n = spec["vals"]["n"]
    a0 = rng.uniform(0, 2 * math.pi)
    if spec["regular"]:
        ps = [(math.cos(a0 + 2 * math.pi * i / n), math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]
    else:
        while True:
            cuts = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
            gaps = [(cuts[(i + 1) % n] - cuts[i]) % (2 * math.pi) for i in range(n)]
            if min(gaps) > 2 * math.pi / n * 0.45:
                break
        ex = rng.uniform(1.0, 1.5)
        ps = [(ex * math.cos(c), math.sin(c)) for c in cuts]
    f = Fig()
    f.poly(ps)
    for p in ps:
        f.dot(p, 3)
    levels = [{"value": i, "description": f"{i + 3} sides"} for i in range(10)]
    d = 2 if n <= 6 else 3 if n <= 9 else 4
    return built(f, "How many sides does the polygon have?", {"figure": "a polygon"}, rng.choice(CANVAS), d, note=False, levels=levels)


def solve_count(spec):
    return spec["vals"]["n"] - 3


# ------------------------------------------------------------------------------------------------ registry
# name: (family, sample, build, choices(None = noul/score/fixed), solve, answer type, formatter kind, weight, unknown child maker)
def fmt_angle(spec):
    return deg


def fmt_len(spec):
    u = spec.get("unit", "")
    return lambda v: length(v, u)


def fmt_area(spec):
    u = spec.get("unit", "")
    return lambda v: length(v, (u + "²") if u else "")


def _child_hide(key_choices):
    def mk(spec, rng):
        cands = key_choices(spec)
        if not cands:
            return None
        k = rng.choice(cands)
        c = dict(spec, shown=[x for x in spec["shown"] if x != k])
        return c, f"removed given {k}"
    return mk


def _child_mark(mark):
    def mk(spec, rng):
        c = dict(spec, marks=dict(spec["marks"], **{mark: False}))
        return c, f"removed mark {mark}"
    return mk


def child_parallel(spec, rng):
    if spec["kind"] == "relation":
        return None
    p1, p2 = spec["perm"][spec["gi"] - 1], spec["perm"][spec["aj"] - 1]
    if POS[p1][0] == POS[p2][0]:
        if spec["kind"] == "find":
            return _child_hide(lambda s: ["g"])(spec, rng)
        return None
    return _child_mark("parallel")(spec, rng)


def child_classify(spec, rng):
    v = spec["vals"]
    if spec["kind"] == "iso":
        if len(spec["shown"]) < 3:
            return None
        s = [v["s0"], v["s1"], v["s2"]]
        for i in range(3):
            rest = [s[j] for j in range(3) if j != i]
            if rest[0] != rest[1]:
                cands = [j for j in range(3) if [s[k] for k in range(3) if k != j][0] != [s[k] for k in range(3) if k != j][1]]
                j = rng.choice(cands)
                return dict(spec, shown=[f"s{k}" for k in range(3) if k != j]), f"removed given s{j}"
        return None
    if spec["kind"] == "right":
        j = rng.randrange(3)
        return dict(spec, shown=[f"s{k}" for k in range(3) if k != j]), f"removed given s{j}"
    known = sorted(spec["shown"], key=lambda k: v[k])
    if v[known[0]] >= 90:
        return None
    return dict(spec, shown=[known[0]]), f"removed given {known[1]}"


def child_parallelogram(spec, rng):
    if spec["kind"] == "area":
        return dict(spec, shown=[k for k in spec["shown"] if k != "h"]), "removed given h"
    return _child_mark("parallel")(spec, rng)


def child_area(spec, rng):
    if spec["shape"] == "rect":
        k = rng.choice(["w", "h"])
        return dict(spec, shown=[x for x in spec["shown"] if x != k]), f"removed given {k}"
    return dict(spec, shown=[x for x in spec["shown"] if x != "h"]), "removed given h"


TEMPLATES = {
    "tri_angle_sum": ("geometry_angle", s_tri_angle, b_tri_angle, c_tri_angle, solve_tri_angle, "choice", fmt_angle, 9,
                      _child_hide(lambda s: list(s["shown"]))),
    "tri_exterior": ("geometry_angle", s_tri_ext, b_tri_ext, c_tri_ext, solve_tri_ext, "choice", fmt_angle, 7,
                     _child_hide(lambda s: ["A"] if s["kind"] == "rem" else ["A", "B"])),
    "isosceles": ("geometry_angle", s_isosceles, b_isosceles, c_isosceles, solve_isosceles, "choice", fmt_angle, 7, _child_mark("ticks")),
    "parallel": ("geometry_relation", s_parallel, b_parallel, c_parallel, solve_parallel, "mixed", fmt_angle, 14, child_parallel),
    "lines_algebra": ("geometry_angle", s_lines_alg, b_lines_alg, c_lines_alg, solve_lines_alg, "choice",
                      lambda s: (num if s["kind"] == "x" else deg), 6, None),
    "pythagoras": ("geometry_length", s_pythag, b_pythag, c_pythag, solve_pythag, "choice", fmt_len, 9, _child_mark("right")),
    "circle_measure": ("geometry_circle", s_circle, b_circle, c_circle, solve_circle, "choice", fmt_circle, 6,
                       _child_hide(lambda s: list(s["shown"]))),
    "inscribed": ("geometry_circle", s_inscribed, b_inscribed, c_inscribed, solve_inscribed, "choice", fmt_angle, 6,
                  _child_hide(lambda s: list(s["shown"]))),
    "tangent": ("geometry_circle", s_tangent, b_tangent, c_tangent, solve_tangent, "choice",
                lambda s: (deg if s["kind"] == "ang" else fmt_len(s)), 6,
                _child_hide(lambda s: ["alpha"] if s["kind"] == "ang" else ["r"])),
    "parallelogram": ("geometry_area", s_parallelogram, b_parallelogram, c_parallelogram, solve_parallelogram, "choice",
                      lambda s: (deg if s["kind"] == "angle" else fmt_area(s) if s["kind"] == "area" else fmt_len(s)), 8, child_parallelogram),
    "regular_polygon": ("geometry_polygon", s_regular, b_regular, c_regular, solve_regular, "choice", fmt_angle, 5, None),
    "polygon_missing": ("geometry_polygon", s_poly_missing, b_poly_missing, c_poly_missing, solve_poly_missing, "choice", fmt_angle, 5,
                        _child_hide(lambda s: list(s["shown"]))),
    "area_shape": ("geometry_area", s_area, b_area, c_area, solve_area, "choice",
                   lambda s: (fmt_area(s) if s["kind"] == "area" else fmt_len(s)), 9, child_area),
    "coord": ("geometry_coord", s_coord, b_coord, c_coord, solve_coord, "mixed", None, 14, None),
    "classify": ("geometry_classify", s_classify, b_classify, None, solve_classify, "mixed", None, 8, child_classify),
    "similar": ("geometry_length", s_similar, b_similar, c_similar, solve_similar, "choice", fmt_len, 5, _child_mark("parallel")),
    "count_sides": ("geometry_polygon", s_count, b_count, None, solve_count, "score", None, 3, None),
}


def answer_type(spec) -> str:
    t = spec["t"]
    if t == "parallel":
        return "noul" if spec["kind"] == "equal" else "choice"
    if t == "coord":
        return "noul" if spec["kind"] in ("right", "para") else "choice"
    if t == "classify":
        return "choice" if spec["kind"] == "type" else "noul"
    if t == "count_sides":
        return "score"
    return "choice"


def formatter(spec):
    t = spec["t"]
    if t == "parallel" and spec["kind"] == "relation":
        return str
    if t == "coord":
        k = spec["kind"]
        return {"dist": num, "slope": frac_txt, "mid": pt_txt, "area": num, "quad": str}.get(k, str)
    if t == "classify":
        return str
    fm = TEMPLATES[t][6]
    return fm(spec)
