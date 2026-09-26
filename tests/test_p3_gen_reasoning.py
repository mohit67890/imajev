"""Phase-3 source-A reasoning generators (scripts/p3/gen_reasoning.py): validity, determinism, gold re-computed independently
from provenance.spec for every kind, unknown variants really unanswerable, option uniqueness after rounding, variants path."""
import calendar
import datetime as dt
import itertools
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p3"))
from candidate import validate  # noqa: E402
import gen_reasoning as G  # noqa: E402

N_TOTAL = 1400


def roundtrip(rs):
    return [json.loads(json.dumps(r, ensure_ascii=False)) for r in rs]


@pytest.fixture(scope="module")
def rows():
    return roundtrip(G.generate("pytest", N_TOTAL, workers=1))


# ------------------------------------------------------------------------------------------------ helpers
def fr(x):
    if isinstance(x, bool):
        return x
    return F(str(x)) if not isinstance(x, F) else x


def dec(x):
    with localcontext() as c:
        c.prec = 60
        if isinstance(x, F):
            return Decimal(x.numerator) / Decimal(x.denominator)
        if isinstance(x, float):
            return Decimal(repr(x))
        return Decimal(x)


def rq(x, places):
    return dec(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def D(s):
    return dt.date.fromisoformat(s)


def DT(s):
    return dt.datetime.fromisoformat(s)


def workdays_after(start, n, closed, days=(0, 1, 2, 3, 4)):
    """n-th working day strictly after start (independent loop)."""
    found, cur = [], start
    while len(found) < n:
        cur = cur + dt.timedelta(days=1)
        if cur.weekday() in days and cur not in closed:
            found.append(cur)
    return found[-1] if found else start


# ------------------------------------------------------------------------------------------------ independent solvers
def s_tab_growth(i, q):
    def m(y, name):
        r, c, s, d_, g = (fr(i[f"{k}|{y}"]) for k in ("rev", "cogs", "sm", "rd", "ga"))
        one = fr(i.get(f"oneoff|{y}", 0))
        op = r - c - s - d_ - g
        return {"rev": r, "opex": s + d_ + g, "gp": r - c, "opinc": op, "adj": op + one, "gm": (r - c) / r * 100, "om": op / r * 100,
                "adjm": (op + one) / r * 100}[name]
    a, b = m(q["a"], q["metric"]), m(q["b"], q["metric"])
    return (b - a) / a * 100 if q["form"] == "growth" else b - a


def s_tab_cagr(i, q):
    def v(y):
        return fr(i[f"cust|{y}"]) * fr(i[f"arpa|{y}"]) if q["derived"] else fr(i[f"v|{y}"])
    return (float(v(q["b"]) / v(q["a"])) ** (1 / q["k"]) - 1) * 100


def s_tab_compound(i, q):
    m, yrs = q["m"], int(fr(i["years"]))
    bal = fr(i["P"])
    for yr in range(yrs):
        for per in range(m):
            if "D" in i and yr == int(fr(i["dy"])) and per == 0:
                bal += fr(i["D"])
            rate = fr(i["r2"]) if ("r2" in i and yr >= int(fr(i["ky"]))) else fr(i["r"])
            bal *= 1 + rate / 100 / m
    return bal - fr(i.get("fee", 0))


def s_tab_fx(i, q):
    tot = F(0)
    for j, c, dd in q["lines"]:
        tot += F(str(rq(fr(i[f"amt|{j}"]) * fr(i[f"rate|{c}|{dd}"]), 2)))
    if "fee" in i:
        tot -= F(str(rq(tot * fr(i["fee"]) / 100, 2)))
    return tot


def s_tab_weighted(i, q):
    num = den = F(0)
    for j in range(q["n"]):
        if q["status"][str(j)] != q["ok"]:
            continue
        units = fr(i[f"q|{j}"]) * (q["case"] if q["cases"][str(j)] else 1)
        num += units * fr(i[f"p|{j}"])
        den += units
    return num / den


def s_tab_ratio(i, q):
    y, ya = q["yb"], q["ya"]
    g = lambda k, yy=y: fr(i[f"{k}|{yy}"])  # noqa: E731
    cl = g("pay") + g("std") + g("accr")
    met = q["metric"]
    if met == "current":
        return (g("cash") + g("recv") + g("inv") + g("prepaid")) / cl
    if met == "quick":
        return (g("cash") + g("recv")) / cl
    if met == "de":
        return (g("std") + g("ltd")) / g("eq")
    turn = fr(i["cogs"]) / ((g("inv", ya) + g("inv")) / 2)
    return turn if met == "turnover" else 365 / turn


def s_tab_variance(i, q):
    n = q["n"]
    a = {j: fr(i[f"a|{j}"]) for j in range(n)}
    b = {j: fr(i[f"rb|{j}"]) if f"rb|{j}" in i else fr(i[f"b|{j}"]) for j in range(n)}
    if q["reclass"]:
        s, t = q["reclass"]
        a[s] -= fr(i["rc"])
        a[t] += fr(i["rc"])
    pct = {j: (a[j] - b[j]) / b[j] for j in range(n)}
    if q["type"] == "choice":
        best = max(pct.values())
        winners = [j for j in pct if pct[j] == best]
        assert len(winners) == 1
        return ("argmax", winners[0])
    return ("bool", pct[q["dept"]] <= 0)


def s_tab_inventory(i, q):
    def one(j):
        tr = fr(i.get(f"tr|{j}", 0))
        recv = fr(i[f"ru|{j}"]) - tr
        units = fr(i[f"ou|{j}"]) + recv + fr(i[f"rt|{j}"]) - fr(i[f"sh|{j}"]) - fr(i[f"wo|{j}"])
        cost = (fr(i[f"ou|{j}"]) * fr(i[f"oc|{j}"]) + recv * fr(i[f"rc|{j}"])) / (fr(i[f"ou|{j}"]) + recv)
        return units, units * cost
    if q["mode"] == "units":
        return one(q["target"])[0]
    if q["mode"] == "value":
        return one(q["target"])[1]
    return sum(one(j)[1] for j in range(q["n"]))


def s_tab_saas(i, q):
    st, new, exp, con, ch = (fr(i[k]) for k in ("start", "new", "exp", "con", "churn"))
    if "mis" in i:
        exp, new = exp - fr(i["mis"]), new + fr(i["mis"])
    return {"end": st + new + exp - con - ch, "grr": (st - con - ch) / st * 100, "nrr": (st + exp - con - ch) / st * 100,
            "nrr_arr": 12 * (st + new + exp - con - ch)}[q["metric"]]


def s_tab_breakeven(i, q):
    p = fr(i["price"])
    c = p - fr(i["mat"]) - fr(i["lab"]) - fr(i.get("pack", 0)) - p * fr(i.get("comm", 0)) / 100
    need = fr(i["rent"]) + fr(i["sal"]) + fr(i["sw"])
    if "target" in i:
        need += fr(i["target"]) / (1 - fr(i["tax"]) / 100)
    u = 0
    while u * c < need:  # smallest whole number of units (independent of ceil)
        u += 1
    return F(u)


def s_tm_deadline(i, q):
    recv = DT(i["recv"])
    closed = {D(x) for x in i["closures"]}
    base = recv.date()
    if i["cutoff"] is not None and (recv.time() >= dt.time(i["cutoff"]) or base.weekday() >= 5 or base in closed):
        base = workdays_after(base, 1, closed)
    return workdays_after(base, i["n"], closed)


def s_tm_leave(i, q):
    a, b = D(i["a"]), D(i["b"])
    hol = {D(x) for x in i["hol"]}
    k = 0
    for x in range((b - a).days + 1):
        day = a + dt.timedelta(days=x)
        if day.weekday() >= 5 or day in hol or (i["off"] is not None and day.weekday() == i["off"]) or (i["sick"] and day == D(i["sick"])):
            continue
        k += 1
    return k


def s_tm_timezone(i, q):
    t = DT(i["t"])
    total = i.get("d1", 0) + i.get("lay", 0) + i.get("d2", 0)
    return t + dt.timedelta(minutes=i["ob"] - i["oa"] + total)


def s_tm_sla(i, q):
    """Interval arithmetic over business windows minus pause and holidays (independent of the minute loop)."""
    left = dt.timedelta(hours=i["h"])
    t = DT(i["opened"])
    hol = {D(x) for x in i["hol"]}
    pause = (DT(i["pause"][0]), DT(i["pause"][1])) if i["pause"] else None
    day = t.date()
    while True:
        if day.weekday() < 5 and day not in hol:
            ws = max(t, dt.datetime.combine(day, dt.time(i["oh"])))
            we = dt.datetime.combine(day, dt.time(i["ch"]))
            segs = [(ws, we)] if ws < we else []
            if pause:
                out = []
                for a, b in segs:
                    if pause[1] <= a or pause[0] >= b:
                        out.append((a, b))
                    else:
                        if a < pause[0]:
                            out.append((a, pause[0]))
                        if pause[1] < b:
                            out.append((pause[1], b))
                segs = out
            for a, b in segs:
                if b - a >= left:
                    return a + left
                left -= b - a
        day += dt.timedelta(days=1)


def s_tm_shift_pay(i, q):
    rate = fr(i["rate"])
    reg, pay = 0, F(0)
    for j, wd in enumerate(q["days"]):
        s, e, b = int(i[f"s|{j}"]), int(i[f"e|{j}"]), int(i[f"b|{j}"])
        m = (e - s) % 1440 or 1440
        m -= b
        if wd == 6:
            pay += F(m, 60) * rate * 2
            continue
        normal = max(0, min(m, 2400 - reg))
        pay += F(normal, 60) * rate + F(m - normal, 60) * rate * F(3, 2)
        reg += m
    return pay


def s_tm_units(i, q):
    lb = F("0.45359237")
    kg = {"kg": F(1), "lb": lb, "g": F(1, 1000), "oz": lb / 16, "t": F(1000)}
    if q["mode"] == "fuel":
        km = sum(fr(i[f"l|{j}"]) * (F("1.609344") if u == "mi" else 1) for j, u in enumerate(q["units"]))
        return km * fr(i["cons"]) / 100 / F("3.785411784") * fr(i["price"])
    w = sum(fr(i[f"w|{j}"]) * kg[u] for j, u in enumerate(q["units"]))
    if q["mode"] == "weight":
        return w
    return fr(i["limit"]) - w - fr(i["tare"]) * int(fr(i["pallets"]))


def s_tm_fx_dated(i, q):
    tot = F(0)
    for j, due in enumerate(q["pays"]):
        x = D(due)
        while f"r|{x.isoformat()}" not in i:
            x -= dt.timedelta(days=1)
        rate = fr(i[f"r|{x.isoformat()}"]) * (1 + fr(i.get("m", 0)) / 100)
        tot += F(str(rq(fr(i[f"a|{j}"]) * rate, 2)))
    return tot


def s_tm_tenure(i, q):
    h = D(i["hire"])
    m = h.month - 1 + i["n"]
    y, mo = h.year + m // 12, m % 12 + 1
    base = dt.date(y, mo, min(h.day, calendar.monthrange(y, mo)[1]))
    return base + dt.timedelta(days=sum((D(b) - D(a)).days + 1 for a, b in i["leave"]))


def s_tm_recurring(i, q):
    if q["mode"] == "bstep":
        closed = {D(x) for x in i["closed"]}
        x = D(i["s"])
        for _ in range(i["m"] - 1):
            x = workdays_after(x, i["k"], closed)
        return x
    mo = i["mo"] - 1 + i["m"] - 1
    y, mo = i["y"] + mo // 12, mo % 12 + 1
    days = [dt.date(y, mo, dd) for dd in range(1, calendar.monthrange(y, mo)[1] + 1) if dt.date(y, mo, dd).weekday() == i["wd"]]
    x = days[i["nth"] - 1]
    return x + dt.timedelta(days=1) if i["shift"] else x


def s_pr_draws(i, q):
    """Sequential draws without replacement as a probability tree (independent of the hypergeometric formula)."""
    N, K, k = int(fr(i["N"])), int(fr(i["K"])), int(fr(i["k"]))
    dist = {0: F(1)}
    for t in range(k):
        nd = {}
        for f, p in dist.items():
            pf = F(K - f, N - t)
            nd[f + 1] = nd.get(f + 1, 0) + p * pf
            nd[f] = nd.get(f, 0) + p * (1 - pf)
        dist = nd
    good = sum(p for f, p in dist.items() if (f == q["j"] if q["mode"] == "exactly" else f >= q["j"]))
    return good * 100


def s_pr_table(i, q):
    c = lambda a, b: fr(i[f"c|{a}|{b}"])  # noqa: E731
    ra, cb, rows, cols = q["ra"], q["cb"], q["rows"], q["cols"]
    if q["mode"] == "col_given_row":
        return c(ra, cb) / sum(c(ra, b) for b in cols)
    if q["mode"] == "row_given_col":
        return c(ra, cb) / sum(c(a, cb) for a in rows)
    others = [a for a in rows if a != ra]
    return sum(c(a, cb) for a in others) / sum(c(a, b) for a in others for b in cols)


def s_pr_bayes(i, q):
    p, s, f = fr(i["p"]), fr(i["s"]), fr(i["f"])
    s2, f2 = fr(i.get("s2", 1)), fr(i.get("f2", 1))
    return p * s * s2 / (p * s * s2 + (1 - p) * f * f2)


def s_pr_reliability(i, q):
    qs = {int(k.split("|")[1]): fr(v) for k, v in i.items()}
    comps = sorted(qs)
    shape = q["shape"]
    good = F(0)
    for states in itertools.product([True, False], repeat=len(comps)):
        up = dict(zip(comps, states))
        pr = math.prod((1 - qs[c]) if up[c] else qs[c] for c in comps)
        if shape == "series2":
            ok = up[0] and up[1]
        elif shape == "parallel2":
            ok = up[0] or up[1]
        elif shape == "par_series":
            ok = (up[0] or up[1]) and up[2]
        elif shape == "two_groups":
            ok = (up[0] or up[1]) and (up[2] or up[3] or up[4])
        else:
            ok = sum([up[0], up[1], up[2]]) >= 2 and up[3]
        good += pr * ok
    return good


def s_pr_binomial(i, q):
    n, p = int(fr(i["n"])), fr(i["p"])
    tot = F(0)
    for outcome in itertools.product([0, 1], repeat=n):
        s = sum(outcome)
        pr = p ** s * (1 - p) ** (n - s)
        ok = {"atleast": s >= q["k"], "exactly": s == q["k"], "atmost": s <= q["k"], "between": q["k"] <= s <= min(q["k2"], n)}[q["mode"]]
        tot += pr * ok
    return tot


def s_pr_wheels(i, q):
    wheels = q["wheels"]
    ds = []
    for w, vals in enumerate(wheels):
        tot = sum(fr(i[f"w|{w}|{v}"]) for v in vals)
        ds.append([(v, fr(i[f"w|{w}|{v}"]) / tot) for v in vals])
    num = den = F(0)
    for combo in itertools.product(*ds):
        pr = math.prod(p for _, p in combo)
        vs = [v for v, _ in combo]
        m = q["mode"]
        if m == "cond_even":
            if sum(vs) % 2 == 0:
                den += pr
                num += pr * (vs[0] > vs[-1])
            continue
        num += pr * {"sum_ge": sum(vs) >= q["t"], "sum_eq": sum(vs) == q["t"], "first_gt": vs[0] > vs[1], "prod_even": math.prod(vs) % 2 == 0}[m]
    return num / den if q["mode"] == "cond_even" else num


def s_pr_expected(i, q):
    tot = F(0)
    for m in range(q["k"]):
        loss = fr(i[f"L|{m}"])
        pay = max(F(0), loss - fr(i.get("ded", 0)))
        if "cap" in i:
            pay = min(pay, fr(i["cap"]))
        tot += fr(i[f"p|{m}"]) * pay
    return tot * int(fr(i["events"]))


def s_pr_markov(i, q):
    S, start, target = q["S"], q["start"], q["target"]
    tot = F(0)
    for path in itertools.product(S, repeat=q["steps"]):
        pr, cur = F(1), start
        for nxt in path:
            if cur == S[2]:
                pr *= 1 if nxt == S[2] else 0
            else:
                pr *= fr(i[f"t|{cur}|{nxt}"])
            cur = nxt
        if cur == target:
            tot += pr
    return tot


def s_pr_ev(i, q):
    evs = {}
    for j in range(q["n"]):
        k = q["outs"][str(j)]
        ev = sum(fr(i[f"p|{j}|{m}"]) * fr(i[f"v|{j}|{m}"]) for m in range(k))
        fee = fr(i.get(f"fee|{j}", 0))
        evs[j] = ev + fee if q["goal"] == "cost" else ev - fee
    best = (min if q["goal"] == "cost" else max)(evs.values())
    w = [j for j in evs if evs[j] == best]
    assert len(w) == 1
    if q["type"] == "choice":
        return ("argmax", w[0])
    return ("bool", w[0] == q["plan"])


# logic: clause form + plain backtracking search with pruning (no unit propagation; independent of the generator's DPLL)
def lclauses(rules, facts, ent):
    cl = []
    for r in rules:
        cl.append([(l[0], not l[1]) for l in r["if"]] + [(l[0], l[1]) for l in r.get("unless", [])] + [(r["then"][0], r["then"][1])])
    for f in facts:
        if f["ent"] == ent:
            cl.append([tuple(f["lit"])] if "lit" in f else [tuple(x) for x in f["or"]])
    return cl


def lsat(cl):
    preds = sorted({p for c in cl for p, _ in c})
    idx = {p: k for k, p in enumerate(preds)}
    last = [max(idx[p] for p, _ in c) for c in cl]

    def rec(k, a):
        for c, lk in zip(cl, last):
            if lk == k - 1 and not any(a[p] == b for p, b in c):
                return False
        if k == len(preds):
            return True
        for v in (True, False):
            a[preds[k]] = v
            if rec(k + 1, a):
                return True
        del a[preds[k]]
        return False
    return rec(0, {})


def entailed(rules, facts, ent, lit):
    cl = lclauses(rules, facts, ent)
    assert lsat(cl), "inconsistent"
    p, b = lit[0], lit[1]
    if not lsat(cl + [[(p, not b)]]):
        return True
    if not lsat(cl + [[(p, b)]]):
        return False
    return None


def walk(facts, start, rels):
    kb = {(a, r): b for a, r, b in facts}
    x = start
    for r in rels:
        if (x, r) not in kb:
            return None
        x = kb[(x, r)]
    return x


# constraint solvers (independent brute force)
def c_holds(c, pos, n):
    t = c[0]
    return {"before": lambda: pos[c[1]] < pos[c[2]], "imm": lambda: pos[c[2]] - pos[c[1]] == 1, "adj": lambda: abs(pos[c[1]] - pos[c[2]]) == 1,
            "nadj": lambda: abs(pos[c[1]] - pos[c[2]]) != 1, "at": lambda: pos[c[1]] == c[2], "nat": lambda: pos[c[1]] != c[2],
            "end": lambda: pos[c[1]] in (0, n - 1), "gap": lambda: abs(pos[c[1]] - pos[c[2]]) > 2,
            "between": lambda: (pos[c[1]] < pos[c[2]] < pos[c[3]]) or (pos[c[3]] < pos[c[2]] < pos[c[1]])}[t]()


def order_sols(items, cons):
    n = len(items)
    return [dict(zip(items, p)) for p in itertools.permutations(range(n)) if all(c_holds(c, dict(zip(items, p)), n) for c in cons)]


def a_holds(c, A, B):
    t = c[0]
    ppl = list(A)
    if t == "a_is":
        return A[c[1]] == c[2]
    if t == "a_not":
        return A[c[1]] != c[2]
    if t == "b_is":
        return B[c[1]] == c[2]
    if t == "b_not":
        return B[c[1]] != c[2]
    if t == "a_before":
        return A[c[1]] < A[c[2]]
    if t == "a_next":
        return A[c[2]] - A[c[1]] == 1
    who = [p for p in ppl if A[p] == c[1]] if t in ("b_of_a", "b_of_a_not") else [p for p in ppl if B[p] == c[1]]
    if t == "b_of_a":
        return B[who[0]] == c[2]
    if t == "b_of_a_not":
        return B[who[0]] != c[2]
    return A[who[0]] < A[c[2]]


def assign_sols(people, n, cons):
    out = []
    for pa in itertools.permutations(range(n)):
        for pb in itertools.permutations(range(n)):
            A, B = dict(zip(people, pa)), dict(zip(people, pb))
            if all(a_holds(c, A, B) for c in cons):
                out.append((A, B))
    return out


def alloc_best(inp, n, projs, rules):
    best, key, count = None, None, 0
    for s in itertools.chain.from_iterable(itertools.combinations(range(n), k) for k in range(n + 1)):
        if sum(int(inp[f"c|{j}"]) for j in s) > int(inp["budget"]):
            continue
        if any((r[0] == "req" and r[1] in s and r[2] not in s) or (r[0] == "excl" and r[1] in s and r[2] in s) or (r[0] == "must" and r[1] not in s)
               for r in rules):
            continue
        k = (sum(int(inp[f"v|{j}"]) for j in s), -sum(int(inp[f"c|{j}"]) for j in s))
        if key is None or k > key:
            best, key, count = s, k, 1
        elif k == key:
            count += 1
    return best, count


def label(projs, s):
    return ", ".join(sorted(projs[j] for j in s)) if s else "none"


# ordinal solvers
METRIC_HB = None


def o_points(i, q):
    from gen_reasoning_families.ordinal import BOOLS, METRICS
    tot = 0
    for m in q["ms"]:
        cuts = [fr(c) for c in q["cuts"][m]]
        v = fr(i[m])
        pts = sum(1 for c in cuts if (v >= c if METRICS[m][5] else v <= c))
        tot += pts * q["weights"][m]
    lvl = sum(1 for b in q["bands"] if tot >= b)
    if q["cap"]:
        b, c = q["cap"]
        if i[b] != BOOLS[b][1]:
            lvl = min(lvl, c)
    return lvl


def o_ladder(i, q):
    from gen_reasoning_families.ordinal import METRICS
    lvl = 0
    for k in range(1, q["n"]):
        ok = True
        for kind, key, c in q["reqs"][str(k)]:
            if kind == "bool":
                ok &= i[key] == c
            else:
                ok &= (fr(i[key]) >= fr(c)) if METRICS[key][5] else (fr(i[key]) <= fr(c))
        if not ok:
            break
        lvl = k
    return lvl


def o_weighted(i, q):
    sc = {}
    for c in q["crits"]:
        v = fr(i[f"s|{c}"])
        if q["two"] == c:
            v = (v + fr(i[f"s2|{c}"])) / 2
        sc[c] = v
    avg = sum(sc[c] * w for c, w in zip(q["crits"], q["ws"])) / 100
    lvl = sum(1 for t in q["th"] if avg >= fr(t))
    if q["cap"] and min(sc.values()) <= q["cap"][0]:
        lvl = min(lvl, q["cap"][1])
    return lvl


def o_matrix(i, q):
    u, r = i["users"], i["revenue"]
    imp_u = sum(1 for c in q["u_cuts"] if u >= c)
    rc = q["r_cuts"]
    imp_r = 3 if r >= rc[1] * 2 else 2 if r >= rc[1] else 1 if r >= rc[0] else 0
    imp = max(imp_u, imp_r)
    urg = 2 if i["deadline_h"] <= q["dl"][0] else 1 if i["deadline_h"] <= q["dl"][1] else 0
    if i["workaround"]:
        urg = max(0, urg - 1)
    p = q["mat"][imp][urg]
    if i.get("security"):
        p = min(q["nl"] - 1, p + 1)
    if i.get("vip") and i.get("repeat"):
        p = min(q["nl"] - 1, p + 1)
    return p


def o_count(i, q):
    tot = 0
    for j, (name, rules) in enumerate(q["obs"]):
        v = fr(i[f"o|{j}"])
        sev = next((s for t, s in rules if v >= fr(t)), None)
        if sev:
            pts = q["w"][sev] * (2 if (i.get(f"rep|{j}") and sev != "minor") else 1)
            tot += pts
    return sum(1 for b in q["bands"] if tot >= b)


NUMERIC = {"tab_growth": s_tab_growth, "tab_cagr": s_tab_cagr, "tab_compound": s_tab_compound, "tab_fx": s_tab_fx, "tab_weighted": s_tab_weighted,
           "tab_ratio": s_tab_ratio, "tab_inventory": s_tab_inventory, "tab_saas": s_tab_saas, "tab_breakeven": s_tab_breakeven,
           "tm_shift_pay": s_tm_shift_pay, "tm_units": s_tm_units, "tm_fx_dated": s_tm_fx_dated, "pr_expected": s_pr_expected}
PROB = {"pr_draws": lambda i, q: s_pr_draws(i, q) / 100, "pr_table": s_pr_table, "pr_bayes": s_pr_bayes, "pr_reliability": s_pr_reliability,
        "pr_binomial": s_pr_binomial, "pr_wheels": s_pr_wheels, "pr_markov": s_pr_markov}
VALUE = {"tm_deadline": (s_tm_deadline, lambda v, p: D(p) <= v), "tm_leave": (s_tm_leave, lambda v, p: v > p),
         "tm_timezone": (s_tm_timezone, lambda v, p: DT(p) >= v), "tm_sla": (s_tm_sla, lambda v, p: DT(p) <= v),
         "tm_tenure": (s_tm_tenure, lambda v, p: D(p) >= v), "tm_recurring": (s_tm_recurring, lambda v, p: D(p) == v)}
ORD = {"or_points": o_points, "or_ladder": o_ladder, "or_weighted": o_weighted, "or_matrix": o_matrix, "or_count": o_count}


def enc_val(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, F):
        return str(v)
    return v


def answer(row, inputs=None):
    """Independent answer for a row's spec (optionally with overridden inputs): the option key (choice), bool (noul) or level."""
    sp = row["provenance"]["spec"]
    kind = sp["kind"]
    i = dict(sp.get("inputs", {}))
    if inputs:
        i.update(inputs)
    q = sp.get("q", {})
    f = row["field"]
    ov = sp.get("option_values")

    def pick_numeric(val):
        if "threshold" in sp:
            return dec(val) > Decimal(sp["threshold"])
        g = rq(val, sp["places"])
        hits = [k for k, v in ov.items() if Decimal(v) == g]
        return hits[0] if len(hits) == 1 else ("AMBIGUOUS" if hits else "NONE")

    if kind in NUMERIC:
        return pick_numeric(NUMERIC[kind](i, q))
    if kind in PROB:
        v = PROB[kind](i, q)
        if "places" in sp:
            return pick_numeric(v * 100)
        if "probe" in sp:
            return v > fr(sp["probe"])
        hits = [k for k, x in ov.items() if fr(x) == v]
        return hits[0] if len(hits) == 1 else "NONE"
    if kind in VALUE:
        fn, judge = VALUE[kind]
        v = fn(i, q)
        if "probe" in sp:
            return judge(v, sp["probe"])
        hits = [k for k, x in ov.items() if x == enc_val(v)]
        return hits[0] if len(hits) == 1 else "NONE"
    if kind in ("tab_variance", "pr_ev"):
        t, x = (s_tab_variance if kind == "tab_variance" else s_pr_ev)(i, q)
        if t == "bool":
            return x
        hits = [k for k, v in ov.items() if int(v) == x]
        return hits[0]
    if kind in ORD:
        return ORD[kind](i, q)
    if kind == "logic_noul":
        return entailed(sp["rules"], sp["facts"], sp["ent"], sp["query"])
    if kind == "logic_statement":
        hits = [k for k, lit in ov.items() if entailed(sp["rules"], sp["facts"], sp["ent"], lit) is True]
        return hits[0] if len(hits) == 1 else (None if not hits else "AMBIGUOUS")
    if kind == "logic_entity":
        hits = [k for k, e in ov.items() if entailed(sp["rules"], sp["facts"], e, sp["query"]) is True]
        return hits[0] if len(hits) == 1 else (None if not hits else "AMBIGUOUS")
    if kind == "mh_path":
        end = walk(sp["facts"], sp["start"], sp["rels"])
        if end is None:
            return None
        if "probe" in sp:
            return end == sp["probe"]
        hits = [k for k, v in ov.items() if v == end]
        return hits[0] if hits else "NONE"
    if kind == "mh_compare":
        ends = [walk(sp["facts"], s, r) for s, r in sp["chains"]]
        if None in ends or any(e not in sp["years"] for e in ends):
            return None
        older = min(ends, key=lambda e: sp["years"][e])
        return [k for k, v in ov.items() if v == older][0]
    if kind == "cs_order":
        sols = order_sols(sp["items"], sp["cons"])
        if sp["qt"] == "slot":
            occ = {[x for x in sp["items"] if s[x] == sp["slot"]][0] for s in sols}
            occ &= set(ov.values())
            return [k for k, v in ov.items() if v in occ][0] if len(occ) == 1 else None
        if sp["qt"] == "full":
            orders = {sp["sep"].join(sorted(sp["items"], key=lambda x: s[x])) for s in sols} & set(ov.values())
            return [k for k, v in ov.items() if v in orders][0] if len(orders) == 1 else None
        vals = {c_holds(sp["probe"], s, len(sp["items"])) for s in sols}
        return vals.pop() if len(vals) == 1 else None
    if kind == "cs_assign":
        sols = assign_sols(sp["people"], sp["n"], sp["cons"])
        if sp["ask"] == "b_of_person":
            vals = {B[sp["who"]] for _, B in sols}
            if len(vals) != 1:
                return None
            x = vals.pop()
            return [k for k, v in ov.items() if v == x][0]
        if sp["ask"] == "person_of_a":
            vals = {[p for p in A if A[p] == sp["k"]][0] for A, _ in sols}
            if len(vals) != 1:
                return None
            x = vals.pop()
            return [k for k, v in ov.items() if v == x][0]
        vals = {B[sp["who"]] == sp["m"] for _, B in sols}
        return vals.pop() if len(vals) == 1 else None
    if kind == "cs_alloc":
        best, count = alloc_best(i, sp["n"], sp["projs"], sp["rules"])
        if count > 1:
            return "TIE"
        return [k for k, v in ov.items() if v == label(sp["projs"], best)][0]
    raise AssertionError(f"no solver for {kind}")


# ------------------------------------------------------------------------------------------------ tests
def test_all_rows_valid_and_shaped(rows):
    ids = set()
    for r in rows:
        assert validate(r) == [], r["id"]
        assert r["source"] == "A" and r["gold_kind"] == "constructed" and r["provenance"]["licence"] == "generated"
        assert r["provenance"]["generator"].startswith("scripts/p3/gen_reasoning.py@")
        assert r["difficulty"] in (3, 4, 5)
        assert r["family"] in G.FAMILIES
        assert r["id"] not in ids
        ids.add(r["id"])
        assert G.ID_RE.match(r["id"]), r["id"]
        if r["family"] == "ordinal":
            assert r["field"]["type"] == "score" and 3 <= len(r["field"]["levels"]) <= 10
        n_words = len(r["state"].split())
        assert 60 <= n_words <= 1500, (r["id"], n_words)


def test_every_family_and_kind_present(rows):
    fams = {r["family"] for r in rows}
    assert fams == set(G.FAMILIES)
    kinds = {r["provenance"]["kind"] for r in rows}
    assert kinds == set(G.KINDS), set(G.KINDS) - kinds


def test_family_balance_and_unknown_share(rows):
    t = G.targets(len(rows))
    import collections
    c = collections.Counter(r["family"] for r in rows)
    for f in G.FAMILIES:
        assert c[f] == t[f]
        unk = sum(1 for r in rows if r["family"] == f and r["gold"] is None)
        assert 0.10 <= unk / c[f] <= 0.20, (f, unk / c[f])
    assert c["ordinal"] / len(rows) == pytest.approx(0.2, abs=0.01)


def test_options_unique_after_display(rows):
    for r in rows:
        f = r["field"]
        if f["type"] != "choice":
            continue
        texts = [o["text"].strip().lower() for o in f["options"]]
        assert len(set(texts)) == len(texts), r["id"]
        sp = r["provenance"]["spec"]
        ov = sp.get("option_values") or {}
        if "places" in sp:
            vals = [Decimal(v) for v in ov.values()]
            assert len(set(vals)) == len(vals), r["id"]
            if r["gold"] is not None:
                g = Decimal(ov[r["gold"]])
                unit = Decimal(1).scaleb(-sp["places"])
                assert all(abs(v - g) >= unit for v in vals if v != g), r["id"]


def test_gold_recomputed_independently(rows):
    checked = 0
    for r in rows:
        if r["parent_id"] is not None:
            continue
        got = answer(r)
        assert got == r["gold"], (r["id"], got, r["gold"])
        checked += 1
    assert checked > 200


def test_unknown_variants_are_unanswerable(rows):
    by_id = {r["id"]: r for r in rows}
    n = 0
    for r in rows:
        if r["parent_id"] is None:
            continue
        n += 1
        parent = by_id[r["parent_id"]]
        assert r["gold"] is None and r["unknown_reason"] == "insufficient_evidence"
        assert r["family"] == parent["family"] and r["difficulty"] == parent["difficulty"]
        assert r["state"] != parent["state"]
        sp = r["provenance"]["spec"]
        if "witness_alt" in sp:
            # two values of the removed fact, both consistent with the redacted state, give different answers
            a0 = answer(r)
            a1 = answer(r, {k: v for k, v in sp["witness_alt"].items()})
            assert a0 != a1, (r["id"], a0, a1)
            assert a0 == parent["gold"], r["id"]
            assert any(m in r["state"] for m in ("n/a*", "[")), r["id"]
        else:
            # structural removal (logic premise, KB fact, puzzle clue): the independent solver cannot determine an answer
            assert answer(r) is None, r["id"]
    assert n > 30


def test_natural_unknowns_are_undetermined(rows):
    nat = [r for r in rows if r["gold"] is None and r["parent_id"] is None]
    assert nat, "expected natural cannot-be-determined logic items"
    for r in nat:
        assert r["family"] == "logic"
        assert answer(r) is None, r["id"]


def test_deterministic_by_seed():
    a = G.generate("det", 70, families=["logic", "probability"], workers=1)
    b = G.generate("det", 70, families=["logic", "probability"], workers=1)
    c = G.generate("det2", 70, families=["logic", "probability"], workers=1)
    assert a == b
    assert [r["state"] for r in a] != [r["state"] for r in c]


def test_variant_of_makes_fresh_same_kind_items(rows):
    parent = next(r for r in rows if r["parent_id"] is None and r["gold"] is not None and r["family"] == "table_arithmetic")
    vs = roundtrip(G.variants_of(parent["id"], 2))
    assert len(vs) == 2
    for v in vs:
        assert validate(v) == []
        assert v["parent_id"] == parent["id"] and v["family"] == parent["family"] and v["difficulty"] == parent["difficulty"]
        assert v["provenance"]["kind"] == parent["provenance"]["kind"] and v["state"] != parent["state"]
        assert answer(v) == v["gold"]
    assert roundtrip(G.variants_of(parent["id"], 2)) == vs  # deterministic
    unk = next(r for r in rows if r["parent_id"] is not None)
    uv = roundtrip(G.variants_of(unk["id"], 1))[0]
    assert uv["gold"] is None and uv["unknown_reason"] == "insufficient_evidence"


def test_fresh_seed_heldout_has_no_13gram_leakage(tmp_path, rows):
    """The Stage-1 leakage rule (13-gram near-duplicates between held-out and train) must pass for fresh-seed items of every family."""
    import decontam
    from candidate import write
    train, held = tmp_path / "train.jsonl", tmp_path / "held.jsonl"
    write(train, rows)
    write(held, G.generate("pytest-heldout", 350, workers=1))
    assert decontam.main(["--leakage", "--train", str(train), "--heldout", str(held), "--report", str(tmp_path / "leak.md")]) == 0
