"""Family probability: exact-answer probability and expected-value items in our own phrasing (sampling audits, contingency
tables, screening with base rates, expected-cost choices, redundancy, repeated trials, custom wheels, expected payouts,
monthly state transitions)."""
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as F

from .common import (CURRENCIES, DOMAINS, Names, Skip, T, assemble, choice_field, ffrac, fmoney, fnum, fpct, noul_field, rq)
from .numeric_base import jenc, md_table, numeric_item, value_item

FAM = "probability"


def _dom(rng):
    k = rng.choice(sorted(DOMAINS))
    return k, DOMAINS[k]


def pperturb(k: str, inp: dict) -> list[dict]:
    v = inp[k]
    out = []
    if isinstance(v, F) and 0 < v < 1:
        cands = [v / 2, (1 + v) / 2, v * F(3, 4), v + (1 - v) / 4]
    elif isinstance(v, F):
        cands = [v * F(3, 2), v * F(2, 3), v * 2, v / 2]
    elif isinstance(v, int):
        cands = [v + 1, v + 2, v - 1, v + 3]
    else:
        return []
    for c in cands:
        if c != v and (not isinstance(c, int) or c > 0):
            out.append(dict(inp, **{k: c}))
    return out


def prob_item(rng, kind, *, inp, q, compute, wrongs, render, qc, qn_about, decisive, style=None, perturb=None):
    """compute returns an exact probability (Fraction in [0,1]). Rendered as a percentage (1-2 dp) or a reduced fraction."""
    perturb = perturb or pperturb
    val = compute(inp)
    if not (0 < val < 1):
        raise Skip("degenerate probability")
    style = style or rng.choice(["pct1", "pct2", "frac"] if val.denominator <= 5000 else ["pct1", "pct2"])
    if style == "frac":
        def fill(k):
            num = val.numerator + ((k + 1) // 2) * (-1) ** k * max(1, val.numerator // 6)
            c = F(num, val.denominator)
            return c if 0 < c < 1 else None

        def noul(r, v):
            for den in (2, 3, 4, 5, 6, 8, 10, 12, 20):
                for num in range(1, den):
                    p = F(num, den)
                    if p != v and abs(p - v) >= F(1, 100) and abs(p - v) < F(1, 6) and r.random() < 0.5:
                        return qn_about(ffrac(p)), p
            raise Skip("no probe")
        return value_item(rng, FAM, kind, inp=inp, q=q, compute=compute, wrongs=lambda i: [w for w in wrongs(i) if w is not None and 0 < w < 1],
                          render=render, q_choice=qc, fmt=ffrac, decisive=decisive, perturb=perturb, noul=noul, judge=lambda v, p: v > p,
                          fill=fill)
    places = 1 if style == "pct1" else 2
    return numeric_item(rng, FAM, kind, inp=inp, q=q, compute=lambda i: compute(i) * 100,
                        wrongs=lambda i: [w * 100 for w in wrongs(i) if w is not None and 0 <= w <= 1], render=render, q_choice=qc,
                        q_noul=lambda t: qn_about(t), places=places, fmt=lambda v: fpct(v, places), decisive=decisive, perturb=perturb)


# ======================================================================================= sampling without replacement
SAMPLING = [("invoices", "contain a coding error", "contains a coding error", "the internal-audit sample"),
            ("parcels", "are damaged", "is damaged", "the quality spot-check"),
            ("patient files", "lack a signed consent form", "lacks a signed consent form", "the records audit"),
            ("laptops", "have outdated firmware", "has outdated firmware", "the security sweep"),
            ("expense claims", "breach policy", "breaches policy", "the compliance review"),
            ("meters", "read incorrectly", "reads incorrectly", "the field test"),
            ("contracts", "are missing a renewal clause", "is missing a renewal clause", "the legal sample"),
            ("exam scripts", "were mis-graded", "was mis-graded", "the moderation check"),
            ("pallets", "are mislabelled", "is mislabelled", "the dock inspection"),
            ("support tickets", "were closed without a fix", "was closed without a fix", "the QA review")]


def pr_draws(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    what, bad, bad1, proc = rng.choice(SAMPLING)
    org = names.company()
    N = rng.randint(12, {3: 30, 4: 60, 5: 80}[d])
    K = rng.randint(2, max(3, N // 3))
    k = rng.randint(2, {3: 3, 4: 5, 5: 6}[d])
    mode = {3: "atleast1", 4: rng.choice(["exactly", "atleast1"]), 5: rng.choice(["atleast2", "exactly"])}[d]
    j = rng.randint(1, min(k, K) - (0 if mode == "exactly" else 0)) if mode == "exactly" else (1 if mode == "atleast1" else 2)
    if mode == "atleast2" and min(k, K) < 2:
        raise Skip("impossible")
    inp = {"N": N, "K": K, "k": k}
    q = {"mode": mode, "j": j}

    def hyper(i, x):
        return F(math.comb(i["K"], x) * math.comb(i["N"] - i["K"], i["k"] - x), math.comb(i["N"], i["k"]))

    def compute(i):
        if i["K"] >= i["N"] or i["k"] > i["N"]:
            raise Skip("bad")
        if mode == "exactly":
            return hyper(i, j)
        return 1 - sum(hyper(i, x) for x in range(j))

    def wrongs(i):
        p = F(i["K"], i["N"])
        rep_at1 = 1 - (1 - p) ** i["k"]
        out = [rep_at1, p * i["k"] if p * i["k"] < 1 else None, hyper(i, j), 1 - hyper(i, 0), hyper(i, 1)]
        out.append(F(math.comb(i["k"], j)) * p ** j * (1 - p) ** (i["k"] - j))
        return out

    def render(i, missing, r):
        lines = [T(r, "{A batch of $N $w is in scope|The population in scope holds $N $w|There are $N $w in scope}. {Of these|Among them}, $K $b.",
                   N=i["N"], w=what, K="[an unconfirmed number]" if "K" in missing else i["K"], b=bad),
                 T(r, "For $p, $k $w are {picked|drawn|selected} at random without replacement{, every set of that size being equally likely|}.",
                   p=proc, k="[a number still to be agreed]" if "k" in missing else i["k"], w=what)]
        head = [T(r, "{Sampling plan|Audit sampling note|Spot-check design} - $o", o=org)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    subj = {"atleast1": f"at least one of the {k} sampled {what} {bad1}", "atleast2": f"at least two of the {k} sampled {what} {bad}",
            "exactly": f"exactly {j} of the {k} sampled {what} {bad1 if j == 1 else bad}"}[mode]
    subj = subj.replace("sampled", "sampled " + org, 1)
    qc = T(rng, "What is the probability that $s?", s=subj)
    return prob_item(rng, "pr_draws", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "Is the probability that $s greater than $t?", s=subj, t=t), decisive=["K", "k"])


# ======================================================================================= contingency table
TABLES = [("orders", ["web", "app", "phone"], ["returned", "kept"]), ("applicants", ["referral", "job board", "agency"], ["hired", "not hired"]),
          ("claims", ["motor", "home", "travel"], ["disputed", "settled"]), ("tickets", ["email", "chat", "phone"], ["escalated", "resolved at first line"]),
          ("shipments", ["road", "rail", "sea"], ["late", "on time"]), ("students", ["full-time", "part-time", "distance"], ["passed", "failed"]),
          ("invoices", ["domestic", "EU", "overseas"], ["paid late", "paid on time"])]


def pr_table(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    unit, rows, cols = rng.choice(TABLES)
    rows = rows[: 2 if d == 3 else 3]
    inp = {f"c|{a}|{b}": rng.randint(5, 400) for a in rows for b in cols}
    ra, cb = rng.choice(rows), cols[0]
    mode = {3: "col_given_row", 4: rng.choice(["row_given_col", "col_given_row"]), 5: rng.choice(["row_given_col", "row_given_notrow"])}[d]
    q = {"mode": mode, "ra": ra, "cb": cb, "rows": rows, "cols": cols}

    def compute(i):
        cnt = lambda a, b: i[f"c|{a}|{b}"]  # noqa: E731
        if mode == "col_given_row":
            return F(cnt(ra, cb), sum(cnt(ra, b) for b in cols))
        if mode == "row_given_col":
            return F(cnt(ra, cb), sum(cnt(a, cb) for a in rows))
        others = [a for a in rows if a != ra]
        return F(sum(cnt(a, cb) for a in others), sum(cnt(a, b) for a in others for b in cols))

    def wrongs(i):
        cnt = lambda a, b: i[f"c|{a}|{b}"]  # noqa: E731
        tot = sum(i.values())
        return [F(cnt(ra, cb), sum(cnt(ra, b) for b in cols)), F(cnt(ra, cb), sum(cnt(a, cb) for a in rows)), F(cnt(ra, cb), tot),
                F(sum(cnt(a, cb) for a in rows), tot), F(sum(cnt(ra, b) for b in cols), tot)]

    def render(i, missing, r):
        tbl = md_table([unit.capitalize()] + cols, [[a] + ["n/a*" if f"c|{a}|{b}" in missing else fnum(i[f"c|{a}|{b}"]) for b in cols] for a in rows])
        notes = [T(r, "{Every|Each} one of the $u {counted|tallied} {falls in|appears in} exactly one cell.", u=unit)]
        if missing:
            notes.append("\\* Count not yet extracted from the system.")
        head = [T(r, "{Quarterly|Monthly|Year-to-date} breakdown of $u by channel and outcome - $o (extract $t)", u=unit, o=org, t=tag)]
        return assemble(r, head, [tbl] + notes, dom, Names(r, names.used), d)

    org = names.company()
    tag = names.code(rng.choice(["Q", "S", "R"]))
    if mode == "col_given_row":
        s = f"a random {ra} {unit[:-1]} from {org} ({tag}) is {cb}"
        qc = T(rng, "Pick one $ra $u from $o at random ($t). What is the probability it is $cb?", u=unit[:-1], ra=ra, cb=cb, o=org, t=tag)
    elif mode == "row_given_col":
        s = f"a random {cb} {unit[:-1]} from {org} ({tag}) came through {ra}"
        qc = T(rng, "Pick one $cb $u from $o at random ($t). What is the probability it came through $ra?", u=unit[:-1], cb=cb, ra=ra, o=org, t=tag)
    else:
        s = f"a random non-{ra} {unit[:-1]} from {org} ({tag}) is {cb}"
        qc = T(rng, "Pick one $o $u outside the $ra group at random ($t). What is the probability it is $cb?", u=unit[:-1], ra=ra, cb=cb, o=org, t=tag)
    decisive = [f"c|{ra}|{cb}"] + [k for k in inp if k.startswith(f"c|{ra}|")] + [k for k in inp if k.endswith(f"|{cb}")]
    return prob_item(rng, "pr_table", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "Is the probability that $s above $t?", s=s, t=t), decisive=decisive)


# ======================================================================================= screening with base rates
SCREENS = [("transactions", "fraudulent", "the fraud model raises an alert"), ("emails", "phishing", "the filter quarantines it"),
           ("welds", "defective", "the ultrasonic scan flags it"), ("claims", "inflated", "the triage rule flags it"),
           ("log-ins", "account takeovers", "the anomaly detector fires"), ("meters", "tampered with", "the analytics flag it"),
           ("applications", "fabricated", "the document check flags it")]


def pr_bayes(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    unit, bad, flag = rng.choice(SCREENS)
    inp = {"p": F(rng.randint(5, 120), 1000), "s": F(rng.randint(700, 990), 1000), "f": F(rng.randint(10, 150), 1000)}
    if d == 5:
        inp["s2"] = F(rng.randint(600, 980), 1000)
        inp["f2"] = F(rng.randint(20, 250), 1000)
    natural = d == 4 and rng.random() < 0.6
    if natural:
        inp["n"] = rng.choice([10_000, 20_000, 50_000, 100_000])

    def compute(i):
        a = i["p"] * i["s"] * i.get("s2", 1)
        b = (1 - i["p"]) * i["f"] * i.get("f2", 1)
        return a / (a + b)

    def wrongs(i):
        a1 = i["p"] * i["s"]
        return [i["s"], 1 - i["f"], a1, i["p"] * i["s"] / (i["p"] * i["s"] + i["f"]), a1 / (a1 + (1 - i["p"]) * i["f"]) if "s2" in i else None,
                1 - i["f"] * i.get("f2", 1)]

    def render(i, missing, r):
        pct = lambda k: "[not measured]" if k in missing else f"{float(i[k] * 100):g}%"  # noqa: E731
        if natural:
            lines = [T(r, "Out of every $n $u {processed|reviewed}, historically $pp are $b.", n=fnum(i["n"]), u=unit, pp=pct("p"), b=bad)]
        else:
            lines = [T(r, "{Historically|Over the last two years}, $pp of $u {turn out to be|are} $b.", pp=pct("p"), u=unit, b=bad)]
        lines.append(T(r, "When one is $b, $f in $s of cases; when it is not, {the same thing happens|$f anyway} in $fp of cases.", b=bad, f=flag,
                       s=pct("s"), fp=pct("f")))
        if "s2" in i:
            lines.append(T(r, "Every flagged item then goes to an independent second check, which {confirms|flags} $s2 of $b ones and $f2 of clean ones "
                              "(its errors are independent of the first check's, given the true status).", s2=pct("s2"), b=bad, f2=pct("f2")))
        head = [T(r, "{Detection performance note|Screening statistics|Model monitoring summary} - $o", o=names.company())]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    code = names.code(rng.choice(["ID", "REF", "CASE", "ITEM"]))
    cond = f"both checks flag {code}" if d == 5 else flag.replace(" it", " " + code).replace(" fires", f" fires on {code}")
    qc = T(rng, "{Suppose|Say} $c. What is the probability that $x is actually $b?", c=cond, x=code, b=bad)
    return prob_item(rng, "pr_bayes", inp=inp, q={}, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "{Suppose|Say} $c. Is the probability that $x is actually $b above $t?", c=cond, x=code, b=bad, t=t),
                     decisive=["f", "p", "s"] + (["f2", "s2"] if d == 5 else []), style=rng.choice(["pct1", "pct2"]))


# ======================================================================================= expected-value choice
def pr_ev(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    cur = rng.choice(CURRENCIES[:7])
    nplans = {3: 3, 4: 3, 5: 4}[d]
    goal = rng.choice(["cost", "profit"])
    plan_names = rng.sample(["Plan A", "Plan B", "Plan C", "Plan D"] if rng.random() < 0.4 else
                            [f"Option {x}" for x in "PQRS"] if rng.random() < 0.5 else ["Carrier North", "Carrier Delta", "Carrier Vista", "Carrier Summit"], nplans)
    inp, outs = {}, {}
    for j in range(nplans):
        k = {3: 2, 4: rng.randint(2, 3), 5: 3}[d]
        cuts = sorted(rng.sample(range(1, 20), k - 1))
        probs = [F(b - a, 20) for a, b in zip([0] + cuts, cuts + [20])]
        outs[j] = k
        for m in range(k):
            inp[f"p|{j}|{m}"] = probs[m]
            inp[f"v|{j}|{m}"] = rng.randint(1, 400) * 100
        if d == 5:
            inp[f"fee|{j}"] = rng.randint(0, 50) * 100

    def ev(i, j):
        return sum(i[f"p|{j}|{m}"] * i[f"v|{j}|{m}"] for m in range(outs[j])) + (i.get(f"fee|{j}", 0) if goal == "cost" else -i.get(f"fee|{j}", 0))

    def best(i, margin=F(1, 100)):
        evs = [ev(i, j) for j in range(nplans)]
        order = sorted(range(nplans), key=lambda j: evs[j], reverse=(goal == "profit"))
        a, b = evs[order[0]], evs[order[1]]
        if abs(a - b) < margin * max(abs(a), abs(b), 1):
            raise Skip("tie")
        return order[0]

    g = best(inp)
    naive = (min if goal == "cost" else max)(range(nplans), key=lambda j: min(inp[f"v|{j}|{m}"] for m in range(outs[j])) if goal == "cost"
                                              else max(inp[f"v|{j}|{m}"] for m in range(outs[j])))
    rseed = rng.random()

    def render(i, missing, r):
        word = "cost" if goal == "cost" else "profit"
        lines = []
        for j in range(nplans):
            parts = []
            for m in range(outs[j]):
                v = "[not yet quoted]" if f"v|{j}|{m}" in missing else fmoney(i[f"v|{j}|{m}"], cur, 0)
                parts.append(f"{ffrac(i[f'p|{j}|{m}']) if r.random() < 0.3 else str(int(i[f'p|{j}|{m}'] * 100)) + '%'} chance of a {word} of {v}")
            fee = ""
            if f"fee|{j}" in i:
                fee = T(r, " There is also a fixed {set-up|onboarding|mobilisation} {charge|cost} of $f{, paid in every outcome|}.",
                        f="[fee not confirmed]" if f"fee|{j}" in missing else fmoney(i[f"fee|{j}"], cur, 0))
            lines.append(f"{plan_names[j]}: " + "; ".join(parts) + "." + fee)
        intro = T(r, "The team compared $n {alternatives|plans|options} for the {rollout|contract|peak season|migration}; the "
                     "{probabilities|chances} {below|listed} {come from|are based on} {last year's data|the vendor's track record|the pilot}.", n=nplans)
        head = [T(r, "{Decision memo|Options appraisal|Risk-weighted comparison} - $o", o=names.company())]
        return assemble(r, head, [intro] + lines, dom, Names(r, names.used), d)

    typ = rng.choice(["choice", "choice", "noul"])
    spec = {"kind": "pr_ev", "inputs": jenc(inp), "q": {"goal": goal, "outs": outs, "n": nplans, "type": typ}}
    if typ == "choice":
        qtext = T(rng, "Which {option|plan} has the {lowest expected cost|best expected cost}?" if goal == "cost" else
                  "Which {option|plan} has the highest expected profit?")
        field, gold, ov = choice_field(rng, qtext, plan_names[g], [plan_names[j] for j in range(nplans) if j != g],
                                       values={p: str(j) for j, p in enumerate(plan_names)})
        spec["option_values"] = ov

        def ans(i):
            return best(i, margin=F(0))
    else:
        j0 = rng.choice([naive, g] + list(range(nplans)))
        gold = j0 == g
        field = noul_field(T(rng, "Does $p have the {lowest|best} expected $w of all the options?" if goal == "cost" else
                             "Does $p have the highest expected $w of all the options?", p=plan_names[j0], w=goal))
        spec["q"]["plan"] = j0

        def ans(i):
            return best(i, margin=F(0)) == j0
    import random as _r
    state = render(inp, set(), _r.Random(rseed))
    unk = None
    keys = [k for k in inp if k.startswith("v|")]
    rng.shuffle(keys)
    for k in keys:
        base = ans(inp)
        for alt in (dict(inp, **{k: inp[k] * f}) for f in (2, 3, F(1, 3), F(1, 2), 5)):
            try:
                if ans(alt) != base:
                    unk = {"state": render(inp, {k}, _r.Random(rseed)), "reason": "insufficient_evidence",
                           "spec": {**spec, "removed": k, "witness": [str(inp[k]), str(alt[k])], "witness_alt": {k: str(alt[k])}}}
                    break
            except Skip:
                continue
        if unk:
            break
    return {"family": FAM, "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


# ======================================================================================= reliability
def pr_reliability(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["saas_ops", "energy", "manufacturing", "logistics", "healthcare_admin"])]
    comps = ["power supply", "network link", "database node", "cooling unit", "pump", "router", "UPS battery", "controller", "generator", "sensor hub"]
    shape = {3: rng.choice(["series2", "parallel2"]), 4: "par_series", 5: rng.choice(["two_groups", "kofn_series"])}[d]
    inp = {}
    names_c = rng.sample(comps, 6)
    for j in range(6):
        inp[f"q|{j}"] = F(rng.randint(1, 30), 100)

    def works(i, j):
        return 1 - i[f"q|{j}"]

    def compute(i, wrong=None):
        w = lambda j: works(i, j)  # noqa: E731
        if shape == "series2":
            return w(0) * w(1) if wrong != "add" else 1 - (i["q|0"] + i["q|1"])
        if shape == "parallel2":
            return 1 - i["q|0"] * i["q|1"] if wrong != "series" else w(0) * w(1)
        if shape == "par_series":
            par = 1 - i["q|0"] * i["q|1"] if wrong != "series" else w(0) * w(1)
            return par * w(2)
        if shape == "two_groups":
            g1 = 1 - i["q|0"] * i["q|1"]
            g2 = 1 - i["q|2"] * i["q|3"] * i["q|4"]
            if wrong == "series":
                return w(0) * w(1) * w(2) * w(3) * w(4)
            if wrong == "one":
                return g1
            return g1 * g2
        ps = [w(0), w(1), w(2)]
        two = sum(math.prod(ps[t] if t in c else 1 - ps[t] for t in range(3)) for c in itertools.combinations(range(3), 2)) + math.prod(ps)
        if wrong == "all":
            two = math.prod(ps)
        if wrong == "any":
            two = 1 - math.prod(1 - p for p in ps)
        return two * w(3)

    def wrongs(i):
        return [compute(i, x) for x in ("add", "series", "one", "all", "any")] + [1 - compute(i)]

    def render(i, missing, r):
        q = lambda j: "[failure rate not yet measured]" if f"q|{j}" in missing else f"{int(i[f'q|{j}'] * 100)}%"  # noqa: E731
        c = names_c
        if shape == "series2":
            desc = [f"The service needs both the {c[0]} and the {c[1]} to be up."]
            used = [0, 1]
        elif shape == "parallel2":
            desc = [f"The service stays up if at least one of two redundant units is up: a primary {c[0]} and a standby {c[1]}."]
            used = [0, 1]
        elif shape == "par_series":
            desc = [f"The site has two redundant {c[0]}s (unit A and unit B); it runs if at least one of them works AND the single {c[2]} works."]
            used = [0, 1, 2]
            c = [f"{names_c[0]} A", f"{names_c[0]} B", names_c[2]]
        elif shape == "two_groups":
            desc = [f"The system needs group 1 AND group 2. Group 1 is a pair of {c[0]}s (1a, 1b) and works if either works; group 2 is three "
                    f"{c[2]}s (2a, 2b, 2c) and works if any of them works."]
            used = [0, 1, 2, 3, 4]
            c = [f"{names_c[0]} 1a", f"{names_c[0]} 1b", f"{names_c[2]} 2a", f"{names_c[2]} 2b", f"{names_c[2]} 2c"]
        else:
            desc = [f"Three {c[0]}s (X, Y, Z) run in a 2-out-of-3 arrangement: that stage works if at least two of them work. The system also needs the "
                    f"single {c[3]}."]
            used = [0, 1, 2, 3]
            c = [f"{names_c[0]} X", f"{names_c[0]} Y", f"{names_c[0]} Z", names_c[3]]
        rows = [[c[j], q(j)] for j in used]
        tbl = md_table(["Component", "Probability of failing during the window"], rows)
        notes = [T(r, "Failures are independent of each other{ over the maintenance window|}.")] + desc
        head = [T(r, "{Resilience review|Availability model|Redundancy assessment} - $o, system $s", o=names.company(), s=sysname)]
        return assemble(r, head, notes + [tbl], dom, Names(r, names.used), d)

    used = {"series2": 2, "parallel2": 2, "par_series": 3, "two_groups": 5, "kofn_series": 4}[shape]
    for j in range(used, 6):
        inp.pop(f"q|{j}")
    sysname = names.code(rng.choice(["SYS", "SITE", "SVC"]))
    qc = T(rng, "What is the probability that $s stays up through the {window|maintenance window}?", s=sysname)
    return prob_item(rng, "pr_reliability", inp=inp, q={"shape": shape}, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "Is the probability that $s stays up through the window above $t?", s=sysname, t=t),
                     decisive=list(inp), style=rng.choice(["pct2", "pct2", "frac"]))


# ======================================================================================= repeated independent trials
TRIALS = [("delivery attempts", "succeeds", "succeed"), ("sales calls", "converts", "convert"), ("server restarts", "comes back cleanly", "come back cleanly"),
          ("job applications", "gets an interview", "get an interview"), ("test batches", "passes first time", "pass first time"),
          ("grant bids", "is funded", "are funded"), ("payment retries", "clears", "clear")]


def pr_binomial(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    what, succ, succ_pl = rng.choice(TRIALS)
    n = rng.randint(3, {3: 5, 4: 8, 5: 12}[d])
    den = rng.choice([2, 3, 4, 5, 6, 10])
    p = F(rng.randint(1, den - 1), den)
    mode = {3: "atleast", 4: rng.choice(["atleast", "exactly"]), 5: rng.choice(["atmost", "between"])}[d]
    k = rng.randint(1, n - 1)
    k2 = min(n, k + rng.randint(1, 3))
    inp = {"n": n, "p": p}
    q = {"mode": mode, "k": k, "k2": k2}

    def pmf(i, x):
        return math.comb(i["n"], x) * i["p"] ** x * (1 - i["p"]) ** (i["n"] - x)

    def compute(i):
        if mode == "atleast":
            return sum(pmf(i, x) for x in range(k, i["n"] + 1))
        if mode == "exactly":
            return pmf(i, k)
        if mode == "atmost":
            return sum(pmf(i, x) for x in range(0, k + 1))
        return sum(pmf(i, x) for x in range(k, min(k2, i["n"]) + 1))

    def wrongs(i):
        return [pmf(i, k), sum(pmf(i, x) for x in range(k + 1, i["n"] + 1)), 1 - compute(i), sum(pmf(i, x) for x in range(0, k)),
                i["p"] ** k]

    def render(i, missing, r):
        lines = [T(r, "{Each|Every one} of the next $n $w {independently |}$s with probability $p.", n=i["n"] if "n" not in missing else "[several]",
                   w=what, s=succ, p="[not estimated]" if "p" in missing else (ffrac(i["p"]) if (r.random() < 0.5 or (i["p"] * 100).denominator != 1)
                                                            else f"{int(i['p'] * 100)}%")),
                 T(r, "{Outcomes|Results} {do not affect each other|are independent}.")]
        head = [T(r, "{Pipeline forecast|Planning assumptions|Capacity note} - $o", o=names.company())]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    owner = names.person()
    what = f"{what} {owner} has lined up"
    s = {"atleast": f"at least {k} of the {n} {what} {succ_pl}", "exactly": f"exactly {k} of the {n} {what} {succ if k == 1 else succ_pl}",
         "atmost": f"at most {k} of the {n} {what} {succ_pl}", "between": f"between {k} and {min(k2, n)} (inclusive) of the {n} {what} {succ_pl}"}[mode]
    qc = T(rng, "What is the probability that $s?", s=s)
    return prob_item(rng, "pr_binomial", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "Is the probability that $s greater than $t?", s=s, t=t), decisive=["p"],
                     style=rng.choice(["pct2", "pct2", "frac"]))


# ======================================================================================= custom wheels
def pr_wheels(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    nw = 3 if d == 5 else 2
    inp = {}
    wheels = []
    for w in range(nw):
        k = rng.randint(3, 6)
        vals = sorted(rng.sample(range(1, 13), k))
        wts = [rng.randint(1, 4) for _ in range(k)]
        wheels.append(vals)
        for v, wt in zip(vals, wts):
            inp[f"w|{w}|{v}"] = wt
    mode = {3: rng.choice(["sum_ge", "first_gt"]), 4: rng.choice(["sum_ge", "prod_even", "sum_eq"]), 5: rng.choice(["sum_ge", "cond_even"])}[d]
    tgt = rng.randint(4, 12 * nw - 4)
    q = {"mode": mode, "t": tgt, "wheels": wheels}

    def dist(i, w):
        tot = sum(i[f"w|{w}|{v}"] for v in wheels[w])
        return {v: F(i[f"w|{w}|{v}"], tot) for v in wheels[w]}

    def compute(i, eq_weights=False):
        ds = [dist(i, w) if not eq_weights else {v: F(1, len(wheels[w])) for v in wheels[w]} for w in range(nw)]
        num = den = F(0)
        for combo in itertools.product(*[list(x.items()) for x in ds]):
            pr = math.prod(p for _, p in combo)
            vs = [v for v, _ in combo]
            s = sum(vs)
            if mode == "sum_ge":
                num += pr * (s >= tgt)
            elif mode == "sum_eq":
                num += pr * (s == tgt)
            elif mode == "first_gt":
                num += pr * (vs[0] > vs[1])
            elif mode == "prod_even":
                num += pr * (math.prod(vs) % 2 == 0)
            else:
                if s % 2 == 0:
                    den += pr
                    num += pr * (vs[0] > vs[-1])
        if mode == "cond_even":
            if den == 0:
                raise Skip("empty condition")
            return num / den
        return num

    def wrongs(i):
        out = [compute(i, eq_weights=True)]
        v = compute(i)
        out += [1 - v, v * F(3, 4), v + (1 - v) / 5]
        return out

    def render(i, missing, r):
        lines = [T(r, "{At the staff raffle|For the stand-up icebreaker|In the onboarding game} each player spins $n {wheels|spinners}. Each wheel is divided "
                      "into sectors; a sector's chance of being hit is proportional to its number of equal slices.", n=nw)]
        for w in range(nw):
            parts = []
            for v in wheels[w]:
                k = f"w|{w}|{v}"
                parts.append(f"value {v}: {'[slice count smudged]' if k in missing else i[k]} slice{'s' if i[k] != 1 else ''}")
            lines.append(f"Wheel {'ABC'[w]} - " + "; ".join(parts) + ".")
        lines.append(T(r, "The wheels are spun independently."))
        head = [T(r, "{Game rules|Raffle mechanics|Icebreaker instructions} - $o", o=names.company())]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    s = {"sum_ge": f"the values shown add up to at least {tgt}", "sum_eq": f"the values shown add up to exactly {tgt}",
         "first_gt": "wheel A shows a higher value than wheel B", "prod_even": "the product of the values shown is even",
         "cond_even": f"wheel A shows a higher value than wheel {'ABC'[nw - 1]}"}[mode]
    player = names.person()
    s = s.replace("the values shown", f"the values {player} gets").replace("wheel A shows", f"{player}'s wheel A shows")
    pre = f"Given that {player}'s values add up to an even number, " if mode == "cond_even" else ""
    qc = (pre + "what is the probability that " + s + "?")
    qc = qc[:1].upper() + qc[1:]
    return prob_item(rng, "pr_wheels", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: pre + ("i" if pre else "I") + T(rng, "is the probability that $s {greater than|above} $t?", s=s, t=t)[1:],
                     decisive=list(inp),
                     style=rng.choice(["frac", "frac", "pct2"]))


# ======================================================================================= expected payout / expected cost
def pr_expected(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    cur = rng.choice(CURRENCIES[:7])
    k = {3: 3, 4: 4, 5: 5}[d]
    cuts = sorted(rng.sample(range(1, 40), k - 1))
    probs = [F(b - a, 40) for a, b in zip([0] + cuts, cuts + [40])]
    inp = {}
    for m in range(k):
        inp[f"p|{m}"] = probs[m]
        inp[f"L|{m}"] = 0 if m == 0 else rng.randint(1, 120) * 250
    if d >= 4:
        inp["ded"] = rng.randint(1, 20) * 250
    if d == 5:
        inp["cap"] = rng.randint(20, 100) * 500
    inp["events"] = rng.randint(1, 4) if d >= 4 else 1

    def payout(i, L, ded=True, cap=True):
        x = L - (i.get("ded", 0) if ded else 0)
        x = max(0, x)
        if cap and "cap" in i:
            x = min(x, i["cap"])
        return x

    def compute(i, **kw):
        return i["events"] * sum(i[f"p|{m}"] * payout(i, i[f"L|{m}"], **kw) for m in range(k))

    def wrongs(i):
        return [compute(i, ded=False), compute(i, cap=False), sum(i[f"p|{m}"] * payout(i, i[f"L|{m}"]) for m in range(k)),
                i["events"] * (sum(i[f"p|{m}"] * i[f"L|{m}"] for m in range(k)) - i.get("ded", 0))]

    def render(i, missing, r):
        rows = [["no loss" if i[f"L|{m}"] == 0 else ("[assessment pending]" if f"L|{m}" in missing else fmoney(i[f"L|{m}"], cur, 0)),
                 f"{int(i[f'p|{m}'] * 100 * 10) / 10:g}%" if (i[f'p|{m}'] * 1000).denominator == 1 else ffrac(i[f"p|{m}"])] for m in range(k)]
        tbl = md_table(["Loss per incident", "Probability"], rows)
        lines = [T(r, "The {cover|policy|service credit scheme} is expected to see $e {covered incident|incident}$s this year; each one independently "
                      "follows the loss distribution below.", e=i["events"], s="s" if i["events"] != 1 else "")]
        if "ded" in i:
            lines.append(T(r, "For each incident the {insurer|provider} pays the loss minus a deductible of $x (nothing if the loss is at or below it)$c.",
                           x="[deductible not agreed]" if "ded" in missing else fmoney(i["ded"], cur, 0),
                           c=(", up to a maximum payout of " + ("[cap not agreed]" if "cap" in missing else fmoney(i["cap"], cur, 0)) + " per incident") if "cap" in i else ""))
        else:
            lines.append(T(r, "The {insurer|provider} pays the full loss for each incident."))
        head = [T(r, "{Insurance renewal note|Risk budget|Exposure estimate} - $o, $p", o=names.company(), p=pol)]
        return assemble(r, head, lines + [tbl], dom, Names(r, names.used), d)

    pol = names.code(rng.choice(["POL", "COV", "SCH"]))
    qc = T(rng, "What is the expected total payout under $p for the year?", p=pol)
    qn = lambda t: T(rng, "Is the expected total payout under $p for the year above $t?", p=pol, t=t)  # noqa: E731
    decisive = [f"L|{m}" for m in range(1, k)] + (["ded"] if "ded" in inp else []) + (["cap"] if "cap" in inp else [])
    return numeric_item(rng, FAM, "pr_expected", inp=inp, q={"k": k}, compute=compute, wrongs=wrongs, render=render, q_choice=qc, q_noul=qn,
                        places=2, fmt=lambda v: fmoney(v, cur), decisive=decisive, witness_factors=[2, 0.5, 3, 1.5])


# ======================================================================================= monthly transitions
STATES = [("Active", "At risk", "Churned"), ("Current", "Overdue", "Written off"), ("Healthy", "Degraded", "Failed"), ("Enrolled", "Suspended", "Withdrawn")]


def pr_markov(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    S = list(rng.choice(STATES))
    steps = {3: 2, 4: 2, 5: 3}[d]
    inp = {}
    for a in S[:2]:
        cuts = sorted(rng.sample(range(1, 20), 2))
        ps = [F(x, 20) for x in (cuts[0], cuts[1] - cuts[0], 20 - cuts[1])]
        for b, p in zip(S, ps):
            inp[f"t|{a}|{b}"] = p
    start = S[0] if d < 5 else rng.choice(S[:2])
    target = rng.choice(S)
    q = {"S": S, "steps": steps, "start": start, "target": target}

    def compute(i, st=steps, absorbing=True):
        v = {s: F(int(s == start)) for s in S}
        for _ in range(st):
            nv = {s: F(0) for s in S}
            for a in S:
                if a == S[2]:
                    nv[a] += v[a] if absorbing else 0
                    continue
                for b in S:
                    nv[b] += v[a] * i[f"t|{a}|{b}"]
            v = nv
        return v[target]

    def wrongs(i):
        return [compute(i, st=steps - 1), compute(i, st=steps + 1), compute(i, absorbing=False), i[f"t|{start}|{target}"] ** steps if start != S[2] else None]

    def render(i, missing, r):
        blank = {k.split("|")[1] for k in missing}
        rows = [[a] + ["n/a*" if a in blank else f"{int(i[f't|{a}|{b}'] * 100)}%" for b in S] for a in S[:2]]
        rows.append([S[2]] + ["0%", "0%", "100%"])
        tbl = md_table(["From \\ to (next month)"] + S, rows)
        lines = [T(r, "Each month every {account|record|unit} moves between states according to the table, independently of its history. "
                      "$c is permanent.", c=S[2])]
        if missing:
            lines.append("\\* The monitoring feed for this row was down; its values are unknown.")
        head = [T(r, "{Portfolio health model|Monthly state transitions|Status migration matrix} - $o", o=names.company())]
        return assemble(r, head, lines + [tbl], dom, Names(r, names.used), d)

    acc = names.code(rng.choice(["ACC", "REC", "UNIT"]))
    qc = T(rng, "$a is $s now. What is the probability that $a is $t exactly $n months from now?", a=acc, s=start, t=target, n=steps)
    decisive = [f"t|{a}|{b}" for a in S[:2] for b in S]

    def mperturb(k, i):
        a = k.split("|")[1]
        out = []
        for b in S:
            k2 = f"t|{a}|{b}"
            if k2 != k and i[k2] != i[k]:
                out.append(dict(i, **{k: i[k2], k2: i[k]}))
        return out
    return prob_item(rng, "pr_markov", perturb=mperturb, inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, qc=qc,
                     qn_about=lambda t: T(rng, "$a is $s now. Is the probability that $a is $tt in $n months above $t?", a=acc, s=start, tt=target, n=steps, t=t),
                     decisive=decisive)


KINDS = {"pr_draws": pr_draws, "pr_table": pr_table, "pr_bayes": pr_bayes, "pr_ev": pr_ev, "pr_reliability": pr_reliability,
         "pr_binomial": pr_binomial, "pr_wheels": pr_wheels, "pr_expected": pr_expected, "pr_markov": pr_markov}
KIND_FAMILY = {k: FAM for k in KINDS}
