"""Family ordinal (type score, 3-10 levels 0..n-1 with level descriptions): graded evidence mapped to a level by explicit
rubric rules. Kinds: points bands with caps, cumulative ladders (highest level whose requirements all hold), weighted
sub-scores with caps, impact x urgency priority matrices with overrides, and demerit counting from an inspection log.
All thresholds are compared exactly ("at least" / "at most"); unknown variants blank one decisive piece of evidence."""
from __future__ import annotations

import random
from fractions import Fraction as F

from .common import DOMAINS, Names, Skip, T, assemble, score_field
from .numeric_base import jenc, md_table

FAM = "ordinal"

# metric id -> (label, unit, lo, hi, decimals, higher_is_better)
METRICS = {
    "otd": ("On-time delivery rate", "%", 70, 100, 1, True), "defect": ("Defect rate", "%", 0.1, 6, 1, False),
    "resp": ("Median first-response time", " hours", 1, 72, 0, False), "csat": ("Customer satisfaction", " / 5", 2.0, 5.0, 1, True),
    "uptime": ("Service uptime", "%", 97.0, 100.0, 2, True), "findings": ("Open audit findings", "", 0, 12, 0, False),
    "training": ("Mandatory training completion", "%", 50, 100, 0, True), "nps": ("Net promoter score", "", -30, 70, 0, True),
    "cov": ("Automated test coverage", "%", 30, 95, 0, True), "lead": ("Average lead time", " days", 2, 40, 0, False),
    "attend": ("Attendance", "%", 60, 100, 0, True), "compl": ("Complaints per 1,000 orders", "", 0, 15, 1, False),
    "fill": ("Order fill rate", "%", 80, 100, 1, True), "stale": ("Tickets open longer than 30 days", "", 0, 40, 0, False),
    "margin": ("Gross margin", "%", 5, 60, 1, True), "dso": ("Days sales outstanding", " days", 20, 90, 0, False),
    "turnover": ("Staff turnover (annualised)", "%", 2, 35, 1, False), "yield": ("First-pass yield", "%", 80, 100, 1, True),
    "patch": ("Critical patches applied within 14 days", "%", 40, 100, 0, True), "backlog": ("Claims backlog", " claims", 0, 400, 0, False),
}
BOOLS = {"iso": ("Holds a current ISO 9001 certificate", True), "dpa": ("Signed data-processing agreement on file", True),
         "bcp": ("Business-continuity plan tested in the last 12 months", True), "major": ("Major incident during the period", False),
         "sanction": ("Regulatory sanction during the period", False), "mfa": ("Multi-factor authentication enforced for all staff", True),
         "insurance": ("Professional indemnity insurance in force", True), "breach": ("Personal-data breach reported during the period", False)}
CONTEXTS = [("supplier", "Supplier scorecard", "supplier rating", "procurement"), ("vendor", "Vendor risk assessment", "risk tier", "legal_compliance"),
            ("service", "Service health review", "health grade", "saas_ops"), ("branch", "Branch performance review", "performance band", "retail"),
            ("programme", "Programme maturity assessment", "maturity level", "nonprofit"), ("account", "Account health check", "account health tier", "saas_ops"),
            ("site", "Site audit", "audit grade", "manufacturing"), ("clinic", "Clinic operations review", "operations grade", "healthcare_admin"),
            ("depot", "Depot performance review", "depot rating", "logistics"), ("school", "Department quality review", "quality level", "education")]
LEVEL_NAMES = {3: ["Weak", "Adequate", "Strong"], 4: ["Poor", "Fair", "Good", "Excellent"], 5: ["Critical", "Weak", "Adequate", "Good", "Outstanding"],
               6: ["Failing", "Poor", "Below standard", "Meets standard", "Good", "Exemplary"],
               7: ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6"]}


def lname(n, k):
    if n in LEVEL_NAMES:
        return LEVEL_NAMES[n][k]
    return f"Tier {k}"


def fmt_val(m, v):
    label, unit, lo, hi, dp, hb = METRICS[m]
    return f"{float(v):.{dp}f}{unit}" if dp else f"{int(v)}{unit}"


def rand_val(rng, m):
    label, unit, lo, hi, dp, hb = METRICS[m]
    if dp == 0:
        return F(rng.randint(int(lo), int(hi)))
    sc = 10 ** dp
    return F(rng.randint(int(lo * sc), int(hi * sc)), sc)


def cut_text(m, c):
    return fmt_val(m, c)


def score_item(rng, kind, *, inp, q, compute, render, levels, question, decisive, perturb, d):
    ref_names = Names(rng)
    ref = T(rng, "{Assessment reference|Review ID|Reference} $c; {completed by|assessor:|lead reviewer} $p; {dated|signed off} $dd.",
            c=ref_names.code(rng.choice(["AR", "RV", "QA"])), p=ref_names.person(), dd=f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(2024, 2028)}")
    base_render = render

    def render(i, missing, r):  # noqa: F811
        body = base_render(i, missing, r)
        head, _, rest = body.partition("\n\n")
        return head + "\n\n" + ref + "\n\n" + rest
    val = compute(inp)
    if not (0 <= val < len(levels)):
        raise Skip("level out of range")
    field = score_field(question, levels)
    spec = {"kind": kind, "inputs": jenc(inp), "q": jenc(q)}
    rseed = rng.random()
    state = render(inp, set(), random.Random(rseed))
    unk = None
    keys = [k for k in decisive if k in inp]
    rng.shuffle(keys)
    for k in keys:
        for alt in perturb(k, inp):
            try:
                if compute(alt) != val:
                    unk = {"state": render(inp, {k}, random.Random(rseed)), "reason": "insufficient_evidence",
                           "spec": {**spec, "removed": k, "witness_alt": {kk: jenc(alt[kk]) for kk in alt if alt[kk] != inp[kk]}}}
                    break
            except Skip:
                continue
        if unk:
            break
    return {"family": FAM, "state": state, "field": field, "gold": val, "unknown_reason": None, "spec": spec, "unk": unk}


def generic_perturb(k, inp, metric_of=None):
    v = inp[k]
    out = []
    if isinstance(v, bool):
        return [dict(inp, **{k: not v})]
    m = metric_of(k) if metric_of else None
    if m:
        label, unit, lo, hi, dp, hb = METRICS[m]
        for x in (lo, hi, (lo + hi) / 2, lo + (hi - lo) / 4, hi - (hi - lo) / 4):
            xv = F(round(x * 10 ** dp), 10 ** dp)
            if xv != v:
                out.append(dict(inp, **{k: xv}))
        return out
    if isinstance(v, (int, F)):
        for x in (v + 1, v - 1, v + 2, v * 2, v // 2 if isinstance(v, int) else v / 2):
            if x != v and x >= 0:
                out.append(dict(inp, **{k: x}))
    return out


# ======================================================================================= points bands
def tiers_for(rng, m, npts):
    """Cut points c_1 < ... < c_npts (for higher-is-better: points = number of cuts reached)."""
    label, unit, lo, hi, dp, hb = METRICS[m]
    sc = 10 ** dp
    cuts = sorted({F(rng.randint(int((lo + (hi - lo) * 0.15) * sc), int((hi - (hi - lo) * 0.1) * sc)), sc) for _ in range(npts * 3)})
    if len(cuts) < npts:
        raise Skip("cuts")
    cuts = sorted(rng.sample(cuts, npts))
    return cuts


def points(m, v, cuts):
    hb = METRICS[m][5]
    if hb:
        return sum(1 for c in cuts if v >= c)
    return sum(1 for c in cuts if v <= c)


def or_points(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    who, title, rating, domk = rng.choice(CONTEXTS)
    dom = DOMAINS.get(domk, DOMAINS["finance"]) if domk in DOMAINS else DOMAINS["finance"]
    subj = names.company() if who in ("supplier", "vendor", "account") else f"{names.city()} {who} {names.code('ID')}"
    nm = {3: 3, 4: rng.randint(4, 5), 5: rng.randint(5, 6)}[d]
    ms = rng.sample(sorted(METRICS), nm)
    inp, cuts, weights = {}, {}, {}
    for m in ms:
        inp[m] = rand_val(rng, m)
        cuts[m] = tiers_for(rng, m, rng.randint(2, 4))
        weights[m] = 2 if (d == 5 and rng.random() < 0.3) else 1
    pmax = sum(len(cuts[m]) * weights[m] for m in ms)
    n = rng.randint(3, min(10, pmax + 1))
    bands = sorted(rng.sample(range(1, pmax + 1), n - 1))  # level k needs total >= bands[k-1]
    cap = None
    if d >= 4:
        b = rng.choice(sorted(BOOLS))
        inp[b] = rng.random() < 0.5
        cap = (b, rng.randint(0, max(0, n - 2)))
    q = {"ms": ms, "cuts": {m: [str(c) for c in cuts[m]] for m in ms}, "weights": weights, "bands": bands, "cap": cap, "n": n}

    def total(i):
        return sum(points(m, i[m], cuts[m]) * weights[m] for m in ms)

    def compute(i):
        t = total(i)
        lvl = sum(1 for b in bands if t >= b)
        if cap is not None:
            b, c = cap
            bad = i[b] != BOOLS[b][1]
            if bad:
                lvl = min(lvl, c)
        return lvl

    def render(i, missing, r):
        rows = [[METRICS[m][0], "[not reported]" if m in missing else fmt_val(m, i[m])] for m in ms]
        if cap is not None:
            b = cap[0]
            rows.append([BOOLS[b][0], "[not confirmed]" if b in missing else ("Yes" if i[b] else "No")])
        r.shuffle(rows)
        tbl = md_table(["Evidence", "Value for the period"], rows)
        rules = []
        for m in ms:
            hb = METRICS[m][5]
            cs = cuts[m] if not hb else list(reversed(cuts[m]))
            parts = []
            for k, c in enumerate(cs):
                pts = len(cs) - k
                parts.append(f"{pts} point{'s' if pts != 1 else ''} if {'at least' if hb else 'at most'} {cut_text(m, c)}")
            txt = f"{METRICS[m][0]}: " + "; otherwise ".join(parts) + "; otherwise 0 points."
            if weights[m] == 2:
                txt += " Points for this criterion count double."
            rules.append("- " + txt)
        tail = T(r, "The {$r|overall $r} is the highest level whose minimum total the points reach (see the level definitions).", r=rating)
        if cap is not None:
            b, c = cap
            cond = BOOLS[b][0].lower() if not BOOLS[b][1] else "no " + BOOLS[b][0][0].lower() + BOOLS[b][0][1:]
            tail += " " + T(r, "{Override|Cap}: if the record shows $c, the $r cannot be higher than level $k ($nm), whatever the points.",
                            c=("a " + cond) if not BOOLS[b][1] else f"'{BOOLS[b][0]}' answered No", r=rating, k=c, nm=lname(n, c))
        head = [T(r, "$t - $s ({period|reporting period|review period}: $p)", t=title, s=subj, p=r.choice(["Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY"]) +
                  f" {r.randint(2024, 2028)}"),
                T(r, "{Scoring rules|How points are awarded} (each criterion is scored separately, then the points are added):") + "\n" + "\n".join(rules), tail]
        return assemble(r, head, [tbl], dom, Names(r, names.used), d)

    levels = []
    for k in range(n):
        lo_ = 0 if k == 0 else bands[k - 1]
        hi_ = (bands[k] - 1) if k < n - 1 else pmax
        levels.append(f"Level {k} - {lname(n, k)}: total of {lo_} to {hi_} points" if lo_ != hi_ else f"Level {k} - {lname(n, k)}: exactly {lo_} point{'s' if lo_ != 1 else ''}")
    question = T(rng, "{Which level|What $r} does $s {receive|get} for the period under these rules?", r=rating, s=subj)

    def pert(k, i):
        return generic_perturb(k, i, metric_of=lambda kk: kk if kk in METRICS else None)
    return score_item(rng, "or_points", inp=inp, q=q, compute=compute, render=render, levels=levels, question=question,
                      decisive=list(inp), perturb=pert, d=d)


# ======================================================================================= cumulative ladder
def or_ladder(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    who, title, rating, domk = rng.choice(CONTEXTS)
    dom = DOMAINS.get(domk, DOMAINS["finance"])
    subj = names.company() if who in ("supplier", "vendor", "account") else f"{names.city()} {who} {names.code('ID')}"
    n = {3: rng.randint(3, 4), 4: rng.randint(4, 6), 5: rng.randint(5, 8)}[d]
    pool_m = rng.sample(sorted(METRICS), 7)
    pool_b = rng.sample(sorted(BOOLS), 3)
    reqs = {}  # level k -> list of requirement tuples
    inp = {}
    used = []
    for k in range(1, n):
        rk = []
        for _ in range(1 if d == 3 else rng.randint(1, 2)):
            if rng.random() < 0.25 and pool_b:
                b = pool_b.pop()
                rk.append(("bool", b, BOOLS[b][1]))
                inp[b] = rng.random() < 0.6 if BOOLS[b][1] else rng.random() < 0.4
            else:
                prev = [x for x in used if x[0] == "num"]
                if prev and rng.random() < 0.35:
                    # tighten an earlier metric (a higher level needs a stricter threshold)
                    _, m, c = rng.choice(prev)
                    hb = METRICS[m][5]
                    label, unit, lo, hi, dp, _ = METRICS[m]
                    step = F(max(1, int(rng.randint(1, 5) * (hi - lo) * 10 ** dp / 20)), 10 ** dp)
                    c2 = c + step if hb else c - step
                    rk.append(("num", m, c2))
                else:
                    if not pool_m:
                        continue
                    m = pool_m.pop()
                    label, unit, lo, hi, dp, hb = METRICS[m]
                    sc = 10 ** dp
                    c = F(rng.randint(int((lo + (hi - lo) * 0.3) * sc), int((hi - (hi - lo) * 0.2) * sc)), sc)
                    rk.append(("num", m, c))
                    inp[m] = rand_val(rng, m)
        reqs[k] = rk
        used += rk
    for k in range(1, n):
        if not reqs[k]:
            raise Skip("empty level")
    # choose the target level first, then evidence that passes levels 1..L and fails one requirement of L+1
    L = rng.randrange(n)
    fail = rng.choice(reqs[L + 1]) if L + 1 < n else None
    for key in list(inp):
        if key in BOOLS:
            need = [x[2] for kk in range(1, L + 1) for x in reqs[kk] if x[1] == key]
            if need:
                inp[key] = need[0]
            elif fail is not None and fail[1] == key:
                inp[key] = not fail[2]
            continue
        label, unit, lo, hi, dp, hb = METRICS[key]
        sc = 10 ** dp
        lo_u, hi_u = int(lo * sc), int(hi * sc)
        for kk in range(1, L + 1):
            for x in reqs[kk]:
                if x[1] == key:
                    cu = int(x[2] * sc)
                    lo_u, hi_u = (max(lo_u, cu), hi_u) if hb else (lo_u, min(hi_u, cu))
        if fail is not None and fail[1] == key:
            cu = int(fail[2] * sc)
            lo_u, hi_u = (lo_u, min(hi_u, cu - 1)) if hb else (max(lo_u, cu + 1), hi_u)
        if lo_u > hi_u:
            raise Skip("infeasible target")
        inp[key] = F(rng.randint(lo_u, hi_u), sc)

    def ok(i, r_):
        if r_[0] == "bool":
            return i[r_[1]] == r_[2]
        m, c = r_[1], r_[2]
        return i[m] >= c if METRICS[m][5] else i[m] <= c

    def compute(i):
        lvl = 0
        for k in range(1, n):
            if all(ok(i, r_) for r_ in reqs[k]):
                lvl = k
            else:
                break
        return lvl

    if compute(inp) != L:
        raise Skip("target missed")

    def rtext(r_):
        if r_[0] == "bool":
            return f"'{BOOLS[r_[1]][0]}' must be {'Yes' if r_[2] else 'No'}"
        m, c = r_[1], r_[2]
        return f"{METRICS[m][0].lower()} {'at least' if METRICS[m][5] else 'at most'} {fmt_val(m, c)}"

    def render(i, missing, r):
        ev = []
        for k in inp:
            if k in METRICS:
                ev.append([METRICS[k][0], "[awaiting data]" if k in missing else fmt_val(k, i[k])])
            else:
                ev.append([BOOLS[k][0], "[not yet verified]" if k in missing else ("Yes" if i[k] else "No")])
        r.shuffle(ev)
        tbl = md_table(["Evidence", "Recorded value"], ev)
        rules = [f"- Level {k} ({lname(n, k)}) requires: " + "; ".join(rtext(x) for x in reqs[k]) + "." for k in range(1, n)]
        head = [T(r, "$t - $s", t=title, s=subj),
                T(r, "{Levels are cumulative|The levels build on each other}: a {$w|subject} is at the highest level $k such that the requirements of "
                     "every level from 1 up to $k are all met. {Meeting a higher level's requirements does not help if a lower level's are missed.|"
                     "Missing any lower level's requirement stops the climb there.} Level 0 ($z) has no requirements.", w=who, k="k", z=lname(n, 0)),
                "\n".join(rules)]
        return assemble(r, head, [tbl], dom, Names(r, names.used), d)

    levels = [f"Level 0 - {lname(n, 0)}: level 1 not met"] + \
             [f"Level {k} - {lname(n, k)}: levels 1-{k} met" + (f", level {k + 1} not" if k < n - 1 else "") for k in range(1, n)]
    question = T(rng, "{At which level|On which level} {does $s sit|is $s placed} {under this ladder|according to the requirements}?", s=subj)

    def pert(k, i):
        return generic_perturb(k, i, metric_of=lambda kk: kk if kk in METRICS else None)
    q = {"n": n, "reqs": {str(k): [[x[0], x[1], str(x[2]) if not isinstance(x[2], bool) else x[2]] for x in v] for k, v in reqs.items()}}
    return score_item(rng, "or_ladder", inp=inp, q=q, compute=compute, render=render, levels=levels, question=question, decisive=list(inp), perturb=pert, d=d)


# ======================================================================================= weighted sub-scores
CRIT = ["documentation", "responsiveness", "technical quality", "safety culture", "value for money", "innovation", "communication", "reliability",
        "accessibility", "sustainability", "data handling", "training", "scalability", "stakeholder engagement"]


def or_weighted(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    who, title, rating, domk = rng.choice(CONTEXTS)
    dom = DOMAINS.get(domk, DOMAINS["finance"])
    subj = names.company() if rng.random() < 0.6 else names.product()
    k = {3: 3, 4: rng.randint(4, 5), 5: rng.randint(5, 6)}[d]
    crits = rng.sample(CRIT, k)
    scale = rng.choice([5, 10])
    cuts = sorted(rng.sample(range(5, 95), k - 1))
    ws = [b - a for a, b in zip([0] + cuts, cuts + [100])]
    inp = {f"s|{c}": rng.randint(1, scale) for c in crits}
    two = None
    if d == 5:
        two = rng.choice(crits)
        inp[f"s2|{two}"] = rng.randint(1, scale)
    n = rng.randint(3, 7)
    th = sorted({F(rng.randint(15 * scale, 90 * scale), 100) for _ in range(20)})
    if len(th) < n - 1:
        raise Skip("thresholds")
    th = sorted(rng.sample(th, n - 1))
    cap = (rng.randint(1, 2), rng.randint(0, n - 2)) if d >= 4 else None  # any sub-score at or below cap[0] -> level <= cap[1]

    def subs(i):
        out = {}
        for c in crits:
            v = F(i[f"s|{c}"])
            if two == c:
                v = (v + i[f"s2|{two}"]) / 2
            out[c] = v
        return out

    def compute(i):
        sc = subs(i)
        avg = sum(sc[c] * w for c, w in zip(crits, ws)) / 100
        lvl = sum(1 for t in th if avg >= t)
        if cap is not None and any(sc[c] <= cap[0] for c in crits):
            lvl = min(lvl, cap[1])
        return lvl

    def render(i, missing, r):
        rows = []
        for c, w in zip(crits, ws):
            sv = "[score missing]" if f"s|{c}" in missing else str(i[f"s|{c}"])
            if two == c:
                s2 = "[score missing]" if f"s2|{c}" in missing else str(i[f"s2|{c}"])
                sv = f"assessor 1: {sv}; assessor 2: {s2}"
            rows.append([c.capitalize(), f"{w}%", sv])
        tbl = md_table(["Criterion", "Weight", f"Score (1-{scale})"], rows)
        rules = [T(r, "The overall score is the weighted average of the criterion scores (weights as shown, summing to 100%).")]
        if two:
            rules.append(T(r, "Where two assessors scored a criterion, the criterion score is the mean of their two scores."))
        rules.append("Level thresholds on the overall score: " + "; ".join(f"level {j + 1} ({lname(n, j + 1)}) needs at least {float(t):.2f}" for j, t in enumerate(th))
                     + f"; below {float(th[0]):.2f} is level 0 ({lname(n, 0)}).")
        if cap:
            rules.append(T(r, "{Cap|Safeguard}: if any criterion score is $a or lower, the level cannot exceed $b ($nm).", a=cap[0], b=cap[1], nm=lname(n, cap[1])))
        head = [T(r, "{Evaluation panel summary|Assessment grid|Scoring sheet} - $s", s=subj)]
        return assemble(r, head, [tbl] + rules, dom, Names(r, names.used), d)

    for t in th:
        if (t * 100).denominator != 1:
            raise Skip("threshold precision")
    levels = [f"Level 0 - {lname(n, 0)}: overall score below {float(th[0]):.2f}"] + \
             [f"Level {j} - {lname(n, j)}: overall score at least {float(th[j - 1]):.2f}" + (f" and below {float(th[j]):.2f}" if j < n - 1 else "")
              for j in range(1, n)]
    question = T(rng, "{Which level does $s receive|What level is $s awarded} under the scoring rules?", s=subj)

    def pert(k, i):
        return [dict(i, **{k: x}) for x in (1, scale, (scale + 1) // 2, max(1, i[k] - 2), min(scale, i[k] + 2)) if x != i[k]]
    q = {"crits": crits, "ws": ws, "th": [str(t) for t in th], "cap": cap, "two": two, "n": n, "scale": scale}
    return score_item(rng, "or_weighted", inp=inp, q=q, compute=compute, render=render, levels=levels, question=question, decisive=list(inp), perturb=pert, d=d)


# ======================================================================================= priority matrix
def or_matrix(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["saas_ops", "energy", "healthcare_admin", "logistics", "finance"])]
    thing = rng.choice(["incident", "service request", "defect report", "outage ticket"])
    u_cuts = [rng.randint(5, 80), rng.randint(90, 900), rng.randint(1000, 6000)]
    r_cuts = [rng.randint(3, 40) * 100, rng.randint(50, 300) * 100]
    inp = {"users": rng.randint(1, 8000), "revenue": rng.randint(0, 60000), "workaround": rng.random() < 0.5,
           "deadline_h": rng.choice([2, 4, 8, 12, 24, 48, 72, 120])}
    if d >= 4:
        inp["security"] = rng.random() < 0.3
    if d == 5:
        inp["vip"] = rng.random() < 0.4
        inp["repeat"] = rng.random() < 0.4
    nl = rng.choice([4, 5])
    # matrix impact (0..3) x urgency (0..2) -> priority level 0..nl-1 (higher = more urgent)
    mat = [[min(nl - 1, max(0, (im + ur) * (nl - 1) // 5)) for ur in range(3)] for im in range(4)]
    mat[3][2] = nl - 1
    mat[0][0] = 0
    dl = sorted(rng.sample(range(2, 73), 2))

    def impact(i):
        a = sum(1 for c in u_cuts if i["users"] >= c)
        b = sum(1 for c in r_cuts if i["revenue"] >= c) + (1 if i["revenue"] >= r_cuts[-1] * 2 else 0)
        return min(3, max(a, b))

    def urgency(i):
        u = 2 if i["deadline_h"] <= dl[0] else 1 if i["deadline_h"] <= dl[1] else 0
        if i["workaround"]:
            u = max(0, u - 1)
        return u

    def compute(i):
        p = mat[impact(i)][urgency(i)]
        if i.get("security"):
            p = min(nl - 1, p + 1)
        if i.get("vip") and i.get("repeat"):
            p = min(nl - 1, p + 1)
        return p

    pname = [f"P{nl - k}" for k in range(nl)]

    def render(i, missing, r):
        imp_lbl = ["Low", "Medium", "High", "Critical"]
        urg_lbl = ["Low", "Medium", "High"]
        rules = [T(r, "{Impact|Impact band}: {Critical if users affected is at least $u3, High if at least $u2, Medium if at least $u1, otherwise Low|"
                      "$u3 or more affected users is Critical, $u2 or more High, $u1 or more Medium, fewer is Low}. {Revenue at risk per day can raise "
                      "the band|A revenue test applies too}: Medium from $r1, High from $r2, Critical from $r3. {Take whichever band is higher.|The higher "
                      "of the two bands applies.}", u1=f"{u_cuts[0]:,}", u2=f"{u_cuts[1]:,}", u3=f"{u_cuts[2]:,}", r1=f"{r_cuts[0]:,}",
                   r2=f"{r_cuts[1]:,}", r3=f"{r_cuts[1] * 2:,}"),
                 T(r, "{Urgency|Urgency band}: {High if the business deadline is $a hours away or less, Medium if $b hours or less, otherwise Low|"
                      "a deadline within $a hours is High, within $b hours Medium, anything later Low}. {A working workaround lowers urgency by one band "
                      "(never below Low).|If a workaround is confirmed working, drop urgency one band, but not below Low.}", a=dl[0], b=dl[1])]
        tbl = md_table(["Impact \\ Urgency"] + urg_lbl, [[imp_lbl[im]] + [pname[mat[im][u]] for u in range(3)] for im in range(4)])
        if "security" in i:
            rules.append("A confirmed security element raises the priority by one step (P-number minus one), to at most " + pname[-1] + ".")
        if "vip" in i:
            rules.append("If the customer is on the strategic-accounts list AND this is a repeat of an issue from the last 30 days, raise the priority by one "
                         "more step (to at most " + pname[-1] + ").")
        ev = [f"Users affected: {'[count not yet known]' if 'users' in missing else format(i['users'], ',')}",
              f"Revenue at risk per day: {'[finance estimate pending]' if 'revenue' in missing else format(i['revenue'], ',')}",
              f"Workaround available: {'[being tested]' if 'workaround' in missing else ('yes, confirmed working' if i['workaround'] else 'no')}",
              f"Business deadline: {'[not stated]' if 'deadline_h' in missing else str(i['deadline_h']) + ' hours from now'}"]
        if "security" in i:
            ev.append(f"Security element: {'[under investigation]' if 'security' in missing else ('confirmed' if i['security'] else 'ruled out')}")
        if "vip" in i:
            ev.append(f"Strategic account: {'[unknown]' if 'vip' in missing else ('yes' if i['vip'] else 'no')}; repeat within 30 days: "
                      f"{'[unknown]' if 'repeat' in missing else ('yes' if i['repeat'] else 'no')}")
        r.shuffle(ev)
        head = [T(r, "{Triage policy|Prioritisation rules|Severity matrix} and $t record $c", t=thing, c=names.code(r.choice(["INC", "REQ", "DEF"])))]
        return assemble(r, head, rules + [tbl, "Record:\n" + "\n".join("- " + e for e in ev)], dom, Names(r, names.used), d)

    mid = ["low priority, next-business-day response", "moderate priority, same-day response", "high priority, response within the hour"]
    levels = [f"Level {k} = {pname[k]}: " + ("lowest priority, handled in the normal queue" if k == 0 else
                                            "highest priority, immediate response" if k == nl - 1 else mid[min(len(mid) - 1, k - 1 + (3 - (nl - 2)))])
              for k in range(nl)]
    question = T(rng, "{What priority does this $t get|Which priority applies to this $t} under the policy?", t=thing)

    def pert(k, i):
        v = i[k]
        if isinstance(v, bool):
            return [dict(i, **{k: not v})]
        if k == "users":
            return [dict(i, users=x) for x in (1, u_cuts[0], u_cuts[1], u_cuts[2], 9000) if x != v]
        if k == "revenue":
            return [dict(i, revenue=x) for x in (0, r_cuts[0], r_cuts[1], r_cuts[1] * 2) if x != v]
        return [dict(i, deadline_h=x) for x in (1, dl[0], dl[1], 200) if x != v]
    q = {"u_cuts": u_cuts, "r_cuts": r_cuts, "dl": dl, "mat": mat, "nl": nl}
    return score_item(rng, "or_matrix", inp=inp, q=q, compute=compute, render=render, levels=levels, question=question, decisive=list(inp), perturb=pert, d=d)


# ======================================================================================= demerit counting
OBS = [("chiller temperature", "°C", [(F(8), "critical"), (F(5), "major")], True), ("fire-exit obstruction", " exits", [(F(2), "critical"), (F(1), "major")], True),
       ("unlabelled chemical containers", " containers", [(F(5), "major"), (F(1), "minor")], True),
       ("expired first-aid items", " items", [(F(4), "major"), (F(1), "minor")], True), ("guard-rail gap", " cm", [(F(30), "critical"), (F(10), "major")], True),
       ("emergency lighting failures", " fittings", [(F(3), "major"), (F(1), "minor")], True), ("hand-wash station out of soap", " stations", [(F(2), "major"), (F(1), "minor")], True),
       ("records missing a signature", " records", [(F(10), "major"), (F(1), "minor")], True), ("trip hazards in walkways", " hazards", [(F(3), "major"), (F(1), "minor")], True)]


def or_count(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["manufacturing", "hospitality", "healthcare_admin", "logistics", "retail", "education"])]
    site = names.city() + " " + rng.choice(["plant", "kitchen", "clinic", "warehouse", "store", "campus"])
    k = {3: rng.randint(3, 4), 4: rng.randint(4, 6), 5: rng.randint(6, 8)}[d]
    obs = []
    for name, unit, rules, flag in rng.sample(OBS, k):
        (t1, s1), (t2, s2) = rules
        if unit == "°C":
            n1, n2 = t1 + F(rng.randint(-3, 3), 2), t2 + F(rng.randint(-3, 2), 2)
        else:
            n1, n2 = max(2, int(t1 * F(rng.randint(70, 140), 100))), max(1, int(t2 * F(rng.randint(80, 200), 100)))
        if n2 >= n1:
            n2 = n1 - (1 if unit != "°C" else F(1, 2))
        obs.append((name, unit, [(F(n1), s1), (F(n2), s2)], flag))
    w = {"minor": 1, "major": rng.choice([3, 4, 5]), "critical": rng.choice([8, 10, 12])}
    inp = {}
    for j, (name, unit, rules, _) in enumerate(obs):
        hi = int(rules[0][0] * 2) + 2
        inp[f"o|{j}"] = F(rng.randint(0, hi * 10), 10) if unit == "°C" else F(rng.randint(0, hi))
        if d == 5:
            inp[f"rep|{j}"] = rng.random() < 0.3
    n = rng.randint(4, 8)
    bands = sorted(rng.sample(range(1, 40), n - 1))  # level k (worse) needs demerits >= bands[k-1]; level 0 = best

    def sev(j, v):
        for t, s in obs[j][2]:
            if v >= t:
                return s
        return None

    def compute(i):
        tot = 0
        for j in range(k):
            s = sev(j, i[f"o|{j}"])
            if s:
                pts = w[s]
                if i.get(f"rep|{j}") and s != "minor":
                    pts *= 2
                tot += pts
        return sum(1 for b in bands if tot >= b)

    def render(i, missing, r):
        rl = []
        for j, (name, unit, rules, _) in enumerate(obs):
            a1, a2 = f"{float(rules[0][0]):g}{unit}", f"{float(rules[1][0]):g}{unit}"
            rl.append("- " + T(r, "{$n: $s1 if $a1 or more; $s2 if $a2 or more; otherwise no finding.|$n counts as $s1 from $a1 upwards and as $s2 from $a2; "
                                  "below $a2 it is not a finding.|$n - at least $a1 is $s1, at least $a2 is $s2, anything lower is not recorded.}",
                               n=name.capitalize(), s1=rules[0][1], s2=rules[1][1], a1=a1, a2=a2))
        rules_txt = "\n".join(rl)
        sc = T(r, "{Demerit points per finding|Each finding scores demerits}: {minor $a, major $b, critical $c|a critical one $c, a major one $b, a minor one $a}.",
               a=w["minor"], b=w["major"], c=w["critical"])
        if d == 5:
            sc += " A major or critical finding that repeats one from the previous inspection scores double; repeated minor findings score normally."
        log = []
        for j, (name, unit, rules, _) in enumerate(obs):
            v = "[reading lost]" if f"o|{j}" in missing else f"{float(i[f'o|{j}']):g}{unit}"
            rep = ""
            if d == 5:
                rep = " (repeat finding)" if i[f"rep|{j}"] else ""
                if f"rep|{j}" in missing:
                    rep = " [repeat status not checked]"
            log.append(f"- {name.capitalize()}: {v}{rep}")
        bands_txt = T(r, "The grade is set by total demerits (level 0 is the best grade).")
        head = [T(r, "{Inspection report|Site inspection|Compliance walk-round} - $s", s=site),
                "Classification rules:\n" + rules_txt, sc, bands_txt]
        return assemble(r, head, ["Inspection log:\n" + "\n".join(log)], dom, Names(r, names.used), d)

    lab = ["Excellent", "Good", "Satisfactory", "Needs improvement", "Poor", "Serious concern", "Unacceptable", "Closure recommended"]
    levels = []
    for kk in range(n):
        lo_ = 0 if kk == 0 else bands[kk - 1]
        levels.append(f"Level {kk} - {lab[kk]}: " + (f"{lo_} to {bands[kk] - 1} demerits" if kk < n - 1 else f"{lo_} demerits or more"))
    question = T(rng, "{Which grade does|What grade should} $s {receive|be given} for this inspection?", s=site)

    def pert(key, i):
        v = i[key]
        if isinstance(v, bool):
            return [dict(i, **{key: not v})]
        j = int(key.split("|")[1])
        ts = [t for t, _ in obs[j][2]]
        return [dict(i, **{key: x}) for x in (F(0), ts[1], ts[0], ts[0] * 2) if x != v]
    q = {"obs": [[o[0], [[str(t), s] for t, s in o[2]]] for o in obs], "w": w, "bands": bands, "n": n, "k": k}
    return score_item(rng, "or_count", inp=inp, q=q, compute=compute, render=render, levels=levels, question=question, decisive=list(inp), perturb=pert, d=d)


KINDS = {"or_points": or_points, "or_ladder": or_ladder, "or_weighted": or_weighted, "or_matrix": or_matrix, "or_count": or_count}
KIND_FAMILY = {k: FAM for k in KINDS}
