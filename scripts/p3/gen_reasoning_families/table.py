"""Family table_arithmetic (+ temporal_numeric for the time-based financial kinds): realistic statements and tables with
multi-step computations (2-6 ops) and distractors from typical mistakes (wrong op, wrong row/year, scale/sign/percent slips)."""
from __future__ import annotations

import math
import random
from fractions import Fraction as F

from .common import (CURRENCIES, DOMAINS, Names, Skip, T, assemble, fmoney, fnum, fpct, rq)
from .numeric_base import md_table, numeric_item

MISSING_NOTES = ["Figure withheld pending audit sign-off.", "Not yet reported by the business unit.", "To be confirmed after the year-end close.",
                 "Awaiting reconciliation; intentionally left blank.", "Source ledger unavailable at the time of writing.",
                 "Cell blank: the regional submission was late."]


def _cur(rng):
    return rng.choice(CURRENCIES[:8])


def _dom(rng):
    k = rng.choice(sorted(DOMAINS))
    return k, DOMAINS[k]


def _cell(inp, k, missing, fmt):
    return "n/a*" if k in missing else fmt(inp[k])


def _missing_note(missing, r):
    note = r.choice(MISSING_NOTES)
    return [f"\\* {note}"] if missing else []


# ======================================================================================= income-statement growth / margins
PNL_LABELS = {"rev": ["Revenue", "Net sales", "Total revenue", "Turnover", "Net revenue"],
              "cogs": ["Cost of goods sold", "Cost of sales", "Cost of revenue", "Direct costs"],
              "sm": ["Sales and marketing", "Selling expenses", "Commercial costs", "Go-to-market spend"],
              "rd": ["Research and development", "Product development", "R&D", "Engineering"],
              "ga": ["General and administrative", "Admin and overheads", "G&A", "Central costs"]}
METRIC_NAME = {"rev": "revenue", "opex": "total operating expenses", "gp": "gross profit", "opinc": "operating income",
               "adj": "adjusted operating income"}


def _pnl_metric(inp, y, m):
    r, c, s, d_, g = (inp[f"{k}|{y}"] for k in ("rev", "cogs", "sm", "rd", "ga"))
    one = inp.get(f"oneoff|{y}", 0)
    return {"rev": r, "opex": s + d_ + g, "gp": r - c, "opinc": r - c - s - d_ - g, "opinc_nog": r - c - s - d_,
            "adj": r - c - s - d_ - g + one, "gm": F(r - c) * 100 / r, "om": F(r - c - s - d_ - g) * 100 / r,
            "adjm": F(r - c - s - d_ - g + one) * 100 / r}[m]


def tab_growth(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    n = rng.randint(3, 5)
    y0 = rng.randint(2018, 2023)
    years = [f"FY{y0 + i}" for i in range(n)]
    inp = {}
    rev = rng.randint(8_000, 90_000)
    for y in years:
        rev = int(rev * rng.uniform(0.9, 1.35))
        inp[f"rev|{y}"] = rev
        inp[f"cogs|{y}"] = int(rev * rng.uniform(0.35, 0.62))
        inp[f"sm|{y}"] = int(rev * rng.uniform(0.08, 0.18))
        inp[f"rd|{y}"] = int(rev * rng.uniform(0.04, 0.14))
        inp[f"ga|{y}"] = int(rev * rng.uniform(0.04, 0.1))
    ia = rng.randrange(n - 1)
    ib = ia + 1 if d < 5 or rng.random() < 0.5 else rng.randrange(ia + 1, n)
    a, b = years[ia], years[ib]
    if d == 3:
        metric, form = rng.choice([("rev", "growth"), ("opex", "growth"), ("gp", "growth")])
    elif d == 4:
        metric, form = rng.choice([("gp", "growth"), ("opinc", "growth"), ("gm", "pp"), ("opex", "growth")])
    else:
        metric, form = rng.choice([("opinc", "growth"), ("om", "pp"), ("adj", "growth"), ("adjm", "pp")])
    if metric in ("adj", "adjm"):
        yo = rng.choice([a, b])
        inp[f"oneoff|{yo}"] = int(inp[f"ga|{yo}"] * rng.uniform(0.15, 0.45))
    q = {"metric": metric, "form": form, "a": a, "b": b, "years": years}
    for y in (a, b):
        if _pnl_metric(inp, y, "opinc") <= 0 or _pnl_metric(inp, y, "gp") <= 0:
            raise Skip("non-positive base")

    def compute(i):
        ma, mb = _pnl_metric(i, a, metric), _pnl_metric(i, b, metric)
        if form == "growth":
            if ma <= 0:
                raise Skip("base")
            return F(mb - ma) * 100 / ma
        return F(mb) - F(ma)

    def wrongs(i):
        out = []
        ma, mb = _pnl_metric(i, a, metric), _pnl_metric(i, b, metric)
        if form == "growth":
            out += [F(mb - ma) * 100 / mb, F(mb) * 100 / ma, -F(mb - ma) * 100 / ma]
            for x, y in ((ia - 1, ia), (ib, ib + 1)):
                if 0 <= x < n and 0 <= y < n and x != y:
                    m1, m2 = _pnl_metric(i, years[x], metric), _pnl_metric(i, years[y], metric)
                    if m1 > 0:
                        out.append(F(m2 - m1) * 100 / m1)
            alt = {"opinc": "opinc_nog", "adj": "opinc", "gp": "rev", "opex": "rev", "rev": "gp"}.get(metric)
            if alt:
                m1, m2 = _pnl_metric(i, a, alt), _pnl_metric(i, b, alt)
                if m1 > 0:
                    out.append(F(m2 - m1) * 100 / m1)
        else:
            out += [(F(mb) - F(ma)) * 100 / ma if ma else None, F(ma) - F(mb)]
            alt = {"om": "gm", "gm": "om", "adjm": "om"}.get(metric)
            out.append(_pnl_metric(i, b, alt) - _pnl_metric(i, a, alt))
            if metric == "om":
                out.append(F(_pnl_metric(i, b, "opinc_nog")) * 100 / i[f"rev|{b}"] - F(_pnl_metric(i, a, "opinc_nog")) * 100 / i[f"rev|{a}"])
        rng.shuffle(out)
        return out

    lab = {k: rng.choice(v) for k, v in PNL_LABELS.items()}
    unit_word = rng.choice(["in thousands", "000s", "thousands"])

    def render(i, missing, r):
        f = lambda v: fnum(v)  # noqa: E731
        rows = [[lab[k]] + [_cell(i, f"{k}|{y}", missing, f) for y in years] for k in ("rev", "cogs", "sm", "rd", "ga")]
        extra = []
        if d >= 4:
            extra.append([r.choice(["Other income (non-operating)", "Interest income", "Finance income"])] +
                         [fnum(r.randint(50, 900)) for _ in years])
            extra.append([r.choice(["Headcount (year end)", "Employees at year end", "FTEs"])] + [fnum(r.randint(40, 900)) for _ in years])
        if d == 5:
            extra.append(["Capital expenditure"] + [fnum(r.randint(200, 4000)) for _ in years])
        body_rows = rows + extra
        r.shuffle(body_rows) if d == 5 else None
        tbl = md_table([f"{cur[0]} {unit_word}"] + years, body_rows)
        head = [T(r, "{Condensed|Summary|Abridged} income statement {for|of} $co {- all amounts|- figures|, figures} $u",
                  co=co, u=f"{cur[0]} {unit_word}")]
        defs = T(r, "{Definitions used in this pack|For this pack|Reading notes}: gross profit {is|means} {revenue|the top line} "
                    "{less|minus} {cost of goods sold|the cost line}; operating income {is|means} gross profit {less|minus} {all|every} "
                    "operating expense {line|row} {shown|listed} ({the three|all three} expense rows). Non-operating items "
                    "{sit|are reported} below operating income and {are|should be} {excluded|left out} from it.")
        notes = [defs]
        if metric in ("adj", "adjm"):
            yo = next(k.split("|")[1] for k in i if k.startswith("oneoff|"))
            notes.append(T(r, "{Note|Footnote} {3|4|7}: the $yo general and administrative {line|figure} includes a one-off "
                              "{legal settlement|restructuring charge|write-down of a disputed receivable} of $amt thousand. "
                              "Adjusted operating income {adds this back|excludes this charge}; {no other adjustments are made|it is the only adjustment}.",
                           yo=yo, amt=fnum(i[f"oneoff|{yo}"])))
        if d >= 4:
            notes.append(T(r, "{Depreciation|Amortisation of software} is already {included|embedded} in the expense lines above and is "
                              "{not|never} shown separately."))
        notes += _missing_note(missing, r)
        return assemble(r, head, [tbl] + notes, dom, Names(r, names.used), d)

    mname = {"gm": "gross margin", "om": "operating margin", "adjm": "adjusted operating margin"}.get(metric, METRIC_NAME.get(metric))
    if form == "growth":
        qc = T(rng, "{By what percentage did $pc $m change|What was the percentage change in $pc $m} from $a to $b{, to one decimal place|}?"
                    "{| Round to one decimal place.}", co=co, m=mname, a=a, b=b)
        qn = lambda t: T(rng, "Was the percentage change in $pc $m from $a to $b {greater than|above} $t?", co=co, m=mname, a=a, b=b, t=t)  # noqa: E731
    else:
        qc = T(rng, "{By how many percentage points did $pc $m change|What was the change, in percentage points, in $pc $m} "
                    "between $a and $b? {Round to one decimal place.|Give one decimal place.}", co=co, m=mname, a=a, b=b)
        qn = lambda t: T(rng, "Was the change in $pc $m from $a to $b {greater than|above} $t?",  # noqa: E731
                         co=co, m=mname, a=a, b=b, t=t)
    cells = {"rev": ["rev"], "opex": ["sm", "rd", "ga"], "gp": ["rev", "cogs"], "opinc": ["cogs", "sm", "rd", "ga"],
             "adj": ["cogs", "sm", "ga", "rev"], "gm": ["rev", "cogs"], "om": ["cogs", "sm", "rd", "ga"], "adjm": ["cogs", "ga", "rd"]}[metric]
    decisive = [f"{c}|{y}" for c in cells for y in (a, b)]
    fam = "temporal_numeric"
    fmt = (lambda v: fpct(v, 1)) if form == "growth" else (lambda v: f"{rq(v, 1):.1f} pp")
    return numeric_item(rng, fam, "tab_growth", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, q_choice=qc, q_noul=qn,
                        places=1, fmt=fmt, decisive=decisive, allow_nonpos=True)


# ======================================================================================= CAGR
SERIES = [("annual recurring revenue", "ARR", "money"), ("active customers", "customers", "count"), ("parcels shipped", "parcels", "count"),
          ("enrolled students", "enrolments", "count"), ("patient visits", "visits", "count"), ("donations received", "donations", "money"),
          ("megawatt-hours sold", "MWh", "count"), ("room nights sold", "room nights", "count"), ("units produced", "units", "count")]


def tab_cagr(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    sname, short, kindv = rng.choice(SERIES)
    n = rng.randint(4, 7)
    y0 = rng.randint(2016, 2022)
    years = [str(y0 + i) for i in range(n)]
    derived = d == 5 and rng.random() < 0.6
    inp = {}
    v = rng.randint(2_000, 400_000)
    for y in years:
        v = int(v * rng.uniform(0.92, 1.45))
        if derived:
            cust = max(50, v // rng.randint(80, 300))
            inp[f"cust|{y}"] = cust
            inp[f"arpa|{y}"] = rng.randint(400, 9000)
        else:
            inp[f"v|{y}"] = v
    ia = rng.randrange(0, n - 2)
    ib = rng.randrange(ia + 2, n)
    a, b = years[ia], years[ib]
    k = ib - ia

    def val(i, y):
        return i[f"cust|{y}"] * i[f"arpa|{y}"] if derived else i[f"v|{y}"]

    def compute(i):
        return ((val(i, b) / val(i, a)) ** (1 / k) - 1) * 100

    def wrongs(i):
        va, vb = val(i, a), val(i, b)
        out = [(vb / va - 1) * 100 / k, ((vb / va) ** (1 / (k + 1)) - 1) * 100, (vb / va - 1) * 100]
        yoy = [(val(i, years[j + 1]) / val(i, years[j]) - 1) * 100 for j in range(ia, ib)]
        out.append(sum(yoy) / len(yoy))
        if ia + 1 < ib:
            out.append(((vb / val(i, years[ia + 1])) ** (1 / (k - 1)) - 1) * 100)
        if derived:
            out.append(((i[f"cust|{b}"] / i[f"cust|{a}"]) ** (1 / k) - 1) * 100)
        return out

    def render(i, missing, r):
        if derived:
            rows = [[r.choice(["Paying customers (year end)", "Customer count", "Accounts billed"])] + [_cell(i, f"cust|{y}", missing, fnum) for y in years],
                    [r.choice(["Average revenue per account", "ARPA", "Revenue per customer"]) + f" ({cur[0]})"] +
                    [_cell(i, f"arpa|{y}", missing, fnum) for y in years]]
        else:
            lab = sname.capitalize() + (f" ({cur[0]})" if kindv == "money" else "")
            rows = [[lab] + [_cell(i, f"v|{y}", missing, fnum) for y in years]]
        if d >= 4:
            rows.append([r.choice(["Staff at year end", "Sites operated", "Support tickets"])] + [fnum(r.randint(30, 3000)) for _ in years])
        tbl = md_table(["Metric"] + years, rows)
        head = [T(r, "{Multi-year|Historical|Five-year|Long-run} {performance|trend} {summary|table} {for|of} $co", co=co)]
        notes = []
        if derived:
            notes.append(T(r, "{Revenue|Annual revenue} for a year {equals|is} {paying customers|the customer count} {multiplied by|times} "
                              "{average revenue per account|ARPA} for that year."))
        notes.append(T(r, "{Compound annual growth rate|CAGR} {is defined as|means} (end value / start value)^(1 / number of years) - 1, "
                          "where the number of years {is|counts} the {annual steps|year-to-year intervals} between the two {years|columns}."))
        notes += _missing_note(missing, r)
        return assemble(r, head, [tbl] + notes, dom, Names(r, names.used), d)

    what = "revenue" if derived else sname
    qc = T(rng, "{What was|Calculate} the compound annual growth rate of $pc $w from $a to $b{, to one decimal place|}?", co=co, w=what, a=a, b=b)
    qn = lambda t: T(rng, "Was $pc compound annual growth rate {in|of} $w from $a to $b above $t?", co=co, w=what, a=a, b=b, t=t)  # noqa: E731
    decisive = ([f"cust|{a}", f"cust|{b}", f"arpa|{a}", f"arpa|{b}"] if derived else [f"v|{a}", f"v|{b}"])
    return numeric_item(rng, "temporal_numeric", "tab_cagr", inp=inp, q={"a": a, "b": b, "k": k, "derived": derived}, compute=compute,
                        wrongs=wrongs, render=render, q_choice=qc, q_noul=qn, places=1, fmt=lambda v: fpct(v, 1), decisive=decisive,
                        allow_nonpos=True)


# ======================================================================================= compounding
def tab_compound(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    who = names.person()
    _, dom = _dom(rng)
    m = rng.choice([1, 2, 4, 12])
    per = {1: "annually", 2: "semi-annually", 4: "quarterly", 12: "monthly"}[m]
    inp = {"P": rng.randint(20, 900) * 100, "r": F(rng.randint(150, 950), 100), "years": rng.randint(2, 6)}
    if d >= 4:
        inp["D"] = rng.randint(5, 300) * 100
        inp["dy"] = rng.randint(1, inp["years"] - 1)
    if d == 5:
        inp["r2"] = F(rng.randint(100, 950), 100)
        inp["ky"] = rng.randint(1, inp["years"] - 1)
        inp["fee"] = rng.randint(1, 30) * 5
    q = {"m": m}

    def bal(i, simple=False, mm=None, ignore_d=False, one_rate=False):
        mm = mm or m
        yrs = i["years"]
        b = F(i["P"])
        for step in range(yrs * mm):
            yr = step // mm
            if "D" in i and not ignore_d and step == i["dy"] * mm:
                b += i["D"]
            rate = i["r"]
            if "r2" in i and not one_rate and yr >= i["ky"]:
                rate = i["r2"]
            if simple:
                continue
            b = b * (1 + rate / 100 / mm)
        if simple:
            b = F(i["P"]) * (1 + i["r"] / 100 * yrs) + (i.get("D", 0) if not ignore_d else 0)
        if "fee" in i:
            b -= i["fee"]
        return b

    def compute(i):
        return bal(i)

    def wrongs(i):
        out = [bal(i, simple=True), bal(i, mm=1) if m != 1 else bal(i, mm=12)]
        if "D" in i:
            out.append(bal(i, ignore_d=True))
        if "r2" in i:
            out.append(bal(i, one_rate=True))
            out.append(bal(i) + 2 * i["fee"])
        b = F(i["P"]) * (1 + i["r"] / 100) ** (i["years"] * m) if m != 1 else None
        out.append(b)
        return out

    def render(i, missing, r):
        c = lambda k, fm: "[not stated]" if k in missing else fm(i[k])  # noqa: E731
        lines = [T(r, "$who {opened|started} a {fixed-term savings account|reserve deposit|sinking-fund account} with an {initial|opening} "
                      "deposit of $P {on|at the start of} {year 1|the first day of year 1}.", who=who, P=c("P", lambda v: fmoney(v, cur, 0))),
                 T(r, "The account {pays|earns} a nominal annual rate of $r, compounded $per{ (the annual rate is divided by the number of "
                      "compounding periods per year)|; each period applies the annual rate divided by the periods per year}.",
                   r=c("r", lambda v: f"{float(v):.2f}%"), per=per),
                 T(r, "{The funds stay invested|No withdrawals are made} for $y {full |}years{ in total|}.", y=c("years", str))]
        if "D" in i:
            lines.append(T(r, "{At the start of year $dy1|At the beginning of year $dy1}, a further $D {was added|is deposited}; it compounds "
                              "{from that point on|from then onwards}.", dy1=i["dy"] + 1, D=c("D", lambda v: fmoney(v, cur, 0))))
        if "r2" in i:
            lines.append(T(r, "{From the start of year $ky1|Beginning in year $ky1} the nominal rate {changes|is reset} to $r2 (same compounding "
                              "frequency) for the remaining years.", ky1=i["ky"] + 1, r2=c("r2", lambda v: f"{float(v):.2f}%")))
            lines.append(T(r, "{A flat closing fee of|An account-closure charge of} $fee is deducted {once |}{at the very end|when the balance is paid out}.",
                           fee=c("fee", lambda v: fmoney(v, cur, 0))))
        lines.append(T(r, "{Interest is never withdrawn|All interest is reinvested}; {no tax is deducted|tax is handled separately and ignored here}."))
        head = [T(r, "{Savings|Treasury|Deposit} {note|memo|summary}: $who", who=who)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    qc = T(rng, "{What balance is paid out to $w|How much does $w receive|What final amount does $w receive} at the end?", w=who)
    qn = lambda t: T(rng, "Is the {final|closing} amount paid to $w {more than|above} $t?", w=who, t=t)  # noqa: E731
    dec_keys = ["r", "P", "years"] + (["D"] if "D" in inp else []) + (["r2", "fee"] if "r2" in inp else [])
    return numeric_item(rng, "temporal_numeric", "tab_compound", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                        q_noul=qn, places=2, fmt=lambda v: fmoney(v, cur), decisive=dec_keys)


# ======================================================================================= FX invoices
def _rp(c, rep):
    return 6 if FX[c] / FX[rep] < F(1, 5) else 4


FX = {"USD": F(1), "EUR": F(108, 100), "GBP": F(127, 100), "CHF": F(113, 100), "CAD": F(73, 100), "AUD": F(66, 100), "SGD": F(74, 100),
      "SEK": F(95, 1000), "JPY": F(67, 10000)}


def tab_fx(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    rep = rng.choice(["USD", "EUR", "GBP", "CHF"])
    others = [c for c in FX if c != rep]
    ccys = rng.sample(others, rng.randint(2, 3))
    dates = ["3 Mar", "17 Mar", "31 Mar"] if rng.random() < 0.5 else ["9 Jun", "23 Jun", "7 Jul"]
    pay = dates[2]
    nlines = {3: 3, 4: 4, 5: rng.randint(5, 6)}[d]
    inp = {}
    for c in ccys:
        for dd in dates:
            base = FX[c] / FX[rep]
            rate = base * F(rng.randint(9600, 10400), 10000)
            rp = _rp(c, rep)
            inp[f"rate|{c}|{dd}"] = F(round(float(rate) * 10 ** rp), 10 ** rp)
    lines = []
    for j in range(nlines):
        c = rng.choice(ccys)
        dd = rng.choice(dates[:2])
        amt = F(rng.randint(40_000, 2_000_000), 100)
        if c == "JPY":
            amt = F(rng.randint(100_000, 9_000_000))
        inp[f"amt|{j}"] = amt
        lines.append((j, c, dd))
    credit = d == 5
    if credit:
        inp[f"amt|{nlines - 1}"] = -inp[f"amt|{nlines - 1}"] / 4
    inv_quote = ccys[0] if d == 5 else None
    fee = F(rng.choice([10, 15, 20, 25, 35, 50]), 100) if d >= 4 else None
    if fee is not None:
        inp["fee"] = fee
    q = {"lines": lines, "rep": rep, "pay": pay}

    def conv(i, j, c, dd, use_pay=False, invert=False):
        rate = i[f"rate|{c}|{pay if use_pay else dd}"]
        v = i[f"amt|{j}"] * (1 / rate if invert else rate)
        return F(rq(v, 2))

    def compute(i, **kw):
        s = sum(conv(i, j, c, dd, **kw) for j, c, dd in lines)
        if "fee" in i:
            s = s - F(rq(s * i["fee"] / 100, 2))
        return s

    def wrongs(i):
        out = [compute(i, use_pay=True), sum(conv(i, j, c, dd, invert=True) for j, c, dd in lines)]
        if "fee" in i:
            out.append(sum(conv(i, j, c, dd) for j, c, dd in lines))
            s = sum(conv(i, j, c, dd) for j, c, dd in lines)
            out.append(s + F(rq(s * i["fee"] / 100, 2)))
        if credit:
            j, c, dd = lines[-1]
            out.append(compute(i) - 2 * conv(i, j, c, dd))
        return out

    def render(i, missing, r):
        def rate_txt(c, dd):
            k = f"rate|{c}|{dd}"
            if k in missing:
                return "n/a*"
            v = i[k]
            if c == inv_quote:
                return f"1 {rep} = {float(rq(1 / v, 4)):.4f} {c}"
            return f"1 {c} = {float(v):.{_rp(c, rep)}f} {rep}"
        if inv_quote:
            # inverse quotes must be exact: store the displayed inverse as the true rate
            pass
        rows = [[f"{c}"] + [rate_txt(c, dd) for dd in dates] for c in ccys]
        fx_tbl = md_table(["Currency"] + [f"Rate on {dd}" for dd in dates], rows)
        irows = []
        for j, c, dd in lines:
            amt = i[f"amt|{j}"]
            desc = r.choice(["Consulting services", "Freight charges", "Licence renewal", "Spare parts", "Marketing retainer", "Warehouse rent",
                             "Translation work", "Sensor modules", "Catering", "Cloud hosting", "Legal review", "Packaging materials"])
            if credit and j == len(lines) - 1:
                desc = "Credit note (returned goods)"
            irows.append([names.code("INV"), desc, dd, c, _cell(i, f"amt|{j}", missing, lambda v: fnum(v, 0 if c == "JPY" else 2))])
        inv_tbl = md_table(["Document", "Description", "Invoice date", "Currency", "Amount"], irows)
        rules = [T(r, "{Conversion policy|Group FX rule}: each document is converted into $rep at the rate {published|in force} on its "
                      "{invoice date|document date} (not the payment date), and each converted amount is rounded to the cent before the lines "
                      "are added.", rep=rep)]
        if credit:
            rules.append(T(r, "Credit notes are shown as negative amounts and {reduce|are netted against} the total."))
        if inv_quote:
            rules.append(T(r, "{Note|Be aware} that the $c rate is quoted {the other way round|as units of $c per $rep}; a quote of "
                              "\"1 $rep = x $c\" means one $c is worth 1/x $rep. Use the {quoted|published} figure as given.", c=inv_quote, rep=rep))
        if "fee" in i:
            rules.append(T(r, "The bank {deducts|charges} a {transfer|conversion} fee of $f of the converted total (rounded to the cent) "
                              "{before settlement|from the amount credited}.", f="n/a*" if "fee" in missing else f"{float(i['fee']):.2f}%"))
        rules.append(T(r, "Payment {is scheduled for|will be made on} $pay; the $pay rates are {shown for treasury planning only|listed for reference}.",
                       pay=pay))
        rules += _missing_note(missing, r)
        head = [T(r, "{Supplier batch|Payables batch|Settlement run} $b {for|at} $co {(reporting currency $rep)|- reporting currency $rep}", b=bref, co=co, rep=rep)]
        return assemble(r, head, [inv_tbl, fx_tbl] + rules, dom, Names(r, names.used), d)

    if inv_quote:
        # make the inverse-quoted currency exact: the published figure is 1 rep = X c with 4 dp, so the true rate is 1/X.
        for dd in dates:
            k = f"rate|{inv_quote}|{dd}"
            x = F(rq(1 / inp[k], 4))
            inp[k] = 1 / x
    cur = next(c for c in CURRENCIES if c[0] == rep)
    bref = names.code("BATCH")
    qc = T(rng, "{What net amount in $rep does $co settle for $b|What is the net $rep amount $co settles for $b}?", rep=rep, co=co, b=bref)
    qn = lambda t: T(rng, "Is the net $rep amount $co settles for $b {greater than|above} $t?", rep=rep, co=co, b=bref, t=t)  # noqa: E731
    decisive = [f"rate|{c}|{dd}" for _, c, dd in lines] + [f"amt|{j}" for j, _, _ in lines]
    return numeric_item(rng, "table_arithmetic", "tab_fx", inp=inp, q=q, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                        q_noul=qn, places=2, fmt=lambda v: fmoney(v, cur), decisive=decisive,
                        witness_factors=[1.2, 0.8, 1.5, 0.6, 2, 0.5])


# ======================================================================================= weighted averages
WAVG_CTX = [("supplier deliveries", "Supplier", "units", "unit price", "accepted", "rejected at goods-in"),
            ("store purchase orders", "Store", "units", "unit cost", "received", "cancelled"),
            ("coffee lots", "Lot", "kg", "price per kg", "passed grading", "failed grading"),
            ("training cohorts", "Cohort", "participants", "cost per participant", "completed", "withdrawn"),
            ("fuel purchases", "Depot", "litres", "price per litre", "posted", "disputed"),
            ("licence blocks", "Reseller", "seats", "price per seat", "activated", "voided")]


def tab_weighted(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    what, col, qty, price, ok, bad = rng.choice(WAVG_CTX)
    n = {3: rng.randint(4, 5), 4: rng.randint(5, 7), 5: rng.randint(6, 9)}[d]
    inp, status, cases = {}, {}, {}
    rows_ids = [names.stem() for _ in range(n)]
    for j, rid in enumerate(rows_ids):
        inp[f"q|{j}"] = rng.randint(20, 900)
        inp[f"p|{j}"] = F(rng.randint(150, 9000), 100)
        status[j] = ok if (d == 3 or rng.random() < 0.72) else bad
        cases[j] = d == 5 and rng.random() < 0.35
    if sum(1 for j in status if status[j] == ok) < 2 or all(status[j] == ok for j in status) and d >= 4:
        raise Skip("filter degenerate")
    case = rng.choice([12, 24, 6, 10]) if d == 5 else 1

    def units(i, j):
        return i[f"q|{j}"] * (case if cases[j] else 1)

    def compute(i, filt=True, conv=True):
        js = [j for j in range(n) if not filt or status[j] == ok]
        w = [units(i, j) if conv else i[f"q|{j}"] for j in js]
        return sum(F(wj) * i[f"p|{j}"] for wj, j in zip(w, js)) / sum(w)

    def wrongs(i):
        js = [j for j in range(n) if status[j] == ok]
        return [compute(i, filt=False), compute(i, conv=False), sum(i[f"p|{j}"] for j in js) / len(js),
                sum(i[f"p|{j}"] for j in range(n)) / n]

    def render(i, missing, r):
        rows = []
        for j, rid in enumerate(rows_ids):
            qv = _cell(i, f"q|{j}", missing, fnum)
            if cases[j]:
                qv += f" cases of {case}"
            rows.append([rid, qv, _cell(i, f"p|{j}", missing, lambda v: fmoney(v, cur)), status[j]])
        tbl = md_table([col, qty.capitalize(), price.capitalize() + " (per single " + qty.rstrip("s") + ")" if d == 5 else price.capitalize(), "Status"], rows)
        notes = [T(r, "{Only rows marked|Rows count only when marked} \"$ok\" {enter|count towards} the weighted average; rows marked \"$bad\" "
                      "{are excluded|do not count}.", ok=ok, bad=bad)] if d >= 4 else []
        if d == 5:
            notes.append(T(r, "Quantities marked \"cases of $c\" are {case counts|counted in cases}: each case holds $c {single |}$u, and the "
                              "price column is always per single {unit|item}.", c=case, u=qty))
        notes.append(T(r, "The weighted average $p {weights|is weighted by} {quantity|the number of $u}{ in single $u|}.", p=price, u=qty))
        notes += _missing_note(missing, r)
        head = [T(r, "{Summary of|Log of|Register of} $w {at|for} $co", w=what, co=co)]
        return assemble(r, head, [tbl] + notes, dom, Names(r, names.used), d)

    qc = T(rng, "{What is|Calculate} $pc weighted average $p {across qualifying rows|under these rules}?", co=co, p=price)
    qn = lambda t: T(rng, "Is $pc weighted average $p {higher than|above} $t?", co=co, p=price, t=t)  # noqa: E731
    decisive = [f"{x}|{j}" for j in range(n) if status[j] == ok for x in ("q", "p")]
    return numeric_item(rng, "table_arithmetic", "tab_weighted", inp=inp, q={"status": status, "ok": ok, "cases": cases, "case": case, "n": n},
                        compute=compute, wrongs=wrongs, render=render, q_choice=qc, q_noul=qn, places=2, fmt=lambda v: fmoney(v, cur),
                        decisive=decisive)


# ======================================================================================= balance-sheet ratios
def tab_ratio(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    y0 = rng.randint(2019, 2025)
    ya, yb = str(y0), str(y0 + 1)
    inp = {}
    for y in (ya, yb):
        for k, lo, hi in (("cash", 200, 9000), ("recv", 300, 9000), ("inv", 300, 12000), ("prepaid", 20, 900), ("ppe", 2000, 40000),
                          ("pay", 300, 8000), ("std", 0, 3000), ("accr", 50, 2000), ("ltd", 500, 20000), ("eq", 3000, 40000)):
            inp[f"{k}|{y}"] = rng.randint(lo, hi)
    inp["cogs"] = rng.randint(4000, 60000)
    choices = {3: ["current", "de"], 4: ["quick", "turnover", "current"], 5: ["dio", "quick", "turnover", "de"]}[d]
    metric = rng.choice(choices)

    def ca(i, y, with_inv=True, with_ppe=False):
        return i[f"cash|{y}"] + i[f"recv|{y}"] + (i[f"inv|{y}"] if with_inv else 0) + i[f"prepaid|{y}"] + (i[f"ppe|{y}"] if with_ppe else 0)

    def cl(i, y):
        return i[f"pay|{y}"] + i[f"std|{y}"] + i[f"accr|{y}"]

    def compute(i, y=yb, variant=None):
        if metric == "current":
            return F(ca(i, y, with_ppe=variant == "ppe")) / cl(i, y)
        if metric == "quick":
            return F(i[f"cash|{y}"] + i[f"recv|{y}"] + (i[f"inv|{y}"] if variant == "inv" else 0)) / cl(i, y)
        if metric == "de":
            return F(i[f"std|{y}"] + i[f"ltd|{y}"] + (i[f"pay|{y}"] if variant == "pay" else 0)) / i[f"eq|{y}"]
        avg = F(i[f"inv|{ya}"] + i[f"inv|{yb}"], 2) if variant != "closing" else F(i[f"inv|{yb}"])
        t = F(i["cogs"]) / avg
        if metric == "turnover":
            return t
        return 365 / t

    def wrongs(i):
        out = [compute(i, y=ya)]
        out += [compute(i, variant=v) for v in ("ppe", "inv", "pay", "closing")]
        if metric in ("turnover", "dio"):
            out.append(F(i["cogs"]) / i[f"inv|{ya}"] if metric == "turnover" else 365 * F(i[f"inv|{yb}"]) / i["cogs"] * 2)
        if metric == "dio":
            out.append(F(i["cogs"]) / (F(i[f"inv|{ya}"] + i[f"inv|{yb}"], 2)))
        return out

    labels = {"cash": "Cash and equivalents", "recv": "Trade receivables", "inv": "Inventories", "prepaid": "Prepayments",
              "ppe": "Property, plant and equipment", "pay": "Trade payables", "std": "Short-term borrowings", "accr": "Accrued liabilities",
              "ltd": "Long-term borrowings", "eq": "Total equity"}

    def render(i, missing, r):
        rows = [[labels[k]] + [_cell(i, f"{k}|{y}", missing, fnum) for y in (ya, yb)] for k in labels]
        tbl = md_table([f"{cur[0]} thousands, at year end", ya, yb], rows)
        sec = T(r, "{Classification|Balance-sheet grouping}: cash, receivables, inventories and prepayments are current assets; property, plant "
                   "and equipment is non-current. Payables, short-term borrowings and accrued liabilities are current liabilities; long-term "
                   "borrowings are non-current.")
        defs = {"current": "The current ratio is current assets divided by current liabilities.",
                "quick": "The quick ratio is (cash and equivalents + trade receivables) divided by current liabilities; inventories and prepayments are left out.",
                "de": "Debt-to-equity is total borrowings (short-term plus long-term) divided by total equity; trade payables are not borrowings.",
                "turnover": "Inventory turnover is cost of goods sold for the year divided by average inventory (the mean of opening and closing inventories).",
                "dio": "Days inventory outstanding is 365 divided by inventory turnover, where turnover is cost of goods sold divided by the average of opening and closing inventories."}[metric]
        pl = T(r, "Cost of goods sold for $yb {was|came to|totalled} $c thousand.", yb=yb, c=_cell(i, "cogs", missing, fnum))
        head = [T(r, "{Balance-sheet extract|Selected balance-sheet lines|Statement of financial position (extract)} - $co", co=co)]
        return assemble(r, head, [tbl, sec, pl, defs] + _missing_note(missing, r), dom, Names(r, names.used), d)

    mname = {"current": "current ratio", "quick": "quick ratio", "de": "debt-to-equity ratio", "turnover": "inventory turnover",
             "dio": "days inventory outstanding"}[metric]
    places = 1 if metric == "dio" else 2
    fmt = (lambda v: f"{rq(v, 1):.1f} days") if metric == "dio" else (lambda v: f"{rq(v, 2):.2f}x")
    when = f"for {yb}" if metric in ("turnover", "dio") else f"at the end of {yb}"
    qc = T(rng, "{What is|Compute} $pc $m $w{, using the definitions given|}?", co=co, m=mname, w=when)
    qn = lambda t: T(rng, "Using the stated definitions, is $pc $m $w above $t?", co=co, m=mname, w=when, t=t)  # noqa: E731
    dk = {"current": ["cash", "recv", "inv", "prepaid", "pay", "std", "accr"], "quick": ["cash", "recv", "pay", "std", "accr"],
          "de": ["std", "ltd", "eq"], "turnover": ["inv"], "dio": ["inv"]}[metric]
    decisive = [f"{k}|{yb}" for k in dk] + ([f"inv|{ya}", "cogs"] if metric in ("turnover", "dio") else [])
    return numeric_item(rng, "table_arithmetic", "tab_ratio", inp=inp, q={"metric": metric, "ya": ya, "yb": yb}, compute=compute,
                        wrongs=wrongs, render=render, q_choice=qc, q_noul=qn, places=places, fmt=fmt, decisive=decisive)


# ======================================================================================= budget variance (choice over departments)
DEPTS = ["Marketing", "Engineering", "Facilities", "Customer Support", "Finance", "Legal", "Operations", "Sales", "HR", "IT", "Procurement",
         "Logistics", "Research", "Quality", "Training", "Security"]


def tab_variance(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    n = {3: 5, 4: rng.randint(5, 7), 5: rng.randint(6, 8)}[d]
    depts = rng.sample(DEPTS, n)
    inp = {}
    for j in range(n):
        b = rng.randint(80, 2000) * 1000
        inp[f"b|{j}"] = b
        inp[f"a|{j}"] = int(b * rng.uniform(0.85, 1.22) / 100) * 100
    reclass = None
    if d >= 4:
        src, dst = rng.sample(range(n), 2)
        inp["rc"] = int(inp[f"a|{src}"] * rng.uniform(0.04, 0.12) / 100) * 100
        reclass = (src, dst)
    revise = None
    if d == 5:
        rv = rng.randrange(n)
        inp[f"rb|{rv}"] = int(inp[f"b|{rv}"] * rng.uniform(1.03, 1.15) / 1000) * 1000
        revise = rv

    def adj(i, use_rc=True, use_rev=True):
        a = [F(i[f"a|{j}"]) for j in range(n)]
        b = [F(i[f"rb|{j}"] if (use_rev and f"rb|{j}" in i) else i[f"b|{j}"]) for j in range(n)]
        if reclass and use_rc:
            s, t = reclass
            a[s] -= i["rc"]
            a[t] += i["rc"]
        return a, b

    def pcts(i, **kw):
        a, b = adj(i, **kw)
        return [(a[j] - b[j]) * 100 / b[j] for j in range(n)]

    def argmax_unique(v, margin=F(3, 10)):
        s = sorted(range(n), key=lambda j: v[j], reverse=True)
        if v[s[0]] - v[s[1]] < margin or v[s[0]] <= 0:
            raise Skip("no clear max")
        return s[0]

    typ = rng.choice(["choice", "choice", "noul"])
    rseed = rng.random()

    def render(i, missing, r):
        rows = [[depts[j], _cell(i, f"b|{j}", missing, fnum), _cell(i, f"a|{j}", missing, fnum)] for j in range(n)]
        tbl = md_table(["Department", f"Original budget ({cur[0]})", f"Actual spend ({cur[0]})"], rows)
        notes = []
        if reclass:
            s, t = reclass
            notes.append(T(r, "{Reclassification|Correction} {after the ledger closed|found at review}: $amt {booked to|recorded under} $s "
                              "{belongs to|should have been charged to} $t. The actuals above {do not yet reflect|still include} this {move|correction}.",
                           amt=_cell(i, "rc", missing, lambda v: fmoney(v, cur, 0)), s=depts[s], t=depts[t]))
        if revise is not None:
            notes.append(T(r, "{Mid-year|In the second quarter}, the board {approved|signed off} a revised budget of $rb for $dname; the revised figure "
                              "{replaces|supersedes} the original for variance purposes.", rb=_cell(i, f"rb|{revise}", missing, lambda v: fmoney(v, cur, 0)),
                           dname=depts[revise]))
        notes.append(T(r, "Overspend {is measured|is expressed} as (actual - budget) {divided by|as a share of} budget{, in percent|}."))
        notes += _missing_note(missing, r)
        head = [T(r, "{Departmental|Cost-centre} {budget review|spend versus budget} {for|at} $co{, full year|, year to date|}", co=co)]
        return assemble(r, head, [tbl] + notes, dom, Names(r, names.used), d)

    spec = {"kind": "tab_variance", "inputs": {k: str(v) for k, v in inp.items()}, "q": {"n": n, "reclass": reclass, "revise": revise, "type": typ}}
    if typ == "choice":
        g = argmax_unique(pcts(inp))
        from .common import choice_field
        qtext = T(rng, "{Which $co department|Which cost centre at $co} had the largest percentage overspend?", co=co)
        field, gold, ov = choice_field(rng, qtext, depts[g], [depts[j] for j in range(n) if j != g], values={x: str(k) for k, x in enumerate(depts)})
        spec["option_values"] = ov

        def ans(i):
            return argmax_unique(pcts(i), margin=F(0))
        decisive = [f"a|{j}" for j in range(n)] + [f"b|{j}" for j in range(n)]
    else:
        j0 = rng.randrange(n)
        if reclass and rng.random() < 0.6:
            j0 = rng.choice(reclass)
        v = pcts(inp)[j0]
        if abs(v) < F(1, 5):
            raise Skip("too close to zero")
        gold = bool(v <= 0)
        from .common import noul_field
        field = noul_field(T(rng, "Did $pc $dname {team|department} finish {within|at or under} its {applicable |}budget?", co=co, dname=depts[j0]))
        spec["q"]["dept"] = j0

        def ans(i):
            return pcts(i)[j0] <= 0
        decisive = [f"a|{j0}", f"b|{j0}"] + (["rc"] if reclass and j0 in reclass else [])
    state = render(inp, set(), random.Random(rseed))
    unk = None
    from .common import flip_witness
    rng.shuffle(decisive)
    for k in decisive:
        if k.startswith("b|") and f"r{k}" in inp:
            continue
        try:
            wit = flip_witness(rng, inp, k, ans, [1.1, 0.9, 1.2, 0.8, 1.35, 0.7])
        except Skip:
            continue
        unk = {"state": render(inp, {k}, random.Random(rseed)), "reason": "insufficient_evidence", "spec": {**spec, "removed": k, "witness": wit, "witness_alt": {k: wit[1]}}}
        break
    return {"family": "table_arithmetic", "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


# ======================================================================================= inventory roll-forward and valuation
def tab_inventory(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    n = {3: rng.randint(2, 3), 4: rng.randint(3, 4), 5: rng.randint(3, 5)}[d]
    skus = [names.code(rng.choice(["SKU", "PN", "ITM"])) for _ in range(n)]
    inp = {}
    for j in range(n):
        ou = rng.randint(50, 900)
        ru = rng.randint(40, 1200)
        inp.update({f"ou|{j}": ou, f"oc|{j}": F(rng.randint(200, 9000), 100), f"ru|{j}": ru, f"rc|{j}": F(rng.randint(200, 9000), 100),
                    f"sh|{j}": rng.randint(20, (ou + ru) * 3 // 4), f"wo|{j}": rng.randint(0, 30), f"rt|{j}": rng.randint(0, 40)})
        if d == 5:
            inp[f"tr|{j}"] = rng.choice([0, 0, rng.randint(10, ru // 2 + 10)])
        if ou + ru - inp.get(f"tr|{j}", 0) - inp[f"sh|{j}"] - inp[f"wo|{j}"] < 5:
            raise Skip("negative stock")
    target = rng.randrange(n)
    mode = {3: "units", 4: "value", 5: "total"}[d]

    def end_units(i, j, wo=True, rt_sign=1, transit=True):
        recv = i[f"ru|{j}"] - (i.get(f"tr|{j}", 0) if transit else 0)
        return i[f"ou|{j}"] + recv + rt_sign * i[f"rt|{j}"] - i[f"sh|{j}"] - (i[f"wo|{j}"] if wo else 0)

    def avg_cost(i, j, transit=True, latest=False):
        if latest:
            return i[f"rc|{j}"]
        ru = i[f"ru|{j}"] - (i.get(f"tr|{j}", 0) if transit else 0)
        return (i[f"ou|{j}"] * i[f"oc|{j}"] + ru * i[f"rc|{j}"]) / (i[f"ou|{j}"] + ru)

    def value(i, j, **kw):
        latest = kw.pop("latest", False)
        transit = kw.get("transit", True)
        return end_units(i, j, **kw) * avg_cost(i, j, transit=transit, latest=latest)

    def compute(i):
        if mode == "units":
            return F(end_units(i, target))
        if mode == "value":
            return value(i, target)
        return sum(value(i, j) for j in range(n))

    def wrongs(i):
        if mode == "units":
            return [end_units(i, target, wo=False), end_units(i, target, rt_sign=-1), end_units(i, target) + 2 * i[f"wo|{target}"],
                    i[f"ou|{target}"] + i[f"ru|{target}"] - i[f"sh|{target}"]]
        if mode == "value":
            return [value(i, target, wo=False), value(i, target, latest=True), value(i, target, rt_sign=-1),
                    end_units(i, target) * i[f"oc|{target}"]]
        return [sum(value(i, j, transit=False) for j in range(n)), sum(value(i, j, latest=True) for j in range(n)),
                sum(value(i, j, wo=False) for j in range(n)), value(i, target)]

    def render(i, missing, r):
        cols = ["Item", "Opening units", "Opening unit cost", "Units received", "Receipt unit cost", "Units shipped", "Written off", "Customer returns"]
        if d == 5:
            cols.append("Of receipts: in transit")
        rows = []
        for j in range(n):
            row = [skus[j], _cell(i, f"ou|{j}", missing, fnum), _cell(i, f"oc|{j}", missing, lambda v: fmoney(v, cur)),
                   _cell(i, f"ru|{j}", missing, fnum), _cell(i, f"rc|{j}", missing, lambda v: fmoney(v, cur)),
                   _cell(i, f"sh|{j}", missing, fnum), _cell(i, f"wo|{j}", missing, fnum), _cell(i, f"rt|{j}", missing, fnum)]
            if d == 5:
                row.append(_cell(i, f"tr|{j}", missing, fnum))
            rows.append(row)
        tbl = md_table(cols, rows)
        rules = [T(r, "{Stock rule|Roll-forward rule}: closing units = opening units + units received + customer returns (restocked as new) "
                      "- units shipped - units written off.")]
        if mode != "units":
            rules.append(T(r, "{Valuation|Costing}: closing units are valued at the weighted average cost of the opening stock and the units "
                              "{received|physically received} in the period (returns and write-offs do not change the average)."))
        if d == 5:
            rules.append(T(r, "Units {marked|listed} as in transit are included in \"units received\" on the supplier invoice but have "
                              "{not arrived|not been delivered}; {they are not stock yet|exclude them} from both the roll-forward and the average cost."))
        rules += _missing_note(missing, r)
        head = [T(r, "{Period-end stock report|Inventory roll-forward|Stock movement summary} - $co", co=co)]
        return assemble(r, head, [tbl] + rules, dom, Names(r, names.used), d)

    if mode == "units":
        qc = T(rng, "How many units of $s {are on hand|remain in stock} at period end?", s=skus[target])
        qn = lambda t: T(rng, "Does $s close the period with more than $t units on hand?", s=skus[target], t=t)  # noqa: E731
        fmt, places = (lambda v: fnum(v)), 0
    elif mode == "value":
        qc = T(rng, "What is the closing {inventory value|stock value} of $s{ under the costing rule|}?", s=skus[target])
        qn = lambda t: T(rng, "Is the closing stock value of $s above $t?", s=skus[target], t=t)  # noqa: E731
        fmt, places = (lambda v: fmoney(v, cur)), 2
    else:
        qc = T(rng, "What is the total closing inventory value across all items{ listed|}?")
        qn = lambda t: T(rng, "Is the total closing inventory value across all items above $t?", t=t)  # noqa: E731
        fmt, places = (lambda v: fmoney(v, cur)), 2
    js = [target] if mode != "total" else list(range(n))
    decisive = [f"{k}|{j}" for j in js for k in ("sh", "wo", "ou", "ru", "rt") + (("oc", "rc") if mode != "units" else ())]
    return numeric_item(rng, "table_arithmetic", "tab_inventory", inp=inp, q={"n": n, "target": target, "mode": mode}, compute=compute,
                        wrongs=wrongs, render=render, q_choice=qc, q_noul=qn, places=places, fmt=fmt, decisive=decisive,
                        min_gap=2 if places else 1)


# ======================================================================================= SaaS MRR bridge
def tab_saas(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company("Software")
    dom = DOMAINS["saas_ops"]
    inp = {"start": rng.randint(200, 5000) * 100}
    s = inp["start"]
    inp.update({"new": int(s * rng.uniform(0.02, 0.12)), "exp": int(s * rng.uniform(0.02, 0.1)), "con": int(s * rng.uniform(0.005, 0.05)),
                "churn": int(s * rng.uniform(0.01, 0.07))})
    metric = {3: "end", 4: rng.choice(["nrr", "grr"]), 5: rng.choice(["nrr", "nrr_arr"])}[d]
    if d == 5:
        inp["mis"] = int(inp["exp"] * rng.uniform(0.15, 0.5))

    def parts(i, fix=True):
        exp, new = i["exp"], i["new"]
        if "mis" in i and fix:
            exp, new = exp - i["mis"], new + i["mis"]
        return exp, new

    def compute(i):
        exp, new = parts(i)
        if metric == "end":
            return F(i["start"] + new + exp - i["con"] - i["churn"])
        if metric == "grr":
            return F(i["start"] - i["con"] - i["churn"]) * 100 / i["start"]
        if metric == "nrr":
            return F(i["start"] + exp - i["con"] - i["churn"]) * 100 / i["start"]
        return F(i["start"] + new + exp - i["con"] - i["churn"]) * 12

    def wrongs(i):
        exp, new = parts(i)
        st, c, ch = i["start"], i["con"], i["churn"]
        endv = st + new + exp - c - ch
        if metric == "end":
            return [st + new + exp - c, st + exp - c - ch, st + new + exp + c + ch, (st + new + exp - c - ch) * 12]
        if metric == "grr":
            return [F(st + exp - c - ch) * 100 / st, F(st - ch) * 100 / st, F(st - c - ch) * 100 / endv]
        if metric == "nrr":
            e2, _ = parts(i, fix=False)
            return [F(st + exp + new - c - ch) * 100 / st, F(st - c - ch) * 100 / st, F(st + e2 - c - ch) * 100 / st, F(st + exp - c - ch) * 100 / endv]
        return [F(endv), F(st + exp - c - ch) * 12, F(st + new + exp - c) * 12]

    def render(i, missing, r):
        month = r.choice(["March", "April", "July", "October", "November"])
        rows = [[lab, _cell(i, k, missing, lambda v: fmoney(v, cur, 0))] for k, lab in
                (("start", f"MRR at 1 {month}"), ("new", "New-logo MRR"), ("exp", "Expansion MRR (upgrades, extra seats)"),
                 ("con", "Contraction MRR (downgrades)"), ("churn", "Churned MRR (cancellations)"))]
        tbl = md_table(["MRR bridge", month], rows)
        defs = [T(r, "{Definitions|Metric definitions}: net revenue retention (NRR) = (starting MRR + expansion - contraction - churn) / starting MRR; "
                     "new-logo MRR is excluded. Gross revenue retention (GRR) = (starting MRR - contraction - churn) / starting MRR. "
                     "Ending MRR = starting MRR + new + expansion - contraction - churn. ARR = 12 x ending MRR.")]
        if "mis" in i:
            defs.append(T(r, "{Correction from RevOps|Data note}: $m of the expansion line {came from|was booked for} a brand-new customer "
                             "{signed this month|who first subscribed this month} and {belongs|should sit} under new-logo MRR instead.",
                          m=_cell(i, "mis", missing, lambda v: fmoney(v, cur, 0))))
        defs += _missing_note(missing, r)
        head = [T(r, "{Monthly revenue bridge|MRR movement report|Revenue retention snapshot} - $co", co=co)]
        return assemble(r, head, [tbl] + defs, dom, Names(r, names.used), d)

    txt = {"end": "ending MRR", "grr": "gross revenue retention", "nrr": "net revenue retention", "nrr_arr": "ARR at month end"}[metric]
    pct_m = metric in ("grr", "nrr")
    fmt = (lambda v: fpct(v, 1)) if pct_m else (lambda v: fmoney(v, cur, 0))
    places = 1 if pct_m else 0
    qc = T(rng, "{What is|Compute} $pc $t for the month{, after any corrections|}?", co=co, t=txt)
    qn = lambda t: T(rng, "Is $pc $m for the month above $t?", co=co, m=txt, t=t)  # noqa: E731
    decisive = ["con", "churn", "exp"] + (["new"] if metric in ("end", "nrr_arr") else []) + (["mis"] if "mis" in inp else [])
    return numeric_item(rng, "table_arithmetic", "tab_saas", inp=inp, q={"metric": metric}, compute=compute, wrongs=wrongs, render=render,
                        q_choice=qc, q_noul=qn, places=places, fmt=fmt, decisive=decisive, min_gap=2)


# ======================================================================================= break-even
def tab_breakeven(rng: random.Random, d: int) -> dict:
    cur = _cur(rng)
    names = Names(rng)
    co = names.company()
    _, dom = _dom(rng)
    product = names.product()
    inp = {"price": F(rng.randint(1500, 30000), 100)}
    p = inp["price"]
    inp["mat"] = F(int(p * 100 * F(rng.randint(15, 30), 100)), 100)
    inp["lab"] = F(int(p * 100 * F(rng.randint(8, 18), 100)), 100)
    if d >= 4:
        inp["pack"] = F(rng.randint(20, 200), 100)
        inp["comm"] = F(rng.choice([3, 4, 5, 6, 7, 8, 10]))
    inp["rent"] = rng.randint(20, 150) * 100
    inp["sal"] = rng.randint(80, 600) * 100
    inp["sw"] = rng.randint(3, 40) * 100
    if d == 5:
        inp["target"] = rng.randint(10, 200) * 100
        inp["tax"] = F(rng.choice([20, 21, 25, 28, 30]))

    def contrib(i, comm=True):
        c = i["price"] - i["mat"] - i["lab"] - i.get("pack", 0)
        if comm and "comm" in i:
            c -= i["price"] * i["comm"] / 100
        return c

    def need(i, comm=True, fixed_skip=None, gross_up=True, rounding=math.ceil):
        fixed = sum(i[k] for k in ("rent", "sal", "sw") if k != fixed_skip)
        req = F(fixed)
        if "target" in i:
            req += i["target"] / (1 - i["tax"] / 100) if gross_up else i["target"]
        c = contrib(i, comm)
        if c <= 0:
            raise Skip("negative contribution")
        return F(rounding(req / c))

    def compute(i):
        return need(i)

    def wrongs(i):
        out = [need(i) - 1, need(i, fixed_skip="sw"), need(i, fixed_skip="rent")]
        if "comm" in i:
            out.append(need(i, comm=False))
        if "target" in i:
            out.append(need(i, gross_up=False))
        out.append(F(math.ceil(F(i["rent"] + i["sal"] + i["sw"]) / i["price"])))
        return out

    def render(i, missing, r):
        mm = lambda k: _cell(i, k, missing, lambda v: fmoney(v, cur))  # noqa: E731
        lines = [T(r, "$co sells the $prod at $p per unit.", co=co, prod=product, p=mm("price")),
                 T(r, "{Variable costs per unit|Per-unit costs}: materials $m, direct labour $l$pk.", m=mm("mat"), l=mm("lab"),
                   pk=(", packaging " + mm("pack")) if "pack" in i else "")]
        if "comm" in i:
            lines.append(T(r, "{Resellers|Channel partners} {earn|are paid} a commission of $c of the selling price on every unit.",
                           c="n/a*" if "comm" in missing else f"{int(i['comm'])}%"))
        lines.append(T(r, "{Fixed monthly costs|Monthly overheads}: {premises rent|rent} $a, salaries $b, {software subscriptions|tooling} $s.",
                       a=_cell(i, "rent", missing, lambda v: fmoney(v, cur, 0)), b=_cell(i, "sal", missing, lambda v: fmoney(v, cur, 0)),
                       s=_cell(i, "sw", missing, lambda v: fmoney(v, cur, 0))))
        if "target" in i:
            lines.append(T(r, "{Management wants|The plan requires} an after-tax monthly profit of at least $t; profit is taxed at $x (tax applies "
                              "only to positive profit).", t=_cell(i, "target", missing, lambda v: fmoney(v, cur, 0)),
                           x="n/a*" if "tax" in missing else f"{int(i['tax'])}%"))
        lines.append(T(r, "{Units are sold whole|Only whole units can be sold}; {no inventory is carried over|everything produced is sold}."))
        lines += _missing_note(missing, r)
        head = [T(r, "{Unit economics|Pricing and cost sheet|Contribution analysis} - $prod", prod=product)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    goal = "reach the after-tax profit target" if d == 5 else "break even"
    qc = T(rng, "{What is the minimum number of whole units $co must sell in a month to $g|At minimum, how many whole units must $co sell in a month to $g}?", co=co, g=goal)
    qn = lambda t: T(rng, "Does $co need to sell more than $t units in a month to $g?", co=co, t=t, g=goal)  # noqa: E731
    decisive = ["price", "mat", "lab", "rent", "sal"] + (["comm"] if "comm" in inp else []) + (["target", "tax"] if "target" in inp else [])
    return numeric_item(rng, "table_arithmetic", "tab_breakeven", inp=inp, q={}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                        q_noul=qn, places=0, fmt=lambda v: fnum(v), decisive=decisive, min_gap=1)


KINDS = {"tab_growth": tab_growth, "tab_cagr": tab_cagr, "tab_compound": tab_compound, "tab_fx": tab_fx, "tab_weighted": tab_weighted,
         "tab_ratio": tab_ratio, "tab_variance": tab_variance, "tab_inventory": tab_inventory, "tab_saas": tab_saas, "tab_breakeven": tab_breakeven}
KIND_FAMILY = {"tab_growth": "temporal_numeric", "tab_cagr": "temporal_numeric", "tab_compound": "temporal_numeric"}
