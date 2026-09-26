"""Family temporal_numeric: deadlines, business days, time zones, SLA windows, shift pay, unit conversions, dated FX,
tenure eligibility and recurring schedules. Answers are exact (dates, datetimes, integers or money)."""
from __future__ import annotations

import calendar
import datetime as dt
import random
from fractions import Fraction as F

from .common import (CURRENCIES, DOMAINS, MONTHS, WEEKDAYS, Names, Skip, T, assemble, fdate, fmoney, fnum, rand_date, rq)
from .numeric_base import md_table, numeric_item, value_item

FAM = "temporal_numeric"


def _dom(rng):
    k = rng.choice(sorted(DOMAINS))
    return k, DOMAINS[k]


def is_bday(d: dt.date, closed) -> bool:
    return d.weekday() < 5 and d not in closed


def add_bdays(d: dt.date, n: int, closed=(), workdays=(0, 1, 2, 3, 4)) -> dt.date:
    cur, k = d, 0
    while k < n:
        cur += dt.timedelta(days=1)
        if cur.weekday() in workdays and cur not in closed:
            k += 1
    return cur


def fdt(t: dt.datetime, style: int) -> str:
    return f"{WEEKDAYS[t.weekday()][:3]} {t.day} {MONTHS[t.month][:3]} {t.year}, {t:%H:%M}" if style % 2 == 0 else \
        f"{t.day} {MONTHS[t.month]} {t.year} at {t:%H:%M}"


def foff(m: int) -> str:
    s = "+" if m >= 0 else "-"
    m = abs(m)
    return f"UTC{s}{m // 60:02d}:{m % 60:02d}"


def _shift_dates(inp, key, days):
    out = []
    for dd in days:
        a = dict(inp)
        a[key] = inp[key] + dt.timedelta(days=dd)
        out.append(a)
    return out


def _num_perturb(inp, key, deltas):
    out = []
    for dd in deltas:
        a = dict(inp)
        v = inp[key] + dd
        if v > 0:
            a[key] = v
            out.append(a)
    return out


# ======================================================================================= business-day deadline
def tm_deadline(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    org = names.company()
    ds = rng.randrange(5)
    recv_d = rand_date(rng)
    recv_t = dt.time(rng.randint(7, 19), rng.choice([0, 10, 25, 40, 55]))
    n = {3: rng.randint(3, 8), 4: rng.randint(8, 15), 5: rng.randint(10, 20)}[d]
    span = [recv_d + dt.timedelta(days=k) for k in range(1, n * 2 + 6)]
    wk = [x for x in span if x.weekday() < 5]
    closures = sorted(set(rng.sample(wk[: max(2, len(wk) * 2 // 3)], {3: 1, 4: 2, 5: 3}[d])))
    if d >= 4:
        we = [x for x in span if x.weekday() >= 5]
        closures = sorted(set(closures + [rng.choice(we)]))
    cutoff = rng.choice([15, 16, 17]) if d == 5 else None
    inp = {"recv": dt.datetime.combine(recv_d, recv_t), "n": n, "closures": closures, "cutoff": cutoff}
    thing = rng.choice(["a written response", "an acknowledgement", "a decision letter", "a formal reply", "a remediation plan"])
    subject = rng.choice(["complaint", "subject-access request", "supplier dispute", "warranty claim", "appeal", "grievance"])

    def start(i, use_cut=True):
        r = i["recv"]
        base = r.date()
        if use_cut and i["cutoff"] is not None and (r.time() >= dt.time(i["cutoff"]) or not is_bday(base, set(i["closures"]))):
            base = add_bdays(base, 1, set(i["closures"]))
        return base

    def compute(i):
        return add_bdays(start(i), i["n"], set(i["closures"]))

    def wrongs(i):
        cl = set(i["closures"])
        return [start(i) + dt.timedelta(days=i["n"]), add_bdays(start(i), i["n"]), add_bdays(start(i), i["n"] - 1, cl),
                add_bdays(start(i, use_cut=False), i["n"], cl), add_bdays(start(i), i["n"] + 1, cl),
                add_bdays(start(i), i["n"], {x for x in cl if x.weekday() < 5} | {x + dt.timedelta(days=2) for x in cl if x.weekday() >= 5})]

    def pending(i):
        inw = [c for c in i["closures"] if c.weekday() < 5 and start(i) < c <= compute(i)]
        return inw[0] if inw else None

    def render(i, missing, r):
        pend = pending(i) if "closures" in missing else None
        cl_txt = [fdate(c, ds) for c in i["closures"] if c != pend]
        lines = [T(r, "{Under|Per} {clause|section} $cl of the $org {service charter|complaints procedure|response policy}, $thing {must be sent|is due} "
                      "{within|no later than} $n business days of receipt. Business days are Monday to Friday, excluding the office "
                      "closure dates listed below. Day 1 is the first business day after the day of receipt.",
                   cl=f"{r.randint(2, 9)}.{r.randint(1, 9)}", org=org, thing=thing, n="[see the escalation schedule]" if "n" in missing else i["n"]),
                 T(r, "The $s from $p was received on $when.", s=subject, p=names.person(),
                   when="[date stamp illegible]" if "recv" in missing else fdt(i["recv"], ds)),
                 T(r, "Office closure dates{ this period|}: $c.", c="; ".join(cl_txt) or "none")]
        if pend is not None:
            lines.append(T(r, "{Facilities has proposed|There is a proposal} to also close the office on $p; {the decision has not been taken yet|"
                              "it is still awaiting approval}.", p=fdate(pend, ds)))
        if i["cutoff"] is not None:
            lines.append(T(r, "Items received at or after $h:00 on a business day, or on a non-business day, are treated as received on the "
                              "next business day.", h=i["cutoff"]))
        if d >= 4:
            lines.append(T(r, "{Closures that fall on a weekend are listed for completeness only|A closure on a Saturday or Sunday does not move any other day}."))
        head = [T(r, "{Case file|Correspondence log|Tracking note} $c", c=case_id)]
        return assemble(r, head, lines, dom, Names(r, names.used), d, anchor=recv_d)

    case_id = names.code(rng.choice(["CASE", "REF", "CMP"]))
    qc = T(rng, "{What is the last day|On which date is the last day} on which $t on $c can be sent on time?", t=thing, c=case_id)

    def noul(r, val):
        probe = add_bdays(val, r.choice([1, 2]), set(inp["closures"])) if r.random() < 0.5 else val - dt.timedelta(days=r.choice([1, 2, 3]))
        return T(r, "$t on $c {went out|was sent} on $p. Was it on time?", t=thing, c=case_id, p=fdate(probe, ds)), probe

    def perturb(k, i):
        if k == "n":
            return _num_perturb(i, "n", [1, -1, 2])
        if k == "recv":
            return _shift_dates(i, "recv", [1, 2, -1, 3])
        p = pending(i)
        if p is None:
            return []
        a = dict(i)
        a["closures"] = [c for c in i["closures"] if c != p]
        return [a]

    return value_item(rng, FAM, "tm_deadline", inp=inp, q={}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: fdate(v, ds), decisive=["n", "recv", "closures"], perturb=perturb, noul=noul,
                      judge=lambda v, p: p <= v, fill=lambda k: compute(inp) + dt.timedelta(days=(k + 1) // 2 * (-1) ** k))


# ======================================================================================= leave-day count
def tm_leave(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    who = names.person()
    ds = rng.randrange(5)
    a = rand_date(rng)
    while a.weekday() >= 5:
        a += dt.timedelta(days=1)
    b = a + dt.timedelta(days=rng.randint(6, 24))
    days = [a + dt.timedelta(days=k) for k in range((b - a).days + 1)]
    wk = [x for x in days if x.weekday() < 5]
    hol = sorted(rng.sample(wk, {3: 1, 4: 1, 5: 2}[d]))
    we = [x for x in days if x.weekday() >= 5]
    if d >= 4 and we:
        hol = sorted(set(hol + [rng.choice(we)]))
    off = rng.choice([0, 4]) if d >= 4 else None  # weekday index not worked (part-time)
    sick = rng.choice([x for x in wk if x not in hol and x.weekday() != off]) if d == 5 else None
    inp = {"a": a, "b": b, "hol": hol, "off": off, "sick": sick}

    def count(i, use_hol=True, use_off=True, use_sick=True, weekend_hol=False):
        k = 0
        x = i["a"]
        while x <= i["b"]:
            wd = x.weekday() < 5
            if use_off and i["off"] is not None and x.weekday() == i["off"]:
                wd = False
            if use_hol and x in i["hol"]:
                wd = False
            if use_sick and i["sick"] is not None and x == i["sick"]:
                wd = False
            k += wd
            x += dt.timedelta(days=1)
        if weekend_hol:
            k -= sum(1 for h in i["hol"] if h.weekday() >= 5)
        return k

    def compute(i):
        return count(i)

    def wrongs(i):
        return [(i["b"] - i["a"]).days + 1, count(i, use_hol=False), count(i, use_off=False), count(i, use_sick=False), count(i, weekend_hol=True),
                count(i) + 1, count(i) - 1]

    def render(i, missing, r):
        lines = [T(r, "$w has booked leave from $a to $b{ inclusive|, both days included}.", w=who, a=fdate(i["a"], ds),
                   b="[end date to be confirmed]" if "b" in missing else fdate(i["b"], ds)),
                 T(r, "Leave is deducted only for days $w would normally work; public holidays and company closure days are never deducted.", w=who.split()[0])]
        hol_txt = "; ".join(fdate(h, ds) for h in i["hol"])
        lines.append(T(r, "{Public holidays|Company closure days} in the period: $h.", h="[calendar not yet published]" if "hol" in missing else hol_txt))
        if i["off"] is not None:
            lines.append(T(r, "$w works a four-day pattern and does not work on $o.", w=who.split()[0],
                           o="[pattern not recorded]" if "off" in missing else WEEKDAYS[i["off"]] + "s"))
        elif d >= 4:
            lines.append(T(r, "$w works a standard Monday-to-Friday week.", w=who.split()[0]))
        if i["sick"] is not None:
            lines.append(T(r, "On $s $w was signed off sick; that day is recorded as sick leave and not taken from the holiday allowance.",
                           s=fdate(i["sick"], ds), w=who.split()[0]))
        head = [T(r, "{Absence request|Leave booking|Holiday form} $b - $w", b=bk, w=who)]
        return assemble(r, head, lines, dom, Names(r, names.used), d, anchor=a)

    bk = names.code("LV")
    qc = T(rng, "How many days {are deducted from|come off} $pw holiday allowance for booking $b?", pw=who + "'s", b=bk)

    def noul(r, val):
        p = val + r.choice([-2, -1, 1, 2])
        return T(r, "Does booking $b use more than $p days of $pw allowance?", b=bk, p=p, pw=who + "'s"), p

    def perturb(k, i):
        if k == "b":
            return _shift_dates(i, "b", [1, 3, -2, 5])
        if k == "off" and i["off"] is not None:
            return [dict(i, off=o) for o in (0, 1, 2, 3, 4) if o != i["off"]]
        if k == "hol":
            return [dict(i, hol=[])]
        return []

    return value_item(rng, FAM, "tm_leave", inp=inp, q={}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: f"{v} days" if v != 1 else "1 day", decisive=["b", "off", "hol"], perturb=perturb, noul=noul,
                      judge=lambda v, p: v > p, fill=lambda k: compute(inp) + (k + 1) // 2 * (-1) ** k)


# ======================================================================================= time zones and travel
CITIES_TZ = [("Lisbon", 0), ("Madrid", 60), ("Athens", 120), ("Nairobi", 180), ("Dubai", 240), ("Karachi", 300), ("Mumbai", 330),
             ("Kathmandu", 345), ("Dhaka", 360), ("Bangkok", 420), ("Singapore", 480), ("Osaka", 540), ("Adelaide", 570), ("Brisbane", 600),
             ("Auckland", 720), ("Sao Paulo", -180), ("Halifax", -240), ("Bogota", -300), ("Chicago", -360), ("Denver", -420),
             ("Vancouver", -480), ("Anchorage", -540), ("Honolulu", -600), ("Reykjavik", 0), ("Tehran", 210), ("Chatham Islands", 765)]


def tm_timezone(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    ds = rng.randrange(2)
    (ca, oa), (cb, ob), (cc, oc) = rng.sample(CITIES_TZ, 3)
    dep = dt.datetime.combine(rand_date(rng), dt.time(rng.randint(0, 23), rng.choice([0, 5, 15, 20, 30, 45, 50])))
    inp = {"t": dep, "oa": oa, "ob": ob}
    mode = {3: "convert", 4: "fly", 5: "two_legs"}[d]
    traveller = names.person()
    if mode != "convert":
        inp["d1"] = rng.randint(55, 14 * 60)
    if mode == "two_legs":
        inp["oc"] = oc
        inp["lay"] = rng.randint(45, 300)
        inp["d2"] = rng.randint(50, 10 * 60)

    def compute(i, sign=1, skip_lay=False, no_conv=False):
        utc = i["t"] - dt.timedelta(minutes=i["oa"]) * sign
        tot = i.get("d1", 0) + (0 if skip_lay else i.get("lay", 0)) + i.get("d2", 0)
        arr_utc = utc + dt.timedelta(minutes=tot)
        return arr_utc + dt.timedelta(minutes=i["ob"]) * sign if not no_conv else i["t"] + dt.timedelta(minutes=tot)

    def wrongs(i):
        g = compute(i)
        out = [compute(i, sign=-1), compute(i, no_conv=True), g.replace(day=i["t"].day, month=i["t"].month, year=i["t"].year) if g.date() != i["t"].date() else g + dt.timedelta(days=1)]
        if "lay" in i:
            out.append(compute(i, skip_lay=True))
            out.append(g + dt.timedelta(minutes=i["oc"] - i["ob"]))
        out.append(g + dt.timedelta(hours=1))
        out.append(g - dt.timedelta(hours=1))
        return out

    def render(i, missing, r):
        off = lambda k, v: "[offset not listed]" if k in missing else foff(v)  # noqa: E731
        tz = [f"{ca}: {off('oa', i['oa'])}", f"{cb}: {off('ob', i['ob'])}"]
        if "oc" in i:
            tz.append(f"{cc}: {off('oc', i['oc'])}")
        r.shuffle(tz)
        lines = [T(r, "{UTC offsets in force on the travel dates|Offsets valid for these dates (already adjusted for daylight saving)}: $z.", z="; ".join(tz))]
        p = traveller
        if mode == "convert":
            lines.insert(0, T(r, "{The|A} {vendor call|board briefing|go-live check-in|incident bridge} {is set|is scheduled} to start at $t local time in $a. "
                                 "$p {will join|dials in} from $b.", t=fdt(i["t"], ds), a=ca, p=p, b=cb))
        else:
            lines.insert(0, T(r, "$p {departs|leaves} $a at $t local time{ on flight| on service} $f.", p=p, a=ca, t=fdt(i["t"], ds),
                              f=f"{r.choice('BKQRSTVZ')}{r.choice('AEHJLMOX')}{r.randint(100, 999)}"))
            dur = lambda k: "[duration not in the itinerary]" if k in missing else f"{i[k] // 60} h {i[k] % 60:02d} min"  # noqa: E731
            if mode == "fly":
                lines.insert(1, T(r, "The flight to $b {is scheduled at|takes} $d{ gate to gate| block time}.", b=cb, d=dur("d1")))
            else:
                lines.insert(1, T(r, "The first leg to $c {takes|is scheduled at} $d1; the connection in $c {is|lasts} $l; the second leg to $b "
                                     "{takes|is scheduled at} $d2.", c=cc, d1=dur("d1"),
                                  l="[layover not stated]" if "lay" in missing else f"{i['lay']} minutes", d2=dur("d2"), b=cb))
        head = [T(r, "{Itinerary|Travel note|Meeting logistics} for $p", p=p)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    if mode == "convert":
        qc = T(rng, "When the call starts, what is the local date and time for $p in $b?", p=traveller, b=cb)
    else:
        qc = T(rng, "{At what local date and time|When, in local time,} {does|will} $p {arrive|land} in $b?", p=traveller, b=cb)

    def noul(r, val):
        probe = val + dt.timedelta(minutes=r.choice([-150, -90, -45, 45, 90, 150]))
        what = "the call has started by" if mode == "convert" else traveller + " has landed in " + cb + " by"
        return T(r, "Is it true that $w $p local time in $b?", w=what, p=fdt(probe, ds), b=cb), probe

    def perturb(k, i):
        if k in ("oa", "ob", "oc"):
            return [dict(i, **{k: i[k] + x}) for x in (60, -60, 120, 180)]
        if k in ("d1", "d2", "lay"):
            return _num_perturb(i, k, [60, -40, 120, 200])
        return []

    return value_item(rng, FAM, "tm_timezone", inp=inp, q={"mode": mode}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: fdt(v, ds), decisive=["ob", "oa", "d1", "lay", "d2"], perturb=perturb, noul=noul,
                      judge=lambda v, p: p >= v, fill=lambda k: compute(inp) + dt.timedelta(minutes=30 * ((k + 1) // 2) * (-1) ** k))


# ======================================================================================= SLA in business hours
def sla_due(opened: dt.datetime, minutes: int, oh: int, ch: int, hol=(), pauses=(), all_days=False, ignore_bh=False) -> dt.datetime:
    t = opened
    left = minutes
    step = dt.timedelta(minutes=1)
    guard = 0
    while left > 0:
        guard += 1
        if guard > 60 * 24 * 60:
            raise Skip("sla loop")
        ok = True
        if not ignore_bh:
            ok = (all_days or t.weekday() < 5) and t.date() not in hol and oh * 60 <= t.hour * 60 + t.minute < ch * 60
        for a, b in pauses:
            if a <= t < b:
                ok = False
        t += step
        if ok:
            left -= 1
    return t


def tm_sla(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["saas_ops", "healthcare_admin", "finance", "legal_compliance", "logistics", "energy"])]
    ds = rng.randrange(2)
    oh, ch = rng.choice([(8, 18), (9, 17), (7, 19), (8, 17), (9, 18)])
    base = rand_date(rng)
    opened = dt.datetime.combine(base, dt.time(rng.randint(0, 23), rng.choice([0, 7, 12, 20, 33, 41, 58])))
    prio = rng.choice(["P2", "P3", "P4"])
    hours = {"P2": rng.choice([4, 6, 8]), "P3": rng.choice([12, 16, 20]), "P4": rng.choice([24, 32, 40])}
    inp = {"opened": opened, "h": hours[prio], "oh": oh, "ch": ch, "hol": [], "pause": None}
    if d >= 4:
        ps = opened + dt.timedelta(minutes=rng.randint(60, 60 * 30))
        inp["pause"] = [ps, ps + dt.timedelta(minutes=rng.randint(45, 60 * 26))]
    if d == 5:
        span = [base + dt.timedelta(days=k) for k in range(1, 8)]
        inp["hol"] = [rng.choice([x for x in span if x.weekday() < 5])]

    def compute(i, pause=True, hol=True, bh=True):
        pz = [tuple(i["pause"])] if (i["pause"] and pause) else []
        return sla_due(i["opened"], i["h"] * 60, i["oh"], i["ch"], set(i["hol"]) if hol else set(), pz, ignore_bh=not bh)

    def wrongs(i):
        return [compute(i, bh=False), compute(i, pause=False), compute(i, hol=False),
                sla_due(i["opened"], i["h"] * 60, i["oh"], i["ch"], set(i["hol"]), [tuple(i["pause"])] if i["pause"] else [], all_days=True),
                compute(i) + dt.timedelta(hours=1)]

    def render(i, missing, r):
        tbl = md_table(["Priority", "Resolution target (business hours)"],
                       [[p, "[under review]" if (p == prio and "h" in missing) else str(hours[p])] for p in ("P2", "P3", "P4")])
        lines = [T(r, "{The|Our} support {desk|team} {works|is staffed} Monday to Friday, $oh:00-$ch:00 local time; the resolution clock runs only during "
                      "these hours.", oh=f"{i['oh']:02d}", ch=f"{i['ch']:02d}"),
                 T(r, "Ticket $t ($p) was {opened|logged} on $o.", t=tkt, p=prio, o=fdt(i["opened"], ds))]
        if i["pause"]:
            ps, pe = i["pause"]
            lines.append(T(r, "The ticket was set to \"awaiting customer\" at $a and {moved back to|returned to} \"in progress\" at $b; "
                              "the clock {does not run|is paused} while a ticket awaits the customer.", a=fdt(ps, ds),
                           b="[timestamp not logged]" if "pause" in missing else fdt(pe, ds)))
        if i["hol"]:
            lines.append(T(r, "$h is a public holiday and the desk is closed all day.", h=fdate(i["hol"][0], 2)))
        head = [T(r, "{SLA policy extract|Service-level terms|Support SLA} and ticket history")]
        return assemble(r, head, [tbl] + lines, dom, Names(r, names.used), d)

    tkt = names.code("TKT")
    qc = T(rng, "{By when|At what date and time} must $t be resolved to meet its {resolution target|target}?", t=tkt)

    def noul(r, val):
        probe = val + dt.timedelta(minutes=r.choice([-120, -50, -20, 20, 50, 120]))
        return T(r, "$t was resolved on $p. Was {its|the} resolution target met?", t=tkt, p=fdt(probe, ds)), probe

    def perturb(k, i):
        if k == "h":
            return [dict(i, h=i["h"] + x) for x in (2, 4, -2, 8) if i["h"] + x > 0]
        if k == "pause" and i["pause"]:
            return [dict(i, pause=[i["pause"][0], i["pause"][1] + dt.timedelta(minutes=x)]) for x in (240, 600, 1440)]
        return []

    return value_item(rng, FAM, "tm_sla", inp=inp, q={"prio": prio}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: fdt(v, ds), decisive=["h", "pause"], perturb=perturb, noul=noul, judge=lambda v, p: p <= v,
                      fill=lambda k: compute(inp) + dt.timedelta(minutes=45 * ((k + 1) // 2) * (-1) ** k))


# ======================================================================================= shift pay
def tm_shift_pay(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["retail", "logistics", "hospitality", "healthcare_admin", "manufacturing", "energy"])]
    cur = rng.choice(CURRENCIES[:7])
    who = names.person()
    monday = rand_date(rng)
    monday -= dt.timedelta(days=monday.weekday())
    nsh = {3: rng.randint(4, 5), 4: rng.randint(5, 6), 5: rng.randint(5, 7)}[d]
    days = sorted(rng.sample(range(7 if d == 5 else 6), nsh))
    inp = {"rate": F(rng.randint(1400, 3800), 100)}
    for j, wd in enumerate(days):
        st = rng.choice([6, 7, 8, 9, 12, 14, 16, 18, 20, 21, 22]) * 60 + rng.choice([0, 15, 30, 45])
        length = rng.randint(6, 11) * 60 + rng.choice([0, 15, 30, 45])
        inp[f"s|{j}"] = st
        inp[f"e|{j}"] = (st + length) % 1440
        inp[f"b|{j}"] = rng.choice([0, 20, 30, 30, 45, 60]) if d >= 4 else 30
    ot = 40 * 60

    def mins(i, j, midnight=True, brk=True):
        a, b = i[f"s|{j}"], i[f"e|{j}"]
        m = b - a
        if m <= 0:
            m = m + 1440 if midnight else abs(m)
        return m - (i[f"b|{j}"] if brk else 0)

    def pay(i, midnight=True, brk=True, use_ot=True, sunday=True, ot_mult=F(3, 2)):
        reg, tot = 0, F(0)
        for j, wd in enumerate(days):
            m = mins(i, j, midnight, brk)
            if wd == 6 and sunday:
                tot += F(m, 60) * i["rate"] * 2
                continue
            if wd == 6:
                pass
            normal = max(0, min(m, ot - reg)) if use_ot else m
            extra = m - normal
            reg += m
            tot += F(normal, 60) * i["rate"] + F(extra, 60) * i["rate"] * ot_mult
        return tot

    def compute(i):
        return pay(i)

    def wrongs(i):
        return [pay(i, brk=False), pay(i, use_ot=False), pay(i, midnight=False), pay(i, sunday=False), pay(i, ot_mult=F(2))]

    def render(i, missing, r):
        rows = []
        for j, wd in enumerate(days):
            day = monday + dt.timedelta(days=wd)
            rows.append([f"{WEEKDAYS[day.weekday()][:3]} {day.day} {MONTHS[day.month][:3]}", f"{i[f's|{j}'] // 60:02d}:{i[f's|{j}'] % 60:02d}",
                         "[no clock-out]" if f"e|{j}" in missing else f"{i[f'e|{j}'] // 60:02d}:{i[f'e|{j}'] % 60:02d}",
                         "[not recorded]" if f"b|{j}" in missing else f"{i[f'b|{j}']} min"])
        tbl = md_table(["Shift date", "Clock-in", "Clock-out", "Unpaid break"], rows)
        lines = [T(r, "$w is paid $rate per hour for paid time (shift length minus the unpaid break). A clock-out earlier than the clock-in means "
                      "the shift ended after midnight.", w=who, rate="[rate card pending]" if "rate" in missing else fmoney(i["rate"], cur))]
        lines.append(T(r, "{Overtime rule|Premium rule}: paid time beyond 40 hours in the week (counted in shift order) is paid at 1.5 times the hourly rate."))
        if d == 5:
            lines.append(T(r, "Shifts that start on a Sunday are paid at double the hourly rate and do not count towards the 40-hour threshold."))
        head = [T(r, "{Timesheet|Weekly time record|Payroll input} - $w, week commencing $m", w=who, m=fdate(monday, 0))]
        return assemble(r, head, [tbl] + lines, dom, Names(r, names.used), d)

    qc = T(rng, "What is $pw gross pay for the week?", pw=who.split()[0] + "'s")
    qn = lambda t: T(rng, "Is $pw gross pay for the week more than $t?", pw=who.split()[0] + "'s", t=t)  # noqa: E731
    decisive = ["rate"] + [f"e|{j}" for j in range(nsh)] + ([f"b|{j}" for j in range(nsh)] if d >= 4 else [])
    return numeric_item(rng, FAM, "tm_shift_pay", inp=inp, q={"days": days}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                        q_noul=qn, places=2, fmt=lambda v: fmoney(v, cur), decisive=decisive, witness_factors=[1.1, 0.9, 1.25, 0.8])


# ======================================================================================= unit conversions
LB = F("0.45359237")
MI = F("1.609344")
USGAL = F("3.785411784")
IMPGAL = F("4.54609")


def tm_units(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["logistics", "retail", "manufacturing", "energy"])]
    cur = rng.choice(CURRENCIES[:7])
    mode = {3: "weight", 4: "capacity", 5: "fuel"}[d]
    inp = {}
    if mode in ("weight", "capacity"):
        n = rng.randint(3, 6)
        units = []
        for j in range(n):
            u = rng.choice(["kg", "lb", "g", "oz", "t"]) if mode == "capacity" else rng.choice(["kg", "lb", "g", "oz"])
            v = {"kg": F(rng.randint(50, 90000), 100), "lb": F(rng.randint(100, 150000), 100), "g": F(rng.randint(500, 900000)),
                 "oz": F(rng.randint(10, 3000)), "t": F(rng.randint(5, 400), 100)}[u]
            inp[f"w|{j}"] = v
            units.append(u)
        if mode == "capacity":
            inp["tare"] = F(rng.randint(15, 30))
            inp["pallets"] = rng.randint(2, 6)
            inp["limit"] = F(rng.randint(3, 12) * 500)
    else:
        legs = rng.randint(2, 4)
        units = [rng.choice(["mi", "km"]) for _ in range(legs)]
        for j in range(legs):
            inp[f"l|{j}"] = F(rng.randint(200, 6000), 10)
        inp["cons"] = F(rng.randint(60, 340), 10)
        inp["price"] = F(rng.randint(250, 750), 100)

    def kg(v, u, lbf=LB, ozf=LB / 16):
        return {"kg": v, "lb": v * lbf, "g": v / 1000, "oz": v * ozf, "t": v * 1000}[u]

    def compute(i, tare_mult=1, **kw):
        if mode == "weight":
            return sum(kg(i[f"w|{j}"], u, **kw) for j, u in enumerate(units))
        if mode == "capacity":
            load = sum(kg(i[f"w|{j}"], u, **kw) for j, u in enumerate(units)) + i["tare"] * i["pallets"] * tare_mult
            return i["limit"] - load
        return fuel(i)

    def fuel(i, milesf=MI, gal=USGAL, legs_as_km=False):
        km = sum(i[f"l|{j}"] * (milesf if (u == "mi" and not legs_as_km) else 1) for j, u in enumerate(units))
        litres = km * i["cons"] / 100
        return litres / gal * i["price"]

    def wrongs(i):
        if mode == "fuel":
            return [fuel(i, gal=IMPGAL), fuel(i, legs_as_km=True), fuel(i, milesf=1 / MI), fuel(i) * USGAL]
        out = [compute(i, lbf=1 / LB), compute(i, ozf=LB / 12), compute(i, lbf=F(1, 2))]
        if mode == "capacity":
            out += [compute(i, tare_mult=0), compute(i, tare_mult=2)]
        return out

    def render(i, missing, r):
        if mode == "fuel":
            rows = [[f"Leg {j + 1}", "[odometer not logged]" if f"l|{j}" in missing else f"{float(i[f'l|{j}']):,.1f} {u}"] for j, u in enumerate(units)]
            tbl = md_table(["Route leg", "Distance"], rows)
            lines = [T(r, "The {van|truck|generator tender} {uses|burns} $c litres of diesel per 100 km.",
                       c="[consumption figure missing]" if "cons" in missing else f"{float(i['cons']):.1f}"),
                     T(r, "Diesel {costs|is priced at} $p per US gallon{ on the fuel card|}.", p="[price not quoted]" if "price" in missing else fmoney(i["price"], cur)),
                     T(r, "Conversion factors: 1 mile = 1.609344 km; 1 US gallon = 3.785411784 litres; 1 imperial gallon = 4.54609 litres.")]
            head = [T(r, "{Route costing|Trip fuel estimate|Delivery run budget} for route $v", v=ld)]
            return assemble(r, head, [tbl] + lines, dom, Names(r, names.used), d)
        rows = []
        for j, u in enumerate(units):
            v = i[f"w|{j}"]
            txt = "[scale reading lost]" if f"w|{j}" in missing else (f"{float(v):,.2f} {u}" if u in ("kg", "lb", "t") else f"{int(v):,} {u}")
            rows.append([names.code("PKG"), r.choice(["spare parts", "fixtures", "cartons of stock", "cable drums", "sample kits", "tools"]), txt])
        tbl = md_table(["Consignment", "Contents", "Declared weight"], rows)
        lines = [T(r, "Conversion factors: 1 lb = 0.45359237 kg; 1 oz = 1/16 lb; 1 t = 1,000 kg; 1 kg = 1,000 g.")]
        if mode == "capacity":
            lines.append(T(r, "The consignments travel on $p pallets, each weighing $t kg empty.", p=i["pallets"],
                           t="[tare not stated]" if "tare" in missing else fnum(i["tare"])))
            lines.append(T(r, "The {vehicle|trailer|lift} has a maximum payload of $l kg, including pallets.", l=fnum(i["limit"])))
        head = [T(r, "{Load sheet|Weighbridge summary|Dispatch manifest} $c", c=ld)]
        return assemble(r, head, [tbl] + lines, dom, Names(r, names.used), d)

    ld = names.code("LD")
    if mode == "weight":
        qc, qn = T(rng, "What is the total declared weight of load $l in kilograms?", l=ld), \
            (lambda t: T(rng, "Is the total declared weight of load $l more than $t?", l=ld, t=t))
        fmt = lambda v: f"{rq(v, 2):,.2f} kg"  # noqa: E731
    elif mode == "capacity":
        qc, qn = T(rng, "Once load $l and its pallets are on board, how many kilograms of payload remain?", l=ld), \
            (lambda t: T(rng, "With load $l and its pallets on board, is more than $t of payload left?", l=ld, t=t))
        fmt = lambda v: f"{rq(v, 2):,.2f} kg"  # noqa: E731
    else:
        qc, qn = T(rng, "What is the diesel cost of route $l?", l=ld), (lambda t: T(rng, "Does the diesel for route $l cost more than $t?", l=ld, t=t))
        fmt = lambda v: fmoney(v, cur)  # noqa: E731
    val = compute(inp)
    if mode == "capacity" and val <= 50:
        raise Skip("overloaded")
    decisive = [k for k in inp if k.startswith(("w|", "l|"))] + ["cons", "price", "tare"]
    return numeric_item(rng, FAM, "tm_units", inp=inp, q={"mode": mode, "units": units}, compute=compute, wrongs=wrongs, render=render,
                        q_choice=qc, q_noul=qn, places=2, fmt=fmt, decisive=decisive, allow_nonpos=False)


# ======================================================================================= dated FX
def tm_fx_dated(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    ds = rng.randrange(5)
    pair = rng.sample(["USD", "EUR", "GBP", "CHF", "CAD", "AUD", "SGD"], 2)
    base_ccy, quote = pair
    qcur = next(c for c in CURRENCIES if c[0] == quote)
    start = rand_date(rng)
    bdays = [start + dt.timedelta(days=k) for k in range(18) if (start + dt.timedelta(days=k)).weekday() < 5]
    hol = rng.choice(bdays[3:-3]) if d == 5 else None
    pub = [x for x in bdays if x != hol]
    r0 = F(rng.randint(6000, 16000), 10000)
    inp = {}
    for x in pub:
        r0 = r0 * F(rng.randint(9900, 10100), 10000)
        inp[f"r|{x.isoformat()}"] = F(round(float(r0) * 10000), 10000)
    npay = 1 if d == 3 else 2
    pays = []
    for j in range(npay):
        due = bdays[0] + dt.timedelta(days=rng.randint(4, 18))
        if due > bdays[-1]:
            due = bdays[-1] + dt.timedelta(days=1)
        inp[f"a|{j}"] = F(rng.randint(100_000, 5_000_000), 100)
        pays.append(due)
    if d >= 4:
        inp["m"] = F(rng.choice([15, 20, 25, 30, 40, 50]), 100)

    def rate_on(i, x, nxt=False):
        step = 1 if nxt else -1
        for _ in range(30):
            k = f"r|{x.isoformat()}"
            if k in i:
                return i[k]
            x += dt.timedelta(days=step)
        raise Skip("no rate")

    def compute(i, nxt=False, margin=1, invert=False):
        tot = F(0)
        for j, due in enumerate(pays):
            r_ = rate_on(i, due, nxt)
            if invert:
                r_ = 1 / r_
            r_ = r_ * (1 + margin * i.get("m", 0) / 100)
            tot += F(rq(i[f"a|{j}"] * r_, 2))
        return tot

    def wrongs(i):
        return [compute(i, nxt=True), compute(i, margin=0), compute(i, margin=-1), compute(i, invert=True)]

    def render(i, missing, r):
        rows = [[fdate(x, ds), "[feed outage]" if f"r|{x.isoformat()}" in missing else f"{float(i[f'r|{x.isoformat()}']):.4f}"] for x in pub]
        tbl = md_table([f"Business date", f"{quote} per 1 {base_ccy} (reference rate)"], rows)
        lines = []
        for j, due in enumerate(pays):
            lines.append(T(r, "{Payment|Instalment} $n of $a {falls due|is due} on $dd.", n=j + 1,
                           a="[amount redacted]" if f"a|{j}" in missing else f"{base_ccy} {float(i[f'a|{j}']):,.2f}", dd=fdate(due, ds)))
        lines.append(T(r, "{Treasury policy|FX rule}: each payment is converted at the reference rate of its due date; if no rate is published for that date "
                          "(weekend or holiday), use the rate of the {closest|most recent} earlier business date. Each converted payment is rounded to the cent."))
        if "m" in i:
            lines.append(T(r, "The bank {adds|applies} a margin of $m on top of the reference rate (the rate paid is reference x (1 + margin)).",
                           m="[margin under negotiation]" if "m" in missing else f"{float(i['m']):.2f}%"))
        if hol:
            lines.append(T(r, "No reference rate was published on $h (bank holiday).", h=fdate(hol, ds)))
        head = [T(r, "{Settlement schedule|FX conversion memo|Cross-border payment plan} - $c", c=names.company())]
        return assemble(r, head, lines + [tbl], dom, Names(r, names.used), d)

    qc = T(rng, "What is the total cost in $q of the {payment|payments} {listed|above}?", q=quote)
    qn = lambda t: T(rng, "Does the total cost in $q of the {payment|payments} exceed $t?", q=quote, t=t)  # noqa: E731
    needed = []
    for due in pays:
        x = due
        while f"r|{x.isoformat()}" not in inp:
            x -= dt.timedelta(days=1)
        needed.append(f"r|{x.isoformat()}")
    decisive = needed + [f"a|{j}" for j in range(npay)] + (["m"] if "m" in inp else [])
    return numeric_item(rng, FAM, "tm_fx_dated", inp=inp, q={"pays": [p.isoformat() for p in pays], "base": base_ccy, "quote": quote},
                        compute=compute, wrongs=wrongs, render=render, q_choice=qc, q_noul=qn, places=2, fmt=lambda v: fmoney(v, qcur),
                        decisive=decisive, witness_factors=[1.05, 0.95, 1.1, 0.9, 1.2])


# ======================================================================================= tenure eligibility
def add_months(d: dt.date, n: int) -> dt.date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def tm_tenure(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(["hr", "education", "finance", "retail", "healthcare_admin"])]
    who = names.person()
    ds = rng.randrange(5)
    hire = rand_date(rng, 2022, 2026)
    if rng.random() < 0.3:
        hire = hire.replace(day=min(rng.choice([29, 30, 31]), calendar.monthrange(hire.year, hire.month)[1]))
    nmon = rng.choice([3, 6, 9, 12, 18, 24])
    inp = {"hire": hire, "n": nmon, "leave": []}
    blocks = {3: 0, 4: 1, 5: 2}[d]
    cur = hire + dt.timedelta(days=20)
    for _ in range(blocks):
        a = cur + dt.timedelta(days=rng.randint(10, 90))
        b = a + dt.timedelta(days=rng.randint(3, 40))
        inp["leave"].append([a, b])
        cur = b
    if any(b >= add_months(hire, nmon) for _, b in inp["leave"]):
        raise Skip("leave after qualifying date")
    benefit = rng.choice(["the pension match", "the private medical plan", "the share scheme", "enhanced parental pay", "the study-support fund"])

    def compute(i, use_leave=True, n_off=0, weeks=False):
        base = add_months(i["hire"], i["n"] + n_off)
        extra = sum((b - a).days + 1 for a, b in i["leave"]) if use_leave else 0
        if weeks:
            extra = sum(((b - a).days + 1) // 7 * 7 for a, b in i["leave"])
        return base + dt.timedelta(days=extra)

    def wrongs(i):
        return [compute(i, use_leave=False), compute(i, n_off=-1), compute(i, n_off=1), compute(i, weeks=True) if i["leave"] else None,
                compute(i) - dt.timedelta(days=len(i["leave"])) if i["leave"] else compute(i) + dt.timedelta(days=1)]

    def render(i, missing, r):
        lines = [T(r, "$w {joined|started with} $o on $h.", w=who, o=names.company(), h="[start date missing from the file]" if "hire" in missing else fdate(i["hire"], ds)),
                 T(r, "{Eligibility|The eligibility rule} for $b: an employee qualifies after $n {calendar |}months of service. The qualifying date is the same "
                      "day of the month $n months after the start date (or the last day of that month if it has no such day), pushed back by one calendar "
                      "day for each day of unpaid leave taken before it.", b=benefit, n="[N - see schedule B, not attached]" if "n" in missing else i["n"])]
        for j, (a, b) in enumerate(i["leave"]):
            lines.append(T(r, "$w took unpaid leave from $a to $b{ inclusive|, both dates included}.", w=who.split()[0], a=fdate(a, ds),
                           b="[return date not recorded]" if f"leave{j}" in missing else fdate(b, ds)))
        if d >= 4:
            lines.append(T(r, "Paid annual leave and sick leave {do not affect|have no effect on} the qualifying date."))
        head = [T(r, "{Benefits eligibility check|HR eligibility note|Service record extract} - $w", w=who)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    qc = T(rng, "On what date does $w first qualify for $b?", w=who.split()[0], b=benefit)

    def noul(r, val):
        probe = val + dt.timedelta(days=r.choice([-9, -3, -1, 1, 4, 12]))
        return T(r, "Does $w qualify for $b on $p?", w=who.split()[0], b=benefit, p=fdate(probe, ds)), probe

    def perturb(k, i):
        if k == "hire":
            return _shift_dates(i, "hire", [3, -5, 10])
        if k == "n":
            return _num_perturb(i, "n", [3, 6, -3])
        return []

    return value_item(rng, FAM, "tm_tenure", inp=inp, q={}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: fdate(v, ds), decisive=["hire", "n"], perturb=perturb, noul=noul, judge=lambda v, p: p >= v,
                      fill=lambda k: compute(inp) + dt.timedelta(days=((k + 1) // 2) * (-1) ** k))


# ======================================================================================= recurring schedules
def nth_weekday(y, m, wd, nth):
    d0 = dt.date(y, m, 1)
    off = (wd - d0.weekday()) % 7
    x = d0 + dt.timedelta(days=off + 7 * (nth - 1))
    return x if x.month == m else None


def tm_recurring(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    _, dom = _dom(rng)
    ds = rng.randrange(5)
    mode = rng.choice(["bstep", "monthly"]) if d >= 4 else "bstep"
    job = rng.choice(["the ledger backup", "the stock count", "the payroll export", "the fire-alarm test", "the vendor sync", "the data-quality sweep",
                      "the courier collection", "the compliance sample"])
    s = rand_date(rng)
    while s.weekday() >= 5:
        s += dt.timedelta(days=1)
    if mode == "bstep":
        inp = {"s": s, "k": rng.randint(2, 5), "m": rng.randint(4, {3: 6, 4: 9, 5: 12}[d]), "closed": []}
        span = [s + dt.timedelta(days=x) for x in range(1, inp["k"] * inp["m"] * 2)]
        inp["closed"] = sorted(rng.sample([x for x in span if x.weekday() < 5], {3: 1, 4: 2, 5: 3}[d]))
    else:
        inp = {"y": s.year, "mo": s.month, "wd": rng.randrange(5), "nth": rng.randint(1, 4), "m": rng.randint(3, 9 if d == 4 else 14),
               "shift": d == 5}

    def compute(i, variant=None):
        if mode == "bstep":
            cl = set(i["closed"]) if variant != "nocl" else set()
            x = i["s"]
            runs = 1
            target = i["m"] - (1 if variant == "zero" else 0)
            while runs < target:
                x = add_bdays(x, i["k"], cl)
                runs += 1
            if variant == "cal":
                x = i["s"] + dt.timedelta(days=i["k"] * (i["m"] - 1))
            return x
        y, mo = i["y"], i["mo"]
        mcount = i["m"] - 1 + (1 if variant == "zero" else 0)
        mo2 = mo - 1 + mcount
        y2, mo2 = y + mo2 // 12, mo2 % 12 + 1
        nth = i["nth"] + (1 if variant == "nth" else 0) if i["nth"] < 4 else i["nth"] - (1 if variant == "nth" else 0)
        x = nth_weekday(y2, mo2, i["wd"], nth)
        if i["shift"] and variant != "noshift":
            x = x + dt.timedelta(days=1)
        return x

    def wrongs(i):
        vs = ["zero", "cal", "nocl"] if mode == "bstep" else ["zero", "nth", "noshift"]
        return [compute(i, v) for v in vs] + [compute(i) + dt.timedelta(days=7), compute(i) - dt.timedelta(days=1)]

    def render(i, missing, r):
        if mode == "bstep":
            lines = [T(r, "$j {first ran|ran for the first time} on $s (run 1) and then runs every $k business days (business days are Monday to Friday, "
                          "excluding closure days; a run never happens on a closure day, and the count skips it).", j=job, s=fdate(i["s"], ds),
                       k="[interval to be set by ops]" if "k" in missing else i["k"]),
                     T(r, "Closure days: $c.", c="; ".join(fdate(x, ds) for x in i["closed"]))]
            q_ = T(r, "On which date does run $m of $j take place?", m=i["m"], j=job)
        else:
            ords = ["first", "second", "third", "fourth"]
            lines = [T(r, "$j {takes place|happens} on the $o $w of every month, starting in $mon $y (that month's run is run 1).", j=job,
                       o="[ordinal not specified]" if "nth" in missing else ords[i["nth"] - 1], w=WEEKDAYS[i["wd"]], mon=MONTHS[i["mo"]], y=i["y"])]
            if i["shift"]:
                lines.append(T(r, "{Because of a clash with the month-end close|To avoid the change-freeze window}, each run is {moved|shifted} to the day "
                                  "after its scheduled date."))
            q_ = T(r, "On which date does run $m of $j take place?", m=i["m"], j=job)
        head = [T(r, "{Operations calendar|Recurring task schedule|Run book extract} - $w: $j", w=where, j=job)]
        return assemble(r, head, lines, dom, Names(r, names.used), d)

    where = names.company()
    qc = T(rng, "On which date does run $m of $j at $w take place?", m=inp["m"], j=job, w=where)

    def noul(r, val):
        probe = val + dt.timedelta(days=r.choice([-7, -1, 0, 0, 1, 7]))
        return T(r, "Does run $m of $j at $w take place on $p?", m=inp["m"], j=job, w=where, p=fdate(probe, ds)), probe

    def perturb(k, i):
        if k == "k":
            return _num_perturb(i, "k", [1, -1, 2])
        if k == "nth":
            return [dict(i, nth=x) for x in (1, 2, 3, 4) if x != i["nth"]]
        return []

    return value_item(rng, FAM, "tm_recurring", inp=inp, q={"mode": mode}, compute=compute, wrongs=wrongs, render=render, q_choice=qc,
                      fmt=lambda v: fdate(v, ds), decisive=["k", "nth"], perturb=perturb, noul=noul, judge=lambda v, p: p == v,
                      fill=lambda k: compute(inp) + dt.timedelta(days=((k + 1) // 2) * (-1) ** k))


KINDS = {"tm_deadline": tm_deadline, "tm_leave": tm_leave, "tm_timezone": tm_timezone, "tm_sla": tm_sla, "tm_shift_pay": tm_shift_pay,
         "tm_units": tm_units, "tm_fx_dated": tm_fx_dated, "tm_tenure": tm_tenure, "tm_recurring": tm_recurring}
KIND_FAMILY = {k: FAM for k in KINDS}
