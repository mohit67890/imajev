"""rule_exception: a record (or several) + a short policy with thresholds, exceptions, exceptions to exceptions, two-step
conditions and tier tables -> eligible? which action? which window? which band? whose approval? which record?

Gold comes from `_eligible` / `_action` / `_window` / `_band` / `_approver` over structured params. `recheck(spec)` is a separate
re-implementation of the same rules used by the tests. Unknown children blank one decisive record field ("not recorded") and are
kept only when enumerating plausible values of that field yields at least two different answers.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from decimal import Decimal

from .common import (DOMAINS, DOMAIN_IDS, P, People, chance, choice_field, code, fd, money, noul_field, ordered_choice_field,
                     org_name, rand_date, score_field, item, cap)

FAMILY = "rule_exception"
KINDS = ["eligibility", "action", "applicable_window", "band_score", "approver", "which_record", "not_eligible"]

# indexes into DOMAINS[d]["outcomes"]: full, partial, deny, and two distractor outcomes
OUTMAP = {"retail": (0, 1, 3, 2, 4), "travel": (0, 1, 3, 2, 4)}
OUTMAP_DEFAULT = (0, 1, 2, 3, 4)
MISSING = None
NR = ["not recorded", "not recorded", "(blank)", "not provided", "—"]

# band scenario per domain: measure, unit, benefit noun, reducing condition, exception to the reduction
BAND = {
    "retail": ("delivery delay beyond the promised date", "days", "late-delivery credit",
               "the delay was caused by an address error in the customer's order", "the customer corrected the address before dispatch"),
    "logistics": ("late arrival at the consignee beyond the booked slot", "hours", "service rebate",
                  "the consignee's site was closed at the booked slot", "the carrier had been told about the closure in advance"),
    "hr": ("overtime worked in the pay period", "hours", "overtime premium",
           "the overtime was not approved in advance", "the overtime was worked during a declared site emergency"),
    "healthcare_admin": ("wait beyond the booked appointment time", "minutes", "goodwill credit",
                         "the patient checked in late", "the late check-in was caused by clinic-arranged transport"),
    "insurance": ("delay in settling the claim beyond the service standard", "days", "delay payment",
                  "the policyholder had not supplied documents that were requested", "the request for documents was sent to an out-of-date address"),
    "finance": ("service outage affecting the client", "hours", "outage compensation",
                "the client's own device or software caused the failure", "the failure followed an update pushed by the bank"),
    "saas_ops": ("unplanned downtime in the billing month", "minutes", "service credit",
                 "the downtime fell inside an announced maintenance window", "the maintenance overran its announced end time"),
    "travel": ("arrival delay at the final destination", "minutes", "delay compensation",
               "the delay was caused by extraordinary circumstances", "the passenger was not offered rerouting"),
    "public_sector": ("delay in issuing the decision beyond the statutory period", "days", "fee refund",
                      "the applicant did not supply information that was requested", "the information request was issued late"),
    "legal": ("overrun of the agreed completion date", "days", "fee reduction",
              "the client was late in providing instructions", "the late instructions followed a change of scope made by the firm"),
}
UNIT_STEPS = {"days": [1, 2, 3, 5, 7, 10, 14, 21, 28], "hours": [2, 3, 4, 6, 8, 12, 16, 24, 36], "minutes": [30, 45, 60, 90, 120, 180, 240, 360]}
BAND_PCTS = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]


# ================================================================================================ params
# which dom["flags"] indexes read naturally as (exception that re-admits an excluded category, eligibility requirement)
FLAG_ROLES = {"retail": ([0, 2], [1, 2]), "logistics": ([2, 0], [0, 1]), "hr": ([1, 0], [0, 2]), "healthcare_admin": ([1, 2], [0, 3]),
              "insurance": ([0, 1], [3, 1, 0]), "finance": ([1, 0], [3]), "saas_ops": ([2], [1]), "travel": ([1, 3], [3, 1]),
              "public_sector": ([0, 3], [3, 0]), "legal": ([1, 2], [0, 2])}
TIER_OVERRIDE = {"public_sector": ("applicant category", ["Category A", "Category B", "Category C", "Category D"])}


def plural(w: str) -> str:
    return w + "es" if w.endswith(("s", "x", "ch", "sh")) else w + "s"


def ev_date(dom) -> str:
    """'delivery date' / 'invoice date' / 'date of loss' (no 'date date')."""
    ev = dom["event"]
    return ev if "date" in ev else f"{ev} date"


def _dom(rng):
    d = rng.choice(DOMAIN_IDS)
    dom = dict(DOMAINS[d])
    if d in TIER_OVERRIDE:
        dom["tier_name"], dom["tiers"] = TIER_OVERRIDE[d]
    if d == "hr":
        dom["amount_noun"] = "estimated cost"
    dom["_id"] = d
    return d, dom


def _policy(rng, dom, diff) -> dict:
    W = rng.choice([10, 14, 21, 28, 30, 45, 60, 90])
    cats = list(dom["categories"])
    rng.shuffle(cats)
    exc_ok, req_ok = FLAG_ROLES[dom["_id"]]
    f_exc = rng.choice(exc_ok)
    f_req = rng.choice([x for x in req_ok if x != f_exc] or [x for x in range(4) if x != f_exc])
    p = {"W": W, "cmp": rng.choice(["le", "lt"]), "t_star": rng.randint(1, 3), "E": rng.choice([7, 10, 14, 15, 30]),
         "excl": cats[0], "other": cats[1], "f_exc": f_exc, "f_req": f_req if diff >= 4 else None,
         "S": rng.choice([25, 40, 50, 75, 100, 150]) if diff >= 4 else None,
         "Wf": rng.choice([x for x in (3, 5, 7, 10, 14, 20) if x != W]) if diff >= 5 else None,
         "T": rng.choice([250, 300, 400, 500, 750, 1000, 1500]), "Wfull": rng.choice([x for x in (3, 5, 7, 10, 14) if x < W]),
         "cats": cats[2:]}
    return p


def _within(days, window, cmp):
    return days <= window if cmp == "le" else days < window


def _window(p, r):
    """The time limit that applies, or None when the category is excluded outright."""
    if r["cat"] == p["excl"]:
        if not r["flags"][p["f_exc"]]:
            return None
        if p["Wf"] is not None:
            return p["Wf"]
    return p["W"] + (p["E"] if r["tier"] >= p["t_star"] else 0)


def _eligible(p, r) -> bool:
    w = _window(p, r)
    if w is None:
        return False
    if not _within(r["days"], w, p["cmp"]):
        return False
    if p["f_req"] is not None and not r["flags"][p["f_req"]]:
        if not (p["S"] is not None and r["amount"] <= p["S"]):
            return False
    return True


def _naive(p, r) -> bool:
    """What a reader who applies only the headline rule would say."""
    return r["cat"] != p["excl"] and _within(r["days"], p["W"], p["cmp"])


def _action(p, r) -> str:
    if not _eligible(p, r):
        return "deny"
    if r["amount"] > p["T"]:
        return "refer"
    if r["days"] > p["Wfull"]:
        return "partial"
    return "full"


# ================================================================================================ records
def _rand_record(rng, p, allowed_cats, people, dom, ref_prefix) -> dict:
    marks = [p["W"], p["W"] + p["E"], p["Wfull"]] + ([p["Wf"]] if p["Wf"] else [])
    if chance(rng, 0.75):
        days = max(0, rng.choice(marks) + rng.choice([-3, -1, 0, 0, 1, 1, 2, 4]))
    else:
        days = rng.randint(0, p["W"] + p["E"] + 20)
    amt_marks = [x for x in (p["S"], p["T"]) if x]
    if chance(rng, 0.5) and amt_marks:
        base = Decimal(rng.choice(amt_marks))
        amount = base + Decimal(rng.choice([-15, -5, -1, 0, 0, 1, 5, 20, 60])) + Decimal(rng.choice([0, 0, 50, 99])) / 100
        amount = max(Decimal("5"), amount)
    else:
        amount = Decimal(rng.randint(8, 2400)) + Decimal(rng.choice([0, 0, 25, 50, 99])) / 100
    return {"name": people(), "ref": code(rng, ref_prefix), "tier": rng.randrange(4),
            "cat": p["excl"] if chance(rng, 0.35) else rng.choice(allowed_cats),
            "amount": amount, "days": days, "flags": [chance(rng, 0.55) for _ in range(4)]}


def _sample(rng, p, dom, people, pred, prefer=None, tries=400):
    best = None
    for _ in range(tries):
        r = _rand_record(rng, p, p["cats"], people, dom, dom["ref"])
        if not pred(r):
            continue
        if prefer is None or prefer(r):
            return r
        best = best or r
    if best is None:
        raise ValueError("no record satisfies the target")
    return best


# ================================================================================================ rendering
def _cmp_phrase(rng, p, W, event):
    if p["cmp"] == "le":
        return P(rng, f"within {W} days of the {event}", f"no later than {W} days after the {event}", f"{W} days or fewer after the {event}",
                 f"on or before day {W} after the {event}")
    return P(rng, f"fewer than {W} days after the {event}", f"less than {W} days after the {event}", f"before day {W} after the {event}")


def _policy_text(rng, p, dom, org, kind, eff) -> tuple[str, dict]:
    """Numbered clauses. Returns (text, {tag: clause label})."""
    ev, req, reqs = dom["event"], dom["request"], dom["requests"]
    tiers, tn = dom["tiers"], dom["tier_name"]
    flags = dom["flags"]
    clauses: list[tuple[str, str]] = []
    style = rng.randrange(4)

    def lab(i):
        return [f"{i}", f"4.{i}", f"R{i}", f"§{i}"][style]

    clauses.append(("count", P(rng, f"Days are calendar days; the {ev_date(dom)} itself is day 0.",
                                f"All periods below are counted in calendar days starting with the day after the {ev}.",
                                f"For every period in this section, day 1 is the calendar day after the {ev}.",
                                f"Count calendar days from the {ev}: the day after it is day 1.")))
    clauses.append(("tiers", P(rng, f"{cap(plural(tn))}, from lowest to highest: {', '.join(tiers)}.",
                                f"The {plural(tn)} rank in this order (lowest first): {', '.join(tiers)}.",
                                f"Ranking of {plural(tn)}, lowest to highest: {' < '.join(tiers)}.")))
    base = P(rng, "A {req} is eligible only if it is made {w}.", "{Reqs} are accepted when they are made {w}; later {reqs} are not eligible.",
             "To be eligible, a {req} must be received {w}.", "The general rule: a {req} qualifies only if it arrives {w}.")
    clauses.append(("base", base.format(req=req, reqs=reqs, Reqs=cap(reqs), w=_cmp_phrase(rng, p, p["W"], ev))))
    clauses.append(("ext", P(rng, "Where the {tn} is {t} or higher, the period in clause {base} is extended by {E} days.",
                             "Holders of {tn} {t} and above get {E} extra days on top of the period in clause {base}.",
                             "Clause {base} is lengthened by {E} days for anyone whose {tn} is {t} or above.",
                             "An additional {E} days applies to the clause {base} period when the {tn} is at least {t}.")
                     .format(tn=tn, t=tiers[p["t_star"]], E=p["E"], base="{base}")))
    exc_tail = (P(rng, " In that case the {req} must be made {w} instead, and no {tn} extension applies.",
                  " Such {reqs} then have their own limit: they must be made {w}, with no extension for {tn}.")
                .format(req=req, reqs=reqs, tn=tn, w=_cmp_phrase(rng, p, p["Wf"], ev)) if p["Wf"] else
                P(rng, " In that case the ordinary rules in this section apply.", " Such {reqs} are then handled under the ordinary rules.").format(reqs=reqs))
    clauses.append(("excl", P(rng, "Cases concerning {cat} are excluded, and no {req} about them is eligible, unless {f}.",
                              "No {req} is eligible where it concerns {cat}. This exclusion does not apply where {f}.",
                              "{Cat}: excluded from this section, except where {f}.",
                              "{Reqs} concerning {cat} are not accepted; the only exception is where {f}.")
                      .format(cat=p["excl"], Cat=cap(p["excl"]), req=req, reqs=reqs, Reqs=cap(reqs), f=flags[p["f_exc"]][0]) + exc_tail))
    if p["f_req"] is not None:
        waiver = (P(rng, " This requirement is waived when the {amt} is {S} or less.", " It does not apply if the {amt} does not exceed {S}.",
                    " (Waived for a {amt} of at most {S}.)")
                  .format(amt=dom["amount_noun"], S="{S}"))
        clauses.append(("req", P(rng, "A {req} is eligible only if {f}.", "In addition, {f} is a condition of eligibility.",
                                 "Eligibility further requires that {f}.")
                        .format(req=req, f=flags[p["f_req"]][0]) + waiver))
    if kind in ("action",):
        clauses.append(("refer", P(rng, "An eligible {req} whose {amt} is above {T} is not settled by the handler: {o}.",
                                   "Step two: once a {req} is found eligible, check the {amt}; if it exceeds {T}, {o}.",
                                   "Where an eligible {req} has a {amt} greater than {T}, {o}.")
                        .format(req=req, amt=dom["amount_noun"], T="{T}", o="{refer}")))
        clauses.append(("partial", P(rng, "Other eligible {reqs} made more than {Wf} days after the {ev} are settled on reduced terms: {o}.",
                                     "If an eligible {req} (not referred under clause {refer_lab}) is made after day {Wf}, {o}.",
                                     "Eligible {reqs} not referred and made later than day {Wf} after the {ev}: {o}.")
                        .format(req=req, reqs=reqs, ev=ev, Wf=p["Wfull"], o="{partial}", refer_lab="{refer_lab}")))
        clauses.append(("full", P(rng, "All remaining eligible {reqs}: {o}.", "In every other eligible case, {o}.",
                                  "Any other eligible {req}: {o}.").format(req=req, reqs=reqs, o="{full}")))
        clauses.append(("deny", P(rng, "A {req} that is not eligible: {o}.", "Where a {req} fails any eligibility condition, {o}.",
                                  "Ineligible {reqs}: {o}.").format(req=req, reqs=reqs, o="{deny}")))
    # distractor clause about a category that never appears on the records
    clauses.append(("other", P(rng, "Cases concerning {cat} follow a separate procedure run by {dept} and are outside this section.",
                               "This section does not cover {cat}; see the {dept} guidance instead.",
                               "For {req}s about {cat}, a shorter period of {x} days applies, and the {tn} extension does not.")
                    .format(cat=p["other"], dept=rng.choice(dom["depts"]), x=rng.choice([5, 7, 10]), tn=tn, req=req)))
    clauses[-1] = (clauses[-1][0], cap(clauses[-1][1]))
    body = [c for c in clauses if c[0] not in ("count", "tiers")]
    rng.shuffle(body)
    # keep base before ext and exclusion order varied; count/tiers first or last
    head = [c for c in clauses if c[0] in ("count", "tiers")]
    ordered = head + body if chance(rng, 0.6) else body + head
    labels = {tag: lab(i + 1) for i, (tag, _) in enumerate(ordered)}
    title = P(rng, f"{org} — {rng.choice(dom['doc_nouns'])} (extract)", f"{org}: {rng.choice(dom['doc_nouns'])}, section 4",
              f"{rng.choice(dom['doc_nouns'])} — {org}", f"{org} · {rng.choice(dom['doc_nouns'])} (in force from {eff})")
    lines = [title, ""]
    for tag, txt in ordered:
        lines.append(f"{labels[tag]}. {txt}")
    return "\n".join(lines), labels


def _fmt_rec_fields(rng, p, dom, r, style, missing: set, sym, extras):
    tiers = dom["tiers"]
    flags = dom["flags"]
    nr = rng.choice(NR)
    ev_d = dt.date.fromisoformat(r["event"])
    req_date = ev_d + dt.timedelta(days=r["days"]) if r["days"] is not None else None
    use_days = r.get("_show_days", False)
    f = [(P(rng, "Reference", "Case ref", "Ref"), r["ref"]),
         (P(rng, "Name", cap(dom["party"]), "Submitted by"), r["name"]),
         (cap(dom["tier_name"]), nr if "tier" in missing else tiers[r["tier"]]),
         (P(rng, "Category", "Relates to", "Product/service line"), nr if "cat" in missing else r["cat"]),
         (cap(dom["amount_noun"]), nr if "amount" in missing else money(r["amount"], sym))]
    if use_days:
        f.append((P(rng, f"Days since {dom['event']}", f"Days after {dom['event']}"), nr if "days" in missing else str(r["days"])))
    else:
        f.append((cap(ev_date(dom)), fd(ev_d, style)))
        f.append((P(rng, "Received", "Date submitted", "Request date"), nr if "days" in missing else fd(req_date, style)))
    for i in range(4):
        f.append((cap(flags[i][1]), nr if f"flag{i}" in missing else ("Yes" if r["flags"][i] else "No")))
    f.extend(extras)
    return f


def _extras(rng, dom, people):
    ex = [(P(rng, "Channel", "Received via"), rng.choice(["web form", "phone", "email", "in person", "post"])),
          (P(rng, "Handler", "Assigned to"), people()),
          (P(rng, "Region", "Site"), rng.choice(["North", "South", "East", "West", "Central", "Coastal"]))]
    rng.shuffle(ex)
    return ex[: rng.randint(1, 3)]


def _render_single(rng, p, dom, r, missing, sym, fmt, people, style):
    fields = _fmt_rec_fields(rng, p, dom, r, style, missing, sym, _extras(rng, dom, people))
    if fmt == "dict":
        return {k: v for k, v in fields}
    if fmt == "table":
        return "| Field | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in fields)
    if fmt == "form":
        return "\n".join(f"{k}: {v}" for k, v in fields)
    # memo prose
    nr = rng.choice(NR)
    tiers, flags = dom["tiers"], dom["flags"]
    ev = dt.date.fromisoformat(r["event"])
    tier = "an unrecorded " + dom["tier_name"] if "tier" in missing else f"{dom['tier_name']} {tiers[r['tier']]}"
    cat = f"a category that was {nr}" if "cat" in missing else r["cat"]
    amt = f"{dom['amount_noun']} {nr}" if "amount" in missing else f"{dom['amount_noun']} of {money(r['amount'], sym)}"
    if r.get("_show_days"):
        when = (f"the number of days since the {dom['event']} is {nr}" if "days" in missing else
                f"it was made {r['days']} days after the {dom['event']}")
    else:
        when = (f"the {dom['event']} was on {fd(ev, style)} and the date the {dom['request']} was received is {nr}" if "days" in missing else
                f"the {dom['event']} was on {fd(ev, style)} and the {dom['request']} arrived on {fd(ev + dt.timedelta(days=r['days']), style)}")
    checks = "; ".join(f"{flags[i][1]}: " + (nr if f"flag{i}" in missing else ("yes" if r["flags"][i] else "no")) for i in range(4))
    intro = P(rng, "File note. ", "Handler's summary: ", "Case note — ")
    return (f"{intro}{r['name']} ({tier}) raised {r['ref']}, a {dom['request']} concerning {cat}, with a {amt}. "
            f"{cap(when)}. Checklist — {checks}.")


def _render_multi(rng, p, dom, recs, missing_map, sym, fmt, style):
    rows = []
    for r in recs:
        fields = _fmt_rec_fields(random.Random(0), p, dom, r, style, missing_map.get(r["ref"], set()), sym, [])
        rows.append(fields)
    # stable labels across rows: rebuild with fixed label choices
    heads = [k for k, _ in rows[0]]
    if fmt == "dict":
        return [{k: v for k, v in fs} for fs in rows]
    if fmt in ("table", "memo"):
        out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
        for fs in rows:
            out.append("| " + " | ".join(str(v) for _, v in fs) + " |")
        return "\n".join(out)
    return "\n\n".join("\n".join(f"{k}: {v}" for k, v in fs) for fs in rows)


# ================================================================================================ unknown support
def _vals(p, field, dom):
    if field == "tier":
        return list(range(4))
    if field == "cat":
        return [p["excl"]] + p["cats"]
    if field == "days":
        s = {0, 200}
        for m in (p["W"], p["W"] + p["E"], p["Wfull"], p["Wf"] or 0):
            s |= {max(0, m - 1), m, m + 1}
        return sorted(s)
    if field == "amount":
        s = {Decimal("5"), Decimal("5000")}
        for m in (p["S"], p["T"]):
            if m:
                s |= {Decimal(m) - 1, Decimal(m), Decimal(m) + 1}
        return sorted(s)
    if field.startswith("flag"):
        return [False, True]
    raise KeyError(field)


def _set(r, field, v):
    r = dict(r)
    r["flags"] = list(r["flags"])
    if field.startswith("flag"):
        r["flags"][int(field[4:])] = v
    else:
        r[field] = v
    return r


FIELDS = ["days", "tier", "cat", "amount", "flag0", "flag1", "flag2", "flag3"]


def _blank_one(rng, r, fn, valsfn, fields=FIELDS):
    """Choose a field of r whose plausible values give >= 2 outcomes under fn; raises when none does."""
    fs = list(fields)
    rng.shuffle(fs)
    for f in fs:
        if len({repr(fn(_set(r, f, v))) for v in valsfn(f)}) >= 2:
            return f
    raise ValueError("no decisive field")


def _spec_rec(r, missing=()):
    out = {"ref": r["ref"], "tier": None if "tier" in missing else r["tier"], "cat": None if "cat" in missing else r["cat"],
           "amount": None if "amount" in missing else str(r["amount"]), "days": None if "days" in missing else r["days"],
           "flags": [None if f"flag{i}" in missing else r["flags"][i] for i in range(4)]}
    return out


def _spec_policy(p):
    q = {k: v for k, v in p.items() if k != "cats"}
    q["cats"] = p["cats"]
    return q


# ================================================================================================ generate
def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    d_id, dom = _dom(rng)
    p = _policy(rng, dom, difficulty)
    people = People(rng)
    sym = rng.choice(["€", "£", "$", "A$", "C$", "S$"])
    org = org_name(rng, dom)
    style = rng.randrange(5)
    eff = fd(rand_date(rng, dt.date(2024, 1, 1), dt.date(2025, 6, 1)), style)
    fmt = rng.choice(["dict", "table", "form", "memo"])
    show_days = chance(rng, 0.3)
    base_date = rand_date(rng)
    hard = {3: 0.45, 4: 0.7, 5: 0.85}[difficulty]

    def fin(r):
        r["event"] = (base_date - dt.timedelta(days=rng.randint(0, 40))).isoformat()
        r["_show_days"] = show_days
        return r

    if kind in ("which_record", "not_eligible"):
        return _gen_multi(rng, difficulty, kind, want_unknown, d_id, dom, p, people, sym, org, style, eff, fmt, fin, hard)
    if kind == "band_score":
        return _gen_band(rng, difficulty, want_unknown, d_id, dom, people, sym, org, style, eff, fmt, show_days, base_date)
    if kind == "approver":
        return _gen_approver(rng, difficulty, want_unknown, d_id, dom, people, sym, org, style, eff, fmt, base_date)

    if kind == "applicable_window":
        p["cmp"] = "le"
    ptext, labels = _policy_text(rng, p, dom, org, kind, eff)
    ptext = ptext.replace("{base}", labels["base"]).replace("{S}", money(p["S"] or 0, sym)).replace("{T}", money(p["T"], sym))
    ref_lab = labels.get("refer", "")
    ptext = ptext.replace("{refer_lab}", ref_lab)

    if kind == "eligibility":
        target = chance(rng, 0.5)
        r = fin(_sample(rng, p, dom, people, lambda r: _eligible(p, r) == target, prefer=lambda r: chance(rng, 1 - hard) or _naive(p, r) != target))
        fn = lambda rr: _eligible(p, rr)  # noqa: E731
        gold = _eligible(p, r)
        q = P(rng, "Under the policy, is {ref} eligible?", "Does {name}'s {req} ({ref}) meet the eligibility rules above?",
              "Applying every clause of the extract, is {ref} an eligible {req}?", "Should {ref} be treated as eligible under this section?",
              "Is the {req} {ref} eligible under the rules quoted?").format(ref=r["ref"], name=r["name"], req=dom["request"])
        negq = P(rng, "Under the policy, is {ref} ineligible?", "Should {ref} be rejected as ineligible under this section?",
                 "Does {ref} fail at least one eligibility condition?").format(ref=r["ref"])
        field = noul_field(q)
        hints = {"neg": {"question": negq, "gold": not gold}}
    elif kind == "action":
        o = dom["outcomes"]
        mi = OUTMAP.get(d_id, OUTMAP_DEFAULT)
        roles = dom["roles"][1:]
        role = rng.choice(roles)
        texts = {"full": o[mi[0]], "partial": o[mi[1]], "deny": o[mi[2]], "refer": f"refer the {dom['request']} to the {role} for approval"}
        for k2, v in texts.items():
            ptext = ptext.replace("{" + k2 + "}", v)
        target = rng.choice(["full", "partial", "deny", "refer", "deny", "refer"])
        r = fin(_sample(rng, p, dom, people, lambda r: _action(p, r) == target,
                        prefer=lambda r: chance(rng, 1 - hard) or (_naive(p, r) != _eligible(p, r)) or r["days"] in (p["Wfull"], p["Wfull"] + 1)))
        fn = lambda rr: _action(p, rr)  # noqa: E731
        gold_t = texts[_action(p, r)]
        others = [role2 for role2 in roles if role2 != role]
        near = [f"refer the {dom['request']} to the {rng.choice(others)} for approval"]
        wrong = [texts[k2] for k2 in texts if texts[k2] != gold_t] + [o[mi[3]], o[mi[4]]]
        if difficulty >= 5:
            wrong.append(near[0])
            near = [f"refer the {dom['request']} to the {x} for approval" for x in others if x not in near[0]][:1] or near
        q = P(rng, "Which action does the policy require for {ref}?", "What must the handler do with {ref}?",
              "Under these rules, what is the correct outcome for {ref}?", "Which outcome follows from the policy for {ref}?").format(ref=r["ref"])
        field, gold = choice_field(rng, q, cap(gold_t), [cap(w) for w in wrong])
        hints = {"near_miss": [cap(x) for x in near]}
    else:  # applicable_window
        ev = dom["event"]
        excl_txt = f"No time limit applies: {dom['requests']} about this category are excluded"
        def wtxt(n):
            return f"Within {n} calendar days of the {ev}" if n is not None else excl_txt
        target = rng.choices(["base", "ext", "none", "wf"], weights=[2, 2, 1, 1.5 if p["Wf"] else 0])[0]

        def tw(r):
            if _window(p, r) is None:
                return "none"
            if r["cat"] == p["excl"] and p["Wf"]:
                return "wf"
            return "ext" if r["tier"] >= p["t_star"] else "base"
        r = fin(_sample(rng, p, dom, people, lambda r: tw(r) == target))
        fn = lambda rr: _window(p, rr)  # noqa: E731
        w = _window(p, r)
        cand = {p["W"], p["W"] + p["E"], p["W"] + 2 * p["E"], p["W"] - p["E"] if p["W"] > p["E"] else p["W"] + 1}
        if p["Wf"]:
            cand.add(p["Wf"])
        wrong = [wtxt(n) for n in sorted(cand) if n != w and n > 0] + ([excl_txt] if w is not None else [])
        q = P(rng, "Which time limit applies to {ref}?", "What period does {name} have to make the {req} ({ref})?",
              "Under the rules above, which deadline governs {ref}?", "Which window applies to the {req} {ref}?").format(
            ref=r["ref"], name=r["name"], req=dom["request"])
        field, gold = choice_field(rng, q, wtxt(w), wrong, k=6)
        near = [f"Within {w if w is not None else p['W']} business days of the {ev}"]
        if w is not None:
            near.append(f"Within {w + 1} calendar days of the {ev}")
        hints = {"near_miss": near}
    missing = set()
    rec_state = _render_single(rng, p, dom, r, missing, sym, fmt, people, style)
    state = _state(rng, ptext, rec_state, fmt, dom)
    spec = {"kind": kind, "policy": _spec_policy(p), "record": _spec_rec(r)}
    if kind == "action":
        spec["texts"] = {k: cap(v) for k, v in texts.items()}
    if kind == "applicable_window":
        spec["event"] = dom["event"]
        spec["excl_txt"] = excl_txt
    hints["decisive"] = [s for s in _decisive_strings(r, dom) if _in(state, s)]
    gold_val = gold
    items = [item(state, field, gold_val, kind, d_id, spec, hints)]
    if want_unknown:
        f = _blank_one(rng, r, fn, lambda f: _vals(p, f, dom))
        missing = {f}
        rec_state2 = _render_single(random.Random(rng.random()), p, dom, r, missing, sym, fmt, People(random.Random(1)), style)
        state2 = _state(random.Random(0), ptext, rec_state2, fmt, dom)
        spec2 = dict(spec)
        spec2["record"] = _spec_rec(r, missing)
        items.append(item(state2, dict(field), None, kind, d_id, spec2, {}, child_of=0))
    return items


def _in(state, s):
    return s in (json.dumps(state, ensure_ascii=False) if isinstance(state, dict) else state)


def _decisive_strings(r, dom):
    return [r["ref"], r["cat"]]


def _state(rng, ptext, rec_state, fmt, dom):
    if fmt == "dict":
        key = P(rng, "records", "cases") if isinstance(rec_state, list) else P(rng, "record", "case", "request")
        return {"policy": ptext, key: rec_state}
    head = P(rng, "CASE RECORD", "Record under review", "The case", f"{cap(dom['request'])} details")
    return f"{ptext}\n\n{head}\n{rec_state}"


# ------------------------------------------------------------------------------------------------ several records
def _gen_multi(rng, difficulty, kind, want_unknown, d_id, dom, p, people, sym, org, style, eff, fmt, fin, hard):
    n = {3: 4, 4: 5, 5: rng.choice([6, 7])}[difficulty]
    one_true = kind == "which_record"
    special = fin(_sample(rng, p, dom, people, lambda r: _eligible(p, r) == one_true,
                          prefer=lambda r: chance(rng, 1 - hard) or _naive(p, r) != one_true))
    rest = []
    for _ in range(n - 1):
        rest.append(fin(_sample(rng, p, dom, people, lambda r: _eligible(p, r) != one_true,
                                prefer=lambda r: chance(rng, 1 - hard) or _naive(p, r) == one_true)))
    recs = rest + [special]
    rng.shuffle(recs)
    refs = [r["ref"] for r in recs]
    if len(set(refs)) != len(refs):
        raise ValueError("dup refs")
    gi = recs.index(special)
    ptext, labels = _policy_text(rng, p, dom, org, kind, eff)
    ptext = ptext.replace("{base}", labels["base"]).replace("{S}", money(p["S"] or 0, sym)).replace("{T}", money(p["T"], sym))
    texts = [f"{r['ref']} ({r['name']})" for r in recs]
    if one_true:
        q = P(rng, "Exactly one of the {reqs} below is eligible. Which one?", "Which of these {reqs} meets every eligibility condition?",
              "Only one listed {req} qualifies under the policy. Which is it?", "Which {req} in the list is eligible?")
    else:
        q = P(rng, "Exactly one of the {reqs} below is NOT eligible. Which one?", "Which of these {reqs} fails the eligibility rules?",
              "All but one listed {req} qualify under the policy. Which one does not?", "Which {req} in the list is ineligible?")
    q = q.format(reqs=dom["requests"], req=dom["request"])
    field, gold = ordered_choice_field(q, texts, gi)
    rec_state = _render_multi(rng, p, dom, recs, {}, sym, fmt, style)
    state = _state(rng, ptext, rec_state, fmt, dom)
    spec = {"kind": kind, "policy": _spec_policy(p), "records": [_spec_rec(r) for r in recs], "target": one_true}
    hints = {"decisive": [special["ref"]]}
    items = [item(state, field, gold, kind, d_id, spec, hints)]
    if want_unknown:
        # blank a field of the special record that could flip it AND a field of another record that could flip that one
        def uniq(rs):
            hits = [x["ref"] for x in rs if _eligible(p, x) == one_true]
            return hits[0] if len(hits) == 1 else "none"
        fs = list(FIELDS)
        rng.shuffle(fs)
        f1 = next((f for f in fs if len({_eligible(p, _set(special, f, v)) for v in _vals(p, f, dom)}) >= 2), None)
        if f1 is None:
            raise ValueError("no decisive field")
        other = None
        for o in rng.sample(recs, len(recs)):
            if o is special:
                continue
            f2 = next((f for f in fs if len({_eligible(p, _set(o, f, v)) for v in _vals(p, f, dom)}) >= 2), None)
            if f2:
                other = (o, f2)
                break
        if other is None:
            raise ValueError("no second record")
        o, f2 = other
        answers = set()
        for v1 in _vals(p, f1, dom):
            for v2 in _vals(p, f2, dom):
                rs = [(_set(x, f1, v1) if x is special else _set(x, f2, v2) if x is o else x) for x in recs]
                a = uniq(rs)
                if a != "none":
                    answers.add(a)
        if len(answers) < 2:
            raise ValueError("unknown not undetermined")
        mm = {special["ref"]: {f1}, o["ref"]: {f2}}
        rec_state2 = _render_multi(rng, p, dom, recs, mm, sym, fmt, style)
        state2 = _state(random.Random(0), ptext, rec_state2, fmt, dom)
        spec2 = dict(spec)
        spec2["records"] = [_spec_rec(r, mm.get(r["ref"], ())) for r in recs]
        items.append(item(state2, dict(field), None, kind, d_id, spec2, {}, child_of=0))
    return items


# ------------------------------------------------------------------------------------------------ band score
def _band(b, r):
    base = sum(1 for t in b["thr"] if r["q"] >= t)
    top = len(b["thr"])
    if b["uplift"] and r["tier"] >= b["t_star"]:
        base = min(top, base + 1)
    if b["reduce"] and r["red"]:
        base = max(0, base - 1) if (b["reduce_exc"] and r["exc"]) else 0
    if b["cap_cat"] is not None and r["cat"] == b["cap_cat"]:
        base = min(base, b["cap"])
    return base


def _gen_band(rng, difficulty, want_unknown, d_id, dom, people, sym, org, style, eff, fmt, show_days, base_date):
    measure, unit, benefit, red_t, exc_t = BAND[d_id]
    nb = rng.randint(4, 6)
    steps = sorted(rng.sample(UNIT_STEPS[unit], nb - 1))
    pcts = sorted(rng.sample(BAND_PCTS, nb - 1))
    cats = list(dom["categories"])
    rng.shuffle(cats)
    mods = ["uplift", "reduce", "cap"]
    rng.shuffle(mods)
    on = set(mods[: {3: 1, 4: 2, 5: 3}[difficulty]])
    b = {"thr": steps, "uplift": "uplift" in on, "t_star": rng.randint(1, 3), "reduce": "reduce" in on, "reduce_exc": "reduce" in on and difficulty >= 4,
         "cap_cat": cats[0] if "cap" in on else None, "cap": rng.randint(1, nb - 2), "cats": cats[:5]}

    def rand_r():
        if chance(rng, 0.7):
            q = max(0, rng.choice(steps) + rng.choice([-1, 0, 0, 1, 2]) * (1 if unit != "minutes" else 5))
        else:
            q = rng.randint(0, steps[-1] + steps[0] * 2)
        return {"ref": code(rng, dom["ref"]), "name": people(), "q": q, "tier": rng.randrange(4),
                "cat": b["cap_cat"] if (b["cap_cat"] and chance(rng, 0.4)) else rng.choice(cats[1:5]),
                "red": chance(rng, 0.45) if b["reduce"] else False, "exc": chance(rng, 0.5) if b["reduce_exc"] else False}

    def naive(r):
        return sum(1 for t in b["thr"] if r["q"] >= t)
    r = None
    for _ in range(300):
        c = rand_r()
        if naive(c) != _band(b, c) or chance(rng, 0.25 if difficulty == 3 else 0.1):
            r = c
            break
    if r is None:
        raise ValueError("no band record")
    gold = _band(b, r)
    # policy text
    tn, tiers = dom["tier_name"], dom["tiers"]
    lines = [P(rng, f"{org} — {cap(benefit)} rules", f"{org}: how the {benefit} is calculated", f"{cap(benefit)} scheme ({org})"), ""]
    lines.append(P(rng, f"Step 1. Find the base band from the {measure} using the table (a row applies when the {measure} is at least its lower bound).",
                   f"Step 1. Use the table: the base band is the highest row whose 'at least' value the {measure} reaches.",
                   f"Step 1. Read the base band off the table below; each band starts at the {measure} shown (inclusive)."))
    lines.append("")
    lines.append(f"| Band | {cap(measure)} ({unit}) | {cap(benefit)} |")
    lines.append("|---|---|---|")
    lines.append(f"| 0 | under {steps[0]} | none |")
    for i, t in enumerate(steps):
        lines.append(f"| {i + 1} | at least {t} | {pcts[i]}% of the {dom['amount_noun']} |")
    lines.append("")
    k = 2
    if b["uplift"]:
        lines.append(P(rng, f"Step {k}. If the {tn} is {tiers[b['t_star']]} or higher (order: {', '.join(tiers)}), move up one band (never above Band {nb - 1}).",
                       f"Step {k}. Ranked {plural(tn)}, lowest first: {', '.join(tiers)}. At {tiers[b['t_star']]} or above, add one band, to a maximum of Band {nb - 1}."))
        k += 1
    if b["reduce"]:
        s = P(rng, f"Step {k}. If {red_t}, the band becomes 0", f"Step {k}. Where {red_t}, no {benefit} is due (Band 0)")
        if b["reduce_exc"]:
            s += P(rng, f"; however, if {exc_t}, the band is only reduced by one instead (not below 0).",
                   f" — unless {exc_t}, in which case reduce the band by one step only (minimum Band 0).")
        else:
            s += "."
        lines.append(s)
        k += 1
    if b["cap_cat"]:
        lines.append(P(rng, f"Step {k}. For {b['cap_cat']}, the band cannot exceed Band {b['cap']}.",
                       f"Step {k}. Cases about {b['cap_cat']} are capped at Band {b['cap']}."))
        k += 1
    lines.append(P(rng, "Apply the steps in the order listed.", "The steps are applied strictly in the order above.", "Always work through the steps in numerical order."))
    ptext = "\n".join(lines)
    nr = rng.choice(NR)

    def rec_fields(missing):
        f = [(P(random.Random(1), "Reference", "Ref"), r["ref"]), ("Name", r["name"]),
             (f"{cap(measure)} ({unit})", nr if "q" in missing else str(r["q"])), (cap(tn), nr if "tier" in missing else tiers[r["tier"]]),
             ("Category", nr if "cat" in missing else r["cat"])]
        if b["reduce"]:
            f.append((cap(red_t) + "?", nr if "red" in missing else ("Yes" if r["red"] else "No")))
        if b["reduce_exc"]:
            f.append((cap(exc_t) + "?", nr if "exc" in missing else ("Yes" if r["exc"] else "No")))
        return f

    def render(missing):
        f = rec_fields(missing)
        if fmt == "dict":
            return {"policy": ptext, "case": {k2: v for k2, v in f}}
        body = "\n".join(f"{k2}: {v}" for k2, v in f) if fmt != "table" else "| Field | Value |\n|---|---|\n" + "\n".join(f"| {k2} | {v} |" for k2, v in f)
        return f"{ptext}\n\nCase\n{body}"
    levels = ["Band 0: no " + benefit] + [f"Band {i + 1}: {pcts[i]}% of the {dom['amount_noun']}" for i in range(nb - 1)]
    q = P(rng, "Which band applies to {ref}?", "What {benefit} band does {ref} fall into after all steps?",
          "After applying every step, which band is {name}'s case ({ref}) in?", "Which band should be paid for {ref}?").format(
        ref=r["ref"], benefit=benefit, name=r["name"])
    field = score_field(q, levels)
    spec = {"kind": "band_score", "band": {k2: v for k2, v in b.items() if k2 != "cats"}, "record": {k2: r[k2] for k2 in ("q", "tier", "cat", "red", "exc")}}
    state = render(set())
    items = [item(state, field, gold, "band_score", d_id, spec, {"decisive": [r["ref"]]})]
    if want_unknown:
        cands = ["q", "tier", "cat"] + (["red"] if b["reduce"] else []) + (["exc"] if b["reduce_exc"] else [])
        rng.shuffle(cands)
        vals = {"q": sorted({0, steps[-1] * 3} | {t + dd for t in steps for dd in (-1, 0)}), "tier": [0, 1, 2, 3], "cat": b["cats"],
                "red": [False, True], "exc": [False, True]}
        for f in cands:
            if len({_band(b, {**r, f: v}) for v in vals[f]}) >= 2:
                spec2 = {"kind": "band_score", "band": spec["band"], "record": {**spec["record"], f: None}}
                items.append(item(render({f}), dict(field), None, "band_score", d_id, spec2, {}, child_of=0))
                break
        else:
            raise ValueError("no decisive band field")
    return items


# ------------------------------------------------------------------------------------------------ approver
def _approver(a, r):
    total = r["amount"] + sum(pa for pd, pa in r["priors"] if pd is not None and 0 <= pd <= a["N"])
    lvl = sum(1 for t in a["thr"] if total > t)
    if a["tier_rel"] and r["tier"] >= a["t_star"]:
        lvl = max(0, lvl - 1)
    if a["ov_cat"] is not None and r["cat"] == a["ov_cat"]:
        lvl = max(lvl, a["ov_lvl"])
    return lvl


def _gen_approver(rng, difficulty, want_unknown, d_id, dom, people, sym, org, style, eff, fmt, base_date):
    roles = dom["roles"]
    ladder = [roles[0]] + rng.sample(roles[1:5], 3)
    ladder = [ladder[0]] + sorted(ladder[1:], key=roles.index)
    base = rng.choice([100, 200, 250, 500, 1000])
    thr = [base, base * rng.choice([2, 3, 4, 5]), 0]
    thr[2] = thr[1] * rng.choice([2, 3, 4])
    cats = list(dom["categories"])
    rng.shuffle(cats)
    a = {"thr": thr, "N": rng.choice([7, 14, 30, 31, 60]), "tier_rel": difficulty >= 5, "t_star": rng.randint(2, 3),
         "ov_cat": cats[0] if difficulty >= 4 else None, "ov_lvl": rng.randint(2, 3), "cats": cats[:5]}

    def rand_r():
        tgt = rng.choice(thr)
        amt = Decimal(max(10, tgt + rng.choice([-1, 0, 1, -40, 30]) * rng.choice([1, 1, 2]) - rng.randint(0, tgt // 2 if chance(rng, 0.5) else 0)))
        priors = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            pd = a["N"] + rng.choice([-2, -1, 0, 1, 2, 5]) if chance(rng, 0.7) else rng.randint(0, a["N"] * 2)
            pa = Decimal(rng.choice([base // 2, base, base * 2, int(tgt * 0.3) + 1]))
            priors.append((max(1, pd), pa))
        return {"ref": code(rng, dom["ref"]), "name": people(), "amount": amt, "priors": priors, "tier": rng.randrange(4),
                "cat": a["ov_cat"] if (a["ov_cat"] and chance(rng, 0.35)) else rng.choice(cats[1:5])}

    def naive(r):
        return sum(1 for t in thr if r["amount"] > t)
    target = rng.randrange(4)
    r = None
    for _ in range(500):
        c = rand_r()
        if _approver(a, c) != target:
            continue
        if naive(c) != target or chance(rng, 0.2):
            r = c
            break
    if r is None:
        raise ValueError("no approver record")
    gold_l = _approver(a, r)
    req_date = base_date
    tn, tiers = dom["tier_name"], dom["tiers"]
    st = rng.randrange(3)
    opts = [[f"No further approval: the {ladder[0]} decides alone", f"No sign-off needed beyond the {ladder[0]}",
             f"The {ladder[0]} can approve it alone"][st]] + \
           [[f"Approval by the {x}", f"Sign-off from the {x}", f"The {x} must approve"][st] for x in ladder[1:]]
    lines = [P(rng, f"{org} — approval limits for {dom['requests']}", f"{org}: who signs off {dom['requests']}", f"Delegated authority ({org}, {dom['requests']})"), ""]
    lines.append(P(rng, "Approval level by amount (after aggregation):", "Use the aggregated amount to find the approver:", "Authority table:"))
    lines.append(f"- up to and including {money(thr[0], sym)}: the {ladder[0]} decides alone")
    lines.append(f"- above {money(thr[0], sym)} and up to {money(thr[1], sym)}: {ladder[1]}")
    lines.append(f"- above {money(thr[1], sym)} and up to {money(thr[2], sym)}: {ladder[2]}")
    lines.append(f"- above {money(thr[2], sym)}: {ladder[3]}")
    lines.append("")
    lines.append(P(rng, "Aggregation: add to the amount every earlier {req} by the same {party} made no more than {N} days before this one (earlier requests by other people never count).",
                   "Before using the table, combine this {req} with any previous {reqs} from the same {party} dated within the preceding {N} days (inclusive); {reqs} by anyone else are ignored.",
                   "Split requests: the amounts of the same {party}'s earlier {reqs} made {N} days or fewer before this one are added before the table is applied.")
                 .format(req=dom["request"], reqs=dom["requests"], party=dom["party"], N=a["N"]))
    if a["tier_rel"]:
        lines.append(P(rng, f"Then, for a {tn} of {tiers[a['t_star']]} or higher ({plural(tn)} from lowest: {', '.join(tiers)}), go down one approval level (never below the first).",
                       f"Next, {plural(tn)} ranked {' < '.join(tiers)}: at {tiers[a['t_star']]} or above the required level drops by one, but not below the {ladder[0]} level."))
    if a["ov_cat"]:
        lines.append(P(rng, f"Finally, anything concerning {a['ov_cat']} needs at least the {ladder[a['ov_lvl']]}, whatever the amount.",
                       f"Last step: {a['ov_cat']} always require sign-off at the {ladder[a['ov_lvl']]} level or higher."))
    ptext = "\n".join(lines)
    nr = rng.choice(NR)

    def render(missing):
        f = [("Reference", r["ref"]), (cap(dom["party"]), r["name"]), ("Date", fd(req_date, style)),
             ("Amount", nr if "amount" in missing else money(r["amount"], sym)), ("Category", nr if "cat" in missing else r["cat"]),
             (cap(tn), nr if "tier" in missing else tiers[r["tier"]])]
        hist = []
        for i, (pd, pa) in enumerate(r["priors"]):
            dtxt = nr if (missing and f"prior{i}" in missing) else fd(req_date - dt.timedelta(days=pd), style)
            hist.append(f"{dtxt} — {money(pa, sym)} ({r['name']})")
        for nm, pd, pa in r["_decoys"]:
            hist.append(f"{fd(req_date - dt.timedelta(days=pd), style)} — {money(pa, sym)} ({nm})")
        body = "\n".join(f"{k2}: {v}" for k2, v in f)
        hist_txt = "\n".join("- " + h for h in sorted(hist)) if hist else "- none"
        if fmt == "dict":
            return {"policy": ptext, "request": {k2: v for k2, v in f}, "earlier_requests": [h for h in sorted(hist)]}
        return f"{ptext}\n\nRequest\n{body}\n\n{P(random.Random(3), 'Earlier requests on file (date — amount — requester)', 'Request history')}:\n{hist_txt}"
    r["_decoys"] = [(people(), rng.randint(0, a["N"]), Decimal(rng.choice([base, base * 2]))) for _ in range(rng.randint(0, 2))]
    q = P(rng, "Whose approval does {ref} need?", "Which approval level applies to {ref}?", "Who must approve {ref} under the limits above?",
          "Applying aggregation and every rule, who approves {ref}?").format(ref=r["ref"])
    near = [[f"Approval by the {x}", f"Sign-off from the {x}", f"The {x} must approve"][st] for x in roles if x not in ladder][:1]
    field, gold = ordered_choice_field(q, opts, gold_l)
    spec = {"kind": "approver", "auth": {k2: (v if k2 != "thr" else v) for k2, v in a.items() if k2 != "cats"},
            "record": {"amount": str(r["amount"]), "priors": [[pd, str(pa)] for pd, pa in r["priors"]], "tier": r["tier"], "cat": r["cat"]}}
    items = [item(render(set()), field, gold, "approver", d_id, spec, {"near_miss": near, "decisive": [r["ref"]]})]
    if want_unknown:
        cands = ["amount", "cat"] + (["tier"] if a["tier_rel"] else []) + [f"prior{i}" for i in range(len(r["priors"]))]
        rng.shuffle(cands)
        for f in cands:
            outs = set()
            if f == "amount":
                for v in [Decimal(1)] + [Decimal(t + dd) for t in thr for dd in (0, 1)] + [Decimal(thr[2] * 3)]:
                    outs.add(_approver(a, {**r, "amount": v}))
            elif f == "cat":
                for v in a["cats"]:
                    outs.add(_approver(a, {**r, "cat": v}))
            elif f == "tier":
                for v in range(4):
                    outs.add(_approver(a, {**r, "tier": v}))
            else:
                i = int(f[5:])
                for v in (1, a["N"], a["N"] + 1, a["N"] * 3):
                    pr = list(r["priors"])
                    pr[i] = (v, pr[i][1])
                    outs.add(_approver(a, {**r, "priors": pr}))
            if len(outs) >= 2:
                rec2 = dict(spec["record"])
                if f.startswith("prior"):
                    i = int(f[5:])
                    rec2["priors"] = [[None if j == i else pd, pa] for j, (pd, pa) in enumerate(rec2["priors"])]
                else:
                    rec2[f] = None
                items.append(item(render({f}), dict(field), None, "approver", d_id, {**spec, "record": rec2}, {}, child_of=0))
                break
        else:
            raise ValueError("no decisive approver field")
    return items


# ================================================================================================ independent re-check
def recheck(spec):
    """Re-derive the gold from the spec (separate implementation). Returns option index/text key semantics as follows:
    eligibility -> bool; action -> outcome text; applicable_window -> window text; band_score -> int; approver -> level int;
    which_record / not_eligible -> index of the answer record; None when a needed fact is missing."""
    k = spec["kind"]
    if k == "band_score":
        b, r = spec["band"], spec["record"]
        if r["q"] is None or r["tier"] is None or r["cat"] is None or (b["reduce"] and r["red"] is None) or (b["reduce_exc"] and r["red"] and r["exc"] is None):
            return None
        n = len([t for t in b["thr"] if t <= r["q"]])
        if b["uplift"] and r["tier"] >= b["t_star"]:
            n = n + 1 if n < len(b["thr"]) else n
        if b["reduce"] and r["red"]:
            n = (n - 1 if n > 0 else 0) if (b["reduce_exc"] and r["exc"]) else 0
        if b["cap_cat"] == r["cat"] and b["cap_cat"] is not None and n > b["cap"]:
            n = b["cap"]
        return n
    if k == "approver":
        a, r = spec["auth"], spec["record"]
        if r["amount"] is None or r["cat"] is None or (a["tier_rel"] and r["tier"] is None) or any(pd is None for pd, _ in r["priors"]):
            return None
        tot = Decimal(r["amount"])
        for pd, pa in r["priors"]:
            if pd <= a["N"]:
                tot += Decimal(pa)
        lvl = 0
        while lvl < 3 and tot > a["thr"][lvl]:
            lvl += 1
        if a["tier_rel"] and r["tier"] >= a["t_star"] and lvl > 0:
            lvl -= 1
        if a["ov_cat"] and r["cat"] == a["ov_cat"]:
            lvl = a["ov_lvl"] if lvl < a["ov_lvl"] else lvl
        return lvl
    p = spec["policy"]

    def elig(r):
        if r["cat"] is None:
            return None
        if r["cat"] == p["excl"]:
            if r["flags"][p["f_exc"]] is None:
                return None
            if r["flags"][p["f_exc"]] is False:
                return False
            if p["Wf"]:
                lim = p["Wf"]
            else:
                if r["tier"] is None:
                    return None
                lim = p["W"] + p["E"] * (r["tier"] >= p["t_star"])
        else:
            if r["tier"] is None:
                return None
            lim = p["W"] + (p["E"] if r["tier"] >= p["t_star"] else 0)
        if r["days"] is None:
            return None
        ok = r["days"] <= lim if p["cmp"] == "le" else r["days"] < lim
        if not ok:
            return False
        if p["f_req"] is not None:
            fr = r["flags"][p["f_req"]]
            if fr is None:
                return None
            if not fr:
                if p["S"] is None:
                    return False
                if r["amount"] is None:
                    return None
                return Decimal(r["amount"]) <= p["S"]
        return True
    if k == "eligibility":
        return elig(spec["record"])
    if k == "action":
        r = spec["record"]
        e = elig(r)
        if e is None:
            return None
        if not e:
            return spec["texts"]["deny"]
        if r["amount"] is None:
            return None
        if Decimal(r["amount"]) > p["T"]:
            return spec["texts"]["refer"]
        return spec["texts"]["partial"] if r["days"] > p["Wfull"] else spec["texts"]["full"]
    if k == "applicable_window":
        r = spec["record"]
        if r["cat"] is None:
            return None
        if r["cat"] == p["excl"]:
            if r["flags"][p["f_exc"]] is None:
                return None
            if not r["flags"][p["f_exc"]]:
                return spec["excl_txt"]
            if p["Wf"]:
                return f"Within {p['Wf']} calendar days of the {spec['event']}"
        if r["tier"] is None:
            return None
        return f"Within {p['W'] + (p['E'] if r['tier'] >= p['t_star'] else 0)} calendar days of the {spec['event']}"
    if k in ("which_record", "not_eligible"):
        vals = [elig(r) for r in spec["records"]]
        want = spec["target"]
        if any(v is None for v in vals):
            return None
        hits = [i for i, v in enumerate(vals) if v == want]
        return hits[0] if len(hits) == 1 else "invalid"
    raise KeyError(k)


# ================================================================================================ enumerated completions
def _rec_domains(p, rec):
    """Plausible values of each withheld (None) field of a policy record, as the unknown proof uses them (_vals), in spec form."""
    out = []
    for f in ("days", "tier", "cat", "amount"):
        if rec.get(f) is None:
            vals = _vals(p, f, None)
            out.append((f, [str(v) for v in vals] if f == "amount" else vals))
    for i, v in enumerate(rec.get("flags") or []):
        if v is None:
            out.append((f"flag{i}", [False, True]))
    return out


def _fill_rec(rec, assign):
    r = dict(rec)
    r["flags"] = list(rec.get("flags") or [])
    for f, v in assign:
        if f.startswith("flag"):
            r["flags"][int(f[4:])] = v
        else:
            r[f] = v
    return r


def worlds(spec) -> list:
    """recheck() over every completion of the withheld record fields (None in the spec), over the same plausible values the
    generator's unknown proof enumerates. One world for an answerable spec. Used by gen_traps.py to re-derive trap golds."""
    import itertools
    k = spec["kind"]
    if k == "band_score":
        b, r = spec["band"], spec["record"]
        doms = {"q": sorted({0, b["thr"][-1] * 3} | {t + dd for t in b["thr"] for dd in (-1, 0)}), "tier": [0, 1, 2, 3],
                "cat": ([b["cap_cat"]] if b["cap_cat"] else []) + ["__other__"], "red": [False, True], "exc": [False, True]}
        miss = [f for f in ("q", "tier", "cat", "red", "exc") if r.get(f) is None]
        return [recheck({**spec, "record": {**r, **dict(zip(miss, combo))}}) for combo in itertools.product(*[doms[f] for f in miss])]
    if k == "approver":
        a, r = spec["auth"], spec["record"]
        doms = []
        if r["amount"] is None:
            doms.append(("amount", [str(v) for v in [1] + [t + dd for t in a["thr"] for dd in (0, 1)] + [a["thr"][2] * 3]]))
        if r["cat"] is None:
            doms.append(("cat", ([a["ov_cat"]] if a["ov_cat"] else []) + ["__other__"]))
        if r.get("tier") is None and a["tier_rel"]:
            doms.append(("tier", [0, 1, 2, 3]))
        for i, (pd, _) in enumerate(r["priors"]):
            if pd is None:
                doms.append((f"prior{i}", [1, a["N"], a["N"] + 1, a["N"] * 3]))
        out = []
        for combo in itertools.product(*[v for _, v in doms]):
            rec = dict(r)
            rec["priors"] = [list(x) for x in r["priors"]]
            for (f, _), v in zip(doms, combo):
                if f.startswith("prior"):
                    rec["priors"][int(f[5:])][0] = v
                else:
                    rec[f] = v
            out.append(recheck({**spec, "record": rec}))
        return out
    p = spec["policy"]
    if k in ("which_record", "not_eligible"):
        per = [_rec_domains(p, r) for r in spec["records"]]
        flat = [(i, f, vals) for i, d in enumerate(per) for f, vals in d]
        out = []
        for combo in itertools.product(*[vals for _, _, vals in flat]):
            recs = [_fill_rec(r, [(f, v) for (j, f, _), v in zip(flat, combo) if j == i]) for i, r in enumerate(spec["records"])]
            out.append(recheck({**spec, "records": recs}))
        return out
    doms = _rec_domains(p, spec["record"])
    return [recheck({**spec, "record": _fill_rec(spec["record"], list(zip([f for f, _ in doms], combo)))})
            for combo in itertools.product(*[v for _, v in doms])]
