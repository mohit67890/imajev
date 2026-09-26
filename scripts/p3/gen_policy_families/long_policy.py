"""long_policy: 1,500-2,800-word policies, contracts, SOPs, terms, schedules and incident logs, plus a case or record.

Every question needs two or more clauses (an exception beats the rule, a later amendment overrides, a definition changes the scope,
a deemed-receipt rule moves a date). Six schemas, each with its own evaluator and an `ignore=` switch that produces the
wrong-path answers used as near-miss options:

    window     request deadline: base period, category exception, exception-to-exception, tier extension, amendment,
               business-day definition, submitted = sent vs received, grace, referral threshold
    banded     amount from a banded table over a measure with definitional exclusions, special-tier schedule, amendment, cap
    authority  who must approve: amount bands, category override, aggregation of related requests, amendment, delegation
    scope      which records fall under a defined term (criteria + exclusion + amended threshold)
    sla        incident log against an SLA: priority matrix, re-prioritisation, clock pauses, resolution target
    notice     contract termination: initial term, renewals, notice period, deemed receipt, amendment, service address

Gold comes from the evaluator on structured parameters; `recheck(spec)` is an independent re-implementation used by the tests.
"""
from __future__ import annotations

import datetime as dt
import random
from decimal import Decimal

from .common import (DOMAINS, DOMAIN_IDS, P, add_bdays, add_days, chance, choice_field, closure_days, code, item,
                     n_days, noul_field, q2, rand_date, score_field)
from .docbuild import Doc, front_matter, render_record

FAMILY = "long_policy"
UNKNOWN_P = 0.19  # driver re-rolls a failed unknown draw, so ask a little above 15/85
KINDS = ["window_outcome", "window_deadline", "window_length", "window_ontime",
         "banded_amount", "banded_band", "banded_over",
         "authority_role", "authority_valid",
         "scope_which", "scope_one", "scope_count",
         "sla_met", "sla_due", "sla_priority",
         "notice_end", "notice_valid"]

EVENT = {"retail": "the date of delivery", "logistics": "the date of delivery", "hr": "the date of the qualifying event",
         "healthcare_admin": "the date of service", "insurance": "the date of loss", "finance": "the transaction date",
         "saas_ops": "the date the incident was resolved", "travel": "the scheduled departure date",
         "public_sector": "the date of the decision notice", "legal": "the invoice date"}
EVENT_LABEL = {"retail": "Delivery date", "logistics": "Delivered on", "hr": "Date of qualifying event", "healthcare_admin": "Date of service",
               "insurance": "Date of loss", "finance": "Transaction date", "saas_ops": "Incident resolved on", "travel": "Scheduled departure",
               "public_sector": "Decision notice issued", "legal": "Invoice date"}
DOC_TITLES = {"policy": ["Policy", "Policy Statement", "Operating Policy"], "sop": ["Standard Operating Procedure", "Operating Procedure", "Handling Procedure"],
              "terms": ["Terms and Conditions", "Customer Terms", "General Terms"], "contract": ["Agreement", "Services Agreement", "Framework Agreement"],
              "schedule": ["Schedule of Charges and Entitlements", "Service Schedule", "Tariff Schedule"],
              "incident_log": ["Incident Record and Service Standard", "Incident File", "Service Incident Pack"]}


def _dom(rng):
    d = rng.choice(DOMAIN_IDS)
    return d, DOMAINS[d]


def _outcomes(dom: dict, request: str, refer_role: str) -> dict:
    o = dom["outcomes"]
    decline = next(x for x in o if any(w in x for w in ("decline", "reject", "refuse", "uphold")))
    return {"approve": o[0], "reduced": o[1], "decline": decline, "refer": f"refer the {request} to the {refer_role}"}


def add_months(d: dt.date, n: int) -> dt.date:
    import calendar
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _iso(d):
    return d.isoformat() if d is not None else None


def _d(s):
    return dt.date.fromisoformat(s) if s else None


# ================================================================================================ window
def w_eval(p: dict, c: dict, ignore=frozenset()) -> dict:
    """Returns {"outcome", "deadline", "window", "unit"}. `ignore` names twists a careless reader would miss."""
    hol = [_d(x) for x in p["holidays"]]
    unit = p["unit"]
    if "unitdef" in ignore and p["unit_via_def"]:
        unit = "calendar"
    ev = c["event"]
    amended = p["amend"] is not None and ev >= p["amend"]["date"] and "amend" not in ignore
    base = p["amend"]["value"] if amended and p["amend"]["what"] == "base" else p["base"]
    window = base
    if c["category"] in p["exc_cats"] and "exc" not in ignore:
        exc_win = p["amend"]["value"] if amended and p["amend"]["what"] == "exc" else p["exc"]
        flag_rule = p["flag_rule"] and not (amended and p["amend"]["what"] == "drop_flag") and "flag" not in ignore
        if not (flag_rule and c["flag"]):
            window = exc_win
    qual = c["tier"] in p["qual_tiers"] and add_months(c["member_since"], p["qual_months"]) <= ev
    if "qualmonths" in ignore:
        qual = c["tier"] in p["qual_tiers"]
    if p["ext"] and qual and "ext" not in ignore:
        window += p["ext"]
    deadline = add_bdays(ev, window, hol) if unit == "business" else add_days(ev, window)
    sub = c["received"] if (p["received_rule"] != ("received" in ignore)) else c["sent"]
    if p["refer"] is not None and c["amount"] > p["refer"] and "refer" not in ignore:
        return {"outcome": "refer", "deadline": deadline, "window": window, "unit": unit}
    if sub <= deadline:
        out = "approve"
    elif p["grace"] and sub <= add_days(deadline, p["grace"]) and (p["grace_for"] == "all" or qual) and "grace" not in ignore:
        out = "reduced"
    else:
        out = "decline"
    return {"outcome": out, "deadline": deadline, "window": window, "unit": unit}


W_TWISTS = ["unitdef", "amend", "exc", "flag", "qualmonths", "ext", "received", "grace", "refer"]


def _w_params(rng, dom_id, dom, difficulty):
    cats = list(dom["categories"])
    rng.shuffle(cats)
    base = rng.choice([10, 14, 15, 20, 21, 28, 30, 45, 60])
    unit = rng.choice(["business", "calendar"]) if difficulty >= 4 else rng.choice(["business", "calendar", "calendar"])
    exc = rng.choice([x for x in (5, 7, 10, 14, 20, 30, 45, 90) if x != base])
    tiers = dom["tiers"]
    y0 = rng.choice([2025, 2026, 2027])
    p = {"base": base, "unit": unit, "unit_via_def": unit == "business" and (difficulty >= 4 or chance(rng, 0.4)),
         "exc_cats": cats[:rng.randint(1, 3)], "exc": exc, "flag_idx": rng.randrange(len(dom["flags"])),
         "flag_rule": difficulty >= 4 or chance(rng, 0.5), "qual_tiers": tiers[-2:], "qual_months": rng.choice([3, 6, 12, 18, 24]),
         "ext": rng.choice([0, 5, 7, 10, 14]) if difficulty >= 4 else rng.choice([0, 0, 7, 10]),
         "received_rule": chance(rng, 0.5), "grace": rng.choice([0, 0, 3, 5, 7]) if difficulty >= 4 else 0,
         "grace_for": rng.choice(["all", "qualifying"]), "refer": None, "amend": None,
         "other_cats": cats[3:7], "year": y0}
    if difficulty == 5 or chance(rng, 0.3):
        p["refer"] = rng.choice([500, 750, 1000, 1500, 2000, 2500, 5000])
    if difficulty >= 4 or chance(rng, 0.6):
        what = rng.choice(["base", "exc", "drop_flag"] if p["flag_rule"] else ["base", "exc"])
        val = rng.choice([x for x in (7, 10, 14, 21, 30, 40, 45, 60) if x not in (base, exc)]) if what != "drop_flag" else None
        p["amend"] = {"date": dt.date(y0, rng.randint(2, 11), rng.randint(1, 28)), "what": what, "value": val}
    hol = closure_days(rng, y0 - 1, 5) + closure_days(rng, y0, 7) + closure_days(rng, y0 + 1, 6)
    p["holidays"] = [h.isoformat() for h in sorted(set(hol))]
    return p


def _w_case(rng, p, dom, difficulty):
    ev_lo = dt.date(p["year"], 1, 10)
    if p["amend"]:
        ad = p["amend"]["date"]
        ev = ad + dt.timedelta(days=rng.randint(-40, 40))
    else:
        ev = ev_lo + dt.timedelta(days=rng.randrange(300))
    if rng.random() < 0.55:
        cat = rng.choice(p["exc_cats"])
    else:
        cat = rng.choice(p["other_cats"] + [x for x in dom["categories"] if x not in p["exc_cats"]])
    tier = rng.choice(dom["tiers"])
    member_since = add_months(ev, -rng.choice([1, 2, 4, 5, 7, 11, 13, 17, 19, 23, 25, 30, 40])) - dt.timedelta(days=rng.randint(0, 20))
    c = {"event": ev, "category": cat, "tier": tier, "member_since": member_since, "flag": chance(rng, 0.5),
         "amount": Decimal(rng.randint(40, 6000)) + Decimal(rng.choice([0, 0, 50, 99, 25])) / 100}
    # submission near an interesting deadline: pick one of the candidate deadlines and jitter
    cands = [w_eval(p, {**c, "sent": ev, "received": ev}, frozenset(ig))["deadline"] for ig in ([],) + tuple([t] for t in W_TWISTS)]
    target = rng.choice(cands)
    sent = target + dt.timedelta(days=rng.randint(-3, 3))
    if sent <= ev:
        sent = ev + dt.timedelta(days=1)
    lag = rng.choice([0, 1, 2, 3, 4, 6]) if p["received_rule"] or chance(rng, 0.5) else rng.choice([0, 1, 2])
    c["sent"], c["received"] = sent, sent + dt.timedelta(days=lag)
    return c


def _w_render(rng, dom_id, dom, p, c, difficulty, doc_kind, missing=None):
    """Render the window document. `missing` removes one decisive fact (unknown variants)."""
    doc = Doc(rng, dom_id, doc_kind)
    req, reqs = dom["request"], dom["requests"]
    ev_phrase = EVENT[dom_id]
    flag_text, flag_short = dom["flags"][p["flag_idx"]]
    tiers = p["qual_tiers"]
    role_up = rng.choice([r for r in dom["roles"][2:5] if r != doc.ctx["role"]])
    p["_role_up"] = role_up
    days_word = "Days" if p["unit_via_def"] else ("business days" if p["unit"] == "business" else "calendar days")
    if not p["unit_via_def"] and p["unit"] == "calendar" and chance(rng, 0.5):
        days_word = "days"
    dw = lambda n: n_days(rng, n).replace("days", days_word).replace("day", days_word.rstrip("s")) if days_word != "days" else n_days(rng, n)

    # --- definitions
    defs = []
    qm_phrase = f"for at least {p['qual_months']} complete months on {ev_phrase}"
    defs.append(("d_qual", P(rng, f"\"Qualifying Member\" means a {dom['party']} whose {dom['tier_name']} is {tiers[0]} or {tiers[1]} and who has held that {dom['tier_name']} {qm_phrase}.",
                             f"\"Qualifying Member\": a {dom['party']} holding {dom['tier_name']} {tiers[0]} or {tiers[1]}, provided that status has been held {qm_phrase}.",
                             f"A {dom['party']} is a \"Qualifying Member\" only if (a) their {dom['tier_name']} is {tiers[0]} or {tiers[1]}, and (b) they have held it {qm_phrase}.")))
    if p["unit_via_def"]:
        defs.append(("d_day", P(rng, "\"Day\" or \"Days\" means a business day, that is any day other than a Saturday, a Sunday or a public holiday listed in {ref:sched_hol}.",
                                "In this document \"Days\" refers to business days only: Saturdays, Sundays and the public holidays in {ref:sched_hol} are not counted.",
                                "\"Days\" are working days. A day is not a working day if it is a Saturday or Sunday or appears in the holiday list at {ref:sched_hol}.")))
    elif p["unit"] == "business":
        defs.append(("d_bday", P(rng, "\"Business day\" means any day other than a Saturday, a Sunday or a public holiday listed in {ref:sched_hol}.",
                                 "A business day is a weekday that is not one of the public holidays listed in {ref:sched_hol}.")))
    sub_rule = (P(rng, f"A {req} is treated as submitted on the date it is received by {doc.ctx['system']}, not the date it was sent.",
                  f"The submission date of a {req} is the date on which {doc.org} receives it; the date of sending is disregarded.",
                  f"For every time limit in this document, a {req} counts as made when it is received, whatever date it was sent.")
                if p["received_rule"] else
                P(rng, f"A {req} is treated as submitted on the date it is sent, as shown by the postmark or electronic timestamp, even if it is received later.",
                  f"The submission date of a {req} is the date it was sent by the {dom['party']}; any delay in receipt is disregarded.",
                  f"For every time limit in this document, a {req} counts as made on the date of sending, not the date of receipt."))
    defs.append(("d_sub", sub_rule))
    defs.append(("d_count", P(rng, f"Periods are counted from the day after {ev_phrase}, and the last day of a period is included.",
                             f"A period \"of\" or \"from\" {ev_phrase} starts on the following day; a {req} submitted on the last day of the period is in time.",
                             f"When counting a period, {ev_phrase} itself is day zero and the final day of the period still counts as within it.")))
    other_def = P(rng, f"\"Case Handler\" means the {doc.ctx['role']} assigned to a {req} in {doc.ctx['system']}.",
                  f"\"Record\" means the case entry held in {doc.ctx['system']}, including attachments.",
                  f"\"Working hours\" means 09:00 to 17:00 on business days, for the purposes of staff rotas only.")
    defs.append(other_def)
    rng.shuffle(defs)
    sec_def = doc.add(P(rng, "Definitions", "Definitions and interpretation", "Key terms", "Interpretation"), defs)

    # --- core rule
    ev_cap = ev_phrase[0].upper() + ev_phrase[1:]
    base_txt = P(rng, f"A {req} must be submitted within {dw(p['base'])} of {ev_phrase}. {req[0].upper() + req[1:]}s submitted after that period are declined, except as set out in this document.",
                 f"The standard period for submitting a {req} is {dw(p['base'])} from {ev_phrase}; a {req} outside this period will be declined unless another clause of this document provides otherwise.",
                 f"Subject to the remainder of this document, {dom['parties']} have {dw(p['base'])} from {ev_phrase} to submit a {req}. Late {reqs} are declined.")
    exc_list = ", ".join(p["exc_cats"][:-1]) + (" and " if len(p["exc_cats"]) > 1 else "") + p["exc_cats"][-1]
    exc_val = "the period set out in {ref:sched_exc}" if missing == "exc" else dw(p["exc"])
    exc_txt = P(rng, f"Notwithstanding {{ref:base}}, a {req} relating to {exc_list} must be submitted within {exc_val} of {ev_phrase}.",
                f"By way of exception to {{ref:base}}, the period for {reqs} concerning {exc_list} is {exc_val} from {ev_phrase}.",
                f"{{ref:base}} does not apply to {exc_list}. For those, the {req} must reach us within {exc_val} of {ev_phrase}." if p["received_rule"] else
                f"{{ref:base}} does not apply to {exc_list}. For those, the {req} must be sent within {exc_val} of {ev_phrase}.")
    core1 = [("base", base_txt), ("exc", exc_txt)]
    for oc in rng.sample(p["other_cats"], min(len(p["other_cats"]), 2 if difficulty >= 4 else 1)):
        n_oc = rng.choice([x for x in (7, 14, 21, 30, 60, 90) if x not in (p["base"], p["exc"])])
        core1.append(P(rng, f"For {oc}, the period is {dw(n_oc)} from {ev_phrase}, and the {req} must include the supporting documents listed in the evidence section.",
                       f"Where {oc} were supplied as part of a bundle, the period is {dw(n_oc)} from {ev_phrase} and applies to the whole bundle.",
                       f"{reqs[0].upper() + reqs[1:]} about {oc} follow a separate timetable: {dw(n_oc)} from {ev_phrase}.",
                       f"A {req} relating to {oc} is subject to a period of {dw(n_oc)} from {ev_phrase}; {{ref:base}} does not apply to it."))
    sec_core = doc.add(P(rng, "Time limits", f"Submitting a {req}", "Periods for submission", "When a request can be made", "Deadlines"), core1)

    later = []
    if p["flag_rule"]:
        later.append(("flag", P(rng, f"The period in {{ref:exc}} does not apply where {flag_text}; in that case the standard period in {{ref:base}} applies instead.",
                                f"Where {flag_text}, {{ref:exc}} is disregarded and the {req} is assessed under {{ref:base}}.",
                                f"If {flag_text}, the {dom['party']} is not bound by the shorter or longer period in {{ref:exc}}; {{ref:base}} governs.")))
    if p["ext"]:
        later.append(("ext", P(rng, f"Qualifying Members receive an additional {dw(p['ext'])} on top of whichever period applies under this section.",
                               f"The applicable period (whether under {{ref:base}} or {{ref:exc}}) is extended by {dw(p['ext'])} for Qualifying Members.",
                               f"For a Qualifying Member, add {dw(p['ext'])} to the period that would otherwise apply.")))
    if p["grace"]:
        who = "any" if p["grace_for"] == "all" else "a Qualifying Member's"
        later.append(("grace", P(rng, f"Where {who} {req} is submitted no more than {n_days(rng, p['grace'])} (calendar days) after the end of the applicable period, it is not declined; instead the handler will {_outcomes(dom, req, role_up)['reduced']}.",
                                 f"A late {req} is not declined if it is {'from a Qualifying Member and ' if p['grace_for'] != 'all' else ''}submitted within {n_days(rng, p['grace'])} (calendar days) after the period ends; the outcome in that case is to {_outcomes(dom, req, role_up)['reduced']}.")))
    if p["refer"] is not None:
        later.append(("refer", P(rng, f"Where the {dom['amount_noun']} exceeds {doc.m(p['refer'])}, the handler must not decide the {req} and must refer it to the {role_up}, whether or not it was submitted in time.",
                                 f"Any {req} with a {dom['amount_noun']} above {doc.m(p['refer'])} is referred to the {role_up} for decision; this applies before any question of timing is considered.")))
    for oc in rng.sample(p["other_cats"], min(len(p["other_cats"]), 1)):
        later.append(P(rng, f"A {doc.ctx['role']} may accept {reqs} for {oc} by telephone, but must record the call reference in {doc.ctx['system']}.",
                       f"Photographs are required for {reqs} concerning {oc}; the absence of photographs does not change any time limit.",
                       f"{reqs[0].upper() + reqs[1:]} for {oc} made by an authorised representative are handled in the same way as those made directly."))
    rng.shuffle(later)
    sec_later = doc.add(P(rng, "Exceptions and extensions", "Special cases", "Variations to the time limits", "Further rules", "Adjustments"), later) if later else None

    # --- amendment
    sec_am = None
    if p["amend"]:
        a = p["amend"]
        if a["what"] == "base":
            t = f"With effect for {reqs} where the {ev_phrase[4:]} is on or after {doc.fd(a['date'])}, the period in {{ref:base}} is changed to {dw(a['value'])}. Earlier cases keep the previous period."
        elif a["what"] == "exc":
            t = f"For cases where the {ev_phrase[4:]} falls on or after {doc.fd(a['date'])}, the period in {{ref:exc}} becomes {dw(a['value'])}; for earlier cases the original period continues to apply."
        else:
            t = f"{{ref:flag}} is withdrawn for cases where the {ev_phrase[4:]} is on or after {doc.fd(a['date'])}; from that date {{ref:exc}} applies whether or not {flag_text}."
        sec_am = doc.add(P(rng, "Amendments", "Amendment record", "Changes to this document", "Schedule of amendments"),
                         [("amend", t), P(rng, f"The contact address for postal {reqs} changed on {doc.fd(a['date'] - dt.timedelta(days=rng.randint(20, 90)))}; mail sent to the old address is forwarded.",
                                           f"Template letters were updated on {doc.fd(a['date'] + dt.timedelta(days=rng.randint(5, 40)))}; this did not change any time limit.")])
    # --- schedules
    sched = []
    hol = [_d(x) for x in p["holidays"]]
    if p["unit"] == "business":
        sched.append(("sched_hol", "Public holidays observed: " + "; ".join(doc.fd(h) for h in hol) + "."))
    else:
        sched.append(("sched_hol", "Office closure days (for staffing only; they do not affect any period in this document): " + "; ".join(doc.fd(h) for h in hol[:6]) + "."))
    if missing == "exc":
        sched.append(("sched_exc", f"[The table of category periods referred to in {{ref:exc}} is not reproduced in this extract.]"))
    sec_sched = doc.add(P(rng, "Schedules", "Annex", "Appendix", "Reference tables"), sched)

    doc.boiler(rng.randint(4, 7))
    core = [sec_def, sec_core] + ([sec_later] if sec_later else []) + ([sec_am] if sec_am else []) + [sec_sched]
    doc.order(core[:-2] + core[-2:] if sec_am else core, boiler_front=rng.randint(1, 2))
    # keep amendments + schedules at the very end
    tail_secs = [s for s in (sec_am, sec_sched) if s is not None]
    doc.sections = [s for s in doc.sections if s not in tail_secs] + tail_secs
    eff = add_months(p["amend"]["date"], -rng.randint(8, 20)) if p["amend"] else dt.date(p["year"] - 1, rng.randint(1, 12), 1)
    title = f"{P(rng, *dom['doc_nouns'])}"
    doc.front = front_matter(doc, title, eff, f"{rng.randint(2, 6)}.{rng.randint(0, 4)}")

    # --- case record
    people = doc.people
    party_name = people()
    ref = code(rng, dom["ref"])
    fields = [("Reference", ref), (dom["party"].capitalize(), party_name),
              (dom["tier_name"].capitalize(), c["tier"]),
              (f"{dom['tier_name'].capitalize()} held since", doc.fd(c["member_since"]) if missing != "member_since" else "not recorded"),
              ("Category", c["category"]),
              (dom["amount_noun"].capitalize(), doc.m(c["amount"])),
              (EVENT_LABEL[dom_id], doc.fd(c["event"]) if missing != "event" else "not recorded"),
              (f"{req.capitalize()} sent", doc.fd(c["sent"]) if missing != "sent" else "not recorded"),
              (f"{req.capitalize()} received", doc.fd(c["received"]) if missing != "received" else "not recorded"),
              (flag_short.capitalize() + "?", ("yes" if c["flag"] else "no") if missing != "flag" else "not stated"),
              ("Channel", rng.choice(["online form", "email", "telephone, confirmed in writing", "post", "in person"])),
              ("Assigned to", f"{people()} ({doc.ctx['role']})")]
    head, rest = fields[:2], fields[2:]
    rng.shuffle(rest)
    notes = rng.sample([f"{party_name} asked for an update by email.", f"Previous {req} on this account closed without action.",
                        f"Duplicate submission merged into {ref}.", "Acknowledgement letter sent.", "Awaiting internal review.",
                        f"{party_name} mentioned a possible complaint if the matter is not resolved quickly."], 2)
    doc.tail = render_record(rng, P(rng, "Case record", f"{req.capitalize()} {ref}", "Case file extract", "Record under review"), head + rest, notes=notes)
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("base", "exc", "flag", "ext", "amend", "refer", "grace") if doc.rendered(a)]
    return text, doc, ref, party_name, decisive


def _w_outcome_texts(dom, req, role_up):
    return _outcomes(dom, req, role_up)


def gen_window(rng, difficulty, kind, want_unknown):
    dom_id, dom = _dom(rng)
    need = {3: 1, 4: 2, 5: 3}[difficulty]
    want = chance(rng, 0.5)
    for _ in range(200):
        p = _w_params(rng, dom_id, dom, difficulty)
        c = _w_case(rng, p, dom, difficulty)
        g = w_eval(p, c)
        wrong = {t: w_eval(p, c, frozenset([t])) for t in W_TWISTS}
        if kind == "window_ontime" and g["outcome"] == "refer":
            continue
        if kind == "window_ontime" and _ < 150 and (g["outcome"] == "approve") != want:
            continue
        if kind == "window_length" and g["window"] == p["base"] and rng.random() < 0.6:
            continue
        fk = W_KEY[kind]
        diff = [t for t, w in wrong.items() if fk(w) != fk(g)]
        if len(diff) >= need:
            break
    else:
        raise ValueError("no interesting window case")
    doc_kind = rng.choice(["policy", "terms", "sop", "policy"])
    text, doc, ref, pname, decisive = _w_render(random.Random(rng.random()), dom_id, dom, p, c, difficulty, doc_kind)
    req = dom["request"]
    outs = _outcomes(dom, req, p["_role_up"])
    field, gold, hints, ov = _w_question(rng, kind, p, c, g, wrong, doc, ref, pname, dom, outs)
    spec = _w_spec(p, c) | {"kind": kind, "opt_values": ov}
    items = [item(text, field, gold, kind, dom_id, spec, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _w_unknown(rng, p, c, kind, dom_id, dom, difficulty, doc_kind, field)
        if u is None:
            raise ValueError("no unknown variant")
        items.append(u)
    return items


def _w_spec(p, c):
    return {"schema": "window", "p": {k: (_iso(v) if isinstance(v, dt.date) else v) for k, v in p.items() if not k.startswith("_") and k not in ("other_cats",)}
            | {"amend": ({"date": _iso(p["amend"]["date"]), "what": p["amend"]["what"], "value": p["amend"]["value"]} if p["amend"] else None)},
            "c": {k: (_iso(v) if isinstance(v, dt.date) else (str(v) if isinstance(v, Decimal) else v)) for k, v in c.items()}}


def cf(rng, q, right, wrongs, fmt, k=None):
    """Choice field from canonical values: (field, gold_key, {key: canonical}). Wrong values equal to `right` are dropped."""
    seen, vals = {fmt(right)}, [right]
    for w in wrongs:
        t = fmt(w)
        if w != right and t not in seen:
            seen.add(t)
            vals.append(w)
        if k and len(vals) >= k:
            break
    texts = [fmt(v) for v in vals]
    field, gold = choice_field(rng, q, texts[0], texts[1:])
    tv = {fmt(v): v for v in vals}
    return field, gold, {o["key"]: _canon(tv[o["text"]]) for o in field["options"]}


def _canon(v):
    if isinstance(v, dt.datetime):
        return v.isoformat(timespec="minutes")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(q2(v))
    if isinstance(v, tuple):
        return list(v)
    return v


def _w_question(rng, kind, p, c, g, wrong, doc, ref, pname, dom, outs):
    """(field, gold, hints, opt_values)."""
    req = dom["request"]
    if kind == "window_outcome":
        q = P(rng, f"Applying the document to {req} {ref}, what should the handler do?",
              f"Under the {doc.org} rules above, which outcome is correct for {ref}?",
              f"What is the correct decision on {pname}'s {req} ({ref})?",
              f"Which action does the document require for {req} {ref}?")
        keys = ["approve", "reduced", "decline", "refer"]
        extra = [x for x in dom["outcomes"] if x not in outs.values()][:1]
        txt = dict(outs, **{f"x{i}": e for i, e in enumerate(extra)})
        field, gold, ov = cf(rng, q, g["outcome"], [k for k in keys if k != g["outcome"]] + list(txt)[4:], lambda k: txt[k])
        nm = [outs[w["outcome"]] for w in wrong.values() if w["outcome"] != g["outcome"]]
        return field, gold, {"near_miss": list(dict.fromkeys(nm))[:3]}, ov
    if kind == "window_deadline":
        q = P(rng, f"What was the last day on which {req} {ref} could be submitted and still be within the applicable period?",
              f"By which date did {pname}'s {req} have to be submitted to be in time?",
              f"On what date did the period for submitting {ref} end?")
        cands = [w["deadline"] for w in wrong.values()] + [g["deadline"] + dt.timedelta(days=k) for k in (-1, 1, 2, -2)]
        field, gold, ov = cf(rng, q, g["deadline"], cands, doc.fd, k=rng.randint(4, 6))
        shown = {o["text"] for o in field["options"]}
        near = [doc.fd(w["deadline"]) for w in wrong.values() if w["deadline"] != g["deadline"] and doc.fd(w["deadline"]) not in shown]
        return field, gold, {"near_miss": list(dict.fromkeys(near))[:3]}, ov
    if kind == "window_length":
        q = P(rng, f"How long was the period that applied to {req} {ref}?",
              f"Which submission period applies to {ref} under the document?",
              f"What is the applicable period for {pname}'s {req}?")
        ev = EVENT[doc.dom_id]
        fmt = lambda v: f"{v[0]} {'business' if v[1] == 'business' else 'calendar'} days from {ev}"
        other_u = "calendar" if g["unit"] == "business" else "business"
        right = (g["window"], g["unit"])
        cands = [(g["window"], other_u)] + [(w["window"], w["unit"]) for w in wrong.values()] + \
                [(w["window"], other_u) for w in wrong.values()] + [(p["base"], g["unit"]), (p["exc"], g["unit"])]
        field, gold, ov = cf(rng, q, right, cands, fmt, k=rng.randint(4, 6))
        shown = {o["text"] for o in field["options"]}
        return field, gold, {"near_miss": [fmt(x) for x in dict.fromkeys(cands) if fmt(x) not in shown and x != right][:3]}, ov
    # window_ontime
    within = g["outcome"] == "approve"
    q = P(rng, f"Was {req} {ref} submitted within the applicable period, without relying on any late-submission allowance?",
          f"Is {pname}'s {req} ({ref}) in time under the period that applies to it (ignoring any grace allowance)?",
          f"Did {ref} meet the time limit that applies to it, before any grace provision is considered?")
    neg = P(rng, f"Was {req} {ref} submitted after the applicable period had ended?", f"Is {ref} late under the period that applies to it?")
    return noul_field(q), within, {"neg": {"question": neg, "gold": not within}}, None


W_KEY = {"window_outcome": lambda r: r["outcome"], "window_ontime": lambda r: r["outcome"] == "approve",
         "window_deadline": lambda r: r["deadline"].isoformat() if isinstance(r["deadline"], dt.date) else r["deadline"],
         "window_length": lambda r: [r["window"], r["unit"]]}


# ------------------------------------------------------------------------------------------------ unknown-proof trace
# When WORLD_TRACE is a list, every unknown child records (child spec, the spec of every world its proof enumerated + the true world).
# Pure observation (no rng use, no output change); worlds_of_row() regenerates an item under it to re-derive trap golds.
WORLD_TRACE: list | None = None


def _pc(a, p, c):
    return (a[1], c) if isinstance(a, tuple) else (p, a)


def _trace(child_spec, world_specs, opt_values=None):
    if WORLD_TRACE is not None:
        WORLD_TRACE.append((child_spec, world_specs, opt_values))


def _w_unknown(rng, p, c, kind, dom_id, dom, difficulty, doc_kind, parent_field):
    """Remove one decisive fact and prove (by enumeration) that the answer is no longer determined."""
    fk = W_KEY[kind]
    opts = ["event", "exc", "member_since", "received", "sent", "flag"]
    rng.shuffle(opts)
    for miss in opts:
        alts = []
        if miss == "event":
            alts = [dict(c, event=c["event"] + dt.timedelta(days=k)) for k in range(-60, 30, 3) if c["event"] + dt.timedelta(days=k) < c["sent"]]
        elif miss == "exc":
            if c["category"] not in p["exc_cats"]:
                continue
            alts = [("p", dict(p, exc=x)) for x in (3, 5, 7, 10, 14, 20, 30, 45, 60, 90, 120)]
        elif miss == "member_since":
            if not p["ext"] and not p["grace"]:
                continue
            alts = [dict(c, member_since=add_months(c["event"], -m)) for m in (1, 2, 5, 11, 13, 25, 40)]
        elif miss == "received":
            if not p["received_rule"]:
                continue
            alts = [dict(c, received=c["sent"] + dt.timedelta(days=k)) for k in range(0, 15)]
        elif miss == "sent":
            if p["received_rule"]:
                continue
            alts = [dict(c, sent=c["received"] - dt.timedelta(days=k)) for k in range(0, 15) if c["received"] - dt.timedelta(days=k) > c["event"]]
        elif miss == "flag":
            if not p["flag_rule"] or c["category"] not in p["exc_cats"]:
                continue
            alts = [dict(c, flag=True), dict(c, flag=False)]
        vals = set()
        for a in alts:
            pp, cc = (a[1], c) if isinstance(a, tuple) else (p, a)
            vals.add(str(fk(w_eval(pp, cc))))
        if len(vals) < 2:
            continue
        text, doc, ref, pname, _ = _w_render(random.Random(rng.random()), dom_id, dom, p, c, difficulty, doc_kind, missing=miss)
        # re-ask the parent's question against the new document (names/refs differ, so re-render the question text)
        f2, _, _, _ov = _w_question(random.Random(rng.random()), kind, p, c, w_eval(p, c), {t: w_eval(p, c, frozenset([t])) for t in W_TWISTS},
                               doc, ref, pname, dom, _outcomes(dom, dom["request"], p["_role_up"]))
        spec = _w_spec(p, c) | {"missing": miss, "kind": kind}
        _trace(spec, [_w_spec(*_pc(a, p, c)) | {"kind": kind} for a in alts] + [_w_spec(p, c) | {"kind": kind}], _ov)
        return item(text, f2, None, kind, dom_id, spec, {}, child_of=0)
    return None


def recheck_window(spec):
    """Independent re-derivation of the window gold from the spec (separate code from w_eval)."""
    p, c = spec["p"], spec["c"]
    if spec.get("missing"):
        return None
    D = dt.date.fromisoformat
    ev = D(c["event"])
    hol = {D(h) for h in p["holidays"]}
    am = p["amend"]
    is_am = bool(am) and ev >= D(am["date"])
    length = am["value"] if is_am and am["what"] == "base" else p["base"]
    if c["category"] in p["exc_cats"]:
        use_exc = True
        if p["flag_rule"] and c["flag"] and not (is_am and am["what"] == "drop_flag"):
            use_exc = False
        if use_exc:
            length = am["value"] if is_am and am["what"] == "exc" else p["exc"]
    ms = D(c["member_since"])
    y, m = ms.year + (ms.month - 1 + p["qual_months"]) // 12, (ms.month - 1 + p["qual_months"]) % 12 + 1
    import calendar
    ms_plus = dt.date(y, m, min(ms.day, calendar.monthrange(y, m)[1]))
    qual = c["tier"] in p["qual_tiers"] and ms_plus <= ev
    if p["ext"] and qual:
        length += p["ext"]
    if p["unit"] == "business":
        d, left = ev, length
        while left:
            d += dt.timedelta(days=1)
            if d.weekday() < 5 and d not in hol:
                left -= 1
    else:
        d = ev + dt.timedelta(days=length)
    sub = D(c["received"]) if p["received_rule"] else D(c["sent"])
    if p["refer"] is not None and Decimal(c["amount"]) > p["refer"]:
        outcome = "refer"
    elif sub <= d:
        outcome = "approve"
    elif p["grace"] and (sub - d).days <= p["grace"] and (p["grace_for"] == "all" or qual):
        outcome = "reduced"
    else:
        outcome = "decline"
    return {"window_outcome": outcome, "window_ontime": outcome == "approve", "window_deadline": d.isoformat(),
            "window_length": [length, p["unit"]]}[spec["kind"]]


# ================================================================================================ banded
# per domain: the measured quantity, which logged components count (definition) and which do not (exclusions), the amount
BANDED = {
    "saas_ops": dict(measure="Downtime", unit="minutes", qty=(5, 240), kind="pct", benefit="service credit", base="monthly subscription fee",
                     counted=["complete outage of the service", "maintenance announced less than 48 hours in advance", "outage while a failover was in progress"],
                     excluded=["partial degradation with the service still responding", "maintenance announced at least 48 hours in advance",
                               "outage caused by the customer's own integration"],
                     exempt=("the customer's account was suspended for non-payment when the downtime occurred", "Account suspended at the time")),
    "logistics": dict(measure="Delay", unit="hours", qty=(1, 30), kind="pct", benefit="late-delivery rebate", base="freight charge",
                      counted=["time past the booked slot while the vehicle was en route", "time lost to a vehicle breakdown", "time waiting for a replacement driver"],
                      excluded=["time the consignee's site was closed", "time held at customs inspection", "time waiting for the consignee to unload beyond the free period"],
                      exempt=("the shipper failed to supply the delivery booking reference", "Booking reference missing")),
    "travel": dict(measure="Qualifying delay", unit="minutes", qty=(10, 200), kind="pct", benefit="delay compensation", base="fare paid",
                   counted=["delay caused by a technical fault", "delay caused by industrial action by the operator's own staff", "delay from the late arrival of the incoming service"],
                   excluded=["delay caused by severe weather", "delay caused by air traffic control or signalling restrictions imposed by a third party",
                             "delay caused by a passenger medical emergency"],
                   exempt=("the passenger travelled on a complimentary or staff ticket", "Complimentary ticket")),
    "hr": dict(measure="Qualifying on-call time", unit="hours", qty=(1, 24), kind="pct", benefit="on-call allowance", base="weekly base pay",
               counted=["time spent responding to a callout", "travel time to site for a callout", "time on a callout that ran past midnight"],
               excluded=["standby time at home without a callout", "time on a public-holiday rota that is paid separately", "training attended during the on-call week"],
               exempt=("the employee was on a fixed-term contract of less than three months", "Fixed-term contract under three months")),
    "legal": dict(measure="Overdue period", unit="days", qty=(2, 40), kind="pct", benefit="late payment interest", base="invoice total",
                  counted=["days after the due date while the invoice was undisputed", "days after a dispute was resolved in the firm's favour", "days after a failed direct debit"],
                  excluded=["days while a written dispute was open", "days after a payment plan was agreed", "days during which the firm had put the matter on hold"],
                  exempt=("the client is a registered charity with a fee-waiver agreement", "Charity fee waiver")),
    "retail": dict(measure="Days of use", unit="days", qty=(1, 25), kind="pct", benefit="restocking fee", base="purchase price",
                   counted=["days the item was in the customer's possession", "days the item was on loan to a family member", "days in the customer's possession after a repair"],
                   excluded=["days the item was with the repair partner", "days in transit back to the warehouse", "days the item was held in store awaiting collection"],
                   exempt=("the item was faulty on arrival", "Faulty on arrival")),
    "finance": dict(measure="Charged overdraft days", unit="days", qty=(1, 12), kind="fixed", benefit="monthly overdraft charge", base=None,
                    counted=["days overdrawn beyond the fee-free buffer", "days overdrawn after the arranged limit expired", "days overdrawn on an unarranged basis"],
                    excluded=["days overdrawn within the fee-free buffer", "days overdrawn because of a bank error", "days overdrawn while a salary credit was in the clearing cycle"],
                    exempt=("the client is registered under the financial hardship scheme", "Hardship scheme")),
    "healthcare_admin": dict(measure="Chargeable pages", unit="pages", qty=(5, 90), kind="fixed", benefit="records copying charge", base=None,
                             counted=["pages of clinical notes", "pages of imaging reports", "pages of correspondence with other providers"],
                             excluded=["duplicate pages", "pages already supplied free of charge in the last 12 months", "cover sheets and indexes"],
                             exempt=("the request is made by a patient receiving palliative care", "Palliative care")),
    "public_sector": dict(measure="Assessable floor area", unit="square metres", qty=(8, 160), kind="fixed", benefit="application fee", base=None,
                          counted=["internal floor area", "mezzanine storage", "enclosed conservatory"],
                          excluded=["open external terraces", "car parking bays", "plant rooms"],
                          exempt=("the applicant is a registered community interest organisation", "Community organisation")),
    "insurance": dict(measure="Covered delay", unit="hours", qty=(1, 20), kind="fixed", benefit="travel delay benefit", base=None,
                      counted=["delay caused by strike action", "delay caused by mechanical breakdown of the aircraft", "delay caused by severe weather"],
                      excluded=["delay caused by the insured person missing check-in", "delay announced before the trip was booked", "delay on a connecting journey booked separately"],
                      exempt=("the trip was booked after the policy lapsed", "Booked after lapse")),
}


def b_eval(p, c, ignore=frozenset()):
    counted = set(p["counted"]) if "excl" not in ignore else set(p["counted"]) | set(p["excluded"])
    meas = sum(q for t, q in c["comps"] if t in counted)
    band = max(i for i, t in enumerate(p["thresholds"]) if (meas >= t if "boundary" not in ignore else meas > t or t == 0))
    if c["tier"] == p["special_tier"] and "special" not in ignore:
        vals = p["special"]
    elif p["amend"] and c["event"] >= p["amend"]["date"] and "amend" not in ignore:
        vals = p["amend"]["values"]
    else:
        vals = p["standard"]
    v = Decimal(vals[band])
    amt = q2(v * c["base"] / 100) if p["kind"] == "pct" else q2(v)
    if p["cap"] is not None and "cap" not in ignore:
        capv = q2(Decimal(p["cap"]) * c["base"] / 100) if p["kind"] == "pct" else q2(Decimal(p["cap"]))
        amt = min(amt, capv)
    if c["exempt"] and "exempt" not in ignore:
        amt = q2(0)
    return {"measure": meas, "band": band, "amount": amt}


B_TWISTS = ["excl", "boundary", "special", "amend", "cap", "exempt"]


def _b_params(rng, dom_id, difficulty):
    cfg = BANDED[dom_id]
    dom = DOMAINS[dom_id]
    lo, hi = cfg["qty"]
    nb = rng.choice([3, 4, 4, 5])
    step = max(1, (hi * 2) // nb)
    th = [0]
    for _ in range(nb - 1):
        th.append(th[-1] + rng.randint(max(1, step // 2), step))
    if cfg["kind"] == "pct":
        std = sorted(rng.sample([0, 5, 10, 15, 20, 25, 30, 40, 50], nb))
        if std[0] != 0 and chance(rng, 0.5):
            std[0] = 0
        spec_ = [min(100, x + rng.choice([5, 10, 15])) for x in std]
        am = [x + rng.choice([-5, 5, 10]) if x else 0 for x in std]
        am = sorted(max(0, a) for a in am)
        spec_ = sorted(spec_)
        cap = rng.choice([None, 20, 25, 30, 35]) if difficulty >= 4 else rng.choice([None, None, 30])
    else:
        unitp = {"finance": (5, 12), "healthcare_admin": (8, 25), "public_sector": (60, 200), "insurance": (40, 120)}[dom_id]
        base_v = rng.randint(*unitp)
        std = [0 if (i == 0 and chance(rng, 0.4)) else base_v * (i + 1) + rng.randint(0, base_v // 2) for i in range(nb)]
        spec_ = [x // 2 for x in std] if dom_id in ("finance", "healthcare_admin", "public_sector") else [x + base_v for x in std]
        am = sorted(x + rng.choice([2, 5, 10]) if x else 0 for x in std)
        spec_ = sorted(spec_)
        cap = None
    y0 = rng.choice([2025, 2026, 2027])
    p = {"dom": dom_id, "thresholds": th, "standard": std, "special": spec_, "special_tier": rng.choice(dom["tiers"][2:]),
         "cap": cap, "kind": cfg["kind"], "counted": rng.sample(cfg["counted"], 3), "excluded": rng.sample(cfg["excluded"], 3),
         "amend": None, "year": y0}
    if difficulty >= 4 or chance(rng, 0.5):
        p["amend"] = {"date": dt.date(y0, rng.randint(2, 11), rng.randint(1, 28)), "values": am}
    return p


def _b_case(rng, p, dom_id, difficulty):
    cfg = BANDED[dom_id]
    dom = DOMAINS[dom_id]
    lo, hi = cfg["qty"]
    k = rng.randint(3, 6)
    comps = []
    types = p["counted"] + p["excluded"]
    for i in range(k):
        t = rng.choice(p["counted"]) if i < 2 else rng.choice(types)
        comps.append((t, rng.randint(lo, hi)))
    rng.shuffle(comps)
    # sometimes land exactly on a threshold (difficulty 5): adjust the first counted component
    meas = sum(q for t, q in comps if t in p["counted"])
    if difficulty == 5 and chance(rng, 0.5):
        t_hit = min((t for t in p["thresholds"][1:]), key=lambda t: abs(t - meas))
        delta = t_hit - meas
        for j, (t, q) in enumerate(comps):
            if t in p["counted"] and q + delta >= 1:
                comps[j] = (t, q + delta)
                break
    ev = (p["amend"]["date"] + dt.timedelta(days=rng.randint(-45, 45))) if p["amend"] else dt.date(p["year"], rng.randint(1, 12), rng.randint(1, 28))
    base = Decimal(rng.randint(30, 4000)) + Decimal(rng.choice([0, 0, 50, 99])) / 100 if cfg["kind"] == "pct" else None
    return {"comps": comps, "tier": rng.choice(dom["tiers"]), "event": ev, "base": base if base is not None else Decimal(0),
            "exempt": chance(rng, 0.12 if difficulty < 5 else 0.2)}


B_KEY = {"banded_amount": lambda r: str(r["amount"]), "banded_band": lambda r: r["band"], "banded_over": lambda r: str(r["amount"])}


def gen_banded(rng, difficulty, kind, want_unknown):
    dom_id = rng.choice(list(BANDED))
    dom = DOMAINS[dom_id]
    cfg = BANDED[dom_id]
    need = {3: 1, 4: 2, 5: 2}[difficulty]
    fk = B_KEY[kind]
    for _ in range(300):
        p = _b_params(rng, dom_id, difficulty)
        c = _b_case(rng, p, dom_id, difficulty)
        g = b_eval(p, c)
        if kind == "banded_band" and c["exempt"]:
            continue
        if g["measure"] > p["thresholds"][-1] * 1.8:
            continue
        wrong = {t: b_eval(p, c, frozenset([t])) for t in B_TWISTS}
        diff = [t for t, w in wrong.items() if fk(w) != fk(g)]
        if len(diff) >= need and (kind != "banded_amount" or g["amount"] > 0 or chance(rng, 0.2)):
            break
    else:
        raise ValueError("no interesting banded case")
    doc_kind = rng.choice(["terms", "contract", "schedule", "policy"])
    rs = rng.random()
    text, doc, ref, pname, decisive = _b_render(random.Random(rs), dom_id, dom, cfg, p, c, difficulty, doc_kind)
    field, gold, hints, ov = _b_question(random.Random(rs + 1), kind, p, c, g, wrong, doc, ref, pname, cfg)
    spec = _b_spec(p, c) | {"kind": kind, "opt_values": ov}
    if kind == "banded_over":
        spec["x"] = hints.pop("_x")
    items = [item(text, field, gold, kind, dom_id, spec, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _b_unknown(rng, p, c, kind, dom_id, dom, cfg, difficulty, doc_kind, spec.get("x"))
        if u is None:
            raise ValueError("no banded unknown")
        items.append(u)
    return items


def _b_spec(p, c):
    pp = {k: v for k, v in p.items() if k != "amend"}
    pp["amend"] = {"date": _iso(p["amend"]["date"]), "values": p["amend"]["values"]} if p["amend"] else None
    return {"schema": "banded", "p": pp, "c": {"comps": [list(x) for x in c["comps"]], "tier": c["tier"], "event": _iso(c["event"]),
                                               "base": str(c["base"]), "exempt": c["exempt"]}}


def _b_render(rng, dom_id, dom, cfg, p, c, difficulty, doc_kind, missing=None, missing_idx=None):
    doc = Doc(rng, dom_id, doc_kind)
    M, unit, ben = cfg["measure"], cfg["unit"], cfg["benefit"]
    ben_c = ben[0].upper() + ben[1:]
    counted, excluded = p["counted"], p["excluded"]
    th = p["thresholds"]
    # definitions
    defs = [("d_measure", P(rng, f"\"{M}\" means the total number of {unit} of {counted[0]}, {counted[1]} and {counted[2]} recorded for the case.",
                            f"\"{M}\" is calculated by adding together the {unit} of: (a) {counted[0]}; (b) {counted[1]}; and (c) {counted[2]}.",
                            f"For this document, {M} consists of {counted[0]}, {counted[1]} and {counted[2]}, measured in {unit}.")),
            P(rng, f"\"Case Record\" means the entry held for the case in {doc.ctx['system']}.",
              f"\"Statement Period\" means a calendar month.", f"\"Account\" means the account under which the relevant service is provided.")]
    if p["amend"]:
        defs.append(P(rng, f"\"Relevant Date\" means the date on which the {M.lower()} began.", f"The \"Relevant Date\" of a case is the date the {M.lower()} began."))
    rng.shuffle(defs)
    sec_def = doc.add(P(rng, "Definitions", "Interpretation", "Defined terms"), defs)
    # calculation
    tabA = "{ref:sched_a}"
    calc = [("calc", P(rng, f"The {ben} is determined by the band in {tabA} into which the {M} falls.",
                       f"The {ben} for a case is the amount shown in {tabA} for the band containing the {M}.",
                       f"To find the {ben}, identify the band of {tabA} that contains the case's {M}."))]
    calc.append(("bound", P(rng, f"A {M} that is exactly equal to the lower limit of a band falls into that band.",
                            f"Band limits are inclusive at the lower end: a {M} equal to a band's starting value belongs to that band.",
                            f"Where the {M} equals the figure at which a band starts, the higher band applies.")))
    if cfg["kind"] == "pct":
        calc.append(P(rng, f"Percentages in the schedules are applied to the {cfg['base']} for the period concerned and rounded to two decimal places.",
                      f"The {ben} is the percentage shown, multiplied by the {cfg['base']}, rounded to the nearest cent."))
    if p["cap"] is not None:
        calc.append(("cap", P(rng, f"The {ben} for a single case may not exceed {p['cap']}% of the {cfg['base']}.",
                              f"In no case will the {ben} be more than {p['cap']}% of the {cfg['base']}; any excess is disregarded.",
                              f"The {ben} is capped at {p['cap']}% of the {cfg['base']}.")))
    calc.append(("special", P(rng, f"Where the {dom['tier_name']} is {p['special_tier']}, {{ref:sched_b}} applies instead of {tabA}.",
                              f"{dom['parties'].capitalize()} on the {p['special_tier']} {dom['tier_name']} are assessed under {{ref:sched_b}}, not {tabA}.",
                              f"{tabA} does not apply to the {p['special_tier']} {dom['tier_name']}; use {{ref:sched_b}} for those cases.")))
    calc.append(("exempt", P(rng, f"No {ben} is payable where {cfg['exempt'][0]}." if cfg['kind'] == 'pct' and dom_id not in ('retail', 'legal') else f"The {ben} is not charged where {cfg['exempt'][0]}.",
                             f"Where {cfg['exempt'][0]}, the {ben} is zero." )))
    rng.shuffle(calc)
    sec_calc = doc.add(P(rng, f"Calculating the {ben}", f"How the {ben} is worked out", f"{ben_c}", "Calculation"), calc)
    excl = [("excl", P(rng, f"For the avoidance of doubt, the following do not count towards the {M}: {excluded[0]}; {excluded[1]}; {excluded[2]}.",
                       f"The {M} does not include {excluded[0]}, {excluded[1]} or {excluded[2]}, even if these are shown on the Case Record.",
                       f"Excluded from the {M}: (i) {excluded[0]}; (ii) {excluded[1]}; (iii) {excluded[2]}.")),
            P(rng, f"Where the Case Record does not state the cause of a period, the {doc.ctx['role']} asks the {dept_or(doc)} team to confirm it before the {ben} is calculated.",
              f"Periods are measured from the timestamps in {doc.ctx['system']}; manual adjustments must be approved by the {doc.ctx['role2']}.")]
    sec_excl = doc.add(P(rng, "Exclusions", "What does not count", "Exclusions and adjustments"), excl)
    sec_am = None
    if p["amend"]:
        sec_am = doc.add(P(rng, "Amendments", "Amendment log", "Changes since first issue"),
                         [("amend", P(rng, f"For cases whose Relevant Date is on or after {doc.fd(p['amend']['date'])}, {tabA} is replaced by {{ref:sched_a1}}. {{ref:sched_b}} is not affected by this change.",
                                      f"With effect from {doc.fd(p['amend']['date'])} (by Relevant Date), the values in {{ref:sched_a1}} apply in place of {tabA}; the {p['special_tier']} schedule is unchanged."))])

    def table(vals, title):
        rows = [f"{title}", f"| Band | {M} ({unit}) | {ben_c} |", "|---|---|---|"]
        for i, v in enumerate(vals):
            rng_txt = f"{th[i]} to less than {th[i + 1]}" if i + 1 < len(th) else f"{th[i]} or more"
            val = f"{v}% of the {cfg['base']}" if cfg["kind"] == "pct" else doc.m(v)
            rows.append(f"| {i + 1} | {rng_txt} | {val} |")
        return "\n".join(rows)
    sched = [("sched_a", table(p["standard"], P(rng, "Standard table:", f"Table of {ben} bands (standard):", "Standard rates:"))),
             ("sched_b", table(p["special"], f"{p['special_tier']} table:"))]
    if p["amend"]:
        sched.append(("sched_a1", table(p["amend"]["values"], P(rng, "Revised standard table:", "Replacement standard table:"))))
    if missing == "tables":
        sched = [(a, t if a != missing_idx else f"[{t.splitlines()[0][:-1]} — not included in this extract]") for a, t in sched]
    sec_s = doc.add(P(rng, "Schedules", "Tables", "Rate schedules"), sched)
    doc.boiler(rng.randint(4, 7))
    core = [sec_def, sec_calc, sec_excl]
    doc.order(core, boiler_front=rng.randint(1, 2))
    tail = [s for s in (sec_am, sec_s) if s]
    doc.sections = [s for s in doc.sections if s not in tail] + tail
    eff = dt.date(p["year"] - 1, rng.randint(1, 12), 1)
    doc.front = front_matter(doc, P(rng, *dom["doc_nouns"]), eff, f"{rng.randint(1, 5)}.{rng.randint(0, 9)}")
    ref = code(rng, dom["ref"])
    pname = doc.people()
    fields = [("Reference", ref), (dom["party"].capitalize(), pname),
              (dom["tier_name"].capitalize(), c["tier"] if missing != "tier" else "not recorded"),
              (f"Relevant Date" if p["amend"] else "Date", doc.fd(c["event"]) if missing != "event" else "not recorded"),
              (cfg["exempt"][1], "yes" if c["exempt"] else "no")]
    if cfg["kind"] == "pct":
        fields.append((cfg["base"].capitalize(), doc.m(c["base"]) if missing != "base" else "see order form (not attached)"))
    rng.shuffle(fields[2:])
    log = [f"| # | Period | {unit.capitalize()} |", "|---|---|---|"]
    for j, (t, q) in enumerate(c["comps"]):
        tt = t if not (missing == "comp" and j == missing_idx) else "cause not recorded"
        qq = q if not (missing == "compq" and j == missing_idx) else "not recorded"
        log.append(f"| {j + 1} | {tt} | {qq} |")
    doc.tail = render_record(rng, P(rng, "Case record", f"Claim {ref}", "Case summary"), fields) + ["", P(rng, "Logged periods:", "Period log:", f"Breakdown of recorded {unit}:")] + log
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("d_measure", "excl", "special", "cap", "exempt", "amend", "bound") if doc.rendered(a)]
    return text, doc, ref, pname, decisive


def dept_or(doc):
    return doc.ctx["dept2"]


def _b_question(rng, kind, p, c, g, wrong, doc, ref, pname, cfg):
    ben = cfg["benefit"]
    if kind == "banded_amount":
        q = P(rng, f"What {ben} applies to {ref}?", f"Under the document, how much is the {ben} for {pname}'s case ({ref})?",
              f"Which figure is the correct {ben} for case {ref}?")
        cands = [w["amount"] for w in wrong.values()]
        # neighbouring bands under the applicable table
        tabs = [p["standard"], p["special"]] + ([p["amend"]["values"]] if p["amend"] else [])
        for b in (g["band"] - 1, g["band"] + 1, g["band"]):
            for vals in tabs:
                if 0 <= b < len(p["thresholds"]):
                    v = Decimal(vals[b])
                    cands.append(q2(v * c["base"] / 100) if p["kind"] == "pct" else q2(v))
        field, gold, ov = cf(rng, q, g["amount"], cands, lambda x: doc.m(x, cents=True), k=rng.randint(4, 6))
        shown = {o["text"] for o in field["options"]}
        near = [doc.m(w["amount"], cents=True) for w in wrong.values() if w["amount"] != g["amount"] and doc.m(w["amount"], cents=True) not in shown]
        return field, gold, {"near_miss": list(dict.fromkeys(near))[:3]}, ov
    if kind == "banded_band":
        th = p["thresholds"]
        lv = [f"Band {i + 1}: {cfg['measure']} of {th[i]} to less than {th[i + 1]} {cfg['unit']}" if i + 1 < len(th) else
              f"Band {i + 1}: {cfg['measure']} of {th[i]} {cfg['unit']} or more" for i in range(len(th))]
        q = P(rng, f"Which band does the {cfg['measure']} for {ref} fall into?", f"Into which band of the schedules does case {ref} fall?",
              f"What band applies to {pname}'s case ({ref})?")
        return score_field(q, lv), g["band"], {}, None
    # banded_over
    others = sorted({w["amount"] for w in wrong.values() if w["amount"] != g["amount"]})
    o = rng.choice(others)
    x = q2((g["amount"] + o) / 2)
    if x == g["amount"]:
        raise ValueError("degenerate threshold")
    q = P(rng, f"Is the {ben} for {ref} more than {doc.m(x, cents=True)}?", f"Does the {ben} due on case {ref} exceed {doc.m(x, cents=True)}?")
    neg = P(rng, f"Is the {ben} for {ref} {doc.m(x, cents=True)} or less?", f"Does the {ben} due on case {ref} stay at or below {doc.m(x, cents=True)}?")
    return noul_field(q), g["amount"] > x, {"neg": {"question": neg, "gold": not (g["amount"] > x)}, "_x": str(x)}, None


def _b_unknown(rng, p, c, kind, dom_id, dom, cfg, difficulty, doc_kind, x):
    fk = B_KEY[kind] if kind != "banded_over" else (lambda r: r["amount"] > Decimal(x))
    lo, hi = cfg["qty"]
    opts = ["comp", "compq", "tier", "event", "base", "tables"]
    rng.shuffle(opts)
    for miss in opts:
        idx, alts = None, []
        if miss in ("comp", "compq"):
            idx = rng.randrange(len(c["comps"]))
            t0, q0 = c["comps"][idx]
            if miss == "comp":
                alts = [dict(c, comps=c["comps"][:idx] + [(t, q0)] + c["comps"][idx + 1:]) for t in p["counted"] + p["excluded"]]
            else:
                if t0 not in p["counted"]:
                    continue
                alts = [dict(c, comps=c["comps"][:idx] + [(t0, q)] + c["comps"][idx + 1:]) for q in range(lo, hi * 3)]
        elif miss == "tier":
            alts = [dict(c, tier=t) for t in dom["tiers"]]
        elif miss == "event":
            if not p["amend"]:
                continue
            alts = [dict(c, event=p["amend"]["date"] + dt.timedelta(days=k)) for k in (-30, -1, 0, 30)]
        elif miss == "base":
            if p["kind"] != "pct" or kind == "banded_band":
                continue
            alts = [dict(c, base=Decimal(b)) for b in (30, 100, 500, 1000, 2500, 4000)]
        elif miss == "tables":
            # the schedule actually used is not reproduced
            if c["tier"] == p["special_tier"]:
                idx = "sched_b"; key = "special"
            elif p["amend"] and c["event"] >= p["amend"]["date"]:
                idx = "sched_a1"; key = None
            else:
                idx = "sched_a"; key = "standard"
            if kind == "banded_band" or c["exempt"]:
                continue
            alts = []
            for shift in (0, 5, 10, 20, 40):
                pp = dict(p)
                if key:
                    pp[key] = [v + shift for v in p[key]]
                else:
                    pp["amend"] = dict(p["amend"], values=[v + shift for v in p["amend"]["values"]])
                alts.append(("p", pp))
        vals = set()
        for a in alts:
            pp, cc = (a[1], c) if isinstance(a, tuple) else (p, a)
            vals.add(str(fk(b_eval(pp, cc))))
        if len(vals) < 2:
            continue
        rs = rng.random()
        _ov = None
        text, doc, ref, pname, _ = _b_render(random.Random(rs), dom_id, dom, cfg, p, c, difficulty, doc_kind, missing=miss, missing_idx=idx)
        g = b_eval(p, c)
        wrong = {t: b_eval(p, c, frozenset([t])) for t in B_TWISTS}
        if kind == "banded_over":
            f2 = noul_field(P(rng, f"Is the {cfg['benefit']} for {ref} more than {doc.m(Decimal(x), cents=True)}?",
                              f"Does the {cfg['benefit']} due on case {ref} exceed {doc.m(Decimal(x), cents=True)}?"))
        else:
            f2, _, _, _ov = _b_question(random.Random(rs + 1), kind, p, c, g, wrong, doc, ref, pname, cfg)
        spec = _b_spec(p, c) | {"missing": miss, "kind": kind} | ({"x": x} if x is not None else {})
        ext = {"kind": kind} | ({"x": x} if x is not None else {})
        _trace(spec, [_b_spec(*_pc(a, p, c)) | ext for a in alts] + [_b_spec(p, c) | ext], _ov)
        return item(text, f2, None, kind, dom_id, spec, {}, child_of=0)
    return None


def recheck_banded(spec):
    if spec.get("missing"):
        return None
    p, c = spec["p"], spec["c"]
    total = 0
    for t, q in c["comps"]:
        if t in p["counted"]:
            total += q
    band = 0
    for i, t in enumerate(p["thresholds"]):
        if total >= t:
            band = i
    if spec["kind"] == "banded_band":
        return band
    if c["tier"] == p["special_tier"]:
        table = p["special"]
    elif p["amend"] and dt.date.fromisoformat(c["event"]) >= dt.date.fromisoformat(p["amend"]["date"]):
        table = p["amend"]["values"]
    else:
        table = p["standard"]
    base = Decimal(c["base"])
    amount = (Decimal(table[band]) * base / 100) if p["kind"] == "pct" else Decimal(table[band])
    amount = amount.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
    if p["cap"] is not None:
        amount = min(amount, (Decimal(p["cap"]) * base / 100).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))
    if c["exempt"]:
        amount = Decimal("0.00")
    if spec["kind"] == "banded_over":
        return amount > Decimal(spec["x"])
    return str(amount)


# ================================================================================================ authority
SPEND = {"retail": ("store fit-out purchase", "supplier"), "logistics": ("subcontractor spend", "subcontractor"), "hr": ("training spend", "training provider"),
         "healthcare_admin": ("equipment purchase", "supplier"), "insurance": ("loss adjuster instruction", "adjusting firm"), "finance": ("marketing spend", "agency"),
         "saas_ops": ("vendor purchase", "vendor"), "travel": ("ground handling purchase", "handling agent"), "public_sector": ("procurement", "supplier"),
         "legal": ("disbursement", "payee")}


def a_eval(p, c, ignore=frozenset()):
    th = p["amend"]["th"] if p["amend"] and c["date"] >= p["amend"]["date"] and "amend" not in ignore else p["th"]
    eff = c["amount"]
    if "agg" not in ignore:
        for r in c["related"]:
            if r["counter"] == c["counter"] and r["requester"] == c["requester"] and 0 <= (c["date"] - r["date"]).days <= p["agg_days"]:
                eff += r["amount"]
    lvl = len(th)
    for i, t in enumerate(th):
        if (eff <= t) if "boundary" not in ignore else (eff < t):
            lvl = i
            break
    if c["category"] in p["cats"] and "cat" not in ignore:
        lvl = max(lvl, p["cat_level"])
    valid = None
    ap = c.get("approver")
    if ap:
        if ap["name"] == c["requester"] and "self" not in ignore:
            valid = False
        elif ap["level"] >= lvl:
            valid = True
        else:
            valid = False
            if "deleg" not in ignore:
                for d in p["deleg"]:
                    if d["delegate"] == ap["name"] and d["for_level"] == lvl and d["start"] <= c["approved_on"] <= d["end"] and eff <= d["limit"]:
                        valid = True
    return {"level": lvl, "eff": eff, "valid": valid}


A_TWISTS = ["agg", "boundary", "cat", "amend", "deleg", "self"]


def _a_params(rng, dom_id, dom, difficulty, people):
    unit = rng.choice([500, 1000, 2500, 5000])
    th = sorted(rng.sample([unit * k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)], 3))
    y0 = rng.choice([2025, 2026, 2027])
    cats = rng.sample(dom["categories"], 6)
    p = {"th": th, "cats": cats[:rng.randint(1, 2)], "other_cats": cats[2:], "cat_level": rng.choice([2, 3]),
         "agg_days": rng.choice([7, 14, 30, 60, 90]), "amend": None, "deleg": [], "year": y0, "unit": unit}
    if difficulty >= 4 or chance(rng, 0.5):
        k = rng.randrange(3)
        new = list(th)
        new[k] = max(unit // 2, th[k] + rng.choice([-1, 1]) * unit * rng.choice([1, 2]))
        if sorted(new) == new and len(set(new)) == 3 and new != th:
            p["amend"] = {"date": dt.date(y0, rng.randint(2, 11), rng.randint(1, 28)), "th": new}
    return p


def _a_case(rng, p, dom, difficulty, people, kind):
    spend, counter = SPEND[next(k for k, v in DOMAINS.items() if v is dom)]
    date = (p["amend"]["date"] + dt.timedelta(days=rng.randint(-30, 30))) if p["amend"] else dt.date(p["year"], rng.randint(1, 12), rng.randint(1, 28))
    requester = people()
    counters = [f"{rng.choice(['Apex', 'Brightline', 'Cedar', 'Delta', 'Evergreen', 'Fulcrum', 'Granite', 'Horizon'])} {rng.choice(['Services', 'Supplies', 'Ltd', 'Partners', 'Group'])}" for _ in range(3)]
    counters = list(dict.fromkeys(counters))
    cnt = counters[0]
    t = rng.choice(p["th"])
    amt = Decimal(t) if chance(rng, 0.15) else Decimal(max(50, t + rng.randint(-p["unit"], p["unit"] // 2))) + Decimal(rng.choice([0, 0, 50])) / 100
    related = []
    others = [people() for _ in range(2)]
    for _ in range(rng.randint(1, 4) if difficulty >= 4 else rng.randint(0, 2)):
        related.append({"date": date - dt.timedelta(days=rng.choice([0, 2, 5, p["agg_days"] - 1, p["agg_days"], p["agg_days"] + 1, p["agg_days"] + 9, 3])),
                        "counter": rng.choice(counters), "requester": rng.choice([requester, requester, others[0]]),
                        "amount": Decimal(rng.randint(p["unit"] // 5, p["unit"] * 3))})
    cat = rng.choice(p["cats"]) if chance(rng, 0.4) else rng.choice(p["other_cats"])
    c = {"date": date, "requester": requester, "counter": cnt, "amount": amt, "related": related, "category": cat, "approver": None,
         "approved_on": date + dt.timedelta(days=rng.randint(0, 5))}
    return c


def gen_authority(rng, difficulty, kind, want_unknown):
    dom_id = rng.choice(DOMAIN_IDS)
    dom = DOMAINS[dom_id]
    need = {3: 1, 4: 2, 5: 2}[difficulty]
    from .common import People
    for _ in range(300):
        people = People(rng)
        p = _a_params(rng, dom_id, dom, difficulty, people)
        c = _a_case(rng, p, dom, difficulty, people, kind)
        roles = dom["roles"][:4]
        g0 = a_eval(p, c)
        lvl = g0["level"]
        # delegation: the holder of the required level (if not the top) is absent for a period
        holders = {i: people() for i in range(4)}
        if lvl >= 1 and (difficulty >= 4 or chance(rng, 0.5)):
            start = c["approved_on"] - dt.timedelta(days=rng.randint(0, 6))
            end = start + dt.timedelta(days=rng.randint(3, 14))
            if chance(rng, 0.3):
                start, end = c["approved_on"] + dt.timedelta(days=1), c["approved_on"] + dt.timedelta(days=rng.randint(3, 9))
            lim = rng.choice([g0["eff"] - Decimal(rng.randint(1, 300)), g0["eff"] + Decimal(rng.randint(0, 3000))])
            p["deleg"] = [{"absent": holders[lvl], "for_level": lvl, "delegate": holders[lvl - 1], "start": start, "end": end, "limit": q2(max(Decimal(100), lim))}]
        if kind == "authority_valid":
            opts = [("exact", lvl), ("below", lvl - 1), ("above", lvl + 1)]
            if p["deleg"]:
                opts += [("delegate", lvl - 1)] * 3
            if difficulty == 5:
                opts.append(("self", None))
            how, al = rng.choice(opts)
            if how == "self":
                c["approver"] = {"name": c["requester"], "level": rng.choice([lvl, min(3, lvl + 1)]), "how": how}
            elif al is None or not 0 <= al <= 3:
                continue
            else:
                c["approver"] = {"name": holders[al], "level": al, "how": how}
        g = a_eval(p, c)
        wrong = {t: a_eval(p, c, frozenset([t])) for t in A_TWISTS}
        key = "valid" if kind == "authority_valid" else "level"
        diff = [t for t, w in wrong.items() if w[key] != g[key]]
        if len(diff) >= need:
            break
    else:
        raise ValueError("no interesting authority case")
    doc_kind = rng.choice(["policy", "sop", "policy", "schedule"])
    rs = rng.random()
    text, doc, ref, decisive = _a_render(random.Random(rs), dom_id, dom, p, c, holders, difficulty, doc_kind)
    field, gold, hints, ov = _a_question(random.Random(rs + 1), kind, p, c, g, wrong, doc, ref, dom, holders)
    spec = _a_spec(p, c) | {"kind": kind, "opt_values": ov}
    items = [item(text, field, gold, kind, dom_id, spec, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _a_unknown(rng, p, c, kind, dom_id, dom, holders, difficulty, doc_kind)
        if u is None:
            raise ValueError("no authority unknown")
        items.append(u)
    return items


def _a_spec(p, c):
    ser = lambda v: _iso(v) if isinstance(v, dt.date) else (str(v) if isinstance(v, Decimal) else v)
    pp = {k: v for k, v in p.items() if k not in ("amend", "deleg", "other_cats")}
    pp["amend"] = {"date": _iso(p["amend"]["date"]), "th": p["amend"]["th"]} if p["amend"] else None
    pp["deleg"] = [{k: ser(v) for k, v in d.items()} for d in p["deleg"]]
    cc = {k: ser(v) for k, v in c.items() if k not in ("related", "approver")}
    cc["related"] = [{k: ser(v) for k, v in r.items()} for r in c["related"]]
    cc["approver"] = {k: v for k, v in c["approver"].items() if k != "how"} if c["approver"] else None
    return {"schema": "authority", "p": pp, "c": cc}


def _a_render(rng, dom_id, dom, p, c, holders, difficulty, doc_kind, missing=None, miss_idx=None):
    doc = Doc(rng, dom_id, doc_kind)
    spend, counter = SPEND[dom_id]
    roles = dom["roles"][:4]
    th = p["th"]
    m = doc.m
    band_rows = [f"| Amount of the {spend} | Lowest approver |", "|---|---|",
                 f"| up to and including {m(th[0])} | {roles[0]} |", f"| more than {m(th[0])}, up to and including {m(th[1])} | {roles[1]} |",
                 f"| more than {m(th[1])}, up to and including {m(th[2])} | {roles[2]} |", f"| more than {m(th[2])} | {roles[3]} |"]
    ladder = " < ".join(roles)
    core = [("bands", P(rng, f"Each {spend} must be approved before it is committed by an approver at or above the level shown in the table below.",
                        f"A {spend} may only be committed once approved by the lowest approver shown below for its amount, or by a more senior approver.",
                        f"Approval authority for a {spend} depends on its amount, as follows.") + "\n" + "\n".join(band_rows)),
            ("ladder", P(rng, f"Seniority runs, from lowest to highest: {ladder}. Any more senior approver may approve in place of a junior one.",
                         f"For this document the order of seniority is {ladder}; an approver may approve anything a less senior approver could approve.")),
            ("self", P(rng, f"No one may approve a {spend} that they requested themselves, whatever their level.",
                       f"An approval given by the person who raised the request is void, regardless of seniority.",
                       f"Self-approval is not permitted: the approver must be a different person from the requester."))]
    cat_txt = ", ".join(p["cats"])
    core2 = [("cat", P(rng, f"Whatever the amount, a {spend} relating to {cat_txt} requires approval at {roles[p['cat_level']]} level or above.",
                       f"{cat_txt[0].upper() + cat_txt[1:]}: approval by the {roles[p['cat_level']]} (or a more senior approver) is always required, even below the thresholds in {{ref:bands}}.",
                       f"The thresholds in {{ref:bands}} do not reduce the approval level for {cat_txt}, which must always go to at least the {roles[p['cat_level']]}.")),
             ("agg", P(rng, f"To prevent requests being split, the amount of a {spend} includes every other {spend} raised by the same requester for the same {counter} on the same day or in the {p['agg_days']} days before it.",
                       f"Related requests are added together: when a requester has raised other {spend}s for the same {counter} within the {p['agg_days']} days before (or on the same day as) a request, the approval level is set by the combined amount.",
                       f"For the purposes of {{ref:bands}}, the amount is the total of the request and any earlier {spend} by the same requester to the same {counter} dated no more than {p['agg_days']} days before it.")),
             P(rng, f"For {rng.choice(p['other_cats'])}, a quote must be attached before approval is requested.",
               f"Recurring {spend}s for {rng.choice(p['other_cats'])} are reviewed every quarter by the {doc.ctx['dept2']} team.",
               f"A {spend} for {rng.choice(p['other_cats'])} must name a budget code, but this does not change who approves it.")]
    rng.shuffle(core2)
    deleg_lines = []
    for d in p["deleg"]:
        if missing == "deleg":
            deleg_lines.append(f"[The delegation rota for this period is held separately and is not attached.]")
        else:
            deleg_lines.append(f"| {d['absent']} ({roles[d['for_level']]}) | {doc.fd(d['start'])} to {doc.fd(d['end'])} | {d['delegate']} ({roles[d['for_level'] - 1]}) | {m(d['limit'], cents=True)} |")
    deleg_txt = P(rng, "During a recorded absence, the named delegate may approve on behalf of the absent approver, but only within the dates and up to the limit shown in the delegation register. Outside those dates or above that limit, the delegate has only their own authority.",
                  "A delegate acts for an absent approver only between the dates in the delegation register and only up to the limit stated there; otherwise the delegate's own level applies.")
    sec_core = doc.add(P(rng, "Approval authority", "Who can approve", "Authorisation limits", "Delegated authority limits"), core)
    sec_core2 = doc.add(P(rng, "Special rules", "Additional requirements", "Aggregation and category rules", "Further conditions"), core2)
    reg = ["| Absent approver | Dates | Delegate | Limit |", "|---|---|---|---|"] + deleg_lines if p["deleg"] and missing != "deleg" else deleg_lines
    sec_del = doc.add(P(rng, "Delegation during absence", "Cover arrangements", "Absence cover"),
                      [("deleg", deleg_txt), ("reg", (P(rng, "Delegation register:", "Current delegation register:") + "\n" + "\n".join(reg)) if reg else "No delegations are currently registered.")])
    sec_am = None
    if p["amend"]:
        new = p["amend"]["th"]
        chg = next(i for i in range(3) if new[i] != th[i])
        sec_am = doc.add(P(rng, "Amendments", "Amendment record", "Changes"),
                         [("amend", P(rng, f"For requests dated on or after {doc.fd(p['amend']['date'])}, the figure of {m(th[chg])} in {{ref:bands}} is replaced by {m(new[chg])} wherever it appears. Earlier requests are unaffected.",
                                      f"With effect for requests dated {doc.fd(p['amend']['date'])} or later, {m(new[chg])} replaces {m(th[chg])} in the table in {{ref:bands}}."))])
    doc.boiler(rng.randint(4, 7), exclude=("roles",))
    doc.order([sec_core, sec_core2, sec_del], boiler_front=rng.randint(1, 2))
    if sec_am:
        doc.sections = [s for s in doc.sections if s is not sec_am] + [sec_am]
    doc.front = front_matter(doc, P(rng, f"{spend.capitalize()} Approval Policy", "Scheme of Delegated Authority", f"Authorisation Procedure for {spend}s"),
                             dt.date(p["year"] - 1, rng.randint(1, 12), 1), f"{rng.randint(1, 5)}.{rng.randint(0, 9)}")
    ref = code(rng, "REQ")
    fields = [("Request", ref), ("Requested by", c["requester"]), ("Date of request", doc.fd(c["date"]) if missing != "date" else "not recorded"),
              (counter.capitalize(), c["counter"]), ("Category", c["category"] if missing != "category" else "not stated"),
              ("Amount", m(c["amount"], cents=True)), ("Cost centre", f"{rng.randint(100, 999)}-{rng.choice(['OPS', 'ADM', 'FIN', 'SVC'])}")]
    if c["approver"]:
        fields.append(("Approved by", f"{c['approver']['name']} ({roles[c['approver']['level']] if 0 <= c['approver']['level'] < 4 else roles[-1]}) on {doc.fd(c['approved_on'])}"))
    tail = render_record(rng, P(rng, "Request form", f"Approval request {ref}", "Request details"), fields)
    if c["related"]:
        tail += ["", P(rng, f"Other {spend}s on file (last 120 days):", "Earlier requests found in the ledger:", "Related entries:"),
                 f"| Date | Requested by | {counter.capitalize()} | Amount |", "|---|---|---|---|"]
        for j, r in enumerate(c["related"]):
            dd = doc.fd(r["date"]) if not (missing == "rel_date" and j == miss_idx) else "date not recorded"
            tail.append(f"| {dd} | {r['requester']} | {r['counter']} | {m(r['amount'], cents=True)} |")
    doc.tail = tail
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("bands", "cat", "agg", "deleg", "amend", "self") if doc.rendered(a)]
    return text, doc, ref, decisive


def _a_question(rng, kind, p, c, g, wrong, doc, ref, dom, holders):
    roles = dom["roles"][:4]
    spend, _ = SPEND[doc.dom_id]
    if kind == "authority_role":
        q = P(rng, f"Who is the least senior approver who can approve request {ref}?", f"What is the lowest approval level required for {ref}?",
              f"Under the policy, which approver level does request {ref} need, at minimum?")
        extra = [r for r in dom["roles"][4:]]
        field, gold, ov = cf(rng, q, g["level"], [0, 1, 2, 3] + [4, 5][:len(extra)], lambda i: dom["roles"][i])
        return field, gold, {}, ov
    ap = c["approver"]
    q = P(rng, f"Was the approval of {ref} valid under the policy?", f"Is the approval recorded on request {ref} a valid approval?",
          f"Could {ap['name']} validly approve {ref}?")
    neg = P(rng, f"Was the approval of {ref} invalid under the policy?", f"Is the approval recorded on request {ref} void or outside authority?")
    return noul_field(q), g["valid"], {"neg": {"question": neg, "gold": not g["valid"]}}, None


def _a_unknown(rng, p, c, kind, dom_id, dom, holders, difficulty, doc_kind):
    key = "valid" if kind == "authority_valid" else "level"
    opts = ["rel_date", "category", "date", "deleg"]
    rng.shuffle(opts)
    for miss in opts:
        idx, alts = None, []
        if miss == "rel_date":
            if not c["related"]:
                continue
            idx = rng.randrange(len(c["related"]))
            alts = [dict(c, related=c["related"][:idx] + [dict(c["related"][idx], date=c["date"] - dt.timedelta(days=k))] + c["related"][idx + 1:])
                    for k in (0, p["agg_days"], p["agg_days"] + 1, 200)]
        elif miss == "category":
            alts = [dict(c, category=x) for x in p["cats"] + p["other_cats"]]
        elif miss == "date":
            if not p["amend"]:
                continue
            alts = [dict(c, date=p["amend"]["date"] + dt.timedelta(days=k)) for k in (-20, -1, 0, 20)]
        elif miss == "deleg":
            if not p["deleg"] or kind != "authority_valid":
                continue
            alts = [("p", dict(p, deleg=[dict(p["deleg"][0], limit=l)])) for l in (Decimal(100), Decimal(10 ** 7))] + [("p", dict(p, deleg=[]))]
        vals = set()
        for a in alts:
            pp, cc = (a[1], c) if isinstance(a, tuple) else (p, a)
            vals.add(a_eval(pp, cc)[key])
        if len(vals) < 2:
            continue
        rs = rng.random()
        text, doc, ref, _ = _a_render(random.Random(rs), dom_id, dom, p, c, holders, difficulty, doc_kind, missing=miss, miss_idx=idx)
        g = a_eval(p, c)
        f2, _, _, _ov = _a_question(random.Random(rs + 1), kind, p, c, g, {}, doc, ref, dom, holders)
        _trace(_a_spec(p, c) | {"missing": miss, "kind": kind}, [_a_spec(*_pc(a, p, c)) | {"kind": kind} for a in alts] + [_a_spec(p, c) | {"kind": kind}], _ov)
        return item(text, f2, None, kind, dom_id, _a_spec(p, c) | {"missing": miss, "kind": kind}, {}, child_of=0)
    return None


def recheck_authority(spec):
    if spec.get("missing"):
        return None
    p, c = spec["p"], spec["c"]
    D = dt.date.fromisoformat
    th = p["amend"]["th"] if p["amend"] and D(c["date"]) >= D(p["amend"]["date"]) else p["th"]
    total = Decimal(c["amount"]) + sum((Decimal(r["amount"]) for r in c["related"]
                                        if r["counter"] == c["counter"] and r["requester"] == c["requester"]
                                        and 0 <= (D(c["date"]) - D(r["date"])).days <= p["agg_days"]), Decimal(0))
    level = sum(1 for t in th if total > t)
    if c["category"] in p["cats"]:
        level = max(level, p["cat_level"])
    if spec["kind"] == "authority_role":
        return level
    a = c["approver"]
    if a["name"] == c["requester"]:
        return False
    if a["level"] >= level:
        return True
    return any(d["delegate"] == a["name"] and d["for_level"] == level and D(d["start"]) <= D(c["approved_on"]) <= D(d["end"])
               and total <= Decimal(d["limit"]) for d in p["deleg"])


# ================================================================================================ scope
SCOPE = {
    "retail": dict(term="High-Value Return", noun="return", enum="Reason code", enums=["damaged in transit", "changed mind", "wrong size", "faulty", "not as described", "late delivery"],
                   num="Item value", money=True, rng=(50, 3000), excl=("the return was processed through a marketplace partner", "Via marketplace partner"), pre="RET", notify="the Loss Prevention team"),
    "logistics": dict(term="Reportable Loss", noun="consignment", enum="Incident type", enums=["theft", "water ingress", "crush damage", "temperature excursion", "misdelivery", "short delivery"],
                      num="Declared value", money=True, rng=(200, 20000), excl=("the consignment was carried under a customer-arranged subcontract", "Customer-arranged subcontract"), pre="CN", notify="the insurer"),
    "hr": dict(term="Notifiable Absence", noun="absence", enum="Absence reason", enums=["sickness", "injury at work", "family emergency", "bereavement", "unauthorised absence", "medical appointment"],
               num="Consecutive working days", money=False, unit="working days", rng=(1, 25), excl=("the absence was pre-approved as annual leave", "Pre-approved leave"), pre="ABS", notify="the People Team"),
    "healthcare_admin": dict(term="Serious Incident", noun="incident report", enum="Category", enums=["medication error", "patient fall", "delayed diagnosis", "equipment failure", "records breach", "missed referral"],
                             num="Harm score", money=False, unit="points", rng=(0, 10), excl=("the error was identified and corrected before reaching the patient", "Corrected before reaching patient"), pre="IR", notify="the Clinical Governance team"),
    "insurance": dict(term="Large Loss", noun="claim", enum="Peril", enums=["fire", "escape of water", "storm", "theft", "subsidence", "accidental damage"],
                      num="Reserve", money=True, rng=(1000, 250000), excl=("the claim is fully reinsured under a facultative arrangement", "Facultatively reinsured"), pre="CLM", notify="the Chief Underwriter"),
    "finance": dict(term="Reviewable Transaction", noun="transaction", enum="Transaction type", enums=["cash deposit", "international wire", "crypto exchange transfer", "cheque", "card payment", "standing order"],
                    num="Amount", money=True, rng=(500, 60000), excl=("both accounts belong to the same verified client", "Own-account transfer"), pre="TX", notify="the Financial Crime team"),
    "saas_ops": dict(term="Major Incident", noun="incident", enum="Affected component", enums=["authentication service", "API gateway", "billing portal", "reporting module", "webhook delivery", "file storage"],
                     num="Customers affected", money=False, unit="customers", rng=(1, 5000), excl=("the incident was confined to a sandbox environment", "Sandbox only"), pre="INC", notify="the incident commander"),
    "travel": dict(term="Significant Disruption", noun="service", enum="Disruption type", enums=["cancellation", "delay", "diversion", "downgrade", "missed connection", "overbooking"],
                   num="Delay at final destination", money=False, unit="minutes", rng=(0, 600), excl=("the disruption was announced more than 14 days before departure", "Announced over 14 days ahead"), pre="SVC", notify="Customer Relations"),
    "public_sector": dict(term="Major Application", noun="application", enum="Development type", enums=["change of use", "new dwelling", "extension", "demolition", "advertisement", "telecommunications mast"],
                          num="Floor area", money=False, unit="square metres", rng=(10, 3000), excl=("the application is a resubmission within 12 months of a refusal", "Resubmission within 12 months"), pre="PA", notify="the planning committee"),
    "legal": dict(term="High-Risk Matter", noun="matter", enum="Practice area", enums=["litigation", "corporate transactions", "regulatory investigations", "conveyancing", "employment advice", "intellectual property"],
                  num="Estimated fees", money=True, rng=(2000, 400000), excl=("the client has been a client of the firm for more than five years", "Client for over five years"), pre="MAT", notify="the Risk Committee"),
}


def s_in(p, r, ignore=frozenset()):
    """Is record r within the defined term?"""
    am = p["amend"] and r["date"] >= p["amend"]["date"] and "amend" not in ignore
    types = set(p["types"]) | ({p["amend"]["add_type"]} if am and p["amend"].get("add_type") else set())
    T = p["amend"]["T"] if am and p["amend"].get("T") is not None else p["T"]
    strict = p["strict"] != ("boundary" in ignore)
    ok_num = r["num"] > T if strict else r["num"] >= T
    ex = r["excl"] and "excl" not in ignore
    if ex and p["override"] is not None and r["type"] == p["override"] and "override" not in ignore:
        ex = False
    return r["type"] in types and ok_num and not ex


S_TWISTS = ["amend", "boundary", "excl", "override"]


def _s_params(rng, dom_id, difficulty):
    cfg = SCOPE[dom_id]
    lo, hi = cfg["rng"]
    enums = list(cfg["enums"])
    rng.shuffle(enums)
    T = rng.randint(lo + (hi - lo) // 5, lo + (hi - lo) * 3 // 5)
    if cfg["money"]:
        T = int(round(T, -2)) or 100
    y0 = rng.choice([2025, 2026, 2027])
    p = {"dom": dom_id, "types": enums[:3], "others": enums[3:], "T": T, "strict": chance(rng, 0.5), "override": None, "amend": None, "year": y0}
    if difficulty >= 4 or chance(rng, 0.4):
        p["override"] = rng.choice(p["types"])
    if difficulty >= 4 or chance(rng, 0.5):
        if chance(rng, 0.5):
            T2 = T + rng.choice([-1, 1]) * max(1, (hi - lo) // rng.choice([8, 10, 12]))
            if cfg["money"]:
                T2 = int(round(T2, -2)) or 100
            if T2 != T and T2 > lo:
                p["amend"] = {"date": dt.date(y0, rng.randint(3, 10), rng.randint(1, 28)), "T": T2, "add_type": None}
        else:
            p["amend"] = {"date": dt.date(y0, rng.randint(3, 10), rng.randint(1, 28)), "T": None, "add_type": p["others"][0]}
    return p


def _s_records(rng, p, dom_id, n):
    cfg = SCOPE[dom_id]
    lo, hi = cfg["rng"]
    Ts = [p["T"]] + ([p["amend"]["T"]] if p["amend"] and p["amend"].get("T") is not None else [])
    recs = []
    ids = set()
    for _ in range(n):
        while True:
            rid = f"{cfg['pre']}-{rng.randint(1000, 9999)}"
            if rid not in ids:
                ids.add(rid); break
        mode = rng.random()
        if mode < 0.25:
            num = rng.choice(Ts)
        elif mode < 0.6:
            num = rng.choice(Ts) + rng.choice([-1, 1]) * rng.randint(1, max(1, (hi - lo) // 10))
        else:
            num = rng.randint(lo, hi)
        num = max(lo, min(hi, num))
        if cfg["money"] and num > 1000 and chance(rng, 0.5):
            num = int(round(num, -1))
        date = (p["amend"]["date"] + dt.timedelta(days=rng.randint(-40, 40))) if p["amend"] else dt.date(p["year"], rng.randint(1, 12), rng.randint(1, 28))
        typ = rng.choice(p["types"] + p["others"][:2] + p["types"])
        recs.append({"id": rid, "date": date, "type": typ, "num": num, "excl": chance(rng, 0.3)})
    return recs


S_KEY = {"scope_which": None, "scope_one": None, "scope_count": None}


def gen_scope(rng, difficulty, kind, want_unknown):
    dom_id = rng.choice(DOMAIN_IDS)
    cfg = SCOPE[dom_id]
    n = {3: rng.randint(4, 5), 4: rng.randint(5, 6), 5: rng.randint(6, 8)}[difficulty]
    need = {3: 1, 4: 2, 5: 2}[difficulty]
    want = chance(rng, 0.5)
    for _ in range(400):
        p = _s_params(rng, dom_id, difficulty)
        recs = _s_records(rng, p, dom_id, n)
        ins = [s_in(p, r) for r in recs]
        target = None
        if kind == "scope_which":
            if sum(ins) != 1:
                continue
            gold_i = ins.index(True)
            # near misses: other records that a careless reader would include
            nm = [i for i, r in enumerate(recs) if not ins[i] and any(s_in(p, r, frozenset([t])) for t in S_TWISTS)]
            # the gold record itself must depend on a twist, or be surrounded by twist-dependent near misses
            dep = sum(1 for t in S_TWISTS if [s_in(p, r, frozenset([t])) for r in recs] != ins)
            if len(nm) >= min(need, 2) and dep >= need:
                break
        elif kind == "scope_one":
            target = rng.randrange(n)
            r = recs[target]
            if _ < 250 and ins[target] != want:
                continue
            dep = [t for t in S_TWISTS if s_in(p, r, frozenset([t])) != ins[target]]
            if len(dep) >= min(need, 2) or (need == 1 and dep):
                break
        else:
            cnt = sum(ins)
            dep = [t for t in S_TWISTS if sum(s_in(p, r, frozenset([t])) for r in recs) != cnt]
            if len(dep) >= need:
                break
    else:
        raise ValueError("no interesting scope case")
    doc_kind = rng.choice(["policy", "sop", "incident_log", "policy"])
    rs = rng.random()
    text, doc, decisive = _s_render(random.Random(rs), dom_id, cfg, p, recs, difficulty, doc_kind)
    field, gold, hints, ov = _s_question(random.Random(rs + 1), kind, p, recs, ins, target, cfg, doc)
    spec = _s_spec(p, recs) | {"kind": kind, "target": target, "opt_values": ov}
    items = [item(text, field, gold, kind, dom_id, spec, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _s_unknown(rng, p, recs, ins, kind, target, dom_id, cfg, difficulty, doc_kind)
        if u is None:
            raise ValueError("no scope unknown")
        items.append(u)
    return items


def _s_spec(p, recs):
    pp = {k: v for k, v in p.items() if k != "amend"}
    pp["amend"] = ({"date": _iso(p["amend"]["date"]), "T": p["amend"]["T"], "add_type": p["amend"]["add_type"]} if p["amend"] else None)
    return {"schema": "scope", "p": pp, "recs": [dict(r, date=_iso(r["date"])) for r in recs]}


def _s_render(rng, dom_id, cfg, p, recs, difficulty, doc_kind, missing=None, miss_idx=None):
    dom = DOMAINS[dom_id]
    doc = Doc(rng, dom_id, doc_kind)
    term, noun = cfg["term"], cfg["noun"]
    fmtn = (lambda v: doc.m(v)) if cfg["money"] else (lambda v: f"{v} {cfg['unit']}")
    types = p["types"]
    tl = f"{types[0]}, {types[1]} or {types[2]}"
    cmp_ = P(rng, "more than", "greater than", "above") if p["strict"] else P(rng, "at least", "not less than", "equal to or more than")
    defs = [("d_term", P(rng, f"\"{term}\" means a {noun} where (a) the {cfg['enum'].lower()} is {tl}; and (b) the {cfg['num'].lower()} is {cmp_} {fmtn(p['T'])}; but excluding any {noun} covered by {{ref:excl}}.",
                         f"A {noun} is a \"{term}\" if both of the following apply: its {cfg['enum'].lower()} is {tl}, and its {cfg['num'].lower()} is {cmp_} {fmtn(p['T'])}. {{ref:excl}} takes some {noun}s out of this definition.",
                         f"\"{term}\": any {noun} whose {cfg['enum'].lower()} is one of {tl} and whose {cfg['num'].lower()} is {cmp_} {fmtn(p['T'])}, subject to the exclusion in {{ref:excl}}.")),
            P(rng, f"\"Record Date\" means the date shown on the {noun} record in {doc.ctx['system']}.", f"The \"Record Date\" of a {noun} is the date it was first logged."),
            P(rng, f"\"Notification\" means a written report to {cfg['notify']} using the standard form.", f"\"Business day\" has its usual meaning.")]
    rng.shuffle(defs)
    sec_def = doc.add(P(rng, "Definitions", "Interpretation", "Key terms"), defs)
    oblig = [P(rng, f"Every {term} must be notified to {cfg['notify']} within 72 hours of being identified.",
               f"{cfg['notify'][0].upper() + cfg['notify'][1:]} must receive a Notification for each {term} within three business days.",
               f"A {term} is escalated to {cfg['notify']} and tracked to closure."),
             P(rng, f"{noun.capitalize()}s that are not {term}s are handled locally and reviewed in the monthly report.",
               f"Other {noun}s follow the standard handling route and do not require a Notification."),
             P(rng, f"A {noun} whose {cfg['enum'].lower()} is {p['others'][-1]} is always reviewed by the {doc.ctx['role2']}, but this review does not make it a {term}.",
               f"{p['others'][-1].capitalize()} {noun}s are sampled for quality review; the sample is not related to the definition of a {term}.")]
    sec_ob = doc.add(P(rng, "Notification", "Escalation", f"Handling a {term}", "Reporting duties"), oblig)
    ex = [("excl", P(rng, f"A {noun} is not a {term} where {cfg['excl'][0]}.", f"The definition of {term} does not cover any {noun} where {cfg['excl'][0]}.",
                     f"Excluded: any {noun} where {cfg['excl'][0]}; such {noun}s are not {term}s even if they meet {{ref:d_term}}."))]
    if p["override"] is not None:
        ex.append(("override", P(rng, f"{{ref:excl}} does not apply to a {noun} whose {cfg['enum'].lower()} is {p['override']}; such a {noun} can be a {term} even where {cfg['excl'][0]}.",
                                 f"However, where the {cfg['enum'].lower()} is {p['override']}, the exclusion in {{ref:excl}} is disregarded.")))
    sec_ex = doc.add(P(rng, "Exclusions", "Scope limits", "What is not covered"), ex)
    sec_am = None
    if p["amend"]:
        a = p["amend"]
        if a["T"] is not None:
            t = P(rng, f"For {noun}s with a Record Date on or after {doc.fd(a['date'])}, the figure of {fmtn(p['T'])} in {{ref:d_term}} is replaced by {fmtn(a['T'])}.",
                  f"From {doc.fd(a['date'])} (by Record Date), {{ref:d_term}} applies with {fmtn(a['T'])} in place of {fmtn(p['T'])}.")
        else:
            t = P(rng, f"For {noun}s with a Record Date on or after {doc.fd(a['date'])}, {a['add_type']} is added to the list in paragraph (a) of {{ref:d_term}}.",
                  f"From {doc.fd(a['date'])} (by Record Date), a {cfg['enum'].lower()} of {a['add_type']} also satisfies part (a) of {{ref:d_term}}.")
        sec_am = doc.add(P(rng, "Amendments", "Changes to the definition", "Amendment log"), [("amend", t)])
    doc.boiler(rng.randint(4, 7))
    doc.order([sec_def, sec_ob, sec_ex], boiler_front=rng.randint(1, 2))
    if sec_am:
        doc.sections = [s for s in doc.sections if s is not sec_am] + [sec_am]
    doc.front = front_matter(doc, P(rng, f"{term} Procedure", f"{noun.capitalize()} Escalation Policy", *dom["doc_nouns"]),
                             dt.date(p["year"] - 1, rng.randint(1, 12), 1), f"{rng.randint(1, 6)}.{rng.randint(0, 9)}")
    hdr = f"| {noun.capitalize()} | Record Date | {cfg['enum']} | {cfg['num']} | {cfg['excl'][1]} | Handler |"
    rows = [hdr, "|---|---|---|---|---|---|"]
    for j, r in enumerate(recs):
        typ = r["type"] if not (missing == "type" and j == miss_idx) else "not recorded"
        num = fmtn(r["num"]) if not (missing == "num" and j == miss_idx) else "not recorded"
        exc = ("yes" if r["excl"] else "no") if not (missing == "excl" and j == miss_idx) else "not recorded"
        dd = doc.fd(r["date"]) if not (missing == "date" and j == miss_idx) else "not recorded"
        rows.append(f"| {r['id']} | {dd} | {typ} | {num} | {exc} | {doc.people()} |")
    doc.tail = ["", P(rng, f"{noun.capitalize()}s logged this period:", f"Extract from the {noun} log:", f"Open {noun}s under review:")] + rows
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("d_term", "excl", "override", "amend") if doc.rendered(a)]
    return text, doc, decisive


def _s_question(rng, kind, p, recs, ins, target, cfg, doc):
    term, noun = cfg["term"], cfg["noun"]
    if kind == "scope_which":
        q = P(rng, f"Which of the listed {noun}s is a {term}?", f"Exactly one {noun} in the log meets the definition of {term}. Which one?",
              f"Which {noun} must be notified as a {term}?")
        gi = ins.index(True)
        ids = [r["id"] for r in recs]
        field, gold, ov = cf(rng, q, ids[gi], [x for x in ids if x != ids[gi]], lambda x: x)
        return field, gold, {}, ov
    if kind == "scope_one":
        r = recs[target]
        q = P(rng, f"Is {noun} {r['id']} a {term}?", f"Does {r['id']} meet the definition of a {term}?", f"Must {r['id']} be treated as a {term}?")
        neg = P(rng, f"Is {noun} {r['id']} outside the definition of a {term}?", f"Does {r['id']} fall outside the definition of {term}?")
        return noul_field(q), ins[target], {"neg": {"question": neg, "gold": not ins[target]}}, None
    q = P(rng, f"How many of the listed {noun}s are {term}s?", f"Count the {term}s in the log above.", f"How many {noun}s in the extract meet the definition of {term}?")
    return score_field(q, [f"{i} {noun}{'s' if i != 1 else ''}" for i in range(len(recs) + 1)]), sum(ins), {}, None


def _s_unknown(rng, p, recs, ins, kind, target, dom_id, cfg, difficulty, doc_kind):
    lo, hi = cfg["rng"]
    def ans(rs_):
        v = [s_in(p, r) for r in rs_]
        if kind == "scope_which":
            return tuple(v)
        if kind == "scope_one":
            return v[target]
        return sum(v)
    cands = [target] if kind == "scope_one" else ([ins.index(True)] if kind == "scope_which" else list(range(len(recs))))
    rng.shuffle(cands)
    for j in cands:
        for miss in rng.sample(["num", "type", "excl", "date"], 4):
            if miss == "date" and not p["amend"]:
                continue
            r = recs[j]
            if miss == "num":
                vals = [lo, hi] + [x for x in (p["T"], p["T"] - 1, p["T"] + 1) if lo <= x <= hi]
                alts = [dict(r, num=v) for v in vals]
            elif miss == "type":
                alts = [dict(r, type=t) for t in p["types"] + p["others"]]
            elif miss == "excl":
                alts = [dict(r, excl=True), dict(r, excl=False)]
            else:
                alts = [dict(r, date=p["amend"]["date"] + dt.timedelta(days=k)) for k in (-10, 10)]
            outs = {str(ans(recs[:j] + [a] + recs[j + 1:])) for a in alts}
            if len(outs) < 2:
                continue
            rs = rng.random()
            text, doc, _ = _s_render(random.Random(rs), dom_id, cfg, p, recs, difficulty, doc_kind, missing=miss, miss_idx=j)
            f2, _, _, _ov = _s_question(random.Random(rs + 1), kind, p, recs, ins, target, cfg, doc)
            spec = _s_spec(p, recs) | {"kind": kind, "target": target, "missing": miss, "miss_idx": j}
            _trace(spec, [_s_spec(p, recs[:j] + [a] + recs[j + 1:]) | {"kind": kind, "target": target} for a in alts]
                   + [_s_spec(p, recs) | {"kind": kind, "target": target}], _ov)
            return item(text, f2, None, kind, dom_id, spec, {}, child_of=0)
    return None


def recheck_scope(spec):
    if spec.get("missing"):
        return None
    p = spec["p"]
    D = dt.date.fromisoformat
    res = []
    for r in spec["recs"]:
        a = p["amend"]
        after = a is not None and D(r["date"]) >= D(a["date"])
        allowed = list(p["types"]) + ([a["add_type"]] if after and a["add_type"] else [])
        thr = a["T"] if after and a["T"] is not None else p["T"]
        big = (r["num"] > thr) if p["strict"] else (r["num"] >= thr)
        excluded = r["excl"] and r["type"] != p["override"]
        res.append(r["type"] in allowed and big and not excluded)
    if spec["kind"] == "scope_which":
        return spec["recs"][res.index(True)]["id"] if res.count(True) == 1 else "__none__"
    if spec["kind"] == "scope_one":
        return res[spec["target"]]
    return sum(res)


# ================================================================================================ sla
SLA_SERVICES = {"retail": ["checkout", "click-and-collect", "loyalty app", "stock lookup", "gift card service", "store tills", "returns portal", "price feed"],
                "logistics": ["tracking API", "route planner", "depot scanners", "booking portal", "EDI gateway", "driver app", "label printing", "yard management"],
                "hr": ["payroll run", "self-service portal", "time clock", "recruitment site", "benefits portal", "learning platform", "rota planner", "expenses app"],
                "healthcare_admin": ["appointment booking", "results viewer", "e-prescribing", "patient portal", "referral inbox", "bed management", "PACS viewer", "switchboard"],
                "insurance": ["claims intake", "quote engine", "policy documents", "broker portal", "payments", "fraud scoring", "call routing", "renewals batch"],
                "finance": ["card authorisations", "online banking", "faster payments", "statement service", "ATM network", "mobile app", "fraud alerts", "branch teller system"],
                "saas_ops": ["authentication service", "API gateway", "billing portal", "reporting module", "webhook delivery", "file storage", "search index", "admin console"],
                "travel": ["booking engine", "check-in kiosks", "departure boards", "mobile boarding passes", "baggage tracking", "loyalty platform", "crew rostering", "call centre IVR"],
                "public_sector": ["online payments", "planning portal", "housing register", "contact centre telephony", "permit issuing", "waste booking", "council tax portal", "library catalogue"],
                "legal": ["document management", "time recording", "client portal", "e-billing", "conflict search", "e-signature", "matter intake", "court filing gateway"]}
PRIO = ["P1", "P2", "P3", "P4"]


def _matrix(impact, critical):
    # impact 0=High,1=Medium,2=Low
    return {(0, True): 0, (0, False): 1, (1, True): 1, (1, False): 2, (2, True): 2, (2, False): 3}[(impact, critical)]


def sla_eval(p, c, ignore=frozenset()):
    def impact(u):
        if (u >= p["U_high"]) if "boundary" not in ignore else (u > p["U_high"]):
            return 0
        if (u >= p["U_med"]) if "boundary" not in ignore else (u > p["U_med"]):
            return 1
        return 2
    crit = c["service"] in p["critical"]
    pr = _matrix(impact(c["u0"]), crit)
    start = c["t0"]
    if c.get("raise") and "reprio" not in ignore:
        pr2 = _matrix(impact(c["raise"]["u"]), crit)
        if pr2 < pr:
            pr, start = pr2, c["raise"]["t"]
    targets = p["amend"]["targets"] if p["amend"] and c["t0"].date() >= p["amend"]["date"] and "amend" not in ignore else p["targets"]
    remaining = targets[pr]
    t = start
    pause = pr != 0 or "p1" in ignore
    if "pause" in ignore:
        pause = False
    if pause:
        for a, b in sorted(c["pauses"]):
            if b <= t:
                continue
            a = max(a, t)
            gap = int((a - t).total_seconds() // 60)
            if gap >= remaining:
                break
            remaining -= gap
            t = b
    due = t + dt.timedelta(minutes=remaining)
    return {"prio": pr, "due": due, "met": c["tf"] <= due}


SLA_TWISTS = ["boundary", "reprio", "amend", "p1", "pause"]


def _sla_params(rng, dom_id, difficulty):
    svcs = list(SLA_SERVICES[dom_id])
    rng.shuffle(svcs)
    U_med = rng.choice([10, 20, 25, 50])
    U_high = U_med * rng.choice([4, 5, 8, 10])
    tg = [rng.choice([120, 180, 240, 360]), rng.choice([480, 600, 720]), rng.choice([1440, 1800, 2880]), rng.choice([4320, 5760, 7200])]
    y0 = rng.choice([2025, 2026, 2027])
    p = {"U_med": U_med, "U_high": U_high, "critical": svcs[:3], "noncritical": svcs[3:], "targets": tg, "amend": None, "year": y0}
    if difficulty >= 4 or chance(rng, 0.5):
        k = rng.choice([0, 1, 2])
        new = list(tg)
        new[k] = tg[k] + rng.choice([-1, 1]) * rng.choice([60, 120, 240])
        if new[k] > 0 and new == sorted(new) and len(set(new)) == 4:
            p["amend"] = {"date": dt.date(y0, rng.randint(2, 11), rng.randint(1, 28)), "targets": new}
    return p


def _sla_case(rng, p, difficulty):
    svc = rng.choice(p["critical"] + p["noncritical"])
    base_day = (p["amend"]["date"] + dt.timedelta(days=rng.randint(-20, 20))) if p["amend"] else dt.date(p["year"], rng.randint(1, 12), rng.randint(1, 28))
    t0 = dt.datetime.combine(base_day, dt.time(rng.randint(6, 20), rng.choice([0, 5, 12, 20, 33, 41, 48, 55])))
    pick = lambda: rng.choice([p["U_med"], p["U_high"], p["U_med"] - rng.randint(1, 5), p["U_high"] + rng.randint(1, 30), rng.randint(1, p["U_med"] - 1),
                                p["U_med"] + rng.randint(1, 20), p["U_high"] - rng.randint(1, 10)])
    u0 = max(1, pick())
    c = {"service": svc, "t0": t0, "u0": u0, "raise": None, "pauses": []}
    if difficulty >= 4 or chance(rng, 0.5):
        c["raise"] = {"t": t0 + dt.timedelta(minutes=rng.randint(20, 300)), "u": max(u0 + 1, pick())}
    t = t0 + dt.timedelta(minutes=rng.randint(15, 90))
    for _ in range(rng.randint(1, 3) if difficulty >= 4 else rng.randint(0, 2)):
        a = t + dt.timedelta(minutes=rng.randint(10, 240))
        b = a + dt.timedelta(minutes=rng.randint(20, 600))
        c["pauses"].append((a, b))
        t = b
    g = sla_eval(p, dict(c, tf=t0))
    # resolve near the true due time so the pause/raise rules matter
    anchor = rng.choice([g["due"]] + [sla_eval(p, dict(c, tf=t0), frozenset([x]))["due"] for x in SLA_TWISTS])
    tf = anchor + dt.timedelta(minutes=rng.choice([-45, -20, -5, 0, 5, 20, 45, 90]))
    last_pause_end = max([b for a, b in c["pauses"]], default=t0)
    if tf <= last_pause_end or (c["raise"] and tf <= c["raise"]["t"]):
        raise ValueError("resolution before log events")
    c["tf"] = tf
    return c


def gen_sla(rng, difficulty, kind, want_unknown):
    dom_id = rng.choice(DOMAIN_IDS)
    need = {3: 1, 4: 2, 5: 2}[difficulty]
    key = {"sla_met": "met", "sla_due": "due", "sla_priority": "prio"}[kind]
    want = chance(rng, 0.5)
    for _ in range(400):
        p = _sla_params(rng, dom_id, difficulty)
        try:
            c = _sla_case(rng, p, difficulty)
        except ValueError:
            continue
        g = sla_eval(p, c)
        if kind == "sla_met" and _ < 300 and g["met"] != want:
            continue
        wrong = {t: sla_eval(p, c, frozenset([t])) for t in SLA_TWISTS}
        diff = [t for t, w in wrong.items() if w[key] != g[key]]
        if len(diff) >= need:
            break
    else:
        raise ValueError("no interesting sla case")
    doc_kind = rng.choice(["incident_log", "contract", "incident_log"])
    rs = rng.random()
    text, doc, inc, decisive = _sla_render(random.Random(rs), dom_id, p, c, difficulty, doc_kind)
    field, gold, hints, ov = _sla_question(random.Random(rs + 1), kind, p, c, g, wrong, doc, inc)
    items = [item(text, field, gold, kind, dom_id, _sla_spec(p, c) | {"kind": kind, "opt_values": ov}, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _sla_unknown(rng, p, c, kind, dom_id, difficulty, doc_kind)
        if u is None:
            raise ValueError("no sla unknown")
        items.append(u)
    return items


def _sla_spec(p, c):
    iso = lambda t: t.isoformat(timespec="minutes")
    pp = {k: v for k, v in p.items() if k != "amend"}
    pp["amend"] = {"date": _iso(p["amend"]["date"]), "targets": p["amend"]["targets"]} if p["amend"] else None
    return {"schema": "sla", "p": pp, "c": {"service": c["service"], "t0": iso(c["t0"]), "u0": c["u0"], "tf": iso(c["tf"]),
                                           "raise": {"t": iso(c["raise"]["t"]), "u": c["raise"]["u"]} if c["raise"] else None,
                                           "pauses": [[iso(a), iso(b)] for a, b in c["pauses"]]}}


def _hm(m):
    h, mm = divmod(m, 60)
    if mm == 0:
        return f"{h} hours" if h != 1 else "1 hour"
    return f"{h} h {mm} min"


def _sla_render(rng, dom_id, p, c, difficulty, doc_kind, missing=None, miss_idx=None):
    dom = DOMAINS[dom_id]
    doc = Doc(rng, dom_id, doc_kind)
    ts = lambda t: f"{doc.fd(t.date())} {t:%H:%M}"
    crit = ", ".join(p["critical"])
    defs = [("impact", P(rng, f"Impact is High where the confirmed number of affected users is at least {p['U_high']}, Medium where it is at least {p['U_med']} but below {p['U_high']}, and Low otherwise.",
                         f"Impact levels: High — {p['U_high']} or more confirmed affected users; Medium — {p['U_med']} to {p['U_high'] - 1}; Low — fewer than {p['U_med']}.")),
            ("confirmed", P(rng, "Only counts recorded in an \"Impact assessment\" entry of the incident log are confirmed; estimates in customer messages or chat are not.",
                            "The confirmed number of affected users is the figure in the latest \"Impact assessment\" entry; unverified estimates are ignored.")),
            ("critical", P(rng, f"The following are Critical Services: {crit}. All other services are standard services.",
                           f"Critical Services means {crit}; every other service is non-critical for the purposes of this document."))]
    rng.shuffle(defs)
    sec_def = doc.add(P(rng, "Definitions", "Interpretation", "Terms used in this document"), defs)
    mat = ["| Impact | Critical Service | Other service |", "|---|---|---|", "| High | P1 | P2 |", "| Medium | P2 | P3 |", "| Low | P3 | P4 |"]
    tg = p["targets"]
    tgt = ["| Priority | Resolution target |", "|---|---|"] + [f"| {PRIO[i]} | {_hm(tg[i])} |" for i in range(4)]
    sec_p = doc.add(P(rng, "Priority and targets", "Service levels", "Incident priorities", "Resolution targets"),
                    [("matrix", P(rng, "The priority of an incident is set from its impact and the service affected:", "Priority is determined as follows:") + "\n" + "\n".join(mat)),
                     ("targets", P(rng, "Each priority has a resolution target, measured on a 24-hour clock including weekends:", "Resolution targets (elapsed time, 24x7):") + "\n" + "\n".join(tgt))])
    clock = [("start", P(rng, "The clock starts when the incident is opened in the log.", "Measurement begins at the time the incident is first logged.")),
             ("reprio", P(rng, "If a later Impact assessment raises the priority, the incident takes the higher priority from the time of that assessment, and its resolution target is measured afresh from that time.",
                          "When a new Impact assessment moves an incident to a higher priority, the higher priority's target applies and is counted from the time of the assessment, not from when the incident was opened.")),
             ("pause", P(rng, "Time during which the incident status is \"Awaiting Customer\" does not count towards the resolution target.",
                         "The clock is paused while the status is \"Awaiting Customer\" and resumes when the status changes back.")),
             ("p1", P(rng, "The clock is never paused for a P1 incident, whatever its status.", "P1 incidents are measured without any pause: {ref:pause} does not apply to them.")),
             P(rng, "Priorities are never lowered during an incident.", "A priority can be raised but not lowered while the incident is open."),
             P(rng, "An incident is resolved when the log records the status \"Resolved\".", "Resolution time is the time of the \"Resolved\" entry in the log.")]
    rng.shuffle(clock)
    sec_c = doc.add(P(rng, "Measuring the clock", "Clock rules", "How time is counted"), clock)
    sec_am = None
    if p["amend"]:
        k = next(i for i in range(4) if p["amend"]["targets"][i] != tg[i])
        sec_am = doc.add(P(rng, "Amendments", "Change log", "Variations"),
                         [("amend", P(rng, f"For incidents opened on or after {doc.fd(p['amend']['date'])}, the {PRIO[k]} resolution target in {{ref:targets}} is {_hm(p['amend']['targets'][k])} instead of {_hm(tg[k])}.",
                                      f"With effect for incidents opened from {doc.fd(p['amend']['date'])}, {PRIO[k]} incidents have a resolution target of {_hm(p['amend']['targets'][k])}; the other targets are unchanged."))])
    doc.boiler(rng.randint(3, 6))
    doc.order([sec_def, sec_p, sec_c], boiler_front=rng.randint(1, 2))
    if sec_am:
        doc.sections = [s for s in doc.sections if s is not sec_am] + [sec_am]
    doc.front = front_matter(doc, P(rng, "Incident Management Service Standard", "Service Level Agreement — Incident Handling", "Major Incident Procedure"),
                             dt.date(p["year"] - 1, rng.randint(1, 12), 1), f"{rng.randint(1, 4)}.{rng.randint(0, 9)}")
    inc = code(rng, "INC", 6)
    people = [doc.people() for _ in range(4)]
    ev = []
    ev.append((c["t0"], f"Incident opened by {people[0]}: users report problems with the {c['service']}. Status: In Progress."))
    ev.append((c["t0"] + dt.timedelta(minutes=rng.randint(2, 9)),
               f"Impact assessment: {c['u0'] if missing != 'u0' else '[figure pending]'} users confirmed affected."))
    ev.append((c["t0"] + dt.timedelta(minutes=rng.randint(10, 14)), P(rng, f"Customer message: \"this is hitting {c['u0'] * rng.randint(3, 9)} or more of our people\".",
                                                                      f"Chat: {people[1]} thinks the real number could be closer to {c['u0'] * rng.randint(2, 6)}, not yet verified.")))
    if c["raise"]:
        ev.append((c["raise"]["t"], f"Impact assessment: {c['raise']['u'] if missing != 'u1' else '[figure pending]'} users confirmed affected."))
    for j, (a, b) in enumerate(c["pauses"]):
        ev.append((a, f"Status changed to Awaiting Customer ({P(rng, 'logs requested', 'waiting for test account', 'awaiting confirmation of fix', 'need screenshots')})."))
        ev.append((b, "Customer replied. Status changed to In Progress." if not (missing == "pause_end" and j == miss_idx) else "Customer replied (time of reply not captured). Status changed to In Progress."))
    t_last = c["tf"]
    noise = [f"{people[2]} restarted a worker node; no change.", f"Monitoring shows error rate {rng.randint(3, 60)}% on the {rng.choice(p['critical'] + p['noncritical'])}.",
             f"{people[3]} joined the bridge.", "Rollback candidate identified.", f"Vendor ticket raised with reference V-{rng.randint(10000, 99999)}.",
             "Workaround shared with the service desk.", f"Status page updated by {people[1]}.", "Fix deployed to staging.", "Fix deployed to production; monitoring."]
    for _ in range(rng.randint(5, 10)):
        span = max(1, int((t_last - c["t0"]).total_seconds() // 60) - 2)
        ev.append((c["t0"] + dt.timedelta(minutes=rng.randint(1, span)), rng.choice(noise)))
    ev.append((c["tf"], "Status changed to Resolved." if missing != "tf" else "Status changed to Resolved (entry time missing from export)."))
    ev.sort(key=lambda x: x[0])
    lines = ["", P(rng, f"Incident log — {inc}", f"{inc}: timeline export", f"Log extract for {inc}"), f"Service: {c['service']}"]
    for t, e in ev:
        if (missing == "tf" and e.startswith("Status changed to Resolved")) or (missing == "pause_end" and "time of reply not captured" in e):
            lines.append(f"[time not recorded] {e}")
        else:
            lines.append(f"[{ts(t)}] {e}")
    doc.tail = lines
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("impact", "critical", "reprio", "pause", "p1", "amend", "confirmed") if doc.rendered(a)]
    return text, doc, inc, decisive


def _sla_question(rng, kind, p, c, g, wrong, doc, inc):
    ts = lambda t: f"{doc.fd(t.date())} {t:%H:%M}"
    if kind == "sla_met":
        q = P(rng, f"Was {inc} resolved within its resolution target?", f"Did the team meet the resolution target for {inc}?", f"Was the resolution target for {inc} met?")
        neg = P(rng, f"Did {inc} breach its resolution target?", f"Was the resolution target for {inc} missed?")
        return noul_field(q), g["met"], {"neg": {"question": neg, "gold": not g["met"]}}, None
    if kind == "sla_due":
        q = P(rng, f"By what time did {inc} have to be resolved to meet its target?", f"When did the resolution target for {inc} expire?",
              f"What was the latest resolution time that would have met the target for {inc}?")
        cands = [w["due"] for w in wrong.values()] + [g["due"] + dt.timedelta(minutes=k) for k in (-60, 60, 120)]
        field, gold, ov = cf(rng, q, g["due"], cands, ts, k=rng.randint(4, 6))
        ov = {k: v.isoformat(timespec="minutes") if isinstance(v, dt.datetime) else v for k, v in ov.items()}
        shown = {o["text"] for o in field["options"]}
        near = [ts(w["due"]) for w in wrong.values() if w["due"] != g["due"] and ts(w["due"]) not in shown]
        return field, gold, {"near_miss": list(dict.fromkeys(near))[:3]}, ov
    q = P(rng, f"What priority applied to {inc} at the time it was resolved?", f"Which priority should {inc} carry when it was resolved?",
          f"At resolution, what was the correct priority of {inc}?")
    field, gold, ov = cf(rng, q, g["prio"], [0, 1, 2, 3], lambda i: PRIO[i])
    return field, gold, {}, ov


def _sla_unknown(rng, p, c, kind, dom_id, difficulty, doc_kind):
    key = {"sla_met": "met", "sla_due": "due", "sla_priority": "prio"}[kind]
    opts = ["u0", "u1", "tf", "pause_end"]
    rng.shuffle(opts)
    for miss in opts:
        idx, alts = None, []
        if miss == "u0":
            alts = [dict(c, u0=u) for u in (1, p["U_med"], p["U_high"], p["U_high"] * 3)]
        elif miss == "u1":
            if not c["raise"]:
                continue
            alts = [dict(c, raise_=None, **{"raise": dict(c["raise"], u=u)}) for u in (1, p["U_med"], p["U_high"], p["U_high"] * 3)]
        elif miss == "tf":
            if kind != "sla_met":
                continue
            alts = [dict(c, tf=c["tf"] + dt.timedelta(minutes=k)) for k in (-600, 0, 600, 3000)]
            alts = [a for a in alts if a["tf"] > max([b for _, b in c["pauses"]], default=c["t0"])]
        elif miss == "pause_end":
            if not c["pauses"]:
                continue
            idx = rng.randrange(len(c["pauses"]))
            a0, b0 = c["pauses"][idx]
            nxt = c["pauses"][idx + 1][0] if idx + 1 < len(c["pauses"]) else c["tf"]
            alts = [dict(c, pauses=c["pauses"][:idx] + [(a0, b)] + c["pauses"][idx + 1:]) for b in (a0 + dt.timedelta(minutes=1), nxt - dt.timedelta(minutes=1))]
        vals = {str(sla_eval(p, a)[key]) for a in alts}
        if len(vals) < 2:
            continue
        rs = rng.random()
        text, doc, inc, _ = _sla_render(random.Random(rs), dom_id, p, c, difficulty, doc_kind, missing=miss, miss_idx=idx)
        g = sla_eval(p, c)
        f2, _, _, _ov = _sla_question(random.Random(rs + 1), kind, p, c, g, {t: sla_eval(p, c, frozenset([t])) for t in SLA_TWISTS}, doc, inc)
        _trace(_sla_spec(p, c) | {"kind": kind, "missing": miss}, [_sla_spec(p, a) | {"kind": kind} for a in alts] + [_sla_spec(p, c) | {"kind": kind}], _ov)
        return item(text, f2, None, kind, dom_id, _sla_spec(p, c) | {"kind": kind, "missing": miss}, {}, child_of=0)
    return None


def recheck_sla(spec):
    if spec.get("missing"):
        return None
    p, c = spec["p"], spec["c"]
    T = dt.datetime.fromisoformat
    def level(u):
        return 0 if u >= p["U_high"] else (1 if u >= p["U_med"] else 2)
    crit = c["service"] in p["critical"]
    table = {(0, True): 0, (0, False): 1, (1, True): 1, (1, False): 2, (2, True): 2, (2, False): 3}
    prio, start = table[(level(c["u0"]), crit)], T(c["t0"])
    if c["raise"] and table[(level(c["raise"]["u"]), crit)] < prio:
        prio, start = table[(level(c["raise"]["u"]), crit)], T(c["raise"]["t"])
    if spec["kind"] == "sla_priority":
        return prio
    use = p["amend"]["targets"] if p["amend"] and T(c["t0"]).date() >= dt.date.fromisoformat(p["amend"]["date"]) else p["targets"]
    # minute-by-minute walk (independent of the interval arithmetic in sla_eval)
    paused = [(T(a), T(b)) for a, b in c["pauses"]] if prio != 0 else []
    need, t = use[prio], start
    step = dt.timedelta(minutes=1)
    while need > 0:
        if not any(a <= t < b for a, b in paused):
            need -= 1
        t += step
    # the target expires at the end of the last counted minute; skip forward over a pause starting exactly then is not needed
    due = t
    if spec["kind"] == "sla_due":
        return due.isoformat(timespec="minutes")
    return T(c["tf"]) <= due


# ================================================================================================ notice
AGREEMENTS = {"retail": "Supply Agreement", "logistics": "Haulage Services Agreement", "hr": "Benefits Administration Agreement",
              "healthcare_admin": "Clinical Services Agreement", "insurance": "Delegated Claims Handling Agreement", "finance": "Card Processing Agreement",
              "saas_ops": "Subscription Agreement", "travel": "Ground Services Agreement", "public_sector": "Framework Services Contract", "legal": "Outsourced Support Agreement"}
METHODS = ["email", "courier", "post"]


def n_receipt(p, c, ignore=frozenset()):
    hol = {_d(h) for h in p["holidays"]}
    isb = lambda d: d.weekday() < 5 and d not in hol
    if "deemed" in ignore:
        return c["sent"].date()
    m = c["method"]
    if m == "email":
        d = c["sent"].date()
        if isb(d) and c["sent"].time() < dt.time(17, 0):
            return d
        return add_bdays(d, 1, hol)
    if m == "courier":
        return c["delivered"]
    return add_bdays(c["sent"].date(), p["post_days"], hol)


def n_eval(p, c, ignore=frozenset()):
    valid = c["address_ok"] or "address" in ignore
    r = n_receipt(p, c, ignore)
    N = p["amend"]["N"] if p["amend"] and r >= p["amend"]["date"] and "amend" not in ignore else p["N"]
    k = 0
    while True:
        end = add_months(p["start"], p["M"] + k * p["R"]) - dt.timedelta(days=0 if "offbyone" in ignore else 1)
        if end >= r and (end - r).days >= N:
            break
        k += 1
    return {"end": end if valid else None, "valid": valid, "receipt": r}


N_TWISTS = ["address", "deemed", "amend", "offbyone"]


def _n_params(rng, difficulty):
    y0 = rng.choice([2022, 2023, 2024, 2025])
    start = dt.date(y0, rng.randint(1, 12), rng.randint(1, 28))
    p = {"start": start, "M": rng.choice([12, 24, 36]), "R": rng.choice([6, 12, 12, 3]), "N": rng.choice([30, 45, 60, 90]),
         "post_days": rng.choice([2, 3, 5]), "amend": None}
    hol = []
    for y in range(y0, y0 + 6):
        hol += closure_days(rng, y, 5)
    p["holidays"] = [h.isoformat() for h in hol]
    if difficulty >= 4 or chance(rng, 0.5):
        p["amend"] = {"date": None, "N": rng.choice([x for x in (30, 45, 60, 90, 120) if x != p["N"]])}
    return p


def _n_case(rng, p, difficulty):
    # pick a term end within a few years and send notice close to the notice deadline for it
    k = rng.randint(0, 4)
    end = add_months(p["start"], p["M"] + k * p["R"]) - dt.timedelta(days=1)
    N = p["N"]
    if p["amend"]:
        p["amend"]["date"] = end - dt.timedelta(days=rng.randint(max(N, p["amend"]["N"]) - 10, max(N, p["amend"]["N"]) + 40))
        N = rng.choice([p["N"], p["amend"]["N"]])
    d = end - dt.timedelta(days=N + rng.randint(-4, 4))
    method = rng.choice(METHODS)
    sent = dt.datetime.combine(d, dt.time(rng.choice([9, 11, 14, 16, 17, 18, 21]), rng.choice([0, 15, 30, 45, 55])))
    c = {"method": method, "sent": sent, "delivered": d + dt.timedelta(days=rng.randint(1, 4)) if method == "courier" else None,
         "address_ok": not chance(rng, 0.15 if difficulty < 5 else 0.25)}
    return c


def gen_notice(rng, difficulty, kind, want_unknown):
    dom_id = rng.choice(DOMAIN_IDS)
    need = {3: 1, 4: 2, 5: 2}[difficulty]
    want = chance(rng, 0.5)
    for _ in range(400):
        p = _n_params(rng, difficulty)
        c = _n_case(rng, p, difficulty)
        g = n_eval(p, c)
        wrong = {t: n_eval(p, c, frozenset([t])) for t in N_TWISTS}
        if kind == "notice_end":
            diff = [t for t, w in wrong.items() if w["end"] != g["end"]]
        else:
            # "will the agreement end on X": X is either the true end or a wrong-path end
            xs = [w["end"] for w in wrong.values() if w["end"]] + ([g["end"]] if g["end"] else [])
            if not xs:
                continue
            x = rng.choice(xs)
            if _ < 300 and (g["end"] == x) != want:
                continue
            c["x"] = x
            diff = [t for t, w in wrong.items() if (w["end"] == x) != (g["end"] == x)]
        if len(diff) >= need:
            break
    else:
        raise ValueError("no interesting notice case")
    rs = rng.random()
    text, doc, decisive, names = _n_render(random.Random(rs), dom_id, p, c, difficulty)
    field, gold, hints, ov = _n_question(random.Random(rs + 1), kind, p, c, g, wrong, doc, names)
    items = [item(text, field, gold, kind, dom_id, _n_spec(p, c) | {"kind": kind, "opt_values": ov}, {**hints, "decisive": decisive})]
    if want_unknown:
        u = _n_unknown(rng, p, c, kind, dom_id, difficulty)
        if u is None:
            raise ValueError("no notice unknown")
        items.append(u)
    return items


def _n_spec(p, c):
    pp = dict(p, start=_iso(p["start"]), amend=({"date": _iso(p["amend"]["date"]), "N": p["amend"]["N"]} if p["amend"] else None))
    cc = {"method": c["method"], "sent": c["sent"].isoformat(timespec="minutes"), "delivered": _iso(c["delivered"]), "address_ok": c["address_ok"],
          "x": _iso(c.get("x"))}
    return {"schema": "notice", "p": pp, "c": cc}


def _n_render(rng, dom_id, p, c, difficulty, missing=None):
    dom = DOMAINS[dom_id]
    doc = Doc(rng, dom_id, "contract")
    title = AGREEMENTS[dom_id]
    supplier = doc.org
    customer = f"{rng.choice(['Aldergate', 'Brookvale', 'Castleford', 'Dunmere', 'Elmstead', 'Farrowdale', 'Greystone', 'Hartwell'])} {rng.choice(['Holdings', 'Group', 'Trust', 'Services', 'plc', 'Ltd'])}"
    notices_email = f"notices@{doc.org.split()[0].lower()}.example"
    wrong_email = f"{doc.people().split()[0].lower()}@{doc.org.split()[0].lower()}.example"
    addr = f"{rng.randint(2, 180)} {rng.choice(['Harbour Road', 'Mill Lane', 'Station Street', 'Kings Parade', 'Quarry Way'])}, {rng.choice(['Leeds', 'Cork', 'Adelaide', 'Halifax', 'Dunedin', 'Leicester'])}"
    other_addr = f"{rng.randint(2, 180)} {rng.choice(['Canal Walk', 'Bridge Street', 'Market Square'])}, {rng.choice(['Bristol', 'Galway', 'Hobart'])}"
    defs = [("d_bday", P(rng, "\"Business Day\" means a day other than a Saturday, a Sunday or a public holiday listed in {ref:sched_hol}.",
                         "\"Business Day\": any weekday that is not a public holiday listed in {ref:sched_hol}.")),
            ("d_start", f"\"Commencement Date\" means {doc.fd(p['start'])}."),
            P(rng, f"\"Services\" means the services described in the Order Form.", f"\"Charges\" means the charges set out in the Order Form."),
            P(rng, f"\"Supplier\" means {supplier}; \"Customer\" means {customer}.", f"The parties are {supplier} (the \"Supplier\") and {customer} (the \"Customer\").")]
    rng.shuffle(defs)
    sec_def = doc.add(P(rng, "Definitions", "Definitions and interpretation"), defs)
    renew_txt = f"{p['R']} months" if missing != "R" else "the renewal period stated in the Order Form"
    term = [("term", P(rng, f"The Initial Term starts on the Commencement Date and lasts {p['M']} months, ending on the day before the date {p['M']} months after the Commencement Date.",
                       f"This Agreement runs for an Initial Term of {p['M']} months from the Commencement Date; the Initial Term ends on the day immediately before the {p['M']}-month anniversary of the Commencement Date.")),
            ("renew", P(rng, f"At the end of the Initial Term, and of each Renewal Term, this Agreement renews automatically for a further Renewal Term of {renew_txt}, starting on the day after the previous term ends.",
                        f"Unless terminated under {{ref:notice}}, this Agreement continues after the Initial Term for successive Renewal Terms of {renew_txt} each.")),
            ("notice", P(rng, f"Either party may terminate this Agreement with effect from the last day of the Initial Term or of any Renewal Term by giving the other at least {p['N']} days' written notice before that day.",
                         f"A party may end this Agreement at the end of the current term only by written notice received by the other party at least {p['N']} days before the term ends; otherwise the notice takes effect at the end of the next term in which it is timely.")),
            ("count", P(rng, "When counting a notice period in days, the day on which notice is received is not counted and the last day of the term is counted.",
                        "Notice periods are counted in calendar days from the day after receipt up to and including the last day of the term."))]
    sec_term = doc.add(P(rng, "Term and termination", "Duration", "Term, renewal and termination"), term)
    addr_txt = (f"by email to {notices_email} or by courier or post to {addr}" if missing != "address" else "to the notices address set out in the Order Form")
    notices = [("service", P(rng, f"Notices under this Agreement must be in writing and sent to the Supplier {addr_txt}. Notices sent to any other address or person are not validly given.",
                             f"Any notice to the Supplier is valid only if sent {addr_txt}; a notice delivered anywhere else has no effect.")),
               ("deemed", P(rng, f"A notice is deemed received: (a) if sent by email, on the day it is sent if that is a Business Day and it is sent before 17:00, and otherwise on the next Business Day; (b) if sent by courier, on the date of delivery; (c) if sent by post, on the {num_word(p['post_days'])} Business Day after posting.",
                            f"Deemed receipt: email — the day of sending when sent before 17:00 on a Business Day, otherwise the next Business Day; courier — the day it is delivered; post — {p['post_days']} Business Days after the day of posting.")),
               P(rng, "Notices given by the Supplier to the Customer are sent to the address the Customer last notified in writing.",
                 "A party may change its notices address by giving notice under this clause.")]
    sec_n = doc.add(P(rng, "Notices", "Giving notice", "Communications"), notices)
    other = [P(rng, "Either party may terminate immediately by notice if the other commits a material breach that is not remedied within 30 days of a request to remedy it.",
               "A party may terminate immediately if the other becomes insolvent."),
             P(rng, f"The Charges are reviewed once a year and may rise by no more than the published inflation index.",
               f"Invoices are payable within 30 days of the invoice date."),
             P(rng, "Termination does not affect rights that have already accrued.", "Clauses on confidentiality and liability survive termination.")]
    sec_o = doc.add(P(rng, "Other rights to terminate", "General", "Charges and other terms"), other)
    sec_am = None
    if p["amend"]:
        sec_am = doc.add(P(rng, "Variation", "Amendment No. 1", "Deed of variation"),
                         [("amend", P(rng, f"By a variation agreed in writing, for any notice received on or after {doc.fd(p['amend']['date'])} the notice period in {{ref:notice}} is {p['amend']['N']} days instead of {p['N']} days.",
                                      f"With effect from {doc.fd(p['amend']['date'])}, {{ref:notice}} is amended so that {p['amend']['N']} days' notice is required; this applies to notices received on or after that date."))])
    hol = [_d(h) for h in p["holidays"]]
    sec_s = doc.add("Schedule", [("sched_hol", "Public holidays: " + "; ".join(doc.fd(h) for h in hol) + ".")])
    doc.boiler(rng.randint(3, 6), exclude=("complaints",))
    doc.order([sec_def, sec_term, sec_n, sec_o], boiler_front=rng.randint(0, 2))
    tail = [x for x in (sec_am, sec_s) if x]
    doc.sections = [x for x in doc.sections if x not in tail] + tail
    doc.front = front_matter(doc, title, p["start"], f"{rng.randint(1, 3)}.{rng.randint(0, 3)}")
    signer = doc.people()
    m = c["method"]
    to = (notices_email if m == "email" else addr) if c["address_ok"] else (wrong_email if m == "email" else other_addr)
    via = {"email": "by email", "courier": "by courier", "post": "by post"}[m]
    sent_txt = f"{doc.fd(c['sent'].date())} at {c['sent']:%H:%M}" if not (missing == "time") else f"{doc.fd(c['sent'].date())} (time not recorded)"
    if m != "email":
        sent_txt = doc.fd(c["sent"].date()) if missing != "sentdate" else "date not recorded"
    lines = ["", P(rng, "Correspondence file", "Notice record", "Termination notice — file note"),
             f"On behalf of {customer}, {signer} sent a written notice of termination {via} to {to}.",
             f"{'Posted' if m == 'post' else 'Sent'}: {sent_txt}."]
    if m == "courier":
        lines.append(f"Courier tracking shows delivery on {doc.fd(c['delivered']) if missing != 'delivered' else '(delivery scan missing)'}.")
    lines.append(P(rng, f"The notice asks for the Agreement to end \"as soon as the contract allows\".", "The notice states that the Customer wishes to terminate at the earliest date permitted."))
    lines.append(P(rng, f"{doc.people()} (account manager) acknowledged the notice by phone the following week.", "No reply has yet been sent.",
                   f"An internal ticket was opened for the account team."))
    doc.tail = lines
    text = doc.fit()
    decisive = [doc.rendered(a) for a in ("term", "renew", "notice", "service", "deemed", "amend", "count") if doc.rendered(a)]
    return text, doc, decisive, {"customer": customer, "title": title}


def num_word(n):
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[n]


NOT_EFFECTIVE = "The notice is not validly given, so the Agreement does not end as a result of it"


def _n_question(rng, kind, p, c, g, wrong, doc, names):
    if kind == "notice_end":
        q = P(rng, f"As a result of this notice, on what date does the {names['title']} end?", f"What is the earliest date on which the Agreement ends because of {names['customer']}'s notice?",
              "On which date does the termination take effect?")
        cands = [w["end"] for w in wrong.values()] + ([g["end"] + dt.timedelta(days=1), g["end"] - dt.timedelta(days=p["R"] * 30)] if g["end"] else [])
        cands += [add_months(p["start"], p["M"] + k * p["R"]) - dt.timedelta(days=1) for k in range(0, 4)]
        fmt = lambda v: NOT_EFFECTIVE if v is None else doc.fd(v)
        right = g["end"]
        cands = [None] + cands if right is not None else cands
        field, gold, ov = cf(rng, q, right, cands, fmt, k=rng.randint(4, 6))
        if right is not None and not any(o["text"] == NOT_EFFECTIVE for o in field["options"]):
            pass
        shown = {o["text"] for o in field["options"]}
        near = [fmt(w["end"]) for w in wrong.values() if w["end"] != g["end"] and fmt(w["end"]) not in shown]
        return field, gold, {"near_miss": list(dict.fromkeys(near))[:3]}, ov
    x = c["x"]
    q = P(rng, f"Does the Agreement end on {doc.fd(x)} as a result of this notice?", f"Will the {names['title']} terminate on {doc.fd(x)} because of the notice described?")
    neg = P(rng, f"Does the Agreement continue beyond {doc.fd(x)} despite this notice?", f"Will the {names['title']} still be in force after {doc.fd(x)}?")
    gold = g["end"] == x
    return noul_field(q), gold, {"neg": {"question": neg, "gold": not gold}}, None


def _n_unknown(rng, p, c, kind, dom_id, difficulty):
    key = (lambda r: str(r["end"])) if kind == "notice_end" else (lambda r: r["end"] == c["x"])
    opts = ["time", "sentdate", "delivered", "address", "R"]
    rng.shuffle(opts)
    for miss in opts:
        alts = []
        if miss == "time":
            if c["method"] != "email":
                continue
            alts = [dict(c, sent=dt.datetime.combine(c["sent"].date(), dt.time(h, 0))) for h in (9, 19)]
        elif miss == "sentdate":
            if c["method"] != "post":
                continue
            alts = [dict(c, sent=c["sent"] + dt.timedelta(days=k)) for k in (-20, 0, 20)]
        elif miss == "delivered":
            if c["method"] != "courier":
                continue
            alts = [dict(c, delivered=c["sent"].date() + dt.timedelta(days=k)) for k in (1, 25)]
        elif miss == "address":
            alts = [dict(c, address_ok=True), dict(c, address_ok=False)]
        elif miss == "R":
            alts = [("p", dict(p, R=r)) for r in (1, 3, 6, 12, 24)]
        vals = set()
        for a in alts:
            pp, cc = (a[1], c) if isinstance(a, tuple) else (p, a)
            vals.add(str(key(n_eval(pp, cc))))
        if len(vals) < 2:
            continue
        rs = rng.random()
        text, doc, _, names = _n_render(random.Random(rs), dom_id, p, c, difficulty, missing=miss)
        g = n_eval(p, c)
        f2, _, _, _ov = _n_question(random.Random(rs + 1), kind, p, c, g, {t: n_eval(p, c, frozenset([t])) for t in N_TWISTS}, doc, names)
        _trace(_n_spec(p, c) | {"kind": kind, "missing": miss}, [_n_spec(*_pc(a, p, c)) | {"kind": kind} for a in alts] + [_n_spec(p, c) | {"kind": kind}], _ov)
        return item(text, f2, None, kind, dom_id, _n_spec(p, c) | {"kind": kind, "missing": miss}, {}, child_of=0)
    return None


def recheck_notice(spec):
    if spec.get("missing"):
        return None
    p, c = spec["p"], spec["c"]
    D = dt.date.fromisoformat
    hol = {D(h) for h in p["holidays"]}
    def next_bd(d, n):
        while n:
            d += dt.timedelta(days=1)
            if d.weekday() < 5 and d not in hol:
                n -= 1
        return d
    sent = dt.datetime.fromisoformat(c["sent"])
    if c["method"] == "email":
        sd = sent.date()
        rec = sd if (sd.weekday() < 5 and sd not in hol and sent.hour < 17) else next_bd(sd, 1)
    elif c["method"] == "courier":
        rec = D(c["delivered"])
    else:
        rec = next_bd(sent.date(), p["post_days"])
    period = p["amend"]["N"] if p["amend"] and rec >= D(p["amend"]["date"]) else p["N"]
    import calendar
    st = D(p["start"])
    def anniversary(months):
        y, m = st.year + (st.month - 1 + months) // 12, (st.month - 1 + months) % 12 + 1
        return dt.date(y, m, min(st.day, calendar.monthrange(y, m)[1]))
    months = p["M"]
    while True:
        end = anniversary(months) - dt.timedelta(days=1)
        if (end - rec).days >= period:
            break
        months += p["R"]
    result = end if c["address_ok"] else None
    if spec["kind"] == "notice_end":
        return _iso(result)
    return result is not None and result.isoformat() == c["x"]


# ================================================================================================ dispatcher
def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    schema = kind.split("_")[0]
    fn = {"window": gen_window, "banded": gen_banded, "authority": gen_authority, "scope": gen_scope, "sla": gen_sla, "notice": gen_notice}.get(schema)
    if fn is None:
        raise ValueError(f"schema {schema} not implemented")
    return fn(rng, difficulty, kind, want_unknown)


def recheck(spec):
    return {"window": recheck_window, "banded": recheck_banded, "authority": recheck_authority, "scope": recheck_scope, "sla": recheck_sla, "notice": recheck_notice}[spec["schema"]](spec)


# ================================================================================================ enumerated completions
def worlds(spec, traced=None) -> list:
    """recheck() of every completion of the removed fact. One world for an answerable spec. An unknown spec ("missing") does not
    store its proof's alternatives: pass `traced`, the world specs WORLD_TRACE recorded when the item was generated
    (gen_traps.regen_trace regenerates the item to get them)."""
    if not spec.get("missing"):
        return [recheck(spec)]
    if traced is None:
        raise LookupError("long_policy unknown: its completions come from the generator trace (gen_traps.regen_trace)")
    return [recheck(w) for w in traced]
