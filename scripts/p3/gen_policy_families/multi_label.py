"""multi_label: several labels apply; ONE noul item per label ("Does label X apply?"), all label items of a state share a group.

Kinds
    ticket_tags      support ticket text + 4-10 tag definitions; built from single-topic sentences (positive, near-miss, filler)
    review_aspects   customer review + aspect definitions (late delivery, damage, wrong item, assembly, returns, ...)
    bug_areas        bug report + product-area definitions ("applies when X is described as not working")
    risk_flags       structured transaction/application record + flag definitions with thresholds, exceptions and two-step rules
    contract_issues  contract extract generated from parameters + a compliance checklist (notice, cap, renewal, law, data ...)

Gold is constructed: text kinds know which sentence meets which definition (near-miss sentences mention the topic and explicitly
fail it); structured kinds compute each flag from the record. Unknown children (child_of = the label's parent item) come from a copy
of the state where one sentence is redacted (text kinds) or one field / clause is withheld (structured kinds); a child is emitted only
for labels whose value is not determined by what remains (checked by enumeration in recheck()).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from .common import DOMAINS, P, People, chance, item, noul_field

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from vision_decision.jev_api import multi_label_question  # noqa: E402  (serving wording: training must match it exactly)

NEG_TAIL = ["Is it correct that this label does NOT apply?", "Should this label be left OFF?"]


def serving_q(instructions: str, label: str, description: str) -> str:
    """The per-label noul question exactly as the server's `multi` type sends it."""
    return multi_label_question(instructions, label, description)


def negated_q(rng, instructions: str, label: str, description: str) -> str:
    q = serving_q(instructions, label, description)
    tail = "Does this label apply?"
    assert q.endswith(tail)
    return q[: -len(tail)] + rng.choice(NEG_TAIL)

FAMILY = "multi_label"
KINDS = ["ticket_tags", "review_aspects", "bug_areas", "risk_flags", "contract_issues"]
UNKNOWN_P = 0.62


class Retry(ValueError):
    pass


def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    for _ in range(30):
        try:
            return GEN[kind](rng, difficulty, want_unknown)
        except Retry:
            continue
    raise ValueError(f"{kind}: failed")


def _n_labels(rng, diff, avail):
    lo, hi = {3: (4, 6), 4: (6, 8), 5: (8, 10)}[diff]
    return min(avail, rng.randint(lo, hi))


def _label_name(rng, lid, style):
    if style == 0:
        return lid
    if style == 1:
        return lid.upper().replace("_", "-")
    return lid.replace("_", " ").capitalize()


Q_POOL = [
    "Does the label {L} apply to this {noun}?", "Should this {noun} be tagged {L}?", "Is {L} one of the labels that applies here?",
    "Under the definitions above, does {L} apply?", "Would a correct labeller attach {L} to this {noun}?",
]
NEG_POOL = ["Is it correct that the label {L} does NOT apply to this {noun}?", "Should {L} be left OFF this {noun}?"]


# ================================================================================================ text kinds
# label -> (definition variants, positive sentence templates, near-miss templates)
TICKET = {
    "billing_error": (["The customer says they were charged a different amount from what they agreed to, or charged twice for the same thing.",
                       "Applies when the customer reports an incorrect or duplicate charge."],
                      ["I was charged {m1} but my plan is {m2} a month.", "There are two identical charges of {m1} on my statement for the same order.",
                       "The invoice says {m1}, yet the quote I accepted was {m2}."],
                      ["My invoice of {m1} matches the plan price; I only need a copy for my records.", "The {m1} charge is correct, that's what I signed up for."]),
    "refund_request": (["The customer explicitly asks for money to be paid back.", "Applies only when the customer asks for a refund or reimbursement."],
                       ["Please refund the {m1}.", "I would like my money back for the {thing}.", "Can you reimburse the {m1} I paid for the {thing}?"],
                       ["I'm not after a refund, I just want the {thing} fixed.", "No need to refund anything, a replacement is fine."]),
    "cancellation_intent": (["The customer states that they want to cancel, close the account or not renew.",
                             "Applies when the customer says they intend to cancel or end the service."],
                            ["Please cancel my subscription at the end of this billing period.", "I want to close my account.", "We will not be renewing next month."],
                            ["I'm not planning to cancel, I just need this sorted.", "I've no intention of leaving; I've been a customer for years."]),
    "security_concern": (["The message mentions unauthorised access, a sign-in the customer did not make, or account details changed without their knowledge.",
                          "Applies to reports of account activity the customer did not perform."],
                         ["I got a sign-in alert from {city} and it wasn't me.", "Someone changed the payout bank details without my knowledge.",
                          "There's a login from a device I don't own in my history."],
                         ["I changed my password myself yesterday, that part is fine.", "The new-device alert was me logging in from my work laptop."]),
    "data_loss": (["The customer reports that data they previously saved is now missing.", "Applies when previously stored content has disappeared."],
                  ["All the {thing}s I saved last week have disappeared.", "Half of my project history is gone since the update."],
                  ["Nothing is missing, everything is still there, it's just slow.", "I checked and all my saved {thing}s are intact."]),
    "outage": (["The customer reports the service being completely unavailable to them.", "Applies when the service does not load or respond at all."],
               ["The site won't load at all since {time}.", "The app shows a blank screen for everyone in our office."],
               ["It loads fine, only one button is greyed out.", "The service is up; my question is about something else."]),
    "feature_request": (["The customer asks for a capability the product does not currently have.", "Applies to requests for new functionality."],
                        ["It would be great if you could add export to {fmt}.", "Could you build a way to schedule reports weekly?"],
                        ["I found the export to {fmt} option, thanks.", "The scheduling feature you already have works for me."]),
    "accessibility": (["The message reports a barrier for someone using assistive technology or with a disability.",
                       "Applies to problems using the product with a screen reader, keyboard-only navigation or similar."],
                      ["My screen reader skips the {elem} on the payment page.", "I navigate by keyboard and can't reach the {elem}."],
                      ["I don't use any assistive technology; I just prefer the old colour scheme.", "No accessibility issues for me, the {elem} is fine."]),
    "account_access": (["The customer cannot sign in to their own account.", "Applies when the customer is locked out or unable to log in."],
                       ["I can't log in, the reset email never arrives.", "My account is locked after the password change and I can't get back in."],
                       ["I can log in fine on my phone.", "Signing in works; the problem starts after that."]),
    "staff_complaint": (["The customer complains about how an employee treated them.", "Applies to complaints about the behaviour of a staff member."],
                        ["The agent I spoke to on {day} hung up on me.", "Your colleague in the {city} branch was rude to me."],
                        ["The agent on the phone was really helpful.", "Your staff have always been polite to me."]),
    "legal_threat": (["The customer says they will take legal action or complain to a regulator or ombudsman.",
                      "Applies when the message threatens a lawsuit or a regulator complaint."],
                     ["If this isn't fixed I'll be contacting the ombudsman.", "My solicitor will be in touch if I don't hear back."],
                     ["I don't want to make a formal complaint, I just want this sorted.", "I'm not going to escalate this, don't worry."]),
}
TICKET_FILL = ["Hi, I hope you can help.", "Thanks in advance.", "My account number is {acct}.", "I've been using {org} for about {n} years.",
               "I tried the help centre first but couldn't find an answer.", "Best regards,", "Sorry for the long message.",
               "I'm writing from my phone so excuse any typos.", "This is my {n2} time writing about this.", "My order reference is {ref}.",
               "The weather has been awful here, so I've been working from home.", "I use the {thing} feature every day for work."]

REVIEW = {
    "late_delivery": (["The reviewer states the order arrived after the promised date.", "Applies when delivery came later than promised."],
                      ["It turned up {n} days after the date on the confirmation.", "Promised for {day}, arrived the following week."],
                      ["It came the day it was promised.", "Delivery was bang on the date they gave."]),
    "damaged_item": (["The reviewer says the product arrived broken or damaged.", "Applies when the item itself was damaged on arrival."],
                     ["One of the {part}s was cracked when I opened the box.", "The {product} had a deep scratch across the top out of the box."],
                     ["The box was a bit scuffed but the {product} inside was perfect.", "Packaging was dented; the item itself was undamaged."]),
    "wrong_item": (["The reviewer received a different product, size or colour from the one ordered.", "Applies when the wrong item or variant was sent."],
                   ["I ordered the {c1} one and got {c2}.", "They sent a size {s2} instead of the {s1} I ordered."],
                   ["It's exactly the colour I ordered.", "Right size, right model, no complaints there."]),
    "price_complaint": (["The reviewer says they paid more than the advertised price.", "Applies when the reviewer was charged above the listed price."],
                        ["The checkout added {m1} that wasn't on the product page.", "I was charged {m1} although the listing said {m2}."],
                        ["The price was what the page said.", "Paid exactly the listed {m1}, which I thought was fair."]),
    "praise_staff": (["The reviewer explicitly praises a named employee or team member.", "Applies when a staff member is singled out for praise."],
                     ["{name} at the {place} went out of their way to help.", "Big thanks to {name}, who sorted everything in one call."],
                     ["I never needed to speak to anyone.", "Didn't deal with any staff, it all happened online."]),
    "assembly_difficulty": (["The reviewer reports the product was hard to assemble or the instructions were missing or wrong.",
                             "Applies to problems putting the product together."],
                            ["Step {n} of the instructions shows a part that isn't in the box.", "It took two of us three hours and the diagrams made no sense."],
                            ["Assembly took ten minutes and the instructions were clear.", "It came fully assembled, which was a relief."]),
    "would_not_recommend": (["The reviewer explicitly says they would not recommend the product or would not buy again.",
                             "Applies to an explicit statement of not recommending or not buying again."],
                            ["I wouldn't buy from them again.", "Can't recommend it, sorry."],
                            ["I'd happily buy from them again.", "Would recommend to friends."]),
    "returns_experience": (["The reviewer describes returning, or trying to return, the item.", "Applies when the review describes a return."],
                           ["I sent it back and the refund took {n} weeks.", "Returning it meant printing a label and queueing at the depot."],
                           ["I didn't need to return anything.", "Kept it, so no idea what returns are like."]),
}
REVIEW_FILL = ["Bought this for the {room}.", "Overall mixed feelings.", "My {rel} recommended the brand.", "Ordered on a {day}.",
               "It looks nice in the {room}.", "Four stars would have been possible.", "The colour matches the photos.",
               "We've had it for {n} weeks now.", "It replaced an old one that finally gave up."]

BUG = {
    "checkout": (["The report describes a checkout or payment step not working as expected.", "Applies to malfunctions in checkout or payment."],
                 ["Clicking Pay does nothing and the spinner never stops.", "The card form rejects valid expiry dates."],
                 ["Checkout itself worked fine once I got there.", "Payment went through normally."]),
    "search": (["The report describes search returning wrong, missing or no results.", "Applies to search not working as expected."],
               ["Searching for exact product codes returns nothing.", "Search results show items from the wrong category."],
               ["Search found the product straight away.", "The search bar works as expected."]),
    "notifications": (["The report describes emails, texts or push notifications not arriving or arriving wrong.", "Applies to notification failures."],
                      ["I never receive the order confirmation email.", "Push alerts arrive hours late."],
                      ["Notifications arrive promptly.", "I got the confirmation email within seconds."]),
    "mobile_app": (["The report describes a malfunction in the iOS or Android app.", "Applies to faults specific to the mobile app."],
                   ["The Android app crashes when I open my basket.", "On iOS the app freezes on the splash screen."],
                   ["The mobile app works; this happens on desktop.", "No problems on the phone app."]),
    "admin_console": (["The report describes a problem in the admin or back-office console.", "Applies to admin console faults."],
                      ["The admin console won't save user role changes.", "Exporting users from the console times out."],
                      ["Admin console is fine, it's the storefront.", "I could change the setting in the console without trouble."]),
    "api": (["The report describes the public API returning errors or wrong data.", "Applies to API faults."],
            ["GET /orders returns 500 for any date filter.", "The API sends amounts in cents while the docs say units."],
            ["The API responses look correct.", "Our API integration is unaffected."]),
    "reporting": (["The report describes dashboards or reports showing wrong or missing figures.", "Applies to reporting and analytics faults."],
                  ["The monthly revenue report double-counts refunds.", "Dashboards stop at yesterday's date."],
                  ["Reports match our own figures.", "The dashboards are up to date."]),
    "login": (["The report describes being unable to sign in or being signed out unexpectedly.", "Applies to sign-in and session faults."],
              ["Users get logged out every few minutes.", "Single sign-on loops back to the login page."],
              ["Logging in works fine.", "Sessions stay open as expected."]),
}
BUG_FILL = ["Steps to reproduce are below.", "Browser: {browser}.", "Started after the release on {day}.", "Affects {n} of our users.",
            "Priority: medium from our side.", "Attached a HAR file.", "We are on the {tier} plan.", "Happy to jump on a call.",
            "Environment: production.", "Reported by our {team} team."]

TEXT_KINDS = {"ticket_tags": (TICKET, TICKET_FILL, "ticket"), "review_aspects": (REVIEW, REVIEW_FILL, "review"), "bug_areas": (BUG, BUG_FILL, "bug report")}


def _slots(rng, dom):
    sym = rng.choice(["€", "$", "£"])
    people = People(rng)
    a = rng.randint(12, 400)
    # m2 is always LOWER than m1 (the listed / agreed price when m1 is what was charged)
    return {"m1": f"{sym}{a}", "m2": f"{sym}{a - rng.choice([5, 10, 15, 30])}", "thing": rng.choice(["invoice", "report", "booking", "playlist", "template", "photo"]),
            "city": rng.choice(["Lagos", "Minsk", "Lima", "Hanoi", "Porto", "Tunis"]), "time": rng.choice(["9am", "last night", "Monday morning"]),
            "fmt": rng.choice(["CSV", "PDF", "Excel"]), "elem": rng.choice(["card number field", "submit button", "date picker"]),
            "day": rng.choice(["Monday", "Tuesday", "Friday"]), "n": rng.randint(2, 9), "n2": rng.choice(["second", "third", "fourth"]),
            "acct": f"AC-{rng.randint(10000, 99999)}", "org": f"{rng.choice(['Northwind', 'Kestrel', 'Halcyon', 'Juniper'])} {rng.choice(DOMAINS[dom]['org_suffix'])}",
            "ref": f"R{rng.randint(100000, 999999)}", "part": rng.choice(["shelf", "leg", "panel", "hinge"]), "product": rng.choice(["table", "lamp", "cabinet", "desk"]),
            "c1": "green", "c2": "grey", "s1": "M", "s2": "L", "name": people().split()[0], "place": rng.choice(["Leeds store", "help desk", "warehouse"]),
            "room": rng.choice(["kitchen", "study", "hallway"]), "rel": rng.choice(["sister", "neighbour", "colleague"]),
            "browser": rng.choice(["Firefox 131", "Chrome 129", "Safari 18"]), "tier": rng.choice(["Growth", "Business", "Enterprise"]),
            "team": rng.choice(["support", "finance", "ops"])}


def _gen_text(rng: random.Random, diff: int, want_unknown: bool, kind: str) -> list[dict]:
    table, fill, noun = TEXT_KINDS[kind]
    dom = {"ticket_tags": rng.choice(["saas_ops", "retail", "finance", "travel"]), "review_aspects": "retail", "bug_areas": "saas_ops"}[kind]
    ids = list(table)
    n = _n_labels(rng, diff, len(ids))
    labels = rng.sample(ids, n)
    n_pos = rng.randint(1, max(1, min(n - 1, 4)))
    pos = set(rng.sample(labels, n_pos))
    neg = [l for l in labels if l not in pos]
    near = set(rng.sample(neg, min(len(neg), rng.randint(1, {3: 2, 4: 3, 5: 4}[diff]))))
    sl = _slots(rng, dom)
    comps = []  # [label, polarity, text]
    for l in labels:
        if l in pos:
            comps.append([l, "pos", rng.choice(table[l][1]).format(**sl)])
        if l in near:
            comps.append([l, "near", rng.choice(table[l][2]).format(**sl)])
    for f in rng.sample(fill, rng.randint(3, {3: 4, 4: 6, 5: 8}[diff])):
        comps.append([None, "fill", f.format(**sl)])
    rng.shuffle(comps)
    style = rng.randrange(3)
    names = {l: _label_name(rng, l, style) for l in labels}
    defs = {l: rng.choice(table[l][0]) for l in labels}
    order = list(labels)
    rng.shuffle(order)
    instr = P(rng, f"Which labels apply to this {noun}? A label applies only if the {noun} meets its definition; several labels may apply.",
              f"Tag the {noun} with every label whose definition it satisfies.", f"Label this {noun}. Apply each label's definition literally.")
    head = {"ticket": P(rng, "Support ticket", "Incoming ticket", "Customer message"), "review": P(rng, "Customer review", "Product review"),
            "bug report": P(rng, "Bug report", "Issue filed by a customer")}[noun]

    def render(redact=None):
        body = " ".join("[one sentence removed by the privacy filter]" if i == redact else c[2] for i, c in enumerate(comps))
        return f"{head}:\n\"{body}\""

    q_style = rng.randrange(len(Q_POOL))
    state = render()
    out = []
    spec_base = {"k": "text", "kind": kind, "comps": [[c[0], c[1]] for c in comps], "redact": None}
    for l in labels:
        q = serving_q(instr, names[l], defs[l])
        gold = l in pos
        hints = {"neg": {"question": negated_q(rng, instr, names[l], defs[l]), "gold": not gold}}
        if gold:
            hints["decisive"] = [c[2] for c in comps if c[0] == l and c[1] == "pos"]
        it = item(state, noul_field(q), gold, kind, dom, dict(spec_base, ask=l), hints, group="g")
        out.append(it)
    if want_unknown:
        cand = [i for i, c in enumerate(comps) if c[1] == "pos"]
        if not cand:
            raise Retry("no pos")
        r = rng.choice(cand)
        state_u = render(redact=r)
        spec_u = dict(spec_base, redact=r)
        kids = []
        for j, l in enumerate(labels):
            s2 = dict(spec_u, ask=l)
            # a visible near-miss sentence argues against the label; keep unknown children to labels with no visible evidence
            if recheck(s2) is None and not any(c[0] == l and c[1] == "near" for c in comps):
                kids.append((j, s2))
        rl = comps[r][0]
        first = [k for k in kids if labels[k[0]] == rl]
        rest = [k for k in kids if labels[k[0]] != rl]
        rng.shuffle(rest)
        for j, s2 in (first + rest)[:3]:
            out.append(item(state_u, out[j]["field"], None, kind, dom, s2, {}, child_of=j))
    return out


# ================================================================================================ risk_flags
EU = ["France", "Germany", "Spain", "Italy", "Netherlands", "Poland", "Portugal", "Ireland"]
NON_EU = ["United Kingdom", "United States", "Canada", "Brazil", "India", "Nigeria", "Singapore", "Turkey"]
HR_COUNTRIES = ["Nigeria", "Turkey", "Brazil"]
DISPOSABLE = ["mailinator.com", "tempmail.io", "10minute.email"]
EMAILS = ["gmail.com", "outlook.com", "proton.me", "example.org"] + DISPOSABLE


def _risk_defs(rng, diff):
    t_val = rng.choice([2000, 5000, 10000])
    d_age = rng.choice([14, 30, 60])
    v_cnt = rng.choice([5, 8, 10])
    flags = {
        "high_value": ({"t": t_val}, [f"the amount is {t_val:,} or more", f"amount at or above {t_val:,}"]),
        "new_account": ({"d": d_age}, [f"the account is less than {d_age} days old", f"account age under {d_age} days (exactly {d_age} does not count)"]),
        "cross_border": ({"eu_exempt": diff >= 4}, ["billing country differs from shipping country" + (", except when both are EU member states (" + ", ".join(EU) + ")" if diff >= 4 else ""),
                                                     "shipping country is not the billing country" + ("; moves between two EU members listed in the appendix do not count" if diff >= 4 else "")]),
        "velocity": ({"v": v_cnt, "approved_only": diff >= 4}, [f"{v_cnt} or more " + ("approved " if diff >= 4 else "") + "transactions in the last 24 hours" + (" (declined attempts are not counted)" if diff >= 4 else ""),
                                                               f"at least {v_cnt} " + ("approved " if diff >= 4 else "") + "transactions within 24 hours"]),
        "name_mismatch": ({}, ["cardholder name differs from the account holder name (ignoring upper/lower case)", "cardholder and account names are not the same, case-insensitively"]),
        "pep_match": ({}, ["the screening result lists the customer as a politically exposed person", "PEP screening = match"]),
        "round_amount": ({"r": 1000}, ["the amount is an exact multiple of 1,000", "amount divisible by 1,000 with no remainder"]),
        "night_time": ({}, ["the local transaction time is from 00:00 up to and including 04:59", "placed between 00:00 and 04:59 local time"]),
        "new_device": ({}, ["the device has not been seen on this account before", "first-time device"]),
        "disposable_email": ({}, ["the email domain is one of " + ", ".join(DISPOSABLE), "email uses a disposable domain (" + ", ".join(DISPOSABLE) + ")"]),
        "high_risk_country": ({}, ["the shipping country is one of " + ", ".join(HR_COUNTRIES), "ships to a listed high-risk country (" + ", ".join(HR_COUNTRIES) + ")"]),
    }
    return flags


RISK_FIELDS = {"high_value": ["amount"], "round_amount": ["amount"], "new_account": ["account_age_days"], "cross_border": ["billing_country", "shipping_country"],
               "velocity": ["txns_24h", "declined_24h"], "name_mismatch": ["cardholder", "account_holder"], "pep_match": ["pep"], "night_time": ["local_time"],
               "new_device": ["device_seen"], "disposable_email": ["email"], "high_risk_country": ["shipping_country"]}


def _risk_eval(fl, p, rec):
    """Generator-side flag evaluation."""
    if fl == "high_value":
        return rec["amount"] >= p["t"]
    if fl == "round_amount":
        return rec["amount"] % 1000 == 0
    if fl == "new_account":
        return rec["account_age_days"] < p["d"]
    if fl == "cross_border":
        if rec["billing_country"] == rec["shipping_country"]:
            return False
        return not (p["eu_exempt"] and rec["billing_country"] in EU and rec["shipping_country"] in EU)
    if fl == "velocity":
        n = rec["txns_24h"] - (rec["declined_24h"] if p["approved_only"] else 0)
        return n >= p["v"]
    if fl == "name_mismatch":
        return rec["cardholder"].lower() != rec["account_holder"].lower()
    if fl == "pep_match":
        return rec["pep"] == "match"
    if fl == "night_time":
        return rec["local_time"] < "05:00"
    if fl == "new_device":
        return rec["device_seen"] == "no"
    if fl == "disposable_email":
        return rec["email"].split("@")[1] in DISPOSABLE
    if fl == "high_risk_country":
        return rec["shipping_country"] in HR_COUNTRIES
    raise KeyError(fl)


FIELD_VALUES = {"amount": [150, 1000, 1999, 2000, 4999, 5000, 9999, 10000, 12000, 3000], "account_age_days": [3, 13, 14, 29, 30, 59, 60, 400],
                "billing_country": EU + NON_EU, "shipping_country": EU + NON_EU, "txns_24h": [1, 4, 5, 7, 8, 9, 10, 14], "declined_24h": [0, 1, 3, 6],
                "pep": ["match", "no match"], "local_time": ["00:10", "03:30", "04:59", "05:00", "11:45", "23:59"], "device_seen": ["yes", "no"],
                "email": [f"x@{d}" for d in EMAILS], "cardholder": ["J. SMITH", "Maria Lopez"], "account_holder": ["J. Smith", "Mario Lopez"]}


def gen_risk(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    people = People(rng)
    defs = _risk_defs(rng, diff)
    n = _n_labels(rng, diff, len(defs))
    flags = rng.sample(list(defs), n)
    p = {f: defs[f][0] for f in flags}
    # record with deliberate boundary values
    name = people()
    sym = rng.choice(["€", "$", "£"])
    t = defs["high_value"][0]["t"]
    amount = rng.choice([t - 1, t, t + rng.randint(1, 3000), rng.randint(50, t - 1), rng.choice([1000, 2000, 3000, 6000, 12000])])
    d = defs["new_account"][0]["d"]
    bc = rng.choice(EU + NON_EU)
    sc = bc if chance(rng, 0.4) else rng.choice(EU + NON_EU)
    v = defs["velocity"][0]["v"]
    tx = rng.choice([v - 1, v, v + 2, rng.randint(1, v)])
    rec = {"amount": amount, "account_age_days": rng.choice([d - 1, d, d + 5, rng.randint(1, 500)]), "billing_country": bc, "shipping_country": sc,
           "txns_24h": tx, "declined_24h": rng.randint(0, min(4, tx)), "pep": "match" if chance(rng, 0.2) else "no match",
           "local_time": rng.choice(["00:10", "03:30", "04:59", "05:00", "05:01", "11:45", "23:59", f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"]),
           "device_seen": rng.choice(["yes", "no"]), "email": f"{name.split()[0].lower()}@{rng.choice(EMAILS)}",
           "account_holder": name, "cardholder": rng.choice([name, name.upper(), people()])}
    gold = {f: _risk_eval(f, p[f], rec) for f in flags}
    if all(gold.values()) or not any(gold.values()):
        raise Retry("all same")
    style = rng.randrange(3)
    names = {f: _label_name(rng, f, style) for f in flags}
    ddef = {f: rng.choice(defs[f][1]) for f in flags}
    subject = rng.choice(["card payment", "online order", "wire transfer request", "loan application"])
    labels_txt = {"amount": "Amount", "account_age_days": "Account age (days)", "billing_country": "Billing country", "shipping_country": "Shipping / destination country",
                  "txns_24h": "Transactions in last 24h (all attempts)", "declined_24h": "Of which declined", "pep": "PEP screening", "local_time": "Local time",
                  "device_seen": "Device seen before on this account", "email": "Email", "account_holder": "Account holder", "cardholder": "Cardholder name"}
    fmt = rng.randrange(2)
    art = "an" if subject[0] in "aeiou" else "a"

    def render(missing=None):
        vals = {k: (f"{sym}{rec[k]:,}" if k == "amount" else str(rec[k])) for k in rec}
        if missing:
            for m in missing:
                vals[m] = "not provided"
        lines = [P(rng, f"Screening record for {art} {subject}.", f"{subject.capitalize()} under review.")]
        if fmt == 0:
            lines.append("Record:")
            lines += [f"  {labels_txt[k]}: {vals[k]}" for k in labels_txt]
        else:
            lines.append("Record: {" + ", ".join(f'"{k}": "{vals[k]}"' for k in labels_txt) + "}")
        return "\n".join(lines)

    state = render()
    q_style = rng.randrange(len(Q_POOL))
    out = []
    spec_base = {"k": "risk", "params": p, "rec": rec, "missing": None}
    instr = P(rng, f"Which risk flags apply to this {subject}? A flag applies only when its condition is met; apply each condition literally.",
              f"Raise every risk flag whose condition holds for this {subject}.")
    for f in flags:
        q = serving_q(instr, names[f], ddef[f])
        out.append(item(state, noul_field(q), gold[f], "risk_flags", "finance", dict(spec_base, ask=f),
                        {"neg": {"question": negated_q(rng, instr, names[f], ddef[f]), "gold": not gold[f]}}, group="g"))
    if want_unknown:
        fields = sorted({x for f in flags for x in RISK_FIELDS[f]})
        rng.shuffle(fields)
        for fld in fields:
            miss = [fld]
            kids = [(j, dict(spec_base, missing=miss, ask=f)) for j, f in enumerate(flags) if fld in RISK_FIELDS[f]]
            kids = [(j, s2) for j, s2 in kids if recheck(s2) is None]
            if kids:
                st_u = render(miss)
                for j, s2 in kids[:3]:
                    out.append(item(st_u, out[j]["field"], None, "risk_flags", "finance", s2, {}, child_of=j))
                break
        else:
            raise Retry("no undetermined flag")
    return out


# ================================================================================================ contract_issues
HOME = ["England and Wales", "Ireland", "the Netherlands", "Ontario", "New South Wales"]


def gen_contract(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    home = rng.choice(HOME)
    region = rng.choice(["the EEA", "the United Kingdom", "Canada", "Australia"])
    pr = {"auto_renew": chance(rng, 0.6), "reminder_days": rng.choice([0, 14, 29, 30, 45, 60]), "cap": rng.choice([None, 100, 150, 200]),
          "term_notice": rng.choice([7, 14, 29, 30, 60, 90]), "price_change": rng.choice(["unilateral", "consent", "index_capped"]),
          "law": home if chance(rng, 0.55) else rng.choice([h for h in HOME if h != home] + ["New York", "Singapore"]),
          "exclusive": chance(rng, 0.35), "ip": rng.choice(["customer", "supplier", "licence"]),
          "data": rng.choice(["inside", "outside_scc", "outside_none"]), "pay_days": rng.choice([14, 30, 45, 60, 61, 90])}
    checks = {
        "auto_renewal_trap": ["the contract renews automatically and the supplier's renewal reminder comes less than 30 days before renewal (or not at all)",
                              "automatic renewal with under 30 days' advance reminder"],
        "uncapped_liability": ["the supplier's total liability is not capped", "no cap on the supplier's liability is stated"],
        "short_termination_notice": ["the supplier may terminate for convenience on less than 30 days' notice", "supplier termination notice shorter than 30 days"],
        "unilateral_price_change": ["the supplier may raise prices without the customer's consent and without a cap", "uncapped price increases without consent"],
        "foreign_law": [f"the governing law is not that of {home}", f"governing law other than {home}"],
        "exclusivity": ["the customer may not buy the same services from anyone else", "exclusive dealing obligation on the customer"],
        "ip_to_supplier": ["ownership of the deliverables stays with the supplier (a licence alone does not transfer ownership)",
                           "the customer does not become owner of the deliverables"],
        "unsafe_data_transfer": [f"personal data may be processed outside {region} without standard contractual clauses",
                                 f"transfers outside {region} with no SCCs"],
        "long_payment_terms": ["invoices are payable more than 60 days after receipt", "payment terms longer than 60 days"],
    }
    gold_all = {
        "auto_renewal_trap": pr["auto_renew"] and pr["reminder_days"] < 30,
        "uncapped_liability": pr["cap"] is None,
        "short_termination_notice": pr["term_notice"] < 30,
        "unilateral_price_change": pr["price_change"] == "unilateral",
        "foreign_law": pr["law"] != home,
        "exclusivity": pr["exclusive"],
        "ip_to_supplier": pr["ip"] in ("supplier", "licence"),
        "unsafe_data_transfer": pr["data"] == "outside_none",
        "long_payment_terms": pr["pay_days"] > 60,
    }
    n = _n_labels(rng, diff, len(checks))
    labels = rng.sample(list(checks), n)
    gold = {l: gold_all[l] for l in labels}
    if all(gold.values()) or not any(gold.values()):
        raise Retry("all same")
    people = People(rng)
    sup = f"{rng.choice(['Northwind', 'Kestrel', 'Halcyon', 'Juniper', 'Meridian', 'Cobalt'])} {rng.choice(['Systems', 'Services', 'Cloud', 'Logistics'])} Ltd"
    cus = f"{rng.choice(['Ashdown', 'Birchfield', 'Larkspur', 'Tidewell', 'Penrose'])} {rng.choice(['Retail', 'Health', 'Foods', 'Group'])} plc"
    clauses = {
        "term": (["Term and renewal", "Duration"],
                 (f"This Agreement runs for an initial term of {rng.choice([12, 24, 36])} months and renews automatically for successive 12-month periods unless either party gives notice. "
                  + (f"The Supplier will send a renewal reminder {pr['reminder_days']} days before each renewal date." if pr["reminder_days"] else "The Supplier is not obliged to send any renewal reminder."))
                 if pr["auto_renew"] else f"This Agreement runs for {rng.choice([12, 24, 36])} months and ends automatically at the end of that term unless the parties sign an extension."),
        "liability": (["Limitation of liability", "Liability"],
                      f"The Supplier's total aggregate liability under this Agreement shall not exceed {pr['cap']}% of the fees paid in the 12 months preceding the claim."
                      if pr["cap"] else "Nothing in this Agreement limits the amount of either party's liability for losses arising from its breach."),
        "termination": (["Termination for convenience", "Termination"],
                        f"The Supplier may terminate this Agreement for convenience by giving the Customer not less than {pr['term_notice']} days' written notice. "
                        f"The Customer may terminate on {rng.choice([30, 60, 90])} days' notice."),
        "prices": (["Charges", "Fees and price changes"],
                   {"unilateral": "The Supplier may amend its charges at any time by notice, and the amended charges apply from the next invoice.",
                    "consent": "Charges may only be changed by written agreement signed by both parties.",
                    "index_capped": f"The Supplier may increase charges once a year by no more than the published inflation index, capped at {rng.choice([3, 4, 5])}%."}[pr["price_change"]]),
        "law": (["Governing law", "Law and jurisdiction"], f"This Agreement is governed by the laws of {pr['law']}, and its courts have exclusive jurisdiction."),
        "exclusivity": (["Exclusivity", "Other suppliers"],
                        "During the term the Customer shall not procure services the same as or similar to the Services from any third party."
                        if pr["exclusive"] else "Nothing in this Agreement prevents the Customer from obtaining similar services from other suppliers."),
        "ip": (["Intellectual property", "Ownership of deliverables"],
               {"customer": "All rights in the Deliverables are assigned to the Customer on creation.",
                "supplier": "All rights in the Deliverables remain with the Supplier.",
                "licence": "The Supplier retains all rights in the Deliverables and grants the Customer a perpetual, non-exclusive licence to use them."}[pr["ip"]]),
        "data": (["Data protection", "International transfers"],
                 {"inside": f"The Supplier shall process Customer personal data only within {region}.",
                  "outside_scc": f"The Supplier may process Customer personal data outside {region}, provided the standard contractual clauses are in place.",
                  "outside_none": f"The Supplier may process Customer personal data in any country where it or its subcontractors operate, including outside {region}."}[pr["data"]]),
        "payment": (["Payment", "Invoicing"], f"The Customer shall pay each undisputed invoice within {pr['pay_days']} days of receipt."),
    }
    dep = {"auto_renewal_trap": "term", "uncapped_liability": "liability", "short_termination_notice": "termination", "unilateral_price_change": "prices",
           "foreign_law": "law", "exclusivity": "exclusivity", "ip_to_supplier": "ip", "unsafe_data_transfer": "data", "long_payment_terms": "payment"}
    corder = list(clauses)
    rng.shuffle(corder)
    style = rng.randrange(3)
    names = {l: _label_name(rng, l, style) for l in labels}
    cdef = {l: rng.choice(checks[l]) for l in labels}

    def render(missing=None):
        lines = [P(rng, f"Extract from the services agreement between {sup} (the Supplier) and {cus} (the Customer).",
                   f"Services Agreement — {sup} (\"Supplier\") and {cus} (\"Customer\"). Relevant clauses:"), ""]
        for i, c in enumerate(corder):
            t, body = clauses[c]
            head = f"{i + 1}. {t[0] if style != 1 else t[-1]}"
            lines.append(f"{head}. " + (f"[Clause {i + 1} is not included in this extract.]" if c == missing else body))
        return "\n".join(lines)

    state = render()
    q_style = rng.randrange(len(Q_POOL))
    out = []
    spec_base = {"k": "contract", "pr": pr, "home": home, "missing": None}
    instr = P(rng, f"Which review flags apply to this agreement for {cus}? Flag every item whose condition is met by the extract.",
              "Contract review: which flags apply? Each applies only if its condition is satisfied by the clauses above.")
    for l in labels:
        q = serving_q(instr, names[l], cdef[l])
        out.append(item(state, noul_field(q), gold[l], "contract_issues", "legal", dict(spec_base, ask=l),
                        {"neg": {"question": negated_q(rng, instr, names[l], cdef[l]), "gold": not gold[l]},
                         "decisive": [clauses[dep[l]][1]]}, group="g"))
    if want_unknown:
        opts = sorted({dep[l] for l in labels})
        rng.shuffle(opts)
        for c in opts:
            kids = [(j, dict(spec_base, missing=c, ask=l)) for j, l in enumerate(labels) if dep[l] == c]
            kids = [(j, s2) for j, s2 in kids if recheck(s2) is None]
            if kids:
                st_u = render(c)
                for j, s2 in kids:
                    out.append(item(st_u, out[j]["field"], None, "contract_issues", "legal", s2, {}, child_of=j))
                break
        else:
            raise Retry("no undetermined clause")
    return out


GEN = {"ticket_tags": lambda r, d, u: _gen_text(r, d, u, "ticket_tags"), "review_aspects": lambda r, d, u: _gen_text(r, d, u, "review_aspects"),
       "bug_areas": lambda r, d, u: _gen_text(r, d, u, "bug_areas"), "risk_flags": gen_risk, "contract_issues": gen_contract}


# ================================================================================================ recheck
def _risk_check(fl, p, r):
    """Independent flag logic (written separately from _risk_eval)."""
    if fl == "high_value":
        return not r["amount"] < p["t"]
    if fl == "round_amount":
        return r["amount"] / 1000 == int(r["amount"] / 1000)
    if fl == "new_account":
        return p["d"] - r["account_age_days"] > 0
    if fl == "cross_border":
        both_eu = r["billing_country"] in EU and r["shipping_country"] in EU
        return r["billing_country"] != r["shipping_country"] and not (both_eu and p["eu_exempt"])
    if fl == "velocity":
        approved = r["txns_24h"] - r["declined_24h"]
        return (approved if p["approved_only"] else r["txns_24h"]) >= p["v"]
    if fl == "name_mismatch":
        return r["cardholder"].casefold() != r["account_holder"].casefold()
    if fl == "pep_match":
        return r["pep"].strip() == "match"
    if fl == "night_time":
        h, m = map(int, r["local_time"].split(":"))
        return h * 60 + m <= 4 * 60 + 59
    if fl == "new_device":
        return r["device_seen"] != "yes"
    if fl == "disposable_email":
        return any(r["email"].endswith("@" + d) for d in DISPOSABLE)
    if fl == "high_risk_country":
        return r["shipping_country"] in set(HR_COUNTRIES)
    raise KeyError(fl)


def _contract_check(l, pr, home):
    return {"auto_renewal_trap": lambda: bool(pr["auto_renew"]) and (pr["reminder_days"] == 0 or pr["reminder_days"] < 30),
            "uncapped_liability": lambda: not pr["cap"],
            "short_termination_notice": lambda: pr["term_notice"] <= 29,
            "unilateral_price_change": lambda: pr["price_change"] not in ("consent", "index_capped"),
            "foreign_law": lambda: pr["law"] != home,
            "exclusivity": lambda: pr["exclusive"] is True,
            "ip_to_supplier": lambda: pr["ip"] != "customer",
            "unsafe_data_transfer": lambda: pr["data"] == "outside_none",
            "long_payment_terms": lambda: pr["pay_days"] >= 61}[l]()


CONTRACT_ALTS = {"term": [("auto_renew", [True, False]), ("reminder_days", [0, 14, 45])], "liability": [("cap", [None, 150])],
                 "termination": [("term_notice", [7, 90])], "prices": [("price_change", ["unilateral", "consent", "index_capped"])],
                 "law": [("law", HOME + ["New York"])], "exclusivity": [("exclusive", [True, False])], "ip": [("ip", ["customer", "supplier", "licence"])],
                 "data": [("data", ["inside", "outside_scc", "outside_none"])], "payment": [("pay_days", [30, 90])]}


def recheck(spec: dict):
    k = spec["k"]
    if k == "text":
        l = spec["ask"]
        vis_pos = any(c[0] == l and c[1] == "pos" for i, c in enumerate(spec["comps"]) if i != spec["redact"])
        if vis_pos:
            return True
        return None if spec["redact"] is not None else False
    if k == "risk":
        f = spec["ask"]
        if not spec["missing"]:
            return _risk_check(f, spec["params"][f], spec["rec"])
        outs = set()
        for m in spec["missing"]:
            vals = FIELD_VALUES[m]
            if m in ("cardholder", "account_holder"):
                other = spec["rec"]["account_holder" if m == "cardholder" else "cardholder"]
                vals = [other, other.upper(), "Zed Quarry"]
            for v in vals:
                r2 = dict(spec["rec"], **{m: v})
                if m == "declined_24h" and v > r2["txns_24h"]:
                    continue
                outs.add(_risk_check(f, spec["params"][f], r2))
        return None if len(outs) > 1 else outs.pop()
    if k == "contract":
        l = spec["ask"]
        if not spec["missing"]:
            return _contract_check(l, spec["pr"], spec["home"])
        outs = set()
        alts = CONTRACT_ALTS[spec["missing"]]
        import itertools
        for combo in itertools.product(*[vals for _, vals in alts]):
            pr2 = dict(spec["pr"], **{name: v for (name, _), v in zip(alts, combo)})
            outs.add(_contract_check(l, pr2, spec["home"]))
        return None if len(outs) > 1 else outs.pop()
    raise KeyError(k)


# ================================================================================================ enumerated completions
def worlds(spec: dict) -> list:
    """Every answer a completion of the withheld fact can give (one world when nothing is withheld):
      text      the redacted sentence may or may not be evidence for the asked label -> {True, False} unless visible evidence decides;
      risk      each missing record field over FIELD_VALUES (as recheck enumerates it);
      contract  the missing clause over CONTRACT_ALTS.
    Used by gen_traps.py to re-derive trap golds."""
    import itertools
    k = spec["k"]
    if k == "text":
        v = recheck(spec)
        return [v] if v is not None else [True, False]
    if k == "risk":
        f = spec["ask"]
        if not spec["missing"]:
            return [recheck(spec)]
        doms = []
        for m in spec["missing"]:
            vals = FIELD_VALUES[m]
            if m in ("cardholder", "account_holder"):
                other = spec["rec"]["account_holder" if m == "cardholder" else "cardholder"]
                vals = [other, other.upper(), "Zed Quarry"]
            doms.append(vals)
        out = []
        for combo in itertools.product(*doms):
            r2 = dict(spec["rec"], **dict(zip(spec["missing"], combo)))
            if "declined_24h" in spec["missing"] and r2["declined_24h"] > r2["txns_24h"]:
                continue
            out.append(_risk_check(f, spec["params"][f], r2))
        return out
    if k == "contract":
        l = spec["ask"]
        if not spec["missing"]:
            return [recheck(spec)]
        alts = CONTRACT_ALTS[spec["missing"]]
        return [_contract_check(l, dict(spec["pr"], **{name: v for (name, _), v in zip(alts, combo)}), spec["home"])
                for combo in itertools.product(*[vals for _, vals in alts])]
    raise KeyError(k)
