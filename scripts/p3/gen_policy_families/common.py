"""Shared helpers for the phase-3 policy generators (scripts/p3/gen_policy.py).

Every family module exposes

    FAMILY: str                      # candidate "family" (long_policy, rule_exception, ...)
    KINDS: list[str]                 # sub-schemas; the driver balances over them
    def generate(rng, difficulty, kind, want_unknown) -> list[Item]

An Item is a plain dict:
    state, field, gold, unknown_reason, kind, domain,
    spec        compact structured parameters (tests re-derive the gold from them independently),
    hints       optional trap hints for scripts/p3/gen_traps.py: near_miss (wrong option texts), neg (negated question + gold),
                decisive (substrings of the state that decide the answer),
    child_of    index (within the returned list) of the parent this item is an unknown variant of, or None,
    group       optional group tag (multi_label: every label item of one state shares it).

Gold is always computed by code from the structured spec, never read back from rendered text.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import random
import re
from decimal import ROUND_HALF_UP, Decimal

# ------------------------------------------------------------------------------------------------ rng
def item_rng(*parts) -> random.Random:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def P(rng: random.Random, *variants):
    """Pick one paraphrase."""
    return rng.choice(variants)


def chance(rng: random.Random, p: float) -> bool:
    return rng.random() < p


# ------------------------------------------------------------------------------------------------ names
FIRST = """Amara Bilal Chiara Dmitri Esther Farid Greta Hiroshi Ines Jonas Kavya Lars Maribel Nnamdi Olga Pedro Quentin Rania Soren Tamsin
Umar Valentina Wen Ximena Yusuf Zofia Aditi Bram Carmen Declan Elif Fionn Gabriela Hamza Isla Jae-won Kiri Leila Mateo Noor Oskar Priya
Rafael Saoirse Tobias Ulla Vikram Wanjiru Yara Zain Abebe Beatriz Callum Divya Emeka Freya Gustavo Hana Ivan Josefina Kofi Lucia Malik
Nadia Omar Paloma Rhys Sunita Thiago Uzma Vera Wiremu Xavier Yasmin Zoltan Anouk Benedikt Chidi Dagny Eamon Fatima Giorgos Helga""".split()
LAST = """Okafor Lindqvist Moreau Tanaka Haddad Kowalski Fernandes O'Brien Nakamura Achterberg Castellanos Mbeki Novak Petrov Quispe Rossi
Sandoval Thorsen Umarov Vasquez Whitfield Xu Yilmaz Zeller Abara Brennan Chakraborty Dufresne Eriksen Fitzgerald Gallagher Hosseini Ibarra
Jankowski Kapoor Lemaire Mwangi Nieminen Oyelaran Pellegrini Rahman Sato Takahashi Urquhart Valdivia Wojcik Yamamoto Zamora Adeyemi
Bergstrom Cardenas Delacroix Esposito Farrow Gunawardena Halloran Iyer Jovanovic Kariuki Lombardi Marchetti Nwosu Oduya Pacheco Reinholt
Szabo Tembo Uchida Varga Wainwright Yeboah Zhou Abernathy Bauer Costa Drummond Engel Fontaine Grünwald Holm Ishikawa Juarez""".split()

ORG_STEMS = """Northwind Bluestone Kestrel Meridian Larchmont Halcyon Brightwater Cobalt Everly Foxglove Granite Harrowgate Ironbridge Juniper
Kingsmere Lumen Marlow Nettlefield Oakhaven Pinecrest Quayside Redfern Silverline Thornbury Upland Valemont Westmarch Yarrow Zephyr Ashdown
Birchfield Coldharbour Dunmore Eastleigh Fairhaven Glenmoor Highgate Kelvedon Lowther Millbrook Norcross Oldcastle Penrose Riverton Stanway
Tidewell Underhill Wexford Alder Beacon Caldera Driftwood Ember Fernhill Greyfriars Hollowell Isling Larkspur Moorcroft Nightingale""".split()

CURRENCIES = [("EUR", "€"), ("GBP", "£"), ("USD", "$"), ("CAD", "C$"), ("AUD", "A$"), ("SGD", "S$"), ("CHF", "CHF "), ("NZD", "NZ$")]

WEEKDAYS = list(calendar.day_name)
MONTHS = list(calendar.month_name)


class People:
    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def __call__(self) -> str:
        for _ in range(200):
            p = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            if p not in self.used:
                self.used.add(p)
                return p
        raise RuntimeError("ran out of names")


def org_name(rng: random.Random, dom: dict) -> str:
    return f"{rng.choice(ORG_STEMS)} {rng.choice(dom['org_suffix'])}"


def code(rng: random.Random, prefix: str, digits: int = 5) -> str:
    return f"{prefix}-{rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"


# ------------------------------------------------------------------------------------------------ dates / money
def fd(d: dt.date, style: int) -> str:
    style %= 5
    if style == 0:
        return f"{d.day} {MONTHS[d.month]} {d.year}"
    if style == 1:
        return d.isoformat()
    if style == 2:
        return f"{WEEKDAYS[d.weekday()][:3]} {d.day} {MONTHS[d.month][:3]} {d.year}"
    if style == 3:
        return f"{MONTHS[d.month]} {d.day}, {d.year}"
    return f"{d.day} {MONTHS[d.month][:3]} {d.year}"


def fdt(t: dt.datetime, style: int) -> str:
    return f"{fd(t.date(), style)} {t:%H:%M}"


def rand_date(rng: random.Random, lo: dt.date = dt.date(2025, 1, 6), hi: dt.date = dt.date(2027, 10, 30)) -> dt.date:
    return lo + dt.timedelta(days=rng.randrange((hi - lo).days))


def weekday_on_or_after(d: dt.date, closed=()) -> dt.date:
    while d.weekday() >= 5 or d in closed:
        d += dt.timedelta(days=1)
    return d


def add_bdays(d: dt.date, n: int, closed=()) -> dt.date:
    """The n-th business day after d (weekends and `closed` dates skipped)."""
    cur, k = d, 0
    while k < n:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5 and cur not in closed:
            k += 1
    return cur


def add_days(d: dt.date, n: int) -> dt.date:
    return d + dt.timedelta(days=n)


D2 = Decimal("0.01")


def q2(x) -> Decimal:
    return Decimal(x).quantize(D2, rounding=ROUND_HALF_UP)


def money(x, sym: str, cents: bool | None = None) -> str:
    x = q2(x)
    if cents is False or (cents is None and x == x.to_integral_value()):
        return f"{sym}{int(x):,}"
    return f"{sym}{x:,.2f}"


def num_words(n: int) -> str:
    words = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    if 0 <= n <= 20:
        return words[n]
    tens = {30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 90: "ninety"}
    return tens.get(n, str(n))


def n_days(rng: random.Random, n: int, unit: str = "") -> str:
    """'14 days' / 'fourteen (14) days' / '14 business days'."""
    u = f"{unit} days" if unit else "days"
    if n == 1:
        u = f"{unit} day" if unit else "day"
    if n <= 20 and chance(rng, 0.3):
        return f"{num_words(n)} ({n}) {u}"
    return f"{n} {u}"


# ------------------------------------------------------------------------------------------------ text utils
def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def approx_tokens(text: str) -> int:
    """Token estimate for English prose (Qwen ~3.3-4 chars/token). NOT conservative on logs, ids and tables, where the real
    tokenizer counts up to ~1.8x more; long_input uses its own calibrated table and gen_traps uses len/1.8 for its cap."""
    return int(len(text) / 3.3) + 1


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def option_key(text: str, taken: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")[:40] or "option"
    if key[0].isdigit():
        key = ("v_" + key)[:40]
    if key in ("unknown", "__unknown__", "option_unknown"):
        key = "opt_" + key
    base, n = key, 2
    while key in taken:
        suffix = f"_{n}"
        key = base[:40 - len(suffix)] + suffix
        n += 1
    taken.add(key)
    return key


def choice_field(rng: random.Random, question: str, correct: str, wrong: list[str], k: int | None = None,
                 desc: dict[str, str] | None = None) -> tuple[dict, str]:
    """(field, gold_key). Options = the correct text + distinct wrong texts, shuffled. Wrong texts equal to the correct one
    (case-insensitive) are dropped, so the gold stays unique."""
    seen, opts = {correct.strip().lower()}, [correct]
    for w in wrong:
        if w.strip().lower() not in seen:
            seen.add(w.strip().lower())
            opts.append(w)
        if k and len(opts) >= k:
            break
    if len(opts) < 2:
        raise ValueError("need at least one wrong option")
    rng.shuffle(opts)
    taken: set[str] = set()
    out, gold = [], None
    for t in opts:
        key = option_key(t, taken)
        o = {"key": key, "text": t}
        if desc and t in desc:
            o["description"] = desc[t]
        out.append(o)
        if t == correct:
            gold = key
    return {"type": "choice", "question": question, "options": out}, gold


def ordered_choice_field(question: str, texts: list[str], correct_index: int, desc: list[str] | None = None) -> tuple[dict, str]:
    """Choice field with a fixed option order (lists of records, actions, UI elements)."""
    taken: set[str] = set()
    out = []
    for i, t in enumerate(texts):
        o = {"key": option_key(t, taken), "text": t}
        if desc:
            o["description"] = desc[i]
        out.append(o)
    return {"type": "choice", "question": question, "options": out}, out[correct_index]["key"]


def noul_field(question: str) -> dict:
    return {"type": "noul", "question": question}


def score_field(question: str, level_desc: list[str]) -> dict:
    return {"type": "score", "question": question, "levels": [{"value": i, "description": d} for i, d in enumerate(level_desc)]}


def item(state, field, gold, kind: str, domain: str, spec: dict, hints: dict | None = None, child_of: int | None = None,
         unknown_reason: str | None = None, group: str | None = None) -> dict:
    if gold is None and unknown_reason is None:
        unknown_reason = "insufficient_evidence"
    return {"state": state, "field": field, "gold": gold, "unknown_reason": unknown_reason if gold is None else None,
            "kind": kind, "domain": domain, "spec": spec, "hints": hints or {}, "child_of": child_of, "group": group}


# ------------------------------------------------------------------------------------------------ domains
# One vocabulary per domain; families pick what they need. Keep lists long so wording varies across items.
DOMAINS: dict[str, dict] = {
    "retail": dict(
        org_suffix=["Stores", "Retail Group", "Outfitters", "Home & Living", "Department Stores", "Marketplace"],
        party="customer", parties="customers", member="loyalty member", tier_name="loyalty tier",
        tiers=["Standard", "Silver", "Gold", "Platinum"],
        request="return request", requests="return requests", event="delivery", event_past="delivered",
        categories=["small electrical goods", "large appliances", "clothing and footwear", "perishable groceries", "made-to-order furniture",
                    "clearance items", "cosmetics", "toys and games", "garden equipment", "mobile phones"],
        depts=["Customer Services", "Returns Desk", "Loss Prevention", "Finance", "E-commerce Operations", "Merchandising"],
        roles=["Customer Service Adviser", "Returns Supervisor", "Store Manager", "Area Manager", "Regional Director", "Chief Commercial Officer"],
        systems=["the returns portal", "the point-of-sale system", "the order management system", "the CRM"],
        ref="RMA", amount_noun="purchase price", doc_nouns=["Returns and Refunds Policy", "Customer Terms of Sale", "Returns Desk Procedure"],
        outcomes=["issue a full refund to the original payment method", "refund less a restocking fee", "offer store credit only",
                  "decline the return", "refer the case to the Returns Supervisor"],
        flags=[("the item was faulty or damaged on arrival", "faulty on arrival"), ("proof of purchase is provided", "proof of purchase"),
               ("the item is unopened with the manufacturer's seal intact", "unopened"), ("the order was placed online", "online order")],
    ),
    "logistics": dict(
        org_suffix=["Freight", "Logistics", "Haulage", "Shipping Lines", "Distribution", "Cargo Services"],
        party="shipper", parties="shippers", member="contract account", tier_name="account band",
        tiers=["Spot", "Bronze", "Silver", "Strategic"],
        request="damage claim", requests="damage claims", event="delivery", event_past="delivered",
        categories=["palletised dry goods", "temperature-controlled cargo", "hazardous materials", "oversized loads", "parcel freight",
                    "high-value electronics", "bulk liquids", "livestock feed", "automotive parts", "pharmaceuticals"],
        depts=["Claims Unit", "Operations Control", "Customer Accounts", "Fleet Management", "Warehouse Operations", "Compliance"],
        roles=["Claims Handler", "Senior Claims Handler", "Claims Manager", "Head of Operations", "Operations Director", "Chief Executive"],
        systems=["the claims portal", "the transport management system", "the warehouse management system", "the tracking platform"],
        ref="CLM", amount_noun="declared value", doc_nouns=["Conditions of Carriage", "Claims Handling Procedure", "Master Services Agreement"],
        outcomes=["pay the claim in full", "pay the claim subject to the liability cap", "reject the claim", "refer the claim to the Claims Manager",
                  "offer a freight credit instead of payment"],
        flags=[("the damage was noted on the proof of delivery", "noted on POD"), ("photographs were supplied with the claim", "photos supplied"),
               ("the goods were packed by the carrier", "carrier-packed"), ("the consignee signed 'unchecked'", "signed unchecked")],
    ),
    "hr": dict(
        org_suffix=["Group", "Holdings", "Industries", "Partners", "Technologies", "Foods"],
        party="employee", parties="employees", member="permanent employee", tier_name="grade",
        tiers=["Grade A", "Grade B", "Grade C", "Grade D"],
        request="leave request", requests="leave requests", event="qualifying event", event_past="occurred",
        categories=["annual leave", "parental leave", "compassionate leave", "study leave", "unpaid leave", "sabbatical", "jury service",
                    "volunteering leave", "medical appointments", "relocation days"],
        depts=["People Team", "Payroll", "Talent Acquisition", "Employee Relations", "Learning and Development", "Workforce Planning"],
        roles=["Line Manager", "HR Business Partner", "Department Head", "HR Director", "Chief People Officer", "Chief Executive"],
        systems=["the HR information system", "the payroll platform", "the absence tracker", "the intranet"],
        ref="HR", amount_noun="salary", doc_nouns=["Leave and Absence Policy", "Employee Handbook", "Flexible Working Procedure"],
        outcomes=["approve the request with full pay", "approve the request as unpaid leave", "decline the request",
                  "refer the request to the HR Business Partner", "approve half the requested days"],
        flags=[("the employee has completed probation", "probation complete"), ("supporting evidence was attached", "evidence attached"),
               ("the request was made through the HR system", "submitted in system"), ("the employee works part-time", "part-time")],
    ),
    "healthcare_admin": dict(
        org_suffix=["Health Partners", "Medical Group", "Clinics", "Community Health Trust", "Care Network", "Diagnostics"],
        party="patient", parties="patients", member="registered patient", tier_name="coverage plan",
        tiers=["Essential", "Standard", "Enhanced", "Complete"],
        request="billing adjustment request", requests="billing adjustment requests", event="date of service", event_past="took place",
        categories=["outpatient imaging", "physiotherapy", "specialist consultations", "laboratory tests", "day surgery", "dental hygiene",
                    "mental health sessions", "vaccinations", "home visits", "telehealth appointments"],
        depts=["Patient Accounts", "Referrals Office", "Medical Records", "Clinical Governance", "Revenue Cycle", "Patient Experience"],
        roles=["Patient Accounts Officer", "Billing Team Lead", "Practice Manager", "Clinical Director", "Medical Director", "Chief Operating Officer"],
        systems=["the patient administration system", "the billing platform", "the referral management system", "the patient portal"],
        ref="PAT", amount_noun="invoiced amount", doc_nouns=["Patient Billing Policy", "Referral and Authorisation Procedure", "Patient Charter"],
        outcomes=["waive the charge in full", "reduce the charge to the plan rate", "uphold the charge", "refer the request to the Practice Manager",
                  "set up an instalment plan"],
        flags=[("a referral letter is on file", "referral on file"), ("the appointment was cancelled by the clinic", "clinic-cancelled"),
               ("the patient holds an exemption certificate", "exemption certificate"), ("prior authorisation was obtained", "prior authorisation")],
    ),
    "insurance": dict(
        org_suffix=["Insurance", "Mutual", "Assurance", "Underwriting", "General Insurance", "Cover"],
        party="policyholder", parties="policyholders", member="policyholder", tier_name="cover level",
        tiers=["Basic", "Standard", "Plus", "Premier"],
        request="claim", requests="claims", event="date of loss", event_past="occurred",
        categories=["accidental damage", "theft", "water damage", "storm damage", "fire", "personal liability", "travel delay",
                    "lost baggage", "medical expenses", "vehicle glass"],
        depts=["Claims", "Underwriting", "Fraud Investigation", "Customer Retention", "Loss Adjusting", "Complaints"],
        roles=["Claims Assessor", "Senior Assessor", "Claims Team Leader", "Head of Claims", "Chief Underwriter", "Claims Director"],
        systems=["the claims system", "the policy administration system", "the document portal", "the fraud screening tool"],
        ref="CLM", amount_noun="claimed amount", doc_nouns=["Policy Wording", "Claims Settlement Guidelines", "Schedule of Cover"],
        outcomes=["settle the claim in full", "settle the claim less the excess", "decline the claim", "refer the claim to Fraud Investigation",
                  "settle up to the sub-limit"],
        flags=[("a police or incident report number was supplied", "report supplied"), ("the loss was reported within 48 hours", "reported promptly"),
               ("the property was left unoccupied for more than 30 days", "unoccupied"), ("receipts were provided", "receipts provided")],
    ),
    "finance": dict(
        org_suffix=["Bank", "Capital", "Credit Union", "Finance", "Payments", "Building Society"],
        party="client", parties="clients", member="premium account holder", tier_name="account tier",
        tiers=["Everyday", "Plus", "Premier", "Private"],
        request="fee waiver request", requests="fee waiver requests", event="transaction date", event_past="posted",
        categories=["overdraft fees", "foreign transaction fees", "late payment charges", "wire transfer fees", "card replacement fees",
                    "returned payment fees", "account maintenance fees", "cash advance fees", "stop payment fees", "statement copy fees"],
        depts=["Client Services", "Credit Risk", "Payments Operations", "Complaints Handling", "Financial Crime", "Retail Banking"],
        roles=["Client Services Officer", "Team Supervisor", "Branch Manager", "Regional Credit Manager", "Head of Retail", "Chief Risk Officer"],
        systems=["the core banking system", "the case management tool", "online banking", "the payments hub"],
        ref="FWR", amount_noun="fee amount", doc_nouns=["Fees and Charges Policy", "Account Terms and Conditions", "Complaint Resolution Procedure"],
        outcomes=["refund the fee in full", "refund 50% of the fee", "decline the waiver", "refer the request to the Branch Manager",
                  "convert the fee to a goodwill credit"],
        flags=[("the client has a vulnerability marker on file", "vulnerability marker"), ("the fee arose from a bank system error", "bank error"),
               ("the account is in arrears", "in arrears"), ("the client requested the waiver in writing", "written request")],
    ),
    "saas_ops": dict(
        org_suffix=["Cloud", "Software", "Labs", "Systems", "Platforms", "Data"],
        party="customer", parties="customers", member="annual subscriber", tier_name="plan",
        tiers=["Starter", "Growth", "Business", "Enterprise"],
        request="service credit claim", requests="service credit claims", event="incident", event_past="was resolved",
        categories=["API gateway", "web dashboard", "data export service", "authentication service", "reporting module", "mobile SDK",
                    "webhook delivery", "search index", "file storage", "billing portal"],
        depts=["Site Reliability", "Customer Success", "Billing Operations", "Security", "Support Engineering", "Product Operations"],
        roles=["Support Engineer", "Support Team Lead", "Customer Success Manager", "Head of Support", "VP Engineering", "Chief Technology Officer"],
        systems=["the status page", "the support desk", "the billing system", "the incident tracker"],
        ref="SC", amount_noun="monthly subscription fee", doc_nouns=["Service Level Agreement", "Subscription Terms", "Incident Response Runbook"],
        outcomes=["grant the service credit in full", "grant the credit subject to the cap", "reject the claim",
                  "escalate the claim to the Customer Success Manager", "grant a pro-rated credit"],
        flags=[("the outage was caused by the customer's own integration", "customer-caused"), ("the claim cites the incident ID", "incident cited"),
               ("the customer is on an annual prepaid contract", "annual prepaid"), ("the downtime fell in a scheduled maintenance window", "maintenance window")],
    ),
    "travel": dict(
        org_suffix=["Airways", "Rail", "Travel", "Coaches", "Ferries", "Holidays"],
        party="passenger", parties="passengers", member="frequent traveller member", tier_name="membership status",
        tiers=["Blue", "Silver", "Gold", "Diamond"],
        request="compensation claim", requests="compensation claims", event="scheduled departure", event_past="was scheduled",
        categories=["domestic flights", "short-haul international flights", "long-haul flights", "sleeper trains", "regional rail",
                    "coach tours", "ferry crossings", "package holidays", "hotel-only bookings", "car hire"],
        depts=["Customer Relations", "Revenue Accounting", "Ground Operations", "Loyalty Programme", "Travel Support", "Legal Affairs"],
        roles=["Customer Relations Agent", "Senior Agent", "Customer Relations Manager", "Head of Customer Care", "Commercial Director", "Chief Executive"],
        systems=["the booking engine", "the claims form", "the loyalty platform", "the disruption tool"],
        ref="TRV", amount_noun="fare paid", doc_nouns=["Conditions of Carriage", "Passenger Compensation Scheme", "Booking Terms"],
        outcomes=["pay cash compensation", "issue a travel voucher", "refund the unused fare", "decline the claim",
                  "refer the claim to Customer Relations Manager"],
        flags=[("the disruption was caused by extraordinary circumstances", "extraordinary circumstances"), ("the passenger accepted rerouting", "accepted rerouting"),
               ("the booking was made through a third-party agent", "third-party booking"), ("the passenger checked in on time", "checked in on time")],
    ),
    "public_sector": dict(
        org_suffix=["Borough Council", "County Council", "City Authority", "Regional Agency", "Licensing Authority", "Department of Housing"],
        party="applicant", parties="applicants", member="registered resident", tier_name="band",
        tiers=["Band 1", "Band 2", "Band 3", "Band 4"],
        request="application", requests="applications", event="decision notice", event_past="was issued",
        categories=["street trading licences", "residential parking permits", "housing repair grants", "planning appeals",
                    "business rates relief", "event licences", "taxi driver licences", "community grants", "waste collection exemptions", "building control"],
        depts=["Licensing", "Revenues and Benefits", "Planning Services", "Housing Services", "Environmental Health", "Customer Contact Centre"],
        roles=["Case Officer", "Senior Case Officer", "Team Manager", "Service Head", "Director of Services", "Chief Executive"],
        systems=["the online applications portal", "the case management system", "the payments gateway", "the public register"],
        ref="APP", amount_noun="fee", doc_nouns=["Licensing Policy", "Fees and Exemptions Scheme", "Customer Service Standards"],
        outcomes=["grant the application", "grant the application with conditions", "refuse the application",
                  "refer the application to the licensing committee", "defer pending further information"],
        flags=[("the applicant receives a means-tested benefit", "means-tested benefit"), ("the site is in a conservation area", "conservation area"),
               ("objections were received during consultation", "objections received"), ("the applicant holds a current licence", "current licence")],
    ),
    "legal": dict(
        org_suffix=["LLP", "& Partners", "Legal", "Solicitors", "Chambers", "Law Group"],
        party="client", parties="clients", member="retained client", tier_name="engagement type",
        tiers=["Ad hoc", "Standard retainer", "Enhanced retainer", "Strategic partner"],
        request="fee dispute", requests="fee disputes", event="invoice date", event_past="was issued",
        categories=["litigation", "conveyancing", "employment advice", "corporate transactions", "family matters", "wills and probate",
                    "intellectual property", "regulatory investigations", "immigration", "commercial leases"],
        depts=["Client Accounts", "Risk and Compliance", "Knowledge Management", "Practice Support", "Billing", "Conflicts Team"],
        roles=["Associate", "Senior Associate", "Supervising Partner", "Practice Group Head", "Managing Partner", "General Counsel"],
        systems=["the matter management system", "the time recording system", "the document management system", "the client portal"],
        ref="MAT", amount_noun="invoice total", doc_nouns=["Terms of Engagement", "Client Care Letter", "Billing and Fee Dispute Procedure"],
        outcomes=["write off the disputed amount", "reduce the invoice by the disputed time entries", "uphold the invoice",
                  "refer the dispute to the Managing Partner", "offer mediation"],
        flags=[("the client raised the dispute in writing", "written dispute"), ("the work was outside the agreed scope", "out of scope"),
               ("the client approved the estimate", "estimate approved"), ("the matter is subject to a fixed-fee agreement", "fixed fee")],
    ),
}
DOMAIN_IDS = list(DOMAINS)

DOC_KINDS = ["policy", "contract", "sop", "terms", "schedule", "incident_log"]


def closure_days(rng: random.Random, year: int, k: int = 6) -> list[dt.date]:
    """k plausible non-weekend closure dates in a year (public holidays of an invented calendar)."""
    fixed = [(1, 1), (1, 2), (4, 18), (4, 21), (5, 1), (5, 26), (8, 25), (10, 12), (11, 11), (12, 25), (12, 26), (6, 9), (3, 17), (7, 1)]
    rng.shuffle(fixed)
    out = []
    for m, d in fixed:
        x = dt.date(year, m, d)
        if x.weekday() < 5:
            out.append(x)
        if len(out) >= k:
            break
    return sorted(out)
