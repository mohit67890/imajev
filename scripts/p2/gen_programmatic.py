"""Phase 2c: deterministic, seeded generators for EXACT-ANSWER families (no model in the loop).

    python scripts/p2/gen_programmatic.py --docs 3000 --seed prog-v1 --out data/decision-p2/teacher/writer-prog.jsonl

Three families, each a set of question KINDS (a computation with randomised entities, numbers, dates and a domain from
`domains.py`) rendered through several question wordings; a TEMPLATE is one (kind, question type, wording) triple:

- `temporal_arithmetic`   business-day deadlines with closure days, calendar roll-overs, time-zone ordering and cut-offs,
                          business-hour SLAs, notice periods with deemed postal receipt, recurring schedules, grace periods.
- `probability_exact`     small-number combinatorics and conditional probability: draws without replacement, 2x2 tables,
                          Bayes with natural frequencies, independent events, counting, expected values, exact bands.
- `numeric_reconciliation` invoice totals (discount, tax, shipping), PO-vs-invoice mismatches, bank reconciliations,
                          stacked discounts, unit conversions, proration, VAT extraction, budgets, FX fees, stock roll-forwards.

Every row has exactly the writer-output schema of `gen_write.py` ({doc_id, batch, plan, writer, output{document, document_kind,
questions[3]}}) so `gen_answer.py` and `assemble_p2.py` consume it unchanged. The intended answer is computed by code; each
question carries a `justification` with the computation, and `plan.params[i]` stores the inputs, the canonical answer and the
canonical value of every option so a test can recompute it independently. Batch tags are `prog-<family>`.

Distractors are the typical mistakes (calendar instead of business days, local time read as UTC, draws with replacement, P(B|A)
for P(A|B), tax on the undiscounted subtotal, additive instead of stacked discounts, ...). A small share of questions withhold a
needed input (marked pending in the document) and are intended `unknown` (insufficient_evidence).

Templated documents share boilerplate by construction, so assemble them with the within-batch near-duplicate lint disabled:
`assemble_p2.py --batch-filter 'prog-*' --dedupe-threshold 100000` (the JevBench 8-gram lint still runs; this script also
checks it and regenerates any document that shares a single 8-gram with the public JevBench files).
"""
from __future__ import annotations
import argparse, calendar, datetime as dt, hashlib, json, math, random, sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import WriterOutput, contamination_count, jevbench_public_files, option_key, reference_ngrams  # noqa: E402
from domains import DOMAINS  # noqa: E402

GENERATOR = {"model": "programmatic", "generator": "scripts/p2/gen_programmatic.py", "version": "prog-v1", "family": "programmatic"}
PROG_FAMILIES = ("temporal_arithmetic", "probability_exact", "numeric_reconciliation")
BATCH = {"temporal_arithmetic": "prog-temporal", "probability_exact": "prog-probability", "numeric_reconciliation": "prog-numeric"}
UNKNOWN_P = 0.25  # per question, for kinds that can withhold an input

# ------------------------------------------------------------------------------------------------- entities and domain vocabulary
FIRST = ["Amara", "Bastian", "Céline", "Dario", "Elin", "Farid", "Greta", "Hollis", "Ines", "Jonah", "Keiko", "Lorcan", "Maren", "Nikhil",
         "Odile", "Pavel", "Quinn", "Rosalind", "Soren", "Tamsin", "Ulla", "Viren", "Wren", "Xiomara", "Yusuf", "Zelda", "Anouk", "Bram",
         "Cosima", "Desmond", "Esme", "Florian", "Gideon", "Halima", "Ivo", "Jolene", "Kasimir", "Leonie", "Mateo", "Noor"]
LAST = ["Achterberg", "Brannigan", "Castellano", "Dunmore", "Eskildsen", "Fairweather", "Galloway", "Hartigan", "Ingersoll", "Jaskolski",
        "Kovalenko", "Lindqvist", "Marchetti", "Nakashima", "Okonkwo", "Pemberton", "Quarshie", "Rasmussen", "Szabo", "Thorne",
        "Umberti", "Valcourt", "Whitlock", "Yarrow", "Zeller", "Abernethy", "Bellamy", "Coldwell", "Delacroix", "Ekwueme"]
ORG_STEMS = ["Norvale", "Brightfen", "Calder Reach", "Oakhollow", "Tessmere", "Rivenhall", "Glenmarsh", "Harrowgate", "Silverbeck",
             "Kestrel Point", "Amberlea", "Duskwater", "Fernhill", "Larkspur", "Mossbank", "Quillon", "Redstart", "Westerby"]

V = lambda case, cases, prefix, org, team, items, events: dict(case=case, cases=cases, prefix=prefix, org=org, team=team, items=items, events=events)  # noqa: E731
DOMAIN_VOCAB: dict[str, dict] = {
    "ecommerce_listings": V("order", "orders", "ORD", "Marketplace", "seller support", [("ceramic mug set", 12, 40), ("desk lamp", 18, 75), ("phone stand", 6, 22), ("yoga mat", 15, 48), ("canvas backpack", 25, 90)], ["buyer complaint", "listing takedown", "refund request"]),
    "logistics": V("shipment", "shipments", "SHP", "Freight", "dispatch", [("pallet handling", 18, 45), ("cold-storage day", 30, 80), ("customs filing", 45, 120), ("carton pack", 4, 12), ("liftgate delivery", 35, 90)], ["pickup", "customs release", "delivery attempt"]),
    "insurance_claims": V("claim", "claims", "CLM", "Mutual", "claims handling", [("windscreen replacement", 180, 520), ("towing", 60, 180), ("rental car day", 35, 70), ("water extraction", 250, 900), ("drywall repair", 120, 600)], ["first notice of loss", "adjuster visit", "settlement offer"]),
    "hr_policy": V("expense claim", "expense claims", "EXP", "Group", "people operations", [("hotel night", 85, 240), ("rail ticket", 20, 140), ("client dinner", 40, 180), ("taxi fare", 12, 60), ("conference pass", 150, 700)], ["leave request", "grievance filing", "expense submission"]),
    "it_support": V("ticket", "tickets", "INC", "Systems", "service desk", [("laptop dock", 90, 230), ("licence seat", 8, 45), ("headset", 25, 110), ("monitor", 140, 380), ("SSD upgrade", 60, 180)], ["access request", "outage report", "password reset"]),
    "finance_ops": V("invoice", "invoices", "INV", "Holdings", "accounts payable", [("consulting hour", 90, 220), ("audit support day", 600, 1400), ("bookkeeping retainer", 300, 900), ("payroll run", 120, 400), ("tax filing", 200, 750)], ["payment run", "approval request", "vendor query"]),
    "healthcare_admin": V("referral", "referrals", "REF", "Clinic", "patient services", [("consultation slot", 60, 180), ("imaging fee", 120, 450), ("lab panel", 35, 140), ("follow-up visit", 45, 120), ("records copy fee", 10, 30)], ["referral intake", "appointment request", "billing query"]),
    "travel": V("booking", "bookings", "BKG", "Journeys", "travel desk", [("economy fare", 90, 480), ("hotel night", 80, 260), ("airport transfer", 25, 70), ("seat upgrade", 30, 150), ("checked bag fee", 20, 60)], ["change request", "cancellation", "rebooking"]),
    "contracts": V("agreement", "agreements", "AGR", "Services", "contract management", [("support hour", 70, 160), ("on-site visit", 250, 700), ("licence module", 400, 1500), ("training day", 600, 1300), ("retainer month", 900, 2500)], ["termination notice", "renewal", "amendment"]),
    "education": V("enrolment", "enrolments", "ENR", "Academy", "registry", [("course module", 150, 600), ("lab kit", 25, 90), ("exam fee", 40, 160), ("course reader", 15, 55), ("tutoring hour", 30, 75)], ["extension request", "assignment submission", "appeal"]),
    "real_estate": V("lease", "leases", "LSE", "Properties", "lettings", [("monthly rent", 700, 2400), ("parking bay", 40, 150), ("storage unit", 30, 90), ("cleaning visit", 60, 160), ("key replacement", 20, 65)], ["inspection", "notice to vacate", "deposit claim"]),
    "hospitality": V("reservation", "reservations", "RSV", "Hotels", "front office", [("deluxe room night", 120, 380), ("breakfast cover", 12, 28), ("late checkout", 20, 60), ("spa session", 55, 160), ("meeting room hour", 30, 95)], ["booking", "incident report", "group enquiry"]),
    "telecom": V("account", "accounts", "ACC", "Mobile", "customer care", [("data add-on", 5, 25), ("roaming day pass", 4, 15), ("handset instalment", 18, 60), ("line rental", 10, 35), ("minutes bundle", 5, 20)], ["outage credit request", "dispute", "port-out request"]),
    "energy": V("meter account", "meter accounts", "MTR", "Energy", "metering", [("standing charge day", 0.3, 0.9), ("meter inspection", 40, 120), ("reconnection fee", 60, 180), ("smart meter install", 80, 200), ("tariff switch fee", 10, 40)], ["meter reading", "outage", "safety visit"]),
    "government_forms": V("application", "applications", "APP", "Borough Council", "licensing", [("filing fee", 25, 150), ("inspection fee", 60, 240), ("certified copy", 8, 30), ("expedite fee", 40, 120), ("late filing surcharge", 15, 75)], ["permit application", "objection", "renewal filing"]),
    "manufacturing_qa": V("batch", "batches", "BAT", "Components", "quality assurance", [("machined bracket", 3, 18), ("gasket", 0.5, 4), ("pump housing", 25, 90), ("fastener kit", 6, 20), ("sensor module", 30, 110)], ["non-conformance report", "batch release", "customer return"]),
    "retail_returns": V("return", "returns", "RMA", "Outfitters", "returns desk", [("rain jacket", 60, 220), ("hiking boots", 80, 240), ("fleece", 35, 110), ("water bottle", 10, 35), ("daypack", 40, 130)], ["return request", "warranty claim", "exchange"]),
    "saas_billing": V("subscription", "subscriptions", "SUB", "Cloud", "billing operations", [("Pro seat", 12, 45), ("storage block", 5, 30), ("API overage pack", 20, 90), ("SSO add-on", 50, 200), ("premium support plan", 100, 500)], ["plan change", "seat increase", "dunning notice"]),
    "security_incidents": V("incident", "incidents", "SEC", "Secure", "security operations", [("forensics hour", 150, 350), ("endpoint licence", 3, 12), ("log retention block", 40, 160), ("penetration test day", 900, 2000), ("incident retainer month", 1000, 3000)], ["alert", "containment action", "access review"]),
    "procurement": V("purchase order", "purchase orders", "PO", "Industrial", "procurement", [("safety gloves box", 9, 30), ("hi-vis vest", 4, 15), ("pallet jack", 250, 700), ("shelving unit", 80, 260), ("label printer", 150, 480)], ["RFQ", "goods receipt", "supplier query"]),
    "events": V("event booking", "event bookings", "EVT", "Venues", "events office", [("banquet cover", 35, 110), ("AV package", 300, 1200), ("stage hire", 400, 1500), ("coffee break", 6, 18), ("security staff hour", 25, 55)], ["site visit", "final numbers", "vendor cutoff"]),
    "fleet": V("vehicle", "vehicles", "VEH", "Fleet", "fleet maintenance", [("oil service", 70, 180), ("brake pad set", 60, 200), ("tyre", 80, 260), ("wiper set", 15, 45), ("diagnostic check", 40, 120)], ["service booking", "breakdown", "inspection"]),
    "agriculture_supply": V("lot", "lots", "LOT", "Growers", "grading", [("crate of apples", 14, 40), ("sack of potatoes", 9, 25), ("tray of berries", 12, 35), ("bale of hay", 5, 15), ("bag of seed", 20, 70)], ["harvest intake", "grading check", "cold-chain alarm"]),
    "nonprofit_grants": V("grant", "grants", "GRT", "Foundation", "grants team", [("workshop session", 150, 600), ("volunteer stipend", 40, 150), ("printed materials", 60, 300), ("venue hire", 200, 800), ("travel reimbursement", 30, 250)], ["grant report", "budget variation", "site visit"]),
}
assert set(DOMAIN_VOCAB) == {d["id"] for d in DOMAINS}

CURRENCIES = [("EUR", "€"), ("GBP", "£"), ("USD", "$"), ("CAD", "C$"), ("AUD", "A$"), ("CHF", "CHF ")]
# (site, UTC offset in minutes) -- the document always states the offset in force, so no daylight-saving knowledge is needed
SITES = [("Lisbon", 0), ("Reykjavik", 0), ("Madrid", 60), ("Warsaw", 60), ("Athens", 120), ("Nairobi", 180), ("Dubai", 240), ("Karachi", 300),
         ("Mumbai", 330), ("Kathmandu", 345), ("Dhaka", 360), ("Bangkok", 420), ("Singapore", 480), ("Osaka", 540), ("Adelaide", 570),
         ("Brisbane", 600), ("Noumea", 660), ("Auckland", 720), ("Sao Paulo", -180), ("Halifax", -240), ("Bogota", -300), ("Denver", -420),
         ("Chicago", -360), ("Anchorage", -540), ("Honolulu", -600)]
WEEKDAYS = list(calendar.day_name)
MONTHS = list(calendar.month_name)

D2 = Decimal("0.01")


def q2(x) -> Decimal:
    return Decimal(x).quantize(D2, rounding=ROUND_HALF_UP)


def money(x: Decimal, sym: str) -> str:
    return f"{sym}{q2(x):,.2f}"


def fd(d: dt.date, style: int = 0) -> str:
    return [f"{d.day} {MONTHS[d.month]} {d.year}", d.isoformat(), f"{WEEKDAYS[d.weekday()][:3]} {d.day} {MONTHS[d.month][:3]} {d.year}",
            f"{MONTHS[d.month]} {d.day}, {d.year}"][style % 4]


def fdt(t: dt.datetime, style: int = 0) -> str:
    return f"{fd(t.date(), style)}, {t:%H:%M}"


def fmt_off(m: int) -> str:
    sign = "+" if m >= 0 else "-"; m = abs(m)
    return f"UTC{sign}{m // 60:02d}:{m % 60:02d}"


def add_bdays(d: dt.date, n: int, closed=()) -> dt.date:
    """n business days after d (day 1 = first business day after d); weekends and `closed` dates do not count."""
    cur, k = d, 0
    while k < n:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5 and cur not in closed:
            k += 1
    return cur


def next_bday(d: dt.date, closed=()) -> dt.date:
    while d.weekday() >= 5 or d in closed:
        d += dt.timedelta(days=1)
    return d


def add_months(d: dt.date, n: int) -> dt.date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def fr(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def pct(f: Fraction, dp: int = 1) -> str:
    v = (Decimal(f.numerator) * 100 / Decimal(f.denominator)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return f"{v}%"


def pct_exact(f: Fraction) -> str:
    """An exact percentage with the fewest decimals (only for fractions whose percentage terminates within 4 places)."""
    v = Decimal(f.numerator) * 100 / Decimal(f.denominator)
    s = f"{v.quantize(Decimal('0.0001')):f}".rstrip("0").rstrip(".")
    return f"{s}%"


# ------------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, rng: random.Random, dom: dict, family: str):
        self.rng, self.dom, self.family = rng, dom, family
        self.v = DOMAIN_VOCAB[dom["id"]]
        self.cur, self.sym = rng.choice(CURRENCIES)
        self.org = f"{rng.choice(ORG_STEMS)} {self.v['org']}"
        self.dstyle = rng.randrange(4)
        self._people: list[str] = []
        self.ref = self.code(self.v["prefix"])

    def person(self) -> str:
        while True:
            p = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            if p not in self._people:
                self._people.append(p); return p

    def code(self, prefix: str | None = None) -> str:
        prefix = prefix or "".join(self.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
        return f"{prefix}-{self.rng.randint(10, 99)}{self.rng.randint(100, 999)}"

    def date(self, lo: int = 2025, hi: int = 2027) -> dt.date:
        start = dt.date(lo, 1, 1); span = (dt.date(hi, 12, 31) - start).days
        return start + dt.timedelta(days=self.rng.randrange(span))

    def weekday_date(self) -> dt.date:
        d = self.date()
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        return d

    def fd(self, d: dt.date) -> str:
        return fd(d, self.dstyle)

    def m(self, x) -> str:
        return money(x, self.sym)

    def price(self, lo, hi) -> Decimal:
        cents = self.rng.choice([0, 0, 50, 95, 25, 75, 99, 40])
        return q2(Decimal(self.rng.randint(int(lo * 100), int(hi * 100)) // 100) + Decimal(cents) / 100) if hi >= 2 else q2(Decimal(self.rng.randint(int(lo * 100), int(hi * 100))) / 100)

    def unknown(self) -> bool:
        return self.rng.random() < UNKNOWN_P


# ------------------------------------------------------------------------------------------------- question helpers
def make_choice(c: Ctx, correct: tuple[str, str], distractors: list[tuple[str, str]], more: Callable[[int], tuple[str, str]] | None = None,
                k: int = 5, desc: Callable[[str], str] | None = None) -> dict:
    """Options from (text, canonical value) pairs: unique texts, unique option keys and unique canonical values; shuffled."""
    seen_t, seen_k, seen_v, chosen = set(), set(), set(), []
    def add(t, v):
        key = option_key(t)
        if t.lower() in seen_t or key in seen_k or v in seen_v:
            return
        seen_t.add(t.lower()); seen_k.add(key); seen_v.add(v); chosen.append((t, v))
    add(*correct)
    for d in distractors:
        if len(chosen) >= k: break
        add(*d)
    i = 1
    while len(chosen) < 3 and more is not None and i < 50:
        add(*more(i)); i += 1
    if len(chosen) < 3:
        raise ValueError("could not build three distinct options")
    first, rest = chosen[0], chosen[1:]
    c.rng.shuffle(rest); pos = c.rng.randrange(len(chosen)); opts = rest[:pos] + [first] + rest[pos:]
    desc = desc or (lambda t: t)
    return {"type": "choice", "options": [{"text": t, "description": desc(t)} for t, _ in opts], "intended": correct[0],
            "option_values": {t: v for t, v in opts}}


def Q(c: Ctx, kind: str, qtype: str, variant: int, title: str, lines: list[str], question: str, justification: str, params: dict,
      table: list[dict] | None = None, unknown: bool = False, **shape) -> dict:
    q = {"kind": kind, "template": f"{c.family}.{kind}.{qtype}{variant}", "title": title, "lines": lines, "table": table,
         "question": question, "type": qtype, "justification": justification[:800], "params": params}
    q.update(shape)
    if unknown:
        q["intended"] = None; q["unknown_reason"] = "insufficient_evidence"
    else:
        q["unknown_reason"] = None
    return q


def pick(c: Ctx, wordings: dict[str, list[str]], weights: dict[str, float] | None = None) -> tuple[str, int, str]:
    types = list(wordings)
    t = c.rng.choices(types, weights=[(weights or {}).get(x, 1.0) for x in types], k=1)[0]
    i = c.rng.randrange(len(wordings[t]))
    return t, i, wordings[t][i]


# ================================================================================================= temporal_arithmetic
TW: dict[str, dict[str, list[str]]] = {}


def _w(kind, **wordings):
    TW[kind] = wordings
    return wordings


W_BD = _w("business_deadline",
          choice=["By which date must the written response on {ref} be sent?", "On what date does the response window for {ref} close under rule {rule}?",
                  "What is the last day on which {ref} can be answered within the rule?", "Which date is the deadline for the reply on {ref}?"],
          noul=["The reply on {ref} went out on {x}. Was it within the {n}-business-day window?", "Was the response on {ref}, sent on {x}, on time under rule {rule}?"])


def t_business_deadline(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_BD)
    start = c.date(); n = c.rng.randint(3, 15); rule = c.code("R").split("-")[1][:3]
    naive = add_bdays(start, n)
    cands = [start + dt.timedelta(i) for i in range(1, (naive - start).days + 1) if (start + dt.timedelta(i)).weekday() < 5]
    closed = sorted(c.rng.sample(cands, k=min(len(cands), c.rng.randint(1, 2))))
    due = add_bdays(start, n, set(closed))
    event = c.rng.choice(c.v["events"])
    unk = c.unknown()
    recv = (f"The {event} for {c.v['case']} {c.ref} is logged as received on {c.fd(start)}." if not unk else
            f"The {event} for {c.v['case']} {c.ref} has no receipt date yet; the mailroom stamp is pending and the field reads 'TBC'.")
    lines = [recv, f"Rule {rule}: {c.v['team']} must send a written response within {n} business days of receipt. Day 1 is the first business day after the day of receipt.",
             f"Business days are Monday to Friday, excluding site closure days. Closure days this period: {', '.join(c.fd(x) for x in closed)}."]
    if c.rng.random() < 0.5:
        lines.append(f"A handover note from {c.person()} says the reply is due {c.fd(start + dt.timedelta(n))}; that note counted calendar days.")
    params = {"start": start.isoformat(), "n": n, "closed": [x.isoformat() for x in closed], "answer": due.isoformat(), "withheld": "start" if unk else None}
    just = f"Receipt {c.fd(start)} + {n} business days, skipping weekends and closures {', '.join(c.fd(x) for x in closed)} = {c.fd(due)}."
    if qtype == "choice":
        d = [(c.fd(start + dt.timedelta(n)), (start + dt.timedelta(n)).isoformat()), (c.fd(naive), naive.isoformat()),
             (c.fd(add_bdays(due, 1, set(closed))), add_bdays(due, 1, set(closed)).isoformat()), (c.fd(due - dt.timedelta(1)), (due - dt.timedelta(1)).isoformat())]
        ch = make_choice(c, (c.fd(due), due.isoformat()), d, more=lambda i: (c.fd(due + dt.timedelta(i + 1)), (due + dt.timedelta(i + 1)).isoformat()))
        params["option_values"] = ch.pop("option_values")
        return Q(c, "business_deadline", "choice", var, "Response deadline", lines, wording.format(ref=c.ref, rule=rule), just, params, unknown=unk, **ch)
    x = c.rng.choice([due, add_bdays(due, 1, set(closed)), naive if naive != due else add_bdays(due, 1, set(closed)), due - dt.timedelta(days=1)])
    params.update(action=x.isoformat(), answer=x <= due)
    return Q(c, "business_deadline", "noul", var, "Response deadline", lines, wording.format(ref=c.ref, rule=rule, x=c.fd(x), n=n),
             just + f" Sent {c.fd(x)}: {'on time' if x <= due else 'late'}.", params, unknown=unk, intended=x <= due)


W_ROLL = _w("calendar_rollover",
            choice=["On what date does the {thing} for {ref} expire?", "What is the expiry date of the {thing} on {ref}?",
                    "Counting as the terms require, which date is the final day of the {thing} for {ref}?", "When does the {thing} attached to {ref} lapse?"])


def t_calendar_rollover(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_ROLL)
    start = c.date(); n = c.rng.choice([30, 45, 60, 90, 100, 120, 14, 21, 28, 75])
    thing = c.rng.choice(["hold", "quote", "price lock", "authorisation", "reservation hold", "credit note"])
    end = start + dt.timedelta(days=n)
    lines = [f"The {thing} on {c.v['case']} {c.ref} was issued on {c.fd(start)}.",
             f"It remains valid for {n} calendar days; the day of issue is day 0, so it expires at the end of day {n}.",
             f"Extensions require a new {thing}; none has been issued for {c.ref}."]
    if c.rng.random() < 0.5:
        lines.append(f"Previous {thing}s at {c.org} ran for {c.rng.choice([n + 15, max(7, n - 15)])} days; that term no longer applies.")
    mon = add_months(start, round(n / 30))
    d = [((start + dt.timedelta(n - 1)), "n-1"), ((start + dt.timedelta(n + 1)), "n+1"), (mon, "months")]
    ch = make_choice(c, (c.fd(end), end.isoformat()), [(c.fd(x), x.isoformat()) for x, _ in d],
                     more=lambda i: (c.fd(end + dt.timedelta(i + 2)), (end + dt.timedelta(i + 2)).isoformat()))
    params = {"start": start.isoformat(), "n": n, "answer": end.isoformat(), "option_values": ch.pop("option_values")}
    return Q(c, "calendar_rollover", "choice", var, f"{thing.capitalize()} validity", lines, wording.format(thing=thing, ref=c.ref),
             f"{c.fd(start)} + {n} days = {c.fd(end)}.", params, **ch)


W_TZO = _w("tz_order",
           noul=["Did the {a} happen before the {b}?", "Was the {a} logged earlier in real time than the {b}?", "Taking the stated UTC offsets into account, did the {a} precede the {b}?"],
           choice=["How many minutes passed between the {a} and the {b}?", "What was the real elapsed time between the {a} and the {b}?"])


def t_tz_order(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_TZO)
    while True:
        (s1, o1), (s2, o2) = c.rng.sample(SITES, 2)
        if abs(o1 - o2) >= 120: break
    base = dt.datetime.combine(c.date(), dt.time(c.rng.randint(0, 23), c.rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])))
    flip = c.rng.random() < 0.7
    hi, lo = ((s1, o1), (s2, o2)) if o1 > o2 else ((s2, o2), (s1, o1))
    diff = hi[1] - lo[1]
    gap = c.rng.randint(15, diff - 10) if flip else c.rng.randint(15, 300)
    first, second = (hi, lo) if flip else (lo, hi)   # true order: `first` happens `gap` minutes before `second`
    names = c.rng.sample(["approval", "release", "cancellation", "escalation", "handover", "sign-off", "dispatch", "rollback"], 2)
    ev = [(f"{names[0]} at the {first[0]} site", first, base), (f"{names[1]} at the {second[0]} site", second, base + dt.timedelta(minutes=gap))]
    c.rng.shuffle(ev)
    lines = []
    for label, (site, off), utc in ev:
        local = utc + dt.timedelta(minutes=off)
        lines.append(f"{label[0].upper() + label[1:]}: recorded {fdt(local, c.dstyle)} local time ({site}, {fmt_off(off)}).")
    lines.append("Each site's log records local wall-clock time; the offsets above are the ones in force on those dates.")
    a, b = ev[0][0], ev[1][0]
    true_a, true_b = ev[0][2], ev[1][2]
    loc = [ev[i][2] + dt.timedelta(minutes=ev[i][1][1]) for i in range(2)]
    params = {"events": [{"label": l, "site": s, "offset": o, "local": (u + dt.timedelta(minutes=o)).isoformat()} for l, (s, o), u in ev]}
    just = f"In UTC the {a} is {true_a:%Y-%m-%d %H:%M} and the {b} is {true_b:%Y-%m-%d %H:%M}."
    if qtype == "noul":
        params["answer"] = true_a < true_b
        return Q(c, "tz_order", "noul", var, "Cross-site log", lines, wording.format(a=a, b=b), just, params, intended=true_a < true_b)
    real = int(abs((true_b - true_a).total_seconds()) // 60); naive = int(abs((loc[1] - loc[0]).total_seconds()) // 60)
    f = lambda m: f"{m // 60} h {m % 60:02d} min"  # noqa: E731
    ch = make_choice(c, (f(real), str(real)), [(f(naive), str(naive)), (f(real + 60), str(real + 60)), (f(abs(real - 60)), str(abs(real - 60))), (f(real + diff), str(real + diff))],
                     more=lambda i: (f(real + 30 * i), str(real + 30 * i)))
    params.update(answer=real, option_values=ch.pop("option_values"))
    return Q(c, "tz_order", "choice", var, "Cross-site log", lines, wording.format(a=a, b=b), just + f" Elapsed {f(real)}.", params, **ch)


W_TZC = _w("tz_cutoff",
           noul=["Was the submission for {ref} received before the cut-off?", "Did {ref} make the {site} cut-off?", "Is the filing for {ref} within the deadline?"],
           choice=["What is the cut-off for {ref} expressed in UTC?", "At what UTC time does the {site} cut-off fall?"])


def t_tz_cutoff(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_TZC)
    site, off = c.rng.choice([s for s in SITES if s[1] != 0])
    day = c.date(); hh = c.rng.choice([12, 16, 17, 18, 23]); mm = 59 if hh == 23 else 0
    cutoff_local = dt.datetime.combine(day, dt.time(hh, mm)); cutoff_utc = cutoff_local - dt.timedelta(minutes=off)
    delta = c.rng.choice([-1, 1]) * c.rng.randint(10, 200)
    sub_utc = cutoff_utc + dt.timedelta(minutes=delta)
    unk = qtype == "noul" and c.unknown()
    offtxt = f"{site} is on {fmt_off(off)}" if not unk else f"the {site} office's UTC offset for that date is still to be confirmed by facilities"
    lines = [f"Cut-off: submissions for {c.v['case']} {c.ref} must reach the {site} office by {cutoff_local:%H:%M} local time on {c.fd(day)} ({offtxt}).",
             f"Portal receipt log: submission received {sub_utc:%Y-%m-%d %H:%M} UTC.",
             f"The portal clock runs on UTC; the {site} office works in local time."]
    params = {"site": site, "offset": off, "cutoff_local": cutoff_local.isoformat(), "received_utc": sub_utc.isoformat()}
    just = f"{cutoff_local:%H:%M} {fmt_off(off)} = {cutoff_utc:%Y-%m-%d %H:%M} UTC; receipt {sub_utc:%Y-%m-%d %H:%M} UTC."
    if qtype == "noul":
        params.update(answer=sub_utc <= cutoff_utc, withheld="offset" if unk else None)
        return Q(c, "tz_cutoff", "noul", var, "Submission cut-off", lines, wording.format(ref=c.ref, site=site), just, params, unknown=unk, intended=sub_utc <= cutoff_utc)
    f = lambda t: f"{t:%Y-%m-%d %H:%M} UTC"  # noqa: E731
    wrong = [cutoff_local, cutoff_local + dt.timedelta(minutes=off), cutoff_utc + dt.timedelta(hours=1), cutoff_utc - dt.timedelta(hours=1)]
    ch = make_choice(c, (f(cutoff_utc), cutoff_utc.isoformat()), [(f(t), t.isoformat()) for t in wrong])
    params.update(answer=cutoff_utc.isoformat(), option_values=ch.pop("option_values"))
    return Q(c, "tz_cutoff", "choice", var, "Submission cut-off", lines, wording.format(ref=c.ref, site=site), just, params, **ch)


W_DUR = _w("duration_sum",
           choice=["What is the total time logged against {ref}?", "How much time in total do the entries for {ref} add up to?"],
           noul=["Does the time logged against {ref} exceed the {cap}-hour cap?", "Is the total logged on {ref} above the approved {cap} hours?"])


def t_duration_sum(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_DUR)
    entries, total, naive = [], 0, 0
    day0 = c.date()
    for j in range(c.rng.randint(4, 6)):
        form = c.rng.randrange(4)
        if form == 0:
            h, m = c.rng.randint(0, 3), c.rng.choice([5, 10, 20, 25, 35, 40, 50]); txt, mins, nv = f"{h} h {m} min", 60 * h + m, 60 * h + m
        elif form == 1:
            m = c.rng.randint(35, 170); txt, mins, nv = f"{m} minutes", m, m
        elif form == 2:
            q = c.rng.choice([0.25, 0.5, 0.75, 1.25, 1.5, 2.75]); txt, mins = f"{q} h", int(q * 60)
            nv = int(round((q % 1) * 100)) + 60 * int(q)  # misread 0.75 h as 75 minutes
        else:
            h, m = c.rng.randint(1, 3), c.rng.choice([10, 15, 20, 45]); txt, mins = f"{h}:{m:02d} (h:mm)", 60 * h + m
            nv = int(round((h + m / 100) * 60))  # read 2:10 as 2.10 hours
        entries.append({"date": c.fd(day0 + dt.timedelta(days=j + c.rng.randint(0, 1))), "by": c.person(), "logged": txt}); total += mins; naive += nv
    f = lambda m: f"{m // 60} h {m % 60:02d} min"  # noqa: E731
    lines = [f"Time entries booked to {c.v['case']} {c.ref} (formats differ because entries come from three tools).",
             "Decimal hours are fractions of an hour; h:mm entries are hours and minutes."]
    params = {"entries": [e["logged"] for e in entries], "answer": total}
    just = f"Converted to minutes the entries sum to {total} min = {f(total)}."
    if qtype == "choice":
        ch = make_choice(c, (f(total), str(total)), [(f(naive), str(naive)), (f(total + 15), str(total + 15)), (f(total - 15), str(total - 15)), (f(total + 60), str(total + 60))])
        params["option_values"] = ch.pop("option_values")
        return Q(c, "duration_sum", "choice", var, "Time log", lines, wording.format(ref=c.ref), just, params, table=entries, **ch)
    cap_min = (total + c.rng.choice([-40, -25, -10, 10, 25, 40])) // 15 * 15
    if cap_min == total:
        cap_min += 15
    cap_txt = f"{cap_min / 60:g}"
    cap_val = f"{cap_min // 60} h {cap_min % 60:02d} min"
    params.update(cap_minutes=cap_min, answer=total > cap_min)
    return Q(c, "duration_sum", "noul", var, "Time log", lines + [f"Approved effort for {c.ref}: {cap_txt} hours."], wording.format(ref=c.ref, cap=cap_txt),
             just + f" Cap {cap_val} = {cap_min} min.", params, table=entries, intended=total > cap_min)


W_SHIFT = _w("overnight_shift",
             choice=["How much paid working time did {p} record on the shift?", "What is {p}'s paid time for the shift once the break is removed?"],
             noul=["Did {p}'s paid time on the shift exceed {lim} hours?", "Is {p} owed the long-shift allowance for this shift?"])


def t_overnight_shift(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_SHIFT)
    p = c.person(); day = c.date()
    s = dt.datetime.combine(day, dt.time(c.rng.randint(18, 23), c.rng.choice([0, 10, 15, 30, 40, 45])))
    e = s + dt.timedelta(minutes=c.rng.randint(6 * 60, 12 * 60) // 5 * 5)
    brk = c.rng.choice([20, 30, 45, 60])
    paid = int((e - s).total_seconds() // 60) - brk
    lim = c.rng.choice([8, 9, 10])
    lines = [f"Shift record for {p}, {c.v['team']}: clocked in {fdt(s, c.dstyle)}, clocked out {fdt(e, c.dstyle)}.",
             f"Unpaid break taken: {brk} minutes. Paid time = time between clock-in and clock-out minus the unpaid break.",
             f"A long-shift allowance is paid when paid time is more than {lim} hours."]
    f = lambda m: f"{m // 60} h {m % 60:02d} min"  # noqa: E731
    params = {"start": s.isoformat(), "end": e.isoformat(), "break": brk, "limit_h": lim}
    just = f"{s:%H:%M} to {e:%H:%M} next day is {paid + brk} min; minus {brk} min break = {f(paid)}."
    if qtype == "choice":
        naive_same_day = abs(int((dt.datetime.combine(day, e.time()) - s).total_seconds() // 60)) - brk
        ch = make_choice(c, (f(paid), str(paid)), [(f(paid + brk), str(paid + brk)), (f(naive_same_day), str(naive_same_day)), (f(paid - 60), str(paid - 60)), (f(paid + 30), str(paid + 30))])
        params.update(answer=paid, option_values=ch.pop("option_values"))
        return Q(c, "overnight_shift", "choice", var, "Shift record", lines, wording.format(p=p), just, params, **ch)
    params["answer"] = paid > lim * 60
    return Q(c, "overnight_shift", "noul", var, "Shift record", lines, wording.format(p=p, lim=lim), just + f" Limit {lim} h.", params, intended=paid > lim * 60)


W_NOT = _w("notice_by_post",
           choice=["What is the latest date a posted non-renewal notice for {ref} can be sent?", "By which posting date must the notice on {ref} go out to stop the renewal?"],
           noul=["A non-renewal notice for {ref} was posted on {x}. Does it stop the renewal?", "Is a notice on {ref} posted on {x} effective in time?"])


def t_notice_by_post(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_NOT)
    renew = c.date(); n = c.rng.choice([30, 45, 60, 90]); k = c.rng.choice([2, 3])
    agr = c.code(c.rng.choice(["AGR", "SVC", "MSA"])); what = c.rng.choice(["service agreement", "supply contract", "maintenance contract", "subscription agreement"])
    last_recv = renew - dt.timedelta(days=n)
    s = last_recv
    while add_bdays(s, k) > last_recv:
        s -= dt.timedelta(days=1)
    lines = [f"The {what} {agr} between {c.org} and its {c.v['team']} provider renews automatically on {c.fd(renew)}.",
             f"Either party may stop the renewal by written notice received at least {n} days before the renewal date.",
             f"A notice sent by post is deemed received {k} business days after posting (Monday to Friday; public holidays are ignored for this purpose)."]
    params = {"renewal": renew.isoformat(), "n": n, "k": k}
    just = f"Notice must be received by {c.fd(last_recv)} ({c.fd(renew)} - {n} days); posting on {c.fd(s)} is deemed received {c.fd(add_bdays(s, k))}."
    if qtype == "choice":
        wrong = [last_recv, last_recv - dt.timedelta(days=k), add_bdays(s, 1), s - dt.timedelta(days=1)]
        ch = make_choice(c, (c.fd(s), s.isoformat()), [(c.fd(x), x.isoformat()) for x in wrong], more=lambda i: (c.fd(s - dt.timedelta(days=i + 1)), (s - dt.timedelta(days=i + 1)).isoformat()))
        params.update(answer=s.isoformat(), option_values=ch.pop("option_values"))
        return Q(c, "notice_by_post", "choice", var, "Renewal and notice", lines, wording.format(ref=agr), just, params, **ch)
    x = c.rng.choice([s, add_bdays(s, 1), last_recv, s - dt.timedelta(days=2)])
    ok = add_bdays(x, k) <= last_recv
    params.update(posted=x.isoformat(), answer=ok)
    return Q(c, "notice_by_post", "noul", var, "Renewal and notice", lines, wording.format(ref=agr, x=c.fd(x)),
             just + f" Posted {c.fd(x)} is deemed received {c.fd(add_bdays(x, k))}.", params, intended=ok)


def sla_due(t: dt.datetime, mins: int, oh: int, ch: int) -> dt.datetime:
    cur = t
    while True:
        if cur.weekday() >= 5:
            cur = dt.datetime.combine(cur.date() + dt.timedelta(days=7 - cur.weekday()), dt.time(oh)); continue
        start = cur.replace(hour=oh, minute=0); end = cur.replace(hour=ch, minute=0)
        if cur < start:
            cur = start
        if cur >= end:
            cur = dt.datetime.combine(cur.date() + dt.timedelta(days=1), dt.time(oh)); continue
        avail = int((end - cur).total_seconds() // 60)
        if mins <= avail:
            return cur + dt.timedelta(minutes=mins)
        mins -= avail; cur = dt.datetime.combine(cur.date() + dt.timedelta(days=1), dt.time(oh))


W_SLA = _w("sla_business_hours",
           choice=["When is the first response on {ref} due under the SLA?", "At what date and time does the SLA clock for {ref} run out?",
                   "What is the SLA due time for {ref}?"],
           noul=["{team} first replied on {ref} at {x}. Was the SLA met?"])


def t_sla_business_hours(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_SLA)
    oh, chh = c.rng.choice([(9, 17), (8, 18), (8, 16), (7, 15)])
    opened = dt.datetime.combine(c.date(), dt.time(c.rng.randint(0, 23), c.rng.choice([0, 10, 20, 30, 40, 50])))
    h = c.rng.choice([2, 4, 6, 8, 12, 16])
    due = sla_due(opened, h * 60, oh, chh)
    pr = c.rng.choice(["P2", "P3", "high", "standard"])
    lines = [f"{c.v['case'].capitalize()} {c.ref} opened {WEEKDAYS[opened.weekday()]} {fdt(opened, c.dstyle)}, priority {pr}.",
             f"SLA for priority {pr}: first response within {h} business hours.",
             f"Business hours are {oh:02d}:00 to {chh:02d}:00, Monday to Friday. The SLA clock runs only inside business hours and starts at the next opening if the {c.v['case']} arrives outside them."]
    params = {"opened": opened.isoformat(), "hours": h, "open": oh, "close": chh}
    just = f"Opened {opened:%a %H:%M}; {h} business hours inside {oh:02d}:00-{chh:02d}:00 Mon-Fri run out at {due:%a %Y-%m-%d %H:%M}."
    f = lambda t: f"{WEEKDAYS[t.weekday()][:3]} {fdt(t, c.dstyle)}"  # noqa: E731
    if qtype == "choice":
        wall = opened + dt.timedelta(hours=h)
        noweekend = opened
        # distractor: treat every day as a business day
        cur, m = opened, h * 60
        while True:
            st, en = cur.replace(hour=oh, minute=0), cur.replace(hour=chh, minute=0)
            if cur < st: cur = st
            if cur >= en: cur = dt.datetime.combine(cur.date() + dt.timedelta(days=1), dt.time(oh)); continue
            av = int((en - cur).total_seconds() // 60)
            if m <= av: noweekend = cur + dt.timedelta(minutes=m); break
            m -= av; cur = dt.datetime.combine(cur.date() + dt.timedelta(days=1), dt.time(oh))
        ch = make_choice(c, (f(due), due.isoformat()), [(f(wall), wall.isoformat()), (f(noweekend), noweekend.isoformat()), (f(due + dt.timedelta(hours=1)), (due + dt.timedelta(hours=1)).isoformat()),
                                                        (f(sla_due(opened, h * 60 - 60, oh, chh)), sla_due(opened, h * 60 - 60, oh, chh).isoformat())])
        params.update(answer=due.isoformat(), option_values=ch.pop("option_values"))
        return Q(c, "sla_business_hours", "choice", var, "Service level", lines, wording.format(ref=c.ref), just, params, **ch)
    x = due + dt.timedelta(minutes=c.rng.choice([-50, -20, 25, 70]))
    params.update(replied=x.isoformat(), answer=x <= due)
    return Q(c, "sla_business_hours", "noul", var, "Service level", lines, wording.format(ref=c.ref, team=c.v["team"].capitalize(), x=f(x)), just, params, intended=x <= due)


W_ELIG = _w("tenure_months",
            choice=["From which date is {p} eligible for the {perk}?", "On what date does {p} first qualify for the {perk}?"],
            noul=["Is {p} eligible for the {perk} on {x}?", "On {x}, has {p} completed the service needed for the {perk}?"])


def t_tenure_months(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_ELIG)
    p = c.person(); s = c.date(); s = s.replace(day=min(s.day, 28)); n = c.rng.choice([3, 6, 9, 12, 18, 24])
    perk = c.rng.choice(["priority tier", "loyalty rate", "extended warranty", "volume discount", "flexible-hours scheme", "reduced deposit"])
    elig = add_months(s, n)
    lines = [f"{p} joined the {c.org} programme on {c.fd(s)} (record {c.code()}).",
             f"The {perk} is available once {n} full calendar months have been completed; a month is completed on the same day-of-month as the joining date.",
             f"Periods are counted from the joining date; the {perk} applies from the day the last required month is completed."]
    params = {"start": s.isoformat(), "months": n}
    just = f"{c.fd(s)} + {n} months = {c.fd(elig)}."
    if qtype == "choice":
        wrong = [s + dt.timedelta(days=30 * n), elig - dt.timedelta(days=1), elig + dt.timedelta(days=1), add_months(s, n + 1)]
        ch = make_choice(c, (c.fd(elig), elig.isoformat()), [(c.fd(x), x.isoformat()) for x in wrong])
        params.update(answer=elig.isoformat(), option_values=ch.pop("option_values"))
        return Q(c, "tenure_months", "choice", var, "Eligibility", lines, wording.format(p=p, perk=perk), just, params, **ch)
    x = c.rng.choice([elig, elig - dt.timedelta(days=1), s + dt.timedelta(days=30 * n), elig + dt.timedelta(days=3)])
    params.update(on=x.isoformat(), answer=x >= elig)
    return Q(c, "tenure_months", "noul", var, "Eligibility", lines, wording.format(p=p, perk=perk, x=c.fd(x)), just, params, intended=x >= elig)


W_WD = _w("weekday_offset",
          choice=["On which weekday does the {b} fall?", "What day of the week is the {b} scheduled for?", "Which weekday will the {b} take place on?"])


def t_weekday_offset(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_WD)
    a = c.date(); n = c.rng.randint(9, 200); b = a + dt.timedelta(days=n)
    an, bn = c.rng.sample(["kick-off meeting", "site survey", "audit visit", "go-live", "handover", "review board", "stock count", "renewal call"], 2)
    lines = [f"The {an} for {c.ref} is on {WEEKDAYS[a.weekday()]}, {fd(a, 0)}.",
             f"The {bn} is scheduled exactly {n} days after the {an}.",
             f"Scheduling note from {c.person()}: the {bn} date was fixed by counting calendar days, not business days."]
    wrong = [WEEKDAYS[(b.weekday() + k) % 7] for k in (1, -1, 2, 3)]
    ch = make_choice(c, (WEEKDAYS[b.weekday()], WEEKDAYS[b.weekday()]), [(w, w) for w in wrong])
    params = {"anchor": a.isoformat(), "n": n, "answer": WEEKDAYS[b.weekday()], "option_values": ch.pop("option_values")}
    return Q(c, "weekday_offset", "choice", var, "Schedule", lines, wording.format(b=bn), f"{fd(a, 0)} + {n} days = {fd(b, 0)}, a {WEEKDAYS[b.weekday()]}.", params, **ch)


W_REC = _w("recurring_schedule",
           choice=["On what date does occurrence number {k} of the {what} take place?", "When is the {k_ord} {what} held?", "What is the date of {what} number {k}?"])


def t_recurring_schedule(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_REC)
    d0 = c.weekday_date(); N = c.rng.choice([7, 10, 14, 21, 28, 9]); k = c.rng.randint(3, 8)
    raw = d0 + dt.timedelta(days=(k - 1) * N)
    closed = []
    if raw.weekday() < 5 and c.rng.random() < 0.7:
        closed.append(raw)
    others = [d0 + dt.timedelta(days=j * N) for j in range(1, k - 1)]
    others = [x for x in others if x.weekday() < 5]
    if others and c.rng.random() < 0.6:
        closed.append(c.rng.choice(others))
    closed = sorted(set(closed))
    ans = next_bday(raw, set(closed))
    what = c.rng.choice(["compliance check", "stock rotation", "backup test", "supplier review", "safety walk", "cash count"])
    lines = [f"The {what} for {c.ref} started on {c.fd(d0)} (occurrence 1) and repeats every {N} days.",
             "If an occurrence falls on a weekend or a closure day it moves to the next business day; later occurrences keep the original cycle.",
             f"Closure days: {', '.join(c.fd(x) for x in closed) if closed else 'none this year'}."]
    ords = {3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
    wrong = [next_bday(d0 + dt.timedelta(days=k * N), set(closed)), raw, next_bday(raw + dt.timedelta(days=1), set(closed)), next_bday(d0 + dt.timedelta(days=(k - 2) * N), set(closed))]
    ch = make_choice(c, (c.fd(ans), ans.isoformat()), [(c.fd(x), x.isoformat()) for x in wrong], more=lambda i: (c.fd(ans + dt.timedelta(days=i + 1)), (ans + dt.timedelta(days=i + 1)).isoformat()))
    params = {"first": d0.isoformat(), "every": N, "k": k, "closed": [x.isoformat() for x in closed], "answer": ans.isoformat(), "option_values": ch.pop("option_values")}
    return Q(c, "recurring_schedule", "choice", var, "Recurring schedule", lines, wording.format(k=k, k_ord=ords[k], what=what),
             f"{c.fd(d0)} + {k - 1} x {N} days = {c.fd(raw)}, moved to the next business day if closed: {c.fd(ans)}.", params, **ch)


BANDS_DELAY = [(0, "on time: delivered on or before the promised date"), (1, "1 to 2 business days late"), (2, "3 to 5 business days late"),
               (3, "6 to 10 business days late"), (4, "more than 10 business days late")]
W_DELAY = _w("delay_band",
             score=["Which lateness band applies to {ref}?", "How late was {ref} on the lateness scale?", "Which credit band does the delivery of {ref} fall into?"])


def bdays_between(a: dt.date, b: dt.date, closed=()) -> int:
    """Business days in (a, b]."""
    return sum(1 for i in range(1, (b - a).days + 1) if (a + dt.timedelta(i)).weekday() < 5 and (a + dt.timedelta(i)) not in closed)


def t_delay_band(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_DELAY)
    prom = c.weekday_date(); late = c.rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20])
    deliv = prom + dt.timedelta(days=late) if late else prom - dt.timedelta(days=c.rng.randint(0, 2))
    closed = []
    if late > 3 and c.rng.random() < 0.6:
        closed = [x for x in (prom + dt.timedelta(days=i) for i in range(1, late)) if x.weekday() < 5][:1]
    bd = bdays_between(prom, deliv, set(closed)) if deliv > prom else 0
    level = 0 if bd == 0 else 1 if bd <= 2 else 2 if bd <= 5 else 3 if bd <= 10 else 4
    lines = [f"{c.v['case'].capitalize()} {c.ref}: promised for {c.fd(prom)}, completed {c.fd(deliv)}.",
             "Lateness is counted in business days after the promised date (Monday to Friday, excluding closure days).",
             f"Closure days: {', '.join(c.fd(x) for x in closed) if closed else 'none in the period'}."]
    lines += [f"Band {v}: {d}." for v, d in BANDS_DELAY]
    params = {"promised": prom.isoformat(), "delivered": deliv.isoformat(), "closed": [x.isoformat() for x in closed], "business_days_late": bd, "answer": level}
    return Q(c, "delay_band", "score", var, "Lateness bands", lines, wording.format(ref=c.ref),
             f"{bd} business days after {c.fd(prom)} (calendar gap {max(0, (deliv - prom).days)}) -> band {level}.", params,
             levels=[{"value": v, "description": d} for v, d in BANDS_DELAY], intended=level)


W_GRACE = _w("grace_period",
             choice=["What is the last date on which payment of {ref} arrives without a late fee?", "Until which date can {ref} be paid fee-free?"],
             noul=["Payment for {ref} arrived on {x}. Is the late fee charged?", "Does {ref}, paid on {x}, attract the late fee?"])


def t_grace_period(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_GRACE)
    inv = c.date(); net = c.rng.choice([14, 21, 30, 45, 60]); g = c.rng.choice([3, 5, 7, 10])
    due = inv + dt.timedelta(days=net); last = due + dt.timedelta(days=g)
    unk = c.unknown()
    lines = [f"Invoice {c.ref} issued {c.fd(inv)}; payment terms net {net} days from the invoice date." if not unk else
             f"Invoice {c.ref}: payment terms net {net} days from the invoice date; the invoice date is missing from the scanned copy (re-scan requested).",
             f"A grace period of {g} calendar days follows the due date; a late fee of {c.rng.choice([2, 3, 5])}% applies only to payments arriving after the grace period.",
             f"Payments are credited on the day they arrive."]
    params = {"invoice": inv.isoformat(), "net": net, "grace": g, "withheld": "invoice" if unk else None}
    just = f"{c.fd(inv)} + {net} days = due {c.fd(due)}; + {g} days grace = {c.fd(last)}."
    if qtype == "choice":
        wrong = [due, last + dt.timedelta(days=1), inv + dt.timedelta(days=net + g - 1), add_bdays(due, g)]
        ch = make_choice(c, (c.fd(last), last.isoformat()), [(c.fd(x), x.isoformat()) for x in wrong], more=lambda i: (c.fd(last - dt.timedelta(days=i + 1)), (last - dt.timedelta(days=i + 1)).isoformat()))
        params.update(answer=last.isoformat(), option_values=ch.pop("option_values"))
        return Q(c, "grace_period", "choice", var, "Payment terms", lines, wording.format(ref=c.ref), just, params, unknown=unk, **ch)
    x = c.rng.choice([last, last + dt.timedelta(days=1), due + dt.timedelta(days=1), last + dt.timedelta(days=4)])
    params.update(paid=x.isoformat(), answer=x > last)
    return Q(c, "grace_period", "noul", var, "Payment terms", lines, wording.format(ref=c.ref, x=c.fd(x)), just, params, unknown=unk, intended=x > last)


TEMPORAL_KINDS = {"business_deadline": t_business_deadline, "calendar_rollover": t_calendar_rollover, "tz_order": t_tz_order, "tz_cutoff": t_tz_cutoff,
                  "duration_sum": t_duration_sum, "overnight_shift": t_overnight_shift, "notice_by_post": t_notice_by_post,
                  "sla_business_hours": t_sla_business_hours, "tenure_months": t_tenure_months, "weekday_offset": t_weekday_offset,
                  "recurring_schedule": t_recurring_schedule, "delay_band": t_delay_band, "grace_period": t_grace_period}
TEMPORAL_WORDINGS = dict(TW); TW.clear()


# ================================================================================================= probability_exact
def comb(n, k) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def fmt_p(f: Fraction, style: str) -> str:
    return fr(f) if style == "fraction" else pct(f)


STYLE_TXT = {"fraction": "as a fraction in lowest terms", "percent": "as a percentage rounded to one decimal place"}


def prob_choice(c: Ctx, correct: Fraction, wrong: list[Fraction], style: str) -> dict:
    cands = [(fmt_p(w, style), fr(w)) for w in wrong if 0 <= w <= 1 and w != correct]
    more = lambda i: (fmt_p(min(Fraction(1), correct * (1 + Fraction(i, 7))), style), fr(min(Fraction(1), correct * (1 + Fraction(i, 7)))))  # noqa: E731
    return make_choice(c, (fmt_p(correct, style), fr(correct)), cands, more=more)


W_P1 = _w("single_draw", choice=["If one {case} is picked at random from the pool, what is the probability that it is {what} ({style})?",
                                  "What is the chance that a randomly selected {case} from the pool is {what} ({style})?",
                                  "A {case} is drawn uniformly at random for the spot check. How likely is it to be {what} ({style})?",
                                  "For the random spot check, what is P({what}) for the selected {case} ({style})?"])


def _pool(c: Ctx, k: int, lo=2, hi=12):
    labels = c.rng.sample(["cleared", "flagged", "on hold", "escalated", "pending review", "rejected", "reworked", "approved with notes"], k)
    return {l: c.rng.randint(lo, hi) for l in labels}


def p_single_draw(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P1)
    pool = _pool(c, c.rng.randint(3, 4)); n = sum(pool.values()); style = c.rng.choice(["fraction", "percent"])
    labs = list(pool); target = c.rng.sample(labs, c.rng.choice([1, 1, 2]))
    a = sum(pool[t] for t in target); p = Fraction(a, n)
    what = " or ".join(target)
    lines = [f"Spot-check pool for {c.org}, {c.v['team']}: {n} {c.v['cases']} in total, broken down by status below.",
             "The auditor picks the spot-check item uniformly at random from the whole pool."]
    table = [{"status": l, "count": pool[l]} for l in labs]
    ch = prob_choice(c, p, [Fraction(a, n - a) if a < n else Fraction(1), 1 - p, Fraction(1, len(labs)) * len(target), Fraction(a, n - 1), Fraction(pool[target[0]], n)], style)
    params = {"pool": pool, "target": target, "style": style, "answer": fr(p), "option_values": ch.pop("option_values")}
    return Q(c, "single_draw", "choice", var, "Spot-check pool", lines, wording.format(case=c.v["case"], what=what, style=STYLE_TXT[style]),
             f"{a} of {n} {c.v['cases']} are {what}: {a}/{n} = {fr(p)}.", params, table=table, **ch)


W_P2 = _w("pair_without_replacement", choice=["Two different {cases} are picked at random. What is the probability that both are {what} ({style})?",
                                              "The reviewer samples two distinct {cases} at random. How likely is it that both are {what} ({style})?",
                                              "What is the chance that both {cases} in a random pair are {what} ({style})?",
                                              "A random pair of distinct {cases} is pulled from the queue. What is the probability that the pair is entirely {what} ({style})?"])


def p_pair(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P2)
    n = c.rng.randint(8, 30); a = c.rng.randint(2, max(2, n // 2)); what = c.rng.choice(["flagged", "missing a signature", "over the threshold", "from the night shift"])
    style = c.rng.choice(["fraction", "percent"]); p = Fraction(a * (a - 1), n * (n - 1))
    lines = [f"{c.org} holds {n} {c.v['cases']} in the review queue; {a} of them are {what}.",
             "Two different items are drawn at random from the queue without replacement; every pair is equally likely."]
    ch = prob_choice(c, p, [Fraction(a, n) ** 2, Fraction(a, n) * Fraction(a - 1, n), Fraction(a, n), Fraction(a * (a - 1), n * n), Fraction(comb(a, 2), n)], style)
    params = {"n": n, "a": a, "style": style, "answer": fr(p), "option_values": ch.pop("option_values")}
    return Q(c, "pair_without_replacement", "choice", var, "Review queue", lines, wording.format(cases=c.v["cases"], what=what, style=STYLE_TXT[style]),
             f"({a}/{n}) x ({a - 1}/{n - 1}) = {fr(p)}.", params, **ch)


W_P3 = _w("at_least_one", choice=["What is the probability that the sample contains at least one {what} {case} ({style})?",
                                  "How likely is it that at least one sampled {case} is {what} ({style})?",
                                  "What is the chance the inspection sample includes one or more {what} {cases} ({style})?",
                                  "What is the probability that the inspector finds a {what} {case} in the sample ({style})?"])


def _atleast(n, d, k) -> Fraction:
    return 1 - Fraction(comb(n - d, k), comb(n, k))


def p_at_least_one(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P3)
    n = c.rng.randint(10, 25); d = c.rng.randint(1, 4); k = c.rng.randint(2, 4); style = c.rng.choice(["fraction", "percent"])
    what = c.rng.choice(["defective", "mislabelled", "non-compliant", "damaged"])
    p = _atleast(n, d, k)
    unk = c.unknown()
    lines = [f"Lot {c.code()} contains {n} {c.v['cases']}; {d} of them are {what} (confirmed after the fact by full inspection)." if not unk else
             f"Lot {c.code()} contains {n} {c.v['cases']}; the count of {what} items is still being confirmed by full inspection and is not in this report.",
             f"The inspector draws {k} different items at random without replacement."]
    ch = prob_choice(c, p, [1 - Fraction(n - d, n) ** k, Fraction(comb(n - d, k), comb(n, k)), Fraction(k * d, n), Fraction(d, n)], style)
    params = {"n": n, "d": d, "k": k, "style": style, "answer": fr(p), "option_values": ch.pop("option_values"), "withheld": "d" if unk else None}
    return Q(c, "at_least_one", "choice", var, "Sampling plan", lines, wording.format(case=c.v["case"], cases=c.v["cases"], what=what, style=STYLE_TXT[style]),
             f"1 - C({n - d},{k})/C({n},{k}) = {fr(p)}.", params, unknown=unk, **ch)


W_P4 = _w("conditional_table", choice=["Among {cases} handled by {g}, what is the probability that one picked at random was {o} ({style})?",
                                       "Given that a {case} came from {g}, how likely is it to have been {o} ({style})?",
                                       "What share of {g}'s {cases} were {o}, as a probability ({style})?",
                                       "If a random {case} is known to be from {g}, what is P({o}) ({style})?"])


def p_conditional_table(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P4)
    groups = c.rng.sample(["North hub", "South hub", "Day team", "Night team", "Partner channel", "Direct channel", "Site A", "Site B"], 2)
    outs = c.rng.sample([("resolved in one contact", "needed follow-up"), ("on time", "late"), ("accepted", "disputed"), ("passed", "failed")], 1)[0]
    cells = [[c.rng.randint(3, 40) for _ in range(2)] for _ in range(2)]
    style = c.rng.choice(["fraction", "percent"])
    gi, oi = c.rng.randrange(2), c.rng.randrange(2)
    a = cells[gi][oi]; row = sum(cells[gi]); col = cells[0][oi] + cells[1][oi]; N = sum(map(sum, cells))
    p = Fraction(a, row)
    table = [{"group": groups[i], outs[0]: cells[i][0], outs[1]: cells[i][1]} for i in range(2)]
    lines = [f"Outcome counts for {c.v['cases']} closed last quarter at {c.org}.", f"Each {c.v['case']} appears once, under the group that handled it."]
    ch = prob_choice(c, p, [Fraction(a, col), Fraction(a, N), Fraction(row, N), Fraction(col, N)], style)
    params = {"cells": cells, "group": gi, "outcome": oi, "style": style, "answer": fr(p), "option_values": ch.pop("option_values")}
    return Q(c, "conditional_table", "choice", var, "Outcome table", lines, wording.format(cases=c.v["cases"], case=c.v["case"], g=groups[gi], o=outs[oi], style=STYLE_TXT[style]),
             f"{a} of the {row} from {groups[gi]} were {outs[oi]}: {fr(p)}.", params, table=table, **ch)


W_P5 = _w("bayes_counts", choice=["If the screen flags a {case}, what is the probability that it really is {bad} ({style})?",
                                  "What fraction of flagged {cases} are actually {bad} ({style})?",
                                  "Given a flag from the screen, how likely is the {case} to be {bad} ({style})?",
                                  "A {case} has just been flagged. What is the probability that the flag is correct, i.e. the {case} is {bad} ({style})?"])


def p_bayes(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P5)
    while True:
        N = c.rng.choice([500, 1000, 2000, 2500, 4000]); m = c.rng.choice([10, 20, 25, 40, 50, 60, 80, 100])
        s = c.rng.choice([80, 90, 95, 75, 60]); f = c.rng.choice([2, 4, 5, 10, 8])
        if (s * m) % 100 == 0 and (f * (N - m)) % 100 == 0 and m < N // 5:
            break
    tp, fp = s * m // 100, f * (N - m) // 100; p = Fraction(tp, tp + fp); style = c.rng.choice(["fraction", "percent"])
    bad = c.rng.choice(["fraudulent", "a duplicate", "misrouted", "non-compliant"])
    lines = [f"Last month {N:,} {c.v['cases']} passed through the automated screen at {c.org}; {m} of them were later confirmed {bad}.",
             f"The screen flagged {s}% of the {bad} {c.v['cases']} and also flagged {f}% of the others.",
             f"Assume the same rates apply to the next {c.v['case']}."]
    ch = prob_choice(c, p, [Fraction(s, 100), Fraction(100 - f, 100), Fraction(m, N), Fraction(tp, N), Fraction(tp, fp) if fp else Fraction(1)], style)
    params = {"N": N, "m": m, "sens": s, "fpr": f, "style": style, "answer": fr(p), "option_values": ch.pop("option_values")}
    return Q(c, "bayes_counts", "choice", var, "Screening results", lines, wording.format(case=c.v["case"], cases=c.v["cases"], bad=bad, style=STYLE_TXT[style]),
             f"True flags {s}% x {m} = {tp}; false flags {f}% x {N - m} = {fp}; P = {tp}/{tp + fp}" + (f" = {fr(p)}." if fr(p) != f"{tp}/{tp + fp}" else "."), params, **ch)


W_P6 = _w("independent_both", choice=["What is the probability that the {case} passes both checks (exact percentage)?",
                                      "How likely is a {case} to clear the {x} and the {y} (exact percentage)?",
                                      "What is the chance that both independent checks succeed for the next {case} (exact percentage)?"],
          noul=["Is the chance of passing both checks at least {t}%?"])


def p_independent_both(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P6)
    p, q = c.rng.choice(range(50, 100, 5)), c.rng.choice(range(50, 100, 5))
    x, y = c.rng.sample(["document check", "identity check", "credit check", "address check", "stock check", "fraud screen"], 2)
    P = Fraction(p * q, 10000)
    lines = [f"The {x} passes {p}% of {c.v['cases']}; the {y} passes {q}%.", f"The two checks are run by different teams and are independent of each other.",
             f"A {c.v['case']} proceeds only if it passes both."]
    params = {"p": p, "q": q}
    just = f"{p}% x {q}% = {pct_exact(P)}."
    if qtype == "noul":
        t = int(P * 100) + c.rng.choice([-3, -1, 1, 2, 4])
        params.update(threshold=t, answer=P * 100 >= t)
        return Q(c, "independent_both", "noul", var, "Checks", lines, wording.format(t=t), just, params, intended=bool(P * 100 >= t))
    wrong = [Fraction(min(p, q), 100), Fraction(p + q - 100, 100), Fraction(p + q, 200), 1 - (1 - Fraction(p, 100)) * (1 - Fraction(q, 100))]
    ch = make_choice(c, (pct_exact(P), fr(P)), [(pct_exact(w), fr(w)) for w in wrong if 0 <= w <= 1], more=lambda i: (pct_exact(P + Fraction(i, 100)), fr(P + Fraction(i, 100))))
    params.update(answer=fr(P), option_values=ch.pop("option_values"))
    return Q(c, "independent_both", "choice", var, "Checks", lines, wording.format(case=c.v["case"], x=x, y=y), just, params, **ch)


W_P7 = _w("independent_any", choice=["What is the probability that at least one of the two {things} fires during the window (exact percentage)?",
                                     "How likely is it that one or both {things} trigger (exact percentage)?",
                                     "What is P(at least one {thing} fires), as an exact percentage?"])


def p_independent_any(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P7)
    p, q = c.rng.choice(range(5, 60, 5)), c.rng.choice(range(5, 60, 5))
    thing = c.rng.choice(["alarm", "reminder", "fallback", "escalation rule"])
    P = 1 - (1 - Fraction(p, 100)) * (1 - Fraction(q, 100))
    lines = [f"Two independent {thing}s cover {c.v['case']} {c.ref}: the first fires in {p}% of windows, the second in {q}%.",
             f"They are triggered by unrelated signals, so treat them as independent."]
    wrong = [Fraction(p + q, 100), Fraction(max(p, q), 100), Fraction(p * q, 10000), Fraction(p + q, 100) - Fraction(p * q, 100)]
    ch = make_choice(c, (pct_exact(P), fr(P)), [(pct_exact(w), fr(w)) for w in wrong if 0 <= w <= 1], more=lambda i: (pct_exact(P - Fraction(i, 100)), fr(P - Fraction(i, 100))))
    params = {"p": p, "q": q, "answer": fr(P), "option_values": ch.pop("option_values")}
    return Q(c, "independent_any", "choice", var, "Coverage", lines, wording.format(things=thing + "s", thing=thing),
             f"1 - (1 - {p}%)(1 - {q}%) = {pct_exact(P)}.", params, **ch)


W_P8 = _w("committee_count", choice=["How many different review panels can be formed under these rules?", "How many valid panels of {k} are there?",
                                     "Counting only panels that satisfy both rules, how many choices of {k} people are possible?",
                                     "How many distinct line-ups of {k} panel members comply with the chair and conflict rules?"])


def p_committee(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P8)
    n = c.rng.randint(6, 12); k = c.rng.randint(3, min(5, n - 2)); lead, conflicted = c.person(), c.person()
    ans = comb(n - 2, k - 1)
    lines = [f"{n} people are eligible to sit on the review panel for {c.ref}, including {lead} and {conflicted}.",
             f"The panel has exactly {k} members and the order of members does not matter.",
             f"{lead} chairs every panel and must be included; {conflicted} has declared a conflict and must be excluded."]
    wrong = [comb(n, k), comb(n - 1, k - 1), comb(n - 1, k), comb(n - 2, k), math.perm(n - 2, k - 1)]
    ch = make_choice(c, (str(ans), str(ans)), [(str(w), str(w)) for w in wrong], more=lambda i: (str(ans + i), str(ans + i)))
    params = {"n": n, "k": k, "answer": ans, "option_values": ch.pop("option_values")}
    return Q(c, "committee_count", "choice", var, "Panel rules", lines, wording.format(k=k), f"Choose the other {k - 1} from the {n - 2} remaining: C({n - 2},{k - 1}) = {ans}.", params, **ch)


W_P9 = _w("ordered_slots", choice=["In how many different ways can the {k} slots be filled?", "How many distinct running orders are possible?",
                                   "How many ways are there to assign presenters to the {k} slots?",
                                   "Counting different presenters in different slots as different schedules, how many schedules exist?"])


def p_ordered(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P9)
    n = c.rng.randint(5, 9); k = c.rng.randint(2, 4); ans = math.perm(n, k)
    lines = [f"{n} team members volunteered to present at the {c.org} review.", f"There are {k} consecutive slots; each slot gets a different presenter and the order matters.",
             "No one may present twice."]
    wrong = [comb(n, k), n ** k, math.factorial(k), math.perm(n - 1, k), math.factorial(n)]
    ch = make_choice(c, (str(ans), str(ans)), [(str(w), str(w)) for w in wrong])
    params = {"n": n, "k": k, "answer": ans, "option_values": ch.pop("option_values")}
    return Q(c, "ordered_slots", "choice", var, "Running order", lines, wording.format(k=k), f"{n}!/({n}-{k})! = {ans}.", params, **ch)


W_P10 = _w("expected_cost", choice=["What is the expected cost per {case} under these rates?", "What is the probability-weighted average cost of one {case}?",
                                    "What expected cost should be budgeted for each {case}?"])


def p_expected(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P10)
    while True:
        ps = [c.rng.choice(range(5, 90, 5)) for _ in range(2)]
        if sum(ps) < 100: break
    ps.append(100 - sum(ps))
    costs = [Decimal(c.rng.randint(1, 60) * 10) for _ in range(3)]
    labels = c.rng.sample(["resolved remotely", "technician visit", "replacement", "partial refund", "full refund", "no action"], 3)
    ev = q2(sum(Decimal(p) * cc for p, cc in zip(ps, costs)) / 100)
    lines = [f"Cost model for {c.v['cases']} at {c.org}, based on last year's closed cases:"]
    table = [{"outcome": l, "share of cases": f"{p}%", "cost": c.m(cc)} for l, p, cc in zip(labels, ps, costs)]
    wrong = [q2(sum(costs) / 3), costs[ps.index(max(ps))], q2(sum(costs)), q2(sum(Decimal(p) * cc for p, cc in zip(ps[:2], costs[:2])) / 100)]
    ch = make_choice(c, (c.m(ev), str(ev)), [(c.m(w), str(q2(w))) for w in wrong], more=lambda i: (c.m(ev + i * 5), str(q2(ev + i * 5))))
    params = {"shares": ps, "costs": [str(x) for x in costs], "answer": str(ev), "option_values": ch.pop("option_values")}
    return Q(c, "expected_cost", "choice", var, "Cost model", lines + ["Shares add up to 100%."], wording.format(case=c.v["case"]),
             " + ".join(f"{p}% x {c.m(cc)}" for p, cc in zip(ps, costs)) + f" = {c.m(ev)}.", params, table=table, **ch)


W_P11 = _w("assignment_none", choice=["What is the probability that {p} receives none of the {k} {cases} ({style})?",
                                      "How likely is it that {p} is assigned no {cases} at all ({style})?",
                                      "What is the chance that {p}'s queue stays empty after the {k} assignments ({style})?"])


def p_assignment(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P11)
    n = c.rng.randint(3, 6); k = c.rng.randint(2, 4); p = c.person(); style = c.rng.choice(["fraction", "percent"])
    P = Fraction(n - 1, n) ** k
    lines = [f"{k} new {c.v['cases']} arrive and are each assigned to one of {n} handlers, including {p}.",
             "Each assignment is made independently and uniformly at random; a handler can receive several."]
    ch = prob_choice(c, P, [Fraction(n - 1, n), 1 - Fraction(1, n) ** k, Fraction(max(n - k, 0), n), Fraction(1, n ** k), Fraction(1, n)], style)
    params = {"n": n, "k": k, "style": style, "answer": fr(P), "option_values": ch.pop("option_values")}
    return Q(c, "assignment_none", "choice", var, "Assignment", lines, wording.format(p=p, k=k, cases=c.v["cases"], style=STYLE_TXT[style]),
             f"({n - 1}/{n})^{k} = {fr(P)}.", params, **ch)


W_P12 = _w("clean_sample_threshold", noul=["Is the probability that the sample contains no {what} {cases} above {t}%?",
                                           "Does the chance of a fully clean sample exceed {t}%?",
                                           "Is a clean sample (no {what} items) more likely than {t}%?"])


def p_clean_threshold(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P12)
    while True:
        n = c.rng.randint(10, 30); d = c.rng.randint(1, 5); k = c.rng.randint(2, 5)
        P = Fraction(comb(n - d, k), comb(n, k))
        t = int(P * 100) + c.rng.choice([-4, -2, 1, 3, 5])
        if 0 < t < 100 and P * 100 != t: break
    what = c.rng.choice(["defective", "late", "mis-keyed", "unsigned"])
    lines = [f"Of {n} {c.v['cases']} in the batch, {d} are {what}.", f"Quality control draws {k} different items at random without replacement."]
    params = {"n": n, "d": d, "k": k, "threshold": t, "answer": P * 100 > t}
    return Q(c, "clean_sample_threshold", "noul", var, "Sampling", lines, wording.format(what=what, cases=c.v["cases"], t=t),
             f"C({n - d},{k})/C({n},{k}) = {fr(P)} = {pct(P)} vs {t}%.", params, intended=bool(P * 100 > t))


W_P13 = _w("probability_band", score=["Which probability band does the chance of at least one {what} item in the sample fall into?",
                                      "Place the probability that the sample catches a {what} item on the band scale.",
                                      "What band does P(sample contains a {what} item) belong to?"])


def p_band(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P13)
    width = c.rng.choice([10, 20, 25])
    while True:
        n = c.rng.randint(8, 30); d = c.rng.randint(1, 5); k = c.rng.randint(2, 5)
        P = _atleast(n, d, k)
        if (P * 100) % width != 0 and 0 < P < 1: break
    nb = 100 // width
    levels = [{"value": i, "description": f"at least {i * width}% and below {(i + 1) * width}%" if i < nb - 1 else f"at least {i * width}% up to 100%"} for i in range(nb)]
    lvl = min(int(P * 100 // width), nb - 1)
    what = c.rng.choice(["defective", "non-compliant", "tampered", "expired"])
    lines = [f"{n} {c.v['cases']} are on hand and {d} are {what}.", f"An auditor selects {k} of them at random without replacement."]
    params = {"n": n, "d": d, "k": k, "width": width, "answer": lvl}
    return Q(c, "probability_band", "score", var, "Audit sample", lines, wording.format(what=what),
             f"1 - C({n - d},{k})/C({n},{k}) = {fr(P)} = {pct(P)} -> band {lvl}.", params, levels=levels, intended=lvl)


W_P14 = _w("binomial_exactly_one", choice=["What is the probability that exactly one of the {n} {cases} is {what} ({style})?",
                                           "How likely is it that precisely one of the next {n} {cases} turns out {what} ({style})?",
                                           "What is P(exactly one {what} among {n}) ({style})?"])


def p_binomial(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_P14)
    n = c.rng.choice([2, 3, 4]); p = c.rng.choice([Fraction(1, 5), Fraction(1, 4), Fraction(1, 10), Fraction(3, 10), Fraction(2, 5), Fraction(1, 2)])
    style = c.rng.choice(["fraction", "percent"]); P = n * p * (1 - p) ** (n - 1)
    what = c.rng.choice(["disputed", "returned", "escalated", "rejected"])
    ptxt = c.rng.choice([f"{int(p * 100)}%", f"{p.numerator} in {p.denominator}"])
    lines = [f"Historically {ptxt} of {c.v['cases']} at {c.org} are {what}, independently of one another.", f"The next {n} {c.v['cases']} are about to be processed."]
    ch = prob_choice(c, P, [p, n * p, p * (1 - p) ** (n - 1), 1 - (1 - p) ** n, (1 - p) ** n], style)
    params = {"n": n, "p": fr(p), "style": style, "answer": fr(P), "option_values": ch.pop("option_values")}
    return Q(c, "binomial_exactly_one", "choice", var, "Base rates", lines, wording.format(n=n, cases=c.v["cases"], what=what, style=STYLE_TXT[style]),
             f"{n} x {fr(p)} x ({fr(1 - p)})^{n - 1} = {fr(P)}.", params, **ch)


PROBABILITY_KINDS = {"single_draw": p_single_draw, "pair_without_replacement": p_pair, "at_least_one": p_at_least_one, "conditional_table": p_conditional_table,
                     "bayes_counts": p_bayes, "independent_both": p_independent_both, "independent_any": p_independent_any, "committee_count": p_committee,
                     "ordered_slots": p_ordered, "expected_cost": p_expected, "assignment_none": p_assignment, "clean_sample_threshold": p_clean_threshold,
                     "probability_band": p_band, "binomial_exactly_one": p_binomial}
PROBABILITY_WORDINGS = dict(TW); TW.clear()


# ================================================================================================= numeric_reconciliation
def _lines_items(c: Ctx, k: int):
    items = c.rng.sample(c.v["items"], k)
    out = []
    for name, lo, hi in items:
        qty = c.rng.randint(1, 12); price = c.price(lo, hi)
        out.append({"item": name, "qty": qty, "unit_price": price, "amount": q2(qty * price)})
    return out


def _table(c, rows):
    return [{"item": r["item"], "qty": r["qty"], "unit price": c.m(r["unit_price"]), "line total": c.m(r["amount"])} for r in rows]


def money_choice(c: Ctx, correct: Decimal, wrong: list[Decimal]) -> dict:
    return make_choice(c, (c.m(correct), str(q2(correct))), [(c.m(w), str(q2(w))) for w in wrong if w >= 0],
                       more=lambda i: (c.m(correct + Decimal(i) * Decimal("7.50")), str(q2(correct + Decimal(i) * Decimal("7.50")))))


W_N1 = _w("invoice_total", choice=["What is the amount due on invoice {ref}?", "What total should accounts payable approve for {ref}?",
                                   "Applying the invoice terms, what does {ref} come to?"],
          noul=["Is the amount due on {ref} above the {lim} single-approval limit?"])


def n_invoice_total(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N1)
    rows = _lines_items(c, c.rng.randint(3, 4)); sub = sum(r["amount"] for r in rows)
    d = c.rng.choice([5, 8, 10, 12, 15]); t = c.rng.choice([Decimal(5), Decimal("7.5"), Decimal(8), Decimal(10), Decimal(20), Decimal(19)])
    ship = q2(Decimal(c.rng.randint(5, 45)) + Decimal(c.rng.choice([0, 50, 95])) / 100)
    disc = q2(sub * d / 100); after = sub - disc; tax = q2(after * t / 100); total = after + tax + ship
    unk = c.unknown()
    lines = [f"Invoice {c.ref} from {c.org} to {c.person()}.",
             f"Order discount: {d}% of the subtotal." ,
             f"Tax at {t}% is charged on the subtotal after discount." if not unk else "Tax: the applicable rate is under review by the tax team and is not stated on this draft.",
             f"Delivery charge {c.m(ship)} is added after tax and is not taxed.", "Each step is rounded to the cent."]
    params = {"lines": [[r["qty"], str(r["unit_price"])] for r in rows], "discount_pct": d, "tax_pct": str(t), "shipping": str(ship), "answer": str(total),
              "withheld": "tax" if unk else None}
    just = f"Subtotal {c.m(sub)} - {d}% ({c.m(disc)}) = {c.m(after)}; tax {t}% = {c.m(tax)}; + delivery {c.m(ship)} = {c.m(total)}."
    if qtype == "choice":
        wrong = [sub - disc + q2(sub * t / 100) + ship, after + q2((after + ship) * t / 100) + ship, after + tax, sub + q2(sub * t / 100) + ship, total + disc]
        ch = money_choice(c, total, wrong); params["option_values"] = ch.pop("option_values")
        return Q(c, "invoice_total", "choice", var, "Invoice", lines, wording.format(ref=c.ref), just, params, table=_table(c, rows), unknown=unk, **ch)
    lim = q2(total + Decimal(c.rng.choice([-40, -15, -5, 5, 15, 40])))
    params.update(limit=str(lim), answer=total > lim)
    return Q(c, "invoice_total", "noul", var, "Invoice", lines, wording.format(ref=c.ref, lim=c.m(lim)), just, params, table=_table(c, rows), unknown=unk, intended=total > lim)


W_N2 = _w("po_mismatch", choice=["Which line explains the difference between purchase order and invoice {ref}?", "Which item accounts for the gap between the PO and invoice totals?"],
          choice_amount=["By how much does invoice {ref} differ from its purchase order?", "What is the difference between the invoice and PO totals for {ref}?",
                         "How far apart are the invoice total and the PO total for {ref}, and in which direction?"])


def n_po_mismatch(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N2)
    po = _lines_items(c, c.rng.randint(4, 5)); inv = [dict(r) for r in po]
    i = c.rng.randrange(len(inv)); r = inv[i]
    if c.rng.random() < 0.5:
        r["qty"] = r["qty"] + c.rng.choice([-1, 1, 2]) if r["qty"] > 1 else r["qty"] + 1
    else:
        r["unit_price"] = q2(r["unit_price"] + Decimal(c.rng.choice([1, 2, 5, -1])) * Decimal("0.50") * c.rng.randint(1, 4))
        if r["unit_price"] <= 0: r["unit_price"] = q2(po[i]["unit_price"] + 1)
    r["amount"] = q2(r["qty"] * r["unit_price"])
    decoy = c.rng.choice([j for j in range(len(inv)) if j != i])
    inv[decoy] = dict(inv[decoy], item=inv[decoy]["item"] + " (renamed on invoice)")
    diff = sum(x["amount"] for x in inv) - sum(x["amount"] for x in po)
    lines = [f"Purchase order and supplier invoice for {c.ref}. Item descriptions on the invoice follow the supplier's catalogue and may be worded differently.",
             "Purchase order lines:", *[f"PO: {x['item']} | qty {x['qty']} | {c.m(x['unit_price'])} each | {c.m(x['amount'])}" for x in po],
             "Invoice lines:", *[f"Invoice: {x['item']} | qty {x['qty']} | {c.m(x['unit_price'])} each | {c.m(x['amount'])}" for x in inv],
             f"Totals: PO {c.m(sum(x['amount'] for x in po))}; invoice {c.m(sum(x['amount'] for x in inv))}."]
    params = {"po": [[x["qty"], str(x["unit_price"])] for x in po], "invoice": [[x["qty"], str(x["unit_price"])] for x in inv], "items": [x["item"] for x in po]}
    just = f"Only the {po[i]['item']} line changes amount ({c.m(po[i]['amount'])} on the PO, {c.m(r['amount'])} on the invoice); the renamed line has the same amount."
    if qtype == "choice":
        ch = make_choice(c, (po[i]["item"], po[i]["item"]), [(x["item"], x["item"]) for j, x in enumerate(po) if j != i], k=6)
        params.update(answer=po[i]["item"], option_values=ch.pop("option_values"))
        return Q(c, "po_mismatch", "choice", var, "PO vs invoice", lines, wording.format(ref=c.ref), just, params, **ch)
    f = lambda x: f"Invoice {'higher' if x > 0 else 'lower'} by {c.m(abs(x))}"  # noqa: E731
    wrong = [-diff, diff * 2, diff + po[decoy]["amount"] if diff > 0 else diff - po[decoy]["amount"], po[i]["amount"] - r["amount"] + (po[i]["unit_price"] if diff > 0 else -po[i]["unit_price"])]
    ch = make_choice(c, (f(diff), str(diff)), [(f(w), str(w)) for w in wrong if w != 0], more=lambda k: (f(diff + k), str(diff + k)))
    params.update(answer=str(diff), option_values=ch.pop("option_values"))
    return Q(c, "po_mismatch", "choice_amount", var, "PO vs invoice", lines, wording.format(ref=c.ref), just + f" Difference {f(diff)}.", params, **ch)


W_N3 = _w("bank_reconciliation", noul=["After the reconciling items, do the ledger and the bank statement agree for {acct}?", "Does account {acct} reconcile once the listed items are taken into account?"],
          choice=["What unexplained difference remains on account {acct} after the reconciling items?", "How much of the gap on {acct} is still unexplained?",
                   "Once outstanding cheques, deposits in transit and the unbooked charge are applied, what difference is left on {acct}?"])


def n_bank_rec(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N3)
    acct = c.code("ACCT")
    true_bal = q2(Decimal(c.rng.randint(5000, 90000)) + Decimal(c.rng.randint(0, 99)) / 100)
    chq = [q2(Decimal(c.rng.randint(50, 3000)) + Decimal(c.rng.randint(0, 99)) / 100) for _ in range(c.rng.randint(1, 3))]
    dep = [q2(Decimal(c.rng.randint(100, 5000))) for _ in range(c.rng.randint(1, 2))]
    fee = q2(Decimal(c.rng.randint(5, 60)) + Decimal(c.rng.choice([0, 50])) / 100)
    unexplained = c.rng.choice([Decimal(0), Decimal(0), Decimal(9 * c.rng.randint(1, 90)), Decimal(c.rng.randint(1, 200)) / 2])
    unexplained = q2(unexplained)
    ledger = true_bal + fee        # the ledger has not yet booked the bank fee
    bank = true_bal - sum(dep) + sum(chq) + unexplained
    lines = [f"Month-end reconciliation for account {acct} ({c.org}).", f"Balance per ledger: {c.m(ledger)}. Balance per bank statement: {c.m(bank)}.",
             "Outstanding cheques (issued, not yet presented): " + ", ".join(c.m(x) for x in chq) + ".",
             "Deposits in transit (booked, not yet on the statement): " + ", ".join(c.m(x) for x in dep) + ".",
             f"Bank charge on the statement not yet booked in the ledger: {c.m(fee)}.",
             "Adjusted bank balance = statement + deposits in transit - outstanding cheques; adjusted ledger = ledger - unbooked charges."]
    adj_bank = bank + sum(dep) - sum(chq); adj_led = ledger - fee; gap = adj_bank - adj_led
    params = {"ledger": str(ledger), "bank": str(bank), "cheques": [str(x) for x in chq], "deposits": [str(x) for x in dep], "fee": str(fee), "answer_gap": str(gap)}
    just = f"Adjusted bank {c.m(adj_bank)}, adjusted ledger {c.m(adj_led)}; difference {c.m(abs(gap))}."
    if qtype == "noul":
        params["answer"] = gap == 0
        return Q(c, "bank_reconciliation", "noul", var, "Bank reconciliation", lines, wording.format(acct=acct), just, params, intended=gap == 0)
    f = lambda x: "None; they agree" if x == 0 else f"{c.m(abs(x))} ({'bank' if x > 0 else 'ledger'} higher)"  # noqa: E731
    wrong = [bank - ledger, gap + fee, gap - 2 * sum(dep) + 2 * sum(chq) if chq else gap + 10, gap + fee * 2]
    ch = make_choice(c, (f(gap), str(gap)), [(f(w), str(w)) for w in wrong], more=lambda k: (f(gap + k * 10), str(gap + k * 10)))
    params.update(answer=str(gap), option_values=ch.pop("option_values"))
    return Q(c, "bank_reconciliation", "choice", var, "Bank reconciliation", lines, wording.format(acct=acct), just, params, **ch)


W_N4 = _w("stacked_discounts", choice=["What does the customer pay for the {item}?", "What is the final price of the {item} after all reductions?",
                                       "Applying the reductions in the stated order, what is the {item}'s price?"])


def n_stacked(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N4)
    name, lo, hi = c.rng.choice(c.v["items"]); P = q2(c.price(max(lo, 20), max(hi, 60)) * c.rng.randint(1, 5))
    a, b = c.rng.choice([10, 15, 20, 25, 30]), c.rng.choice([5, 10, 12, 15])
    fixed = c.rng.random() < 0.5; coupon = q2(Decimal(c.rng.choice([5, 10, 15, 20])))
    s1 = q2(P * (100 - a) / 100)
    if fixed:
        order = c.rng.choice(["percent_first", "coupon_first"])
        final = q2(s1 - coupon) if order == "percent_first" else q2((P - coupon) * (100 - a) / 100)
        other = q2((P - coupon) * (100 - a) / 100) if order == "percent_first" else q2(s1 - coupon)
        lines = [f"List price of the {name} bundle for {c.ref}: {c.m(P)}.", f"Seasonal markdown: {a}% off.", f"Voucher: {c.m(coupon)} off.",
                 "The markdown is applied first and the voucher is taken off the marked-down price." if order == "percent_first" else "The voucher is taken off the list price first; the markdown then applies to what remains."]
        wrong = [other, q2(P * (100 - a) / 100), q2(P - coupon), q2(final + coupon)]
        just = f"{'(' + c.m(P) + ' less ' + str(a) + '%) less ' + c.m(coupon) if order == 'percent_first' else '(' + c.m(P) + ' less ' + c.m(coupon) + ') less ' + str(a) + '%'} = {c.m(final)}."
        params = {"price": str(P), "a": a, "coupon": str(coupon), "order": order}
    else:
        final = q2(s1 * (100 - b) / 100)
        lines = [f"List price of the {name} bundle for {c.ref}: {c.m(P)}.", f"Trade discount: {a}%.", f"Loyalty discount: a further {b}%, applied to the already-discounted price.",
                 "Prices are rounded to the cent after each discount."]
        wrong = [q2(P * (100 - a - b) / 100), s1, q2(P * (100 - b) / 100), q2(P - P * a * b / 10000)]
        just = f"{c.m(P)} x {100 - a}% = {c.m(s1)}; x {100 - b}% = {c.m(final)}."
        params = {"price": str(P), "a": a, "b": b}
    ch = money_choice(c, final, wrong); params.update(answer=str(final), option_values=ch.pop("option_values"))
    return Q(c, "stacked_discounts", "choice", var, "Pricing", lines, wording.format(item=f"{name} bundle"), just, params, **ch)


W_N5 = _w("weight_conversion", choice=["What is the total weight of the consignment in kilograms, to one decimal place?", "What does the manifest for {ref} weigh in kg (one decimal)?",
                                 "Converted to kilograms and rounded to one decimal, what is the consignment weight for {ref}?"],
          noul=["Is consignment {ref} within the {lim} kg limit?", "Does the total weight of {ref} stay at or under {lim} kg?"])


def n_weight(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N5)
    F = Decimal("0.4536"); parcels = []
    for j in range(c.rng.randint(4, 6)):
        if c.rng.random() < 0.5:
            parcels.append({"parcel": f"P{j + 1}", "weight": f"{c.rng.randint(3, 60)}.{c.rng.randint(0, 9)} kg"})
        else:
            parcels.append({"parcel": f"P{j + 1}", "weight": f"{c.rng.randint(5, 120)} lb"})
    def kg(w):
        v, u = w.split(); return Decimal(v) if u == "kg" else Decimal(v) * F
    total = sum(kg(p["weight"]) for p in parcels); t1 = total.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    naive = sum(Decimal(p["weight"].split()[0]) for p in parcels).quantize(Decimal("0.1"))
    inv = sum(Decimal(p["weight"].split()[0]) / F if p["weight"].endswith("lb") else Decimal(p["weight"].split()[0]) for p in parcels).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    lines = [f"Manifest for {c.v['case']} {c.ref}; the two packing stations record weights in different units.", "Conversion: 1 lb = 0.4536 kg.",
             "Total weight is the sum of all parcels, converted to kilograms and rounded to one decimal place at the end."]
    params = {"parcels": [p["weight"] for p in parcels], "answer_kg": str(t1)}
    just = f"Sum in kg = {total.quantize(Decimal('0.001'))} -> {t1} kg."
    if qtype == "choice":
        f = lambda x: f"{x} kg"  # noqa: E731
        ch = make_choice(c, (f(t1), str(t1)), [(f(naive), str(naive)), (f(inv), str(inv)), (f(t1 + 1), str(t1 + 1)), (f(t1 - Decimal('0.5')), str(t1 - Decimal('0.5')))])
        params.update(answer=str(t1), option_values=ch.pop("option_values"))
        return Q(c, "weight_conversion", "choice", var, "Manifest", lines, wording.format(ref=c.ref), just, params, table=parcels, **ch)
    lim = int(t1) + c.rng.choice([-6, -2, 1, 3, 8])
    params.update(limit=lim, answer=total <= lim)
    return Q(c, "weight_conversion", "noul", var, "Manifest", lines, wording.format(ref=c.ref, lim=lim), just + f" Limit {lim} kg.", params, table=parcels, intended=total <= lim)


W_N6 = _w("proration", choice=["What is the prorated charge for {ref} for {month}?", "How much is billed for the partial month of {month} on {ref}?",
                               "Under the proration rule, what does {ref} pay for {month}?",
                               "What is the first-month charge on {ref} ({month}) after proration?"])


def n_proration(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N6)
    d0 = c.date(); D = calendar.monthrange(d0.year, d0.month)[1]; day = c.rng.randint(2, D - 1)
    F = q2(Decimal(c.rng.randint(20, 900)) + Decimal(c.rng.choice([0, 0, 50, 99])) / 100)
    rem = D - day + 1; ans = q2(F * rem / D)
    month = f"{MONTHS[d0.month]} {d0.year}"
    start = d0.replace(day=day)
    lines = [f"{c.v['case'].capitalize()} {c.ref} started on {c.fd(start)} at a monthly price of {c.m(F)}.",
             "The first month is prorated by calendar days: monthly price x (days from the start date to the end of the month, both included) / days in that month.",
             "The result is rounded to the cent."]
    wrong = [q2(F * rem / 30), q2(F * (rem - 1) / D), q2(F * (day - 1) / D), q2(F * (rem + 1) / D)]
    ch = money_choice(c, ans, wrong)
    params = {"monthly": str(F), "start": start.isoformat(), "answer": str(ans), "option_values": ch.pop("option_values")}
    return Q(c, "proration", "choice", var, "Proration", lines, wording.format(ref=c.ref, month=month), f"{c.m(F)} x {rem}/{D} = {c.m(ans)}.", params, **ch)


W_N7 = _w("net_from_gross", choice=["What is the net price of the {item}, excluding tax?", "How much of the {item} price is tax?"])


def n_net_from_gross(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N7)
    name, lo, hi = c.rng.choice(c.v["items"]); G = q2(c.price(max(lo, 10), max(hi, 40)) * c.rng.randint(1, 4)); t = c.rng.choice([5, 10, 15, 20, 21, 25, 8])
    net = q2(G * 100 / (100 + t)); tax = G - net
    lines = [f"Receipt line for {c.ref}: {name}, {c.m(G)} including tax.", f"Tax rate: {t}%, charged on the net price.",
             "Receipts show tax-inclusive prices only; net and tax amounts are rounded to the cent."]
    params = {"gross": str(G), "rate": t}
    if var == 0:
        wrong = [q2(G * (100 - t) / 100), q2(G - G * t / 100 - 1), q2(G * 100 / (100 + t + 5)), tax]
        ch = money_choice(c, net, wrong); params.update(answer=str(net), ask="net", option_values=ch.pop("option_values"))
    else:
        wrong = [q2(G * t / 100), net, q2(G * t / (100 + t + 5)), q2(tax + 1)]
        ch = money_choice(c, tax, wrong); params.update(answer=str(tax), ask="tax", option_values=ch.pop("option_values"))
    return Q(c, "net_from_gross", "choice", var, "Tax-inclusive price", lines, wording.format(item=name), f"Net = {c.m(G)} x 100 / {100 + t} = {c.m(net)}; tax = {c.m(G)} - {c.m(net)} = {c.m(tax)}.", params, **ch)


W_N8 = _w("budget_remaining", choice=["How much of the {line} budget remains?", "What is the remaining balance on budget line {line}?"],
          noul=["Is budget line {line} overspent?", "Has committed spend on {line} exceeded its budget?"])


def n_budget(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N8)
    line = c.code("BL"); B = q2(Decimal(c.rng.randint(20, 200) * 100)); rows = []
    for name, lo, hi in c.rng.sample(c.v["items"], 3):
        rows.append({"item": name, "amount": q2(c.price(lo, hi) * c.rng.randint(3, 25)), "status": "approved"})
    pend = {"item": c.rng.choice(["additional order", "draft requisition", "quote awaiting sign-off"]), "amount": q2(Decimal(c.rng.randint(5, 40) * 50)), "status": "pending approval"}
    rows.insert(c.rng.randrange(len(rows) + 1), pend)
    tr = q2(Decimal(c.rng.randint(1, 30) * 100)); t_in = c.rng.random() < 0.5
    spent = sum(r["amount"] for r in rows if r["status"] == "approved")
    remaining = B + (tr if t_in else -tr) - spent
    unk = c.unknown() and abs(remaining) < tr   # withholding the transfer amount must leave the answer genuinely open
    lines = [f"Budget line {line} ({c.v['team']}, {c.org}): original budget {c.m(B)}.",
             f"Transfer {'into' if t_in else 'out of'} this line approved by {c.person()}: {c.m(tr)}." if not unk else
             f"A transfer {'into' if t_in else 'out of'} this line was approved by {c.person()}; the amount has not yet been entered.",
             "Only approved spend is committed against the budget; pending items are not counted until approved."]
    table = [{"item": r["item"], "amount": c.m(r["amount"]), "status": r["status"]} for r in rows]
    params = {"budget": str(B), "transfer": str(tr), "transfer_in": t_in, "spend": [[str(r["amount"]), r["status"]] for r in rows]}
    just = f"{c.m(B)} {'+' if t_in else '-'} {c.m(tr)} - approved {c.m(spent)} = {c.m(remaining)}."
    if qtype == "noul":
        params.update(answer=remaining < 0, withheld="transfer" if unk else None)
        return Q(c, "budget_remaining", "noul", var, "Budget", lines, wording.format(line=line), just, params, table=table, unknown=unk, intended=remaining < 0)
    f = lambda x: f"{c.m(x)} remaining" if x >= 0 else f"Overspent by {c.m(-x)}"  # noqa: E731
    wrong = [remaining - pend["amount"], B - spent, B + (-tr if t_in else tr) - spent, remaining + pend["amount"]]
    ch = make_choice(c, (f(remaining), str(remaining)), [(f(w), str(w)) for w in wrong], more=lambda k: (f(remaining + k * 25), str(remaining + k * 25)))
    params.update(answer=str(remaining), option_values=ch.pop("option_values"), withheld="transfer" if unk else None)
    return Q(c, "budget_remaining", "choice", var, "Budget", lines, wording.format(line=line), just, params, table=table, unknown=unk, **ch)


W_N9 = _w("fx_fee", choice=["How much is charged in {to} for the payment on {ref}?", "What is the total {to} debit for {ref}, including fees?",
                               "After conversion and card fees, how much leaves the {to} account for {ref}?"])


def n_fx(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N9)
    (fc, fs), (tc, ts) = c.rng.sample(CURRENCIES, 2)
    A = q2(Decimal(c.rng.randint(100, 9000)) + Decimal(c.rng.choice([0, 50, 25])) / 100); r = Decimal(c.rng.randint(5000, 19000)) / 10000
    p = c.rng.choice([Decimal("1"), Decimal("1.5"), Decimal("2"), Decimal("2.5"), Decimal("3")]); fixed = q2(Decimal(c.rng.choice([0, 2, 3, 5, 10])))
    conv = q2(A * r); fee = q2(conv * p / 100); total = conv + fee + fixed
    lines = [f"Supplier payment for {c.ref}: {money(A, fs)} ({fc}), paid from a {tc} account.", f"Rate applied: 1 {fc} = {r} {tc}.",
             f"Card fee: {p}% of the converted {tc} amount plus a fixed {money(fixed, ts)} per payment." if fixed else f"Card fee: {p}% of the converted {tc} amount; no fixed fee.",
             "Each step is rounded to the cent."]
    wrong = [q2(A / r) + q2(q2(A / r) * p / 100) + fixed, conv + q2(A * p / 100) + fixed, conv + fee, conv + fixed]
    ch = make_choice(c, (money(total, ts), str(total)), [(money(w, ts), str(q2(w))) for w in wrong],
                     more=lambda k: (money(total + k * 3, ts), str(q2(total + k * 3))))
    params = {"amount": str(A), "rate": str(r), "pct": str(p), "fixed": str(fixed), "answer": str(total), "option_values": ch.pop("option_values")}
    return Q(c, "fx_fee", "choice", var, "Foreign-currency payment", lines, wording.format(to=tc, ref=c.ref),
             f"{money(A, fs)} x {r} = {money(conv, ts)}; fee {p}% = {money(fee, ts)}; + {money(fixed, ts)} = {money(total, ts)}.", params, **ch)


W_N10 = _w("stock_variance", choice=["What is the stock variance for {sku} (counted minus expected)?", "By how much does the physical count of {sku} differ from the book figure?"],
           noul=["Is the count variance on {sku} within the tolerance?", "Does {sku} pass the stock-count tolerance check?"])


def n_stock(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N10)
    sku = c.code("SKU"); O = c.rng.randint(50, 900); R = [c.rng.randint(10, 200) for _ in range(2)]; S = [c.rng.randint(10, 250) for _ in range(2)]
    W = c.rng.randint(0, 20); Rt = c.rng.randint(0, 15)
    exp = O + sum(R) - sum(S) - W + Rt
    if exp < 0:
        O += -exp + 20; exp = O + sum(R) - sum(S) - W + Rt
    var_ = c.rng.choice([0, -1, -3, -7, 2, 5, -12, 9]); counted = exp + var_; tol = c.rng.choice([2, 3, 5, 8])
    lines = [f"Stock roll-forward for {sku} at the {c.org} store, period ending {c.fd(c.date())}.", f"Opening balance: {O} units.",
             f"Receipts: {R[0]} and {R[1]} units.", f"Shipments out: {S[0]} and {S[1]} units.", f"Write-offs (damaged): {W} units.",
             f"Customer returns put back into saleable stock: {Rt} units.", f"Physical count at period end: {counted} units.",
             f"Tolerance: a variance of up to {tol} units either way is accepted without investigation."]
    params = {"opening": O, "receipts": R, "shipments": S, "writeoffs": W, "returns": Rt, "counted": counted, "tolerance": tol, "variance": var_}
    just = f"Expected {O} + {sum(R)} - {sum(S)} - {W} + {Rt} = {exp}; counted {counted}; variance {var_:+d}."
    if qtype == "noul":
        params["answer"] = abs(var_) <= tol
        return Q(c, "stock_variance", "noul", var, "Stock count", lines, wording.format(sku=sku), just, params, intended=abs(var_) <= tol)
    f = lambda x: "No variance" if x == 0 else f"{'Shortage' if x < 0 else 'Surplus'} of {abs(x)} units"  # noqa: E731
    wrong = [-var_, var_ - W, var_ + Rt, var_ - 2 * Rt, var_ + W]
    ch = make_choice(c, (f(var_), str(var_)), [(f(w), str(w)) for w in wrong], more=lambda k: (f(var_ + k), str(var_ + k)))
    params.update(answer=var_, option_values=ch.pop("option_values"))
    return Q(c, "stock_variance", "choice", var, "Stock count", lines, wording.format(sku=sku), just, params, **ch)


W_N11 = _w("change_band", score=["Which change band does {metric} for {ref} fall into?", "How should the change in {metric} be classified on the band scale?",
                                 "On the review scale, where does the period-on-period change in {metric} sit?"])
CHANGE_BANDS = [(-100, -10, "fell by more than 10%"), (-10, 0, "fell by up to 10%"), (0, 10, "grew by less than 10%"), (10, 25, "grew by at least 10% but less than 25%"), (25, 10 ** 9, "grew by 25% or more")]


def n_change_band(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N11)
    metric = c.rng.choice(["monthly volume", "net revenue", "processed items", "billable hours"])
    while True:
        v1 = c.rng.randint(200, 20000); v2 = int(v1 * c.rng.uniform(0.7, 1.45))
        ch_ = Fraction(v2 - v1, v1) * 100
        if ch_ not in (-10, 0, 10, 25) and v2 != v1: break
    # boundaries: band 0 = change < -10; band 1 = -10 <= change < 0; band 2 = 0 <= change < 10; band 3 = 10 <= change < 25; band 4 = >= 25
    lvl = 0 if ch_ < -10 else 1 if ch_ < 0 else 2 if ch_ < 10 else 3 if ch_ < 25 else 4
    levels = [{"value": i, "description": d} for i, (_, _, d) in enumerate(CHANGE_BANDS)]
    lines = [f"{metric.capitalize()} for {c.ref}: previous period {v1:,}; current period {v2:,}.",
             "Change is measured against the previous period: (current - previous) / previous.",
             f"A note from {c.person()} quotes the change 'relative to the current period'; that is not the reporting convention."]
    params = {"previous": v1, "current": v2, "answer": lvl}
    return Q(c, "change_band", "score", var, "Period comparison", lines, wording.format(metric=metric, ref=c.ref),
             f"({v2} - {v1}) / {v1} = {float(ch_):.2f}% -> band {lvl}.", params, levels=levels, intended=lvl)


W_N12 = _w("usage_split", choice=["What share of the {cost} is charged to {team}?", "How much does {team} pay towards the {cost}?", "Under the allocation rule, what is {team}'s portion of the {cost}?",
                                  "Allocating by usage, what amount of the {cost} lands on {team}?"])


def n_usage_split(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N12)
    teams = c.rng.sample(["Operations", "Finance", "Sales", "Support", "Warehouse", "Marketing", "Legal"], c.rng.randint(3, 4))
    usage = [c.rng.randint(5, 400) for _ in teams]; heads = [c.rng.randint(2, 40) for _ in teams]
    C = q2(Decimal(c.rng.randint(500, 20000)) + Decimal(c.rng.choice([0, 50])) / 100); i = c.rng.randrange(len(teams))
    cost = c.rng.choice(["shared platform invoice", "courier contract", "print room bill", "cloud storage bill"])
    unit = c.rng.choice(["GB-months", "consignments", "pages", "API calls (thousands)"])
    ans = q2(C * usage[i] / sum(usage))
    lines = [f"The {cost} for {c.ref} is {c.m(C)}.", f"It is allocated to teams in proportion to recorded usage ({unit}); headcount is shown for information only.",
             "Each team's share is rounded to the cent."]
    table = [{"team": t, "usage": u, "headcount": h} for t, u, h in zip(teams, usage, heads)]
    wrong = [q2(C / len(teams)), q2(C * heads[i] / sum(heads)), q2(C * usage[(i + 1) % len(teams)] / sum(usage)), q2(C * usage[i] / (sum(usage) - usage[i]))]
    ch = money_choice(c, ans, wrong)
    params = {"total": str(C), "usage": usage, "i": i, "answer": str(ans), "option_values": ch.pop("option_values")}
    return Q(c, "usage_split", "choice", var, "Cost allocation", lines, wording.format(cost=cost, team=teams[i]),
             f"{c.m(C)} x {usage[i]}/{sum(usage)} = {c.m(ans)}.", params, table=table, **ch)


W_N13 = _w("free_shipping", noul=["Does order {ref} qualify for free delivery?", "Is delivery on {ref} free under the threshold rule?"],
           choice=["What is the total payable for order {ref}?", "How much will the buyer be charged for {ref} in total?"])


def n_free_shipping(c: Ctx) -> dict:
    qtype, var, wording = pick(c, W_N13)
    rows = _lines_items(c, c.rng.randint(2, 3)); sub = sum(r["amount"] for r in rows)
    coupon = q2(Decimal(c.rng.choice([5, 10, 15, 20, 25])))
    T = q2(Decimal(int(sub - coupon) + c.rng.choice([-12, -3, 2, 6, 15])))
    fee = q2(Decimal(c.rng.choice([4, 5, 6, 8])) + Decimal("0.99"))
    after = sub - coupon; free = after >= T
    total = after + (0 if free else fee)
    lines = [f"Order {c.ref} (prices include tax).", f"Voucher applied: {c.m(coupon)} off the goods total.",
             f"Free delivery applies when the goods total after vouchers is at least {c.m(T)}; otherwise delivery costs {c.m(fee)}."]
    params = {"lines": [[r["qty"], str(r["unit_price"])] for r in rows], "coupon": str(coupon), "threshold": str(T), "fee": str(fee)}
    just = f"Goods {c.m(sub)} - {c.m(coupon)} = {c.m(after)} vs threshold {c.m(T)}: delivery {'free' if free else c.m(fee)}."
    if qtype == "noul":
        params["answer"] = bool(free)
        return Q(c, "free_shipping", "noul", var, "Order", lines, wording.format(ref=c.ref), just, params, table=_table(c, rows), intended=bool(free))
    wrong = [after + (fee if free else 0), sub + (0 if sub >= T else fee), sub, after + fee + fee]
    ch = money_choice(c, total, wrong); params.update(answer=str(total), option_values=ch.pop("option_values"))
    return Q(c, "free_shipping", "choice", var, "Order", lines, wording.format(ref=c.ref), just + f" Total {c.m(total)}.", params, table=_table(c, rows), **ch)


NUMERIC_KINDS = {"invoice_total": n_invoice_total, "po_mismatch": n_po_mismatch, "bank_reconciliation": n_bank_rec, "stacked_discounts": n_stacked,
                 "weight_conversion": n_weight, "proration": n_proration, "net_from_gross": n_net_from_gross, "budget_remaining": n_budget, "fx_fee": n_fx,
                 "stock_variance": n_stock, "change_band": n_change_band, "usage_split": n_usage_split, "free_shipping": n_free_shipping}
NUMERIC_WORDINGS = dict(TW); TW.clear()

KINDS = {"temporal_arithmetic": TEMPORAL_KINDS, "probability_exact": PROBABILITY_KINDS, "numeric_reconciliation": NUMERIC_KINDS}
WORDINGS = {"temporal_arithmetic": TEMPORAL_WORDINGS, "probability_exact": PROBABILITY_WORDINGS, "numeric_reconciliation": NUMERIC_WORDINGS}


def templates(family: str) -> list[str]:
    """Every (kind, question type, wording) template id for a family."""
    return [f"{family}.{k}.{t}{i}" for k, ws in WORDINGS[family].items() for t, lst in ws.items() for i in range(len(lst))]


# ------------------------------------------------------------------------------------------------- document assembly
INTROS = [
    "{kind} prepared by {person} ({team}) for {org}. Figures below are copied from the source systems without adjustment.",
    "{org} | {kind}. Owner: {person}, {team}. This file gathers the records needed to settle the open points on {ref}.",
    "Working copy of the {kind} for {ref}, maintained by {person} in {team} at {org}.",
    "{kind} for {org}. Compiled by {person} ({team}); the sections below are independent unless they say otherwise.",
]
NOTES = [
    "Entries were exported from the {system} system on {date} and were not re-keyed.",
    "This version replaces the draft circulated on {date}; earlier printouts should be discarded.",
    "Queries go to {person} in {team}; responses are logged against {ref}.",
    "Cost centre {code} is charged for any follow-up work arising from this file.",
    "{person} reviewed the entries on {date}; no manual overrides were recorded.",
    "Attachments {code} and {code2} are held in the shared drive and are not reproduced here.",
    "Records older than {months} months are archived and do not appear in this file.",
    "The {team} team reviews open {cases} every {weekday}.",
]
SYSTEMS = ["ledger", "case-management", "warehouse", "booking", "ticketing", "billing", "scheduling"]


def assemble_document(c: Ctx, doc_kind: str, qs: list[dict], shape: str):
    intro = c.rng.choice(INTROS).format(kind=doc_kind.capitalize(), person=c.person(), team=c.v["team"], org=c.org, ref=c.ref)
    notes = []
    body_words = sum(len(" ".join(q["lines"]).split()) + 6 * len(q.get("table") or []) for q in qs)
    for tmpl in c.rng.sample(NOTES, 3 if body_words > 110 else 6):
        notes.append(tmpl.format(system=c.rng.choice(SYSTEMS), date=c.fd(c.date()), person=c.person(), team=c.v["team"], ref=c.ref, code=c.code(),
                                 code2=c.code(), months=c.rng.choice([12, 18, 24, 36]), cases=c.v["cases"], weekday=c.rng.choice(WEEKDAYS[:5])))
    header = {"reference": c.ref, "organisation": c.org, "currency": c.cur}
    if shape == "object":
        doc = {"document_type": doc_kind, **header, "summary": intro, "sections": []}
        for i, q in enumerate(qs, 1):
            sec = {"section": i, "heading": q["title"], "details": q["lines"]}
            if q.get("table"):
                sec["rows"] = q["table"]
            doc["sections"].append(sec)
        doc["notes"] = notes
        return doc
    parts = [f"{doc_kind.upper()} | {c.org} | Ref {c.ref} | Currency {c.cur}", intro, ""]
    for i, q in enumerate(qs, 1):
        parts.append(f"Section {i}. {q['title']}")
        parts += q["lines"]
        if q.get("table"):
            cols = list(q["table"][0])
            parts.append(" | ".join(cols))
            parts += [" | ".join(str(r[k]) for k in cols) for r in q["table"]]
        parts.append("")
    parts.append("Notes")
    parts += [f"- {n}" for n in notes]
    return "\n".join(parts)


_A_AN = __import__("re").compile(r"(?:(?<=^)|(?<= )|(?<=\n))([Aa]) (?=[aeioAEIO]|[uU](?!ni|se|su))")


def fix_articles(text: str) -> str:
    """'a expense claim' -> 'an expense claim' (lowercase 'a' anywhere; capital 'A' only at a sentence start)."""
    def sub(m):
        start = m.start()
        if m.group(1) == "A" and not (start == 0 or text[max(0, start - 2):start] in (". ", ": ", "; ", "? ", "! ") or text[start - 1:start] == "\n"):
            return m.group(0)
        return ("An " if m.group(1) == "A" else "an ")
    return _A_AN.sub(sub, text)


def _fix_all(value):
    if isinstance(value, str):
        return fix_articles(value)
    if isinstance(value, list):
        return [_fix_all(v) for v in value]
    if isinstance(value, dict):
        return {k: _fix_all(v) for k, v in value.items()}
    return value


def generate_document(index: int, seed: str = "prog-v1", family: str = "temporal_arithmetic", attempt: int = 0, shape: str | None = None) -> dict:
    """One writer-schema row: a domain document with three exact-answer questions of `family` (three distinct kinds)."""
    rng = random.Random(f"{seed}\0{family}\0{index}\0{attempt}")
    dom = DOMAINS[(index // len(PROG_FAMILIES)) % len(DOMAINS)]
    c = Ctx(rng, dom, family)
    doc_kind = rng.choice(dom["kinds"]); shape = shape or rng.choice(("string", "object"))
    kinds = rng.sample(sorted(KINDS[family]), 3)
    qs = [KINDS[family][k](c) for k in kinds]
    for q in qs:
        q["lines"] = _fix_all(q["lines"]); q["question"] = fix_articles(q["question"])
    doc = assemble_document(c, doc_kind, qs, shape)
    questions = []
    for q in qs:
        wq = {"family": family, "type": "choice" if q["type"].startswith("choice") else q["type"], "question": q["question"],
              "options": q.get("options"), "levels": q.get("levels"), "intended": q["intended"], "unknown_reason": q["unknown_reason"], "justification": q["justification"]}
        questions.append(wq)
    out = WriterOutput.model_validate({"document": doc, "document_kind": doc_kind, "questions": questions}).model_dump()
    h = hashlib.sha256(f"prog\0{seed}\0{family}\0{index}".encode()).hexdigest()[:12]
    batch = BATCH[family]
    plan = {"doc_id": f"{batch}-{dom['id']}-{h}", "index": index, "domain": dom["id"], "doc_kind": doc_kind, "batch": batch, "families": [family] * 3,
            "types": [q["type"] for q in questions], "unknowns": [q["intended"] is None for q in questions], "state_shape": shape,
            "kinds": kinds, "templates": [q["template"] for q in qs], "params": [q["params"] for q in qs], "seed": seed, "attempt": attempt,
            "generator": GENERATOR["version"]}
    return {"doc_id": plan["doc_id"], "batch": batch, "plan": plan, "writer": dict(GENERATOR), "output": out, "attempt": attempt}


def row_text(row: dict) -> str:
    doc = row["output"]["document"]
    return (doc if isinstance(doc, str) else json.dumps(doc)) + "\n" + "\n".join(q["question"] for q in row["output"]["questions"])


def generate(n_docs: int, seed: str = "prog-v1", families: list[str] | None = None, start: int = 0, reference: set | None = None,
             max_attempts: int = 8) -> tuple[list[dict], dict]:
    """Round-robin over families by index. Documents sharing any 8-gram with `reference` (or failing validation) are regenerated."""
    fams = list(families or PROG_FAMILIES)
    rows, stats = [], {"regenerated_contaminated": 0, "regenerated_invalid": 0, "by_family": {}, "by_type": {}, "unknown": 0}
    for i in range(start, start + n_docs):
        fam = fams[i % len(fams)]
        row, last = None, "contaminated"
        for attempt in range(max_attempts):
            try:
                row = generate_document(i, seed, fam, attempt)
            except (ValueError, ZeroDivisionError) as exc:  # a degenerate draw (e.g. too few distinct options): redraw
                stats["regenerated_invalid"] += 1; last = repr(exc); row = None; continue
            if reference and contamination_count(row_text(row), reference) > 0:
                stats["regenerated_contaminated"] += 1; row = None; continue
            break
        if row is None:
            raise RuntimeError(f"document {i} ({fam}) failed {max_attempts} attempts: {last}")
        rows.append(row)
        stats["by_family"][fam] = stats["by_family"].get(fam, 0) + 1
        for q in row["output"]["questions"]:
            stats["by_type"][q["type"]] = stats["by_type"].get(q["type"], 0) + 1
            stats["unknown"] += q["intended"] is None
    stats["docs"] = len(rows); stats["questions"] = 3 * len(rows)
    return rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", type=int, required=True); ap.add_argument("--seed", default="prog-v1"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", default=",".join(PROG_FAMILIES), help="comma-separated subset of " + ", ".join(PROG_FAMILIES))
    ap.add_argument("--start", type=int, default=0, help="first document index (a later top-up uses new indices)")
    ap.add_argument("--no-lint", action="store_true", help="skip the JevBench 8-gram check (tests only)")
    a = ap.parse_args(argv)
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in fams if f not in PROG_FAMILIES]
    if bad:
        ap.error(f"unknown programmatic families: {bad}")
    reference = None if a.no_lint else reference_ngrams(jevbench_public_files())
    rows, stats = generate(a.docs, a.seed, fams, a.start, reference)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    stats["reference_ngrams"] = len(reference) if reference else 0
    stats["templates"] = {f: len(templates(f)) for f in fams}
    print(json.dumps(stats))
    print(f"programmatic: {len(rows)} documents -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
