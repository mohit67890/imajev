"""Shared helpers for the phase-3 reasoning generators: seeded RNG, spintax, fictional names, domains, filler prose,
number formatting and option construction.

Every generator kind is a function ``kind(rng, difficulty) -> dict`` returning

    {"family": str, "state": str, "field": {...}, "gold": key|bool|int|None, "unknown_reason": None|str,
     "spec": {...formal inputs an independent checker can re-solve...},
     "unk": None | {"state": str, "field": {...} (optional), "reason": str, "spec": {...witness...}}}

``unk`` is the unknown variant of an answerable item (the decisive fact removed); ``spec`` of the variant carries a
*witness*: two values of the removed fact that are both consistent with what is left and lead to different answers.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import math
import random
import re
import string
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Callable

VERSION = "reasoning-v1"
GENERATOR = f"scripts/p3/gen_reasoning.py@{VERSION}"


class Skip(Exception):
    """The sampled parameters cannot give a clean item (tie, rounding boundary, non-unique answer); resample."""


def rng_for(*parts) -> random.Random:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ------------------------------------------------------------------------------------------------ spintax + templates
_SPIN = re.compile(r"\{([^{}]*)\}")


def spin(rng: random.Random, s: str) -> str:
    """Expand {a|b|c} groups (innermost first)."""
    while True:
        m = _SPIN.search(s)
        if not m:
            return s
        s = s[:m.start()] + rng.choice(m.group(1).split("|")) + s[m.end():]


def T(_rng: random.Random, _tpl: str, **kv) -> str:
    """Spin the template, then substitute $vars (values are never spun)."""
    if "co" in kv and "pc" not in kv:
        kv["pc"] = poss(str(kv["co"]))
    out = string.Template(spin(_rng, _tpl)).substitute({k: str(v) for k, v in kv.items()})
    return out[:1].upper() + out[1:] if out else out


def poss(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def words(s: str) -> int:
    return len(re.findall(r"\S+", s))


# ------------------------------------------------------------------------------------------------ names (all fictional)
FIRST = ["Amara", "Bastian", "Celine", "Dario", "Elin", "Farid", "Greta", "Hollis", "Ines", "Jonah", "Keiko", "Lorcan", "Maren", "Nikhil",
         "Odile", "Pavel", "Quinn", "Rosalind", "Soren", "Tamsin", "Ulla", "Viren", "Wren", "Ximena", "Yusuf", "Zelda", "Anouk", "Bram",
         "Cosima", "Desmond", "Esme", "Florian", "Gideon", "Halima", "Ivo", "Jolene", "Kasimir", "Leonie", "Mateo", "Noor", "Oskar",
         "Priya", "Rafael", "Sanne", "Tobias", "Uma", "Vikram", "Wilhelmina", "Yara", "Zoltan", "Adaeze", "Bjorn", "Chiara", "Dmitri",
         "Eshe", "Fionn", "Gulnara", "Hamza", "Isolde", "Jaspreet", "Kofi", "Liesel", "Mireille", "Nadia", "Orla", "Paulo", "Rhea",
         "Stellan", "Thandiwe", "Ulrich", "Valentina", "Wojciech", "Yohannes", "Zainab", "Aurelio", "Bettina", "Cyprian", "Delphine",
         "Emeka", "Freya", "Gustavo", "Hana", "Idris", "Juno", "Kalinda", "Lennart", "Mbali", "Nils", "Ottilie", "Pranav", "Rasmus",
         "Signe", "Tariq", "Undine", "Vesna", "Willem", "Xavier", "Yevgenia", "Zubair", "Agnieszka", "Benedikt", "Carmela", "Dorian"]
LAST = ["Achterberg", "Brannigan", "Castellano", "Dunmore", "Eskildsen", "Fairweather", "Galloway", "Hartigan", "Ingersoll", "Jaskolski",
        "Kovalenko", "Lindqvist", "Marchetti", "Nakashima", "Okonkwo", "Pemberton", "Quarshie", "Rasmussen", "Szabo", "Thorne", "Umberti",
        "Valcourt", "Whitlock", "Yarrow", "Zeller", "Abernethy", "Bellamy", "Coldwell", "Delacroix", "Ekwueme", "Farrow", "Grunwald",
        "Holloway", "Iversen", "Jablonski", "Kariuki", "Lachance", "Mbeki", "Nordin", "Oyelaran", "Petrakis", "Quintero", "Rourke",
        "Sandoval", "Takahashi", "Ueda", "Vasquez", "Wainwright", "Xu", "Yilmaz", "Zamora", "Albrecht", "Brightwater", "Cavanagh",
        "Dimitrov", "Engstrom", "Fonseca", "Gallagher", "Haddad", "Ivanova", "Joubert", "Kowalczyk", "Laurent", "Moreau", "Novak",
        "Olsen", "Pacheco", "Renwick", "Salazar", "Tennant", "Uriarte", "Vidal", "Westergaard", "Yamamoto", "Zdanowicz", "Ashdown",
        "Bergstrom", "Calloway", "Draper", "Esposito", "Fitzroy", "Gunawardena", "Hollander", "Ishikawa", "Kerrigan", "Lindgren", "Mwangi"]
_P1 = ["Vel", "Tor", "Mar", "Quin", "Dor", "Ash", "Bel", "Cor", "Fen", "Hal", "Kel", "Lun", "Mor", "Nor", "Pry", "Ros", "Sal", "Tam", "Ul",
       "Var", "Wex", "Yar", "Zen", "Bry", "Cal", "Ern", "Gal", "Ist", "Jor", "Lyr", "Ost", "Pel", "Rav", "Sev", "Thal", "Brin", "Cres", "Dun"]
_P2 = ["dale", "ford", "mont", "wick", "vane", "holm", "sted", "gate", "ridge", "crest", "mere", "wood", "field", "ton", "ley", "brook",
       "haven", "stone", "vale", "port", "moor", "shaw", "combe", "thorpe", "by", "well", "hurst", "burn"]
INDUSTRY = ["Logistics", "Analytics", "Foods", "Health", "Systems", "Partners", "Robotics", "Textiles", "Freight", "Labs", "Pharma", "Energy",
            "Software", "Capital", "Retail", "Components", "Media", "Outfitters", "Instruments", "Materials", "Networks", "Biotech", "Mobility",
            "Holdings", "Ceramics", "Optics", "Packaging", "Agritech", "Security", "Learning"]
_PROD = ["Aurora", "Nimbus", "Vector", "Halcyon", "Pioneer", "Zephyr", "Atlas", "Lumen", "Orbit", "Cascade", "Ember", "Summit", "Kestrel",
         "Meridian", "Solace", "Tundra", "Quill", "Sable", "Beacon", "Corvid", "Drift", "Fathom", "Glint", "Harbor", "Juniper", "Lattice"]
_CITY_SUF = ["ton", "bury", "ville", "haven", "port", "by", "stead", "field", "mouth", "ford", "wick", "minster", "holt", "brae"]
REGIONS = ["the Northern Reach", "the Amber Coast", "the Lowmarch", "the Eastern Fells", "the Saltplain", "the Riverlands", "the High Moors",
           "the Western Isles", "the Copper Basin", "the Greywater Delta", "the Southern Terraces", "the Linden Valley", "the Frost March",
           "the Cinder Hills", "the Ashen Steppe", "the Verdant Shelf"]


class Names:
    """Unique fictional names within one item."""

    def __init__(self, rng: random.Random, avoid=()):
        self.rng, self.used = rng, set(avoid)

    def _uniq(self, f):
        for _ in range(200):
            x = f()
            if x not in self.used:
                self.used.add(x)
                return x
        raise Skip("name pool exhausted")

    def person(self) -> str:
        return self._uniq(lambda: f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}")

    def first(self) -> str:
        return self._uniq(lambda: self.rng.choice(FIRST))

    def stem(self) -> str:
        return self._uniq(lambda: self.rng.choice(_P1) + self.rng.choice(_P2))

    def company(self, industry: str | None = None) -> str:
        return self._uniq(lambda: f"{self.rng.choice(_P1)}{self.rng.choice(_P2)} {industry or self.rng.choice(INDUSTRY)}")

    def city(self) -> str:
        return self._uniq(lambda: (self.rng.choice(_P1) + self.rng.choice(_CITY_SUF)).capitalize())

    def product(self) -> str:
        return self._uniq(lambda: f"{self.rng.choice(_PROD)} {self.rng.choice('ABCDEFGHKLMNPRSTVXZ')}{self.rng.randint(2, 99)}")

    def code(self, prefix: str) -> str:
        return self._uniq(lambda: f"{prefix}-{self.rng.randint(1000, 99999)}")


# ------------------------------------------------------------------------------------------------ domains
def _dom(org, teams, things, docs, sites, topics, units):
    return dict(org=org, teams=teams, things=things, docs=docs, sites=sites, topics=topics, units=units)


DOMAINS = {
    "retail": _dom(["Outfitters", "Stores", "Market", "Home Goods"], ["merchandising", "store operations", "e-commerce", "loss prevention", "buying"],
                   ["point-of-sale system", "planogram", "loyalty programme", "returns portal", "stockroom layout"],
                   ["weekly trading update", "store performance pack", "category review", "markdown plan"],
                   ["flagship store", "outlet store", "distribution hub", "pop-up unit"], ["till procedures", "fitting-room security", "gift-card fraud"], "units"),
    "logistics": _dom(["Freight", "Logistics", "Haulage", "Distribution"], ["dispatch", "warehouse", "fleet planning", "customs", "yard control"],
                      ["route optimiser", "dock scheduler", "handheld scanners", "cold-store monitor", "telematics feed"],
                      ["carrier scorecard", "lane review", "dock report", "peak-season plan"],
                      ["cross-dock", "north depot", "port terminal", "rail yard"], ["manual handling", "dangerous goods labelling", "forklift safety"], "pallets"),
    "hr": _dom(["People Services", "Group", "Workforce"], ["people operations", "payroll", "talent acquisition", "learning", "employee relations"],
               ["HR portal", "applicant tracking system", "time-clock terminals", "benefits platform", "org chart tool"],
               ["headcount report", "policy handbook", "engagement survey summary", "compensation review"],
               ["head office", "regional office", "shared-services centre", "training suite"], ["unconscious bias", "data protection", "hybrid working"], "employees"),
    "healthcare_admin": _dom(["Clinic", "Health Partners", "Medical Group"], ["patient services", "billing", "medical records", "scheduling", "referrals"],
                             ["appointment system", "claims clearing-house link", "records scanner", "patient portal", "coding software"],
                             ["denials report", "clinic utilisation summary", "billing audit", "intake checklist"],
                             ["outpatient wing", "billing office", "records annex", "satellite clinic"], ["coding updates", "privacy obligations", "front-desk triage"], "claims"),
    "finance": _dom(["Capital", "Holdings", "Finance"], ["accounts payable", "treasury", "financial planning", "tax", "internal audit"],
                    ["ERP ledger", "bank feed", "expense tool", "consolidation workbook", "treasury dashboard"],
                    ["month-end pack", "variance commentary", "cash forecast", "audit memo"],
                    ["finance floor", "treasury office", "shared-services hub", "audit room"], ["segregation of duties", "month-end cut-off", "expense policy"], "invoices"),
    "saas_ops": _dom(["Cloud", "Software", "Platform"], ["site reliability", "customer success", "billing operations", "platform engineering", "support"],
                     ["status page", "billing engine", "feature-flag service", "observability stack", "CRM"],
                     ["incident review", "renewal forecast", "capacity plan", "support-queue digest"],
                     ["us-east region", "eu-central region", "staging cluster", "support hub"], ["on-call etiquette", "secure coding", "customer escalations"], "accounts"),
    "legal_compliance": _dom(["Legal Services", "Compliance", "Advisory"], ["compliance", "contracts", "privacy", "regulatory affairs", "legal operations"],
                             ["contract repository", "policy register", "e-discovery tool", "consent tracker", "sanctions screening tool"],
                             ["compliance attestation", "contract register", "policy exception log", "regulatory calendar"],
                             ["legal floor", "records room", "board suite", "regional office"], ["anti-bribery rules", "records retention", "sanctions screening"], "matters"),
    "education": _dom(["Academy", "College", "Institute"], ["registry", "admissions", "student support", "timetabling", "assessment"],
                      ["learning platform", "timetabling system", "exam board portal", "library catalogue", "attendance tracker"],
                      ["module handbook", "assessment calendar", "progression report", "admissions digest"],
                      ["east campus", "exam hall", "library annex", "science block"], ["academic integrity", "accessibility", "safeguarding"], "students"),
    "manufacturing": _dom(["Components", "Industries", "Works"], ["quality assurance", "production planning", "maintenance", "procurement", "EHS"],
                          ["MES terminal", "calibration log", "press line", "paint booth", "torque tools"],
                          ["shift handover", "non-conformance log", "OEE report", "supplier audit"],
                          ["assembly hall", "machine shop", "goods-in bay", "test lab"], ["lockout-tagout", "5S housekeeping", "chemical storage"], "parts"),
    "hospitality": _dom(["Hotels", "Resorts", "Hospitality"], ["front office", "housekeeping", "food and beverage", "revenue management", "events"],
                        ["property-management system", "keycard encoders", "booking engine", "minibar sensors", "banquet planner"],
                        ["occupancy report", "guest-feedback digest", "rate strategy", "banquet order"],
                        ["lobby", "conference level", "rooftop terrace", "back-of-house"], ["allergen handling", "guest privacy", "fire safety"], "rooms"),
    "energy": _dom(["Energy", "Utilities", "Power"], ["metering", "grid operations", "field services", "customer billing", "asset management"],
                   ["meter-data platform", "outage map", "SCADA console", "work-order app", "tariff engine"],
                   ["outage report", "asset condition survey", "tariff review", "field-crew roster"],
                   ["substation", "control room", "field depot", "customer centre"], ["live-line safety", "vulnerable customers", "working at height"], "meters"),
    "nonprofit": _dom(["Foundation", "Trust", "Charity"], ["grants", "programmes", "fundraising", "volunteering", "finance"],
                      ["donor database", "grant portal", "volunteer rota", "impact tracker", "gift-aid tool"],
                      ["grant report", "trustee pack", "programme review", "donor update"],
                      ["community hub", "head office", "field office", "warehouse"], ["safeguarding", "fundraising regulation", "volunteer induction"], "grants"),
}
DOMAIN_IDS = sorted(DOMAINS)


# ------------------------------------------------------------------------------------------------ dates
WEEKDAYS = list(calendar.day_name)
MONTHS = list(calendar.month_name)


def fdate(d: dt.date, style: int) -> str:
    return [f"{d.day} {MONTHS[d.month]} {d.year}", d.isoformat(), f"{WEEKDAYS[d.weekday()]} {d.day} {MONTHS[d.month]} {d.year}",
            f"{MONTHS[d.month]} {d.day}, {d.year}", f"{WEEKDAYS[d.weekday()][:3]} {d.day} {MONTHS[d.month][:3]} {d.year}"][style % 5]


def rand_date(rng: random.Random, lo: int = 2025, hi: int = 2028) -> dt.date:
    start = dt.date(lo, 1, 1)
    return start + dt.timedelta(days=rng.randrange((dt.date(hi, 12, 31) - start).days))


# ------------------------------------------------------------------------------------------------ filler prose
FILLER = [
    "{Separately|Unrelated to the matter above|For context|On a different note}, $person {of|from} the $team team {noted|mentioned|flagged} that the $thing {review|audit|refresh|upgrade} {is scheduled for|has moved to|is now expected on} $date.",
    "The $team team {closed|resolved|worked through} $num {tickets|requests|queries|cases} {during|over|in} $month, {up from|compared with|against} $num2 the {month|period} before.",
    "$person {will be|is} {out of office|on leave|travelling|at a conference} {from|starting} $date; {queries|questions|anything urgent} should go to $person2 {in the meantime|until then|while they are away}.",
    "{A reminder|Please note|Note} that the $doc {template|format|layout} {changed|was updated|was revised} on $date, {and|so} {older copies|previous versions|earlier drafts} {should not be used|are superseded|have been archived}.",
    "{Parking|Desk booking|Badge access|Lift access} at the $site {will be|is} {limited|restricted|unavailable} on $weekday {because of|due to|owing to} {maintenance|a fire drill|contractor works|a floor move}.",
    "{The|Our} {quarterly|monthly|annual} {town hall|all-hands|offsite|planning day} {is|has been} {booked|set|scheduled} for $date at the $site; $person is {coordinating|organising|running} {catering|the agenda|logistics}.",
    "$num {people|staff|colleagues} {completed|finished|signed up for} the {refresher|induction|awareness} {course|module|session} on $topic {last|this} {week|month}, {with|and} $num2 more {booked|pending|still to go}.",
    "{Feedback|Input} on the {draft|proposed|revised} $doc {is due|should reach $person2} by $date{.|, after which it goes to the $team lead.}",
    "{The|A} {vendor|supplier|contractor} {meeting|call|review} with $org2 {was|has been} {moved|rescheduled|pushed back} to $weekday {at|around} $time.",
    "{According to|Per|As noted in} the {minutes|notes|summary} of the $weekday {stand-up|meeting|call}, $person {agreed to|will|offered to} {follow up on|look into|chase} the $thing {issue|question|backlog}.",
    "The {old|legacy|previous} $thing {will be|is being} {retired|decommissioned|switched off} {at the end of|by|after} $month, once $org2 {completes|signs off} the {migration|cut-over|data transfer}.",
    "{Coffee|The kitchen|The break room} on the {second|third|fourth} floor of the $site {will be|is} {closed|out of use} {until|through} $date {for|during} {refurbishment|a deep clean|plumbing repairs}.",
    "$person2 {asked|requested|suggested} that {future|all|any} $doc {submissions|drafts|versions} {include|carry|list} a {version number|change log|named owner} {from|starting} $month.",
    "The {IT|facilities|security} {desk|team} {logged|recorded|handled} $num {password resets|badge replacements|laptop swaps} {at|across} the $site {last|in the previous} {week|fortnight}.",
    "{Travel|Hotel|Mileage} claims for the $month {offsite|workshop|site visits} {must be filed|are due|should be submitted} {within|inside} $num3 days of {return|travel|the trip}.",
    "{In unrelated housekeeping|Housekeeping|One more item}: the {shared drive|team channel|wiki space} for $team {has been|was} {reorganised|renamed|archived} {by|under} $person.",
    "$org2 {confirmed|announced|told us} {on|by} $date that its {account manager|primary contact|liaison} is now $person2.",
    "The {fire marshal|first-aid|wellbeing champion} {rota|list|roster} for the $site {was refreshed to name|now names} $person {and|plus} $person2 {for|through} $month.",
    "{Survey|Pulse-check|Poll} {responses|results} from $num {respondents|colleagues|participants} {put|rated} the $thing at $num3 out of 10 {on average|overall}.",
    "{Next|The following|This} {quarter|month}'s $doc will be {circulated|shared|published} {on|by} $date {by|from} the $team team.",
    "{Guests|Visitors|Contractors} to the $site {must|should|are asked to} {sign in|register|collect a pass} at {reception|the front desk|the security lodge} {from|starting} $weekday.",
    "$person {closed out|finished|wrapped up} the {training|onboarding|handover} plan for {two|three|four} new {starters|joiners|hires} in $team {ahead of|before} $date.",
    "The {printer|scanner|projector} {in|near} {meeting room|room|suite} $num3 {is|has been} {out of order|replaced|serviced}; {use|try|book} {room|suite} $num4 {instead|for now|meanwhile}.",
    "{Budget|Capacity|Resourcing} {questions|requests|asks} for $month {go to|are handled by|sit with} $person2 {rather than|instead of|and not} the $team {inbox|queue|mailbox}.",
    "{Last|This} $weekday's {walk-through|inspection|site tour} of the $site {found|noted|recorded} $num3 {minor|small|cosmetic} {items|snags|observations}, all {assigned|allocated} to $person.",
]


def filler_para(rng: random.Random, dom: dict, names: Names, n_sent: int | None = None, anchor: dt.date | None = None) -> str:
    n_sent = n_sent or rng.randint(2, 4)
    base = anchor or rand_date(rng)
    out = []
    for t in rng.sample(FILLER, n_sent):
        d = base + dt.timedelta(days=rng.randint(-40, 60))
        out.append(T(rng, t, person=names.person(), person2=names.person(), team=rng.choice(dom["teams"]), thing=rng.choice(dom["things"]),
                     doc=rng.choice(dom["docs"]), site=rng.choice(dom["sites"]), topic=rng.choice(dom["topics"]), org2=names.company(),
                     date=fdate(d, rng.randrange(5)), month=MONTHS[d.month], weekday=rng.choice(WEEKDAYS[:5]), num=rng.randint(12, 480),
                     num2=rng.randint(10, 480), num3=rng.randint(2, 9), num4=rng.randint(10, 60),
                     time=f"{rng.randint(8, 16):02d}:{rng.choice(['00', '15', '30', '45'])}"))
    return " ".join(out)


TARGET_WORDS = {3: (100, 300), 4: (200, 550), 5: (350, 950)}


def assemble(rng: random.Random, head: list[str], body: list[str], dom: dict, names: Names, d: int, shuffle_body: bool = False,
             lo_hi: tuple[int, int] | None = None, anchor: dt.date | None = None) -> str:
    """head blocks stay first in order; body blocks keep relative order (unless shuffled); filler paragraphs are inserted
    at random gaps until the state reaches a word target drawn for the difficulty (never above 1,500 words)."""
    body = list(body)
    if shuffle_body:
        rng.shuffle(body)
    lo, hi = lo_hi or TARGET_WORDS[d]
    target = rng.randint(lo, hi)
    blocks = list(body)
    total = sum(words(b) for b in head + blocks)
    target = min(target, max(lo, 3 * total + 150))  # filler never swamps the content
    guard = 0
    while total < target and guard < 30:
        p = filler_para(rng, dom, names, anchor=anchor)
        if total + words(p) > 1500:
            break
        blocks.insert(rng.randint(0, len(blocks)), p)
        total += words(p)
        guard += 1
    return "\n\n".join(head + blocks)


# ------------------------------------------------------------------------------------------------ numbers
def dec(x) -> Decimal:
    with localcontext() as c:
        c.prec = 60
        if isinstance(x, Fraction):
            return Decimal(x.numerator) / Decimal(x.denominator)
        if isinstance(x, float):
            return Decimal(repr(x))
        return Decimal(x)


def rq(x, places: int) -> Decimal:
    """Round half-up to `places` decimals."""
    return dec(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def near_half(x, places: int, tol: float = 1e-7) -> bool:
    """True when x sits (almost) exactly on a rounding boundary at `places` decimals (then half-up vs half-even could differ)."""
    v = dec(x) * (Decimal(10) ** places)
    frac = v - v.to_integral_value(rounding="ROUND_FLOOR")
    return abs(float(frac) - 0.5) < tol


def fnum(x, places: int = 0) -> str:
    v = rq(x, places)
    return f"{v:,.{places}f}"


CURRENCIES = [("USD", "$", "pre"), ("EUR", "€", "pre"), ("GBP", "£", "pre"), ("CHF", "CHF ", "pre"), ("CAD", "C$", "pre"),
              ("AUD", "A$", "pre"), ("SGD", "S$", "pre"), ("SEK", " SEK", "post"), ("JPY", "¥", "pre"), ("INR", "₹", "pre")]


def fmoney(x, cur, places: int = 2) -> str:
    code, sym, pos = cur
    v = rq(x, places)
    neg = v < 0
    s = f"{abs(v):,.{places}f}"
    s = f"{sym}{s}" if pos == "pre" else f"{s}{sym}"
    return f"-{s}" if neg else s


def fpct(x, places: int = 1) -> str:
    return f"{rq(x, places):.{places}f}%"


def ffrac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


# ------------------------------------------------------------------------------------------------ options / fields
def option_key(text: str, taken: set[str]) -> str:
    t = text.replace("-", " minus ") if re.match(r"^-\s*\S", text) else text
    t = t.replace("%", " pct")
    key = re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")[:40] or "option"
    if key[0].isdigit():
        key = ("v_" + key)[:40]
    if key in ("unknown", "option_unknown"):
        key = "opt_" + key
    base, n = key, 2
    while key in taken:
        suf = f"_{n}"
        key = base[:40 - len(suf)] + suf
        n += 1
    taken.add(key)
    return key


def choice_field(rng: random.Random, question: str, gold_text: str | None, others: list[str], values: dict[str, str] | None = None,
                 descs: dict[str, str] | None = None, shuffle: bool = True) -> tuple[dict, str | None, dict]:
    """Build a choice field. Returns (field, gold_key, option_values{key: canonical value}). Texts must be unique."""
    texts = ([gold_text] if gold_text is not None else []) + list(others)
    low = [t.strip().lower() for t in texts]
    if len(set(low)) != len(low):
        raise Skip("duplicate option text")
    order = list(texts)
    if shuffle:
        rng.shuffle(order)
    taken: set[str] = set()
    opts, gold_key, ov = [], None, {}
    for t in order:
        k = option_key(t, taken)
        opts.append({"key": k, "text": t, "description": (descs or {}).get(t, "")})
        if t == gold_text:
            gold_key = k
        if values is not None:
            ov[k] = values[t]
    return {"type": "choice", "question": question, "options": opts}, gold_key, ov


def numeric_choice(rng: random.Random, question: str, gold, wrongs: list, places: int, fmt: Callable[[Decimal], str],
                   n_opts: int = 5, min_gap: int = 2, allow_nonpos: bool = False, rel_jitter=(0.06, 0.35)) -> tuple[dict, str, dict]:
    """Numeric choice: gold + computed mistakes (+ jitter fill). All options differ from gold by >= min_gap display units and from
    each other by >= 1 unit; the exact gold is never on a rounding boundary."""
    if near_half(gold, places):
        raise Skip("gold on rounding boundary")
    unit = Decimal(1).scaleb(-places)
    g = rq(gold, places)
    chosen = [g]

    def ok(v: Decimal) -> bool:
        if abs(v - g) < min_gap * unit:
            return False
        if any(abs(v - c) < unit for c in chosen):
            return False
        if not allow_nonpos and g > 0 and v <= 0:
            return False
        if g != 0 and v != 0 and (v / g > 5 or v / g < Decimal("0.2")) and (g > 0) == (v > 0):
            return False
        return True

    for w in wrongs:
        if w is None:
            continue
        try:
            v = rq(w, places)
        except Exception:
            continue
        if ok(v) and len(chosen) < n_opts:
            chosen.append(v)
    tries = 0
    while len(chosen) < n_opts and tries < 200:
        tries += 1
        r = rng.uniform(*rel_jitter) * rng.choice([-1, 1])
        base = dec(gold)
        v = rq(base * (1 + dec(r)) if base != 0 else dec(r * 10), places)
        if ok(v):
            chosen.append(v)
    if len(chosen) < 3:
        raise Skip("not enough numeric options")
    texts = [fmt(v) for v in chosen]
    vals = {fmt(v): str(v) for v in chosen}
    return choice_field(rng, question, texts[0], texts[1:], values=vals)


def noul_field(question: str) -> dict:
    return {"type": "noul", "question": question}


def score_field(question: str, level_descs: list[str]) -> dict:
    return {"type": "score", "question": question, "levels": [{"value": i, "description": d} for i, d in enumerate(level_descs)]}


def threshold_near(rng: random.Random, gold, places: int, min_rel: float = 0.02, max_rel: float = 0.15) -> tuple[Decimal, bool]:
    """A threshold T for a noul 'is X above T?' question, clearly separated from gold (>= 2 display units and >= min_rel)."""
    g = dec(gold)
    unit = Decimal(1).scaleb(-places)
    for _ in range(50):
        r = dec(rng.uniform(min_rel, max_rel)) * rng.choice([-1, 1])
        t = rq(g * (1 + r) if g != 0 else r * 10, places)
        if abs(t - g) >= 2 * unit and abs(t - g) >= abs(g) * dec(min_rel) * Decimal("0.9"):
            return t, g > t
    raise Skip("no clean threshold")


def flip_witness(rng: random.Random, inp: dict, key: str, answer: Callable[[dict], object], factors=None) -> list[str]:
    """Two values of inp[key] (the original and an alternative) that give different answers; raises Skip if none found."""
    base = answer(inp)
    orig = inp[key]
    for f in (factors or [1.35, 0.7, 1.8, 0.5, 2.5, 0.3, 1.15, 0.88, 4, 0.2]):
        alt = type(orig)(orig * Fraction(f).limit_denominator(100)) if isinstance(orig, Fraction) else orig * f
        if isinstance(orig, int):
            alt = max(1, int(round(orig * f)))
        if alt == orig:
            continue
        inp2 = dict(inp)
        inp2[key] = alt
        try:
            if answer(inp2) != base:
                return [str(orig), str(alt)]
        except (ZeroDivisionError, ValueError, Skip):
            continue
    raise Skip("no flip witness")


def pick_type(rng: random.Random, weights: dict[str, float]) -> str:
    ks = list(weights)
    return rng.choices(ks, weights=[weights[k] for k in ks])[0]


def ceil_div(a: Fraction, b: Fraction) -> int:
    return math.ceil(Fraction(a) / Fraction(b))
