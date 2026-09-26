"""255-option items (family `large_choice`, source A) so the 256-code readout lane has training rows for code 256.

docs/phase-3-plan.md, "Model-side engineering results", item 5: row 256 of the extended readout starts at its LM-head init and
is read only by 255-option questions, so ~1-2k such items are built by construction. Each item is a catalogue of exactly
255 records (products, staff, spare parts, rail services, rental flats); the state is a request that names 3-5 constraints
(categorical equalities and numeric bounds). The gold is the single record that satisfies every constraint:

- unique by construction: the target is drawn first, ~40% of the other 254 records are near misses that break exactly one
  constraint, the rest are random, and any random record that happens to satisfy all constraints is redrawn; the final
  catalogue is re-checked by brute force (exactly one match);
- unknown twins (~15%): the same catalogue with one constraint changed so that NO record matches (brute-force count 0);
  gold null, unknown_reason `not_listed`, parent_id = the answerable item (same split group). With 255 options + unknown the
  question uses all 256 readout codes, which is exactly what the lane needs;
- every 13-token window of a request sentence carries an item-specific value, so fresh-seed items do not near-duplicate
  each other under the decontam/leakage 13-gram rule.

Option key = the record code (e.g. SKU-40213), text = its display name, description = its attributes. Licence `generated`,
gold_kind `constructed`, difficulty 4 (3 constraints) or 5 (4-5 constraints / numeric bounds). Deterministic by seed.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assembly_common import stable_hash  # noqa: E402

N_OPTIONS = 255
GENERATOR = "scripts/p3/assembly_large_choice.py@large-choice-v1"

_SYL = ["ar", "bel", "cor", "dun", "el", "fen", "gar", "hol", "is", "jor", "kel", "lan", "mor", "nes", "or", "pel", "quin",
        "ros", "sal", "tor", "ul", "ven", "wil", "yar", "zen", "bri", "cas", "dor", "fal", "lin", "mar", "tal"]
_FIRST = ["Amara", "Bilal", "Chen", "Dalia", "Emeka", "Farah", "Goran", "Hana", "Ibrahim", "Jonas", "Keiko", "Lucia", "Mateo",
          "Nadia", "Omar", "Priya", "Quentin", "Rosa", "Sanjay", "Tove", "Umar", "Vera", "Wen", "Ximena", "Yusuf", "Zofia",
          "Arjun", "Beatriz", "Cormac", "Dilnoza", "Eero", "Fatima", "Gideon", "Hyejin", "Ines", "Jakub"]
_LAST = ["Abara", "Brennan", "Castellanos", "Dimitrov", "Eklund", "Fonseca", "Grewal", "Haddad", "Ivanova", "Jaramillo", "Kowalczyk",
         "Lindqvist", "Mbeki", "Nakamura", "Okafor", "Petrovic", "Quispe", "Rahman", "Sorensen", "Tanaka", "Uribe", "Varga",
         "Whitlock", "Xu", "Yilmaz", "Zielinski", "Achebe", "Bergstrom", "Cardoso", "Delacroix", "Esposito", "Fitzgerald"]


def _word(rng: random.Random, n: int = 2) -> str:
    return "".join(rng.choice(_SYL) for _ in range(n)).capitalize()


# A domain: attributes (name, kind, values-or-range, unit), how records are named/coded, and request phrasing.
# kind "cat": categorical equality; "num": integer/decimal with a bound in the request ("at most" / "at least").
DOMAINS = {
    "product": {
        "code": ("SKU-", 5),
        "attrs": [("category", "cat", ["backpack", "desk lamp", "kettle", "rain jacket", "office chair", "blender", "hiking boot",
                                       "throw blanket", "water bottle", "bookshelf", "headphones", "cutting board"]),
                  ("colour", "cat", ["black", "navy", "olive", "sand", "burgundy", "slate grey", "cream", "teal", "mustard",
                                     "charcoal"]),
                  ("material", "cat", ["recycled polyester", "oak", "stainless steel", "bamboo", "wool", "aluminium", "cotton canvas",
                                       "ceramic"]),
                  ("size", "cat", ["XS", "S", "M", "L", "XL", "one size"]),
                  ("price", "num", (12, 240), "GBP"),
                  ("warranty", "num", (1, 5), "years")],
        "name": lambda rng, rec: f"{rec['brand']} {rec['category'].title()} {_word(rng, 1)}{rng.randint(2, 99)}",
        "extra": lambda rng: {"brand": _word(rng)},
        "phrase": {
            "category": ["{who} needs a {v} for order {ref}.", "Item type on ticket {ref}: {v}, nothing else."],
            "colour": ["{who} insists the colour is {v} (see note {ref2}).", "Colour wanted by {who}: {v}."],
            "material": ["It must be made of {v}, {who} says.", "{who} will only accept {v} as the material."],
            "size": ["Size {v} is required for {who}.", "{who} asked for size {v} on {ref2}."],
            "price": ["{who} can pay at most {v} GBP for it.", "Budget cap from {who}: {v} GBP or less."],
            "price>": ["{who} wants a premium one costing at least {v} GBP.", "Nothing under {v} GBP, per {who}."],
            "warranty": ["A warranty of no more than {v} years is fine for {who}.", "{who} accepts at most {v} years of warranty."],
            "warranty>": ["{who} needs at least {v} years of warranty.", "Warranty of {v} years or longer, says {who}."],
        },
        "question": ["Which catalogue entry matches the customer's request?", "Which SKU should be shipped for this request?",
                     "Which product in the catalogue is the one being asked for?"],
        "setting": "Customer request (catalogue of {n} products listed as options)",
    },
    "staff": {
        "code": ("EMP-", 4),
        "attrs": [("department", "cat", ["Finance", "Logistics", "Legal", "Research", "Customer Care", "Procurement", "Facilities",
                                         "Data Platform", "Security", "Training"]),
                  ("office", "cat", ["Lisbon", "Nairobi", "Osaka", "Calgary", "Tallinn", "Recife", "Leeds", "Pune", "Perth"]),
                  ("role", "cat", ["analyst", "team lead", "coordinator", "engineer", "specialist", "manager", "assistant"]),
                  ("language", "cat", ["Portuguese", "Swahili", "Japanese", "French", "Estonian", "Hindi", "Arabic", "Polish"]),
                  ("start_year", "num", (2008, 2025), ""),
                  ("clearance", "num", (1, 5), "")],
        "name": lambda rng, rec: f"{rec['first']} {rec['last']}",
        "extra": lambda rng: {"first": rng.choice(_FIRST), "last": rng.choice(_LAST)},
        "phrase": {
            "department": ["The contact {who} needs sits in {v} (case {ref}).", "Department for case {ref}: {v}."],
            "office": ["They are based in the {v} office, per {who}.", "{who} says the person works out of {v}."],
            "role": ["Their role is {v}, according to {who}.", "{who} remembers the job title as {v}."],
            "language": ["They speak {v}, which {who} needs for {ref2}.", "{who} requires a {v} speaker."],
            "start_year": ["{who} says they joined no later than {v}.", "Start year {v} or earlier, per {who}."],
            "start_year>": ["{who} says they joined in {v} or later.", "Joined no earlier than {v}, notes {who}."],
            "clearance": ["Clearance level at most {v} is enough for {who}.", "{who} notes a clearance of {v} or lower."],
            "clearance>": ["{who} needs clearance level {v} or higher.", "Clearance at least {v}, per {who}."],
        },
        "question": ["Which employee is the person described?", "Which staff record matches the description?",
                     "Who in the directory should the request be routed to?"],
        "setting": "Directory lookup (a staff directory of {n} people listed as options)",
    },
    "part": {
        "code": ("PRT-", 5),
        "attrs": [("part_type", "cat", ["hex bolt", "wing nut", "flange bearing", "hose clamp", "spring washer", "dowel pin",
                                        "rivet", "cable gland", "set screw", "O-ring"]),
                  ("thread", "cat", ["M4", "M5", "M6", "M8", "M10", "M12", "none"]),
                  ("material", "cat", ["brass", "nylon", "zinc-plated steel", "A4 stainless", "titanium", "bronze"]),
                  ("finish", "cat", ["black oxide", "plain", "passivated", "galvanised", "anodised"]),
                  ("length", "num", (6, 120), "mm"),
                  ("pack_size", "num", (10, 500), "pieces")],
        "name": lambda rng, rec: f"{rec['supplier']} {rec['part_type']} {rng.choice('ABCDEFGHJK')}{rng.randint(10, 99)}",
        "extra": lambda rng: {"supplier": _word(rng) + " Industrial"},
        "phrase": {
            "part_type": ["Work order {ref} calls for a {v}.", "{who} is replacing a {v} on line {ref2}."],
            "thread": ["Thread must be {v} according to {who}.", "{who} measured the thread as {v}."],
            "material": ["Material: {v}, as specified by {who}.", "{who} needs it in {v}."],
            "finish": ["Finish {v} is mandatory for {ref2}.", "{who} asks for a {v} finish."],
            "length": ["Length no more than {v} mm, says {who}.", "{who} has room for at most {v} mm."],
            "length>": ["It must be at least {v} mm long for {who}.", "{who} needs {v} mm or longer."],
            "pack_size": ["{who} wants packs of at most {v} pieces.", "No pack bigger than {v} pieces for {ref2}."],
            "pack_size>": ["{who} needs packs of {v} pieces or more.", "Pack size at least {v}, per {who}."],
        },
        "question": ["Which part number fits the work order?", "Which catalogue part should be picked?",
                     "Which spare part matches every requirement in the order?"],
        "setting": "Spare-part order (a parts catalogue of {n} items listed as options)",
    },
    "rail": {
        "code": ("SVC-", 4),
        "attrs": [("operator", "cat", ["Northline", "Coastway", "Valley Express", "Metroflyer", "Highland Rail", "Crossrail South",
                                       "Lakeside Trains"]),
                  ("destination", "cat", ["Harrowgate", "Eastmere", "Kingsbridge", "Portwell", "Ashford Vale", "Millbrook",
                                          "Stonehaven", "Riverton"]),
                  ("fare_class", "cat", ["standard", "first", "saver", "flexi"]),
                  ("departure_hour", "num", (5, 22), ":00"),
                  ("stops", "num", (0, 9), "stops"),
                  ("fare", "num", (9, 180), "GBP")],
        "name": lambda rng, rec: f"{rec['operator']} {rng.randint(100, 999)}",
        "extra": lambda rng: {},
        "phrase": {
            "operator": ["{who} holds a {v} railcard, so it must be a {v} service.", "Operator {v} only (booking {ref})."],
            "destination": ["{who} is travelling to {v} for meeting {ref}.", "Destination on booking {ref}: {v}."],
            "fare_class": ["The ticket must be {v} class for {who}.", "{who} is entitled to a {v} fare."],
            "departure_hour": ["It must leave no later than {v}:00, says {who}.", "{who} has to depart by {v}:00."],
            "departure_hour>": ["{who} cannot leave before {v}:00.", "Departure at {v}:00 or later, per {who}."],
            "stops": ["At most {v} intermediate stops, {who} insists.", "{who} accepts no more than {v} stops."],
            "stops>": ["{who} needs a stopping service with at least {v} stops.", "It must call at {v} or more stops, says {who}."],
            "fare": ["{who} has a travel budget of at most {v} GBP.", "Fare cap of {v} GBP on booking {ref}."],
            "fare>": ["Policy for {who} rules out fares below {v} GBP.", "{who} must book a fare of at least {v} GBP."],
        },
        "question": ["Which service should be booked?", "Which train meets every constraint of the booking?",
                     "Which departure fits the travel request?"],
        "setting": "Travel booking (a timetable of {n} services listed as options)",
    },
    "flat": {
        "code": ("LST-", 5),
        "attrs": [("district", "cat", ["Old Harbour", "Hillcrest", "Canal Quarter", "Northgate", "Southbank", "Elm Park",
                                       "Mill Lane", "Castle View", "Greenway"]),
                  ("heating", "cat", ["heat pump", "gas boiler", "district heating", "electric panels"]),
                  ("pets", "cat", ["pets allowed", "no pets", "cats only"]),
                  ("bedrooms", "num", (1, 5), "bedrooms"),
                  ("floor", "num", (0, 14), ""),
                  ("rent", "num", (650, 3200), "EUR")],
        "name": lambda rng, rec: f"{_word(rng)} {rng.choice(['Court', 'House', 'Row', 'Yard', 'Place', 'Rise'])} {rng.randint(1, 80)}",
        "extra": lambda rng: {},
        "phrase": {
            "district": ["{who} wants to live in {v} (enquiry {ref}).", "Area requested on enquiry {ref}: {v}."],
            "heating": ["{who} needs {v} heating.", "Heating must be {v} for {who}."],
            "pets": ["The listing must say {v}, since {who} asked.", "{who} requires a flat marked {v}."],
            "bedrooms": ["No more than {v} bedrooms for {who}.", "{who} wants at most {v} bedrooms."],
            "bedrooms>": ["{who} needs at least {v} bedrooms.", "{v} bedrooms or more, per {who}."],
            "floor": ["{who} cannot be above floor {v}.", "Floor {v} or lower for {who}."],
            "floor>": ["{who} wants floor {v} or higher.", "Nothing below floor {v}, says {who}."],
            "rent": ["Monthly rent of at most {v} EUR for {who}.", "{who} can pay no more than {v} EUR a month."],
            "rent>": ["{who} is looking at rents of {v} EUR or more.", "Rent at least {v} EUR, per {who}."],
        },
        "question": ["Which listing matches the enquiry?", "Which flat should the agent offer?",
                     "Which rental listing satisfies every requirement?"],
        "setting": "Rental enquiry (a list of {n} listings given as options)",
    },
}
DISTRACTOR_SENTENCES = [
    "{who} also mentioned that the invoice address changed last month.", "Delivery can go to reception at desk {ref2}.",
    "{who} previously asked about loyalty points on account {ref}.", "Please copy {who} on the confirmation for {ref2}.",
    "The previous request {ref2} from {who} was closed without action.",
]


def _attr_value(rng: random.Random, spec) -> object:
    name, kind, vals = spec[0], spec[1], spec[2]
    if kind == "cat":
        return rng.choice(vals)
    lo, hi = vals
    return rng.randint(lo, hi)


def _satisfies(rec: dict, cons: list[tuple]) -> bool:
    for attr, op, v in cons:
        x = rec[attr]
        if op == "==" and x != v:
            return False
        if op == "<=" and not x <= v:
            return False
        if op == ">=" and not x >= v:
            return False
    return True


def _describe(dom: dict, rec: dict) -> str:
    parts = []
    for spec in dom["attrs"]:
        a, kind = spec[0], spec[1]
        v = rec[a]
        unit = spec[3] if kind == "num" else ""
        label = a.replace("_", " ")
        if a == "departure_hour":
            parts.append(f"departs {v:02d}:00")
        elif kind == "num":
            parts.append(f"{label} {v}{(' ' + unit) if unit and not unit.startswith(':') and unit != label else ''}")
        else:
            parts.append(f"{label} {v}")
    return "; ".join(parts)


def _violate(rng: random.Random, dom: dict, rec: dict, con: tuple) -> dict:
    """A copy of rec that breaks exactly this constraint (other attributes kept)."""
    attr, op, v = con
    spec = next(s for s in dom["attrs"] if s[0] == attr)
    out = dict(rec)
    if op == "==":
        out[attr] = rng.choice([x for x in spec[2] if x != v])
    else:
        lo, hi = spec[2]
        if op == "<=" and v < hi:
            out[attr] = rng.randint(v + 1, hi)
        elif op == ">=" and v > lo:
            out[attr] = rng.randint(lo, v - 1)
        else:
            return None
    return out


def _pick_constraints(rng: random.Random, dom: dict, target: dict, k: int) -> list[tuple]:
    specs = rng.sample(dom["attrs"], k)
    cons = []
    for spec in specs:
        a, kind = spec[0], spec[1]
        if kind == "cat":
            cons.append((a, "==", target[a]))
        else:
            lo, hi = spec[2]
            x = target[a]
            if x < hi and (x == lo or rng.random() < 0.5):          # "at most b" with b >= x
                cons.append((a, "<=", rng.randint(x, min(hi - 1, x + max(1, (hi - lo) // 6)))))
            else:                                                     # "at least b" with b <= x
                cons.append((a, ">=", rng.randint(max(lo + 1, x - max(1, (hi - lo) // 6)), x)))
    return cons


def _sentence(rng: random.Random, dom: dict, con: tuple, ctx: dict) -> str:
    attr, op, v = con
    key = attr + (">" if op == ">=" else "")
    return rng.choice(dom["phrase"][key]).format(v=v, **ctx)


def _catalogue(rng: random.Random, dom: dict, target: dict, cons: list[tuple]) -> list[dict] | None:
    seen = {_describe(dom, target)}
    recs = [target]
    tries = 0
    while len(recs) < N_OPTIONS and tries < 20000:
        tries += 1
        if rng.random() < 0.4:
            base = dict(target)
            for spec in dom["attrs"]:                     # vary the unconstrained attributes
                if spec[0] not in {c[0] for c in cons} and rng.random() < 0.7:
                    base[spec[0]] = _attr_value(rng, spec)
            rec = _violate(rng, dom, base, rng.choice(cons))
            if rec is None:
                continue
        else:
            rec = {s[0]: _attr_value(rng, s) for s in dom["attrs"]}
        if _satisfies(rec, cons):
            continue
        d = _describe(dom, rec)
        if d in seen:
            continue
        seen.add(d)
        rec.update(dom["extra"](rng))
        recs.append(rec)
    return recs if len(recs) == N_OPTIONS else None


def _codes(rng: random.Random, prefix: str, digits: int, n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:                 # a list, not set iteration order: identical across processes (PYTHONHASHSEED)
        c = f"{prefix}{rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_item(seed: str, index: int, domain: str | None = None, id_prefix: str = "p3-lc") -> list[dict]:
    """One answerable 255-option item (+ its unknown twin when the item is drawn as a twin parent). Returns 1-2 rows."""
    rng = random.Random(stable_hash("large_choice", seed, index))
    domain = domain or rng.choice(sorted(DOMAINS))
    dom = DOMAINS[domain]
    for _ in range(50):
        target = {s[0]: _attr_value(rng, s) for s in dom["attrs"]}
        target.update(dom["extra"](rng))
        k = rng.choice([3, 3, 4, 4, 5])
        cons = _pick_constraints(rng, dom, target, k)
        recs = _catalogue(rng, dom, target, cons)
        if recs is not None and sum(_satisfies(r, cons) for r in recs) == 1:
            break
    else:
        raise RuntimeError(f"could not build a unique catalogue for {seed}/{index}")
    rng.shuffle(recs)
    codes = _codes(rng, dom["code"][0], dom["code"][1], N_OPTIONS)
    options, names = [], set()
    gold = None
    for rec, code in zip(recs, codes):
        name = dom["name"](rng, rec)
        while name in names:
            name = name + " " + rng.choice("ABCDEFGH")
        names.add(name)
        options.append({"key": code, "text": name, "description": _describe(dom, rec)})
        if rec is target:
            gold = code
    ctx = {"who": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}", "ref": f"{rng.choice('ABCDEFGHKMNPRT')}-{rng.randint(10000, 99999)}",
           "ref2": f"{rng.choice('QRSTUVWXYZ')}{rng.randint(100, 999)}"}
    sentences = [_sentence(rng, dom, c, ctx) for c in cons]
    n_distr = rng.choice([0, 1, 1, 2])
    for _ in range(n_distr):
        sentences.insert(rng.randint(0, len(sentences)), rng.choice(DISTRACTOR_SENTENCES).format(**ctx))
    numeric = any(op != "==" for _, op, _ in cons)
    difficulty = 5 if (k >= 5 or (k == 4 and numeric)) else 4
    question = rng.choice(dom["question"])
    rid = f"{id_prefix}-{domain}-{seed}-{index:05d}"
    base = {
        "id": rid, "source": "A", "dataset": "large_choice", "family": "large_choice", "difficulty": difficulty,
        "state": {"setting": dom["setting"].format(n=N_OPTIONS), "request": " ".join(sentences)},
        "images": [], "field": {"type": "choice", "question": question, "options": options},
        "gold": gold, "unknown_reason": None, "gold_kind": "constructed", "parent_id": None,
        "provenance": {"licence": "generated", "generator": GENERATOR, "seed": seed, "index": index, "domain": domain,
                       "n_options": N_OPTIONS, "constraints": [list(c) for c in cons], "readout_codes_needed": 256},
    }
    rows = [base]
    # unknown twin: change one constraint so that no record satisfies all of them (its sentence is rewritten in place)
    if rng.random() < 0.18:
        where = _sentence_map(sentences, cons, dom, ctx)
        order = [ci for ci in range(len(cons)) if ci in where]
        rng.shuffle(order)
        for ci in order:
            attr, op, v = cons[ci]
            spec = next(s for s in dom["attrs"] if s[0] == attr)
            alts = [x for x in spec[2] if x != v] if op == "==" else [x for x in range(spec[2][0], spec[2][1] + 1) if x != v]
            rng.shuffle(alts)
            alt = next((a for a in alts if not any(_satisfies(r, cons[:ci] + [(attr, op, a)] + cons[ci + 1:]) for r in recs)), None)
            if alt is None:
                continue
            new = cons[:ci] + [(attr, op, alt)] + cons[ci + 1:]
            twin_sentences = list(sentences)
            twin_sentences[where[ci]] = _sentence(rng, dom, new[ci], ctx)
            rows.append({**base, "id": rid + "-u", "gold": None, "unknown_reason": "not_listed", "parent_id": rid, "difficulty": 5,
                         "state": {"setting": base["state"]["setting"], "request": " ".join(twin_sentences)},
                         "provenance": {**base["provenance"], "constraints": [list(c) for c in new], "twin_changed_constraint": ci}})
            break
    return rows


def _sentence_map(sentences: list[str], cons: list[tuple], dom: dict, ctx: dict) -> dict[int, int]:
    """constraint index -> sentence index (the sentence that mentions its value in that constraint's phrasing)."""
    out = {}
    for ci, (attr, op, v) in enumerate(cons):
        key = attr + (">" if op == ">=" else "")
        cands = {t.format(v=v, **ctx) for t in dom["phrase"][key]}
        for si, s in enumerate(sentences):
            if s in cands and si not in out.values():
                out[ci] = si
                break
    return out


def generate(seed: str, count: int, id_prefix: str = "p3-lc") -> list[dict]:
    """Exactly `count` rows (answerable items + ~15% unknown twins), balanced over the five domains. A twin always follows its
    parent, so cutting at `count` can drop a twin but never orphan one."""
    rows: list[dict] = []
    doms = sorted(DOMAINS)
    i = 0
    while len(rows) < count:
        rows += make_item(seed, i, doms[i % len(doms)], id_prefix)
        i += 1
    return rows[:count]


def recheck(row: dict) -> object:
    """Independent re-solve from the options' descriptions and the stored constraints: the gold key, or None if no match."""
    dom = DOMAINS[row["provenance"]["domain"]]
    cons = [tuple(c) for c in row["provenance"]["constraints"]]
    matches = []
    specs = sorted(dom["attrs"], key=lambda sp: -len(sp[0]))   # "fare class" before "fare"
    for o in row["field"]["options"]:
        rec = {}
        for part in o["description"].split("; "):
            if part.startswith("departs "):
                rec["departure_hour"] = int(part.split()[1].split(":")[0])
                continue
            for spec in specs:
                label = spec[0].replace("_", " ")
                if part.startswith(label + " "):
                    val = part[len(label) + 1:]
                    if spec[1] == "num":
                        val = int(val.split()[0])
                    rec[spec[0]] = val
                    break
        if _satisfies(rec, cons):
            matches.append(o["key"])
    if len(matches) > 1:
        raise AssertionError(f"{row['id']}: {len(matches)} matches")
    return matches[0] if matches else None
