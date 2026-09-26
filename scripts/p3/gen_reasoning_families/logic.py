"""Family logic: policy-style premises with quantifiers ("every X who..."), negation, exceptions ("unless"), inclusive
disjunctions and distractor premises; depth 2-6. Questions are noul ("is it true that ...") or choice ("which statement must
be true" / "which of these entities is certain to ..."). Undetermined questions are gold=null (insufficient_evidence).

Formal model: unary predicates over named entities. A rule is (ifs, unless, then) over literals (pred, polarity) and holds
for every entity: if all `ifs` hold and no `unless` literal holds, `then` holds. Facts are unit literals or inclusive
disjunctions of two literals about one entity. Entailment is decided by a small DPLL over the grounded clauses of the
queried entity (the test re-checks by brute-force enumeration)."""
from __future__ import annotations

import random

from .common import DOMAINS, Names, Skip, T, assemble, choice_field, noul_field

FAM = "logic"

CLASSES = {
    "employee": dict(n=("staff member", "staff members"), rel="who", dom="hr", ent="first", preds=[
        ("holds a forklift licence", "does not hold a forklift licence"), ("works night shifts", "does not work night shifts"),
        ("is eligible for the retention bonus", "is not eligible for the retention bonus"),
        ("has completed the safety induction", "has not completed the safety induction"),
        ("is on a fixed-term contract", "is not on a fixed-term contract"), ("must attend the compliance workshop", "is exempt from the compliance workshop"),
        ("can approve purchase orders", "cannot approve purchase orders"), ("is based at the northern depot", "is not based at the northern depot"),
        ("receives the travel allowance", "does not receive the travel allowance"), ("is part of the on-call rota", "is not part of the on-call rota"),
        ("has access to the payroll system", "has no access to the payroll system"),
        ("needs a second-level background check", "does not need a second-level background check"),
        ("mentors a graduate trainee", "does not mentor a graduate trainee"), ("qualifies for remote working", "does not qualify for remote working"),
        ("sits on the health and safety committee", "does not sit on the health and safety committee"),
        ("is covered by the enhanced insurance plan", "is not covered by the enhanced insurance plan")]),
    "supplier": dict(n=("supplier", "suppliers"), rel="that", dom="finance", ent="company", preds=[
        ("is ISO 9001 certified", "is not ISO 9001 certified"), ("is on the preferred-vendor list", "is not on the preferred-vendor list"),
        ("requires an on-site audit", "does not require an on-site audit"), ("ships from outside the customs union", "ships from within the customs union"),
        ("qualifies for 60-day payment terms", "does not qualify for 60-day payment terms"), ("must provide a carbon report", "is exempt from the carbon report"),
        ("has a single-source contract", "does not have a single-source contract"),
        ("is flagged for enhanced due diligence", "is not flagged for enhanced due diligence"), ("handles personal data", "does not handle personal data"),
        ("must sign the data-processing addendum", "does not need to sign the data-processing addendum"),
        ("is paid in foreign currency", "is paid in the home currency"), ("has passed the financial-health check", "has not passed the financial-health check"),
        ("can bid on framework tenders", "cannot bid on framework tenders"), ("holds cyber-insurance", "does not hold cyber-insurance")]),
    "shipment": dict(n=("shipment", "shipments"), rel="that", dom="logistics", ent="code:SHP", preds=[
        ("contains lithium batteries", "does not contain lithium batteries"), ("needs a dangerous-goods declaration", "does not need a dangerous-goods declaration"),
        ("travels by air", "does not travel by air"), ("is temperature controlled", "is not temperature controlled"),
        ("requires a customs broker", "does not require a customs broker"), ("is insured at full value", "is not insured at full value"),
        ("goes through the express lane", "does not go through the express lane"), ("must be signed for on delivery", "can be left without a signature"),
        ("is held for inspection", "is not held for inspection"), ("exceeds 30 kg", "does not exceed 30 kg"),
        ("is bound for a remote postcode", "is not bound for a remote postcode"), ("qualifies for next-day delivery", "does not qualify for next-day delivery"),
        ("carries a fragile label", "does not carry a fragile label")]),
    "account": dict(n=("customer account", "customer accounts"), rel="that", dom="saas_ops", ent="company", preds=[
        ("is on the enterprise tier", "is not on the enterprise tier"), ("has single sign-on enabled", "does not have single sign-on enabled"),
        ("gets a dedicated success manager", "does not get a dedicated success manager"), ("is billed annually", "is not billed annually"),
        ("qualifies for the loyalty discount", "does not qualify for the loyalty discount"), ("stores data in the EU region", "does not store data in the EU region"),
        ("must complete a security review", "is exempt from the security review"), ("has an overdue invoice", "has no overdue invoice"),
        ("is eligible for the beta programme", "is not eligible for the beta programme"), ("is in the renewal window", "is not in the renewal window"),
        ("receives priority support", "does not receive priority support"), ("is flagged as a churn risk", "is not flagged as a churn risk"),
        ("uses the legacy API", "does not use the legacy API")]),
    "student": dict(n=("student", "students"), rel="who", dom="education", ent="first", preds=[
        ("is enrolled full time", "is not enrolled full time"), ("receives the travel bursary", "does not receive the travel bursary"),
        ("must attend the study-skills module", "is exempt from the study-skills module"), ("has submitted the placement form", "has not submitted the placement form"),
        ("can register for the honours track", "cannot register for the honours track"), ("lives in university housing", "does not live in university housing"),
        ("is on academic probation", "is not on academic probation"), ("sits the resit exam", "does not sit the resit exam"),
        ("qualifies for extra exam time", "does not qualify for extra exam time"), ("is a member of the student council", "is not a member of the student council"),
        ("has a library fine outstanding", "has no library fine outstanding"), ("takes the evening timetable", "does not take the evening timetable")]),
    "claim": dict(n=("claim", "claims"), rel="that", dom="healthcare_admin", ent="code:CLM", preds=[
        ("involves a hospital stay", "does not involve a hospital stay"), ("needs pre-authorisation", "does not need pre-authorisation"),
        ("is routed to the senior assessor", "is not routed to the senior assessor"), ("exceeds the fast-track limit", "does not exceed the fast-track limit"),
        ("is paid within ten days", "is not paid within ten days"), ("was filed after the deadline", "was filed on time"),
        ("requires a medical report", "does not require a medical report"), ("is flagged for fraud review", "is not flagged for fraud review"),
        ("is covered under the dental rider", "is not covered under the dental rider"),
        ("comes from an out-of-network provider", "comes from an in-network provider"),
        ("qualifies for direct settlement", "does not qualify for direct settlement"), ("is subject to co-payment", "is not subject to co-payment")]),
}


# ------------------------------------------------------------------------------------------------ solver (DPLL)
def _sat(clauses: list[frozenset]) -> bool:
    assign: dict = {}
    return _dpll([set(c) for c in clauses], assign)


def _dpll(clauses, assign) -> bool:
    clauses = [set(c) for c in clauses]
    while True:
        new = []
        unit = None
        for c in clauses:
            if any((v, b) in c for v, b in assign.items()):
                continue
            rest = {(v, b) for v, b in c if v not in assign}
            if not rest:
                return False
            if len(rest) == 1:
                unit = next(iter(rest))
            new.append(rest)
        clauses = new
        if unit is None:
            break
        assign = dict(assign)
        assign[unit[0]] = unit[1]
    if not clauses:
        return True
    v = next(iter(clauses[0]))[0]
    for b in (True, False):
        a2 = dict(assign)
        a2[v] = b
        if _dpll(clauses, a2):
            return True
    return False


def grounded(rules: list[dict], facts: list[dict], ent: str) -> list[frozenset]:
    cl = []
    for r in rules:
        c = {(p, not b) for p, b in r["if"]} | {(p, b) for p, b in r.get("unless", [])} | {(r["then"][0], r["then"][1])}
        cl.append(frozenset(c))
    for f in facts:
        if f["ent"] != ent:
            continue
        if "lit" in f:
            cl.append(frozenset({tuple(f["lit"])}))
        else:
            cl.append(frozenset(tuple(x) for x in f["or"]))
    return cl


def status(rules, facts, ent, lit) -> bool | None:
    """True if lit is entailed for ent, False if its negation is, None if undetermined. Raises Skip if inconsistent."""
    cl = grounded(rules, facts, ent)
    if not _sat(cl):
        raise Skip("inconsistent theory")
    p, b = lit
    if not _sat(cl + [frozenset({(p, not b)})]):
        return True
    if not _sat(cl + [frozenset({(p, b)})]):
        return False
    return None


# ------------------------------------------------------------------------------------------------ rendering
def vp(cls, lit):
    pos, neg = cls["preds"][lit[0]]
    return pos if lit[1] else neg


def rule_text(r: random.Random, cls: dict, rule: dict) -> str:
    n, rel = cls["n"][0], cls["rel"]
    cond = " and ".join(vp(cls, l) for l in rule["if"])
    then = rule["then"]
    unless = rule.get("unless", [])
    forms = [f"Every {n} {rel} {cond} {vp(cls, then)}", f"Any {n} {rel} {cond} {vp(cls, then)}",
             f"If a {n} {cond}, then that {n} {vp(cls, then)}", f"Whenever a {n} {cond}, that {n} {vp(cls, then)}"]
    if not then[1]:
        forms.append(f"No {n} {rel} {cond} {vp(cls, (then[0], True))}")
    s = r.choice(forms)
    if unless:
        u = " or ".join(vp(cls, l) for l in unless)
        s += r.choice([f", unless that {n} {u}", f", except where the {n} {u}", f" (this does not apply to any {n} {rel} {u})"])
    return s + "."


def fact_text(r, cls, f) -> str:
    if "lit" in f:
        return f"{f['ent']} {vp(cls, f['lit'])}."
    a, b = f["or"]
    return r.choice([f"{f['ent']} either {vp(cls, a)} or {vp(cls, b)} (possibly both).",
                     f"The file shows that {f['ent']} {vp(cls, a)}, or {vp(cls, b)}, or both; it does not say which."])


def render(r: random.Random, cls: dict, rules, facts, d, names, org) -> str:
    dom = DOMAINS[cls["dom"]]
    rules = list(rules)
    r.shuffle(rules)
    facts = list(facts)
    r.shuffle(facts)
    style = r.randrange(3)
    rtxt = [rule_text(r, cls, x) for x in rules]
    ftxt = [fact_text(r, cls, x) for x in facts]
    if style == 0:
        rb = "Rules in force:\n" + "\n".join(f"{k + 1}. {t}" for k, t in enumerate(rtxt))
        fb = "Facts on file:\n" + "\n".join(f"- {t}" for t in ftxt)
        body = [rb, fb]
    elif style == 1:
        rb = T(r, "{The|Our} {policy|handbook|rulebook} {states|sets out} the following. ") + " ".join(rtxt)
        fb = T(r, "{From the records|According to the case notes|The register shows the following}: ") + " ".join(ftxt)
        body = [rb, fb]
    else:
        half = len(rtxt) // 2
        body = ["Section A. " + " ".join(rtxt[:half]), "Records. " + " ".join(ftxt), "Section B. " + " ".join(rtxt[half:])]
    head = [T(r, "{Policy extract|Eligibility rules|Rulebook excerpt|Decision rules} - $o", o=org),
            T(r, "{The rules below apply to every $n. |Each rule applies to every $n without exception, except where the rule itself says otherwise. |}"
                 "{Nothing beyond these rules and records may be assumed.|Treat the rules and records as complete for what they state, but do not assume "
                 "anything they do not state.|Only what is written here counts.}", n=cls["n"][0])]
    return assemble(r, head, body, dom, Names(r, names.used), d)


# ------------------------------------------------------------------------------------------------ generator
def _entity(names: Names, cls: dict) -> str:
    e = cls["ent"]
    if e == "first":
        return names.person()
    if e == "company":
        return names.company()
    return names.code(e.split(":")[1])


def build_chain(rng, cls, d, ent, used_preds):
    """Chain of literals l0..lD for `ent`, with rules and facts. Returns (rules, facts, chain, decisive_premises)."""
    npred = len(cls["preds"])
    depth = {3: rng.randint(2, 3), 4: rng.randint(3, 4), 5: rng.randint(5, 6)}[d]
    avail = [p for p in range(npred) if p not in used_preds]
    if len(avail) < depth + 4:
        raise Skip("few preds")
    ps = rng.sample(avail, depth + 1)
    used_preds.update(ps)
    chain = [(p, rng.random() < 0.65) for p in ps]
    rules, facts, decisive = [], [], []
    for k in range(depth):
        rule = {"if": [chain[k]], "then": chain[k + 1]}
        rules.append(rule)
        decisive.append(("rule", rule))
    # base: plain fact, or (d5) an inclusive disjunction resolved by two rules
    if d == 5 and rng.random() < 0.6:
        a, b = [p for p in range(npred) if p not in used_preds][:2]
        used_preds.update([a, b])
        la, lb = (a, rng.random() < 0.6), (b, rng.random() < 0.6)
        f = {"ent": ent, "or": [la, lb]}
        facts.append(f)
        rules += [{"if": [la], "then": chain[0]}, {"if": [lb], "then": chain[0]}]
        decisive.append(("fact", f))
        decisive.append(("rule", rules[-1]))
    else:
        f = {"ent": ent, "lit": chain[0]}
        facts.append(f)
        decisive.append(("fact", f))
    # exception on one link
    if d >= 4:
        k = rng.randrange(depth)
        ex = [p for p in range(npred) if p not in used_preds][0]
        used_preds.add(ex)
        exl = (ex, rng.random() < 0.5)
        rules[k]["unless"] = [exl]
        if d == 5 and rng.random() < 0.5:
            src = [p for p in range(npred) if p not in used_preds][0]
            used_preds.add(src)
            srcl = (src, rng.random() < 0.6)
            rules.append({"if": [srcl], "then": (ex, not exl[1])})
            f = {"ent": ent, "lit": srcl}
            facts.append(f)
            decisive.append(("fact", f))
            decisive.append(("rule", rules[-1]))
        else:
            f = {"ent": ent, "lit": (ex, not exl[1])}
            facts.append(f)
            decisive.append(("fact", f))
    return rules, facts, chain, decisive


def add_distractors(rng, cls, d, rules, facts, chain, ent, used_preds, others):
    npred = len(cls["preds"])
    free = [p for p in range(npred) if p not in used_preds]
    rng.shuffle(free)
    nd = {3: 2, 4: 3, 5: 5}[d]
    for _ in range(nd):
        kind = rng.choice(["converse", "deny", "other_cond", "other_fact"])
        if kind == "converse" and free:
            k = rng.randrange(1, len(chain))
            x = (free.pop(), rng.random() < 0.5)
            rules.append({"if": [x], "then": chain[k]})  # affirming-the-consequent bait: x -> l_k (x unknown for ent)
        elif kind == "deny" and free:
            k = rng.randrange(0, len(chain) - 1)
            x = (free.pop(), rng.random() < 0.5)
            rules.append({"if": [(chain[k][0], not chain[k][1])], "then": x})  # denying-the-antecedent bait
        elif kind == "other_cond" and free:
            x = (free.pop(), rng.random() < 0.5)
            rules.append({"if": [chain[-1], x] if rng.random() < 0.5 else [x], "then": (rng.choice(free) if free else chain[0][0], rng.random() < 0.5)})
        elif others:
            o = rng.choice(others)
            facts.append({"ent": o, "lit": (rng.choice(list(used_preds)), rng.random() < 0.5)})


def _q_noul(r, ent, cls, lit_pos):
    return T(r, "{Based only on the rules and facts above|According to the rules and records|Going only by what is written}, is it true that $e $v?",
             e=ent, v=vp(cls, (lit_pos, True)))


def logic_noul(rng: random.Random, d: int) -> dict:
    cname = rng.choice(sorted(CLASSES))
    cls = CLASSES[cname]
    names = Names(rng)
    org = names.company()
    ent = _entity(names, cls)
    others = [_entity(names, cls) for _ in range(rng.randint(1, 3))]
    used = set()
    rules, facts, chain, decisive = build_chain(rng, cls, d, ent, used)
    add_distractors(rng, cls, d, rules, facts, chain, ent, used, others)
    target = chain[-1] if rng.random() < 0.8 else chain[rng.randrange(2, len(chain))]
    natural_unknown = rng.random() < 0.12
    if natural_unknown:
        # break the chain before the target: drop one decisive premise (the item is undetermined from the start)
        kind, prem = rng.choice(decisive)
        (rules if kind == "rule" else facts).remove(prem)
    st = status(rules, facts, ent, (target[0], True))
    if natural_unknown:
        if st is not None:
            raise Skip("still determined")
        gold, reason = None, "insufficient_evidence"
    else:
        if st is None or st != target[1]:
            raise Skip("not entailed as built")
        gold, reason = st, None
    rseed = rng.random()
    state = render(random.Random(rseed), cls, rules, facts, d, names, org)
    field = noul_field(_q_noul(rng, ent, cls, target[0]))
    spec = {"kind": "logic_noul", "class": cname, "rules": rules, "facts": facts, "ent": ent, "query": [target[0], True]}
    unk = None
    if gold is not None:
        order = list(decisive)
        rng.shuffle(order)
        for kind, prem in order:
            r2 = [x for x in rules if x is not prem]
            f2 = [x for x in facts if x is not prem]
            try:
                if status(r2, f2, ent, (target[0], True)) is None:
                    unk = {"state": render(random.Random(rseed), cls, r2, f2, d, names, org), "reason": "insufficient_evidence",
                           "spec": {**spec, "rules": r2, "facts": f2, "removed": prem}}
                    break
            except Skip:
                continue
    return {"family": FAM, "state": state, "field": field, "gold": gold, "unknown_reason": reason, "spec": spec, "unk": unk}


def logic_statement(rng: random.Random, d: int) -> dict:
    """Which statement about ent must be true? Exactly one option is entailed (or none -> unknown)."""
    cname = rng.choice(sorted(CLASSES))
    cls = CLASSES[cname]
    names = Names(rng)
    org = names.company()
    ent = _entity(names, cls)
    others = [_entity(names, cls) for _ in range(rng.randint(1, 2))]
    used = set()
    rules, facts, chain, decisive = build_chain(rng, cls, d, ent, used)
    add_distractors(rng, cls, d, rules, facts, chain, ent, used, others)
    target = chain[-1] if rng.random() < 0.7 else chain[rng.randrange(2, len(chain))]
    if status(rules, facts, ent, target) is not True:
        raise Skip("not entailed")
    # candidate distractor statements: negated target, undetermined literals over used/unused preds, chain literals negated
    cands = [(target[0], not target[1])]
    for p in range(len(cls["preds"])):
        for b in (True, False):
            if (p, b) == target:
                continue
            try:
                s_ = status(rules, facts, ent, (p, b))
            except Skip:
                continue
            if s_ is not True and (p, b) not in cands:
                cands.append((p, b))
    k = rng.choice([3, 4, 4, 5])
    mentioned = {l[0] for r in rules for l in r["if"] + [r["then"]] + r.get("unless", [])}
    rest = sorted(cands[1:], key=lambda x: (x[0] not in mentioned, rng.random()))
    opts = [cands[0]] if rng.random() < 0.5 else []
    for c in rest:
        if len(opts) >= k - 1:
            break
        if c[0] != target[0] and all(c[0] != o[0] for o in opts):
            opts.append(c)
    if len(opts) < 2:
        raise Skip("few distractors")
    natural_unknown = rng.random() < 0.1
    texts = {f"{ent} {vp(cls, l)}": l for l in [target] + opts}
    rseed = rng.random()
    qtext = T(rng, "{Which statement about $e must be true|Which statement about $e is guaranteed by the rules|What must be true of $e}?", e=ent)
    spec = {"kind": "logic_statement", "class": cname, "ent": ent}
    if natural_unknown:
        kind, prem = rng.choice(decisive)
        (rules if kind == "rule" else facts).remove(prem)
        if any(status(rules, facts, ent, l) is True for l in texts.values()):
            raise Skip("still determined")
        field, _, ov = choice_field(rng, qtext, None, list(texts), values={t: list(l) for t, l in texts.items()})
        spec.update(rules=rules, facts=facts, option_values=ov)
        return {"family": FAM, "state": render(random.Random(rseed), cls, rules, facts, d, names, org), "field": field, "gold": None,
                "unknown_reason": "insufficient_evidence", "spec": spec, "unk": None}
    gtxt = f"{ent} {vp(cls, target)}"
    field, gold, ov = choice_field(rng, qtext, gtxt, [t for t in texts if t != gtxt], values={t: list(l) for t, l in texts.items()})
    spec.update(rules=rules, facts=facts, option_values=ov)
    unk = None
    order = list(decisive)
    rng.shuffle(order)
    for kind, prem in order:
        r2 = [x for x in rules if x is not prem]
        f2 = [x for x in facts if x is not prem]
        try:
            if not any(status(r2, f2, ent, l) is True for l in texts.values()):
                unk = {"state": render(random.Random(rseed), cls, r2, f2, d, names, org), "reason": "insufficient_evidence",
                       "spec": {**spec, "rules": r2, "facts": f2, "removed": prem}}
                break
        except Skip:
            continue
    return {"family": FAM, "state": render(random.Random(rseed), cls, rules, facts, d, names, org), "field": field, "gold": gold,
            "unknown_reason": None, "spec": spec, "unk": unk}


def logic_entity(rng: random.Random, d: int) -> dict:
    """Which of these entities is certain to satisfy Q? One entity has the full chain; the others hit a trap."""
    cname = rng.choice(sorted(CLASSES))
    cls = CLASSES[cname]
    names = Names(rng)
    org = names.company()
    n_ent = {3: 3, 4: 4, 5: rng.randint(4, 5)}[d]
    ents = [_entity(names, cls) for _ in range(n_ent)]
    gold_e = ents[0]
    used = set()
    rules, facts, chain, decisive = build_chain(rng, cls, d, gold_e, used)
    add_distractors(rng, cls, d, rules, facts, chain, gold_e, used, [])
    q = (chain[-1][0], True)
    target_true = chain[-1][1]
    if not target_true:
        # ask about the literal that holds: rephrase the question on the negative VP by flipping the predicate sense
        q = (chain[-1][0], False)
    # traps for the other entities: each breaks the chain in a different way
    ex_rules = [r for r in rules if r.get("unless")]
    free = [p for p in range(len(cls["preds"])) if p not in used]
    for e in ents[1:]:
        trap = rng.choice(["denied", "exception", "mid_negated", "disj_fresh"])
        if trap == "exception" and ex_rules:
            facts.append({"ent": e, "lit": chain[0]})
            facts.append({"ent": e, "lit": ex_rules[0]["unless"][0]})
        elif trap == "mid_negated" and len(chain) > 2:
            m = rng.randrange(1, len(chain) - 1)
            facts.append({"ent": e, "lit": (chain[m][0], not chain[m][1])})
        elif trap == "disj_fresh" and free:
            facts.append({"ent": e, "or": [chain[0], (rng.choice(free), rng.random() < 0.5)]})
        else:
            facts.append({"ent": e, "lit": (chain[0][0], not chain[0][1])})
    try:
        stats = {e: status(rules, facts, e, q) for e in ents}
    except Skip:
        raise
    winners = [e for e in ents if stats[e] is True]
    if winners != [gold_e]:
        raise Skip("not unique")
    rseed = rng.random()
    rel = cls["rel"]
    shown = list(ents)
    rng.shuffle(shown)
    listed = ", ".join(shown[:-1]) + " and " + shown[-1]
    qtext = T(rng, "{Of|Among} $l, which one {certainly|definitely|provably} $v?", l=listed, v=vp(cls, q))
    field, gold, ov = choice_field(rng, qtext, gold_e, ents[1:], values={e: e for e in ents})
    spec = {"kind": "logic_entity", "class": cname, "rules": rules, "facts": facts, "query": list(q), "ents": ents, "option_values": ov}
    unk = None
    order = list(decisive)
    rng.shuffle(order)
    for kind, prem in order:
        r2 = [x for x in rules if x is not prem]
        f2 = [x for x in facts if x is not prem]
        try:
            if not any(status(r2, f2, e, q) is True for e in ents):
                unk = {"state": render(random.Random(rseed), cls, r2, f2, d, names, org), "reason": "insufficient_evidence",
                       "spec": {**spec, "rules": r2, "facts": f2, "removed": prem}}
                break
        except Skip:
            continue
    return {"family": FAM, "state": render(random.Random(rseed), cls, rules, facts, d, names, org), "field": field, "gold": gold,
            "unknown_reason": None, "spec": spec, "unk": unk}


KINDS = {"logic_noul": logic_noul, "logic_statement": logic_statement, "logic_entity": logic_entity}
KIND_FAMILY = {k: FAM for k in KINDS}
