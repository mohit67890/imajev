"""Family multi_hop: a generated fictional knowledge base (companies, people, cities, regions, products, universities) written
as profiles, news briefs and directory entries; questions follow 2-5 relation hops. Distractors: facts about other entities,
"former" values (a previous CEO / old head office), and near-miss answers reached by taking a wrong turn. Unknown variants drop
one hop's fact so the chain breaks.

Every relation is functional (one current value per subject), so the gold is the unique value reached by following the
current facts; the spec stores the fact triples and the path so a test can re-walk it."""
from __future__ import annotations

import random

from .common import DOMAINS, REGIONS, Names, Skip, T, assemble, choice_field, noul_field

FAM = "multi_hop"

# relation -> (src type, dst type, fact templates, question noun phrase with {x})
REL = {
    "made_by": ("product", "company", ["The {a} is manufactured by {b}.", "{b} makes the {a}.", "{b} is the sole manufacturer of the {a}."],
                "the manufacturer of {x}"),
    "parent": ("company", "company", ["{a} is a wholly owned subsidiary of {b}.", "{b} owns {a} outright.", "{a} sits inside the {b} group as a subsidiary."],
               "the parent company of {x}"),
    "hq": ("company", "city", ["{a} is headquartered in {b}.", "{a}'s head office is in {b}.", "{a} runs its headquarters out of {b}."],
           "the city where {x} is headquartered"),
    "region": ("city", "region", ["{a} lies in {b}.", "{a} is a town in {b}.", "{a} is one of the larger towns of {b}."], "the region in which {x} lies"),
    "ceo": ("company", "person", ["{b} is the chief executive of {a}.", "{a} is run by chief executive {b}.", "{b} currently leads {a} as CEO."],
            "the chief executive of {x}"),
    "born": ("person", "city", ["{a} was born in {b}.", "{a} is a native of {b}, where they were born.", "{a}'s birthplace is {b}."], "the birthplace of {x}"),
    "supplier": ("company", "company", ["{a} buys all of its key components from {b}.", "{b} is the main supplier to {a}.",
                                        "{a} sources its core parts exclusively from {b}."], "the main supplier of {x}"),
    "auditor": ("company", "company", ["{b} audits the accounts of {a}.", "{a}'s external auditor is {b}.", "{b} signs off {a}'s annual accounts."],
                "the auditor of {x}"),
    "mentor": ("person", "person", ["{a} is mentored by {b}.", "{b} is the mentor of {a}.", "{a} meets their mentor, {b}, every month."],
               "the mentor of {x}"),
    "flagship": ("company", "product", ["The flagship product of {a} is the {b}.", "{a}'s best-known product is the {b}.", "{a} is best known for the {b}."],
                 "the flagship product of {x}"),
    "founder": ("company", "person", ["{a} was founded by {b}.", "{b} founded {a}.", "{b} started {a} from a spare room."], "the founder of {x}"),
    "alma": ("person", "university", ["{a} studied at {b}.", "{a} is a graduate of {b}.", "{a} holds a degree from {b}."],
             "the university that {x} attended"),
    "campus": ("university", "city", ["{a} is located in {b}.", "{a} has its only campus in {b}.", "{a} sits on the edge of {b}."],
               "the city where {x} is located"),
}
FORMER = {"ceo": ["{b} served as chief executive of {a} until {y}.", "Before {y}, {a} was led by {b}."],
          "hq": ["Until {y}, {a} was headquartered in {b}.", "{a} moved its head office out of {b} in {y}."],
          "parent": ["{a} was part of {b} until it was sold in {y}."],
          "supplier": ["{a} stopped buying from {b} in {y}."],
          "auditor": ["{b} audited {a} until {y}, when the mandate was retendered."]}
TYPE_WORD = {"company": "company", "person": "person", "city": "city", "region": "region", "product": "product", "university": "university"}


class World:
    def __init__(self, rng: random.Random, names: Names, d: int):
        self.rng, self.names = rng, names
        n = {3: 1, 4: 1.4, 5: 1.8}[d]
        self.ents = {"company": [names.company() for _ in range(int(8 * n))], "person": [names.person() for _ in range(int(8 * n))],
                     "city": [names.city() for _ in range(int(6 * n))], "region": rng.sample(REGIONS, min(len(REGIONS), int(4 * n))),
                     "product": [names.product() for _ in range(int(5 * n))],
                     "university": [f"{rng.choice(['University of ', ''])}{names.stem()}{rng.choice([' University', ' Institute of Technology', ' College'])}".replace("University of ", "University of ", 1)
                                    for _ in range(int(3 * n))]}
        self.ents["university"] = [u if not u.startswith("University of ") else u.replace(" University", "").replace(" Institute of Technology", "").replace(" College", "")
                                   for u in self.ents["university"]]
        self.facts: dict[tuple[str, str], str] = {}
        for rel, (st, dt_, _, _) in REL.items():
            for a in self.ents[st]:
                cands = [b for b in self.ents[dt_] if b != a]
                if rel == "parent":
                    # parents form a forest: parent index strictly lower
                    ci = self.ents["company"].index(a)
                    cands = self.ents["company"][:ci]
                    if not cands or rng.random() < 0.3:
                        continue
                if cands:
                    self.facts[(a, rel)] = rng.choice(cands)
        # a company's flagship product is always one it manufactures (so flagship and made_by never disagree)
        for c in self.ents["company"]:
            self.facts.pop((c, "flagship"), None)
            made = [p for p in self.ents["product"] if self.facts.get((p, "made_by")) == c]
            if made:
                self.facts[(c, "flagship")] = rng.choice(made)
        self.former = {}
        for (a, rel), b in list(self.facts.items()):
            if rel in FORMER and rng.random() < 0.35:
                st, dt_, _, _ = REL[rel]
                others = [x for x in self.ents[dt_] if x not in (a, b)]
                if rel == "parent":
                    others = [x for x in others if self.ents["company"].index(x) < self.ents["company"].index(a)]
                if others:
                    self.former[(a, rel)] = (rng.choice(others), rng.randint(2012, 2024))

    def typ(self, e):
        for t, xs in self.ents.items():
            if e in xs:
                return t
        raise KeyError(e)

    def walk(self, start, rels, facts=None, former_at=None):
        facts = self.facts if facts is None else facts
        x = start
        for k, r in enumerate(rels):
            if former_at == k and (x, r) in self.former:
                x = self.former[(x, r)][0]
                continue
            if (x, r) not in facts:
                return None
            x = facts[(x, r)]
        return x


def random_path(rng, w: World, hops: int, end_types=None):
    for _ in range(200):
        start_t = rng.choice(["product", "company", "person", "company"])
        start = rng.choice(w.ents[start_t])
        x, rels, seen = start, [], {start}
        ok = True
        for _ in range(hops):
            opts = [r for r, (st, _, _, _) in REL.items() if st == w.typ(x) and (x, r) in w.facts and w.facts[(x, r)] not in seen]
            if rels:
                opts = [r for r in opts if not (r == rels[-1] and r in ("region", "hq", "born", "campus"))]
            if not opts:
                ok = False
                break
            r = rng.choice(opts)
            rels.append(r)
            x = w.facts[(x, r)]
            seen.add(x)
        if ok and (end_types is None or w.typ(x) in end_types):
            return start, rels, x
    raise Skip("no path")


STEP = {"made_by": "the company that manufactures it", "parent": "its parent company", "hq": "the city where it is headquartered",
        "region": "the region that city lies in", "ceo": "its chief executive", "born": "the city where that person was born",
        "supplier": "its main supplier", "auditor": "its auditor", "mentor": "that person's mentor", "flagship": "its flagship product",
        "founder": "its founder", "alma": "the university that person attended", "campus": "the city where that university is located"}


def nested_np(start, rels, w: World) -> str:
    x = ("the " + start) if w.typ(start) == "product" else start
    for r in rels:
        x = REL[r][3].format(x=x)
    return x


def stepwise(r: random.Random, start, rels, w: World) -> str:
    s0 = ("the " + start) if w.typ(start) == "product" else start
    parts = [r.choice(["Start from {s}.", "Begin with {s}.", "Take {s} as the starting point."]).format(s=s0)]
    for k, rel in enumerate(rels):
        parts.append(r.choice(["Take {np}.", "Move to {np}.", "Then go to {np}.", "Next, {np}."]).format(np=STEP[rel]))
    return " ".join(parts)


def render_world(r: random.Random, w: World, keep: set, drop: set, d: int, names: Names, dom) -> str:
    """Render the facts in `keep` (minus `drop`), grouped by subject into profiles, plus former facts for those subjects."""
    by_subj: dict[str, list[str]] = {}
    for (a, rel) in sorted(keep):
        if (a, rel) in drop or (a, rel) not in w.facts:
            continue
        b = w.facts[(a, rel)]
        by_subj.setdefault(a, []).append(r.choice(REL[rel][2]).format(a=a, b=b))
        if (a, rel) in w.former:
            ob, y = w.former[(a, rel)]
            by_subj[a].append(r.choice(FORMER[rel]).format(a=a, b=ob, y=y))
    for (a, rel) in sorted(drop):
        if (a, rel) in w.former:
            ob, y = w.former[(a, rel)]
            by_subj.setdefault(a, []).append(r.choice(FORMER[rel]).format(a=a, b=ob, y=y))
    blocks = []
    subjects = list(by_subj)
    r.shuffle(subjects)
    for s in subjects:
        sents = by_subj[s]
        r.shuffle(sents)
        t = w.typ(s)
        lead = {"company": r.choice(["Company profile: ", "Directory entry - ", "From the business register: ", ""]),
                "person": r.choice(["Bio: ", "People file - ", "Who's who: ", ""]), "city": r.choice(["Gazetteer: ", "Place note - ", ""]),
                "product": r.choice(["Product sheet: ", "Catalogue note - ", ""]), "university": r.choice(["Education register: ", ""]),
                "region": ""}[t]
        blocks.append((lead + " ".join(sents)).replace("s's ", "s' "))
    head = [T(r, "{Background briefing|Reference pack|Due-diligence notes|Market map} {compiled by|prepared by|from} $o{ for the review|}", o=names.company()),
            T(r, "{The entries below are current unless they say otherwise.|All statements describe the present situation unless a date says otherwise.|"
                 "Where an entry mentions a past arrangement, it no longer applies.}")]
    return assemble(r, head, blocks, dom, Names(r, names.used), d)


def linked_facts(w: World, fact, keep) -> set:
    """Facts that would let a reader recover a dropped link: made_by(P)=C is implied by flagship(C)=P."""
    a, rel = fact
    out = set()
    if rel == "made_by":
        out |= {(c, "flagship") for (c, r) in keep if r == "flagship" and w.facts.get((c, r)) == a}
    if rel == "flagship":
        out |= {(p, "made_by") for (p, r) in keep if r == "made_by" and p == w.facts.get(fact)}
    return out


def _keep_set(rng, w: World, path_facts, d):
    keep = set(path_facts)
    extra = [k for k in w.facts if k not in keep]
    rng.shuffle(extra)
    keep.update(extra[: {3: 8, 4: 16, 5: 26}[d]])
    return keep


def _options(rng, w: World, start, rels, gold, keep, n_opts):
    typ = w.typ(gold)
    traps = []
    for k in range(len(rels)):
        t = w.walk(start, rels, former_at=k)
        if t and t != gold and w.typ(t) == typ:
            traps.append(t)
    # sibling / one-hop-off answers
    for k in range(len(rels)):
        alt = rels[:k] + [r for r in REL if REL[r][0] == REL[rels[k]][0] and REL[r][1] == REL[rels[k]][1] and r != rels[k]] + rels[k + 1:]
        if len(alt) == len(rels) + 0:
            t = w.walk(start, alt)
            if t and t != gold and w.typ(t) == typ:
                traps.append(t)
    mentioned = {w.facts[k] for k in keep if k in w.facts} | {a for a, _ in keep}
    pool = [e for e in w.ents[typ] if e != gold and e in mentioned] + [e for e in w.ents[typ] if e != gold]
    out = []
    for e in traps + rng.sample(pool, len(pool)):
        if e not in out and e != gold:
            out.append(e)
        if len(out) >= n_opts - 1:
            break
    if len(out) < 2:
        raise Skip("few options")
    return out


def lettered(rng, start, rels, w, final_typ):
    """'Let K be the manufacturer of the X. Let Q be the auditor of K. ... Which region is Z?' with random letters."""
    letters = rng.sample(list("BCDFGHJKLMNPQRSTVWXZ"), len(rels))
    s0 = ("the " + start) if w.typ(start) == "product" else start
    parts, prev = [], s0
    for L, rel in zip(letters, rels):
        np = REL[rel][3].format(x=prev)
        parts.append(rng.choice(["Let {L} be {np}.", "Call {np} {L}.", "{L} denotes {np}."]).format(L=L, np=np))
        prev = L
    ask = {"region": "Which region is", "city": "Which city is", "person": "Who is", "company": "Which company is", "product": "Which product is",
           "university": "Which university is"}[final_typ]
    return " ".join(parts) + f" {ask} {letters[-1]}?"


def _question(rng, start, rels, w, final_typ):
    if len(rels) <= 2 and rng.random() < 0.6:
        np = nested_np(start, rels, w)
        return T(rng, "{What is|Name|Identify} $np{, according to the notes|}?", np=np), np
    if rng.random() < 0.9:
        return lettered(rng, start, rels, w, final_typ), None
    step = stepwise(rng, start, rels, w)
    return step + " " + T(rng, "{Following these steps using only the notes|Going only by the notes}, which $t do you {end up with|reach}?", t=final_typ), None


def mh_path(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    dom = DOMAINS[rng.choice(sorted(DOMAINS))]
    w = World(rng, names, d)
    hops = {3: 2, 4: rng.randint(3, 4), 5: rng.randint(4, 5)}[d]
    start, rels, gold_e = random_path(rng, w, hops)
    path_facts, x = [], start
    for r_ in rels:
        path_facts.append((x, r_))
        x = w.facts[(x, r_)]
    keep = _keep_set(rng, w, path_facts, d)
    typ = w.typ(gold_e)
    rseed = rng.random()
    state = render_world(random.Random(rseed), w, keep, set(), d, names, dom)
    spec_facts = [[a, r_, w.facts[(a, r_)]] for (a, r_) in sorted(keep) if (a, r_) in w.facts]
    spec = {"kind": "mh_path", "facts": spec_facts, "start": start, "rels": rels}
    typ_choice = "noul" if rng.random() < 0.3 else "choice"
    if typ_choice == "choice":
        qtext, _ = _question(rng, start, rels, w, typ)
        others = _options(rng, w, start, rels, gold_e, keep, rng.choice([4, 5, 5, 6]))
        field, gold, ov = choice_field(rng, qtext, gold_e, others, values={e: e for e in [gold_e] + others})
        spec["option_values"] = ov
    else:
        probe = gold_e if rng.random() < 0.5 else rng.choice(_options(rng, w, start, rels, gold_e, keep, 4))
        if len(rels) <= 2:
            np = nested_np(start, rels, w)
            field = noul_field(T(rng, "{According to the notes|Based on the briefing}, is $np $p?", np=np, p=probe))
        else:
            lq = lettered(rng, start, rels, w, typ)
            last = lq.rsplit(" ", 1)[1].rstrip("?")
            field = noul_field(lq.rsplit(".", 1)[0] + "." + T(rng, " {According to the notes|Going by the briefing}, is $l $p?", l=last, p=probe))
        gold = probe == gold_e
        spec["probe"] = probe
    # unknown variant: drop one hop's fact (the chain breaks there; no other statement gives that link)
    k = rng.randrange(len(path_facts))
    drop = {path_facts[k]} | linked_facts(w, path_facts[k], keep)
    unk = {"state": render_world(random.Random(rseed), w, keep, drop, d, names, dom), "reason": "insufficient_evidence",
           "spec": {**spec, "facts": [f for f in spec_facts if (f[0], f[1]) not in drop], "removed": list(path_facts[k])}}
    return {"family": FAM, "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


def mh_compare(rng: random.Random, d: int) -> dict:
    """Two chains ending in companies; which one was founded earlier (founding years stated in the notes)."""
    names = Names(rng)
    dom = DOMAINS[rng.choice(sorted(DOMAINS))]
    w = World(rng, names, d)
    hops = {3: 1, 4: 2, 5: rng.randint(2, 3)}[d]
    years = {c: rng.randint(1950, 2022) for c in w.ents["company"]}
    s1, r1, e1 = random_path(rng, w, hops, end_types={"company"})
    for _ in range(50):
        s2, r2, e2 = random_path(rng, w, hops, end_types={"company"})
        if e2 != e1 and s2 != s1 and abs(years[e1] - years[e2]) >= 1:
            break
    else:
        raise Skip("no second chain")
    pf = []
    for s, rels in ((s1, r1), (s2, r2)):
        x = s
        for r_ in rels:
            pf.append((x, r_))
            x = w.facts[(x, r_)]
    keep = _keep_set(rng, w, pf, d)
    # founding years for every company mentioned
    mentioned = {a for a, _ in keep if w.typ(a) == "company"} | {w.facts[k] for k in keep if k in w.facts and w.typ(w.facts[k]) == "company"}
    mentioned |= {e1, e2}
    yr_drop = set()

    def rend(r, drop, ydrop):
        base = render_world(r, w, keep, drop, d, names, dom)
        yl = [f"{c} was founded in {years[c]}." if r.random() < 0.5 else f"{c} (est. {years[c]})." for c in sorted(mentioned) if c not in ydrop]
        r.shuffle(yl)
        return base + "\n\n" + T(r, "{Founding dates|Company ages|Year of incorporation}: ") + " ".join(yl)
    rseed = rng.random()
    state = rend(random.Random(rseed), set(), yr_drop)
    np1 = nested_np(s1, r1, w)
    np2 = nested_np(s2, r2, w)
    np1, np2 = np1[:1].upper() + np1[1:], np2[:1].upper() + np2[1:]
    older_np = np1 if years[e1] < years[e2] else np2
    qtext = T(rng, "{Which of these two companies was founded earlier|Which company is older, going by the founding years}?")
    field, gold, ov = choice_field(rng, qtext, older_np, [x for x in (np1, np2) if x != older_np], values={np1: e1, np2: e2})
    # also offer the start entities as distractors to punish stopping early (when they are companies)
    spec = {"kind": "mh_compare", "facts": [[a, r_, w.facts[(a, r_)]] for (a, r_) in sorted(keep) if (a, r_) in w.facts],
            "years": {c: years[c] for c in sorted(mentioned)}, "chains": [[s1, r1], [s2, r2]], "option_values": ov}
    if rng.random() < 0.5:
        k = rng.randrange(len(pf))
        drop, ydrop = {pf[k]} | linked_facts(w, pf[k], keep), set()
        rem = list(pf[k])
    else:
        drop, ydrop = set(), {rng.choice([e1, e2])}
        rem = ["year", list(ydrop)[0]]
    unk = {"state": rend(random.Random(rseed), drop, ydrop), "reason": "insufficient_evidence",
           "spec": {**spec, "facts": [f for f in spec["facts"] if (f[0], f[1]) not in drop],
                    "years": {c: y for c, y in spec["years"].items() if c not in ydrop}, "removed": rem}}
    return {"family": FAM, "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


KINDS = {"mh_path": mh_path, "mh_compare": mh_compare}
KIND_FAMILY = {k: FAM for k in KINDS}
