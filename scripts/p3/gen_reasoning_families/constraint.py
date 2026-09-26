"""Family constraint: ordering / seating / scheduling / ranking puzzles, two-attribute assignment grids and budgeted
allocation, each with a unique solution checked by brute force. Unknown variants drop a clue so that the asked-for fact is
no longer fixed (at least two solutions disagree on it)."""
from __future__ import annotations

import itertools
import random

from .common import CURRENCIES, DOMAINS, Names, Skip, T, assemble, choice_field, fmoney, noul_field

FAM = "constraint"
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


# ------------------------------------------------------------------------------------------------ ordering constraints
def holds(c, pos, n) -> bool:
    t = c[0]
    if t == "before":
        return pos[c[1]] < pos[c[2]]
    if t == "imm":
        return pos[c[2]] == pos[c[1]] + 1
    if t == "adj":
        return abs(pos[c[1]] - pos[c[2]]) == 1
    if t == "nadj":
        return abs(pos[c[1]] - pos[c[2]]) != 1
    if t == "at":
        return pos[c[1]] == c[2]
    if t == "nat":
        return pos[c[1]] != c[2]
    if t == "end":
        return pos[c[1]] in (0, n - 1)
    if t == "gap":
        return abs(pos[c[1]] - pos[c[2]]) >= 3
    if t == "between":
        return min(pos[c[1]], pos[c[3]]) < pos[c[2]] < max(pos[c[1]], pos[c[3]])
    raise ValueError(t)


def order_solutions(items, cons, limit=None):
    n = len(items)
    out = []
    for perm in itertools.permutations(range(n)):
        pos = {items[i]: perm[i] for i in range(n)}
        if all(holds(c, pos, n) for c in cons):
            out.append(pos)
            if limit and len(out) >= limit:
                break
    return out


def random_constraint(rng, items, pos, n, d):
    a, b, c = rng.sample(items, 3) if n >= 3 else (items[0], items[1], items[0])
    kinds = ["before", "before", "imm", "adj", "nadj", "at", "nat", "end", "gap", "between"]
    if d == 3:
        kinds = ["before", "before", "imm", "at", "nat", "adj"]
    for _ in range(50):
        t = rng.choice(kinds)
        if t == "before":
            x = (t, a, b) if pos[a] < pos[b] else (t, b, a)
        elif t == "imm":
            nxt = [y for y in items if pos[y] == pos[a] + 1]
            if not nxt:
                a, b, c = rng.sample(items, 3)
                continue
            x = (t, a, nxt[0])
        elif t == "adj":
            nb = [y for y in items if abs(pos[y] - pos[a]) == 1]
            x = (t, a, rng.choice(nb))
        elif t == "nadj":
            if abs(pos[a] - pos[b]) == 1:
                a, b, c = rng.sample(items, 3)
                continue
            x = (t, a, b)
        elif t == "at":
            x = (t, a, pos[a])
        elif t == "nat":
            k = rng.choice([k for k in range(n) if k != pos[a]])
            x = (t, a, k)
        elif t == "end":
            if pos[a] not in (0, n - 1):
                a, b, c = rng.sample(items, 3)
                continue
            x = (t, a)
        elif t == "gap":
            if abs(pos[a] - pos[b]) < 3:
                a, b, c = rng.sample(items, 3)
                continue
            x = (t, a, b)
        else:
            s = sorted([a, b, c], key=lambda y: pos[y])
            x = (t, s[0], s[1], s[2]) if rng.random() < 0.5 else (t, s[2], s[1], s[0])
        if holds(x, pos, n):
            return x
    raise Skip("constraint")


THEMES = {
    "talks": dict(items="people", slot=lambda k, n: f"the {ORD[k]} talk", order_word="speaking order", dom="education",
                  intro="{Six|Several} {speakers|presenters} {are booked|will present} at the {research day|quarterly showcase|induction morning}, one after another. "
                        "Each gives exactly one talk and no two talks overlap.",
                  t=dict(before="{a} speaks before {b}.", imm="{a} speaks immediately before {b}.", adj="{a} and {b} speak one right after the other (in either order).",
                         nadj="{a} and {b} do not speak one right after the other.", at="{a} gives {k}.", nat="{a} does not give {k}.",
                         end="{a} speaks either first or last.", gap="At least two other talks come between those of {a} and {b}.",
                         between="{b}'s talk comes somewhere between those of {a} and {c}."),
                  q_slot="Who gives {k}?", q_item="{a}", noun="speaker"),
    "stops": dict(items="stores", slot=lambda k, n: f"stop {k + 1}", order_word="route order", dom="logistics",
                  intro="A {delivery van|courier|service vehicle} visits each of the following {sites|customers|branches} exactly once on today's route, "
                        "numbered stop 1, stop 2 and so on.",
                  t=dict(before="The van reaches {a} before {b}.", imm="{b} is the stop straight after {a}.", adj="{a} and {b} are consecutive stops (in either order).",
                         nadj="{a} and {b} are not consecutive stops.", at="{a} is {k}.", nat="{a} is not {k}.", end="{a} is either the first or the last stop.",
                         gap="There are at least two other stops between {a} and {b}.", between="{b} is visited at some point between {a} and {c}."),
                  q_slot="Which site is {k}?", q_item="{a}", noun="site"),
    "bays": dict(items="people", slot=lambda k, n: f"bay {k + 1}", order_word="bay allocation", dom="hr",
                 intro="{Parking bays|Hot desks|Lockers} numbered 1 to N are in a single row, 1 at the left. Each of the following people is given exactly one, "
                       "and every one is used.",
                 t=dict(before="{a} is somewhere to the left of {b}.", imm="{a} is directly to the left of {b}.", adj="{a} and {b} are next to each other.",
                        nadj="{a} and {b} are not next to each other.", at="{a} has {k}.", nat="{a} does not have {k}.", end="{a} is at one end of the row.",
                        gap="At least two others are between {a} and {b}.", between="{b} is somewhere between {a} and {c}."),
                 q_slot="Who has {k}?", q_item="{a}", noun="person"),
    "floors": dict(items="depts", slot=lambda k, n: f"floor {k + 1}", order_word="floor plan", dom="manufacturing",
                   intro="The new building has one department per floor, floors numbered from 1 (lowest) upwards; every floor is occupied.",
                   t=dict(before="{a} is on a lower floor than {b}.", imm="{a} is directly below {b}.", adj="{a} and {b} are on adjacent floors.",
                          nadj="{a} and {b} are not on adjacent floors.", at="{a} is on {k}.", nat="{a} is not on {k}.", end="{a} is on either the lowest or the highest floor.",
                          gap="At least two floors separate {a} and {b} (at least two other departments sit between them).",
                          between="{b} is on a floor between those of {a} and {c}."),
                   q_slot="Which department is on {k}?", q_item="{a}", noun="department"),
    "ranking": dict(items="branches", slot=lambda k, n: f"rank {k + 1}", order_word="ranking", dom="retail",
                    intro="The quarterly league table ranks the branches by net sales, rank 1 being the highest; there were no ties.",
                    t=dict(before="{a} sold more than {b}.", imm="{a} finished exactly one place above {b}.", adj="{a} and {b} finished in neighbouring places.",
                           nadj="{a} and {b} did not finish in neighbouring places.", at="{a} finished at {k}.", nat="{a} did not finish at {k}.",
                           end="{a} finished either top or bottom.", gap="At least two branches finished between {a} and {b}.",
                           between="{b} finished somewhere between {a} and {c}."),
                    q_slot="Which branch finished at {k}?", q_item="{a}", noun="branch"),
}
DEPTS = ["Finance", "Legal", "Design", "Research", "Sales", "Support", "Procurement", "Security", "Training", "Facilities", "Analytics", "Marketing"]


def _items(rng, names, kind, n):
    if kind == "people":
        return [names.first() for _ in range(n)]
    if kind == "depts":
        return rng.sample(DEPTS, n)
    if kind == "branches":
        return [names.city() for _ in range(n)]
    return [names.stem() + rng.choice([" Depot", " Market", " Clinic", " Store", " Works"]) for _ in range(n)]


def con_text(th, c, n):
    t = th["t"][c[0]]
    if c[0] in ("at", "nat"):
        return t.format(a=c[1], k=th["slot"](c[2], n))
    if c[0] == "between":
        return t.format(a=c[1], b=c[2], c=c[3])
    if c[0] == "end":
        return t.format(a=c[1])
    return t.format(a=c[1], b=c[2])


def cs_order(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    tname = rng.choice(sorted(THEMES))
    th = THEMES[tname]
    n = {3: 4, 4: 5, 5: rng.choice([6, 6, 7])}[d]
    items = _items(rng, names, th["items"], n)
    perm = list(range(n))
    rng.shuffle(perm)
    pos = {items[i]: perm[i] for i in range(n)}
    cons = []
    for _ in range(40):
        cons.append(random_constraint(rng, items, pos, n, d))
        if len(order_solutions(items, cons, limit=2)) == 1:
            break
    else:
        raise Skip("not unique")
    # remove redundant clues (harder), keep uniqueness
    for c in list(cons):
        if rng.random() < 0.8:
            trial = [x for x in cons if x is not c]
            if len(order_solutions(items, trial, limit=2)) == 1:
                cons = trial
    if d >= 4 and sum(1 for c in cons if c[0] == "at") > 1:
        raise Skip("too many direct placements")
    sol = order_solutions(items, cons)[0]
    by_slot = sorted(items, key=lambda x: sol[x])
    rseed = rng.random()

    def render(cs, r):
        cs = list(cs)
        r.shuffle(cs)
        intro = spin_intro(r, th, n)
        clues = "\n".join(f"- {con_text(th, c, n)}" for c in cs)
        head = [T(r, "{Puzzle brief|Planning constraints|Scheduling note} - $o", o=names.company()), intro,
                "Participants: " + ", ".join(sorted(items, key=lambda _: r.random())) + "."]
        return assemble(r, head, [T(r, "{The following all hold|Known constraints|Confirmed facts}:") + "\n" + clues], DOMAINS[th["dom"]], Names(r, names.used), d)

    qt = rng.choice(["slot", "slot", "full", "noul"])
    spec = {"kind": "cs_order", "items": items, "cons": [list(c) for c in cons], "qt": qt}
    if qt == "slot":
        k = rng.randrange(n)
        gold_item = by_slot[k]
        others = [x for x in items if x != gold_item]
        rng.shuffle(others)
        others = others[: rng.choice([3, 4, 5])]
        field, gold, ov = choice_field(rng, th["q_slot"].format(k=th["slot"](k, n)), gold_item, others, values={x: x for x in items})
        spec.update(slot=k, option_values=ov)
    elif qt == "full":
        gold_txt = " > ".join(by_slot) if tname == "ranking" else ", ".join(by_slot)
        cands = []
        for _ in range(200):
            a, b = rng.sample(range(n), 2)
            o = list(by_slot)
            o[a], o[b] = o[b], o[a]
            p2 = {x: i for i, x in enumerate(o)}
            if sum(1 for c in cons if not holds(c, p2, n)) >= 1 and o not in cands:
                cands.append(o)
            if len(cands) >= 4:
                break
        sep = " > " if tname == "ranking" else ", "
        texts = [sep.join(o) for o in cands]
        qtext = T(rng, "Which $w fits all the clues ({from $s on|starting at $s})?", w=th["order_word"], s=th["slot"](0, n))
        field, gold, ov = choice_field(rng, qtext, gold_txt, texts, values={t: t for t in [gold_txt] + texts})
        spec.update(option_values=ov, sep=sep)

        def answer(sols):
            return {sep.join(sorted(items, key=lambda x: s[x])) for s in sols}
    else:
        a, b = rng.sample(items, 2)
        form = rng.choice(["before", "imm", "adj"])
        c = (form, a, b)
        gold = holds(c, sol, n)
        stxt = con_text(th, c, n)[:-1]
        if stxt.split()[0] not in items:
            stxt = stxt[:1].lower() + stxt[1:]
        field = noul_field(T(rng, "{Based on the constraints|Given the constraints above|From the clues}, is it true that $s?", s=stxt))
        spec.update(probe=list(c))

        def answer(sols):
            return {holds(c, s, n) for s in sols}
    # unknown: drop one clue so that the answer is no longer fixed (among options for choice questions)
    unk = None
    order = list(cons)
    rng.shuffle(order)
    for cdrop in order:
        trial = [x for x in cons if x is not cdrop]
        sols = order_solutions(items, trial)
        if qt == "slot":
            occ = {[x for x in items if s[x] == spec["slot"]][0] for s in sols}
            ok = len(occ & set(ov.values())) >= 2
        elif qt == "full":
            ok = len(answer(sols) & set(ov.values())) >= 2
        else:
            ok = len(answer(sols)) == 2
        if ok:
            unk = {"state": render(trial, random.Random(rseed)), "reason": "insufficient_evidence",
                   "spec": {**spec, "cons": [list(x) for x in trial], "removed": list(cdrop)}}
            break
    return {"family": FAM, "state": render(cons, random.Random(rseed)), "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


def spin_intro(r, th, n):
    return T(r, th["intro"]).replace("Six", {4: "Four", 5: "Five", 6: "Six", 7: "Seven"}[n]).replace("1 to N", f"1 to {n}")


# ------------------------------------------------------------------------------------------------ two-attribute assignment
ASSIGN = [dict(a="day", avals=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], b="room", bvals=["Room A", "Room B", "Room C", "Room D", "Room E"],
               ctx="Each auditor visits on a different weekday (Monday to Friday order matters) and works from a different meeting room.", ordered_a=True, dom="finance",
               who="auditor"),
          dict(a="shift", avals=["06:00", "08:00", "10:00", "12:00", "14:00"], b="station", bvals=["packing", "labelling", "loading", "returns", "inspection"],
               ctx="Each operative starts at a different time (earlier to later as listed) and staffs a different station.", ordered_a=True, dom="logistics",
               who="operative"),
          dict(a="slot", avals=["09:00", "10:00", "11:00", "13:00", "14:00"], b="panel", bvals=["panel North", "panel South", "panel East", "panel West", "panel Central"],
               ctx="Each candidate has a different interview time (in the order listed) and faces a different panel.", ordered_a=True, dom="hr",
               who="candidate")]


def acons_holds(c, asg, n):
    t = c[0]
    A, B = asg
    if t == "a_is":
        return A[c[1]] == c[2]
    if t == "a_not":
        return A[c[1]] != c[2]
    if t == "b_is":
        return B[c[1]] == c[2]
    if t == "b_not":
        return B[c[1]] != c[2]
    if t == "a_before":
        return A[c[1]] < A[c[2]]
    if t == "a_next":
        return A[c[2]] == A[c[1]] + 1
    if t == "b_of_a":  # whoever has a-value k has b-value m
        return any(A[p] == c[1] and B[p] == c[2] for p in A)
    if t == "b_of_a_not":
        return not any(A[p] == c[1] and B[p] == c[2] for p in A)
    if t == "b_before":  # person with b-value m comes (in a-order) before person x
        return any(B[p] == c[1] and A[p] < A[c[2]] for p in A)
    raise ValueError(t)


def assign_solutions(people, n, cons, limit=None):
    out = []
    for pa in itertools.permutations(range(n)):
        A = dict(zip(people, pa))
        # quick prune on a-only clues
        if not all(acons_holds(c, (A, {p: -1 for p in people}), n) for c in cons if c[0] in ("a_is", "a_not", "a_before", "a_next")):
            continue
        for pb in itertools.permutations(range(n)):
            B = dict(zip(people, pb))
            if all(acons_holds(c, (A, B), n) for c in cons):
                out.append((A, B))
                if limit and len(out) >= limit:
                    return out
    return out


def cs_assign(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    g = rng.choice(ASSIGN)
    n = {3: 3, 4: 4, 5: 5}[d]
    people = [names.first() for _ in range(n)]
    pa = list(range(n))
    pb = list(range(n))
    rng.shuffle(pa)
    rng.shuffle(pb)
    A, B = dict(zip(people, pa)), dict(zip(people, pb))
    av, bv = g["avals"][:n], g["bvals"][:n]
    cons = []

    def rand_con():
        p, q = rng.sample(people, 2)
        t = rng.choice(["a_not", "b_not", "a_before", "a_next", "b_of_a", "b_of_a_not", "b_before", "a_not", "b_not"] if d >= 4 else
                       ["a_not", "b_not", "a_before", "b_of_a", "b_is", "a_is", "a_not", "b_not"])
        if t == "a_is":
            return (t, p, A[p])
        if t == "b_is":
            return (t, p, B[p])
        if t == "a_not":
            return (t, p, rng.choice([k for k in range(n) if k != A[p]]))
        if t == "b_not":
            return (t, p, rng.choice([k for k in range(n) if k != B[p]]))
        if t == "a_before":
            return (t, p, q) if A[p] < A[q] else (t, q, p)
        if t == "a_next":
            nxt = [x for x in people if A[x] == A[p] + 1]
            return (t, p, nxt[0]) if nxt else None
        if t == "b_of_a":
            return (t, A[p], B[p])
        if t == "b_of_a_not":
            return (t, A[p], rng.choice([k for k in range(n) if k != B[p]]))
        if A[p] < A[q]:
            return (t, B[p], q)
        return None

    for _ in range(60):
        c = rand_con()
        if c is None or c in cons:
            continue
        cons.append(c)
        if len(assign_solutions(people, n, cons, limit=2)) == 1:
            break
    else:
        raise Skip("not unique")
    for c in list(cons):
        if rng.random() < 0.7:
            trial = [x for x in cons if x is not c]
            if len(assign_solutions(people, n, trial, limit=2)) == 1:
                cons = trial
    if sum(1 for c in cons if c[0] in ("a_is", "b_is")) > (2 if d == 3 else 1):
        raise Skip("too direct")

    def ctext(c):
        t = c[0]
        if t == "a_is":
            return f"{c[1]}'s {g['a']} is {av[c[2]]}."
        if t == "a_not":
            return f"{c[1]}'s {g['a']} is not {av[c[2]]}."
        if t == "b_is":
            return f"{c[1]} is assigned {bv[c[2]]}."
        if t == "b_not":
            return f"{c[1]} is not assigned {bv[c[2]]}."
        if t == "a_before":
            return f"{c[1]}'s {g['a']} comes earlier than {c[2]}'s."
        if t == "a_next":
            return f"{c[2]}'s {g['a']} is the one immediately after {c[1]}'s."
        if t == "b_of_a":
            return f"The {g['who']} whose {g['a']} is {av[c[1]]} is assigned {bv[c[2]]}."
        if t == "b_of_a_not":
            return f"The {g['who']} whose {g['a']} is {av[c[1]]} is not assigned {bv[c[2]]}."
        return f"The {g['who']} assigned {bv[c[1]]} has an earlier {g['a']} than {c[2]}."

    rseed = rng.random()

    def render(cs, r):
        cs = list(cs)
        r.shuffle(cs)
        head = [T(r, "{Rota|Allocation|Schedule} {worksheet|draft} - $o", o=names.company()),
                g["ctx"] + f" The {g['who']}s are {', '.join(people[:-1])} and {people[-1]}; the {g['a']}s are {', '.join(av)}; the {g['b']}s are {', '.join(bv)}. "
                "Every value is used exactly once."]
        return assemble(r, head, ["Constraints:\n" + "\n".join(f"- {ctext(c)}" for c in cs)], DOMAINS[g["dom"]], Names(r, names.used), d)

    sols = assign_solutions(people, n, cons)
    A1, B1 = sols[0]
    p = rng.choice(people)
    ask = rng.choice(["b_of_person", "person_of_a", "noul"])
    spec = {"kind": "cs_assign", "people": people, "n": n, "cons": [list(c) for c in cons], "ask": ask, "who": p}
    if ask == "b_of_person":
        gt = bv[B1[p]]
        field, gold, ov = choice_field(rng, f"Which {g['b']} is {p} assigned?", gt, [x for x in bv if x != gt], values={x: i for i, x in enumerate(bv)})

        def answer(sl):
            return {B[p] for _, B in sl}
    elif ask == "person_of_a":
        k = rng.randrange(n)
        gp = [x for x in people if A1[x] == k][0]
        field, gold, ov = choice_field(rng, f"Whose {g['a']} is {av[k]}?", gp, [x for x in people if x != gp], values={x: x for x in people})
        spec["k"] = k

        def answer(sl):
            return {[x for x in people if A[x] == k][0] for A, _ in sl}
    else:
        m = rng.randrange(n)
        gold = B1[p] == m
        field = noul_field(f"Is {p} assigned {bv[m]}?")
        spec["m"] = m

        def answer(sl):
            return {B[p] == m for _, B in sl}
        ov = None
    if ov is not None:
        spec["option_values"] = ov
    unk = None
    order = list(cons)
    rng.shuffle(order)
    for c in order:
        trial = [x for x in cons if x is not c]
        if len(answer(assign_solutions(people, n, trial))) >= 2:
            unk = {"state": render(trial, random.Random(rseed)), "reason": "insufficient_evidence",
                   "spec": {**spec, "cons": [list(x) for x in trial], "removed": list(c)}}
            break
    return {"family": FAM, "state": render(cons, random.Random(rseed)), "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


# ------------------------------------------------------------------------------------------------ budgeted allocation
PROJ = ["warehouse robotics pilot", "CRM migration", "solar canopy", "staff wellbeing app", "fraud analytics", "fleet electrification", "data-centre cooling",
        "customer portal redesign", "cyber-security audit", "training academy", "packaging redesign", "route optimiser", "ERP upgrade", "clinic extension",
        "library digitisation", "call-centre automation"]


def best_bundle(inp, n, rules, strict_tie=True):
    best, key = None, None
    for mask in range(1 << n):
        s = [j for j in range(n) if mask >> j & 1]
        if sum(inp[f"c|{j}"] for j in s) > inp["budget"]:
            continue
        ok = True
        for r in rules:
            if r[0] == "req" and r[1] in s and r[2] not in s:
                ok = False
            if r[0] == "excl" and r[1] in s and r[2] in s:
                ok = False
            if r[0] == "must" and r[1] not in s:
                ok = False
        if not ok:
            continue
        k = (sum(inp[f"v|{j}"] for j in s), -sum(inp[f"c|{j}"] for j in s))
        if key is None or k > key:
            best, key, tie = s, k, False
        elif k == key:
            tie = True
    if best is None:
        raise Skip("infeasible")
    if strict_tie and tie:
        raise Skip("tie")
    return tuple(best)


def cs_alloc(rng: random.Random, d: int) -> dict:
    names = Names(rng)
    cur = rng.choice(CURRENCIES[:7])
    dom = DOMAINS[rng.choice(sorted(DOMAINS))]
    n = {3: 5, 4: 6, 5: rng.randint(7, 8)}[d]
    projs = [f"{rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.randint(10, 99)} {p}" for p in rng.sample(PROJ, n)]
    org = names.company()
    inp = {}
    for j in range(n):
        inp[f"c|{j}"] = rng.randint(4, 60) * 5000
        inp[f"v|{j}"] = rng.randint(10, 99)
    tot = sum(inp[f"c|{j}"] for j in range(n))
    inp["budget"] = int(tot * rng.uniform(0.35, 0.6) / 5000) * 5000
    rules = []
    if d >= 4:
        a, b = rng.sample(range(n), 2)
        rules.append(("req", a, b))
        c1, c2 = rng.sample([j for j in range(n) if j not in (a,)], 2)
        rules.append(("excl", c1, c2))
    if d == 5:
        rules.append(("must", rng.randrange(n)))
        a, b = rng.sample(range(n), 2)
        if ("req", b, a) not in rules:
            rules.append(("req", a, b))
    g = best_bundle(inp, n, rules)
    if not g:
        raise Skip("empty")

    def label(s):
        return ", ".join(sorted(projs[j] for j in s)) if s else "none"

    # near-miss bundles: best ignoring each rule, best ignoring budget by one project, greedy by value
    cands = []
    for k in range(len(rules)):
        try:
            b = best_bundle(inp, n, rules[:k] + rules[k + 1:], strict_tie=False)
            cands.append(b)
        except Skip:
            pass
    try:
        cands.append(best_bundle(dict(inp, budget=inp["budget"] + 5000 * rng.randint(2, 8)), n, rules, strict_tie=False))
    except Skip:
        pass
    order = sorted(range(n), key=lambda j: -inp[f"v|{j}"])
    greedy, spent = [], 0
    for j in order:
        if spent + inp[f"c|{j}"] <= inp["budget"]:
            greedy.append(j)
            spent += inp[f"c|{j}"]
    cands.append(tuple(sorted(greedy)))
    ratio = sorted(range(n), key=lambda j: -inp[f"v|{j}"] / inp[f"c|{j}"])
    rg, spent = [], 0
    for j in ratio:
        if spent + inp[f"c|{j}"] <= inp["budget"]:
            rg.append(j)
            spent += inp[f"c|{j}"]
    cands.append(tuple(sorted(rg)))
    for _ in range(20):
        s = tuple(sorted(rng.sample(range(n), rng.randint(1, n - 1))))
        cands.append(s)
    texts, seen = [], {label(g)}
    for s in cands:
        t = label(s)
        if t not in seen:
            seen.add(t)
            texts.append(t)
        if len(texts) >= rng.choice([3, 4, 4]):
            break
    if len(texts) < 2:
        raise Skip("few bundles")
    rseed = rng.random()

    def render(i, missing, r):
        rows = "\n".join(f"| {projs[j]} | {'[quote pending]' if f'c|{j}' in missing else fmoney(i[f'c|{j}'], cur, 0)} | "
                         f"{'[not scored]' if f'v|{j}' in missing else i[f'v|{j}']} |" for j in range(n))
        tbl = "| Project | Cost | Benefit score |\n|---|---|---|\n" + rows
        rt = [T(r, "The {investment committee|steering group|board} can spend at most $b in total{ this year|}.", b=fmoney(i["budget"], cur, 0)),
              T(r, "{It funds|The aim is to fund} the set of projects with the highest total benefit score within budget; if two sets tie on benefit, the cheaper "
                   "set wins. Projects cannot be part-funded.")]
        for rr in rules:
            if rr[0] == "req":
                rt.append(T(r, "The $a can only go ahead if the $b is also funded.", a=projs[rr[1]], b=projs[rr[2]]))
            elif rr[0] == "excl":
                rt.append(T(r, "The $a and the $b {cannot both be funded|are mutually exclusive} (they need the same team).", a=projs[rr[1]], b=projs[rr[2]]))
            else:
                rt.append(T(r, "The $a is {mandatory|a regulatory commitment} and must be funded.", a=projs[rr[1]]))
        head = [T(r, "{Capital allocation round|Project prioritisation|Budget bid review} - $o", o=org)]
        return assemble(r, head, [tbl] + rt, dom, Names(r, names.used), d)

    qtext = T(rng, "Which projects should $pc committee fund?", co=org)
    field, gold, ov = choice_field(rng, qtext, label(g), texts, values={t: t for t in [label(g)] + texts})
    spec = {"kind": "cs_alloc", "inputs": dict(inp), "n": n, "projs": projs, "rules": [list(x) for x in rules], "option_values": ov}
    unk = None
    keys = [f"c|{j}" for j in g] + [f"v|{j}" for j in range(n)]
    rng.shuffle(keys)
    opt_labels = set(ov.values())
    for k in keys:
        for f in (0.5, 2, 3, 0.3, 1.5):
            alt = dict(inp, **{k: int(inp[k] * f)})
            try:
                b = best_bundle(alt, n, rules, strict_tie=False)
            except Skip:
                continue
            if label(b) != label(g) and label(b) in opt_labels:
                unk = {"state": render(inp, {k}, random.Random(rseed)), "reason": "insufficient_evidence",
                       "spec": {**spec, "removed": k, "witness": [inp[k], alt[k]], "witness_alt": {k: alt[k]}}}
                break
        if unk:
            break
    return {"family": FAM, "state": render(inp, set(), random.Random(rseed)), "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


KINDS = {"cs_order": cs_order, "cs_assign": cs_assign, "cs_alloc": cs_alloc}
KIND_FAMILY = {k: FAM for k in KINDS}
