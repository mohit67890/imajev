"""long_input: 4k-16k-token states (logs, call transcripts, record exports, audit trails) with the decisive fact buried mid-way.

Kinds
    final_status   application log; final status of entity X (near-duplicate id X' keeps changing later; X is mentioned again
                   later without a status)                                                            -> choice
    agreed_date    support-call transcript; the date finally booked after proposals and a change of mind  -> choice
    record_lookup  a 150-450-row export; one field of record X (near-duplicate X' and later amendments of other rows) -> choice / noul
    count_after    service log; how many times job Z failed after worker W's LAST restart (earlier restart, Z' failures,
                   W' restarts as distractors)                                                        -> score
    audit_last     configuration audit trail; who last changed setting Y before the incident (Y' look-alike, changes of Y after
                   the incident)                                                                      -> choice

Unknown children replace a block of lines around the decisive line(s) with an explicit gap marker ("lines not retained"). The
spec records the decisive events and the gap; recheck() recomputes the answer from them and returns None when the gap can hide a
change that alters the answer.

Sizes: common.approx_tokens (conservative, len/3.3) between 4,000 and 15,500; the decisive line sits between 25% and 75% of the text.
"""
from __future__ import annotations

import datetime as dt
import random

from .common import DOMAIN_IDS, DOMAINS, P, People, approx_tokens, chance, choice_field, item, noul_field, score_field

FAMILY = "long_input"
KINDS = ["final_status", "agreed_date", "record_lookup", "count_after", "audit_last"]
MIN_TOK, MAX_TOK = 4000, 15500


class Retry(ValueError):
    pass


def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    for _ in range(20):
        try:
            items = GEN[kind](rng, difficulty, want_unknown)
        except Retry:
            continue
        return _with_opt_values(items)
    raise ValueError(f"{kind}: failed")


def _with_opt_values(items):
    """Test convention: for choice items recheck() returns the option key, so the canonical value of each option is its key."""
    for it in items:
        if it["field"]["type"] == "choice":
            it["spec"] = dict(it["spec"], opt_values={o["key"]: o["key"] for o in it["field"]["options"]})
    return items


def _budget(rng, diff, kind):
    """Character budget: at least ~13.6k chars (approx_tokens >= 4,000 with margin), at most ~14k real tokens."""
    lo, hi = {3: (4800, 8000), 4: (6500, 11500), 5: (9000, 14000)}[diff]
    lo_c = max(13800, int(lo * CPT[kind]))
    hi_c = max(int(hi * CPT[kind]), lo_c + 3000)
    return rng.randint(lo_c, hi_c)


def _near_dup(rng, ident: str) -> str:
    """Same prefix, two adjacent digits swapped (or one digit changed)."""
    pre, num = ident.rsplit("-", 1)
    ds = list(num)
    for _ in range(20):
        i = rng.randrange(len(ds) - 1)
        if ds[i] != ds[i + 1]:
            ds[i], ds[i + 1] = ds[i + 1], ds[i]
            return f"{pre}-{''.join(ds)}"
    ds[-1] = str((int(ds[-1]) + 1) % 10)
    return f"{pre}-{''.join(ds)}"


def _fill_to(rng, make_line, budget_chars: int, reserve: int = 0, size=None) -> list:
    out, n = [], reserve
    while n < budget_chars:
        l = make_line()
        out.append(l)
        n += size(l) + 1 if size else (len(" ".join(map(str, l))) + 12 if isinstance(l, tuple) else len(l) + 1)
    return out


def _insert(lines: list, specials: list[tuple[float, object]]) -> tuple[list, list[int]]:
    """Insert special lines at fractional positions (sorted by fraction); returns new list and their indices."""
    specials = sorted(specials, key=lambda x: x[0])
    out, idx, j = [], [], 0
    n = len(lines)
    for i, l in enumerate(lines):
        while j < len(specials) and specials[j][0] * n <= i:
            idx.append(len(out))
            out.append(specials[j][1])
            j += 1
        out.append(l)
    while j < len(specials):
        idx.append(len(out))
        out.append(specials[j][1])
        j += 1
    return out, idx


def _char_frac(lines: list[str], i: int) -> float:
    before = sum(len(l) + 1 for l in lines[:i])
    return before / max(1, sum(len(l) + 1 for l in lines))


def _gap(lines: list[str], lo: int, hi: int, marker: str) -> list[str]:
    return lines[:lo] + [marker.format(n=hi - lo)] + lines[hi:]


# Real Qwen tokenizer density measured on these formats (ids, timestamps and numbers tokenise densely): chars per token, rounded
# down for margin. common.approx_tokens (len/3.3) under-counts logs by ~1.5-1.8x, so sizes are controlled with these instead.
CPT = {"final_status": 2.0, "agreed_date": 2.7, "record_lookup": 1.65, "count_after": 2.0, "audit_last": 2.0}
EST_MIN, EST_MAX = 4300, 15000


def est_tokens(text: str, kind: str) -> int:
    return int(len(text) / CPT[kind]) + 1


def _check_size(text: str, kind: str):
    t = est_tokens(text, kind)
    if not EST_MIN <= t <= EST_MAX or not MIN_TOK <= approx_tokens(text) <= MAX_TOK:
        raise Retry(f"size {t}")


def _ids(rng, prefix, digits=5):
    return f"{prefix}-{rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"


def _fresh_id(rng, prefix, avoid, digits=5):
    while True:
        x = _ids(rng, prefix, digits)
        if x not in avoid:
            return x


# ------------------------------------------------------------------------------------------------ log helpers
SERVICES = ["order-svc", "billing-api", "notify-worker", "auth-gw", "inventory", "search-idx", "report-gen", "webhook-relay", "pricing", "ledger"]
USERS = ["svc-batch", "a.kowalski", "m.tanaka", "r.haddad", "j.okafor", "s.lindqvist", "p.moreau", "d.novak", "k.iyer", "e.rossi", "t.brennan",
         "f.nakamura", "l.mbeki", "c.petrov", "h.zamora", "b.yilmaz"]
RES = ["orders", "customers", "invoices", "shipments", "tickets", "sessions", "carts", "refunds", "payouts", "claims"]
TASKS = ["nightly-reconcile", "cache-warmup", "sitemap-build", "token-cleanup", "fx-rate-sync", "stock-snapshot", "sla-report", "audit-export"]
HOSTS = ["hooks.partner-a.example", "api.carrier.example", "erp.internal", "crm.internal", "events.example.net"]


def _log_fmt(rng):
    f = rng.randrange(3)
    if f == 0:
        return lambda t, lvl, svc, msg: f"{t:%Y-%m-%dT%H:%M:%S}.{t.microsecond // 1000:03d}Z {lvl:<5} [{svc}] {msg}"
    if f == 1:
        return lambda t, lvl, svc, msg: f"[{t:%H:%M:%S}] {svc} | {lvl} | {msg}"
    return lambda t, lvl, svc, msg: '{"ts":"' + f"{t:%Y-%m-%dT%H:%M:%S}" + f'","level":"{lvl}","svc":"{svc}","msg":"' + msg.replace('"', "'") + '"}'


def _generic_msg(rng, avoid_ids=(), extra=None):
    r = rng.random()
    if extra and r < 0.35:
        return extra()
    k = rng.randrange(14)
    if k == 0:
        return f"GET /api/v2/{rng.choice(RES)}/{rng.randint(10000, 99999)} 200 {rng.randint(3, 900)}ms"
    if k == 1:
        return f"cache miss for key {rng.choice(RES)}:{rng.randint(1000, 99999)}"
    if k == 2:
        return f"worker-{rng.randint(1, 40)} heartbeat ok (queue depth {rng.randint(0, 300)})"
    if k == 3:
        return f"scheduled task {rng.choice(TASKS)} completed in {rng.randint(40, 90000)}ms"
    if k == 4:
        return f"connection pool stats: active={rng.randint(1, 80)} idle={rng.randint(0, 40)} waiting={rng.randint(0, 5)}"
    if k == 5:
        return f"user {rng.choice(USERS)} signed in from 10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if k == 6:
        return f"retrying webhook delivery to {rng.choice(HOSTS)} (attempt {rng.randint(1, 5)})"
    if k == 7:
        return f"POST /api/v2/{rng.choice(RES)} 201 {rng.randint(10, 700)}ms"
    if k == 8:
        return f"rate limit warning for client app-{rng.randint(100, 999)} ({rng.randint(80, 99)}% of quota)"
    if k == 9:
        return f"gc pause {rng.randint(2, 250)}ms, heap {rng.randint(200, 3800)}MB"
    if k == 10:
        return f"slow query on {rng.choice(RES)} ({rng.randint(900, 9000)}ms), plan cached"
    if k == 11:
        return f"feature flag {rng.choice(['new-checkout', 'dark-mode', 'bulk-export', 'fast-refund', 'smart-routing'])} evaluated for {rng.randint(1, 900)} users"
    if k == 12:
        return f"TLS certificate for {rng.choice(HOSTS)} valid for {rng.randint(10, 300)} more days"
    return f"queue {rng.choice(['emails', 'exports', 'invoices', 'labels', 'webhooks'])} drained ({rng.randint(1, 5000)} messages)"


def _timeline(rng, n, start=None):
    t = start or dt.datetime(2026, rng.randint(1, 12), rng.randint(1, 27), rng.randint(0, 9), rng.randint(0, 59), rng.randint(0, 59))
    out = []
    for _ in range(n):
        t += dt.timedelta(milliseconds=rng.randint(200, 25000))
        out.append(t)
    return out


# ================================================================================================ final_status
STATUS_SETS = [
    (["RECEIVED", "PICKING", "PACKED", "DISPATCHED", "OUT_FOR_DELIVERY", "DELIVERED", "RETURNED_TO_SENDER", "ON_HOLD"], "ORD", "order"),
    (["QUEUED", "RUNNING", "RETRYING", "FAILED", "SUCCEEDED", "CANCELLED", "PAUSED"], "JOB", "job"),
    (["SUBMITTED", "UNDER_REVIEW", "AWAITING_DOCS", "APPROVED", "DECLINED", "PAID", "WITHDRAWN"], "CLM", "claim"),
    (["CREATED", "AUTHORISED", "CAPTURED", "SETTLED", "REFUNDED", "CHARGEBACK", "VOIDED"], "PAY", "payment"),
    (["OPEN", "TRIAGED", "IN_PROGRESS", "WAITING_ON_CUSTOMER", "RESOLVED", "CLOSED", "REOPENED"], "TKT", "ticket"),
]


def gen_final_status(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    statuses, prefix, noun = rng.choice(STATUS_SETS)
    X = _ids(rng, prefix, 5)
    X2 = _near_dup(rng, X)
    fmt = _log_fmt(rng)
    svc = rng.choice(SERVICES)
    others = set()
    phr = rng.randrange(3)

    def status_msg(e, s):
        return [f"{noun} {e} status -> {s}", f"{noun} {e}: transition to {s}", f"state change {e} => {s}"][phr]

    def other_status():
        e = _fresh_id(rng, prefix, {X, X2})
        others.add(e)
        return status_msg(e, rng.choice(statuses))

    budget = _budget(rng, diff, "final_status")
    t_dummy = dt.datetime(2026, 1, 1, 12, 0, 0)
    fill = _fill_to(rng, lambda: (rng.choice(["INFO", "INFO", "INFO", "DEBUG", "WARN"]), rng.choice(SERVICES), _generic_msg(rng, extra=other_status)),
                    budget, 0, size=lambda l: len(fmt(t_dummy, *l)))
    n_x = {3: 3, 4: 4, 5: 5}[diff]
    x_stat = [rng.choice(statuses) for _ in range(n_x)]
    for i in range(1, n_x):
        while x_stat[i] == x_stat[i - 1]:
            x_stat[i] = rng.choice(statuses)
    final = x_stat[-1]
    fin_frac = rng.uniform(0.33, 0.68)
    specials = []
    for i, s in enumerate(x_stat[:-1]):
        specials.append((rng.uniform(0.05, fin_frac - 0.04) if i else rng.uniform(0.03, 0.2), ("x", s)))
    specials = sorted(specials, key=lambda x: x[0])
    specials.append((fin_frac, ("xfinal", final)))
    x2_final = rng.choice([s for s in statuses if s != final])
    specials += [(rng.uniform(0.1, 0.9), ("x2", rng.choice(statuses))) for _ in range(diff - 1)]
    specials.append((rng.uniform(fin_frac + 0.08, 0.97), ("x2", x2_final)))
    specials.append((rng.uniform(fin_frac + 0.05, 0.95), ("xmention", None)))
    lines, idx = _insert(fill, specials)
    times = _timeline(rng, len(lines))
    out, x_events = [], []
    sp_sorted = sorted(specials, key=lambda x: x[0])
    sp_pos = {i: sp_sorted[k][1] for k, i in enumerate(idx)}
    for i, l in enumerate(lines):
        if i in sp_pos:
            tag, s = sp_pos[i]
            if tag in ("x", "xfinal"):
                out.append(fmt(times[i], "INFO", svc, status_msg(X, s)))
                x_events.append([i, s])
            elif tag == "x2":
                out.append(fmt(times[i], "INFO", svc, status_msg(X2, s)))
            else:
                out.append(fmt(times[i], "INFO", rng.choice(SERVICES), P(rng, f"customer contact logged for {noun} {X} (no status change)",
                                                                         f"note added to {X}: customer asked for an update",
                                                                         f"{noun} {X} viewed by support agent {rng.choice(USERS)}")))
        else:
            lvl, s2, msg = l
            out.append(fmt(times[i], lvl, s2, msg))
    fin_i = x_events[-1][0]
    fr = _char_frac(out, fin_i)
    if not 0.25 <= fr <= 0.75:
        raise Retry("position")
    header = P(rng, f"Excerpt of the {svc} service log (UTC), exported for the incident review. {len(out)} lines.",
               f"Combined application log from the {svc} cluster, UTC timestamps, {len(out)} lines, oldest first.",
               f"Log export ({len(out)} lines, chronological) from {svc} and neighbouring services.")
    text = header + "\n\n" + "\n".join(out)
    _check_size(text, "final_status")
    q = P(rng, f"According to the log, what is the final status of {noun} {X}?", f"What status did {X} end up in by the end of this log?",
          f"Which status is the last one recorded for {noun} {X}?", f"By the last line of the export, what state is {X} in?")
    wrong = [x2_final] + [s for s in statuses if s != final]
    field, gold = choice_field(rng, q, final, wrong, k={3: 5, 4: 6, 5: 7}[diff])
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "status", "x": X, "events": x_events, "gap": None, "opt": keymap}
    hints = {"near_miss": [x2_final, x_stat[-2]] if x_stat[-2] != final else [x2_final], "decisive": [out[fin_i]]}
    parent = item(text, field, gold, "final_status", "logistics" if prefix == "ORD" else "saas_ops", spec, hints)
    res = [parent]
    if want_unknown:
        prev_i = x_events[-2][0]
        lo = max(prev_i + 1, fin_i - rng.randint(3, 25))
        hi = fin_i + rng.randint(3, 25)
        marker = P(rng, "... [{n} lines not retained: log shipper buffer overflow] ...", "<<< {n} log lines missing from the export (collector restart) >>>",
                   "[gap: {n} lines dropped by the log pipeline]")
        # the gap must not swallow other decisive X lines
        if any(lo <= i < hi for i, _ in x_events[:-1]):
            raise Retry("gap overlaps")
        out_u = _gap(out, lo, hi, marker)
        text_u = header + "\n\n" + "\n".join(out_u)
        spec_u = dict(spec, gap=[lo, hi])
        if recheck(spec_u) is not None:
            raise Retry("unknown decided")
        _check_size(text_u, "final_status")
        res.append(item(text_u, field, None, "final_status", parent["domain"], spec_u, {}, child_of=0))
    return res


# ================================================================================================ agreed_date
PRODUCTS = ["washing machine", "sofa", "fridge-freezer", "desk", "wardrobe", "dishwasher", "mattress", "bookcase", "bike", "oven"]
AGENT_FILL = [
    "Let me just pull up the account, bear with me a moment.", "I can see the {p} order here on the system.", "Thanks for holding, the system is a bit slow today.",
    "Could you confirm the first line of the address for me?", "Our {d} team normally deals with that within {n} working days.",
    "I'll add a note so the next person can see what we discussed.", "That's no problem at all.", "Is the {p} going to the same address as last time?",
    "I've sent a text with the tracking link to the number ending {n4}.", "We had a lot of calls about the {p} range this week.",
    "Just so you know, the call may be recorded for training.", "The warranty on the {p} runs for {n} years from delivery.",
    "I can see an older order for a {p2} as well, but that one's completed.", "Our drivers will call about thirty minutes before arriving.",
    "Let me check with the warehouse whether it has left the depot yet.", "I understand, that must be frustrating.",
    "The {p2} you asked about is back in stock next month.", "I'll email you a summary of this call afterwards.",
    "If anything changes you can reply to the confirmation email.", "The two-person team will need clear access to the room.",
    "I'm going to put you on a short hold while I check something.", "I've updated your phone number on the account.",
]
CUST_FILL = [
    "Sure, no rush.", "It's the {p} I ordered a couple of weeks ago.", "My neighbour had the same problem with a {p2}, funnily enough.",
    "Yes, that's right.", "Sorry, the line is a bit noisy here.", "Could you repeat that last bit?", "The reference is on the email somewhere, hang on.",
    "I work from home most days, so that's usually fine.", "Do the drivers take the packaging away?", "We're on the third floor, but there's a lift.",
    "I think my partner placed the order, actually.", "Last time the {p2} arrived with a dent, so I'm a bit nervous.", "Okay, great.",
    "Can I pay the delivery charge by card over the phone?", "Is there a way to get a text on the day?", "I'll be around in the mornings, mostly.",
    "That's fine, thank you.", "Mm-hm.", "Oh, and the old {p2} needs collecting too, if that's possible.", "Let me just grab a pen.",
    "Right, okay, go on.", "The parking outside can be tricky on weekdays.",
]
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _dstr(d, style):
    return [f"{WEEK[d.weekday()] if d.weekday() < 6 else 'Sunday'} {d.day} {d.strftime('%B')}", f"{d.strftime('%A')} the {d.day}{'th' if 10 <= d.day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d.day % 10, 'th')}",
            f"{d.day} {d.strftime('%B')}"][style]


def gen_agreed_date(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    people = People(rng)
    agent, cust = people().split()[0], people()
    prod = rng.choice(PRODUCTS)
    base = dt.date(2026, rng.randint(1, 12), rng.randint(1, 20))
    days = sorted(rng.sample(range(2, 30), 7))
    ds = [base + dt.timedelta(days=x) for x in days]
    ds = [d for d in ds if d.weekday() < 6]
    if len(ds) < 5:
        raise Retry("dates")
    d1, d2, d3, d4, d5 = rng.sample(ds, 5)
    changed = chance(rng, 0.5 if diff == 3 else 0.75)
    final = d4 if changed else d3
    style = rng.randrange(3)
    S = lambda d: _dstr(d, style)  # noqa: E731
    A, C = f"Agent ({agent})", f"Customer ({cust.split()[0]})"
    # decisive block
    block = [
        (A, P(rng, f"For the {prod}, the first slot I have is {S(d1)}.", f"The earliest the {prod} can come is {S(d1)}.")),
        (C, P(rng, f"Hmm, {S(d1)} doesn't work, I'm away that day.", f"I can't do {S(d1)}, sorry, I've got an appointment.")),
        (A, P(rng, f"Okay. I could do {S(d2)} or {S(d3)} instead.", f"Alternatively there's {S(d2)}, or {S(d3)}.")),
        (C, P(rng, f"Let's go with {S(d3)}.", f"{S(d3)} is better for me, please.")),
        (A, P(rng, f"Great, that's booked for {S(d3)}, all-day slot.", f"Done — the {prod} is now booked for {S(d3)}.")),
    ]
    later = []
    if changed:
        later = [
            (C, P(rng, f"Actually, sorry — thinking about it, is there anything on {S(d4)}? {S(d3)} might clash after all.",
                  f"Oh wait, I've just remembered {S(d3)} won't work. Could we try {S(d4)}?")),
            (A, P(rng, f"Let me look... yes, {S(d4)} is free. I've moved it, so it's {S(d4)} now and {S(d3)} is cancelled.",
                  f"No problem, I've changed the booking to {S(d4)}; the {S(d3)} slot is released.")),
        ]
    distract = (C, P(rng, f"By the way, my sister's {rng.choice(PRODUCTS)} is being delivered on {S(d5)}, not by you though.",
                     f"Unrelated, but the plumber's coming on {S(d5)}, so not that day for anything else."))
    budget = _budget(rng, diff, "agreed_date")
    p2s = [p for p in PRODUCTS if p != prod]

    def filler():
        if chance(rng, 0.5):
            return (A, rng.choice(AGENT_FILL).format(p=prod, p2=rng.choice(p2s), d=rng.choice(["delivery", "installation", "returns"]), n=rng.randint(2, 5), n4=rng.randint(1000, 9999)))
        return (C, rng.choice(CUST_FILL).format(p=prod, p2=rng.choice(p2s)))

    fill = _fill_to(rng, filler, budget, 0)
    fin_frac = rng.uniform(0.35, 0.62)
    specials = [(fin_frac - 0.002 * (len(block) - k), ("blk", k)) for k in range(len(block))]
    if changed:
        specials += [(fin_frac + 0.04 + 0.001 * k, ("later", k)) for k in range(len(later))]
    specials.append((rng.uniform(0.7, 0.92), ("dis", 0)))
    lines, idx = _insert(fill, specials)
    sp_sorted = sorted(specials, key=lambda x: x[0])
    sp_pos = {i: sp_sorted[k][1] for k, i in enumerate(idx)}
    t0 = dt.datetime(2026, 1, 1, rng.randint(8, 16), rng.randint(0, 59), 0)
    out, events = [], []
    sec = 0
    for i, l in enumerate(lines):
        if i in sp_pos:
            tag, k = sp_pos[i]
            spk, txt = block[k] if tag == "blk" else (later[k] if tag == "later" else distract)
            if tag == "blk" and k == 4:
                events.append([len(out), "confirm", S(d3)])
            if tag == "later" and k == 1:
                events.append([len(out), "confirm", S(d4)])
        else:
            spk, txt = l
        sec += rng.randint(3, 25)
        out.append(f"[{sec // 60:02d}:{sec % 60:02d}] {spk}: {txt}")
    conf_i = events[-1][0]
    fr = _char_frac(out, conf_i)
    if not 0.25 <= fr <= 0.75:
        raise Retry("position")
    header = P(rng, f"Transcript of a customer call about a {prod} delivery (automatic speech-to-text, lightly cleaned).",
               f"Call recording transcript — delivery booking for a {prod}. Timestamps are minutes:seconds from the start of the call.",
               f"Support call log (transcribed). Topic: arranging delivery of a {prod}.")
    text = header + "\n\n" + "\n".join(out)
    _check_size(text, "agreed_date")
    q = P(rng, f"On which date is the {prod} delivery finally booked at the end of the call?", f"What delivery date for the {prod} was ultimately agreed?",
          f"Which day did the customer end up with for the {prod} delivery?")
    opts = [S(d) for d in (d1, d2, d3, d4, d5)]
    if len({o.lower() for o in opts}) != 5:
        raise Retry("date strings clash")
    field, gold = choice_field(rng, q, S(final), [o for o in opts if o != S(final)])
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "agreed", "events": events, "gap": None, "opt": keymap}
    near = [S(d3)] if changed else [S(d2)]
    parent = item(text, field, gold, "agreed_date", "retail", spec, {"near_miss": near + [S(d5)], "decisive": [out[conf_i]]})
    res = [parent]
    if want_unknown:
        # drop everything from the customer's choice onward through the last confirmation
        first_blk = idx[[k for k, s in enumerate(sp_sorted) if s[1] == ("blk", 3)][0]]
        lo, hi = first_blk, conf_i + 1 + rng.randint(1, 6)
        marker = P(rng, "[... audio dropped for about {n} turns; not transcribed ...]", "<< recording gap: {n} turns missing >>",
                   "[transcription failed for the next {n} turns]")
        out_u = _gap(out, lo, hi, marker)
        text_u = header + "\n\n" + "\n".join(out_u)
        spec_u = dict(spec, gap=[lo, hi], events=[e for e in events])
        if recheck(spec_u) is not None:
            raise Retry("unknown decided")
        _check_size(text_u, "agreed_date")
        res.append(item(text_u, field, None, "agreed_date", "retail", spec_u, {}, child_of=0))
    return res


# ================================================================================================ record_lookup
CARRIERS = ["DPX Freight", "Northline", "Coastal Express", "Hartmann Logistik", "Rapidus", "BlueRoute", "Kite Couriers", "Meridian Haulage"]
REC_STATUS = ["Delivered", "In transit", "Delayed", "Held at customs", "Returned", "Awaiting pickup", "Damaged", "Cancelled"]
REGIONS = ["EU-West", "EU-North", "UK", "US-East", "US-West", "APAC", "LATAM", "MEA"]


def gen_record_lookup(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    people = People(rng)
    X = _ids(rng, "SHP", 6)
    X2 = _near_dup(rng, X)
    used = {X, X2}
    custs = [people() for _ in range(40)]
    budget = _budget(rng, diff, "record_lookup")
    sep = rng.choice([" | ", "; ", ","])

    def row(i=None, vals=None):
        rid = vals["id"] if vals else _fresh_id(rng, "SHP", used, 6)
        used.add(rid)
        v = vals or {"id": rid, "customer": rng.choice(custs), "region": rng.choice(REGIONS), "carrier": rng.choice(CARRIERS),
                     "weight": f"{rng.randint(1, 900)}.{rng.randint(0, 9)}", "status": rng.choice(REC_STATUS),
                     "updated": (dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 360))).isoformat()}
        return v

    def fmt(v):
        return sep.join([v["id"], v["customer"], v["region"], v["carrier"], v["weight"], v["status"], v["updated"]])

    rows = []
    n = 0
    while n < budget:
        v = row()
        rows.append(v)
        n += len(fmt(v)) + 1
    field_name = rng.choice(["carrier", "status"])
    xv = row(vals={"id": X, "customer": rng.choice(custs), "region": rng.choice(REGIONS), "carrier": rng.choice(CARRIERS),
                   "weight": f"{rng.randint(1, 900)}.{rng.randint(0, 9)}", "status": rng.choice(REC_STATUS),
                   "updated": (dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 360))).isoformat()})
    pool = CARRIERS if field_name == "carrier" else REC_STATUS
    x2v = dict(xv, id=X2, **{field_name: rng.choice([p for p in pool if p != xv[field_name]])})
    x2v["customer"] = xv["customer"] if chance(rng, 0.6) else rng.choice(custs)
    pos = rng.uniform(0.32, 0.68)
    ix = int(pos * len(rows))
    rows.insert(ix, xv)
    iy = rng.randrange(len(rows))
    while abs(iy - ix) < 3:
        iy = rng.randrange(len(rows))
    rows.insert(iy, x2v)
    ix = rows.index(xv)
    # amendments at the end: X2 and some other rows, never X
    amend = [(X2, field_name, rng.choice([p for p in pool if p not in (xv[field_name], x2v[field_name])]))]
    for r in rng.sample(rows, 4):
        if r["id"] not in (X, X2):
            f2 = rng.choice(["carrier", "status"])
            amend.append((r["id"], f2, rng.choice(CARRIERS if f2 == "carrier" else REC_STATUS)))
    rng.shuffle(amend)
    header_cols = sep.join(["shipment_id", "customer", "region", "carrier", "weight_kg", "status", "last_updated"])
    title = P(rng, f"Shipment register export ({len(rows)} rows). Columns: {header_cols}",
              f"Carrier allocation report — {len(rows)} shipments, one per line. Fields: {header_cols}",
              f"Operations extract, {len(rows)} records ({header_cols}).")
    body = [fmt(r) for r in rows]
    amend_txt = [P(rng, "Amendments received after the export (these override the rows above):", "Post-export corrections (take precedence over the table):",
                   "Late corrections to the register:")] + [f"- {a}: {f} corrected to {v}" for a, f, v in amend]
    text = title + "\n\n" + "\n".join(body) + "\n\n" + "\n".join(amend_txt)
    fr = (len(title) + 2 + sum(len(l) + 1 for l in body[:ix])) / len(text)
    if not 0.25 <= fr <= 0.75:
        raise Retry("position")
    _check_size(text, "record_lookup")
    qtype = "noul" if chance(rng, 0.35) else "choice"
    fname = {"carrier": "carrier", "status": "status"}[field_name]
    if qtype == "choice":
        q = P(rng, f"Which {fname} is recorded for shipment {X}, taking any corrections into account?",
              f"What is the {fname} of {X} according to this export and its amendments?",
              f"For shipment {X}, which {fname} does the register show once corrections are applied?")
        wrong = [x2v[field_name], amend[0][2] if amend[0][0] == X2 else next(v for a, f, v in amend if a == X2)] + [p for p in pool if p != xv[field_name]]
        field, gold = choice_field(rng, q, xv[field_name], wrong, k={3: 5, 4: 6, 5: 8}[diff])
        keymap = {o["text"]: o["key"] for o in field["options"]}
        spec = {"k": "record", "x": X, "field": field_name, "value": xv[field_name], "qtype": "choice", "opt": keymap, "gap": None,
                "amend": [list(a) for a in amend]}
        near = [x2v[field_name]]
    else:
        ask = xv[field_name] if chance(rng, 0.5) else x2v[field_name]
        q = P(rng, f"Is the {fname} of shipment {X} '{ask}' (after applying corrections)?", f"Does the register, with its amendments, list {X} with {fname} '{ask}'?")
        field = noul_field(q)
        gold = ask == xv[field_name]
        spec = {"k": "record", "x": X, "field": field_name, "value": xv[field_name], "qtype": "noul", "ask": ask, "gap": None,
                "amend": [list(a) for a in amend]}
        near = []
    hints = {"near_miss": near, "decisive": [fmt(xv)]}
    if qtype == "noul":
        hints["neg"] = {"question": P(rng, f"Is it true that the {fname} of shipment {X} is NOT '{ask}' (after corrections)?",
                                      f"Does the register, with amendments, show {X} with a {fname} OTHER than '{ask}'?"), "gold": not gold}
    parent = item(text, field, gold, "record_lookup", "logistics", spec, hints)
    res = [parent]
    if want_unknown:
        lo = max(0, ix - rng.randint(4, 15))
        hi = ix + rng.randint(4, 15)
        if lo <= rows.index(x2v) < hi:
            raise Retry("gap swallows X2")
        marker = P(rng, "[{n} rows could not be exported: checksum error]", "... {n} rows missing from this extract (export timeout) ...",
                   "<{n} records omitted: page failed to load>")
        body_u = _gap(body, lo, hi, marker)
        text_u = title + "\n\n" + "\n".join(body_u) + "\n\n" + "\n".join(amend_txt)
        spec_u = dict(spec, gap=[lo, hi], x_row=ix)
        if recheck(spec_u) is not None:
            raise Retry("unknown decided")
        _check_size(text_u, "record_lookup")
        res.append(item(text_u, field, None, "record_lookup", "logistics", spec_u, {}, child_of=0, unknown_reason="insufficient_evidence"))
    return res


# ================================================================================================ count_after
JOBS = ["invoice-export", "label-print", "fx-sync", "stock-import", "report-build", "email-digest", "payout-batch", "search-reindex"]


def gen_count_after(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    fmt = _log_fmt(rng)
    Wn = rng.randint(2, 19)
    W, W2 = f"worker-{Wn}", f"worker-{Wn + 10}"
    Z = rng.choice(JOBS)
    Z2 = Z + rng.choice(["-v2", "-eu", "-retry", "-legacy"])
    others_jobs = [j for j in JOBS if j != Z]
    budget = _budget(rng, diff, "count_after")

    def extra():
        k = rng.randrange(3)
        if k == 0:
            return ("INFO", f"job {rng.choice(others_jobs)} completed on worker-{rng.choice([x for x in range(1, 40) if x not in (Wn, Wn + 10)])}")
        if k == 1:
            return ("ERROR", f"job {rng.choice(others_jobs)} failed: {rng.choice(['timeout', 'upstream 502', 'lock not acquired', 'bad input row'])}")
        return ("INFO", f"job {Z} completed")

    def filler():
        e = extra() if chance(rng, 0.3) else (rng.choice(["INFO", "INFO", "DEBUG", "WARN"]), _generic_msg(rng))
        return (e[0], rng.choice(SERVICES), e[1])

    t_dummy = dt.datetime(2026, 1, 1, 12, 0, 0)
    fill = _fill_to(rng, filler, budget, 0, size=lambda l: len(fmt(t_dummy, *l)))
    last_r = rng.uniform(0.3, 0.62)
    count = rng.randint(0, 5)
    specials = [(rng.uniform(0.05, last_r - 0.1), ("restart", W)), (last_r, ("restart", W))]
    specials += [(rng.uniform(0.02, last_r - 0.01), ("fail", Z)) for _ in range(rng.randint(1, 4))]
    specials += [(rng.uniform(last_r + 0.01, 0.99), ("fail", Z)) for _ in range(count)]
    specials += [(rng.uniform(last_r + 0.01, 0.99), ("fail", Z2)) for _ in range(rng.randint(1, 3))]
    specials += [(rng.uniform(last_r + 0.05, 0.95), ("restart", W2)) for _ in range(rng.randint(1, 2))]
    lines, idx = _insert(fill, specials)
    sp_sorted = sorted(specials, key=lambda x: x[0])
    sp_pos = {i: sp_sorted[k][1] for k, i in enumerate(idx)}
    times = _timeline(rng, len(lines))
    out, restarts, fails = [], [], []
    svc = rng.choice(SERVICES)
    for i, l in enumerate(lines):
        if i in sp_pos:
            tag, who = sp_pos[i]
            if tag == "restart":
                out.append(fmt(times[i], "WARN", svc, P(rng, f"{who} restarted (exit code 137, OOM)", f"supervisor: restarting {who}", f"{who} process restarted by watchdog")))
                if who == W:
                    restarts.append(len(out) - 1)
            else:
                out.append(fmt(times[i], "ERROR", svc, f"job {who} failed on {W}: " + rng.choice(["timeout after 300s", "connection reset", "upstream 503", "deadlock detected"])))
                if who == Z:
                    fails.append(len(out) - 1)
        else:
            lvl, s2, msg = l
            out.append(fmt(times[i], lvl, s2, msg))
    fr = _char_frac(out, restarts[-1])
    if not 0.25 <= fr <= 0.75:
        raise Retry("position")
    header = P(rng, f"Scheduler and worker log ({len(out)} lines, chronological).", f"Batch platform log export, {len(out)} lines, oldest first.",
               f"Job runner log for the {svc} cluster ({len(out)} lines).")
    text = header + "\n\n" + "\n".join(out)
    _check_size(text, "count_after")
    levels = [f"{k} failure{'s' if k != 1 else ''}" for k in range(6)] + ["6 or more failures"]
    q = P(rng, f"How many times did job {Z} (exactly that job name) fail after the last restart of {W}?",
          f"Counting only failures of job {Z} logged after {W}'s most recent restart, how many are there?",
          f"After {W} was last restarted, how many failures of {Z} does the log show?")
    field = score_field(q, levels)
    gold = sum(1 for f in fails if f > restarts[-1])
    assert gold == count
    spec = {"k": "count", "restarts": restarts, "fails": fails, "gap": None}
    before = sum(1 for f in fails if f > restarts[0])
    parent = item(text, field, gold, "count_after", "saas_ops", spec, {"decisive": [out[restarts[-1]]], "near_miss": [levels[min(6, before)]] if before != count else []})
    res = [parent]
    if want_unknown:
        r = restarts[-1]
        prev = max([f for f in fails if f < r] + [restarts[0]])
        nxt = min([f for f in fails if f > r] + [len(out)])
        lo = max(prev + 1, r - rng.randint(3, 20))
        hi = min(nxt, r + rng.randint(3, 20))
        if hi <= r:
            raise Retry("gap")
        marker = P(rng, "... [{n} lines lost during log rotation] ...", "[{n} lines missing: disk full on the collector]", "<gap of {n} lines in the export>")
        out_u = _gap(out, lo, hi, marker)
        spec_u = dict(spec, gap=[lo, hi])
        if recheck(spec_u) is not None:
            raise Retry("unknown decided")
        _check_size(header + "\n\n" + "\n".join(out_u), "count_after")
        res.append(item(header + "\n\n" + "\n".join(out_u), field, None, "count_after", "saas_ops", spec_u, {}, child_of=0))
    return res


# ================================================================================================ audit_last
SETTINGS = [("db.pool.max_size", "db.pool.max_idle"), ("payments.retry.limit", "payments.retry.limit_ms"), ("cache.ttl_seconds", "cache.ttl_jitter"),
            ("api.rate_limit.per_min", "api.rate_limit.burst"), ("search.shards", "search.shard_replicas"), ("checkout.timeout_ms", "checkout.timeout_retry_ms"),
            ("queue.max_inflight", "queue.max_inflight_per_host"), ("auth.session_minutes", "auth.session_idle_minutes")]
OTHER_SETTINGS = ["log.level", "feature.dark_mode", "email.batch_size", "cdn.purge_on_deploy", "export.max_rows", "metrics.sample_rate", "alerts.pager_threshold",
                  "billing.grace_days", "images.max_px", "sso.enforce", "reports.timezone", "backup.window"]


def gen_audit_last(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    Y, Y2 = rng.choice(SETTINGS)
    users = rng.sample(USERS[1:], {3: 5, 4: 8, 5: 11}[diff])
    fmt_k = rng.randrange(2)

    def line(t, u, s, old, new, action="update"):
        if fmt_k == 0:
            return f"{t:%Y-%m-%d %H:%M:%S} user={u} action={action} setting={s} old={old} new={new}"
        return '{"time":"' + f"{t:%Y-%m-%dT%H:%M:%S}" + f'","actor":"{u}","op":"{action}","key":"{s}","from":"{old}","to":"{new}"' + "}"

    budget = _budget(rng, diff, "audit_last")

    def filler():
        k = rng.random()
        if k < 0.75:
            s = rng.choice(OTHER_SETTINGS)
            return ("chg", rng.choice(users), s)
        return ("view", rng.choice(users), rng.choice(OTHER_SETTINGS + [Y, Y2]))

    t_dummy = dt.datetime(2026, 1, 1, 12, 0, 0)
    fill = _fill_to(rng, filler, budget, 0, size=lambda l: len(line(t_dummy, l[1], l[2], 10, 20, "update" if l[0] == "chg" else "read")))
    dec = rng.uniform(0.33, 0.62)
    inc = rng.uniform(dec + 0.04, min(0.9, dec + 0.25))
    changers = rng.sample(users, 4)
    specials = [(rng.uniform(0.03, dec - 0.05), ("y", changers[1])), (rng.uniform(0.03, dec - 0.02), ("y", changers[2])),
                (dec, ("y", changers[0])), (rng.uniform(dec + 0.005, inc - 0.005), ("y2", changers[3])),
                (inc, ("incident", None)), (rng.uniform(inc + 0.01, 0.98), ("y", changers[3]))]
    lines, idx = _insert(fill, specials)
    sp_sorted = sorted(specials, key=lambda x: x[0])
    sp_pos = {i: sp_sorted[k][1] for k, i in enumerate(idx)}
    times = _timeline(rng, len(lines))
    out, changes, inc_i = [], [], None
    yval = rng.randint(10, 400)
    for i, l in enumerate(lines):
        t = times[i]
        if i in sp_pos:
            tag, u = sp_pos[i]
            if tag in ("y", "y2"):
                s = Y if tag == "y" else Y2
                if tag == "y":
                    new = max(1, yval + rng.choice([-50, -20, 20, 50, 100]))
                    out.append(line(t, u, s, yval, new))
                    yval = new
                else:
                    y2 = rng.randint(1, 60)
                    out.append(line(t, u, s, y2, y2 + rng.choice([5, 10, 15])))
                if tag == "y":
                    changes.append([len(out) - 1, u])
            else:
                inc_i = len(out)
                out.append(P(rng, f"{t:%Y-%m-%d %H:%M:%S} *** INCIDENT OPENED: elevated error rate on checkout (INC-{rng.randint(1000, 9999)}) ***",
                             f"{t:%Y-%m-%d %H:%M:%S} [pager] incident declared — p95 latency breach, all later changes are part of the response"))
        else:
            tag, u, s = l[0], l[1], l[2]
            if tag == "chg":
                out.append(line(t, u, s, rng.randint(0, 50), rng.randint(0, 50)))
            else:
                out.append(line(t, u, s, "-", "-", action="read"))
    gold_u = [u for p, u in changes if p < inc_i][-1]
    dec_i = [p for p, u in changes if p < inc_i][-1]
    fr = _char_frac(out, dec_i)
    if not 0.25 <= fr <= 0.75:
        raise Retry("position")
    header = P(rng, f"Configuration audit trail ({len(out)} entries, oldest first). 'read' entries do not change anything.",
               f"Settings change log exported from the admin console — {len(out)} events in time order; action=read means the value was only viewed.",
               f"Admin audit log, {len(out)} events (chronological). Only update operations change a setting.")
    text = header + "\n\n" + "\n".join(out)
    _check_size(text, "audit_last")
    q = P(rng, f"Who made the last change to {Y} before the incident was declared?", f"Which user last updated the setting {Y} prior to the incident?",
          f"Before the incident started, who was the most recent person to modify {Y}?")
    wrong = [changers[3], changers[1], changers[2]] + [u for u in users if u != gold_u]
    field, gold = choice_field(rng, q, gold_u, [w for w in wrong if w != gold_u], k={3: 5, 4: 7, 5: 10}[diff])
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "audit", "changes": changes, "incident": inc_i, "gap": None, "opt": keymap}
    parent = item(text, field, gold, "audit_last", "saas_ops", spec, {"near_miss": [changers[3]], "decisive": [out[dec_i]]})
    res = [parent]
    if want_unknown:
        prev = max([p for p, u in changes if p < dec_i] + [0])
        lo = max(prev + 1, dec_i - rng.randint(2, 15))
        hi = min(inc_i, dec_i + rng.randint(2, 15))
        if hi <= dec_i:
            raise Retry("gap")
        marker = P(rng, "[{n} audit entries unavailable: archive segment corrupted]", "... {n} events not exported ...", "<{n} entries redacted by retention policy>")
        out_u = _gap(out, lo, hi, marker)
        spec_u = dict(spec, gap=[lo, hi])
        if recheck(spec_u) is not None:
            raise Retry("unknown decided")
        _check_size(header + "\n\n" + "\n".join(out_u), "audit_last")
        res.append(item(header + "\n\n" + "\n".join(out_u), field, None, "audit_last", "saas_ops", spec_u, {}, child_of=0))
    return res


GEN = {"final_status": gen_final_status, "agreed_date": gen_agreed_date, "record_lookup": gen_record_lookup, "count_after": gen_count_after,
       "audit_last": gen_audit_last}


# ================================================================================================ recheck
def recheck(spec: dict):
    """Recompute the answer from the decisive events. A gap (line range [lo, hi)) hides whatever it covers; the answer is None when a
    change inside the gap could alter it."""
    k, gap = spec["k"], spec.get("gap")
    inside = (lambda p: gap is not None and gap[0] <= p < gap[1])
    if k == "status":
        vis = [(p, s) for p, s in spec["events"] if not inside(p)]
        if gap is not None and (not vis or gap[0] > vis[-1][0]):
            return None
        return spec["opt"][max(vis)[1]]
    if k == "agreed":
        vis = [e for e in spec["events"] if not inside(e[0])]
        if gap is not None and (not vis or gap[0] > vis[-1][0]):
            return None
        return spec["opt"][max(vis)[2]]
    if k == "record":
        if gap is not None and gap[0] <= spec["x_row"] < gap[1]:
            return None
        v = spec["value"]
        for a, f, nv in spec["amend"]:
            if a == spec["x"] and f == spec["field"]:
                v = nv
        return spec["opt"][v] if spec["qtype"] == "choice" else (v == spec["ask"])
    if k == "count":
        rs = [r for r in spec["restarts"] if not inside(r)]
        if gap is not None and gap[0] > max(rs):
            return None
        last = max(rs)
        return min(6, sum(1 for f in spec["fails"] if f > last and not inside(f)))
    if k == "audit":
        inc = spec["incident"]
        vis = [(p, u) for p, u in spec["changes"] if p < inc and not inside(p)]
        if gap is not None and gap[0] < inc and (not vis or gap[1] > vis[-1][0]):
            return None
        return spec["opt"][max(vis)[1]]
    raise KeyError(k)


# ================================================================================================ enumerated completions
UNLISTED = "__not_an_option__"


def worlds(spec: dict) -> list:
    """Every answer a completion of the gap can give. A gap (line range [lo, hi)) is a run of hi - lo unseen log lines that may hold
    anything; recheck() returns None exactly when such lines could change the answer. One world when recheck() decides.
      status / agreed / audit: an unseen line can set any status / confirm any date / be a change by anyone -> every option (and a
                               value outside the options);
      record:                  the hidden row holds any value (amendments never touch X) -> every option, resp. ask / not ask;
      count:                   the unseen lines may hold a restart and up to hi - lo failures -> the exact reachable range.
    Used by gen_traps.py to re-derive trap golds (e.g. "is the count something other than L?" is decided when L is unreachable)."""
    v = recheck(spec)
    if v is not None:
        return [v]
    k, (lo, hi) = spec["k"], spec["gap"]
    if k in ("status", "agreed", "audit"):
        return sorted(set(spec["opt"].values())) + [UNLISTED]
    if k == "record":
        if spec["qtype"] == "choice":
            return sorted(set(spec["opt"].values())) + [UNLISTED]
        return [True, False]
    if k == "count":
        n = hi - lo
        inside = (lambda p: lo <= p < hi)
        after = sum(1 for f in spec["fails"] if f >= hi)                        # visible failures after the gap
        vis_r = [r for r in spec["restarts"] if not inside(r)]
        last = max(vis_r)                                                        # recheck() is None only when the gap follows it
        between = sum(1 for f in spec["fails"] if last < f < lo)
        reach = {min(6, after + g) for g in range(n)}                           # a restart inside the gap, then g unseen failures
        reach |= {min(6, after + between + g) for g in range(n + 1)}            # no restart inside the gap
        return sorted(reach)
    raise KeyError(k)
