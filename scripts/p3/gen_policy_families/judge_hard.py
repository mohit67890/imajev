"""judge_hard: rubric-based judging of a candidate reply against explicit criteria, with subtle violations.

A customer message (concrete facts: reference, approved amount, a second amount as a distractor, two questions), a short reply
policy, an explicit rubric, and one or two candidate replies assembled from components whose correctness we control (wrong digit
order in a reference, the order total quoted as the refund, "10 days" for "10 business days", a missed question, a missing review
statement, a request for card details, a word limit counted by whitespace). Gold = the rubric applied in code to the components.

Kinds: rubric_points (score, with a cap for factual errors), rubric_levels (score, mandatory/optional/disqualifying), compliance
(noul), which_failed (choice, exactly one criterion fails), pairwise (choice, highest-ranked separating criterion decides).
Unknown children hide one fact the verdict depends on (the approved amount, the reference or the payment period) and are kept only
when enumerating possible values of that fact changes the verdict.
"""
from __future__ import annotations

import random

from .common import (DOMAINS, DOMAIN_IDS, P, People, chance, choice_field, code, fd, item, money, noul_field, ordered_choice_field,
                     org_name, rand_date, score_field, cap)

FAMILY = "judge_hard"
KINDS = ["rubric_points", "rubric_levels", "compliance", "which_failed", "pairwise"]
# which_failed has no single-fact unknown (its premise "exactly one criterion fails" pins the answer), so the other four kinds
# carry the family's unknown share: 0.15 / 0.85 * 5 / 4
UNKNOWN_P = 0.27  # which_failed never has unknown children; failed unknown draws are re-rolled by the driver
FACTS = ("amount", "ref", "timeline")

# domain scenario: matter noun, reference label, money noun, total noun, second question + answers, reviewer
JD = {
    "retail": ("order", "order number", "refund", "order total", "Can I swap the second item for a larger size instead of returning it?",
               ["You can exchange the second item for a larger size from the Returns page in your account.",
                "A size exchange on the second item is possible; choose 'exchange' on the Returns page."], "our Customer Care Manager"),
    "logistics": ("consignment", "consignment note number", "claim payment", "invoice value of the consignment",
                  "Do I need to keep the damaged pallets for inspection?",
                  ["Please keep the damaged pallets for 14 days in case our assessor needs to inspect them.",
                   "Yes, hold on to the damaged pallets for two weeks so an assessor can inspect them if needed."], "the Claims Manager"),
    "hr": ("pay query", "case number", "back-pay adjustment", "gross monthly salary", "Will the adjustment show on a separate payslip?",
           ["The adjustment will appear as a separate line on your next regular payslip rather than on its own payslip.",
            "It will be itemised on your next regular payslip, not on a separate one."], "the HR Business Partner"),
    "healthcare_admin": ("account", "patient account number", "billing credit", "original invoice", "Will this affect my next appointment booking?",
                         ["Your next appointment booking is not affected and stays exactly as scheduled.",
                          "The credit has no effect on your upcoming appointment, which remains booked."], "the Practice Manager"),
    "insurance": ("claim", "claim number", "settlement", "total loss estimate", "Do I still need to send the repair invoice?",
                  ["Yes, please upload the final repair invoice through the document portal once you have it.",
                   "We still need the final repair invoice; you can upload it on the document portal."], "the Head of Claims"),
    "finance": ("account", "case reference", "fee refund", "total charges on the statement", "Will this affect my credit file?",
                ["The refunded fee has no effect on your credit file.", "Nothing about this refund is reported to credit reference agencies."],
                "our Complaints Manager"),
    "saas_ops": ("workspace", "ticket ID", "service credit", "monthly invoice", "Can the credit be applied to next month's invoice instead of refunded?",
                 ["Yes, we can apply the credit to next month's invoice if you prefer; just reply to confirm.",
                  "The credit can go against your next invoice instead of being refunded, if you let us know."], "the Head of Support"),
    "travel": ("booking", "booking reference", "compensation payment", "total fare", "Can I have it as travel vouchers instead of cash?",
               ["You can choose vouchers instead of cash; they are worth 20% more and valid for a year.",
                "Vouchers are available instead of cash if you prefer; they carry a 20% uplift and last one year."], "the Customer Relations Manager"),
    "public_sector": ("application", "application reference", "fee refund", "total fee paid", "Do I need to reapply for the permit?",
                      ["No reapplication is needed; your original application continues to be processed.",
                       "You do not need to reapply — the existing application stays open."], "the Service Head"),
    "legal": ("matter", "matter number", "invoice credit", "invoice total", "Will the credit show on the next invoice?",
              ["The credit will appear as a separate line on your next invoice.", "You will see the credit itemised on the next invoice we send."],
              "the Managing Partner"),
}
FILLER = ["I'm sorry for the inconvenience this has caused.", "We appreciate your patience while we looked into this.",
          "Thank you for bringing this to our attention.", "I understand how frustrating this must have been.",
          "Our team has checked the details carefully.", "I hope this helps to clear things up.",
          "Please let me know if anything here is unclear.", "It was a pleasure looking into this for you.",
          "We've added a note to your account so colleagues can see the update.", "I really do appreciate you taking the time to write to us.",
          "Getting this right for you matters to the whole team.", "Thank you again for your understanding."]
BANNED = ["For verification, please reply with your full card number and the security code.",
          "Could you send us your online account password so we can check the payment?",
          "To speed things up, please confirm your PIN in your reply.",
          "Please include the full 16-digit card number when you reply so we can match the payment."]
CRITS = ["name", "ref", "amount", "timeline", "q2", "disc", "banned", "length"]


def _labels(rng, x, L):
    return {
        "name": P(rng, f"Greets the {x['party']} by the first name they signed with, spelled exactly",
                  f"Uses the {x['party']}'s signed first name, correctly spelled", f"Addresses the {x['party']} by their first name exactly as they wrote it"),
        "ref": P(rng, f"Quotes the {x['ref_label']} exactly", f"States the correct {x['ref_label']}", f"Includes the {x['ref_label']} without errors"),
        "amount": P(rng, f"States the approved {x['money']} amount correctly", f"Gives the correct approved {x['money']} figure",
                    f"Confirms the exact approved {x['money']} amount"),
        "timeline": P(rng, "Gives the payment period exactly as the policy states it, including the type of days",
                      "States the policy payment period precisely (number and type of days)", "Quotes the payment period from the policy without changing it"),
        "q2": P(rng, f"Answers every question the {x['party']} asked", f"Responds to all of the {x['party']}'s questions",
                f"Leaves none of the {x['party']}'s questions unanswered"),
        "disc": P(rng, f"Tells the {x['party']} they can ask {x['reviewer']} to review the decision", "Includes the review-rights statement required by the policy",
                  "Mentions the right to have the decision reviewed, as the policy requires"),
        "banned": P(rng, "Never asks for a password, PIN or full card number", "Does not request passwords, PINs or full card numbers",
                    "Requests no security credentials (password, PIN or full card number)"),
        "length": P(rng, f"Is at most {L} words long", f"Stays within {L} words", f"Does not exceed {L} words"),
    }


# ================================================================================================ world
def _misspell(rng, name):
    for _ in range(20):
        i = rng.randrange(1, len(name) - 1)
        if rng.random() < 0.5 and name[i] != name[i + 1] and name[i + 1].isalpha():
            w = name[:i] + name[i + 1] + name[i] + name[i + 2:]
        else:
            w = name[:i] + name[i] + name[i:]
        if w != name and w.lower() != name.lower():
            return w
    raise ValueError("cannot misspell")


def _transpose_digits(rng, s):
    idx = [i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]]
    if not idx:
        raise ValueError("no transposable digits")
    i = rng.choice(idx)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _world(rng, difficulty):
    d_id = rng.choice(DOMAIN_IDS)
    dom = DOMAINS[d_id]
    thing, ref_label, money_n, total_n, q2q, q2a, reviewer = JD[d_id]
    people = People(rng)
    cust = people()
    first, last = cust.split(" ", 1)
    if len(first) < 4 or not first.isalpha():
        raise ValueError("short or hyphenated name")
    sym = rng.choice(["€", "£", "$", "A$", "C$", "NZ$"])
    amt = rng.randint(30, 900) + rng.choice([0, 0.5, 0.25, 0.99, 0.4])
    total = amt + rng.randint(15, 400) + rng.choice([0, 0.5, 0.75])
    amount = money(amt, sym, cents=True)
    x = {"d_id": d_id, "dom": dom, "thing": thing, "ref_label": ref_label, "money": money_n, "total_n": total_n, "q2q": q2q, "q2a": q2a,
         "reviewer": reviewer, "party": dom["party"], "first": first, "last": last, "ref": code(rng, dom["ref"], rng.choice([5, 6])),
         "amount": amount, "total": money(total, sym, cents=True), "N": rng.choice([3, 5, 7, 10, 14]), "sym": sym, "agent": people(),
         "dept": rng.choice(dom["depts"]), "org": org_name(rng, dom), "date": fd(rand_date(rng), rng.randrange(5)), "style": rng.randrange(3)}
    if len(set(x["ref"][x["ref"].index("-") + 1:])) < 2:
        raise ValueError("degenerate ref")
    return x


def _truth(x):
    return {"name": x["first"], "ref": x["ref"], "amount": x["amount"], "N": x["N"]}


def _tl_ok(tl, N):
    return tl is not None and tl[0] == "n" and tl[1] == N and tl[2] == "business"


def _check(c, t, comp, words, L):
    if c == "name":
        return comp["name"] == t["name"]
    if c == "ref":
        return comp["ref"] == t["ref"]
    if c == "amount":
        return comp["amount"] == t["amount"]
    if c == "timeline":
        return _tl_ok(comp["tl"], t["N"])
    if c == "q2":
        return comp["q2"]
    if c == "disc":
        return comp["disc"]
    if c == "banned":
        return not comp["banned"]
    if c == "length":
        return words <= L
    raise KeyError(c)


def _factual_error(t, comp):
    return ((comp["ref"] is not None and comp["ref"] != t["ref"]) or (comp["amount"] is not None and comp["amount"] != t["amount"])
            or (comp["tl"] is not None and not _tl_ok(comp["tl"], t["N"])))


# ================================================================================================ replies
def _comp_ok(x):
    return {"name": x["first"], "ref": x["ref"], "amount": x["amount"], "tl": ("n", x["N"], "business"), "q2": True, "disc": True,
            "banned": False, "filler": 0}


def _break(rng, x, comp, c, subtle):
    comp = dict(comp)
    if c == "name":
        comp["name"] = _misspell(rng, x["first"]) if subtle else None
    elif c == "ref":
        comp["ref"] = _transpose_digits(rng, x["ref"]) if subtle else None
    elif c == "amount":
        if subtle:
            comp["amount"] = rng.choice([x["total"], x["total"], _transpose_digits(rng, x["amount"])])
            if comp["amount"] == x["amount"]:
                raise ValueError("amount unchanged")
        else:
            comp["amount"] = None
    elif c == "timeline":
        if subtle:
            comp["tl"] = rng.choice([("n", x["N"], ""), ("n", x["N"], ""), ("n", x["N"] + rng.choice([-2, 2, 3]), "business")])
        else:
            comp["tl"] = rng.choice([("promise",), None])
    elif c == "q2":
        comp["q2"] = False
    elif c == "disc":
        comp["disc"] = False
    elif c == "banned":
        comp["banned"] = True
    elif c == "length":
        comp["_long"] = True
    return comp


def _render_reply(rng, x, comp) -> str:
    th = f"your {x['thing']}"
    g = (P(rng, f"Dear {comp['name']},", f"Hi {comp['name']},", f"Hello {comp['name']},") if comp["name"] else
         P(rng, f"Dear {x['party']},", "Hello,", "Hi there,", "Good afternoon,"))
    ack = (P(rng, f"Thank you for your message about {th} {comp['ref']}.", f"Thanks for getting in touch regarding {th} {comp['ref']}.",
             f"I've looked into {th} {comp['ref']} for you.", f"I have now reviewed {th}, reference {comp['ref']}.") if comp["ref"] else
           P(rng, f"Thank you for your message about {th}.", f"Thanks for getting in touch regarding {th}.", f"I've looked into {th} for you."))
    body = []
    if comp["amount"]:
        body.append(P(rng, f"I can confirm that your {x['money']} of {comp['amount']} has been approved.", f"The approved {x['money']} is {comp['amount']}.",
                      f"We have authorised {an(x['money'])} of {comp['amount']}.", f"Your {x['money']} for {comp['amount']} is confirmed."))
    else:
        body.append(P(rng, f"I can confirm that your {x['money']} has been approved.", f"Your {x['money']} is confirmed.",
                      f"We have authorised the {x['money']}."))
    tl = comp["tl"]
    if tl is not None:
        if tl[0] == "promise":
            body.append(P(rng, "You will have it by tomorrow at the latest.", "It will be with you within 24 hours.", "Expect it in your account by tomorrow."))
        else:
            span = f"{tl[1]} business days" if tl[2] == "business" else f"{tl[1]} days"
            body.append(P(rng, f"It will be paid within {span} of approval.", f"You should receive it within {span}.",
                          f"Please allow {span} for it to arrive.", f"It should reach you within {span} from today's approval."))
    if comp["q2"]:
        body.append(rng.choice(x["q2a"]))
    fill = rng.sample(FILLER, comp["filler"])
    for f in fill:
        body.insert(rng.randrange(len(body) + 1), f)
    if comp["banned"]:
        body.insert(rng.randrange(1, len(body) + 1), rng.choice(BANNED))
    if comp["disc"]:
        who = x["reviewer"]
        body.append(P(rng, f"If you are unhappy with how we handled this, you can ask {who} to review it.",
                      f"You have the right to ask for this decision to be reviewed by {who}.",
                      f"Should you disagree with the outcome, a review by {who} is available on request."))
    close = P(rng, "Kind regards,", "Best wishes,", "Warm regards,", "With thanks,")
    return "\n".join([g, "", ack + " " + " ".join(body), "", close, f"{x['agent']}, {x['dept']}"])


def an(w: str) -> str:
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def _nwords(text):
    return len(text.split())


def _customer_msg(rng, x, hide):
    greet = P(rng, "Hello,", "Hi there,", "Good morning,", "Hi team,", "To whom it may concern,")
    th = x["thing"]
    ref_part = (P(rng, f"I'm writing about my {th}; I don't have the {x['ref_label']} to hand, sorry.",
                  f"This is about my {th} (I can't find the {x['ref_label']} right now).") if hide == "ref" else
                P(rng, f"I'm writing about my {th}, {x['ref_label']} {x['ref']}.", f"This concerns {th} {x['ref']}.",
                  f"My {x['ref_label']} is {x['ref']}."))
    amt_part = (P(rng, f"Your email of {x['date']} said the {x['money']} had been approved (the {x['total_n']} was {x['total']}), but it didn't give the amount.",
                  f"On {x['date']} I was told {an(x['money'])} was approved; the {x['total_n']} had been {x['total']}.") if hide == "amount" else
                P(rng, f"Your email of {x['date']} said {an(x['money'])} of {x['amount']} had been approved (the {x['total_n']} was {x['total']}).",
                  f"On {x['date']} you approved {an(x['money'])} of {x['amount']} against {an(x['total_n'])} of {x['total']}.",
                  f"I was told on {x['date']} that {x['amount']} would come back to me as {an(x['money'])}; for reference the {x['total_n']} was {x['total']}."))
    q1 = P(rng, f"Could you tell me when the {x['money']} will actually reach me?", f"When should I expect the {x['money']} to arrive?",
           f"How long will the {x['money']} take to come through?")
    sign = P(rng, "Thanks,", "Many thanks,", "Best,", "Regards,")
    return f"{greet}\n{ref_part} {amt_part} {q1} {x['q2q']}\n{sign}\n{x['first']} {x['last']}"


def _policy_lines(rng, x, hide):
    M = cap(x["money"])
    tl = (P(rng, f"{M}s are paid within the period set out in Schedule 3 (not included in this pack).",
            f"The payment period for {x['money']}s is in the payments annex, which is not attached here.") if hide == "timeline" else
          P(rng, f"{M}s are paid within {x['N']} business days of approval.", f"Payment of an approved {x['money']} takes up to {x['N']} business days.",
            f"We pay approved {x['money']}s within {x['N']} business days."))
    lines = [tl,
             P(rng, "Replies must never ask for passwords, PINs or full card numbers.", "Staff must not request security credentials of any kind by email.",
               "Never ask a customer for a password, PIN or a full card number."),
             P(rng, f"Every decision letter must tell the {x['party']} they can ask {x['reviewer']} for a review.",
               f"Replies must include the right to a review by {x['reviewer']}."),
             P(rng, "Office hours are 08:30 to 17:30, Monday to Friday.", "Telephone lines are open 8am to 6pm on weekdays.",
               "Live chat is staffed between 09:00 and 20:00.")]
    rng.shuffle(lines)
    return lines


# ================================================================================================ scoring
def _points(crits, t, comp, words, L, capv):
    s = sum(_check(c, t, comp, words, L) for c in crits)
    return min(s, capv) if _factual_error(t, comp) else s


def _levels(M, O, t, comp, words, L):
    if comp["banned"]:
        return 0
    fm = sum(not _check(c, t, comp, words, L) for c in M)
    if fm >= 2:
        return 1
    if fm == 1:
        return 2
    return 3 if any(not _check(c, t, comp, words, L) for c in O) else 4


def _pair(ranked, t, ca, wa, cb, wb, L):
    for c in ranked:
        a, b = _check(c, t, ca, wa, L), _check(c, t, cb, wb, L)
        if a != b:
            return "A" if a else "B"
    return "equal"


def _pick_L(rng, wlist, wants, margin):
    """A word limit (multiple of 5) such that each reply passes/fails as wanted, with |words - L| >= margin."""
    for L in range(20, 400, 5):
        ok = True
        for w, want in zip(wlist, wants):
            if want and not (w <= L - margin):
                ok = False
            if not want and not (w >= L + margin):
                ok = False
        if ok:
            return L
    raise ValueError("no word limit fits")


# ================================================================================================ generate
def _build_reply(rng, x, fails, subtle_p, difficulty):
    comp = _comp_ok(x)
    comp["filler"] = rng.randint(0, 2) + (2 if "length" in fails else 0)
    for c in fails:
        comp = _break(rng, x, comp, c, chance(rng, subtle_p))
    return comp


def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    x = _world(rng, difficulty)
    subtle_p = {3: 0.5, 4: 0.75, 5: 0.9}[difficulty]
    margin = {3: 8, 4: 4, 5: 2}[difficulty]
    k = {3: 5, 4: 6, 5: 7}[difficulty]
    t = _truth(x)
    crits = rng.sample(CRITS, k)
    if kind == "which_failed":
        want_unknown = False
    if want_unknown and not any(c in crits for c in FACTS):
        crits[-1] = rng.choice(FACTS)
    capv = rng.choice([1, 2, 3])
    M = O = ranked = None
    if kind == "rubric_points":
        nf = rng.choice([0, 1, 1, 2, 2, 3])
        fails = rng.sample(crits, nf)
        if chance(rng, 0.35):  # a factual error outside the listed criteria still triggers the cap
            extra = [c for c in ("ref", "amount", "timeline") if c not in crits]
            if extra:
                fails.append(rng.choice(extra))
        comps = [_build_reply(rng, x, fails, 0.95 if any(c not in crits for c in fails) else subtle_p, difficulty)]
    elif kind == "rubric_levels":
        pool = [c for c in crits if c != "banned"]
        nm = rng.randint(3, min(4, len(pool) - 1))
        M, O = pool[:nm], pool[nm:]
        target = rng.choice([1, 2, 2, 3, 3, 4] if want_unknown else [0, 1, 2, 2, 3, 3, 4])
        fails = {0: ["banned"] + rng.sample(pool, rng.randint(0, 1)), 1: rng.sample(M, 2) + rng.sample(O, rng.randint(0, 1)),
                 2: rng.sample(M, 1) + rng.sample(O, rng.randint(0, len(O))), 3: rng.sample(O, rng.randint(1, len(O))), 4: []}[target]
        crits = M + O + ["banned"]
        comps = [_build_reply(rng, x, fails, subtle_p, difficulty)]
    elif kind == "compliance":
        fails = [] if chance(rng, 0.45) else [rng.choice(crits)]
        comps = [_build_reply(rng, x, fails, max(subtle_p, 0.8), difficulty)]
    elif kind == "which_failed":
        fails = [rng.choice(crits)]
        comps = [_build_reply(rng, x, fails, subtle_p, difficulty)]
    else:  # pairwise
        ranked = crits[: rng.randint(4, 6)]
        crits = ranked
        i = rng.randrange(len(ranked) - 1)  # deciding criterion (not the last, so lower ones can favour the loser)
        fact_i = [j for j in range(len(ranked) - 1) if ranked[j] in FACTS]
        if want_unknown and fact_i:
            i = rng.choice(fact_i)
        lower = ranked[i + 1:]
        win_f = rng.sample(lower, min(len(lower), rng.randint(1, 2)))
        lose_f = [ranked[i]]
        common = rng.sample(ranked[:i], rng.randint(0, min(1, i))) if i else []
        comps = [_build_reply(rng, x, win_f + common, subtle_p, difficulty), _build_reply(rng, x, lose_f + common, subtle_p, difficulty)]
        comps[1]["filler"] += rng.randint(1, 2)  # the losing reply reads more polished
        if chance(rng, 0.5):
            comps.reverse()
    # render, fix word limit
    texts = [_render_reply(random.Random(rng.random()), x, c) for c in comps]
    words = [_nwords(tx) for tx in texts]
    L = _pick_L(rng, words, [not c.get("_long") for c in comps], margin) if "length" in crits else rng.choice([150, 180, 200, 250])
    labels = _labels(rng, x, L)
    for c in comps:
        c.pop("_long", None)

    def verdict(tt):
        if kind == "rubric_points":
            return _points(crits, tt, comps[0], words[0], L, capv)
        if kind == "rubric_levels":
            return _levels(M, O, tt, comps[0], words[0], L)
        if kind == "compliance":
            return all(_check(c, tt, comps[0], words[0], L) for c in crits)
        if kind == "which_failed":
            f = [c for c in crits if not _check(c, tt, comps[0], words[0], L)]
            return f[0] if len(f) == 1 else "invalid"
        return _pair(ranked, tt, comps[0], words[0], comps[1], words[1], L)
    gold_v = verdict(t)
    if kind == "which_failed" and gold_v == "invalid":
        raise ValueError("not exactly one failure")
    if kind == "pairwise" and gold_v == "equal":
        raise ValueError("replies tie")

    def state_for(hide):
        rr = random.Random(f"{x['ref']}|{x['agent']}|{hide}")
        msg = _customer_msg(rr, x, hide)
        pol = _policy_lines(rr, x, hide)
        wc = P(rr, "Word counts cover the whole reply between the dashed lines, greeting and sign-off included; a word is any run of characters between spaces.",
               "To count words, split the reply (everything between the dashed lines, sign-off included) on whitespace.")
        rub = []
        if kind == "rubric_points":
            rub.append(P(rr, "Award one point for each criterion the reply meets:", "Score one point per criterion met:", "Criteria (1 point each):"))
            rub += [f"- {labels[c]}" for c in crits]
            rub.append(P(rr, f"Cap: if the reply states any wrong fact (a wrong reference, a wrong amount or a wrong payment period), the score cannot exceed {capv}, however many criteria are met.",
                         f"Factual-error cap: a reply that states a wrong reference, amount or payment period scores at most {capv}, whether or not that item is a listed criterion."))
        elif kind == "rubric_levels":
            rub.append("Mandatory criteria:")
            rub += [f"- {labels[c]}" for c in M]
            rub.append("Optional criteria:")
            rub += [f"- {labels[c]}" for c in O]
            rub.append(f"Disqualifying: the reply asks for a password, PIN or full card number.")
            rub.append("Levels: 4 = every criterion met; 3 = all mandatory met, at least one optional missed; 2 = exactly one mandatory missed; "
                       "1 = two or more mandatory missed; 0 = disqualifying content present (overrides everything else).")
        elif kind in ("compliance", "which_failed"):
            rub.append(P(rr, "Quality criteria for this reply:", "The reply must meet all of these criteria:", "Checklist:"))
            rub += [f"- {labels[c]}" for c in crits]
        else:
            rub.append(P(rr, "Criteria in priority order (1 = most important):", "Ranked criteria (highest priority first):"))
            rub += [f"{j + 1}. {labels[c]}" for j, c in enumerate(ranked)]
            rub.append(P(rr, "Prefer the reply that meets the highest-ranked criterion the other reply fails; lower-ranked criteria only matter when all higher ones are tied. If they tie on every criterion, they are equally good.",
                         "Compare the replies criterion by criterion from the top: the first criterion that one meets and the other fails decides; if none separates them, they are equally good."))
        if "length" in crits:
            rub.append(wc)
        head = f"{x['org']} — {x['dept']}"
        parts = [head, "", P(rr, "Customer message", f"Message from the {x['party']}", "Incoming message") + f" ({x['date']}):", msg, "",
                 P(rr, "Reply policy", "Policy for replies", "Relevant policy") + ":"] + [f"- {p}" for p in pol] + ["", P(rr, "Rubric", "Grading rubric", "Assessment criteria") + ":"] + rub + [""]
        if kind == "pairwise":
            parts += ["Reply A:", "-" * 12, texts[0], "-" * 12, "", "Reply B:", "-" * 12, texts[1], "-" * 12]
        else:
            parts += [P(rr, f"Draft reply by {x['agent']}:", "Candidate reply:", "Reply to assess:"), "-" * 12, texts[0], "-" * 12]
        return "\n".join(parts)

    # question field
    who = x["party"]
    if kind == "rubric_points":
        q = P(rng, "What score does the reply earn under the rubric?", "How many points does the draft reply score, after any cap?",
              "Applying the rubric, including the cap, what is the reply's score?")
        field = score_field(q, [f"{v} point{'s' if v != 1 else ''}" for v in range(len(crits) + 1)])
        gold = gold_v
        hints = {}
    elif kind == "rubric_levels":
        q = P(rng, "Which level does the reply reach?", "What level should the assessor give this reply?", "Under the rubric, what is the reply's level?")
        field = score_field(q, ["0: disqualifying content", "1: two or more mandatory criteria missed", "2: exactly one mandatory criterion missed",
                                "3: all mandatory met, an optional one missed", "4: every criterion met"])
        gold = gold_v
        hints = {}
    elif kind == "compliance":
        q = P(rng, "Does the reply meet every criterion in the checklist?", f"Is the draft reply ready to send to the {who} under the criteria?",
              "Does the candidate reply comply with all the criteria?")
        field = noul_field(q)
        gold = gold_v
        hints = {"neg": {"question": P(rng, "Does the reply fail at least one criterion in the checklist?", "Does the draft breach any of the criteria?"),
                         "gold": not gold_v}}
    elif kind == "which_failed":
        q = P(rng, "Exactly one criterion is not met. Which one?", "Which criterion does the reply fail?", "The reply misses one criterion. Which?")
        opts = [labels[c] for c in crits]
        field, gold = ordered_choice_field(q, opts, crits.index(gold_v))
        hints = {}
    else:
        q = P(rng, "Which reply better satisfies the rubric?", "Under the ranked criteria, which reply should be sent?",
              "Which of the two replies does the rubric prefer?")
        texts_o = ["Reply A", "Reply B", "Both replies are equally good under the rubric"]
        field, gold = ordered_choice_field(q, texts_o, 0 if gold_v == "A" else 1)
        hints = {}
    state = state_for(None)
    dec = []
    for c in comps:
        for key in ("amount", "ref", "name"):
            if c[key] and c[key] != t[key] and c[key] in state:
                dec.append(c[key])
    hints["decisive"] = sorted(set(dec))
    spec = {"kind": kind, "truth": t, "crits": crits, "M": M, "O": O, "ranked": ranked, "capv": capv, "L": L,
            "replies": [{**{k2: (list(v) if isinstance(v, tuple) else v) for k2, v in c.items() if k2 != "filler"}, "words": w} for c, w in zip(comps, words)]}
    items = [item(state, field, gold, kind, x["d_id"], spec, hints)]
    if want_unknown:
        hs = ["amount", "ref", "timeline"]
        rng.shuffle(hs)
        for h in hs:
            key = "N" if h == "timeline" else h
            if key == "N":
                vals = {c["tl"][1] for c in comps if c["tl"] and c["tl"][0] == "n"} | {x["N"], 99}
            else:
                vals = {c[key] for c in comps if c[key]} | {t[key], "__other__"}
            outs = {repr(verdict({**t, key: v})) for v in vals}
            outs.discard(repr("invalid"))
            outs.discard(repr("equal"))
            if len(outs) >= 2:
                spec2 = {**spec, "truth": {**t, key: None}}
                items.append(item(state_for(h), dict(field), None, kind, x["d_id"], spec2, {}, child_of=0))
                break
        else:
            raise ValueError("no decisive hidden fact")
    return items


# ================================================================================================ independent re-check
def recheck(spec):
    """Re-derive the verdict from the spec. rubric_points / rubric_levels -> int; compliance -> bool; which_failed -> index into crits;
    pairwise -> 0 (A) / 1 (B); None when a fact the verdict needs is hidden."""
    t = spec["truth"]
    if any(v is None for v in t.values()):
        return None
    L = spec["L"]

    def ok(c, r):
        tl = r["tl"]
        return {"name": r["name"] == t["name"], "ref": r["ref"] == t["ref"], "amount": r["amount"] == t["amount"],
                "timeline": bool(tl) and list(tl) == ["n", t["N"], "business"], "q2": r["q2"] is True, "disc": r["disc"] is True,
                "banned": r["banned"] is False, "length": r["words"] <= L}[c]

    def wrong_fact(r):
        tl = r["tl"]
        return ((r["ref"] not in (None, t["ref"])) or (r["amount"] not in (None, t["amount"])) or
                (tl is not None and list(tl) != ["n", t["N"], "business"]))
    k = spec["kind"]
    r0 = spec["replies"][0]
    if k == "rubric_points":
        s = len([c for c in spec["crits"] if ok(c, r0)])
        return s if not wrong_fact(r0) else min(s, spec["capv"])
    if k == "rubric_levels":
        if r0["banned"]:
            return 0
        miss_m = len([c for c in spec["M"] if not ok(c, r0)])
        if miss_m:
            return 1 if miss_m > 1 else 2
        return 4 if all(ok(c, r0) for c in spec["O"]) else 3
    if k == "compliance":
        return all(ok(c, r0) for c in spec["crits"])
    if k == "which_failed":
        bad = [i for i, c in enumerate(spec["crits"]) if not ok(c, r0)]
        return bad[0] if len(bad) == 1 else "invalid"
    ra, rb = spec["replies"]
    for c in spec["ranked"]:
        if ok(c, ra) != ok(c, rb):
            return 0 if ok(c, ra) else 1
    return "equal"


# ================================================================================================ enumerated completions
def worlds(spec) -> list:
    """recheck() over every completion of the hidden truth fact (None in spec["truth"]): the values the replies state for it plus
    one value no reply states ("__other__"; 99 for the timeline), as the unknown proof enumerates them. One world when nothing is
    hidden. Used by gen_traps.py to re-derive trap golds."""
    import itertools
    t = spec["truth"]
    miss = [k for k, v in t.items() if v is None]
    doms = []
    for key in miss:
        if key == "N":
            vals = {r["tl"][1] for r in spec["replies"] if r.get("tl") and r["tl"][0] == "n"} | {99}
        else:
            vals = {r[key] for r in spec["replies"] if r.get(key)} | {"__other__"}
        doms.append(sorted(vals, key=repr))
    return [recheck({**spec, "truth": {**t, **dict(zip(miss, combo))}}) for combo in itertools.product(*doms)]
