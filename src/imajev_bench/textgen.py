"""Programmatic text-track items for imajev-bench v2; every answer is computed, never judged.

Families: numerical_reconciliation, policy_precedence, missing_conflicting_evidence. Unknown answers
arise only from explicit structure (an undefined tie, conflicting sources with no authority), and the
wording never mentions unknown or abstention.
"""
from __future__ import annotations

import random

GOODS = ["printer paper", "stapler", "desk lamp", "USB cable", "notebook", "marker set", "cable ties", "label tape"]
DESTINATIONS = ["shelf_a", "shelf_b", "cold_room", "returns_bay"]


def reconciliation(rng, iid):
    lines = [{"item": g, "quantity": rng.randint(1, 6), "unit_price": round(rng.randrange(80, 2500) / 100, 2)}
             for g in rng.sample(GOODS, rng.randint(2, 4))]
    true_total = round(sum(x["quantity"] * x["unit_price"] for x in lines), 2)
    matches = rng.random() < 0.5
    stated = true_total if matches else round(true_total + rng.choice([-1, 1]) * rng.choice([0.1, 1.0, 2.5, 10.0]), 2)
    return {"id": iid, "track": "text", "family": "numerical_reconciliation",
            "state": {"invoice_lines": lines, "stated_total": stated,
                      "rule": "The invoice passes if and only if the stated total equals the sum of quantity times unit price."},
            "field": {"id": "decision", "type": "boolean", "question": "Does the invoice pass the check?"},
            "draft_gold": matches, "evidence": f"computed total {true_total:.2f}; stated {stated:.2f}"}


def precedence(rng, iid):
    attrs = {"fragile": rng.random() < 0.5, "weight_kg": round(rng.uniform(0.5, 12), 1), "chilled": rng.random() < 0.4}
    conditions = [("item is fragile", lambda a: a["fragile"]), ("item weighs under 5 kg", lambda a: a["weight_kg"] < 5),
                  ("item is chilled", lambda a: a["chilled"]), ("item weighs 5 kg or more", lambda a: a["weight_kg"] >= 5)]
    chosen = rng.sample(conditions, 3)
    rules = [{"priority": rng.randint(1, 3), "condition": text, "action": dest}
             for (text, _), dest in zip(chosen, rng.sample(DESTINATIONS, 3))]
    matching = [r for r, (_, test) in zip(rules, chosen) if test(attrs)]
    if matching:
        best = min(r["priority"] for r in matching)
        winners = {r["action"] for r in matching if r["priority"] == best}
        gold = winners.pop() if len(winners) == 1 else None
    else:
        gold = "general_store"
    options = sorted({r["action"] for r in rules} | {"general_store"})
    rng.shuffle(options)
    return {"id": iid, "track": "text", "family": "policy_precedence",
            "state": {"policy": "Apply the matching rule with the smallest priority number. When none of the rules match, send "
                                "the item to general_store. The policy defines no tie-breaker between matching rules "
                                "with the same priority.", "rules": rules, "item": attrs},
            "field": {"id": "decision", "type": "choice", "question": "Which destination does the policy determine?",
                      "options": [{"value": o} for o in options]},
            "draft_gold": gold, "evidence": f"matching rules: {[(r['priority'], r['action']) for r in matching]}"}


def conflicting(rng, iid):
    def level(marks):
        return 1 if marks <= 2 else 2 if marks <= 5 else 3
    first = rng.randint(0, 8)
    agree = rng.random() < 0.45
    second = first if agree else rng.choice([m for m in range(0, 9) if level(m) != level(first)])
    authority = rng.choice([None, None, "morning sheet", "evening sheet"]) if not agree else None
    reports = [{"source": "morning sheet", "marks": first}, {"source": "evening sheet", "marks": second}]
    if agree:
        gold = level(first)
    elif authority:
        gold = level(first if authority == "morning sheet" else second)
    else:
        gold = None
    return {"id": iid, "track": "text", "family": "missing_conflicting_evidence",
            "state": {"rubric": "Level 1 for 0-2 marks, level 2 for 3-5 marks, level 3 for 6 or more marks. If the "
                                "reports disagree on the level, only a report named as authoritative settles it.",
                      "reports": reports, "authoritative_source": authority},
            "field": {"id": "decision", "type": "ordinal", "question": "Which level applies?",
                      "levels": [{"value": 1, "description": "0-2 marks"}, {"value": 2, "description": "3-5 marks"},
                                 {"value": 3, "description": "6 or more marks"}]},
            "draft_gold": gold, "evidence": f"reports {first}/{second}, authority {authority}"}


FAMILIES = (reconciliation, precedence, conflicting)


def text_items(count: int, seed: int) -> list[dict]:
    rng = random.Random(f"imajev-bench-v2-text\0{seed}")
    items = []
    for i in range(count):
        item = FAMILIES[i % len(FAMILIES)](rng, f"text-{seed:02d}-{i:04d}")
        item["construction"] = {"truth_source": "programmatic", "truth": item["draft_gold"]}
        items.append(item)
    return items


# --- hard text families -------------------------------------------------------------------------------
import datetime as _dt
from decimal import ROUND_HALF_UP, Decimal


def _money_round(x):
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def reconciliation_hard(rng, iid):
    lines = [{"item": g, "quantity": rng.randint(1, 9), "unit_price": f"{rng.randrange(95, 4999) / 100:.2f}"}
             for g in rng.sample(GOODS, rng.randint(3, 5))]
    discounted = rng.randrange(len(lines))
    percent = rng.choice([5, 10, 12.5, 15, 20])
    tax = rng.choice([0, 5, 8, 20])
    subtotal = sum((Decimal(l["unit_price"]) * l["quantity"] * (Decimal(1) - Decimal(str(percent)) / 100 if i == discounted else 1))
                   for i, l in enumerate(lines))
    total = _money_round(_money_round(subtotal) * (1 + Decimal(tax) / 100))
    matches = rng.random() < 0.5
    stated = total if matches else total + Decimal(rng.choice(["0.01", "-0.01", "0.10", "-1.00", "2.50"]))
    return {"id": iid, "track": "text", "family": "numerical_reconciliation",
            "state": {"invoice_lines": lines, "line_discount": {"item": lines[discounted]["item"], "percent": percent},
                      "tax_percent": tax, "stated_total": f"{stated:.2f}",
                      "rule": "Apply the line discount to that line only, sum all lines, round the subtotal to cents "
                              "(half up), add tax on the rounded subtotal, and round again to cents. The invoice passes "
                              "if and only if the stated total equals that result."},
            "field": {"id": "decision", "type": "boolean", "question": "Does the invoice pass the check?"},
            "draft_gold": matches, "evidence": f"computed total {total}; stated {stated:.2f}"}


def precedence_hard(rng, iid):
    attrs = {"fragile": rng.random() < 0.5, "weight_kg": round(rng.uniform(0.5, 12), 1), "chilled": rng.random() < 0.4,
             "express": rng.random() < 0.4}
    conditions = [("item is fragile", lambda a: a["fragile"]), ("item weighs under 5 kg", lambda a: a["weight_kg"] < 5),
                  ("item is chilled", lambda a: a["chilled"]), ("item weighs 5 kg or more", lambda a: a["weight_kg"] >= 5),
                  ("item is marked express", lambda a: a["express"])]
    exceptions = [("unless the item is chilled", lambda a: a["chilled"]), ("unless the item is marked express", lambda a: a["express"]),
                  ("unless the item weighs 8 kg or more", lambda a: a["weight_kg"] >= 8), (None, lambda a: False)]
    chosen = rng.sample(conditions, 4)
    rules = []
    for (text, test), dest in zip(chosen, rng.choices(DESTINATIONS, k=4)):
        exc_text, exc_test = rng.choice(exceptions)
        rules.append(({"priority": rng.randint(1, 3), "condition": text + (f", {exc_text}" if exc_text else ""),
                       "action": dest}, lambda a, t=test, e=exc_test: t(a) and not e(a)))
    matching = [r for r, applies in rules if applies(attrs)]
    if matching:
        best = min(r["priority"] for r in matching)
        winners = {r["action"] for r in matching if r["priority"] == best}
        gold = winners.pop() if len(winners) == 1 else None
    else:
        gold = "general_store"
    options = sorted({r["action"] for r, _ in rules} | {"general_store"})
    rng.shuffle(options)
    return {"id": iid, "track": "text", "family": "policy_precedence",
            "state": {"policy": "A rule applies when its condition holds and its 'unless' clause, if any, does not. Apply "
                                "the applicable rule with the smallest priority number. When none of the rules apply, send "
                                "the item to general_store. Applicable rules with the same smallest priority and different "
                                "actions have no tie-breaker.", "rules": [r for r, _ in rules], "item": attrs},
            "field": {"id": "decision", "type": "choice", "question": "Which destination does the policy determine?",
                      "options": [{"value": o} for o in options]},
            "draft_gold": gold, "evidence": f"applicable: {[(r['priority'], r['action']) for r in matching]}"}


def business_days(rng, iid):
    start = _dt.date(rng.choice([2027, 2028]), rng.randint(1, 12), rng.randint(1, 28))
    end = start + _dt.timedelta(days=rng.randint(5, 30))
    holidays = sorted({start + _dt.timedelta(days=rng.randint(1, (end - start).days)) for _ in range(rng.randint(1, 3))})
    count, day = 0, start + _dt.timedelta(days=1)
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            count += 1
        day += _dt.timedelta(days=1)
    options = sorted({count, count + 1, max(0, count - 1), count + 2})
    rng.shuffle(options)
    return {"id": iid, "track": "text", "family": "date_arithmetic",
            "state": {"order_date": start.isoformat(), "delivery_date": end.isoformat(),
                      "holidays": [h.isoformat() for h in holidays],
                      "rule": "Count business days after the order date up to and including the delivery date. Business "
                              "days are Monday to Friday, excluding the listed holidays."},
            "field": {"id": "decision", "type": "choice", "question": "How many business days does the delivery take?",
                      "options": [{"value": str(o)} for o in options]},
            "draft_gold": str(count), "evidence": f"counted {count} business days"}


HARD_FAMILIES = (reconciliation_hard, precedence_hard, business_days, conflicting)


def text_items(count: int, seed: int, hard: bool = False) -> list[dict]:  # noqa: F811 - adds the hard tier
    rng = random.Random(f"imajev-bench-v2-text\0{seed}\0{'hard' if hard else 'standard'}")
    families = HARD_FAMILIES if hard else FAMILIES
    items = []
    for i in range(count):
        item = families[i % len(families)](rng, f"text-{'h' if hard else 's'}{seed:02d}-{i:04d}")
        item["construction"] = {"truth_source": "programmatic", "truth": item["draft_gold"]}
        items.append(item)
    return items
