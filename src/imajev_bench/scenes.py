"""Scene specifications and question templates for synthetic imajev-bench items.

A scene is a structured fact set (prices, counts, lid states...) plus the prompt that renders it, the
statements an image must satisfy, and one-change variants (edits). Questions are templates with
parameters fixed from the original scene; their answers are computed from the facts of whichever image
they are asked about, so an edit that changes, hides or leaves a fact untouched yields the right answer
automatically. Nothing here calls a model.

Unknown (None) arises only when the facts cannot determine the answer: a hidden value, or a value the
scene does not contain. Question and state wording never mentions unknown or abstention.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

VISUAL_STATE = {"instruction": "Use only the supplied image."}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
# Street scenes: generators otherwise add pedestrians, car badges and shop signs, which fail hygiene checks.
QUIET_STREET = "Set on a quiet, empty street early in the morning: no people, no traffic, no shopfronts or signage other than this."
# Scene-level realism: clutter and viewpoint so images are not all clean, centred studio shots.
REALISM = ["photographed at a slight angle", "with some everyday clutter around it", "in soft overcast light",
           "with a mildly busy background", "photographed from a normal standing height", "in warm indoor light"]


@dataclass
class Variant:
    name: str
    relation: str              # change | same  (relative to the contrast question on the original)
    edit: str
    facts: dict
    checks: list[str]


@dataclass
class Scene:
    id: str
    kind: str
    generator: str
    prompt: str
    facts: dict
    checks: list[str]
    variants: list[Variant] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)   # [0] is the contrast question

    def to_json(self):
        return asdict(self)


def _money(rng, low, high):
    return f"{rng.randrange(int(low * 10), int(high * 10) + 1) / 10:.2f}"


def _options(rng, truth_values, distractors, count=4):
    """Distinct string options containing every truth value, shuffled."""
    chosen = list(dict.fromkeys(v for v in truth_values if v is not None))
    for d in distractors:
        if len(chosen) >= count:
            break
        if d not in chosen:
            chosen.append(d)
    rng.shuffle(chosen)
    return chosen


def choice_field(question, options):
    return {"id": "decision", "type": "choice", "question": question, "options": [{"value": o} for o in options]}


def bool_field(question):
    return {"id": "decision", "type": "boolean", "question": question}


# --- answer functions: (facts, params) -> gold -------------------------------------------------------

def _menu_price(facts, p):
    return dict(facts["items"]).get(p["item"], "absent")


def answer(template, facts, params):
    if template == "menu_price":
        value = _menu_price(facts, params)
        return None if value in (None, "absent") else value
    if template == "menu_budget":
        prices = [dict(facts["items"]).get(i) for i in params["items"]]
        return None if any(v is None for v in prices) else round(sum(float(v) for v in prices), 2) <= params["limit"]
    if template == "menu_count":
        return str(len(facts["items"]))
    if template == "menu_listed":
        return params["item"] in dict(facts["items"])
    if template == "shelf_colour_count":
        return str(facts["sequence"].count(params["colour"]))
    if template == "shelf_position":
        seq = facts["sequence"]
        return seq[params["index"]] if params["index"] < len(seq) else None
    if template == "shelf_more":
        return facts["sequence"].count(params["a"]) > facts["sequence"].count(params["b"])
    if template == "shelf_pair_more":
        return facts[0]["sequence"].count(params["colour"]) > facts[1]["sequence"].count(params["colour"])
    if template == "timetable_next":
        times, now = facts["times"], params["now"]
        for k, t in enumerate(times):
            if t is None:
                # The hidden time lies between its visible neighbours; if the current time falls in that gap the
                # hidden departure may or may not be the next one.
                lower = times[k - 1] if k > 0 else "00:00"
                upper = times[k + 1] if k + 1 < len(times) else "24:00"
                if lower <= now < upper:
                    return None
        later = [t for t in times if t is not None and t > now]
        return min(later) if later else None
    if template == "timetable_before":
        times = facts["times"]
        return None if any(t is None for t in times) else str(sum(t < params["cutoff"] for t in times))
    if template == "timetable_window":
        times = facts["times"]
        if any(t is None for t in times):
            return None
        return any(params["start"] <= t <= params["end"] for t in times)
    if template == "sign_stay":
        within_limit = params["stay_minutes"] <= facts["limit_hours"] * 60
        if params["weekday"] not in facts["days_list"] or within_limit:
            return True                       # allowed whether or not the (possibly hidden) hours apply
        if facts["window"] is None:
            return None                       # over the limit on a restricted day: depends on the hidden hours
        start, end = facts["window"]
        inside = start <= params["arrival_hour"] < end and params["weekday"] in facts["days_list"]
        return (not inside) or params["stay_minutes"] <= facts["limit_hours"] * 60
    if template == "sign_limit":
        return f"{facts['limit_hours']} HOUR"
    if template == "bins_open":
        return params["colour"] in facts["open"]
    if template == "bins_open_count":
        return str(len(facts["open"]))
    if template == "bins_all_closed":
        return not facts["open"]
    if template == "thermo_policy":
        return facts["temp"] <= params["max_temp"] and facts["mode"] == params["mode"]
    if template == "thermo_read":
        return f"{facts['temp']:.1f}"
    if template == "parcel_qualifies":
        return facts["weight"] <= params["max_weight"]
    if template == "parcel_code":
        return facts["code"]
    if template == "nutrition_low":
        return None if facts["sugars"] is None else facts["sugars"] <= params["max_sugars"]
    if template == "nutrition_salt":
        return facts["salt"]
    if template == "nutrition_lists":
        return params["nutrient"] in facts["listed"]
    raise KeyError(template)


def render(template, params):
    """Return (track, family, state, field) for a question template; identical for every image in a set."""
    if template == "menu_price":
        return "visual", "text_reading", VISUAL_STATE, choice_field(
            f"What price is written next to {params['item']} on the board?", params["options"])
    if template == "menu_budget":
        a, b = params["items"]
        return "joint", "threshold_rule", {"rule": f"Approve the order if and only if the listed prices of {a} and {b} "
                                           f"add up to at most {params['limit']:.2f}.", "order": [a, b]}, bool_field(
            "Is the order approved under the rule?")
    if template == "menu_listed":
        return "visual", "answerability", VISUAL_STATE, bool_field(f"Does the board list {params['item']}?")
    if template == "menu_count":
        return "visual", "counting", VISUAL_STATE, choice_field("How many menu items are listed on the board?",
                                                                params["options"])
    if template == "shelf_colour_count":
        return "visual", "counting", VISUAL_STATE, choice_field(
            f"How many {params['colour']} {params['object']}s are on the shelf?", params["options"])
    if template == "shelf_position":
        return "visual", "spatial_relation", VISUAL_STATE, choice_field(
            f"What colour is the {ORDINALS[params['index']]} {params['object']} from the left?", params["options"])
    if template == "shelf_more":
        return "joint", "multi_clause_rule", {"rule": f"Restock if and only if there are more {params['a']} {params['object']}s "
                                              f"than {params['b']} {params['object']}s on the shelf."}, bool_field(
            "Does the rule call for a restock?")
    if template == "shelf_pair_more":
        return "joint", "two_image_comparison", {"rule": f"Compare the {params['colour']} {params['object']}s in Image 1 "
                                                 "with those in Image 2."}, bool_field(
            f"Does Image 1 contain more {params['colour']} {params['object']}s than Image 2?")
    if template == "timetable_next":
        return "joint", "threshold_rule", {"current_time": params["now"], "route": params["route"]}, choice_field(
            "Which listed departure is the next one after the current time?", params["options"])
    if template == "timetable_before":
        return "visual", "counting", VISUAL_STATE, choice_field(
            f"How many departures are listed before {params['cutoff']}?", params["options"])
    if template == "timetable_window":
        return "joint", "threshold_rule", {"rule": f"A traveller can board only a departure between {params['start']} "
                                           f"and {params['end']}, inclusive."}, bool_field(
            "Can the traveller board a listed departure?")
    if template == "sign_stay":
        return "joint", "rule_exception", {"arrival": f"{params['weekday']} {params['arrival_hour']:02d}:00",
                                           "planned_stay_minutes": params["stay_minutes"],
                                           "rule": "The stay is allowed if the sign's restriction does not apply at "
                                                   "arrival, or if the planned stay is within the posted limit."}, bool_field(
            "Is the planned stay allowed?")
    if template == "sign_limit":
        return "visual", "text_reading", VISUAL_STATE, choice_field("What maximum stay does the sign show?", params["options"])
    if template == "bins_open":
        return "visual", "attribute_state", VISUAL_STATE, bool_field(f"Is the lid of the {params['colour']} bin open?")
    if template == "bins_open_count":
        return "visual", "counting", VISUAL_STATE, choice_field("How many bins have an open lid?", params["options"])
    if template == "bins_all_closed":
        return "joint", "multi_clause_rule", {"rule": "Collection goes ahead if and only if every bin lid is closed."}, bool_field(
            "Does collection go ahead?")
    if template == "thermo_policy":
        return "joint", "multi_clause_rule", {"rule": f"Approve the setting if and only if the displayed temperature is at most "
                                              f"{params['max_temp']:.1f} °C and the mode is {params['mode']}."}, bool_field(
            "Is the setting approved?")
    if template == "thermo_read":
        return "visual", "text_reading", VISUAL_STATE, choice_field("What temperature does the display show, in °C?",
                                                                    params["options"])
    if template == "parcel_qualifies":
        return "joint", "threshold_rule", {"rule": f"A parcel qualifies for standard post if and only if it weighs at most {params['max_weight']:.1f} kg."}, \
            bool_field("Does this parcel qualify for standard post?")
    if template == "parcel_code":
        return "visual", "text_reading", VISUAL_STATE, choice_field("What tracking code is printed on the label?",
                                                                    params["options"])
    if template == "nutrition_low":
        return "joint", "threshold_rule", {"rule": f"The product is labelled low sugar if and only if its sugars are at most "
                                           f"{params['max_sugars']} g."}, bool_field("Is the product labelled low sugar under the rule?")
    if template == "nutrition_salt":
        return "visual", "text_reading", VISUAL_STATE, choice_field("What salt value is printed on the panel, in grams?",
                                                                    params["options"])
    if template == "nutrition_lists":
        return "visual", "answerability", VISUAL_STATE, bool_field(f"Does the panel list a {params['nutrient']} value?")
    raise KeyError(template)


# --- scene builders ----------------------------------------------------------------------------------

MENU_ITEMS = ["Tomato Soup", "Cheese Toastie", "Flat White", "Carrot Cake", "Lemonade", "Espresso", "Iced Tea",
              "Club Sandwich", "Chocolate Muffin", "Green Salad", "Hot Chocolate", "Bagel", "Veggie Wrap",
              "Apple Juice", "Banana Bread", "Latte", "Minestrone", "Scone"]
COLOURS = ["blue", "red", "yellow", "green"]
BIN_COLOURS = ["blue", "green", "yellow", "grey", "black", "brown"]
IRRELEVANT = ["Add a closed red umbrella leaning against the wall nearby. Change nothing else.",
              "Add a folded cardboard box on the ground at one side. Change nothing else.",
              "Add a small potted plant at one edge of the scene. Change nothing else."]


def _checks_menu(facts):
    lines = [f"The board shows '{n} {facts['currency']}{p}'" if p else
             f"The item {n} is listed but no price can be read next to it" for n, p in facts["items"]]
    return lines + [f"Exactly {len(facts['items'])} items are listed on the board"]


def build_menu(rng, sid, generator):
    items = rng.sample(MENU_ITEMS, rng.randint(4, 6))
    prices = [_money(rng, 1.5, 12.9) for _ in items]
    currency = rng.choice(["£", "€", "$"])
    facts = {"currency": currency, "items": [[n, p] for n, p in zip(items, prices)]}
    target = rng.randrange(len(items))
    hidden = rng.choice([i for i in range(len(items)) if i != target]) if rng.random() < 0.35 else None
    lines = ", ".join(f"'{n} {currency}{p}'" if i != hidden else f"'{n}'" for i, (n, p) in enumerate(facts["items"]))
    surface = rng.choice(["chalkboard A-frame outside a small cafe", "printed menu board on a cafe wall",
                          "whiteboard menu behind a counter"])
    cover_note = ""
    if hidden is not None:
        cover_note = (f" A blank yellow sticky note covers the place where the price of {items[hidden]} would be, so "
                      "no price can be read for it; the item name stays visible.")
        facts["items"][hidden][1] = None
    prompt = (f"A {surface} listing exactly these {len(items)} lines: {lines}.{cover_note} No other items or prices "
              f"anywhere. Scene {rng.choice(REALISM)}.")
    name, price = facts["items"][target]
    new_price = _money(rng, 1.5, 12.9)
    while new_price == price:
        new_price = _money(rng, 1.5, 12.9)
    changed, covered = copy.deepcopy(facts), copy.deepcopy(facts)
    changed["items"][target][1] = new_price
    covered["items"][target][1] = None
    variants = [
        Variant("change", "change", f"Change only the price of {name} from {currency}{price} to {currency}{new_price}. "
                "Keep everything else identical.", changed, _checks_menu(changed)),
        Variant("cover", "change", f"Place one thick, fully opaque yellow sticky note over the price of {name} so that its price is "
                "completely hidden. Keep everything else identical.", covered, _checks_menu(covered)),
        Variant("same", "same", rng.choice(IRRELEVANT), copy.deepcopy(facts), _checks_menu(facts)),
    ]
    distractors = [p for _, p in facts["items"] if p and p not in (price, new_price)] + [_money(rng, 1.5, 12.9) for _ in range(6)]
    questions = [{"template": "menu_price", "params": {"item": name, "options": _options(rng, [price, new_price], distractors)}}]
    a, b = rng.sample([n for n, p in facts["items"] if p], 2)
    total = round(float(dict(facts["items"])[a]) + float(dict(facts["items"])[b]), 2)
    approve = rng.random() < 0.5
    limit = total + rng.choice([0.0, 0.5, 1.0, 2.0]) if approve else total - rng.choice([0.1, 0.5, 1.0, 2.0])
    questions.append({"template": "menu_budget", "params": {"items": [a, b], "limit": round(limit, 2)}})
    if hidden is not None:
        questions.append({"template": "menu_price", "params": {"item": items[hidden], "options": _options(
            rng, [prices[hidden]], [p for _, p in facts["items"] if p] + [_money(rng, 1.5, 12.9) for _ in range(3)])}})
    elif rng.random() < 0.5:
        listed = rng.random() < 0.5
        item = rng.choice(items) if listed else rng.choice([x for x in MENU_ITEMS if x not in items])
        questions.append({"template": "menu_listed", "params": {"item": item}})
    else:
        n = len(items)
        questions.append({"template": "menu_count", "params": {"options": _options(rng, [str(n)], [str(n - 1), str(n + 1), str(n + 2)])}})
    return Scene(sid, "menu", generator, prompt, facts, _checks_menu(facts), variants, questions)


def _checks_shelf(facts):
    seq, obj = facts["sequence"], facts["object"]
    checks = [f"Exactly {len(seq)} {obj}s are on the shelf"]
    checks += [f"Exactly {seq.count(c)} {obj}s are {c}" for c in sorted(set(seq))]
    return checks + [f"From left to right the {obj} colours are {', '.join(seq)}"]


def build_shelf(rng, sid, generator):
    obj = rng.choice(["glass jar", "ceramic mug", "tin can", "storage box"])
    palette = rng.sample(COLOURS, rng.choice([2, 2, 3]))
    seq = [rng.choice(palette) for _ in range(rng.randint(5, 8))]
    for colour in palette:                        # every palette colour appears at least once
        if colour not in seq:
            seq[rng.randrange(len(seq))] = colour
    facts = {"object": obj, "sequence": seq}
    prompt = (f"A plain wooden shelf holding exactly {len(seq)} {obj}s in one row, clearly separated, left to right: "
              f"{', '.join(seq)}. Each colour is bold and unmistakable. No other {obj}s. Scene {rng.choice(REALISM)}.")
    colour = rng.choice([c for c in palette if seq.count(c) >= 1])
    idx = max(i for i, c in enumerate(seq) if c == colour)
    removed = copy.deepcopy(facts)
    del removed["sequence"][idx]
    variants = [Variant("change", "change", f"Remove only the {ORDINALS[idx]} {obj} from the left (the {colour} one), "
                        "leaving an empty gap. Keep everything else identical.", removed, _checks_shelf(removed)),
                Variant("same", "same", "Add one folded tea towel lying flat on the shelf at the far right end, apart from "
                        f"the {obj}s. Change nothing else.", copy.deepcopy(facts), _checks_shelf(facts))]
    n = seq.count(colour)
    questions = [{"template": "shelf_colour_count", "params": {"colour": colour, "object": obj, "options": _options(
        rng, [str(n), str(n - 1)], [str(n + 1), str(n + 2), str(max(0, n - 2))])}}]
    index = rng.randrange(len(seq))
    questions.append({"template": "shelf_position", "params": {"index": index, "object": obj,
                                                                "options": _options(rng, [seq[index]], COLOURS)}})
    if len(palette) >= 2:
        a, b = rng.sample(palette, 2)
        questions.append({"template": "shelf_more", "params": {"a": a, "b": b, "object": obj}})
    return Scene(sid, "shelf", generator, prompt, facts, _checks_shelf(facts), variants, questions)


def _hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _checks_timetable(facts):
    shown = [t for t in facts["times"] if t]
    checks = [f"The route number shown is {facts['route']}"] + [f"{t} is listed" for t in shown]
    if any(t is None for t in facts["times"]):
        checks.append("Exactly one departure row's time is covered by a sticker and cannot be read")
    return checks + [f"Exactly {len(facts['times'])} departure rows are listed"]


def build_timetable(rng, sid, generator):
    route = rng.randint(10, 99)
    starts = sorted(rng.sample(range(6 * 12, 10 * 12), rng.randint(4, 6)))
    times = [_hhmm(s * 5) for s in starts]
    facts = {"route": route, "times": times}
    prompt = (f"A bus stop timetable panel headed 'Route {route} - Weekdays' listing exactly these departure times, one "
              f"per row: {', '.join(times)}. No other times. {QUIET_STREET} Scene {rng.choice(REALISM)}.")
    k = rng.randrange(1, len(times))
    lo = int(times[k - 1][:2]) * 60 + int(times[k - 1][3:])
    hi = int(times[k + 1][:2]) * 60 + int(times[k + 1][3:]) if k + 1 < len(times) else lo + 60
    new_minutes = rng.randrange(lo + 5, hi, 5) if hi - lo > 10 else None
    now = _hhmm(lo + 1)
    variants = []
    if new_minutes and _hhmm(new_minutes) != times[k]:
        changed = copy.deepcopy(facts)
        changed["times"][k] = _hhmm(new_minutes)
        variants.append(Variant("change", "change", f"Change only the departure time {times[k]} to {_hhmm(new_minutes)}. "
                                "Keep everything else identical.", changed, _checks_timetable(changed)))
    covered = copy.deepcopy(facts)
    covered["times"][k] = None
    variants.append(Variant("cover", "change", f"Place a thick, fully opaque white sticker over the departure time {times[k]} so that it "
                            "is completely unreadable. Keep everything else identical.", covered, _checks_timetable(covered)))
    variants.append(Variant("same", "same", rng.choice(IRRELEVANT), copy.deepcopy(facts), _checks_timetable(facts)))
    truths = [times[k]] + ([_hhmm(new_minutes)] if new_minutes else [])
    questions = [{"template": "timetable_next", "params": {"now": now, "route": route, "options": _options(
        rng, truths, [t for t in times if t != times[k]] + [_hhmm(lo + 30)])}}]
    cutoff = rng.choice(["07:30", "08:00", "08:30", "09:00"])
    c = sum(t < cutoff for t in times)
    questions.append({"template": "timetable_before", "params": {"cutoff": cutoff, "options": _options(
        rng, [str(c)], [str(c + 1), str(max(0, c - 1)), str(c + 2), "0"])}})
    return Scene(sid, "timetable", generator, prompt, facts, _checks_timetable(facts), variants, questions)


def _checks_sign(facts):
    checks = [f"The sign says {facts['limit_hours']} HOUR PARKING", f"The sign says {facts['days']}"]
    if facts["window"]:
        s, e = facts["window"]
        checks.append(f"The sign says {s}AM - {e - 12}PM")
    else:
        checks.append("The line with the hours is covered by a sticker and no times can be read")
    return checks


def build_sign(rng, sid, generator):
    limit = rng.randint(1, 4)
    window = [rng.randint(7, 9), rng.randint(17, 19)]
    days, days_list = rng.choice([("MON - FRI", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                                  ("MON - SAT", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])])
    facts = {"limit_hours": limit, "window": window, "days": days, "days_list": days_list}
    prompt = (f"A street parking sign on a pole that reads exactly, on three lines: '{limit} HOUR PARKING', "
              f"'{window[0]}AM - {window[1] - 12}PM', '{days}'. No other text on the sign. {QUIET_STREET} "
              f"Scene {rng.choice(REALISM)}.")
    new_limit = rng.choice([x for x in (1, 2, 3, 4) if x != limit])
    changed, covered = copy.deepcopy(facts), copy.deepcopy(facts)
    changed["limit_hours"] = new_limit
    covered["window"] = None
    variants = [Variant("change", "change", f"Change only '{limit} HOUR' to '{new_limit} HOUR' on the sign. Keep everything "
                        "else identical.", changed, _checks_sign(changed)),
                Variant("cover", "change", f"Cover only the line '{window[0]}AM - {window[1] - 12}PM' with a plain grey "
                        "sticker so the times are unreadable. Keep everything else identical.", covered, _checks_sign(covered)),
                Variant("same", "same", "Add a bicycle locked to the pole below the sign. Change nothing else.",
                        copy.deepcopy(facts), _checks_sign(facts))]
    weekday = rng.choice(days_list[:5])
    hour = rng.randint(window[0], window[1] - 1)
    lo, hi = sorted((limit, new_limit))
    stay = rng.randrange(lo * 60 + 15, hi * 60 + 1, 15)  # between the limits: the one-change edit flips the answer
    questions = [{"template": "sign_stay", "params": {"weekday": weekday, "arrival_hour": hour, "stay_minutes": stay}},
                 {"template": "sign_limit", "params": {"options": _options(rng, [f"{limit} HOUR"],
                                                                             [f"{x} HOUR" for x in (1, 2, 3, 4)])}}]
    return Scene(sid, "sign", generator, prompt, facts, _checks_sign(facts), variants, questions)


def _checks_bins(facts):
    checks = [f"Exactly {len(facts['colours'])} wheelie bins are visible",
              f"From left to right the bin colours are {', '.join(facts['colours'])}"]
    checks += [f"The {c} bin's lid is {'open' if c in facts['open'] else 'closed'}" for c in facts["colours"]]
    return checks


def build_bins(rng, sid, generator):
    colours = rng.sample(BIN_COLOURS, rng.randint(4, 5))
    opened = rng.sample(colours, rng.choice([0, 1, 1, 2]))
    facts = {"colours": colours, "open": opened}
    states = "; ".join(f"the {c} bin's lid is {'open' if c in opened else 'closed'}" for c in colours)
    prompt = (f"Exactly {len(colours)} wheelie bins standing in one row against a wall, left to right coloured "
              f"{', '.join(colours)}; {states}. Scene {rng.choice(REALISM)}.")
    target = rng.choice(colours)
    toggled = copy.deepcopy(facts)
    toggled["open"] = [c for c in opened if c != target] if target in opened else opened + [target]
    action = "Close" if target in opened else "Open"
    variants = [Variant("change", "change", f"{action} only the lid of the {target} bin. Keep everything else identical.",
                        toggled, _checks_bins(toggled)),
                Variant("same", "same", rng.choice(IRRELEVANT), copy.deepcopy(facts), _checks_bins(facts))]
    k = len(opened)
    questions = [{"template": "bins_open", "params": {"colour": target}},
                 {"template": "bins_all_closed", "params": {}},
                 {"template": "bins_open_count", "params": {"options": _options(rng, [str(k)], ["0", "1", "2", "3"])}}]
    return Scene(sid, "bins", generator, prompt, facts, _checks_bins(facts), variants, questions)


def _checks_thermo(facts):
    return [f"The display reads {facts['temp']:.1f}°C", f"The display shows the mode {facts['mode']}"]


def build_thermostat(rng, sid, generator):
    temp = rng.randrange(32, 53) / 2
    mode = rng.choice(["HEAT", "COOL", "AUTO"])
    facts = {"temp": temp, "mode": mode}
    prompt = (f"A wall-mounted digital thermostat whose display reads exactly '{temp:.1f}°C' and '{mode}', and nothing "
              f"else. Scene {rng.choice(REALISM)}.")
    max_temp = rng.choice([temp - 1.0, temp - 0.5, temp, temp + 0.5, temp + 1.5])
    new_temp = temp + 2.0 if max_temp >= temp else max_temp - 1.0
    changed = copy.deepcopy(facts)
    changed["temp"] = new_temp
    variants = [Variant("change", "change", f"Change only the displayed temperature from {temp:.1f}°C to {new_temp:.1f}°C. "
                        "Keep everything else identical.", changed, _checks_thermo(changed)),
                Variant("same", "same", "Add a small framed picture on the wall beside the thermostat. Change nothing else.",
                        copy.deepcopy(facts), _checks_thermo(facts))]
    policy_mode = mode if rng.random() < 0.7 else rng.choice([m for m in ("HEAT", "COOL", "AUTO") if m != mode])
    questions = [{"template": "thermo_policy", "params": {"max_temp": max_temp, "mode": policy_mode}},
                 {"template": "thermo_read", "params": {"options": _options(
                     rng, [f"{temp:.1f}"], [f"{temp + d:.1f}" for d in (0.5, -0.5, 1.0, -1.0, 10.0)])}}]
    return Scene(sid, "thermostat", generator, prompt, facts, _checks_thermo(facts), variants, questions)


def _checks_parcel(facts):
    return [f"The label shows 'WEIGHT: {facts['weight']:.1f} kg'", f"The label shows 'TRACKING: {facts['code']}'",
            f"The label shows '{facts['address']}'"]


def build_parcel(rng, sid, generator):
    weight = rng.randrange(5, 100) / 10
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
    digits = "".join(rng.choice("0123456789") for _ in range(4))
    code = f"{letters}-{digits}"
    address = f"Unit {rng.randint(2, 48)}, {rng.choice(['Harbour', 'Mill', 'Station', 'Orchard', 'Canal'])} Road"
    facts = {"weight": weight, "code": code, "address": address}
    prompt = (f"A cardboard parcel with a printed white shipping label showing exactly: 'TO: {address}', "
              f"'WEIGHT: {weight:.1f} kg', 'TRACKING: {code}', and a barcode. No other text. Scene {rng.choice(REALISM)}.")
    max_weight = rng.choice([weight, weight + 0.5, weight - 0.3, weight + 2.0, weight - 1.0])
    new_weight = round(max_weight + 1.2, 1) if weight <= max_weight else round(max(0.3, max_weight - 0.8), 1)
    changed = copy.deepcopy(facts)
    changed["weight"] = new_weight
    variants = [Variant("change", "change", f"Change only the weight on the label from {weight:.1f} kg to {new_weight:.1f} kg. "
                        "Keep everything else identical.", changed, _checks_parcel(changed)),
                Variant("same", "same", "Add a roll of brown packing tape beside the parcel. Change nothing else.",
                        copy.deepcopy(facts), _checks_parcel(facts))]
    near = [f"{letters}-{digits[::-1]}", f"{letters[::-1]}-{digits}", f"{letters}-{digits[:2]}{digits[3]}{digits[2]}"]
    questions = [{"template": "parcel_qualifies", "params": {"max_weight": round(max_weight, 1)}},
                 {"template": "parcel_code", "params": {"options": _options(rng, [code], near + [f"{letters}-{int(digits) + 7:04d}"[-7:]])}}]
    return Scene(sid, "parcel", generator, prompt, facts, _checks_parcel(facts), variants, questions)


def _checks_nutrition(facts):
    lines = [f"Energy is {facts['energy']} kcal", f"Fat is {facts['fat']} g", f"Salt is {facts['salt']} g"]
    lines.append(f"Sugars is {facts['sugars']} g" if facts["sugars"] is not None else
                 "The sugars line is covered and no sugars value can be read")
    return lines + ["No protein value is printed"]


def build_nutrition(rng, sid, generator):
    facts = {"energy": rng.randrange(80, 520, 10), "fat": rng.randint(1, 30), "sugars": rng.randint(1, 40),
             "salt": f"{rng.randrange(1, 30) / 10:.1f}", "listed": ["energy", "fat", "sugars", "salt"]}
    prompt = (f"Close-up of a plain food package nutrition panel listing exactly four lines: 'Energy {facts['energy']} kcal', "
              f"'Fat {facts['fat']} g', 'Sugars {facts['sugars']} g', 'Salt {facts['salt']} g'. No protein line and no "
              f"other values. Scene {rng.choice(REALISM)}.")
    max_sugars = facts["sugars"] + rng.choice([-3, -1, 0, 2, 5])
    new_sugars = max_sugars + 4 if facts["sugars"] <= max_sugars else max(0, max_sugars - 2)
    changed, covered = copy.deepcopy(facts), copy.deepcopy(facts)
    changed["sugars"] = new_sugars
    covered["sugars"] = None
    variants = [Variant("change", "change", f"Change only the sugars value from {facts['sugars']} g to {new_sugars} g. "
                        "Keep everything else identical.", changed, _checks_nutrition(changed)),
                Variant("cover", "change", "Cover only the sugars line with a strip of opaque tape so its value is "
                        "unreadable. Keep everything else identical.", covered, _checks_nutrition(covered))]
    salt = facts["salt"]
    questions = [{"template": "nutrition_low", "params": {"max_sugars": max_sugars}},
                 {"template": "nutrition_salt", "params": {"options": _options(
                     rng, [salt], [f"{float(salt) + d:.1f}" for d in (0.1, -0.1, 0.5, 1.0) if float(salt) + d > 0])}}]
    nutrient = rng.choice(["protein", "fibre", "fat", "salt"])
    questions.append({"template": "nutrition_lists", "params": {"nutrient": nutrient}})
    return Scene(sid, "nutrition", generator, prompt, facts, _checks_nutrition(facts), variants, questions)


BUILDERS = {"menu": build_menu, "shelf": build_shelf, "timetable": build_timetable, "sign": build_sign,
            "bins": build_bins, "thermostat": build_thermostat, "parcel": build_parcel, "nutrition": build_nutrition}


def plan_scenes(count: int, seed: int, generators=("flare", "nano-banana-2"), kinds=None, first_share=0.5) -> list[Scene]:
    """Round-robin over scene kinds; within each kind, the first generator gets `first_share` of scenes.

    The split is spread evenly (not front-loaded) so every kind has scenes from both generators.
    """
    rng = random.Random(f"imajev-bench-v2-scenes\0{seed}")
    kinds = list(kinds or BUILDERS)
    scenes = []
    for i in range(count):
        kind = kinds[i % len(kinds)]
        k = i // len(kinds)                       # k-th scene of this kind
        other = int((k + 1) * (1 - first_share)) > int(k * (1 - first_share))
        generator = generators[1] if len(generators) > 1 and other else generators[0]
        scenes.append(BUILDERS[kind](rng, f"scene-{seed:02d}-{i:04d}", generator))
    return scenes


# --- hard tier -----------------------------------------------------------------------------------------
# Composition and careful reading rather than degraded images: denser scenes, near-miss distractors,
# multi-step rules, and hidden values that may or may not matter.

def _hard_answer(template, facts, params):
    if template == "menuh_order":
        prices = dict(facts["items"])
        needed = [prices.get(name) for name, _ in params["order"]]
        if any(p is None for p in needed):
            return None
        total = round(sum(float(p) * qty for p, (_, qty) in zip(needed, params["order"])), 2)
        return total <= params["limit"]
    if template == "menuh_current":
        return dict(facts["items"]).get(params["item"])
    if template == "menuh_cheapest":
        if any(p is None for _, p in facts["items"]):
            return None                       # a hidden price could be the lowest
        prices = sorted((float(p), n) for n, p in facts["items"])
        return None if prices[0][0] == prices[1][0] else prices[0][1]
    if template == "shelfd_right_of":
        seq = facts["sequence"]
        if params["anchor"] not in seq:
            return None
        first = seq.index(params["anchor"])
        return str(seq[first + 1:].count(params["colour"]))
    if template == "signx_stay":
        if params["weekday"] not in facts["days_list"]:
            return True
        start = params["arrival"]
        end = start + params["stay_minutes"]
        if facts["exception"] is None:
            # The exception times are hidden: the answer is determined only if every possible placement agrees,
            # which we do not assume; the limit alone can still forbid the stay.
            if facts["window"][0] * 60 <= start < facts["window"][1] * 60 and params["stay_minutes"] > facts["limit_hours"] * 60:
                return False
            return None
        ex_start, ex_end = facts["exception"][0] * 60, facts["exception"][1] * 60
        if start < ex_end and end > ex_start:
            return False
        in_window = facts["window"][0] * 60 <= start < facts["window"][1] * 60
        return (not in_window) or params["stay_minutes"] <= facts["limit_hours"] * 60
    if template == "nutric_serving":
        return facts["per_serving"]["sugars"]
    if template == "nutric_rule":
        value = facts["per_serving"]["sugars"]
        return None if value is None else float(value) <= params["max"]
    if template == "parcels_heavier":
        a, b = facts["parcels"]
        if a["weight"] == b["weight"]:
            return None
        return (a if a["weight"] > b["weight"] else b)["code"]
    if template == "parcels_count":
        return str(sum(p["weight"] <= params["max_weight"] for p in facts["parcels"]))
    raise KeyError(template)


def _hard_render(template, params):
    if template == "menuh_order":
        order = ", ".join(f"{qty} × {name}" for name, qty in params["order"])
        return "joint", "multi_step_rule", {"order": order, "rule": f"Approve the order if and only if its total at the "
                                            f"board's current prices is at most {params['limit']:.2f}."}, bool_field(
            "Is the order approved under the rule?")
    if template == "menuh_current":
        return "visual", "text_reading", VISUAL_STATE, choice_field(
            f"What is the current price of {params['item']} on the board?", params["options"])
    if template == "menuh_cheapest":
        return "visual", "comparison", VISUAL_STATE, choice_field("Which item has the lowest current price on the board?",
                                                                  params["options"])
    if template == "shelfd_right_of":
        return "visual", "spatial_counting", VISUAL_STATE, choice_field(
            f"How many {params['colour']} {params['object']}s are to the right of the first {params['anchor']} "
            f"{params['object']}?", params["options"])
    if template == "signx_stay":
        hh, mm = divmod(params["arrival"], 60)
        return "joint", "rule_exception", {"arrival": f"{params['weekday']} {hh:02d}:{mm:02d}",
                                           "planned_stay_minutes": params["stay_minutes"],
                                           "rule": "The stay is allowed if and only if it does not overlap any no-parking "
                                                   "period on the signs, and, when the time limit applies at arrival, it "
                                                   "is within that limit."}, bool_field("Is the planned stay allowed?")
    if template == "nutric_serving":
        return "visual", "text_reading", VISUAL_STATE, choice_field(
            "How many grams of sugars are in one serving, according to the panel?", params["options"])
    if template == "nutric_rule":
        return "joint", "threshold_rule", {"rule": f"The product is labelled low sugar if and only if one serving "
                                           f"contains at most {params['max']} g of sugars."}, bool_field(
            "Is the product labelled low sugar under the rule?")
    if template == "parcels_heavier":
        return "visual", "comparison", VISUAL_STATE, choice_field(
            "Which tracking code is on the heavier parcel, according to the labels?", params["options"])
    if template == "parcels_count":
        return "joint", "threshold_rule", {"rule": f"A parcel qualifies for standard post if and only if its label weight "
                                           f"is at most {params['max_weight']:.1f} kg."}, choice_field(
            "How many of the parcels shown qualify for standard post?", params["options"])
    raise KeyError(template)


_base_answer, _base_render = answer, render


def answer(template, facts, params):  # noqa: F811 - extends the standard templates
    try:
        return _base_answer(template, facts, params)
    except KeyError:
        return _hard_answer(template, facts, params)


def render(template, params):  # noqa: F811
    try:
        return _base_render(template, params)
    except KeyError:
        return _hard_render(template, params)


SIMILAR = [("Latte", "Iced Latte"), ("Tea", "Iced Tea"), ("Mocha", "White Mocha"), ("Soup", "Soup of the Day"),
           ("Toastie", "Ham Toastie"), ("Brownie", "Vegan Brownie"), ("Cola", "Diet Cola")]


def _checks_menuh(facts):
    checks = []
    for name, price in facts["items"]:
        if price is None:
            checks.append(f"The item {name} is listed but no price can be read next to it")
        elif name == facts.get("struck", [None])[0]:
            checks.append(f"{name} shows the old price {facts['currency']}{facts['struck'][1]} crossed out and the new price "
                          f"{facts['currency']}{price}")
        else:
            checks.append(f"The board shows '{name} {facts['currency']}{price}'")
    return checks + [f"Exactly {len(facts['items'])} items are listed"]


def build_menu_hard(rng, sid, generator):
    pairs = rng.sample(SIMILAR, 2)
    others = rng.sample([m for m in MENU_ITEMS if all(m not in p for p in pairs)], rng.randint(3, 5))
    names = [n for p in pairs for n in p] + others
    rng.shuffle(names)
    currency = rng.choice(["£", "€", "$"])
    prices = rng.sample([f"{x / 10:.2f}" for x in range(15, 100)], len(names))  # distinct: a unique cheapest item
    facts = {"currency": currency, "items": [[n, p] for n, p in zip(names, prices)]}
    struck_i = rng.randrange(len(names))
    old = _money(rng, 1.5, 9.9)
    while old == facts["items"][struck_i][1]:
        old = _money(rng, 1.5, 9.9)
    facts["struck"] = [names[struck_i], old]
    lines = []
    for i, (n, p) in enumerate(facts["items"]):
        lines.append(f"'{n}' with the old price {currency}{old} crossed out and the new price {currency}{p} written next to it"
                     if i == struck_i else f"'{n} {currency}{p}'")
    prompt = (f"A busy handwritten cafe menu board listing exactly these {len(names)} items: {'; '.join(lines)}. No other "
              f"items or prices. Scene {rng.choice(REALISM)}.")
    order_names = rng.sample([n for n in names], 2)
    order = [[order_names[0], rng.randint(1, 3)], [order_names[1], rng.randint(1, 2)]]
    total = round(sum(float(dict(facts["items"])[n]) * q for n, q in order), 2)
    limit = round(total + rng.choice([0.0, 0.3, 1.0]) if rng.random() < 0.5 else total - rng.choice([0.1, 0.4, 1.5]), 2)
    in_order = order_names[0]
    not_in_order = rng.choice([n for n in names if n not in order_names])
    new_price = _money(rng, 1.5, 9.9)
    changed, covered, covered_irrelevant = copy.deepcopy(facts), copy.deepcopy(facts), copy.deepcopy(facts)
    for f, name, value in ((changed, in_order, new_price), (covered, in_order, None), (covered_irrelevant, not_in_order, None)):
        for row in f["items"]:
            if row[0] == name:
                row[1] = value
    old_in = dict(facts["items"])[in_order]
    variants = [
        Variant("change", "change", f"Change only the price shown for {in_order} from {currency}{old_in} to "
                f"{currency}{new_price}. Keep everything else identical.", changed, _checks_menuh(changed)),
        Variant("cover", "change", f"Place one thick, fully opaque yellow sticky note over the price of {in_order} so it cannot be read. "
                "Keep everything else identical.", covered, _checks_menuh(covered)),
        Variant("cover_other", "same", f"Place one thick, fully opaque yellow sticky note over the price of {not_in_order} so it cannot be "
                "read. Keep everything else identical.", covered_irrelevant, _checks_menuh(covered_irrelevant)),
    ]
    struck_name, current = names[struck_i], facts["items"][struck_i][1]
    prices = [p for _, p in facts["items"]]
    cheapest = min(facts["items"], key=lambda x: float(x[1]))[0]
    questions = [
        {"template": "menuh_order", "params": {"order": order, "limit": limit}},
        {"template": "menuh_current", "params": {"item": struck_name, "options": _options(
            rng, [current, old], prices + [_money(rng, 1.5, 9.9)])}},
        {"template": "menuh_cheapest", "params": {"options": _options(rng, [cheapest], names)}},
    ]
    return Scene(sid, "menu_hard", generator, prompt, facts, _checks_menuh(facts), variants, questions)


def build_shelf_dense(rng, sid, generator):
    obj = rng.choice(["glass jar", "ceramic mug", "tin can"])
    palette = rng.sample(COLOURS, 3)
    seq = [rng.choice(palette) for _ in range(rng.randint(9, 12))]
    for colour in palette:
        if colour not in seq:
            seq[rng.randrange(len(seq))] = colour
    facts = {"object": obj, "sequence": seq}
    prompt = (f"A long plain wooden shelf holding exactly {len(seq)} {obj}s in one row, evenly spaced and clearly separated, "
              f"left to right: {', '.join(seq)}. Each colour is bold and unmistakable. No other {obj}s. "
              f"Scene {rng.choice(REALISM)}.")
    colour = rng.choice(palette)
    idx = max(i for i, c in enumerate(seq) if c == colour)
    removed = copy.deepcopy(facts)
    del removed["sequence"][idx]
    variants = [Variant("change", "change", f"Remove only the {ORDINALS[idx] if idx < len(ORDINALS) else str(idx + 1) + 'th'} "
                        f"{obj} from the left (a {colour} one), leaving an empty gap. Keep everything else identical.",
                        removed, _checks_shelf(removed)),
                Variant("same", "same", f"Add one folded tea towel on the shelf at the far right end, apart from the {obj}s. "
                        "Change nothing else.", copy.deepcopy(facts), _checks_shelf(facts))]
    n = seq.count(colour)
    anchor, target = rng.sample(palette, 2)
    first = seq.index(anchor)
    k = seq[first + 1:].count(target)
    index = rng.randrange(5, min(9, len(seq)))
    questions = [
        {"template": "shelf_colour_count", "params": {"colour": colour, "object": obj, "options": _options(
            rng, [str(n), str(n - 1)], [str(n + 1), str(n + 2), str(max(0, n - 2))])}},
        {"template": "shelfd_right_of", "params": {"colour": target, "anchor": anchor, "object": obj, "options": _options(
            rng, [str(k)], [str(k + 1), str(max(0, k - 1)), str(k + 2), str(seq.count(target))])}},
        {"template": "shelf_position", "params": {"index": index, "object": obj, "options": _options(rng, [seq[index]], COLOURS)}},
    ]
    return Scene(sid, "shelf_dense", generator, prompt, facts, _checks_shelf(facts), variants, questions)


def _checks_signx(facts):
    checks = [f"The main sign says {facts['limit_hours']} HOUR PARKING",
              f"The main sign says {facts['window'][0]}AM - {facts['window'][1] - 12}PM", f"The main sign says {facts['days']}"]
    if facts["exception"]:
        s, e = facts["exception"]
        checks.append(f"A second plate says NO PARKING {s - 12}PM - {e - 12}PM")
    else:
        checks.append("A second plate says NO PARKING but its times are covered and cannot be read")
    return checks


def build_sign_exception(rng, sid, generator):
    limit = rng.randint(1, 3)
    window = [rng.randint(7, 9), rng.randint(18, 19)]
    ex_start = rng.randint(15, 16)
    exception = [ex_start, ex_start + 2]
    days, days_list = ("MON - FRI", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    facts = {"limit_hours": limit, "window": window, "days": days, "days_list": days_list, "exception": exception}
    prompt = (f"A parking sign pole with two plates. The upper plate reads exactly, on three lines: '{limit} HOUR PARKING', "
              f"'{window[0]}AM - {window[1] - 12}PM', '{days}'. The lower plate reads exactly: 'NO PARKING', "
              f"'{exception[0] - 12}PM - {exception[1] - 12}PM'. No other text. {QUIET_STREET} Scene {rng.choice(REALISM)}.")
    weekday = rng.choice(days_list)
    # Arrive before the exception; choose a stay that ends just before, inside, or well within it.
    arrival = ex_start * 60 - rng.choice([150, 90, 60, 45])
    stay = rng.choice([30, 45, 60, 75, 90, 120, 150])
    shifted = copy.deepcopy(facts)
    shifted["exception"] = [ex_start + 1, ex_start + 3] if ex_start == 15 else [ex_start - 1, ex_start + 1]
    covered = copy.deepcopy(facts)
    covered["exception"] = None
    variants = [Variant("change", "change", f"Change only the lower plate's times to '{shifted['exception'][0] - 12}PM - "
                        f"{shifted['exception'][1] - 12}PM'. Keep everything else identical.", shifted, _checks_signx(shifted)),
                Variant("cover", "change", "Cover only the times on the lower NO PARKING plate with a thick, fully opaque grey sticker so "
                        "they cannot be read. Keep everything else identical.", covered, _checks_signx(covered)),
                Variant("same", "same", "Add a bicycle locked to the pole below the signs. Change nothing else.",
                        copy.deepcopy(facts), _checks_signx(facts))]
    questions = [{"template": "signx_stay", "params": {"weekday": weekday, "arrival": arrival, "stay_minutes": stay}},
                 {"template": "sign_limit", "params": {"options": _options(rng, [f"{limit} HOUR"],
                                                                             [f"{x} HOUR" for x in (1, 2, 3, 4)])}}]
    return Scene(sid, "sign_exception", generator, prompt, facts, _checks_signx(facts), variants, questions)


def _checks_nutric(facts):
    ps = facts["per_serving"]
    checks = [f"The panel has a 'per 100 g' column and a 'per serving ({facts['serving']} g)' column",
              f"Per 100 g, sugars is {facts['per_100']['sugars']} g", f"Per 100 g, fat is {facts['per_100']['fat']} g",
              f"Per serving, fat is {ps['fat']} g"]
    checks.append(f"Per serving, sugars is {ps['sugars']} g" if ps["sugars"] is not None else
                  "The per-serving sugars value is covered and cannot be read")
    return checks


def build_nutrition_columns(rng, sid, generator):
    serving = rng.choice([25, 30, 40, 45, 60])
    per_100 = {"sugars": rng.randint(8, 45), "fat": rng.randint(2, 30)}
    per_serving = {k: f"{v * serving / 100:.1f}" for k, v in per_100.items()}
    per_100 = {k: str(v) for k, v in per_100.items()}
    facts = {"serving": serving, "per_100": per_100, "per_serving": per_serving}
    prompt = (f"Close-up of a plain food package nutrition table with two columns headed 'per 100 g' and "
              f"'per serving ({serving} g)'. Rows: 'Fat' {per_100['fat']} g and {per_serving['fat']} g; 'Sugars' "
              f"{per_100['sugars']} g and {per_serving['sugars']} g. No other rows or values. Scene {rng.choice(REALISM)}.")
    value = float(per_serving["sugars"])
    maximum = round(value + rng.choice([-2.0, -0.5, 0.0, 1.0, 3.0]), 1)
    new_value = f"{maximum + 2.5:.1f}" if value <= maximum else f"{max(0.5, maximum - 1.5):.1f}"
    changed = copy.deepcopy(facts)
    changed["per_serving"]["sugars"] = new_value
    # No covered variant: the per-serving value is derivable from per 100 g and the serving size, so hiding it
    # would not make the answer unknown (a model caught this in the hard trial).
    variants = [Variant("change", "change", f"Change only the per-serving sugars value from {per_serving['sugars']} g to "
                        f"{new_value} g. Keep everything else identical.", changed, _checks_nutric(changed)),
                Variant("same", "same", "Add a small spoon lying beside the package. Change nothing else.",
                        copy.deepcopy(facts), _checks_nutric(facts))]
    questions = [{"template": "nutric_rule", "params": {"max": maximum}},
                 {"template": "nutric_serving", "params": {"options": _options(
                     rng, [per_serving["sugars"]], [per_100["sugars"], per_serving["fat"], f"{value + 1:.1f}"])}}]
    return Scene(sid, "nutrition_columns", generator, prompt, facts, _checks_nutric(facts), variants, questions)


def _checks_parcels(facts):
    return [f"The {side} parcel's label shows 'WEIGHT: {p['weight']:.1f} kg' and 'TRACKING: {p['code']}'"
            for side, p in zip(("left", "right"), facts["parcels"])] + ["Exactly two parcels are visible"]


def build_parcels_pair(rng, sid, generator):
    def code():
        return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2)) + "-" + \
            "".join(rng.choice("0123456789") for _ in range(4))
    weights = rng.sample([x / 10 for x in range(5, 100)], 2)
    parcels = [{"weight": w, "code": code()} for w in weights]
    facts = {"parcels": parcels}
    prompt = (f"Two cardboard parcels side by side, each with a printed white shipping label. The left label shows exactly "
              f"'WEIGHT: {parcels[0]['weight']:.1f} kg' and 'TRACKING: {parcels[0]['code']}'; the right label shows exactly "
              f"'WEIGHT: {parcels[1]['weight']:.1f} kg' and 'TRACKING: {parcels[1]['code']}'. Place the lighter parcel's box "
              f"so it looks bigger than the heavier one. No other text. Scene {rng.choice(REALISM)}.")
    heavier = max(parcels, key=lambda p: p["weight"])
    lighter = min(parcels, key=lambda p: p["weight"])
    swapped = copy.deepcopy(facts)
    for p in swapped["parcels"]:
        p["weight"] = lighter["weight"] if p["code"] == heavier["code"] else heavier["weight"]
    variants = [Variant("change", "change", f"Swap only the two weight values on the labels, so the parcel with "
                        f"{heavier['code']} shows {lighter['weight']:.1f} kg and the one with {lighter['code']} shows "
                        f"{heavier['weight']:.1f} kg. Keep everything else identical.", swapped, _checks_parcels(swapped)),
                Variant("same", "same", "Add a roll of brown packing tape beside the parcels. Change nothing else.",
                        copy.deepcopy(facts), _checks_parcels(facts))]
    max_weight = round(rng.choice([min(weights) + 0.2, (min(weights) + max(weights)) / 2, max(weights) + 0.5,
                                   min(weights) - 0.3]), 1)
    questions = [{"template": "parcels_heavier", "params": {"options": _options(
                     rng, [p["code"] for p in parcels], [code(), code()])}},
                 {"template": "parcels_count", "params": {"max_weight": max_weight, "options": ["0", "1", "2"]}}]
    return Scene(sid, "parcels_pair", generator, prompt, facts, _checks_parcels(facts), variants, questions)


HARD_BUILDERS = {"menu_hard": build_menu_hard, "shelf_dense": build_shelf_dense, "sign_exception": build_sign_exception,
                 "nutrition_columns": build_nutrition_columns, "parcels_pair": build_parcels_pair}
BUILDERS.update(HARD_BUILDERS)
