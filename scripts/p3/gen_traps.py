#!/usr/bin/env python3
"""Phase-3 source D: trap and near-miss variants of items with a constructed gold (docs/phase-3-plan.md, source D).

Works on any scripts/p3/candidate.py row whose gold_kind is "constructed" (gen_policy output, gen_reasoning output, ...). Four
transforms, each keeping the gold exact by construction:

  near_miss  (a) add an option that almost fits: the parent's generator hint (right rule, wrong exception) when present, else the
                 gold option's text with one quantity changed (right entity, wrong number). The added option differs from the gold
                 in a stated value, so it is wrong whenever the question has one correct option. Gold key unchanged.
  negation   (b) negated wording: the generator's natural negated question when it supplies one (noul, gold recomputed by the
                 generator); otherwise an exact frame -- noul: "is the correct answer 'no'?" (gold flipped); choice: two proposed
                 answers (the gold and a wrong option) plus "both"/"neither", asking which one is NOT supported (gold = the wrong
                 one); score: "is the correct level something other than L?" (gold = level != L).
  bury       (c) the decisive fact is buried: plausible but irrelevant operational notices (other offices, facilities, IT, staffing;
                 never about the item's entities, records or rules) are inserted next to the decisive clause, or mid-state.
  qualifier  (d) an option that differs from the gold only in a qualifier ("30 days" vs "30 business days", "excluding" vs
                 "including", "at least" vs "more than"...). Gold key unchanged.

Gold re-derivation (fix 2026-09-26): EVERY trap's gold is re-derived from the parent's provenance.spec under the TRANSFORMED question,
never inherited. The parent's undetermined facts are enumerated exactly as its unknown was proven (the family module's worlds():
every plausible value of the withheld fact; long_policy regenerates the item and reads the proof's alternatives from its trace), each
completion's answer is mapped through the transform, and the gold is the unique answer when all completions agree, else null
(parent's unknown_reason). This matters for unknown parents: "is the band something other than Band 3?" is YES when the missing
engagement type only leaves Bands 0 and 1 possible, and "which proposed answer is NOT supported?" is "neither" when no completion
gives either proposed answer. Answerable parents have one completion (their gold is re-checked too). Parents without a solver
(foreign rows) keep the by-construction gold and are refused the two transforms that can decide an unknown parent (score / choice
negation). ~15% of the output comes from unknown parents.

    .venv/bin/python scripts/p3/gen_traps.py --in data/p3/candidates/A-policy.jsonl --out data/p3/candidates/D-traps.jsonl --count 8000
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate import read, validate, write  # noqa: E402

KINDS = ("near_miss", "negation", "bury", "qualifier")
MAX_OPTIONS = 254
TOKEN_CAP = 15800  # keep long-input states under 16k real tokens


def rng_for(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def approx_tokens(text: str) -> int:
    """Conservative for logs/tables/ids: the Qwen tokenizer counts up to ~1.8x more than len/3.3 on such text."""
    return int(len(text) / 1.8) + 1


def option_key(text: str, taken: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")[:40] or "option"
    if key[0].isdigit():
        key = ("v_" + key)[:40]
    if key in ("unknown", "__unknown__", "option_unknown"):
        key = "opt_" + key
    base, n = key, 2
    while key in taken:
        suffix = f"_{n}"
        key = base[:40 - len(suffix)] + suffix
        n += 1
    taken.add(key)
    return key


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


# ------------------------------------------------------------------------------------------------ (a) near miss
MONTHS = "january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec".split()
_DATEISH = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\b(" + "|".join(MONTHS) + r")\b", re.I)
_NUM = re.compile(r"(?<![\w\-/.:#])([$€£]|[A-Z]{1,3}\$|CHF )?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%| ?(?:days?|hours?|minutes?|business days|working days|calendar days|months?|years?|km|kg|pages|points|users|customers|square metres))?(?![\w\-/])")


def perturb_number(text: str, rng: random.Random) -> str | None:
    """The same text with one quantity changed; None if the text has no safe quantity (dates, times and ids are left alone)."""
    if _DATEISH.search(text):
        return None
    cands = []
    for m in _NUM.finditer(text):
        cur, whole, dec, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        n = int(whole.replace(",", ""))
        if not (cur or unit or dec or n >= 10):
            continue
        cands.append(m)
    if not cands:
        return None
    m = rng.choice(cands)
    cur, whole, dec, unit = m.group(1) or "", m.group(2), m.group(3) or "", m.group(4) or ""
    n = int(whole.replace(",", ""))
    if dec and rng.random() < 0.5:
        d = int(dec[1:])
        nd = (d + rng.choice([1, 2, 5, 9, 10, 20, 50]) * rng.choice([1, -1])) % (10 ** (len(dec) - 1))
        if nd == d:
            return None
        new_dec = "." + str(nd).zfill(len(dec) - 1)
        new_whole = whole
    else:
        step = rng.choice([1, 2, 5, 10]) if n < 100 else rng.choice([max(1, n // 10), max(1, n // 20), 10, 100])
        nn = n + step * rng.choice([1, -1])
        if nn < 0 or nn == n:
            nn = n + step
        # swapped digits is a classic near miss for money
        if n >= 100 and rng.random() < 0.3:
            s = str(n)
            i = rng.randrange(len(s) - 1)
            sw = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            if sw != s and sw[0] != "0":
                nn = int(sw)
        new_whole = f"{nn:,}" if "," in whole or (nn >= 1000 and cur) else str(nn)
        new_dec = dec
    new = f"{cur}{new_whole}{new_dec}{unit}"
    out = text[:m.start()] + new + text[m.end():]
    return out if out != text else None


def t_near_miss(row: dict, rng: random.Random) -> dict | None:
    f = row["field"]
    if f["type"] != "choice" or len(f["options"]) >= MAX_OPTIONS:
        return None
    existing = {_norm(o["text"]) for o in f["options"]}
    hints = ((row.get("provenance") or {}).get("trap_hints") or {})
    cand, how = None, None
    for t in hints.get("near_miss") or []:
        if _norm(t) not in existing:
            cand, how = t, "hint"
            break
    if cand is None and row.get("gold") is not None:
        gold_text = next(o["text"] for o in f["options"] if o["key"] == row["gold"])
        for _ in range(6):
            t = perturb_number(gold_text, rng)
            if t and _norm(t) not in existing:
                cand, how = t, "number"
                break
    if cand is None and row.get("gold") is None:
        # unknown parent: perturb any option (the item stays unanswerable)
        for o in rng.sample(f["options"], len(f["options"])):
            t = perturb_number(o["text"], rng)
            if t and _norm(t) not in existing:
                cand, how = t, "number"
                break
    if cand is None:
        return None
    new = copy.deepcopy(row)
    taken = {o["key"] for o in new["field"]["options"]}
    opt = {"key": option_key(cand, taken), "text": cand}
    if any("description" in o for o in new["field"]["options"]):
        opt["description"] = cand
    pos = rng.randrange(len(new["field"]["options"]) + 1)
    new["field"]["options"].insert(pos, opt)
    return _mark(new, "near_miss", {"added_option": opt["key"], "source": how})


# ------------------------------------------------------------------------------------------------ (b) negation
def t_negation(row: dict, rng: random.Random) -> dict | None:
    f = row["field"]
    hints = ((row.get("provenance") or {}).get("trap_hints") or {})
    gold = row.get("gold")
    new = copy.deepcopy(row)
    if f["type"] == "noul":
        neg = hints.get("neg")
        if neg and gold is not None and isinstance(neg.get("gold"), bool) and neg["gold"] == (not gold):
            new["field"]["question"] = neg["question"]
            new["gold"] = neg["gold"]
            return _mark(new, "negation", {"source": "hint"})
        if neg and gold is None:
            new["field"]["question"] = neg["question"]
            return _mark(new, "negation", {"source": "hint"})
        q = f["question"].strip()
        new["field"]["question"] = rng.choice([
            f"Consider the question \"{q}\" Is the correct answer to it \"no\"?",
            f"Someone answered \"yes\" to this question: \"{q}\" Were they wrong?",
            f"Would it be wrong to answer \"yes\" to the following question? {q}"])
        new["gold"] = None if gold is None else (not gold)
        return _mark(new, "negation", {"source": "frame"})
    if f["type"] == "choice":
        opts = f["options"]
        if gold is not None:
            gold_opt = next(o for o in opts if o["key"] == gold)
            nm = [t for t in hints.get("near_miss") or [] if _norm(t) != _norm(gold_opt["text"])]
            wrong_opts = [o for o in opts if o["key"] != gold]
            wrong_text = nm[0] if nm and rng.random() < 0.5 else rng.choice(wrong_opts)["text"]
            pair = [gold_opt["text"], wrong_text]
        else:
            a, b = rng.sample(opts, 2)
            pair = [a["text"], b["text"]]
        rng.shuffle(pair)
        both, neither = "Both answers are supported", "Neither answer is supported"
        q = f["question"].strip()
        texts = [f"Answer 1: {pair[0]}", f"Answer 2: {pair[1]}", both, neither]
        taken: set[str] = set()
        new_opts = [{"key": option_key(t, taken), "text": t} for t in texts]
        new["field"] = {"type": "choice", "options": new_opts,
                        "question": rng.choice([f"Two answers were proposed to the question \"{q}\" Which proposed answer is NOT supported by the material above?",
                                                f"For the question \"{q}\", which of the two proposed answers is NOT correct?",
                                                f"Question: \"{q}\" One of the two answers below is wrong. Which answer is NOT correct?"])}
        if gold is not None:
            wrong_idx = 0 if pair[0] != gold_opt["text"] else 1
            new["gold"] = new_opts[wrong_idx]["key"]
        return _mark(new, "negation", {"source": "pair"})
    if f["type"] == "score":
        levels = f["levels"]
        if gold is not None:
            L = gold if rng.random() < 0.5 else rng.choice([v for v in (gold - 1, gold + 1) if 0 <= v < len(levels)] or [gold])
        else:
            L = rng.randrange(len(levels))
        desc = levels[L]["description"]
        new["field"] = {"type": "noul", "question": rng.choice([
            f"{f['question'].strip()} Is the correct level something other than \"{desc}\"?",
            f"Question: \"{f['question'].strip()}\" Is it wrong to answer \"{desc}\"?"])}
        new["gold"] = None if gold is None else (L != gold)
        return _mark(new, "negation", {"source": "score_level", "level": L})
    return None


# ------------------------------------------------------------------------------------------------ (c) bury
CITIES = ["Aberdeen", "Tallinn", "Porto", "Graz", "Tromsø", "Lyon", "Utrecht", "Brno", "Cádiz", "Turku", "Cork", "Malmö", "Ghent", "Bergen",
          "Lucerne", "Trieste", "Bilbao", "Plovdiv", "Kaunas", "Salzburg"]
NOTICES = [
    "The {city} office car park will be resurfaced on {date}; staff should use the overflow car park on {street}.",
    "The quarterly fire drill at the {city} site is scheduled for {date} at {time}.",
    "{name} joins the facilities team in {city} on {date} and will look after meeting-room bookings.",
    "The canteen in the {city} building will open at {time} instead of 08:00 from {date}.",
    "Laptops will receive a security update overnight on {date}; please leave them connected to power.",
    "The intranet search function will be unavailable between {time} and {time2} on {date} for maintenance.",
    "Printer {code} on floor {floor} of the {city} office has been replaced; the old one will be collected on {date}.",
    "The staff social committee in {city} is collecting nominations for the annual volunteering award until {date}.",
    "Visitor badges at the {city} reception now need to be returned by {time}.",
    "The {city} team's weekly stand-up moves from Monday to Wednesday from {date}.",
    "Cycle storage at {street} in {city} will be closed for repairs from {date} for {n} days.",
    "{name} will cover the switchboard in {city} on {date} while the regular operator is on training.",
    "The heating in the {city} annex will be switched to winter mode on {date}.",
    "A charity bake sale raised {cur}{n2} for the local food bank in {city}.",
    "The lift at the {city} office will be serviced on {date}; allow extra time between floors.",
    "Desk booking for the {city} hot-desk area opens every Friday at {time}.",
    "The {city} office will trial recycled paper towels for {n} weeks from {date}.",
    "Parking permits for the {city} site expire on {date} and can be renewed at reception.",
    "The first-aid kit on floor {floor} of the {city} building was restocked on {date}.",
    "{name} is running an optional spreadsheet refresher course in {city} on {date} at {time}.",
    "Water will be shut off at {street}, {city}, on {date} between {time} and {time2}.",
    "The {city} office's new coffee machine takes {n} minutes to warm up in the morning.",
]
STREETS = ["Linden Street", "Quay Road", "Albert Terrace", "Ropewalk", "Chapel Row", "Orchard Lane", "Ferry Street", "Weaver's Yard"]
FIRST = ["Ines", "Tomasz", "Aoife", "Rohan", "Mette", "Kwame", "Lucía", "Anders", "Yuki", "Farah", "Emil", "Nadia", "Oisín", "Priya", "Sven", "Leona"]
LAST = ["Varga", "Oakes", "Halvorsen", "Mendes", "Kerr", "Lindgren", "Osei", "Brandt", "Ruiz", "Novak", "Quinlan", "Ferreira", "Haas", "Duarte"]


def _notice(rng: random.Random, avoid: str) -> str:
    for _ in range(20):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        city = rng.choice(CITIES)
        if name in avoid or city in avoid:
            continue
        m = rng.randint(1, 12)
        date = f"{rng.randint(1, 28)} {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][m - 1]}"
        h = rng.randint(7, 17)
        return rng.choice(NOTICES).format(city=city, date=date, time=f"{h:02d}:{rng.choice(['00', '15', '30', '45'])}",
                                          time2=f"{h + 2:02d}:00", name=name, street=rng.choice(STREETS), code=f"PR-{rng.randint(100, 999)}",
                                          floor=rng.randint(1, 6), n=rng.randint(2, 6), n2=rng.randint(120, 900), cur=rng.choice(["£", "€", "$"]))
    return "The staff newsletter is published on the first working day of each month."


def _block(rng: random.Random, avoid: str, k: int) -> list[str]:
    head = rng.choice(["Office notices (for information):", "Unrelated operational notes:", "General notices:", "Noticeboard:",
                       "Housekeeping announcements:", "Other updates this week:"])
    seen, out = set(), []
    while len(out) < k:
        n = _notice(rng, avoid)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return [head] + [f"- {n}" for n in out]


def _insert_at(lines: list[str], idx: int, block: list[str]) -> list[str]:
    """Insert a block at a paragraph boundary at or after idx (never inside a table or JSON block)."""
    n = len(lines)
    j = idx
    while j < n and not (lines[j].strip() == "" and (j + 1 >= n or not lines[j + 1].lstrip().startswith("|"))):
        j += 1
    if j >= n:
        j = idx
        while j > 0 and lines[j - 1].strip() != "":
            j -= 1
    return lines[:j] + [""] + block + [""] + lines[j:]


def t_bury(row: dict, rng: random.Random) -> dict | None:
    st = row["state"]
    k = rng.randint(4, 8)
    new = copy.deepcopy(row)
    if isinstance(st, dict):
        avoid = json.dumps(st, ensure_ascii=False)
        notes = [x[2:] for x in _block(rng, avoid, k)[1:]]
        items = list(st.items())
        pos = len(items) // 2
        key = next(x for x in ("office_notices", "unrelated_notes", "general_notices", "noticeboard_items") if x not in st)
        new["state"] = dict(items[:pos] + [(key, notes)] + items[pos:])
        return _mark(new, "bury", {"inserted": k, "where": "dict_middle"})
    if not isinstance(st, str):
        return None
    block = _block(rng, st, k)
    if approx_tokens(st) + approx_tokens("\n".join(block)) > TOKEN_CAP:
        return None
    lines = st.split("\n")
    hints = ((row.get("provenance") or {}).get("trap_hints") or {})
    idx, where = None, "middle"
    for d in hints.get("decisive") or []:
        probe = d.strip()[:60]
        if not probe:
            continue
        hit = [i for i, ln in enumerate(lines) if probe in ln]
        if hit:
            idx, where = hit[0] + 1, "after_decisive"
            break
    if idx is None:
        idx = len(lines) // 2
    new["state"] = "\n".join(_insert_at(lines, idx, block))
    # a second, smaller block before the decisive clause buries it from both sides
    if where == "after_decisive" and rng.random() < 0.5:
        lines2 = new["state"].split("\n")
        b2 = _block(rng, st, rng.randint(2, 4))
        start = max(0, idx - 6)
        new["state"] = "\n".join(_insert_at(lines2, start, b2))
    return _mark(new, "bury", {"inserted": k, "where": where})


# ------------------------------------------------------------------------------------------------ (d) qualifier
SWAPS = [(r"\bbusiness days?\b", {"business day": "calendar day", "business days": "calendar days"}),
         (r"\bcalendar days?\b", {"calendar day": "business day", "calendar days": "business days"}),
         (r"\bworking days?\b", {"working day": "calendar day", "working days": "calendar days"}),
         (r"\bexcluding\b", {"excluding": "including"}), (r"\bincluding\b", {"including": "excluding"}),
         (r"\bat least\b", {"at least": "more than"}), (r"\bmore than\b", {"more than": "at least"}),
         (r"\bon or before\b", {"on or before": "before"}), (r"\bon or after\b", {"on or after": "after"}),
         (r"\bor more\b", {"or more": "or less"}), (r"\bper month\b", {"per month": "per year"}), (r"\bper year\b", {"per year": "per month"}),
         (r"\bnet\b", {"net": "gross"}), (r"\bgross\b", {"gross": "net"}), (r"\bpre-tax\b", {"pre-tax": "post-tax"}),
         (r"\bweekdays\b", {"weekdays": "all days"}), (r"\bup to and including\b", {"up to and including": "up to but not including"})]
_PLAIN_DAYS = re.compile(r"\b(\d+) days\b")


def qualifier_variant(text: str, rng: random.Random) -> str | None:
    opts = []
    for pat, rep in SWAPS:
        for m in re.finditer(pat, text, re.I):
            src = m.group(0)
            dst = rep.get(src.lower())
            if dst:
                if src[0].isupper():
                    dst = dst[0].upper() + dst[1:]
                opts.append(text[:m.start()] + dst + text[m.end():])
    if not opts:
        for m in _PLAIN_DAYS.finditer(text):
            opts.append(text[:m.start()] + f"{m.group(1)} {rng.choice(['business', 'working'])} days" + text[m.end():])
    return rng.choice(opts) if opts else None


def t_qualifier(row: dict, rng: random.Random) -> dict | None:
    f = row["field"]
    if f["type"] != "choice" or len(f["options"]) >= MAX_OPTIONS:
        return None
    existing = {_norm(o["text"]) for o in f["options"]}
    base = next((o["text"] for o in f["options"] if o["key"] == row["gold"]), None) if row.get("gold") is not None else None
    pool = [base] if base else [o["text"] for o in rng.sample(f["options"], len(f["options"]))]
    for t in pool:
        v = qualifier_variant(t, rng)
        if v and _norm(v) not in existing:
            new = copy.deepcopy(row)
            taken = {o["key"] for o in new["field"]["options"]}
            opt = {"key": option_key(v, taken), "text": v}
            if any("description" in o for o in new["field"]["options"]):
                opt["description"] = v
            new["field"]["options"].insert(rng.randrange(len(new["field"]["options"]) + 1), opt)
            return _mark(new, "qualifier", {"added_option": opt["key"]})
    return None


TRANSFORMS = {"near_miss": t_near_miss, "negation": t_negation, "bury": t_bury, "qualifier": t_qualifier}


def _mark(new: dict, kind: str, info: dict) -> dict:
    new["_trap"] = (kind, info)
    return new


def make_trap(row: dict, kind: str, seed: int = 0) -> dict | None:
    """One trap variant of `row` (or None when the transform does not apply). Output is a source-D candidate row whose gold is
    re-derived by enumerating the parent's undetermined facts under the transformed question (rederive_gold)."""
    if row.get("gold_kind") != "constructed":
        return None
    rng = rng_for(seed, row["id"], kind)
    out = TRANSFORMS[kind](row, rng)
    if out is None:
        return None
    kind, info = out.pop("_trap")
    prov = dict(out.get("provenance") or {})
    prov.pop("trap_hints", None)
    prov.update({"generator": "scripts/p3/gen_traps.py", "trap": kind, "trap_info": info, "parent_source": row["source"],
                 "parent_dataset": row.get("dataset")})
    if prov.get("group_id"):
        prov["parent_group_id"] = prov.pop("group_id")
    out.update({"id": f"p3-trap-{kind}-{row['id']}", "source": "D", "dataset": "gen_traps", "parent_id": row["id"],
                "difficulty": min(5, int(row.get("difficulty", 3)) + 1), "provenance": prov, "gold_kind": "constructed"})
    try:
        gold, reason, how = rederive_gold(row, out)
    except NoSolver:
        if row.get("gold") is None and kind == "negation" and row["field"]["type"] in ("choice", "score"):
            return None          # could have become decidable; without a solver we cannot tell
        gold, reason, how = out.get("gold"), (out.get("unknown_reason") or row.get("unknown_reason") or "insufficient_evidence"), \
            {"method": "construction"}
    out["gold"] = gold
    out["unknown_reason"] = None if gold is not None else reason
    prov["gold_derivation"] = how
    errs = validate(out)
    if errs:
        raise ValueError(f"trap {out['id']}: {errs}")
    return out


# ------------------------------------------------------------------------------------------------ gold re-derivation
INVALID = "__invalid__"    # a completion with no single answer to the (transformed) question -> the trap is not decided
OUTSIDE = "__outside__"    # a completion whose answer is none of the listed options
PREMISE = "__premise__"    # a completion the question's own premise rules out ("exactly one ... is eligible"): ignored


class NoSolver(LookupError):
    """The parent has no spec / family solver to enumerate (foreign candidate rows)."""


_MODS: dict = {}


def _family_module(row: dict):
    """(family name, module) of a gen_policy row, else raise NoSolver."""
    if row.get("dataset") != "gen_policy" and (row.get("provenance") or {}).get("parent_dataset") != "gen_policy":
        raise NoSolver(row.get("dataset"))
    if not _MODS:
        import gen_policy as gp
        for fam in gp.FAMILY_NAMES:
            _MODS[fam] = gp.MODULES[fam]
    for fam, mod in _MODS.items():
        if row.get("family") in getattr(mod, "FAMILIES", (fam,)):
            return fam, mod
    raise NoSolver(row.get("family"))


def _canon(v) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def regen_trace(row: dict) -> tuple:
    """(world specs, opt_values) of a long_policy unknown row: regenerate its item (gen_one for pool rows, the variants_of rng for
    Stage-2 variants, keyed by provenance seed/index) with long_policy.WORLD_TRACE on and take the entry whose child spec is the row's."""
    import gen_policy as gp
    from gen_policy_families import long_policy as lp
    from gen_policy_families.common import item_rng
    prov = row["provenance"]
    want = _canon({k: v for k, v in prov["spec"].items() if k != "opt_values"})
    seed = prov.get("seed")
    lp.WORLD_TRACE = []
    try:
        if isinstance(seed, str) and seed.startswith("variant:"):
            pid, k = seed[len("variant:"):].rsplit(":", 1)
            for attempt in range(40):
                try:
                    lp.generate(item_rng("variant", pid, int(k), attempt), prov["difficulty"], prov["kind"], True)
                except (ValueError, IndexError, KeyError, ZeroDivisionError):
                    continue
                if any(_canon(e[0]) == want for e in lp.WORLD_TRACE):
                    break
        else:
            gp.gen_one("long_policy", int(seed), int(prov["index"]))
        hits = [e for e in lp.WORLD_TRACE if _canon(e[0]) == want]
    finally:
        lp.WORLD_TRACE = None
    if not hits:
        raise NoSolver(f"long_policy trace not reproduced for {row['id']}")
    return hits[0][1], hits[0][2]


def parent_worlds(parent: dict) -> tuple[list, dict | None, str]:
    """(raw answer of every completion, opt_values mapping keys to canonical answers or None, family)."""
    fam, mod = _family_module(parent)
    spec = (parent.get("provenance") or {}).get("spec")
    if not spec:
        raise NoSolver("no spec")
    if fam == "long_policy" and spec.get("missing"):
        traced, ov = regen_trace(parent)
        return mod.worlds(spec, traced), (spec.get("opt_values") or ov), fam
    return mod.worlds(spec), spec.get("opt_values"), fam


def _answer_key(v, field: dict, ov: dict | None, fam: str):
    """A completion's raw answer (recheck semantics) -> the parent's answer: bool (noul), level (score), option key (choice), or a
    marker. Mirrors tests/test_p3_gen_policy.py::recheck_matches."""
    t = field["type"]
    if isinstance(v, str) and v == "invalid":
        return PREMISE
    if t == "noul":
        return v if isinstance(v, bool) else INVALID
    if t == "score":
        return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(field["levels"]) else INVALID
    opts = field["options"]
    if isinstance(v, str) and v in ("__not_an_option__", "__none__"):
        return OUTSIDE        # that completion's answer is none of the listed options
    if ov:
        hit = [k for k, cv in ov.items() if _canon(cv) == _canon(v)]
        if len(hit) == 1 and any(o["key"] == hit[0] for o in opts):
            return hit[0]
        if isinstance(v, str) and v.startswith("__") or v is None:
            return INVALID
        return OUTSIDE
    if isinstance(v, str) and v.startswith("__") or v is None:
        return INVALID
    if isinstance(v, bool):
        return INVALID
    if isinstance(v, int):
        return opts[v]["key"] if 0 <= v < len(opts) else INVALID
    if fam == "judge_hard" and v == "equal":
        return opts[2]["key"] if len(opts) > 2 else INVALID
    if isinstance(v, str):
        hit = [o["key"] for o in opts if o["text"].strip().lower() == v.strip().lower()]
        return hit[0] if len(hit) == 1 else OUTSIDE
    return INVALID


def _pair_keys(parent: dict, trap: dict) -> tuple:
    """Parent option keys of the two proposed answers of a choice-negation trap (None for a near-miss text that is no option)."""
    texts = {o["text"].strip().lower(): o["key"] for o in parent["field"]["options"]}
    out = []
    for o in trap["field"]["options"][:2]:
        t = re.sub(r"^Answer [12]: ", "", o["text"]).strip().lower()
        out.append(texts.get(t))
    return tuple(out)


def rederive_gold(parent: dict, trap: dict) -> tuple:
    """(gold, unknown_reason, derivation info) of `trap`, a transform of `parent`, from the parent's enumerated completions."""
    raw, ov, fam = parent_worlds(parent)
    pf = parent["field"]
    answers = [_answer_key(v, pf, ov, fam) for v in raw]
    answers = [a for a in answers if a != PREMISE]
    prov = trap["provenance"]
    kind, info = prov["trap"], prov.get("trap_info") or {}
    tf = trap["field"]
    added_text = None
    if kind in ("near_miss", "qualifier"):
        added_text = next((o["text"] for o in tf["options"] if o["key"] == info.get("added_option")), None)
    mapped = []
    for v, a in zip([v for v in raw if not (isinstance(v, str) and v == "invalid")], answers):
        if a == OUTSIDE and kind == "negation" and pf["type"] == "choice":
            mapped.append([o["key"] for o in tf["options"]][3])      # neither proposed answer is right in this completion
        elif a in (INVALID, OUTSIDE):
            mapped.append(a)
        elif kind in ("bury",):
            mapped.append(a)
        elif kind in ("near_miss", "qualifier"):
            # the added option must not also be right in this completion (text-valued answers are compared)
            same = isinstance(v, str) and added_text is not None and added_text.strip().lower() == v.strip().lower()
            mapped.append(INVALID if same else a)
        elif tf["type"] == "noul" and pf["type"] == "noul":
            mapped.append(not a)
        elif pf["type"] == "score":
            mapped.append(a != info["level"])
        elif pf["type"] == "choice":
            k1, k2 = _pair_keys(parent, trap)
            keys = [o["key"] for o in tf["options"]]      # Answer 1, Answer 2, both, neither
            mapped.append(keys[1] if a == k1 else keys[0] if a == k2 else keys[3])
        else:
            raise NoSolver(f"transform {kind} of {pf['type']}")
    how = {"method": "enumerated", "completions": len(mapped)}
    if kind == "negation" and pf["type"] == "choice" and len(set(map(_canon, answers))) > 1:
        # "which proposed answer is NOT supported?" over several completions: "partial" = one proposed answer is possible and the
        # other is not (every completion answers differently -> unknown, but a reader may call the impossible one "not supported")
        inpair = set(answers) & set(_pair_keys(parent, trap))
        how["pair_overlap"] = "none" if not inpair else ("full" if set(answers) <= inpair else "partial")
    if parent.get("gold") is not None and set(map(_canon, answers)) != {_canon(parent["gold"])}:
        how["parent_gold_mismatch"] = sorted(set(map(str, answers)))
    decided = bool(mapped) and not any(m in (INVALID, OUTSIDE) for m in mapped) and len(set(map(_canon, mapped))) == 1
    if decided:
        return mapped[0], None, how
    reason = parent.get("unknown_reason") or "insufficient_evidence"
    if mapped and all(m == OUTSIDE for m in mapped):
        reason = "not_listed"
    return None, reason, how


def build(rows: list[dict], count: int, seed: int = 0, unknown_share: float = 0.15, weights: dict | None = None) -> list[dict]:
    """~count traps: parents sampled evenly across families, one transform per parent (the kind rotates, with fallbacks)."""
    weights = weights or {"near_miss": 0.3, "negation": 0.25, "bury": 0.25, "qualifier": 0.2}
    rows = [r for r in rows if r.get("gold_kind") == "constructed"]
    known = [r for r in rows if r.get("gold") is not None]
    unk = [r for r in rows if r.get("gold") is None]
    rng = rng_for("build", seed)
    n_unk = int(round(count * unknown_share))
    out: list[dict] = []

    def take(pool, n, tag):
        by_fam = defaultdict(list)
        for r in pool:
            by_fam[r["family"]].append(r)
        for f in by_fam.values():
            rng.shuffle(f)
        fams = sorted(by_fam)
        got, i = [], 0
        kinds = list(weights)
        counts = Counter()
        while len(got) < n and any(by_fam.values()):
            f = fams[i % len(fams)]
            i += 1
            if not by_fam[f]:
                continue
            # look at up to 4 parents of this family; prefer one where the most under-quota transform applies
            tot = max(1, len(got))
            want = sorted(kinds, key=lambda k: counts[k] / tot - weights[k])
            looked = []
            best = None
            for _ in range(min(4, len(by_fam[f]))):
                r = by_fam[f].pop()
                cands = {k: t for k in kinds if (t := make_trap(r, k, seed)) is not None}
                looked.append((r, cands))
                if cands and want[0] in cands:
                    best = looked[-1]
                    break
            if best is None:
                best = next((x for x in looked if x[1]), None)
            for x in looked:
                if x is not best:
                    by_fam[f].insert(0, x[0])
            if best is None:
                continue
            cands = best[1]
            k = min(cands, key=lambda k: (counts[k] / tot - weights[k], rng.random()))
            counts[k] += 1
            got.append(cands[k])
        return got
    out += take(unk, n_unk, "unknown")
    out += take(known, count - len(out), "known")
    return out


# ------------------------------------------------------------------------------------------------ downstream gold fixes
FIX_SOURCES = ["data/p3/candidates/D-traps.jsonl", "data/p3/candidates-clean/D-traps.jsonl", "data/p3/pool/shards/pool-*.jsonl",
               "data/p3/pool/heldout-*.jsonl", "data/p3/variants/constructed.jsonl", "data/p3/stream/queue.jsonl",
               "data/p3/stream/queue-variants.jsonl", "data/p3/mine-pulled/*.flagged.jsonl", "data/p3/labeltrain/*.jsonl",
               "data/p3/teacher/results.jsonl", "data/p3/teacher/kept.jsonl", "data/p3/teacher/heldout-results.jsonl"]


def _gold_str(v) -> str:
    if v is None or (isinstance(v, str) and v.lower() in ("unknown", "__unknown__")):
        return "unknown"
    return str(v).lower() if isinstance(v, bool) else str(v)


def label_contradicts(target, gold) -> bool:
    """Does a (teacher) label target disagree with a gold (bool / key / level / None=unknown, label or value form)?"""
    return _gold_str(target) != _gold_str(gold)


def _iter_d_rows(path: Path):
    """(row wrapper, candidate item) for every source-D row of a JSONL file (candidate rows, queue / flagged / verdict wrappers).
    A torn last line (file being appended) is skipped."""
    with open(path) as fh:
        for line in fh:
            if "gen_traps" not in line:
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = w.get("item") if isinstance(w.get("item"), dict) else w
            if it.get("dataset") == "gen_traps" and "field" in it:
                yield w, it


def _variant_base(a_row: dict, seed: str) -> dict | None:
    """The gen_policy Stage-2 variant of `a_row` whose provenance seed is `seed` (variants.py Programmatic.policy path)."""
    import tempfile
    import gen_policy as gp
    k = int(seed.rsplit(":", 1)[1])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "parent.jsonl"
        tmp.write_text(json.dumps(a_row, ensure_ascii=False) + "\n")
        for r in gp.variants_of(tmp, k + 1):
            if r["provenance"].get("seed") == seed:
                return r
    return None


def _regen_policy_row(rid: str, family: str, seed, index) -> dict | None:
    """A gen_policy pool row regenerated from (seed, index) (e.g. the fresh-seed held-out parents, not in A-policy.jsonl)."""
    import gen_policy as gp
    fam = next((f for f in gp.FAMILY_NAMES if family in getattr(gp.MODULES[f], "FAMILIES", (f,))), None)
    if fam is None:
        return None
    try:
        return next((r for r in gp.gen_one(fam, int(seed), int(index)) if r["id"] == rid), None)
    except (RuntimeError, ValueError):
        return None


def resolve_parent(trap: dict, a_rows: dict) -> dict:
    """The parent a trap row was made from: its gen_policy row (a_rows by id, else regenerated from provenance seed/index), or, for
    a Stage-2 trap variant (variants.py traps path: a trap of a fresh gen_policy variant -- an unknown one for "unknown" variants),
    that variant regenerated from its provenance seed. The parent's spec must equal the trap's (NoSolver otherwise)."""
    prov = trap.get("provenance") or {}
    want = _canon(prov.get("spec"))
    ok = (lambda p: p is not None and _canon((p.get("provenance") or {}).get("spec")) == want)
    p = a_rows.get(trap.get("parent_id"))
    seed = str(prov.get("seed") if prov.get("seed") is not None else "")
    if not ok(p) and seed.startswith("variant:"):
        a = a_rows.get(seed[len("variant:"):].rsplit(":", 1)[0])
        p = None
        for src in ([a, dict(a, gold=None, unknown_reason="insufficient_evidence")] if a else []):
            p = _variant_base(src, seed)
            if ok(p):
                break
    if not ok(p) and seed.lstrip("-").isdigit() and prov.get("index") is not None:
        p = _regen_policy_row(trap.get("parent_id"), trap.get("family"), seed, prov["index"])
    if not ok(p):
        raise NoSolver(f"parent of {trap['id']} not found")
    return p


def gold_fixes(sources: list[str], a_policy: list[str], log=print) -> tuple[list[dict], dict]:
    """Re-derive the gold of every source-D row in `sources` (globs, READ-ONLY) -> ([fix rows], stats). A fix row
    {id, old_gold, new_gold, new_unknown_reason, where, action, trap, family, parent_unknown} is written for every occurrence whose
    stored gold differs from the re-derived one. action: "drop" for a teacher-KEPT verdict whose kept label contradicts the new gold,
    else "relabel" (candidate / pool / held-out / queue / constructed-variant copies, and kept verdicts that agree)."""
    import glob as _glob
    occ = []          # (path, wrapper, item)
    for pat in sources:
        for path in sorted(_glob.glob(str(ROOT / pat))):
            for w, it in _iter_d_rows(Path(path)):
                occ.append((str(Path(path).relative_to(ROOT)), w, it))
    need = set()
    for _, _, it in occ:
        need.add(it.get("parent_id"))
        seed = str((it.get("provenance") or {}).get("seed") or "")
        if seed.startswith("variant:"):
            need.add(seed[len("variant:"):].rsplit(":", 1)[0])
    a_rows = {}
    for path in a_policy:
        with open(path) as fh:
            for line in fh:
                rid = line[8:line.find('"', 8)] if line.startswith('{"id": "') else None
                if rid in need:
                    a_rows[rid] = json.loads(line)
    log(f"{len(occ):,} source-D occurrences in {len(sources)} source patterns; {len(a_rows):,} gen_policy parents loaded")
    cache: dict = {}
    fixes, stats = [], Counter()
    for where, w, it in occ:
        key = (it["id"], _canon(it["field"]), _canon((it.get("provenance") or {}).get("spec")))
        if key not in cache:
            try:
                parent = resolve_parent(it, a_rows)
                cache[key] = rederive_gold(parent, it)[:2] + (parent.get("gold") is None,)
            except NoSolver as e:
                cache[key] = e
        res = cache[key]
        if isinstance(res, NoSolver):
            stats["unresolved"] += 1
            continue
        new, reason, punk = res
        stats["rows"] += 1
        if _canon(new) == _canon(it.get("gold")):
            continue
        tag = where
        action = "relabel"
        if "item" in w and "keep" in w:       # a teacher verdict
            tag += "#kept" if w.get("keep") else "#rejected"
            lab = (w.get("label") or {}).get("target") if w.get("label") else ((w.get("labels") or [None])[0])
            if w.get("keep") and label_contradicts(lab, new):
                action = "drop"
        fixes.append({"id": it["id"], "old_gold": it.get("gold"), "new_gold": new, "new_unknown_reason": reason, "where": tag,
                      "action": action, "trap": it["provenance"].get("trap"), "family": it.get("family"), "parent_unknown": punk})
        short = "pool/shards" if "/shards/" in tag else "mine-pulled/*.flagged" if "mine-pulled" in tag else tag.replace("data/p3/", "")
        stats[f"fix:{action}:{short}"] += 1
    return fixes, dict(stats)


def load_gold_fixes(path) -> tuple[dict, set, dict]:
    """(relabel {id: (new_gold, new_unknown_reason)}, drop ids, counts) from a gold-fixes JSONL (build_manifest / make_extra_queue)."""
    relabel, drop = {}, set()
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = json.loads(line)
            if f["action"] == "drop":
                drop.add(f["id"])
            elif f["action"] == "relabel":
                relabel[f["id"]] = (f["new_gold"], f.get("new_unknown_reason"))
    return relabel, drop, {"relabel_ids": len(relabel), "drop_ids": len(drop)}


def apply_gold_fix(row: dict, relabel: dict) -> bool:
    """Set a candidate / verified row's gold (and a constructed label's target) from the relabel map. True when changed."""
    if row.get("id") not in relabel:
        return False
    new, reason = relabel[row["id"]]
    row["gold"] = new
    row["unknown_reason"] = None if new is not None else (reason or row.get("unknown_reason") or "insufficient_evidence")
    lab = row.get("label")
    if isinstance(lab, dict) and lab.get("target_kind") == "constructed":
        row["label"] = dict(lab, target=new)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inputs", nargs="+", default=[str(ROOT / "data/p3/candidates/A-policy.jsonl")])
    ap.add_argument("--out", default=str(ROOT / "data/p3/candidates/D-traps.jsonl"))
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unknown-share", type=float, default=0.15)
    ap.add_argument("--previous", default=None, help="an earlier trap file: write --mapping (every id whose gold changed, old -> new)")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--gold-fixes", default=None, help="instead of generating: re-derive every source-D row found in --fix-sources "
                    "(read-only) and write the fixes JSONL here (build_manifest.py / make_extra_queue.py --gold-fixes)")
    ap.add_argument("--fix-sources", nargs="*", default=FIX_SOURCES)
    a = ap.parse_args(argv)
    if a.gold_fixes:
        fixes, stats = gold_fixes(a.fix_sources, a.inputs)
        Path(a.gold_fixes).parent.mkdir(parents=True, exist_ok=True)
        with open(a.gold_fixes, "w") as fh:
            for f in fixes:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"wrote {len(fixes)} fixes -> {a.gold_fixes}; {stats}")
        return 0
    rows = [r for p in a.inputs for r in read(p)]
    prev = {r["id"]: r for r in read(a.previous)} if a.previous else None
    traps = build(rows, a.count, a.seed, a.unknown_share)
    ids = Counter(t["id"] for t in traps)
    assert max(ids.values()) == 1, "duplicate trap ids"
    n = write(a.out, traps)
    kc = Counter(t["provenance"]["trap"] for t in traps)
    fc = Counter(t["family"] for t in traps)
    unk = sum(t["gold"] is None for t in traps)
    print(f"wrote {n} traps -> {a.out}; kinds {dict(kc)}; families {dict(fc)}; unknown {unk / max(1, n):.1%}")
    if prev is not None and a.mapping:
        changed = []
        for t in traps:
            o = prev.get(t["id"])
            if o is None:
                changed.append({"id": t["id"], "status": "new", "old_gold": None, "new_gold": t["gold"]})
            elif _canon(o["gold"]) != _canon(t["gold"]):
                changed.append({"id": t["id"], "status": "gold_changed", "old_gold": o["gold"], "new_gold": t["gold"],
                                "new_unknown_reason": t["unknown_reason"], "trap": t["provenance"]["trap"], "family": t["family"],
                                "field_type": t["field"]["type"], "parent_unknown": o["gold"] is None})
        gone = [{"id": i, "status": "gone", "old_gold": o["gold"], "new_gold": None} for i, o in prev.items() if i not in ids]
        with open(a.mapping, "w") as fh:
            for c in changed + gone:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"mapping -> {a.mapping}: {Counter(c['status'] for c in changed + gone)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
