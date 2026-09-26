"""Phase-3 Stage 0-I: rendered document images with constructed gold (families `docimg_*`, source I, licence "generated").

We lay out fictional documents in HTML and render them to PNG / WebP with the local headless Chromium (the renderer the GUI
and chart generators use: scripts/p3/gui_templates/render.mjs), then ask typed questions whose answer follows from the
document data we generated:

  documents  invoice, receipt, purchase order, bank-statement snippet, form (checkboxes, consent, signature box), shipping
             label, letter, membership / staff / event card (fictional, never a government ID), timetable
  artefacts  scan look on ~35%: slight rotation, blur, noise, low contrast, paper tint, rubber stamps; handwriting fonts
             for filled-in form values and signatures; receipts in thermal-printer style
  tasks      arith_total / balance_check / change_ok (noul)   do the printed figures add up
             line_calc / max_item / largest_debit (choice)      which line is wrong / largest
             overdue / expired / deadline_passed (noul)         date checks against "today" in the state
             threshold_amt / allowance / budget_ok / overweight / lead_time / access (noul)  document + policy in the state
             currency / doc_type / service / dest_city / signer / tier / ask / payment / deadline (choice)
             blank_field / ticked (choice), signed / consent / adult / international / merchant / overdrawn (noul)
             tt_latest / tt_duration (choice), tt_stops (noul), tt_count / count_over (score)
Unknown variants (~15%, gold null, parent_id = the answerable item): a field the answer needs is covered (sticky note,
ink blot, correction tape), cut off by a cropped scan, or not printed at all; a policy compares amounts in two currencies
with no rate given; a "which is blank / wrong" question where nothing is (false_premise); a payment asked about on page 1
of a multi-page statement. `decide()` reads the document through a Reader that refuses hidden or unprinted fields, so an
item is unknown exactly when the answer needs one; the renderer then proves every field the answer used is fully in view
and on top, and every hidden one is not.

    .venv/bin/python scripts/p3/gen_docimages.py --seed d1 --count 5000 [--out data/p3/candidates/I-docimg.jsonl]
    .venv/bin/python scripts/p3/gen_docimages.py --variant-of parents.jsonl --variants-per 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_charts import (BRAND_A, BRAND_B, E, FIRST, LAST, Missing, P, Skip, bench_image_shas, brand,  # noqa: E402
                        contact_sheets, finish_image, rel_dir, render_pages, rng_for, town)

GENERATOR = "scripts/p3/gen_docimages.py"
IMG_DIR = ROOT / "data" / "p3" / "images" / "docimg"
OUT = ROOT / "data" / "p3" / "candidates" / "I-docimg.jsonl"
UNKNOWN_SHARE = 0.15
VIEW_H = 1900

# ------------------------------------------------------------------------------------------------ content pools
COUNTRIES = ["Netherlands", "Germany", "France", "Spain", "Italy", "Ireland", "Portugal", "Belgium", "Austria", "Denmark",
             "Sweden", "Norway", "Finland", "Poland", "Czechia", "United Kingdom", "Canada", "Australia", "New Zealand",
             "Japan", "Singapore", "South Africa", "Mexico", "Chile"]
STREETS = ["Linden", "Harbour", "Quarry", "Mill", "Orchard", "Station", "Beacon", "Willow", "Granary", "Tanner", "Cedar",
           "Foundry", "Meadow", "Chapel", "Lark", "Copper", "Ferry", "Kiln"]
STREET_T = ["Street", "Road", "Lane", "Way", "Avenue", "Close", "Row", "Square"]
CO_SUFFIX = ["Ltd", "GmbH", "& Co", "Studio", "Supplies", "Logistics", "Partners", "Works", "Trading", "B.V."]
CURRENCIES = {"EUR": "€", "GBP": "£", "USD": "$", "CHF": "CHF ", "SEK": "kr ", "CAD": "C$", "AUD": "A$", "INR": "₹", "NZD": "NZ$"}
AMBIG = {"USD", "CAD", "AUD", "NZD"}          # "$"-style symbols: the code is always printed as well
INVOICE_ITEMS = ["Consulting hours", "Website maintenance", "Printer paper A4 (box)", "Toner cartridge", "Office chair",
                 "Delivery fee", "Software licence (annual)", "Cleaning service", "Catering for 20", "Translation (per page)",
                 "Design mock-ups", "Server hosting (month)", "Training workshop", "Safety boots", "Cable ties (pack)",
                 "LED panel light", "Oak desk", "Monitor 27 inch", "Wireless keyboard", "Room hire (half day)",
                 "Photography session", "Data backup service", "Legal review", "Plant care visit", "Window cleaning"]
RECEIPT_ITEMS = ["Flat white", "Croissant", "Sparkling water", "Pasta salad", "Bananas 1 kg", "Oat milk", "Sourdough loaf",
                 "Cheddar 200 g", "Cherry tomatoes", "Green tea", "Dark chocolate", "Soap refill", "AA batteries 4x",
                 "Notebook A5", "Ballpoint pens 3x", "Orange juice 1 L", "Rice 2 kg", "Olive oil 500 ml", "Muesli",
                 "Club sandwich", "Soup of the day", "Lemonade", "Espresso", "Carrot cake", "Tap shoes polish"]
PO_ITEMS = ["Steel bolts M8 (100)", "Pallet wrap roll", "Safety gloves (pair)", "Cardboard boxes 40x30", "Hydraulic oil 5 L",
            "Label rolls", "Copper wire 50 m", "Hard hats", "Hi-vis vests", "Packing tape", "Cable drums", "Hex nuts M10 (200)",
            "Floor cleaner 10 L", "Ear defenders", "Shelf brackets", "Drill bits set", "Zip bags (1000)", "Air filters"]
DOC_TYPES = {"invoice": "invoice", "receipt": "receipt", "po": "purchase order", "statement": "bank statement",
             "form": "application form", "label": "shipping label", "letter": "letter", "card": "membership card",
             "timetable": "timetable"}
HAND_FONTS = ["'Bradley Hand', cursive", "'Noteworthy', cursive", "'Marker Felt', fantasy", "'Chalkboard SE', cursive"]
SIG_FONTS = ["'Snell Roundhand', cursive", "'Savoye LET', cursive", "'Brush Script MT', cursive", "'Bradley Hand', cursive"]
DOC_FONTS = ["Helvetica, Arial, sans-serif", "Georgia, serif", "'Times New Roman', serif", "Verdana, sans-serif",
             "'Avenir Next', sans-serif", "Palatino, serif", "Optima, sans-serif", "'Trebuchet MS', sans-serif",
             "'American Typewriter', serif", "Baskerville, serif", "Futura, sans-serif"]
ACCENTS = ["#1f4e79", "#7a1f3d", "#0f6b4f", "#5b3c99", "#a3470f", "#2d2d2d", "#006d77", "#8a1c1c", "#34568b"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]


def company(rng):
    return brand(rng) + " " + rng.choice(CO_SUFFIX)


def person(rng):
    return rng.choice(FIRST) + " " + rng.choice(LAST)


def street(rng):
    return f"{rng.randint(1, 240)} {rng.choice(STREETS)} {rng.choice(STREET_T)}"


def postcode(rng):
    return rng.choice([f"{rng.randint(1000, 9999)} {rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}{rng.choice('ABCDEFGHJKLMNPRSTVWXZ')}",
                       f"{rng.randint(10000, 99999)}", f"{rng.choice('BCDFGHKLMNPRSTW')}{rng.randint(1, 29)} {rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"])


def rdate(rng, y0=2024, y1=2027):
    return dt.date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))


def dfmt(d: dt.date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "long":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "us":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"


def money(v: float, cur: str, style: str) -> str:
    s = f"{abs(v):,.2f}"
    neg = "-" if v < 0 else ""
    if style == "sym":
        return f"{neg}{CURRENCIES[cur]}{s}"
    if style == "code":
        return f"{neg}{s} {cur}"
    return f"{neg}{s}"


def r2(x):
    return round(x + 1e-9, 2)


# ------------------------------------------------------------------------------------------------ spec + reader
def hidden_keys(sp) -> set:
    out = set(sp.get("hide", {}))
    crop = sp.get("crop_at")
    if crop:
        order = sp["section_order"]
        cut = set(order[order.index(crop):])
        out |= {k for k, s in sp["sections"].items() if s in cut}
    return out


class Reader:
    """Reads printed fields; refuses hidden or unprinted ones (Missing). Records what was read."""

    def __init__(self, sp):
        self.sp = sp
        self.hidden = hidden_keys(sp)
        self.used = []

    def __call__(self, key):
        if key not in self.sp["f"] or key in self.hidden:
            raise Missing("insufficient_evidence")
        if key not in self.used:
            self.used.append(key)
        return self.sp["f"][key]

    def has(self, key) -> bool:
        """Printed and visible (reading it)."""
        try:
            self(key)
            return True
        except Missing:
            return False


def num(x):
    return float(str(x).replace(",", "").split()[0]) if not isinstance(x, (int, float)) else float(x)


def D(s):
    return dt.date.fromisoformat(s)


def solve(sp, task, p, R: Reader):
    """The answer (option text / bool / level) from the document as the image shows it."""
    if task == "arith_total":
        tot = sum(num(R(k)) for k in p["parts"])
        return abs(tot - num(R(p["total"]))) < 0.005
    if task in ("balance_check", "largest_debit", "count_over") and int(R(p.get("pages", "pages"))) > 1:
        raise Missing("insufficient_evidence")
    if task == "balance_check":
        v = num(R(p["open"])) + sum(num(R(k)) for k in p["credits"]) - sum(num(R(k)) for k in p["debits"])
        return abs(v - num(R(p["close"]))) < 0.005
    if task == "change_ok":
        return abs(num(R(p["tendered"])) - num(R(p["total"])) - num(R(p["change"]))) < 0.005
    if task == "line_calc":
        bad = [lab for lab, q, u, a in p["rows"] if abs(num(R(q)) * num(R(u)) - num(R(a))) >= 0.005]
        if not bad:
            raise Missing("false_premise")
        return bad[0] if len(bad) == 1 else "__multi__"
    if task in ("max_item", "largest_debit"):
        vals = [(num(R(k)), lab) for lab, k in p["rows"]]
        best = max(vals)
        return "__tie__" if sum(1 for v, _ in vals if v == best[0]) > 1 else best[1]
    if task == "overdue":
        if p.get("paid") and R.has(p["paid"]):
            return False
        if "due" in p and p["due"] in R.sp["f"]:
            due = D(R(p["due"]))
        else:
            due = D(R(p["issue"])) + dt.timedelta(days=int(R(p["terms"])))
        return D(p["today"]) > due
    if task in ("expired", "deadline_passed"):
        return D(p["today"]) > D(R(p["key"]))
    if task == "late_band":
        if R.has(p["paid"]):
            return 0
        due = D(R("due")) if "due" in R.sp["f"] else D(R("issue")) + dt.timedelta(days=int(R("terms")))
        late = (D(p["today"]) - due).days
        return 0 if late <= 0 else 1 if late <= 30 else 2 if late <= 60 else 3
    if task == "count_items":
        return sum(1 for k in p["rows"] if num(R(k)) > p["T"])
    if task == "threshold_amt":
        amt = num(R(p["amount"]))
        cur = R(p["cur"])
        if cur != p["limit_cur"]:
            rate = (p.get("rates") or {}).get(cur)
            if rate is None:
                raise Missing("insufficient_evidence")
            amt = amt * rate
        return amt > p["limit"]
    if task == "allowance":
        return num(R(p["total"])) <= p["per_person"] * p["people"]
    if task == "budget_ok":
        return num(R(p["total"])) <= p["budget"]
    if task == "overweight":
        return num(R(p["key"])) > p["limit"]
    if task == "lead_time":
        return D(p["today"]) + dt.timedelta(days=p["days"]) <= D(R(p["key"]))
    if task == "access":
        return R(p["tier"]) in p["allowed"] and D(p["today"]) <= D(R(p["until"]))
    if task in ("value", "currency", "dest_city", "signer", "tier", "ask", "payment", "deadline"):
        v = R(p["key"])
        if v in ("", None):
            raise Missing("insufficient_evidence")
        return str(v)
    if task == "doc_type":
        R("title")
        return DOC_TYPES[sp["doc"]]
    if task == "tax_rate":
        return abs(num(R(p["tax"])) - r2(num(R(p["sub"])) * p["rate"] / 100)) < 0.015
    if task == "signed":
        return R(p["key"]) != ""
    if task == "consent":
        return bool(R(p["key"]))
    if task == "blank_field":
        blank = [lab for lab, k in p["fields"] if R(k) == ""]
        if not blank:
            raise Missing("false_premise")
        return blank[0] if len(blank) == 1 else "__multi__"
    if task in ("ticked", "service"):
        on = [lab for lab, k in p["group"] if R(k)]
        if not on:
            raise Missing("insufficient_evidence")
        return on[0] if len(on) == 1 else "__multi__"
    if task == "adult":
        dob = R(p["key"])
        if dob == "":
            raise Missing("insufficient_evidence")
        b, t = D(dob), D(p["today"])
        age = t.year - b.year - ((t.month, t.day) < (b.month, b.day))
        return age >= p["age"]
    if task == "international":
        return R(p["from"]) != R(p["to"])
    if task == "merchant":
        found = False
        unsure = False
        for k in p["rows"]:
            try:
                if p["name"] in R(k):
                    found = True
            except Missing:
                unsure = True
        if found:
            return True
        if unsure or int(R(p["pages"])) > 1:
            raise Missing("insufficient_evidence")
        return False
    if task == "overdrawn":
        if any(num(R(k)) < 0 for k in p["rows"]):
            return True
        if int(R("pages")) > 1:
            raise Missing("insufficient_evidence")
        return False
    if task == "count_over":
        return sum(1 for k in p["rows"] if num(R(k)) > p["T"])
    if task == "tt_latest":
        best = None
        for c in p["cols"]:
            a = R(f"tt_{p['to']}_{c}")
            dep = R(f"tt_{p['frm']}_{c}")
            if a == "—" or dep == "—":
                continue
            if hm(a) <= p["by"] and (best is None or hm(dep) > hm(best)):
                best = dep
        return best if best is not None else "__none__"
    if task == "tt_stops":
        return R(f"tt_{p['row']}_{p['col']}") != "—"
    if task == "tt_count":
        return sum(1 for c in p["cols"] if (lambda v: v != "—" and hm(v) < p["before"])(R(f"tt_{p['row']}_{c}")))
    if task == "tt_duration":
        a, b = R(f"tt_{p['frm']}_{p['col']}"), R(f"tt_{p['to']}_{p['col']}")
        if "—" in (a, b):
            raise Missing("false_premise")
        return f"{hm(b) - hm(a)} min"
    raise KeyError(task)


def hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def decide(sp, task, p):
    """(gold, unknown_reason, keys read)."""
    R = Reader(sp)
    try:
        g = solve(sp, task, p, R)
    except Missing as m:
        return None, m.reason, R.used
    if isinstance(g, str) and g.startswith("__"):
        return "__ill__", None, R.used
    return g, None, R.used


# ------------------------------------------------------------------------------------------------ page chrome
def style(rng, doc):
    scan = rng.random() < 0.35
    return {"font": rng.choice(DOC_FONTS), "fs": rng.choice([13, 14, 14, 15, 16]), "accent": rng.choice(ACCENTS),
            "paper": rng.choice(["#ffffff", "#fffdf7", "#fbfaf5", "#f7f5ef"]) if not scan else rng.choice(["#f4f1e8", "#efece4", "#f6f3ea"]),
            "ink": rng.choice(["#111111", "#222222", "#2a2a2a"]) if not (scan and rng.random() < 0.4) else "#4a4a4a",
            "scan": scan, "rot": round(rng.uniform(-1.2, 1.2), 2) if scan else 0.0,
            "blur": rng.choice([0, 0.3, 0.5]) if scan else 0, "noise": rng.choice([4, 6, 8]) if scan else 0,
            "contrast": rng.choice([1.0, 0.85, 0.75]) if scan else 1.0, "hand": rng.choice(HAND_FONTS),
            "sig": rng.choice(SIG_FONTS), "layout": rng.randrange(3), "bed": rng.choice(["#d8d6d0", "#c9c7c2", "#e6e4df", "#b9b7b2"])}


def V(sp, key, extra_style="", cls="v"):
    """A printed value with its element tag; hidden keys get an opaque cover."""
    if key not in sp["f"]:
        return ""
    val = sp["f"][key]
    txt = sp.get("disp", {}).get(key, val)
    inner = f'<span class="{cls}" data-el="f_{E(key)}" style="{extra_style}">{E(str(txt))}</span>'
    how = sp.get("hide", {}).get(key)
    if how:
        inner = f'<span style="position:relative;display:inline-block">{inner}{cover_html(how)}</span>'
    return inner


def cover_html(how):
    if how == "sticky":
        st = "background:#f7e27a;box-shadow:1px 2px 4px #0005;transform:rotate(-2deg)"
    elif how == "ink":
        st = "background:radial-gradient(circle at 40% 45%,#1d2a55 0,#1d2a55 55%,#2b3a6e 70%,#2b3a6e 100%);border-radius:45% 55% 50% 40%"
    elif how == "tape":
        st = "background:repeating-linear-gradient(90deg,#fbfbf8 0,#f1f1ec 3px,#fbfbf8 6px);box-shadow:0 0 2px #0003"
    else:
        st = "background:#6b4a2b;border-radius:40%;opacity:1"
    return f'<i style="position:absolute;left:-9px;right:-9px;top:-5px;bottom:-5px;{st}"></i>'


def stamp_html(sp):
    s = sp.get("stamp")
    if not s:
        return ""
    tag = ' data-el="f_paid"' if s["text"] == "PAID" and sp["f"].get("paid") else ""
    return (f'<div{tag} style="position:absolute;{s["pos"]};transform:rotate({s["rot"]}deg);border:3px double {s["col"]};color:{s["col"]};'
            f'padding:4px 10px;font:bold {s["size"]}px/1.1 Courier, monospace;letter-spacing:2px;opacity:.72;text-align:center">'
            f'{E(s["text"])}{"<br><span style=font-size:11px>" + E(s["sub"]) + "</span>" if s.get("sub") else ""}</div>')


def page(sp, body, width=None, extra_css=""):
    st = sp["style"]
    w = width or sp["size"][0]
    bed_pad = 26 if st["scan"] else 0
    bg = st["bed"] if st["scan"] else st["paper"]
    if sp.get("stamp") and "bottom" in sp["stamp"]["pos"]:
        body += '<div style="height:64px"></div>'
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>'
            f'html,body{{margin:0;background:{bg}}} body{{padding:{bed_pad}px}} *{{box-sizing:border-box}}'
            f'.page{{position:relative;width:{w}px;background:{st["paper"]};color:{st["ink"]};font-family:{st["font"]};font-size:{st["fs"]}px;'
            f'line-height:1.35;padding:34px 40px;transform:rotate({st["rot"]}deg);{"box-shadow:0 2px 8px #0004;" if st["scan"] else ""}}}'
            f'.acc{{color:{st["accent"]}}} .lab{{color:#666;font-size:.85em;text-transform:uppercase;letter-spacing:.04em}}'
            f'table{{border-collapse:collapse;width:100%}} td,th{{padding:5px 6px;text-align:left;vertical-align:top}}'
            f'th{{border-bottom:2px solid {st["accent"]};font-size:.85em;text-transform:uppercase;color:#555}}'
            f'td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.l td{{border-bottom:1px solid #ddd}}'
            f'.hand{{font-family:{st["hand"]};font-size:1.12em;color:#1b2f6b}} .sig{{font-family:{st["sig"]};font-size:1.9em;color:#14285f}}'
            f'.box{{display:inline-block;min-width:180px;border-bottom:1px solid #777;padding:0 4px;min-height:1.3em}}'
            f'.cb{{display:inline-block;width:15px;height:15px;border:1.5px solid #333;margin-right:6px;vertical-align:-2px;text-align:center;line-height:13px;font-weight:bold}}'
            f'{extra_css}</style></head><body><div class="page" data-el="page">{body}{stamp_html(sp)}</div></body></html>')


def maybe_stamp(rng, sp, texts, pos_choices):
    if rng.random() < (0.45 if sp["style"]["scan"] else 0.15):
        t = rng.choice(texts)
        sp["stamp"] = {"text": t, "sub": dfmt(rdate(rng), "short").upper() if t in ("RECEIVED", "SCANNED", "FILED") else "",
                       "pos": rng.choice(pos_choices), "rot": rng.choice([-14, -9, -6, 7, 11]),
                       "col": rng.choice(["#b3261e", "#1f4fa3", "#6a2c91"]), "size": rng.choice([18, 22, 26])}


def sec(name, html_):
    return f'<div data-el="sec_{name}">{html_}</div>'


# ------------------------------------------------------------------------------------------------ invoice / receipt / PO
def items_block(rng, pool, n, lo, hi, qty_hi=6):
    names = rng.sample(pool, n)
    rows = []
    for nm in names:
        q = rng.randint(1, qty_hi)
        u = r2(rng.uniform(lo, hi))
        rows.append([nm, q, u, r2(q * u)])
    return rows


def make_invoice(rng, sid):
    sp = {"id": sid, "doc": "invoice", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "invoice")
    sp["style"] = st
    W = rng.choice([720, 780, 840, 900, 960])
    cur = rng.choice(list(CURRENCIES))
    mstyle = rng.choice(["sym", "code", "plain"])
    if cur in AMBIG and mstyle == "sym":
        mstyle = rng.choice(["code", "plain"])
    n = rng.randint(2, 6)
    items = items_block(rng, INVOICE_ITEMS, n, 8, 900)
    issue = rdate(rng)
    has_due = rng.random() < 0.75
    terms = rng.choice([14, 21, 30, 45, 60])
    rate = rng.choice([0, 5, 7, 10, 12, 19, 20, 21, 25])
    sub = r2(sum(r[3] for r in items))
    tax = r2(sub * rate / 100)
    total = r2(sub + tax)
    f = sp["f"]
    f["title"] = rng.choice(["INVOICE", "Invoice", "TAX INVOICE", "Sales invoice"])
    f["seller"] = company(rng)
    f["buyer"] = company(rng)
    f["number"] = rng.choice(["INV-", "", "No. ", "F"]) + str(rng.randint(10000, 99999))
    f["issue"] = issue.isoformat()
    ds = rng.choice(["iso", "long", "us", "short"])
    sp["disp"]["issue"] = dfmt(issue, ds)
    if has_due:
        f["due"] = (issue + dt.timedelta(days=terms)).isoformat()
        sp["disp"]["due"] = dfmt(issue + dt.timedelta(days=terms), ds)
    else:
        f["terms"] = terms
        sp["disp"]["terms"] = f"{terms} days from invoice date"
    f["cur"] = cur
    sp["disp"]["cur"] = cur
    for i, (nm, q, u, a) in enumerate(items):
        f[f"it_{i}_desc"], f[f"it_{i}_qty"], f[f"it_{i}_unit"], f[f"it_{i}_amount"] = nm, q, u, a
        sp["disp"][f"it_{i}_unit"] = money(u, cur, mstyle)
        sp["disp"][f"it_{i}_amount"] = money(a, cur, mstyle)
    f["sub"], f["rate"], f["tax"], f["total"] = sub, rate, tax, total
    sp["disp"].update({"sub": money(sub, cur, mstyle), "tax": money(tax, cur, mstyle), "total": money(total, cur, mstyle),
                       "rate": f"{rate}%"})
    sp["n_items"] = n
    sp["meta"] = {"mstyle": mstyle, "W": W, "tax_word": rng.choice(["VAT", "Tax", "Sales tax", "GST"]), "addr": [street(rng), town(rng), postcode(rng)],
                  "baddr": [street(rng), town(rng), postcode(rng)], "bank": f"Account {rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{rng.randint(100, 999)}"}
    if rng.random() < 0.15:
        f["paid"] = "PAID"
    # errors to check: a wrong total or a wrong line amount (never both)
    err = rng.random()
    if err < 0.4:
        delta = rng.choice([items[0][3], 10.0, 100.0, 1.0, 0.9, -10.0, 50.0])
        f["total"] = r2(total + delta)
        sp["disp"]["total"] = money(f["total"], cur, mstyle)
    elif err < 0.6:
        i = rng.randrange(n)
        a = r2(items[i][3] + rng.choice([10.0, -5.0, 1.0, 100.0, items[i][2]]))
        if a > 0:
            f[f"it_{i}_amount"] = a
            sp["disp"][f"it_{i}_amount"] = money(a, cur, mstyle)
            f["sub"] = r2(sum(f[f"it_{k}_amount"] for k in range(n)))
            f["tax"] = r2(f["sub"] * rate / 100)
            f["total"] = r2(f["sub"] + f["tax"])
            sp["disp"].update({"sub": money(f["sub"], cur, mstyle), "tax": money(f["tax"], cur, mstyle), "total": money(f["total"], cur, mstyle)})
    elif err < 0.72 and rate:
        f["tax"] = r2(f["sub"] * rng.choice([r for r in (5, 7, 10, 12, 19, 20, 21, 25) if r != rate]) / 100)
        f["total"] = r2(f["sub"] + f["tax"])
        sp["disp"].update({"tax": money(f["tax"], cur, mstyle), "total": money(f["total"], cur, mstyle)})
    sp["sections"] = {**{k: "head" for k in ("title", "seller", "number", "issue", "due", "terms", "cur", "paid")},
                      "buyer": "parties", **{k: "items" for k in f if k.startswith("it_")},
                      **{k: "totals" for k in ("sub", "rate", "tax", "total")}}
    sp["section_order"] = ["head", "parties", "items", "totals", "foot"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    if "paid" in f:
        sp["stamp"] = {"text": "PAID", "sub": "", "pos": "right:46px;bottom:12px", "rot": -12, "col": "#b3261e", "size": 30}
    else:
        maybe_stamp(rng, sp, ["RECEIVED", "COPY", "SCANNED"], ["right:46px;bottom:12px", "left:40px;bottom:10px"])
    return sp


def html_invoice(sp):
    f, m, st = sp["f"], sp["meta"], sp["style"]
    show_line = m["mstyle"] != "sym" or rngless(sp, 2)
    cur_line = f'<div class="lab" style="margin-top:4px">Currency: {V(sp, "cur")}</div>' if show_line else ""
    due = (f'<tr><td class="lab">Due date</td><td>{V(sp, "due")}</td></tr>' if "due" in f else
           f'<tr><td class="lab">Payment terms</td><td>{V(sp, "terms")}</td></tr>')
    head = (f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'<div><div style="font-size:1.6em;font-weight:bold" class="acc">{V(sp, "seller")}</div>'
            f'<div>{E(m["addr"][0])}<br>{E(m["addr"][2])} {E(m["addr"][1])}</div></div>'
            f'<div style="text-align:right"><div style="font-size:1.9em;letter-spacing:.05em;font-weight:bold">{V(sp, "title")}</div>'
            f'<table style="width:auto;margin-left:auto"><tr><td class="lab">Invoice no.</td><td>{V(sp, "number")}</td></tr>'
            f'<tr><td class="lab">Date</td><td>{V(sp, "issue")}</td></tr>{due}</table>{cur_line}</div></div>')
    # with symbols and no currency line, the currency evidence is the printed total itself
    total_html = V(sp, "total") if show_line else f'<span data-el="f_cur" style="display:inline-block">{V(sp, "total")}</span>'
    parties = (f'<div style="margin:22px 0 14px"><div class="lab">Bill to</div><div style="font-weight:bold">{V(sp, "buyer")}</div>'
               f'<div>{E(m["baddr"][0])}<br>{E(m["baddr"][2])} {E(m["baddr"][1])}</div></div>')
    rows = "".join(f'<tr class="l"><td>{V(sp, f"it_{i}_desc")}</td><td class="n">{V(sp, f"it_{i}_qty")}</td>'
                   f'<td class="n">{V(sp, f"it_{i}_unit")}</td><td class="n">{V(sp, f"it_{i}_amount")}</td></tr>' for i in range(sp["n_items"]))
    items = f'<table><tr><th>Description</th><th class="n">Qty</th><th class="n">Unit price</th><th class="n">Amount</th></tr>{rows}</table>'
    totals = (f'<table style="width:48%;margin:14px 0 0 auto"><tr><td>Subtotal</td><td class="n">{V(sp, "sub")}</td></tr>'
              f'<tr><td>{E(m["tax_word"])} {V(sp, "rate")}</td><td class="n">{V(sp, "tax")}</td></tr>'
              f'<tr style="font-weight:bold;border-top:2px solid #333"><td>Total due</td><td class="n">{total_html}</td></tr></table>')
    foot = f'<div style="margin-top:26px;font-size:.85em;color:#555">Please pay by bank transfer to {E(m["bank"])}. Thank you for your business.</div>'
    return page(sp, sec("head", head) + sec("parties", parties) + sec("items", items) + sec("totals", totals) + sec("foot", foot), m["W"])


def rngless(sp, k):
    """Deterministic per-spec coin (render-time choices must come from the spec)."""
    return int(hashlib.sha1(sp["id"].encode()).hexdigest(), 16) % k == 0


def make_receipt(rng, sid):
    sp = {"id": sid, "doc": "receipt", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "receipt")
    st["font"] = rng.choice(["Menlo, monospace", "'Courier New', monospace", "Courier, monospace", "'Andale Mono', monospace"])
    st["fs"] = rng.choice([13, 14, 15])
    sp["style"] = st
    W = rng.choice([340, 370, 400, 430])
    cur = rng.choice(["EUR", "GBP", "CHF", "SEK", "USD", "CAD"])
    n = rng.randint(2, 7)
    items = items_block(rng, RECEIPT_ITEMS, n, 0.8, 24, qty_hi=3)
    rate = rng.choice([0, 6, 7, 9, 10, 13, 20])
    sub = r2(sum(r[3] for r in items))
    tax = r2(sub * rate / (100 + rate))       # prices include tax: tax shown for information
    total = sub
    f = sp["f"]
    f["title"] = rng.choice(["RECEIPT", "Receipt", "SALES RECEIPT", "Till receipt"])
    f["store"] = brand(rng) + " " + rng.choice(["Market", "Café", "Deli", "Grocer", "Kiosk", "Pharmacy"])
    f["number"] = f"{rng.randint(100, 999)}-{rng.randint(10000, 99999)}"
    day = rdate(rng)
    f["date"] = day.isoformat()
    sp["disp"]["date"] = dfmt(day, rng.choice(["iso", "short"])) + f"  {rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}"
    f["cur"] = cur
    for i, (nm, q, u, a) in enumerate(items):
        f[f"it_{i}_desc"], f[f"it_{i}_qty"], f[f"it_{i}_unit"], f[f"it_{i}_amount"] = nm, q, u, a
        sp["disp"][f"it_{i}_amount"] = f"{a:,.2f}"
        sp["disp"][f"it_{i}_unit"] = f"{u:,.2f}"
    f["total"], f["tax"] = total, tax
    sp["disp"]["total"] = f"{total:,.2f}"
    sp["disp"]["tax"] = f"{tax:,.2f}"
    pay = rng.choice(["card", "card", "cash", "mobile wallet", "voucher"])
    f["payment"] = pay
    sp["disp"]["payment"] = {"card": f"CARD **** {rng.randint(1000, 9999)}", "cash": "CASH", "mobile wallet": "MOBILE PAY",
                             "voucher": "GIFT VOUCHER"}[pay]
    if pay == "cash":
        tend = r2(max(total, 5 * (int(total / 5) + rng.randint(1, 4))))
        f["tendered"] = tend
        chg = r2(tend - total)
        if rng.random() < 0.35:
            chg = r2(chg + rng.choice([1.0, -1.0, 0.5, 10.0, -0.1]))
        f["change"] = chg
        sp["disp"]["tendered"] = f"{tend:,.2f}"
        sp["disp"]["change"] = f"{chg:,.2f}"
    if rng.random() < 0.35:
        delta = rng.choice([items[-1][3], 1.0, 0.1, 2.0, -1.0])
        f["total"] = r2(total + delta)
        sp["disp"]["total"] = f"{f['total']:,.2f}"
        if pay == "cash" and "change" in f:
            pass
    sp["n_items"] = n
    sp["meta"] = {"W": W, "addr": f"{street(rng)}, {town(rng)}", "rate": rate}
    sp["sections"] = {**{k: "head" for k in ("title", "store", "number", "date", "cur")}, **{k: "items" for k in f if k.startswith("it_")},
                      **{k: "totals" for k in ("total", "tax", "payment", "tendered", "change")}}
    sp["section_order"] = ["head", "items", "totals", "foot"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    maybe_stamp(rng, sp, ["COPY", "DUPLICATE"], ["right:16px;bottom:16px"])
    return sp


def html_receipt(sp):
    f, m = sp["f"], sp["meta"]
    dash = '<div style="border-top:1px dashed #555;margin:8px 0"></div>'
    head = (f'<div style="text-align:center"><div style="font-size:1.3em;font-weight:bold">{V(sp, "store")}</div><div>{E(m["addr"])}</div>'
            f'<div style="margin-top:6px">{V(sp, "title")} {V(sp, "number")}</div><div>{V(sp, "date")}</div>'
            f'<div>Prices in {V(sp, "cur")}</div></div>{dash}')
    rows = "".join(f'<tr><td>{V(sp, f"it_{i}_desc")}<br><span style="color:#555">{V(sp, f"it_{i}_qty")} x {V(sp, f"it_{i}_unit")}</span></td>'
                   f'<td class="n">{V(sp, f"it_{i}_amount")}</td></tr>' for i in range(sp["n_items"]))
    items = f'<table>{rows}</table>{dash}'
    tot = (f'<table><tr style="font-weight:bold;font-size:1.15em"><td>TOTAL</td><td class="n">{V(sp, "total")}</td></tr>'
           f'<tr><td>incl. tax {m["rate"]}%</td><td class="n">{V(sp, "tax")}</td></tr>'
           f'<tr><td>Paid by</td><td class="n">{V(sp, "payment")}</td></tr>')
    if "tendered" in f:
        tot += f'<tr><td>Cash given</td><td class="n">{V(sp, "tendered")}</td></tr><tr><td>Change</td><td class="n">{V(sp, "change")}</td></tr>'
    tot += "</table>"
    foot = f'{dash}<div style="text-align:center">Thank you!<br>Keep this receipt for returns.</div>'
    css = f'.page{{padding:22px 20px;box-shadow:0 1px 4px #0003}}'
    return page(sp, sec("head", head) + sec("items", items) + sec("totals", tot) + sec("foot", foot), m["W"], css)


def make_po(rng, sid):
    sp = {"id": sid, "doc": "po", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "po")
    sp["style"] = st
    W = rng.choice([760, 820, 880, 940])
    cur = rng.choice(["EUR", "GBP", "USD", "CHF", "SEK"])
    n = rng.randint(2, 6)
    items = items_block(rng, PO_ITEMS, n, 2, 400, qty_hi=40)
    f = sp["f"]
    f["title"] = rng.choice(["PURCHASE ORDER", "Purchase Order", "Purchase order"])
    f["number"] = f"PO-{rng.randint(1000, 99999)}"
    f["buyer"] = company(rng)
    f["vendor"] = company(rng)
    d0 = rdate(rng)
    f["date"] = d0.isoformat()
    f["required"] = (d0 + dt.timedelta(days=rng.randint(7, 60))).isoformat()
    ds = rng.choice(["iso", "long", "short"])
    sp["disp"]["date"] = dfmt(d0, ds)
    sp["disp"]["required"] = dfmt(D(f["required"]), ds)
    f["cur"] = cur
    for i, (nm, q, u, a) in enumerate(items):
        f[f"it_{i}_desc"], f[f"it_{i}_qty"], f[f"it_{i}_unit"], f[f"it_{i}_amount"] = nm, q, u, a
        sp["disp"][f"it_{i}_unit"] = f"{u:,.2f}"
        sp["disp"][f"it_{i}_amount"] = f"{a:,.2f}"
    ship = r2(rng.choice([0, 0, 25, 40, 75.5]))
    f["ship"] = ship
    sp["disp"]["ship"] = f"{ship:,.2f}"
    total = r2(sum(r[3] for r in items) + ship)
    if rng.random() < 0.35:
        total = r2(total + rng.choice([items[0][3], 10.0, 100.0, -20.0, 1.0]))
    f["total"] = total
    sp["disp"]["total"] = f"{total:,.2f} {cur}"
    if rng.random() < 0.2:
        i = rng.randrange(n)
        a = r2(items[i][3] + rng.choice([10.0, -5.0, 100.0, items[i][2]]))
        if a > 0:
            f[f"it_{i}_amount"] = a
            sp["disp"][f"it_{i}_amount"] = f"{a:,.2f}"
    appr = person(rng)
    f["approver"] = appr
    f["sig"] = appr if rng.random() < 0.6 else ""
    sp["n_items"] = n
    sp["meta"] = {"W": W, "vaddr": f"{street(rng)}, {town(rng)}", "ship_to": f"{street(rng)}, {town(rng)} {postcode(rng)}"}
    sp["sections"] = {**{k: "head" for k in ("title", "number", "buyer", "vendor", "date", "required", "cur")},
                      **{k: "items" for k in f if k.startswith("it_")}, "ship": "totals", "total": "totals",
                      "approver": "sign", "sig": "sign"}
    sp["section_order"] = ["head", "items", "totals", "sign"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    maybe_stamp(rng, sp, ["RECEIVED", "FILED", "COPY"], ["right:46px;bottom:12px"])
    return sp


def html_po(sp):
    f, m = sp["f"], sp["meta"]
    head = (f'<div style="display:flex;justify-content:space-between"><div><div style="font-size:1.5em;font-weight:bold" class="acc">{V(sp, "buyer")}</div>'
            f'<div class="lab" style="margin-top:14px">Vendor</div><div>{V(sp, "vendor")}<br>{E(m["vaddr"])}</div></div>'
            f'<div style="text-align:right"><div style="font-size:1.7em;font-weight:bold">{V(sp, "title")}</div>'
            f'<div>{V(sp, "number")}</div><div class="lab" style="margin-top:8px">Order date</div><div>{V(sp, "date")}</div>'
            f'<div class="lab">Deliver by</div><div>{V(sp, "required")}</div></div></div>'
            f'<div style="margin:14px 0"><span class="lab">Ship to</span> {E(m["ship_to"])} · <span class="lab">Currency</span> {V(sp, "cur")}</div>')
    rows = "".join(f'<tr class="l"><td>{V(sp, f"it_{i}_desc")}</td><td class="n">{V(sp, f"it_{i}_qty")}</td>'
                   f'<td class="n">{V(sp, f"it_{i}_unit")}</td><td class="n">{V(sp, f"it_{i}_amount")}</td></tr>' for i in range(sp["n_items"]))
    items = f'<table><tr><th>Item</th><th class="n">Qty</th><th class="n">Unit price</th><th class="n">Line total</th></tr>{rows}</table>'
    totals = (f'<table style="width:46%;margin:12px 0 0 auto"><tr><td>Shipping</td><td class="n">{V(sp, "ship")}</td></tr>'
              f'<tr style="font-weight:bold"><td>Order total</td><td class="n">{V(sp, "total")}</td></tr></table>')
    sig_val = f'<span class="sig" data-el="f_sig">{E(f["sig"])}</span>' if f["sig"] else '<span data-el="f_sig" style="display:inline-block;width:200px;height:1.9em"></span>'
    if sp.get("hide", {}).get("sig"):
        sig_val = f'<span style="position:relative;display:inline-block">{sig_val}{cover_html(sp["hide"]["sig"])}</span>'
    sign = (f'<div style="margin-top:34px;display:flex;gap:40px;align-items:flex-end"><div><div style="border-bottom:1px solid #333;min-width:260px;min-height:2.6em">{sig_val}</div>'
            f'<div class="lab">Authorised signature</div></div><div><div>{V(sp, "approver")}</div><div class="lab">Approver</div></div></div>')
    return page(sp, sec("head", head) + sec("items", items) + sec("totals", totals) + sec("sign", sign), m["W"])


# ------------------------------------------------------------------------------------------------ bank statement
def make_statement(rng, sid):
    sp = {"id": sid, "doc": "statement", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "statement")
    sp["style"] = st
    W = rng.choice([760, 820, 900, 980])
    cur = rng.choice(["EUR", "GBP", "USD", "CHF", "SEK", "NZD"])
    f = sp["f"]
    f["title"] = rng.choice(["Account statement", "STATEMENT OF ACCOUNT", "Current account statement"])
    f["bank"] = brand(rng) + rng.choice([" Bank", " Savings", " Credit Union"])
    f["holder"] = person(rng)
    y, mth = rng.randint(2024, 2027), rng.randint(1, 12)
    n = rng.randint(4, 9)
    bal = r2(rng.uniform(-150, 3000))
    f["open"] = bal
    merchants = [brand(rng) + " " + rng.choice(["Supermarket", "Energy", "Telecom", "Coffee", "Books", "Garage", "Pharmacy", "Gym"]) for _ in range(n + 3)]
    merchants = list(dict.fromkeys(merchants))
    days = sorted(rng.sample(range(1, 29), n))
    for i in range(n):
        credit = rng.random() < 0.25
        if credit:
            desc = rng.choice([f"Salary {company(rng)}", f"Transfer from {person(rng)}", f"Refund {merchants[i]}"])
            amt = r2(rng.uniform(20, 2600))
            f[f"tx_{i}_credit"] = amt
            bal = r2(bal + amt)
        else:
            desc = rng.choice([f"Card payment {merchants[i]}", f"Direct debit {merchants[i]}", f"{merchants[i]}", "ATM withdrawal"])
            amt = r2(rng.uniform(3, 900))
            f[f"tx_{i}_debit"] = amt
            bal = r2(bal - amt)
        f[f"tx_{i}_desc"] = desc
        f[f"tx_{i}_date"] = dt.date(y, mth, days[i]).isoformat()
        sp["disp"][f"tx_{i}_date"] = f"{days[i]:02d} {MONTHS[mth - 1][:3]}"
        f[f"tx_{i}_bal"] = bal
    close = bal
    if rng.random() < 0.35:
        close = r2(close + rng.choice([10.0, -10.0, 100.0, 1.0, -0.5]))
    f["close"] = close
    f["pages"] = 1              # multi-page snippets only come from the partial_statement unknown variant
    f["cur"] = cur
    sp["disp"]["pages"] = f"Page 1 of {f['pages']}"
    for k in list(f):
        if k.startswith(("tx_", "open", "close")) and not k.endswith(("desc", "date")):
            sp["disp"][k] = f"{f[k]:,.2f}"
    sp["n_tx"] = n
    sp["meta"] = {"W": W, "acct": f"•••• {rng.randint(1000, 9999)}", "period": f"1–{28 if mth == 2 else 30} {MONTHS[mth - 1]} {y}",
                  "absent": [m for m in merchants if not any(m in str(f.get(f"tx_{i}_desc", "")) for i in range(n))][:3]}
    sp["sections"] = {**{k: "head" for k in ("title", "bank", "holder", "open", "cur", "pages")}, **{k: "rows" for k in f if k.startswith("tx_")},
                      "close": "foot"}
    sp["section_order"] = ["head", "rows", "foot"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    return sp


def html_statement(sp):
    f, m = sp["f"], sp["meta"]
    head = (f'<div style="display:flex;justify-content:space-between"><div><div style="font-size:1.5em;font-weight:bold" class="acc">{V(sp, "bank")}</div>'
            f'<div style="font-size:1.2em">{V(sp, "title")}</div></div><div style="text-align:right">{V(sp, "pages")}<br>'
            f'<span class="lab">Account</span> {E(m["acct"])}<br><span class="lab">Period</span> {E(m["period"])}</div></div>'
            f'<div style="margin:14px 0"><span class="lab">Account holder</span> {V(sp, "holder")} · <span class="lab">Currency</span> {V(sp, "cur")}</div>'
            f'<div style="margin-bottom:6px"><b>Opening balance</b> {V(sp, "open")}</div>')
    rows = ""
    for i in range(sp["n_tx"]):
        rows += (f'<tr class="l"><td>{V(sp, f"tx_{i}_date")}</td><td>{V(sp, f"tx_{i}_desc")}</td><td class="n">{V(sp, f"tx_{i}_debit")}</td>'
                 f'<td class="n">{V(sp, f"tx_{i}_credit")}</td><td class="n">{V(sp, f"tx_{i}_bal")}</td></tr>')
    tbl = f'<table><tr><th>Date</th><th>Description</th><th class="n">Paid out</th><th class="n">Paid in</th><th class="n">Balance</th></tr>{rows}</table>'
    foot = f'<div style="margin-top:12px;text-align:right;font-weight:bold">Closing balance {V(sp, "close")}</div>'
    if f["pages"] > 1:
        foot += '<div style="text-align:right;color:#666;font-size:.85em">Continued overleaf</div>'
    return page(sp, sec("head", head) + sec("rows", tbl) + sec("foot", foot), m["W"])


# ------------------------------------------------------------------------------------------------ form
FORM_KINDS = [
    ("Library membership application", [("Membership type", ["Adult", "Student", "Family", "Senior"])]),
    ("Parking permit application", [("Permit length", ["1 month", "3 months", "12 months"])]),
    ("Volunteer registration", [("Availability", ["Weekdays", "Weekends", "Evenings", "Any time"])]),
    ("Sports club enrolment", [("Membership level", ["Junior", "Standard", "Premium"])]),
    ("Community garden plot request", [("Plot size", ["Small", "Medium", "Large"])]),
    ("Course enrolment form", [("Study mode", ["Full-time", "Part-time", "Online"])]),
]


def make_form(rng, sid):
    sp = {"id": sid, "doc": "form", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "form")
    sp["style"] = st
    W = rng.choice([720, 780, 840, 900])
    kind, groups = rng.choice(FORM_KINDS)
    f = sp["f"]
    f["title"] = kind
    f["org"] = town(rng) + rng.choice([" Council", " Library Trust", " Sports Club", " College", " Community Centre"])
    first, last = rng.choice(FIRST), rng.choice(LAST)
    today = rdate(rng, 2025, 2027)
    age = rng.choice([rng.randint(12, 17), rng.randint(19, 80), 18])
    bday = dt.date(today.year - age, rng.randint(1, 12), rng.randint(1, 28))
    fields = [("First name", "fn", first), ("Last name", "ln", last), ("Date of birth", "dob", bday.isoformat()),
              ("Email", "email", f"{first.lower()}.{last.lower()}@mail.test"), ("Phone", "phone", f"0{rng.randint(100000000, 999999999)}"),
              ("Address", "addr", f"{street(rng)}, {town(rng)}"), ("Postcode", "pc", postcode(rng))]
    fields = fields[:3] + rng.sample(fields[3:], rng.randint(2, 4))
    req = [k for _, k, _ in fields if k in ("fn", "ln", "dob") or rng.random() < 0.6]
    blank = None
    mode = rng.random()
    if mode < 0.45:
        blank = rng.choice(req)
    for lab, k, v in fields:
        f[k] = "" if k == blank else v
        if k == "dob" and v:
            sp["disp"][k] = rng.choice([dfmt(bday, "short"), dfmt(bday, "long"), bday.isoformat()]) if k != blank else ""
    gname, gopts = groups[0]
    tick = rng.choice(gopts) if rng.random() < 0.88 else None
    for o in gopts:
        f[f"opt_{o}"] = o == tick
    f["consent"] = rng.random() < 0.7
    f["sig"] = f"{first} {last}" if rng.random() < 0.6 else ""
    f["sig_date"] = (today - dt.timedelta(days=rng.randint(0, 20))).isoformat()
    sp["disp"]["sig_date"] = dfmt(D(f["sig_date"]), "short")
    sp["meta"] = {"W": W, "fields": [[lab, k] for lab, k, _ in fields], "req": req, "group": gname, "gopts": gopts,
                  "today": today.isoformat(), "consent_text": rng.choice(["I confirm the details above are correct.",
                                                                         "I agree to the terms and the privacy notice.",
                                                                         "I consent to being contacted about this application."])}
    sp["sections"] = {**{k: "head" for k in ("title", "org")}, **{k: "fields" for _, k, _ in fields},
                      **{f"opt_{o}": "choice" for o in gopts}, "consent": "sign", "sig": "sign", "sig_date": "sign"}
    sp["section_order"] = ["head", "fields", "choice", "sign"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    maybe_stamp(rng, sp, ["RECEIVED", "FILED"], ["right:40px;top:22px"])
    return sp


def html_form(sp):
    f, m, st = sp["f"], sp["meta"], sp["style"]
    head = (f'<div class="acc" style="font-weight:bold">{V(sp, "org")}</div><div style="font-size:1.55em;font-weight:bold;margin:4px 0 4px">{V(sp, "title")}</div>'
            f'<div class="lab" style="margin-bottom:12px">Fields marked * are required</div>')
    rows = ""
    for lab, k in m["fields"]:
        star = ' <span style="color:#b00">*</span>' if k in m["req"] else ""
        rows += (f'<div style="display:flex;align-items:flex-end;margin:9px 0"><div style="width:170px">{E(lab)}{star}</div>'
                 f'{V(sp, k, cls="box hand")}</div>')
    ch = f'<div style="margin-top:14px"><b>{E(m["group"])}</b> (tick one)<div style="margin-top:6px">'
    for o in m["gopts"]:
        key = f"opt_{o}"
        mark = "✓" if f[key] else ""
        inner = f'<span class="cb hand" data-el="f_{E(key)}" style="color:#1b2f6b">{mark}</span>'
        if sp.get("hide", {}).get(key):
            inner = f'<span style="position:relative;display:inline-block">{inner}{cover_html(sp["hide"][key])}</span>'
        ch += f'<span style="margin-right:22px;white-space:nowrap">{inner}{E(o)}</span>'
    ch += "</div></div>"
    cmark = "✓" if f["consent"] else ""
    cons = f'<span class="cb hand" data-el="f_consent" style="color:#1b2f6b">{cmark}</span>'
    if sp.get("hide", {}).get("consent"):
        cons = f'<span style="position:relative;display:inline-block">{cons}{cover_html(sp["hide"]["consent"])}</span>'
    sig_val = f'<span class="sig" data-el="f_sig">{E(f["sig"])}</span>' if f["sig"] else '<span data-el="f_sig" style="display:inline-block;width:220px;height:1.9em"></span>'
    if sp.get("hide", {}).get("sig"):
        sig_val = f'<span style="position:relative;display:inline-block">{sig_val}{cover_html(sp["hide"]["sig"])}</span>'
    sign = (f'<div style="margin-top:18px">{cons}{E(m["consent_text"])}</div>'
            f'<div style="margin-top:26px;display:flex;gap:40px;align-items:flex-end"><div><div style="border-bottom:1px solid #333;min-width:260px;min-height:2.4em">{sig_val}</div>'
            f'<div class="lab">Signature of applicant</div></div><div><div class="hand">{V(sp, "sig_date")}</div><div class="lab">Date</div></div></div>')
    return page(sp, sec("head", head) + sec("fields", rows) + sec("choice", ch) + sec("sign", sign), m["W"])


# ------------------------------------------------------------------------------------------------ shipping label
def make_label(rng, sid):
    sp = {"id": sid, "doc": "label", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "label")
    st["font"] = rng.choice(["Helvetica, Arial, sans-serif", "Verdana, sans-serif", "'Arial Narrow', Arial, sans-serif", "Futura, sans-serif"])
    sp["style"] = st
    W = rng.choice([560, 620, 680, 740])
    f = sp["f"]
    f["title"] = rng.choice(["SHIPPING LABEL", "PARCEL", "DELIVERY LABEL"])
    f["carrier"] = brand(rng) + rng.choice([" Express", " Parcel", " Post", " Freight"])
    c1 = rng.choice(COUNTRIES)
    c2 = c1 if rng.random() < 0.5 else rng.choice([c for c in COUNTRIES if c != c1])
    cities = [town(rng) for _ in range(4)]
    f["from_name"], f["from_city"], f["from_country"] = company(rng), cities[0], c1
    f["to_name"], f["to_city"], f["to_country"] = person(rng), cities[1], c2
    f["weight"] = round(rng.uniform(0.2, 31.5), 1)
    sp["disp"]["weight"] = f"{f['weight']:.1f} kg"
    svc = rng.choice(["Express", "Standard", "Economy"])
    for o in ("Express", "Standard", "Economy"):
        f[f"svc_{o}"] = o == svc
    f["service"] = svc
    f["tracking"] = f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.randint(100000000, 999999999)}"
    sp["meta"] = {"W": W, "from_addr": f"{street(rng)}, {postcode(rng)}", "to_addr": f"{street(rng)}, {postcode(rng)}", "others": cities[2:],
                  "dims": f"{rng.randint(10, 80)} x {rng.randint(10, 60)} x {rng.randint(5, 50)} cm", "bars": [rng.randint(1, 4) for _ in range(46)],
                  "fragile": rng.random() < 0.3}
    sp["sections"] = {**{k: "top" for k in ("title", "carrier", "from_name", "from_city", "from_country")},
                      **{k: "to" for k in ("to_name", "to_city", "to_country")}, **{k: "svc" for k in f if k.startswith("svc_")},
                      "service": "svc", "weight": "svc", "tracking": "code"}
    sp["section_order"] = ["top", "to", "svc", "code"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    return sp


def html_label(sp):
    f, m = sp["f"], sp["meta"]
    top = (f'<div style="display:flex;justify-content:space-between;border-bottom:3px solid #111;padding-bottom:8px">'
           f'<div style="font-size:1.4em;font-weight:bold">{V(sp, "carrier")}</div><div style="font-weight:bold">{V(sp, "title")}</div></div>'
           f'<div style="margin-top:10px"><div class="lab">From</div>{V(sp, "from_name")}<br>{E(m["from_addr"])}<br>{V(sp, "from_city")}, {V(sp, "from_country")}</div>')
    to = (f'<div style="margin-top:14px;border:2px solid #111;padding:10px 14px"><div class="lab">Deliver to</div>'
          f'<div style="font-size:1.35em;font-weight:bold">{V(sp, "to_name")}</div><div style="font-size:1.2em">{E(m["to_addr"])}<br>'
          f'{V(sp, "to_city")}, {V(sp, "to_country")}</div></div>')
    svc = '<div style="margin-top:14px;display:flex;gap:20px;align-items:center;flex-wrap:wrap">'
    for o in ("Express", "Standard", "Economy"):
        k = f"svc_{o}"
        inner = f'<span class="cb" data-el="f_{k}">{"X" if f[k] else ""}</span>'
        if sp.get("hide", {}).get(k):
            inner = f'<span style="position:relative;display:inline-block">{inner}{cover_html(sp["hide"][k])}</span>'
        svc += f'<span>{inner}{o}</span>'
    svc += (f'<span style="margin-left:auto"><span class="lab">Weight</span> {V(sp, "weight", "font-weight:bold")}</span></div>'
            f'<div style="margin-top:6px"><span class="lab">Dimensions</span> {E(m["dims"])}{" · <b>FRAGILE</b>" if m["fragile"] else ""}</div>')
    bars = "".join(f'<i style="display:inline-block;width:{b}px;height:62px;background:#111;margin-right:{5 - b}px"></i>' for b in m["bars"])
    code = f'<div style="margin-top:14px;text-align:center"><div>{bars}</div><div style="letter-spacing:.2em">{V(sp, "tracking")}</div></div>'
    css = ".page{border:2px solid #222}"
    return page(sp, sec("top", top) + sec("to", to) + sec("svc", svc) + sec("code", code), m["W"], css)


# ------------------------------------------------------------------------------------------------ letter
ASKS = [("return the signed agreement", "Please sign the enclosed agreement and return it to us"),
        ("confirm attendance at the meeting", "Please let us know whether you will attend the meeting"),
        ("pay the outstanding balance", "Please settle the outstanding balance on your account"),
        ("send the missing documents", "We still need a copy of the documents listed below; please send them"),
        ("choose a delivery slot", "Please choose a delivery slot using the reply form"),
        ("renew the membership", "Please renew your membership to keep your benefits"),
        ("update the contact details", "Please check and update the contact details we hold for you"),
        ("book a service appointment", "Your appliance is due for its yearly service; please book an appointment")]


def make_letter(rng, sid):
    sp = {"id": sid, "doc": "letter", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "letter")
    sp["style"] = st
    W = rng.choice([720, 780, 840, 900])
    f = sp["f"]
    f["title"] = "letter"
    f["org"] = company(rng)
    d0 = rdate(rng)
    f["date"] = d0.isoformat()
    ds = rng.choice(["long", "us", "short"])
    sp["disp"]["date"] = dfmt(d0, ds)
    f["recipient"] = person(rng)
    f["signer"] = person(rng)
    while f["signer"] == f["recipient"]:
        f["signer"] = person(rng)
    f["cc"] = person(rng)
    ask, sentence = rng.choice(ASKS)
    f["ask"] = ask
    sp["disp"]["ask"] = sentence
    has_dl = rng.random() < 0.85
    if has_dl:
        f["deadline"] = (d0 + dt.timedelta(days=rng.randint(7, 45))).isoformat()
        sp["disp"]["deadline"] = dfmt(D(f["deadline"]), ds)
    other = d0 + dt.timedelta(days=rng.randint(46, 90))
    f["other_date"] = other.isoformat()
    sp["disp"]["other_date"] = dfmt(other, ds)
    sp["meta"] = {"W": W, "sentence": sentence, "addr": f"{street(rng)}\n{town(rng)} {postcode(rng)}", "ref": f"Our ref: {rng.choice('ABCDEFGHKLMNPRST')}{rng.randint(1000, 9999)}",
                  "role": rng.choice(["Customer Services Manager", "Office Manager", "Membership Secretary", "Accounts Team Lead", "Operations Director"]),
                  "other_what": rng.choice(["Our offices will be closed on", "The new rates apply from", "Your next statement is due on", "The annual meeting takes place on"]),
                  "ds": ds, "asks": [a for a, _ in ASKS],
                  "dl_lead": rng.choice(["Please do this by", "We need your response by", "The deadline for this is", "Kindly reply by"])}
    sp["sections"] = {"org": "head", "date": "head", "title": "head", "recipient": "head", "ask": "body", "deadline": "body",
                      "other_date": "body", "signer": "sign", "cc": "sign"}
    sp["section_order"] = ["head", "body", "sign"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    maybe_stamp(rng, sp, ["RECEIVED", "SCANNED"], ["right:46px;bottom:12px"])
    return sp


def html_letter(sp):
    f, m = sp["f"], sp["meta"]
    head = (f'<div style="border-bottom:2px solid;padding-bottom:8px" class="acc"><span style="font-size:1.5em;font-weight:bold">{V(sp, "org")}</span></div>'
            f'<span data-el="f_title" style="display:none"></span>'
            f'<div style="margin-top:18px">{V(sp, "recipient")}<br>{E(m["addr"]).replace(chr(10), "<br>")}</div>'
            f'<div style="margin-top:14px;text-align:right">{V(sp, "date")}<br><span class="lab">{E(m["ref"])}</span></div>')
    dl = (f'<p>{E(m["dl_lead"])} {V(sp, "deadline")}.</p>' if "deadline" in f else "")
    body = (f'<p style="margin-top:18px">Dear {E(f["recipient"].split()[0])},</p>'
            f'<p>Thank you for being a customer of {E(f["org"])}.</p><p>{V(sp, "ask", "display:inline-block")}.</p>{dl}'
            f'<p>{E(m["other_what"])} {V(sp, "other_date")}. If you have any questions, reply to this letter or call our office.</p>')
    sign = (f'<p style="margin-top:22px">Yours sincerely,</p><div class="sig" style="margin:4px 0">{E(f["signer"])}</div>'
            f'<div>{V(sp, "signer")}<br>{E(m["role"])}</div><div style="margin-top:14px;font-size:.9em">cc: {V(sp, "cc")}</div>')
    return page(sp, sec("head", head) + sec("body", body) + sec("sign", sign), m["W"])


# ------------------------------------------------------------------------------------------------ card
CARD_KINDS = [("Gym membership", ["Basic", "Plus", "Premium"]), ("Library card", ["Standard", "Reader+", "Research"]),
              ("Staff badge", ["Visitor", "Staff", "Manager"]), ("Conference pass", ["Day pass", "Full pass", "Speaker"]),
              ("Club card", ["Bronze", "Silver", "Gold", "Platinum"])]


def make_card(rng, sid):
    sp = {"id": sid, "doc": "card", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "card")
    sp["style"] = st
    W = rng.choice([540, 580, 620, 660])
    kind, tiers = rng.choice(CARD_KINDS)
    f = sp["f"]
    f["title"] = kind
    f["org"] = brand(rng) + rng.choice([" Fitness", " Library", " Group", " Summit", " Club"])
    f["holder"] = person(rng)
    f["number"] = f"{rng.randint(100, 999)} {rng.randint(1000, 9999)} {rng.randint(10, 99)}"
    f["tier"] = rng.choice(tiers)
    v0 = rdate(rng, 2023, 2026)
    f["from"] = v0.isoformat()
    f["until"] = (v0 + dt.timedelta(days=rng.choice([180, 365, 730]))).isoformat()
    ds = rng.choice(["iso", "short", "mmyy"])
    for k in ("from", "until"):
        d = D(f[k])
        sp["disp"][k] = f"{d.month:02d}/{d.year}" if ds == "mmyy" else dfmt(d, ds)
    sp["meta"] = {"W": W, "tiers": tiers, "mmyy": ds == "mmyy", "bg": rng.choice(["#123a5c", "#5c1236", "#0d4d3a", "#3b2a6b", "#222222", "#f2efe6"])}
    sp["sections"] = {k: "card" for k in f}
    sp["section_order"] = ["card"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    return sp


def html_card(sp):
    f, m = sp["f"], sp["meta"]
    dark = m["bg"] != "#f2efe6"
    fg = "#ffffff" if dark else "#1a1a1a"
    sil = ('<svg width="110" height="130" viewBox="0 0 110 130"><rect width="110" height="130" rx="6" fill="#cfd3d8"/>'
           '<circle cx="55" cy="48" r="24" fill="#9aa1a9"/><path d="M15 125 Q55 70 95 125 Z" fill="#9aa1a9"/></svg>')
    body = (f'<div style="display:flex;justify-content:space-between;align-items:center"><div style="font-size:1.3em;font-weight:bold">{V(sp, "org")}</div>'
            f'<div style="font-weight:bold;letter-spacing:.06em">{V(sp, "title")}</div></div>'
            f'<div style="display:flex;gap:22px;margin-top:16px"><div>{sil}</div><div style="flex:1">'
            f'<div class="lab" style="color:{fg};opacity:.75">Name</div><div style="font-size:1.25em;font-weight:bold">{V(sp, "holder")}</div>'
            f'<div class="lab" style="color:{fg};opacity:.75;margin-top:6px">Member no.</div><div>{V(sp, "number")}</div>'
            f'<div style="display:flex;gap:26px;margin-top:6px"><div><div class="lab" style="color:{fg};opacity:.75">Level</div><div style="font-weight:bold">{V(sp, "tier")}</div></div>'
            f'<div><div class="lab" style="color:{fg};opacity:.75">Valid from</div><div>{V(sp, "from")}</div></div>'
            f'<div><div class="lab" style="color:{fg};opacity:.75">Valid until</div><div>{V(sp, "until")}</div></div></div></div></div>'
            f'<div style="margin-top:12px;font-size:.75em;opacity:.8">This card is not an identity document. If found, please return to {E(f["org"])}.</div>')
    css = f".page{{background:{m['bg']};color:{fg};border-radius:18px;padding:24px 28px}}"
    return page(sp, sec("card", body), m["W"], css)


# ------------------------------------------------------------------------------------------------ timetable
def make_timetable(rng, sid):
    sp = {"id": sid, "doc": "timetable", "f": {}, "sections": {}, "hide": {}, "disp": {}}
    st = style(rng, "timetable")
    sp["style"] = st
    nstop, ncol = rng.randint(4, 7), rng.randint(5, 9)
    W = min(1000, 240 + ncol * 78 + rng.randint(0, 80))
    f = sp["f"]
    mode = rng.choice(["Bus", "Tram", "Train", "Ferry", "Coach"])
    f["title"] = f"{mode} {rng.choice(['', 'Line ', 'Route '])}{rng.randint(1, 99)}{rng.choice(['', 'A', 'X'])}".replace("  ", " ")
    stops = []
    while len(stops) < nstop:
        t = town(rng) + rng.choice(["", " Station", " Square", " Hospital", " Park", " Harbour"])
        if t not in stops:
            stops.append(t)
    gaps = [rng.randint(4, 22) for _ in range(nstop - 1)]
    dep = rng.randint(5 * 60, 9 * 60)
    head = rng.choice([15, 20, 30, 45, 60])
    for c in range(ncol):
        if c:
            dep += head + rng.randint(-4, 6)
        t = dep
        express = rng.random() < 0.3
        for r in range(nstop):
            if r > 0:
                t += gaps[r - 1] + rng.randint(0, 3)
            skip = express and 0 < r < nstop - 1 and rng.random() < 0.5
            if skip:
                t -= 2          # an express saves a little time per skipped stop
            f[f"tt_{r}_{c}"] = "—" if skip else f"{(t // 60) % 24:02d}:{t % 60:02d}"
    for r, s in enumerate(stops):
        f[f"stop_{r}"] = s
    sp["n_stops"], sp["n_cols"] = nstop, ncol
    sp["meta"] = {"W": W, "days": rng.choice(["Monday to Friday", "Mondays to Fridays except holidays", "Saturdays", "Daily"]),
                  "valid": f"Valid from {dfmt(rdate(rng), 'long')}", "outside": town(rng) + " Station"}
    sp["sections"] = {k: "table" for k in f}
    sp["section_order"] = ["table"]
    sp["size"] = [W + (52 if st["scan"] else 0), VIEW_H]
    return sp


def html_timetable(sp):
    f, m = sp["f"], sp["meta"]
    head = (f'<div style="font-size:1.6em;font-weight:bold" class="acc">{V(sp, "title")}</div>'
            f'<div>{E(m["days"])} · {E(m["valid"])}</div><div class="lab" style="margin:6px 0 10px">— = does not stop</div>')
    hdr = "<tr><th>Stop</th>" + "".join(f'<th class="n"></th>' for _ in range(sp["n_cols"])) + "</tr>"
    rows = ""
    for r in range(sp["n_stops"]):
        rows += f'<tr class="l"><td style="font-weight:bold;white-space:nowrap">{V(sp, f"stop_{r}")}</td>'
        for c in range(sp["n_cols"]):
            rows += f'<td class="n">{V(sp, f"tt_{r}_{c}")}</td>'
        rows += "</tr>"
    return page(sp, sec("table", head + f"<table>{hdr}{rows}</table>"), m["W"])


DOCS = {"invoice": (0.2, make_invoice, html_invoice), "receipt": (0.13, make_receipt, html_receipt), "po": (0.1, make_po, html_po),
        "statement": (0.11, make_statement, html_statement), "form": (0.13, make_form, html_form), "label": (0.09, make_label, html_label),
        "letter": (0.09, make_letter, html_letter), "card": (0.07, make_card, html_card), "timetable": (0.08, make_timetable, html_timetable)}


def render_html(sp) -> str:
    return DOCS[sp["doc"]][2](sp)


# ------------------------------------------------------------------------------------------------ questions
def case_id(rng, pre):
    return f"{pre}-{rng.randint(10000, 99999)}"


def pick_others(rng, gold, pool, k=3):
    others = [x for x in dict.fromkeys(pool) if x != gold]
    if len(others) < k:
        return None
    return rng.sample(others, k)


def opts_with(rng, gold, others):
    opts = [gold] + others
    rng.shuffle(opts)
    return opts


def today_near(rng, d: dt.date, spread=40):
    t = d + dt.timedelta(days=rng.choice([-1, 1]) * rng.randint(2, spread))
    return t


def tasks_for(rng, sp):
    """Candidates: (task, qtype, question, options|levels|None, params, state)."""
    f, doc, out = sp["f"], sp["doc"], []
    cur = f.get("cur")
    other_docs = [v for k, v in DOC_TYPES.items() if k != doc]
    if rng.random() < 0.07:
        opts = opts_with(rng, DOC_TYPES[doc], rng.sample(other_docs, 3))
        out.append(("doc_type", "choice", rng.choice(["What kind of document is this?", "Which type of document is shown?",
                                                      "How would you file this document?"]), opts, {}, {}))
    if doc in ("invoice", "receipt", "po"):
        n = sp["n_items"]
        name = {"invoice": f"invoice {f.get('number')}", "receipt": f"the {f.get('store')} receipt", "po": f"purchase order {f.get('number')}"}[doc]
        if doc == "invoice":
            parts = [f"it_{i}_amount" for i in range(n)] + ["tax"]
        elif doc == "po":
            parts = [f"it_{i}_amount" for i in range(n)] + ["ship"]
        else:
            parts = [f"it_{i}_amount" for i in range(n)]
        q = rng.choice([f"Do the line amounts{' and tax' if doc == 'invoice' else (' and shipping' if doc == 'po' else '')} on {name} add up to the printed total?"
                        if doc != "receipt" else f"Do the {f.get('store')} receipt lines add up to its total?",
                        f"Is the total on {name} arithmetically correct?", f"Does {name} total up correctly?"])
        out.append(("arith_total", "noul", q, None, {"parts": parts, "total": "total"}, {}))
        labs = [f[f"it_{i}_desc"] for i in range(n)]
        if doc != "receipt":
            rows = [[f[f"it_{i}_desc"], f"it_{i}_qty", f"it_{i}_unit", f"it_{i}_amount"] for i in range(n)]
            q = rng.choice([f"Which line on {name} has an amount that is not quantity times unit price?",
                            f"On {name}, which item's line total is miscalculated?"])
            out.append(("line_calc", "choice", q, labs, {"rows": rows}, {}))
        amts = sorted(f[f"it_{i}_amount"] for i in range(n))
        T = round(amts[len(amts) // 2] * rng.uniform(0.7, 1.3), 0)
        if n >= 3 and all(abs(a - T) >= 0.5 for a in amts):
            levels = ["none" if k == 0 else f"{k} line{'s' if k > 1 else ''}" for k in range(n + 1)]
            out.append(("count_items", "score", rng.choice([f"How many lines on {name} are over {T:,.0f} {cur}?",
                                                            f"Count the lines of {name} costing more than {T:,.0f} {cur}."]),
                        levels, {"rows": [f"it_{i}_amount" for i in range(n)], "T": T}, {}))
        rows = [[f[f"it_{i}_desc"], f"it_{i}_amount"] for i in range(n)]
        q = rng.choice([f"Which item on {name} has the largest line amount?", f"What is the most expensive line on {name}?",
                        f"Which line of {name} costs the most in total?"])
        out.append(("max_item", "choice", q, labs, {"rows": rows}, {}))
        if doc == "po":
            rows = [[f[f"it_{i}_desc"], f"it_{i}_qty"] for i in range(n)]
            out.append(("max_item", "choice", rng.choice([f"Which item on {name} is ordered in the largest quantity?",
                                                          f"On {name}, what is ordered in the greatest number?"]), labs, {"rows": rows}, {}))
    if doc == "invoice":
        name = f"invoice {f['number']}"
        issue = D(f["issue"])
        due = D(f["due"]) if "due" in f else issue + dt.timedelta(days=f["terms"])
        today = today_near(rng, due)
        out.append(("overdue", "noul", rng.choice([f"As of today, is {name} overdue?", f"Is payment of {name} late as of today's date?",
                                                   f"Has the payment deadline for {name} passed unpaid?"]), None,
                    {"due": "due", "issue": "issue", "terms": "terms", "paid": "paid", "today": today.isoformat()},
                    {"today": today.isoformat(), "case": case_id(rng, "AP")}))
        # approval limit, sometimes in another currency (with or without a rate)
        tot = f["total"]
        lim = round(tot * rng.choice([0.5, 0.7, 0.85, 1.15, 1.3, 2.0]), -1) or 10.0
        if abs(lim - tot) / tot < 0.02:
            lim = round(tot * 1.25, -1)
        lcur = cur if rng.random() < 0.8 else rng.choice([c for c in CURRENCIES if c != cur])
        state = {"second_approval_above": f"{lim:,.2f} {lcur}", "case": case_id(rng, "AP")}
        p = {"amount": "total", "cur": "cur", "limit": lim, "limit_cur": lcur}
        if lcur != cur and rng.random() < 0.6:
            rate = round(rng.uniform(0.5, 1.6), 4)
            p["rates"] = {cur: rate}
            state[f"rate_{cur}_to_{lcur}"] = str(rate)
        if not ("rates" in p and abs(tot * p["rates"][cur] - lim) / lim < 0.02):
            out.append(("threshold_amt", "noul", rng.choice([f"Does {name} need a second approval under the policy?",
                                                             f"Under the approval rule given, must {name} go to a second approver?"]), None, p, state))
        due_t = due + dt.timedelta(days=rng.choice([d for d in range(-40, 121) if d not in (-1, 0, 1, 29, 30, 31, 32, 59, 60, 61, 62)]))
        out.append(("late_band", "score", rng.choice([f"How late is payment of {name} as of today?", f"Grade how overdue {name} is today."]),
                    ["paid or not yet due", "1 to 30 days late", "31 to 60 days late", "more than 60 days late"],
                    {"paid": "paid", "today": due_t.isoformat()}, {"today": due_t.isoformat(), "ledger": case_id(rng, "AR")}))
        opts = pick_others(rng, cur, list(CURRENCIES))
        if opts:
            out.append(("currency", "choice", rng.choice([f"In which currency is {name} billed?", f"What currency are the amounts on {name}?"]),
                        opts_with(rng, cur, opts), {"key": "cur"}, {}))
        R = rng.choice([f["rate"], rng.choice([5, 7, 10, 12, 19, 20, 21, 25])])
        out.append(("tax_rate", "noul", rng.choice([f"Is the tax on {name} exactly {R}% of the subtotal?", f"Was tax charged at {R}% on {name}?"]),
                    None, {"sub": "sub", "tax": "tax", "rate": R}, {}))
    if doc == "receipt":
        people = rng.randint(1, 4)
        per = round(f["total"] / people * rng.choice([0.6, 0.8, 1.2, 1.5]), 0) or 5.0
        if abs(per * people - f["total"]) / f["total"] > 0.02:
            out.append(("allowance", "noul", rng.choice(["Is this receipt within the meal allowance for the group?", "Can this receipt be reimbursed in full under the allowance?"]),
                        None, {"total": "total", "per_person": per, "people": people},
                        {"allowance_per_person": f"{per:,.2f} {cur}", "people": people, "claim": case_id(rng, "EXP")}))
        pays = ["card", "cash", "mobile wallet", "voucher"]
        out.append(("payment", "choice", rng.choice(["How was this purchase paid?", "What payment method does the receipt show?"]),
                    pays, {"key": "payment"}, {}))
        if "change" in f:
            out.append(("change_ok", "noul", rng.choice(["Is the change on this receipt correct?", "Did the customer get the right change?"]),
                        None, {"tendered": "tendered", "total": "total", "change": "change"}, {}))
    if doc == "po":
        name = f"purchase order {f['number']}"
        out.append(("signed", "noul", rng.choice([f"Has {name} been signed by the approver?", f"Is there a signature on {name}?"]),
                    None, {"key": "sig"}, {}))
        b = round(f["total"] * rng.choice([0.6, 0.85, 1.2, 1.6]), 0)
        if abs(b - f["total"]) / f["total"] > 0.02:
            out.append(("budget_ok", "noul", rng.choice([f"Can {name} be paid from the remaining budget?", f"Does the remaining budget cover {name}?"]),
                        None, {"total": "total", "budget": b}, {"budget_remaining": f"{b:,.2f} {cur}", "cost_centre": case_id(rng, "CC")}))
        today = D(f["date"]) + dt.timedelta(days=rng.randint(0, 5))
        days = (D(f["required"]) - today).days + rng.choice([-9, -4, -2, 2, 4, 9])
        if days > 0:
            out.append(("lead_time", "noul", rng.choice([f"Can the vendor deliver {name} by its deliver-by date?", f"Will {name} arrive in time if ordered today?"]),
                        None, {"key": "required", "days": days, "today": today.isoformat()},
                        {"today": today.isoformat(), "vendor_lead_time_days": days, "ref": case_id(rng, "PR")}))
    if doc == "statement":
        n = sp["n_tx"]
        debits = [[f[f"tx_{i}_desc"], f"tx_{i}_debit"] for i in range(n) if f"tx_{i}_debit" in f]
        if len(debits) >= 2 and len({d[0] for d in debits}) == len(debits):
            out.append(("largest_debit", "choice", rng.choice(["Which payment out is the largest on this statement?", "What was the biggest debit in the period?"]),
                        [d[0] for d in debits], {"rows": debits}, {}))
        who = P(f["holder"])
        out.append(("balance_check", "noul", rng.choice([f"Does {who} closing balance match the transactions shown?",
                                                         f"Is {who} closing balance consistent with the listed transactions?"]), None,
                    {"open": "open", "close": "close", "credits": [f"tx_{i}_credit" for i in range(n) if f"tx_{i}_credit" in f],
                     "debits": [f"tx_{i}_debit" for i in range(n) if f"tx_{i}_debit" in f]}, {}))
        present = [f[f"tx_{i}_desc"] for i in range(n) if not f[f"tx_{i}_desc"].startswith(("ATM", "Salary", "Transfer"))]
        cands = [x.replace("Card payment ", "").replace("Direct debit ", "").replace("Refund ", "") for x in present]
        if f["pages"] == 1 and sp["meta"]["absent"] and rng.random() < 0.5:
            name = rng.choice(sp["meta"]["absent"])
        elif cands:
            name = rng.choice(cands)
        else:
            name = None
        if name:
            out.append(("merchant", "noul", rng.choice([f"Is there a transaction with {name} on this statement?", f"Did the account holder pay {name} in this period?"]),
                        None, {"rows": [f"tx_{i}_desc" for i in range(n)], "name": name, "pages": "pages"}, {}))
        out.append(("overdrawn", "noul", rng.choice(["Did the balance go below zero at any point in this period?", "Was the account overdrawn at any time shown?"]),
                    None, {"rows": [f"tx_{i}_bal" for i in range(n)]}, {}))
        if len(debits) >= 3:
            vals = sorted(f[k] for _, k in debits)
            T = round(vals[len(vals) // 2] * rng.uniform(0.8, 1.2), 0)
            if all(abs(v - T) > 0.5 for v in vals):
                levels = ["none" if k == 0 else f"{k} payment{'s' if k > 1 else ''}" for k in range(len(debits) + 1)]
                out.append(("count_over", "score", rng.choice([f"How many payments out are larger than {T:,.0f} {cur}?",
                                                               f"Count the debits above {T:,.0f} {cur} on this statement."]), levels,
                            {"rows": [k for _, k in debits], "T": T}, {}))
    if doc == "form":
        m = sp["meta"]
        req = [[lab, k] for lab, k in m["fields"] if k in m["req"]]
        if len(req) >= 2:
            out.append(("blank_field", "choice", rng.choice([f"Which required field on this {f['title'].lower()} was left empty?",
                                                             "Which required box on the form has not been filled in?"]),
                        [x[0] for x in req], {"fields": req}, {}))
        out.append(("signed", "noul", rng.choice(["Has the applicant signed the form?", "Is this form signed?"]), None, {"key": "sig"}, {}))
        group = [[o, f"opt_{o}"] for o in m["gopts"]]
        out.append(("ticked", "choice", rng.choice([f"Which {m['group'].lower()} option is ticked?", f"What {m['group'].lower()} did the applicant choose?"]),
                    m["gopts"], {"group": group}, {}))
        today = m["today"]
        out.append(("adult", "noul", rng.choice(["Was the applicant at least 18 years old on today's date?", "Is the applicant an adult (18 or over) as of today?"]),
                    None, {"key": "dob", "today": today, "age": 18}, {"today": today, "file": case_id(rng, "APP")}))
        out.append(("consent", "noul", rng.choice(["Has the consent box been ticked?", "Did the applicant tick the declaration box?"]), None, {"key": "consent"}, {}))
    if doc == "label":
        out.append(("service", "choice", rng.choice(["Which delivery service is marked on the label?", "What shipping speed was chosen for this parcel?"]),
                    ["Express", "Standard", "Economy"], {"group": [[o, f"svc_{o}"] for o in ("Express", "Standard", "Economy")]}, {}))
        cities = opts_with(rng, f["to_city"], [f["from_city"]] + sp["meta"]["others"])
        out.append(("dest_city", "choice", rng.choice(["Which town is this parcel going to?", "What is the destination town on the label?"]),
                    cities, {"key": "to_city"}, {}))
        out.append(("international", "noul", rng.choice(["Is this an international shipment?", "Is the parcel crossing a national border?"]),
                    None, {"from": "from_country", "to": "to_country"}, {}))
        lim = rng.choice([2, 5, 10, 20, 25, 30])
        if abs(f["weight"] - lim) >= 0.3:
            out.append(("overweight", "noul", rng.choice(["Is the parcel over the weight limit for this service?", "Does this parcel exceed the carrier's weight limit?"]),
                        None, {"key": "weight", "limit": lim}, {"max_weight_kg": lim, "depot": case_id(rng, "DP")}))
    if doc == "letter":
        m = sp["meta"]
        if "deadline" in f:
            dl = f["deadline"]
            others = [f["date"], f["other_date"], (D(dl) + dt.timedelta(days=rng.choice([-7, 7, 14]))).isoformat()]
            disp = {d: dfmt(D(d), m["ds"]) for d in [dl] + others}
            opts = opts_with(rng, disp[dl], [disp[o] for o in others])
            out.append(("deadline", "choice", rng.choice(["By what date does the letter ask for a response?", "What is the deadline given in this letter?"]),
                        opts, {"key": "deadline", "map": disp}, {}))
            today = today_near(rng, D(dl), 20)
            out.append(("deadline_passed", "noul", rng.choice(["Has the deadline in this letter already passed?", "Is it too late to act on this letter as of today?"]),
                        None, {"key": "deadline", "today": today.isoformat()}, {"today": today.isoformat(), "ref": case_id(rng, "L")}))
        out.append(("signer", "choice", rng.choice(["Who signed this letter?", "Who is the sender of this letter?"]),
                    opts_with(rng, f["signer"], [f["recipient"], f["cc"], person(rng)]), {"key": "signer"}, {}))
        asks = opts_with(rng, f["ask"], rng.sample([a for a in m["asks"] if a != f["ask"]], 3))
        out.append(("ask", "choice", rng.choice(["What does the letter ask the reader to do?", "What action is requested in this letter?"]),
                    asks, {"key": "ask"}, {}))
    if doc == "card":
        until = D(f["until"])
        today = today_near(rng, until, 200)
        if sp["meta"]["mmyy"] and today.year == until.year and today.month == until.month:
            today = today + dt.timedelta(days=40)
        out.append(("expired", "noul", rng.choice(["Has this card expired as of today?", "Is the card out of date today?"]), None,
                    {"key": "until", "today": today.isoformat()}, {"today": today.isoformat(), "desk": case_id(rng, "D")}))
        tiers = sp["meta"]["tiers"]
        out.append(("tier", "choice", rng.choice(["What level is shown on this card?", "Which membership level does the holder have?"]),
                    tiers, {"key": "tier"}, {}))
        allowed = rng.sample(tiers, rng.randint(1, len(tiers) - 1))
        out.append(("access", "noul", rng.choice(["Can the holder use the members' lounge today under the rule given?", "Does this card grant lounge access today?"]),
                    None, {"tier": "tier", "until": "until", "allowed": allowed, "today": today.isoformat()},
                    {"today": today.isoformat(), "lounge_levels": ", ".join(allowed)}))
    if doc == "timetable":
        ns, nc = sp["n_stops"], sp["n_cols"]
        a, b = sorted(rng.sample(range(ns), 2))
        by_col = rng.randrange(nc)
        arr = f[f"tt_{b}_{by_col}"]
        stop_a, stop_b = f[f"stop_{a}"], f[f"stop_{b}"]
        deps = [f[f"tt_{a}_{c}"] for c in range(nc) if f[f"tt_{a}_{c}"] != "—"]
        if arr != "—" and len(deps) >= 3:
            by = hm(arr) + rng.randint(0, 4)
            by_s = f"{(by // 60) % 24:02d}:{by % 60:02d}"
            opts = sorted(set(deps), key=hm)
            if len(opts) > 6:
                opts = opts[:6] if hm(opts[5]) > by else opts[-6:]
            out.append(("tt_latest", "choice", rng.choice([f"What is the last departure from {stop_a} that reaches {stop_b} by {by_s}?",
                                                           f"To be at {stop_b} by {by_s}, which is the latest service to catch at {stop_a}?"]),
                        opts, {"frm": a, "to": b, "by": by, "cols": list(range(nc))}, {}))
        c = rng.randrange(nc)
        r = rng.randrange(1, ns)
        dep0 = f[f"tt_0_{c}"]
        out.append(("tt_stops", "noul", rng.choice([f"Does the {dep0} from {f['stop_0']} stop at {f[f'stop_{r}']}?",
                                                    f"Will the {dep0} service from {f['stop_0']} call at {f[f'stop_{r}']}?"]), None, {"row": r, "col": c}, {}))
        if nc <= 9 and deps:
            before = hm(rng.choice(deps)) + rng.choice([-3, 2, 5])
            bs = f"{(before // 60) % 24:02d}:{before % 60:02d}"
            levels = ["none" if k == 0 else f"{k} departure{'s' if k > 1 else ''}" for k in range(nc + 1)]
            out.append(("tt_count", "score", rng.choice([f"How many departures leave {stop_a} before {bs}?", f"Count the services from {stop_a} leaving earlier than {bs}."]),
                        levels, {"row": a, "before": before, "cols": list(range(nc))}, {}))
        c = rng.randrange(nc)
        da, db = f[f"tt_{a}_{c}"], f[f"tt_{b}_{c}"]
        if da != "—" and db != "—":
            dur = hm(db) - hm(da)
            opts = sorted({dur, dur + rng.choice([4, 6, 9]), max(1, dur - rng.choice([3, 5, 8])), dur + rng.choice([12, 15, 20])})
            if len(opts) == 4:
                out.append(("tt_duration", "choice", rng.choice([f"How long does the {da} from {stop_a} take to reach {stop_b}?",
                                                                 f"What is the journey time from {stop_a} to {stop_b} on the {da} service?"]),
                            [f"{x} min" for x in opts], {"frm": a, "to": b, "col": c}, {}))
    return out


FAMILY = {"arith_total": "docimg_arithmetic", "balance_check": "docimg_arithmetic", "change_ok": "docimg_arithmetic",
          "line_calc": "docimg_arithmetic", "tax_rate": "docimg_arithmetic", "count_over": "docimg_arithmetic",
          "max_item": "docimg_extract", "largest_debit": "docimg_extract", "currency": "docimg_extract", "service": "docimg_extract",
          "dest_city": "docimg_extract", "signer": "docimg_extract", "tier": "docimg_extract", "payment": "docimg_extract",
          "deadline": "docimg_extract", "ask": "docimg_extract", "doc_type": "docimg_category",
          "overdue": "docimg_date", "late_band": "docimg_date", "count_items": "docimg_arithmetic", "expired": "docimg_date", "deadline_passed": "docimg_date", "adult": "docimg_date",
          "threshold_amt": "docimg_policy", "allowance": "docimg_policy", "budget_ok": "docimg_policy", "overweight": "docimg_policy",
          "lead_time": "docimg_policy", "access": "docimg_policy",
          "signed": "docimg_form", "consent": "docimg_form", "blank_field": "docimg_form", "ticked": "docimg_form",
          "international": "docimg_extract", "merchant": "docimg_extract", "overdrawn": "docimg_arithmetic",
          "tt_latest": "docimg_timetable", "tt_stops": "docimg_timetable", "tt_count": "docimg_timetable", "tt_duration": "docimg_timetable"}
BASE_DIFF = {"doc_type": 2, "currency": 2, "service": 2, "dest_city": 2, "signer": 2, "tier": 2, "payment": 2, "signed": 2,
             "consent": 2, "ticked": 2, "international": 3, "ask": 3, "deadline": 3, "max_item": 3, "largest_debit": 3,
             "blank_field": 3, "merchant": 3, "overdrawn": 3, "arith_total": 4, "balance_check": 5, "change_ok": 3, "line_calc": 4,
             "tax_rate": 4, "count_over": 4, "overdue": 4, "late_band": 4, "count_items": 4, "expired": 3, "deadline_passed": 3, "adult": 4, "threshold_amt": 4,
             "allowance": 4, "budget_ok": 3, "overweight": 3, "lead_time": 5, "access": 4, "tt_latest": 5, "tt_stops": 3,
             "tt_count": 4, "tt_duration": 4}


def difficulty(sp, task, unknown):
    d = BASE_DIFF[task]
    if sp["style"]["scan"]:
        d += 1
    if task == "threshold_amt" and False:
        d += 1
    if unknown:
        d += 1
    return max(2, min(5, d))


def make_item(sp, task, qtype, q, opts, p, state, parent=None, variant=None):
    gold, unk, used = decide(sp, task, p)
    if gold == "__ill__":
        return None
    if gold is None and variant is None:
        return None
    field = {"type": qtype, "question": q}
    g = gold
    if qtype == "choice":
        if len({o.lower() for o in opts}) != len(opts):
            return None
        if gold is not None:
            gtxt = p["map"][gold] if p.get("map") else gold
            if gtxt not in opts:
                return None
        taken = set()
        keyed = [{"key": option_key(o, taken), "text": o} for o in opts]
        field["options"] = keyed
        if gold is not None:
            g = {o["text"]: o["key"] for o in keyed}[p["map"][gold] if p.get("map") else gold]
    elif qtype == "score":
        field["levels"] = [{"value": i, "description": d} for i, d in enumerate(opts)]
        if gold is not None and not 0 <= gold < len(opts):
            return None
    return {"spec_id": sp["id"], "task": task, "field": field, "gold": g, "unknown_reason": unk, "params": p, "state": state,
            "used": used, "parent": parent, "variant": variant, "difficulty": difficulty(sp, task, gold is None)}


# a generic answer key for choice tasks that read a value ("service" reads a checkbox group)
def _fix_service(p):
    return p


ABSENT_OK = {"invoice": {"due", "terms"}, "letter": {"deadline"}}
CROP_SECTIONS = {"invoice": ["totals"], "receipt": ["totals"], "po": ["totals", "sign"], "statement": ["foot"],
                 "form": ["sign"], "letter": ["sign"]}


def row_keys(sp, i):
    return [k for k in (f"it_{i}_desc", f"it_{i}_qty", f"it_{i}_unit", f"it_{i}_amount") if k in sp["f"]]


def cover_groups(sp, task, p, it):
    """Groups of keys whose cover makes the answer unknowable. Evidence printed twice (currency codes on every amount,
    typed + signed names, running balances that restate a transaction, qty x unit that restates a line amount) is never
    the only thing covered."""
    f, doc = sp["f"], sp["doc"]
    plain_cur = sp.get("meta", {}).get("mstyle", "plain") == "plain" and doc in ("invoice", "receipt", "statement")
    if doc == "receipt":
        plain_cur = True
    g = []
    if task == "arith_total":
        g = [["total"]] + ([["tax"]] if doc == "invoice" else []) + ([["ship"]] if doc == "po" else []) + \
            [row_keys(sp, i) for i in range(sp.get("n_items", 0))]
    elif task == "max_item":
        g = [row_keys(sp, i) for i in range(sp.get("n_items", 0))]
    elif task == "balance_check":
        g = [["open"], ["close"]]
    elif task == "change_ok":
        g = [["tendered"], ["change"], ["total"]]
    elif task in ("overdue", "late_band"):
        g = [["due"]] if "due" in f else [["terms"], ["issue"]]
    elif task == "count_items":
        g = [row_keys(sp, i) for i in range(sp.get("n_items", 0))]
    elif task == "threshold_amt":
        g = [["total"]] + ([["cur"]] if plain_cur else [])
    elif task in ("allowance", "budget_ok"):
        g = [["total"]]
    elif task == "lead_time":
        g = [["required"]]
    elif task == "currency":
        g = [["cur"]] if plain_cur else []
    elif task in ("signed", "consent"):
        g = [[p["key"]]]
    elif task in ("ticked", "service"):
        g = [[k] for _, k in p["group"] if f.get(k)]
    elif task == "adult":
        g = [["dob"]]
    elif task in ("expired",):
        g = [["until"]]
    elif task == "access":
        g = [["until"], ["tier"]]
    elif task == "tier":
        g = [["tier"]]
    elif task in ("deadline", "deadline_passed"):
        g = [["deadline"]]
    elif task == "ask":
        g = [["ask"]]
    elif task == "dest_city":
        g = [["to_city"]]
    elif task == "international":
        g = [["to_country"], ["from_country"]]
    elif task == "overweight":
        g = [["weight"]]
    elif task == "merchant":
        g = [[k] for k in p["rows"]] + [["pages"]]
    elif task == "tt_stops":
        g = [[f"tt_{p['row']}_{p['col']}"]]
    elif task == "tt_duration":
        g = [[f"tt_{p['frm']}_{p['col']}"], [f"tt_{p['to']}_{p['col']}"]]
    elif task == "tt_latest" and it["gold"] is not None:
        c = next(c for c in p["cols"] if f[f"tt_{p['frm']}_{c}"] == gold_text(it))
        g = [[f"tt_{p['to']}_{c}"]]
    return [x for x in g if x and all(k in f for k in x)]


def gold_text(it):
    if it["field"]["type"] != "choice" or it["gold"] is None:
        return it["gold"]
    return {o["key"]: o["text"] for o in it["field"]["options"]}[it["gold"]]


def unknown_variant(rng, sp, it):
    """(new spec or None, params, state, question, how) whose answer needs something the image lacks."""
    task, p, used = it["task"], it["params"], [k for k in it["used"] if k != "title"]
    ways = ["crop", "absent", "rate", "none", "pages"]
    rng.shuffle(ways)
    ways = ways + ["cover"] if rng.random() < 0.6 else ["cover"] + ways
    for how in ways:
        if how == "cover":
            groups = cover_groups(sp, task, p, it)
            if not groups:
                continue
            keys = rng.choice(groups)
            sp2 = copy.deepcopy(sp)
            look = rng.choice(["sticky", "ink", "tape", "stain"])
            sp2["hide"] = dict(sp2.get("hide", {}), **{k: look for k in keys})
            return sp2, p, it["state"], it["field"]["question"], "covered"
        if how == "crop" and sp["doc"] in CROP_SECTIONS and not sp["style"]["rot"]:
            secs = {sp["sections"].get(k) for k in used}
            cands = [s for s in CROP_SECTIONS[sp["doc"]] if s in secs]
            if not cands:
                continue
            sp2 = copy.deepcopy(sp)
            sp2["crop_at"] = cands[0]
            sp2.pop("stamp", None)
            return sp2, p, it["state"], it["field"]["question"], "cropped"
        if how == "absent" and sp["doc"] in ABSENT_OK:
            ks = [k for k in used if k in ABSENT_OK[sp["doc"]]]
            if not ks:
                continue
            sp2 = copy.deepcopy(sp)
            for k in ABSENT_OK[sp["doc"]]:
                sp2["f"].pop(k, None)
            return sp2, p, it["state"], it["field"]["question"], "not_printed"
        if how == "rate" and task == "threshold_amt" and p["limit_cur"] == sp["f"]["cur"]:
            lcur = rng.choice([c for c in CURRENCIES if c != sp["f"]["cur"]])
            p2 = dict(p, limit_cur=lcur)
            p2.pop("rates", None)
            st2 = {k: v for k, v in it["state"].items() if not k.startswith("rate_")}
            st2["second_approval_above"] = f"{p['limit']:,.2f} {lcur}"
            return None, p2, st2, it["field"]["question"], "no_exchange_rate"
        if how == "none" and task in ("blank_field", "line_calc", "ticked", "service"):
            sp2 = copy.deepcopy(sp)
            if task == "blank_field":
                for lab, k in p["fields"]:
                    if sp2["f"][k] == "":
                        sp2["f"][k] = {"fn": rng.choice(FIRST), "ln": rng.choice(LAST), "dob": "1990-05-17", "email": "reader@mail.test",
                                       "phone": "0612345678", "addr": street(rng), "pc": postcode(rng)}.get(k, "x")
            elif task == "line_calc":
                n = sp2["n_items"]
                for i in range(n):
                    a = r2(sp2["f"][f"it_{i}_qty"] * sp2["f"][f"it_{i}_unit"])
                    sp2["f"][f"it_{i}_amount"] = a
                    sp2["disp"][f"it_{i}_amount"] = sp["disp"][f"it_{i}_amount"].replace(
                        f"{sp['f'][f'it_{i}_amount']:,.2f}", f"{a:,.2f}")
            else:
                for _, k in p["group"]:
                    sp2["f"][k] = False
            return sp2, p, it["state"], it["field"]["question"], "none_applies"
        if how == "pages" and sp["doc"] == "statement" and it["gold"] is not None:
            f = sp["f"]
            if task == "merchant" and (any(p["name"] in str(f.get(k, "")) for k in p["rows"]) or not sp["meta"]["absent"]):
                continue
            if task == "overdrawn" and it["gold"] is True:
                continue
            sp2 = copy.deepcopy(sp)
            sp2["f"]["pages"] = 2
            sp2["disp"]["pages"] = "Page 1 of 2"
            return sp2, p, it["state"], it["field"]["question"], "partial_statement"
    return None


# ------------------------------------------------------------------------------------------------ plan / render / build
def make_spec(rng, sid, doc=None):
    doc = doc or rng.choices(list(DOCS), weights=[v[0] for v in DOCS.values()])[0]
    sp = DOCS[doc][1](rng, sid)
    if rng.random() < 0.06:          # an irrelevant cover (decide() keeps the items it does not affect)
        keys = [k for k in sp["f"] if k not in ("title", "paid", "cur", "signer", "service") and not k.startswith(("svc_", "opt_"))]
        sp["hide"][rng.choice(keys)] = rng.choice(["sticky", "ink", "tape"])
    return sp


def plan(seed: str, count: int, unknown_share: float = UNKNOWN_SHARE, prefix: str = "d"):
    specs, items = {}, []
    n_base = int(round(count * (1 - unknown_share)))
    task_ct, gold_ct = Counter(), Counter()
    i = 0
    while len(items) < n_base * 1.15 and i < count * 3:
        rng = rng_for("p3-docimg", seed, "spec", i)
        sid = f"{prefix}{seed}-{i:05d}"
        i += 1
        try:
            sp = make_spec(rng, sid)
        except Skip:
            continue
        qrng = rng_for("p3-docimg", seed, "q", sid)
        cands = tasks_for(qrng, sp)
        qrng.shuffle(cands)
        kept = 0
        for task, qtype, q, opts, p, state in cands:
            if kept >= 3:
                break
            if task_ct[task] > 1.8 * (sum(task_ct.values()) / max(1, len(task_ct))) + 30:
                continue
            it = make_item(sp, task, qtype, q, opts, p, state)
            if it is None:
                continue
            if qtype == "noul":
                if gold_ct[it["gold"]] > gold_ct[not it["gold"]] * 1.08 + 10:
                    continue
                gold_ct[it["gold"]] += 1
            kept += 1
            task_ct[task] += 1
            items.append(it)
        if kept:
            specs[sid] = sp
    urng = rng_for("p3-docimg", seed, "unknowns")
    base = [x for x in items if x["variant"] is None]
    urng.shuffle(base)
    target = int(round(len(base) * unknown_share / (1 - unknown_share)))
    made = 0
    for k, it in enumerate(base):
        if made >= target:
            break
        sp = specs[it["spec_id"]]
        v = unknown_variant(rng_for("p3-docimg", seed, "unk", it["spec_id"], k), sp, it)
        if v is None:
            continue
        sp2, p2, st2, q2, how = v
        if sp2 is not None:
            sp2["id"] = f"{sp['id']}-u{k}"
        opts = [o["text"] for o in it["field"].get("options", [])] or [l["description"] for l in it["field"].get("levels", [])] or None
        u = make_item(sp2 or sp, it["task"], it["field"]["type"], q2, opts, p2, st2, parent=it, variant=how)
        if u is None or u["gold"] is not None:
            continue
        if sp2 is not None:
            specs[sp2["id"]] = sp2
        items.append(u)
        made += 1
    return specs, items


def post_for(sp) -> dict:
    st = sp["style"]
    r = rng_for("p3-docimg-post", sp["id"])
    if st["scan"]:
        return {"noise": st["noise"], "noise_seed": r.randrange(2 ** 31), "blur": st["blur"], "contrast": st["contrast"], "format": "webp"}
    return {"format": "png"}


def check_render(sp, els, used_by_item):
    """(ok, crop_h, reason): every printed field in view, hidden ones not; crop height for the page."""
    pg = els.get("page")
    if not pg:
        return False, None, "no page"
    bottom = pg["y"] + pg["h"] + (26 if sp["style"]["scan"] else 14)
    if bottom > VIEW_H - 4:
        return False, None, "page too tall"
    crop_h = int(bottom)
    hidden = hidden_keys(sp)
    if sp.get("crop_at"):
        s = els.get(f"sec_{sp['crop_at']}")
        if not s:
            return False, None, "no section"
        crop_h = int(s["y"] - 6)
    for k, e in els.items():
        if not k.startswith("f_"):
            continue
        key = k[2:]
        if key == "title" and sp["doc"] == "letter":
            continue
        if key in hidden:
            if sp.get("crop_at") and sp["sections"].get(key) in sp["section_order"][sp["section_order"].index(sp["crop_at"]):]:
                if e["y"] < crop_h:
                    return False, None, f"cropped key above cut {key}"
            elif e["visible"]:
                return False, None, f"cover leaks {key}"
            continue
        if not e["visible"] and e["w"] > 2:
            return False, None, f"not visible {key}"
        if e["y"] + e["h"] > crop_h:
            return False, None, f"below crop {key}"
    for key in used_by_item:
        if key == "title" and sp["doc"] == "letter":
            continue
        e = els.get(f"f_{key}")
        if e is None or (not e["visible"] and sp["f"].get(key) not in ("", False)):
            if sp["f"].get(key) in ("", False) and e is not None:
                continue          # a blank box / empty signature line: its container is what the reader sees
            return False, None, f"used not visible {key}"
    return True, crop_h, ""


def build(seed: str, count: int, work: Path, img_dir: Path = IMG_DIR, id_prefix: str = "p3-I-docimg", unknown_share=UNKNOWN_SHARE):
    specs, items = plan(seed, count, unknown_share)
    used = {}
    for it in items:
        used.setdefault(it["spec_id"], set()).update(it["used"])
    pages = [{"id": sid, "html": render_html(sp), "width": sp["size"][0], "height": VIEW_H} for sid, sp in specs.items()]
    html_sha = {p["id"]: hashlib.sha256(p["html"].encode()).hexdigest() for p in pages}
    res = render_pages(pages, work)
    bench = bench_image_shas()
    rel_base = rel_dir(img_dir)
    img_of, stats = {}, Counter()
    for sid, sp in specs.items():
        r = res.get(sid)
        if not r or not r["ok"]:
            stats["render_failed"] += 1
            continue
        els = {e["el"]: e for e in r["elements"]}
        ok, crop_h, why = check_render(sp, els, used.get(sid, ()))
        if not ok:
            stats["rejected:" + why.split(" ")[0] + "_" + (why.split(" ")[1] if len(why.split(" ")) > 1 else "")] += 1
            continue
        post = dict(post_for(sp), crop_h=crop_h)
        got = finish_image(work / f"{sid}.png", post, img_dir, rel_base, bench)
        if got is None:
            stats["bench_sha"] += 1
            continue
        img_of[sid] = (got[0], got[1], post)
    rows, id_of = [], {}
    for it in items:
        if it["spec_id"] not in img_of or (it["parent"] is not None and id(it["parent"]) not in id_of):
            continue
        sp = specs[it["spec_id"]]
        rel, sha, post = img_of[it["spec_id"]]
        rid = f"{id_prefix}-{seed}-{len(rows):06d}"
        id_of[id(it)] = rid
        rows.append(make_row(rid, it, sp, rel, sha, post, html_sha[it["spec_id"]], seed,
                             id_of[id(it["parent"])] if it["parent"] is not None else None))
    from gen_charts import trim
    return trim(rows, count, unknown_share), stats


def make_row(rid, it, sp, rel, sha, post, hsha, seed, parent_id):
    return {"id": rid, "source": "I", "dataset": "docimg_synthetic", "family": FAMILY[it["task"]], "difficulty": it["difficulty"],
            "state": it["state"], "images": [rel], "field": it["field"], "gold": it["gold"], "unknown_reason": it["unknown_reason"],
            "gold_kind": "constructed", "parent_id": parent_id,
            "provenance": {"licence": "generated", "generator": GENERATOR, "seed": seed,
                           "renderer": "HTML -> puppeteer-core + headless Chromium (scripts/p3/gui_templates/render.mjs) + PIL",
                           "task": it["task"], "doc_type": sp["doc"], "spec": sp, "params": it["params"], "fields_read": it["used"],
                           "html_sha256": hsha, "post": post, "image_sha256": [sha], "viewport": [sp["size"][0], VIEW_H],
                           "unknown_construction": it["variant"], "upstream_split": "generated",
                           "content": "invented organisations, people, addresses and numbers; no real identity documents"}}


def variants_of(path: str, per: int, work: Path, img_dir: Path = IMG_DIR) -> list[dict]:
    """Stage-2 variants: fresh documents of the same type and task, same difficulty, rng keyed by the parent id."""
    from candidate import read
    specs, planned = {}, []
    for par in read(path):
        if par.get("dataset") != "docimg_synthetic":
            continue
        task, doc = par["provenance"]["task"], par["provenance"]["doc_type"]
        want_unk = par["gold"] is None
        made = 0
        for k in range(200):
            if made >= per:
                break
            rng = rng_for("p3-docimg-variant", par["id"], k)
            sp = make_spec(rng, f"v{hashlib.sha1(par['id'].encode()).hexdigest()[:10]}-{k}", doc)
            for t, qtype, q, opts, p, state in [c for c in tasks_for(rng, sp) if c[0] == task]:
                it = make_item(sp, t, qtype, q, opts, p, state)
                if it is None:
                    continue
                if want_unk:
                    v = unknown_variant(rng, sp, it)
                    if v is None:
                        continue
                    sp2, p2, st2, q2, how = v
                    if sp2 is not None:
                        sp2["id"] = sp["id"] + "u"
                    optl = [o["text"] for o in it["field"].get("options", [])] or [l["description"] for l in it["field"].get("levels", [])] or None
                    it = make_item(sp2 or sp, t, qtype, q2, optl, p2, st2, variant=how)
                    if it is None or it["gold"] is not None:
                        continue
                    sp = sp2 or sp
                it.update({"difficulty": par["difficulty"], "vparent": par["id"], "vk": made + 1})
                specs[sp["id"]] = sp
                planned.append(it)
                made += 1
                break
    pages = [{"id": sid, "html": render_html(sp), "width": sp["size"][0], "height": VIEW_H} for sid, sp in specs.items()]
    res = render_pages(pages, work) if pages else {}
    bench, rel_base = bench_image_shas(), rel_dir(img_dir)
    rows = []
    for it in planned:
        sp = specs[it["spec_id"]]
        r = res.get(sp["id"])
        if not r or not r["ok"]:
            continue
        ok, crop_h, _ = check_render(sp, {e["el"]: e for e in r["elements"]}, it["used"])
        if not ok:
            continue
        post = dict(post_for(sp), crop_h=crop_h)
        got = finish_image(work / f"{sp['id']}.png", post, img_dir, rel_base, bench)
        if got is None:
            continue
        row = make_row(f"{it['vparent']}-v{it['vk']}", it, sp, got[0], got[1], post, hashlib.sha256(render_html(sp).encode()).hexdigest(),
                       "variant", it["vparent"])
        row["provenance"]["variant_of"] = it["vparent"]
        rows.append(row)
    return rows


def summary(rows):
    return {"rows": len(rows), "unknown": sum(r["gold"] is None for r in rows),
            "by_family": dict(Counter(r["family"] for r in rows)), "by_task": dict(Counter(r["provenance"]["task"] for r in rows)),
            "by_doc": dict(Counter(r["provenance"]["doc_type"] for r in rows)), "by_type": dict(Counter(r["field"]["type"] for r in rows)),
            "by_difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
            "unknown_by_construction": dict(Counter(r["provenance"]["unknown_construction"] for r in rows if r["gold"] is None)),
            "images": len({r["images"][0] for r in rows})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", default="d1")
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--img-dir", default=str(IMG_DIR))
    ap.add_argument("--id-prefix", default="p3-I-docimg")
    ap.add_argument("--heldout-tag", default=None)
    ap.add_argument("--variant-of", default=None)
    ap.add_argument("--variants-per", type=int, default=2)
    ap.add_argument("--keep-work", default=None)
    a = ap.parse_args()
    work = Path(a.keep_work) if a.keep_work else Path(tempfile.mkdtemp(prefix="p3docimg-"))
    work.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.img_dir).resolve()
    if a.variant_of:
        rows, stats = variants_of(a.variant_of, a.variants_per, work, img_dir), {}
    else:
        rows, stats = build(a.seed, a.count, work, img_dir, a.id_prefix)
    if a.heldout_tag:
        for r in rows:
            r["provenance"].update({"heldout": "fresh", "heldout_tag": a.heldout_tag, "heldout_generator": f"gen_docimages.py --seed {a.seed}"})
    n = write(a.out, rows)
    print(json.dumps({"wrote": n, "out": a.out, "stats": dict(stats), **summary(rows)}, indent=1))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
