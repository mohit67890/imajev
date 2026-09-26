"""Phase-3 Stage 0-I: synthetic GUI screenshots (family `gui_action`, source I).

We render web/app screens ourselves (HTML -> PNG with a local headless Chromium through puppeteer-core, the same tool
site/readme_shots.mjs uses), with invented brands, people and content, and ask questions whose gold the generator knows:

- `form_invalid_field`  (choice) which field breaks the rule printed under it; unknown variant: none does (false_premise)
- `form_ready`          (noul)   is the form ready to submit (all fields valid, required ones filled, consent ticked)
- `toolbar_goal`        (choice) which control achieves goal G; unknown variant: no control on screen does (not_listed)
- `toolbar_available`   (noul)   can you do G from this screen right now (the control may be disabled)
- `settings_which`      (choice) which setting controls G; unknown variant: that setting is not on screen (not_listed)
- `settings_state`      (noul)   will G happen with the switches as shown (map G to the setting, read its switch)
- `dialog_button`       (choice) which dialog button does G; unknown variant: G is not something the dialog offers
- `cart_free_shipping`  (noul)   does the basket reach the free-shipping threshold printed on the page (sum it)
- `cart_over_limit`     (choice) which item is over its per-customer limit; unknown variant: none is (false_premise)

Options are the element labels exactly as rendered; the renderer reports every labelled element's box and an item is
kept only when every option's element is fully inside the screenshot and on top (not covered by an overlay).
PNGs go to data/p3/images/gui/<sha256>.png (<= 1280 px wide). Licence: "generated" (our own renders, system fonts).

    .venv/bin/python scripts/p3/gen_gui.py [--screens 2600] [--out data/p3/candidates/I-gui.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_image_joint import bench_image_shas, rng_for  # noqa: E402

SEED = "p3-gui-v1"
IMG_DIR = ROOT / "data" / "p3" / "images" / "gui"
OUT = ROOT / "data" / "p3" / "candidates" / "I-gui.jsonl"
RENDER = ROOT / "scripts" / "p3" / "gui_templates" / "render.mjs"
E = html.escape

# ------------------------------------------------------------------------------------------------ invented content
BRAND_A = ["Vel", "Quor", "Tami", "Brin", "Oda", "Zell", "Poma", "Kiro", "Lume", "Sorb", "Yent", "Fenn", "Marro", "Tovi",
           "Wexa", "Ulma", "Rask", "Pello", "Nibu", "Cora", "Dessa", "Gault", "Hesk", "Ivor", "Jemba", "Olvi"]
BRAND_B = ["wick", "nest", "loop", "pad", "deck", "ora", "ix", "hub", "field", "mint", "works", "base", "port", "lane",
           "grove", "stack", "tide", "yard"]
FIRST = ["Anna", "Tomas", "Leila", "Marek", "Priya", "Jonah", "Sofia", "Kwame", "Ines", "Ravi", "Helga", "Omar",
         "Yuki", "Bruno", "Maeve", "Tariq", "Lena", "Diego", "Nadia", "Felix", "Chiara", "Emeka", "Ruth", "Sven"]
LAST = ["Kerr", "Novak", "Haddad", "Lindqvist", "Okafor", "Brandt", "Moreau", "Sato", "Ferreira", "Iqbal", "Duarte",
        "Keane", "Varga", "Mbeki", "Olsen", "Rossi", "Tanaka", "Wolfe", "Petrov", "Achterberg"]
PRODUCTS = ["Linen tea towel", "Ceramic mug", "Desk lamp", "Notebook, dotted", "Oak cutting board", "Wool socks",
            "Glass carafe", "Canvas tote", "Phone stand", "Plant pot, small", "Beeswax candle", "Travel mug",
            "Cotton napkin set", "Steel water bottle", "Bamboo brush", "Cork coaster set", "Picture frame A4",
            "Enamel pin", "Hand cream 50 ml", "Pocket umbrella"]

THEMES = [  # bg, surface, text, muted, primary, on_primary, border, danger, font
    ("#f6f7f9", "#ffffff", "#1d2330", "#6b7383", "#2f6fed", "#ffffff", "#d9dde5", "#c62828", "-apple-system, Helvetica, Arial, sans-serif"),
    ("#fbf8f3", "#ffffff", "#2b2622", "#7a7068", "#b4532a", "#ffffff", "#e6ddd2", "#b00020", "Georgia, 'Times New Roman', serif"),
    ("#eef3f1", "#ffffff", "#15302a", "#5b726c", "#12805c", "#ffffff", "#cfdcd7", "#c0392b", "Verdana, Geneva, sans-serif"),
    ("#121417", "#1c1f24", "#e8eaed", "#9aa0a6", "#7aa2ff", "#0b1020", "#2e333a", "#ff6b6b", "-apple-system, Helvetica, Arial, sans-serif"),
    ("#f3f0fa", "#ffffff", "#221a36", "#6e6485", "#6b3fd4", "#ffffff", "#ddd5ee", "#c2185b", "'Trebuchet MS', Helvetica, sans-serif"),
    ("#0f1a1f", "#16252c", "#e3eef2", "#8fa7b0", "#3cc4b4", "#06201c", "#27404a", "#ff8a80", "Menlo, Monaco, monospace"),
    ("#ffffff", "#f7f7f7", "#111111", "#666666", "#111111", "#ffffff", "#e2e2e2", "#d32f2f", "'Avenir Next', Avenir, Helvetica, sans-serif"),
]
SIZES = [(1280, 800), (1280, 720), (1180, 820), (1024, 768), (1100, 760), (1280, 900)]


def brand(rng) -> str:
    return rng.choice(BRAND_A) + rng.choice(BRAND_B)


def person(rng) -> tuple[str, str]:
    return rng.choice(FIRST), rng.choice(LAST)


def page(theme, title: str, body: str, extra_css: str = "") -> str:
    bg, surf, text, muted, prim, onp, border, danger, font = theme
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{E(title)}</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:{bg};color:{text};font-family:{font};font-size:15px}}
.top{{display:flex;align-items:center;gap:18px;padding:12px 24px;background:{surf};border-bottom:1px solid {border}}}
.logo{{font-weight:700;font-size:19px;color:{prim}}} .muted{{color:{muted};font-size:13px}}
.card{{background:{surf};border:1px solid {border};border-radius:10px;padding:22px}}
.btn{{font:inherit;padding:8px 14px;border-radius:7px;border:1px solid {border};background:{surf};color:{text};cursor:pointer}}
.btn.primary{{background:{prim};border-color:{prim};color:{onp}}} .btn:disabled{{opacity:.38;cursor:not-allowed}}
.btn.danger{{color:{danger};border-color:{danger}}}
label.f{{display:block;font-weight:600;margin:14px 0 5px}} input.i{{font:inherit;width:100%;padding:9px 11px;border:1px solid {border};border-radius:7px;background:{bg};color:{text}}}
.help{{color:{muted};font-size:12.5px;margin-top:4px}} table{{border-collapse:collapse;width:100%}}
td,th{{text-align:left;padding:9px 10px;border-bottom:1px solid {border}}} th{{color:{muted};font-weight:600;font-size:13px}}
.sw{{width:44px;height:24px;border-radius:12px;background:{border};position:relative;flex:none}}
.sw.on{{background:{prim}}} .sw i{{position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#fff;box-shadow:0 1px 2px #0004}}
.sw.on i{{left:23px}} .req{{color:{danger}}} .nav a{{color:{muted};text-decoration:none;margin-right:16px}} .nav a.cur{{color:{text};font-weight:600}}
{extra_css}</style></head><body>{body}</body></html>"""


def topbar(rng, theme, name, nav=None):
    nav = nav or rng.sample(["Home", "Projects", "Reports", "Inbox", "Team", "Billing", "Help"], 4)
    links = "".join(f'<a class="{"cur" if i == 0 else ""}">{E(n)}</a>' for i, n in enumerate(nav))
    f, l = person(rng)
    return (f'<div class="top"><span class="logo">{E(name)}</span><span class="nav">{links}</span>'
            f'<span style="margin-left:auto" class="muted">{E(f)} {E(l[0])}.</span></div>')


# ------------------------------------------------------------------------------------------------ forms
def _digits(rng, n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def field_bank(rng, first, last, today):
    """Each: (key, label, help text, valid value, invalid value). The rule a value breaks is always printed."""
    d, m, y = today
    user = (first + last[:3]).lower() + _digits(rng, 2)
    email = f"{first.lower()}.{last.lower()}@mailbox.test"
    fb = [
        ("fullname", "Full name", "", f"{first} {last}", ""),
        ("email", "Email", "Format: name@domain", email, rng.choice([email.replace("@", ""), email.replace("@mailbox.test", "@"), email.replace(".", " ", 1)])),
        ("username", "Username", "4-16 letters or digits, no spaces", user, rng.choice([user[:2], f"{first} {last}".lower(), user + "x" * 12])),
        ("phone", "Phone", "Digits only, exactly 10", _digits(rng, 10), rng.choice([_digits(rng, 9), _digits(rng, 6) + "-" + _digits(rng, 4), _digits(rng, 11)])),
        ("postcode", "Postcode", "5 digits", _digits(rng, 5), rng.choice([_digits(rng, 4), "AB" + _digits(rng, 3), _digits(rng, 6)])),
        ("dob", "Date of birth", "DD/MM/YYYY", f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2004)}",
         rng.choice([f"{rng.randint(1950, 2004)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(50, 99)}"])),
        ("start", "Start date", f"Must be after today ({d:02d}/{m:02d}/{y})", f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{y + 1}",
         f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{y - 1}"),
        ("qty", "Number of guests", "Between 1 and 8", str(rng.randint(1, 8)), str(rng.choice([0, 9, 10, 12, 20]))),
        ("amount", "Amount", "Up to 500.00", f"{rng.uniform(5, 499):.2f}", f"{rng.uniform(501, 900):.2f}"),
        ("code", "Referral code", "8 characters: capital letters and digits", "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)),
         rng.choice(["".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8)), "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(6))])),
        ("expiry", "Card expiry", f"MM/YY, not expired (today {m:02d}/{str(y)[2:]})", f"{rng.randint(1, 12):02d}/{str(y + rng.randint(1, 4))[2:]}",
         f"{rng.randint(1, 12):02d}/{str(y - rng.randint(1, 3))[2:]}"),
        ("company", "Company", "", brand(rng) + " Ltd", ""),
    ]
    return fb


FORM_TITLES = ["Create your account", "Book a table", "Event registration", "Checkout details", "Request a callback",
               "Join the waiting list", "Update your profile", "Rent a bike"]


def build_form(rng, sid):
    theme, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    name = brand(rng)
    first, last = person(rng)
    today = (rng.randint(1, 28), rng.randint(1, 12), rng.choice((2025, 2026, 2027)))
    bank = field_bank(rng, first, last, today)
    n = rng.randint(4, 6)
    chosen = rng.sample(bank, n)
    mode = rng.choices(["one_bad", "empty_required", "all_ok"], weights=(35, 15, 50))[0]
    bad = set()
    if mode == "one_bad":
        cands = [i for i, f in enumerate(chosen) if f[4] != "" or True]
        bad = {rng.choice(cands)}
    required = {i for i, f in enumerate(chosen) if rng.random() < 0.6 or f[0] == "fullname"}
    values = []
    for i, (k, lab, hlp, good, badv) in enumerate(chosen):
        if i in bad:
            values.append(badv)          # "" for a required field without a printed rule = left empty
            if badv == "":
                required.add(i)
        else:
            values.append(good)
    if mode == "empty_required":
        if not required:
            required.add(rng.randrange(len(chosen)))
        i = rng.choice(sorted(required))
        values[i] = ""
        bad = {i}
    has_terms = rng.random() < 0.6
    terms_on = rng.random() < 0.75
    rows = []
    for i, (k, lab, hlp, good, badv) in enumerate(chosen):
        star = ' <span class="req">*</span>' if i in required else ""
        help_ = f'<div class="help">{E(hlp)}</div>' if hlp else ""
        rows.append(f'<label class="f" data-el="fld_{k}">{E(lab)}{star}</label><input class="i" value="{E(values[i])}">{help_}')
    terms = ""
    if has_terms:
        terms = (f'<label style="display:flex;gap:9px;align-items:center;margin-top:18px" data-el="terms">'
                 f'<input type="checkbox" {"checked" if terms_on else ""} style="width:18px;height:18px">'
                 f'I accept the {E(name)} terms (required)</label>')
    title = rng.choice(FORM_TITLES)
    body = (topbar(rng, theme, name) + f'<div style="max-width:560px;margin:24px auto" class="card">'
            f'<h2 style="margin:0 0 4px">{E(title)}</h2><div class="muted">Fields marked * are required.</div>'
            + "".join(rows) + terms +
            f'<div style="margin-top:20px;display:flex;gap:10px"><button class="btn primary" data-el="submit">Continue</button>'
            f'<button class="btn">Cancel</button></div></div>')
    labels = [f[1] for f in chosen]
    invalid = sorted(bad)
    ready = not invalid and (not has_terms or terms_on)
    meta = {"kind": "form", "labels": labels, "invalid": invalid, "ready": ready, "has_terms": has_terms,
            "terms_on": terms_on, "values": values, "rules": [f[2] for f in chosen], "required": sorted(required),
            "keys": {f[1]: f"fld_{f[0]}" for f in chosen}}
    return {"id": sid, "html": page(theme, title, body), "width": w, "height": h, "meta": meta}


def form_questions(rng, s):
    m = s["meta"]
    out = []
    if len(m["invalid"]) == 1 or (not m["invalid"] and rng.random() < 0.35):
        q = rng.choice(["Which field has a value that breaks the rule shown for it (or a required field left empty)?",
                        "Which field would the form reject as invalid?", "Which field needs correcting before this form can be sent?"])
        gold = m["labels"][m["invalid"][0]] if m["invalid"] else None
        out.append(("form_invalid_field", "choice", q, m["labels"], gold, None if gold else "false_premise", 3))
    q = rng.choice(["Is this form ready to submit as filled in?", "Can this form be submitted now without any error?",
                    "Would pressing Continue go through with the form as shown?"])
    out.append(("form_ready", "noul", q, None, m["ready"], None, 4))
    return out


# ------------------------------------------------------------------------------------------------ toolbar goals
GOALS = [
    ("download this table as a spreadsheet file", "Export CSV", ["Import CSV", "Print", "Copy link"]),
    ("let a colleague open this report", "Share", ["Export PDF", "Duplicate", "Archive"]),
    ("make a copy of this project", "Duplicate", ["Rename", "Move", "Export PDF"]),
    ("hide this item from the list without deleting it", "Archive", ["Delete", "Rename", "Print"]),
    ("permanently remove the selected rows", "Delete", ["Archive", "Clear filters", "Refresh"]),
    ("bring in records from a spreadsheet file", "Import CSV", ["Export CSV", "Refresh", "Duplicate"]),
    ("invite a new teammate to the workspace", "Invite", ["Share", "Duplicate", "Refresh"]),
    ("see only the rows that match a condition", "Filter", ["Sort", "Refresh", "Print"]),
    ("put the rows in date order", "Sort", ["Filter", "Refresh", "Rename"]),
    ("reverse your last edit", "Undo", ["Redo", "Refresh", "Delete"]),
    ("reload the latest data from the server", "Refresh", ["Undo", "Import CSV", "Sort"]),
    ("get a paper copy", "Print", ["Export PDF", "Share", "Copy link"]),
    ("save the report as a PDF file", "Export PDF", ["Print", "Export CSV", "Save draft"]),
    ("keep your edits without making them public", "Save draft", ["Publish", "Preview", "Share"]),
    ("make the page live for visitors now", "Publish", ["Save draft", "Preview", "Schedule"]),
    ("see how the page will look before it goes live", "Preview", ["Publish", "Save draft", "Print"]),
    ("set the post to go live at a later date", "Schedule", ["Publish", "Save draft", "Preview"]),
    ("give this task to a teammate", "Assign", ["Invite", "Share", "Comment"]),
    ("leave a note for your team on this task", "Comment", ["Assign", "Share", "Rename"]),
    ("mark the task as finished", "Mark done", ["Archive", "Assign", "Delete"]),
    ("change the file's name", "Rename", ["Duplicate", "Move", "Copy link"]),
    ("put the file in another folder", "Move", ["Rename", "Copy link", "Duplicate"]),
    ("copy a web address that points to this file", "Copy link", ["Share", "Download", "Move"]),
    ("save the original file to your computer", "Download", ["Copy link", "Export CSV", "Print"]),
]
OTHER_LABELS = sorted({g[1] for g in GOALS} | {x for g in GOALS for x in g[2]})
TABLE_HEAD = [("Customer", "Invoice", "Due", "Amount"), ("Task", "Owner", "Status", "Updated"), ("File", "Owner", "Size", "Modified")]


def build_toolbar(rng, sid):
    theme, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    name = brand(rng)
    goal, correct, near = rng.choice(GOALS)
    mode = rng.choices(["present", "disabled", "absent"], weights=(50, 32, 18))[0]
    labels = list(near) + rng.sample([l for l in OTHER_LABELS if l != correct and l not in near], rng.randint(1, 3))
    if mode != "absent":
        labels.append(correct)
    rng.shuffle(labels)
    btns = []
    for lab in labels:
        dis = "disabled" if (mode == "disabled" and lab == correct) or (lab != correct and rng.random() < 0.12) else ""
        cls = "btn primary" if lab in ("Publish", "Invite", "Share") and not dis and rng.random() < 0.5 else "btn"
        btns.append(f'<button class="{cls}" data-el="b_{option_key(lab)}" {dis}>{E(lab)}</button>')
    hi = rng.randrange(len(TABLE_HEAD))
    head = TABLE_HEAD[hi]
    rows = []
    for _ in range(rng.randint(4, 8)):
        f, l = person(rng)
        date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}"
        cells = [(f"{f} {l}", f"INV-{rng.randint(1000, 9999)}", date, f"{rng.uniform(20, 2000):.2f}"),
                 (rng.choice(["Update pricing page", "Call supplier", "Draft newsletter", "Fix login bug", "Book venue",
                              "Review contract", "Order samples"]), f"{f} {l}", rng.choice(["Open", "In progress", "Blocked", "Done"]), date),
                 (rng.choice(["budget", "notes", "photos", "plan", "invoice", "slides"]) + f"-{rng.randint(1, 99)}." + rng.choice(["pdf", "xlsx", "png", "docx"]),
                  f"{f} {l}", f"{rng.uniform(0.1, 30):.1f} MB", date)][hi]
        rows.append("<tr>" + "".join(f"<td>{E(x)}</td>" for x in cells) + "</tr>")
    sel = rng.random() < 0.5
    body = (topbar(rng, theme, name) + f'<div style="padding:20px 24px"><div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap">'
            f'<h2 style="margin:0 16px 0 0">{E(rng.choice(["Invoices", "Tasks", "Files", "Reports", "Pages"]))}</h2>' + "".join(btns) +
            f'</div><div class="card" style="padding:6px 12px"><table><tr>' + "".join(f"<th>{E(x)}</th>" for x in head) + "</tr>"
            + "".join(rows) + f'</table></div><div class="muted" style="margin-top:10px">{"1 row selected" if sel else "No rows selected"}</div></div>')
    meta = {"kind": "toolbar", "goal": goal, "correct": correct, "mode": mode, "labels": labels}
    return {"id": sid, "html": page(theme, name, body), "width": w, "height": h, "meta": meta}


def toolbar_questions(rng, s):
    m = s["meta"]
    out = []
    if m["mode"] in ("present", "absent"):
        q = rng.choice([f"Which control would you click to {m['goal']}?", f"Which button lets you {m['goal']}?"])
        gold = m["correct"] if m["mode"] == "present" else None
        out.append(("toolbar_goal", "choice", q, m["labels"], gold, None if gold else "not_listed", 3))
    if m["mode"] in ("present", "disabled"):
        q = rng.choice([f"Can you {m['goal']} from this screen right now?", f"Is the control to {m['goal']} available to click right now?"])
        out.append(("toolbar_available", "noul", q, None, m["mode"] == "present", None, 4))
    return out


# ------------------------------------------------------------------------------------------------ settings
SETTINGS = [
    ("Email digest", "One summary of activity each day", "get one summary email a day"),
    ("Push notifications", "Alerts on your phone", "receive alerts on your phone"),
    ("Two-step sign-in", "Ask for a code when you sign in", "be asked for a code when you sign in"),
    ("Auto-save", "Save changes as you type", "have your edits saved as you type"),
    ("Public profile", "Anyone can see your profile", "let anyone see your profile"),
    ("Read receipts", "Show others when you have read a message", "let senders know you have read their messages"),
    ("Location sharing", "Share your location with your team", "show your team where you are"),
    ("Dark theme", "Use dark colours", "see the app in dark colours"),
    ("Compact view", "Show more rows on screen", "fit more rows on the screen"),
    ("Sound effects", "Play a sound when an action completes", "hear a sound when an action completes"),
    ("Weekly report", "A usage report every Monday", "get a usage report each Monday"),
    ("Auto-update", "Install updates automatically", "have updates installed automatically"),
    ("Spell check", "Underline misspelled words", "see misspelled words underlined"),
    ("Calendar sync", "Copy events to your calendar", "have events copied into your calendar"),
]


def build_settings(rng, sid):
    theme, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    name = brand(rng)
    shown = rng.sample(SETTINGS, rng.randint(5, 7))
    target = rng.choice(SETTINGS)
    present = target in shown
    if rng.random() < 0.8 and not present:
        shown[rng.randrange(len(shown))] = target
        present = True
    state = {s[0]: rng.random() < 0.5 for s in shown}
    text_labels = rng.random() < 0.5
    rows = []
    for lab, desc, _ in shown:
        on = state[lab]
        txt = f'<span class="muted" style="width:28px">{"On" if on else "Off"}</span>' if text_labels else ""
        rows.append(f'<div style="display:flex;align-items:center;gap:14px;padding:13px 0;border-bottom:1px solid {theme[6]}">'
                    f'<div style="flex:1"><div data-el="s_{option_key(lab)}" style="font-weight:600;display:inline-block">{E(lab)}</div>'
                    f'<div class="muted">{E(desc)}</div></div>{txt}<div class="sw{" on" if on else ""}"><i></i></div></div>')
    body = (topbar(rng, theme, name, ["Settings", "Account", "Billing", "Help"]) +
            f'<div style="max-width:640px;margin:24px auto" class="card"><h2 style="margin:0 0 6px">Preferences</h2>'
            f'<div class="muted">Changes are saved automatically.</div>' + "".join(rows) + "</div>")
    meta = {"kind": "settings", "labels": [s[0] for s in shown], "state": state, "target": target[0], "goal": target[2],
            "present": present, "switch_text": text_labels}
    return {"id": sid, "html": page(theme, "Preferences", body), "width": w, "height": h, "meta": meta}


def settings_questions(rng, s):
    m = s["meta"]
    out = []
    q = rng.choice([f"Which setting controls whether you {m['goal']}?", f"Which switch would you use to {m['goal']}?"])
    gold = m["target"] if m["present"] else None
    out.append(("settings_which", "choice", q, m["labels"], gold, None if gold else "not_listed", 3))
    q = rng.choice([f"With the switches as shown, will you {m['goal']}?", f"As currently set, do you {m['goal']}?"])
    g2 = m["state"][m["target"]] if m["present"] else None
    out.append(("settings_state", "noul", q, None, g2, None if m["present"] else "insufficient_evidence", 4))
    return out


# ------------------------------------------------------------------------------------------------ dialogs
DIALOGS = [
    ("Delete 3 files?", "Deleted files cannot be recovered.", ["Cancel", "Delete", "Move to archive"],
     {"remove the files for good": "Delete", "keep the files but take them out of this list": "Move to archive",
      "close the dialog without changing anything": "Cancel"}),
    ("Unsaved changes", "You have unsaved changes in “Budget draft”.", ["Save", "Don't save", "Cancel"],
     {"close the document and throw away your edits": "Don't save", "keep your edits and close the document": "Save",
      "go back to editing the document": "Cancel"}),
    ("Leave the call?", "The others can keep talking after you go.", ["Stay", "Leave", "End call for everyone"],
     {"exit the call but let the others continue": "Leave", "stop the call for every participant": "End call for everyone",
      "remain in the call": "Stay"}),
    ("Cancel your plan?", "Your plan renews on the 1st of next month.", ["Keep plan", "Pause for 1 month", "Cancel plan"],
     {"stop paying for a month and then continue": "Pause for 1 month", "end the plan at renewal": "Cancel plan",
      "change nothing about the plan": "Keep plan"}),
    ("Update ready", "Version 4.2 can be installed now.", ["Install now", "Remind me tomorrow", "Skip this version"],
     {"get the update right away": "Install now", "be asked again tomorrow": "Remind me tomorrow",
      "never be offered version 4.2 again": "Skip this version"}),
    ("Show notifications?", "This site wants to send you notifications.", ["Allow", "Block", "Not now"],
     {"receive notifications from the site": "Allow", "refuse and never be asked again": "Block",
      "decide later": "Not now"}),
]
UNRELATED_GOALS = ["rename the file", "change your password", "print the page", "invite a teammate", "export the table"]


def build_dialog(rng, sid):
    theme, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    name = brand(rng)
    title, msg, buttons, goals = rng.choice(DIALOGS)
    buttons = list(buttons)
    if rng.random() < 0.5:
        rng.shuffle(buttons)
    bg_btns = "".join(f'<button class="btn" data-el="bg_{i}">{E(x)}</button>' for i, x in enumerate(rng.sample(OTHER_LABELS, 4)))
    btn_html = "".join(f'<button class="btn{" primary" if i == len(buttons) - 1 else ""}" data-el="d_{option_key(b)}">{E(b)}</button>'
                       for i, b in enumerate(buttons))
    body = (topbar(rng, theme, name) + f'<div style="padding:24px;display:flex;gap:8px">{bg_btns}</div>'
            f'<div style="padding:0 24px" class="muted">' + "<p>" + "</p><p>".join(E(f"{person(rng)[0]} updated item {rng.randint(10, 99)}") for _ in range(6)) + "</p></div>"
            f'<div style="position:fixed;inset:0;background:#0008;display:flex;align-items:center;justify-content:center">'
            f'<div class="card" style="width:440px"><h3 style="margin:0 0 8px">{E(title)}</h3><div class="muted" style="margin-bottom:20px">{E(msg)}</div>'
            f'<div style="display:flex;gap:10px;justify-content:flex-end">{btn_html}</div></div></div>')
    if rng.random() < 0.82:
        goal = rng.choice(sorted(goals))
        gold = goals[goal]
    else:
        goal, gold = rng.choice(UNRELATED_GOALS), None
    meta = {"kind": "dialog", "labels": buttons, "goal": goal, "gold": gold}
    return {"id": sid, "html": page(theme, title, body), "width": w, "height": h, "meta": meta}


def dialog_questions(rng, s):
    m = s["meta"]
    q = rng.choice([f"Which button in the dialog should you press to {m['goal']}?", f"To {m['goal']}, which dialog button do you click?"])
    return [("dialog_button", "choice", q, m["labels"], m["gold"], None if m["gold"] else "not_listed", 3)]


# ------------------------------------------------------------------------------------------------ cart
def build_cart(rng, sid):
    theme, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    name = brand(rng)
    items = rng.sample(PRODUCTS, rng.randint(2, 4))
    lines, limits = [], {}
    for it in items:
        price = round(rng.uniform(3, 40), 2)
        qty = rng.randint(1, 4)
        lim = rng.choice((None, None, 2, 3, 5))
        lines.append([it, price, qty, lim])
    mode = rng.choice(["one_over", "one_over", "none_over"])
    with_lim = [i for i, l in enumerate(lines) if l[3]]
    if not with_lim:
        lines[0][3] = rng.choice((2, 3))
        with_lim = [0]
    for l in lines:
        if l[3] and l[2] > l[3]:
            l[2] = l[3]
    over = None
    if mode == "one_over":
        over = rng.choice(with_lim)
        lines[over][2] = lines[over][3] + rng.randint(1, 2)
    subtotal = round(sum(l[1] * l[2] for l in lines), 2)
    if rng.random() < 0.6:   # put the threshold near the subtotal so the sum matters
        thr = max(10, int(subtotal) + rng.choice((-6, -3, -1, 1, 2, 5)))
    else:
        thr = max(10, int(round(subtotal * rng.uniform(0.5, 1.5) / 10.0)) * 10)
    rows = []
    for it, price, qty, lim in lines:
        note = f'<div class="muted">Limit {lim} per customer</div>' if lim else ""
        rows.append(f'<tr><td><div data-el="c_{option_key(it)}" style="display:inline-block;font-weight:600">{E(it)}</div>{note}</td>'
                    f'<td>{price:.2f}</td><td>{qty}</td></tr>')
    body = (topbar(rng, theme, name, ["Shop", "New in", "Gifts", "Basket"]) +
            f'<div style="max-width:760px;margin:24px auto" class="card"><h2 style="margin:0 0 10px">Your basket</h2>'
            f'<div style="padding:8px 12px;border-radius:7px;background:{theme[0]};margin-bottom:12px">Free shipping on orders of {thr:.2f} or more (before shipping).</div>'
            f'<table><tr><th>Item</th><th>Unit price</th><th>Qty</th></tr>' + "".join(rows) + '</table>'
            f'<div style="margin-top:16px;display:flex;justify-content:flex-end"><button class="btn primary" data-el="checkout">Go to checkout</button></div></div>')
    meta = {"kind": "cart", "labels": [l[0] for l in lines], "lines": lines, "subtotal": subtotal, "threshold": thr,
            "over": None if over is None else lines[over][0]}
    return {"id": sid, "html": page(theme, "Basket", body), "width": w, "height": h, "meta": meta}


def cart_questions(rng, s):
    m = s["meta"]
    q1 = rng.choice(["Does this basket get free shipping under the offer shown?", "Is shipping free for this basket as it stands?"])
    q2 = rng.choice(["Which item's quantity is over its per-customer limit?", "Which item exceeds the limit printed under it?"])
    return [("cart_free_shipping", "noul", q1, None, round(m["subtotal"], 2) >= m["threshold"], None, 4),
            ("cart_over_limit", "choice", q2, m["labels"], m["over"], None if m["over"] else "false_premise", 3)]


KINDS = {"form": (0.28, build_form, form_questions), "toolbar": (0.26, build_toolbar, toolbar_questions),
         "settings": (0.20, build_settings, settings_questions), "dialog": (0.12, build_dialog, dialog_questions),
         "cart": (0.14, build_cart, cart_questions)}
ELEMENT_PREFIX = {"form": "fld_", "toolbar": "b_", "settings": "s_", "dialog": "d_", "cart": "c_"}


# ------------------------------------------------------------------------------------------------ build
def render(screens: list[dict], work: Path, conc: int = 6) -> dict:
    jobs = []
    for s in screens:
        hp = work / f"{s['id']}.html"
        hp.write_text(s["html"])
        jobs.append({"id": s["id"], "html": str(hp), "out": str(work / f"{s['id']}.png"), "width": s["width"], "height": s["height"]})
    jp, rp = work / "jobs.jsonl", work / "results.jsonl"
    jp.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    subprocess.run(["node", str(RENDER), str(jp), str(rp), str(conc)], check=True)
    return {r["id"]: r for r in (json.loads(l) for l in rp.read_text().splitlines() if l.strip())}


def element_for(kind: str, label: str, form_keys: dict | None = None) -> str:
    if kind == "form":
        return form_keys[label]
    return ELEMENT_PREFIX[kind] + option_key(label)


def build(n_screens: int, work: Path) -> tuple[list[dict], Counter]:
    rng = rng_for(SEED)
    screens = []
    for i in range(n_screens):
        kind = rng.choices(list(KINDS), weights=[v[0] for v in KINDS.values()])[0]
        s = KINDS[kind][1](rng_for(SEED, "screen", i), f"s{i:05d}")
        s["kind"] = kind
        screens.append(s)
    results = render(screens, work)
    bench = bench_image_shas()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows, stats = [], Counter()
    for s in screens:
        r = results.get(s["id"])
        if not r or not r["ok"]:
            stats["render_failed"] += 1
            continue
        png = work / f"{s['id']}.png"
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        if sha in bench:
            stats["bench_sha"] += 1
            continue
        els = {e["el"]: e for e in r["elements"]}
        kind = s["kind"]
        form_keys = s["meta"].get("keys")
        qrng = rng_for(SEED, "q", s["id"])
        kept_any = False
        for fam, ftype, q, labels, gold, unk, diff in KINDS[kind][2](qrng, s):
            if labels is not None:
                need = [element_for(kind, l, form_keys) for l in labels]
                if not all(n in els and els[n]["visible"] for n in need):
                    stats["option_not_visible"] += 1
                    continue
            if fam == "form_ready" and s["meta"]["has_terms"] and not els.get("terms", {}).get("visible"):
                stats["terms_not_visible"] += 1
                continue
            if fam == "form_ready":
                need = [element_for(kind, l, form_keys) for l in s["meta"]["labels"]]
                if not all(n in els and els[n]["visible"] for n in need):
                    stats["field_not_visible"] += 1
                    continue
            if fam in ("cart_free_shipping",):
                need = [element_for(kind, l) for l in s["meta"]["labels"]]
                if not all(n in els and els[n]["visible"] for n in need):
                    stats["item_not_visible"] += 1
                    continue
            field = {"type": ftype, "question": q}
            g = gold
            if ftype == "choice":
                taken = set()
                opts = [{"key": option_key(l, taken), "text": l} for l in labels]
                keymap = {l: o["key"] for l, o in zip(labels, opts)}
                field["options"] = opts
                g = keymap[gold] if gold is not None else None
            kept_any = True
            rows.append({
                "id": None, "source": "I", "dataset": "gui_synthetic", "family": "gui_action",
                "difficulty": diff + (1 if g is None and diff < 5 else 0),
                "state": {"screen": f"a screenshot of a {kind} screen in a web app"},
                "images": [f"p3/images/gui/{sha}.png"], "field": field, "gold": g, "unknown_reason": unk,
                "gold_kind": "constructed", "parent_id": None,
                "provenance": {"licence": "generated", "renderer": "puppeteer-core + headless Chromium (scripts/p3/gui_templates/render.mjs)",
                               "screen_id": s["id"], "screen_kind": kind, "task": fam, "image_sha256": [sha],
                               "viewport": [s["width"], s["height"]], "screen_meta": s["meta"], "upstream_split": "generated",
                               "content": "invented brands, names and products"},
            })
        if kept_any:
            dst = IMG_DIR / f"{sha}.png"
            if not dst.exists():
                shutil.copyfile(png, dst)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, default=2600)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-work", default=None, help="directory for the HTML/PNG work files (default: a temp dir)")
    a = ap.parse_args()
    work = Path(a.keep_work) if a.keep_work else Path(tempfile.mkdtemp(prefix="p3gui-"))
    work.mkdir(parents=True, exist_ok=True)
    rows, stats = build(a.screens, work)
    # the screen-level context line is enough; the question never names the answer
    for i, r in enumerate(rows):
        r["id"] = f"p3-I-gui-{i:06d}"
    n = write(a.out, rows)
    print(json.dumps({"rows": n, "screens": a.screens, "stats": dict(stats),
                      "by_task": dict(Counter(r["provenance"]["task"] for r in rows)),
                      "unknown": sum(r["gold"] is None for r in rows),
                      "images": len({r["images"][0] for r in rows})}, indent=1))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
