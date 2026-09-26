"""Desktop / web screens for gen_screens.py family screen_locate (where is X / which marked box is X).

Templates: email client, analytics dashboard, cloud file browser, shop product page, calendar. Invented content.
"""
from __future__ import annotations

from base import E, Screen, doc
from icons import svg

SIZES = [(1280, 800), (1280, 720), (1180, 820), (1024, 768), (1100, 760), (1280, 900)]
THEMES = [  # bg, surface, text, muted, accent, border, on_accent, font, dark
    ("#f6f7f9", "#ffffff", "#1d2330", "#6b7383", "#2f6fed", "#d9dde5", "#ffffff", "-apple-system, Helvetica, Arial, sans-serif", False),
    ("#fbf8f3", "#ffffff", "#2b2622", "#7a7068", "#b4532a", "#e6ddd2", "#ffffff", "Georgia, 'Times New Roman', serif", False),
    ("#eef3f1", "#ffffff", "#15302a", "#5b726c", "#12805c", "#cfdcd7", "#ffffff", "Verdana, Geneva, sans-serif", False),
    ("#121417", "#1c1f24", "#e8eaed", "#9aa0a6", "#7aa2ff", "#2e333a", "#0b1020", "-apple-system, Helvetica, Arial, sans-serif", True),
    ("#f3f0fa", "#ffffff", "#221a36", "#6e6485", "#6b3fd4", "#ddd5ee", "#ffffff", "'Trebuchet MS', Helvetica, sans-serif", False),
    ("#ffffff", "#f7f7f7", "#111111", "#666666", "#111111", "#e2e2e2", "#ffffff", "'Avenir Next', Avenir, Helvetica, sans-serif", False),
]
FIRST = ["Anna", "Tomas", "Leila", "Marek", "Priya", "Jonah", "Sofia", "Kwame", "Ines", "Ravi", "Helga", "Omar", "Yuki", "Bruno"]
LAST = ["Kerr", "Novak", "Haddad", "Okafor", "Brandt", "Moreau", "Sato", "Ferreira", "Iqbal", "Duarte", "Varga", "Olsen"]
BRAND_A = ["Vel", "Quor", "Tami", "Brin", "Oda", "Zell", "Poma", "Kiro", "Lume", "Sorb", "Yent", "Fenn", "Marro", "Tovi"]
BRAND_B = ["wick", "nest", "loop", "pad", "deck", "ora", "hub", "field", "mint", "works", "base", "port", "lane", "grove"]


def brand(rng):
    return rng.choice(BRAND_A) + rng.choice(BRAND_B)


def css(t):
    bg, surf, text, muted, acc, border, onacc, font, dark = t
    return f"""
body{{color:{text};font-size:14px}}
.top{{display:flex;align-items:center;gap:14px;padding:10px 18px;background:{surf};border-bottom:1px solid {border};height:56px}}
.logo{{font-weight:700;font-size:18px;color:{acc}}}
.btn{{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:7px;border:1px solid {border};background:{surf};color:{text};font-size:13.5px;white-space:nowrap}}
.btn.pri{{background:{acc};border-color:{acc};color:{onacc}}}
.ib{{width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;border-radius:8px;color:{muted}}}
.side{{width:210px;background:{surf};border-right:1px solid {border};padding:12px 8px;flex:none}}
.side a{{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:7px;color:{text};text-decoration:none;margin-bottom:2px}}
.side a.cur{{background:{bg};font-weight:600;color:{acc}}}
.card{{background:{surf};border:1px solid {border};border-radius:10px;padding:14px}}
.muted{{color:{muted}}} .fld{{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid {border};border-radius:8px;background:{bg};color:{muted}}}
table{{border-collapse:collapse;width:100%}} td,th{{text-align:left;padding:8px 10px;border-bottom:1px solid {border}}} th{{color:{muted};font-size:12.5px}}
.scrim{{position:fixed;inset:0;background:#0009;display:flex;align-items:center;justify-content:center}}
.dlg{{background:{surf};border-radius:12px;padding:22px;width:440px;box-shadow:0 10px 40px #0006}}
"""


def page(t, s: Screen, body: str) -> str:
    return doc(css(t), body, t[0], t[7], s.width, s.height)


def maybe_dialog(rng, s: Screen, p=0.14) -> str:
    if rng.random() >= p:
        return ""
    title, msg, b1, b2 = rng.choice([("Your session is about to expire", "Stay signed in to keep your changes.", "Sign out", "Stay signed in"),
                                     ("We use cookies", "Choose which cookies we may store.", "Reject all", "Accept all"),
                                     ("Try the new layout?", "You can switch back at any time.", "No thanks", "Try it"),
                                     ("Unsaved changes", "Leave this page without saving?", "Stay", "Leave")])
    s.dialog = "dlg"
    s.meta["dialog_title"] = title
    a1 = s.el("dlg_b1", f"the {b1} button in the dialog", b1, [], None)
    a2 = s.el("dlg_b2", f"the {b2} button in the dialog", b2, [], None)
    w = rng.choice([440, 520, 600])
    return (f'<div class="scrim"><div class="dlg" style="width:{w}px;padding:{rng.choice([22, 34])}px" {s.el("dlg", None, role="dialog")}>'
            f'<h3 style="margin:0 0 8px">{E(title)}</h3><div class="muted" style="margin-bottom:18px">{E(msg)}</div>'
            f'<div style="display:flex;gap:10px;justify-content:flex-end"><span class="btn" {a1}>{E(b1)}</span><span class="btn pri" {a2}>{E(b2)}</span></div></div></div>')


def topbar(rng, s, t, name, search_ph="Search"):
    srch = s.el("search", "the search field", None, ["search"], None)
    bell = s.el("bell", "the notifications bell icon", None, ["see notifications"], None, small=True)
    help_ = s.el("help", "the help (question mark) icon", None, ["open help"], None, small=True)
    f, l = rng.choice(FIRST), rng.choice(LAST)
    av = s.el("avatar", "the profile avatar", None, ["open your profile"], None, small=True)
    return (f'<div class="top"><span class="logo">{E(name)}</span><div class="fld" style="width:{rng.randint(260, 420)}px;margin-left:{rng.randint(10, 80)}px" {srch}>'
            f'{svg("search", 16)}{E(search_ph)}</div><span style="margin-left:auto" class="ib" {help_}>{svg("help", 20)}</span>'
            f'<span class="ib" {bell}>{svg("bell", 20)}</span><span {av} style="width:32px;height:32px;border-radius:16px;background:{t[4]};color:{t[6]};'
            f'display:inline-flex;align-items:center;justify-content:center;font-weight:700">{f[0]}{l[0]}</span></div>')


def sidebar(rng, s, t, items, cur=0):
    out = []
    for i, (lab, ic) in enumerate(items):
        a = s.el(f"side_{i}", f"the {lab} link in the sidebar", lab, [f"open {lab}"], None)
        out.append(f'<a class="{"cur" if i == cur else ""}" {a}>{svg(ic, 17)}{E(lab)}</a>')
    return f'<div class="side">{"".join(out)}</div>'


# ------------------------------------------------------------------------------------------------ email
def email(rng) -> Screen:
    t, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    s = Screen("desktop", "email", w, h)
    s.screen_desc = "a web email client"
    name = brand(rng) + " Mail"
    folders = [("Inbox", "mail"), ("Starred", "star"), ("Sent", "send"), ("Drafts", "file"), ("Archive", "archive"), ("Spam", "help"), ("Trash", "trash")]
    folders = folders[:rng.randint(5, 7)]
    comp = s.el("compose", "the Compose button", "Compose", [], None)
    side = sidebar(rng, s, t, folders).replace('<div class="side">', f'<div class="side"><div class="btn pri" style="margin:4px 8px 14px;padding:10px 18px" {comp}>{svg("pencil", 16)}Compose</div>', 1)
    tools = rng.sample([("Archive", "archive"), ("Delete", "trash"), ("Mark as read", "mail"), ("Move to", "folder"), ("Label", "tag"),
                        ("Refresh", "refresh"), ("Reply", "reply")], rng.randint(4, 6))
    tb = "".join(f'<span class="btn" {s.el("tb_" + str(i), f"the {lab} button in the toolbar", lab, [], None)}>{svg(ic, 15)}{E(lab)}</span>' for i, (lab, ic) in enumerate(tools))
    rows = []
    for i in range(rng.randint(8, 12)):
        f, l = rng.choice(FIRST), rng.choice(LAST)
        subj = rng.choice(["Quarterly figures", "Lunch on Friday?", "Your invoice #", "Re: venue booking", "Flight change", "Draft contract",
                           "Team offsite plan", "Password reset", "Photos from the trip", "Delivery update"])
        subj = subj + (str(rng.randint(100, 999)) if subj.endswith("#") else "")
        star = s.el(f"star_{i}", f"the star icon on the email from {f} {l}", None, [], None, small=True) if i < 4 and f"the star icon on the email from {f} {l}" not in [e["desc"] for e in s.els.values()] else ""
        rows.append(f'<tr><td style="width:28px">{("<span " + star + ">" + svg("star", 16) + "</span>") if star else ""}</td><td style="font-weight:{600 if rng.random() < .4 else 400}">{E(f)} {E(l)}</td>'
                    f'<td>{E(subj)}</td><td class="muted" style="text-align:right">{rng.randint(1, 28)} {rng.choice(["Mar", "Apr", "May", "Jun"])}</td></tr>')
    body = (topbar(rng, s, t, name, "Search mail") + f'<div style="display:flex;height:calc(100% - 56px)">{side}'
            f'<div style="flex:1;padding:14px 18px;overflow:hidden"><div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">{tb}</div>'
            f'<div class="card" style="padding:0 6px"><table>{"".join(rows)}</table></div></div></div>')
    for d in ["the Snooze button in the toolbar", "the Print button in the toolbar", "the Calendar link in the sidebar", "the Forward button in the toolbar"]:
        s.add_absent(d)
    s.html = page(t, s, body + maybe_dialog(rng, s))
    return s


# ------------------------------------------------------------------------------------------------ dashboard
def dashboard(rng) -> Screen:
    t, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    s = Screen("desktop", "dashboard", w, h)
    s.screen_desc = "a web analytics dashboard"
    name = brand(rng)
    items = rng.sample([("Overview", "home"), ("Reports", "chart"), ("Customers", "user"), ("Orders", "cart"), ("Products", "tag"),
                        ("Invoices", "file"), ("Team", "user"), ("Settings", "gear")], 6)
    side = sidebar(rng, s, t, items)
    rng_btn = s.el("range", "the date range selector", "Last 30 days", [], None)
    exp = s.el("export", "the Export button", "Export", [], None)
    flt = s.el("filter", "the Filter button", "Filter", [], None)
    kpis = rng.sample(["Revenue", "New customers", "Orders", "Refunds", "Active users", "Avg. basket"], 3)
    cards = []
    for i, k in enumerate(kpis):
        v = s.el(f"kpi_{i}", f"the View report link on the {k} card", "View report", [], None)
        cards.append(f'<div class="card" style="flex:1"><div class="muted">{E(k)}</div><div style="font-size:26px;font-weight:700;margin:6px 0">'
                     f'{rng.randint(10, 990)}{rng.choice(["", "k", ".4k", "%"])}</div><span style="color:{t[4]};font-size:13px" {v}>View report →</span></div>')
    bars = "".join(f'<div style="flex:1;margin:0 4px;background:{t[4]};opacity:.8;height:{rng.randint(20, 160)}px;align-self:flex-end"></div>' for _ in range(12))
    dl = s.el("chart_dl", "the download icon on the chart", None, [], None, small=True)
    body = (topbar(rng, s, t, name) + f'<div style="display:flex;height:calc(100% - 56px)">{side}<div style="flex:1;padding:16px 20px;overflow:hidden">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px"><h2 style="margin:0;flex:1">{E(items[0][0])}</h2>'
            f'<span class="btn" {rng_btn}>{svg("calendar", 15)}Last 30 days</span><span class="btn" {flt}>{svg("filter", 15)}Filter</span>'
            f'<span class="btn pri" {exp}>{svg("download", 15)}Export</span></div><div style="display:flex;gap:14px">{"".join(cards)}</div>'
            f'<div class="card" style="margin-top:14px"><div style="display:flex;justify-content:space-between"><b>Sales by month</b><span class="ib" {dl}>{svg("download", 18)}</span></div>'
            f'<div style="display:flex;height:180px;margin-top:10px">{bars}</div></div></div></div>')
    for d in ["the Share button", "the Print button", "the Import button", "the Add widget button"]:
        s.add_absent(d)
    s.html = page(t, s, body + maybe_dialog(rng, s))
    return s


# ------------------------------------------------------------------------------------------------ files
def files(rng) -> Screen:
    t, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    s = Screen("desktop", "files", w, h)
    s.screen_desc = "a cloud file browser"
    name = brand(rng) + " Drive"
    items = [("My files", "folder"), ("Shared with me", "user"), ("Recent", "refresh"), ("Starred", "star"), ("Bin", "trash")]
    new = s.el("new", "the New button", "New", [], None)
    side = sidebar(rng, s, t, items).replace('<div class="side">', f'<div class="side"><div class="btn pri" style="margin:4px 8px 14px;padding:10px 18px" {new}>{svg("plus", 16)}New</div>', 1)
    up = s.el("upload", "the Upload button", "Upload", [], None)
    shr = s.el("share", "the Share button", "Share", [], None)
    gv = s.el("gridview", "the grid view icon", None, [], None, small=True)
    lv = s.el("listview", "the list view icon", None, [], None, small=True)
    srt = s.el("sort", "the Sort by name dropdown", "Sort by name", [], None)
    tiles = []
    names = rng.sample(["Budget 2026.xlsx", "Team photo.jpg", "Contract draft.docx", "Slides - Q3.pptx", "Receipts", "Invoices", "Logo final.png",
                        "Meeting notes.txt", "Travel plan.pdf", "Archive 2025", "Recipes", "Floor plan.pdf", "Podcast ep 12.mp3", "Survey results.csv"], rng.randint(8, 12))
    for i, n in enumerate(names):
        folder = "." not in n
        a = s.el(f"f{i}", f"the file tile '{n}'" if not folder else f"the folder tile '{n}'", n, [], None)
        tiles.append(f'<div class="card" style="width:170px;padding:12px" {a}><div style="color:{t[4]}">{svg("folder" if folder else "file", 34)}</div>'
                     f'<div style="margin-top:8px;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{E(n)}</div></div>')
    used = rng.randint(10, 95)
    body = (topbar(rng, s, t, name, "Search in Drive") + f'<div style="display:flex;height:calc(100% - 56px)">{side}<div style="flex:1;padding:16px 20px;overflow:hidden">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px"><h2 style="margin:0;flex:1">My files</h2><span class="btn" {srt}>{svg("sort", 15)}Sort by name</span>'
            f'<span class="ib" {gv}>{svg("grid", 18)}</span><span class="ib" {lv}>{svg("menu", 18)}</span><span class="btn" {up}>{svg("upload", 15)}Upload</span>'
            f'<span class="btn" {shr}>{svg("share", 15)}Share</span></div><div style="display:flex;flex-wrap:wrap;gap:12px">{"".join(tiles)}</div>'
            f'<div class="muted" style="margin-top:14px">{used}% of 15 GB used</div></div></div>')
    for d in ["the Download button", "the New folder button", "the Rename button", "the Details (info) icon"]:
        s.add_absent(d)
    s.html = page(t, s, body + maybe_dialog(rng, s))
    return s


# ------------------------------------------------------------------------------------------------ product page
def product(rng) -> Screen:
    t, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    s = Screen("desktop", "product", w, h)
    s.screen_desc = "an online shop product page"
    name = brand(rng)
    cart = s.el("cart", "the basket (cart) icon", None, [], None, small=True)
    srch = s.el("search", "the search field", None, [], None)
    acc = s.el("account", "the Account link", "Account", [], None)
    nav = "".join(f'<span style="margin-right:18px" {s.el(f"cat_{i}", f"the {c} menu link", c, [], None)}>{E(c)}</span>'
                  for i, c in enumerate(rng.sample(["New in", "Women", "Men", "Home", "Kids", "Sale", "Gifts"], 5)))
    prod = rng.choice(["Linen overshirt", "Wool beanie", "Canvas tote", "Trail runner", "Rain jacket", "Cotton hoodie"])
    price = f"£{rng.randint(15, 140)}.{rng.choice(['00', '50', '99'])}"
    sizes = "".join(f'<span class="btn" style="min-width:44px;justify-content:center" {s.el(f"size_{z}", f"the size {z} button", z, [], None)}>{z}</span>'
                    for z in ["XS", "S", "M", "L", "XL"][rng.randint(0, 1):])
    add = s.el("add", "the Add to basket button", "Add to basket", [], None)
    wish = s.el("wish", "the heart (wishlist) icon", None, [], None, small=True)
    minus = s.el("qminus", "the quantity minus button", None, [], None, small=True)
    plus = s.el("qplus", "the quantity plus button", None, [], None, small=True)
    tabs = "".join(f'<span style="margin-right:24px;padding-bottom:6px;{"border-bottom:2px solid " + t[4] if i == 0 else ""}" {s.el(f"tab_{i}", f"the {x} tab", x, [], None)}>{E(x)}</span>'
                   for i, x in enumerate(["Description", "Reviews", "Delivery & returns"]))
    rev = s.el("reviews", "the reviews link", f"{rng.randint(12, 480)} reviews", [], None)
    body = (f'<div class="top"><span class="logo">{E(name)}</span><span style="margin-left:24px">{nav}</span>'
            f'<div class="fld" style="margin-left:auto;width:240px" {srch}>{svg("search", 15)}Search</div><span {acc}>Account</span>'
            f'<span class="ib" {cart}>{svg("cart", 22)}</span></div>'
            f'<div style="display:flex;gap:40px;padding:28px {rng.randint(30, 90)}px">'
            f'<div style="width:{rng.randint(380, 480)}px;height:{rng.randint(380, 480)}px;border-radius:12px;background:linear-gradient(135deg,#c9b8a6,#8a7968)"></div>'
            f'<div style="flex:1"><h1 style="margin:0 0 8px">{E(prod)}</h1><div style="display:flex;gap:14px;align-items:center"><span style="font-size:22px">{price}</span>'
            f'<span class="muted" {rev}>★★★★☆ reviews</span></div><div class="muted" style="margin:18px 0 8px">Size</div><div style="display:flex;gap:8px">{sizes}</div>'
            f'<div class="muted" style="margin:18px 0 8px">Quantity</div><div style="display:flex;align-items:center;gap:12px"><span class="ib" style="border:1px solid {t[5]}" {minus}>{svg("minus", 16)}</span>'
            f'<b>1</b><span class="ib" style="border:1px solid {t[5]}" {plus}>{svg("plus", 16)}</span></div>'
            f'<div style="display:flex;gap:10px;margin-top:22px"><span class="btn pri" style="padding:12px 26px;font-size:15px" {add}>Add to basket</span>'
            f'<span class="ib" style="border:1px solid {t[5]};width:46px;height:46px" {wish}>{svg("heart", 22)}</span></div>'
            f'<div style="margin-top:28px">{tabs}</div><p class="muted" style="max-width:460px">Made from soft, durable fabric. Machine washable at 30°.</p></div></div>')
    for d in ["the Buy now button", "the size guide link", "the Compare button", "the share icon"]:
        s.add_absent(d)
    s.html = page(t, s, body + maybe_dialog(rng, s))
    return s


# ------------------------------------------------------------------------------------------------ calendar
def calendar(rng) -> Screen:
    t, (w, h) = rng.choice(THEMES), rng.choice(SIZES)
    s = Screen("desktop", "calendar", w, h)
    s.screen_desc = "a web calendar in month view"
    name = brand(rng) + " Calendar"
    month = rng.choice(["March", "April", "June", "September", "October", "November"])
    prev = s.el("prev", "the previous-month arrow", None, [], None, small=True)
    nxt = s.el("next", "the next-month arrow", None, [], None, small=True)
    today = s.el("today", "the Today button", "Today", [], None)
    views = "".join(f'<span class="btn" style="border-radius:0;{"background:" + t[4] + ";color:" + t[6] if v == "Month" else ""}" {s.el("v_" + v, f"the {v} view button", v, [], None)}>{v}</span>'
                    for v in ["Day", "Week", "Month"])
    create = s.el("create", "the Create event button", "Create event", [], None)
    start = rng.randrange(7)
    cells = []
    events = {}
    evnames = rng.sample(["Dentist", "Team lunch", "Gym", "Book club", "Rent due", "Flight to Oslo", "Piano lesson", "Sprint review", "Dinner with Ines", "Car service"], 5)
    days = rng.sample(range(1, 29), 5)
    for d, e in zip(days, evnames):
        events[d] = e
    for i in range(35):
        d = i - start + 1
        inner = ""
        if 1 <= d <= 30:
            inner = f'<div class="muted" style="font-size:12px">{d}</div>'
            if d in events:
                e = events[d]
                inner += (f'<div {s.el("ev_" + str(d), f"the {e} event", e, [], None)} style="margin-top:4px;background:{t[4]};color:{t[6]};'
                          f'border-radius:4px;padding:2px 6px;font-size:12px;white-space:nowrap;overflow:hidden">{E(e)}</div>')
        cells.append(f'<div style="border-right:1px solid {t[5]};border-bottom:1px solid {t[5]};padding:6px;min-width:0">{inner}</div>')
    heads = "".join(f'<div class="muted" style="padding:6px;font-size:12px">{d}</div>' for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    body = (topbar(rng, s, t, name) + f'<div style="padding:12px 20px;display:flex;align-items:center;gap:10px"><span class="btn pri" {create}>{svg("plus", 15)}Create event</span>'
            f'<span class="btn" {today}>Today</span><span class="ib" {prev}>{svg("chev_l", 18)}</span><span class="ib" {nxt}>{svg("chev_r", 18)}</span>'
            f'<h2 style="margin:0 0 0 8px;flex:1">{month}</h2><span style="display:flex">{views}</span></div>'
            f'<div style="margin:0 20px;border-left:1px solid {t[5]};border-top:1px solid {t[5]};display:grid;grid-template-columns:repeat(7,1fr)">{heads}</div>'
            f'<div style="margin:0 20px;border-left:1px solid {t[5]};display:grid;grid-template-columns:repeat(7,1fr);grid-auto-rows:{(h - 56 - 64 - 40) // 5}px">{"".join(cells)}</div>')
    for d in ["the Year view button", "the Print button", "the Tasks panel", "the Settings gear"]:
        s.add_absent(d)
    s.meta["events"] = {str(k): v for k, v in events.items()}
    s.html = page(t, s, body + maybe_dialog(rng, s))
    return s


TEMPLATES = {"email": (email, 22), "dashboard": (dashboard, 20), "files": (files, 20), "product": (product, 18), "calendar": (calendar, 20)}
