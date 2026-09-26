"""Phone-sized screens (390x844 CSS px, rendered at 1.5x -> 585x1266 PNG) for gen_screens.py (families mobile_*).

Templates: settings, chat (optionally with the keyboard up), chat list with a bottom nav, map-like place list, checkout,
permission dialog over an app, notification shade with quick tiles, music player. All brands, people and places are
invented. Every clickable thing we may ask about carries data-el and a registry entry (base.Screen.el).
"""
from __future__ import annotations

from base import E, Screen, doc
from icons import svg

W, H, DPR = 390, 844, 1.5

THEMES = [  # bg, surface, text, muted, accent, border, on_accent, danger, font, dark
    ("#f2f2f7", "#ffffff", "#111111", "#6b6b70", "#0a7cff", "#d8d8dc", "#ffffff", "#e5322d", "-apple-system, Helvetica, Arial, sans-serif", False),
    ("#f7f9f7", "#ffffff", "#1b1f1c", "#5f6b63", "#1f8a4c", "#dde3de", "#ffffff", "#c62828", "Roboto, 'Helvetica Neue', Arial, sans-serif", False),
    ("#000000", "#1c1c1e", "#f2f2f2", "#9a9aa0", "#0a84ff", "#2c2c2e", "#ffffff", "#ff453a", "-apple-system, Helvetica, Arial, sans-serif", True),
    ("#f6f3fb", "#ffffff", "#1f1830", "#6c6480", "#6b3fd4", "#e1dbee", "#ffffff", "#c2185b", "'Avenir Next', Avenir, Helvetica, sans-serif", False),
    ("#0e1a1f", "#16262d", "#e3eef2", "#8fa7b0", "#35c2b0", "#27404a", "#04201c", "#ff8a80", "Verdana, Geneva, sans-serif", True),
    ("#fffaf4", "#ffffff", "#2a2420", "#7a6f66", "#d0551f", "#eadfd3", "#ffffff", "#b00020", "'Trebuchet MS', Helvetica, sans-serif", False),
]
FIRST = ["Anna", "Tomas", "Leila", "Marek", "Priya", "Jonah", "Sofia", "Kwame", "Ines", "Ravi", "Helga", "Omar", "Yuki",
         "Bruno", "Maeve", "Tariq", "Lena", "Diego", "Nadia", "Felix", "Chiara", "Emeka", "Ruth", "Sven"]
LAST = ["Kerr", "Novak", "Haddad", "Okafor", "Brandt", "Moreau", "Sato", "Ferreira", "Iqbal", "Duarte", "Varga", "Olsen"]
APPS = ["Splitly", "Snapnote", "Tidewalk", "Quorra", "Pellomap", "Brinbank", "Velvox", "Kirochat", "Lumegram", "Sorbfit",
        "Nibustore", "Dessacam", "Wexaride", "Hesknotes", "Olvimusic"]
PLACES = ["Café Lumen", "Brindle Bakery", "Quorn Street Pizza", "The Salt Yard", "Marrow & Vine", "Tovi Noodle Bar",
          "Harbour Books", "Kiro Coffee", "Pello Pharmacy", "Olvi Hotel", "Ulma Petrol", "Fenn Grocers", "Wexa Gym",
          "Dessa Dumplings", "Rask Tapas"]


def css(t):
    bg, surf, text, muted, acc, border, onacc, danger, font, dark = t
    return f"""
.ph{{display:flex;flex-direction:column;width:{W}px;height:{H}px;color:{text};font-size:16px}}
.sb{{height:44px;display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-weight:600;font-size:15px;flex:none}}
.sb .ic{{display:flex;gap:6px;align-items:center}}
.ct{{flex:1;overflow:hidden;position:relative}}
.hd{{display:flex;align-items:center;gap:12px;padding:8px 16px 10px;flex:none}}
.hd h1{{font-size:22px;margin:0;flex:1}} .hd h2{{font-size:18px;margin:0;flex:1;font-weight:600}}
.ib{{width:40px;height:40px;display:flex;align-items:center;justify-content:center;color:{acc};border-radius:20px;flex:none}}
.ib.dim{{color:{muted};opacity:.45}}
.card{{background:{surf};border-radius:12px;margin:8px 14px;overflow:hidden}}
.row{{display:flex;align-items:center;gap:12px;padding:12px 14px;border-bottom:1px solid {border};min-height:50px}}
.row:last-child{{border-bottom:none}} .row .lb{{flex:1}} .row .val{{color:{muted};font-size:14px}}
.row.dis{{opacity:.45}} .sub{{color:{muted};font-size:12.5px}}
.tg{{width:50px;height:30px;border-radius:15px;background:{border};position:relative;flex:none}}
.tg.on{{background:{acc}}} .tg i{{position:absolute;top:3px;left:3px;width:24px;height:24px;border-radius:50%;background:#fff;box-shadow:0 1px 3px #0005}}
.tg.on i{{left:23px}}
.chip{{display:inline-flex;align-items:center;padding:7px 13px;border-radius:17px;border:1px solid {border};background:{surf};font-size:14px;white-space:nowrap}}
.chip.sel{{background:{acc};color:{onacc};border-color:{acc}}}
.btn{{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:10px 16px;border-radius:10px;border:1px solid {border};background:{surf};color:{text};font-size:15px;font-weight:600}}
.btn.pri{{background:{acc};border-color:{acc};color:{onacc}}} .btn.dis{{opacity:.38}}
.btn.sm{{padding:6px 11px;font-size:13px;border-radius:8px}}
.nav{{height:62px;display:flex;border-top:1px solid {border};background:{surf};flex:none}}
.nav div{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;font-size:11px;color:{muted}}}
.nav div.cur{{color:{acc};font-weight:600}}
.home{{height:26px;display:flex;align-items:center;justify-content:center;flex:none}} .home i{{width:134px;height:5px;border-radius:3px;background:{text};opacity:.85}}
.fld{{display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:10px;background:{bg};border:1px solid {border};font-size:15px}}
.ph .muted{{color:{muted}}} .av{{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex:none}}
.badge{{background:{acc};color:{onacc};font-size:12px;font-weight:700;border-radius:10px;padding:1px 7px}}
.scrim{{position:absolute;left:0;top:0;right:0;bottom:0;background:#000a;display:flex;align-items:center;justify-content:center}}
.dlg{{background:{surf};border-radius:16px;width:310px;padding:20px 18px;box-shadow:0 10px 40px #0007}}
.dlg h3{{margin:0 0 8px;font-size:18px}}
.kb{{background:{"#2b2b2e" if dark else "#d1d4db"};padding:6px 3px 4px;flex:none}}
.kr{{display:flex;justify-content:center;gap:5px;margin:6px 0}}
.key{{width:33px;height:42px;border-radius:6px;background:{"#5a5a5e" if dark else "#fff"};display:flex;align-items:center;justify-content:center;font-size:19px;box-shadow:0 1px 0 #0004;color:{text}}}
.key.w{{width:48px;font-size:13px}} .key.sp{{width:180px;font-size:14px}}
"""


def chrome(rng, t, s: Screen, content: str, nav: str = "", kb: str = "", overlay: str = "") -> str:
    tm = f"{rng.randint(7, 22)}:{rng.randint(0, 59):02d}"
    sb = (f'<div class="sb"><span>{tm}</span><span class="ic">{svg("wifi", 16)}'
          f'<span style="display:inline-block;width:24px;height:12px;border:1.5px solid currentColor;border-radius:3px;position:relative">'
          f'<span style="position:absolute;left:1px;top:1px;bottom:1px;width:{rng.randint(4, 19)}px;background:currentColor;border-radius:1px"></span></span></span></div>')
    body = f'<div class="ph">{sb}<div class="ct">{content}{overlay}</div>{kb}{nav}<div class="home"><i></i></div></div>'
    return doc(css(t), body, t[0], t[8], W, H)


def bottom_nav(s: Screen, tabs, cur: int, t) -> str:
    out = []
    for i, (lab, ic, goal) in enumerate(tabs):
        if i == cur:
            out.append(f'<div class="cur">{svg(ic, 22)}<span>{E(lab)}</span></div>')
        else:
            a = s.el(f"nav_{i}", f"the {lab} tab", lab, [goal], "switches to another tab", group="nav")
            out.append(f'<div {a}>{svg(ic, 22)}<span>{E(lab)}</span></div>')
    return f'<div class="nav">{"".join(out)}</div>'


def maybe_dialog(rng, s: Screen, t, p=0.14) -> str:
    """Optional modal card over the middle of the screen (occlusion unknowns)."""
    if rng.random() >= p:
        return ""
    title, msg, b1, b2 = rng.choice([("Enjoying the app?", "Tell us what you think in a quick rating.", "Not now", "Rate"),
                                     ("Turn on backups?", "Keep your data safe if you lose your phone.", "Later", "Turn on"),
                                     ("New features", "Swipe left on a row to see more actions.", "Skip", "Got it"),
                                     ("Update available", "Version 5.2 is ready to install.", "Later", "Update")])
    s.dialog = "dlg"
    s.meta["dialog_title"] = title
    a1 = s.el("dlg_b1", f"the {b1} button in the dialog", b1, ["dismiss the dialog"], "closes the dialog")
    a2 = s.el("dlg_b2", f"the {b2} button in the dialog", b2, [], None)
    return (f'<div class="scrim"><div class="dlg" {s.el("dlg", None, role="dialog")}><h3>{E(title)}</h3>'
            f'<div class="muted" style="font-size:14.5px;margin-bottom:18px">{E(msg)}</div>'
            f'<div style="display:flex;gap:10px;justify-content:flex-end"><div class="btn" {a1}>{E(b1)}</div>'
            f'<div class="btn pri" {a2}>{E(b2)}</div></div></div></div>')


# ------------------------------------------------------------------------------------------------ settings
SETTINGS_ROWS = [
    ("Wi-Fi", "chevron", "wifi", "choose a Wi-Fi network"),
    ("Bluetooth", "toggle", "bt", None),
    ("Airplane mode", "toggle", "plane", None),
    ("Mobile data", "toggle", "chart", None),
    ("Notifications", "chevron", "bell", "choose which apps can send notifications"),
    ("Display & brightness", "chevron", "moon", "change the screen brightness"),
    ("Sounds & vibration", "chevron", "volume", "change the ringtone"),
    ("Battery", "chevron", "torch", "see which apps use the most battery"),
    ("Privacy", "chevron", "lock", "review which apps can use the camera"),
    ("Do not disturb", "toggle", "moon", None),
    ("Dark mode", "toggle", "eye", None),
    ("Auto-rotate", "toggle", "refresh", None),
    ("Location services", "toggle", "pin", None),
    ("Storage", "chevron", "folder", "see how much storage is free"),
    ("About phone", "chevron", "help", "find the phone's model number"),
    ("Accounts", "chevron", "user", "add another email account"),
    ("Wallpaper", "chevron", "grid", "change the home screen picture"),
]


def tname(lab: str) -> str:
    return lab if lab in ("Bluetooth", "Wi-Fi") else lab.lower()


def settings(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "settings", W, H, DPR)
    s.screen_desc = "a phone settings screen"
    rows = rng.sample(SETTINGS_ROWS, rng.randint(8, 11))
    managed = rng.random() < 0.3
    toggles = [r for r in rows if r[1] == "toggle"]
    man_row = rng.choice(toggles)[0] if managed and toggles else None
    back = s.el("back", "the back arrow", None, ["go back to the previous screen"], "goes back to the previous screen", small=True)
    srch = s.el("search", "the search icon", None, ["search the settings"], "opens another screen", small=True)
    head = (f'<div class="hd"><div class="ib" {back}>{svg("back", 22)}</div><h2>Settings</h2>'
            f'<div class="ib" {srch}>{svg("search", 22)}</div></div>')
    blocks, cur = [], []
    split = rng.randint(3, len(rows) - 3)
    state = {}
    for i, (lab, kind, ic, goal) in enumerate(rows):
        k = f"r{i}"
        icon = f'<div style="width:30px;height:30px;border-radius:8px;background:{t[4]};color:#fff;display:flex;align-items:center;justify-content:center">{svg(ic, 18)}</div>'
        if kind == "toggle":
            on = rng.random() < 0.5
            dis = lab == man_row
            state[lab] = on
            g = f"turn {'off' if on else 'on'} {tname(lab)}"
            act = "nothing, it is disabled" if dis else ("turns a setting off" if on else "turns a setting on")
            a_lab = s.el(k, f"the {lab} row", lab, [g], act, disabled=dis, group="rows")
            a_tg = s.el(k + "_tg", f"the {lab} switch", None, [g], act, disabled=dis, small=True)
            if not dis:
                s.possible.append(k)
            else:
                s.possible.append(k)
            sub = '<div class="sub">Managed by your organisation</div>' if dis else ""
            cur.append(f'<div class="row{" dis" if dis else ""}">{icon}<div class="lb"><span {a_lab}>{E(lab)}</span>{sub}</div>'
                       f'<div class="tg{" on" if on else ""}" {a_tg}><i></i></div></div>')
        else:
            val = {"Wi-Fi": rng.choice(["Home-5G", "Vel-Guest", "Office", "Off"]), "Accounts": f"{rng.randint(1, 3)}"}.get(lab, "")
            a_lab = s.el(k, f"the {lab} row", lab, [goal], "opens another screen", group="rows")
            cur.append(f'<div class="row">{icon}<div class="lb" {a_lab}>{E(lab)}</div><span class="val">{E(val)}</span>'
                       f'<span class="muted">{svg("chev_r", 16)}</span></div>')
        if i + 1 == split:
            blocks.append(cur)
            cur = []
    blocks.append(cur)
    body = head + "".join(f'<div class="card">{"".join(b)}</div>' for b in blocks)
    for lab, kind, ic, goal in SETTINGS_ROWS:
        if lab not in [r[0] for r in rows]:
            s.add_absent(f"the {lab} row", goal or f"turn {rng.choice(['on', 'off'])} {tname(lab)}")
    s.meta.update({"toggles": state, "managed": man_row})
    s.html = chrome(rng, t, s, body, overlay=maybe_dialog(rng, s, t))
    return s


# ------------------------------------------------------------------------------------------------ chat
def chat(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "chat", W, H, DPR)
    s.screen_desc = "a phone messaging conversation"
    f = rng.choice(FIRST)
    l = rng.choice(LAST)
    typed = rng.random() < 0.5
    kb_up = rng.random() < 0.5
    back = s.el("back", "the back arrow", None, ["go back to the chat list"], "goes back to the previous screen", small=True)
    call = s.el("call", "the phone icon", None, [f"call {f} with voice only"], "starts a call", small=True, group="head")
    vid = s.el("video", "the video camera icon", None, [f"start a video call with {f}"], "starts a call", small=True, group="head")
    col = rng.choice(["#e07a5f", "#3d8bfd", "#8e6cd1", "#2a9d8f", "#d4a017"])
    head = (f'<div class="hd" style="border-bottom:1px solid {t[5]}"><div class="ib" {back}>{svg("back", 22)}</div>'
            f'<div class="av" style="background:{col}">{f[0]}{l[0]}</div><h2>{E(f)} {E(l)}</h2>'
            f'<div class="ib" {call}>{svg("phone", 21)}</div><div class="ib" {vid}>{svg("video", 22)}</div></div>')
    lines = ["Are we still on for Saturday?", "Yes! 10 am at the market?", "Perfect. Bring the blue bag.", "Did you get my photos?",
             "Running 5 min late", "No worries", "Can you send the address again?", "It's 14 Fenn Road, flat 2", "See you soon",
             "Thanks for yesterday!", "The train is delayed again", "Want me to pick up bread?"]
    bubbles = []
    for i, ln in enumerate(rng.sample(lines, rng.randint(4, 7))):
        me = rng.random() < 0.5
        st = (f"margin-left:auto;background:{t[4]};color:{t[6]}" if me else f"background:{t[1]}")
        bubbles.append(f'<div style="max-width:72%;padding:9px 13px;border-radius:18px;margin:6px 14px;{st};font-size:15px;width:fit-content">{E(ln)}</div>')
    msg = rng.choice(["On my way", "Sounds good, see you then", "I'll bring snacks", "Can we make it 11?"]) if typed else ""
    att = s.el("attach", "the plus (attach) button", None, ["attach a photo or file"], "opens another screen", small=True)
    fld = s.el("field", "the message text field", None, ["type a message"], None)
    emo = s.el("emoji", "the emoji icon", None, ["insert an emoji"], "opens another screen", small=True)
    snd = s.el("send", "the send button", None, ["send the message"], "sends the message" if typed else "nothing, it is disabled",
               disabled=not typed, small=True)
    s.possible.append("send")
    txt = f'<span>{E(msg)}</span>' if typed else '<span class="muted">Message</span>'
    bar = (f'<div style="display:flex;align-items:center;gap:6px;padding:8px 10px;border-top:1px solid {t[5]};background:{t[1]};position:absolute;left:0;right:0;bottom:0">'
           f'<div class="ib" {att}>{svg("plus", 24)}</div><div class="fld" style="flex:1" {fld}>{txt}</div>'
           f'<div class="ib" {emo}>{svg("smile", 22)}</div>'
           f'<div class="ib{"" if typed else " dim"}" {snd} style="background:{t[4] if typed else "transparent"};color:{t[6] if typed else t[3]}">{svg("send", 20)}</div></div>')
    kb = ""
    if kb_up:
        rows_ = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        krs = []
        for ri, r in enumerate(rows_):
            keys = []
            if ri == 2:
                keys.append(f'<div class="key w" {s.el("k_shift", "the shift key", None, [], None, small=True)}>⇧</div>')
            for ch in r:
                keys.append(f'<div class="key" {s.el("k_" + ch, f"the {ch.upper()} key", None, [f"type the letter {ch}"], "types a letter", small=True)}>{ch}</div>')
            if ri == 2:
                keys.append(f'<div class="key w" {s.el("k_del", "the delete (backspace) key", None, ["delete the last letter"], None, small=True)}>⌫</div>')
            krs.append(f'<div class="kr">{"".join(keys)}</div>')
        krs.append(f'<div class="kr"><div class="key w" {s.el("k_123", "the 123 key", None, [], None, small=True)}>123</div>'
                   f'<div class="key sp" {s.el("k_space", "the space bar", None, ["type a space"], "types a letter")}>space</div>'
                   f'<div class="key w" {s.el("k_ret", "the return key", None, [], None, small=True)}>return</div></div>')
        kb = f'<div class="kb">{"".join(krs)}</div>'
    body = head + f'<div style="padding-top:6px">{"".join(bubbles)}</div>' + bar
    for d, g in [("the block contact button", f"block {f}"), ("the delete conversation button", "delete this conversation"),
                 ("the search in conversation icon", "search within this conversation"), ("the mute button", "mute this chat")]:
        s.add_absent(d, g)
    s.meta.update({"typed": typed, "keyboard": kb_up, "contact": f"{f} {l}"})
    s.html = chrome(rng, t, s, body, kb=kb, overlay="" if kb_up else maybe_dialog(rng, s, t, 0.1))
    return s


# ------------------------------------------------------------------------------------------------ chat list + bottom nav
def chat_list(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "chat_list", W, H, DPR)
    s.screen_desc = "a phone chat list with a bottom navigation bar"
    app = rng.choice(APPS)
    srch = s.el("search", "the search icon", None, ["search your chats"], "opens another screen", small=True, group="head")
    cam = s.el("camera", "the camera icon", None, ["take a photo to send"], "opens another screen", small=True, group="head")
    head = (f'<div class="hd"><h1>{E(app)}</h1><div class="ib" {cam}>{svg("camera", 22)}</div>'
            f'<div class="ib" {srch}>{svg("search", 22)}</div></div>')
    chips = []
    for i, c in enumerate(rng.sample(["All", "Unread", "Groups", "Favourites", "Work"], 4)):
        sel = i == 0
        a = "" if sel else s.el(f"chip_{i}", f"the {c} filter", c, [f"show only {c.lower()} chats" if c != "Favourites" else "show only favourite chats"],
                                "switches to another tab", group="chips")
        chips.append(f'<span class="chip{" sel" if sel else ""}" {a}>{E(c)}</span>')
    rows = []
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], 7)
    for i, n in enumerate(names):
        col = rng.choice(["#e07a5f", "#3d8bfd", "#8e6cd1", "#2a9d8f", "#d4a017", "#c05780"])
        unread = rng.random() < 0.35
        prev = rng.choice(["See you tomorrow!", "Photo", "Voice message (0:14)", "Thanks, got it", "Where are you?",
                           "The meeting moved to 3", "Happy birthday!!", "Can you call me?"])
        tm = f"{rng.randint(7, 22)}:{rng.randint(0, 59):02d}"
        a = s.el(f"c{i}", f"the chat with {n}", n, [f"open the chat with {n}"], "opens another screen")
        badge = f'<span class="badge">{rng.randint(1, 9)}</span>' if unread else ""
        rows.append(f'<div class="row" {a}><div class="av" style="background:{col}">{n.split()[0][0]}{n.split()[1][0]}</div>'
                    f'<div class="lb"><div style="font-weight:600">{E(n)}</div><div class="sub">{E(prev)}</div></div>'
                    f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px"><span class="sub">{tm}</span>{badge}</div></div>')
    fab = s.el("compose", "the new chat (pencil) button", None, ["start a new chat"], "opens another screen")
    body = (head + f'<div style="display:flex;gap:8px;padding:2px 14px 8px">{"".join(chips)}</div>'
            f'<div class="card" style="margin-top:4px">{"".join(rows)}</div>'
            f'<div {fab} style="position:absolute;right:18px;bottom:18px;width:56px;height:56px;border-radius:16px;background:{t[4]};color:{t[6]};display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px #0004">{svg("pencil", 24)}</div>')
    tabs = [("Chats", "mail", "see your chats"), ("Calls", "phone", "see your recent calls"),
            ("Updates", "star", "see status updates"), ("Settings", "gear", "open the app's settings")]
    nav = bottom_nav(s, tabs, 0, t)
    s.add_absent("the archive button", "archive the selected chat")
    s.add_absent("the Contacts tab", "open your contacts list")
    s.add_absent("the microphone icon", "record a voice message")
    s.html = chrome(rng, t, s, body, nav=nav, overlay=maybe_dialog(rng, s, t))
    return s


# ------------------------------------------------------------------------------------------------ map-like place list
def places(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "places", W, H, DPR)
    s.screen_desc = "a phone maps app showing nearby places"
    fld = s.el("search", "the Search here field", None, ["search for a place"], "opens another screen")
    prof = s.el("profile", "the profile picture", None, ["open your account"], "opens another screen", small=True)
    mp = (f'<div style="height:250px;position:relative;background:{"#20303a" if t[9] else "#e4eadf"};overflow:hidden">'
          + "".join(f'<div style="position:absolute;left:{rng.randint(-40, 360)}px;top:{rng.randint(-20, 240)}px;width:{rng.randint(120, 420)}px;height:{rng.randint(6, 14)}px;'
                    f'background:{"#3b4d58" if t[9] else "#fff"};transform:rotate({rng.randint(-60, 60)}deg)"></div>' for _ in range(7))
          + "".join(f'<div style="position:absolute;left:{rng.randint(20, 340)}px;top:{rng.randint(60, 220)}px;color:#e0342b">{svg("pin", 26)}</div>' for _ in range(4))
          + f'<div style="position:absolute;left:12px;right:12px;top:10px;display:flex;gap:8px"><div class="fld" style="flex:1;background:{t[1]};box-shadow:0 2px 8px #0003" {fld}>'
          f'{svg("search", 18)}<span class="muted">Search here</span></div><div class="av" {prof} style="background:#8e6cd1">{rng.choice(FIRST)[0]}</div></div>')
    loc = s.el("myloc", "the my-location button", None, ["centre the map on your location"], None, small=True)
    mp += (f'<div {loc} style="position:absolute;right:12px;bottom:12px;width:44px;height:44px;border-radius:22px;background:{t[1]};'
           f'display:flex;align-items:center;justify-content:center;color:{t[4]};box-shadow:0 2px 6px #0004">{svg("pin", 22)}</div></div>')
    cats = rng.sample([("Restaurants", "restaurants"), ("Coffee", "coffee places"), ("Petrol", "petrol stations"),
                       ("Hotels", "hotels"), ("Pharmacies", "pharmacies"), ("Groceries", "grocery shops")], 4)
    chips = "".join(f'<span class="chip" {s.el(f"cat_{i}", f"the {c} chip", c, [f"show only {g} on the map"], None, group="chips")}>{E(c)}</span>'
                    for i, (c, g) in enumerate(cats))
    cards = []
    for i, p in enumerate(rng.sample(PLACES, 2)):
        rating = f"{rng.uniform(3.5, 4.9):.1f}"
        dist = f"{rng.uniform(0.2, 4.5):.1f} km"
        acts = [("Directions", "pin", f"get directions to {p}"), ("Call", "phone", f"phone {p}"), ("Save", "star", f"save {p} to a list"),
                ("Share", "share", f"send {p} to a friend")]
        btns = "".join(f'<span class="btn sm{" pri" if j == 0 else ""}" {s.el(f"p{i}_{a[0].lower()}", f"the {a[0]} button for {p}", a[0], [a[2]], "opens another screen" if a[0] != "Call" else "starts a call")}>'
                       f'{svg(a[1], 15)}{a[0]}</span>' for j, a in enumerate(acts))
        name = s.el(f"p{i}", f"the name {p}", p, [f"see details for {p}"], "opens another screen")
        cards.append(f'<div class="card" style="padding:12px 14px"><div style="font-weight:700;font-size:17px" {name}>{E(p)}</div>'
                     f'<div class="sub" style="margin:3px 0 10px">★ {rating} · {dist} · Open until {rng.randint(17, 23)}:00</div>'
                     f'<div style="display:flex;gap:7px;flex-wrap:wrap">{btns}</div></div>')
    body = mp + f'<div style="display:flex;gap:8px;padding:10px 12px;overflow:hidden">{chips}</div>' + "".join(cards)
    tabs = [("Explore", "pin", "explore the map"), ("Saved", "star", "see your saved places"), ("Contribute", "plus", "add a review or photo"),
            ("Updates", "bell", "see updates from places you follow")]
    nav = bottom_nav(s, tabs, 0, t)
    s.add_absent("the traffic layer button", "show live traffic on the map")
    s.add_absent("the Book a table button", "book a table for tonight")
    s.add_absent("the Order online button", "order food for delivery")
    s.html = chrome(rng, t, s, body, nav=nav, overlay=maybe_dialog(rng, s, t))
    return s


# ------------------------------------------------------------------------------------------------ checkout
ITEMS = ["Oat latte", "Rye loaf", "Linen tea towel", "Ceramic mug", "Wool socks", "Hand cream", "Beeswax candle",
         "Travel mug", "Notebook", "Cork coasters"]


def checkout(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "checkout", W, H, DPR)
    s.screen_desc = "a phone shopping checkout screen"
    back = s.el("back", "the back arrow", None, ["go back to the basket"], "goes back to the previous screen", small=True)
    head = f'<div class="hd"><div class="ib" {back}>{svg("back", 22)}</div><h2>Checkout</h2></div>'
    items = rng.sample(ITEMS, rng.randint(2, 3))
    rows, total = [], 0.0
    for i, it in enumerate(items):
        pr = round(rng.uniform(2.5, 24), 2)
        q = rng.randint(1, 3)
        total += pr * q
        mn = s.el(f"i{i}_minus", f"the minus button for {it}", None, [f"remove one {it.lower()}"], "removes one item", small=True)
        pl = s.el(f"i{i}_plus", f"the plus button for {it}", None, [f"add one more {it.lower()}"], "adds one more item", small=True)
        rows.append(f'<div class="row"><div class="lb"><div style="font-weight:600">{E(it)}</div><div class="sub">£{pr:.2f} each</div></div>'
                    f'<div style="display:flex;align-items:center;gap:10px"><div class="ib" style="width:30px;height:30px;border:1px solid {t[5]}" {mn}>{svg("minus", 16)}</div>'
                    f'<b>{q}</b><div class="ib" style="width:30px;height:30px;border:1px solid {t[5]}" {pl}>{svg("plus", 16)}</div></div></div>')
    chg = s.el("addr_change", "the Change link for the delivery address", "Change", ["change the delivery address"], "opens another screen", group="links")
    f_, l_ = rng.choice(FIRST), rng.choice(LAST)
    addr = (f'<div class="card" style="padding:12px 14px"><div style="display:flex;justify-content:space-between"><b>Deliver to</b>'
            f'<span style="color:{t[4]};font-weight:600" {chg}>Change</span></div><div class="sub" style="margin-top:4px">{E(f_)} {E(l_)}, '
            f'{rng.randint(2, 90)} {rng.choice(["Fenn Road", "Marro Lane", "Quor Street", "Tovi Close"])}</div></div>')
    card_filled = rng.random() < 0.6
    promo_typed = rng.random() < 0.4
    terms = rng.random() < 0.7
    cardno = f"•••• •••• •••• {rng.randint(1000, 9999)}" if card_filled else ""
    cf = s.el("card_field", "the card number field", None, ["enter your card number"], None)
    pf = s.el("promo_field", "the promo code field", None, ["type a promo code"], None)
    ap = s.el("promo_apply", "the Apply button", "Apply", ["apply the promo code"], "applies the promo code" if promo_typed else "nothing, it is disabled",
              disabled=not promo_typed, group="links")
    tc = s.el("terms", "the terms checkbox", None, ["accept the terms"], None, small=True)
    pay_ok = card_filled and terms
    total = round(total + 3.5, 2)
    pay = s.el("pay", "the Pay button", f"Pay £{total:.2f}", ["place the order"], "places the order" if pay_ok else "nothing, it is disabled",
               disabled=not pay_ok)
    s.possible += ["pay", "promo_apply"]
    pay_form = (f'<div class="card" style="padding:12px 14px"><b>Payment</b><div class="sub" style="margin:8px 0 4px">Card number <span style="color:{t[7]}">*</span></div>'
                f'<div class="fld" {cf}>{E(cardno) if cardno else "<span class=muted>Enter card number</span>"}</div>'
                f'<div class="sub" style="margin:10px 0 4px">Promo code</div><div style="display:flex;gap:8px"><div class="fld" style="flex:1" {pf}>'
                f'{E(rng.choice(["SPRING10", "WELCOME5", "FREEDEL"])) if promo_typed else "<span class=muted>Enter code</span>"}</div>'
                f'<span class="btn{"" if promo_typed else " dis"}" {ap}>Apply</span></div>'
                f'<label style="display:flex;gap:10px;align-items:center;margin-top:12px;font-size:14px"><span {tc} style="width:22px;height:22px;border-radius:5px;'
                f'border:2px solid {t[4] if terms else t[3]};background:{t[4] if terms else "transparent"};color:{t[6]};display:flex;align-items:center;justify-content:center;font-size:14px">{"✓" if terms else ""}</span>'
                f'I accept the terms of sale <span style="color:{t[7]}">*</span></label></div>')
    body = (head + f'<div class="card">{"".join(rows)}</div>' + addr + pay_form +
            f'<div style="padding:10px 14px"><div class="btn pri{"" if pay_ok else " dis"}" style="width:100%;padding:14px" {pay}>Pay £{total:.2f}</div>'
            f'<div class="sub" style="text-align:center;margin-top:6px">Fields marked * are required</div></div>')
    s.add_absent("the gift wrap option", "add gift wrapping")
    s.add_absent("the Pay later button", "pay in three instalments")
    s.meta.update({"card_filled": card_filled, "promo_typed": promo_typed, "terms": terms, "pay_enabled": pay_ok})
    s.html = chrome(rng, t, s, body)
    return s


# ------------------------------------------------------------------------------------------------ permission dialog
PERMS = ["Camera", "Microphone", "Location", "Contacts", "Photos and videos", "Calendar", "Notifications", "Call logs",
         "SMS messages", "Nearby devices"]
PERM_ICON = {"Camera": "camera", "Microphone": "mic", "Location": "pin", "Contacts": "user", "Photos and videos": "grid",
             "Calendar": "calendar", "Notifications": "bell", "Call logs": "phone", "SMS messages": "mail", "Nearby devices": "bt"}
TASKS = [("Scan a paper receipt to split a bill", {"Camera"}), ("Record a voice memo", {"Microphone"}),
         ("Find cafés near where you are now", {"Location"}), ("Invite friends from your address book", {"Contacts"}),
         ("Add a photo from your gallery to a post", {"Photos and videos"}), ("Scan the QR code on a table", {"Camera"}),
         ("Get walking directions from here", {"Location"}), ("Get a reminder alert before a meeting", {"Notifications"}),
         ("Make a video call", {"Camera", "Microphone"}), ("Add a dentist appointment to your calendar", {"Calendar"}),
         ("Connect to your wireless earbuds", {"Nearby devices"}), ("Send a voice message", {"Microphone"}),
         ("Take a profile picture", {"Camera"}), ("Tag your location on a photo", {"Location"})]


def permission(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "permission", W, H, DPR)
    s.screen_desc = "a phone app asking for permissions"
    app = rng.choice(APPS)
    task, need = rng.choice(TASKS)
    extra = rng.random() < 0.55
    req = list(need)
    if extra:
        req.append(rng.choice([p for p in PERMS if p not in need]))
    rng.shuffle(req)
    bg = "".join(f'<div class="row"><div class="av" style="background:#999">{c}</div><div class="lb">Item {c}{rng.randint(10, 99)}</div></div>'
                 for c in "ABCDEFG")
    items = "".join(f'<div style="display:flex;gap:10px;align-items:center;padding:6px 0" {s.el("perm_" + str(i), f"the {p} permission line", p, [], None)}>'
                    f'<span style="color:{t[4]}">{svg(PERM_ICON[p], 20)}</span><span>{E(p)}</span></div>' for i, p in enumerate(req))
    allow = s.el("allow", "the Allow button", "Allow", ["grant the requested access"], "closes the dialog", group="dlg")
    deny = s.el("deny", "the Don't allow button", "Don't allow", ["refuse the requested access"], "closes the dialog", group="dlg")
    once = ""
    if rng.random() < 0.5:
        once = f'<div class="btn" style="width:100%;margin-top:8px" {s.el("once", "the Only this time button", "Only this time", ["allow access just for now"], "closes the dialog", group="dlg")}>Only this time</div>'
    dlg = (f'<div class="scrim"><div class="dlg" {s.el("dlg", None, role="dialog")}><h3>Allow {E(app)} to access:</h3>{items}'
           f'<div style="margin-top:14px"><div class="btn pri" style="width:100%" {allow}>Allow</div>{once}'
           f'<div class="btn" style="width:100%;margin-top:8px" {deny}>Don\'t allow</div></div></div></div>')
    s.meta.update({"app": app, "task": task, "needed": sorted(need), "requested": req, "extra": sorted(set(req) - need)})
    s.html = chrome(rng, t, s, f'<div class="hd"><h1>{E(app)}</h1></div><div class="card">{bg}</div>', overlay=dlg)
    return s


# ------------------------------------------------------------------------------------------------ notification shade
def shade(rng) -> Screen:
    t = rng.choice([th for th in THEMES if th[9]] + [THEMES[0]])
    s = Screen("mobile", "shade", W, H, DPR)
    s.screen_desc = "a phone notification shade with quick settings"
    tiles = rng.sample([("Wi-Fi", "wifi"), ("Bluetooth", "bt"), ("Torch", "torch"), ("Do not disturb", "moon"),
                        ("Airplane mode", "plane"), ("Auto-rotate", "refresh"), ("Location", "pin"), ("Dark mode", "eye")], 6)
    tl = []
    state = {}
    for i, (lab, ic) in enumerate(tiles):
        on = rng.random() < 0.5
        state[lab] = on
        nm = lab if lab in ("Wi-Fi", "Bluetooth") else lab.lower()
        g = f"turn {'off' if on else 'on'} {'the torch' if lab == 'Torch' else nm}"
        a = s.el(f"t{i}", f"the {lab} tile", lab, [g], "turns a setting off" if on else "turns a setting on", group="tiles")
        s.possible.append(f"t{i}")
        tl.append(f'<div {a} style="width:84px;display:flex;flex-direction:column;align-items:center;gap:6px;font-size:12px">'
                  f'<div style="width:56px;height:56px;border-radius:28px;display:flex;align-items:center;justify-content:center;'
                  f'background:{t[4] if on else t[5]};color:{t[6] if on else t[2]}">{svg(ic, 24)}</div><span style="text-align:center">{E(lab)}</span></div>')
    notes = []
    senders = rng.sample(FIRST, 3)
    for i, who in enumerate(senders):
        app = rng.choice(APPS)
        kind = rng.choice(["msg", "mail", "remind"])
        if kind == "msg":
            text, acts = rng.choice(["Are you coming tonight?", "Sent a photo", "Call me when free"]), [("Reply", f"reply to {who}"), ("Mark as read", f"mark {who}'s message as read")]
        elif kind == "mail":
            text, acts = rng.choice(["Invoice for September", "Your order has shipped", "Meeting notes"]), [("Archive", f"archive the email from {who}"), ("Reply", f"reply to {who}")]
        else:
            text, acts = rng.choice(["Water the plants", "Dentist at 4 pm", "Pay the rent"]), [("Snooze", "be reminded again later"), ("Done", "mark the reminder as done")]
        btns = "".join(f'<span style="color:{t[4]};font-weight:600;font-size:14px;margin-right:18px" '
                       f'{s.el(f"n{i}_{j}", f"the {a} button on the notification from {who}", a, [g], None)}>{E(a)}</span>'
                       for j, (a, g) in enumerate(acts))
        notes.append(f'<div class="card" style="padding:12px 14px"><div class="sub">{E(app)} · {rng.randint(1, 59)} min</div>'
                     f'<div style="font-weight:600;margin:3px 0">{E(who)}</div><div style="font-size:14.5px;margin-bottom:8px">{E(text)}</div>{btns}</div>')
    clr = s.el("clear", "the Clear all button", "Clear all", ["dismiss all notifications"], None)
    gear = s.el("gear", "the settings gear icon", None, ["open the phone's settings"], "opens another screen", small=True)
    bright = s.el("bright", "the brightness slider", None, ["change the screen brightness"], None)
    body = (f'<div style="padding:4px 14px;display:flex;justify-content:space-between;align-items:center"><b style="font-size:20px">'
            f'{rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])} {rng.randint(1, 28)} {rng.choice(["Mar", "Jun", "Sep", "Nov"])}</b>'
            f'<div class="ib" {gear}>{svg("gear", 22)}</div></div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:10px 8px;justify-content:center;padding:8px 10px">{"".join(tl)}</div>'
            f'<div style="margin:6px 18px 12px;height:30px;border-radius:15px;background:{t[5]};position:relative" {bright}>'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:{rng.randint(20, 90)}%;border-radius:15px;background:{t[4]}"></div></div>'
            + "".join(notes) + f'<div style="text-align:center;margin-top:10px"><span class="btn sm" {clr}>Clear all</span></div>')
    s.add_absent("the Hotspot tile", "turn on the mobile hotspot")
    s.add_absent("the Screen record tile", "record the screen")
    s.meta.update({"tiles": state})
    s.html = chrome(rng, t, s, body)
    return s


# ------------------------------------------------------------------------------------------------ music player
def player(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("mobile", "player", W, H, DPR)
    s.screen_desc = "a phone music player"
    playing = rng.random() < 0.5
    song = rng.choice(["Glass Harbour", "Paper Lanterns", "Slow Tide", "North Window", "Copper Sky", "Late Tram"])
    artist = rng.choice(["The Vellums", "Marra Quinn", "Oda & Fenn", "Kiro Lights", "Tovi Sands"])
    down = s.el("down", "the collapse (down arrow) button", None, ["close the full-screen player"], None, small=True)
    more = s.el("more", "the three-dot menu", None, [], None, small=True)
    col = rng.choice(["#e07a5f", "#3d8bfd", "#8e6cd1", "#2a9d8f", "#d4a017"])
    like = s.el("like", "the heart (like) button", None, ["add the song to your liked songs"], None, small=True, group="ctl")
    shuf = s.el("shuffle", "the shuffle button", None, ["play the queue in random order"], None, small=True, group="ctl")
    prev = s.el("prev", "the previous-track button", None, ["go back to the previous song"], None, small=True, group="ctl")
    pp = s.el("pp", "the pause button" if playing else "the play button", None, ["pause the music" if playing else "start the music"],
              "pauses playback" if playing else "starts playback", group="ctl")
    nxt = s.el("next", "the next-track button", None, ["skip to the next song"], None, small=True, group="ctl")
    rep = s.el("repeat", "the repeat button", None, ["repeat the current song"], None, small=True, group="ctl")
    q = s.el("queue", "the queue (list) icon", None, ["see what plays next"], "opens another screen", small=True)
    shr = s.el("share", "the share icon", None, ["share the song"], "opens another screen", small=True)
    body = (f'<div class="hd"><div class="ib" {down}>{svg("chev_d", 24)}</div><h2 style="text-align:center;font-size:14px" class="muted">PLAYING FROM ALBUM</h2>'
            f'<div class="ib" {more}>{svg("dots", 22)}</div></div>'
            f'<div style="margin:18px 34px;height:322px;border-radius:14px;background:linear-gradient(135deg,{col},#222)"></div>'
            f'<div style="display:flex;align-items:center;padding:0 30px"><div style="flex:1"><div style="font-size:22px;font-weight:700">{E(song)}</div>'
            f'<div class="muted">{E(artist)}</div></div><div class="ib" {like}>{svg("heart", 24)}</div></div>'
            f'<div style="margin:18px 30px 4px;height:4px;border-radius:2px;background:{t[5]}"><div style="width:{rng.randint(5, 90)}%;height:4px;background:{t[2]}"></div></div>'
            f'<div style="display:flex;justify-content:space-between;padding:0 30px" class="sub"><span>{rng.randint(0, 2)}:{rng.randint(10, 59)}</span><span>{rng.randint(3, 5)}:{rng.randint(10, 59)}</span></div>'
            f'<div style="display:flex;align-items:center;justify-content:space-between;padding:14px 26px">'
            f'<div class="ib" {shuf}>{svg("shuffle", 22)}</div><div class="ib" {prev} style="color:{t[2]}">{svg("prev", 30)}</div>'
            f'<div {pp} style="width:68px;height:68px;border-radius:34px;background:{t[2]};color:{t[0]};display:flex;align-items:center;justify-content:center">{svg("pause" if playing else "play", 30)}</div>'
            f'<div class="ib" {nxt} style="color:{t[2]}">{svg("next", 30)}</div><div class="ib" {rep}>{svg("loop", 22)}</div></div>'
            f'<div style="display:flex;justify-content:space-between;padding:10px 34px"><div class="ib" {shr}>{svg("share", 20)}</div><div class="ib" {q}>{svg("menu", 22)}</div></div>')
    s.add_absent("the lyrics button", "show the song lyrics")
    s.add_absent("the equaliser button", "adjust the bass and treble")
    s.add_absent("the sleep timer icon", "stop the music after 30 minutes")
    s.meta.update({"playing": playing})
    s.html = chrome(rng, t, s, body, overlay=maybe_dialog(rng, s, t, 0.1))
    return s


TEMPLATES = {"settings": (settings, 20), "chat": (chat, 15), "chat_list": (chat_list, 12), "places": (places, 13),
             "checkout": (checkout, 14), "permission": (permission, 12), "shade": (shade, 8), "player": (player, 8)}
