"""Phase-3 Stage 0-I: screen grounding / location items (families screen_locate, mobile_*, pro_*; source I; constructed gold).

We render desktop, phone and professional-app screens ourselves (HTML -> PNG, local headless Chromium through
scripts/p3/screen_templates/render.mjs), measure every labelled element's box, then ask:

  screen_locate      desktop / web: "in which part of the screen is X?" (3x3 named regions, 4x4 row/column, or a drawn
                     grid with labelled cells A1..D4) and "which marked box (A-F drawn on the screenshot) is X?"
  mobile_locate      the same on phone screens (390x844 viewport rendered at 1.5x)
  mobile_tap         which visible label / which marked box to tap for goal G
  mobile_possible    can you do G right now (disabled buttons, required fields, managed settings) (noul)
  mobile_next        what happens if you tap X (action classes: opens another screen, turns a setting on, nothing -
                     it is disabled ...)
  mobile_permission  a permission dialog + the user's task in the state: does it ask for more than the task needs
                     (noul); which requested permission does the task not need (choice; none -> false_premise)
  pro_locate         grid questions on dense pro screens (IDE, spreadsheet, image / video editor, CAD, DAW)
  pro_ground         which marked box is X / would you click to do G (small toolbar icons, track buttons, cells)
  pro_menu           which item of the open menu does G; can you do G from this menu right now (disabled items)

Gold comes from the measured boxes: a grid item is kept only when the element's whole box lies inside one cell
(elements straddling a border are rejected; drawn cell badges never touch the target); marked boxes never overlap and
their letter badges touch no other mark. Unknown variants (~15%): the named control is not on this screen (not_listed),
or it is covered by a modal dialog (insufficient_evidence), or no marked box / listed option does it (not_listed).
provenance keeps the element boxes (image px), the grid / marks spec and the screen seed, so the tests re-derive every
gold (tests/test_p3_geometry_screens.py). PNGs <= 1280 px, licence "generated", invented brands, people and places.

    .venv/bin/python scripts/p3/gen_screens.py [--seed ...] [--count 5000] [--out data/p3/candidates/I-screens.jsonl]
    .venv/bin/python scripts/p3/gen_screens.py --heldout [--count 330]
    .venv/bin/python scripts/p3/gen_screens.py --variant-of rows.jsonl --variants-per 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
sys.path.insert(0, str(ROOT / "scripts" / "p3" / "screen_templates"))
from candidate import read, write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_image_joint import bench_image_shas, rng_for  # noqa: E402
import desktop  # noqa: E402
import mobile  # noqa: E402
import pro  # noqa: E402
from base import ACTIONS  # noqa: E402
from render_common import render  # noqa: E402

GENERATOR = "scripts/p3/gen_screens.py"
IMG_DIR = ROOT / "data" / "p3" / "images" / "screens"
OUT = ROOT / "data" / "p3" / "candidates" / "I-screens.jsonl"
HELDOUT_OUT = ROOT / "data" / "p3" / "pool" / "heldout-fresh-screens.jsonl"
PLATFORMS = {"desktop": (desktop.TEMPLATES, 0.28), "mobile": (mobile.TEMPLATES, 0.42), "pro": (pro.TEMPLATES, 0.30)}
NAMES3 = [["top left", "top centre", "top right"], ["middle left", "centre", "middle right"], ["bottom left", "bottom centre", "bottom right"]]
MARK_COLOURS = ["#e6007e", "#ff4f00", "#d10000", "#7a00e6"]
LETTERS = "ABCDEF"
P_UNKNOWN = 0.22
CONFLICT = [{"opens another screen", "switches to another tab", "starts a call", "places the order", "goes back to the previous screen",
             "closes the dialog", "applies the promo code", "sends the message"},
]
COUNTERPART = {"turns a setting on": "turns a setting off", "turns a setting off": "turns a setting on", "starts playback": "pauses playback",
               "pauses playback": "starts playback", "adds one more item": "removes one item", "removes one item": "adds one more item"}
FAMILY = {("desktop", "locate"): "screen_locate", ("desktop", "ground"): "screen_locate",
          ("mobile", "locate"): "mobile_locate", ("mobile", "ground"): "mobile_locate", ("mobile", "tap_box"): "mobile_tap",
          ("mobile", "tap_label"): "mobile_tap", ("mobile", "possible"): "mobile_possible", ("mobile", "next"): "mobile_next",
          ("mobile", "perm_more"): "mobile_permission", ("mobile", "perm_which"): "mobile_permission",
          ("pro", "locate"): "pro_locate", ("pro", "ground"): "pro_ground", ("pro", "ground_goal"): "pro_ground",
          ("pro", "menu"): "pro_menu", ("pro", "tap_label"): "pro_ground", ("pro", "menu_possible"): "pro_menu"}
PLATFORM_WORD = {"desktop": "on a computer", "mobile": "on a phone", "pro": "on a computer"}


# ------------------------------------------------------------------------------------------------ geometry helpers
def img_box(m, dpr):
    return [round(m["x"] * dpr, 1), round(m["y"] * dpr, 1), round(m["w"] * dpr, 1), round(m["h"] * dpr, 1)]


def inter(a, b, gap=0.0):
    return not (a[0] + a[2] + gap <= b[0] or b[0] + b[2] + gap <= a[0] or a[1] + a[3] + gap <= b[1] or b[1] + b[3] + gap <= a[1])


def inside(a, b):
    return a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]


def cell_of(box, W, H, n, margin=2.0):
    """(row, col) of the grid cell that wholly contains box (image px), else None (straddles a border)."""
    cw, ch = W / n, H / n
    c0, c1 = math.floor((box[0] - margin) / cw), math.floor((box[0] + box[2] + margin) / cw)
    r0, r1 = math.floor((box[1] - margin) / ch), math.floor((box[1] + box[3] + margin) / ch)
    if c0 != c1 or r0 != r1 or not (0 <= c0 < n and 0 <= r0 < n):
        return None
    return r0, c0


def cell_label(r, c, n, style):
    if style == "named":
        return NAMES3[r][c]
    if style == "rowcol":
        return f"row {r + 1}, column {c + 1}"
    return f"{'ABCD'[r]}{c + 1}"


# ------------------------------------------------------------------------------------------------ screens
def build_screen(rng, platform=None, template=None):
    if platform is None:
        platform = rng.choices(list(PLATFORMS), weights=[v[1] for v in PLATFORMS.values()])[0]
    tmpls = PLATFORMS[platform][0]
    if template is None:
        template = rng.choices(list(tmpls), weights=[v[1] for v in tmpls.values()])[0]
    return tmpls[template][0](rng)


class Ctx:
    """A measured screen: the Screen + measured element boxes."""

    def __init__(self, sid, s, meas, seed_parts):
        self.sid, self.s, self.seed_parts = sid, s, seed_parts
        self.m = {e["el"]: e for e in meas}
        self.dpr = s.dpr
        self.Wi, self.Hi = round(s.width * s.dpr), round(s.height * s.dpr)

    def box(self, eid):
        return img_box(self.m[eid], self.dpr)

    def vis(self, eid):
        e = self.m.get(eid)
        return bool(e and e["visible"] and e["w"] >= 6 and e["h"] >= 6)

    def candidates(self):
        area = self.s.width * self.s.height
        return [k for k, v in self.s.els.items() if v["desc"] and self.vis(k) and self.m[k]["w"] * self.m[k]["h"] <= 0.2 * area]

    def occluded(self):
        d = self.s.dialog
        if not d or d not in self.m:
            return []
        db = self.box(d)
        return [k for k, v in self.s.els.items() if v["desc"] and k in self.m and not self.m[k]["visible"] and not k.startswith("dlg")
                and self.m[k]["w"] >= 6 and inside(self.box(k), db)]


# ------------------------------------------------------------------------------------------------ overlays
def grid_overlay(ctx, n, colour):
    s = ctx.s
    sh = []
    for i in range(1, n):
        sh.append({"t": "vline", "p": round(s.width * i / n, 2), "c": colour})
        sh.append({"t": "hline", "p": round(s.height * i / n, 2), "c": colour})
    bw, bh = (26, 17) if s.platform != "mobile" else (24, 16)
    badges = []
    for r in range(n):
        for c in range(n):
            x, y = s.width * c / n + 3, s.height * r / n + 3
            sh.append({"t": "badge", "x": round(x, 1), "y": round(y, 1), "w": bw, "h": bh, "text": cell_label(r, c, n, "drawn"), "c": colour})
            badges.append([round(x * s.dpr, 1), round(y * s.dpr, 1), round(bw * s.dpr, 1), round(bh * s.dpr, 1)])
    return sh, badges


def place_marks(ctx, eids, colour, rng):
    """Boxes around eids (CSS px, padded) with letter badges placed where they touch no other mark. -> (shapes, marks) or None."""
    s = ctx.s
    pad = 3
    boxes = []
    for k in eids:
        e = ctx.m[k]
        x, y = max(0.0, e["x"] - pad), max(0.0, e["y"] - pad)
        w = min(s.width, e["x"] + e["w"] + pad) - x
        h = min(s.height, e["y"] + e["h"] + pad) - y
        boxes.append([x, y, w, h])
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if inter(boxes[i], boxes[j], 2):
                return None
    bw = bh = 16 if s.platform != "mobile" else 17
    letters = list(LETTERS[:len(eids)])
    rng.shuffle(letters)
    placed = []
    shapes = []
    marks = []
    for k, b, L in zip(eids, boxes, letters):
        x, y, w, h = b
        cands = [(x, y - bh - 1), (x + w - bw, y - bh - 1), (x, y + h + 1), (x - bw - 1, y), (x + w + 1, y), (x + 1, y + 1)]
        ok = None
        for cx, cy in cands:
            bb = [cx, cy, bw, bh]
            if cx < 0 or cy < 0 or cx + bw > s.width or cy + bh > s.height:
                continue
            if any(inter(bb, ob) for ob, kk in zip(boxes, eids) if kk != k):
                continue
            if any(inter(bb, pb) for pb in placed):
                continue
            ok = bb
            break
        if ok is None:
            return None
        placed.append(ok)
        shapes.append({"t": "box", "x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1), "c": colour})
        shapes.append({"t": "badge", "x": round(ok[0], 1), "y": round(ok[1], 1), "w": bw, "h": bh, "text": L, "c": colour})
        d = s.dpr
        marks.append({"letter": L, "el": k, "box": [round(v * d, 1) for v in (x, y, w, h)], "badge": [round(v * d, 1) for v in ok],
                      "el_box": ctx.box(k)})
    return shapes, sorted(marks, key=lambda m: m["letter"])


# ------------------------------------------------------------------------------------------------ tasks
def q(task, ftype, question, options, gold, unk, diff, state_extra, spec):
    return {"task": task, "type": ftype, "question": question, "options": options, "gold": gold, "unknown_reason": unk,
            "difficulty": max(2, min(5, diff)), "state_extra": state_extra, "spec": spec}


def small(ctx, eid):
    e = ctx.m[eid]
    return e["w"] * e["h"] < 900


def t_locate(ctx, rng, style, n, badges, want_unknown=None, used=()):
    s = ctx.s
    occ_all = ctx.occluded()
    p_unk = 0.5 if occ_all else P_UNKNOWN
    unk = (rng.random() < p_unk) if want_unknown is None else want_unknown
    labels = [cell_label(r, c, n, style) for r in range(n) for c in range(n)]
    extra = {"named": {"regions": "the screen split into a 3 x 3 grid"},
             "rowcol": {"regions": f"{n} x {n} grid; rows top to bottom, columns left to right"},
             "drawn": {"regions": "grid drawn on the screenshot, cells labelled A1 to " + cell_label(n - 1, n - 1, n, "drawn")}}[style]
    base_d = {"named": 2, "rowcol": 3, "drawn": 3}[style] + (1 if s.platform == "pro" else 0)
    qtext = rng.choice(["In which part of the screen is {d}?", "Where on the screen is {d}?"]) if style != "drawn" else \
        rng.choice(["In which grid cell is {d}?", "Which labelled cell contains {d}?"])
    grid = {"rows": n, "cols": n, "style": style, "image_size": [ctx.Wi, ctx.Hi], "badges": badges}
    if unk:
        occ = occ_all
        if occ and rng.random() < 0.75:
            k = rng.choice(occ)
            spec = {"grid": grid, "target": k, "target_desc": s.els[k]["desc"], "target_box": ctx.box(k), "target_visible": False,
                    "dialog_box": ctx.box(s.dialog), "unknown_construction": "occluded"}
            return q("locate", "choice", qtext.format(d=s.els[k]["desc"]), labels, None, "insufficient_evidence", base_d + 1, extra, spec)
        if s.absent:
            d, _ = rng.choice(s.absent)
            spec = {"grid": grid, "target": None, "target_desc": d, "target_box": None, "unknown_construction": "absent",
                    "present_descs": sorted(v["desc"] for v in s.els.values() if v["desc"])}
            return q("locate", "choice", qtext.format(d=d), labels, None, "not_listed", base_d + 1, extra, spec)
        return None
    cands = [k for k in ctx.candidates() if k not in used]
    rng.shuffle(cands)
    for k in cands:
        b = ctx.box(k)
        rc = cell_of(b, ctx.Wi, ctx.Hi, n)
        if rc is None:
            continue
        if badges and any(inter(b, bb) for bb in badges):
            continue
        gold = cell_label(rc[0], rc[1], n, style)
        spec = {"grid": grid, "target": k, "target_desc": s.els[k]["desc"], "target_box": b, "cell": list(rc)}
        return q("locate", "choice", qtext.format(d=s.els[k]["desc"]), labels, gold, None, base_d + (1 if small(ctx, k) else 0), extra, spec)
    return None


def pick_marks(ctx, rng, target, k_total, pool=None):
    cands = [c for c in (pool or ctx.candidates()) if c != target]
    if target is not None:
        tb = ctx.box(target)
        cands.sort(key=lambda c: (ctx.box(c)[0] - tb[0]) ** 2 + (ctx.box(c)[1] - tb[1]) ** 2)
        cands = cands[:14]
    rng.shuffle(cands)
    for _ in range(30):
        need = k_total - (1 if target is not None else 0)
        if len(cands) < need:
            return None
        pick = rng.sample(cands, need) + ([target] if target is not None else [])
        rng.shuffle(pick)
        yield pick


def t_ground(ctx, rng, colour, form="desc", want_unknown=None):
    """-> (question dict, overlay shapes, marks) or None."""
    s = ctx.s
    unk = (rng.random() < P_UNKNOWN) if want_unknown is None else want_unknown
    k_total = rng.choice([4, 4, 5, 6]) if s.platform != "mobile" else rng.choice([3, 4, 4, 5])
    cands = ctx.candidates()
    if form == "goal":
        cands_t = [k for k in cands if s.els[k]["goals"] and not s.els[k]["disabled"]]
    else:
        cands_t = cands
    if unk:
        pool = [a for a in s.absent if (a[1] if form == "goal" else a[0])]
        if not pool:
            return None
        d, g = rng.choice(pool)
        target, tdesc, tgoal = None, d, g
    else:
        if not cands_t:
            return None
        target = rng.choice(cands_t)
        tdesc, tgoal = s.els[target]["desc"], (rng.choice(s.els[target]["goals"]) if form == "goal" else None)
    for pick in pick_marks(ctx, rng, target, k_total) or []:
        if form == "goal" and tgoal:
            # the goal must belong to exactly the target among the marked elements
            others = [p for p in pick if p != target and tgoal in s.els[p]["goals"]]
            if others:
                continue
        res = place_marks(ctx, pick, colour, rng)
        if res is None:
            continue
        shapes, marks = res
        gold = next((m["letter"] for m in marks if m["el"] == target), None) if target else None
        letters = [m["letter"] for m in marks]
        verb = "tap" if s.platform == "mobile" else "click"
        if form == "goal":
            qtext = f"Which marked box would you {verb} to {tgoal}?"
            task = {"mobile": "tap_box", "pro": "ground_goal"}.get(s.platform, "ground")
        else:
            qtext = f"Which marked box is {tdesc}?"
            task = "ground"
        base = {"desktop": 2, "mobile": 3, "pro": 4}[s.platform] + (1 if form == "goal" else 0)
        if target and small(ctx, target) and s.platform != "pro":
            base += 1
        spec = {"marks": marks, "target": target, "target_desc": tdesc, "goal": tgoal, "target_box": ctx.box(target) if target else None,
                "image_size": [ctx.Wi, ctx.Hi]}
        if unk:
            spec["unknown_construction"] = "absent"
            spec["present_descs"] = sorted(v["desc"] for v in s.els.values() if v["desc"])
            spec["present_goals"] = sorted({g for v in s.els.values() for g in v["goals"]})
        extra = {"marks": f"boxes labelled {letters[0]} to {letters[-1]} drawn on the screenshot"}
        return q(task, "choice", qtext, letters, gold, "not_listed" if unk else None, base + (1 if unk else 0), extra, spec), shapes, marks
    return None


def t_tap_label(ctx, rng, want_unknown=None, only=None):
    s = ctx.s
    groups = [(g, [k for k in ids if ctx.vis(k) and s.els[k]["label"]]) for g, ids in s.groups.items()]
    groups = [(g, ids) for g, ids in groups if len(ids) >= 3 and len({s.els[k]["label"] for k in ids}) == len(ids)]
    if only == "menu":
        groups = [x for x in groups if x[0] == "menu"]
    elif only == "nomenu":
        groups = [x for x in groups if x[0] != "menu"]
    if not groups:
        return None
    g, ids = rng.choice(groups)
    ids = ids[:12]
    unk = (rng.random() < P_UNKNOWN) if want_unknown is None else want_unknown
    menu = g == "menu"
    labels = [s.els[k]["label"] for k in ids]
    if unk:
        pool = [a for a in s.absent if a[1]]
        if not pool:
            return None
        _, goal = rng.choice(pool)
        target = None
    else:
        cands = [k for k in ids if s.els[k]["goals"] and not (menu and s.els[k]["disabled"])]
        cands = [k for k in cands if all(s.els[k]["goals"][0] not in s.els[o]["goals"] for o in ids if o != k)]
        if not cands:
            return None
        target = rng.choice(cands)
        goal = s.els[target]["goals"][0]
    qtext = f"Which menu item would you use to {goal}?" if menu else (f"Which should you tap to {goal}?" if s.platform == "mobile" else f"Which should you click to {goal}?")
    spec = {"group": g, "option_elements": dict(zip(labels, ids)), "target": target, "goal": goal,
            "option_boxes": {s.els[k]["label"]: ctx.box(k) for k in ids}}
    if unk:
        spec["unknown_construction"] = "absent"
        spec["option_goals"] = {s.els[k]["label"]: s.els[k]["goals"] for k in ids}
    task = "menu" if menu else "tap_label"
    d = (3 if menu else 2) + (1 if unk else 0)
    return q(task, "choice", qtext, labels, s.els[target]["label"] if target else None, "not_listed" if unk else None, d, {}, spec)


def t_possible(ctx, rng):
    s = ctx.s
    cands = [k for k in s.possible if ctx.vis(k) and s.els[k]["goals"]]
    if not cands:
        return None
    dis = [k for k in cands if s.els[k]["disabled"]]
    k = rng.choice(dis) if dis and rng.random() < 0.5 else rng.choice(cands)
    goal = s.els[k]["goals"][0]
    menu = s.platform == "pro"
    qtext = f"Can you {goal} from this menu right now?" if menu else f"Can you {goal} right now?"
    spec = {"target": k, "target_desc": s.els[k]["desc"], "goal": goal, "disabled": s.els[k]["disabled"], "target_box": ctx.box(k),
            "screen_state": s.meta}
    return q("menu_possible" if menu else "possible", "noul", qtext, None, not s.els[k]["disabled"], None, 4 if menu else 3, {}, spec)


def t_next(ctx, rng):
    s = ctx.s
    cands = [k for k in ctx.candidates() if s.els[k]["action"]]
    if not cands:
        return None
    acts = sorted({s.els[k]["action"] for k in cands})     # pick the action class first: no "opens another screen" prior
    act = rng.choice(acts)
    k = rng.choice([c for c in cands if s.els[c]["action"] == act])
    gold = s.els[k]["action"]
    conflict = set().union(*[g for g in CONFLICT if gold in g]) if any(gold in g for g in CONFLICT) else {gold}
    others = [a for a in ACTIONS if a not in conflict and a != gold and a != COUNTERPART.get(gold)]
    n_opt = rng.choice([4, 5])
    opts = [gold] + ([COUNTERPART[gold]] if gold in COUNTERPART else [])
    opts += rng.sample(others, n_opt - len(opts))
    rng.shuffle(opts)
    spec = {"target": k, "target_desc": s.els[k]["desc"], "action": gold, "disabled": s.els[k]["disabled"], "target_box": ctx.box(k)}
    return q("next", "choice", f"What happens if you tap {s.els[k]['desc']}?", opts, gold, None, 3, {}, spec)


def t_perm(ctx, rng, kind):
    s = ctx.s
    mt = s.meta
    lines = [k for k in s.els if k.startswith("perm_")]
    if not all(ctx.vis(k) for k in lines):
        return None
    extra = mt["extra"]
    spec = {"task_text": mt["task"], "needed": mt["needed"], "requested": mt["requested"], "extra": extra,
            "option_boxes": {s.els[k]["label"]: ctx.box(k) for k in lines}}
    st = {"task": mt["task"]}
    if kind == "perm_more":
        return q("perm_more", "noul", "Does this request ask for more access than the task needs?", None, bool(extra), None, 4, st, spec)
    if len(mt["requested"]) < 2:
        return None
    gold = extra[0] if extra else None
    return q("perm_which", "choice", "Which requested permission does the task not need?", list(mt["requested"]), gold,
             None if gold else "false_premise", 4 + (0 if gold else 1), st, spec)


# ------------------------------------------------------------------------------------------------ planning
def plan(ctx, rng, want=None):
    """-> list of images: {"overlay": shapes, "questions": [...], "variant": ...}. want = (task, unknown) restricts."""
    s = ctx.s
    colour = rng.choice(MARK_COLOURS)
    out = []
    wt, wu = want if want else (None, None)

    def wanted(task):
        return wt is None or wt == task

    if wt is None:
        if s.platform == "desktop":
            v = rng.choices(["plain", "drawn", "boxes"], weights=(35, 25, 40))[0]
        elif s.platform == "mobile":
            v = rng.choices(["plain", "drawn", "boxes"], weights=(58, 12, 30))[0]
            if s.template == "permission":
                v = rng.choices(["plain", "boxes"], weights=(80, 20))[0]
        else:
            v = rng.choices(["plain", "drawn", "boxes"], weights=(45, 15, 40))[0]
    else:
        v = {"locate": rng.choice(["plain", "drawn"]), "ground": "boxes", "ground_goal": "boxes", "tap_box": "boxes"}.get(wt, "plain")
    qs, shapes = [], []
    if v == "drawn":
        n = rng.choice([3, 4])
        shapes, badges = grid_overlay(ctx, n, colour)
        if wanted("locate"):
            used = set()
            for _ in range(rng.choice([1, 2]) if wt is None else 1):
                x = t_locate(ctx, rng, "drawn", n, badges, wu, used)
                if x:
                    used.add(x["spec"]["target"])
                    qs.append(x)
        return [{"overlay": shapes, "questions": qs, "variant": f"grid{n}"}] if qs else []
    if v == "boxes":
        forms = ["desc"]
        if s.platform in ("mobile", "pro"):
            forms = ["desc", "goal"]
        form = rng.choice(forms)
        if wt in ("tap_box", "ground_goal"):
            form = "goal"
        elif wt == "ground":
            form = "desc"
        x = t_ground(ctx, rng, colour, form, wu)
        if not x:
            return []
        qd, shapes, marks = x
        qs.append(qd)
        # a second question on the same marks: another marked element by description
        if wt is None and qd["gold"] is not None and rng.random() < 0.5:
            other = [m for m in marks if m["el"] != qd["spec"]["target"]]
            if other:
                m = rng.choice(other)
                d = s.els[m["el"]]["desc"]
                spec = dict(qd["spec"], target=m["el"], target_desc=d, goal=None, target_box=m["el_box"])
                base = {"desktop": 2, "mobile": 3, "pro": 4}[s.platform]
                qs.append(q("ground", "choice", f"Which marked box is {d}?", qd["options"], m["letter"], None, base, qd["state_extra"], spec))
        return [{"overlay": shapes, "questions": qs, "variant": "boxes"}]
    # plain
    tasks = []
    if s.platform == "mobile":
        if s.template == "permission":
            tasks = ["perm_more", "perm_which", "tap_label"]
            rng.shuffle(tasks)
            tasks = ["perm_which", "perm_more"] + [t for t in tasks if not t.startswith("perm")]
        else:
            tasks = ["tap_label", "possible", "next", "locate"]
            if s.dialog:
                tasks = ["locate"]
    elif s.platform == "pro":
        tasks = ["locate", "tap_label"]
        if s.meta.get("menu_open"):
            tasks = ["menu", "menu_possible", "locate"]
    else:
        tasks = ["locate"]
    if wt is not None:
        tasks = [wt] if wt in tasks or wt in ("menu", "menu_possible", "tap_label", "possible", "next", "perm_more", "perm_which", "locate") else []
    if s.template != "permission":
        rng.shuffle(tasks)
    style = rng.choice(["named", "rowcol"]) if s.platform != "mobile" else rng.choice(["named", "named", "rowcol"])
    n = 3 if style == "named" else 4
    k = 1 if wt is not None else rng.choice([1, 2, 2, 3])
    used = set()
    for task in tasks:
        if len(qs) >= k:
            break
        x = None
        if task == "locate":
            x = t_locate(ctx, rng, style, n, [], wu, used)
        elif task in ("tap_label", "menu"):
            x = t_tap_label(ctx, rng, wu, "menu" if task == "menu" else "nomenu")
        elif task in ("possible", "menu_possible"):
            x = t_possible(ctx, rng)
        elif task == "next":
            x = t_next(ctx, rng)
        elif task in ("perm_more", "perm_which"):
            x = t_perm(ctx, rng, task)
        if x and (wu is None or (x["gold"] is None) == wu):
            if x["spec"].get("target"):
                used.add(x["spec"]["target"])
            qs.append(x)
    return [{"overlay": [], "questions": qs, "variant": "plain"}] if qs else []


# ------------------------------------------------------------------------------------------------ rows
def make_rows(ctx, img, sha, rel, seed):
    s = ctx.s
    rows = []
    for qd in img["questions"]:
        field = {"type": qd["type"], "question": qd["question"]}
        gold = qd["gold"]
        if qd["type"] == "choice":
            taken = set()
            opts = [{"key": option_key(o, taken), "text": o} for o in qd["options"]]
            field["options"] = opts
            gold = next(o["key"] for o in opts if o["text"] == qd["gold"]) if qd["gold"] is not None else None
        state = {"screen": f"a screenshot of {s.screen_desc}"}
        state.update(qd["state_extra"])
        fam = FAMILY[(s.platform, qd["task"])]
        spec = dict(qd["spec"])
        rows.append({"id": None, "source": "I", "dataset": "screens_synthetic", "family": fam, "difficulty": qd["difficulty"],
                     "state": state, "images": [rel], "field": field, "gold": gold, "unknown_reason": qd["unknown_reason"],
                     "gold_kind": "constructed", "parent_id": None,
                     "provenance": {"licence": "generated", "generator": GENERATOR,
                                    "renderer": "puppeteer-core + headless Chromium (scripts/p3/screen_templates/render.mjs)",
                                    "screen_id": ctx.sid, "screen_seed": list(ctx.seed_parts), "screen_template": s.template,
                                    "platform": s.platform, "task": qd["task"], "variant": img["variant"],
                                    "viewport": [s.width, s.height], "dpr": s.dpr, "image_size": [ctx.Wi, ctx.Hi],
                                    "spec": spec, "screen_meta": s.to_meta(), "unknown_construction": spec.get("unknown_construction"),
                                    "image_sha256": [sha], "upstream_split": "generated", "seed": seed,
                                    "content": "invented brands, people, places and apps"}})
    return rows


def run(screens: list[tuple], work: Path, img_dir: Path, seed: str, conc: int, wants=None):
    """screens: [(sid, Screen, seed_parts)] -> (rows grouped per screen, stats)."""
    stats = Counter()
    jobs = [{"id": sid, "html": s.html, "width": s.width, "height": s.height, "dpr": s.dpr, "shot": False} for sid, s, _ in screens]
    meas = render(jobs, work, conc=conc, tag="measure")
    plans = []
    jobs2 = []
    for i, (sid, s, parts) in enumerate(screens):
        r = meas.get(sid)
        if not r or not r["ok"]:
            stats["render_failed"] += 1
            plans.append(None)
            continue
        ctx = Ctx(sid, s, r["elements"], parts)
        prng = rng_for(seed, "plan", sid)
        imgs = plan(ctx, prng, wants[i] if wants else None)
        if not imgs:
            stats["no_question"] += 1
        plans.append((ctx, imgs))
        for j, img in enumerate(imgs):
            jid = f"{sid}_{j}"
            img["job"] = jid
            jobs2.append({"id": jid, "html": s.html, "width": s.width, "height": s.height, "dpr": s.dpr, "overlay": img["overlay"], "shot": True})
    shots = render(jobs2, work, conc=conc, tag="shot")
    bench = bench_image_shas()
    img_dir.mkdir(parents=True, exist_ok=True)
    grouped = []
    for p in plans:
        if p is None:
            grouped.append([])
            continue
        ctx, imgs = p
        rows = []
        for img in imgs:
            r = shots.get(img["job"])
            if not r or not r["ok"]:
                stats["shot_failed"] += 1
                continue
            m2 = {e["el"]: e for e in r["elements"]}
            if any(abs(m2[k][f] - ctx.m[k][f]) > 0.6 for k in ctx.m for f in ("x", "y", "w", "h") if k in m2):
                stats["layout_moved"] += 1
                continue
            png = work / f"{img['job']}.png"
            sha = hashlib.sha256(png.read_bytes()).hexdigest()
            if sha in bench:
                stats["bench_sha"] += 1
                continue
            dst = img_dir / f"{sha}.png"
            if not dst.exists():
                shutil.copyfile(png, dst)
            rel = os.path.relpath(dst, ROOT / "data")
            rows += make_rows(ctx, img, sha, rel, seed)
        grouped.append(rows)
    return grouped, stats


def generate(seed, count, work, img_dir, prefix, conc):
    per = 1.4
    n = int(count / per) + 1
    screens = []
    for i in range(n):
        rng = rng_for(seed, "screen", i)
        sid = f"s{i:05d}"
        screens.append((sid, build_screen(rng), (seed, "screen", i)))
    grouped, stats = run(screens, work, img_dir, seed, conc)
    rows = [r for g in grouped for r in g]
    for i, r in enumerate(rows):
        r["id"] = f"{prefix}{i:06d}"
    return rows, stats


def variants_of(path, per, work, img_dir, conc):
    screens, wants, parents = [], [], []
    for row in read(path):
        pv = row.get("provenance") or {}
        if pv.get("generator") != GENERATOR:
            continue
        for k in range(per):
            rng = rng_for("variant", row["id"], k)
            sid = f"v{len(screens):05d}"
            screens.append((sid, build_screen(rng, pv["platform"], pv["screen_template"]), ("variant", row["id"], k)))
            wants.append((pv["task"], row["gold"] is None))
            parents.append((row["id"], k))
    grouped, _ = run(screens, work, img_dir, "variant", conc, wants)
    out = []
    for (pid, k), rows in zip(parents, grouped):
        if not rows:
            continue
        r = rows[0]
        r["id"] = f"{pid}-v{k + 1}"
        r["parent_id"] = pid
        r["provenance"]["variant_of"] = pid
        out.append(r)
    return out


def summary(rows):
    c = Counter
    return {"rows": len(rows), "by_family": dict(c(r["family"] for r in rows)), "by_task": dict(c(r["provenance"]["task"] for r in rows)),
            "by_template": dict(c(r["provenance"]["screen_template"] for r in rows)),
            "by_type": dict(c(r["field"]["type"] for r in rows)), "by_difficulty": dict(sorted(c(r["difficulty"] for r in rows).items())),
            "unknown": sum(r["gold"] is None for r in rows), "unknown_share": round(sum(r["gold"] is None for r in rows) / max(1, len(rows)), 3),
            "unknown_by_reason": dict(c(r["unknown_reason"] for r in rows if r["gold"] is None)),
            "images": len({r["images"][0] for r in rows}), "screens": len({r["provenance"]["screen_id"] for r in rows})}


def heldout_rows(rows, train_paths, target, spec_key=None):
    """Leakage-filter fresh-seed rows against training files (screen_templates/heldout_filter.py); prune unused PNGs."""
    import heldout_filter as HF
    train_imgs, train_specs = set(), set()
    for p in train_paths:
        if Path(p).name.startswith(("I-geometry", "I-screens")):
            for r in read(p):
                train_imgs.add(Path(r["images"][0]).name)
                if spec_key:
                    train_specs.add(spec_key(r))
    if spec_key:
        bad = {r["parent_id"] or r["id"] for r in rows if spec_key(r) in train_specs}
        rows = [r for r in rows if (r["parent_id"] or r["id"]) not in bad]
    kept, stats = HF.filter_heldout(rows, [Path(p) for p in train_paths], train_imgs, target)
    used = {Path(r["images"][0]).name for r in kept}
    img_dir = ROOT / "data" / Path(kept[0]["images"][0]).parent if kept else None
    if img_dir and img_dir.name == "heldout":
        for f in img_dir.glob("*.png"):
            if f.name not in used:
                f.unlink()
    return kept, stats


def default_train(own: Path) -> list[str]:
    pool = ROOT / "data" / "p3" / "pool"
    return [str(own)] + sorted(str(p) for p in (pool / "shards").glob("pool-*.jsonl")) + [str(pool / "large-choice.jsonl")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--heldout-tag", default="hf1")
    ap.add_argument("--train", nargs="*", default=None,
                    help="held-out mode: training files to leakage-filter against (default: the training candidates file + "
                         "data/p3/pool/shards/*.jsonl + large-choice.jsonl)")
    ap.add_argument("--variant-of", default=None)
    ap.add_argument("--variants-per", type=int, default=2)
    ap.add_argument("--keep-work", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    img_dir = Path(a.img_dir).resolve() if a.img_dir else IMG_DIR
    work = Path(a.keep_work) if a.keep_work else Path(tempfile.mkdtemp(prefix="p3scr-"))
    if a.variant_of:
        rows = variants_of(a.variant_of, a.variants_per, work, img_dir, a.conc)
        out, stats = a.out or str(OUT.with_name("I-screens-var.jsonl")), {}
    elif a.heldout:
        seed = a.seed or f"p3-screens-heldout-{a.heldout_tag}"
        target = a.count or 250
        rows, stats = generate(seed, int(target * 1.4), work, (img_dir / "heldout" if not a.img_dir else img_dir), f"p3-{a.heldout_tag}-I-screens-", a.conc)
        for r in rows:
            r["provenance"].update({"heldout": "fresh", "heldout_tag": a.heldout_tag, "heldout_generator": "gen_screens.py --heldout"})
        rows, hstats = heldout_rows(rows, a.train or default_train(OUT), target, None)
        stats = dict(stats, heldout_filter=hstats)
        print(json.dumps(hstats))
        out = a.out or str(HELDOUT_OUT)
    else:
        seed = a.seed or "p3-screens-v1"
        rows, stats = generate(seed, a.count or 5000, work, img_dir, "p3-I-screens-", a.conc)
        out = a.out or str(OUT)
    n = write(out, rows)
    print(json.dumps({"out": out, "written": n, "stats": dict(stats), **summary(rows)}, indent=1, ensure_ascii=False))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
