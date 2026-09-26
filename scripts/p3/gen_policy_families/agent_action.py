"""agent_action + routing_hard: synthetic state machines where exactly one next action / element / queue is correct by construction.

Kinds
    grid_move      grid world with walls, a key and a door; first move on a shortest path, or which plan is shortest (BFS-verified)
    dom_target     accessibility tree (duplicates, disabled/hidden/readonly copies, collapsed menus, modal dialogs); which element next
    craft_next     recipes + inventory + goal; the one recipe that is both possible now and still needed
    tool_next      tool-call log for two similar entities (failed calls, lost responses); the next API call with the right arguments
    wizard_step    multi-step form with validation rules; which field blocks "Next" on the current step
    routing_table  (routing_hard) ticket fields + prioritised routing rules with exceptions; which queue
    oncall_route   (routing_hard) on-call rota in one UTC offset, alert in another, swaps and leave; who is paged
    tool_select    (routing_hard) tool catalogue with scope limits (age, amount, region, verification); which tool handles the request

Every generator builds a structured world, computes the answer with one code path and stores a compact spec; `recheck(spec)`
recomputes it with a separate code path (and, for unknown children, enumerates the hidden fact's values and returns None when
they disagree).
"""
from __future__ import annotations

import datetime as dt
import random
from collections import deque

from .common import DOMAIN_IDS, DOMAINS, P, People, chance, choice_field, code, fd, item, ordered_choice_field

FAMILY = "agent_action"
FAMILIES = ("agent_action", "routing_hard")
KINDS = ["grid_move", "dom_target", "craft_next", "tool_next", "wizard_step", "routing_table", "oncall_route", "tool_select"]
ROUTING_KINDS = {"routing_table", "oncall_route", "tool_select"}


class Retry(ValueError):
    pass


def generate(rng: random.Random, difficulty: int, kind: str, want_unknown: bool) -> list[dict]:
    fn = GEN[kind]
    for _ in range(60):
        try:
            items = fn(rng, difficulty, want_unknown)
        except Retry:
            continue
        for it in items:
            it["family"] = "routing_hard" if kind in ROUTING_KINDS else "agent_action"
            # test convention: recheck() returns the option key, so each option's canonical value is its key
            it["spec"] = dict(it["spec"], opt_values={o["key"]: o["key"] for o in it["field"]["options"]})
        return items
    raise ValueError(f"{kind}: no valid world")


def _unknown_child(parent: dict, state, spec: dict, hints: dict | None = None) -> dict:
    return item(state, parent["field"], None, parent["kind"], parent["domain"], spec, hints or {}, child_of=0,
                unknown_reason="insufficient_evidence")


# ================================================================================================ grid_move
DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
DIR_WORD = {"N": "north", "S": "south", "E": "east", "W": "west"}
GRID_FLAVOUR = [
    ("warehouse robot", "the pick-up bay", "a shelving rack"), ("maze runner in a puzzle game", "the exit tile", "a hedge"),
    ("cleaning robot", "its charging dock", "a piece of furniture"), ("delivery rover on a campus map", "the drop-off point", "a building"),
    ("rescue drone flying at fixed altitude", "the survivor's location", "a collapsed section"), ("hospital courier robot", "the pharmacy hatch", "a wall"),
    ("museum guide robot", "the gallery entrance", "a display case"), ("farm rover", "the feed store", "a fence section"),
]


def _g_step(w: dict, s: tuple, d: str, locked: bool):
    x, y, k = s
    dx, dy = DIRS[d]
    nx, ny = x + dx, y + dy
    if not (0 <= nx < w["W"] and 0 <= ny < w["H"]) or w["rows"][ny][nx] == "#":
        return None
    if [nx, ny] == w["door"] and locked and not k:
        return None
    return (nx, ny, k or [nx, ny] == w["key"])


def _g_first_moves(w: dict, locked: bool, order: str = "NSEW"):
    """Forward BFS propagating the set of first moves along the shortest-path DAG. Returns (L, first_moves, one_path)."""
    s0 = (w["start"][0], w["start"][1], False)
    dist, fm, par = {s0: 0}, {s0: set()}, {s0: None}
    q = deque([s0])
    while q:
        s = q.popleft()
        for d in order:
            t = _g_step(w, s, d, locked)
            if t is None:
                continue
            f = {d} if s == s0 else fm[s]
            if t not in dist:
                dist[t], fm[t], par[t] = dist[s] + 1, set(f), (s, d)
                q.append(t)
            elif dist[t] == dist[s] + 1:
                fm[t] |= f
    goals = [s for s in dist if [s[0], s[1]] == w["goal"]]
    if not goals:
        return None, set(), None
    L = min(dist[s] for s in goals)
    out = set()
    for s in goals:
        if dist[s] == L:
            out |= fm[s]
    end = next(s for s in goals if dist[s] == L)
    path = []
    while par[end] is not None:
        end, d = par[end]
        path.append(d)
    return L, out, path[::-1]


def _g_sim(w: dict, seq: list[str], locked: bool):
    s = (w["start"][0], w["start"][1], False)
    for d in seq:
        s = _g_step(w, s, d, locked)
        if s is None:
            return False, False
    return True, [s[0], s[1]] == w["goal"]


def _fmt_plan(seq: list[str], style: int) -> str:
    if style == 0:
        return ", ".join(seq)
    if style == 1:
        return " ".join(DIR_WORD[d] for d in seq)
    runs, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append(f"{j - i}×{seq[i]}")
        i = j
    return " then ".join(runs)


def _grid_render(w: dict, flav, locked, fmt: int, door_unknown=False) -> str:
    rows = [list(r) for r in w["rows"]]
    wall = "#" if fmt == 0 else "X"
    for y in range(w["H"]):
        for x in range(w["W"]):
            rows[y][x] = wall if w["rows"][y][x] == "#" else "."
    sx, sy = w["start"]; gx, gy = w["goal"]; kx, ky = w["key"]; dx, dy = w["door"]
    rows[sy][sx], rows[gy][gx], rows[ky][kx] = "A", "G", "k"
    rows[dy][dx] = "?" if door_unknown else ("D" if locked else "d")
    head = "    " + "".join(str(x) for x in range(w["W"]))
    body = [f"{y:>2}  " + "".join(r) for y, r in enumerate(rows)]
    agent, goal, obst = flav
    door_line = ("?  door; whether it is locked was not recorded in this snapshot" if door_unknown else
                 ("D  locked door (passable only while carrying the key)" if locked else "d  door, currently unlocked (ordinary floor)"))
    legend = [f"A  the {agent} (start)", f"G  {goal}", "k  key (picked up automatically when the agent steps on it)", door_line,
              f"{wall}  {obst} (impassable)", ".  open floor"]
    rules = P(random.Random(w["seed"]),
              "Each action moves the agent one cell north (up, row - 1), south (down, row + 1), east (right, column + 1) or west (left, column - 1). "
              "It cannot enter impassable cells or leave the map.",
              "Moves are single steps: north decreases the row number, south increases it, east increases the column number, west decreases it. "
              "Impassable cells and the map edge block movement.",
              "The agent acts one step at a time in the four compass directions (north = towards row 0, west = towards column 0) and can never "
              "step onto an impassable cell or off the grid.")
    coords = (f"\nPositions (column, row): agent {tuple(w['start'])}, goal {tuple(w['goal'])}, key {tuple(w['key'])}, door {tuple(w['door'])}."
              if fmt == 1 else "")
    return (f"Map of the area for a {agent} (columns across, rows down):\n\n" + "\n".join([head] + body) + "\n\nLegend:\n" +
            "\n".join("  " + l for l in legend) + "\n\nRules: " + rules + coords)


def gen_grid(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    W, H = {3: (7, 6), 4: (9, 7), 5: (10, 9)}[diff]
    dens = {3: 0.2, 4: 0.24, 5: 0.28}[diff]
    min_l = {3: 6, 4: 8, 5: 11}[diff]
    qkind = "first" if chance(rng, 0.5) else "plan"
    for _ in range(400):
        rows = ["".join("#" if rng.random() < dens else "." for _ in range(W)) for _ in range(H)]
        cells = [[x, y] for y in range(H) for x in range(W) if rows[y][x] == "."]
        if len(cells) < 10:
            continue
        start, goal = rng.sample(cells, 2)
        w = {"W": W, "H": H, "rows": rows, "start": start, "goal": goal, "key": [-9, -9], "door": [-9, -9], "seed": rng.randrange(10 ** 6)}
        L0, _, path0 = _g_first_moves(w, False)
        if L0 is None or L0 < 4:
            continue
        # put the door on the open shortest route, the key somewhere else
        s = (start[0], start[1])
        pcells = []
        for d in path0[:-1]:
            s = (s[0] + DIRS[d][0], s[1] + DIRS[d][1])
            pcells.append(list(s))
        pcells = pcells[1:] or pcells
        w["door"] = rng.choice(pcells)
        rest = [c for c in cells if c not in (start, goal, w["door"])]
        w["key"] = rng.choice(rest)
        locked = chance(rng, 0.65)
        L, fm, path = _g_first_moves(w, locked, "".join(rng.sample("NSEW", 4)))
        if L is None or L < min_l or len(fm) != 1:
            continue
        L2, fm2, _ = _g_first_moves(w, not locked)
        if want_unknown and qkind == "first" and (L2 is None or fm2 == fm):
            continue
        break
    else:
        raise Retry("grid")
    dom = rng.choice(DOMAIN_IDS)
    flav = rng.choice(GRID_FLAVOUR)
    fmt = rng.randrange(2)
    state = _grid_render(w, flav, locked, fmt)
    agent = flav[0]
    if qkind == "first":
        q = P(rng, f"Which move should the {agent} make first to reach G in as few moves as possible?",
              f"What is the first action on a shortest route from A to G for the {agent}?",
              f"The {agent} must reach G using the minimum number of moves. Which single move comes first?",
              "Choose the first step of the fastest legal route from the agent's position to G.")
        opts = {"N": "Move north", "S": "Move south", "E": "Move east", "W": "Move west", "none": "No move: G cannot be reached"}
        gold_m = next(iter(fm))
        field, gold = choice_field(rng, q, opts[gold_m], [v for k, v in opts.items() if k != gold_m])
        keymap = {m: next(o["key"] for o in field["options"] if o["text"] == t) for m, t in opts.items()}
        spec = {"k": "grid", "q": "first", "w": w, "locked": locked, "opt": keymap}
        near = [opts[m] for m in fm2 if m != gold_m] if L2 is not None and len(fm2) == 1 else []
    else:
        style = rng.randrange(3)
        cands = []
        def add(seq):
            valid, reach = _g_sim(w, seq, locked)
            if not (valid and reach and len(seq) == L) and seq != path and seq not in cands:
                cands.append(seq)
        for _ in range(60):
            i = rng.randrange(len(path) - 1)
            add(path[:i] + [path[i + 1], path[i]] + path[i + 2:])
            j = rng.randrange(len(path))
            add(path[:j] + [rng.choice([d for d in "NSEW" if d != path[j]])] + path[j + 1:])
        add(path[:-1])
        _, _, p_open = _g_first_moves(w, not locked)
        if p_open and p_open != path:
            add(p_open)
        # a longer legal route through a random waypoint
        for _ in range(10):
            mid = rng.choice(cells)
            w2 = dict(w, goal=mid)
            Lm, _, pm = _g_first_moves(w2, locked)
            if Lm:
                s_mid = (mid[0], mid[1], None)
                w3 = dict(w, start=mid)
                Lr, _, pr = _g_first_moves(w3, locked)
                if Lr and len(pm) + len(pr) > L:
                    seq = pm + pr
                    valid, reach = _g_sim(w, seq, locked)
                    if valid and reach:
                        add(seq)
                        break
        n_wrong = {3: 4, 4: 6, 5: 9}[diff]
        rng.shuffle(cands)
        # keep the near misses (illegal but same length) first so they are always present
        cands.sort(key=lambda s: 0 if len(s) == L else 1)
        wrong = cands[:n_wrong]
        if len(wrong) < 4:
            raise Retry("plan")
        texts = {tuple(s): _fmt_plan(s, style) for s in wrong + [path]}
        if len(set(texts.values())) != len(texts):
            raise Retry("plan text clash")
        q = P(rng, f"Which plan gets the {agent} from A to G in the fewest moves while obeying the rules?",
              "Which of these move sequences is a legal route to G of minimum length?",
              f"Several plans have been proposed for the {agent}. Which one reaches G legally with the fewest possible moves?",
              "Pick the shortest legal plan that ends on G.")
        field, gold = choice_field(rng, q, texts[tuple(path)], [texts[tuple(s)] for s in wrong])
        seqs = {o["key"]: next(list(s) for s in wrong + [path] if texts[tuple(s)] == o["text"]) for o in field["options"]}
        spec = {"k": "grid", "q": "plan", "w": w, "locked": locked, "seqs": seqs}
        near = [texts[tuple(s)] for s in wrong if len(s) == L][:3]
    door_txt = "D  locked door" if locked else "d  door, currently unlocked"
    parent = item(state, field, gold, "grid_move", dom, spec, {"near_miss": near, "decisive": [door_txt]})
    out = [parent]
    if want_unknown:
        spec_u = dict(spec, unk="door")
        if recheck(spec_u) is not None:
            raise Retry("door not decisive")
        out.append(_unknown_child(parent, _grid_render(w, flav, locked, fmt, door_unknown=True), spec_u))
    return out


def _rc_grid_dist(w, s, locked):
    seen, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        if [u[0], u[1]] == w["goal"]:
            return seen[u]
        x, y, k = u
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w["W"] and 0 <= ny < w["H"] and w["rows"][ny][nx] != "#" and not ([nx, ny] == w["door"] and locked and not k):
                t = (nx, ny, k or [nx, ny] == w["key"])
                if t not in seen:
                    seen[t] = seen[u] + 1
                    q.append(t)
    return None


def _rc_grid(spec, locked):
    w = spec["w"]
    s0 = (w["start"][0], w["start"][1], False)
    D = _rc_grid_dist(w, s0, locked)
    if spec["q"] == "first":
        if D is None:
            return spec["opt"]["none"]
        good = []
        for m, (dx, dy) in DIRS.items():
            nx, ny = s0[0] + dx, s0[1] + dy
            if 0 <= nx < w["W"] and 0 <= ny < w["H"] and w["rows"][ny][nx] != "#" and not ([nx, ny] == w["door"] and locked):
                t = (nx, ny, [nx, ny] == w["key"])
                dt_ = _rc_grid_dist(w, t, locked)
                if dt_ is not None and dt_ == D - 1:
                    good.append(m)
        return spec["opt"][good[0]] if len(good) == 1 else "__ambiguous__"
    good = []
    for key, seq in spec["seqs"].items():
        x, y, k, ok = s0[0], s0[1], False, True
        for d in seq:
            dx, dy = DIRS[d]
            x, y = x + dx, y + dy
            if not (0 <= x < w["W"] and 0 <= y < w["H"]) or w["rows"][y][x] == "#" or ([x, y] == w["door"] and locked and not k):
                ok = False
                break
            k = k or [x, y] == w["key"]
        if ok and [x, y] == w["goal"] and D is not None and len(seq) == D:
            good.append(key)
    return good[0] if len(good) == 1 else "__ambiguous__"


# ================================================================================================ dom_target
THEMES = {
    "retail": ("Checkout", ["Delivery address", "Billing address", "Order summary", "Payment", "Gift options", "Loyalty points", "Promotions"]),
    "saas_ops": ("Workspace settings", ["Profile", "Notifications", "API keys", "Billing", "Team members", "Security", "Integrations"]),
    "travel": ("Manage booking", ["Passenger details", "Seats", "Baggage", "Contact details", "Payment", "Special assistance", "Meals"]),
    "hr": ("Leave request", ["Employee details", "Leave dates", "Cover arrangements", "Approver", "Attachments", "Notes", "Pay options"]),
    "healthcare_admin": ("Patient portal", ["Personal details", "Appointments", "Prescriptions", "Insurance", "Messages", "Consent", "Referrals"]),
    "insurance": ("Claim form", ["Policy details", "Incident details", "Third parties", "Bank details", "Documents", "Declaration", "Witnesses"]),
    "finance": ("Payments", ["New payee", "Scheduled payments", "Standing orders", "Card controls", "Statements", "Alerts", "Limits"]),
    "logistics": ("Shipment console", ["Sender", "Recipient", "Parcel details", "Pickup", "Customs", "Label options", "Insurance cover"]),
    "public_sector": ("Permit application", ["Applicant", "Vehicle details", "Address", "Supporting evidence", "Fee payment", "Declaration", "Contact preferences"]),
    "legal": ("Client portal", ["Matter details", "Documents", "Invoices", "Contacts", "Engagement terms", "Messages", "Billing preferences"]),
}
TEXTBOXES = ["Full name", "Email address", "Phone number", "Postcode", "Reference number", "Promo code", "Notes", "Card number", "Date of birth",
             "Company name", "Account number", "Street address", "City", "Membership number", "Comments"]
BUTTONS = ["Save", "Apply", "Update", "Submit", "Continue", "Remove", "Edit", "Add another", "Confirm", "Verify"]
CHECKS = ["Remember my choice", "Send me updates", "I agree to the terms", "Use this as my default", "Mark as urgent"]
NAV_LINKS = ["Home", "Help centre", "Contact us", "Privacy", "Terms", "Order history", "Billing history", "Account settings", "Notification settings",
             "Security settings", "Saved cards", "Documents", "Messages", "Accessibility", "Status page", "Downloads"]
MENU_LINKS = [("Billing history", "Billing settings"), ("Security settings", "Security log"), ("Saved addresses", "Saved cards"),
              ("Export data", "Import data"), ("Team roles", "Team invites"), ("Download invoices", "Download receipts"),
              ("Travel documents", "Travel insurance"), ("Claim history", "Claim documents")]
BENIGN = ["Not now", "Close", "Keep editing", "Continue", "Dismiss", "Back to form"]
HARM = [("Discard changes", "closes this dialog and discards everything entered on the page"), ("Sign out", "ends the session immediately"),
        ("Start over", "clears the form and returns to step 1"), ("Cancel request", "cancels the request you are working on"),
        ("Leave page", "navigates away and loses unsaved input")]
MODAL_TITLES = ["Session timeout warning", "Cookie preferences", "Unsaved changes", "Survey invitation", "New feature announcement", "Security notice"]


def _dom_build(rng, dom, diff, scenario):
    """Nodes: [id, role, name, parent_index, flags]. Returns nodes, meta about the scenario."""
    title, secs = THEMES[dom]
    nodes: list[list] = []

    def add(role, name, parent, **flags):
        nodes.append([None, role, name, parent, {k: v for k, v in flags.items() if v is not None}])
        return len(nodes) - 1

    root_b = add("banner", "", None)
    nav = add("navigation", "Main", root_b)
    for l in rng.sample(NAV_LINKS, {3: 4, 4: 6, 5: 9}[diff]):
        add("link", l, nav)
    main = add("main", title, None)
    n_secs = {3: 3, 4: 5, 5: 7}[diff]
    sec_names = rng.sample(secs, n_secs)
    sec_idx = {}
    per = {3: (2, 3), 4: (3, 4), 5: (3, 6)}[diff]
    for s in sec_names:
        si = add(rng.choice(["form", "region", "group"]), s, main)
        sec_idx[s] = si
        for tb in rng.sample(TEXTBOXES, rng.randint(1, per[0])):
            add("textbox", tb, si, readonly=True if chance(rng, 0.08) else None)
        if chance(rng, 0.4):
            add("checkbox", rng.choice(CHECKS), si, checked=chance(rng, 0.5))
        for b in rng.sample(BUTTONS, rng.randint(1, per[1] - 1)):
            add("button", b, si, disabled=True if chance(rng, 0.12) else None)
    foot = add("contentinfo", "", None)
    for l in rng.sample(NAV_LINKS, 3):
        add("link", l, foot)
    meta = {"secs": sec_names, "sec_idx": sec_idx}
    return nodes, meta


def _dom_ancestors(nodes, i):
    out = []
    p = nodes[i][3]
    while p is not None:
        out.append(p)
        p = nodes[p][3]
    return out


def _dom_render(nodes, fmt, unk_flags=None):
    unk_flags = unk_flags or {}
    kids: dict = {}
    for i, n in enumerate(nodes):
        kids.setdefault(n[3], []).append(i)
    lines = []

    def flag_txt(i):
        f = nodes[i][4]
        if i in unk_flags:
            return [unk_flags[i]]
        parts = []
        for k in ("disabled", "hidden", "readonly", "modal", "checked"):
            if k in f:
                if k == "modal" or k == "checked":
                    parts.append(f"{k}={'true' if f[k] else 'false'}")
                elif f[k]:
                    parts.append(k)
        if "expanded" in f:
            parts.append(f"expanded={'true' if f['expanded'] else 'false'}")
        if "controls" in f and "hide_controls" not in f:
            parts.append(f"controls={nodes[f['controls']][0]}")
        if "desc" in f:
            parts.append(f'description="{f["desc"]}"')
        return parts

    def walk(i, depth):
        n = nodes[i]
        name = f' "{n[2]}"' if n[2] else ""
        fl = flag_txt(i)
        if fmt == 0:
            lines.append("  " * depth + f"[{n[0]}] {n[1]}{name}" + (" " + " ".join(fl) if fl else ""))
        else:
            lines.append("  " * depth + f"- {n[1]}{name} [ref={n[0]}]" + (" {" + ", ".join(fl) + "}" if fl else ""))
        for c in kids.get(i, []):
            walk(c, depth + 1)

    for r in kids.get(None, []):
        walk(r, 0)
    return "\n".join(lines)


INTERACTIVE = {"button", "link", "textbox", "checkbox", "combobox"}


def _dom_next(nodes, goal, overrides=None):
    """Generator-side resolution: the element to activate next (index) or None if not unique."""
    ov = overrides or {}

    def fl(i):
        f = dict(nodes[i][4])
        f.update(ov.get(i, {}))
        return f

    def hidden(i):
        return any(fl(j).get("hidden") for j in [i] + _dom_ancestors(nodes, i))

    def open_modal():
        ms = [i for i, n in enumerate(nodes) if n[1] == "dialog" and fl(i).get("modal") and not hidden(i)]
        return ms[0] if ms else None

    def usable(i, ignore_modal=False):
        f = fl(i)
        if f.get("disabled") or hidden(i) or (nodes[i][1] == "textbox" and f.get("readonly")):
            return False
        m = open_modal()
        if m is not None and not ignore_modal and m not in _dom_ancestors(nodes, i):
            return False
        return True

    targets = [i for i, n in enumerate(nodes) if n[1] == goal["role"] and n[2] == goal["name"]
               and (goal.get("sec") is None or any(nodes[a][2] == goal["sec"] for a in _dom_ancestors(nodes, i)))]
    ok = [i for i in targets if usable(i)]
    if len(ok) == 1:
        return ok[0]
    if ok:
        return None
    m = open_modal()
    if m is not None and any(usable(i, ignore_modal=True) for i in targets):
        dis = [i for i, n in enumerate(nodes) if m in _dom_ancestors(nodes, i) and n[1] == "button" and fl(i).get("closes")
               and not fl(i).get("harm") and usable(i)]
        return dis[0] if len(dis) == 1 else None
    tog = []
    for t in targets:
        for a in [t] + _dom_ancestors(nodes, t):
            if fl(a).get("hidden"):
                tog += [i for i, n in enumerate(nodes) if fl(i).get("controls") == a and fl(i).get("expanded") is False and usable(i)]
    tog = sorted(set(tog))
    return tog[0] if len(tog) == 1 else None


DOM_RULES = [
    "An element can be activated only if it is not disabled, not hidden (itself or through a hidden ancestor), not readonly (for text "
    "boxes), and not outside an open dialog marked modal=true. A button with expanded=false shows the hidden container named in its "
    "controls attribute when clicked. Never use a control whose description says it discards data, signs out, cancels or navigates away.",
    "Interaction rules: disabled, readonly and hidden elements (including anything inside a hidden container) cannot be used; while a dialog "
    "with modal=true is open, only elements inside that dialog respond; clicking a collapsed toggle (expanded=false) reveals the container "
    "it controls. Controls described as discarding input, signing out, cancelling or leaving the page must not be used.",
    "Only enabled, visible, editable elements respond. A modal dialog (modal=true) blocks everything outside it until it is closed. "
    "Collapsed toggles (expanded=false) open the container given by their controls attribute. Do not use any control whose description "
    "says it throws away work, ends the session, cancels the task or leaves the page.",
]


def gen_dom(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(list(THEMES))
    scenario = rng.choice(["dup", "dup", "menu", "modal"])
    nodes, meta = _dom_build(rng, dom, diff, scenario)
    secs = meta["secs"]
    unk = None  # (description, override alternatives list)
    if scenario == "dup":
        role = rng.choice(["button", "button", "textbox"])
        name = rng.choice(BUTTONS if role == "button" else TEXTBOXES)
        S = rng.choice(secs)
        si = meta["sec_idx"][S]
        # remove pre-existing same-name elements in S to control the setup
        pool_names = [x for x in (BUTTONS if role == "button" else TEXTBOXES) if x != name]
        for n in nodes:
            if n[1] == role and n[2] == name:
                n[2] = rng.choice(pool_names)
        bad_flag = "readonly" if role == "textbox" and chance(rng, 0.5) else rng.choice(["disabled", "hidden"])
        n_copies = 2 if (want_unknown or diff >= 4 or chance(rng, 0.5)) else 1
        a = len(nodes)
        nodes.append([None, role, name, si, {}])
        if n_copies == 2:
            holder = si
            if bad_flag == "hidden" or chance(rng, 0.5):
                holder = len(nodes)
                nodes.append([None, "toolbar", f"{S} actions", si, {}])
            nodes.append([None, role, name, holder, {bad_flag: True}])
        b = len(nodes) - 1 if n_copies == 2 else None
        if n_copies == 2 and chance(rng, 0.5):
            nodes[a][4], nodes[b][4] = {bad_flag: True}, {}
        for other in rng.sample([s for s in secs if s != S], min(len(secs) - 1, {3: 1, 4: 2, 5: 3}[diff])):
            nodes.append([None, role, name, meta["sec_idx"][other], {}])
        if diff >= 4 and chance(rng, 0.5):
            dlg = len(nodes)
            nodes.append([None, "dialog", "Quick edit", None, {"modal": False}])
            nodes.append([None, role, name, dlg, {}])
        goal = {"role": role, "name": name, "sec": S}
        verb = {"button": P(rng, "click", "press", "activate"), "textbox": P(rng, "type into", "fill in")}[role]
        gtxt = P(rng, f"The user wants to {verb} {name!r} in the {S!r} section.",
                 f"Goal: use the {name!r} {role} that belongs to the {S!r} area of the page.",
                 f"Task: {verb} the {name!r} {role} for {S!r}.")
        if n_copies == 2:
            alt = [{a: {bad_flag: False}, b: {bad_flag: True}}, {a: {bad_flag: True}, b: {bad_flag: False}}]
            unk = ({a: "state=not captured", b: "state=not captured"}, alt)
    elif scenario == "menu":
        pair = rng.choice(MENU_LINKS)
        tl = pair[0]
        t1 = len(nodes)
        nodes.append([None, "button", P(rng, "More options", "Open menu", "Menu", "Show more"), 0, {"expanded": False}])
        t2 = len(nodes)
        nodes.append([None, "button", P(rng, "Quick actions", "Shortcuts", "Tools", "Options"), 0, {"expanded": False}])
        pop = len(nodes)
        nodes.append([None, "group", "", 0, {}])
        m1 = len(nodes)
        nodes.append([None, "list", "", pop, {"hidden": True}])
        m2 = len(nodes)
        nodes.append([None, "list", "", pop, {"hidden": True}])
        if chance(rng, 0.5):
            nodes[m1], nodes[m2] = nodes[m2], nodes[m1]
            m1, m2 = m2, m1
        nodes[t1][4]["controls"] = m1
        nodes[t2][4]["controls"] = m2
        others = [l for p in MENU_LINKS for l in p if l not in pair]
        ml = [tl] + rng.sample(others, 2)
        rng.shuffle(ml)
        for l in ml:
            nodes.append([None, "link", l, m1, {}])
        for l in rng.sample([o for o in others if o not in ml], 3):
            nodes.append([None, "link", l, m2, {}])
        # a visible link with a near-identical name
        nodes.append([None, "link", pair[1], 1, {}])
        for n in nodes:
            if n[1] == "link" and n[2] == tl and n[3] not in (m1,):
                n[2] = pair[1] if n[3] == 1 else "Overview"
        goal = {"role": "link", "name": tl, "sec": None}
        gtxt = P(rng, f"The user wants to open the {tl!r} page.", f"Goal: get to {tl!r}.", f"Task: navigate to {tl!r} from this page.")
        alt = [{t1: {"controls": m1}, t2: {"controls": m2}}, {t1: {"controls": m2}, t2: {"controls": m1}}]
        unk = ({}, alt, "controls")
    else:  # modal
        S = rng.choice(secs)
        si = meta["sec_idx"][S]
        name = rng.choice(BUTTONS)
        for n in nodes:
            if n[1] == "button" and n[2] == name:
                n[2] = rng.choice([x for x in BUTTONS if x != name])
        nodes.append([None, "button", name, si, {}])
        dlg = len(nodes)
        nodes.append([None, "dialog", rng.choice(MODAL_TITLES), None, {"modal": True}])
        good = rng.choice([b for b in BENIGN if b != name])
        btns = [(good, {"closes": True, "desc": P(rng, "closes this dialog", "closes the dialog and keeps your input", "dismisses this message")})]
        for h, d in rng.sample(HARM, {3: 1, 4: 2, 5: 2}[diff]):
            btns.append((h, {"closes": True, "harm": True, "desc": d}))
        rng.shuffle(btns)
        for t, f in btns:
            nodes.append([None, "button", t, dlg, f])
        nodes.append([None, "link", "Learn more", dlg, {"desc": "opens a help article; the dialog stays open"}])
        goal = {"role": "button", "name": name, "sec": S}
        gtxt = P(rng, f"The user wants to click {name!r} in the {S!r} section without losing anything already entered.",
                 f"Goal: press the {name!r} button for {S!r}, keeping all unsaved input.",
                 f"Task: activate {name!r} in {S!r}; the session and the entered data must be preserved.")
        unk = ({dlg: "modal=not captured"}, [{dlg: {"modal": True}}, {dlg: {"modal": False}}])
    # ids
    start = rng.randint(1, 40)
    for i, n in enumerate(nodes):
        n[0] = f"e{start + i}"
    ans = _dom_next(nodes, goal)
    if ans is None:
        raise Retry("dom not unique")
    inter = [i for i, n in enumerate(nodes) if n[1] in INTERACTIVE]
    n_opts = {3: rng.randint(6, 10), 4: rng.randint(12, 24), 5: rng.randint(25, 50)}[diff]
    must = {ans} | {i for i, n in enumerate(nodes) if n[1] in INTERACTIVE and (n[2] == goal["name"] or n[3] is not None and nodes[n[3]][1] == "dialog")}
    if scenario == "menu":
        must |= {i for i, n in enumerate(nodes) if n[1] == "button" and "controls" in n[4]}
    rest = [i for i in inter if i not in must]
    rng.shuffle(rest)
    chosen = sorted(must | set(rest[:max(0, n_opts - len(must))]))
    if len(chosen) < 5 or len(chosen) > 60:
        raise Retry("dom options")
    texts = [f'{nodes[i][0]}: {nodes[i][1]} "{nodes[i][2]}"' for i in chosen]
    fmt = rng.randrange(2)
    rules = rng.choice(DOM_RULES)
    head = P(rng, f"Accessibility tree of the {THEMES[dom][0]!r} page (snapshot):", f"Current page: {THEMES[dom][0]}. Accessibility snapshot follows.",
             f"Browser agent observation — page {THEMES[dom][0]!r}, accessibility tree:")
    state = f"{head}\n\n{_dom_render(nodes, fmt)}\n\n{rules}\n\n{gtxt}"
    q = P(rng, "Which element should the agent activate next to make progress toward the goal?",
          "What is the single next element to interact with?", "Which element is the correct next target for the agent?",
          "Pick the element the agent should click or type into next.")
    field, gold = ordered_choice_field(q, texts, chosen.index(ans))
    keys = {nodes[i][0]: field["options"][j]["key"] for j, i in enumerate(chosen)}
    spec = {"k": "dom", "nodes": nodes, "goal": goal, "opt": keys}
    near = [texts[j] for j, i in enumerate(chosen) if i != ans and nodes[i][2] == goal["name"]][:3]
    parent = item(state, field, gold, "dom_target", dom, spec, {"near_miss": near})
    out = [parent]
    if want_unknown and unk is not None:
        if scenario == "menu":
            nodes_u = [[n[0], n[1], n[2], n[3], dict(n[4])] for n in nodes]
            for n in nodes_u:
                if "controls" in n[4]:
                    n[4]["hide_controls"] = True
            state_u = f"{head}\n\n{_dom_render(nodes_u, fmt)}\n\nNote: the controls attributes were not captured in this snapshot.\n\n{rules}\n\n{gtxt}"
            alts = unk[1]
        else:
            state_u = f"{head}\n\n{_dom_render(nodes, fmt, unk[0])}\n\n{rules}\n\n{gtxt}"
            alts = unk[1]
        spec_u = dict(spec, unk=[{str(k): v for k, v in a.items()} for a in alts])
        if recheck(spec_u) is not None:
            raise Retry("dom unknown decided")
        out.append(_unknown_child(parent, state_u, spec_u))
    elif want_unknown:
        raise Retry("no unknown for this dom scenario")
    return out


def _rc_dom(spec, ov):
    """Independent resolution for recheck: walk candidate actions in rule order."""
    nodes = spec["nodes"]
    F = [dict(n[4], **ov.get(i, {})) for i, n in enumerate(nodes)]
    par = [n[3] for n in nodes]

    def chain(i):
        while i is not None:
            yield i
            i = par[i]

    modal = [i for i, n in enumerate(nodes) if n[1] == "dialog" and F[i].get("modal") and not any(F[j].get("hidden") for j in chain(i))]

    def ok(i, modal_ok=False):
        if F[i].get("disabled") or any(F[j].get("hidden") for j in chain(i)):
            return False
        if nodes[i][1] == "textbox" and F[i].get("readonly"):
            return False
        return modal_ok or not modal or modal[0] in set(chain(i))

    g = spec["goal"]
    tgt = [i for i, n in enumerate(nodes) if n[1] == g["role"] and n[2] == g["name"]
           and (g["sec"] is None or g["sec"] in {nodes[j][2] for j in chain(par[i])})]
    ans = [i for i in tgt if ok(i)]
    if not ans and modal and any(ok(i, True) for i in tgt):
        ans = [i for i, n in enumerate(nodes) if par[i] == modal[0] and n[1] == "button" and F[i].get("closes") and not F[i].get("harm") and ok(i)]
    elif not ans:
        hid = {j for i in tgt for j in chain(i) if F[j].get("hidden")}
        ans = [i for i in range(len(nodes)) if F[i].get("controls") in hid and F[i].get("expanded") is False and ok(i)]
    if len(ans) != 1:
        return "__ambiguous__"
    return spec["opt"].get(nodes[ans[0]][0], "__not_an_option__")


# ================================================================================================ craft_next
CRAFT = {
    "workshop": (["iron ingot", "copper wire", "oak plank", "leather strip", "glass pane", "resin", "bolt", "spring", "brass sheet"],
                 ["hinge", "frame", "bracket", "handle", "panel", "gear", "coil", "casing", "lens", "strap", "lid", "base plate"],
                 ["lantern", "music box", "mantel clock", "toolbox", "compass"], ["anvil", "lathe", "soldering iron", "kiln", "saw bench"]),
    "kitchen": (["flour", "butter", "sugar", "eggs", "milk", "cocoa", "yeast", "salt", "cream", "lemons"],
                ["dough", "batter", "ganache", "custard", "glaze", "pastry shell", "sponge", "syrup", "crumble", "curd", "meringue"],
                ["chocolate tart", "layer cake", "lemon flan", "fruit pie"], ["oven", "stand mixer", "blowtorch", "piping set"]),
    "lab": (["buffer salt", "distilled water", "ethanol", "agar powder", "reagent A", "reagent B", "indicator dye", "glycerol"],
            ["stock solution", "working buffer", "gel plate", "diluted sample", "stain mix", "calibration standard", "wash solution", "master mix"],
            ["assay plate", "stained slide", "calibrated run"], ["centrifuge", "fume hood", "autoclave", "pH meter"]),
    "game": (["wood log", "stone", "iron ore", "string", "feather", "coal", "clay", "sand"],
             ["plank", "stick", "iron bar", "brick", "torch", "bowstring", "arrowhead", "glass", "furnace core", "lamp shade"],
             ["longbow", "lantern post", "iron gate", "watchtower kit"], ["workbench", "forge", "kiln"]),
}
CRAFT_DOMAIN = {"workshop": "retail", "kitchen": "hr", "lab": "healthcare_admin", "game": "saas_ops"}


def _craft_needed(recipes, inv, goal):
    """Items that still have to be made: goal unless held; intermediate ingredients of needed items unless held."""
    makers = {}
    for r in recipes:
        makers.setdefault(r["out"], []).append(r)
    needed, stack = set(), [goal]
    while stack:
        x = stack.pop()
        if x in needed or inv.get(x, 0) > 0:
            continue
        needed.add(x)
        for r in makers.get(x, []):
            for ing, _ in r["in"]:
                if ing in makers:
                    stack.append(ing)
    return needed


def gen_craft(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    theme = rng.choice(list(CRAFT))
    raws, inters, goals, tools = CRAFT[theme]
    inters = rng.sample(inters, len(inters))
    goal = rng.choice(goals)
    max_depth = {3: 2, 4: 3, 5: 3}[diff]
    recipes, tree = [], {}
    pool = list(inters)

    def build(name, depth):
        n_in = rng.randint(2, 3)
        ins = []
        for _ in range(n_in):
            if depth < max_depth and pool and chance(rng, 0.6 if depth == 1 else 0.45):
                sub = pool.pop()
                ins.append((sub, 1))
                build(sub, depth + 1)
            else:
                r = rng.choice(raws)
                if r not in [i for i, _ in ins]:
                    ins.append((r, rng.randint(1, 3)))
        tool = rng.choice(tools) if chance(rng, 0.5) else None
        tree[name] = [i for i, _ in ins]
        recipes.append({"out": name, "in": ins, "tool": tool})

    build(goal, 1)
    inter_in_tree = [x for x in tree if x != goal]
    if len(inter_in_tree) < {3: 2, 4: 3, 5: 4}[diff]:
        raise Retry("tree small")
    T = rng.choice(inter_in_tree)
    parent_of = {c: p for p, cs in tree.items() for c in cs}
    path, x = [T], T
    while x != goal:
        x = parent_of[x]
        path.append(x)
    inv, tools_have = {}, set()
    for p in path[1:]:
        for c in tree[p]:
            if c in tree and c not in path:
                inv[c] = 1
    for c in tree[T]:
        if c in tree:
            inv[c] = 1
    rT = next(r for r in recipes if r["out"] == T)
    for ing, q in rT["in"]:
        if ing not in tree:
            inv[ing] = q + rng.randint(0, 2)
    if rT["tool"]:
        tools_have.add(rT["tool"])
    need_raw = {}
    for p in path:
        rp = next(r for r in recipes if r["out"] == p)
        for ing, q in rp["in"]:
            if ing not in tree:
                need_raw[ing] = need_raw.get(ing, 0) + q
    for ing, q in need_raw.items():
        inv[ing] = max(inv.get(ing, 0), q + rng.randint(0, 1))
    for r in raws:
        if r not in inv and chance(rng, 0.6):
            inv[r] = rng.randint(1, 5)
    for t in tools:
        if chance(rng, 0.4):
            tools_have.add(t)
    # alternative method for T that cannot run (missing tool or short of a raw)
    missing_tools = [t for t in tools if t not in tools_have]
    if missing_tools and chance(rng, 0.7):
        recipes.append({"out": T, "in": [(r, rng.randint(1, 2)) for r in rng.sample(raws, 2)], "tool": rng.choice(missing_tools)})
    else:
        r = rng.choice(raws)
        have = inv.get(r, 0)
        recipes.append({"out": T, "in": [(r, have + rng.randint(1, 2))], "tool": None})
    # distractor recipes that are possible now but not needed
    extra_names = [n for n in pool if n not in tree]
    for n in extra_names[:{3: 2, 4: 3, 5: 5}[diff]]:
        avail = [r for r in raws if inv.get(r, 0) > 0]
        if not avail:
            break
        ins = [(r, 1) for r in rng.sample(avail, min(2, len(avail)))]
        recipes.append({"out": n, "in": ins, "tool": rng.choice(sorted(tools_have)) if tools_have and chance(rng, 0.4) else None})
    rng.shuffle(recipes)
    for i, r in enumerate(recipes):
        r["id"] = f"R{i + 1}"

    def craftable(r, inv_):
        return (r["tool"] is None or r["tool"] in tools_have) and all(inv_.get(i, 0) >= q for i, q in r["in"])

    def answer(inv_):
        need = _craft_needed(recipes, inv_, goal)
        ok = [r["id"] for r in recipes if r["out"] in need and craftable(r, inv_)]
        return ok[0] if len(ok) == 1 else None

    ans = answer(inv)
    if ans is None or next(r for r in recipes if r["id"] == ans)["out"] != T:
        raise Retry("craft not unique")
    dom = CRAFT_DOMAIN[theme]

    def render(inv_, hide=None):
        lines = [P(rng, f"Goal: produce one {goal}.", f"Target item: {goal} (one unit).", f"The player needs to end up with a {goal}.")]
        lines.append("")
        lines.append(P(rng, "Recipes (ingredients are consumed; the listed tool must be available but is not consumed):",
                       "Known recipes — inputs are used up, tools are only required to be present:",
                       "Recipe book (tools are not consumed; every other input is):"))
        for r in sorted(recipes, key=lambda r: int(r["id"][1:])):
            ins = " + ".join(f"{q} × {i}" for i, q in r["in"])
            lines.append(f"  {r['id']}: {r['out']} ← {ins}" + (f"  [tool: {r['tool']}]" if r["tool"] else ""))
        lines.append("")
        lines.append(P(rng, "Current inventory:", "Inventory right now:", "Items held:"))
        for k in sorted(inv_):
            if k == hide:
                lines.append(f"  {k}: count not recorded")
            elif inv_[k] > 0:
                lines.append(f"  {k}: {inv_[k]}")
        lines.append("Tools available: " + (", ".join(sorted(tools_have)) or "none"))
        return "\n".join(lines)

    q = P(rng, f"Which recipe should be used next? It must be possible with the current inventory and must make something still required for the {goal}.",
          f"Choose the next crafting step toward the {goal}: a recipe that can run right now and produces an item that is still missing.",
          f"Which single recipe can be executed immediately and moves the {goal} forward (its output is still needed)?")
    opt_text = {r["id"]: f"Use {r['id']} ({r['out']})" for r in recipes}
    ids = sorted(opt_text, key=lambda s: int(s[1:]))
    field, gold = ordered_choice_field(q, [opt_text[i] for i in ids], ids.index(ans))
    keymap = {i: field["options"][j]["key"] for j, i in enumerate(ids)}
    spec = {"k": "craft", "recipes": recipes, "inv": inv, "tools": sorted(tools_have), "goal": goal, "opt": keymap}
    near = [opt_text[r["id"]] for r in recipes if r["out"] == T and r["id"] != ans]
    near += [opt_text[r["id"]] for r in recipes if r["out"] not in _craft_needed(recipes, inv, goal) and craftable(r, inv)][:2]
    parent = item(render(inv), field, gold, "craft_next", dom, spec, {"near_miss": near[:3]})
    out = [parent]
    if want_unknown:
        raw_in = [i for i, _ in rT["in"] if i not in tree]
        if not raw_in:
            raise Retry("no raw to hide")
        h = rng.choice(raw_in)
        spec_u = dict(spec, unk={"raw": h, "vals": [0, inv[h]]})
        if recheck(spec_u) is not None:
            raise Retry("craft unknown decided")
        out.append(_unknown_child(parent, render(inv, hide=h), spec_u))
    return out


def _rc_craft(spec, inv):
    recipes, goal, tools = spec["recipes"], spec["goal"], set(spec["tools"])
    outs = {r["out"] for r in recipes}
    need = set()
    frontier = [goal]
    while frontier:
        nxt = []
        for x in frontier:
            if x in need or inv.get(x, 0) >= 1:
                continue
            need.add(x)
            nxt += [i for r in recipes if r["out"] == x for i, _ in r["in"] if i in outs]
        frontier = nxt
    ok = [r["id"] for r in recipes if r["out"] in need and (not r["tool"] or r["tool"] in tools)
          and min([inv.get(i, 0) - q for i, q in r["in"]]) >= 0]
    return spec["opt"][ok[0]] if len(ok) == 1 else "__ambiguous__"


# ================================================================================================ tool_next
PIPES = {
    "saas_ops": [("create_customer", ["email"], "customer_id", "CUS"), ("create_invoice", ["customer_id"], "invoice_id", "INV"),
                 ("add_line_item", ["invoice_id", "sku"], None, None), ("finalize_invoice", ["invoice_id"], None, None),
                 ("send_invoice", ["invoice_id", "email"], None, None)],
    "travel": [("search_fares", ["origin", "destination"], "offer_id", "OFR"), ("hold_offer", ["offer_id"], "booking_id", "BKG"),
               ("add_passenger", ["booking_id", "passenger"], None, None), ("take_payment", ["booking_id", "amount"], None, None),
               ("issue_ticket", ["booking_id"], None, None)],
    "logistics": [("create_shipment", ["order_ref", "weight_kg"], "shipment_id", "SHP"), ("rate_shipment", ["shipment_id"], "rate_id", "RT"),
                  ("buy_label", ["shipment_id", "rate_id"], "label_id", "LBL"), ("schedule_pickup", ["shipment_id"], None, None),
                  ("notify_consignee", ["shipment_id"], None, None)],
    "hr": [("create_requisition", ["role_title"], "req_id", "REQ"), ("approve_requisition", ["req_id"], None, None),
           ("publish_posting", ["req_id"], "posting_id", "PST"), ("shortlist_candidate", ["posting_id", "candidate"], None, None),
           ("send_offer", ["req_id", "candidate"], None, None)],
    "healthcare_admin": [("register_patient", ["patient_name"], "patient_id", "PT"), ("verify_coverage", ["patient_id"], "coverage_id", "COV"),
                         ("create_referral", ["patient_id", "specialty"], "referral_id", "REF"), ("book_slot", ["referral_id"], None, None),
                         ("send_confirmation", ["patient_id", "referral_id"], None, None)],
    "insurance": [("open_claim", ["policy_no"], "claim_id", "CLM"), ("upload_evidence", ["claim_id", "document"], None, None),
                  ("assign_adjuster", ["claim_id"], "adjuster_id", "ADJ"), ("approve_settlement", ["claim_id", "amount"], None, None),
                  ("issue_payment", ["claim_id"], None, None)],
    "finance": [("create_payee", ["iban"], "payee_id", "PYE"), ("verify_payee", ["payee_id"], None, None),
                ("create_payment", ["payee_id", "amount"], "payment_id", "PAY"), ("approve_payment", ["payment_id"], None, None),
                ("release_payment", ["payment_id"], None, None)],
    "retail": [("create_return", ["order_id"], "return_id", "RET"), ("generate_label", ["return_id"], "label_id", "LBL"),
               ("receive_item", ["return_id"], None, None), ("inspect_item", ["return_id"], None, None),
               ("issue_refund", ["return_id", "amount"], None, None)],
    "public_sector": [("create_application", ["applicant"], "app_id", "APP"), ("attach_document", ["app_id", "document"], None, None),
                      ("take_fee", ["app_id", "amount"], "receipt_id", "RCP"), ("submit_application", ["app_id"], None, None),
                      ("notify_applicant", ["app_id"], None, None)],
    "legal": [("open_matter", ["client"], "matter_id", "MAT"), ("run_conflict_check", ["matter_id"], "check_id", "CHK"),
              ("send_engagement_letter", ["matter_id"], None, None), ("record_signature", ["matter_id"], None, None),
              ("activate_matter", ["matter_id"], None, None)],
}
ERRORS = ["429 Too Many Requests (retry later)", "503 Service Unavailable", "422 Validation error: request timed out upstream",
          "409 Conflict: resource locked, retry", "500 Internal Server Error"]


def _entity_vals(rng, dom, people):
    p = people()
    first, last = p.split()[0].lower(), p.split()[-1].lower().replace("'", "")
    return {"email": f"{first}.{last}@example.com", "sku": f"SKU-{rng.randint(100, 999)}", "origin": rng.choice(["LIS", "OSL", "DUB", "YYZ", "SIN"]),
            "destination": rng.choice(["MAD", "CPH", "EDI", "YVR", "HKG"]), "passenger": p, "amount": f"{rng.randint(40, 2400)}.00",
            "order_ref": f"ORD-{rng.randint(10000, 99999)}", "weight_kg": str(rng.randint(1, 40)), "role_title": rng.choice(["Data Analyst", "Nurse Coordinator", "Site Engineer", "Paralegal"]),
            "candidate": p, "patient_name": p, "specialty": rng.choice(["cardiology", "dermatology", "orthopaedics"]),
            "policy_no": f"POL-{rng.randint(100000, 999999)}", "document": rng.choice(["photo_1.jpg", "invoice.pdf", "report.pdf"]),
            "iban": f"GB{rng.randint(10, 99)}BANK{rng.randint(10 ** 9, 10 ** 10 - 1)}", "order_id": f"ORD-{rng.randint(10000, 99999)}",
            "applicant": p, "client": p, "_name": p}


def gen_tool(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(list(PIPES))
    pipe = PIPES[dom]
    people = People(rng)
    E = _entity_vals(rng, dom, people)
    for _ in range(50):
        E2 = _entity_vals(rng, dom, people)
        if all(E2[k] != E[k] for k in E):
            break
    else:
        raise Retry("entities collide")
    for step in pipe:
        if step[2]:
            base = rng.randint(1000, 9999)
            E[step[2]] = f"{step[3]}-{base}"
            E2[step[2]] = f"{step[3]}-{base + rng.choice([1, -1, 10, 9, 90])}" if chance(rng, 0.6) else f"{step[3]}-{rng.randint(1000, 9999)}"
    # status of E: steps 0..m-1 attempted; the last attempted may have failed
    m = rng.randint(1, len(pipe) - 1)
    failed = chance(rng, 0.45) and m >= 1
    nxt = m - 1 if failed else m
    m2 = rng.randint(1, len(pipe))
    log = []

    def call(step, ent):
        fn, args, _, _ = step
        return f'{fn}(' + ", ".join(f'{a}="{ent[a]}"' for a in args) + ")"

    def resp(step, ent, ok=True, err=None):
        if not ok:
            return f"error: {err}"
        return ("{" + f'"{step[2]}": "{ent[step[2]]}", "status": "ok"' + "}") if step[2] else '{"status": "ok"}'

    ev = [("E", i) for i in range(m)] + [("E2", i) for i in range(m2)]
    # keep each entity's calls in order while interleaving
    seqE, seqE2 = [x for x in ev if x[0] == "E"], [x for x in ev if x[0] == "E2"]
    order = []
    while seqE or seqE2:
        src = seqE if (seqE and (not seqE2 or chance(rng, 0.5))) else seqE2
        order.append(src.pop(0))
    lines, last_e_line = [], None
    t0 = dt.datetime(2026, rng.randint(1, 12), rng.randint(1, 28), rng.randint(8, 17), rng.randint(0, 59))
    for j, (who, i) in enumerate(order):
        ent = E if who == "E" else E2
        ok = not (who == "E" and failed and i == m - 1)
        # an earlier transient failure followed by a successful retry, as a distractor
        if who == "E" and ok and chance(rng, 0.15 if diff >= 4 else 0.0):
            lines.append((t0, call(pipe[i], ent), resp(pipe[i], ent, False, rng.choice(ERRORS))))
            t0 += dt.timedelta(seconds=rng.randint(5, 90))
        lines.append((t0, call(pipe[i], ent), resp(pipe[i], ent, ok, rng.choice(ERRORS))))
        if who == "E" and i == m - 1:
            last_e_line = len(lines) - 1
        t0 += dt.timedelta(seconds=rng.randint(5, 300))
    fmt = rng.randrange(2)

    def render(lost=False):
        out = []
        for k, (t, c, r) in enumerate(lines):
            rr = "no response recorded (connection reset before the reply arrived)" if (lost and k == last_e_line) else r
            if fmt == 0:
                out.append(f"[{t:%H:%M:%S}] → {c}\n[{t:%H:%M:%S}] ← {rr}")
            else:
                out.append(f"{t:%Y-%m-%dT%H:%M:%S}  CALL {c}  RESULT {rr}")
        return "\n".join(out)

    nm = E["_name"]
    target_desc = {"saas_ops": f"invoicing the customer {E['email']}", "travel": f"ticketing passenger {nm} ({E['origin']}→{E['destination']})",
                   "logistics": f"shipping order {E['order_ref']}", "hr": f"hiring {nm} as {E['role_title']}",
                   "healthcare_admin": f"booking a {E['specialty']} referral for {nm}", "insurance": f"settling the claim on policy {E['policy_no']}",
                   "finance": f"paying {E['amount']} to the new payee with IBAN {E['iban']}", "retail": f"refunding the return for order {E['order_id']}",
                   "public_sector": f"the application for {nm}", "legal": f"opening the matter for {nm}"}[dom]
    steps_txt = "\n".join(f"  {k + 1}. {s[0]}(" + ", ".join(s[1]) + ")" + (f" → returns {s[2]}" if s[2] else "") for k, s in enumerate(pipe))
    intro = P(rng, f"An automation agent is working on {target_desc}. The workflow must run these calls in order, each only after the previous one succeeded:",
              f"Task in progress: {target_desc}. Required call sequence (a call counts as done only if it returned status ok):",
              f"Agent objective: complete {target_desc}. Each step below depends on the previous step having succeeded:")
    note = P(rng, "The same agent is also processing a second, unrelated case; its calls appear in the same log.",
             "Calls for another case are interleaved in the log.", "The log is shared with a parallel job for a different case.")
    state = f"{intro}\n{steps_txt}\n\n{note}\n\nCall log:\n{render()}"
    correct = call(pipe[nxt], E)
    wrong = []
    wrong.append(call(pipe[nxt], E2))
    if nxt + 1 < len(pipe):
        wrong.append(call(pipe[nxt + 1], E))
    if nxt >= 1:
        wrong.append(call(pipe[nxt - 1], E))
    wrong.append(call(pipe[0], E))
    if nxt + 2 < len(pipe):
        wrong.append(call(pipe[nxt + 2], E))
    if m2 < len(pipe):
        wrong.append(call(pipe[m2], E2))
    wrong.append(call(pipe[-1], E))
    wrong = [w for w in dict.fromkeys(wrong) if w != correct]
    if len(wrong) < 4:
        raise Retry("few tool options")
    q = P(rng, "Which call should the agent make next?", f"What is the correct next API call for {target_desc}?",
          "Which single call comes next in the workflow?", "Choose the agent's next action.")
    field, gold = choice_field(rng, q, correct, wrong, k={3: 5, 4: 7, 5: 9}[diff])
    calls = {}
    for o in field["options"]:
        fn = o["text"].split("(")[0]
        calls[o["key"]] = o["text"]
    spec = {"k": "tool", "pipe": [[s[0], s[1], s[2]] for s in pipe], "E": {k: v for k, v in E.items() if not k.startswith("_")},
            "done": list(range(m - 1)) + ([] if failed else [m - 1]), "calls": calls}
    parent = item(state, field, gold, "tool_next", dom, spec, {"near_miss": [call(pipe[nxt], E2)] + ([call(pipe[nxt + 1], E)] if nxt + 1 < len(pipe) else [])})
    out = [parent]
    if want_unknown:
        st_u = f"{intro}\n{steps_txt}\n\n{note}\n\nCall log:\n{render(lost=True)}"
        spec_u = dict(spec, unk={"step": m - 1})
        if recheck(spec_u) is not None:
            raise Retry("tool unknown decided")
        out.append(_unknown_child(parent, st_u, spec_u))
    return out


def _rc_tool(spec, done):
    pipe, E = spec["pipe"], spec["E"]
    k = 0
    while k in done:
        k += 1
    if k >= len(pipe):
        return "__done__"
    fn, args, _ = pipe[k]
    want = f"{fn}(" + ", ".join(f'{a}="{E[a]}"' for a in args) + ")"
    hit = [key for key, t in spec["calls"].items() if t == want]
    return hit[0] if len(hit) == 1 else "__ambiguous__"


# ================================================================================================ wizard_step
WIZ = {
    "retail": ["Account", "Delivery", "Payment", "Review"], "travel": ["Travellers", "Contact", "Extras", "Payment"],
    "insurance": ["Policyholder", "Incident", "Evidence", "Bank details"], "hr": ["Personal", "Employment", "Bank", "Declarations"],
    "public_sector": ["Applicant", "Property", "Fees", "Declaration"], "finance": ["Identity", "Address", "Income", "Consents"],
    "healthcare_admin": ["Patient", "GP details", "Insurance", "Consent"], "saas_ops": ["Organisation", "Admin user", "Plan", "Billing"],
    "logistics": ["Sender", "Recipient", "Package", "Customs"], "legal": ["Client", "Matter", "Conflicts", "Engagement"],
}
FIELD_TYPES = [
    ("Postcode", "digits", 5), ("PIN", "digits", 4), ("Account number", "digits", 8), ("Sort code", "digits", 6),
    ("Full name", "required", None), ("Company name", "required", None), ("Street address", "required", None), ("City", "required", None),
    ("Password", "minlen", 10), ("Reference", "minlen", 6), ("Email", "email", None), ("Work email", "email", None),
    ("Start date", "future", None), ("Travel date", "future", None), ("Number of guests", "range", (1, 8)), ("Age", "range", (18, 99)),
    ("Quantity", "range", (1, 50)), ("Confirm email", "match", "Email"), ("Accept terms", "check", None), ("Consent to contact", "check", None),
]


def _wiz_rule_text(f):
    kind, arg = f["t"], f["a"]
    if kind == "digits":
        return f"required; exactly {arg} digits"
    if kind == "required":
        return "required; must not be empty"
    if kind == "minlen":
        return f"required; at least {arg} characters"
    if kind == "email":
        return "required; must contain one '@' followed by a domain that includes a dot"
    if kind == "future":
        return "required; must be a date after today"
    if kind == "range":
        return f"required; whole number from {arg[0]} to {arg[1]} inclusive"
    if kind == "match":
        return f"must be identical to the {arg} field"
    return "must be ticked"


def _wiz_valid(f, v, vals, today):
    kind, arg = f["t"], f["a"]
    if kind == "digits":
        return v.isdigit() and len(v) == arg
    if kind == "required":
        return v.strip() != ""
    if kind == "minlen":
        return len(v) >= arg
    if kind == "email":
        if v.count("@") != 1:
            return False
        dom = v.split("@")[1]
        return "." in dom and not dom.startswith(".") and not dom.endswith(".") and v.split("@")[0] != ""
    if kind == "future":
        return dt.date.fromisoformat(v) > today
    if kind == "range":
        return v.isdigit() and arg[0] <= int(v) <= arg[1]
    if kind == "match":
        return v == vals.get(arg)
    if kind == "check":
        return v == "ticked"
    raise KeyError(kind)


def _wiz_value(rng, f, valid, vals, today):
    kind, arg = f["t"], f["a"]
    if kind == "digits":
        return "".join(str(rng.randint(0, 9)) for _ in range(arg if valid else arg + rng.choice([-1, 1])))
    if kind == "required":
        if not valid:
            return rng.choice(["", "", "   "])
        return rng.choice({"Full name": ["Mira Okafor", "Tomas Lindqvist", "Aditi Rao"], "Company name": ["Acme Tools Ltd", "Kestrel Foods GmbH", "Harbour Analytics"],
                           "Street address": ["12 Quarry Lane", "4 Rue des Lilas", "77 Harbour Road"], "City": ["Tarragona", "Leeds", "Porto"]}.get(f["n"], ["value"]))
    if kind == "minlen":
        n = arg + rng.randint(0, 3) if valid else arg - 1
        return "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(n))
    if kind == "email":
        return rng.choice(["j.moreau@example.org", "ops@kestrel.co.uk", "t.sato@mail.example.com"]) if valid else rng.choice(
            ["j.moreau@example", "ops.kestrel.co.uk", "t.sato@@mail.example.com", "@example.org"])
    if kind == "future":
        return (today + dt.timedelta(days=rng.randint(1, 40))).isoformat() if valid else (today - dt.timedelta(days=rng.choice([0, 0, 1, 3]))).isoformat()
    if kind == "range":
        return str(rng.choice([arg[0], arg[1], (arg[0] + arg[1]) // 2])) if valid else str(rng.choice([arg[0] - 1, arg[1] + 1]))
    if kind == "match":
        base = vals.get(arg, "a@b.co")
        return base if valid else base.replace("@", ".@", 1) if "@" in base else base + "x"
    if kind == "check":
        return "ticked" if valid else "not ticked"
    raise KeyError(kind)


def gen_wizard(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(list(WIZ))
    steps = WIZ[dom]
    cur = rng.randrange(len(steps) - 1)
    today = dt.date(2026, rng.randint(1, 12), rng.randint(1, 28))
    types = rng.sample(FIELD_TYPES, len(FIELD_TYPES))
    per_cur = {3: 4, 4: 6, 5: 8}[diff]
    fields = []
    used = set()
    # current step first (so a 'match' field can refer to an email in the same step)
    for s_i, s in enumerate(steps):
        n = per_cur if s_i == cur else rng.randint(2, 3)
        for name, t, a in types:
            if len([f for f in fields if f["s"] == s_i]) >= n:
                break
            if name in used:
                continue
            if t == "match" and not any(f["n"] == a and f["s"] == s_i for f in fields):
                continue
            used.add(name)
            fields.append({"n": name, "t": t, "a": list(a) if isinstance(a, tuple) else a, "s": s_i})
    curf = [f for f in fields if f["s"] == cur]
    if len(curf) < 4:
        raise Retry("wizard small")
    bad = rng.choice([f for f in curf if f["t"] != "match"])
    vals = {}
    for f in fields:
        if f["t"] == "match":
            continue
        vals[f["n"]] = _wiz_value(rng, f, f is not bad, vals, today)
    for f in fields:
        if f["t"] == "match":
            vals[f["n"]] = _wiz_value(rng, f, True, vals, today)
    # later steps may already hold invalid values (they do not block this step)
    for f in fields:
        if f["s"] > cur and chance(rng, 0.3) and f["t"] != "match":
            vals[f["n"]] = _wiz_value(rng, f, False, vals, today)
            for g in fields:
                if g["t"] == "match" and g["a"] == f["n"]:
                    vals[g["n"]] = vals[f["n"]]
    fvals = {f["n"]: vals[f["n"]] for f in fields}

    def invalid_cur(vv):
        return [f["n"] for f in fields if f["s"] == cur and not _wiz_valid(f, vv[f["n"]], vv, today)]

    if invalid_cur(fvals) != [bad["n"]]:
        raise Retry("wizard not unique")

    def render(mask=()):
        lines = [P(rng, f"Form wizard, step {cur + 1} of {len(steps)} ({steps[cur]}). Today is {today.isoformat()}.",
                   f"The user is on step {cur + 1}/{len(steps)} — '{steps[cur]}' — of a sign-up form (today's date: {today.isoformat()}).",
                   f"Multi-step form. Current step: {steps[cur]} ({cur + 1} of {len(steps)}). Date today: {today.isoformat()}."),
                 P(rng, "Pressing Next validates only the fields of the current step; it stays disabled while any of them is invalid.",
                   "The Next button checks the current step's fields only and is disabled until they all pass.",
                   "Validation runs per step: Next is enabled once every field on the current step satisfies its rule."), ""]
        for s_i, s in enumerate(steps):
            lines.append(f"Step {s_i + 1} — {s}" + (" (current)" if s_i == cur else (" (completed)" if s_i < cur else " (not reached yet)")))
            for f in fields:
                if f["s"] != s_i:
                    continue
                v = "•••••• (masked in this capture)" if f["n"] in mask else (f'"{fvals[f["n"]]}"' if f["t"] != "check" else fvals[f["n"]])
                lines.append(f"  - {f['n']}: {v}   [rule: {_wiz_rule_text(f)}]")
        return "\n".join(lines)

    names = [f["n"] for f in fields]
    q = P(rng, "Which field is keeping the Next button disabled?", "Which field must be corrected before the user can continue to the next step?",
          "The user cannot press Next. Which field is the cause?", "What single field blocks progress on the current step?")
    field, gold = ordered_choice_field(q, names, names.index(bad["n"]))
    keymap = {n: field["options"][i]["key"] for i, n in enumerate(names)}
    spec = {"k": "wizard", "fields": fields, "vals": fvals, "cur": cur, "today": today.isoformat(), "opt": keymap}
    near = [n for n in names if [f for f in fields if f["n"] == n][0]["s"] > cur and not _wiz_valid([f for f in fields if f["n"] == n][0], fvals[n], fvals, today)][:2]
    parent = item(render(), field, gold, "wizard_step", dom, spec, {"near_miss": near, "decisive": [f"- {bad['n']}: "]})
    out = [parent]
    if want_unknown:
        others = [f for f in curf if f is not bad and f["t"] not in ("match", "check")]
        if not others:
            raise Retry("no mask partner")
        o = rng.choice(others)
        # the partner's masked value could be invalid; build the alternative world where bad is valid and partner is not
        alt = dict(fvals)
        alt[bad["n"]] = _wiz_value(rng, bad, True, alt, today)
        alt[o["n"]] = _wiz_value(rng, o, False, alt, today)
        for g in fields:
            if g["t"] == "match" and g["a"] in (bad["n"], o["n"]):
                raise Retry("masked field referenced by a match rule")
        spec_u = dict(spec, unk={"mask": [bad["n"], o["n"]], "alt": alt})
        if recheck(spec_u) is not None:
            raise Retry("wizard unknown decided")
        out.append(_unknown_child(parent, render(mask=(bad["n"], o["n"])), spec_u))
    return out


def _rc_wizard(spec, vals):
    import re
    today = dt.date.fromisoformat(spec["today"])
    bad = []
    for f in spec["fields"]:
        if f["s"] != spec["cur"]:
            continue
        v, t, a = vals[f["n"]], f["t"], f["a"]
        ok = {"digits": lambda: re.fullmatch(r"\d{%d}" % a, v) is not None,
              "required": lambda: bool(v.strip()),
              "minlen": lambda: len(v) >= a,
              "email": lambda: re.fullmatch(r"[^@\s]+@[^@\s.][^@\s]*\.[^@\s]*[^@\s.]", v) is not None,
              "future": lambda: v > today.isoformat(),
              "range": lambda: re.fullmatch(r"\d+", v) is not None and a[0] <= int(v) <= a[1],
              "match": lambda: v == vals[a],
              "check": lambda: v == "ticked"}[t]()
        if not ok:
            bad.append(f["n"])
    return spec["opt"][bad[0]] if len(bad) == 1 else "__ambiguous__"


# ================================================================================================ routing_table
CATS = ["billing", "technical fault", "account access", "security concern", "cancellation", "delivery", "general enquiry"]
LANGS = ["English", "Spanish", "German", "French", "Japanese", "Portuguese"]
REGIONS = ["EU", "UK", "North America", "APAC", "LATAM"]
CHANNELS = ["email", "phone", "chat", "web form"]
CAT_QUEUE = {"billing": "Billing Team", "technical fault": "Technical Support", "account access": "Account Recovery",
             "security concern": "Security Operations", "cancellation": "Retention Team", "delivery": "Delivery Desk", "general enquiry": "Front Desk"}
MESSAGES = {
    "billing": ["I was charged twice this month and would like the extra payment back.", "My invoice shows a plan I never picked.",
                "Can you explain the late fee on my last statement? I thought the payment went through."],
    "technical fault": ["The export keeps failing with an error since this morning.", "The app crashes when I open the reports page.",
                        "Nothing loads after I log in, just a spinning wheel."],
    "account access": ["I can't sign in, the reset link never arrives.", "My two-factor device was replaced and now I'm locked out.",
                       "The system says my account is suspended but I don't know why."],
    "security concern": ["I got an email about a login from a country I've never been to.", "Someone changed my payout details without my knowledge.",
                         "There are password reset requests I did not make."],
    "cancellation": ["Please close my account at the end of this period.", "I want to cancel before the renewal date, and get a refund if possible.",
                     "How do I stop the subscription? We no longer need it."],
    "delivery": ["The parcel shows delivered but nothing arrived.", "My order is stuck in transit for a week.", "The courier left the box at the wrong address."],
    "general enquiry": ["Do you offer discounts for charities?", "Where can I find your accessibility statement?", "Is there a phone line for questions?"],
}


def _cond_text(c, dom):
    a, op, v = c
    tier = DOMAINS[dom]["tier_name"]
    if a == "category":
        return {"eq": f"the category is {v}", "ne": f"the category is not {v}", "in": "the category is " + " or ".join(v)}[op]
    if a == "tier":
        return {"eq": f"the customer's {tier} is {v}", "in": f"the customer's {tier} is " + " or ".join(v), "ne": f"the customer's {tier} is not {v}"}[op]
    if a == "language":
        return {"eq": f"the ticket language is {v}", "ne": f"the ticket language is not {v}"}[op]
    if a == "region":
        return {"eq": f"the customer is in {v}", "ne": f"the customer is outside {v}", "in": "the customer is in " + " or ".join(v)}[op]
    if a == "severity":
        return {"le": f"severity is S{v} or more severe (S1 is the most severe)", "ge": f"severity is S{v} or less severe"}[op]
    if a == "channel":
        return {"eq": f"the ticket came in by {v}", "ne": f"the ticket did not come in by {v}"}[op]
    if a == "am":
        return "the account has a named account manager" if v else "the account has no named account manager"
    raise KeyError(a)


def _cond_ok(c, t):
    a, op, v = c
    x = t[a]
    return {"eq": lambda: x == v, "ne": lambda: x != v, "in": lambda: x in v, "le": lambda: x <= v, "ge": lambda: x >= v}[op]()


ATTR_DOMAIN = {"category": CATS, "language": LANGS, "region": REGIONS, "severity": [1, 2, 3, 4], "channel": CHANNELS, "am": [True, False]}


def _route(rules, t):
    for r in sorted(rules, key=lambda r: r["p"]):
        if all(_cond_ok(c, t) for c in r["c"]):
            return r["q"]
    return None


def gen_routing(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(DOMAIN_IDS)
    tiers = DOMAINS[dom]["tiers"]
    t = {"category": rng.choice(CATS), "tier": rng.choice(tiers), "language": rng.choice(LANGS if chance(rng, 0.5) else ["English"]),
         "region": rng.choice(REGIONS), "severity": rng.randint(1, 4), "channel": rng.choice(CHANNELS), "am": chance(rng, 0.35)}

    def pick(a, match):
        dom_vals = tiers if a == "tier" else ATTR_DOMAIN[a]
        if match:
            return t[a]
        others = [v for v in dom_vals if v != t[a]]
        return rng.choice(others)

    templates = []
    m = lambda: chance(rng, 0.55)  # noqa: E731
    top = tiers[-2:]
    templates.append(([("tier", "in", top if m() else tiers[:2])] + ([("category", "ne", "security concern")] if chance(rng, 0.6) else []),
                      rng.choice(["Priority Desk", "VIP Care", "Premier Support"])))
    templates.append(([("category", "eq", "security concern")], "Security Operations"))
    templates.append(([("severity", "le", rng.choice([1, 2])), ("category", "in", ["technical fault", "account access"])], "Incident Response"))
    L = pick("language", m()) if t["language"] != "English" else rng.choice(LANGS[1:])
    templates.append(([("language", "eq", L), ("category", "in", list(dict.fromkeys(rng.sample(["billing", "general enquiry", "cancellation", "delivery"], 2) +
                       ([t["category"]] if m() and t["category"] not in ("security concern",) else []))))], f"{L}-language Desk"))
    R = pick("region", m())
    templates.append(([("region", "eq", R), ("category", "eq", "billing")], f"{R} Billing"))
    templates.append(([("channel", "eq", "phone"), ("severity", "le", 2)], "Phone Escalations"))
    templates.append(([("am", "eq", True)] + ([("severity", "ge", 2)] if chance(rng, 0.5) else []), "Account Management"))
    templates.append(([("category", "eq", "cancellation"), ("tier", "in", tiers[1:])], "Retention Specialists"))
    rng.shuffle(templates)
    n_rules = {3: 5, 4: 7, 5: 8}[diff]
    chosen = templates[:n_rules - 1]
    cat_rule = ([("category", "eq", t["category"])], CAT_QUEUE[t["category"]])
    if cat_rule[1] not in [q for _, q in chosen]:
        chosen.append(cat_rule)
    rng.shuffle(chosen)
    rules = [{"c": [list(x) for x in c], "q": q, "p": i + 1} for i, (c, q) in enumerate(chosen)]
    rules.append({"c": [], "q": "General Queue", "p": len(rules) + 1})
    matches = [r for r in sorted(rules, key=lambda r: r["p"]) if all(_cond_ok(c, t) for c in r["c"])]
    if len(matches) < {3: 2, 4: 3, 5: 3}[diff]:
        raise Retry("few matches")
    gold_q = matches[0]["q"]
    naive = CAT_QUEUE[t["category"]]
    if gold_q == naive and chance(rng, 0.85):
        raise Retry("naive right")
    queues = list(dict.fromkeys(r["q"] for r in rules))
    numbered = chance(rng, 0.5)
    shown = rules[:-1][:]
    if numbered:
        rng.shuffle(shown)
    lines = []
    for r in shown:
        cond = " and ".join(_cond_text(c, dom) for c in r["c"])
        pre = f"Priority {r['p']}: " if numbered else f"{r['p']}. "
        lines.append(f"{pre}if {cond} → {r['q']}")
    lines.append((f"Priority {rules[-1]['p']}: " if numbered else f"{rules[-1]['p']}. ") + "otherwise → General Queue")
    how = (P(rng, "When several rules match, the rule with the lowest priority number wins.", "Rules are listed in no particular order; among matching rules the smallest priority number decides.")
           if numbered else P(rng, "Rules are checked from top to bottom and the first rule that matches decides the queue.",
                                "The first matching rule, reading down the list, wins."))
    org = f"{rng.choice(['Northwind', 'Halcyon', 'Kestrel', 'Meridian', 'Juniper', 'Larchmont'])} {rng.choice(DOMAINS[dom]['org_suffix'])}"
    tier_name = DOMAINS[dom]["tier_name"]
    tid = code(rng, "TCK", 6)

    def ticket(hide=None):
        rows = [("Ticket", tid), ("Category (set by triage)", t["category"]), (f"Customer {tier_name}", t["tier"]), ("Language", t["language"]),
                ("Customer region", t["region"]), ("Severity", f"S{t['severity']}"), ("Channel", t["channel"]),
                ("Named account manager", "yes" if t["am"] else "no")]
        keymap = {"category": 1, "tier": 2, "language": 3, "region": 4, "severity": 5, "channel": 6, "am": 7}
        if hide:
            k = keymap[hide]
            rows[k] = (rows[k][0], "not captured")
        rng2 = random.Random(tid)
        body = "\n".join(f"{a}: {b}" for a, b in rows)
        msg = rng2.choice(MESSAGES[rng2.choice([c for c in CATS if c != t["category"]])] if chance(rng2, 0.4) else MESSAGES[t["category"]])
        return body + f'\nCustomer message: "{msg}"'

    head = P(rng, f"{org} support routing rules. Routing uses only the ticket fields; the free-text message is never used for routing.",
             f"Queue assignment policy at {org} (applied by the router to the structured ticket fields only, not to the message text):",
             f"{org} — ticket router configuration. Only the fields below are evaluated; message wording is ignored.")
    state = f"{head}\n{how}\n\n" + "\n".join(lines) + "\n\nIncoming ticket:\n" + ticket()
    q = P(rng, "Which queue should this ticket be routed to?", f"Where does the router send ticket {tid}?", "Which team receives this ticket under the rules?",
          "What is the correct destination queue for the ticket?")
    k = min(len(queues), {3: 5, 4: 8, 5: 12}[diff])
    wrong = [x["q"] for x in matches[1:]] + [naive] + [x for x in queues if x != gold_q]
    field, gold = choice_field(rng, q, gold_q, list(dict.fromkeys(w for w in wrong if w != gold_q)), k=max(4, k))
    if len(field["options"]) < 4:
        raise Retry("few queues")
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "route", "rules": rules, "t": t, "tiers": tiers, "opt": keymap}
    near = [x["q"] for x in matches[1:3] if x["q"] != gold_q] + ([naive] if naive != gold_q else [])
    parent = item(state, field, gold, "routing_table", dom, spec, {"near_miss": list(dict.fromkeys(near))[:3]})
    out = [parent]
    if want_unknown:
        attrs = list(dict.fromkeys(c[0] for c in matches[0]["c"]))
        rng.shuffle(attrs)
        for a in attrs:
            spec_u = dict(spec, unk=a)
            if recheck(spec_u) is None:
                state_u = f"{head}\n{how}\n\n" + "\n".join(lines) + "\n\nIncoming ticket:\n" + ticket(hide=a)
                out.append(_unknown_child(parent, state_u, spec_u))
                break
        else:
            raise Retry("routing unknown decided")
    return out


def _rc_route(spec, t):
    best = None
    for r in spec["rules"]:
        ok = True
        for a, op, v in r["c"]:
            x = t[a]
            if op == "eq":
                ok &= x == v
            elif op == "ne":
                ok &= x != v
            elif op == "in":
                ok &= x in v
            elif op == "le":
                ok &= x <= v
            elif op == "ge":
                ok &= x >= v
        if ok and (best is None or r["p"] < best["p"]):
            best = r
    return spec["opt"].get(best["q"], "__not_an_option__")


# ================================================================================================ oncall_route
def _off(m):
    s = "+" if m >= 0 else "-"
    return f"UTC{s}{abs(m) // 60:02d}:{abs(m) % 60:02d}"


def _oncall_resolve(sp, when_utc: dt.datetime, swap_override=None):
    """Generator path: local time -> shift -> swap -> leave -> secondary -> manager."""
    loc = when_utc + dt.timedelta(minutes=sp["tz"])
    week0 = dt.date.fromisoformat(sp["week0"])
    shifts = sp["shifts"]  # list of [name, start_hour, hours]
    for back in (0, 1):
        day = loc.date() - dt.timedelta(days=back)
        for si, (_, sh, hrs) in enumerate(shifts):
            st = dt.datetime.combine(day, dt.time(sh))
            if st <= loc < st + dt.timedelta(hours=hrs):
                d_idx = (day - week0).days
                if not 0 <= d_idx < 7:
                    return None
                key = f"{d_idx}:{si}"
                prim = sp["rota"][key][0]
                sec = sp["rota"][key][1]
                sw = sp["swaps"].get(key)
                if swap_override is not None and key == sp.get("swap_key"):
                    sw = swap_override
                if sw:
                    prim = sw
                ld = loc.date()
                if any(dt.date.fromisoformat(a) <= ld <= dt.date.fromisoformat(b) for p_, a, b in sp["leave"] if p_ == prim):
                    prim = sec
                    if any(dt.date.fromisoformat(a) <= ld <= dt.date.fromisoformat(b) for p_, a, b in sp["leave"] if p_ == prim):
                        prim = sp["manager"]
                return prim
    return None


def gen_oncall(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(DOMAIN_IDS)
    people = People(rng)
    engs = [people() for _ in range({3: 5, 4: 7, 5: 9}[diff])]
    manager = people()
    tz = rng.choice([60, 120, 330, -300, -420, 480, 0, 180, 570, 600])
    shifts = [["Day", 8, 12], ["Night", 20, 12]] if diff < 5 or chance(rng, 0.5) else [["Early", 0, 8], ["Late", 8, 8], ["Evening", 16, 8]]
    week0 = dt.date(2026, 1, 5) + dt.timedelta(weeks=rng.randint(0, 90))
    rota = {}
    for d in range(7):
        for si in range(len(shifts)):
            a, b = rng.sample(engs, 2)
            rota[f"{d}:{si}"] = [a, b]
    # the alert
    d_alert = rng.randint(1, 5)
    si_alert = rng.randrange(len(shifts))
    sh = shifts[si_alert]
    start_local = dt.datetime.combine(week0 + dt.timedelta(days=d_alert), dt.time(sh[1]))
    loc = start_local + dt.timedelta(minutes=rng.randint(10, sh[2] * 60 - 10))
    alert_tz = rng.choice([o for o in [0, -300, 60, 330, 540, -240, 120] if o != tz])
    alert_utc = loc - dt.timedelta(minutes=tz)
    key = f"{d_alert}:{si_alert}"
    swaps, leave = {}, []
    cover = rng.choice([e for e in engs if e not in rota[key]])
    if chance(rng, 0.7) or want_unknown:
        swaps[key] = cover
    # distractor swaps on other shifts
    for _ in range({3: 1, 4: 2, 5: 3}[diff]):
        k2 = f"{rng.randrange(7)}:{rng.randrange(len(shifts))}"
        if k2 != key:
            swaps[k2] = rng.choice([e for e in engs if e not in rota[k2]])
    eff = swaps.get(key, rota[key][0])
    if chance(rng, 0.4):
        d0 = loc.date() - dt.timedelta(days=rng.randint(0, 2))
        leave.append([eff, d0.isoformat(), (d0 + dt.timedelta(days=rng.randint(2, 4))).isoformat()])
    for _ in range(rng.randint(1, 2)):
        p = rng.choice(engs)
        d0 = week0 + dt.timedelta(days=rng.randint(0, 6))
        if p != eff and all(p != x[0] for x in leave):
            leave.append([p, d0.isoformat(), (d0 + dt.timedelta(days=rng.randint(0, 2))).isoformat()])
    sp = {"tz": tz, "week0": week0.isoformat(), "shifts": shifts, "rota": rota, "swaps": swaps, "leave": leave, "manager": manager, "swap_key": key}
    gold_p = _oncall_resolve(sp, alert_utc)
    if gold_p is None:
        raise Retry("oncall none")
    alert_local_str = (alert_utc + dt.timedelta(minutes=alert_tz)).strftime("%Y-%m-%d %H:%M")
    service = rng.choice(["payments API", "booking engine", "claims portal", "warehouse scanners", "patient portal", "document store"])

    def render(hide_swap=False):
        days = [week0 + dt.timedelta(days=d) for d in range(7)]
        lines = [P(rng, f"On-call rota for the {service} team. All rota times are local to the team ({_off(tz)}).",
                   f"{service.capitalize()} on-call schedule — times below are in {_off(tz)}.",
                   f"Rota (team time zone {_off(tz)}) for alerts on the {service}."),
                 "Shifts: " + "; ".join(f"{n} {h:02d}:00 for {hrs} hours" for n, h, hrs in shifts) +
                 ". A shift that runs past midnight belongs to the date on which it starts.", "",
                 "| Date | Shift | Primary | Secondary |", "|---|---|---|---|"]
        for d, day in enumerate(days):
            for si, (n, _, _) in enumerate(shifts):
                a, b = rota[f"{d}:{si}"]
                lines.append(f"| {day.isoformat()} ({day.strftime('%a')}) | {n} | {a} | {b} |")
        lines.append("")
        lines.append("Agreed swaps (the covering engineer replaces the primary for that shift only):")
        for k, c in swaps.items():
            d, si = map(int, k.split(":"))
            who = rota[k][0]
            if hide_swap and k == key:
                lines.append(f"  - {days[d].isoformat()} {shifts[si][0]}: {who} swapped out; covering engineer not recorded in the export")
            else:
                lines.append(f"  - {days[d].isoformat()} {shifts[si][0]}: {c} covers for {who}")
        lines.append("Leave (inclusive dates, team time zone): " + ("; ".join(f"{p} {a} to {b}" for p, a, b in leave) if leave else "none"))
        lines.append(P(rng, f"If the engineer due to be paged is on leave, the shift's secondary is paged instead; if the secondary is also on leave, {manager} (engineering manager) is paged.",
                       f"Leave rule: page the secondary when the primary (after swaps) is on leave; page {manager}, the engineering manager, when both are away."))
        lines.append("")
        lines.append(f"Alert: {service} error rate above threshold at {alert_local_str} ({_off(alert_tz)}).")
        return "\n".join(lines)

    q = P(rng, "Who should be paged for this alert?", "Which person receives the page?", "Who is the right person to page under the rota?",
          "Under the schedule, whom does the alert go to?")
    wrong = [e for e in engs + [manager] if e != gold_p]
    rng.shuffle(wrong)
    naive_local = _oncall_resolve(dict(sp, tz=alert_tz), alert_utc)
    pri = rota[key][0]
    firsts = [x for x in [pri, naive_local, rota[key][1]] if x and x != gold_p]
    wrong = list(dict.fromkeys(firsts + wrong))
    field, gold = choice_field(rng, q, gold_p, wrong, k={3: 5, 4: 8, 5: 11}[diff])
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "oncall", "sp": sp, "alert_utc": alert_utc.isoformat(), "opt": keymap}
    parent = item(render(), field, gold, "oncall_route", dom, spec, {"near_miss": firsts[:3]})
    out = [parent]
    if want_unknown:
        spec_u = dict(spec, unk={"swap_key": key, "cands": engs})
        if recheck(spec_u) is not None:
            raise Retry("oncall unknown decided")
        out.append(_unknown_child(parent, render(hide_swap=True), spec_u))
    return out


def _rc_oncall(spec, swaps):
    sp = spec["sp"]
    t = dt.datetime.fromisoformat(spec["alert_utc"]) + dt.timedelta(minutes=sp["tz"])
    week0 = dt.datetime.fromisoformat(sp["week0"])
    mins = int((t - week0).total_seconds() // 60)
    cand = None
    for d in range(-1, 8):
        for si, (_, h, hrs) in enumerate(sp["shifts"]):
            s = d * 1440 + h * 60
            if s <= mins < s + hrs * 60:
                cand = (d, si)
    if cand is None or not 0 <= cand[0] < 7:
        return "__none__"
    k = f"{cand[0]}:{cand[1]}"
    who = swaps.get(k) or sp["rota"][k][0]
    day = t.date().isoformat()
    on_leave = lambda p: any(x[0] == p and x[1] <= day <= x[2] for x in sp["leave"])  # noqa: E731
    if on_leave(who):
        who = sp["rota"][k][1]
        if on_leave(who):
            who = sp["manager"]
    return spec["opt"].get(who, "__not_an_option__")


# ================================================================================================ tool_select
CAPS = {
    "look up an order": ["order_lookup", "order_history_archive", "marketplace_order_lookup", "b2b_order_lookup"],
    "issue a refund": ["refund_auto", "refund_manual_review", "refund_marketplace", "refund_store_credit"],
    "change a delivery address": ["address_change_prepick", "address_change_intransit", "carrier_redirect"],
    "reset multi-factor authentication": ["mfa_reset_self_service", "mfa_reset_agent", "mfa_reset_admin_console"],
    "export account data": ["export_standard", "export_bulk_async", "export_regulated_region"],
    "reschedule an appointment": ["reschedule_online", "reschedule_clinic_desk", "reschedule_specialist"],
}
DIMS = {
    "age": ("days since the order or event", lambda lo, hi: f"only for items {lo}–{hi} days old"),
    "amount": ("amount", lambda lo, hi: f"amounts from {lo} to {hi}"),
    "region": ("customer region", None),
    "verified": ("identity verified", None),
    "channel": ("sales channel", None),
}


def gen_tool_select(rng: random.Random, diff: int, want_unknown: bool) -> list[dict]:
    dom = rng.choice(DOMAIN_IDS)
    cap = rng.choice(list(CAPS))
    names = list(CAPS[cap])
    rng.shuffle(names)
    req = {"age": rng.randint(1, 400), "amount": rng.randint(5, 3000), "region": rng.choice(REGIONS), "verified": chance(rng, 0.6),
           "channel": rng.choice(["own website", "marketplace", "wholesale account", "in store"])}
    dims = rng.sample(["age", "amount", "region", "verified", "channel"], {3: 2, 4: 3, 5: 4}[diff])

    def constraint_true(d):
        if d == "age":
            lo = rng.randint(0, req["age"])
            return ("age", [lo, rng.randint(req["age"], req["age"] + 200)])
        if d == "amount":
            return ("amount", [rng.randint(0, req["amount"]), rng.randint(req["amount"], req["amount"] + 2000)])
        if d == "region":
            return ("region", sorted({req["region"]} | set(rng.sample(REGIONS, 1))))
        if d == "verified":
            return ("verified", chance(rng, 0.5) if req["verified"] else False)
        return ("channel", sorted({req["channel"]} | set(rng.sample(["own website", "marketplace", "wholesale account", "in store"], 1))))

    def constraint_false(d):
        if d == "age":
            if req["age"] > 30 and chance(rng, 0.6):
                return ("age", [0, req["age"] - rng.randint(1, min(30, req["age"]))])
            return ("age", [req["age"] + rng.randint(1, 60), req["age"] + 400])
        if d == "amount":
            if req["amount"] > 50 and chance(rng, 0.6):
                return ("amount", [0, req["amount"] - rng.randint(1, min(200, req["amount"] - 1))])
            return ("amount", [req["amount"] + rng.randint(1, 300), req["amount"] + 5000])
        if d == "region":
            return ("region", sorted(rng.sample([r for r in REGIONS if r != req["region"]], 2)))
        if d == "verified":
            if req["verified"]:
                return constraint_false("amount")
            return ("verified", True)
        return ("channel", sorted(rng.sample([c for c in ["own website", "marketplace", "wholesale account", "in store"] if c != req["channel"]], 2)))

    tools = []
    gold_name = names[0]
    tools.append({"n": gold_name, "cap": cap, "c": [list(constraint_true(d)) for d in dims]})
    for nm in names[1:]:
        cs = [list(constraint_true(d)) for d in dims]
        j = rng.randrange(len(dims))
        cs[j] = list(constraint_false(dims[j]))
        tools.append({"n": nm, "cap": cap, "c": cs})
    other_caps = rng.sample([c for c in CAPS if c != cap], {3: 1, 4: 2, 5: 3}[diff])
    for oc in other_caps:
        for nm in rng.sample(CAPS[oc], 2):
            tools.append({"n": nm, "cap": oc, "c": [list(constraint_true(d)) for d in dims]})
    rng.shuffle(tools)

    def sat(tl, r):
        for d, v in tl["c"]:
            if d in ("age", "amount") and not (v[0] <= r[d] <= v[1]):
                return False
            if d in ("region", "channel") and r[d] not in v:
                return False
            if d == "verified" and v is True and not r[d]:
                return False
        return True

    good = [tl["n"] for tl in tools if tl["cap"] == cap and sat(tl, req)]
    if good != [gold_name]:
        raise Retry("tool_select not unique")
    sym = rng.choice(["€", "$", "£"])

    def ctext(d, v):
        if d == "age":
            return P(rng, f"only for orders/events {v[0]}–{v[1]} days old", f"covers ages between {v[0]} and {v[1]} days")
        if d == "amount":
            return P(rng, f"amounts {sym}{v[0]}–{sym}{v[1]}", f"handles values from {sym}{v[0]} up to {sym}{v[1]}")
        if d == "region":
            return "regions: " + ", ".join(v)
        if d == "verified":
            return "requires a verified identity" if v else P(rng, "no identity verification needed", "usable without identity verification")
        return "channels: " + ", ".join(v)

    # the constraint texts must be fixed once (P uses rng)
    ctexts = {tl["n"]: "; ".join(ctext(d, v) for d, v in tl["c"]) for tl in tools}

    def render(hide=None):
        lines = [P(rng, "Tool catalogue available to the support agent:", "Tools the assistant may call (each tool may be used only within its stated limits):",
                   "Available functions and their scopes:")]
        for tl in tools:
            lines.append(f"- {tl['n']}: {tl['cap']} — {ctexts[tl['n']]}")
        lines.append("")
        vals = {"age": f"{req['age']} days", "amount": f"{sym}{req['amount']}", "region": req["region"],
                "verified": "yes" if req["verified"] else "no", "channel": req["channel"]}
        labels = {"age": "Days since the order/event", "amount": "Amount involved", "region": "Customer region", "verified": "Identity verified",
                  "channel": "Sales channel"}
        lines.append(P(rng, f"Customer request: the customer wants to {cap}.", f"Request: {cap}.", f"The agent needs to {cap} for this customer."))
        for d in ["age", "amount", "region", "verified", "channel"]:
            lines.append(f"  {labels[d]}: " + ("not stated in the request" if d == hide else vals[d]))
        return "\n".join(lines)

    q = P(rng, "Which tool should the agent call to handle this request?", "Which single tool is permitted and suitable for this request?",
          "Pick the tool that can handle the request within its limits.", "Which function should be used?")
    wrong = [tl["n"] for tl in tools if tl["n"] != gold_name]
    field, gold = choice_field(rng, q, gold_name, wrong, k={3: 5, 4: 8, 5: 12}[diff])
    keymap = {o["text"]: o["key"] for o in field["options"]}
    spec = {"k": "toolsel", "tools": tools, "req": req, "cap": cap, "opt": keymap}
    near = [tl["n"] for tl in tools if tl["cap"] == cap and tl["n"] != gold_name and tl["n"] in keymap][:3]
    parent = item(render(), field, gold, "tool_select", dom, spec, {"near_miss": near})
    out = [parent]
    if want_unknown:
        for d in rng.sample(dims, len(dims)):
            spec_u = dict(spec, unk=d)
            if recheck(spec_u) is None:
                out.append(_unknown_child(parent, render(hide=d), spec_u))
                break
        else:
            raise Retry("toolsel unknown decided")
    return out


def _rc_toolsel(spec, req):
    ok = []
    for tl in spec["tools"]:
        if tl["cap"] != spec["cap"]:
            continue
        good = True
        for d, v in tl["c"]:
            x = req[d]
            if isinstance(v, bool):
                good &= (x or not v)
            elif isinstance(v, list) and v and isinstance(v[0], int) and d in ("age", "amount"):
                good &= v[0] <= x <= v[1]
            else:
                good &= x in v
        if good:
            ok.append(tl["n"])
    if len(ok) != 1:
        return "__ambiguous__"
    return spec["opt"].get(ok[0], "__not_an_option__")


GEN = {"grid_move": gen_grid, "dom_target": gen_dom, "craft_next": gen_craft, "tool_next": gen_tool, "wizard_step": gen_wizard,
       "routing_table": gen_routing, "oncall_route": gen_oncall, "tool_select": gen_tool_select}


# ================================================================================================ recheck
def _agree(golds):
    """None (undetermined) when the enumerated worlds disagree or any world has no unique answer."""
    if any(g is None or str(g).startswith("__") for g in golds) or len(set(golds)) > 1:
        return None
    return golds[0]


def worlds(spec: dict) -> list:
    """Every enumerated completion's answer (option key, or a "__..." marker when that world has no unique listed answer): one world
    for an answerable spec; for an unknown spec (spec["unk"]) one per value of the withheld fact -- the enumeration recheck() agrees
    over. gen_traps.py re-derives trap golds from these."""
    k, unk = spec["k"], spec.get("unk")
    if k == "grid":
        return [_rc_grid(spec, True), _rc_grid(spec, False)] if unk else [_rc_grid(spec, spec["locked"])]
    if k == "dom":
        return [_rc_dom(spec, {int(i): v for i, v in alt.items()}) for alt in unk] if unk else [_rc_dom(spec, {})]
    if k == "craft":
        return [_rc_craft(spec, dict(spec["inv"], **{unk["raw"]: v})) for v in unk["vals"]] if unk else [_rc_craft(spec, spec["inv"])]
    if k == "tool":
        if unk:
            s = unk["step"]
            done = [d for d in spec["done"] if d != s]
            return [_rc_tool(spec, done), _rc_tool(spec, done + [s])]
        return [_rc_tool(spec, spec["done"])]
    if k == "wizard":
        return [_rc_wizard(spec, spec["vals"]), _rc_wizard(spec, unk["alt"])] if unk else [_rc_wizard(spec, spec["vals"])]
    if k == "route":
        if unk:
            dom_vals = spec["tiers"] if unk == "tier" else ATTR_DOMAIN[unk]
            return [_rc_route(spec, dict(spec["t"], **{unk: v})) for v in dom_vals]
        return [_rc_route(spec, spec["t"])]
    if k == "oncall":
        if unk:
            outs = []
            for c in unk["cands"]:
                sw = dict(spec["sp"]["swaps"])
                sw[unk["swap_key"]] = c
                outs.append(_rc_oncall(spec, sw))
            return outs
        return [_rc_oncall(spec, spec["sp"]["swaps"])]
    if k == "toolsel":
        if unk:
            vals = {"age": [1, 45, 95, 200, 400], "amount": [5, 50, 250, 1000, 3000, 6000], "region": REGIONS, "verified": [True, False],
                    "channel": ["own website", "marketplace", "wholesale account", "in store"]}[unk]
            return [_rc_toolsel(spec, dict(spec["req"], **{unk: v})) for v in vals]
        return [_rc_toolsel(spec, spec["req"])]
    raise KeyError(k)


def recheck(spec: dict):
    w = worlds(spec)
    return _agree(w) if spec.get("unk") else w[0]
