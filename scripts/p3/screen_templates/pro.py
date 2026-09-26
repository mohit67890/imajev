"""Dense professional-app screens (1280x800) for gen_screens.py families pro_* (ScreenSpot-Pro-like small targets).

Templates: IDE, spreadsheet, image editor, video editor, CAD, DAW. Toolbars of small icons (16-18 px glyphs in 24-28 px
buttons), tiny labels, one visible tooltip sometimes, an open menu sometimes (menu items with shortcuts, some disabled).
Every control we may ask about has a unique desc and, for "which one does X" questions, a unique goal.
"""
from __future__ import annotations

from base import E, Screen, doc
from icons import svg

W, H = 1280, 800
THEMES = [  # bg, panel, text, muted, accent, border, sel, font, dark
    ("#1e1e1e", "#252526", "#d4d4d4", "#8b8b8b", "#3794ff", "#3c3c3c", "#094771", "'Segoe UI', Helvetica, Arial, sans-serif", True),
    ("#2b2b2b", "#323232", "#dddddd", "#9a9a9a", "#4a9df8", "#444444", "#3d5a80", "Helvetica, Arial, sans-serif", True),
    ("#f3f3f3", "#ffffff", "#1f1f1f", "#6e6e6e", "#0f6cbd", "#d6d6d6", "#cfe4fa", "'Segoe UI', Helvetica, Arial, sans-serif", False),
    ("#e9ecef", "#f8f9fa", "#212529", "#6c757d", "#107c41", "#ced4da", "#d1e7dd", "Arial, Helvetica, sans-serif", False),
    ("#1a1d23", "#22262e", "#e6e6e6", "#8a93a3", "#e0a030", "#343a46", "#4a3b1a", "Helvetica, Arial, sans-serif", True),
]


def css(t):
    bg, panel, text, muted, acc, border, sel, font, dark = t
    return f"""
body{{color:{text};font-size:12px}}
.mb{{display:flex;gap:2px;height:26px;align-items:center;padding:0 6px;background:{panel};border-bottom:1px solid {border}}}
.mb span{{padding:3px 8px;border-radius:3px}} .mb span.open{{background:{sel}}}
.tb{{display:flex;align-items:center;gap:2px;padding:3px 6px;background:{panel};border-bottom:1px solid {border}}}
.ti{{width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;border-radius:4px;color:{text}}}
.ti.on{{background:{sel}}} .ti.sm{{width:22px;height:22px}}
.sep{{width:1px;height:20px;background:{border};margin:0 4px}}
.pn{{background:{panel};border:1px solid {border}}}
.muted{{color:{muted}}} .lab{{font-size:10.5px;color:{muted}}}
.menu{{position:absolute;background:{panel};border:1px solid {border};box-shadow:0 6px 20px #0006;padding:4px 0;min-width:230px;z-index:10}}
.menu div{{display:flex;justify-content:space-between;gap:24px;padding:4px 16px;font-size:12.5px}}
.menu div.dis{{opacity:.4}} .menu hr{{border:none;border-top:1px solid {border};margin:3px 0}}
.tip{{position:absolute;background:#ffffe1;color:#111;border:1px solid #888;font-size:11.5px;padding:3px 6px;z-index:9;white-space:nowrap}}
.sbar{{position:absolute;left:0;right:0;bottom:0;height:22px;display:flex;align-items:center;gap:2px;padding:0 6px;background:{acc};color:#fff;font-size:11.5px}}
.sbar span{{padding:1px 7px}}
.tbtn{{font-size:10.5px;font-weight:700;width:18px;height:17px;display:inline-flex;align-items:center;justify-content:center;border-radius:3px;border:1px solid {border}}}
"""


def page(t, body):
    return doc(css(t), f'<div style="position:relative;width:{W}px;height:{H}px;overflow:hidden">{body}</div>', t[0], t[7], W, H)


def menubar(s, names, open_idx=None, prefix="m"):
    out = []
    for i, n in enumerate(names):
        a = s.el(f"{prefix}_{i}", f"the {n} menu", n, [], None, small=True)
        out.append(f'<span class="{"open" if i == open_idx else ""}" {a}>{E(n)}</span>')
    return f'<div class="mb">{"".join(out)}</div>'


def icon_btn(s, eid, glyph, name, goal, size=17, cls="ti", on=False, desc=None):
    a = s.el(eid, desc or f"the {name} button", None, [goal] if goal else [], None, small=True)
    s.meta.setdefault("icons", {})[eid] = {"glyph": glyph, "name": name}
    return f'<span class="{cls}{" on" if on else ""}" {a} title="{E(name)}">{svg(glyph, size)}</span>'


def open_menu(rng, s, t, items, x, y):
    """items: (label, shortcut, goal, disabled). Registers group 'menu'."""
    rows = []
    for i, it in enumerate(items):
        if it is None:
            rows.append("<hr>")
            continue
        lab, sc, goal, dis = it
        a = s.el(f"mi_{i}", f"the {lab} menu item", lab, [goal], None, disabled=dis, group="menu")
        s.possible.append(f"mi_{i}")
        rows.append(f'<div class="{"dis" if dis else ""}" {a}><span>{E(lab)}</span><span class="muted">{E(sc)}</span></div>')
    s.meta["menu_open"] = True
    return f'<div class="menu" style="left:{x}px;top:{y}px">{"".join(rows)}</div>'


def tooltip(rng, s, eid_name_pairs, positions):
    """Show one tooltip next to one icon (by computed position hint)."""
    return ""


# ------------------------------------------------------------------------------------------------ IDE
def ide(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("pro", "ide", W, H)
    s.screen_desc = "a code editor (IDE)"
    menu_kind = rng.choice([None, "Edit", "File", "Run"])
    names = ["File", "Edit", "Selection", "View", "Go", "Run", "Terminal", "Help"]
    mb = menubar(s, names, names.index(menu_kind) if menu_kind else None)
    act = [("files", "Explorer", "browse the project's files"), ("search", "Search", "search across all files"),
           ("branch", "Source Control", "commit your changes"), ("bug", "Run and Debug", "start a debugging session"),
           ("blocks", "Extensions", "install an extension")]
    ab = "".join(f'<div style="padding:6px 0">{icon_btn(s, f"act_{i}", g, n, goal, 22, desc=f"the {n} icon in the activity bar")}</div>'
                 for i, (g, n, goal) in enumerate(act))
    files = ["app.py", "models.py", "utils.py", "config.yaml", "README.md", "test_app.py", "requirements.txt", "routes.py", "db.py"]
    open_files = rng.sample(files, rng.randint(3, 4))
    cur = open_files[0]
    tree = "".join(f'<div style="padding:3px 10px 3px {18 if f.endswith(".py") else 10}px;{"background:" + t[6] if f == cur else ""}" '
                   f'{s.el("tree_" + str(i), f"the {f} file in the explorer", f, [f"open {f}"], None)}>{E(f)}</div>'
                   for i, f in enumerate(files))
    tabs = "".join(f'<div style="display:flex;align-items:center;gap:6px;padding:6px 10px;border-right:1px solid {t[5]};{"background:" + t[0] if f == cur else ""}">'
                   f'<span {s.el("tab_" + str(i), f"the {f} tab", f, [f"switch to {f}"], None)}>{E(f)}</span>'
                   f'{icon_btn(s, "tabx_" + str(i), "close", f"close {f}", f"close the {f} tab", 12, "ti sm", desc=f"the close button on the {f} tab")}</div>'
                   for i, f in enumerate(open_files))
    run_tools = [("play", "Run", "run the program"), ("stop", "Stop", "stop the running program"), ("refresh", "Restart", "restart the program"),
                 ("step_over", "Step Over", "step over the current line"), ("grid", "Split Editor", "split the editor in two")]
    rng.shuffle(run_tools)
    rt = "".join(icon_btn(s, f"run_{i}", g, n, goal, 16) for i, (g, n, goal) in enumerate(run_tools[:4]))
    code_lines = ["import os", "from db import connect", "", "def load(path):", "    with open(path) as fh:", "        return fh.read()", "",
                  "class App:", "    def __init__(self, cfg):", "        self.cfg = cfg", "        self.db = connect(cfg['db'])", "",
                  "    def run(self):", "        data = load(self.cfg['input'])", "        for row in data.splitlines():", "            self.handle(row)",
                  "", "if __name__ == '__main__':", "    App({'db': 'local', 'input': 'in.txt'}).run()"]
    code = "".join(f'<div style="display:flex;font-family:Menlo,monospace;font-size:12.5px;line-height:19px"><span class="muted" style="width:40px;text-align:right;padding-right:12px">{i + 1}</span>'
                   f'<span style="white-space:pre">{E(l)}</span></div>' for i, l in enumerate(code_lines))
    ptabs = [("Problems", "see the list of errors and warnings"), ("Output", "see the program's output log"), ("Terminal", "type shell commands"),
             ("Debug Console", "evaluate expressions while debugging")]
    pt = "".join(f'<span style="padding:4px 10px;{"border-bottom:1px solid " + t[4] if i == 2 else ""}" {s.el("ptab_" + str(i), f"the {n} tab in the bottom panel", n.upper(), [g], None, group="panel", small=True)}>{n.upper()}</span>'
                 for i, (n, g) in enumerate(ptabs))
    err, warn = rng.randint(0, 5), rng.randint(0, 9)
    sb_items = [("branch", "main", "switch to another git branch"), ("probs", f"⊗ {err}  ⚠ {warn}", "open the problems list"),
                ("pos", f"Ln {rng.randint(1, 19)}, Col {rng.randint(1, 30)}", "jump to a line number"), ("indent", "Spaces: 4", "change the indentation size"),
                ("enc", "UTF-8", "change the file encoding"), ("lang", "Python", "change the language mode")]
    sb = "".join(f'<span {s.el("sb_" + k, f"the {lab} item in the status bar", lab, [g], None, small=True)}>{E(lab)}</span>' + ('<span style="flex:1"></span>' if k == "probs" else "")
                 for k, lab, g in sb_items)
    menu = ""
    if menu_kind == "Edit":
        sel = rng.random() < 0.5
        s.meta["selection"] = sel
        menu = open_menu(rng, s, t, [("Undo", "Ctrl+Z", "undo the last change", False), ("Redo", "Ctrl+Y", "redo the change you undid", rng.random() < 0.4), None,
                                     ("Cut", "Ctrl+X", "cut the selected text", not sel), ("Copy", "Ctrl+C", "copy the selected text", not sel),
                                     ("Paste", "Ctrl+V", "paste from the clipboard", False), None, ("Find", "Ctrl+F", "find text in this file", False),
                                     ("Replace", "Ctrl+H", "replace text in this file", False), ("Find in Files", "Ctrl+Shift+F", "search every file in the project", False),
                                     None, ("Toggle Line Comment", "Ctrl+/", "comment out the current line", False)], 46, 26)
    elif menu_kind == "File":
        dirty = rng.random() < 0.6
        menu = open_menu(rng, s, t, [("New File", "Ctrl+N", "create a new file", False), ("Open File…", "Ctrl+O", "open an existing file", False),
                                     ("Open Folder…", "Ctrl+K Ctrl+O", "open a project folder", False), None, ("Save", "Ctrl+S", "save the current file", not dirty),
                                     ("Save As…", "Ctrl+Shift+S", "save a copy under a new name", False), ("Save All", "Ctrl+K S", "save every open file", not dirty),
                                     None, ("Revert File", "", "discard unsaved changes in this file", not dirty), ("Close Editor", "Ctrl+F4", "close the current editor", False),
                                     None, ("Exit", "", "quit the editor", False)], 6, 26)
    elif menu_kind == "Run":
        running = rng.random() < 0.5
        menu = open_menu(rng, s, t, [("Start Debugging", "F5", "start the program in the debugger", running),
                                     ("Run Without Debugging", "Ctrl+F5", "run the program without the debugger", running),
                                     ("Stop Debugging", "Shift+F5", "stop the debugging session", not running), ("Restart Debugging", "Ctrl+Shift+F5", "restart the debugging session", not running),
                                     None, ("Toggle Breakpoint", "F9", "set a breakpoint on the current line", False),
                                     ("Disable All Breakpoints", "", "turn off every breakpoint", False), None, ("Add Configuration…", "", "add a launch configuration", False)], 186, 26)
    body = (mb + f'<div style="display:flex;height:{H - 26 - 22}px"><div class="pn" style="width:46px;display:flex;flex-direction:column;align-items:center;padding-top:4px">{ab}</div>'
            f'<div class="pn" style="width:220px;padding-top:6px"><div class="muted" style="padding:4px 10px;font-weight:700">EXPLORER</div>{tree}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;min-width:0"><div class="pn" style="display:flex;align-items:center">{tabs}<span style="flex:1"></span>{rt}</div>'
            f'<div style="flex:1;padding-top:8px;overflow:hidden">{code}</div>'
            f'<div class="pn" style="height:190px"><div style="display:flex;gap:4px;padding:2px 6px">{pt}</div>'
            f'<div style="font-family:Menlo,monospace;padding:8px 12px" class="muted">$ python app.py<br>Loaded {rng.randint(10, 999)} rows</div></div></div></div>'
            f'<div class="sbar">{sb}</div>{menu}')
    for d, g in [("the Format Document button", "format the whole document"), ("the Git Pull button", "pull the latest commits"),
                 ("the Docker icon", "manage containers"), ("the Zen Mode button", "hide every panel")]:
        s.add_absent(d, g)
    s.html = page(t, body)
    return s


# ------------------------------------------------------------------------------------------------ spreadsheet
def sheet(rng) -> Screen:
    t = rng.choice([x for x in THEMES if not x[8]])
    s = Screen("pro", "spreadsheet", W, H)
    s.screen_desc = "a spreadsheet application"
    menu_kind = rng.choice([None, "Data"])
    names = ["File", "Home", "Insert", "Formulas", "Data", "Review", "View"]
    mb = menubar(s, names, names.index(menu_kind) if menu_kind else 1)
    groups = [("Clipboard", [("clip", "Paste", "paste the copied cells"), ("scissors", "Cut", "cut the selected cells"), ("files", "Copy", "copy the selected cells")]),
              ("Font", [("bold", "Bold", "make the text bold"), ("italic", "Italic", "make the text italic"), ("underline", "Underline", "underline the text")]),
              ("Alignment", [("align_l", "Align Left", "align the text to the left"), ("align_c", "Center", "center the text in the cell"),
                             ("align_r", "Align Right", "align the text to the right"), ("merge", "Merge Cells", "merge the selected cells into one")]),
              ("Number", [("tag", "Currency", "format the cells as currency"), ("chart", "Percent", "format the cells as a percentage")]),
              ("Editing", [("sigma", "AutoSum", "add up the selected numbers"), ("sort", "Sort", "sort the rows"), ("filter", "Filter", "add filter buttons to the headers"),
                           ("search", "Find", "find a value in the sheet")])]
    rib = []
    for gname, tools in groups:
        btns = "".join(f'<div style="display:flex;flex-direction:column;align-items:center;width:{44 if len(n) > 6 else 38}px">{icon_btn(s, "rb_" + n.replace(" ", "").lower(), g, n, goal, 17)}'
                       f'<span class="lab" style="font-size:9.5px;white-space:nowrap">{E(n)}</span></div>' for g, n, goal in tools)
        rib.append(f'<div style="display:flex;flex-direction:column;align-items:center;padding:4px 8px;border-right:1px solid {t[5]}"><div style="display:flex;gap:2px">{btns}</div>'
                   f'<span class="lab" style="margin-top:3px">{gname}</span></div>')
    shr = s.el("share", "the Share button", "Share", ["share the workbook with a colleague"], None)
    ribbon = f'<div class="tb" style="padding:2px 4px">{"".join(rib)}<span style="flex:1"></span><span {shr} style="background:{t[4]};color:#fff;padding:5px 12px;border-radius:4px">Share</span></div>'
    nb = s.el("namebox", "the Name Box", None, ["jump to a cell by its address"], None)
    fx = s.el("fx", "the insert function (fx) button", None, ["insert a function into the cell"], None, small=True)
    selcell = f"{rng.choice('BCDEF')}{rng.randint(2, 12)}"
    fbar = (f'<div class="tb" style="gap:8px"><span class="pn" style="width:70px;padding:2px 6px" {nb}>{selcell}</span>'
            f'<span {fx} style="font-style:italic;font-weight:700;padding:0 6px">fx</span><span class="pn" style="flex:1;padding:2px 8px">=SUM(B2:B{rng.randint(5, 14)})</span></div>')
    cols = "ABCDEFGHIJKL"
    data = [["Region", "Q1", "Q2", "Q3", "Q4", "Total"], ["North", 120, 135, 128, 160, None], ["South", 98, 102, 110, 125, None],
            ["East", 143, 150, 149, 170, None], ["West", 87, 90, 96, 104, None], ["Central", 110, 118, 121, 130, None]]
    picked = set()
    while len(picked) < 5:
        picked.add(f"{rng.choice(cols[:10])}{rng.randint(1, 22)}")
    grid = [f'<tr><th style="width:36px"></th>' + "".join(f'<th style="text-align:center;font-weight:400">{c}</th>' for c in cols) + "</tr>"]
    for r in range(1, 23):
        cells = []
        for ci, c in enumerate(cols):
            v = ""
            if r <= len(data) and ci < 6:
                v = data[r - 1][ci]
                v = "" if v is None else v
            ref = f"{c}{r}"
            a = s.el("cell_" + ref, f"the sheet cell in column {c}, row {r}", None, [], None, small=True) if ref in picked else ""
            cells.append(f'<td {a} style="border:1px solid {t[5]};padding:0 4px;height:21px;{"background:" + t[6] if ref == selcell else ""}">{E(str(v))}</td>')
        grid.append(f'<tr><td class="muted" style="text-align:center;border:1px solid {t[5]}">{r}</td>{"".join(cells)}</tr>')
    tabs = "".join(f'<span style="padding:3px 12px;{"background:" + t[1] + ";font-weight:700" if i == 0 else ""}" {s.el("sheet_" + str(i), f"the {n} sheet tab", n, [f"switch to the {n} sheet"], None, small=True)}>{E(n)}</span>'
                   for i, n in enumerate(rng.sample(["Sheet1", "Q3 data", "Budget", "Summary", "Raw"], 3)))
    addsh = icon_btn(s, "addsheet", "plus", "New Sheet", "add a new sheet", 14, "ti sm", desc="the add sheet (plus) button")
    menu = ""
    if menu_kind == "Data":
        has_filter = rng.random() < 0.5
        menu = open_menu(rng, s, t, [("Sort A to Z", "", "sort from smallest to largest", False), ("Sort Z to A", "", "sort from largest to smallest", False),
                                     None, ("Filter", "Ctrl+Shift+L", "show filter arrows on the header row", False),
                                     ("Clear Filter", "", "remove the current filter", not has_filter), None,
                                     ("Remove Duplicates", "", "delete repeated rows", False), ("Data Validation…", "", "restrict what can be typed in a cell", False),
                                     ("Text to Columns…", "", "split one column into several", False), None, ("Group Rows", "", "collapse rows into a group", False)], 262, 26)
    body = (mb + ribbon + fbar + f'<div style="overflow:hidden;height:{H - 26 - 76 - 34 - 50}px"><table style="border-collapse:collapse;width:100%;font-size:12px">{"".join(grid)}</table></div>'
            f'<div class="tb" style="position:absolute;bottom:0;left:0;right:0;height:26px">{tabs}{addsh}<span style="flex:1"></span>'
            f'<span class="muted">Sum: {rng.randint(100, 900)}</span></div>{menu}')
    for d, g in [("the Insert Chart button", "insert a chart"), ("the Wrap Text button", "wrap long text in the cell"),
                 ("the Freeze Panes button", "keep the header row visible"), ("the Pivot Table button", "summarise data in a pivot table")]:
        s.add_absent(d, g)
    s.html = page(t, body)
    return s


# ------------------------------------------------------------------------------------------------ image editor
def image_editor(rng) -> Screen:
    t = rng.choice([x for x in THEMES if x[8]])
    s = Screen("pro", "image_editor", W, H)
    s.screen_desc = "an image editing application"
    menu_kind = rng.choice([None, "Image"])
    names = ["File", "Edit", "Image", "Layer", "Select", "Filter", "View", "Window", "Help"]
    mb = menubar(s, names, names.index(menu_kind) if menu_kind else None)
    tools = [("move", "Move tool", "move the selected layer"), ("rect", "Rectangle Select tool", "select a rectangular area"),
             ("crop", "Crop tool", "crop the image"), ("brush", "Brush tool", "paint with a brush"), ("eraser", "Eraser tool", "erase pixels"),
             ("text", "Text tool", "add text to the image"), ("zoom", "Zoom tool", "zoom in on the image"), ("hand", "Hand tool", "pan around the canvas"),
             ("pencil", "Pencil tool", "draw hard-edged lines"), ("circle", "Ellipse tool", "draw an ellipse shape")]
    rng.shuffle(tools)
    tools = tools[:rng.randint(7, 10)]
    cur = rng.randrange(len(tools))
    tcol = "".join(f'<div>{icon_btn(s, "tool_" + str(i), g, n, goal, 18, on=i == cur, desc=f"the {n}")}</div>' for i, (g, n, goal) in enumerate(tools))
    tip = ""
    if rng.random() < 0.5:
        k = rng.randrange(len(tools))
        tip = f'<div class="tip" style="left:44px;top:{26 + 34 + 6 + 28 * k + 4}px">{E(tools[k][1])}</div>'
        s.meta["tooltip"] = tools[k][1]
    opts = (f'<div class="tb" style="gap:10px;height:34px"><span class="muted">Size:</span><span class="pn" style="padding:1px 6px" {s.el("opt_size", "the brush size field", None, ["change the brush size"], None, small=True)}>{rng.randint(5, 80)} px</span>'
            f'<span class="muted">Opacity:</span><span class="pn" style="padding:1px 6px" {s.el("opt_opacity", "the opacity field", None, ["change the brush opacity"], None, small=True)}>{rng.randint(20, 100)}%</span>'
            f'<span class="muted">Mode:</span><span class="pn" style="padding:1px 6px">Normal ▾</span></div>')
    layers = rng.sample(["Background", "Sky", "Portrait", "Shadows", "Logo", "Text overlay", "Colour fix"], rng.randint(4, 5))
    lrows = []
    for i, ln in enumerate(layers):
        eye = icon_btn(s, f"eye_{i}", "eye", f"visibility of {ln}", f"hide the {ln} layer", 14, "ti sm", desc=f"the eye icon on the {ln} layer")
        lock = icon_btn(s, f"lock_{i}", "lock", f"lock {ln}", f"lock the {ln} layer", 13, "ti sm", desc=f"the lock icon on the {ln} layer")
        lrows.append(f'<div style="display:flex;align-items:center;gap:6px;padding:4px 6px;border-bottom:1px solid {t[5]};{"background:" + t[6] if i == 0 else ""}">'
                     f'{eye}<span style="width:34px;height:24px;background:#777;display:inline-block"></span><span style="flex:1" {s.el("layer_" + str(i), f"the {ln} layer", ln, [f"select the {ln} layer"], None)}>{E(ln)}</span>{lock}</div>')
    lbtns = (icon_btn(s, "layer_add", "plus", "New Layer", "add a new layer", 14, "ti sm", desc="the new layer (plus) button") +
             icon_btn(s, "layer_del", "trash", "Delete Layer", "delete the selected layer", 14, "ti sm", desc="the delete layer (bin) button") +
             icon_btn(s, "layer_grp", "folder", "New Group", "put layers into a group", 14, "ti sm", desc="the new group (folder) button"))
    menu = ""
    if menu_kind == "Image":
        has_sel = rng.random() < 0.5
        menu = open_menu(rng, s, t, [("Adjustments", "▸", "change brightness or colour balance", False), None, ("Image Size…", "Alt+Ctrl+I", "change the image resolution", False),
                                     ("Canvas Size…", "Alt+Ctrl+C", "add space around the image", False), ("Rotate 90° Clockwise", "", "turn the image a quarter turn right", False),
                                     ("Flip Horizontal", "", "mirror the image left to right", False), None, ("Crop to Selection", "", "cut the image down to the selection", not has_sel),
                                     ("Trim…", "", "remove transparent borders", False), ("Duplicate", "", "make a copy of the image", False)], 96, 26)
    body = (mb + opts + f'<div style="display:flex;height:{H - 26 - 34 - 22}px"><div class="pn" style="width:40px;display:flex;flex-direction:column;align-items:center;gap:2px;padding-top:6px">{tcol}</div>'
            f'<div style="flex:1;display:flex;align-items:center;justify-content:center;background:#3a3a3a"><div style="width:{rng.randint(560, 700)}px;height:{rng.randint(400, 520)}px;background:linear-gradient(160deg,#8fb3d9,#e8c49a 60%,#6b5d4f)"></div></div>'
            f'<div class="pn" style="width:260px"><div style="padding:6px;font-weight:700">Layers</div>{"".join(lrows)}<div style="display:flex;gap:4px;padding:6px;justify-content:flex-end">{lbtns}</div></div></div>'
            f'<div style="position:absolute;bottom:0;left:0;right:0;height:22px;padding:3px 10px" class="pn muted">{rng.randint(25, 200)}% · {rng.choice(["3000 × 2000", "1920 × 1080", "4032 × 3024"])} px · RGB/8</div>{tip}{menu}')
    for d, g in [("the Lasso tool", "draw a freehand selection"), ("the Gradient tool", "fill with a colour gradient"),
                 ("the Clone Stamp tool", "copy pixels from one area to another"), ("the Eyedropper tool", "pick a colour from the image")]:
        s.add_absent(d, g)
    s.html = page(t, body)
    return s


# ------------------------------------------------------------------------------------------------ video editor
def video_editor(rng) -> Screen:
    t = rng.choice([x for x in THEMES if x[8]])
    s = Screen("pro", "video_editor", W, H)
    s.screen_desc = "a video editing application"
    mb = menubar(s, ["File", "Edit", "Clip", "Sequence", "Markers", "Window", "Help"])
    exp = s.el("export", "the Export button", "Export", ["render the final video file"], None)
    tr = [("prev", "Go to Start", "jump to the start of the timeline"), ("rewind", "Rewind", "rewind the playback"),
          ("play", "Play", "play the timeline"), ("forward", "Fast Forward", "fast-forward the playback"), ("next", "Go to End", "jump to the end of the timeline")]
    transport = "".join(icon_btn(s, "tr_" + str(i), g, n, goal, 16) for i, (g, n, goal) in enumerate(tr))
    tools = [("cursor", "Selection tool", "select and move clips"), ("scissors", "Razor tool", "cut a clip in two"),
             ("hand", "Hand tool", "scroll the timeline"), ("zoom", "Zoom tool", "zoom into the timeline"), ("text", "Type tool", "add a title")]
    tcol = "".join(f'<div>{icon_btn(s, "vt_" + str(i), g, n, goal, 16, on=i == 0, desc=f"the {n}")}</div>' for i, (g, n, goal) in enumerate(tools))
    tracks = ["V2", "V1", "A1", "A2"]
    rows = []
    for tk in tracks:
        vis = "eye" if tk.startswith("V") else "mute"
        vname = f"hide track {tk}" if vis == "eye" else f"mute track {tk}"
        vdesc = f"the eye icon on track {tk}" if vis == "eye" else f"the mute (speaker) icon on track {tk}"
        hd = (f'<div style="width:120px;display:flex;align-items:center;gap:3px;padding:0 6px;border-right:1px solid {t[5]}"><b style="width:26px">{tk}</b>'
              f'{icon_btn(s, "lk_" + tk, "lock", f"lock {tk}", f"lock track {tk}", 13, "ti sm", desc=f"the lock icon on track {tk}")}'
              f'{icon_btn(s, "vs_" + tk, vis, vname, vname, 13, "ti sm", desc=vdesc)}</div>')
        clips = []
        x = rng.randint(0, 60)
        while x < 900:
            wd = rng.randint(90, 260)
            col = "#4f6fb3" if tk.startswith("V") else "#3f8f5f"
            nm = rng.choice(["intro.mp4", "interview_02.mov", "b-roll_city.mp4", "drone_07.mp4", "music_bed.wav", "vo_take3.wav", "outro.mp4"])
            clips.append(f'<div style="position:absolute;left:{x}px;top:3px;height:32px;width:{wd}px;background:{col};border-radius:3px;font-size:10.5px;padding:2px 5px;overflow:hidden;color:#fff">{nm}</div>')
            x += wd + rng.randint(0, 80)
        rows.append(f'<div style="display:flex;height:40px;border-bottom:1px solid {t[5]}">{hd}<div style="flex:1;position:relative;overflow:hidden">{"".join(clips)}</div></div>')
    tc = f"00:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 24):02d}"
    body = (mb + f'<div class="tb" style="justify-content:flex-end;gap:8px"><span class="muted">Project: {rng.choice(["Harbour doc", "Spring promo", "Wedding cut", "Launch teaser"])}</span>'
            f'<span {exp} style="background:{t[4]};color:#fff;padding:4px 14px;border-radius:4px">Export</span></div>'
            f'<div style="display:flex;height:430px"><div class="pn" style="width:300px;padding:8px"><b>Media</b>'
            + "".join(f'<div style="display:flex;gap:6px;align-items:center;margin:6px 0"><span style="width:48px;height:28px;background:#666;display:inline-block"></span>{n}</div>'
                      for n in rng.sample(["intro.mp4", "interview_02.mov", "b-roll_city.mp4", "drone_07.mp4", "music_bed.wav", "vo_take3.wav"], 5))
            + f'</div><div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#000">'
            f'<div style="width:560px;height:315px;background:linear-gradient(135deg,#2c3e50,#b08d57)"></div>'
            f'<div style="display:flex;align-items:center;gap:6px;margin-top:8px;color:#ddd">{transport}<span style="margin-left:12px;font-family:Menlo,monospace">{tc}</span></div></div></div>'
            f'<div style="display:flex;height:{H - 26 - 32 - 430}px"><div class="pn" style="width:34px;display:flex;flex-direction:column;align-items:center;padding-top:4px">{tcol}</div>'
            f'<div style="flex:1"><div class="muted" style="height:22px;padding:3px 130px;font-family:Menlo,monospace;font-size:10px">00:00 · · · 00:10 · · · 00:20 · · · 00:30 · · · 00:40</div>{"".join(rows)}</div></div>')
    for d, g in [("the Slip tool", "slip a clip's contents"), ("the Colour wheels panel", "grade the colours"),
                 ("the Add Marker button", "drop a marker at the playhead"), ("the Audio Mixer button", "open the audio mixer")]:
        s.add_absent(d, g)
    s.html = page(t, body)
    return s


# ------------------------------------------------------------------------------------------------ CAD
def cad(rng) -> Screen:
    t = rng.choice(THEMES)
    s = Screen("pro", "cad", W, H)
    s.screen_desc = "a CAD drafting application"
    mb = menubar(s, ["File", "Edit", "View", "Insert", "Format", "Tools", "Draw", "Dimension", "Modify", "Help"])
    draw = [("line", "Line", "draw a straight line"), ("circle", "Circle", "draw a circle"), ("arc", "Arc", "draw an arc"),
            ("rect", "Rectangle", "draw a rectangle"), ("polygon", "Polygon", "draw a regular polygon"), ("hatch", "Hatch", "fill an area with a hatch pattern"),
            ("text", "Text", "add a text note")]
    modify = [("move", "Move", "move the selected objects"), ("rotate", "Rotate", "rotate the selected objects"), ("mirror", "Mirror", "mirror the selected objects"),
              ("scissors", "Trim", "trim lines at a cutting edge"), ("eraser", "Erase", "delete the selected objects"), ("dimension", "Dimension", "add a linear dimension"),
              ("zoom", "Zoom", "zoom into the drawing")]
    rng.shuffle(draw)
    rng.shuffle(modify)
    draw, modify = draw[:rng.randint(5, 7)], modify[:rng.randint(5, 7)]
    labels = rng.random() < 0.4
    def grp(lst, pre):
        return "".join((f'<div style="display:flex;flex-direction:column;align-items:center;width:36px">' if labels else "<div>") +
                       icon_btn(s, f"{pre}_{i}", g, n, goal, 17, desc=f"the {n} tool") + (f'<span class="lab" style="font-size:9px">{n}</span>' if labels else "") + "</div>"
                       for i, (g, n, goal) in enumerate(lst))
    tb = f'<div class="tb">{grp(draw, "dr")}<span class="sep"></span>{grp(modify, "md")}<span class="sep"></span><span class="muted">Layer:</span><span class="pn" style="padding:2px 8px" {s.el("layer", "the layer dropdown", None, ["change the current layer"], None, small=True)}>{rng.choice(["Walls", "Doors", "Furniture", "Dimensions"])} ▾</span></div>'
    tip = ""
    if not labels and rng.random() < 0.6:
        k = rng.randrange(len(draw))
        tip = f'<div class="tip" style="left:{10 + 28 * k}px;top:{26 + 34 + 2}px">{E(draw[k][1])}</div>'
        s.meta["tooltip"] = draw[k][1]
    toggles = ["SNAP", "GRID", "ORTHO", "POLAR", "OSNAP", "LWT"]
    st = {}
    tg = []
    for i, n in enumerate(toggles):
        on = rng.random() < 0.5
        st[n] = on
        full = {"SNAP": "snap mode", "GRID": "the grid display", "ORTHO": "ortho mode", "POLAR": "polar tracking", "OSNAP": "object snap", "LWT": "lineweight display"}[n]
        a = s.el(f"tg_{n}", f"the {n} toggle in the status bar", n, [f"turn {'off' if on else 'on'} {full}"], None, small=True, group="toggles")
        tg.append(f'<span {a} style="padding:1px 6px;margin:0 2px;border:1px solid {t[5]};{"background:" + t[6] if on else "opacity:.6"}">{n}</span>')
    lines = "".join(f'<line x1="{rng.randint(100, 900)}" y1="{rng.randint(80, 500)}" x2="{rng.randint(100, 900)}" y2="{rng.randint(80, 500)}" stroke="{t[2]}" stroke-width="1"/>' for _ in range(14))
    circles = "".join(f'<circle cx="{rng.randint(150, 850)}" cy="{rng.randint(120, 480)}" r="{rng.randint(15, 70)}" fill="none" stroke="{t[4]}" stroke-width="1"/>' for _ in range(3))
    body = (mb + tb + f'<div style="position:relative;height:{H - 26 - 36 - 90}px;background:{"#1b1f24" if t[8] else "#fbfbfb"}">'
            f'<svg width="{W}" height="{H - 152}" style="position:absolute;left:0;top:0">{lines}{circles}</svg></div>'
            f'<div class="pn" style="height:64px;padding:6px 10px;font-family:Menlo,monospace;font-size:11.5px"><div class="muted">Command: _circle Specify center point</div>'
            f'<div>Command: <span {s.el("cmdline", "the command line input", None, ["type a command"], None)} style="border-bottom:1px solid {t[3]};display:inline-block;width:300px">&nbsp;</span></div></div>'
            f'<div class="pn" style="height:26px;display:flex;align-items:center;gap:0;padding:0 8px;font-size:11px"><span class="muted" style="margin-right:12px">{rng.randint(100, 9999)}.{rng.randint(10, 99)}, {rng.randint(100, 9999)}.{rng.randint(10, 99)}, 0.00</span>{"".join(tg)}</div>{tip}')
    for d, g in [("the Ellipse tool", "draw an ellipse"), ("the Fillet tool", "round a corner between two lines"), ("the Offset tool", "draw a parallel copy of a line"),
                 ("the Array tool", "repeat objects in a grid")]:
        s.add_absent(d, g)
    s.meta.update({"toggles": st, "icon_labels": labels})
    s.html = page(t, body)
    return s


# ------------------------------------------------------------------------------------------------ DAW
def daw(rng) -> Screen:
    t = rng.choice([x for x in THEMES if x[8]])
    s = Screen("pro", "daw", W, H)
    s.screen_desc = "a music production (DAW) application"
    mb = menubar(s, ["File", "Edit", "Track", "Clip", "Transport", "Mix", "View", "Help"])
    tr = [("rewind", "Rewind", "go back to the start of the song"), ("play", "Play", "start playback"), ("stop", "Stop", "stop playback"),
          ("record", "Record", "start recording"), ("loop", "Loop", "loop the selected region"), ("metronome", "Metronome", "turn the click track on or off")]
    transport = "".join(icon_btn(s, "tp_" + str(i), g, n, goal, 17) for i, (g, n, goal) in enumerate(tr))
    bpm = s.el("tempo", "the tempo field", None, ["change the tempo"], None, small=True)
    names = rng.sample(["Drums", "Bass", "Keys", "Vocals", "Guitar", "Strings", "Synth pad", "Backing vox"], rng.randint(5, 7))
    rows = []
    for i, n in enumerate(names):
        m = s.el(f"mute_{i}", f"the M (mute) button on the {n} track", None, [f"mute the {n} track"], None, small=True)
        so = s.el(f"solo_{i}", f"the S (solo) button on the {n} track", None, [f"hear only the {n} track"], None, small=True)
        r = s.el(f"arm_{i}", f"the R (record arm) button on the {n} track", None, [f"arm the {n} track for recording"], None, small=True)
        vol = rng.randint(20, 90)
        col = rng.choice(["#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"])
        clips = "".join(f'<div style="position:absolute;left:{x}px;top:4px;width:{rng.randint(80, 220)}px;height:44px;background:{col};opacity:.85;border-radius:3px"></div>'
                        for x in sorted(rng.sample(range(0, 820, 40), rng.randint(2, 4))))
        rows.append(f'<div style="display:flex;height:54px;border-bottom:1px solid {t[5]}"><div class="pn" style="width:250px;display:flex;align-items:center;gap:6px;padding:0 8px">'
                    f'<span style="width:4px;height:36px;background:{col}"></span><span style="width:78px;white-space:nowrap;overflow:hidden" {s.el(f"tn_{i}", f"the {n} track name", n, [], None)}>{E(n)}</span>'
                    f'<span class="tbtn" {m}>M</span><span class="tbtn" {so}>S</span><span class="tbtn" {r} style="color:#e0342b">R</span>'
                    f'<span style="flex:1;height:4px;background:{t[5]};position:relative;margin-left:6px"><span style="position:absolute;left:{vol}%;top:-5px;width:6px;height:14px;background:{t[2]}"></span></span></div>'
                    f'<div style="flex:1;position:relative;overflow:hidden">{clips}</div></div>')
    add = icon_btn(s, "addtrack", "plus", "Add Track", "add a new track", 15, desc="the add track (plus) button")
    body = (mb + f'<div class="tb" style="gap:6px;height:40px">{transport}<span class="sep"></span><span class="pn" style="padding:2px 8px;font-family:Menlo,monospace" {bpm}>{rng.randint(70, 160)}.00 BPM</span>'
            f'<span class="pn" style="padding:2px 8px">4/4</span><span class="pn" style="padding:2px 8px;font-family:Menlo,monospace">{rng.randint(1, 64)}.{rng.randint(1, 4)}.1</span>'
            f'<span style="flex:1"></span>{add}</div><div class="muted" style="height:20px;padding:2px 260px;font-size:10px;font-family:Menlo,monospace">1 · · · 5 · · · 9 · · · 13 · · · 17 · · · 21</div>'
            + "".join(rows))
    for d, g in [("the Quantize button", "snap notes to the beat grid"), ("the Mixer window button", "open the mixer"),
                 ("the Punch-in button", "record only between two markers"), ("the Freeze track button", "freeze a track to save CPU")]:
        s.add_absent(d, g)
    s.meta["tracks"] = names
    s.html = page(t, body)
    return s


TEMPLATES = {"ide": (ide, 22), "spreadsheet": (sheet, 18), "image_editor": (image_editor, 18), "video_editor": (video_editor, 14),
             "cad": (cad, 16), "daw": (daw, 12)}
