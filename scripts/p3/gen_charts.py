"""Phase-3 Stage 0-I: rendered charts with constructed gold (families `chart_*`, source I, licence "generated").

We draw the charts ourselves (SVG / HTML -> PNG with the local headless Chromium through puppeteer-core, the renderer the
GUI generator uses: scripts/p3/gui_templates/render.mjs) from a data spec we invent (fictional brands, towns, people and
numbers), then ask typed questions whose answer follows from the spec:

  kinds   vbar, hbar (single / grouped), grouped, stacked, line (multi-series), pie, donut, scatter, table (as image), dual
          (bars on the left axis + a line on the right axis)
  tasks   extremum (choice)  which category / series / period is highest or lowest
          threshold (noul)   is a value above / below T (T in the question or in the state)
          readoff (choice)   value read to the nearest labelled tick (options one tick apart; the true value sits on a tick)
          diff / ratio (choice) difference or "about N times" between two values (both on ticks, options >= one tick apart)
          rank (choice)      order three items; trend (choice) shape of a series; crossover (choice) first period A > B
          claim (noul)       does the chart support a claim; relation (choice) scatter correlation; share (choice) pie share
          band (score)       severity level from printed bands; count (score) how many periods above T
Readability is part of the construction: without printed values every compared gap is >= 6% of the axis range (4 points
on a pie), read-off values sit exactly on a labelled tick >= 26 px apart, and the renderer reports every tick label, legend
entry and mark so a chart with a clipped, covered or overlapping label is dropped.

Unknown variants (~15%, gold null, parent_id = the answerable item): the question names a series / category that is not in
the chart (false_premise) or a period outside the axis (insufficient_evidence); or the chart is re-rendered with the
decisive part missing: a sticky note over the needed column, the legend removed from a multi-series chart, or the value
axis unlabelled (insufficient_evidence). `decide()` evaluates every world consistent with what the image shows (any value
under the note, any series-to-colour assignment, any axis scale) and the gold is kept only when all worlds agree, so the
same machinery keeps answerable questions on charts that carry an irrelevant note or no axis numbers.

Every row stores provenance.spec (the full render spec), params, html_sha256 and the image sha256, so tests re-derive the
gold and re-build the page.

    .venv/bin/python scripts/p3/gen_charts.py --seed c1 --count 5000 [--out data/p3/candidates/I-charts.jsonl]
    .venv/bin/python scripts/p3/gen_charts.py --variant-of parents.jsonl --variants-per 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html as _html
import io
import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
from candidate import validate, write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_image_joint import bench_image_shas, rng_for  # noqa: E402

GENERATOR = "scripts/p3/gen_charts.py"
IMG_DIR = ROOT / "data" / "p3" / "images" / "charts"
OUT = ROOT / "data" / "p3" / "candidates" / "I-charts.jsonl"
RENDER = ROOT / "scripts" / "p3" / "gui_templates" / "render.mjs"
E = _html.escape
UNKNOWN_SHARE = 0.15


class Skip(Exception):
    pass


class Missing(Exception):
    """The question needs something the chart does not show."""

    def __init__(self, reason="insufficient_evidence"):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------------------------------------------ invented names
BRAND_A = ["Vel", "Quor", "Tami", "Brin", "Oda", "Zell", "Poma", "Kiro", "Lume", "Sorb", "Yent", "Fenn", "Marro", "Tovi",
           "Wexa", "Ulma", "Rask", "Pello", "Nibu", "Cora", "Dessa", "Gault", "Hesk", "Ivor", "Jemba", "Olvi", "Praxa", "Seno"]
BRAND_B = ["wick", "nest", "loop", "pad", "deck", "ora", "ix", "hub", "field", "mint", "works", "base", "port", "lane",
           "grove", "stack", "tide", "yard", "line", "vale"]
TOWN_A = ["Nor", "Vel", "Kas", "Tor", "Bel", "Ard", "Mor", "Lin", "Hal", "Wes", "Fen", "Cal", "Dun", "Ros", "Ost", "Kel",
          "Bram", "Sel", "Tav", "Quen", "Arl", "Esk", "Grim", "Hol"]
TOWN_B = ["beck", "ford", "wick", "mere", "ton", "stad", "holm", "vale", "by", "port", "field", "moor", "haven", "crest",
          "brook", "ridge"]
FIRST = ["Anna", "Tomas", "Leila", "Marek", "Priya", "Jonah", "Sofia", "Kwame", "Ines", "Ravi", "Helga", "Omar", "Yuki",
         "Bruno", "Maeve", "Tariq", "Lena", "Diego", "Nadia", "Felix", "Chiara", "Emeka", "Ruth", "Sven", "Aiko", "Pavel"]
LAST = ["Kerr", "Novak", "Haddad", "Lindqvist", "Okafor", "Brandt", "Moreau", "Sato", "Ferreira", "Iqbal", "Duarte",
        "Keane", "Varga", "Mbeki", "Olsen", "Rossi", "Tanaka", "Wolfe", "Petrov", "Achterberg", "Quist", "Ncube"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def brand(rng):
    return rng.choice(BRAND_A) + rng.choice(BRAND_B)


def town(rng):
    return rng.choice(TOWN_A) + rng.choice(TOWN_B)


def uniq(rng, fn, n, avoid=()):
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        x = fn(rng)
        if x not in out and x not in avoid:
            out.append(x)
        if tries > 500:
            raise Skip("names")
    return out


def pool_names(kind):
    """Name generator for a series / category pool."""
    fixed = {
        "region": ["North", "South", "East", "West", "Central", "Coastal", "Highlands", "Islands"],
        "channel": ["Search", "Social", "Email", "Direct", "Referral", "Partners", "Video", "Ads"],
        "team": ["Billing", "Accounts", "Technical", "Shipping", "Returns", "Onboarding", "Payments"],
        "plant": ["Solar", "Wind", "Hydro", "Biogas", "Tidal", "Geothermal"],
        "department": ["Sales", "Finance", "Research", "Operations", "Marketing", "Legal", "Support", "Logistics"],
        "plan": ["Free", "Starter", "Team", "Business", "Enterprise", "Student"],
        "platform": ["iOS", "Android", "Web", "Desktop", "Tablet", "Smart TV"],
        "line": ["Line A", "Line B", "Line C", "Line D", "Line E", "Line F"],
        "budget": ["Salaries", "Rent", "Marketing", "IT", "Travel", "Training", "Insurance", "Utilities"],
        "answer": ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"],
        "mode": ["Car", "Bus", "Bicycle", "Walking", "Train", "Tram", "Scooter"],
    }
    if kind in fixed:
        return lambda rng, n, avoid=(): (lambda xs: xs)(_sample_fixed(rng, fixed[kind], n, avoid))
    gen = {"town": town, "brand": brand,
           "store": lambda r: brand(r) + " " + r.choice(["Store", "Outlet", "Market", "Shop"]),
           "product": lambda r: brand(r) + " " + r.choice(["S", "X", "One", "Pro", "Mini", "Air", "Go", "Max"]),
           "person": lambda r: r.choice(FIRST) + " " + r.choice(LAST),
           "clinic": lambda r: town(r) + " Clinic",
           "site": lambda r: town(r) + " site"}[kind]
    return lambda rng, n, avoid=(): uniq(rng, gen, n, avoid)


def _sample_fixed(rng, xs, n, avoid):
    xs = [x for x in xs if x not in avoid]
    if len(xs) < n:
        raise Skip("pool")
    return rng.sample(xs, n)


# domains: metric phrase, axis title, unit suffix for options, series pool, category kinds, typical max, decimals
DOMAINS = [
    dict(key="revenue", metric="revenue", axis="Revenue (k€)", unit=" k€", series="region", cats=["months", "quarters", "years"], top=(80, 900), dec=0, titles=["Revenue by region", "{brand} revenue", "Regional revenue, {year}"]),
    dict(key="units", metric="units sold", axis="Units sold", unit=" units", series="product", cats=["months", "quarters", "store"], top=(60, 2000), dec=0, titles=["Units sold", "{brand} unit sales", "Sales volume by product"]),
    dict(key="visits", metric="website visits", axis="Visits (thousands)", unit="k visits", series="channel", cats=["weeks", "months"], top=(20, 400), dec=0, titles=["Website visits by channel", "{brand}.com traffic", "Visits per channel"]),
    dict(key="tickets", metric="support tickets", axis="Tickets opened", unit=" tickets", series="team", cats=["days", "weeks"], top=(40, 600), dec=0, titles=["Support tickets", "Tickets opened per team", "{brand} help desk load"]),
    dict(key="energy", metric="energy output", axis="Output (MWh)", unit=" MWh", series="plant", cats=["months", "years"], top=(100, 5000), dec=0, titles=["Energy output by source", "{town} grid output", "Generation mix output"]),
    dict(key="rain", metric="rainfall", axis="Rainfall (mm)", unit=" mm", series="town", cats=["months"], top=(40, 300), dec=0, titles=["Monthly rainfall", "Rainfall in {year}", "Rainfall by town"]),
    dict(key="temp", metric="average temperature", axis="Temperature (°C)", unit=" °C", series="town", cats=["months"], top=(15, 40), dec=1, titles=["Average temperature", "Temperatures, {year}", "Mean daily temperature"]),
    dict(key="wait", metric="average wait time", axis="Wait (minutes)", unit=" min", series="clinic", cats=["days", "months"], top=(20, 120), dec=0, titles=["Average wait time", "Waiting times by clinic", "Patient wait, {year}"]),
    dict(key="headcount", metric="headcount", axis="Employees", unit=" staff", series="department", cats=["years", "quarters"], top=(20, 800), dec=0, titles=["Headcount by department", "{brand} staff numbers", "Employees per team"]),
    dict(key="orders", metric="online orders", axis="Orders", unit=" orders", series="store", cats=["days", "weeks", "months"], top=(50, 3000), dec=0, titles=["Online orders", "Orders per store", "{brand} orders"]),
    dict(key="costs", metric="operating costs", axis="Costs (k$)", unit=" k$", series="site", cats=["quarters", "years"], top=(50, 1200), dec=0, titles=["Operating costs by site", "Site costs", "{brand} operating costs"]),
    dict(key="subs", metric="subscribers", axis="Subscribers (thousands)", unit="k subscribers", series="plan", cats=["months", "quarters"], top=(10, 500), dec=0, titles=["Subscribers by plan", "{brand} subscriptions", "Paying users by plan"]),
    dict(key="defects", metric="defect rate", axis="Defects per 1,000 units", unit=" per 1,000", series="line", cats=["weeks", "months"], top=(8, 40), dec=1, titles=["Defect rate by line", "Quality report", "{town} plant defects"]),
    dict(key="score", metric="satisfaction score", axis="Score (0-100)", unit=" points", series="product", cats=["quarters", "years"], top=(100, 100), dec=0, titles=["Customer satisfaction", "Satisfaction by product", "{brand} survey scores"]),
    dict(key="downloads", metric="app downloads", axis="Downloads (thousands)", unit="k downloads", series="platform", cats=["months", "weeks"], top=(20, 900), dec=0, titles=["App downloads", "{brand} installs", "Downloads by platform"]),
    dict(key="trips", metric="trips", axis="Trips (thousands)", unit="k trips", series="mode", cats=["months", "years", "town"], top=(20, 400), dec=0, titles=["Trips by transport mode", "{town} travel survey", "Commuter trips"]),
    dict(key="sales_person", metric="sales", axis="Sales (k€)", unit=" k€", series="person", cats=["quarters", "months"], top=(30, 400), dec=0, titles=["Sales by rep", "Team sales, {year}", "{brand} sales team"]),
    dict(key="footfall", metric="visitors", axis="Visitors (thousands)", unit="k visitors", series="brand", cats=["days", "months", "town"], top=(5, 120), dec=0, titles=["Visitors per venue", "{town} footfall", "Weekly footfall"]),
]
PIE_DOMAINS = [
    dict(key="budget", title=["{brand} budget split", "Spending by category", "Annual budget, {year}"], series="budget", noun="category", whole="{po} budget", unit="%"),
    dict(key="traffic", title=["Traffic sources", "{brand}.com visits by source", "Where visitors came from"], series="channel", noun="source", whole="all {o} visits", unit="%"),
    dict(key="survey", title=["Survey answers", "How satisfied are you?", "{brand} customer survey"], series="answer", noun="answer", whole="all {o} survey responses", unit="%"),
    dict(key="mix", title=["Energy mix, {year}", "{town} electricity sources", "Generation by source"], series="plant", noun="source", whole="{po} total generation", unit="%"),
    dict(key="market", title=["Market share, {year}", "Share of sales by brand", "Smart speaker market"], series="brand", noun="brand", whole="the {o} market", unit="%"),
    dict(key="commute", title=["How staff commute", "{town} commuting survey", "Main travel mode"], series="mode", noun="mode", whole="all {o} commuters", unit="%"),
]
SCATTER_DOMAINS = [
    dict(key="ads", noun="store", names="store", x="Advertising spend (k€)", y="Sales (k€)", xm="advertising spend", ym="sales", xu=" k€", yu=" k€", xtop=(20, 200), ytop=(100, 900)),
    dict(key="price", noun="product", names="product", x="Price (€)", y="Units sold", xm="price", ym="units sold", xu=" €", yu=" units", xtop=(20, 200), ytop=(200, 2000)),
    dict(key="training", noun="employee", names="person", x="Training hours", y="Assessment score", xm="training hours", ym="assessment score", xu=" h", yu=" points", xtop=(20, 80), ytop=(100, 100)),
    dict(key="size", noun="shop", names="store", x="Floor area (m²)", y="Revenue (k€)", xm="floor area", ym="revenue", xu=" m²", yu=" k€", xtop=(200, 2000), ytop=(100, 1500)),
    dict(key="distance", noun="depot", names="town", x="Distance to hub (km)", y="Delivery time (h)", xm="distance to the hub", ym="delivery time", xu=" km", yu=" h", xtop=(50, 500), ytop=(10, 60)),
]
DUAL_DOMAINS = [
    dict(key="climate", bars="rainfall", bars_axis="Rainfall (mm)", bu=" mm", line="temperature", line_axis="Temperature (°C)", lu=" °C", btop=(60, 250), ltop=(20, 40), cats="months", title=["{town} climate", "Rain and temperature, {year}"]),
    dict(key="margin", bars="revenue", bars_axis="Revenue (k€)", bu=" k€", line="profit margin", line_axis="Margin (%)", lu="%", btop=(100, 1000), ltop=(20, 50), cats="quarters", title=["{brand} revenue and margin", "Revenue vs margin"]),
    dict(key="conv", bars="visits", bars_axis="Visits (thousands)", bu="k visits", line="conversion rate", line_axis="Conversion (%)", lu="%", btop=(20, 400), ltop=(5, 10), cats="months", title=["Traffic and conversion", "{brand}.com funnel"]),
    dict(key="basket", bars="orders", bars_axis="Orders", bu=" orders", line="average basket", line_axis="Average basket (€)", lu=" €", btop=(200, 3000), ltop=(40, 120), cats="weeks", title=["Orders and basket size", "{brand} shop weekly"]),
]

PALETTES = [
    ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#9c755f"],
    ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#000000"],
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#8d99ae", "#f4a261", "#6d597a", "#b56576"],
    ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00"],
    ["#5b8ff9", "#5ad8a6", "#5d7092", "#f6bd16", "#e8684a", "#6dc8ec", "#9270ca", "#ff9d4d"],
]
FONTS = [  # css family, width factor
    ("Helvetica, Arial, sans-serif", 0.58), ("Georgia, 'Times New Roman', serif", 0.58), ("Verdana, Geneva, sans-serif", 0.66),
    ("'Trebuchet MS', Helvetica, sans-serif", 0.58), ("Menlo, Monaco, monospace", 0.63), ("'Avenir Next', Avenir, sans-serif", 0.6),
    ("Tahoma, Verdana, sans-serif", 0.6), ("'Times New Roman', Times, serif", 0.52),
    ("'Arial Narrow', Arial, sans-serif", 0.5),
]
BGS = [("#ffffff", "#222222", "#dddddd", "#555555"), ("#fbfaf6", "#2b2b2b", "#e2ded3", "#6a6559"),
       ("#f4f6f9", "#1d2330", "#d5dbe5", "#5d6677"), ("#1e2127", "#e6e6e6", "#3a3f48", "#a9afb9"),
       ("#ffffff", "#000000", "#e8e8e8", "#444444"), ("#fffdf5", "#333333", "#ece6cf", "#6b6552")]


# ------------------------------------------------------------------------------------------------ formatting
def fmt(v: float, dec: int = 0) -> str:
    return f"{v:,.{dec}f}"


def fmt_opt(v: float, dec: int, unit: str) -> str:
    return fmt(v, dec) + unit


def nice_steps(top: float, dec: int):
    """(step, n_ticks) options whose ymax covers `top`."""
    out = []
    for e in range(-1, 5):
        for m in (1, 2, 5, 2.5):
            s = m * 10 ** e
            if dec == 0 and (s < 1 or s != int(s)):
                continue
            if dec == 1 and round(s * 10) != s * 10:
                continue
            for n in (4, 5, 6, 8):
                if top <= s * n <= top * 1.6:
                    out.append((s, n))
    return out


def spoken_cat(spec, c):
    return spec.get("spoken", {}).get(c, c)


# ------------------------------------------------------------------------------------------------ categories
def owned_title(rng, templates):
    """(title, owner): every chart names its owner (a brand or a town) in the title; questions refer to it."""
    owner = brand(rng) if rng.random() < 0.6 else town(rng)
    t = rng.choice(templates)
    year = rng.randint(2018, 2026)
    if "{brand}" in t or "{town}" in t:
        return t.format(brand=owner, town=owner, year=year), owner
    t = t.format(year=year)
    return (rng.choice([f"{owner}: {t[:1].lower() + t[1:]}", f"{t} · {owner}", f"{owner} — {t[:1].lower() + t[1:]}"]), owner)


def make_categories(rng, kind, n, sp_pool=None):
    """(labels, spoken names, kind noun, noun plural, out-of-range candidates, ordered?)"""
    if kind == "months":
        n = min(n, 12)
        start = rng.randint(0, 12 - n)
        idx = list(range(start, start + n))
        labs = [MONTHS[i][:3] for i in idx]
        spoken = {MONTHS[i][:3]: MONTHS[i] for i in idx}
        outside = [MONTHS[i] for i in range(12) if i not in idx]
        return labs, spoken, "month", "months", outside, True
    if kind == "quarters":
        y = rng.randint(2019, 2026)
        q0 = rng.randint(1, 4)
        labs = []
        for k in range(n):
            q = (q0 - 1 + k) % 4 + 1
            yy = y + (q0 - 1 + k) // 4
            labs.append(f"Q{q} {yy}")
        after = [(q0 - 1 + n) % 4 + 1, y + (q0 - 1 + n) // 4]
        before = [(q0 - 2) % 4 + 1, y + (q0 - 2) // 4]
        return labs, {}, "quarter", "quarters", [f"Q{after[0]} {after[1]}", f"Q{before[0]} {before[1]}"], True
    if kind == "years":
        y = rng.randint(2008, 2026 - n + 1)
        labs = [str(y + k) for k in range(n)]
        return labs, {}, "year", "years", [str(y + n), str(y - 1)], True
    if kind == "weeks":
        w = rng.randint(1, 40)
        labs = [f"Week {w + k}" for k in range(n)]
        return labs, {}, "week", "weeks", [f"Week {w + n}", f"Week {w - 1}" if w > 1 else f"Week {w + n + 1}"], True
    if kind == "days":
        n = min(n, 7)
        labs = [d[:3] for d in DAYS[:n]]
        spoken = {d[:3]: d for d in DAYS[:n]}
        return labs, spoken, "day", "days", [d for d in DAYS[n:]] or [], True
    gen = pool_names(kind)
    names = gen(rng, n + 2)
    noun = {"store": "store", "town": "town", "product": "product", "brand": "venue"}.get(kind, kind)
    return names[:n], {}, noun, noun + "s", names[n:], False


# ------------------------------------------------------------------------------------------------ spec sampling
def pick_style(rng, W=None, H=None, small=False):
    bg = rng.choice(BGS)
    font, wf = rng.choice(FONTS)
    fs = rng.choice([11, 12, 12, 13, 13, 14, 15]) if not small else rng.choice([10, 11])
    return {"bg": bg[0], "fg": bg[1], "grid_c": bg[2], "muted": bg[3], "font": font, "wf": wf, "fs": fs,
            "palette": rng.randrange(len(PALETTES)), "pal_shift": rng.randrange(3),
            "bar_round": rng.choice([0, 0, 2, 4]), "line_w": rng.choice([2, 2.5, 3]), "markers": rng.random() < 0.7,
            "legend_pos": rng.choice(["top", "top", "right", "bottom"]), "rotate": rng.random() < 0.15,
            "title_align": rng.choice(["left", "left", "center"]), "frame": rng.random() < 0.3,
            "noise": rng.random() < 0.12}


def sample_values(rng, n, ymax, step, dec, labelled, on_tick_share=0.5, lo=0.08, hi=0.97):
    vals = []
    grain = step / 2 if (dec == 0 and (step / 2) == int(step / 2)) or dec == 1 else step
    for _ in range(n):
        if labelled:
            v = round(rng.uniform(lo, hi) * ymax, dec)
        elif rng.random() < on_tick_share:
            v = step * rng.randint(max(1, math.ceil(lo * ymax / step)), max(1, int(hi * ymax / step)))
        else:
            v = grain * rng.randint(max(1, math.ceil(lo * ymax / grain)), max(1, int(hi * ymax / grain)))
        vals.append(round(v, dec))
    return vals


def shaped_series(rng, n, ymax, step, dec, labelled, shape):
    """Values with a clear shape (every step >= 8% of the axis) for trend / crossover questions."""
    grain = step / 2 if (dec == 0 and (step / 2) == int(step / 2)) or dec == 1 else step
    minstep = max(grain, 0.08 * ymax)
    for _ in range(40):
        k = rng.randint(1, n - 2) if n >= 3 else 1
        signs = {"rise": [1] * (n - 1), "fall": [-1] * (n - 1),
                 "peak": [1] * k + [-1] * (n - 1 - k), "valley": [-1] * k + [1] * (n - 1 - k)}[shape]
        span = sum(1 for _ in signs)
        budget = 0.85 * ymax
        stepv = min(budget / max(1, max(sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0))), 0.3 * ymax)
        if stepv < minstep:
            return None
        v = rng.uniform(0.08, 0.2) * ymax if signs[0] > 0 else rng.uniform(0.8, 0.95) * ymax
        if shape == "valley":
            v = rng.uniform(0.75, 0.95) * ymax
        if shape == "peak":
            v = rng.uniform(0.08, 0.25) * ymax
        out = [v]
        for s in signs:
            d = rng.uniform(minstep, max(minstep, stepv)) * s
            out.append(out[-1] + d)
        if min(out) < 0.04 * ymax or max(out) > 0.99 * ymax:
            continue
        if labelled:
            out = [round(x, dec) for x in out]
        else:
            out = [round(round(x / grain) * grain, dec) for x in out]
        diffs = [b - a for a, b in zip(out, out[1:])]
        if all(abs(d) >= minstep * 0.99 and (d > 0) == (s > 0) for d, s in zip(diffs, signs)) and span:
            return out
    return None


def base_spec(rng, kind, sid):
    small = rng.random() < 0.18
    st = pick_style(rng, small=small)
    W = rng.choice([640, 720, 800, 880, 960, 1024, 1120, 1280])
    H = rng.choice([420, 460, 500, 540, 600, 640, 720])
    if W / H > 2.2:
        H = int(W / 2)
    if small and W > 880:
        st["fs"] = rng.choice([11, 12])
    st["fs"] += int((W - 640) / 256)
    return {"id": sid, "kind": kind, "size": [W, H], "style": st, "cover": None,
            "show": {"legend": True, "data_labels": rng.random() < 0.4, "y_labels": True, "grid": rng.random() < 0.75,
                     "direct_labels": False}}


NON_ADDITIVE = {"temp", "wait", "defects", "score"}


def make_cartesian(rng, sid, kind):
    d = rng.choice([x for x in DOMAINS if kind != "stacked" or x["key"] not in NON_ADDITIVE])
    sp = base_spec(rng, kind, sid)
    ck = rng.choice(d["cats"])
    if kind in ("vbar", "hbar"):
        n_series = 1 if kind == "vbar" or rng.random() < 0.6 else rng.randint(2, 3)
        ncat = rng.randint(4, 9 if kind == "vbar" else 8)
    elif kind == "grouped":
        n_series, ncat = rng.randint(2, 4), rng.randint(3, 6)
    elif kind == "stacked":
        n_series, ncat = rng.randint(2, 4), rng.randint(3, 8)
    else:  # line
        n_series, ncat = rng.randint(1, 4), rng.randint(5, 12)
        if ck not in ("months", "quarters", "years", "weeks", "days"):
            ck = "months"
    if kind == "line" and ck == "days":
        ncat = min(ncat, 7)
    labs, spoken, cnoun, cnouns, outside, ordered = make_categories(rng, ck, ncat)
    ser_gen = pool_names(d["series"])
    snames = ser_gen(rng, n_series + 2)
    series_names, absent = snames[:n_series], snames[n_series:]
    top = rng.uniform(*d["top"]) if d["top"][0] != d["top"][1] else d["top"][0]
    if kind == "stacked":
        top *= 1.0
    steps = nice_steps(top, d["dec"])
    if not steps:
        raise Skip("steps")
    step, nt = rng.choice(steps)
    ymax = round(step * nt, d["dec"])
    labelled = sp["show"]["data_labels"]
    if kind == "stacked":
        sp["show"]["data_labels"] = labelled = rng.random() < 0.5    # totals printed on top
    series = []
    shapes = None
    if kind == "line" or (kind in ("vbar", "grouped") and ordered and rng.random() < 0.35):
        shapes = [rng.choice(["rise", "fall", "peak", "valley", None]) for _ in series_names]
    for j, nm in enumerate(series_names):
        vals = None
        if shapes and shapes[j]:
            vals = shaped_series(rng, len(labs), ymax, step, d["dec"], labelled, shapes[j])
        if vals is None:
            if kind == "stacked":
                vals = sample_values(rng, len(labs), ymax / n_series, step / 2 if step / 2 == int(step / 2) or d["dec"] else step,
                                     d["dec"], labelled, lo=0.15, hi=0.95)
            else:
                vals = sample_values(rng, len(labs), ymax, step, d["dec"], labelled)
        series.append({"name": nm, "values": vals})
    if kind == "stacked":
        tot = [sum(s["values"][i] for s in series) for i in range(len(labs))]
        if max(tot) > ymax:
            raise Skip("stack")
    # ratio pair: two single-series bars on ticks with an exact ratio
    if kind in ("vbar", "hbar") and n_series == 1 and rng.random() < 0.35:
        r = rng.choice([1.5, 2, 3, 4])
        bmax = int(nt / r)
        if bmax >= 1:
            b = step * rng.randint(1, bmax) * (2 if r == 1.5 else 1)
            a = b * r
            if a <= ymax and a == round(a, d["dec"]):
                i, k = rng.sample(range(len(labs)), 2)
                series[0]["values"][i], series[0]["values"][k] = round(a, d["dec"]), round(b, d["dec"])
    if kind == "line" and len(series) >= 2 and rng.random() < 0.4:   # crossing pair
        a = shaped_series(rng, len(labs), ymax, step, d["dec"], labelled, "rise")
        b = shaped_series(rng, len(labs), ymax, step, d["dec"], labelled, "fall")
        if a and b:
            series[0]["values"], series[1]["values"] = a, b
    if n_series == 1:
        title, owner = owned_title(rng, ["{brand} " + d["metric"], cap(d["metric"]) + ", {year}", cap(d["metric"]) + " at {town}",
                                         cap(d["metric"]) + " by " + cnoun])
    else:
        title, owner = owned_title(rng, d["titles"])
    sp.update({"domain": d["key"], "title": title, "owner": owner,
               "y_title": d["axis"], "unit": d["unit"], "metric": d["metric"], "dec": d["dec"], "categories": labs,
               "spoken": spoken, "cat_noun": cnoun, "cat_nouns": cnouns, "ordered": ordered, "outside": outside[:3],
               "absent_series": absent, "series_noun": d["series"], "series": series, "y_max": ymax, "y_step": step})
    if n_series == 1:
        sp["show"]["legend"] = False
    elif kind == "stacked" and rng.random() < 0.3:
        sp["show"]["legend"] = False        # stack totals stay answerable; questions about a segment become unknown
    elif kind == "line" and rng.random() < 0.35:
        sp["show"]["direct_labels"] = True
        sp["show"]["legend"] = False
    if kind == "line" and n_series > 2:
        sp["show"]["data_labels"] = False
    if rng.random() < 0.1 and not sp["show"]["data_labels"] and kind != "stacked":
        sp["show"]["data_labels"] = True
        sp["show"]["y_labels"] = False          # values printed, axis numbers dropped
        sp["show"]["grid"] = rng.random() < 0.3
    if rng.random() < 0.06 and kind in ("vbar", "grouped", "line", "hbar"):
        sp["cover"] = rng.choice(labs)           # an irrelevant-or-not sticky note: decide() sorts it out
    return sp


def make_pie(rng, sid, donut):
    d = rng.choice(PIE_DOMAINS)
    sp = base_spec(rng, "donut" if donut else "pie", sid)
    n = rng.randint(3, 7)
    names = pool_names(d["series"])(rng, n + 2)
    labelled = rng.random() < 0.6
    for _ in range(50):
        cuts = sorted(rng.sample(range(3, 97), n - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [100])]
        if min(parts) >= 3:
            break
    else:
        raise Skip("pie")
    sp["show"].update({"data_labels": labelled, "legend": True})
    title, owner = owned_title(rng, d["title"])
    sp.update({"domain": d["key"], "title": title, "owner": owner,
               "unit": "%", "dec": 0, "categories": names[:n], "spoken": {}, "cat_noun": d["noun"], "cat_nouns": d["noun"] + "s",
               "whole": d["whole"].format(o=owner, po=P(owner)), "ordered": False, "outside": names[n:], "absent_series": [], "series_noun": d["noun"],
               "series": [{"name": "share", "values": [float(p) for p in parts]}], "y_max": 100, "y_step": 10, "metric": "share",
               "start_angle": rng.choice([0, 0, 30, 90, 180, 270]), "label_mode": rng.choice(["inside", "legend"])})
    return sp


def make_scatter(rng, sid):
    d = rng.choice(SCATTER_DOMAINS)
    sp = base_spec(rng, "scatter", sid)
    n = rng.randint(5, 10)
    names = pool_names(d["names"])(rng, n + 2)
    xs_steps = nice_steps(rng.uniform(*d["xtop"]), 0)
    ys_steps = nice_steps(rng.uniform(*d["ytop"]) if d["ytop"][0] != d["ytop"][1] else 100, 0)
    if not xs_steps or not ys_steps:
        raise Skip("steps")
    xstep, xn = rng.choice(xs_steps)
    ystep, yn = rng.choice(ys_steps)
    xmax, ymax = xstep * xn, ystep * yn
    rel = rng.choice(["positive", "negative", "none"])
    for _ in range(80):
        xs = [round(rng.uniform(0.08, 0.95) * xmax) for _ in range(n)]
        if rel == "none":
            ys = [round(rng.uniform(0.1, 0.92) * ymax) for _ in range(n)]
        else:
            sl = 1 if rel == "positive" else -1
            ys = [round(max(0.05, min(0.95, 0.5 + sl * (x / xmax - 0.5) * 0.9 + rng.gauss(0, 0.05))) * ymax) for x in xs]
        r = pearson(xs, ys)
        if (rel == "positive" and r > 0.85) or (rel == "negative" and r < -0.85) or (rel == "none" and abs(r) < 0.12):
            break
    else:
        raise Skip("scatter")
    sp["show"].update({"legend": False, "data_labels": False})
    title, owner = owned_title(rng, [f"{d['ym'].capitalize()} vs {d['xm']}", "{brand}: " + d["ym"] + " vs " + d["xm"]])
    sp.update({"domain": d["key"], "title": title, "owner": owner, "x_title": d["x"], "y_title": d["y"],
               "unit": d["yu"], "x_unit": d["xu"], "metric": d["ym"], "x_metric": d["xm"], "dec": 0, "categories": names[:n],
               "spoken": {}, "cat_noun": d["noun"], "cat_nouns": d["noun"] + "s", "ordered": False, "outside": names[n:],
               "absent_series": [], "series_noun": d["noun"],
               "series": [{"name": "x", "values": [float(x) for x in xs]}, {"name": "y", "values": [float(y) for y in ys]}],
               "x_max": xmax, "x_step": xstep, "y_max": ymax, "y_step": ystep})
    return sp


def make_table(rng, sid):
    d = rng.choice(DOMAINS)
    sp = base_spec(rng, "table", sid)
    ck = rng.choice(["months", "quarters", "years", "weeks"])
    ncol = rng.randint(3, 6)
    cols, spoken, cnoun, cnouns, outside, _ = make_categories(rng, ck, ncol)
    nrow = rng.randint(3, 8)
    rows = pool_names(d["series"])(rng, nrow + 2) if d["series"] not in ("plant", "platform", "plan", "line") or nrow <= 4 else None
    if rows is None:
        raise Skip("rows")
    top = rng.uniform(*d["top"]) if d["top"][0] != d["top"][1] else d["top"][0]
    series = [{"name": c, "values": [round(rng.uniform(0.1, 1.0) * top, d["dec"]) for _ in range(nrow)]} for c in cols]
    sp["show"].update({"legend": False, "data_labels": True})
    title, owner = owned_title(rng, d["titles"])
    sp.update({"domain": d["key"], "title": title, "owner": owner,
               "y_title": d["axis"], "unit": d["unit"], "metric": d["metric"], "dec": d["dec"], "categories": rows[:nrow],
               "spoken": spoken, "row_pool": d["series"], "cat_noun": d["series"] if d["series"] not in ("person",) else "person",
               "cat_nouns": d["series"] + "s", "col_noun": cnoun, "ordered": False, "outside": rows[nrow:],
               "absent_series": outside[:2], "series_noun": cnoun, "series": series, "y_max": top, "y_step": 1,
               "table_style": {"zebra": rng.random() < 0.5, "lines": rng.choice(["rows", "grid", "none"]),
                               "head_bg": rng.random() < 0.6, "total_row": False}})
    if rng.random() < 0.08:
        sp["cover"] = {"row": rng.choice(rows[:nrow]), "col": rng.choice(cols)}
    # size the canvas to the table (the page is cropped to the viewport)
    fs, wf = sp["style"]["fs"] + 1, sp["style"]["wf"]
    name_w = max(tw(x, fs, wf) for x in rows[:nrow] + [sp["cat_noun"]]) + 1.6 * fs
    col_w = sum(max(tw(c, fs, wf), tw(fmt(max(s["values"]), d["dec"]), fs, wf)) + 1.6 * fs for c, s in zip(cols, series))
    need_w = name_w + col_w + 60
    W = max(need_w + rng.randint(20, 160), rng.choice([520, 600, 680]))
    H = 22 + (fs + 4) * 1.35 + fs * 1.4 + 12 + (nrow + 1) * (fs * 1.25 + 0.9 * fs + 2) + rng.randint(30, 70)
    if W > 1280:
        raise Skip("table wide")
    sp["size"] = [int(W), int(H)]
    return sp


def make_dual(rng, sid):
    d = rng.choice(DUAL_DOMAINS)
    sp = base_spec(rng, "dual", sid)
    n = rng.randint(4, 12 if d["cats"] == "months" else 8)
    labs, spoken, cnoun, cnouns, outside, _ = make_categories(rng, d["cats"], n)
    bs = nice_steps(rng.uniform(*d["btop"]), 0)
    ls = nice_steps(rng.uniform(*d["ltop"]), 0)
    if not bs or not ls:
        raise Skip("steps")
    bstep, bn = rng.choice(bs)
    lstep, ln = rng.choice(ls)
    # both axes get the same number of ticks so gridlines are shared
    if bn != ln:
        ln = bn
    bmax, lmax = bstep * bn, lstep * ln
    labelled = False
    sp["show"].update({"data_labels": False, "legend": True, "grid": True})
    bars = sample_values(rng, n, bmax, bstep, 0, labelled)
    line = sample_values(rng, n, lmax, lstep, 0, labelled, on_tick_share=0.7, lo=0.12, hi=0.95)
    title, owner = owned_title(rng, d["title"])
    sp.update({"domain": d["key"], "title": title, "owner": owner,
               "y_title": d["bars_axis"], "y2_title": d["line_axis"], "unit": d["bu"], "unit2": d["lu"], "metric": d["bars"],
               "metric2": d["line"], "dec": 0, "categories": labs, "spoken": spoken, "cat_noun": cnoun, "cat_nouns": cnouns,
               "ordered": True, "outside": outside[:2], "absent_series": [], "series_noun": "measure",
               "series": [{"name": d["bars"], "values": bars}, {"name": d["line"], "values": line}],
               "y_max": bmax, "y_step": bstep, "y2_max": lmax, "y2_step": lstep})
    return sp


KIND_WEIGHTS = {"vbar": 0.17, "hbar": 0.1, "grouped": 0.13, "stacked": 0.08, "line": 0.18, "pie": 0.07, "donut": 0.04,
                "scatter": 0.08, "table": 0.09, "dual": 0.06}


def make_spec(rng, sid):
    kind = rng.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
    if kind in ("pie", "donut"):
        sp = make_pie(rng, sid, kind == "donut")
    elif kind == "scatter":
        sp = make_scatter(rng, sid)
    elif kind == "table":
        sp = make_table(rng, sid)
    elif kind == "dual":
        sp = make_dual(rng, sid)
    else:
        sp = make_cartesian(rng, sid, kind)
    if sp["kind"] == "line" and len(sp["series"]) > 1:
        nudge_lines(sp)
    if len(colours(sp)) < max(len(sp["series"]), len(sp["categories"]) if sp["kind"] in ("pie", "donut") else 0):
        raise Skip("palette")
    lay = layout(sp)          # may switch off labels that would collide, or Skip
    sp["show"] = lay["show"]
    check_marks(sp, lay)
    return sp


def nudge_lines(sp):
    """Move a line point that would sit on top of another series' point (same x) up or down by one grain."""
    ymax, step, dec = sp["y_max"], sp["y_step"], sp["dec"]
    grain = step / 2 if (dec == 0 and (step / 2) == int(step / 2)) or dec == 1 else step
    g = 0.04 * ymax
    mv = math.ceil(g / grain) * grain
    for i in range(len(sp["categories"])):
        for _ in range(6):
            order = sorted(range(len(sp["series"])), key=lambda j: sp["series"][j]["values"][i])
            clash = [(a, b) for a, b in zip(order, order[1:])
                     if sp["series"][b]["values"][i] - sp["series"][a]["values"][i] < g]
            if not clash:
                break
            a, b = clash[0]
            vb = sp["series"][b]["values"][i]
            up = round(vb + mv, dec)
            dn = round(sp["series"][a]["values"][i] - mv, dec)
            if up <= 0.99 * ymax:
                sp["series"][b]["values"][i] = up
            elif dn >= 0.04 * ymax:
                sp["series"][a]["values"][i] = dn
            else:
                raise Skip("line clash")


def scatter_boxes(sp, L):
    """(label boxes, point boxes) exactly as svg_cartesian draws them."""
    fs, wf = L["fs"], sp["style"]["wf"]
    x0, x1, y0, y1 = L["x0"], L["x1"], L["y0"], L["y1"]
    xs, ys = sp["series"][0]["values"], sp["series"][1]["values"]
    labs, pts = [], []
    for i, c in enumerate(sp["categories"]):
        px = x0 + (x1 - x0) * xs[i] / sp["x_max"]
        py = y1 - (y1 - y0) * ys[i] / sp["y_max"]
        w = tw(c, fs - 1, wf)
        lx = px + 8 if px + 8 + w < x1 else px - 8
        bx = (lx, lx + w) if lx > px else (lx - w, lx)
        labs.append((bx[0] - 2, py - fs * 0.6, bx[1] + 2, py + fs * 0.5))
        pts.append((px - 7, py - 7, px + 7, py + 7))
    return labs, pts


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check_marks(sp, L):
    """Reject specs whose marks would hide each other: scatter labels / points colliding, line points on top of each other."""
    if sp["kind"] == "scatter":
        labs, pts = scatter_boxes(sp, L)
        boxes = labs + pts
        n = len(labs)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if j == i + n:          # a label and its own point
                    continue
                if _overlap(boxes[i], boxes[j]):
                    raise Skip("scatter overlap")
        for b in labs:
            if b[0] < L["x0"] or b[2] > L["x1"] or b[1] < L["y0"] - 4 or b[3] > L["y1"]:
                raise Skip("scatter label outside")
    if sp["kind"] == "line" and len(sp["series"]) > 1:
        g = 0.04 * sp["y_max"]
        for i in range(len(sp["categories"])):
            vals = sorted(s["values"][i] for s in sp["series"])
            if any(b - a < g for a, b in zip(vals, vals[1:])):
                raise Skip("line points coincide")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# ------------------------------------------------------------------------------------------------ layout + SVG
def tw(s, fs, wf):
    return len(str(s)) * fs * wf + 2


def _lum(hexc):
    c = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def colours(sp):
    """The palette rotated by the style shift, without colours too close to the background (contrast < 2)."""
    pal = PALETTES[sp["style"]["palette"]]
    k = sp["style"]["pal_shift"]
    pal = pal[k:] + pal[:k]
    return [c for c in pal if contrast(c, sp["style"]["bg"]) >= 2.0]


def layout(sp):
    """Pure geometry shared by planning and rendering. Returns the plot box, positions and the final `show` flags."""
    kind, st, show = sp["kind"], sp["style"], dict(sp["show"])
    W, H = sp["size"]
    fs, wf = st["fs"], st["wf"]
    ft = fs + 5
    pad = 16
    L = {"W": W, "H": H, "fs": fs, "ft": ft, "pad": pad, "show": show}
    top = pad + ft + 8
    if kind == "table":
        return L
    if kind in ("pie", "donut"):
        names = sp["categories"]
        vals = sp["series"][0]["values"]
        leg = [f"{n} ({fmt(v)}%)" if show["data_labels"] and sp["label_mode"] == "legend" else n for n, v in zip(names, vals)]
        if show["data_labels"] and sp["label_mode"] == "inside" and min(vals) < 7:
            sp_mode = "legend"
            leg = [f"{n} ({fmt(v)}%)" for n, v in zip(names, vals)]
        else:
            sp_mode = sp["label_mode"]
        legw = max(tw(x, fs, wf) for x in leg) + 30
        if W - legw - 2 * pad < 260:
            raise Skip("pie width")
        r = min((W - legw - 3 * pad) / 2, (H - top - pad) / 2) - 6
        if r < 110:
            raise Skip("pie small")
        L.update({"cx": pad + r + 10, "cy": top + (H - top - pad) / 2, "r": r, "legend": leg, "legend_x": pad + 2 * r + 50,
                  "mode": sp_mode})
        if L["legend_x"] + legw > W - 4:
            raise Skip("pie legend")
        if len(leg) * (fs + 12) > H - top - pad:
            raise Skip("pie legend h")
        return L
    cats = sp["categories"]
    n = len(cats)
    ser = sp["series"]
    # legend
    leg_items = [s["name"] for s in ser] if show["legend"] and kind not in ("scatter",) else []
    if kind == "dual" and show["legend"]:
        leg_items = [f"{sp['series'][0]['name'].capitalize()} (bars, left axis)", f"{sp['series'][1]['name'].capitalize()} (line, right axis)"]
    legend_rows, right_w = [], 0
    pos = st["legend_pos"] if kind != "dual" else "top"
    if leg_items:
        if pos == "right":
            right_w = max(tw(x, fs, wf) for x in leg_items) + 36
            if right_w > W * 0.3:
                pos = "top"
                right_w = 0
        if pos in ("top", "bottom"):
            row, x = [], 0
            for it in leg_items:
                w_ = tw(it, fs, wf) + 34
                if x + w_ > W - 2 * pad and row:
                    legend_rows.append(row)
                    row, x = [], 0
                row.append((it, x))
                x += w_
            legend_rows.append(row)
    L["legend_pos"], L["legend_rows"], L["legend_items"] = pos, legend_rows, leg_items
    if pos == "top" and legend_rows:
        top += len(legend_rows) * (fs + 10) + 4
    bottom_leg = len(legend_rows) * (fs + 10) + 6 if pos == "bottom" and legend_rows else 0
    horiz = kind == "hbar"
    # value ticks
    vmax, vstep = sp["y_max"], sp["y_step"]
    ticks = [round(k * vstep, 6) for k in range(int(round(vmax / vstep)) + 1)]
    L["ticks"] = ticks
    tick_lab_w = max(tw(fmt(t, 1 if vstep < 1 else 0), fs, wf) for t in ticks) if show["y_labels"] else 0
    y2w = 0
    if kind == "dual":
        t2 = [round(k * sp["y2_step"], 6) for k in range(int(round(sp["y2_max"] / sp["y2_step"])) + 1)]
        L["ticks2"] = t2
        y2w = max(tw(fmt(t), fs, wf) for t in t2) + 8 + fs + 10
    dl_w = 0
    if kind == "line" and show["direct_labels"]:
        dl_w = max(tw(s["name"], fs, wf) for s in ser) + 16
    if horiz:
        cat_w = max(tw(c, fs, wf) for c in cats) + 10
        x0 = pad + cat_w
        x1 = W - pad - right_w - (tw(fmt(vmax, sp["dec"]), fs - 1, wf) + 10 if show["data_labels"] else 12)
        y0 = top + 4
        y1 = H - pad - bottom_leg - (fs + 10 if show["y_labels"] else 4) - (fs + 12)
        if x1 - x0 < 240 or y1 - y0 < 150:
            raise Skip("hbar small")
        band = (y1 - y0) / n
        nser = len(ser)
        bh = band * 0.72 / nser
        if bh < fs * 0.9:
            raise Skip("hbar band")
        L.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "band": band, "bw": bh, "rotate": False})
        L["px_per_step"] = (x1 - x0) / (vmax / vstep)
        return L
    ytitle_w = fs + 10 if sp.get("y_title") and show["y_labels"] else (fs + 10 if sp.get("y_title") else 0)
    x0 = pad + ytitle_w + tick_lab_w + 8
    x1 = W - pad - right_w - y2w - dl_w - 6
    band = (x1 - x0) / n
    labw = max(tw(c, fs, wf) for c in cats)
    rotate = st["rotate"] or labw > band * 0.95
    if rotate:
        ang = math.radians(40)
        if band * math.sin(ang) < fs + 2:
            raise Skip("x labels crowd")
        xl_h = labw * math.sin(ang) + fs * math.cos(ang) + 12
        # the first label runs left from its tick
        need_left = labw * math.cos(ang) - band / 2 + 4
        if x0 < need_left + pad / 2:
            x0 = need_left + pad / 2
            band = (x1 - x0) / n
    else:
        xl_h = fs + 12
    if kind == "scatter":
        rotate, xl_h = False, fs + 12
    xt_h = fs + 10 if sp.get("x_title") else 0
    y0 = top + fs * 0.6 + 6
    y1 = H - pad - bottom_leg - xl_h - xt_h
    if y1 - y0 < 170 or x1 - x0 < 260:
        raise Skip("plot small")
    L.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "band": band, "rotate": rotate, "xl_h": xl_h})
    L["px_per_step"] = (y1 - y0) / (vmax / vstep)
    if kind == "scatter":
        L["px_per_xstep"] = (x1 - x0) / (sp["x_max"] / sp["x_step"])
        return L
    nser = len(ser) if kind in ("grouped", "vbar") else 1
    bw = band * 0.7 / nser
    L["bw"] = bw
    if kind in ("grouped", "vbar", "stacked", "dual") and bw < 6:
        raise Skip("bars thin")
    # data labels must fit: bar labels need the bar width; line labels must not collide between series
    if show["data_labels"] and kind in ("grouped", "vbar"):
        wmax = max(tw(fmt(v, sp["dec"]), fs - 1, wf) for s in ser for v in s["values"])
        if wmax > bw * (1.0 if nser == 1 else 1.05) + 4:
            show["data_labels"] = False if show["y_labels"] else show["data_labels"]
            if not show["y_labels"]:
                raise Skip("labels do not fit")
    if show["data_labels"] and kind == "stacked":
        wmax = max(tw(fmt(sum(s["values"][i] for s in ser), sp["dec"]), fs - 1, wf) for i in range(n))
        if wmax > band * 0.95:
            show["data_labels"] = False
    if show["data_labels"] and kind == "line":
        wmax = max(tw(fmt(v, sp["dec"]), fs - 1, wf) for s in ser for v in s["values"])
        if wmax > band * 0.95:
            if show["y_labels"]:
                show["data_labels"] = False
            else:
                raise Skip("line labels")
        elif len(ser) == 2:
            pp = (y1 - y0) / vmax
            if any(abs(a - b) * pp < fs + 4 for a, b in zip(ser[0]["values"], ser[1]["values"])):
                if show["y_labels"]:
                    show["data_labels"] = False
                else:
                    raise Skip("line labels collide")
    if kind == "line" and show["direct_labels"]:
        pp = (y1 - y0) / vmax
        last = sorted(s["values"][-1] * pp for s in ser)
        if any(b - a < fs + 3 for a, b in zip(last, last[1:])):
            show["direct_labels"] = False
            show["legend"] = True
            sp2 = dict(sp, show=show)
            return layout(sp2)
    return L


def _t(x, y, s, fs, fill, anchor="start", weight="normal", el=None, extra=""):
    ea = f' data-el="{E(el)}"' if el else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
            f'{ea} {extra}>{E(str(s))}</text>')


def render_html(sp) -> str:
    """The page for a spec (deterministic; its sha256 is stored in provenance)."""
    L = layout(copy.deepcopy(sp))
    st = sp["style"]
    W, H, fs, ft, pad = L["W"], L["H"], L["fs"], L["ft"], L["pad"]
    fg, muted, grid_c, bg = st["fg"], st["muted"], st["grid_c"], st["bg"]
    show = sp["show"]
    cols = colours(sp)
    out = []
    kind = sp["kind"]
    if kind == "table":
        return render_table(sp)
    tx = pad if st["title_align"] == "left" else W / 2
    out.append(_t(tx, pad + ft, sp["title"], ft, fg, "start" if st["title_align"] == "left" else "middle", "bold", el="title"))
    overlays = []
    if kind in ("pie", "donut"):
        out += svg_pie(sp, L, cols)
    else:
        out += svg_cartesian(sp, L, cols, overlays)
    frame = f'border:1px solid {grid_c};' if st["frame"] else ""
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:{bg}}}'
            f'body{{font-family:{st["font"]};}} svg{{display:block;font-family:{st["font"]}}}'
            f'.note{{position:absolute;background:#f7e27a;box-shadow:1px 2px 4px #0005;border-radius:2px;transform:rotate(-2deg)}}'
            f'</style></head><body><div style="position:relative;width:{W}px;height:{H}px;{frame}box-sizing:border-box">'
            f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">' + "".join(out) +
            "</svg>" + "".join(overlays) + "</div></body></html>")


def svg_legend(sp, L, cols, kinds=None):
    out = []
    fs, st = L["fs"], sp["style"]
    items = L["legend_items"]
    if not items:
        return out
    kinds = kinds or ["box"] * len(items)

    def sw(x, y, j):
        if kinds[j] == "line":
            return f'<line x1="{x}" y1="{y - fs * 0.35:.1f}" x2="{x + 18}" y2="{y - fs * 0.35:.1f}" stroke="{cols[j]}" stroke-width="3"/>'
        return f'<rect x="{x}" y="{y - fs * 0.8:.1f}" width="13" height="13" fill="{cols[j]}" rx="2"/>'
    if L["legend_pos"] == "right":
        x = L["W"] - L["pad"] - (max(tw(i, fs, st["wf"]) for i in items) + 30)
        for j, it in enumerate(items):
            y = L["y0"] + 10 + j * (fs + 12)
            out.append(sw(x, y, j) + _t(x + 22 if kinds[j] == "box" else x + 26, y, it, fs, st["fg"], el=f"lg_{j}"))
    else:
        base_y = L["pad"] + L["ft"] + 8 + fs + 4 if L["legend_pos"] == "top" else L["H"] - L["pad"] - (len(L["legend_rows"]) - 1) * (fs + 10) - 2
        j = 0
        for r_i, row in enumerate(L["legend_rows"]):
            y = base_y + r_i * (fs + 10)
            for it, x in row:
                xx = L["pad"] + x
                out.append(sw(xx, y, j) + _t(xx + (22 if kinds[j] == "box" else 26), y, it, fs, st["fg"], el=f"lg_{j}"))
                j += 1
    return out


def svg_cartesian(sp, L, cols, overlays):
    st, show, kind = sp["style"], sp["show"], sp["kind"]
    fs, wf, fg, muted, grid_c = L["fs"], st["wf"], st["fg"], st["muted"], st["grid_c"]
    x0, x1, y0, y1 = L["x0"], L["x1"], L["y0"], L["y1"]
    cats, ser = sp["categories"], sp["series"]
    n = len(cats)
    vmax, dec = sp["y_max"], sp["dec"]
    tdec = 1 if sp["y_step"] < 1 else 0
    out = []
    horiz = kind == "hbar"
    if horiz:
        def vx(v):
            return x0 + (x1 - x0) * v / vmax
        for k, t in enumerate(L["ticks"]):
            x = vx(t)
            if show["grid"] and k:
                out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{grid_c}" stroke-width="1"/>')
            if show["y_labels"]:
                out.append(_t(x, y1 + fs + 6, fmt(t, tdec), fs, muted, "middle", el=f"yt_{k}"))
        out.append(_t((x0 + x1) / 2, L["H"] - L["pad"] - (len(L["legend_rows"]) * (fs + 10) + 6 if L["legend_pos"] == "bottom" and L["legend_rows"] else 0),
                      sp["y_title"], fs, muted, "middle", el="ytitle"))
        out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{muted}" stroke-width="1.2"/>')
        band, bh = L["band"], L["bw"]
        for i, c in enumerate(cats):
            yc = y0 + band * (i + 0.5)
            out.append(_t(x0 - 8, yc + fs * 0.35, c, fs, fg, "end", el=f"x_{i}"))
            for j, s in enumerate(ser):
                v = s["values"][i]
                yy = yc - bh * len(ser) / 2 + j * bh
                out.append(f'<rect data-el="mk_{j}_{i}" x="{x0}" y="{yy + 1:.1f}" width="{vx(v) - x0:.1f}" height="{bh - 2:.1f}" fill="{cols[j]}" rx="{st["bar_round"]}"/>')
                if show["data_labels"]:
                    out.append(_t(vx(v) + 5, yy + bh / 2 + (fs - 1) * 0.35, fmt(v, dec), fs - 1, fg, el=f"dl_{j}_{i}"))
        out += svg_legend(sp, L, cols)
        if sp["cover"] is not None:
            i = cats.index(sp["cover"])
            overlays.append(f'<div class="note" style="transform:none;left:{x0 - 3:.0f}px;top:{y0 + band * i - 1:.0f}px;width:{x1 - x0 + 8:.0f}px;height:{band + 2:.0f}px"></div>')
        return out

    def vy(v, top_=vmax):
        return y1 - (y1 - y0) * v / top_
    for k, t in enumerate(L["ticks"]):
        y = vy(t)
        if show["grid"] and k:
            out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{grid_c}" stroke-width="1"/>')
        if show["y_labels"]:
            out.append(_t(x0 - 7, y + fs * 0.35, fmt(t, tdec), fs, muted, "end", el=f"yt_{k}"))
    if sp.get("y_title"):
        cx = L["pad"] + fs * 0.8
        cy = (y0 + y1) / 2
        out.append(_t(cx, cy, sp["y_title"], fs, muted, "middle", el="ytitle", extra=f'transform="rotate(-90 {cx:.1f} {cy:.1f})"'))
    if kind == "dual":
        for k, t in enumerate(L["ticks2"]):
            y = vy(t, sp["y2_max"])
            out.append(_t(x1 + 7, y + fs * 0.35, fmt(t), fs, muted, "start", el=f"yt2_{k}"))
        cx = L["W"] - L["pad"] - fs * 0.4
        cy = (y0 + y1) / 2
        out.append(_t(cx, cy, sp["y2_title"], fs, muted, "middle", el="y2title", extra=f'transform="rotate(90 {cx:.1f} {cy:.1f})"'))
        out.append(f'<line x1="{x1}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{muted}" stroke-width="1.2"/>')
    if sp.get("x_title"):
        out.append(_t((x0 + x1) / 2, y1 + L["xl_h"] + fs + 2, sp["x_title"], fs, muted, "middle", el="xtitle"))
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{muted}" stroke-width="1.2"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{muted}" stroke-width="1.2"/>')
    band = L["band"]
    if kind == "scatter":
        xmax = sp["x_max"]
        xs, ys = ser[0]["values"], ser[1]["values"]

        def sx(v):
            return x0 + (x1 - x0) * v / xmax
        for k in range(int(round(xmax / sp["x_step"])) + 1):
            t = k * sp["x_step"]
            if show["grid"] and k:
                out.append(f'<line x1="{sx(t):.1f}" y1="{y0}" x2="{sx(t):.1f}" y2="{y1}" stroke="{grid_c}" stroke-width="1"/>')
            out.append(_t(sx(t), y1 + fs + 6, fmt(t), fs, muted, "middle", el=f"xt_{k}"))
        for i, c in enumerate(cats):
            px, py = sx(xs[i]), vy(ys[i])
            out.append(f'<circle data-el="mk_0_{i}" cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{cols[0]}" stroke="{st["bg"]}" stroke-width="1"/>')
            lx = px + 8 if px + 8 + tw(c, fs - 1, wf) < x1 else px - 8
            out.append(_t(lx, py + (fs - 1) * 0.35, c, fs - 1, fg, "start" if lx > px else "end", el=f"x_{i}"))
        return out
    # category labels
    for i, c in enumerate(cats):
        xc = x0 + band * (i + 0.5)
        if L["rotate"]:
            yy = y1 + fs + 4
            out.append(_t(xc, yy, c, fs, fg, "end", el=f"x_{i}", extra=f'transform="rotate(-40 {xc:.1f} {yy:.1f})"'))
        else:
            out.append(_t(xc, y1 + fs + 6, c, fs, fg, "middle", el=f"x_{i}"))
    if kind in ("vbar", "grouped"):
        bw = L["bw"]
        for i in range(n):
            xc = x0 + band * (i + 0.5)
            for j, s in enumerate(ser):
                v = s["values"][i]
                xx = xc - bw * len(ser) / 2 + j * bw
                out.append(f'<rect data-el="mk_{j}_{i}" x="{xx + 1:.1f}" y="{vy(v):.1f}" width="{bw - 2:.1f}" height="{y1 - vy(v):.1f}" fill="{cols[j]}" rx="{st["bar_round"]}"/>')
                if show["data_labels"]:
                    out.append(_t(xx + bw / 2, vy(v) - 5, fmt(v, dec), fs - 1, fg, "middle", el=f"dl_{j}_{i}"))
        out += svg_legend(sp, L, cols)
    elif kind == "stacked":
        bw = L["bw"]
        for i in range(n):
            xc = x0 + band * (i + 0.5)
            acc = 0
            for j, s in enumerate(ser):
                v = s["values"][i]
                out.append(f'<rect data-el="mk_{j}_{i}" x="{xc - bw / 2:.1f}" y="{vy(acc + v):.1f}" width="{bw:.1f}" height="{vy(acc) - vy(acc + v):.1f}" fill="{cols[j]}" stroke="{st["bg"]}" stroke-width="1"/>')
                acc += v
            if show["data_labels"]:
                out.append(_t(xc, vy(acc) - 5, fmt(acc, dec), fs - 1, fg, "middle", "bold", el=f"dl_t_{i}"))
        out += svg_legend(sp, L, cols)
    elif kind in ("line", "dual"):
        if kind == "dual":
            bw = L["bw"]
            s = ser[0]
            for i in range(n):
                xc = x0 + band * (i + 0.5)
                v = s["values"][i]
                out.append(f'<rect data-el="mk_0_{i}" x="{xc - bw / 2:.1f}" y="{vy(v):.1f}" width="{bw:.1f}" height="{y1 - vy(v):.1f}" fill="{cols[0]}" opacity="0.85"/>')
            lines = [(1, ser[1], sp["y2_max"])]
        else:
            lines = [(j, s, vmax) for j, s in enumerate(ser)]
        for j, s, topv in lines:
            pts = [(x0 + band * (i + 0.5), vy(v, topv)) for i, v in enumerate(s["values"])]
            d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
            dash = ' stroke-dasharray="7 4"' if kind == "line" and j == 2 and st["palette"] % 2 else ""
            out.append(f'<path d="{d}" fill="none" stroke="{cols[j]}" stroke-width="{st["line_w"]}"{dash}/>')
            for i, (x, y) in enumerate(pts):
                if st["markers"] or kind == "dual":
                    out.append(f'<circle data-el="mk_{j}_{i}" cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{cols[j]}"/>')
                if show["data_labels"] and kind == "line":
                    other = [t["values"][i] for t in ser if t is not s]
                    above = not other or s["values"][i] >= max(other)
                    out.append(_t(x, y - 8 if above else y + fs + 4, fmt(s["values"][i], dec), fs - 1, fg, "middle", el=f"dl_{j}_{i}"))
            if show["direct_labels"]:
                x, y = pts[-1]
                out.append(_t(x + 9, y + fs * 0.35, s["name"], fs, cols[j], "start", "bold", el=f"lg_{j}"))
        out += svg_legend(sp, L, cols, ["box", "line"] if kind == "dual" else ["line"] * len(ser))
    if sp["cover"] is not None and kind != "dual":
        i = cats.index(sp["cover"])
        overlays.append(f'<div class="note" style="transform:rotate(-1deg);left:{x0 + band * i - 1:.0f}px;top:{y0 - 8:.0f}px;width:{band + 2:.0f}px;height:{y1 - y0 + 6:.0f}px"></div>')
    return out


def svg_pie(sp, L, cols):
    st, fs = sp["style"], L["fs"]
    cx, cy, r = L["cx"], L["cy"], L["r"]
    vals = sp["series"][0]["values"]
    out = []
    a = math.radians(sp["start_angle"] - 90)
    for j, v in enumerate(vals):
        b = a + 2 * math.pi * v / 100
        large = 1 if b - a > math.pi else 0
        p1 = (cx + r * math.cos(a), cy + r * math.sin(a))
        p2 = (cx + r * math.cos(b), cy + r * math.sin(b))
        out.append(f'<path data-el="mk_0_{j}" d="M{cx:.1f},{cy:.1f} L{p1[0]:.1f},{p1[1]:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {p2[0]:.1f},{p2[1]:.1f} Z" '
                   f'fill="{cols[j]}" stroke="{st["bg"]}" stroke-width="2"/>')
        if sp["show"]["data_labels"] and L["mode"] == "inside":
            m = (a + b) / 2
            rr = r * (0.8 if sp["kind"] == "donut" else 0.64)
            out.append(_t(cx + rr * math.cos(m), cy + rr * math.sin(m) + fs * 0.35, f"{fmt(v)}%", fs, "#ffffff", "middle", "bold",
                          el=f"dl_0_{j}", extra='paint-order="stroke" stroke="#0007" stroke-width="2"'))
        a = b
    if sp["kind"] == "donut":
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.55:.1f}" fill="{st["bg"]}"/>')
    lx = L["legend_x"]
    n = len(vals)
    y_start = cy - n * (fs + 12) / 2 + fs
    for j, it in enumerate(L["legend"]):
        y = y_start + j * (fs + 12)
        out.append(f'<rect x="{lx}" y="{y - fs * 0.8:.1f}" width="13" height="13" fill="{cols[j]}" rx="2"/>')
        out.append(_t(lx + 21, y, it, fs, st["fg"], el=f"x_{j}"))
    return out


def render_table(sp) -> str:
    st, ts = sp["style"], sp["table_style"]
    W, H = sp["size"]
    fs = st["fs"] + 1
    cols = [s["name"] for s in sp["series"]]
    rows = sp["categories"]
    cover = sp["cover"]
    border = {"rows": f"border-bottom:1px solid {st['grid_c']}", "grid": f"border:1px solid {st['grid_c']}", "none": ""}[ts["lines"]]
    head_bg = f"background:{colours(sp)[0]};color:#fff" if ts["head_bg"] else f"color:{st['muted']}"
    h = [f'<tr><th style="text-align:left;{border};{head_bg}" data-el="th_row">{E(sp["cat_noun"].capitalize())}</th>'
         + "".join(f'<th style="text-align:right;{border};{head_bg}" data-el="col_{k}">{E(c)}</th>' for k, c in enumerate(cols)) + "</tr>"]
    for i, rname in enumerate(rows):
        zebra = f"background:{st['grid_c']}55;" if ts["zebra"] and i % 2 else ""
        cells = []
        for k, c in enumerate(cols):
            v = fmt(sp["series"][k]["values"][i], sp["dec"])
            inner = f'<span data-el="cell_{k}_{i}">{E(v)}</span>'
            if cover and cover["row"] == rname and cover["col"] == c:
                inner = f'<span style="position:relative">{inner}<i style="position:absolute;left:-10px;right:-10px;top:-6px;bottom:-6px;background:#f7e27a;box-shadow:1px 2px 4px #0005;transform:rotate(-3deg)"></i></span>'
            cells.append(f'<td style="text-align:right;{border};{zebra}font-variant-numeric:tabular-nums">{inner}</td>')
        h.append(f'<tr><td style="{border};{zebra}" data-el="x_{i}">{E(rname)}</td>' + "".join(cells) + "</tr>")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:{st["bg"]};color:{st["fg"]};'
            f'font-family:{st["font"]};font-size:{fs}px}} td,th{{padding:{fs * 0.45:.0f}px {fs * 0.8:.0f}px}} table{{border-collapse:collapse}}'
            f'</style></head><body><div style="padding:22px 26px;width:{W}px;box-sizing:border-box">'
            f'<div style="font-weight:bold;font-size:{fs + 4}px;margin-bottom:4px" data-el="title">{E(sp["title"])}</div>'
            f'<div style="color:{st["muted"]};margin-bottom:12px" data-el="ytitle">{E(sp["y_title"])}</div>'
            f'<table>{"".join(h)}</table></div></body></html>')


# ------------------------------------------------------------------------------------------------ truth + worlds
def get(sp, ref):
    """Value of a ref {"s": series name | None (= stack total), "c": category}; Missing if the chart lacks it."""
    cats = sp["categories"]
    c = ref["c"]
    if c not in cats:
        raise Missing("insufficient_evidence")
    i = cats.index(c)
    if ref.get("s") is None:
        return sum(s["values"][i] for s in sp["series"])
    for s in sp["series"]:
        if s["name"] == ref["s"]:
            return s["values"][i]
    raise Missing("false_premise")


def shape_of(vals):
    d = [b - a for a, b in zip(vals, vals[1:])]
    if any(x == 0 for x in d):
        return None
    sg = [x > 0 for x in d]
    if all(sg):
        return "rise"
    if not any(sg):
        return "fall"
    ch = sum(1 for a, b in zip(sg, sg[1:]) if a != b)
    if ch == 1:
        return "peak" if sg[0] else "valley"
    return None


def solve(sp, task, p, opts=None, levels=None):
    """The true answer on a fully known spec (option text / bool / level int)."""
    if task == "extremum":
        vals = [(get(sp, r), lab) for lab, r in p["cands"]]
        best = (max if p["mode"] == "max" else min)(vals, key=lambda t: t[0])
        if sum(1 for v, _ in vals if v == best[0]) > 1:
            return "__tie__"
        return best[1]
    if task == "threshold":
        v = get(sp, p["ref"])
        return v > p["T"] if p["op"] == ">" else v < p["T"]
    if task in ("readoff", "diff", "share"):
        v = get(sp, p["ref"]) if task != "diff" else get(sp, p["a"]) - get(sp, p["b"])
        hits = [o for o, x in zip(opts, p["opt_values"]) if abs(x - v) < 1e-6]
        return hits[0] if hits else "__none__"
    if task == "ratio":
        b = get(sp, p["b"])
        v = get(sp, p["a"]) / b if b else float("inf")
        hits = [o for o, x in zip(opts, p["opt_values"]) if abs(x - v) < 1e-6]
        return hits[0] if hits else "__none__"
    if task == "rank":
        vals = [(get(sp, r), lab) for lab, r in p["cands"]]
        if len({v for v, _ in vals}) < len(vals):
            return "__tie__"
        return ", ".join(lab for _, lab in sorted(vals, key=lambda t: -t[0]))
    if task == "trend":
        vals = [get(sp, {"s": p["s"], "c": c}) for c in sp["categories"]]
        return {"rise": p["labels"]["rise"], "fall": p["labels"]["fall"], "peak": p["labels"]["peak"],
                "valley": p["labels"]["valley"], None: "__other__"}[shape_of(vals)]
    if task == "crossover":
        a = [get(sp, {"s": p["a"], "c": c}) for c in sp["categories"]]
        b = [get(sp, {"s": p["b"], "c": c}) for c in sp["categories"]]
        above = [x > y for x, y in zip(a, b)]
        if above[0] or not any(above):
            return "__none__"
        k = above.index(True)
        return spoken_cat(sp, sp["categories"][k])
    if task == "claim":
        k = p["claim"]
        if k == "above_every":
            return all(get(sp, {"s": p["a"], "c": c}) > get(sp, {"s": p["b"], "c": c}) for c in sp["categories"])
        if k == "doubled":
            return get(sp, p["r2"]) >= 2 * get(sp, p["r1"])
        if k == "higher":
            return get(sp, p["r1"]) > get(sp, p["r2"])
        if k == "over_half":
            return get(sp, p["ref"]) > 50
        if k == "sum_above":
            return get(sp, p["r1"]) + get(sp, p["r2"]) > p["T"]
        if k == "every_above":
            return all(get(sp, {"s": p["s"], "c": c}) > p["T"] for c in sp["categories"])
        raise KeyError(k)
    if task == "band":
        v = get(sp, p["ref"])
        return sum(1 for e in p["edges"] if v >= e)
    if task == "count":
        return sum(1 for r in p["refs"] if (get(sp, r) > p["T"]))
    if task == "relation":
        r = pearson(sp["series"][0]["values"], sp["series"][1]["values"])
        if r > 0.7:
            return p["labels"]["positive"]
        if r < -0.7:
            return p["labels"]["negative"]
        if abs(r) < 0.2:
            return p["labels"]["none"]
        return "__other__"
    raise KeyError(task)


def worlds(sp):
    """Every data set consistent with what the rendered chart shows (the spec itself first)."""
    out = [sp]
    kind, show = sp["kind"], sp["show"]
    cov = sp.get("cover")
    if cov is not None:
        new = []
        if kind == "table":
            k = [s["name"] for s in sp["series"]].index(cov["col"])
            i = sp["categories"].index(cov["row"])
            vals = sorted({0.0, sp["series"][k]["values"][i], 10 * max(max(s["values"]) for s in sp["series"])})
            for w in out:
                for v in vals:
                    w2 = copy.deepcopy(w)
                    w2["series"][k]["values"][i] = v
                    new.append(w2)
        else:
            i = sp["categories"].index(cov)
            top = sp["y_max"]
            choices = [0.0, top / 2, top]
            for w in out:
                for combo in itertools.product(choices, repeat=len(sp["series"])):
                    w2 = copy.deepcopy(w)
                    for j, v in enumerate(combo):
                        w2["series"][j]["values"][i] = v
                    new.append(w2)
        out = new
    multi = len(sp["series"]) > 1 and kind in ("grouped", "stacked", "line", "hbar")
    if multi and not show["legend"] and not show["direct_labels"]:
        names = [s["name"] for s in sp["series"]]
        new = []
        for w in out:
            for perm in itertools.permutations(names):
                w2 = copy.deepcopy(w)
                for s, nm in zip(w2["series"], perm):
                    s["name"] = nm
                new.append(w2)
        out = new
    if kind not in ("pie", "donut", "table", "scatter", "dual") and not show["y_labels"] and not show["data_labels"]:
        new = []
        for w in out:
            for k in (1.0, 0.5, 2.0):
                w2 = copy.deepcopy(w)
                for s in w2["series"]:
                    s["values"] = [v * k for v in s["values"]]
                new.append(w2)
        out = new
    return out


def decide(sp, task, p, opts=None, levels=None):
    """(gold, unknown_reason): the answer when every world consistent with the image agrees, else unknown."""
    outs = set()
    reason = "insufficient_evidence"
    for w in worlds(sp):
        try:
            outs.add(solve(w, task, p, opts, levels))
        except Missing as m:
            return None, m.reason
    if len(outs) != 1:
        return None, reason
    g = outs.pop()
    if isinstance(g, str) and g.startswith("__"):
        return None, "insufficient_evidence" if g != "__tie__" else "insufficient_evidence"
    return g, None


# ------------------------------------------------------------------------------------------------ margins (readability)
def gap_needed(sp, ref=None, axis2=False):
    """Smallest value difference a reader can resolve reliably on this chart."""
    if sp["kind"] == "table" or sp["show"]["data_labels"] and sp["kind"] not in ("stacked",):
        return 10 ** -sp["dec"] if sp["dec"] else 1
    if sp["kind"] == "stacked" and sp["show"]["data_labels"] and ref is not None and ref.get("s") is None:
        return 10 ** -sp["dec"] if sp["dec"] else 1
    if sp["kind"] in ("pie", "donut"):
        return 4
    top = sp["y2_max"] if axis2 else sp["y_max"]
    return 0.06 * top


def exact_readable(sp, v, axis2=False):
    """A value can be read exactly: printed, or sitting on a labelled tick far enough from its neighbours."""
    if sp["kind"] == "table":
        return True
    if sp["show"]["data_labels"] and sp["kind"] not in ("stacked", "dual"):
        return True
    if not sp["show"]["y_labels"]:
        return False
    step = sp["y2_step"] if axis2 else sp["y_step"]
    L = layout(copy.deepcopy(sp))
    return abs(v / step - round(v / step)) < 1e-9 and L.get("px_per_step", 0) >= 26


def margin_ok(sp, task, p):
    ax2 = p.get("axis2", False)
    g = gap_needed(sp, p.get("ref") or (p.get("cands") or [[None, None]])[0][1], ax2)
    try:
        if task == "extremum":
            vals = sorted((get(sp, r) for _, r in p["cands"]), reverse=p["mode"] == "max")
            return abs(vals[0] - vals[1]) >= g
        if task == "threshold":
            return abs(get(sp, p["ref"]) - p["T"]) >= g
        if task == "rank":
            vals = sorted(get(sp, r) for _, r in p["cands"])
            return all(b - a >= g for a, b in zip(vals, vals[1:]))
        if task == "trend":
            vals = [get(sp, {"s": p["s"], "c": c}) for c in sp["categories"]]
            return all(abs(b - a) >= g for a, b in zip(vals, vals[1:]))
        if task == "crossover":
            return all(abs(get(sp, {"s": p["a"], "c": c}) - get(sp, {"s": p["b"], "c": c})) >= g for c in sp["categories"])
        if task == "claim":
            k = p["claim"]
            if k == "above_every":
                return all(abs(get(sp, {"s": p["a"], "c": c}) - get(sp, {"s": p["b"], "c": c})) >= g for c in sp["categories"])
            if k == "doubled":
                return abs(get(sp, p["r2"]) - 2 * get(sp, p["r1"])) >= 2 * g
            if k == "higher":
                return abs(get(sp, p["r1"]) - get(sp, p["r2"])) >= g
            if k == "over_half":
                return abs(get(sp, p["ref"]) - 50) >= (g if sp["show"]["data_labels"] else 4)
            if k == "sum_above":
                return abs(get(sp, p["r1"]) + get(sp, p["r2"]) - p["T"]) >= 2 * g
            if k == "every_above":
                return all(abs(get(sp, {"s": p["s"], "c": c}) - p["T"]) >= g for c in sp["categories"])
        if task == "band":
            v = get(sp, p["ref"])
            return all(abs(v - e) >= g for e in p["edges"])
        if task == "count":
            return all(abs(get(sp, r) - p["T"]) >= g for r in p["refs"])
        if task in ("readoff", "share"):
            return exact_readable(sp, get(sp, p["ref"]), ax2)
        if task in ("diff", "ratio"):
            return exact_readable(sp, get(sp, p["a"]), ax2) and exact_readable(sp, get(sp, p["b"]), ax2)
        if task == "relation":
            return True
    except Missing:
        return True
    return True


# ------------------------------------------------------------------------------------------------ questions
SNOUN = {"person": "sales rep", "brand": "venue", "line": "production line", "plan": "plan", "mode": "travel mode",
         "clinic": "clinic", "site": "site", "store": "store", "product": "product"}


def ctx(sp):
    sn = SNOUN.get(sp["series_noun"], sp["series_noun"])
    return {"m": sp["metric"], "cn": sp["cat_noun"], "cns": sp["cat_nouns"], "sn": sn, "u": sp["unit"].strip()}


FIXED_POOLS = {"region", "channel", "team", "plant", "department", "plan", "platform", "line", "budget", "answer", "mode"}
PLURAL = {"units sold", "website visits", "support tickets", "online orders", "operating costs", "subscribers",
          "app downloads", "trips", "sales", "visitors", "visits", "orders"}


def P(x: str) -> str:
    """Possessive: Islands' / North's."""
    return x + ("'" if x.endswith("s") else "'s")


def WAS(sp, ax2=False, now=False):
    m = sp.get("metric2") if ax2 else sp["metric"]
    pl = m in PLURAL
    return ("are" if pl else "is") if now else ("were" if pl else "was")


def whose(sp, s):
    """'North's revenue' / 'the revenue' (single series)."""
    o = sp.get("owner")
    if s is None or len(sp["series"]) == 1 and sp["kind"] not in ("table",):
        return f"{P(o)} {sp['metric']}" if o else f"the {sp['metric']}"
    if sp["series_noun"] == "mode":
        return f"{sp['metric']} by {s.lower()}" + (f" in {o}" if o else "")
    if o and sp["series_noun"] in FIXED_POOLS:
        return f"{P(o)} {s} {sp['metric']}"
    return f"{P(s)} {sp['metric']}"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def at(sp, c):
    if sp["kind"] == "table":
        return f"for {c}"
    c2 = spoken_cat(sp, c)
    if sp["cat_noun"] in ("month", "year"):
        return f"in {c2}"
    if sp["cat_noun"] in ("quarter", "week"):
        return f"in {c2}"
    if sp["cat_noun"] == "day":
        return f"on {c2}"
    return f"at {c2}" if sp["cat_noun"] in ("store", "town", "venue") else f"for {c2}"


def T_txt(sp, T, ax2=False):
    return fmt(T, sp["dec"]) + (sp.get("unit2") if ax2 else sp["unit"])


def num_opts(rng, v, step, lo, hi, dec, unit, n=4):
    """Numeric options one step apart around v, all inside [lo, hi]."""
    cand = sorted({round(v + k * step, dec) for k in range(-3, 4) if lo <= v + k * step <= hi})
    if v not in cand or len(cand) < 3:
        return None
    for _ in range(20):
        pick = sorted(rng.sample(cand, min(n, len(cand))))
        if v in pick:
            break
    else:
        pick = sorted(set([v] + rng.sample([c for c in cand if c != v], min(n - 1, len(cand) - 1))))
    return [fmt_opt(x, dec, unit) for x in pick], pick


def q_extremum(rng, sp):
    kind, cats, ser = sp["kind"], sp["categories"], sp["series"]
    c = ctx(sp)
    mode = rng.choice(["max", "max", "min"])
    hi = "highest" if mode == "max" else "lowest"
    most = rng.choice(["most", "highest"]) if mode == "max" else rng.choice(["least", "lowest"])
    if kind in ("pie", "donut"):
        cands = [[x, {"s": "share", "c": x}] for x in cats]
        q = rng.choice([f"Which {c['cn']} has the {'largest' if mode == 'max' else 'smallest'} share of {sp['whole']}?",
                        f"Which slice of the chart is {'biggest' if mode == 'max' else 'smallest'}?",
                        f"Which {c['cn']} accounts for the {'biggest' if mode == 'max' else 'smallest'} part of {sp['whole']}?"])
        return q, cands, {"mode": mode}
    if kind == "scatter":
        axis = rng.choice(["y", "x"])
        m = sp["metric"] if axis == "y" else sp["x_metric"]
        cands = [[x, {"s": axis, "c": x}] for x in cats]
        q = rng.choice([f"Which {c['cn']} has the {hi} {m}?", f"Among {P(sp['owner'])} {c['cns']}, which one shows the {hi} {m}?"])
        return q, cands, {"mode": mode}
    if kind == "table":
        col = rng.choice([s["name"] for s in ser])
        cands = [[x, {"s": col, "c": x}] for x in cats]
        q = rng.choice([f"Which {c['cn']} had the {hi} {c['m']} in {spoken_cat(sp, col)}?",
                        f"In the {spoken_cat(sp, col)} column, which row shows the {hi} value?",
                        f"Looking at {spoken_cat(sp, col)}, which {c['cn']} recorded the {most} {c['m']}?"])
        return q, cands, {"mode": mode, "col": col}
    if kind == "dual":
        j = rng.randrange(2)
        s = ser[j]["name"]
        cands = [[spoken_cat(sp, x), {"s": s, "c": x}] for x in cats]
        q = rng.choice([f"In which {c['cn']} was the {s} {hi}?", f"Which {c['cn']} shows the {hi} {s}?"])
        return q, cands, {"mode": mode, "axis2": j == 1}
    if len(ser) > 1 and rng.random() < 0.5:
        x = rng.choice(cats)
        if kind == "stacked":
            cands = [[s["name"], {"s": s["name"], "c": x}] for s in ser]
            q = rng.choice([f"Which {c['sn']} makes up the {'largest' if mode == 'max' else 'smallest'} part of the {spoken_cat(sp, x)} bar?",
                            f"{cap(at(sp, x))}, which {c['sn']} contributed the {most} {c['m']}?"])
        else:
            cands = [[s["name"], {"s": s["name"], "c": x}] for s in ser]
            q = rng.choice([f"Which {c['sn']} had the {hi} {c['m']} {at(sp, x)}?", f"{cap(at(sp, x))}, which {c['sn']} recorded the {most} {c['m']}?",
                            f"Which {c['sn']} came {'top' if mode == 'max' else 'bottom'} for {c['m']} {at(sp, x)}?"])
        return q, cands, {"mode": mode}
    if kind == "stacked" and rng.random() < 0.6:
        cands = [[spoken_cat(sp, x), {"s": None, "c": x}] for x in cats]
        q = rng.choice([f"Which {c['cn']} had the {hi} total {c['m']}?", f"Adding up all {c['sn']}s, which {c['cn']} is {hi}?"])
        return q, cands, {"mode": mode}
    s = rng.choice(ser)["name"]
    cands = [[spoken_cat(sp, x), {"s": s, "c": x}] for x in cats]
    if len(ser) == 1:
        q = rng.choice([f"Which {c['cn']} had the {hi} {c['m']}?", f"In which {c['cn']} {WAS(sp)} {c['m']} at {'their' if WAS(sp) == 'were' else 'its'} {hi}?",
                        f"According to the chart, which {c['cn']} shows the {most} {c['m']}?"])
    else:
        q = rng.choice([f"In which {c['cn']} {WAS(sp)} {whose(sp, s)} {hi}?", f"When did {s} record its {hi} {c['m']}?",
                        f"Which {c['cn']} shows the {hi} value for {s}?"])
    return q, cands, {"mode": mode, "s": s}


def plan_questions(rng, sp):
    """Candidate (task, type, question, options|levels, params) for one chart; margins are checked by the caller."""
    out = []
    kind, cats, ser = sp["kind"], sp["categories"], sp["series"]
    c = ctx(sp)
    dec = sp["dec"]
    # 1. extremum
    q, cands, p = q_extremum(rng, sp)
    out.append(("extremum", "choice", q, [x[0] for x in cands], dict(p, cands=cands)))
    # refs available for value questions
    refs = []
    if kind in ("pie", "donut"):
        refs = [({"s": "share", "c": x}, False) for x in cats]
    elif kind == "scatter":
        refs = [({"s": "y", "c": x}, False) for x in cats]
    elif kind == "dual":
        refs = [({"s": s["name"], "c": x}, j == 1) for j, s in enumerate(ser) for x in cats]
    elif kind == "stacked":
        refs = [({"s": None, "c": x}, False) for x in cats] + [({"s": s["name"], "c": x}, False) for s in ser for x in cats]
    else:
        refs = [({"s": s["name"], "c": x}, False) for s in ser for x in cats]
    rng.shuffle(refs)

    def rdesc(ref, ax2=False):
        if kind in ("pie", "donut"):
            return f"{P(ref['c'])} share"
        if kind == "scatter":
            return f"{P(ref['c'])} {sp['metric']}"
        if kind == "dual":
            return f"{P(sp['owner'])} {ref['s']} {at(sp, ref['c'])}"
        if kind == "table":
            who = f"{P(sp['owner'])} {ref['c']}" if sp["series_noun"] in FIXED_POOLS or sp.get("row_pool") in FIXED_POOLS else P(ref["c"])
            return f"{who} {sp['metric']} in {spoken_cat(sp, ref['s'])}"
        if ref["s"] is None:
            return f"{P(sp['owner'])} total {sp['metric']} {at(sp, ref['c'])}"
        return f"{whose(sp, ref['s'])} {at(sp, ref['c'])}"

    def short_b(a_ref, b_ref, ax2):
        if kind in ("pie", "donut", "scatter", "table", "dual"):
            return rdesc(b_ref, ax2)
        if a_ref["s"] == b_ref["s"]:
            return at(sp, b_ref["c"])
        if b_ref["s"] is not None and a_ref["c"] == b_ref["c"]:
            return P(b_ref["s"])
        return rdesc(b_ref, ax2)

    def unit_of(ax2):
        return sp.get("unit2") if ax2 else sp["unit"]

    def top_of(ax2):
        return sp["y2_max"] if ax2 else sp["y_max"]

    def step_of(ax2):
        return sp["y2_step"] if ax2 else sp["y_step"]
    # 2. threshold (T in the question or in the state)
    if refs:
        ref, ax2 = refs[0]
        v = get(sp, ref)
        top, st_ = top_of(ax2), step_of(ax2)
        T = None
        for _ in range(12):
            if kind in ("pie", "donut"):
                t = rng.choice([10, 15, 20, 25, 30, 40, 50])
            elif kind == "table" or sp["show"]["data_labels"] and kind != "stacked":
                t = round(v + rng.choice([-1, 1]) * rng.uniform(0.02, 0.3) * max(v, 1), dec)
                t = round(t) if dec == 0 and t > 20 else t
            else:
                t = st_ * rng.randint(1, int(round(top / st_))) / rng.choice([1, 1, 2])
                t = round(t, dec)
            if t > 0 and t != v:
                T = t
                break
        if T is not None:
            op = rng.choice([">", ">", "<"])
            word = rng.choice(["above", "more than", "over", "higher than"]) if op == ">" else rng.choice(["below", "less than", "under"])
            if rng.random() < 0.25 and kind not in ("pie", "donut"):
                q = rng.choice([f"{WAS(sp, ax2).capitalize()} {rdesc(ref, ax2)} {word} the alert level in the notes?",
                                f"Did {rdesc(ref, ax2)} come in {word} the target given in the brief?"])
                state = {"alert_level" if "alert" in q else "target": T_txt(sp, T, ax2), "note_ref": f"N-{rng.randint(1000, 99999)}"}
                out.append(("threshold", "noul", q, None, {"ref": ref, "T": T, "op": op, "axis2": ax2, "state": state}))
            else:
                if kind in ("pie", "donut"):
                    q = rng.choice([f"Is {P(ref['c'])} share of {sp['whole']} {word} {fmt(T)}%?", f"Does {ref['c']} account for {word} {fmt(T)}% of {sp['whole']}?"])
                else:
                    q = rng.choice([f"{WAS(sp, ax2).capitalize()} {rdesc(ref, ax2)} {word} {T_txt(sp, T, ax2)}?", f"{WAS(sp, ax2, True).capitalize()} {rdesc(ref, ax2)} {word} {T_txt(sp, T, ax2)}?",
                                    f"Did {rdesc(ref, ax2)} exceed {T_txt(sp, T, ax2)}?" if op == ">" else f"Did {rdesc(ref, ax2)} stay under {T_txt(sp, T, ax2)}?"])
                out.append(("threshold", "noul", q, None, {"ref": ref, "T": T, "op": op, "axis2": ax2}))
    # 3. readoff
    for ref, ax2 in refs[1:4]:
        v = get(sp, ref)
        if kind in ("pie", "donut"):
            if not sp["show"]["data_labels"]:
                continue
            got = num_opts(rng, v, rng.choice([3, 5, 7]), 1, 99, 0, "%")
            task = "share"
            q = rng.choice([f"What share of {sp['whole']} does {ref['c']} have?", f"What percentage is shown for {ref['c']}?"])
        else:
            step = step_of(ax2) if not (sp["show"]["data_labels"] or kind == "table") else max(step_of(ax2), 10 ** -dec * rng.choice([5, 10, 20]))
            if kind == "table":
                step = round(max(10 ** -dec, abs(v) * rng.choice([0.05, 0.1, 0.2])), dec)
            got = num_opts(rng, v, step, step, top_of(ax2) if kind != "table" else v * 3, dec, unit_of(ax2))
            task = "readoff"
            q = rng.choice([f"What {WAS(sp, ax2)} {rdesc(ref, ax2)}?", f"What value does the chart show for {rdesc(ref, ax2)}?",
                            f"Reading the chart, how much {WAS(sp, ax2)} {rdesc(ref, ax2)}?"])
        if got:
            opts, vals = got
            out.append((task, "choice", q, opts, {"ref": ref, "axis2": ax2, "opt_values": vals}))
            break
    # 4. diff / ratio between two values of one series (or the same category)
    if kind not in ("pie", "donut", "scatter") and len(refs) >= 2:
        for a_ref, ax2 in refs[:6]:
            b_cands = [r for r, a2 in refs if a2 == ax2 and r != a_ref and (r["s"] == a_ref["s"] or r["c"] == a_ref["c"])]
            if not b_cands:
                continue
            b_ref = rng.choice(b_cands)
            a, b = get(sp, a_ref), get(sp, b_ref)
            if a < b:
                a_ref, b_ref, a, b = b_ref, a_ref, b, a
            if a == b or b <= 0:
                continue
            if abs(a / b - round(a / b * 2) / 2) < 1e-9 and a / b in (1.5, 2, 3, 4):
                labels = {1.5: "about 1.5 times", 2: "about twice", 3: "about 3 times", 4: "about 4 times"}
                opts = list(labels.values())
                q = rng.choice([f"How does {rdesc(a_ref, ax2)} compare with {rdesc(b_ref, ax2)}?",
                                f"Roughly how many times {rdesc(b_ref, ax2)} {WAS(sp, ax2)} {rdesc(a_ref, ax2)}?"])
                out.append(("ratio", "choice", q, opts, {"a": a_ref, "b": b_ref, "axis2": ax2, "opt_values": list(labels)}))
                break
            d = round(a - b, dec)
            step = step_of(ax2) if not (sp["show"]["data_labels"] or kind == "table") else round(max(10 ** -dec, d * rng.choice([0.25, 0.5])), dec)
            if kind == "table":
                step = round(max(10 ** -dec, d * rng.choice([0.2, 0.3, 0.5])), dec)
            got = num_opts(rng, d, step, 10 ** -dec, top_of(ax2) if kind != "table" else d * 4, dec, unit_of(ax2))
            if not got:
                continue
            q = rng.choice([f"By how much did {rdesc(a_ref, ax2)} exceed {rdesc(b_ref, ax2)}?",
                            f"What is the gap between {rdesc(a_ref, ax2)} and {rdesc(b_ref, ax2)}?",
                            f"How much higher {WAS(sp, ax2)} {rdesc(a_ref, ax2)} than {short_b(a_ref, b_ref, ax2)}?"])
            out.append(("diff", "choice", q, got[0], {"a": a_ref, "b": b_ref, "axis2": ax2, "opt_values": got[1]}))
            break
    # 5. rank three items
    if len(cats) >= 3 and kind not in ("scatter",):
        three = rng.sample(cats, 3)
        s = ser[0]["name"] if len(ser) == 1 or kind in ("pie", "donut") else rng.choice([x["name"] for x in ser])
        if kind == "stacked":
            s = None
        cands = [[spoken_cat(sp, x), {"s": s, "c": x}] for x in three]
        perms = [", ".join(pp) for pp in itertools.permutations([x[0] for x in cands])]
        opts = rng.sample(perms, 4)
        lab = " and ".join([", ".join(x[0] for x in cands[:2]), cands[2][0]])
        what = f"{P(sp['owner'])} total {c['m']}" if s is None else whose(sp, s)
        if kind in ("pie", "donut"):
            what = f"share of {sp['whole']}"
        if kind == "table":
            what = f"{P(sp['owner'])} {c['m']} in {spoken_cat(sp, s)}"
            cands = [[x, {"s": s, "c": x}] for x in three]
            perms = [", ".join(pp) for pp in itertools.permutations(three)]
            opts = rng.sample(perms, 4)
            lab = " and ".join([", ".join(three[:2]), three[2]])
        true = ", ".join(x[0] for x in sorted(cands, key=lambda t: -get(sp, t[1])))
        opts = [true] + rng.sample([x for x in perms if x != true], 3)
        rng.shuffle(opts)
        q = rng.choice([f"Rank {lab} by {what}, highest first.", f"Which ordering of {lab} runs from highest to lowest {what}?"])
        out.append(("rank", "choice", q, opts, {"cands": cands, "s": s}))
    # 6. trend / crossover (ordered axes)
    if kind in ("line", "vbar", "grouped") and sp["ordered"] and len(cats) >= 4:
        s = rng.choice(ser)["name"]
        labels = {"rise": f"rose every {c['cn']}", "fall": f"fell every {c['cn']}", "peak": "rose, then fell", "valley": "fell, then rose"}
        q = rng.choice([f"How did {whose(sp, s)} change across the {c['cns']} shown?", f"Which description fits {whose(sp, s)} over the period?",
                        f"What pattern does {whose(sp, s)} follow from {spoken_cat(sp, cats[0])} to {spoken_cat(sp, cats[-1])}?"])
        out.append(("trend", "choice", q, list(labels.values()), {"s": s, "labels": labels}))
        if len(ser) >= 2:
            a, b = rng.sample([x["name"] for x in ser], 2)
            q = rng.choice([f"In which {c['cn']} did {a} first move above {b}?", f"When did {P(a)} {c['m']} first overtake {P(b)}?"])
            out.append(("crossover", "choice", q, [spoken_cat(sp, x) for x in cats[1:]], {"a": a, "b": b}))
    # 7. claim
    claim = None
    if kind in ("line", "grouped") and len(ser) >= 2:
        a, b = rng.sample([x["name"] for x in ser], 2)
        claim = ("above_every", {"a": a, "b": b}, [f"For {sp['owner']}, does the chart support the claim that {a} beat {b} on {c['m']} in every {c['cn']}?",
                                                   f"At {sp['owner']}, was {a} ahead of {b} in each {c['cn']} shown?"])
    elif kind in ("pie", "donut"):
        x = rng.choice(cats)
        claim = ("over_half", {"ref": {"s": "share", "c": x}}, [f"Does the chart show {x} making up more than half of {sp['whole']}?",
                                                                 f"Is {x} over 50% of {sp['whole']} according to the chart?"])
    elif kind in ("vbar", "hbar", "line", "table", "stacked") and len(cats) >= 2:
        s = rng.choice(ser)["name"] if kind != "stacked" else None
        c1, c2 = rng.sample(cats, 2)
        if kind == "table":
            s = rng.choice(ser)["name"]
            r1, r2 = {"s": s, "c": c1}, {"s": s, "c": c2}
            claim = ("doubled", {"r1": r1, "r2": r2}, [f"Does {P(sp['owner'])} table show {c2} with at least double the {c['m']} of {c1} in {spoken_cat(sp, s)}?",
                                                     f"In {spoken_cat(sp, s)}, was {P(c2)} figure at least twice {P(c1)}?"])
        elif sp["ordered"]:
            i1, i2 = sorted([cats.index(c1), cats.index(c2)])
            c1, c2 = cats[i1], cats[i2]
            r1, r2 = {"s": s, "c": c1}, {"s": s, "c": c2}
            if rng.random() < 0.5:
                claim = ("doubled", {"r1": r1, "r2": r2}, [f"Is it true that {whose(sp, s)} at least doubled from {spoken_cat(sp, c1)} to {spoken_cat(sp, c2)}?",
                                                         f"Does the chart back the claim that {whose(sp, s)} doubled or more between {spoken_cat(sp, c1)} and {spoken_cat(sp, c2)}?"])
            else:
                claim = ("higher", {"r1": r2, "r2": r1}, [f"{WAS(sp).capitalize()} {whose(sp, s)} higher {at(sp, c2)} than {at(sp, c1)}?",
                                                         f"Does the chart show {whose(sp, s)} {at(sp, c2)} above its level {at(sp, c1)}?"])
        else:
            r1, r2 = {"s": s, "c": c1}, {"s": s, "c": c2}
            claim = ("higher", {"r1": r1, "r2": r2}, [f"Does the chart support the claim that {c1} outperformed {c2} on {c['m']}?",
                                                     f"{WAS(sp, now=True).capitalize()} {P(c1)} {c['m']} higher than {P(c2)} in this chart?"])
    if claim:
        k, p, qs = claim
        out.append(("claim", "noul", rng.choice(qs), None, dict(p, claim=k)))
    # 8. severity band (score)
    for ref, ax2 in refs[4:8]:
        if kind in ("pie", "donut"):
            break
        v = get(sp, ref)
        top, st_ = top_of(ax2), step_of(ax2)
        if kind == "table":
            top = max(max(s["values"]) for s in ser) * 1.1
            st_ = 10 ** math.floor(math.log10(max(top / 5, 1)))
        nb = rng.randint(2, 4)
        grid = [round(st_ * k, dec) for k in range(1, int(top / st_))] if top / st_ >= nb + 1 else []
        if len(grid) < nb:
            continue
        edges = sorted(rng.sample(grid, nb))
        u = unit_of(ax2).strip()
        names = rng.choice([["normal", "elevated", "high", "critical", "extreme"], ["low", "moderate", "high", "very high", "severe"],
                            ["green", "amber", "red", "black", "purple"]])
        levels = [f"{names[0]}: under {fmt(edges[0], dec)} {u}".strip()]
        for e1, e2 in zip(edges, edges[1:]):
            levels.append(f"{names[len(levels)]}: {fmt(e1, dec)} to under {fmt(e2, dec)} {u}".strip())
        levels.append(f"{names[len(levels)]}: {fmt(edges[-1], dec)} {u} or more".replace("  ", " "))
        q = rng.choice([f"Which band does {rdesc(ref, ax2)} fall into?", f"Using the bands given, how severe {WAS(sp, ax2)} {rdesc(ref, ax2)}?",
                        f"Rate {rdesc(ref, ax2)} on the scale below."])
        out.append(("band", "score", q, levels, {"ref": ref, "edges": edges, "axis2": ax2}))
        break
    # 9. count above T (score)
    if kind in ("vbar", "line", "grouped", "hbar", "dual", "table", "scatter") and 3 <= len(cats) <= 9:
        if kind == "table":
            s = rng.choice(ser)["name"]
        elif kind == "scatter":
            s = "y"
        elif kind == "dual":
            s = rng.choice(ser)["name"]
        else:
            s = rng.choice(ser)["name"]
        ax2 = kind == "dual" and s == ser[1]["name"]
        rs = [{"s": s, "c": x} for x in cats]
        vals = [get(sp, r) for r in rs]
        top, st_ = top_of(ax2), step_of(ax2)
        if kind == "table":
            T = round(sorted(vals)[len(vals) // 2] * rng.uniform(0.85, 1.15), dec)
        else:
            T = round(st_ * rng.randint(1, int(round(top / st_)) - 1) / rng.choice([1, 2]), dec)
        cn = sp["cat_noun"]
        levels = [f"none of them" if k == 0 else f"{k} {cn if k == 1 else sp['cat_nouns']}" for k in range(len(cats) + 1)]
        if kind == "scatter":
            what = f"{sp['metric']} above {T_txt(sp, T)}"
            q = rng.choice([f"How many {sp['cat_nouns']} have {what}?", f"Count the {sp['cat_nouns']} with {what}."])
        elif kind == "table":
            q = rng.choice([f"How many {sp['cat_nouns']} had more than {T_txt(sp, T)} in {spoken_cat(sp, s)}?",
                            f"In {spoken_cat(sp, s)}, how many rows exceed {T_txt(sp, T)}?"])
        else:
            who = whose(sp, s)
            if kind == "dual":
                who = f"{P(sp['owner'])} {s}"
            q = rng.choice([f"In how many {sp['cat_nouns']} {WAS(sp, ax2)} {who} above {T_txt(sp, T, ax2)}?",
                            f"How many {sp['cat_nouns']} show {who} over {T_txt(sp, T, ax2)}?"])
        out.append(("count", "score", q, levels, {"refs": rs, "T": T, "axis2": ax2}))
    # 10. scatter relation
    if kind == "scatter":
        labels = {"positive": f"higher {sp['x_metric']} goes with higher {sp['metric']}",
                  "negative": f"higher {sp['x_metric']} goes with lower {sp['metric']}", "none": "no clear relationship"}
        q = rng.choice([f"What does the plot of {P(sp['owner'])} {sp['cat_nouns']} show about {sp['x_metric']} and {sp['metric']}?",
                        f"For {P(sp['owner'])} {sp['cat_nouns']}, how are {sp['x_metric']} and {sp['metric']} related?"])
        out.append(("relation", "choice", q, list(labels.values()), {"labels": labels}))
    return out


FAMILY = {"extremum": "chart_extremum", "threshold": "chart_threshold", "readoff": "chart_readoff", "share": "chart_readoff",
          "diff": "chart_compare", "ratio": "chart_compare", "rank": "chart_compare", "trend": "chart_trend",
          "crossover": "chart_trend", "claim": "chart_claim", "relation": "chart_trend", "band": "chart_score", "count": "chart_score"}
BASE_DIFF = {"extremum": 2, "threshold": 3, "readoff": 3, "share": 2, "diff": 4, "ratio": 4, "rank": 3, "trend": 3,
             "crossover": 4, "claim": 4, "relation": 3, "band": 4, "count": 4}


def difficulty(sp, task, unknown):
    d = BASE_DIFF[task]
    if len(sp["series"]) >= 3 and sp["kind"] not in ("table",):
        d += 1
    if sp["style"]["fs"] <= 11 or sp["style"]["rotate"] or sp["kind"] == "dual":
        d += 1
    if unknown:
        d += 1
    return max(2, min(5, d))


def make_item(sp, task, qtype, q, opts, p, parent=None, variant=None):
    """One planned row (no image yet); None when the gold is ill-posed or unreadable."""
    levels = opts if qtype == "score" else None
    options = opts if qtype == "choice" else None
    gold, unk = decide(sp, task, p, options, levels)
    if gold is None and variant is None:
        return None          # base items are answerable unless decide() found the chart itself undecidable on purpose
    if gold is not None and not margin_ok(sp, task, p):
        return None
    if qtype == "choice":
        if len(set(o.lower() for o in options)) != len(options):
            return None
    field = {"type": qtype, "question": q}
    g = gold
    if qtype == "choice":
        taken = set()
        keyed = [{"key": option_key(o, taken), "text": o} for o in options]
        field["options"] = keyed
        if gold is not None:
            if gold not in options:
                return None
            g = {o["text"]: o["key"] for o in keyed}[gold]
    elif qtype == "score":
        field["levels"] = [{"value": i, "description": d} for i, d in enumerate(levels)]
    state = p.get("state") or {}
    return {"spec_id": sp["id"], "task": task, "field": field, "gold": g, "unknown_reason": unk, "params": p, "state": state,
            "parent": parent, "variant": variant, "difficulty": difficulty(sp, task, gold is None)}


# ------------------------------------------------------------------------------------------------ unknown variants
def refs_in(p):
    out = []
    for k in ("ref", "a", "b", "r1", "r2"):
        if isinstance(p.get(k), dict):
            out.append(p[k])
    for k in ("cands",):
        out += [x[1] for x in p.get(k, [])]
    out += p.get("refs", [])
    return out


def subst(obj, old, new):
    """Replace every string exactly equal to `old` (params only; never substrings)."""
    if isinstance(obj, str):
        return new if obj == old else obj
    if isinstance(obj, list):
        return [subst(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v, old, new) for k, v in obj.items()}
    return obj


def unknown_variant(rng, sp, it, qtext_fn=None):
    """(new spec or None, new params, question, how) with the decisive part missing, or None."""
    task, p = it["task"], it["params"]
    ways = ["cover", "absent", "outside", "no_legend", "no_scale"]
    rng.shuffle(ways)
    q = it["field"]["question"]
    for how in ways:
        if how == "absent" and sp["absent_series"] and len(sp["series"]) > 1 and sp["kind"] not in ("table", "scatter", "pie", "donut", "dual"):
            names = {r.get("s") for r in refs_in(p)} | {p.get(k) for k in ("s", "a", "b") if isinstance(p.get(k), str)}
            names = [n for n in names if n and n in [s["name"] for s in sp["series"]]]
            if not names:
                continue
            old = names[0]
            new = rng.choice(sp["absent_series"])
            if old not in q:
                continue
            p2 = subst(p, old, new)
            q2 = q.replace(old, new)
            if task in ("extremum", "rank") and any(x[0] == new for x in p2.get("cands", [])):
                continue          # the new name would also be an option label: keep it simple
            return None, p2, q2, "absent_series"
        if how == "outside" and sp["outside"] and sp["kind"] not in ("scatter", "pie", "donut"):
            cs = [r["c"] for r in refs_in(p)]
            if not cs or task in ("extremum", "rank", "count") and "cands" in p and len(p["cands"]) and p["cands"][0][1]["c"] != p["cands"][-1][1]["c"]:
                continue
            old = cs[0]
            old_sp = spoken_cat(sp, old)
            if old_sp not in q:
                continue
            new = rng.choice(sp["outside"])
            p2 = subst(p, old, new)
            q2 = q.replace(old_sp, new)
            return None, p2, q2, "outside_axis"
        if how == "cover" and sp["kind"] in ("vbar", "grouped", "line", "hbar", "table") and sp["cover"] is None:
            cs = [r["c"] for r in refs_in(p) if r["c"] in sp["categories"]]
            if task in ("trend", "crossover") or (task == "claim" and p.get("claim") == "above_every"):
                cs = list(sp["categories"])
            if not cs:
                continue
            sp2 = copy.deepcopy(sp)
            c = rng.choice(cs)
            if sp["kind"] == "table":
                col = next((r["s"] for r in refs_in(p) if r["c"] == c and r.get("s")), None) or p.get("col")
                if col is None:
                    continue
                sp2["cover"] = {"row": c, "col": col}
            else:
                sp2["cover"] = c
            return sp2, p, q, "covered"
        if how == "no_legend" and sp["kind"] in ("grouped", "stacked", "line", "hbar") and len(sp["series"]) > 1 and (sp["show"]["legend"] or sp["show"]["direct_labels"]):
            sp2 = copy.deepcopy(sp)
            sp2["show"]["legend"] = False
            sp2["show"]["direct_labels"] = False
            return sp2, p, q, "no_legend"
        if how == "no_scale" and sp["kind"] in ("vbar", "grouped", "line", "hbar", "stacked") and sp["show"]["y_labels"]:
            sp2 = copy.deepcopy(sp)
            sp2["show"]["y_labels"] = False
            sp2["show"]["data_labels"] = False
            sp2["show"]["grid"] = False
            return sp2, p, q, "no_scale"
    return None


# ------------------------------------------------------------------------------------------------ plan (pure) + render
def plan(seed: str, count: int, unknown_share: float = UNKNOWN_SHARE, prefix: str = "c"):
    """Specs + items (no rendering). Deterministic in (seed, count)."""
    specs, items = {}, []
    n_base = int(round(count * (1 - unknown_share)))
    i = 0
    task_ct, gold_ct = Counter(), Counter()
    while len(items) < n_base * 1.12 and i < count * 3:
        rng = rng_for("p3-charts", seed, "spec", i)
        sid = f"{prefix}{seed}-{i:05d}"
        i += 1
        try:
            sp = make_spec(rng, sid)
        except Skip:
            continue
        qrng = rng_for("p3-charts", seed, "q", sid)
        cands = plan_questions(qrng, sp)
        qrng.shuffle(cands)
        kept = 0
        for task, qtype, q, opts, p in cands:
            if kept >= 3:
                break
            # balance the task mix: extremum/threshold are always available, the rest are rarer
            if task_ct[task] > 1.6 * (sum(task_ct.values()) / max(1, len(task_ct))) + 30:
                continue
            it = make_item(sp, task, qtype, q, opts, p)
            if it is None:
                continue
            if qtype == "noul" and gold_ct[it["gold"]] > gold_ct[not it["gold"]] * 1.08 + 10:
                continue          # keep yes / no golds balanced
            gold_ct[it["gold"]] += qtype == "noul"
            kept += 1
            task_ct[task] += 1
            items.append(it)
        if kept:
            specs[sid] = sp
    # unknown variants
    urng = rng_for("p3-charts", seed, "unknowns")
    base = [x for x in items if x["variant"] is None]
    urng.shuffle(base)
    n_unk_target = int(round(n_base * 1.12 * unknown_share / (1 - unknown_share)))
    made = 0
    for k, it in enumerate(base):
        if made >= n_unk_target:
            break
        sp = specs[it["spec_id"]]
        vrng = rng_for("p3-charts", seed, "unk", it["spec_id"], k)
        v = unknown_variant(vrng, sp, it)
        if v is None:
            continue
        sp2, p2, q2, how = v
        if sp2 is not None:
            sp2["id"] = f"{sp['id']}-u{k}"
            try:
                sp2["show"] = layout(copy.deepcopy(sp2))["show"]
            except Skip:
                continue
        target = sp2 or sp
        opts = [o["text"] for o in it["field"].get("options", [])] or [l["description"] for l in it["field"].get("levels", [])] or None
        u = make_item(target, it["task"], it["field"]["type"], q2, opts, p2, parent=it, variant=how)
        if u is None or u["gold"] is not None:
            continue
        if sp2 is not None:
            specs[sp2["id"]] = sp2
        items.append(u)
        made += 1
    return specs, items


def render_pages(pages: list[dict], work: Path, conc: int = 6) -> dict:
    """pages: {"id", "html", "width", "height"} -> {id: render result} (PNG at work/<id>.png)."""
    jobs = []
    for s in pages:
        hp = work / f"{s['id']}.html"
        hp.write_text(s["html"])
        jobs.append({"id": s["id"], "html": str(hp), "out": str(work / f"{s['id']}.png"), "width": s["width"], "height": s["height"]})
    res = {}
    for k in range(0, len(jobs), 400):
        chunk = jobs[k:k + 400]
        jp, rp = work / f"jobs-{k}.jsonl", work / f"results-{k}.jsonl"
        jp.write_text("\n".join(json.dumps(j) for j in chunk) + "\n")
        subprocess.run(["node", str(RENDER), str(jp), str(rp), str(conc)], check=True, capture_output=True)
        res.update({r["id"]: r for r in (json.loads(l) for l in rp.read_text().splitlines() if l.strip())})
    return res


def finish_image(png: Path, post: dict, out_dir: Path, rel_base: str, bench: set) -> tuple[str, str] | None:
    """Apply post-processing (noise / blur / rotation / crop) with PIL, save as PNG or WebP named by sha256."""
    from PIL import Image, ImageFilter
    import numpy as np
    im = Image.open(png).convert("RGB")
    if post.get("crop_h"):
        im = im.crop((0, 0, im.width, min(im.height, int(post["crop_h"]))))
    if post.get("rotate"):
        im = im.rotate(post["rotate"], resample=Image.BICUBIC, expand=False, fillcolor=tuple(post.get("fill", (250, 250, 248))))
    if post.get("blur"):
        im = im.filter(ImageFilter.GaussianBlur(post["blur"]))
    if post.get("contrast"):
        from PIL import ImageEnhance
        im = ImageEnhance.Contrast(im).enhance(post["contrast"])
    if post.get("noise"):
        rs = np.random.RandomState(post["noise_seed"])
        a = np.asarray(im).astype(np.int16)
        a = a + rs.normal(0, post["noise"], a.shape[:2])[..., None].astype(np.int16)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if im.width > 1280:
        im = im.resize((1280, int(im.height * 1280 / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    ext = post.get("format", "png")
    if ext == "webp":
        im.save(buf, "WEBP", quality=88, method=4)
    else:
        im.save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    if sha in bench:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{sha}.{ext}"
    if not dst.exists():
        dst.write_bytes(data)
    return f"{rel_base}/{sha}.{ext}", sha


def rel_dir(img_dir: Path) -> str:
    """Image paths are relative to data/ (a scratch directory elsewhere is kept absolute, for dry runs only)."""
    try:
        return str(img_dir.resolve().relative_to(ROOT / "data"))
    except ValueError:
        return str(img_dir.resolve())


def post_for(sp) -> dict:
    if sp["style"].get("noise"):
        r = rng_for("p3-charts-post", sp["id"])
        return {"noise": r.choice([4, 6, 8]), "noise_seed": r.randrange(2 ** 31), "blur": r.choice([0, 0.4, 0.6]), "format": "webp"}
    return {"format": "png"}


def required_elements(sp, els):
    """Every tick label, legend entry, category label and the title must be fully in view and on top; data labels too."""
    bad = []
    for k, e in els.items():
        if k.startswith(("x_", "lg_", "yt_", "yt2_", "xt_", "title", "dl_", "col_", "th_", "ytitle", "y2title")):
            if not e["visible"]:
                bad.append(k)
    return bad


def covered_ok(sp, els):
    """Every mark (and data label) of the covered column is hidden under the note; the category label is not."""
    cov = sp.get("cover")
    if cov is None:
        return True
    if sp["kind"] == "table":
        k = [s["name"] for s in sp["series"]].index(cov["col"])
        i = sp["categories"].index(cov["row"])
        e = els.get(f"cell_{k}_{i}")
        return e is not None and not e["visible"]
    i = sp["categories"].index(cov)
    for j in range(len(sp["series"])):
        e = els.get(f"mk_{j}_{i}")
        if e is not None and e["visible"]:
            return False
    return True


def build(seed: str, count: int, work: Path, img_dir: Path = IMG_DIR, id_prefix: str = "p3-I-chart", unknown_share=UNKNOWN_SHARE):
    specs, items = plan(seed, count, unknown_share)
    pages = []
    for sid, sp in specs.items():
        pages.append({"id": sid, "html": render_html(sp), "width": sp["size"][0], "height": sp["size"][1]})
    res = render_pages(pages, work)
    bench = bench_image_shas()
    rel_base = rel_dir(img_dir)
    img_of, stats = {}, Counter()
    html_sha = {p["id"]: hashlib.sha256(p["html"].encode()).hexdigest() for p in pages}
    for sid, sp in specs.items():
        r = res.get(sid)
        if not r or not r["ok"]:
            stats["render_failed"] += 1
            continue
        els = {e["el"]: e for e in r["elements"]}
        cov_mask = set()
        if sp.get("cover") is not None and sp["kind"] != "table":
            i = sp["categories"].index(sp["cover"])
            cov_mask = {f"dl_{j}_{i}" for j in range(len(sp["series"]))} | {f"dl_t_{i}"}
        bad = [b for b in required_elements(sp, els) if b not in cov_mask]
        if bad:
            stats["clipped_or_overlapping"] += 1
            continue
        if not covered_ok(sp, els):
            stats["cover_leaks"] += 1
            continue
        post = post_for(sp)
        got = finish_image(work / f"{sid}.png", post, img_dir, rel_base, bench)
        if got is None:
            stats["bench_sha"] += 1
            continue
        img_of[sid] = (got[0], got[1], post)
    rows, id_of = [], {}
    for it in items:
        if it["spec_id"] not in img_of:
            continue
        if it["parent"] is not None and id(it["parent"]) not in id_of:
            continue
        sp = specs[it["spec_id"]]
        rel, sha, post = img_of[it["spec_id"]]
        rid = f"{id_prefix}-{seed}-{len(rows):06d}"
        id_of[id(it)] = rid
        prov = {"licence": "generated", "generator": GENERATOR, "seed": seed,
                "renderer": "SVG/HTML -> puppeteer-core + headless Chromium (scripts/p3/gui_templates/render.mjs) + PIL",
                "task": it["task"], "chart_kind": sp["kind"], "spec": sp, "params": {k: v for k, v in it["params"].items() if k != "state"},
                "html_sha256": html_sha[it["spec_id"]], "post": post, "image_sha256": [sha], "viewport": sp["size"],
                "unknown_construction": it["variant"], "upstream_split": "generated",
                "content": "invented brands, places, people and numbers"}
        rows.append({"id": rid, "source": "I", "dataset": "chart_synthetic", "family": FAMILY[it["task"]],
                     "difficulty": it["difficulty"], "state": it["state"], "images": [rel], "field": it["field"],
                     "gold": it["gold"], "unknown_reason": it["unknown_reason"], "gold_kind": "constructed",
                     "parent_id": id_of[id(it["parent"])] if it["parent"] is not None else None, "provenance": prov})
    return trim(rows, count, unknown_share), stats


def trim(rows, count, unknown_share):
    """Cut to `count` rows keeping the unknown share and whole parent/child pairs."""
    if len(rows) <= count:
        return rows
    n_unk = int(round(count * unknown_share))
    unk = [r for r in rows if r["gold"] is None][:n_unk]
    keep_par = {r["parent_id"] for r in unk if r["parent_id"]}
    rest = [r for r in rows if r["gold"] is not None]
    must = [r for r in rest if r["id"] in keep_par]
    other = [r for r in rest if r["id"] not in keep_par]
    chosen = {r["id"] for r in must + other[:max(0, count - len(unk) - len(must))]} | {r["id"] for r in unk}
    return [r for r in rows if r["id"] in chosen]


# ------------------------------------------------------------------------------------------------ variants
def variants_of(path: str, per: int, work: Path, img_dir: Path = IMG_DIR) -> list[dict]:
    """Stage-2 variants: fresh charts of the same kind and task, same difficulty, rng keyed by the parent id.
    An unknown parent yields unknown variants."""
    from candidate import read
    out_specs, planned = {}, []
    for par in read(path):
        if not str(par.get("dataset")) == "chart_synthetic":
            continue
        task, kind = par["provenance"]["task"], par["provenance"]["chart_kind"]
        want_unk = par["gold"] is None
        made = 0
        for k in range(200):
            if made >= per:
                break
            rng = rng_for("p3-charts-variant", par["id"], k)
            try:
                sp = make_spec_kind(rng, f"v{hashlib.sha1(par['id'].encode()).hexdigest()[:10]}-{k}", kind)
            except Skip:
                continue
            cands = [c for c in plan_questions(rng, sp) if c[0] == task]
            for task_, qtype, q, opts, p in cands:
                it = make_item(sp, task_, qtype, q, opts, p)
                if it is None:
                    continue
                if want_unk:
                    v = unknown_variant(rng, sp, it)
                    if v is None:
                        continue
                    sp2, p2, q2, how = v
                    tgt = sp2 or sp
                    if sp2 is not None:
                        sp2["id"] = sp["id"] + "u"
                        try:
                            sp2["show"] = layout(copy.deepcopy(sp2))["show"]
                        except Skip:
                            continue
                    optl = [o["text"] for o in it["field"].get("options", [])] or [l["description"] for l in it["field"].get("levels", [])] or None
                    it = make_item(tgt, task_, qtype, q2, optl, p2, parent=None, variant=how)
                    if it is None or it["gold"] is not None:
                        continue
                    sp = tgt
                it["difficulty"] = par["difficulty"]
                it["vparent"] = par["id"]
                it["vk"] = made + 1
                out_specs[sp["id"]] = sp
                planned.append(it)
                made += 1
                break
    pages = [{"id": sid, "html": render_html(sp), "width": sp["size"][0], "height": sp["size"][1]} for sid, sp in out_specs.items()]
    res = render_pages(pages, work) if pages else {}
    bench = bench_image_shas()
    rel_base = rel_dir(img_dir)
    rows = []
    for it in planned:
        sp = out_specs[it["spec_id"]]
        r = res.get(sp["id"])
        if not r or not r["ok"]:
            continue
        els = {e["el"]: e for e in r["elements"]}
        if required_elements(sp, els) and not sp.get("cover") or not covered_ok(sp, els):
            continue
        post = post_for(sp)
        got = finish_image(work / f"{sp['id']}.png", post, img_dir, rel_base, bench)
        if got is None:
            continue
        html_sha = hashlib.sha256(render_html(sp).encode()).hexdigest()
        rows.append({"id": f"{it['vparent']}-v{it['vk']}", "source": "I", "dataset": "chart_synthetic", "family": FAMILY[it["task"]],
                     "difficulty": it["difficulty"], "state": it["state"], "images": [got[0]], "field": it["field"], "gold": it["gold"],
                     "unknown_reason": it["unknown_reason"], "gold_kind": "constructed", "parent_id": it["vparent"],
                     "provenance": {"licence": "generated", "generator": GENERATOR, "task": it["task"], "chart_kind": sp["kind"],
                                    "spec": sp, "params": {k: v for k, v in it["params"].items() if k != "state"}, "html_sha256": html_sha,
                                    "post": post, "image_sha256": [got[1]], "viewport": sp["size"], "unknown_construction": it["variant"],
                                    "upstream_split": "generated", "variant_of": it["vparent"],
                                    "content": "invented brands, places, people and numbers"}})
    return rows


def make_spec_kind(rng, sid, kind):
    for _ in range(30):
        try:
            if kind in ("pie", "donut"):
                sp = make_pie(rng, sid, kind == "donut")
            elif kind == "scatter":
                sp = make_scatter(rng, sid)
            elif kind == "table":
                sp = make_table(rng, sid)
            elif kind == "dual":
                sp = make_dual(rng, sid)
            else:
                sp = make_cartesian(rng, sid, kind)
            sp["show"] = layout(sp)["show"]
            return sp
        except Skip:
            continue
    raise Skip("kind")


def contact_sheets(rows: list[dict], out_prefix: Path, n: int = 24, per_sheet: int = 6, seed: str = "sheet") -> list[Path]:
    """Visual QA: PNG sheets of n random rows (image + question + options, gold marked) for a human to look at."""
    import random
    from PIL import Image
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="p3sheet-"))
    pages, paths = [], []
    for k in range(0, len(pick), per_sheet):
        cells, hmax = [], 0
        for r in pick[k:k + per_sheet]:
            im = r["images"][0]
            ip = Path(im) if im.startswith("/") else ROOT / "data" / im
            w, h = Image.open(ip).size
            f = r["field"]
            if f["type"] == "choice":
                opts = "".join(f'<li{" class=g" if o["key"] == r["gold"] else ""}>{E(o["text"])}</li>' for o in f["options"])
            elif f["type"] == "score":
                opts = "".join(f'<li{" class=g" if l["value"] == r["gold"] else ""}>{l["value"]}: {E(l["description"])}</li>' for l in f["levels"])
            else:
                opts = ""
            gold = "UNKNOWN (" + str(r["unknown_reason"]) + ")" if r["gold"] is None else json.dumps(r["gold"])
            st = json.dumps(r["state"], ensure_ascii=False) if r["state"] else ""
            scale = min(1.0, 580 / w)
            hmax = max(hmax, h * scale)
            cells.append(f'<div class=c><img src="file://{ip}" style="width:{w * scale:.0f}px;height:{h * scale:.0f}px">'
                         f'<div class=m>{E(r["id"])} · {E(r["provenance"].get("task", ""))} · {f["type"]} · d{r["difficulty"]}'
                         f'{" · " + E(str(r["provenance"].get("unknown_construction"))) if r["gold"] is None else ""}</div>'
                         f'<div class=q>{E(f["question"])}</div>{"<div class=s>state: " + E(st) + "</div>" if st else ""}<ol>{opts}</ol>'
                         f'<div class=a>gold: {E(gold)}</div></div>')
        rows_n = (len(cells) + 2) // 3
        H = int(rows_n * (hmax + 250)) + 30
        html = ('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font:14px Helvetica,Arial,sans-serif;background:#eee}'
                '.w{display:grid;grid-template-columns:repeat(3,600px);gap:10px;padding:10px}.c{background:#fff;padding:8px;border:1px solid #ccc}'
                'img{display:block;border:1px solid #999}.m{color:#666;font-size:12px;margin-top:4px}.q{font-weight:bold;margin:4px 0}'
                '.s{color:#555;font-size:12px}ol{margin:4px 0;padding-left:22px}li.g{background:#bdf5c0;font-weight:bold}.a{color:#a00}'
                '</style></head><body><div class=w>' + "".join(cells) + "</div></body></html>")
        pages.append({"id": f"sheet{k // per_sheet}", "html": html, "width": 1830, "height": H})
    render_pages(pages, work, conc=2)
    for pg in pages:
        dst = Path(f"{out_prefix}-{pg['id'][5:]}.png")
        shutil.copyfile(work / f"{pg['id']}.png", dst)
        paths.append(dst)
    shutil.rmtree(work, ignore_errors=True)
    return paths


def summary(rows):
    return {"rows": len(rows), "unknown": sum(r["gold"] is None for r in rows),
            "by_family": dict(Counter(r["family"] for r in rows)), "by_task": dict(Counter(r["provenance"]["task"] for r in rows)),
            "by_kind": dict(Counter(r["provenance"]["chart_kind"] for r in rows)), "by_type": dict(Counter(r["field"]["type"] for r in rows)),
            "by_difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
            "unknown_by_construction": dict(Counter(r["provenance"]["unknown_construction"] for r in rows if r["gold"] is None)),
            "images": len({r["images"][0] for r in rows})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", default="c1")
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--img-dir", default=str(IMG_DIR))
    ap.add_argument("--id-prefix", default="p3-I-chart")
    ap.add_argument("--heldout-tag", default=None, help="mark rows as fresh held-out items (provenance.heldout*)")
    ap.add_argument("--variant-of", default=None, help="candidate JSONL of chart parents: write Stage-2 variants instead")
    ap.add_argument("--variants-per", type=int, default=2)
    ap.add_argument("--keep-work", default=None)
    a = ap.parse_args()
    work = Path(a.keep_work) if a.keep_work else Path(tempfile.mkdtemp(prefix="p3charts-"))
    work.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.img_dir).resolve()
    if a.variant_of:
        rows = variants_of(a.variant_of, a.variants_per, work, img_dir)
        stats = {}
    else:
        rows, stats = build(a.seed, a.count, work, img_dir, a.id_prefix)
    if a.heldout_tag:
        for r in rows:
            r["provenance"].update({"heldout": "fresh", "heldout_tag": a.heldout_tag, "heldout_generator": f"gen_charts.py --seed {a.seed}"})
    n = write(a.out, rows)
    print(json.dumps({"wrote": n, "out": a.out, "stats": dict(stats), **summary(rows)}, indent=1))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
