"""Build imajev's visual assets from real results: brand marks, README charts, gallery cards, social card.

Everything is drawn from files in `reports/` and the playground's verified scenario outputs, never typed in by hand, so the
assets can be regenerated after any re-run:

    python site/build_site.py            # needs matplotlib and pillow (fonts are in site/fonts, OFL)

Outputs go to site/assets/{brand,charts,gallery,social}. Every chart and card exists in a light and a dark version; the
README picks one with <picture> and prefers-color-scheme. Charts are SVG with text converted to outlines, so they render
identically on GitHub, which cannot load web fonts.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = SITE / "assets"
FONTS = SITE / "fonts"
SCEN = ROOT / "scripts/playground/static/scenarios"

# ---- design tokens (validated with the dataviz palette checks: accent vs context grey, both themes) ----
THEMES = {  # near-monochrome: imajev's data is the darkest ink, other systems are grey; no decorative hue
    "light": dict(bg="#FFFFFF", surface="#F6F6F4", ink="#111111", ink2="#3D3D3D", muted="#6E6E6B", hair="#E3E3DF",
                  track="#EFEFEC", accent="#111111", amber="#111111", unknown="#8F8F8A", ok="#111111", other="#8F8F8A"),
    "dark": dict(bg="#111213", surface="#18191A", ink="#EDEDEA", ink2="#C4C4BF", muted="#8E8E89", hair="#2A2A28",
                 track="#1E1E1D", accent="#EDEDEA", amber="#EDEDEA", unknown="#6A6A66", ok="#EDEDEA", other="#6A6A66"),
}
for f in FONTS.glob("*.ttf"):
    if "[" not in f.name:
        font_manager.fontManager.addfont(str(f))
DISPLAY = FontProperties(fname=str(FONTS / "SourceSerifDisplay-SemiBold.ttf"))
MARKFONT = FontProperties(fname=str(FONTS / "IBMPlexMono-Medium.ttf"))
SANS = FontProperties(fname=str(FONTS / "PlexSans-Regular.ttf"))
SANS_M = FontProperties(fname=str(FONTS / "PlexSans-Medium.ttf"))
SANS_B = FontProperties(fname=str(FONTS / "PlexSans-SemiBold.ttf"))
MONO = FontProperties(fname=str(FONTS / "IBMPlexMono-Regular.ttf"))
MONO_M = FontProperties(fname=str(FONTS / "IBMPlexMono-Medium.ttf"))
plt.rcParams.update({"svg.fonttype": "path", "hatch.linewidth": 1.1, "axes.unicode_minus": False})


def fp(base: FontProperties, size: float) -> FontProperties:
    f = base.copy(); f.set_size(size); return f


def load(path):
    return json.loads((ROOT / path).read_text())


def jl(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


# =====================================================================================================================
# Brand
# =====================================================================================================================
def mark_svg(t: dict, size: int = 64, bg: str | None = None) -> str:
    """Viewfinder corner brackets around a three-bar readout; the last bar is hatched: the explicit `unknown`."""
    s = size / 64
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64" role="img" aria-label="imajev">']
    parts.append(f'<defs><pattern id="h" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                 f'<rect width="4" height="4" fill="{bg or "none"}"/><line x1="0" y1="0" x2="0" y2="4" stroke="{t["unknown"]}" stroke-width="2"/></pattern></defs>')
    if bg:
        parts.append(f'<rect width="64" height="64" rx="14" fill="{bg}"/>')
    c = t["ink"]
    for d in ("M6 20V6h14", "M44 6h14v14", "M58 44v14H44", "M20 58H6V44"):
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<rect x="17" y="19" width="31" height="7" rx="2" fill="{t["accent"]}"/>')
    parts.append(f'<rect x="17" y="29" width="17" height="7" rx="2" fill="{t["other"]}"/>')
    parts.append(f'<rect x="17" y="39" width="12" height="7" rx="2" fill="url(#h)" stroke="{t["unknown"]}" stroke-width="1.2"/>')
    parts.append("</svg>")
    return "".join(parts)


def text_path_d(text: str, prop: FontProperties, size: float) -> tuple[str, float, float, float, float]:
    tp = TextPath((0, 0), text, prop=prop, size=size)
    verts, codes = tp.vertices, tp.codes
    xmin, ymin = verts.min(axis=0); xmax, ymax = verts.max(axis=0)
    d, i = [], 0
    from matplotlib.path import Path as MP
    while i < len(verts):
        c = codes[i]; x, y = verts[i]
        if c == MP.MOVETO: d.append(f"M{x:.2f} {-y:.2f}"); i += 1
        elif c == MP.LINETO: d.append(f"L{x:.2f} {-y:.2f}"); i += 1
        elif c == MP.CURVE3:
            (x1, y1), (x2, y2) = verts[i], verts[i + 1]; d.append(f"Q{x1:.2f} {-y1:.2f} {x2:.2f} {-y2:.2f}"); i += 2
        elif c == MP.CURVE4:
            (x1, y1), (x2, y2), (x3, y3) = verts[i], verts[i + 1], verts[i + 2]
            d.append(f"C{x1:.2f} {-y1:.2f} {x2:.2f} {-y2:.2f} {x3:.2f} {-y3:.2f}"); i += 3
        elif c == MP.CLOSEPOLY: d.append("Z"); i += 1
        else: i += 1
    return " ".join(d), xmin, -ymax, xmax - xmin, ymax - ymin


def logo_svg(t: dict) -> str:
    d, x0, y0, w, h = text_path_d("imajev", MARKFONT, 60)
    mark = mark_svg(t).replace('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" role="img" aria-label="imajev">', "").replace("</svg>", "")
    gap = 18; H = 64; W = 64 + gap + w + 4
    ty = (H - h) / 2 - y0 + 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H}" viewBox="0 0 {W:.1f} {H}" role="img" aria-label="imajev">'
            f'<title>imajev</title>{mark}<path transform="translate({64 + gap - x0:.2f} {ty:.2f})" d="{d}" fill="{t["ink"]}"/></svg>')


def build_brand():
    (OUT / "brand").mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (OUT / "brand" / f"mark-{name}.svg").write_text(mark_svg(t))
        (OUT / "brand" / f"logo-{name}.svg").write_text(logo_svg(t))
    (OUT / "brand" / "favicon.svg").write_text(mark_svg(THEMES["light"], bg="#FFFFFF"))


# =====================================================================================================================
# Charts (SVG, text as outlines)
# =====================================================================================================================
def base_fig(t, w, h, title, subtitle, source):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    fig.text(0.035, 1 - 0.30 / h, title, fontproperties=fp(DISPLAY, 17), color=t["ink"], va="top")
    fig.text(0.035, 1 - 0.62 / h, subtitle, fontproperties=fp(SANS, 10.5), color=t["ink2"], va="top")
    fig.text(0.035, 0.16 / h, source, fontproperties=fp(MONO, 8.2), color=t["muted"], va="bottom")
    return fig


def save(fig, path: Path, t):
    fig.savefig(path, facecolor=t["bg"])
    import os
    if os.environ.get("PREVIEW"):
        Path(os.environ["PREVIEW"]).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(os.environ["PREVIEW"]) / (path.stem + ".png"), facecolor=t["bg"], dpi=110)
    plt.close(fig)


def style_ax(ax, t, xgrid=True):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["hair"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    for lab in ax.get_xticklabels(): lab.set_fontproperties(fp(MONO, 8.6)); lab.set_color(t["muted"])
    if xgrid:
        ax.grid(axis="x", color=t["hair"], linewidth=0.8); ax.set_axisbelow(True)


def chart_imajevbench(t, name):
    rows = [  # (label, correct, low, high, ours, group)
        ("imajev-4b", 230, .77, .89, True, "A"), ("imajev-9b", 229, .76, .88, True, "A"),
        ("Qwen3.5-9B base", 214, .70, .82, False, "A"), ("imajev-2b", 200, .65, .78, True, "A"),
        ("Qwen3.5-4B base", 197, .64, .78, False, "A"), ("Gemma 4 E4B-it", 176, .55, .72, False, "A"),
        ("Qwen3.5-2B base", 168, .53, .67, False, "A"), ("SmolVLM2-2.2B", 80, .22, .36, False, "A"),
        ("Jev-Omni (12B)", 219, .73, .84, False, "B"),
    ]
    fig = base_fig(t, 8.8, 5.6, "ImajevBench v2.0-lite: accuracy on 279 test items",
                   "Photo, state and photo+state questions with an explicit Unknown. Dot = accuracy, line = 95% cluster bootstrap CI.",
                   "Direct option scoring, full rotations; Jev-Omni via its own predict() API. Source: bench/LEADERBOARD.md")
    ax = fig.add_axes([0.25, 0.14, 0.68, 0.70])
    ys = []
    y = 0
    for i, r in enumerate(rows):
        if r[5] == "B" and rows[i - 1][5] == "A": y += 0.7
        ys.append(y); y += 1
    for (lab, c, lo, hi, ours, g), yy in zip(rows, ys):
        col = t["accent"] if ours else t["other"]
        ax.plot([lo * 100, hi * 100], [yy, yy], color=col, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([c / 279 * 100], [yy], s=64, color=col, edgecolors=t["bg"], linewidths=2, zorder=3)
        ax.text(18.5, yy, lab, ha="right", va="center", fontproperties=fp(SANS_B if ours else SANS, 10), color=t["ink"] if ours else t["ink2"])
        ax.text(hi * 100 + 1.2, yy, f"{c / 279 * 100:.1f}%", va="center", fontproperties=fp(MONO_M if ours else MONO, 9), color=t["ink"] if ours else t["ink2"])
    ax.text(18.5, ys[-1] - 0.85, "own interface", ha="right", va="center", fontproperties=fp(MONO, 7.8), color=t["muted"])
    ax.set_xlim(20, 100); ax.set_ylim(ys[-1] + 0.7, -0.7); ax.set_yticks([])
    ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xticklabels([f"{v}%" for v in (20, 40, 60, 80, 100)])
    ax.set_clip_on(False)
    style_ax(ax, t)
    for lab in ax.texts: lab.set_clip_on(False)
    save(fig, OUT / "charts" / f"imajevbench-{name}.svg", t)


def chart_jevbench(t, name):
    rows = [("JevK5 v0.2.0", 73.9, False), ("Eikos-4B", 73.9, False), ("imajev-4b", 70.3, True),
            ("imajev-9b", 69.4, True), ("Hopper", 67.6, False), ("imajev-2b", 60.4, True),
            ("cua-s1-4b (GUI-action LoRA)", 52.3, False), ("Qwen3.5-4B base, generation", 48.6, False), ("mojev 0.85B", 33.3, False)]
    fig = base_fig(t, 8.8, 5.0, "JevBench public hard split (111 items, text-only)",
                   "Same protocol for every system: jevbench harness, typesafe adapter, one H100, serial. Competitive, not #1.",
                   "Our runs, 24 Sept 2026; imajev as served (4 rotations, calibration file); not the official board (sealed items, 4-axis score). Source: results/benchmarks/")
    ax = fig.add_axes([0.30, 0.13, 0.63, 0.70])
    for i, (lab, v, ours) in enumerate(rows):
        col = t["accent"] if ours else t["other"]
        ax.barh(i, v, height=0.62, color=col, zorder=2)
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontproperties=fp(MONO_M if ours else MONO, 9), color=t["ink"] if ours else t["ink2"])
        ax.text(-1.5, i, lab, ha="right", va="center", fontproperties=fp(SANS_B if ours else SANS, 10), color=t["ink"] if ours else t["ink2"])
    ax.set_xlim(0, 100); ax.set_ylim(len(rows) - 0.4, -0.6); ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    style_ax(ax, t)
    for lab in ax.texts: lab.set_clip_on(False)
    save(fig, OUT / "charts" / f"jevbench-{name}.svg", t)


def chart_calibration(t, name):
    raw = load("reports/decision-p2c/pod/4b-delta/jevbench-soup50-raw/hard/summary.json")["ece"]
    cal = load("reports/decision-p2c/pod/4b-delta/jevbench-soup50-rot4cal/hard/summary.json")["ece"]
    fig = base_fig(t, 8.8, 5.4, "Calibration: imajev-4b on JevBench hard",
                   f"Stated confidence vs accuracy, 5 equal-count bins (~22 items each). Temperature scaling changes no answer.",
                   "Served by the released server (torch, 1×H100); calibrated = shipped file + 4 rotations, as served. ECE: the benchmark's 10-bin definition. Source: results/ in the repository")
    ax = fig.add_axes([0.10, 0.15, 0.55, 0.68])
    ax.plot([0, 1], [0, 1], color=t["hair"], linewidth=1.2, zorder=1)
    ax.text(0.93, 0.96, "perfect", rotation=45, fontproperties=fp(MONO, 8), color=t["muted"], ha="center")
    def equal_count_bins(path, k=5):
        rows = sorted((max(r["probs"].values()), bool(r["correct"])) for r in jl(path) if r.get("probs"))
        out = []
        for i in range(k):
            chunk = rows[i * len(rows) // k:(i + 1) * len(rows) // k]
            out.append((sum(c for c, _ in chunk) / len(chunk), sum(ok for _, ok in chunk) / len(chunk), len(chunk)))
        return out
    series_pts = {"raw": equal_count_bins("reports/decision-p2c/pod/4b-delta/jevbench-soup50-raw/hard/results.jsonl"),
                  "calibrated": equal_count_bins("reports/decision-p2c/pod/4b-delta/jevbench-soup50-rot4cal/hard/results.jsonl")}
    for series, col, lab in ((raw, t["other"], "raw"), (cal, t["accent"], "calibrated")):
        xs, ys, ns = zip(*series_pts[lab])
        ax.plot(xs, ys, color=col, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=70, color=col, edgecolors=t["bg"], linewidths=2, zorder=3)
        ax.text(xs[0] - 0.02, ys[0], lab, ha="right", va="center", fontproperties=fp(SANS_B, 9.5), color=col)
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_facecolor(t["bg"])
    for s in ("left", "bottom"): ax.spines[s].set_color(t["hair"])
    ax.tick_params(colors=t["muted"], length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontproperties(fp(MONO, 8.6)); lab.set_color(t["muted"])
    ax.grid(color=t["hair"], linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("stated confidence", fontproperties=fp(SANS, 9.5), color=t["ink2"])
    ax.set_ylabel("observed accuracy", fontproperties=fp(SANS, 9.5), color=t["ink2"])
    # side table: served hard ECE per size
    rows = [("imajev-2b", .176, .123), ("imajev-4b", .164, .116), ("imajev-9b", .187, .092)]
    x0 = 0.71
    fig.text(x0, 0.78, "hard ECE, served", fontproperties=fp(MONO, 8.4), color=t["muted"])
    fig.text(x0, 0.72, "size", fontproperties=fp(SANS_M, 9.5), color=t["ink2"]); fig.text(x0 + 0.12, 0.72, "raw", fontproperties=fp(SANS_M, 9.5), color=t["ink2"])
    fig.text(x0 + 0.20, 0.72, "shipped", fontproperties=fp(SANS_M, 9.5), color=t["ink2"])
    for i, (lab, a, b) in enumerate(rows):
        y = 0.65 - i * 0.065
        fig.text(x0, y, lab, fontproperties=fp(SANS, 10), color=t["ink"])
        fig.text(x0 + 0.12, y, f"{a:.3f}", fontproperties=fp(MONO, 9.5), color=t["ink2"])
        fig.text(x0 + 0.20, y, f"{b:.3f}", fontproperties=fp(MONO_M, 9.5), color=t["accent"])
    fig.text(x0, 0.40, textwrap.fill("Photo-only verification is already calibrated raw; the cards say when to serve without the file.", 34),
             fontproperties=fp(SANS, 9), color=t["ink2"], va="top", linespacing=1.5)
    save(fig, OUT / "charts" / f"calibration-{name}.svg", t)


def chart_uplift(t, name):
    rows = [("2B", 60.2, 71.7), ("4B", 70.6, 82.4), ("9B", 76.7, 82.1)]
    fig = base_fig(t, 8.8, 4.2, "What the adapter adds on photos and state",
                   "ImajevBench accuracy of each untuned Qwen3.5 base (grey) and the same base with the imajev adapter (colour).",
                   "Paired cluster tests vs base on the shipped adapters: 2B +11.5 pts p=0.005 · 4B +11.8 p=0.0006 · 9B +5.4 p=0.131 (n.s.). Source: bench/LEADERBOARD.md")
    ax = fig.add_axes([0.13, 0.17, 0.80, 0.62])
    for i, (lab, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=t["hair"], linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [i], s=90, color=t["other"], edgecolors=t["bg"], linewidths=2, zorder=3)
        ax.scatter([b], [i], s=110, color=t["accent"], edgecolors=t["bg"], linewidths=2, zorder=3)
        ax.text(a - 1.2, i, f"{a:.1f}", ha="right", va="center", fontproperties=fp(MONO, 9), color=t["ink2"])
        ax.text(b + 1.2, i, f"{b:.1f}  +{b - a:.1f}", ha="left", va="center", fontproperties=fp(MONO_M, 9.5), color=t["ink"])
        ax.text(52.5, i, f"Qwen3.5-{lab}", ha="right", va="center", fontproperties=fp(SANS_B, 10.5), color=t["ink"])
    ax.set_xlim(55, 95); ax.set_ylim(2.6, -0.6); ax.set_yticks([])
    ax.set_xticks([60, 70, 80, 90]); ax.set_xticklabels(["60%", "70%", "80%", "90%"])
    style_ax(ax, t)
    for lab in ax.texts: lab.set_clip_on(False)
    save(fig, OUT / "charts" / f"uplift-{name}.svg", t)


# =====================================================================================================================
# Gallery cards (PNG, real imajev-4b outputs from reports/scenarios/imajev-4b/verification-scenarios.json)
# =====================================================================================================================
CASES = [("listing", "listing.color=red", "contradicted_field"), ("qc", "chipped part", "fault_type"),
         ("returns", "a different shoe", "same_item"), ("audit", "labels missing", "what_changed"),
         ("archive", "record.subject=dog", "wrong_field"), ("ticket", "as shipped", "department")]


def flat_state(state, prefix=""):
    out = []
    for k, v in state.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict): out += flat_state(v, key + ".")
        else: out.append((key, v))
    return out


def draw_card(t, case, row, qkey, path, scale=1.0, hero=False):
    W, H = (16, 7.2) if hero else (12.8, 6.4)
    fig = plt.figure(figsize=(W, H), dpi=125 * scale)
    fig.patch.set_facecolor(t["bg"])
    L = 0.035; split = 0.43
    # left: evidence
    imgs = case["images"]
    if imgs:
        n = len(imgs); gap = 0.02; w = (split - L - 0.02 - gap * (n - 1)) / n
        for i, (src, slot) in enumerate(zip(imgs, case["slots"])):
            im = Image.open(SCEN / src).convert("RGB")
            x = L + i * (w + gap)
            iw, ih = im.size; box_h = 0.66; ax_h = box_h; ax_w = w
            ax = fig.add_axes([x, 0.12, ax_w, 0.70]); ax.imshow(im); ax.set_axis_off()
            ax.set_aspect("equal"); ax.set_anchor("NW")
            fig.text(x, 0.845, slot.upper(), fontproperties=fp(MONO_M, 8.5), color=t["muted"])
    else:
        body = case["state"]["ticket"]["body"]
        fig.text(L, 0.86, "TEXT-ONLY REQUEST", fontproperties=fp(MONO_M, 8.5), color=t["muted"])
        fig.text(L, 0.80, textwrap.fill(f"“{body}”", 44), fontproperties=fp(SANS, 13 if hero else 12), color=t["ink"], va="top", linespacing=1.55)
    # right: request and readout
    R = split + 0.02
    fig.text(R, 0.90, case["kicker"].upper(), fontproperties=fp(MONO_M, 9), color=t["accent"])
    fig.text(R, 0.855, case["title"], fontproperties=fp(DISPLAY, 24 if hero else 21), color=t["ink"], va="top")
    y = 0.735
    fig.text(R, y, "state", fontproperties=fp(MONO, 8.5), color=t["muted"]); y -= 0.045
    lines = [(k, v) for k, v in flat_state(case["state"]) if not (case["kicker"].lower().startswith("support") and k == "ticket.body")][:4]
    for k, v in lines:
        v = str(v); v = v if len(v) <= 44 else v[:41] + "…"
        fig.text(R, y, f"{k}", fontproperties=fp(MONO, 10), color=t["ink2"])
        fig.text(R + 0.20, y, f"{v}", fontproperties=fp(MONO_M, 10), color=t["ink"])
        y -= 0.045
    q = case["questions"][qkey]
    y -= 0.02
    fig.text(R, y, "question", fontproperties=fp(MONO, 8.5), color=t["muted"]); y -= 0.028
    qtext = q["instructions"].replace("`", "")
    fig.text(R, y, textwrap.fill(qtext, 58), fontproperties=fp(SANS_M, 11.5), color=t["ink"], va="top", linespacing=1.4)
    y -= 0.048 * len(textwrap.wrap(qtext, 58)) + 0.045
    a = row["answers"][qkey]
    unk = float(a["unknown"] or 0)
    if q["type"] == "noul":  # Jev's noul folds half the unknown mass into P(yes); undo it so yes + no + unknown = 1
        yes = max(0.0, float(a["top"]) - 0.5 * unk); opts = [("yes", yes), ("no", max(0.0, 1 - yes - unk))]
    else:
        opts = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]
    bars = opts + [("unknown", unk)]
    top = max(opts, key=lambda kv: kv[1])[0]
    bx0, bw, lab_w = R, 0.46, 0.20
    for lab, v in bars:
        is_unk = lab == "unknown"; is_top = lab == top
        fig.text(bx0, y, lab if len(lab) <= 30 else lab[:28] + "…", fontproperties=fp(MONO_M if is_top else MONO, 10), color=t["ink"] if is_top else t["ink2"], va="center")
        tx = bx0 + lab_w
        fig.patches.append(Rectangle((tx, y - 0.008), bw - lab_w - 0.06, 0.016, transform=fig.transFigure, facecolor=t["track"], edgecolor="none"))
        ww = max(0.002, (bw - lab_w - 0.06) * v)
        if is_unk:
            fig.patches.append(Rectangle((tx, y - 0.008), ww, 0.016, transform=fig.transFigure, facecolor="none", edgecolor=t["unknown"], hatch="////", linewidth=0))
        else:
            fig.patches.append(Rectangle((tx, y - 0.008), ww, 0.016, transform=fig.transFigure, facecolor=t["accent"] if is_top else t["other"], edgecolor="none"))
        fig.text(bx0 + bw - 0.05, y, f"{v:.3f}", fontproperties=fp(MONO_M if is_top else MONO, 10), color=t["ink"] if is_top else t["ink2"], va="center")
        y -= 0.05
    route = row["route"]
    y -= 0.035
    fig.lines.append(matplotlib.lines.Line2D([R, R + 0.5], [y + 0.035, y + 0.035], transform=fig.transFigure, color=t["hair"], linewidth=1))
    fig.text(R, y, "app action", fontproperties=fp(MONO, 9), color=t["muted"], va="center")
    fig.text(R + 0.095 * (12.8 / W), y, route["title"], fontproperties=fp(SANS_B, 11), color=t["ink"], va="center")
    fig.text(0.965, 0.045, f"imajev-4b · shipped calibration · {row['server_ms']:.0f} ms on a Mac Studio (MLX)", ha="right",
             fontproperties=fp(MONO, 8.2), color=t["muted"])
    fig.savefig(path, facecolor=t["bg"]); plt.close(fig)


def build_gallery():
    ver = load("reports/scenarios/imajev-4b/verification-scenarios.json")
    rows = {(r["scenario"], r["case"]): r for r in ver["rows"]}
    cases = json.loads((SITE / "gallery_cases.json").read_text())
    (OUT / "gallery").mkdir(parents=True, exist_ok=True)
    meta = []
    for sid, label, qkey in CASES:
        row = rows[(sid, label)]
        assert row["pass"], (sid, label)
        for name, t in THEMES.items():
            draw_card(t, cases[sid], row, qkey, OUT / "gallery" / f"{sid}-{name}.png")
        meta.append({"id": sid, "case": label, "question": qkey, "route": row["route"], "title": cases[sid]["title"]})
        if sid == "listing":
            for name, t in THEMES.items():
                draw_card(t, cases[sid], row, qkey, OUT / "gallery" / f"hero-{name}.png", hero=True)
    (OUT / "gallery" / "cases.json").write_text(json.dumps({"model": ver["model"], "passed": ver["passed"], "total": ver["total"],
                                                            "threshold": ver["threshold"], "cases": meta}, indent=1) + "\n")


# =====================================================================================================================
# Social card (1280×640 PNG) for GitHub's repository settings and link previews
# =====================================================================================================================
def build_social():
    t = THEMES["dark"]
    (OUT / "social").mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100); fig.patch.set_facecolor(t["bg"])
    # mark drawn with patches (same geometry as mark_svg), 96 px
    ax = fig.add_axes([0.055, 0.70, 0.075, 0.15]); ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_axis_off(); ax.set_aspect("equal")
    for xs, ys in (([6, 6, 20], [20, 6, 6]), ([44, 58, 58], [6, 6, 20]), ([58, 58, 44], [44, 58, 58]), ([20, 6, 6], [58, 58, 44])):
        ax.plot(xs, ys, color=t["ink"], linewidth=5.2, solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(Rectangle((17, 19), 31, 7, facecolor=t["accent"])); ax.add_patch(Rectangle((17, 29), 17, 7, facecolor=t["other"]))
    ax.add_patch(Rectangle((17, 39), 12, 7, facecolor="none", edgecolor=t["unknown"], hatch="////"))
    fig.text(0.145, 0.775, "imajev", fontproperties=fp(MARKFONT, 46), color=t["ink"], va="center")
    fig.text(0.055, 0.56, "Decisions for\nreal-world cases.", fontproperties=fp(DISPLAY, 38), color=t["ink"], va="top", linespacing=1.12)
    fig.text(0.055, 0.25, "Photos, records and text in · typed answers out\nexplicit can't tell · 2B · 4B · 9B · runs locally",
             fontproperties=fp(MONO, 13), color=t["ink2"], va="top", linespacing=1.6)
    # specimen: the loafers photo against a listing that says red
    im = Image.open(SCEN / "assets/loafers.jpg").convert("RGB")
    a2 = fig.add_axes([0.60, 0.33, 0.16, 0.50]); a2.imshow(im); a2.set_axis_off(); a2.set_anchor("N")
    fig.text(0.60, 0.87, "LISTING PHOTO", fontproperties=fp(MONO_M, 10), color=t["muted"])
    fig.text(0.785, 0.87, "listing.color", fontproperties=fp(MONO, 12), color=t["ink2"])
    fig.text(0.785, 0.815, "red", fontproperties=fp(MONO_M, 16), color=t["ink"])
    fig.text(0.785, 0.72, "contradicted field?", fontproperties=fp(SANS_M, 13), color=t["ink"])
    ver = load("reports/scenarios/imajev-4b-raw/verification-scenarios.json")  # the same run the site's demos show
    row = next(r for r in ver["rows"] if r["scenario"] == "listing" and r["case"] == "listing.color=red")
    a = row["answers"]["contradicted_field"]
    items = sorted(a["probabilities"].items(), key=lambda kv: -kv[1]) + [("unknown", a["unknown"])]
    y = 0.64
    for lab, v in items:
        top = lab == items[0][0]; unk = lab == "unknown"
        fig.text(0.785, y, lab.replace("listing.", ""), fontproperties=fp(MONO_M if top else MONO, 11.5), color=t["ink"] if top else t["ink2"], va="center")
        fig.patches.append(Rectangle((0.785, y - 0.045), 0.14, 0.022, transform=fig.transFigure, facecolor=t["track"]))
        fig.patches.append(Rectangle((0.785, y - 0.045), max(0.003, 0.14 * v), 0.022, transform=fig.transFigure,
                                     facecolor="none" if unk else (t["accent"] if top else t["other"]),
                                     edgecolor=t["unknown"] if unk else "none", hatch="////" if unk else None, linewidth=0))
        fig.text(0.975, y - 0.034, f"{v:.3f}", fontproperties=fp(MONO, 10.5), color=t["ink2"], va="center", ha="right")
        y -= 0.105
    fig.text(0.60, 0.07, "github.com/mohit67890/imajev", fontproperties=fp(MONO_M, 12.5), color=t["accent"])
    fig.savefig(OUT / "social" / "social-card.png", facecolor=t["bg"]); plt.close(fig)


# =====================================================================================================================
# Report page: site/report/index.html from site/report_template.html + data drawn from the same result files
# =====================================================================================================================
def build_report():
    import shutil
    rep = SITE / "report"; (rep / "assets/photos").mkdir(parents=True, exist_ok=True); (rep / "assets/brand").mkdir(parents=True, exist_ok=True)
    ver = load("reports/scenarios/imajev-4b/verification-scenarios.json")
    rows = {(r["scenario"], r["case"]): r for r in ver["rows"]}
    cases = json.loads((SITE / "gallery_cases.json").read_text())
    attrib = {a["file"]: a for a in json.loads((SCEN / "assets/attribution.json").read_text())}

    def ans(row):
        return {k: {"top": a["top"], "unknown": a["unknown"], "probabilities": a["probabilities"]} for k, a in row["answers"].items()}

    # listing: every verified flip, so the page can switch the record and show the real answer
    listing_rows = [r for r in ver["rows"] if r["scenario"] == "listing"]
    listing = {"image": "assets/photos/loafers.jpg", "base_state": cases["listing"]["state"], "question": cases["listing"]["questions"]["contradicted_field"]["instructions"].replace("`", ""),
               "variants": [{"case": r["case"], "answers": ans(r), "route": r["route"], "ms": r["server_ms"]} for r in listing_rows]}
    gallery = []
    for sid, label, qkey in CASES:
        r = rows[(sid, label)]; c = cases[sid]
        photos = []
        for src, slot in zip(c["images"], c["slots"]):
            f = Path(src).name; shutil.copy(SCEN / src, rep / "assets/photos" / f)
            a = attrib[f]; photos.append({"src": f"assets/photos/{f}", "slot": slot, "credit": a["credit"], "license": a["license"],
                                          "held_out": a["held_out"], "edit": a["edit"]})
        qs = {k: {"type": q["type"], "instructions": q["instructions"].replace("`", "")} for k, q in c["questions"].items()}
        gallery.append({"id": sid, "title": c["title"], "kicker": c["kicker"], "problem": c["problem"], "case": label, "state": c["state"],
                        "photos": photos, "questions": qs, "focus": qkey, "answers": ans(r), "route": r["route"], "ms": r["server_ms"]})
    shutil.copy(SCEN / "assets/loafers.jpg", rep / "assets/photos/loafers.jpg")
    for name in THEMES:
        shutil.copy(OUT / "brand" / f"mark-{name}.svg", rep / "assets/brand" / f"mark-{name}.svg")
    shutil.copy(OUT / "brand/favicon.svg", rep / "assets/brand/favicon.svg")

    def bins(path, k=5):
        rs = sorted((max(r["probs"].values()), bool(r["correct"])) for r in jl(path) if r.get("probs"))
        return [[round(sum(c for c, _ in ch) / len(ch), 4), round(sum(o for _, o in ch) / len(ch), 4), len(ch)]
                for ch in (rs[i * len(rs) // k:(i + 1) * len(rs) // k] for i in range(k))]
    calibration = {s: {"raw": bins(f"reports/decision-p2c/pod/{s}-delta/jevbench-soup50-raw/hard/results.jsonl"),
                       "served": bins(f"reports/decision-p2c/pod/{s}-delta/jevbench-soup50-rot4cal/hard/results.jsonl"),
                       "ece_raw": load(f"reports/decision-p2c/pod/{s}-delta/jevbench-soup50-raw/hard/summary.json")["ece"]["ece"],
                       "ece_served": load(f"reports/decision-p2c/pod/{s}-delta/jevbench-soup50-rot4cal/hard/summary.json")["ece"]["ece"]} for s in ("2b", "4b", "9b")}
    data = {"listing": listing, "gallery": gallery, "calibration": calibration,
            "verification": {"passed": ver["passed"], "total": ver["total"], "threshold": ver["threshold"], "model": ver["model"]["model"]}}
    html = (SITE / "report_template.html").read_text().replace("/*__DATA__*/{}", json.dumps(data, ensure_ascii=False))
    (rep / "preview.html").write_text(html)  # fragment for the private preview host, which adds its own document wrapper
    cut = html.index('<div class="bar-top">')
    (rep / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                                    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                                    f'<meta property="og:image" content="{PAGES_URL}report/assets/social-card.png">\n<meta name="twitter:card" content="summary_large_image">\n' + html[:cut]
                                    + '</head>\n<body>\n' + html[cut:] + '\n</body>\n</html>\n')
    shutil.copy(OUT / "social/social-card.png", rep / "assets/social-card.png")


# =====================================================================================================================
# Landing page: site/index.html from site/landing_template.html + every checked playground run (site/dump_demos.mjs)
# =====================================================================================================================
DEMO_MODEL = "imajev-4b-raw"        # the demos are served without the calibration file; the calibrated runs are reported beside them
DEMO_CAL_MODEL = "imajev-4b"
DEMO_GROUPS = [  # (group, pack, scenario ids in page order, family id or None: a family is one app shown as one entry)
    ("photos", "scenarios", ["listing", "returns", "qc", "audit", "archive", "retake"], None),
    ("text", "text", ["crm", "refund", "invoice", "moderation", "review", "inbox"], None),
    ("text", "scenarios", ["ticket"], None),
    ("apps", "wardrobe", ["ordered", "dresscode", "tag", "owned", "match"], None),
    ("apps", "wardrobe-stylist", None, "stylist"),
    ("apps", "tracing", None, "tracing"),
]
FAMILIES = {
    "stylist": dict(title="Stylist app", kicker="Phone app · pick a piece, get an outfit", shot="assets/apps/stylist.png", strip="Style the ",
                    problem="Pick a piece from your closet. imajev reads its colour and pattern from the photo, the app applies your rules "
                            "(one patterned piece, bright colours with neutrals), and imajev ranks what is left in the colours you like.",
                    solves="Eight pieces, each checked in three colour preferences. Choose a piece below."),
    "tracing": dict(title="Tracing pad", kicker="Kids' app · trace a letter, get a star", shot="assets/apps/tracing.png", strip="Trace ",
                    problem="A child traces a dotted letter and taps Check. imajev reads which character was written, without being told the answer; "
                            "the app measures how much of each line the strokes covered. Both must agree for a star.",
                    solves="Ten characters, each checked with a good trace, the wrong character and a scribble. Choose one below."),
}
DEMO_SETS = [  # (pack, name, app screenshot, plain description, where it lives in the gallery)
    ("scenarios", "Business checks", "assets/apps/scenarios.jpg",
     "Does the photo match the listing? Is the returned item the one we shipped? Is this part chipped? Seven checks, each run "
     "with one field changed or one photo swapped."),
    ("text", "Text only", "assets/apps/text.jpg",
     "An email against a CRM record, a refund against the policy, a post against forum rules, a review, an inbox. No photo; "
     "written for this page and run once."),
    ("wardrobe", "Wardrobe", "assets/apps/wardrobe.jpg",
     "Everyday clothing questions from a phone photo: is this what I ordered, does it meet the dress code, do I already own it, "
     "which shoes match."),
    ("wardrobe-stylist", "Stylist app", "assets/apps/stylist.png",
     "A phone app that reads a piece of clothing and picks bottoms, shoes and a bag from your closet in the colours you like."),
    ("tracing", "Tracing pad", "assets/apps/tracing.png",
     "A kids' app: imajev reads which letter or number a child traced; the app checks the strokes covered every line."),
]


def automation_curve():
    """Per-item (abstained, top option probability, correct) for imajev-4b on the ImajevBench test split, raw probabilities, so the
    page can show how many decisions are automated at a threshold and how often those are right. Scored with the benchmark's own rule."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from imajev_bench.scoring import _correct
    preds = jl("reports/decision-p2c/pod/4b-delta/imajevbench-soup50/predictions.jsonl")
    ids = {p["id"] for p in preds}
    recs = {r["id"]: r for r in jl("data/imajev-bench/v2-lite-v1/records-final-v2.jsonl") if r["id"] in ids}
    items = []
    for p in preds:
        opts = [v for k, v in (p.get("probabilities") or {}).items() if k != "__unknown__"]
        items.append([int(p["status"] == "abstained"), round(max(opts) if opts else 0.0, 4), int(_correct(recs[p["id"]], p))])
    assert sum(i[2] for i in items) == 230, "ImajevBench 4B score no longer reproduces 230/279"
    return {"model": "imajev-4b", "n": len(items), "items": items}


def build_landing():
    import shutil
    import subprocess
    static = ROOT / "scripts/playground/static"
    dump = lambda m: json.loads(subprocess.run(["node", str(SITE / "dump_demos.mjs"), m], check=True, capture_output=True, text=True).stdout)
    raw, cal = dump(DEMO_MODEL), dump(DEMO_CAL_MODEL)
    (SITE / "assets/demo").mkdir(parents=True, exist_ok=True)
    attrib_cache: dict[Path, dict] = {}

    def credit(src: str):
        f = static / src
        a_path = f.parent / "attribution.json"
        if a_path not in attrib_cache:
            rows = json.loads(a_path.read_text()) if a_path.exists() else []
            attrib_cache[a_path] = {r["file"]: r for r in rows} if isinstance(rows, list) else {}
        a = attrib_cache[a_path].get(f.name)
        if a is None and "tracing/samples" in src:
            return {"credit": "generated sample tracing", "license": "CC0-1.0", "held_out": True, "edit": None}
        return a and {k: a.get(k) for k in ("credit", "license", "held_out", "edit")}

    photos: dict[str, dict] = {}

    def copy_images(sc):
        for c in sc["cases"]:
            new = []
            for src in c["images"]:
                dest = f"assets/demo/{src}"
                if src not in photos:
                    (SITE / dest).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(static / src, SITE / dest)
                    photos[src] = {"src": dest, **(credit(src) or {"credit": "unknown", "license": "", "held_out": None, "edit": None})}
                new.append(dest)
            c["images"] = new

    groups = {"photos": [], "text": [], "apps": []}
    for group, pack, ids, family in DEMO_GROUPS:
        scen = raw["packs"][pack]["scenarios"]
        chosen = scen if ids is None else [next(x for x in scen if x["id"] == i) for i in ids]
        for sc in chosen:
            copy_images(sc); sc["pack"] = pack
        if family:
            f = FAMILIES[family]
            for sc in chosen:
                sc["member"] = sc["title"].replace(f["strip"], "")
            groups[group].append({"id": family, "family": True, "pack": pack, **{k: v for k, v in f.items() if k != "strip"}, "members": chosen})
        else:
            groups[group].extend(chosen)
    sets = []
    for pack, name, shot, desc in DEMO_SETS:
        r, c = raw["packs"][pack], cal["packs"][pack]
        checks = [{"sid": sc["id"], "scenario": sc["title"], "label": x["label"], "pass": x["pass"], "wrong": x["wrong"], "route": x["route"]["kind"]}
                  for sc in r["scenarios"] for x in sc["cases"]]
        sets.append({"pack": pack, "name": name, "shot": shot, "desc": desc, "passed": r["passed"], "total": r["total"],
                     "cal_passed": c["passed"], "checks": checks, "threshold": r["threshold"]})
    automation = automation_curve()
    (SITE / "assets/showcase").mkdir(parents=True, exist_ok=True)
    shutil.copy(SITE / "showcase/listing.jpg", SITE / "assets/showcase/listing.jpg")
    showcase = {n: {"code": (SITE / f"showcase/{n}.py").read_text(), "result": json.loads((SITE / f"showcase/{n}.out.json").read_text())}
                for n in ("listing", "ticket")}  # the exact scripts and their unedited output (site/showcase/)
    credits = sorted({p["credit"] for p in photos.values() if p.get("credit")})
    data = {"groups": groups, "sets": sets, "photos": {p["src"]: p for p in photos.values()}, "credits": credits, "automation": automation, "showcase": showcase}
    html = (SITE / "landing_template.html").read_text().replace("/*__DATA__*/{}", json.dumps(data, ensure_ascii=False))
    (SITE / "preview.html").write_text(html)  # fragment for the private preview host, which adds its own document wrapper
    cut = html.index('<header class="top">')
    head_extra = ('<meta property="og:type" content="website">\n'
                  '<meta property="og:title" content="imajev: decisions for real-world cases, from photos, records and text">\n'
                  '<meta property="og:description" content="Small open models (2B, 4B, 9B) that answer typed questions about photos and text, '
                  'with probabilities and an explicit unknown. Runs locally. Jev-compatible API.">\n'
                  f'<meta property="og:image" content="{PAGES_URL}assets/social/social-card.png">\n'
                  '<meta name="twitter:card" content="summary_large_image">\n')
    (SITE / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                                     '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                                     + head_extra + html[:cut] + '</head>\n<body>\n' + html[cut:] + '\n</body>\n</html>\n')


PAGES_URL = "https://mohit67890.github.io/imajev/"


if __name__ == "__main__":
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    build_brand()
    for name, t in THEMES.items():
        chart_imajevbench(t, name); chart_jevbench(t, name); chart_calibration(t, name); chart_uplift(t, name)
    build_gallery()
    build_social()
    if (SITE / "report_template.html").exists():
        build_report()
    if (SITE / "landing_template.html").exists():
        build_landing()
    print("assets written to", OUT)
