"""Tiny SVG figure builder for scripts/p3/gen_geometry.py (math coordinates, y up; fitted into the canvas on render).

Primitives: segments (solid / dashed), full lines through two points (clipped to the figure box), circles, points,
labels placed away from a reference point, angle arcs with a value, right-angle squares, equal-length ticks and
parallel arrow marks. Everything is plain SVG in an HTML page rendered by screen_templates/render.mjs.
"""
from __future__ import annotations

import html
import math

E = html.escape
INK = "#1b1b1b"
ACC = "#1f5fbf"


def unit(v):
    n = math.hypot(*v) or 1.0
    return (v[0] / n, v[1] / n)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, k):
    return (a[0] * k, a[1] * k)


def rot(p, ang, c=(0.0, 0.0)):
    x, y = p[0] - c[0], p[1] - c[1]
    ca, sa = math.cos(ang), math.sin(ang)
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


class Fig:
    def __init__(self):
        self.items = []          # (kind, data)
        self.extent = []         # math points that must fit in the canvas

    # ---------------------------------------------------------------- primitives (math coords)
    def seg(self, p, q, dash=False, w=2.2, color=INK):
        self.items.append(("seg", (p, q, dash, w, color)))
        self.extent += [p, q]

    def poly(self, pts, w=2.2):
        for i in range(len(pts)):
            self.seg(pts[i], pts[(i + 1) % len(pts)], w=w)

    def circle(self, c, r, w=2.2):
        self.items.append(("circle", (c, r, w)))
        self.extent += [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]

    def dot(self, p, r=3.5):
        self.items.append(("dot", (p, r)))
        self.extent.append(p)

    def label(self, p, text, away=None, dist=16, size=22, italic=True, color=INK, anchor_px=None):
        """Text near p, pushed `dist` px away from `away` (a math point) or in direction anchor_px (screen dx, dy)."""
        self.items.append(("label", (p, text, away, dist, size, italic, color, anchor_px)))
        self.extent.append(p)

    def angle(self, v, p, q, text=None, r=26, right=False, color=ACC, size=18, arcs=1, tdist=None):
        """Mark the (smaller) angle p-v-q: an arc (or a right-angle square) and optional text on the bisector."""
        self.items.append(("angle", (v, p, q, text, r, right, color, size, arcs, tdist)))

    def tick(self, p, q, n=1):
        self.items.append(("tick", (p, q, n)))

    def arrow(self, p, q, n=1, at=0.5):
        self.items.append(("arrow", (p, q, n, at)))

    def text_px(self, x, y, text, size=16, color="#444", anchor="start", italic=False, weight=400):
        """Free text in canvas pixels (notes, given lists)."""
        self.items.append(("textpx", (x, y, text, size, color, anchor, italic, weight)))

    def raw_px(self, svg):
        self.items.append(("rawpx", svg))

    # ---------------------------------------------------------------- render
    def svg(self, W, H, margin=(60, 60, 60, 60), fit=None, flip_y=True):
        """margin = (left, top, right, bottom) px. fit = explicit (minx, miny, maxx, maxy) in math coords (grids)."""
        ml, mt, mr, mb = margin
        if fit is None:
            xs = [p[0] for p in self.extent]
            ys = [p[1] for p in self.extent]
            minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        else:
            minx, miny, maxx, maxy = fit
        sw = (W - ml - mr) / max(maxx - minx, 1e-9)
        sh = (H - mt - mb) / max(maxy - miny, 1e-9)
        s = min(sw, sh)
        ox = ml + ((W - ml - mr) - s * (maxx - minx)) / 2
        oy = mt + ((H - mt - mb) - s * (maxy - miny)) / 2

        def T(p):
            return (ox + (p[0] - minx) * s, oy + (maxy - p[1]) * s)

        self.T, self.scale = T, s
        out = []
        for kind, d in self.items:
            if kind == "seg":
                p, q, dash, w, color = d
                a, b = T(p), T(q)
                da = ' stroke-dasharray="7 5"' if dash else ""
                out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" '
                           f'stroke-width="{w}" stroke-linecap="round"{da}/>')
            elif kind == "circle":
                c, r, w = d
                a = T(c)
                out.append(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="{r * s:.1f}" fill="none" stroke="{INK}" stroke-width="{w}"/>')
            elif kind == "dot":
                p, r = d
                a = T(p)
                out.append(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="{r}" fill="{INK}"/>')
            elif kind == "label":
                p, text, away, dist, size, italic, color, anchor_px = d
                a = T(p)
                if anchor_px is not None:
                    dx, dy = unit(anchor_px)
                elif away is not None:
                    b = T(away)
                    dx, dy = unit((a[0] - b[0], a[1] - b[1]))
                else:
                    dx, dy = 0.0, -1.0
                if away is not None or anchor_px is not None:
                    dist = max(dist, 5 + abs(dx) * 0.29 * size * len(text) + abs(dy) * 0.55 * size)
                x, y = a[0] + dx * dist, a[1] + dy * dist
                st = "italic" if italic else "normal"
                fam = "Georgia, 'Times New Roman', serif" if italic else "Helvetica, Arial, sans-serif"
                out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-style="{st}" font-family="{fam}" fill="{color}" '
                           f'text-anchor="middle" dominant-baseline="central">{E(text)}</text>')
            elif kind == "angle":
                v, p, q, text, r, right, color, size, arcs, tdist = d
                V, P, Q = T(v), T(p), T(q)
                u1 = unit((P[0] - V[0], P[1] - V[1]))
                u2 = unit((Q[0] - V[0], Q[1] - V[1]))
                bis = unit((u1[0] + u2[0], u1[1] + u2[1]))
                if right:
                    k = 15
                    a = (V[0] + u1[0] * k, V[1] + u1[1] * k)
                    c = (V[0] + u2[0] * k, V[1] + u2[1] * k)
                    m = (V[0] + (u1[0] + u2[0]) * k, V[1] + (u1[1] + u2[1]) * k)
                    out.append(f'<polyline points="{a[0]:.1f},{a[1]:.1f} {m[0]:.1f},{m[1]:.1f} {c[0]:.1f},{c[1]:.1f}" '
                               f'fill="none" stroke="{INK}" stroke-width="1.8"/>')
                else:
                    cross = u1[0] * u2[1] - u1[1] * u2[0]
                    sweep = 1 if cross > 0 else 0
                    for k in range(arcs):
                        rr = r + 5 * k
                        a = (V[0] + u1[0] * rr, V[1] + u1[1] * rr)
                        c = (V[0] + u2[0] * rr, V[1] + u2[1] * rr)
                        out.append(f'<path d="M{a[0]:.1f},{a[1]:.1f} A{rr},{rr} 0 0 {sweep} {c[0]:.1f},{c[1]:.1f}" '
                                   f'fill="none" stroke="{color}" stroke-width="1.8"/>')
                if text:
                    half = math.acos(max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))) / 2
                    need = (5 + 4.6 * len(text)) / max(math.sin(half), 0.08)
                    dd = tdist if tdist else min(95, max(r + 10 + 3.2 * len(text) + 5 * (arcs - 1), need))
                    x, y = V[0] + bis[0] * dd, V[1] + bis[1] * dd
                    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" '
                               f'fill="{color}" text-anchor="middle" dominant-baseline="central">{E(text)}</text>')
            elif kind == "tick":
                p, q, n = d
                a, b = T(p), T(q)
                m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                u = unit((b[0] - a[0], b[1] - a[1]))
                nrm = (-u[1], u[0])
                for k in range(n):
                    off = (k - (n - 1) / 2) * 6
                    c = (m[0] + u[0] * off, m[1] + u[1] * off)
                    out.append(f'<line x1="{c[0] - nrm[0] * 8:.1f}" y1="{c[1] - nrm[1] * 8:.1f}" x2="{c[0] + nrm[0] * 8:.1f}" '
                               f'y2="{c[1] + nrm[1] * 8:.1f}" stroke="{INK}" stroke-width="2"/>')
            elif kind == "arrow":
                p, q, n, at = d
                a, b = T(p), T(q)
                m = (a[0] + (b[0] - a[0]) * at, a[1] + (b[1] - a[1]) * at)
                u = unit((b[0] - a[0], b[1] - a[1]))
                nrm = (-u[1], u[0])
                for k in range(n):
                    off = (k - (n - 1) / 2) * 8
                    tip = (m[0] + u[0] * (off + 5), m[1] + u[1] * (off + 5))
                    l1 = (tip[0] - u[0] * 9 + nrm[0] * 6, tip[1] - u[1] * 9 + nrm[1] * 6)
                    l2 = (tip[0] - u[0] * 9 - nrm[0] * 6, tip[1] - u[1] * 9 - nrm[1] * 6)
                    out.append(f'<polyline points="{l1[0]:.1f},{l1[1]:.1f} {tip[0]:.1f},{tip[1]:.1f} {l2[0]:.1f},{l2[1]:.1f}" '
                               f'fill="none" stroke="{INK}" stroke-width="2.2"/>')
            elif kind == "textpx":
                x, y, text, size, color, anchor, italic, weight = d
                st = ' font-style="italic"' if italic else ""
                out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" fill="{color}" '
                           f'text-anchor="{anchor}" font-weight="{weight}"{st}>{E(text)}</text>')
            elif kind == "rawpx":
                out.append(d)
        body = "\n".join(out)
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>'


def page(svg: str, W: int, H: int) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}'
            f'svg{{display:block}}</style></head><body style="width:{W}px;height:{H}px">{svg}</body></html>')
