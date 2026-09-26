"""Long-document builder: numbered sections, cross-references, boilerplate padding to a word target, and case records."""
from __future__ import annotations

import json
import random
import re

from . import prose
from .common import DOMAINS, CURRENCIES, P, People, chance, fd, money, org_name, words

REF_WORDS = ["clause", "paragraph", "section", "§", "clause", "paragraph"]
HEADING_STYLES = ["md", "caps", "num", "article"]


class Doc:
    """Collects sections; renders numbering (k / k.j), resolves {ref:anchor} placeholders, pads to a word target."""

    def __init__(self, rng: random.Random, dom_id: str, doc_kind: str):
        self.rng, self.dom_id, self.dom, self.doc_kind = rng, dom_id, DOMAINS[dom_id], doc_kind
        self.people = People(rng)
        self.org = org_name(rng, self.dom)
        self.cur, self.sym = rng.choice(CURRENCIES)
        self.dstyle = rng.randrange(5)
        self.refword = rng.choice(REF_WORDS)
        self.hstyle = rng.choice(HEADING_STYLES)
        d = self.dom
        depts = rng.sample(d["depts"], 2)
        roles = rng.sample(d["roles"][:4], 2)
        self.ctx = {"org": self.org, "dept": depts[0], "dept2": depts[1], "role": roles[0], "role2": roles[1],
                    "system": rng.choice(d["systems"]), "party": d["party"], "parties": d["parties"], "requests": d["requests"],
                    "person": self.people()}
        self.front: list[str] = []
        self.sections: list[dict] = []   # {"title", "clauses": [{"anchor", "text"}], "fixed": bool, "kind"}
        self.tail: list[str] = []        # rendered after sections (case record etc.)
        self.labels: dict[str, str] = {}
        self.used_topics: set[str] = set()

    # formatting
    def fd(self, x) -> str:
        return fd(x, self.dstyle)

    def m(self, x, cents=None) -> str:
        return money(x, self.sym, cents)

    def ref(self, anchor: str) -> str:
        return "{ref:" + anchor + "}"

    def add(self, title: str, clauses: list, kind: str = "core", anchors: list | None = None) -> dict:
        cl = []
        for i, c in enumerate(clauses):
            if isinstance(c, tuple):
                cl.append({"anchor": c[0], "text": c[1]})
            else:
                cl.append({"anchor": (anchors[i] if anchors and i < len(anchors) else None), "text": c})
        s = {"title": title, "clauses": cl, "kind": kind}
        self.sections.append(s)
        return s

    def _fresh_topic(self):
        left = [t for t in prose.TOPIC_IDS if t not in self.used_topics]
        if not left:
            return None
        t = self.rng.choice(left)
        self.used_topics.add(t)
        return t

    def boiler(self, n: int, exclude=()) -> None:
        self.used_topics.update(exclude)
        for _ in range(n):
            t = self._fresh_topic()
            if t is None:
                return
            title, sents = prose.section(self.rng, t, self.ctx)
            # group sentences into 1-3 clauses
            k = self.rng.randint(1, min(3, len(sents)))
            groups = [sents[i::k] for i in range(k)]
            self.add(title, [" ".join(g) for g in groups], kind="boiler")

    def heading(self, num: int, title: str) -> str:
        if self.hstyle == "md":
            return f"## {num}. {title}"
        if self.hstyle == "caps":
            return f"{num}. {title.upper()}"
        if self.hstyle == "article":
            return f"Article {num} — {title}"
        return f"{num}  {title}"

    def order(self, core_order: list[dict], boiler_front: int = 2) -> None:
        """Interleave boilerplate sections between core sections, keeping the core sections' relative order."""
        boil = [s for s in self.sections if s["kind"] == "boiler"]
        self.rng.shuffle(boil)
        front, rest = boil[:boiler_front], boil[boiler_front:]
        out = list(front)
        slots = [[] for _ in range(len(core_order) + 1)]
        for b in rest:
            slots[self.rng.randrange(1, len(core_order) + 1)].append(b)
        for i, c in enumerate(core_order):
            out.append(c)
            out.extend(slots[i + 1])
        self.sections = out

    def render(self) -> str:
        self.labels = {}
        for i, s in enumerate(self.sections, 1):
            s["num"] = i
            for j, c in enumerate(s["clauses"], 1):
                c["label"] = f"{i}.{j}" if len(s["clauses"]) > 1 or s["kind"] != "boiler" else f"{i}"
                if c["anchor"]:
                    self.labels[c["anchor"]] = c["label"]
            if s.get("anchor"):
                self.labels[s["anchor"]] = str(i)
        rw = self.refword

        def sub(m):
            lab = self.labels[m.group(1)]
            return f"§{lab}" if rw == "§" else f"{rw} {lab}"
        lines = list(self.front)
        for s in self.sections:
            lines.append("")
            lines.append(self.heading(s["num"], s["title"]))
            for c in s["clauses"]:
                txt = re.sub(r"\{ref:([a-z0-9_]+)\}", sub, c["text"])
                txt = txt[:1].upper() + txt[1:]
                c["rendered"] = txt
                if "\n" in txt and txt.lstrip().startswith("|"):
                    lines.append(txt)
                elif len(s["clauses"]) == 1 and s["kind"] == "boiler":
                    lines.append(txt)
                else:
                    lines.append(f"{c['label']} {txt}")
        lines.extend(self.tail)
        return "\n".join(lines).strip() + "\n"

    def rendered(self, anchor: str) -> str:
        for s in self.sections:
            for c in s["clauses"]:
                if c["anchor"] == anchor:
                    return c.get("rendered", "")
        return ""

    def fit(self, lo: int = 1500, hi: int = 2800, target: int | None = None, render_tail=None) -> str:
        """Add or drop boilerplate sections until the rendered document is within [lo, hi] words (target in between)."""
        target = target or self.rng.randint(lo + 120, hi - 180)
        for _ in range(40):
            if render_tail:
                render_tail()
            text = self.render()
            n = words(text)
            if lo <= n <= hi and abs(n - target) < 260:
                return text
            if n < target:
                t = self._fresh_topic()
                if t is None:
                    if "version_history" in self.used_topics:
                        break
                    self.used_topics.add("version_history")
                    import datetime as _dt
                    ds = [_dt.date(2019 + i, self.rng.randint(1, 12), self.rng.randint(1, 28)) for i in range(self.rng.randint(4, 7))]
                    rows = prose.version_history(self.rng, self.ctx, ds, self.fd)
                    sec = {"title": self.rng.choice(["Version history", "Document control", "Revision history"]),
                           "clauses": [{"anchor": None, "text": "Editorial history only; substantive changes are recorded in the amendment provisions.\n" + "\n".join(rows)}],
                           "kind": "boiler"}
                    self.sections.insert(max(1, min(self.rng.randrange(1, len(self.sections) + 1),
                                                    max(i for i, s in enumerate(self.sections) if s["kind"] != "boiler"))), sec)
                    continue
                title, sents = prose.section(self.rng, t, self.ctx)
                k = self.rng.randint(1, min(3, len(sents)))
                sec = {"title": title, "clauses": [{"anchor": None, "text": " ".join(sents[i::k])} for i in range(k)], "kind": "boiler"}
                pos = self.rng.randrange(1, len(self.sections) + 1)
                # never insert after the amendments/schedules at the end
                last_core = max(i for i, s in enumerate(self.sections) if s["kind"] != "boiler")
                pos = min(pos, last_core)
                self.sections.insert(max(1, pos), sec)
            else:
                boil = [i for i, s in enumerate(self.sections) if s["kind"] == "boiler"]
                if not boil:
                    break
                del self.sections[self.rng.choice(boil)]
        if render_tail:
            render_tail()
        text = self.render()
        if not lo <= words(text) <= hi:
            raise ValueError(f"could not fit document ({words(text)} words)")
        return text


def render_record(rng: random.Random, title: str, fields: list[tuple[str, str]], style: int | None = None,
                  notes: list[str] | None = None) -> list[str]:
    style = rng.randrange(4) if style is None else style
    out = [""]
    if style == 0:
        out.append(f"{title}")
        out.append("-" * min(60, len(title)))
        out += [f"{k}: {v}" for k, v in fields]
    elif style == 1:
        out.append(f"### {title}")
        out.append("| Field | Value |")
        out.append("|---|---|")
        out += [f"| {k} | {v} |" for k, v in fields]
    elif style == 2:
        out.append(f"{title} (exported record)")
        out.append(json.dumps({k: v for k, v in fields}, indent=2, ensure_ascii=False))
    else:
        out.append(f"=== {title.upper()} ===")
        out += [f"- {k}: {v}" for k, v in fields]
    if notes:
        out.append("")
        out.append(P(rng, "Notes on file:", "Case notes:", "Additional notes:", "Handler notes:"))
        out += [f"* {n}" for n in notes]
    return out


def front_matter(doc: Doc, title: str, eff, version: str) -> list[str]:
    rng = doc.rng
    owner = doc.ctx["person"]
    style = rng.randrange(3)
    if style == 0:
        return [f"# {doc.org} — {title}", f"Version {version} | Effective {doc.fd(eff)} | Owner: {owner}, {doc.ctx['dept']}"]
    if style == 1:
        return [f"{doc.org.upper()}", f"{title}", f"Document owner: {doc.ctx['dept']} ({owner})", f"Version: {version}", f"In force from: {doc.fd(eff)}"]
    return [f"{title}", f"Issued by {doc.org}, {doc.ctx['dept']}. Version {version}, effective {doc.fd(eff)}."]
