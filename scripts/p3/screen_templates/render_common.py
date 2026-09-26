"""Shared helpers for gen_geometry.py and gen_screens.py: HTML -> PNG through screen_templates/render.mjs.

render(jobs, work) writes every job's HTML into `work`, runs the node renderer once, and returns {id: result}.
A job: {"id", "html", "width", "height", "dpr"?, "overlay"?, "shot": bool}. PNGs land at work/<id>.png when shot is true.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

RENDER = Path(__file__).resolve().parent / "render.mjs"
TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")


def render(jobs: list[dict], work: Path, conc: int = 4, tag: str = "r") -> dict:
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in jobs:
        hp = work / f"{j['id']}.html"
        if not hp.exists() or hp.read_text() != j["html"]:
            hp.write_text(j["html"])
        rows.append({"id": j["id"], "html": str(hp), "out": str(work / f"{j['id']}.png") if j.get("shot", True) else None,
                     "width": j["width"], "height": j["height"], "dpr": j.get("dpr", 1), "overlay": j.get("overlay") or []})
    jp, rp = work / f"jobs-{tag}.jsonl", work / f"results-{tag}.jsonl"
    jp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    subprocess.run(["node", str(RENDER), str(jp), str(rp), str(conc)], check=True)
    return {r["id"]: r for r in (json.loads(l) for l in rp.read_text().splitlines() if l.strip())}


def ntokens(text: str) -> int:
    """Token count as the decontam 13-gram checker sees it (lower-case [a-z0-9]+ runs)."""
    return len(TOKEN_RE.findall(str(text).lower()))
