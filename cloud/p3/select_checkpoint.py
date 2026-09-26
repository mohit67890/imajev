"""Phase-3 checkpoint selector (docs/phase-3-plan.md Stage 4 "Pick").

Reads ONLY <eval-root>/<checkpoint>/selection.json (own held-out hard set, fresh-seed + flagged halves, and the human/Kimi dev
slice) and <eval-root>/<checkpoint>/gates.json. It never opens the DecisionBench tracking files: they live under a separate
`tracking-only` root, every read goes through `_read_json`, which refuses any path under a `tracking-only` directory or naming
DecisionBench, and directory listing skips such directories (tests/test_p3_pod.py enforces this by instrumenting open()).

Rule:
  score  = mean(fresh-seed held-out accuracy, human-slice accuracy)   (fresh only when the human slice is missing)
  pool   = checkpoints whose gates.json says shippable (all gates pass; no soup) -- or every checkpoint with --no-gates
           (round-2 start point, before the gates have run)
  pick   = among pool checkpoints within noise of the best score (band = sqrt(2) x SE of the best score), the LATEST
           (largest training order: phase 2c's dev-picked early step had not absorbed the image replay)
  none   = no checkpoint passes -> {"pick": null, "best_by_score": ...}; the pod script then soups best_by_score with the shipped
           adapter and re-runs the gates (the soup is the fallback only).

    python cloud/p3/select_checkpoint.py --eval-root <O>/eval [--names a,b] [--no-gates] --out <O>/pick.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FORBIDDEN_PARTS = ("tracking-only",)
FORBIDDEN_NAMES = ("decisionbench",)


def _allowed(path: Path) -> bool:
    parts = [p.lower() for p in Path(path).parts]
    return not any(f in parts for f in FORBIDDEN_PARTS) and not any(n in parts[-1] for n in FORBIDDEN_NAMES)


def _read_json(path: Path) -> dict | None:
    if not _allowed(path):
        raise PermissionError(f"the selector may not read {path} (DecisionBench is tracking only)")
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def load_candidates(eval_root: Path, names: list[str] | None = None) -> list[dict]:
    if not _allowed(eval_root):
        raise PermissionError(f"eval root {eval_root} is under a tracking-only directory")
    dirs = [eval_root / n for n in names] if names else sorted(d for d in Path(eval_root).iterdir() if d.is_dir() and _allowed(d))
    out = []
    for d in dirs:
        sel = _read_json(d / "selection.json")
        if not sel or not sel.get("fresh") or sel["fresh"].get("acc") is None:
            continue
        gates = _read_json(d / "gates.json")
        out.append({"name": sel["name"], "dir": str(d), "ckpt": sel.get("ckpt"), "order": sel.get("order", 0.0), "round": sel.get("round"),
                    "step": sel.get("step"), "fresh": sel["fresh"], "human": sel.get("human"), "flagged": sel.get("flagged"),
                    "shippable": bool(gates and gates.get("shippable")), "gates_present": gates is not None,
                    "soup": bool(gates and gates.get("soup"))})
    return out


def score(c: dict) -> tuple[float, float]:
    """(score in points, standard error in points)."""
    pf, nf = c["fresh"]["acc"] / 100.0, max(1, c["fresh"]["n"])
    h = c.get("human")
    if h and h.get("acc") is not None and h.get("n"):
        ph, nh = h["acc"] / 100.0, h["n"]
        return 100 * (pf + ph) / 2, 100 * 0.5 * math.sqrt(pf * (1 - pf) / nf + ph * (1 - ph) / nh)
    return 100 * pf, 100 * math.sqrt(pf * (1 - pf) / nf)


def select(cands: list[dict], require_gates: bool = True) -> dict:
    for c in cands:
        c["score"], c["se"] = score(c)
    ranked = sorted(cands, key=lambda c: (-c["score"], -c["order"]))
    pool = [c for c in cands if c["shippable"]] if require_gates else list(cands)
    res = {"rule": "score = mean(fresh held-out acc, human-slice acc); pool = all gates pass" + ("" if require_gates else " (gates not required)") +
                   "; latest checkpoint within sqrt(2) x SE of the best score",
           "candidates": [{k: c[k] for k in ("name", "score", "se", "order", "shippable", "soup")} for c in ranked],
           "best_by_score": ranked[0]["name"] if ranked else None, "best_by_score_ckpt": ranked[0]["ckpt"] if ranked else None}
    if not pool:
        res.update(pick=None, reason="no checkpoint passes every gate" if require_gates else "no candidates")
        return res
    best = max(pool, key=lambda c: (c["score"], c["order"]))
    band = math.sqrt(2) * best["se"]
    within = [c for c in pool if c["score"] >= best["score"] - band]
    pick = max(within, key=lambda c: (c["order"], c["score"]))
    res.update(pick=pick["name"], pick_ckpt=pick["ckpt"], pick_dir=pick["dir"], best=best["name"], band=band,
               within_noise=[c["name"] for c in within], reason=f"{pick['name']}: latest of {len(within)} within {band:.2f} pts of the best ({best['name']})")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--names", default="", help="comma list of checkpoint dirs to consider (default: all)")
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not _allowed(a.out):
        ap.error("--out may not be under tracking-only")
    res = select(load_candidates(a.eval_root, [n for n in a.names.split(",") if n] or None), require_gates=not a.no_gates)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"pick: {res.get('pick')} ({res.get('reason')}); best by score: {res.get('best_by_score')}")
    return 0 if res.get("pick") else 3


if __name__ == "__main__":
    raise SystemExit(main())
