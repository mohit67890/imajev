"""Phase-3 Stage 0-I: Geometry3K-style diagram questions (families `geometry_*`, source I, constructed gold).

We draw the diagrams ourselves (SVG -> PNG through the local headless Chromium renderer
scripts/p3/screen_templates/render.mjs) from a numeric spec, and compute the gold from the same spec:

  geometry_angle     triangle angle sum, exterior angle, isosceles base/apex, intersecting lines with (ax+b)° labels,
                     polygon with one missing angle
  geometry_relation  parallel lines cut by a transversal: find an angle, name the pair (corresponding / alternate ...),
                     "is m∠i = m∠j" (noul)
  geometry_length    Pythagoras (legs / hypotenuse, exact or 1 d.p.), similar triangles with DE ∥ BC
  geometry_circle    radius / diameter / circumference / area (in π), inscribed vs central angle, Thales, tangent length
                     and tangent angle
  geometry_area      rectangle / triangle / trapezoid / parallelogram area and perimeter, parallelogram angles
  geometry_polygon   regular polygon interior / exterior / sum (count the sides in the figure), count sides (score)
  geometry_coord     points on a drawn grid (to scale): distance, slope, midpoint, triangle area, quadrant,
                     "is ABC a right triangle", "is ABCD a parallelogram" (noul)
  geometry_classify  isosceles? / right triangle by sides? (noul); acute, right or obtuse (choice)

Options are computed so exactly one is correct; distractors are the usual mistakes (supplement vs complement, radius vs
diameter, forgetting the ½, a²+b² vs a²-b², miscounting sides, the wrong angle-pair relation ...). Figures that the
question answers from given numbers carry "Figure not drawn to scale" (and are jittered so the picture never decides);
coordinate grids, regular polygons and the count-sides items are drawn to scale because the reader must read them.
Givens are drawn on the figure, and ~25% of them are moved into the state instead (the reader must join both).

Unknown children (~15% of rows): the parent spec with one decisive given or mark removed (an angle, the height, the
right-angle square, the parallel arrows, the equal-side ticks), re-drawn; gold null, unknown_reason
insufficient_evidence, parent_id = the answerable parent, same options. provenance.spec holds the whole spec so the
tests recompute every gold independently (tests/test_p3_geometry_screens.py).

    .venv/bin/python scripts/p3/gen_geometry.py [--seed g1] [--count 3000] [--out data/p3/candidates/I-geometry.jsonl]
    .venv/bin/python scripts/p3/gen_geometry.py --heldout [--count 250]      # fresh-seed held-out file
    .venv/bin/python scripts/p3/gen_geometry.py --variant-of rows.jsonl --variants-per 2 --out variants.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))
sys.path.insert(0, str(ROOT / "scripts" / "p3" / "geometry_templates"))
sys.path.insert(0, str(ROOT / "scripts" / "p3" / "screen_templates"))
from candidate import read, write  # noqa: E402
from convert_common import option_key  # noqa: E402
from gen_image_joint import bench_image_shas, rng_for  # noqa: E402
from fig import page  # noqa: E402
import templates as T  # noqa: E402
from render_common import render  # noqa: E402

GENERATOR = "scripts/p3/gen_geometry.py"
IMG_DIR = ROOT / "data" / "p3" / "images" / "geometry"
OUT = ROOT / "data" / "p3" / "candidates" / "I-geometry.jsonl"
HELDOUT_OUT = ROOT / "data" / "p3" / "pool" / "heldout-fresh-geometry.jsonl"
P_CHILD = 0.26
N_OPTIONS = (4, 4, 4, 5)


# ------------------------------------------------------------------------------------------------ options
def _valid(spec, v) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(v) or v <= 0:
            return False
        if T.formatter(spec) is T.deg and v >= 360:
            return False
    if isinstance(v, Fraction):
        return True
    return True


def options_for(spec, answer, rng) -> list[str] | None:
    """Option texts (answer first, before shuffling), exactly one formatted like the answer."""
    t = spec["t"]
    fm = T.formatter(spec)
    if t == "parallel" and spec["kind"] == "relation":
        others = [r for r in T.RELATIONS if r != answer]
        return [answer] + rng.sample(others, rng.choice([3, 4]))
    if t == "classify" and spec["kind"] == "type":
        return ["acute", "right", "obtuse"]
    if t == "coord" and spec["kind"] == "quad":
        return ["Quadrant I", "Quadrant II", "Quadrant III", "Quadrant IV"]
    cands = TT[t][3](spec, rng)
    want = rng.choice(N_OPTIONS)
    texts = [fm(answer)]
    for v in cands[1:]:
        if not _valid(spec, v):
            continue
        s = fm(v)
        if s not in texts:
            texts.append(s)
        if len(texts) == want:
            break
    step = 1
    base = answer
    while len(texts) < want and isinstance(base, (int, float, Fraction)) and not isinstance(base, bool):
        for v in (base + step, base - step) if not isinstance(base, Fraction) else (base + Fraction(step, 2), base - Fraction(step, 2)):
            if _valid(spec, v) and fm(v) not in texts and len(texts) < want:
                texts.append(fm(v))
        step += 1
        if step > 40:
            break
    return texts if len(texts) >= 3 else None


TT = T.TEMPLATES


# ------------------------------------------------------------------------------------------------ items
def make_item(rng, template: str | None = None, kind: str | None = None, want_unknown: bool | None = None):
    """-> list of item dicts: the parent, plus an unknown child (P_CHILD) when the template has a child maker."""
    names = list(TT)
    for _ in range(400):
        t = template or rng.choices(names, weights=[TT[n][7] for n in names])[0]
        spec = TT[t][1](rng)
        if kind is not None and spec.get("kind", spec.get("shape")) != kind:
            continue
        break
    else:
        raise RuntimeError(f"could not sample {template}/{kind}")
    fam = TT[t][0]
    answer = TT[t][4](spec)
    assert answer is not None, spec
    ftype = T.answer_type(spec)
    opts = options_for(spec, answer, rng) if ftype == "choice" else None
    if ftype == "choice" and opts is None:
        return []
    if opts:
        gold_text = answer if isinstance(answer, str) else T.formatter(spec)(answer)
        assert opts.count(gold_text) == 1, (gold_text, opts)
        rng.shuffle(opts)
    parent = {"spec": spec, "template": t, "family": fam, "type": ftype, "answer": answer, "options": opts,
              "draw_seed": rng.randrange(1 << 30), "unknown_construction": None}
    out = [parent]
    maker = TT[t][8]
    make_child = want_unknown if want_unknown is not None else (maker is not None and rng.random() < P_CHILD)
    if make_child and maker is not None:
        res = maker(spec, rng)
        if res is not None:
            cspec, how = res
            if TT[t][4](cspec) is None:
                out.append({"spec": cspec, "template": t, "family": fam, "type": ftype, "answer": None, "options": opts,
                            "draw_seed": rng.randrange(1 << 30), "unknown_construction": how})
    if want_unknown and len(out) < 2:
        return []
    return out


def draw(item) -> tuple[str, dict]:
    import random
    spec = item["spec"]
    rng = random.Random(item["draw_seed"])
    b = TT[item["template"]][2](spec, rng, child=item["answer"] is None)
    W, H = b["canvas"]
    margin = b["margin"] or (70, 64, 70, 70)
    svg = b["fig"].svg(W, H, margin=margin, fit=b["fit"])
    extra = []
    if b["note"]:
        extra.append(f'<text x="14" y="{H - 14}" font-size="15" font-style="italic" font-family="Helvetica, Arial, sans-serif" '
                     f'fill="#555">{T.NOTE}</text>')
    for i, g in enumerate(b.get("given_px") or []):
        extra.append(f'<text x="16" y="{30 + 24 * i}" font-size="19" font-family="Helvetica, Arial, sans-serif" fill="#1b1b1b">'
                     f'{g}</text>')
    if extra:
        svg = svg.replace("</svg>", "".join(extra) + "</svg>")
    return page(svg, W, H), b


def to_row(item, b, sha, img_rel, seed) -> dict:
    spec, ftype = item["spec"], item["type"]
    field = {"type": ftype, "question": b["question"]}
    gold = None
    if ftype == "choice":
        taken = set()
        opts = [{"key": option_key(o, taken), "text": o} for o in item["options"]]
        field["options"] = opts
        if item["answer"] is not None:
            fm = T.formatter(spec)
            at = item["answer"] if isinstance(item["answer"], str) else fm(item["answer"])
            gold = next(o["key"] for o in opts if o["text"] == at)
    elif ftype == "score":
        field["levels"] = b["levels"]
        gold = item["answer"]
    else:
        gold = item["answer"]
    unk = None if gold is not None else "insufficient_evidence"
    diff = b["difficulty"] + (1 if gold is None else 0) + (1 if b["state"].get("given") and b["difficulty"] < 4 else 0)
    diff = max(2, min(5, diff))
    return {"id": None, "source": "I", "dataset": "geometry_synthetic", "family": item["family"], "difficulty": diff,
            "state": b["state"], "images": [img_rel], "field": field, "gold": gold, "unknown_reason": unk,
            "gold_kind": "constructed", "parent_id": None,
            "provenance": {"licence": "generated", "generator": GENERATOR,
                           "renderer": "SVG in headless Chromium (scripts/p3/screen_templates/render.mjs)",
                           "template": item["template"], "spec": spec, "canvas": list(b["canvas"]),
                           "drawn_to_scale": not b["note"], "figure_note": T.NOTE if b["note"] else None,
                           "unknown_construction": item["unknown_construction"], "image_sha256": [sha],
                           "upstream_split": "generated", "seed": seed, "draw_seed": item["draw_seed"],
                           "content": "our own diagrams; fictional labels"}}


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, Fraction):
        return str(x)
    return x


def render_items(groups: list[list[dict]], work: Path, img_dir: Path, seed: str, conc: int) -> tuple[list[list[dict]], Counter]:
    """groups: [[parent, child?], ...] -> rows per group (child parent_id filled later)."""
    jobs, meta = [], []
    for gi, grp in enumerate(groups):
        for k, it in enumerate(grp):
            html, b = draw(it)
            jid = f"g{gi:06d}_{k}"
            W, H = b["canvas"]
            jobs.append({"id": jid, "html": html, "width": W, "height": H})
            meta.append((gi, k, it, b, jid))
    res = render(jobs, work, conc=conc, tag="geo")
    bench = bench_image_shas()
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    out = [[None] * len(g) for g in groups]
    for gi, k, it, b, jid in meta:
        r = res.get(jid)
        if not r or not r["ok"]:
            stats["render_failed"] += 1
            continue
        png = work / f"{jid}.png"
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        if sha in bench:
            stats["bench_sha"] += 1
            continue
        dst = img_dir / f"{sha}.png"
        if not dst.exists():
            shutil.copyfile(png, dst)
        rel = __import__("os").path.relpath(dst, ROOT / "data")
        it = dict(it, spec=_json_safe(it["spec"]))
        if isinstance(it["answer"], Fraction):
            pass
        out[gi][k] = to_row(it, b, sha, rel, seed)
    return out, stats


def assign_ids(groups_rows, prefix: str) -> list[dict]:
    rows, seen = [], set()
    n = 0
    for grp in groups_rows:
        if grp[0] is None:
            continue
        key = (grp[0]["images"][0], grp[0]["field"]["question"])
        if key in seen:
            continue
        seen.add(key)
        pid = f"{prefix}{n:06d}"
        grp[0]["id"] = pid
        rows.append(grp[0])
        n += 1
        for ch in grp[1:]:
            if ch is None:
                continue
            ch["id"] = f"{prefix}{n:06d}"
            ch["parent_id"] = pid
            rows.append(ch)
            n += 1
    return rows


def generate(seed: str, count: int, work: Path, img_dir: Path, prefix: str, conc: int = 4) -> tuple[list[dict], Counter]:
    groups, i = [], 0
    total = 0
    while total < count:
        rng = rng_for(seed, "geo", i)
        i += 1
        g = make_item(rng)
        if not g:
            continue
        groups.append(g)
        total += len(g)
    grs, stats = render_items(groups, work, img_dir, seed, conc)
    rows = assign_ids(grs, prefix)
    return rows, stats


def variants_of(path: str, per: int, work: Path, img_dir: Path, conc: int = 4) -> list[dict]:
    """Stage-2 variants: same template, kind and unknown-ness; fresh rng keyed by the parent id; parent_id = that row."""
    groups, parents = [], []
    for row in read(path):
        pv = row.get("provenance") or {}
        if pv.get("generator") != GENERATOR:
            continue
        spec = pv["spec"]
        want_unk = row["gold"] is None
        for k in range(per):
            for attempt in range(60):
                rng = rng_for("variant", row["id"], k, attempt)
                g = make_item(rng, spec["t"], spec.get("kind", spec.get("shape")), want_unknown=want_unk)
                if g:
                    break
            else:
                continue
            groups.append([g[1]] if want_unk else [g[0]])
            parents.append((row["id"], k))
    grs, _ = render_items(groups, work, img_dir, "variant", conc)
    out = []
    for (pid, k), grp in zip(parents, grs):
        r = grp[0]
        if r is None:
            continue
        r["id"] = f"{pid}-v{k + 1}"
        r["parent_id"] = pid
        r["provenance"]["variant_of"] = pid
        out.append(r)
    return out


def summary(rows):
    return {"rows": len(rows), "by_family": dict(Counter(r["family"] for r in rows)),
            "by_template": dict(Counter(r["provenance"]["template"] for r in rows)),
            "by_type": dict(Counter(r["field"]["type"] for r in rows)),
            "by_difficulty": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
            "unknown": sum(r["gold"] is None for r in rows),
            "unknown_share": round(sum(r["gold"] is None for r in rows) / max(len(rows), 1), 3),
            "images": len({r["images"][0] for r in rows}),
            "given_in_state": sum("given" in r["state"] for r in rows)}


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
    ap.add_argument("--heldout", action="store_true", help="fresh-seed held-out file (images under images/geometry/heldout)")
    ap.add_argument("--heldout-tag", default="hf1")
    ap.add_argument("--train", nargs="*", default=None,
                    help="held-out mode: training files to leakage-filter against (default: the training candidates file + "
                         "data/p3/pool/shards/*.jsonl + large-choice.jsonl)")
    ap.add_argument("--variant-of", default=None, help="candidate JSONL with gen_geometry parents -> Stage-2 variants")
    ap.add_argument("--variants-per", type=int, default=2)
    ap.add_argument("--keep-work", default=None)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--img-dir", default=None, help="override the image directory (smoke tests)")
    a = ap.parse_args()
    global IMG_DIR
    if a.img_dir:
        IMG_DIR = Path(a.img_dir).resolve()
    work = Path(a.keep_work) if a.keep_work else Path(tempfile.mkdtemp(prefix="p3geo-"))
    if a.variant_of:
        rows = variants_of(a.variant_of, a.variants_per, work, IMG_DIR, a.conc)
        out = a.out or str(OUT.with_name("I-geometry-var.jsonl"))
    elif a.heldout:
        seed = a.seed or f"p3-geometry-heldout-{a.heldout_tag}"
        target = a.count or 250
        rows, stats = generate(seed, int(target * 1.4), work, IMG_DIR / "heldout", f"p3-{a.heldout_tag}-I-geometry-", a.conc)
        for r in rows:
            r["provenance"].update({"heldout": "fresh", "heldout_tag": a.heldout_tag, "heldout_generator": "gen_geometry.py --heldout"})
        rows, hstats = heldout_rows(rows, a.train or default_train(OUT), target, lambda r: json.dumps(r["provenance"]["spec"], sort_keys=True))
        stats = dict(stats, heldout_filter=hstats)
        print(json.dumps(hstats))
        out = a.out or str(HELDOUT_OUT)
    else:
        seed = a.seed or "p3-geometry-v1"
        rows, stats = generate(seed, a.count or 3000, work, IMG_DIR, "p3-I-geometry-", a.conc)
        out = a.out or str(OUT)
    n = write(out, rows)
    print(json.dumps({"out": out, "written": n, **summary(rows)}, indent=1, ensure_ascii=False))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
