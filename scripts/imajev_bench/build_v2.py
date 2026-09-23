"""Build imajev-bench v2 synthetic items end to end: plan -> images -> items.

  plan    Write scene specs (facts, prompts, checks, variants, questions) and text items. No API calls.
  images  For each scene: generate the original with its assigned generator, check every fact with two
          model families, retry up to --retries times; then make each variant by editing the accepted
          original and check it the same way. Resumable: finished scenes are skipped. Writes a cost ledger.
  items   Turn accepted images plus text items into an `imajev_bench assemble` spec. Answers come from the
          scene facts (construction), never from a model. Variants whose check failed are left out; a
          scene is never dropped because some model answers its questions wrongly.

Typical run:
  build_v2.py plan   --out data/imajev-bench/v2-build/pilot --scenes 24 --text 12 --seed 1
  build_v2.py images --out data/imajev-bench/v2-build/pilot --workers 6
  build_v2.py items  --out data/imajev-bench/v2-build/pilot
  python -m imajev_bench assemble --spec data/imajev-bench/v2-build/pilot/spec.json --output data/imajev-bench/v2-lite-pilot
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imajev_bench import generation  # noqa: E402
from imajev_bench.scenes import Scene, Variant, answer, plan_scenes, render  # noqa: E402
from imajev_bench.textgen import text_items  # noqa: E402

LOCK = threading.Lock()


def save_jpeg(data: bytes, path: Path):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(path, "JPEG", quality=92, optimize=True)


def cmd_plan(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    scenes = plan_scenes(args.scenes, args.seed, kinds=args.kinds, first_share=args.flare_share)
    (out / "plan.json").write_text(json.dumps({"seed": args.seed, "style": generation.STYLE,
                                               "scenes": [s.to_json() for s in scenes]}, indent=1))
    text = text_items(args.text, args.seed) + text_items(getattr(args, "text_hard", 0), args.seed, hard=True)
    (out / "text_items.json").write_text(json.dumps(text, indent=1))
    print(json.dumps({"scenes": len(scenes), "variants": sum(len(s.variants) for s in scenes), "text_items": args.text}))


def _load_scene(raw):
    raw = dict(raw)
    raw["variants"] = [Variant(**v) for v in raw["variants"]]
    return Scene(**raw)


def _attempt(clients, backend, model, prompt, facts, source=None, retries=2):
    attempts = []
    for _ in range(retries + 1):
        try:
            image, meta = clients[backend].image(model, prompt, source=source)
        except Exception as exc:
            attempts.append({"error": str(exc)[:300]})
            continue
        check = generation.verify(clients, image, facts)
        attempts.append({"estimated_usd": meta["estimated_usd"], "checker_usd": check["checker_usd"],
                         "checkers": [model for _, model in generation.CHECKERS], "per_fact": check["per_fact"],
                         "hygiene": check["hygiene"], "notes": {k: v.get("note") for k, v in check["verdicts"].items()}})
        if check["passed"]:
            return image, attempts
    return None, attempts


def _png(path: Path) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def retry_variants(scene: Scene, folder: Path, result: dict, clients, retries):
    """Re-edit variants that never got a real check (quota or network errors) from the saved original."""
    backend, model = generation.GENERATORS[scene.generator]
    source = folder / "base-source.png"
    base = source.read_bytes() if source.exists() else _png(folder / "base.jpg")
    changed = False
    for variant in scene.variants:
        tries = result["attempts"].get(variant.name, [])
        if variant.name in result["images"] or not tries or not any("error" in a for a in tries):
            continue
        image, attempts = _attempt(clients, backend, model, f"{variant.edit} {generation.STYLE}", variant.checks,
                                   source=base, retries=retries)
        result["attempts"][variant.name] = tries + attempts
        changed = True
        if image is not None:
            save_jpeg(image, folder / f"{variant.name}.jpg")
            result["images"][variant.name] = f"{variant.name}.jpg"
    return changed


def build_scene(scene: Scene, out: Path, clients, retries, retry_missing=False):
    folder = out / "scenes" / scene.id
    done = folder / "result.json"
    if done.exists():
        result = json.loads(done.read_text())
        if retry_missing and "base" in result["images"] and retry_variants(scene, folder, result, clients, retries):
            result["estimated_usd"] = round(sum(a.get("estimated_usd", 0) for t in result["attempts"].values() for a in t), 4)
            done.write_text(json.dumps(result, indent=1))
            with LOCK:
                print(json.dumps({"scene": scene.id, "kind": scene.kind, "images": sorted(result["images"]),
                                  "usd": result["estimated_usd"], "retried_variants": True}), flush=True)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    backend, model = generation.GENERATORS[scene.generator]
    result = {"scene": scene.id, "generator": scene.generator, "model": model, "images": {}, "attempts": {},
              "checkers": [m for _, m in generation.CHECKERS]}
    base, attempts = _attempt(clients, backend, model, f"{scene.prompt} {generation.STYLE}", scene.checks, retries=retries)
    result["attempts"]["base"] = attempts
    if base is not None:
        save_jpeg(base, folder / "base.jpg")
        (folder / "base-source.png").write_bytes(base)  # lossless source for edits and later retries
        result["images"]["base"] = "base.jpg"
        for variant in scene.variants:
            image, attempts = _attempt(clients, backend, model, f"{variant.edit} {generation.STYLE}", variant.checks,
                                       source=base, retries=retries)
            result["attempts"][variant.name] = attempts
            if image is not None:
                save_jpeg(image, folder / f"{variant.name}.jpg")
                result["images"][variant.name] = f"{variant.name}.jpg"
    result["estimated_usd"] = round(sum(a.get("estimated_usd", 0) for tries in result["attempts"].values() for a in tries), 4)
    result["checker_usd"] = round(sum(a.get("checker_usd", 0) for tries in result["attempts"].values() for a in tries), 4)
    if base is None and all("error" in a for a in result["attempts"]["base"]):
        # Infrastructure failure only (quota, network): leave the scene unfinished so a resume retries it.
        (folder / f"errors-{int(time.time())}.json").write_text(json.dumps(result, indent=1))
        with LOCK:
            print(json.dumps({"scene": scene.id, "kind": scene.kind, "retry_later": True}), flush=True)
        return result
    done.write_text(json.dumps(result, indent=1))
    with LOCK:
        print(json.dumps({"scene": scene.id, "kind": scene.kind, "images": sorted(result["images"]),
                          "usd": result["estimated_usd"]}), flush=True)
    return result


def cmd_images(args):
    plan = json.loads((args.out / "plan.json").read_text())
    scenes = [_load_scene(s) for s in plan["scenes"]
              if not args.generator or s["generator"] == args.generator][:args.limit]
    clients = {"azure": generation.AzureClient(), "vertex": generation.VertexClient()}
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda s: build_scene(s, args.out, clients, args.retries, args.retry_missing), scenes))
    ledger = {"scenes": len(results), "accepted_originals": sum("base" in r["images"] for r in results),
              "accepted_variants": sum(len(r["images"]) - ("base" in r["images"]) for r in results),
              "planned_variants": sum(len(s.variants) for s in scenes),
              "estimated_generation_usd": round(sum(r["estimated_usd"] for r in results), 2),
              "estimated_checker_usd": round(sum(r.get("checker_usd", 0) for r in results), 2),
              "note": "Estimates from list prices; checker cost is logged only for runs after 2026-09-23. Bills are authoritative.",
              "minutes": round((time.time() - start) / 60, 1)}
    (args.out / f"ledger-{args.generator or 'all'}-{int(time.time())}.json").write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger))


def _facts_sha(facts):
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def cmd_items(args):
    plan = json.loads((args.out / "plan.json").read_text())
    scenes = {s["id"]: _load_scene(s) for s in plan["scenes"]}
    sources, items = [], []
    usage = {}
    accepted = {}
    for sid in scenes:
        path = args.out / "scenes" / sid / "result.json"
        if path.exists() and "base" in json.loads(path.read_text())["images"]:
            accepted[sid] = json.loads(path.read_text())
    # Pair accepted shelf scenes that share an object type and a colour, for two-image comparisons.
    pairs, waiting = [], None
    for sid in accepted:
        scene = scenes[sid]
        if scene.kind != "shelf" or not args.pairs:
            continue
        if waiting and scenes[waiting].facts["object"] == scene.facts["object"] \
                and set(scenes[waiting].facts["sequence"]) & set(scene.facts["sequence"]):
            pairs.append((scenes[waiting], scene))
            waiting = None
        else:
            waiting = sid
    paired = {s.id for pair in pairs for s in pair}
    for sid, scene in scenes.items():
        if sid not in accepted:
            continue
        result = accepted[sid]
        variants = {v.name: v for v in scene.variants}
        for name, file in result["images"].items():
            facts = scene.facts if name == "base" else variants[name].facts
            sources.append({"id": f"{sid}-{name}", "path": f"scenes/{sid}/{file}", "source_cluster": sid, "synthetic": True,
                            "provenance": {"generator": result["model"], "generator_family": result["generator"],
                                           "scene_kind": scene.kind, "variant": name, "facts_sha256": _facts_sha(facts),
                                           "image_checks": "passed: every stated fact confirmed by " + " and ".join(
                                               result.get("checkers") or ["two checker models"]) + "; no frame, identifiable "
                                               "faces or brands",
                                           "license": "CC0-1.0 (AI-generated image)"}})
        # Paired shelf images keep two questions so the comparison stays within three items per image.
        questions = list(scene.questions)[:2] if sid in paired else list(scene.questions)
        for qi, question in enumerate(questions):
            track, family, state, field = render(question["template"], question["params"])
            asked_on = ["base"] + ([n for n in result["images"] if n != "base"] if qi == 0 else [])
            original_gold = answer(question["template"], scene.facts, question["params"])
            for name in asked_on:
                facts = scene.facts if name == "base" else variants[name].facts
                gold = answer(question["template"], facts, question["params"])
                item = {"id": f"{sid}-{name}-q{qi}", "track": track, "family": family, "images": [f"{sid}-{name}"],
                        "state": state, "field": field, "draft_gold": gold,
                        "evidence": f"construction from scene facts ({scene.kind}, {name})",
                        "construction": {"truth_source": "generator_spec", "truth": gold, "scene": sid, "variant": name,
                                         "template": question["template"], "facts_sha256": _facts_sha(facts),
                                         "image_checks": "passed"}}
                # An Unknown that rests on something being covered needs a human to confirm it truly cannot be read;
                # checkers accepted faintly readable covers in the hard trial.
                if gold is None and (name.startswith("cover") or any(p is None for _, p in facts.get("items", []))):
                    item["construction"]["occlusion_check"] = True
                if qi == 0 and len(asked_on) > 1:
                    relation = None if name == "base" else ("same" if gold == original_gold and type(gold) is type(original_gold) else "change")
                    item["contrast"] = {"set_id": f"{sid}-q0", "role": "original"} if name == "base" else \
                        {"set_id": f"{sid}-q0", "role": "variant", "relation": relation}
                items.append(item)
                usage[f"{sid}-{name}"] = usage.get(f"{sid}-{name}", 0) + 1
    # Two-image comparisons between consecutive shelf scenes that share an object type and a colour.
    for a, b in pairs:
        colour = sorted(set(a.facts["sequence"]) & set(b.facts["sequence"]))[0]
        params = {"colour": colour, "object": a.facts["object"]}
        track, family, state, field = render("shelf_pair_more", params)
        gold = answer("shelf_pair_more", [a.facts, b.facts], params)
        items.append({"id": f"{a.id}-{b.id}-pair", "track": track, "family": family,
                      "images": [f"{a.id}-base", f"{b.id}-base"], "state": state, "field": field, "draft_gold": gold,
                      "evidence": "construction from both scenes' facts",
                      "construction": {"truth_source": "generator_spec", "truth": gold, "scene": f"{a.id}+{b.id}",
                                       "template": "shelf_pair_more", "image_checks": "passed"}})
    for item in json.loads((args.out / "text_items.json").read_text()):
        items.append(item)
    spec = {"dataset": args.dataset, "version": args.version, "split_salt": args.split_salt,
            "split_fractions": {"dev": args.dev, "calibration": args.calibration, "test": 1 - args.dev - args.calibration},
            "max_records_per_image": 3, "sources": sources, "items": items}
    (args.out / "spec.json").write_text(json.dumps(spec, indent=1))
    print(json.dumps({"sources": len(sources), "items": len(items),
                      "unknown": sum(i["draft_gold"] is None for i in items), "max_items_per_image": max(usage.values(), default=0)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", type=int, required=True)
    p.add_argument("--text", type=int, default=0)
    p.add_argument("--text-hard", type=int, default=0, help="Hard-tier text items (discounts/tax, exceptions, dates)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kinds", nargs="+", help="Scene kinds to plan (default: all standard and hard kinds)")
    p.add_argument("--flare-share", type=float, default=0.75,
                   help="Share of scenes generated with Flare; the rest use Nano Banana 2 (reported per generator)")
    i = sub.add_parser("images")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--workers", type=int, default=6)
    i.add_argument("--retries", type=int, default=2)
    i.add_argument("--limit", type=int)
    i.add_argument("--retry-missing", action="store_true", help="Re-edit variants that only hit quota/network errors")
    i.add_argument("--generator", choices=sorted(generation.GENERATORS),
                   help="Only scenes assigned to this generator (run one pool per generator so quotas don't block)")
    t = sub.add_parser("items")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--dataset", default="imajev-bench-v2-lite")
    t.add_argument("--version", default="0.1.0")
    t.add_argument("--split-salt", required=True)
    t.add_argument("--dev", type=float, default=0.3)
    t.add_argument("--calibration", type=float, default=0.15)
    t.add_argument("--no-pairs", dest="pairs", action="store_false")
    args = parser.parse_args(argv)
    {"plan": cmd_plan, "images": cmd_images, "items": cmd_items}[args.cmd](args)


if __name__ == "__main__":
    main()
