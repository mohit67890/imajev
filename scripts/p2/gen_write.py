"""Stage 1 of decision-p2: the WRITER model produces documents + three hard typed questions each.

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 python scripts/p2/gen_write.py --docs 6000 --out data/decision-p2/teacher/writer.jsonl --workers 32

Resumable: documents already present in --out (by doc id) are skipped. Every accepted document is validated with the pydantic
schema; rejected outputs are logged to <out>.rejects.jsonl with the reason and are retried once with a new seed offset.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import normalise_writer_output, loads_lenient, ChatClient, WRITER_MODEL, WriterOutput, append_jsonl, doc_id, read_jsonl, run_parallel
from families import BRIEFS, FAMILIES, FAMILY_IDS, STATE_SHAPES, WRITER_SYSTEM, forced_type, intended_unknown, pick_families, pick_types, writer_prompt
from domains import DOMAINS, domain as get_domain


def restricted_families(families: list[str], index: int, rng: random.Random, n: int = 3) -> list[str]:
    """Round-robin over a restricted family list: start at `index`, take distinct families cyclically; when the list is
    shorter than `n`, top up with a random distinct family from the full set so every document still gets three."""
    fams: list[str] = []
    for j in range(len(families)):
        f = families[(index + j) % len(families)]
        if f not in fams:
            fams.append(f)
        if len(fams) == n:
            break
    while len(fams) < n:
        f = rng.choice([x for x in FAMILY_IDS if x not in fams]); fams.append(f)
    return fams


def plan_document(index: int, seed: str = "decision-p2", families: list[str] | None = None, tag: str | None = None) -> dict:
    """Deterministic plan for document `index`: domain, kind, families, types, unknown flags, state shape, briefs.

    `families` restricts the plan to those families (round-robin by index); `tag` prefixes the doc id so a top-up batch never
    collides with an earlier batch and is stored on the plan as `batch`."""
    rng = random.Random(f"{seed}\0{tag or ''}\0{index}")
    d = DOMAINS[index % len(DOMAINS)]
    fams = restricted_families(families, index, rng) if families else pick_families(rng)
    types = [forced_type(f) or ty for f, ty in zip(fams, pick_types(rng))]; unks = [intended_unknown(rng, f) for f in fams]
    briefs = [rng.choice(BRIEFS[f]) for f in fams]
    did = doc_id(d["id"], index)
    if tag:
        did = f"{tag}-{did}"
    return {"doc_id": did, "index": index, "domain": d["id"], "doc_kind": rng.choice(d["kinds"]), "batch": tag or "p2",
            "families": fams, "types": types, "unknowns": unks, "state_shape": rng.choice(STATE_SHAPES), "briefs": briefs,
            "seed": rng.randrange(10 ** 9)}


def render(plan: dict) -> list[dict]:
    d = get_domain(plan["domain"])
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": writer_prompt(d, plan["doc_kind"], plan["families"], plan["types"], plan["unknowns"],
                                                       plan["state_shape"], plan["briefs"], plan["seed"])}]


def validate(raw: str, plan: dict) -> tuple[dict | None, str | None]:
    """Parse + validate the writer's JSON; enforce that families/types match the plan (the writer may not swap them)."""
    try:
        obj = normalise_writer_output(loads_lenient(raw))
        out = WriterOutput.model_validate(obj)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"
    for q, fam, typ, unk in zip(out.questions, plan["families"], plan["types"], plan["unknowns"]):
        if q.family != fam: return None, f"family mismatch {q.family} != {fam}"
        if q.type != typ: return None, f"type mismatch {q.type} != {typ}"
        if (q.intended is None) != unk: return None, f"unknown flag mismatch for {fam}"
    if plan["state_shape"] == "object" and not isinstance(out.document, dict): return None, "state shape: expected object"
    if plan["state_shape"] == "string" and not isinstance(out.document, str): return None, "state shape: expected string"
    return out.model_dump(), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=6000); ap.add_argument("--thinking", action="store_true", help="let the writer think (slow: ~5k reasoning tokens per document)"); ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=WRITER_MODEL["served"]); ap.add_argument("--seed", default="decision-p2")
    ap.add_argument("--start", type=int, default=0, help="first document index (for a second pass with new indices)")
    ap.add_argument("--families", default="", help="comma-separated family ids to restrict the plan to (round-robin); default: all")
    ap.add_argument("--tag", default="", help="batch tag: prefixes every doc id (no collision with earlier batches) and is stored as `batch`")
    a = ap.parse_args(argv)
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    unknown_fams = [f for f in fams if f not in FAMILIES]
    if unknown_fams:
        ap.error(f"unknown families: {unknown_fams}")
    done = {r["doc_id"] for r in read_jsonl(a.out)}
    plans = [p for p in (plan_document(i, a.seed, fams or None, a.tag or None) for i in range(a.start, a.start + a.docs)) if p["doc_id"] not in done]
    print(f"writer: {len(done)} done, {len(plans)} to generate", flush=True)
    client = ChatClient(a.base_url, a.model)
    rejects = a.out.with_suffix(".rejects.jsonl"); started = time.time(); ok = bad = 0

    def work(plan):
        for attempt in range(2):
            p = dict(plan, seed=plan["seed"] + attempt)
            raw = client.complete(render(p), temperature=a.temperature, max_tokens=a.max_tokens, json_mode=True, extra=None if a.thinking else {"chat_template_kwargs": {"enable_thinking": False}})
            out, err = validate(raw, p)
            if out is not None:
                return {"doc_id": p["doc_id"], "batch": p.get("batch", "p2"), "plan": p,
                        "writer": {"model": a.model, "thinking": bool(a.thinking), **{k: v for k, v in WRITER_MODEL.items() if k != "served"}},
                        "output": out, "attempt": attempt}
            append_jsonl(rejects, {"doc_id": p["doc_id"], "attempt": attempt, "error": err, "raw": raw[:2000]})
        return None

    for plan, result in run_parallel(plans, work, workers=a.workers):
        if isinstance(result, Exception):
            bad += 1; append_jsonl(rejects, {"doc_id": plan["doc_id"], "error": f"exception: {result}"})
        elif result is None:
            bad += 1
        else:
            ok += 1; append_jsonl(a.out, result)
        n = ok + bad
        if n % 50 == 0:
            rate = n / max(time.time() - started, 1e-6)
            print(f"writer: {ok} ok, {bad} rejected, {rate*60:.1f} docs/min, eta {(len(plans)-n)/max(rate,1e-6)/60:.0f} min", flush=True)
    print(f"writer done: {ok} ok, {bad} rejected -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
