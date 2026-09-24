"""Build the phase-2c manifests (docs/phase-2c-plan.md, amendments included).

    python scripts/p2/build_p2c_manifest.py \
        --p2b-records data/decision-p2/teacher/records-p2b.jsonl \
        --relabel data/decision-p2/teacher/answers-p2c-relabel-qwen35.jsonl \
        --p2c-records data/decision-p2/teacher/records-p2c.jsonl \
        --eikos-records data/external/eikos-decisions/records.jsonl \
        --image-replay-fresh data/decision-p2c/image-replay/replay-fresh.jsonl \
        --image-replay-delta data/decision-p2c/image-replay/replay-delta.jsonl

Sources (each row gets `mix`: p2b | p2c | eikos | image_replay):
  p2b           phase-2 + 2b teacher records. With --relabel (a `gen_answer.py --mode distribution` file) each row gets `target_probs` +
                `rationale` by (doc_id, qi) when the soft teacher's argmax equals the existing target. An answerable TRAIN row whose argmax
                differs is dropped (Eikos agreement rule); an unknown-gold row is never dropped (it keeps its hard target); dev/test/
                calibration rows are never dropped; rows without a usable distribution keep their hard target.
  p2c           judge + programmatic records from assemble_p2.py (with --soft-from they already carry target_probs).
  eikos         the filtered Eikos slice; its `target_probs` lists (aligned with `target_probs_keys`) become label dicts, unknown = 0.
  image_replay  pre-sampled image-decision rows (scripts/p2/make_image_replay.py) copied unchanged into partition train:
                --image-replay-delta into decision-p2c.jsonl, --image-replay-fresh into decision-p2c-fresh.jsonl.
Partitions: p2b/p2c rows in --holdout-domains -> `calibration` (all partitions, as build_p2b_manifest.py); dev = p2b dev + p2c dev rows
of the judge families; other p2c dev rows and the Eikos dev split -> `extra_dev` (never trained on, not used for selection); test kept.
Every `target_probs` carries every label of its question and an `unknown` key (0 when the teacher gave none).

--cap-per-family (default 2,500) caps the TRAIN rows of each templated family (family or batch matching --templated, default
`eikos.rules_*,prog-*`), keeping the rows with the smallest sha256(seed, id). Why 2,500: templated rows share boilerplate by
construction, so past a few thousand rows a family adds template repetition rather than new reasoning, and it starts to steer the
readout (Eikos caps its templated sources for the same reason, docs/eikos-analysis.md §7). 2,500 is about 1.8x our largest teacher
family (multi_step_lookup, ~1,400 rows) and about 8% of the ~30k-row text mix, so no single template family can dominate, while the
programmatic generators (3,000 docs x 3 questions per family before agreement) are the ones the cap actually binds on.

Gates: every record is contract-validated (decision_data.expand_fields + render, soft targets included) and must keep `unknown` among
its rendered choices, or it is dropped and counted; duplicate ids abort; every row of every output is checked with the JevBench 8-gram
lint used by assemble_p2.py (state + question shared 8-grams >= 2) and ANY hit aborts the build.
Outputs: decision-p2c.jsonl (delta lane), decision-p2c-fresh.jsonl (fresh lane), decision-p2c-judge-dev.jsonl (judge-family dev rows,
partition dev, for --dev2-version) and a JSON report (also printed).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import contamination_count, jevbench_public_files, project_probs, reference_ngrams, value_to_label  # noqa: E402
from assemble_p2 import soft_label  # noqa: E402

DEFAULT_HOLDOUT = "telecom,hospitality,nonprofit_grants"
JUDGE_FAMILIES = ("judge_pairwise", "judge_rubric_score")
TEMPLATED = "eikos.rules_*,prog-*"


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [json.loads(l) for l in path.open() if l.strip()]


def h(seed, rid) -> str:
    return hashlib.sha256(f"{seed}\0{rid}".encode()).hexdigest()


def doc_qi(rec: dict) -> tuple[str, int]:
    return rec["source_group"], int(rec["id"].rsplit(":", 1)[1])


def relabel_p2b(rows: list[dict], dist_rows: list[dict], holdout: tuple, counts: collections.Counter) -> list[dict]:
    """Attach target_probs/rationale from distribution rows; drop answerable train rows whose argmax disagrees (see module doc)."""
    by_key: dict = {}
    for r in dist_rows:
        if r.get("mode") != "distribution":
            continue
        k = (r["doc_id"], r["qi"]); prev = by_key.get(k)
        if prev is None or r.get("probs") and not r.get("parse_error") or not prev.get("probs") or prev.get("parse_error"):
            by_key[k] = r
    out = []
    for rec in rows:
        field = rec["request"]["fields"][0]; row = by_key.get(doc_qi(rec))
        status, probs, unk = soft_label(field, rec["target"], row)
        trained = rec["partition"] == "train" and rec.get("domain") not in holdout
        counts[f"relabel_{status}"] += 1
        if status == "disagree":
            if rec["target"] is not None and trained:
                counts["relabel_dropped_answerable_train"] += 1; continue
            counts["relabel_disagree_kept_hard_" + ("unknown_gold" if rec["target"] is None else "eval_row")] += 1
        if status == "agree":
            rec = dict(rec, target_probs=probs)
            if isinstance(row.get("rationale"), str) and row["rationale"].strip():
                rec["rationale"] = row["rationale"].strip()
            rec["provenance"] = dict(rec.get("provenance") or {}, soft_teacher={
                "answerer": row.get("answerer"), "repo": row.get("repo"), "revision": row.get("revision"), "status": "agree", "unknown_mass": unk,
                **({"unknown_mass_high": unk > 0.2} if rec["target"] is not None else {})})
        out.append(rec)
    return out


def with_unknown_key(rec: dict) -> dict:
    """Normalise `target_probs` to a label dict over the question's labels + unknown (lists with `target_probs_keys` included)."""
    tp = rec.get("target_probs")
    if tp is None:
        return rec
    rec = dict(rec); field = rec["request"]["fields"][0]
    if isinstance(tp, list):
        keys = rec.pop("target_probs_keys")
        if len(keys) != len(tp):
            raise ValueError(f"{rec['id']}: target_probs / target_probs_keys length mismatch")
        tp = {value_to_label(k): v for k, v in zip(keys, tp)}
    probs, err = project_probs(tp, field, strict=True)
    if err:
        raise ValueError(f"{rec['id']}: {err}")
    rec["target_probs"] = probs
    return rec


def cap_families(rows: list[dict], cap: int, patterns: list[str], seed: str, counts: collections.Counter) -> list[dict]:
    def templated(r):
        return any(fnmatch(str(r.get("family", "")), p) or fnmatch(str(r.get("batch", "")), p) for p in patterns)
    groups = collections.defaultdict(list)
    for r in rows:
        if r["partition"] == "train" and templated(r):
            groups[r.get("family")].append(r)
    drop = set()
    for fam, g in groups.items():
        if len(g) > cap:
            keep = sorted(g, key=lambda r: h(f"{seed}-cap", r["id"]))[:cap]
            kept_ids = {r["id"] for r in keep}
            drop |= {r["id"] for r in g if r["id"] not in kept_ids}
            counts[f"capped:{fam}"] = len(g) - cap
    return [r for r in rows if r["id"] not in drop]


def validate(rows: list[dict], counts: collections.Counter) -> list[dict]:
    from decision_data import expand_fields, render
    from vision_decision.contracts import UNKNOWN
    out = []
    for r in rows:
        try:
            for item in expand_fields(r):
                _, choices, _, _ = render(item, soft_targets="target_probs" in item)
                if not any(v == UNKNOWN for v, _ in choices):
                    raise ValueError("no unknown choice")
            out.append(r)
        except Exception as exc:
            counts[f"contract_dropped:{r.get('mix')}"] += 1; counts["contract_dropped"] += 1
            counts[f"contract_reason:{str(exc).splitlines()[0][:60]}"] += 1
    return out


def lint(rows: list[dict], reference: set, min_shared: int = 2) -> list[tuple[str, int]]:
    hits = []
    for r in rows:
        state = r["request"].get("state", "")
        text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
        n = contamination_count(text, reference) + sum(contamination_count(f.get("question", ""), reference) for f in r["request"]["fields"])
        if n >= min_shared:
            hits.append((r["id"], n))
    return hits


def items_unknown(r: dict) -> tuple[int, int]:
    """(expanded items, items whose gold is unknown)."""
    if r.get("targets"):
        vals = list(r["targets"].values())
        return len(vals), sum(v is None for v in vals)
    return 1, int(r.get("target") is None)


def build(p2b, p2c, eikos, replay_delta, replay_fresh, dist_rows=None, holdout=(), cap=2500, templated=TEMPLATED, seed="p2c",
          judge_families=JUDGE_FAMILIES, reference=frozenset(), min_shared=2, do_validate=True):
    counts = collections.Counter()
    if dist_rows is not None:
        p2b = relabel_p2b(p2b, dist_rows, holdout, counts)
    text = []
    for r in p2b:
        r = with_unknown_key(dict(r, mix="p2b"))
        if r.get("domain") in holdout: r["partition"] = "calibration"
        text.append(r)
    for r in p2c:
        r = with_unknown_key(dict(r, mix="p2c"))
        if r.get("domain") in holdout: r["partition"] = "calibration"
        elif r["partition"] == "dev" and r.get("family") not in judge_families: r["partition"] = "extra_dev"
        text.append(r)
    for r in eikos:
        r = with_unknown_key(dict(r, mix="eikos"))
        if r["partition"] != "train": r["partition"] = "extra_dev"
        text.append(r)
    text = cap_families(text, cap, [p for p in templated.split(",") if p], seed, counts)
    replay_d = [dict(r, mix="image_replay", partition="train") for r in replay_delta]
    replay_f = [dict(r, mix="image_replay", partition="train") for r in replay_fresh]
    if do_validate:
        text = validate(text, counts); replay_d = validate(replay_d, counts); replay_f = validate(replay_f, counts)
    delta, fresh = text + replay_d, text + replay_f
    for name, rows in (("decision-p2c", delta), ("decision-p2c-fresh", fresh)):
        dup = [k for k, v in collections.Counter(r["id"] for r in rows).items() if v > 1]
        if dup:
            raise SystemExit(f"duplicate ids in {name}: {dup[:5]}")
    if reference:
        hits = lint({r["id"]: r for r in delta + fresh}.values(), reference, min_shared)
        if hits:
            raise SystemExit(f"JevBench 8-gram lint: {len(hits)} manifest rows share >= {min_shared} 8-grams with the public files, e.g. {hits[:10]}")
    judge_dev = [r for r in text if r["partition"] == "dev" and r["mix"] == "p2c" and r.get("family") in judge_families]
    return delta, fresh, judge_dev, counts


def report(delta, fresh, judge_dev, counts, min_unknown_share=0.15) -> dict:
    def summary(rows):
        train = [r for r in rows if r["partition"] == "train"]; text_train = [r for r in train if r["mix"] != "image_replay"]
        n_items = sum(items_unknown(r)[0] for r in train); n_unk = sum(items_unknown(r)[1] for r in train)
        by_mix = {}
        for m in sorted({r["mix"] for r in rows}):
            mr = [r for r in rows if r["mix"] == m]; mt = [r for r in mr if r["partition"] == "train"]
            it = sum(items_unknown(r)[0] for r in mt); iu = sum(items_unknown(r)[1] for r in mt)
            by_mix[m] = {"rows": len(mr), "partitions": dict(collections.Counter(r["partition"] for r in mr)),
                         "unknown_gold_rows": sum(items_unknown(r)[1] > 0 for r in mr), "train_unknown_gold_items": iu,
                         "train_unknown_share": round(iu / it, 4) if it else None,
                         "train_with_target_probs": round(sum("target_probs" in r for r in mt) / len(mt), 4) if mt else None,
                         "train_with_rationale": round(sum(bool(r.get("rationale")) for r in mt) / len(mt), 4) if mt else None}
        return {"rows": len(rows), "partitions": dict(collections.Counter(r["partition"] for r in rows)),
                "train_rows": len(train), "image_replay_train_rows": len(train) - len(text_train),
                "image_replay_share_of_train": round((len(train) - len(text_train)) / len(train), 4) if train else None,
                "train_unknown_share": round(n_unk / n_items, 4) if n_items else None, "train_unknown_gold_items": n_unk,
                "text_train_with_target_probs": round(sum("target_probs" in r for r in text_train) / len(text_train), 4) if text_train else None,
                "text_train_with_rationale": round(sum(bool(r.get("rationale")) for r in text_train) / len(text_train), 4) if text_train else None,
                "by_mix": by_mix}
    text = [r for r in delta if r["mix"] != "image_replay"]
    rep = {"decision-p2c": summary(delta), "decision-p2c-fresh": summary(fresh),
           "judge_dev": {"rows": len(judge_dev), "by_family": dict(collections.Counter(r["family"] for r in judge_dev))},
           "text_train_by_family": dict(sorted(collections.Counter(r.get("family") for r in text if r["partition"] == "train").items())),
           "text_by_type": dict(collections.Counter(r["request"]["fields"][0]["type"] for r in text)),
           "text_by_mix_partition": {f"{m}/{p}": n for (m, p), n in sorted(collections.Counter((r["mix"], r["partition"]) for r in text).items())},
           "counts": dict(sorted(counts.items())), "min_unknown_share": min_unknown_share}
    share = rep["decision-p2c"]["train_unknown_share"]
    rep["unknown_share_warning"] = share is not None and share < min_unknown_share
    return rep


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    M = ROOT / "data/manifests"
    ap.add_argument("--p2b-records", type=Path, default=ROOT / "data/decision-p2/teacher/records-p2b.jsonl")
    ap.add_argument("--relabel", type=Path, nargs="*", default=None, help="distribution-mode answer files (gen_answer.py --mode distribution)")
    ap.add_argument("--p2c-records", type=Path, nargs="*", default=[ROOT / "data/decision-p2/teacher/records-p2c.jsonl"])
    ap.add_argument("--eikos-records", type=Path, default=ROOT / "data/external/eikos-decisions/records.jsonl")
    ap.add_argument("--image-replay-fresh", type=Path, default=ROOT / "data/decision-p2c/image-replay/replay-fresh.jsonl")
    ap.add_argument("--image-replay-delta", type=Path, default=ROOT / "data/decision-p2c/image-replay/replay-delta.jsonl")
    ap.add_argument("--cap-per-family", type=int, default=2500); ap.add_argument("--templated", default=TEMPLATED)
    ap.add_argument("--holdout-domains", default=DEFAULT_HOLDOUT); ap.add_argument("--seed", default="p2c")
    ap.add_argument("--judge-families", default=",".join(JUDGE_FAMILIES)); ap.add_argument("--min-unknown-share", type=float, default=0.15)
    ap.add_argument("--reference", type=Path, nargs="*", help="JevBench public files (default: .cache/external/jevbench/datasets/public/*.jsonl)")
    ap.add_argument("--out", type=Path, default=M / "decision-p2c.jsonl"); ap.add_argument("--out-fresh", type=Path, default=M / "decision-p2c-fresh.jsonl")
    ap.add_argument("--out-judge-dev", type=Path, default=M / "decision-p2c-judge-dev.jsonl"); ap.add_argument("--report", type=Path, default=M / "decision-p2c-report.json")
    a = ap.parse_args(argv)
    ref_files = a.reference if a.reference is not None else jevbench_public_files()
    reference = reference_ngrams(ref_files)
    if not reference:
        raise SystemExit("no JevBench reference 8-grams: the lint cannot run (expected .cache/external/jevbench/datasets/public/*.jsonl)")
    dist = [r for p in a.relabel for r in read_jsonl(p)] if a.relabel else None
    hold = tuple(d for d in a.holdout_domains.split(",") if d)
    delta, fresh, judge_dev, counts = build(
        read_jsonl(a.p2b_records), [r for p in a.p2c_records for r in read_jsonl(p)], read_jsonl(a.eikos_records),
        read_jsonl(a.image_replay_delta), read_jsonl(a.image_replay_fresh), dist, hold, a.cap_per_family, a.templated, a.seed,
        tuple(f for f in a.judge_families.split(",") if f), reference)
    for path, rows in ((a.out, delta), (a.out_fresh, fresh), (a.out_judge_dev, judge_dev)):
        write(path, rows)
    rep = report(delta, fresh, judge_dev, counts, a.min_unknown_share)
    rep["inputs"] = {k: str(v) for k, v in vars(a).items() if k not in ("reference",)}
    rep["jevbench_lint"] = {"files": [str(p) for p in ref_files], "ngrams": len(reference), "rule": "state + question shared 8-grams >= 2 -> abort", "hits": 0}
    rep["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.out, a.out_fresh, a.out_judge_dev)}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    if rep["unknown_share_warning"]:
        print(f"WARNING: combined decision-p2c train unknown share {rep['decision-p2c']['train_unknown_share']:.1%} below {a.min_unknown_share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
