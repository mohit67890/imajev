"""Our own labelled TEXT decisions -> phase-3 candidates (docs/phase-3-plan.md, Stage 0 source C).

Stage-1 mining runs the shipped imajev-4b over these to find the items it still gets wrong. Sources (text rows only; the image
rows of our pools are the image branch's job):

  p2-teacher   data/manifests/decision-p2.jsonl + decision-p2b.jsonl, source p2_teacher: phase 2/2b hard typed questions over long
               synthetic business documents (written by Qwen3.6-27B, kept only when 2-3 open answerers agreed).
               p2b's 30% replay rows are the p2 rows again (same ids) and are read once.
  p2-human     the same two manifests, the public "human half" (ARC, CSQA, CosmosQA, DROP, GSM8K, QASC, StrategyQA, TruthfulQA,
               WANLI; MMLU-Pro sits only in test and never appears). StrategyQA items already in source B
               (data/p3/candidates/B-strategyqa.jsonl) are skipped so one upstream item is not in the pool twice.
  eikos        data/external/eikos-decisions/records.jsonl, the phase-2c CC-BY strict slice (program-labelled; per-row CC-BY-4.0 or
               MIT, attribution kept in provenance).
  p2c-prog     phase-2c programmatic questions, regenerated exactly as the pod did (`scripts/p2/gen_programmatic.py --docs 3000
               --seed 7`, sha256 checked against reports/benchmarks/decisionbench/contamination-check.md) and partitioned exactly as
               `assemble_p2.py` did (stable_partition(doc_id, seed="decision-p2")). Gold is the generator's computed answer.
  v2.1         text-only rows (no images) of data/manifests/decision-v2.1-4b.jsonl, multi-question requests expanded per field
               (decision_data.expand_fields semantics), sampled for family balance. banking77, CLINC150, civil_comments and ESCI
               are excluded: DecisionBench draws on the same upstream items (contamination-check.md).

Split rule: only rows whose manifest partition is `train`. Rows of the phase-2 hold-out domains (telecom, hospitality,
nonprofit_grants) are also excluded everywhere: phase 2b/2c moved them to `calibration`.

Mapping (manifest record -> candidate):
  boolean -> noul (gold true/false; yes/no descriptions kept on the field); choice -> choice (key = the option `value`, so the
  gold is the record's `target` unchanged); ordinal -> score (levels re-indexed 0..n-1 in their listed order, gold = the index of
  the target's level; the original level values are kept in provenance.level_values). target None -> gold null with
  unknown_reason = the record's abstention cause. A `target_probs` / `target_distribution` is kept verbatim in
  provenance.target_probs (keyed by the ORIGINAL option / level values). gold_kind is "dataset" except p2c-prog ("constructed":
  the answer is computed by the generator).

Selection (--target, default 25,000): each source has a quota; inside a source, rows are allocated across families by weighted
water-filling (harder families weigh more: long documents, judge, numeric/temporal, multi-hop, rules), and inside a family
unknown-gold rows come first (up to --unknown-cap of the family's share), the rest in a stable hash order.

    .venv/bin/python scripts/p3/convert_own_text.py [--target 25000] [--out data/p3/candidates/C-text.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/p2"))
from candidate import UNKNOWN_REASONS, validate, write  # noqa: E402
from convert_common import approx_tokens  # noqa: E402

OUT = ROOT / "data/p3/candidates/C-text.jsonl"
MANIFESTS = ROOT / "data/manifests"
EIKOS = ROOT / "data/external/eikos-decisions/records.jsonl"
EIKOS_REPORT = ROOT / "data/external/eikos-decisions/report.json"
PROG_CACHE = ROOT / "data/p3/raw/own-text/writer-prog-seed7.jsonl"
PROG_SHA256 = "b7e27800cfd3e316ba1c577fb1e6e3d08ef4993878c85a05eef83ffc563bd6c8"
PROG_PARTITION_SEED = "decision-p2"  # assemble_p2.py default; the pod ran it without --seed
B_STRATEGYQA = ROOT / "data/p3/candidates/B-strategyqa.jsonl"
HOLDOUT_DOMAINS = frozenset({"telecom", "hospitality", "nonprofit_grants"})
V21_EXCLUDED_SOURCES = frozenset({"banking77", "clinc150", "civil_comments", "esci"})
TYPE_MAP = {"boolean": "noul", "choice": "choice", "ordinal": "score"}
CAUSE_MAP = {c: c for c in UNKNOWN_REASONS}

# Source quotas for the default 25k target (scaled proportionally for another --target). p2-teacher is our hardest own data
# (long documents, rules, judge, numeric/temporal) and is taken whole when it fits.
QUOTAS = {"p2-teacher": 11500, "eikos": 4500, "p2c-prog": 3000, "p2-human": 2000, "v2.1": 4000}

# Family weights for the water-filling allocation inside a source (default 1.0).
HARD = 2.0
FAMILY_WEIGHTS = {
    # p2 / p2b teacher
    "judge_answer": HARD, "rubric": HARD, "multi_step_lookup": HARD, "numerical_reconciliation": HARD, "temporal_ordering": HARD,
    "date_number_trap": HARD, "rule_precedence": HARD, "policy_exception": HARD, "insufficient_evidence": HARD, "contradiction": HARD,
    "probability_estimate": HARD, "misleading_note": 1.5, "tradeoff": 1.5, "ambiguity": 1.5,
    # human half
    "implicit_multihop": HARD, "numerical_reasoning": HARD, "arithmetic_word_problem": HARD, "fact_combination": HARD,
    "truthfulness_trap": 1.5, "nli": 0.7, "commonsense": 0.7, "science_reasoning": 0.7, "reading_commonsense": 0.7,
    # eikos
    "eikos.judge": 2.5, "eikos.temporal_numeric": 2.5, "eikos.finance_exact": HARD, "eikos.trade_exact": HARD,
    "eikos.rules_book": 1.0, "eikos.rules_permit": 1.0, "eikos.rules_route": 1.0, "eikos.rules_tier": 1.0, "eikos.rules_score": 1.0,
    # v2.1 text
    "multi_question": HARD, "adequacy_of_answer": HARD, "numeric_comparison": HARD, "squad2": HARD, "fever": HARD,
    "response_rubric": HARD, "boolq": 1.5, "contains_claim": 1.5, "stance": 1.5, "next_action_routing": 1.5, "severity_urgency": 1.5,
    "emotion": 0.5, "intent_routing": 0.5, "moderation_taxonomy": 0.5, "content_safety": 0.7, "snli": 0.7, "abo_structured": 0.5,
}
HARD_FAMILIES = {f for f, w in FAMILY_WEIGHTS.items() if w >= HARD}


def h(*parts) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()


def read_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------------------------------------------------ mapping
def map_field(field: dict) -> tuple[dict, dict]:
    """Manifest request field -> (candidate field, extras for provenance: level_values)."""
    t = TYPE_MAP[field["type"]]
    out = {"type": t, "question": field["question"]}
    extra = {}
    if t == "choice":
        out["options"] = []
        for o in field["options"]:
            opt = {"key": o["value"], "text": str(o["value"])}
            if o.get("description"):
                opt["description"] = o["description"]
            out["options"].append(opt)
    elif t == "score":
        out["levels"] = [{"value": i, "description": l.get("description", "")} for i, l in enumerate(field["levels"])]
        extra["level_values"] = [l["value"] for l in field["levels"]]
    else:
        for k in ("yes_description", "no_description"):
            if field.get(k):
                out[k] = field[k]
    return out, extra


def map_gold(field: dict, target, cause) -> tuple[object, str | None, str | None]:
    """(gold, unknown_reason, original cause when it was not one of ours)."""
    if target is None:
        reason = CAUSE_MAP.get(cause)
        return None, reason or "insufficient_evidence", (None if reason else cause)
    if field["type"] == "ordinal":
        vals = [l["value"] for l in field["levels"]]
        return vals.index(target), None, None
    return target, None, None


def probs_dict(tp) -> dict | None:
    if not tp:
        return None
    return {("true" if k is True else "false" if k is False else str(k)): float(v) for k, v in tp.items()}


def difficulty(family: str, state, base: int = 3, batch: str | None = None) -> int:
    d = base + (1 if family in HARD_FAMILIES else 0)
    toks = approx_tokens(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
    if toks > 1500:
        d += 1
    if batch == "p2b-hard":
        d += 1
    return max(1, min(5, d))


def candidate(*, group: str, dataset: str, family: str, record: dict, field: dict, target, cause, licence: str, manifest: str,
              base_difficulty: int, gold_kind: str = "dataset", target_probs=None, extra_prov: dict | None = None) -> dict:
    cfield, extra = map_field(field)
    gold, reason, raw_cause = map_gold(field, target, cause)
    state = record["request"]["state"]
    rid = record["id"] if len(record["request"]["fields"]) == 1 else f"{record['id']}:{field['id']}"
    prov = {"licence": licence, "manifest": manifest, "record_id": record["id"], "field_id": field.get("id"),
            "partition": record.get("partition", "train"), "upstream_split": record.get("source_split"),
            "record_source": record.get("source"), "record_family": record.get("family"), "group_id": record.get("source_group"),
            "own_pool": group}
    if isinstance(record.get("license"), dict):
        prov["source_licence"] = {k: record["license"][k] for k in ("spdx", "evidence", "attribution") if k in record["license"]}
    if record.get("domain"):
        prov["domain"] = record["domain"]
    for k in ("batch", "pseudo_label", "label_route", "template_id"):
        if record.get(k) is not None:
            prov[k] = record[k]
    if target_probs:
        prov["target_probs"] = target_probs
    if raw_cause is not None:
        prov["abstention_cause_original"] = raw_cause
    prov.update(extra)
    prov.update(extra_prov or {})
    return {"id": f"p3-C-{dataset.replace('/', '-')}-{h(manifest, rid)[:14]}", "source": "C", "dataset": dataset,
            "family": family, "difficulty": difficulty(family, state, base_difficulty, record.get("batch")), "state": state,
            "images": [], "field": cfield, "gold": gold, "unknown_reason": reason, "gold_kind": gold_kind, "parent_id": None,
            "provenance": prov}


# ------------------------------------------------------------------------------------------------------------ loaders
def load_p2(manifests=("decision-p2", "decision-p2b"), b_strategyqa: Path = B_STRATEGYQA, why=None) -> tuple[list, list]:
    """(teacher rows, human rows). Train partition only, hold-out domains out, replay duplicates read once."""
    why = why if why is not None else collections.Counter()
    b_ids = {r["provenance"]["upstream_id"] for r in read_jsonl(b_strategyqa)} if Path(b_strategyqa).exists() else set()
    seen, teacher, human = set(), [], []
    for name in manifests:
        for r in read_jsonl(MANIFESTS / f"{name}.jsonl"):
            if r["id"] in seen:
                why[f"{name}:replay_duplicate"] += 1
                continue
            seen.add(r["id"])
            if r["partition"] != "train":
                why[f"{name}:partition_{r['partition']}"] += 1
                continue
            if r.get("images"):
                why[f"{name}:has_images"] += 1
                continue
            if r.get("domain") in HOLDOUT_DOMAINS:
                why[f"{name}:holdout_domain"] += 1
                continue
            field = r["request"]["fields"][0]
            if r["source"] == "p2_teacher":
                teacher.append(candidate(group="p2-teacher", dataset=f"{name}/teacher", family=r["family"], record=r, field=field,
                                         target=r["target"], cause=r["abstention_cause"], licence="own", manifest=name,
                                         base_difficulty=3, extra_prov={"writer": (r.get("provenance") or {}).get("writer", {}).get("model")}))
            else:
                ds = r["source"].removeprefix("p2_")
                uid = r["source_group"].split(":", 1)[1] if ":" in r["source_group"] else r["source_group"]
                if ds == "strategyqa" and uid in b_ids:
                    why[f"{name}:strategyqa_in_B"] += 1
                    continue
                human.append(candidate(group="p2-human", dataset=f"{name}/{ds}", family=r["family"], record=r, field=field,
                                       target=r["target"], cause=r["abstention_cause"], licence=r["license"]["spdx"], manifest=name,
                                       base_difficulty=2, extra_prov={"upstream_dataset": ds, "upstream_id": uid}))
            why[f"{name}:ok"] += 1
    return teacher, human


def load_eikos(path: Path = EIKOS, why=None) -> list:
    why = why if why is not None else collections.Counter()
    attribution = json.loads(EIKOS_REPORT.read_text())["attribution"] if EIKOS_REPORT.exists() else None
    out = []
    for r in read_jsonl(path):
        if r["partition"] != "train":
            why[f"eikos:partition_{r['partition']}"] += 1
            continue
        field = r["request"]["fields"][0]
        tp = dict(zip(r.get("target_probs_keys") or [], r.get("target_probs") or []))
        pv = r.get("provenance") or {}
        extra = {"upstream_dataset": pv.get("dataset"), "upstream_revision": pv.get("revision"), "upstream_id": pv.get("upstream_id"),
                 "upstream_origin": pv.get("upstream"), "label_source": pv.get("label_source"), "eikos_difficulty": pv.get("difficulty"),
                 "attribution": r["license"].get("attribution") or attribution,
                 "url": "https://huggingface.co/datasets/caiovicentino1/eikos-decisions"}
        base = 3 if pv.get("difficulty") == "hard" else 2
        out.append(candidate(group="eikos", dataset="eikos-decisions", family=r["family"], record=r, field=field, target=r["target"],
                             cause=r["abstention_cause"], licence=r["license"]["spdx"], manifest="eikos-decisions/records",
                             base_difficulty=base, target_probs=probs_dict(tp), extra_prov=extra))
        why["eikos:ok"] += 1
    return out


def ensure_prog(path: Path = PROG_CACHE) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/p2/gen_programmatic.py"), "--docs", "3000", "--seed", "7", "--out", str(path)],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != PROG_SHA256:
        raise SystemExit(f"{path}: sha256 {got} != the phase-2c build {PROG_SHA256}")
    return path


def load_prog(path: Path | None = None, why=None) -> list:
    """Writer rows of gen_programmatic.py -> records exactly as assemble_p2.py builds them, then candidates."""
    from p2_common import WQuestion, to_field  # noqa: E402
    from v1_text.common import stable_partition  # noqa: E402
    why = why if why is not None else collections.Counter()
    path = path or ensure_prog()
    out = []
    for w in read_jsonl(path):
        part = stable_partition(w["doc_id"], seed=PROG_PARTITION_SEED)
        dom = w["plan"]["domain"]
        for qi, q in enumerate(w["output"]["questions"]):
            if part != "train":
                why[f"p2c-prog:partition_{part}"] += 1
                continue
            if dom in HOLDOUT_DOMAINS:
                why["p2c-prog:holdout_domain"] += 1
                continue
            field, intended = to_field(WQuestion.model_validate(q), field_id="decision")
            if intended is None and q["intended"] is not None:
                why["p2c-prog:intended_not_an_option"] += 1
                continue
            rec = {"id": f"p2_teacher:{w['doc_id']}:{qi}", "source": "p2_teacher", "source_split": "generated", "source_group": w["doc_id"],
                   "partition": part, "domain": dom, "batch": w.get("batch"), "request": {"state": w["output"]["document"], "fields": [field]}}
            out.append(candidate(group="p2c-prog", dataset="p2c-programmatic", family=q["family"], record=rec, field=field,
                                 target=intended, cause=q.get("unknown_reason"), licence="own", manifest="p2c-programmatic-seed7",
                                 base_difficulty=3, gold_kind="constructed",
                                 extra_prov={"generator": "scripts/p2/gen_programmatic.py --docs 3000 --seed 7", "generator_sha256": PROG_SHA256,
                                             "template": w["plan"]["templates"][qi], "justification": q.get("justification")}))
            why["p2c-prog:ok"] += 1
    return out


def v21_family(record: dict, field: dict) -> str:
    fam = (record.get("field_families") or {}).get(field["id"]) or record["family"]
    return "emotion" if fam.startswith("emotion_") else fam


def load_v21(path: Path | None = None, why=None) -> list:
    why = why if why is not None else collections.Counter()
    path = path or MANIFESTS / "decision-v2.1-4b.jsonl"
    out = []
    with open(path) as fh:
        for line in fh:
            if '"images": [{' in line:  # fast skip of image rows
                continue
            r = json.loads(line)
            if r.get("images"):
                continue
            if r["partition"] != "train":
                why[f"v2.1:partition_{r['partition']}"] += 1
                continue
            if r["source"] in V21_EXCLUDED_SOURCES:
                why[f"v2.1:excluded_{r['source']}"] += 1
                continue
            fields = r["request"]["fields"]
            targets, dists, causes = r.get("targets") or {}, r.get("target_distributions") or {}, r.get("abstention_causes") or {}
            for f in fields:
                if len(fields) == 1 and not targets:
                    tgt, cause, tp = r.get("target"), r.get("abstention_cause"), r.get("target_distribution") or r.get("target_probs")
                else:
                    if f["id"] not in targets:
                        why["v2.1:missing_field_target"] += 1
                        continue
                    tgt, cause, tp = targets[f["id"]], causes.get(f["id"]), dists.get(f["id"])
                if f["type"] not in TYPE_MAP:
                    why[f"v2.1:field_type_{f['type']}"] += 1
                    continue
                if f["type"] == "choice" and not 2 <= len(f.get("options") or []) <= 254:
                    why["v2.1:option_count"] += 1
                    continue
                if tgt is None and cause is None:
                    cause = "insufficient_evidence"
                src = r["source"]
                extra = {"upstream_dataset": src}
                if src not in ("stackexchange", "wikipedia_paragraphs", "support_reviews"):
                    extra["upstream_id"] = r["id"].split(":", 1)[1] if ":" in r["id"] else r["id"]
                out.append(candidate(group="v2.1", dataset=f"v2.1/{src}", family=v21_family(r, f), record=r, field=f, target=tgt,
                                     cause=cause, licence=r["license"]["spdx"], manifest="decision-v2.1-4b", base_difficulty=2,
                                     target_probs=probs_dict(tp), extra_prov=extra))
                why["v2.1:ok"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------ selection
def allocate(counts: dict, quota: int, weights: dict) -> dict:
    """Weighted water-filling: alloc_f = min(n_f, w_f * c), with c the largest level whose total fits the quota."""
    if sum(counts.values()) <= quota:
        return dict(counts)
    lo, hi = 0.0, float(max(counts.values())) / min(weights.get(f, 1.0) for f in counts) + 1
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(min(n, int(weights.get(f, 1.0) * mid)) for f, n in counts.items()) <= quota:
            lo = mid
        else:
            hi = mid
    alloc = {f: min(n, int(weights.get(f, 1.0) * lo)) for f, n in counts.items()}
    # hand out the rounding remainder one row at a time, heaviest unsaturated families first
    rest = quota - sum(alloc.values())
    for f in sorted(counts, key=lambda f: (-weights.get(f, 1.0), f)):
        if rest <= 0:
            break
        if alloc[f] < counts[f]:
            alloc[f] += 1
            rest -= 1
    return alloc


def select(rows: list, quota: int, weights: dict = FAMILY_WEIGHTS, unknown_cap: float = 0.35, seed: str = "p3-C-text") -> list:
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    alloc = allocate({f: len(v) for f, v in by_fam.items()}, quota, weights)
    out = []
    for fam in sorted(by_fam):
        k = alloc[fam]
        ordered = sorted(by_fam[fam], key=lambda r: h(seed, r["id"]))
        unk = [r for r in ordered if r["gold"] is None]
        known = [r for r in ordered if r["gold"] is not None]
        n_unk = min(len(unk), max(int(unknown_cap * k), k - len(known)))
        out += unk[:n_unk] + known[:k - n_unk]
    return out


def build(target: int = 25000, quotas: dict | None = None, v21_path: Path | None = None, prog_path: Path | None = None) -> tuple[list, dict]:
    why = collections.Counter()
    teacher, human = load_p2(why=why)
    pools = {"p2-teacher": teacher, "p2-human": human, "eikos": load_eikos(why=why), "p2c-prog": load_prog(prog_path, why=why),
             "v2.1": load_v21(v21_path, why=why)}
    quotas = dict(quotas or QUOTAS)
    scale = target / sum(QUOTAS.values())
    quotas = {k: int(round(v * scale)) for k, v in quotas.items()}
    # quota a source cannot fill goes to v2.1 (the largest pool)
    spare = sum(max(0, q - len(pools[k])) for k, q in quotas.items() if k != "v2.1")
    quotas["v2.1"] += spare
    rows, seen = [], set()
    for k in ("p2-teacher", "eikos", "p2c-prog", "p2-human", "v2.1"):
        for r in select(pools[k], min(quotas[k], len(pools[k]))):
            if r["id"] in seen:
                raise ValueError(f"duplicate id {r['id']}")
            seen.add(r["id"])
            errs = validate(r)
            if errs:
                raise ValueError(f"{r['id']}: {errs}")
            rows.append(r)
    stats = {"available": {k: len(v) for k, v in pools.items()}, "quotas": quotas, "selected": dict(collections.Counter(r["provenance"]["own_pool"] for r in rows)),
             "reasons": dict(sorted(why.items()))}
    return rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--target", type=int, default=25000)
    for k in QUOTAS:
        ap.add_argument(f"--quota-{k.replace('.', '').replace('-', '_')}", type=int, default=None, help=f"override the {k} quota (at the 25k scale)")
    ap.add_argument("--stats", type=Path, default=None, help="write the selection stats JSON here")
    a = ap.parse_args(argv)
    quotas = dict(QUOTAS)
    for k in QUOTAS:
        v = getattr(a, f"quota_{k.replace('.', '').replace('-', '_')}")
        if v is not None:
            quotas[k] = v
    rows, stats = build(a.target, quotas)
    stats["written"] = write(a.out, rows)
    if a.stats:
        a.stats.parent.mkdir(parents=True, exist_ok=True)
        a.stats.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
