"""Filter the public `caiovicentino1/eikos-decisions` dataset (CC-BY-4.0) into an attributed imajev training slice.

    HF_HOME=.cache/huggingface .venv/bin/python scripts/p2/filter_eikos.py            # default --teacher-policy strict
    .venv/bin/python scripts/p2/filter_eikos.py --teacher-policy writer-only           # see docs/eikos-decisions-usage.md

Deterministic: the snapshot is pinned to REVISION and verified against the dataset's MANIFEST.sha256; rows are processed in file
order; no randomness. Outputs (all under data/external/eikos-decisions/):

    train.jsonl, validation.jsonl   filtered upstream rows, unchanged except for one added `_imajev` provenance object
    heldout.jsonl                   the upstream `core/heldout` split, byte-identical (evaluation only; never training)
    records.jsonl                   the survivors in imajev's decision record schema (partition train / dev)
    long_context-*.jsonl, views_pt_en-train.jsonl, records-long_context.jsonl, records-views_pt_en.jsonl
                                    only when a policy lets rows of those configs survive (none do under `strict`)
    licenses/                       licence evidence + receipts consumed by v1_text.common.verified_license
    report.json                     counts per filter step, per family / teacher / language / licence, plus what-if counts

Filters, applied in this order to `core/train` and `core/validation` (and to long_context / views_pt_en):
  1. split       `heldout` rows never reach a training file; they are copied untouched to heldout.jsonl.
  2. heldout_by_authors   Spanish rows and rows with empty `used_in` are dropped (the authors hold these out; today all live in
                 `heldout`, so this is a guard).
  3. licence     per-row SPDX from `upstream` (null -> the dataset's own CC-BY-4.0) must be in COMMERCIAL_ALLOWLIST
                 (scripts/v1_text/common.py). FinEntity rows are ODC-BY-1.0, which is not on the list -> dropped.
  4. teacher     per-row provenance of `expected` / `target_probs` / `teacher_probs` / `rationale`; see TEACHER_POLICIES. Rows whose
                 provenance cannot be derived from the recorded fields are dropped as `unknown`.
  5. jevbench    JevBench public 8-gram lint, identical to scripts/p2/assemble_p2.py: shared word 8-grams of the state plus shared
                 8-grams of the question (`instructions`) against .cache/external/jevbench/datasets/public/*.jsonl; >= 2 -> drop.
  6. heldout_overlap   rows whose normalised state+question exactly equals a heldout row are dropped, so heldout.jsonl stays a
                 clean evaluation set.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, hashlib, json, os, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2_common import ROOT, contamination_count, jevbench_public_files, reference_ngrams
from v1_text.common import COMMERCIAL_ALLOWLIST, verified_license, write_jsonl

REPO = "caiovicentino1/eikos-decisions"
REVISION = "26d9a680efdf5c05a0aa12d97231b0cc70898392"  # pinned 2026-09-24
SOURCE = "eikos_decisions"
OUT = ROOT / "data/external/eikos-decisions"
ATTRIBUTION = ("Contains data from \"Eikos Decisions\" by Caio Vicentino (Hugging Face: caiovicentino1), "
               "https://huggingface.co/datasets/caiovicentino1/eikos-decisions (revision " + REVISION[:12] + "), licensed CC BY 4.0 "
               "(https://creativecommons.org/licenses/by/4.0/). Rows were filtered and converted by the imajev project; changes: rows removed, "
               "fields renamed into imajev's record schema.")
UPSTREAM_SPDX = {None: "CC-BY-4.0", "TAT-QA (CC BY 4.0)": "CC-BY-4.0", "GSM8K train (MIT)": "MIT", "FinEntity (ODC-BY 1.0)": "ODC-BY-1.0"}
UPSTREAM_CREDIT = {
    "TAT-QA (CC BY 4.0)": "Zhu et al. (2021), TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
    "GSM8K train (MIT)": "Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems (GSM8K, train split)",
    "FinEntity (ODC-BY 1.0)": "Tang et al. (2023), FinEntity: Entity-level Sentiment Classification for Financial Texts"}
# Models named on the dataset card. GLM-5.3-Flash has MIT open weights (zai-org/GLM-5.3-Flash) and the card says the authors ran it on
# their own infrastructure, but the phase-2c rule (docs/phase-2c-plan.md) treats it as an API teacher, so `strict` excludes it.
GLM, QWEN_WRITER = "GLM-5.3-Flash", "Qwen3.8-27B"
TEACHER_POLICIES = {
    # nothing GLM-5.3-Flash wrote, labelled, scored or selected
    "strict": {"disallowed": {GLM}, "strip_teacher": False},
    # also keep open-model-written items, but drop GLM's probabilities and use the writer's own label (GLM still *selected* them)
    "writer-only": {"disallowed": {GLM}, "strip_teacher": True},
    # GLM-5.3-Flash accepted as an open-weight MIT model run by the authors
    "open-weights": {"disallowed": set(), "strip_teacher": False},
}


# ------------------------------------------------------------------------------------------------- snapshot
def snapshot(revision: str = REVISION) -> Path:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(REPO, repo_type="dataset", revision=revision))
    for line in (path / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        if hashlib.sha256((path / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{rel}: sha256 does not match the dataset's MANIFEST.sha256")
    return path


def read(path: Path) -> list[dict]:
    # split("\n"), not splitlines(): some states contain U+2028, which splitlines() treats as a line break
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()] if path.is_file() else []


# ------------------------------------------------------------------------------------------------- provenance
def provenance(row: dict) -> dict:
    """Who produced this row's text, gold label, soft target and rationale, from the recorded fields only.

    Returns {"teacher", "writer", "target_from", "models", "known"}; `models` is every model that touched the row's content or labels.
    """
    writer, ls, tp = row.get("item_writer"), row.get("label_source"), row.get("teacher_probs")
    p = {"writer": writer, "label_source": ls, "teacher": None, "target_from": None, "models": set(), "known": True}
    if writer == "program" and ls in ("program", "exact_probability") and tp is None:
        p.update(teacher="program", target_from="program (exact answer, label-smoothed)" if ls == "program" else "program (exact distribution)")
        if row.get("upstream") == "GSM8K train (MIT)":
            p["models"].add("Qwen3.5-0.8B")  # sampled candidate solutions (Apache-2.0); the label is graded exactly against GSM8K
    elif writer == "FinEntity (human annotators)" and ls == "human" and tp is None:
        p.update(teacher="human (FinEntity)", target_from="human label, 0.85 smoothed")
    elif writer in (GLM, QWEN_WRITER) and ls in ("teacher", "writer_exact_probs") and tp is not None:
        # every generated item was blind-labelled by GLM-5.3-Flash and kept only if GLM's argmax matched the writer's gold
        p.update(teacher=GLM, target_from=f"{GLM} teacher_probs" if ls == "teacher" else f"{writer} exact distribution")
        p["models"] |= {writer, GLM}
    else:
        p.update(teacher="unknown", known=False)
    if row.get("_translator"):
        p["models"].add(QWEN_WRITER)  # views_pt_en: Qwen3.8-27B translated the state/question/options
    return p


def teacher_decision(row: dict, prov: dict, policy: dict) -> tuple[bool, str]:
    if not prov["known"]:
        return False, "unknown_provenance"
    bad = prov["models"] & policy["disallowed"]
    if not bad:
        return True, "ok"
    if str(row.get("source", "")).startswith("long"):
        return False, "long_context_distractors:" + ",".join(sorted(policy["disallowed"]))  # other items' states (any writer) are embedded
    if policy["strip_teacher"] and row.get("item_writer") not in policy["disallowed"]:
        return True, "stripped_teacher"  # writer-authored label survives; GLM's distribution is discarded
    return False, "disallowed_model:" + ",".join(sorted(bad))


# ------------------------------------------------------------------------------------------------- lint helpers
def norm_key(state: str, question: str) -> str:
    return hashlib.sha256(" ".join(re.findall(r"[a-z0-9]+", f"{state}\n{question}".lower())).encode()).hexdigest()


def jevbench_hits(row: dict, reference: set) -> int:  # same sum as assemble_p2.build_records: document hits + question hits
    return contamination_count(row["state"], reference) + contamination_count(row["instructions"], reference)


# ------------------------------------------------------------------------------------------------- conversion
def license_objects(directory: Path, card: Path) -> dict:
    """Evidence (the dataset card's licence section) + receipt per SPDX we keep, in the form verified_license checks."""
    directory.mkdir(parents=True, exist_ok=True); text = card.read_text()
    section = text[text.index("## Licenses and attribution"):text.index("## Citation")].strip()
    out = {}
    for spdx in sorted(set(UPSTREAM_SPDX.values()) & COMMERCIAL_ALLOWLIST):
        ev = directory / f"EIKOS-DECISIONS-{spdx}.md"
        ev.write_text("\n".join([f"# eikos-decisions rows licensed {spdx}: licence evidence", "",
                                 f"Source: https://huggingface.co/datasets/{REPO} (revision {REVISION}). Dataset licence: CC-BY-4.0.",
                                 f"Rows filed under {spdx}: " + ", ".join(k or "rows with no `upstream` (the dataset's own licence)"
                                                                          for k, v in UPSTREAM_SPDX.items() if v == spdx) + ".", "",
                                 "Attribution required by CC BY 4.0 (reproduce in model cards and release notes):", "", ATTRIBUTION, "",
                                 "Verbatim licence section of the dataset card:", "", section, ""]))
        rel = ev.relative_to(ROOT) if ev.is_relative_to(ROOT) else ev  # repo-relative so the audit passes on a pod
        receipt = {"evidence_sha256": hashlib.sha256(ev.read_bytes()).hexdigest(), "spdx": spdx, "commercial_use_reviewed": True,
                   "reviewed_at": dt.date.today().isoformat(), "reviewed_by": "imajev filter_eikos.py (automated; owner to confirm)",
                   "scope": f"{REPO} rows filed under {spdx} that pass scripts/p2/filter_eikos.py",
                   "source_url": f"https://huggingface.co/datasets/{REPO}",
                   "review_note": "Dataset card licence is CC-BY-4.0 with per-row upstream exceptions; allow-listed SPDX only. Attribution required."}
        ev.with_name(ev.name + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        out[spdx] = {**verified_license(rel, spdx), "attribution": ATTRIBUTION}
    return out


def to_record(row: dict, prov: dict, lic: dict, stripped: bool, config: str) -> dict:
    qt, opts, exp = row["question_type"], row["options"], row["expected"]
    probs = None if row.get("target_probs") is None else [float(x) for x in row["target_probs"]]
    if qt == "noul":
        assert [o["label"] for o in opts] == ["yes", "no"], row["id"]
        field = {"id": "decision", "type": "boolean", "question": row["instructions"]}; target = exp == "yes"; keys = [True, False]
    elif qt == "choice":
        field = {"id": "decision", "type": "choice", "question": row["instructions"],
                 "options": [{"value": o["label"], "description": o["description"]} for o in opts]}; target = exp; keys = [o["label"] for o in opts]
    elif qt == "score":
        field = {"id": "decision", "type": "ordinal", "question": row["instructions"],
                 "levels": [{"value": int(o["label"]), "description": o["description"]} for o in opts]}; target = int(exp); keys = [int(o["label"]) for o in opts]
    else:
        raise ValueError(f"{row['id']}: unknown question_type {qt}")
    group = row.get("origin_id") or re.sub(r"-long\d+k$", "", row["id"])  # a view / dossier shares its original item's group
    rid = f"{SOURCE}:{config}:{row['id']}"
    rec = {"id": rid, "source": SOURCE, "source_split": f"{config}/{row['split']}", "source_group": group,
           "partition": "train" if row["split"] == "train" else "dev", "family": f"eikos.{row['family']}", "domain": row.get("topic"),
           "heldout_family": False, "license": lic, "images": [],
           "request": {"schema_version": "1.0", "request_id": rid.replace(":", "-"), "state": row["state"], "fields": [field]},
           "target": target, "abstention_cause": None, "source_answer": exp,
           "pseudo_label": f"eikos-{row['label_source']}" + ("-hard" if stripped else ""), "template_id": f"eikos.{row['source']}",
           "unknown_by_construction": {"decision": False}, "state_variant": "string",
           "rationale": row.get("rationale") or None,
           "provenance": {"dataset": REPO, "revision": REVISION, "config": config, "upstream_id": row["id"], "item_writer": row.get("item_writer"),
                          "label_source": row.get("label_source"), "teacher": prov["teacher"], "target_from": prov["target_from"],
                          "models": sorted(prov["models"]), "upstream": row.get("upstream"), "lang": row.get("lang"),
                          "difficulty": row.get("difficulty"), "format": row.get("format"), "used_in": row.get("used_in")}}
    if probs is not None:
        rec["target_probs"] = probs; rec["target_probs_keys"] = keys  # aligned with the listed options/levels; no unknown mass
    return rec


# ------------------------------------------------------------------------------------------------- pipeline
STEPS = ("heldout_by_authors", "licence", "teacher", "jevbench", "heldout_overlap")


def run_filters(rows: list[dict], policy: dict, reference: set, heldout_keys: set, spdx_ok=COMMERCIAL_ALLOWLIST):
    """Yield (row, prov, stripped) for survivors; return per-step counters via the `stats` dict."""
    stats = {"in": len(rows), "after": {}, "dropped": collections.Counter(), "jevbench_max_hits": 0}
    kept = []
    for row in rows:
        if row.get("lang") == "Spanish" or not row.get("used_in"):
            stats["dropped"]["heldout_by_authors"] += 1; continue
        spdx = UPSTREAM_SPDX.get(row.get("upstream"), "unknown")
        if spdx not in spdx_ok:
            stats["dropped"][f"licence:{spdx}"] += 1; continue
        prov = provenance(row); ok, why = teacher_decision(row, prov, policy)
        if not ok:
            stats["dropped"][f"teacher:{why}"] += 1; continue
        hits = jevbench_hits(row, reference); stats["jevbench_max_hits"] = max(stats["jevbench_max_hits"], hits)
        if hits >= 2:
            stats["dropped"]["jevbench_8gram"] += 1; continue
        if norm_key(row["state"], row["instructions"]) in heldout_keys:
            stats["dropped"]["heldout_overlap"] += 1; continue
        if why == "stripped_teacher":  # discard the disallowed teacher's numbers so no output file can carry them
            prov = {**prov, "teacher": f"none ({row.get('item_writer')} gold; {GLM} probs discarded)",
                    "target_from": f"{row.get('item_writer')} " + ("exact distribution" if row.get("label_source") == "writer_exact_probs" else "gold label (hard)")}
            row = {**row, "teacher_probs": None, "target_probs": row["target_probs"] if row.get("label_source") == "writer_exact_probs" else None}
        kept.append((row, prov, why == "stripped_teacher"))
    n = len(rows)
    for step in STEPS:
        n -= sum(v for k, v in stats["dropped"].items() if k.startswith(step)); stats["after"][step] = n
    stats["dropped"] = dict(stats["dropped"]); return kept, stats


def tables(rows: list[dict], provs: list[dict] | None = None) -> dict:
    t = {k: collections.Counter() for k in ("family", "teacher", "item_writer", "lang", "licence", "label_source", "question_type")}
    for i, r in enumerate(rows):
        p = provs[i] if provs is not None else provenance(r)
        t["family"][r.get("family")] += 1; t["teacher"][p["teacher"]] += 1; t["item_writer"][r.get("item_writer")] += 1
        t["lang"][r.get("lang")] += 1; t["licence"][UPSTREAM_SPDX.get(r.get("upstream"), "unknown")] += 1
        t["label_source"][r.get("label_source")] += 1; t["question_type"][r.get("question_type")] += 1
    return {k: dict(v.most_common()) for k, v in t.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, help="local snapshot dir (default: download the pinned revision via huggingface_hub)")
    ap.add_argument("--out", type=Path, help="output dir (default: data/external/eikos-decisions for strict, data/external/eikos-decisions-<policy> otherwise, so a what-if run never overwrites the strict slice)")
    ap.add_argument("--teacher-policy", choices=sorted(TEACHER_POLICIES), default="strict")
    a = ap.parse_args(argv)
    snap = a.snapshot or snapshot()
    out = a.out or (OUT if a.teacher_policy == "strict" else OUT.with_name(f"{OUT.name}-{a.teacher_policy}")); out.mkdir(parents=True, exist_ok=True)
    policy = TEACHER_POLICIES[a.teacher_policy]
    ref_files = jevbench_public_files(); reference = reference_ngrams(ref_files)
    n_ref_items = sum(1 for p in ref_files for l in p.read_text().splitlines() if l.strip())
    if n_ref_items != 231:
        raise SystemExit(f"expected the 231 JevBench public items, found {n_ref_items} in {[str(p) for p in ref_files]}")

    core = {s: read(snap / f"data/core/{s}.jsonl") for s in ("train", "validation", "heldout")}
    heldout = core["heldout"]
    shutil.copyfile(snap / "data/core/heldout.jsonl", out / "heldout.jsonl")  # byte-identical; evaluation only
    heldout_keys = {norm_key(r["state"], r["instructions"]) for r in heldout}
    by_id = {r["id"]: r for rows in core.values() for r in rows}
    long = {s: read(snap / f"data/long_context/{s}.jsonl") for s in ("train", "validation")}
    views = []
    for v in read(snap / "data/views_pt_en/train.jsonl"):  # a view inherits the original row's labels and provenance
        o = by_id.get(v["id"], {})
        views.append({**{k: o.get(k) for k in ("family", "topic", "difficulty", "format", "source", "upstream", "item_writer", "label_source",
                                               "teacher_probs", "rationale")},
                      **v, "split": "train", "origin_id": v["id"], "id": f"{v['id']}-view-{v['lang'][:2].lower()}",
                      "item_writer": o.get("item_writer") if o else None, "rationale": None})
        views[-1]["_translator"] = v.get("writer")

    licenses = license_objects(out / "licenses", snap / "README.md")
    report = {"dataset": REPO, "revision": REVISION, "teacher_policy": a.teacher_policy, "generated_at": dt.date.today().isoformat(),
              "jevbench_reference": {"files": [str(p.relative_to(ROOT)) for p in ref_files], "items": n_ref_items, "ngrams": len(reference),
                                     "rule": "state 8-gram hits + question 8-gram hits >= 2 -> drop (as scripts/p2/assemble_p2.py)"},
              "licence_allowlist": sorted(COMMERCIAL_ALLOWLIST), "attribution": ATTRIBUTION, "configs": {}, "what_if": {}}

    all_records = []
    jobs = [("core", "train", core["train"]), ("core", "validation", core["validation"]),
            ("long_context", "train", long["train"]), ("long_context", "validation", long["validation"]), ("views_pt_en", "train", views)]
    for config, split, rows in jobs:
        kept, stats = run_filters(rows, policy, reference, heldout_keys)
        survivors = [r for r, _, _ in kept]
        name = f"{split}.jsonl" if config == "core" else f"{config}-{split}.jsonl"
        if config == "core" or survivors:
            write_jsonl(out / name, [{**{k: v for k, v in r.items() if not k.startswith("_")},
                                      "_imajev": {"teacher": p["teacher"], "target_from": p["target_from"], "models": sorted(p["models"]),
                                                  "licence": UPSTREAM_SPDX.get(r.get("upstream")), "teacher_stripped": s,
                                                  **({"translator": r.get("_translator")} if config == "views_pt_en" else {})}}
                                     for r, p, s in kept])
        recs = [to_record(r, p, licenses[UPSTREAM_SPDX.get(r.get("upstream"))], s, config) for r, p, s in kept]
        if config == "core":
            all_records += recs
        elif recs:
            write_jsonl(out / f"records-{config}.jsonl", recs)
        report["configs"][f"{config}/{split}"] = {**stats, "kept": len(survivors), "input_tables": tables(rows), "kept_tables": tables(survivors, [p for _, p, _ in kept])}
        # what the other policies would keep (no files written)
        report["what_if"][f"{config}/{split}"] = {pol: len(run_filters(rows, TEACHER_POLICIES[pol], reference, heldout_keys)[0])
                                                  for pol in TEACHER_POLICIES if pol != a.teacher_policy}
    write_jsonl(out / "records.jsonl", all_records)
    h_hits = [jevbench_hits(r, reference) for r in heldout]
    report["configs"]["core/heldout"] = {"in": len(heldout), "written_untouched": True, "note": "evaluation only; never in a training output",
                                         "sha256": hashlib.sha256((out / "heldout.jsonl").read_bytes()).hexdigest(),
                                         "jevbench_rows_ge2": sum(h >= 2 for h in h_hits), "input_tables": tables(heldout)}
    report["records"] = {"file": "records.jsonl", "n": len(all_records), "by_partition": dict(collections.Counter(r["partition"] for r in all_records)),
                         "by_field_type": dict(collections.Counter(r["request"]["fields"][0]["type"] for r in all_records)),
                         "with_target_probs": sum("target_probs" in r for r in all_records)}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # ---- summary
    print(f"eikos-decisions @ {REVISION[:12]}  policy={a.teacher_policy}  JevBench ref: {n_ref_items} items, {len(reference)} 8-grams")
    for key, c in report["configs"].items():
        if "after" not in c:
            print(f"  {key:26s} in={c['in']:6d}  -> heldout.jsonl untouched (JevBench >=2 hits: {c['jevbench_rows_ge2']})"); continue
        chain = "  ".join(f"{s}={c['after'][s]}" for s in STEPS)
        print(f"  {key:26s} in={c['in']:6d}  {chain}  kept={c['kept']}   what-if {report['what_if'][key]}")
        if c["dropped"]:
            print(f"  {'':26s} dropped: {c['dropped']}")
    for key in ("core/train", "core/validation"):
        c = report["configs"][key]
        for t in ("teacher", "family", "lang", "licence"):
            inp, kep = c["input_tables"][t], c["kept_tables"][t]
            print(f"\n  {key} by {t} (kept/in): " + ", ".join(f"{k}: {kep.get(k, 0)}/{inp.get(k, 0)}" for k in [*inp, *(k for k in kep if k not in inp)]))
    print(f"\n  records.jsonl: {report['records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
