"""Markdown reports for the phase-3 pool assembly (reports/phase3/pool.md) and the manifest builder (data/p3/manifests/report.md)."""
from __future__ import annotations

import collections
from pathlib import Path


def _t(head: list[str], rows: list[list], align: str | None = None) -> list[str]:
    align = align or "|".join(["---"] + ["---:"] * (len(head) - 1))
    return ["| " + " | ".join(head) + " |", "|" + align + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def write_pool_report(m: dict, q: dict, path: Path) -> None:
    c = m["counts"]
    h = m["heldout"]
    sel = h["selection"]
    L = ["# Phase 3: the assembled candidate pool", "",
         f"Built {m['built']} by `scripts/p3/assemble_pool.py` (helpers `scripts/p3/assembly_*.py`; tests "
         "`tests/test_p3_assembly.py`). Plan: docs/phase-3-plan.md rev 4 (Stage 0 -> Stage 1, streamed mining + teacher). "
         "Numbers: `data/p3/pool/pool-manifest.json`. No model was run.", "",
         "## Result", "",
         f"- **Input:** {_n(c['clean_input_rows'])} clean candidate rows from {len(m['inputs'])} source files, plus "
         f"{_n(c['large_choice_generated'])} generated 255-option items.",
         f"- **Dropped by the trainer/serving contract:** {_n(c['contract_dropped'])} rows that `decision_data.render()` "
         "rejects (reasons below).",
         f"- **Cross-source near-duplicates removed:** {_n(c['dedup_removed'])}.",
         f"- **Held out:** {_n(c['heldout_rows'])} rows in three files (below). Another {_n(c['removed_by_links'])} pool rows "
         "left the pool with their whole group because they still linked to a held-out row.",
         f"- **Mining pool:** {_n(c['shard_rows'])} rows in {len(m['shards'])} family-balanced shards "
         f"(`data/p3/pool/shards/`), plus {_n(c['large_choice_pool_rows'])} 255-option rows in `data/p3/pool/large-choice.jsonl` "
         "(these are not mined; see below).",
         f"- **Leakage gate** (`decontam.py --leakage`, pool + large-choice vs heldout-fresh + heldout-fresh-large-choice): "
         f"**{'PASS' if m['gate']['main']['exit'] == 0 else 'FAIL'}**. `{m['gate']['main']['result']}`",
         f"- **Teacher quotas:** {_n(q['first_solve_total'])} first solves ({_n(q['first_solve_text'])} text, "
         f"{_n(q['first_solve_image'])} image), expected {_n(q['expected_total_calls'])} Azure calls incl. second solves and B/C "
         f"variant checks, under the {_n(q['cap_total_teacher_calls'])} cap (`data/p3/pool/quotas.json`).", "",
         "## Per source file", ""]
    rows = []
    for f, v in m["by_file"].items():
        rows.append([f"`{f}`", _n(v["input"]), _n(v["contract_dropped"]), _n(v["dedup_removed"]), _n(v["heldout_slice"]),
                     _n(v["removed_by_links"]), _n(v["pool"])])
    tot = [sum(v[k] for v in m["by_file"].values())
           for k in ("input", "contract_dropped", "dedup_removed", "heldout_slice", "removed_by_links", "pool")]
    rows.append(["**total**"] + [f"**{_n(x)}**" for x in tot])
    L += _t(["file", "input", "contract drop", "near-dup removed", "held out (B/C slice)", "removed (linked)", "pool"], rows)
    L += ["", "Contract drops (every row is mapped to a trainer record with its own gold and run through "
          "`decision_data.expand_fields` + `render()`, i.e. the pydantic serving contract):", ""]
    L += _t(["file: reason", "rows"], [[k, _n(v)] for k, v in sorted(m["contract_drops"]["by_file_reason"].items(), key=lambda x: -x[1])])
    L += ["", "The long_input generator writes states up to 16k tokens, but the request contract "
          "(`src/vision_decision/contracts.py`) caps a state at 32,768 bytes (~8k tokens). Those rows cannot be served or "
          "trained until the contract is raised or the generator is capped. This is an open decision for the plan's "
          "long-input item (item 3); the rows are listed in pool-manifest.json `contract_drops.ids`."]
    L += ["", "`A-large-choice` is generated here (not a clean input file). Fresh-seed held-out items are new rows, so they do "
          "not appear in this table; the B/C slice rows do.", "",
          "## a. Cross-source near-duplicates", "",
          "Rule (`scripts/p3/assembly_dedup.py`): decontam's normalisation (lower-case `[a-z0-9]+` tokens per string leaf, "
          "distinct 13-grams per unit). Two rows from different source files and different split groups are near-duplicates "
          "when their states share more than 50% of either side's 13-grams (both text or both image rows), or, for text rows "
          "with no state 13-gram at all, when their question+options do. 13-grams in more than 200 rows are template "
          "boilerplate and are ignored for pair finding. The kept copy is the public-licensed one, then source B, then a "
          "checkable gold, then the longer state. Pairs: `data/p3/pool/dedup-pairs.jsonl`.", ""]
    L += _t(["kept file > dropped file (unit)", "pairs"], [[k, _n(v)] for k, v in sorted(m["dedup"]["pairs_by_kind"].items(), key=lambda x: -x[1])])
    L += ["", "Rows removed by dataset:", ""]
    L += _t(["dataset", "rows"], [[k, _n(v)] for k, v in sorted(m["dedup"]["removed_by_dataset"].items(), key=lambda x: -x[1])])
    nc = m["dedup"]["not_counted"]
    L += ["", f"Not counted as duplicates: {_n(nc.get('qopts_only_template', 0))} pairs that share only a question/option "
          f"template over different states (mostly D traps against unrelated A-policy items with the same option set), and "
          f"{_n(nc.get('cross_modality', 0))} text-vs-image pairs.", "",
          "**Known case confirmed:** the Eikos judge rows derived from TAT-QA (C-text) duplicate source B's TAT-QA conversion; "
          "the B copy is kept. SQuAD2 paragraphs from v2.1 (C-text) that sit inside a MuSiQue item's passages are the other "
          "large group; the MuSiQue copy is kept.", ""]
    lc = m["large_choice"]
    L += ["## b. 255-option items (family `large_choice`)", "",
          f"{_n(c['large_choice_generated'])} generated (`scripts/p3/assembly_large_choice.py`, seed `{m['parameters']['lc_seed']}`); "
          f"{_n(lc['rows'])} are in `data/p3/pool/large-choice.jsonl` after the held-out links ({_n(lc['unknown_gold'])} unknown "
          "twins). Each item is a catalogue of exactly 255 records (products, staff, spare parts, rail services, rental flats) "
          "and a request with 3-5 constraints (equalities and numeric bounds). The gold is the one record that meets every "
          "constraint. The generator draws near misses that break exactly one constraint, redraws any accidental second match, "
          "and re-checks the finished catalogue by brute force. `recheck()` re-solves every row from the option descriptions. "
          "An unknown twin changes one constraint so that no record matches; its gold is null with `not_listed`, and its "
          "`parent_id` is the answerable item, so both are in the same split. With 255 options + unknown, a question uses all "
          "256 readout codes.", "",
          "- `scripts/p3/candidate.py` now has `MAX_OPTIONS = 255` (was 254).",
          "- These rows are not in the mining shards: the shipped 255-code model cannot serve 255 options, and the gold is "
          "constructed, so there is nothing to mine or label.",
          "- `scripts/p3/build_manifest.py` adds them to the 256-code lanes and drops every 255-option row from the 255-code lanes.",
          f"- By difficulty: {lc['by_difficulty']}.", ""]
    L += ["## c. Held-out sets", "",
          "Split unit: the **group**, a union-find over `provenance.group_id` / `source_group` / `parent_group_id`, the "
          "`parent_id` chain, GUI screen ids and upstream item keys (a FinQA filing page, a MuSiQue pair, an ALFWorld game, an "
          "ABO photo item). Images are not a group key: ABO photos are shared by thousands of rows across families. Instead, "
          "photo reuse is reported below.", "",
          "| file | rows | what |", "|---|---:|---|",
          f"| `{h['fresh']['path']}` | {_n(h['fresh']['rows'])} | fresh-seed generator items + the B/C group slice "
          f"({', '.join(f'{k} {_n(v)}' for k, v in h['fresh']['by_origin'].items())}) |",
          f"| `{h['gui']['path']}` | {_n(h['gui']['rows'])} | fresh-seed GUI screens (PNGs in `{h['gui']['images_dir']}/`) |",
          f"| `{h['large_choice']['path']}` | {_n(h['large_choice']['rows'])} | fresh-seed 255-option items (256-code lanes only) |",
          f"| `{h['flagged']['path']}` | later | the flagged half: `assemble_pool.py take-flagged --mined ...` after mining |", ""]
    fr = h["fresh"]
    fresh_by = collections.Counter()
    for k, v in fr["by_source_family"].items():
        fresh_by[k] = v
    L += ["### heldout-fresh by source and family", ""]
    L += _t(["source:family", "rows"], [[k, _n(v)] for k, v in sorted(fresh_by.items())])
    L += ["", f"Unknown gold: {_n(fr['unknown_gold'])} of {_n(fr['rows'])}; image rows {_n(fr['image_rows'])}; difficulty "
          f"{fr['by_difficulty']}; types {fr['by_type']}.", "",
          "**How the fresh items were made.** Every generator was run with a seed the pool never used "
          f"(`--tag {m['parameters']['tag']}`, `--policy-seed {m['parameters']['policy_seed']}`; the pool used `r1` and `0`):",
          "`gen_reasoning.py --seed`, `gen_policy.py --seed --count`, `gen_traps.py --in <the fresh policy file> --seed`, and, since "
          "they have no seed flag, `gen_image_joint` and `gen_gui` imported with their `SEED` constant replaced (ids and GUI "
          "screen ids re-prefixed, the GUI PNGs written under `data/p3/pool/heldout-images/gui/`). Each generator was "
          "**oversampled** and the groups that link to the fewest pool rows were kept per family, up to these targets: "
          f"{sel['targets']}. The reason is that some families share templated text between seeds. The multi_label "
          "question wording, the judge rubric and the image_joint policy sentences all share 13-grams with pool items of the "
          "same template, and every such link would force pool rows out.", "",
          "Achieved against the targets (main + GUI + large-choice files):", ""]
    got = collections.Counter()
    for part in ("fresh", "gui", "large_choice"):
        for k, v in h[part]["by_family"].items():
            got[k] += v
    slice_fam = collections.Counter()
    L += _t(["family", "target", "held out"], [[f, _n(t), _n(got.get(f, 0))] for f, t in sorted(sel["targets"].items())])
    L += ["", "(heldout-fresh also holds the B/C slice, whose families are not in this table. The fresh counts can exceed a "
          "target by less than one group.)", "",
          f"**B/C slice.** {_n(sel['bc_candidate_groups'])} candidate groups were drawn at 3x {m['parameters']['bc_share']:.0%} by a "
          "stable hash (a group with any A/D/I row is never drawn). Per dataset, the candidates with the lowest link cost were "
          f"kept up to {m['parameters']['bc_share']:.0%} of the dataset's groups (plumb, FinQA and TAT-QA capped at "
          f"{m['parameters']['scarce_cap']:.0%}): {_n(sel['bc_groups_selected'])} groups, {_n(sel['bc_groups_kept'])} after link "
          f"resolution ({_n(sel['bc_groups_discarded'])} discarded). By dataset:", ""]
    L += _t(["dataset", "rows"], [[k, _n(v)] for k, v in sorted(sel["bc_slice_by_dataset"].items(), key=lambda x: -x[1])])
    L += ["",
          "**Link resolution.** Every held-out row is checked against the pool under the rules of `decontam.py --leakage`: "
          "same id or parent family, a 13-gram near-duplicate of the state or the question+options, the same whole state, or "
          "the same upstream item. A linked pool row leaves the pool with its whole group. A held-out group that would force "
          f"more than {m['parameters']['max_links']} pool rows out is dropped from the held-out set instead. Rounds:", ""]
    L += _t(["round", "links", "by rule", "pool rows scanned", "held-out rows", "held-out groups dropped (total)",
             "pool rows removed (total)"],
            [[r["round"], _n(r["links"]), r["by_reason"] or "-", _n(r["pool_rows_scanned"]), _n(r["heldout_rows"]),
              _n(r.get("heldout_groups_dropped_total", "-")), _n(r.get("pool_rows_removed_total", "-"))] for r in sel["link_rounds"]])
    rb = m["removed_by_links"]
    if rb.get("rows"):
        L += ["", "Pool rows removed for links, by source:family:", ""]
        L += _t(["source:family", "rows"], [[k, _n(v)] for k, v in sorted(rb["by_source_family"].items(), key=lambda x: -x[1])[:25]])
    g = m["gate"]
    L += ["", "### Gate", "",
          f"- **Main:** `{g['main']['result']}` (report `{g['main']['report']}`). The build fails on any link.",
          f"- **GUI:** `{g['gui']['result']}` (report `{g['gui']['report']}`). A GUI row's state is one of five fixed strings "
          "(\"a screenshot of a form screen in a web app\"), so the exact-whole-state rule links every GUI item to every pool GUI "
          "item of the same screen kind, although the screenshots differ. The build therefore requires that GUI links are "
          f"**only** fixed-state links: {'yes' if g['gui']['only_fixed_state_links'] else 'NO'}. Screenshots shared with the "
          f"pool: {g['gui']['shared_screenshots_with_pool']}. Every other rule (13-gram question+options, upstream, id/family) "
          "was resolved as for the main set.",
          f"- **Photo reuse (reported, not gated):** {_n(g['image_reuse']['heldout_fresh_rows_with_a_pool_photo'])} of "
          f"{_n(g['image_reuse']['heldout_fresh_image_rows'])} fresh image rows use a photo that some pool row also uses (in a "
          "different question). The photo pool is finite, and the leakage rule is textual plus the upstream item.", "",
          "### heldout-flagged (hook)", "",
          f"Every pool group gets a stable bucket: `sha256(\"{h['flagged']['bucket_salt']}\", group) < {h['flagged']['bucket_share']}` "
          f"(`pool.heldout_flagged_bucket` on each shard row; {_n(h['flagged']['shard_rows_in_bucket'])} shard rows). A flagged row "
          "in the bucket is the flagged held-out half: the streamer does not send it to the teacher for training labels. "
          "After mining, `assemble_pool.py take-flagged --mined <files>` writes `heldout-flagged.jsonl` (GUI rows in "
          "`heldout-flagged-gui.jsonl`, with the fixed-state exemption). Non-bucket pool rows that still link to it go to "
          "`heldout-flagged-exclude.json`. A flagged group that would exclude more than the cost limit of training rows is "
          "not held out; it stays in the bucket and is never trained on either. The hook then runs the same gate. "
          "`build_manifest.py` drops every bucket group and every excluded id from training. A smoke test on a fake mining "
          "file (8 shards, a third flagged) held out 313 of 431 bucket rows and excluded 503 training rows; the gate "
          "passed.", ""]
    L += ["**Not held out, and why.**",
          "- `multi_label` (fresh): every label question uses the fixed wording `jev_api.multi_label_question()` requires "
          "(training must match serving) over a finite label vocabulary, so a fresh item's question+options equals pool "
          "questions with the same label. The cheapest fresh group would force 141 pool rows out. Evaluate multi-label on "
          "the flagged half or on a dedicated check.",
          "- ALFWorld (B slice): games share room descriptions and observation strings, and the 13-gram rule links groups "
          "across scenes as well. No ALFWorld group links to fewer than the cost limit. Agent actions are covered by the "
          "fresh `agent_action` / `routing_hard` items and the GUI set.",
          "- Image rows whose state is a fixed string (many C-images templates) link to every pool row with the same state "
          "and are skipped by the cost rule, so the B/C image slice holds the rows with item-specific states.", "",
          "## d. Streaming quotas", "", "From `data/p3/pool/quotas.json` (`scripts/p3/assembly_quotas.py`). The quotas are "
          "upper bounds per family on first solves; a family that flags less leaves budget unused.", "",
          "```", q["why"], "```", "",
          f"Totals: first solves {_n(q['first_solve_total'])} (text {_n(q['first_solve_text'])}, image "
          f"{_n(q['first_solve_image'])}); expected second solves {_n(q['expected_second_solves'])}; expected B/C variant "
          f"re-verifications {_n(q['expected_bc_variant_calls'])}; expected total {_n(q['expected_total_calls'])} of the "
          f"{_n(q['cap_total_teacher_calls'])} cap. Parameters: {q['parameters']}.", "",
          "Largest quotas (all families are in quotas.json):", ""]
    fams = sorted(q["families"].items(), key=lambda x: -x[1]["quota"])
    L += _t(["quota key", "weight", "pool rows", "quota", "rate of pool rows", "2nd-solve share"],
            [[f"`{k}`", v["weight"], _n(v["pool_rows"]), _n(v["quota"]), v["rate"], v["second_solve_share"]]
             for k, v in fams[:40]])
    pri = {k: v for k, v in q["families"].items() if v["weight"] == 3.0}
    L += ["", f"The three priority themes (weight 3: long_policy, judge_hard, temporal_numeric and their own-pool "
          f"counterparts) hold {_n(sum(v['quota'] for v in pri.values()))} of the {_n(q['first_solve_text'])} text first solves "
          f"from {_n(sum(v['pool_rows'] for v in pri.values()))} pool rows.", ""]
    p = m["pool"]
    sizes = [s["rows"] for s in m["shards"]]
    L += ["## Shards", "",
          f"{len(m['shards'])} shards, {min(sizes):,}-{max(sizes):,} rows each. Every quota key's rows are dealt round-robin "
          "over the shards in stable-hash order, so each shard holds ~1/64 of every family. Inside a shard the order is a "
          "stable hash, so families are interleaved. sha256 per shard is in pool-manifest.json.", "",
          "### Pool (shards) by source", ""]
    L += _t(["source", "rows"], [[k, _n(v)] for k, v in p["by_source"].items()])
    L += ["", "### Pool by difficulty and type", "",
          f"Difficulty {p['by_difficulty']}; types {p['by_type']}; unknown gold {_n(p['unknown_gold'])}; image rows {_n(p['image_rows'])}.", "",
          "## Verification (2026-09-25)", "",
          "- `tests/test_p3_assembly.py` (20 tests) covers group-level split integrity (also on this pool: no held-out row "
          "shares any group key with a shard or large-choice row), the leakage gate failing on family and 13-gram links and "
          "passing when clean, exhaustive link scanning and resolution, 255-option validity (unique keys, a brute-force-unique "
          "gold, twins with no match, 256-code rendering, identical output across processes), quota sums, trainer records "
          "for every question type through `decision_data.render()`, and an end-to-end `build_manifest.py` run on a fixture "
          "pool (mixture shares within 2 points, no 255-option row in the 255 lane, heldout-flagged bucket groups never "
          "trained, gate PASS). All phase-3 tests pass (`tests/test_p3_*.py`).",
          "- Reproducibility: the whole build was run with `PYTHONHASHSEED=1` and `=2`, and all 311 output files (shards, "
          "held-out files, quotas, GUI PNGs) are byte-identical. `gen_reasoning.py`'s multi_hop states depend on the hash "
          "seed, so the assembler pins `PYTHONHASHSEED=0` for the generator subprocesses. Without the pin, 110 fresh rows "
          "differed between two runs. The generator itself may want a fix.",
          "- `build_manifest.py` dry run on this pool with 60k synthetic teacher labels (`--synthetic 60000`, output in the "
          "session scratchpad, not in data/p3/manifests): 91,451 train rows (limited by hard text), shares exactly "
          "45/20/20/15 in both lanes, 1,500 255-option rows in the 256 lane and 0 in the 255 lane, gate PASS for both lanes "
          "(GUI: fixed-state links only). The real manifests are built after the teacher.", "",
          "## Caveats", "",
          "- The fresh held-out selection prefers low-link groups, so templated families are represented by their "
          "less-templated items. The targets per family are met unless a family ran out of low-link groups (see the table).",
          "- Near-duplicates inside one source file are not removed here; they were already the generators' and converters' concern.",
          "- The quotas assume a 35% flag rate and a 70% keep rate. They are streaming caps, not predictions; retune them "
          "after the 1k-item pilot.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def write_manifest_report(r: dict, path: Path) -> None:
    L = ["# Phase 3: training manifests" + (" (DRY RUN on synthetic teacher labels)" if r["synthetic"] else ""), "",
         f"Built {r['built']} by `scripts/p3/build_manifest.py {r['command']}`. Numbers: `manifest-report.json` next to this file.", "",
         "## Mixture (train partition)", "",
         f"Total {_n(r['total_train'])} train rows (limited by **{r['limited_by']}**; the plan's cap is 130k). Supply after "
         f"exclusions: {', '.join(f'{k} {_n(v)}' for k, v in r['supply'].items())}.", ""]
    rows = []
    for lane, v in r["lanes"].items():
        rows.append([f"{lane}-code lane", _n(v["rows"]), _n(v["train"]), _n(v["dev"])] +
                    [f"{v['shares'][k]:.1%} (want {r['shares_wanted'][k]:.0%})" for k in r["shares_wanted"]] +
                    ["yes" if v["shares_ok"] else "**NO**", _n(v["options_255"])])
    L += _t(["lane", "rows", "train", "dev"] + list(r["shares_wanted"]) + ["shares within 2 pts", "255-option rows"], rows)
    for lane, v in r["lanes"].items():
        L += ["", f"### Lane {lane} (`{v['path']}`, sha256 `{v['sha256'][:16]}…`)", "",
              f"- hard rows by source: {v['hard_by_source']}; by difficulty: {v['hard_by_difficulty']}",
              f"- unknown-gold share of hard text: {v['unknown_share_hard_text']:.1%}; with target_probs {_n(v['with_target_probs'])}; "
              f"with rationale {_n(v['with_rationale'])}",
              f"- replay by source: {v['replay_by_source']}",
              f"- image directories referenced (ship these to the pod): {', '.join(f'`{d}`' for d in v['image_dirs'][:20])}"
              + (" …" if len(v["image_dirs"]) > 20 else ""), "",
              "Hard rows by family (top 30):", ""]
        L += _t(["family", "rows"], [[k, _n(c)] for k, c in list(v["hard_by_family"].items())[:30]])
    L += ["", "## Dev / eval manifests (partition dev, never trained on)", ""]
    L += _t(["name", "rows", "255-option rows", "path"], [[k, _n(v["rows"]), _n(v["options_255"]), f"`{v['path']}`"]
                                                         for k, v in r["dev_manifests"].items()])
    L += ["", "## Gate (`decontam.py --leakage`, every lane vs every held-out file)", ""]
    for k, v in r["gate"].items():
        L.append(f"- **{k}**: `{v['result']}`" + (f" (only fixed-state links: {v['only_fixed_state_links']})" if "only_fixed_state_links" in v else ""))
    L += ["", "## Dropped", ""]
    L += _t(["reason", "rows"], [[k, _n(c)] for k, c in sorted(r["dropped"].items(), key=lambda x: -x[1])])
    L += ["", "The trainer reads `data/manifests/<version>.jsonl`: use `--version ../p3/manifests/p3-train-256` (or copy the file).",
          "Lanes: `p3-train-256` for `--readout-codes 256`, `p3-train-255` for the shipped 255-code readout (no 255-option row).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
