"""Phase-3 pool assembly and manifest builder (scripts/p3/assemble_pool.py, scripts/p3/build_manifest.py, scripts/p3/assembly_*.py).

Unit tests run on small fixtures; the checks on the real pool (data/p3/pool/) are skipped when it has not been built.
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts/p3", ROOT / "scripts/p2"):
    sys.path.insert(0, str(p))
import assembly_common as ac  # noqa: E402
import assembly_heldout as ah  # noqa: E402
import assembly_large_choice as alc  # noqa: E402
import assembly_manifest as am  # noqa: E402
import assembly_quotas as aq  # noqa: E402
import candidate  # noqa: E402
from decision_data import render, SoftTarget  # noqa: E402

POOL = ac.POOL_DIR
REAL = (POOL / "pool-manifest.json").exists()
real = pytest.mark.skipif(not REAL, reason="data/p3/pool not built")


def cand(rid, state="A plain state about invoice 12 and its due date on 3 March.", ftype="noul", gold=True, parent=None, **pv):
    field = {"type": ftype, "question": "Is the invoice overdue on the review date?"}
    if ftype == "choice":
        field["options"] = [{"key": "a", "text": "Option A"}, {"key": "b", "text": "Option B", "description": "the second"}]
    if ftype == "score":
        field["levels"] = [{"value": 0, "description": "low"}, {"value": 1, "description": "mid"}, {"value": 2, "description": "high"}]
    return {"id": rid, "source": pv.pop("source", "A"), "dataset": pv.pop("dataset", "fixture"), "family": pv.pop("family", "fam"),
            "difficulty": 3, "state": state, "images": pv.pop("images", []), "field": field, "gold": gold,
            "unknown_reason": None if gold is not None else "insufficient_evidence", "gold_kind": "constructed",
            "parent_id": parent, "provenance": {"licence": "generated", **pv}}


LONG = ("The quarterly reconciliation for the northern warehouse lists pallet counts per aisle and the auditor noted that "
        "aisle seven had three damaged pallets which were written off after the insurance assessor visited on Tuesday morning "
        "and signed the release form for the claim number recorded in the ledger")


# ------------------------------------------------------------------------------------------------ groups / split integrity
def test_groups_union_parent_group_id_screen_and_upstream():
    rows = [cand("p1"), cand("c1", parent="p1"), cand("c2", parent="c1"),
            cand("g1", group_id="doc-9"), cand("g2", group_id="doc-9"),
            cand("f1", source="B", dataset="finqa", upstream_dataset="finqa", upstream_id="ABC/2010/page_3.pdf-1"),
            cand("f2", source="B", dataset="finqa", upstream_dataset="finqa", upstream_id="ABC/2010/page_3.pdf-2"),
            cand("s1", dataset="gui_synthetic", screen_id="s00001"), cand("s2", dataset="gui_synthetic", screen_id="s00001"),
            cand("lone")]
    g = ac.assign_groups(rows)
    assert g["p1"] == g["c1"] == g["c2"]
    assert g["g1"] == g["g2"] != g["p1"]
    assert g["f1"] == g["f2"]                  # a FinQA filing page is one upstream item
    assert g["s1"] == g["s2"]                  # questions over one rendered screen
    assert len({g["p1"], g["g1"], g["f1"], g["s1"], g["lone"]}) == 5
    assert ac.assign_groups(list(reversed(rows))) == g   # stable representative


def test_bc_candidates_are_whole_groups_of_b_and_c_only():
    rows = [cand(f"b{i}", source="B", group_id=f"doc{i // 2}") for i in range(400)] + \
           [cand(f"a{i}", source="A", group_id=f"mix{i}") for i in range(50)] + \
           [cand(f"m{i}", source="C", group_id=f"mix{i}") for i in range(50)]
    g = ac.assign_groups(rows)
    chosen = ah.bc_candidate_groups(rows, g, 0.3, "salt")
    assert chosen and all(not gid.startswith("g:mix") for gid in chosen)
    members = collections.defaultdict(set)
    for r in rows:
        members[g[r["id"]]].add(r["id"])
    for gid in chosen:                                   # both rows of a doc come together
        assert len(members[gid]) == 2


def test_select_bc_respects_share_and_scarce_cap_and_cost():
    rows = [cand(f"p{i}", source="B", dataset="plumb", group_id=f"pd{i}") for i in range(100)] + \
           [cand(f"q{i}", source="B", dataset="other", group_id=f"od{i}") for i in range(100)]
    g = ac.assign_groups(rows)
    cands = {g[r["id"]] for r in rows}
    costs = {g["q0"]: 999}
    sel = ah.select_bc_groups(rows, g, cands, costs, share=0.10, scarce_cap=0.05, salt="s", max_links=25)
    assert sum(1 for x in sel if x.startswith("g:pd")) == 5          # plumb capped at 5%
    assert sum(1 for x in sel if x.startswith("g:od")) == 10
    assert g["q0"] not in sel                                        # too many pool links


@real
def test_real_heldout_groups_never_share_a_group_key_with_the_pool():
    held = [r for f in ("heldout-fresh", "heldout-fresh-gui", "heldout-fresh-large-choice") for r in ac.read_jsonl(POOL / f"{f}.jsonl")]
    held_keys = {k for r in held for k in ac.group_keys(r)}
    held_ids = {r["id"] for r in held}
    n = 0
    for p in sorted((POOL / "shards").glob("pool-*.jsonl")) + [POOL / "large-choice.jsonl"]:
        for r in ac.read_jsonl(p):
            n += 1
            assert r["id"] not in held_ids
            shared = set(ac.group_keys(r)) & held_keys
            assert not shared, (r["id"], shared)
    assert n > 100_000


@real
def test_real_shards_match_manifest_and_carry_pool_fields():
    m = json.loads((POOL / "pool-manifest.json").read_text())
    total = 0
    fam_per_shard = collections.defaultdict(collections.Counter)
    for s in m["shards"]:
        p = ROOT / s["path"]
        assert ac.sha256_file(p) == s["sha256"]
        for r in ac.read_jsonl(p):
            total += 1
            assert set(r["pool"]) == {"group", "quota_key", "shard", "heldout_flagged_bucket"}
            assert r["dataset"] != "large_choice" and len(r["field"].get("options") or []) <= 254
            fam_per_shard[r["pool"]["quota_key"]][r["pool"]["shard"]] += 1
            want = ac.stable_unit("p3-heldout-flagged-v1", r["pool"]["group"]) < 0.05
            assert r["pool"]["heldout_flagged_bucket"] == want
    assert total == m["counts"]["shard_rows"]
    for k, c in fam_per_shard.items():                  # family-balanced: every family spread evenly over the shards
        if sum(c.values()) >= 64:
            assert max(c.values()) - min(c.get(i, 0) for i in range(64)) <= 1, k


# ------------------------------------------------------------------------------------------------ leakage
def _write(path, rows):
    ac.write_jsonl(path, rows)
    return path


def _leakage(train, held):
    res = subprocess.run([sys.executable, str(ROOT / "scripts/p3/decontam.py"), "--leakage", "--train", str(train), "--heldout",
                          str(held)], cwd=ROOT, capture_output=True, text=True)
    return res.returncode, res.stdout


def test_leakage_gate_fails_on_family_and_near_dup_and_passes_when_clean(tmp_path):
    held = _write(tmp_path / "h.jsonl", [cand("h1", state=LONG + " alpha"), cand("h2", state="Short fresh state about route 66.")])
    ok = _write(tmp_path / "ok.jsonl", [cand("t1", state="Completely different text about a bakery order of 40 loaves."),
                                        cand("t2", state="Another unrelated passage on train timetables and platform 4.")])
    assert _leakage(ok, held)[0] == 0
    fam = _write(tmp_path / "fam.jsonl", [cand("t3", parent="h2", state="Variant text that shares nothing at all.")])
    rc, out = _leakage(fam, held)
    assert rc == 1 and "family" in out
    dup = _write(tmp_path / "dup.jsonl", [cand("t4", state=LONG + " beta")])
    rc, out = _leakage(dup, held)
    assert rc == 1 and "near_dup" in out


def test_scan_links_reports_every_linked_heldout_row_not_only_the_best():
    held = [cand("h1", state=LONG + " one"), cand("h2", state=LONG + " two")]
    lines = [json.dumps(cand("t1", state=LONG + " three")), json.dumps(cand("t2", state="nothing in common here at all"))]
    links = ac.scan_links(lines, held, workers=1)
    assert {(i, h) for i, r, h in links if r == "near_dup"} == {(0, "h1"), (0, "h2")}


def test_resolve_links_removes_cheap_links_and_drops_expensive_heldout():
    pool = [cand(f"p{i}", state=LONG + f" copy {i}", group_id=f"pg{i}") for i in range(30)] + \
           [cand("q1", state="The ferry to the island leaves at 7 from pier 2 on weekdays only " * 2, group_id="qg")]
    held = [cand("h1", state=LONG + " held"), cand("h2", state="The ferry to the island leaves at 7 from pier 2 on weekdays only " * 2)]
    pg = ac.assign_groups(pool)
    lines = [json.dumps(r) for r in pool]
    hg = {"h1": "fresh:h1", "h2": "fresh:h2"}
    removed, dropped, rounds = ah.resolve_links(lines, [pg[r["id"]] for r in pool], held, hg, set(), max_links=5, log=lambda *a: None)
    assert dropped == {"fresh:h1"}                 # would force 30 pool rows out
    assert removed == {pg["q1"]}                   # one cheap link: the pool row leaves
    assert rounds[-1]["links"] == 0


# ------------------------------------------------------------------------------------------------ 255-option items
def test_candidate_max_options_is_255():
    assert candidate.MAX_OPTIONS == 255


def test_large_choice_items_are_valid_with_a_unique_constructed_gold():
    rows = alc.generate("pytest", 60)
    assert len(rows) == 60
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    twins = [r for r in rows if r["gold"] is None]
    assert twins and all(r["unknown_reason"] == "not_listed" and r["parent_id"] in ids for r in twins)
    assert {r["provenance"]["domain"] for r in rows} == set(alc.DOMAINS)
    for r in rows:
        assert candidate.validate(r) == []
        opts = r["field"]["options"]
        assert len(opts) == 255 and len({o["key"] for o in opts}) == 255
        assert len({o["description"] for o in opts}) == 255
        assert alc.recheck(r) == r["gold"]       # exactly one record meets every constraint (None for a twin)
        assert r["family"] == "large_choice" and r["source"] == "A" and r["provenance"]["licence"] == "generated"
    assert alc.generate("pytest", 60) == rows    # deterministic


def test_large_choice_is_identical_across_processes():
    code = ("import sys,json,hashlib; sys.path.insert(0,'scripts/p3'); import assembly_large_choice as a; "
            "print(hashlib.sha256(json.dumps(a.generate('xproc', 12)).encode()).hexdigest())")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}).stdout for seed in (1, 2, 3)}
    assert len(outs) == 1 and len(next(iter(outs))) > 10


def test_large_choice_renders_with_256_codes():
    from vision_decision.scoring import labels_for_count
    r = alc.generate("pytest-render", 1)[0]
    rec = am.candidate_to_record(r, None, group="g", partition="train", mix="hard_text", image_meta=False)
    header, choices, texts, target = render(rec)
    assert len(choices) == 256 and choices[target][0] == r["gold"]
    assert len(labels_for_count(len(choices), 256)) == 256
    with pytest.raises(ValueError):
        labels_for_count(len(choices), 255)


@real
def test_real_large_choice_rows_resolve():
    rows = list(ac.read_jsonl(POOL / "large-choice.jsonl"))
    assert 1000 <= len(rows) <= 2000
    for r in rows[::25]:
        assert alc.recheck(r) == r["gold"]


# ------------------------------------------------------------------------------------------------ quotas
def test_quotas_sum_to_totals_and_fit_the_cap():
    rows = []
    for fam, n, src, img in (("long_policy", 3000, "A", False), ("judge_hard", 800, "B", False), ("intent", 2000, "C", False),
                             ("logic", 4000, "A", False), ("image_joint_rule", 5000, "I", True), ("color", 900, "C", True),
                             ("large_choice", 500, "A", False)):
        for i in range(n):
            r = cand(f"{fam}-{i}", family=fam, source=src, dataset="plumb" if src == "B" else fam,
                     images=["p3/x.png"] if img else [])
            r["gold_kind"] = "dataset" if src in ("B", "C") else "constructed"
            rows.append(r)
    q = aq.compute(rows, cap_total=5000, image_share=0.36)
    fams = q["families"]
    assert sum(v["quota"] for v in fams.values()) == q["first_solve_total"]
    assert q["first_solve_text"] + q["first_solve_image"] == q["first_solve_total"]
    assert q["expected_total_calls"] <= 5000
    assert fams["large_choice"]["quota"] == 0 and not fams["large_choice"]["teacher"]
    assert all(v["quota"] <= v["available_at_flag_rate"] for v in fams.values())
    # weight 3 family takes a higher rate of its rows than weight 1.5 and 0.5 families
    assert fams["long_policy"]["rate"] > fams["logic"]["rate"] > fams["intent"]["rate"]
    assert abs(q["first_solve_image"] - 0.36 * q["first_solve_total"]) <= 2


@real
def test_real_quotas_file_is_consistent():
    q = json.loads((POOL / "quotas.json").read_text())
    assert sum(v["quota"] for v in q["families"].values()) == q["first_solve_total"]
    assert q["expected_total_calls"] <= q["cap_total_teacher_calls"]
    assert "large_choice" not in q["families"]           # not mined, not in the shards


# ------------------------------------------------------------------------------------------------ manifest records
def test_records_load_through_render_for_every_type_and_level_mapping():
    lab = {"target": "b", "probs": {"a": 0.2, "b": 0.7, "unknown": 0.1}, "rationale": "B fits.", "target_kind": "teacher", "review": None}
    rec = am.candidate_to_record(cand("c", ftype="choice", gold="b"), lab, group="g", partition="train", mix="hard_text")
    assert rec["request"]["fields"][0]["options"][0] == {"value": "a", "description": "Option A"}
    assert rec["request"]["fields"][0]["options"][1] == {"value": "b", "description": "Option B — the second"}
    h, ch, tx, t = render(rec, soft_targets=True)
    assert isinstance(t, SoftTarget) and ch[t.gold][0] == "b" and abs(t.probs[1] - 0.7) < 1e-9
    assert rec["rationale"] == "B fits."
    # score with 1-based level values (C-text p2 rows)
    s = cand("s", ftype="score", gold=2, level_values=[1, 2, 3])
    rec = am.candidate_to_record(s, {"target": 2, "probs": {"0": 0.1, "1": 0.1, "2": 0.8}, "rationale": None,
                                     "target_kind": "teacher", "review": None}, group="g", partition="train", mix="hard_text")
    assert [l["value"] for l in rec["request"]["fields"][0]["levels"]] == [1, 2, 3] and rec["target"] == 3
    assert rec["target_probs"] == {"1": 0.1, "2": 0.1, "3": 0.8, "unknown": 0.0}
    h, ch, tx, t = render(rec, soft_targets=True)
    assert ch[t.gold][0] == 3
    # noul unknown
    u = cand("u", gold=None)
    rec = am.candidate_to_record(u, None, group="g", partition="dev", mix="heldout")
    assert rec["target"] is None and rec["abstention_cause"] == "insufficient_evidence"
    h, ch, tx, t = render(rec)
    assert ch[t][0] == "__unknown__"
    assert am.check_record(rec) is None


def test_label_contract_rejects_bad_labels():
    c = cand("c", ftype="choice", gold="a")
    with pytest.raises(ValueError):
        am.candidate_to_record(c, {"target": "zzz", "probs": None, "rationale": None, "target_kind": "gold", "review": None},
                               group="g", partition="train", mix="hard_text")
    with pytest.raises(ValueError):
        am.normalise_probs(c, {"a": 0.5, "nope": 0.5})
    assert am.trainer_problem(cand("big", state="x " * 70000)) == "state over the serving contract's state limit"


def test_proportional_and_tiered_selection():
    rows = [{"id": f"r{i}", "s": "big" if i < 800 else "small", "u": i % 10 == 0} for i in range(1000)]
    got = am.stratified_take(rows, 100, lambda r: r["s"], lambda r: r["u"], "t")
    assert len(got) == 100 and all(r["u"] for r in got)            # 100 must-keep rows fill it
    got = am.stratified_take(rows, 500, lambda r: r["s"], lambda r: r["u"], "t")
    assert len(got) == 500 and sum(r["u"] for r in got) == 100
    rest = [r for r in got if not r["u"]]
    assert abs(sum(r["s"] == "big" for r in rest) / len(rest) - 0.8) < 0.01
    t = am.tiered_take([rows[:50], rows[50:]], 80, lambda r: r["s"], "t")
    assert len(t) == 80 and set(r["id"] for r in rows[:50]) <= {r["id"] for r in t}


# ------------------------------------------------------------------------------------------------ build_manifest end to end (fixture pool)
def _fixture_pool(tmp: Path):
    import gen_reasoning as gr
    pool_rows = gr.generate("pytest-pool", 420, None, 1)
    img = []
    for i in range(160):
        sha = f"{i:064x}"
        r = cand(f"img{i}", family="image_joint_rule", source="I", dataset="image_joint", images=[f"p3/fake/{sha}.png"],
                 image_sha256=[sha], state={"photo": f"item number {i} in the returns bay"}, gold=bool(i % 2))
        img.append(r)
    lc = alc.generate("pytest-pool", 30)
    rows = pool_rows + img
    g = ac.assign_groups(rows + lc)
    (tmp / "shards").mkdir(parents=True)
    for r in rows:
        r["pool"] = {"group": g[r["id"]], "quota_key": aq.quota_key(r), "shard": 0,
                     "heldout_flagged_bucket": ac.stable_unit("p3-heldout-flagged-v1", g[r["id"]]) < 0.05}
    ac.write_jsonl(tmp / "shards" / "pool-00.jsonl", rows)
    for r in lc:
        r["pool"] = {"group": g[r["id"]], "quota_key": "large_choice", "shard": None, "heldout_flagged_bucket": False}
    ac.write_jsonl(tmp / "large-choice.jsonl", lc)
    ac.write_jsonl(tmp / "heldout-fresh.jsonl", gr.generate("pytest-heldout", 30, None, 1))
    ac.write_jsonl(tmp / "heldout-flagged.jsonl", [r for r in rows if r["pool"]["heldout_flagged_bucket"]][:5])
    ac.write_jsonl(tmp / "heldout-fresh-large-choice.jsonl", alc.generate("pytest-heldout", 3, id_prefix="p3-hf-lc"))
    replay_text = [{"id": f"rt{i}", "source": "fixture", "partition": "train", "family": f"f{i % 3}", "images": [],
                    "request": {"schema_version": "1.0", "request_id": f"rt{i}", "state": f"Replay row {i} about topic {i * 7}.",
                                "fields": [{"id": "d", "type": "boolean", "question": f"Is replay statement {i} true?"}]},
                    "target": bool(i % 2), "abstention_cause": None} for i in range(400)]
    replay_img = [{"id": f"ri{i}", "source": "fixture", "partition": "train", "family": "img",
                   "images": [{"image": f"data/fake/{i:064x}.jpg", "sha256": f"{i + 10**6:064x}", "width": 64, "height": 64}],
                   "request": {"schema_version": "1.0", "request_id": f"ri{i}", "state": {},
                               "fields": [{"id": "d", "type": "boolean", "question": f"Is object {i} red?"}]},
                   "target": bool(i % 2), "abstention_cause": None} for i in range(400)]
    ac.write_jsonl(tmp / "replay-text.jsonl", replay_text)
    ac.write_jsonl(tmp / "replay-img.jsonl", replay_img)


def test_build_manifest_end_to_end_shares_lanes_and_gate(tmp_path):
    import build_manifest as bm
    _fixture_pool(tmp_path)
    out = tmp_path / "out"
    rc = bm.main(["--synthetic", "600", "--pool-dir", str(tmp_path), "--out-dir", str(out), "--total", "100000",
                  "--replay-p2", str(tmp_path / "replay-text.jsonl"), "--replay-eikos", "", "--replay-v21", "",
                  "--replay-images", str(tmp_path / "replay-img.jsonl"), "--dev-share", "0.05"])
    assert rc == 0
    rep = json.loads((out / "manifest-report.json").read_text())
    want = rep["shares_wanted"]
    for lane in ("255", "256"):
        rows = list(ac.read_jsonl(out / f"p3-train-{lane}.jsonl"))
        tr = [r for r in rows if r["partition"] == "train"]
        mix = collections.Counter(r["mix"] for r in tr)
        for k, s in want.items():
            assert abs(mix[k] / len(tr) - s) <= 0.02, (lane, k, mix[k] / len(tr))
        n255 = sum(len(r["request"]["fields"][0].get("options") or []) == 255 for r in rows)
        assert (n255 == 0) if lane == "255" else (n255 > 0)
        assert any(r["partition"] == "dev" for r in rows)
        for r in rows[:300]:                              # every record goes through the trainer's renderer
            render(r, soft_targets=True)
            render(r)
        assert rep["gate"][lane]["exit"] == 0
    held = {r["id"] for r in ac.read_jsonl(tmp_path / "heldout-fresh.jsonl")}
    train = list(ac.read_jsonl(out / "p3-train-256.jsonl"))
    assert not held & {r["id"] for r in train}
    bucket_groups = {r["pool"]["group"] for r in ac.read_jsonl(tmp_path / "shards" / "pool-00.jsonl")
                     if r["pool"]["heldout_flagged_bucket"]}
    assert bucket_groups and not bucket_groups & {r.get("source_group") for r in train}
    assert rep["dev_manifests"]["heldout-flagged"]["rows"] >= 1
    assert (out / "p3-dev-heldout-fresh.jsonl").exists() and (out / "p3-dev-heldout-fresh-large-choice.jsonl").exists()


def test_variant_inherits_the_parent_group():
    import build_manifest as bm

    class P:
        rows = {"par": {"id": "par", "pool": {"group": "g:doc"}}}

        def group(self, rid):
            r = self.rows.get(rid)
            return r["pool"]["group"] if r else None
    v = {"id": "par-v1", "parent_id": "par"}
    vv = {"id": "par-v1-v1", "parent_id": "par-v1"}
    assert bm.root_group(P(), v, {"par-v1": v}) == "g:doc"
    assert bm.root_group(P(), vv, {"par-v1": v, "par-v1-v1": vv}) == "g:doc"


def test_build_manifest_reads_teacher_rows_heldout_verdicts_drops_and_publishes_pod_names(tmp_path):
    """Integration: azure_teacher kept rows ({item, label}), constructed variants (target_kind constructed), heldout-results
    (teacher verdicts on the flagged half), review drop-ids -> the pod-facing manifest names (cloud/p3/pod_run_train.sh)."""
    import azure_teacher as T
    import build_manifest as bm
    _fixture_pool(tmp_path)
    rows = list(ac.read_jsonl(tmp_path / "shards" / "pool-00.jsonl"))
    train_rows = [r for r in rows if not r["pool"]["heldout_flagged_bucket"] and not r.get("images")][:301]
    train_rows = train_rows[:300] + [r for r in rows if not r["pool"]["heldout_flagged_bucket"] and r.get("images")][:120] + train_rows[300:]
    verified = tmp_path / "kept.jsonl"
    with open(verified, "w") as fh:
        for r in train_rows[:420]:
            labs = am.candidate_labels(r)
            gold = am.to_label(r, r["gold"])
            probs = {l: (0.9 if l == gold else 0.1 / (len(labs) - 1)) for l in labs}
            verdict = {"keep": True, "target": "teacher_dist", "target_probs": probs}
            fh.write(json.dumps({"id": r["id"], "keep": True, "rule": "A", "origin": "mine", "item": r,
                                 "label": T.label_for(r, verdict, ["Two sentences. Of rationale."])}) + "\n")
        fh.write(json.dumps({"id": train_rows[420]["id"], "keep": False, "item": train_rows[420], "label": None}) + "\n")
    parent = train_rows[0]
    var = {k: v for k, v in parent.items() if k != "pool"}
    var.update(id=parent["id"] + "-var1", parent_id=parent["id"],
               label={"target": parent["gold"], "probs": None, "rationale": None, "target_kind": "constructed", "review": None})
    constructed = tmp_path / "constructed.jsonl"
    constructed.write_text(json.dumps(var) + "\n")
    flagged = list(ac.read_jsonl(tmp_path / "heldout-flagged.jsonl"))
    hres = tmp_path / "heldout-results.jsonl"
    with open(hres, "w") as fh:
        fh.write(json.dumps({"id": flagged[0]["id"], "keep": True, "origin": "heldout", "item": flagged[0],
                             "label": {"target": flagged[0]["gold"], "probs": None, "rationale": None, "target_kind": "gold"}}) + "\n")
        fh.write(json.dumps({"id": flagged[1]["id"], "keep": False, "origin": "heldout", "item": flagged[1], "label": None}) + "\n")
    drop = tmp_path / "drop-ids.json"
    drop.write_text(json.dumps({"ids": [train_rows[5]["id"]]}))
    man = tmp_path / "manifests"
    rc = bm.main(["--verified", str(verified), str(constructed), "--pool-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
                  "--total", "100000", "--replay-p2", str(tmp_path / "replay-text.jsonl"), "--replay-eikos", "", "--replay-v21", "",
                  "--replay-images", str(tmp_path / "replay-img.jsonl"), "--heldout-results", str(hres), "--drop-ids", str(drop),
                  "--publish", "decision-p3", "--publish-dir", str(man)])
    assert rc == 0
    rep = json.loads((tmp_path / "out" / "manifest-report.json").read_text())
    assert rep["dropped"]["teacher_dropped_row_in_verified"] == 1 and rep["dropped"]["review_says_wrong_or_disputed"] == 1
    assert rep["dropped"]["heldout_flagged_teacher_rejected"] == 1
    names = {"decision-p3", "decision-p3-lane255", "decision-p3-labelled", "decision-p3-heldout-fresh",
             "decision-p3-heldout-fresh-large-choice", "decision-p3-heldout-flagged", "decision-p3-human-dev"}
    assert names == set(rep["published"])
    for n in names - {"decision-p3", "decision-p3-lane255", "decision-p3-labelled"}:
        assert all(r["partition"] == "test" for r in ac.read_jsonl(man / f"{n}.jsonl"))
    fresh = list(ac.read_jsonl(man / "decision-p3-heldout-fresh.jsonl"))
    assert fresh and max(len(r["request"]["fields"][0].get("options") or []) for r in fresh) <= 254
    assert not any(len(r["request"]["fields"][0].get("options") or []) > 254 for r in ac.read_jsonl(man / "decision-p3-lane255.jsonl"))
    labelled = {r["id"]: r for r in ac.read_jsonl(man / "decision-p3-labelled.jsonl")}
    assert var["id"] in labelled and labelled[var["id"]]["p3"]["target_kind"] == "constructed"     # not capped as hard gold
    assert train_rows[5]["id"] not in labelled and train_rows[420]["id"] not in labelled
    assert any("target_probs" in r and r.get("rationale") for r in labelled.values())
    fl = [r["id"] for r in ac.read_jsonl(man / "decision-p3-heldout-flagged.jsonl")]
    assert flagged[0]["id"] in fl and flagged[1]["id"] not in fl
