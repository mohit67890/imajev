"""Phase-3 pod tooling: scripts/p3/{transfer,eval_checkpoint,make_decisionbench_subset}.py and cloud/p3/* (no GPU, no network)."""
from __future__ import annotations

import builtins
import importlib.util
import io
import json
import math
import os
import random
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


transfer = load("scripts/p3/transfer.py", "p3_transfer")
evalck = load("scripts/p3/eval_checkpoint.py", "p3_eval_checkpoint")
dbsub = load("scripts/p3/make_decisionbench_subset.py", "p3_db_subset")
pilot = load("cloud/p3/pilot_decision.py", "p3_pilot_decision")
selector = load("cloud/p3/select_checkpoint.py", "p3_select_checkpoint")
prep = load("cloud/p3/prep_manifests.py", "p3_prep_manifests")
r2 = load("cloud/p3/round2_manifest.py", "p3_round2_manifest")
watch = load("cloud/p3/ckpt_watch.py", "p3_ckpt_watch")
plan_steps = load("cloud/p3/plan_steps.py", "p3_plan_steps")
final_table = load("cloud/p3/final_table.py", "p3_final_table")


# ------------------------------------------------------------------------------------------------ shell scripts
@pytest.mark.parametrize("script", ["cloud/p3/bootstrap_train.sh", "cloud/p3/pod_run_train.sh", "cloud/p3/bootstrap_mine.sh"])
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", ["cloud/p3/bootstrap_train.sh", "cloud/p3/pod_run_train.sh", "cloud/p3/bootstrap_mine.sh"])
def test_scripts_never_terminate_the_pod_or_print_tokens(script):
    text = (ROOT / script).read_text()
    for bad in ("runpodctl remove", "delete-pod", "shutdown", "poweroff", "echo $HF_TOKEN", "echo \"$HF_TOKEN", "--token $", "set -x"):
        assert bad not in text, (script, bad)
    assert "ALL_DONE" in text or "MINE_DONE" in text or "BOOTSTRAP_DONE" in text


def test_pod_run_uses_the_stage3_recipe_and_the_selector_only():
    text = (ROOT / "cloud/p3/pod_run_train.sh").read_text()
    for flag in ("--soft-targets --soft-weight 1.0 --permute-options --rationale-weight 0.3 --rationale-max-tokens 192 --max-length 16384",
                 "--ordinal-weight $W --readout-codes $CODES", "--init-adapter $SHIPPED", "parity_readout256.py --backend torch"):
        assert flag in text
    # the pick is made by select_checkpoint.py; DecisionBench numbers only reach final_table.py (for a person)
    pick_block = text[text.index("# ---------------------------------------------------------------------------------------------------------------- 8 pick"):
                      text.index("# ---------------------------------------------------------------------------------------------------------------- 9 final")]
    assert "select_checkpoint.py" in pick_block and "decisionbench" not in pick_block.lower() and "$TR" not in pick_block


# ------------------------------------------------------------------------------------------------ selector never reads DecisionBench
def _ckpt(root: Path, name: str, fresh: float, human: float, order: float, shippable: bool, n=400):
    d = root / name
    d.mkdir(parents=True)
    (d / "selection.json").write_text(json.dumps({"name": name, "ckpt": f"/ckpts/{name}", "order": order, "round": int(order), "step": 1,
                                                  "fresh": {"acc": fresh, "n": n}, "human": {"acc": human, "n": 100}, "flagged": {"acc": 10.0, "n": n}}))
    (d / "gates.json").write_text(json.dumps({"shippable": shippable, "soup": False}))
    (d / "decisionbench.json").write_text(json.dumps({"primary_accuracy": 99.9}))   # decoy next to the allowed files


def test_selector_never_opens_decisionbench_files(tmp_path, monkeypatch):
    ev, tr = tmp_path / "eval", tmp_path / "tracking-only"
    _ckpt(ev, "r1-s000100", 60.0, 60.0, 1.2, True); _ckpt(ev, "r1-s000200", 61.0, 61.0, 1.4, True); _ckpt(ev, "r2-s000050", 55.0, 50.0, 2.5, True)
    for n in ("r1-s000100", "r1-s000200", "r2-s000050"):
        (tr / n).mkdir(parents=True); (tr / n / "decisionbench.json").write_text(json.dumps({"primary_accuracy": 1.0}))
    (ev / "tracking-only").mkdir(); (ev / "tracking-only" / "selection.json").write_text("{}")   # never listed as a candidate
    opened = []
    real_open, real_io_open = builtins.open, io.open

    def spy(file, *a, **k):
        opened.append(str(file)); return real_open(file, *a, **k)

    def spy_io(file, *a, **k):
        opened.append(str(file)); return real_io_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", spy); monkeypatch.setattr(io, "open", spy_io)
    rc = selector.main(["--eval-root", str(ev), "--out", str(tmp_path / "pick.json")])
    monkeypatch.undo()
    assert rc == 0 and opened
    assert not [p for p in opened if "decisionbench" in p.lower() or "tracking-only" in p], opened
    assert all(Path(p).name in ("selection.json", "gates.json", "pick.json") for p in opened), opened
    pick = json.loads((tmp_path / "pick.json").read_text())
    assert pick["pick"] == "r1-s000200"


def test_selector_refuses_tracking_paths(tmp_path):
    with pytest.raises(PermissionError):
        selector._read_json(tmp_path / "tracking-only" / "x" / "selection.json")
    with pytest.raises(PermissionError):
        selector._read_json(tmp_path / "eval" / "x" / "decisionbench.json")
    with pytest.raises(PermissionError):
        selector.load_candidates(tmp_path / "tracking-only")
    src = (ROOT / "cloud/p3/select_checkpoint.py").read_text()
    code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    assert "decisionbench.json" not in code and "tracking_root" not in code


def test_selector_rules(tmp_path):
    ev = tmp_path / "eval"
    _ckpt(ev, "a", 70.0, 70.0, 1.1, True); _ckpt(ev, "b", 69.5, 70.0, 1.9, True)        # within noise, later -> b
    _ckpt(ev, "c", 80.0, 80.0, 1.5, False)                                                # best but fails a gate
    res = selector.select(selector.load_candidates(ev))
    assert res["pick"] == "b" and "c" not in res["within_noise"] and res["best_by_score"] == "c"
    res2 = selector.select(selector.load_candidates(ev), require_gates=False)
    assert res2["pick"] == "c"
    ev2 = tmp_path / "eval2"; _ckpt(ev2, "x", 50.0, 50.0, 1.0, False)
    res3 = selector.select(selector.load_candidates(ev2))
    assert res3["pick"] is None and res3["best_by_score"] == "x"
    ev3 = tmp_path / "eval3"; _ckpt(ev3, "old", 75.0, 75.0, 1.2, True, n=2000); _ckpt(ev3, "new", 70.0, 70.0, 2.0, True, n=2000)
    assert selector.select(selector.load_candidates(ev3))["pick"] == "old"                 # a clear gap beats "later"


def test_decisionbench_scores_go_only_to_tracking_root(tmp_path, monkeypatch):
    sub = tmp_path / "tracking-only" / "decisionbench-3k.jsonl"; sub.parent.mkdir(parents=True)
    rows = []
    for i, prim in enumerate(["binary_classification", "candidate_selection", "ordinal_scoring"]):
        cands = [{"id": f"c{j}", "label": f"L{j}", "description": None, "ordinal_value": float(j)} for j in range(3 if prim != "binary_classification" else 2)]
        rows.append({"row_id": f"r{i}", "task_id": f"t{i}", "primitive": prim, "family": "reasoning" if i == 1 else "f", "instruction": "Pick.",
                     "state_json": json.dumps({"x": i}), "candidates_json": json.dumps(cands), "gold_candidate_id": "c1"})
    sub.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (sub.parent / "decisionbench-3k.meta.json").write_text(json.dumps({"full_counts": {"t0": 100, "t1": 200, "t2": 100}}))

    def fake_post(port, req, timeout=600):
        q = req["questions"]["decision"]
        if q["type"] == "noul":
            return {"answers": {"decision": {"noul": 0.2}}}                        # -> c1 (false) wins
        if q["type"] == "choice":
            return {"answers": {"decision": {"probabilities": {"c0": 0.1, "c1": 0.8, "c2": 0.1}}}}
        return {"answers": {"decision": {"probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}}}}
    monkeypatch.setattr(evalck, "post", fake_post)

    class A:
        name, tracking_root, decisionbench_subset, concurrency, out = "ck", str(sub.parent), str(sub), 1, str(tmp_path / "eval" / "ck")
    ctx = type("C", (), {"a": A(), "out": Path(A.out)})()
    summ = evalck.run_decisionbench(ctx, 0)
    out = sub.parent / "ck" / "decisionbench.json"
    assert out.exists() and not (tmp_path / "eval").exists()
    assert summ["primary_accuracy"] == pytest.approx(200 / 3) and summ["ordinal"] == 0.0 and summ["reasoning"] == 100.0
    assert summ["full_suite_equivalent"] == pytest.approx((100 * 100 + 100 * 200 + 0 * 100) / 400)


def test_eval_checkpoint_refuses_out_under_tracking(tmp_path):
    with pytest.raises(SystemExit):
        evalck.main(["--ckpt", "x", "--name", "n", "--out", str(tmp_path / "tracking-only" / "n"), "--model", "m",
                     "--tracking-root", str(tmp_path / "tracking-only")])
    with pytest.raises(SystemExit):
        evalck.main(["--ckpt", "x", "--name", "n", "--out", str(tmp_path / "e"), "--model", "m", "--tracking-root", str(tmp_path / "scores")])


# ------------------------------------------------------------------------------------------------ transfer bundles
BENCH_TEXT = "The leak ran four to six weeks inside the wall and the insured did not know about it; exception 4.3.1 restores cover."


def _index(min_chars=40):
    idx = transfer.BenchmarkIndex(min_chars=min_chars)
    idx.add_string("jevbench-public", BENCH_TEXT)
    return idx


def _tree(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True); (root / "src" / "a.py").write_text("print('hi')\n")
    (root / "src" / "._a.py").write_text("appledouble")
    (root / "data/manifests").mkdir(parents=True)
    (root / "data/manifests/decision-p3.jsonl").write_text(json.dumps({"id": "r1", "partition": "train", "state": "a clean training state " * 5}) + "\n")
    return root


def test_bundle_fails_on_benchmark_text(tmp_path):
    root = _tree(tmp_path)
    (root / "data/leak.jsonl").write_text(json.dumps({"id": "x", "state": {"doc": "  " + BENCH_TEXT.upper() + "  "}}) + "\n")
    entries = transfer.collect("none", ["src", "data"], [], (), root=root)
    with pytest.raises(transfer.BundleError, match="benchmark text"):
        transfer.build("t", entries, _index(), tmp_path / "out", root=root)
    assert not (tmp_path / "out").exists()                       # nothing written


def test_bundle_fails_on_benchmark_json_inside_a_string(tmp_path):
    root = _tree(tmp_path)
    (root / "data/leak2.jsonl").write_text(json.dumps({"id": "x", "state_json": json.dumps({"k": BENCH_TEXT})}) + "\n")
    with pytest.raises(transfer.BundleError, match="benchmark text"):
        transfer.check_entries(transfer.collect("none", ["data"], [], (), root=root), _index(), root=root)


def test_bundle_fails_on_decontam_sources_and_benchmark_paths(tmp_path):
    root = _tree(tmp_path)
    for rel in ("data/p3/decontam-sources/jevbench/hard.jsonl", "reports/benchmarks/x/preds.jsonl", "data/imajev-bench/private-1/r.jsonl"):
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("{}\n")
        with pytest.raises(transfer.BundleError, match="benchmark path"):
            transfer.check_entries([transfer.Entry(p, rel)], _index(), root=root)
    # an innocuous name elsewhere is fine; copying a benchmark file under another name is caught by its sha256
    p = root / "data/copy.jsonl"; p.write_text("same bytes\n")
    idx = _index(); idx.shas[transfer.sha256_file(p)] = "fast-decisions-dev"
    with pytest.raises(transfer.BundleError, match="pinned benchmark file"):
        transfer.check_entries([transfer.Entry(p, "data/copy.jsonl")], idx, root=root)


def test_clean_bundle_builds_verifies_and_extracts(tmp_path):
    root = _tree(tmp_path)
    entries = transfer.collect("none", ["src", "data"], [], (), root=root)
    assert not any("._" in e.arc for e in entries)
    m = transfer.build("t", entries, _index(), tmp_path / "out", root=root, part_bytes=150)   # force several parts
    assert m["check"]["status"] == "PASS" and len(m["parts"]) >= 2
    with pytest.raises(transfer.BundleError, match="gatekit"):
        pass_through = dict(m); pass_through["gatekit"] = True
        (tmp_path / "gk").mkdir(); (tmp_path / "gk" / "t.manifest.json").write_text(json.dumps(pass_through))
        transfer.verify_and_extract("t", tmp_path / "gk", None)
    dest = tmp_path / "x"
    transfer.verify_and_extract("t", tmp_path / "out", dest)
    assert (dest / "src/a.py").read_text() == "print('hi')\n" and (dest / "data/manifests/decision-p3.jsonl").exists()


def test_tampered_bundle_is_refused(tmp_path):
    root = _tree(tmp_path)
    transfer.build("t", transfer.collect("none", ["src"], [], (), root=root), _index(), tmp_path / "out", root=root)
    tgz = tmp_path / "out/t.tgz"; data = bytearray(tgz.read_bytes()); data[-5] ^= 0xFF; tgz.write_bytes(bytes(data))
    with pytest.raises(transfer.BundleError, match="sha256"):
        transfer.verify_and_extract("t", tmp_path / "out", tmp_path / "x")


def test_archive_has_no_owner_or_xattr_metadata(tmp_path):
    root = _tree(tmp_path)
    transfer.build("t", transfer.collect("none", ["src"], [], (), root=root), _index(), tmp_path / "out", root=root)
    with tarfile.open(tmp_path / "out/t.tgz") as tar:
        for m in tar.getmembers():
            assert m.uid == 0 and m.gid == 0 and not m.uname and not any("LIBARCHIVE" in k for k in m.pax_headers)


def test_gatekit_allow_list(tmp_path):
    root = tmp_path / "repo"
    rec = root / "data/imajev-bench/v2-lite-v1/records-audited.jsonl"; rec.parent.mkdir(parents=True); rec.write_text(json.dumps({"q": BENCH_TEXT}) + "\n")
    (rec.parent / "assets").mkdir(); (rec.parent / "assets/a.jpg").write_bytes(b"\xff\xd8img")
    idx = _index(); idx.add_string("imajevbench-public", "gold-bearing imajevbench record text that is long enough to index " * 2)
    entries = transfer.collect("gatekit", [], [], (), root=root)
    idx2 = transfer.BenchmarkIndex(min_chars=40); idx2.add_string("imajevbench-public", BENCH_TEXT)
    assert transfer.check_entries(entries, idx2, gatekit=True, root=root)["status"] == "PASS"
    extra = root / "data/other.jsonl"; extra.write_text("{}\n")
    with pytest.raises(transfer.BundleError, match="allow-list"):
        transfer.check_entries(entries + [transfer.Entry(extra, "data/other.jsonl")], idx2, gatekit=True, root=root)
    with pytest.raises(transfer.BundleError):   # JevBench text is still forbidden in a gatekit
        transfer.check_entries(entries, _index(), gatekit=True, root=root)


def test_index_requires_sources(tmp_path):
    with pytest.raises(transfer.BundleError, match="no benchmark sources"):
        transfer.BenchmarkIndex.load(root=tmp_path, pins=tmp_path / "none.json", hf_cache=False)


def test_hf_token_comes_from_env_and_is_never_printed(tmp_path, monkeypatch, capsys):
    secret = "hf_SECRET-token-0123"   # shaped like a token but too short for the release secret scan
    be = transfer.HFBackend("P3_TEST_TOKEN")
    monkeypatch.delenv("P3_TEST_TOKEN", raising=False)
    with pytest.raises(transfer.BundleError, match="P3_TEST_TOKEN"):
        be.fetch("t", "o/r", tmp_path)
    calls = []

    def fake_download(repo, filename, repo_type=None, token=None, local_dir=None, **k):
        calls.append(token); p = Path(local_dir) / filename
        p.write_text(json.dumps({"parts": []}) if filename.endswith(".json") else "x"); return str(p)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("P3_TEST_TOKEN", secret)
    be.fetch("t", "o/r", tmp_path)
    out = capsys.readouterr()
    assert calls == [secret] and secret not in out.out + out.err
    assert "--token" not in (ROOT / "cloud/p3/bootstrap_train.sh").read_text().replace("--token-env", "")


def test_push_refuses_a_bundle_without_a_passing_check(tmp_path):
    (tmp_path / "t.manifest.json").write_text(json.dumps({"check": {"status": "FAIL"}, "parts": []}))
    with pytest.raises(transfer.BundleError, match="no passing check"):
        transfer.HFBackend(dry=True).push("t", tmp_path, "o/r")


@pytest.mark.skipif(not (ROOT / "data/p3/decontam-sources").is_dir(), reason="needs the local benchmark sources (Mac)")
def test_real_code_bundle_passes_the_check():
    entries = transfer.collect("code", [], [], ())
    assert not [e.arc for e in entries if transfer.forbidden_path(e.arc)]
    assert transfer.check_entries(entries, transfer.BenchmarkIndex.load())["status"] == "PASS"


# ------------------------------------------------------------------------------------------------ pilot decision rule
def M(acc, loss, acc2):
    return {"dev_accuracy": acc, "dev_loss": loss, "dev2_accuracy": acc2}


def test_pilot_ordinal_wins_and_256_needs_parity():
    base = M(0.70, 1.00, 0.40)
    d = pilot.decide({"baseline": base, "ordinal": M(0.70, 1.01, 0.45), "codes256": M(0.705, 0.99, 0.40), "both": M(0.70, 1.00, 0.46)}, True, True)
    assert (d["ordinal_weight"], d["readout_codes"]) == (0.15, 256)
    d = pilot.decide({"baseline": base, "ordinal": M(0.70, 1.01, 0.45), "codes256": M(0.705, 0.99, 0.40), "both": M(0.70, 1.00, 0.46)}, False, True)
    assert (d["ordinal_weight"], d["readout_codes"]) == (0.15, 255) and any("parity" in r for r in d["reasons"])
    d = pilot.decide({"baseline": base, "ordinal": M(0.70, 1.01, 0.45), "codes256": M(0.705, 0.99, 0.40), "both": None}, True, False)
    assert d["readout_codes"] == 255 and any("255-option" in r for r in d["reasons"])


def test_pilot_ordinal_loses_on_harm_or_no_gain():
    base = M(0.70, 1.00, 0.40)
    assert pilot.decide({"baseline": base, "ordinal": M(0.68, 1.00, 0.50)}, False, False)["ordinal_weight"] == 0.0   # overall acc -2 pts
    assert pilot.decide({"baseline": base, "ordinal": M(0.70, 1.05, 0.50)}, False, False)["ordinal_weight"] == 0.0   # dev loss +5%
    assert pilot.decide({"baseline": base, "ordinal": M(0.71, 0.99, 0.40)}, False, False)["ordinal_weight"] == 0.0   # score acc flat
    assert pilot.decide({"baseline": base, "ordinal": None}, False, False)["ordinal_weight"] == 0.0
    d = pilot.decide({"baseline": None, "ordinal": M(0.9, 0.5, 0.9)}, True, True)
    assert (d["ordinal_weight"], d["readout_codes"]) == (0.0, 255)


def test_pilot_factors_are_decided_independently_and_combined():
    base = M(0.70, 1.00, 0.40)
    lanes = {"baseline": base, "ordinal": M(0.70, 1.00, 0.44), "codes256": M(0.72, 0.98, 0.40)}
    d = pilot.decide(lanes, True, True)          # two factors win on their own lanes: the full run combines them, no lane to resume
    assert (d["ordinal_weight"], d["readout_codes"], d["expand_lora_rank"]) == (0.15, 256, 0)
    assert d["lane"] is None and d["combined"] and d["winners"] == ["ordinal", "codes256"]
    assert any("combined winners" in r for r in d["reasons"])
    d = pilot.decide(lanes, False, True)         # parity failed: ordinal alone -> its lane is continued
    assert (d["ordinal_weight"], d["readout_codes"], d["lane"], d["combined"]) == (0.15, 255, "ordinal", False)


def test_pilot_lane_metrics_average_last_two_dev_passes(tmp_path):
    entries = [{"step": 0, "dev_loss": 9, "dev_accuracy": 0.1, "dev2_accuracy": 0.1}, {"step": 10, "loss": 1.0, "of": 200},
               {"step": 50, "dev_loss": 1.2, "dev_accuracy": 0.6, "dev2_accuracy": 0.3}, {"step": 150, "dev_loss": 1.0, "dev_accuracy": 0.7, "dev2_accuracy": 0.4},
               {"step": 200, "dev_loss": 0.8, "dev_accuracy": 0.8, "dev2_accuracy": 0.5, "ordinal_loss": 0.05}]
    m = pilot.lane_metrics(entries)
    assert m["dev_loss"] == pytest.approx(0.9) and m["dev_accuracy"] == pytest.approx(0.75) and m["steps"] == [150, 200]
    log = tmp_path / "baseline/train/log.jsonl"; log.parent.mkdir(parents=True); log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    rc = pilot.main(["--pilot-dir", str(tmp_path), "--out", str(tmp_path / "d.json"), "--steps", "200"])
    d = json.loads((tmp_path / "d.json").read_text())
    assert rc == 0 and d["readout_codes"] == 255 and d["lanes"]["baseline"]["last_step"] == 200
    pilot.main(["--pilot-dir", str(tmp_path), "--out", str(tmp_path / "d2.json"), "--steps", "250"])   # crashed before 250 -> missing
    assert json.loads((tmp_path / "d2.json").read_text())["lanes"]["baseline"] is None


# ------------------------------------------------------------------------------------------------ per-type calibration
def _rows(kind, k, n, rng, options=4):
    rows = []
    for i in range(n):
        m = 2 if kind == "boolean" else options
        z = [rng.gauss(0, 1.5) for _ in range(m + 1)]
        p = [math.exp(x) for x in z]; s = sum(p); p = [x / s for x in p]
        t = rng.choices(range(m + 1), weights=p)[0]
        rows.append({"id": f"{kind}-{i}", "decision_type": kind, "option_count": m, "logits": [k * x for x in z], "target_index": t, "correct": True})
    return rows


def test_per_type_temperature_recovers_each_type():
    rng = random.Random(0)
    rows = _rows("choice", 2.0, 1500, rng) + _rows("boolean", 1.5, 1500, rng) + _rows("ordinal", 3.0, 1500, rng, options=5)
    pay = evalck.fit_per_type(rows, "test")
    f = pay["fit"]["types"]
    assert f["choice"]["fitted"] == pytest.approx(2.0, rel=0.12)
    assert f["noul"]["fitted"] == pytest.approx(1.5, rel=0.12)
    assert f["score"]["fitted"] == pytest.approx(3.0, rel=0.12)
    from vision_decision.calibration import TemperatureCalibrator
    cal = TemperatureCalibrator.from_dict(pay)
    assert cal.temperature("choice", 4) == pytest.approx(f["choice"]["shipped"])
    assert cal.temperature("noul", 2) == pytest.approx(f["noul"]["shipped"])
    assert cal.temperature("score", 5) == pytest.approx(f["score"]["shipped"])
    assert cal.temperature("choice", 200) == pytest.approx(f["choice"]["shipped"])      # every bucket of a type


def test_per_type_clamp_and_pooled_fallback():
    rng = random.Random(1)
    rows = _rows("choice", 0.5, 800, rng) + _rows("boolean", 2.0, 10, rng)              # under-confident choice; too few noul rows
    pay = evalck.fit_per_type(rows, "t")
    f = pay["fit"]["types"]
    assert f["choice"]["fitted"] < 0.8 and f["choice"]["shipped"] == 1.0                  # T < 1 ships as 1.0
    assert f["noul"]["source"] == "pooled" and f["noul"]["fitted"] == pytest.approx(pay["fit"]["pooled_fitted"])
    assert f["score"]["rows"] == 0 and f["score"]["source"] == "pooled"


def test_ece_and_pooled_guard(tmp_path):
    assert evalck.ece([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert evalck.ece([0.9, 0.9], [False, False]) == pytest.approx(0.9)
    # an over-confident run: raw pooled ECE is high, a per-type T of 3 pulls it down
    jb = tmp_path / "jb"; rng = random.Random(2); types = {}
    for tier in ("easy", "original", "hard"):
        d = jb / tier; d.mkdir(parents=True); lines = []
        for i in range(150):
            tid = f"{tier}-{i}"; types[tid] = "choice"
            z = [rng.gauss(0, 1) for _ in range(4)]; p = [math.exp(x) for x in z]; s = sum(p); p = [x / s for x in p]
            gold = rng.choices(range(4), weights=p)[0]
            q = [math.exp(3 * x) for x in z]; s = sum(q); q = [x / s for x in q]
            probs = {f"o{j}": q[j] for j in range(4)}; top = max(probs, key=probs.get)
            lines.append(json.dumps({"task_id": tid, "probs": probs, "predicted": top, "correct": top == f"o{gold}"}))
        (d / "results.jsonl").write_text("\n".join(lines) + "\n")
    raw = evalck.pooled_ece_guard(jb, types, None)
    pay = {"temperatures": {f"choice:{b}": 3.0 for b in evalck.BUCKET_LABELS} | {f"boolean:{b}": 1.0 for b in evalck.BUCKET_LABELS}}
    fixed = evalck.pooled_ece_guard(jb, types, pay)
    assert raw["rows"] == 450 and fixed["pooled_ece"] < raw["pooled_ece"] and fixed["pooled_ece"] < 0.06
    choice, payload, why = evalck.choose_calibration({"k": "pt"}, {"k": "au"}, {"pass": True, "pooled_ece": 0.02})
    assert choice == "per_type" and payload == {"k": "pt"}
    choice, payload, why = evalck.choose_calibration({"k": "pt"}, {"k": "au"}, {"pass": False, "pooled_ece": 0.05})
    assert choice == "authored_dev" and payload == {"k": "au"} and "0.05" in why


def test_gates_same_rules_as_p2c():
    ref = json.loads((ROOT / "cloud/p2c_gates_reference.json").read_text())
    R = ref["sizes"]["4b"]
    ok_rows = lambda acc, n=100: [{"correct": i < acc * n / 100} for i in range(n)]
    p2b = [{"target": "unknown", "prediction": "unknown"}] * 14 + [{"target": "a", "prediction": "a"}] * 421
    panels = {"state_probe": ok_rows(R["state_probe"]), "pairs_probe": ok_rows(R["pairs_probe"]), "irrelevance": ok_rows(R["irrelevance"]), "p2b_test": p2b}
    tracks = {"visual": R["visual"], "joint": R["joint"], "text": [100, 100]}
    g = evalck.compute_gates(panels, tracks, ref)
    assert g["shippable"], [c for c in g["checks"] if not c["pass"]]
    assert g["targets"]["joint_items"]["met"] == (R["joint"][0] >= 99)
    tracks_bad = dict(tracks, joint=[R["joint"][0] - 3, R["joint"][1]])
    g2 = evalck.compute_gates(panels, tracks_bad, ref)
    assert not g2["shippable"] and not g2["image_gate"]
    g3 = evalck.compute_gates({}, None, ref)
    assert not g3["shippable"] and all(not c["pass"] for c in g3["checks"])


# ------------------------------------------------------------------------------------------------ DecisionBench subset
def test_decisionbench_allocation_is_equal_with_water_filling():
    counts = {"a": 80, "b": 100, "c": 2222, "d": 30}
    alloc = dbsub.allocate(counts, 300)
    assert sum(alloc.values()) == 300 and alloc["d"] == 30 and all(alloc[k] <= counts[k] for k in counts)
    assert max(alloc["a"], alloc["b"], alloc["c"]) - min(alloc["a"], alloc["b"], alloc["c"]) <= 1 or alloc["a"] == 80
    rows = [{"row_id": f"{k}-{i}", "task_id": k} for k, n in counts.items() for i in range(n)]
    s1, s2 = dbsub.stratified(rows, 300, seed=0), dbsub.stratified(rows, 300, seed=0)
    assert [r["row_id"] for r in s1] == [r["row_id"] for r in s2] and len(s1) == 300
    assert dbsub.allocate({"a": 5}, 10) == {"a": 5}


def test_decisionbench_subset_requires_tracking_dir(tmp_path):
    with pytest.raises(SystemExit):
        dbsub.main(["--out", str(tmp_path / "subset.jsonl"), "--parquet", str(tmp_path / "x.parquet")])


def test_decisionbench_request_mapping_matches_the_harness():
    cands = [{"id": "c2", "label": "High", "description": None, "ordinal_value": 2.0}, {"id": "c0", "label": "Low", "description": "small", "ordinal_value": 0.0}]
    row = {"primitive": "ordinal_scoring", "instruction": "Rate.", "state_json": json.dumps({"s": 1}), "candidates_json": json.dumps(cands)}
    req, order = evalck.decisionbench_request(row)
    assert req["questions"]["decision"] == {"type": "score", "instructions": "Rate.", "criteria": ["Low: small", "High"]} and order == ["c0", "c2"]
    probs = evalck.decisionbench_probs(row, {"probabilities": {"0": 0.3, "1": 0.7}}, order)
    assert probs == [0.7, 0.3]


# ------------------------------------------------------------------------------------------------ manifests, round 2, watcher, soup, steps
def _mrow(i, part, nopt=3, kind="choice"):
    f = {"id": "decision", "type": kind, "question": "q?"}
    if kind == "choice":
        f["options"] = [{"value": f"o{j}"} for j in range(nopt)]
    else:
        f["levels"] = [{"value": j} for j in range(3)]
    return {"id": f"m{i}", "partition": part, "request": {"state": {}, "fields": [f]}, "target": "o0" if kind == "choice" else 1}


def test_prep_manifests_splits_255_rows_and_builds_ordinal_dev(tmp_path):
    man = tmp_path / "manifests"; man.mkdir()
    rows = [_mrow(0, "train"), _mrow(1, "train", 255), _mrow(2, "dev"), _mrow(3, "dev", 255), _mrow(4, "dev", kind="ordinal")]
    (man / "v.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = prep.prep("v", man)
    assert info["rows_255"] == 1 and info["version_255"] == "v-r255" and info["version_256"] == "v-c256" and info["ordinal_dev_rows"] == 1
    ids = lambda v: [json.loads(l)["id"] for l in (man / f"{v}.jsonl").read_text().split("\n") if l.strip()]
    assert ids("v-r255") == ["m0", "m2", "m4"] and ids("v-c256") == ["m0", "m1", "m2", "m4"] and ids("v-ordinal-dev") == ["m4"]
    (man / "w.jsonl").write_text(json.dumps(_mrow(0, "train")) + "\n")
    assert prep.prep("w", man)["version_255"] == "w" and prep.prep("w", man)["rows_255"] == 0


def test_round2_only_train_rows_never_heldout(tmp_path):
    source = [dict(_mrow(i, "train"), candidate_id=f"c{i}") for i in range(6)] + [dict(_mrow(9, "dev"), candidate_id="c9")]
    base = [_mrow(0, "train"), _mrow(1, "train"), _mrow(7, "train"), _mrow(8, "dev")]
    mined = {"c0", "c2", "c3", "c9", "m5"}
    rows, rep = r2.build(mined, source, base, exclude={"c3"}, max_new=10, replay_ratio=1.0, codes=255)
    new = [r for r in rows if r["id"] in ("m0", "m2", "m3", "m5", "m9")]
    assert {r["id"] for r in new} == {"m0", "m2", "m5"}                 # c3 held out, c9 is dev
    assert rep["refused_heldout_overlap"] == 1 and rep["never_trained"] == 2 and rep["retrained"] == 1
    assert [r["id"] for r in rows if r["partition"] == "dev"] == ["m8"]
    f = tmp_path / "s-000.flagged.jsonl"; f.write_text(json.dumps({"id": "c1", "shard": "s"}) + "\n")
    g = tmp_path / "other.jsonl"; g.write_text(json.dumps({"id": "c4", "flag": False}) + "\n" + json.dumps({"id": "c5", "flagged": True}) + "\n")
    assert r2.flagged_ids([str(f), str(g)]) == {"c1", "c5"}


def test_ckpt_watch_snapshots_each_dev_step(tmp_path):
    train, ck = tmp_path / "train", tmp_path / "ckpts"
    (train / "last").mkdir(parents=True)
    for f in ("adapter_model.safetensors", "decision_readout.safetensors", "trainer.pt"):
        (train / "last" / f).write_text(f)
    log = train / "log.jsonl"
    log.write_text(json.dumps({"step": 0, "dev_loss": 1}) + "\n" + json.dumps({"step": 30, "of": 200, "dev_loss": 0.9, "dev_accuracy": 0.5}) + "\n")
    watch.sweep(train, ck, "r1")
    d = ck / "r1-s000030"
    assert d.is_dir() and not (d / "trainer.pt").exists() and json.loads((d / "ckpt.json").read_text())["of"] == 200
    with log.open("a") as f:
        f.write(json.dumps({"step": 35, "loss": 1}) + "\n" + json.dumps({"step": 60, "of": 200, "dev_loss": 0.8}) + "\n")
    watch.sweep(train, ck, "r1"); watch.sweep(train, ck, "r1")
    assert sorted(p.name for p in ck.iterdir()) == ["r1-s000030", "r1-s000060"]


def test_soup_extends_a_256_row_readout():
    torch = pytest.importorskip("torch")
    shipped = {"w": torch.ones(255, 4), "l": torch.zeros(2, 2)}
    new = {"w": torch.cat([torch.full((255, 4), 3.0), torch.full((1, 4), 7.0)]), "l": torch.ones(2, 2)}
    soup = load("cloud/p3/soup.py", "p3_soup")
    out = soup.soup_tensors(shipped, new, 0.5)
    assert out["w"].shape == (256, 4) and float(out["w"][0, 0]) == 2.0 and float(out["w"][255, 0]) == 7.0 and float(out["l"][0, 0]) == 0.5
    with pytest.raises(ValueError):
        soup.soup_tensors({"w": torch.ones(3, 4)}, {"w": torch.ones(5, 4)}, 0.5)


def test_plan_steps_matches_trainer_arithmetic():
    assert plan_steps.steps_for([100, 100], 2.0, 1, 8) == 25
    assert plan_steps.steps_for([100, 100], 1.5, 2, 4) == 19      # round(200 * 1.5 / 2) = 150 microbatches / 8
    assert plan_steps.steps_for([37], 1.0, 1, 8) == 5


def test_final_table_targets(tmp_path):
    base = {"fresh": 40.0, "db_full_equiv": 77.0, "db_ordinal": 40.0, "db_reasoning": 60.0}
    p = {"hard_single_cal": 73.5, "hard_rot4_cal": 74.0, "fresh": 49.0, "db_full_equiv": 79.5, "db_ordinal": 46.0, "db_reasoning": 64.0,
         "gates": True, "soup": False, "joint": 100, "hard_ece_single_cal": 0.07, "pooled_ece_single_cal": 0.028}
    t = {x["target"]: x["met"] for x in final_table.targets(p, base)}
    assert all(v for k, v in t.items() if "reasoning" not in k) and t["DecisionBench reasoning vs shipped"] is False


def test_parity_script_has_a_cuda_torch_path():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/p3/parity_readout256.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--backend {mlx,torch}" in r.stdout
    par = load("scripts/p3/parity_readout256.py", "p3_parity")
    assert hasattr(par, "TorchEngine") and callable(par.TorchEngine.score)


def test_prep_manifests_uses_the_builders_size_matched_255_lane(tmp_path):
    man = tmp_path / "manifests"; man.mkdir()
    rows = [_mrow(0, "train"), _mrow(1, "train", 255), _mrow(2, "dev")]
    lane = [_mrow(0, "train"), _mrow(5, "train"), _mrow(2, "dev")]           # the builder refilled the 255-option slot
    (man / "v.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (man / "v-lane255.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lane))
    info = prep.prep("v", man)
    assert info["lane255_prebuilt"] and info["version_255"] == "v-r255"
    ids = lambda v: [json.loads(l)["id"] for l in (man / f"{v}.jsonl").read_text().split("\n") if l.strip()]
    assert ids("v-r255") == ["m0", "m5", "m2"] and ids("v-c256") == ["m0", "m1", "m2"]
    (man / "v-lane255.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lane + [_mrow(6, "train", 255)]))
    with pytest.raises(SystemExit):
        prep.prep("v", man)


def test_bootstrap_mine_defaults_to_gcs_phase3_and_marks_workers_running():
    s = (ROOT / "cloud/p3/bootstrap_mine.sh").read_text()
    assert "BACKEND=${BACKEND:-gcs}" in s and "GCS_PREFIX=${GCS_PREFIX:-phase3}" in s and "gs://<bucket>/" in s
    assert "WORKERS_RUNNING" in s and "MINE_OUT=${MINE_OUT:-data/p3/mine}" in s
    t = (ROOT / "cloud/p3/bootstrap_train.sh").read_text()
    assert "BACKEND=${BACKEND:-gcs}" in t and "GCS_PREFIX=${GCS_PREFIX:-phase3}" in t
    tr = (ROOT / "scripts/p3/transfer.py").read_text()
    assert 'p.add_argument("--prefix", default="phase3"' in tr
    # the coordinator pulls from where the workers write
    sc = load("scripts/p3/stream_coordinator.py", "p3_stream_coordinator")
    assert sc.parser().parse_args(["--no-pull"]).remote_dir == "data/p3/mine"


# ------------------------------------------------------------------------------------------------ pilot lane -> full run (resume)
lane_resume = load("cloud/p3/lane_resume.py", "p3_lane_resume")
eval_lane = load("cloud/p3/eval_lane.py", "p3_eval_lane")


def _pilot_lane(o: Path, lane: str, world: int = 2, acc: int = 4, step: int = 100, w: float = 0.0, codes: int = 255, done: bool = True):
    t = o / "pilot" / lane / "train"; (t / "last").mkdir(parents=True)
    for f in ("trainer.pt", "adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json"):
        (t / "last" / f).write_text("x")
    cfg = {"world_size": world, "accumulate": acc, "epochs": 2}
    if w:
        cfg["ordinal_weight"] = w
    if codes != 255:
        cfg["readout_codes"] = codes          # the trainer records recipe flags only when they differ from the default
    (t / "config.json").write_text(json.dumps(cfg))
    (t / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in
                                         [{"step": 0, "dev_loss": 1.0}, {"step": step - 1, "loss": 1}, {"step": step, "of": 3200, "dev_loss": 0.9}]))
    if done:
        (o / f"pilot-{lane}.DONE").touch()
    return t / "last"


def test_pilot_decision_names_the_winning_lane():
    base = {"dev_loss": 1.0, "dev_accuracy": 0.60, "dev2_accuracy": 0.40}
    d = pilot.decide({"baseline": base, "ordinal": {"dev_loss": 1.0, "dev_accuracy": 0.60, "dev2_accuracy": 0.45}}, False, False)
    assert d["lane"] == "ordinal" and (d["ordinal_weight"], d["readout_codes"]) == (0.15, 255)
    assert pilot.decide({}, False, False)["lane"] == "baseline"
    r64 = {"dev_loss": 0.95, "dev_accuracy": 0.63, "dev2_accuracy": 0.40}
    d = pilot.decide({"baseline": base, "rank64": r64}, True, True)
    assert d["lane"] == "rank64" and (d["expand_lora_rank"], d["lora_rank"], d["readout_codes"]) == (64, 64, 255)


def test_lane_resume_exact_on_8_gpus_and_inexact_with_the_eval_lane(tmp_path):
    src = _pilot_lane(tmp_path, "ordinal", w=0.15)
    dec = {"ordinal_weight": 0.15, "readout_codes": 255, "lane": "ordinal"}
    r = lane_resume.plan(tmp_path / "pilot", dec, 100, 8, 1)          # 2 GPUs x 4 = 8 GPUs x 1
    assert r["ok"] and r["mode"] == "exact" and r["flags"] == ["--resume-from", str(src)]
    assert r["start_step"] == 100 and r["source_microbatches"] == 800 and r["repeated_microbatches"] == 0 and r["skipped_microbatches"] == 0
    r = lane_resume.plan(tmp_path / "pilot", dec, 100, 7, 1)          # EVAL_DURING_TRAIN=1: 7 training GPUs
    assert r["ok"] and r["mode"] == "inexact" and r["flags"][-1] == "--resume-inexact"
    assert r["start_step"] == 114 and r["repeated_microbatches"] == 2 and r["repeated_microbatches"] < 7     # 800 = 114 x 7 + 2
    # a lane retry that ran past PILOT_STEPS is still a prefix of the full run: used, with a note
    src2 = _pilot_lane(tmp_path, "baseline", step=130)
    r = lane_resume.plan(tmp_path / "pilot", {"ordinal_weight": 0.0, "readout_codes": 255}, 100, 8, 1)
    assert r["ok"] and r["lane"] == "baseline" and r["start_step"] == 130 and "note" in r and r["flags"][1] == str(src2)


@pytest.mark.parametrize("breakage", ["no_done", "no_trainer_state", "switches", "no_dev", "unknown_lane"])
def test_lane_resume_falls_back_when_anything_is_missing(tmp_path, breakage):
    src = _pilot_lane(tmp_path, "codes256", codes=256, done=breakage != "no_done")
    dec = {"ordinal_weight": 0.0, "readout_codes": 256, "lane": "codes256"}
    if breakage == "no_trainer_state":
        (src / "trainer.pt").unlink()
    if breakage == "switches":
        (src.parent / "config.json").write_text(json.dumps({"world_size": 2, "accumulate": 4}))    # a 255-code lane under that name
    if breakage == "no_dev":
        (src.parent / "log.jsonl").write_text(json.dumps({"step": 0, "dev_loss": 1}) + "\n")
    if breakage == "unknown_lane":
        dec = {"ordinal_weight": 0.3, "readout_codes": 255}
    r = lane_resume.plan(tmp_path / "pilot", dec, 100, 8, 1)
    assert not r["ok"] and r["flags"] == [] and "FALLBACK" in r["summary"]


def test_lane_resume_cli_prints_only_the_flags(tmp_path, capsys):
    src = _pilot_lane(tmp_path, "ordinal", w=0.15)
    (tmp_path / "d.json").write_text(json.dumps({"ordinal_weight": 0.15, "readout_codes": 255, "lane": "ordinal"}))
    args = ["--pilot-dir", str(tmp_path / "pilot"), "--decision", str(tmp_path / "d.json"), "--steps", "100", "--world", "8", "--accumulate", "1",
            "--out", str(tmp_path / "r.json")]
    assert lane_resume.main(args) == 0
    assert capsys.readouterr().out.strip() == f"--resume-from {src}" and json.loads((tmp_path / "r.json").read_text())["mode"] == "exact"
    (tmp_path / "d.json").write_text("not json")                       # any surprise: empty flags, exit 0 (the old path)
    assert lane_resume.main(args) == 0 and capsys.readouterr().out.strip() == ""
    assert "FALLBACK" in json.loads((tmp_path / "r.json").read_text())["summary"]


def test_pod_run_pilot_lanes_match_the_full_run_step():
    s = (ROOT / "cloud/p3/pod_run_train.sh").read_text()
    assert "PER_STEP=$((TRAIN_WORLD*ACC)); PILOT_ACC=${PILOT_ACC:-$(( (PER_STEP+1)/2 ))}" in s
    assert "PILOT_STEPS=${PILOT_STEPS:-100}" in s and "--max-steps $PILOT_STEPS --dev-every $PILOT_DEV_EVERY" in s
    lane_block = s[s.index("lane(){"):s.index("if ! skip pilot")]
    full_block = s[s.index("train_run(){"):s.index("ckpt_meta(){")]
    # the lanes and the full run share every flag that shapes the data order or the optimisation
    for flag in ("--seed 0", "--pad-multiple 64 --token-budget $TOKEN_BUDGET", "--batch-size $BATCH", "$(ckflag) $RECIPE", "$DEV2"):
        assert flag in lane_block and flag in full_block, flag
    assert "--init-adapter $SHIPPED --epochs $EPOCHS --lr $LR --warmup $WARMUP --seed 0" in lane_block
    assert "--init-adapter $init --epochs $ep --lr $lr --warmup $wu --seed 0" in full_block
    assert "train_run r1 $VFULL $SHIPPED $EPOCHS $LR $WARMUP" in s
    assert 'lane_resume.py --pilot-dir $O/pilot --decision $O/pilot-decision.json --steps $PILOT_STEPS --world $TRAIN_WORLD' in s
    assert "EVAL_DURING_TRAIN=${EVAL_DURING_TRAIN:-0}" in s and "RESUME_LANE=${RESUME_LANE:-1}" in s


def _train_run_harness(tmp_path: Path, trainer_body: str) -> subprocess.CompletedProcess:
    """Run the pod script's own train_run() with stubbed python / torchrun (no GPU): the fallback path."""
    s = (ROOT / "cloud/p3/pod_run_train.sh").read_text()
    funcs = s[s.index("cleanup_train(){"):s.index("ckpt_meta(){")]
    o = tmp_path / "run"; (o / "ckpts/r1-s000010").mkdir(parents=True)
    harness = f'''set -uo pipefail
O={o}; TRAIN_GPUS=(0 1 2 3 4 5 6 7); TRAIN_WORLD=8; TOKEN_BUDGET=16384; BATCH=40; ACC=1; M4=m; DEV2=""; CK_FLAG=""; RECIPE="--x"; W=0.0; CODES=255
log(){{ echo "LOG $*"; }}; stage(){{ log "stage $1"; }}; skip(){{ [ -f $O/$1.DONE ]; }}; csv(){{ local IFS=,; echo "$*"; }}
python(){{
  case "$*" in
    *plan_steps.py*) echo '{{"steps": 40, "dev_every": 10}}';;
    *ckpt_watch.py*) return 0;;
    *"-c import json,sys"*) command python3 "$@";;
    *torch.distributed.run*) echo "$*" >> $O/calls.txt; {trainer_body};;
    *) command python3 "$@";;
  esac; }}
{funcs}
cleanup_train(){{ :; }}
train_run r1 v /shipped 2 2e-5 10 0.25 --resume-from $O/pilot/ordinal/train/last && echo TRAIN_OK
'''
    (tmp_path / "h.sh").write_text(harness)
    return subprocess.run(["bash", str(tmp_path / "h.sh")], capture_output=True, text=True, timeout=60)


def test_train_run_falls_back_to_the_restart_path_when_the_resume_fails(tmp_path):
    # the resumed trainer exits before its first save (e.g. a config mismatch); the plain restart then succeeds
    body = 'out=$(echo "$*" | sed -E "s/.*--output ([^ ]+).*/\\1/"); mkdir -p $out; case "$*" in *--resume-from*) return 1;; *) return 0;; esac'
    r = _train_run_harness(tmp_path, body)
    assert "TRAIN_OK" in r.stdout, r.stdout + r.stderr
    assert "RESUME_FALLBACK" in r.stdout
    calls = (tmp_path / "run/calls.txt").read_text().strip().split("\n")
    assert len(calls) == 2 and "--resume-from" in calls[0] and "--resume-from" not in calls[1]
    assert "CUDA_VISIBLE_DEVICES" not in calls[0] and "--nproc_per_node=8" in calls[0]
    assert json.loads((tmp_path / "run/r1/resume-fallback.json").read_text())["fallback"] is True
    assert (tmp_path / "run/train-r1.DONE").exists()


def test_train_run_keeps_the_resumed_trajectory_after_a_later_crash(tmp_path):
    # the resume succeeded (resumed_from.json written), then the run crashed once: the retry keeps --resume-from (the trainer
    # resumes from its own last/ first), and no fallback happens
    body = ('out=$(echo "$*" | sed -E "s/.*--output ([^ ]+).*/\\1/"); mkdir -p $out; echo {} > $out/resumed_from.json; '
            '[ -f $O/crashed ] && return 0; touch $O/crashed; return 1')
    r = _train_run_harness(tmp_path, body)
    assert "TRAIN_OK" in r.stdout and "RESUME_FALLBACK" not in r.stdout, r.stdout + r.stderr
    calls = (tmp_path / "run/calls.txt").read_text().strip().split("\n")
    assert len(calls) == 2 and all("--resume-from" in c for c in calls)


# ------------------------------------------------------------------------------------------------ eval lane (EVAL_DURING_TRAIN=1)
def _snap(ckpts: Path, name: str, step: int, of: int):
    d = ckpts / name; d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_text("x"); (d / "ckpt.json").write_text(json.dumps({"step": step, "of": of}))


def test_eval_lane_queue_order_and_final_snapshots(tmp_path):
    ck, st = tmp_path / "ckpts", tmp_path / "lane"
    _snap(ck, "r1-s000800", 800, 3200); _snap(ck, "r1-s001600", 1600, 3200); _snap(ck, "r1-s003200", 3200, 3200)
    _snap(ck, "r2-s000100", 100, 300); _snap(ck, "r2-s000300", 300, 300)
    (ck / "r1-s002400.tmp").mkdir(); (ck / "soup-r1-s001600-w50").mkdir(); (ck / "r1-s002400").mkdir()   # in the making / soups / no ckpt.json
    # shipped first; then intermediate snapshots by (round, step); the last snapshot of each round is left to the pipeline
    assert eval_lane.eligible(ck, set()) == ["shipped", "r1-s000800", "r1-s001600", "r2-s000100"]
    assert eval_lane.eligible(ck, {1}) == ["shipped", "r1-s000800", "r1-s001600", "r1-s003200", "r2-s000100"]
    marker = tmp_path / "sel-r1.DONE"
    assert eval_lane.final_rounds([f"r1={marker}"]) == set(); marker.touch(); assert eval_lane.final_rounds([f"r1={marker}"]) == {1}
    got = [eval_lane.claim_next(st, ck, set()) for _ in range(5)]
    assert got == ["shipped", "r1-s000800", "r1-s001600", "r2-s000100", None]      # each name claimed once
    assert eval_lane.running(st) == ["r1-s000800", "r1-s001600", "r2-s000100", "shipped"]
    m = eval_lane.meta("r2-s000100", ck)
    assert (m["round"], m["step"], m["of"]) == (2, 100, 300) and abs(m["order"] - (2 + 100 / 300)) < 1e-12


def test_eval_lane_pause_stop_status_left_and_reset(tmp_path):
    ck, st = tmp_path / "ckpts", tmp_path / "lane"
    _snap(ck, "r1-s000800", 800, 3200); _snap(ck, "r1-s001600", 1600, 3200)
    assert eval_lane.claim_next(st, ck, set()) == "shipped"
    assert eval_lane.set_flag(st, "PAUSE") == ["shipped"]                          # pause reports what is still running
    assert eval_lane.claim_next(st, ck, set()) is None
    eval_lane.set_flag(st, "PAUSE", on=False)
    assert eval_lane.claim_next(st, ck, set()) == "r1-s000800"
    (st / "shipped.ok").write_text("{}"); (st / "r1-s000800.failed").write_text("{}")
    assert [eval_lane.status(st, n) for n in ("shipped", "r1-s000800", "r1-s001600")] == ["ok", "failed", "unclaimed"]
    assert eval_lane.claim_next(st, ck, set()) == "r1-s001600"
    assert eval_lane.set_flag(st, "STOP") == ["r1-s001600"] and eval_lane.claim_next(st, ck, set()) is None
    # final queue: everything that is neither done nor still running (a failed lane eval is redone)
    assert eval_lane.left(st, ["shipped", "r1-s000800", "r1-s001600", "r1-s003200", "r2-s000300"]) == ["r1-s000800", "r1-s003200", "r2-s000300"]
    # once the lane process has ended, a claim without an outcome (the lane died mid-way) is evaluated again
    assert eval_lane.left(st, ["shipped", "r1-s001600"], lane_exited=True) == ["r1-s001600"]
    eval_lane.reset(st)                                                             # a new lane forgets unfinished claims, keeps ok
    assert eval_lane.status(st, "shipped") == "ok" and eval_lane.status(st, "r1-s000800") == "unclaimed"
    assert eval_lane.status(st, "r1-s001600") == "unclaimed" and not (st / "STOP").exists()


def test_eval_lane_ok_needs_summary_and_no_failed_panel(tmp_path):
    out = tmp_path / "e"; out.mkdir()
    assert not eval_lane.eval_ok(out)
    (out / "summary.json").write_text("{}"); (out / "status.json").write_text(json.dumps({"heldout_fresh": {"ok": True}}))
    assert eval_lane.eval_ok(out)
    (out / "status.json").write_text(json.dumps({"heldout_fresh": {"ok": True}, "imajevbench": {"ok": False}}))
    assert not eval_lane.eval_ok(out)


FAKE_EVAL = '''import json, sys
from pathlib import Path
a = sys.argv[1:]; get = lambda k: a[a.index(k) + 1]
out = Path(get("--out")); out.mkdir(parents=True, exist_ok=True)
Path(sys.argv[0]).with_name("calls.jsonl").open("a").write(json.dumps(a) + "\\n")
if get("--name") == "r1-s000800":
    sys.exit(3)                                   # a failing eval
(out / "summary.json").write_text("{}"); (out / "status.json").write_text("{}")
'''


def test_eval_lane_run_evaluates_until_stop(tmp_path):
    ck, st, ev = tmp_path / "ckpts", tmp_path / "lane", tmp_path / "eval"
    _snap(ck, "r1-s000800", 800, 3200); _snap(ck, "r1-s001600", 1600, 3200); _snap(ck, "r1-s003200", 3200, 3200)
    fake = tmp_path / "fake_eval.py"; fake.write_text(FAKE_EVAL)
    st.mkdir(); (st / "STOP").touch()     # reset() clears a stale STOP at start; STOP is set again below, after the first poll
    import threading
    t = threading.Timer(1.5, lambda: (st / "STOP").touch()); t.start()
    rc = eval_lane.main(["run", "--state", str(st), "--ckpts", str(ck), "--eval-root", str(ev), "--gpu", "7", "--shipped", str(tmp_path / "shipped"),
                         "--poll", "0.2", "--eval-args", "--model /m --tracking-root /t/tracking-only", "--eval-cmd", f"{shlex.quote(sys.executable)} {shlex.quote(str(fake))}"])
    t.cancel()
    assert rc == 0
    calls = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().split("\n") if l.strip()]
    names = [c[c.index("--name") + 1] for c in calls]
    assert names == ["shipped", "r1-s000800", "r1-s001600"]                                   # never the final r1 snapshot
    shipped = calls[0]
    assert shipped[shipped.index("--ckpt") + 1] == str(tmp_path / "shipped") and "--fixed-calibration" in shipped
    assert shipped[shipped.index("--port") + 1] == "8772" and shipped[shipped.index("--panels") + 1] == "all" and "--model" in shipped
    c = calls[2]
    assert (c[c.index("--round") + 1], c[c.index("--step") + 1]) == ("1", "1600") and float(c[c.index("--order") + 1]) == 1.5
    assert [eval_lane.status(st, n) for n in names] == ["ok", "failed", "ok"]
    assert eval_lane.left(st, ["shipped", "r1-s000800", "r1-s001600", "r1-s003200"]) == ["r1-s000800", "r1-s003200"]


def test_eval_lane_exits_when_the_pod_script_is_gone(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    rc = eval_lane.main(["run", "--state", str(tmp_path / "lane"), "--ckpts", str(tmp_path / "ckpts"), "--eval-root", str(tmp_path / "eval"),
                         "--gpu", "0", "--shipped", "/s", "--parent-pid", str(p.pid), "--poll", "0.1"])
    assert rc == 0 and not (tmp_path / "eval").exists()


def test_pod_run_eval_lane_wiring():
    s = (ROOT / "cloud/p3/pod_run_train.sh").read_text()
    # lane mode: training leaves the lane GPU out; the selection stage pauses the lane; the final queue covers only what is left
    assert 'TRAIN_GPUS=($(for g in "${ALL_GPUS[@]}"; do [ "$g" = "$LANE_GPU" ] || echo $g; done))' in s
    assert "CUDA_VISIBLE_DEVICES=$(csv \"${TRAIN_GPUS[@]}\") python -m torch.distributed.run --nproc_per_node=$TRAIN_WORLD" in s
    assert "eval_lane.py pause --state $O/lane" in s and "eval_lane.py resume --state $O/lane" in s
    assert "eval_lane.py stop --state $O/lane" in s and "eval_lane.py left --state $O/lane" in s
    assert "--final-ok r1=$O/sel-r1.DONE --parent-pid $$" in s
    assert 'for g in "${TRAIN_GPUS[@]}"; do\n      cmd=${MINE_CMD' in s                          # re-mine leaves the lane GPU alone
    assert "[ -d $O/lane ] && touch $O/lane/STOP" in s                                          # die() stops the lane
    # the lane never reads the tracking root's DecisionBench files: it only passes --tracking-root on, as eval_one does
    lane_src = (ROOT / "cloud/p3/eval_lane.py").read_text().lower()
    assert "decisionbench" not in lane_src.replace("decisionbench tracking", "")


def test_hard_linked_files_are_stored_and_extracted_as_plain_files(tmp_path):
    """data/decision-v1 image dirs are hard-link farms: the same inode under two arcnames must not become a tar link member
    (the 2026-09-26 p3-train fetch failed on exactly this), and an older archive holding an in-archive hard link must still extract."""
    import os, tarfile
    root = _tree(tmp_path)
    a = root / "data/a/img.jpg"; a.parent.mkdir(parents=True); a.write_bytes(b"jpegbytes" * 100)
    b = root / "data/b/img.jpg"; b.parent.mkdir(parents=True); os.link(a, b)
    entries = transfer.collect("none", ["src", "data"], [], (), root=root)
    m = transfer.build("t", entries, _index(), tmp_path / "out", root=root)
    with tarfile.open(tmp_path / "out/t.tgz") as tar:
        members = {x.name: x for x in tar.getmembers()}
    assert members["data/b/img.jpg"].isreg() and members["data/b/img.jpg"].size == a.stat().st_size and not members["data/b/img.jpg"].islnk()
    dest = tmp_path / "x"; transfer.verify_and_extract("t", tmp_path / "out", dest)
    assert (dest / "data/b/img.jpg").read_bytes() == a.read_bytes()
    # an archive with an in-archive hard link (built by the old code) extracts; one linking outside the tree is refused
    for linkname, ok in (("data/a/img.jpg", True), ("../outside", False)):
        out = tmp_path / f"old-{ok}"; out.mkdir(); tgz = out / "t.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(a, arcname="data/a/img.jpg")
            ti = tarfile.TarInfo("data/b/img.jpg"); ti.type = tarfile.LNKTYPE; ti.linkname = linkname; tar.addfile(ti)
        sha = transfer.sha256_file(tgz)
        (out / "t.manifest.json").write_text(json.dumps({"name": "t", "gatekit": False, "archive_sha256": sha, "archive_bytes": tgz.stat().st_size,
            "parts": [{"file": "t.tgz", "sha256": sha, "bytes": tgz.stat().st_size}], "check": {"status": "PASS"}, "files": []}))
        if ok:
            transfer.verify_and_extract("t", out, out / "x"); assert (out / "x/data/b/img.jpg").read_bytes() == a.read_bytes()
        else:
            with pytest.raises(transfer.BundleError, match="unsafe hard link"):
                transfer.verify_and_extract("t", out, out / "x")
