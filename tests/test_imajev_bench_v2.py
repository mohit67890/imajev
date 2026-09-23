import json
import math

import pytest
from PIL import Image

from imajev_bench.assemble import assemble
from imajev_bench.cli import main, read_jsonl
from imajev_bench.harness import NEUTRAL_UNKNOWN, build_question, run_local
from imajev_bench.lint import lint
from imajev_bench.release import annotator_agreement, release_check
from imajev_bench.runner import verify_run
from imajev_bench.schema import validate_records
from imajev_bench.scoring import score
from imajev_bench.stats import (cluster_bootstrap, constant_baseline_contrast, contrast_metrics, evidence_clusters,
                                paired_cluster_test, required_clusters)
from vision_decision.contracts import ChoiceField


def text_record(item_id, gold=True, group=None, question="Is the count above 2?", family="policy", contrast=None,
                split="dev"):
    provenance = {"contrast": contrast} if contrast else {}
    return {"id": item_id, "group_id": group or f"g-{item_id}", "track": "text", "family": family, "split": split,
            "images": [], "request": {"request_id": item_id, "state": {"count": 3}, "fields": [
                {"id": "decision", "type": "boolean", "question": question}]},
            "gold": gold, "annotation_status": "draft", "provenance": provenance}


def image(path, color):
    Image.new("RGB", (32, 24), color).save(path)
    return path


# --- statistics -----------------------------------------------------------------------------------

def test_clusters_join_shared_images_sources_and_contrast_sets():
    a = text_record("a", contrast={"set_id": "s1", "role": "original"})
    b = text_record("b", False, contrast={"set_id": "s1", "role": "variant", "relation": "change"})
    c = text_record("c")
    c["images"] = [{"path": "x.png", "sha256": "0" * 64}]
    d = text_record("d")
    d["images"] = [{"path": "y.png", "sha256": "0" * 64}]
    e = text_record("e")
    e["provenance"]["source_clusters"] = ["scene-1"]
    f = text_record("f")
    f["provenance"]["source_cluster"] = "scene-1"
    clusters = evidence_clusters([a, b, c, d, e, f])
    assert clusters["a"] == clusters["b"] and clusters["c"] == clusters["d"] and clusters["e"] == clusters["f"]
    assert len(set(clusters.values())) == 3


def test_cluster_bootstrap_suppresses_small_cluster_counts():
    correct = {str(i): i % 2 == 0 for i in range(30)}
    clusters = {str(i): f"c{i % 3}" for i in range(30)}
    report = cluster_bootstrap(correct, clusters)
    assert report["clusters"] == 3 and report["low"] is None and "suppressed_reason" in report
    wide = cluster_bootstrap(correct, {k: k for k in correct}, samples=500)
    assert wide["low"] < 0.5 < wide["high"]


def test_paired_test_counts_clusters_not_rows():
    ids = [str(i) for i in range(20)]
    a = {i: True for i in ids}
    b = {i: int(i) >= 10 for i in ids}
    one_cluster = paired_cluster_test(a, b, {i: "same" for i in ids})
    assert one_cluster["discordant_clusters"] == 1 and one_cluster["p_value"] == 1.0
    independent = paired_cluster_test(a, b, {i: i for i in ids})
    assert independent["a_only_correct"] == 10 and independent["p_value"] < 0.01
    assert independent["low"] > 0


def test_contrast_metrics_give_constant_answers_no_change_credit():
    records = [text_record("o", True, contrast={"set_id": "s", "role": "original"}),
               text_record("flip", False, contrast={"set_id": "s", "role": "variant", "relation": "change"}),
               text_record("same", True, contrast={"set_id": "s", "role": "variant", "relation": "same"})]
    constant = contrast_metrics(records, {"o": True, "flip": True, "same": True})
    assert constant["constant_sets"] == 1 and constant["change_pairs_correct"] == 0 and constant["same_unchanged"] == 1
    assert constant["all_correct"] == 0
    perfect = contrast_metrics(records, {"o": {"status": "answered", "value": True},
                                         "flip": {"status": "answered", "value": False},
                                         "same": {"status": "answered", "value": True}})
    assert perfect["all_correct"] == 1 and perfect["change_pairs_correct"] == 1
    assert constant_baseline_contrast(records)["best_constant_row_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        contrast_metrics([records[1]], {})


def test_power_analysis_grows_with_clustering():
    flat = required_clusters(0.05, 0.25, 1, 0.0)
    clustered = required_clusters(0.05, 0.25, 4, 0.3)
    assert flat["effective_items"] == clustered["effective_items"]
    assert clustered["items"] > flat["items"]
    assert math.isclose(clustered["design_effect"], 1.9)


# --- lint -----------------------------------------------------------------------------------------

def test_lint_flags_lexical_unknown_cue_and_family_majority():
    records = [text_record(f"u{i}", None, question="Which? Return unknown if not printed.", family="gap")
               for i in range(10)]
    records += [text_record(f"a{i}", True, family="rule") for i in range(10)]
    report = lint(records)
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["abstention_cues"]["status"] == "fail"
    assert checks["abstention_cues"]["cue_only_unknown_detection_accuracy"] == 1.0
    assert checks["label_balance"]["status"] == "fail"
    assert report["status"] == "fail"


def test_lint_passes_balanced_uncued_records_and_checks_contrast_relations():
    records = [text_record(f"r{i}", i % 2 == 0, family="rule") for i in range(12)]
    records += [text_record("u1", None, question="Is the count above 7?"),
                text_record("u2", None, question="Is the count above 8?")]
    checks = {c["check"]: c for c in lint(records)["checks"]}
    assert checks["abstention_cues"]["status"] == "pass"
    assert checks["label_balance"]["status"] == "pass"
    wrong = [text_record("o", True, contrast={"set_id": "s", "role": "original"}),
             text_record("v", True, contrast={"set_id": "s", "role": "variant", "relation": "change"})]
    assert {c["check"]: c for c in lint(wrong)["checks"]}["contrast_sets"]["status"] == "fail"


def test_lint_image_reuse_split_isolation_and_provenance(tmp_path):
    path = image(tmp_path / "a.png", "red")
    sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    records = []
    for i in range(4):
        r = text_record(f"v{i}", split="dev" if i < 3 else "test")
        r.update(track="visual", images=[{"path": "a.png", "sha256": sha}])
        records.append(r)
    checks = {c["check"]: c for c in lint(records, tmp_path)["checks"]}
    assert checks["image_reuse"]["status"] == "fail"
    assert checks["split_isolation"]["status"] == "fail"
    assert checks["provenance"]["status"] == "fail"
    assert checks["near_duplicates"]["status"] == "pass"


# --- harness --------------------------------------------------------------------------------------

class FakeBackend:
    """Prefers the candidate text containing 'yes'; records the prompts it saw."""

    def __init__(self):
        self.prompts, self.images_seen, self.resets = [], [], 0

    def labels(self, header, count, n_images):
        return [chr(65 + i) for i in range(count)]

    def score(self, images, prompt, labels, choices):
        self.prompts.append(prompt)
        self.images_seen.append(len(images))
        lines = prompt.split("\n")[-len(labels):]
        return [2.0 if ": yes" in line else 0.0 for line in lines], {"preprocess_seconds": 0.01, "forward_seconds": 0.02}

    def reset_peak_memory(self):
        self.resets += 1

    def peak_memory_bytes(self):
        return 123

    def describe(self):
        return {"name": "fake"}


def test_neutral_profile_removes_premise_escape_hatch():
    field = ChoiceField(id="d", type="choice", question="Which?", options=[{"value": "a"}, {"value": "b"}])
    _, _, native = build_question(field, {}, "imajev-native")
    _, _, neutral = build_question(field, {}, "benchmark-neutral")
    assert "premise is false" in native[-1]
    assert neutral[-1] == NEUTRAL_UNKNOWN and "premise" not in neutral[-1]


def test_harness_run_is_scorable_and_uses_full_rotations(tmp_path):
    img = image(tmp_path / "p.png", "blue")
    sha = __import__("hashlib").sha256(img.read_bytes()).hexdigest()
    record = text_record("v1")
    record.update(track="visual", images=[{"path": "p.png", "sha256": sha}])
    backend = FakeBackend()
    busy = iter([{"busy": []}, {"busy": [{"pid": 1, "pcpu": 90.0, "command": "other"}]}])
    out = run_local([record], tmp_path, tmp_path / "run", backend, warmup=0, repeats=2, monitor=lambda: next(busy))
    row = read_jsonl(out)[0]
    assert row["value"] is True and row["rotation_agreement"] == 1.0
    assert row["timing_contaminated"] is True and len(row["latency_repeats_ms"]) == 2
    assert len(backend.prompts) == 2 * 3  # full rotations over yes/no/unknown, two timing repeats
    assert backend.resets == 1
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["profile"] == "benchmark-neutral" and manifest["rotations"] == "full"
    assert (tmp_path / "run" / "pip-freeze.txt").exists()
    verify_run([record], out)
    report = score([record], {row["id"]: row}, 0)
    assert report["capability"]["all_records"]["correct"] == 1
    assert report["probability_quality"]["valid"] == 1


def test_harness_no_image_condition_drops_images_only(tmp_path):
    img = image(tmp_path / "p.png", "blue")
    sha = __import__("hashlib").sha256(img.read_bytes()).hexdigest()
    record = text_record("v1")
    record.update(track="visual", images=[{"path": "p.png", "sha256": sha}])
    backend = FakeBackend()
    run_local([record], tmp_path, tmp_path / "blind", backend, condition="no_image", warmup=0, repeats=1,
              monitor=lambda: {"busy": []})
    assert set(backend.images_seen) == {0}
    assert all('"count": 3' in prompt for prompt in backend.prompts)


# --- assembly and release -------------------------------------------------------------------------

def spec_for(tmp_path, items_per_image=2):
    sources, items = [], []
    for s in range(6):
        image(tmp_path / f"s{s}.png", (s * 40, 0, 0))
        sources.append({"id": f"src{s}", "path": f"s{s}.png", "source_cluster": f"scene-{s}",
                        "provenance": {"source": "test", "source_page": "https://example.org", "creator": "me",
                                       "license": "CC0 1.0"}})
        for q in range(items_per_image):
            items.append({"id": f"item-{s}-{q}", "track": "visual", "family": "reading", "images": [f"src{s}"],
                          "state": {}, "field": {"id": "decision", "type": "boolean", "question": f"Q{q}?"},
                          "draft_gold": q % 2 == 0, "evidence": "visible"})
    return {"dataset": "t", "version": "0", "split_salt": "salt", "sources": sources, "items": items,
            "split_fractions": {"dev": 0.5, "calibration": 0.0, "test": 0.5}}


def test_assemble_assigns_whole_clusters_to_one_split(tmp_path):
    receipt = assemble(spec_for(tmp_path), tmp_path, tmp_path / "out")
    records = read_jsonl(tmp_path / "out" / "records.jsonl")
    assert receipt["records"] == 12 and receipt["evidence_clusters"] == 6
    by_scene = {}
    for r in records:
        by_scene.setdefault(r["provenance"]["source_clusters"][0], set()).add(r["split"])
        assert r["annotation_status"] == "draft" and r["images"][0]["path"].startswith("assets/")
    assert all(len(splits) == 1 for splits in by_scene.values())
    again = assemble(spec_for(tmp_path), tmp_path, tmp_path / "out2")
    assert again["split_records"] == receipt["split_records"]


def test_assemble_enforces_reuse_cap(tmp_path):
    with pytest.raises(ValueError, match="max_records_per_image"):
        assemble(spec_for(tmp_path, items_per_image=4), tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_release_check_blocks_draft_small_and_unverified_datasets(tmp_path):
    assemble(spec_for(tmp_path), tmp_path, tmp_path / "out")
    records = read_jsonl(tmp_path / "out" / "records.jsonl")
    report = release_check(records, tmp_path / "out")
    failed = {g["gate"] for g in report["gates"] if not g["passed"]}
    assert report["release_ready"] is False
    assert {"human_review", "independent_clusters", "image_necessity"} <= failed
    agreement = next(g for g in report["gates"] if g["gate"] == "annotator_agreement")
    assert agreement["passed"] and agreement["detail"].startswith("Not applicable")


def test_annotator_agreement_kappa():
    def reviewed(i, a, b):
        r = text_record(str(i))
        r["provenance"]["reviews"] = [{"value": a}, {"value": b}]
        return r
    records = [reviewed(0, True, True), reviewed(1, False, False), reviewed(2, True, False), reviewed(3, False, False)]
    result = annotator_agreement(records)
    assert result["agreement"] == 0.75 and result["kappa"] == pytest.approx(0.5)


def test_cli_lint_and_compare(tmp_path):
    records = validate_records([text_record(str(i)) for i in range(12)], tmp_path)
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    main(["lint", "--records", str(path), "--allow-draft", "--output", str(tmp_path / "lint.json")])
    assert json.loads((tmp_path / "lint.json").read_text())["checks"]
    runs = []
    for name, prefer_yes in (("a", True), ("b", False)):
        backend = FakeBackend()
        if not prefer_yes:
            backend.score = lambda images, prompt, labels, choices: (
                [2.0 if ": no" in line else 0.0 for line in prompt.split("\n")[-len(labels):]], {})
        runs.append(run_local(records, tmp_path, tmp_path / name, backend, warmup=0, repeats=1,
                              monitor=lambda: {"busy": []}))
    main(["compare", "--records", str(path), "--allow-draft", "--split", "dev", "--predictions-a", str(runs[0]),
          "--predictions-b", str(runs[1]), "--output", str(tmp_path / "compare.json")])
    compared = json.loads((tmp_path / "compare.json").read_text())
    assert compared["difference"] == 1.0 and compared["clusters"] == 12
    main(["score", "--records", str(path), "--allow-draft", "--split", "dev", "--predictions", str(runs[0]),
          "--output", str(tmp_path / "score.json")])
    scored = json.loads((tmp_path / "score.json").read_text())
    assert scored["cluster_accuracy_ci"]["clusters"] == 12 and scored["contrast_sets"]["sets"] == 0


def test_process_monitor_ignores_desktop_processes(monkeypatch):
    import subprocess
    from imajev_bench import harness
    listing = ("  417  30.0 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer\n"
               " 9001  95.0 /opt/homebrew/bin/python3\n"
               " 9002   3.0 /usr/bin/idle\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=listing))
    snapshot = harness.process_monitor()
    assert [b["pid"] for b in snapshot["busy"]] == [9001]
    assert [b["pid"] for b in snapshot["ignored"]] == [417]
