import json

import pytest
from PIL import Image

from imajev_bench.harness import run_local
from imajev_bench.release import blind_baseline
from imajev_bench.runner import file_digest
from test_imajev_bench_v2 import FakeBackend, text_record


def setup_run(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "blue").save(image)
    record = text_record("item")
    record.update(track="visual", images=[{"path": image.name, "sha256": file_digest(image)}])
    path = run_local([record], tmp_path, tmp_path / "run", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    return record, path


@pytest.mark.parametrize("artifact", ["completion.json", "raw.jsonl"])
def test_blind_baseline_rejects_unfinished_or_missing_artifacts(tmp_path, artifact):
    record, path = setup_run(tmp_path)
    (path.parent / artifact).unlink()
    with pytest.raises(ValueError, match="completed|integrity"):
        blind_baseline([record], path)


def test_blind_baseline_rejects_modified_predictions_and_other_dataset(tmp_path):
    record, path = setup_run(tmp_path)
    different = {**record, "gold": False}
    with pytest.raises(ValueError, match="dataset hash"):
        blind_baseline([different], path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["value"] = False
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="integrity"):
        blind_baseline([record], path)


def test_blind_baseline_accepts_verified_no_image_run(tmp_path):
    record, path = setup_run(tmp_path)
    report = blind_baseline([record], path)
    assert report["items"] == 1 and report["blind_accuracy"] == 1.0


def release_records(tmp_path):
    rows = [text_record("dev-text"), text_record("test-text", split="test")]
    for i, (track, split, gold) in enumerate((("visual", "test", True), ("visual", "test", False),
                                             ("joint", "calibration", None))):
        image = tmp_path / f"image-{i}.png"
        Image.new("RGB", (2, 2), (i * 50, 0, 0)).save(image)
        record = text_record(f"image-{i}", gold=gold, split=split)
        record.update(track=track, images=[{"path": image.name, "sha256": file_digest(image)}])
        rows.append(record)
    return rows


@pytest.mark.parametrize("selection", ["all", "images", "test", "test-images"])
def test_blind_baseline_verifies_complete_protocol_selections(tmp_path, selection):
    records = release_records(tmp_path)
    selected = records
    if selection.startswith("test"):
        selected = [r for r in selected if r["split"] == "test"]
    if selection.endswith("images"):
        selected = [r for r in selected if r["track"] in ("visual", "joint")]
    path = run_local(selected, tmp_path, tmp_path / "run", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    report = blind_baseline(records, path)
    assert report["items"] == 2
    assert report["blind_accuracy"] == .5


def test_blind_baseline_rejects_missing_eligible_record_within_selected_split(tmp_path):
    records = release_records(tmp_path)
    selected = [r for r in records if r["track"] == "visual"][:1]
    path = run_local(selected, tmp_path, tmp_path / "run", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    with pytest.raises(ValueError):
        blind_baseline(records, path)


def test_blind_baseline_rejects_changed_dataset_for_same_size_selection(tmp_path):
    records = release_records(tmp_path)
    selected = [r for r in records if r["track"] in ("visual", "joint")]
    path = run_local(selected, tmp_path, tmp_path / "run", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    records[2]["gold"] = False
    with pytest.raises(ValueError):
        blind_baseline(records, path)


def test_blind_baseline_rejects_text_only_split(tmp_path):
    records = release_records(tmp_path)
    selected = [r for r in records if r["split"] == "dev"]
    path = run_local(selected, tmp_path, tmp_path / "run", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    with pytest.raises(ValueError, match="answerable visual or joint"):
        blind_baseline(records, path)


def test_blind_baseline_rejects_image_selection_with_only_unknown_gold(tmp_path):
    record, _ = setup_run(tmp_path)
    record["gold"] = None
    path = run_local([record], tmp_path, tmp_path / "unknown", FakeBackend(), condition="no_image",
                     warmup=0, repeats=1, monitor=lambda: {"busy": []})
    with pytest.raises(ValueError, match="answerable visual or joint"):
        blind_baseline([record], path)
