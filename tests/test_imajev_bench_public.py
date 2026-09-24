"""Public releases withhold test gold: such records run, never score locally, and the label holders can score the run."""
import json

import pytest

from imajev_bench.cli import main, read_jsonl
from imajev_bench.runner import digest, run, verify_run
from imajev_bench.schema import validate_records


def full(record_id="t1", split="test", gold=True):
    return {"id": record_id, "group_id": f"g-{record_id}", "track": "text", "family": "policy", "split": split,
            "images": [], "request": {"request_id": record_id, "state": "Visible count is 3", "fields": [
                {"id": "decision", "type": "boolean", "question": "Is count greater than 2?"}]},
            "gold": gold, "annotation_status": "draft", "provenance": {"construction": {"truth": gold}}}


def public(record):
    row = json.loads(json.dumps(record))
    if row["split"] == "test":
        row.update(gold=None, gold_withheld=True, annotation_status="reviewed", provenance={})
    return row


def test_withheld_records_validate_and_keep_the_flag(tmp_path):
    rows = validate_records([public(full())], tmp_path)
    assert rows[0]["gold_withheld"] is True and rows[0]["gold"] is None
    bad = public(full()); bad["gold"] = True
    with pytest.raises(ValueError, match="gold null"):
        validate_records([bad], tmp_path)


def test_full_label_records_hash_as_before(tmp_path):
    rows = validate_records([full(split="dev")], tmp_path)
    assert "gold_withheld" not in rows[0]
    assert digest(rows) == digest(validate_records(rows, tmp_path))


def test_public_run_is_scored_by_label_holders_only(tmp_path):
    pub = validate_records([public(full())], tmp_path)
    path = run(pub, tmp_path, tmp_path / "run", adapter="first")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["gold_withheld"] is True and manifest["inputs_sha256"]
    # the maintainers' full records verify the run through the input hash
    assert verify_run(validate_records([full()], tmp_path), path)["completion"]["status"] == "complete"
    changed = full(); changed["request"]["state"] = "Visible count is 1"
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run(validate_records([changed], tmp_path), path)
    # local scoring on withheld records is refused
    records = tmp_path / "public.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in pub))
    with pytest.raises(SystemExit):
        main(["score", "--records", str(records), "--allow-draft", "--split", "test", "--predictions", str(path),
              "--output", str(tmp_path / "score.json")])
    assert not (tmp_path / "score.json").exists()
    assert read_jsonl(path)[0]["value"] is True
