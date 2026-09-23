import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.imajev_bench.build_exposure_inventory import (
    build_inventory,
    contains,
    discover_inputs,
    extract_exposures,
    occurrence_count,
)


def _write_jsonl(path: Path, rows: list[object], suffix: bytes = b"") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows) + suffix
    path.write_bytes(content)
    return content


def test_extracts_canonical_ids_and_nested_image_metadata_without_payloads():
    record = {
        "id": "vqav2:42",
        "source": "vqav2",
        "source_group": "42",
        "images": [{"image": "images/a.jpg", "sha256": "AB" * 32, "nested": {"path": "thumb/a.jpg"}}],
        "request": {"request_id": "not-a-record-id"},
        "preview": "data:image/png;base64,do-not-index",
    }
    found = {(kind, value) for kind, value, _ in extract_exposures(record)}
    assert ("record_id", "vqav2:42") in found
    assert ("source_id", "42") in found
    assert ("canonical_source_id", "vqav2\x1f42") in found
    assert ("image_ref", "images/a.jpg") in found
    assert ("image_ref", "thumb/a.jpg") in found
    assert ("image_sha256", "ab" * 32) in found
    assert all("not-a-record-id" not in value and not value.startswith("data:") for _, value in found)


def test_build_inventory_streams_deduplicates_and_records_file_receipts(tmp_path: Path):
    manifest = tmp_path / "data/manifests/train.jsonl"
    raw = _write_jsonl(
        manifest,
        [
            {"id": "a", "source": "demo", "source_group": "g1", "images": [{"image": "a.jpg", "sha256": "1" * 64}]},
            {"id": "b", "source": "demo", "source_group": "g1", "images": [{"path": "a.jpg", "sha256": "1" * 64}]},
        ],
        b"{invalid json\n",
    )
    database = tmp_path / "out/exposure.sqlite"
    summary = build_inventory([manifest], database, root=tmp_path)

    assert summary["files_scanned"] == 1
    assert summary["files"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["files"][0]["records"] == 2
    assert summary["files"][0]["invalid_json_lines"] == 1
    assert contains(database, "canonical_source_id", "demo\x1fg1")
    assert contains(database, "image_sha256", "1" * 64)
    assert occurrence_count(database, "image_ref", "a.jpg") == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_discovery_is_narrow_and_sorted(tmp_path: Path):
    expected = [
        tmp_path / "data/decision-v1/demo/records.jsonl",
        tmp_path / "data/manifests/z.jsonl",
    ]
    for path in reversed(expected):
        _write_jsonl(path, [])
    _write_jsonl(tmp_path / "data/other/records.jsonl", [])
    assert discover_inputs(tmp_path) == expected
