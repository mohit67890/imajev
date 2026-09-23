#!/usr/bin/env python3
"""Build a streaming inventory of locally observed evaluation/training examples.

The inventory is evidence of local overlap only.  It cannot establish that an
example was absent from model pretraining or from files outside the scan roots.
Image files are never opened: only references and stored digests in JSONL are
indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1
DEFAULT_PATTERNS = ("data/manifests/*.jsonl", "data/decision-v1*/**/records.jsonl")
SOURCE_KEYS = frozenset(("source", "source_name", "dataset"))
SOURCE_ID_KEYS = frozenset(("source_id", "source_group", "source_group_id"))
IMAGE_REF_KEYS = frozenset(("image", "path", "image_path", "image_ref", "uri", "url"))
SHA_KEYS = frozenset(("sha256", "image_sha256"))
MAX_VALUE_LENGTH = 16_384


def _scalar(value: Any) -> str | None:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        text = str(value).strip()
        if text and len(text) <= MAX_VALUE_LENGTH and not text.startswith("data:"):
            return text
    return None


def _walk_images(value: Any, path: str = "images") -> Iterator[tuple[str, str, str]]:
    """Yield (kind, value, json_path) beneath an images field."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_images(child, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            text = _scalar(child)
            if key in SHA_KEYS and text:
                yield "image_sha256", text.lower(), child_path
            elif key in IMAGE_REF_KEYS and text:
                yield "image_ref", text, child_path
            elif isinstance(child, (dict, list)):
                yield from _walk_images(child, child_path)
    else:
        text = _scalar(value)
        if text:
            yield "image_ref", text, path


def extract_exposures(record: Any) -> Iterator[tuple[str, str, str]]:
    """Extract stable identifiers and stored image metadata from one record."""
    if not isinstance(record, dict):
        return
    sources: list[str] = []
    source_ids: list[str] = []
    for key, value in record.items():
        text = _scalar(value)
        if key in SOURCE_KEYS and text:
            sources.append(text)
            yield "source", text, key
        elif key in SOURCE_ID_KEYS and text:
            source_ids.append(text)
            yield "source_id", text, key
        elif key == "id" and text:
            yield "record_id", text, key
        elif key in ("images", "image"):
            if key == "image" and text:
                yield "image_ref", text, key
            else:
                yield from _walk_images(value, key)

    # Namespace the group because identifiers such as numeric COCO ids collide
    # across datasets. The unit separator makes the representation unambiguous.
    for source in sources:
        for source_id in source_ids:
            yield "canonical_source_id", f"{source}\x1f{source_id}", "source+source_group"


def discover_inputs(root: Path, patterns: Iterable[str] = DEFAULT_PATTERNS) -> list[Path]:
    paths = {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    return sorted(paths, key=lambda path: path.as_posix())


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE kinds (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        CREATE TABLE entries (
            kind_id INTEGER NOT NULL,
            value_sha256 BLOB NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY (kind_id, value_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            lines INTEGER NOT NULL,
            records INTEGER NOT NULL,
            invalid_json_lines INTEGER NOT NULL,
            extracted_occurrences INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )


def build_inventory(inputs: Iterable[Path], output: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Stream JSONL inputs into an atomically replaced SQLite inventory."""
    root = (root or Path.cwd()).resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    totals: Counter[str] = Counter()
    files_summary: list[dict[str, Any]] = []
    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)
        kind_ids: dict[str, int] = {}
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("entry_encoding", "SHA-256(kind UTF-8 value); exact values omitted for compactness"))
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("coverage_limitations", "Local JSONL metadata only; does not establish pretraining absence; referenced images are not decoded or hashed."),
        )
        for input_path in sorted((Path(p).resolve() for p in inputs), key=lambda p: p.as_posix()):
            digest = hashlib.sha256()
            stats: Counter[str] = Counter()
            try:
                relative = input_path.relative_to(root).as_posix()
            except ValueError:
                relative = input_path.as_posix()
            with input_path.open("rb") as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    stats["bytes"] += len(raw_line)
                    stats["lines"] += 1
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        stats["invalid_json_lines"] += 1
                        continue
                    stats["records"] += 1
                    for kind, value, _json_path in extract_exposures(record):
                        kind_id = kind_ids.get(kind)
                        if kind_id is None:
                            connection.execute("INSERT OR IGNORE INTO kinds(name) VALUES (?)", (kind,))
                            kind_id = connection.execute("SELECT id FROM kinds WHERE name=?", (kind,)).fetchone()[0]
                            kind_ids[kind] = kind_id
                        connection.execute(
                            """INSERT INTO entries(kind_id,value_sha256,occurrences) VALUES(?,?,1)
                               ON CONFLICT(kind_id,value_sha256) DO UPDATE SET occurrences=occurrences+1""",
                            (kind_id, hashlib.sha256(value.encode("utf-8")).digest()),
                        )
                        stats["extracted_occurrences"] += 1
                        totals[f"{kind}_occurrences"] += 1
            receipt = {
                "path": relative,
                "sha256": digest.hexdigest(),
                **{name: stats[name] for name in ("bytes", "lines", "records", "invalid_json_lines", "extracted_occurrences")},
            }
            files_summary.append(receipt)
            connection.execute(
                "INSERT INTO files VALUES (?,?,?,?,?,?,?)",
                tuple(receipt[name] for name in ("path", "sha256", "bytes", "lines", "records", "invalid_json_lines", "extracted_occurrences")),
            )
            connection.commit()
        for kind, count in connection.execute(
            "SELECT kinds.name, COUNT(*) FROM entries JOIN kinds ON kinds.id=entries.kind_id GROUP BY kinds.name"
        ):
            totals[f"{kind}_distinct"] = count
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("scan_root", root.as_posix()))
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, output)
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "database": output.as_posix(),
        "files_scanned": len(files_summary),
        "totals": dict(sorted(totals.items())),
        "files": files_summary,
        "coverage_limitations": [
            "The inventory covers only the selected local JSONL files.",
            "It detects local identifier or stored-digest overlap; it cannot prove pretraining cleanliness.",
            "Image references and stored SHA-256 values are indexed without reading image bytes.",
        ],
    }


def contains(database: Path, kind: str, value: str) -> bool:
    """Return whether an exact kind/value pair is in an inventory."""
    with sqlite3.connect(database) as connection:
        return connection.execute(
            """SELECT 1 FROM entries JOIN kinds ON kinds.id=entries.kind_id
               WHERE kinds.name=? AND entries.value_sha256=?""",
            (kind, hashlib.sha256(value.encode("utf-8")).digest()),
        ).fetchone() is not None


def occurrence_count(database: Path, kind: str, value: str) -> int:
    """Return the number of extracted occurrences of an exact kind/value pair."""
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """SELECT occurrences FROM entries JOIN kinds ON kinds.id=entries.kind_id
               WHERE kinds.name=? AND entries.value_sha256=?""",
            (kind, hashlib.sha256(value.encode("utf-8")).digest()),
        ).fetchone()
    return 0 if row is None else int(row[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/imajev-bench-fresh-v1/exposure.sqlite"))
    parser.add_argument("--summary", type=Path, default=Path("reports/imajev-bench-fresh-v1/exposure.json"))
    parser.add_argument("--pattern", action="append", dest="patterns", help="Root-relative glob; repeatable")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = discover_inputs(root, args.patterns or DEFAULT_PATTERNS)
    output = args.output if args.output.is_absolute() else root / args.output
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    summary = build_inventory(inputs, output, root=root)
    summary["database"] = os.path.relpath(output, summary_path.parent)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"database": output.as_posix(), "summary": summary_path.as_posix(), "files_scanned": len(inputs), "totals": summary["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
