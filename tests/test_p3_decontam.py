"""Tests for the phase-3 decontamination filter and leakage checker (scripts/p3/decontam.py). Synthetic fixtures only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/p3"))
import decontam as dc  # noqa: E402
import decontam_index as di  # noqa: E402

PUB_STATE = ("The quarterly revenue of the northern warehouse rose from 4.2 million to 5.9 million dollars while freight "
             "costs doubled after the harbour strike closed the eastern terminal for eleven weeks in the spring")
PRIV_STATE = ("Zorblax heliotrope manifests quivered beneath the cobalt aqueduct whenever seventeen lanterns flickered over "
              "the marzipan observatory near the vermilion causeway at dusk")
PRIV_FIELD = "which lantern count unlocks the vermilion marzipan gate"
PUB_FIELD = "Choose the terminal that reopened first after the strike ended"


def bench_rows():
    return [
        di.BenchRow(di.B_DB, "db-row-1", state=[PUB_STATE], qopts=["Which cost rose the most?"],
                    upstream=di.upstream_keys("dgslibisey/MuSiQue", "2hop__111_222", "validation")),
        di.BenchRow(di.B_DB, "db-row-2", state=["short"], qopts=[PUB_FIELD],
                    upstream=di.upstream_keys("dreamerdeo/finqa", "AAP/2011/page_28.pdf-1", "validation")),
        di.BenchRow(di.B_DB, "db-row-3", state=["x"], qopts=["y"], upstream=di.upstream_keys("tasksource/folio", "0", "validation")),
        di.BenchRow(di.B_PRIV, "secret-id-777", state=[PRIV_STATE], qopts=[PRIV_FIELD], images=["ab" * 32]),
    ]


def cand(i, state, question="Pick the best option for this case please", options=("a", "b"), upstream=None, dataset="gen",
         split="train", parent=None, images=()):
    return {"id": i, "source": "B", "dataset": dataset, "family": "f", "difficulty": 3, "state": state,
            "images": list(images),
            "field": {"type": "choice", "question": question, "options": [{"key": k, "text": k} for k in options]},
            "gold": options[0], "unknown_reason": None, "gold_kind": "dataset", "parent_id": parent,
            "provenance": {"licence": "MIT", "upstream_split": split, "upstream_id": upstream}}


@pytest.fixture()
def ix():
    return di.build(bench_rows())


def reasons(ix, row):
    st, q = dc.cand_units(row)
    m = dc.match(ix, di.Tok(), st, q, dc.cand_upstream(row), [], 0.5)
    return {r for r, _, _ in m["drop"]}, m


def test_clean_row_kept(ix):
    r, m = reasons(ix, cand("c1", "A brand new generated scenario about invoices with nothing in common at all today"))
    assert r == set() and m["flag"] is None


def test_upstream_id_drop(ix):
    assert "upstream" in reasons(ix, cand("c2", "some text", upstream="2hop__111_222", dataset="musique"))[0]
    # FinQA: another question on the same filing page is the same upstream item
    assert "upstream" in reasons(ix, cand("c3", "t", upstream="AAP/2011/page_28.pdf-3", dataset="finqa", split="train"))[0]
    # numeric ids only join together with their split
    assert "upstream" in reasons(ix, cand("c4", "t", upstream="0", dataset="FOLIO", split="validation"))[0]
    assert "upstream" not in reasons(ix, cand("c5", "t", upstream="0", dataset="folio", split="train"))[0]
    assert "upstream" not in reasons(ix, cand("c6", "t", upstream="2hop__999_222", dataset="musique"))[0]


def test_near_dup_drop(ix):
    words = PUB_STATE.split()
    words[5] = "southern"  # one edit: most 13-grams survive
    r, _ = reasons(ix, cand("c7", " ".join(words)))
    assert "near_dup" in r
    # a candidate that merely contains a single 13-gram of a long benchmark text is flagged, not dropped
    tail = " ".join(PUB_STATE.split()[:13])
    long_new = tail + " " + " ".join(f"word{i}" for i in range(60))
    r, m = reasons(ix, cand("c8", long_new))
    assert "near_dup" not in r and m["flag"] is not None


def test_exact_field_drop(ix):
    r, _ = reasons(ix, cand("c9", {"note": "a fresh state that is unrelated"}, question=PUB_FIELD.upper() + "!"))
    assert "field" in r
    r, _ = reasons(ix, cand("c10", "fresh state entirely new", question="Choose the terminal that reopened"))  # 5 tokens
    assert "field" not in r


def test_exact_state_drop(ix):
    r, _ = reasons(ix, cand("c11", {"a": PUB_STATE.lower()}))
    assert "state" in r


def _setup_index(tmp_path, monkeypatch, rows):
    ix = di.build(rows, {"built_utc": "test"})
    idx_dir = tmp_path / "index"
    ix.save(idx_dir)
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    (src_dir / "sources.json").write_text(json.dumps({"sources": {}}))
    monkeypatch.setattr(di, "INDEX_DIR", idx_dir)
    monkeypatch.setattr(di, "SOURCES_DIR", src_dir)
    monkeypatch.setattr(di, "get_index", lambda rebuild=False, verbose=True: di.Index.load(idx_dir))


def test_filter_cli_and_private_text_never_written(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch, bench_rows())
    inp = tmp_path / "X.jsonl"
    rows = [cand("keep-1", "A brand new generated scenario about invoices with nothing in common at all"),
            cand("drop-priv-state", PRIV_STATE),
            cand("drop-priv-field", "unrelated", question=PRIV_FIELD),
            cand("drop-priv-image", "unrelated again", images=[{"path": "x.jpg", "sha256": "ab" * 32}]),
            cand("drop-pub", "t", upstream="2hop__111_222", dataset="musique")]
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    inp.write_text("\n".join(lines) + "\n")
    out, rep = tmp_path / "clean", tmp_path / "rep.md"
    assert dc.main(["--in", str(inp), "--out-dir", str(out), "--report", str(rep), "--workers", "2", "--batch", "2"]) == 0
    kept = (out / "X.jsonl").read_text().splitlines()
    assert kept == [lines[0]]  # unchanged bytes
    drops = [json.loads(x) for x in (out / "drops.jsonl").read_text().splitlines()]
    assert {d["id"] for d in drops} == {"drop-priv-state", "drop-priv-field", "drop-priv-image", "drop-pub"}
    by = {d["id"]: d for d in drops}
    assert by["drop-priv-state"]["benchmark"] == di.B_PRIV
    assert by["drop-priv-image"]["reason"] == "image"
    assert by["drop-pub"]["benchmark"] == di.B_DB and by["drop-pub"]["matches"][0]["benchmark_row"] == "db-row-1"
    written = rep.read_text() + (out / "drops.jsonl").read_text() + (out / "flagged.jsonl").read_text() + \
        (di.INDEX_DIR / "meta.json").read_text()
    for secret in ("secret-id-777", "zorblax", "Zorblax", "marzipan", "vermilion"):
        assert secret not in written
    assert "imajevbench-private-1" in rep.read_text()


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_leakage_pass_and_fail(tmp_path):
    train = [cand("t1", "Invoice 4411 from Kestrel Supplies lists twelve pallets of copper wire delivered on the third of May to depot nine"),
             cand("t1-var", "Invoice 4411 variant", parent="t1")]
    held = [cand("h1", "Seven hikers left the northern trailhead at dawn and two returned before noon because the river crossing had flooded badly")]
    tr, he = _write(tmp_path / "train.jsonl", train), _write(tmp_path / "held.jsonl", held)
    assert dc.main(["--leakage", "--train", tr, "--heldout", he]) == 0
    # a variant of a held-out parent in train -> fail
    tr2 = _write(tmp_path / "train2.jsonl", train + [cand("h1-var", "totally different words here", parent="h1")])
    assert dc.main(["--leakage", "--train", tr2, "--heldout", he]) == 1
    # a near-duplicate of a held-out item in train -> fail
    near = held[0]["state"].replace("Seven", "Eight")
    tr3 = _write(tmp_path / "train3.jsonl", train + [cand("t9", near)])
    assert dc.main(["--leakage", "--train", tr3, "--heldout", he]) == 1
    # held-out variant whose parent is in train -> fail (the family is split across)
    he2 = _write(tmp_path / "held2.jsonl", held + [cand("h2", "fresh unrelated words about ferries", parent="t1")])
    assert dc.main(["--leakage", "--train", tr, "--heldout", he2]) == 1


def test_template_field_flagged_not_dropped():
    label = "balance not updated after cheque or cash deposit"
    rows = [di.BenchRow(di.B_DB, f"r{i}", state=[f"message number {i}"], qopts=["Route it", label]) for i in range(60)]
    ix = di.build(rows)
    row = cand("c1", "a new customer message about something else", options=(label, "other option"))
    st, q = dc.cand_units(row)
    m = dc.match(ix, di.Tok(), st, q, [], [], 0.5, True, 50)
    assert not m["drop"] and m["template"]
    m = dc.match(ix, di.Tok(), st, q, [], [], 0.5, True, 0)
    assert [d[0] for d in m["drop"]] == ["field"]
