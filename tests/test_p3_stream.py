"""Phase-3 streamed mining + Azure teacher tooling (scripts/p3/{mine,stream_coordinator,azure_teacher,variants,mock_azure}.py).

Everything runs against the local mock (scripts/p3/mock_azure.py) with a FAKE key; no real Azure endpoint is ever called."""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts/p3", ROOT / "scripts/p2", ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import azure_teacher as T  # noqa: E402
import mine as M  # noqa: E402
import stream_coordinator as C  # noqa: E402
import variants as V  # noqa: E402
from candidate import validate  # noqa: E402
from mock_azure import MockAzure  # noqa: E402

KEY = "sk-FAKE-imajev-test-key-7f3a9c"


# ------------------------------------------------------------------------------------------------ fixtures

def item(i, source="B", gold_kind="dataset", ftype="choice", gold="b", family="fam_a", images=None, question=None):
    q = question or f"Which option is right for case {i}?"
    field = {"type": ftype, "question": q}
    if ftype == "choice":
        field["options"] = [{"key": k, "text": f"option {k}"} for k in ("a", "b", "c")]
    elif ftype == "score":
        field["levels"] = [{"value": v, "description": f"level {v}"} for v in range(3)]
    row = {"id": f"it-{i}", "source": source, "dataset": "test", "family": family, "difficulty": 4,
           "state": {"case": i, "note": f"document for case {i}"}, "images": images or [], "field": field, "gold": gold,
           "unknown_reason": "insufficient_evidence" if gold is None else None, "gold_kind": gold_kind, "parent_id": None,
           "provenance": {"licence": "generated"}}
    assert validate(row) == [], validate(row)
    return row


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


@pytest.fixture
def mock():
    m = MockAzure(KEY, policy="gold").start()
    yield m
    m.stop()


@pytest.fixture
def keyfile(tmp_path):
    p = tmp_path / "fake-key"
    p.write_text(KEY + "\n")
    return p


def teacher_args(tmp_path, keyfile, mock, *extra, queue=None, max_usd="1000"):
    q = queue or tmp_path / "stream/queue.jsonl"
    return ["run", "--deployment", "teacher-35b:1", "--queue", str(q), "--out", str(tmp_path / "teacher"),
            "--base-url", mock.base_url, "--key-file", str(keyfile), "--data-root", str(tmp_path / "data"),
            "--max-usd", max_usd, "--per-instance-concurrency", "4", "--poll-secs", "0.05", "--status-secs", "0.2",
            "--max-attempts", "4", *extra]


def queue_rows(rows, origin="mine"):
    return [{"id": r["id"], "origin": origin, "item": r} for r in rows]


def gold_label(row):
    g = row["gold"]
    return "unknown" if g is None else ("true" if g else "false") if isinstance(g, bool) else str(g)


# ------------------------------------------------------------------------------------------------ keep rules (unit)

def solve(probs, ok=True):
    return {"ok": ok, "probs": probs}


def P(label, labels=("a", "b", "c", "unknown")):
    return {l: (0.91 if l == label else 0.03) for l in labels}


@pytest.mark.parametrize("source,kind,labels,expect_keep,expect_reason,target", [
    ("A", "constructed", ["b"], True, "teacher_matches_gold", "teacher_dist"),
    ("A", "constructed", ["a"], True, "hard_gold_teacher_disagrees", "hard_gold"),
    ("D", "constructed", ["b"], True, "teacher_matches_gold", "teacher_dist"),
    ("D", "constructed", ["a"], False, "teacher_disagrees_with_constructed_gold", None),
    ("B", "dataset", ["b", "b"], True, "two_solves_agree", "teacher_dist"),
    ("B", "dataset", ["a", "a"], False, "solves_disagree_with_gold", None),
    ("B", "dataset", ["b", "a"], False, "solves_disagree", None),
    ("B", "dataset", ["a"], False, "first_solve_failed_rule", None),
    ("C", "dataset", ["b", "b"], True, "two_solves_agree", "teacher_dist"),
    ("C", "dataset", ["c", "b"], False, "solves_disagree", None),
    ("C", "dataset", ["a", "a"], False, "gold_conflict", None),
])
def test_keep_rules_per_source(source, kind, labels, expect_keep, expect_reason, target):
    q = {"id": "x", "item": item(1, source=source, gold_kind=kind)}
    v = T.keep_decision(q, [solve(P(l)) for l in labels])
    assert (v["keep"], v["reason"], v["target"]) == (expect_keep, expect_reason, target)
    if target == "hard_gold":
        assert v["review_first"] and v["target_probs"]["b"] == 1.0
    if expect_keep and len(labels) == 2:
        assert v["target_probs"] == T.mean_probs([solve(P(l)) for l in labels])


def test_keep_rules_images_and_unknown_variants():
    img = item(2, source="I", gold_kind="constructed", images=["x.png"])
    v = T.keep_decision({"id": "i", "item": img}, [solve(P("b"))])
    assert v["keep"] and v["target"] == "hard_gold+dist" and v["hard_label"] == "b"
    assert not T.keep_decision({"id": "i", "item": img}, [solve(P("a"))])["keep"]       # no hard-gold fallback for images
    mined = item(3, source="C", gold_kind="dataset", images=["x.png"])
    assert T.rule_for({"item": mined}) == "image_mined" and T.needs_two({"item": mined})
    v = T.keep_decision({"id": "m", "item": mined}, [solve(P("b")), solve(P("b"))])
    assert v["keep"] and "needs_9b_distribution" not in v and v["hard_label"] == "b"    # no 9B pass (owner decision)
    # programmatic unknown variant (constructed): one solve, kept only with most mass on unknown
    unk = item(4, source="A", gold_kind="constructed", gold=None)
    q = {"id": "u", "item": unk, "variant": {"kind": "unknown"}}
    assert T.keep_decision(q, [solve(P("unknown"))])["keep"]
    assert not T.keep_decision(q, [solve({"a": 0.5, "b": 0.0, "c": 0.0, "unknown": 0.5})])["keep"]
    # writer unknown variant: two solves, both on unknown
    wq = {"id": "w", "item": item(5, source="B", gold_kind="none", gold=None), "variant": {"kind": "unknown"}}
    assert T.keep_decision(wq, [solve(P("unknown")), solve(P("unknown"))])["keep"]
    assert T.keep_decision(wq, [solve(P("unknown"))])["reason"] == "missing_second_solve"
    assert not T.keep_decision(wq, [solve(P("unknown")), solve(P("a"))])["keep"]
    # a failed solve never keeps
    assert T.keep_decision({"id": "f", "item": item(6, source="A", gold_kind="constructed")}, [solve(None, ok=False)])["reason"] == "solve_failed"


def test_finalize_caps_hard_gold_at_quarter_of_A(tmp_path):
    out = tmp_path / "teacher"
    rows = [{"id": f"m{i}", "keep": True, "rule": "A", "target": "teacher_dist", "item": {"source": "A"}} for i in range(6)]
    rows += [{"id": f"h{i}", "keep": True, "rule": "A", "target": "hard_gold", "review_first": True, "item": {"source": "A"}}
             for i in range(10)]
    rows += [{"id": "d0", "keep": False, "rule": "B", "reason": "solves_disagree", "item": {"source": "B"}}]
    write_jsonl(out / "results.jsonl", rows)
    s = T.finalize(out)
    assert s["hard_gold_A_kept"] == 2 and s["hard_gold_A_capped"] == 8     # 2 / (6 + 2) = 25%
    assert len(read_jsonl(out / "kept.jsonl")) == 8


# ------------------------------------------------------------------------------------------------ teacher runner vs mock

def test_second_solve_only_where_gold_not_guaranteed(tmp_path, keyfile, mock):
    rows = [item(1, source="A", gold_kind="constructed"), item(2, source="B"), item(3, source="C")]
    for r in rows:
        mock.answers[r["field"]["question"]] = gold_label(r)
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
    assert T.main(teacher_args(tmp_path, keyfile, mock)) == 0
    by_q = {}
    for body in mock.requests:
        by_q.setdefault(MockAzure.user_text(body).split("QUESTION: ")[1].split("\n")[0], []).append(body)
    assert len(by_q[rows[0]["field"]["question"]]) == 1                    # constructed: one solve
    for r in rows[1:]:
        first, second = by_q[r["field"]["question"]]
        assert first["temperature"] == 0.0 and second["temperature"] == 0.6
        assert first["seed"] != second["seed"]
        assert first["messages"][0]["content"] == T.DISTRIBUTION_SYSTEM
        assert second["messages"][0]["content"] == T.SECOND_SYSTEM != T.DISTRIBUTION_SYSTEM
        assert first["messages"][1] == second["messages"][1]                # same question, independent solve
    for body in mock.requests:
        assert "chat_template_kwargs" not in body and body["model"] == "teacher-35b"
        assert body["response_format"]["json_schema"]["name"] == "decision_distribution"
    res = {r["id"]: r for r in read_jsonl(tmp_path / "teacher/results.jsonl")}
    assert all(res[r["id"]]["keep"] for r in rows)
    assert res["it-2"]["reason"] == res["it-3"]["reason"] == "two_solves_agree"


def test_second_solve_disagreement_drops(tmp_path, keyfile, mock):
    r = item(7, source="C")
    mock.custom = lambda body, labels: P("b" if body["temperature"] == 0.0 else "c", labels)
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([r]))
    T.main(teacher_args(tmp_path, keyfile, mock))
    (res,) = read_jsonl(tmp_path / "teacher/results.jsonl")
    assert not res["keep"] and res["reason"] == "solves_disagree" and res["labels"] == ["b", "c"]


def test_first_solve_failure_skips_second_solve(tmp_path, keyfile, mock):
    r = item(8, source="B")                    # gold b; the mock answers a
    mock.policy = "first"
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([r]))
    T.main(teacher_args(tmp_path, keyfile, mock))
    assert len(mock.requests) == 1
    assert read_jsonl(tmp_path / "teacher/results.jsonl")[0]["reason"] == "first_solve_failed_rule"


def test_image_payload_encoding(tmp_path, keyfile, mock):
    from PIL import Image
    img_path = tmp_path / "data/imgs/photo.png"
    img_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (200, 10, 10)).save(img_path)
    r = item(9, source="I", gold_kind="constructed", images=["imgs/photo.png", "imgs/photo.png"])
    mock.answers[r["field"]["question"]] = "b"
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([r]))
    T.main(teacher_args(tmp_path, keyfile, mock))
    (body,) = mock.requests
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    url = parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == img_path.read_bytes()
    assert "QUESTION: " + r["field"]["question"] in parts[2]["text"]
    (res,) = read_jsonl(tmp_path / "teacher/results.jsonl")
    assert res["keep"] and res["target"] == "hard_gold+dist"


def test_retries_on_429_and_500(tmp_path, keyfile, mock):
    mock.fail_429, mock.fail_500 = 2, 1
    r = item(10, source="A", gold_kind="constructed")
    mock.answers[r["field"]["question"]] = "b"
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([r]))
    T.main(teacher_args(tmp_path, keyfile, mock))
    assert len(mock.requests) == 4
    (res,) = read_jsonl(tmp_path / "teacher/results.jsonl")
    assert res["keep"]
    assert json.loads((tmp_path / "teacher/status.json").read_text())["http"] == {"http_429": 2, "http_500": 1}


def test_cost_cap_from_instance_hours_stops_before_dispatch(tmp_path, keyfile, mock):
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([item(i) for i in range(5)]))
    code = T.main(teacher_args(tmp_path, keyfile, mock, "--billing-start", str(time.time() - 3600), max_usd="5"))
    assert code == 2 and mock.requests == []                                 # 1 instance x 1 h x $7.91 > $5
    st = json.loads((tmp_path / "teacher/status.json").read_text())
    assert st["state"] == "cap_reached" and st["spend_usd"] >= 7.9


def test_cost_cap_from_usage_tokens(tmp_path, keyfile, mock):
    mock.completion_tokens = 2_100_000         # ~$2.20 of instance time per solve at 2,100 tok/s and $7.91/h
    rows = [item(i, source="A", gold_kind="constructed") for i in range(6)]
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
    T.main([*teacher_args(tmp_path, keyfile, mock, max_usd="5"), "--per-instance-concurrency", "1"])
    assert len(mock.requests) == 3                                           # 2.20, 4.39 < 5, then 6.59 >= 5: stop
    solves = read_jsonl(tmp_path / "teacher/solves.jsonl")
    assert all(abs(s["est_usd"] - 2_100_000 * 7.91 / (2100 * 3600) - T.PREFILL_WEIGHT * s["usage"]["prompt_tokens"] * 7.91 / (2100 * 3600)) < 1e-6
               for s in solves)
    assert json.loads((tmp_path / "teacher/status.json").read_text())["state"] == "cap_reached"


def test_ledger_integrates_instance_changes(tmp_path):
    now = [1000.0]
    led = T.Ledger(tmp_path / "instances.jsonl", clock=lambda: now[0])
    led.record(1, t=0.0)                   # pilot: 1 instance for an hour
    led.record(8, t=3600.0)                # scaled to 8
    led.record(0, t=7200.0)                # deleted
    led.record(1, pool="writer", t=0.0)
    led.record(0, pool="writer", t=1800.0)
    now[0] = 99999.0
    assert led.spend() == pytest.approx(7.91 * (1 + 8 + 0.5))


def test_resume_after_crash(tmp_path, keyfile, mock):
    rows = [item(i, source="B") for i in range(6)]
    for r in rows:
        mock.answers[r["field"]["question"]] = "b"
    q = tmp_path / "stream/queue.jsonl"
    write_jsonl(q, queue_rows(rows))
    T.main(teacher_args(tmp_path, keyfile, mock, "--max-items", "2"))
    out = tmp_path / "teacher"
    assert len(read_jsonl(out / "results.jsonl")) == 2
    # crash artefacts: a solve-1 that never got its result, and torn last lines in both files
    s1 = dict(read_jsonl(out / "solves.jsonl")[0], id="it-4")
    with open(out / "solves.jsonl", "a") as fh:
        fh.write(json.dumps(s1) + "\n" + '{"id": "it-5", "solve": 1, "ok": tr')
    with open(out / "results.jsonl", "a") as fh:
        fh.write('{"id": "it-3", "keep": tr')
    n_before = len(mock.requests)
    T.main(teacher_args(tmp_path, keyfile, mock))
    res = read_jsonl(out / "results.jsonl")
    assert sorted(r["id"] for r in res) == [r["id"] for r in rows]           # every item exactly once
    new = mock.requests[n_before:]
    asked = [MockAzure.user_text(b).split("QUESTION: ")[1].split("\n")[0] for b in new]
    assert asked.count(rows[4]["field"]["question"]) == 1                    # it-4: only its missing second solve
    assert len(new) == 2 * 3 + 1                                             # it-2, it-3, it-5 twice; it-4 once
    solves = read_jsonl(out / "solves.jsonl")
    assert len({(s["id"], s["solve"]) for s in solves}) == len(solves)
    # a third run is a no-op
    n = len(mock.requests)
    T.main(teacher_args(tmp_path, keyfile, mock))
    assert len(mock.requests) == n and len(read_jsonl(out / "results.jsonl")) == 6


def test_never_logs_the_key(tmp_path, keyfile, mock, capsys):
    rows = [item(i, source="B") for i in range(3)]
    for r in rows:
        mock.answers[r["field"]["question"]] = "b"
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
    T.main(teacher_args(tmp_path, keyfile, mock))                            # a clean run
    # a wrong-key run against a server that echoes the key it received in its 401 body
    other = tmp_path / "other"
    other.mkdir()
    write_jsonl(other / "stream/queue.jsonl", queue_rows([item(9, source="B")]))
    bad = MockAzure("a-different-server-key").start()
    try:
        code = T.main(teacher_args(other, keyfile, bad))
    finally:
        bad.stop()
    assert code == 2
    st = json.loads((other / "teacher/status.json").read_text())
    assert st["state"] == "fatal" and st["results"] == 0                     # a bad key never drops the queue
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert "<key>" in captured.out                                            # the echoed key was redacted
    for base in (tmp_path, other):
        for f in base.rglob("*"):
            if f.is_file() and f != keyfile:
                assert KEY not in f.read_text(errors="ignore"), f
    for body in mock.requests:
        assert KEY not in json.dumps(body)


# ------------------------------------------------------------------------------------------------ coordinator

def flagged_row(r, prio, shard="s000"):
    return {"id": r["id"], "shard": shard, "item": r, "mine": {"flags": ["wrong"]}, "priority": prio}


def coord_args(tmp_path, *extra):
    return C.parser().parse_args(["--no-pull", "--local-dir", str(tmp_path / "pulled"), "--queue", str(tmp_path / "stream/queue.jsonl"),
                                  "--status", str(tmp_path / "stream/status.json"), "--quotas", str(tmp_path / "quotas.json"),
                                  "--variants-queue", str(tmp_path / "stream/queue-variants.jsonl"),  # never the live queue
                                  "--teacher-status", str(tmp_path / "teacher/status.json"), "--expected-shards", "2", "--once",
                                  *extra])


def test_coordinator_quota_and_global_cap(tmp_path):
    a_rows = [flagged_row(item(i, family="fam_a"), prio=2.0 + i / 10) for i in range(5)]
    b_rows = [flagged_row(item(10 + i, family="fam_b"), prio=1.0) for i in range(4)]
    write_jsonl(tmp_path / "pulled/s000.flagged.jsonl", a_rows + b_rows)
    (tmp_path / "quotas.json").write_text(json.dumps({"global_cap": 5, "families": {"fam_a": 2}}))
    st = C.Coordinator(coord_args(tmp_path)).run()
    q = read_jsonl(tmp_path / "stream/queue.jsonl")
    assert [r["id"] for r in q] == ["it-4", "it-3", "it-10", "it-11", "it-12"]  # most confidently wrong first; fam_a full at 2
    assert st["blocked"] == {"quota:fam_a": 3, "global_cap": 1}
    # raising the quota and the cap admits waiting items on a restart, never twice
    (tmp_path / "quotas.json").write_text(json.dumps({"global_cap": 100, "families": {"fam_a": 3}}))
    C.Coordinator(coord_args(tmp_path)).run()
    ids = [r["id"] for r in read_jsonl(tmp_path / "stream/queue.jsonl")]
    assert len(ids) == len(set(ids)) == 7 and "it-2" in ids and "it-13" in ids and "it-1" not in ids


def test_coordinator_stops_on_teacher_cap_and_closes_when_mined(tmp_path):
    write_jsonl(tmp_path / "pulled/s000.flagged.jsonl", [flagged_row(item(i), 2.0) for i in range(3)])
    (tmp_path / "teacher").mkdir()
    (tmp_path / "teacher/status.json").write_text(json.dumps({"state": "cap_reached"}))
    st = C.Coordinator(coord_args(tmp_path)).run()
    assert st["stopped"] == "teacher cap_reached" and read_jsonl(tmp_path / "stream/queue.jsonl") == []
    (tmp_path / "teacher/status.json").write_text(json.dumps({"state": "running"}))
    for s in ("s000", "s001"):
        (tmp_path / f"pulled/{s}.done").write_text("{}")
    st = C.Coordinator(coord_args(tmp_path)).run()
    assert st["queued"] == 3 and st["mining_done"]
    assert (tmp_path / "stream/queue.jsonl.closed").exists()


def test_coordinator_ignores_partial_lines_and_excluded_ids(tmp_path):
    rows = [flagged_row(item(i), 2.0) for i in range(3)]
    path = tmp_path / "pulled/s000.flagged.jsonl"
    write_jsonl(path, rows[:2])
    with open(path, "a") as fh:
        fh.write(json.dumps(rows[2])[:40])                   # rsync mid-transfer / torn write
    (tmp_path / "heldout.txt").write_text("it-0\n")
    co = C.Coordinator(coord_args(tmp_path, "--exclude-ids", str(tmp_path / "heldout.txt")))
    co.cycle()
    assert [r["id"] for r in read_jsonl(tmp_path / "stream/queue.jsonl")] == ["it-1"]
    with open(path, "a") as fh:
        fh.write(json.dumps(rows[2])[40:] + "\n")
    co.cycle()
    assert [r["id"] for r in read_jsonl(tmp_path / "stream/queue.jsonl")] == ["it-1", "it-2"]


def test_pull_command_uses_ssh_alias_and_filters(tmp_path):
    seen = {}

    class R:
        returncode, stderr = 0, ""

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    assert C.pull("imajev-mine", "x/", tmp_path / "pulled", runner=runner)
    cmd = seen["cmd"]
    assert cmd[0] == "rsync" and "imajev-mine:x/" in cmd and "--include=*.flagged.jsonl" in cmd
    assert "--include=*.scores.jsonl" not in cmd and cmd.index("--exclude=*") > cmd.index("--include=*.done")
    C.pull("imajev-mine", "x", tmp_path / "pulled", scores=True, runner=runner)
    assert "--include=*.scores.jsonl" in seen["cmd"]


# ------------------------------------------------------------------------------------------------ mining worker

def mine_args(tmp_path, *extra):
    return M.parser().parse_args(["--backend", "fake", "--shards", str(tmp_path / "pool/*.jsonl"), "--out", str(tmp_path / "mine"),
                                  "--data-root", str(tmp_path / "data"), "--worker-id", "w0", "--chunk", "3", *extra])


def fake_fn(row, offset):
    i = int(row["id"].split("-")[1])
    labels = ["a", "b", "c", "__unknown__"]
    if i % 4 == 0:
        top = "a"                                       # wrong (gold b)
    elif i % 4 == 1:
        return {"a": 0.3, "b": 0.45, "c": 0.2, "__unknown__": 0.05}       # right but p(gold) < 0.6
    else:
        top = "b" if offset == 0 or i % 4 == 2 else "c"                   # i%4==3 flips under the rotation
    return {l: (0.85 if l == top else 0.05) for l in labels}


def test_mine_flags_and_outputs(tmp_path):
    rows = [item(i) for i in range(12)]
    write_jsonl(tmp_path / "pool/s000.jsonl", rows)
    M.run(mine_args(tmp_path, "--order-sample", "1.0"), scorer=M.FakeScorer(fake_fn))
    scores = {r["id"]: r for r in read_jsonl(tmp_path / "mine/s000.scores.jsonl")}
    flagged = {r["id"]: r for r in read_jsonl(tmp_path / "mine/s000.flagged.jsonl")}
    assert len(scores) == 12
    assert scores["it-0"]["flags"] == ["wrong"] and scores["it-1"]["flags"] == ["low_conf"]
    assert scores["it-2"]["flags"] == [] and scores["it-3"]["flags"] == ["order_flip"]
    assert set(flagged) == {f"it-{i}" for i in range(12) if i % 4 != 2}
    assert flagged["it-0"]["item"] == rows[0] and flagged["it-0"]["priority"] > flagged["it-1"]["priority"] > flagged["it-3"]["priority"]
    done = json.loads((tmp_path / "mine/s000.done").read_text())
    assert done["total_scored"] == 12 and done["total_flagged"] == 9


def test_mine_order_sample_is_about_twenty_percent():
    share = sum(M.in_order_sample(f"p3-x-{i}") for i in range(5000)) / 5000
    assert 0.18 < share < 0.22


def test_mine_resume_and_lock(tmp_path):
    rows = [item(i) for i in range(10)]
    write_jsonl(tmp_path / "pool/s000.jsonl", rows)
    write_jsonl(tmp_path / "pool/s001.jsonl", [item(100 + i) for i in range(4)])
    # another worker holds s001
    lock = M.ShardLock(tmp_path / "mine/locks/s001.lock", "other")
    assert lock.acquire()
    M.run(mine_args(tmp_path, "--limit", "4"), scorer=M.FakeScorer(fake_fn))
    assert len(read_jsonl(tmp_path / "mine/s000.scores.jsonl")) == 4 and not (tmp_path / "mine/s000.done").exists()
    assert not (tmp_path / "mine/s001.scores.jsonl").exists()
    # crash: a torn last line
    with open(tmp_path / "mine/s000.scores.jsonl", "a") as fh:
        fh.write('{"id": "it-9", "sta')
    lock.release()
    M.run(mine_args(tmp_path), scorer=M.FakeScorer(fake_fn))
    ids = [r["id"] for r in read_jsonl(tmp_path / "mine/s000.scores.jsonl")]
    assert sorted(ids) == sorted(r["id"] for r in rows) and len(ids) == 10
    flagged = [r["id"] for r in read_jsonl(tmp_path / "mine/s000.flagged.jsonl")]
    assert len(flagged) == len(set(flagged))
    assert (tmp_path / "mine/s000.done").exists() and (tmp_path / "mine/s001.done").exists()
    # a done shard is never rescored
    M.run(mine_args(tmp_path), scorer=M.FakeScorer(fake_fn))
    assert len(read_jsonl(tmp_path / "mine/s000.scores.jsonl")) == 10


def test_mine_image_items_and_invalid_rows(tmp_path):
    from PIL import Image
    (tmp_path / "data/imgs").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "data/imgs/a.png")
    good = item(1, images=["imgs/a.png"])
    missing = item(2, images=["imgs/missing.png"])
    unknown_gold = item(3, source="A", gold_kind="constructed", gold=None)
    write_jsonl(tmp_path / "pool/s000.jsonl", [good, missing, unknown_gold])
    seen = []

    class Spy(M.FakeScorer):
        def score(self, entries):
            seen.extend((e[0]["id"], len(e[2])) for e in entries)
            return super().score(entries)

    M.run(mine_args(tmp_path, "--order-sample", "0"), scorer=Spy(lambda r, o: {"a": 0.1, "b": 0.8, "c": 0.05, "__unknown__": 0.05}))
    scores = {r["id"]: r for r in read_jsonl(tmp_path / "mine/s000.scores.jsonl")}
    assert ("it-1", 1) in seen and scores["it-1"]["has_images"] and scores["it-1"]["flags"] == []
    assert scores["it-2"]["status"] == "invalid" and not scores["it-2"]["flagged"]
    assert scores["it-3"]["gold"] == "__unknown__" and scores["it-3"]["flags"] == ["wrong"]  # unknown gold, answered b


# ------------------------------------------------------------------------------------------------ variants

def variant_args(tmp_path, *extra):
    return V.parser().parse_args(["--results", str(tmp_path / "teacher/results.jsonl"), "--teacher-out", str(tmp_path / "teacher"),
                                  "--out", str(tmp_path / "variants"), "--queue", str(tmp_path / "stream/queue-variants.jsonl"),
                                  "--pool-shards", str(tmp_path / "pool/*.jsonl"), *extra])


def kept(row):
    return {"id": row["id"], "keep": True, "origin": "mine", "item": row}


def reasoning_parent():
    import gen_reasoning as gr
    for kind, (fn, fam, m) in sorted(gr.KINDS.items()):
        if fam == "table_arithmetic":
            rid = f"p3-ar-{kind}-d4-t1-000000"
            it = gr.gen_one(kind, 4, rid, 0)
            return gr.make_row(it, kind, 4, "t1", rid)
    raise AssertionError("no table kind")


def test_variants_programmatic_inherit_split_and_skip_teacher(tmp_path):
    parent = dict(reasoning_parent(), split="train")
    held = reasoning_parent()
    held = dict(held, id=held["id"].replace("-t1-", "-t2-"))
    write_jsonl(tmp_path / "teacher/results.jsonl", [kept(parent), kept(held), {"id": "x", "keep": False, "item": parent}])
    (tmp_path / "heldout.txt").write_text(held["id"] + "\n")
    a = variant_args(tmp_path, "--no-writer", "--unknown-share", "1.0", "--heldout-ids", str(tmp_path / "heldout.txt"))
    a.writer_name, a.writer_concurrency = None, 4
    V.Variants(a, None).run()
    cons = read_jsonl(tmp_path / "variants/constructed.jsonl")
    rej = read_jsonl(tmp_path / "variants/rejects.jsonl")
    assert len(cons) + len(rej) == 4 and all("overlap" in r["reason"] for r in rej)
    for r in cons:
        assert validate(r) == [] and r["gold_kind"] == "constructed" and r["gold"] is not None
        assert r["split"] == ("heldout" if r["parent_id"] == held["id"] else "train")
    q = read_jsonl(tmp_path / "stream/queue-variants.jsonl")                 # unknown variants go to the teacher
    assert len(q) == 2 and all(r["variant"]["kind"] == "unknown" and r["item"]["gold"] is None for r in q)
    assert {r["item"]["split"] for r in q} == {"train", "heldout"}
    # idempotent
    V.Variants(a, None).run()
    assert len(read_jsonl(tmp_path / "variants/constructed.jsonl")) == len(cons) and len(read_jsonl(tmp_path / "stream/queue-variants.jsonl")) == 2


def test_variants_writer_path_then_teacher(tmp_path, keyfile, mock):
    parent = item(1, source="B")
    parent["state"] = "The Fern Street depot shipped 41 crates on Monday and 17 on Tuesday; the contract caps weekly shipments at 60 crates."
    write_jsonl(tmp_path / "teacher/results.jsonl", [kept(parent)])
    a = variant_args(tmp_path, "--unknown-share", "1.0", "--close")
    a.writer_name, a.writer_concurrency = "writer-27b", 4
    V.Variants(a, T.AzureClient(mock.base_url, KEY, sleep=lambda s: None)).run()
    q = read_jsonl(tmp_path / "stream/queue-variants.jsonl")
    kinds = sorted(r["variant"]["kind"] for r in q)
    assert kinds == ["perturb", "perturb", "unknown"]
    for r in q:
        it = r["item"]
        assert validate(it) == [] and it["parent_id"] == parent["id"] and it["gold_kind"] == "none"
        assert "42 crates" in it["state"] or "42" in it["state"]
    writer_bodies = [b for b in mock.requests if b["response_format"]["json_schema"]["name"] == "decision_variant"]
    assert len(writer_bodies) == 3 and all(b["model"] == "writer-27b" for b in writer_bodies)
    assert (tmp_path / "stream/queue-variants.jsonl.closed").exists()
    # the teacher verifies them: perturb variants need two solves agreeing with the writer's answer; unknown needs unknown mass
    mock.custom = lambda body, labels: P("b", labels)      # the teacher always answers b: perturb kept, unknown dropped
    T.main(teacher_args(tmp_path, keyfile, mock, queue=tmp_path / "stream/queue-variants.jsonl"))
    res = {r["id"]: r for r in read_jsonl(tmp_path / "teacher/results.jsonl") if r.get("origin") == "variant"}
    assert len(res) == 3
    for r in res.values():
        if r["variant"]["kind"] == "perturb":
            assert r["keep"] and r["rule"] == "writer_variant"
        else:
            assert not r["keep"] and r["reason"] == "teacher_not_unknown"


def test_variants_overlap_rule():
    p = item(1)
    p["state"] = " ".join(f"word{i}" for i in range(100))
    near = dict(p, id="v")
    far = dict(p, id="w", state=" ".join(f"other{i}" for i in range(100)))
    assert V.overlap(near, p) > 0.7 and V.overlap(far, p) < 0.05     # only the shared question tail overlaps


# ------------------------------------------------------------------------------------------------ integration (phase-3 dry run)

def gate_args(tmp_path, keyfile, mock, *extra, **kw):
    return teacher_args(tmp_path, keyfile, mock, "--deployment", "imajev-teacher:8", *extra, **kw)


def test_kept_rows_carry_the_manifest_label_contract(tmp_path, keyfile, mock):
    import assembly_manifest as am
    rows = [item(1, source="A", gold_kind="constructed"), item(2, source="B"), item(3, source="A", gold_kind="constructed", gold="c"),
            item(4, source="C", gold_kind="dataset", ftype="score", gold=2)]
    for r in rows:
        mock.answers[r["field"]["question"]] = gold_label(r) if r["id"] != "it-3" else "a"     # it-3: teacher disagrees
    write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
    assert T.main(teacher_args(tmp_path, keyfile, mock)) == 0
    kept = {r["id"]: r for r in read_jsonl(tmp_path / "teacher/results.jsonl")}      # finalize would cap it-3 (25% of A)
    assert all(r["keep"] for r in kept.values()) and set(kept) == {"it-1", "it-2", "it-3", "it-4"}
    assert kept["it-1"]["label"]["target_kind"] == "teacher" and kept["it-1"]["label"]["probs"]["b"] > 0.5
    assert kept["it-3"]["label"] == {"target": "c", "probs": None, "rationale": None, "target_kind": "gold", "review": None}
    assert kept["it-4"]["label"]["target"] == 2 and set(kept["it-4"]["label"]["probs"]) == {"0", "1", "2", "unknown"}
    for r in kept.values():                                  # what build_manifest.py does with them
        cand = am.unwrap_verified(r)
        rec = am.candidate_to_record(cand, am.parse_label(cand), group="g", partition="train", mix="hard_text")
        assert am.check_record(rec) is None
    assert am.unwrap_verified({"id": "x", "keep": False, "item": rows[0]}) is None


def test_pilot_gate_pauses_dispatch_until_resume(tmp_path, keyfile):
    import threading
    m = MockAzure(KEY, policy="realistic", parse_error_rate=0.5, latency=0.02).start()
    try:
        rows = [item(i, source="A", gold_kind="constructed") for i in range(40)]
        write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
        rc = {}
        t = threading.Thread(target=lambda: rc.setdefault("rc", T.main(gate_args(tmp_path, keyfile, m, "--gate-items", "10",
                                                                                       "--per-instance-concurrency", "1"))))
        t.start()
        gate = tmp_path / "teacher/pilot-gate.json"
        for _ in range(300):
            if gate.exists():
                break
            time.sleep(0.05)
        st = json.loads(gate.read_text())
        assert st["decision"] == "paused" and "parse errors" in st["why"][0]
        assert (tmp_path / "teacher/PILOT_GATE_ALERT.txt").read_text().startswith("PILOT GATE: DISPATCH PAUSED")
        time.sleep(1.0)                                              # in-flight requests drain
        n = len(m.requests)
        time.sleep(1.0)
        assert len(m.requests) == n and t.is_alive()                 # nothing new is sent while paused
        assert json.loads((tmp_path / "teacher/status.json").read_text())["state"] == T.PAUSED
        assert T.main(["resume", "--out", str(tmp_path / "teacher")]) == 0
        t.join(60)
        assert not t.is_alive() and rc["rc"] == 0
        assert json.loads(gate.read_text())["decision"] == "resumed"
        assert len(read_jsonl(tmp_path / "teacher/results.jsonl")) == 40
    finally:
        m.stop()


def test_pilot_gate_passes_once_and_persists(tmp_path, keyfile):
    m = MockAzure(KEY, policy="realistic", accuracy=1.0).start()
    try:
        rows = [item(i, source="A", gold_kind="constructed") for i in range(12)]
        write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows(rows))
        assert T.main(gate_args(tmp_path, keyfile, m, "--gate-items", "10")) == 0
        st = json.loads((tmp_path / "teacher/pilot-gate.json").read_text())
        assert st["decision"] == "pass" and st["metrics"]["parse_error_rate"] == 0.0
        assert set(st["metrics"]) >= {"truncation_rate", "keep_rate_by_source", "constructed_keep_rate", "mean_completion_tokens",
                                      "usd_per_item"}
        # a restart does not re-evaluate, and --no-pilot-gate never writes one
        write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([item(99, source="A", gold_kind="constructed")]))
        assert T.main(gate_args(tmp_path, keyfile, m, "--gate-items", "10")) == 0
        assert json.loads((tmp_path / "teacher/pilot-gate.json").read_text())["t"] == st["t"]
        other = tmp_path / "t2"
        assert T.main(gate_args(other, keyfile, m, "--no-pilot-gate", queue=tmp_path / "stream/queue.jsonl")) == 0
        assert not (other / "teacher/pilot-gate.json").exists()
    finally:
        m.stop()


def test_pilot_gate_rules():
    g = T.PilotGate.__new__(T.PilotGate)
    g.th = dict(T.GATE)
    ok = {"parse_error_rate": 0.01, "truncation_rate": 0.02, "constructed_keep_rate": 0.8, "constructed_items": 200}
    assert g.evaluate(ok) == []
    assert "truncation" in g.evaluate(dict(ok, truncation_rate=0.031))[0]
    assert "constructed gold" in g.evaluate(dict(ok, constructed_keep_rate=0.59))[0]
    assert g.evaluate(dict(ok, constructed_keep_rate=0.1, constructed_items=5)) == []      # too few constructed items to judge
    assert T.MAX_USD == 145.0 and T.parser().parse_args(["run", "--deployment", "x:8"]).max_usd == 145.0


def test_truncation_and_parse_errors_are_counted_per_response(tmp_path, keyfile):
    m = MockAzure(KEY, policy="realistic", truncation_rate=1.0).start()
    try:
        write_jsonl(tmp_path / "stream/queue.jsonl", queue_rows([item(1, source="A", gold_kind="constructed")]))
        T.main(teacher_args(tmp_path, keyfile, m, "--no-pilot-gate"))
        s = read_jsonl(tmp_path / "teacher/solves.jsonl")[0]              # every dispatch truncates; retried, then dropped
        assert s["responses"] == 2 and s["truncated"] == 2 and s["parse_errors"] == 0 and not s["ok"]
    finally:
        m.stop()


def test_coordinator_routes_bucket_rows_to_heldout_and_skips_fresh_ids(tmp_path):
    def pooled(r, bucket=False, key=None):
        return dict(r, pool={"group": "g:" + r["id"], "quota_key": key or r["family"], "shard": 0, "heldout_flagged_bucket": bucket})
    rows = [flagged_row(pooled(item(i)), 2.0) for i in range(4)]
    rows += [flagged_row(pooled(item(10 + i), bucket=True), 2.5) for i in range(3)]
    rows += [flagged_row(pooled(item(20), key="img:fam_a"), 2.9)]
    write_jsonl(tmp_path / "pulled/s000.flagged.jsonl", rows)
    (tmp_path / "fresh.jsonl").write_text(json.dumps({"id": "it-0", "state": "x"}) + "\n")
    (tmp_path / "quotas.json").write_text(json.dumps({"cap_total_teacher_calls": 100, "heldout_flagged_cap": 2,
                                                      "families": {"fam_a": {"quota": 2}, "img:fam_a": {"quota": 1}}}))
    st = C.Coordinator(coord_args(tmp_path, "--exclude-ids", str(tmp_path / "fresh.jsonl"), "--no-default-exclude")).run()
    q = read_jsonl(tmp_path / "stream/queue.jsonl")
    held = [r for r in q if r["origin"] == "heldout"]
    mine = [r for r in q if r["origin"] == "mine"]
    assert len(held) == 2 and all(r["item"]["pool"]["heldout_flagged_bucket"] for r in held)
    assert not any(r["item"]["pool"]["heldout_flagged_bucket"] for r in mine)
    assert "it-0" not in {r["id"] for r in q} and st["heldout_fresh_ids_skipped"] == 1
    assert [r["id"] for r in mine] == ["it-20", "it-1", "it-2"]          # img: quota key counted apart; fam_a quota 2 of it-1..3
    assert st["by_family"]["fam_a"] == {"queued": 2, "quota": 2} and st["blocked"]["heldout_cap"] == 1


def test_coordinator_default_exclude_reads_pool_heldout_fresh(tmp_path):
    (tmp_path / "pool/shards").mkdir(parents=True)
    (tmp_path / "pool/heldout-fresh.jsonl").write_text(json.dumps({"id": "it-1"}) + "\n")
    (tmp_path / "pool/heldout-fresh-gui.jsonl").write_text(json.dumps({"id": "it-2"}) + "\n")
    write_jsonl(tmp_path / "pulled/s000.flagged.jsonl", [flagged_row(item(i), 2.0) for i in range(4)])
    st = C.Coordinator(coord_args(tmp_path, "--pool-shards", str(tmp_path / "pool/shards/*.jsonl"))).run()
    assert [r["id"] for r in read_jsonl(tmp_path / "stream/queue.jsonl")] == ["it-0", "it-3"] and st["heldout_fresh_ids_skipped"] == 2


def test_coordinator_cap_counts_teacher_calls(tmp_path):
    write_jsonl(tmp_path / "pulled/s000.flagged.jsonl", [flagged_row(item(i), 2.0) for i in range(10)])
    (tmp_path / "quotas.json").write_text(json.dumps({"cap_total_teacher_calls": 8}))
    (tmp_path / "teacher").mkdir()
    (tmp_path / "teacher/status.json").write_text(json.dumps({"state": "running", "second_solves": 3}))
    write_jsonl(tmp_path / "vq.jsonl", [{"id": "v1"}, {"id": "v2"}])
    st = C.Coordinator(coord_args(tmp_path, "--variants-queue", str(tmp_path / "vq.jsonl"))).run()
    assert st["queued"] == 3 and st["estimated_calls"] == 8 and st["blocked"]["global_cap"] == 7


def test_pull_from_a_local_directory_uses_no_ssh(tmp_path):
    src = tmp_path / "pod/mine"
    write_jsonl(src / "s000.flagged.jsonl", [flagged_row(item(1), 2.0)])
    (src / "s000.done").write_text("{}")
    write_jsonl(src / "s000.scores.jsonl", [{"id": "it-1"}])
    assert C.pull(None, str(src), tmp_path / "pulled")
    assert (tmp_path / "pulled/s000.flagged.jsonl").exists() and (tmp_path / "pulled/s000.done").exists()
    assert not (tmp_path / "pulled/s000.scores.jsonl").exists()


def test_finalize_routes_heldout_verdicts_apart(tmp_path):
    out = tmp_path / "teacher"
    r1 = item(1, source="A", gold_kind="constructed")
    rows = [{"id": "it-1", "keep": True, "rule": "A", "target": "teacher_dist", "target_probs": P("b"), "origin": "mine", "item": r1},
            {"id": "it-2", "keep": True, "rule": "A", "target": "teacher_dist", "target_probs": P("b"), "origin": "heldout",
             "item": item(2, source="A", gold_kind="constructed")},
            {"id": "it-3", "keep": False, "rule": "B", "reason": "solves_disagree", "origin": "heldout", "item": item(3)}]
    write_jsonl(out / "results.jsonl", rows)
    s = T.finalize(out)
    assert [r["id"] for r in read_jsonl(out / "kept.jsonl")] == ["it-1"]
    assert read_jsonl(out / "kept.jsonl")[0]["label"]["target"] == "b"                      # backfilled label
    assert sorted(r["id"] for r in read_jsonl(out / "heldout-results.jsonl")) == ["it-2", "it-3"]
    assert s["heldout_results"] == 2 and s["heldout_kept"] == 1


def test_variants_no_writer_finishes_and_labels_constructed(tmp_path):
    parent = dict(reasoning_parent(), split="train")
    bc = item(5, source="B")
    write_jsonl(tmp_path / "teacher/results.jsonl", [kept(parent), kept(bc),
                                                     {"id": "h", "keep": True, "origin": "heldout", "item": item(6, source="B")}])
    (tmp_path / "stream").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stream/queue.jsonl.closed").write_text("{}")
    (tmp_path / "teacher/status.json").write_text(json.dumps({"state": "done", "pending": 0, "inflight": 0, "results": 3}))
    a = variant_args(tmp_path, "--no-writer", "--follow", "--interval", "0.01", "--stop-when-closed", "--close",
                     "--upstream-queues", str(tmp_path / "stream/queue.jsonl"))
    a.writer_name, a.writer_concurrency = None, 4
    counts = V.Variants(a, None).run()                                # must terminate
    assert counts["skipped_no_writer"] == 1 and (tmp_path / "stream/queue-variants.jsonl.closed").exists()
    parents = {r["parent_id"]: r["method"] for r in read_jsonl(tmp_path / "variants/parents.jsonl")}
    assert parents == {parent["id"]: "programmatic", bc["id"]: "skipped_no_writer"}          # the held-out item is never a parent
    for r in read_jsonl(tmp_path / "variants/constructed.jsonl"):
        assert r["label"] == {"target": r["gold"], "probs": None, "rationale": None, "target_kind": "constructed", "review": None}
    assert V.main(["--results", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "v2"), "--queue", str(tmp_path / "vq.jsonl"),
                   "--teacher-out", str(tmp_path / "teacher")]) == 0          # no writer deployment: clean no-writer mode


def test_mine_fake_accuracy_gives_a_realistic_flag_rate(tmp_path):
    rows = [item(i) for i in range(400)]
    write_jsonl(tmp_path / "pool/pool-00.jsonl", rows)
    M.run(mine_args(tmp_path, "--fake-accuracy", "0.7", "--chunk", "100"))
    sc = read_jsonl(tmp_path / "mine/pool-00.scores.jsonl")
    wrong = sum("wrong" in r["flags"] for r in sc) / len(sc)
    assert 0.2 < wrong < 0.4
