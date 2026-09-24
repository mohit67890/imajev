"""decision-p2 teacher pipeline: prompt rendering, writer JSON validation, option keys, answer parsing, agreement filter,
unknown share, partition, licence receipt and contamination lint. No model is run; clients are faked."""
import json, random, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p2")); sys.path.insert(0, str(ROOT / "scripts"))
from p2_common import (ANSWERERS, ChatClient, WQuestion, WRITER_MODEL, WriterOutput, contamination_count, option_key, parse_answer,  # noqa: E402
                       reference_ngrams, to_field)
import families, domains, gen_write, gen_answer, assemble_p2  # noqa: E402


def make_writer_output(doc_words=250, shape="string"):
    doc = " ".join(f"word{i}" for i in range(doc_words))
    document = doc if shape == "string" else {"ticket": {"id": "T-1", "body": doc}, "rules": [{"id": 1, "text": "Refunds within 14 days."}]}
    return {"document": document, "document_kind": "ticket", "questions": [
        {"family": "policy_exception", "type": "choice", "question": "Which outcome applies to the case described in the ticket?",
         "options": [{"text": "Full refund", "description": "Refund the whole amount."}, {"text": "Partial refund", "description": "Refund part."},
                     {"text": "No refund", "description": "Refund nothing."}], "intended": "Partial refund", "unknown_reason": None,
         "justification": "Section 2 exception."},
        {"family": "date_number_trap", "type": "noul", "question": "Was the claim filed within the 14-day window?", "intended": False,
         "unknown_reason": None, "justification": "Filed on day 16."},
        {"family": "insufficient_evidence", "type": "score", "question": "How severe is the incident on the stated scale?",
         "levels": [{"value": 1, "description": "low"}, {"value": 2, "description": "medium"}, {"value": 3, "description": "high"}],
         "intended": None, "unknown_reason": "insufficient_evidence", "justification": "Severity field is blank."}]}


def test_domains_and_families_are_complete():
    assert len(domains.DOMAINS) == 24 and len(families.FAMILY_IDS) == 17
    for fam in families.FAMILY_IDS:
        assert len(families.BRIEFS[fam]) >= 6
    rng = random.Random(0)
    fams = families.pick_families(rng)
    assert len(fams) == 3 and len(set(fams)) == 3 and all(f in families.FAMILIES for f in fams)


def test_plan_is_deterministic_and_prompt_renders():
    a, b = gen_write.plan_document(7), gen_write.plan_document(7)
    assert a == b and a["domain"] == domains.DOMAINS[7 % 24]["id"] and len(a["families"]) == 3
    messages = gen_write.render(a)
    assert messages[0]["role"] == "system" and "JSON" in messages[1]["content"]
    for fam in a["families"]:
        assert f'family "{fam}"' in messages[1]["content"]
    assert ("JSON object under \"document\"" in messages[1]["content"]) == (a["state_shape"] == "object")


def test_unknown_share_of_plans_is_in_range():
    plans = [gen_write.plan_document(i) for i in range(2000)]
    share = sum(u for p in plans for u in p["unknowns"]) / (3 * len(plans))
    assert 0.15 <= share <= 0.30, share


def test_writer_output_validation_and_option_keys():
    out = WriterOutput.model_validate(make_writer_output())
    field, target = to_field(out.questions[0])
    assert field["type"] == "choice" and [o["value"] for o in field["options"]] == ["full_refund", "partial_refund", "no_refund"]
    assert target == "partial_refund"
    assert to_field(out.questions[1]) == ({"id": "decision", "type": "boolean", "question": "Was the claim filed within the 14-day window?"}, False)
    f3, t3 = to_field(out.questions[2]); assert f3["type"] == "ordinal" and t3 is None
    assert option_key("Unknown") == "option_unknown"
    taken = set(); assert option_key("A b", taken) == "a_b" and option_key("a-b", taken) == "a_b_2"
    bad = make_writer_output(); bad["questions"][0]["intended"] = "Nope"
    with pytest.raises(Exception): WriterOutput.model_validate(bad)
    bad = make_writer_output(); bad["questions"][2]["unknown_reason"] = None
    with pytest.raises(Exception): WriterOutput.model_validate(bad)
    short = make_writer_output(doc_words=30)
    with pytest.raises(Exception): WriterOutput.model_validate(short)


def test_validate_enforces_plan():
    plan = {"families": ["policy_exception", "date_number_trap", "insufficient_evidence"], "types": ["choice", "noul", "score"],
            "unknowns": [False, False, True], "state_shape": "string"}
    out, err = gen_write.validate(json.dumps(make_writer_output()), plan); assert err is None and out["questions"][0]["family"] == "policy_exception"
    plan2 = dict(plan, types=["noul", "noul", "score"]); assert gen_write.validate(json.dumps(make_writer_output()), plan2)[1].startswith("type mismatch")
    plan3 = dict(plan, state_shape="object"); assert "state shape" in gen_write.validate(json.dumps(make_writer_output()), plan3)[1]
    assert gen_write.validate("not json", plan)[0] is None


def test_parse_answer_all_types():
    choice = {"type": "choice", "options": [{"value": "full_refund", "description": "Refund the whole amount."}, {"value": "no_refund", "description": "Refund nothing."}]}
    assert parse_answer('{"answer": "full_refund", "confidence": 0.8}', choice) == ("full_refund", 0.8, None)
    assert parse_answer('```json\n{"answer": "Full refund"}\n```', choice)[0] == "full_refund"
    assert parse_answer('{"answer": "unknown", "confidence": 0.5}', choice) == (None, 0.5, None)
    assert parse_answer('{"answer": "maybe"}', choice)[2].startswith("bad option")
    boolean = {"type": "boolean"}
    assert parse_answer('{"answer": true}', boolean)[0] is True and parse_answer('{"answer": "no"}', boolean)[0] is False
    ordinal = {"type": "ordinal", "levels": [{"value": 1, "description": "low"}, {"value": 2, "description": "high"}]}
    assert parse_answer('{"answer": 2}', ordinal)[0] == 2 and parse_answer('{"answer": "2"}', ordinal)[0] == 2
    assert parse_answer('{"answer": 7}', ordinal)[2].startswith("bad level")
    assert parse_answer("garbage", ordinal)[2].startswith("json")


def test_chat_client_retries_and_returns_content():
    calls = []

    class Resp:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(self.body).encode()

    def opener(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1:
            raise TimeoutError("first call times out")
        return Resp({"choices": [{"message": {"content": "{\"answer\": true}"}}]})

    client = ChatClient("http://x/v1", "m", opener=opener, sleep=lambda s: None)
    assert client.complete([{"role": "user", "content": "hi"}], extra={"reasoning_effort": "medium"}) == '{"answer": true}'
    assert len(calls) == 2 and calls[-1]["reasoning_effort"] == "medium" and calls[-1]["response_format"]["type"] == "json_object"


def test_agreement_filter_partition_licence_and_report(tmp_path):
    writer_rows = []
    for i in range(40):
        plan = gen_write.plan_document(i)
        writer_rows.append({"doc_id": plan["doc_id"], "plan": plan, "writer": WRITER_MODEL, "output": make_writer_output(shape=plan["state_shape"])})
    answers = []
    for w in writer_rows:
        for qi, intended in enumerate(["partial_refund", False, None]):
            for name in ("qwen", "gptoss"):
                value = intended
                if qi == 0 and name == "gptoss" and w["plan"]["index"] % 4 == 0:
                    value = "no_refund"  # disagreement on a quarter of the choice questions
                answers.append({"doc_id": w["doc_id"], "qi": qi, "answerer": name, "repo": ANSWERERS[name]["repo"],
                                "revision": ANSWERERS[name]["revision"], "raw": "", "value": value, "confidence": 0.9, "parse_error": None})
    evidence = assemble_p2.write_license_evidence(tmp_path / "lic", WRITER_MODEL, ANSWERERS)
    from v1_text.common import verified_license
    lic = verified_license(evidence, "Apache-2.0"); assert lic["spdx"] == "Apache-2.0"
    kept, report = assemble_p2.build_records(writer_rows, answers, 2, lic, reference=set())
    assert report["counts"]["questions"] == 120 and report["counts"]["disagreed"] == 10 and len(kept) == 110
    assert abs(report["unknown_share"] - 40 / 110) < 1e-9
    parts = {r["source_group"]: r["partition"] for r in kept}
    assert set(parts.values()) <= {"train", "dev", "test"} and all(r["partition"] == parts[r["source_group"]] for r in kept)
    rec = next(r for r in kept if r["target"] is None)
    assert rec["abstention_cause"] == "insufficient_evidence" and rec["request"]["fields"][0]["type"] == "ordinal"
    rec = next(r for r in kept if r["request"]["fields"][0]["type"] == "choice")
    assert rec["target"] == "partial_refund" and rec["provenance"]["agreement"] is True and len(rec["provenance"]["answerers"]) == 2
    from vision_decision.contracts import Request
    for r in kept[:10]:
        Request.model_validate(r["request"])


def test_contamination_lint_drops_overlaps(tmp_path):
    ref = tmp_path / "hard.jsonl"
    ref.write_text(json.dumps({"id": "x", "state": "the quick brown fox jumps over the lazy dog every single morning before breakfast"}) + "\n")
    grams = reference_ngrams([ref])
    assert grams and contamination_count("The quick brown fox jumps over the lazy dog every single morning", grams) >= 2
    assert contamination_count("An unrelated invoice for three boxes of paper", grams) == 0
    plan = gen_write.plan_document(1)
    w = {"doc_id": plan["doc_id"], "plan": plan, "writer": WRITER_MODEL, "output": make_writer_output(shape=plan["state_shape"])}
    doc = "the quick brown fox jumps over the lazy dog every single morning before breakfast " + " ".join(f"w{i}" for i in range(200))
    w["output"]["document"] = doc if plan["state_shape"] == "string" else {"body": doc}
    answers = [{"doc_id": w["doc_id"], "qi": qi, "answerer": n, "value": v, "confidence": 1.0, "parse_error": None}
               for qi, v in enumerate(["partial_refund", False, None]) for n in ("qwen", "gptoss")]
    evidence = assemble_p2.write_license_evidence(tmp_path / "lic", WRITER_MODEL, ANSWERERS)
    from v1_text.common import verified_license
    kept, report = assemble_p2.build_records([w], answers, 2, verified_license(evidence, "Apache-2.0"), grams)
    assert kept == [] and report["counts"]["contaminated"] == 3


def test_gen_answer_questions_and_prompt():
    plan = gen_write.plan_document(3)
    rows = [{"doc_id": plan["doc_id"], "plan": plan, "writer": WRITER_MODEL, "output": make_writer_output(shape=plan["state_shape"])}]
    qs = list(gen_answer.questions(rows)); assert len(qs) == 3 and qs[0]["field"]["type"] == "choice"
    prompt = families.answerer_prompt(qs[0]["state"], qs[0]["field"])
    assert "full_refund" in prompt and "unknown" in prompt and "intended" not in prompt.lower()


# ---------------------------------------------------------------------------------------------------------------------
# phase-2b: restricted families + batch tags, relaxed unknown rule, batch filter / append, near-duplicate lint, torch rotations

def test_plan_families_and_tag():
    fams = ["probability_estimate", "tradeoff", "multi_step_lookup"]
    plans = [gen_write.plan_document(i, families=fams, tag="p2b-hard") for i in range(12)]
    for p in plans:
        assert p["doc_id"].startswith("p2b-hard-") and p["batch"] == "p2b-hard" and set(p["families"]) <= set(fams) and len(set(p["families"])) == 3
        for fam, ty in zip(p["families"], p["types"]):
            assert families.forced_type(fam) in (None, ty)
        assert p["types"][p["families"].index("probability_estimate")] == "score" and p["types"][p["families"].index("tradeoff")] == "choice"
    assert len({p["doc_id"] for p in plans}) == 12
    assert gen_write.plan_document(3, tag="p2b-hard")["doc_id"] != gen_write.plan_document(3)["doc_id"]
    a, b = gen_write.plan_document(3, tag="p2b-hard"), gen_write.plan_document(3, tag="p2b-unknown")
    assert a["doc_id"] != b["doc_id"] and a["domain"] == b["domain"]  # the tag reseeds the plan; the domain stays index-based
    assert gen_write.plan_document(5)["batch"] == "p2" and not gen_write.plan_document(5)["doc_id"].startswith("p2b")
    two = gen_write.plan_document(0, families=["ambiguity", "contradiction"], tag="p2b-unknown")
    assert {"ambiguity", "contradiction"} <= set(two["families"]) and len(set(two["families"])) == 3
    with pytest.raises(SystemExit):
        gen_write.main(["--docs", "1", "--out", "/dev/null", "--families", "no_such_family"])


def test_hedged_and_relaxed_unknown_rule():
    assert assemble_p2.hedged("The document does not say; the answer cannot be determined.") and not assemble_p2.hedged('{"answer": "no_refund"}')
    unknown = {"answerer": "qwen", "value": None, "parse_error": None, "raw": '{"answer": "unknown"}'}
    hedged_guess = {"answerer": "gptoss", "value": "no_refund", "parse_error": None, "raw": 'Not stated explicitly; {"answer": "no_refund", "confidence": 0.4}'}
    confident = {"answerer": "gptoss", "value": "no_refund", "parse_error": None, "raw": '{"answer": "no_refund", "confidence": 0.95}'}
    key = ("d", 0)
    assert assemble_p2.agreement({key: [unknown, hedged_guess]}, key, None, 2, "strict") == (False, [unknown, hedged_guess])
    assert assemble_p2.agreement({key: [unknown, hedged_guess]}, key, None, 2, "relaxed")[0] is True
    assert assemble_p2.agreement({key: [unknown, confident]}, key, None, 2, "relaxed")[0] is False
    assert assemble_p2.agreement({key: [confident, dict(confident, answerer="qwen")]}, key, None, 2, "relaxed")[0] is False  # nobody says unknown
    assert assemble_p2.agreement({key: [unknown, dict(unknown, answerer="gptoss")]}, key, None, 2, "relaxed")[0] is True
    assert assemble_p2.agreement({key: [unknown, hedged_guess]}, key, "no_refund", 2, "relaxed")[0] is False  # answerable questions stay strict
    assert assemble_p2.agreement({key: [unknown]}, key, None, 2, "relaxed")[0] is False  # missing answerer


def _writer_row(i, tag=None, families=None, doc=None):
    plan = gen_write.plan_document(i, families=families, tag=tag)
    out = make_writer_output(shape=plan["state_shape"])
    if doc is not None:
        out["document"] = doc if plan["state_shape"] == "string" else {"body": doc}
    return {"doc_id": plan["doc_id"], "batch": plan["batch"], "plan": plan, "writer": WRITER_MODEL, "output": out}


def _answers(rows, values=("partial_refund", False, None)):
    return [{"doc_id": w["doc_id"], "qi": qi, "answerer": n, "repo": ANSWERERS[n]["repo"], "revision": ANSWERERS[n]["revision"], "raw": "",
             "value": v, "confidence": 0.9, "parse_error": None} for w in rows for qi, v in enumerate(values) for n in ("qwen", "gptoss")]


def test_batch_filter_and_append_to(tmp_path):
    from p2_common import read_jsonl
    from v1_text.common import write_jsonl
    def doc(k): return " ".join(f"w{k}_{j}" for j in range(220))   # distinct documents, or the within-batch near-duplicate lint drops them
    old = [_writer_row(i, doc=doc(i)) for i in range(4)]
    new = [_writer_row(i, tag="p2b-hard", doc=doc(10 + i)) for i in range(4)] + [_writer_row(i, tag="p2b-unknown", doc=doc(20 + i)) for i in range(2)]
    write_jsonl(tmp_path / "writer.jsonl", old + new); write_jsonl(tmp_path / "answers.jsonl", _answers(old + new))
    existing = [{"id": f"p2_teacher:{new[0]['doc_id']}:0", "source_group": new[0]["doc_id"], "partition": "train"}, {"id": "p2_teacher:old-doc:0", "partition": "train"}]
    write_jsonl(tmp_path / "existing.jsonl", existing)
    argv = ["--writer", str(tmp_path / "writer.jsonl"), "--answers", str(tmp_path / "answers.jsonl"), "--out", str(tmp_path / "records.jsonl"),
            "--license-dir", str(tmp_path / "lic"), "--reference", "--dedupe-against", "--batch-filter", "p2b-*", "--append-to", str(tmp_path / "existing.jsonl"), "--unknown-rule", "relaxed"]
    assert assemble_p2.main(argv) == 0
    rows = read_jsonl(tmp_path / "records.jsonl"); report = json.loads((tmp_path / "assembly-report.json").read_text())
    assert report["counts"]["writer_docs"] == 6 and report["new_records"] == 17 and report["existing_records"] == 2 and len(rows) == 19
    assert len({r["id"] for r in rows}) == 19 and rows[:2] == existing
    assert {r["batch"] for r in rows[2:]} == {"p2b-hard", "p2b-unknown"}
    assert all(r["provenance"]["unknown_rule"] == ("relaxed" if r["target"] is None else "strict") for r in rows[2:])
    assert not any(r.get("source_group") == old[0]["doc_id"] for r in rows)   # untagged rows were filtered out


def test_near_duplicate_lint_drops_lookalike_documents(tmp_path):
    from v1_text.common import write_jsonl
    base = " ".join(f"token{i}" for i in range(220))
    lookalike = base.replace("token100", "tokenX")             # shares hundreds of 8-grams with base
    fresh = " ".join(f"other{i}" for i in range(220))         # shares none
    prior = _writer_row(0, doc=base)                          # an earlier (phase-2) document
    a, b, c = _writer_row(1, tag="p2b-hard", doc=lookalike), _writer_row(2, tag="p2b-hard", doc=fresh), _writer_row(3, tag="p2b-hard", doc=fresh.replace("other7 ", "other7b "))
    write_jsonl(tmp_path / "prior-writer.jsonl", [prior])
    human = [{"id": "h1", "source_group": "hg1", "request": {"state": fresh.replace("other0 ", "zzz ")}}]
    write_jsonl(tmp_path / "human.jsonl", human)
    idx = assemble_p2.NearDuplicateIndex.from_files([tmp_path / "prior-writer.jsonl"], threshold=5)
    assert idx.is_duplicate(lookalike)[0] and not idx.is_duplicate(fresh)[0]
    from v1_text.common import verified_license
    lic = verified_license(assemble_p2.write_license_evidence(tmp_path / "lic", WRITER_MODEL, ANSWERERS), "Apache-2.0")
    kept, report = assemble_p2.build_records([a, b, c], _answers([a, b, c]), 2, lic, reference=set(), dedupe=idx)
    assert report["counts"]["near_duplicates_dropped"] == 2 and report["counts"]["near_duplicate_questions_dropped"] == 6
    assert {r["source_group"] for r in kept} == {b["doc_id"]} and len(kept) == 3        # a: vs prior; c: vs b within the batch
    pairs = report["near_duplicate_pairs"]; assert {(p["new"], p["existing"]) for p in pairs} == {(a["doc_id"], prior["doc_id"]), (c["doc_id"], b["doc_id"])} and all(p["shared_8grams"] >= 5 for p in pairs)
    idx2 = assemble_p2.NearDuplicateIndex.from_files([tmp_path / "prior-writer.jsonl", tmp_path / "human.jsonl"], exclude_ids={prior["doc_id"]}, threshold=5)
    assert idx2.is_duplicate(fresh)[0] and not idx2.is_duplicate(lookalike)[0]   # human states count; excluded ids do not


def _load_server():
    import importlib.util
    pytest.importorskip("fastapi")
    spec = importlib.util.spec_from_file_location("playground_server", ROOT / "scripts/playground/server.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class _Tensor:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return list(self.values)


class _Ids:
    shape = (1, 17)


class _StubEngine:
    """Scores each option text by a fixed preference plus a large bias on the FIRST presented position."""
    prefer = {"alpha": 1.0, "beta": 1.5, "gamma": 0.0}

    def __init__(self): self.calls = []
    def labels(self, n, n_images): return [chr(65 + i) for i in range(n)]
    def prepare(self, images, prompt, labels):
        texts = [prompt.split(f"\n{label}: ")[1].split("\n")[0] if f"\n{label}: " in prompt else prompt.split(f"{label}: ")[1].split("\n")[0] for label in labels]
        self.calls.append(texts); return None, {"input_ids": _Ids()}, list(range(len(labels)))
    def candidate_logits(self, inputs, token_ids):
        texts = self.calls[-1]
        return _Tensor([next((v for k, v in self.prefer.items() if k in t), -1.0) + (3.0 if i == 0 else 0.0) for i, t in enumerate(texts)])


class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _torch_backend(server, rotations):
    backend = object.__new__(server.TorchBackend)
    backend.engine = _StubEngine(); backend.rotations = rotations; backend.torch = type("T", (), {"no_grad": staticmethod(lambda: _NoGrad())})
    return backend


def test_torch_backend_rotation_averaging_with_stub_engine():
    server = _load_server()
    from vision_decision.contracts import Request
    request = Request.model_validate({"request_id": "r1", "state": "A: not an option line", "fields": [{"id": "decision", "type": "choice", "question": "Pick one?",
                                      "options": [{"value": "alpha", "description": "alpha"}, {"value": "beta", "description": "beta"}, {"value": "gamma", "description": "gamma"}]}]})
    single = _torch_backend(server, 1); results, stats = single.score([], request)
    assert results[0].value == "alpha" and len(single.engine.calls) == 1 and stats["rotations"] == 1  # position bias wins in one pass
    full = _torch_backend(server, 4); results, stats = full.score([], request)
    assert len(full.engine.calls) == 4 and stats["rotations"] == 4 and stats["input_tokens"] == 17
    assert [c[0] for c in full.engine.calls] != [full.engine.calls[0][0]] * 4          # every rotation presents a different first option
    assert results[0].value == "beta" and results[0].status == "answered"                # the bias cancels; the preferred option wins
    from vision_decision.scoring import compile_question, cyclic_offsets, rotate, combine_rotations
    _, choices, texts = compile_question(request.fields[0], request.state)
    expected = combine_rotations(choices, [(o, [_StubEngine.prefer.get(next((k for k in _StubEngine.prefer if k in t), ""), -1.0) + (3.0 if i == 0 else 0.0)
                                                for i, t in enumerate(rotate(texts, o))]) for o in cyclic_offsets(len(choices), 4)])
    assert results[0].raw_logits == expected.raw_logits
    assert server.build_backend.__code__.co_varnames[:5] == ("kind", "adapter", "no_adapter", "bundle", "rotations")


def test_author_jevstyle_dev_is_valid_and_deterministic():
    import author_jevstyle_dev as aj
    rows = aj.author(per_family=3)
    assert len(rows) == 30 and {r["family"] for r in rows} == set(aj.JEVSTYLE_FAMILIES) and all(r["partition"] == "dev" for r in rows)
    assert [r["request"]["state"] for r in rows] == [r["request"]["state"] for r in aj.author(per_family=3)]
    assert any(r["target"] is None and r["abstention_cause"] == "insufficient_evidence" for r in rows if r["family"] == "ambiguous")
    from vision_decision.contracts import Request
    for r in rows: Request.model_validate(r["request"])


def test_fit_single_temperature_and_manifest_builder(tmp_path):
    import fit_p2_temperature as ft, build_p2b_manifest as bm, math
    rng = random.Random(0); rows = []
    for i in range(60):
        n = rng.choice([2, 3, 5]); t = rng.randrange(n + 1); logits = [rng.gauss(0, 1) for _ in range(n + 1)]; logits[t] += 4.0  # over-confident
        rows.append({"id": f"p2_teacher:d{i}:0", "decision_type": rng.choice(["choice", "boolean"]) if n == 2 else "choice", "option_count": n, "logits": logits, "target_index": t})
    payload = ft.fit_single_temperature(rows, "p2b-test")
    assert payload["schema_version"] == "1.0" and len(set(payload["temperatures"].values())) == 1 and set(payload["counts"].values()) == {60}
    assert payload["fit"]["temperature"] > 0 and payload["fit"]["nll_calibrated"] <= payload["fit"]["nll_raw"] and math.isfinite(payload["fit"]["temperature"])
    assert payload["fit"]["temperature"] != 1.0   # the synthetic logits are miscalibrated, so the fit moves off 1
    from vision_decision.calibration import TemperatureCalibrator
    TemperatureCalibrator.from_dict(payload)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps({"id": f"p2_teacher:d{i}:0", "domain": "telecom" if i % 2 else "travel"}) + "\n" for i in range(60)))
    assert len(ft.select_rows(rows, manifest, ("telecom",))) == 30 and len(ft.select_rows(rows, None, ())) == 60
    # manifest builder: new p2b rows by partition, held-out domains -> calibration, 30% deterministic replay of phase-2 train rows
    p2 = [{"id": f"p2:{i}", "partition": "train" if i % 10 else "dev", "request": {"state": "s", "fields": [{"id": "decision", "type": "boolean", "question": "q?"}]}, "target": True} for i in range(1000)]
    new = []
    for i in range(40):
        plan = gen_write.plan_document(i, tag="p2b-hard")
        new.append({"id": f"p2_teacher:{plan['doc_id']}:0", "batch": "p2b-hard", "domain": plan["domain"], "partition": "train" if i % 5 else "dev",
                    "request": {"state": "s", "fields": [{"id": "decision", "type": "boolean", "question": "q?"}]}, "target": False, "family": "tradeoff"})
    new.append(dict(new[0], id="p2_teacher:oldbatch:0", batch="p2"))
    rows, report = bm.build(p2, new, "p2b-*", 0.30, ("telecom", "hospitality", "nonprofit_grants"), validate=False)
    parts = {}
    for r in rows: parts.setdefault(r["partition"], []).append(r)
    assert report["replay_train"] == len(parts["train"]) - report["new_train"] and 230 <= report["replay_train"] <= 310
    assert all(r["domain"] in ("telecom", "hospitality", "nonprofit_grants") for r in parts.get("calibration", [])) and report.get("calibration", 0) == sum(n["domain"] in ("telecom", "hospitality", "nonprofit_grants") for n in new[:40])
    assert not any(r["id"] == "p2_teacher:oldbatch:0" for r in rows) and not any(r["id"].startswith("p2:") and r["partition"] != "train" for r in rows)
    assert [r["id"] for r in rows if r.get("replay")] == [r["id"] for r in bm.build(p2, new, "p2b-*", 0.30, (), validate=False)[0] if r.get("replay")]
    strict_rows, strict_report = bm.build(p2, new, "p2b-*", 0.30, (), validate=True)
    assert strict_rows == [] and strict_report["contract_dropped"] == len(rows)   # the fake requests lack request ids: contract validation drops them


# ---------------------------------------------------------------------------------------------------------------------
# phase-2c: judge writer families (opt-in) and deterministic exact-answer generators (gen_programmatic.py)

import re  # noqa: E402
import gen_programmatic as gp  # noqa: E402

JUDGES = ["judge_pairwise", "judge_rubric_score"]


def _fake_writer_reply(prompt: str) -> str:
    """A schema-valid writer reply built from the rendered prompt (families, types, unknown flags, state shape)."""
    specs = re.findall(r'Question \d: family "(\w+)".*?"type": "(\w+)".*?(must be UNKNOWN|must be one specific)', prompt, flags=re.S)
    body = " ".join(f"clause{i} applies to the request and both candidate responses" for i in range(40))
    doc = body if "single string" in prompt else {"request": body, "rubric": ["correctness first", "format second"], "responses": {"A": "x", "B": "y"}}
    qs = []
    for fam, typ, unk in specs:
        q = {"family": fam, "type": typ, "question": f"Which verdict applies for {fam}?", "unknown_reason": None, "justification": "Criterion 1 decides it."}
        if typ == "choice":
            q["options"] = [{"text": "Response A", "description": "A is better."}, {"text": "Response B", "description": "B is better."},
                            {"text": "Both equally", "description": "A tie."}]
            q["intended"] = "Response B"
        elif typ == "score":
            q["levels"] = [{"value": v, "description": f"level {v}"} for v in range(9)]; q["intended"] = 5
        else:
            q["intended"] = True
        if unk == "must be UNKNOWN":
            q["intended"] = None; q["unknown_reason"] = "insufficient_evidence"
        qs.append(q)
    return json.dumps({"document": doc, "questions": qs})


def test_judge_families_are_opt_in_and_render_valid_prompts():
    for fam in JUDGES:
        assert fam in families.FAMILIES and fam in families.ALL_FAMILY_IDS and fam not in families.FAMILY_IDS and len(families.BRIEFS[fam]) >= 6
    assert families.forced_type("judge_pairwise") == "choice" and families.forced_type("judge_rubric_score") == "score"
    assert len(families.FAMILY_IDS) == 17 and all(f in families.FAMILY_IDS for f in families.pick_families(random.Random(1)))
    plans = [gen_write.plan_document(i, families=JUDGES, tag="p2c-judge") for i in range(12)]
    for p in plans:
        assert p["batch"] == "p2c-judge" and p["doc_id"].startswith("p2c-judge-") and set(JUDGES) <= set(p["families"])
        assert p["types"][p["families"].index("judge_pairwise")] == "choice" and p["types"][p["families"].index("judge_rubric_score")] == "score"
        text = gen_write.render(p)[1]["content"]
        assert 'family "judge_pairwise"' in text and "Response A" in text and "Response B" in text and "between 2 and 9" in text
        assert text.count("the answerers are checked against it") == 2      # judge-specific justification requirement on both judge questions
        other = next(f for f in p["families"] if f not in JUDGES)
        assert FAMILIES_DEFAULT_JUST in text.split(f'family "{other}"')[1]    # other families keep the original wording
        out, err = gen_write.validate(_fake_writer_reply(text), p)
        assert err is None and [q["family"] for q in out["questions"]] == p["families"]
    # a 9-level rubric score validates and maps to an ordinal field
    q = WQuestion.model_validate({"family": "judge_rubric_score", "type": "score", "question": "Which rubric level does the reply earn?",
                                  "levels": [{"value": v, "description": f"level {v}"} for v in range(9)], "intended": 4, "unknown_reason": None, "justification": "cap"})
    assert to_field(q)[0]["type"] == "ordinal" and len(to_field(q)[0]["levels"]) == 9
    # the judge texts themselves pass the JevBench 8-gram lint
    grams = reference_ngrams(__import__("p2_common").jevbench_public_files())
    for fam in JUDGES:
        text = " ".join([families.FAMILIES[fam]["rule"], families.FAMILIES[fam]["justification"], *families.FAMILIES[fam]["shape"].values(), *families.BRIEFS[fam]])
        assert contamination_count(text, grams) == 0, fam


FAMILIES_DEFAULT_JUST = 'Also give "justification": one sentence citing the parts of the document that decide it.'


def test_gen_write_cli_runs_judge_families(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def complete(self, messages, **kw): return _fake_writer_reply(messages[1]["content"])
    monkeypatch.setattr(gen_write, "ChatClient", FakeClient)
    out = tmp_path / "writer-judge.jsonl"
    assert gen_write.main(["--docs", "6", "--out", str(out), "--families", "judge_pairwise,judge_rubric_score", "--tag", "p2c-judge", "--workers", "2"]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 6 and all(r["batch"] == "p2c-judge" for r in rows) and not (tmp_path / "writer-judge.rejects.jsonl").exists()
    assert all(set(JUDGES) <= {q["family"] for q in r["output"]["questions"]} for r in rows)


# --- independent recomputation of every programmatic kind (brute force / numpy / tz-aware datetimes, not the generator's helpers)
import calendar as _cal, datetime as _dt, itertools as _it  # noqa: E402
from decimal import Decimal as _D, ROUND_HALF_UP as _HU  # noqa: E402
from fractions import Fraction as _F  # noqa: E402
import numpy as _np  # noqa: E402

_d = _dt.date.fromisoformat


def _c2(x):
    return _D(x).quantize(_D("0.01"), rounding=_HU)


def _busday_after(start, n, closed=()):
    return _np.busday_offset(_np.datetime64(start), n, roll="backward", holidays=list(closed)).astype(object)


def _band(value, edges):
    return sum(value >= e for e in edges)


def _minutes(entry):
    m = re.fullmatch(r"(\d+) h (\d+) min|(\d+) minutes|([\d.]+) h|(\d+):(\d+) \(h:mm\)", entry)
    if m.group(1): return int(m.group(1)) * 60 + int(m.group(2))
    if m.group(3): return int(m.group(3))
    if m.group(4): return int(_F(m.group(4)) * 60)
    return int(m.group(5)) * 60 + int(m.group(6))


def _kg(w):
    v, u = w.split(); return _D(v) * (_D("0.4536") if u == "lb" else 1)


def _recompute(kind, p):
    """The canonical answer (string for choice, bool for noul, int for score) from the stored inputs only."""
    if kind == "business_deadline":
        due = _busday_after(_d(p["start"]), p["n"], [_d(x) for x in p["closed"]])
        return (_d(p["action"]) <= due) if "action" in p else due.isoformat()
    if kind == "calendar_rollover":
        return _dt.date.fromordinal(_d(p["start"]).toordinal() + p["n"]).isoformat()
    if kind == "tz_order":
        utc = [_dt.datetime.fromisoformat(e["local"]).replace(tzinfo=_dt.timezone(_dt.timedelta(minutes=e["offset"]))) for e in p["events"]]
        return str(int(abs((utc[1] - utc[0]).total_seconds()) // 60)) if "option_values" in p else utc[0] < utc[1]
    if kind == "tz_cutoff":
        cut = _dt.datetime.fromisoformat(p["cutoff_local"]).replace(tzinfo=_dt.timezone(_dt.timedelta(minutes=p["offset"])))
        rec = _dt.datetime.fromisoformat(p["received_utc"]).replace(tzinfo=_dt.timezone.utc)
        return cut.astimezone(_dt.timezone.utc).replace(tzinfo=None).isoformat() if "option_values" in p else rec <= cut
    if kind == "duration_sum":
        total = sum(_minutes(e) for e in p["entries"])
        return total > p["cap_minutes"] if "cap_minutes" in p else str(total)
    if kind == "overnight_shift":
        paid = int((_dt.datetime.fromisoformat(p["end"]) - _dt.datetime.fromisoformat(p["start"])).total_seconds() // 60) - p["break"]
        return str(paid) if "option_values" in p else paid > 60 * p["limit_h"]
    if kind == "notice_by_post":
        last = _d(p["renewal"]) - _dt.timedelta(days=p["n"])
        ok = lambda s: _busday_after(s, p["k"]) <= last  # noqa: E731
        if "posted" in p: return ok(_d(p["posted"]))
        return max(last - _dt.timedelta(days=i) for i in range(0, 30) if ok(last - _dt.timedelta(days=i))).isoformat()
    if kind == "sla_business_hours":
        t, left = _dt.datetime.fromisoformat(p["opened"]), p["hours"] * 60
        while left:
            if t.weekday() < 5 and p["open"] <= t.hour < p["close"]: left -= 1
            t += _dt.timedelta(minutes=1)
        return (_dt.datetime.fromisoformat(p["replied"]) <= t) if "replied" in p else t.isoformat()
    if kind == "tenure_months":
        s = _d(p["start"]); m = s.month - 1 + p["months"]; elig = _dt.date(s.year + m // 12, m % 12 + 1, s.day)
        return (_d(p["on"]) >= elig) if "on" in p else elig.isoformat()
    if kind == "weekday_offset":
        return (_d(p["anchor"]) + _dt.timedelta(days=p["n"])).strftime("%A")
    if kind == "recurring_schedule":
        x = _d(p["first"]) + _dt.timedelta(days=(p["k"] - 1) * p["every"])
        while x.weekday() >= 5 or x.isoformat() in p["closed"]: x += _dt.timedelta(days=1)
        return x.isoformat()
    if kind == "delay_band":
        pr, de = _d(p["promised"]), _d(p["delivered"])
        bd = int(_np.busday_count(pr + _dt.timedelta(days=1), de + _dt.timedelta(days=1), holidays=p["closed"])) if de > pr else 0
        return _band(bd, [1, 3, 6, 11])
    if kind == "grace_period":
        last = _d(p["invoice"]) + _dt.timedelta(days=p["net"] + p["grace"])
        return (_d(p["paid"]) > last) if "paid" in p else last.isoformat()
    if kind == "single_draw":
        items = [lab for lab, k in p["pool"].items() for _ in range(k)]
        f = _F(sum(i in p["target"] for i in items), len(items)); return f"{f.numerator}/{f.denominator}"
    if kind in ("pair_without_replacement", "at_least_one", "clean_sample_threshold", "probability_band"):
        n = p["n"]; bad = p.get("a", p.get("d")); k = 2 if kind == "pair_without_replacement" else p["k"]
        combos = list(_it.combinations(range(n), k)); hits = lambda cmb: sum(i < bad for i in cmb)  # noqa: E731
        if kind == "pair_without_replacement": f = _F(sum(hits(x) == k for x in combos), len(combos))
        elif kind == "at_least_one": f = _F(sum(hits(x) > 0 for x in combos), len(combos))
        elif kind == "clean_sample_threshold": return _F(sum(hits(x) == 0 for x in combos), len(combos)) * 100 > p["threshold"]
        else: return min(int(_F(sum(hits(x) > 0 for x in combos), len(combos)) * 100 // p["width"]), 100 // p["width"] - 1)
        return f"{f.numerator}/{f.denominator}"
    if kind == "conditional_table":
        row = p["cells"][p["group"]]; f = _F(row[p["outcome"]], sum(row)); return f"{f.numerator}/{f.denominator}"
    if kind == "bayes_counts":
        tp, fp = _F(p["sens"] * p["m"], 100), _F(p["fpr"] * (p["N"] - p["m"]), 100); f = tp / (tp + fp); return f"{f.numerator}/{f.denominator}"
    if kind == "independent_both":
        f = _F(p["p"] * p["q"], 10000)
        return f * 100 >= p["threshold"] if "threshold" in p else f"{f.numerator}/{f.denominator}"
    if kind == "independent_any":
        outcomes = [(a, b) for a in range(100) for b in range(100)]  # brute force over a 100 x 100 grid of equally likely cells
        f = _F(sum(a < p["p"] or b < p["q"] for a, b in outcomes), len(outcomes)); return f"{f.numerator}/{f.denominator}"
    if kind == "committee_count":
        return str(sum(1 for cmb in _it.combinations(range(p["n"]), p["k"]) if 0 in cmb and 1 not in cmb))
    if kind == "ordered_slots":
        return str(sum(1 for _ in _it.permutations(range(p["n"]), p["k"])))
    if kind == "expected_cost":
        return str(_c2(sum(_D(s) * _D(c) for s, c in zip(p["shares"], p["costs"])) / 100))
    if kind == "assignment_none":
        allx = list(_it.product(range(p["n"]), repeat=p["k"])); f = _F(sum(0 not in a for a in allx), len(allx)); return f"{f.numerator}/{f.denominator}"
    if kind == "binomial_exactly_one":
        pr = _F(p["p"]); f = sum(_np.prod([pr if o else 1 - pr for o in out]) for out in _it.product([0, 1], repeat=p["n"]) if sum(out) == 1)
        f = _F(f); return f"{f.numerator}/{f.denominator}"
    if kind == "invoice_total":
        sub = sum(_c2(q * _D(u)) for q, u in p["lines"]); after = sub - _c2(sub * p["discount_pct"] / 100)
        total = after + _c2(after * _D(p["tax_pct"]) / 100) + _D(p["shipping"])
        return total > _D(p["limit"]) if "limit" in p else str(total)
    if kind == "po_mismatch":
        amt = lambda rows: [_c2(q * _D(u)) for q, u in rows]  # noqa: E731
        po, inv = amt(p["po"]), amt(p["invoice"]); diff = [i for i in range(len(po)) if po[i] != inv[i]]
        assert len(diff) == 1
        return p["items"][diff[0]] if p["answer"] in p["items"] else str(sum(inv) - sum(po))
    if kind == "bank_reconciliation":
        gap = (_D(p["bank"]) + sum(map(_D, p["deposits"])) - sum(map(_D, p["cheques"]))) - (_D(p["ledger"]) - _D(p["fee"]))
        return str(gap) if "option_values" in p else gap == 0
    if kind == "stacked_discounts":
        P, a = _D(p["price"]), p["a"]
        if "coupon" in p:
            cp = _D(p["coupon"])
            return str(_c2(_c2(P * (100 - a) / 100) - cp) if p["order"] == "percent_first" else _c2((P - cp) * (100 - a) / 100))
        return str(_c2(_c2(P * (100 - a) / 100) * (100 - p["b"]) / 100))
    if kind == "weight_conversion":
        t = sum(_kg(w) for w in p["parcels"])
        return t <= p["limit"] if "limit" in p else str(t.quantize(_D("0.1"), rounding=_HU))
    if kind == "proration":
        s = _d(p["start"]); days = _cal.monthrange(s.year, s.month)[1]
        return str(_c2(_D(p["monthly"]) * (days - s.day + 1) / days))
    if kind == "net_from_gross":
        g = _D(p["gross"]); net = _c2(g * 100 / (100 + p["rate"])); return str(net if p["ask"] == "net" else g - net)
    if kind == "budget_remaining":
        rem = _D(p["budget"]) + (_D(p["transfer"]) if p["transfer_in"] else -_D(p["transfer"])) - sum(_D(a) for a, st in p["spend"] if st == "approved")
        return str(rem) if "option_values" in p else rem < 0
    if kind == "fx_fee":
        conv = _c2(_D(p["amount"]) * _D(p["rate"])); return str(conv + _c2(conv * _D(p["pct"]) / 100) + _D(p["fixed"]))
    if kind == "stock_variance":
        v = p["counted"] - (p["opening"] + sum(p["receipts"]) - sum(p["shipments"]) - p["writeoffs"] + p["returns"])
        return str(v) if "option_values" in p else abs(v) <= p["tolerance"]
    if kind == "change_band":
        ch = _F(p["current"] - p["previous"], p["previous"]) * 100
        return 0 if ch < -10 else 1 if ch < 0 else 2 if ch < 10 else 3 if ch < 25 else 4
    if kind == "usage_split":
        return str(_c2(_D(p["total"]) * p["usage"][p["i"]] / sum(p["usage"])))
    if kind == "free_shipping":
        after = sum(_c2(q * _D(u)) for q, u in p["lines"]) - _D(p["coupon"]); free = after >= _D(p["threshold"])
        return str(after + (0 if free else _D(p["fee"]))) if "option_values" in p else free
    raise AssertionError(f"no recompute for {kind}")


def test_programmatic_documents_are_schema_valid_unique_and_deterministic():
    rows, stats = gp.generate(240, "test-seed")
    assert stats["docs"] == 240 and stats["by_family"] == {f: 80 for f in gp.PROG_FAMILIES}
    ids = [r["doc_id"] for r in rows]; assert len(set(ids)) == len(ids)
    for r in rows:
        WriterOutput.model_validate(r["output"])
        assert r["batch"] == gp.BATCH[r["plan"]["families"][0]] and r["batch"].startswith("prog-") and r["doc_id"].startswith(r["batch"] + "-")
        assert r["plan"]["domain"] in domains.DOMAIN_IDS and len(set(r["plan"]["kinds"])) == 3 and r["writer"]["model"] == "programmatic"
        assert isinstance(r["output"]["document"], str) == (r["plan"]["state_shape"] == "string")
    assert {r["plan"]["domain"] for r in rows} == set(domains.DOMAIN_IDS)
    again, _ = gp.generate(240, "test-seed")
    assert [json.dumps(r, sort_keys=True) for r in again] == [json.dumps(r, sort_keys=True) for r in rows]      # deterministic per seed
    other, _ = gp.generate(30, "other-seed")
    assert [r["output"]["document"] for r in other] != [r["output"]["document"] for r in rows[:30]]
    assert gp.generate(10, "test-seed", start=5)[0] == rows[5:15]                                                # indices are stable across --start
    for fam in gp.PROG_FAMILIES:
        assert len(gp.templates(fam)) >= 50 and len(set(gp.templates(fam))) == len(gp.templates(fam))
    # gen_answer consumes the rows unchanged
    qs = list(gen_answer.questions(rows[:10])); assert len(qs) == 30 and {q["field"]["type"] for q in qs} <= {"boolean", "choice", "ordinal"}


def test_programmatic_intended_answers_recompute_independently():
    rows, _ = gp.generate(1500, "recompute")
    seen_kinds, checked, unknowns = set(), 0, 0
    for r in rows:
        for q, kind, p, template in zip(r["output"]["questions"], r["plan"]["kinds"], r["plan"]["params"], r["plan"]["templates"]):
            seen_kinds.add(kind); assert template.startswith(f"{q['family']}.{kind}.")
            if q["intended"] is None:
                unknowns += 1; assert q["unknown_reason"] == "insufficient_evidence" and p.get("withheld"); continue
            expected = _recompute(kind, p)
            if q["type"] == "choice":
                vals = p["option_values"]; assert len(set(vals.values())) == len(vals) == len(q["options"])     # exactly one option is right
                assert vals[q["intended"]] == str(p["answer"]) == expected, (kind, p, q["intended"])
            else:
                assert q["intended"] == expected == p["answer"], (kind, p, q["intended"])
                if q["type"] == "score":
                    assert 2 <= len(q["levels"]) <= 10
            checked += 1
    assert seen_kinds == {k for f in gp.PROG_FAMILIES for k in gp.KINDS[f]} and checked > 4000 and 0 < unknowns < 0.1 * 4500


def test_programmatic_output_has_no_jevbench_overlap_and_assembles(tmp_path):
    from p2_common import jevbench_public_files, read_jsonl
    grams = reference_ngrams(jevbench_public_files())
    if not grams:
        pytest.skip("JevBench public files not cached")
    rows, _ = gp.generate(600, "lint-check")                           # generated WITHOUT the reference: nothing was regenerated to hide a hit
    assert all(contamination_count(gp.row_text(r), grams) == 0 for r in rows)
    out = tmp_path / "writer-prog.jsonl"
    assert gp.main(["--docs", "30", "--seed", "cli", "--out", str(out)]) == 0
    first = out.read_text(); assert gp.main(["--docs", "30", "--seed", "cli", "--out", str(out)]) == 0 and out.read_text() == first
    wrows = read_jsonl(out)
    answers = []
    for w in wrows:
        for qi, q in enumerate(w["output"]["questions"]):
            _, target = to_field(WQuestion.model_validate(q))
            answers += [{"doc_id": w["doc_id"], "qi": qi, "answerer": n, "value": target, "confidence": 0.9, "parse_error": None, "raw": ""} for n in ("qwen", "gptoss")]
    from v1_text.common import write_jsonl
    write_jsonl(tmp_path / "answers.jsonl", answers)
    argv = ["--writer", str(out), "--answers", str(tmp_path / "answers.jsonl"), "--out", str(tmp_path / "records.jsonl"), "--license-dir", str(tmp_path / "lic"),
            "--dedupe-against", "--dedupe-threshold", "100000", "--batch-filter", "prog-*"]
    assert assemble_p2.main(argv) == 0
    recs = read_jsonl(tmp_path / "records.jsonl"); report = json.loads((tmp_path / "assembly-report.json").read_text())
    assert len(recs) == 90 and report["counts"].get("contaminated", 0) == 0 and {r["batch"] for r in recs} <= set(gp.BATCH.values())
    from vision_decision.contracts import Request
    for r in recs: Request.model_validate(r["request"])
    with pytest.raises(SystemExit):
        gp.main(["--docs", "1", "--out", str(out), "--families", "judge_pairwise"])


# ---------------------------------------------------------------------------------------------------------------------
# phase-2c soft targets: gen_answer --mode distribution, assemble_p2 --soft-from, make_image_replay, build_p2c_manifest

CHOICE_FIELD = {"id": "decision", "type": "choice", "question": "Which outcome applies?",
                "options": [{"value": "full_refund", "description": "all"}, {"value": "partial_refund", "description": "part"},
                            {"value": "no_refund", "description": "none"}]}
BOOL_FIELD = {"id": "decision", "type": "boolean", "question": "Was it on time?"}
ORD_FIELD = {"id": "decision", "type": "ordinal", "question": "How severe?", "levels": [{"value": 1, "description": "low"}, {"value": 2, "description": "mid"},
                                                                                       {"value": 3, "description": "high"}]}


def test_parse_distribution_renormalises_and_maps_labels():
    pd = gen_answer.parse_distribution
    v, p, rat, err = pd(json.dumps({"probabilities": {"full_refund": 10, "partial_refund": 70, "no_refund": 10, "unknown": 10},
                                    "rationale": "Section 2 applies. The claim was late. Extra sentence dropped."}), CHOICE_FIELD)
    assert err is None and v == "partial_refund" and abs(sum(p.values()) - 1) < 1e-6 and list(p) == ["full_refund", "partial_refund", "no_refund", "unknown"]
    assert p["partial_refund"] == 0.7 and rat == "Section 2 applies. The claim was late."
    # missing labels count 0, unknown always present, aliases and option text normalise, unrecognised keys are ignored
    v, p, _, err = pd('<think>{"scratch": 1}</think>{"probabilities": {"Partial refund": 0.5, "bogus": 3}, "rationale": "x"}', CHOICE_FIELD)
    assert err is None and v == "partial_refund" and p == {"full_refund": 0.0, "partial_refund": 1.0, "no_refund": 0.0, "unknown": 0.0}
    v, p, _, err = pd('{"probabilities": {"yes": 0.2, "no": 0.2, "__unknown__": 0.6}, "rationale": "not stated"}', BOOL_FIELD)
    assert err is None and v is None and p == {"true": 0.2, "false": 0.2, "unknown": 0.6}
    v, p, _, err = pd('{"probabilities": {"true": 0.9, "false": 0.1, "unknown": 0}, "rationale": "r"}', BOOL_FIELD)
    assert v is True and p["unknown"] == 0.0
    v, p, _, err = pd('{"probabilities": {"1": 0.1, "2": 0.45, "3": 0.45}, "rationale": "r"}', ORD_FIELD)
    assert err is None and v == 2 and p["unknown"] == 0.0            # tie -> first maximum in label order, unknown last
    v, p, _, err = pd('{"probabilities": {"1": 0.5, "unknown": 0.5}, "rationale": "r"}', ORD_FIELD)
    assert v == 1                                                     # ties never resolve to unknown
    assert pd('{"probabilities": {"1": -0.1, "2": 1}}', ORD_FIELD)[3].startswith("bad probability")
    assert pd('{"probabilities": {"1": 0, "2": 0}}', ORD_FIELD)[3] == "probabilities sum to zero"
    assert pd('{"answer": 2}', ORD_FIELD)[3] == "no probabilities key" and pd("not json", ORD_FIELD)[3].startswith("json")
    schema = gen_answer.distribution_schema(CHOICE_FIELD)["json_schema"]["schema"]["properties"]["probabilities"]
    assert schema["required"] == ["full_refund", "partial_refund", "no_refund", "unknown"] and schema["additionalProperties"] is False
    assert "- unknown:" in gen_answer.distribution_prompt("doc", BOOL_FIELD) and "- true: yes" in gen_answer.distribution_prompt("doc", BOOL_FIELD)


def test_gen_answer_distribution_mode_with_fake_client(tmp_path, monkeypatch):
    from p2_common import read_jsonl
    from v1_text.common import write_jsonl
    rows = [_writer_row(i, doc=" ".join(f"w{i}_{j}" for j in range(200))) for i in range(3)]
    write_jsonl(tmp_path / "writer.jsonl", rows)
    calls = []

    class FakeClient:
        def __init__(self, base_url, model): self.model = model
        def complete(self, messages, temperature=0.9, max_tokens=6000, json_mode=True, extra=None, response_format=None):
            calls.append({"max_tokens": max_tokens, "extra": extra, "response_format": response_format, "system": messages[0]["content"]})
            if response_format is None:
                return '{"answer": "unknown", "confidence": 0.5}'
            labels = response_format["json_schema"]["schema"]["properties"]["probabilities"]["required"]
            if "full_refund" in labels:
                return json.dumps({"probabilities": {"full_refund": 1, "partial_refund": 6, "no_refund": 1, "unknown": 2}, "rationale": "Section 2."})
            if labels[0] == "true":
                return "garbage"
            return json.dumps({"probabilities": {"1": 0.1, "2": 0.1, "3": 0.1, "unknown": 0.7}, "rationale": "Blank field."})

    monkeypatch.setattr(gen_answer, "ChatClient", FakeClient)
    out = tmp_path / "dist.jsonl"
    assert gen_answer.main(["--mode", "distribution", "--answerer", "qwen35", "--writer", str(tmp_path / "writer.jsonl"), "--out", str(out), "--workers", "2"]) == 0
    got = sorted(read_jsonl(out), key=lambda r: (r["doc_id"], r["qi"]))
    assert len(got) == 9 and all(set(r) == {"doc_id", "qi", "answerer", "model", "repo", "revision", "mode", "value", "probs", "rationale", "raw", "parse_error"} for r in got)
    assert all(r["mode"] == "distribution" and r["model"] == "qwen3.6-35b-a3b" and r["repo"] == ANSWERERS["qwen35"]["repo"] for r in got)
    c = [r for r in got if r["qi"] == 0]; b = [r for r in got if r["qi"] == 1]; o = [r for r in got if r["qi"] == 2]
    assert all(r["value"] == "partial_refund" and r["probs"]["partial_refund"] == 0.6 and r["probs"]["unknown"] == 0.2 and r["rationale"] == "Section 2." for r in c)
    assert all(r["value"] is None and r["probs"] is None and r["parse_error"] for r in b)
    assert all(r["value"] is None and r["probs"]["unknown"] == 0.7 and r["parse_error"] is None for r in o)
    assert all(k["max_tokens"] == 8000 and k["extra"]["chat_template_kwargs"] == {"enable_thinking": True} and k["system"] == gen_answer.DISTRIBUTION_SYSTEM for k in calls)
    n = len(calls); assert gen_answer.main(["--mode", "distribution", "--answerer", "qwen35", "--writer", str(tmp_path / "writer.jsonl"), "--out", str(out)]) == 0
    assert len(calls) == n and len(read_jsonl(out)) == 9                 # resumable by (doc_id, qi)
    # default mode is unchanged: same row keys, plain JSON mode, no thinking kwargs added
    out2 = tmp_path / "answers.jsonl"; calls.clear()
    assert gen_answer.main(["--answerer", "qwen35", "--writer", str(tmp_path / "writer.jsonl"), "--out", str(out2), "--shard", "0/3"]) == 0
    assert len(read_jsonl(out2)) == 3 and all(set(r) == {"doc_id", "qi", "answerer", "model", "repo", "revision", "raw", "value", "confidence", "parse_error"} for r in read_jsonl(out2))
    assert all(k["response_format"] is None and k["extra"] == {} and k["system"] == families.ANSWERER_SYSTEM for k in calls)


def test_chat_client_json_schema_response_format():
    sent = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    client = ChatClient("http://x/v1", "m", opener=lambda req, timeout: sent.append(json.loads(req.data)) or Resp(), sleep=lambda s: None)
    rf = gen_answer.distribution_schema(BOOL_FIELD); client.complete([{"role": "user", "content": "q"}], response_format=rf)
    client.complete([{"role": "user", "content": "q"}])
    assert sent[0]["response_format"] == rf and sent[1]["response_format"] == {"type": "json_object"}


def _dist(w, qi, probs, rationale="Because the table says so.", answerer="qwen35", **kw):
    return {"doc_id": w["doc_id"], "qi": qi, "answerer": answerer, "model": "qwen3.6-35b-a3b", "repo": ANSWERERS[answerer]["repo"],
            "revision": ANSWERERS[answerer]["revision"], "mode": "distribution", "value": None, "probs": probs, "rationale": rationale, "raw": "", "parse_error": None, **kw}


def test_assemble_soft_from_keeps_argmax_agreement_and_audits_unknown_mass(tmp_path):
    from v1_text.common import verified_license, write_jsonl
    from p2_common import read_jsonl
    lic = verified_license(assemble_p2.write_license_evidence(tmp_path / "lic", WRITER_MODEL, ANSWERERS), "Apache-2.0")
    ws = [_writer_row(i, doc=" ".join(f"d{i}_{j}" for j in range(200))) for i in range(4)]
    hard = _answers(ws)                                # qwen + gptoss agree with the intended answers (partial_refund, False, None)
    P = {"full_refund": 0.05, "partial_refund": 0.85, "no_refund": 0.05, "unknown": 0.05}
    soft = [
        _dist(ws[0], 0, P), _dist(ws[0], 1, {"true": 0.1, "false": 0.6, "unknown": 0.3}), _dist(ws[0], 2, {"1": 0.1, "unknown": 0.9}),
        _dist(ws[1], 0, {"partial_refund": 0.3, "unknown": 0.7}),          # answerable, unknown mass > 0.5 -> argmax unknown -> dropped
        _dist(ws[1], 1, {"true": 0.9, "false": 0.1}),                       # answerable, wrong argmax -> dropped
        _dist(ws[1], 2, {"2": 0.8, "unknown": 0.2}),                        # unknown gold, argmax a level -> kept with the hard target only
        _dist(ws[2], 0, None, parse_error="json: JSONDecodeError"),         # unusable -> kept hard
        # ws[2] qi 1/2 and ws[3]: no distribution row -> kept hard
    ]
    kept, rep = assemble_p2.build_records(ws, hard + soft, 2, lic, reference=set(), soft_from="qwen35")
    by = {r["id"]: r for r in kept}; rid = lambda w, qi: f"p2_teacher:{w['doc_id']}:{qi}"
    assert rep["counts"]["soft_disagreed"] == 2 and rid(ws[1], 0) not in by and rid(ws[1], 1) not in by
    assert len(kept) == 10 and rep["counts"]["soft_agree"] == 3 and rep["counts"]["soft_disagree_kept_hard"] == 1
    r = by[rid(ws[0], 0)]
    assert r["target"] == "partial_refund" and r["target_probs"] == P and r["rationale"] == "Because the table says so."
    assert r["provenance"]["soft_teacher"]["unknown_mass"] == 0.05 and r["provenance"]["soft_teacher"]["unknown_mass_high"] is False
    r = by[rid(ws[0], 1)]; assert r["target_probs"]["unknown"] == 0.3 and r["provenance"]["soft_teacher"]["unknown_mass_high"] is True  # flagged, kept
    r = by[rid(ws[0], 2)]; assert r["target"] is None and r["target_probs"] == {"1": 0.1, "2": 0.0, "3": 0.0, "unknown": 0.9}
    r = by[rid(ws[1], 2)]; assert r["target"] is None and "target_probs" not in r and r["provenance"]["soft_teacher"]["status"] == "disagree_kept_hard"
    assert all("target_probs" not in by[rid(w, qi)] for w, qi in ((ws[2], 0), (ws[2], 1), (ws[3], 0)))
    for r in kept:
        if "target_probs" in r:
            assert "unknown" in r["target_probs"] and abs(sum(r["target_probs"].values()) - 1) < 1e-6
    um = rep["unknown_mass"]
    assert um["answerable_rows"] == 4 and abs(um["mean"] - (0.05 + 0.3 + 0.7 + 0.0) / 4) < 1e-9
    assert um["over_0_2"] == 2 and um["over_0_5"] == 1 and um["over_0_5_share"] == 0.25 and um["kept_over_0_5"] == 0 and um["kept_over_0_2"] == 1
    assert rep["with_target_probs"] == 3 and rep["with_rationale"] == 3
    # the intended-unknown disagreement is dropped only with --soft-unknown-policy drop
    kept2, rep2 = assemble_p2.build_records(ws, hard + soft, 2, lic, reference=set(), soft_from="qwen35", soft_unknown_policy="drop")
    assert len(kept2) == 9 and rep2["counts"]["soft_disagreed"] == 3
    # without --soft-from nothing changes (distribution rows would vote, so they are simply not passed)
    base, base_rep = assemble_p2.build_records(ws, hard, 2, lic, reference=set())
    assert len(base) == 12 and not any("target_probs" in r for r in base) and "unknown_mass" not in base_rep
    # soft teacher as the only answerer: a question without a usable distribution is not kept
    only, only_rep = assemble_p2.build_records(ws, soft, 0, lic, reference=set(), soft_from="qwen35")
    assert {r["id"] for r in only} == {rid(ws[0], 0), rid(ws[0], 1), rid(ws[0], 2), rid(ws[1], 2)}
    # CLI: distribution rows passed in --answers are split off and do not count as an agreement answerer
    write_jsonl(tmp_path / "writer.jsonl", ws); write_jsonl(tmp_path / "hard.jsonl", hard); write_jsonl(tmp_path / "soft.jsonl", soft)
    argv = ["--writer", str(tmp_path / "writer.jsonl"), "--answers", str(tmp_path / "hard.jsonl"), str(tmp_path / "soft.jsonl"), "--out", str(tmp_path / "rec.jsonl"),
            "--license-dir", str(tmp_path / "lic"), "--reference", "--dedupe-against", "--soft-from", "qwen35"]
    assert assemble_p2.main(argv) == 0
    recs = read_jsonl(tmp_path / "rec.jsonl"); report = json.loads((tmp_path / "assembly-report.json").read_text())
    assert len(recs) == 10 and report["answerers"] == 2 and report["unknown_mass"]["over_0_5"] == 1
    from decision_data import render
    for r in recs:
        if "target_probs" in r:
            target = render(r, soft_targets=True)[3]
            assert abs(sum(target.probs) - 1) < 1e-6 and len(target.probs) == len(render(r)[1])


def _rec(rid, field, target, partition="train", domain="travel", family="policy_exception", state="A short fictional document.", **kw):
    return {"id": rid, "source": "p2_teacher", "source_group": rid.split(":")[1], "partition": partition, "domain": domain, "family": family,
            "images": [], "request": {"schema_version": "1.0", "request_id": rid.replace(":", "-"), "state": state, "fields": [field]},
            "target": target, "abstention_cause": None if target is not None else "insufficient_evidence", **kw}


def test_make_image_replay_is_stratified_deterministic_and_rewrites_paths(tmp_path):
    import make_image_replay as mir
    from PIL import Image
    img_dir = tmp_path / "src"; img_dir.mkdir(); rows = []
    for i in range(60):
        src = "big" if i < 40 else "small"
        p = img_dir / f"im{i}.jpg"; Image.new("RGB", (8, 8), (i, 0, 0)).save(p)
        rows.append({"id": f"{src}:{i}", "source": src, "partition": "train" if i % 10 else "dev", "images": [{"image": str(p), "sha256": f"{i:064x}", "width": 8, "height": 8}],
                     "request": {"schema_version": "1.0", "request_id": f"r{i}", "state": {}, "fields": [BOOL_FIELD]}, "target": bool(i % 2)})
    rows.append({"id": "text:0", "source": "big", "partition": "train", "images": [], "request": rows[1]["request"], "target": True})   # no image
    rows[3]["images"][0]["image"] = str(img_dir / "missing.jpg")                                                                 # skipped
    rows.append({"id": "multi:0", "source": "small", "partition": "train", "images": [], "targets": {"a": True, "b": None},           # multi-field, untouched
                 "request": {"schema_version": "1.0", "request_id": "m", "state": {}, "fields": [dict(BOOL_FIELD, id="a"), dict(BOOL_FIELD, id="b")]}})
    rows[-1]["images"] = [{"image": str(img_dir / "im41.jpg"), "sha256": f"{41:064x}"}]
    man = tmp_path / "m.jsonl"; man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    argv = ["--manifest", str(man), "--n-fresh", "20", "--n-delta", "8", "--seed", "7", "--out-dir", str(out), "--reference"]
    assert mir.main(argv) == 0
    fresh = [json.loads(l) for l in (out / "replay-fresh.jsonl").read_text().splitlines()]
    delta = [json.loads(l) for l in (out / "replay-delta.jsonl").read_text().splitlines()]
    rep = json.loads((out / "replay-report.json").read_text())
    assert len(fresh) == 20 and len(delta) == 8 and {r["id"] for r in delta} <= {r["id"] for r in fresh}
    assert rep["skipped_candidates"].get("missing_image", 0) <= 1 and rep["eligible_train_rows_with_images"] == 55
    src_counts = collections.Counter(r["source"] for r in fresh); assert src_counts == {"big": 13, "small": 7}   # 36 : 19 eligible -> 13.1 : 6.9
    assert all(r["replay"] == "image-v2.1" and r["partition"] == "train" for r in fresh)
    orig = {r["id"]: r for r in rows}
    for r in fresh:
        o = orig[r["id"]]
        assert {k: v for k, v in r.items() if k not in ("images", "replay")} == {k: v for k, v in o.items() if k != "images"}   # labels untouched
        for im, oim in zip(r["images"], o["images"]):
            assert Path(im["image"]).parent == out / "images" and Path(im["image"]).name == f"{oim['sha256']}.jpg" and Path(im["image"]).is_file()
    first = (out / "replay-fresh.jsonl").read_text()
    assert mir.main(argv) == 0 and (out / "replay-fresh.jsonl").read_text() == first            # deterministic
    small = tmp_path / "out2"; assert mir.main(argv[:-3] + ["--out-dir", str(small), "--max-bytes", "3000", "--reference"]) == 0
    rep2 = json.loads((small / "replay-report.json").read_text())
    assert rep2["n_fresh"] < 20 and rep2["fresh"]["image_bytes"] <= 3000 and rep2["n_delta"] <= rep2["n_fresh"]


import collections  # noqa: E402


def _p2c_inputs():
    judge_field = {"id": "decision", "type": "choice", "question": "Which response better satisfies the request under the rubric?",
                   "options": [{"value": "response_a", "description": "A"}, {"value": "response_b", "description": "B"}, {"value": "both_or_neither", "description": "tie"}]}
    p2b = [
        _rec("p2_teacher:d0:0", CHOICE_FIELD, "partial_refund"),                             # relabel agrees -> soft
        _rec("p2_teacher:d0:1", BOOL_FIELD, False),                                           # relabel disagrees, answerable train -> dropped
        _rec("p2_teacher:d0:2", ORD_FIELD, None),                                             # relabel disagrees, unknown gold -> kept hard
        _rec("p2_teacher:d1:0", BOOL_FIELD, True, partition="dev"),                           # relabel disagrees, dev -> kept hard
        _rec("p2_teacher:d2:0", BOOL_FIELD, True, domain="telecom"),                          # held out -> calibration
        _rec("p2_teacher:d3:0", BOOL_FIELD, True, partition="test"),                          # no relabel row -> kept hard
    ]
    dist = [{"doc_id": "d0", "qi": 0, "answerer": "qwen35", "mode": "distribution", "probs": {"partial_refund": 0.8, "no_refund": 0.1, "unknown": 0.1}, "rationale": "Clause 2.", "parse_error": None},
            {"doc_id": "d0", "qi": 1, "answerer": "qwen35", "mode": "distribution", "probs": {"true": 0.7, "false": 0.3}, "rationale": "r", "parse_error": None},
            {"doc_id": "d0", "qi": 2, "answerer": "qwen35", "mode": "distribution", "probs": {"2": 0.7, "unknown": 0.3}, "rationale": "r", "parse_error": None},
            {"doc_id": "d1", "qi": 0, "answerer": "qwen35", "mode": "distribution", "probs": {"false": 1.0}, "rationale": "r", "parse_error": None},
            {"doc_id": "d0", "qi": 0, "answerer": "qwen35", "mode": "answer", "value": "no_refund"}]           # answer-mode rows are ignored
    p2c = [_rec(f"p2_teacher:j{i}:0", judge_field, "response_b", partition="dev" if i < 3 else "train", family="judge_pairwise", batch="p2c-judge",
                target_probs={"response_a": 0.1, "response_b": 0.9}) for i in range(5)]
    p2c += [_rec("p2_teacher:j0:2", BOOL_FIELD, True, partition="dev", family="routing", batch="p2c-judge")]              # non-judge dev -> extra_dev
    p2c += [_rec(f"p2_teacher:prog-temporal-{i}:0", BOOL_FIELD, bool(i % 2), family="temporal_arithmetic", batch="prog-temporal") for i in range(12)]
    eikos = [dict(_rec(f"eikos_decisions:core:e{i}", BOOL_FIELD, bool(i % 2), family="eikos.rules_book", domain="rules: lending", partition="train" if i else "dev"),
                  source="eikos_decisions", target_probs=[0.95, 0.05] if i % 2 else [0.05, 0.95], target_probs_keys=[True, False]) for i in range(8)]
    eikos += [dict(_rec("eikos_decisions:core:o0", ORD_FIELD, 2, family="eikos.judge"), source="eikos_decisions", target_probs=[0.02, 0.96, 0.02], target_probs_keys=[1, 2, 3])]
    img = {"image": "data/decision-p2c/image-replay/images/abc.jpg", "sha256": "ab" * 32, "width": 8, "height": 8}
    replay = [dict(_rec(f"vqav2:{i}", BOOL_FIELD, None if i == 0 else True, state={}), images=[img], replay="image-v2.1",
                   target_distribution={"true": 0.8, "false": 0.2} if i == 1 else None) for i in range(5)]
    for r in replay:
        if r["target_distribution"] is None: r.pop("target_distribution")
    return p2b, dist, p2c, eikos, replay


def test_build_p2c_manifest_partitions_soft_targets_caps_and_gates(tmp_path):
    import build_p2c_manifest as bm
    p2b, dist, p2c, eikos, replay = _p2c_inputs()
    delta, fresh, judge_dev, counts = bm.build(p2b, p2c, eikos, replay[:2], replay, dist, ("telecom",), cap=4, seed="t")
    by = {r["id"]: r for r in fresh}
    # relabel: agree -> target_probs + rationale (unknown key present); answerable train disagreement dropped; unknown gold and eval rows kept hard
    r = by["p2_teacher:d0:0"]; assert r["target_probs"] == {"full_refund": 0.0, "partial_refund": 0.8, "no_refund": 0.1, "unknown": 0.1} and r["rationale"] == "Clause 2."
    assert "p2_teacher:d0:1" not in by and counts["relabel_dropped_answerable_train"] == 1
    assert "target_probs" not in by["p2_teacher:d0:2"] and by["p2_teacher:d0:2"]["target"] is None and counts["relabel_disagree_kept_hard_unknown_gold"] == 1
    assert "target_probs" not in by["p2_teacher:d1:0"] and by["p2_teacher:d1:0"]["partition"] == "dev" and counts["relabel_missing"] == 2   # d2, d3
    # partitions
    assert by["p2_teacher:d2:0"]["partition"] == "calibration" and by["p2_teacher:d3:0"]["partition"] == "test"
    assert [r["id"] for r in judge_dev] == [f"p2_teacher:j{i}:0" for i in range(3)] and all(r["partition"] == "dev" for r in judge_dev)
    assert by["p2_teacher:j0:2"]["partition"] == "extra_dev" and by["eikos_decisions:core:e0"]["partition"] == "extra_dev"
    assert {r["id"] for r in fresh if r["partition"] == "dev"} == {"p2_teacher:d1:0"} | {r["id"] for r in judge_dev}
    # target_probs: eikos lists -> label dicts with unknown = 0; p2c dicts gain the unknown key
    e = next(r for r in fresh if r["family"] == "eikos.rules_book" and r["partition"] == "train")
    assert e["target_probs"] == ({"true": 0.95, "false": 0.05, "unknown": 0.0} if e["target"] else {"true": 0.05, "false": 0.95, "unknown": 0.0}) and "target_probs_keys" not in e
    assert by["eikos_decisions:core:o0"]["target_probs"] == {"1": 0.02, "2": 0.96, "3": 0.02, "unknown": 0.0}
    assert by["p2_teacher:j3:0"]["target_probs"] == {"response_a": 0.1, "response_b": 0.9, "both_or_neither": 0.0, "unknown": 0.0}
    assert all("unknown" in r["target_probs"] for r in fresh if "target_probs" in r)
    # caps: templated families (prog-* batch, eikos.rules_*) keep 4 train rows each, deterministically; others untouched
    tr = [r for r in fresh if r["partition"] == "train"]
    assert sum(r["family"] == "temporal_arithmetic" for r in tr) == 4 and sum(r["family"] == "eikos.rules_book" for r in tr) == 4
    assert counts["capped:temporal_arithmetic"] == 8 and counts["capped:eikos.rules_book"] == 3 and sum(r["family"] == "judge_pairwise" for r in tr) == 2
    assert [r["id"] for r in bm.build(p2b, p2c, eikos, replay[:2], replay, dist, ("telecom",), cap=4, seed="t")[1]] == [r["id"] for r in fresh]
    # image replay: 2 rows in the delta lane, 5 in the fresh lane, labels untouched, unknown-gold replay rows kept
    assert sum(r["mix"] == "image_replay" for r in delta) == 2 and sum(r["mix"] == "image_replay" for r in fresh) == 5
    assert by["vqav2:1"]["target_distribution"] == {"true": 0.8, "false": 0.2} and by["vqav2:0"]["target"] is None and "target_probs" not in by["vqav2:0"]
    assert [r for r in delta if r["mix"] != "image_replay"] == [r for r in fresh if r["mix"] != "image_replay"]
    # every row renders through the trainer's code path, soft targets included, with unknown among the choices
    from decision_data import render
    from vision_decision.contracts import UNKNOWN
    for r in fresh:
        _, choices, _, target = render(r, soft_targets=True)
        assert choices[-1][0] == UNKNOWN
        if "target_probs" in r:
            assert abs(sum(target.probs) - 1) < 1e-6 and target.gold == [v for v, _ in choices].index(UNKNOWN if r["target"] is None else r["target"])
    rep = bm.report(delta, fresh, judge_dev, counts, 0.15)
    d = rep["decision-p2c"]
    assert d["by_mix"]["p2b"]["unknown_gold_rows"] == 1 and d["by_mix"]["image_replay"]["unknown_gold_rows"] == 1 and d["image_replay_train_rows"] == 2
    assert rep["unknown_share_warning"] is True and rep["judge_dev"]["rows"] == 3 and 0 < d["text_train_with_target_probs"] <= 1
    # gates: contract failures are dropped and counted; duplicate ids and JevBench overlap abort
    bad = dict(p2c[-1], id="p2_teacher:bad:0"); bad["request"] = dict(bad["request"], fields=[dict(CHOICE_FIELD, options=CHOICE_FIELD["options"][:1])])
    d2, _, _, c2 = bm.build(p2b, p2c + [bad], eikos, [], [], dist, ("telecom",), cap=100)
    assert "p2_teacher:bad:0" not in {r["id"] for r in d2} and c2["contract_dropped:p2c"] == 1
    with pytest.raises(SystemExit, match="duplicate"):
        bm.build(p2b, p2c + [p2c[0]], eikos, [], [], dist, ("telecom",), cap=100)
    leak = "the quarterly compliance officer must countersign every refund above the regional threshold before release"
    ref = tmp_path / "hard.jsonl"; ref.write_text(json.dumps({"state": leak}) + "\n")
    with pytest.raises(SystemExit, match="JevBench 8-gram lint"):
        bm.build(p2b, p2c + [dict(p2c[-1], id="p2_teacher:leak:0", request=dict(p2c[-1]["request"], state=leak))], eikos, [], [], dist, ("telecom",),
                 cap=100, reference=reference_ngrams([ref]))
    with pytest.raises(SystemExit, match="JevBench 8-gram lint"):   # replay rows are linted too
        bm.build(p2b, p2c, eikos, [], [dict(replay[1], id="vqav2:leak", request=dict(replay[1]["request"], fields=[dict(BOOL_FIELD, question=leak)]))],
                 dist, ("telecom",), cap=100, reference=reference_ngrams([ref]))


def test_build_p2c_manifest_cli_writes_three_manifests(tmp_path):
    import build_p2c_manifest as bm
    from v1_text.common import write_jsonl
    p2b, dist, p2c, eikos, replay = _p2c_inputs()
    for name, rows in (("p2b", p2b), ("dist", dist), ("p2c", p2c), ("eikos", eikos), ("fresh", replay), ("delta", replay[:2])):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    ref = tmp_path / "ref.jsonl"; ref.write_text(json.dumps({"state": "an unrelated reference sentence with more than eight words in it"}) + "\n")
    out = tmp_path / "m"
    argv = ["--p2b-records", str(tmp_path / "p2b.jsonl"), "--relabel", str(tmp_path / "dist.jsonl"), "--p2c-records", str(tmp_path / "p2c.jsonl"),
            "--eikos-records", str(tmp_path / "eikos.jsonl"), "--image-replay-fresh", str(tmp_path / "fresh.jsonl"), "--image-replay-delta", str(tmp_path / "delta.jsonl"),
            "--holdout-domains", "telecom", "--cap-per-family", "4", "--reference", str(ref), "--out", str(out / "decision-p2c.jsonl"),
            "--out-fresh", str(out / "decision-p2c-fresh.jsonl"), "--out-judge-dev", str(out / "decision-p2c-judge-dev.jsonl"), "--report", str(out / "report.json")]
    assert bm.main(argv) == 0
    rep = json.loads((out / "report.json").read_text())
    n = lambda f: len((out / f).read_text().splitlines())
    assert n("decision-p2c.jsonl") == rep["decision-p2c"]["rows"] and n("decision-p2c-fresh.jsonl") == rep["decision-p2c"]["rows"] + 3 and n("decision-p2c-judge-dev.jsonl") == 3
    assert rep["jevbench_lint"]["hits"] == 0 and rep["decision-p2c-fresh"]["image_replay_train_rows"] == 5
    with pytest.raises(SystemExit, match="missing input"):
        bm.main(argv[:4] + ["--p2c-records", str(tmp_path / "nope.jsonl")] + argv[6:])
