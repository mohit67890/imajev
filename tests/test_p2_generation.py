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
