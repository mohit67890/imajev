import copy
import io
import json
import urllib.error

import pytest

from imajev_bench.annotations import import_prelabels, merge_reviews
from imajev_bench.api_models import (AzureOpenAIProvider, GeminiProvider, OpenAIProvider, build_prompt, load_key,
                                     GcloudToken, VertexGeminiProvider, make_provider, parse_reply, prelabel_export,
                                     run_api)
from imajev_bench.release import release_check
from imajev_bench.runner import digest, verify_run
from imajev_bench.schema import model_payload, validate_records
from imajev_bench.scoring import score
from imajev_bench.triage import audit_report, consensus, packet_sha256, plan_reviews


def record(i, gold=True, split="dev", **provenance):
    return {"id": f"r{i}", "group_id": f"g{i}", "track": "text", "family": "policy", "split": split, "images": [],
            "request": {"request_id": f"r{i}", "state": {"count": i}, "fields": [
                {"id": "decision", "type": "boolean", "question": f"Is the count above {i % 3}?"}]},
            "gold": gold, "annotation_status": "draft", "provenance": dict(provenance)}


class Reply:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeProvider:
    def __init__(self, name, answers, model="m-1"):
        self.name, self.model, self.answers = name, model, answers

    def complete(self, text, images, tokens, constrained):
        answer = self.answers(text)
        return json.dumps({"answer": answer, "evidence": "state count"}), f"{self.model}-served", {"tokens": 1}


# --- API adapter ----------------------------------------------------------------------------------

def test_openai_request_shape_and_parse():
    seen = {}

    def opener(request, timeout):
        seen["url"], seen["body"], seen["auth"] = request.full_url, json.loads(request.data), request.headers["Authorization"]
        return Reply({"model": "gpt-x-2026", "choices": [{"message": {"content": '{"answer": "yes", "evidence": "e"}'}}]})

    provider = OpenAIProvider("gpt-x", "sk-test", opener=opener)
    text, options = build_prompt(model_payload(record(1)), order_seed=None)
    reply, model, _ = provider.complete(text, [("image/png", "QUJD")], [t for t, _, _ in options], True)
    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-test"
    content = seen["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Image 1:"} and content[1]["image_url"]["url"].startswith("data:image/png")
    assert seen["body"]["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["yes", "no", "unknown"]
    assert parse_reply(reply, options) == {"status": "answered", "value": True, "evidence": "e"} and model == "gpt-x-2026"


def test_gemini_request_shape_and_retry():
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, io.BytesIO(b"slow down"))
        return Reply({"modelVersion": "gemini-y-001", "candidates": [{"content": {"parts": [{"text": '{"answer":"unknown","evidence":""}'}]}}]})

    provider = GeminiProvider("gemini-y", "g-key", opener=opener, sleep=sleeps.append)
    text, options = build_prompt(model_payload(record(1)), order_seed=None)
    reply, model, _ = provider.complete(text, [], ["yes", "no"], True)
    body = json.loads(calls[-1].data)
    assert calls[-1].full_url.endswith("/models/gemini-y:generateContent") and calls[-1].headers["X-goog-api-key"] == "g-key"
    assert body["generationConfig"]["responseSchema"]["properties"]["answer"]["enum"] == ["yes", "no", "unknown"]
    assert sleeps == [15.0] and model == "gemini-y-001"  # 429: minute-scale backoff
    assert parse_reply(reply, options)["status"] == "abstained"


def test_parse_reply_never_turns_garbage_into_abstention():
    _, options = build_prompt(model_payload(record(1)), order_seed=None)
    assert parse_reply("I think yes", options)["error_type"] == "no_json"
    assert parse_reply('{"answer": "maybe"}', options)["error_type"] == "out_of_domain"
    assert parse_reply('```json\n{"answer": "no", "evidence": "x"}\n```', options)["value"] is False


def test_load_key_prefers_environment_then_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env.local"
    env.write_text('OPENAI_API_KEY="from-file"\n')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_key("openai", env) == "from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert load_key("openai", env) == "from-env"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        load_key("gemini", env)


def test_run_api_is_scorable_and_guards_hidden_test(tmp_path):
    records = validate_records([record(i) for i in range(4)], tmp_path)
    provider = FakeProvider("openai", lambda text: "yes")
    path = run_api(records, tmp_path, tmp_path / "run", provider)
    verify_run(records, path)
    rows = {json.loads(line)["id"]: json.loads(line) for line in path.read_text().splitlines()}
    assert score(records, rows, 0)["capability"]["all_records"]["correct"] == 4
    assert "sk-" not in (tmp_path / "run" / "manifest.json").read_text()
    hidden = validate_records([record(9, split="test")], tmp_path)
    with pytest.raises(ValueError, match="hidden-test"):
        run_api(hidden, tmp_path, tmp_path / "blocked", provider)


# --- triage workflow ------------------------------------------------------------------------------

def human_export(records, reviewer, values, packet_records=None):
    return {"format_version": "0.0.1", "protocol": "imajev-bench-blind-review-v1", "reviewer_id": reviewer,
            "purpose": "independent_review", "input_sha256": packet_sha256(packet_records or records), "flags": [],
            "reviews": [{"id": rid, "reviewer_id": reviewer, "value": value, "evidence": "checked",
                         "input_sha256": digest(model_payload(next(r for r in records if r["id"] == rid)))}
                        for rid, value in values.items()]}


def prelabelled(tmp_path, n=10, gemini_disagrees=("r3",)):
    records = validate_records([record(i) for i in range(n)], tmp_path)
    runs = [run_api(records, tmp_path, tmp_path / "openai", FakeProvider("openai", lambda t: "yes")),
            run_api(records, tmp_path, tmp_path / "gemini", FakeProvider(
                "gemini", lambda t: "no" if any(f"above {int(r[1:]) % 3}?" in t and f'"count": {int(r[1:])}' in t
                                                  for r in gemini_disagrees) else "yes"))]
    for run in runs:
        records = import_prelabels(records, prelabel_export(run.parent, records))
    return validate_records(records, tmp_path)


def test_prelabels_do_not_count_as_reviews_and_model_exports_are_rejected(tmp_path):
    records = prelabelled(tmp_path)
    assert all(r["annotation_status"] == "draft" and "reviews" not in r["provenance"] for r in records)
    assert consensus(records[0]) == (True, True) and consensus(records[3]) == (False, None)
    with pytest.raises(ValueError, match="already has a pre-label"):
        import_prelabels(records, prelabel_export(tmp_path / "openai", records))
    model_export = human_export(records, "bot", {"r0": True}) | {"annotator_type": "model"}
    with pytest.raises(ValueError, match="Model annotations"):
        merge_reviews(records, [model_export], tmp_path)


def test_single_route_promotes_only_on_human_agreement(tmp_path):
    records, summary = plan_reviews(prelabelled(tmp_path), audit_share=0.0)
    assert summary["single"] == 9 and summary["double"] == 1
    values = {r["id"]: True for r in records}
    values["r1"] = False  # human disagrees with the model consensus
    merged = merge_reviews(records, [human_export(records, "human-a", values)], tmp_path)
    by_id = {r["id"]: r for r in merged}
    assert by_id["r0"]["annotation_status"] == "reviewed" and by_id["r0"]["provenance"]["review_route"] == "consensus_verified"
    assert by_id["r1"]["annotation_status"] == "draft" and by_id["r1"]["provenance"]["requires_second_review"]
    assert by_id["r3"]["annotation_status"] == "draft"  # models disagreed: needs a second human
    second = [r for r in merged if r["provenance"]["review_plan"] == "double"]
    assert {r["id"] for r in second} == {"r1", "r3"}
    final = merge_reviews(merged, [human_export(merged, "human-b", {"r1": True, "r3": True}, second)], tmp_path)
    by_id = {r["id"]: r for r in final}
    assert by_id["r3"]["annotation_status"] == "reviewed" and by_id["r3"]["provenance"]["review_route"] == "double_human"
    assert by_id["r1"]["provenance"]["requires_adjudication"]


def test_schema_rejects_weak_consensus(tmp_path):
    records, _ = plan_reviews(prelabelled(tmp_path), audit_share=0.0)
    merged = merge_reviews(records, [human_export(records, "human-a", {"r0": True})], tmp_path)
    good = next(r for r in merged if r["id"] == "r0")
    validate_records([good], tmp_path, require_reviewed=True)
    one_provider = copy.deepcopy(good)
    one_provider["provenance"]["model_prelabels"] = one_provider["provenance"]["model_prelabels"][:1]
    audited = copy.deepcopy(good)
    audited["provenance"]["audit_sample"] = True
    for bad in (one_provider, audited):
        with pytest.raises(ValueError):
            validate_records([bad], tmp_path)


def test_audit_sample_routes_whole_clusters_and_measures_consensus_error(tmp_path):
    records, summary = plan_reviews(prelabelled(tmp_path, n=20, gemini_disagrees=()), audit_share=0.25, seed=3)
    assert summary["audit_sample"] == 5 and summary["double"] == 5
    audited = [r for r in records if r["provenance"]["audit_sample"]]
    first = {r["id"]: True for r in records}
    first[audited[0]["id"]] = False
    merged = merge_reviews(records, [human_export(records, "a", first)], tmp_path)
    second_packet = [r for r in merged if r["provenance"]["review_plan"] == "double"]
    second = {r["id"]: (False if r["id"] == audited[0]["id"] else True) for r in second_packet}
    final = merge_reviews(merged, [human_export(merged, "b", second, second_packet)], tmp_path)
    report = audit_report(final)
    assert report["audited"] == 5 and report["consensus_errors"] == 1 and report["error_ids"] == [audited[0]["id"]]
    gates = {g["gate"]: g for g in release_check(final, tmp_path)["gates"]}
    assert gates["consensus_audit"]["passed"] is False


def test_azure_routes_and_factory(tmp_path):
    seen = []

    def opener(request, timeout):
        seen.append((request.full_url, request.headers.get("Api-key"), json.loads(request.data)))
        return Reply({"model": "gpt-deploy", "choices": [{"message": {"content": '{"answer":"no","evidence":"e"}'}}]})

    v1 = AzureOpenAIProvider("my-deploy", "az-key", "https://res.openai.azure.com/", opener=opener)
    v1.complete("q", [], ["yes", "no"], True)
    dated = AzureOpenAIProvider("my-deploy", "az-key", "https://res.openai.azure.com", "2024-10-21", opener=opener)
    dated.complete("q", [], ["yes", "no"], True)
    assert seen[0][0] == "https://res.openai.azure.com/openai/v1/chat/completions" and seen[0][2]["model"] == "my-deploy"
    assert seen[1][0].endswith("/openai/deployments/my-deploy/chat/completions?api-version=2024-10-21")
    assert "model" not in seen[1][2] and seen[0][1] == "az-key"
    env = tmp_path / ".env.local"
    env.write_text("AZURE_OPENAI_ENDPOINT=https://res.openai.azure.com\nAZURE_OPENAI_API_KEY=k\n")
    provider = make_provider("azure-openai", "my-deploy", env)
    assert provider.endpoint == "https://res.openai.azure.com" and provider.api_version is None


def test_azure_endpoint_with_path_is_normalised():
    provider = AzureOpenAIProvider("d", "k", "https://res.openai.azure.com/openai/v1/")
    assert provider.endpoint == "https://res.openai.azure.com"


def test_vertex_uses_gcloud_token_and_refreshes_on_401(tmp_path):
    tokens = iter(["tok-1", "tok-2"])
    token = GcloudToken(command=["true"])
    token.get = lambda refresh=False: next(tokens)
    seen = []

    def opener(request, timeout):
        seen.append((request.full_url, request.headers["Authorization"]))
        if len(seen) == 1:
            raise urllib.error.HTTPError(request.full_url, 401, "expired", {}, io.BytesIO(b"expired"))
        return Reply({"modelVersion": "gemini-z", "candidates": [{"content": {"parts": [{"text": '{"answer":"yes","evidence":"e"}'}]}}]})

    provider = VertexGeminiProvider("gemini-z", "proj-1", "global", token=token, opener=opener)
    reply, model, _ = provider.complete("q", [("image/png", "QUJD")], ["yes", "no"], True)
    assert seen[0][0] == ("https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/"
                          "publishers/google/models/gemini-z:generateContent")
    assert [auth for _, auth in seen] == ["Bearer tok-1", "Bearer tok-2"] and model == "gemini-z"
    regional = VertexGeminiProvider("m", "p", "us-central1", token=token)
    assert regional.base_url.startswith("https://us-central1-aiplatform.googleapis.com/")


def test_runs_survive_review_bookkeeping_but_not_gold_changes(tmp_path):
    records = validate_records([record(i) for i in range(3)], tmp_path)
    path = run_api(records, tmp_path, tmp_path / "run", FakeProvider("openai", lambda t: "yes"))
    touched = copy.deepcopy(records)
    touched[0]["provenance"]["audit_sample"] = True
    verify_run(touched, path)
    corrected = copy.deepcopy(records)
    corrected[0]["gold"] = False
    with pytest.raises(ValueError, match="dataset hash"):
        verify_run(corrected, path)
