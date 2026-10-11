"""Offline tests for the native torch cache stack (no model, no server).

Covers ResultCache (LRU, copies, sqlite persist+reopen, worker threads) and the
per-question keying, namespace/config invalidation, reuse-old-only-scores-new
and microbatch bounding of the TorchBackend cache layer.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from torch_caches import (ResultCache, artifact_namespace, cache_context_digest,  # noqa: E402
                          field_cache_key)


class FakeImage:
    mode = "RGB"
    size = (2, 2)

    def __init__(self, value=b"pixels"):
        self.value = value

    def tobytes(self):
        return self.value


@dataclass
class FakeField:
    id: str
    question: str

    def model_dump(self, mode=None):
        return {"id": self.id, "type": "boolean", "question": self.question}


class FakeRequest:
    def __init__(self, fields, state=None):
        self.fields = list(fields)
        self.state = state or {}

    def model_copy(self, update=None):
        update = update or {}
        return FakeRequest(update.get("fields", self.fields), update.get("state", self.state))


class FakeResult:
    def __init__(self, value):
        self.value = value

    def model_copy(self, deep=False):
        return FakeResult(self.value)

    def model_dump(self, mode=None):
        return {"value": self.value}

    @classmethod
    def model_validate(cls, payload):
        return cls(payload["value"])


class FakeBackend:
    """Minimal TorchBackend.score cache-layer double: result cache + microbatch chunking."""

    def __init__(self):
        self.model, self.adapter, self.rotations, self.fast = "imajev-test", "adapter", 1, False
        self.prompt_layout, self.readout_codes = "standard", 256
        self.merge_lora, self.shared_prefix_requested = False, True
        self.question_microbatch = 8
        self.result_cache_namespace = "artifact-v1"
        self.result_cache = ResultCache(16)
        self.score_calls, self.batch_calls, self._prefix_cache_metadata = [], [], None

    def context(self, images, state):
        return cache_context_digest(namespace=self.result_cache_namespace, model=self.model,
                                    adapter=self.adapter, rotations=self.rotations, fast=self.fast,
                                    merge_lora=self.merge_lora, shared_prefix=self.shared_prefix_requested,
                                    microbatch=self.question_microbatch, prompt_layout=self.prompt_layout,
                                    readout_codes=self.readout_codes, state=state, images=images)

    def score(self, images, request, thinking=None):
        if thinking is not None and getattr(thinking, "active", False):
            return self._uncached(images, request)
        context = self.context(images, request.state)
        keys = [field_cache_key(context, field) for field in request.fields]
        results, miss_idx, miss_fields = [None] * len(request.fields), [], []
        for i, (key, field) in enumerate(zip(keys, request.fields)):
            hit = self.result_cache.get(key)
            (miss_idx.append(i), miss_fields.append(field)) if hit is None else results.__setitem__(i, hit)
        if miss_fields:
            fresh, usage = self._uncached(images, request.model_copy(update={"fields": miss_fields}))
            for i, result in zip(miss_idx, fresh):
                results[i] = result
                self.result_cache.put(keys[i], result)
        else:
            usage = {"questions_ms": 0.0, "input_tokens": 0, "rotations": self.rotations}
        return results, {**usage, "cache_hits": len(request.fields) - len(miss_fields),
                         "cache_misses": len(miss_fields), "cache_entries": len(self.result_cache),
                         "cache_capacity": self.result_cache.capacity,
                         "cache_persistent": self.result_cache.persistent}

    def _uncached(self, images, request):
        self.score_calls.append([field.id for field in request.fields])
        return [FakeResult(field.id) for field in request.fields], {"questions_ms": 1.0, "input_tokens": 10}

    def _score_batched(self, images, compiled):
        size = self.question_microbatch
        if len(compiled) <= size:
            return self._score_chunk(images, compiled)
        combined = []
        for start in range(0, len(compiled), size):
            combined.extend(self._score_chunk(images, compiled[start:start + size]))
        return combined

    def _score_chunk(self, images, compiled):
        self.batch_calls.append(len(compiled))
        return list(compiled)


def _context(backend, images=((),), state=None):
    return backend.context(list(images), state or {"screen": "same"})


def test_result_cache_is_lru_and_returns_copies():
    cache = ResultCache(2)
    original = FakeResult("a")
    cache.put("a", original)
    cache.put("b", FakeResult("b"))

    hit = cache.get("a")
    assert hit.value == "a"
    assert hit is not original

    cache.put("c", FakeResult("c"))
    assert cache.get("b") is None
    assert cache.get("a").value == "a"
    assert cache.get("c").value == "c"


def test_persistent_cache_survives_new_instance(tmp_path):
    path = tmp_path / "results.sqlite3"
    first = ResultCache(4, path=path, result_type=FakeResult)
    first.put("a", FakeResult("persisted"))
    first.close()

    second = ResultCache(4, path=path, result_type=FakeResult)
    hit = second.get("a")
    assert hit is not None
    assert hit.value == "persisted"
    second.close()


def test_persistent_cache_works_from_worker_thread(tmp_path):
    path = tmp_path / "results.sqlite3"
    cache = ResultCache(4, path=path, result_type=FakeResult)
    cache.put("a", FakeResult("thread-safe"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        hit = pool.submit(cache.get, "a").result()

    assert hit is not None
    assert hit.value == "thread-safe"
    cache.close()


def test_cache_key_is_per_question_over_shared_evidence_context():
    backend = FakeBackend()
    context = _context(backend, [FakeImage()])
    assert _context(backend, [FakeImage()]) == context
    assert _context(backend, [FakeImage(b"other")]) != context

    first = field_cache_key(context, FakeField("a", "Question A?"))
    assert field_cache_key(context, FakeField("a", "Question A?")) == first
    assert field_cache_key(context, FakeField("b", "Question B?")) != first

    backend.result_cache_namespace = "artifact-v2"
    assert _context(backend, [FakeImage()]) != context

    backend.result_cache_namespace = "artifact-v1"
    backend.merge_lora = True
    assert _context(backend, [FakeImage()]) != context


def test_result_cache_reuses_old_questions_and_only_scores_new_ones():
    backend = FakeBackend()
    first = FakeRequest([FakeField("a", "A?"), FakeField("b", "B?")], {"state": 1})
    second = FakeRequest([FakeField("a", "A?"), FakeField("b", "B?"), FakeField("c", "C?")], {"state": 1})

    results1, usage1 = backend.score([], first)
    results2, usage2 = backend.score([], second)

    assert [result.value for result in results1] == ["a", "b"]
    assert [result.value for result in results2] == ["a", "b", "c"]
    assert backend.score_calls == [["a", "b"], ["c"]]
    assert usage1["cache_hits"] == 0
    assert usage1["cache_misses"] == 2
    assert usage2["cache_hits"] == 2
    assert usage2["cache_misses"] == 1
    assert usage2["cache_persistent"] is False


def test_persistent_install_reuses_answers_after_backend_restart(tmp_path):
    path = tmp_path / "results.sqlite3"
    request = FakeRequest([FakeField("a", "A?"), FakeField("b", "B?")], {"state": 1})

    first_backend = FakeBackend()
    first_backend.result_cache = ResultCache(16, path=path, result_type=FakeResult)
    first_backend.score([], request)
    first_backend.result_cache.close()

    second_backend = FakeBackend()
    second_backend.result_cache = ResultCache(16, path=path, result_type=FakeResult)
    results, usage = second_backend.score([], request)

    assert [result.value for result in results] == ["a", "b"]
    assert second_backend.score_calls == []
    assert usage["cache_hits"] == 2
    assert usage["cache_misses"] == 0
    assert usage["cache_persistent"] is True
    second_backend.result_cache.close()


def test_thinking_bypasses_result_cache():
    class Thinking:
        active = True

    backend = FakeBackend()
    request = FakeRequest([FakeField("a", "A?")], {"state": 1})
    backend.score([], request)
    backend.score([], request, thinking=Thinking())
    assert backend.score_calls == [["a"], ["a"]]


def test_artifact_namespace_changes_with_adapter(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "w.bin").write_bytes(b"v1")
    first = artifact_namespace(bundle, adapter)
    (adapter / "w.bin").write_bytes(b"v22!")
    assert artifact_namespace(bundle, adapter) != first


def test_microbatching_keeps_large_logical_panel_bounded():
    backend = FakeBackend()
    backend.question_microbatch = 8

    assert backend._score_batched([], list(range(21))) == list(range(21))
    assert backend.batch_calls == [8, 8, 5]


def test_probe_pair_needs_same_type():
    from types import SimpleNamespace

    from torch_prefix_cache import PrefixScorer, PrefixUnsuitable

    def field(kind):
        return SimpleNamespace(type=kind)

    mixed = [(field("boolean"), None, None, None, None), (field("choice"), None, None, None, None)]
    scorer = PrefixScorer.__new__(PrefixScorer)
    scorer.enabled, scorer.validated_text, scorer.validated_visual = True, False, False
    scorer.max_delta, scorer.error, scorer.metadata = {}, None, {}
    scorer.engine = SimpleNamespace(device="cpu")
    scorer.fast, scorer.microbatch = False, 8
    calls = []
    out = scorer.maybe_validate_and_score(None, mixed, 1, lambda images, compiled: calls.append("fallback") or "ok")
    assert out == "ok" and calls == ["fallback"] and scorer.enabled
    same = [(field("boolean"), None, None, None, None), (field("boolean"), None, None, None, None)]
    import torch

    scorer.render_groups = lambda images, compiled, rotations: ([[(("r", [1]),)], [(("r", [1]),)]], None)
    scorer.reference_logits = lambda images, examples: [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    import torch_prefix_cache as prefix_cache

    real = prefix_cache.score_rendered_prefix_cached_hierarchical
    prefix_cache.score_rendered_prefix_cached_hierarchical = (
        lambda *a, **k: ([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])], {}))
    try:
        scorer.score = lambda images, compiled, rotations: "scored"
        assert scorer.maybe_validate_and_score(None, same, 1, lambda images, compiled: "fallback") == "scored"
        assert scorer.validated_text
    finally:
        prefix_cache.score_rendered_prefix_cached_hierarchical = real



def test_degenerate_hidden_falls_back_without_disabling():
    import torch

    from torch_prefix_cache import PrefixScorer, PrefixUnsuitable, _read_hidden

    try:
        _read_hidden(torch.zeros(()), 0, 0)
    except PrefixUnsuitable:
        pass
    else:
        raise AssertionError("0D hidden must raise PrefixUnsuitable")
