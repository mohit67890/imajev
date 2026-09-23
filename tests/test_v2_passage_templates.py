"""Tests for the decision-v2 TEXT passage templates and the records they produce.

Run with:  PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v2_passage_templates.py
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decision_data import expand_fields, render                  # noqa: E402
from v2.templates import passage as T                            # noqa: E402
from vision_decision.contracts import UNKNOWN, Request           # noqa: E402

SOURCES = ["stackexchange", "wikipedia_paragraphs", "support_reviews"]

QUESTION_BANKS = {
    "topic": T.TOPIC_QUESTIONS, "intent": T.INTENT_QUESTIONS, "stance": T.STANCE_QUESTIONS,
    "adequacy_of_answer": T.ADEQUACY_QUESTIONS, "sentiment": T.SENTIMENT_QUESTIONS,
    "severity_urgency": T.URGENCY_QUESTIONS, "contains_claim": T.CLAIM_QUESTIONS,
    "entity_type": T.ENTITY_QUESTIONS, "next_action_routing": T.ROUTING_QUESTIONS,
    "pii_present": T.PII_QUESTIONS, "numeric_comparison": T.NUMERIC_QUESTIONS,
}


def sample_inputs(kind="qa", **over):
    base = {
        "id": "demo-1", "source": "stackexchange", "kind": kind,
        "title": "How long should I proof this dough?",
        "passage": ("I mixed a 65% hydration dough this morning and left it on the counter. "
                    "The kitchen is about 22 degrees and the recipe says two hours. "
                    "It has been 3 hours and the dough has barely risen. "
                    "The yeast was opened 6 months ago and cost $4 at the corner shop. "
                    "Should I throw it out and start again with fresh yeast?"),
        "answer": ("Old yeast loses potency. Proof a teaspoon in warm water with sugar for ten "
                   "minutes; if it does not foam, replace it and start the dough again."),
        "topic_hints": ["cooking and recipes"],
        "tags": ["bread", "yeast"],
        "entities": ["Red Star Yeast", "King Arthur Flour"],
        "numbers": [{"raw": "$4", "value": 4.0, "unit": "USD", "quantity": "price"},
                    {"raw": "3 hours", "value": 3.0, "unit": "hours", "quantity": "duration in hours"}],
        "clauses": ["The kitchen is about 22 degrees and the recipe says two hours",
                    "It has been 3 hours and the dough has barely risen"],
        "foreign_clauses": ["The rear derailleur needs a new cable before the hanger will align",
                            "The landlord must serve two months notice before the tenancy ends"],
    }
    base.update(over)
    return base


# ------------------------------------------------------------------------------- paraphrases

@pytest.mark.parametrize("family,bank", sorted(QUESTION_BANKS.items()))
def test_at_least_eight_distinct_paraphrases(family, bank):
    assert len(bank) >= 8, f"{family} has only {len(bank)} phrasings"
    assert len(set(bank)) == len(bank), f"{family} repeats a phrasing"


def test_every_family_is_reachable_and_named():
    assert set(T.SINGLE_FAMILIES) == set(QUESTION_BANKS)
    assert T.UNKNOWN_CAPABLE <= set(T.SINGLE_FAMILIES)


def test_templates_are_pure_and_deterministic():
    """Same seed, same field; and the call must not mutate its inputs."""
    inputs = sample_inputs()
    before = json.dumps(inputs, sort_keys=True)
    first = [q.field for q in T.make_one("topic", inputs, random.Random(11))]
    second = [q.field for q in T.make_one("topic", inputs, random.Random(11))]
    assert first == second
    assert json.dumps(inputs, sort_keys=True) == before


# ------------------------------------------------------------------------- option plausibility

def _check_field(field):
    Request.model_validate({"schema_version": "1.0", "request_id": "t", "state": {}, "fields": [field]})
    assert field["question"].strip() == field["question"]
    assert "{" not in field["question"] and "}" not in field["question"], "unfilled template slot"
    if field["type"] == "choice":
        values = [o["value"] for o in field["options"]]
        assert 2 <= len(values) <= 12, f"{len(values)} options"
        assert len(set(values)) == len(values)
        assert UNKNOWN not in values, "unknown is added by the loader, never listed"
        assert all(0 < len(v) <= 128 for v in values)
        assert all(v.strip() == v for v in values)
        for a in values:
            for b in values:
                assert a == b or a not in b, f"{a!r} is a substring of {b!r}"
    if field["type"] == "ordinal":
        levels = [lv["value"] for lv in field["levels"]]
        assert levels == sorted(set(levels))
        assert all(lv.get("description") for lv in field["levels"])
    if field["type"] == "boolean":
        assert field.get("yes_description") != field.get("no_description")


@pytest.mark.parametrize("family", sorted(QUESTION_BANKS))
@pytest.mark.parametrize("kind", ["qa", "article", "conversation"])
def test_option_sets_are_valid_and_plausible(family, kind):
    made = 0
    for seed in range(40):
        rng = random.Random(f"{family}:{kind}:{seed}")
        inputs = sample_inputs(kind=kind, source={"qa": "stackexchange", "article": "wikipedia_paragraphs",
                                                  "conversation": "support_reviews"}[kind])
        for unknown in (False, True):
            for q in T.make_one(family, inputs, rng, unknown=unknown):
                _check_field(q.field)
                assert q.family
                assert q.template_id
                made += 1
    assert made > 0, f"{family} produced nothing for kind={kind}"


def test_topic_option_count_is_six_to_twelve():
    for seed in range(60):
        for q in T.topic(sample_inputs(), random.Random(seed)):
            assert 6 <= len(q.field["options"]) <= 12


def test_topic_unknown_variant_excludes_the_passage_topic_group():
    hint = "cooking and recipes"
    group = T.TOPIC_GROUP_OF[hint]
    for seed in range(60):
        for q in T.topic(sample_inputs(), random.Random(seed), unknown=True):
            assert q.unknown_by_construction
            for option in q.field["options"]:
                assert T.TOPIC_GROUP_OF[option["value"]] != group


def test_routing_other_option_is_never_paired_with_an_unknown_construction():
    seen_other = seen_unknown = 0
    for seed in range(200):
        for q in T.routing(sample_inputs(), random.Random(seed)):
            seen_other += int(q.notes["user_supplied_other"])
        for q in T.routing(sample_inputs(), random.Random(seed), unknown=True):
            assert not q.notes["user_supplied_other"]
            seen_unknown += 1
    assert seen_other > 0 and seen_unknown > 0


def test_contains_claim_absent_uses_a_claim_from_another_passage():
    inputs = sample_inputs()
    absent = [q for seed in range(30)
              for q in T.contains_claim(inputs, random.Random(seed), absent=True)]
    present = [q for seed in range(30)
               for q in T.contains_claim(inputs, random.Random(seed), absent=False)]
    assert absent and present
    assert all(q.notes["claim_absent"] for q in absent)
    assert all(any(c[:40] in q.field["question"] for c in inputs["foreign_clauses"]) for q in absent)
    assert all(any(c[:40] in q.field["question"] for c in inputs["clauses"]) for q in present)


def test_about_thirty_per_cent_of_choice_questions_carry_option_criteria():
    described = total = 0
    for seed in range(400):
        rng = random.Random(seed)
        for q in T.topic(sample_inputs(), rng) + T.intent(sample_inputs(), rng):
            total += 1
            described += int(any("description" in o for o in q.field["options"]))
    assert 0.15 <= described / total <= 0.45, described / total


# ---------------------------------------------------------------------------- state variants

def test_state_is_a_string_about_forty_per_cent_of_the_time():
    counts = {"string": 0, "object": 0, "object+irrelevant": 0}
    for seed in range(4000):
        _, variant = T.build_state(sample_inputs(), random.Random(seed))
        counts[variant] += 1
    total = sum(counts.values())
    assert 0.35 <= counts["string"] / total <= 0.45, counts
    assert 0.06 <= counts["object+irrelevant"] / total <= 0.14, counts


def test_object_state_uses_named_fields_and_irrelevant_ones_are_extra():
    plain = [T.build_state(sample_inputs(), random.Random(s)) for s in range(500)]
    objects = [state for state, variant in plain if variant == "object"]
    noisy = [state for state, variant in plain if variant == "object+irrelevant"]
    assert objects and noisy
    assert all("question_body" in s for s in objects + noisy)
    irrelevant = {k for k, _ in T.IRRELEVANT_FIELDS}
    assert all(not irrelevant & set(s) for s in objects)
    assert all(irrelevant & set(s) for s in noisy)


# -------------------------------------------------------------------- multi-question integrity

def test_multi_question_requests_are_two_to_five_unique_fields():
    families = ["topic", "sentiment", "severity_urgency", "pii_present", "entity_type", "intent"]
    for seed in range(60):
        rng = random.Random(seed)
        questions = T.multi_question(sample_inputs(), rng, families)
        assert 2 <= len(questions) <= 5
        ids = [q.field["id"] for q in questions]
        assert len(set(ids)) == len(ids)
        request = {"schema_version": "1.0", "request_id": "m", "state": {"a": "b"},
                   "fields": [q.field for q in questions]}
        Request.model_validate(request)
        for q in questions:
            _check_field(q.field)


# ------------------------------------------------------------------- the converted records

def _records(source, cap=4000):
    path = ROOT / "data" / "decision-v2" / source / "records.jsonl"
    if not path.is_file():
        pytest.skip(f"{source} has not been converted yet")
    rows = []
    with path.open() as handle:
        for n, line in enumerate(handle):
            if n >= cap:
                break
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("source", SOURCES)
def test_converted_unknown_construction_share_is_fifteen_to_twentyfive_per_cent(source):
    report = ROOT / "data" / "decision-v2" / source / "conversion.json"
    if not report.is_file():
        pytest.skip(f"{source} has not been converted yet")
    share = json.loads(report.read_text())["unknown_share"]
    assert 0.15 <= share <= 0.25, f"{source} unknown-by-construction share is {share}"


@pytest.mark.parametrize("source", SOURCES)
def test_converted_records_render_through_the_training_path(source):
    rng = random.Random(3)
    for r in _records(source, cap=1500):
        probe = json.loads(json.dumps(r))
        if len(probe["request"]["fields"]) > 1:
            probe["targets"] = {f["id"]: None for f in probe["request"]["fields"]}
        items = expand_fields(probe)
        assert len(items) == len(r["request"]["fields"])
        for item in items:
            header, choices, texts, index = render(item, rng)
            assert choices[index][0] == UNKNOWN          # pending rows have no target yet
            assert len(texts) == len(choices)
            assert header.startswith("Inspect the available evidence")


@pytest.mark.parametrize("source", SOURCES)
def test_converted_multi_question_records_keep_field_ids_unique(source):
    multi = [r for r in _records(source) if len(r["request"]["fields"]) > 1]
    assert multi, f"{source} produced no multi-question records in the sample"
    for r in multi:
        ids = [f["id"] for f in r["request"]["fields"]]
        assert 2 <= len(ids) <= 5
        assert len(set(ids)) == len(ids)
        assert r["family"] == "multi_question"
        assert set(r["field_families"]) == set(ids)
        assert set(r["unknown_by_construction"]) == set(ids)
        assert r["template_id"].count(";") == len(ids) - 1


@pytest.mark.parametrize("source", SOURCES)
def test_converted_records_are_pending_candidates(source):
    for r in _records(source, cap=500):
        assert r["target"] is None and r["abstention_cause"] is None
        assert r["pseudo_label"] == "pending"
        assert r["images"] == []
        assert r["license"]["spdx"] in {"CC-BY-SA-4.0", "CC-BY-4.0", "MIT"}
        assert (r["partition"] == "test") == bool(r.get("pseudo_label_test"))
