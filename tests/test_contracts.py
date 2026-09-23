import json
import math
from pathlib import Path
import pytest
from pydantic import ValidationError
from vision_decision.contracts import Request, BooleanField, UNKNOWN
from vision_decision.scoring import compile_question, candidates, result_from_logits, verified_label_ids, cyclic_offsets, rotate, combine_rotations
from vision_decision.images import load_image
from PIL import Image

def request_data():
    return json.loads(Path("examples/product.json").read_text())

@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra=1),
    lambda d: d.update(fields=[]),
    lambda d: d["fields"].append(d["fields"][0]),
    lambda d: d["fields"][0]["options"][0].update(value=UNKNOWN),
    lambda d: d["fields"][0]["options"][0].update(value="shirt"),
    lambda d: d["execution"].update(mode="gated"),
    lambda d: d["execution"].update(allow_external_fallback=True),
    lambda d: d.update(state={"bad": float("nan")}),
    lambda d: d.update(state={"huge": "x" * 32769}),
])
def test_reject_invalid_requests(mutation):
    d = request_data(); mutation(d)
    with pytest.raises((ValidationError, ValueError)): Request.model_validate(d)

def test_boolean_and_abstention_preserve_types():
    field = BooleanField(id="damaged", type="boolean", question="Damaged?")
    choices = candidates(field)
    assert result_from_logits(choices, [1000., -1000., 0.]).value is True
    result = result_from_logits(choices, [0., 0., 20.])
    assert result.value is None and result.status == "abstained"
    assert math.isclose(sum(result.scores.values()), 1)

@pytest.mark.parametrize("logits", [[float("nan"), 0., 0.], [float("inf"), 0., 0.], [1.]])
def test_invalid_logits_are_errors(logits):
    with pytest.raises(ValueError): result_from_logits([(True,""),(False,""),(UNKNOWN,"")], logits)

def test_ordinal():
    d = request_data(); d["fields"] = [{"id":"quality", "type":"ordinal", "question":"Usability?", "levels":[{"value":1,"description":"Poor"},{"value":3,"description":"Clear"}]}]
    f = Request.model_validate(d).fields[0]
    assert result_from_logits(candidates(f), [0., 4., 0.]).value == 3
    d["fields"][0]["levels"][0]["value"] = True
    with pytest.raises(ValidationError): Request.model_validate(d)

def test_token_ties_follow_vocabulary_order_not_candidate_order():
    choices = [(True, "yes"), (False, "no"), (UNKNOWN, "unknown")]
    assert result_from_logits(choices, [26., 26., 23.], [90, 40, 100]).value is False
    assert result_from_logits(list(reversed(choices)), [23., 26., 26.], [100, 40, 90]).value is False
    assert result_from_logits(choices, [26.01, 26., 23.], [90, 40, 100]).value is True
    with pytest.raises(ValueError):
        result_from_logits(choices, [26., 26., 23.], [40, 40, 100])

def test_token_boundary_checked():
    class Tokenizer:
        def encode(self, s, **kwargs): return list(s.encode())
    assert verified_label_ids(Tokenizer(), "answer:", ["A", "B"]) == [65,66]
    with pytest.raises(ValueError): verified_label_ids(Tokenizer(), "answer:", ["AB"])

def test_image_loading(tmp_path):
    p = tmp_path / "image.png"; Image.new("RGB", (20,30)).save(p)
    image, meta = load_image(p)
    assert image.size == (20,30) and len(meta["sha256"]) == 64
    p.write_bytes(b"corrupt")
    with pytest.raises(OSError): load_image(p)

def test_animation_rejected(tmp_path):
    p = tmp_path / "image.png"
    Image.new("RGB", (10,10), "red").save(p, save_all=True, append_images=[Image.new("RGB", (10,10), "blue")])
    with pytest.raises(ValueError, match="Multi-frame"): load_image(p)

def test_rotation_offsets_cover_positions_evenly():
    assert cyclic_offsets(3, 4) == [0, 1, 2]
    assert cyclic_offsets(16, 4) == [0, 4, 8, 12]
    assert cyclic_offsets(5, None) == [0, 1, 2, 3, 4]
    assert rotate(["a", "b", "c"], 1) == ["b", "c", "a"]
    with pytest.raises(ValueError): cyclic_offsets(3, 0)

def test_full_rotation_cancels_a_pure_position_preference():
    choices = [("red", ""), ("blue", ""), (UNKNOWN, "")]
    content, position = [0.2, 0.5, -1.0], [3.0, 0.0, 0.0]  # strong first-slot bias
    passes = []
    for offset in cyclic_offsets(3, None):
        shown = rotate(content, offset)
        passes.append((offset, [shown[i] + position[i] for i in range(3)]))
    assert result_from_logits(choices, passes[0][1]).value == "red"  # bias wins in a single order
    combined = combine_rotations(choices, passes)
    assert combined.value == "blue"
    expected = result_from_logits(choices, content).scores
    assert all(math.isclose(combined.scores[k], expected[k], abs_tol=1e-9) for k in expected)

def test_rotation_inputs_validated():
    choices = [(True, ""), (False, ""), (UNKNOWN, "")]
    with pytest.raises(ValueError): combine_rotations(choices, [])
    with pytest.raises(ValueError): combine_rotations(choices, [(0, [0., 0., 0.]), (0, [0., 0., 0.])])
    with pytest.raises(ValueError): combine_rotations(choices, [(3, [0., 0., 0.])])

def test_option_description_is_optional_and_rendered_bare():
    d = request_data(); d["fields"] = [{"id":"color", "type":"choice", "question":"Color?", "options":[{"value":"red"},{"value":"blue","description":"Any blue shade."}]}]
    f = Request.model_validate(d).fields[0]
    _, choices, texts = compile_question(f, {})
    assert texts[:2] == ["red", "blue — Any blue shade."] and choices[-1][0] == UNKNOWN

def test_jev_style_request_and_response_round_trip():
    from vision_decision.jev_api import to_request, to_response
    payload = {"state": {"listing": {"color": "red"}}, "questions": {
        "matches": {"type": "noul", "instructions": "The product matches `listing.color`.", "criteria": {"true": "same colour family"}},
        "kind": {"type": "choice", "instructions": "What is it?", "criteria": {"mug": "a cup with a handle", "bowl": None}},
        "wear": {"type": "score", "instructions": "How worn is it?", "criteria": ["like new", "light wear", "heavy wear"]}}}
    request = to_request(payload)
    assert [f.type for f in request.fields] == ["boolean", "choice", "ordinal"]
    header, choices, texts = compile_question(request.fields[0], request.state)
    assert '"color": "red"' in header and texts[0] == "yes — same colour family" and texts[1] == "no"
    results = [result_from_logits(candidates(request.fields[0]), [2., 0., -1.]),
               result_from_logits(candidates(request.fields[1]), [0., 3., 0.]),
               result_from_logits(candidates(request.fields[2]), [0., 2., 2., -5.])]
    out = to_response(request, results)["answers"]
    assert 0.84 < out["matches"]["noul"] < 0.88 and out["kind"]["choice"] == "bowl" and out["kind"]["type"] == "choice"
    assert 1.3 < out["wear"]["score"] < 1.5 and math.isclose(sum(out["wear"]["probabilities"].values()), 1)
    assert all("unknown_probability" in a for a in out.values())
    with pytest.raises(ValueError): to_request({"questions": {"x": {"type": "essay", "instructions": "?"}}})

def test_object_form_instructions_are_flattened_in_order():
    from vision_decision.jev_api import to_request, flatten_instructions
    text = flatten_instructions({"question": "How old is the item, as of today?", "today": "September 15, 2026", "units": ["years"]})
    assert text == 'question: How old is the item, as of today?\ntoday: September 15, 2026\nunits: ["years"]'
    request = to_request({"state": {"subject": "mug"}, "questions": {"age": {"type": "score", "instructions": {"question": "How worn is `subject`?"}, "criteria": ["None", "Minorly", "Majorly"]}}})
    assert request.fields[0].question == "question: How worn is `subject`?" and [l.value for l in request.fields[0].levels] == [0, 1, 2]
    with pytest.raises(ValueError): flatten_instructions({})

def test_confidence_matches_the_published_formula_and_unknown_pulls_noul_to_half():
    from vision_decision.jev_api import concentration, to_request, to_response, flatten_description
    assert math.isclose(concentration({"a": .82, "b": .17, "c": .01, "d": 0.}), (4 * .82 - 1) / 3)  # 76% in the playground
    assert concentration({"a": .5, "b": .5}) == 0 and concentration({"a": 1., "b": 0.}) == 1
    assert flatten_description({"what": "Returns", "examples": ["worn once"]}) == 'what: Returns; examples: ["worn once"]'
    request = to_request({"state": "a plain string state", "questions": {"q": {"type": "noul", "instructions": ["Context line.", "The mug is chipped."]}}})
    assert request.state == "a plain string state" and request.fields[0].question == "Context line.\nThe mug is chipped."
    abstaining = result_from_logits(candidates(request.fields[0]), [0., -1., 8.])
    assert abs(to_response(request, [abstaining])["answers"]["q"]["noul"] - 0.5) < 0.01

def test_text_only_request_compiles_without_an_image():
    """A request with no image uses the same fields, header and candidate lines as an image request.

    Zero-image support is a property of the prompt and the model call, not of the contract: nothing
    here mentions an image, so `to_request` / `compile_question` need no special case (no model loaded).
    """
    from vision_decision.jev_api import to_request, to_response
    payload = {"state": {"ticket": {"subject": "Charged twice", "plan": "Team"}}, "questions": {
        "department": {"type": "choice", "instructions": "Which department should own this ticket?",
                       "criteria": {"billing": "invoices and refunds", "technical_support": None}},
        "urgent": {"type": "noul", "instructions": "This ticket needs a reply today."},
        "frustration": {"type": "score", "instructions": {"question": "How frustrated is the customer?", "today": "September 22, 2026"},
                        "criteria": ["calm", "impatient", "angry"]}}}
    request = to_request(payload)
    assert [f.type for f in request.fields] == ["choice", "boolean", "ordinal"]
    assert request.fields[2].question == "question: How frustrated is the customer?\ntoday: September 22, 2026"
    headers = [compile_question(field, request.state)[0] for field in request.fields]
    assert all('"subject": "Charged twice"' in header for header in headers)
    # Every prompt shares the instruction and state prefix, which is what the zero-image prefill caches.
    shared = headers[0][:headers[0].index("Question:")]
    assert all(header.startswith(shared) for header in headers) and len(shared) > 100
    header, choices, texts = compile_question(request.fields[0], request.state)
    assert [c[0] for c in choices] == ["billing", "technical_support", UNKNOWN]
    assert texts[:2] == ["billing — invoices and refunds", "technical_support"]
    results = [result_from_logits(candidates(f), [3.0] + [0.0] * (len(candidates(f)) - 1)) for f in request.fields]
    answers = to_response(request, results)["answers"]
    assert answers["department"]["choice"] == "billing" and answers["urgent"]["noul"] > 0.5
    assert answers["frustration"]["legend"] == {"0": "calm", "1": "impatient", "2": "angry"}
