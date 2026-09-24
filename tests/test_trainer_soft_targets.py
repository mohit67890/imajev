"""v1.1 recipe flags for train_decision_lora_torch.py: soft targets, option permutation, rationale auxiliary loss.

CPU only, tiny tensors and a fake tokenizer: no model is loaded.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
from decision_data import (SoftTarget, approx_rationale_tokens, approx_tokens, batch_plan, permute_options,  # noqa: E402
                           rationale_text, render, resolve_probs_target, with_noise_state)
from decision_recipe import (IGNORE, append_rationale, collate_with_rationale, decision_loss, gold,  # noqa: E402
                             rationale_ids, rationale_lm_loss)
from vision_decision.contracts import UNKNOWN  # noqa: E402

OPTIONS = ['red', 'green', 'blue', 'amber', 'violet']


def choice(target='blue', rid='c1', **extra):
    field = {'id': 'q', 'type': 'choice', 'question': 'Which colour?', 'options': [{'value': v} for v in OPTIONS]}
    return dict({'id': rid, 'request': {'schema_version': '1.0', 'request_id': rid, 'state': 'text', 'fields': [field]},
                 'target': target, 'abstention_cause': None, 'images': []}, **extra)


def ordinal(target=2, rid='o1', **extra):
    field = {'id': 'q', 'type': 'ordinal', 'question': 'How bright?', 'levels': [{'value': v, 'description': f'level {v}'} for v in (1, 2, 3, 4)]}
    return dict({'id': rid, 'request': {'schema_version': '1.0', 'request_id': rid, 'state': 'text', 'fields': [field]},
                 'target': target, 'abstention_cause': None, 'images': []}, **extra)


def boolean(target=True, rid='b1', **extra):
    field = {'id': 'q', 'type': 'boolean', 'question': 'Is it lit?'}
    return dict({'id': rid, 'request': {'schema_version': '1.0', 'request_id': rid, 'state': 'text', 'fields': [field]},
                 'target': target, 'abstention_cause': None, 'images': []}, **extra)


# ---- (A) soft targets ------------------------------------------------------------------------------

def original_one(x, t):  # the trainer's pre-change loss, verbatim
    if isinstance(t, list): return -(torch.tensor(t, device=x.device, dtype=x.dtype) * x.log_softmax(0)).sum()
    return torch.nn.functional.cross_entropy(x[None], torch.tensor([t], device=x.device))[None].squeeze()


def test_default_losses_are_bitwise_unchanged():
    x = torch.randn(6, generator=torch.Generator().manual_seed(1))
    assert torch.equal(decision_loss(x, 3), original_one(x, 3))
    dist = [0.1, 0.2, 0.3, 0.0, 0.3, 0.1]
    assert torch.equal(decision_loss(x, dist), original_one(x, dist))
    assert gold(3) == 3 and gold(dist) == 2


@pytest.mark.parametrize('weight', [1.0, 0.5, 0.0])
def test_soft_loss_equals_hard_loss_for_one_hot(weight):
    x = torch.randn(6, generator=torch.Generator().manual_seed(2))
    one_hot = [0.0] * 6; one_hot[4] = 1.0
    hard = original_one(x, 4)
    assert torch.allclose(decision_loss(x, SoftTarget(4, one_hot), weight), hard, atol=1e-6)
    assert torch.allclose(decision_loss(x, one_hot), hard, atol=1e-6)


def test_soft_weight_mixes_soft_and_hard():
    x = torch.randn(4, generator=torch.Generator().manual_seed(3)); probs = [0.5, 0.3, 0.2, 0.0]
    soft = -(torch.tensor(probs) * x.log_softmax(0)).sum(); hard = torch.nn.functional.cross_entropy(x[None], torch.tensor([1]))
    assert torch.allclose(decision_loss(x, SoftTarget(1, probs), 0.25), 0.25 * soft + 0.75 * hard, atol=1e-6)
    assert torch.allclose(decision_loss(x, SoftTarget(1, probs), 0.0), hard, atol=1e-6)


def test_soft_cross_entropy_has_the_kl_gradient():
    x = torch.randn(5, requires_grad=True); probs = [0.4, 0.1, 0.2, 0.2, 0.1]
    decision_loss(x, SoftTarget(0, probs)).backward()
    assert torch.allclose(x.grad, x.detach().softmax(0) - torch.tensor(probs), atol=1e-6)


def test_target_probs_render_only_with_the_flag_and_gold_is_the_target():
    probs = {'red': 0.1, 'green': 0.5, 'blue': 0.3, 'unknown': 0.1}
    record = choice(target='blue', target_probs=probs)
    assert render(record)[3] == 2  # flag off: target_probs ignored, hard index
    _, choices, _, t = render(record, soft_targets=True)
    assert isinstance(t, SoftTarget) and t.gold == 2 and gold(t) == 2  # gold is target, not the argmax (green)
    assert [c[0] for c in choices][-1] == UNKNOWN and t.probs == pytest.approx([0.1, 0.5, 0.3, 0.0, 0.0, 0.1])


def test_gold_is_argmax_of_target_probs_when_target_is_absent():
    record = choice(target_probs={'amber': 0.6, '__unknown__': 0.4}); del record['target']
    resolved = resolve_probs_target(record)
    assert resolved['target'] == 'amber' and resolved['abstention_cause'] is None and 'target' not in record
    _, _, _, t = render(record, soft_targets=True)
    assert t.gold == 3 and t.probs[-1] == pytest.approx(0.4)
    unknown = choice(target_probs={'red': 0.2, 'unknown': 0.8}); del unknown['target']
    assert resolve_probs_target(unknown)['target'] is None and render(unknown, soft_targets=True)[3].gold == len(OPTIONS)


def test_boolean_and_ordinal_target_probs_use_value_aliases():
    t = render(boolean(target_probs={'true': 0.7, 'false': 0.2, 'unknown': 0.1}), soft_targets=True)[3]
    assert t.gold == 0 and t.probs == pytest.approx([0.7, 0.2, 0.1])
    t = render(ordinal(target=3, target_probs={'2': 0.25, '3': 0.75}), soft_targets=True)[3]
    assert t.gold == 2 and t.probs == pytest.approx([0, 0.25, 0.75, 0, 0])
    with pytest.raises(ValueError, match='unknown keys'):
        render(choice(target_probs={'mauve': 1.0}), soft_targets=True)


def test_no_match_relabel_drops_stale_target_probs():
    class Rng:  # forces the not_listed -> user "other" option relabel path
        def random(self): return 0.0
        def choice(self, seq): return seq[0]
    record = choice(target=None, abstention_cause='not_listed', target_probs={'unknown': 1.0})
    record['request']['state'] = 'x'
    out = with_noise_state(record, Rng())
    assert out['target'] == 'other' and 'target_probs' not in out and 'target_probs' in record
    assert render(out, soft_targets=True)[3] == len(OPTIONS)  # hard index of the added "other" option


# ---- (B) option permutation -----------------------------------------------------------------------

def test_permutation_is_deterministic_and_remaps_target_and_probs():
    probs = {'red': 0.05, 'green': 0.15, 'blue': 0.6, 'amber': 0.1, 'violet': 0.05, 'unknown': 0.05}
    record = choice(target='blue', target_probs=probs)
    a, b = permute_options(record, 0, 1), permute_options(record, 0, 1)
    assert a == b and record['request']['fields'][0]['options'] == [{'value': v} for v in OPTIONS]  # input untouched
    orders = {tuple(o['value'] for o in permute_options(record, 0, e)['request']['fields'][0]['options']) for e in range(8)}
    assert len(orders) > 1 and all(sorted(o) == sorted(OPTIONS) for o in orders)
    other = choice(target='blue', rid='c2', target_probs=probs)
    assert any(permute_options(record, 0, e)['request']['fields'][0]['options'] != permute_options(other, 0, e)['request']['fields'][0]['options'] for e in range(8))
    for epoch in range(8):
        _, choices, _, t = render(permute_options(record, 0, epoch), soft_targets=True)
        values = [c[0] for c in choices]
        assert values[-1] == UNKNOWN and values[t.gold] == 'blue'
        assert dict(zip(values, t.probs)) == pytest.approx({**{k: v for k, v in probs.items() if k != 'unknown'}, UNKNOWN: 0.05})
        assert values[render(permute_options(record, 0, epoch))[3]] == 'blue'


def test_permutation_leaves_ordinal_and_boolean_untouched():
    for record in (ordinal(target_probs={'1': 0.5, '4': 0.5}), boolean()):
        for epoch in range(5):
            assert permute_options(record, 3, epoch) is record
    assert [c[0] for c in render(permute_options(ordinal(), 0, 2))[1]] == [1, 2, 3, 4, UNKNOWN]
    assert [c[0] for c in render(permute_options(boolean(), 0, 2))[1]] == [True, False, UNKNOWN]


def test_render_shuffle_flag_disables_only_the_builtin_shuffle():
    import random
    record = choice()
    assert [c[0] for c in render(record, random.Random(5), shuffle=False)[1]] == OPTIONS + [UNKNOWN]
    assert render(record, random.Random(5))[:3] == render(record, random.Random(5), shuffle=True)[:3]


# ---- (C) rationale auxiliary loss -----------------------------------------------------------------

class Tokenizer:  # one token per character, ids offset so 0 is free for padding
    def encode(self, text, add_special_tokens=False): return [ord(c) % 90 + 10 for c in text]


def test_rationale_ids_prefix_and_cap():
    record = choice(rationale='  The sky region is saturated blue.  ')
    assert rationale_text(record) == ' Because: The sky region is saturated blue.'
    ids = rationale_ids(Tokenizer(), record, 64)
    assert ids == Tokenizer().encode(' Because: The sky region is saturated blue.')
    assert len(rationale_ids(Tokenizer(), record, 12)) == 12
    assert rationale_ids(Tokenizer(), choice(), 64) == [] and rationale_ids(Tokenizer(), choice(rationale='  '), 64) == []
    assert rationale_ids(Tokenizer(), record, 0) == []


def left_padded(lengths, width):
    ids = torch.zeros(len(lengths), width, dtype=torch.long); mask = torch.zeros_like(ids)
    for i, n in enumerate(lengths):
        ids[i, width - n:] = torch.arange(1, n + 1) + 100 * i; mask[i, width - n:] = 1
    return {'input_ids': ids, 'attention_mask': mask, 'mm_token_type_ids': torch.zeros_like(ids),
            'pixel_values': torch.ones(3, 4), 'image_grid_thw': torch.tensor([[1, 2, 2]])}


def test_rationale_is_appended_after_the_decision_position_and_excluded_from_readout():
    inputs = left_padded([5, 3], 5); rationales = [[7, 8, 9], [4]]
    out, position, labels = append_rationale(inputs, rationales, pad_id=0)
    assert out['input_ids'].shape == (2, 8) and position == 4
    assert torch.equal(out['input_ids'][:, :5], inputs['input_ids'])  # prompt (and so the decision token) untouched
    assert torch.equal(out['input_ids'][:, position], inputs['input_ids'][:, -1])  # readout still reads the last prompt token
    assert out['input_ids'][0, 5:].tolist() == [7, 8, 9] and out['input_ids'][1, 5:].tolist() == [4, 0, 0]
    assert out['attention_mask'][0, 5:].tolist() == [1, 1, 1] and out['attention_mask'][1, 5:].tolist() == [1, 0, 0]
    assert out['mm_token_type_ids'].shape == (2, 8) and not out['mm_token_type_ids'][:, 5:].any()
    assert out['pixel_values'] is inputs['pixel_values'] and out['image_grid_thw'] is inputs['image_grid_thw']
    # column 0 would be predicted from the decision position: never an LM target
    assert labels.tolist() == [[IGNORE, 8, 9], [IGNORE, IGNORE, IGNORE]]


def test_no_rationale_leaves_the_batch_untouched():
    inputs = left_padded([4, 2], 4)
    out, position, labels = append_rationale(inputs, [[], []], pad_id=0)
    assert out is inputs and position == -1 and labels is None
    examples = [('p', [], [11, 12], 0), ('p', [], [11, 12], 1, [])]
    assert len(collate_with_rationale(lambda ex: (inputs, [e[2] for e in ex], [e[3] for e in ex]), examples, 0)) == 3


def test_rationale_lm_loss_targets_only_rationale_tokens():
    torch.manual_seed(0)
    hidden = torch.randn(2, 8, 6); head = torch.randn(20, 6)
    labels = torch.tensor([[IGNORE, 8, 9], [IGNORE, IGNORE, IGNORE]])
    loss = rationale_lm_loss(hidden, head, labels, start=5)
    manual = torch.nn.functional.cross_entropy((hidden[0, 5:7] @ head.T), torch.tensor([8, 9]))
    assert loss.shape == (2,) and torch.allclose(loss[0], manual, atol=1e-6) and loss[1] == 0
    perturbed = hidden.clone(); perturbed[:, :5] += 100  # prompt and decision-position states do not enter the LM loss
    assert torch.allclose(rationale_lm_loss(perturbed, head, labels, start=5), loss)


def test_causal_model_decision_state_is_unchanged_by_appended_rationale():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True).eval(); embed = torch.nn.Embedding(200, 8)
    inputs = left_padded([5, 3], 5)
    out, position, _ = append_rationale(inputs, [[7, 8, 9], [4]], pad_id=0)
    def run(batch):
        n = batch['input_ids'].shape[1]
        return layer(embed(batch['input_ids']), src_mask=torch.nn.Transformer.generate_square_subsequent_mask(n))
    # (padding mask omitted: with causal attention the decision state already cannot see later positions)
    assert torch.allclose(run(inputs)[:, -1], run(out)[:, position], atol=1e-5)


def test_collate_with_rationale_counts_rationale_tokens_in_the_padded_length():
    prompts = left_padded([6, 4], 6)
    fake_collate = lambda ex: (prompts, [e[2] for e in ex], [e[3] for e in ex])
    examples = [('p', [], [11, 12], 0, [1, 2, 3, 4]), ('p', [], [11, 12], SoftTarget(1, [0.5, 0.5]), [5])]
    inputs, ids, targets, extra = collate_with_rationale(fake_collate, examples, pad_id=0)
    assert inputs['input_ids'].shape[-1] == 6 + 4  # the trainer's budget check reads this length
    assert extra['position'] == 5 and targets[1].gold == 1 and ids == [[11, 12], [11, 12]]


def test_token_budget_estimate_includes_capped_rationale():
    short = choice(rationale='Blue.'); long = choice(rid='c2', rationale='word ' * 200)
    base = approx_tokens(choice(), 400000)
    assert approx_tokens(short, 400000) == base and approx_tokens(short, 400000, 0) == base  # default unchanged
    assert approx_tokens(short, 400000, 64) == base + approx_rationale_tokens(short, 64) > base
    assert approx_rationale_tokens(long, 64) == 64 and approx_tokens(long, 400000, 64) == base + 64
    assert approx_rationale_tokens(choice(), 64) == 0


def test_batch_plan_packs_fewer_examples_when_rationales_count():
    records = [choice(rid=f'r{i}', rationale='The swatch is clearly blue under daylight. ' * 3) for i in range(40)]
    per = approx_tokens(records[0], 400000); budget = per * 8
    plain = batch_plan(records, 400000, budget, 64, 0, 0)
    assert plain == batch_plan(records, 400000, budget, 64, 0, 0, rationale_tokens=0)
    with_rationale = batch_plan(records, 400000, budget, 64, 0, 0, rationale_tokens=64)
    assert max(map(len, plain)) == 8 and max(map(len, with_rationale)) < 8
    assert sorted(i for b in with_rationale for i in b) == list(range(40))
    cost = approx_tokens(records[0], 400000, 64)
    assert all(cost * len(b) <= budget for b in with_rationale)
