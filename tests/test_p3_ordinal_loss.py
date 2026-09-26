"""Phase 3 --ordinal-weight: squared earth-mover term for score (ordinal) fields.

CPU only, tiny tensors: no model is loaded.
"""
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
from decision_data import SoftTarget, permute_options, render, with_noise_state  # noqa: E402
from decision_recipe import (OrdinalTarget, decision_loss, gold, ordinal_emd2, ordinal_level_target,  # noqa: E402
                             ordinal_loss)
from vision_decision.contracts import UNKNOWN  # noqa: E402


def ordinal(target=3, rid='o1', levels=(1, 2, 3, 4, 5), **extra):
    field = {'id': 'q', 'type': 'ordinal', 'question': 'How bright?',
             'levels': [{'value': v, 'description': f'level {v}'} for v in levels]}
    return dict({'id': rid, 'request': {'schema_version': '1.0', 'request_id': rid, 'state': 'text', 'fields': [field]},
                 'target': target, 'abstention_cause': None, 'images': []}, **extra)


def choice(target='blue', rid='c1'):
    field = {'id': 'q', 'type': 'choice', 'question': 'Which colour?',
             'options': [{'value': v} for v in ('red', 'green', 'blue', 'amber')]}
    return {'id': rid, 'request': {'schema_version': '1.0', 'request_id': rid, 'state': 'text', 'fields': [field]},
            'target': target, 'abstention_cause': None, 'images': []}


def logits_for(probs):
    """Logits whose softmax is exactly `probs` (strictly positive entries)."""
    return torch.tensor(probs, dtype=torch.float32).log()


def batch_total(logits, targets, weight, soft_weight=1.0):
    """The trainer's batch_loss arithmetic for the decision + ordinal part."""
    loss = torch.stack([decision_loss(x, t, soft_weight) for x, t in zip(logits, targets)]).sum()
    if weight > 0:
        terms = [o for o in (ordinal_loss(x, t, soft_weight) for x, t in zip(logits, targets)) if o is not None]
        if terms:
            loss = loss + weight * torch.stack(terms).sum()
    return loss


def test_zero_when_prediction_equals_target():
    level_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
    x = logits_for([p * 0.8 for p in level_probs] + [0.2])  # 20% on unknown: renormalised away
    t = OrdinalTarget(SoftTarget(2, [p * 0.9 for p in level_probs] + [0.1]), 5)
    assert float(ordinal_loss(x, t)) == pytest.approx(0.0, abs=1e-7)
    # a confident correct prediction against a hard gold level is ~0 too
    x = torch.tensor([-30.0, -30.0, 30.0, -30.0, -30.0, -30.0])
    assert float(ordinal_loss(x, OrdinalTarget(2, 5))) == pytest.approx(0.0, abs=1e-12)


def test_far_off_mass_costs_more_than_near_off_mass():
    t = OrdinalTarget(4, 5)  # gold = top level
    near = logits_for([0.01, 0.01, 0.01, 0.9, 0.06, 0.01])  # most mass one level below gold
    far = logits_for([0.9, 0.01, 0.01, 0.01, 0.06, 0.01])   # the same mass at the opposite end
    assert float(ordinal_loss(far, t)) > 3 * float(ordinal_loss(near, t))
    # plain CE cannot tell the two apart (same probability on gold)
    assert float(decision_loss(near, 4)) == pytest.approx(float(decision_loss(far, 4)), abs=1e-5)


def test_formula_and_normalisation():
    # all mass on level 0 versus gold at level n-1: every CDF gap is 1, so the normalised value is 1
    for n in (2, 5, 10):
        x = torch.full((n + 1,), -40.0); x[0] = 40.0
        assert float(ordinal_loss(x, OrdinalTarget(n - 1, n))) == pytest.approx(1.0, abs=1e-6)
    p = [0.5, 0.25, 0.25]; q = [0.0, 0.0, 1.0]
    expected = ((0.5 - 0) ** 2 + (0.75 - 0) ** 2) / 2
    assert float(ordinal_emd2(logits_for(p), q)) == pytest.approx(expected, abs=1e-6)


def test_unknown_gold_and_non_ordinal_fields_get_no_term():
    x = torch.randn(6, generator=torch.Generator().manual_seed(0))
    assert ordinal_loss(x, OrdinalTarget(5, 5)) is None  # hard unknown gold (index n)
    assert ordinal_loss(x, OrdinalTarget(SoftTarget(5, [0.1, 0, 0, 0, 0, 0.9]), 5)) is None  # soft, gold unknown
    assert ordinal_loss(x, OrdinalTarget([0, 0, 0, 0, 0, 1.0], 5)) is None  # distribution with its mode on unknown
    assert ordinal_loss(x, 3) is None and ordinal_loss(x, [0.2] * 5 + [0.0]) is None  # choice / boolean targets
    assert ordinal_loss(x, SoftTarget(1, [0, 1.0, 0, 0, 0, 0])) is None
    # soft mass ~0 on the levels although the gold index is a level: skipped
    assert ordinal_level_target(OrdinalTarget(SoftTarget(1, [0, 0, 0, 0, 0, 1.0]), 5)) is None
    # a batch whose ordinal rows all have unknown gold: the loss equals the plain loss exactly
    targets = [OrdinalTarget(5, 5), 2]
    logits = [x, torch.randn(4, generator=torch.Generator().manual_seed(1))]
    assert torch.equal(batch_total(logits, targets, 0.15), batch_total(logits, [5, 2], 0.0))


def test_weight_zero_is_bit_identical():
    g = torch.Generator().manual_seed(3)
    logits = [torch.randn(6, generator=g), torch.randn(4, generator=g), torch.randn(6, generator=g)]
    plain = [3, 1, SoftTarget(2, [0.1, 0.2, 0.4, 0.2, 0.05, 0.05])]
    wrapped = [OrdinalTarget(3, 5), 1, OrdinalTarget(plain[2], 5)]
    for sw in (1.0, 0.5):
        reference = batch_total(logits, plain, 0.0, sw)
        assert torch.equal(batch_total(logits, wrapped, 0.0, sw), reference)  # wrapped, flag off
        assert torch.equal(batch_total(logits, plain, 0.15, sw), reference)   # flag on, nothing wrapped (dev rows)
        for x, a, b in zip(logits, plain, wrapped):
            assert torch.equal(decision_loss(x, b, sw), decision_loss(x, a, sw))
            assert gold(a) == gold(b)
    assert not torch.equal(batch_total(logits, wrapped, 0.15), batch_total(logits, plain, 0.0))


def test_soft_target_path_restricts_renormalises_and_mixes():
    record = ordinal(target=4, target_probs={'3': 0.3, '4': 0.5, 'unknown': 0.2})
    _, choices, _, t = render(record, soft_targets=True)
    assert [c[0] for c in choices] == [1, 2, 3, 4, 5, UNKNOWN] and isinstance(t, SoftTarget) and t.gold == 3
    wrapped = OrdinalTarget(t, len(choices) - 1)
    assert ordinal_level_target(wrapped, 1.0) == pytest.approx([0, 0, 0.375, 0.625, 0])
    assert ordinal_level_target(wrapped, 0.5) == pytest.approx([0, 0, 0.1875, 0.8125, 0])
    assert ordinal_level_target(wrapped, 0.0) == pytest.approx([0, 0, 0, 1.0, 0])
    x = logits_for([0.01, 0.01, 0.21, 0.35, 0.01, 0.41])
    x_levels = x[:5].softmax(0)
    q = torch.tensor([0, 0, 0.375, 0.625, 0])
    expected = ((x_levels.cumsum(0) - q.cumsum(0))[:4] ** 2).sum() / 4
    assert torch.allclose(ordinal_loss(x, wrapped), expected, atol=1e-6)
    # target_distribution rows (rating histograms) use the histogram restricted to the levels
    hist = ordinal(target=3, target_distribution={'1': 0.1, '3': 0.6, '5': 0.3})
    _, _, _, t = render(hist)
    assert ordinal_level_target(OrdinalTarget(t, 5)) == pytest.approx([0.1, 0, 0.6, 0, 0.3])


def test_gradient_flows_to_level_logits_only():
    x = torch.randn(6, generator=torch.Generator().manual_seed(4), requires_grad=True)
    ordinal_loss(x, OrdinalTarget(0, 5)).backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:5].abs().sum() > 0
    assert float(x.grad[5]) == 0.0  # unknown is not on the scale
    # descending the gradient moves mass toward the gold end
    with torch.no_grad():
        y = x - 0.5 * x.grad
    assert float(ordinal_loss(y, OrdinalTarget(0, 5))) < float(ordinal_loss(x.detach(), OrdinalTarget(0, 5)))


def test_bf16_logits_are_computed_in_float32():
    x = torch.tensor([20.0, -20.0, -20.0, 5.0], dtype=torch.bfloat16, requires_grad=True)
    loss = ordinal_loss(x, OrdinalTarget(2, 3))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad.float()).all()
    big = torch.tensor([1e4, -1e4, 0.0, 0.0])
    assert torch.isfinite(ordinal_loss(big, OrdinalTarget(1, 3)))


def test_level_order_survives_rendering_permutation_and_noise():
    """Levels stay ascending at indices 0..n-1 and unknown last under render's shuffle, --permute-options and
    noise augmentation, so OrdinalTarget(target, len(choices) - 1) always points at the right bins."""
    record = ordinal(target=4)
    for epoch in range(6):
        rng = random.Random(f'0:{epoch}:7')
        for permute in (False, True):
            r = with_noise_state(record, rng)
            if permute:
                r = permute_options(r, 0, epoch)
            _, choices, _, t = render(r, rng, shuffle=not permute)
            assert [c[0] for c in choices] == [1, 2, 3, 4, 5, UNKNOWN]
            assert t == 3 and choices[t][0] == 4
    # choice fields are the ones that move; the trainer never wraps them
    orders = {tuple(c[0] for c in render(permute_options(choice(), 0, e), random.Random(e), shuffle=False)[1]) for e in range(6)}
    assert len(orders) > 1


def test_mismatched_candidate_count_is_an_error():
    with pytest.raises(ValueError):
        ordinal_loss(torch.zeros(5), OrdinalTarget(1, 5))
