"""Opt-in v1.1 recipe pieces for train_decision_lora_torch.py: soft-target loss and the rationale auxiliary LM loss.

Kept out of the trainer (which runs on import) so they can be unit-tested on CPU with tiny tensors.
With every flag off the trainer's numbers are unchanged: decision_loss reproduces the original
hard / target_distribution expressions exactly, and nothing here is called for the rationale path.
"""
import torch
from decision_data import SoftTarget, rationale_text

IGNORE = -100


def gold(t):
    """Index dev accuracy scores against: the hard target, the SoftTarget's gold (the record's `target`
    when present, else the argmax of `target_probs`), or the mode of a `target_distribution` list."""
    if isinstance(t, SoftTarget):
        return t.gold
    return max(range(len(t)), key=t.__getitem__) if isinstance(t, list) else t


def soft_cross_entropy(x, probs):
    """-sum p log softmax(x). Equals KL(p || softmax x) + H(p): same gradient as KL, and equal to hard CE for one-hot p."""
    return -(torch.tensor(probs, device=x.device, dtype=x.dtype) * x.log_softmax(0)).sum()


def decision_loss(x, t, soft_weight=1.0):
    """Loss of one example's candidate logits x (1-D, float32).

    int -> cross-entropy (unchanged); list (`target_distribution`) -> soft CE (unchanged);
    SoftTarget (`target_probs` under --soft-targets) -> soft_weight * soft CE + (1 - soft_weight) * hard CE on gold."""
    if isinstance(t, SoftTarget):
        soft = soft_cross_entropy(x, t.probs)
        if soft_weight >= 1:
            return soft
        hard = torch.nn.functional.cross_entropy(x[None], torch.tensor([t.gold], device=x.device))[None].squeeze()
        return soft_weight * soft + (1 - soft_weight) * hard
    if isinstance(t, list):
        return soft_cross_entropy(x, t)
    return torch.nn.functional.cross_entropy(x[None], torch.tensor([t], device=x.device))[None].squeeze()


def rationale_ids(tokenizer, record, cap):
    """Token ids of ' Because: <rationale>' capped at `cap`, or [] when the record has no rationale."""
    text = rationale_text(record)
    return tokenizer.encode(text, add_special_tokens=False)[:cap] if text and cap > 0 else []


def append_rationale(inputs, rationales, pad_id):
    """Append a right-padded rationale block after the (left-padded) decision prompts.

    Every prompt row ends at the decision position P-1 (left padding), so after appending the decision
    position is still P-1 for every row; causal attention means its hidden state cannot see the
    rationale. Returns (inputs, decision_position, labels[B, R]) where labels hold the rationale ids
    with IGNORE on padding and on column 0: the first rationale token would be predicted from the
    decision position itself, whose loss must stay the readout loss alone.
    With no rationale in the batch, returns the inputs untouched, position -1 and labels None."""
    ids = inputs['input_ids']
    batch, length = ids.shape
    width = max((len(r) for r in rationales), default=0)
    if width == 0:
        return inputs, -1, None
    block = torch.full((batch, width), pad_id, dtype=ids.dtype)
    mask = torch.zeros((batch, width), dtype=ids.dtype)
    labels = torch.full((batch, width), IGNORE, dtype=torch.long)
    for i, r in enumerate(rationales):
        if r:
            block[i, :len(r)] = torch.tensor(r, dtype=ids.dtype)
            mask[i, :len(r)] = 1
            labels[i, 1:len(r)] = torch.tensor(r[1:], dtype=torch.long)
    out = dict(inputs)
    for key, value in inputs.items():
        if not (torch.is_tensor(value) and value.dim() == 2 and tuple(value.shape) == (batch, length)):
            continue  # pixel_values, image_grid_thw, ... are per image, not per token
        if key == 'input_ids':
            extra = block
        elif key == 'attention_mask':
            extra = mask.to(value.dtype)
        else:  # e.g. mm_token_type_ids: rationale tokens are text
            extra = torch.zeros((batch, width), dtype=value.dtype)
        out[key] = torch.cat([value, extra], dim=1)
    return out, length - 1, labels


def collate_with_rationale(collate, examples, pad_id):
    """engine.collate plus the rationale block. examples: (rendered, images, token_ids, target[, rationale_ids]).

    Returns engine.collate's 3-tuple when no example carries a rationale, else a 4-tuple whose last item is
    dict(position=decision position, labels=rationale labels). The padded length (and so the token-budget
    check in the trainer) includes the rationale tokens."""
    inputs, token_ids, targets = collate(examples)
    rationales = [e[4] if len(e) > 4 else [] for e in examples]
    inputs, position, labels = append_rationale(inputs, rationales, pad_id)
    if labels is None:
        return inputs, token_ids, targets
    return inputs, token_ids, targets, dict(position=position, labels=labels)


def rationale_lm_loss(hidden, head_weight, labels, start):
    """Per-example mean next-token cross-entropy over the rationale, shape [B] (0 for rows with no labelled token).

    hidden: [B, T, H] final hidden states; head_weight: [V, H] LM head; labels: [B, R] from append_rationale;
    start: index of the first rationale position (decision position + 1). Label column j (j >= 1) is predicted
    from position start + j - 1; column 0 is IGNORE, so the decision position gets no LM loss. Rows are
    projected one at a time to keep the [R, V] logits small."""
    width = labels.shape[1]
    losses = []
    for i in range(hidden.shape[0]):
        target = labels[i, 1:].to(hidden.device)
        count = int((target != IGNORE).sum())
        if count == 0 or width < 2:
            losses.append(hidden.new_zeros((), dtype=torch.float32))
            continue
        logits = (hidden[i, start:start + width - 1] @ head_weight.T).float()
        losses.append(torch.nn.functional.cross_entropy(logits, target, ignore_index=IGNORE, reduction='sum') / count)
    return torch.stack(losses)
