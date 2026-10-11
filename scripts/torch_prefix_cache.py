"""Torch shared-prefix/KV scorer for the torch serving path.

For a physical batch of unique questions over identical evidence it:
1. prepares each prompt without running the model;
2. finds the longest token-identical prefix, including all visual placeholders;
3. runs multimodal embedding + the language-model prefix exactly once;
4. expands the resulting transformer KV cache to the suffix batch;
5. runs only the unique text suffixes in one padded batch;
6. reads each decision at its own final non-padding position.

PrefixScorer is a plain helper owned by TorchBackend (no monkey-patching): the
backend calls self._prefix_scorer.score(...) from _score_batched once the
parity gate has validated each evidence mode (text vs visual) against the
serial scorer.
"""

from __future__ import annotations

import copy
from time import perf_counter

import torch

PARITY_ATOL = 0.02

class PrefixUnsuitable(Exception):
    """Batch cannot use prefix reuse (no shared prefix, no validatable pair,
    degenerate model output). Fall back WITHOUT disabling the path."""


class ParityError(Exception):
    """Prefix path produced wrong logits. Disables the path."""


def _unpadded_ids(inputs):
    ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    if ids.shape[0] != 1:
        raise ValueError("prefix scorer expects individually prepared prompts")
    if mask is None:
        return ids[0]
    keep = mask[0].to(dtype=torch.bool)
    return ids[0][keep]


def _longest_common_prefix(rows):
    if len(rows) < 2:
        return 0
    limit = min(int(row.numel()) for row in rows)
    if limit < 2:
        return 0
    first = rows[0][:limit]
    same = torch.ones(limit, dtype=torch.bool, device=first.device)
    for row in rows[1:]:
        same &= row[:limit].eq(first)
    mismatch = (~same).nonzero(as_tuple=False)
    shared = int(mismatch[0].item()) if mismatch.numel() else limit
    # Every branch must retain at least one suffix token for a decision state.
    return min(shared, limit - 1)


def _repeat_cache(cache, batch_size):
    """Clone one-prefix transformers cache and expand it to batch_size."""
    branch = copy.deepcopy(cache)
    repeat = getattr(branch, "batch_repeat_interleave", None)
    if callable(repeat):
        try:
            repeat(batch_size)
        except AttributeError:
            # Hybrid Qwen3.5 stack: full-attention layers repeat, linear-attention
            # conv/recurrent states repeat per layer instead.
            for layer in getattr(branch, "layers", []):
                entry = getattr(layer, "batch_repeat_interleave", None)
                if callable(entry):
                    try:
                        entry(batch_size)
                    except AttributeError:
                        _repeat_linear_layer(layer, batch_size)
                else:
                    _repeat_linear_layer(layer, batch_size)
        return branch

    # Compatibility with legacy tuple/list past_key_values.
    if isinstance(branch, (tuple, list)):
        layers = []
        for layer in branch:
            layers.append(
                tuple(
                    value.expand(batch_size, *value.shape[1:])
                    for value in layer
                )
            )
        return type(branch)(layers)
    raise TypeError(
        f"unsupported transformers cache type: {type(branch).__name__}"
    )


def _repeat_linear_layer(layer, batch_size):
    """Expand one linear-attention layer's conv/recurrent states to batch_size."""
    for name in ("conv_states", "recurrent_states"):
        states = getattr(layer, name, None)
        if not states:
            continue
        for key, value in list(states.items()):
            if value is not None and value.shape[0] == 1 and batch_size > 1:
                states[key] = value.repeat_interleave(batch_size, dim=0)



def _visual_token_ids(engine):
    config = engine._base().config
    ids = set()
    for name in (
        "image_token_id",
        "video_token_id",
        "image_token_index",
        "video_token_index",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int):
            ids.add(value)
    model_config = getattr(engine._base().model, "config", None)
    for name in (
        "image_token_id",
        "video_token_id",
        "image_token_index",
        "video_token_index",
    ):
        value = getattr(model_config, name, None)
        if isinstance(value, int):
            ids.add(value)
    return ids


def _suffix_positions(prefix_positions, lengths, max_suffix, device):
    """Qwen M-RoPE text after a completed multimodal prefix advances uniformly."""
    last = prefix_positions[:, :, -1:].to(device)
    steps = torch.arange(
        1,
        max_suffix + 1,
        device=device,
        dtype=last.dtype,
    ).view(1, 1, -1)
    positions = last + steps
    return positions.expand(-1, len(lengths), -1).contiguous()


def score_rendered_prefix_cached(
    engine,
    images,
    rendered_examples,
    *,
    fast=False,
    microbatch=8,
):
    """Return candidate-logit tensors for rendered (prompt, labels) examples.

    rendered_examples is a list of (rendered_prompt, labels). The images must be
    identical for every example. This function performs no answer/result cache.
    """
    if len(rendered_examples) < 2:
        raise ValueError("prefix reuse requires at least two prompts")

    prepared = []
    rows = []
    for rendered, labels in rendered_examples:
        if fast:
            token_ids = engine.label_ids(rendered, labels)
            image_list = (
                []
                if images is None
                else images
                if isinstance(images, list)
                else [images]
            )
            inputs = dict(
                engine.processor(
                    text=[rendered],
                    images=image_list or None,
                    return_tensors="pt",
                    do_rescale=False,
                    do_normalize=False,
                )
            )
        else:
            image_list = (
                []
                if images is None
                else images
                if isinstance(images, list)
                else [images]
            )
            token_ids = engine.label_ids(rendered, labels)
            inputs = engine.processor(
                text=[rendered],
                images=image_list or None,
                return_tensors="pt",
            )
        prepared.append((inputs, token_ids))
        rows.append(_unpadded_ids(inputs))

    shared = _longest_common_prefix(rows)
    if shared < 1:
        raise PrefixUnsuitable("prompts have no reusable token prefix")

    visual_ids = _visual_token_ids(engine)
    if visual_ids:
        for row in rows:
            suffix = row[shared:]
            if any(bool(suffix.eq(token).any()) for token in visual_ids):
                raise ValueError(
                    "visual placeholders escaped the common prefix"
                )

    device = engine.device
    first_inputs = {
        key: value.to(device)
        for key, value in prepared[0][0].items()
    }
    if (
        fast
        and first_inputs.get("pixel_values") is not None
        and first_inputs["pixel_values"].dtype == torch.uint8
    ):
        first_inputs["pixel_values"] = engine.normalize_patches(
            first_inputs["pixel_values"]
        )

    # The expensive vision tower is called only here, for the first prompt.
    full_embeds, full_positions = engine._embeds_positions(first_inputs)
    prefix_embeds = full_embeds[:, :shared]
    prefix_positions = full_positions[:, :, :shared]

    base = engine._base()
    language = base.model.language_model
    with torch.inference_mode():
        prefix_out = language(
            inputs_embeds=prefix_embeds,
            position_ids=prefix_positions,
            use_cache=True,
            return_dict=True,
        )
    cache = getattr(prefix_out, "past_key_values", None)
    if cache is None:
        raise RuntimeError("language model did not return past_key_values")

    suffix_ids = [row[shared:].to(device) for row in rows]
    lengths = [int(row.numel()) for row in suffix_ids]
    if min(lengths) < 1:
        raise ValueError("every prompt must retain a non-empty suffix")
    if isinstance(microbatch, bool) or not isinstance(microbatch, int):
        raise ValueError("microbatch must be an integer")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")

    embed_tokens = base.model.get_input_embeddings()
    head = base.lm_head.weight
    results = []
    suffix_batches = 0
    for batch_start in range(0, len(rows), microbatch):
        batch_ids = suffix_ids[batch_start : batch_start + microbatch]
        batch_lengths = lengths[batch_start : batch_start + microbatch]
        max_suffix = max(batch_lengths)
        batch_size = len(batch_ids)
        hidden_size = int(prefix_embeds.shape[-1])
        suffix_embeds = torch.zeros(
            (batch_size, max_suffix, hidden_size),
            device=device,
            dtype=prefix_embeds.dtype,
        )
        suffix_mask = torch.zeros(
            (batch_size, max_suffix),
            device=device,
            dtype=torch.long,
        )
        for local_index, ids in enumerate(batch_ids):
            count = batch_lengths[local_index]
            suffix_embeds[local_index, :count] = embed_tokens(ids)
            suffix_mask[local_index, :count] = 1

        attention_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, shared),
                    device=device,
                    dtype=torch.long,
                ),
                suffix_mask,
            ],
            dim=1,
        )
        position_ids = _suffix_positions(
            prefix_positions,
            batch_lengths,
            max_suffix,
            device,
        )
        branch_cache = _repeat_cache(cache, batch_size)

        with torch.inference_mode():
            output = language(
                inputs_embeds=suffix_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=branch_cache,
                use_cache=False,
                return_dict=True,
            )

        states = output.last_hidden_state.float()
        for local_index, global_index in enumerate(
            range(batch_start, batch_start + batch_size)
        ):
            token_ids = prepared[global_index][1]
            hidden = _read_hidden(
                states, local_index, batch_lengths[local_index] - 1
            )
            indices = engine._readout_indices(token_ids)
            if engine.readout is not None and indices is not None:
                logits = engine.readout(hidden)[indices]
            else:
                rows_tensor = torch.tensor(token_ids, device=device)
                logits = hidden @ head[rows_tensor].float().T
            results.append(logits)
        suffix_batches += 1

    return results, {
        "shared_prefix_tokens": shared,
        "suffix_tokens": lengths,
        "vision_forwards": 1 if first_inputs.get("pixel_values") is not None else 0,
        "prefix_prefills": 1,
        "suffix_batches": suffix_batches,
        "microbatch": microbatch,
        "prefix_cache": True,
    }



def _prepare_rendered_example(engine, images, rendered, labels, *, fast=False):
    image_list = (
        []
        if images is None
        else images
        if isinstance(images, list)
        else [images]
    )
    token_ids = engine.label_ids(rendered, labels)
    if fast:
        inputs = dict(
            engine.processor(
                text=[rendered],
                images=image_list or None,
                return_tensors="pt",
                do_rescale=False,
                do_normalize=False,
            )
        )
    else:
        inputs = engine.processor(
            text=[rendered],
            images=image_list or None,
            return_tensors="pt",
        )
    return inputs, token_ids, _unpadded_ids(inputs)


def score_rendered_prefix_cached_hierarchical(
    engine,
    images,
    rendered_groups,
    *,
    fast=False,
    microbatch=8,
):
    """Reuse one global multimodal prefix, then one prefix per rotation group.

    Each group is normally one question containing all of that question's
    candidate-order rotations. The expensive image/evidence prefix is evaluated
    once for the whole request; the question-specific header is evaluated once
    per question; only the reordered candidate suffix is evaluated per rotation.
    """
    prepared_groups = []
    rows = []
    for group in rendered_groups:
        prepared = [
            _prepare_rendered_example(
                engine,
                images,
                rendered,
                labels,
                fast=fast,
            )
            for rendered, labels in group
        ]
        prepared_groups.append(prepared)
        rows.extend(item[2] for item in prepared)

    if len(rows) < 2:
        raise ValueError("hierarchical prefix reuse requires at least two prompts")

    shared = _longest_common_prefix(rows)
    if shared < 1:
        raise PrefixUnsuitable("prompts have no reusable global token prefix")

    visual_ids = _visual_token_ids(engine)
    if visual_ids:
        for row in rows:
            suffix = row[shared:]
            if any(bool(suffix.eq(token).any()) for token in visual_ids):
                raise ValueError("visual placeholders escaped the global prefix")

    device = engine.device
    first_inputs = {
        key: value.to(device)
        for key, value in prepared_groups[0][0][0].items()
    }
    if (
        fast
        and first_inputs.get("pixel_values") is not None
        and first_inputs["pixel_values"].dtype == torch.uint8
    ):
        first_inputs["pixel_values"] = engine.normalize_patches(
            first_inputs["pixel_values"]
        )

    full_embeds, full_positions = engine._embeds_positions(first_inputs)
    prefix_embeds = full_embeds[:, :shared]
    prefix_positions = full_positions[:, :, :shared]

    base = engine._base()
    language = base.model.language_model
    embed_tokens = base.model.get_input_embeddings()
    head = base.lm_head.weight
    with torch.inference_mode():
        prefix_out = language(
            inputs_embeds=prefix_embeds,
            position_ids=prefix_positions,
            use_cache=True,
            return_dict=True,
        )
    global_cache = getattr(prefix_out, "past_key_values", None)
    if global_cache is None:
        raise RuntimeError("language model did not return past_key_values")

    logits_out = []
    question_prefix_tokens = []
    suffix_batches = 0
    max_effective_tokens = shared
    question_prefills = 0

    for prepared in prepared_groups:
        group_rows = [item[2] for item in prepared]
        relative = [row[shared:] for row in group_rows]
        question_shared = _longest_common_prefix(relative)
        question_prefix_tokens.append(question_shared)

        branch_cache = global_cache
        branch_positions = prefix_positions
        branch_length = shared

        if question_shared > 0:
            segment_ids = group_rows[0][
                shared : shared + question_shared
            ].to(device)
            segment_embeds = embed_tokens(segment_ids).unsqueeze(0)
            segment_positions = _suffix_positions(
                prefix_positions,
                [question_shared],
                question_shared,
                device,
            )
            attention_mask = torch.ones(
                (1, shared + question_shared),
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                question_out = language(
                    inputs_embeds=segment_embeds,
                    position_ids=segment_positions,
                    attention_mask=attention_mask,
                    past_key_values=_repeat_cache(global_cache, 1),
                    use_cache=True,
                    return_dict=True,
                )
            branch_cache = getattr(question_out, "past_key_values", None)
            if branch_cache is None:
                raise RuntimeError(
                    "question-prefix pass did not return past_key_values"
                )
            branch_positions = segment_positions
            branch_length += question_shared
            question_prefills += 1

        suffix_ids = [
            row[shared + question_shared :].to(device)
            for row in group_rows
        ]
        suffix_lengths = [int(row.numel()) for row in suffix_ids]
        if not suffix_lengths or min(suffix_lengths) < 1:
            raise ValueError("every rotation must retain a non-empty suffix")

        for batch_start in range(0, len(suffix_ids), microbatch):
            batch_ids = suffix_ids[batch_start : batch_start + microbatch]
            batch_prepared = prepared[
                batch_start : batch_start + microbatch
            ]
            batch_lengths = suffix_lengths[
                batch_start : batch_start + microbatch
            ]
            batch_size = len(batch_ids)
            max_suffix = max(batch_lengths)
            hidden_size = int(prefix_embeds.shape[-1])

            suffix_embeds = torch.zeros(
                (batch_size, max_suffix, hidden_size),
                device=device,
                dtype=prefix_embeds.dtype,
            )
            suffix_mask = torch.zeros(
                (batch_size, max_suffix),
                device=device,
                dtype=torch.long,
            )
            for local_index, ids in enumerate(batch_ids):
                count = batch_lengths[local_index]
                suffix_embeds[local_index, :count] = embed_tokens(ids)
                suffix_mask[local_index, :count] = 1

            attention_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, branch_length),
                        device=device,
                        dtype=torch.long,
                    ),
                    suffix_mask,
                ],
                dim=1,
            )
            position_ids = _suffix_positions(
                branch_positions,
                batch_lengths,
                max_suffix,
                device,
            )
            with torch.inference_mode():
                output = language(
                    inputs_embeds=suffix_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=_repeat_cache(
                        branch_cache,
                        batch_size,
                    ),
                    use_cache=False,
                    return_dict=True,
                )

            states = output.last_hidden_state.float()
            for local_index, item in enumerate(batch_prepared):
                token_ids = item[1]
                hidden = _read_hidden(
                    states, local_index, batch_lengths[local_index] - 1
                )
                indices = engine._readout_indices(token_ids)
                if engine.readout is not None and indices is not None:
                    logits = engine.readout(hidden)[indices]
                else:
                    rows_tensor = torch.tensor(
                        token_ids,
                        device=device,
                    )
                    logits = hidden @ head[rows_tensor].float().T
                logits_out.append(logits)
            suffix_batches += 1
            max_effective_tokens = max(
                max_effective_tokens,
                branch_length + max_suffix,
            )

    return logits_out, {
        "shared_prefix_tokens": shared,
        "question_shared_prefix_tokens": question_prefix_tokens,
        "vision_forwards": (
            1 if first_inputs.get("pixel_values") is not None else 0
        ),
        "prefix_prefills": 1 + question_prefills,
        "question_prefix_prefills": question_prefills,
        "suffix_batches": suffix_batches,
        "microbatch": microbatch,
        "prefix_cache": True,
        "max_effective_tokens": max_effective_tokens,
    }


def _select_probe_pair(compiled):
    """Indices of two same-type questions, or None if no validatable pair.

    Cross-type pairs (e.g. noul + choice) share no prompt prefix, so probing
    them proves nothing; the batch falls back and the path stays enabled."""
    by_type = {}
    for i, row in enumerate(compiled):
        by_type.setdefault(getattr(row[0], "type", None), []).append(i)
    for idx in by_type.values():
        if len(idx) >= 2:
            return idx[0], idx[1]
    return None


def _read_hidden(states, local_index, pos):
    """Decision hidden state with a loud failure on degenerate model output.

    A 0D `states` (collapsed batch dim upstream) cannot be indexed at all, so
    guard before indexing; raise PrefixUnsuitable so the gate falls back
    cleanly instead of exploding inside F.linear."""
    if states.dim() == 0:
        raise PrefixUnsuitable("degenerate 0D model output; falling back")
    hidden = states[local_index, pos]
    if hidden.dim() < 1:
        raise PrefixUnsuitable(
            f"degenerate hidden state dim={hidden.dim()}; falling back"
        )
    return hidden


def _logits_match(candidate, reference, *, atol=PARITY_ATOL):
    if len(candidate) != len(reference):
        return False, float("inf")
    max_delta = 0.0
    for left, right in zip(candidate, reference):
        if left.shape != right.shape:
            return False, float("inf")
        delta = float((left.float() - right.float()).abs().max().item())
        max_delta = max(max_delta, delta)
        if int(left.argmax().item()) != int(right.argmax().item()):
            return False, max_delta
    return max_delta <= atol, max_delta


class PrefixScorer:
    """Shared-prefix KV scorer owned by TorchBackend.

    The backend renders (prompt, labels) groups and calls score(); the serial
    scorer stays the fallback. The first eligible batch per evidence mode
    (text vs visual) is double-scored against the serial path; the prefix path
    enables only on identical argmax AND max abs logit delta <= PARITY_ATOL.
    """

    def __init__(self, engine, *, fast=False, microbatch=8):
        self.engine = engine
        self.fast = bool(fast)
        self.microbatch = max(1, int(microbatch))
        self.enabled = True
        self.validated_text = False
        self.validated_visual = False
        self.error = None
        self.max_delta = {}
        self.metadata = None

    @property
    def validated(self):
        return bool(self.validated_text or self.validated_visual)

    def render_groups(self, images, compiled, rotations):
        from vision_decision.scoring import cyclic_offsets, rotate

        groups, owners = [], []
        for qi, (_, header, choices, texts, labels) in enumerate(compiled):
            group = []
            for offset in cyclic_offsets(len(choices), rotations):
                prompt = header + "\n".join(
                    f"{label}: {text}" for label, text in zip(labels, rotate(texts, offset)))
                group.append((self.engine.render(prompt, len(images)), labels))
                owners.append((qi, offset))
            groups.append(group)
        return groups, owners

    def reference_logits(self, images, examples):
        engine = self.engine
        out = []
        for rendered, labels in examples:
            image_list = [] if images is None else images if isinstance(images, list) else [images]
            token_ids = engine.label_ids(rendered, labels)
            if self.fast:
                inputs = dict(engine.processor(text=[rendered], images=image_list or None,
                                               return_tensors="pt", do_rescale=False, do_normalize=False))
                with torch.inference_mode():
                    out.append(engine.candidate_logits_batch_fast(inputs, [token_ids])[0])
            else:
                inputs = engine.processor(text=[rendered], images=image_list or None, return_tensors="pt")
                with torch.inference_mode():
                    out.append(engine.candidate_logits_batch(inputs, [token_ids])[0])
        return out

    def score(self, images, compiled, rotations):
        """-> (per-question [(offset, logits)] lists, metadata)."""
        from vision_decision.scoring import combine_rotations, result_from_logits

        start = perf_counter()
        groups, owners = self.render_groups(images, compiled, rotations)
        logits, metadata = score_rendered_prefix_cached_hierarchical(
            self.engine, images, groups, fast=self.fast, microbatch=self.microbatch)
        per_q = [[] for _ in compiled]
        for (qi, offset), tensor in zip(owners, logits):
            per_q[qi].append((offset, [float(v) for v in tensor.cpu().tolist()]))
        self.metadata = {**metadata, "enabled": True,
                         "validated": True, "logical_questions": len(compiled), "rotated_suffixes": len(logits)}
        self.engine._last_batch_tokens = metadata["max_effective_tokens"]
        out = []
        for qi, (_, _, choices, _, _) in enumerate(compiled):
            passes = per_q[qi]
            out.append(result_from_logits(choices, passes[0][1]) if len(passes) == 1
                       else combine_rotations(choices, [(o, v) for o, v in passes]))
        self.engine._batch_seconds = perf_counter() - start
        return out

    def maybe_validate_and_score(self, images, compiled, rotations, fallback):
        """Prefix path with parity gate; falls back to `fallback` on any doubt.

        Unsuitable batches (no shared prefix, no validatable same-type pair,
        degenerate model output) fall back WITHOUT disabling the path; only
        genuine parity mismatches and unexpected errors disable it."""
        if not self.enabled or len(compiled) < 2:
            return fallback(images, compiled)
        mode = "visual" if images else "text"
        try:
            if (mode == "visual" and not self.validated_visual) or (mode == "text" and not self.validated_text):
                # Validate the hierarchical branch itself (rotations included); a
                # flat-prefix check would not prove the per-question KV branch.
                # Cross-type pairs share no prefix, so only same-type pairs prove
                # the mechanics; without one, fall back and stay enabled.
                pair = _select_probe_pair(compiled)
                if pair is None:
                    return fallback(images, compiled)
                first, second = pair
                groups, _ = self.render_groups(images, [compiled[first], compiled[second]], rotations)
                probe, probe_examples = groups, [item for group in groups for item in group]
                candidate, _ = score_rendered_prefix_cached_hierarchical(
                    self.engine, images, probe, fast=self.fast, microbatch=self.microbatch)
                matches, delta = _logits_match(candidate, self.reference_logits(images, probe_examples))
                self.max_delta[mode] = delta
                if not matches:
                    raise ParityError(f"parity mismatch: max_delta={delta:.6f} tolerance={PARITY_ATOL:.6f}")
                if mode == "visual":
                    self.validated_visual = True
                else:
                    self.validated_text = True
            out = self.score(images, compiled, rotations)
            self.metadata["validated_mode"] = mode
            self.metadata["parity_max_delta"] = self.max_delta.get(mode)
            return out
        except PrefixUnsuitable:
            return fallback(images, compiled)
        except Exception as exc:
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"
            return fallback(images, compiled)

