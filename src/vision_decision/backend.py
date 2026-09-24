"""MLX direct-choice prototype; one unpadded request per forward pass."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from .scoring import compile_question, readout_codes, verified_label_ids, result_from_logits, cyclic_offsets, rotate, combine_rotations

class MLXDirect:
    def __init__(self, bundle="artifacts/model.json", adapter=None, max_input_tokens=4096):
        import mlx.core as mx
        self.max_input_tokens = int(max_input_tokens)
        from mlx_vlm import load
        self.mx = mx
        self.bundle = json.loads(Path(bundle).read_text())
        if not Path(self.bundle["path"]).is_dir():
            raise ValueError("Local model snapshot is missing; run scripts/download_model.py")
        start = perf_counter()
        self.model, self.processor = load(self.bundle["path"], trust_remote_code=False)
        if adapter is not None:
            from mlx_vlm.trainer.utils import apply_lora_layers
            self.model = apply_lora_layers(self.model, adapter)
        self.adapter = None if adapter is None else str(adapter)
        self._codebook = None
        self.readout = None
        if adapter is not None:
            head = Path(adapter) / "decision_readout.safetensors"
            if head.exists():
                weights = mx.load(str(head))
                if (set(weights) != {"weight"} or weights["weight"].ndim != 2 or weights["weight"].shape[0] != 255
                        or not bool(mx.all(mx.isfinite(weights["weight"])).item())):
                    raise ValueError("Invalid decision_readout.safetensors")
                self.readout = weights["weight"]
                manifest_path = Path(adapter) / "decision_readout.json"
                if not manifest_path.exists():
                    raise ValueError("Trained readout is missing decision_readout.json tokenizer binding")
                self._readout_manifest = json.loads(manifest_path.read_text())
        self.model.eval()
        mx.eval(self.model.parameters())
        self.load_seconds = perf_counter() - start

    def score(self, image, field, state):
        header, choices, texts = compile_question(field, state)
        labels = self._labels(header, len(choices), 0 if image is None else len(image) if isinstance(image, list) else 1)
        prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
        return self.score_compiled(image, prompt, labels, choices)

    def _labels(self, prompt, count, n_images):
        if self._codebook is None:
            from mlx_vlm.prompt_utils import apply_chat_template
            rendered = apply_chat_template(self.processor, self.model.config, prompt, num_images=n_images, enable_thinking=False)
            self._codebook = readout_codes(self.processor.tokenizer, rendered, 255)
            if self.readout is not None:
                actual = [{"code": c, "token_id": i} for c, i in self._codebook]
                if self._readout_manifest.get("codes") != actual:
                    raise ValueError("Decision readout code/token binding does not match this tokenizer")
        return [code for code, _ in self._codebook[:count]]

    def score_request(self, image, fields, state, rotations=4):
        """All fields of one request: one shared prefill, rotated candidate orders per field.

        `image` is one image, a list of one or two, or None/[] for a text-only request.
        """
        return self.score_questions(image, [compile_question(field, state) for field in fields], rotations)

    def score_compiled(self, image, prompt, labels, choices):
        """Research hook for controlled labels and one- or two-image diagnostics."""
        if len(labels) != len(choices) or not labels:
            raise ValueError("Labels and choices must have matching nonzero length")
        logits, metadata, _ = self._label_logits(image, prompt, labels)
        metadata["tie_break_policy"] = "lowest_vocabulary_token_id"
        return result_from_logits(choices, logits, token_ids=metadata["choice_token_ids"]), metadata

    def score_rotated(self, image, header, choices, texts, rotations=4, shared_prefix=True):
        """Average candidate log-probabilities over cyclic presentation orders of one question."""
        results, metadata = self.score_questions(image, [(header, choices, texts)], rotations, shared_prefix)
        return results[0], metadata["questions"][0] | {k: v for k, v in metadata.items() if k != "questions"}

    def score_questions(self, image, questions, rotations=4, shared_prefix=True):
        """Score every (header, choices, texts) question for one image, or for no image at all.

        Each question is shown in up to `rotations` cyclic candidate orders and its
        per-candidate log-probabilities are averaged. With shared_prefix the image (when there
        is one) and the leading text common to all prompts are evaluated once for the whole
        request; a text-only request shares the state and question header the same way.
        """
        prompts, owners = [], []
        for index, (header, choices, texts) in enumerate(questions):
            if len(texts) != len(choices) or len(choices) < 2:
                raise ValueError("Texts and choices must have matching length of at least two")
            n_images = 0 if image is None else len(image) if isinstance(image, list) else 1
            labels = self._labels(header, len(choices), n_images)
            for offset in cyclic_offsets(len(choices), rotations):
                lines = "\n".join(f"{label}: {text}" for label, text in zip(labels, rotate(texts, offset)))
                prompts.append((header + lines, labels))
                owners.append((index, offset))
        if shared_prefix:
            scored, shared = self._shared_prefix_logits(image, prompts)
        else:
            scored, shared, features = [], {"shared_prefix_tokens": 0}, None
            for prompt, labels in prompts:
                logits, metadata, features = self._label_logits(image, prompt, labels, features)
                scored.append((logits, metadata))
        results, details = [], []
        for index, (_, choices, _) in enumerate(questions):
            passes = [(offset, *scored[k]) for k, (owner, offset) in enumerate(owners) if owner == index]
            result = combine_rotations(choices, [(offset, logits) for offset, logits, _ in passes])
            values = list(result.raw_logits.values())
            final = max(range(len(choices)), key=values.__getitem__)
            winners = [(max(range(len(logits)), key=logits.__getitem__) + offset) % len(choices) for offset, logits, _ in passes]
            results.append(result)
            details.append({"rotations": [{"offset": offset, "logits": logits, **meta} for offset, logits, meta in passes],
                            "rotation_agreement": sum(w == final for w in winners) / len(winners)})
        return results, {"questions": details, "language_model_passes": len(prompts),
                         "logit_precision": "float32_candidate_head", **shared}

    def _candidate_logits(self, hidden, token_ids, readout_indices=None):
        # The BF16 head quantizes logits near 20 to 1/16 steps and creates exact ties;
        # project only the candidate rows in float32 instead.
        if readout_indices is not None and hasattr(self.model, "decision_readout"):
            return self.model.decision_readout(hidden.astype(self.mx.float32))[self.mx.array(readout_indices)]
        if readout_indices is not None and self.readout is not None:
            return hidden.astype(self.mx.float32) @ self.readout[self.mx.array(readout_indices)].astype(self.mx.float32).T
        language = self.model.language_model
        head = language.model.embed_tokens if language.args.tie_word_embeddings else language.lm_head
        if "scales" in head or "bias" in head:
            raise ValueError("Float32 candidate scoring requires an unquantized bias-free output head")
        mx = self.mx
        return hidden.astype(mx.float32) @ head.weight[mx.array(token_ids)].astype(mx.float32).T

    def _prepare(self, image, prompt, labels):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs
        mx = self.mx
        start = perf_counter()
        images = [] if image is None else (image if isinstance(image, list) else [image])
        if len(images) > 2:
            raise ValueError("Research scoring supports at most two images")
        rendered = apply_chat_template(self.processor, self.model.config, prompt, num_images=len(images), enable_thinking=False)
        # Qwen non-thinking template must close its empty reasoning block before the answer.
        if not rendered.endswith("<think>\n\n</think>\n\n"):
            raise ValueError("Unexpected Qwen non-thinking template boundary; refusing to score an unverified position")
        tokenizer = self.processor.tokenizer
        token_ids = verified_label_ids(tokenizer, rendered, labels)
        codebook = self._codebook or readout_codes(tokenizer, rendered, 255)
        self._codebook = codebook
        if self.readout is not None:
            actual = [{"code": c, "token_id": i} for c, i in codebook]
            if self._readout_manifest.get("version") != 1 or self._readout_manifest.get("codes") != actual:
                raise ValueError("Decision readout code/token binding does not match this tokenizer")
        lookup = {code: i for i, (code, _) in enumerate(codebook)}
        readout_indices = [lookup[x] for x in labels] if all(x in lookup for x in labels) else None
        # A text-only request (no image) goes through the same decision position; prepare_inputs
        # then returns ids and mask only, so there is no pixel tensor and no image grid to check.
        inputs = prepare_inputs(self.processor, images=images or None, prompts=rendered,
                                image_token_index=self.model.config.image_token_index)
        ids = inputs.pop("input_ids")
        pixels = inputs.pop("pixel_values", None)
        mask = inputs.pop("attention_mask", None)
        if ids.shape[0] != 1 or (mask is not None and not bool(mx.all(mask == 1).item())):
            raise ValueError("This adapter supports only one unpadded example")
        # Verify the actual processed sequence's answer suffix survived image-token expansion.
        suffix = tokenizer.encode("</think>\n\n", add_special_tokens=False)
        if ids[0, -len(suffix):].tolist() != suffix:
            raise ValueError("Processed decision-position suffix mismatch")
        if ids.shape[-1] > self.max_input_tokens:
            raise ValueError(f"Processed request exceeds the {self.max_input_tokens}-token limit")
        if images:
            if pixels is None or "image_grid_thw" not in inputs:
                raise ValueError("Processor returned no image grid")
            grids = inputs["image_grid_thw"].tolist()
            if len(grids) != len(images):
                raise ValueError("Image grid count mismatch")
            if any(g[0] * g[1] * g[2] // 4 > 2048 for g in grids):
                raise ValueError("Image exceeds the 2048-visual-token research limit")
        else:
            if pixels is not None or inputs:
                raise ValueError("A text-only request must not carry vision inputs")
            grids = []
        metadata = {"preprocess_seconds": perf_counter() - start, "input_tokens": ids.shape[-1],
                    "choice_token_ids": token_ids, "template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "readout_indices": readout_indices, "readout": "trained_255" if self.readout is not None and readout_indices is not None else "language_model_rows",
                    "template_suffix": rendered[-64:], "logit_precision": "float32_candidate_head",
                    "image_grid_thw": grids}
        return ids, pixels, mask, inputs, metadata

    def _image_features(self, pixels, inputs):
        tower = self.model.vision_tower
        features, _ = tower(pixels.astype(tower.patch_embed.proj.weight.dtype), inputs["image_grid_thw"])
        self.mx.eval(features)
        return features

    def _label_logits(self, image, prompt, labels, image_features=None):
        """Uncached reference path: one full forward per prompt."""
        mx = self.mx
        ids, pixels, mask, inputs, metadata = self._prepare(image, prompt, labels)
        start = perf_counter()
        if image_features is None and pixels is not None:  # no vision tower on a text-only request
            image_features = self._image_features(pixels, inputs)
        output = self.model(ids, pixels, mask=mask, cached_image_features=image_features,
                            return_hidden=True, skip_logits=True, **inputs)
        selected = self._candidate_logits(output.hidden_states[-1][0, -1], metadata["choice_token_ids"], metadata["readout_indices"])
        mx.eval(selected)
        logits = selected.tolist()
        metadata.update(forward_seconds=perf_counter() - start, peak_memory_bytes=mx.get_peak_memory(),
                        logical_forward_evaluations=1)
        del output, selected
        return logits, metadata, image_features

    def _shared_prefix_logits(self, image, prompts):
        """Score several (prompt, labels) pairs for one image with a single prefill.

        The image (when there is one) and every token the prompts share are evaluated once;
        each prompt then continues from a private copy of that cache with only its own
        remaining tokens.
        """
        mx = self.mx
        prepared = [self._prepare(image, prompt, labels) for prompt, labels in prompts]
        rows = [p[0][0].tolist() for p in prepared]
        shared = min(len(r) for r in rows) - 1  # every prompt keeps at least its decision position
        for position in range(shared):
            if len({r[position] for r in rows}) != 1:
                shared = position
                break
        config = self.model.config
        visual = {config.image_token_index, config.video_token_index}
        if shared < 1 or any(token in visual for r in rows for token in r[shared:]):
            raise ValueError("Prompts must share the image and a leading text segment")
        ids, pixels, mask, inputs, _ = prepared[0]
        start = perf_counter()
        language = self.model.language_model
        prefix = language.make_cache()
        features = self._image_features(pixels, inputs) if pixels is not None else None
        self.model(ids[:, :shared], pixels, cache=prefix, cached_image_features=features,
                   skip_logits=True, **inputs)
        mx.eval([x for c in prefix for x in c.state if x is not None])
        prefill_seconds = perf_counter() - start
        scored = []
        for (ids, _, _, inputs, metadata) in prepared:
            start = perf_counter()
            branch = language.make_cache()
            for source, target in zip(prefix, branch):
                state = source.state
                target.state = list(state) if isinstance(state, list) else state
            positions, _ = language.get_rope_index(ids, inputs.get("image_grid_thw"))
            output = language(ids[:, shared:], cache=branch, position_ids=positions, return_hidden=True, skip_logits=True)
            selected = self._candidate_logits(output.hidden_states[-1][0, -1], metadata["choice_token_ids"], metadata["readout_indices"])
            mx.eval(selected)
            metadata.update(forward_seconds=perf_counter() - start, suffix_tokens=ids.shape[-1] - shared)
            scored.append((selected.tolist(), metadata))
            del output, selected, branch
        shared_metadata = {"shared_prefix_tokens": shared, "prefill_seconds": prefill_seconds,
                           "peak_memory_bytes": mx.get_peak_memory()}
        return scored, shared_metadata

    def generated_reference(self, image, field, state):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        images = [] if image is None else (image if isinstance(image, list) else [image])
        header, choices, texts = compile_question(field, state)
        labels = self._labels(header, len(choices), len(images))
        prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
        rendered = apply_chat_template(self.processor, self.model.config, prompt, num_images=len(images), enable_thinking=False)
        start = perf_counter()
        output = generate(self.model, self.processor, rendered, image=images or None, max_tokens=8, temperature=0, verbose=False)
        answer = output.text.strip()
        valid = answer in labels
        return {"text": answer, "format_valid": valid,
                "value": choices[labels.index(answer)][0] if valid else None,
                "seconds": perf_counter() - start}
