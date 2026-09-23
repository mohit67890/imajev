"""Label decision-v2 candidates with the calibrated teacher (see docs/decision-v2-pseudolabel-spec.md).

Every candidate decision is scored twice: once in the served order and once with the options
rotated by one position.  Each order's logits are turned into probabilities with the calibrator's
temperature for that (decision type, option count) bucket, realigned onto the served order, and
averaged.

Keep rule (spec): the two orders must agree on the argmax AND the averaged top probability must be
>= 0.6, OR the argmax must be ``unknown`` with probability >= 0.5.  Kept decisions get a soft
``target_distribution`` keyed by option value exactly as ``decision_data.render`` expects; dropped
decisions are written to ``<output>.dropped.jsonl`` with their reason.

Rotation is applied to the *presented candidate lines*, not to ``request.fields[0].options``:
boolean fields have no option list and ordinal levels must stay ascending inside the request, so
presentation order is the only place where every decision type can be rotated.  ``--rotate
options`` (the default) rotates the listed options and leaves ``unknown`` on the last line, which
is where training and serving always put it (``decision_data.render``: "unknown stays last as
served").  ``--rotate candidates`` rotates the whole list, unknown included; that is a much harsher
probe -- rotating by one always promotes unknown to the first line -- and in the 2B smoke it turned
essentially every text decision into an order disagreement.  Both mirror
``vision_decision.scoring.rotate`` / ``combine_rotations``.

Two modes:

- **label** (default): candidates with ``target: null`` are pseudo-labelled by the fine-tuned
  teacher, and decisions that fail the keep rule are dropped.
- **blend** (``--blend ALPHA``): records that are *already* labelled (e.g. the training rows of
  data/manifests/decision-v1.1.jsonl) are regularised toward the model's own distributions --
  self-distillation, to stop the fine-tune erasing the base model's reasoning.  Each field's
  ``target_distribution`` becomes ``ALPHA * model + (1 - ALPHA) * existing target`` (the existing
  target being a one-hot over the listed options plus unknown, or the record's own soft
  distribution when it has one).  ``target``/``targets`` and every other field are untouched, the
  keep rule does not apply, and every input record is written out.  Pair it with ``--adapter none``
  to regularise toward the *base* model.

``--adapter none`` (or omitting it) scores with the base model through the engine's legacy lm_head
readout -- no LoRA, no trained 255-row head.  ``--calibration`` is optional in that mode (T = 1).

Batched (left-padded) scoring needs CUDA; on Apple MPS left padding produces NaN, so --max-batch is
clamped to 1 and the one-at-a-time path is used (docs/full-run-runbook.md).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from vision_decision.calibration import TemperatureCalibrator, softmax
from vision_decision.contracts import UNKNOWN
from vision_decision.scoring import key as candidate_key
from vision_decision.scoring import rotate

from decision_data import approx_tokens, expand_fields, load_image, pixel_budget, render

TOP_THRESHOLD = 0.6
UNKNOWN_THRESHOLD = 0.5
OFFSETS = (0, 1)  # served order, rotated by one
ORDER_NAMES = {0: "served", 1: "rotated"}
ROTATE_SCOPES = ("options", "candidates")
ROTATE_DEFAULT = "options"


# ---------------------------------------------------------------- candidates

def read_candidates(paths, shard=0, shards=1, limit=0):
    """Rows for this shard. Without --limit the shard filter is applied while streaming.

    v1.1 is 862 MB; four blend shards that each parse the whole file would hold four copies of it.
    """
    stream = shards > 1 and not limit
    rows, index = [], 0
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                if not stream or index % shards == shard:
                    rows.append(json.loads(line))
                index += 1
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
    return rows if stream else rows[shard::shards]


def expand_candidates(record):
    """[(field_id, one-field copy)] for scoring.

    ``decision_data.expand_fields`` cannot be used here: it requires a ``targets`` map, which a
    candidate (``target: null``) does not have.  The copies carry ``target: None`` so that
    ``render`` resolves the target index onto ``unknown``, which is always present.
    """
    fields = record["request"]["fields"]
    out = []
    for field in fields:
        copy = json.loads(json.dumps(record))
        copy["request"]["fields"] = [field]
        copy["id"] = record["id"] if len(fields) == 1 else f"{record['id']}:{field['id']}"
        copy["target"] = None
        copy["abstention_cause"] = None
        for stale in ("targets", "target_distributions", "abstention_causes", "target_distribution"):
            copy.pop(stale, None)
        out.append((field["id"], copy))
    return out


# ------------------------------------------------------------- distributions

def option_keys(choices):
    """Distribution keys for a candidate list, in served order; the last entry is unknown.

    ``render`` accepts ``__unknown__``/``unknown``/``null`` for the unknown mass and ``str(value)``
    (plus the lowercase form for booleans) for everything else.  A listed option whose own key would
    be ``unknown`` or ``null`` would make the unknown mass ambiguous, so the reserved identifier is
    used instead in that (rare) case.
    """
    if not choices or choices[-1][0] != UNKNOWN:
        raise ValueError("Candidate list must end with the reserved unknown candidate")
    keys = [candidate_key(value) for value, _ in choices[:-1]]
    unknown = UNKNOWN if {"unknown", "null"} & set(keys) else "unknown"
    if unknown in keys:
        raise ValueError("Listed options shadow the unknown key")
    return keys + [unknown]


def values_by_key(choices):
    """key -> the value ``render`` will compare against ``record['target']`` (bool/int stay typed)."""
    keys = option_keys(choices)
    return {k: (None if value == UNKNOWN else value) for k, (value, _) in zip(keys, choices)}


def presented(choices, texts, offset, scope=ROTATE_DEFAULT):
    """The candidate lines as they are shown to the model for this rotation."""
    if scope == "candidates":
        return rotate(choices, offset), rotate(texts, offset)
    n = len(choices) - 1  # unknown keeps the last line, as in training and serving
    return rotate(choices[:n], offset) + [choices[n]], rotate(texts[:n], offset) + [texts[n]]


def realign(probabilities, offset, scope=ROTATE_DEFAULT):
    """Rotated position p holds served candidate (p + offset) % n (see scoring.combine_rotations)."""
    n = len(probabilities)
    served = [0.0] * n
    span = n if scope == "candidates" else n - 1
    for position, value in enumerate(probabilities[:span]):
        served[(position + offset) % span] = value
    if span < n:
        served[span] = probabilities[span]
    return served


def average_orders(choices, logits_by_offset, temperature, scope=ROTATE_DEFAULT):
    """(distribution keyed by option value, per-order argmax keys, agreement)."""
    keys = option_keys(choices)
    n = len(choices)
    per_order = {}
    for offset, logits in sorted(logits_by_offset.items()):
        if len(logits) != n:
            raise ValueError("Each rotation must score every candidate")
        per_order[offset] = realign(softmax(list(logits), temperature or 1.0), offset, scope)
    orders = list(per_order.values())
    mean = [sum(o[i] for o in orders) / len(orders) for i in range(n)]
    total = sum(mean)
    distribution = {k: v / total for k, v in zip(keys, mean)}
    argmaxes = {ORDER_NAMES.get(offset, str(offset)): keys[max(range(n), key=o.__getitem__)]
                for offset, o in per_order.items()}
    pairs = [(a, b) for i, a in enumerate(orders) for b in orders[i + 1:]]
    agreement = min((1.0 - 0.5 * sum(abs(x - y) for x, y in zip(a, b)) for a, b in pairs), default=1.0)
    return distribution, argmaxes, agreement


def keep_decision(distribution, argmaxes, unknown_key, top=TOP_THRESHOLD, unknown_floor=UNKNOWN_THRESHOLD):
    """(keep?, reason, argmax key, top probability). Reason is None when the decision is kept."""
    best = max(distribution, key=lambda k: (distribution[k], k))
    probability = distribution[best]
    if len(set(argmaxes.values())) > 1 or any(v != best for v in argmaxes.values()):
        return False, "order_disagreement", best, probability
    if probability >= top:
        return True, None, best, probability
    if best == unknown_key and probability >= unknown_floor:
        return True, None, best, probability
    return False, ("low_confidence_unknown" if best == unknown_key else "low_confidence"), best, probability


# ------------------------------------------------------------------ assembly

def assemble(record, kept, teacher):
    """Build the labelled record from ``kept`` = {field_id: verdict}. Fields that failed are dropped."""
    out = json.loads(json.dumps(record))
    fields = [f for f in record["request"]["fields"] if f["id"] in kept]
    if not fields:
        raise ValueError(f"{record['id']}: no field passed the keep rule")
    out["request"]["fields"] = fields
    for stale in ("targets", "target_distributions", "abstention_causes", "target_distribution"):
        out.pop(stale, None)
    out["pseudo_label"] = teacher
    out["teacher_fields"] = {f["id"]: {k: v for k, v in kept[f["id"]].items() if k != "value"} for f in fields}
    out["teacher_confidence"] = min(kept[f["id"]]["confidence"] for f in fields)
    out["teacher_agreement"] = min(kept[f["id"]]["agreement"] for f in fields)
    if len(fields) == 1:
        verdict = kept[fields[0]["id"]]
        out["target"] = verdict["value"]
        out["abstention_cause"] = "teacher_unknown" if verdict["value"] is None else None
        out["target_distribution"] = verdict["distribution"]
    else:
        out["target"] = None
        out["abstention_cause"] = None
        out["targets"] = {f["id"]: kept[f["id"]]["value"] for f in fields}
        out["target_distributions"] = {f["id"]: kept[f["id"]]["distribution"] for f in fields}
        out["abstention_causes"] = {f["id"]: ("teacher_unknown" if kept[f["id"]]["value"] is None else None)
                                    for f in fields}
    return out


# ------------------------------------------------- self-distillation blending

def label_vector(choices, target):
    """The record's own label as a vector over ``choices``.

    ``target`` is what ``render`` returned: an index (hard label) or an already-normalised weight
    vector (the record carried a soft ``target_distribution``).  Going through ``render`` is what
    makes the existing soft keys -- ``True``/``true``, ``__unknown__``/``unknown``/``null``, the
    stringified ordinal levels -- resolve onto the served candidate order for free.
    """
    if isinstance(target, list):
        if len(target) != len(choices):
            raise ValueError("Existing distribution does not cover the candidate list")
        return list(target)
    return [1.0 if i == target else 0.0 for i in range(len(choices))]


def blend(model, label, alpha):
    """alpha * model + (1 - alpha) * label, renormalised."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--blend alpha must be in [0, 1]")
    if len(model) != len(label):
        raise ValueError("Blend operands must cover the same candidates")
    mixed = [alpha * m + (1.0 - alpha) * l for m, l in zip(model, label)]
    total = sum(mixed)
    if total <= 0 or not all(x >= 0 for x in mixed):
        raise ValueError("Blend produced an empty or negative distribution")
    return [x / total for x in mixed]


def has_not_listed(record):
    """Rows whose honest answer is an option the *augmenter* will append at training time.

    ``with_noise_state`` turns an ``abstention_cause: "not_listed"`` row into one with an extra
    "none of the above" option and moves the target onto it.  A soft distribution written before
    that happens has no key for the appended option, so its mass would stay on ``unknown`` and
    contradict the hard target.  Those rows are passed through unblended unless --blend-not-listed.
    """
    causes = [record.get("abstention_cause"), *(record.get("abstention_causes") or {}).values()]
    return "not_listed" in causes


def should_blend(record, partitions=("train",), blend_not_listed=False):
    if partitions is not None and record.get("partition") not in set(partitions):
        return False, "partition"
    if not blend_not_listed and has_not_listed(record):
        return False, "not_listed"
    return True, None


def blend_records(records, score, calibrator, base, alpha, partitions=("train",), scope=ROTATE_DEFAULT,
                  rotations=1, blend_not_listed=False, progress=None):
    """Regularise already-labelled records toward the base model's own distributions.

    Nothing is dropped: every input record is written out.  ``target``/``targets`` and every other
    field (``image_role``, control flags, licences) are left exactly as they were; only
    ``target_distribution`` / ``target_distributions`` are rewritten, plus ``regularizer``,
    ``regularizer_alpha`` and the per-field ``regularizer_fields`` diagnostic.
    """
    rows, passthrough = [], []
    summary = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    by_reason = collections.Counter()
    confidences, agreements = [], []
    for n, record in enumerate(records):
        out = json.loads(json.dumps(record))
        family = str(record.get("family", "unknown"))
        wanted, why = should_blend(record, partitions, blend_not_listed)
        try:
            items = expand_fields(out) if wanted else []
        except (ValueError, KeyError) as exc:
            wanted, why = False, "invalid_record"
            passthrough.append({"id": record.get("id"), "reason": why, "error": f"{type(exc).__name__}: {exc}"})
        if not wanted:
            if why != "invalid_record":
                passthrough.append({"id": record.get("id"), "reason": why})
            by_reason[why] += 1
            by_family[family]["passthrough"] += 1
            summary["passthrough_records"] += 1
            rows.append(out)
            summary["records"] += 1
            if progress:
                progress(n + 1, len(records))
            continue
        distributions, detail = {}, {}
        failed = None
        for item in items:
            field = item["request"]["fields"][0]
            try:
                _, choices, texts, target = render(item)
                keys = option_keys(choices)
                label = label_vector(choices, target)
                if alpha > 0:
                    temperature = calibrator.temperature(field["type"], len(choices) - 1) if calibrator else None
                    logits = {offset: list(score(item, *presented(choices, texts, offset, scope), offset))
                              for offset in OFFSETS[:rotations]}
                    model_scores, argmaxes, agreement = average_orders(choices, logits, temperature, scope)
                    model = [model_scores[k] for k in keys]
                else:  # a pure-label blend never needs the model
                    model, argmaxes, agreement = [0.0] * len(choices), {}, 1.0
                mixed = blend(model, label, alpha)
            except (ValueError, KeyError, AssertionError, OSError) as exc:
                failed = f"{type(exc).__name__}: {exc}"
                break
            distributions[field["id"]] = dict(zip(keys, mixed))
            best = max(range(len(model)), key=model.__getitem__)
            label_best = max(range(len(label)), key=label.__getitem__)
            detail[field["id"]] = {"model_argmax": keys[best] if alpha > 0 else None,
                                   "model_confidence": model[best] if alpha > 0 else None,
                                   "label_argmax": keys[label_best],
                                   "label_was_soft": isinstance(target, list),
                                   "base_agrees": bool(best == label_best) if alpha > 0 else None,
                                   "order_argmax": argmaxes, "order_agreement": agreement}
            if alpha > 0:
                confidences.append(model[best])
                agreements.append(1.0 if best == label_best else 0.0)
            summary["blended_decisions"] += 1
        if failed is not None:
            passthrough.append({"id": record.get("id"), "reason": "invalid_record", "error": failed})
            by_reason["invalid_record"] += 1
            by_family[family]["passthrough"] += 1
            summary["passthrough_records"] += 1
            rows.append(json.loads(json.dumps(record)))
        else:
            fields = out["request"]["fields"]
            if len(fields) == 1:
                out["target_distribution"] = distributions[fields[0]["id"]]
            else:
                out["target_distributions"] = distributions
            out["regularizer"] = f"base-{base}"
            out["regularizer_alpha"] = alpha
            out["regularizer_fields"] = detail
            by_family[family]["blended"] += 1
            summary["blended_records"] += 1
            rows.append(out)
        summary["records"] += 1
        if progress:
            progress(n + 1, len(records))
    report = {"mode": "blend", "records": summary["records"], "blended_records": summary["blended_records"],
              "blended_decisions": summary["blended_decisions"],
              "passthrough_records": summary["passthrough_records"],
              "passthrough_by_reason": dict(sorted(by_reason.items(), key=lambda x: str(x[0]))),
              "by_family": {f: dict(c) for f, c in sorted(by_family.items())},
              "regularizer": f"base-{base}", "regularizer_alpha": alpha,
              "base_agreement_rate": (sum(agreements) / len(agreements)) if agreements else None,
              "mean_model_confidence": (sum(confidences) / len(confidences)) if confidences else None,
              "rotate": scope, "rotations": rotations,
              "blend_partitions": "all" if partitions is None else sorted(partitions),
              "blend_not_listed": blend_not_listed}
    return rows, passthrough, report


def label_records(records, score, calibrator, teacher, top=TOP_THRESHOLD, unknown_floor=UNKNOWN_THRESHOLD,
                  progress=None, scope=ROTATE_DEFAULT):
    """Label every candidate decision.

    ``score(item, choices, texts, offset) -> logits`` scores one rotation of one decision and returns
    the logits in *presented* order.  Returns (labelled records, dropped rows, summary).
    """
    labelled, dropped = [], []
    summary = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    by_reason = collections.Counter()
    unknown_kept = 0
    confidences = []
    for n, record in enumerate(records):
        kept = {}
        try:
            expanded = expand_candidates(record)
        except (ValueError, KeyError) as exc:
            by_reason["invalid_candidate"] += 1
            summary["decisions"] += 1
            summary["dropped_records"] += 1
            summary["records"] += 1
            dropped.append({"id": record.get("id"), "record_id": record.get("id"), "field_id": None,
                            "source": record.get("source"), "family": str(record.get("family", "unknown")),
                            "reason": "invalid_candidate", "error": f"{type(exc).__name__}: {exc}"})
            continue
        for field_id, item in expanded:
            summary["decisions"] += 1
            family = str(record.get("family", "unknown"))
            try:
                _, choices, texts, _ = render(item)
                field = item["request"]["fields"][0]
                option_count = len(choices) - 1
                temperature = calibrator.temperature(field["type"], option_count) if calibrator else None
                logits = {offset: list(score(item, *presented(choices, texts, offset, scope), offset))
                          for offset in OFFSETS}
                distribution, argmaxes, agreement = average_orders(choices, logits, temperature, scope)
                keys = option_keys(choices)
            except (ValueError, KeyError, AssertionError, OSError) as exc:
                # A malformed candidate must not kill a multi-hour pod run; it is counted, loudly.
                by_reason["invalid_candidate"] += 1
                by_family[family]["dropped"] += 1
                dropped.append({"id": item["id"], "record_id": record["id"], "field_id": field_id,
                                "source": record.get("source"), "family": family,
                                "reason": "invalid_candidate", "error": f"{type(exc).__name__}: {exc}"})
                continue
            ok, reason, best, probability = keep_decision(distribution, argmaxes, keys[-1], top, unknown_floor)
            if not ok:
                by_reason[reason] += 1
                by_family[family]["dropped"] += 1
                dropped.append({"id": item["id"], "record_id": record["id"], "field_id": field_id,
                                "source": record.get("source"), "family": family,
                                "decision_type": field["type"], "option_count": option_count,
                                "reason": reason, "argmax": best, "confidence": probability,
                                "agreement": agreement, "order_argmax": argmaxes,
                                "distribution": distribution})
                continue
            value = values_by_key(choices)[best]
            kept[field_id] = {"value": value, "distribution": distribution, "confidence": probability,
                              "agreement": agreement, "argmax": best, "order_argmax": argmaxes,
                              "option_count": option_count, "decision_type": field["type"],
                              "temperature": temperature,
                              "calibration_version": calibrator.version if (calibrator and temperature) else None}
            by_family[family]["kept"] += 1
            summary["kept_decisions"] += 1
            confidences.append(probability)
            if value is None:
                unknown_kept += 1
        if kept:
            labelled.append(assemble(record, kept, teacher))
            summary["kept_records"] += 1
        else:
            summary["dropped_records"] += 1
        summary["records"] += 1
        if progress:
            progress(n + 1, len(records))
    report = {"mode": "label", "candidate_records": summary["records"], "candidate_decisions": summary["decisions"],
              "kept_records": summary["kept_records"], "dropped_records": summary["dropped_records"],
              "kept_decisions": summary["kept_decisions"],
              "dropped_decisions": summary["decisions"] - summary["kept_decisions"],
              "dropped_by_reason": dict(sorted(by_reason.items())),
              "by_family": {f: dict(c) for f, c in sorted(by_family.items())},
              "unknown_kept": unknown_kept,
              "unknown_share": (unknown_kept / summary["kept_decisions"]) if summary["kept_decisions"] else 0.0,
              "keep_rate": (summary["kept_decisions"] / summary["decisions"]) if summary["decisions"] else 0.0,
              "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
              "teacher": teacher, "top_threshold": top, "unknown_threshold": unknown_floor, "rotate": scope}
    return labelled, dropped, report


# -------------------------------------------------------------------- engine

def resolve_model(value, revision=None):
    """--model takes a bundle json (artifacts/model.json), a local snapshot directory, or a repo id."""
    path = Path(value)
    if path.suffix == ".json" and path.is_file():
        return json.loads(path.read_text())["path"]
    if path.is_dir():
        return str(path)
    from huggingface_hub import snapshot_download
    return snapshot_download(value, revision=revision)


class Engine:
    """Thin wrapper over TorchDecision that scores one rotation of one decision."""

    def __init__(self, model, device, adapter=None, pixels=400000, max_batch=1, token_budget=16000, revision=None, workers=0):
        import torch
        from torch_decision import TorchDecision
        self.torch = torch
        self.pixels = pixels
        self.workers = workers if device.startswith("cuda") else 0  # CPU-side batch preparation in worker processes (fork)
        self.max_batch = max_batch if device.startswith("cuda") else 1
        self.token_budget = token_budget
        self.engine = TorchDecision(resolve_model(model, revision), device)
        self.readout = "base_lm_head"
        if adapter:
            from peft import PeftModel
            self.engine.model = PeftModel.from_pretrained(self.engine.model, adapter).eval()
            # A False return means a legacy adapter without a trained readout: keep the lm_head path.
            self.readout = "trained" if self.engine.enable_readout(adapter, trainable=False) else "base_lm_head"
        # Pin the decision codebook to the no-image prompt, as enable_readout does, so that base-model
        # runs use the same candidate codes for image and text records.
        self.engine.labels(1, 0)

    def prompt(self, item, choices, texts):
        labels = self.engine.labels(len(choices), len(item["images"]))
        header, _, _, _ = render(item)
        return header + "\n".join(f"{l}: {t}" for l, t in zip(labels, texts)), labels

    def images(self, item):
        return [load_image(x["image"], pixel_budget(item, self.pixels)) for x in item["images"]]

    def score(self, item, choices, texts, offset):
        prompt, labels = self.prompt(item, choices, texts)
        _, inputs, token_ids = self.engine.prepare(self.images(item), prompt, labels)
        with self.torch.no_grad():
            return self.engine.candidate_logits(inputs, token_ids).float().cpu().tolist()

    def score_batched(self, jobs):
        """jobs: [(item, choices, texts, offset)] -> logits per job, length-bucketed like the evaluator."""
        order = sorted(range(len(jobs)), key=lambda i: approx_tokens(jobs[i][0], self.pixels))
        out = [None] * len(jobs)
        batch, longest = [], 0
        batches = []
        for i in order:
            n = approx_tokens(jobs[i][0], self.pixels)
            if batch and (max(longest, n) * (len(batch) + 1) > self.token_budget or len(batch) >= self.max_batch):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            batches.append(batch)
        def prepare(indices):  # CPU work: image decode/resize, rendering, tokenising, collating
            examples = []
            for i in indices:
                item, choices, texts, _ = jobs[i]
                prompt, labels = self.prompt(item, choices, texts)
                examples.append((*self.engine.render_example(self.images(item), prompt, labels), 0))
            inputs, token_ids, _ = self.engine.collate(examples)
            return indices, inputs, token_ids
        if self.workers:
            engine = self
            class _Prep(self.torch.utils.data.Dataset):
                def __len__(_):
                    return len(batches)
                def __getitem__(_, b):
                    return prepare(batches[b])
            stream = self.torch.utils.data.DataLoader(_Prep(), batch_size=None, shuffle=False, num_workers=self.workers, prefetch_factor=4)
        else:
            stream = (prepare(b) for b in batches)
        started, done, seen = time.monotonic(), 0, 0
        for indices, inputs, token_ids in stream:
            with self.torch.no_grad():
                logits = self.engine.candidate_logits_batch(inputs, token_ids)
            for i, x in zip(indices, logits):
                out[i] = x.float().cpu().tolist()
            done += 1; seen += len(indices)
            if done % 25 == 0 or done == len(batches):
                el = time.monotonic() - started
                print(f"scored {seen}/{len(jobs)} decisions in {done}/{len(batches)} batches, {seen / max(1e-9, el):.1f} dec/s, "
                      f"eta {(len(jobs) - seen) / max(1e-9, seen / max(1e-9, el)) / 60:.0f} min", flush=True)
        return out


def batched_scorer(engine, records, scope=ROTATE_DEFAULT, offsets=OFFSETS):
    """Score every rotation of every decision up front, then serve label_records from the cache."""
    jobs, index = [], {}
    for record in records:
        try:
            expanded = expand_candidates(record)
        except (ValueError, KeyError) as exc:  # re-raised per decision by label_records
            print(f"skipping malformed candidate {record.get('id')}: {exc}", flush=True)
            continue
        for _, item in expanded:
            try:
                _, choices, texts, _ = render(item)
            except (ValueError, KeyError, AssertionError, OSError):
                continue
            for offset in offsets:
                index[(item["id"], offset)] = len(jobs)
                jobs.append((item, *presented(choices, texts, offset, scope), offset))
    results = engine.score_batched(jobs)
    return lambda item, choices, texts, offset: results[index[(item["id"], offset)]]


# ---------------------------------------------------------------------- main

def shard_path(output, shard, shards):
    out = Path(output)
    return out if shards == 1 else out.with_name(f"{out.stem}.shard{shard}of{shards}{out.suffix}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True, help="candidate records.jsonl (one or more)")
    p.add_argument("--model", required=True, help="bundle json (artifacts/model-qwen9b.json), snapshot dir, or repo id")
    p.add_argument("--revision")
    p.add_argument("--adapter", help="trained adapter directory (with decision_readout.safetensors); "
                                     "'none' or omitted scores with the base model and its lm_head readout")
    p.add_argument("--calibration", help="temperature artifact; without it every bucket uses T=1")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default="0/1", help="i/n: label every n-th candidate record starting at i")
    p.add_argument("--token-budget", type=int, default=16000)
    p.add_argument("--limit", type=int, default=0, help="evenly spaced sample of candidate records")
    p.add_argument("--max-batch", type=int, default=32, help="clamped to 1 off CUDA (left padding NaNs on MPS)")
    p.add_argument("--pixels", type=int, default=400000)
    p.add_argument("--workers", type=int, default=8, help="batch-preparation worker processes on CUDA (0 = in-process; forced to 0 off CUDA)")
    p.add_argument("--teacher-name", default="9b-v1.1", help="written as pseudo_label: teacher-<name>")
    p.add_argument("--rotate", choices=ROTATE_SCOPES, default=ROTATE_DEFAULT,
                   help="options: rotate the listed options, unknown stays last (spec). "
                        "candidates: rotate the whole list, unknown included.")
    p.add_argument("--top-threshold", type=float, default=TOP_THRESHOLD)
    p.add_argument("--unknown-threshold", type=float, default=UNKNOWN_THRESHOLD)
    p.add_argument("--rotations", type=int, choices=(1, 2), default=None,
                   help="presentation orders to score; default 2 when labelling, 1 when blending")
    blend = p.add_argument_group("self-distillation blend (already-labelled records)")
    blend.add_argument("--blend", type=float, metavar="ALPHA",
                       help="regularise toward the model: target_distribution = ALPHA * model + "
                            "(1 - ALPHA) * existing target. No keep rule; every record is written out.")
    blend.add_argument("--blend-partitions", default="train",
                       help="comma-separated partitions to blend, or 'all'; the rest pass through unchanged")
    blend.add_argument("--blend-not-listed", action="store_true",
                       help="also blend abstention_cause='not_listed' rows (see has_not_listed: the "
                            "trainer's augmenter appends an option those soft targets cannot name)")
    blend.add_argument("--base-name", default="9b-base", help="written as regularizer: base-<name>")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    shard, shards = map(int, args.shard.split("/"))
    records = read_candidates(args.manifest, shard, shards, args.limit)
    adapter = None if args.adapter in (None, "", "none", "None") else args.adapter
    calibrator = TemperatureCalibrator.load(args.calibration) if args.calibration else None
    blending = args.blend is not None
    rotations = args.rotations or (1 if blending else 2)
    partitions = None if args.blend_partitions == "all" else tuple(
        x for x in args.blend_partitions.split(",") if x)
    if blending and not 0.0 <= args.blend <= 1.0:
        raise SystemExit("--blend alpha must be in [0, 1]")
    scored = [r for r in records if should_blend(r, partitions, args.blend_not_listed)[0]] if blending else records
    # A pure-label blend (alpha 0) never consults the model, so it does not load one.
    engine = None if (blending and args.blend == 0) else Engine(
        args.model, args.device, adapter, args.pixels, args.max_batch, args.token_budget, args.revision, workers=args.workers)
    started = time.monotonic()
    progress = None
    if engine is None:
        def score(*_):
            raise AssertionError("alpha 0 must not score")
    elif engine.max_batch > 1:
        score = batched_scorer(engine, scored, args.rotate, OFFSETS[:rotations])
    else:
        score = engine.score

        def progress(done, total):
            if done % 10 == 0 or done == total:
                print(f"{done}/{total} records {done / max(1e-9, time.monotonic() - started):.2f} rec/s", flush=True)
    if blending:
        rows, notes, report = blend_records(records, score, calibrator, args.base_name, args.blend, partitions,
                                            args.rotate, rotations, args.blend_not_listed, progress)
        sidecar = "passthrough"
    else:
        rows, notes, report = label_records(records, score, calibrator, f"teacher-{args.teacher_name}",
                                            args.top_threshold, args.unknown_threshold, progress, args.rotate)
        sidecar = "dropped"
    out = shard_path(args.output, shard, shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n" for r in rows))
    Path(f"{out}.{sidecar}.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in notes))
    seconds = time.monotonic() - started
    report.update(output=str(out), shard=args.shard, device=args.device,
                  max_batch=engine.max_batch if engine else 0, readout=engine.readout if engine else None,
                  model=args.model, adapter=adapter, rotations=rotations,
                  calibration_version=calibrator.version if calibrator else None,
                  seconds=round(seconds, 1),
                  records_per_second=round(len(records) / seconds, 3) if seconds > 0 else None)
    Path(str(out) + ".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
