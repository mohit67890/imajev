"""Gate 0 for phase-3 item 5: the 256-code readout must reproduce the shipped 255-code model bit for bit.

Loads the SHIPPED imajev-4b adapter twice, one model at a time in this one process (the Mac rule: never two models
resident): first as shipped (Linear[255, hidden]), then extended to 256 codes (row 256 appended from its LM-head row).
Every question with <= 254 options is scored single-pass (the uncached full forward, no rotations) by both, and the
full candidate logit and probability vectors are compared for exact equality. A determinism control re-scores a slice
with the first load, so a mismatch can be told apart from kernel run-to-run noise.

    .venv/bin/python scripts/p3/parity_readout256.py            # MLX, ~700 questions, writes reports/phase3/

CUDA (the pod's torch path; phase-3 plan item 5, "rerun the parity check on the pod's CUDA torch path before the +256 lane"):
    python scripts/p3/parity_readout256.py --backend torch --device cuda --adapter adapters/4b-shipped \
        --bundle <json with "path": Qwen3.5-4B snapshot> --out p3/run/parity
The torch engine (scripts/torch_decision.py TorchDecision, the path the trainer, evaluator and server use) runs the full
Linear[codes, hidden] and then indexes it, so on CUDA a 256-row GEMM could in principle round differently from 255 rows;
this run measures exactly that. Exit 0 = bit-identical (the pod enables the +256 lanes), 1 = not (they stay off).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_decision.contracts import Request  # noqa: E402
from vision_decision.jev_api import to_request  # noqa: E402

ADAPTER = ROOT.parent / "imajev-release/hf/imajev-4b/mlx"
BUNDLE = ROOT / "artifacts/model-qwen4b-local.json"
JEVBENCH = ROOT / ".cache/external/jevbench/datasets/public"
JEVSTYLE_DEV = ROOT / "data/manifests/decision-p2b-jevstyle-dev.jsonl"
CANDIDATES = ROOT / "data/p3/candidates-clean"


def candidate_request(row: dict) -> Request:
    """A phase-3 candidate row (scripts/p3/candidate.py) as a served request (one field)."""
    f = row["field"]
    if f["type"] == "choice":
        field = {"type": "choice", "options": [
            {"value": o["key"], "description": (f"{o['text']}: {o['description']}" if o.get("description") else o["text"]) or None}
            for o in f["options"]]}
    elif f["type"] == "noul":
        field = {"type": "boolean"}
    else:
        field = {"type": "ordinal", "levels": [{"value": l["value"], "description": l["description"]} for l in f["levels"]]}
    state = row["state"] if isinstance(row["state"], (dict, str)) else {"state": row["state"]}
    return Request.model_validate({"request_id": row["id"], "state": state,
                                   "fields": [{"id": "decision", "question": f["question"], **field}]})


def load_items(n_candidates: int, seed: int = 0, candidates=CANDIDATES, jevbench=JEVBENCH, jevstyle_dev=JEVSTYLE_DEV) -> list[tuple[str, str, Request]]:
    """(set, id, request) for: a seeded sample of phase-3 candidates, JevBench public, and the p2b JevBench-style dev set,
    plus synthetic wide-choice questions that reach the two-letter codes (27..254 options)."""
    items = []
    files = sorted(Path(candidates).glob("B-*.jsonl"))
    per_file = max(1, n_candidates // max(1, len(files)))
    for path in files:
        rows = [json.loads(x) for x in path.open() if x.strip()]
        for row in random.Random(f"{seed}:{path.name}").sample(rows, min(per_file, len(rows))):
            try:
                items.append((f"candidates/{path.stem}", row["id"], candidate_request(row)))
            except Exception as exc:  # a row the served contract refuses is reported, never silently dropped
                print(f"skip {row['id']}: {exc}", file=sys.stderr)
    for tier in ("easy", "original", "hard"):
        if not (Path(jevbench) / f"{tier}.jsonl").exists():
            print(f"missing {Path(jevbench) / f'{tier}.jsonl'} (skipped)", file=sys.stderr); continue
        for row in (json.loads(x) for x in (Path(jevbench) / f"{tier}.jsonl").open() if x.strip()):
            items.append((f"jevbench/{tier}", row["id"], to_request({"state": row["state"], "questions": {"decision": row["question"]}})))
    for row in (json.loads(x) for x in (Path(jevstyle_dev).open() if Path(jevstyle_dev).exists() else []) if x.strip()):
        items.append(("p2b-jevstyle-dev", row["id"], Request.model_validate(row["request"])))
    for n in (27, 60, 128, 200, 254):
        criteria = {f"item_{i:03d}": f"catalogue entry {i}" for i in range(n)}
        items.append(("synthetic-wide", f"wide-{n}", to_request({
            "state": {"order": {"sku": f"item_{(n * 7) % n:03d}", "note": "customer asked for the listed entry"}},
            "questions": {"decision": {"type": "choice", "instructions": "Which catalogue entry does `order.sku` name?",
                                       "criteria": criteria}}})))
    return items


class TorchEngine:
    """TorchDecision behind the MLXDirect.score(image, field, state) -> (result, meta) interface used below."""

    def __init__(self, bundle, adapter, readout_codes, device="cuda", max_input_tokens=16384):
        import torch
        from peft import PeftModel
        from torch_decision import TorchDecision
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False   # the float32 candidate head must stay float32
        start = time.monotonic()
        path = json.loads(Path(bundle).read_text())["path"]
        self.engine = TorchDecision(path, device, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
                                    max_length=max_input_tokens)
        self.engine.model = PeftModel.from_pretrained(self.engine.model, str(adapter)).eval()
        if not self.engine.enable_readout(adapter, trainable=False, codes=readout_codes):
            raise ValueError(f"{adapter} has no decision readout")
        self.codes, self.prompt_layout = self.engine.codes, self.engine.prompt_layout
        self.readout = self.engine.readout.weight
        self.load_seconds = time.monotonic() - start

    @property
    def _codebook(self):
        return self.engine._ensure_codebook(0)

    def readout_sha256(self):
        return hashlib.sha256(self.readout[:255].detach().float().cpu().numpy().tobytes()).hexdigest()

    def score(self, image, field, state):
        from vision_decision.scoring import compile_question, result_from_logits
        header, choices, texts = compile_question(field, state, self.prompt_layout)
        images = [] if image is None else image if isinstance(image, list) else [image]
        labels = self.engine.labels(len(choices), len(images))
        prompt = header + "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
        with self.torch.no_grad():
            rendered, inputs, token_ids = self.engine.prepare(images, prompt, labels)
            logits = [float(x) for x in self.engine.candidate_logits(inputs, token_ids).cpu().tolist()]
        indices = self.engine._readout_indices(token_ids)
        meta = {"choice_token_ids": list(token_ids), "readout_indices": indices,
                "readout": f"trained_{self.codes}" if indices is not None else "vocabulary",
                "template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "input_tokens": int(inputs["input_ids"].shape[-1])}
        return result_from_logits(choices, logits, token_ids=token_ids), meta


def score_all(engine, items, label):
    rows, started = [], time.monotonic()
    for k, (group, item_id, request) in enumerate(items):
        result, meta = engine.score(None, request.fields[0], request.state)
        rows.append({"set": group, "id": item_id, "labels": list(result.raw_logits), "logits": list(result.raw_logits.values()),
                     "probabilities": list(result.scores.values()), "value": result.value,
                     "choice_token_ids": meta["choice_token_ids"], "readout_indices": meta["readout_indices"],
                     "readout": meta["readout"], "template_sha256": meta["template_sha256"], "input_tokens": meta["input_tokens"]})
        if (k + 1) % 50 == 0:
            print(f"[{label}] {k + 1}/{len(items)}  {time.monotonic() - started:.0f}s", flush=True)
    return rows, time.monotonic() - started


def compare(a_rows, b_rows):
    exact_logits = exact_probs = 0
    max_logit = max_prob = 0.0
    mismatches = []
    for a, b in zip(a_rows, b_rows):
        assert (a["id"], a["labels"], a["template_sha256"], a["choice_token_ids"]) == \
               (b["id"], b["labels"], b["template_sha256"], b["choice_token_ids"]), a["id"]
        dl = max(abs(x - y) for x, y in zip(a["logits"], b["logits"]))
        dp = max(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"]))
        exact_logits += a["logits"] == b["logits"]
        exact_probs += a["probabilities"] == b["probabilities"]
        max_logit, max_prob = max(max_logit, dl), max(max_prob, dp)
        if a["logits"] != b["logits"]:
            mismatches.append({"id": a["id"], "set": a["set"], "max_abs_logit_diff": dl, "max_abs_prob_diff": dp})
    return {"questions": len(a_rows), "bit_identical_logits": exact_logits, "bit_identical_probabilities": exact_probs,
            "max_abs_logit_diff": max_logit, "max_abs_probability_diff": max_prob, "mismatches": mismatches[:20]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default=str(ADAPTER))
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--control", type=int, default=40, help="questions re-scored by the first load (determinism control)")
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--out", default=None, help="default reports/phase3/parity-readout256 (MLX) or parity-readout256-<device> (torch)")
    parser.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--device", default="cuda", help="torch backend: cuda (pod), mps or cpu")
    parser.add_argument("--candidates-dir", default=str(CANDIDATES)); parser.add_argument("--jevbench", default=str(JEVBENCH))
    parser.add_argument("--jevstyle-dev", default=str(JEVSTYLE_DEV))
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(ROOT / ("reports/phase3/parity-readout256" if args.backend == "mlx" else f"reports/phase3/parity-readout256-{args.device}"))
    if args.backend == "mlx":
        import mlx.core as mx
        from vision_decision.backend import MLXDirect
        make_engine = lambda codes: MLXDirect(args.bundle, adapter=args.adapter, max_input_tokens=args.max_input_tokens, readout_codes=codes)
        readout_sha = lambda engine: hashlib.sha256(__import__("numpy").array(engine.readout[:255].astype(mx.float32)).tobytes()).hexdigest()
        clear, version = mx.clear_cache, mx.__version__
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        import torch
        make_engine = lambda codes: TorchEngine(args.bundle, args.adapter, codes, args.device, args.max_input_tokens)
        readout_sha = lambda engine: engine.readout_sha256()
        clear = (lambda: torch.cuda.empty_cache()) if torch.cuda.is_available() else (lambda: None)
        version = torch.__version__

    items = load_items(args.candidates, candidates=args.candidates_dir, jevbench=args.jevbench, jevstyle_dev=args.jevstyle_dev)
    assert all(len(r.fields[0].options) <= 254 for _, _, r in items if r.fields[0].type == "choice")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} questions", flush=True)
    runs = {}
    for codes in (255, 256):   # strictly sequential: the first model is freed before the second loads
        engine = make_engine(codes)
        print(f"loaded codes={codes} readout={tuple(engine.readout.shape)} in {engine.load_seconds:.1f}s", flush=True)
        rows, seconds = score_all(engine, items, f"codes={codes}")
        runs[codes] = {"rows": rows, "seconds": seconds, "readout_shape": list(engine.readout.shape), "readout_sha256": readout_sha(engine)}
        if codes == 255:
            control, _ = score_all(engine, items[:args.control], "control")
            runs["control"] = compare(rows[:args.control], control)
        else:
            wide = to_request({"questions": {"decision": {"type": "choice", "instructions": "Which entry is item_254?",
                                                          "criteria": {f"item_{i:03d}": None for i in range(255)}}}},
                              max_options=255)
            result, meta = engine.score(None, wide.fields[0], wide.state)
            runs["wide255"] = {"options": 255, "candidates": len(result.scores), "unknown_code": engine._codebook[255][0],
                               "unknown_token_id": meta["choice_token_ids"][-1], "readout": meta["readout"],
                               "value": result.value, "p_unknown": result.scores["__unknown__"],
                               "finite": all(math.isfinite(x) for x in result.raw_logits.values())}
        for key in ("rows",):
            with (out / f"codes{codes}.jsonl").open("w") as fh:
                for row in runs[codes][key]:
                    fh.write(json.dumps(row) + "\n")
        del engine
        gc.collect()
        clear()
    summary = {"adapter": args.adapter, "bundle": json.loads(Path(args.bundle).read_text())["repo"],
               "backend": args.backend, "device": args.device if args.backend == "torch" else "metal", args.backend: version,
               "path": "MLXDirect.score (uncached full forward, single pass)" if args.backend == "mlx" else
                       "TorchDecision.prepare + candidate_logits (full Linear then index; single pass)",
               "parity": compare(runs[255]["rows"], runs[256]["rows"]), "determinism_control": runs["control"],
               "readout": {c: {k: runs[c][k] for k in ("readout_shape", "readout_sha256", "seconds")} for c in (255, 256)},
               "wide255_smoke": runs["wide255"],
               "sets": {s: sum(1 for r in runs[255]["rows"] if r["set"] == s) for s in dict.fromkeys(r["set"] for r in runs[255]["rows"])},
               "input_tokens": {"mean": sum(r["input_tokens"] for r in runs[255]["rows"]) / len(items),
                                "max": max(r["input_tokens"] for r in runs[255]["rows"])}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("parity", "determinism_control", "wide255_smoke")}, indent=2))
    ok = summary["parity"]["bit_identical_logits"] == summary["parity"]["questions"]
    print("GATE 0:", "PASS (bit-identical)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
