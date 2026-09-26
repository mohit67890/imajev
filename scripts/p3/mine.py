#!/usr/bin/env python3
"""Phase-3 Stage 1 mining worker (docs/phase-3-plan.md, "Stage 1" + "Streamed mining + teacher").

Scores every pool item with the shipped imajev-4b (single pass, raw probabilities, standard layout, 255 codes) and flags it when
    * the argmax is wrong (any item with a gold, unknown gold included)                                   -> "wrong"
    * it is correct but the raw p(gold) < 0.6                                                             -> "low_conf"
    * a 2-order check (the serving rotation by n//2) changes the argmax; run on a deterministic 20% sample -> "order_flip"
    * the item has no gold (gold_kind "none") and the raw top probability < 0.6                           -> "unsure"

Input: pool shards `data/p3/pool/shards/*.jsonl` (scripts/p3/candidate.py rows). Output per shard, append-only JSONL, every row
keyed by the item id:
    <out>/<shard>.scores.jsonl    one row per scored item (flagged or not), for later analysis
    <out>/<shard>.flagged.jsonl   {"id", "shard", "item": <candidate row>, "mine": <score row>, "priority"}
    <out>/<shard>.done            JSON summary, written atomically when the shard is complete
    <out>/progress/<worker>.json  heartbeat + counters (atomic rewrite every ~30 s)
Workers (one per GPU) claim shards with an exclusive flock on <out>/locks/<shard>.lock, so a crashed worker's claim is released
by the kernel; a re-claimed shard resumes after the ids already in its scores file (a torn last line is repaired first).

Pod (CUDA torch, batched, one worker per GPU):
    for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python scripts/p3/mine.py --backend torch \\
        --worker-id gpu$g > logs/mine-gpu$g.log 2>&1 & done
Mac smoke test (ONE model process only; the Mac kernel-panicked with several resident):
    .venv/bin/python scripts/p3/mine.py --backend mlx --shards 'data/p3/pool/shards/*-000.jsonl' --limit 20 --out data/p3/mine-smoke
Plumbing test without any model: --backend fake.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

LOW_CONF = 0.6
ORDER_SAMPLE = 0.2
UNKNOWN_KEY = "__unknown__"          # vision_decision.contracts.UNKNOWN: the key result scores use for unknown


# ------------------------------------------------------------------------------------------------ small file helpers

def repair_tail(path: Path) -> None:
    """Drop a torn last line (a crash mid-write) so appends start on a clean line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
        fh.truncate(data.rfind(b"\n") + 1)


def read_ids(path: Path) -> set[str]:
    ids = set()
    if path.exists():
        with open(path) as fh:
            for line in fh:
                if line.endswith("\n") and line.strip():
                    try:
                        ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


class Appender:
    """Line-buffered append-only JSONL writer; flush + fsync on every batch."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        repair_tail(path)
        self.fh = open(path, "a")

    def write(self, row: dict) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sync(self) -> None:
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def close(self) -> None:
        self.sync()
        self.fh.close()


# ------------------------------------------------------------------------------------------------ item helpers

def gold_key(row: dict) -> str | None:
    """The candidate gold in the result-score key space ("true"/"false", option key, level as str, __unknown__);
    None when the item has no gold at all (gold_kind "none")."""
    if row.get("gold_kind") == "none":
        return None
    g = row.get("gold")
    if g is None:
        return UNKNOWN_KEY
    if isinstance(g, bool):
        return "true" if g else "false"
    return str(g)


def in_order_sample(item_id: str, share: float = ORDER_SAMPLE) -> bool:
    h = int(hashlib.sha1(f"order:{item_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def flag_item(row: dict, scores: dict, scores_rot: dict | None, low_conf: float = LOW_CONF) -> dict:
    """Apply the Stage-1 flag rules to one item's raw probabilities (key -> p). Returns the score-row fields."""
    pred = max(scores, key=lambda k: scores[k])
    gk = gold_key(row)
    out = {"pred": pred, "p_pred": round(scores[pred], 6), "p_unknown": round(scores.get(UNKNOWN_KEY, 0.0), 6),
           "top": [[k, round(p, 6)] for k, p in sorted(scores.items(), key=lambda kv: -kv[1])[:5]], "gold": gk}
    flags = []
    if gk is not None:
        out["p_gold"] = round(scores.get(gk, 0.0), 6)
        out["correct"] = pred == gk
        if not out["correct"]:
            flags.append("wrong")
        elif out["p_gold"] < low_conf:
            flags.append("low_conf")
    elif scores[pred] < low_conf:
        flags.append("unsure")
    out["order_checked"] = scores_rot is not None
    if scores_rot is not None:
        pred_rot = max(scores_rot, key=lambda k: scores_rot[k])
        out["pred_rot"] = pred_rot
        out["flip"] = pred_rot != pred
        if gk is not None:
            out["p_gold_rot"] = round(scores_rot.get(gk, 0.0), 6)
        if out["flip"]:
            flags.append("order_flip")
    out["flags"] = flags
    out["flagged"] = bool(flags)
    return out


def priority(mine: dict) -> float:
    """Most confidently wrong first, then least confident correct, then order flips / unsure."""
    f = mine.get("flags") or []
    if "wrong" in f:
        return 2.0 + mine.get("p_pred", 0.0)
    if "low_conf" in f:
        return 1.0 + (LOW_CONF - mine.get("p_gold", LOW_CONF))
    if "order_flip" in f:
        return 0.5
    return 0.25 + (LOW_CONF - mine.get("p_pred", LOW_CONF)) / 4


def load_images(row: dict, data_root: Path):
    from vision_decision.images import load_image
    return [load_image(data_root / rel)[0] for rel in row.get("images") or []]


# ------------------------------------------------------------------------------------------------ scorers
# A scorer takes a list of (row, request, images, offset) and returns, per entry, (scores {key: p}, input_tokens) or an Exception.

class TorchScorer:
    """Batched CUDA path on the shipped PEFT adapter (scripts/torch_decision.py): left padding puts every decision position
    at index -1; batches are length-sorted and capped by a padded-token budget. offset 0 keeps the serving tie break."""

    def __init__(self, model_path: str, adapter: Path, max_input_tokens: int, batch_tokens: int, max_batch: int, image_batch: int,
                 device: str | None = None):
        # The same load as the playground TorchBackend (PEFT adapter + trained 255-code readout), without its web dependencies.
        import torch
        from peft import PeftModel
        from torch_decision import TorchDecision
        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = TorchDecision(model_path, device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                    max_length=max_input_tokens)
        self.engine.model = PeftModel.from_pretrained(self.engine.model, str(adapter)).eval()
        if not self.engine.enable_readout(adapter, trainable=False, codes=255):
            raise ValueError(f"{adapter} has no decision_readout.safetensors: not the shipped imajev-4b adapter")
        if self.engine.prompt_layout != "standard":
            raise ValueError(f"adapter layout {self.engine.prompt_layout!r}; mining uses the shipped standard layout")
        self.batch_tokens, self.max_batch, self.image_batch = batch_tokens, max_batch, image_batch
        self.tok = self.engine.processor.tokenizer

    def _example(self, entry):
        from vision_decision.scoring import compile_question, rotate
        row, request, images, offset = entry
        field = request.fields[0]
        header, choices, texts = compile_question(field, request.state, "standard")
        labels = self.engine.labels(len(choices), len(images))
        prompt = header + "\n".join(f"{l}: {t}" for l, t in zip(labels, rotate(texts, offset)))
        rendered, imgs, token_ids = self.engine.render_example(images, prompt, labels)
        n = len(self.tok.encode(rendered, add_special_tokens=False))   # text tokens: the batching estimate (images add more)
        return rendered, imgs, token_ids, rotate(choices, offset), n

    def _run(self, batch):
        from vision_decision.scoring import result_from_logits
        inputs, token_ids, targets = self.engine.collate([(r, i, t, c) for r, i, t, c, _ in batch])
        with self.torch.no_grad():
            logits = self.engine.candidate_logits_batch(inputs, token_ids)
        lengths = [int(x) for x in inputs["attention_mask"].sum(-1).tolist()]     # real tokens, image tokens included
        out = []
        for (_, _, tids, choices, _), lg, n in zip(batch, logits, lengths):
            res = result_from_logits(choices, [float(x) for x in lg.cpu().tolist()], token_ids=tids)
            out.append((dict(res.scores), n))
        return out

    def _run_safe(self, batch):
        try:
            return self._run(batch)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(batch) == 1:
                return [RuntimeError("cuda out of memory on a single item")]
        except ValueError as exc:  # e.g. over the token limit after image expansion
            if len(batch) == 1:
                return [exc]
        mid = len(batch) // 2
        return self._run_safe(batch[:mid]) + self._run_safe(batch[mid:])

    def score(self, entries):
        prepared = [None] * len(entries)
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(self._example, e): i for i, e in enumerate(entries)}
            for f, i in futs.items():
                try:
                    prepared[i] = f.result()
                except Exception as exc:  # noqa: BLE001 - reported per item
                    prepared[i] = exc
        results: list = [None] * len(entries)
        ok = [i for i, p in enumerate(prepared) if not isinstance(p, Exception)]
        for i, p in enumerate(prepared):
            if isinstance(p, Exception):
                results[i] = p
            elif p[4] > self.engine.max_length:
                results[i] = ValueError(f"too_long: {p[4]} tokens > {self.engine.max_length}")
        ok = [i for i in ok if results[i] is None]
        ok.sort(key=lambda i: prepared[i][4])
        batch: list[int] = []

        def flush():
            if batch:
                for i, r in zip(batch, self._run_safe([prepared[j] for j in batch])):
                    results[i] = r
                batch.clear()

        for i in ok:
            has_img = bool(prepared[i][1])
            cap = self.image_batch if has_img else self.max_batch
            longest = max([prepared[j][4] for j in batch] + [prepared[i][4]])
            if batch and (longest * (len(batch) + 1) > self.batch_tokens or len(batch) >= cap
                          or bool(prepared[batch[0]][1]) != has_img):
                flush()
            batch.append(i)
        flush()
        return results


class MLXScorer:
    """Local Mac smoke path: MLXDirect, one uncached forward per item. Never run next to another model process."""

    def __init__(self, bundle: Path, adapter: Path, max_input_tokens: int):
        from vision_decision.backend import MLXDirect
        self.engine = MLXDirect(str(bundle), adapter=str(adapter), max_input_tokens=max_input_tokens, readout_codes=255,
                                prompt_layout="standard")

    def score(self, entries):
        from vision_decision.scoring import compile_question, rotate
        out = []
        for row, request, images, offset in entries:
            try:
                field = request.fields[0]
                header, choices, texts = compile_question(field, request.state, "standard")
                labels = self.engine._labels(header, len(choices), len(images))
                prompt = header + "\n".join(f"{l}: {t}" for l, t in zip(labels, rotate(texts, offset)))
                image = None if not images else images[0] if len(images) == 1 else list(images)
                res, meta = self.engine.score_compiled(image, prompt, labels, rotate(choices, offset))
                out.append((dict(res.scores), int(meta.get("input_tokens", 0))))
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
        return out


class FakeScorer:
    """Deterministic model-free scorer for plumbing tests: `fn(row, offset) -> {key: p}` or a hash-based default. With
    `accuracy` (dry runs: --fake-accuracy) the gold gets the top probability on that share of items (hash-chosen), so the flag
    rate looks like a real model's (~35% at 0.7) instead of ~80% for a random scorer."""

    def __init__(self, fn=None, accuracy: float | None = None):
        self.fn, self.accuracy = fn, accuracy

    def score(self, entries):
        from vision_decision.scoring import candidates, key
        out = []
        for row, request, images, offset in entries:
            if self.fn is not None:
                out.append((self.fn(row, offset), 100))
                continue
            keys = [key(v) for v, _ in candidates(request.fields[0])]
            gk = gold_key(row)
            if self.accuracy is not None and gk in keys:
                u = int(hashlib.sha1(f"acc:{row['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                v = int(hashlib.sha1(f"conf:{row['id']}:{offset}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                top = gk if u < self.accuracy else keys[int(v * 1e6) % len(keys)]
                p_top = 0.45 + 0.5 * v
                rest = [k for k in keys if k != top]
                out.append(({**{k: (1 - p_top) / max(1, len(rest)) for k in rest}, top: p_top if rest else 1.0}, 100))
                continue
            h = [int(hashlib.sha1(f"{row['id']}:{k}:{offset}".encode()).hexdigest()[:6], 16) for k in keys]
            z = sum(math.exp(x / 2 ** 22) for x in h)
            out.append(({k: math.exp(x / 2 ** 22) / z for k, x in zip(keys, h)}, 100))
        return out


# ------------------------------------------------------------------------------------------------ shard processing

def build_entries(rows, data_root: Path, order_share: float, max_options: int = 254):
    """-> list of (row, request, images, needs_order_check) or (row, Exception)."""
    from parity_readout256 import candidate_request
    out = []
    for row in rows:
        try:
            n_opts = len(row["field"].get("options") or [])
            if n_opts > max_options:
                raise ValueError(f"{n_opts} options exceed the 255-code readout")
            request = candidate_request(row)
            images = load_images(row, data_root)
            out.append((row, request, images, in_order_sample(row["id"], order_share)))
        except Exception as exc:  # noqa: BLE001
            out.append((row, exc))
    return out


def score_rows(scorer, rows, data_root: Path, order_share: float, low_conf: float, shard: str):
    """-> list of (score_row, flagged_row | None) for one chunk of rows."""
    from vision_decision.scoring import candidates, cyclic_offsets
    entries = build_entries(rows, data_root, order_share)
    first = [(e[0], e[1], e[2], 0) for e in entries if len(e) == 4]
    second = []
    for e in entries:
        if len(e) == 4 and e[3]:
            n = len(candidates(e[1].fields[0]))
            second.append((e[0], e[1], e[2], cyclic_offsets(n, 2)[1]))
    t0 = time.time()
    res1 = dict(zip((x[0]["id"] for x in first), scorer.score(first))) if first else {}
    res2 = dict(zip((x[0]["id"] for x in second), scorer.score(second))) if second else {}
    per_item = (time.time() - t0) / max(1, len(first))
    out = []
    for e in entries:
        row = e[0]
        base = {"id": row["id"], "shard": shard, "source": row.get("source"), "dataset": row.get("dataset"),
                "family": row.get("family"), "difficulty": row.get("difficulty"), "type": row["field"].get("type"),
                "n_options": len(row["field"].get("options") or row["field"].get("levels") or []) or 2,
                "gold_kind": row.get("gold_kind"), "has_images": bool(row.get("images"))}
        if len(e) == 2 or isinstance(res1.get(row["id"]), Exception):
            exc = e[1] if len(e) == 2 else res1[row["id"]]
            msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            status = "too_long" if "too_long" in msg or "token limit" in msg else "invalid" if len(e) == 2 else "error"
            out.append(({**base, "status": status, "error": msg, "flagged": False}, None))
            continue
        scores, ntok = res1[row["id"]]
        rot = res2.get(row["id"])
        scores_rot = rot[0] if rot is not None and not isinstance(rot, Exception) else None
        rec = {**base, "status": "ok", **flag_item(row, scores, scores_rot, low_conf), "input_tokens": ntok,
               "secs": round(per_item, 4)}
        flagged = ({"id": row["id"], "shard": shard, "item": row, "mine": rec, "priority": round(priority(rec), 6)}
                   if rec["flagged"] else None)
        out.append((rec, flagged))
    return out


class ShardLock:
    """Exclusive, crash-released claim on one shard (fcntl.flock on a local file)."""

    def __init__(self, path: Path, worker: str):
        self.path, self.worker, self.fh = path, worker, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"worker": self.worker, "pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}))
        fh.flush()
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


def iter_rows(path: Path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def process_shard(scorer, shard: Path, out: Path, args, progress) -> dict | None:
    stem = shard.stem
    done_path = out / f"{stem}.done"
    if done_path.exists():
        return None
    lock = ShardLock(out / "locks" / f"{stem}.lock", args.worker_id)
    if not lock.acquire():
        return None
    try:
        if done_path.exists():
            return None
        scores_path, flagged_path = out / f"{stem}.scores.jsonl", out / f"{stem}.flagged.jsonl"
        for p in (scores_path, flagged_path):
            repair_tail(p)
        done_ids, flagged_ids = read_ids(scores_path), read_ids(flagged_path)
        sw, fw = Appender(scores_path), Appender(flagged_path)
        t0, n, nf, chunk = time.time(), 0, 0, []
        limit = args.limit
        progress.update(shard=stem, resumed=len(done_ids))

        def run(chunk):
            nonlocal n, nf
            nf0 = nf
            for rec, flagged in score_rows(scorer, chunk, Path(args.data_root), args.order_sample, args.low_conf, stem):
                if flagged is not None and rec["id"] not in flagged_ids:
                    fw.write(flagged)          # flagged before score: a crash in between re-scores, never loses a flag
                    flagged_ids.add(rec["id"])
                    nf += 1
                sw.write(rec)
                n += 1
            fw.sync()
            sw.sync()
            progress.update(items=progress.state["items"] + len(chunk), flagged=progress.state["flagged"] + nf - nf0)

        for row in iter_rows(shard):
            if row["id"] in done_ids:
                continue
            if limit is not None and n + len(chunk) >= limit:
                break
            chunk.append(row)
            if len(chunk) >= args.chunk:
                run(chunk)
                chunk = []
        if chunk:
            run(chunk)
        sw.close()
        fw.close()
        summary = {"shard": stem, "worker": args.worker_id, "scored_this_run": n, "flagged_this_run": nf,
                   "resumed_from": len(done_ids), "total_scored": len(read_ids(scores_path)),
                   "total_flagged": len(read_ids(flagged_path)), "secs": round(time.time() - t0, 1), "finished": time.time()}
        if limit is None:
            write_atomic(done_path, summary)
        return summary
    finally:
        lock.release()


class Progress:
    def __init__(self, path: Path, worker: str):
        self.path, self.last = path, 0.0
        self.state = {"worker": worker, "pid": os.getpid(), "host": socket.gethostname(), "started": time.time(),
                      "items": 0, "flagged": 0, "shards_done": 0, "shard": None}

    def update(self, force=False, **kw):
        self.state.update(kw)
        self.state["t"] = time.time()
        el = max(1e-6, self.state["t"] - self.state["started"])
        self.state["items_per_hour"] = round(self.state["items"] * 3600 / el)
        if force or time.time() - self.last > 30:
            write_atomic(self.path, self.state)
            self.last = time.time()


def make_scorer(args):
    if args.backend == "fake":
        return FakeScorer(accuracy=args.fake_accuracy)
    bundle = Path(args.model_bundle)
    if args.backend == "mlx":
        return MLXScorer(bundle, Path(args.adapter or ROOT.parent / "imajev-release/hf/imajev-4b/mlx"), args.max_input_tokens)
    model_path = args.model_path or json.loads(bundle.read_text())["path"]
    return TorchScorer(model_path, Path(args.adapter or ROOT.parent / "imajev-release/hf/imajev-4b"), args.max_input_tokens,
                       args.batch_tokens, args.max_batch, args.image_batch, device=args.device)


def run(args, scorer=None) -> list[dict]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(p) for p in glob.glob(args.shards))
    if not shards:
        print(f"no shards match {args.shards}", flush=True)
        return []
    scorer = scorer or make_scorer(args)
    progress = Progress(out / "progress" / f"{args.worker_id}.json", args.worker_id)
    progress.update(force=True, shards_total=len(shards))
    summaries = []
    # Each worker starts at a different offset so workers do not all contend for shard 0.
    k = int(hashlib.sha1(args.worker_id.encode()).hexdigest()[:6], 16) % len(shards)
    for shard in shards[k:] + shards[:k]:
        s = process_shard(scorer, shard, out, args, progress)
        if s is not None:
            summaries.append(s)
            progress.update(force=True, shards_done=progress.state["shards_done"] + 1)
            print(f"[{args.worker_id}] {s['shard']}: {s['scored_this_run']} scored, {s['flagged_this_run']} flagged, "
                  f"{s['secs']}s", flush=True)
    progress.update(force=True, shard=None, finished=True)
    return summaries


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("torch", "mlx", "fake"), default="torch")
    ap.add_argument("--shards", default=str(ROOT / "data/p3/pool/shards/*.jsonl"), help="glob of pool shards")
    ap.add_argument("--out", default=str(ROOT / "data/p3/mine"))
    ap.add_argument("--data-root", default=str(ROOT / "data"), help="image paths in rows are relative to this")
    ap.add_argument("--model-bundle", default=str(ROOT / "artifacts/model-qwen4b.json"),
                    help="base model bundle json (pod: artifacts/model-qwen4b.json; Mac: artifacts/model-qwen4b-local.json)")
    ap.add_argument("--model-path", help="torch: base model snapshot dir (pod: $(cat model_path)); overrides the bundle")
    ap.add_argument("--adapter", help="shipped imajev-4b adapter dir (torch: PEFT dir; mlx: the mlx/ subdir)")
    ap.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    ap.add_argument("--device", default=None, help="torch device (default: cuda when available; cpu for a local check - "
                    "never mps, where left-padded batches give NaNs)")
    ap.add_argument("--max-input-tokens", type=int, default=32768)
    ap.add_argument("--batch-tokens", type=int, default=65536, help="padded tokens per torch batch (longest x size)")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--image-batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256, help="rows scored (and fsynced) per step")
    ap.add_argument("--order-sample", type=float, default=ORDER_SAMPLE)
    ap.add_argument("--low-conf", type=float, default=LOW_CONF)
    ap.add_argument("--limit", type=int, default=None, help="score at most N new rows per shard and do not mark it done (smoke)")
    ap.add_argument("--fake-accuracy", type=float, default=None, help="fake backend: share of items where the gold is on top")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.backend == "mlx" and args.model_bundle.endswith("model-qwen4b.json"):
        args.model_bundle = str(ROOT / "artifacts/model-qwen4b-local.json")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
