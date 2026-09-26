"""Phase-3 eval lane for ONE checkpoint on ONE GPU (docs/phase-3-plan.md Stage 4; cloud/p3/pod_run_train.sh runs one per GPU).

Panels (each cached: a rerun skips what is already on disk):
  selection   own held-out hard set, fresh-seed half (the target) and flagged half, + the human/Kimi dev slice
              -> <out>/selection.json  (the ONLY file besides gates.json that the checkpoint selector reads)
  gates       decision-p2b test (unknown / false abstention), state + pairs probes, irrelevance test, ImajevBench v2.0-lite test
              -> <out>/gates.json with the phase-2c rules (cloud/p2c_gates_reference.json, size 4b; unchanged tolerances),
                 plus the phase-3 joint target (>= 99/122) reported beside them. No soup here: the pod script soups only if
                 no checkpoint passes.
  calibration one temperature PER ANSWER TYPE (choice / noul / score) fitted by held-out NLL on the FLAGGED held-out half,
              single pass and rot4 separately (fitted T < 1 ships as 1.0, the phase-2c rule), plus the authored-dev single-T fit;
              the per-type fit is kept only if pooled public JevBench ECE (all tiers, single pass re-tempered offline) stays
              <= 0.03, else the authored-dev fit is used (plan Stage 4).
  jevbench    JevBench public (easy/original/hard) single pass raw, single pass calibrated, rot4 raw, rot4 calibrated
  fastdec     fastino/fast-decisions dev (off-distribution check; same request shapes as
              reports/benchmarks/fast-decisions/imajev-4b/run_fastdec.py)
  decisionbench  the 3k stratified subset (scripts/p3/make_decisionbench_subset.py), single pass raw, TRACKING ONLY:
              written to <tracking-root>/<name>/decisionbench.json, a directory the selector never opens.

    python scripts/p3/eval_checkpoint.py --ckpt <adapter dir> --name r1-s000420 --out <eval root>/r1-s000420 --gpu 3 --port 8768 \
        --model <Qwen3.5-4B snapshot> --tracking-root p3/tracking-only --panels all
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vision_decision.calibration import BUCKET_LABELS, TemperatureCalibrator, _nll, fit_temperature  # noqa: E402

UNK = (None, "unknown", "__unknown__", "null")
TYPE_OF = {"choice": "choice", "boolean": "noul", "noul": "noul", "ordinal": "score", "score": "score"}      # evaluator / JevBench -> plan names
CAL_TYPE = {"choice": "choice", "noul": "boolean", "score": "ordinal"}                                      # plan names -> calibration key names
POOLED_ECE_MAX = 0.03
HARD_ECE_TARGET = 0.08
MIN_ROWS_PER_TYPE = 30
JEV_TIERS = ("easy", "original", "hard")
GATE_PANELS = {"p2b_test": ("decision-p2b", "test"), "state_probe": ("decision-v2-state-probe", "test"),
               "pairs_probe": ("decision-v2-pairs-probe", "test"), "irrelevance": ("decision-v1.1-irrelevance-test", "test")}
AUTHORED_DEV = ("decision-p2b-jevstyle-dev", "dev")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().split("\n") if l.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1) + "\n"); tmp.rename(path)


def accuracy(rows) -> dict:
    n = len(rows); c = sum(bool(r["correct"]) for r in rows)
    return {"acc": 100.0 * c / n if n else None, "correct": c, "n": n}


def by_family(rows) -> dict:
    """Accuracy per family (prediction rows carry `family` when the manifest has it)."""
    fam: dict = {}
    for r in rows:
        fam.setdefault(r.get("family") or "?", []).append(r)
    return {k: accuracy(v) for k, v in sorted(fam.items())}


def spec(value: str) -> tuple[str, str]:
    """'manifest[:partition]' -> (manifest, partition); partition defaults to test."""
    name, _, part = value.partition(":")
    return name, part or "test"


# ------------------------------------------------------------------------------------------------ temperatures and ECE
def ece(conf, correct, bins: int = 10) -> float:
    """Top-label ECE with equal-width bins (lo, hi], as the JevBench summary computes it (a 0 confidence falls in bin 0)."""
    n = len(conf)
    if not n:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return total


def fit_per_type(rows: list[dict], version: str, clamp_min: float = 1.0, min_rows: int = MIN_ROWS_PER_TYPE) -> dict:
    """One temperature per answer type (choice / noul / score) by NLL on evaluator rows (logits incl. unknown, target_index).

    A type with fewer than min_rows rows gets the pooled temperature. Fitted T < clamp_min ships as clamp_min (phase-2c rule).
    Returns a schema-1.0 TemperatureCalibrator payload (every bucket of a type carries that type's T) plus a `fit` block.
    """
    by = collections.defaultdict(list)
    for r in rows:
        t = TYPE_OF.get(r["decision_type"])
        if t is None:
            raise ValueError(f"{r.get('id')}: unknown decision_type {r['decision_type']!r}")
        if len(r["logits"]) != r["option_count"] + 1:
            raise ValueError(f"{r.get('id')}: option_count must equal len(logits) - 1")
        by[t].append(([float(x) for x in r["logits"]], int(r["target_index"])))
    pooled = [s for v in by.values() for s in v]
    if not pooled:
        raise ValueError("no calibration rows")
    t_pooled = fit_temperature(pooled)
    temps, counts, fit = {}, {}, {"rule": f"per-type NLL fit on the flagged held-out half; types with < {min_rows} rows use the pooled T; "
                                           f"fitted T < {clamp_min} ships as {clamp_min}", "pooled_fitted": t_pooled, "types": {}}
    for t in ("choice", "noul", "score"):
        samples = by.get(t, [])
        fitted = fit_temperature(samples) if len(samples) >= min_rows else t_pooled
        shipped = max(clamp_min, fitted)
        s = samples or pooled
        fit["types"][t] = {"rows": len(samples), "fitted": fitted, "shipped": shipped, "source": "own" if len(samples) >= min_rows else "pooled",
                           "nll_raw": _nll(0.0, s), "nll_calibrated": _nll(math.log(shipped), s)}
        for label in BUCKET_LABELS:
            key = f"{CAL_TYPE[t]}:{label}"
            temps[key] = shipped; counts[key] = max(1, len(samples))
    payload = TemperatureCalibrator(version, temps, counts, None).to_dict()
    payload["fit"] = fit
    TemperatureCalibrator.from_dict(payload)  # round-trip check
    return payload


def type_temperature(payload: dict, qtype: str) -> float:
    key = f"{CAL_TYPE[qtype]}:{BUCKET_LABELS[0]}"
    return float(payload["temperatures"].get(key) or next(iter(payload["temperatures"].values())))


def retemper(probs: dict, t: float) -> dict:
    """Re-temper a returned distribution offline: p_i^(1/T) renormalised (== logits / T over the returned candidates)."""
    lg = {k: math.log(max(float(v), 1e-12)) / t for k, v in probs.items()}
    m = max(lg.values()); z = sum(math.exp(v - m) for v in lg.values())
    return {k: math.exp(v - m) / z for k, v in lg.items()}


def jevbench_types(jevbench_dir: Path) -> dict:
    out = {}
    for tier in JEV_TIERS:
        p = jevbench_dir / "datasets/public" / f"{tier}.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                out[r["id"]] = TYPE_OF.get(r["question"].get("type"), "choice")
    return out


def jevbench_conf(results_dir: Path, types: dict, payload: dict | None, tiers=JEV_TIERS):
    """(confidence, correct) per scorable JevBench row, optionally re-tempered per type with a calibration payload."""
    conf, cor, per = [], [], {}
    for tier in tiers:
        p = results_dir / tier / "results.jsonl"
        if not p.exists():
            continue
        c_t, k_t = [], []
        for r in read_jsonl(p):
            probs = r.get("probs")
            if not probs:
                continue
            q = retemper(probs, type_temperature(payload, types.get(r["task_id"], "choice"))) if payload else probs
            top = max(q, key=q.get)
            c_t.append(q[top]); k_t.append(bool(r.get("correct")) and top == r.get("predicted"))
        per[tier] = (c_t, k_t); conf += c_t; cor += k_t
    return conf, cor, per


def pooled_ece_guard(results_dir: Path, types: dict, payload: dict | None) -> dict:
    conf, cor, per = jevbench_conf(results_dir, types, payload)
    return {"pooled_ece": ece(conf, cor) if conf else None, "rows": len(conf),
            "per_tier": {t: (ece(c, k) if c else None) for t, (c, k) in per.items()},
            "pass": bool(conf) and ece(conf, cor) <= POOLED_ECE_MAX}


def choose_calibration(per_type: dict | None, authored: dict | None, guard_per_type: dict | None) -> tuple[str, dict | None, str]:
    """Plan Stage 4: keep the hard-set (per-type) fit unless it raises pooled public ECE above 0.03; then the authored-dev fit."""
    if per_type is not None and guard_per_type is not None and guard_per_type.get("pass"):
        return "per_type", per_type, f"per-type fit keeps pooled public ECE {guard_per_type['pooled_ece']:.4f} <= {POOLED_ECE_MAX}"
    if authored is not None:
        why = "no per-type fit" if per_type is None else f"per-type fit gives pooled public ECE {guard_per_type and guard_per_type.get('pooled_ece')} > {POOLED_ECE_MAX}"
        return "authored_dev", authored, why + " -> authored-dev single T"
    if per_type is not None:
        return "per_type", per_type, "authored-dev fit unavailable; per-type fit kept although the pooled guard failed"
    return "none", None, "no calibration available"


# ------------------------------------------------------------------------------------------------ gates (phase-2c rules)
def imajevbench_tracks(score_path: Path) -> dict | None:
    if not score_path.exists():
        return None
    s = json.loads(score_path.read_text())
    return {t: [sum(f["correct"] for f in v["families"].values()), sum(f["total"] for f in v["families"].values())]
            for t, v in s["capability"]["tracks"].items()}


def compute_gates(panels: dict, tracks: dict | None, ref: dict, size: str = "4b", joint_target: int = 99) -> dict:
    """panels: name -> prediction rows (or None). Same rules and tolerances as cloud/pod_run_p2c_train.sh gates()."""
    tol, R = ref["tolerances"], ref["sizes"][size]
    checks = []

    def check(name, value, op, bound, detail=""):
        if value is None or bound is None:
            checks.append({"gate": name, "value": value, "bound": bound, "pass": False,
                           "note": "missing measurement" if value is None else "missing reference"}); return
        v, b = round(value, 2), round(bound, 2)
        ok = v >= b - 1e-9 if op == ">=" else v <= b + 1e-9
        checks.append({"gate": name, "value": round(value, 2), "op": op, "bound": round(bound, 2), "pass": ok, "note": detail})

    acc = lambda rows: 100.0 * sum(r["correct"] for r in rows) / len(rows) if rows else None
    if tracks:
        tot = sum(v[0] for v in tracks.values()); n = sum(v[1] for v in tracks.values())
        check("a.imajevbench_acc", 100.0 * tot / n, ">=", R["imajevbench_acc"] - tol["imajevbench_acc_points"], f"{tot}/{n}")
        for t in ("visual", "joint"):
            check(f"a.{t}_items", (tracks.get(t) or [None])[0], ">=", R[t][0] - tol["track_items"], f"ref {R[t][0]}/{R[t][1]}")
    else:
        for g in ("a.imajevbench_acc", "a.visual_items", "a.joint_items"):
            check(g, None, ">=", 0)
    check("b.state_probe", acc(panels.get("state_probe") or []), ">=", R["state_probe"] - tol["probe_points"])
    check("b.pairs_probe", acc(panels.get("pairs_probe") or []), ">=", R["pairs_probe"] - tol["probe_points"])
    rows = panels.get("p2b_test")
    if rows:
        unk = [r for r in rows if r.get("target") in UNK]; ans = [r for r in rows if r.get("target") not in UNK]
        cu = 100.0 * sum(r.get("prediction") in UNK for r in unk) / len(unk) if unk else None
        fa = 100.0 * sum(r.get("prediction") in UNK for r in ans) / len(ans) if ans else None
        check("c.unknown_correct_rate", cu, ">=", R.get("unknown_correct_rate"), f"{len(unk)} unknown-gold rows")
        fr = R.get("false_abstention_rate")
        check("c.false_abstention_rate", fa, "<=", None if fr is None else fr + tol["false_abstention_points"], f"{len(ans)} answerable rows")
    else:
        check("c.unknown_correct_rate", None, ">=", 0); check("c.false_abstention_rate", None, "<=", 0)
    check("d.irrelevance", acc(panels.get("irrelevance") or []), ">=", R["irrelevance"] - tol["irrelevance_points"])
    ok = all(c["pass"] for c in checks)
    joint = (tracks or {}).get("joint")
    return {"size": size, "shippable": ok, "image_gate": all(c["pass"] for c in checks if c["gate"].startswith("a.")),
            "checks": checks, "targets": {"joint_items": {"value": joint and joint[0], "target": joint_target,
                                                          "met": bool(joint and joint[0] >= joint_target)}},
            "reference": R, "tolerances": tol, "soup": False}


# ------------------------------------------------------------------------------------------------ subprocess helpers
class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}", CUDA_VISIBLE_DEVICES=str(a.gpu),
                        TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
        self.status = json.loads((self.out / "status.json").read_text()) if (self.out / "status.json").exists() else {}

    def mark(self, panel, ok, seconds, note=""):
        self.status[panel] = {"ok": ok, "seconds": round(seconds, 1), "note": note[-400:]}
        write_json(self.out / "status.json", self.status)

    def run(self, cmd, logf: Path, cwd=ROOT, env=None) -> bool:
        logf.parent.mkdir(parents=True, exist_ok=True)
        with open(logf, "a") as f:
            return subprocess.run(cmd, cwd=cwd, env=env or self.env, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def evaluate(ctx: Ctx, key: str, version: str, partition: str) -> list[dict] | None:
    """Batched single-pass predictions of one manifest partition (scripts/evaluate_decision_model_torch.py, max length raised)."""
    d = ctx.out / "panels" / key; pred = d / "predictions.jsonl"
    if pred.exists():
        return read_jsonl(pred)
    if not (ROOT / "data/manifests" / f"{version}.jsonl").exists():
        log(f"  {key}: manifest {version} missing"); ctx.mark(key, False, 0, f"manifest {version} missing"); return None
    t = time.monotonic()
    ok = ctx.run([sys.executable, __file__, "_evaluate", "--max-length", str(ctx.a.max_length), "--", "--version", version, "--partition", partition,
                  "--output", str(d), "--model", ctx.a.model, "--adapter", ctx.a.ckpt, "--token-budget", str(ctx.a.token_budget)], d / "eval.log")
    ctx.mark(key, ok and pred.exists(), time.monotonic() - t, "" if ok else (d / "eval.log").read_text()[-400:])
    if not (ok and pred.exists()):
        return None
    rows = read_jsonl(pred); log(f"  {key}: {len(rows)} rows, acc {accuracy(rows)['acc']:.2f}")
    return rows


class Server:
    """scripts/playground/server.py (torch) in its own process group on this GPU; stopped by group kill."""

    def __init__(self, ctx: Ctx, tag: str, rotations: int = 1, calibration: str | None = None):
        self.ctx, self.tag, self.rot, self.cal = ctx, tag, rotations, calibration
        self.port = ctx.a.port; self.proc = None

    def __enter__(self):
        d = self.ctx.out / "servers"; d.mkdir(parents=True, exist_ok=True)
        bundle = d / "bundle.json"; bundle.write_text(json.dumps({"repo": "local", "revision": "local", "path": self.ctx.a.model}))
        cmd = [sys.executable, str(ROOT / "scripts/playground/server.py"), "--backend", "torch", "--model-bundle", str(bundle), "--adapter", self.ctx.a.ckpt,
               "--host", "127.0.0.1", "--port", str(self.port), "--rotations", str(self.rot), "--max-input-tokens", str(self.ctx.a.server_max_tokens),
               "--model-name", f"imajev-p3-{self.ctx.a.name}"] + (["--calibration", self.cal] if self.cal else [])
        self.logf = open(d / f"{self.tag}.log", "a")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=self.ctx.env, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.tag} exited (see {d / (self.tag + '.log')})")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2); return self
            except (urllib.error.URLError, OSError):
                time.sleep(3)
        raise RuntimeError(f"server {self.tag} did not come up")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(3)
        self.logf.close()


def post(port: int, req: dict, timeout: float = 600) -> dict:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def run_jevbench(ctx: Ctx, variant: str, port: int) -> dict:
    R = ctx.out / "jevbench" / variant; res = {}
    jb = Path(ctx.a.jevbench)
    env = dict(ctx.env, TYPESAFE_API_KEY="local")
    for tier in JEV_TIERS:
        d = R / tier; d.mkdir(parents=True, exist_ok=True)
        if not (d / "summary.json").exists():
            ctx.run([sys.executable, "-m", "jevbench.cli", "run", "--tasks", f"datasets/public/{tier}.jsonl", "--adapter", "typesafe",
                     "--endpoint", f"http://127.0.0.1:{port}", "--model", f"imajev-p3-{ctx.a.name}-{variant}", "--price-in-per-m", "0",
                     "--price-out-per-m", "0", "--cap-usd", "1", "--results", str(d / "results.jsonl"), "--raw-dir", str(d / "raw"),
                     "--manifest", str(d / "manifest.json")], d / "run.log", cwd=jb, env=env)
            ctx.run([sys.executable, "-m", "jevbench.cli", "summarize", "--tasks", f"datasets/public/{tier}.jsonl", "--results", str(d / "results.jsonl"),
                     "--public-export", str(d / "summary.json")], d / "summarize.log", cwd=jb, env=env)
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            res[tier] = {"acc": 100 * s["accuracy"], "ece": s["ece"]["ece"], "n": s["ece"].get("n")}
    types = jevbench_types(jb)
    conf, cor, _ = jevbench_conf(R, types, None)
    res["pooled_ece"] = ece(conf, cor) if conf else None
    write_json(R / "result.json", res)
    log(f"  jevbench {variant}: " + ", ".join(f"{t} {v['acc']:.1f}/{v['ece']:.3f}" for t, v in res.items() if isinstance(v, dict)) +
        (f", pooled ECE {res['pooled_ece']:.3f}" if res.get("pooled_ece") is not None else ""))
    return res


# ------------------------------------------------------------------------------------------------ DecisionBench (tracking only)
def decisionbench_request(row: dict) -> tuple[dict, list[str]]:
    """The benchmark's own SystemOne mapping (decision_bench.models.jev_openrouter.build_jev_request, MIT)."""
    cands = json.loads(row["candidates_json"]) if isinstance(row.get("candidates_json"), str) else row["candidates"]
    state = json.loads(row["state_json"]) if isinstance(row.get("state_json"), str) else row["state"]
    desc = lambda c: c["label"] if c.get("description") is None else f"{c['label']}: {c['description']}"
    if row["primitive"] == "binary_classification":
        q = {"type": "noul", "instructions": row["instruction"], "criteria": {"true": desc(cands[0]), "false": desc(cands[1])}}
        order = [c["id"] for c in cands]
    elif row["primitive"] == "candidate_selection":
        q = {"type": "choice", "instructions": row["instruction"], "criteria": {c["id"]: desc(c) for c in cands}}
        order = [c["id"] for c in cands]
    else:
        ordered = sorted(cands, key=lambda c: float(c.get("ordinal_value") or 0.0))
        q = {"type": "score", "instructions": row["instruction"], "criteria": [desc(c) for c in ordered]}
        order = [c["id"] for c in ordered]
    return {"model": "imajev", "state": state, "questions": {"decision": q}}, order


def decisionbench_probs(row: dict, answer: dict, order: list[str]) -> list[float]:
    cands = json.loads(row["candidates_json"]) if isinstance(row.get("candidates_json"), str) else row["candidates"]
    if row["primitive"] == "binary_classification":
        yes = float(answer["noul"]); return [yes if c["id"] == cands[0]["id"] else 1.0 - yes for c in cands]
    raw = answer["probabilities"]
    by = {cid: float(raw[cid]) for cid in order} if row["primitive"] == "candidate_selection" else {cid: float(raw[str(i)]) for i, cid in enumerate(order)}
    return [by[c["id"]] for c in cands]


def summarize_decisionbench(rows: list[dict], full_counts: dict | None = None) -> dict:
    by_task, by_prim, by_fam = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r["correct"]); by_prim[r["primitive"]].append(r["correct"]); by_fam[r["family"]].append(r["correct"])
    pct = lambda v: 100.0 * sum(v) / len(v) if v else None
    out = {"rows": len(rows), "primary_accuracy": pct([r["correct"] for r in rows]), "errors": sum(bool(r.get("error")) for r in rows),
           "unsupported": sum(bool(r.get("unsupported")) for r in rows),
           "macro_task_accuracy": sum(pct(v) for v in by_task.values()) / len(by_task) if by_task else None,
           "by_primitive": {k: pct(v) for k, v in sorted(by_prim.items())}, "by_family": {k: pct(v) for k, v in sorted(by_fam.items())},
           "by_task": {k: pct(v) for k, v in sorted(by_task.items())},
           "ordinal": pct(by_prim.get("ordinal_scoring", [])), "reasoning": pct(by_fam.get("reasoning", []))}
    if full_counts:
        tot = sum(full_counts[k] for k in by_task if k in full_counts)
        out["full_suite_equivalent"] = sum(pct(v) * full_counts[k] for k, v in by_task.items() if k in full_counts) / tot if tot else None
    return out


def run_decisionbench(ctx: Ctx, port: int) -> dict | None:
    sub = Path(ctx.a.decisionbench_subset)
    out = Path(ctx.a.tracking_root) / ctx.a.name / "decisionbench.json"
    if out.exists():
        return json.loads(out.read_text())
    if not sub.exists():
        log("  decisionbench: subset missing (tracking only; skipped)"); return None
    rows = read_jsonl(sub)
    meta_p = sub.with_name(sub.stem + ".meta.json"); full = json.loads(meta_p.read_text())["full_counts"] if meta_p.exists() else None

    def one(row):
        rec = {"row_id": row["row_id"], "task_id": row["task_id"], "primitive": row["primitive"], "family": row["family"], "correct": False}
        try:
            req, order = decisionbench_request(row)
            probs = decisionbench_probs(row, post(port, req)["answers"]["decision"], order)
            cands = json.loads(row["candidates_json"])
            rec["correct"] = cands[max(range(len(probs)), key=probs.__getitem__)]["id"] == row["gold_candidate_id"]
        except urllib.error.HTTPError as e:
            rec["error"] = f"HTTP {e.code}"; rec["unsupported"] = e.code in (400, 413, 422)
        except Exception as e:  # counted as a miss (primary accuracy rule)
            rec["error"] = repr(e)[:200]
        return rec

    with ThreadPoolExecutor(ctx.a.concurrency) as pool:
        preds = list(pool.map(one, rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / "decisionbench-rows.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    summ = summarize_decisionbench(preds, full); summ["rule"] = "tracking only; never read by the checkpoint selector"
    write_json(out, summ)
    log(f"  decisionbench (tracking only): written to {out}")
    return summ


# ------------------------------------------------------------------------------------------------ fast-decisions dev
def fastdec_build(row: dict):
    """Same request shapes as reports/benchmarks/fast-decisions/imajev-4b/run_fastdec.py."""
    human = lambda t: t.replace("_", " ")
    questions, plan = {}, []
    for i, h in enumerate(row["output"]["classifications"]):
        if h["multi_label"]:
            ids = []
            for j, lab in enumerate(h["labels"]):
                qid = f"h{i}_l{j}"
                questions[qid] = {"type": "noul", "instructions": f"Does the {human(h['task'])} label \"{lab}\" apply to this text?"}
                ids.append((qid, lab))
            plan.append((h, ids))
        else:
            qid = f"h{i}"
            questions[qid] = {"type": "choice", "instructions": f"Which {human(h['task'])} label fits this text?", "criteria": {lab: None for lab in h["labels"]}}
            plan.append((h, qid))
    return {"model": "imajev", "state": row["input"], "questions": questions}, plan


def fastdec_heads(plan, answers) -> list[dict]:
    heads = []
    for h, q in plan:
        if isinstance(q, list):
            probs = {lab: answers[qid]["noul"] for qid, lab in q}
            pred = [l for l, p in probs.items() if p > 0.5] or [max(probs, key=probs.get)]
        else:
            pred = [answers[q]["choice"]]
        heads.append({"task": h["task"], "multi_label": h["multi_label"], "correct": sorted(map(str, pred)) == sorted(map(str, h["true_label"]))})
    return heads


def run_fastdec(ctx: Ctx, port: int) -> dict | None:
    out = ctx.out / "fastdec" / "result.json"
    if out.exists():
        return json.loads(out.read_text())
    d = Path(ctx.a.fastdec_dir)
    files = sorted(p for p in d.glob("*.jsonl")) if d.is_dir() else []
    if not files:
        log("  fastdec: dev files missing (skipped)"); return None

    def one(item):
        dom, row = item
        req, plan = fastdec_build(row); items = list(req["questions"].items()); answers = {}
        try:
            for k in range(0, len(items), 8):
                answers.update(post(port, {**req, "questions": dict(items[k:k + 8])})["answers"])
            return dom, fastdec_heads(plan, answers)
        except Exception:
            return dom, [{"task": h["task"], "multi_label": h["multi_label"], "correct": False, "error": True} for h, _ in plan]

    work = [(p.stem, json.loads(l)) for p in files for l in p.read_text().split("\n") if l.strip()]
    with ThreadPoolExecutor(ctx.a.concurrency) as pool:
        res = list(pool.map(one, work))
    dom = collections.defaultdict(lambda: [0, 0])
    for dname, heads in res:
        for h in heads:
            dom[dname][0] += h["correct"]; dom[dname][1] += 1
    accs = {k: 100.0 * c / n for k, (c, n) in dom.items()}
    C = sum(v[0] for v in dom.values()); H = sum(v[1] for v in dom.values())
    summ = {"rows": len(work), "macro": sum(accs.values()) / len(accs), "pooled_heads": 100.0 * C / H, "heads": H, "by_domain": accs,
            "errors": sum(any(h.get("error") for h in heads) for _, heads in res)}
    write_json(out, summ); log(f"  fastdec: macro {summ['macro']:.1f}, pooled heads {summ['pooled_heads']:.1f}")
    return summ


# ------------------------------------------------------------------------------------------------ ImajevBench
def run_imajevbench(ctx: Ctx) -> dict | None:
    out = ctx.out / "imajevbench"; score = out / "score"
    if not score.exists():
        recs, root = Path(ctx.a.bench_root) / "records/records-eval.jsonl", Path(ctx.a.bench_root)
        if not recs.exists():
            log("  imajevbench: records missing"); ctx.mark("imajevbench", False, 0, "records missing"); return None
        t = time.monotonic()
        ok = ctx.run([sys.executable, "scripts/imajev_bench/run_local_v2.py", "--records", str(recs), "--root", str(root), "--split", "test",
                      "--profile", "benchmark-neutral", "--rotations", "full", "--warmup", "1", "--repeats", "1", "--allow-draft", "--backend", "torch",
                      "--device", "cuda", "--gpu-coordinated", "--model", "qwen4b", "--adapter", ctx.a.ckpt, "--base-path", ctx.a.model,
                      "--output", str(out)], ctx.out / "imajevbench.log")
        ok = ok and ctx.run([sys.executable, "-m", "imajev_bench", "score", "--records", str(recs), "--root", str(root), "--split", "test", "--allow-draft",
                             "--predictions", str(out / "predictions.jsonl"), "--output", str(score)], ctx.out / "imajevbench-score.log")
        ctx.mark("imajevbench", ok and score.exists(), time.monotonic() - t)
    tracks = imajevbench_tracks(score)
    if tracks:
        tot = sum(v[0] for v in tracks.values()); n = sum(v[1] for v in tracks.values())
        log(f"  imajevbench: {tot}/{n} = {100 * tot / n:.1f}% tracks {tracks}")
    return tracks


# ------------------------------------------------------------------------------------------------ rot4 predictions on the flagged half
def rot4_predictions(ctx: Ctx, version: str, partition: str) -> list[dict] | None:
    pred = ctx.out / "panels" / "heldout_flagged_rot4" / "predictions.jsonl"
    if pred.exists():
        return read_jsonl(pred)
    t = time.monotonic()
    ok = ctx.run([sys.executable, __file__, "_rot4", "--version", version, "--partition", partition, "--model", ctx.a.model, "--adapter", ctx.a.ckpt,
                  "--out", str(pred), "--max-rows", str(ctx.a.rot4_cal_rows), "--max-length", str(ctx.a.max_length)], pred.parent / "rot4.log")
    ctx.mark("heldout_flagged_rot4", ok and pred.exists(), time.monotonic() - t)
    return read_jsonl(pred) if ok and pred.exists() else None


def _rot4_main(argv):
    """Internal: rot4 raw logits (the served combine_rotations path) for a sample of one manifest partition."""
    ap = argparse.ArgumentParser(); ap.add_argument("--version"); ap.add_argument("--partition"); ap.add_argument("--model"); ap.add_argument("--adapter")
    ap.add_argument("--out", type=Path); ap.add_argument("--max-rows", type=int, default=1200); ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    from decision_data import load_records, load_image, pixel_budget
    from vision_decision.contracts import Request
    sys.path.insert(0, str(ROOT / "scripts/playground"))
    from server import TorchBackend  # noqa: E402
    rows = load_records(a.partition, a.version)
    rows = sorted(rows, key=lambda r: r["id"]); random.Random(0).shuffle(rows); rows = rows[:a.max_rows]
    bundle = a.out.parent / "bundle.json"; a.out.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(json.dumps({"repo": "local", "revision": "local", "path": a.model}))
    be = TorchBackend(bundle, a.adapter, device="cuda", rotations=4, max_input_tokens=a.max_length)
    from decision_data import render
    from vision_decision.scoring import key as score_key
    ftype = {"choice": "choice", "boolean": "boolean", "ordinal": "ordinal"}
    with a.out.with_suffix(".tmp").open("w") as w:
        for r in rows:
            try:
                header, choices, texts, target = render(r)   # served order, no shuffle; unknown last
                target = max(range(len(target)), key=target.__getitem__) if isinstance(target, list) else int(target)
                req = Request.model_validate(r["request"])
                images = [load_image(ROOT / x["image"], pixel_budget(r, 400000)) for x in r.get("images", [])]
                res = be.score(images, req)[0][0]
                logits = [float(res.raw_logits[score_key(v)]) for v, _ in choices]
                f0 = r["request"]["fields"][0]
                w.write(json.dumps({"id": r["id"], "decision_type": ftype.get(f0["type"], f0["type"]), "option_count": len(choices) - 1,
                                    "logits": logits, "target_index": target,
                                    "correct": max(range(len(logits)), key=logits.__getitem__) == target}) + "\n")
            except Exception as e:  # a row the served contract refuses is skipped (reported in the log)
                print("skip", r.get("id"), repr(e)[:160], flush=True)
    a.out.with_suffix(".tmp").rename(a.out)
    return 0


def _evaluate_main(argv):
    """Internal: run the batched evaluator with TorchDecision's max length raised to --max-length (phase-3 long inputs)."""
    ap = argparse.ArgumentParser(); ap.add_argument("--max-length", type=int, default=16384)
    i = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:i]); rest = argv[i + 1:]
    os.chdir(ROOT)
    import runpy
    import torch_decision
    init = torch_decision.TorchDecision.__init__
    def patched(self, path, device, dtype=None, max_length=4096, pad_multiple=0):
        import torch
        init(self, path, device, dtype=dtype or torch.bfloat16, max_length=max(max_length, a.max_length), pad_multiple=pad_multiple)
    torch_decision.TorchDecision.__init__ = patched
    sys.argv = ["evaluate_decision_model_torch.py"] + rest
    runpy.run_path(str(ROOT / "scripts/evaluate_decision_model_torch.py"), run_name="__main__")
    return 0


def _load_p2_fit():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("fit_p2_temperature", ROOT / "scripts/p2/fit_p2_temperature.py")
    mod = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(mod)
    return mod.fit_single_temperature


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_evaluate":
        return _evaluate_main(argv[1:])
    if argv and argv[0] == "_rot4":
        return _rot4_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="adapter directory (PEFT LoRA + decision_readout)")
    ap.add_argument("--name", required=True, help="checkpoint name, e.g. r1-s000420, r2-last, shipped, soup-r1-s000420-w50")
    ap.add_argument("--out", required=True, help="this checkpoint's eval directory (selection.json, gates.json, summary.json)")
    ap.add_argument("--round", type=int, default=0); ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--order", type=float, default=0.0, help="training order (later = larger), used by the selector's tie rule")
    ap.add_argument("--model", required=True); ap.add_argument("--gpu", default="0"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--panels", default="all", help="selection | all")
    ap.add_argument("--skip", default="", help="comma list of: gates,imajevbench,calibration,rot4,jevbench,fastdec,decisionbench")
    ap.add_argument("--heldout-fresh", default="decision-p3-heldout-fresh:test"); ap.add_argument("--heldout-flagged", default="decision-p3-heldout-flagged:test")
    ap.add_argument("--human-dev", default="decision-p3-human-dev:test")
    ap.add_argument("--direct-groups", default="charts,docimg,inventory,safety,geometry,screens",
                    help="direct constructed image groups: <prefix><group>:<partition> each is a separate REPORT panel (skipped "
                         "quietly when its manifest is absent), never read by the selector")
    ap.add_argument("--direct-prefix", default="decision-p3-heldout-"); ap.add_argument("--direct-partition", default="test")
    ap.add_argument("--gates-ref", default=str(ROOT / "cloud/p2c_gates_reference.json")); ap.add_argument("--size", default="4b")
    ap.add_argument("--bench-root", default="bench"); ap.add_argument("--jevbench", default="evalsets/jevbench")
    ap.add_argument("--fastdec-dir", default="evalsets/fast-decisions")
    ap.add_argument("--decisionbench-subset", default="p3/tracking-only/decisionbench-3k.jsonl")
    ap.add_argument("--tracking-root", default="p3/tracking-only")
    ap.add_argument("--fixed-calibration", help="use this calibration artifact for the calibrated runs (the shipped 1.0 has its own)")
    ap.add_argument("--max-length", type=int, default=16384); ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--server-max-tokens", type=int, default=32768); ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--rot4-cal-rows", type=int, default=1200)
    ap.add_argument("--soup", action="store_true", help="this checkpoint is a soup with the shipped adapter (recorded in gates.json)")
    a = ap.parse_args(argv)
    if "tracking-only" not in Path(a.tracking_root).parts:
        ap.error("--tracking-root must be a directory named tracking-only")
    if Path(a.tracking_root).resolve() in Path(a.out).resolve().parents or "tracking-only" in Path(a.out).parts:
        ap.error("--out must not live under the tracking-only root")
    ctx = Ctx(a); skip = set(filter(None, a.skip.split(",")))
    log(f"eval {a.name} ({a.ckpt}) on GPU {a.gpu}, panels {a.panels}")

    # 1. selection panels
    fresh = evaluate(ctx, "heldout_fresh", *spec(a.heldout_fresh))
    flagged = evaluate(ctx, "heldout_flagged", *spec(a.heldout_flagged))
    human = evaluate(ctx, "human_dev", *spec(a.human_dev))
    sel = {"name": a.name, "ckpt": a.ckpt, "round": a.round, "step": a.step, "order": a.order,
           "fresh": accuracy(fresh) if fresh else None, "flagged": accuracy(flagged) if flagged else None, "human": accuracy(human) if human else None}
    write_json(ctx.out / "selection.json", sel)
    if a.panels == "selection":
        return 0 if fresh else 1

    panels = {}
    direct = {}
    if "direct" not in skip:       # report-only panels; a missing manifest (that group not in this run) is skipped quietly
        for grp in filter(None, a.direct_groups.split(",")):
            version = f"{a.direct_prefix}{grp}"
            if (ROOT / "data/manifests" / f"{version}.jsonl").exists():
                rows = evaluate(ctx, f"heldout_{grp}", version, a.direct_partition)
                if rows:
                    direct[grp] = {**accuracy(rows), "by_family": by_family(rows)}
    if "gates" not in skip:
        for key, (v, part) in GATE_PANELS.items():
            panels[key] = evaluate(ctx, key, v, part)
    try:
        tracks = run_imajevbench(ctx) if "imajevbench" not in skip else None
    except Exception as e:
        log(f"  imajevbench FAILED: {e!r}"); ctx.mark("imajevbench", False, 0, repr(e)); tracks = None

    # 2. calibration fits (single pass on the flagged half; authored-dev single T; rot4 on a flagged sample)
    cal_dir = ctx.out / "calibration"; cal_dir.mkdir(exist_ok=True)
    per_type = authored = per_type_rot4 = None
    try:
        if "calibration" not in skip:
            if flagged:
                per_type = fit_per_type(flagged, f"p3-{a.name}-pertype"); write_json(cal_dir / "single-pertype.json", per_type)
            adev = evaluate(ctx, "authored_dev", *AUTHORED_DEV)
            if adev:
                fit_single_temperature = _load_p2_fit()
                authored = fit_single_temperature(adev, f"p3-{a.name}-authored")
                if authored["fit"]["temperature"] < 1.0:
                    authored["fit"]["fitted_temperature"] = authored["fit"]["temperature"]
                    authored["temperatures"] = {k: 1.0 for k in authored["temperatures"]}
                write_json(cal_dir / "single-authored.json", authored)
            if "rot4" not in skip:
                r4 = rot4_predictions(ctx, *spec(a.heldout_flagged))
                if r4:
                    per_type_rot4 = fit_per_type(r4, f"p3-{a.name}-pertype-rot4"); write_json(cal_dir / "rot4-pertype.json", per_type_rot4)

    except Exception as e:   # a failed fit leaves that calibration out; the choice logic falls back
        log(f"  calibration FAILED: {e!r}"); ctx.mark("calibration", False, 0, repr(e))
    # 3. server passes: single raw (JevBench, fast-decisions, DecisionBench tracking), single cal, rot4 raw, rot4 cal
    types = jevbench_types(Path(a.jevbench))
    summary = {"name": a.name, "selection": sel, "calibration": {}, "jevbench": {}}
    fixed = a.fixed_calibration
    try:
        with Server(ctx, "single-raw") as s:
            if "jevbench" not in skip:
                summary["jevbench"]["single_raw"] = run_jevbench(ctx, "single-raw", s.port)
            if "fastdec" not in skip:
                summary["fastdec"] = run_fastdec(ctx, s.port)
            if "decisionbench" not in skip:
                run_decisionbench(ctx, s.port)   # tracking only: never copied into summary/selection
        if "jevbench" not in skip:
            raw_dir = ctx.out / "jevbench" / "single-raw"
            g_pt = pooled_ece_guard(raw_dir, types, per_type) if per_type else None
            g_au = pooled_ece_guard(raw_dir, types, authored) if authored else None
            if fixed:
                choice, payload, why = "fixed", json.loads(Path(fixed).read_text()), f"fixed calibration {fixed}"
            else:
                choice, payload, why = choose_calibration(per_type, authored, g_pt)
            summary["calibration"]["single"] = {"choice": choice, "why": why, "guard_per_type": g_pt, "guard_authored": g_au}
            if payload:
                cal_path = cal_dir / "single-chosen.json"; write_json(cal_path, payload)
                with Server(ctx, "single-cal", calibration=str(cal_path)) as s:
                    summary["jevbench"]["single_cal"] = run_jevbench(ctx, "single-cal", s.port)
            if "rot4" not in skip:
                with Server(ctx, "rot4-raw", rotations=4) as s:
                    summary["jevbench"]["rot4_raw"] = run_jevbench(ctx, "rot4-raw", s.port)
                r4_dir = ctx.out / "jevbench" / "rot4-raw"
                g4 = pooled_ece_guard(r4_dir, types, per_type_rot4) if per_type_rot4 else None
                if fixed:
                    c4, p4, w4 = "fixed", json.loads(Path(fixed).read_text()), f"fixed calibration {fixed}"
                else:
                    c4, p4, w4 = choose_calibration(per_type_rot4, authored, g4)
                summary["calibration"]["rot4"] = {"choice": c4, "why": w4, "guard_per_type": g4}
                if p4:
                    cal4 = cal_dir / "rot4-chosen.json"; write_json(cal4, p4)
                    with Server(ctx, "rot4-cal", rotations=4, calibration=str(cal4)) as s:
                        summary["jevbench"]["rot4_cal"] = run_jevbench(ctx, "rot4-cal", s.port)
    except Exception as e:
        log(f"  server panels FAILED: {e!r}"); ctx.mark("server_panels", False, 0, repr(e))

    # 4. gates
    if "gates" not in skip:
        ref = json.loads(Path(a.gates_ref).read_text())
        gates = compute_gates(panels, tracks, ref, a.size); gates["name"] = a.name; gates["soup"] = bool(a.soup)
        write_json(ctx.out / "gates.json", gates)
        for c in gates["checks"]:
            log(f"  gate {a.name} {c['gate']}: {'PASS' if c['pass'] else 'FAIL'} value {c['value']} {c.get('op', '')} {c['bound']} {c.get('note', '')}")
        log(f"  {a.name} {'SHIPPABLE (all gates pass)' if gates['shippable'] else 'NOT_SHIPPABLE'}; joint {gates['targets']['joint_items']}")
        summary["gates"] = {"shippable": gates["shippable"], "image_gate": gates["image_gate"], "joint": gates["targets"]["joint_items"]}
        summary["imajevbench"] = tracks
    summary["panels"] = {k: accuracy(v) if v else None for k, v in panels.items()}
    if direct:
        summary["direct_panels"] = direct                # reported separately (final table); the selector never reads it
    write_json(ctx.out / "summary.json", summary)
    log(f"done {a.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
