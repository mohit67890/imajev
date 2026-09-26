"""Phase-3 pod: scripts/train_decision_lora_torch.py --resume-from (continue the winning pilot lane into the full run).

The REAL trainer runs on the CPU with a tiny stand-in for torch_decision.TorchDecision (a PEFT LoRA on a 2-layer module, a
float readout; no model download). Multi-rank runs use torchrun with gloo. The stand-in logs which records every training
micro-batch held, per rank, so the tests check the data position exactly, not only the weights.

  reference   1 rank x accumulate 4, uninterrupted                          (per step: 4 micro-batches)
  pilot lane  2 ranks x accumulate 2, --max-steps 3                         (the pod: 2 GPUs x PILOT_ACC)
  full run    4 ranks x accumulate 1, --resume-from <pilot>/last            (the pod: 8 GPUs x ACC 1)
The resumed run must train the reference's remaining steps on the same micro-batches and end on the same weights.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from safetensors.torch import load_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_decision_lora_torch.py"

FAKE_ENGINE = r'''
"""Test stand-in for scripts/torch_decision.py (same interface the trainer uses)."""
import json, os, re
from pathlib import Path
import torch
TARGETS = ['proj']

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.emb = torch.nn.Embedding(512, 16); self.proj = torch.nn.Linear(16, 16)
    def forward(self, ids):
        return torch.tanh(self.proj(self.emb(ids).mean(1)))

class TorchDecision:
    def __init__(self, path, device, dtype=None, max_length=4096, pad_multiple=0):
        self.device = device; self.max_length = max_length; self.model = Tiny(); self.readout = None; self.prompt_layout = 'standard'
        self.processor = type('P', (), {'tokenizer': type('T', (), {'pad_token_id': 0})()})()
    def add_lora(self, rank=16, alpha=32, targets=None):
        from peft import LoraConfig, get_peft_model
        self.model = get_peft_model(self.model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=['proj'], init_lora_weights=False))
        return self.model
    def enable_readout(self, adapter=None, trainable=True, codes=None):
        self.readout = torch.nn.Linear(16, codes or 255, bias=False)
        self.readout.weight.requires_grad_(trainable); return True
    def save_readout(self, directory):
        from safetensors.torch import save_file
        save_file({'weight': self.readout.weight.detach().cpu().contiguous()}, str(Path(directory) / 'decision_readout.safetensors'))
        (Path(directory) / 'decision_readout.json').write_text(json.dumps({'version': 1}) + '\n')
    def labels(self, count, n_images=0):
        return [f'L{i}' for i in range(count)]
    def render_example(self, images, prompt, labels):
        return prompt, images or [], list(range(len(labels)))
    def collate(self, examples):
        rids = [int(re.search(r'RID (\d+)', e[0]).group(1)) for e in examples]
        ids = torch.tensor([[r + 1, (r * 7) % 500 + 1, (r * 13) % 500 + 1] for r in rids])
        return {'input_ids': ids}, [e[2] for e in examples], [e[3] for e in examples]
    def candidate_logits_batch(self, inputs, token_ids):
        ids = inputs['input_ids']
        if torch.is_grad_enabled():   # a training micro-batch (dev passes run under no_grad)
            with open(os.environ['FAKE_TRACE'] + '.' + os.environ.get('RANK', '0'), 'a') as f:
                f.write(json.dumps(sorted(int(x) - 1 for x in ids[:, 0])) + '\n')
        h = self.model(ids)
        return [self.readout(h[i])[:len(t)] for i, t in enumerate(token_ids)]
'''

OPTIONS = ["red", "green", "blue", "amber"]


def _record(i: int, part: str) -> dict:
    field = {"id": "q", "type": "choice", "question": "Which colour?", "options": [{"value": v} for v in OPTIONS]}
    return {"id": f"r{i:03d}", "partition": part, "target": OPTIONS[(i * 3) % 4], "abstention_cause": None, "images": [],
            "request": {"schema_version": "1.0", "request_id": f"r{i:03d}", "state": f"RID {i} item", "fields": [field]}}


@pytest.fixture()
def work(tmp_path):
    (tmp_path / "data/manifests").mkdir(parents=True)
    rows = [_record(i, "train") for i in range(40)] + [_record(100 + i, "dev") for i in range(6)]
    (tmp_path / "data/manifests/tiny.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake/torch_decision.py").write_text(FAKE_ENGINE)
    (tmp_path / "model").mkdir()
    # a wrapper: `python scripts/train_decision_lora_torch.py` would put scripts/ (the real torch_decision) first on sys.path
    (tmp_path / "run_trainer.py").write_text(f"import runpy, sys\nsys.argv[0] = {str(TRAINER)!r}\nrunpy.run_path({str(TRAINER)!r}, run_name='__main__')\n")
    return tmp_path


def train(work: Path, out: str, world: int, acc: int, *extra: str, lr: str = "0.05", check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=f"{work / 'fake'}:{ROOT / 'src'}:{ROOT / 'scripts'}", FAKE_TRACE=str(work / f"{out}.trace"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    args = [str(work / "run_trainer.py"), "--version", "tiny", "--output", str(work / out), "--model", str(work / "model"), "--device", "cpu",
            "--epochs", "2", "--lr", lr, "--warmup", "2", "--seed", "0", "--batch-size", "2", "--accumulate", str(acc), "--dev-cases", "6",
            "--dev-every", "4", "--rank", "4", "--alpha", "8", "--permute-options", "--soft-targets", *extra]
    cmd = [sys.executable] + (["-m", "torch.distributed.run", "--nproc_per_node", str(world), "--master_addr", "127.0.0.1",
                               "--master_port", str(29600 + (abs(hash(out)) % 300))] if world > 1 else []) + args
    r = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=300)
    if check:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def steps_of(work: Path, out: str, world: int, acc: int, first_step: int = 0) -> dict[int, list[int]]:
    """{optimizer step (1-based): sorted record indices trained in it}, merged over ranks."""
    by: dict[int, list[int]] = {}
    for r in range(world):
        p = work / f"{out}.trace.{r}"
        lines = [json.loads(l) for l in p.read_text().split("\n") if l.strip()] if p.exists() else []
        for k, rids in enumerate(lines):
            by.setdefault(first_step + k // acc + 1, []).extend(rids)
    return {s: sorted(v) for s, v in by.items()}


def weights(d: Path) -> dict:
    w = load_file(str(d / "adapter_model.safetensors"))
    w["readout"] = load_file(str(d / "decision_readout.safetensors"))["weight"]
    return w


def assert_same_weights(a: dict, b: dict, tol: float = 2e-5):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=tol, rtol=1e-4), (k, float((a[k] - b[k]).abs().max()))


def test_pilot_lane_resumes_exactly_on_more_ranks(work):
    # 40 train rows, batch 2 -> 20 micro-batches per epoch, 2 epochs -> 40; 4 per step -> 10 steps
    train(work, "ref", 1, 4)
    train(work, "lane", 2, 2, "--max-steps", "3", "--select", "dev_loss")
    lane_state = torch.load(work / "lane/last/trainer.pt")
    assert lane_state["step"] == 3
    train(work, "full", 4, 1, "--resume-from", str(work / "lane/last"), "--select", "mean_accuracy", "--workers", "0")

    info = json.loads((work / "full/resumed_from.json").read_text())
    assert info["mode"] == "exact" and info["step"] == 3 and info["source_microbatches"] == 12 and info["first_microbatch"] == 12
    assert info["repeated_microbatches"] == 0 and info["skipped_microbatches"] == 0 and info["of"] == 10

    ref = steps_of(work, "ref", 1, 4)
    lane = steps_of(work, "lane", 2, 2)
    full = steps_of(work, "full", 4, 1, first_step=3)
    assert sorted(ref) == list(range(1, 11)) and sorted(lane) == [1, 2, 3] and sorted(full) == list(range(4, 11))
    # every optimizer step holds exactly the reference's micro-batches (balancing reorders only inside a step)
    for s in range(1, 11):
        assert (lane if s <= 3 else full)[s] == ref[s], s
    # epoch 1 = the first 20 micro-batches = steps 1-5: every train row exactly once, none twice, none skipped
    epoch1 = Counter(x for s in range(1, 6) for x in (lane if s <= 3 else full)[s])
    assert set(epoch1) == set(range(40)) and set(epoch1.values()) == {1}
    assert_same_weights(weights(work / "full/last"), weights(work / "ref/last"))
    # the resumed log starts with the resume entry (no step-0 dev pass, so the snapshot watcher sees nothing new)
    first = json.loads((work / "full/log.jsonl").read_text().split("\n")[0])
    assert first["resumed_from"] and "dev_loss" not in first
    # config.json of the resumed run is the one a fresh full run writes (no resume keys)
    assert "resume_from" not in json.loads((work / "full/config.json").read_text())


def test_resumed_run_restarts_from_its_own_last(work):
    train(work, "lane", 2, 2, "--max-steps", "3")
    train(work, "full", 1, 4, "--resume-from", str(work / "lane/last"), "--max-steps", "2")      # stops at step 5
    assert torch.load(work / "full/last/trainer.pt")["step"] == 5
    r = train(work, "full", 1, 4, "--resume-from", str(work / "lane/last"))                         # the pod's retry: same command
    assert "Resumed at step 5" in r.stdout and "Resumed from" not in r.stdout
    train(work, "ref", 1, 4)
    assert_same_weights(weights(work / "full/last"), weights(work / "ref/last"))


def test_resume_refuses_a_different_recipe_or_per_step(work):
    train(work, "lane", 2, 2, "--max-steps", "3")
    r = train(work, "bad-lr", 1, 4, "--resume-from", str(work / "lane/last"), lr="0.01", check=False)
    assert r.returncode != 0 and "differs in ['lr']" in (r.stdout + r.stderr)
    assert not (work / "bad-lr/resumed_from.json").exists() and not (work / "bad-lr/last").exists()
    r = train(work, "bad-per-step", 1, 3, "--resume-from", str(work / "lane/last"), check=False)
    assert r.returncode != 0 and "--resume-inexact" in (r.stdout + r.stderr)
    r = train(work, "no-source", 1, 4, "--resume-from", str(work / "missing/last"), check=False)
    assert r.returncode != 0


def test_inexact_resume_repeats_less_than_one_step_and_skips_nothing(work):
    # lane: 4 micro-batches per step, 2 steps -> micro-batches 0-7 trained. Full run: 5 per step -> starts at step 1
    # (micro-batch 5): micro-batches 5-7 are trained twice (3 < one step of 5), nothing is skipped.
    train(work, "lane", 2, 2, "--max-steps", "2")
    train(work, "full", 1, 5, "--resume-from", str(work / "lane/last"), "--resume-inexact", "--max-steps", "1")
    info = json.loads((work / "full/resumed_from.json").read_text())
    assert info["mode"] == "inexact" and info["step"] == 1 and info["first_microbatch"] == 5 and info["repeated_microbatches"] == 3
    assert info["skipped_microbatches"] == 0 and info["of"] == 8
    st = torch.load(work / "full/last/trainer.pt")
    assert st["step"] == 2 and st["scheduler"]["last_epoch"] == 2       # this run's schedule (the lane's would be at 3 by now)
    train(work, "full", 1, 5, "--resume-from", str(work / "lane/last"), "--resume-inexact")      # retry: continues its own last/
    train(work, "ref5", 1, 5)
    got = Counter(x for v in steps_of(work, "lane", 2, 2).values() for x in v) + Counter(x for v in steps_of(work, "full", 1, 5).values() for x in v)
    want = Counter(x for v in steps_of(work, "ref5", 1, 5).values() for x in v)
    assert not (want - got)                                 # nothing the uninterrupted 5-per-step run trains is missing
    assert sum((got - want).values()) == 3 * 2              # the 3 repeated micro-batches x batch size 2, nothing else extra


def test_default_run_is_unchanged_by_the_new_flags(work):
    src = TRAINER.read_text()
    # config.json never records the resume flags, so existing outputs keep resuming and a resumed run's config equals a fresh one
    assert "'max_steps','resume_from','resume_inexact')" in src
    assert "parser.add_argument('--resume-from',default=''" in src and "parser.add_argument('--resume-inexact',action='store_true'" in src
    r = subprocess.run([sys.executable, str(TRAINER), "--output", str(work / "x"), "--model", "m", "--resume-inexact"], capture_output=True, text=True)
    assert r.returncode != 0 and "--resume-inexact needs --resume-from" in r.stderr
