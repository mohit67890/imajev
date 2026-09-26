"""LoRA rank expansion (scripts/lora_expand.py, trainer --expand-lora-rank) and the rank64 pilot lane (cloud/p3/pilot_decision.py,
cloud/p3/lane_resume.py, cloud/p3/soup.py, cloud/p3/pod_run_train.sh).

The expanded adapter must compute the SAME function at step 0 (new B columns zero, alpha scaled so alpha/rank is unchanged), take
gradients in its new columns, save / reload through PEFT (what scripts/playground/server.py, scripts/evaluate_decision_model_torch.py
and scripts/p3/mine.py use: PeftModel.from_pretrained) and convert to MLX with the same effective delta."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", Path(__file__).parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
import lora_expand as LX  # noqa: E402


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = load("cloud/p3/pilot_decision.py", "p3_pilot_rx")
lane_resume = load("cloud/p3/lane_resume.py", "p3_lane_resume_rx")
soup = load("cloud/p3/soup.py", "p3_soup_rx")


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.q_proj = torch.nn.Linear(12, 10)
        self.down_proj = torch.nn.Linear(10, 6)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.q_proj(x)))


def lora(rank, alpha, init=True):
    return get_peft_model(Net(), LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=["q_proj", "down_proj"],
                                            init_lora_weights=init))


def trained_r4():
    m = lora(4, 8, init=False)               # random A AND B: a "trained" adapter with a non-zero delta
    return m, {k: v.detach().clone() for k, v in get_peft_model_state_dict(m).items()}


def test_expanded_adapter_is_identical_at_step_0_and_new_columns_learn():
    small, sd = trained_r4()
    x = torch.randn(5, 12)
    want = small(x).detach()
    alpha = LX.expanded_alpha(4, 8, 16)
    assert alpha == 32.0                                      # alpha/rank: 8/4 == 32/16
    big = lora(16, alpha)                                     # fresh rank-16 model: PEFT init (kaiming A, zero B)
    cur = {k: v.detach().clone() for k, v in get_peft_model_state_dict(big).items()}
    new, stats = LX.expand_state_dict(sd, cur)
    assert stats["expanded_pairs"] == 2 and (stats["from_rank"], stats["to_rank"]) == (4, 16)
    set_peft_model_state_dict(big, new)
    got = big(x)
    assert torch.allclose(got, want, atol=1e-6), float((got - want).abs().max())
    loaded = get_peft_model_state_dict(big)
    for k, v in loaded.items():
        if ".lora_B." in k:
            assert torch.equal(v[:, :4], sd[k]) and not v[:, 4:].any()        # trained columns kept, new ones zero
        if ".lora_A." in k:
            assert torch.equal(v[:4], sd[k]) and v[4:].abs().sum() > 0        # trained rows kept, new rows initialised
    got.pow(2).sum().backward()
    for name, p in big.named_parameters():
        if "lora_B" in name:
            assert p.grad[:, 4:].abs().sum() > 0, name                        # gradients reach the new columns at step 0


def test_expanded_adapter_saves_and_reloads_through_peft(tmp_path):
    small, sd = trained_r4()
    x = torch.randn(3, 12)
    want = small(x).detach()
    big = lora(16, LX.expanded_alpha(4, 8, 16))
    new, _ = LX.expand_state_dict(sd, {k: v.detach().clone() for k, v in get_peft_model_state_dict(big).items()})
    set_peft_model_state_dict(big, new)
    big.save_pretrained(tmp_path / "exp")
    cfg = json.loads((tmp_path / "exp/adapter_config.json").read_text())
    assert cfg["r"] == 16 and float(cfg["lora_alpha"]) == 32.0
    again = PeftModel.from_pretrained(Net(), str(tmp_path / "exp")).eval()     # the server / evaluator / mine.py load path
    assert torch.allclose(again(x), want, atol=1e-6)


def test_offline_expansion_cli_and_mlx_conversion_keep_the_delta(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import convert_peft_adapter_to_mlx as conv
    peft_prefix = "base_model.model.model.language_model.layers"
    src = tmp_path / "r4"
    src.mkdir()
    g = torch.Generator().manual_seed(3)
    tensors = {f"{peft_prefix}.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8, generator=g),
               f"{peft_prefix}.0.self_attn.q_proj.lora_B.weight": torch.randn(6, 4, generator=g)}
    save_file(tensors, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                                                         "use_dora": False, "fan_in_fan_out": False, "bias": "none"}))
    assert LX.main(["--adapter", str(src), "--rank", "16", "--out", str(tmp_path / "r16")]) == 0
    cfg = json.loads((tmp_path / "r16/adapter_config.json").read_text())
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32.0
    summary = conv.convert(tmp_path / "r16", tmp_path / "mlx")
    assert summary["rank"] == 16 and summary["scale"] == 2.0
    w = mx.load(str(tmp_path / "mlx/adapters.safetensors"))
    key = "language_model.model.layers.0.self_attn.q_proj"
    mlx_delta = (2.0 * w[f"{key}.lora_b"].T) @ w[f"{key}.lora_a"].T
    a, b = tensors[f"{peft_prefix}.0.self_attn.q_proj.lora_A.weight"], tensors[f"{peft_prefix}.0.self_attn.q_proj.lora_B.weight"]
    want = (8 / 4) * (b @ a)
    assert mx.allclose(mx.array(want.numpy()), mlx_delta, atol=1e-5).item()


def test_expand_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        LX.expand_state_dict({"m.lora_A.weight": torch.zeros(4, 8)}, {"m.lora_A.weight": torch.zeros(16, 9)})
    with pytest.raises(KeyError):
        LX.expand_state_dict({"other.lora_A.weight": torch.zeros(4, 8)}, {"m.lora_A.weight": torch.zeros(16, 8)})


# ------------------------------------------------------------------------------------------------ the real trainer (tiny stand-in)
def _fake_engine() -> str:
    from test_p3_resume import FAKE_ENGINE
    old = """    def enable_readout(self, adapter=None, trainable=True, codes=None):
        self.readout = torch.nn.Linear(16, codes or 255, bias=False)
"""
    new = """    def enable_readout(self, adapter=None, trainable=True, codes=None):
        self.readout = torch.nn.Linear(16, codes or 255, bias=False)
        if adapter and (Path(adapter) / 'decision_readout.safetensors').exists():   # as the real engine: the init adapter's readout
            from safetensors.torch import load_file
            self.readout.weight.data.copy_(load_file(str(Path(adapter) / 'decision_readout.safetensors'))['weight'])
"""
    assert old in FAKE_ENGINE
    return FAKE_ENGINE.replace(old, new)


@pytest.fixture()
def work(tmp_path):
    import test_p3_resume as R
    (tmp_path / "data/manifests").mkdir(parents=True)
    rows = [R._record(i, "train") for i in range(24)] + [R._record(100 + i, "dev") for i in range(6)]
    (tmp_path / "data/manifests/tiny.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake/torch_decision.py").write_text(_fake_engine())
    (tmp_path / "model").mkdir()
    trainer = ROOT / "scripts/train_decision_lora_torch.py"
    (tmp_path / "run_trainer.py").write_text(f"import runpy, sys\nsys.argv[0] = {str(trainer)!r}\nrunpy.run_path({str(trainer)!r}, run_name='__main__')\n")
    return tmp_path


def run(work: Path, out: str, *extra: str):
    env = dict(os.environ, PYTHONPATH=f"{work / 'fake'}:{ROOT / 'src'}:{ROOT / 'scripts'}", FAKE_TRACE=str(work / f"{out}.trace"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    cmd = [sys.executable, str(work / "run_trainer.py"), "--version", "tiny", "--output", str(work / out), "--model", str(work / "model"),
           "--device", "cpu", "--epochs", "1", "--lr", "0.05", "--warmup", "1", "--seed", "0", "--batch-size", "2", "--accumulate", "2",
           "--dev-cases", "6", "--dev-every", "1", "--rank", "4", "--alpha", "8", *extra]
    r = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return [json.loads(l) for l in (work / out / "log.jsonl").read_text().splitlines() if l.strip()], r.stdout


def test_trainer_expand_lora_rank_is_identical_at_step_0(work):
    run(work, "base", "--max-steps", "2")                                        # a trained r4 adapter (+ readout) to start from
    same, _ = run(work, "same", "--init-adapter", str(work / "base/last"), "--max-steps", "1")
    exp, out = run(work, "exp", "--init-adapter", str(work / "base/last"), "--expand-lora-rank", "8", "--max-steps", "1", "--dev-slices")
    assert "expanded the init adapter's LoRA rank 4 -> 8" in out
    d_same = next(e for e in same if e.get("step") == 0 and "dev_loss" in e)
    d_exp = next(e for e in exp if e.get("step") == 0 and "dev_loss" in e)
    assert d_exp["dev_loss"] == pytest.approx(d_same["dev_loss"], rel=1e-6, abs=1e-7)
    assert d_exp["dev_accuracy"] == d_same["dev_accuracy"]
    assert set(d_exp["dev_slices"]) == {"text"} and d_exp["dev_slices"]["text"]["n"] == 6
    cfg = json.loads((work / "exp/config.json").read_text())
    assert (cfg["rank"], cfg["alpha"], cfg["expand_lora_rank"]) == (8, 16.0, 8)
    ac = json.loads((work / "exp/last/adapter_config.json").read_text())
    assert ac["r"] == 8 and float(ac["lora_alpha"]) == 16.0
    w = load_file(str(work / "exp/last/adapter_model.safetensors"))
    b = next(v for k, v in w.items() if "lora_B" in k)
    assert b.shape[1] == 8 and b[:, 4:].abs().sum() > 0                       # the new columns trained in step 1
    # the r2 path: a later run from the expanded checkpoint takes its rank from the adapter (no flag needed)
    again, _ = run(work, "again", "--init-adapter", str(work / "exp/last"), "--max-steps", "1")
    cfg2 = json.loads((work / "again/config.json").read_text())
    assert (cfg2["rank"], cfg2["alpha"]) == (8, 16.0) and "expand_lora_rank" not in cfg2


def test_trainer_refuses_shrinking_or_expansion_without_init(work):
    run(work, "base", "--max-steps", "1")
    env = dict(os.environ, PYTHONPATH=f"{work / 'fake'}:{ROOT / 'src'}:{ROOT / 'scripts'}", FAKE_TRACE=str(work / "x.trace"))
    for extra in (["--init-adapter", str(work / "base/last"), "--expand-lora-rank", "2"], ["--expand-lora-rank", "8"]):
        r = subprocess.run([sys.executable, str(work / "run_trainer.py"), "--version", "tiny", "--output", str(work / "bad"), "--model",
                            str(work / "model"), "--device", "cpu", *extra], cwd=work, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode != 0 and "--expand-lora-rank" in r.stderr


# ------------------------------------------------------------------------------------------------ pilot rules, lane resume, soup
def M(acc, loss, acc2=0.40, img=None, txt=None):
    m = {"dev_accuracy": acc, "dev_loss": loss, "dev2_accuracy": acc2}
    if img is not None:
        m["dev_slices"] = {"image": img, "text": txt}
    return m


def test_rank64_needs_a_clear_gain_and_no_slice_regression():
    base = M(0.70, 1.00, img=0.65, txt=0.72)
    d = pilot.decide({"baseline": base, "rank64": M(0.72, 0.99, img=0.67, txt=0.74)}, False, False)
    assert d["expand_lora_rank"] == 64 and d["lane"] == "rank64" and not d["combined"]           # +2 pts, loss lower
    d = pilot.decide({"baseline": base, "rank64": M(0.71, 0.99, img=0.66, txt=0.73)}, False, False)
    assert d["expand_lora_rank"] == 0 and any("not beyond noise" in r for r in d["reasons"])      # +1 pt: noise, keep r16
    d = pilot.decide({"baseline": base, "rank64": M(0.70, 0.96, img=0.66, txt=0.71)}, False, False)
    assert d["expand_lora_rank"] == 64                                                           # loss -4%, accuracy held
    d = pilot.decide({"baseline": base, "rank64": M(0.73, 0.98, img=0.60, txt=0.78)}, False, False)
    assert d["expand_lora_rank"] == 0 and any("slice regression" in r for r in d["reasons"])      # images -5 pts
    d = pilot.decide({"baseline": base, "rank64": None}, False, False)
    assert d["expand_lora_rank"] == 0 and d["lora_rank"] == 16


def test_rank64_with_ordinal_is_combined_and_restarts(tmp_path):
    base = M(0.70, 1.00, 0.40, img=0.65, txt=0.72)
    d = pilot.decide({"baseline": base, "ordinal": M(0.70, 1.00, 0.45, img=0.65, txt=0.72),
                      "rank64": M(0.73, 0.98, 0.40, img=0.66, txt=0.75)}, False, False)
    assert (d["ordinal_weight"], d["expand_lora_rank"], d["lane"], d["combined"]) == (0.15, 64, None, True)
    r = lane_resume.plan(tmp_path / "pilot", d, 100, 8, 1)
    assert not r["ok"] and r["flags"] == [] and "COMBINED" in r["summary"]


def test_lane_resume_continues_the_rank64_lane(tmp_path):
    t = tmp_path / "pilot/rank64/train"
    (t / "last").mkdir(parents=True)
    for f in ("trainer.pt", "adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json"):
        (t / "last" / f).write_text("x")
    (t / "config.json").write_text(json.dumps({"world_size": 2, "accumulate": 4, "expand_lora_rank": 64, "rank": 64, "alpha": 128.0}))
    (t / "log.jsonl").write_text(json.dumps({"step": 100, "dev_loss": 0.9}) + "\n")
    (tmp_path / "pilot-rank64.DONE").touch()
    dec = {"ordinal_weight": 0.0, "readout_codes": 255, "expand_lora_rank": 64, "lane": "rank64", "combined": False}
    r = lane_resume.plan(tmp_path / "pilot", dec, 100, 8, 1)
    assert r["ok"] and r["mode"] == "exact" and r["flags"] == ["--resume-from", str(t / "last")]
    (t / "config.json").write_text(json.dumps({"world_size": 2, "accumulate": 4}))     # a r16 lane under that name
    assert not lane_resume.plan(tmp_path / "pilot", dec, 100, 8, 1)["ok"]


def test_soup_pads_the_shipped_r16_to_the_expanded_rank():
    shipped = {"m.lora_A.weight": torch.ones(16, 8), "m.lora_B.weight": torch.ones(6, 16), "weight": torch.ones(255, 4)}
    new = {"m.lora_A.weight": torch.full((64, 8), 3.0), "m.lora_B.weight": torch.full((6, 64), 3.0), "weight": torch.ones(256, 4)}
    out = soup.soup_tensors(shipped, new, 0.5)
    assert out["m.lora_A.weight"].shape == (64, 8) and float(out["m.lora_A.weight"][0, 0]) == 2.0 and float(out["m.lora_A.weight"][20, 0]) == 1.5
    assert out["m.lora_B.weight"].shape == (6, 64) and float(out["m.lora_B.weight"][0, 20]) == 1.5
    assert out["weight"].shape == (256, 4)


def test_pod_run_has_the_rank64_lane_and_passes_the_expansion_to_r1_only():
    s = (ROOT / "cloud/p3/pod_run_train.sh").read_text()
    assert "lane rank64 4,5 29523 0.0 255 $V255 --expand-lora-rank $RANK_X" in s and "lane both" not in s
    assert "--dev-slices" in s[s.index("lane(){"):s.index("if ! skip pilot")]
    assert "RANKX=$(python -c" in s and '$( [ "$tag" = r1 ] && echo $R1_XFLAGS )' in s
