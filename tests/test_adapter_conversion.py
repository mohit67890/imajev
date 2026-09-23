"""Fast checks for the PEFT -> mlx-vlm LoRA conversion: names, shapes, config, and the delta it implies.

No model is loaded; a handful of synthetic PEFT-style tensors is enough to pin the contract.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_peft_adapter_to_mlx import convert, map_module, split_peft_key  # noqa: E402

RANK, ALPHA = 4, 8
PEFT = "base_model.model.model.language_model.layers"
# (peft module suffix, in_features, out_features)
MODULES = [("0.self_attn.q_proj", 8, 6), ("0.mlp.down_proj", 6, 8), ("10.linear_attn.in_proj_qkv", 8, 12)]


def write_peft_adapter(directory, extra_weights=None, **config_overrides):
    directory.mkdir(parents=True, exist_ok=True)
    weights = {}
    for index, (suffix, fan_in, fan_out) in enumerate(MODULES):
        weights[f"{PEFT}.{suffix}.lora_A.weight"] = mx.random.normal((RANK, fan_in), key=mx.random.key(index))
        weights[f"{PEFT}.{suffix}.lora_B.weight"] = mx.random.normal((fan_out, RANK), key=mx.random.key(100 + index))
    weights.update(extra_weights or {})
    mx.save_safetensors(str(directory / "adapter_model.safetensors"), weights)
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA, "lora_dropout": 0.0,
              "use_dora": False, "fan_in_fan_out": False, "bias": "none"}
    config.update(config_overrides)
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return weights


@pytest.fixture
def converted(tmp_path):
    source = write_peft_adapter(tmp_path / "peft")
    summary = convert(tmp_path / "peft", tmp_path / "mlx")
    weights = mx.load(str(tmp_path / "mlx" / "adapters.safetensors"))
    config = json.loads((tmp_path / "mlx" / "adapter_config.json").read_text())
    return source, summary, weights, config


def test_key_names_follow_the_mlx_module_tree(converted):
    _, summary, weights, config = converted
    expected = [f"language_model.model.layers.{suffix}" for suffix, _, _ in MODULES]
    assert sorted(config["lora_parameters"]["keys"]) == sorted(expected)
    assert set(weights) == {f"{key}.{side}" for key in expected for side in ("lora_a", "lora_b")}
    assert summary["tensors_converted"] == 2 * len(MODULES)
    assert summary["modules"] == len(MODULES)
    assert summary["unmatched_keys"] == []


def test_config_fields_match_what_apply_lora_layers_reads(converted):
    _, _, _, config = converted
    assert config["fine_tune_type"] == "lora"
    assert config["num_layers"] == -1
    parameters = config["lora_parameters"]
    assert parameters["rank"] == RANK
    assert parameters["dropout"] == 0.0
    assert parameters["scale"] == ALPHA / RANK  # LoRALinear applies this, as PEFT applies lora_alpha / r
    assert isinstance(parameters["keys"], list) and all(isinstance(k, str) for k in parameters["keys"])


def test_factors_are_transposed_into_loralinear_orientation(converted):
    _, _, weights, _ = converted
    for suffix, fan_in, fan_out in MODULES:
        key = f"language_model.model.layers.{suffix}"
        assert weights[f"{key}.lora_a"].shape == (fan_in, RANK)  # LoRALinear: x @ lora_a
        assert weights[f"{key}.lora_b"].shape == (RANK, fan_out)  # ... @ lora_b
        assert weights[f"{key}.lora_a"].dtype == mx.float32


def test_effective_weight_delta_is_preserved(converted):
    source, _, weights, config = converted
    scale = config["lora_parameters"]["scale"]
    for suffix, _, _ in MODULES:
        a = source[f"{PEFT}.{suffix}.lora_A.weight"]
        b = source[f"{PEFT}.{suffix}.lora_B.weight"]
        peft_delta = (ALPHA / RANK) * (b @ a)  # [out, in], as PEFT adds to the base weight
        key = f"language_model.model.layers.{suffix}"
        mlx_delta = (scale * weights[f"{key}.lora_b"].T) @ weights[f"{key}.lora_a"].T  # LoRALinear.fuse
        assert mx.allclose(peft_delta, mlx_delta, atol=1e-6).item()


def test_keys_outside_the_language_model_are_reported_not_dropped(tmp_path):
    stray = {"base_model.model.visual.blocks.0.attn.qkv.lora_A.weight": mx.zeros((RANK, 8)),
             "base_model.model.visual.blocks.0.attn.qkv.lora_B.weight": mx.zeros((8, RANK))}
    write_peft_adapter(tmp_path / "peft", extra_weights=stray)
    summary = convert(tmp_path / "peft", tmp_path / "mlx")
    assert sorted(summary["unmatched_keys"]) == sorted(stray)
    assert summary["tensors_converted"] == 2 * len(MODULES)


def test_unsupported_adapter_kinds_refuse(tmp_path):
    write_peft_adapter(tmp_path / "dora", use_dora=True)
    with pytest.raises(ValueError, match="DoRA"):
        convert(tmp_path / "dora", tmp_path / "out")
    write_peft_adapter(tmp_path / "rs", use_rslora=True)
    with pytest.raises(ValueError, match="rslora"):
        convert(tmp_path / "rs", tmp_path / "out")
    write_peft_adapter(tmp_path / "patterned", rank_pattern={"q_proj": 8})
    with pytest.raises(ValueError, match="rank_pattern"):
        convert(tmp_path / "patterned", tmp_path / "out")


def test_key_parsing_helpers():
    assert split_peft_key(f"{PEFT}.0.mlp.up_proj.lora_A.weight") == (f"{PEFT}.0.mlp.up_proj", "lora_a")
    assert split_peft_key(f"{PEFT}.0.mlp.up_proj.lora_B.default.weight") == (f"{PEFT}.0.mlp.up_proj", "lora_b")
    assert split_peft_key(f"{PEFT}.0.mlp.up_proj.base_layer.weight") is None
    assert map_module(f"{PEFT}.3.self_attn.k_proj") == "language_model.model.layers.3.self_attn.k_proj"
    assert map_module("base_model.model.visual.blocks.0.attn.qkv") is None


def test_joint_readout_and_token_binding_are_copied(tmp_path):
    source = tmp_path / "peft"
    write_peft_adapter(source)
    head = mx.arange(255 * 8).reshape(255, 8).astype(mx.float32)
    mx.save_safetensors(str(source / "decision_readout.safetensors"), {"weight": head})
    codes = [{"code": f"code-{i}", "token_id": i} for i in range(255)]
    (source / "decision_readout.json").write_text(json.dumps({"version": 1, "codes": codes}))
    summary = convert(source, tmp_path / "mlx")
    converted = mx.load(str(tmp_path / "mlx" / "decision_readout.safetensors"))["weight"]
    assert summary["readout"] is True
    assert mx.array_equal(converted, head).item()
    assert json.loads((tmp_path / "mlx" / "decision_readout.json").read_text())["codes"] == codes
