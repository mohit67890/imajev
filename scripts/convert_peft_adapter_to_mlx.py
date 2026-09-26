"""Convert a PyTorch/PEFT LoRA adapter into the adapter format mlx-vlm loads.

The v1 decision adapter was trained with PyTorch + PEFT on the language layers of
Qwen3.5-2B. Our fast serving path is MLX, whose `mlx_vlm.trainer.utils.apply_lora_layers`
expects the layout written by `save_adapter` after `get_peft_model`:

    adapters.safetensors   {"<mlx module path>.lora_a": [in, r], "<...>.lora_b": [r, out]}
    adapter_config.json    {"fine_tune_type": "lora", "num_layers": -1,
                            "lora_parameters": {"rank", "dropout", "scale", "keys": [...]}}

Two things have to change per tensor.

Name. PEFT wraps the HF model twice, so a parameter lives at
    base_model.model. model.language_model. layers.N.<module>. lora_A.weight
while the MLX tree puts the decoder under `language_model.model`:
    language_model.model. layers.N.<module>. lora_a

Orientation. PEFT stores lora_A as [r, in] and lora_B as [out, r] and computes
    delta(x) = scaling * (x @ A.T) @ B.T
MLX's LoRALinear stores lora_a as [in, r] and lora_b as [r, out] and computes
    delta(x) = scale * (x @ lora_a) @ lora_b
so lora_a = A.T and lora_b = B.T, and the effective weight delta is identical as long as
scale == scaling. Both sides use alpha/rank (no rslora), which is 32/16 = 2.0 here, so the
scale carries over unchanged and the weights need only a transpose.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx

PEFT_WRAPPER = "base_model.model."
HF_LANGUAGE_PREFIX = "model.language_model."
MLX_LANGUAGE_PREFIX = "language_model.model."
SIDES = {"lora_A": "lora_a", "lora_B": "lora_b"}
READOUT_ROWS = (255, 256)  # shipped readout, extended readout (options + unknown codes)
# PEFT emits these for adapter kinds LoRALinear cannot represent; fail loudly instead of dropping them.
UNSUPPORTED = ("lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector", "modules_to_save")


def split_peft_key(key):
    """(module path, 'lora_a'|'lora_b') for a PEFT LoRA weight, or None if the key is not one."""
    parts = key.split(".")
    for position, part in enumerate(parts):
        if part in SIDES:
            tail = parts[position + 1:]
            # Either "<...>.lora_A.weight" or, for a named adapter, "<...>.lora_A.<name>.weight".
            if tail[-1:] != ["weight"] or len(tail) > 2:
                return None
            return ".".join(parts[:position]), SIDES[part]
    return None


def map_module(module):
    """PEFT module path -> MLX module path, or None when the module is outside the language model."""
    if module.startswith(PEFT_WRAPPER):
        module = module[len(PEFT_WRAPPER):]
    if not module.startswith(HF_LANGUAGE_PREFIX):
        return None
    return MLX_LANGUAGE_PREFIX + module[len(HF_LANGUAGE_PREFIX):]


def sort_key(module):
    """Deterministic ordering: by layer index, then by the rest of the path."""
    parts = module.split(".")
    for position, part in enumerate(parts):
        if part == "layers" and position + 1 < len(parts) and parts[position + 1].isdigit():
            return (0, int(parts[position + 1]), ".".join(parts[position + 2:]))
    return (1, 0, module)


def convert_weights(weights, rank):
    """{peft key: array} -> ({mlx key: array}, sorted mlx module list, unmatched peft keys)."""
    converted, modules, unmatched = {}, {}, []
    for key in sorted(weights):
        if any(marker in key for marker in UNSUPPORTED):
            unmatched.append(key)
            continue
        split = split_peft_key(key)
        if split is None:
            unmatched.append(key)
            continue
        module, side = split
        mapped = map_module(module)
        if mapped is None:
            unmatched.append(key)
            continue
        tensor = weights[key]
        if tensor.ndim != 2:
            raise ValueError(f"{key}: expected a 2-D LoRA factor, got shape {tuple(tensor.shape)}")
        # PEFT lora_A is [r, in] and lora_B is [out, r]; MLX wants [in, r] and [r, out].
        axis = 0 if side == "lora_a" else 1
        if tensor.shape[axis] != rank:
            raise ValueError(f"{key}: rank axis is {tensor.shape[axis]}, expected {rank}")
        converted[f"{mapped}.{side}"] = tensor.T.astype(mx.float32)
        modules.setdefault(mapped, set()).add(side)
    incomplete = sorted(m for m, sides in modules.items() if sides != {"lora_a", "lora_b"})
    if incomplete:
        raise ValueError(f"Modules missing an A or B factor: {incomplete[:5]}")
    return converted, sorted(modules, key=sort_key), unmatched


def mlx_config(rank, alpha, dropout, keys):
    """The config `_lora_config` would have produced, plus the key list `_apply_lora_layers` needs."""
    return {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {"rank": rank, "dropout": dropout, "scale": alpha / rank, "keys": keys},
    }


def read_peft_config(src):
    config = json.loads((src / "adapter_config.json").read_text())
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"Only plain LoRA adapters convert; got peft_type={config.get('peft_type')!r}")
    for flag, message in [("use_dora", "DoRA has no LoRALinear equivalent"),
                          ("fan_in_fan_out", "fan_in_fan_out transposes the base weight"),
                          ("use_qalora", "QALoRA is not supported"),
                          ("lora_bias", "LoRA bias terms are not supported by LoRALinear")]:
        if config.get(flag):
            raise ValueError(f"Cannot convert: {message} (adapter_config.{flag} is set)")
    for field, message in [("rank_pattern", "per-module ranks"), ("alpha_pattern", "per-module alphas"),
                           ("modules_to_save", "extra fully-saved modules")]:
        if config.get(field):
            raise ValueError(f"Cannot convert: {message} are not expressible in one mlx-vlm adapter ({field})")
    if config.get("use_rslora"):
        # rslora scales by alpha/sqrt(r); MLX's single `scale` could carry it, but the trained run did not use it.
        raise ValueError("Cannot convert: use_rslora changes the scaling rule")
    rank, alpha = int(config["r"]), float(config["lora_alpha"])
    return rank, alpha, float(config.get("lora_dropout") or 0.0), config


def convert(src, dst):
    src, dst = Path(src), Path(dst)
    rank, alpha, dropout, peft_config = read_peft_config(src)
    source = src / "adapter_model.safetensors"
    weights = mx.load(str(source))
    converted, keys, unmatched = convert_weights(weights, rank)
    if not converted:
        raise ValueError(f"No convertible LoRA tensors found in {source}")
    config = mlx_config(rank, alpha, dropout, keys)
    readout_source = src / "decision_readout.safetensors"
    has_readout = readout_source.exists()
    head = payload = None
    if has_readout:
        head = mx.load(str(readout_source))
        q_inputs = {tensor.shape[1] for key, tensor in weights.items()
                    if ".self_attn.q_proj.lora_A" in key and tensor.ndim == 2}
        expected_hidden = next(iter(q_inputs)) if len(q_inputs) == 1 else None
        # 255 rows: the shipped readout; 256 rows: the extended readout (phase-3 `--readout-codes 256`)
        if (set(head) != {"weight"} or head["weight"].ndim != 2 or head["weight"].shape[0] not in READOUT_ROWS
                or (expected_hidden is not None and head["weight"].shape[1] != expected_hidden)
                or not bool(mx.all(mx.isfinite(head["weight"])).item())):
            raise ValueError("decision_readout.safetensors must contain a finite weight with shape [255 or 256, hidden_size]")
        rows = head["weight"].shape[0]
        manifest = src / "decision_readout.json"
        if not manifest.exists():
            raise ValueError("Trained readout is missing decision_readout.json tokenizer binding")
        payload = json.loads(manifest.read_text())
        codes = payload.get("codes", [])
        valid_rows = all(isinstance(x, dict) and set(x) == {"code", "token_id"}
                         and isinstance(x["code"], str) and isinstance(x["token_id"], int) for x in codes)
        if (payload.get("version") != 1 or len(codes) != rows or not valid_rows
                or len({x["code"] for x in codes}) != rows or len({x["token_id"] for x in codes}) != rows
                or payload.get("prompt_layout", "standard") not in ("standard", "compact")):
            raise ValueError("Invalid decision_readout.json")
    dst.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mx.save_safetensors(str(dst / "adapters.safetensors"), converted,
                        metadata={"converted_from": str(src), "source_sha256": digest,
                                  "converter": "scripts/convert_peft_adapter_to_mlx.py",
                                  "peft_version": str(peft_config.get("peft_version", "unknown"))})
    if has_readout:
        mx.save_safetensors(str(dst / "decision_readout.safetensors"), {"weight": head["weight"].astype(mx.float32)},
                            metadata={"converted_from": str(readout_source),
                                      "source_sha256": hashlib.sha256(readout_source.read_bytes()).hexdigest()})
        (dst / "decision_readout.json").write_text(json.dumps(payload, indent=2) + "\n")
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"src": str(src), "dst": str(dst), "source_sha256": digest,
            "peft_tensors": len(weights), "tensors_converted": len(converted), "modules": len(keys),
            "rank": rank, "alpha": alpha, "scale": config["lora_parameters"]["scale"], "dropout": dropout,
            "readout": has_readout, "unmatched_keys": unmatched}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="PEFT adapter directory")
    parser.add_argument("--dst", required=True, help="output mlx-vlm adapter directory")
    args = parser.parse_args()
    summary = convert(args.src, args.dst)
    print(f"{summary['src']} -> {summary['dst']}")
    print(f"  peft tensors read      {summary['peft_tensors']}")
    print(f"  tensors converted      {summary['tensors_converted']} ({summary['modules']} linear layers)")
    print(f"  rank / alpha / scale   {summary['rank']} / {summary['alpha']:g} / {summary['scale']:g}")
    print(f"  dropout                {summary['dropout']:g}")
    print(f"  decision readout       {'copied' if summary['readout'] else 'absent (legacy adapter)'}")
    print(f"  unmatched keys         {len(summary['unmatched_keys'])}"
          + ("" if not summary["unmatched_keys"] else " -> " + ", ".join(summary["unmatched_keys"][:10])))
    if summary["unmatched_keys"]:
        raise SystemExit("Refusing to claim a clean conversion: some PEFT keys were not mapped")


if __name__ == "__main__":
    main()
