#!/usr/bin/env python3
"""LoRA rank expansion with an identical function at step 0 (phase 3 pilot lane "rank64"; trainer --expand-lora-rank R).

An adapter of rank r (the shipped imajev-4b soup50: r16, alpha 32) becomes rank R > r:
  * every LoRA pair keeps its r trained rows of A and r trained columns of B;
  * the R - r new rows of A keep the fresh module's own init (PEFT: kaiming-uniform), the R - r new columns of B are ZERO,
    so B_new @ A_new == B_old @ A_old exactly and the model's output is unchanged;
  * alpha becomes alpha * R / r, so the scaling alpha / rank (PEFT's default rule; rslora is refused) stays the same;
  * the decision readout is untouched.
The new B columns get gradients from the first step (their gradient is (A_new x) g, and A_new's new rows are not zero); the new A
rows start receiving gradients once those B columns move, as in any freshly initialised LoRA.

    from lora_expand import read_init_config, expanded_alpha, expand_state_dict
    python scripts/lora_expand.py --adapter <peft dir> --rank 64 --out <dir>     # offline copy (new A rows = 0: exact, not trainable)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_init_config(adapter: Path | str) -> dict:
    cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
    if cfg.get("use_rslora"):
        raise ValueError("rank expansion keeps alpha/r; an rslora adapter (alpha/sqrt(r)) is not supported")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("rank expansion needs one rank for every module (rank_pattern / alpha_pattern set)")
    return {"r": int(cfg["r"]), "alpha": float(cfg["lora_alpha"]), "config": cfg}


def expanded_alpha(r: int, alpha: float, new_rank: int) -> float:
    """alpha_new / new_rank == alpha / r."""
    return float(alpha) * new_rank / r


def is_a(key: str) -> bool:
    return ".lora_A." in key or key.endswith(".lora_A.weight")


def is_b(key: str) -> bool:
    return ".lora_B." in key or key.endswith(".lora_B.weight")


def expand_state_dict(saved: dict, current: dict) -> tuple[dict, dict]:
    """saved: the rank-r adapter tensors (PEFT file keys); current: the fresh rank-R model's tensors under the same keys
    (peft.get_peft_model_state_dict). -> (tensors to load, stats). A: current[:r] <- saved; B: zeros, [:, :r] <- saved."""
    import torch
    out, stats = {}, {"expanded_pairs": 0, "unchanged": 0, "from_rank": None, "to_rank": None}
    for k, v in saved.items():
        cur = current.get(k)
        if cur is None:
            raise KeyError(f"{k}: not in the rank-expanded model (targets differ?)")
        if tuple(cur.shape) == tuple(v.shape):
            out[k] = v
            stats["unchanged"] += 1
            continue
        if is_a(k):
            r, big = v.shape[0], cur.shape[0]
            if cur.shape[1:] != v.shape[1:] or big < r:
                raise ValueError(f"{k}: cannot expand {tuple(v.shape)} to {tuple(cur.shape)}")
            t = cur.detach().clone().to(v.dtype)
            t[:r] = v
        elif is_b(k):
            r, big = v.shape[1], cur.shape[1]
            if cur.shape[0] != v.shape[0] or big < r:
                raise ValueError(f"{k}: cannot expand {tuple(v.shape)} to {tuple(cur.shape)}")
            t = torch.zeros(tuple(cur.shape), dtype=v.dtype)
            t[:, :r] = v
            stats["expanded_pairs"] += 1
        else:
            raise ValueError(f"{k}: shape {tuple(v.shape)} vs {tuple(cur.shape)} and not a LoRA factor")
        stats["from_rank"], stats["to_rank"] = r, big
        out[k] = t.contiguous()
    return out, stats


def pad_to_rank(tensors: dict, new_rank: int) -> dict:
    """Zero-pad every LoRA factor of a lower rank to new_rank (A rows, B columns): the same function, for soups / offline copies."""
    import torch
    out = {}
    for k, v in tensors.items():
        if is_a(k) and v.dim() == 2 and v.shape[0] < new_rank:
            out[k] = torch.cat([v, torch.zeros((new_rank - v.shape[0], v.shape[1]), dtype=v.dtype)]).contiguous()
        elif is_b(k) and v.dim() == 2 and v.shape[1] < new_rank:
            out[k] = torch.cat([v, torch.zeros((v.shape[0], new_rank - v.shape[1]), dtype=v.dtype)], dim=1).contiguous()
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", type=Path, required=True); ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from safetensors.torch import load_file, save_file
    info = read_init_config(a.adapter)
    if a.rank < info["r"]:
        raise SystemExit(f"--rank {a.rank} < the adapter's rank {info['r']}")
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.adapter.iterdir():
        if f.is_file() and f.name not in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(f, a.out / f.name)
    save_file(pad_to_rank(load_file(str(a.adapter / "adapter_model.safetensors")), a.rank), str(a.out / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    cfg = dict(info["config"], r=a.rank, lora_alpha=expanded_alpha(info["r"], info["alpha"], a.rank))
    (a.out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"expanded {a.adapter} r{info['r']} alpha {info['alpha']:g} -> r{a.rank} alpha {cfg['lora_alpha']:g} in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
