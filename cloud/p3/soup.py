"""Phase-3 fallback only: weight-average a phase-3 checkpoint with the shipped 1.0 adapter (LoRA + decision readout), as the
phase-2c soup (cloud/p2c-side/soup_gen.sh). W is the weight of the NEW checkpoint. A 256-row readout soups its first 255 rows
with the shipped 255-row readout and keeps row 256 from the new checkpoint (the shipped model never had it). A rank-expanded
checkpoint (--expand-lora-rank 64: alpha/rank kept equal to the shipped 32/16) is souped with the shipped r16 factors zero-padded
to its rank (scripts/lora_expand.py pad_to_rank: the same function), and keeps its own adapter_config (r64, alpha 128).

    python cloud/p3/soup.py --shipped adapters/4b-shipped --new <ckpt> --weight 0.5 --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def soup_tensors(a: dict, b: dict, w: float) -> dict:
    import torch
    ranks = {v.shape[0] for k, v in b.items() if ".lora_A." in k and v.dim() == 2}
    if len(ranks) == 1:
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "scripts"))
        from lora_expand import pad_to_rank
        a = pad_to_rank(a, next(iter(ranks)))
    if a.keys() != b.keys():
        raise ValueError(f"key mismatch: {sorted(set(a) ^ set(b))[:5]}")
    out = {}
    for k in a:
        x, y = a[k].float(), b[k].float()
        if x.shape != y.shape:
            if x.dim() == 2 and y.dim() == 2 and x.shape[1] == y.shape[1] and y.shape[0] == x.shape[0] + 1:
                mixed = (1 - w) * x + w * y[: x.shape[0]]
                out[k] = torch.cat([mixed, y[x.shape[0]:]]).to(b[k].dtype).contiguous(); continue
            raise ValueError(f"{k}: shapes {tuple(x.shape)} vs {tuple(y.shape)}")
        out[k] = ((1 - w) * x + w * y).to(b[k].dtype).contiguous()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shipped", type=Path, required=True); ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--weight", type=float, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from safetensors.torch import load_file, save_file
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_config.json", "decision_readout.json"):
        shutil.copyfile(a.new / f, a.out / f)   # the new checkpoint's code binding (255 or 256 codes) and layout
    for f in ("adapter_model.safetensors", "decision_readout.safetensors"):
        save_file(soup_tensors(load_file(str(a.shipped / f)), load_file(str(a.new / f)), a.weight), str(a.out / f), metadata={"format": "pt"})
    print(f"soup W={a.weight} of {a.new} with {a.shipped} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
