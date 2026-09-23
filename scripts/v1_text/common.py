from __future__ import annotations

import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COMMERCIAL_ALLOWLIST={"Apache-2.0","MIT","BSD-2-Clause","BSD-3-Clause","CC0-1.0","CC-BY-2.0","CC-BY-3.0","CC-BY-4.0","CC-BY-SA-3.0","CC-BY-SA-4.0"}  # CC-BY-2.0 added 2026-09-23 for Open Images photos (attribution-only, commercial use permitted)

def stable_partition(source_group:str,seed:str="decision-v1-text",dev=5,test=10)->str:
    """Group-preserving deterministic split, with percentages independent of input order."""
    n=int(hashlib.sha256(f"{seed}\0{source_group}".encode()).hexdigest()[:8],16)%100
    return "dev" if n<dev else "test" if n<dev+test else "train"

def fit_partition(source_group:str,seed:str="decision-v1-text",dev=5,calibration=5)->str:
    """Split an upstream training group; calibration is never used as training data."""
    n=int(hashlib.sha256(f"{seed}\0fit\0{source_group}".encode()).hexdigest()[:8],16)%100
    return "dev" if n<dev else "calibration" if n<dev+calibration else "train"

def read_rows(path:Path):
    path=Path(path)
    if path.suffix==".jsonl":
        return [json.loads(line) for line in path.open() if line.strip()]
    if path.suffix==".json":
        value=json.loads(path.read_text());return value if isinstance(value,list) else value.get("data",value.get("rows",[]))
    if path.suffix==".csv":
        with path.open(newline="") as f:return list(csv.DictReader(f))
    if path.suffix==".parquet" or not path.suffix:
        try:
            import pandas as pd
        except ImportError as exc:raise ValueError("Parquet input requires pandas and pyarrow") from exc
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"unsupported local source format: {path.suffix}")

def verified_license(evidence:Path,expected_spdx:str)->dict:
    """Require copied license evidence and an explicit reviewed receipt.

    Merely finding a Hub metadata tag is deliberately insufficient. The receipt must
    name the exact evidence file and record commercial-use review.
    """
    evidence=Path(evidence);receipt=evidence.with_name(evidence.name+".receipt.json")
    # Manifests carry repo-relative evidence paths so the same audit passes on a pod.
    located=evidence if evidence.is_absolute() else ROOT/evidence
    located_receipt=located.with_name(located.name+".receipt.json")
    if not located.is_file() or not located_receipt.is_file():
        raise ValueError("license evidence and its .receipt.json are required")
    meta=json.loads(located_receipt.read_text())
    digest=hashlib.sha256(located.read_bytes()).hexdigest()
    if meta.get("evidence_sha256")!=digest or meta.get("spdx")!=expected_spdx or meta.get("commercial_use_reviewed") is not True:
        raise ValueError("license receipt does not verify this evidence/SPDX/commercial-use review")
    if expected_spdx not in COMMERCIAL_ALLOWLIST:
        raise ValueError(f"license {expected_spdx} is not in the commercial allowlist")
    return {"spdx":expected_spdx,"evidence":str(evidence),"evidence_sha256":digest,"receipt":str(receipt)}

def write_jsonl(path:Path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=True,allow_nan=False)+"\n" for r in rows))
