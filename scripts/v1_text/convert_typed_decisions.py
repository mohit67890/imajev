"""Convert a local Jev/typed-decisions JSON/JSONL export after its license is verified."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .common import fit_partition,read_rows,verified_license,write_jsonl
from vision_decision.jev_api import to_request

def _internal_request(row):
    """Translate the published HF row's Jev questions into the internal request."""
    state=json.loads(row["state"]) if isinstance(row["state"],str) and row["state"].lstrip().startswith(("{","[")) else row["state"]
    questions=json.loads(row["questions"]) if isinstance(row["questions"],str) else row["questions"]
    return to_request({"state":state,"questions":questions},request_id=str(row["id"])).model_dump(mode="json")

def convert(source:Path,evidence:Path,spdx="Apache-2.0",split=None):
    license_info=verified_license(evidence,spdx);out=[]
    for n,row in enumerate(read_rows(source)):
        request=row.get("request",row.get("input"));answers=row.get("targets",row.get("answers"))
        gold=None
        if request is None and "questions" in row:
            request=_internal_request(row);gold=json.loads(row["gold"]) if isinstance(row["gold"],str) else row["gold"]
            answers={k:(v["label"].lower()=="true" if v["type"]=="noul" else int(v["label"]) if v["type"]=="score" else v["label"]) for k,v in gold.items()}
        if not isinstance(request,dict) or not isinstance(request.get("fields"),list):raise ValueError(f"row {n}: request.fields missing")
        if not isinstance(answers,dict):
            if len(request["fields"])!=1:raise ValueError(f"row {n}: multi-question row needs targets by field id")
            answers={request["fields"][0]["id"]:row.get("target")}
        group=str(row.get("source_group",row.get("id",n)));rid=str(row.get("id",hashlib.sha256(f'{group}:{n}'.encode()).hexdigest()[:16]))
        distributions={k:v["probabilities"] for k,v in gold.items()} if gold else row.get("target_distributions",{})
        if row.get("target_distribution") is not None and len(request["fields"])==1:
            distributions={request["fields"][0]["id"]:row["target_distribution"]}
        causes=row.get("abstention_causes",{});official=str(row.get("split",split or "")).lower()
        if official not in {"train","test","validation","valid","dev"}:raise ValueError(f"row {n}: official split is required")
        partition="test" if official=="test" else "dev" if official in {"validation","valid","dev"} else fit_partition(group)
        out.append({"id":f"typed_decisions:{rid}","source":"typed_decisions","source_split":official,
          "source_group":group,"heldout_family":partition=="test","family":str(row.get("workflow",row.get("family","typed_decision"))),"license":license_info,
          "images":[],"request":request,"targets":answers,"target_distributions":distributions,
          "abstention_causes":causes,"partition":partition,"source_answer":answers})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("license_evidence",type=Path);p.add_argument("output",type=Path);p.add_argument("--spdx",default="Apache-2.0");p.add_argument("--split",choices=("train","validation","test"));a=p.parse_args()
    rows=convert(a.source,a.license_evidence,a.spdx,a.split);write_jsonl(a.output,rows);print(json.dumps({"records":len(rows),"output":str(a.output)}))
if __name__=="__main__":main()
