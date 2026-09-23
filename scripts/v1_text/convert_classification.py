"""Convert local CSV/JSON(L) text-classification exports to choice decisions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .common import fit_partition,read_rows,verified_license,write_jsonl

def convert(source,evidence,spdx,text_key="text",label_key="label",group_key="id",source_name="classification",heldout=False):
    license_info=verified_license(evidence,spdx);raw=read_rows(source);labels=sorted({str(r[label_key]) for r in raw});out=[]
    if len(labels)<2:raise ValueError("classification source needs at least two labels")
    for n,r in enumerate(raw):
        group=str(r.get(group_key,n));official=str(r.get("split","")).lower()
        if heldout:partition="test"
        elif official in {"test","validation","valid","dev"}:partition="dev" if official in {"validation","valid","dev"} else "test"
        elif official in {"train",""}:partition=fit_partition(group)
        else:raise ValueError(f"row {n}: unsupported official split {official!r}")
        out.append({"id":f"{source_name}:{group}","source":source_name,"source_split":str(r.get("split","local")),"source_group":group,
          "family":source_name,"heldout_family":heldout,"license":license_info,"images":[],"request":{"schema_version":"1.0","request_id":f"{source_name}-{group}",
          "state":str(r[text_key]),"fields":[{"id":"label","type":"choice","question":"Classify the text.","options":[{"value":x} for x in labels]}]},
          "target":str(r[label_key]),"abstention_cause":None,"partition":partition,"source_answer":r[label_key]})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("license_evidence",type=Path);p.add_argument("output",type=Path);p.add_argument("--spdx",required=True);p.add_argument("--source-name",required=True);p.add_argument("--text-key",default="text");p.add_argument("--label-key",default="label");p.add_argument("--group-key",default="id");p.add_argument("--heldout",action="store_true");a=p.parse_args()
    rows=convert(a.source,a.license_evidence,a.spdx,a.text_key,a.label_key,a.group_key,a.source_name,a.heldout);write_jsonl(a.output,rows);print(json.dumps({"records":len(rows),"output":str(a.output)}))
if __name__=="__main__":main()
