"""Keep precomputed synthetic examples only when two blind labels agree."""
from __future__ import annotations
import argparse,json
import math
from pathlib import Path
from .common import write_jsonl

def agreed(rows):
    out=[]
    for r in rows:
        labels=r.get("blind_labels")
        if not isinstance(labels,list) or len(labels)!=2:raise ValueError(f"{r.get('id')}: exactly two blind_labels required")
        # JSON canonicalization supports scalar and structured multi-question labels.
        def valid(v):
            if isinstance(v,float):return math.isfinite(v)
            if isinstance(v,dict):return all(isinstance(k,str) and valid(x) for k,x in v.items())
            if isinstance(v,list):return all(valid(x) for x in v)
            return v is None or type(v) in (str,int,bool)
        if not all(valid(x) for x in labels):raise ValueError(f"{r.get('id')}: blind label contains a non-finite/unsupported value")
        if json.dumps(labels[0],sort_keys=True,ensure_ascii=False)==json.dumps(labels[1],sort_keys=True,ensure_ascii=False):
            copy=dict(r);copy.pop("blind_labels");label=labels[0]
            if isinstance(label,dict):copy["targets"]=label;copy.pop("target",None)
            else:copy["target"]=label;copy.pop("targets",None)
            copy["agreement_filter"]={"blind_labelers":2,"agreed":True};out.append(copy)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    rows=[json.loads(x) for x in a.source.read_text().splitlines() if x.strip()];kept=agreed(rows);write_jsonl(a.output,kept);print(json.dumps({"input":len(rows),"kept":len(kept)}))
if __name__=="__main__":main()
